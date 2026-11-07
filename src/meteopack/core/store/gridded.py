"""zarr v3 store for ``(time, y, x)`` fields: one frame per chunk, a day per shard.

Layout under ``<root>/<layer>/<category>/<source>/<product>/v<n>/``: a zarr v3 group holding
one array per variable plus a ``time`` coordinate, with the grid in the attrs.

Four decisions worth stating:

*One frame per inner chunk, many frames per shard.* Measured on the KNMI radar
grid: a packed ``[288, 765, 700]`` chunk serves 3-5 warm frame reads/s, because
every read inflates a whole block to return one frame; 1-frame inner chunks
inside a 576-frame shard serve 63-89/s. Sharding is what keeps that affordable —
one file per shard instead of one per frame — and costs nothing on the read,
since the shard index addresses a single inner chunk.

*The time axis is created whole and never grown.* :func:`initialise` lays down
the full timeline; frames nobody wrote read back as NaN from the array's fill
value without occupying a byte on disk. Every later write is therefore a pure
region write into an index range that already exists — no resize, no append, no
coordinate rewrite.

*Writes are region writes keyed on the timeline.* :func:`write` maps incoming
times with :meth:`Timeline.index_of` and writes only the rows it touches, so
re-running a day overwrites it in place: idempotent, and order-independent.
Times that are not exactly on the cadence get index -1 and are dropped rather
than snapped, so a cadence mismatch surfaces as missing data in validation
instead of a silently shifted series.

*Concurrency is safe per file, which means per shard.* Two tasks writing
disjoint shard-aligned time regions touch disjoint files and cannot interfere —
that is the Slurm-array case this layout is for, one task per day with
``shard["time"]`` set to a day of frames. Two tasks writing *different frames of
the same shard* are not safe: zarr has to read, merge and rewrite the whole
shard file, and it replaces that file atomically, so the slower writer's frames
simply vanish — measured here as whole half-shards reading back NaN, never as a
half-written file. Split work on shard boundaries, or write a shard from a
single task. Nothing in this module takes a lock.
"""
from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from meteopack.core import grids
from meteopack.core.config import Layer, ProductConfig, StoreConfig, layer_path
from meteopack.core.encoding import Encoding
from meteopack.core.timeline import Timeline

if TYPE_CHECKING:  # xarray is a caller-facing type here, not an import-time cost
    import xarray as xr

#: Gridded stores are canonical-layer artefacts; training sets are derived from them.
DEFAULT_LAYER: Layer = "packed"

logger = logging.getLogger(__name__)

TIME_NAME = "time"
DIMS: tuple[str, str, str] = ("time", "y", "x")

#: Seconds, not nanoseconds: CF-decodable by any reader, and exact for an integer
#: cadence. Sub-second axes are rejected in `initialise` rather than rounded.
#: Nanoseconds, not seconds. numpy's datetime64[ns] is what every reader here
#: already uses, so storing seconds forces a conversion on the way in and out
#: and loses sub-second resolution the moment a product has any.
TIME_UNITS = "nanoseconds since 1970-01-01T00:00:00Z"

#: Cell-centre coordinates in the projection's own units, plus per-cell
#: geographic coordinates, so the store is georeferenced without a sidecar.
COORD_NAMES: tuple[str, ...] = ("y", "x", "lat", "lon")

#: Per-frame provenance. Written alongside the fields because the questions it
#: answers - was this frame absent, was it unreadable, which upstream
#: generation produced it - are unanswerable from the imagery itself: an absent
#: frame and an empty one both read as missing.
PROVENANCE: dict[str, tuple[str, object]] = {
    # True by default: a slot nobody has written has no source file behind it,
    # which is exactly what missing means. `write` clears it for the rows it
    # touches, so correctness here needs nothing from the source.
    "is_missing_frame": ("bool", True),
    "is_corrupt_frame": ("bool", False),
    "source_layout_id": ("int16", -1),
    "source_record_id": ("int64", -1),
}

#: The zarr group sits in a subdirectory of the store so that `manifest.json`
#: and `README.md` are beside it rather than inside it. Zarr warns about every
#: foreign object it finds in a hierarchy, and it is right to.
ZARR_NAME = "dataset.zarr"

#: The CF grid-mapping variable name, referenced by every field.
GRID_MAPPING_NAME = "crs"

_NS_PER_S = 1_000_000_000
_SECONDS_PER_DAY = 86_400


def store_path(root: str | os.PathLike[str], store_id: str, *, layer: Layer = DEFAULT_LAYER) -> Path:
    """Where a store with this id lives. ``store_id`` is ``source/product/vN``."""
    return layer_path(Path(root), layer, *store_id.split("/"))


def _compressors(store: StoreConfig) -> Any:
    """Codec chain for the data arrays. zarr 3 defaults to zstd; we pin the level."""
    name = (store.compressor or "none").lower()
    level = store.compression_level
    if name in ("none", "null", "raw"):
        return None
    if name == "zstd":
        return ZstdCodec(level=level)
    if name == "gzip":
        return GzipCodec(level=level)
    if name == "blosc":
        return BloscCodec(cname="zstd", clevel=level)
    raise ValueError(f"unsupported compressor {store.compressor!r}; use zstd, blosc, gzip or none")


def _geometry(
    cfg: ProductConfig, shape: tuple[int, int, int]
) -> tuple[tuple[int, int, int], tuple[int, int, int] | None]:
    """Inner chunk and shard shapes: from config where given, benchmarked defaults otherwise.

    Defaults are one frame per chunk and one day of frames per shard — a day
    being the natural unit of work for a packing job, and therefore the grain at
    which parallel writers stay on disjoint files.
    """
    _, ny, nx = shape
    frames_per_day = max(1, _SECONDS_PER_DAY // cfg.cadence_seconds)
    wanted = cfg.store.chunks or {}
    chunks = (
        max(1, int(wanted.get("time", 1))),
        max(1, int(wanted.get("y", ny))),
        max(1, int(wanted.get("x", nx))),
    )
    sharded = cfg.store.shard or {}
    shards = (
        max(1, int(sharded.get("time", frames_per_day))),
        max(1, int(sharded.get("y", chunks[1]))),
        max(1, int(sharded.get("x", chunks[2]))),
    )
    for dim, chunk, shard in zip(DIMS, chunks, shards, strict=True):
        if shard % chunk:
            raise ValueError(
                f"{cfg.store_id}: shard {shard} on {dim!r} is not a multiple of chunk {chunk}"
            )
    # One chunk per shard is pure overhead: a shard index wrapping a single chunk.
    return chunks, None if shards == chunks else shards


def _variable_names(cfg: ProductConfig, names: Sequence[str] | None) -> list[str]:
    declared = [v.name for v in cfg.variables]
    if names is None:
        if not declared:
            raise ValueError(f"{cfg.store_id} declares no variables")
        return declared
    unknown = [n for n in names if n not in declared]
    if unknown:
        raise ValueError(f"{cfg.store_id} does not declare {unknown}; declared: {declared}")
    return list(names)


def zarr_path(root: str | os.PathLike[str], store_id: str, *,
              layer: Layer = DEFAULT_LAYER) -> Path:
    """The zarr group inside the store directory."""
    return store_path(root, store_id, layer=layer) / ZARR_NAME


def _open(store_id: str, root: str | os.PathLike[str], layer: Layer, mode: str) -> zarr.Group:
    path = zarr_path(root, store_id, layer=layer)
    if not (path / "zarr.json").exists():
        legacy = store_path(root, store_id, layer=layer)
        if (legacy / "zarr.json").exists():
            # Stores written before the group moved into a subdirectory.
            return zarr.open_group(legacy, mode=mode, zarr_format=3)
        raise FileNotFoundError(f"no gridded store at {path}; run initialise first")
    return zarr.open_group(path, mode=mode, zarr_format=3)


def _timeline_of(group: zarr.Group) -> Timeline:
    """Rebuild the axis from the attrs rather than from the time array.

    Writers only need `index_of`, and taking it from the attrs keeps `write`
    from touching the coordinate at all — one less array shared between tasks.
    """
    attrs = group.attrs
    return Timeline(
        start=datetime.fromisoformat(str(attrs["time_start"])),
        end=datetime.fromisoformat(str(attrs["time_end"])),
        cadence=timedelta(seconds=int(attrs["cadence_seconds"])),
    )


#: Everything that is not a field: the time axis, the coordinates and the
#: per-frame provenance. Kept in one place because several functions mean
#: "the variables" and none of them mean these.
_NON_FIELD = frozenset({TIME_NAME, *COORD_NAMES, *PROVENANCE})


def _data_arrays(group: zarr.Group) -> list[str]:
    return [name for name, _ in group.arrays() if name not in _NON_FIELD]


def initialise(
    cfg: ProductConfig,
    root: str | os.PathLike[str],
    timeline: Timeline,
    variables: Sequence[str] | None = None,
    *,
    layer: Layer = DEFAULT_LAYER,
    overwrite: bool = False,
) -> Path:
    """Create the store with its full time axis, so every later write is a region write.

    Nothing is filled in: the arrays are created at their final shape with a NaN
    fill value, which costs no disk and reads back as missing. Call this once,
    from the planner; the array tasks only ever call :func:`write`.
    """
    if cfg.store.kind != "gridded":
        raise ValueError(f"{cfg.store_id} is a {cfg.store.kind} store, not gridded")
    if not cfg.grid_id:
        raise ValueError(f"{cfg.store_id} has no grid_id")
    if int(timeline.cadence.total_seconds()) != cfg.cadence_seconds:
        raise ValueError(
            f"timeline cadence {timeline.cadence} != cfg.cadence_seconds {cfg.cadence_seconds}"
        )
    if timeline.step_ns % _NS_PER_S or timeline.start_ns % _NS_PER_S:
        raise ValueError(f"{TIME_UNITS!r} cannot hold {timeline.start} at {timeline.cadence}")

    names = _variable_names(cfg, variables)
    grid = grids.get(cfg.grid_id)
    ny, nx = grid.shape
    shape = (len(timeline), ny, nx)
    chunks, shards = _geometry(cfg, shape)

    path = zarr_path(root, cfg.store_id, layer=layer)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to rebuild it")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.attrs.update(_root_attrs(cfg, grid, timeline, shape[0]))

    time = group.create_array(
        TIME_NAME,
        shape=(shape[0],),
        chunks=(shape[0],),
        dtype="int64",
        dimension_names=(TIME_NAME,),
        attributes={
            "units": TIME_UNITS,
            "calendar": "proleptic_gregorian",
            "standard_name": "time",
            "long_name": "time",
            "axis": "T",
        },
    )
    time[:] = timeline.values().astype("int64")

    _create_coordinates(group, grid)
    _create_provenance(group, shape[0], shards[0] if shards else chunks[0])

    declared = {v.name: v for v in cfg.variables}
    for name in names:
        var = declared[name]
        encoding = Encoding.from_config(var.encoding, name=name)
        attributes = _field_attrs(var, encoding)
        if encoding is None:
            dtype: object = "float32"
            fill: object = np.float32("nan")
        else:
            dtype = encoding.dtype
            fill = encoding.missing
        group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype=dtype,
            fill_value=fill,
            compressors=_compressors(cfg.store),
            dimension_names=DIMS,
            attributes=attributes,
        )
    return store_path(root, cfg.store_id, layer=layer)


def _root_attrs(cfg: ProductConfig, grid, timeline: Timeline, n_times: int) -> dict:
    """CF discovery metadata, the grid, and how this store was built.

    A store outlives the checkout that wrote it. It gets copied to another
    machine and opened by someone who has never seen this repository, so
    everything needed to know what it is goes here rather than in a config.
    """
    declared = {
        "Conventions": "CF-1.10",
        "title": str(cfg.attrs.get("title")
                     or cfg.extra.get("title")
                     or f"{cfg.source} {cfg.product} v{cfg.version}"),
        "store_id": cfg.store_id,
        "source_name": cfg.source,
        "product": cfg.product,
        "version": cfg.version,
        "grid_id": grid.id,
        "proj4": grid.proj4,
        "cadence_seconds": cfg.cadence_seconds,
        "time_start": timeline.start.isoformat(),
        "time_end": timeline.end.isoformat(),
        "n_times": n_times,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    # Whatever the config publishes - institution, source, history, references,
    # comment - wins, except the keys this store owns and must stay true.
    reserved = {"store_id", "grid_id", "proj4", "cadence_seconds",
                "time_start", "time_end", "n_times"}
    declared.update({k: v for k, v in (cfg.attrs or {}).items() if k not in reserved})
    mapping = {
        "grid_mapping_name": "polar_stereographic",
        "proj4_params": grid.proj4,
        "shape": [grid.ny, grid.nx],
        "corners_lonlat": [list(c) for c in (grid.corners_lonlat or ())],
    }
    try:
        (x0, x1), (y0, y1) = grid.projected_extent()
    except (ValueError, ImportError) as exc:
        # A grid may be declared by shape alone. Say so in the metadata rather
        # than refusing to build the store: the imagery is still the imagery.
        logger.warning("grid %s has no projected extent (%s)", grid.id, exc)
        mapping["extent_note"] = f"not recorded: {exc}"
    else:
        mapping["extent_x"] = [x0, x1]
        mapping["extent_y"] = [y0, y1]
    declared[GRID_MAPPING_NAME] = mapping
    return declared


def _create_coordinates(group: zarr.Group, grid) -> None:
    """Projected cell centres and per-cell lon/lat.

    radar_v3 stored `x`/`y` as `0..n-1`, which carries nothing the shape does
    not already carry. Real coordinates make the store open correctly in
    xarray and rioxarray, and `lat`/`lon` turn co-location with a station into
    a lookup instead of a projection.
    """
    try:
        y, x = grid.coordinates()
    except (ValueError, ImportError) as exc:
        logger.warning("grid %s: no projected coordinates written (%s)", grid.id, exc)
        return
    units = "km" if "+units=km" in grid.proj4 else "m"
    for name, values, axis, std in (
        ("y", y, "Y", "projection_y_coordinate"),
        ("x", x, "X", "projection_x_coordinate"),
    ):
        array = group.create_array(
            name, shape=values.shape, chunks=values.shape, dtype="float64",
            dimension_names=(name,),
            attributes={"units": units, "standard_name": std, "axis": axis,
                        "long_name": f"{axis.lower()} coordinate of projection"},
        )
        array[:] = values

    try:
        lon, lat = grid.lonlat()
    except Exception as exc:  # noqa: BLE001 - a grid may have no corners to project
        logger.warning("grid %s: no lon/lat coordinates written (%s)", grid.id, exc)
        return
    for name, values, std, units in (
        ("lat", lat, "latitude", "degrees_north"),
        ("lon", lon, "longitude", "degrees_east"),
    ):
        array = group.create_array(
            name, shape=values.shape, chunks=values.shape, dtype="float32",
            dimension_names=("y", "x"),
            attributes={"units": units, "standard_name": std, "long_name": std},
        )
        array[:] = values


def _field_attrs(var, encoding: Encoding | None) -> dict:
    """The attributes of one field array, from its config. Used at creation and refresh."""
    attributes: dict = {
        **var.cf_attrs,
        "grid_mapping": GRID_MAPPING_NAME,
        "coordinates": "lat lon",
    }
    if var.no_echo is not None:
        # The commonest pixel in a radar frame. Without this a reader has to
        # know the product to tell "no echo" from a small value.
        attributes["no_echo_value"] = float(var.no_echo)
        attributes["comment"] = (f"Where the radar saw no echo the value is {var.no_echo:g} "
                                 f"{var.units}; that is an observation, not missing data.")
    if encoding is not None:
        attributes.update({
            "scale_factor": encoding.scale,
            "add_offset": encoding.offset,
            "calibration_formula": encoding.formula,
            "calibration_missing_data": encoding.missing,
            "calibration_out_of_image": encoding.out_of_image,
            "_FillValue": encoding.missing,
            # CF allows one _FillValue but a vector missing_value, and
            # xarray's mask_and_scale honours both. Without this, a plain
            # `xr.open_zarr(...)` decodes out_of_image as a real value:
            # 65535 * 0.01 = 655.35 mm of rain over the whole footprint
            # mask. The two codes stay distinguishable in the raw array and
            # in the two named attributes above.
            "missing_value": [encoding.missing, encoding.out_of_image],
        })
    return attributes


#: What each per-frame provenance array means, on the array itself.
PROVENANCE_DESCRIPTIONS = {
    "is_missing_frame": "no source file covered this time step",
    "is_corrupt_frame": "a source file covered this time step but its frame could not be "
                        "used - unreadable, truncated, or a calibration the product config "
                        "lists as broken (extra.corrupt_layouts); its data is left missing",
    "source_layout_id": "which upstream layout produced this frame; the codes are in "
                        "flag_values and flag_meanings",
    "source_record_id": "the raw file this frame was packed from: its row in "
                        "provenance.parquet beside dataset.zarr (column "
                        "source_record_id); -1 if never packed",
}


def _layout_flags(names: Mapping[int, str]) -> dict:
    """CF flag attributes for `source_layout_id` from a source's code -> name table."""
    codes = sorted(names)
    return {"flag_values": [int(c) for c in codes],
            "flag_meanings": " ".join(names[c].replace(",", "").replace(" ", "_")
                                      for c in codes)}


def _create_provenance(group: zarr.Group, n_times: int, chunk: int) -> None:
    """One array per provenance field, chunked like the shards.

    Tiny - four arrays of n_times - and they answer what the imagery cannot.
    Measured on the KNMI rtcor archive: 2,013 of 772,129 slots had no file at
    all and 36 held a file that would not read. Without these, both are
    indistinguishable from a frame that was present and empty.
    """
    descriptions = PROVENANCE_DESCRIPTIONS
    for name, (dtype, fill) in PROVENANCE.items():
        group.create_array(
            name,
            shape=(n_times,),
            chunks=(max(1, min(n_times, chunk)),),
            dtype=dtype,
            fill_value=fill,
            dimension_names=(TIME_NAME,),
            attributes={"long_name": descriptions[name]},
        )


@dataclass(frozen=True, slots=True)
class WriteResult:
    """What a write actually did, as opposed to what it was handed.

    `written` used to be assumed equal to the number of frames passed in, and
    the two diverge in the one case that matters: a store's time axis is fixed
    when it is created, so packing a later window into an existing store drops
    every frame past the end of the axis. Packing four days one day at a time
    reported "wrote 576 frames" three times and kept 288 - two thirds of the
    data gone, with nothing in the output to say so. Callers must look at
    `outside` rather than counting their input.
    """

    path: Path
    written: int = 0
    outside: int = 0
    off_cadence: int = 0
    #: The first and last frame that fell outside the axis, ISO-8601. A bare
    #: count says a frame was dropped but not which end of the axis is wrong,
    #: and hunting for it means re-reading the archive by hand.
    outside_first: str | None = None
    outside_last: str | None = None

    def __fspath__(self) -> str:
        return str(self.path)


def write(
    ds: xr.Dataset,
    cfg: ProductConfig,
    root: str | os.PathLike[str],
    *,
    layer: Layer = DEFAULT_LAYER,
) -> WriteResult:
    """Scatter ``ds`` onto the store's time axis. Idempotent and order-independent.

    Frames land at ``Timeline.index_of(ds.time)``; a repeated timestamp
    resolves last-wins, and only the rows actually touched are written. The
    concurrency rule is in the module docstring: disjoint shard-aligned
    regions, or a single writer.

    Frames that do not fall on the axis are *counted and returned*, not raised
    on: one worker holding a stray frame should not fail a whole pack, and the
    caller is the only one that knows whether it is a stray or a window that
    the store was never sized for. `meteopack pack` fails on a non-zero
    `outside`.
    """
    group = _open(cfg.store_id, root, layer, "r+")
    path = store_path(root, cfg.store_id, layer=layer)
    timeline = _timeline_of(group)
    if TIME_NAME not in ds.coords and TIME_NAME not in ds.variables:
        raise ValueError("gridded dataset needs a time coordinate")

    present = set(_data_arrays(group)) | set(PROVENANCE)
    unknown = sorted(str(n) for n in ds.data_vars if str(n) not in present)
    if unknown:
        raise ValueError(f"{cfg.store_id} has no array for {unknown}; present: {sorted(present)}")

    stamps = ds[TIME_NAME].values
    rows = timeline.index_of(stamps)
    keep = rows >= 0
    n_outside, n_off, lost = _classify_misses(stamps, rows, timeline)
    if not keep.any():
        return WriteResult(path, 0, n_outside, n_off, *lost)

    # Sorting makes the kept rows contiguous however the input was ordered, so a
    # day of frames is one region write rather than 288. Stable, so a timestamp
    # repeated inside one dataset resolves to the last frame given.
    hit = rows[keep]
    order = np.argsort(hit, kind="stable")
    hit = hit[order]
    breaks = (np.flatnonzero(np.diff(hit) != 1) + 1).tolist()

    # Encode inner chunks in batches rather than one async round per chunk:
    # 5-10% of write time on all three radar products, and the shard files
    # come out byte-identical (checked at batch sizes 1, 16, 64 and 288).
    # Scoped to this call so nothing else in the process changes behaviour.
    with zarr.config.set({"codec_pipeline.batch_size": WRITE_BATCH_SIZE}):
        _write_fields(group, ds, cfg, keep, hit, order, breaks)

    _write_provenance(group, ds, rows, keep, hit, order, breaks)
    return WriteResult(path, int(np.unique(hit).size), n_outside, n_off, *lost)


#: Inner chunks encoded per batch when writing. Flat beyond ~16; see `write`.
WRITE_BATCH_SIZE = 64


def _write_fields(group, ds, cfg, keep, hit, order, breaks) -> None:
    for name in ds.data_vars:
        target = str(name)
        if target in PROVENANCE:
            continue
        array = group[target]
        # Cast to the array's own dtype rather than to float32: the store may
        # hold integer codes with their own calibration, and converting those
        # to float and back would round values the source had exactly.
        block = np.asarray(ds[name].transpose(*DIMS).values)
        if block.shape[1:] != tuple(array.shape[1:]):
            raise ValueError(
                f"{name} frames are {block.shape[1:]}, store is {tuple(array.shape[1:])}"
            )
        if block.dtype != array.dtype:
            if array.dtype.kind in "ui" and block.dtype.kind == "f":
                raise ValueError(
                    f"{cfg.store_id}/{target} stores {array.dtype} codes, but was handed "
                    f"{block.dtype}. Encode with meteopack.core.encoding before writing; "
                    f"casting here would silently round every value."
                )
            block = block.astype(array.dtype, copy=False)
        block = block[keep]
        for run, picks in zip(np.split(hit, breaks), np.split(order, breaks), strict=True):
            array[int(run[0]) : int(run[-1]) + 1] = block[picks]


#: The per-file provenance table, beside dataset.zarr. One row per raw file.
PROVENANCE_TABLE = "provenance.parquet"


def record_id(name: str) -> int:
    """A stable id for one raw file, from its name: what `source_record_id` holds.

    A hash rather than a counter, so parallel workers and day-by-day packs agree
    on it without coordinating, and moving the archive does not change it. 63
    bits: collisions among a few million files are vanishingly unlikely, and it
    stays a non-negative int64 (-1 means "never packed").
    """
    import hashlib

    digest = hashlib.blake2b(name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def write_provenance_table(
    root: str | os.PathLike[str], store_id: str, records: Sequence[Mapping],
    *, layer: Layer = DEFAULT_LAYER,
) -> Path:
    """Upsert one row per raw file into provenance.parquet beside the store.

    Re-packing a file replaces its row; files packed earlier keep theirs, so a
    store grown day by day carries the provenance of every day.
    """
    import polars as pl

    target = store_path(root, store_id, layer=layer) / PROVENANCE_TABLE
    new = pl.DataFrame(list(records), schema_overrides={"source_record_id": pl.Int64})
    if target.exists() and len(new):
        old = pl.read_parquet(target)
        old = old.filter(~pl.col("source_record_id").is_in(new["source_record_id"].implode()))
        new = pl.concat([old, new], how="diagonal_relaxed")
    new = new.sort("first_time") if "first_time" in new.columns else new
    tmp = target.with_suffix(".parquet.part")
    new.write_parquet(tmp)
    os.replace(tmp, target)
    return target


def _write_provenance(group, ds, rows, keep, hit, order, breaks) -> None:
    """Scatter whatever provenance the dataset carries onto the same rows.

    Optional per field: a source that cannot say which layout a frame came from
    simply leaves `source_layout_id` at -1, which is what the array was
    created with and reads as "not recorded" rather than as a real layout.
    """
    for name in PROVENANCE:
        if name == "is_missing_frame" and name not in ds.variables:
            # Written unconditionally: these rows now have data behind them.
            array = group[name]
            blank = np.zeros(int(keep.sum()), dtype=array.dtype)
            for run, picks in zip(np.split(hit, breaks), np.split(order, breaks), strict=True):
                array[int(run[0]) : int(run[-1]) + 1] = blank[picks]
            continue
        if name not in ds.variables:
            continue
        array = group[name]
        values = np.asarray(ds[name].values).reshape(-1)
        if values.shape[0] != rows.shape[0]:
            raise ValueError(
                f"{name} has {values.shape[0]} entries for {rows.shape[0]} frames"
            )
        values = values.astype(array.dtype, copy=False)[keep]
        for run, picks in zip(np.split(hit, breaks), np.split(order, breaks), strict=True):
            array[int(run[0]) : int(run[-1]) + 1] = values[picks]


def _classify_misses(
    stamps: np.ndarray, rows: np.ndarray, timeline: Timeline
) -> tuple[int, int, tuple[str | None, str | None]]:
    """Split the rejected frames into "past the end of the axis" and "off grid".

    They are different problems with different fixes - one needs a bigger
    store, the other means the data is not on the cadence it claims - and a
    single "dropped" count would send you looking in the wrong place.

    Also returns the first and last frame that missed the axis. One dropped
    frame out of 770,115 took a manual walk through the archive to place; the
    two timestamps say immediately which end of the axis is short.
    """
    missed = rows < 0
    if not missed.any():
        return 0, 0, (None, None)
    when = np.asarray(stamps, dtype="datetime64[ns]").astype("int64")[missed]
    start = timeline.start_ns
    end = start + timeline.step_ns * len(timeline)
    off_axis = (when < start) | (when >= end)
    outside = int(off_axis.sum())
    span: tuple[str | None, str | None] = (None, None)
    if outside:
        lost = when[off_axis].astype("datetime64[ns]")
        span = (str(lost.min()), str(lost.max()))
    return outside, int(missed.sum()) - outside, span


def refresh_attrs(
    cfg: ProductConfig,
    root: str | os.PathLike[str],
    *,
    layer: Layer = DEFAULT_LAYER,
    layout_names: Mapping[int, str] | None = None,
) -> None:
    """Re-publish the metadata from the current config: root, fields and provenance.

    The root attributes used to be written once, when the store was created,
    so editing the config afterwards - adding a title, an institution, a
    reference - changed nothing on disk and the store kept describing itself
    with the fallback. The time bounds are left alone: those belong to the
    store, not to the config, and `extend` maintains them. Only attributes are
    touched, never data.

    `layout_names` (a source's code -> description table) becomes CF
    flag_values/flag_meanings on `source_layout_id`.
    """
    group = _open(cfg.store_id, root, layer, "r+")
    present = {n for n, _ in group.arrays()}
    for var in cfg.variables:
        if var.name in present:
            encoding = Encoding.from_config(var.encoding, name=var.name)
            group[var.name].attrs.update(_field_attrs(var, encoding))
    for name, description in PROVENANCE_DESCRIPTIONS.items():
        if name in present:
            group[name].attrs.update({"long_name": description})
    if layout_names and "source_layout_id" in present:
        group["source_layout_id"].attrs.update(_layout_flags(layout_names))
    grid = grids.get(cfg.grid_id) if cfg.grid_id else None
    if grid is None:
        return
    keep = {k: group.attrs[k] for k in ("time_start", "time_end", "n_times", "created_at")
            if k in group.attrs}
    timeline = _timeline_of(group)
    fresh = _root_attrs(cfg, grid, timeline, int(keep.get("n_times", len(timeline))))
    fresh.update(keep)
    fresh["updated_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    group.attrs.update(fresh)


def extend(
    cfg: ProductConfig,
    root: str | os.PathLike[str],
    timeline: Timeline,
    *,
    layer: Layer = DEFAULT_LAYER,
) -> int:
    """Grow an existing store's time axis forward to cover `timeline`.

    A gridded store is created with its whole axis so that every write is a
    region write into space that already exists. That is right for a backfill
    planned in one go and wrong for the operational case, where tomorrow does
    not exist yet: without this, packing day N+1 into a store sized for day N
    silently discarded the frames.

    Only forward growth is supported, and only on the cadence the store already
    has. Zarr appends on the first dimension, so existing data keeps its index
    and nothing already written moves. Extending by whole shards is not
    required for correctness here - the new region is untouched until something
    writes it - but it is what keeps a half-written shard from being shared by
    two jobs, so the growth is rounded up to a shard boundary.

    Returns the number of time steps added (0 if the axis already covered it).
    """
    group = _open(cfg.store_id, root, layer, "r+")
    current = _timeline_of(group)
    if int(timeline.cadence.total_seconds()) != int(current.cadence.total_seconds()):
        raise ValueError(
            f"{cfg.store_id} has cadence {current.cadence}, asked to extend at {timeline.cadence}"
        )
    if timeline.start_ns < current.start_ns:
        raise ValueError(
            f"{cfg.store_id} starts at {current.start}; extending backwards to "
            f"{timeline.start} would move every frame already written. Rebuild instead."
        )
    if (timeline.start_ns - current.start_ns) % current.step_ns:
        raise ValueError(
            f"{timeline.start} is not on {cfg.store_id}'s {current.cadence} grid"
        )

    have = len(current)
    need = int((timeline.start_ns - current.start_ns) // current.step_ns) + len(timeline)
    if need <= have:
        return 0

    arrays = _data_arrays(group)
    step = 1
    if arrays:
        shards = group[arrays[0]].shards
        if shards:
            step = int(shards[0])
    need = -(-need // step) * step  # whole shards

    # Every per-frame array grows, not just the fields. The provenance arrays
    # are per-frame too; leaving them at the old length made the store
    # unopenable ("conflicting sizes for dimension 'time'") and zarr silently
    # dropped every provenance write past the old end. New space reads as each
    # array's fill value - is_missing_frame=True - which is what it is.
    present = {name for name, _ in group.arrays()}
    for name in [*arrays, *(n for n in PROVENANCE if n in present)]:
        array = group[name]
        array.resize((need, *array.shape[1:]))

    # Nanoseconds, exactly as `initialise` writes them and as TIME_UNITS says.
    # This once wrote seconds, and every timestamp in a grown store read 1970.
    stamps = current.start_ns + np.arange(need, dtype="int64") * current.step_ns
    time = group[TIME_NAME]
    time.resize((need,))
    time[:] = stamps
    # `time_end` is the half-open end and timezone-aware, exactly as
    # `initialise` writes it: `_timeline_of` reads it straight back into a
    # Timeline, which rejects a naive or inclusive value.
    end = datetime.fromtimestamp(int(stamps[-1]) // _NS_PER_S, UTC) + current.cadence
    group.attrs.update({"n_times": need, "time_end": end.isoformat()})
    return need - have


def _bounds(times: np.ndarray, time_range: tuple[datetime, datetime] | None) -> tuple[int, int]:
    """Half-open ``[start, end)`` as array positions, matching the points store."""
    if time_range is None:
        return 0, len(times)
    naive = [t.astimezone(UTC).replace(tzinfo=None) if t.tzinfo else t for t in time_range]
    lo, hi = (np.datetime64(t, "ns") for t in naive)
    return int(np.searchsorted(times, lo, "left")), int(np.searchsorted(times, hi, "left"))


def read(
    store_id: str,
    root: str | os.PathLike[str],
    *,
    time_range: tuple[datetime, datetime] | None = None,
    variables: Sequence[str] | None = None,
    layer: Layer = DEFAULT_LAYER,
    decode: bool = True,
) -> xr.Dataset:
    """Read a window of the store as a ``(time, y, x)`` dataset.

    The window is resolved to positions before any chunk is touched, so
    ``time_range`` is what bounds the read: a whole-store read of a multi-year
    radar store will not fit in memory, and is not meant to.

    Integer-encoded fields come back decoded - physical values, NaN for both
    the missing and the out-of-image code - by xarray's own CF decoder, from
    the same attributes `xr.open_zarr` reads. So this and `xr.open_zarr` give
    the same numbers by construction. They did not: this returned the stored
    codes, and training bricks built from it held 1234 for 12.34 mm and
    treated the 65534/65535 sentinels as rain. `decode=False` returns the
    codes, for a caller that wants to copy them exactly.
    """
    import xarray as xr

    group = _open(store_id, root, layer, "r")
    timeline = _timeline_of(group)
    times = timeline.values()
    lo, hi = _bounds(times, time_range)
    hi = max(lo, hi)

    available = _data_arrays(group)
    names = available if variables is None else [str(n) for n in variables]
    missing = [n for n in names if n not in available]
    if missing:
        raise ValueError(f"{store_id} has no array for {missing}; present: {available}")

    data: dict = {}
    for name in names:
        variable = xr.Variable(DIMS, np.asarray(group[name][lo:hi]), dict(group[name].attrs))
        if decode:
            # One variable at a time: decoding the whole dataset would also
            # apply CF masking to the provenance coordinates.
            variable = xr.conventions.decode_cf_variable(name, variable, decode_times=False)
        data[name] = variable
    present = {n for n, _ in group.arrays()}
    coords: dict = {TIME_NAME: times[lo:hi]}
    # Provenance rides along as coordinates, not as data variables. It is
    # per-frame metadata, and a consumer asked for `data_vars` means the
    # fields - putting four bookkeeping arrays in there made `derive` try to
    # build training bricks out of is_corrupt_frame.
    for name in PROVENANCE:
        if name in present:
            coords[name] = ((TIME_NAME,), np.asarray(group[name][lo:hi]),
                            dict(group[name].attrs))
    for name in ("y", "x"):
        if name in present:
            coords[name] = ((name,), np.asarray(group[name][:]), dict(group[name].attrs))
    for name in ("lat", "lon"):
        if name in present:
            coords[name] = (("y", "x"), np.asarray(group[name][:]), dict(group[name].attrs))
    return xr.Dataset(data, coords=coords, attrs=dict(group.attrs))


def info(store_id: str, root: str | os.PathLike[str], *, layer: Layer = DEFAULT_LAYER) -> dict:
    """Describe the store from its metadata alone — not one chunk is read.

    This is what a catalog listing and `validate` print, so it has to answer
    "what shape, chunked how, covering when" without the cost of opening data.
    """
    group = _open(store_id, root, layer, "r")
    timeline = _timeline_of(group)
    names = _data_arrays(group)
    variables = {}
    for name in names:
        array = group[name]
        variables[name] = {
            "shape": tuple(int(n) for n in array.shape),
            "chunks": tuple(int(n) for n in array.chunks),
            "shards": tuple(int(n) for n in array.shards) if array.shards else None,
            "dtype": str(array.dtype),
            "units": array.attrs.get("units"),
            "long_name": array.attrs.get("long_name"),
        }
    first = variables[names[0]] if names else {}
    return {
        "store_id": store_id,
        "layer": layer,
        "path": str(store_path(root, store_id, layer=layer)),
        "grid_id": group.attrs.get("grid_id"),
        "proj4": group.attrs.get("proj4"),
        "title": group.attrs.get("title"),
        "dims": DIMS,
        "shape": first.get("shape"),
        "chunks": first.get("chunks"),
        "shards": first.get("shards"),
        "cadence_seconds": int(group.attrs["cadence_seconds"]),
        "n_times": len(timeline),
        "time_start": timeline.start.isoformat(),
        "time_end": timeline.end.isoformat(),
        "variables": variables,
    }


__all__ = [
    "DEFAULT_LAYER",
    "DIMS",
    "TIME_NAME",
    "TIME_UNITS",
    "info",
    "initialise",
    "read",
    "store_path",
    "write",
]
