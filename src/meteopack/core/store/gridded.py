"""zarr v3 store for ``(time, y, x)`` fields: one frame per chunk, a day per shard.

Layout under ``<root>/<layer>/<source>/<product>/v<n>/``: a zarr v3 group holding
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

import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import zarr
from zarr.codecs import BloscCodec, GzipCodec, ZstdCodec

from meteopack.core import grids
from meteopack.core.config import Layer, ProductConfig, StoreConfig, layer_path
from meteopack.core.timeline import Timeline

if TYPE_CHECKING:  # xarray is a caller-facing type here, not an import-time cost
    import xarray as xr

#: Gridded stores are canonical-layer artefacts; training sets are derived from them.
DEFAULT_LAYER: Layer = "canonical"

TIME_NAME = "time"
DIMS: tuple[str, str, str] = ("time", "y", "x")

#: Seconds, not nanoseconds: CF-decodable by any reader, and exact for an integer
#: cadence. Sub-second axes are rejected in `initialise` rather than rounded.
TIME_UNITS = "seconds since 1970-01-01T00:00:00"

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


def _open(store_id: str, root: str | os.PathLike[str], layer: Layer, mode: str) -> zarr.Group:
    path = store_path(root, store_id, layer=layer)
    if not (path / "zarr.json").exists():
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


def _data_arrays(group: zarr.Group) -> list[str]:
    return [name for name, _ in group.arrays() if name != TIME_NAME]


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

    path = store_path(root, cfg.store_id, layer=layer)
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists; pass overwrite=True to rebuild it")
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    group = zarr.open_group(path, mode="w", zarr_format=3)
    group.attrs.update(
        {
            "Conventions": "CF-1.10",
            "title": str(cfg.extra.get("title") or f"{cfg.source} {cfg.product} v{cfg.version}"),
            "store_id": cfg.store_id,
            "source": cfg.source,
            "product": cfg.product,
            "grid_id": grid.id,
            "proj4": grid.proj4,
            "cadence_seconds": cfg.cadence_seconds,
            "time_start": timeline.start.isoformat(),
            "time_end": timeline.end.isoformat(),
            "n_times": shape[0],
        }
    )

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
            "axis": "T",
        },
    )
    time[:] = timeline.values().astype("int64") // _NS_PER_S

    declared = {v.name: v for v in cfg.variables}
    for name in names:
        var = declared[name]
        group.create_array(
            name,
            shape=shape,
            chunks=chunks,
            shards=shards,
            dtype="float32",
            fill_value=np.float32("nan"),
            compressors=_compressors(cfg.store),
            dimension_names=DIMS,
            attributes={
                "units": var.units,
                "long_name": var.long_name or name,
                "grid_id": grid.id,
                "coordinates": TIME_NAME,
            },
        )
    return path


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

    present = set(_data_arrays(group))
    unknown = sorted(str(n) for n in ds.data_vars if str(n) not in present)
    if unknown:
        raise ValueError(f"{cfg.store_id} has no array for {unknown}; present: {sorted(present)}")

    stamps = ds[TIME_NAME].values
    rows = timeline.index_of(stamps)
    keep = rows >= 0
    n_outside, n_off = _classify_misses(stamps, rows, timeline)
    if not keep.any():
        return WriteResult(path, 0, n_outside, n_off)

    # Sorting makes the kept rows contiguous however the input was ordered, so a
    # day of frames is one region write rather than 288. Stable, so a timestamp
    # repeated inside one dataset resolves to the last frame given.
    hit = rows[keep]
    order = np.argsort(hit, kind="stable")
    hit = hit[order]
    breaks = (np.flatnonzero(np.diff(hit) != 1) + 1).tolist()

    for name in ds.data_vars:
        array = group[str(name)]
        block = np.asarray(ds[name].transpose(*DIMS).values, dtype="float32")
        if block.shape[1:] != tuple(array.shape[1:]):
            raise ValueError(
                f"{name} frames are {block.shape[1:]}, store is {tuple(array.shape[1:])}"
            )
        block = block[keep]
        for run, picks in zip(np.split(hit, breaks), np.split(order, breaks), strict=True):
            array[int(run[0]) : int(run[-1]) + 1] = block[picks]
    return WriteResult(path, int(np.unique(hit).size), n_outside, n_off)


def _classify_misses(
    stamps: np.ndarray, rows: np.ndarray, timeline: Timeline
) -> tuple[int, int]:
    """Split the rejected frames into "past the end of the axis" and "off grid".

    They are different problems with different fixes - one needs a bigger
    store, the other means the data is not on the cadence it claims - and a
    single "dropped" count would send you looking in the wrong place.
    """
    missed = rows < 0
    if not missed.any():
        return 0, 0
    when = np.asarray(stamps, dtype="datetime64[ns]").astype("int64")[missed]
    start = timeline.start_ns
    end = start + timeline.step_ns * len(timeline)
    outside = int(((when < start) | (when >= end)).sum())
    return outside, int(missed.sum()) - outside


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

    for name in arrays:
        array = group[name]
        array.resize((need, *array.shape[1:]))

    stamps = (
        current.start_ns + np.arange(need, dtype="int64") * current.step_ns
    ) // _NS_PER_S
    time = group[TIME_NAME]
    time.resize((need,))
    time[:] = stamps
    # `time_end` is the half-open end and timezone-aware, exactly as
    # `initialise` writes it: `_timeline_of` reads it straight back into a
    # Timeline, which rejects a naive or inclusive value.
    end = datetime.fromtimestamp(int(stamps[-1]), UTC) + current.cadence
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
) -> xr.Dataset:
    """Read a window of the store as a ``(time, y, x)`` dataset.

    The window is resolved to positions before any chunk is touched, so
    ``time_range`` is what bounds the read: a whole-store read of a multi-year
    radar store will not fit in memory, and is not meant to.
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

    data = {
        name: (DIMS, np.asarray(group[name][lo:hi]), dict(group[name].attrs)) for name in names
    }
    return xr.Dataset(data, coords={TIME_NAME: times[lo:hi]}, attrs=dict(group.attrs))


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
