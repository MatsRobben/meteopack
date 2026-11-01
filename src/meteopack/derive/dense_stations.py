"""Points store -> dense ``(time, station)`` matrices on the product time axis.

Stations are irregular in both axes: a station reports when it reports, and the
set of reporting stations changes over the years. A model wants a rectangle. So
this module projects the parquet store onto the target timeline with
:func:`meteopack.core.store.points.to_matrix` — gaps stay NaN, because an
absent observation is not a zero — and writes the result as a small zarr store.

Station metadata travels with the matrix as coordinates on the ``station``
dimension (``station_id``, ``lat``, ``lon``, ``height``, ``name``, ``source``),
in exactly the column order of the matrix. That is the whole point: a model that
is handed a column of temperatures and cannot say where they were measured
cannot learn anything spatial, and a coordinate array that is merely *sorted the
same way* is one upstream change away from being silently misaligned.

Several input stores are concatenated along ``station``. Their station ids must
be disjoint — a collision is raised rather than resolved, because two networks
that happen to both use "00044" are not the same station.
"""
from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import xarray as xr

from meteopack.core.config import Layer, layer_path
from meteopack.core.store import manifest as manifest_mod
from meteopack.core.store import points
from meteopack.derive.base import DeriveSpec

DEFAULT_LAYER: Layer = "training"
SOURCE_LAYER: Layer = "packed"

MATRIX_DIRNAME = "matrix.zarr"

TIME_DIM = "time"
STATION_DIM = "station"

#: Carried from the station table onto the station dimension.
STATION_COORDS = ("lat", "lon", "height", "name")

#: Never a model input, always present in the observation frame.
_NOT_VARIABLES = {"time", "station_id"}


def store_path(spec: DeriveSpec, root: str | os.PathLike[str], *,
               layer: Layer = DEFAULT_LAYER) -> Path:
    return layer_path(Path(root), layer, *spec.store_id.split("/"))


def matrix_path(spec: DeriveSpec, root: str | os.PathLike[str], *,
                layer: Layer = DEFAULT_LAYER) -> Path:
    return store_path(spec, root, layer=layer) / MATRIX_DIRNAME


def available_variables(
    store_id: str,
    root: str | os.PathLike[str],
    *,
    layer: Layer = SOURCE_LAYER,
) -> list[str]:
    """Value columns of a points store, read from a parquet footer.

    Cheap on purpose: the alternative is scanning the store to find out what is
    in it, and these stores are years of observations.
    """
    directory = points.store_path(root, store_id, layer=layer) / points.OBSERVATIONS_DIR
    shard = next(iter(sorted(directory.rglob("*.parquet"))), None)
    if shard is None:
        raise FileNotFoundError(f"no parquet files under {directory}")
    schema = pl.scan_parquet(shard).collect_schema()
    return [name for name in schema.names() if name not in _NOT_VARIABLES]


def _observed_stations(
    store_id: str, root: Path, spec: DeriveSpec, *, layer: Layer
) -> list[str]:
    """Stations that actually report inside the target window, sorted."""
    frame = points.read(
        store_id,
        root,
        time_range=(spec.timeline.start, spec.timeline.end),
        columns=["station_id"],
        layer=layer,
    )
    if frame.is_empty():
        return []
    return sorted(frame["station_id"].unique().to_list())


def _station_frame(
    ids: list[str], store_id: str, root: Path, *, layer: Layer
) -> pl.DataFrame:
    """Metadata for ``ids``, in ``ids`` order, with nulls where unknown."""
    wanted = pl.DataFrame({"station_id": pl.Series(ids, dtype=pl.String)})
    known = points.read_stations(store_id, root, layer=layer)
    if known.is_empty():
        known = pl.DataFrame(schema=points.STATIONS_SCHEMA)
    joined = wanted.join(known.drop("updated_at"), on="station_id", how="left")
    return joined.with_columns(pl.lit(store_id).alias("source"))


def to_dataset(
    spec: DeriveSpec,
    root: str | os.PathLike[str],
    *,
    variables: Sequence[str] | None = None,
    station_ids: Sequence[str] | None = None,
    source_layer: Layer = SOURCE_LAYER,
) -> xr.Dataset:
    """Build the ``(time, station)`` dataset in memory. No output is written.

    ``root`` is where the *input* points stores live. A variable a store does
    not carry is NaN for that store's columns rather than an error, so two
    networks with different instrument sets can be concatenated.
    """
    root = Path(root)
    timeline = spec.timeline
    blocks: list[tuple[str, list[str], dict[str, np.ndarray]]] = []
    for store_id in spec.input_store_ids:
        have = set(available_variables(store_id, root, layer=source_layer))
        names = [n for n in (variables if variables is not None else sorted(have)) if n in have]
        if not names:
            raise ValueError(f"{store_id} provides none of {list(variables or sorted(have))}")
        # One station list for the whole store, fixed before any matrix is built:
        # letting to_matrix derive it per variable would give two variables
        # different column sets whenever they are reported by different stations.
        ids = (
            list(station_ids)
            if station_ids is not None
            else _observed_stations(store_id, root, spec, layer=source_layer)
        )
        columns = {
            name: points.to_matrix(
                store_id, root, name, timeline, layer=source_layer, station_ids=ids
            )[0]
            for name in names
        }
        blocks.append((store_id, ids, columns))

    all_ids = [sid for _, ids, _ in blocks for sid in ids]
    if len(set(all_ids)) != len(all_ids):
        duplicated = sorted({s for s in all_ids if all_ids.count(s) > 1})
        raise ValueError(f"station ids collide across inputs: {duplicated}")

    all_names = sorted({name for _, _, columns in blocks for name in columns})
    data: dict[str, tuple[tuple[str, str], np.ndarray]] = {}
    for name in all_names:
        parts = [
            columns[name]
            if name in columns
            else np.full((len(timeline), len(ids)), np.nan, dtype=np.float32)
            for _, ids, columns in blocks
        ]
        data[name] = ((TIME_DIM, STATION_DIM), np.concatenate(parts, axis=1))

    meta = pl.concat(
        [_station_frame(ids, store_id, root, layer=source_layer) for store_id, ids, _ in blocks]
    ) if blocks else pl.DataFrame()

    coords: dict[str, object] = {
        TIME_DIM: timeline.values(),
        STATION_DIM: np.asarray(all_ids, dtype=object),
    }
    for name in (*STATION_COORDS, "source"):
        column = meta[name] if name in meta.columns else None
        if column is None:
            continue
        if column.dtype == pl.String:
            values = np.asarray(["" if v is None else v for v in column.to_list()], dtype=object)
        else:
            values = column.cast(pl.Float64).to_numpy()
        coords[name] = (STATION_DIM, values)

    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(
        {
            "store_id": spec.store_id,
            "fingerprint": spec.fingerprint,
            "inputs": list(spec.input_store_ids),
            "cadence_seconds": spec.cfg.cadence_seconds,
            "time_start": timeline.start.isoformat(),
            "time_end": timeline.end.isoformat(),
        }
    )
    return ds


def build(
    spec: DeriveSpec,
    root: str | os.PathLike[str],
    *,
    variables: Sequence[str] | None = None,
    station_ids: Sequence[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
    source_layer: Layer = SOURCE_LAYER,
    layer: Layer = DEFAULT_LAYER,
    time_chunk: int = 4320,
) -> Path:
    """Write the matrices to ``<root>/derived/<name>/vN/matrix.zarr``.

    Rewritten whole rather than appended to: one of these is a few floats per
    station per step, and rebuilding is cheaper than reasoning about a partial
    overwrite.
    """
    source_root = Path(source_root) if source_root is not None else Path(root)
    ds = to_dataset(
        spec,
        source_root,
        variables=variables,
        station_ids=station_ids,
        source_layer=source_layer,
    )
    store = store_path(spec, root, layer=layer)
    store.mkdir(parents=True, exist_ok=True)
    path = store / MATRIX_DIRNAME
    n_time, n_station = len(ds[TIME_DIM]), len(ds[STATION_DIM])
    encoding = {
        str(name): {"chunks": (min(time_chunk, n_time) or 1, n_station or 1)}
        for name in ds.data_vars
    }
    ds.to_zarr(path, mode="w", zarr_format=3, consolidated=False, encoding=encoding)

    existing = manifest_mod.read(store)
    entry = manifest_mod.Manifest.new(
        spec.store_id,
        layer,
        "dense_stations",
        spec.cfg,
        n_records=n_time * n_station,
        time_min=spec.timeline.start.isoformat(),
        time_max=spec.timeline.end.isoformat(),
        inputs=list(spec.input_store_ids),
        extra={
            "fingerprint": spec.fingerprint,
            "n_stations": n_station,
            "n_times": n_time,
            "variables": sorted(str(v) for v in ds.data_vars),
        },
    )
    if existing is not None:
        entry.created_at = existing.created_at
    manifest_mod.write(store, entry)
    return path


def read(spec: DeriveSpec, root: str | os.PathLike[str], *,
         layer: Layer = DEFAULT_LAYER) -> xr.Dataset:
    path = matrix_path(spec, root, layer=layer)
    if not path.exists():
        raise FileNotFoundError(f"no station matrix at {path}")
    return xr.open_zarr(path, consolidated=False)


__all__ = [
    "DEFAULT_LAYER",
    "MATRIX_DIRNAME",
    "STATION_COORDS",
    "STATION_DIM",
    "TIME_DIM",
    "available_variables",
    "build",
    "matrix_path",
    "read",
    "store_path",
    "to_dataset",
]
