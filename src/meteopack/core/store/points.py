"""Partitioned parquet store for station observations, read through DuckDB.

Layout under ``<root>/<layer>/<source>/<product>/v<n>/``::

    observations/year=2012/data.parquet      # one file per partition, or
    observations/year=2012/part-00044.parquet  # one per writer, if part_key
    stations/part-<key>.parquet

Three decisions worth stating:

*Partition values live in the path only.* They are not written into the file
body, and reads do not use hive partitioning — the body is exactly the frame
that was handed to :func:`write`, so nothing in the schema depends on how the
store happens to be partitioned. Pruning still happens: each file holds one
partition, so its parquet row-group statistics on ``time`` and ``station_id``
let DuckDB skip it without reading data. Partitioning is therefore about
bounding the unit of rewrite and keeping file sizes sane, not about the scan.

*A partition is replaced, never appended to.* Re-running a job writes a hidden
temp file and ``os.replace``-es it over the partition's file, which is atomic on
POSIX: a concurrent reader sees the old file or the new one, and re-running
cannot duplicate rows. When several Slurm array tasks each produce part of one
partition, they pass ``part_key`` (their unit key, typically a station) and each
owns a separate file inside the partition — same guarantee, finer grain, and no
task can clobber another's output.

*Station ids are text.* Upstream ids are zero-padded codes ("00044"), so they
are stored and returned as strings; that makes the matrix columns and the
station metadata joinable without a dtype negotiation.
"""
from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

from meteopack.core.config import Layer, ProductConfig, layer_path
from meteopack.core.timeline import Timeline

#: Points stores are canonical-layer artefacts; derived datasets are built from
#: them rather than written here.
DEFAULT_LAYER: Layer = "packed"

OBSERVATIONS_DIR = "observations"
STATIONS_DIR = "stations"

#: Enforced so every shard unions cleanly, whatever subset a caller supplies.
STATIONS_SCHEMA: dict[str, pl.DataType] = {
    "station_id": pl.String,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "height": pl.Float64,
    "name": pl.String,
    "updated_at": pl.Datetime("us"),
}

#: Partition keys derivable from ``time``; anything else must be a real column.
_DERIVED = {
    "year": (lambda c: c.dt.year(), "{:04d}"),
    "month": (lambda c: c.dt.month(), "{:02d}"),
    "day": (lambda c: c.dt.day(), "{:02d}"),
    "hour": (lambda c: c.dt.hour(), "{:02d}"),
    "date": (lambda c: c.dt.date(), "{}"),
}

_POLARS_TYPE: dict[str, pl.DataType] = {
    "BOOLEAN": pl.Boolean,
    "TINYINT": pl.Int8,
    "SMALLINT": pl.Int16,
    "INTEGER": pl.Int32,
    "BIGINT": pl.Int64,
    "UTINYINT": pl.UInt8,
    "USMALLINT": pl.UInt16,
    "UINTEGER": pl.UInt32,
    "UBIGINT": pl.UInt64,
    "FLOAT": pl.Float32,
    "DOUBLE": pl.Float64,
    "VARCHAR": pl.String,
    "DATE": pl.Date,
    "TIMESTAMP": pl.Datetime("us"),
    "TIMESTAMP_NS": pl.Datetime("ns"),
    "TIMESTAMP_MS": pl.Datetime("ms"),
    "BLOB": pl.Binary,
}


def store_path(root: str | os.PathLike[str], store_id: str, *, layer: Layer = DEFAULT_LAYER) -> Path:
    """Where a store with this id lives. ``store_id`` is ``source/product/vN``."""
    return layer_path(Path(root), layer, *store_id.split("/"))


def _to_polars(rel: duckdb.DuckDBPyRelation) -> pl.DataFrame:
    """DuckDB result -> polars without pyarrow (which is not a dependency).

    ``fetchnumpy`` keeps the null mask that ``fetchall`` would flatten into
    Python objects, so columns keep their types through the round trip.
    """
    names = [d[0] for d in rel.description]
    types = [str(d[1]) for d in rel.description]
    columns = rel.fetchnumpy()
    out: list[pl.Series] = []
    for name, type_name in zip(names, types, strict=True):
        arr = columns[name]
        dtype = _POLARS_TYPE.get(type_name)
        if len(arr) == 0:
            out.append(pl.Series(name, [], dtype=dtype or pl.Null))
            continue
        data = np.ma.getdata(arr)
        mask = np.ma.getmaskarray(arr)
        if data.dtype == object:
            values = [None if m else v for v, m in zip(data.tolist(), mask, strict=True)]
            out.append(pl.Series(name, values, dtype=dtype, strict=False))
            continue
        series = pl.Series(name, data)
        if mask.any():
            series = series.scatter(np.flatnonzero(mask), None)
        if dtype is not None and series.dtype != dtype:
            series = series.cast(dtype, strict=False)
        out.append(series)
    return pl.DataFrame(out)


def _slug(value: object) -> str:
    """Make a path component out of a partition value without inventing one."""
    text = str(value)
    for bad in "/\\ \t\n=":
        text = text.replace(bad, "_")
    return text or "none"


def _naive_utc_time(frame: pl.DataFrame) -> pl.DataFrame:
    """Store ``time`` as naive UTC microseconds, like :mod:`~meteopack.core.timeline`.

    A tz-aware column would make every later read depend on the reader's session
    timezone, which is exactly the class of bug that silently shifts a series.
    """
    dtype = frame.schema["time"]
    if not isinstance(dtype, pl.Datetime):
        return frame.with_columns(pl.col("time").cast(pl.Datetime("us")))
    expr = pl.col("time")
    if dtype.time_zone is not None:
        expr = expr.dt.convert_time_zone("UTC").dt.replace_time_zone(None)
    return frame.with_columns(expr.cast(pl.Datetime("us")))


def _write_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    frame.write_parquet(tmp, statistics=True)
    os.replace(tmp, path)


SCHEMA_NAME = "schema.json"


def write_schema(cfg: ProductConfig, root: str | os.PathLike[str], *,
                 layer: Layer = DEFAULT_LAYER) -> Path:
    """Publish the dataset's metadata beside the parquet.

    A gridded store carries CF attributes on its zarr group; parquet has no
    equivalent place for them, so they go in a file. Without this the points
    stores were columns of numbers with no units, no standard names and no
    statement of what the dataset is - which is exactly what the previous
    in-situ store got right, carrying KNMI's own title, institution, source,
    history and references straight through.

    Also records each variable's name in the source, so KNMI's `ta` and `dd`
    stay greppable after the rename to CF names.
    """
    payload = {
        "Conventions": "CF-1.10",
        "featureType": "timeSeries",
        "title": str(cfg.attrs.get("title") or cfg.description
                     or f"{cfg.source} {cfg.product} v{cfg.version}"),
        "store_id": cfg.store_id,
        "source_name": cfg.source,
        "product": cfg.product,
        "version": cfg.version,
        "cadence_seconds": cfg.cadence_seconds,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        **{k: v for k, v in (cfg.attrs or {}).items() if k != "title"},
        "variables": {
            var.name: {
                k: v for k, v in {
                    "units": var.units,
                    "long_name": var.long_name,
                    "standard_name": var.cf_standard_name,
                    "source_name": var.legacy_name or var.source_key,
                    "valid_min": var.valid_min,
                    "valid_max": var.valid_max,
                }.items() if v is not None
            }
            for var in cfg.variables
        },
        "columns": {
            "time": {"units": "UTC", "long_name": "observation time"},
            "station_id": {"long_name": "station identifier", "dtype": "string"},
        },
    }
    target = store_path(root, cfg.store_id, layer=layer) / SCHEMA_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, target)
    return target


def partition_labels(frame: pl.DataFrame, cfg: ProductConfig) -> pl.Series:
    """One label per row naming the partition it belongs to.

    Exposed so a writer can route rows to a partition *before* materialising
    the whole dataset. `write` derives the same labels internally; this is the
    same expressions, so the two cannot disagree about where a row goes.
    """
    keys = list(cfg.store.partition_by)
    if not keys:
        return pl.Series("__part", ["all"] * frame.height, dtype=pl.String)
    parts = []
    for key in keys:
        if key in _DERIVED:
            expr, fmt = _DERIVED[key]
            value = expr(pl.col("time"))
            parts.append(value.map_elements(lambda v, f=fmt: f.format(v), return_dtype=pl.String))
        elif key in frame.columns:
            parts.append(pl.col(key).cast(pl.String))
        else:
            raise ValueError(f"partition key {key!r} is neither derived from time nor a column")
    labelled = frame.with_columns(
        pl.concat_str([p for p in parts], separator="/").alias("__part")
    )
    return labelled["__part"]


def write(
    df: pl.DataFrame,
    cfg: ProductConfig,
    root: str | os.PathLike[str],
    *,
    layer: Layer = DEFAULT_LAYER,
    part_key: str | None = None,
) -> list[Path]:
    """Write observations into ``cfg``'s store, replacing the partitions touched.

    ``df`` needs ``time`` and ``station_id`` plus any value columns. Returns the
    files written, which is what the caller records in the catalog.
    """
    if cfg.store.kind != "points":
        raise ValueError(f"{cfg.store_id} is a {cfg.store.kind} store, not points")
    missing = {"time", "station_id"} - set(df.columns)
    if missing:
        raise ValueError(f"points frame needs {sorted(missing)}")
    if df.is_empty():
        return []

    frame = _naive_utc_time(df).with_columns(pl.col("station_id").cast(pl.String))
    keys = list(cfg.store.partition_by)
    base = store_path(root, cfg.store_id, layer=layer) / OBSERVATIONS_DIR
    filename = "data.parquet" if part_key is None else f"part-{_slug(part_key)}.parquet"

    if not keys:
        target = base / filename
        _write_atomic(frame.sort("time", "station_id"), target)
        return [target]

    helpers = []
    for i, key in enumerate(keys):
        if key in _DERIVED:
            expr = _DERIVED[key][0](pl.col("time"))
        elif key in frame.columns:
            expr = pl.col(key)
        else:
            raise ValueError(f"partition key {key!r} is neither derived from time nor a column")
        helpers.append(expr.alias(f"__part{i}"))
    tagged = frame.with_columns(helpers)
    helper_names = [f"__part{i}" for i in range(len(keys))]

    written: list[Path] = []
    for values, group in tagged.partition_by(helper_names, as_dict=True).items():
        values = values if isinstance(values, tuple) else (values,)
        parts = []
        for key, value in zip(keys, values, strict=True):
            fmt = _DERIVED[key][1] if key in _DERIVED else "{}"
            parts.append(f"{key}={_slug(fmt.format(value))}")
        target = base.joinpath(*parts, filename)
        _write_atomic(group.drop(helper_names).sort("time", "station_id"), target)
        written.append(target)
    return sorted(written)


def write_stations(
    stations: pl.DataFrame,
    store_id: str,
    root: str | os.PathLike[str],
    *,
    layer: Layer = DEFAULT_LAYER,
    part_key: str | None = None,
) -> Path:
    """Write station metadata beside the observations.

    Kept in its own parquet so a matrix can be labelled with coordinates without
    touching the observation files, which are orders of magnitude larger.
    """
    if "station_id" not in stations.columns:
        raise ValueError("station metadata needs a station_id column")
    frame = stations.with_columns(
        [
            pl.col(name).cast(dtype) if name in stations.columns else pl.lit(None, dtype).alias(name)
            for name, dtype in STATIONS_SCHEMA.items()
            if name != "updated_at"
        ]
    )
    if "updated_at" not in stations.columns:
        frame = frame.with_columns(
            pl.lit(datetime.now(UTC).replace(tzinfo=None)).alias("updated_at")
        )
    frame = frame.select(list(STATIONS_SCHEMA)).cast(STATIONS_SCHEMA)  # type: ignore[arg-type]
    name = "data.parquet" if part_key is None else f"part-{_slug(part_key)}.parquet"
    target = store_path(root, store_id, layer=layer) / STATIONS_DIR / name
    _write_atomic(frame.unique(subset=["station_id"], keep="last"), target)
    return target


def read_stations(
    store_id: str, root: str | os.PathLike[str], *, layer: Layer = DEFAULT_LAYER
) -> pl.DataFrame:
    """Station metadata, one row per station — latest ``updated_at`` wins."""
    directory = store_path(root, store_id, layer=layer) / STATIONS_DIR
    shards = sorted(directory.glob("*.parquet"))
    if not shards:
        return pl.DataFrame(schema=STATIONS_SCHEMA)
    frame = pl.read_parquet(shards)
    return (
        frame.sort("updated_at")
        .unique(subset=["station_id"], keep="last")
        .sort("station_id")
    )


def _sql_timestamp(value: datetime) -> str:
    if value.tzinfo is not None:
        value = value.astimezone(UTC).replace(tzinfo=None)
    return f"TIMESTAMP '{value.isoformat(sep=' ', timespec='microseconds')}'"


def _sql_text(value: object) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def read(
    store_id: str,
    root: str | os.PathLike[str],
    *,
    time_range: tuple[datetime, datetime] | None = None,
    station_ids: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
    layer: Layer = DEFAULT_LAYER,
) -> pl.DataFrame:
    """Read observations, pushing the filters down into the parquet scan.

    ``time_range`` is half-open ``[start, end)``, matching
    :class:`~meteopack.core.timeline.Timeline` and ``TimeWindow``. Both filters
    and the column projection go into the SQL, so DuckDB skips row groups and
    columns instead of materialising the store and filtering afterwards.
    """
    directory = store_path(root, store_id, layer=layer) / OBSERVATIONS_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"no points store at {directory}")
    if next(directory.rglob("*.parquet"), None) is None:
        return pl.DataFrame()

    pattern = str(directory / "**" / "*.parquet").replace("'", "''")
    projection = ", ".join(f'"{c}"' for c in columns) if columns else "*"
    where: list[str] = []
    if time_range is not None:
        start, end = time_range
        where.append(f'"time" >= {_sql_timestamp(start)} AND "time" < {_sql_timestamp(end)}')
    if station_ids is not None:
        wanted = [_sql_text(s) for s in station_ids]
        where.append(f'"station_id" IN ({", ".join(wanted)})' if wanted else "FALSE")
    clause = f" WHERE {' AND '.join(where)}" if where else ""

    con = duckdb.connect()
    try:
        sql = (
            f"SELECT {projection} FROM "
            f"read_parquet('{pattern}', union_by_name=true, hive_partitioning=false)"
            f'{clause} ORDER BY "time", "station_id"'
        )
        return _to_polars(con.sql(sql))
    finally:
        con.close()


def to_matrix(
    store_id: str,
    root: str | os.PathLike[str],
    variable: str,
    timeline: Timeline,
    *,
    layer: Layer = DEFAULT_LAYER,
    station_ids: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Dense ``(time, station)`` float32 array for ``variable`` on ``timeline``.

    Gaps stay NaN — an absent observation is not a zero. Rows whose timestamp is
    not exactly on the cadence get index -1 from
    :meth:`Timeline.index_of` and are dropped rather than snapped, so a cadence
    mismatch shows up as missing data in validation instead of a shifted series.
    Placement is a numpy integer scatter, which is why no pandas reindex (and no
    pandas) appears anywhere in this module.
    """
    frame = read(
        store_id,
        root,
        time_range=(timeline.start, timeline.end),
        station_ids=station_ids,
        columns=["time", "station_id", variable],
        layer=layer,
    )
    ids = list(station_ids) if station_ids is not None else []
    if frame.is_empty() and not ids:
        return np.empty((len(timeline), 0), dtype=np.float32), []
    if not ids:
        ids = sorted(frame["station_id"].unique().to_list())

    values = np.full((len(timeline), len(ids)), np.nan, dtype=np.float32)
    if frame.is_empty():
        return values, ids

    rows = timeline.index_of(frame["time"].to_numpy())
    # object dtype on purpose: a fixed-width "<U" array would silently truncate
    # a station id longer than the longest one in `ids` and mismatch it.
    lookup = np.asarray(ids, dtype=object)
    order = np.argsort(lookup, kind="stable")
    observed = frame["station_id"].to_numpy().astype(object)
    hit = np.clip(np.searchsorted(lookup[order], observed), 0, len(ids) - 1)
    pos = order[hit]
    known = lookup[pos] == observed
    keep = (rows >= 0) & known
    data = frame[variable].cast(pl.Float64).to_numpy().astype(np.float32)
    values[rows[keep], pos[keep]] = data[keep]
    return values, ids


__all__ = [
    "DEFAULT_LAYER",
    "OBSERVATIONS_DIR",
    "STATIONS_DIR",
    "STATIONS_SCHEMA",
    "read",
    "read_stations",
    "store_path",
    "to_matrix",
    "write",
    "write_stations",
]
