"""Parquet catalog + job ledger: what was fetched, what was built, what is done.

Three tables live under ``<data_root>/catalog/``:

* ``files``  — one row per fetched raw file (provenance + checksum).
* ``stores`` — one row per built store (what it covers, and from which config
  and code version), which is what lets a garbage collector decide whether a
  layer is safely reproducible before deleting it.
* ``jobs``   — the ledger a re-runnable stage consults so ``plan`` can skip
  work that already completed.

Concurrency
-----------
Jobs run as Slurm array tasks: dozens of processes on different nodes append to
these tables at the same time, over a shared filesystem with no lock manager we
trust. So a table is not a file, it is a *directory of shards* — one parquet
file per :class:`Catalog` instance, named after host/pid/array-task. A writer
only ever rewrites its own shard, and does so by writing a hidden temp file and
``os.replace``-ing it into place, which is atomic on POSIX. Consequences:

* no process ever reads-modifies-writes a file another process may write, so
  there is no lost-update window and no lock;
* a reader globbing ``*.parquet`` sees each shard either fully before or fully
  after a flush, never torn (the temp file does not match the glob);
* the tables are append-only, so history is kept and a crashed task leaves at
  worst a stale-but-valid shard.

The price is file count and duplicate rows. File count is bounded by the number
of writer processes (not rows), and duplicates are resolved on read: rows are
ordered by their timestamp and the last row per key wins. Shards can be
compacted offline at any time by rewriting the deduplicated frame as a single
shard and deleting the rest — nothing here depends on shard identity.

Timestamps are stored as naive UTC microseconds, matching
:mod:`meteopack.core.timeline`, so nothing depends on a reader's session
timezone.
"""
from __future__ import annotations

import os
import socket
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Self

import duckdb
import numpy as np
import polars as pl

from meteopack.core.config import Layer
from meteopack.sources.base import RawFile

Status = Literal["pending", "done", "failed"]

CATALOG_DIRNAME = "_bookkeeping"

#: The pre-rename name. Read from it when it is the only one present, so an
#: existing data root does not lose its ledger and start redoing finished work.
LEGACY_CATALOG_DIRNAME = "catalog"

#: Table name -> column schema. Written explicitly on every shard so that a
#: shard whose column happens to be all-null does not drift to a different type
#: and break the union across shards.
SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "files": {
        "unit_key": pl.String,
        "source": pl.String,
        "product": pl.String,
        "url": pl.String,
        "path": pl.String,
        "sha256": pl.String,
        "n_bytes": pl.Int64,
        "valid_time": pl.Datetime("us"),
        "fetched_at": pl.Datetime("us"),
        # What the file holds, from the source's integrity check at fetch time
        # (null when the source has none). See `sources.base.Content`.
        "n_records": pl.Int64,
        "time_min": pl.Datetime("us"),
        "time_max": pl.Datetime("us"),
    },
    "stores": {
        "store_id": pl.String,
        "layer": pl.String,
        "kind": pl.String,
        "path": pl.String,
        "version": pl.Int64,
        "n_records": pl.Int64,
        "time_min": pl.Datetime("us"),
        "time_max": pl.Datetime("us"),
        "config_hash": pl.String,
        "code_version": pl.String,
        "created_at": pl.Datetime("us"),
    },
    "jobs": {
        "job_key": pl.String,
        "source": pl.String,
        "product": pl.String,
        "stage": pl.String,
        "unit_key": pl.String,
        "status": pl.String,
        "attempts": pl.Int64,
        "error": pl.String,
        "updated_at": pl.Datetime("us"),
        # Why a unit ended where it did - `status` says only whether to redo it.
        # One of OUTCOMES; null on rows written before the column existed.
        "outcome": pl.String,
        "http_status": pl.Int64,
        "error_class": pl.String,
    },
}

#: `jobs.outcome` values. `status` decides whether a re-run redoes the unit
#: (only `done` is skipped); `outcome` records what actually happened:
#:
#: * ``done``   - downloaded (or already on disk) and passed its check
#: * ``empty``  - upstream answered, with nothing in it (EA's pre-opening years)
#: * ``absent`` - upstream says it does not exist (HTTP 404)
#: * ``failed`` - gave up; `error_class`, `http_status`, `error`, `attempts` say why
#:
#: A `done` or `empty` unit still changing upstream keeps `status = pending`.
OUTCOMES = ("done", "empty", "absent", "failed")

#: (key column, ordering column) per table — the "latest row wins" rule.
_DEDUP: dict[str, tuple[str, str]] = {
    "files": ("unit_key", "fetched_at"),
    "stores": ("store_id", "created_at"),
    "jobs": ("job_key", "updated_at"),
}

_DUCKDB_TYPE = {
    pl.String: "VARCHAR",
    pl.Int64: "BIGINT",
    pl.Datetime("us"): "TIMESTAMP",
}

_TS_LOCK = threading.Lock()
_LAST_TS = datetime(1970, 1, 1)
_MICROSECOND = timedelta(microseconds=1)


def _now() -> datetime:
    """A strictly increasing naive-UTC clock, per process.

    ``datetime.now`` has microsecond resolution, which two marks on the same job
    can share; the dedup rule ("latest row wins") would then be ambiguous
    exactly where it matters — pending -> done. Nudging forward by a microsecond
    keeps ordering total within a process, which is where same-key races happen.
    """
    global _LAST_TS
    with _TS_LOCK:
        now = datetime.now(UTC).replace(tzinfo=None)
        if now <= _LAST_TS:
            now = _LAST_TS + _MICROSECOND
        _LAST_TS = now
        return now


def _naive_utc(value: datetime | None) -> datetime | None:
    """Coerce an aware datetime to naive UTC; pass naive ones through."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def default_writer_id() -> str:
    """Identify this writer's shard: when, host, pid, Slurm array task, nonce.

    The nonce matters: two :class:`Catalog` objects in one process (threads, or
    a re-run after a crash) must not share a shard path, or they would clobber
    each other's rewrites.

    The timestamp leads so that a plain `ls` of the table directory is in
    chronological order. These names are otherwise unreadable, and a directory
    of a few hundred opaque names is the first thing that makes the catalog
    look like something has gone wrong.
    """
    task = os.environ.get("SLURM_ARRAY_TASK_ID") or os.environ.get("SLURM_PROCID") or "0"
    when = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{when}-{socket.gethostname()}-{os.getpid()}-t{task}-{uuid.uuid4().hex[:6]}"


def job_key(stage: str, unit_key: str) -> str:
    """The ledger key for one unit at one stage — stable across runs."""
    return f"{stage}:{unit_key}"


def split_job_key(key: str) -> tuple[str, str]:
    """Inverse of `job_key`: ("fetch", "dwd/00044/2012") from "fetch:dwd/00044/2012".

    The key is the single source of truth for which stage a row belongs to. The
    `stage` column is derived from it rather than supplied separately, because a
    caller that passed the key but forgot `stage=` would leave the column empty,
    `pending()` would never see the unit as done, and every run would re-fetch
    it — the one failure mode a rate-limited source cannot absorb.
    """
    stage, _, unit = key.partition(":")
    return stage, unit


def _to_polars(rel: duckdb.DuckDBPyRelation) -> pl.DataFrame:
    """DuckDB result -> polars without pyarrow (which is not a dependency).

    ``fetchnumpy`` gives masked arrays, which carry the null mask that
    ``fetchall`` would flatten into Python objects; the mask is reapplied here.
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
            # Strings and nested types come back as object arrays, which polars
            # will not scatter into; go via a Python list once, here only.
            values = [None if m else v for v, m in zip(data.tolist(), mask, strict=True)]
            series = pl.Series(name, values, dtype=dtype, strict=False)
        else:
            series = pl.Series(name, data)
            if mask.any():
                series = series.scatter(np.flatnonzero(mask), None)
            if dtype is not None and series.dtype != dtype:
                series = series.cast(dtype, strict=False)
        out.append(series)
    return pl.DataFrame(out)


_POLARS_TYPE: dict[str, pl.DataType] = {
    "BOOLEAN": pl.Boolean,
    "TINYINT": pl.Int8,
    "SMALLINT": pl.Int16,
    "INTEGER": pl.Int32,
    "BIGINT": pl.Int64,
    "HUGEINT": pl.Int128,
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
    "TIMESTAMP_S": pl.Datetime("ms"),
    "BLOB": pl.Binary,
}


def _empty_select(schema: dict[str, pl.DataType]) -> str:
    """A typed, row-less relation for a table that has no shards yet.

    DuckDB's ``read_parquet`` errors on a glob that matches nothing, and an
    empty catalog is the normal state on the first run, so queries are served
    from this instead of from a seed file written to disk.
    """
    cols = ", ".join(f"CAST(NULL AS {_DUCKDB_TYPE[dt]}) AS {name}" for name, dt in schema.items())
    return f"SELECT {cols} WHERE 1=0"


class Catalog:
    """The catalog and ledger rooted at ``<root>/catalog/``.

    One instance owns one shard per table; it is cheap to create and safe to
    create many, but not safe to share across processes (each process makes its
    own, which is the point).
    """

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        writer_id: str | None = None,
        autoflush: bool = True,
    ) -> None:
        self.root = catalog_path(root)
        self.writer_id = writer_id or default_writer_id()
        self.autoflush = autoflush
        self._rows: dict[str, list[dict]] = {name: [] for name in SCHEMAS}
        self._dirty: set[str] = set()
        self._attempts: dict[str, int] = {}
        self._lock = threading.Lock()
        # Directories are made on first write, not here: opening the catalog to
        # read it - `list`, `validate`, a dry run - must not create a data root
        # in whatever directory the command happened to run in.

    # ------------------------------------------------------------------ write

    def shard_path(self, table: str) -> Path:
        return self.root / table / f"{self.writer_id}.parquet"

    def _append(self, table: str, row: dict) -> None:
        with self._lock:
            self._rows[table].append(row)
            self._dirty.add(table)
        if self.autoflush:
            self.flush()

    def flush(self) -> None:
        """Rewrite this writer's dirty shards atomically.

        The whole shard is rewritten rather than appended to, because parquet
        has a trailing footer and cannot be appended to in place. That is O(rows
        this process wrote), which is bounded by one task's work.
        """
        with self._lock:
            tables = sorted(self._dirty)
            snapshots = {t: list(self._rows[t]) for t in tables}
            self._dirty.clear()
        for table in tables:
            rows = snapshots[table]
            if not rows:
                continue
            frame = pl.DataFrame(rows, schema=SCHEMAS[table])
            path = self.shard_path(table)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
            frame.write_parquet(tmp)
            os.replace(tmp, path)

    def record_file(
        self,
        raw: RawFile,
        *,
        source: str,
        product: str,
        url: str,
        valid_time: datetime | None = None,
        fetched_at: datetime | None = None,
    ) -> None:
        """Record a fetched unit. Takes the :class:`RawFile` the fetch returned
        so the checksum recorded here is the one computed at fetch time."""
        content = getattr(raw, "content", None)
        self._append(
            "files",
            {
                "unit_key": raw.unit_key,
                "source": source,
                "product": product,
                "url": url,
                "path": str(raw.path),
                "sha256": raw.sha256,
                "n_bytes": int(raw.n_bytes),
                "valid_time": _naive_utc(valid_time),
                "fetched_at": _naive_utc(fetched_at) or _now(),
                "n_records": None if content is None else int(content.n_records),
                "time_min": None if content is None else _naive_utc(content.time_min),
                "time_max": None if content is None else _naive_utc(content.time_max),
            },
        )

    def record_store(
        self,
        *,
        store_id: str,
        layer: Layer,
        kind: str,
        path: str | os.PathLike[str],
        version: int,
        n_records: int,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        config_hash: str = "",
        code_version: str = "",
        created_at: datetime | None = None,
    ) -> None:
        """Record a built store. ``config_hash`` and ``code_version`` are what
        make the claim "this layer can be rebuilt" checkable rather than hoped."""
        self._append(
            "stores",
            {
                "store_id": store_id,
                "layer": layer,
                "kind": kind,
                "path": str(path),
                "version": int(version),
                "n_records": int(n_records),
                "time_min": _naive_utc(time_min),
                "time_max": _naive_utc(time_max),
                "config_hash": config_hash,
                "code_version": code_version,
                "created_at": _naive_utc(created_at) or _now(),
            },
        )

    def mark(
        self,
        key: str,
        status: Status,
        *,
        source: str = "",
        product: str = "",
        stage: str = "",
        unit_key: str = "",
        attempts: int | None = None,
        error: str | None = None,
        updated_at: datetime | None = None,
        outcome: str | None = None,
        http_status: int | None = None,
        error_class: str | None = None,
    ) -> None:
        """Append a ledger row for ``key``.

        ``attempts`` defaults to this process's count for the key, which is the
        number a retry loop actually cares about; pass it explicitly to carry a
        count across processes. `fetch` passes the number of HTTP requests
        the unit took. ``outcome`` is one of :data:`OUTCOMES`.
        """
        if outcome is not None and outcome not in OUTCOMES:
            raise ValueError(f"outcome must be one of {OUTCOMES}, got {outcome!r}")
        with self._lock:
            self._attempts[key] = self._attempts.get(key, 0) + 1
            counted = self._attempts[key]
        derived_stage, derived_unit = split_job_key(key)
        self._append(
            "jobs",
            {
                "job_key": key,
                "source": source,
                "product": product,
                "stage": stage or derived_stage,
                "unit_key": unit_key or derived_unit,
                "status": status,
                "attempts": int(counted if attempts is None else attempts),
                "error": error,
                "updated_at": _naive_utc(updated_at) or _now(),
                "outcome": outcome,
                "http_status": None if http_status is None else int(http_status),
                "error_class": error_class,
            },
        )

    # ------------------------------------------------------------------- read

    def _shards(self, table: str) -> list[str]:
        return sorted(str(p) for p in (self.root / table).glob("*.parquet"))

    def _relation(self, table: str) -> str:
        """SQL for one deduplicated table: latest row per key wins."""
        shards = self._shards(table)
        if not shards:
            return _empty_select(SCHEMAS[table])
        key, order = _DEDUP[table]
        files = ", ".join("'" + s.replace("'", "''") + "'" for s in shards)
        return (
            f"SELECT * FROM read_parquet([{files}], union_by_name=true) "
            f"QUALIFY row_number() OVER (PARTITION BY {key} ORDER BY {order} DESC) = 1"
        )

    def _connect(self) -> duckdb.DuckDBPyConnection:
        self.flush()
        con = duckdb.connect()
        for table in SCHEMAS:
            con.execute(f"CREATE VIEW {table} AS {self._relation(table)}")
            con.execute(f"CREATE VIEW {table}_all AS {self._raw_relation(table)}")
        return con

    def _raw_relation(self, table: str) -> str:
        shards = self._shards(table)
        if not shards:
            return _empty_select(SCHEMAS[table])
        files = ", ".join("'" + s.replace("'", "''") + "'" for s in shards)
        return f"SELECT * FROM read_parquet([{files}], union_by_name=true)"

    def query(self, sql: str) -> pl.DataFrame:
        """Run SQL over the catalog.

        Views ``files``/``stores``/``jobs`` are deduplicated; ``*_all`` expose
        every appended row, which is how you audit a job's retry history.
        """
        con = self._connect()
        try:
            return _to_polars(con.sql(sql))
        finally:
            con.close()

    def files(self) -> pl.DataFrame:
        return self.query("SELECT * FROM files")

    def stores(self) -> pl.DataFrame:
        return self.query("SELECT * FROM stores")

    def jobs(self) -> pl.DataFrame:
        return self.query("SELECT * FROM jobs")

    def is_done(self, key: str) -> bool:
        escaped = key.replace("'", "''")
        frame = self.query(f"SELECT status FROM jobs WHERE job_key = '{escaped}'")
        return bool(len(frame)) and frame["status"][0] == "done"

    def pending(self, stage: str, keys: Iterable[str]) -> list[str]:
        """``keys`` minus the ones already done at ``stage``, in input order.

        A set difference rather than a per-key lookup because ``plan`` asks
        about every unit in a window at once, and that is one scan instead of
        thousands.
        """
        wanted = list(dict.fromkeys(keys))
        if not wanted:
            return []
        escaped = stage.replace("'", "''")
        done = self.query(
            f"SELECT job_key FROM jobs WHERE status = 'done' AND stage = '{escaped}'"
        )
        finished = set(done["job_key"].to_list()) if len(done) else set()
        return [k for k in wanted if k not in finished]

    def partial(self, stage: str) -> set[str]:
        """Job keys at ``stage`` fetched while still changing and not since finished.

        `fetch` records a unit whose content was still filling upstream as
        `pending` rather than `done`, so the next run fetches it again - over
        the old copy - and only a fetch made after it stopped changing marks it
        done. Without that, an hour harvested at 10:37 was done for good, and
        10:37-11:00 was never requested again.
        """
        escaped = stage.replace("'", "''")
        rows = self.query(
            f"SELECT job_key FROM jobs WHERE stage = '{escaped}' GROUP BY job_key "
            f"HAVING bool_or(status = 'pending') AND NOT bool_or(status = 'done')"
        )
        return set(rows["job_key"].to_list()) if len(rows) else set()

    # ----------------------------------------------------------------- sugar

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.flush()

    def __repr__(self) -> str:
        return f"Catalog(root={str(self.root)!r}, writer_id={self.writer_id!r})"


def catalog_path(root: str | os.PathLike[str]) -> Path:
    """The bookkeeping directory, honouring a pre-rename one if that is what exists."""
    base = Path(root)
    current = base / CATALOG_DIRNAME
    legacy = base / LEGACY_CATALOG_DIRNAME
    if not current.is_dir() and legacy.is_dir():
        return legacy
    return current


__all__ = [
    "CATALOG_DIRNAME",
    "OUTCOMES",
    "SCHEMAS",
    "Catalog",
    "Status",
    "catalog_path",
    "default_writer_id",
    "job_key",
    "split_job_key",
]
