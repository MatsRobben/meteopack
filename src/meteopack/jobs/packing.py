"""Packing workers: parse raw files and write them into a canonical store.

Runs in processes, not threads. Measured on cn104 over six real radar tars:
serial 71.9 s, 4 threads 36.5 s, 8 threads 42.0 s (slower than four — the work
is GIL-bound, so more threads only add contention), 4 processes 16.6 s, 8
processes 12.4 s. The serial run also grew 1328 MiB of RSS across repeats,
which pooling hides by discarding workers.

Gridded tasks write their own shard from inside the worker rather than sending
arrays back: a day of radar is 616 MiB per variable, which is far too much to
push through IPC, and the planner has already guaranteed that two tasks never
share a zarr shard.
"""
from __future__ import annotations

import concurrent.futures as cf
import functools
import logging
import multiprocessing
import os
import resource
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PackTask:
    """Everything a worker needs. Must be picklable, so it carries paths."""

    source: str
    product: str
    raw_path: str
    root: str
    configs: str | None = None
    #: Where a points worker spills its parsed rows, one directory per store
    #: partition. Unset means "hand the frame back inline", which is fine for a
    #: handful of files and is what broke on 8,495 of them.
    spill: str | None = None
    #: Further raw files this task must pack *itself*, because they write into
    #: a shard this task already owns. A zarr shard is one file: two processes
    #: writing different frames of it both read-modify-write the whole thing
    #: and the slower one's frames vanish - measured here as whole half-shards
    #: reading back empty. Grouping them into one task is what makes the
    #: parallel pack equal the serial one.
    also: tuple[str, ...] = ()
    #: A piece of the raw file(s) the source parses on its own (`parse_part`),
    #: for sources whose files are too big to be a unit of work: one UTC day of
    #: the WOW-NL yearly zips. None means "the whole file", as before.
    part: str | None = None
    #: The store partitions this task's rows belong in, declared up front. It
    #: is what lets `run_pipeline` write a partition the moment its last task
    #: is done instead of after the whole archive is parsed.
    partitions: tuple[str, ...] = ()
    #: Arrow IPC compression of the spill: "uncompressed", "lz4" or "zstd".
    spill_compression: str = "uncompressed"
    #: Hand the spilled rows back through the pool's pipe (compressed IPC
    #: bytes) instead of writing files. For `run_pipeline`, whose spill is only
    #: the few partitions in flight: it then costs no filesystem traffic at all.
    spill_inline: bool = False

    @property
    def paths(self) -> tuple[str, ...]:
        return (self.raw_path, *self.also)


@functools.lru_cache(maxsize=8)
def _load_cached(source: str, product: str | None, configs: str | None):
    from meteopack.jobs.plan import load_product_config, load_source_config
    from meteopack.sources import registry

    cfg = load_product_config(source, product, configs=configs)
    src = registry.build(source, load_source_config(source, configs=configs))
    return src, cfg


def _load(task: PackTask):
    """The source and config for this task, built once per worker process.

    This re-read and re-parsed the YAML on every task: 19.6 ms each, almost
    all of it in the YAML scanner, which over dwd's 8,494 files is 166 seconds
    of pure setup. Cached, it is 0.3 us. A source is reusable across tasks by
    design - it holds a session and its own caches, which is exactly what
    should persist - and the cache stays small because one worker only ever
    sees one product.
    """
    return _load_cached(task.source, task.product, task.configs)


def _raw_file(path: str):
    from meteopack.sources.base import RawFile

    p = Path(path)
    return RawFile(path=p, sha256="", n_bytes=p.stat().st_size, unit_key=p.name)


def parse_points(task: PackTask) -> dict[str, Any]:
    """Parse one raw file and spill it to disk, one file per store partition.

    Returning the frame to the parent is what broke the DWD backfill: 8,495
    files, every parsed frame held in the parent at once and then concatenated,
    74 GB peak RSS and a SIGABRT from the allocator. Routing each worker's rows
    into per-partition spill files instead means the parent only ever holds one
    partition, and the pickling of large payloads through the pool's pipe goes
    away with it.

    `task.spill` is where those files go. Without it the frame comes back
    inline, which is what the tests and small runs use.
    """
    import io

    started = time.time()
    src, cfg = _load(task)
    records: list[dict] = []
    if task.part is not None:
        # The source returns rows plus whatever it wants recorded about the
        # piece (WOW: one row per hour member for its hours table).
        frame, records = src.parse_part(list(task.paths), task.part, cfg)
    else:
        frame = src.parse(_raw_file(task.raw_path), cfg)
    t_parse = time.time() - started
    if frame is None or not len(frame):
        return {"path": task.raw_path, "part": task.part, "rows": 0, "ipc": None, "parts": {},
                "sizes": {}, "records": records,
                "timing": _timing(started, t_parse, 0.0)}

    if not task.spill and not task.spill_inline:
        buf = io.BytesIO()
        frame.write_ipc(buf)
        return {"path": task.raw_path, "part": task.part, "rows": frame.height,
                "ipc": buf.getvalue(), "parts": {}, "records": records,
                "timing": _timing(started, t_parse, 0.0)}

    t = time.time()
    spill = Path(task.spill) if task.spill else None
    parts: dict[str, str | bytes] = {}
    sizes: dict[str, int] = {}
    for label, group in _by_partition(frame, cfg, task.partitions):
        if task.spill_inline:
            buf = io.BytesIO()
            group.write_ipc(buf, compression=task.spill_compression)
            parts[label] = buf.getvalue()
        else:
            directory = spill / label
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / f"{uuid.uuid4().hex}.arrow"
            group.write_ipc(target, compression=task.spill_compression)
            parts[label] = str(target)
        # What the rows occupy in memory once read back - with a compressed
        # spill the file size says little about what writing it will cost.
        sizes[label] = int(group.estimated_size())
    return {"path": task.raw_path, "part": task.part, "rows": frame.height, "ipc": None,
            "parts": parts, "sizes": sizes, "records": records,
            "timing": _timing(started, t_parse, time.time() - t)}


def _by_partition(frame, cfg, declared: tuple[str, ...]):
    """`(label, rows)` per store partition the frame touches.

    When the task declared one partition and the frame's first and last
    times both fall in it, every row does (partitions are time ranges), and
    labelling each row is skipped: building a label string per row cost
    ~0.5 s of every WOW-NL day task, 3 % of a full pack.
    """
    from meteopack.core.store.points import (
        partition_label_of,
        partition_labels,
        time_partitioned,
    )

    if len(declared) == 1 and time_partitioned(cfg) and frame["time"].null_count() == 0:
        lo, hi = frame["time"].min(), frame["time"].max()
        if partition_label_of(cfg, lo) == partition_label_of(cfg, hi) == declared[0]:
            yield declared[0], frame
            return
    tagged = frame.with_columns(partition_labels(frame, cfg).alias("__part"))
    for value, group in tagged.partition_by("__part", as_dict=True).items():
        yield str(value[0] if isinstance(value, tuple) else value), group.drop("__part")


def _timing(started: float, parse_s: float, write_s: float) -> dict[str, float]:
    return {"pid": os.getpid(), "start": started, "end": time.time(), "parse_s": parse_s,
            "write_s": write_s,
            "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


@dataclass(frozen=True, slots=True)
class PartitionTask:
    """One store partition's spilled frames, ready to merge and write.

    Partitions are independent - a `(time, station_id)` pair belongs to
    exactly one, because the key is derived from `time` - so they can be
    written concurrently. Doing them in a loop left fifteen cores idle through
    the most expensive phase of a points pack: dwd coalesces ~60 million
    duplicate rows per partition, because its five variable groups each
    contribute a row for the same (time, station).
    """

    label: str
    #: Spill files, or inline IPC bytes (`PackTask.spill_inline`).
    files: tuple[str | bytes, ...]
    source: str
    product: str | None
    root: str
    configs: str | None = None
    #: Bytes of spilled Arrow IPC this partition reads back - what the memory
    #: planner in `partition_workers` sizes the pool by.
    n_bytes: int = 0


#: Peak RSS of writing a partition, as a multiple of its spilled IPC bytes:
#: the frames read back, their concatenation, and the group-by that coalesces
#: duplicate (time, station) rows each hold a copy. Measured on the full dwd
#: pack (2026-11-05): 17.5 GiB peaks for ~2.7 GiB partitions (6.5x) and 29.2
#: GiB for the largest, 5.2 GiB (5.6x); the job peaked at 103.6 of 110 GiB with
#: a factor of 4. Guessing low costs the whole pack to the OOM killer.
PARTITION_RSS_FACTOR = 7.0


def _read_limit(path: Path) -> int | None:
    try:
        v = path.read_text().strip()
    except OSError:
        return None
    return int(v) if v.isdigit() and int(v) < 1 << 60 else None     # "max", v1's huge sentinel


def memory_limit_bytes() -> int | None:
    """The memory this process may use: the cgroup limit, else Slurm's, else RAM.

    Slurm enforces `--mem` through the job's cgroup, and the OOM killer acts on
    that limit, not on what the node has free. Exceeding it killed a dwd pack
    that wrote seventeen partitions at once (127 GB against a 110 GB request).
    """
    limits = []
    try:  # cgroup v2: the job's group and its ancestors (Slurm sets it on job_N, not the step)
        rel = Path("/proc/self/cgroup").read_text().splitlines()[-1].split(":", 2)[2]
        group = Path("/sys/fs/cgroup" + rel)
        for d in (group, *group.parents):
            if d == Path("/sys/fs"):
                break
            v = _read_limit(d / "memory.max")
            if v:
                limits.append(v)
    except (OSError, IndexError):
        pass
    v = _read_limit(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))   # cgroup v1
    if v:
        limits.append(v)
    if limits:
        return min(limits)
    mb = os.environ.get("SLURM_MEM_PER_NODE")
    if mb and mb.isdigit():
        return int(mb) << 20
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):  # pragma: no cover - non-POSIX
        return None


def partition_workers(tasks: list[PartitionTask], workers: int, *,
                      limit: int | None = None, fraction: float = 0.7,
                      factor: float = PARTITION_RSS_FACTOR) -> int:
    """How many partitions can be written at once without exceeding memory.

    Sized so the largest `k` partitions, written together, fit in `fraction`
    of the limit. Always at least one: a single partition that does not fit
    will fail on its own, and saying so is better than refusing to try.
    """
    workers = max(1, min(workers, len(tasks)))
    if limit is None:
        limit = memory_limit_bytes()
    if not limit:
        return workers
    budget = fraction * limit
    sizes = sorted((factor * t.n_bytes for t in tasks), reverse=True)
    k, used = 0, 0.0
    for s in sizes[:workers]:
        if k and used + s > budget:
            break
        used += s
        k += 1
    return max(1, k)


def write_partition(task: PartitionTask) -> dict[str, Any]:
    """Merge one partition's spill files, coalesce them, and write it."""
    import polars as pl

    from meteopack.core.store import points

    started = time.time()
    src, cfg = _load_cached(task.source, task.product, task.configs)
    import io

    # A spilled part is a file, or - from an inline-spilling pipeline - the
    # compressed IPC bytes themselves.
    frame = pl.concat([pl.read_ipc(io.BytesIO(f) if isinstance(f, bytes) else f)
                       for f in task.files],
                      how="diagonal_relaxed", rechunk=False)
    t_read = time.time() - started
    finish = getattr(src, "finish_partition", None)
    info: dict[str, Any] = {}
    if callable(finish):
        # The source owns what a partition means: an irregular store must not
        # have its reports "coalesced" by (time, station) - two reports a
        # station sent in the same second are two reports.
        frame, info = finish(frame, cfg)
        merged = 0
        presorted = True
    else:
        frame, merged = coalesce_observations(frame)
        presorted = False
    paths = points.write(frame, cfg, Path(task.root), presorted=presorted)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss << 10
    return {"label": task.label, "rows": frame.height, "merged": merged,
            "paths": [str(p) for p in paths], "n_bytes": task.n_bytes, "peak_rss": peak,
            "info": info,
            "timing": {"pid": os.getpid(), "start": started, "end": time.time(),
                       "read_s": t_read, "parse_s": 0.0,
                       "write_s": time.time() - started - t_read,
                       "rss_mib": peak / 2**20}}


def task_name(task) -> str:
    """What to call a task in an error: its raw file, or its partition label."""
    return str(getattr(task, "raw_path", None) or getattr(task, "label", None) or task)


def coalesce_observations(df):
    """Merge rows describing the same (time, station) into one.

    Sources differ in what a unit contains. DWD ships a station-year of every
    variable in its group, so the same (time, station) arrives once per group;
    DMI ships one PARAMETER per station-window. Concatenating stacks them
    instead of merging, and the store then holds a row per parameter rather
    than per observation - 711,570 duplicate pairs on the first real DMI pack,
    and ~60 million per partition on dwd.

    Only pays for itself when duplicates exist, so the common case is a cheap
    check. Where two files genuinely disagree the first non-null wins, which
    is arbitrary but deterministic; the count is returned so a source
    producing real conflicts is visible rather than silently reconciled.
    """
    import polars as pl

    keys = [c for c in ("time", "station_id") if c in df.columns]
    if len(keys) < 2 or df.is_empty():
        return df, 0
    n_dup = int(df.select(keys).is_duplicated().sum())
    if not n_dup:
        return df, 0
    values = [c for c in df.columns if c not in keys]
    merged = (
        df.group_by(keys)
        .agg([pl.col(c).drop_nulls().first().alias(c) for c in values])
        .sort(keys)
    )
    return merged, n_dup


def write_gridded(task: PackTask) -> dict[str, Any]:
    """Parse one raw file and write it straight into the store.

    Reports what the store took, not what the file held. Those differ whenever
    a frame falls outside the store's fixed time axis, and counting the input
    turned a two-thirds data loss into a cheerful "wrote 576 frames".
    """
    import numpy as np

    from meteopack.core.store import gridded

    started = time.time()
    src, cfg = _load(task)
    frames = outside = off_cadence = 0
    lost: list[str] = []
    t_parse = t_write = 0.0
    # Schema drift is reported, not raised: a frame whose layout is not pinned
    # still reads correctly, and discarding its whole tar lost 5,508 frames on
    # the real archive. `pack` fails at the end on anything collected here.
    drift: set[str] = set()
    layouts: dict[str, dict] = {}
    # One bad file fails itself, not the files after it in the same task. A
    # task can hold several files - everything that shares a shard - and a
    # single unreadable tar used to abort the rest unreported.
    file_errors: list[dict[str, str]] = []
    records: list[dict[str, Any]] = []
    damaged: list[str] = []
    for raw_path in task.paths:
        t = time.perf_counter()
        try:
            ds = src.parse(_raw_file(raw_path), cfg)
        except Exception as exc:  # noqa: BLE001 - reported per file, never silent
            # A config that disagrees with itself fails here too - for every
            # file, before any frame is written, and every one is reported.
            file_errors.append({"path": raw_path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        finally:
            t_parse += time.perf_counter() - t
        if ds is None or not ds.sizes.get("time"):
            continue
        # Which raw file every one of these slots came from; the table row
        # that says what that file was is assembled by `pack` from `records`.
        rid = gridded.record_id(Path(raw_path).name)
        ds["source_record_id"] = (("time",), np.full(ds.sizes["time"], rid, dtype="int64"))
        drift.update(ds.attrs.get("schema_drift") or ())
        damaged.extend(ds.attrs.get("damaged") or ())
        for entry in ds.attrs.get("layouts_seen") or ():
            fp = str(entry.get("fingerprint"))
            seen = layouts.get(fp)
            if seen is None:
                layouts[fp] = dict(entry)
            else:
                seen["count"] += int(entry.get("count", 0))
        stamps = ds["time"].values
        missing = int(np.asarray(ds["is_missing_frame"].values).sum()) \
            if "is_missing_frame" in ds else 0
        t = time.perf_counter()
        try:
            result = gridded.write(ds, cfg, Path(task.root))
        except Exception as exc:  # noqa: BLE001 - reported per file, never silent
            file_errors.append({"path": raw_path, "error": f"{type(exc).__name__}: {exc}"})
            continue
        finally:
            t_write += time.perf_counter() - t
            del ds
        frames += result.written
        stat = Path(raw_path).stat()
        records.append({
            "source_record_id": rid, "file": Path(raw_path).name, "bytes": stat.st_size,
            "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(timespec="seconds"),
            "first_time": str(stamps.min()), "last_time": str(stamps.max()),
            "slots": int(stamps.size), "frames_missing": missing,
            "slots_written": int(result.written),
            "packed_at": datetime.now(UTC).isoformat(timespec="seconds"),
        })
        outside += result.outside
        off_cadence += result.off_cadence
        lost += [t for t in (result.outside_first, result.outside_last) if t]
    return {"path": task.raw_path, "frames": frames, "outside": outside,
            "off_cadence": off_cadence, "drift": sorted(drift),
            "outside_span": [min(lost), max(lost)] if lost else None,
            "layouts": list(layouts.values()),
            "file_errors": file_errors,
            "records": records,
            "damaged": damaged,
            # Wall-clock, so tasks from different workers line up: where a
            # pack's time goes between start-up, work and waiting.
            "timing": {"pid": os.getpid(), "start": started, "end": time.time(),
                       "parse_s": t_parse, "write_s": t_write,
                       # Peak so far for this worker process, in MiB (Linux: KiB).
                       "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}}


#: How long `run` waits for *any* task to finish before concluding the pool is
#: wedged - per file in the largest task in flight. Generous on purpose: the
#: slowest unit measured here is a SEVIRI scene at ~20 s and a radar tar at
#: ~13 s, so half an hour with nothing at all completing is not slowness.
#:
#: Per file, because a task can legitimately hold many files (everything that
#: shares a shard), and such a task makes progress without reporting any. A
#: flat limit killed two ~2,000-tar echotop tasks part-way through: the pool
#: was working, the store was left half-written, and the frames those tasks had
#: already written were missing from the reported total.
DEFAULT_STALL_TIMEOUT = 1800.0


def scan_files(task: PackTask) -> dict[str, Any]:
    """Scan this task's raw files and return the partial report.

    One task per batch of files rather than per file: a scan report is a
    dictionary that has to be merged, and merging a few dozen is cheaper than
    merging thousands.
    """
    src, _cfg = _load(task)
    scan = getattr(src, "scan", None)
    if scan is None:
        return {"error": f"{task.source} does not implement scan()"}
    return scan([_raw_file(p) for p in task.paths])


#: Tasks per worker before it is replaced. 32 came from the radar benchmark,
#: where RSS climbed 1,328 MiB across repeats. It is the wrong number for
#: cheap tasks: a respawn costs interpreter startup plus the source import,
#: and dwd's 8,494 small files would pay that 265 times. The caller picks.
DEFAULT_MAX_TASKS_PER_CHILD = 32


def run(tasks: list[PackTask], fn, *, workers: int | None = None,
        max_tasks_per_child: int = DEFAULT_MAX_TASKS_PER_CHILD,
        stall_timeout: float | None = DEFAULT_STALL_TIMEOUT) -> list[dict[str, Any]]:
    """Map `fn` over `tasks`, and fail rather than hang when a worker dies.

    `max_tasks_per_child` recycles workers. The serial benchmark showed RSS
    climbing 1328 MiB across repeats; whatever holds that memory would
    accumulate in a long-lived worker too, and a 2680-task backfill is exactly
    where it would matter.

    Two things here are about surviving a long run rather than about speed.

    *A dead worker must not wedge the job.* Observed: a SEVIRI pack with six
    workers lost all six - almost certainly to the OOM killer on a loaded node
    - and the parent sat in `futex_wait_queue` for 1 h 53 m with no children,
    0% CPU and no output. A cluster job that hangs silently is worse than one
    that fails, because nothing notices. `stall_timeout` bounds the wait for
    the *next* result, not for a task, and a `BrokenProcessPool` ends the run
    with every unfinished task reported rather than being retried into the same
    dead pool.

    *Only a bounded number of tasks are in flight.* Submitting 2,680 at once
    builds the whole queue up front for no gain; a window of a few per worker
    keeps the queue small and lets a stall be noticed early.
    """
    if not tasks:
        return []
    workers = workers or min(len(tasks), len(os.sched_getaffinity(0)))
    if workers <= 1:
        return [fn(t) for t in tasks]

    ctx = multiprocessing.get_context("spawn")
    kwargs: dict[str, Any] = {"max_workers": workers, "mp_context": ctx}
    if max_tasks_per_child:
        try:
            cf.ProcessPoolExecutor(max_workers=1, mp_context=ctx, max_tasks_per_child=1).shutdown()
            kwargs["max_tasks_per_child"] = max_tasks_per_child
        except TypeError:  # pragma: no cover - Python < 3.11
            pass

    out: list[dict[str, Any]] = []
    queued = iter(tasks)
    broken: str | None = None

    # Deliberately not a context manager: `__exit__` calls shutdown(wait=True),
    # which joins the worker processes - and joining a pool that is already
    # wedged is the hang this function exists to avoid. Shutdown is explicit
    # below, and forceful on the broken path.
    pool = cf.ProcessPoolExecutor(**kwargs)
    try:
        pending: dict[cf.Future, PackTask] = {}

        def submit_more() -> None:
            while len(pending) < workers * 2:
                task = next(queued, None)
                if task is None:
                    return
                pending[pool.submit(fn, task)] = task

        submit_more()
        while pending:
            limit = None
            if stall_timeout is not None:
                limit = stall_timeout * max(_n_files(t) for t in pending.values())
            done, _ = cf.wait(pending, timeout=limit, return_when=cf.FIRST_COMPLETED)
            if not done:
                broken = (
                    f"no task finished in {limit:.0f}s with {len(pending)} in "
                    f"flight; the worker pool is not making progress"
                )
                break
            for fut in done:
                task = pending.pop(fut)
                try:
                    out.append(fut.result())
                except cf.process.BrokenProcessPool as exc:
                    broken = (
                        f"a worker died ({exc}); on a shared node this is usually the "
                        f"OOM killer - lower --workers or raise the memory request"
                    )
                    # This future is already out of `pending`, so it has to be
                    # reported here or it vanishes from the accounting. When a
                    # pool breaks, every in-flight future breaks at once.
                    out.append({"path": task_name(task), "error": broken})
                except Exception as exc:  # noqa: BLE001 - reported per file, never silent
                    logger.error("pack failed for %s: %s: %s",
                                 task_name(task), type(exc).__name__, exc)
                    out.append({"path": task_name(task), "error": f"{type(exc).__name__}: {exc}"})
            if broken:
                break
            submit_more()

        if broken:
            logger.error("%s", broken)
            for fut, task in pending.items():
                fut.cancel()
                out.append({"path": task_name(task), "error": broken})
            for task in queued:
                out.append({"path": task_name(task), "error": f"not attempted: {broken}"})
    finally:
        _shutdown(pool, force=broken is not None)
    return out


def _n_files(task) -> int:
    """How many raw files a task covers; 1 for tasks that are not per-file."""
    return len(getattr(task, "paths", ())) or 1


def _shutdown(pool, *, force: bool) -> None:
    """Close the pool; on the broken path, do not wait for it.

    A wedged worker will not answer a join, so the forceful path kills the
    processes outright first. Reaching into `_processes` is reaching into the
    standard library's internals, justified only because the alternative is the
    hang this exists to prevent - and guarded, so a future layout change
    degrades to the ordinary shutdown rather than raising.
    """
    if not force:
        pool.shutdown(wait=True)
        return
    try:
        for proc in list(getattr(pool, "_processes", {}).values()):
            if proc.is_alive():
                proc.kill()
    except Exception as exc:  # noqa: BLE001 - best effort, never fatal
        logger.debug("could not kill worker processes: %s", exc)
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception as exc:  # noqa: BLE001
        logger.debug("pool shutdown after a break: %s", exc)


def memory_peak_bytes() -> int | None:
    """The job's peak memory so far, from the cgroup (tmpfs pages included).

    `ru_maxrss` sees one process; what the OOM killer acts on is the job's
    cgroup, which also counts a spill on /dev/shm. cgroup v2 records the peak
    in `memory.peak` on the group that carries the limit. None if unknown.
    """
    try:
        rel = Path("/proc/self/cgroup").read_text().splitlines()[-1].split(":", 2)[2]
        group = Path("/sys/fs/cgroup" + rel)
    except (OSError, IndexError):
        return None
    best = None
    for d in (group, *group.parents):
        if d == Path("/sys/fs"):
            break
        peak = _read_limit(d / "memory.peak")
        if peak is not None:
            best = peak
            if _read_limit(d / "memory.max"):
                return peak
    return best


#: What one parse worker is assumed to hold while it works, when sizing how
#: many partitions may be written beside the parsers. Measured on WOW-NL
#: (2026-11-06): a day of 2025 reports - 240 k reports, 323 MiB of JSON -
#: peaked at 0.45 GB in a single-threaded worker (1.2 GB for a 2019 day that
#: decodes two copies of every hour).
PARSE_WORKER_BYTES = 0.6 * 2**30


@dataclass(slots=True)
class PipelineResult:
    """What `run_pipeline` did: every parse result, every partition written, and why
    any partition was not."""

    parsed: list[dict[str, Any]]
    written: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    #: label -> reason, for partitions that were declared but not written.
    skipped: dict[str, str]
    stats: dict[str, Any]


def _set_threads(n: int) -> None:
    """Pool initializer: fix polars' thread pool before anything imports polars.

    polars sizes its pool once, at import, from POLARS_MAX_THREADS. A parse
    worker spends its time in single-threaded Python (msgspec, list building),
    and 48 of them each starting a 48-thread pool cost 4x in context switches
    (measured: 17 MiB/s per worker against 69 alone, 2.4 M involuntary
    switches for one month). A partition writer is the opposite: sort and
    parquet encoding parallelise (a 7 M-row month: 47.5 s on 1 thread, 10.4 s
    on 8, 8.9 s on 16 - cn114, 2026-11-06).
    """
    os.environ["POLARS_MAX_THREADS"] = str(n)
    os.environ.setdefault("OMP_NUM_THREADS", "1")


def _hold(seconds: float) -> None:
    time.sleep(seconds)


def _prestart(pool: cf.ProcessPoolExecutor, n: int) -> None:
    """Make `pool` start all `n` workers now rather than on demand.

    A submit spawns a worker only when none is idle, so `n` tasks that each
    hold a worker for a moment start all of them.
    """
    for fut in [pool.submit(_hold, 0.5) for _ in range(n)]:
        fut.result()


def run_pipeline(tasks: list[PackTask], *, workers: int, make_partition,
                 allowed: set[str] | None = None, limit: int | None = None,
                 fraction: float = 0.7, factor: float = PARTITION_RSS_FACTOR,
                 parse_bytes: float = PARSE_WORKER_BYTES, spill_in_memory: bool = False,
                 allow_partial: bool = False, max_tasks_per_child: int = 0,
                 write_workers: int = 3, write_threads: int = 8,
                 max_spill: float | None = None, progress_every: float = 60.0,
                 stall_timeout: float | None = DEFAULT_STALL_TIMEOUT,
                 parse_fn=None, write_fn=None) -> PipelineResult:
    """Parse and write in one pass: each partition is written as soon as its
    last parse task is done.

    The two-phase pack (`run` the parsers, then `run` the partition writers)
    holds the whole archive in the spill between the phases and leaves the
    CPUs idle while the slowest parse finishes, and then the NFS idle while
    partitions are written. For WOW-NL that is ~230 GB of rows - more than the
    job's memory if the spill is on /dev/shm, which on cn114 it must be (its
    /scratch had 18 GB free). Tasks declare their partitions (`partitions`)
    and are handed over in time order, so months complete one after another
    and the spill only ever holds the few months in flight.

    Rules that keep this from losing data:

    * A partition is written only when every task that declared it succeeded.
      If one failed, the partition is left exactly as it was (unless
      `allow_partial`), and `skipped` says why.
    * Rows for a partition no task declared (a report stamped outside its own
      hour) are accepted until that partition is written; arriving after it
      was, or outside `allowed` (a windowed pack), they are an error - never
      a silent overwrite of a partition with a fraction of its rows.
    * Writers run only while their estimated peak (`factor` x the rows'
      in-memory size) fits in `fraction` of the memory limit beside the
      parsers (`parse_bytes` each) and - when the spill is RAM - the spill.

    Parsers and writers are separate pools, because they want opposite
    things: `workers` single-threaded parsers, and at most `write_workers`
    writers with `write_threads` polars threads each.

    `make_partition(label, files, n_bytes)` builds the PartitionTask.
    `parse_fn`/`write_fn` default to `parse_points`/`write_partition`.
    """
    parse_fn = parse_fn or parse_points
    write_fn = write_fn or write_partition
    import collections

    started = time.time()
    if limit is None:
        limit = memory_limit_bytes()
    budget = fraction * limit if limit else float("inf")

    declared = collections.Counter(p for t in tasks for p in t.partitions)
    remaining = dict(declared)
    files: dict[str, list[str]] = collections.defaultdict(list)
    sizes: dict[str, int] = collections.defaultdict(int)
    spill_of: dict[str, int] = collections.defaultdict(int)
    failed: dict[str, str] = {}
    submitted: set[str] = set()
    ready: collections.deque[str] = collections.deque()
    parsed: list[dict] = []
    written: list[dict] = []
    errors: list[dict] = []
    skipped: dict[str, str] = {}
    queued = iter(tasks)
    order = {id(t): i for i, t in enumerate(tasks)}
    pending: dict[cf.Future, tuple[str, Any, float]] = {}
    live = {"spill": 0, "writing": 0.0, "parsing": 0, "max_spill": 0, "max_writing": 0.0,
            "max_writers": 0}
    broken: str | None = None

    ctx = multiprocessing.get_context("spawn")
    kwargs: dict[str, Any] = {"max_workers": workers, "mp_context": ctx,
                              "initializer": _set_threads, "initargs": (1,)}
    if max_tasks_per_child:
        kwargs["max_tasks_per_child"] = max_tasks_per_child
    pool = cf.ProcessPoolExecutor(**kwargs)
    # Writers are never recycled: each holds one partition at a time and
    # returns its memory when the frame is dropped.
    wpool = cf.ProcessPoolExecutor(max_workers=max(1, write_workers), mp_context=ctx,
                                   initializer=_set_threads, initargs=(write_threads,))
    # Start every worker now, while the pools' semaphores exist. On this
    # cluster everything a user owns in /dev/shm - including the named
    # semaphores multiprocessing creates there - is deleted ~10 s after any
    # job of theirs on the node ends; a worker spawned after that dies
    # rebuilding its semaphore (SemLock._rebuild: FileNotFoundError) and the
    # pool waits for it. Seen on cn117, 2026-11-06, when another job ended
    # during a re-pack. Workers already running are unaffected.
    _prestart(pool, workers)
    _prestart(wpool, max(1, write_workers))

    def drop_spill(label: str) -> None:
        for _, f in files.pop(label, []):
            if isinstance(f, bytes):
                continue
            try:
                os.unlink(f)
            except OSError:
                pass
        live["spill"] -= spill_of.pop(label, 0)
        sizes.pop(label, None)

    def complete(label: str) -> None:
        if label in failed and not allow_partial:
            skipped[label] = f"not rewritten: {failed[label]}"
            drop_spill(label)
        elif files.get(label):
            ready.append(label)
        else:
            skipped[label] = "no rows"

    def submit() -> None:
        n_writers = sum(1 for k, _, _ in pending.values() if k == "write")
        while ready:
            label = ready[0]
            need = factor * sizes[label]
            used = (live["writing"] + live["parsing"] * parse_bytes
                    + (live["spill"] if spill_in_memory else 0))
            if n_writers >= max(1, write_workers) or (n_writers and used + need > budget):
                break
            ready.popleft()
            # In task order, not completion order: the partition's rows then
            # arrive in the same order on every run, and a stable sort makes
            # the file the same however the workers happened to finish.
            pieces = tuple(f for _, f in sorted(files[label], key=lambda kv: kv[0]))
            task = make_partition(label, pieces, sizes[label])
            pending[wpool.submit(write_fn, task)] = ("write", (label, task), need)
            submitted.add(label)
            live["writing"] += need
            n_writers += 1
        live["max_writers"] = max(live["max_writers"], n_writers)
        live["max_writing"] = max(live["max_writing"], live["writing"])
        # Keep the parse queue short: a stall must be noticed early, and a
        # failed pool must not have thousands of tasks to cancel. And hold it
        # back while the spill is over `max_spill`: parsing outran writing on
        # the full WOW-NL archive (2026-11-06) - 22 GB of spill, writers
        # starved of CPU, and 7 of 17 minutes spent writing after the last
        # parse. Pausing the parsers hands their CPUs to the writers.
        while live["parsing"] < workers + 2 and (
                max_spill is None or live["spill"] < max_spill or not pending):
            task = next(queued, None)
            if task is None:
                break
            pending[pool.submit(parse_fn, task)] = ("parse", task, 0.0)
            live["parsing"] += 1

    def after_parse(task: PackTask, result: dict) -> None:
        live["parsing"] -= 1
        parsed.append(result)
        if result.get("error"):
            errors.append({"path": task_name(task), "part": task.part, "error": result["error"]})
            for label in task.partitions:
                failed.setdefault(label, f"task {task.part or task.raw_path} failed")
        for label, path in (result.get("parts") or {}).items():
            if isinstance(path, bytes):
                size = len(path)
            else:
                try:
                    size = os.path.getsize(path)
                except OSError:
                    size = 0
            if label not in task.partitions and (
                    (allowed is not None and label not in allowed) or label in submitted):
                why = ("outside the window being packed" if allowed is not None
                       and label not in allowed else "after that partition was written")
                errors.append({"path": task_name(task), "part": task.part,
                               "error": f"rows for partition {label} arrived {why}; "
                                        f"they were NOT written"})
                failed.setdefault(label, f"stray rows from {task.part}")
                if not isinstance(path, bytes):
                    os.unlink(path)
                continue
            files[label].append((order[id(task)], path))
            sizes[label] += int((result.get("sizes") or {}).get(label, size))
            spill_of[label] += size
            live["spill"] += size
        live["max_spill"] = max(live["max_spill"], live["spill"])
        for label in task.partitions:
            remaining[label] -= 1
            if remaining[label] == 0:
                complete(label)

    last_note = time.time()
    n_parsed = n_written = 0

    def progress(force: bool = False) -> None:
        nonlocal last_note
        if not force and time.time() - last_note < progress_every:
            return
        last_note = time.time()
        logger.info("pack: %d/%d parts parsed, %d/%d partitions written, %.1f GiB spilled, "
                    "%d writer(s) busy, %.0f s", n_parsed, len(tasks), n_written,
                    len(declared), live["spill"] / 2**30,
                    sum(1 for k, _, _ in pending.values() if k == "write"),
                    time.time() - started)

    try:
        submit()
        while pending:
            wait = None
            if stall_timeout is not None:
                wait = stall_timeout * max(_n_files(t) if k == "parse" else 1
                                           for k, t, _ in pending.values())
            done, _ = cf.wait(pending, timeout=wait, return_when=cf.FIRST_COMPLETED)
            if not done:
                broken = f"no task finished in {wait:.0f}s; the worker pool is not making progress"
                break
            for fut in done:
                kind, task, need = pending.pop(fut)
                try:
                    result = fut.result()
                except cf.process.BrokenProcessPool as exc:
                    broken = (f"a worker died ({exc}); on a shared node this is usually the "
                              f"OOM killer - lower --workers or raise the memory request")
                    result = {"error": broken}
                except Exception as exc:  # noqa: BLE001 - reported per task, never silent
                    result = {"error": f"{type(exc).__name__}: {exc}"}
                if kind == "parse":
                    after_parse(task, result)
                    n_parsed += 1
                    continue
                n_written += 1
                label, _ptask = task
                live["writing"] -= need
                if result.get("error"):
                    errors.append({"path": label, "error": result["error"]})
                    failed.setdefault(label, "partition write failed")
                    skipped[label] = f"write failed: {result['error']}"
                else:
                    written.append(result)
                drop_spill(label)
            if broken:
                break
            progress()
            if next_is_empty(pending, queued):
                # Every parse is done: partitions only stray rows reached are
                # complete now too.
                for label in list(files):
                    if label not in submitted and label not in ready and label not in skipped:
                        complete(label)
            submit()
        if broken:
            for kind, task, _ in pending.values():
                errors.append({"path": task_name(task) if kind == "parse" else task[0],
                               "error": broken})
            for task in queued:
                errors.append({"path": task_name(task), "error": f"not attempted: {broken}"})
            for label in list(files):
                skipped.setdefault(label, f"not written: {broken}")
                drop_spill(label)
    finally:
        _shutdown(pool, force=broken is not None)
        _shutdown(wpool, force=broken is not None)

    parse_t = [r["timing"] for r in parsed if r.get("timing")]
    write_t = [r["timing"] for r in written if r.get("timing")]
    wall = time.time() - started
    stats = {
        "wall_s": wall, "workers": workers, "write_workers": write_workers,
        "write_threads": write_threads, "parse_tasks": len(tasks),
        "partitions_written": len(written),
        "parse_busy_s": sum(t["end"] - t["start"] for t in parse_t),
        "parse_cpu_s": sum(t["parse_s"] for t in parse_t),
        "spill_write_s": sum(t["write_s"] for t in parse_t),
        "write_busy_s": sum(t["end"] - t["start"] for t in write_t),
        "parse_last_end_s": (max(t["end"] for t in parse_t) - started) if parse_t else 0.0,
        "max_parse_worker_rss_mib": max((t["rss_mib"] for t in parse_t), default=0.0),
        "max_write_worker_rss_mib": max((t["rss_mib"] for t in write_t), default=0.0),
        "max_spill_bytes": live["max_spill"], "max_writers": live["max_writers"],
        "max_writing_estimate_bytes": live["max_writing"],
        "memory_budget_bytes": budget if limit else None,
        "job_memory_peak_bytes": memory_peak_bytes(),
    }
    return PipelineResult(parsed=parsed, written=written, errors=errors, skipped=skipped,
                          stats=stats)


def next_is_empty(pending: dict, queued) -> bool:
    """True when no parse task is in flight and none is left to submit.

    `queued` is an iterator over a list, whose `__length_hint__` is exactly
    the number of items left - so this looks without consuming one.
    """
    if any(k == "parse" for k, _, _ in pending.values()):
        return False
    return getattr(queued, "__length_hint__", lambda: 0)() == 0
