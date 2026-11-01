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

    src, cfg = _load(task)
    frame = src.parse(_raw_file(task.raw_path), cfg)
    if frame is None or not len(frame):
        return {"path": task.raw_path, "rows": 0, "ipc": None, "parts": {}}

    if not task.spill:
        buf = io.BytesIO()
        frame.write_ipc(buf)
        return {"path": task.raw_path, "rows": frame.height, "ipc": buf.getvalue(), "parts": {}}

    from meteopack.core.store.points import partition_labels

    spill = Path(task.spill)
    labels = partition_labels(frame, cfg)
    tagged = frame.with_columns(labels.alias("__part"))
    parts: dict[str, str] = {}
    for value, group in tagged.partition_by("__part", as_dict=True).items():
        label = str(value[0] if isinstance(value, tuple) else value)
        directory = spill / label
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"{uuid.uuid4().hex}.arrow"
        group.drop("__part").write_ipc(target)
        parts[label] = str(target)
    return {"path": task.raw_path, "rows": frame.height, "ipc": None, "parts": parts}


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
    files: tuple[str, ...]
    source: str
    product: str | None
    root: str
    configs: str | None = None


def write_partition(task: PartitionTask) -> dict[str, Any]:
    """Merge one partition's spill files, coalesce them, and write it."""
    import polars as pl

    from meteopack.core.store import points

    _, cfg = _load_cached(task.source, task.product, task.configs)
    frame = pl.concat([pl.read_ipc(f) for f in task.files], how="diagonal_relaxed")
    frame, merged = coalesce_observations(frame)
    paths = points.write(frame, cfg, Path(task.root))
    return {"label": task.label, "rows": frame.height, "merged": merged,
            "paths": [str(p) for p in paths]}


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
    for raw_path in task.paths:
        t = time.perf_counter()
        ds = src.parse(_raw_file(raw_path), cfg)
        t_parse += time.perf_counter() - t
        if ds is None or not ds.sizes.get("time"):
            continue
        drift.update(ds.attrs.get("schema_drift") or ())
        for entry in ds.attrs.get("layouts_seen") or ():
            fp = str(entry.get("fingerprint"))
            seen = layouts.get(fp)
            if seen is None:
                layouts[fp] = dict(entry)
            else:
                seen["count"] += int(entry.get("count", 0))
        t = time.perf_counter()
        result = gridded.write(ds, cfg, Path(task.root))
        t_write += time.perf_counter() - t
        del ds
        frames += result.written
        outside += result.outside
        off_cadence += result.off_cadence
        lost += [t for t in (result.outside_first, result.outside_last) if t]
    return {"path": task.raw_path, "frames": frames, "outside": outside,
            "off_cadence": off_cadence, "drift": sorted(drift),
            "outside_span": [min(lost), max(lost)] if lost else None,
            "layouts": list(layouts.values()),
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
                    out.append({"path": task.raw_path, "error": broken})
                except Exception as exc:  # noqa: BLE001 - reported per file, never silent
                    logger.error("pack failed for %s: %s: %s",
                                 task.raw_path, type(exc).__name__, exc)
                    out.append({"path": task.raw_path, "error": f"{type(exc).__name__}: {exc}"})
            if broken:
                break
            submit_more()

        if broken:
            logger.error("%s", broken)
            for fut, task in pending.items():
                fut.cancel()
                out.append({"path": task.raw_path, "error": broken})
            for task in queued:
                out.append({"path": task.raw_path, "error": f"not attempted: {broken}"})
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
