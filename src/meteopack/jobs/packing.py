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
import logging
import multiprocessing
import os
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


def _load(task: PackTask):
    from meteopack.jobs.plan import load_product_config, load_source_config
    from meteopack.sources import registry

    cfg = load_product_config(task.source, task.product, configs=task.configs)
    src = registry.build(task.source, load_source_config(task.source, configs=task.configs))
    return src, cfg


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


def write_gridded(task: PackTask) -> dict[str, Any]:
    """Parse one raw file and write it straight into the store.

    Reports what the store took, not what the file held. Those differ whenever
    a frame falls outside the store's fixed time axis, and counting the input
    turned a two-thirds data loss into a cheerful "wrote 576 frames".
    """
    from meteopack.core.store import gridded

    src, cfg = _load(task)
    frames = outside = off_cadence = 0
    # Schema drift is reported, not raised: a frame whose layout is not pinned
    # still reads correctly, and discarding its whole tar lost 5,508 frames on
    # the real archive. `pack` fails at the end on anything collected here.
    drift: set[str] = set()
    for raw_path in task.paths:
        ds = src.parse(_raw_file(raw_path), cfg)
        if ds is None or not ds.sizes.get("time"):
            continue
        drift.update(ds.attrs.get("schema_drift") or ())
        result = gridded.write(ds, cfg, Path(task.root))
        del ds
        frames += result.written
        outside += result.outside
        off_cadence += result.off_cadence
    return {"path": task.raw_path, "frames": frames, "outside": outside,
            "off_cadence": off_cadence, "drift": sorted(drift)}


#: How long `run` waits for *any* task to finish before concluding the pool is
#: wedged. Generous on purpose: the slowest unit measured here is a SEVIRI
#: scene at ~20 s and a radar tar at ~13 s, so half an hour with nothing at all
#: completing is not slowness.
DEFAULT_STALL_TIMEOUT = 1800.0


def run(tasks: list[PackTask], fn, *, workers: int | None = None,
        max_tasks_per_child: int = 32,
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
            done, _ = cf.wait(pending, timeout=stall_timeout,
                              return_when=cf.FIRST_COMPLETED)
            if not done:
                broken = (
                    f"no task finished in {stall_timeout:.0f}s with {len(pending)} in "
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
