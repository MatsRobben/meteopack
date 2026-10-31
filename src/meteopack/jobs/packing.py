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
    """Parse one raw file to a DataFrame, returned as arrow-friendly bytes.

    Polars frames pickle poorly across a spawn boundary, so the worker hands
    back IPC bytes and the parent concatenates. Station data is small enough
    that this is cheap; gridded never does it.
    """
    import io

    src, cfg = _load(task)
    frame = src.parse(_raw_file(task.raw_path), cfg)
    if frame is None or not len(frame):
        return {"path": task.raw_path, "rows": 0, "ipc": None}
    buf = io.BytesIO()
    frame.write_ipc(buf)
    return {"path": task.raw_path, "rows": frame.height, "ipc": buf.getvalue()}


def write_gridded(task: PackTask) -> dict[str, Any]:
    """Parse one raw file and write it straight into the store."""
    from meteopack.core.store import gridded

    src, cfg = _load(task)
    ds = src.parse(_raw_file(task.raw_path), cfg)
    if ds is None or not ds.sizes.get("time"):
        return {"path": task.raw_path, "frames": 0}
    gridded.write(ds, cfg, Path(task.root))
    n = int(ds.sizes["time"])
    del ds
    return {"path": task.raw_path, "frames": n}


def run(tasks: list[PackTask], fn, *, workers: int | None = None,
        max_tasks_per_child: int = 32) -> list[dict[str, Any]]:
    """Map `fn` over `tasks`.

    `max_tasks_per_child` recycles workers. The serial benchmark showed RSS
    climbing 1328 MiB across repeats; whatever holds that memory would
    accumulate in a long-lived worker too, and a 2680-task backfill is exactly
    where it would matter.
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
    with cf.ProcessPoolExecutor(**kwargs) as pool:
        futures = {pool.submit(fn, t): t for t in tasks}
        for fut in cf.as_completed(futures):
            task = futures[fut]
            try:
                out.append(fut.result())
            except Exception as exc:  # noqa: BLE001 - reported per file, never silent
                logger.error("pack failed for %s: %s: %s", task.raw_path, type(exc).__name__, exc)
                out.append({"path": task.raw_path, "error": f"{type(exc).__name__}: {exc}"})
    return out
