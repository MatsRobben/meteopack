"""Timing harness that records the things which break at scale.

Wall time is the easy part. The numbers that actually decided past designs here
were memory high-water marks, temp-directory growth and I/O saturation — none
of which show up in a single warm microbenchmark on an idle node. So every
result carries its environment, its peak RSS, and the spread across repeats,
and the runner refuses to report a single sample as if it were a measurement.

Read benchmarks/README.md before trusting any number in results/.
"""
from __future__ import annotations

import json
import os
import platform
import resource
import shutil
import socket
import statistics
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent / "results"


def peak_rss_mib() -> float:
    """High-water mark for this process, in MiB. Linux reports KiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def current_rss_mib() -> float:
    try:
        with open("/proc/self/statm") as fh:
            pages = int(fh.read().split()[1])
        return pages * os.sysconf("SC_PAGE_SIZE") / 2**20
    except OSError:
        return float("nan")


def environment() -> dict[str, Any]:
    """Everything needed to know whether two results are comparable."""
    cpus = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else range(os.cpu_count() or 1)
    env = {
        "host": socket.gethostname(),
        "cpus_available": len(cpus),
        "cpu_model": "unknown",
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "python": platform.python_version(),
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "tmpdir": os.environ.get("TMPDIR", "/tmp"),
    }
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                env["cpu_model"] = line.split(":", 1)[1].strip()
                break
        env["has_avx2"] = "avx2" in Path("/proc/cpuinfo").read_text()
    except OSError:
        pass
    for name in ("zarr", "numpy", "h5py", "polars"):
        try:
            env[name] = __import__(name).__version__
        except Exception as exc:  # noqa: BLE001 - a missing version is not a failure
            env[name] = f"unavailable ({type(exc).__name__})"
    return env


def disk_free_gib(path: str | Path) -> float:
    try:
        return shutil.disk_usage(path).free / 2**30
    except OSError:
        return float("nan")


@dataclass
class Result:
    name: str
    variant: str
    seconds: list[float] = field(default_factory=list)
    units: float = 0.0
    unit_name: str = "items"
    peak_rss_mib: float = 0.0
    rss_growth_mib: float = 0.0
    bytes_written: int = 0
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def best(self) -> float:
        return min(self.seconds) if self.seconds else float("nan")

    @property
    def median(self) -> float:
        return statistics.median(self.seconds) if self.seconds else float("nan")

    @property
    def spread(self) -> float:
        """max/min. Anything far from 1.0 means the node was not quiet."""
        return (max(self.seconds) / min(self.seconds)) if self.seconds and min(self.seconds) else float("nan")

    @property
    def rate(self) -> float:
        return self.units / self.median if self.median else float("nan")

    def row(self) -> str:
        leak = " LEAK?" if self.rss_growth_mib > 64 else ""
        noisy = " NOISY" if self.spread > 1.5 else ""
        return (
            f"{self.variant:34s} {self.median:8.3f}s  {self.rate:9.2f} {self.unit_name}/s"
            f"  peak {self.peak_rss_mib:7.0f} MiB  growth {self.rss_growth_mib:+6.0f} MiB"
            f"  x{self.spread:.2f}{leak}{noisy}"
        )


@contextmanager
def timed(result: Result):
    rss_before = current_rss_mib()
    t0 = time.perf_counter()
    yield
    result.seconds.append(time.perf_counter() - t0)
    result.peak_rss_mib = max(result.peak_rss_mib, peak_rss_mib())
    result.rss_growth_mib = current_rss_mib() - rss_before


def run(name: str, variant: str, fn: Callable[[], Any], *, repeats: int = 3,
        units: float = 1.0, unit_name: str = "items", notes: dict | None = None) -> Result:
    """Run `fn` several times. One sample is not a measurement.

    RSS growth is measured across the whole set of repeats, because a leak that
    only shows on the third pass is exactly the kind that survives a benchmark
    and kills a 2680-tar backfill.
    """
    res = Result(name=name, variant=variant, units=units, unit_name=unit_name, notes=notes or {})
    rss_start = current_rss_mib()
    for _ in range(repeats):
        with timed(res):
            out = fn()
        if isinstance(out, dict):
            res.notes.update(out)
    res.rss_growth_mib = current_rss_mib() - rss_start
    return res


def report(title: str, results: Iterable[Result], *, extra: dict | None = None) -> Path:
    results = list(results)
    env = environment()
    print(f"\n=== {title} ===")
    print(f"host={env['host']} cpu={env['cpu_model'][:38]} cpus={env['cpus_available']} "
          f"tmpdir={env['tmpdir']}")
    for r in results:
        print("  " + r.row())
    RESULTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS / f"{title.replace(' ', '_').lower()}_{stamp}.json"
    path.write_text(json.dumps({
        "title": title, "environment": env, "extra": extra or {},
        "results": [asdict(r) | {"median": r.median, "rate": r.rate, "spread": r.spread}
                    for r in results],
    }, indent=2, default=str))
    print(f"  -> {path}")
    return path
