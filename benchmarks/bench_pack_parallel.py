"""Serial vs threads vs processes for packing real radar tars.

`pack` currently runs as a plain loop. The previous implementation used a
ProcessPoolExecutor with a spawn context, and its notes record why: the work is
GIL-bound (h5py decode plus numpy), and an earlier version that used threads
starved itself — it asked Slurm for 4 CPUs and then ran 32 threads.

Scale caveats this benchmark cannot see, from the previous effort:
  * temp storage. Extraction once went to /tmp, filled it, and crashed the job.
    This harness puts scratch under $TMPDIR (set to /dev/shm by the wrappers).
  * memory. Each process holds a full (288, 765, 700) float32 block = 616 MiB
    per variable. Eight workers is ~5 GiB before zarr's own buffers. The rss
    columns below are per-process, so multiply by the worker count yourself.
  * I/O saturation. One node reading one tar at a time says nothing about 30
    array tasks hitting the same NFS mount.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import Result, report, run

ARCHIVE = Path("/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi/nl_rdr_data_rtcor_5m_tar")
N_TARS = int(os.environ.get("BENCH_N_TARS", "6"))


def parse_one(tar_path: str) -> int:
    """Top level so a spawn context can pickle it."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from meteopack.jobs.plan import load_product_config
    from meteopack.sources.base import RawFile
    from meteopack.sources.knmi_radar import KnmiRadar

    cfg = load_product_config("knmi_radar", "rtcor_5min")
    src = KnmiRadar()
    p = Path(tar_path)
    ds = src.parse(RawFile(path=p, sha256="", n_bytes=p.stat().st_size, unit_key=p.name), cfg)
    n = int(ds.sizes.get("time", 0))
    del ds
    return n


def main() -> int:
    if not ARCHIVE.exists():
        print(f"archive not found: {ARCHIVE}")
        return 2
    tars = sorted(str(p) for p in ARCHIVE.glob("*2024*.tar"))[:N_TARS]
    if not tars:
        print("no tars matched")
        return 2
    print(f"{len(tars)} tars, {sum(Path(t).stat().st_size for t in tars)/2**20:.0f} MiB total")

    results: list[Result] = []
    results.append(run("pack", "serial loop", lambda: {"frames": sum(parse_one(t) for t in tars)},
                       repeats=2, units=len(tars), unit_name="tars"))

    for n in (4, 8):
        def threads(n=n):
            with cf.ThreadPoolExecutor(max_workers=n) as pool:
                return {"frames": sum(pool.map(parse_one, tars))}
        results.append(run("pack", f"ThreadPoolExecutor({n})", threads, repeats=2,
                           units=len(tars), unit_name="tars"))

        def procs(n=n):
            ctx = multiprocessing.get_context("spawn")
            with cf.ProcessPoolExecutor(max_workers=n, mp_context=ctx) as pool:
                return {"frames": sum(pool.map(parse_one, tars))}
        results.append(run("pack", f"ProcessPoolExecutor({n}, spawn)", procs, repeats=2,
                           units=len(tars), unit_name="tars"))

    report("pack parallelism", results,
           extra={"n_tars": len(tars), "cpus": len(os.sched_getaffinity(0))})
    return 0


if __name__ == "__main__":
    sys.exit(main())
