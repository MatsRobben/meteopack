"""Where one radar tar's time goes, through the real parse and write path.

    python benchmarks/radar_profile.py --root SAMPLE_ROOT --product echotop_5min

For each tar in the sample:

    io     evict it from the page cache, then read the whole file: cold I/O
    parse  `KnmiRadar.parse` on the now-cached file: the CPU cost alone
    write  `gridded.write` of the parsed frames into a fresh store

Splitting I/O from CPU is the point. A pack that is I/O-bound needs fewer or
larger reads; one that is CPU-bound needs less work per frame or more cores;
and the two have been confused here before - a "cold" benchmark that was in
fact reading page cache reported 148 MiB/s on a link that tops out at 119.

Eviction uses posix_fadvise(DONTNEED), which needs no privileges and on an
NFS client drops that file's cached pages. The first tar is used to check
that it works on this machine (cold, warm, evicted reads), and the result is
printed: if "evicted" looks like "warm", every io number below is a lie.

`--cprofile N` also runs cProfile over the parse and write of N tars spread
evenly across the sample (so every generation is in it), after one untimed
warm-up parse: the first parse in a process pays ~12 s of imports from the
venv on NFS, which otherwise buries everything else in the profile.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import environment


def evict(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def read_all(path: Path, block: int = 4 << 20) -> float:
    t = time.perf_counter()
    with open(path, "rb", buffering=0) as fh:
        while fh.read(block):
            pass
    return time.perf_counter() - t


def check_eviction(path: Path) -> dict[str, float]:
    mib = path.stat().st_size / 2**20
    evict(path)
    cold = read_all(path)
    warm = read_all(path)
    evict(path)
    again = read_all(path)
    out = {"cold_mib_s": mib / cold, "warm_mib_s": mib / warm, "evicted_mib_s": mib / again}
    out["eviction_works"] = again > 3 * warm
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, required=True, help="sample data root")
    ap.add_argument("--product", required=True)
    ap.add_argument("--limit", type=int, default=None, help="at most this many tars")
    ap.add_argument("--cprofile", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="data root to write into (default <root>/_profile)")
    args = ap.parse_args()

    from meteopack import cli
    from meteopack.core.store import gridded
    from meteopack.sources.base import RawFile

    out_root = args.out or args.root / "_profile"
    if out_root.exists():
        shutil.rmtree(out_root)
    (out_root / "raw" / "knmi_radar").mkdir(parents=True)
    (out_root / "raw" / "knmi_radar" / args.product).symlink_to(
        (args.root / "raw" / "knmi_radar" / args.product).resolve())

    ctx = cli.Context(cli.build_parser().parse_args(
        ["pack", "knmi_radar", "--product", args.product, "--data-root", str(out_root)]))
    tars = ctx.raw_files()[: args.limit]
    timeline = cli._timeline_for(ctx)
    t = time.perf_counter()
    gridded.initialise(ctx.cfg, ctx.root, timeline)
    t_init = time.perf_counter() - t

    src, cfg = ctx.src, ctx.cfg
    ev = check_eviction(tars[0].resolve())
    print(f"eviction check on {tars[0].name}: cold {ev['cold_mib_s']:.0f}, warm "
          f"{ev['warm_mib_s']:.0f}, evicted {ev['evicted_mib_s']:.0f} MiB/s -> "
          f"{'works' if ev['eviction_works'] else 'DOES NOT WORK: io numbers are cached reads'}")

    # Warm-up: pay the imports and first-call costs outside every measurement.
    warm = tars[0]
    src.parse(RawFile(path=warm, sha256="", n_bytes=warm.resolve().stat().st_size,
                      unit_key=warm.name), cfg)

    prof = cProfile.Profile() if args.cprofile else None
    stride = max(len(tars) // max(args.cprofile, 1), 1)
    profiled = set(range(0, len(tars), stride)[: args.cprofile]) if args.cprofile else set()
    rows = []
    for k, tar in enumerate(tars):
        real = tar.resolve()
        evict(real)
        t_io = read_all(real)
        raw = RawFile(path=tar, sha256="", n_bytes=real.stat().st_size, unit_key=tar.name)
        profiling = prof is not None and k in profiled
        if profiling:
            prof.enable()
        t = time.perf_counter()
        ds = src.parse(raw, cfg)
        t_parse = time.perf_counter() - t
        t = time.perf_counter()
        result = gridded.write(ds, cfg, ctx.root) if ds is not None else None
        t_write = time.perf_counter() - t
        if profiling:
            prof.disable()
        frames = result.written if result else 0
        rows.append({"tar": tar.name, "mib": real.stat().st_size / 2**20, "frames": frames,
                     "io": t_io, "parse": t_parse, "write": t_write})
        del ds

    tot = {k: sum(r[k] for r in rows) for k in ("mib", "frames", "io", "parse", "write")}
    busy = tot["io"] + tot["parse"] + tot["write"]
    env = environment()
    print(f"\n{args.product}: {len(rows)} tars, {tot['frames']:,} frames, {tot['mib']:,.0f} MiB "
          f"on {env['host']} ({env['cpus_available']} cpus); store init {t_init:.1f}s")
    print(f"{'phase':8s} {'seconds':>8s} {'share':>6s} {'per frame':>10s}  note")
    for k in ("io", "parse", "write"):
        note = f"{tot['mib'] / tot['io']:.0f} MiB/s cold, one stream" if k == "io" else ""
        print(f"{k:8s} {tot[k]:8.1f} {100 * tot[k] / busy:5.0f}% "
              f"{1000 * tot[k] / max(tot['frames'], 1):8.2f} ms  {note}")
    print(f"{'serial':8s} {busy:8.1f}        "
          f"{1000 * busy / max(tot['frames'], 1):8.2f} ms  "
          f"= {tot['frames'] / busy:.0f} frames/s per worker")

    if prof is not None:
        buf = io.StringIO()
        stats = pstats.Stats(prof, stream=buf).strip_dirs()
        stats.sort_stats("tottime").print_stats(30)
        print("\n--- cProfile, by own time ---")
        print("\n".join(buf.getvalue().splitlines()[4:42]))

    results = Path(__file__).resolve().parent / "results"
    results.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (results / f"radar_profile_{args.product}_{stamp}.json").write_text(json.dumps(
        {"env": env, "eviction": ev, "totals": tot, "init_s": t_init, "rows": rows},
        indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
