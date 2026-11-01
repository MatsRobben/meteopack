"""Where the time goes, stage by stage, on the real radar archive.

The point is not to produce one number but to split one tar's journey into the
pieces that can be optimised independently:

    read      pull the member bytes out of the tar (NFS + tar seek)
    open      h5py parse of the member's metadata
    decode    read the image arrays, which is where zlib runs
    encode    re-encode onto the canonical calibration
    write     put the frames into the zarr store

Run it with a tar count and a worker count:

    python benchmarks/bench_stages.py --tars 6 --workers 1
"""
from __future__ import annotations

import argparse
import io
import os
import tarfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ARCHIVE = Path(os.environ.get(
    "KNMI_RADAR_ARCHIVE",
    "/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi/nl_rdr_data_rtcor_5m_tar",
))


def time_one_tar(path: Path, limit: int | None = None) -> dict[str, float]:
    """Split one tar's processing into named phases."""
    import h5py

    from meteopack.core.encoding import Encoding, encode
    from meteopack.sources.knmi_radar import _image_specs

    target = Encoding(np.dtype("uint16"), 0.01, 0.0, 65534, 65535)
    spent: dict[str, float] = defaultdict(float)
    frames = 0

    t0 = time.perf_counter()
    with tarfile.open(path, "r") as tar:
        members = [m for m in tar if m.isfile()]
        spent["list_members"] += time.perf_counter() - t0
        for member in members[:limit]:
            t = time.perf_counter()
            blob = tar.extractfile(member).read()
            spent["read"] += time.perf_counter() - t

            t = time.perf_counter()
            with h5py.File(io.BytesIO(blob), "r") as frame:
                specs = _image_specs(frame)
                spent["open"] += time.perf_counter() - t
                if not specs:
                    continue
                for spec in specs.values():
                    t = time.perf_counter()
                    counts = np.asarray(frame[f"{spec.group}/image_data"][:])
                    spent["decode"] += time.perf_counter() - t

                    t = time.perf_counter()
                    encode(counts, target, source_scale=spec.scale,
                           source_offset=spec.offset,
                           source_out_of_image={65535})
                    spent["encode"] += time.perf_counter() - t
            frames += 1
    spent["frames"] = frames
    spent["bytes"] = path.stat().st_size
    return dict(spent)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tars", type=int, default=4)
    ap.add_argument("--members", type=int, default=None,
                    help="members per tar; default all")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0,
                    help="pick a different, disjoint set of tars; the page cache "
                         "makes a second run over the same files meaningless")
    args = ap.parse_args()

    tars = sorted(ARCHIVE.glob("*.tar"))
    if not tars:
        raise SystemExit(f"no tars under {ARCHIVE}")
    # A different, disjoint set per seed. Re-running over the same tars
    # measures the page cache: the first pass here went from 29 MiB/s cold to
    # 483 MiB/s warm, which would have made any comparison nonsense. Striding
    # by seed also keeps every product generation represented in each sample.
    stride = max(len(tars) // max(args.tars, 1), 1)
    picks = [tars[(i * stride + args.seed) % len(tars)] for i in range(args.tars)]

    started = time.perf_counter()
    if args.workers > 1:
        import concurrent.futures as cf
        import multiprocessing

        ctx = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx) as pool:
            results = list(pool.map(time_one_tar, picks, [args.members] * len(picks)))
    else:
        results = [time_one_tar(p, args.members) for p in picks]
    wall = time.perf_counter() - started

    total = defaultdict(float)
    for r in results:
        for k, v in r.items():
            total[k] += v
    frames = int(total.pop("frames", 0))
    nbytes = total.pop("bytes", 0)

    print(f"{len(picks)} tars, {frames} frames, {nbytes / 2**30:.1f} GiB, "
          f"{args.workers} worker(s)")
    print(f"wall {wall:.1f} s  ->  {frames / wall:.1f} frames/s, "
          f"{nbytes / 2**20 / wall:.0f} MiB/s")
    cpu = sum(total.values())
    print(f"\n{'phase':14s} {'seconds':>9s} {'% of cpu':>9s}  {'per frame':>10s}")
    for name, seconds in sorted(total.items(), key=lambda kv: -kv[1]):
        print(f"{name:14s} {seconds:9.1f} {100 * seconds / cpu:8.1f}% "
              f"{1000 * seconds / max(frames, 1):9.1f} ms")
    print(f"{'TOTAL cpu':14s} {cpu:9.1f}")
    if args.workers > 1:
        print(f"parallel efficiency: {cpu / wall / args.workers:.2f} "
              f"(1.0 = perfect)")


if __name__ == "__main__":
    main()
