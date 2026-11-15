"""Equivalence of the meteopack RTCOR cleaner v14 with the research cleaner, bit for bit, on real windows.

Run each side in its own process (the research code changes global OpenCV state on import):

    python scripts/rtcor_clean_v14_equivalence.py --side research    # research outputs -> --cache
    python scripts/rtcor_clean_v14_equivalence.py --side port        # port outputs, compared with them

Research side: proto/v12/cleaner.py clean_window(codes, times, aggr=1.0, period="chain") from the clutter study
workspace (--work); with --research-module another cleaner file with the same call (e.g. the final v14).
Port side: meteopack.datasets.rtcor_clean_v14.clean_window(codes, times, 1.0), and clean(codes, times,
context=PAD) (one block) on the same window. Windows (PAD = 7 context frames on each side, 24 core frames):
  heat   2022-07-19 22:15 (heatwave night; research cache raw_heat_n2215.npz)
  flood  2021-07-14 11:00 (research cache raw_flood_2021-07-14.npz)
  spikes 2021-07-08 06:30 (research cache raw_spikes_2021-07-08.npz)
  new    2025-07-15 12:00 (new chain, read from the local RTCOR store; must pass unchanged)
The v13 clutter-amount score (research proto/v13/score.py clutter_score vs rtcor_clean_v14.score) is compared
on the same windows. Exit code 1 if any window differs.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import resource
import sys
import time
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np

WORK = Path.home() / 'weather_research/research/rtcor_clutter_v8'
CACHE = Path.home() / '.cache/meteopack/rtcor_clean_v14_equiv'
WINDOWS = {'heat': 'raw_heat_n2215.npz', 'flood': 'raw_flood_2021-07-14.npz',
           'spikes': 'raw_spikes_2021-07-08.npz', 'new': '2025-07-15T12:00'}
PAD, CORE = 7, 24


def window(name: str, work: Path, cache: Path):
    src = WINDOWS[name]
    if src.endswith('.npz'):
        z = np.load(work / 'proto/v12/cache' / src)
        return z['codes'], z['times']
    f = cache / f'raw_{name}.npz'
    if not f.exists():
        sys.path.insert(0, str(work / 'lib'))
        from v8common import load, open_store
        t0 = np.datetime64(src)
        codes, times = load(open_store(), t0, t0 + np.timedelta64(5 * CORE, 'm'), pad=PAD)
        np.savez_compressed(f, codes=codes, times=times)
    z = np.load(f)
    return z['codes'], z['times']


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--side', choices=('research', 'port'), required=True)
    p.add_argument('--work', type=Path, default=WORK)
    p.add_argument('--cache', type=Path, default=CACHE)
    p.add_argument('--research-module', type=Path, default=None, help='default WORK/proto/v12/cleaner.py')
    p.add_argument('--windows', nargs='*', default=list(WINDOWS))
    args = p.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    if args.side == 'research':
        f = args.research_module or args.work / 'proto/v12/cleaner.py'
        sys.path.insert(0, str(f.parent))
        spec = importlib.util.spec_from_file_location('research_cleaner', f)
        C = importlib.util.module_from_spec(spec); spec.loader.exec_module(C)
        spec = importlib.util.spec_from_file_location('research_score', args.work / 'proto/v13/score.py')
        RS = importlib.util.module_from_spec(spec); spec.loader.exec_module(RS)
        for name in args.windows:
            codes, times = window(name, args.work, args.cache)
            t = time.perf_counter()
            out = C.clean_window(codes, times, aggr=1.0, period='chain')
            dt = time.perf_counter() - t
            np.save(args.cache / f'research_{name}.npy', out)
            sc = RS.clutter_score(codes, times)
            np.savez(args.cache / f'research_score_{name}.npz', **sc)
            print(f'{name}: research {dt / len(codes):.2f} s/frame, changed px {int((out != codes).sum())}, '
                  f'peak RSS {rss_mb():.0f} MB', flush=True)
        return 0
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from meteopack.datasets import rtcor_clean_v14 as V14
    bad = 0
    for name in args.windows:
        codes, times = window(name, args.work, args.cache)
        ref = np.load(args.cache / f'research_{name}.npy')
        t = time.perf_counter()
        out = V14.clean_window(codes, times, 1.0)
        dt = time.perf_counter() - t
        out2, info = V14.clean(codes, times, context=PAD)
        same = out.dtype == ref.dtype and np.array_equal(out, ref)
        core = slice(PAD, len(codes) - PAD)
        same2 = np.array_equal(out2[core], ref[core]) and np.array_equal(out2[:PAD], codes[:PAD]) and \
            np.array_equal(out2[-PAD:], codes[-PAD:])
        mask = np.array_equal(out >= 65534, codes >= 65534) and np.array_equal(out[codes >= 65534], codes[codes >= 65534])
        raised = int((out > codes).sum())
        new = times >= np.datetime64('2023-01-31T10:45')
        newok = np.array_equal(out[new], codes[new])
        rsc = np.load(args.cache / f'research_score_{name}.npz')
        psc = V14.score.clutter_score(codes, times)
        score_ok = all(np.array_equal(psc[k], rsc[k]) for k in rsc.files)
        ok = same and same2 and mask and raised == 0 and newok and score_ok
        bad += not ok
        nd = int((out != ref).sum())
        print(f'{name}: {"PASS" if ok else "FAIL"} clean_window==research {same} ({nd} px differ), '
              f'clean()==research on core {same2}, mask unchanged {mask}, raised px {raised}, '
              f'new chain unchanged {newok} ({int(new.sum())} frames), score==research {score_ok}, changed px {int((out != codes).sum())}, '
              f'{dt / len(codes):.2f} s/frame, peak RSS {rss_mb():.0f} MB', flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
