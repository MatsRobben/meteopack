"""Pack the fixed inputs of the RTCOR clutter cleaner v14 into src/meteopack/datasets/assets/rtcor_clean_v14.npz.

Inputs come from the clutter study workspace (research/rtcor_clutter_v8, --work):
  proto/fast/fixed.npz        geography on the RTCOR grid (sea, static map 2021-22, radar distances, azimuth, ...)
  proto/fast/azmap.npz        recurring interference azimuths per NL radar
  proto/fast/model.pkl        object classifier: two scikit-learn HistGradientBoostingClassifier members. Exported
                              here to flat tree tables (assets.HGB); needs scikit-learn only at build time.
  proto/climmap/counts/       monthly RTCOR counts; the hot-spot maps are built per processing chain from every
                              month on disk (research climmap window_map, period="chain", min_months=2)
  proto/v13/sea.npy, model.json  clutter-amount score (sea mask, logistic pixel model)
and from the v6 workspace (--v6): assets/cover_eff.npy (effective radar coverage).

Run: python scripts/rtcor_clean_v14_assets.py [--work DIR] [--v6 DIR] [--out PATH]
The script checks that the exported trees reproduce the pickled models' predict_proba on random inputs.
"""
from __future__ import annotations

import argparse
import importlib.util
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'src/meteopack/datasets/assets/rtcor_clean_v14.npz'
WORK = Path.home() / 'weather_research/research/rtcor_clutter_v8'
V6 = Path.home() / 'weather_research/research/rtcor_clutter_v6'
FIXED_KEYS = ('sea', 'static', 'd_nl', 'terr', 'd_any', 'az', 'sites', 'sq_ratio', 'sq_logf', 'nl_rc')
AZMAP_KEYS = ('cand', 'occ', 'nval', 'nframes', 'sites')


def export_hgb(clf) -> dict:
    """Flat tree tables of a fitted binary HistGradientBoostingClassifier (same arrays as the research FastHGB)."""
    trees = [p[0].nodes for p in clf._predictors]
    if any(len(p) != 1 for p in clf._predictors):
        raise ValueError('only binary classifiers are supported')
    off = np.cumsum([0] + [len(t) for t in trees[:-1]])
    nd = np.concatenate(trees)
    if nd['is_categorical'].any():
        raise ValueError('categorical splits are not supported')
    o = np.repeat(off, [len(t) for t in trees])
    leaf = nd['is_leaf'].astype(bool)
    return {'leaf': leaf, 'feat': nd['feature_idx'].astype(np.int64), 'thr': nd['num_threshold'].astype(np.float64),
            'mleft': nd['missing_go_to_left'].astype(bool),
            'left': np.where(leaf, np.arange(len(nd)), nd['left'].astype(np.int64) + o),
            'right': np.where(leaf, np.arange(len(nd)), nd['right'].astype(np.int64) + o),
            'value': nd['value'].astype(np.float64), 'roots': off.astype(np.int64),
            'depth': np.int64(nd['depth'].max()), 'base': np.float64(np.ravel(clf._baseline_prediction)[0])}


def hot_maps(work: Path) -> dict:
    """Old- and new-chain hot-spot maps exactly as the research window_map(period="chain") builds them."""
    d = work / 'proto/climmap'
    sys.path.insert(0, str(d))
    spec = importlib.util.spec_from_file_location('climmap_build', d / 'climmap.py')
    cm = importlib.util.module_from_spec(spec); spec.loader.exec_module(cm)
    allm = sorted(f.stem.split('_')[1] for f in (d / 'counts').glob('climcounts_????-??.npz'))
    cut = np.datetime64('2023-02', 'M')
    out = {}
    for key, new in (('old', False), ('new', True)):
        months = tuple(m for m in allm if (np.datetime64(m, 'M') >= cut) == new)
        M = cm.cached_hot(str(d / 'counts'), months, (('min_months', 2),))
        if M is None:
            raise ValueError(f'no {key}-chain monthly counts in {d / "counts"}')
        out[key] = M['hot'].astype(bool)
        out[key + '_months'] = np.array(months)
        print(f'hot map {key}: months {", ".join(months)}: {int(out[key].sum())} hot px')
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work', type=Path, default=WORK, help='research/rtcor_clutter_v8 workspace')
    p.add_argument('--v6', type=Path, default=V6, help='research/rtcor_clutter_v6 workspace')
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    arrays: dict[str, np.ndarray] = {}
    with np.load(args.work / 'proto/fast/fixed.npz') as z:
        for k in FIXED_KEYS:
            arrays[f'fixed.{k}'] = z[k]
    with np.load(args.work / 'proto/fast/azmap.npz') as z:
        for k in AZMAP_KEYS:
            arrays[f'azmap.{k}'] = z[k]
    arrays['quality.cover_eff'] = np.load(args.v6 / 'assets/cover_eff.npy').astype(bool)
    arrays['score.sea'] = np.load(args.work / 'proto/v13/sea.npy')
    arrays['score.model_json'] = np.str_((args.work / 'proto/v13/model.json').read_text())
    for k, v in hot_maps(args.work).items():
        arrays[f'hot.{k}'] = v
    with open(args.work / 'proto/fast/model.pkl', 'rb') as fh:
        m = pickle.load(fh)
    mem = m['members'] if 'members' in m else [(m['clf'], m['cols'], m['thr'])]
    if m.get('split', False) or m.get('gate') is not None:
        raise ValueError('model options split / gate are not supported by the port')
    arrays['model.feats'] = np.array([str(f) for f in m['feats']])
    arrays['model.alpha'] = np.float64(m['alpha'])
    arrays['model.ks'] = np.array(m.get('ks', (-2, -1, 1, 2)), np.int64)
    arrays['model.nmembers'] = np.int64(len(mem))
    sys.path.insert(0, str(ROOT / 'src'))
    from meteopack.datasets.rtcor_clean_v14.assets import HGB
    rng = np.random.default_rng(0)
    for i, (clf, cols, thr) in enumerate(mem):
        t = export_hgb(clf)
        t['cols'] = np.array(cols, np.int64)
        t['cut'] = np.float64(thr)
        for k, v in t.items():
            arrays[f'hgb{i}.{k}'] = v
        X = rng.normal(size=(5000, len(cols))) * 3
        X[rng.random(X.shape) < 0.05] = np.nan
        ref = clf.predict_proba(X)[:, 1]
        got = HGB(t).proba(X)
        err = float(np.abs(ref - got).max())
        print(f'member {i}: {len(clf._predictors)} trees, {len(t["leaf"])} nodes, max |p - sklearn| = {err:.2e}')
        if err > 1e-9:
            raise ValueError('exported trees do not reproduce the classifier')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print(f'{args.out}: {args.out.stat().st_size / 1e6:.2f} MB, {len(arrays)} arrays')


if __name__ == '__main__':
    main()
