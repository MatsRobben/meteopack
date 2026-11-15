"""Fixed inputs of the cleaner, loaded from one npz file (assets/rtcor_clean_v14.npz).

The file is built by scripts/rtcor_clean_v14_assets.py from the research workspace. Keys are grouped by a
prefix (`<group>.<name>`):

    fixed.*     geography on the RTCOR grid: sea, static clutter 2021-22, distance to the NL radars / any
                radar, territory, azimuth, radar sites, static-map ratio features (object classifier)
    azmap.*     recurring interference azimuths per NL radar (thin-ray candidate rate per polar cell)
    quality.*   cover_eff: effective radar coverage (pixel-quality analysis)
    hot.*       fixed-clutter hot-spot maps: old (old chain) and new (new chain), bool
    model.*     object classifier: feature names, smoothing alpha, members
    hgb<i>.*    member i of the object classifier: a binary HistGradientBoosting model exported as flat
                tree tables (see HGB); cols = feature columns, cut = member threshold

No pickle and no scikit-learn are needed at run time.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "rtcor_clean_v14.npz"
SHAPE = (765, 700)


@dataclass(frozen=True, eq=False)
class Assets:
    arrays: dict
    path: str = ""

    @property
    def shape(self) -> tuple[int, int]:
        return SHAPE

    def group(self, prefix: str) -> dict:
        p = prefix + "."
        return {k[len(p):]: v for k, v in self.arrays.items() if k.startswith(p)}


@lru_cache(maxsize=2)
def _load(path: str) -> Assets:
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    a = Assets(arrays, path)
    if a.arrays["fixed.sea"].shape != SHAPE:
        raise ValueError(f"cleaner assets {path} are not on the {SHAPE} RTCOR grid")
    return a


def load_assets(path: str | Path = ASSETS) -> Assets:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} missing: build it with scripts/rtcor_clean_v14_assets.py")
    return _load(str(p.resolve()))


def sha256(path: str | Path = ASSETS) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ------------------------------------------------------------------ active asset set
_ACTIVE: list = [None]
_CACHES: list = []          # callables that clear a module cache derived from the assets


def register_cache(clear) -> None:
    _CACHES.append(clear)


def use(a: Assets | None) -> Assets:
    """Make `a` (default: the packaged file) the asset set the stages read; clears derived caches on change."""
    a = a or load_assets()
    if a is not _ACTIVE[0]:
        for clear in _CACHES:
            clear()
        _ACTIVE[0] = a
    return a


def active() -> Assets:
    return _ACTIVE[0] if _ACTIVE[0] is not None else use(None)


def group(prefix: str) -> dict:
    return active().group(prefix)


# ------------------------------------------------------------------ tree model
class HGB:
    """Vectorised evaluation of a binary HistGradientBoosting classifier exported to flat tables: all trees at
    once, one numpy step per tree level (same decisions as scikit-learn: x <= threshold goes left, NaN follows
    missing_go_to_left). Arrays (concatenated over trees): feat, thr, mleft, left, right (absolute node index;
    a leaf points to itself), value, leaf; roots (first node of each tree), depth, base (baseline logit)."""

    def __init__(self, t: dict):
        self.leaf = t["leaf"].astype(bool)
        self.feat = t["feat"].astype(np.int64)
        self.thr = t["thr"].astype(np.float64)
        self.mleft = t["mleft"].astype(bool)
        self.left = t["left"].astype(np.int64)
        self.right = t["right"].astype(np.int64)
        self.value = t["value"].astype(np.float64)
        self.roots = t["roots"].astype(np.int64)
        self.depth = int(t["depth"])
        self.base = float(t["base"])

    def proba(self, X):
        X = np.asarray(X, np.float64)
        n = len(X)
        node = np.broadcast_to(self.roots, (n, len(self.roots))).copy()
        rows = np.arange(n)[:, None]
        for d in range(self.depth):
            x = X[rows, self.feat[node]]
            left = np.where(np.isnan(x), self.mleft[node], x <= self.thr[node])
            node = np.where(left, self.left[node], self.right[node])
            if d >= 3 and self.leaf[node].all():
                break
        raw = self.base + self.value[node].sum(1)
        return 1.0 / (1.0 + np.exp(-raw))


_MODEL: dict = {}
register_cache(_MODEL.clear)


def model() -> dict:
    """The object classifier: {"members": [(HGB, cols, cut)], "feats", "alpha", "ks", "split", "gate"}."""
    if not _MODEL:
        g = group("model")
        mem = []
        for i in range(int(g["nmembers"])):
            t = group(f"hgb{i}")
            mem.append((HGB(t), [int(c) for c in t["cols"]], float(t["cut"])))
        _MODEL.update(members=mem, feats=[str(f) for f in g["feats"]], alpha=float(g["alpha"]),
                      ks=tuple(int(k) for k in g["ks"]), split=False, gate=None)
    return _MODEL
