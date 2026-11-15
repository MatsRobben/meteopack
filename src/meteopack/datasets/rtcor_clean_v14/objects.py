"""Per-frame object evidence (research proto/fast): one object classifier on fast per-object features.

    labs, ev, flr = per_frame(codes)

Objects are echo connected after closing 1-px gaps. Per frame: one Farneback flow (4 km grid, t -> t+2),
~50 per-object features (feats.py: intensity, texture, shape, geography incl. the 2021-22 static map, temporal
overlap in place / moved along the flow at +-5/10 min, persistence +-30 min, beam-ray evidence +-60 min), and
P(clutter) from each member of the classifier (two HistGradientBoosting models, assets.HGB).
ev[t] = logit(p_member) - logit(thr_member) per object (> 0: that member says clutter); the temporal pooling
(tracks.py) turns it into decisions.
"""
from __future__ import annotations

import numpy as np

from . import assets as _assets
from . import feats as F


def predict(clf, X):
    if not len(X):
        return np.zeros(0)
    return clf.proba(X)


def members(m):
    """[(clf, cols, thr)] of the classifier."""
    return m["members"]


def combine(S, mem):
    """union score: 0.5 + max over members of (smoothed score - member threshold); clutter iff > 0.5."""
    return 0.5 + np.max([S[i] - mem[i][2] for i in range(len(mem))], 0)


def frame_scores(codes, *, alpha=None, raw=False, smooth=True):
    """Yield (t, lab, n, p, flow) for t = 0..T-1. smooth=False: p from the unsmoothed member probabilities
    (for a temporal layer that pools evidence itself). raw=True also yields the unsmoothed member probabilities
    (n, members) and the Frame: (t, lab, n, p, flow, p_raw, frame)."""
    m = _assets.model()
    alpha = m["alpha"] if alpha is None else alpha
    ks = tuple(m.get("ks", (-2, -1, 1, 2)))
    codes = np.asarray(codes)
    T = len(codes)
    win = F.Win(codes)
    need = set(m["feats"])
    mem = members(m)
    frames, P = {}, {}
    lag = max(ks)
    for t in range(T + lag):
        if t < T:
            fr = F.frame_features(win, t, need=need, split=m.get("split", False))
            frames[t] = fr
            P[t] = np.stack([predict(c, fr.X[:, cols]) for c, cols, _ in mem], 1) if fr.n else np.zeros((0, len(mem)))
        u = t - lag
        if u < 0:
            continue
        fr = frames[u]
        if fr.n and not smooth:
            s = combine([P[u][:, i] for i in range(len(mem))], mem)
        elif fr.n:
            maps = F.nb_maps(fr, frames, ks, gate=m.get("gate"))            # the flow mapping: once, for every member
            S = [F.smooth(P[u][:, i], maps, {v: P[v][:, i] for v in P}, alpha) for i in range(len(mem))]
            s = combine(S, mem)
        else:
            s = np.zeros(0)
        flow = win.F4(u)
        if raw:
            yield u, fr.lab, fr.n, s, flow, P[u], fr
        else:
            yield u, fr.lab, fr.n, s, flow
        frames.pop(u + min(ks), None); P.pop(u + min(ks), None)
        win.evict(u - 13)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def per_frame(codes, times=None):
    """Unsmoothed evidence for the temporal layer. Returns labs [(lab, n)] per frame, ev [(n, members)] =
    logit(p_member) - logit(thr_member) (> 0 = that member says clutter), flr {t: (4 km flow / 4, 1.0)}
    (tracks.flow_at gives the motion in px per 5 min t -> t+1)."""
    m = _assets.model()
    mem = members(m)
    lt = np.array([_logit(th) for _, _, th in mem])
    labs, ev, flr = [], [], {}
    for t, lab, n, p, flow, praw, fr in frame_scores(codes, raw=True, smooth=False):
        labs.append((lab, n))
        ev.append(_logit(praw) - lt if n else np.zeros((0, len(mem))))
        flr[t] = (flow / F.DS, 1.0) if flow is not None else (np.zeros((F.NY4, F.NX4, 2), np.float32), 1.0)
    return labs, ev, flr
