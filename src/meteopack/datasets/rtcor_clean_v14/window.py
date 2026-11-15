"""The cleaner pipeline on one window (research proto/v12 cleaner.py, round 6; see the package docstring).

    clean_window(codes, times, aggr=1.0, **kw) -> codes        (T, 765, 700) uint16 in, same out

aggr (0 .. 2, clipped) sets the aggressiveness: 0 = stages 1-5 only (research v11), 0 .. 1 = classifier shift
on small tracked objects plus the pixel stages 6-8 and the vote, 1 .. 2 = support stage extended, outlier and
lines stages. aggr may be a scalar, an array (T,) (one value per frame) or (T, ny, nx) (per frame and region on
any coarse grid, bilinear upsampling to 765 x 700, e.g. from score.clutter_score tiles). Every decision reads
the local aggr: per object the mean over its pixels, per pixel the pixel's value. A varying aggr should be
smooth in time; `aggr_smooth` (frames, 0 = off) applies a centred running mean along time.

Stages: 1 object decision (decide: track-pooled classifier margin + 3 min(aggr, 1) g(size) logit units,
g = 1 up to 50 px, 0 from 500 px; hysteresis along tracks), 2 spikes (v10.spike_stage), 3 frozen copy,
4 hot spots (hotspots.hot_stage), 5 weak transient echo (weak.weak_stage), then the pixel stages named in
`stages` (stages.<name>_stage or register_stage), the vote (stages.consistency), the realism guard
(stages.realism_guard), the stages named in `post`, and the frozen-half copy. New-chain frames (times >=
2023-01-31 10:45 UTC) are returned unchanged; without `times` every frame is old chain. Codes >= 65534 pass
unchanged; values are never raised; no echo is added.
"""
from __future__ import annotations

import warnings

import cv2
import numpy as np

from . import hotspots as CMC
from . import stages as SG
from . import tracks as TRK
from . import v10 as V10
from . import weak as WK
from .stale import removed_mask

MISSING = 65534
H, W = 765, 700
NEW_CHAIN = np.datetime64("2023-01-31T10:45", "ns")
DEFAULT_STAGES = ("support", "dash", "texture", "outlier", "lines")
OBJ_DEFAULTS = dict(dm=3.0, dm2=0.0, g0=50, g1=500, gmin=0.0)


# ------------------------------------------------------------------ aggr field
class Aggr:
    """aggr as scalar / (T,) / (T, ny, nx). px(t) -> float or (H, W) float32; obj(t, lab, n) -> (n,) mean per object."""

    def __init__(self, aggr, T, smooth=0):
        a = np.asarray(aggr, np.float32)
        if a.ndim == 0:
            a = np.full(T, float(a), np.float32)
        if a.shape[0] != T:
            raise ValueError(f"aggr has {a.shape[0]} frames, window has {T}")
        if a.ndim not in (1, 3):
            raise ValueError("aggr must be scalar, (T,) or (T, ny, nx)")
        a = np.clip(a, 0.0, 2.0)
        if smooth and smooth > 1:
            k = int(smooth)
            pad = np.concatenate([a[:1].repeat(k // 2, 0), a, a[-1:].repeat(k // 2, 0)])
            c = np.cumsum(np.concatenate([np.zeros_like(a[:1]), pad]), 0)
            a = ((c[k:] - c[:-k]) / k)[:T].astype(np.float32)
        self.a = a
        self.field = a.ndim == 3
        self._px = {}

    def px(self, t):
        if not self.field:
            return float(self.a[t])
        if t not in self._px:
            self._px = {k: v for k, v in self._px.items() if abs(k - t) <= 3}
            self._px[t] = cv2.resize(self.a[t], (W, H), interpolation=cv2.INTER_LINEAR)
        return self._px[t]

    def max(self, t=None):
        return float(self.a.max() if t is None else self.a[t].max())

    def obj(self, t, lab, n):
        if not self.field:
            return np.full(n, float(self.a[t]))
        s = np.bincount(lab.ravel(), weights=self.px(t).ravel(), minlength=n + 1)[1:]
        c = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        return s / np.maximum(c, 1)


# ------------------------------------------------------------------ 1. object decision with aggr
def decide(st, A, *, lam=0.8, margin=0.0, hyst=1.0, hown=1.0, bigm=1.0, bigs=1000, hshare=0.5, dm=3.0, dm2=0.0, g0=50, g1=500,
           gmin=0.0):
    """proto/tracks decide() with a per-object shift dm * aggr * g(size) of the pooled and own margins; g = 1 up to
    g0 px, falling log-linearly to gmin at >= g1 px, size = track-pooled object size (geometric mean along the track,
    same pooling as the evidence, so g does not jump when an object splits or merges). Whole-object removal of large
    objects destroys whole rain systems (heatwave convection scores near the classifier threshold); large mixed
    objects are left to the pixel stages. aggr 0: the plain tracks decision.
    Returns (RM, info: pooled margins incl. shift, links)."""
    labs, ev, flr, flags = st["labs"], st["ev"], st["flr"], st["flags"]
    L = [x[0] for x in labs]; N = [x[1] for x in labs]
    T = len(L)
    flows = TRK.flow_at(flr)
    links = TRK.build_links(L, N, flows)
    w = TRK.frozen_weight(L, N, flags)
    P, _ = TRK.pool(links, ev, w, lam)
    lsz = [np.log(np.maximum(np.bincount(L[t].ravel(), minlength=N[t] + 1)[1:], 1)).reshape(-1, 1) if N[t]
           else np.zeros((0, 1)) for t in range(T)]
    PS, _ = TRK.pool(links, lsz, [np.ones(N[t]) for t in range(T)], lam)
    ma, sh = [], []
    for t in range(T):
        if N[t] == 0:
            ma.append(np.zeros(0)); sh.append(np.zeros(0)); continue
        size = np.bincount(L[t].ravel(), minlength=N[t] + 1)[1:]
        a = A.obj(t, L[t], N[t])
        g = np.clip(1 - (1 - gmin) * (PS[t][:, 0] - np.log(g0)) / np.log(g1 / g0), gmin, 1)
        s = (dm * np.minimum(a, 1) + dm2 * np.maximum(a - 1, 0)) * g      # aggr 2: 5 logit units
        ma.append(P[t].max(1) - margin - bigm * (size >= bigs) + s)
        sh.append(s)
    RM = [m > 0 for m in ma]
    if hyst > 0:
        LO = [(m > -hyst) & (ev[t].max(1) + sh[t] > -hown) if N[t] else np.zeros(0, bool) for t, m in enumerate(ma)]
        for _ in range(2):
            for t in range(1, T):
                if N[t] and N[t - 1]:
                    RM[t] |= LO[t] & ((links[t - 1][1] @ np.r_[0.0, RM[t - 1]])[1:] > hshare)
            for t in range(T - 2, -1, -1):
                if N[t] and N[t + 1]:
                    RM[t] |= LO[t] & ((links[t][0] @ np.r_[0.0, RM[t + 1]])[1:] > hshare)
    return RM, {"ma": ma, "links": links}


def frozen_copy(codes, out, flags):
    for t in range(1, len(codes)):
        for bit, sl in SG.HALVES:
            if flags[t] & bit:
                same = codes[t][sl] == codes[t - 1][sl]
                out[t][sl] = np.where(same, out[t - 1][sl], out[t][sl])
    return out


# ------------------------------------------------------------------ stage registry (hook for added stages)
STAGES: dict = {}


def register_stage(name, fn):
    """Add a pixel stage: fn(ctx, out, **kw) -> (out, level). ctx: stages.Context (codes = raw window, st with
    labs / flr flows / frozen flags, A = local aggr via ctx.apx(t, py, px), valid, ok rows, flow(t, py, px));
    out: the current output (uint16, may be lowered, never raised); level: per-pixel aggr at which the pixel
    was removed (inf where kept) or None. Selected by name in clean_window(stages=...) or post=..."""
    STAGES[name] = fn


def stage_fn(name):
    """stages.<name>_stage, or a stage added with register_stage."""
    if name in STAGES:
        return STAGES[name]
    if hasattr(SG, name + "_stage"):
        return getattr(SG, name + "_stage")
    raise ValueError(f"unknown stage {name!r} (no stages.{name}_stage, none registered)")


# ------------------------------------------------------------------ main
def clean_window(codes, times=None, aggr=1.0, *, st=None, weak=True, hotspots=True,
                 aggr_smooth=0, stages=DEFAULT_STAGES, post=(), guard=True, ret_info=False, **kw):
    """Clean one window (T, 765, 700) uint16 -> same; equals the research v12 clean_window(..., period="chain")
    bit for bit when post=(). kw: track kw (tracks.DEFAULTS), spike_kw, wedge, wedge_kw, streak_kw, hot-spot
    STAGE_DEFAULTS keys, weak.DEFAULTS keys, OBJ_DEFAULTS keys, stage kw dicts <name>_kw, smooth_kw, guard_kw.
    post: stage names run after the realism guard (not reverted by it), before the frozen-half copy.
    st: a precomputed v10 stage (it is not modified)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _clean_window(codes, times, aggr, st=st, weak=weak, hotspots=hotspots, aggr_smooth=aggr_smooth,
                             stages=stages, post=post, guard=guard, ret_info=ret_info, **kw)


def _clean_window(codes, times, aggr, *, st, weak, hotspots, aggr_smooth, stages, post, guard, ret_info, **kw):
    codes = np.asarray(codes)
    T = len(codes)
    old = np.ones(T, bool) if times is None else (np.asarray(times).astype("datetime64[ns]") < NEW_CHAIN)
    if not old.any():
        return (codes.copy(), {}) if ret_info else codes.copy()
    A = Aggr(aggr, T, aggr_smooth)
    valid = codes < MISSING
    # --- 1-3: v10 with the aggr object decision
    if st is None:
        st = V10.stage(codes, times, kw.get("streak_kw"))
    else:
        st = dict(st); st["sms"] = st["sms"].copy()
    tkw = {k: v for k, v in kw.items() if k in TRK.DEFAULTS}
    okw = {k: kw.get(k, v) for k, v in OBJ_DEFAULTS.items()}
    RM, dinfo = decide(st, A, **{**TRK.DEFAULTS, **tkw}, **okw)
    rmo = removed_mask(codes, st, RM)
    v9, _ = V10.spike_stage(codes, st, rmo, wedge=kw.get("wedge", True), wedge_kw=kw.get("wedge_kw"),
                            spike_kw=kw.get("spike_kw"))
    out = V10.frozen_apply(codes, st, v9)
    # --- 4: hot spots
    if hotspots:
        M = CMC.window_map(times)
        if M is not None:
            p = {**CMC.STAGE_DEFAULTS, **{k: v for k, v in kw.items() if k in CMC.STAGE_DEFAULTS}}
            geo = CMC.cached_geo(id(M), M["hot"], p["zone"], p["ring"])
            out, _ = CMC.hot_stage(codes, out, st, geo, **p)
    # --- 5: weak Q0
    if weak:
        out = WK.weak_stage(codes, out, times, flags=st["flags"], **{k: v for k, v in kw.items() if k in WK.DEFAULTS})
    v11ish = out
    info = {"v11ish": v11ish}
    # --- 6-10: new stages
    if A.max() > 0 and stages:
        ctx = SG.Context(codes, st, A, valid)
        lvmin = np.full(codes.shape, 60000, np.float16)
        sid = np.zeros(codes.shape, np.uint8)                  # earliest stage that changed the pixel (1-based)
        for si, name in enumerate(stages):
            fn = stage_fn(name)
            before = out
            out, m = fn(ctx, out, **kw.get(name + "_kw", {}))
            sid[(sid == 0) & (out < before)] = si + 1
            if m is not None:
                m16 = np.minimum(m, 60000).astype(np.float16)            # levels (inf = kept); float16 for memory
                np.minimum(lvmin, m16, out=lvmin)
                if ret_info:
                    info[name] = m16
                del m16
            del m, before
        out = SG.consistency(ctx, v11ish, out, lvmin=lvmin, **kw.get("smooth_kw", {}))
        del lvmin
        if guard:
            out, gst = SG.realism_guard(ctx, v11ish, out, sid, names=tuple(stages), **kw.get("guard_kw", {}))
            info["guard"] = gst
        del sid
        for name in post:
            out, _ = stage_fn(name)(ctx, out, **kw.get(name + "_kw", {}))
        out = frozen_copy(codes, out, st["flags"])
        out = np.where(valid, np.minimum(out, v11ish), codes).astype(codes.dtype)
    out = np.where(old[:, None, None], out, codes).astype(codes.dtype)
    return (out, info) if ret_info else out
