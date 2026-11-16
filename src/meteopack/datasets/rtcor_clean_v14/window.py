"""v14: final old-chain RTCOR clutter cleaner = frozen v12 base + external stages + automatic aggr (night 2026-11-15).

    clean_window(codes, times, aggr=None, **kw) -> codes       (T, 765, 700) uint16 in, same out

Changes against v12 (proto/v12/cleaner.py, md5 ce231133...):
  - aggr=None (default) = automatic: auto_aggr() maps the v13 clutter load of 32-km tiles to a (T, ny, nx) aggr map
    (1 + smoothstep((load - 0.20) / 0.25), rate limit 0.1 per 5 min); every stage reads it through ctx.A.
  - three stage slots as in the meteopack port: stages (default BASE_STAGES = v12 stages without texture, before
    the vote and the guard), post (default POST_STAGES = fixclut, spikes2, stc from proto/<name>/stage.py, after
    the guard) and final (on the finished output); missing external stages are skipped (info["skipped"]).
    aggr may also be a callable.
  - memory: after the base, ctx.st keeps only flr and flags; ret_info stores per-stage level maps only with
    ret_info="levels".
  - fix of atlas A042 (base, also in v11): wedge seeds that move with the rain are dropped (moving_seeds) before the
    spike stage; a far-range rain band radial to Herwijnen was zeroed / cut into a block with slits.
  - period defaults to "chain" (as every v12 evaluation ran).

v12 description (unchanged below):

aggr scale (0 .. 2, values clipped to [0, 2]):
    0      = v11 exactly (proto/v11/cleaner.py)
    0 .. 1 = stages 1, 6-8, 10 (classifier shift, temporal support, radial dashes, speckle texture, vote)
    1 .. 2 = top range (user feedback 2026-11-15, then cut back after reviews R1/R2): support stage extended
             (larger unsupported objects, lower support share), hot-pixel lowering (12) and thin lines in weak
             surroundings (13). The classifier shift does NOT grow above 1 (dm2 = 0), texture and dash do not grow
             above 1, and the polar ray stage (11) is off by default (it cut slits / deleted radial rain bands).
             Every top-range rule passed the rain-as-old gate (NOTES.md).
aggr may be
    a scalar,
    an array (T,)              one value per frame,
    an array (T, ny, nx)       per frame and region on any coarse grid (bilinear upsampling to 765 x 700).
Every v12 decision reads the local aggr: per object the mean aggr over its pixels, per pixel the pixel's value.
Spatially/temporally varying aggr is supported, but decisions are only as steady in time as aggr itself: a
caller (v13) should give aggr that is smooth in time (e.g. per track or >= 30 min running mean); `aggr_smooth`
(frames, default 0 = off) applies a centred running mean along time inside clean_window.

New-chain frames (times >= 2023-01-31 10:45 UTC) are returned unchanged. Without `times` every frame is old chain.
Codes >= 65534 pass unchanged; values are never raised; no echo is added where the raw has none.

Pipeline (stages 1-5 = v11, identical at aggr 0; 6-12 new, active only where aggr > 0 (11-12: > 1)):
 1. proto/fast per-frame object evidence -> proto/tracks track pooling; object removed when its pooled margin
    + shift > 0, shift = 3 min(aggr, 1) g(size) logit units (g = 1 up to 50 px, 0 from 500 px, track-pooled size);
    hysteresis along tracks as v11 with the same shift.
 2. full-radial spike stage (proto/spikes), 3. frozen-half copy, 4. hot-spot stage (proto/climmap), 5. weak Q0.
 6. support stage: echo without temporal support along the motion (or in place) at t+-1..3 on both sides (bursts,
    one-frame dash fields, flicker); unsupported fringe pixels of large objects. Missing data and proto/quality
    DROPOUT pixels in neighbour frames are no evidence; proto/quality SUPPORT pixels are never removed.
 7. radial-dash stage: thin lines radial to an NL radar (width 3-4 px, >= 12 px long, not moving with the flow).
 8. speckle-texture stage: fragmented echo inside objects >= 30 px; score averaged along the motion, Schmitt
    trigger along the motion.
 9. (optional) value cap (proto/bgsub logic), 11. (optional, off) polar ray stage.
10. consistency vote along the motion over t-1, t, t+1 on the v12-only decisions (adds a removal only where some
    stage nearly fired).
14. REALISM GUARD over all of 6-13 (stages.realism_guard, user rule 2026-11-15): edited regions that carve inside
    coherent rain (create a hole, or touch kept echo on >= 35 % of their boundary without being fragmented
    speckle) are reverted, and lowered pixels with strong neighbours are restored. info["guard"] (ret_info=True)
    counts regions / reverted regions / reverted px per stage. Then frozen halves copy.
External stages: any name in `stages` that is not in stages.py is loaded from proto/<name>/stage.py (hook for
the proto/spikes2 spike stage).
12. outlier stage (aggr > 1): single hot pixels > 3x their second-highest neighbour and absent before and after.
13. lines stage (aggr > 1): thin bright lines of any orientation in weak surroundings, absent before and after.
Stages run in the order of `stages` (v12 default support, dash, texture, outlier, lines; v14 default without texture);
the vote runs after all.
"""
# Ported from research proto/v14/cleaner.py (md5 1b1c444cc06b323b70502c4b6af028d9),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations

import functools
import warnings

import cv2
import numpy as np

from . import ext_stages
from . import hotspots as CMC
from . import score as V13
from . import stages as SG
from . import tracks as TRK
from . import v10 as V10
from . import weak as WK
from .stale import removed_mask

MISSING = 65534
H, W = 765, 700
NEW_CHAIN = np.datetime64("2023-01-31T10:45", "ns")
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
    objects are left to the pixel stages. aggr 0: identical to TR.decide.
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


# ------------------------------------------------------------------ 2. spike stage with the moving-seed check (v14)
SEED_MC_DEFAULTS = dict(ks=(1, 2), rad=2, share=0.6, min_use=0.5, copy_max=0.9, min_shift=3.0)


def moving_seeds(codes, seeds, flr, flags, *, ks=(1, 2), rad=2, share=0.6, min_use=0.5, copy_max=0.9, min_shift=3.0):
    """Wedge seeds that move with the rain (fix of atlas A042, v14). seeds: bool (n_sites, T, H, W).
    Returns a bool array of the same shape: the seed components to DROP.

    The wedge test's novelty check compares polar cells in place (Eulerian). At far range a cell is ~2.5 km wide,
    so a rain band that is radial to a radar and drifts across its azimuth looks "novel" and becomes a seed; the
    spike stage then zeroes the band and subtracts a ray from what is left (A042: a winter band 300 km N of
    Herwijnen, removed except for a block with slits, 2022-01-05 07:30, also in v11). Interference does not move
    with the flow; rain does. Per seed component (8-connected, per site and frame) and side (past, future): the
    best over k in ks of the share of its pixels with echo within rad px of the motion-shifted position
    p + (u - t) v(p) in frame u = t -+ k. Frozen-half rows, missing data and frames that are a copy of t
    (> copy_max of the component's pixels identical in place) are no evidence; a side counts when >= min_use of
    the pixels can be sampled, and only when the shift is real (mean |k v| >= min_shift px; without motion,
    moving rain and a fixed ray cannot be told apart and the seed stays). The component is dropped when every
    usable side (>= 1) reaches `share`. Real rays are rarely affected: at least half of a wedge ray lies in clear
    air (wedge ca_min), and that part does not reappear along the motion on both sides."""
    n_s, T = seeds.shape[:2]
    drop = np.zeros_like(seeds)
    valid = codes < MISSING
    ok = np.ones((T, H), bool)
    for u in range(T):
        for bit, sl in SG.HALVES:
            if flags[u] & bit:
                ok[u, sl] = False
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    wetd = {}

    def wd(u):
        if u not in wetd:
            wetd[u] = cv2.dilate(((codes[u] > 0) & valid[u]).astype(np.uint8), ker).astype(bool)
        return wetd[u]

    for si in range(n_s):
        for t in range(T):
            if not seeds[si, t].any():
                continue
            n, lab = cv2.connectedComponents(seeds[si, t].astype(np.uint8), connectivity=8)
            fl, s = flr[min(t, T - 1)]
            for j in range(1, n):
                py, px = np.nonzero(lab == j)
                v = fl[np.minimum(py // SG.DS, fl.shape[0] - 1), np.minimum(px // SG.DS, fl.shape[1] - 1)] * (SG.DS * s)
                vx, vy = v[:, 0], v[:, 1]
                sides = []
                for sg in (-1, 1):
                    best = None
                    for k in ks:
                        u = t + sg * k
                        if not 0 <= u < T:
                            continue
                        same = (codes[u][py, px] == codes[t][py, px])
                        if same.mean() > copy_max:
                            continue                                  # a copy of t is no evidence
                        ys = np.rint(py + sg * k * vy).astype(np.int64); xs = np.rint(px + sg * k * vx).astype(np.int64)
                        ins = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                        yi, xi = np.where(ins, ys, 0), np.where(ins, xs, 0)
                        use = ins & ok[u][yi] & valid[u][yi, xi]
                        if use.mean() < min_use or k * float(np.hypot(vx, vy).mean()) < min_shift:
                            continue
                        sc = float(wd(u)[yi[use], xi[use]].mean())
                        best = sc if best is None else max(best, sc)
                    if best is not None:
                        sides.append(best)
                if sides and min(sides) >= share:
                    drop[si, t][py, px] = True
    return drop


def spike_stage(codes, st, rmo, *, wedge=True, wedge_kw=None, spike_kw=None, seed_mc=True, seed_mc_kw=None):
    """v10 spike stage (proto/spikes/cleaner.py spike_stage) with moving wedge seeds dropped (seed_mc, v14).
    Returns (v9, info)."""
    valid = codes < MISSING
    base = np.where(rmo & valid, 0, codes).astype(codes.dtype)          # objects removed
    seeds = st.pop("sms")                                               # (n_sites, T, H, W)
    ndrop = 0
    if wedge:
        wm = V10.WG.wedge_mask(codes, **(wedge_kw or {}))
        if seed_mc and wm.any():
            dr = moving_seeds(codes, wm, st["flr"], st["flags"], **{**SEED_MC_DEFAULTS, **(seed_mc_kw or {})})
            ndrop = int(dr.sum())
            wm &= ~dr
        seeds |= wm
    est, hit = V10.SP.spike_fix(codes, seeds, ref=base, **(spike_kw or {}))
    v9 = np.where(st["sm"] & ~hit & valid, 0, base).astype(codes.dtype)
    v9 = V10.SP.apply(v9, est, hit, codes)
    return v9, {"seed_mc_dropped_px": ndrop}


def frozen_copy(codes, out, flags):
    for t in range(1, len(codes)):
        for bit, sl in SG.HALVES:
            if flags[t] & bit:
                same = codes[t][sl] == codes[t - 1][sl]
                out[t][sl] = np.where(same, out[t - 1][sl], out[t][sl])
    return out


# ------------------------------------------------------------------ stage lookup (hook for external stages)
STAGES: dict = {}


def register_stage(name, fn):
    """Add a stage: fn(ctx, out, **kw) -> (out, level) (port replacement of loading proto/<name>/stage.py)."""
    STAGES[name] = fn


def stage_fn(name, skip_missing=False):
    """stages.<name>_stage, or an external stage proto/<name>/stage.py: stage(ctx, out, **kw) -> (out, level)
    (level: per-pixel aggr at which the pixel was removed, inf where kept, or None). ctx: stages.Context (codes,
    st with flr (flows) / flags (frozen halves) only, A = the local aggr map: ctx.A.px(t) -> float or (H, W),
    ctx.apx(t, py, px), valid, ok rows, good frames, flow(t, py, px), flow_full(t), times). Every stage reads its strength from ctx.A, so
    all stages follow the same (automatic) aggr map. skip_missing: None instead of an error when the external
    stage does not exist (yet)."""
    if hasattr(SG, name + "_stage"):
        return getattr(SG, name + "_stage")
    if name not in STAGES:
        if skip_missing:
            return None
        raise ValueError(f"unknown stage {name!r} (no stages.{name}_stage, none registered)")
    return STAGES[name]


# ------------------------------------------------------------------ automatic aggr (v13 clutter score)
AUTO_DEFAULTS = dict(a_min=1.0, a_max=2.0, lo=0.20, width=0.25, rate=0.1)
# a_min 1: every old-chain frame at least v12 aggr 1 (decision 2026-11-16: clean targets; passes the rain gates).
# lo / width: v13 tile load on the atlas cases: normal days wet-tile p90 0.02-0.17 (p99 0.15-0.32), heatwave days
# p50 0.13-0.43 / p90 0.56-0.60, tiles of the anaprop / sea / speckle instances 0.40-0.66 -> those reach aggr 2.
_V13 = None


def auto_aggr(codes, times=None, *, a_min=1.0, a_max=2.0, lo=0.20, width=0.25, rate=0.1, score=None):
    """Per-tile aggr map (T, ny, nx) from the v13 clutter load of 32-km tiles (proto/v13/score.py, smoothed +-45 min):
    aggr = a_min + (a_max - a_min) * smoothstep((load_tile - lo) / width) (a_min: old-chain floor), then
    rate-limited in time to <= `rate` per 5 min as the upper envelope a'(t) = max_s a(s) - rate |t - s| (two
    passes): a ramp starts before a clutter episode and ends after it, never cutting its peak. Aggr upsamples the map bilinearly to the pixel grid. score: a precomputed
    clutter_score dict. Returns (aggr (T, ny, nx) float32, score)."""
    global _V13
    if score is None:
        if _V13 is None:
            _V13 = V13
        score = _V13.clutter_score(codes, times)
    x = np.clip((np.asarray(score["load_tile"], np.float32) - lo) / width, 0.0, 1.0)
    a = (a_min + (a_max - a_min) * x * x * (3.0 - 2.0 * x)).astype(np.float32)
    T = len(a)
    if times is None:
        dt = np.ones(T, np.float32)
    else:
        tm = np.asarray(times).astype("datetime64[s]").astype(np.int64)
        dt = np.r_[1.0, np.maximum(np.diff(tm) / 300.0, 1.0)].astype(np.float32)
    for t in range(1, T):
        np.maximum(a[t], a[t - 1] - rate * dt[t], out=a[t])
    for t in range(T - 2, -1, -1):
        np.maximum(a[t], a[t + 1] - rate * dt[t + 1], out=a[t])
    return a, score


# ------------------------------------------------------------------ main
# v12 stages before the vote and the guard. The v12 speckle-texture stage ("texture", still available) is not in the
# default (decision 2026-11-16, stc iteration 3 in post): it decides per pixel and carved atlas A052 (half a shower
# in one frame); without it A052 is fixed, rain-as-old wet px lost 0.40 -> 0.22 %, TEST rain kept 80.3 -> 80.7 %,
# for one atlas speckle instance (21 -> 20 of 47) and one 38-px TEST speckle object.
BASE_STAGES = ("support", "dash", "outlier", "lines")
POST_STAGES = ("fixclut", "spikes2", "stc")                              # after the guard (not reverted by it)
FINAL_STAGES = ()                                                        # on the finished output


def _clean_window(codes, times=None, aggr=None, *, st=None, weak=True, hotspots=True, period="chain", maps=None, hot=None,
                 aggr_smooth=0, stages=BASE_STAGES, post=POST_STAGES, final=FINAL_STAGES, skip_missing=True, guard=True,
                 auto_kw=None, ret_info=False, **kw):
    """See module doc. Three stage slots, each run in the given order (names of stages.py or external
    proto/<name>/stage.py; the same slots as the meteopack port):
      stages  before the consistency vote and the realism guard (v12 stages; only where aggr > 0);
      post    after the guard, before the frozen-half copy (not reverted by the guard; run also where aggr = 0);
      final   on the finished output (after the frozen-half copy), clipped to the raw codes.
    Missing external stages are skipped when skip_missing (info["skipped"]).
    aggr: None = automatic (auto_aggr, auto_kw = AUTO_DEFAULTS keys); a callable aggr(codes, times) -> array; or
    scalar / (T,) / (T, ny, nx) as v12. Every stage reads its strength from ctx.A (ctx.A.px(t), ctx.apx(t, py, px)).
    kw: v11 kw (track kw, spike_kw, wedge, wedge_kw, seed_mc, seed_mc_kw, streak_kw, climmap STAGE_DEFAULTS keys,
    weak.DEFAULTS keys), OBJ_DEFAULTS keys, per-stage kw dicts <stage>_kw, smooth_kw, guard_kw. st: a precomputed
    v10 stage (dev cache; it is not modified). ret_info: info with v11ish, aggr, stages (run order), skipped,
    sid (earliest stage index + 1 that lowered each pixel), guard stats; ret_info="levels" adds the per-stage
    level maps (float16, T x H x W each)."""
    codes = np.asarray(codes)
    T = len(codes)
    old = np.ones(T, bool) if times is None else (np.asarray(times).astype("datetime64[ns]") < NEW_CHAIN)
    if not old.any():
        return (codes.copy(), {}) if ret_info else codes.copy()
    score = None
    if aggr is None:
        aggr, score = auto_aggr(codes, times, **{**AUTO_DEFAULTS, **(auto_kw or {})})
    elif callable(aggr):
        aggr = aggr(codes, times)
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
    v9, sinfo = spike_stage(codes, st, rmo, wedge=kw.get("wedge", True), wedge_kw=kw.get("wedge_kw"),
                            spike_kw=kw.get("spike_kw"), seed_mc=kw.get("seed_mc", True), seed_mc_kw=kw.get("seed_mc_kw"))
    out = V10.frozen_apply(codes, st, v9)
    # --- 4: hot spots
    if hotspots:
        M = CMC.window_map(times, maps, period, hot)
        if M is not None:
            p = {**CMC.STAGE_DEFAULTS, **{k: v for k, v in kw.items() if k in CMC.STAGE_DEFAULTS}}
            geo = CMC.cached_geo(id(M), M["hot"], p["zone"], p["ring"])
            out, _ = CMC.hot_stage(codes, out, st, geo, **p)
    # --- 5: weak Q0
    if weak:
        out = WK.weak_stage(codes, out, times, flags=st["flags"], **{k: v for k, v in kw.items() if k in WK.DEFAULTS})
    v11ish = out
    info = {"v11ish": v11ish, "seed_mc_dropped_px": sinfo["seed_mc_dropped_px"]}
    st = {"flr": st["flr"], "flags": st["flags"]}       # stages need only motion + frozen flags (frees labels, masks)
    del RM, dinfo, rmo, v9
    slots = {}
    for slot, names in (("stages", stages), ("post", post), ("final", final)):
        slots[slot] = [(n, f) for n, f in ((n, stage_fn(n, skip_missing)) for n in names) if f is not None]
    info["skipped"] = [n for names in (stages, post, final) for n in names if stage_fn(n, True) is None]
    pre_run = slots["stages"] if A.max() > 0 else []
    run = pre_run + slots["post"] + slots["final"]
    info["stages"] = [n for n, _ in run]
    if not run:
        out = np.where(old[:, None, None], out, codes).astype(codes.dtype)
        return (out, {**info, "aggr": A.a}) if ret_info else out
    ctx = SG.Context(codes, st, A, valid)
    ctx.times = times
    sid = np.zeros(codes.shape, np.uint8)                      # earliest stage that lowered the pixel (1-based)

    def apply(si, name, fn, out, lvmin=None):
        before = out
        out, m = fn(ctx, out, **kw.get(name + "_kw", {}))
        sid[(sid == 0) & (out < before)] = si + 1
        if m is not None:
            m16 = np.minimum(m, 60000).astype(np.float16)                # levels (inf = kept); float16 for memory
            if lvmin is not None:
                np.minimum(lvmin, m16, out=lvmin)
            if ret_info == "levels":                                     # per-stage level maps (memory!)
                info[name] = m16
        return out

    # --- 6-13: v12 stages, vote, realism guard
    if pre_run:
        lvmin = np.full(codes.shape, 60000, np.float16)
        for si, (name, fn) in enumerate(pre_run):
            out = apply(si, name, fn, out, lvmin)
        out = SG.consistency(ctx, v11ish, out, lvmin=lvmin, **kw.get("smooth_kw", {}))
        del lvmin
        if guard:
            out, gst = SG.realism_guard(ctx, v11ish, out, sid, names=tuple(n for n, _ in pre_run), **kw.get("guard_kw", {}))
            info["guard"] = gst
    # --- post stages (after the guard), frozen-half copy
    for j, (name, fn) in enumerate(slots["post"]):
        out = apply(len(pre_run) + j, name, fn, out)
    if pre_run or slots["post"]:
        out = frozen_copy(codes, out, st["flags"])
        out = np.where(valid, np.minimum(out, v11ish), codes).astype(codes.dtype)
    # --- final stages (on the finished output)
    for j, (name, fn) in enumerate(slots["final"]):
        out = apply(len(pre_run) + len(slots["post"]) + j, name, fn, out)
    if slots["final"]:
        out = np.where(valid, np.minimum(out, codes), codes).astype(codes.dtype)
    out = np.where(old[:, None, None], out, codes).astype(codes.dtype)
    if ret_info:
        info["sid"] = sid
        info["aggr"] = A.a
        if score is not None:
            info["score_load"] = score["load"]
    return (out, info) if ret_info else out


for _n in ("fixclut", "spikes2", "stc"):
    register_stage(_n, getattr(ext_stages, _n + "_stage"))


@functools.wraps(_clean_window)
def clean_window(codes, times=None, aggr=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _clean_window(codes, times, aggr, **kw)
