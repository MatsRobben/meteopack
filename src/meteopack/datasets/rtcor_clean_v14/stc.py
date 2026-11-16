"""stc: motion-compensated spatio-temporal consistency filter for old-chain RTCOR (v14 NOISE stage).

    stc_clean(cur, raw, times, strength=1.0, ret_info=False, ccf_gate=None, **kw) -> out  (T, 765, 700) uint16

ccf_gate: optional (T, ny, nx) or (T,) gate in [0, 1] for step 7 (default: from proto/v13 clutter_score of raw).

cur = the cleaner output so far (v12), raw = the raw codes of the same window, times = frame times.
out <= cur everywhere, codes >= 65534 unchanged, new-chain frames (>= 2023-01-31 10:45 UTC) unchanged.

Idea: rain moves with the flow and evolves smoothly over 5-10 min; flashes, granular speckle and dash fields do
not have a counterpart of similar strength nearby along the motion in the neighbouring frames.

1. MOTION (motion()): Farneback flow on the 4 km grid of log rain (as proto/fast, proto/tracks) between
   consecutive CONTENT frames per half (north rows 0-379 / south): a frame that is a frozen copy of its
   predecessor in a half (lib/stale) is skipped there, so alternating copies give 10-min pairs (flow / 2). Pairs
   with midpoints within 2 frames of t are averaged with structure weights (wet share of both frames), then
   spatially smoothed by normalised convolution; without structure the motion falls back to zero. The spread of
   the pair flows gives a motion uncertainty sigma (px per 5 min) that widens the search radius.
2. NEIGHBOURS u = t+k, k in leads (+-1, +-2), plus reserve leads (+-3) only where fewer than need2_min primary
   neighbours are available. Copies: X[u] marks pixels where u is a copy of u-1 (frozen half, or per-pixel
   copies: a 7 x 7 box where >= copy_share of >= copy_min wet px have exactly the previous code). Per pixel the
   content time ct[u] follows the copy chain; neighbour u is excluded where ct[u] = ct[t] (same data as t) or
   where it duplicates the neighbour next to it toward t. It is sampled at p + (ct[u] - ct[t]) v(p).
   Not available either: poor frame (coverage < 50 % of the window's best), outside the grid, or more than
   inv_max of the search disk missing/outside with nothing wet found (missing data is no evidence, not "dry").
3. SUPPORT from neighbour u, radius r_k = r0(s) + r_lead (|k| - 1) + r_sig |k| sigma (capped). All fields are
   taken from cur (the v12 output: clutter v12 already removed in a neighbour cannot support anything; src /
   src_level = "raw" use the raw codes instead):
     presence = local max over the disk of q(cur[u]) (q = log code, 20 units per e-fold, 0 = dry);
     mean     = local max over the disk of q(lev_box x lev_box mean of cur[u]) (a granular field is low);
     level    = local max over the disk of the grey OPENING (open_box = 2: 2 x 2 square) of q(cur[u]): 1 px wide
                dashes, specks and hot pixels of the neighbour are ignored, smooth rain and 2 px wide convective
                cores are not (open_box 3 flattened real shower cores: -20 % amount above 36 mm/h on 2021-07-30;
                it is used only for strength >= s_open3 = 1.5, where 2 px wide dashes go too);
                lev_mode="mean": = mean.
   Dropouts: a neighbour tile (drop_tile px) that is (nearly) dry while rain is there on both sides in time
   (old-chain radar / far-range dropouts read 0, not missing) is not available unless it has echo.
4. COMBINE: the need-th largest of the available supports, need = 2 when >= need2_min neighbours are available
   (two must agree; a cell present at t,t+1,t+2 or t-2,t-1,t survives), need = 1 with fewer (window ends,
   dropouts, copies); s < s_need2 uses need = 1. Removal needs >= 2 available neighbours (>= 1 for s >= s_rm1);
   clamping needs >= min_nav_clamp.
5. CLAMP, do not cut: allowed = level + delta(s) (rate x m(s) = 1 + m_k / s); out = min(cur, allowed); the
   reduction (log units) is widened slightly (max with its Gaussian, sigma = smooth, where >= smooth_min) so a
   clamped spot has no seam. REMOVE where the robust support is dry: mean = 0 (rm_on_level) or presence = 0,
   subject to:
     - weak echo is never removed: objects of cur[t] with max code <= weak0 / s (drizzle fragments flicker in
       the old chain but are rain; stc targets flashes, speckle and intense noise);
     - object consistency (obj): per 8-connected object of cur[t] the supported share f; f < obj_rm and
       <= obj_big px: the whole object goes (no salt); f >= obj_rm: unsupported parts <= app_max px (x s) go
       (dashes / flashes attached to rain), larger unsupported parts are clamped to the geometric mean of the
       supported pixels instead of cut (no half showers); bigger mostly-unsupported objects: pixelwise;
     - no pin holes: a removed pixel with >= hole_nb kept wet 8-neighbours is clamped to their median;
     - flashes (flash_rm): an object of cur[t] (<= obj_big px) none of whose pixels kept its value had no support at
       its level anywhere; clamped it would stay as a weak residue (scorecard weak-isolated / blink), so it goes.
6. Frozen halves: where a half is a frozen copy and the raw code equals the previous frame, the output copies
   the previous frame's output (as v12), so a copy stays a copy.

7. COHERENT-CORE FILTER (iteration 2; ccf, only where clutter is heavy). At 1 km rain is spatially coherent (bodies
   >= 2-3 px wide, ragged rim <= 1-2 px); speckle, dash fields and sea-clutter granules are 1-2 px wide and broken,
   and they re-appear in place, so temporal support cannot remove them. Level L = s x gate, gate = smoothstep of
   the proto/v13 tile load between ccf_lo = 0.30 and ccf_hi = 0.65 (32 km tiles, smoothed in time by v13), then a
   running max over +-ccf_tmax frames (stays on 15 min around a heavy-clutter judgement: no on/off blinking).
     L < ccf_on (0.25): off (clean scenes and the new chain stay as v12);
     core = binary opening of wet with a 2 x 2 (L < ccf_l3 = 0.5) or 3 x 3 square, restricted to SMOOTH px (5 x 5
       mean of the 3 x 3 std of ln(code) < ccf_rough = 0.55: dense granular fields form solid patches, not smooth
       ones); keep = wet within ccf_rim px (2; 1 for L >= ccf_l1 = 1.5) of the core, through wet px;
     objects losing >= ccf_obj (0.5) of their px go whole, otherwise only their thin parts (outside the 2 x 2
       opening) go: dashes / spokes / granules attached to rain are cut at the junction, no interior holes;
     temporal consistency per object along the motion: removed at t but kept at t-1 and t+1 -> kept; kept at t
       but removed at t-1 and t+1 -> removed.
   HARD (hard_rm, everywhere): px the clamp lowered by more than 1 / hard_ratio (0.3) are removed unless they lie
   in a 3 x 3 (smooth) core, so clearly unsupported intense dashes do not stay as faint lines.

STRENGTH s (scalar, (T,) or (T, ny, nx) on any coarse grid, bilinear): 0 = off, 1 = default, 2 = aggressive.
  r0(s) = r_base / sqrt(s) (clipped [r_min, r_max]), delta = ln(1 + m_k / s), need-2 rule from s >= s_need2.
  The disk radius uses the frame's mean s over wet px; delta and the need rule use the per-pixel s.
"""
# Ported from research proto/stc/stc.py (md5 e7d0c34f90e5edb29699719eae3b75ee),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations

import time

import cv2
import numpy as np

from .stale import frozen_flags

MISSING = 65534
H, W = 765, 700
DS = 4
NY4, NX4 = H // DS, W // DS
NEW_CHAIN = np.datetime64("2023-01-31T10:45", "ns")
HALVES = ((1, slice(0, 380)), (2, slice(380, H)))
SC = 20.0                                                # q units per e-fold of rain rate

_LUT = np.zeros(65536, np.uint8)
_c = np.arange(1, MISSING)
_LUT[1:MISSING] = np.clip(1 + np.rint(SC * np.log(_c)), 1, 255).astype(np.uint8)
# q -> largest code with that q (inverse for clamping; codes are integers)
_INV = np.zeros(256, np.int64)
for _q in range(1, 256):
    _m = np.nonzero(_LUT[1:MISSING] <= _q)[0]
    _INV[_q] = (_m[-1] + 1) if len(_m) else 0
del _c, _m, _q

DEFAULTS = dict(
    leads=(-2, -1, 1, 2), reserve=(-3, 3), src="cur", src_level="cur", lev_box=3, lev_mode="open", open_box=2, s_open3=1.5,
    r_base=2.5, r_min=1.5, r_max=5.0, r_lead=0.75, r_sig=0.5, r_cap=7.0,
    m_k=2.0, s_need2=0.75, need2_min=3, s_rm1=1.5, min_nav_clamp=2, rm_on_level=True,
    copy_share=0.8, copy_min=10, copy_box=7,
    inv_max=0.5, smooth=1.0, smooth_min=2.0, hole_nb=5, obj=True, obj_rm=0.3, obj_big=60, app_max=12, weak0=2.5, flash_rm=True,
    dropout=True, drop_tile=16, drop_ratio=0.3, drop_min=0.05,
    hard_rm=True, hard_ratio=0.3,
    ccf=True, ccf_lo=0.30, ccf_hi=0.65, ccf_on=0.25, ccf_l3=0.5, ccf_l1=1.5, ccf_rim=2, ccf_rough=None,
    ccf_tc=True, ccf_tc_dil=2, ccf_tc_share=0.5, ccf_obj=0.5, ccf_tmax=3, ccf_core_keep=30,
    ccf_gran=True, ccf_gon=0.05, ccf_gran_g=0.5, ccf_gran_d=0.2, ccf_gran_nmin=10, ccf_gran_lo=0.8,
    ccf_pix=False, ccf_p0=0.5, ccf_p1=0.8, ccf_pbox=9, ccf_pmin=8,
    flow_sigma=3.0, flow_eps=0.05, flow_vmax=12.0,
)


# ------------------------------------------------------------------ helpers
def qlog(c):
    return _LUT[c]


def _disk(r):
    k = int(round(r))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)), k


def _quarter(a):
    return cv2.resize(np.ascontiguousarray(a[: NY4 * DS, : NX4 * DS], np.float32), (NX4, NY4), interpolation=cv2.INTER_AREA)


def _flow_prep(c):
    c32 = np.where(c < MISSING, c, 0).astype(np.float32)
    db = 10 * np.log10(np.maximum(_quarter(c32) * np.float32(0.12), 0.03))
    return np.clip((db + 15) * 6, 0, 255).astype(np.uint8)


def strength_field(strength, T):
    """-> (kind, array): scalar per frame (T,) or field (T, ny, nx)."""
    s = np.asarray(strength, np.float32)
    if s.ndim == 0:
        s = np.full(T, float(s), np.float32)
    if s.shape[0] != T:
        raise ValueError(f"strength has {s.shape[0]} frames, window has {T}")
    if s.ndim not in (1, 3):
        raise ValueError("strength must be scalar, (T,) or (T, ny, nx)")
    return np.clip(s, 0.0, 2.0)


def _s_px(S, t, ys, xs):
    if S.ndim == 1:
        return np.full(len(ys), S[t], np.float32)
    f = cv2.resize(S[t], (W, H), interpolation=cv2.INTER_LINEAR)
    return f[ys, xs]


# ------------------------------------------------------------------ 1. motion
def motion(raw, flags, good, *, flow_sigma=3.0, flow_eps=0.05, flow_vmax=12.0, **_):
    """-> V (T, NY4, NX4, 2) float32 motion (dx, dy) in px per 5 min at frame t, SIG (T, NY4, NX4) float32 spread."""
    T = len(raw)
    prep = [_flow_prep(raw[u]) for u in range(T)]
    wet4 = [_quarter(((raw[u] > 0) & (raw[u] < MISSING)).astype(np.float32)) for u in range(T)]
    rows4 = np.arange(NY4) * DS
    flows, pairs = {}, []                                    # pairs: (midpoint, flow per 5 min, weight map)
    for j in range(T - 1):
        if not good[j]:
            continue
        for bit, sl in HALVES:
            if flags[j] & bit:
                continue                                     # j is a copy in this half: not a content frame
            b = next((u for u in range(j + 1, min(j + 4, T)) if good[u] and not flags[u] & bit), None)
            if b is None:
                continue
            if (j, b) not in flows:
                f = cv2.calcOpticalFlowFarneback(prep[j], prep[b], None, 0.5, 3, 13, 3, 5, 1.1, 0) * (DS / (b - j))
                w = np.minimum(cv2.blur(wet4[j], (5, 5)), cv2.blur(wet4[b], (5, 5)))
                flows[(j, b)] = (f, w)
            f, w = flows[(j, b)]
            hm = ((rows4 >= sl.start) & (rows4 < sl.stop)).astype(np.float32)[:, None]
            pairs.append((0.5 * (j + b), f, w * hm))
    V = np.zeros((T, NY4, NX4, 2), np.float32)
    SIG = np.zeros((T, NY4, NX4), np.float32)
    for t in range(T):
        num = np.zeros((NY4, NX4, 2), np.float32); den = np.zeros((NY4, NX4), np.float32)
        sq = np.zeros((NY4, NX4), np.float32)
        for m, f, w0 in pairs:
            d = abs(m - t)
            if d > 2:
                continue
            w = (1.0 if d <= 1 else 0.5) * w0
            num += w[..., None] * f; den += w; sq += w * (f ** 2).sum(-1)
        nums = cv2.GaussianBlur(num, (0, 0), flow_sigma)
        dens = cv2.GaussianBlur(den, (0, 0), flow_sigma)
        sqs = cv2.GaussianBlur(sq, (0, 0), flow_sigma)
        v = nums / (dens + flow_eps)[..., None]           # normalised convolution, -> 0 without structure
        mean = nums / np.maximum(dens, 1e-6)[..., None]
        var = np.maximum(sqs / np.maximum(dens, 1e-6) - (mean ** 2).sum(-1), 0)
        sp = np.hypot(v[..., 0], v[..., 1])
        v *= np.minimum(1, flow_vmax / np.maximum(sp, 1e-6))[..., None]
        V[t] = v
        SIG[t] = np.where(dens > 1e-3, np.sqrt(var), 2.0)
    return V, SIG


# ------------------------------------------------------------------ 2. copy / evidence masks
def copy_masks(raw, valid, flags, *, copy_share=0.8, copy_min=10, copy_box=7, **_):
    """X (T, H, W) bool: frame u's content at p is (part of) a copy of frame u-1 (frozen half or per-pixel copy)."""
    T = len(raw)
    X = np.zeros((T, H, W), bool)
    k = (copy_box, copy_box)
    for u in range(1, T):
        wet = (raw[u] > 0) & valid[u]
        eq = wet & (raw[u] == raw[u - 1])
        nw = cv2.boxFilter(wet.view(np.uint8), cv2.CV_32F, k, normalize=False)
        ne = cv2.boxFilter(eq.view(np.uint8), cv2.CV_32F, k, normalize=False)
        X[u] = (nw >= copy_min) & (ne >= copy_share * nw)
        for bit, sl in HALVES:
            if flags[u] & bit:
                X[u, sl] = True
    return X


# ------------------------------------------------------------------ 2b. dropout tiles
def dropout_tiles(raw, valid, *, drop_tile=16, drop_ratio=0.3, drop_min=0.05, **_):
    """(T, ny, nx) bool on a drop_tile grid: frame u is (nearly) dry in a tile while rain is there on BOTH sides
    (max over u-2..u-1 and over u+1..u+2): the old chain dropped a radar / the far range there (it reads 0, not
    missing). Such a neighbour is no evidence of "no echo". A one-frame burst at t is only on one side of t+-1."""
    T = len(raw)
    ny, nx = H // drop_tile, W // drop_tile
    wf = np.stack([cv2.resize(((raw[u] > 0) & valid[u]).astype(np.float32)[: ny * drop_tile, : nx * drop_tile],
                              (nx, ny), interpolation=cv2.INTER_AREA) for u in range(T)])
    D = np.zeros((T, ny, nx), bool)
    for u in range(T):
        a = wf[max(u - 2, 0):u].max(0) if u > 0 else None
        b = wf[u + 1:u + 3].max(0) if u < T - 1 else None
        if a is None or b is None:
            continue
        m = np.minimum(a, b)
        D[u] = (m >= drop_min) & (wf[u] < drop_ratio * m)
        D[u] = cv2.dilate(D[u].view(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return D


# ------------------------------------------------------------------ 7. coherent-core filter (heavy clutter only)
def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def ccf_level(raw, times, S, gate=None, *, ccf_lo=0.30, ccf_hi=0.65, ccf_tmax=3, ccf_pix=True, ccf_p0=0.5, ccf_p1=0.8,
              ccf_pbox=9, ccf_pmin=8, **_):
    """(T, ny, nx) float32 coherent-core level = strength x gate. gate: given (T, ny, nx) / (T,) in [0, 1], or from
    proto/v13 clutter_score: smoothstep((load_tile - ccf_lo) / (ccf_hi - ccf_lo)) (v13 NOTES proposal: 0 below
    0.30, the new-chain echo-tile p99; 1 above 0.65, the extreme-day tile p95; load is smoothed in time, sigma 15 min)."""
    if gate is None:
        from . import score as m
        sc = m.clutter_score(raw, times, keep_p=bool(ccf_pix))
        lt = sc["load_tile"]
        gate = _smoothstep((lt - ccf_lo) / (ccf_hi - ccf_lo)).astype(np.float32)
        if ccf_pix:
            # local pixel gate: v13 pixel clutter probability averaged over the wet px of a ccf_pbox box (granular
            # texture / density / sea), smoothstep between ccf_p0 and ccf_p1. Catches dash fields inside rain scenes
            # where the 32 km tile load is diluted by the rain.
            P = sc["p"]
            PG = np.zeros(P.shape, np.float16)
            k = (ccf_pbox, ccf_pbox)
            for t in range(len(P)):
                pt = P[t].astype(np.float32); w = np.isfinite(pt)
                if not w.any():
                    continue
                num = cv2.boxFilter(np.where(w, pt, 0), -1, k, normalize=False)
                den = cv2.boxFilter(w.astype(np.float32), -1, k, normalize=False)
                pb = num / np.maximum(den, 1)
                PG[t] = _smoothstep((pb - ccf_p0) / (ccf_p1 - ccf_p0)) * (den >= ccf_pmin)
            if ccf_tmax > 0:
                T = len(PG)
                PG = np.stack([PG[max(t - ccf_tmax, 0):t + ccf_tmax + 1].max(0) for t in range(T)])
            del P
    gate = np.asarray(gate, np.float32)
    if gate.ndim == 1:
        gate = gate[:, None, None]
    if ccf_tmax > 0:
        # running max over +-ccf_tmax frames: once a region is judged heavy clutter the filter stays on for 15 min
        # around it, so an object is not filtered in some frames and spared in the next (no on/off blinking)
        T = len(gate)
        gate = np.stack([gate[max(t - ccf_tmax, 0):t + ccf_tmax + 1].max(0) for t in range(T)])
    s = S if S.ndim == 3 else S[:, None, None]
    if s.ndim == 3 and gate.ndim == 3 and s.shape[1:] != gate.shape[1:] and s.shape[1:] != (1, 1):
        s = np.stack([cv2.resize(s[t], (gate.shape[2], gate.shape[1]), interpolation=cv2.INTER_LINEAR) for t in range(len(s))])
    G = (gate * s).astype(np.float32)
    if gate is not None and "PG" in locals() and ccf_pix:
        sp = s if s.shape[1:] == (1, 1) else None
        return (G, PG, s)
    return (G, None, s)


def _px(GG, t):
    G, PG, s = GG
    g = _px_tile(G, t)
    if PG is not None:
        sv = float(s[t, 0, 0]) if s.shape[1:] == (1, 1) else cv2.resize(s[t], (W, H), interpolation=cv2.INTER_LINEAR)
        g = np.maximum(g, PG[t].astype(np.float32) * sv)
    return g


def _px_tile(G, t):
    g = G[t]
    if g.shape == (1, 1):
        return np.full((H, W), float(g[0, 0]), np.float32)
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)


def _keep_mask(wet, core_k, rim, val=None, rough=None):
    """wet px within `rim` px (geodesic, through wet px) of the binary opening of wet with a core_k square."""
    k = np.ones((core_k, core_k), np.uint8)
    core = cv2.morphologyEx(wet.view(np.uint8), cv2.MORPH_OPEN, k)
    if rough is not None and val is not None:
        # core px must be smooth too: the 5 x 5 mean of the 3 x 3 std of ln(code) below `rough`
        lv = np.where(wet, np.log(np.maximum(val, 1)).astype(np.float32), 0)
        n = cv2.boxFilter(wet.view(np.uint8).astype(np.float32), -1, (3, 3), normalize=False)
        m1 = cv2.boxFilter(lv, -1, (3, 3), normalize=False) / np.maximum(n, 1)
        m2 = cv2.boxFilter(lv * lv, -1, (3, 3), normalize=False) / np.maximum(n, 1)
        r = np.where(wet, np.sqrt(np.maximum(m2 - m1 * m1, 0)), 0).astype(np.float32)
        n5 = cv2.boxFilter(wet.view(np.uint8).astype(np.float32), -1, (5, 5), normalize=False)
        sm = cv2.boxFilter(r, -1, (5, 5), normalize=False) / np.maximum(n5, 1) < rough   # granular texture, 5 x 5
        core = cv2.morphologyEx((core.astype(bool) & sm).view(np.uint8), cv2.MORPH_OPEN, k)
    keep = core.astype(bool)
    w8 = wet.view(np.uint8)
    for _ in range(rim):
        keep = cv2.dilate(keep.view(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & wet
    return keep, core.astype(bool)


def _warp(mask, v4, sgn):
    """mask of a neighbour frame moved onto t: sgn=+1 for t-1 (forward by v), -1 for t+1 (back by v)."""
    v = cv2.resize(v4, (NX4 * DS, NY4 * DS), interpolation=cv2.INTER_LINEAR)
    full = np.zeros((H, W, 2), np.float32); full[: v.shape[0], : v.shape[1]] = v
    full[v.shape[0]:] = full[v.shape[0] - 1:v.shape[0]]; full[:, v.shape[1]:] = full[:, v.shape[1] - 1:v.shape[1]]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return cv2.remap(mask.view(np.uint8), gx - sgn * full[..., 0], gy - sgn * full[..., 1], cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def coherent_core(out, valid, old, V, G, HARD, *, ccf_on=0.25, ccf_l3=0.75, ccf_l1=1.5, ccf_rim=2, ccf_rough=None,
                  ccf_tc=True, ccf_tc_dil=2, ccf_tc_share=0.5, ccf_obj=0.5, ccf_core_keep=30, ccf_gran=True, ccf_gon=0.05,
                  ccf_gran_g=0.5, ccf_gran_d=0.2, ccf_gran_nmin=10, ccf_gran_lo=0.8, **_):
    """At 1 km rain is coherent: bodies >= 2-3 px wide with a ragged rim <= 1-2 px; speckle, dash fields and sea
    clutter granules are 1-2 px wide and broken. Per pixel level L = strength x v13 gate (G, coarse grid):
      L < ccf_on: off;  ccf_on <= L < ccf_l3: core = 2 x 2 opening, rim ccf_rim;  L >= ccf_l3: 3 x 3 opening, rim
      ccf_rim;  L >= ccf_l1: 3 x 3 opening, rim 1.   keep = wet within rim px (through wet px) of the core.
    Temporal consistency (ccf_tc) per 8-connected object of the frame: removed by the filter at t but kept (overlap
    share >= ccf_tc_share with the advected, ccf_tc_dil-dilated kept px) at t-1 AND t+1 -> restored; kept (partly)
    at t but removed at t-1 AND t+1 (and not kept there) -> removed whole. HARD px (clamped by > 1/hard_ratio) are
    removed everywhere unless inside a 3 x 3 core (coherent rain)."""
    T = len(out)
    out = out.copy()
    wet = (out > 0) & (out < MISSING) & valid
    KEEP = np.ones(out.shape, bool); RMV = np.zeros(out.shape, bool)
    GHI = np.zeros(out.shape, bool); GLO = np.zeros(out.shape, bool)
    st = {"ccf_removed": 0, "ccf_restored_tc": 0, "ccf_removed_tc": 0, "hard_removed": 0, "gran_removed": 0}
    for t in range(T):
        if not old[t] or not wet[t].any():
            continue
        w = wet[t]
        k3, core3 = _keep_mask(w, 3, ccf_rim, out[t], ccf_rough)
        if HARD is not None:
            h = HARD[t] & w & ~core3
            RMV[t] |= h; KEEP[t] &= ~h
        if G is None:
            continue
        L = _px(G, t)
        if (L >= ccf_on).any() or (ccf_gran and (L > ccf_gon).any()):
            k2, _ = _keep_mask(w, 2, ccf_rim, out[t], ccf_rough)
            k31, _ = _keep_mask(w, 3, 1, out[t], ccf_rough)
            keep = np.where(L >= ccf_l1, k31, np.where(L >= ccf_l3, k3, k2))
            r = w & ~keep & (L >= ccf_on)
            gran = None
            if ccf_gran and (L > ccf_gon).any():
                # object granularity (iteration 3): over interior px (3 x 3 all wet) the deviation of ln(code) from its
                # 3 x 3 median; G = mean deviation, D = share > 0.7. Objects above BOTH thresholds (set above the
                # p99 of labelled rain and rain-as-old convection, NOTES) inside the gate (L > ccf_gon) go whole;
                # the solid-body safety does not protect them.
                q = np.zeros(w.shape, np.uint8)
                q[w] = np.clip(1 + np.rint(SC * np.log(out[t][w])), 1, 255)
                dv = np.abs(q.astype(np.float32) - cv2.medianBlur(q, 3)) / SC
                inter = cv2.erode(w.view(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
                n_, lab_ = cv2.connectedComponents(w.view(np.uint8), connectivity=8)
                ni = np.bincount(lab_[inter], minlength=n_)
                Gm = np.bincount(lab_[inter], weights=dv[inter], minlength=n_) / np.maximum(ni, 1)
                Dm = np.bincount(lab_[inter], weights=(dv[inter] > 0.7), minlength=n_) / np.maximum(ni, 1)
                Lm = np.bincount(lab_[w], weights=L[w], minlength=n_) / np.maximum(np.bincount(lab_[w], minlength=n_), 1)
                base = (ni >= ccf_gran_nmin) & (Lm > ccf_gon)
                hi = base & (Gm >= ccf_gran_g) & (Dm >= ccf_gran_d)
                lo = base & (Gm >= ccf_gran_g * ccf_gran_lo) & (Dm >= ccf_gran_d * ccf_gran_lo)
                hi[0] = lo[0] = False
                GHI[t] = hi[lab_] & w
                GLO[t] = lo[lab_] & w
            if r.any() and ccf_obj > 0:
                # object consistency: an object losing >= ccf_obj of its px goes whole; otherwise only its thin
                # parts (outside the 2 x 2 opening: dashes, spokes, granules attached to it) go, no interior holes
                n, lab = cv2.connectedComponents(w.view(np.uint8), connectivity=8)
                npx = np.maximum(np.bincount(lab[w], minlength=n), 1)
                fr = np.bincount(lab[r], minlength=n) / npx
                whole = fr >= ccf_obj; whole[0] = False
                # safety: an object with a solid body (>= ccf_core_keep px of plain 3 x 3 opening) is never removed
                # whole and never loses its body: only its thin parts can go (TEST rnd11 #3: a 1281 px convective
                # shower was removed whole when a texture test disqualified its steep-gradient cores)
                plain = cv2.morphologyEx(w.view(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
                body = np.bincount(lab[plain], minlength=n) >= ccf_core_keep
                whole &= ~body
                thin = w & ~cv2.morphologyEx(w.view(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)).astype(bool)
                r = (whole[lab] & w & (L >= ccf_on)) | (r & thin)
            RMV[t] |= r; KEEP[t] &= ~r
    if ccf_gran and G is not None and GHI.any():
        # temporal hysteresis of the granular decision along the motion: candidate at t = HI object, or LO object
        # overlapping an advected HI object at t-1 / t+1; final = candidate in >= 2 of t-1, t, t+1 (an object is
        # not removed in one frame and kept in the next because its granularity wobbles around the threshold)
        K8g = np.ones((2 * ccf_tc_dil + 1, 2 * ccf_tc_dil + 1), np.uint8)
        adv = lambda M, t, u: cv2.dilate(_warp(M[u], V[t], +1 if u < t else -1).view(np.uint8), K8g).astype(bool)

        def objshare(t, M):
            n, lab = cv2.connectedComponents(wet[t].view(np.uint8), connectivity=8)
            npx = np.maximum(np.bincount(lab[wet[t]], minlength=n), 1)
            return (np.bincount(lab[wet[t] & M], minlength=n) / npx)[lab] * wet[t]
        CAND = np.zeros(out.shape, bool)
        for t in range(T):
            if not old[t] or not (GLO[t].any() or GHI[t].any()):
                continue
            nb = np.zeros((H, W), bool)
            for u in (t - 1, t + 1):
                if 0 <= u < T and GHI[u].any():
                    nb |= adv(GHI, t, u)
            CAND[t] = GHI[t] | (GLO[t] & (objshare(t, nb) >= 0.3))
        def ov(t, u):
            if not (0 <= u < T) or not CAND[u].any():
                return np.zeros((H, W), bool)
            k = u - t
            m = cv2.dilate(_warp(CAND[u], V[t] * abs(k), -1 if k > 0 else +1).view(np.uint8), K8g).astype(bool)
            return objshare(t, m) >= 0.3
        for t in range(T):
            if not old[t] or not wet[t].any():
                continue
            o1, o_1, o2, o_2 = ov(t, t + 1), ov(t, t - 1), ov(t, t + 2), ov(t, t - 2)
            # granular at t and at a neighbour; or granular on both sides; or in the two frames before / after (the
            # first / last frame of a granular object is not spared because it measured smooth just then)
            g = ((CAND[t] & (o1 | o_1)) | (o1 & o_1) | (o_1 & o_2) | (o1 & o2)) & wet[t]
            if g.any():
                st["gran_removed"] += int((g & ~RMV[t]).sum())
                RMV[t] |= g; KEEP[t] &= ~g
    if ccf_tc and G is not None:
        K8 = np.ones((2 * ccf_tc_dil + 1, 2 * ccf_tc_dil + 1), np.uint8)
        RM2 = RMV.copy()
        for t in range(1, T - 1):
            if not old[t] or not wet[t].any():
                continue
            kp = cv2.dilate((_warp(wet[t - 1] & KEEP[t - 1], V[t], +1)).view(np.uint8), K8).astype(bool)
            kn = cv2.dilate((_warp(wet[t + 1] & KEEP[t + 1], V[t], -1)).view(np.uint8), K8).astype(bool)
            rp = cv2.dilate((_warp(RMV[t - 1], V[t], +1)).view(np.uint8), K8).astype(bool)
            rn = cv2.dilate((_warp(RMV[t + 1], V[t], -1)).view(np.uint8), K8).astype(bool)
            n, lab = cv2.connectedComponents(wet[t].view(np.uint8), connectivity=8)
            lw = lab[wet[t]]
            npx = np.maximum(np.bincount(lw, minlength=n), 1)
            f = lambda m: np.bincount(lw, weights=m[wet[t]], minlength=n) / npx
            fkp, fkn, frp, frn = f(kp), f(kn), f(rp), f(rn)
            frm = f(RMV[t])
            restore = (frm > 0) & (fkp >= ccf_tc_share) & (fkn >= ccf_tc_share)
            kill = (frm < 1) & (frp >= ccf_tc_share) & (frn >= ccf_tc_share) & (fkp < 0.2) & (fkn < 0.2)
            plain = cv2.morphologyEx(wet[t].view(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
            kill &= np.bincount(lab[plain], minlength=n) < ccf_core_keep          # same safety as above
            restore[0] = kill[0] = False
            if restore.any():
                m = restore[lab] & wet[t] & ~(HARD[t] if HARD is not None else False)
                st["ccf_restored_tc"] += int((m & RMV[t]).sum()); RM2[t] &= ~m
            if kill.any():
                m = kill[lab] & wet[t]
                st["ccf_removed_tc"] += int((m & ~RMV[t]).sum()); RM2[t] |= m
        RMV = RM2
    for t in range(T):
        if RMV[t].any():
            if HARD is not None:
                st["hard_removed"] += int((RMV[t] & HARD[t]).sum())
            st["ccf_removed"] += int((RMV[t] & wet[t]).sum())
            out[t][RMV[t] & wet[t]] = 0
    return out, st


# ------------------------------------------------------------------ main
def stc_clean(cur, raw, times=None, strength=1.0, ret_info=False, *, flags=None, V=None, SIG=None, ccf_gate=None, **kw):
    """See module doc. kw: DEFAULTS keys. flags: lib/stale frozen flags of raw (computed if None).
    V, SIG: precomputed motion (motion()). Returns out, or (out, info) with info = dict(level (T,H,W) float16
    (strength-free diagnostic: 0 removed, 1 clamped, inf kept), n_removed, n_clamped, secs)."""
    p = {**DEFAULTS, **{k: v for k, v in kw.items() if k in DEFAULTS}}
    t0 = time.time()
    cur = np.asarray(cur); raw = np.asarray(raw)
    T = len(cur)
    old = np.ones(T, bool) if times is None else (np.asarray(times).astype("datetime64[ns]") < NEW_CHAIN)
    S = strength_field(strength, T)
    out = cur.copy()
    info = {"n_removed": 0, "n_clamped": 0}
    if not old.any() or S.max() <= 0:
        info["secs"] = time.time() - t0
        return (out, info) if ret_info else out
    valid = raw < MISSING
    if flags is None:
        flags = frozen_flags(raw)
    cov = valid.reshape(T, -1).mean(1)
    good = cov >= 0.5 * max(cov.max(), 1e-9)
    if V is None:
        V, SIG = motion(raw, flags, good, **p)
    X = copy_masks(raw, valid, flags, **p)
    DROP = dropout_tiles(raw, valid, **p) if p["dropout"] else None
    dt = p["drop_tile"]
    src = np.where(valid, raw if p["src"] == "raw" else cur, 0)
    Q = qlog(src)                                            # (T, H, W) uint8 presence evidence
    lsrc = src if p["src_level"] == p["src"] else np.where(valid, raw if p["src_level"] == "raw" else cur, 0)
    INV = (~valid).view(np.uint8)
    leads = tuple(p["leads"]); reserve = tuple(p["reserve"])
    kmax = max(abs(l) for l in leads + reserve)
    level = np.full(cur.shape, np.inf, np.float16) if ret_info else None
    HARD = np.zeros(cur.shape, bool) if p["hard_rm"] else None     # clamped by more than 1 / hard_ratio
    dil = {}                                                 # (u, radius px) -> (presence, level, invalid share)

    def dilated(u, r, ob):
        key = (u, r, ob)
        if key not in dil:
            ker, kr = _disk(r)
            d = cv2.dilate(Q[u], ker)
            sm = cv2.blur(np.minimum(lsrc[u], 4000).astype(np.float32), (p["lev_box"], p["lev_box"]))
            lm = cv2.dilate(qlog(np.rint(sm).astype(np.uint16)), ker)       # local mean level (removal test)
            if p["lev_mode"] == "open":
                # robust clamp level: grey opening (lev_box square) removes 1-2 px wide dashes, specks and hot pixels
                # of the neighbour (they cannot support anything), then the local max over the disk
                ko = np.ones((ob, ob), np.uint8)
                lv = cv2.dilate(cv2.morphologyEx(qlog(lsrc[u]), cv2.MORPH_OPEN, ko), ker)
            else:
                lv = lm
            iv = cv2.boxFilter(INV[u], cv2.CV_32F, (2 * kr + 1, 2 * kr + 1))
            dil[key] = (d, lv, lm, iv)
        return dil[key]

    for t in range(T):
        dil = {k_: v_ for k_, v_ in dil.items() if abs(k_[0] - t) <= kmax}
        if not old[t]:
            continue
        c = cur[t]
        wet = (c > 0) & (c < MISSING)
        if not wet.any():
            continue
        ys, xs = np.nonzero(wet)
        s = _s_px(S, t, ys, xs)
        act = s > 0
        if not act.any():
            continue
        ys, xs, s = ys[act], xs[act], s[act]
        sbar = float(s.mean())
        r0 = float(np.clip(p["r_base"] / np.sqrt(max(sbar, 1e-3)), p["r_min"], p["r_max"]))
        ob = 3 if sbar >= p["s_open3"] else p["open_box"]     # aggressive: 2 px wide dashes cannot support either
        yi = np.minimum(ys // DS, NY4 - 1); xi = np.minimum(xs // DS, NX4 - 1)
        vx, vy = V[t][yi, xi, 0], V[t][yi, xi, 1]
        sig_med = float(np.median(SIG[t][yi, xi]))
        # content time per pixel: a copied pixel carries the content of the frame before (chain of copies)
        ct = {}
        for w_ in range(max(t - kmax - 2, 0), min(t + kmax, T - 1) + 1):
            ct[w_] = np.where(X[w_][ys, xs], ct[w_ - 1], w_) if (w_ - 1) in ct else np.full(len(ys), w_)
        cols = {}
        for k in leads + reserve:
            u = t + k
            if not (0 <= u < T) or not good[u]:
                continue
            ke = (ct[u] - ct[t]).astype(np.float32)             # effective lead in 5-min steps
            r = min(r0 + p["r_lead"] * (abs(k) - 1) + p["r_sig"] * abs(k) * sig_med, p["r_cap"])
            D, LV, LM, IV = dilated(u, int(round(r)), ob)
            sx = np.rint(xs + ke * vx).astype(np.int64); sy = np.rint(ys + ke * vy).astype(np.int64)
            ins = (sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)
            sxc, syc = np.clip(sx, 0, W - 1), np.clip(sy, 0, H - 1)
            sup = D[syc, sxc]; lev = LV[syc, sxc]; lmn = LM[syc, sxc]; iv = IV[syc, sxc]
            # a copy is no independent evidence: same content as t, or as the neighbour next to it toward t
            ex = (ct[u] == ct[t]) | (X[u][ys, xs] if k > 0 else X[u + 1][ys, xs])
            ok = iv <= p["inv_max"]
            if DROP is not None:
                ok &= ~DROP[u][np.minimum(syc // dt, DROP.shape[1] - 1), np.minimum(sxc // dt, DROP.shape[2] - 1)]
            av = ins & ~ex & (ok | (sup > 0))
            cols[k] = (sup, lev, lmn, av)
        if not cols:
            continue
        prim = [k for k in leads if k in cols]
        nprim = sum(cols[k][-1] for k in prim) if prim else np.zeros(len(ys), int)   # available primaries
        SU, LE, LMN, AV = [], [], [], []
        for k, (sup, lev, lmn, av) in cols.items():
            if k in reserve:
                av = av & (nprim < p["need2_min"])          # reserve leads only where primaries are short
            SU.append(sup); LE.append(lev); LMN.append(lmn); AV.append(av)
        SU = np.stack(SU).astype(np.int16); LE = np.stack(LE).astype(np.int16); AV = np.stack(AV)
        LMN = np.stack(LMN).astype(np.int16)
        nav = AV.sum(0)
        SUm = -np.sort(-np.where(AV, SU, -1), axis=0)        # descending; unavailable = -1 at the end
        LEm = -np.sort(-np.where(AV, LE, -1), axis=0)
        LMm = -np.sort(-np.where(AV, LMN, -1), axis=0)
        need = np.where((s >= p["s_need2"]) & (nav >= p["need2_min"]), 2, 1)
        need = np.minimum(need, np.maximum(nav, 1))
        sel = np.take_along_axis(SUm, (need - 1)[None], 0)[0]
        lsel = np.take_along_axis(LEm, (need - 1)[None], 0)[0]
        msel = np.take_along_axis(LMm, (need - 1)[None], 0)[0]
        qc = 1 + SC * np.log(c[ys, xs].astype(np.float32))           # exact log value of cur
        delta = SC * np.log1p(p["m_k"] / np.maximum(s, 1e-3))
        allowed = np.maximum(lsel, 1) + delta
        min_rm = np.where(s >= p["s_rm1"], 1, 2)
        rem = (nav >= min_rm) & ((msel <= 0) if p["rm_on_level"] else (sel <= 0))
        clamp = (nav >= p["min_nav_clamp"]) & ~rem & (qc > allowed)
        if not (rem.any() or clamp.any()):
            continue
        # reduction field in q units (inf for removed)
        red = np.zeros((H, W), np.float32)
        red[ys[clamp], xs[clamp]] = (qc - allowed)[clamp]
        if HARD is not None:
            hc = clamp & ((qc - allowed) > SC * np.log(1.0 / p["hard_ratio"]))
            HARD[t][ys[hc], xs[hc]] = True
        if p["smooth"] > 0 and clamp.any():
            rs = cv2.GaussianBlur(red, (0, 0), p["smooth"])
            red = np.maximum(red, np.where(rs >= p["smooth_min"], rs, 0))
        remm = np.zeros((H, W), bool)
        remm[ys[rem], xs[rem]] = True
        objclamp = None
        if p["weak0"] > 0 and rem.any():
            # weak echo is not what stc is for (flashes, speckle, intense noise): objects of cur[t] whose max code is
            # <= weak0 / s are never removed (old-chain drizzle fragments flicker but are rain)
            n_, lab_ = cv2.connectedComponents(wet.view(np.uint8), connectivity=8)
            mx = np.zeros(n_, np.int64)
            np.maximum.at(mx, lab_[ys, xs], c[ys, xs].astype(np.int64))
            weak = mx[lab_[ys, xs]] <= np.floor(p["weak0"] / np.maximum(s, 1e-3))
            rem &= ~weak
            remm[ys[weak], xs[weak]] = False
        if p["obj"] and rem.any():
            # object-level consistency (no half showers): per 8-connected object of cur[t], supported share f.
            # f >= obj_rm: unsupported pixels are not cut but clamped to the geometric mean of the supported ones;
            # f < obj_rm and object <= obj_big px: the whole object goes; larger objects: pixelwise as above.
            n, lab = cv2.connectedComponents(wet.view(np.uint8), connectivity=8)
            lw = lab[ys, xs]
            npx = np.bincount(lw, minlength=n)
            nrm = np.bincount(lw, weights=rem, minlength=n)
            sup = ~rem
            qsum = np.bincount(lw, weights=np.where(sup, qc, 0), minlength=n)
            f = 1 - nrm / np.maximum(npx, 1)
            gm = qsum / np.maximum(npx - nrm, 1)
            part = (f >= p["obj_rm"]) & (nrm > 0)
            whole = (f < p["obj_rm"]) & (npx <= p["obj_big"])
            whole[0] = part[0] = False
            pc = rem & part[lw]
            if pc.any() and p["app_max"] > 0:
                # small unsupported appendages of a supported object (dashes / flashes attached to rain) still go;
                # only larger unsupported parts are clamped (a growing / decaying piece of a real cell)
                pm = np.zeros((H, W), np.uint8); pm[ys[pc], xs[pc]] = 1
                n2, l2, st2, _ = cv2.connectedComponentsWithStats(pm, connectivity=8)
                small = st2[:, 4] <= p["app_max"] * np.maximum(sbar, 0.25)
                small[0] = False
                pc &= ~small[l2[ys, xs]]
            if pc.any():
                remm[ys[pc], xs[pc]] = False
                objclamp = (ys[pc], xs[pc], gm[lw[pc]])
            wh = whole[lw] & ~rem
            remm[ys[wh], xs[wh]] = True
        # no pin holes: a removed pixel with >= hole_nb kept wet 8-neighbours is clamped to their median instead
        keptw = wet & ~remm
        nb = cv2.filter2D(keptw.view(np.uint8).astype(np.float32), -1, _K8, borderType=cv2.BORDER_CONSTANT)
        pin = remm & (nb >= p["hole_nb"])
        lc = np.log(np.maximum(c, 1).astype(np.float32))
        newc = np.where(red > 0, np.maximum(np.floor(np.exp(lc - red / SC)), 1), c).astype(np.int64)
        newc = np.minimum(newc, c)
        if objclamp is not None:
            oy, ox, oq = objclamp
            newc[oy, ox] = np.minimum(c[oy, ox], np.maximum(np.floor(np.exp((oq - 1) / SC)), 1))
        if pin.any():
            py, px = np.nonzero(pin)
            cp = np.pad(np.where(keptw, newc, 0), 1)
            nbv = np.stack([cp[py + 1 + dy, px + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]).astype(np.float32)
            nbv[nbv == 0] = np.nan
            med = np.nanmedian(nbv, 0)
            newc[py, px] = np.minimum(c[py, px], np.maximum(np.floor(med), 1)).astype(c.dtype)
            remm &= ~pin
        newc[remm] = 0
        if p["flash_rm"]:
            # an object of cur[t] (<= obj_big px) of which NO pixel kept its value had no support at its level
            # anywhere: a flash / dash / hot spot. Clamped, it would stay as a weak residue: it goes entirely.
            chg = wet & (newc < c)
            n3, l3, st3, _ = cv2.connectedComponentsWithStats(wet.view(np.uint8), connectivity=8)
            unchanged = np.bincount(l3[wet & ~chg], minlength=n3)
            fl = (unchanged == 0) & (st3[:, 4] <= p["obj_big"])
            fl[0] = False
            if fl.any():
                fm = fl[l3] & wet
                remm |= fm
                newc[fm] = 0
        newc = np.where(wet, newc, c)
        out[t] = newc.astype(cur.dtype)
        info["n_removed"] += int(remm.sum()); info["n_clamped"] += int(((newc < c) & (newc > 0)).sum())
        if level is not None:
            level[t][(newc < c) & (newc > 0)] = 1
            level[t][remm] = 0
    # coherent-core filter in heavy-clutter regions + removal of hard-clamped dashes outside cores
    if p["ccf"] or HARD is not None:
        G = ccf_level(raw, times, S, ccf_gate, **p) if p["ccf"] else None
        out, cst = coherent_core(out, valid, old, V, G, HARD, **p)
        info.update(cst)
        if level is not None:
            level[(out < cur) & (out == 0) & np.isinf(level)] = 0
    # frozen halves stay copies
    for t in range(1, T):
        if not old[t]:
            continue
        for bit, sl in HALVES:
            if flags[t] & bit:
                same = raw[t][sl] == raw[t - 1][sl]
                out[t][sl] = np.where(same, np.minimum(out[t - 1][sl], cur[t][sl]), out[t][sl])
    out = np.where(valid, np.minimum(out, cur), cur).astype(cur.dtype)
    out[~old] = cur[~old]
    info["secs"] = time.time() - t0
    if level is not None:
        info["level"] = level
    return (out, info) if ret_info else out


_K8 = np.ones((3, 3), np.float32); _K8[1, 1] = 0
