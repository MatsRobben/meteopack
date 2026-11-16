"""fixclut: earth-fixed (stationary) clutter stage for old-chain RTCOR (anaprop, static spots, sea clutter).

    fixclut_clean(cur, raw, times, strength=1.0, ret_info=False, **kw) -> out [, info]

cur: (T, 765, 700) uint16 output of the base cleaner (v12), raw: the raw codes, times: (T,) datetime64.
Output <= cur, codes >= 65534 unchanged, new-chain frames unchanged.

Idea: earth-fixed echo persists IN PLACE while the rain around it moves with the flow.
 1. usable pixels: valid, not in a frozen half (lib/stale.py rule), not a per-pixel copy of the previous frame.
 2. scene flow from the moving echo of `cur` (Farneback on a 4 km grid, pairs 10 min apart, frozen frames skipped),
    echo-weighted smoothing over ~40 km, temporal median.
 3. dense persistence at a 13 km scale for pairs (t, t+k), k = 3..6 (15-30 min): fuzzy overlap of the soft echo
    field with itself at the same place (Eulerian, E) and with the frame advected by k * flow (Lagrangian, L).
    Only pairs with a clear displacement (k |v| >= dmin px) count. F = E - L.
 4. texture: local share of loose (<= 3 wet 8-neighbours) wet px, and pixel-to-pixel jumps.
 5. decision over time at a fixed place: F and texture are averaged along time at the same pixel (centred
    window), then a Schmitt trigger along time (forward and backward); echo objects of cur that lie mostly
    (>= obj_share) on flagged area are removed whole (21..1500 px; smaller only as fragments next to such an object).
Iteration 2 (2026-11-16):
 6. clear-air earth-fixed rule (static=True): per object of cur, against the raw frames t-+3..8 per side (frozen /
    missing px excluded, the stabler side counts): Eulerian corr of the 3x3-smoothed log-rate pattern over the
    object (+2 px) >= 0.6, footprint retention (1 px tolerance) >= 0.6, internal-pattern corr >= 0.3, AND a sharp
    zero-shift peak: corr at zero shift minus the best corr at a 3 px shift >= 0.1 (slow / evolving rain has a flat
    or offset peak). Two vectorised prechecks keep it cheap.
 7. propagation over time in both directions (prop=True): an object at the same place (>= 60 % inside the removed
    footprint of t-+1/2, dilated 2 px) and inside the window's earth-fixed seed region is the same object -> removed
    too, so a blob is clutter for its whole lifetime (no start lag, no on/off).
 8. lobes (lobe=True): the part of a larger object inside the footprint removed BOTH before and after t (within
    -+4 frames) is cut when <= 30 % of its boundary touches the rest of the object (clutter attached to rain).
"""
# Ported from research proto/fixclut/fixclut.py (md5 add03a0cc84dc68e78a91163c2be6135),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations

import time as _time

import cv2
import numpy as np
from scipy import ndimage as ndi

MISSING = 65534
H, W = 765, 700
NEW_CHAIN = np.datetime64("2023-01-31T10:45", "ns")
HALVES = ((1, slice(0, 380)), (2, slice(380, 765)))
DS = 4

DEFAULTS = dict(
    ks=(3, 4, 5, 6),        # pair lags (frames)
    box=13,                 # persistence scale (px)
    dmin=5.0,               # min displacement k|v| for a pair to count (px)
    twin=6,                 # temporal half window for the decision average (frames)
    f_on=0.30, f_off=0.18,  # Schmitt thresholds on the combined score
    w_tex=0.5,              # weight of the texture term
    min_pairs=3,            # min usable pairs (in the time window) for a decision
    obj_share=0.6,          # object removed whole when >= this share of its px is flagged
    obj_max=1500,           # objects larger than this are never removed whole (capped instead)
    cap=False,              # (removed in iteration 2; kept as a no-op key)
    static=True,            # clear-air earth-fixed rule (object footprint + log-rate pattern fixed for 30-45 min)
    st_min=8, st_nl=3, st_c=0.6, st_j=0.6, st_ci=0.3, st_pk=0.1, st_pre=0.5, st_pre_c=0.2,
    prop=True, prop_share=0.6,
    lobe=True, lobe_min=20, lobe_contact=0.3, lobe_c=-1.0,   # two-sided lobe cut; lobe_c > -1 adds a pattern test
    obj_min=21,             # smaller objects are removed only when inside the flagged area of a removed object's
                            # 8 px surroundings (fragments of the clutter blob), never on their own
)


def frozen_flags(codes, min_wet=300, share=0.85):
    flags = np.zeros(len(codes), np.uint8)
    for t in range(1, len(codes)):
        for bit, sl in HALVES:
            a, b = codes[t][sl], codes[t - 1][sl]
            w = (a > 0) & (a < MISSING)
            if w.sum() >= min_wet and (a[w] == b[w]).mean() > share:
                flags[t] |= bit
    return flags


def usable_masks(raw):
    """(T, H, W) bool: pixel is usable evidence (valid, not a frozen half, not a per-pixel copy)."""
    T = len(raw)
    U = raw < MISSING
    fl = frozen_flags(raw)
    for t in range(T):
        for bit, sl in HALVES:
            if fl[t] & bit:
                U[t, sl] = False
        if t > 0:
            cp = (raw[t] == raw[t - 1]) & (raw[t] > 0) & (raw[t] < MISSING)
            # per-pixel copies only matter where many wet px repeat exactly (Den Helder side); a lone equal code is
            # chance. Mark copies where the local share of exact repeats among wet px is high.
            wet = ((raw[t] > 0) & (raw[t] < MISSING)).astype(np.float32)
            rep = cv2.boxFilter(cp.astype(np.float32), -1, (9, 9)) / np.maximum(cv2.boxFilter(wet, -1, (9, 9)), 1e-3)
            U[t] &= ~(cp & (rep > 0.5))
    return U, fl


_CODES = np.arange(65536, dtype=np.float32)
_SOFT_LUT = np.where(_CODES < MISSING, np.clip(np.log1p(_CODES * 0.12 * 4) / np.log1p(12.0), 0, 1), 0).astype(np.float32)
_LOGR_LUT = np.where(_CODES < MISSING, np.log1p(_CODES), 0).astype(np.float32)


def soft(c):
    """soft echo field in [0, 1]: rate 0 -> 0, >= ~3 mm/h -> 1, log scale (lookup table on the codes)."""
    return _SOFT_LUT[c]


def _small(x):
    h, w = x.shape
    return cv2.resize(x, (w // DS, h // DS), interpolation=cv2.INTER_AREA)


def scene_flow(cur, U, fl, gap=2, sig=10.0, vmov=0.3):
    """(T, 2, H, W) flow in px per 5 min (x, y) from the moving echo of cur. Farneback on a 4 km grid between
    frames t and t+gap (shifted to the nearest pair whose halves are not frozen), echo-weighted smoothing
    (sigma `sig` cells), then a temporal median over 5 pairs."""
    T = len(cur)
    S = [_small(soft(cur[t]) * U[t]) for t in range(T)]
    ok = np.array([fl[t] != 3 for t in range(T)])
    pairs = []
    for t in range(T):
        a = t
        b = min(t + gap, T - 1)
        if b <= a:
            a = max(T - 1 - gap, 0); b = T - 1
        # skip fully frozen frames
        while a > 0 and not ok[a]:
            a -= 1
        while b < T - 1 and not ok[b]:
            b += 1
        pairs.append((a, b))
    cache = {}
    Fs = np.zeros((T, 2) + S[0].shape, np.float32)
    Ws = np.zeros((T,) + S[0].shape, np.float32)
    for t, (a, b) in enumerate(pairs):
        if (a, b) not in cache:
            A8 = (S[a] * 255).astype(np.uint8); B8 = (S[b] * 255).astype(np.uint8)
            f = cv2.calcOpticalFlowFarneback(A8, B8, None, 0.5, 3, 9, 3, 5, 1.1, 0) / max(b - a, 1)
            w = np.minimum(S[a], S[b]) + 0.5 * np.maximum(S[a], S[b])
            w = w * (np.hypot(f[..., 0], f[..., 1]) > vmov)      # only moving echo sets the flow
            ws = cv2.GaussianBlur(w, (0, 0), sig) + 1e-4
            fx = cv2.GaussianBlur(f[..., 0] * w, (0, 0), sig) / ws
            fy = cv2.GaussianBlur(f[..., 1] * w, (0, 0), sig) / ws
            # global fallback where little echo: weighted mean of the frame
            g = w.sum() + 1e-6
            gx, gy = (f[..., 0] * w).sum() / g, (f[..., 1] * w).sum() / g
            lam = np.clip(ws / (np.percentile(ws, 99) * 0.1 + 1e-6), 0, 1)
            fx = lam * fx + (1 - lam) * gx; fy = lam * fy + (1 - lam) * gy
            cache[(a, b)] = (np.stack([fx, fy]), ws)
        Fs[t], Ws[t] = cache[(a, b)]
    # temporal median over 5 pairs
    Fm = np.empty_like(Fs)
    for t in range(T):
        lo, hi = max(0, t - 2), min(T, t + 3)
        Fm[t] = np.median(Fs[lo:hi], axis=0)
    return Fm * DS  # in full-res px per 5 min, on the small grid


def up(f):
    """(2, h, w) small-grid field -> half grid."""
    x = cv2.resize(f[0], (W2, H2), interpolation=cv2.INTER_LINEAR)
    y = cv2.resize(f[1], (W2, H2), interpolation=cv2.INTER_LINEAR)
    return x, y


H2, W2 = (H + 1) // 2, W // 2
GX, GY = np.meshgrid(np.arange(W2, dtype=np.float32), np.arange(H2, dtype=np.float32))
K3 = np.ones((3, 3), np.float32); K3[1, 1] = 0


def persistence(S, U, Fm, ks, box, dmin):
    """Dense sums over pairs: num/den of (E - L) at each (t, px), counting each pair at both of its frames."""
    T = len(S)
    acc = np.zeros((T, H2, W2), np.float32)     # sum of (E - L) * weight
    wsum = np.zeros((T, H2, W2), np.float32)    # sum of weights
    esum = np.zeros((T, H2, W2), np.float32)    # sum of E * weight
    bx = (box, box)
    for t in range(T):
        vx, vy = up(Fm[t])
        sp = np.hypot(vx, vy)
        for k in ks:
            u = t + k
            if u >= T:
                break
            A, B = S[t], S[u]
            M = (U[t] & U[u]).astype(np.float32)
            # Eulerian
            mn = cv2.boxFilter(np.minimum(A, B) * M, -1, bx); mx = cv2.boxFilter(np.maximum(A, B) * M, -1, bx)
            E = mn / np.maximum(mx, 1e-3)
            # Lagrangian: B sampled at x + k v
            mapx = GX + k * vx; mapy = GY + k * vy
            Bw = cv2.remap(B, mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            Mw = cv2.remap(U[u].astype(np.float32), mapx, mapy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT,
                           borderValue=0) * U[t]
            mnl = cv2.boxFilter(np.minimum(A, Bw) * Mw, -1, bx); mxl = cv2.boxFilter(np.maximum(A, Bw) * Mw, -1, bx)
            L = mnl / np.maximum(mxl, 1e-3)
            wt = ((k * sp >= dmin) & (mx > 0.02) & (mxl > 0.02)).astype(np.float32)
            d = (E - L) * wt
            for s in (t, u):
                acc[s] += d; wsum[s] += wt; esum[s] += E * wt
    return acc, wsum, esum


def texture(cur, box):
    """per frame: share of loose wet px (<= 3 wet 8-neighbours) and mean |jump| / level in a box window."""
    T = len(cur)
    tex = np.zeros((T, H2, W2), np.float32)
    for t in range(T):
        w = ((cur[t] > 0) & (cur[t] < MISSING)).astype(np.float32)
        n8 = cv2.filter2D(w, -1, K3, borderType=cv2.BORDER_CONSTANT)
        bw = cv2.boxFilter(w, -1, (box, box), normalize=False)
        loose = cv2.boxFilter(w * (n8 <= 4), -1, (box, box), normalize=False) / np.maximum(bw, 1)
        tex[t] = half(loose)
    return tex


def time_mean(x, wsum, twin):
    """centred running sum along time (window 2*twin+1)."""
    c = np.cumsum(np.concatenate([np.zeros((1,) + x.shape[1:], x.dtype), x]), axis=0)
    T = len(x)
    lo = np.clip(np.arange(T) - twin, 0, T); hi = np.clip(np.arange(T) + twin + 1, 0, T)
    return c[hi] - c[lo]


def half(x, interp=cv2.INTER_AREA):
    return cv2.resize(x, (W2, H2), interpolation=interp)


def features(cur, raw, times=None, **kw):
    p = {**DEFAULTS, **kw}
    U, fl = usable_masks(raw)
    Fm = scene_flow(cur, U, fl)
    S = np.stack([half(soft(c)) for c in raw])        # persistence of the RAW echo (cur switches clutter on/off)
    U2 = np.stack([half(u.astype(np.float32)) > 0.75 for u in U])
    acc, wsum, esum = persistence(S, U2, Fm / 2, p["ks"], max(p["box"] // 2 | 1, 3), p["dmin"] / 2)
    tex = texture(raw, p["box"])
    return dict(U=U, U2=U2, fl=fl, S=S, Fm=Fm, acc=acc, wsum=wsum, esum=esum, tex=tex)


def fixclut_clean(cur, raw, times=None, strength=1.0, ret_info=False, **kw):
    t0 = _time.time()
    p = {**DEFAULTS, **kw}
    cur = np.asarray(cur); raw = np.asarray(raw)
    T = len(cur)
    old = np.ones(T, bool) if times is None else (np.asarray(times).astype("datetime64[ns]") < NEW_CHAIN)
    if strength <= 0 or not old.any():
        return (cur.copy(), {}) if ret_info else cur.copy()
    f = features(cur, raw, times, **p)
    tm = {"features": _time.time() - t0}
    A = time_mean(f["acc"], None, p["twin"]); Wt = time_mean(f["wsum"], None, p["twin"])
    F = A / np.maximum(Wt, 1e-3)
    wet = (cur > 0) & (cur < MISSING)
    wet2 = np.stack([half(w.astype(np.float32)) for w in wet])
    # texture averaged over time at the same place, weighted by echo presence
    tx = time_mean(f["tex"] * wet2, None, p["twin"]) / np.maximum(time_mean(wet2, None, p["twin"]), 1e-3)
    score = F + p["w_tex"] * (tx - 0.3)
    score = np.where(Wt >= p["min_pairs"], score, -1).astype(np.float32)
    on = p["f_on"] / strength; off = p["f_off"] / strength
    # Schmitt trigger along time at each pixel (forward and backward pass, union)
    flag2 = np.zeros((T, H2, W2), bool)
    st = np.zeros((H2, W2), bool)
    for t in range(T):
        st = np.where(st, score[t] > off, score[t] > on); flag2[t] = st
    st = np.zeros((H2, W2), bool)
    for t in range(T - 1, -1, -1):
        st = np.where(st, score[t] > off, score[t] > on); flag2[t] |= st
    out = cur.copy()
    flag = np.zeros((T, H, W), bool)
    labs, rmobj, cnts = [], [], []
    for t in range(T):
        flag[t] = cv2.resize(flag2[t].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        w = wet[t]
        lab, n = ndi.label(w, structure=np.ones((3, 3)))
        cnt = np.bincount(lab.ravel(), minlength=n + 1)
        labs.append(lab); cnts.append(cnt)
        whole = np.zeros(n + 1, bool)
        if old[t] and (flag[t] & w).any():
            fw = flag[t] & w
            fc = np.bincount(lab.ravel(), weights=fw.ravel(), minlength=n + 1)
            share = fc / np.maximum(cnt, 1)
            whole = (share >= p["obj_share"]) & (cnt <= p["obj_max"]); whole[0] = False
        rmobj.append(whole)
    tm["labels"] = _time.time() - t0 - sum(tm.values())
    # small fragments only next to a removed object of >= obj_min px
    for t in range(T):
        lab, cnt, whole = labs[t], cnts[t], rmobj[t]
        if not whole.any():
            continue
        big = whole & (cnt >= p["obj_min"])
        near_big = cv2.dilate(big[lab].astype(np.uint8), np.ones((17, 17), np.uint8)).astype(bool)
        frag = np.bincount(lab.ravel(), weights=near_big.ravel(), minlength=len(cnt)) > 0
        rmobj[t] = whole & ((cnt >= p["obj_min"]) | frag)
    info = {}
    tm["motion_decision"] = _time.time() - t0 - sum(tm.values())
    # clear-air earth-fixed rule: footprint AND log-rate pattern stay in place for 30-45 min
    if p["static"]:
        U = f["U"]
        Lr = np.stack([cv2.blur(logr(c), (3, 3)) for c in raw]); Wr = (raw > 0) & (raw < MISSING)
        k3 = np.ones((3, 3), np.uint8)
        Wd = np.stack([cv2.dilate(w.astype(np.uint8), k3) for w in Wr]).astype(bool)
        nst = 0
        for t in range(T):
            if not old[t]:
                continue
            lab, cnt = labs[t], cnts[t]
            # cheap precheck (vectorised): footprint retention at the same place at t-+3 and t-+6 on the better side;
            # objects of moving / evolving echo fail it and skip the per-object test
            szok = (cnt >= p["st_min"]) & (cnt <= p["obj_max"]) & ~rmobj[t]; szok[0] = False
            if not szok.any():
                continue
            idx = np.flatnonzero(szok[lab].ravel())          # px of candidate objects only (fast)
            li0 = lab.ravel()[idx]
            wt = Wr[t].ravel()[idx] & U[t].ravel()[idx]
            nw = np.bincount(li0, weights=wt, minlength=len(cnt))
            pre = np.zeros(len(cnt))
            for sgn in (-1, 1):
                side = np.ones(len(cnt))
                for grp in ((3, 4), (6, 7)):
                    g = np.full(len(cnt), -1.0)          # best usable retention in this lag group (-1: none usable)
                    for k in grp:
                        u = t + sgn * k
                        if not (0 <= u < T):
                            continue
                        uu = wt & U[u].ravel()[idx]
                        den = np.bincount(li0, weights=uu, minlength=len(cnt))
                        num = np.bincount(li0, weights=uu & Wd[u].ravel()[idx], minlength=len(cnt))
                        g = np.maximum(g, np.where(den >= 0.8 * nw, num / np.maximum(den, 1), -1.0))
                    side = np.minimum(side, g)
                pre = np.maximum(pre, side)
            cand = (pre >= p["st_pre"]) & (cnt >= p["st_min"]) & (cnt <= p["obj_max"]) & ~rmobj[t]; cand[0] = False
            if not cand.any():
                continue
            # second cheap precheck: Eulerian log-rate correlation over the object dilated by 2 px at t-+6/7
            labf = lab.astype(np.float32)
            labd = np.where(lab > 0, labf, cv2.dilate(labf, np.ones((5, 5), np.uint8))).astype(np.int64)
            labd = np.where(cand[labd], labd, 0).ravel()
            id2 = np.flatnonzero(labd)
            ld = labd[id2]; xr = Lr[t].ravel()[id2].astype(np.float64); ut = U[t].ravel()[id2]
            prec = np.full(len(cnt), -1.0)
            for u in (t - 7, t - 6, t + 6, t + 7):
                if not (0 <= u < T):
                    continue
                P = ut & U[u].ravel()[id2]
                if not P.any():
                    continue
                li = ld[P]; xv = xr[P]; yv = Lr[u].ravel()[id2][P].astype(np.float64)
                bc = lambda w: np.bincount(li, weights=w, minlength=len(cnt))
                n_ = bc(None) if False else np.bincount(li, minlength=len(cnt)).astype(np.float64)
                sx, sy, sxx, syy, sxy = bc(xv), bc(yv), bc(xv * xv), bc(yv * yv), bc(xv * yv)
                nn = np.maximum(n_, 1)
                cov = sxy / nn - sx * sy / nn ** 2
                vx = sxx / nn - (sx / nn) ** 2; vy = syy / nn - (sy / nn) ** 2
                c = np.where((vx > 1e-9) & (vy > 1e-9), cov / np.sqrt(np.maximum(vx * vy, 1e-18)), 0.0)
                prec = np.maximum(prec, c)
            cand &= prec >= p["st_c"] / strength - p["st_pre_c"]
            if not cand.any():
                continue
            sls = ndi.find_objects(lab)
            for j in np.flatnonzero(cand):
                s0 = sls[j - 1]
                sl = tuple(slice(max(a.start - 4, 0), a.stop + 4) for a in s0)
                m = lab[sl] == j
                nl, cE, J, cI, pk = object_static_feats(Lr, Wr, U, t, sl, m)   # C=raw tried: worse (NOTES)
                if (nl >= p["st_nl"] and cE >= p["st_c"] / strength and J >= p["st_j"]
                        and (np.isnan(cI) or cI >= p["st_ci"]) and pk >= p["st_pk"]):
                    rmobj[t][j] = True; nst += 1
        info["n_static_obj"] = nst
        tm["static"] = _time.time() - t0 - sum(tm.values())
        del Lr, Wr, Wd
    # propagation over time in both directions: an object at the same place as a removed object at t-1/t-2 or
    # t+1/t+2 (>= prop_share of its px inside that footprint dilated by 2 px) and inside the window's earth-fixed
    # seed region (union of the primary removals, dilated by 3 px) is the same earth-fixed object -> removed too.
    if p["prop"]:
        anyrm = np.zeros((H, W), bool)
        for t in range(T):
            if rmobj[t].any():
                anyrm |= rmobj[t][labs[t]]
        npr = 0
        if anyrm.any():
            ys, xs = np.nonzero(anyrm)
            S = (slice(max(ys.min() - 16, 0), ys.max() + 17), slice(max(xs.min() - 16, 0), xs.max() + 17))
            seed = cv2.dilate(anyrm[S].astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            labc = [labs[t][S] for t in range(T)]
            rmm = [rmobj[t][labc[t]] for t in range(T)]
            k5 = np.ones((5, 5), np.uint8)
            for order in (range(T), range(T - 1, -1, -1), range(T)):
                for t in order:
                    if not old[t]:
                        continue
                    lab, cnt = labc[t], cnts[t]
                    nb = [rmm[u] for u in (t - 2, t - 1, t + 1, t + 2) if 0 <= u < T and rmm[u].any()]
                    if not nb:
                        continue
                    foot = cv2.dilate(np.any(nb, axis=0).astype(np.uint8), k5).astype(bool)
                    fin = np.bincount(lab.ravel(), weights=foot.ravel(), minlength=len(cnt)) / np.maximum(cnt, 1)
                    sin = np.bincount(lab.ravel(), weights=seed.ravel(), minlength=len(cnt)) / np.maximum(cnt, 1)
                    new = (fin >= p["prop_share"]) & (sin >= 0.8) & (cnt <= p["obj_max"]) & ~rmobj[t]; new[0] = False
                    if new.any():
                        rmobj[t] = rmobj[t] | new; rmm[t] = rmobj[t][lab]; npr += int(new.sum())
        info["n_prop_obj"] = npr
        tm["prop"] = _time.time() - t0 - sum(tm.values())
    nrm = 0
    RM = [rmobj[t][labs[t]] if rmobj[t].any() else None for t in range(T)]
    for t in range(T):
        if old[t] and RM[t] is not None:
            out[t][RM[t]] = 0; nrm += int(RM[t].sum())
    # lobes: the same earth-fixed object, attached in this frame to a larger (rain) object. The part of that object
    # inside the clutter footprint of the neighbouring frames (t-+1..4, where it was removed as a separate object)
    # is removed when (a) its log-rate pattern matches the removed clutter at the same place (corr >= lobe_c, so rain
    # passing over the spot is kept) and (b) it is a lobe: <= lobe_contact of its boundary touches the rest.
    nlobe = 0
    if p["lobe"]:
        k3 = np.ones((3, 3), np.uint8)
        for t in range(T):
            if not old[t]:
                continue
            nb = [(abs(u - t), u) for u in range(t - 4, t + 5) if u != t and 0 <= u < T and RM[u] is not None]
            bef = [RM[u] for _, u in nb if u < t]; aft = [RM[u] for _, u in nb if u > t]
            if not bef or not aft:
                continue
            # footprint removed both before AND after t at the same place (earth-fixed on both sides)
            foot = cv2.dilate((np.any(bef, axis=0) & np.any(aft, axis=0)).astype(np.uint8), k3).astype(bool)
            lab, cnt = labs[t], cnts[t]
            fin = np.bincount(lab.ravel(), weights=foot.ravel(), minlength=len(cnt))
            cand = (fin >= p["lobe_min"]) & ~rmobj[t]; cand[0] = False
            if not cand.any():
                continue
            near = [u for _, u in sorted(nb)[:3]]
            sls = ndi.find_objects(lab)
            for j in np.flatnonzero(cand):
                s0 = sls[j - 1]
                sl = tuple(slice(max(a.start - 2, 0), a.stop + 2) for a in s0)
                m = lab[sl] == j
                lobe = m & foot[sl]
                rest = m & ~lobe
                if lobe.sum() < p["lobe_min"] or rest.sum() < lobe.sum():
                    continue
                # contact: lobe px with a 4-neighbour in the rest of the object, against the lobe's boundary length
                rd = cv2.dilate(rest.astype(np.uint8), np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)).astype(bool)
                bnd = lobe & ~cv2.erode(lobe.astype(np.uint8), k3).astype(bool)
                if (lobe & rd).sum() > p["lobe_contact"] * max(bnd.sum(), 1):
                    continue
                # pattern match with the removed clutter at the same place
                a = cv2.blur(logr(raw[t][sl]), (3, 3))[lobe]
                cs = []
                for u in near:
                    b = cv2.blur(logr(raw[u][sl]), (3, 3))[lobe]
                    if a.std() > 0 and b.std() > 0:
                        cs.append(float(np.corrcoef(a, b)[0, 1]))
                if not cs or np.median(cs) < p["lobe_c"]:
                    continue
                o = out[t][sl]; o[lobe] = 0
                nlobe += 1; nrm += int(lobe.sum())
    info["n_lobe"] = nlobe
    out = np.where(cur >= MISSING, cur, np.minimum(out, cur)).astype(cur.dtype)
    out = np.where(old[:, None, None], out, cur)
    if ret_info:
        return out, dict(score=score, flag=flag, F=F, tx=tx, Wt=Wt, Fm=f["Fm"], secs=_time.time() - t0, nrm=nrm, tm=tm, **info)
    return out


# ------------------------------------------------------------------ clear-air earth-fixed rule (iteration 2)
def logr(c):
    return _LOGR_LUT[c]


def object_static_feats(Lr, Wr, U, t, sl, m, lags=(3, 4, 5, 6, 7, 8), dil=2, minfrac=0.8, tol=1, min_side=3, sh=3, C=None,
                        pkmode="max"):
    """Eulerian stability of one object (mask m inside slice sl of frame t) against frames t+-k, per side
    (past / future); the stabler side counts, so an object that lives 45-60 min is judged at both ends.
    Lr: log-rate fields already smoothed by a 3x3 box (1 px position tolerance); Wr: wet masks.
    Per lag: c = corr of the log-rate pattern over the object dilated by `dil` px, J = share of the object's wet px
    with wet px within `tol` px at t+k, cI = corr over the object's px only (internal pattern).
    Returns (n usable lags on the best side, median c, median J, median cI) for the side with the larger median c;
    lags whose px are < minfrac usable (frozen copies, missing data) are skipped; a side needs >= min_side lags.
    With C (raw codes): frame t itself may be a copy of t-1 (static echo repeats its codes exactly, which the
    per-pixel copy rule marks unusable); a lag is then skipped only when > 50 % of the object's wet codes at t
    and t+k are identical (a copy is never compared with itself)."""
    R = cv2.dilate(m.astype(np.uint8), np.ones((2 * dil + 1, 2 * dil + 1), np.uint8)).astype(bool)
    T = len(Lr)
    a = Lr[t][sl]
    wt = Wr[t][sl] & m
    best = (0, np.nan, np.nan, np.nan, 0, [])
    kt = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    for sgn in (-1, 1):
        cs, js, ci, us = [], [], [], []
        for k in lags:
            u = t + sgn * k
            if not (0 <= u < T):
                continue
            P = R & U[u][sl] if C is not None else R & U[t][sl] & U[u][sl]
            if P.sum() < minfrac * R.sum():
                continue
            if C is not None:
                ow = m & Wr[t][sl]
                if (ow & (C[t][sl] == C[u][sl])).sum() > 0.5 * max(ow.sum(), 1):
                    continue
            b = Lr[u][sl]
            x, y = a[P], b[P]
            sx, sy = x.std(), y.std()
            cs.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 0 and sy > 0 else 0.0)
            us.append(u)
            wu = cv2.dilate(Wr[u][sl].astype(np.uint8), kt).astype(bool)
            js.append(float((wt & wu & P).sum() / max((wt & P).sum(), 1)))
            mm = m & P
            if mm.sum() >= 8:
                x, y = a[mm], b[mm]
                sx, sy = x.std(), y.std()
                ci.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 0 and sy > 0 else 0.0)
        if len(cs) >= min_side:
            c = float(np.median(cs))
            if np.isnan(best[1]) or c > best[1]:
                best = (len(cs), c, float(np.median(js)), float(np.median(ci)) if ci else np.nan, sgn, us)
    if best[0] == 0:
        return (0, np.nan, np.nan, np.nan, np.nan)
    # peak test on the best side's two largest usable lags: corr at zero shift minus the best corr at a 3 px shift
    # (8 directions). Earth-fixed echo peaks sharply at zero; slow or evolving rain has a flat or offset peak.
    pk = []
    r0, c0 = sl[0].start, sl[1].start
    r1, c1 = r0 + a.shape[0], c0 + a.shape[1]
    for u in best[5][-2:]:
        P = R & U[u][sl] if C is not None else R & U[t][sl] & U[u][sl]
        x = a[P]; sx = x.std()
        if sx == 0:
            continue
        cc = []
        for dy, dx in ((0, 0), (-sh, 0), (sh, 0), (0, -sh), (0, sh), (-sh, -sh), (-sh, sh), (sh, -sh), (sh, sh)):
            if r0 + dy < 0 or c0 + dx < 0 or r1 + dy > H or c1 + dx > W:
                continue
            y = Lr[u][r0 + dy:r1 + dy, c0 + dx:c1 + dx][P]; sy = y.std()
            cc.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sy > 0 else 0.0)
        if len(cc) > 1:
            pk.append(cc[0] - (np.mean(cc[1:]) if pkmode == "mean" else max(cc[1:])))
    return best[:4] + (float(np.median(pk)) if pk else np.nan,)
