"""Small/transient echo stage on top of v10 output (proto/weak).

    out2 = weak_stage(codes, out, times=None, **kw)          (ret_info=True: (out2, info))

codes: raw window (T, 765, 700) uint16; out: v10 output of the same window. Only pixels that v10 kept wet are ever
set to 0; values are never raised, codes >= 65534 never touched; frozen halves (lib/stale) re-copy the previous
output afterwards (as v10 does).

Temporal support of a point at t from frame u = t +- k (k = 1..K): a wet pixel of v10's output at u within
r_k = r0 + rk (k - 1) px of the point shifted by k * v (v = the window's median motion, phase correlation of the v10
output, |v| <= 8 px/frame). No evidence (neither support nor its absence): frames with < half the window's best
coverage, frozen halves, and points with missing data / outside the grid within r_k.

(1) Objects (8-connected, size <= smax) are removed when no pixel of them is supported in any evidence frame of
    lags 1..K on either side, and both sides have >= `need` evidence frames. Candidates:
    A  max code >= cmin (default 2: transient objects with a pixel >= 0.24 mm/h; in the new chain transient small
       objects are almost all code 1, in the old chain a quarter are not), size >= smin, and more than dnear px
       (default 15) from any other object of >= nearmin (21) px (keeps transient bits next to showers/rain edges);
    B  code-1-only objects (max code < cmin) only if size >= w1smin, >= w1dbig px from the nearest >= 100-px object
       and mean wet density in the 31 x 31 box < w1dens (isolated clear-air specks). Off by default (w1smin=99).
(2) fringe=True: pixels with code >= fcmin of larger objects with no support at lags 1..fK (radius fr + rk (k-1)) on
    both sides, only in objects whose unsupported share is <= fobj and in frames where <= fmax_share of the
    large-object pixels are unsupported ("blink frames", whole one-frame rain areas, are left alone).

Operating points (proto/weak/NOTES.md): Q0 = defaults; Q1 = + B (w1smin=2, w1dbig=60, w1dens=0.005) + fringe
(fcmin=1, fK=1); Q2 = + B (w1smin=1, w1dbig=60, w1dens=0.01) + fringe; P3 = cmin=1 dnear=0 + fringe (aggressive).
"""
from __future__ import annotations

import cv2
import numpy as np

from .stale import frozen_flags

MISSING = 65534
H, W = 765, 700
HALVES = ((1, slice(0, 380)), (2, slice(380, 765)))
DEFAULTS = dict(smin=1, cmin=2, w1smin=99, w1dbig=0, w1dens=1.0, dnear=15, nearmin=21, smax=20, cmax=65533, K=3, r0=3.0, rk=1.0, need=1, fringe=False, fcmin=2, fr=3.0, fK=2, fmax_share=0.1, fobj=0.3)
_HANN = None


def motion(out, valid):
    """Median 5-min displacement (dx, dy) of the echo field (phase correlation of consecutive frames), |v| <= 8."""
    global _HANN
    if _HANN is None:
        _HANN = cv2.createHanningWindow((W, H), cv2.CV_32F)
    F = np.where(valid, np.minimum(out, 60), 0).astype(np.float32)
    vs = []
    for t in range(0, len(out) - 1, 2):
        if (F[t] > 0).sum() < 3000 or (F[t + 1] > 0).sum() < 3000:
            continue
        a, b = cv2.GaussianBlur(F[t], (0, 0), 2), cv2.GaussianBlur(F[t + 1], (0, 0), 2)
        (dx, dy), resp = cv2.phaseCorrelate(a, b, _HANN)
        if resp > 0.05:
            vs.append((dx, dy))
    if not vs:
        return 0.0, 0.0
    v = np.median(np.array(vs), 0)
    return float(np.clip(v[0], -8, 8)), float(np.clip(v[1], -8, 8))


def evidence_rows(codes, flags, valid):
    T = len(codes)
    nv = valid.sum((1, 2))
    okf = nv >= 0.5 * nv.max()                                 # frames with < half the window's best coverage: no evidence
    ok = np.repeat(okf[:, None], H, 1)
    for u in range(T):
        for bit, sl in HALVES:
            if flags[u] & bit:
                ok[u, sl] = False
    return ok


def sample(dist_u, ok_u, valid_u, py, px, dx, dy, rad=0.0):
    """distance to the nearest wet pixel of frame u at the shifted points; NaN where no evidence. valid_u may be a
    distance-to-invalid map (float): then a point is evidence only if no invalid pixel lies within rad."""
    yy = np.rint(py + dy).astype(np.int64)
    xx = np.rint(px + dx).astype(np.int64)
    inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W)
    yy = np.clip(yy, 0, H - 1); xx = np.clip(xx, 0, W - 1)
    d = dist_u[yy, xx].astype(np.float32)
    vv = valid_u[yy, xx]
    bad = ~inside | ~ok_u[yy] | ((vv <= rad) if vv.dtype != bool else ~vv)
    d[bad] = np.nan
    return d


def support(dist, ok, dval, t, py, px, mv, K, r0, rk):
    """per point: supported (some evidence frame within lags 1..K, either side, has wet within r0 + rk (k-1) px of the
    advected point) and number of evidence frames per side (2, n)."""
    T = len(dist)
    sup = np.zeros(len(py), bool)
    nev = np.zeros((2, len(py)), np.int16)
    for si, sgn in enumerate((-1, 1)):
        for k in range(1, K + 1):
            u = t + sgn * k
            if u < 0 or u >= T:
                continue
            rad = r0 + rk * (k - 1)
            d = sample(dist[u], ok[u], dval[u], py, px, sgn * k * mv[0], sgn * k * mv[1], rad)
            fin = np.isfinite(d)
            nev[si] += fin
            sup |= fin & (d <= rad)
    return sup, nev


def weak_stage(codes, out, times=None, *, flags=None, mv=None, ret_info=False, **kw):
    p = {**DEFAULTS, **kw}
    codes = np.asarray(codes)
    T = len(codes)
    valid = codes < MISSING
    wet = (out > 0) & valid
    if flags is None:
        flags = frozen_flags(codes)
    ok = evidence_rows(codes, flags, valid)
    if mv is None:
        mv = motion(out, valid)
    dist = [cv2.distanceTransform((~wet[u]).astype(np.uint8), cv2.DIST_L2, 3) for u in range(T)]
    dval = [cv2.distanceTransform(np.pad(valid[u], 1, constant_values=False).astype(np.uint8), cv2.DIST_L2, 3)[1:-1, 1:-1] for u in range(T)]
    res = out.copy()
    nrem = np.zeros((2, T), np.int64)
    for t in range(T):
        wt = wet[t]
        if not wt.any():
            continue
        live = ok[t].copy()                                   # rows of t that are not a frozen copy
        n, lab, st, _ = cv2.connectedComponentsWithStats(wt.astype(np.uint8), connectivity=8)
        size = st[:, cv2.CC_STAT_AREA].copy(); size[0] = 0
        py, px = np.nonzero(wt)
        L = lab[py, px]
        c = out[t][py, px].astype(np.int64)
        mx = np.zeros(n, np.int64); np.maximum.at(mx, L, c)
        kill = np.zeros(len(py), bool)
        # (1) small weak objects without temporal support
        # A: objects with a pixel >= cmin (old-chain specific); B: code-below-cmin objects, only with stricter gates
        cA = (size >= p["smin"]) & (mx >= p["cmin"])
        cB = (size >= p["w1smin"]) & (mx < p["cmin"])
        cobj = (size <= p["smax"]) & (mx <= p["cmax"]) & (size > 0) & (cA | cB)
        if p["dnear"] > 0 and cA.any():                       # A objects near other echo (>= nearmin px) are kept
            near_src = size >= p["nearmin"]
            if near_src[1:].any():
                dn = cv2.distanceTransform((~near_src[lab]).astype(np.uint8), cv2.DIST_L2, 3)
                dmin = np.full(n, np.inf, np.float32); np.minimum.at(dmin, L, dn[py, px])
                cobj &= ~(cA & ~cB) | (dmin > p["dnear"])
        if p["w1dbig"] > 0 and cB.any():
            big = size >= 100
            dbig = cv2.distanceTransform((~big[lab]).astype(np.uint8), cv2.DIST_L2, 3) if big.any() else np.full((H, W), 1e9, np.float32)
            far = np.full(n, np.inf, np.float32); np.minimum.at(far, L, dbig[py, px])
            cobj &= ~cB | (far >= p["w1dbig"])
        if p["w1dens"] < 1 and cB.any():
            dens = cv2.boxFilter(wt.astype(np.float32), -1, (31, 31), normalize=True, borderType=cv2.BORDER_CONSTANT)
            dsum = np.bincount(L, weights=dens[py, px], minlength=n)
            cobj &= ~cB | (dsum < p["w1dens"] * np.maximum(size, 1))
        m = cobj[L] & live[py]
        if m.any():
            s1, ev1 = support(dist, ok, dval, t, py[m], px[m], mv, p["K"], p["r0"], p["rk"])
            osup = np.zeros(n, bool); osup[L[m][s1]] = True
            oev = np.zeros((2, n), bool)
            oev[0, L[m][ev1[0] >= 1]] = True; oev[1, L[m][ev1[1] >= 1]] = True
            okill = cobj & ~osup & oev[0] & oev[1]
            if p["need"] > 1:
                cnt = np.zeros((2, n), np.int64)
                np.maximum.at(cnt[0], L[m], ev1[0]); np.maximum.at(cnt[1], L[m], ev1[1])
                okill &= (cnt[0] >= p["need"]) & (cnt[1] >= p["need"])
            kill |= okill[L] & live[py]
            nrem[0, t] = int(kill.sum())
        # (2) fringe: unsupported pixels of larger, supported objects, outside blink frames
        if p["fringe"]:
            m = (size[L] > p["smax"]) & live[py] & (c >= p["fcmin"])
            mm = (size[L] > p["smax"]) & live[py]
            s2, ev2 = support(dist, ok, dval, t, py[mm], px[mm], mv, p["fK"], p["fr"], p["rk"])
            have = (ev2[0] >= 1) & (ev2[1] >= 1)
            uns = ~s2 & have
            if have.sum() and uns.sum() <= p["fmax_share"] * have.sum():      # not a blink frame
                Lm = L[mm]
                nu = np.bincount(Lm, weights=uns, minlength=n); nh = np.bincount(Lm, weights=have, minlength=n)
                osup = nu <= p["fobj"] * np.maximum(nh, 1)                    # object as a whole is supported
                fk = np.zeros(len(py), bool)
                fk[np.nonzero(mm)[0][uns & osup[Lm]]] = True
                fk &= m
                kill |= fk
                nrem[1, t] = int(fk.sum())
        if kill.any():
            res[t][py[kill], px[kill]] = 0
    for t in range(1, T):                                           # frozen halves copy the previous output
        for bit, sl in HALVES:
            if flags[t] & bit:
                same = codes[t][sl] == codes[t - 1][sl]
                res[t][sl] = np.where(same, res[t - 1][sl], res[t][sl])
    res = np.where(valid, np.minimum(res, out), codes).astype(codes.dtype)
    return (res, {"mv": mv, "removed": nrem[0] + nrem[1], "removed_obj": nrem[0], "removed_fringe": nrem[1]}) if ret_info else res
