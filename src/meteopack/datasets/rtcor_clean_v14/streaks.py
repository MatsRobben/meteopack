"""Interference / beam-streak detector for old-chain RTCOR (v8 study, proto/beams).

streak_mask(codes, times) -> bool (T, 765, 700): echo pixels to remove as interference streaks.
Meant to be OR-ed into another cleaner, so it flags only thin radial rays in dry surroundings,
never echo inside rain. See NOTES.md for numbers.

Interference shows up as a ray along one azimuth of an NL radar (Den Helder, Herwijnen): 1-2
azimuth bins wide, far along range, continuous or broken into dashes. Per NL radar, each pixel
is binned into a polar cell (0.5 deg x 2 km, out to 420 km) by a precomputed index.
  fill   A = share of valid cells with echo within 22 km along range;
  core   max of A over rows a-1..a+1;
  side   max over left/right of the mean of A over rows 1.5-7 deg away (valid rays only).

1. Transient rays: core >= 0.25, side <= 0.04 in frame t and in t-2..t+2 (motion guard: a
   moving radial band reappears beside the ray). Runs >= 40 km, <= 2.5 deg, isolated (no other
   run within 10 deg: rejects striped far-range sectors). Novel: the ray is mostly absent from
   its own cells in at least one of t-2..t+2. At least 30 km of echo along the run.
2. (off by default, use_recur) thin rays recurring over +-8 frames.
3. Known interference azimuths: azmap.npz (build_azmap.py) is the thin-ray candidate rate per
   cell over 1200 old-chain frames (TEST event days excluded); azimuth rows that stand out
   against their neighbours. In frame t, echo on those cells is flagged where the frame's side
   fill <= 0.15, on thin objects (no pixel > 1.5 px from dry) lying mostly on the ray, with
   >= 14 km of echo along the ray.
Inputs: RTCOR window, NL radar positions, the fixed history map. ~0.015-0.02 s/frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import assets as _assets

# proto/fast: thread count left to the caller (bench: one core)
MISSING = 65534
H, W = 765, 700
# radar sites (row, col) from the v5 assets: Den Helder, Herwijnen, Jabbeke, Wideumont, Emden,
# Essen, Neuheilenbach, Helchteren
ALL_SITES = ((331.495, 333.220), (458.153, 368.942), (543.853, 224.030), (679.498, 416.767),
             (247.800, 460.777), (494.089, 505.969), (648.272, 492.970), (545.410, 396.559))
NL_SITES = ALL_SITES[:2]
NAZ, DR, NR = 720, 2, 210                              # 0.5 deg x 2 km cells, 420 km
_IDX = {}
_AZMAP = {}
_assets.register_cache(_AZMAP.clear)


def azmap(min_rate=0.003, contrast=2.5, row_contrast=2.5, row_excess=0.0015, win=11, side=(3, 10)):
    """Fixed per-site mask (NAZ, NR) of cells on recurring interference rays, from azmap.npz
    (build_azmap.py: thin-ray candidate rate over 1200 old-chain frames, TEST days excluded).
    Rows: azimuths whose range-mean candidate rate is >= row_contrast x and >= row_excess above
    the rows 1.5-5 deg beside them. Cells on those rows (+-1): smoothed rate >= min_rate and
    >= contrast x the side rows."""
    key = (min_rate, contrast, row_contrast, row_excess, win, side)
    if key not in _AZMAP:
        z = _assets.group("azmap")
        out = []
        for s in range(len(z["cand"])):
            R = (z["cand"][s] / np.maximum(z["nval"][s], 1)).astype(np.float32)
            prof = R[:, 8:].mean(1, keepdims=True)
            pl, pr = _rowmean(prof, *side)
            nb = np.maximum(pl, pr)[:, 0]
            rows = (prof[:, 0] >= row_contrast * nb) & (prof[:, 0] - nb >= row_excess)
            rows = np.convolve(rows.astype(float), [1, 1, 1], "same") > 0
            Rs = cv2.blur(R, (win, 1), borderType=cv2.BORDER_CONSTANT)
            l, r = _rowmean(Rs, *side)
            m = (Rs >= min_rate) & (Rs >= contrast * np.maximum(l, r)) & rows[:, None]
            out.append(cv2.dilate(m.astype(np.uint8), np.ones((3, 1), np.uint8)) > 0)
        _AZMAP[key] = out
    return _AZMAP[key]


_KCC = {}


def _known_cc(known, si):
    """connected rays of the fixed known-azimuth mask (cached: the mask is fixed)."""
    key = (id(known[si]), si)
    if key not in _KCC:
        _KCC[key] = cv2.connectedComponents(known[si].astype(np.uint8), connectivity=8)
    return _KCC[key]


def _index(center):
    """Polar cell index (a * NR + r) of every pixel, -1 beyond the last range bin."""
    if center not in _IDX:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        dy, dx = yy - center[0], xx - center[1]
        a = ((np.degrees(np.arctan2(dy, dx)) % 360.0) * NAZ / 360.0 + 0.5).astype(np.int64) % NAZ
        r = (np.hypot(dx, dy) / DR).astype(np.int64)
        _IDX[center] = np.where(r < NR, a * NR + r, -1).ravel()
    return _IDX[center]


def _rowmean(X, lo, hi):
    """(left, right) mean of X over rows a-hi..a-lo and a+lo..a+hi, wrapping in azimuth
    (proto/fast: one row box filter on the wrapped array instead of a float32 cumsum; same values)."""
    X32 = np.ascontiguousarray(X, np.float32)
    n, k = X32.shape[0], hi - lo + 1
    Xp = np.concatenate([X32[-hi:], X32, X32[:hi]], 0)
    B = cv2.boxFilter(Xp, -1, (1, k), anchor=(0, 0), normalize=False, borderType=cv2.BORDER_CONSTANT)
    return B[0:n] / np.float32(k), B[hi + lo: hi + lo + n] / np.float32(k)


class _Site:
    def __init__(self, center, win, side):
        self.idx = _index(center)
        self.inside = self.idx >= 0
        self.win, self.side = win, side
        self.vkey = None

    def valid(self, v):
        """Polar validity and side validity for a radar mask (cached while the mask is unchanged)."""
        if self.vkey is None or not np.array_equal(v, self.vkey):
            vc = np.zeros(NAZ * NR, np.float32)
            vc[self.idx[np.flatnonzero(v & self.inside)]] = 1.0
            vc = vc.reshape(NAZ, NR)
            sv = cv2.blur(vc, (self.win, 1), borderType=cv2.BORDER_CONSTANT)
            okv = (sv >= 0.3).astype(np.float32)
            lV, rV = _rowmean(okv, *self.side)
            self.vkey = v
            self.v = (vc, sv, okv, lV >= 0.75, rV >= 0.75, np.maximum(lV, 1e-6), np.maximum(rV, 1e-6))
        return self.v

    def occupancy(self, wet_t, v):
        """wet_t: bool image (flat) or int array of wet pixel indices (proto/fast: computed once per frame)."""
        o = np.zeros(NAZ * NR, np.float32)
        wp = wet_t if wet_t.dtype != bool else np.flatnonzero(wet_t)
        ci = self.idx[wp]
        o[ci[ci >= 0]] = 1.0
        return o.reshape(NAZ, NR) * self.valid(v)[0]

    def core_side(self, wet_t, v, o=None):
        vc, sv, okv, lok, rok, lV, rV = self.valid(v)
        if o is None:
            o = self.occupancy(wet_t, v)
        sw = cv2.blur(o, (self.win, 1), borderType=cv2.BORDER_CONSTANT)
        A = np.where(okv > 0, sw / np.maximum(sv, 1e-6), 0).astype(np.float32)
        lA, rA = _rowmean(A, *self.side)
        side = np.maximum(np.where(lok, lA / lV, 1.0), np.where(rok, rA / rV, 1.0))
        core = np.maximum(A, np.maximum(np.roll(A, 1, 0), np.roll(A, -1, 0)))
        return core, side, vc, o


def _runs(cand, min_run, max_rows, isolate):
    """bool (NAZ, NR): long thin isolated runs along range in a candidate map, +-1 row."""
    n, lab, st, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None
    ok = np.zeros(n, bool)
    ok[1:] = (st[1:, cv2.CC_STAT_WIDTH] >= min_run) & (st[1:, cv2.CC_STAT_HEIGHT] <= max_rows)
    ids = np.flatnonzero(ok)
    for i in ids:
        for j in ids:
            if i == j:
                continue
            dy = abs(int(st[j, cv2.CC_STAT_TOP]) - int(st[i, cv2.CC_STAT_TOP])) % NAZ
            x0 = max(st[i, cv2.CC_STAT_LEFT], st[j, cv2.CC_STAT_LEFT])
            x1 = min(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH], st[j, cv2.CC_STAT_LEFT] + st[j, cv2.CC_STAT_WIDTH])
            if min(dy, NAZ - dy) <= isolate and x1 > x0:
                ok[i] = False
                break
    if not ok.any():
        return None
    return cv2.dilate(ok[lab].astype(np.uint8), np.ones((3, 1), np.uint8)) > 0


def _transient(cs, t, T, core, sd, vc, o, core_min, side_max, time_ks, min_run, max_rows, isolate,
               novel_max, echo_min_km):
    """Detector 1: transient long rays, dry around in space and time, novel, enough echo."""
    cand = (core >= core_min) & (sd <= side_max) & (vc > 0)
    if not cand.any():
        return None
    for k in time_ks:
        u = t + k
        if 0 <= u < T and cs[u] is not None:
            cand &= cs[u][1] <= side_max
    if not cand.any():
        return None
    flag = _runs(cand, min_run, max_rows, isolate)
    if flag is None:
        return None
    n, lab = cv2.connectedComponents(flag.astype(np.uint8), connectivity=8)
    e = (lab * (o > 0)).ravel()
    ne = np.bincount(e, minlength=n).astype(float)
    keep = np.ones(n, bool)
    if novel_max < 1:
        ov = np.full(n, np.inf)
        for k in time_ks:
            u = t + k
            if not 0 <= u < T:
                continue
            if cs[u] is None:
                ov[:] = 0
                break
            du = cv2.dilate(cs[u][3], np.ones((3, 5), np.uint8)).ravel() > 0
            ov = np.minimum(ov, np.bincount(e, weights=du, minlength=n) / np.maximum(ne, 1))
        keep &= ov <= novel_max
    if echo_min_km > 0:
        rb = np.zeros((n, NR), bool)
        yy, xx = np.nonzero(lab * (o > 0))
        rb[lab[yy, xx], xx] = True
        keep &= rb.sum(1) * DR >= echo_min_km
    keep[0] = False
    flag = keep[lab] & flag
    return flag if flag.any() else None


def _thin_only(fr, wet_t, st, thin, share, margin=40):
    """Keep flagged polar cells only under thin echo objects (no pixel farther than `thin` px
    from dry) lying mostly (>= share of their pixels) on flagged cells: dashes, not rain cells.
    Works on the bounding box of the flagged echo (+ margin) only."""
    w = np.flatnonzero(wet_t & st.inside)
    hitpx = w[fr.ravel()[st.idx[w]]]
    if not len(hitpx):
        return None
    ys, xs = hitpx // W, hitpx % W
    y0, y1 = max(ys.min() - margin, 0), min(ys.max() + margin + 1, H)
    x0, x1 = max(xs.min() - margin, 0), min(xs.max() + margin + 1, W)
    img = wet_t.reshape(H, W)[y0:y1, x0:x1].astype(np.uint8)
    on = np.zeros(img.shape, np.uint8)
    on[ys - y0, xs - x0] = 1
    n, lab = cv2.connectedComponents(img, connectivity=8)
    edt = cv2.distanceTransform(np.pad(img, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    l = lab.ravel()
    size = np.bincount(l, minlength=n)
    hit = np.bincount(l, weights=on.ravel(), minlength=n)
    thick = np.bincount(l, weights=(edt > thin).ravel(), minlength=n)
    good = (hit >= share * np.maximum(size, 1)) & (thick == 0) & (hit > 0)
    good[0] = False
    keep = good[lab] & (on > 0)
    if not keep.any():
        return None
    ky, kx = np.nonzero(keep)
    frk = np.zeros(NAZ * NR, bool)
    frk[st.idx[(ky + y0) * W + kx + x0]] = True
    return frk.reshape(NAZ, NR)
