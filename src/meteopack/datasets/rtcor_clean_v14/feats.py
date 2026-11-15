"""proto/fast: fast per-object features for the RTCOR old-chain clutter classifier.

Same feature set and meaning as proto/objgbm3/cleaner.py (frame_features), computed faster:
- objects with cv2.connectedComponents on the dilated wet mask (same objects as the bench/labels);
- every per-object mean is ONE sparse (objects x pixels) product with a stack of per-pixel columns;
  maxima / bounding boxes by one sorted reduceat;
- fixed maps are gathered once per frame as one (pixels x k) row gather;
- temporal features sample the neighbour frames only at the object's pixels, moved along the flow
  (point sampling of the cached dilated masks) instead of remapping / dilating whole frames;
- the moved pixel positions are kept, so the +-10 min score smoothing reuses them (mapping computed once,
  for any number of models);
- echo persistence over +-30 min: one uint8 count image per frame, sampled at the object (and +-4 px along /
  across the flow), box filter on the count image.
The flow is used on its 4 km grid (nearest), not upsampled. Codes >= 65534 are never touched here.
"""
from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
import scipy.sparse as sp

from . import assets as _assets

MISSING, OUTSIDE = 65534, 65535
H, WD = 765, 700
HW = H * WD
K3 = np.ones((3, 3), np.uint8)
DS = 4
NY4, NX4 = H // DS, WD // DS
NR, NSEG = 320, 16

LUTX = np.log(np.maximum(np.arange(65536, dtype=np.float32), 1))
LUTX[0] = -1.0
LUTX[MISSING:] = -1.0

FEATS = ["logsize", "mean_x", "max_x", "std_x", "f1", "f12", "peak_ratio", "rough", "interior", "maxdepth",
         "elong", "radial", "d_nl", "d_any", "sea", "static", "wet20", "wet50", "frame_wet", "d_out",
         "e1", "e2", "l1", "l2", "lmin1", "lmin2", "lme1", "p30", "flick", "adiff", "speed", "nb_rain1",
         "radcos", "width", "fill", "hot", "jump_max", "core_corr", "d_big", "scene_small", "flowcos",
         "pmax", "pcontrast", "ptail",
         "sq_max", "sq_mean", "sq_logf", "sq_share", "pol_azw", "pol_rlen", "pol_aspect", "spk20", "spk40",
         "ray_rec", "ray_now"]
FI = {f: i for i, f in enumerate(FEATS)}


# ------------------------------------------------------------------ fixed maps
@lru_cache(maxsize=1)
def fixed():
    return _assets.group("fixed")


FIXCOLS = ("d_nl", "d_any", "sea", "static", "sqr", "sq_logf", "caz", "saz")


@lru_cache(maxsize=1)
def fixed_px():
    """flat float32 fixed maps (gathered at the object pixels) and the territory (int64)."""
    fx = fixed()
    cols = [fx["d_nl"], fx["d_any"], fx["sea"], fx["static"], fx["sq_ratio"], fx["sq_logf"],
            np.cos(fx["az"]), np.sin(fx["az"])]
    return {nm: np.ascontiguousarray(np.asarray(c, np.float32).ravel()) for nm, c in zip(FIXCOLS, cols)} | \
        {"terr": fx["terr"].ravel().astype(np.int64)}


@lru_cache(maxsize=1)
def ray_index():
    fx = fixed()
    azb = np.floor(np.degrees(fx["az"]) % 360).astype(np.int64) % 360
    rb = np.minimum(fx["d_nl"] // (320 // NSEG), NSEG - 1).astype(np.int64)
    return ((fx["terr"].astype(np.int64) * 360 + azb) * NSEG + rb).ravel()


@lru_cache(maxsize=1)
def ray_cells():
    """polar cell of every pixel (ray_index; pixels beyond 320 km -> extra last bin) and pixels per cell."""
    idx = np.where(fixed()["d_nl"].ravel() < NR, ray_index(), 2 * 360 * NSEG)
    cnt = np.maximum(np.bincount(idx, minlength=2 * 360 * NSEG + 1), 1).astype(np.float32)
    return idx, cnt


@lru_cache(maxsize=1)
def terr_u8():
    t = fixed()["terr"]
    return (t == 0).astype(np.uint8), (t == 1).astype(np.uint8)


def quarter(a):
    return cv2.resize(np.ascontiguousarray(a[: NY4 * DS, : NX4 * DS], np.float32), (NX4, NY4), interpolation=cv2.INTER_AREA)


def flow_prep(c):
    """4 km mean rate in dB (-15 dB floor) scaled to uint8: the Farneback input (as objgbm3)."""
    c32 = cv2.threshold(c.astype(np.float32), MISSING - 0.5, 0, cv2.THRESH_TOZERO_INV)[1]   # mask -> 0
    a = quarter(c32)
    db = 10 * np.log10(np.maximum(a * np.float32(0.12), 0.03))
    return np.clip((db + 15) * 6, 0, 255).astype(np.uint8)


def flow4(p0, p1):
    """Farneback motion (dx, dy) in px per frame step on the 4 km grid, (NY4, NX4, 2), from prepared frames."""
    return cv2.calcOpticalFlowFarneback(p0, p1, None, 0.5, 3, 13, 3, 5, 1.1, 0) * DS


def _side(F, lo, hi, op):
    """op(mean of rows a-hi..a-lo, mean of rows a+lo..a+hi), wrapping in azimuth (one row box filter on the
    wrapped array; same as the objgbm3 sum of np.roll)."""
    F32 = np.ascontiguousarray(F, np.float32)
    n, k = F32.shape[0], hi - lo + 1
    Fp = np.concatenate([F32[-hi:], F32, F32[:hi]], 0)
    B = cv2.boxFilter(Fp, -1, (1, k), anchor=(0, 0), normalize=False, borderType=cv2.BORDER_CONSTANT)
    return op(B[0:n], B[hi + lo: hi + lo + n]) / np.float32(k)


@lru_cache(maxsize=1)
def polar_maps():
    """per NL radar: flat source pixel (+1; 0 = outside the image) of every (360 x NR) polar cell, i.e. the
    forward cv2.warpPolar (WARP_POLAR_LINEAR, INTER_NEAREST, maxRadius NR) map computed in numpy. cv2.warpPolar
    itself is not used: with OpenCV 5 it returned garbage now and then (bool views, float images)."""
    out = []
    phi = np.arange(360) * (2 * np.pi / 360)
    rho = np.arange(NR) * (NR / NR)
    for r0, c0 in fixed()["nl_rc"]:
        x = np.rint(c0 + rho[None, :] * np.cos(phi[:, None])).astype(np.int64)
        y = np.rint(r0 + rho[None, :] * np.sin(phi[:, None])).astype(np.int64)
        ok = (x >= 0) & (x < WD) & (y >= 0) & (y < H)
        out.append(np.where(ok, y * WD + x + 1, 0))
    return out


@lru_cache(maxsize=1)
def polar_inverse():
    """per NL radar: polar cell (a * NR + r) of every pixel (inverse warpPolar, nearest), -1 beyond NR."""
    yy, xx = np.mgrid[0:H, 0:WD].astype(np.float64)
    out = []
    for r0, c0 in fixed()["nl_rc"]:
        r = np.hypot(xx - c0, yy - r0)
        a = np.rint(np.degrees(np.arctan2(yy - r0, xx - c0)) % 360).astype(np.int64) % 360
        j = np.rint(r).astype(np.int64)
        out.append(np.where(j < NR, a * NR + j, -1).ravel())
    return out


# ------------------------------------------------------------------ window cache
class Win:
    """Per-window cache of per-frame quantities (each computed once per frame)."""

    def __init__(self, codes):
        self.codes = codes
        self.T = len(codes)
        self._c = {}            # (name, u) -> value
        self._dout = None

    def _get(self, name, u, fn):
        k = (name, u)
        v = self._c.get(k)
        if v is None:
            v = fn(u)
            self._c[k] = v
        return v

    def evict(self, t):
        for k in [k for k in self._c if k[1] < t and not (k[0] == "E" and k[1] >= t - 6)]:
            del self._c[k]

    def ok(self, u):
        if not 0 <= u < self.T:
            return False
        return self._get("ok", u, lambda u: bool(np.count_nonzero(self.codes[u] < MISSING) > 0.3 * HW))

    def nvalid(self, u):
        return self._get("nv", u, lambda u: int(np.count_nonzero(self.codes[u] < MISSING)))

    def W(self, u):
        """flat uint8 wet mask with one trailing 0 (index -1 = outside the image = dry)."""
        def f(u):
            c = self.codes[u]
            out = np.zeros(HW + 1, np.uint8)
            out[:HW] = ((c > 0) & (c < MISSING)).ravel()
            return out
        return self._get("W", u, f)

    def D(self, u):
        def f(u):
            out = np.zeros(HW + 1, np.uint8)
            cv2.dilate(self.W(u)[:HW].reshape(H, WD), K3, dst=out[:HW].reshape(H, WD))
            return out
        return self._get("D", u, f)

    def C(self, u):
        """flat codes with a trailing 0."""
        def f(u):
            out = np.zeros(HW + 1, np.uint16)
            out[:HW] = self.codes[u].ravel()
            return out
        return self._get("C", u, f)

    def F4(self, t):
        """motion per 5 min for t -> t+1 on the 4 km grid: half the flow t -> t+2 (old-chain alternate-frame
        repeats), as objgbm3."""
        def f(t):
            if t + 2 < self.T:
                fl = 0.5 * flow4(self.prep(t), self.prep(t + 2))
            elif t >= 1 and t + 1 < self.T:
                fl = 0.5 * flow4(self.prep(t - 1), self.prep(t + 1))
            elif t >= 2:                                   # last frame (flow output only)
                fl = 0.5 * flow4(self.prep(t - 2), self.prep(t))
            else:
                fl = flow4(self.prep(t), self.prep(min(t + 1, self.T - 1)))
            return fl
        return self._get("F", t, f)

    def disp(self, t, k):
        """integer displacement (dy, dx) per 4 km cell (flat) moving frame t to frame t+k (k*F(t) for k > 0,
        k*F(t-1) for k < 0, rounded): rint(y + k v) = y + rint(k v) for integer pixel positions."""
        def f(key):
            F = (self.F4(t) if k > 0 else self.F4(t - 1)).reshape(-1, 2)
            return np.rint(k * F[:, 1]).astype(np.int32), np.rint(k * F[:, 0]).astype(np.int32)
        return self._get(("disp", k), t, f)

    def prep(self, u):
        return self._get("prep", u, lambda u: flow_prep(self.codes[u]))

    def Q5(self, u):
        """20 km box mean of the dilated echo on the 4 km grid, flat with trailing 0."""
        def f(u):
            q = cv2.boxFilter(quarter(self.D(u)[:HW].reshape(H, WD)), -1, (5, 5))
            return np.concatenate([q.ravel(), np.zeros(1, np.float32)])
        return self._get("Q5", u, f)

    def dup(self, u):
        def f(u):
            out = np.zeros(2, bool)
            if 1 <= u < self.T:
                a, b = self.codes[u - 1], self.codes[u]
                w = (a > 0) & (a < MISSING)
                terr = fixed()["terr"]
                nw = np.bincount(terr[w], minlength=2)
                ns = np.bincount(terr[w & (a == b)], minlength=2)
                out = (nw >= 30) & (ns >= 0.8 * nw)
            return out
        return self._get("dup", u, f)

    def depth(self, u):
        """distance to the nearest dry pixel (outside the image counts as dry), (H, W) float32."""
        return self._get("dep", u, lambda u: cv2.distanceTransform(np.pad(self.W(u)[:HW].reshape(H, WD), 1), cv2.DIST_L2, 3)[1:-1, 1:-1])

    def wetpx(self, u):
        return self._get("wp", u, lambda u: np.flatnonzero(self.W(u)[:HW]))

    def rayE(self, u):
        """(2, 360, 16) narrow-ray excess per NL radar territory (1 deg x 20 km cells, pixels binned by their
        polar cell around the nearest NL radar): wet share of the cell minus the wetter of the two side sectors
        3-8 deg away, clipped at 0 (objgbm3 used warpPolar sampling; same quantity)."""
        def f(u):
            idx, cnt = ray_cells()
            wp = self.wetpx(u)
            if not len(wp):
                return np.zeros((2, 360, NSEG), np.float32)
            A = (np.bincount(idx[wp], minlength=len(cnt))[:-1] / cnt[:-1]).astype(np.float32).reshape(2, 360, NSEG)
            out = np.empty_like(A)
            for i in range(2):
                out[i] = np.clip(A[i] - np.maximum(_side(A[i], 3, 8, np.maximum), 0), 0, None)
            return out
        return self._get("E", u, f)

    def dout(self, t):
        if self._dout is None:
            self._dout = cv2.distanceTransform((self.codes[t] != OUTSIDE).astype(np.uint8), cv2.DIST_L2, 3).ravel()
        return self._dout


# ------------------------------------------------------------------ polar spike test (objgbm2/3, unchanged)
_PV = {}


def spikes(wet, valid, side=(3, 8), core_min=0.6, side_max=0.1, runs=(20,)):
    """objgbm2/3 polar spike test (proto/lagrange): pixels of narrow isolated radial bands (>= run km of echo
    within +-1 deg, valid rays 3-8 deg to both sides nearly dry). wet: flat uint8 with a leading 0 (index 0 =
    outside), valid: bool image. Returns flat bool arrays (length H*W) per run length."""
    outs = [np.zeros((H, WD), np.uint8) for _ in runs]
    v8 = valid.view(np.uint8)
    k = np.ones((1, 9), np.float32)
    if _PV.get("valid") is None or not np.array_equal(_PV["valid"], valid):
        _PV.clear(); _PV["valid"] = valid.copy()
        for i, (r0, c0) in enumerate(fixed()["nl_rc"]):
            pv = np.concatenate([[0], v8.ravel()])[polar_maps()[i]].astype(np.float32)
            sv_ = cv2.filter2D(pv, -1, k, borderType=cv2.BORDER_CONSTANT)
            V = (sv_ > 4).astype(np.float32)
            _PV[i] = (pv, sv_, V, _side(V, *side, np.minimum))
    for i, (r0, c0) in enumerate(fixed()["nl_rc"]):
        pw = wet[polar_maps()[i]].astype(np.float32)
        if pw.sum() < 30:
            continue
        pv, sv_, V, sv = _PV[i]
        F = cv2.filter2D(pw, -1, k, borderType=cv2.BORDER_CONSTANT) / np.maximum(sv_, 1)
        core = np.maximum(F, np.maximum(np.roll(F, 1, 0), np.roll(F, -1, 0)))
        sf = _side(F * V, *side, np.maximum) / np.maximum(sv, 1e-3)
        cand = ((core >= core_min) & (sf <= side_max) & (sv >= 0.75) & (pv > 0)).astype(np.uint8)
        if not cand.any():
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, connectivity=4)
        for out, mr in zip(outs, runs):
            good = np.r_[False, (st[1:, cv2.CC_STAT_WIDTH] >= mr) & (st[1:, cv2.CC_STAT_HEIGHT] <= 5)]
            ids = np.flatnonzero(good)
            if len(ids) > 1:
                y0, x0, x1 = st[ids, cv2.CC_STAT_TOP], st[ids, cv2.CC_STAT_LEFT], st[ids, cv2.CC_STAT_LEFT] + st[ids, cv2.CC_STAT_WIDTH]
                dy = np.abs(y0[:, None] - y0[None, :]) % 360
                near = (np.minimum(dy, 360 - dy) <= 15) & (np.minimum(x1[:, None], x1[None, :]) > np.maximum(x0[:, None], x0[None, :]))
                np.fill_diagonal(near, False)
                good[ids[near.any(1)]] = False
            m = good[lab].astype(np.uint8)
            if m.any():
                m = cv2.dilate(m, np.ones((3, 3), np.uint8))
                inv = polar_inverse()[i]
                out |= np.where(inv >= 0, m.ravel()[np.maximum(inv, 0)], 0).astype(np.uint8).reshape(H, WD)
    return [o.ravel().astype(bool) & (wet[1:] > 0) for o in outs]


# ------------------------------------------------------------------ per-frame objects + features
class Frame:
    """Objects of one frame: lab (H, W int32), n, opx (flat wet px), oid (object index per px), size,
    mp {k: moved flat positions of opx in frame t+k, -1 outside}, X (n x len(FEATS))."""
    __slots__ = ("A", "X", "lab", "mp", "n", "oid", "opx", "size", "t")


def _moved(win, t, k, yy, xx, p4):
    """flat index in frame t+k of the object pixels moved along the flow (nearest 4 km flow), -1 outside."""
    dy, dx = win.disp(t, k)
    yi = yy + dy[p4]
    xi = xx + dx[p4]
    inb = (yi >= 0) & (yi < H) & (xi >= 0) & (xi < WD)
    return np.where(inb, yi * WD + xi, -1)


ELL5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
SPLIT_MIN_RES = 12        # split v2: thin pieces < 12 px stay attached (v1 = 0 split every fringe off)


def _cc(mask):
    """objects of a uint8 mask: connected after closing 1-px gaps (dilate 3x3, 8-connected), on mask pixels."""
    n1, lab = cv2.connectedComponents(cv2.dilate(mask, K3), connectivity=8, ltype=cv2.CV_32S)
    lab[mask == 0] = 0
    return lab, n1 - 1


def objects_of(win, t, split=False, core_depth=1.8, min_res=None):
    """Objects of frame t. split=False: the bench/label objects (echo connected after closing 1-px gaps).
    split=True (sub-objects): bodies = echo within 2 px of a core pixel (>= 2 px from dry, i.e. parts at
    least 3 px wide), and the remaining thin echo (rays, speckle chains, thin protrusions > 2 px from any body)
    as separate objects; each set connected as above. Thin structures merged into rain (interference rays,
    clutter fringes) are scored on their own; objects without a 3-px-wide part are unchanged."""
    w2 = win.W(t)[:HW].reshape(H, WD)
    if not split:
        n1, lab = cv2.connectedComponents(win.D(t)[:HW].reshape(H, WD), connectivity=8, ltype=cv2.CV_32S)
        lab[w2 == 0] = 0
        return lab, n1 - 1
    core = (win.depth(t) >= core_depth).astype(np.uint8)
    body = cv2.dilate(core, ELL5) & w2
    res = w2 & (1 - body)
    lb, nb = _cc(body)
    lr, nr = _cc(res)
    min_res = SPLIT_MIN_RES if min_res is None else min_res
    if min_res and nr:
        # thin pieces smaller than min_res px stay with the body they touch (fringes are not split off)
        rs = np.bincount(lr.ravel(), minlength=nr + 1)
        small = rs < min_res
        small[0] = False
        if small.any():
            near = cv2.dilate(lb.astype(np.float32), ELL5).astype(np.int32)   # a touching body label (max)
            body_of = np.zeros(nr + 1, np.int32)
            m_ = (lr > 0) & (near > 0)
            np.maximum.at(body_of, lr[m_], near[m_])
            join = small & (body_of > 0)                    # whole small piece joins its body
            sm = join[lr]
            lb = np.where(sm, body_of[lr], lb)
            lr = np.where(sm, 0, lr)
            keep = np.unique(lr[lr > 0])
            remap = np.zeros(nr + 1, np.int32); remap[keep] = np.arange(1, len(keep) + 1)
            lr = remap[lr]; nr = len(keep)
    lab = np.where(lr > 0, lr + nb, lb).astype(np.int32)
    return lab, nb + nr


def frame_features(win: Win, t: int, need=None, split=False) -> Frame:
    """Objects of frame t and their features (row i = object i+1). `need`: optional set of feature names;
    expensive groups not needed are skipped (their columns are NaN). split: sub-objects (objects_of)."""
    need = set(FEATS) if need is None else set(need)
    fr = Frame(); fr.t = t; fr.mp = {}
    w = win.W(t)
    lab, n = objects_of(win, t, split)
    fr.lab, fr.n = lab, n
    if n == 0:
        fr.opx = fr.oid = np.zeros(0, np.int64); fr.size = np.zeros(0); fr.X = np.zeros((0, len(FEATS)), np.float32)
        return fr
    opx = win.wetpx(t)
    oid = lab.ravel()[opx] - 1
    order = np.argsort(oid, kind="stable")          # pixels grouped by object: means/maxima by reduceat
    opx, oid = opx[order], oid[order]
    N = len(opx)
    size_i = np.bincount(oid, minlength=n)
    size = size_i.astype(np.float64)
    starts = np.zeros(n, np.int64); np.cumsum(size_i[:-1], out=starts[1:])
    fr.opx, fr.oid, fr.size = opx, oid, size
    yy, xx = np.divmod(opx, WD)
    R32 = np.empty((64, N), np.float32)             # per-pixel columns (float32) -> per-object means
    names = []

    def add(nm, v):
        R32[len(names)] = v; names.append(nm)

    C = win.C(t)
    c = C[opx]
    x = LUTX[c]
    add("x", x); add("x2", x * x); add("c1", c <= 1); add("c2", c <= 2); add("c", c)
    # texture: 4-neighbours (outside the image counts as dry)
    jmp = np.zeros(N, np.float32); nb = np.zeros(N, np.float32); nbs = np.zeros(N, np.float32)
    for off, okn in ((1, xx < WD - 1), (-1, xx > 0), (WD, yy < H - 1), (-WD, yy > 0)):
        x2 = LUTX[C[np.where(okn, opx + off, HW)]]      # C[HW] = 0 -> dry
        w2 = x2 >= 0
        jmp += np.where(w2, np.abs(x - x2), 0); nb += w2; nbs += np.where(w2, x2, 0)
    add("rough", jmp / np.maximum(nb, 1)); add("interior", nb == 4)
    nbmean = np.where(nb > 0, nbs / np.maximum(nb, 1), np.float32(-1.0))
    add("hot", (x - nbmean > 1.5) & (c >= 10))
    dep = win.depth(t).ravel()[opx]
    add("dep", dep); add("dep2", dep * dep); add("depx", dep * x)
    FP = fixed_px()
    for nm in FIXCOLS:
        add(nm, FP[nm][opx])
    dnl, sqr = R32[names.index("d_nl")], R32[names.index("sqr")]
    add("dnl2", dnl * dnl); add("sqshare", sqr > 0.5)
    add("d_out", win.dout(t)[opx])
    terr = FP["terr"][opx]
    p4 = np.minimum(yy // DS, NY4 - 1) * NX4 + np.minimum(xx // DS, NX4 - 1)
    if "wet20" in need or "wet50" in need:
        q = quarter(w[:HW].reshape(H, WD))
        add("wet20", cv2.boxFilter(q, -1, (11, 11), borderType=cv2.BORDER_CONSTANT).ravel()[p4])
        add("wet50", cv2.boxFilter(q, -1, (25, 25), borderType=cv2.BORDER_CONSTANT).ravel()[p4])
    # ---- temporal: neighbour frames sampled at the object's pixels, in place (e) and moved along the flow (l)
    ok = win.ok
    e, l, cps = {}, {}, {}
    for k in (-2, -1, 1, 2):
        if not ok(t + k):
            continue
        Cu = win.C(t + k)
        add(f"cp{k}", Cu[opx] == c)
        add(f"e{k}", win.D(t + k)[opx])
        if ok(t + 1) if k > 0 else ok(t - 1):
            mp = _moved(win, t, k, yy, xx, p4)
            fr.mp[k] = mp
            add(f"l{k}", win.D(t + k)[mp])
            if abs(k) == 1:
                cu = Cu[mp]
                a_ = (cu > 0) & (cu < MISSING)
                add(f"adn{k}", np.where(a_, np.abs(x - LUTX[cu]), 0)); add(f"add{k}", a_)
                if "nb_rain1" in need:
                    mi = np.where(mp >= 0, mp, 0)
                    m4 = np.where(mp >= 0, np.minimum(mi // WD // DS, NY4 - 1) * NX4 + np.minimum(mi % WD // DS, NX4 - 1), NY4 * NX4)
                    add(f"nbr{k}", win.Q5(t + k)[m4])
    seqk = [k for k in range(-3, 4) if ok(t + k) and (k == 0 or not win.dup(t + k).any())]
    if len(seqk) >= 3:
        S = np.stack([win.W(t + k)[opx] for k in seqk]).astype(np.int8)
        add("flick", np.abs(np.diff(S, axis=0)).sum(0) / np.float32(len(seqk) - 1))
    F4 = win.F4(t) if ok(t + 1) else None
    if F4 is not None:
        fp = F4.reshape(-1, 2)
        add("speed", np.hypot(fp[p4, 0], fp[p4, 1]))
    # ---- echo persistence over +-30 min (per radar territory, skipping frames that repeat their predecessor)
    ks = [k for k in range(-6, 7) if k and ok(t + k)]
    Pc = None
    if ks:
        Pc = np.zeros(HW, np.uint8)
        den = np.zeros(2)
        tm = terr_u8()
        for k in ks:
            fresh = ~win.dup(t + k)
            if k == -1:
                fresh = fresh & ~win.dup(t)
            if fresh.all():
                np.add(Pc, win.W(t + k)[:HW], out=Pc)
            elif fresh.any():
                np.add(Pc, win.W(t + k)[:HW] & tm[int(np.argmax(fresh))].ravel(), out=Pc)
            den += fresh
        dpx = (1.0 / np.maximum(den, 1)).astype(np.float32)[terr]
        pp = Pc[opx] * dpx
        add("p30", pp)
        add("pbox", cv2.boxFilter(Pc.reshape(H, WD), cv2.CV_32F, (11, 11)).ravel()[opx] * dpx)
    if "spk20" in need:
        add("spk20", spikes(np.concatenate([np.zeros(1, np.uint8), w[:HW]]), win.codes[t] < MISSING)[0][opx])
    if "ray_now" in need or "ray_rec" in need:
        pidx = ray_index()[opx]
        us = [t + k for k in range(-12, 13) if k and ok(t + k) and not win.dup(t + k).any()]

        def at_px(E):
            Ed = np.maximum(E, np.maximum(np.roll(E, 1, 1), np.roll(E, -1, 1)))
            return Ed.ravel()[pidx]
        add("ray_now", at_px(win.rayE(t)))
        if us and "ray_rec" in need:
            add("ray_rec", at_px(np.mean([win.rayE(u) for u in us], 0)))
    # ---- per-object means (one reduceat) and maxima / bounding box
    Sm = np.add.reduceat(R32[: len(names)], starts, axis=1) / size.astype(np.float32)
    m = {nm: Sm[i].astype(np.float64) for i, nm in enumerate(names)}
    # second moments in float64, relative to the object's first pixel
    yr = (yy - yy[starts][oid]).astype(np.float64); xr = (xx - xx[starts][oid]).astype(np.float64)
    Mo = np.add.reduceat(np.stack([yr, xr, yr * yr, xr * xr, xr * yr]), starts, axis=1) / size
    mxl = [x, dep, yy, -yy, xx, -xx, sqr, x - nbmean] + ([pp] if Pc is not None else [])
    MX = np.maximum.reduceat(np.stack([np.asarray(v, np.float32) for v in mxl]), starts, axis=1).T.astype(np.float64)
    f = {}
    f["logsize"] = np.log(size)
    mx = m["x"]
    f["mean_x"] = mx; f["max_x"] = MX[:, 0]
    f["std_x"] = np.sqrt(np.maximum(m["x2"] - mx * mx, 0))
    f["f1"] = m["c1"]; f["f12"] = m["c2"]
    f["peak_ratio"] = np.exp(f["max_x"]) / np.maximum(m["c"], 1e-6)
    f["rough"] = m["rough"]; f["interior"] = m["interior"]
    f["maxdepth"] = MX[:, 1]
    my, mxx = Mo[0], Mo[1]
    syy = Mo[2] - my * my + 1 / 12; sxx = Mo[3] - mxx * mxx + 1 / 12; sxy = Mo[4] - my * mxx
    my, mxx = my + yy[starts], mxx + xx[starts]
    tr, det = syy + sxx, syy * sxx - sxy * sxy
    l1 = tr / 2 + np.sqrt(np.maximum(tr * tr / 4 - det, 0)); l2 = np.maximum(tr - l1, 1e-3)
    f["elong"] = np.sqrt(l1 / l2)
    ang = 0.5 * np.arctan2(2 * sxy, sxx - syy)
    cy, cx = np.clip(np.rint(my).astype(int), 0, H - 1), np.clip(np.rint(mxx).astype(int), 0, WD - 1)
    azc = fixed()["az"][cy, cx]
    f["radial"] = np.abs(np.cos(ang - azc)) * (1 - 1 / f["elong"])
    for nm in ("d_nl", "d_any", "sea", "static", "wet20", "wet50", "d_out"):
        f[nm] = m[nm] if nm in m else np.full(n, np.nan)
    f["frame_wet"] = np.full(n, N / max(win.nvalid(t), 1))
    nan = np.full(n, np.nan)

    def objcp(k):
        return m[f"cp{k}"] >= 0.9 if f"cp{k}" in m else None
    for k in (-2, -1, 1, 2):
        cp = objcp(k)
        e[k] = np.where(cp, np.nan, m[f"e{k}"]) if cp is not None else nan
        l[k] = np.where(cp, np.nan, m[f"l{k}"]) if (cp is not None and f"l{k}" in m) else nan
        if abs(k) == 1:
            cps[k] = cp
    f["e1"] = np.nanmean([e[-1], e[1]], 0)
    f["e2"] = np.nanmean([e[-2], e[2]], 0)
    f["l1"] = np.nanmean([l[-1], l[1]], 0)
    f["l2"] = np.nanmean([l[-2], l[2]], 0)
    f["lmin1"] = np.fmin(l[-1], l[1]); f["lmin2"] = np.fmin(l[-2], l[2])
    f["lme1"] = np.nanmean([l[-1], l[1], l[-2], l[2]], 0) - np.nanmean([e[-1], e[1], e[-2], e[2]], 0)
    f["flick"] = m["flick"] if "flick" in m else nan
    ad = [np.where(cps[k], np.nan, m[f"adn{k}"] / np.maximum(m[f"add{k}"], 1e-9) * (m[f"add{k}"] > 0))
          for k in (-1, 1) if f"adn{k}" in m]
    f["adiff"] = np.nanmean(ad, 0) if ad else nan
    if F4 is not None:
        f["speed"] = m["speed"]
        nb_ = [m[f"nbr{k}"] for k in (-1, 1) if f"nbr{k}" in m]
        f["nb_rain1"] = np.mean(nb_, 0) if nb_ else nan
        c4 = np.minimum(cy // DS, NY4 - 1) * NX4 + np.minimum(cx // DS, NX4 - 1)
        vx, vy = fp[c4, 0].astype(np.float64), fp[c4, 1].astype(np.float64)
        f["flowcos"] = np.where(np.hypot(vx, vy) > 0.3, np.abs(np.cos(ang - np.arctan2(vy, vx))), np.nan)
    else:
        f["speed"] = f["nb_rain1"] = f["flowcos"] = nan
    f["radcos"] = np.abs(np.cos(ang - azc))
    f["width"] = size / np.maximum(np.sqrt(12 * l1), 1)
    f["fill"] = size / ((MX[:, 2] + MX[:, 3] + 1) * (MX[:, 4] + MX[:, 5] + 1))
    f["hot"] = m["hot"]
    f["jump_max"] = MX[:, 7]
    md = m["dep"]
    sdd = np.sqrt(np.maximum(m["dep2"] - md * md, 0))
    f["core_corr"] = np.where((sdd > 1e-3) & (f["std_x"] > 1e-3), (m["depx"] - md * mx) / np.maximum(sdd * f["std_x"], 1e-6), 0)
    big = size >= 100
    if big.any() and "d_big" in need:
        bigobj = np.concatenate([[False], big])
        dbig = cv2.distanceTransform((~bigobj[lab]).astype(np.uint8), cv2.DIST_L2, 3).ravel()[opx]
        f["d_big"] = np.minimum(np.add.reduceat(dbig, starts) / size, 300)
    else:
        f["d_big"] = np.full(n, 300.0)
    f["scene_small"] = np.full(n, size[size < 20].sum() / size.sum())
    if Pc is not None:
        f["p30"] = m["p30"]; f["pmax"] = MX[:, 8]
        f["pcontrast"] = m["p30"] - m["pbox"]
        if F4 is not None:
            sp_ = np.hypot(vx, vy)
            ux, uy = vx / np.maximum(sp_, 1e-6), vy / np.maximum(sp_, 1e-6)

            def samp(dx_, dy_):              # rint(y + 4 u) = y + rint(4 u): integer offset per object
                yi = np.clip(yy + np.rint(4 * dy_).astype(np.int64)[oid], 0, H - 1)
                xi = np.clip(xx + np.rint(4 * dx_).astype(np.int64)[oid], 0, WD - 1)
                return Pc[yi * WD + xi]
            tail = (samp(ux, uy).astype(np.int16) + samp(-ux, -uy) - samp(-uy, ux) - samp(uy, -ux)) * (0.5 * dpx)
            f["ptail"] = np.where(sp_ > 0.3, np.add.reduceat(tail, starts) / size, np.nan)
        else:
            f["ptail"] = nan
    else:
        f["p30"] = f["pmax"] = f["pcontrast"] = f["ptail"] = nan
    f["sq_max"] = MX[:, 6]; f["sq_mean"] = m["sqr"]; f["sq_logf"] = m["sq_logf"]; f["sq_share"] = m["sqshare"]
    Rb = np.clip(np.hypot(m["caz"], m["saz"]), 1e-6, 1)
    azw = np.degrees(np.sqrt(-2 * np.log(Rb)) * np.sqrt(12)) + 1e-3
    mr = m["d_nl"]
    rlen = np.sqrt(np.maximum(m["dnl2"] - mr * mr, 0) * 12) + 1
    f["pol_azw"] = azw; f["pol_rlen"] = rlen
    f["pol_aspect"] = rlen / np.maximum(np.radians(azw) * mr, 1)
    f["spk40"] = nan
    for nm in ("spk20", "ray_now", "ray_rec"):
        f[nm] = m[nm] if nm in m else nan
    fr.X = np.stack([np.asarray(f[k], np.float32) for k in FEATS], 1)
    return fr


# ------------------------------------------------------------------ smoothing along the flow
def nb_maps(fr: Frame, frames: dict, ks=(-2, -1, 1, 2), min_cover=0.3, gate=None):
    """For each neighbour frame u = t+k with objects: sparse (n_t x n_u) matrix of the share of each object's
    pixels landing (moved along the flow) on objects of u, restricted to objects covered >= min_cover.
    Model independent: smoothing for any score vector is one matvec per neighbour."""
    out = []
    for k in ks:
        u = fr.t + k
        g = frames.get(u)
        if g is None or g.n == 0 or k not in fr.mp:
            continue
        mp = fr.mp[k]
        lu = g.lab.ravel()[np.maximum(mp, 0)] * (mp >= 0)
        if gate:          # ignore neighbour objects > gate x larger than this object (merged into other echo)
            usz = np.concatenate([[0.0], g.size])[lu]
            lu = np.where(usz > gate * fr.size[fr.oid], 0, lu)
        has = lu > 0
        cnt = np.bincount(fr.oid[has], minlength=fr.n)
        okm = cnt >= min_cover * fr.size
        sel = has & okm[fr.oid]
        oi = fr.oid[sel]
        Mk = sp.coo_matrix((1.0 / cnt[oi], (oi, lu[sel] - 1)), shape=(fr.n, g.n))
        out.append((u, Mk, okm))
    return out


def smooth(p, maps, P, alpha):
    """s = (1 - alpha) p + alpha * mean over covering neighbours of their mean score on the object."""
    if alpha == 0 or not maps:
        return p
    sm = np.zeros(len(p)); cnt = np.zeros(len(p))
    for u, Mk, okm in maps:
        sm += Mk @ P[u]; cnt += okm
    return np.where(cnt > 0, (1 - alpha) * p + alpha * sm / np.maximum(cnt, 1), p)


_assets.register_cache(_PV.clear)
for _f in (fixed, fixed_px, ray_index, ray_cells, terr_u8, polar_maps, polar_inverse):
    _assets.register_cache(_f.cache_clear)
