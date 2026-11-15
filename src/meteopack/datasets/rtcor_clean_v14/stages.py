"""v12 stages on top of the v11 output (see cleaner.py). Every stage takes (ctx, out, **kw) and returns
(out, level) where level is the per-pixel aggr at which the pixel was removed (inf where kept), for diagnostics.

Convention: an element (object, segment, pixel) gets a required level L >= 0 from its evidence; it is removed where
the local aggr > L. So aggr = 0 never removes anything here, and larger aggr removes a superset (monotone).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import quality as QA
from . import texfeat as TF

MISSING = 65534
H, W = 765, 700
DS = 4
HALVES = ((1, slice(0, 380)), (2, slice(380, H)))
NL_SITES = ((331.495, 333.220), (458.153, 368.942))       # Den Helder, Herwijnen (row, col) = proto/fast/streaks
EPS = 1e-3


def pw(x, x1, x2, lo=None):
    """piecewise-linear level: x1 -> 1, x2 -> 2 (x beyond x1 toward x2 is weaker evidence), used to extend a
    stage's aggr <= 1 rule to aggr in (1, 2] without changing it below 1."""
    return 1 + (x1 - x) / (x1 - x2)
_K8 = np.ones((3, 3), np.float32); _K8[1, 1] = 0


class Context:
    def __init__(self, codes, st, A, valid):
        self.codes, self.st, self.A, self.valid = codes, st, A, valid
        self.T = len(codes)
        self.flags = st["flags"]
        ok = np.ones((self.T, H), bool)
        for u in range(self.T):
            for bit, sl in HALVES:
                if self.flags[u] & bit:
                    ok[u, sl] = False
        self.ok = ok                                      # rows usable as evidence (not a frozen copy)
        cov = valid.reshape(self.T, -1).mean(1)
        self.good = cov >= 0.5 * max(cov.max(), 1e-9)     # frames with enough coverage to be evidence

    def flow(self, t, py, px):
        """motion t -> t+1 (px / 5 min) at pixels, nearest 4 km cell."""
        fl, s = self.st["flr"][min(t, self.T - 1)]
        v = fl[np.minimum(py // DS, fl.shape[0] - 1), np.minimum(px // DS, fl.shape[1] - 1)] * (DS * s)
        return v[:, 0].astype(np.float32), v[:, 1].astype(np.float32)

    def flow_full(self, t):
        """(2, H, W) motion t -> t+1 in px per 5 min (nearest 4 km cell)."""
        fl, s = self.st["flr"][min(t, self.T - 1)]
        v = np.repeat(np.repeat(fl * (DS * s), DS, 0), DS, 1)
        pad = np.zeros((H, W, 2), np.float32)
        h, w = min(H, v.shape[0]), min(W, v.shape[1])
        pad[:h, :w] = v[:h, :w]
        if h < H:
            pad[h:] = pad[h - 1:h]
        if w < W:
            pad[:, w:] = pad[:, w - 1:w]
        return np.ascontiguousarray(pad.transpose(2, 0, 1))

    def apx(self, t, py, px):
        a = self.A.px(t)
        return np.full(len(py), a, np.float32) if np.isscalar(a) else a[py, px]


def _dist_to_wet(out, valid):
    """per frame: distance (px) of every pixel to the nearest wet pixel of `out`."""
    return [cv2.distanceTransform((~((out[u] > 0) & valid[u])).astype(np.uint8), cv2.DIST_L2, 3) for u in range(len(out))]


def support_dist(ctx, dist, t, py, px, ks=(1, 2, 3), rk=1.0, euler=True):
    """Per pixel and side (-, +): D_s = min over k of (distance to wet at frame t+-k at the position moved by
    +-k v, and at the same place) - (k-1) rk. NaN where no usable sample on that side (outside, missing data
    within 2 px, frozen half, poor frame)."""
    T = ctx.T
    vx, vy = ctx.flow(t, py, px)
    fy, fx = py.astype(np.float32), px.astype(np.float32)
    D = np.full((2, len(py)), np.nan, np.float32)
    for si, s in enumerate((-1, 1)):
        for k in ks:
            u = t + s * k
            if not (0 <= u < T) or not ctx.good[u]:
                continue
            cands = [(np.rint(fx + s * k * vx), np.rint(fy + s * k * vy))]
            if euler:
                cands.append((fx, fy))
            for xs, ys in cands:
                ins = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
                yi = np.where(ins, ys, 0).astype(np.int64); xi = np.where(ins, xs, 0).astype(np.int64)
                d = dist[u][yi, xi] - (k - 1) * rk
                use = ins & ctx.ok[u][yi] & ctx.vm[u][yi, xi]
                d = np.where(use, d, np.nan)
                D[si] = np.fmin(D[si], d)
    return D


def quality_mask(ctx):
    """quality.py pixel mask (uint8 (T, H, W): 8 DROPOUT, 128 SUPPORT, ...) of the raw window."""
    return QA.analyse(ctx.codes)[1]


def _valid_eroded(valid):
    return [cv2.erode(valid[u].astype(np.uint8), np.ones((5, 5), np.uint8), borderValue=1).astype(bool)
            for u in range(len(valid))]


# ------------------------------------------------------------------ 6. support stage
SUPPORT_DEFAULTS = dict(ks=(1, 2, 3), rk=1.0, r_obj=3.0, s0=20, s1=1500, s2=4000, fmax=0.3, fmax2=0.5, f_free=0.1, size_floor=0.3, floor_smooth=0.9, frag_obj=0.3, code1=0.3,
                        fr_hi=6.0, fr_lo=1.5, fr_min_obj=21, fr_k=(1, 2, 3), fr_blink=0.5, fr_a0=0.3)


def support_stage(ctx, out, *, ks=(1, 2, 3), rk=1.0, r_obj=3.0, s0=20, s1=1500, s2=4000, fmax=0.3, fmax2=0.5, f_free=0.1, size_floor=0.3, floor_smooth=0.9,
                  frag_obj=0.3, code1=0.3, fr_hi=6.0, fr_lo=1.5, fr_min_obj=21, fr_k=(1, 2, 3), fr_blink=0.5, fr_a0=0.3):
    """(a) objects whose pixels find no echo along the motion (or at the same place) at t+-1..3 on EITHER side:
    support share f = best side's share of pixels with echo within r_obj + (k-1) rk. Required level
    max(f/fmax, ls * max(min(f/f_free, 1), size_floor)), ls = log(size/s0)/log(s1/s0): at aggr 1 objects up to s1 px
    with <= fmax supported pixels go; objects with almost no support (f < f_free: burst frames, one-frame fields)
    go at any size once aggr > floor * ls, floor = size_floor (0.3) for fragmented objects (>= frag_obj of their
    px with <= 3 wet neighbours), floor_smooth (0.9) for compact ones (real rain in old-chain blink frames). Objects with max code 1 need aggr > code1 (0.3): the new chain
    also has transient code-1 specks in drizzle.
    Both sides need evidence (a window edge or missing data is no evidence).
    (b) fringe pixels of objects >= fr_min_obj px: ds = distance to the nearest echo along the motion (or in place)
    at t-+1..3, the smaller side (both sides must be unsupported): level fr_a0 + (1 - fr_a0) clip((fr_hi - ds) /
    (fr_hi - fr_lo), 0) (ds >= fr_hi: 0.3; ds = fr_lo: 1; ds <= 1 px: never); frames where the unsupported share of
    large-object pixels at the local threshold exceeds fr_blink are left alone (blink / burst frames: their large
    objects are not judged pixel by pixel)."""
    T = ctx.T
    if not hasattr(ctx, "vm"):
        ctx.vm = _valid_eroded(ctx.valid)
        qm = quality_mask(ctx)
        if qm is not None:
            # review R1: missing / dropped data in a neighbour frame is no evidence. proto/quality marks DROPOUT
            # (dry at u where the advected neighbours of u have rain: far-range contribution dropped) and SUPPORT
            # (an apparently one-frame area at t that IS present at t+-2: real rain whose neighbours dropped it)
            for u in range(T):
                ctx.vm[u] &= (qm[u] & 8) == 0
            ctx.qsupport = (qm & 128) > 0
    dist = _dist_to_wet(out, ctx.valid)
    res = out.copy()
    level = np.full(out.shape, np.inf, np.float32)
    for t in range(T):
        wt = (out[t] > 0) & ctx.valid[t]
        if not wt.any() or not ctx.good[t]:
            continue
        n, lab, sts, _ = cv2.connectedComponentsWithStats(wt.view(np.uint8), connectivity=8)
        size = sts[:, cv2.CC_STAT_AREA].astype(np.int64); size[0] = 0
        py, px = np.nonzero(wt)
        Lb = lab[py, px]
        live = ctx.ok[t][py]
        a = ctx.apx(t, py, px)
        D = support_dist(ctx, dist, t, py, px, ks, rk)
        # (a) objects
        thr = r_obj
        have = ~np.isnan(D)
        sup = have & (D <= thr)
        nh = np.stack([np.bincount(Lb, weights=have[s], minlength=n) for s in (0, 1)])
        ns = np.stack([np.bincount(Lb, weights=sup[s], minlength=n) for s in (0, 1)])
        ev = (nh >= 0.5 * np.maximum(size, 1))
        share = np.where(ev, ns / np.maximum(nh, 1), np.nan)
        both = ev[0] & ev[1]
        f = np.nanmax(np.where(ev, share, -1), 0)
        lv_size = np.log(np.maximum(size, 1) / s0) / np.log(s1 / s0)
        lv_size2 = np.where(size > s1, 1 + np.log(np.maximum(size, 1) / s1) / np.log(s2 / s1), lv_size)  # aggr 2: s2 px
        # fragmented objects (share of px with <= 3 wet 8-neighbours >= frag_obj: dash fields, burst speckle) may go
        # at any size when unsupported; compact unsupported objects (rain in old-chain blink frames) only near aggr 1
        n8 = cv2.filter2D(wt.astype(np.float32), -1, _K8, borderType=cv2.BORDER_CONSTANT)[py, px]
        fragm = np.bincount(Lb, weights=(n8 <= 3), minlength=n) / np.maximum(size, 1)
        floor = np.where(fragm >= frag_obj, size_floor, floor_smooth)
        lf = np.where(f <= fmax, f / fmax, 1 + (f - fmax) / (fmax2 - fmax))                # aggr 2: f <= fmax2
        ff = np.maximum(np.clip(f / f_free, 0, 1), floor)
        lv = np.maximum(np.maximum(np.maximum(lv_size, 0) * ff, lf), EPS)
        lv2 = np.maximum(np.maximum(lv_size2, 0) * ff, lf)
        lv = np.where(lv <= 1, lv, np.maximum(lv, lv2))       # identical for aggr <= 1; above 1 the s2 limit also holds
        mxc = np.zeros(n, np.int64); np.maximum.at(mxc, Lb, out[t][py, px].astype(np.int64))
        lv = np.where(mxc <= 1, np.maximum(lv, code1), lv)        # code-1-only objects: the new chain has them too
        lv = np.where(both & (size > 0), lv, np.inf)
        lvp = lv[Lb]
        # (b) fringe pixels of large objects
        big = size[Lb] >= fr_min_obj
        if big.any():
            Dk = support_dist(ctx, dist, t, py[big], px[big], fr_k, rk)
            hb = ~np.isnan(Dk).any(0)
            ds = np.nanmin(Dk, 0)
            lvf = np.where(hb, fr_a0 + (1 - fr_a0) * np.clip((fr_hi - ds) / (fr_hi - fr_lo), 0, None), np.inf)
            lvf = np.where(ds > 1.0, np.maximum(lvf, EPS), np.inf)      # echo within 1 px on a side: never
            # pixels of compact objects that are unsupported as a whole (rain in blink frames) only near aggr 1
            Lbb = Lb[big]
            whole = (f[Lbb] < fmax) & (fragm[Lbb] < frag_obj)
            lvf = np.where(whole, np.maximum(lvf, floor_smooth), lvf)
            rem_f = a[big] > lvf
            if hb.sum() and rem_f[hb].mean() <= fr_blink:
                tmp = lvp[big]
                lvp[big] = np.minimum(tmp, lvf)
        lvp = np.where(live, lvp, np.inf)
        if getattr(ctx, "qsupport", None) is not None:
            lvp = np.where(ctx.qsupport[t][py, px], np.inf, lvp)
        kill = a > lvp
        level[t][py, px] = lvp
        if kill.any():
            res[t][py[kill], px[kill]] = 0
    return res, level


def _advected_both(ctx, out, t, py, px, tol=1, min_disp=1.5):
    """bool per pixel: echo (current output, dilated by tol) at the position moved by -v into t-1 AND by +v into
    t+1, with |v| >= min_disp (otherwise False: stationary echo is not 'moving rain')."""
    vx, vy = ctx.flow(t, py, px)
    ok = np.hypot(vx, vy) >= min_disp
    res = ok.copy()
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    for s_ in (-1, 1):
        u = t + s_
        if not (0 <= u < ctx.T) or not ctx.good[u]:
            return np.zeros(len(py), bool)
        d = cv2.dilate(((out[u] > 0) & ctx.valid[u]).view(np.uint8), k)
        ys = np.clip(np.rint(py + s_ * vy), 0, H - 1).astype(np.int64); xs = np.clip(np.rint(px + s_ * vx), 0, W - 1).astype(np.int64)
        res &= d[ys, xs] > 0
    return res


# ------------------------------------------------------------------ 7. radial dash stage
DASH_DEFAULTS = dict(lmax=40, wmax=3, wcap=4, dtheta=1.6, wloose=2, ratio=4.0, l_hi=60.0, l_lo=12.0, resid=6, r_min=20.0,
                     motion_gate=True)


def _runs(wet, py, px, uy, ux, nmax):
    """number of consecutive wet pixels from (py, px) stepping by (uy, ux) (excluding the start), up to nmax."""
    n = np.zeros(len(py), np.int16)
    alive = np.ones(len(py), bool)
    fy, fx = py.astype(np.float32), px.astype(np.float32)
    for k in range(1, nmax + 1):
        ys = np.rint(fy + k * uy).astype(np.int64); xs = np.rint(fx + k * ux).astype(np.int64)
        ins = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
        w = np.zeros(len(py), bool)
        w[ins] = wet[ys[ins], xs[ins]]
        alive &= w
        if not alive.any():
            break
        n += alive
    return n


def dash_stage(ctx, out, *, lmax=40, wmax=3, wcap=4, dtheta=1.6, wloose=2, ratio=4.0, l_hi=60.0, l_lo=12.0, resid=6, r_min=20.0,
               motion_gate=True):
    """Thin straight echo lines radial to an NL radar: tangential width <= max(wmax, r tan(dtheta)) px (about one
    or two beam widths at range r) and radial run >= ratio x width, the run counted over pixels of width <= that
    limit + wloose (a blob corner is thin, but its run must not pass through the blob).
    Dash pixels form segments (8-connected); a segment of radial length Ls has level (l_hi - Ls) / (l_hi - l_lo)
    (aggr 1: segments >= l_lo px). Small leftovers (<= resid px objects) touching a removed segment go with it.
    Lines inside rain (wide) are not touched; the stub of a ray sticking out of rain is."""
    res = out.copy()
    level = np.full(out.shape, np.inf, np.float32)
    k3 = np.ones((3, 3), np.uint8)
    for t in range(ctx.T):
        wet = (out[t] > 0) & ctx.valid[t]
        if not wet.any():
            continue
        py, px = np.nonzero(wet)
        best_len = np.zeros(len(py), np.int16)
        isd = np.zeros(len(py), bool)
        for sy, sx in NL_SITES:
            dy, dx = py - sy, px - sx
            r = np.hypot(dy, dx).astype(np.float32)
            far = r >= r_min
            uy, ux = (dy / np.maximum(r, 1)).astype(np.float32), (dx / np.maximum(r, 1)).astype(np.float32)
            wlim = np.clip(r * np.float32(np.tan(np.radians(dtheta))), wmax, wcap)   # cap 4 px (review R1)
            nw = int(np.ceil(wlim.max())) + wloose
            wa = _runs(wet, py, px, ux, -uy, nw); wb = _runs(wet, py, px, -ux, uy, nw)
            width = 1 + wa + wb
            thin = far & (width <= wlim)
            if not thin.any():
                continue
            loose = far & (width <= wlim + wloose)
            tm = np.zeros((H, W), bool); tm[py[loose], px[loose]] = True
            ty, tx = py[thin], px[thin]
            ra = _runs(tm, ty, tx, uy[thin], ux[thin], lmax); rb = _runs(tm, ty, tx, -uy[thin], -ux[thin], lmax)
            run = np.zeros(len(py), np.int16); run[thin] = 1 + ra + rb
            d = thin & (run >= ratio * width)
            isd |= d
            best_len = np.where(d, np.maximum(best_len, run), best_len)
        if not isd.any():
            continue
        dm = np.zeros((H, W), np.uint8); dm[py[isd], px[isd]] = 1
        n, lab = cv2.connectedComponents(dm, connectivity=8)
        L = lab[py[isd], px[isd]]
        seglen = np.zeros(n, np.int16); np.maximum.at(seglen, L, best_len[isd])
        lv_seg = np.maximum((l_hi - seglen) / (l_hi - l_lo), EPS)
        lv_seg = np.where(seglen < l_lo, np.inf, lv_seg)          # >= l_lo px at any aggr (review R1: no top-range extension)
        lvp = np.full(len(py), np.inf, np.float32)
        lvp[isd] = lv_seg[L]
        lvp = np.where(ctx.ok[t][py], lvp, np.inf)
        if motion_gate and isd.any():
            # rain bands persist and move: a dash whose pixels find echo at the advected position (1 px tolerance) in
            # both t-1 and t+1, with a displacement >= 1.5 px, is moving rain and not a dash (review R2, FM6)
            moving = _advected_both(ctx, out, t, py[isd], px[isd])
            segmov = np.bincount(L, weights=moving, minlength=n) / np.maximum(np.bincount(L, minlength=n), 1)
            tmp = lvp[isd]; tmp[segmov[L] > 0.5] = np.inf; lvp[isd] = tmp
        a = ctx.apx(t, py, px)
        kill = a > lvp
        if kill.any() and resid:
            # small leftovers attached to removed dash pixels
            km = np.zeros((H, W), np.uint8); km[py[kill], px[kill]] = 1
            rest = wet.copy(); rest[py[kill], px[kill]] = False
            n2, lab2, st2, _ = cv2.connectedComponentsWithStats(rest.view(np.uint8), connectivity=8)
            touch = np.zeros(n2, bool)
            kd = cv2.dilate(km, k3).astype(bool)
            touch[np.unique(lab2[kd & rest])] = True
            small = st2[:, cv2.CC_STAT_AREA] <= resid
            kk = touch & small; kk[0] = False
            extra = kk[lab2[py, px]] & ~kill
            lvp = np.where(extra, EPS, lvp)
            kill |= extra
        level[t][py, px] = lvp
        res[t][py[kill], px[kill]] = 0
    return res, level


# ------------------------------------------------------------------ 8. texture stage
TEXTURE_DEFAULTS = dict(box=9, min_obj=30, s_hi=3.0, s_lo=1.3, frag0=0.1, fs=0.25, rho0=0.6, rs=0.4, delta=0.3)


def _moved(mask, flow_full, sgn):
    """mask of a neighbour frame moved onto frame t's grid: sample at x + sgn * v (nearest)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(mask.view(np.uint8), xx + sgn * flow_full[0], yy + sgn * flow_full[1], cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)


def texture_score(ctx, out, box=9, frag0=0.1, fs=0.25, rho0=0.6, rs=0.4):
    """per frame (H, W) speckle score (frag - frag0) / fs + (rho0 - rho) / rs, NaN where dry. frag = share of
    loosely connected wet px (<= 3 wet 8-neighbours) in the box x box window, rho = wet density there.
    (A temporal-coherence term was tried and dropped: relative to chance it does not separate rain from clutter,
    NOTES.md.) Labelled TRAIN means: rain segments of merged objects frag 0.04 / rho 0.80, clutter segments 0.19 / 0.52."""
    T = ctx.T
    wet = (out > 0) & ctx.valid
    S = []
    for t in range(T):
        if not wet[t].any():
            S.append(None); continue
        rho, frag, _ = TF.features(wet[t], None, None, box)
        sc = (frag - frag0) / fs + (rho0 - rho) / rs
        S.append(np.where(wet[t], sc, np.nan).astype(np.float32))
    return S


def _moved_f(img, flow_full, sgn, fill=np.nan):
    """float image of a neighbour frame moved onto frame t's grid (sample at x + sgn v, nearest)."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    return cv2.remap(img.astype(np.float32), xx + sgn * flow_full[0], yy + sgn * flow_full[1], cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)


def texture_stage(ctx, out, *, box=9, min_obj=30, s_hi=3.0, s_lo=1.3, frag0=0.1, fs=0.25, rho0=0.6, rs=0.4,
                  delta=0.3):
    """Fragmented echo (sea clutter / speckle merged with or lying on rain) inside objects >= min_obj px.
    Score (texture_score) averaged ALONG THE MOTION over t-1, t, t+1 (the neighbours' score maps moved onto t);
    level (s_hi - score) / (s_hi - s_lo): aggr 1 removes score >= s_lo; no extension above aggr 1 (review R2:
    the aggr-2 extension ate rain fringes and caused most flips). Decisions are a Schmitt trigger along the motion:
    a pixel whose advected position at t-1 was removed needs only aggr > level - delta, one that was kept there
    needs aggr > level + delta, so a decision does not toggle while the score hovers near the threshold."""
    T = ctx.T
    S = texture_score(ctx, out, box, frag0, fs, rho0, rs)
    res = out.copy()
    level = np.full(out.shape, np.inf, np.float32)
    prev_state = None                        # (H, W) float: 1 removed, 0 kept, nan unknown (frame t-1 grid)
    for t in range(T):
        if S[t] is None:
            prev_state = None
            continue
        ff = ctx.flow_full(t)
        acc = np.nan_to_num(S[t]); cnt = np.isfinite(S[t]).astype(np.float32)
        for u, sgn in ((t - 1, -1), (t + 1, 1)):
            if 0 <= u < T and S[u] is not None and ctx.good[u]:
                m = _moved_f(S[u], ff, 1) if sgn > 0 else _moved_f(S[u], ctx.flow_full(u), -1)
                acc += np.nan_to_num(m); cnt += np.isfinite(m)
        sc = acc / np.maximum(cnt, 1)
        wt = np.isfinite(S[t])
        n, lab, sts, _ = cv2.connectedComponentsWithStats(wt.view(np.uint8), connectivity=8)
        big = (sts[:, cv2.CC_STAT_AREA] >= min_obj); big[0] = False
        py, px = np.nonzero(wt & big[lab])
        state = np.full((H, W), np.nan, np.float32)
        state[wt] = 0
        if len(py):
            scp = sc[py, px]
            lv = np.where(scp >= s_lo, np.maximum((s_hi - scp) / (s_hi - s_lo), EPS), np.inf)
            lv = np.where(ctx.ok[t][py], lv, np.inf)
            a = ctx.apx(t, py, px)
            if prev_state is not None and t > 0:
                ps = _moved_f(prev_state, ctx.flow_full(t - 1), -1)[py, px]
                thr = np.where(ps == 1, lv - delta, np.where(ps == 0, lv + delta, lv))
            else:
                thr = lv
            kill = (a > thr) & np.isfinite(lv)
            level[t][py, px] = np.where(kill, np.minimum(lv, a - EPS), np.where(a > lv, a + EPS, lv))   # removed <=> aggr > level
            res[t][py[kill], px[kill]] = 0
            state[py[kill], px[kill]] = 1
        prev_state = state
    return res, level


# ------------------------------------------------------------------ 9. value cap
def consistency(ctx, v11, out, *, lags=(1,), on=True, lvmin=None, add_margin=0.5):
    """Vote along the motion on the v12-only decisions. A pixel that v11 kept (raw-wet) is in state 'removed' when
    v12 set it to 0, else 'kept'. For each lag k, the pixel's position moved by -+k v in t-+k (nearest) gives a vote
    when that pixel is also v11-kept (removed or kept there); dry or v11-removed positions give no vote (so a burst
    with dry neighbours keeps its own decision). Final state = majority of own + neighbour votes, ties -> own.
    Adding a removal sets 0; restoring sets the v11 value (cap-lowered values count as kept and stay lowered).
    Only where local aggr > 0. A removal is added only where some stage nearly fired (lowest stage level < aggr +
    add_margin; review R2 FM8: the vote must not add removals no stage made)."""
    if not on:
        return out
    T = ctx.T
    v = ctx.valid
    k11 = (v11 > 0) & v
    rem = k11 & (out == 0)
    res = out.copy()
    for t in range(T):
        if not k11[t].any() or ctx.A.max(t) <= 0:
            continue
        py, px = np.nonzero(k11[t])
        vx, vy = ctx.flow(t, py, px)
        own = rem[t][py, px]
        nr = own.astype(np.int16); nk = (~own).astype(np.int16)
        for k in lags:
            for s in (-1, 1):
                u = t + s * k
                if not (0 <= u < T) or not ctx.good[u]:
                    continue
                ys = np.rint(py + s * k * vy).astype(np.int64); xs = np.rint(px + s * k * vx).astype(np.int64)
                ins = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
                yi, xi = np.where(ins, ys, 0), np.where(ins, xs, 0)
                use = ins & ctx.ok[u][yi] & k11[u][yi, xi]
                r = rem[u][yi, xi]
                nr += use & r; nk += use & ~r
        a = ctx.apx(t, py, px)
        new = np.where(nr > nk, True, np.where(nk > nr, False, own)) & (a > 0)
        new |= own & ~(a > 0)
        add = new & ~own; back = own & ~new & ctx.ok[t][py]
        if lvmin is not None:
            add &= lvmin[t][py, px].astype(np.float32) < a + add_margin
        if add.any():
            res[t][py[add], px[add]] = 0
        if back.any():
            res[t][py[back], px[back]] = v11[t][py[back], px[back]]
    return res


# ------------------------------------------------------------------ 11. polar ray stage (aggr > 1)
OUTLIER_DEFAULTS = dict(d_hi=2.3, d_lo=1.1)
_RING = np.ones((3, 3), np.uint8); _RING[1, 1] = 0


def outlier_stage(ctx, out, *, d_hi=2.3, d_lo=1.1):
    """Single hot pixels, active only for aggr > 1 (rewritten after review R1: the earlier median / ring references
    flattened convective cores). x = log1p(code). Reference = the SECOND-highest of the 8 neighbours (a core has
    several strong pixels; a hot pixel stands alone). Excess d = x - ref2. Temporal guard: x exceeds the MAXIMUM of x
    in the 3 x 3 box at the advected position in t-1 and in t+1 (both available): nothing comparable before or after.
    Level 1 + (d_hi - d) / (d_hi - d_lo): aggr 2 lowers pixels > 3 x (in 1 + code) above their second-highest
    neighbour. The pixel is lowered to that neighbour's value (never to 0 unless the neighbour is dry). Never raised."""
    T = ctx.T
    res = out.copy()
    level = np.full(out.shape, np.inf, np.float32)
    if ctx.A.max() <= 1:
        return res, level
    cache = {}

    def XM(u):
        if u not in cache:
            x_ = np.log1p(np.where(ctx.valid[u], out[u], 0).astype(np.float32))
            cache[u] = (x_, cv2.dilate(x_, np.ones((3, 3), np.uint8)))
            for k in [k for k in cache if k < u - 2]:
                del cache[k]
        return cache[u]
    sh = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    for t in range(T):
        if ctx.A.max(t) <= 1:
            continue
        x = XM(t)[0]
        if not (x > 0).any():
            continue
        nb1 = cv2.dilate(x, _RING)                                  # highest neighbour
        cand = (x > 0) & (x - nb1 > 0)                              # local strict maxima only
        py, px = np.nonzero(cand)
        if not len(py):
            continue
        xp = np.pad(x, 1)
        nbv = np.stack([xp[py + 1 + dy, px + 1 + dx] for dy, dx in sh])
        ref2 = np.partition(nbv, 6, axis=0)[6]                      # second-highest of 8
        d = x[py, px] - ref2
        sel0 = d > 0.5 * d_lo
        py, px, d, ref2 = py[sel0], px[sel0], d[sel0], ref2[sel0]
        if not len(py):
            continue
        vx, vy = ctx.flow(t, py, px)
        guard = np.ones(len(py), bool)
        for s_ in (-1, 1):
            u = t + s_
            if not (0 <= u < T) or not ctx.good[u]:
                guard[:] = False; break
            ys = np.clip(np.rint(py + s_ * vy), 0, H - 1).astype(np.int64); xs = np.clip(np.rint(px + s_ * vx), 0, W - 1).astype(np.int64)
            use = ctx.ok[u][ys] & ctx.valid[u][ys, xs]
            guard &= use & (XM(u)[1][ys, xs] < ref2 + 0.5 * d)
        lv = np.where(guard & ctx.ok[t][py], np.maximum(1 + (d_hi - d) / (d_hi - d_lo), 1 + EPS), np.inf)
        a = ctx.apx(t, py, px)
        sel = a > lv
        level[t][py, px] = lv
        if sel.any():
            nv = np.rint(np.expm1(ref2[sel]))
            res[t][py[sel], px[sel]] = np.minimum(res[t][py[sel], px[sel]], nv).astype(out.dtype)
    return res, level


# ------------------------------------------------------------------ 13. orientation-free thin-line stage (aggr > 1)
LINES_DEFAULTS = dict(c_hi=1.8, c_lo=0.92, min_px=3, min_len=5, tol=1, bg_max=2.2)


def lines_stage(ctx, out, *, c_hi=1.8, c_lo=0.92, min_px=3, min_len=5, tol=1, bg_max=2.2):
    """Thin bright structures of any orientation (anaprop dashes across the beam, streaks, strobes' rims), active only
    for aggr > 1. x = log1p(code); thin part = x - grey opening of x with a 3 x 3 square (structures <= 2 px wide);
    candidates: thin part > c_lo*0.5, in connected groups that are LINES (>= min_len px long, area <= 2.5 x length;
    blob-like convective peaks are not lines, review R1). Motion gate: the pixel finds no echo of
    comparable strength (3 x 3 median of x at the advected position >= x - c_lo) in t-1 AND in t+1 (a transient
    streak not carried along by the flow; a one-side test made persistent fringes toggle). Level 1 + (c_hi - c) /
    (c_hi - c_lo), c = thin part: aggr 2 removes thin structures > 2.5 x (in 1 + code) above their across-line
    surroundings. The pixel is lowered to the opened value (an isolated line goes to 0, a streak in rain to the rain
    level). Only where the opened surroundings are weak (log1p <= bg_max, code <= 8, ~1 mm/h). Never raised."""
    T = ctx.T
    res = out.copy()
    level = np.full(out.shape, np.inf, np.float32)
    if ctx.A.max() <= 1:
        return res, level
    k3 = np.ones((3, 3), np.uint8)
    cache = {}

    def XM(u):
        if u not in cache:
            x_ = np.log1p(np.where(ctx.valid[u], out[u], 0).astype(np.float32))
            cache[u] = (x_, cv2.medianBlur(x_, 3))
            for k in [k for k in cache if k < u - 2]:
                del cache[k]
        return cache[u]
    for t in range(T):
        if ctx.A.max(t) <= 1 or not ctx.good[t]:
            continue
        x = XM(t)[0]
        if not (x > 0).any():
            continue
        op = cv2.morphologyEx(x, cv2.MORPH_OPEN, k3)
        th = x - op
        cand = th > 0.5 * c_lo
        if not cand.any():
            continue
        n, lab, sts, _ = cv2.connectedComponentsWithStats(cand.view(np.uint8), connectivity=8)
        ext = np.maximum(sts[:, cv2.CC_STAT_WIDTH], sts[:, cv2.CC_STAT_HEIGHT])
        okc = (sts[:, cv2.CC_STAT_AREA] >= min_px) & (ext >= min_len) & (sts[:, cv2.CC_STAT_AREA] <= 2.5 * ext); okc[0] = False
        py, px = np.nonzero(okc[lab])
        if not len(py):
            continue
        xv = x[py, px]
        vx, vy = ctx.flow(t, py, px)
        absent = np.ones(len(py), bool); have = np.zeros(len(py), bool)
        for s_ in (-1, 1):
            u = t + s_
            if 0 <= u < T and ctx.good[u]:
                ys = np.clip(np.rint(py + s_ * vy), 0, H - 1).astype(np.int64); xs = np.clip(np.rint(px + s_ * vx), 0, W - 1).astype(np.int64)
                use = ctx.ok[u][ys] & ctx.valid[u][ys, xs]
                med = XM(u)[1][ys, xs]
                absent &= ~use | (med < xv - c_lo)          # absent on BOTH sides (persistent structures stay:
                have |= use                                   # fewer holes / flips; review R2 recheck)
        c = th[py, px]
        weak_bg = op[py, px] <= bg_max            # only lines in clear air / weak echo: inside convective rain thin bright
                                                  # filaments are real structure (review R1 gate: cores were lowered)
        lv = np.where(absent & have & weak_bg & ctx.ok[t][py], np.maximum(1 + (c_hi - c) / (c_hi - c_lo), 1 + EPS), np.inf)
        a = ctx.apx(t, py, px)
        sel = a > lv
        level[t][py, px] = lv
        if sel.any():
            nv = np.rint(np.expm1(op[py[sel], px[sel]]))
            res[t][py[sel], px[sel]] = np.minimum(res[t][py[sel], px[sel]], nv).astype(out.dtype)
    return res, level


# ------------------------------------------------------------------ 14. realism guard (final, over all stages)
GUARD_DEFAULTS = dict(touch_max=0.35, frag_ok=0.5, low_nb=1 / 3.0)
_K3 = np.ones((3, 3), np.uint8)


def _holes(wet):
    """bool: dry pixels enclosed by wet pixels (not connected to the frame border through dry pixels)."""
    dry = (~wet).astype(np.uint8)
    n, lab = cv2.connectedComponents(dry, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    enc = np.ones(n, bool); enc[border] = False; enc[0] = False
    return enc[lab] & (dry > 0)


def realism_guard(ctx, ref, out, sid, *, touch_max=0.35, frag_ok=0.5, low_nb=1 / 3.0, names=()):
    """Final guard: an edit may remove a whole coherent structure (an isolated object, a speck, a dash field, a part
    attached to rain by a thin neck) but must never carve inside coherent rain. Per frame, the v12 edits relative to
    `ref` (v11 incl. the shifted object decisions) are split into connected regions of ZEROED pixels and of LOWERED
    pixels.
    Zeroed region R is reverted when it carves:
      - it creates a hole: some pixel of R is enclosed by echo that is kept (a hole in out that was not one in ref), or
      - interior carving: >= touch_max of its outer ring (3 x 3 dilation) is kept echo, unless R itself is fragmented
        (>= frag_ok of its pixels have <= 3 wet 8-neighbours in ref: speckle / sea clutter, not smooth rain).
    Lowered pixel p is reverted when its second-highest 8-neighbour (lowered neighbours at their new value) is
    >= low_nb x its ref value (a core with strong neighbours, not a lone hot pixel or a lowered line).
    Returns (out, stats) with stats[stage] = [regions, reverted regions, reverted px] (stage = earliest stage that
    flagged most pixels of the region; 'vote' when no stage flagged it)."""
    T = ctx.T
    res = out.copy()
    st = {}
    sh = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    for t in range(T):
        v = ctx.valid[t]
        rw = (ref[t] > 0) & v
        ow = (out[t] > 0) & v
        zero = rw & ~ow
        low = ow & (out[t] < ref[t])
        if not zero.any() and not low.any():
            continue
        if zero.any():
            newhole = _holes(ow) & ~_holes(rw)
            n8 = cv2.filter2D(rw.astype(np.float32), -1, _K8, borderType=cv2.BORDER_CONSTANT)
            loose = rw & (n8 <= 3)
            n, lab, sts, _ = cv2.connectedComponentsWithStats(zero.view(np.uint8), connectivity=8)
            for i in range(1, n):
                x0, y0, w_, h_, area = sts[i]
                y0_, x0_ = max(y0 - 1, 0), max(x0 - 1, 0)
                sl = (slice(y0_, min(y0 + h_ + 1, H)), slice(x0_, min(x0 + w_ + 1, W)))
                m = lab[sl] == i
                ring = cv2.dilate(m.view(np.uint8), _K3).astype(bool) & ~m
                touch = ow[sl][ring].mean() if ring.any() else 0.0
                frag = loose[sl][m].mean()
                hole = newhole[sl][m].any()
                s_ = sid[t][sl][m]
                s_ = s_[s_ > 0]
                key = names[np.bincount(s_).argmax() - 1] if len(s_) else "vote"
                rec = st.setdefault(key, [0, 0, 0])
                rec[0] += 1
                if hole or (touch >= touch_max and frag < frag_ok):
                    rec[1] += 1; rec[2] += int(area)
                    res[t][sl][m] = ref[t][sl][m]
        if low.any():
            py, px = np.nonzero(low)
            # neighbours that were lowered themselves count with their NEW value (a lowered line or cluster is judged
            # against what stays around it, not against its own pixels)
            rp = np.pad(np.where(low, out[t], ref[t]).astype(np.float32), 1)
            nbv = np.stack([rp[py + 1 + dy, px + 1 + dx] for dy, dx in sh])
            nb2 = np.partition(nbv, 6, axis=0)[6]
            bad = nb2 >= low_nb * ref[t][py, px]
            s_ = sid[t][py, px]
            for k in np.unique(s_):
                key = names[k - 1] if k > 0 else "vote"
                rec = st.setdefault(key + "(lowered)", [0, 0, 0])
                mk = s_ == k
                rec[0] += int(mk.sum()); rec[1] += int((mk & bad).sum()); rec[2] += int((mk & bad).sum())
            if bad.any():
                res[t][py[bad], px[bad]] = ref[t][py[bad], px[bad]]
    return res, st
