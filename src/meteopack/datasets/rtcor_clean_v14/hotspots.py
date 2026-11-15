"""Stage 4: fixed-clutter hot spots (research proto/climmap).

Hot pixels come from a climatological map (assets hot.old / hot.new: pixels with isolated echo far more often
than their surroundings, persistent over >= 2 months; built from the monthly RTCOR counts of each processing
chain). Spots = connected hot pixels; zone = spot grown by 1 px.
- Spot activity: share of the window's usable frames in which the zone has raw echo while the ring around the
  spot (3..7 px, other hot px excluded) is < 25 % wet. Spots below 0.2 are left alone in this window.
- Candidates: kept echo objects of <= 400 px with >= 60 % of their px in an active spot's zone.
- Rain-through evidence per candidate: its pixels moved back by k v into frame t-k and forward into t+k
  (k = 1..3, v = 4 km flow, median of 5 x 5 cells), share with echo within 2 px; per k the smaller side, then the
  largest over k. Samples on missing data, in frozen halves or inside hot zones are no evidence. No usable sample
  at all (stationary flow): rain if the spot ring is >= 25 % wet now.
- Decision per spot track: a spot is 'rain' at t if any candidate on it at t-1..t+1 is rain (>= 0.25);
  otherwise all its candidates at t are removed. Frozen halves copy the previous output.
Never adds or raises echo, never touches rain moving through.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

from . import assets as _assets
from . import stale as bglayer

MISSING = 65534
H, W = 765, 700
DS = 4
STAGE_DEFAULTS = dict(rain_frac=0.25, ring_frac=0.25, inside_frac=0.6, zone=1, ring=(3, 7), ks=(1, 2, 3), hyst=1,
                      min_side=0.3, flow_cells=2, tol=2, act_min=0.2, max_obj=400, self_excl=False)
NEW_MAP_FROM = np.datetime64("2023-02", "M")


def window_map(times):
    """Hot map for the window: the processing chain of the window's middle time, by month (research
    climmap window_map with period="chain"). None without times."""
    if times is None or len(times) == 0:
        return None
    tm = np.asarray(times)[len(times) // 2]
    new = np.datetime64(tm, "M") >= NEW_MAP_FROM
    key = "new" if new else "old"
    if key not in _HOT:
        _HOT[key] = {"hot": _assets.group("hot")[key].astype(bool)}
    return _HOT[key]


_HOT: dict = {}


def spots(hot, grow=1, min_px=1):
    """Connected hot spots (8-conn, after a 1 px closing so a wind farm's pixels form one spot)."""
    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    lab, n = ndi.label(cv2.dilate(hot.astype(np.uint8), k), structure=np.ones((3, 3)))
    lab = np.where(hot, lab, 0)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), 1):
        if sl is None:
            continue
        yy, xx = np.nonzero(lab[sl] == i)
        if len(yy) >= min_px:
            out.append((yy + sl[0].start, xx + sl[1].start))
    return out


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def spot_geometry(hot, zone=1, ring=(3, 7)):
    """zone label map (spot id + 1 on the spot grown by `zone` px; 0 elsewhere) and per spot the ring pixels
    (ring[0]..ring[1] px from the spot, away from any hot px)."""
    allhot = cv2.dilate(hot.astype(np.uint8), _disk(ring[0])).astype(bool)
    zl = np.zeros((H, W), np.int32)
    rl = np.zeros((H, W), np.int32)
    rings = []
    for i, (yy, xx) in enumerate(spots(hot)):
        y0, y1 = max(yy.min() - ring[1] - 1, 0), min(yy.max() + ring[1] + 2, H)
        x0, x1 = max(xx.min() - ring[1] - 1, 0), min(xx.max() + ring[1] + 2, W)
        m = np.zeros((y1 - y0, x1 - x0), np.uint8); m[yy - y0, xx - x0] = 1
        z = cv2.dilate(m, _disk(zone)).astype(bool)
        sub = zl[y0:y1, x0:x1]
        sub[z & (sub == 0)] = i + 1
        r = cv2.dilate(m, _disk(ring[1])).astype(bool) & ~allhot[y0:y1, x0:x1]
        ry, rx = np.nonzero(r)
        rings.append((ry + y0, rx + x0))
        rsub = rl[y0:y1, x0:x1]
        rsub[r & (rsub == 0)] = i + 1
    return {"zl": zl, "rl": rl, "rings": rings, "n": len(rings)}


def flow_at(flr, t, cy, cx, cells=2):
    """median motion (px / 5 min) over (2 cells + 1)^2 4 km cells around (cy, cx)."""
    fl, sc = flr[t]
    gy, gx = min(cy // DS, fl.shape[0] - 1), min(cx // DS, fl.shape[1] - 1)
    blk = fl[max(gy - cells, 0):gy + cells + 1, max(gx - cells, 0):gx + cells + 1].reshape(-1, 2)
    v = np.median(blk, 0) * (DS * sc)
    return float(v[0]), float(v[1])


def obj_evidence(Dv, okrows, valid, inzone, t, py, px, v, ks, min_side, own=None):
    """Advected support of one object: for k in ks, the share of its pixels whose position moved back by k v has
    (dilated) v10 echo at t-k, and moved forward at t+k; per k the smaller side (one side if the other has no
    evidence), then the largest over k. Samples on missing data, in frozen halves or inside any hot zone (where the
    clutter itself sits) are no evidence, nor are samples on the object's own footprint (own = (y0, x0, mask) of the
    object grown by the tolerance): a stationary echo must not support itself. NaN: no usable side at all."""
    T = len(Dv)
    n = len(py)
    best = np.nan
    for k in ks:
        sides = []
        for s in (-1, 1):
            u = t + s * k
            if not 0 <= u < T:
                continue
            ys = np.rint(py + s * k * v[1]).astype(np.int64); xs = np.rint(px + s * k * v[0]).astype(np.int64)
            ins = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
            ys, xs = ys[ins], xs[ins]
            use = okrows[u][ys] & valid[u][ys, xs] & ~inzone[ys, xs]
            if own is not None:
                oy, ox, om = own
                ly, lx = ys - oy, xs - ox
                inb = (ly >= 0) & (ly < om.shape[0]) & (lx >= 0) & (lx < om.shape[1])
                mine = np.zeros(len(ys), bool)
                mine[inb] = om[ly[inb], lx[inb]]
                use &= ~mine
            if use.sum() < max(1, min_side * n):
                continue
            sides.append(Dv[u][ys[use], xs[use]].mean())
        if sides:
            f = min(sides)
            best = f if np.isnan(best) else max(best, f)
    return best


def spot_activity(codes, wetv, okrows, valid, geo, ring_frac):
    """(S,) share of usable frames in which the spot zone has raw echo while its ring is < ring_frac wet (v10):
    how often the spot shows isolated echo in this window (the per-track 'is it hot here and now' test)."""
    T = len(codes)
    zl, rl, S = geo["zl"], geo["rl"], geo["n"]
    zm, rm_ = zl > 0, rl > 0
    zlab, rlab = zl[zm], rl[rm_]
    act = np.zeros(S + 1); nus = np.zeros(S + 1)
    for t in range(T):
        rows_ok = okrows[t]
        vz = valid[t][zm] & rows_ok[np.nonzero(zm)[0]]
        has = np.bincount(zlab[vz & (codes[t][zm] > 0)], minlength=S + 1) > 0
        usez = np.bincount(zlab[vz], minlength=S + 1) > 0
        vr = valid[t][rm_] & rows_ok[np.nonzero(rm_)[0]]
        rw = np.bincount(rlab[vr], weights=wetv[t][rm_][vr], minlength=S + 1)
        rn = np.bincount(rlab[vr], minlength=S + 1)
        quiet = rw < ring_frac * np.maximum(rn, 1)
        act += has & quiet & usez; nus += usez
    return (act / np.maximum(nus, 1))[1:]


def hot_stage(codes, out, st, geo, *, rain_frac=0.25, ring_frac=0.25, inside_frac=0.6, zone=1, ring=(3, 7),
              ks=(1, 2, 3), hyst=1, min_side=0.3, flow_cells=2, tol=2, act_min=0.2, max_obj=400, self_excl=False):
    """Remove kept echo objects that sit on mapped hot spots unless they are rain moving through (see header).
    Decision per spot track: a spot is 'rain' in frame t when any of its objects at t-hyst..t+hyst had advected
    support >= rain_frac, or (no advected evidence at all) its ring was >= ring_frac wet at t."""
    T = len(codes)
    valid = codes < MISSING
    wetv = (out > 0) & valid
    okrows = bglayer.usable(st["flags"], T)
    res = out.copy()
    info = {"removed_px": np.zeros(T, np.int64), "n_spots": geo["n"], "cand": 0, "rain_obj": 0, "unknown": 0}
    zl = geo["zl"]
    if geo["n"] == 0:
        return res, info
    inzone = zl > 0
    if not (wetv & inzone[None]).any():
        return res, info
    Dv = np.empty_like(wetv)
    kt = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    for t in range(T):
        Dv[t] = cv2.dilate(wetv[t].view(np.uint8), kt).view(bool)
    S = geo["n"]
    act = spot_activity(codes, wetv, okrows, valid, geo, ring_frac)
    info["active_spots"] = int((act >= act_min).sum())
    state = np.full((S, T), -1, np.int8)        # -1 no candidate, 0 dry, 1 rain
    cands = {}
    for t in range(T):
        if not (wetv[t] & inzone).any():
            continue
        n, lab, stats, cen = cv2.connectedComponentsWithStats(wetv[t].view(np.uint8), connectivity=8)
        lz = lab[inzone]; zz = zl[inzone]
        sel = lz > 0
        cnt = np.bincount(lz[sel], minlength=n)
        area = stats[:, cv2.CC_STAT_AREA]
        objs = np.nonzero((cnt > 0) & (cnt >= inside_frac * area) & (area <= max_obj))[0]
        objs = objs[objs > 0]
        if not len(objs):
            continue
        # spot of each object: the most frequent zone label among its zone pixels
        pair = lz[sel].astype(np.int64) * (S + 1) + zz[sel]
        up, uc = np.unique(pair, return_counts=True)
        best = {}
        for pv, c_ in zip(up, uc):
            o, sid = divmod(int(pv), S + 1)
            if c_ > best.get(o, (0, 0))[0]:
                best[o] = (c_, sid - 1)
        frozen_rows = ~okrows[t]
        for o in objs:
            sid = best[int(o)][1]
            if act[sid] < act_min:
                continue                                  # spot not active in this window: leave to v10
            x0, y0, w_, h_ = stats[o, :4]
            if frozen_rows[int(cen[o, 1])]:
                continue                                  # frozen copy: gets t-1's output (copy below)
            loc = lab[y0:y0 + h_, x0:x0 + w_] == o
            yy, xx = np.nonzero(loc)
            py, px = (yy + y0).astype(np.float32), (xx + x0).astype(np.float32)
            om = cv2.dilate(np.pad(loc, tol).view(np.uint8), kt).view(bool)
            v = flow_at(st["flr"], t, int(cen[o, 1]), int(cen[o, 0]), flow_cells)
            e = obj_evidence(Dv, okrows, valid, inzone, t, py, px, v, ks, min_side,
                             own=(y0 - tol, x0 - tol, om) if self_excl else None)
            info["cand"] += 1
            if np.isnan(e):
                info["unknown"] += 1
                ry, rx = geo["rings"][sid]
                use = okrows[t][ry] & valid[t][ry, rx]
                rain = use.sum() > 0 and wetv[t][ry[use], rx[use]].mean() >= ring_frac
            else:
                rain = e >= rain_frac
            info["rain_obj"] += int(rain)
            info.setdefault("cl", []).append((t, sid, len(py), float(e), float(np.hypot(*v)), int(rain)))
            state[sid, t] = max(state[sid, t], int(rain))
            cands.setdefault(t, []).append((sid, int(o), (yy + y0, xx + x0)))
    rain = state == 1
    rr = rain.copy()
    for h in range(1, hyst + 1):
        rr[:, h:] |= rain[:, :-h]; rr[:, :-h] |= rain[:, h:]
    for t, lst in cands.items():
        for sid, o, (yy, xx) in lst:
            if not rr[sid, t]:
                info["removed_px"][t] += len(yy)
                res[t][yy, xx] = 0
    for t in range(1, T):                                       # frozen copy (as v10)
        for bit, sl in bglayer.HALVES:
            if st["flags"][t] & bit:
                same = codes[t][sl] == codes[t - 1][sl]
                res[t][sl] = np.where(same, res[t - 1][sl], res[t][sl])
    res = np.where(valid, res, codes).astype(codes.dtype)
    return res, info


_GEO = {}


def cached_geo(key, hot, zone, ring):
    k = (key, zone, tuple(ring))
    if k not in _GEO:
        _GEO.clear()
        _GEO[k] = spot_geometry(hot, zone, ring)
    return _GEO[k]


def _clear():
    _HOT.clear(); _GEO.clear()


_assets.register_cache(_clear)
