"""Per-azimuth ray test that tolerates rain on the ray and wide wedges (seed detector for spikes.py).

    wedge_mask(codes) -> bool (n_sites, T, H, W): wet pixels on detected rays (seeds; spikes.py handles the whole radial)

Per NL radar, polar cells 0.5 deg x 2 km (proto/fast/streaks.py index); O = wet share of the valid pixels per cell.
  side   for a cell: max of the left/right mean O over rows k..k+6 beside it, the smaller of k = 3 and k = 6
         (1.5 / 3 deg: narrow rays and wedges up to ~5 deg)
  CA     clear-air ray cell: O >= 0.5 and side <= side_max, in frame t and (motion guard) in t-1 or t+1
  WET    O >= 0.5
A ray cell: within a window of win bins along range, >= wet_share of the cells WET (the ray is continuous, also
where it crosses rain) and >= ca_min of them CA (it is seen in clear air with dry rows beside it over >= 2*ca_min km).
Rays: connected candidate cells >= min_run bins long, <= max_deg wide, isolated (no other candidate within
isolate_deg: striped far-range sectors are not rays), novel (in some frame of t-2..t+2 at most nov_max of the
ray's clear-air cells are wet, wet = within 1 row / 2 bins), smooth along range (tex_max) and not fading with range (slope_min); only within
the radar's range (r_max bins = 332 km), running out to within end_bins (30 km) of the end of the radar's data.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import streaks as SK

MISSING = 65534
H, W = 765, 700
NAZ, NR = SK.NAZ, SK.NR
DEFAULTS = dict(side_max=0.1, win=30, wet_share=0.8, ca_min=15, min_run=30, max_deg=5.0, isolate_deg=10.0,
                ks=(3, 6), kw=7, r_min=10, r_max=166, time_guard=True, nov_max=0.5, nov_dil=(3, 5),
                tex_max=0.25, slope_min=-1.0, end_bins=15)
_VAL = {}


def _valid_counts(si, idx, inside, v):
    key = si
    if key not in _VAL or not np.array_equal(_VAL[key][0], v):
        n = np.bincount(idx[v & inside], minlength=NAZ * NR).astype(np.float32).reshape(NAZ, NR)
        _VAL[key] = (v.copy(), np.where(n > 0, 1.0 / np.maximum(n, 1), 0).astype(np.float32))   # 1 / valid count
    return _VAL[key][1]


def _reach_end(flag, inv, r_max, end_bins):
    """keep rays that run out to within end_bins of the end of the radar's data along their azimuth (min of r_max
    and the last valid cell): the apparent power of interference grows with range, so a ray seen at r is seen out
    to the radar's maximum range; a radial rain band ends somewhere."""
    valid = inv > 0
    last = np.where(valid.any(1), NR - 1 - np.argmax(valid[:, ::-1], axis=1), 0)
    end = np.minimum(last, r_max - 1)
    n, lab = cv2.connectedComponents(flag.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        ys, xs = np.nonzero(lab == j)
        keep[j] = xs.max() >= end[np.unique(ys)].min() - end_bins
    return keep[lab] if keep.any() else None


def _smooth_rays(flag, f, idx, inside, tex_max, slope_min):
    """keep rays whose values are smooth along range (mean |step| of log1p(code) between neighbouring 2-km cells
    <= tex_max: interference adds a steady power, rain has cells and cores) and do not fade fast with range
    (slope of the mean log value >= slope_min per 100 km: the apparent rate of interference grows with range)."""
    v = f < MISSING
    m = inside & v & (f > 0)
    L = np.log1p(f[m].astype(np.float32))
    n_all = np.bincount(idx[inside & v], minlength=NAZ * NR)
    s = np.bincount(idx[m], weights=L, minlength=NAZ * NR)
    Pc = (s / np.maximum(n_all, 1)).reshape(NAZ, NR).astype(np.float32)
    n, lab = cv2.connectedComponents(flag.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    for j in range(1, n):
        mm = lab == j
        both = mm[:, 1:] & mm[:, :-1]
        if both.sum() < 10:
            continue
        tex = float(np.abs(np.diff(Pc, axis=1))[both].mean())
        cols = np.flatnonzero(mm.any(0))
        prof = (Pc * mm).sum(0)[cols] / mm.sum(0)[cols]
        slope = float(np.polyfit(cols * 2.0, prof, 1)[0] * 100) if len(cols) > 2 else 0.0
        keep[j] = tex <= tex_max and slope >= slope_min
    if not keep.any():
        return None
    return keep[lab]


def wedge_mask(codes, *, sites=None, ret_cells=False, **kw):
    p = {**DEFAULTS, **kw}
    codes = np.asarray(codes)
    T = len(codes)
    flat = codes.reshape(T, -1)
    out = np.zeros((len(sites) if sites is not None else len(SK.NL_SITES), T, H * W), bool)
    sites = SK.NL_SITES if sites is None else sites
    rows = NAZ / 360.0
    max_rows, iso = int(round(p["max_deg"] * rows)), int(round(p["isolate_deg"] * rows))
    cells = {}
    V = [flat[t] < MISSING for t in range(T)]
    WP = [np.flatnonzero((flat[t] > 0) & V[t]) for t in range(T)]
    for si, center in enumerate(sites):
        idx = SK._index(tuple(center)); inside = idx >= 0
        O, CA0, WET = [], [], []
        for t in range(T):
            v = V[t]
            wp = WP[t]
            wp = wp[inside[wp]]
            if len(wp) < p["ca_min"] * 2:
                O.append(None); CA0.append(None); WET.append(None); continue
            inv = _valid_counts(si, idx, inside, v)
            o = np.bincount(idx[wp], minlength=NAZ * NR).astype(np.float32).reshape(NAZ, NR) * inv
            # side means over rows a+-k..a+-(k+kw-1), wrapped in azimuth, from one cumulative sum
            hi = max(p["ks"]) + p["kw"]
            C = np.zeros((NAZ + 2 * hi + 1, NR), np.float32)
            np.cumsum(np.concatenate([o[-hi:], o, o[:hi]]), axis=0, out=C[1:])
            sd = None
            for k in p["ks"]:
                kw_ = p["kw"]
                rs = C[hi + k + kw_:hi + k + kw_ + NAZ] - C[hi + k:hi + k + NAZ]
                ls = C[hi - k + 1:hi - k + 1 + NAZ] - C[hi - k - kw_ + 1:hi - k - kw_ + 1 + NAZ]
                sk = np.maximum(ls, rs) * np.float32(1.0 / kw_)
                sd = sk if sd is None else np.minimum(sd, sk)
            wet = o >= 0.5
            wet[:, p["r_max"]:] = False                  # beyond the radar's range the pixels are other radars'
            O.append(o); WET.append(wet); CA0.append(sd <= p["side_max"])
        for t in range(T):
            if O[t] is None:
                continue
            ca = WET[t] & CA0[t]
            if p["time_guard"]:
                # motion guard: dry beside the cell also in t-1 or t+1 (a band drifting across the azimuth has
                # echo beside it in both; a fixed ray next to approaching rain in at most one)
                g = np.zeros_like(ca)
                for u in (t - 1, t + 1):
                    if 0 <= u < T:
                        g |= CA0[u] if CA0[u] is not None else True
                ca &= g
            ca[:, :p["r_min"]] = False
            if ca.sum() < p["ca_min"]:
                continue
            win = p["win"]
            sw = cv2.boxFilter(WET[t].astype(np.float32), -1, (win, 1), normalize=False, borderType=cv2.BORDER_CONSTANT)
            sc = cv2.boxFilter(ca.astype(np.float32), -1, (win, 1), normalize=False, borderType=cv2.BORDER_CONSTANT)
            cand = (sw >= p["wet_share"] * win) & (sc >= p["ca_min"])
            if not cand.any():
                continue
            # cells of the window: dilate the window centres back along range, keep wet cells
            cand = cv2.dilate(cand.astype(np.uint8), np.ones((1, win), np.uint8)).astype(bool) & WET[t]
            flag = SK._runs(cand, p["min_run"], max_rows, iso)
            if flag is None:
                continue
            flag &= WET[t]
            if p["nov_max"] < 1:
                # novel: in some frame of t-2..t+2 most of the ray's clear-air cells are not wet (interference comes
                # and goes; a radial rain band persists)
                n, lab = cv2.connectedComponents(flag.astype(np.uint8), connectivity=8)
                l = lab.ravel()
                cl = np.where(ca.ravel(), l, 0)
                ne = np.bincount(cl, minlength=n).astype(float)
                ov = np.full(n, np.inf)
                for u in (t - 2, t - 1, t + 1, t + 2):
                    if 0 <= u < T:
                        if WET[u] is None:
                            wu = np.zeros(NAZ * NR, bool)
                        else:
                            wu = (cv2.dilate(WET[u].astype(np.uint8), np.ones(p["nov_dil"], np.uint8)) > 0).ravel()
                        ov = np.minimum(ov, np.bincount(cl, weights=wu, minlength=n) / np.maximum(ne, 1))
                keep = ov <= p["nov_max"]
                keep[0] = False
                flag = keep[lab]
                if not flag.any():
                    continue
            if p["tex_max"] < 9:
                flag = _smooth_rays(flag, flat[t], idx, inside, p["tex_max"], p["slope_min"])
                if flag is None:
                    continue
            if p["end_bins"] < NR:
                flag = _reach_end(flag, _valid_counts(si, idx, inside, V[t]), p["r_max"], p["end_bins"])
                if flag is None:
                    continue
            cells[(si, t)] = flag
            fl = flag.ravel()
            wp = WP[t]
            wp = wp[inside[wp]]
            out[si, t, wp[fl[idx[wp]]]] = True
    out = out.reshape(len(out), T, H, W)
    return (out, cells) if ret_cells else out
