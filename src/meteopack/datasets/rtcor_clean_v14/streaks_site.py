"""proto/fast/streaks.py streak_mask, split by radar (identical decisions; OR over the sites = streak_mask).

    sm_sites = streak_mask_sites(codes, times)   # bool (n_sites, T, H, W)
"""
from __future__ import annotations

import numpy as np

from .streaks import (
    ALL_SITES,
    DR,
    NAZ,
    NL_SITES,
    NR,
    _known_cc,
    _runs,
    _Site,
    _thin_only,
    _transient,
    azmap,
)

MISSING = 65534
H, W = 765, 700

def streak_mask_sites(codes, times=None, *, sites="nl", core_min=0.25, side_max=0.04, min_run_km=40,
                max_deg=2.5, isolate_deg=10, time_ks=(-2, -1, 1, 2), win_km=22, side_deg=(1.5, 7.0),
                novel_max=0.3, echo_min_km=30, use_recur=False, recur_half=8, recur_min=0.05,
                recur_ratio=0.2, recur_side_abs=0.03, recur_thin=2.0, recur_share=0.5, recur_side_t=0.04, recur_run_km=40,
                use_known=True, known_side_t=0.15, known_thin=1.5, known_share=0.5,
                known_min_rate=0.003, known_contrast=2.5, known_min_km=14):
    """bool (n_sites, T, H, W): streak_mask() split by the radar whose polar test flagged the pixels."""
    if isinstance(sites, str):
        sites = ALL_SITES if sites == "all" else NL_SITES
    codes = np.asarray(codes)
    T = len(codes)
    flat = codes.reshape(T, -1)
    wet = (flat > 0) & (flat < MISSING)
    valid = flat < MISSING
    any_wet = wet.any(1)
    out = np.zeros((len(sites), T, H * W), bool)
    rows = NAZ / 360.0
    win = max(int(round(win_km / DR)) | 1, 3)
    side = (int(round(side_deg[0] * rows)), int(round(side_deg[1] * rows)))
    min_run, max_rows, isolate = int(round(min_run_km / DR)), int(round(max_deg * rows)), int(round(isolate_deg * rows))
    known = azmap(known_min_rate, known_contrast) if use_known else None
    for si, center in enumerate(sites):
        st = _Site(tuple(center), win, side)
        if si == 0:
            wps = [np.flatnonzero(wet[t]) if any_wet[t] else None for t in range(T)]
        cs = [st.core_side(wps[t], valid[t]) if any_wet[t] else None for t in range(T)]
        if use_recur:
            cum = np.zeros((T + 1, NAZ, NR), np.float32)
            for t in range(T):
                cum[t + 1] = cum[t] + (cs[t][3] if cs[t] is not None else 0)
        for t in range(T):
            if cs[t] is None:
                continue
            core, sd, vc, o = cs[t]
            flag = _transient(cs, t, T, core, sd, vc, o, core_min, side_max, time_ks, min_run, max_rows,
                              isolate, novel_max, echo_min_km)
            if use_recur:
                a, b = max(t - recur_half, 0), min(t + recur_half + 1, T)
                O = (cum[b] - cum[a]) / (b - a)
                cO, sO, _, _ = st.core_side(None, valid[t], o=O)
                cand = (cO >= recur_min) & (sO <= recur_ratio * cO) & (sO <= recur_side_abs) & (vc > 0)
                fr = _runs(cand, int(round(recur_run_km / DR)), max_rows, isolate) if cand.any() else None
                if fr is not None:
                    fr &= sd <= recur_side_t                     # dry around in this frame too
                    if fr.any():
                        fr = _thin_only(fr, wet[t], st, recur_thin, recur_share)
                    if fr is not None:
                        flag = fr if flag is None else (flag | fr)
            if known is not None and si < len(known):
                # detector 3: echo on fixed recurring interference rays (history map), not inside
                # rain in this frame, on thin objects lying mostly on the ray
                fk = known[si] & (o > 0) & (sd <= known_side_t)
                if fk.any():
                    fk = _thin_only(fk, wet[t], st, known_thin, known_share)
                    if fk is not None and known_min_km > 0:
                        # a streak, not a lone fragment: >= known_min_km of echo along this ray
                        n, lab = _known_cc(known, si)
                        yy, xx = np.nonzero(fk)
                        rb = np.zeros((n, NR), bool)
                        rb[lab[yy, xx], xx] = True
                        good = rb.sum(1) * DR >= known_min_km
                        good[0] = False
                        fk = fk & good[lab]
                        if not fk.any():
                            fk = None
                    if fk is not None:
                        flag = fk if flag is None else (flag | fk)
            if flag is None or not flag.any():
                continue
            fl = flag.ravel()
            w = wps[t][st.idx[wps[t]] >= 0]
            out[si, t, w[fl[st.idx[w]]]] = True
    return out.reshape(len(sites), T, H, W)
