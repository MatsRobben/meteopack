"""Stages 1-3 of the cleaner (research proto/spikes "v10"): per-frame object evidence, the full-radial spike
stage and the frozen-half copy.

    st = stage(codes, times)                         objects, evidence, 4 km flows, streak seeds, frozen flags
    v9, info = spike_stage(codes, st, rmo)           removed objects zeroed, spike rays handled (spikes.py)
    out = frozen_apply(codes, st, v9)                frozen halves copy the previous output

Spike stage: seeds per NL radar = thin streak pixels (streaks_site.py) | wedge seeds (wedge.py: rays crossing
rain, wide wedges); each seed line becomes a ray handled along its whole radial out to 335 km: clear air ->
removed; rain -> the ray's measured power is subtracted (never below the rain beside the strip, never zeroed);
thick showers on a strip are kept; rays are carried to t +- 2 where the frame shows ray evidence. Streak
pixels outside any strip are zeroed. Codes >= 65534 pass unchanged; values are never raised.
"""
from __future__ import annotations

import numpy as np

from . import objects as FAST
from . import spikes as SP
from . import streaks_site as SS
from . import wedge as WG
from .stale import HALVES, frozen_flags

MISSING = 65534


def stage(codes, times=None, streak_kw=None):
    """Per-frame stage: labs [(lab, n)], ev [(n, members)], flr {t: 4 km flow}, sms (sites, T, H, W) streak
    seeds per radar, sm (their OR), flags (frozen halves)."""
    labs, ev, flr = FAST.per_frame(codes, times)
    sms = SS.streak_mask_sites(codes, times, **(streak_kw or {}))
    return {"labs": labs, "ev": ev, "flr": flr, "sm": sms.any(0), "sms": sms, "flags": frozen_flags(codes)}


def spike_stage(codes, st, rmo, *, wedge=True, wedge_kw=None, spike_kw=None):
    """Objects in rmo zeroed, then the full-radial spike stage. Returns (v9, info). Pops st["sms"]."""
    valid = codes < MISSING
    base = np.where(rmo & valid, 0, codes).astype(codes.dtype)          # objects removed
    seeds = st.pop("sms")                                               # (n_sites, T, H, W), extended in place
    if wedge:
        seeds |= WG.wedge_mask(codes, **(wedge_kw or {}))
    est, hit, rays = SP.spike_fix(codes, seeds, ref=base, ret_rays=True, **(spike_kw or {}))
    v9 = np.where(st["sm"] & ~hit & valid, 0, base).astype(codes.dtype)
    v9 = SP.apply(v9, est, hit, codes)
    return v9, {"est": est, "hit": hit, "rays": rays, "seeds": seeds}


def frozen_apply(codes, st, v9):
    """The research bg_apply with c = 1e9 (no bgsub cap): only the frozen-half copy."""
    T = len(codes)
    valid = codes < MISSING
    out = v9.copy()
    for t in range(1, T):
        for bit, sl in HALVES:
            if st["flags"][t] & bit:
                same = codes[t][sl] == codes[t - 1][sl]
                out[t][sl] = np.where(same, out[t - 1][sl], out[t][sl])
    return np.where(valid, out, codes).astype(codes.dtype)
