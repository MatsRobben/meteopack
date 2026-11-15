"""Frozen (copied) halves of old-chain RTCOR frames, and small helpers built on them.

In the old chain, part of the composite is often an exact copy of the previous 5-minute frame (2022: north /
Den Helder half 28-32 % of frames with echo, whole frame 11-13 %). The rain there stands still for one frame
and then jumps. A copied half is never temporal evidence, and its cleaned output is the previous frame's.

    frozen_flags(codes) -> uint8 (T,)   bit 1: north half (rows 0-379) copied from the previous frame
                                        bit 2: south half (rows 380-764) copied (3 = whole frame)
A half counts as copied when >= 300 of its pixels are wet and > 85 % of them have exactly the previous
frame's code. Frame 0 cannot be judged and gets 0.
"""
from __future__ import annotations

import numpy as np

MISSING = 65534
H = 765
HALVES = ((1, slice(0, 380)), (2, slice(380, 765)))


def frozen_flags(codes, prev=None, min_wet=300, share=0.85):
    codes = np.asarray(codes)
    seq = codes if prev is None else np.concatenate([np.asarray(prev)[None], codes])
    flags = np.zeros(len(seq), np.uint8)
    for t in range(1, len(seq)):
        for bit, sl in HALVES:
            a, b = seq[t][sl], seq[t - 1][sl]
            w = (a > 0) & (a < MISSING)
            if w.sum() >= min_wet and (a[w] == b[w]).mean() > share:
                flags[t] |= bit
    return flags if prev is None else flags[1:]


def usable(flags, T):
    """(T, H) bool: rows usable as evidence (not part of a frozen copy)."""
    ok = np.ones((T, H), bool)
    for u in range(T):
        for bit, sl in HALVES:
            if flags[u] & bit:
                ok[u, sl] = False
    return ok


def removed_mask(codes, st, RM):
    """(T, H, W) bool: pixels of the objects removed by the object decision RM."""
    out = np.zeros(codes.shape, bool)
    for t, ((lab, n), rm) in enumerate(zip(st["labs"], RM)):
        if n and rm.any():
            out[t] = np.r_[False, rm][lab]
    return out
