"""Polar line candidates per NL radar (dev core of spikes2).

Polar grid per radar: NA = 1440 azimuth rows of 0.25 deg x NR = 340 range bins of 1 km (nearest RTCOR pixel).
A spike / strobe is a vertical line in this grid (constant azimuth), with a top-hat cross-section in azimuth.

    cands = line_cands(Pw, Pv, PL)   # per frame and radar: list of dict(a0, a1, r0, r1, ...) (rows a0..a1 incl.)
"""
# Ported from research proto/spikes2/polar.py (md5 88667960f454bde1f2d8c59c6a4d0278),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations

import cv2
import numpy as np

from .spikes2_common import DA, NA, NR, MISSING

WS = (3, 5, 9, 15)          # core widths in rows (0.75, 1.25, 2.25, 3.75 deg)
GAP = 1                     # rows between core and side windows (the ray's rim)


class RowSum:
    """row-window sums (circular in azimuth) from one cumulative sum: rs(lo, hi)[a] = sum X[a+lo .. a+hi]."""
    PAD = 40

    def __init__(self, X):
        p = self.PAD
        Xp = np.concatenate([X[-p:], X, X[:p]], 0)
        self.C = np.zeros((Xp.shape[0] + 1, X.shape[1]), np.float32)
        np.cumsum(Xp, 0, dtype=np.float32, out=self.C[1:])

    def __call__(self, lo, hi):
        p = self.PAD
        return self.C[p + hi + 1:p + hi + 1 + NA] - self.C[p + lo:p + lo + NA]


def rowsum(X, lo, hi):
    return RowSum(X)(lo, hi)


def rbox(X, k):
    """centered running mean along range (axis 1), window k bins."""
    return cv2.blur(X, (k, 1), borderType=cv2.BORDER_REPLICATE)


def line_maps(wet, val, L, ws=WS, side_min=4, r_min=25):
    """per width: clear-air line cells CA (core wet share >= 0.5, both sides <= 0.15) and contrast cells EX
    (core wet share >= 0.7 and core mean log1p(code) above BOTH side means by >= 0.35). Returns union masks and the
    best width index per cell (uint8, 255 = none)."""
    out_ca = np.zeros(wet.shape, bool)
    out_ex = np.zeros(wet.shape, bool)
    best = np.full(wet.shape, 255, np.uint8)
    c0 = r_min
    RV, RW, RL = RowSum(val[:, c0:]), RowSum(wet[:, c0:]), RowSum(L[:, c0:])
    for i, w in enumerate(ws):
        h = (w - 1) // 2
        s = max(side_min, w)
        nv = RV(-h, h)
        inv = 1.0 / np.maximum(nv, 1)
        cw = RW(-h, h) * inv
        okv = nv >= 0.6 * w
        sw, sl = [], []
        for lo, hi in ((-h - GAP - s, -h - GAP - 1), (h + GAP + 1, h + GAP + s)):
            n = RV(lo, hi)
            okv &= n >= 0.5 * s
            invs = 1.0 / np.maximum(n, 1)
            sw.append(RW(lo, hi) * invs)
            sl.append(RL(lo, hi) * invs)
        ca = okv & (cw >= 0.5) & (np.maximum(sw[0], sw[1]) <= 0.15)
        exm = okv & (cw >= 0.7)
        ex = exm & ((RL(-h, h) * inv - np.maximum(sl[0], sl[1])) >= 0.35)
        sub_best = best[:, c0:]
        new = (ca | ex) & (sub_best == 255)
        sub_best[new] = i
        out_ca[:, c0:] |= ca
        out_ex[:, c0:] |= ex
    out_ca[:, :r_min] = False
    out_ex[:, :r_min] = False
    return out_ca, out_ex, best


def components(mask, min_len=20, max_rows=24):
    """connected (8-conn) components of a polar mask that are elongated along range. Returns list of
    (rows (sorted unique, may wrap), r0, r1, n)."""
    m = mask.astype(np.uint8)
    # drop cells in azimuth runs wider than max_rows (rain areas): a line touching a rain cell stays its own component
    wide = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((max_rows + 1, 1), np.uint8))
    m = m & (1 - wide)
    # bridge 2-bin gaps along range (dashes) before labelling
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((1, 5), np.uint8))
    # circular azimuth: label on a padded copy
    pad = 40
    mp = np.concatenate([m[-pad:], m, m[:pad]], 0)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mp, connectivity=8)
    out = []
    seen = set()
    for j in range(1, n):
        x, y, w, h, area = stats[j]
        if w < min_len:
            continue
        if not (pad <= y + h // 2 < NA + pad):
            continue                                   # the copy in the padding: handled by the central one
        sub = lab[y:y + h, x:x + w] == j
        if h <= max_rows + 8:
            pieces = [(0, h)]
        else:
            # a line chained to other echo: keep the azimuth rows that carry a long run along range
            rc = sub.sum(1)
            good = rc >= 0.6 * min_len
            pieces = []
            k = 0
            while k < h:
                if good[k]:
                    k1 = k
                    while k1 + 1 < h and good[k1 + 1]:
                        k1 += 1
                    if k1 - k + 1 <= max_rows:
                        pieces.append((max(k - 1, 0), min(k1 + 2, h)))
                    k = k1 + 1
                else:
                    k += 1
        for (q0, q1) in pieces:
            ss = sub[q0:q1]
            cols = np.flatnonzero(ss.any(0))
            if len(cols) == 0 or cols[-1] - cols[0] + 1 < min_len:
                continue
            ss = ss[:, cols[0]:cols[-1] + 1]
            rows = (np.arange(y + q0, y + q1) - pad) % NA
            key = (int(rows[len(rows) // 2]), int(x + cols[0]))
            if key in seen:
                continue
            seen.add(key)
            out.append(dict(rows=rows, r0=int(x + cols[0]), r1=int(x + cols[-1]), n=int(ss.sum()), sub=ss))
    return out
