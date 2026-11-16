"""proto/spikes2 shared helpers: polar geometry per NL radar, case loading (raw + v11), rendering."""
# Ported from research proto/spikes2/s2common.py (md5 73ff28a657177b98307aec82818372ea),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations


import numpy as np


MISSING, OUTSIDE = 65534, 65535
H, W = 765, 700
SITES = ((331.495, 333.220), (458.153, 368.942))      # Den Helder, Herwijnen (row, col), proto/fast/streaks.py
SITE_NAMES = ("Den Helder", "Herwijnen")
DA = 0.25                                             # polar azimuth step (deg)
NA = int(round(360 / DA))
NR = 340                                              # 1-km range bins, 0..339 km


_G = {}


def polar_geo(si):
    """per site: idx (NA, NR) flat pixel index of each polar cell (nearest pixel, -1 off-grid); back (H*W,) polar
    cell of each pixel (a*NR + r, -1 beyond NR); paz, pr (H*W,) exact azimuth (deg) and range (km) per pixel."""
    if si not in _G:
        cy, cx = SITES[si]
        a = (np.arange(NA) + 0.5) * DA
        r = np.arange(NR) + 0.5
        th = np.radians(a)[:, None]
        y = np.rint(cy + r[None] * np.sin(th)).astype(np.int64)
        x = np.rint(cx + r[None] * np.cos(th)).astype(np.int64)
        ok = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        idx = np.where(ok, y * W + x, -1)
        yy, xx = np.mgrid[0:H, 0:W]
        dy, dx = (yy - cy).ravel(), (xx - cx).ravel()
        paz = (np.degrees(np.arctan2(dy, dx)) % 360.0).astype(np.float32)
        pr = np.hypot(dy, dx).astype(np.float32)
        ai = np.minimum((paz / DA).astype(np.int64), NA - 1)
        ri = pr.astype(np.int64)
        back = np.where(ri < NR, ai * NR + ri, -1)
        _G[si] = dict(idx=idx, ok=ok, back=back, paz=paz, pr=pr)
    return _G[si]


def to_polar(frame, si, fill=OUTSIDE):
    g = polar_geo(si)
    f = frame.reshape(-1)
    return np.where(g["ok"], f[np.maximum(g["idx"], 0)], fill)
