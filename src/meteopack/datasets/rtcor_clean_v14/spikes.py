"""Full-radial interference-ray stage for old-chain RTCOR (v8 study, round 4).

    rays = find_rays(seed_mask_t)                          # list of Ray from any per-frame seed mask (bool H x W)
    est, hit = spike_fix(codes, seeds, ref=None)           # (T,H,W) uint16 replacement values and the ray strips
    out = apply(out, est, hit, codes)                      # = where(hit, min(out, est), out); mask unchanged

A seed is a set of pixels that some detector flagged as interference (here: proto/fast/streaks.py, the v8/v9b
detector, plus the wedge test in wedge.py). Seeds are turned into rays: per NL radar, a contiguous azimuth interval
(centre c, half-width hw) holding the seed pixels, assigned to the radar from which the pixels look narrowest.
Each ray is then handled along its WHOLE radial, from the radar to the edge of the image:
  strip   pixels whose perpendicular distance to the ray's centre line is <= r tan(hw) + rim (rim 1 px: the rims),
          out to the radar's range (r_max = 335 km; beyond it the composite shows other radars);
  est     per strip pixel at range r, from the reference frame at 6 points at the same range just outside the
          strip, at c +- (hw + (rim + d) / r) for d = 2, 4, 6 px: the median of the wet points when >= 2 are wet,
          else 0 (missing/outside points ignored; fewer than 3 usable points -> pixel left unchanged);
  output  min(value, est): in clear air the neighbours are dry, est = 0 and the ray goes (weak tail and near-radar
          part included); through rain est is the rain beside the ray, so the rain stays and the ray's excess goes.
          Never raised; codes >= 65534 never touched.
  in rain the ray's power is subtracted rather than the rain flattened (subtract_ray): the ray's reflectivity is
          measured on its clear-air pixels in the same frame and scaled with r^2 (interference power does not fall
          with range); the pixel keeps max(neighbours' level, code^1.6 - k r^2)^(1/1.6), so convective cores on the
          ray keep their values.
Showers on the ray: a strip pixel about to be zeroed is kept when its echo object is thick (a pixel >= 3 px from
dry) unless its strip pixels are mostly this frame's seed pixels (a detected wide wedge, also touching rain).
Time: a ray found at t is carried to t +- 1..hold where the frame itself shows ray evidence in that strip: >= ev_bins
2-km range bins in which most strip echo is clear-air echo (est == 0) or stands >= ev_ratio x above est. This removes
half-masked stubs in the frames before/after a detection without masking a strip blindly.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import streaks as SK

MISSING = 65534
NONE = 65535                             # est: no estimate
H, W = 765, 700
SITES = SK.NL_SITES                      # Den Helder, Herwijnen (row, col)
DEFAULTS = dict(rim=1.5, offs=(2.0, 4.0, 6.0), hw_pad=0.3, max_w=5.0, n_min=8, ext_min=20.0,
                hold=2, ev_bins=8, ev_ratio=2.0, ev_pad=20.0, r_max=335.0, wet_min=2,
                thick=3.0, seed_share=0.5, subtract=True, b=1.6, zq=75)
_GEO = {}


@dataclass
class Ray:
    site: int
    c: float          # centre azimuth, deg (warpPolar convention: clockwise from east in image coords)
    hw: float         # half-width, deg
    npx: int = 0
    spread: float = 0.0
    r0: float = 0.0   # range (px) of the seed pixels
    r1: float = 1e9


def geo(si):
    """per-site flat arrays: az (deg), r (px), az order for interval lookup."""
    if si not in _GEO:
        cy, cx = SITES[si]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
        dy, dx = yy - cy, xx - cx
        az = (np.degrees(np.arctan2(dy, dx)) % 360.0).ravel()
        r = np.hypot(dy, dx).ravel()
        order = np.argsort(az, kind="stable")
        near = np.flatnonzero(r < 60)
        _GEO[si] = dict(az=az.astype(np.float32), r=r.astype(np.float32), order=order, az_sorted=az[order],
                        near=near, cy=cy, cx=cx)
    return _GEO[si]


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def find_rays(seed, site=None, *, max_w=5.0, n_min=8, ext_min=20.0, hw_pad=0.25, **_):
    """Rays from a seed mask (bool (H, W) or flat pixel indices) flagged by the polar test of radar `site`
    (None: try every radar and keep, per seed line, the radar from which it looks narrowest)."""
    pix = np.flatnonzero(seed) if seed.dtype == bool else np.asarray(seed)
    if len(pix) < n_min:
        return []
    cand = []
    for si in (range(len(SITES)) if site is None else [site]):
        g = geo(si)
        a, r = g["az"][pix], g["r"][pix]
        far = r >= 5
        b = (a * 2).astype(np.int64) % 720
        h = np.bincount(b[far], minlength=720) > 0
        h = h | np.roll(h, 1) & np.roll(h, -1)                  # bridge 1-bin gaps
        if h.all():
            continue
        s0 = int(np.flatnonzero(~h)[0])                         # rotate so a gap is at 0 (circular labelling)
        hr = np.roll(h, -s0)
        edges = np.flatnonzero(np.diff(np.r_[0, hr.astype(np.int8), 0]))
        for lo, hi in zip(edges[::2], edges[1::2]):
            sel = far & (((b - s0) % 720 >= lo) & ((b - s0) % 720 < hi))
            if sel.sum() < n_min:
                continue
            rs = r[sel]
            if rs.max() - rs.min() < ext_min:
                continue
            ref = ((lo + s0) % 720) / 2.0
            da = _wrap(a[sel] - ref)
            w = np.clip(rs / 60.0, 0.2, 1.0)                     # far pixels define the angle best
            p5, p95 = np.percentile(da, [2, 98])
            wid = p95 - p5
            if wid > max_w:
                continue
            c = (ref + float(np.sum(da * w) / np.sum(w))) % 360.0
            spread = np.radians(wid) * float(np.median(rs))       # transverse extent as seen from this radar
            cand.append((Ray(si, c, wid / 2 + hw_pad, int(sel.sum()), spread, float(rs.min()), float(rs.max())), set(pix[sel].tolist()) if len(SITES) > 1 else None))
    # a seed line is a ray of the radar from which it looks narrowest
    keep = [True] * len(cand)
    for i in range(len(cand)):
        for j in range(len(cand)):
            if i != j and keep[i] and keep[j] and cand[i][0].site != cand[j][0].site:
                inter = len(cand[i][1] & cand[j][1])
                if inter >= 0.5 * min(len(cand[i][1]), len(cand[j][1])):
                    if cand[i][0].spread > cand[j][0].spread:
                        keep[i] = False
    return [c[0] for c, k in zip(cand, keep) if k]


def strip(ray, rim=1.0, r_max=335.0, wet_min=2,
                thick=3.0, seed_share=0.5, subtract=True, b=1.6, zq=75):
    """flat pixel indices of the ray strip (whole radial, out to the radar's range r_max px) and their range (px)."""
    g = geo(ray.site)
    marg = ray.hw + 3.0
    lo, hi = (ray.c - marg) % 360.0, (ray.c + marg) % 360.0
    s = g["az_sorted"]
    if lo <= hi:
        idx = g["order"][np.searchsorted(s, lo):np.searchsorted(s, hi)]
    else:
        idx = np.concatenate([g["order"][np.searchsorted(s, lo):], g["order"][:np.searchsorted(s, hi)]])
    nr = g["near"]
    idx = np.concatenate([idx, nr[np.abs(_wrap(g["az"][nr] - ray.c)) >= marg]])   # near-radar pixels outside the window
    d = np.radians(np.abs(_wrap(g["az"][idx] - ray.c)))
    r = g["r"][idx]
    ok = (d < np.pi / 2) & (r * np.sin(d) <= r * np.tan(np.radians(ray.hw)) + rim) & (r >= 0.5) & (r <= r_max)
    return idx[ok], r[ok]


def estimate(ray, idx, r, F, rim=1.0, offs=(2.0, 4.0, 6.0), wet_min=2):
    """rain beside the strip at each pixel's range: of the 6 points of frame F, the median of the wet ones when
    >= wet_min are wet (rain on or beside the ray: never zeroed), else 0 (clear air); NaN where < 3 usable points."""
    g = geo(ray.site)
    hw = np.radians(ray.hw)
    c = np.radians(ray.c)
    vals = []
    for d in offs:
        for s in (-1.0, 1.0):
            th = c + s * (hw + (rim + d) / np.maximum(r, 1.0))
            y = np.rint(g["cy"] + r * np.sin(th)).astype(np.int64)
            x = np.rint(g["cx"] + r * np.cos(th)).astype(np.int64)
            ins = (y >= 0) & (y < H) & (x >= 0) & (x < W)
            v = F[np.where(ins, y, 0), np.where(ins, x, 0)].astype(np.float32)
            vals.append(np.where(ins & (v < MISSING), v, np.nan))
    V = np.stack(vals)
    n = (~np.isnan(V)).sum(0)
    Vw = np.where(V > 0, V, np.nan)                                  # wet samples
    nw = (~np.isnan(Vw)).sum(0)
    Vw = np.sort(Vw, 0)                                              # NaN last
    i0, i1 = np.maximum((nw - 1) // 2, 0), np.maximum(nw // 2, 0)
    cols = np.arange(V.shape[1])
    med = 0.5 * (Vw[i0, cols] + Vw[i1, cols])
    est = np.where(nw >= wet_min, med, 0.0)
    return np.where(n >= 3, est, np.nan)


def subtract_ray(e, idx, r, raw, b=1.6, zq=75):
    """Inside rain (e > 0): remove the ray's power instead of flattening the rain to its neighbours.
    Interference adds a received power that does not depend on range, so after range correction its reflectivity
    grows as r^2: z_int(r) = k r^2, with z = code^b (Z-R power law; constants cancel). k is the zq-th percentile of
    code^b / r^2 over this frame's clear-air ray pixels (strip echo with e == 0). Rain pixel on the ray:
    value = max(e, (code^b - k r^2)^(1/b)); a convective core far above the ray keeps its value, weak rain where the
    ray dominates goes to the neighbours' level. Without >= 10 clear-air ray pixels in this frame the rain on the ray
    is left unchanged (no measured ray power)."""
    v = raw.reshape(-1)[idx].astype(np.float32)
    ca = (e == 0) & (v > 0) & (v < MISSING) & (r >= 5)
    rain = (e > 0) & (v < MISSING)
    if not rain.any():
        return e
    if ca.sum() < 10:                                  # ray not seen in clear air in this frame: rain untouched
        return np.where(rain, np.nan, e)
    k = np.percentile(v[ca] ** b / r[ca] ** 2, zq)
    z = np.maximum(v ** b - k * r ** 2, 0)
    return np.where(rain, np.maximum(e, z ** (1.0 / b)), e)


def evidence(idx, r, raw, est, ev_ratio=2.0, rr=(0.0, 1e9)):
    """number of 2-km range bins (within the range interval rr) in which most strip echo is ray-like (clear air,
    or >= ev_ratio x est)."""
    v = raw.reshape(-1)[idx].astype(np.float32)
    wet = (v > 0) & (v < MISSING) & ~np.isnan(est) & (r >= rr[0]) & (r <= rr[1])
    if not wet.any():
        return 0
    e = np.nan_to_num(est)
    rl = wet & ((e == 0) | (v >= ev_ratio * np.maximum(e, 1)))
    b = (r / 2).astype(np.int64)
    nw = np.bincount(b[wet], minlength=b.max() + 1)
    nr = np.bincount(b[rl], minlength=b.max() + 1)
    return int(((nr * 2 > nw) & (nw > 0)).sum())


def merge(rays):
    """union of overlapping rays of the same radar."""
    out = []
    for ray in sorted(rays, key=lambda x: (x.site, x.c)):
        if out and out[-1].site == ray.site:
            p = out[-1]
            lo1, hi1 = p.c - p.hw, p.c + p.hw
            d = _wrap(ray.c - p.c)
            lo2, hi2 = p.c + d - ray.hw, p.c + d + ray.hw
            if lo2 <= hi1 and hi2 >= lo1:
                lo, hi = min(lo1, lo2), max(hi1, hi2)
                out[-1] = Ray(p.site, ((lo + hi) / 2) % 360.0, (hi - lo) / 2, p.npx + ray.npx, max(p.spread, ray.spread),
                              min(p.r0, ray.r0), max(p.r1, ray.r1))
                continue
        out.append(ray)
    return out


def spike_fix(codes, seeds, ref=None, *, ret_rays=False, **kw):
    """codes (T,H,W) raw window; seeds (n_sites,T,H,W) bool seed pixels per radar (or (T,H,W): radar found per
    line); ref (T,H,W) frames the estimate is read from (default: codes).
    Returns est (T,H,W) uint16 (NONE = 65535 outside strips / no estimate) and hit (T,H,W) bool."""
    p = {**DEFAULTS, **kw}
    codes = np.asarray(codes)
    ref = codes if ref is None else ref
    T = len(codes)
    if seeds.ndim == 3:
        rays = [merge(find_rays(seeds[t], **p)) if seeds[t].any() else [] for t in range(T)]
    else:
        rays = [merge([r for si in range(len(seeds)) if seeds[si, t].any() for r in find_rays(seeds[si, t], si, **p)])
                for t in range(T)]
    est = np.full(codes.shape, NONE, np.uint16)
    hit = np.zeros(codes.shape, bool)
    used = [[] for _ in range(T)]
    strips = {}
    for t in range(T):
        cands = [(ray, True) for ray in rays[t]]
        for k in range(1, p["hold"] + 1):
            for u in (t - k, t + k):
                if 0 <= u < T:
                    cands += [(ray, False) for ray in rays[u]]
        done = []
        for ray, own in cands:
            if any(q.site == ray.site and abs(_wrap(q.c - ray.c)) < 0.5 * (q.hw + ray.hw) for q in done):
                continue                                           # same ray already applied in this frame
            if id(ray) not in strips:
                strips[id(ray)] = strip(ray, p["rim"], p["r_max"])
            idx, r = strips[id(ray)]
            e = estimate(ray, idx, r, ref[t], p["rim"], p["offs"], p["wet_min"])
            if not own and evidence(idx, r, codes[t], e, p["ev_ratio"], (ray.r0 - p["ev_pad"], ray.r1 + p["ev_pad"])) < p["ev_bins"]:
                continue
            done.append(ray)
            if p["subtract"]:
                e = subtract_ray(e, idx, r, codes[t], p["b"], p["zq"])
            ef = est[t].reshape(-1)
            cur = ef[idx]
            ef[idx] = np.where(np.isnan(e), cur, np.minimum(cur, np.rint(np.nan_to_num(e)))).astype(np.uint16)
            hit[t].reshape(-1)[idx[~np.isnan(e)]] = True
        used[t] = done
        if done and p["thick"] > 0:
            sd = seeds[t] if seeds.ndim == 3 else seeds[:, t].any(0)
            protect(codes[t], est[t], hit[t], sd, p["thick"], p["seed_share"])
    return (est, hit, used) if ret_rays else (est, hit)


def protect(f, est, hit, seed, thick=3.0, seed_share=0.5, subtract=True, b=1.6, zq=75):
    """In place: strip pixels about to be zeroed (est == 0) are kept (hit off) when their echo object is thick
    (a pixel >= thick px from dry: a shower, not a ray dash), unless its pixels about to be zeroed are mostly seed
    pixels of this frame (a wide detected wedge is thick, also when it touches rain, but it is its own seed)."""
    z = hit & (est == 0) & (f > 0) & (f < MISSING)
    if not z.any():
        return
    ys, xs = np.nonzero(z)
    m = 12
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, H)
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, W)
    sub = f[y0:y1, x0:x1]
    wet = ((sub > 0) & (sub < MISSING)).astype(np.uint8)
    n, lab = cv2.connectedComponents(wet, connectivity=8)
    edt = cv2.distanceTransform(np.pad(wet, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
    l = lab.ravel()
    big = np.bincount(l, weights=(edt >= thick).ravel(), minlength=n) > 0
    zs = z[y0:y1, x0:x1].ravel()
    nz = np.bincount(l, weights=zs, minlength=n)
    sh = np.bincount(l, weights=(seed[y0:y1, x0:x1].ravel() & zs), minlength=n) / np.maximum(nz, 1)
    keep = big & (sh < seed_share)
    keep[0] = False
    kp = keep[lab] & z[y0:y1, x0:x1]
    hit[y0:y1, x0:x1][kp] = False


def apply(out, est, hit, codes):
    """out lowered to est on the strips; never raised; radar mask unchanged."""
    v = codes < MISSING
    return np.where(hit & v, np.minimum(out, est), out).astype(out.dtype)
