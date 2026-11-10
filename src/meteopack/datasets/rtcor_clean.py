"""RTCOR clutter cleaner, version 5.

The KNMI RTCOR archive changed processing chain on 2023-01-31 10:45 UTC. Frames of the old
chain hold much more non-meteorological echo: sea clutter and anomalous propagation,
interference spikes and wind farms. This module removes it with one procedure for every
period, so that the old chain looks like the new one. Nothing is dropped: clutter becomes
dry, spike pixels are refilled, and pixels outside the common radar footprint become
out-of-image. Frame t is cleaned with the help of frames t-1 and t+1.

1. Motion. Dense Lucas-Kanade optical flow (pySTEPS) on a 2 km copy of each frame.
2. Spikes and static clutter. A ray-based detector per radar site (in the spirit of BALTRAD
   ropo) marks thin radial lines (>= 40 km long, <= 2.5 km wide) whose neighbouring rays are
   dry. In old-chain frames the pixels of a static clutter map (wind farms, permanent
   interference) are added. These pixels are refilled with the mean of a spatial estimate
   (normalised convolution, sigma 2 km) and a temporal estimate (t-1 and t+1 advected along
   the flow): a spike through rain becomes rain, a spike in clear air becomes dry.
3. Clutter probability. Per 16 x 16 km tile, a logistic model on seven features of the wet
   pixels: fragments per area, mean log rate, neighbour jump, local fill, log wet area,
   motion-compensated correlation with t-1 and t+1, and log speed. The tile probability is
   averaged over 3 x 3 tiles and t-1..t+1 and interpolated bilinearly to pixels. Echo whose
   probability exceeds the threshold is set to dry.
4. Scene threshold. The threshold is 0.7. It is 0.4 in clutter-dominated warm-season scenes:
   April-September with a scene clutter fraction (wet-area-weighted mean tile probability)
   >= 0.4, or June-August nights (18-06 UTC) with >= 0.3. Never in October-March, so winter
   sea-effect showers are not hit.
5. Protection. Objects of >= 15 km2 are kept when they travel with the flow (>= 60 % present
   in the advected t-1 and t+1, >= 1.5 km per 5 min, fitting better advected than left in
   place) or have a filled, smooth texture (>= 50 % interior pixels, mean neighbour jump
   <= 0.25), or when they overlap (>= 30 %) such an object of the previous frame.
6. Intense blobs. In clutter-dominated scenes (step 4), objects peaking at >= 20 mm/h with a
   hard edge (>= 10 % of edge pixels >= 5 mm/h) that do not travel with the flow are removed.
7. Specks. Objects of <= 3 km2 without echo nearby along the flow at t-1 and at t+1 are removed.
8. Hot values. Pixels above 30 mm/h exceeding the 5 x 5 median of log(1+R) by more than 1.5
   are capped at that level.
9. Footprint. Pixels outside the common (old-chain) footprint become out-of-image.

The model coefficients, the static clutter map, the footprint and the radar positions are
fixed inputs in assets/rtcor_clean_v5.npz (scripts/rtcor_clean_assets.py builds it from the
research outputs). Protection lineage (step 5) starts empty at the start of every block.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

VERSION = 'v5'
ASSETS = Path(__file__).with_name('assets') / 'rtcor_clean_v5.npz'
MISSING, OUTSIDE = 65534, 65535
MMH_PER_CODE = 0.12                       # one RTCOR code is 0.01 mm per 5 min
NEW_CHAIN = np.datetime64('2023-01-31T10:45', 'ns')

TILE = 16                                 # km
THR, THR_SCENE = 0.7, 0.4                 # clutter probability thresholds (normal, scene)
SCENE_MIN, SCENE_MIN_NIGHT = 0.4, 0.3     # scene clutter fraction for THR_SCENE
MIN_OBJECT = 15                           # km2, smallest protected object
BLOB_PEAK, BLOB_EDGE_RATE, BLOB_EDGE_SHARE = 20.0, 5.0, 0.1
SPECK = 3                                 # km2
HOT_MIN, HOT_LOG = 30.0, 1.5              # mm/h, log(1+R) above the 5 x 5 median
NAZ, DR, NR = 360, 2.0, 160               # spike polar grid: 1 deg, 2 km, 320 km
MINRUN = 20                               # bins (40 km)
MAXW = 2.5                                # km, median distance to dry along a spike


@dataclass(frozen=True, eq=False)
class Assets:
    coef: np.ndarray                      # bias, then one weight per standardised feature
    mu: np.ndarray
    sd: np.ndarray
    feats: tuple[str, ...]
    static: np.ndarray                    # bool (ny, nx), old-chain static clutter
    cover: np.ndarray                     # bool (ny, nx), common footprint
    sites: tuple[tuple[float, float], ...]  # radar positions as fractional (row, col)

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.cover.shape)


@lru_cache(maxsize=4)
def load_assets(path: str | Path = ASSETS) -> Assets:
    z = np.load(path)
    return Assets(coef=z['coef'].astype(float), mu=z['mu'].astype(float), sd=z['sd'].astype(float),
                  feats=tuple(str(f) for f in z['feats']), static=z['static'].astype(bool),
                  cover=z['cover'].astype(bool),
                  sites=tuple((float(r), float(c)) for r, c in z['site_rc']))


def fingerprint() -> dict:
    """Hashes of the code and fixed inputs, for build manifests."""
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    return {'version': VERSION, 'code_sha256': sha(__file__), 'assets_sha256': sha(ASSETS)}


# ---------------------------------------------------------------- helpers
def _box(a, s):
    return ndi.uniform_filter(a.astype(np.float32), size=s, mode='constant')


def _tiles(a):
    ny, nx = a.shape[0] // TILE * TILE, a.shape[1] // TILE * TILE
    return (a[:ny, :nx].reshape(ny // TILE, TILE, nx // TILE, TILE).swapaxes(1, 2)
            .reshape(ny // TILE, nx // TILE, TILE * TILE))


@lru_cache(maxsize=4)
def _grid(shape):
    return np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)


def _warp(field, v, k):
    """`field` advected k steps forward along the flow v (backward semi-Lagrangian, nearest)."""
    yy, xx = _grid(field.shape)
    return ndi.map_coordinates(np.nan_to_num(field, nan=0.0), [yy - k * v[1], xx - k * v[0]],
                               order=0, mode='constant', cval=0.0)


def motion(R):
    """Dense Lucas-Kanade flow (pixels per 5 min) for each frame from (t-1, t), on a 2 km grid."""
    import cv2  # noqa: F401  (pySTEPS needs OpenCV; fail loudly instead of returning zero flow)
    from pysteps.motion.lucaskanade import dense_lucaskanade
    T, ny, nx = R.shape
    ny2, nx2 = ny // 2 * 2, nx // 2 * 2
    V = np.zeros((T, 2, ny, nx), np.float32)
    for t in range(1, T):
        a = R[t - 1:t + 1, :ny2, :nx2].reshape(2, ny2 // 2, 2, nx2 // 2, 2).mean((2, 4))
        db = np.where(a >= 0.1, 10 * np.log10(np.maximum(a, 0.1)), -15.0)
        try:
            v = dense_lucaskanade(db, verbose=False)
        except Exception:                 # too few features to track: no motion
            v = np.zeros((2,) + db.shape[1:], np.float32)
        V[t, :, :ny2, :nx2] = np.repeat(np.repeat(v, 2, 1), 2, 2) * 2.0
    V[0] = V[1]
    return V


# ---------------------------------------------------------------- spikes
@lru_cache(maxsize=4)
def _spike_geometry(shape, sites):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    geom = []
    for r0, c0 in sites:
        dy, dx = -(yy - r0), xx - c0                       # km, north up
        az = (np.degrees(np.arctan2(dx, dy)) % 360).astype(np.int32) % NAZ
        rb = (np.hypot(dx, dy) / DR).astype(np.int32)
        ok = rb < NR
        geom.append((np.where(ok, az * NR + rb, -1).ravel(), ok.ravel()))
    return geom


def detect_spikes(codes, sites):
    """Boolean mask of spike pixels in one frame of RTCOR codes."""
    valid = (codes < MISSING).ravel()
    wet = ((codes > 0) & (codes < MISSING)).ravel()
    out = np.zeros(codes.size, bool)
    for idx, ok in _spike_geometry(codes.shape, sites):
        sel = ok & valid
        nv = np.bincount(idx[sel], minlength=NAZ * NR).reshape(NAZ, NR)
        nwt = np.bincount(idx[sel & wet], minlength=NAZ * NR).reshape(NAZ, NR)
        if nwt.sum() < 30:
            continue
        F = np.where(nv > 0, nwt / np.maximum(nv, 1), 0.0)
        Fs = ndi.uniform_filter1d(F, 3, axis=1)                    # 6 km along range
        val = (nv > 0).astype(float)

        def sidemean(ds, sign):
            num = sum(np.roll(Fs, sign * d, 0) * np.roll(val, sign * d, 0) for d in ds)
            den = sum(np.roll(val, sign * d, 0) for d in ds)
            # neighbours mostly outside the image: not a spike
            return np.where(den >= 0.75 * len(ds), num / np.maximum(den, 1), 1.0)
        side = np.maximum(sidemean((3, 4, 5, 6), 1), sidemean((3, 4, 5, 6), -1))
        wide = np.maximum(sidemean(range(6, 11), 1), sidemean(range(6, 11), -1))
        core = np.maximum(Fs, np.maximum(np.roll(Fs, 1, 0), np.roll(Fs, -1, 0)))
        cand = (core >= 0.5) & (side <= 0.12) & (wide <= 0.2) & (nv > 0)
        lab, n = ndi.label(cand, structure=np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]]))
        if n == 0:
            continue
        length = ndi.sum(np.ones_like(F), lab, np.arange(1, n + 1))
        keep = np.r_[False, length >= MINRUN][lab]
        keep = keep | np.roll(keep, 1, 0) | np.roll(keep, -1, 0)    # +-1 deg
        pix = np.zeros(codes.size, bool)
        pix[sel] = keep.ravel()[idx[sel]]
        out |= pix & wet
    out = out.reshape(codes.shape)
    if out.any():
        # width check: spikes are 1-2 km wide; rain bands that happen to lie along a ray are wider
        edt = ndi.distance_transform_edt(wet.reshape(codes.shape))
        lab, n = ndi.label(out, structure=np.ones((3, 3)))
        med = ndi.median(edt, lab, np.arange(1, n + 1))
        out = np.r_[False, np.asarray(med) <= MAXW][lab]
    return out


def _nc_fill(r, hole, sigma=2.0):
    """Normalised convolution: Gaussian-weighted mean of the known pixels (dry counts as 0)."""
    num = ndi.gaussian_filter(np.where(hole, 0, r), sigma)
    den = ndi.gaussian_filter((~hole).astype(np.float32), sigma)
    return np.where(den > 0.05, num / np.maximum(den, 1e-6), 0)


# ---------------------------------------------------------------- clutter probability
def tile_prob(x, xp, xn, v, assets):
    """x, xp, xn: log(1+R) at t and of t-1 / t+1 advected to t. Tile probabilities (nan: < 5 wet px)."""
    wet = x > 0
    W = _tiles(wet)
    nw = W.sum(-1)

    def tmean(a):
        return (_tiles(a) * W).sum(-1) / np.maximum(nw, 1)
    f = {'fill5': tmean(_box(wet, 5)), 'logr': tmean(x)}
    j = np.zeros_like(x)
    for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        j = np.maximum(j, np.abs(x - np.roll(x, sh, (0, 1))))
    f['jump'] = tmean(j)
    lab, _ = ndi.label(wet)
    srt = np.sort(_tiles(lab), -1)
    nobj = ((np.diff(srt, axis=-1) != 0) & (srt[..., 1:] > 0)).sum(-1) + (srt[..., 0] > 0)
    f['obj_per_100px'] = 100 * nobj / np.maximum(nw, 1)
    f['lnw'] = np.log10(np.maximum(nw, 1))

    def corr(a, b, win=7):
        ma, mb = _box(a, win), _box(b, win)
        cov = _box(a * b, win) - ma * mb
        va, vb = _box(a * a, win) - ma * ma, _box(b * b, win) - mb * mb
        return cov / np.sqrt(np.maximum(va * vb, 1e-6))
    f['corr'] = tmean(0.5 * (corr(x, xp) + corr(x, xn)))
    f['lspeed'] = np.log(np.clip(tmean(np.hypot(v[0], v[1])), 0, 10) + 1)
    X = np.stack([f[k] for k in assets.feats], -1)
    z = assets.coef[0] + ((X - assets.mu) / assets.sd) @ assets.coef[1:]
    p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
    return np.where(nw >= 5, p, np.nan)


# ---------------------------------------------------------------- cleaner
def clean(codes, times, assets: Assets | None = None):
    """Clean frames 1..T-2 of consecutive 5-minute RTCOR frames (uint16 codes, time, y, x).

    `times` are the frame timestamps (UTC). Frames 0 and T-1 are context only and are returned
    unchanged. Returns the cleaned codes and per-frame statistics inside the footprint.
    """
    assets = assets or load_assets()
    codes = np.asarray(codes)
    times = np.asarray(times).astype('datetime64[ns]')
    T = codes.shape[0]
    if codes.ndim != 3 or codes.shape[1:] != assets.shape:
        raise ValueError(f'frames must have shape (time, {assets.shape[0]}, {assets.shape[1]})')
    if len(times) != T or T < 3:
        raise ValueError('need at least three frames and one timestamp per frame')
    ny, nx = assets.shape
    valid = codes < MISSING
    R = np.where(valid, codes.astype(np.float32) * np.float32(MMH_PER_CODE), np.float32(0))
    V = motion(R)
    old = times < NEW_CHAIN
    month = times.astype('datetime64[M]').astype(np.int64) % 12 + 1
    hour = times.astype('datetime64[h]').astype(np.int64) % 24

    # 2. spikes and static clutter, refilled in space and time
    SPK = np.zeros(codes.shape, bool)
    for k in range(T):
        SPK[k] = ndi.binary_dilation(detect_spikes(codes[k], assets.sites), iterations=1) & valid[k]
        if old[k]:
            SPK[k] |= assets.static & valid[k]
    R1 = R.copy()
    for t in range(T):
        if not SPK[t].any():
            continue
        est, wgt = np.zeros_like(R[t]), np.zeros_like(R[t])
        for k in (-1, 1):
            if 0 <= t + k < T:
                v = _warp(np.where(SPK[t + k], np.nan, R[t + k]), V[t], k)
                ok = _warp(np.where(SPK[t + k] | ~valid[t + k], 0.0, 1.0), V[t], k) > 0.5
                est += np.where(ok, v, 0)
                wgt += ok
        spatial = _nc_fill(R[t], SPK[t])
        temporal = np.where(wgt > 0, est / np.maximum(wgt, 1), spatial)
        R1[t] = np.where(SPK[t], 0.5 * spatial + 0.5 * temporal, R[t])

    # 3. tile clutter probability and scene clutter fraction
    x = np.log1p(R1)
    P = np.full((T, ny // TILE, nx // TILE), np.nan)
    S = np.zeros(T)
    for t in range(1, T - 1):
        xp = np.log1p(_warp(R1[t - 1], V[t], 1))
        xn = np.log1p(_warp(R1[t + 1], -V[t], 1))
        P[t] = tile_prob(x[t], xp, xn, V[t], assets)
        nw = _tiles(x[t] > 0).sum(-1)
        ok = np.isfinite(P[t])
        S[t] = (P[t][ok] * nw[ok]).sum() / max(nw[ok].sum(), 1)

    out = codes.copy()
    gy = (np.arange(ny) - (TILE - 1) / 2) / TILE
    gx = (np.arange(nx) - (TILE - 1) / 2) / TILE
    yy, xx = np.meshgrid(gy, gx, indexing='ij')
    k3 = np.ones((3, 3, 3))
    stats = {k: np.zeros(T, np.int64) for k in ('spike_pixels', 'raw_wet_pixels', 'clean_wet_pixels',
                                                'raw_sum_code', 'clean_sum_code')}
    stats['cleaned'] = np.zeros(T, bool)
    stats['low_threshold'] = np.zeros(T, bool)
    stats['scene_fraction'] = S
    prev_prot = np.zeros((ny, nx), bool)
    for t in range(1, T - 1):
        st = P[t - 1:t + 2]                                   # P[0] and P[T-1] are nan
        num = ndi.convolve(np.nan_to_num(st), k3, mode='constant')[1]
        den = ndi.convolve(np.isfinite(st).astype(float), k3, mode='constant')[1]
        ps = np.where(den > 0, num / np.maximum(den, 1e-9), 0.0)
        pp = ndi.map_coordinates(ps, [yy, xx], order=1, mode='nearest')
        # 4. scene threshold (warm season only)
        thr = THR
        if 4 <= month[t] <= 9:
            night = month[t] in (6, 7, 8) and (hour[t] >= 18 or hour[t] < 6)
            if S[t] >= (SCENE_MIN_NIGHT if night else SCENE_MIN):
                thr = THR_SCENE
        r = np.where(pp > thr, 0.0, R1[t])

        # 5. protection of rain-like objects
        wt = R1[t] > 0
        lab0, n0 = ndi.label(wt, structure=np.ones((3, 3)))
        size = np.bincount(lab0.ravel(), minlength=n0 + 1)

        def per_obj(field):
            return np.bincount(lab0.ravel(), weights=field.ravel(), minlength=n0 + 1) / np.maximum(size, 1)
        adv_prev, adv_next = _warp(R1[t - 1], V[t], 1) > 0, _warp(R1[t + 1], -V[t], 1) > 0
        both = ndi.maximum_filter(adv_prev, 3) & ndi.maximum_filter(adv_next, 3)
        still = ndi.maximum_filter(R1[t - 1] > 0, 3) & ndi.maximum_filter(R1[t + 1] > 0, 3)
        ov, ovs, spd = per_obj(both), per_obj(still), per_obj(np.hypot(V[t][0], V[t][1]))
        xt = x[t]
        jmp, nb = np.zeros_like(xt), np.zeros_like(xt)
        for sh in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            y = np.roll(xt, sh, (0, 1))
            jmp = np.maximum(jmp, np.where(wt & (y > 0), np.abs(xt - y), 0))
            nb += y > 0
        fill_int, rough = per_obj(wt & (nb == 4)), per_obj(jmp)
        moving = (ov >= 0.6) & (spd >= 1.5) & (ov >= ovs + 0.15)
        smooth = (fill_int >= 0.5) & (rough <= 0.25)
        lineage = per_obj(_warp(prev_prot.astype(np.float32), V[t], 1) > 0.5)
        primary = (size >= MIN_OBJECT) & (moving | smooth)
        prot = primary | (lineage >= 0.3)

        # 6. intense, hard-edged, non-moving blobs in clutter-dominated scenes
        blob = np.zeros(n0 + 1, bool)
        if thr == THR_SCENE and n0:
            peak = np.asarray(ndi.maximum(R1[t], lab0, np.arange(n0 + 1)))
            edge = wt & ~ndi.binary_erosion(wt, structure=np.ones((3, 3)))
            heavy = np.bincount(lab0.ravel(), weights=(edge & (R1[t] >= BLOB_EDGE_RATE)).ravel(),
                                minlength=n0 + 1)
            heavy /= np.maximum(np.bincount(lab0.ravel(), weights=edge.ravel(), minlength=n0 + 1), 1)
            blob = (peak >= BLOB_PEAK) & (heavy >= BLOB_EDGE_SHARE) & ~moving
            blob[0] = False
        prot &= ~blob
        prot[0] = False
        r = np.where(blob[lab0], 0.0, r)
        r = np.where(prot[lab0], R1[t], r)
        prev_prot = (primary & ~blob)[lab0] & (lab0 > 0)      # one-step lineage

        # 7. specks without support along the flow at t-1 and t+1
        lab, n = ndi.label(r > 0)
        if n:
            size = np.bincount(lab.ravel(), minlength=n + 1)
            sup = np.bincount(lab.ravel(), weights=both.ravel(), minlength=n + 1) > 0
            kill = (size <= SPECK) & ~sup
            kill[0] = False
            r[kill[lab]] = 0

        # 8. isolated hot values
        xr = np.log1p(r)
        med = ndi.median_filter(xr, 5)
        r = np.where((r > HOT_MIN) & (xr - med > HOT_LOG), np.expm1(med + HOT_LOG), r)

        # 9. footprint
        c = np.rint(r / MMH_PER_CODE).astype(np.int64).clip(0, MISSING - 1).astype(codes.dtype)
        c = np.where(valid[t], c, codes[t])
        out[t] = np.where(assets.cover, c, OUTSIDE)

        m = assets.cover & valid[t]
        stats['cleaned'][t] = True
        stats['low_threshold'][t] = thr == THR_SCENE
        stats['spike_pixels'][t] = int((SPK[t] & m).sum())
        stats['raw_wet_pixels'][t] = int(((codes[t] > 0) & m).sum())
        stats['clean_wet_pixels'][t] = int(((out[t] > 0) & m).sum())
        stats['raw_sum_code'][t] = int(codes[t][m].sum(dtype=np.int64))
        stats['clean_sum_code'][t] = int(out[t][m].sum(dtype=np.int64))
    return out, stats
