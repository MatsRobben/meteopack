"""v13 clutter-amount score for old-chain RTCOR (proto/v13). RTCOR only, cheap (~0.03 s/frame), continuous.

    S = clutter_score(codes, times)            # dict, see OUTPUT
    F = frame_pixels(codes, t, ...)            # per-wet-pixel feature table of one frame (for fitting / inspection)

IDEA. Every wet pixel gets a clutter probability p from a small logistic model (model.json, fit.py) on cheap,
explainable pixel features. The fitted model uses 4 of them (the others had ~0 weight and are computed for
inspection only): f31 (echo density around the pixel, -), thin15 and rough15 (granular texture of the surrounding
field, +), sea (+). In words: echo is clutter-like where the surrounding field is granular (many one-pixel-thin
pixels, rough code-to-code texture) for its density, more so over sea.
Per tile (32 px = 32 km) C = sum of p over the wet px (expected clutter px), W = wet px. C and W are smoothed in time
(Gaussian, sigma 15 min, cut at +-45 min, over judged frames only) and over neighbouring tiles (1-2-1 kernel) BEFORE
any ratio, so near-dry frames give no noisy ratios and the score cannot jump between frames. Then
    share = cal(C_s / (W_s + c))          c = 25 px (tile) / 500 px (frame); cal = logit-linear recalibration
                                          (fit.py, train tiles): "fraction of the echo here that is clutter"
    load  = share * W_s / (W_s + c0)      c0 = 100 px (tile, 10 % of its area) / 10000 px (frame, 2.5 % of the
                                          ~4e5 px footprint): the share, but counting the echo as at least c0,
                                          so a few speckles in a dry scene score low and a field of clutter high
THE SCORE is `load` (0..1, continuous): high only when there is both a high clutter share and a real clutter
amount. c0 was set a priori (physical: "a lot of clutter" covers >~10 % of a region), not tuned on test days.

FEATURES (per wet pixel at t; neighbours = t-1 and t+1, but a half (rows <380 / >=380) that is a frozen copy of
its predecessor (lib/stale.py rule) is replaced by the frame before / after it, so a copy is no evidence)
  lsz      log2 size of the 8-connected object (speck: 0-2)
  lonely   no wet px within 2 px at the previous and the next frame (both valid there), else 0
  olonely  share of the object's px that are lonely (blink-like object)
  f5, f31  wet share of the valid px in the 5x5 / 31x31 box around the pixel (speck density, dry surroundings)
  rough    std of ln(code) over the wet px of the 3x3 box (gradient / high-frequency texture of the field)
  thin     not inside any wet 2x2 square (one-px lines: one-range-bin dashes, rays, granular sea clutter)
  lc       ln(code) and  omax  ln(max code of the object) - 0.5 lsz (hot pixels without a weak body)
  sea      over sea (assets score.sea = proto/v13/sea.npy: Natural Earth coastline flood fill)  and  sea_dry  sea * (1 - f31)
  hot      on a fixed hot spot of the period map (proto/climmap, dilated 1 px)
  thin15, rough15, lon15   the same texture measured on the field: mean of thin / rough / lonely over the wet px
           of the 15x15 box (granular sea / anaprop fields vs smooth rain fields of the same density)
Model: assets score.model_json = research proto/v13/model.json (fitted by proto/v13/fit.py on labelled train frames + train crossings; never on TEST/dry_c).

OUTPUT of clutter_score (T frames, tiles of `tile` px, ny = ceil(765/tile), nx = ceil(700/tile))
  load_tile   (T, ny, nx) THE SCORE per tile                    load   (T,) THE SCORE per frame
  share_tile  (T, ny, nx) clutter share per tile                share  (T,) clutter share per frame
  dens_tile   (T, ny, nx) clutter px per tile px (= share*W_s/tile^2)  clut, wet (T,) clutter px / wet px per frame
  raw_C, raw_W  (T, ny, nx) unsmoothed expected clutter px / wet px per tile; raw_ok (T,) frame was judged
  p (optional, keep_p=True): (T, 765, 700) float16 pixel probabilities (NaN where dry/invalid)
Frames without both neighbours or with < 50 % valid px are not judged (raw_ok False) and only borrow from the
smoothing. Codes >= 65534 (missing / outside) are never wet.
"""
from __future__ import annotations

import json
from functools import lru_cache

import cv2
import numpy as np

from . import assets as _assets

MISSING = 65534
H, W = 765, 700
SPLIT = 380                                   # lib/stale.py halves
FEATURES = ["lsz", "lonely", "olonely", "f5", "f31", "rough", "thin", "lc", "omax", "sea", "sea_dry", "hot",
            "thin15", "rough15", "lon15"]
TILE, SIG_T, CUT_T = 32, 3.0, 9               # tile px, time sigma (frames), half width (frames)
C0_TILE, C0_FRAME = 25.0, 500.0              # shrinkage of the share
L0_TILE, L0_FRAME = 100.0, 10000.0            # "at least this much echo" of the load
_K5 = np.ones((5, 5), np.uint8)
_K2 = np.ones((2, 2), np.uint8)


@lru_cache(maxsize=1)
def sea_mask():
    return _assets.group("score")["sea"]


@lru_cache(maxsize=4)
def _hot(key):
    """Hot-spot map of the processing chain of month key 'YYYY-MM' (assets hot.old / hot.new) dilated by 1 px
    (equal to the research proto/v13/hot_<chain>.npy)."""
    chain = "new" if np.datetime64(key, "M") >= np.datetime64("2023-02", "M") else "old"
    return cv2.dilate(_assets.group("hot")[chain].astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def hot_map(times):
    if times is None or len(times) == 0:
        return np.zeros((H, W), bool)
    return _hot(str(np.datetime64(np.asarray(times)[len(times) // 2], "M")))


def _copy_of_prev(codes, t, sl, min_wet=300, share=0.85):
    """lib/stale.py rule for one half: frame t is a frozen copy of t-1 there."""
    if t < 1:
        return False
    a, b = codes[t][sl], codes[t - 1][sl]
    w = (a > 0) & (a < MISSING)
    return bool(w.sum() >= min_wet and (a[w] == b[w]).mean() > share)


def _neighbour(codes, t, step, copied):
    """Frame used as previous (step -1) / next (+1) neighbour of t, assembled per half: skip frames that are
    frozen copies (prev: if t itself is a copy of t-1, look at t-2; next: if t+1 is a copy of t, use t+2).
    Returns (wet bool, valid bool) or None when the neighbour lies outside the window."""
    T = len(codes)
    wet = np.zeros((H, W), bool); val = np.zeros((H, W), bool)
    for h, sl in enumerate((slice(0, SPLIT), slice(SPLIT, H))):
        j = t + step
        if step < 0:
            k = t
            while j >= 0 and copied[k, h] and j > t - 3:
                k, j = j, j - 1
        else:
            while j < T and copied[j, h] and j < t + 3:
                j += 1
        if j < 0 or j >= T:
            return None
        c = codes[j][sl]
        val[sl] = c < MISSING
        wet[sl] = (c > 0) & (c < MISSING)
    return wet, val


def copied_flags(codes):
    """(T, 2) bool: half (north, south) of frame t is a frozen copy of frame t-1."""
    T = len(codes)
    out = np.zeros((T, 2), bool)
    for t in range(1, T):
        for h, sl in enumerate((slice(0, SPLIT), slice(SPLIT, H))):
            out[t, h] = _copy_of_prev(codes, t, sl)
    return out


def _box(x, k):
    return cv2.boxFilter(x, cv2.CV_32F, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)


def frame_maps(codes, t, copied, hot):
    """Feature maps of frame t. Returns (wet bool (H,W), dict name -> float32 (H,W)) or None if not judgeable."""
    c = codes[t]
    valid = c < MISSING
    if valid.mean() < 0.5:
        return None
    wet = valid & (c > 0)
    pv, nx = _neighbour(codes, t, -1, copied), _neighbour(codes, t, 1, copied)
    if pv is None or nx is None:
        return None
    if pv[1].mean() < 0.5 or nx[1].mean() < 0.5:
        return None
    w8 = wet.astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(w8, connectivity=8)
    size = st[:, cv2.CC_STAT_AREA].astype(np.float32); size[0] = 1
    near_p = cv2.dilate(pv[0].astype(np.uint8), _K5) > 0
    near_n = cv2.dilate(nx[0].astype(np.uint8), _K5) > 0
    okn = (cv2.erode(pv[1].astype(np.uint8), _K5) > 0) & (cv2.erode(nx[1].astype(np.uint8), _K5) > 0)
    lonely = (wet & ~near_p & ~near_n & okn).astype(np.float32)
    lr = lab.ravel()
    olon = np.bincount(lr, weights=lonely.ravel(), minlength=n) / size
    lc = np.log(np.maximum(c, 1).astype(np.float32)) * wet
    omax = np.zeros(n, np.float32)
    np.maximum.at(omax, lr, lc.ravel())
    lsz = np.minimum(np.log2(size), 12.0)
    fv = valid.astype(np.float32); fw = wet.astype(np.float32)
    f5 = _box(fw, 5) / np.maximum(_box(fv, 5), 1)
    f31 = _box(fw, 31) / np.maximum(_box(fv, 31), 1)
    b3 = _box(fw, 3)
    m1 = _box(lc, 3) / np.maximum(b3, 1)
    m2 = _box(lc * lc, 3) / np.maximum(b3, 1)
    rough = np.sqrt(np.maximum(m2 - m1 * m1, 0)) * (b3 > 1)
    thin = (wet & ~(cv2.dilate(cv2.erode(w8, _K2, anchor=(0, 0)), _K2, anchor=(1, 1)) > 0)).astype(np.float32)
    sea = sea_mask().astype(np.float32)
    b15 = np.maximum(_box(fw, 15), 1)
    nb = {"thin15": _box(thin, 15) / b15, "rough15": _box(rough, 15) / b15, "lon15": _box(lonely, 15) / b15}
    maps = nb | {"lsz": lsz[lab], "lonely": lonely, "olonely": olon[lab].astype(np.float32), "f5": f5, "f31": f31,
            "rough": rough, "thin": thin, "lc": lc, "omax": (omax - 0.5 * lsz)[lab], "sea": sea,
            "sea_dry": sea * (1 - f31), "hot": hot.astype(np.float32)}
    return wet, maps


def frame_pixels(codes, t, times=None, copied=None, hot=None):
    """(wet mask, (n_wet, n_features) float32 table in FEATURES order) for frame t, or None."""
    copied = copied_flags(codes) if copied is None else copied
    hot = hot_map(times) if hot is None else hot
    r = frame_maps(codes, t, copied, hot)
    if r is None:
        return None
    wet, maps = r
    return wet, np.stack([maps[k][wet] for k in FEATURES], 1).astype(np.float32)


@lru_cache(maxsize=1)
def model():
    m = json.loads(str(_assets.group("score")["model_json"]))
    return (np.array(m["mean"], np.float32), np.array(m["scale"], np.float32), np.array(m["coef"], np.float32),
            np.float32(m["intercept"]), {"iso": m.get("iso"), "platt": m.get("platt")})


def predict(X):
    """Pixel clutter probability from the feature table (logistic model; the share recalibration is applied
    to tile / frame shares, not here)."""
    mu, sc, b, b0, _ = model()
    z = ((X - mu) / sc) @ b + b0
    return 1.0 / (1.0 + np.exp(-z))


def _tile_sum(x, tile):
    ny, nx = -(-H // tile), -(-W // tile)
    pad = np.zeros((ny * tile, nx * tile), np.float32)
    pad[:H, :W] = x
    return pad.reshape(ny, tile, nx, tile).sum((1, 3))


def raw_tiles(codes, times=None, tile=TILE, keep_p=False, frames=None):
    """Unsmoothed per-frame tile sums: C (expected clutter px), Wt (wet px), V (valid px), ok (T,)."""
    T = len(codes)
    copied = copied_flags(codes)
    hot = hot_map(times)
    ny, nx = -(-H // tile), -(-W // tile)
    C = np.zeros((T, ny, nx), np.float32); Wt = np.zeros_like(C); V = np.zeros_like(C)
    ok = np.zeros(T, bool)
    P = np.full((T, H, W), np.nan, np.float16) if keep_p else None
    for t in (range(T) if frames is None else frames):
        V[t] = _tile_sum((codes[t] < MISSING).astype(np.float32), tile)
        r = frame_maps(codes, t, copied, hot)
        if r is None:
            continue
        wet, maps = r
        ok[t] = True
        X = np.stack([maps[k][wet] for k in FEATURES], 1)
        p = predict(X) if len(X) else np.zeros(0, np.float32)
        pm = np.zeros((H, W), np.float32); pm[wet] = p
        C[t] = _tile_sum(pm, tile); Wt[t] = _tile_sum(wet.astype(np.float32), tile)
        if keep_p:
            P[t][wet] = p
    return C, Wt, V, ok, P


def smooth(C, Wt, ok, sig=SIG_T, cut=CUT_T, space=True):
    """NaN-aware smoothing of tile sums: Gaussian in time over judged frames, 1-2-1 over tiles. Returns the
    smoothed per-frame means (C_s, W_s) and the time-weight coverage (T,) (0 = no judged frame within reach)."""
    T = len(C)
    k = np.exp(-0.5 * (np.arange(-cut, cut + 1) / sig) ** 2).astype(np.float32)
    w = ok.astype(np.float32)
    Cs = np.zeros_like(C); Ws = np.zeros_like(Wt); cov = np.zeros(T, np.float32)
    for t in range(T):
        a, b = max(0, t - cut), min(T, t + cut + 1)
        kk = k[a - t + cut:b - t + cut] * w[a:b]
        s = kk.sum()
        cov[t] = s / k.sum()
        if s > 0:
            Cs[t] = np.tensordot(kk, C[a:b], 1) / s
            Ws[t] = np.tensordot(kk, Wt[a:b], 1) / s
    if space:
        Cs = np.stack([cv2.sepFilter2D(x, -1, np.float32([1, 2, 1]) / 4, np.float32([1, 2, 1]) / 4,
                                       borderType=cv2.BORDER_REPLICATE) for x in Cs]) if T else Cs
        Ws = np.stack([cv2.sepFilter2D(x, -1, np.float32([1, 2, 1]) / 4, np.float32([1, 2, 1]) / 4,
                                       borderType=cv2.BORDER_REPLICATE) for x in Ws]) if T else Ws
    return Cs, Ws, cov


def calibrate(share):
    """Recalibration of raw shares (fit.py, train tiles): logit-linear  cal(x) = sigmoid(a + b logit(x));
    the older isotonic form (model.json 'iso') is still read; identity if neither is present."""
    m = model()[4]
    share = np.asarray(share, np.float32)
    if m.get("platt"):
        x = np.clip(share, 1e-4, 1 - 1e-4)
        z = m["platt"]["a"] + m["platt"]["b"] * np.log(x / (1 - x))
        return (1 / (1 + np.exp(-z))).astype(np.float32)
    iso = m.get("iso")
    if not iso:
        return share
    return np.interp(share, iso["x"], iso["y"]).astype(np.float32)


def finish(C, Wt, ok, tile=TILE, sig=SIG_T, cut=CUT_T, c0_tile=C0_TILE, c0_frame=C0_FRAME, l0_tile=L0_TILE,
           l0_frame=L0_FRAME, space=True):
    """Smoothed scores from unsmoothed tile sums (raw_tiles): dict of load/share (tile and frame), dens_tile,
    clut, wet, cover. Usable on stored raw_C / raw_W (e.g. month.py results) to re-derive everything."""
    C = np.asarray(C, np.float32); Wt = np.asarray(Wt, np.float32)
    Cs, Ws, cov = smooth(C, Wt, ok, sig, cut, space)
    Cf, Wf, _ = smooth(C.sum((1, 2))[:, None, None], Wt.sum((1, 2))[:, None, None], ok, sig, cut, space=False)
    Cf, Wf = Cf[:, 0, 0], Wf[:, 0, 0]
    sh_t = calibrate(Cs / (Ws + c0_tile)); sh_f = calibrate(Cf / (Wf + c0_frame))
    return {"load_tile": sh_t * Ws / (Ws + l0_tile), "load": sh_f * Wf / (Wf + l0_frame),
            "share_tile": sh_t, "share": sh_f, "dens_tile": sh_t * Ws / float(tile * tile),
            "clut": sh_f * Wf, "wet": Wf, "cover": cov}


def clutter_score(codes, times=None, tile=TILE, keep_p=False, **kw):
    """Continuous clutter score of a window, see module doc. codes (T, 765, 700) uint16, times (T,).
    kw: finish() keys (sig, cut, c0_tile, c0_frame, l0_tile, l0_frame, space)."""
    C, Wt, V, ok, P = raw_tiles(codes, times, tile, keep_p)
    out = finish(C, Wt, ok, tile, **kw)
    out.update(raw_C=C, raw_W=Wt, raw_V=V, raw_ok=ok)
    if keep_p:
        out["p"] = P
    return out


for _f in (sea_mask, _hot, model):
    _assets.register_cache(_f.cache_clear)
