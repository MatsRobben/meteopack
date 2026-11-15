"""Per-frame quality flags and per-pixel uncertainty masks for old-chain RTCOR (round 5, proto/quality).

The cleaned old chain (v10 = proto/spikes/cleaner.py, c=1e9) is a diffusion-model target. Instead of dropping
frames or windows, the training can use these flags to skip/down-weight frames and these masks for a masked loss.
Flags only: the data are never modified, nothing is interpolated. RTCOR codes only (frame + neighbours in time,
fixed geography: footprint map, the two NL radar sites). Deterministic. ~0.05-0.1 s per frame.

    q = frame_quality(codes, times, cleaned=None)          # dict of (T,) arrays, see FRAME FIELDS
    m = pixel_uncertainty(codes, times, cleaned=None)      # uint8 (T, H, W) bit mask, see MASK BITS
    q, m = analyse(codes, times, cleaned=None)             # both in one pass (cheaper than calling both)
    w = loss_weight(m, codes)                              # float32 (T, H, W) in [0, 1] for a masked loss

codes: (T, 765, 700) uint16 raw RTCOR codes (65534 missing, 65535 outside; frames 5 min apart, missing frames
       all-65534 so time stays regular). times: (T,) datetime64. cleaned: optional cleaned codes of the same window
       (v10 output); used only for the residual-clutter score/mask (else computed on raw codes).
Pass a window with >= 3 frames of context on each side when possible: frame t is judged against t-1, t+1 and
(for support) t-2..t+2, t-3, t+3. Edge frames without both neighbours get FLAG_EDGE and no blink/dropout verdict.

DIAGNOSIS (NOTES.md): in the old chain the "blink frames" are mostly frames where the far range (> ~200-225 km
from the nearest of Den Helder / Herwijnen) carries rain that the neighbouring frames lack. The far-range rain is
coherent at t-2/t+2, advects with the field and continues smoothly, so the frames WITHOUT it are wrong (far-range
contribution dropped, preferentially at minute % 10 == 5). Hence the logic below:
  blink px    wet at t, not wet in the advected t-1 nor t+1 (dilated) (both neighbours valid there)
  supported   blink px whose area is wet in the advected t-2 or t+2 (or t-3/t+3): real rain that the neighbours
              dropped -> NOT masked at t; the neighbours get a dropout mask instead
  blink area  unsupported blink px in components >= `area` px: a genuine one-frame area -> masked (BLINK)
  dropout     dry at t but wet in both advected t-1 and t+1 (eroded), components >= `area` px, or a far-range
              ring sector that is near-dry at t while both neighbours have rain there -> masked (DROPOUT)

FRAME FIELDS (frame_quality / analyse)
  missing      share of the radar footprint that is missing at t
  frozen       lib/stale bits: 1 north half (rows 0-379) copied from t-1, 2 south half, 3 both
  blink        px in unsupported blink areas / footprint px (if `cleaned` is given: only px still wet after
               cleaning, e.g. one-frame interference rays the cleaner removed are not counted/masked)
  dropout      dry px in dropout regions / footprint px (regions: dry at t, wet in the advected t-1 and t+1, or
               in t-1 & t+2 / t-2 & t+1 for two dropped frames in a row; components >= area, holes closed)
  drop_far     share of the far-range (>= 200 km) rain of the neighbours that is absent at t (0..1; NaN if the
               neighbours have < 2000 far px)
  alt          alternation ratio of the wet fraction inside 150 km of the nearest radar (so the far-range dropout
               does not drive it; it measures the even/odd sensitivity alternation): (w_t - (w_t-1 + w_t+1)/2) / ((w_t-1 + w_t+1)/2), NaN when
               the neighbour mean < alt_min = 0.5 % of that area (near-dry frames give noisy ratios); alt_flag: |alt| > 0.15 and neighbours deviate the
               other way (a zig-zag), so a single real change is not counted
  resid        residual-clutter density: px in small weak lonely objects (<= 20 px, max code <= 3, no wet px within
               2 px at t-1 and t+1) per 10^4 footprint px, on `cleaned` if given
  wetfrac      wet px / valid footprint px
  flags        uint16 bit field F_MISSING 1, F_FROZEN 2, F_BLINK 4, F_DROPOUT 8, F_ALT 16, F_RESID 32, F_EDGE 64 at the
               default thresholds THRESH (missing 0.05, blink 0.001, dropout 0.005 or drop_far 0.5, alt 0.15, resid 1.5
               = new-chain q99 on v10 output; resid on raw codes runs ~1.3x higher); use the scores for other ones
MASK BITS (pixel_uncertainty / analyse): 1 MISSING (codes >= 65534 or outside the footprint), 2 FROZEN (half copied
  from t-1; only where rain is at t or at t+1 (advected), dilated by |v| + 3 px: the dry rest of the half is right), 4 BLINK, 8 DROPOUT, 16 FRINGE (pixel-level flicker at rain edges: weak px whose wet/dry state differs
  from both advected neighbours), 32 RESID (pixels of small weak lonely objects in `cleaned`), 64 EDGE (frame lacks
  a neighbour, nothing judged), 128 SUPPORT (info only, not for the loss: one-frame-looking area that IS present at
  t-2/t+2, i.e. real rain whose neighbours dropped it).
"""
from __future__ import annotations

import cv2
import numpy as np

from . import assets as _assets
from .stale import frozen_flags

MISSING = 65534
H, W = 765, 700
_COV: dict = {}
_assets.register_cache(_COV.clear)


def _cov():
    """effective radar coverage (bool (H, W)) from the assets."""
    if "c" not in _COV:
        _COV["c"] = _assets.group("quality")["cover_eff"].astype(bool)
    return _COV["c"]


def _ncov():
    return int(_cov().sum())


SITES = {"DH": (331, 333), "HW": (458, 369)}          # Den Helder, Herwijnen (row, col) on the RTCOR grid
_yy, _xx = np.mgrid[:H, :W]
RANGE = np.minimum(np.hypot(_yy - 331, _xx - 333), np.hypot(_yy - 458, _xx - 369)).astype(np.float32)
del _yy, _xx

M_MISSING, M_FROZEN, M_BLINK, M_DROPOUT, M_FRINGE, M_RESID, M_EDGE, M_SUPPORT = 1, 2, 4, 8, 16, 32, 64, 128
F_MISSING, F_FROZEN, F_BLINK, F_DROPOUT, F_ALT, F_RESID, F_EDGE = 1, 2, 4, 8, 16, 32, 64
THRESH = dict(missing=0.05, blink=0.001, dropout=0.005, drop_far=0.5, alt=0.15, resid=1.5)
DEFAULTS = dict(area=100, tol=2, near=150.0, far=200.0, alt_min=0.005, far_min=2000, resid_size=20, resid_code=3)


# ----------------------------------------------------------------------------------------------- motion
def motion(codes, valid):
    """Per-frame 5-min displacement (dx, dy) px from frame t-1 to t (row t; row 0 = row 1).

    Phase correlation of consecutive smoothed fields (codes capped at 60), estimates with < 3000 wet px or a weak
    peak dropped, then a running median over 13 frames (robust against frozen halves / dropout frames), filled with
    the window median, clipped to |v| <= 8 px. Returns (T, 2) float32."""
    T = len(codes)
    win = cv2.createHanningWindow((W, H), cv2.CV_32F)
    est = np.full((T, 2), np.nan, np.float32)
    prev = None
    for t in range(T):
        F = np.where(valid[t], np.minimum(codes[t], 60), 0).astype(np.float32)
        ok = (F > 0).sum() >= 3000
        F = cv2.GaussianBlur(F, (0, 0), 2) if ok else None
        if prev is not None and F is not None:
            (dx, dy), resp = cv2.phaseCorrelate(prev, F, win)
            if resp > 0.05 and abs(dx) <= 12 and abs(dy) <= 12:
                est[t] = dx, dy
        prev = F
    good = ~np.isnan(est[:, 0])
    if not good.any():
        return np.zeros((T, 2), np.float32)
    med = np.nanmedian(est, 0)
    out = np.empty_like(est)
    for t in range(T):
        seg = est[max(0, t - 6):t + 7]
        seg = seg[~np.isnan(seg[:, 0])]
        out[t] = np.median(seg, 0) if len(seg) >= 3 else med
    return np.clip(out, -8, 8)


def _shift(m, dx, dy):
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(m.view(np.uint8) if m.dtype == bool else m, M, (W, H), flags=cv2.INTER_NEAREST)


def _disk(r):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


# ----------------------------------------------------------------------------------------------- core
def analyse(codes, times=None, cleaned=None, *, prev=None, mv=None, thresh=None, **kw):
    """Frame fields (dict of (T,) arrays) and pixel mask (uint8 (T, H, W)) in one pass. See module docstring.

    prev: frame before codes[0] for the frozen test (optional). mv: precomputed motion (T, 2) (optional).
    thresh: override THRESH entries (frame flags only). kw: override DEFAULTS (area, tol, far, far_min, ...)."""
    p = {**DEFAULTS, **kw}
    th = {**THRESH, **(thresh or {})}
    codes = np.asarray(codes)
    T = len(codes)
    valid = (codes < MISSING) & _cov()
    wet = (codes > 0) & valid
    nval = valid.sum((1, 2))
    okf = nval >= 0.5 * _ncov()                               # frame usable as evidence
    if mv is None:
        mv = motion(codes, valid)
    cum = np.concatenate([[[0, 0]], np.cumsum(mv[1:], 0)]).astype(np.float32)   # position of frame t vs frame 0

    def disp(s, t):                                        # displacement taking frame s onto frame t
        d = cum[t] - cum[s]
        return float(d[0]), float(d[1])

    wet8 = wet.view(np.uint8)
    cache = {}

    def adv(s, t, r):                                      # frame s's wet mask advected to t, dilated by r (r<0: eroded)
        key = (s, t, r)
        if key not in cache:
            m = _shift(wet8[s], *disp(s, t))
            if r > 0:
                m = cv2.dilate(m, _disk(r))
            elif r < 0:
                m = cv2.erode(m, _disk(-r))
            cache[key] = m.astype(bool)
        return cache[key]

    def advv(s, t):                                        # validity of frame s advected to t
        key = (s, t, "v")
        if key not in cache:
            cache[key] = _shift(valid[s].view(np.uint8), *disp(s, t)).astype(bool)
        return cache[key]

    clean = None if cleaned is None else np.asarray(cleaned)
    cw = wet if clean is None else ((clean > 0) & valid)
    mask = np.zeros((T, H, W), np.uint8)
    mask[~valid] |= M_MISSING
    fz = frozen_flags(codes, prev)
    far = RANGE >= p["far"]
    farc = far & _cov()
    q = {k: np.full(T, np.nan, np.float32) for k in ("blink", "dropout", "drop_far", "alt", "resid", "supported")}
    q["missing"] = (1 - nval / _ncov()).astype(np.float32)
    q["frozen"] = fz
    q["wetfrac"] = (wet.sum((1, 2)) / np.maximum(nval, 1)).astype(np.float32)
    q["edge"] = np.zeros(T, bool)
    nearc = _cov() & (RANGE < p["near"])
    q["wf_near"] = ((wet & nearc).sum((1, 2)) / np.maximum((valid & nearc).sum((1, 2)), 1)).astype(np.float32)
    tol, area = p["tol"], p["area"]
    k3 = _disk(1)
    for t in range(T):
        has = [s for s in (t - 1, t + 1) if 0 <= s < T and okf[s]]
        if len(has) < 2 or not okf[t]:
            q["edge"][t] = True
            mask[t] |= M_EDGE
            for k in ("blink", "dropout", "drop_far", "alt", "resid", "supported"):
                q[k][t] = np.nan
            if okf[t]:
                q["resid"][t] = _resid(cw, codes if clean is None else clean, t, [s for s in (t - 1, t + 1) if 0 <= s < T], valid, p, mask)
            continue
        a, b = t - 1, t + 1
        both_v = advv(a, t) & advv(b, t) & valid[t]
        # --- blink px and their support at t +- 2, 3
        bl = wet[t] & ~adv(a, t, tol) & ~adv(b, t, tol) & both_v
        q["blink"][t] = 0.0
        q["supported"][t] = 0.0
        if bl.sum() >= area:
            n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(bl.view(np.uint8), k3), connectivity=8)
            size = np.bincount(lab[bl], minlength=n)
            big = size >= area
            big[0] = False
            if big.any():
                far_t = [s for s in (t - 2, t + 2, t - 3, t + 3) if 0 <= s < T and okf[s]]
                hit = np.zeros(n)
                for s in far_t:
                    hit = np.maximum(hit, np.bincount(lab[bl], weights=adv(s, t, tol + abs(s - t))[bl], minlength=n) / np.maximum(size, 1))
                supported = big & (hit >= 0.3)
                spurious = big & ~supported
                bm = spurious[lab] & bl
                if clean is not None:                      # areas the cleaner already removed need no mask
                    bm &= clean[t] > 0
                if bm.any():
                    mask[t][cv2.dilate(bm.view(np.uint8), _disk(3)) > 0] |= M_BLINK
                q["blink"][t] = bm.sum() / _ncov()
                sm = supported[lab] & bl
                mask[t][sm] |= M_SUPPORT
                q["supported"][t] = sm.sum() / _ncov()
        # --- dropout regions: dry at t, wet in both advected neighbours (eroded by 1 px); also 2-frame gaps
        #     (t-1 & t+2, t-2 & t+1) so two consecutive dropped frames are caught; holes closed
        dr = adv(a, t, -1) & adv(b, t, -1)
        for a2, b2 in ((t - 1, t + 2), (t - 2, t + 1)):
            if 0 <= a2 and b2 < T and okf[a2] and okf[b2]:
                dr |= adv(a2, t, -1) & adv(b2, t, -1)
        dr &= ~wet[t] & valid[t]
        dm = np.zeros((H, W), bool)
        if dr.sum() >= area:
            n, lab, st, _ = cv2.connectedComponentsWithStats(dr.view(np.uint8), connectivity=8)
            big = st[:, cv2.CC_STAT_AREA] >= area
            big[0] = False
            if big.any():
                dm = cv2.morphologyEx(big[lab].view(np.uint8), cv2.MORPH_CLOSE, _disk(4)).astype(bool) & valid[t]
        # far-range share of the neighbours' rain missing at t
        nb_far = adv(a, t, 0) & adv(b, t, 0) & farc & valid[t]
        nf = int(nb_far.sum())
        if nf >= p["far_min"]:
            lost = nb_far & ~adv(t, t, tol)
            q["drop_far"][t] = lost.sum() / nf
        q["dropout"][t] = (dm & ~wet[t]).sum() / _ncov()
        if dm.any():
            mask[t][cv2.dilate(dm.view(np.uint8), _disk(3)) > 0] |= M_DROPOUT
        # --- alternation of the wet fraction
        wa, wb, wt = q["wf_near"][a], q["wf_near"][b], q["wf_near"][t]
        m = 0.5 * (wa + wb)
        if m > p["alt_min"]:
            q["alt"][t] = (wt - m) / m
        # --- fringe flicker: weak px whose wet state differs from both advected neighbours (no dilation), outside
        #     blink/dropout areas, only next to rain (within 3 px of another wet px state)
        na, nb_ = adv(a, t, 1), adv(b, t, 1)
        fl_wet = wet[t] & (codes[t] <= 2) & ~na & ~nb_ & both_v
        fl_dry = ~wet[t] & valid[t] & adv(a, t, -1) & adv(b, t, -1) & ~dm
        sup_d = cv2.dilate(((mask[t] & M_SUPPORT) > 0).view(np.uint8), _disk(3)) > 0
        fr = (fl_wet | fl_dry) & ((mask[t] & (M_BLINK | M_DROPOUT)) == 0) & ~sup_d
        if fr.any():
            mask[t][cv2.dilate(fr.view(np.uint8), k3) > 0] |= M_FRINGE
        # --- residual clutter on the cleaned (or raw) frame
        q["resid"][t] = _resid(cw, codes if clean is None else clean, t, [a, b], valid, p, mask)
        for key in [k for k in cache if k[0] < t - 3]:
            cache.pop(key)
    # frozen halves: mask the copied half where the field can have changed during the 5 min: wet at t (= the copy)
    # or wet at t+1 advected back, dilated by |v| + 3 px. Dry-to-dry areas of a frozen half are left unmasked.
    for t in np.nonzero(fz)[0]:
        r = int(np.ceil(np.hypot(*mv[t]))) + 3
        ch = wet8[t].copy()
        if t + 1 < T and okf[t + 1]:
            ch |= _shift(wet8[t + 1], *disp(t + 1, t))
        ch = cv2.dilate(ch, _disk(r)) > 0
        for bit, sl in ((1, slice(0, 380)), (2, slice(380, H))):
            if fz[t] & bit:
                mask[t, sl][ch[sl]] |= M_FROZEN
    # alternation zig-zag flag
    alt = q["alt"]
    zz = np.zeros(T, bool)
    for t in range(1, T - 1):
        if np.isnan(alt[t]):
            continue
        nb = [alt[s] for s in (t - 1, t + 1) if not np.isnan(alt[s])]
        zz[t] = abs(alt[t]) > th["alt"] and len(nb) > 0 and all(np.sign(x) == -np.sign(alt[t]) for x in nb)
    q["alt_flag"] = zz
    q["flags"] = flags_from(q, th)
    q["mv"] = mv
    return q, mask


def _resid(cw, cc, t, nbs, valid, p, mask):
    """Residual-clutter density at t: px in objects <= resid_size px with max code <= resid_code and no wet px within
    2 px at the neighbours (unadvected; lonely as in evalstats) per 10^4 footprint px; marks them M_RESID."""
    w = cw[t]
    if not w.any():
        return 0.0
    n, lab, st, _ = cv2.connectedComponentsWithStats(w.view(np.uint8), connectivity=8)
    size = st[:, cv2.CC_STAT_AREA]
    small = (size <= p["resid_size"])
    small[0] = False
    if not small.any():
        return 0.0
    mx = np.zeros(n, np.int64)
    np.maximum.at(mx, lab[w], cc[t][w].astype(np.int64))
    cand = small & (mx <= p["resid_code"])
    near = np.zeros((H, W), bool)
    for s in nbs:
        near |= cv2.dilate(cw[s].view(np.uint8), _disk(2)) > 0
    near |= (mask[t] & M_SUPPORT) > 0                      # objects touching real-but-dropped-elsewhere rain
    near = cv2.dilate(near.view(np.uint8), _disk(1)) > 0 if (mask[t] & M_SUPPORT).any() else near
    hitn = np.bincount(lab[w], weights=near[w], minlength=n)
    lonely = cand & (hitn == 0)
    px = lonely[lab] & w
    mask[t][px] |= M_RESID
    return float(px.sum() / _ncov() * 1e4)


def flags_from(q, th=None):
    """uint16 frame flag bits from the scores at thresholds th (default THRESH; th["resid"] = None: never flagged)."""
    th = {**THRESH, **(th or {})}
    f = np.zeros(len(q["missing"]), np.uint16)
    f[q["missing"] > th["missing"]] |= F_MISSING
    f[q["frozen"] > 0] |= F_FROZEN
    f[np.nan_to_num(q["blink"]) > th["blink"]] |= F_BLINK
    f[(np.nan_to_num(q["dropout"]) > th["dropout"]) | (np.nan_to_num(q["drop_far"]) > th["drop_far"])] |= F_DROPOUT
    f[q["alt_flag"]] |= F_ALT
    if th.get("resid") is not None:
        f[np.nan_to_num(q["resid"]) > th["resid"]] |= F_RESID
    f[q["edge"]] |= F_EDGE
    return f
