"""spikes2: interference spike / strobe remover for old-chain KNMI RTCOR, working in each radar's polar grid.

    out = remove_spikes(codes, times, strength=1.0, ret_info=False, raw=None)

codes (T, 765, 700) uint16 window (raw RTCOR or the output of another cleaner, e.g. v11); raw = the window that
detection reads (default: codes; the v12 stage passes the raw window and cleans its current output). Output is
never above the input, codes >= 65534 pass unchanged, no echo is added. New-chain frames (times >= 2023-01-31
10:45) are returned unchanged. strength scales the evidence thresholds (> 1 removes more, < 1 less; 0 = off).

Method (details and numbers in NOTES.md):
1. POLAR GRID. Every frame is resampled per NL radar (Den Helder, Herwijnen) to 1440 azimuth rows of 0.25 deg x
   340 range bins of 1 km (nearest pixel). An interference spike is a vertical line there: constant azimuth, a
   top-hat cross-section in azimuth.
2. CANDIDATES (polar.py). Per width 0.75/1.25/2.25/3.75 deg: clear-air line cells (core wet share >= 0.5, both
   sides <= 0.15) and in-rain contrast cells (core log1p(code) above both sides); components >= 20 km along range.
3. GEOMETRY (features.py): constant azimuth along range (centroid drift <= 1 deg per 100 km, scatter <= 0.25 deg),
   top-hat contrast, sharp edges, clear-air share, width <= 5 deg, distance to the end of coverage.
4. TIME (the decisive part): the azimuth profile at t-1 / t+1 (t-2 / t+2) over the same range:
     fixed   the same profile at shift 0 (earth-fixed in polar coordinates: an emitter);
     flash   the line is absent in place AND nothing similar is nearby (the echo did not move, it vanished), in
             a frame whose surroundings are not much drier (burst frames are no evidence);
     moving  the profile re-appears shifted by >= 0.5 deg (rain moving with the flow) -> never a seed.
   Seeds: straight top-hat lines that flash in all neighbours (A), or reach the coverage edge (B), or are >= 80 km
   long (C), are fixed over t-2..t+2 with very sharp edges (D), fixed on one side and flash on the other (E),
   >= 100 km to the edge (F), short stubs at the coverage edge that flash (H); B, C, E, F need width <= 2.5 deg.
   None of these fires on any rain-as-old candidate (3044 candidates, 144 frames, see NOTES.md).
5. TRACKS: a seed's azimuth is followed through the window (same radar, within 0.5 deg or its half width, +-3
   frames, chained): weaker candidates at that azimuth that do not move are accepted; frames without a candidate
   get the ray carried (applied only where the frame itself shows the ray's excess).
6. REMOVAL along the whole ray, in Cartesian pixels, from the detected range out to the coverage edge (when the
   ray reaches it) and inward/outward through rain while 4-km blocks show the ray's excess:
     I     rain beside the strip at the same range: medians of 3 points 1-3 px outside each edge, linearly
           interpolated across the strip (0 in clear air);
     Zk    the ray's power, k r^2 in Z = code^1.6 units, k measured on the ray's clear-air pixels in this frame
           (else the track's nearest frame, else the in-rain excess over I);
     est   = (code^1.6 - Zk)^(1/1.6), the rain under the ray; where the ray is not small against the rain
           (code^1.6 < 4 Zk) est is held between 0.67 I and 1.5 I (natural variability beside the strip), so in
           clear air est = 0 and in rain the strip takes the level and texture of the rain beside it: no slit, no
           hole, no lowered band; never below 1 code where rain is beside the strip; never above the input.
     rims  1 px outside the measured width: half-way blend between input and est (no seam, no hollow outline).
"""
# Ported from research proto/spikes2/spikes2.py (md5 a2a83f20a9692e6f1725f06446c4fed7),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations

import time as _time

import cv2
import numpy as np

from .spikes2_common import DA, NA, NR, H, W, MISSING, SITES, polar_geo
from . import spikes2_polar as PO
from . import spikes2_features as FE

NEW_CHAIN = np.datetime64("2023-01-31T10:45", "ns")
B_EXP = 1.6
DEFAULTS = dict(drift=1.0, csd=0.25, con=0.25, edge=0.35, ca=0.3, wmax=5.0, lmin=30, lmin_b=40, gap=8, llong=80,
                hold=3, rim=1.5, offs=(1.0, 2.0, 3.0), lo_f=0.67, hi_f=1.5, dom=4.0, blk=4, ev=0.3, r_cov=335,
                carry_min=3, protect=True, shower_len=20.0, shower_x=3.0, stub_frames=6, stub_win=4, sweep=True, sweep_px=2.0, sweep_deg=1.0, coh_px=20)


# ------------------------------------------------------------------------------------------------ polar cache
class Polar:
    def __init__(self, codes):
        self.codes = codes
        self.T = len(codes)
        self.P = {}
        self.cache = {}

    def raw(self, si, t):
        k = (si, t)
        if k not in self.P:
            g = polar_geo(si)
            self.P[k] = np.where(g["ok"], self.codes[t].reshape(-1)[np.maximum(g["idx"], 0)], 65535).astype(np.uint16)
        return self.P[k]

    def get(self, si, t):
        k = (si, t)
        if k not in self.cache:
            if len(self.cache) > 12:
                self.cache.pop(next(iter(self.cache)))
            P = self.raw(si, t)
            val = (P < MISSING).astype(np.float32)
            wet = ((P > 0) & (P < MISSING)).astype(np.float32)
            L = np.where(wet > 0, np.log1p(P.astype(np.float32)), 0).astype(np.float32)
            self.cache[k] = (wet, val, L)
        return self.cache[k]

    def stack(self, si, t):
        d = {u: self.get(si, u) for u in range(t - 2, t + 3) if 0 <= u < self.T}
        d["t"] = t
        return d


# ------------------------------------------------------------------------------------------------ rules
def _g(f, k, d=np.nan):
    v = f.get(k, d)
    return d if v is None else v


def geometry_ok(f, p, s=1.0):
    dtol = p["drift"] * (1.5 if (f["length"] >= 80 or (f["edge"] >= 0.85 and f["con"] >= 0.6)) else 1.0)
    return (abs(_g(f, "drift_r", 9)) <= dtol * s and _g(f, "cent_sd", 9) <= p["csd"] * s and f["con"] >= p["con"] / s
            and f["edge"] >= p["edge"] / s and f["ca"] >= p["ca"] / s and f["width"] <= p["wmax"] and f["length"] >= p["lmin"] / s)


def geometry_lite(f):
    straight = (f.get("nblk", 0) < 3) or (abs(_g(f, "drift_r", 9)) <= 1.5 and _g(f, "cent_sd", 9) <= 0.35)
    return straight and f["con"] >= 0.15 and f["edge"] >= 0.2 and f["width"] <= 5.0 and f["length"] >= 15


def time_class(f, u, con):
    if not f.get(f"{u}_ok", 0):
        return None
    sh, c0, cb = abs(f[f"{u}_shift"]), f[f"{u}_c0"], f[f"{u}_cb"]
    c0 = -1.0 if np.isnan(c0) else c0
    if sh <= 0.25 and c0 >= 0.6 and f[f"{u}_con0"] >= 0.5 * con:
        return "fixed"
    if sh >= 0.5 and cb - c0 >= 0.1 and cb >= 0.5 and f.get(f"{u}_conS", 1.0) >= 0.5 * con:
        return "moving"
    if f[f"{u}_con0"] <= 0.15 and f[f"{u}_flr"] <= 0.35 and f[f"{u}_reg"] >= 0.5:
        return "flash"
    return "other"


def classify(f, p, s=1.0):
    """seed rule letters that fire for candidate f (time features present)."""
    G = geometry_ok(f, p, s)
    tc = {u: time_class(f, u, f["con"]) for u in ("p", "n", "p2", "n2")}
    f["tc"] = tc
    near = [tc["p"], tc["n"]]
    avail = [c for c in near if c is not None]
    moving = "moving" in avail
    allflash = len(avail) > 0 and all(c == "flash" for c in avail)
    stable = len(avail) > 0 and not moving
    tev = any(c in ("fixed", "flash") for c in avail)
    fixed2 = all(tc[u] == "fixed" for u in ("p", "n", "p2", "n2"))
    narrow = f["width"] <= 2.5
    out = []
    if G and allflash:
        out.append("A")
    if G and stable and narrow and f["gap_end"] <= p["gap"] and f["length"] >= p["lmin_b"] / s and \
            (tev or (f["length"] >= 60 and f["edge"] >= 0.6)):
        out.append("B")
    if G and stable and narrow and f["length"] >= p["llong"]:
        out.append("C")
    if G and fixed2 and f["edge"] >= 0.8 and f["length"] >= 45:
        out.append("D")
    if G and stable and narrow and "fixed" in avail and "flash" in avail and f["edge"] >= 0.6 and f["length"] >= 40:
        out.append("E")
    if G and narrow and f["length"] >= 100 and f["gap_end"] <= p["gap"]:
        out.append("F")
    if (f["length"] >= 20 / s and f["gap_end"] <= 5 and allflash and f["width"] <= 2.0 and f["con"] >= 0.4 / s
            and f["edge"] >= 0.5 / s and f["ca"] >= 0.5 / s and abs(_g(f, "drift_r", 0)) <= 1.0 and _g(f, "cent_sd", 0) <= 0.25):
        out.append("H")
    return out


def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


# ------------------------------------------------------------------------------------------------ detection
def detect(src, p, s=1.0, frames=None):
    """candidates with features; seeds; track acceptance. Returns list of accepted ray dicts and all candidates."""
    T = len(src)
    PC = Polar(src)
    cands = []
    frames = range(T) if frames is None else frames
    for t in frames:
        for si in (0, 1):
            wet, val, L = PC.get(si, t)
            ca, ex, _ = PO.line_maps(wet, val, L)
            comps = PO.components(ca | ex)
            if not comps:
                continue
            P = PC.stack(si, t)
            for comp in comps:
                f = FE.features(comp, P, time=False)
                f.update(site=si, t=t)
                f["lite"] = geometry_lite(f)
                if f["lite"]:
                    FE.time_feats(f, P)
                    f["rules"] = classify(f, p, s)
                else:
                    f["rules"] = []
                f.pop("_ctx", None)
                cands.append(f)
    # --- S: short stubs at the coverage edge (e.g. rays sticking out of a rain band) that stay at the same azimuth
    #     for >= stub_frames consecutive frames (one-frame gaps allowed): earth-fixed for 25+ min, rain moves 10+ km
    stubs = [f for f in cands if f["lite"] and not f["rules"] and f["gap_end"] <= 5 and f["length"] >= 15
             and f["width"] <= 2.5 and f["con"] >= 0.4 / s and f["edge"] >= 0.5 / s and f["ca"] >= 0.5 / s
             and "moving" not in (f.get("tc") or {}).values()
             and abs(_g(f, "drift_r", 0)) <= 1.0 and _g(f, "cent_sd", 0) <= 0.25]
    for f in stubs:
        rows0, w = f["rows"]
        rows = (np.arange(rows0 - 10, rows0 + w + 10)) % NA
        a, b = 10, 10 + w - 1
        n = 0
        for u in range(f["t"] - p["stub_win"], f["t"] + p["stub_win"] + 1):
            if not (0 <= u < T):
                continue
            wu, vu, Lu = PC.get(f["site"], u)
            X, _, _ = FE.az_profile(wu, vu, Lu, rows, f["r0"], f["r1"])
            n += FE.tophat(X, a, b)[0] >= 0.25
        f["stub_n"] = int(n)
        if n >= p["stub_frames"]:
            f["rules"] = ["S"]
    # --- tracks: accept non-moving lite candidates at a seed's azimuth within +-hold frames (chained)
    acc = [f for f in cands if f["rules"]]
    for f in acc:
        f["why"] = "seed:" + "".join(f["rules"])
    accepted = {id(f) for f in acc}
    changed = True
    while changed:
        changed = False
        for f in cands:
            if id(f) in accepted or not f["lite"]:
                continue
            tcf = f.get("tc") or {}
            is_moving = tcf.get("p") == "moving" or tcf.get("n") == "moving"
            for g in acc:
                if g["site"] != f["site"] or abs(g["t"] - f["t"]) > p["hold"]:
                    continue
                if g["t"] == f["t"]:
                    # another piece of the same ray in the same frame (e.g. beyond the rain it crosses): collinear
                    # with the seed (same radar and azimuth), passing the geometry test, any width up to 5 deg
                    if g.get("why", "").startswith("seed") and geometry_ok(f, p, s) and \
                            abs(_wrap(g["a_c"] - f["a_c"])) <= max(0.5, 0.5 * g["width"], 0.5 * f["width"]):
                        f["why"] = "collinear"
                        acc.append(f)
                        accepted.add(id(f))
                        changed = True
                        break
                    continue
                if is_moving:
                    continue
                tol = max(0.5, 0.5 * g["width"])
                if abs(_wrap(g["a_c"] - f["a_c"])) <= tol and f["width"] <= g["width"] + 1.5:
                    f["why"] = "track"
                    if g["width"] > f["width"]:                  # a weak piece of a wider ray: use the ray's extent
                        f["a_c"], f["width"] = g["a_c"], g["width"]
                    acc.append(f)
                    accepted.add(id(f))
                    changed = True
                    break
    return acc, cands, PC


def _width_at(X, off, hwr, ext_max, frac):
    core = X[np.abs(off) <= max(hwr, 0.5)]
    if not len(core):
        return None
    cm = float(core.mean())
    far = X[(np.abs(off) >= hwr + 3.0 / DA) & (np.abs(off) <= hwr + 5.0 / DA)]
    side = float(np.median(far)) if len(far) else 0.0
    if cm < 0.4 or cm - side < 0.2 or side > 0.15:
        return None                                   # the line is not seen in clear air in this block: keep the detection width
    thr = max(side + frac * (cm - side), 0.12)
    ext = []
    for sgn in (-1, 1):
        e = hwr
        while e < hwr + ext_max / DA:
            j = np.flatnonzero((sgn * off > e) & (sgn * off <= e + 1))
            if not len(j) or X[j[0]] < thr:
                break
            e += 1
        ext.append(e)
    return max(ext)


def full_width(PC, f, ext_max=2.0, frac=0.25, blk=20, r_cov=335):
    """removal extent per 20-km range block, out to the coverage edge: from the detection core (rows >= half the
    peak wet share) each side grows while the block's azimuth profile of wet share stays >= side + frac (core - side)
    (and >= 0.12): skirts, sidelobes and rims of wide strobes go with the ray (no hollow outline); a ray that is
    wider farther out gets a wider strip there. Blocks where the line is not seen keep the detection width.
    Returns (a_c, rb, hwb): block centres (km) and half widths (deg)."""
    wet, val, L = PC.get(f["site"], f["t"])
    hwr = f["width"] / 2.0 / DA
    c = f["a_c"] / DA - 0.5
    K = int(hwr + (ext_max + 5.0) / DA) + 2
    rows = (np.arange(int(round(c)) - K, int(round(c)) + K + 1)) % NA
    off = np.arange(-K, K + 1) + (int(round(c)) - c)
    rb, hwb = [], []
    for b0 in range(max(f["r0"] - 40, 10), r_cov, blk):
        b1 = min(b0 + blk, NR) - 1
        nv = val[rows, b0:b1 + 1].sum(1)
        if nv[K] == 0:
            continue
        X = wet[rows, b0:b1 + 1].sum(1) / np.maximum(nv, 1)
        e = _width_at(X, off, hwr, ext_max, frac)
        rb.append(0.5 * (b0 + b1))
        hwb.append((e if e is not None else hwr) * DA)
    if not rb:
        return f["a_c"], np.array([0.0]), np.array([f["width"] / 2.0])
    return f["a_c"], np.array(rb), np.array(hwb)


def ray_list(acc, T, p, PC=None):
    """per frame: rays (merged by radar and azimuth), plus carried rays in frames within hold of a ray without one."""
    rays = {t: [] for t in range(T)}
    for f in sorted(acc, key=lambda x: -x["length"]):
        a_c, rb, hwb = full_width(PC, f) if PC is not None else (f["a_c"], np.array([0.0]), np.array([f["width"] / 2.0]))
        hw = f["width"] / 2.0
        r = dict(site=f["site"], a_c=a_c, hw=hw, rb=rb, hwb=hwb, r0=f["r0"], r1=f["r1"], edge=f["gap_end"] <= p["gap"],
                 carry=False, why=f.get("why", ""))
        dup = None
        for q in rays[f["t"]]:
            if q["site"] == r["site"] and abs(_wrap(q["a_c"] - r["a_c"])) <= max(q["hw"], r["hw"], 0.5) \
                    and abs(q["hw"] - r["hw"]) <= 0.5:
                dup = q
                break
        if dup is None:
            rays[f["t"]].append(r)
        else:                                            # same ray, another piece along range (e.g. both sides of rain)
            dup["r0"], dup["r1"] = min(dup["r0"], r["r0"]), max(dup["r1"], r["r1"])
            dup["hwb"] = np.maximum(dup["hwb"], np.interp(dup["rb"], r["rb"], r["hwb"]))
            dup["edge"] = dup["edge"] or r["edge"]
    for t in range(T):
        for u in range(max(t - p["hold"], 0), min(t + p["hold"] + 1, T)):
            if u == t:
                continue
            for r in rays[u]:
                if r["carry"]:
                    continue
                if any(q["site"] == r["site"] and abs(_wrap(q["a_c"] - r["a_c"])) <= max(q["hw"], r["hw"], 0.5) for q in rays[t]):
                    continue
                rays[t].append(dict(r, carry=True, src=u))
    return rays


# ------------------------------------------------------------------------------------------------ removal
_SEL = {}


def _site_sorted(si):
    if si not in _SEL:
        g = polar_geo(si)
        order = np.argsort(g["paz"], kind="stable")
        _SEL[si] = (order, g["paz"][order])
    return _SEL[si]


def strip_pixels(si, a_c, hw, r_in, r_out, rim, rb=None, hwb=None):
    g = polar_geo(si)
    order, azs = _site_sorted(si)
    hmax = max(hw, float(np.max(hwb)) if hwb is not None else hw)
    marg = hmax + np.degrees((rim + 1.0) / max(r_in, 5.0))
    marg = min(marg, 60.0)
    lo, hi = (a_c - marg) % 360.0, (a_c + marg) % 360.0
    if lo <= hi:
        idx = order[np.searchsorted(azs, lo):np.searchsorted(azs, hi)]
    else:
        idx = np.concatenate([order[np.searchsorted(azs, lo):], order[:np.searchsorted(azs, hi)]])
    r = g["pr"][idx]
    d = np.radians(_wrap(g["paz"][idx] - a_c))
    q = r * np.sin(d)
    hwp = np.interp(r, rb, hwb) if rb is not None else hw
    wk = r * np.tan(np.radians(hwp))
    ok = (r >= r_in) & (r <= r_out) & (np.abs(q) <= wk + rim) & (np.cos(d) > 0)
    return idx[ok], r[ok], q[ok], wk[ok]


def fix_ray(out_t, src_t, ray, p, k_hint=None, ref_t=None):
    """returns (pixel idx, new values, info) for one ray in one frame, or None."""
    si, a_c, hw = ray["site"], ray["a_c"], ray["hw"]
    rim = p["rim"]
    r_in = max(ray["r0"] - 60, 10)
    r_out = p["r_cov"]
    idx, r, q, wk = strip_pixels(si, a_c, hw, r_in, r_out, rim, ray.get("rb"), ray.get("hwb"))
    if len(idx) == 0:
        return None
    cur = out_t.reshape(-1)[idx].astype(np.float32)
    raw = src_t.reshape(-1)[idx].astype(np.float32)
    okv = (cur < MISSING) & (raw < MISSING)
    cur = np.where(okv, cur, 0)
    raw = np.where(okv, raw, 0)
    core = np.abs(q) <= wk + 0.5
    # rain beside the strip (from the frame being cleaned), interpolated across
    Wd = wk + rim
    REF = out_t if ref_t is None else ref_t             # sides read from the stage input (not from rays already cleaned)
    offs = np.array(p["offs"], np.float32)
    # vectorised side samples (offset depends on the pixel's own strip half width)
    cy, cx = SITES[si]
    sides = []
    for sign in (-1.0, 1.0):
        vals = []
        for o in offs:
            th = np.radians(a_c) + sign * np.arctan2(Wd + o, np.maximum(r, 1.0))
            y = np.rint(cy + r * np.sin(th)).astype(np.int64)
            x = np.rint(cx + r * np.cos(th)).astype(np.int64)
            ins = (y >= 0) & (y < H) & (x >= 0) & (x < W)
            v = REF[np.where(ins, y, 0), np.where(ins, x, 0)].astype(np.float32)
            vals.append(np.where(ins & (v < MISSING), v, np.nan))
        V = np.sort(np.stack(vals), 0)                     # NaN last
        n = (~np.isnan(V)).sum(0)
        # outside the domain / missing data is unknown, never dry
        med = np.where(n >= 2, V[np.clip((n - 1) // 2, 0, 2), np.arange(V.shape[1])], np.nan)
        sides.append(med)
    Lv, Rv = sides
    known = ~np.isnan(Lv) | ~np.isnan(Rv)
    both = ~np.isnan(Lv) & ~np.isnan(Rv)
    Lv, Rv = np.where(np.isnan(Lv), Rv, Lv), np.where(np.isnan(Rv), Lv, Rv)   # one side unknown: the other side
    Lv, Rv = np.nan_to_num(Lv), np.nan_to_num(Rv)
    frac = np.clip((q + Wd) / np.maximum(2 * Wd, 1e-3), 0, 1)
    I = Lv + (Rv - Lv) * frac
    Zo = cur ** B_EXP
    Zr = raw ** B_EXP
    clear = (Lv == 0) & (Rv == 0) & both              # clear air needs in-domain dry evidence on BOTH sides
    # ray power k (Z / r^2) from this frame's clear-air ray pixels (detection source)
    ca_px = clear & core & (raw > 0)
    k = k95 = np.nan
    if ca_px.sum() >= 10:
        kk = Zr[ca_px] / r[ca_px] ** 2
        k, k95 = float(np.median(kk)), float(np.percentile(kk, 95))
    elif k_hint is not None and np.isfinite(k_hint[0]):
        k, k95 = k_hint
    rain_core = core & ~clear
    if not np.isfinite(k):
        if rain_core.sum() >= 10:
            ex = np.maximum(Zr[rain_core] - I[rain_core] ** B_EXP, 0) / r[rain_core] ** 2
            k = float(np.median(ex))
        if not np.isfinite(k) or k <= 0:
            return None
        k95 = 3.0 * k
    Zk = k * r ** 2
    # block evidence along range (4 km): clear-air block present if the core has echo, rain block if excess
    b = ((r - r_in) // p["blk"]).astype(np.int64)
    nb = int(b.max()) + 1
    cnt = np.bincount(b[core], minlength=nb).astype(np.float32)
    wetc = np.bincount(b[core], weights=(raw[core] > 0).astype(np.float32), minlength=nb)
    nclr = np.bincount(b[core], weights=clear[core].astype(np.float32), minlength=nb)
    exc = np.bincount(b[core], weights=(Zr - I ** B_EXP)[core], minlength=nb)
    zk = np.bincount(b[core], weights=Zk[core], minlength=nb)
    with np.errstate(all="ignore"):
        is_clear = nclr >= 0.5 * np.maximum(cnt, 1)
        pres_clear = is_clear & (wetc >= 0.3 * cnt) & (cnt > 0)
        pres_rain = ~is_clear & (exc >= p["ev"] * zk) & (cnt > 0) & (exc > 0)
    present = pres_clear | pres_rain
    bc = (r_in + (np.arange(nb) + 0.5) * p["blk"])
    det = (bc >= ray["r0"] - 2) & (bc <= ray["r1"] + 2)
    # clear-air evidence runs (>= 2 blocks = 8 km of echo on the strip with dry sides)
    cr = pres_clear.astype(np.int8)
    run2 = cr & (np.r_[cr[1:], 0] | np.r_[0, cr[:-1]]).astype(bool)
    if ray["carry"]:
        # carried ray: needs its own evidence (clear-air run or in-rain excess) over >= carry_min blocks
        if present.sum() < p["carry_min"] or not run2.any():
            return None                            # a carried ray must show itself in clear air (>= 8 km) in this frame
        seg = present.copy()
    else:
        ends = [np.flatnonzero(det)] if det.any() else []
        if run2.any():
            ends.append(np.flatnonzero(run2))
        if not ends:
            return None
        allb = np.concatenate(ends)
        i0, i1 = int(allb.min()), int(allb.max())
        if ray["edge"]:
            i1 = nb - 1
        span = np.zeros(nb, bool)
        span[i0:i1 + 1] = True
        # inside the span: detected blocks, clear-air blocks with echo, rain blocks with the ray's excess
        seg = span & ((det & (cnt > 0)) | present | (is_clear & (cnt > 0)))
    use = seg[b]
    # estimate
    sub = np.maximum(Zo - Zk, 0) ** (1.0 / B_EXP)
    # the ray is small against the echo: subtract it, keep the echo (rain: vs the median ray power; clear air: vs its
    # 95th percentile, so the ray's own strong pixels are not taken for a shower)
    dom = np.where(clear, False, Zo >= p["dom"] * Zk)
    # in rain the strip is held at the level of the WETTER side (a ray crossing the flank of a rain area has one dry
    # side: interpolating towards it made a dark stripe through the rain edge)
    Iw = np.maximum(I, np.maximum(Lv, Rv))
    est = np.where(dom, sub, np.clip(sub, p["lo_f"] * Iw, p["hi_f"] * Iw))
    est = np.where((Iw >= 0.5) & (cur >= 1), np.maximum(est, 1.0), est)
    est = np.minimum(est, cur)
    if p["protect"]:
        est = protect_showers(idx, r, cur, est, sub, clear & use & okv, k, p)
    rimpx = ~core
    est = np.where(rimpx, cur - 0.5 * (cur - est), est)
    new = np.floor(est + 0.5)
    new = np.minimum(new, cur)
    chg = use & okv & known & (new < cur)
    sb = np.flatnonzero(seg)
    ext = (r_in + sb.min() * p["blk"], (p["r_cov"] if (ray["edge"] and not ray["carry"]) else r_in + (sb.max() + 1) * p["blk"]))
    return idx[chg], new[chg].astype(np.uint16), dict(k=(k, k95), nseg=int(seg.sum()), n=int(chg.sum()), ext=ext)


def _side_median(F, si, a_c, r, off):
    """median code of the points at lateral offsets off[k] (km, signed arrays) at range r; NaN-safe, missing -> 0."""
    cy, cx = SITES[si]
    vals = []
    for o in off:
        th = np.radians(a_c) + np.arctan2(o, np.maximum(r, 1.0))
        y = np.rint(cy + r * np.sin(th)).astype(np.int64)
        x = np.rint(cx + r * np.cos(th)).astype(np.int64)
        ins = (y >= 0) & (y < H) & (x >= 0) & (x < W)
        v = F[np.where(ins, y, 0), np.where(ins, x, 0)].astype(np.float32)
        vals.append(np.where(ins & (v < MISSING), v, np.nan))
    V = np.stack(vals)
    ok = (~np.isnan(V)).sum(0) >= 2
    with np.errstate(all="ignore"):
        m = np.nanmedian(np.where(ok[None], V, 0.0), 0)
    return np.where(ok, m, np.nan)                      # outside / missing: unknown


def sweep_corridor(out_t, ray, ext, k, p, ref_t=None):
    """Once a ray is established, every echo in its azimuth corridor (strip + max(2 px, 0.75 deg) each side) over the
    ray's range extent is removed where the rain beside the corridor at the same range is dry on both sides (median of
    3 points 1-3 px outside the corridor): thin bright cores, dashes, stubs and skirts out in clear air go, also when
    they touch rain further along. Where the corridor runs through rain nothing is swept (the strip estimate there
    brings the ray to the rain level). Compact bright showers sitting on the ray (>= 12 px, short along the ray,
    filled, core >= shower_x times the ray's level) are kept. Returns pixel indices to zero."""
    si, a_c, hw = ray["site"], ray["a_c"], ray["hw"]
    rim = p["rim"]
    r0, r1 = max(ext[0] - 4, 5), min(ext[1] + 4, p["r_cov"])
    idx, r, q, wk = strip_pixels(si, a_c, hw, r0, r1, rim + 6.0, ray.get("rb"), ray.get("hwb"))
    if len(idx) == 0:
        return idx
    marg = np.maximum(p["sweep_px"], r * np.tan(np.radians(p["sweep_deg"])))
    wc = wk + rim + marg
    inc = np.abs(q) <= wc
    idx, r, q, wc = idx[inc], r[inc], q[inc], wc[inc]
    v = out_t.reshape(-1)[idx]
    wetp = (v > 0) & (v < MISSING)
    if not wetp.any():
        return idx[:0]
    idx, r, q, wc, v = idx[wetp], r[wetp], q[wetp], wc[wetp], v[wetp]
    REF = out_t if ref_t is None else ref_t
    lft = _side_median(REF, si, a_c, r, [-(wc + d) for d in (1.0, 2.0, 3.0)])
    rgt = _side_median(REF, si, a_c, r, [(wc + d) for d in (1.0, 2.0, 3.0)])
    z = (lft == 0) & (rgt == 0)                         # NaN (outside / missing) is never dry
    if p["protect"] and np.isfinite(k) and z.sum() >= 12:
        ys, xs = idx // W, idx % W
        y0, x0 = ys.min(), xs.min()
        img = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), np.uint8)
        img[ys[z] - y0, xs[z] - x0] = 1
        n, lab = cv2.connectedComponents(img, connectivity=8)
        li = np.where(z, lab[ys - y0, xs - x0], 0)
        lvl = (k * r ** 2) ** (1.0 / B_EXP)
        cnt = np.bincount(li, minlength=n)
        peak = np.zeros(n); np.maximum.at(peak, li, v / np.maximum(lvl, 0.5))
        rmax = np.full(n, -1.0); np.maximum.at(rmax, li, r)
        rmin = np.full(n, 1e9); np.minimum.at(rmin, li, r)
        ymin = np.full(n, 1e9); np.minimum.at(ymin, li, ys); ymax = np.full(n, -1.0); np.maximum.at(ymax, li, ys)
        xmin = np.full(n, 1e9); np.minimum.at(xmin, li, xs); xmax = np.full(n, -1.0); np.maximum.at(xmax, li, xs)
        span_ = np.maximum(ymax - ymin, xmax - xmin) + 1
        keep = (cnt >= 12) & (rmax - rmin <= p["shower_len"]) & (peak >= p["shower_x"]) & (cnt >= 0.125 * span_ ** 2 + 6)
        keep[0] = False
        z &= ~keep[li]
    return idx[z]


def protect_showers(idx, r, cur, est, sub, cand, k, p):
    """Showers sitting on the ray in clear air (the strip's sides dry): a connected echo piece on the strip that is
    short along the ray (<= shower_len km), has >= 6 px and a core >= shower_x times the ray's level at its range
    (rain has cores; the ray's range profile is smooth) keeps its echo minus the ray's median power."""
    z = cand & (cur > 0) & (est < cur)
    if z.sum() < 6:
        return est
    ys, xs = idx // W, idx % W
    y0, x0 = ys.min(), xs.min()
    img = np.zeros((ys.max() - y0 + 1, xs.max() - x0 + 1), np.uint8)
    wet = cand & (cur > 0)
    img[ys[wet] - y0, xs[wet] - x0] = 1
    n, lab = cv2.connectedComponents(img, connectivity=8)
    if n <= 1:
        return est
    li = lab[ys - y0, xs - x0]
    li = np.where(wet, li, 0)
    lvl = (k * r ** 2) ** (1.0 / B_EXP)                     # the ray's median level (codes) at each range
    cnt = np.bincount(li, minlength=n)
    rmax = np.full(n, -1.0); np.maximum.at(rmax, li, r)
    rmin = np.full(n, 1e9); np.minimum.at(rmin, li, r)
    peak = np.zeros(n); np.maximum.at(peak, li, cur / np.maximum(lvl, 0.5))
    keep = (cnt >= 6) & (rmax - rmin <= p["shower_len"]) & (peak >= p["shower_x"])
    if p.get("compact", True):
        ysw, xsw = ys - y0, xs - x0
        ymin = np.full(n, 1e9); np.minimum.at(ymin, li, ysw); ymax = np.full(n, -1.0); np.maximum.at(ymax, li, ysw)
        xmin = np.full(n, 1e9); np.minimum.at(xmin, li, xsw); xmax = np.full(n, -1.0); np.maximum.at(xmax, li, xsw)
        span_ = np.maximum(ymax - ymin, xmax - xmin) + 1
        keep &= cnt >= 0.25 * span_ ** 2 * 0.5 + 6       # filled, not a thin line
    keep[0] = False
    kp = keep[li]
    return np.where(kp, np.minimum(np.maximum(est, sub), cur), est)


def remove_spikes(codes, times=None, strength=1.0, ret_info=False, raw=None, **kw):
    """See module doc. kw: DEFAULTS keys."""
    p = {**DEFAULTS, **kw}
    codes = np.asarray(codes)
    src = codes if raw is None else np.asarray(raw)
    T = len(codes)
    out = codes.copy()
    old = np.ones(T, bool) if times is None else (np.asarray(times).astype("datetime64[ns]") < NEW_CHAIN)
    info = dict(rays={}, secs=0.0)
    if strength <= 0 or not old.any():
        return (out, info) if ret_info else out
    t0 = _time.time()
    s = float(strength)
    acc, cands, PC = detect(np.where(old[:, None, None], src, 0).astype(src.dtype), p, s)
    rays = ray_list(acc, T, p, PC)
    for carry in (False, True):                       # rays seen in the frame first, then carried rays (k from the track)
        for t in range(T):
            if not old[t]:
                continue
            for ray in rays[t]:
                if ray["carry"] != carry:
                    continue
                hint = None
                if carry:
                    hint = info["rays"].get((ray["src"], ray["site"], round(ray["a_c"], 2)), {}).get("k")
                res = fix_ray(out[t], src[t], ray, p, hint, ref_t=codes[t])
                if res is None:
                    continue
                idx, vals, ri = res
                o = out[t].reshape(-1)
                o[idx] = np.minimum(o[idx], vals)
                if p["sweep"] and ri["nseg"] > 0:
                    zi = sweep_corridor(out[t], ray, ri["ext"], ri["k"][0], p, ref_t=codes[t])
                    o[zi] = 0
                    ri["swept"] = int(len(zi))
                info["rays"][(t, ray["site"], round(ray["a_c"], 2))] = dict(ri, carry=carry, hw=ray["hw"], r0=ray["r0"],
                                                                             r1=ray["r1"], why=ray.get("why", ""))
    out = np.where(codes >= MISSING, codes, np.minimum(out, codes)).astype(codes.dtype)
    info["secs"] = _time.time() - t0
    info["cands"] = len(cands)
    info["accepted"] = acc
    return (out, info) if ret_info else out
