"""Per-candidate geometry and time features in a radar's polar grid (dev; see spikes2.py for the final stage).

Polar arrays per frame: wet (float32 0/1), val (float32 0/1), L = log1p(code) on wet cells.
"""
# Ported from research proto/spikes2/features.py (md5 9641fcbf8282c7159b0e07e054318d59),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations

import numpy as np

from .spikes2_common import DA, NA, NR

R_MAX = 335


def _rows(c, k):
    return (np.arange(c - k, c + k + 1)) % NA


def az_profile(wet, val, L, rows, r0, r1):
    v = val[rows, r0:r1 + 1].sum(1)
    X = wet[rows, r0:r1 + 1].sum(1) / np.maximum(v, 1)
    Y = L[rows, r0:r1 + 1].sum(1) / np.maximum(v, 1)
    return X, Y, v


def core_rows(X, k, lo, hi, frac=0.5):
    """contiguous rows around the peak inside [lo, hi] (indices into X) with X >= frac * peak."""
    p = lo + int(np.argmax(X[lo:hi + 1]))
    thr = frac * X[p]
    a = p
    while a - 1 >= 0 and X[a - 1] >= thr and p - a < 24:
        a -= 1
    b = p
    while b + 1 < len(X) and X[b + 1] >= thr and b - p < 24:
        b += 1
    return a, b


def tophat(X, a, b, gap=1, side=8):
    """contrast of rows a..b of X against the sides (rows a-gap-side..a-gap-1 and b+gap+1..b+gap+side)."""
    n = len(X)
    core = X[a:b + 1].mean()
    l = X[max(a - gap - side, 0):max(a - gap, 0)]
    r = X[min(b + gap + 1, n):min(b + gap + 1 + side, n)]
    sl = l.mean() if len(l) else 1.0
    sr = r.mean() if len(r) else 1.0
    return core - max(sl, sr), core, sl, sr


def features(comp, P, K=28, time=True):
    """comp: dict(rows, r0, r1, sub) from polar.components; P: dict t -> (wet, val, L) with keys t-1, t, t+1 ...
    (missing frames absent). Returns a feature dict."""
    t = P["t"]
    wet, val, L = P[t]
    r0, r1 = comp["r0"], comp["r1"]
    rows_c = comp["rows"]
    c = int(rows_c[len(rows_c) // 2])
    rows = _rows(c, K)
    X, Y, v = az_profile(wet, val, L, rows, r0, r1)
    lo = K - len(rows_c) // 2
    hi = lo + len(rows_c) - 1
    a, b = core_rows(X, K, max(lo, 0), min(hi, 2 * K))
    con, core, sl, sr = tophat(X, a, b)
    conY, _, _, _ = tophat(Y, a, b)
    f = dict(r0=r0, r1=r1, length=r1 - r0 + 1, w_rows=b - a + 1, width=(b - a + 1) * DA, con=con, conY=conY,
             core=core, side=max(sl, sr))
    crow = rows[a:b + 1]
    f["a_c"] = float(((rows[a] + (b - a) / 2.0) % NA + 0.5) * DA)
    f["rows"] = (int(rows[a]), int(b - a + 1))      # first core row, number of core rows
    # edges: share just inside the edge vs 2 rows outside
    ein = 0.5 * (X[a] + X[b])
    eout = 0.5 * (X[max(a - 2, 0)] + X[min(b + 2, len(X) - 1)])
    f["edge"] = (ein - eout) / max(core, 1e-3)
    # coverage end along the core rows
    vv = val[crow] > 0
    last = np.where(vv.any(1), NR - 1 - np.argmax(vv[:, ::-1], 1), 0)
    r_end = min(int(last.min()), R_MAX)
    f["gap_end"] = r_end - r1
    # constant azimuth and width per 10-km block
    rr = np.arange(r0, r1 + 1)
    wide = rows[max(a - 4, 0):min(b + 5, len(rows))]
    off = np.arange(len(wide)) - (a - max(a - 4, 0)) - (b - a) / 2.0
    Wb = wet[wide][:, r0:r1 + 1]
    cent, wid, rb = [], [], []
    for s in range(0, len(rr) - 4, 10):
        blk = Wb[:, s:s + 10]
        m = blk.sum(1)
        if m.sum() < 5:
            continue
        cent.append(float((m * off).sum() / m.sum()))
        wid.append(float((m >= 0.5 * blk.shape[1]).sum()))
        rb.append(float(rr[s:s + 10].mean()))
    f["nblk"] = len(rb)
    if len(rb) >= 3:
        rb_, cent_ = np.array(rb), np.array(cent)
        A = np.vstack([rb_ / 100.0, np.ones_like(rb_)]).T
        coef, res, *_ = np.linalg.lstsq(A, cent_, rcond=None)
        f["drift_r"] = float(coef[0] * DA)              # deg per 100 km
        f["cent_sd"] = float(np.std(cent_ - A @ coef) * DA)
        wid_ = np.array(wid)
        h = len(wid_) // 2
        wn, wf = wid_[:h].mean(), wid_[h:].mean()
        f["wratio"] = float((wf + 0.5) / (wn + 0.5))   # angular width far / near (spike ~1, rain ~ r_near / r_far)
        f["rratio"] = float(np.mean(rb_[h:]) / np.mean(rb_[:h]))
    else:
        f["drift_r"], f["cent_sd"], f["wratio"], f["rratio"] = np.nan, np.nan, np.nan, np.nan
    # range profile on the core rows
    nv = val[crow][:, r0:r1 + 1].sum(0)
    m = L[crow][:, r0:r1 + 1].sum(0) / np.maximum(nv, 1)
    mw = wet[crow][:, r0:r1 + 1].sum(0) / np.maximum(nv, 1)
    nb = len(m) // 4
    if nb >= 3:
        mb = m[:nb * 4].reshape(nb, 4).mean(1)
        f["rough"] = float(np.median(np.abs(np.diff(mb))))
        rbk = rr[:nb * 4].reshape(nb, 4).mean(1)
        f["slope"] = float(np.polyfit(np.log(rbk), mb, 1)[0])
        f["peak_ratio"] = float(np.expm1(mb.max()) / max(np.expm1(np.median(mb)), 0.5))
    else:
        f["rough"], f["slope"], f["peak_ratio"] = np.nan, np.nan, np.nan
    f["fill"] = float(mw.mean())
    f["lmean"] = float(m.mean())
    # clear-air share: range bins where both sides are dry
    sl_rows = rows[max(a - 9, 0):max(a - 1, 0)]
    sr_rows = rows[min(b + 2, len(rows)):min(b + 10, len(rows))]
    sd = np.maximum(wet[sl_rows][:, r0:r1 + 1].mean(0) if len(sl_rows) else 1, wet[sr_rows][:, r0:r1 + 1].mean(0) if len(sr_rows) else 1)
    f["ca"] = float(((sd <= 0.15) & (mw >= 0.4)).mean())
    f["_ctx"] = (rows, a, b, crow, X, v, core, con, r0, r1, t)
    if not time:
        return f
    return time_feats(f, P)


def time_feats(f, P):
    """temporal features (t-1, t+1, t-2, t+2) for a candidate whose geometry features() computed."""
    rows, a, b, crow, X, v, core, con, r0, r1, t = f["_ctx"]
    wet, val, L = P[t]
    Kn, Kr = 32, 25                       # +-8 deg beyond the core, +-25 km: a shower that jumped is still inside
    nb_rows = _rows(int(rows[(a + b) // 2]), (b - a) // 2 + Kn)
    rlo, rhi = max(r0 - Kr, 0), min(r1 + Kr, NR - 1)
    reg_rows = _rows(int(rows[(a + b) // 2]), 80)
    rglo, rghi = max(r0 - 40, 0), min(r1 + 40, NR - 1)
    core_t = float(wet[crow][:, r0:r1 + 1].sum())
    nb_t = float(wet[nb_rows][:, rlo:rhi + 1].sum())
    reg_t = float(wet[reg_rows][:, rglo:rghi + 1].sum())
    ia, ib = max(a - 12, 0), min(b + 12, len(X) - 1)
    Xt = X[ia:ib + 1]
    for name, u in (("p", t - 1), ("n", t + 1), ("p2", t - 2), ("n2", t + 2)):
        if u not in P:
            f[f"{name}_ok"] = 0
            continue
        wu, vu, Lu = P[u]
        Xu, Yu, vvu = az_profile(wu, vu, Lu, rows, r0, r1)
        if (vvu[a:b + 1] < 0.5 * v[a:b + 1]).any():
            f[f"{name}_ok"] = 0
            continue
        f[f"{name}_ok"] = 1
        f[f"{name}_con0"] = tophat(Xu, a, b)[0]
        # correlation of the azimuth profile with the profile at u shifted by -12..12 rows (vectorised)
        sfts = np.arange(-12, 13)
        okk = (ia + sfts >= 0) & (ib + sfts < len(Xu))
        sfts = sfts[okk]
        n_ = ib - ia + 1
        Ws = np.stack([Xu[ia + q:ia + q + n_] for q in sfts]).astype(np.float64)
        xt = Xt.astype(np.float64) - Xt.mean()
        wsd = Ws - Ws.mean(1, keepdims=True)
        den = np.sqrt((xt ** 2).sum() * (wsd ** 2).sum(1))
        cc = np.where(den > 1e-9, (wsd @ xt) / np.maximum(den, 1e-12), -1.0)
        cs0 = float(cc[sfts == 0][0]) if (sfts == 0).any() else np.nan
        mx = cc.max()
        cand_ = np.flatnonzero(cc >= mx - 1e-6)
        k_ = cand_[np.argmin(np.abs(sfts[cand_]))]
        best, bs = float(cc[k_]), int(sfts[k_])
        f[f"{name}_conS"] = float(tophat(Xu, a + bs, b + bs)[0]) if (a + bs - 9 >= 0 and b + bs + 9 < len(Xu)) else 0.0
        f[f"{name}_c0"] = cs0
        f[f"{name}_cb"] = best
        f[f"{name}_shift"] = bs * DA
        nb_u = float(wu[nb_rows][:, rlo:rhi + 1].sum())
        reg_u = float(wu[reg_rows][:, rglo:rghi + 1].sum())
        f[f"{name}_flr"] = (nb_u - (nb_t - core_t)) / max(core_t, 1.0)
        f[f"{name}_reg"] = (reg_u + 50.0) / (reg_t - core_t + 50.0)
    return f
