"""The tests behind the WOW-NL quality-control flags, as pure numpy functions.

Nothing here knows about stores, configs or bit numbers: arrays in, boolean or
three-state arrays out, so every test can be checked on a hand-made series
(tests/test_qc.py) and `meteopack.derive.qc` only wires them together.

Conventions:

* Within-station tests take one variable's non-null reports, sorted by
  (station, time): `sid` an integer station code that changes exactly where the
  station does, `t` seconds (float), `x` the values. A neighbour across a
  station boundary does not exist (NaN value, infinite time gap).
* Network tests work on dense matrices `V[s, j]` (station x time bin), NaN where
  a station has no usable value. Neighbour lists `nb[s, k]` are station indices,
  nearest first, padded with -1.
* Three-state results (PWSQC's convention): 1 flagged, 0 passed, -1 not
  evaluable. Robust scales are the median absolute deviation times 1.4826
  (CrowdQC+ uses Qn; MAD is the usual stand-in, with the same 50 % breakdown).
"""
from __future__ import annotations

import numpy as np

MAD_TO_SD = 1.4826


# ----------------------------------------------------------- within station


def same_as_previous(sid: np.ndarray) -> np.ndarray:
    """True where the previous row belongs to the same station."""
    same = np.zeros(len(sid), bool)
    same[1:] = sid[1:] == sid[:-1]
    return same


def prev_next(x: np.ndarray, t: np.ndarray, sid: np.ndarray):
    """Previous/next value of the same station and the time gaps to them."""
    n = len(x)
    sp = same_as_previous(sid)
    sn = np.zeros(n, bool)
    sn[:-1] = sp[1:]
    xp = np.full(n, np.nan)
    xn = np.full(n, np.nan)
    dtp = np.full(n, np.inf)
    dtn = np.full(n, np.inf)
    xp[1:] = x[:-1]
    xn[:-1] = x[1:]
    dtp[1:] = t[1:] - t[:-1]
    dtn[:-1] = t[1:] - t[:-1]
    xp[~sp] = np.nan
    dtp[~sp] = np.inf
    xn[~sn] = np.nan
    dtn[~sn] = np.inf
    return xp, xn, dtp, dtn


def outside(x: np.ndarray, lo: float | None, hi: float | None) -> np.ndarray:
    bad = np.zeros(len(x), bool)
    if lo is not None:
        bad |= x < lo
    if hi is not None:
        bad |= x > hi
    return bad


def sentinel(x: np.ndarray, values, tol: float) -> np.ndarray:
    bad = np.zeros(len(x), bool)
    for v in values:
        bad |= np.abs(x - v) <= tol
    return bad


def spike(x, xp, xn, dtp, dtn, threshold: float, max_dt: float) -> np.ndarray:
    """Up (or down) by more than `threshold` from both neighbours, each at most `max_dt` away."""
    up = (x - xp > threshold) & (x - xn > threshold)
    down = (xp - x > threshold) & (xn - x > threshold)
    return (up | down) & (dtp <= max_dt) & (dtn <= max_dt)


def step(x, xp, dtp, threshold: float, max_dt: float) -> np.ndarray:
    """Changed by more than `threshold` since a previous report at most `max_dt` earlier."""
    return (np.abs(x - xp) > threshold) & (dtp <= max_dt)


def run_durations(x: np.ndarray, t: np.ndarray, sid: np.ndarray) -> np.ndarray:
    """Per report: seconds from the first to the last report of its run of identical values."""
    if len(x) == 0:
        return np.zeros(0)
    same = same_as_previous(sid)
    xp = np.empty_like(x)
    xp[0] = np.nan
    xp[1:] = x[:-1]
    start = ~same | (x != xp)
    rid = np.cumsum(start) - 1
    last = np.ones(len(x), bool)
    last[:-1] = start[1:]
    return (t[last] - t[start])[rid]


def persistence(x, t, sid, min_duration: float | None, zero_min_duration: float | None,
                zero_is_special: bool):
    """(stuck, stuck_zero): runs of one value lasting at least the given durations.

    With `zero_is_special` (wind, rain rate) a run of zeros is judged by
    `zero_min_duration` alone - calm is normal, a calm week is not - and the
    stuck bit is for non-zero values only.
    """
    dur = run_durations(x, t, sid)
    n = len(x)
    stuck = np.zeros(n, bool)
    stuck_zero = np.zeros(n, bool)
    if zero_is_special:
        zero = x == 0
        if min_duration is not None:
            stuck = ~zero & (dur >= min_duration)
        if zero_min_duration is not None:
            stuck_zero = zero & (dur >= zero_min_duration)
    elif min_duration is not None:
        stuck = dur >= min_duration
    return stuck, stuck_zero


def stale_repeat(cols: list[np.ndarray], t: np.ndarray, sid: np.ndarray, gap_s: float):
    """The first report after a gap longer than `gap_s` repeats the previous one exactly.

    Every column is compared (NaN equals NaN) and at least one must hold a
    value: a station that comes back after an outage re-sending its last
    record (de Vos et al. 2017 saw Netatmo do this) has not measured anything.
    """
    n = len(t)
    same = same_as_previous(sid)
    gap = np.zeros(n, bool)
    gap[1:] = (t[1:] - t[:-1]) > gap_s
    eq = same & gap
    anyval = np.zeros(n, bool)
    for c in cols:
        prev = np.empty_like(c)
        prev[0] = np.nan
        prev[1:] = c[:-1]
        eq &= (c == prev) | (np.isnan(c) & np.isnan(prev))
        anyval |= ~np.isnan(c)
    return eq & anyval


def dew_point_magnus(t_c: np.ndarray, rh: np.ndarray) -> np.ndarray:
    """Dew point (degC) from temperature and RH, Magnus form (Alduchov & Eskridge 1996)."""
    a, b = 17.625, 243.04
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.log(rh / 100.0) + a * t_c / (b + t_c)
        return b * g / (a - g)


# ---------------------------------------------------------------- matrices


def bin_last(s: np.ndarray, j: np.ndarray, x: np.ndarray, n_s: int, n_j: int) -> np.ndarray:
    """V[s, j]: the last value (in input order) of each (station, bin); NaN elsewhere."""
    v = np.full(n_s * n_j, np.nan, np.float32)
    if len(x):
        key = s.astype(np.int64) * n_j + j
        order = np.argsort(key, kind="stable")
        key = key[order]
        last = np.ones(len(key), bool)
        last[:-1] = key[1:] != key[:-1]
        v[key[last]] = x[order][last]
    return v.reshape(n_s, n_j)


def ffill_limited(v: np.ndarray, limit: int) -> np.ndarray:
    """Carry each row's last value forward at most `limit` columns."""
    if limit <= 0:
        return v
    n_j = v.shape[1]
    cols = np.arange(n_j)
    idx = np.where(~np.isnan(v), cols[None, :], -1)
    np.maximum.accumulate(idx, axis=1, out=idx)
    ok = (idx >= 0) & (cols[None, :] - idx <= limit)
    out = np.take_along_axis(v, np.maximum(idx, 0), axis=1)
    out[~ok] = np.nan
    return out


def _sorted_median(srt: np.ndarray, n: np.ndarray) -> np.ndarray:
    lo = np.maximum((n - 1) // 2, 0)[..., None]
    hi = np.maximum(n // 2, 0)[..., None]
    a = np.take_along_axis(srt, lo, -1)[..., 0]
    b = np.take_along_axis(srt, hi, -1)[..., 0]
    med = (a + b) / 2
    return np.where(n > 0, med, np.nan)


def nan_median_mad(a: np.ndarray):
    """(median, MAD, count) over the last axis, ignoring NaN; one sort each."""
    n = (~np.isnan(a)).sum(-1)
    med = _sorted_median(np.sort(a, axis=-1), n)
    dev = np.abs(a - med[..., None])
    mad = _sorted_median(np.sort(dev, axis=-1), n)
    return med, mad, n


def network_stats(v: np.ndarray):
    """Per bin (column): median, MAD and count over all stations."""
    return nan_median_mad(np.ascontiguousarray(v.T))


def neighbour_stats(v: np.ndarray, nb: np.ndarray, chunk: int = 256, mad: bool = True):
    """Per (station, bin): median, MAD and count of the station's neighbours' values.

    The station itself is never among its neighbours (leave-one-out), so a bad
    value cannot vouch for itself.
    """
    n_s, n_j = v.shape
    med = np.full((n_s, n_j), np.nan, np.float32)
    dev = np.full((n_s, n_j), np.nan, np.float32)
    cnt = np.zeros((n_s, n_j), np.int16)
    if nb.shape[1] == 0 or n_s == 0:
        return med, dev, cnt
    vp = np.vstack([v, np.full((1, n_j), np.nan, v.dtype)])
    nbp = np.where(nb < 0, n_s, nb)
    for c0 in range(0, n_s, chunk):
        w = np.moveaxis(vp[nbp[c0:c0 + chunk]], 1, 2)          # (c, J, K)
        if mad:
            m, d, c = nan_median_mad(w)
            dev[c0:c0 + chunk] = d
        else:
            c = (~np.isnan(w)).sum(-1)
            m = _sorted_median(np.sort(w, axis=-1), c)
        med[c0:c0 + chunk] = m
        cnt[c0:c0 + chunk] = c
    return med, dev, cnt


def pearson_rows(v: np.ndarray, m: np.ndarray):
    """Pearson r of each row of `v` with the series `m` over bins where both exist; and n."""
    ok = ~np.isnan(v) & ~np.isnan(m)[None, :]
    n = ok.sum(1)
    x = np.where(ok, v, 0.0).astype(np.float64)
    y = np.where(ok, m[None, :], 0.0).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mx = x.sum(1) / n
        my = y.sum(1) / n
        dx = np.where(ok, x - mx[:, None], 0.0)
        dy = np.where(ok, y - my[:, None], 0.0)
        r = (dx * dy).sum(1) / np.sqrt((dx * dx).sum(1) * (dy * dy).sum(1))
    return r, n


def neighbour_lists(xy: np.ndarray, radius: float, k_max: int | None,
                    eligible: np.ndarray | None = None) -> np.ndarray:
    """nb[s, k]: eligible stations within `radius` of s, nearest first, -1 padded.

    `xy` in metres (any local projection). Brute force: a WOW month has at most
    ~1,500 stations.
    """
    n = len(xy)
    if n == 0:
        return np.zeros((0, 0), np.int32)
    d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, np.inf)
    d[np.isnan(d)] = np.inf
    if eligible is not None:
        d[:, ~eligible] = np.inf
    d[d > radius] = np.inf
    k = int(np.isfinite(d).sum(1).max()) if n else 0
    if k_max is not None:
        k = min(k, k_max)
    if k == 0:
        return np.full((n, 0), -1, np.int32)
    order = np.argsort(d, axis=1, kind="stable")[:, :k]
    nb = order.astype(np.int32)
    nb[~np.isfinite(np.take_along_axis(d, order, 1))] = -1
    return nb


# --------------------------------------------------------------------- rain


def spread_increments(s: np.ndarray, t_end: np.ndarray, dt: np.ndarray, inc: np.ndarray,
                      t0: float, step: float, n_s: int, n_j: int, max_span: float,
                      min_coverage: float):
    """Counter increments spread at a constant rate onto bins (de Vos et al. 2017).

    Bin j covers (t0 + j*step, t0 + (j+1)*step]. An increment covers
    (t_end - dt, t_end]; it is spread over the bins it overlaps in proportion to
    the overlap, if `dt <= max_span` (de Vos: intervals longer than 20 min are
    discarded). A bin is a value only if increments cover at least
    `min_coverage` of it; otherwise NaN.

    Returns R (n_s, n_j) and the (report, bin) pairs each increment touched,
    as (report index into the inputs, bin index).
    """
    ok = np.isfinite(inc) & np.isfinite(dt) & (dt > 0) & (dt <= max_span)
    dt = np.where(ok, dt, 1.0)
    a = t_end - dt
    j0 = np.floor((a - t0) / step).astype(np.int64)
    j1 = np.ceil((t_end - t0) / step).astype(np.int64) - 1
    j0 = np.maximum(j0, 0)
    j1 = np.minimum(j1, n_j - 1)
    cnt = np.where(ok, np.maximum(j1 - j0 + 1, 0), 0)
    rep = np.repeat(np.arange(len(inc)), cnt)
    first = np.repeat(np.cumsum(cnt) - cnt, cnt)
    jj = j0[rep] + (np.arange(len(rep)) - first)
    lo_b = t0 + jj * step
    ov = np.minimum(t_end[rep], lo_b + step) - np.maximum(a[rep], lo_b)
    keep = ov > 0
    rep, jj, ov = rep[keep], jj[keep], ov[keep]
    key = s[rep].astype(np.int64) * n_j + jj
    amount = np.bincount(key, weights=inc[rep] * ov / dt[rep], minlength=n_s * n_j)
    cover = np.bincount(key, weights=ov, minlength=n_s * n_j)
    r = amount.reshape(n_s, n_j)
    r[cover.reshape(n_s, n_j) < min_coverage * step - 1e-6] = np.nan
    return r, rep, jj


def pwsqc_high_influx(r: np.ndarray, med: np.ndarray, cnt: np.ndarray, nstat: int,
                      thr_a: float, thr_b: float) -> np.ndarray:
    """PWSQC HI (de Vos et al. 2019; PWSQC_2.R): rain far above the neighbours' median."""
    with np.errstate(invalid="ignore"):
        hi = ((r > thr_b) & (med < thr_a)) | ((med >= thr_a) & (r > thr_b * med / thr_a))
    out = hi.astype(np.int8)
    out[cnt < nstat] = -1
    return out


def _runs(mask: np.ndarray):
    """(start, end) inclusive of each run of True."""
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1


def pwsqc_faulty_zeros(r: np.ndarray, med: np.ndarray, cnt: np.ndarray, nstat: int,
                       nint: int) -> np.ndarray:
    """PWSQC FZ (PWSQC_2.R, line for line).

    In each run of more than `nint` consecutive zeros of the station (a missing
    value breaks a run), find the first run of more than `nint` consecutive
    intervals in which the neighbours' median is above zero; flag from `nint`
    intervals into it to the end of the dry run, then keep flagging through
    zeros and missing values until the station reports rain. Wherever fewer than
    `nstat` neighbours report, -1.
    """
    n_s, n_j = r.shape
    out = np.zeros((n_s, n_j), np.int8)
    ref = np.where(cnt < nstat, np.nan, (med > 0).astype(np.float32))
    for s in range(n_s):
        x = r[s]
        dry = x == 0                         # NaN is not dry: it breaks the run
        if not dry.any():
            continue
        wet_ref = ref[s] == 1
        for a, b in zip(*_runs(dry), strict=True):
            if b - a + 1 <= nint or wet_ref[a:b + 1].sum() <= nint:
                continue
            ra, rb = _runs(wet_ref[a:b + 1])
            long = (rb - ra + 1) > nint
            if not long.any():
                continue
            start = a + ra[long][0] + nint
            out[s, start:b + 1] = 1
            after = np.flatnonzero(x[b + 1:] > 0)
            stop = b + 1 + (after[0] if len(after) else n_j - b - 1)
            out[s, b + 1:stop] = 1
    out[cnt < nstat] = -1
    return out


def pwsqc_station_outlier(r: np.ndarray, nb: np.ndarray, nstat: int, per: int,
                          compare: int, rainy: int, match: int, corthres: float,
                          first_eval: int, chunk: int = 64) -> np.ndarray:
    """PWSQC SO (PWSQC_3.R), evaluated once per block of `per` bins.

    At the end of each block e (from block `first_eval` on) the window is the
    last `compare` blocks, extended further back until the station has at
    least `rainy` wet intervals in it. Each neighbour with more than `match`
    intervals in common with the station counts with its Pearson r. Flag if
    the median r is below `corthres`; -1 if fewer than `nstat` neighbours count
    or the window cannot be formed (not enough history, or not enough rain in
    it). The block's verdict applies to all its bins. `r` must already have the
    FZ/HI-flagged values removed (NaN). PWSQC evaluates at every 5-min
    interval; once per block keeps it affordable.
    """
    import warnings

    n_s, n_j = r.shape
    n_b = n_j // per
    out = np.full((n_s, n_j), -1, np.int8)
    if n_b == 0 or nb.shape[1] == 0 or first_eval >= n_b:
        return out
    rr = r[:, :n_b * per]
    zero = np.zeros((n_s, 1))
    wet = np.concatenate([zero, np.cumsum((rr > 0).reshape(n_s, n_b, per).sum(2), axis=1)],
                         axis=1)                                      # (S, B+1)
    e = np.arange(first_eval, n_b)
    start_b = e - compare + 1                                         # (E,)
    need = wet[:, e + 1] - rainy             # the window must start where wet <= need
    start_a = np.empty((n_s, len(e)), np.int64)
    for s in range(n_s):
        start_a[s] = np.searchsorted(wet[s], need[s], side="right") - 1
    usable = (need >= 0) & (start_a >= 0) & (start_b[None, :] >= 0)
    start = np.clip(np.minimum(start_a, start_b[None, :]), 0, n_b)
    nbp = np.where(nb < 0, n_s, nb)
    rp = np.vstack([rr, np.full((1, rr.shape[1]), np.nan)]).astype(np.float32)

    def blocks(q):                                                    # (c, K, J) -> (c, K, B+1)
        b = q.reshape(q.shape[:2] + (n_b, per)).sum(3, dtype=np.float64)
        return np.concatenate([np.zeros(b.shape[:2] + (1,)), np.cumsum(b, axis=2)], axis=2)

    for c0 in range(0, n_s, chunk):
        idx = np.arange(c0, min(c0 + chunk, n_s))
        x = rp[idx][:, None, :]
        y = rp[nbp[idx]]
        ok = ~np.isnan(x) & ~np.isnan(y)
        xv = np.where(ok, x, np.float32(0))
        yv = np.where(ok, y, np.float32(0))
        del x, y
        st = start[idx][:, None, :]
        en = (e + 1)[None, None, :]
        stats = []
        for q in (ok, xv, yv, xv * xv, yv * yv, xv * yv):
            c = blocks(q)
            a = np.take_along_axis(c, np.broadcast_to(st, (c.shape[0], c.shape[1], len(e))), 2)
            b = np.take_along_axis(c, np.broadcast_to(en, a.shape), 2)
            stats.append(b - a)
        n, sx, sy, sxx, syy, sxy = stats
        with np.errstate(invalid="ignore", divide="ignore"):
            var = (n * sxx - sx * sx) * (n * syy - sy * sy)
            rho = (n * sxy - sx * sy) / np.sqrt(var)
        rho[(n <= match) | ~(var > 0)] = np.nan
        counted = (~np.isnan(rho)).sum(1)                             # (c, E)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(rho, axis=1)
        flag = (med < corthres).astype(np.int8)
        flag[(counted < nstat) | ~usable[idx]] = -1
        out[idx, first_eval * per:n_b * per] = np.repeat(flag, per, axis=1)
    return out


def three_state_per_report(flags: np.ndarray, s_rep: np.ndarray, rep: np.ndarray,
                           jj: np.ndarray, n_reports: int):
    """Collapse bin flags onto the reports whose increments touched them.

    A report fails if any of its bins is flagged, is not evaluable if none is
    flagged but any is -1, and is not evaluable when it touched no bin at all.
    """
    v = flags[s_rep[rep], jj] if len(rep) else np.zeros(0, np.int8)
    fail = np.zeros(n_reports, bool)
    ne = np.zeros(n_reports, bool)
    touched = np.zeros(n_reports, bool)
    np.logical_or.at(fail, rep, v == 1)
    np.logical_or.at(ne, rep, v == -1)
    touched[rep] = True
    return fail, (ne & ~fail) | ~touched
