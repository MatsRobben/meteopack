"""residue: final clean-up of faint leftovers of removed clutter (v14 final slot, night 2026-11-16).

    residue_clean(out, raw, flags, flr, aggr, times=None, s2mask=None, ret_info=False, **kw) -> out [, info]

out   = finished cleaner output (T, 765, 700) uint16; raw = raw codes; flags = frozen-half flags (lib/stale);
flr   = v10 flows (ctx.st["flr"]: per frame (flow on the 4 km grid, scale)); aggr = v14 aggr map (T, ny, nx) or (T,)
s2mask = optional bool (T, H, W): pixels lowered by spikes2 (its corridors).
Returns out' <= out; codes >= 65534 unchanged; new-chain frames unchanged.

The cleaner often removes / lowers nearly all of a clutter instance's AMOUNT but leaves many faint pixels: clamped
remnants, small leftovers next to removed clutter, stubs. Rules (per frame, 8-connected objects of out):
 1. LOWERED / FLOOR REMNANT: >= low_share of the object's px are below their raw value OR at the floor
    (code <= faint = 2, <= 0.24 mm/h: the stamped-on lowest-class blocks the reviewers saw, raw-valued leftovers of
    clamped anaprop) and the object is small (<= low_max px) or has no coherent core (core_kept: 3 x 3 opening of
    the px that are neither lowered nor at the floor) -> removed. Large coherent rain whose values were capped
    (spikes2 strips inside rain) is untouched: big objects with a real core.
 2. FRAGMENT: object <= frag_max px whose raw neighbourhood (frag_box box around its px) holds >= frag_ratio x more
    removed echo px (raw wet, out dry) than kept echo px -> removed.
 3. SPECK: object <= speck_max px without echo of out within speck_r px of its motion-shifted position at t-1 AND
    t+1 (both sides usable: good frame, rows not a frozen copy, not missing) -> removed.
 4. GATE: rules 1-3 only where the v13 tile load is elevated: aggr >= a_gate (aggr = 1 + smoothstep((load -
    0.20) / 0.25), rate limited; aggr 1.8 ~ load 0.38, where the stc coherent-core gate L = g x aggr reaches
    ccf_on 0.25), with a running max over +-gate_tmax frames; the object's mean gate >= 0.5. Rule 1 also applies
    inside the spikes2 corridors (object touching s2mask dilated by 2 px) without the gate.
 5. TEMPORAL CONSISTENCY along the motion (shift k v, search radius tc_r): an object (<= tc_max px, inside the
    gate, a speck or with lowered/floor share >= tc_fshare) whose px at t-1 and t+1 both fall on removed residue
    (>= tc_share) is removed too; an object removed by
    rule 2 or 3 whose px at t-1 and t+1 both fall on kept echo (>= tc_share) is kept (rule 1 removals stay).
Then frozen halves stay copies (as stc / v12).
"""
# Ported from research proto/residue/residue.py (md5 fbc9010583549fac50fa254f606ad2c3),
# by port_tools/sync_research.py: only imports / loading edited.
from __future__ import annotations

import time

import cv2
import numpy as np

MISSING = 65534
H, W = 765, 700
DS = 4
NEW_CHAIN = np.datetime64("2023-01-31T10:45", "ns")
HALVES = ((1, slice(0, 380)), (2, slice(380, H)))

DEFAULTS = dict(
    low_share=0.55, low_max=40, low_core=3, core_kept=True, faint=2, tc_fshare=0.4,
    frag_max=20, frag_box=7, frag_ratio=3.0,
    speck_max=3, speck_r=3,
    a_gate=1.8, gate_tmax=3, gate_share=0.5, s2_dil=2, s2_rule1=True,
    tc=True, tc_max=1500, tc_r=2, tc_share=0.5,
    rules=(1, 2, 3),
)


def _flow(flr, t, py, px):
    fl, s = flr[min(t, len(flr) - 1)]
    v = fl[np.minimum(py // DS, fl.shape[0] - 1), np.minimum(px // DS, fl.shape[1] - 1)] * (DS * s)
    return v[:, 0].astype(np.float32), v[:, 1].astype(np.float32)


def gate_maps(aggr, T, a_gate=1.8, gate_tmax=3):
    a = np.asarray(aggr, np.float32)
    if a.ndim == 0:
        a = np.full(T, float(a), np.float32)
    g = (a >= a_gate).astype(np.float32)
    if gate_tmax > 0:
        g = np.stack([g[max(t - gate_tmax, 0):t + gate_tmax + 1].max(0) for t in range(T)])
    return g


def _gate_px(g, t):
    if g.ndim == 1:
        return np.full((H, W), float(g[t]), np.float32)
    return cv2.resize(g[t], (W, H), interpolation=cv2.INTER_LINEAR)


def _shift_sample(mask_d, ok_rows, valid_u, py, px, vx, vy, k):
    """sample a dilated mask of frame u = t + k at p + k v. Returns (value bool, usable bool)."""
    ys = np.rint(py + k * vy).astype(np.int64); xs = np.rint(px + k * vx).astype(np.int64)
    ins = (ys >= 0) & (ys < H) & (xs >= 0) & (xs < W)
    yi, xi = np.where(ins, ys, 0), np.where(ins, xs, 0)
    use = ins & ok_rows[yi] & valid_u[yi, xi]
    return mask_d[yi, xi] & use, use


def residue_clean(out, raw, flags, flr, aggr, times=None, s2mask=None, ret_info=False, **kw):
    p = {**DEFAULTS, **kw}
    t0 = time.time()
    out = np.asarray(out); raw = np.asarray(raw)
    T = len(out)
    old = np.ones(T, bool) if times is None else (np.asarray(times).astype("datetime64[ns]") < NEW_CHAIN)
    valid = raw < MISSING
    ok = np.ones((T, H), bool)
    for u in range(T):
        for bit, sl in HALVES:
            if flags[u] & bit:
                ok[u, sl] = False
    cov = valid.reshape(T, -1).mean(1)
    good = cov >= 0.5 * max(cov.max(), 1e-9)
    G = gate_maps(aggr, T, p["a_gate"], p["gate_tmax"])
    rules = set(p["rules"])
    k3 = np.ones((p["low_core"], p["low_core"]), np.uint8)
    k8 = np.ones((3, 3), np.uint8)
    ktc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p["tc_r"] + 1, 2 * p["tc_r"] + 1))
    ksp = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p["speck_r"] + 1, 2 * p["speck_r"] + 1))
    ks2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p["s2_dil"] + 1, 2 * p["s2_dil"] + 1))
    wet = (out > 0) & valid
    wetd = {}

    def wd(u):  # out wet dilated by speck_r
        if u not in wetd:
            wetd[u] = cv2.dilate(wet[u].view(np.uint8), ksp).astype(bool)
        return wetd[u]

    # ---- pass 1: per-frame decisions
    F = []                                    # per frame: (lab, n, rm1 (n+1,), rm23 (n+1,), gated (n+1,), area)
    cnt = {"r1": 0, "r2": 0, "r3": 0, "tc_add": 0, "tc_keep": 0}
    for t in range(T):
        if not old[t] or not wet[t].any():
            F.append(None); continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(wet[t].view(np.uint8), connectivity=8)
        area = stats[:, cv2.CC_STAT_AREA].astype(np.int64); area[0] = 0
        lf = lab.ravel()
        gpx = _gate_px(G, t)
        gsh = np.bincount(lf, weights=gpx.ravel(), minlength=n) / np.maximum(area, 1)
        gated = gsh >= p["gate_share"]
        if s2mask is not None and p["s2_rule1"] and s2mask[t].any():
            s2d = cv2.dilate(s2mask[t].view(np.uint8), ks2).astype(bool)
            ins2 = np.bincount(lf, weights=s2d.ravel(), minlength=n) > 0
        else:
            ins2 = np.zeros(n, bool)
        gated[0] = False; ins2[0] = False
        rm1 = np.zeros(n, bool); rm23 = np.zeros(n, bool)
        lsh = np.zeros(n)
        cand = gated | ins2
        if cand.any():
            if 1 in rules:
                low = wet[t] & ((out[t] < raw[t]) | (out[t] <= p["faint"]))
                lsh = np.bincount(lf, weights=low.ravel(), minlength=n) / np.maximum(area, 1)
                cb = (wet[t] & ~low) if p["core_kept"] else wet[t]
                core = cv2.morphologyEx(cb.view(np.uint8), cv2.MORPH_OPEN, k3).astype(bool)
                ncore = np.bincount(lf, weights=core.ravel(), minlength=n)
                rm1 = cand & (lsh >= p["low_share"]) & ((area <= p["low_max"]) | (ncore == 0))
            if 2 in rules:
                rem = (raw[t] > 0) & valid[t] & ~wet[t]
                bx = (p["frag_box"], p["frag_box"])
                br = cv2.boxFilter(rem.astype(np.float32), -1, bx, normalize=False)
                bk = cv2.boxFilter(wet[t].astype(np.float32), -1, bx, normalize=False)
                sr = np.bincount(lf, weights=br.ravel(), minlength=n)
                sk = np.bincount(lf, weights=bk.ravel(), minlength=n)
                rm23 |= gated & (area <= p["frag_max"]) & (sr >= p["frag_ratio"] * np.maximum(sk, 1e-6))
            if 3 in rules:
                sm = gated & (area <= p["speck_max"]) & ~rm23 & ~rm1
                if sm.any():
                    py, px = np.nonzero(sm[lab])
                    vx, vy = _flow(flr, t, py, px)
                    l = lab[py, px]
                    dry_both = np.ones(n, bool); dry_both[0] = False
                    for k in (-1, 1):
                        u = t + k
                        if not (0 <= u < T) or not good[u]:
                            dry_both[:] = False; break
                        hit, use = _shift_sample(wd(u), ok[u], valid[u], py, px, vx, vy, k)
                        nh = np.bincount(l, weights=hit, minlength=n)
                        nu = np.bincount(l, weights=use, minlength=n)
                        dry_both &= (nh == 0) & (nu >= 0.5 * np.maximum(area, 1))
                    rm23 |= sm & dry_both
        cnt["r1"] += int(rm1.sum()); cnt["r2"] += int((rm23 & ~rm1).sum())
        F.append([lab, n, rm1, rm23 & ~rm1, gated, area, lsh])
    del wetd

    # ---- pass 2: temporal consistency along the motion
    RMD, KPD = {}, {}

    def masks(u):
        if u not in RMD:
            f = F[u]
            if f is None:
                RMD[u] = KPD[u] = None
            else:
                lab, n, rm1, rm23, _, _, _ = f
                r = (rm1 | rm23)[lab]
                RMD[u] = cv2.dilate(r.view(np.uint8), ktc).astype(bool)
                KPD[u] = cv2.dilate((wet[u] & ~r).view(np.uint8), ktc).astype(bool)
        return RMD[u], KPD[u]

    final_rm = []
    for t in range(T):
        f = F[t]
        if f is None:
            final_rm.append(None); continue
        lab, n, rm1, rm23, gated, area, lsh = f
        rm = rm1 | rm23
        if p["tc"]:
            chk = gated & (area <= p["tc_max"])
            if chk.any():
                py, px = np.nonzero(chk[lab])
                vx, vy = _flow(flr, t, py, px)
                l = lab[py, px]
                rm_both = chk.copy(); kp_both = chk.copy()
                for k in (-1, 1):
                    u = t + k
                    if not (0 <= u < T) or F[u] is None or not good[u]:
                        rm_both[:] = False; kp_both[:] = False; break
                    R, K = masks(u)
                    hr, use = _shift_sample(R, ok[u], valid[u], py, px, vx, vy, k)
                    hk, _ = _shift_sample(K, ok[u], valid[u], py, px, vx, vy, k)
                    a = np.maximum(area, 1)
                    rm_both &= np.bincount(l, weights=hr, minlength=n) >= p["tc_share"] * a
                    kp_both &= (np.bincount(l, weights=hk, minlength=n) >= p["tc_share"] * a) & \
                               (np.bincount(l, weights=hr, minlength=n) < p["tc_share"] * a)
                add = rm_both & ~rm & ((area <= p["speck_max"]) | (lsh >= p["tc_fshare"]))
                keep = kp_both & rm23 & ~rm1
                cnt["tc_add"] += int(add.sum()); cnt["tc_keep"] += int(keep.sum())
                rm = (rm | add) & ~keep
        rm[0] = False
        final_rm.append(rm[lab] if rm.any() else None)
        for u in [x for x in RMD if x < t - 1]:
            RMD.pop(u); KPD.pop(u)

    res = out.copy()
    for t in range(T):
        if final_rm[t] is not None:
            res[t][final_rm[t]] = 0
    # frozen halves stay copies
    for t in range(1, T):
        if not old[t]:
            continue
        for bit, sl in HALVES:
            if flags[t] & bit:
                same = raw[t][sl] == raw[t - 1][sl]
                res[t][sl] = np.where(same, np.minimum(res[t - 1][sl], out[t][sl]), res[t][sl])
    res = np.where(valid, np.minimum(res, out), out).astype(out.dtype)
    res[~old] = out[~old]
    if ret_info:
        cnt["px_removed"] = int(((res == 0) & (out > 0) & valid).sum())
        cnt["secs"] = round(time.time() - t0, 2)
        return res, cnt
    return res
