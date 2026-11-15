"""Temporal layer: pool per-frame object evidence along object tracks, then decide once per object.

    links = build_links(labs, flows, frozen)                 # object graph between consecutive frames
    S = pool(links, ev, w, lam)                               # track-pooled evidence per object (forward+backward)

Composable: it takes per-frame objects (label images), per-frame object evidence (any number of columns, e.g.
the two classifiers' clutter probabilities) and the motion field; it returns pooled evidence per object.
The caller turns pooled evidence into a removal decision (see cleaner.py).

Tracks. Object i in frame t and object j in frame t+1 are linked when they overlap either at the same place
(Eulerian: clutter is stationary) or after moving t+1 back along the flow (Lagrangian: rain). The link weight
c_ij is the share of i covered by j (and, for the backward pass, the share of j covered by i), using the
better of the two correspondences for that pair, with a 1-px tolerance (3x3 dilation of the other frame's
objects). Splits and merges follow automatically (an object inherits from every object it overlaps, in
proportion to the overlap).

Pooling is a recursive (information-filter) smoother over that graph, forward and backward in time:
  forward   Sf_i = w_i e_i + lam * sum_j c_ij Sf_j        Nf_i = w_i + lam * sum_j c_ij Nf_j    (j in t-1)
  backward  Sb_i likewise over t+1
  pooled    e~_i = (Sf_i + Sb_i - w_i e_i) / (Nf_i + Nb_i - w_i)
so evidence from k frames away along the track counts with weight lam^|k| (times overlap shares).
w_i is the object's own evidence weight: 0 for the part of an object that lies in a frozen (copied) half of
the frame (lib/stale.py), so copies are missing evidence, never confirmation; they still pass messages.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import sparse

K3 = np.ones((3, 3), np.uint8)
_GRID = {}


def _grid(shape):
    if shape not in _GRID:
        yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(np.float32)
        _GRID[shape] = (xx, yy)
    return _GRID[shape]


def _pairs(a, b, na, nb):
    """sparse counts (na+1, nb+1) of label pairs (a, b) over pixels (1-D int arrays)."""
    m = (a > 0) & (b > 0)
    a, b = a[m], b[m]
    if len(a) == 0:
        return sparse.csr_matrix((na + 1, nb + 1))
    return sparse.csr_matrix((np.ones(len(a), np.float32), (a, b)), shape=(na + 1, nb + 1))


def build_links(labs, ns, flows, *, tol=True, lagrange=True, euler=True, norm="own"):
    """For t = 0..T-2: (A_t, B_t) sparse matrices; A_t[i, j] = share of object i (frame t) linked to object j
    (frame t+1), B_t[j, i] = share of object j linked to object i. Index 0 = background (row/col zero).
    n_ij = number of pixels of i that fall on j (3x3-dilated) at the same place (Eulerian) or after moving
    along the flow (Lagrangian); the larger of the two counts is used. A = n/|i|, B = n'/|j|, row sums capped
    at 1. flows[t]: (2, H, W) motion per 5 min for t -> t+1 (dx, dy px)."""
    T = len(labs)
    out = []
    for t in range(T - 1):
        la, lb = labs[t], labs[t + 1]
        na, nb = ns[t], ns[t + 1]
        if na == 0 or nb == 0:
            out.append((sparse.csr_matrix((na + 1, nb + 1)), sparse.csr_matrix((nb + 1, na + 1))))
            continue
        H, W = la.shape
        pa = np.flatnonzero(la)
        ia = la.ravel()[pa].astype(np.int64)
        sza = np.bincount(ia, minlength=na + 1).astype(np.float64)
        szb = np.bincount(lb.ravel(), minlength=nb + 1).astype(np.float64)
        lbd = cv2.dilate(lb.astype(np.uint16) if nb < 65535 else lb.astype(np.float32), K3) if tol else lb
        lbd = lbd.ravel()
        keys = []
        if euler:
            keys.append(ia * (nb + 1) + lbd[pa].astype(np.int64))
        if lagrange and flows is not None:
            y, x = np.divmod(pa, W)
            vx, vy = flows(t, y, x) if callable(flows) else (flows[t][0].ravel()[pa], flows[t][1].ravel()[pa])
            xs = np.clip(np.rint(x + vx), 0, W - 1).astype(np.int64)
            ys = np.clip(np.rint(y + vy), 0, H - 1).astype(np.int64)
            keys.append(ia * (nb + 1) + lbd[ys * W + xs].astype(np.int64))
        nk = (na + 1) * (nb + 1)
        if nk <= 4_000_000:
            cnt = np.bincount(keys[0], minlength=nk)
            for k in keys[1:]:
                cnt = np.maximum(cnt, np.bincount(k, minlength=nk))
            kk = np.flatnonzero(cnt); cc = cnt[kk].astype(np.float64)
        else:
            allk = np.concatenate(keys); tag = np.repeat(np.arange(len(keys)), [len(k) for k in keys])
            u, inv = np.unique(allk * len(keys) + tag, return_inverse=True)
            c_ = np.bincount(inv)
            kk_all = u // len(keys)
            kk, first = np.unique(kk_all, return_index=True)
            cc = np.maximum.reduceat(c_, first).astype(np.float64)
        ii, jj = np.divmod(kk, nb + 1)
        m = jj > 0
        ii, jj, cc = ii[m], jj[m], cc[m]
        if norm == "max":
            # overlap relative to the LARGER of the two objects: a small object merging into (or splitting from)
            # a much larger one does not inherit the large object's verdict, which is about other pixels
            v = np.minimum(cc / np.maximum(np.maximum(sza[ii], szb[jj]), 1), 1)
            A = _cap(sparse.csr_matrix((v, (ii, jj)), shape=(na + 1, nb + 1)))
            Bm = _cap(sparse.csr_matrix((v, (jj, ii)), shape=(nb + 1, na + 1)))
        else:
            A = _cap(sparse.csr_matrix((cc / sza[ii], (ii, jj)), shape=(na + 1, nb + 1)))
            Bm = _cap(sparse.csr_matrix((np.minimum(cc / np.maximum(szb[jj], 1), 1), (jj, ii)), shape=(nb + 1, na + 1)))
        out.append((A, Bm))
    return out


def _cap(M):
    s = np.asarray(M.sum(1)).ravel()
    f = np.where(s > 1, 1 / np.maximum(s, 1e-9), 1.0)
    return (sparse.diags(f) @ M).tocsr()


def pool(links, ev, w, lam=0.8):
    """Track-pooled evidence. ev[t]: (n_t, E) evidence per object; w[t]: (n_t,) own weight. Returns
    (pooled (n_t, E), support N (n_t,)) per frame."""
    T = len(ev)
    E = ev[0].shape[1] if T and ev[0].ndim == 2 else 1
    own = [np.vstack([np.zeros((1, E)), (w[t][:, None] * ev[t].reshape(len(w[t]), E))]) for t in range(T)]
    ownN = [np.r_[0.0, w[t]] for t in range(T)]
    Sf, Nf = [None] * T, [None] * T
    for t in range(T):
        if t == 0:
            Sf[t], Nf[t] = own[t].copy(), ownN[t].copy()
        else:
            Bm = links[t - 1][1]          # (n_t+1, n_{t-1}+1): share of object in t covered by objects in t-1
            Sf[t] = own[t] + lam * (Bm @ Sf[t - 1]); Nf[t] = ownN[t] + lam * (Bm @ Nf[t - 1])
    Sb, Nb = [None] * T, [None] * T
    for t in range(T - 1, -1, -1):
        if t == T - 1:
            Sb[t], Nb[t] = own[t].copy(), ownN[t].copy()
        else:
            A = links[t][0]               # (n_t+1, n_{t+1}+1)
            Sb[t] = own[t] + lam * (A @ Sb[t + 1]); Nb[t] = ownN[t] + lam * (A @ Nb[t + 1])
    out, sup = [], []
    for t in range(T):
        S = Sf[t] + Sb[t] - own[t]
        N = Nf[t] + Nb[t] - ownN[t]
        out.append((S / np.maximum(N, 1e-9)[:, None])[1:])
        sup.append(N[1:])
    return out, sup


def frozen_weight(labs, ns, flags):
    """Own-evidence weight per object: share of its pixels outside the frozen halves (lib/stale.py flags:
    bit 1 rows 0-379, bit 2 rows 380-764)."""
    w = []
    for t, (lab, n) in enumerate(zip(labs, ns)):
        if n == 0:
            w.append(np.zeros(0)); continue
        f = int(flags[t]) if flags is not None else 0
        if f == 0:
            w.append(np.ones(n)); continue
        keep = np.ones(lab.shape[0], np.float32)
        if f & 1:
            keep[:380] = 0
        if f & 2:
            keep[380:] = 0
        pix = np.flatnonzero(lab); ii = lab.ravel()[pix]
        rows = pix // lab.shape[1]
        sz = np.bincount(ii, minlength=n + 1)
        w.append((np.bincount(ii, weights=keep[rows], minlength=n + 1) / np.maximum(sz, 1))[1:])
    return w


# decision defaults of the research tracks layer (proto/tracks/cleaner.py)
DEFAULTS = dict(lam=0.8, margin=0.0, hyst=1.0, hown=1.0, bigm=1.0, bigs=1000, hshare=0.5)
DS = 4


def flow_at(flr):
    """callable (t, y, x) -> (dx, dy) motion per 5 min t -> t+1 at pixels, nearest 4 km cell of the cached
    flow (no full-grid upsampling; links only need ~1 px accuracy)."""
    def f(t, y, x):
        fl, s = flr[t]
        ny, nx = fl.shape[:2]
        yi = np.minimum(y // DS, ny - 1); xi = np.minimum(x // DS, nx - 1)
        v = fl[yi, xi] * (DS * s)
        return v[:, 0], v[:, 1]
    return f
