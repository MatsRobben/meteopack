"""Per-pixel speckle-texture features (shared by stages.texture_stage and the analysis)."""
import cv2
import numpy as np

K3 = np.ones((3, 3), np.float32); K3[1, 1] = 0


def features(w, wp, wn, box=9):
    """w: wet mask at t; wp, wn: wet masks of t-1 / t+1 already moved onto t's grid (None: unknown).
    Returns (rho, frag, kappa): local wet density, share of loosely connected wet px (<= 3 wet 8-neighbours),
    temporal coherence (presence in t-1 or t+1 at the same moved position, relative to chance)."""
    wf = w.astype(np.float32)
    n8 = cv2.filter2D(wf, -1, K3, borderType=cv2.BORDER_CONSTANT)
    bw = cv2.boxFilter(wf, -1, (box, box), normalize=False, borderType=cv2.BORDER_CONSTANT)
    rho = bw / (box * box)
    loose = (wf * (n8 <= 3)).astype(np.float32)
    frag = cv2.boxFilter(loose, -1, (box, box), normalize=False, borderType=cv2.BORDER_CONSTANT) / np.maximum(bw, 1)
    if wp is None and wn is None:
        return rho, frag, np.full(w.shape, np.nan, np.float32)
    pres = np.zeros(w.shape, bool); dens = np.zeros(w.shape, np.float32); m = 0
    for x in (wp, wn):
        if x is None:
            continue
        pres |= x
        dens += cv2.boxFilter(x.astype(np.float32), -1, (box, box), borderType=cv2.BORDER_CONSTANT); m += 1
    dens /= m
    chance = 1 - (1 - dens) ** m
    P = cv2.boxFilter((wf * pres).astype(np.float32), -1, (box, box), normalize=False, borderType=cv2.BORDER_CONSTANT) / np.maximum(bw, 1)
    kappa = (P - chance) / np.maximum(1 - chance, 0.05)
    return rho, frag, kappa
