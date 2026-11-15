"""RTCOR clutter cleaner, version 14 (old processing chain only).

The KNMI RTCOR archive changed processing chain on 2023-01-31 10:45 UTC. Frames of the old chain hold much
more non-meteorological echo: interference spikes, sea clutter and anomalous propagation, wind farms and other
fixed clutter, speckle and one-frame bursts. This cleaner removes that echo from old-chain frames so that they
can serve as clean, temporally consistent training targets. New-chain frames pass unchanged.

Rules. Only RTCOR itself is used (no gauges, no other radars). The radar mask never changes: codes >= 65534
(missing, outside) pass unchanged. A value is never raised and echo is never added: every output pixel is
<= the raw pixel. Frame t is decided with the help of up to PAD = 7 frames on each side.

How it works, in plain words (stages run in this order on one window of frames):
 1. Objects. Echo connected after closing 1-px gaps forms objects. A classifier (two gradient-boosted tree
    models, ~50 features: intensity, texture, shape, geography, how the object overlaps its neighbours in
    time when moved along the motion, ray patterns from the radars) scores each object as clutter or rain.
    The scores are pooled along object tracks over about +-30 min, so one frame cannot flip a decision.
    Small tracked objects get a bias toward removal (aggressiveness aggr); large objects are never removed
    whole, they are left to the pixel stages.
 2. Spikes. Thin radial rays from the Dutch radars (interference) are found per radar in polar coordinates.
    In clear air the ray is removed; over rain, the ray's extra power is subtracted down to the rain beside it.
 3. Frozen halves. In the old chain one half of a frame is often an exact copy of the previous frame. Copied
    halves never count as evidence and get the previous frame's cleaned output.
 4. Hot spots. Pixels that show isolated echo far more often than their surroundings (a climatological map
    per processing chain) are fixed clutter. Echo sitting on such a spot is removed unless it is rain moving
    through (found along the motion before and after).
 5. Weak transient echo. Small objects with no echo along the motion in the frames before and after.
 6. Temporal support. Echo with no support along the motion (or in place) in t-3..t+3 on both sides:
    bursts, one-frame dash fields, flicker. Missing data in a neighbour frame is no evidence either way.
 7. Radial dashes. Thin lines radial to a Dutch radar that do not move with the flow.
 8. Speckle texture. Fragmented, granular echo inside larger objects, judged along the motion.
 9. Outliers and lines (aggr > 1). Single hot pixels and thin bright lines in weak surroundings that are
    absent before and after.
10. Vote. Removals are made consistent along the motion over t-1, t, t+1.
11. Realism guard. An edit of stages 6-9 that carves inside coherent rain (makes a hole, or bites into kept
    echo along most of its boundary) is undone, and lowered pixels with strong neighbours are restored.
    Losing some rain that looks like clutter is accepted; holes, slits and half showers are not.
12. Added stages (window.register_stage): in `stages` (before the vote and guard), in `post` (after the
    guard, before the frozen-half copy) or in `final` (on the finished output).

Entry points:
    clean(codes, times, assets=None, *, context=1)  -> (cleaned, info)   the dataset builder's contract
    clean_window(codes, times, aggr=1.0, ...)        -> cleaned          one window, research-equivalent
`clean` cleans frames context..T-context in blocks of BLOCK frames, each with up to PAD neighbouring frames on
both sides (memory stays ~1.4 GB per process). Frames outside are context only and are returned unchanged.

Port of the research cleaner proto/v12 (frozen; md5 in research proto/v12/final/md5.txt) of the clutter study
research/rtcor_clutter_v8, called as clean_window(codes, times, aggr=1, period="chain"); equal bit for bit
(scripts/rtcor_clean_v14_equivalence.py). The fixed inputs (geography, interference azimuths, coverage,
hot-spot maps, the tree models as plain arrays) are in assets/rtcor_clean_v14.npz, built by
scripts/rtcor_clean_v14_assets.py. Module map: objects/feats (stage 1 evidence), tracks (pooling),
window (decision and pipeline), v10/spikes/streaks/streaks_site/wedge (2-3), hotspots (4), weak (5),
stages/texfeat/quality (6-11), stale (frozen halves), assets (inputs and tree evaluation).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import assets as _assets
from . import score
from .assets import ASSETS, Assets, load_assets
from .window import DEFAULT_STAGES, NEW_CHAIN, clean_window, register_stage

__all__ = [
    'ASSETS',
    'BLOCK',
    'DEFAULT_STAGES',
    'MISSING',
    'OUTSIDE',
    'PAD',
    'STATS',
    'VERSION',
    'Assets',
    'clean',
    'clean_window',
    'fingerprint',
    'load_assets',
    'register_stage',
    'score',
]

VERSION = 'v14-base'            # v12 base; becomes 'v14' when the final stages land
MISSING, OUTSIDE = 65534, 65535
PAD = 7                         # neighbouring frames used on each side of a block
BLOCK = 24                      # frames cleaned per call of clean_window
AGGR = 1.0
STATS = ('changed_pixels', 'raw_wet_pixels', 'clean_wet_pixels', 'raw_sum_code', 'clean_sum_code')


def fingerprint() -> dict:
    """Version and hashes of the code and fixed inputs, for build manifests."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob('*.py')):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return {'version': VERSION, 'code_sha256': h.hexdigest(), 'assets_sha256': _assets.sha256(),
            'aggr': AGGR, 'stages': list(DEFAULT_STAGES), 'pad': PAD, 'block': BLOCK}


def clean(codes, times, assets: Assets | None = None, *, context: int | tuple[int, int] = 1,
          aggr=AGGR, block: int = BLOCK, pad: int = PAD, **kw):
    """Clean frames context..T-context of consecutive 5-minute RTCOR frames (uint16 codes, time, y, x).

    `times` are the frame timestamps (UTC). context = n or (lead, trail): the first lead and last trail frames
    are context only and are returned unchanged. The frames in between are cleaned in blocks of `block`, each
    with up to `pad` neighbouring frames on both sides taken from the whole array. Old-chain frames are cleaned,
    new-chain frames pass unchanged. kw go to clean_window. Returns (cleaned codes, per-frame statistics on
    valid pixels: cleaned, old_chain and STATS)."""
    a = _assets.use(assets)
    codes = np.asarray(codes)
    times = np.asarray(times).astype('datetime64[ns]')
    T = codes.shape[0] if codes.ndim == 3 else 0
    if codes.ndim != 3 or codes.shape[1:] != a.shape:
        raise ValueError(f'frames must have shape (time, {a.shape[0]}, {a.shape[1]})')
    if codes.dtype != np.uint16:
        raise ValueError('codes must be uint16 RTCOR codes')
    lead, trail = (context, context) if np.isscalar(context) else context
    if len(times) != T or lead < 0 or trail < 0 or lead + trail >= T or block < 1 or pad < 0:
        raise ValueError('need one timestamp per frame and at least one frame to clean')
    out = codes.copy()
    old = times < NEW_CHAIN
    c0, c1 = lead, T - trail
    for b0 in range(c0, c1, block):
        b1 = min(b0 + block, c1)
        if not old[b0:b1].any():
            continue
        w0, w1 = max(b0 - pad, 0), min(b1 + pad, T)
        res = clean_window(codes[w0:w1], times[w0:w1], aggr, **kw)
        out[b0:b1] = res[b0 - w0:b1 - w0]
        del res
    valid = codes < MISSING
    info = {'cleaned': np.zeros(T, bool), 'old_chain': old}
    info['cleaned'][c0:c1] = True
    rw, cw = (codes > 0) & valid, (out > 0) & valid
    info['changed_pixels'] = (out != codes).sum((1, 2)).astype(np.int64)
    info['raw_wet_pixels'] = rw.sum((1, 2)).astype(np.int64)
    info['clean_wet_pixels'] = cw.sum((1, 2)).astype(np.int64)
    info['raw_sum_code'] = np.where(valid, codes, 0).sum((1, 2), dtype=np.int64)
    info['clean_sum_code'] = np.where(valid, out, 0).sum((1, 2), dtype=np.int64)
    return out, info
