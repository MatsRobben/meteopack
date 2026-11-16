"""RTCOR clutter cleaner, version 14 (old processing chain only).

The KNMI RTCOR archive changed processing chain on 2023-01-31 10:45 UTC. Frames of the old chain hold much
more non-meteorological echo: interference spikes, sea clutter and anomalous propagation, wind farms and other
fixed clutter, speckle, dash fields, flashes and one-frame bursts. This cleaner removes that echo from old-chain
frames so that they can serve as clean, temporally consistent training targets. New-chain frames pass unchanged.

Rules. Only RTCOR itself is used (no gauges, no other radars). The radar mask never changes: codes >= 65534
(missing, outside) pass unchanged. A value is never raised and echo is never added: every output pixel is
<= the raw pixel. Frame t is decided with the help of up to PAD = 7 frames on each side. Missing data and
frozen copies are never counted as evidence either way. Losing some rain that looks like clutter is accepted;
holes, slits, bites, half showers and rain that blinks on and off are not.

Aggressiveness. One number per pixel, aggr in [1, 2], set automatically: a small model scores how much of the
echo in each 32-km tile looks like clutter (granular texture for its density, more so over sea), smoothed over
+-45 min (score.py). Clean tiles get aggr 1, tiles with a clutter load of 0.20-0.45 ramp up to 2; the map may
change by at most 0.1 per 5 minutes, so decisions do not jump in time. Every stage reads its strength from it.

How it works, in plain words (stages run in this order on one window of frames):
 1. Objects. Echo connected after closing 1-px gaps forms objects. A classifier (two gradient-boosted tree
    models, ~50 features: intensity, texture, shape, geography, how the object overlaps its neighbours in
    time when moved along the motion, ray patterns from the radars) scores each object as clutter or rain.
    The scores are pooled along object tracks over about +-30 min, so one frame cannot flip a decision.
    Small tracked objects get a bias toward removal; large objects are never removed whole.
 2. Spikes (first pass). Thin radial rays from the Dutch radars are found per radar in polar coordinates. In
    clear air a ray is removed; over rain, its extra power is subtracted down to the rain beside it. Seeds
    that move with the rain are dropped first (a far-range rain band radial to a radar is not a ray).
 3. Frozen halves. In the old chain one half of a frame is often an exact copy of the previous frame. Copied
    halves never count as evidence and get the previous frame's cleaned output.
 4. Hot spots. Pixels that show isolated echo far more often than their surroundings (a climatological map
    per processing chain) are fixed clutter; echo on them is removed unless it is rain moving through.
 5. Weak transient echo. Small objects with no echo along the motion in the frames before and after.
 6. Temporal support. Echo with no support along the motion (or in place) in t-3..t+3 on both sides:
    bursts, one-frame dash fields, flicker.
 7. Radial dashes. Thin lines radial to a Dutch radar that do not move with the flow.
 8. Outliers and lines (aggr > 1). Single hot pixels and thin bright lines in weak surroundings that are
    absent before and after. (The research speckle-texture stage, "texture", is still available but not in
    the default list BASE_STAGES: it decided per pixel and could cut half a shower; stage 13 covers speckle.)
 9. Vote. Removals of 6-8 are made consistent along the motion over t-1, t, t+1.
10. Realism guard. An edit of 6-8 that carves inside coherent rain (makes a hole, or bites into kept echo
    along most of its boundary) is undone, and lowered pixels with strong neighbours are restored.
Post stages (after the guard, so it cannot undo them; they run at every aggr):
11. Fixed clutter (fixclut). Earth-fixed echo stays in place while the rain around it moves with the flow:
    echo that persists at the same place over 15-30 min (more than it persists along the motion), or whose
    footprint and internal pattern stay fixed for 30-45 min in clear air, is removed per object for its whole
    lifetime; a clutter lobe attached to rain is cut where it joins.
12. Spikes (second pass, spikes2). Interference rays and strobes in each Dutch radar's polar grid: straight
    top-hat lines at constant azimuth that flash or stay fixed in polar coordinates while rain moves; tracked
    through the window. Under a ray the rain is rebuilt from the rain beside it (no slit, no lowered band):
    the sides are read from the stage input, missing data or the image edge beside a ray counts as unknown
    (never as dry), clear air needs dry evidence on both sides, and where a ray crosses the flank of a rain
    area the strip is held at the level of the wetter side (no dark stripe through the rain edge).
13. Consistency (stc). Rain moves with the flow and changes smoothly over 5-10 minutes; noise does not. Each
    pixel is compared with the neighbouring frames (t+-1, t+-2) along the motion: echo with no similar echo
    nearby there is clamped to the supported level, or removed when nothing supports it (small flashes,
    granular speckle, dash fields, intense noise spots in and around rain). Weak drizzle is never removed and
    whole objects are treated alike (no half showers, no pin holes). Where the clutter load is high, thin
    1-2 px granules and dashes that cannot form a coherent rain body are also removed. A neighbour region
    where a radar contributed only every second or third cycle (a dropout) is no evidence at all, and where
    the clutter load is low, pixels inside a smooth coherent rain core are never lowered or removed.
Then the frozen-half copy again, and the final stage on the finished output:
14. Residue. Faint leftovers of removed clutter, only where the clutter load is high (aggr >= 1.8): small
    objects that are mostly lowered or at the lowest class without a coherent core, small fragments sitting
    in a neighbourhood where much more echo was removed than kept, and small specks with no echo along the
    motion in the frames before and after. Decisions are made consistent along the motion (removed in t-1 and
    t+1 -> removed at t; kept around -> kept). Large coherent rain is never touched. More stages can be added with window.register_stage in the slots `stages`
(before the vote and guard), `post` (after the guard) or `final` (on the finished output).

Entry points:
    clean(codes, times, assets=None, *, context=1)  -> (cleaned, info)   the dataset builder's contract
    clean_window(codes, times, aggr=None, ...)       -> cleaned          one window, research-equivalent
`clean` cleans frames context..T-context in blocks of BLOCK frames, each with up to PAD neighbouring frames on
both sides. Frames outside are context only and are returned unchanged.

Port of the final research cleaner proto/v14 (clean_window(codes, times) with its defaults) of the clutter
study research/rtcor_clutter_v8, equal bit for bit (scripts/rtcor_clean_v14_equivalence.py); research md5s in
RESEARCH_MD5. The fixed inputs (geography, interference azimuths, coverage, hot-spot maps, the clutter-score
model, the tree models as plain arrays) are in assets/rtcor_clean_v14.npz, built by
scripts/rtcor_clean_v14_assets.py. Module map: window (pipeline, decision, automatic aggr, stage registry),
objects/feats (1), tracks (pooling), v10/spikes/streaks/streaks_site/wedge (2-3), hotspots (4), weak (5),
stages/texfeat/quality (6-10), ext_stages (adapters of 11-14), fixclut (11), spikes2* (12), stc (13),
residue (14),
score (aggr), stale (frozen halves), assets (inputs and tree evaluation).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from . import assets as _assets
from . import score
from .assets import ASSETS, Assets, load_assets
from .window import (
    AUTO_DEFAULTS,
    BASE_STAGES,
    FINAL_STAGES,
    NEW_CHAIN,
    POST_STAGES,
    SEED_MC_DEFAULTS,
    auto_aggr,
    clean_window,
    register_stage,
)

__all__ = [
    'ASSETS',
    'BASE_STAGES',
    'BLOCK',
    'FINAL_STAGES',
    'MISSING',
    'OUTSIDE',
    'PAD',
    'POST_STAGES',
    'STATS',
    'VERSION',
    'Assets',
    'auto_aggr',
    'clean',
    'clean_window',
    'fingerprint',
    'load_assets',
    'register_stage',
    'score',
]

VERSION = 'v14'
MISSING, OUTSIDE = 65534, 65535
PAD = 7                         # neighbouring frames used on each side of a block
BLOCK = 24                      # frames cleaned per call of clean_window
AGGR = None                     # automatic: window.auto_aggr from the v13 clutter load of 32-km tiles
# research sources of this port (clutter study research/rtcor_clutter_v8/proto), md5 of the ported file
RESEARCH_MD5 = {
    'v14/cleaner.py': '7f7d852cd05f9a7b62c00dce13a6d7c4', 'v14/stages.py': '89c5730930de76778bd0cf04ec955345',
    'v14/texfeat.py': '389f3f306f3d6604c12895fa354e31b8', 'v13/score.py': 'e788eb185e08ea1c1058ad1bd0dd0bdc', 'v13/model.json': '4490a393b16d16ef9833d9087f8e6cba',
    'fixclut/fixclut.py': 'add03a0cc84dc68e78a91163c2be6135', 'fixclut/stage.py': '6c6272e036690ac6534a0f67dbd919a5',
    'spikes2/spikes2.py': 'a2a83f20a9692e6f1725f06446c4fed7', 'spikes2/polar.py': '88667960f454bde1f2d8c59c6a4d0278',
    'spikes2/features.py': '9641fcbf8282c7159b0e07e054318d59', 'spikes2/s2common.py': '73ff28a657177b98307aec82818372ea',
    'spikes2/stage.py': 'c2f0860fa2bbcd732baa2dfee44ac1fc', 'stc/stc.py': '299746e2d4d1faa7040e489d1b0043f0',
    'stc/stage.py': '4a557bf33a7cf08552e126acdd5b98b9',
    'residue/residue.py': 'fbc9010583549fac50fa254f606ad2c3', 'residue/stage.py': '33642fce33436f316fb042f7e934b787',
}
STATS = ('changed_pixels', 'raw_wet_pixels', 'clean_wet_pixels', 'raw_sum_code', 'clean_sum_code')


def fingerprint() -> dict:
    """Version and hashes of the code and fixed inputs, for build manifests."""
    h = hashlib.sha256()
    for f in sorted(Path(__file__).parent.glob('*.py')):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return {'version': VERSION, 'code_sha256': h.hexdigest(), 'assets_sha256': _assets.sha256(),
            'aggr': 'auto' if AGGR is None else AGGR, 'auto': dict(AUTO_DEFAULTS), 'seed_mc': dict(SEED_MC_DEFAULTS),
            'stages': list(BASE_STAGES), 'post': list(POST_STAGES), 'final': list(FINAL_STAGES),
            'pad': PAD, 'block': BLOCK, 'research_md5': dict(RESEARCH_MD5)}


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
