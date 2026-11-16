"""Adapters of the post stages: research proto/<name>/stage.py bodies, stage(ctx, out, **kw) -> (out, level).

ctx is stages.Context (codes = raw window, A = local aggr, flags = frozen halves, ...); out is the current output.
fixclut  fixed-clutter / anaprop cleaner on the raw window, applied to `out` (no level for the vote)
spikes2  interference rays in each NL radar's polar grid; level 0 where changed
stc      motion-compensated spatio-temporal consistency filter, strength = scale x aggr (ctx.A.a); level 0 where
         changed
residue  final slot: residual specks / flashes / dash remnants on the finished output (no ctx.s2mask in the
         default run, as in the research in-slot run); level 0 where removed
"""
from __future__ import annotations

import numpy as np

from . import fixclut as FC
from . import residue as RS
from . import spikes2 as S2
from . import stc as STC


def fixclut_stage(ctx, out, strength=1.0, **kw):
    return FC.fixclut_clean(out, ctx.codes, None, strength=strength, **kw), None


def spikes2_stage(ctx, out, strength=1.0, times=None, **kw):
    new = S2.remove_spikes(out, times, strength=strength, raw=ctx.codes, **kw)
    level = np.where(new < out, 0.0, np.inf).astype(np.float32)
    return new, level


def stc_stage(ctx, out, *, strength=None, scale=1.0, **kw):
    s = ctx.A.a * scale if strength is None else strength
    new, info = STC.stc_clean(out, ctx.codes, None, strength=s, ret_info=True, flags=ctx.flags, **kw)
    lev = info["level"]
    # v12 convention: level = aggr at which the pixel was changed (inf = kept); stc changes at any aggr > 0
    lev = np.where(np.isfinite(lev), np.float16(0.0), np.float16(np.inf)).astype(np.float16)
    return new, lev


def residue_stage(ctx, out, **kw):
    new = RS.residue_clean(out, ctx.codes, ctx.flags, ctx.st["flr"], ctx.A.a, getattr(ctx, "times", None),
                           s2mask=getattr(ctx, "s2mask", None), **kw)
    level = np.where(new < out, 0.0, np.inf).astype(np.float32)
    return new, level
