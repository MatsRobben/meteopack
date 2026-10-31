"""train/val/test index by time blocks, written next to the store as parquet.

Never a random per-sample split. Consecutive radar frames are nearly the same
picture, so a random split puts a frame in train and the frame five minutes
later in val, and the validation score measures interpolation rather than
forecasting. The error is invisible — the curves look excellent — which is what
makes it worth a module.

So: time is cut into contiguous blocks, a whole block goes to one split, and a
sample belongs to the block that contains it. Two consequences are deliberate.

*Blocks cycle rather than being three slabs.* The default pattern deals ten
consecutive blocks as eight train, one val, one test, so validation covers every
season and every regime instead of whatever the last year of the archive
happened to be. A hold-out year is still expressible — that is what the explicit
intervals in ``cfg.splits`` are for.

*A sample that straddles a block boundary is dropped*, not assigned. Those are
the only samples that could carry information across a split boundary, there are
few of them (a window is minutes, a block is days), and dropping them is the
cheap version of a purge/embargo.

Everything here is a pure function of the spec and the arguments, so the index
is reproducible: same config, same window, same file.
"""
from __future__ import annotations

import itertools
import logging
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

from meteopack.core.config import Layer, layer_path
from meteopack.derive.base import DeriveSpec, sample_table

SPLITS_NAME = "splits.parquet"

#: Ten-block cycle: 80/10/10, with val and test spread through the archive.
DEFAULT_CYCLE: tuple[str, ...] = ("train",) * 8 + ("val", "test")
DEFAULT_BLOCK = timedelta(days=14)

#: Samples that fall in no configured interval, or straddle a boundary between
#: two splits. Kept in the file (the file documents the whole enumeration) but
#: returned by no split.
UNASSIGNED = "unassigned"
EMBARGO = "embargo"


@dataclass(frozen=True, slots=True)
class Block:
    """A contiguous stretch of the target timeline belonging to one split."""

    index: int
    split: str
    start: datetime
    end: datetime

    def __contains__(self, when: datetime) -> bool:
        return self.start <= when < self.end


def _as_utc(value: datetime | str) -> datetime:
    when = datetime.fromisoformat(value) if isinstance(value, str) else value
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)


def parse_intervals(splits: dict[str, list[str]]) -> dict[str, list[tuple[datetime, datetime]]]:
    """``{"train": ["2012-01-01/2020-01-01", ...]}`` -> half-open UTC intervals."""
    out: dict[str, list[tuple[datetime, datetime]]] = {}
    for name, entries in splits.items():
        parsed = []
        for entry in entries:
            if "/" not in entry:
                raise ValueError(f"split {name!r}: {entry!r} is not an ISO interval 'start/end'")
            start, end = (_as_utc(part) for part in entry.split("/", 1))
            if end <= start:
                raise ValueError(f"split {name!r}: empty interval {entry!r}")
            parsed.append((start, end))
        out[name] = sorted(parsed)
    return out


def _check_disjoint(intervals: dict[str, list[tuple[datetime, datetime]]]) -> None:
    flat = [(s, e, name) for name, spans in intervals.items() for s, e in spans]
    flat.sort()
    for (s0, e0, n0), (s1, _, n1) in itertools.pairwise(flat):
        if s1 < e0 and n0 != n1:
            raise ValueError(f"splits {n0!r} and {n1!r} overlap at {max(s0, s1).isoformat()}")


logger = logging.getLogger(__name__)

def blocks(
    spec: DeriveSpec,
    *,
    block: timedelta = DEFAULT_BLOCK,
    cycle: Sequence[str] = DEFAULT_CYCLE,
) -> list[Block]:
    """Contiguous blocks covering the target timeline, each tagged with a split.

    Explicit intervals in ``cfg.splits`` win when present; otherwise the
    timeline is cut into ``block``-long pieces dealt out along ``cycle``.
    """
    start, end = spec.timeline.start, spec.timeline.end
    if spec.cfg.splits:
        intervals = parse_intervals(spec.cfg.splits)
        _check_disjoint(intervals)
        spans = sorted(
            (max(s, start), min(e, end), name)
            for name, entries in intervals.items()
            for s, e in entries
            if max(s, start) < min(e, end)
        )
        return [Block(i, name, s, e) for i, (s, e, name) in enumerate(spans)]

    if block <= timedelta(0):
        raise ValueError("block length must be positive")
    if not cycle:
        raise ValueError("cycle must name at least one split")
    out: list[Block] = []
    edge = start
    while edge < end:
        stop = min(edge + block, end)
        out.append(Block(len(out), cycle[len(out) % len(cycle)], edge, stop))
        edge = stop
    return out


def _warn_if_a_split_is_empty(
    spec: DeriveSpec, layout: Sequence[Block], block: timedelta, cycle: Sequence[str]
) -> None:
    """Say so when a split ends up with no blocks at all.

    Training on that silently gives no validation data, which is the kind of
    mistake you find out about at the end of a run rather than the start.

    The warning is computed from the layout that was actually built, not from
    the cycle arithmetic. When `cfg.splits` names the intervals, the cycle is
    not used at all, and reasoning about it produced a warning that named the
    wrong splits and advised setting `cfg.splits` - which was already set. A
    warning that is wrong about the thing it is warning about is worse than
    silence, because it teaches you to skip the one that is right.
    """
    present = {b.split for b in layout}
    if spec.cfg.splits:
        missing = sorted(set(spec.cfg.splits) - present)
        if missing:
            window = f"{spec.timeline.start:%Y-%m-%dT%H:%M}..{spec.timeline.end:%Y-%m-%dT%H:%M}"
            logger.warning(
                "%s: cfg.splits names %s, but no interval for %s overlaps the window %s, "
                "so %s will be EMPTY. Check the dates in cfg.splits against the input store.",
                spec.store_id, ", ".join(sorted(spec.cfg.splits)),
                ", ".join(missing), window, ", ".join(missing),
            )
        return

    missing = sorted(set(cycle) - present)
    if missing:
        span = spec.timeline.end - spec.timeline.start
        logger.warning(
            "%s: the window is %s but the %s block cycle needs %s to reach every split; "
            "%s will be EMPTY. Widen the window, shorten `block`, or set cfg.splits.",
            spec.store_id, span, block, len(cycle) * block, ", ".join(missing),
        )


def assign(
    spec: DeriveSpec,
    *,
    block: timedelta = DEFAULT_BLOCK,
    cycle: Sequence[str] = DEFAULT_CYCLE,
) -> pl.DataFrame:
    """The split index: one row per sample, in write-slot order."""
    table = sample_table(spec)
    layout = blocks(spec, block=block, cycle=cycle)
    _warn_if_a_split_is_empty(spec, layout, block, cycle)
    if not layout:
        raise ValueError(f"{spec.store_id}: no split block overlaps {spec.timeline.start}..")
    edges = np.array(
        [np.datetime64(b.start.replace(tzinfo=None), "ns") for b in layout], dtype="datetime64[ns]"
    )
    names = np.array([b.split for b in layout], dtype=object)
    stops = np.array(
        [np.datetime64(b.end.replace(tzinfo=None), "ns") for b in layout], dtype="datetime64[ns]"
    )

    step = np.timedelta64(spec.timeline.step_ns, "ns")
    first = table["start_time"]
    last = table["end_time"] - step        # inclusive end: the final frame

    def locate(when: np.ndarray) -> np.ndarray:
        at = np.searchsorted(edges, when, side="right") - 1
        inside = (at >= 0) & (when < stops[np.clip(at, 0, len(layout) - 1)])
        return np.where(inside, at, -1)

    at_first, at_last = locate(first), locate(last)
    split = np.full(len(first), UNASSIGNED, dtype=object)
    found = (at_first >= 0) & (at_last >= 0)
    split[found] = names[at_first[found]]
    # Straddling a boundary is only a problem when the two blocks disagree.
    straddle = found & (names[np.clip(at_first, 0, None)] != names[np.clip(at_last, 0, None)])
    split[straddle] = EMBARGO
    split[(at_first >= 0) & (at_last < 0)] = EMBARGO

    return pl.DataFrame(
        {
            "slot": table["slot"],
            "sample_index": table["sample_index"],
            "t0": table["t0"],
            "y0": table["y0"],
            "x0": table["x0"],
            "start_time": table["start_time"].astype("datetime64[us]"),
            "end_time": table["end_time"].astype("datetime64[us]"),
            "block": np.where(at_first >= 0, at_first, -1).astype("int64"),
            "split": pl.Series([str(s) for s in split], dtype=pl.String),
        }
    )


def store_path(spec: DeriveSpec, root: str | os.PathLike[str], *,
               layer: Layer = "derived") -> Path:
    return layer_path(Path(root), layer, *spec.store_id.split("/"))


def path_for(spec: DeriveSpec, root: str | os.PathLike[str], *,
             layer: Layer = "derived") -> Path:
    return store_path(spec, root, layer=layer) / SPLITS_NAME


def write(
    spec: DeriveSpec,
    root: str | os.PathLike[str],
    *,
    block: timedelta = DEFAULT_BLOCK,
    cycle: Sequence[str] = DEFAULT_CYCLE,
    layer: Layer = "derived",
) -> Path:
    """Write the index beside the store. Overwrites — it is a pure function."""
    frame = assign(spec, block=block, cycle=cycle)
    target = path_for(spec, root, layer=layer)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
    frame.write_parquet(tmp, statistics=True)
    os.replace(tmp, target)
    return target


def read(spec: DeriveSpec, root: str | os.PathLike[str], *,
         layer: Layer = "derived") -> pl.DataFrame:
    target = path_for(spec, root, layer=layer)
    if not target.exists():
        raise FileNotFoundError(f"no split index at {target}")
    return pl.read_parquet(target)


def slots(frame: pl.DataFrame, split: str) -> list[int]:
    """Write slots belonging to ``split`` — what a training loader iterates."""
    return frame.filter(pl.col("split") == split)["slot"].sort().to_list()


def summary(frame: pl.DataFrame) -> dict[str, int]:
    counts = frame.group_by("split").len().sort("split")
    return dict(zip(counts["split"].to_list(), counts["len"].to_list(), strict=True))


__all__ = [
    "DEFAULT_BLOCK",
    "DEFAULT_CYCLE",
    "EMBARGO",
    "SPLITS_NAME",
    "UNASSIGNED",
    "Block",
    "assign",
    "blocks",
    "parse_intervals",
    "path_for",
    "read",
    "slots",
    "store_path",
    "summary",
    "write",
]
