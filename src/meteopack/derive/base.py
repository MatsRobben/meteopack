"""DeriveSpec: a DeriveConfig plus a window, resolved into concrete work.

A derived dataset is a cartesian product of *when* and *where*: time windows of
``n_timesteps`` frames on the target timeline, crossed with tiles on the target
grid. Everything downstream — the brick writer, the split index, the Slurm
planner — reads that enumeration from here, so there is one definition of
"sample 4711" and it cannot drift between stages.

Two decisions carry the design:

*Tiles are the only spatial mode.* ``cfg.tile is None`` means ``tile == the grid
shape``, which yields exactly one tile per window; ``tile=(256, 256)`` yields
patches. Full-field is not a special case in any consumer — it is the degenerate
tiling. Anchors step by the tile size, and the last anchor is pinned to the far
edge when the tile does not divide the extent, so the field is fully covered at
the cost of an overlap in the final row/column rather than a dropped strip.

*Write order is shuffled once, deterministically.* Training reads bricks
sequentially, so the shuffle has to be baked into the layout at build time. Slot
``j`` of the output holds sample ``order[j]``, where ``order`` is a permutation
seeded by the spec fingerprint — the config hash plus the window. The same
config and window give the same layout on any node, so disjoint shards written
by separate processes compose into exactly the single-process result.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timedelta

import numpy as np

from meteopack.core import grids
from meteopack.core.config import DeriveConfig
from meteopack.core.store.manifest import config_hash
from meteopack.core.timeline import Timeline


@dataclass(frozen=True, slots=True)
class Sample:
    """One training sample: a time window crossed with a tile.

    ``index`` is the position in the canonical (unshuffled) enumeration and is
    the sample's identity everywhere — in the split index, in the brick's
    ``sample_index`` array, in job logs. The write slot is a separate number.
    """

    index: int
    t0: int
    y0: int
    x0: int


@dataclass(frozen=True, slots=True)
class Shard:
    """A contiguous run of write slots: the unit of parallelism and of resume.

    Shards partition the *output*, not the input, so two tasks never touch the
    same bytes and re-running one shard rewrites exactly the region it owns.
    """

    index: int
    start: int
    stop: int
    samples: tuple[Sample, ...]

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def slots(self) -> range:
        return range(self.start, self.stop)


def tile_anchors(extent: int, size: int) -> tuple[int, ...]:
    """Start offsets covering ``extent`` with tiles of ``size``.

    Steps by ``size``; when the tile does not divide the extent the last anchor
    is pinned to ``extent - size`` so the edge is covered. ``size == extent``
    (the full-field case) gives ``(0,)``, with no overlap.
    """
    if size <= 0:
        raise ValueError(f"tile size must be positive, got {size}")
    if size > extent:
        raise ValueError(f"tile size {size} exceeds grid extent {extent}")
    out = list(range(0, extent - size + 1, size))
    if out[-1] + size < extent:
        out.append(extent - size)
    return tuple(out)


class DeriveSpec:
    """Resolved work for one derived dataset over one window."""

    __slots__ = (
        "_inverse",
        "_order",
        "_tiles",
        "_time_anchors",
        "cfg",
        "grid",
        "tile",
        "timeline",
    )

    def __init__(
        self,
        cfg: DeriveConfig,
        timeline: Timeline,
        *,
        grid: grids.Grid | None = None,
        grid_registry: str | None = None,
    ) -> None:
        if timeline.cadence != timedelta(seconds=cfg.cadence_seconds):
            raise ValueError(
                f"timeline cadence {timeline.cadence} does not match "
                f"{cfg.name} cadence of {cfg.cadence_seconds}s"
            )
        self.cfg = cfg
        self.timeline = timeline
        self.grid = grid if grid is not None else grids.get(cfg.grid_id, grid_registry)
        ny, nx = self.grid.shape
        self.tile: tuple[int, int] = (
            (int(cfg.tile[0]), int(cfg.tile[1])) if cfg.tile else (int(ny), int(nx))
        )
        n_steps = len(timeline)
        if n_steps < cfg.n_timesteps:
            raise ValueError(
                f"window holds {n_steps} steps, fewer than n_timesteps={cfg.n_timesteps}"
            )
        # Non-overlapping time windows: overlapping ones put near-identical
        # samples in different splits, which is the leak splits.py exists to stop.
        self._time_anchors = tuple(range(0, n_steps - cfg.n_timesteps + 1, cfg.n_timesteps))
        self._tiles = tuple(
            (y0, x0)
            for y0 in tile_anchors(ny, self.tile[0])
            for x0 in tile_anchors(nx, self.tile[1])
        )
        self._order: np.ndarray | None = None
        self._inverse: np.ndarray | None = None

    @classmethod
    def from_config(cls, cfg: DeriveConfig, start: datetime, end: datetime, **kw) -> DeriveSpec:
        """Build the target timeline from the config's cadence and a window."""
        return cls(cfg, Timeline(start, end, timedelta(seconds=cfg.cadence_seconds)), **kw)

    # -- identity ---------------------------------------------------------

    @property
    def store_id(self) -> str:
        return f"{self.cfg.name}/v{self.cfg.version}"

    @property
    def category(self) -> str:
        return self.cfg.category

    @property
    def input_store_ids(self) -> tuple[str, ...]:
        return tuple(self.cfg.inputs)

    @property
    def fingerprint(self) -> str:
        """Config hash + window. Seeds the shuffle and goes in the manifest."""
        blob = "|".join(
            (
                config_hash(self.cfg),
                self.timeline.start.isoformat(),
                self.timeline.end.isoformat(),
                str(self.timeline.step_ns),
            )
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    # -- geometry ---------------------------------------------------------

    @property
    def time_anchors(self) -> tuple[int, ...]:
        return self._time_anchors

    @property
    def tiles(self) -> tuple[tuple[int, int], ...]:
        return self._tiles

    @property
    def sample_shape(self) -> tuple[int, int, int]:
        """``(T, tile_y, tile_x)`` — one sample of one variable."""
        return (self.cfg.n_timesteps, self.tile[0], self.tile[1])

    @property
    def n_samples(self) -> int:
        return len(self._time_anchors) * len(self._tiles)

    def sample(self, index: int) -> Sample:
        """The ``index``-th sample in canonical (time-major) order."""
        if not 0 <= index < self.n_samples:
            raise IndexError(f"sample {index} out of range (n_samples={self.n_samples})")
        t_i, tile_i = divmod(index, len(self._tiles))
        y0, x0 = self._tiles[tile_i]
        return Sample(index=index, t0=self._time_anchors[t_i], y0=y0, x0=x0)

    def samples(self) -> Iterator[Sample]:
        for i in range(self.n_samples):
            yield self.sample(i)

    def window(self, sample: Sample) -> tuple[datetime, datetime]:
        """Half-open ``[start, end)`` of a sample's time window."""
        step = self.timeline.cadence
        start = self.timeline.start + sample.t0 * step
        return start, start + self.cfg.n_timesteps * step

    def times(self, sample: Sample) -> np.ndarray:
        """datetime64[ns] timestamps of a sample's frames."""
        return self.timeline.values()[sample.t0 : sample.t0 + self.cfg.n_timesteps]

    def slices(self, sample: Sample) -> tuple[slice, slice, slice]:
        """``(time, y, x)`` slices of a sample into the full target field."""
        return (
            slice(sample.t0, sample.t0 + self.cfg.n_timesteps),
            slice(sample.y0, sample.y0 + self.tile[0]),
            slice(sample.x0, sample.x0 + self.tile[1]),
        )

    # -- write order and shards -------------------------------------------

    @property
    def order(self) -> np.ndarray:
        """Permutation mapping write slot -> canonical sample index."""
        if self._order is None:
            self._order = np.random.default_rng(int(self.fingerprint, 16)).permutation(
                self.n_samples
            )
        return self._order

    def slot_of(self, index: int) -> int:
        """Inverse of :attr:`order`: the slot sample ``index`` is written to."""
        if not 0 <= index < self.n_samples:
            raise IndexError(f"sample {index} is not in this spec")
        if self._inverse is None:
            self._inverse = np.argsort(self.order, kind="stable")
        return int(self._inverse[index])

    def slot_sample(self, slot: int) -> Sample:
        return self.sample(int(self.order[slot]))

    @property
    def shard_samples(self) -> int:
        return self.cfg.shard_samples

    @property
    def n_shards(self) -> int:
        n, s = self.n_samples, self.cfg.shard_samples
        return (n + s - 1) // s

    def shard(self, index: int) -> Shard:
        if not 0 <= index < self.n_shards:
            raise IndexError(f"shard {index} out of range (n_shards={self.n_shards})")
        start = index * self.cfg.shard_samples
        stop = min(start + self.cfg.shard_samples, self.n_samples)
        return Shard(
            index=index,
            start=start,
            stop=stop,
            samples=tuple(self.slot_sample(s) for s in range(start, stop)),
        )

    def shards(self) -> Iterator[Shard]:
        for i in range(self.n_shards):
            yield self.shard(i)

    def shard_indices(self, task: int = 0, n_tasks: int = 1) -> list[int]:
        """Shards owned by array task ``task`` of ``n_tasks``.

        Strided, not contiguous, so a ragged last shard does not leave one task
        with measurably less work than its neighbours.
        """
        if n_tasks < 1 or not 0 <= task < n_tasks:
            raise ValueError(f"bad task {task} of {n_tasks}")
        return list(range(task, self.n_shards, n_tasks))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"DeriveSpec({self.store_id!r}, samples={self.n_samples}, "
            f"shape={self.sample_shape}, shards={self.n_shards}, fp={self.fingerprint})"
        )


def sample_table(spec: DeriveSpec) -> dict[str, np.ndarray]:
    """Columnar view of the whole enumeration, in write-slot order.

    Written once into the brick at initialise time and read back by the split
    index, so neither side has to re-derive the geometry.
    """
    order = spec.order
    samples = [spec.sample(int(i)) for i in order]
    n = len(samples)
    step_ns = spec.timeline.step_ns
    start_ns = spec.timeline.start_ns
    t0 = np.fromiter((s.t0 for s in samples), dtype="int64", count=n)
    return {
        "slot": np.arange(n, dtype="int64"),
        "sample_index": order.astype("int64"),
        "t0": t0,
        "y0": np.fromiter((s.y0 for s in samples), dtype="int64", count=n),
        "x0": np.fromiter((s.x0 for s in samples), dtype="int64", count=n),
        "start_time": (start_ns + t0 * step_ns).astype("datetime64[ns]"),
        "end_time": (start_ns + (t0 + spec.cfg.n_timesteps) * step_ns).astype("datetime64[ns]"),
    }


__all__ = ["DeriveSpec", "Sample", "Shard", "sample_table", "tile_anchors"]
