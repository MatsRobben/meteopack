"""The one regular-time-axis implementation.

Every store gets its time axis from here so cadence, UTC handling and the
integer-scatter trick are decided once. Writers map observation times to
positions with `index_of` and scatter, which avoids a pandas reindex per block.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

_NS = np.dtype("datetime64[ns]")


def to_utc_ns(values) -> np.ndarray:
    """Coerce anything datetime-ish to tz-naive UTC datetime64[ns].

    Gotcha this exists to absorb: pandas date_range().asi8 yields microseconds,
    not nanoseconds, so always go through .astype('datetime64[ns]').
    """
    arr = np.asarray(values)
    if arr.dtype == object or np.issubdtype(arr.dtype, np.str_):
        arr = arr.astype("datetime64[ns]")
    elif np.issubdtype(arr.dtype, np.datetime64):
        arr = arr.astype(_NS)
    else:
        raise TypeError(f"cannot interpret {arr.dtype} as datetimes")
    return arr


@dataclass(frozen=True, slots=True)
class Timeline:
    """A half-open [start, end) axis at a fixed cadence, in UTC."""

    start: datetime
    end: datetime
    cadence: timedelta

    def __post_init__(self) -> None:
        if self.cadence.total_seconds() <= 0:
            raise ValueError("cadence must be positive")
        if self.end <= self.start:
            raise ValueError(f"empty timeline: {self.start} -> {self.end}")
        for name in ("start", "end"):
            v: datetime = getattr(self, name)
            if v.tzinfo is None or v.utcoffset() != timedelta(0):
                raise ValueError(f"{name} must be timezone-aware UTC, got {v!r}")

    @property
    def step_ns(self) -> int:
        return int(self.cadence.total_seconds() * 1_000_000_000)

    @staticmethod
    def _ns(value: datetime) -> int:
        return int(np.datetime64(value.replace(tzinfo=None), "ns").astype("int64"))

    @property
    def start_ns(self) -> int:
        return self._ns(self.start)

    def __len__(self) -> int:
        return (self._ns(self.end) - self.start_ns) // self.step_ns

    def values(self) -> np.ndarray:
        return (self.start_ns + np.arange(len(self), dtype="int64") * self.step_ns).astype(_NS)

    def index_of(self, times) -> np.ndarray:
        """Positions of `times` on this axis; -1 where off-axis or off-grid.

        Off-grid means not an exact multiple of the cadence — those are dropped
        rather than silently snapped, so a cadence mismatch surfaces as missing
        data in validation instead of a shifted series.
        """
        ns = to_utc_ns(times).astype("int64")
        offset = ns - self.start_ns
        idx = offset // self.step_ns
        exact = (offset % self.step_ns) == 0
        in_range = (idx >= 0) & (idx < len(self))
        return np.where(exact & in_range, idx, -1)


def utc(year: int, month: int = 1, day: int = 1, hour: int = 0, minute: int = 0) -> datetime:
    """Short constructor for tests and configs."""
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)
