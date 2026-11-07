"""The one contract every source implements.

A source knows three things and nothing else: what exists remotely, how to
fetch one unit of it, and how to turn that unit into canonical form. Packing,
storage layout and orchestration are deliberately not its concern — those live
in `meteopack.core.store` and `meteopack.jobs`.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # heavy imports stay out of the hot path
    import polars as pl
    import xarray as xr

    from meteopack.core.config import ProductConfig


@dataclass(frozen=True, slots=True)
class TimeWindow:
    """Half-open [start, end) in UTC."""

    start: datetime
    end: datetime


@dataclass(frozen=True, slots=True)
class Unit:
    """One remotely addressable thing: a file, an API page, a station-year.

    `key` must be stable across runs — the job ledger uses it to decide what is
    already done.
    """

    key: str
    url: str
    valid_time: datetime | None = None
    meta: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class Content:
    """What a downloaded file holds, read by a source's integrity check.

    Recorded beside the checksum in the `files` table so that the gap report
    can tell a file that spans its period from one that does not, without
    re-reading the raw layer. That is the check that would have caught EA's
    station-years running 01-02..12-30 the day they were fetched instead of
    a year later. `n_records == 0` is an empty answer from upstream.
    """

    n_records: int
    time_min: datetime | None = None
    time_max: datetime | None = None


@dataclass(frozen=True, slots=True)
class RawFile:
    """A fetched unit on local disk, with the checksum recorded at fetch time.

    `content` is filled when the source checked the file (`fetch_file(check=)`)
    and None otherwise. `attempts` is how many HTTP requests this fetch took;
    0 means the file was already on disk and nothing was downloaded.
    """

    path: Path
    sha256: str
    n_bytes: int
    unit_key: str
    content: Content | None = None
    attempts: int = 0


@runtime_checkable
class Source(Protocol):
    """Implemented by every module in `meteopack.sources`."""

    name: str

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """What exists remotely in `window`. No downloads."""
        ...

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile:
        """Idempotent and atomic: write `.part`, checksum, rename. Skip if present."""
        ...

    def parse(self, raw: RawFile, cfg: ProductConfig) -> xr.Dataset | pl.DataFrame:
        """Canonical names, units and `grid_id`. Gridded -> Dataset, points -> DataFrame."""
        ...
