"""KNMI 10-minute in-situ meteorological observations (KMDS), as daily tars.

KNMI publishes this feed as **one netCDF file per ten minutes** — 144 files a
day, each a `(station, time=1)` snapshot of the whole Dutch network::

    KMDS__OPER_P___10M_OBS_L2_<YYYYMMDDHHMM>.nc

and bundles each day into one tar on the archive side::

    KMDS__OPER_P___10M_OBS_L2__<YYYYMMDD>T000000_<YYYYMMDD>T000000_0001.tar

That shape is the whole reason `RawLayout(archive="tar", granularity="day")`
exists. KNMI's dataplatform meters on **request count**, not bytes, and 14 years
of this feed is 750k members: fetched loose it is both un-downloadable and a
filesystem the metadata server cannot carry. So the unit here is a *day*, the
raw layer stores the day's tar verbatim, and `parse` reads members out of it.
Nothing in this module ever writes a loose member.

Four pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store if you get it wrong:

* **The station list grows.** Sampled one member per year off the real archive:
  62-66 stations in 2012-2020, 71 in 2026. Reading the station table from a
  single file and reusing it loses every station added later, so `stations`
  unions across a sample of days (`extra.station_scan_days`) instead.
* **The variable set drifts.** The same sample shows 100 variables through 2020
  and 102 from 2021 on, when `tg` and `Q24H` appeared. A per-file fingerprint
  would therefore reject half the archive. What is pinned is the *catalogue* —
  `extra.known_variables` — and a file carrying a subset of it is normal and
  passes; a variable outside it is drift and fails.
* **Missing is NaN, not a sentinel.** Every observation variable is float64 with
  `_FillValue = NaN`. polars does not treat a float NaN as null, so NaN is
  nulled explicitly *before* scale/offset — otherwise `NaN * 60` is still NaN
  but a configured sentinel would have been scaled into a plausible number.
* **The timestamp is the end of the interval**, per the file's own `time`
  attribute, and is read from the `time` variable (seconds since 1950-01-01)
  rather than from the member name. The member name is only how a day's members
  are ordered and selected; see `member_time`.

Station ids are the five-character WMO code the file calls `station`
("06260"), and the WIGOS id (`wsi`, "0-20000-0-06260") is carried alongside in
the station table so this store joins against WIGOS-keyed feeds without a
second lookup.

There is no HTTP here. The archive is already on local disk and a unit's `url`
is a `file://` path, so `fetch` hardlinks (or copies) the day's tar into the raw
layer rather than downloading it; see `fetch` for why a link is the default.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import re
import shutil
import tarfile
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote, urlparse

from meteopack.core import grids
from meteopack.core.config import ProductConfig, RateLimit, SourceConfig, Variable
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, TimeWindow, Unit
from meteopack.sources.common import SchemaDriftError as _SchemaDriftError

if TYPE_CHECKING:  # keep polars and xarray off the import path of `meteopack.sources`
    import polars as pl

logger = logging.getLogger(__name__)

#: Where the daily tars live: $KNMI_INSITU_ARCHIVE (the source config sets
#: `base_url` from the same variable). Empty when unset - there is no built-in
#: default, because the only sensible one was a path on one user's machine.
ARCHIVE_DIR = os.environ.get("KNMI_INSITU_ARCHIVE", "")

#: `KMDS__OPER_P___10M_OBS_L2__<from>T000000_<to>T000000_<counter>.tar`.
#: The counter is matched rather than assumed: every one of the 5211 tars in the
#: archive today reads `_0001`, and hard-coding that would break the day it does
#: not.
_TAR_RE = re.compile(
    r"^KMDS__OPER_P___10M_OBS_L2__(?P<from>\d{8})T000000_"
    r"(?P<to>\d{8})T000000_(?P<counter>\d{4})\.tar$"
)

#: `KMDS__OPER_P___10M_OBS_L2_<YYYYMMDDHHMM>.nc`, one member per ten minutes.
_MEMBER_RE = re.compile(r"^KMDS__OPER_P___10M_OBS_L2_(?P<stamp>\d{12})\.nc$")

#: Variables that describe the file rather than the weather. Excluded from the
#: fingerprint so it tracks what is *measured*, and never offered as a source_key.
STRUCTURAL: frozenset[str] = frozenset(
    {"time", "station", "wsi", "stationname", "lat", "lon", "height", "product", "projection"}
)

#: How many days to sample when building the station table. Eight spreads far
#: enough across a multi-year window to pick up stations that start late,
#: without opening thousands of tars.
DEFAULT_STATION_SCAN_DAYS = 8


#: A member's variables are not what the catalogue pinned in config says.
SchemaDriftError = _SchemaDriftError


# ------------------------------------------------------------------- naming


def station_id(value: str | bytes) -> str:
    """Canonical KNMI station id: the WMO code exactly as the file writes it.

    Deliberately *not* re-padded. The archive writes five characters ("06260")
    and the WIGOS local identifier derived from it is a string by
    specification, so normalising the width here would be a silent way to merge
    or split stations later.
    """
    text = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    return text.strip()


def member_time(name: str) -> datetime:
    """`...L2_202006011200.nc` -> 2020-06-01 12:00 UTC.

    Used to order and select members, not to timestamp observations: the file's
    own `time` variable is authoritative and `parse` reads it from there.
    """
    m = _MEMBER_RE.match(Path(name).name)
    if not m:
        raise ValueError(
            f"{name!r} is not a KMDS member; expected "
            f"'KMDS__OPER_P___10M_OBS_L2_<YYYYMMDDHHMM>.nc'"
        )
    stamp = m.group("stamp")
    return datetime(
        int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]),
        int(stamp[8:10]), int(stamp[10:12]), tzinfo=UTC,
    )


def tar_date(name: str) -> date:
    """The day one archive tar covers, from its filename."""
    m = _TAR_RE.match(Path(name).name)
    if not m:
        raise ValueError(f"{name!r} is not a KMDS daily tar")
    stamp = m.group("from")
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:8]))


def local_path(url: str) -> Path:
    """`file:///a/b` or a bare `/a/b` -> Path. Anything else is a mistake here."""
    parsed = urlparse(str(url))
    if parsed.scheme in ("", "file"):
        return Path(unquote(parsed.path or str(url)))
    raise ValueError(
        f"{url!r} is not a local archive path; this source reads a local tar "
        f"archive and never downloads (see the module docstring)"
    )


# -------------------------------------------------------------- fingerprints


def schema_fingerprint(names: Iterable[str]) -> str:
    """Stable hash of one normalised variable set.

    Normalisation drops the structural variables (coordinates, station table,
    CRS), so the fingerprint tracks what the file *measures*. It is
    order-independent — netCDF variable order is an implementation detail — and
    a genuine addition, removal or rename does move it.
    """
    normalised = sorted({str(n).strip() for n in names} - STRUCTURAL - {""})
    return hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()[:16]


# ----------------------------------------------------------------- variables


def variables_by_key(cfg: ProductConfig) -> list[tuple[str, Variable]]:
    """`[(netCDF variable, canonical Variable), ...]` for everything config wants."""
    out: list[tuple[str, Variable]] = []
    for var in cfg.variables:
        key = var.source_key.strip()
        if not key:
            raise ValueError(f"variable {var.name!r} has an empty source_key")
        if key in STRUCTURAL:
            raise ValueError(
                f"source_key {key!r} is a structural variable, not an observation; "
                f"coordinates and the station table are read automatically"
            )
        out.append((key, var))
    return out


@dataclass(frozen=True, slots=True)
class StationRecord:
    """One row of the station table carried inside every member."""

    station_id: str
    lat: float
    lon: float
    height: float
    name: str
    wigos_id: str


# ------------------------------------------------------------------ the source


class KnmiInsitu:
    """KNMI 10-minute in-situ observations as a `meteopack` source.

    Public API::

        src = KnmiInsitu(source_cfg)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain stations
        src.list(window, product_cfg)    -> list[Unit]        (one per DAY)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == no such day)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        KnmiInsitu.schema_fingerprint(names) -> str

    `session` and `limiter` are accepted and ignored, so a generic driver can
    construct every source the same way: this one reads a local archive and
    issues no requests at all.
    """

    name = "knmi_insitu"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(
            name="knmi_insitu", base_url=f"file://{ARCHIVE_DIR}" if ARCHIVE_DIR else None)
        self._now = now
        self._archive_cache: dict[date, Path] | None = None
        self._station_cache: dict[str, list[StationRecord]] = {}

    # ------------------------------------------------------------------ config

    @property
    def rate_limit(self) -> RateLimit:
        return self.source.rate_limit

    @property
    def archive_dir(self) -> Path:
        """Where the daily tars live. `extra.archive_dir` wins over `base_url`."""
        override = self.source.extra.get("archive_dir")
        if override:
            return Path(str(override)).expanduser()
        where = self.source.base_url or ARCHIVE_DIR
        if not where:
            raise ValueError("no in-situ archive: set $KNMI_INSITU_ARCHIVE, or extra.archive_dir "
                             "in the source config")
        return local_path(where).expanduser()

    def _clock(self) -> datetime:
        return self._now or datetime.now(UTC)

    # ------------------------------------------------------------ archive reads

    def archive(self) -> dict[date, Path]:
        """`{day: tar}` for the whole archive (cached per instance).

        A day with more than one tar keeps the highest counter, which is how a
        re-issued day is meant to be read, and says so rather than picking
        silently.
        """
        if self._archive_cache is not None:
            return self._archive_cache
        found: dict[date, tuple[str, Path]] = {}
        directory = self.archive_dir
        if not directory.is_dir():
            logger.warning("knmi_insitu: no archive directory at %s", directory)
            self._archive_cache = {}
            return self._archive_cache
        for path in directory.iterdir():
            m = _TAR_RE.match(path.name)
            if not m:
                continue
            day = tar_date(path.name)
            counter = m.group("counter")
            previous = found.get(day)
            if previous is not None:
                # Warn whichever order the directory lists them in: iterdir()
                # order is the filesystem's, and warning only when the lower
                # counter came first hid half of the re-issued days.
                keep, drop = sorted((counter, previous[0]), reverse=True)
                logger.warning(
                    "knmi_insitu: %s has several tars; using counter %s over %s",
                    day, keep, drop,
                )
            if previous is None or counter > previous[0]:
                found[day] = (counter, path)
        self._archive_cache = {day: path for day, (_, path) in sorted(found.items())}
        logger.info(
            "knmi_insitu: %d daily tars in %s", len(self._archive_cache), directory
        )
        return self._archive_cache

    # ----------------------------------------------------------------- stations

    def stations(
        self, cfg: ProductConfig, *, window: TimeWindow | None = None
    ) -> pl.DataFrame:
        """In-domain stations, unioned across a sample of days.

        The network grows — 62 stations in 2012, 71 in 2026 — so a table taken
        from one file would silently drop every station commissioned after it.
        Days are sampled evenly across `window` (or the whole archive), the
        latest reading of a station wins, and clipping is the grid's job: a
        station is kept iff `grids.get(cfg.grid_id).contains(lon, lat)`. With no
        `grid_id` the whole network is kept, which is a deliberate escape hatch
        rather than a default.
        """
        import polars as pl

        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        merged: dict[str, StationRecord] = {}
        for tar_path in self._sample_days(cfg, window):
            for rec in self._station_table(tar_path):
                merged[rec.station_id] = rec

        rows = [
            {
                "station_id": rec.station_id,
                "lat": rec.lat,
                "lon": rec.lon,
                # `height`, not DWD's `height_m`: this is the name
                # `core.store.points.STATIONS_SCHEMA` writes, and a column
                # spelled otherwise lands in the store as nulls.
                "height": rec.height,
                "name": rec.name,
                "wigos_id": rec.wigos_id,
            }
            for rec in merged.values()
            if grid is None or grid.contains(rec.lon, rec.lat)
        ]
        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "height": pl.Float64,
            "name": pl.String,
            "wigos_id": pl.String,
        }
        return pl.DataFrame(rows, schema=schema).sort("station_id")

    def _sample_days(self, cfg: ProductConfig, window: TimeWindow | None) -> list[Path]:
        """Evenly spaced tars to read the station table from, newest included."""
        archive = self.archive()
        days = sorted(archive)
        if window is not None:
            start, end = self._window_bounds(window)
            days = [d for d in days if start.date() <= d <= (end - timedelta(days=1)).date()]
        if not days:
            return []
        n = max(1, int(cfg.extra.get("station_scan_days", DEFAULT_STATION_SCAN_DAYS)))
        if len(days) <= n:
            picked = days
        else:
            step = (len(days) - 1) / (n - 1) if n > 1 else 1.0
            picked = sorted({days[round(i * step)] for i in range(n)})
        return [archive[d] for d in picked]

    def _station_table(self, tar_path: Path) -> list[StationRecord]:
        """Station table from the first member of one tar (cached per instance)."""
        key = str(tar_path)
        if key in self._station_cache:
            return self._station_cache[key]
        records: list[StationRecord] = []
        try:
            with tarfile.open(tar_path, "r") as tar:
                names = sorted(_observation_members(tar))
                if names:
                    records = _read_stations(_member_bytes(tar, names[0]))
        except (OSError, tarfile.TarError, KeyError, ValueError) as exc:
            # One unreadable tar must not take out a decade-long station table.
            logger.warning("knmi_insitu: cannot read stations from %s: %s", tar_path, exc)
        self._station_cache[key] = records
        return records

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """One unit per DAY that the archive actually holds.

        A day is the unit because a day is the tar: 144 loose members per day is
        the shape this source exists to avoid, and a half-fetched day would be
        indistinguishable from a whole one in the ledger. Days missing from the
        archive are skipped here rather than emitted and 404'd later.
        """
        start, end = self._window_bounds(window)
        # Validate the window against the product cadence in one place.
        Timeline(start, end, cfg.cadence)

        archive = self.archive()
        first = start.date()
        # Half-open: a window ending exactly at midnight does not want that day.
        last = (end - timedelta(microseconds=1)).date()
        today = self._clock().date()

        units: list[Unit] = []
        missing: list[date] = []
        day = first
        while day <= last:
            path = archive.get(day)
            if path is None:
                missing.append(day)
                day += timedelta(days=1)
                continue
            units.append(
                Unit(
                    key=f"{cfg.source}/{cfg.product}/{day:%Y%m%d}",
                    url=f"file://{path}",
                    valid_time=datetime(day.year, day.month, day.day, tzinfo=UTC),
                    meta={
                        "date": f"{day:%Y%m%d}",
                        "year": f"{day:%Y}",
                        "filename": path.name,
                        "archive": "tar",
                        "granularity": "day",
                        # A closed day's tar is immutable, so its key may be
                        # skipped forever. Today's is still being filled
                        # upstream, and a ledger that marks it done would freeze
                        # the live tail at whatever happened to be in it.
                        "volatile": day >= today,
                    },
                )
            )
            day += timedelta(days=1)
        if missing:
            logger.warning(
                "knmi_insitu: %d day(s) absent from the archive between %s and %s (e.g. %s)",
                len(missing), first, last, ", ".join(f"{d:%Y-%m-%d}" for d in missing[:3]),
            )
        logger.info("knmi_insitu: %d daily unit(s) for %s..%s", len(units), first, last)
        return units

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Put the day's **whole tar** in the raw layer. None if the day is gone.

        A hardlink by default, a copy when the archive is on another filesystem
        or `extra.link_raw` is false: the archive is ~25 MB a day and 5211 days,
        so copying it costs ~130 GB to produce bytes that already exist and are
        never mutated. The link is published with the same `.part` + `os.replace`
        dance a download would use, so an interrupted job cannot leave something
        that looks like a finished unit, and an existing destination is re-hashed
        rather than rewritten — exactly the idempotence `core.http.fetch_file`
        gives the HTTP sources.

        A day that has vanished from the archive is a skip, not a failure, which
        matches how every other source here treats an absent unit.
        """
        source_path = local_path(unit.url)
        if not source_path.is_file():
            logger.info("knmi_insitu: no archive tar for %s (%s)", unit.key, source_path)
            return None

        meta = unit.meta or {}
        dest = Path(raw_root) / meta.get("year", "unknown") / meta.get(
            "filename", f"{unit.key.rsplit('/', 1)[-1]}.tar"
        )
        if dest.exists():
            digest, n = _sha256_file(dest)
            return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit.key)

        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_suffix(f"{dest.suffix}.{os.getpid()}.{uuid.uuid4().hex[:8]}.part")
        link = bool(self.source.extra.get("link_raw", True))
        try:
            if link:
                try:
                    os.link(source_path, part)
                except OSError:
                    # Cross-device, or a filesystem without hardlinks.
                    shutil.copyfile(source_path, part)
            else:
                shutil.copyfile(source_path, part)
            digest, n = _sha256_file(part)
            os.replace(part, dest)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit.key)

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One day's tar -> tidy frame of `time`, `station_id`, canonical values.

        Members are read straight out of the tar; nothing is ever extracted to
        disk. Only the variables `cfg.variables` asks for are read. NaN (this
        feed's `_FillValue`) and any `extra.missing_sentinels` become null
        *before* scale/offset, so a sentinel can never be scaled into a
        plausible-looking number, and the valid range is applied afterwards, so
        it is expressed in canonical units.
        """
        import numpy as np
        import polars as pl

        wanted = variables_by_key(cfg)
        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        in_domain: dict[str, bool] = {}

        times: list[np.ndarray] = []
        ids: list[np.ndarray] = []
        columns: dict[str, list[np.ndarray]] = {var.name: [] for _, var in wanted}

        path = Path(raw.path)
        with tarfile.open(path, "r") as tar:
            members = sorted(_observation_members(tar))
            if not members:
                logger.warning("knmi_insitu: no KMDS members in %s", path)
                return _empty_frame(wanted)
            for name in members:
                data = _member_bytes(tar, name)
                record = _read_member(data, [key for key, _ in wanted], name)
                self._check_schema(record.variables, cfg, name)
                if not wanted:
                    continue
                keep = np.fromiter(
                    (
                        _contains(grid, in_domain, sid, lon, lat)
                        for sid, lon, lat in zip(
                            record.station_ids, record.lon, record.lat, strict=True
                        )
                    ),
                    dtype=bool,
                    count=len(record.station_ids),
                )
                if not keep.any():
                    continue
                n = int(keep.sum())
                times.append(np.full(n, record.time, dtype="datetime64[ns]"))
                ids.append(record.station_ids[keep])
                for key, var in wanted:
                    values = record.values.get(key)
                    # A partial member has no array for a variable it lacks.
                    # Null for those rows, which is what the absence means -
                    # not zero, and not a reason to drop the member.
                    columns[var.name].append(
                        np.full(n, np.nan, dtype="float32")
                        if values is None else values[keep]
                    )

        if not wanted or not times:
            return _empty_frame(wanted)

        frame = pl.DataFrame(
            {
                "time": pl.Series(np.concatenate(times))
                .cast(pl.Datetime("us"))
                .dt.replace_time_zone("UTC"),
                "station_id": pl.Series(np.concatenate(ids), dtype=pl.String),
                **{
                    name: pl.Series(np.concatenate(chunks), dtype=pl.Float64)
                    for name, chunks in columns.items()
                },
            }
        )
        frame = _apply_variables(frame, [var for _, var in wanted], cfg)
        frame = self._drop_off_cadence(frame, cfg)
        return frame.unique(subset=["time", "station_id"], keep="last").sort(
            ["time", "station_id"]
        )

    def _check_schema(self, present: Sequence[str], cfg: ProductConfig, member: str) -> None:
        """Refuse to parse a member reporting something the catalogue never saw.

        Pinning a per-file fingerprint the way DWD does would reject most of the
        archive: `tg` and `Q24H` appear in 2021 and are simply absent before it,
        so every file is a subset or a superset of some other file. What is
        pinned is the union — `extra.known_variables`, taken from `scan` — and
        `cfg.schema_fingerprint` hashes that list, so editing it without
        re-pinning the hash fails loudly instead of quietly widening what counts
        as known.
        """
        known = cfg.extra.get("known_variables") or []
        if not known:
            return
        pinned = cfg.schema_fingerprint
        if pinned:
            seen = schema_fingerprint(known)
            if seen != pinned:
                raise SchemaDriftError(
                    f"config is inconsistent: extra.known_variables hashes to {seen} but "
                    f"schema_fingerprint pins {pinned}. Re-run `scan` and re-pin both together."
                )
        added = sorted((set(present) - STRUCTURAL) - set(known))
        if added:
            raise SchemaDriftError(
                f"{member}: KNMI reports variable(s) {added} that are not in the pinned "
                f"catalogue ({len(known)} names). Re-run `scan` and re-pin "
                f"extra.known_variables and schema_fingerprint after checking what moved."
            )

    def _drop_off_cadence(self, out: pl.DataFrame, cfg: ProductConfig) -> pl.DataFrame:
        """Drop observations that do not sit on the product's cadence grid.

        `Timeline.index_of` returns -1 for an off-grid time. Those rows are
        dropped rather than snapped, so a cadence mismatch shows up as missing
        data in validation instead of a silently shifted series.
        """
        if not cfg.extra.get("drop_off_cadence", True) or out.is_empty():
            return out
        import polars as pl

        first = out["time"].min()
        last = out["time"].max()
        start = datetime(first.year, first.month, first.day, tzinfo=UTC)
        end = datetime(last.year, last.month, last.day, tzinfo=UTC) + timedelta(days=1)
        idx = Timeline(start, end, cfg.cadence).index_of(out["time"].to_numpy())
        n_bad = int((idx < 0).sum())
        if n_bad:
            logger.warning("knmi_insitu: dropping %d off-cadence observation(s)", n_bad)
            return out.filter(pl.Series(idx >= 0))
        return out

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile], *, sample_members: int = 3) -> dict:
        """Report every distinct variable set across `raw_files`.

        Upstream adds variables without announcing it — `tg` and `Q24H` arrived
        in 2021 — so the pipeline pins the catalogue this returns and `parse`
        refuses to run against anything outside it. Only the first
        `sample_members` members of each day are opened: a day's members share a
        schema, and opening 144 of them per tar would make scanning the archive
        cost more than packing it. The report is deliberately a plain dict: it is
        meant to be written to JSON next to the store and diffed between runs.
        """
        variants: dict[tuple[str, ...], dict] = {}
        errors: list[dict[str, str]] = []
        union: set[str] = set()
        for raw in raw_files:
            path = Path(raw.path)
            try:
                with tarfile.open(path, "r") as tar:
                    members = sorted(_observation_members(tar))
                    if not members:
                        raise ValueError("no KMDS members")
                    for name in members[: max(1, sample_members)]:
                        record = _read_member(_member_bytes(tar, name), [], name)
                        columns = tuple(
                            sorted(set(record.variables) - STRUCTURAL)
                        )
                        entry = variants.setdefault(
                            columns,
                            {
                                "variables": list(columns),
                                "n_variables": len(columns),
                                "fingerprint": schema_fingerprint(columns),
                                "count": 0,
                                "n_stations": [],
                                "examples": [],
                            },
                        )
                        entry["count"] += 1
                        union.update(columns)
                        if record.n_stations not in entry["n_stations"]:
                            entry["n_stations"].append(record.n_stations)
                        if len(entry["examples"]) < 3:
                            entry["examples"].append(f"{path.name}:{name}")
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue

        ordered = sorted(variants.values(), key=lambda e: e["fingerprint"])
        for entry in ordered:
            entry["n_stations"].sort()
        # The union, not a per-file variant, is what a config pins: a 2012 file
        # is a strict subset of a 2026 one, so pinning either rejects the other.
        known = sorted(union)
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": sorted(e["fingerprint"] for e in ordered),
            "known_variables": known,
            "fingerprint": schema_fingerprint(known) if known else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ------------------------------------------------------------------- tar helpers


def _observation_members(tar: tarfile.TarFile) -> list[str]:
    """Names of the KMDS members in one tar, ignoring anything else in it."""
    return [m.name for m in tar.getmembers() if m.isfile() and _MEMBER_RE.match(Path(m.name).name)]


def _member_bytes(tar: tarfile.TarFile, name: str) -> bytes:
    handle = tar.extractfile(name)
    if handle is None:
        raise KeyError(name)
    return handle.read()


def _sha256_file(path: Path) -> tuple[str, int]:
    h = hashlib.sha256()
    n = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
            n += len(chunk)
    return h.hexdigest(), n


# ---------------------------------------------------------------- netCDF reads


@dataclass(frozen=True, slots=True)
class Member:
    """One 10-minute member, reduced to arrays."""

    time: object                      # numpy datetime64[ns]
    station_ids: object               # numpy array of str
    lat: object
    lon: object
    variables: tuple[str, ...]
    values: dict                      # netCDF name -> numpy float64 array
    n_stations: int


#: `<unit> since <iso timestamp>`, how CF writes a time axis.
_TIME_UNITS_RE = re.compile(r"^\s*(?P<unit>\w+)\s+since\s+(?P<epoch>.+?)\s*$")

_TIME_SCALE = {
    "seconds": "s", "second": "s", "secs": "s", "sec": "s", "s": "s",
    "minutes": "m", "minute": "m", "mins": "m", "min": "m",
    "hours": "h", "hour": "h", "hrs": "h", "hr": "h",
    "days": "D", "day": "D",
}


def _decode_time(raw, attrs, name: str):
    """CF `<unit> since <epoch>` -> datetime64[ns], without a decoder stack.

    Hand-rolled because the rest of this module reads h5netcdf directly: going
    through xarray's decoding costs ~676 ms per member against ~35 ms, which
    over 144 members a day and 5211 days is the difference between a two-hour
    pack and a three-day one. Measured, not guessed.
    """
    import numpy as np

    units = _text(attrs.get("units", ""))
    m = _TIME_UNITS_RE.match(units)
    if not m:
        raise SchemaDriftError(f"{name}: time has no CF units, got {units!r}")
    scale = _TIME_SCALE.get(m.group("unit").lower())
    if scale is None:
        raise SchemaDriftError(f"{name}: unsupported time unit in {units!r}")
    epoch_text = m.group("epoch").replace("Z", "").strip()
    try:
        epoch = np.datetime64(epoch_text.replace(" ", "T"), "ns")
    except ValueError:
        raise SchemaDriftError(f"{name}: unparseable time epoch {epoch_text!r}") from None
    offsets = np.asarray(raw, dtype="float64").reshape(-1)
    return epoch + (offsets * np.timedelta64(1, scale).astype("timedelta64[ns]").astype(
        "int64"
    )).astype("int64").astype("timedelta64[ns]")


def _scalar(value):
    """h5py gives a 1-element array where h5netcdf gives a scalar."""
    import numpy as np

    arr = np.asarray(value).reshape(-1)
    return arr[0] if arr.size else None


def _values(dataset, name: str, key: str):
    """One variable as float64, honouring CF scale_factor/add_offset if present.

    These files carry neither today and store raw float64 with `_FillValue`
    NaN, but reading HDF5 directly means nothing else would apply them if KNMI
    ever starts packing this feed.
    """
    import numpy as np

    out = np.asarray(dataset[...], dtype="float64").reshape(-1)
    attrs = dataset.attrs
    if "scale_factor" in attrs:
        out = out * float(_scalar(attrs["scale_factor"]))
    if "add_offset" in attrs:
        out = out + float(_scalar(attrs["add_offset"]))
    fill = _scalar(attrs["_FillValue"]) if "_FillValue" in attrs else None
    if fill is not None and np.isfinite(float(fill)):
        # NaN fill needs nothing; a numeric one has to be nulled before it is
        # ever scaled or compared, same rule as everywhere else here.
        out = np.where(out == float(fill), np.nan, out)
    if out.size == 0:
        raise SchemaDriftError(f"{name}: variable {key!r} is empty")
    return out


def _read_member(data: bytes, wanted: Sequence[str], name: str):
    """Open one member from memory and pull out exactly what is needed.

    The bytes stay in memory: writing 144 members a day back to disk is the
    filesystem problem this source exists to avoid.
    """
    import h5py
    import numpy as np

    # h5py rather than h5netcdf, and the difference is not small. netCDF-4 is
    # HDF5, so the variables are plain datasets and this reads the same bytes
    # - verified over 432 arrays spread across the archive, every one
    # identical. What h5netcdf adds is a netCDF *view*: its File() walks the
    # whole hierarchy eagerly to build it, which profiled at 65% of the time
    # for one tar - 36,864 group lookups and 33,264 attribute reads to find
    # the fourteen variables we actually want. 49 ms per member against
    # 13.4 ms, on a source that is CPU-bound rather than I/O-bound.
    with h5py.File(io.BytesIO(data), "r") as ds:
        variables = tuple(str(v) for v in ds)
        missing = [key for key in wanted if key not in ds]
        if missing and len(missing) == len(wanted):
            # Nothing we want at all: this is not a partial member, it is a
            # different kind of file, and reading it would be guessing.
            raise SchemaDriftError(
                f"{name}: none of the configured variables {sorted(wanted)} are "
                f"present; it holds {sorted(set(variables) - STRUCTURAL)}"
            )
        if missing:
            # A partial member. KNMI disseminates these: 2018-07-27 09:20
            # carries the wind and pressure variables and not the temperature
            # ones. Refusing it cost the entire day's tar - 144 members - for
            # the sake of one. The variables that are there are real, so they
            # are read and the rest are null, which is what null means.
            logger.warning(
                "knmi_insitu: %s has no %s; reading the %d variable(s) it does have",
                name, ", ".join(missing), len(wanted) - len(missing),
            )
        when = _decode_time(ds["time"][...], ds["time"].attrs, name)
        if when.size != 1:
            raise SchemaDriftError(
                f"{name}: expected one time step per member, got {when.size}"
            )
        # h5py returns fixed-width strings as bytes; station_id must see text.
        sids = np.array(
            [station_id(_text(s)) for s in np.asarray(ds["station"][...]).reshape(-1)],
            dtype=object,
        )
        lat = np.asarray(ds["lat"][...], dtype="float64").reshape(-1)
        lon = np.asarray(ds["lon"][...], dtype="float64").reshape(-1)
        values = {
            key: _values(ds[key], name, key)
            for key in wanted
            if key in ds
        }
    return Member(
        time=when[0],
        station_ids=sids,
        lat=lat,
        lon=lon,
        variables=variables,
        values=values,
        n_stations=int(sids.size),
    )


def _read_stations(data: bytes) -> list[StationRecord]:
    """The station table carried by one member."""
    import h5py
    import numpy as np

    # h5py for the same reason as `_read_member`: netCDF-4 is HDF5, and
    # h5netcdf's eager hierarchy walk is most of the cost of opening a member.
    with h5py.File(io.BytesIO(data), "r") as ds:
        sids = np.asarray(ds["station"][...]).reshape(-1)
        lat = np.asarray(ds["lat"][...], dtype="float64").reshape(-1)
        lon = np.asarray(ds["lon"][...], dtype="float64").reshape(-1)
        height = np.asarray(ds["height"][...], dtype="float64").reshape(-1)
        names = np.asarray(ds["stationname"][...]).reshape(-1)
        wsi = (
            np.asarray(ds["wsi"][...]).reshape(-1)
            if "wsi" in ds
            else np.array([""] * sids.size)
        )
    return [
        StationRecord(
            station_id=station_id(_text(sids[i])),
            lat=float(lat[i]),
            lon=float(lon[i]),
            height=float(height[i]),
            name=_text(names[i]),
            wigos_id=_text(wsi[i]),
        )
        for i in range(sids.size)
    ]


def _text(value) -> str:
    return (value.decode("utf-8") if isinstance(value, bytes) else str(value)).strip()


def _contains(grid, cache: dict[str, bool], sid: str, lon: float, lat: float) -> bool:
    """Grid footprint test, memoised per station.

    Every member repeats the whole station table, so without this a day costs
    144 x 66 ray casts to answer a question with 66 distinct answers.
    """
    if grid is None:
        return True
    hit = cache.get(sid)
    if hit is None:
        hit = grid.contains(lon, lat)
        cache[sid] = hit
    return hit


# ------------------------------------------------------------- frame assembly


def _empty_frame(wanted: Sequence[tuple[str, Variable]]) -> pl.DataFrame:
    import polars as pl

    schema: dict[str, object] = {
        "time": pl.Datetime("us", "UTC"),
        "station_id": pl.String,
    }
    for _, var in wanted:
        schema[var.name] = _polars_dtype(var.dtype)
    return pl.DataFrame(schema=schema)


def _apply_variables(
    frame: pl.DataFrame, variables: Sequence[Variable], cfg: ProductConfig
) -> pl.DataFrame:
    """NaN/sentinel -> null, then scale/offset, then valid-range mask, then dtype.

    NaN first and explicitly: this feed's `_FillValue` is NaN and polars treats a
    float NaN as a value, not a null, so it would otherwise survive into the
    store and poison every mean computed over it.
    """
    import polars as pl

    sentinels = [float(v) for v in (cfg.extra.get("missing_sentinels") or [])]
    exprs = []
    for var in variables:
        col = pl.col(var.name)
        col = pl.when(col.is_nan()).then(None).otherwise(col)
        if sentinels:
            col = pl.when(col.is_in(sentinels)).then(None).otherwise(col)
        if var.scale is not None:
            col = col * var.scale
        if var.offset is not None:
            col = col + var.offset
        if var.valid_min is not None:
            col = pl.when(col < var.valid_min).then(None).otherwise(col)
        if var.valid_max is not None:
            col = pl.when(col > var.valid_max).then(None).otherwise(col)
        exprs.append(col.cast(_polars_dtype(var.dtype), strict=False).alias(var.name))
    return frame.with_columns(exprs) if exprs else frame


def _polars_dtype(name: str):
    import polars as pl

    lookup = {
        "float32": pl.Float32,
        "float64": pl.Float64,
        "int16": pl.Int16,
        "int32": pl.Int32,
        "int64": pl.Int64,
    }
    try:
        return lookup[name.lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype {name!r} for a KNMI variable") from None
