"""DWD Climate Data Center — 10-minute station observations.

The feed is a plain HTTPS directory tree with no key:

    <base>/<group>/{historical,recent,now}/

Each `<group>` is one parameter group (air_temperature, wind, ...). Every group
directory holds one ZIP per station plus a `*_Beschreibung_Stationen.txt`
station catalogue. Inside each ZIP is a single `produkt_*.txt`: `;`-separated,
latin-1, fields padded with spaces, `-999` for missing, a trailing `eor`
column.

Three pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store if you get it wrong:

* The station catalogue is fixed-width, and *its data rows do not line up with
  its own header rule*: the dashes put the name field at column 66, the rows put
  it at the end of the numeric prefix. Splitting on whitespace instead loses
  station names that contain spaces ("Neuruppin-Alt Ruppin"). We locate the
  numeric prefix with a regex and slice the two 41-character text fields that
  follow it.
* `historical/` timestamps are MEZ (UTC+1) before 2000 and UTC from 2000
  onwards; `recent/` and `now/` are always UTC. See `pre_2000_mez` below.
* Absent station/period combinations are normal — a station that reports wind
  but not solar simply 404s in `solar/`. `fetch` skips those rather than failing
  the job.

The upstream schema drifts without announcement, so `scan` reports every column
set seen and `parse` refuses to run against a pinned fingerprint it does not
match.
"""
from __future__ import annotations

import hashlib
import io
import logging
import re
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from meteopack.core import grids
from meteopack.core.config import ProductConfig, RateLimit, SourceConfig, Variable
from meteopack.core.timeline import Timeline
from meteopack.sources.common import SchemaDriftError as _SchemaDriftError
from meteopack.sources.base import RawFile, TimeWindow, Unit

if TYPE_CHECKING:  # keep polars off the import path of `meteopack.sources`
    import polars as pl

logger = logging.getLogger(__name__)

BASE_URL = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes"
)

#: `-999` in every column of every group, per the DWD dataset descriptions.
MISSING = -999.0

#: `recent/` is documented as a rolling window of the last 500 days.
RECENT_DAYS = 500

_PRODUKT_PREFIX = "produkt_"
_HREF_RE = re.compile(r'href="([^"?]+)"')
_EOR = "eor"

# Columns that are structural rather than observations.
_ID_COL = "STATIONS_ID"
_TIME_COL = "MESS_DATUM"
_QN_COL = "QN"

#: The instant from which `historical/` timestamps are UTC rather than MEZ.
_UTC_FROM = datetime(2000, 1, 1, tzinfo=UTC)


#: A produkt file's columns do not match the fingerprint pinned in config.
SchemaDriftError = _SchemaDriftError


@dataclass(frozen=True, slots=True)
class Group:
    """One 10-minute parameter group: its subdirectory and filename tokens."""

    key: str          # subdirectory under BASE_URL
    file_token: str   # token in `10minutenwerte_<token>_<id>_*.zip`
    desc_token: str   # token in `zehn_{min,now}_<token>_Beschreibung_Stationen.txt`

    def archive_url(self, subdir: str, filename: str) -> str:
        return f"{BASE_URL}/{self.key}/{subdir}/{filename}"

    def description_name(self, subdir: str) -> str:
        stem = "zehn_now" if subdir == "now" else "zehn_min"
        return f"{stem}_{self.desc_token}_Beschreibung_Stationen.txt"

    def description_url(self, subdir: str = "recent") -> str:
        return f"{BASE_URL}/{self.key}/{subdir}/{self.description_name(subdir)}"


GROUPS: tuple[Group, ...] = (
    Group("air_temperature", "TU", "tu"),
    Group("extreme_temperature", "extrema_temp", "tx"),
    Group("extreme_wind", "extrema_wind", "fx"),
    Group("precipitation", "nieder", "rr"),
    Group("solar", "SOLAR", "sd"),
    Group("wind", "wind", "ff"),
)
GROUPS_BY_KEY: dict[str, Group] = {g.key: g for g in GROUPS}
_GROUPS_BY_TOKEN: dict[str, Group] = {g.file_token: g for g in GROUPS}

SUBDIRS: tuple[str, ...] = ("historical", "recent", "now")

# `10minutenwerte_<token>_<id>_<from8>_<to8>_hist.zip` / `_akt.zip` / `_now.zip`
_ARCHIVE_RE = re.compile(
    r"^10minutenwerte_(?P<token>.+?)_(?P<station>\d{4,6})"
    r"(?:_(?P<from>\d{8})_(?P<to>\d{8}))?"
    r"_(?P<kind>hist|akt|now)\.zip$"
)

# The numeric prefix of a station-description row. Everything after it is the
# two 41-character text fields (name, federal state) plus the licence column.
_STATION_RE = re.compile(
    r"^\s*(?P<id>\d{1,6})\s+(?P<from>\d{8})\s+(?P<to>\d{8})\s+"
    r"(?P<height>-?\d+(?:\.\d+)?)\s+(?P<lat>-?\d+(?:\.\d+)?)\s+(?P<lon>-?\d+(?:\.\d+)?)\s"
)
_TEXT_FIELD_WIDTH = 41


@dataclass(frozen=True, slots=True)
class StationRecord:
    """One row of a `*_Beschreibung_Stationen.txt` catalogue."""

    station_id: str
    from_date: date
    to_date: date
    height_m: float
    lat: float
    lon: float
    name: str
    state: str

    def covers(self, start: datetime, end: datetime) -> bool:
        """True if the station's reporting period overlaps the half-open window."""
        return self.from_date <= end.date() and self.to_date >= start.date()


def station_id(value: str | int) -> str:
    """Canonical five-digit station id. DWD pads inconsistently across files."""
    return f"{int(value):05d}"


def parse_station_description(text: str) -> list[StationRecord]:
    """Parse a station catalogue.

    The two trailing text fields are sliced at fixed width rather than split on
    whitespace, because station names contain spaces and commas, and because the
    file's own header rule is misaligned with its data rows. Rows that do not
    match the numeric prefix (header, dashes, blank lines, the occasional
    truncated tail) are skipped rather than raising: a single malformed row must
    not take out a 500-station catalogue.
    """
    records: list[StationRecord] = []
    for line in text.splitlines():
        m = _STATION_RE.match(line)
        if not m:
            continue
        rest = line[m.end() :]
        name = rest[:_TEXT_FIELD_WIDTH].strip()
        state = rest[_TEXT_FIELD_WIDTH : 2 * _TEXT_FIELD_WIDTH].strip()
        if not state:
            # Older/narrower variants: fall back to splitting the tail from the
            # right, which still survives spaces in the station name.
            tail = rest.rsplit(maxsplit=2)
            if len(tail) == 3:
                name, state = tail[0].strip(), tail[1].strip()
        try:
            records.append(
                StationRecord(
                    station_id=station_id(m.group("id")),
                    from_date=_yyyymmdd(m.group("from")),
                    to_date=_yyyymmdd(m.group("to")),
                    height_m=float(m.group("height")),
                    lat=float(m.group("lat")),
                    lon=float(m.group("lon")),
                    name=name,
                    state=state,
                )
            )
        except ValueError:
            logger.warning("unparseable station row: %r", line[:80])
    return records


def _yyyymmdd(value: str) -> date:
    return date(int(value[:4]), int(value[4:6]), int(value[6:8]))


def schema_fingerprint(columns: Iterable[str]) -> str:
    """Stable hash of one normalised produkt column set.

    Normalisation strips the field padding and drops the `eor` terminator, so
    the fingerprint tracks *what* a file contains rather than how it is spaced.
    It is order-independent (DWD has reordered columns between versions without
    changing their meaning); a genuine addition, removal or rename does move it.
    """
    normalised = sorted({c.strip() for c in columns} - {_EOR, ""})
    digest = hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()
    return digest[:16]


def product_fingerprint(by_group: Mapping[str, str]) -> str:
    """Stable hash of a whole product's per-group fingerprints."""
    joined = "|".join(f"{k}={by_group[k]}" for k in sorted(by_group))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _split_source_key(key: str) -> tuple[str, str]:
    """`"air_temperature:TT_10"` -> `("air_temperature", "TT_10")`."""
    group, _, column = key.partition(":")
    if not column:
        raise ValueError(
            f"source_key {key!r} must be '<group>:<DWD column>', e.g. 'air_temperature:TT_10'"
        )
    if group not in GROUPS_BY_KEY:
        raise ValueError(f"unknown DWD group {group!r}; known: {', '.join(GROUPS_BY_KEY)}")
    return group, column


def variables_by_group(cfg: ProductConfig) -> dict[str, list[tuple[str, Variable]]]:
    """`{group: [(dwd_column, variable), ...]}` for everything the config wants."""
    out: dict[str, list[tuple[str, Variable]]] = {}
    for var in cfg.variables:
        group, column = _split_source_key(var.source_key)
        out.setdefault(group, []).append((column, var))
    return out


class DwdStations:
    """DWD CDC 10-minute station observations as a `meteopack` source.

    Public API::

        src = DwdStations(source_cfg, session=..., limiter=...)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain stations
        src.list(window, product_cfg)    -> list[Unit]        (one per station ZIP)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == upstream 404)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        DwdStations.schema_fingerprint(columns) -> str

    `limiter` is whatever `meteopack.core.http` expects; it is passed straight
    through to `fetch_file`. Listing the directory tree and the station
    catalogue uses `requests` directly, because those are HTML/text reads rather
    than archive downloads and `core.http` only owns the latter.
    """

    name = "dwd"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="dwd", base_url=BASE_URL)
        self.base_url = (self.source.base_url or BASE_URL).rstrip("/")
        self._session = session
        self._limiter = limiter
        self._now = now
        self._listing_cache: dict[str, list[str]] = {}
        self._description_cache: dict[str, list[StationRecord]] = {}

    # ------------------------------------------------------------------ config

    @property
    def rate_limit(self) -> RateLimit:
        return self.source.rate_limit

    @property
    def timeout(self) -> float:
        return float(self.source.extra.get("timeout_seconds", 120))

    def _clock(self) -> datetime:
        return self._now or datetime.now(UTC)

    def _requests_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            agent = self.source.extra.get("user_agent", "meteopack/0.1 (+dwd-opendata)")
            self._session.headers["User-Agent"] = str(agent)
        return self._session

    # ------------------------------------------------------------- remote reads

    def _get_text(self, url: str, *, encoding: str = "utf-8") -> str:
        """One plain GET, decoded. Overridable seam for tests."""
        resp = self._requests_session().get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content.decode(encoding)

    def list_directory(self, url: str) -> list[str]:
        """Filenames in one Apache-style directory listing (cached per instance)."""
        if url in self._listing_cache:
            return self._listing_cache[url]
        try:
            html = self._get_text(url)
        except Exception as exc:  # noqa: BLE001 - a missing subdir is not fatal
            logger.warning("directory listing failed for %s: %s", url, exc)
            self._listing_cache[url] = []
            return []
        names = [
            n
            for n in _HREF_RE.findall(html)
            if not n.endswith("/") and not n.startswith((".", "/", "?"))
        ]
        self._listing_cache[url] = names
        return names

    def description(self, group: Group, subdir: str = "recent") -> list[StationRecord]:
        """Station catalogue for one group, decoded latin-1 (cached per instance)."""
        url = group.description_url(subdir)
        if url in self._description_cache:
            return self._description_cache[url]
        try:
            text = self._get_text(url, encoding="latin-1")
        except Exception as exc:  # noqa: BLE001 - one missing catalogue is survivable
            logger.warning("station catalogue fetch failed for %s: %s", group.key, exc)
            self._description_cache[url] = []
            return []
        records = parse_station_description(text)
        self._description_cache[url] = records
        return records

    # ----------------------------------------------------------------- stations

    def stations(
        self,
        cfg: ProductConfig,
        *,
        window: TimeWindow | None = None,
        groups: Sequence[Group] | None = None,
    ) -> pl.DataFrame:
        """In-domain stations across the groups this product needs.

        Clipping is the grid's job: a station is kept iff
        `grids.get(cfg.grid_id).contains(lon, lat)`. With no `grid_id` the whole
        German network is kept, which is a deliberate escape hatch rather than a
        default — every product config here sets one.
        """
        import polars as pl

        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        wanted = list(groups) if groups is not None else self._groups_for(cfg)
        start, end = self._window_bounds(window) if window else (None, None)

        merged: dict[str, dict] = {}
        for group in wanted:
            for rec in self.description(group):
                if grid is not None and not grid.contains(rec.lon, rec.lat):
                    continue
                if start is not None and not rec.covers(start, end):
                    continue
                entry = merged.setdefault(
                    rec.station_id,
                    {
                        "station_id": rec.station_id,
                        "lat": rec.lat,
                        "lon": rec.lon,
                        "height_m": rec.height_m,
                        "name": rec.name,
                        "state": rec.state,
                        "from_date": rec.from_date,
                        "to_date": rec.to_date,
                        "groups": [],
                    },
                )
                entry["from_date"] = min(entry["from_date"], rec.from_date)
                entry["to_date"] = max(entry["to_date"], rec.to_date)
                if group.key not in entry["groups"]:
                    entry["groups"].append(group.key)

        rows = [dict(e, groups=",".join(sorted(e["groups"]))) for e in merged.values()]
        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "height_m": pl.Float64,
            "name": pl.String,
            "state": pl.String,
            "from_date": pl.Date,
            "to_date": pl.Date,
            "groups": pl.String,
        }
        frame = pl.DataFrame(rows, schema=schema)
        return frame.sort("station_id")

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every station ZIP that overlaps `window`, for in-domain stations only.

        No archive is downloaded; only directory listings and the station
        catalogues are read.
        """
        start, end = self._window_bounds(window)
        # Validate the window against the product cadence in one place.
        Timeline(start, end, cfg.cadence)

        groups = self._groups_for(cfg)
        table = self.stations(cfg, window=window, groups=groups)
        keep = set(table["station_id"].to_list())
        only = cfg.extra.get("station_ids")
        if only:
            keep &= {station_id(s) for s in only}
        limit = cfg.extra.get("max_stations")
        if limit:
            keep = set(sorted(keep)[: int(limit)])
        if not keep:
            logger.warning("no in-domain DWD stations for grid %s", cfg.grid_id)
            return []

        subdirs = self._subdirs(start, end, cfg)
        units: list[Unit] = []
        seen: set[str] = set()
        for group in groups:
            for subdir in subdirs:
                url = f"{self.base_url}/{group.key}/{subdir}/"
                for filename in self.list_directory(url):
                    unit = self._unit_for(group, subdir, filename, keep, start, end, cfg)
                    if unit is not None and unit.key not in seen:
                        seen.add(unit.key)
                        units.append(unit)
        logger.info(
            "dwd: %d units across %d groups x %s for %d stations",
            len(units),
            len(groups),
            "/".join(subdirs),
            len(keep),
        )
        return units

    def _unit_for(
        self,
        group: Group,
        subdir: str,
        filename: str,
        stations: set[str],
        start: datetime,
        end: datetime,
        cfg: ProductConfig,
    ) -> Unit | None:
        m = _ARCHIVE_RE.match(filename)
        if not m or m.group("token") != group.file_token:
            return None
        sid = station_id(m.group("station"))
        if sid not in stations:
            return None
        valid_time: datetime | None = None
        if m.group("from"):
            covered_from = _yyyymmdd(m.group("from"))
            covered_to = _yyyymmdd(m.group("to"))
            if covered_to < start.date() or covered_from > end.date():
                return None
            valid_time = datetime(
                covered_from.year, covered_from.month, covered_from.day, tzinfo=UTC
            )
        meta = {
            "group": group.key,
            "subdir": subdir,
            "station_id": sid,
            "filename": filename,
        }
        if m.group("from"):
            meta["covers_from"] = m.group("from")
            meta["covers_to"] = m.group("to")
        return Unit(
            key=f"{cfg.source}/{cfg.product}/{group.key}/{subdir}/{filename[:-4]}",
            url=group.archive_url(subdir, filename),
            valid_time=valid_time,
            meta=meta,
        )

    def _subdirs(self, start: datetime, end: datetime, cfg: ProductConfig) -> list[str]:
        """Which of historical/recent/now can hold data for this window.

        `recent/` is a rolling 500-day window and `now/` is the current day, so
        a 2015 backfill must not pay for 9000 listing-irrelevant station files.
        """
        override = cfg.extra.get("subdirs")
        if override:
            unknown = set(override) - set(SUBDIRS)
            if unknown:
                raise ValueError(f"unknown DWD subdir(s): {sorted(unknown)}")
            return [s for s in SUBDIRS if s in set(override)]
        recent_days = int(cfg.extra.get("recent_days", RECENT_DAYS))
        now = self._clock()
        recent_from = now - timedelta(days=recent_days)
        out: list[str] = []
        if start < recent_from:
            out.append("historical")
        if end > recent_from:
            out.append("recent")
        if end > now - timedelta(days=1):
            out.append("now")
        return out or ["historical"]

    def _groups_for(self, cfg: ProductConfig) -> list[Group]:
        wanted = set(variables_by_group(cfg)) or set(GROUPS_BY_KEY)
        return [g for g in GROUPS if g.key in wanted]

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Download one station ZIP. Returns None if upstream has no such file.

        Station/period combinations missing upstream are routine — a station
        that reports wind need not report solar — so a 404 is a skip, not a
        failure. Everything else (retries, `.part` + atomic rename, checksum,
        rate limiting) belongs to `meteopack.core.http`; the import is local so
        tests can monkeypatch it.
        """
        from meteopack.core.http import NotFoundError, fetch_file

        meta = unit.meta or {}
        dest = Path(raw_root) / meta.get("group", "unknown") / meta.get(
            "subdir", "unknown"
        ) / meta.get("filename", f"{unit.key.rsplit('/', 1)[-1]}.zip")
        rl = self.rate_limit
        try:
            return fetch_file(
                unit.url,
                dest,
                unit_key=unit.key,
                session=self._requests_session(),
                limiter=self._limiter,
                retries=rl.max_retries,
                backoff=rl.backoff_seconds,
                timeout=self.timeout,
                # Rewritten upstream under the same name: the ledger re-queues it,
                # so the download must actually happen rather than skip.
                overwrite=bool((unit.meta or {}).get("volatile")),
            )
        except NotFoundError:
            logger.info("dwd: no upstream file for %s (404)", unit.key)
            return None

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One station ZIP -> tidy frame of `time`, `station_id`, canonical values.

        Only the columns `cfg.variables` asks for are read. `-999` becomes null
        *before* scale/offset, so a sentinel can never be scaled into a
        plausible-looking number, and the valid range is applied afterwards, so
        it is expressed in canonical units.
        """
        import polars as pl

        group = self._group_for(raw)
        wanted = variables_by_group(cfg).get(group.key, [])
        data = _read_produkt(Path(raw.path))
        if data is None:
            logger.warning("no produkt member in %s", raw.path)
            return _empty_frame(wanted)
        header = _header_columns(data)
        self._check_fingerprint(group, header, cfg)
        if not wanted:
            return _empty_frame(wanted)

        by_stripped = {c.strip(): c for c in header}
        missing = [col for col, _ in wanted if col not in by_stripped]
        if missing:
            raise SchemaDriftError(
                f"{raw.path.name}: group {group.key} is missing column(s) "
                f"{missing}; present: {sorted(by_stripped)}"
            )
        for structural in (_ID_COL, _TIME_COL):
            if structural not in by_stripped:
                raise SchemaDriftError(f"{raw.path.name}: no {structural} column")

        select = [by_stripped[_ID_COL], by_stripped[_TIME_COL]]
        select += [by_stripped[col] for col, _ in wanted]
        frame = pl.read_csv(io.BytesIO(data), separator=";", infer_schema=False, columns=select)
        rename = {by_stripped[_ID_COL]: _ID_COL, by_stripped[_TIME_COL]: _TIME_COL}
        frame = frame.rename({k: v for k, v in rename.items() if k != v})

        out = frame.select(
            pl.col(_TIME_COL)
            .str.strip_chars()
            .str.strptime(pl.Datetime("us"), "%Y%m%d%H%M", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("time"),
            pl.col(_ID_COL).str.strip_chars().str.zfill(5).alias("station_id"),
            *[
                pl.col(by_stripped[col]).str.strip_chars().cast(pl.Float64, strict=False).alias(
                    var.name
                )
                for col, var in wanted
            ],
        ).drop_nulls("time")

        out = _apply_variables(out, [var for _, var in wanted])
        out = self._fix_timezone(out, raw, cfg)
        out = out.unique(subset=["time", "station_id"], keep="last").sort("time")
        return self._drop_off_cadence(out, cfg)

    def _group_for(self, raw: RawFile) -> Group:
        meta_group = None
        m = _ARCHIVE_RE.match(Path(raw.path).name)
        if m:
            meta_group = _GROUPS_BY_TOKEN.get(m.group("token"))
        if meta_group is None:
            for group in GROUPS:
                if f"/{group.key}/" in str(raw.path).replace("\\", "/"):
                    meta_group = group
                    break
        if meta_group is None:
            raise ValueError(f"cannot tell which DWD group {raw.path} belongs to")
        return meta_group

    def _check_fingerprint(self, group: Group, header: Sequence[str], cfg: ProductConfig) -> None:
        pinned = (cfg.extra.get("schema_fingerprints") or {}).get(group.key)
        if not pinned:
            return
        seen = schema_fingerprint(header)
        if seen != pinned:
            raise SchemaDriftError(
                f"DWD {group.key} schema changed: pinned {pinned}, got {seen} "
                f"(columns {[c.strip() for c in header]}). Re-run `scan` and re-pin "
                f"extra.schema_fingerprints after checking what moved."
            )

    def _fix_timezone(self, out: pl.DataFrame, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """Shift pre-2000 `historical/` timestamps from MEZ to UTC.

        DWD assigns historical measurements a MEZ (UTC+1, no DST) timestamp
        before 2000 and a UTC timestamp from 2000 onwards. Mixing the two puts a
        one-hour step in the middle of every long series.
        """
        import polars as pl

        if not cfg.extra.get("pre_2000_mez", True):
            return out
        if "_hist" not in Path(raw.path).name:
            return out
        return out.with_columns(
            pl.when(pl.col("time") < _UTC_FROM)
            .then(pl.col("time").dt.offset_by("-1h"))
            .otherwise(pl.col("time"))
            .alias("time")
        )

    def _drop_off_cadence(self, out: pl.DataFrame, cfg: ProductConfig) -> pl.DataFrame:
        """Drop observations that do not sit on the product's cadence grid.

        `Timeline.index_of` returns -1 for an off-grid time. Those rows are
        dropped rather than snapped, so a cadence mismatch shows up as missing
        data in validation instead of a silently shifted series.
        """
        if not cfg.extra.get("drop_off_cadence", True) or out.is_empty():
            return out
        times = out["time"].to_numpy()
        first = out["time"].min()
        last = out["time"].max()
        start = datetime(first.year, first.month, first.day, tzinfo=UTC)
        end = datetime(last.year, last.month, last.day, tzinfo=UTC) + timedelta(days=1)
        idx = Timeline(start, end, cfg.cadence).index_of(times)
        n_bad = int((idx < 0).sum())
        if n_bad:
            logger.warning("dropping %d off-cadence observation(s)", n_bad)
            import polars as pl

            return out.filter(pl.Series(idx >= 0))
        return out

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile], *, sample_rows: int = 200) -> dict:
        """Report every distinct produkt schema across `raw_files`.

        Upstream changes columns without announcing it, so the pipeline pins the
        fingerprint this returns and `parse` refuses to run against anything
        else. The report is deliberately a plain dict: it is meant to be written
        to JSON next to the store and diffed between runs.
        """
        variants: dict[tuple[str, tuple[str, ...]], dict] = {}
        errors: list[dict[str, str]] = []
        for raw in raw_files:
            path = Path(raw.path)
            try:
                group = self._group_for(raw)
                data = _read_produkt(path)
                if data is None:
                    raise ValueError("no produkt_* member")
                header = _header_columns(data)
                product_code = _produkt_code(path)
                dtypes = _infer_dtypes(data, header, sample_rows)
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            columns = tuple(c.strip() for c in header)
            entry = variants.setdefault(
                (group.key, columns),
                {
                    "group": group.key,
                    "product_code": product_code,
                    "columns": list(columns),
                    "dtypes": dtypes,
                    "fingerprint": schema_fingerprint(columns),
                    "count": 0,
                    "examples": [],
                },
            )
            entry["count"] += 1
            if entry["dtypes"] != dtypes:
                entry.setdefault("dtype_variants", [])
                if dtypes not in entry["dtype_variants"]:
                    entry["dtype_variants"].append(dtypes)
            if product_code != entry["product_code"]:
                entry.setdefault("product_code_variants", [])
                if product_code not in entry["product_code_variants"]:
                    entry["product_code_variants"].append(product_code)
            if len(entry["examples"]) < 3:
                entry["examples"].append(path.name)

        ordered = sorted(variants.values(), key=lambda e: (e["group"], e["fingerprint"]))
        by_group: dict[str, list[str]] = {}
        for entry in ordered:
            by_group.setdefault(entry["group"], []).append(entry["fingerprint"])
        single = {g: fps[0] for g, fps in by_group.items() if len(fps) == 1}
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": {g: sorted(fps) for g, fps in by_group.items()},
            "fingerprint": product_fingerprint(single) if len(single) == len(by_group) else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ----------------------------------------------------------------- file helpers


def _read_produkt(path: Path) -> bytes | None:
    """Raw bytes of the single `produkt_*.txt` inside a station ZIP."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if Path(n).name.startswith(_PRODUKT_PREFIX)]
        if not members:
            return None
        return zf.read(min(members))


def _produkt_code(path: Path) -> str:
    """`produkt_zehn_now_tu_...txt` -> `zehn_now_tu`; a dialect marker for scan."""
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if Path(n).name.startswith(_PRODUKT_PREFIX)]
    if not members:
        return ""
    stem = Path(min(members)).stem[len(_PRODUKT_PREFIX) :]
    return "_".join(stem.split("_")[:3])


def _header_columns(data: bytes) -> list[str]:
    line = data.split(b"\n", 1)[0].decode("latin-1").rstrip("\r")
    return line.split(";")


def _infer_dtypes(data: bytes, header: Sequence[str], sample_rows: int) -> dict[str, str]:
    """Best-effort dtype per column from the first `sample_rows` data rows.

    DWD writes every field space-padded, so polars reads the whole file as
    strings; the real dtype has to be inferred after stripping.
    """
    lines = data.split(b"\n")[1 : sample_rows + 1]
    out: dict[str, str] = {}
    cols = [c.strip() for c in header]
    values: dict[str, list[str]] = {c: [] for c in cols}
    for line in lines:
        text = line.decode("latin-1").rstrip("\r")
        if not text.strip():
            continue
        fields = text.split(";")
        for name, field in zip(cols, fields, strict=False):
            values[name].append(field.strip())
    for name in cols:
        out[name] = _dtype_of(values[name])
    return out


def _dtype_of(samples: Sequence[str]) -> str:
    seen = [s for s in samples if s]
    if not seen:
        return "null"
    if all(re.fullmatch(r"[+-]?\d+", s) for s in seen):
        return "i64"
    if all(re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", s) for s in seen):
        return "f64"
    return "str"


def _empty_frame(wanted: Sequence[tuple[str, Variable]]) -> pl.DataFrame:
    import polars as pl

    schema: dict[str, object] = {
        "time": pl.Datetime("us", "UTC"),
        "station_id": pl.String,
    }
    for _, var in wanted:
        schema[var.name] = _polars_dtype(var.dtype)
    return pl.DataFrame(schema=schema)


def _apply_variables(frame: pl.DataFrame, variables: Sequence[Variable]) -> pl.DataFrame:
    """Sentinel -> null, then scale/offset, then valid-range mask, then dtype."""
    import polars as pl

    exprs = []
    for var in variables:
        col = pl.col(var.name)
        col = pl.when(col == MISSING).then(None).otherwise(col)
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
        raise ValueError(f"unsupported dtype {name!r} for a DWD variable") from None
