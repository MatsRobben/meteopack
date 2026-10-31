"""RMI Belgium automatic weather stations, served by opendata.meteo.be (OGC WFS).

The Royal Meteorological Institute publishes its open data as a keyless GeoServer
WFS 2.0 endpoint. Two layers matter here::

    <base>?service=WFS&request=GetFeature&typeNames=aws:aws_station  -> GeoJSON
    <base>?service=WFS&request=GetFeature&typeNames=aws:aws_10min    -> CSV

The `aws_10min` layer is ~15 M features, so it is never asked for whole: a unit
is **one station-year**, selected with a CQL filter
(`code=<id> AND timestamp DURING <year>`) and returned as CSV.

Five pieces of domain knowledge are worth stating explicitly, because each one
has already cost something here:

* **This server fails on CONCURRENCY, not on size.** A single-threaded yearly
  export is reliable and takes a couple of seconds; several at once return 500,
  and sustained load gets our whole egress IP rate-limited for tens of minutes —
  which on a shared cluster takes out everyone's downloads, not just this job.
  `configs/sources/belgium.yaml` therefore pins
  `rate_limit: {requests_per_second: 1, max_workers: 1}`. That is not politeness,
  it is the only setting observed to work. Fourteen stations x ~25 years is ~350
  requests, so one per second is also perfectly sufficient.
* **The data layer's geometry is lon/lat swapped.** `aws_station` returns proper
  GeoJSON (`[3.122, 50.904]` for Beitem), while a CSV row from `aws_10min` reads
  `POINT (50.904 3.122)`. Coordinates come from the station layer only; the
  `the_geom` column is ignored, and the station layer is what `stations`
  publishes.
* **RMI data is RAW, not quality-controlled.** Nonphysical values occur —
  pressures near 500 hPa, humidity over 100 %, negative radiation — so
  `valid_min`/`valid_max` in the variable map are doing real work here rather
  than guarding against a corrupt download. The per-variable `qc_flags` column
  is a JSON blob of `validated` booleans; it is deliberately not packed, since a
  flag that is false means "not yet checked", not "wrong".
* **Station codes are not fixed width.** Nine of the fourteen are five-digit WMO
  codes (`06447` Uccle), but De Haan is `10101002`. Zero-padding them the way
  DWD's are padded would invent a station.
* **"Nothing there" is a header-only CSV with HTTP 200**, not a 404 — verified
  live against a nonexistent station code. A WFS *error*, by contrast, arrives
  as an `ows:ExceptionReport` XML body (HTTP 400 for a bad request, 500 under
  load). `parse` refuses an XML body loudly rather than reading it as a CSV with
  one odd column.

The upstream schema drifts without announcement, so `scan` reports every column
set seen and `parse` refuses to run against a pinned fingerprint it does not
match.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlencode

from meteopack.core import grids
from meteopack.core.config import ProductConfig, RateLimit, SourceConfig, Variable
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, TimeWindow, Unit
from meteopack.sources.common import SchemaDriftError as _SchemaDriftError

if TYPE_CHECKING:  # keep polars off the import path of `meteopack.sources`
    import polars as pl

logger = logging.getLogger(__name__)

BASE_URL = "https://opendata.meteo.be/service/ows"

#: The 10-minute automatic-weather-station product and its station catalogue.
DATA_LAYER = "aws:aws_10min"
STATION_LAYER = "aws:aws_station"

#: Columns that are per-request plumbing rather than observations. `FID` is a
#: GeoServer feature id that changes between requests, and `the_geom` is the
#: lon/lat-swapped duplicate of the station layer — see the module docstring.
STRUCTURAL: frozenset[str] = frozenset({"FID", "the_geom"})

_CODE_COL = "code"
_TIME_COL = "timestamp"

#: A station code as RMI writes it: digits, but not a fixed width.
_CODE_RE = re.compile(r"^\d{1,12}$")

#: The first bytes of an `ows:ExceptionReport`, which some WFS deployments
#: happily return with a 200.
_XML_PREFIXES = (b"<?xml", b"<ows:", b"<ExceptionReport", b"<html")


#: A CSV export's columns, or its body, are not what this reader expects.
SchemaDriftError = _SchemaDriftError


# -------------------------------------------------------------------- naming


def station_id(value: str | int) -> str:
    """Canonical RMI station id: the code, as a string, unpadded.

    Deliberately *not* zero-padded to five like DWD's: De Haan is `10101002`,
    so a fixed width would either truncate it or invent leading zeros on the
    others and break the join against the station catalogue.
    """
    text = str(value).strip()
    if not _CODE_RE.match(text):
        raise ValueError(f"{value!r} is not an RMI station code; expected digits, e.g. '6447'")
    return text


def schema_fingerprint(columns: Iterable[str]) -> str:
    """Stable hash of one normalised CSV column set.

    Normalisation strips whitespace and drops the structural columns, so the
    fingerprint tracks *what the export contains* rather than GeoServer's
    per-request plumbing. It is order-independent (WFS is free to reorder
    columns), and a genuine addition, removal or rename does move it.
    """
    normalised = sorted({c.strip() for c in columns} - STRUCTURAL - {""})
    return hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()[:16]


def variables_by_column(cfg: ProductConfig) -> list[tuple[str, Variable]]:
    """`[(CSV column, canonical Variable), ...]` for everything the config wants."""
    out: list[tuple[str, Variable]] = []
    for var in cfg.variables:
        column = var.source_key.strip()
        if not column:
            raise ValueError(f"variable {var.name!r} has an empty source_key")
        if column in STRUCTURAL or column in (_CODE_COL, _TIME_COL):
            raise ValueError(
                f"source_key {column!r} is a structural column, not an observation"
            )
        out.append((column, var))
    return out


def unit_key(cfg: ProductConfig, station: str, year: int) -> str:
    """`<source>/<product>/<station>/<year>`.

    Stable because a unit is always a *whole* calendar year, never the slice of
    one that a particular run happened to ask for: a key whose content depended
    on the request window would be marked done by a narrow backfill and skipped
    by the wide one that followed.
    """
    return f"{cfg.source}/{cfg.product}/{station}/{year}"


# ------------------------------------------------------------------- stations


@dataclass(frozen=True, slots=True)
class StationRecord:
    """One feature of the `aws:aws_station` GeoJSON layer."""

    station_id: str
    lon: float
    lat: float
    height: float
    name: str
    wigos_id: str
    date_begin: date | None = None
    date_end: date | None = None

    def covers(self, start: datetime, end: datetime) -> bool:
        """True if the station's operating period overlaps the half-open window.

        `date_end` is null for a station still reporting. Honouring this is what
        keeps a 2003-2026 backfill from asking De Haan (commissioned 2025) for
        twenty-two years of header-only CSVs, at one request per second.
        """
        if self.date_begin is not None and self.date_begin >= end.date():
            return False
        return not (self.date_end is not None and self.date_end < start.date())


def _as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        # 3.11+ fromisoformat accepts the trailing 'Z' the layer actually writes.
        return datetime.fromisoformat(str(value)).date()
    except ValueError:
        logger.warning("unparseable station date %r", value)
        return None


def parse_stations(payload: Any) -> list[StationRecord]:
    """Parse the `aws:aws_station` GeoJSON FeatureCollection.

    A feature without a usable code or a finite point geometry is skipped with a
    warning rather than raising: one malformed feature must not take out the
    catalogue. A repeated code keeps the first occurrence, which the synoptic
    sibling layer is known to need.
    """
    features = (payload or {}).get("features") or []
    records: list[StationRecord] = []
    seen: set[str] = set()
    for feature in features:
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        try:
            sid = station_id(props["code"])
            lon, lat = float(coords[0]), float(coords[1])
        except (KeyError, TypeError, ValueError, IndexError):
            logger.warning("skipping unparseable station feature: %r", str(feature)[:120])
            continue
        if not (math.isfinite(lon) and math.isfinite(lat)):
            logger.warning("skipping station %s with non-finite coordinates", sid)
            continue
        if sid in seen:
            continue
        seen.add(sid)
        altitude = props.get("altitude")
        records.append(
            StationRecord(
                station_id=sid,
                lon=lon,
                lat=lat,
                height=float(altitude) if altitude is not None else float("nan"),
                name=str(props.get("name") or ""),
                wigos_id=str(props.get("wigos_id") or ""),
                date_begin=_as_date(props.get("date_begin")),
                date_end=_as_date(props.get("date_end")),
            )
        )
    return records


# ------------------------------------------------------------------ the source


class RmiAws:
    """RMI Belgium 10-minute AWS observations as a `meteopack` source.

    Public API::

        src = RmiAws(source_cfg, session=..., limiter=...)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain stations
        src.list(window, product_cfg)    -> list[Unit]        (one per station-YEAR)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == upstream 404)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        RmiAws.schema_fingerprint(columns) -> str

    `limiter` is whatever `meteopack.core.http` expects and is passed straight
    through to `fetch_file`. Reading the station catalogue uses `requests`
    directly, because that is a 4 kB JSON read rather than an export download
    and `core.http` only owns the latter.
    """

    name = "belgium"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="belgium", base_url=BASE_URL)
        self.base_url = (self.source.base_url or BASE_URL).rstrip("/")
        self.data_layer = str(self.source.extra.get("data_layer", DATA_LAYER))
        self.station_layer = str(self.source.extra.get("station_layer", STATION_LAYER))
        self._session = session
        self._limiter = limiter
        self._now = now
        self._station_cache: dict[str, list[StationRecord]] = {}

    # ------------------------------------------------------------------ config

    @property
    def rate_limit(self) -> RateLimit:
        return self.source.rate_limit

    @property
    def timeout(self) -> float:
        # A whole station-year export is a few MB and the server is unhurried;
        # a short timeout here just turns a slow success into a retry, which is
        # precisely the load this endpoint cannot take.
        return float(self.source.extra.get("timeout_seconds", 180))

    def _clock(self) -> datetime:
        return self._now or datetime.now(UTC)

    def _requests_session(self):
        """The session, built on first use.

        A source constructed without one must still work: passing `self._session`
        straight through when it is None raises deep inside `fetch_file`, which
        has happened here before.
        """
        if self._session is None:
            import requests

            self._session = requests.Session()
            agent = self.source.extra.get("user_agent", "meteopack/0.1 (+rmi-opendata)")
            self._session.headers["User-Agent"] = str(agent)
        return self._session

    # ------------------------------------------------------------- remote reads

    def _get_json(self, url: str) -> Any:
        """One plain GET, decoded as JSON. Overridable seam for tests."""
        resp = self._requests_session().get(url, timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def station_url(self) -> str:
        query = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.station_layer,
            "outputFormat": "application/json",
            "count": "1000",
        }
        return f"{self.base_url}?{urlencode(query, safe=',:/')}"

    def data_url(self, station: str, start: datetime, end: datetime) -> str:
        """CSV export for one station over one closed interval.

        `DURING` is inclusive at both ends, so a year's interval stops one
        second short of the next year's start; otherwise midnight on 1 January
        would land in two units and be written twice.
        """
        cql = (
            f"{_CODE_COL}={station} AND {_TIME_COL} DURING "
            f"{_instant(start)}/{_instant(end - timedelta(seconds=1))}"
        )
        query = {
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": self.data_layer,
            "outputFormat": "csv",
            "sortBy": _TIME_COL,
            "CQL_FILTER": cql,
        }
        return f"{self.base_url}?{urlencode(query, safe=',:/')}"

    def station_records(self) -> list[StationRecord]:
        """The station catalogue (cached per instance)."""
        url = self.station_url()
        if url in self._station_cache:
            return self._station_cache[url]
        try:
            payload = self._get_json(url)
        except Exception as exc:  # noqa: BLE001 - a failed listing is not fatal
            logger.warning("rmi station layer fetch failed: %s", exc)
            self._station_cache[url] = []
            return []
        if payload is None:
            logger.info("belgium: station layer returned 404")
            self._station_cache[url] = []
            return []
        records = parse_stations(payload)
        self._station_cache[url] = records
        return records

    # ----------------------------------------------------------------- stations

    def stations(
        self, cfg: ProductConfig, *, window: TimeWindow | None = None
    ) -> pl.DataFrame:
        """In-domain stations from the catalogue layer.

        Clipping is the grid's job: a station is kept iff
        `grids.get(cfg.grid_id).contains(lon, lat)`. With no `grid_id` the whole
        Belgian network is kept, which is a deliberate escape hatch rather than a
        default — every product config here sets one.
        """
        import polars as pl

        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        start, end = self._window_bounds(window) if window else (None, None)

        rows = []
        for rec in self.station_records():
            if grid is not None and not grid.contains(rec.lon, rec.lat):
                continue
            if start is not None and not rec.covers(start, end):
                continue
            rows.append(
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
            )
        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "height": pl.Float64,
            "name": pl.String,
            "wigos_id": pl.String,
        }
        return pl.DataFrame(rows, schema=schema).sort("station_id")

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every (in-domain station, calendar year) the window touches.

        No observations are downloaded; only the station catalogue is read. A
        unit is a whole year even when the window covers part of it — see
        `unit_key` for why the key must not depend on the request window.
        """
        start, end = self._window_bounds(window)
        # Validate the window against the product cadence in one place.
        Timeline(start, end, cfg.cadence)

        table = self.stations(cfg, window=window)
        keep = table["station_id"].to_list()
        only = cfg.extra.get("station_ids")
        if only:
            wanted = {station_id(s) for s in only}
            keep = [s for s in keep if s in wanted]
        limit = cfg.extra.get("max_stations")
        if limit:
            keep = keep[: int(limit)]
        if not keep:
            logger.warning("no in-domain RMI stations for grid %s", cfg.grid_id)
            return []
        by_id = {rec.station_id: rec for rec in self.station_records()}

        # Half-open: a window ending exactly at midnight on 1 January does not
        # want that year.
        last = (end - timedelta(microseconds=1)).year
        years = range(start.year, last + 1)
        this_year = self._clock().year

        units: list[Unit] = []
        for sid in keep:
            record = by_id.get(sid)
            for year in years:
                year_start = datetime(year, 1, 1, tzinfo=UTC)
                year_end = datetime(year + 1, 1, 1, tzinfo=UTC)
                if record is not None and not record.covers(year_start, year_end):
                    continue
                units.append(
                    Unit(
                        key=unit_key(cfg, sid, year),
                        url=self.data_url(sid, year_start, year_end),
                        valid_time=year_start,
                        meta={
                            "station_id": sid,
                            "year": str(year),
                            "filename": f"{year}.csv",
                            # The current year is still being written: its key
                            # is deliberately stable, so without this the ledger
                            # would mark January's harvest as the whole year and
                            # never collect the rest.
                            "volatile": year >= this_year,
                        },
                    )
                )
        logger.info(
            "belgium: %d station-year unit(s) across %d station(s) for %d..%d",
            len(units), len(keep), start.year, last,
        )
        return units

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Download one station-year CSV. Returns None if upstream has no such file.

        Everything else (retries, `.part` + atomic rename, checksum, rate
        limiting) belongs to `meteopack.core.http`; the import is local so tests
        can monkeypatch it. Note that *empty* is not 404 here: a station-year
        with no observations comes back as a header-only CSV with HTTP 200, and
        `parse` turns that into an empty frame.
        """
        from meteopack.core.http import NotFoundError, fetch_file

        meta = unit.meta or {}
        dest = Path(raw_root) / meta.get("station_id", "unknown") / meta.get(
            "filename", f"{unit.key.rsplit('/', 1)[-1]}.csv"
        )
        rl = self.rate_limit
        try:
            return fetch_file(
                unit.url,
                dest,
                unit_key=unit.key,
                # `_requests_session` rather than `self._session`, so a source
                # constructed without one still downloads instead of raising
                # AttributeError on None deep inside `fetch_file`.
                session=self._requests_session(),
                limiter=self._limiter,
                retries=rl.max_retries,
                backoff=rl.backoff_seconds,
                timeout=self.timeout,
            )
        except NotFoundError:
            logger.info("belgium: no export for %s (404)", unit.key)
            return None

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One station-year CSV -> tidy frame of `time`, `station_id`, values.

        Only the columns `cfg.variables` asks for are read. An empty field —
        which is how RMI writes missing — and any `extra.missing_sentinels`
        become null *before* scale/offset, so a sentinel can never be scaled into
        a plausible-looking number, and the valid range is applied afterwards, so
        it is expressed in canonical units. Those ranges matter more here than
        anywhere else in this package: RMI publishes un-QC'd data.
        """
        import polars as pl

        wanted = variables_by_column(cfg)
        data = Path(raw.path).read_bytes()
        header = _header_columns(data, Path(raw.path).name)
        self._check_fingerprint(header, cfg, Path(raw.path).name)
        if not wanted:
            return _empty_frame(wanted)

        by_stripped = {c.strip(): c for c in header}
        for structural in (_CODE_COL, _TIME_COL):
            if structural not in by_stripped:
                raise SchemaDriftError(f"{Path(raw.path).name}: no {structural} column")
        missing = [col for col, _ in wanted if col not in by_stripped]
        if missing:
            raise SchemaDriftError(
                f"{Path(raw.path).name}: missing column(s) {missing}; "
                f"present: {sorted(set(by_stripped) - STRUCTURAL)}"
            )

        select = [by_stripped[_CODE_COL], by_stripped[_TIME_COL]]
        select += [by_stripped[col] for col, _ in wanted]
        # `infer_schema=False` keeps every field a string, so empty-means-missing
        # is decided here rather than by polars' type inference, which differs
        # between a column that is empty everywhere and one that is not.
        frame = pl.read_csv(io.BytesIO(data), infer_schema=False, columns=select)
        if frame.is_empty():
            # Header-only: a legitimate answer for a station-year before the
            # station existed. Not an error, and not a 404 either.
            return _empty_frame(wanted)

        out = frame.select(
            pl.col(by_stripped[_TIME_COL])
            .str.strip_chars()
            .str.strip_chars_end("Z")
            .str.strptime(pl.Datetime("us"), "%Y-%m-%dT%H:%M:%S", strict=False)
            .dt.replace_time_zone("UTC")
            .alias("time"),
            pl.col(by_stripped[_CODE_COL]).str.strip_chars().alias("station_id"),
            *[
                _blank_to_null(pl.col(by_stripped[col]))
                .cast(pl.Float64, strict=False)
                .alias(var.name)
                for col, var in wanted
            ],
        ).drop_nulls("time")

        out = _apply_variables(out, [var for _, var in wanted], cfg)
        out = self._drop_off_cadence(out, cfg)
        return out.unique(subset=["time", "station_id"], keep="last").sort(
            ["time", "station_id"]
        )

    def _check_fingerprint(
        self, header: Sequence[str], cfg: ProductConfig, filename: str
    ) -> None:
        pinned = cfg.schema_fingerprint
        if not pinned:
            return
        seen = schema_fingerprint(header)
        if seen != pinned:
            raise SchemaDriftError(
                f"{filename}: RMI {self.data_layer} schema changed: pinned {pinned}, got "
                f"{seen} (columns {sorted({c.strip() for c in header} - STRUCTURAL)}). "
                f"Re-run `scan` and re-pin schema_fingerprint after checking what moved."
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
            logger.warning("belgium: dropping %d off-cadence observation(s)", n_bad)
            return out.filter(pl.Series(idx >= 0))
        return out

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile], *, sample_rows: int = 200) -> dict:
        """Report every distinct CSV schema across `raw_files`.

        Upstream changes columns without announcing it, so the pipeline pins the
        fingerprint this returns and `parse` refuses to run against anything
        else. The report is deliberately a plain dict: it is meant to be written
        to JSON next to the store and diffed between runs.
        """
        variants: dict[tuple[str, ...], dict] = {}
        errors: list[dict[str, str]] = []
        for raw in raw_files:
            path = Path(raw.path)
            try:
                data = path.read_bytes()
                header = _header_columns(data, path.name)
                dtypes = _infer_dtypes(data, header, sample_rows)
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            columns = tuple(c.strip() for c in header)
            entry = variants.setdefault(
                columns,
                {
                    "layer": self.data_layer,
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
            if len(entry["examples"]) < 3:
                entry["examples"].append(path.name)

        ordered = sorted(variants.values(), key=lambda e: e["fingerprint"])
        fingerprints = [e["fingerprint"] for e in ordered]
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": sorted(fingerprints),
            "fingerprint": fingerprints[0] if len(fingerprints) == 1 else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ----------------------------------------------------------------- csv helpers


def _instant(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _header_columns(data: bytes, filename: str) -> list[str]:
    """The CSV header, refusing an XML error body.

    GeoServer answers a bad or overloaded request with an `ows:ExceptionReport`.
    Read as a CSV that is a single-column file whose one row happens to be XML,
    which parses cleanly and produces an empty store; this says so instead.
    """
    head = data.lstrip()[:64]
    if head.startswith(_XML_PREFIXES):
        snippet = data.decode("utf-8", "replace")[:200].replace("\n", " ")
        raise SchemaDriftError(
            f"{filename}: body is a WFS error document, not a CSV export: {snippet}"
        )
    if not data.strip():
        raise SchemaDriftError(f"{filename}: empty body")
    line = data.split(b"\n", 1)[0].decode("utf-8", "replace").rstrip("\r")
    return line.split(",")


def _infer_dtypes(data: bytes, header: Sequence[str], sample_rows: int) -> dict[str, str]:
    """Best-effort dtype per column from the first `sample_rows` data rows.

    Deliberately naive about quoting: `qc_flags` is a JSON blob full of commas,
    so it is the one column this can misread, and it is never packed.
    """
    cols = [c.strip() for c in header]
    values: dict[str, list[str]] = {c: [] for c in cols}
    for line in data.split(b"\n")[1 : sample_rows + 1]:
        text = line.decode("utf-8", "replace").rstrip("\r")
        if not text.strip():
            continue
        for name, field in zip(cols, text.split(","), strict=False):
            values[name].append(field.strip())
    return {name: _dtype_of(values[name]) for name in cols}


def _dtype_of(samples: Sequence[str]) -> str:
    seen = [s for s in samples if s]
    if not seen:
        return "null"
    if all(re.fullmatch(r"[+-]?\d+", s) for s in seen):
        return "i64"
    if all(re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?", s) for s in seen):
        return "f64"
    return "str"


def _blank_to_null(col):
    """RMI writes missing as an empty field; make that a null before anything else."""
    import polars as pl

    stripped = col.str.strip_chars()
    return pl.when(stripped.str.len_chars() == 0).then(None).otherwise(stripped)


def _qc_flags(text: str) -> dict[str, bool]:
    """Decode one `qc_flags` cell. Not packed; here for ad-hoc inspection.

    A false flag means "not validated", not "wrong", which is why it is not a
    mask: dropping unvalidated values would quietly delete most of the recent
    tail of every series.
    """
    try:
        return dict((json.loads(text) or {}).get("validated") or {})
    except (TypeError, ValueError):
        return {}


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
    """Sentinel -> null, then scale/offset, then valid-range mask, then dtype.

    Missing is already null by the time this runs (`_blank_to_null`).
    `extra.missing_sentinels` exists for the numeric sentinels RMI occasionally
    pushes through; a -999 that gets scaled first is indistinguishable from data.
    """
    import polars as pl

    sentinels = [float(v) for v in (cfg.extra.get("missing_sentinels") or [])]
    exprs = []
    for var in variables:
        col = pl.col(var.name)
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
        raise ValueError(f"unsupported dtype {name!r} for an RMI variable") from None
