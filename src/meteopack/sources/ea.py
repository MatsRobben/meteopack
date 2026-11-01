"""UK Environment Agency Hydrology API — quality-controlled 15-minute rainfall.

Keyless REST under the Open Government Licence, two endpoints::

    <base>/id/stations?observedProperty=rainfall     # the catalogue, JSON
    <base>/id/measures/<measure>/readings.csv        # the bulk readings, CSV

One variable, one cadence: tipping-bucket rainfall totals over 15 minutes
(`period=900`), `qualified` — the quality-controlled series rather than the
live telemetry — with an archive back to about 1990.

Four pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store, or the ledger, if you get it wrong:

* **`station_id` is the EA `notation`, never `stationReference`.** Verified
  against the live catalogue on 2026-10-31: all 995 rainfall notations are
  distinct, while `stationReference` repeats ("43204", "E14900") and is absent
  for 81 of them. Keying on the reference would merge two gauges into one
  column and drop the 81 that have none. The notation is a bare GUID for most
  stations and `<guid>_<ref>` for 77 of them; both forms are kept verbatim.
* **A station-year that returns zero bytes is normal, not a failure.** The
  readings endpoint answers HTTP 200 with an empty body for a year before the
  gauge opened and for the recent tail that quality control has not reached
  yet — 364 of 9450 station-years in the existing archive. `list` already skips
  years before `dateOpened`, and `parse` turns what is left into an empty frame
  with an info log. Treating it as an error would fail roughly one job in
  twenty-five for no reason.
* **The readings endpoint caps a response at 100 000 rows.** A year of
  15-minute data is ~35 040 rows, so one request per station-year is both safe
  and efficient (~0.8 s each). `parse` warns if a file ever reaches the cap,
  because the truncation is silent — the CSV simply stops.
* **`quality` is the missing-data flag, not just an annotation.** A row flagged
  `Missing` carries an empty `value`; both become null *before* any scaling.
  `Suspect` and `Unchecked` rows are kept by default and can be dropped through
  `extra.missing_quality`, because what counts as usable is a product decision
  rather than a reader one.

EA publishes no elevation for these gauges, so `stations` emits `height` as a
genuine null rather than a plausible-looking zero: a fabricated height would
propagate into every derived product that uses station elevation.

The upstream schema drifts without announcement, so `scan` reports every column
set seen and `parse` refuses to run against a pinned fingerprint it does not
match.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
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

BASE_URL = "https://environment.data.gov.uk/hydrology"

#: The EA `observedProperty` this source reads. Rainfall only, by design.
PARAMETER = "rainfall"

#: Measure period in seconds. 900 is the 15-minute series; EA also publishes a
#: daily (86400) one, which is a different product and not this one.
PERIOD_SECONDS = 900

#: Hard row cap on a `readings.csv` response. A year is ~35 040 rows.
CSV_ROW_CAP = 100_000

#: Columns of `readings.csv`, as served on 2026-10-31.
READINGS_COLUMNS: tuple[str, ...] = (
    "measure", "dateTime", "date", "value", "completeness", "quality", "qcode",
)

_TIME_COL = "dateTime"
_MEASURE_COL = "measure"
_QUALITY_COL = "quality"

#: Columns that are structural rather than observations; a `source_key` naming
#: one of these is a config mistake, not a variable.
_STRUCTURAL = frozenset({_MEASURE_COL, _TIME_COL, "date", _QUALITY_COL})

#: `quality` values whose rows carry no usable number. "Missing" always comes
#: with an empty `value`; the others are judgements, not absences.
DEFAULT_MISSING_QUALITY: tuple[str, ...] = ("Missing",)

#: How long quality control lags real time. A station-year whose end is still
#: inside this window can be rewritten upstream under the same URL, so it is
#: marked volatile and never skipped on key alone.
QC_LAG_DAYS = 180

_MEASURE_PATH = "/id/measures/"
_MEASURE_SUFFIX = f"-{PARAMETER}-t-{PERIOD_SECONDS}-mm-qualified"

#: EA writes naive timestamps and documents them as UTC/GMT.
_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"


#: A readings CSV's columns do not match the fingerprint pinned in config.
SchemaDriftError = _SchemaDriftError


@dataclass(frozen=True, slots=True)
class StationRecord:
    """One item of an `/id/stations` response, reduced to what we use."""

    station_id: str
    lon: float
    lat: float
    name: str
    measure_id: str
    station_reference: str = ""
    date_opened: date | None = None

    def first_year(self, floor: int) -> int:
        """Earliest year worth requesting, given `dateOpened`.

        Years before the gauge opened answer 200 with an empty body. They are
        harmless but they are also most of a long backfill, so they are not
        emitted at all.
        """
        if self.date_opened is None:
            return floor
        return max(floor, self.date_opened.year)


def station_id(value: str) -> str:
    """Canonical EA station id: the `notation`, stripped and nothing else.

    Deliberately not normalised further. Both the bare-GUID form and the
    `<guid>_<reference>` form occur in the live catalogue, and the two are
    different stations rather than two spellings of one.
    """
    text = str(value).strip()
    if not text or "/" in text:
        raise ValueError(
            f"{value!r} is not an EA station notation; expected the catalogue's "
            f"`notation`, e.g. '34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359'"
        )
    return text


def measure_notation(measure_id: str) -> str | None:
    """Station notation carried by a measure id, or None if it has none.

    A measure id reads `<notation>-rainfall-t-900-mm-qualified`, which makes
    every readings CSV self-identifying — see `parse`.
    """
    text = str(measure_id).strip()
    if _MEASURE_PATH in text:
        text = text.split(_MEASURE_PATH, 1)[1]
    if not text.endswith(_MEASURE_SUFFIX):
        return None
    try:
        return station_id(text[: -len(_MEASURE_SUFFIX)])
    except ValueError:
        return None


def schema_fingerprint(columns: Iterable[str]) -> str:
    """Stable hash of one normalised readings column set.

    Normalisation strips whitespace and quoting artefacts and drops empties, so
    the fingerprint tracks *what* a file contains rather than how it is spaced.
    It is order-independent; a genuine addition, removal or rename does move it.
    """
    normalised = sorted({c.strip().strip('"') for c in columns} - {""})
    return hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()[:16]


def _split_source_key(key: str) -> str:
    """`"value"` -> `"value"`, with a useful error for anything else."""
    column = key.strip()
    if column in _STRUCTURAL:
        raise ValueError(
            f"source_key {key!r} names a structural column; EA variables come from "
            f"the readings value column, i.e. source_key: value"
        )
    if not column:
        raise ValueError("source_key must be a readings.csv column name, e.g. 'value'")
    return column


def variables_for(cfg: ProductConfig) -> list[tuple[str, Variable]]:
    """`[(readings column, variable), ...]` for everything the config wants."""
    return [(_split_source_key(var.source_key), var) for var in cfg.variables]


def parse_stations(payload: Any, *, period: int = PERIOD_SECONDS) -> list[StationRecord]:
    """Parse an `/id/stations` response, keeping gauges with a `period` measure.

    A station without coordinates, without a usable notation or without a
    measure at the wanted period is skipped with a warning rather than raising:
    one malformed item must not take out a 995-station catalogue.
    """
    items = (payload or {}).get("items") or []
    records: list[StationRecord] = []
    for item in items:
        try:
            sid = station_id(item["notation"])
            lon, lat = float(item["long"]), float(item["lat"])
        except (KeyError, TypeError, ValueError):
            logger.warning("skipping unparseable EA station item: %r", str(item)[:120])
            continue
        measure = next(
            (
                str(m.get("@id", ""))
                for m in (item.get("measures") or [])
                if m.get("period") == period
            ),
            "",
        )
        if not measure:
            continue
        records.append(
            StationRecord(
                station_id=sid,
                lon=lon,
                lat=lat,
                name=str(item.get("label") or ""),
                measure_id=measure.split(_MEASURE_PATH, 1)[-1],
                station_reference=str(item.get("stationReference") or ""),
                date_opened=_iso_date(item.get("dateOpened")),
            )
        )
    records.sort(key=lambda r: r.station_id)
    return records


def _iso_date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None


def unit_key(cfg: ProductConfig, sid: str, year: int) -> str:
    """`<source>/<product>/<notation>/<year>`.

    Stable across runs: a station-year is the same request whenever it is made,
    which is what lets the ledger skip the ones already done.
    """
    return f"{cfg.source}/{cfg.product}/{sid}/{year}"


class EaRainfall:
    """EA Hydrology 15-minute rainfall as a `meteopack` source.

    Public API::

        src = EaRainfall(source_cfg, session=..., limiter=...)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain gauges
        src.list(window, product_cfg)    -> list[Unit]        (one per station-year)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == upstream 404)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        EaRainfall.schema_fingerprint(columns) -> str

    `limiter` is whatever `meteopack.core.http` expects; it is passed straight
    through to `fetch_file`. Reading the station catalogue uses `requests`
    directly, because that is one small JSON read rather than an archive
    download and `core.http` only owns the latter.
    """

    name = "ea"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="ea", base_url=BASE_URL)
        self.base_url = (self.source.base_url or BASE_URL).rstrip("/")
        self._session = session
        self._limiter = limiter
        self._now = now
        self._catalogue_cache: dict[str, list[StationRecord]] = {}

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
        """The session, fitted out with this source's headers.

        A source built without one must still work, and one the caller supplied
        must still come back wearing the source's User-Agent: the CLI hands
        every source a bare `requests.Session()`, and returning it untouched is
        how `midas` lost its bearer token on every CLI run.
        """
        from meteopack.sources.common import http_session

        self._session = http_session(self.source, self._session)
        return self._session

    # ------------------------------------------------------------- remote reads

    def _get_json(self, url: str) -> Any:
        """One plain GET, decoded as JSON. Overridable seam for tests."""
        resp = self._requests_session().get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def stations_url(self, *, limit: int = 10000) -> str:
        query = {"observedProperty": PARAMETER, "_limit": int(limit)}
        return f"{self.base_url}/id/stations?{urlencode(query)}"

    def readings_url(self, measure_id: str, year: int) -> str:
        """Bulk readings for one measure over one calendar year.

        The window is per-year because the endpoint caps a response at 100 000
        rows and a year of 15-minute data is comfortably under that.

        `mineq-date`/`maxeq-date` are the inclusive bounds. `min-date` and
        `max-date` are EXCLUSIVE, and using them dropped 1 January and
        31 December from every station-year: 386 of 386 complete year files in
        the archive ran 01-02 to 12-30. (Inherited from the original pipeline.)
        """
        query = {"mineq-date": f"{year:04d}-01-01", "maxeq-date": f"{year:04d}-12-31"}
        return f"{self.base_url}{_MEASURE_PATH}{measure_id}/readings.csv?{urlencode(query)}"

    def catalogue(self, cfg: ProductConfig | None = None) -> list[StationRecord]:
        """Every rainfall gauge with a 15-minute measure (cached per instance)."""
        limit = int((cfg.extra.get("catalogue_limit") if cfg else None) or 10000)
        url = self.stations_url(limit=limit)
        if url in self._catalogue_cache:
            return self._catalogue_cache[url]
        try:
            payload = self._get_json(url)
        except Exception as exc:  # noqa: BLE001 - a failed listing is not fatal
            logger.warning("EA station catalogue fetch failed for %s: %s", url, exc)
            self._catalogue_cache[url] = []
            return []
        records = parse_stations(payload, period=PERIOD_SECONDS)
        self._catalogue_cache[url] = records
        return records

    # ----------------------------------------------------------------- stations

    def stations(self, cfg: ProductConfig) -> pl.DataFrame:
        """In-domain gauges, with coordinates, for `pack` to write beside the data.

        Clipping is the grid's job when `cfg.grid_id` is set; the UK sits
        outside every grid in this repo's registry, so the shipped product
        clips with `extra.bbox` instead. With neither, the whole network is
        kept — a deliberate escape hatch rather than a default.

        `height` is emitted as a null column because EA publishes no elevation
        for these gauges. Inventing one would be worse than having none.
        """
        import polars as pl

        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        bbox = self._bbox(cfg)
        rows = []
        for rec in self.catalogue(cfg):
            if grid is not None and not grid.contains(rec.lon, rec.lat):
                continue
            if bbox is not None and not _in_bbox(bbox, rec.lon, rec.lat):
                continue
            rows.append(
                {
                    "station_id": rec.station_id,
                    "lat": rec.lat,
                    "lon": rec.lon,
                    "height": None,
                    "name": rec.name,
                    "station_reference": rec.station_reference,
                    "date_opened": rec.date_opened,
                }
            )
        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "height": pl.Float64,
            "name": pl.String,
            "station_reference": pl.String,
            "date_opened": pl.Date,
        }
        frame = pl.DataFrame(rows, schema=schema)
        return frame.unique(subset=["station_id"]).sort("station_id")

    @staticmethod
    def _bbox(cfg: ProductConfig) -> tuple[float, float, float, float] | None:
        override = cfg.extra.get("bbox")
        if not override:
            return None
        west, south, east, north = (float(v) for v in override)
        return west, south, east, north

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every in-domain station-year overlapping `window`. No downloads.

        Only the station catalogue is read. Years before a gauge's `dateOpened`
        are not emitted: they answer 200 with an empty body, so they are work
        with a guaranteed empty result.
        """
        start, end = self._window_bounds(window)
        # Validate the window against the product cadence in one place.
        Timeline(start, end, cfg.cadence)

        table = self.stations(cfg)
        keep = set(table["station_id"].to_list())
        only = cfg.extra.get("station_ids")
        if only:
            keep &= {station_id(s) for s in only}
        limit = cfg.extra.get("max_stations")
        if limit:
            keep = set(sorted(keep)[: int(limit)])
        if not keep:
            logger.warning("no in-domain EA gauges for grid %s", cfg.grid_id)
            return []

        # `end` is exclusive, so a window ending exactly at midnight on 1 Jan
        # must not pull in the whole of that year.
        last_year = (end - timedelta(microseconds=1)).year
        volatile_from = self._volatile_from(cfg)
        units: list[Unit] = []
        for rec in self.catalogue(cfg):
            if rec.station_id not in keep:
                continue
            for year in range(rec.first_year(start.year), last_year + 1):
                units.append(
                    Unit(
                        key=unit_key(cfg, rec.station_id, year),
                        url=self.readings_url(rec.measure_id, year),
                        valid_time=datetime(year, 1, 1, tzinfo=UTC),
                        meta={
                            "station_id": rec.station_id,
                            "measure_id": rec.measure_id,
                            "year": str(year),
                            "filename": f"{year}.csv",
                            # Quality control rewrites the recent tail in place
                            # under the same URL. Without this the ledger would
                            # freeze the provisional version of the last months.
                            "volatile": year >= volatile_from,
                        },
                    )
                )
        logger.info(
            "ea: %d units across %d gauge(s) for %d..%d", len(units), len(keep), start.year,
            last_year,
        )
        return units

    def _volatile_from(self, cfg: ProductConfig) -> int:
        """First year whose data can still change upstream under the same key."""
        lag = timedelta(days=float(cfg.extra.get("qc_lag_days", QC_LAG_DAYS)))
        return (self._clock() - lag).year

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Download one station-year CSV. None if upstream has no such measure.

        An empty body is *not* a 404 and is not handled here: EA answers 200
        with zero bytes for a year it has nothing for, which `fetch_file`
        records as a perfectly good zero-byte RawFile and `parse` turns into an
        empty frame. Everything else (retries, `.part` + atomic rename,
        checksum, rate limiting) belongs to `meteopack.core.http`; the import is
        local so tests can monkeypatch it.
        """
        from meteopack.core.http import NotFoundError, fetch_file

        meta = unit.meta or {}
        dest = (
            Path(raw_root)
            / meta.get("station_id", unit.key.rsplit("/", 2)[-2])
            / meta.get("filename", f"{unit.key.rsplit('/', 1)[-1]}.csv")
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
            logger.info("ea: no such measure for %s (404)", unit.key)
            return None

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One station-year CSV -> tidy frame of `time`, `station_id`, values.

        Rows flagged by `quality` and rows with an empty `value` become null
        *before* scale/offset, so a flag can never be scaled into a
        plausible-looking number, and the valid range is applied afterwards, so
        it is expressed in canonical units.

        A zero-byte file is a legitimate answer — a year before the gauge
        opened, or the tail quality control has not reached — and returns an
        empty frame rather than raising.
        """
        import polars as pl

        wanted = variables_for(cfg)
        data = Path(raw.path).read_bytes()
        if not data.strip():
            logger.info("ea: %s is empty (pre-opening year or QC lag); skipping", raw.path.name)
            return _empty_frame(wanted)

        header = _header_columns(data)
        self._check_fingerprint(header, cfg, Path(raw.path).name)
        if not wanted:
            return _empty_frame(wanted)

        present = set(header)
        missing = [col for col, _ in wanted if col not in present]
        if missing:
            raise SchemaDriftError(
                f"{Path(raw.path).name}: readings CSV is missing column(s) {missing}; "
                f"present: {sorted(present)}"
            )
        if _TIME_COL not in present:
            raise SchemaDriftError(f"{Path(raw.path).name}: no {_TIME_COL} column")

        select = [c for c in (_TIME_COL, _MEASURE_COL, _QUALITY_COL) if c in present]
        select += [col for col, _ in wanted if col not in select]
        frame = pl.read_csv(io.BytesIO(data), infer_schema=False, columns=select)
        if frame.height >= CSV_ROW_CAP:
            logger.warning(
                "ea: %s has %d rows and the endpoint caps at %d; the year may be truncated",
                raw.path.name, frame.height, CSV_ROW_CAP,
            )

        sid = self._station_of(frame, raw)
        bad = _missing_quality(cfg)
        quality = (
            pl.col(_QUALITY_COL).str.strip_chars() if _QUALITY_COL in present else pl.lit("")
        )
        out = frame.select(
            pl.col(_TIME_COL)
            .str.strip_chars()
            .str.strptime(pl.Datetime("us"), _TIME_FORMAT, strict=False)
            .dt.replace_time_zone("UTC")
            .alias("time"),
            pl.lit(sid, dtype=pl.String).alias("station_id"),
            *[
                # Missing first: a flagged or blank reading must never reach the
                # cast, let alone the scale.
                pl.when(quality.is_in(bad) | (pl.col(col).str.strip_chars() == ""))
                .then(None)
                .otherwise(pl.col(col).str.strip_chars())
                .cast(pl.Float64, strict=False)
                .alias(var.name)
                for col, var in wanted
            ],
        ).drop_nulls("time")

        out = _apply_variables(out, [var for _, var in wanted], cfg)
        out = out.unique(subset=["time", "station_id"], keep="last").sort("time")
        return self._drop_off_cadence(out, cfg)

    def _station_of(self, frame: pl.DataFrame, raw: RawFile) -> str:
        """Notation from the file's own `measure` column, else from the unit key.

        Preferring the file means a CSV found on disk parses correctly whatever
        it is called, which is how `scan` and re-parses actually reach it.
        """
        if _MEASURE_COL in frame.columns and frame.height:
            found = measure_notation(str(frame[_MEASURE_COL][0]))
            if found:
                return found
        parts = (raw.unit_key or "").split("/")
        if len(parts) >= 2:
            try:
                return station_id(parts[-2])
            except ValueError:
                pass
        raise SchemaDriftError(
            f"{Path(raw.path).name}: no usable {_MEASURE_COL} column and the unit key "
            f"{raw.unit_key!r} carries no station notation"
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
                f"{filename}: EA readings schema changed: pinned {pinned}, got {seen} "
                f"(columns {list(header)}). Re-run `scan` and re-pin schema_fingerprint "
                f"after checking what moved."
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

        times = out["time"].to_numpy()
        first, last = out["time"].min(), out["time"].max()
        start = datetime(first.year, first.month, first.day, tzinfo=UTC)
        end = datetime(last.year, last.month, last.day, tzinfo=UTC) + timedelta(days=1)
        idx = Timeline(start, end, cfg.cadence).index_of(times)
        n_bad = int((idx < 0).sum())
        if n_bad:
            logger.warning("ea: dropping %d off-cadence observation(s)", n_bad)
            return out.filter(pl.Series(idx >= 0))
        return out

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile], *, sample_rows: int = 200) -> dict:
        """Report every distinct readings schema across `raw_files`.

        Empty files are counted rather than reported as errors, because that is
        what most of an EA backfill's "failures" turn out to be. The report is
        deliberately a plain dict: it is meant to be written to JSON next to the
        store and diffed between runs.
        """
        variants: dict[tuple[str, ...], dict] = {}
        errors: list[dict[str, str]] = []
        n_empty = 0
        for raw in raw_files:
            path = Path(raw.path)
            try:
                data = path.read_bytes()
                if not data.strip():
                    n_empty += 1
                    continue
                header = tuple(_header_columns(data))
                qualities = _sampled_qualities(data, header, sample_rows)
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            entry = variants.setdefault(
                header,
                {
                    "columns": list(header),
                    "fingerprint": schema_fingerprint(header),
                    "qualities": [],
                    "count": 0,
                    "examples": [],
                },
            )
            entry["count"] += 1
            for value in qualities:
                if value not in entry["qualities"]:
                    entry["qualities"].append(value)
            if len(entry["examples"]) < 3:
                entry["examples"].append(path.name)

        ordered = sorted(variants.values(), key=lambda e: e["fingerprint"])
        for entry in ordered:
            entry["qualities"].sort()
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_empty": n_empty,
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": [e["fingerprint"] for e in ordered],
            "fingerprint": ordered[0]["fingerprint"] if len(ordered) == 1 else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ----------------------------------------------------------------- file helpers


def _header_columns(data: bytes) -> list[str]:
    """Column names of a readings CSV. EA quotes every field, including these."""
    line = data.split(b"\n", 1)[0].decode("utf-8-sig").rstrip("\r")
    return next(csv.reader([line]), [])


def _sampled_qualities(data: bytes, header: Sequence[str], sample_rows: int) -> list[str]:
    """Distinct `quality` values in the first `sample_rows` rows, for `scan`."""
    if _QUALITY_COL not in header:
        return []
    index = list(header).index(_QUALITY_COL)
    text = data.decode("utf-8-sig", errors="replace")
    rows = csv.reader(io.StringIO(text))
    next(rows, None)
    seen: list[str] = []
    for n, row in enumerate(rows):
        if n >= sample_rows:
            break
        if index < len(row) and row[index] and row[index] not in seen:
            seen.append(row[index])
    return seen


def _in_bbox(bbox: tuple[float, float, float, float], lon: float, lat: float) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def _missing_quality(cfg: ProductConfig) -> list[str]:
    values = cfg.extra.get("missing_quality")
    if values is None:
        return list(DEFAULT_MISSING_QUALITY)
    return [str(v) for v in values]


def _empty_frame(wanted: Sequence[tuple[str, Variable]]) -> pl.DataFrame:
    import polars as pl

    schema: dict[str, Any] = {"time": pl.Datetime("us", "UTC"), "station_id": pl.String}
    for _, var in wanted:
        schema[var.name] = _polars_dtype(var.dtype)
    return pl.DataFrame(schema=schema)


def _apply_variables(
    frame: pl.DataFrame, variables: Sequence[Variable], cfg: ProductConfig
) -> pl.DataFrame:
    """Sentinel -> null, then scale/offset, then valid-range mask, then dtype.

    Quality flags and blanks are already null by this point. `extra.missing_sentinels`
    exists for the day EA pushes a numeric sentinel through instead; applying it
    before the scale is what keeps a -999 from becoming a believable total.
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
        raise ValueError(f"unsupported dtype {name!r} for an EA variable") from None
