"""EUMETNET E-SOH surface observations, served by MeteoGate.

MeteoGate publishes the pan-European real-time in-situ feed as an OGC API -
Environmental Data Retrieval (EDR) service. It is keyless, returns JSON, and
answers three endpoints that matter here::

    <base>/collections
    <base>/collections/observations/locations?bbox=&datetime=&standard_name=
    <base>/collections/observations/locations/<wigos id>?...&f=CoverageJSON

Five pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store, or the ledger, if you get it wrong:

* **The collection is a rolling ~24 hour window. There is no archive.** The
  live `/collections` document advertises exactly one temporal interval; on
  2026-10-31 it read `2026-10-30T19:59:48Z/2026-10-31T20:00:00Z`. Anything
  older is gone and cannot be re-requested at any price. `list` is therefore
  built around harvesting *the window that exists right now*: a request reaching
  further back is clamped to the oldest instant still served, with a warning,
  rather than quietly returning units that will all 404. The operational
  consequence is that this source must be run on a schedule that overlaps
  itself — miss a day and that day does not exist.
* **A unit key must encode the harvest window, not just the station.** Unit keys
  are the job ledger's notion of identity. A station-only key would be marked
  done on the first harvest, and every later harvest of the same station would
  be skipped — which, for a feed with no archive, loses the data permanently.
  Keys here are `<source>/<product>/<param digest>/<slot>/<wigos id>`, where the
  slot is a fixed grid of harvest buckets (`extra.harvest_slot_seconds`, default
  one hour). Flooring onto that grid is what makes a key *stable*: re-running
  the same job, or a retry straddling a minute boundary, produces the same key,
  so retries stay idempotent. Advancing time is what makes it *distinct*: the
  next harvest covers later slots and therefore later keys.
* **Cadence is heterogeneous, per station and per parameter.** Nothing here
  assumes ten minutes. A document mixing cadences comes back as a
  `CoverageCollection` with one `Coverage` *per parameter*, each carrying its
  own `t` axis — verified live on 2026-10-31, where one station returned
  hourly air temperature, 3-hourly, 6-hourly and 12-hourly precipitation sums as
  four separate coverages. A single-cadence response is a bare `Coverage`
  instead. `parse` handles both and outer-joins on `(time, station_id)`, so a
  parameter simply has nulls at times it does not report.
* **`level` in a parameter name is usually not the sensor height.** Parameter
  names read `<standard_name>:<level>:<method>:<period>`, but for many feeds the
  level carries the station elevation: the live catalogue lists 1176 distinct
  levels for `air_pressure:*:point:PT0S`, including `1003.0` and `1029.0`.
  Pinning a level in config would therefore match a handful of stations by
  accident. `source_key` accepts `*` for the level (and the product config uses
  it throughout); `method` and `period` are never wildcards, because those are
  what the number *means*.
* **An empty locations query is a 404, not an empty collection.** A bbox with no
  stations answers `404 {"detail": "Query did not return any locations."}`. That
  is a normal answer to a narrow footprint and must not fail a job.

Stations are keyed by WIGOS identifier (`0-20000-0-06260`, `0-276-0-10015`),
which is the natural `station_id` and the join key against every other source
here. Unlike DWD's numeric ids it is *not* normalised by zero-padding: the local
identifier is a string by specification, and `06260` is a different station from
`6260`.

The upstream parameter set drifts without announcement, so `scan` reports every
measurement set it sees plus their union, and `parse` refuses to run against a
document reporting anything outside the catalogue pinned in config. The pinned
thing is the catalogue rather than a per-file fingerprint because no two
stations here carry the same parameter set — see `_check_schema`.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import quote, urlencode

from meteopack.core import grids
from meteopack.core.config import ProductConfig, RateLimit, SourceConfig, Variable
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, TimeWindow, Unit

if TYPE_CHECKING:  # keep polars off the import path of `meteopack.sources`
    import polars as pl

logger = logging.getLogger(__name__)

BASE_URL = "https://observations.meteogate.eu"

#: The one collection this source reads. MeteoGate serves others, but the
#: surface in-situ feed is the whole point of E-SOH.
COLLECTION = "observations"

#: How far back the rolling window reaches, per the collection's own temporal
#: extent. Overridable via `extra.rolling_window_hours` in case it is widened.
ROLLING_WINDOW_HOURS = 24

#: One harvest bucket. An hour keeps a single request small and a single ledger
#: entry meaningful, while still being coarse enough that a 24 hour catch-up is
#: 24 units per station rather than 144.
DEFAULT_SLOT_SECONDS = 3600

#: Wildcard accepted in the `level` field of a `source_key`.
LEVEL_ANY = "*"

#: `<series>-<issuer>-<issue number>-<local id>`, the WIGOS station identifier.
#: The local identifier is alphanumeric plus `-` and `_`, at most 16 characters,
#: and may itself contain dashes — so the id cannot simply be split on "-".
_WIGOS_RE = re.compile(r"^[0-9]+-[0-9]+-[0-9]+-[-_A-Za-z0-9]{1,16}$")

_COVERAGE = "Coverage"
_COVERAGE_COLLECTION = "CoverageCollection"
_WIGOS_KEY = "metocean:wigosId"
_LEVEL_TOL = 1e-9


class SchemaDriftError(RuntimeError):
    """A CoverageJSON document does not have the shape this reader expects."""


# -------------------------------------------------------------- parameter names


@dataclass(frozen=True, slots=True)
class ParameterName:
    """One EDR parameter name, split into its four fields.

    `level` is `None` when it was written as `*`, which means "any level" — see
    the module docstring for why that is the sane default rather than a
    shortcut.
    """

    standard_name: str
    level: float | None
    method: str
    period: str

    @classmethod
    def parse(cls, text: str) -> ParameterName:
        parts = text.split(":")
        if len(parts) != 4:
            raise ValueError(
                f"parameter name {text!r} must be "
                f"'<standard_name>:<level>:<method>:<period>', "
                f"e.g. 'air_temperature:2.0:point:PT0S'"
            )
        name, level, method, period = (p.strip() for p in parts)
        if not (name and method and period):
            raise ValueError(f"parameter name {text!r} has an empty field")
        if level == LEVEL_ANY:
            return cls(name, None, method, period)
        if not level:
            raise ValueError(f"parameter name {text!r} has an empty level; use '*' for any")
        try:
            return cls(name, float(level), method, period)
        except ValueError:
            raise ValueError(
                f"parameter name {text!r} has a non-numeric level {level!r}; use '*' for any"
            ) from None

    @property
    def text(self) -> str:
        level = LEVEL_ANY if self.level is None else repr(self.level)
        return f"{self.standard_name}:{level}:{self.method}:{self.period}"

    @property
    def measurement(self) -> str:
        """Level-free identity: `air_temperature:point:PT0S`.

        This is what a schema is made of. Including the level would make every
        station its own schema variant, since the level is routinely the station
        elevation.
        """
        return f"{self.standard_name}:{self.method}:{self.period}"

    def matches(self, other: ParameterName) -> bool:
        """True if `other` is an acceptable realisation of this selector."""
        if (self.standard_name, self.method, self.period) != (
            other.standard_name,
            other.method,
            other.period,
        ):
            return False
        if self.level is None or other.level is None:
            return True
        return abs(self.level - other.level) <= _LEVEL_TOL


def wigos_id(value: str) -> str:
    """Canonical WIGOS station identifier.

    Deliberately *not* normalised beyond stripping: the local identifier is a
    string by specification, so `0-20000-0-06260` must not be collapsed onto
    `0-20000-0-6260`. Both the WMO-number issuer (`0-20000-...`) and the
    ISO-3166-numeric issuers (`0-276-...` for Germany) occur in the live feed.
    """
    text = str(value).strip()
    if not _WIGOS_RE.match(text):
        raise ValueError(
            f"{value!r} is not a WIGOS id; expected four dash-separated fields "
            f"like '0-20000-0-06260' or '0-276-0-10015'"
        )
    return text


def schema_fingerprint(parameter_names: Iterable[str]) -> str:
    """Stable hash of one normalised measurement set.

    Normalisation drops the level, so the fingerprint tracks *what a station
    measures* rather than how high its barometer sits — otherwise every station
    would be its own schema. It is order-independent, and a genuine addition,
    removal or change of method/period does move it.
    """
    measurements = set()
    for name in parameter_names:
        text = name.strip()
        if not text:
            continue
        try:
            measurements.add(ParameterName.parse(text).measurement)
        except ValueError:
            measurements.add(text)
    digest = hashlib.sha256("|".join(sorted(measurements)).encode("utf-8")).hexdigest()
    return digest[:16]


def product_fingerprint(by_group: Mapping[str, str]) -> str:
    """Stable hash of a whole product's per-standard-name fingerprints."""
    joined = "|".join(f"{k}={by_group[k]}" for k in sorted(by_group))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def _split_source_key(key: str) -> tuple[str, ParameterName]:
    """`"air_temperature:*:point:PT0S"` -> `("air_temperature", ParameterName(...))`.

    The standard name doubles as the group: it is exactly the value passed to
    the API's `standard_name=` filter, so it decides which requests get made,
    the same way a DWD group decides which directory gets listed.
    """
    selector = ParameterName.parse(key)
    return selector.standard_name, selector


def variables_by_standard_name(
    cfg: ProductConfig,
) -> dict[str, list[tuple[ParameterName, Variable]]]:
    """`{standard_name: [(selector, variable), ...]}` for everything the config wants."""
    out: dict[str, list[tuple[ParameterName, Variable]]] = {}
    for var in cfg.variables:
        group, selector = _split_source_key(var.source_key)
        out.setdefault(group, []).append((selector, var))
    return out


# ------------------------------------------------------------------- locations


@dataclass(frozen=True, slots=True)
class StationRecord:
    """One feature of a `/locations` GeoJSON response."""

    station_id: str
    lon: float
    lat: float
    name: str
    parameters: tuple[str, ...]

    def measures(self, selectors: Iterable[ParameterName]) -> bool:
        """True if any advertised parameter satisfies any of `selectors`."""
        available = []
        for text in self.parameters:
            try:
                available.append(ParameterName.parse(text))
            except ValueError:
                continue
        return any(sel.matches(have) for sel in selectors for have in available)


def parse_locations(payload: Any) -> list[StationRecord]:
    """Parse a `/locations` GeoJSON FeatureCollection.

    A feature whose id is not a WIGOS id, or whose geometry is not a finite
    point, is skipped with a warning rather than raising: one malformed feature
    must not take out a continent-sized station list.
    """
    features = (payload or {}).get("features") or []
    records: list[StationRecord] = []
    for feature in features:
        props = feature.get("properties") or {}
        geometry = feature.get("geometry") or {}
        coords = geometry.get("coordinates") or []
        try:
            sid = wigos_id(feature["id"])
            lon, lat = float(coords[0]), float(coords[1])
        except (KeyError, TypeError, ValueError, IndexError):
            logger.warning("skipping unparseable location feature: %r", str(feature)[:120])
            continue
        if not (math.isfinite(lon) and math.isfinite(lat)):
            logger.warning("skipping location %s with non-finite coordinates", sid)
            continue
        names = props.get("parameter-name") or []
        records.append(
            StationRecord(
                station_id=sid,
                lon=lon,
                lat=lat,
                name=str(props.get("name") or ""),
                parameters=tuple(str(n) for n in names),
            )
        )
    return records


# ----------------------------------------------------------------- harvest slots


def _iso_duration(delta: timedelta) -> str:
    """`timedelta` -> the coarsest exact ISO-8601 duration token."""
    seconds = int(delta.total_seconds())
    if seconds % 3600 == 0:
        return f"PT{seconds // 3600}H"
    if seconds % 60 == 0:
        return f"PT{seconds // 60}M"
    return f"PT{seconds}S"


def _instant(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _slot_token(start: datetime, slot: timedelta) -> str:
    return f"{start.strftime('%Y%m%dT%H%M%SZ')}_{_iso_duration(slot)}"


def _parameter_digest(standard_names: Sequence[str]) -> str:
    """Short hash of the standard names a unit asks for.

    It is part of the unit key because it is part of what the raw file
    *contains*: widening the product's variable list changes the payload at the
    same URL, and the ledger has to notice that rather than report the narrower
    harvest as already done.
    """
    joined = "|".join(sorted(standard_names))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:8]


def unit_key(
    cfg: ProductConfig, station_id: str, slot_start: datetime, slot: timedelta, digest: str
) -> str:
    """`<source>/<product>/<digest>/<slot>/<wigos id>`.

    Stable because `slot_start` is floored onto a fixed grid, so the same job
    re-run — or retried a minute later — yields the same key. Distinct across
    harvests because a later run covers later slots. Both properties are load
    bearing for a feed with no archive; see the module docstring.
    """
    return f"{cfg.source}/{cfg.product}/{digest}/{_slot_token(slot_start, slot)}/{station_id}"


# ------------------------------------------------------------------ the source


class EsohObservations:
    """EUMETNET E-SOH observations via MeteoGate as a `meteopack` source.

    Public API::

        src = EsohObservations(source_cfg, session=..., limiter=...)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain stations
        src.list(window, product_cfg)    -> list[Unit]        (one per station-slot)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == upstream 404)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        EsohObservations.schema_fingerprint(names) -> str

    `limiter` is whatever `meteopack.core.http` expects and is passed straight
    through to `fetch_file`. Listing stations uses `requests` directly, because
    that is a small JSON read rather than an archive download and `core.http`
    only owns the latter.
    """

    name = "esoh"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="esoh", base_url=BASE_URL)
        self.base_url = (self.source.base_url or BASE_URL).rstrip("/")
        self.collection = str(self.source.extra.get("collection", COLLECTION))
        self._session = session
        self._limiter = limiter
        self._now = now
        self._locations_cache: dict[str, list[StationRecord]] = {}

    # ------------------------------------------------------------------ config

    @property
    def rate_limit(self) -> RateLimit:
        return self.source.rate_limit

    @property
    def timeout(self) -> float:
        return float(self.source.extra.get("timeout_seconds", 60))

    def _clock(self) -> datetime:
        return self._now or datetime.now(UTC)

    def _requests_session(self):
        if self._session is None:
            import requests

            self._session = requests.Session()
            agent = self.source.extra.get("user_agent", "meteopack/0.1 (+esoh-meteogate)")
            self._session.headers["User-Agent"] = str(agent)
        return self._session

    # ------------------------------------------------------------- remote reads

    def _get_json(self, url: str) -> Any:
        """One plain GET, decoded as JSON. Overridable seam for tests.

        `None` for a 404, which is how this API says "nothing matched" — both
        for an empty bbox and for an unknown station.
        """
        resp = self._requests_session().get(url, timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def locations_url(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        standard_names: Sequence[str] = (),
    ) -> str:
        query: dict[str, str] = {"f": "GeoJSON"}
        if bbox is not None:
            query["bbox"] = ",".join(f"{v:.6f}" for v in bbox)
        if standard_names:
            query["standard_name"] = ",".join(sorted(standard_names))
        return (
            f"{self.base_url}/collections/{self.collection}/locations"
            f"?{urlencode(query, safe=',:/')}"
        )

    def coverage_url(
        self,
        station_id: str,
        *,
        standard_names: Sequence[str],
        start: datetime,
        end: datetime,
    ) -> str:
        """CoverageJSON for one station over one closed interval.

        The requested interval ends one second *before* the slot end. EDR
        `datetime` intervals are closed at both ends, so adjacent half-open
        slots would otherwise both contain the boundary instant and the same
        observation would land in two units.
        """
        query = {
            "f": "CoverageJSON",
            "datetime": f"{_instant(start)}/{_instant(end - timedelta(seconds=1))}",
        }
        if standard_names:
            query["standard_name"] = ",".join(sorted(standard_names))
        return (
            f"{self.base_url}/collections/{self.collection}/locations/{quote(station_id)}"
            f"?{urlencode(query, safe=',:/')}"
        )

    def locations(
        self,
        *,
        bbox: tuple[float, float, float, float] | None = None,
        standard_names: Sequence[str] = (),
    ) -> list[StationRecord]:
        """Stations the API advertises for a bbox and parameter set (cached)."""
        url = self.locations_url(bbox=bbox, standard_names=standard_names)
        if url in self._locations_cache:
            return self._locations_cache[url]
        try:
            payload = self._get_json(url)
        except Exception as exc:  # noqa: BLE001 - a failed listing is not fatal
            logger.warning("locations query failed for %s: %s", url, exc)
            self._locations_cache[url] = []
            return []
        if payload is None:
            logger.info("esoh: no stations for %s (404)", url)
            self._locations_cache[url] = []
            return []
        records = parse_locations(payload)
        self._locations_cache[url] = records
        return records

    # ----------------------------------------------------------------- stations

    def stations(self, cfg: ProductConfig) -> pl.DataFrame:
        """In-domain stations advertising anything this product wants.

        Clipping is the grid's job twice over: the grid's bbox narrows the
        query, then `grids.get(cfg.grid_id).contains(lon, lat)` trims the
        rectangle back to the real footprint. With `grid_id: null` nothing is
        clipped and the whole European network is kept — the deliberate escape
        hatch for a continent-wide run, optionally narrowed by `extra.bbox`.
        """
        import polars as pl

        wanted = variables_by_standard_name(cfg)
        selectors = [sel for pairs in wanted.values() for sel, _ in pairs]
        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        records = self.locations(bbox=self._bbox(cfg), standard_names=sorted(wanted))

        rows = []
        for rec in records:
            if grid is not None and not grid.contains(rec.lon, rec.lat):
                continue
            if selectors and not rec.measures(selectors):
                continue
            rows.append(
                {
                    "station_id": rec.station_id,
                    "lat": rec.lat,
                    "lon": rec.lon,
                    "name": rec.name,
                    "parameters": ",".join(sorted(rec.parameters)),
                }
            )
        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "name": pl.String,
            "parameters": pl.String,
        }
        return pl.DataFrame(rows, schema=schema).unique(subset=["station_id"]).sort("station_id")

    def _bbox(self, cfg: ProductConfig) -> tuple[float, float, float, float] | None:
        """Query rectangle: the grid's bbox, an explicit override, or none."""
        override = cfg.extra.get("bbox")
        if override:
            west, south, east, north = (float(v) for v in override)
            return west, south, east, north
        if cfg.grid_id:
            return grids.get(cfg.grid_id).bbox()
        return None

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every (station, harvest slot) in `window` that the feed can still serve.

        No observations are downloaded; only the station list is read. A window
        reaching back beyond the rolling retention is clamped rather than
        expanded into units that are certain to 404.
        """
        start, end = self._window_bounds(window)
        start, end = self._clamp_to_rolling(start, end, cfg)
        if start is None:
            return []

        slot = self._slot(cfg)
        start = _floor(start, slot)
        end = _ceil(end, slot)
        # Flooring is what makes a retry idempotent, but it can reach back past
        # what the feed still serves. Only the RETENTION horizon may veto a slot —
        # not the requested start, or a run at :37 would drop the slot its :00
        # sibling harvested and every retry would look like new work.
        horizon = self._retention_horizon(cfg)
        if start < horizon:
            # A half-expired slot 404s for every station that reports on the hour,
            # and the ledger would file thousands of units as permanently absent.
            start = _ceil(horizon, slot)
            if start >= end:
                logger.warning("esoh: no whole slot is still inside the rolling window")
                return []
        # Timeline owns cadence and UTC validation for every store here; borrow
        # it to count the slots rather than inventing a second rule for them.
        n_slots = len(Timeline(start, end, slot))
        slots = [start + i * slot for i in range(n_slots)]

        wanted = variables_by_standard_name(cfg)
        names = sorted(wanted)
        digest = _parameter_digest(names)
        table = self.stations(cfg)
        keep = table["station_id"].to_list()
        only = cfg.extra.get("station_ids")
        if only:
            wanted_ids = {wigos_id(s) for s in only}
            keep = [s for s in keep if s in wanted_ids]
        limit = cfg.extra.get("max_stations")
        if limit:
            keep = keep[: int(limit)]
        if not keep:
            logger.warning("no in-domain E-SOH stations for grid %s", cfg.grid_id)
            return []

        units: list[Unit] = []
        for slot_start in slots:
            slot_end = slot_start + slot
            for sid in keep:
                units.append(
                    Unit(
                        key=unit_key(cfg, sid, slot_start, slot, digest),
                        url=self.coverage_url(
                            sid, standard_names=names, start=slot_start, end=slot_end
                        ),
                        valid_time=slot_start,
                        meta={
                            # A slot that has not closed yet is still filling
                            # upstream. Its key is deliberately stable, so without
                            # this the ledger would mark a half-harvested hour done
                            # and the rest would never be collected — and for a
                            # feed with no archive, never is permanent.
                            "volatile": slot_end > self._clock(),
                            "station_id": sid,
                            "slot_start": _instant(slot_start),
                            "slot_end": _instant(slot_end),
                            "slot": _iso_duration(slot),
                            "digest": digest,
                            "standard_names": ",".join(names),
                            "filename": f"{sid}.covjson",
                        },
                    )
                )
        logger.info(
            "esoh: %d units across %d slot(s) x %d station(s) for %d standard name(s)",
            len(units),
            len(slots),
            len(keep),
            len(names),
        )
        return units

    def _slot(self, cfg: ProductConfig) -> timedelta:
        seconds = int(cfg.extra.get("harvest_slot_seconds", DEFAULT_SLOT_SECONDS))
        if seconds <= 0:
            raise ValueError("extra.harvest_slot_seconds must be positive")
        return timedelta(seconds=seconds)

    def _retention_horizon(self, cfg: ProductConfig) -> datetime:
        """The oldest instant the rolling collection still serves."""
        hours = float(cfg.extra.get("rolling_window_hours", ROLLING_WINDOW_HOURS))
        return self._clock() - timedelta(hours=hours)

    def _clamp_to_rolling(
        self, start: datetime, end: datetime, cfg: ProductConfig
    ) -> tuple[datetime | None, datetime]:
        """Pull `start` forward to the oldest instant the feed still serves.

        Returns `(None, end)` when the whole window is already gone — there is
        no archive, so that is a finished job with nothing to do, not an error.
        """
        hours = float(cfg.extra.get("rolling_window_hours", ROLLING_WINDOW_HOURS))
        horizon = self._retention_horizon(cfg)
        if start < horizon:
            logger.warning(
                "esoh: window starts %s but the rolling collection only reaches back to %s; "
                "clamping. Data older than that does not exist upstream.",
                _instant(start),
                _instant(horizon),
            )
            start = horizon
        if start >= end:
            logger.warning(
                "esoh: nothing to harvest — window %s..%s is entirely outside the rolling "
                "%.0f hour collection",
                _instant(start),
                _instant(end),
                hours,
            )
            return None, end
        return start, end

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Download one station-slot CoverageJSON. None if upstream has nothing.

        A station that advertised a parameter but reported nothing in this slot
        answers 404, which is routine for a real-time feed and a skip rather
        than a failure. Everything else (retries, `.part` + atomic rename,
        checksum, rate limiting) belongs to `meteopack.core.http`; the import is
        local so tests can monkeypatch it.
        """
        from meteopack.core.http import NotFoundError, fetch_file

        meta = unit.meta or {}
        dest = (
            Path(raw_root)
            / meta.get("digest", "unknown")
            / meta.get("slot_start", "unknown").replace(":", "")
            / meta.get("filename", f"{unit.key.rsplit('/', 1)[-1]}.covjson")
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
                # Rewritten upstream under the same name: the ledger re-queues it,
                # so the download must actually happen rather than skip.
                overwrite=bool((unit.meta or {}).get("volatile")),
            )
        except NotFoundError:
            logger.info("esoh: no observations for %s (404)", unit.key)
            return None

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One CoverageJSON document -> tidy frame of `time`, `station_id`, values.

        Only the parameters `cfg.variables` asks for are read. Missing values
        (JSON `null`, plus any numeric sentinel listed in
        `extra.missing_sentinels`) become null *before* scale/offset, so a
        sentinel can never be scaled into a plausible-looking number, and the
        valid range is applied afterwards, so it is expressed in canonical
        units.

        Cadences are not aligned or snapped here. A parameter reporting hourly
        and one reporting every ten minutes outer-join on time, leaving nulls
        where the slower one has nothing to say; inventing values for those
        rows is the resampler's decision to make, not the reader's.
        """
        import polars as pl

        wanted = [
            (sel, var)
            for pairs in variables_by_standard_name(cfg).values()
            for sel, var in pairs
        ]
        document = _read_document(Path(raw.path))
        coverages = _coverages(document)
        self._check_schema(coverages, cfg, Path(raw.path).name)
        if not wanted:
            return _empty_frame(wanted)

        fallback = (raw.unit_key or "").rsplit("/", 1)[-1]
        rows: dict[tuple[str, datetime], dict] = {}
        # Candidate range keys per station, so a variable resolves to exactly
        # one series even when several levels of the same measurement exist.
        by_station: dict[str, list[tuple[ParameterName, int, str]]] = {}
        parsed: list[tuple[str, list[datetime], dict[str, list]]] = []

        for index, cov in enumerate(coverages):
            station = _coverage_station(cov, fallback)
            times = _coverage_times(cov, Path(raw.path).name)
            ranges = cov.get("ranges") or {}
            values: dict[str, list] = {}
            for key, nd in ranges.items():
                try:
                    name = ParameterName.parse(key)
                except ValueError:
                    logger.warning("skipping unparseable range key %r", key)
                    continue
                series = _nd_values(nd, len(times), key, Path(raw.path).name)
                values[key] = series
                by_station.setdefault(station, []).append((name, index, key))
            parsed.append((station, times, values))

        chosen: dict[tuple[str, str], str] = {}
        for station, candidates in by_station.items():
            for selector, var in wanted:
                key = _best_match(selector, candidates)
                if key is not None:
                    chosen[(station, var.name)] = key

        for station, times, values in parsed:
            for _, var in wanted:
                key = chosen.get((station, var.name))
                if key is None or key not in values:
                    continue
                for when, value in zip(times, values[key], strict=True):
                    rows.setdefault((station, when), {})[var.name] = _number(value)

        schema: dict[str, Any] = {"time": pl.Datetime("us", "UTC"), "station_id": pl.String}
        for _, var in wanted:
            schema[var.name] = pl.Float64
        records = [
            {"time": when, "station_id": station, **{var.name: None for _, var in wanted}, **vals}
            for (station, when), vals in rows.items()
        ]
        frame = pl.DataFrame(records, schema=schema)
        frame = _apply_variables(frame, [var for _, var in wanted], cfg)
        return frame.unique(subset=["time", "station_id"], keep="last").sort(
            ["time", "station_id"]
        )

    def _check_schema(
        self, coverages: Sequence[dict], cfg: ProductConfig, filename: str
    ) -> None:
        """Refuse to parse a document that reports something the config never saw.

        The pinned thing cannot be a per-file fingerprint the way it is for
        DWD, because no two stations here carry the same parameter set: De Bilt
        reports `air_temperature` as `point:PT0S` *and* `point:PT1M`, its
        neighbour only the first. A station carrying a subset of the catalogue
        is therefore normal and passes. What is pinned is the catalogue itself —
        `extra.known_measurements`, one level-free measurement list per standard
        name, taken from `scan` — and drift means a measurement appearing that
        is not in it.

        `cfg.schema_fingerprint` hashes that catalogue, so editing the list
        without re-pinning the hash fails loudly instead of quietly widening
        what counts as known.
        """
        known = cfg.extra.get("known_measurements") or {}
        if not known:
            return
        catalogue = {group: schema_fingerprint(names) for group, names in known.items()}
        pinned = cfg.schema_fingerprint
        if pinned:
            seen = product_fingerprint(catalogue)
            if seen != pinned:
                raise SchemaDriftError(
                    f"config is inconsistent: extra.known_measurements hashes to {seen} but "
                    f"schema_fingerprint pins {pinned}. Re-run `scan` and re-pin both together."
                )
        for group, measurements in _measurements_by_group(coverages).items():
            allowed = set(known.get(group) or ())
            if not allowed:
                continue
            added = sorted(measurements - allowed)
            if added:
                raise SchemaDriftError(
                    f"{filename}: E-SOH {group} reports measurement(s) {added} that are not in "
                    f"the pinned catalogue {sorted(allowed)}. Re-run `scan` and re-pin "
                    f"extra.known_measurements and schema_fingerprint after checking what moved."
                )

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile]) -> dict:
        """Report every distinct measurement set across `raw_files`.

        Upstream adds and drops parameters without announcing it, so the
        pipeline pins the fingerprints this returns and `parse` refuses to run
        against anything else. The report is deliberately a plain dict: it is
        meant to be written to JSON next to the store and diffed between runs.
        """
        variants: dict[tuple[str, tuple[str, ...]], dict] = {}
        errors: list[dict[str, str]] = []
        for raw in raw_files:
            path = Path(raw.path)
            try:
                coverages = _coverages(_read_document(path))
                by_group = _measurements_by_group(coverages)
                domains = sorted(
                    {str((c.get("domain") or {}).get("domainType", "")) for c in coverages}
                )
                dtypes = _nd_dtypes(coverages)
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            for group, measurements in by_group.items():
                columns = tuple(sorted(measurements))
                entry = variants.setdefault(
                    (group, columns),
                    {
                        "standard_name": group,
                        "measurements": list(columns),
                        "domain_types": domains,
                        "dtypes": {m: dtypes.get(m, "null") for m in columns},
                        "fingerprint": schema_fingerprint(columns),
                        "count": 0,
                        "examples": [],
                    },
                )
                entry["count"] += 1
                for domain in domains:
                    if domain not in entry["domain_types"]:
                        entry["domain_types"].append(domain)
                if len(entry["examples"]) < 3:
                    entry["examples"].append(path.name)

        ordered = sorted(variants.values(), key=lambda e: (e["standard_name"], e["fingerprint"]))
        by_name: dict[str, list[str]] = {}
        union: dict[str, set[str]] = {}
        for entry in ordered:
            by_name.setdefault(entry["standard_name"], []).append(entry["fingerprint"])
            union.setdefault(entry["standard_name"], set()).update(entry["measurements"])
        # The union, not the per-file variants, is what goes in config: one
        # station is always a subset of the catalogue, so pinning a variant
        # would reject its neighbours. See `_check_schema`.
        known = {g: sorted(ms) for g, ms in union.items()}
        catalogue = {g: schema_fingerprint(ms) for g, ms in known.items()}
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": {g: sorted(fps) for g, fps in by_name.items()},
            "known_measurements": known,
            "catalogue_fingerprints": catalogue,
            "fingerprint": product_fingerprint(catalogue) if catalogue else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ---------------------------------------------------------------- slot helpers


def _floor(value: datetime, slot: timedelta) -> datetime:
    step = int(slot.total_seconds())
    epoch = int(value.timestamp())
    return datetime.fromtimestamp(epoch - epoch % step, UTC)


def _ceil(value: datetime, slot: timedelta) -> datetime:
    floored = _floor(value, slot)
    return floored if floored == value else floored + slot


# ------------------------------------------------------------ document helpers


def _read_document(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaDriftError(f"{path.name}: not valid JSON ({exc})") from None


def _coverages(document: Any) -> list[dict]:
    """Both CoverageJSON shapes, flattened to a list of coverages.

    Homogeneous cadence comes back as a bare `Coverage`; mixed cadence as a
    `CoverageCollection` with one coverage per parameter. Treating only one of
    the two as valid is the single easiest way to lose half a station's data.
    """
    kind = (document or {}).get("type")
    if kind == _COVERAGE_COLLECTION:
        return [c for c in (document.get("coverages") or []) if isinstance(c, dict)]
    if kind == _COVERAGE:
        return [document]
    raise SchemaDriftError(
        f"expected a {_COVERAGE} or {_COVERAGE_COLLECTION} document, got type={kind!r}"
    )


def _coverage_station(cov: Mapping[str, Any], fallback: str) -> str:
    """WIGOS id of one coverage, from the document if it says, else the unit key."""
    raw = cov.get(_WIGOS_KEY)
    for candidate in (raw, fallback):
        if not candidate:
            continue
        try:
            return wigos_id(str(candidate))
        except ValueError:
            continue
    raise SchemaDriftError(f"coverage carries no {_WIGOS_KEY} and the unit key has no station id")


def _coverage_times(cov: Mapping[str, Any], filename: str) -> list[datetime]:
    axes = ((cov.get("domain") or {}).get("axes")) or {}
    try:
        values = axes["t"]["values"]
    except (KeyError, TypeError):
        raise SchemaDriftError(f"{filename}: coverage has no domain.axes.t.values") from None
    for axis in ("x", "y"):
        found = (axes.get(axis) or {}).get("values") or []
        if len(found) != 1:
            raise SchemaDriftError(
                f"{filename}: expected a single {axis} coordinate for a station series, "
                f"got {len(found)}"
            )
    out: list[datetime] = []
    for text in values:
        try:
            # 3.11+ fromisoformat accepts the trailing 'Z' the feed actually writes.
            when = datetime.fromisoformat(str(text))
        except ValueError:
            raise SchemaDriftError(f"{filename}: unparseable time {text!r}") from None
        out.append(when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC))
    return out


def _nd_values(nd: Mapping[str, Any], n_times: int, key: str, filename: str) -> list:
    values = nd.get("values")
    if not isinstance(values, list):
        raise SchemaDriftError(f"{filename}: range {key!r} has no values list")
    if len(values) != n_times:
        raise SchemaDriftError(
            f"{filename}: range {key!r} has {len(values)} values for {n_times} times; "
            f"this reader only handles single-point series (shape [t, 1, 1])"
        )
    return values


def _measurements_by_group(coverages: Sequence[Mapping[str, Any]]) -> dict[str, set[str]]:
    """`{standard_name: {level-free measurement, ...}}` across a document."""
    out: dict[str, set[str]] = {}
    for cov in coverages:
        for key in (cov.get("ranges") or {}):
            try:
                name = ParameterName.parse(key)
            except ValueError:
                continue
            out.setdefault(name.standard_name, set()).add(name.measurement)
    return out


def _nd_dtypes(coverages: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for cov in coverages:
        for key, nd in (cov.get("ranges") or {}).items():
            try:
                measurement = ParameterName.parse(key).measurement
            except ValueError:
                continue
            out[measurement] = str((nd or {}).get("dataType", "null"))
    return out


def _best_match(
    selector: ParameterName, candidates: Sequence[tuple[ParameterName, int, str]]
) -> str | None:
    """The one range key a selector resolves to, chosen deterministically.

    A wildcard level can match several series at one station (two barometers,
    two thermometer heights). Picking the lowest level and logging the rest is
    arbitrary but repeatable, which is what matters: the alternative is a column
    whose provenance changes between runs.
    """
    matches = [(name, key) for name, _, key in candidates if selector.matches(name)]
    if not matches:
        return None
    matches.sort(key=lambda pair: (math.inf if pair[0].level is None else pair[0].level, pair[1]))
    if len(matches) > 1:
        logger.info(
            "esoh: %s matches %d series; using %s",
            selector.text,
            len(matches),
            matches[0][1],
        )
    return matches[0][1]


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _empty_frame(wanted: Sequence[tuple[ParameterName, Variable]]) -> pl.DataFrame:
    import polars as pl

    schema: dict[str, Any] = {
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

    E-SOH expresses missing as JSON `null`, which `_number` has already turned
    into a null. `extra.missing_sentinels` exists because national feeds
    occasionally push a numeric sentinel through unchanged, and a -999 that gets
    scaled first is indistinguishable from data.
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
        raise ValueError(f"unsupported dtype {name!r} for an E-SOH variable") from None
