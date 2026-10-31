"""DMI (Denmark) metObs 10-minute station observations.

The feed is an OGC API - Features service, keyless since March 2026:

    <base>/station/items?bbox=&limit=
    <base>/observation/items?stationId=&parameterId=&datetime=&limit=&offset=

Four pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store, or the ledger, if you get it wrong:

* **One feature is one `(stationId, parameterId, observed, value)` record.**
  There is no "give me this station's row for this minute" call: a tidy frame
  has to be built by querying per station x parameterId x time window and
  pivoting the features onto a `(time, station)` grid. `parse` does that pivot,
  keyed on `parameterId`, so a document carrying several parameters widens into
  several columns rather than stacking into duplicate rows.
* **`observation/items` caps a page at 300000 features.** A window wider than
  that is silently truncated — the response is a perfectly valid, short
  FeatureCollection. `list` refuses to emit a window that *could* overflow
  (window seconds / cadence vs the cap) and `parse` raises if a document comes
  back exactly at the cap, so truncation fails loudly instead of showing up as
  a hole three years later.
* **Only the 10-minute-native parameters belong here.** metObs also publishes
  `*_past1h`, 12 h and 24 h aggregates, and `precip_past1min`, which is
  advertised by stations but not populated through this endpoint. They are
  excluded by simply not appearing in the product config's variable map, which
  is also what decides which requests get made.
* **metObs is raw, not quality controlled.** DMI applies no QC before
  publishing, so `valid_min`/`valid_max` are doing real work here rather than
  guarding against a corrupt file — nonphysical values genuinely occur.

Station ids are DMI's own five-character codes ("06041"). They are kept as
text, zero padding included, because that is how both `station/items` and
`observation/items` write them and how they join against DMI's own metadata.

The server is latency-bound rather than bandwidth-bound: measured ~0.23
files/s, with no 429s up to twelve concurrent workers. The source config caps
workers there; raising it buys little and risks the throttle.

The upstream record schema drifts without announcement, so `scan` reports every
feature-property set seen and `parse` refuses to run against a pinned
fingerprint it does not match.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
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

BASE_URL = "https://opendataapi.dmi.dk/v2/metObs/collections"

STATION_COLLECTION = "station/items"
OBSERVATION_COLLECTION = "observation/items"

#: Hard cap on `limit` for `observation/items`; the API 400s above it and
#: silently truncates at it.
PAGE_LIMIT = 300_000

#: Width of one download window, in whole years. Three years of 10-minute data
#: is ~158k features, comfortably inside `PAGE_LIMIT`.
DEFAULT_WINDOW_YEARS = 3

#: Windows are blocked from this year, not from the requested start, so the
#: same station-parameter always lands in the same window whatever window a job
#: happens to ask for. Unit keys are the ledger's notion of identity; anchoring
#: on the request would make them a function of how the backfill was sliced.
DEFAULT_ANCHOR_YEAR = 1950

#: metObs station ids are five characters, zero padded.
_ID_WIDTH = 5

#: Feature property names `parse` depends on.
_OBSERVED = "observed"
_VALUE = "value"
_PARAMETER = "parameterId"
_STATION = "stationId"


#: A metObs document does not have the shape this reader expects.
SchemaDriftError = _SchemaDriftError


class PageOverflowError(RuntimeError):
    """A download window returned a full page, so features were truncated."""


def station_id(value: str | int) -> str:
    """Canonical five-character station id. DMI pads inconsistently by hand."""
    text = str(value).strip()
    if text.isdigit():
        return text.zfill(_ID_WIDTH)
    if not text:
        raise ValueError("empty DMI station id")
    return text


def schema_fingerprint(property_keys: Iterable[str]) -> str:
    """Stable hash of one normalised feature-property set.

    The record schema — not the parameter list — is what `parse` depends on:
    it reads `observed`, `value`, `parameterId` and `stationId` out of every
    feature. Which parameters a document carries is a product-config question
    and varies per unit by construction, so it is deliberately not hashed here.
    Order-independent; an addition, removal or rename does move it.
    """
    normalised = sorted({k.strip() for k in property_keys} - {""})
    digest = hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()
    return digest[:16]


def product_fingerprint(by_group: Mapping[str, str]) -> str:
    """Stable hash of a whole product's per-group fingerprints."""
    joined = "|".join(f"{k}={by_group[k]}" for k in sorted(by_group))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def variables_by_parameter(cfg: ProductConfig) -> dict[str, Variable]:
    """`{parameterId: variable}` for everything the config wants.

    One parameter maps to exactly one canonical column; two variables claiming
    the same `source_key` would make the pivot's output depend on config order,
    so it is rejected rather than resolved.
    """
    out: dict[str, Variable] = {}
    for var in cfg.variables:
        pid = var.source_key.strip()
        if not pid:
            raise ValueError(f"variable {var.name!r} has an empty source_key")
        if pid in out:
            raise ValueError(
                f"parameterId {pid!r} is claimed by both {out[pid].name!r} and {var.name!r}"
            )
        out[pid] = var
    return out


# -------------------------------------------------------------------- stations


@dataclass(frozen=True, slots=True)
class StationRecord:
    """One station of the `station/items` collection, merged across validities.

    metObs lists a station once per validity period, each feature carrying its
    own `parameterId` list. The union is what the station actually measures
    over time; the most recent feature is what it measures *where*.
    """

    station_id: str
    lon: float
    lat: float
    height_m: float | None
    name: str
    kind: str
    parameters: tuple[str, ...]

    def measures(self, wanted: Iterable[str]) -> bool:
        return bool(set(self.parameters) & set(wanted))


def parse_stations(payload: Any) -> list[StationRecord]:
    """Parse a `station/items` GeoJSON FeatureCollection.

    A feature with no station id or no finite point geometry is skipped with a
    warning rather than raising: one malformed feature must not take out the
    whole Danish network.
    """
    features = (payload or {}).get("features") or []
    merged: dict[str, dict] = {}
    for feature in features:
        props = feature.get("properties") or {}
        coords = (feature.get("geometry") or {}).get("coordinates") or []
        try:
            sid = station_id(props[_STATION])
            lon, lat = float(coords[0]), float(coords[1])
        except (KeyError, TypeError, ValueError, IndexError):
            logger.warning("skipping unparseable station feature: %r", str(feature)[:120])
            continue
        if not (math.isfinite(lon) and math.isfinite(lat)):
            logger.warning("skipping station %s with non-finite coordinates", sid)
            continue
        # `validFrom` orders the validity periods; `operationFrom` is the
        # fallback for the handful of features that carry only the latter.
        stamp = str(props.get("validFrom") or props.get("operationFrom") or "")
        entry = merged.setdefault(sid, {"stamp": "", "parameters": set()})
        entry["parameters"].update(str(p) for p in (props.get(_PARAMETER) or []))
        if stamp >= entry["stamp"]:
            entry.update(
                stamp=stamp,
                lon=lon,
                lat=lat,
                height_m=_number(props.get("stationHeight")),
                name=str(props.get("name") or ""),
                kind=str(props.get("type") or ""),
            )

    records = [
        StationRecord(
            station_id=sid,
            lon=e["lon"],
            lat=e["lat"],
            height_m=e["height_m"],
            name=e["name"],
            kind=e["kind"],
            parameters=tuple(sorted(e["parameters"])),
        )
        for sid, e in merged.items()
        if "lon" in e
    ]
    return sorted(records, key=lambda r: r.station_id)


# ------------------------------------------------------------------- windows


@dataclass(frozen=True, slots=True)
class DownloadWindow:
    """One closed download interval, labelled by the block it belongs to."""

    start: datetime
    end: datetime
    label: str


def windows(
    start: datetime, end: datetime, *, years: int, anchor: int = DEFAULT_ANCHOR_YEAR
) -> list[DownloadWindow]:
    """Calendar-year blocks of `years` covering [start, end), anchored on `anchor`.

    Blocking on a fixed anchor rather than on `start` is what makes a unit key
    stable: a 2012-2015 backfill and a 2014-2016 one must agree on which window
    holds 2014, or the ledger sees the same data as two different units and
    fetches it twice.
    """
    if years <= 0:
        raise ValueError("extra.window_years must be positive")
    first = anchor + (start.year - anchor) // years * years
    out: list[DownloadWindow] = []
    year = first
    while year < end.year + 1:
        block_end = year + years
        out.append(
            DownloadWindow(
                start=datetime(year, 1, 1, tzinfo=UTC),
                end=datetime(block_end, 1, 1, tzinfo=UTC),
                label=f"{year}-{block_end - 1}",
            )
        )
        year = block_end
    return out


def unit_key(cfg: ProductConfig, sid: str, parameter: str, label: str) -> str:
    """`<source>/<product>/<station>/<parameterId>/<window label>`."""
    return f"{cfg.source}/{cfg.product}/{sid}/{parameter}/{label}"


# ------------------------------------------------------------------ the source


class DmiStations:
    """DMI metObs 10-minute station observations as a `meteopack` source.

    Public API::

        src = DmiStations(source_cfg, session=..., limiter=...)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain stations
        src.list(window, product_cfg)    -> list[Unit]        (station x parameter x window)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == upstream 404)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        DmiStations.schema_fingerprint(keys) -> str

    `limiter` is whatever `meteopack.core.http` expects; it is passed straight
    through to `fetch_file`. Listing stations uses `requests` directly, because
    that is a small JSON read rather than an archive download and `core.http`
    only owns the latter.
    """

    name = "dmi"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="dmi", base_url=BASE_URL)
        self.base_url = (self.source.base_url or BASE_URL).rstrip("/")
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
        """One plain GET, decoded as JSON. Overridable seam for tests.

        `None` for a 404, which is how this API says "no such collection or
        station" — a normal answer, not a failure.
        """
        resp = self._requests_session().get(url, timeout=self.timeout)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.json()

    def stations_url(self, *, bbox: tuple[float, float, float, float] | None, limit: int) -> str:
        query: dict[str, str] = {"limit": str(limit)}
        if bbox is not None:
            query["bbox"] = ",".join(f"{v:.6f}" for v in bbox)
        return f"{self.base_url}/{STATION_COLLECTION}?{urlencode(query, safe=',:/')}"

    def observation_url(
        self, sid: str, parameter: str, *, start: datetime, end: datetime, limit: int = PAGE_LIMIT
    ) -> str:
        """Features for one station x parameter over one interval.

        The interval ends one second *before* the window end: metObs `datetime`
        ranges are closed at both ends, so adjacent half-open windows would
        otherwise both contain the boundary observation and it would be stored
        twice.
        """
        query = {
            _STATION: sid,
            _PARAMETER: parameter,
            "datetime": f"{_instant(start)}/{_instant(end - timedelta(seconds=1))}",
            "limit": str(limit),
        }
        return f"{self.base_url}/{OBSERVATION_COLLECTION}?{urlencode(query, safe=',:/')}"

    def catalogue(
        self, *, bbox: tuple[float, float, float, float] | None, limit: int
    ) -> list[StationRecord]:
        """Stations the API advertises for a bbox (cached per instance)."""
        url = self.stations_url(bbox=bbox, limit=limit)
        if url in self._station_cache:
            return self._station_cache[url]
        try:
            payload = self._get_json(url)
        except Exception as exc:  # noqa: BLE001 - a failed listing is not fatal
            logger.warning("station catalogue fetch failed for %s: %s", url, exc)
            self._station_cache[url] = []
            return []
        records = [] if payload is None else parse_stations(payload)
        self._station_cache[url] = records
        return records

    # ----------------------------------------------------------------- stations

    def stations(self, cfg: ProductConfig) -> pl.DataFrame:
        """In-domain stations carrying at least one parameter this product wants.

        Clipping is the grid's job twice over: its bbox narrows the query, then
        `grids.get(cfg.grid_id).contains(lon, lat)` trims the rectangle back to
        the real footprint. With `grid_id: null` the whole Danish network is
        kept, which is a deliberate escape hatch rather than a default.

        `pack` writes this beside the observations; without it the store has
        values but no coordinates and every downstream station matrix silently
        drops its columns.
        """
        import polars as pl

        wanted = variables_by_parameter(cfg)
        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        records = self.catalogue(bbox=self._bbox(cfg), limit=self._station_limit(cfg))

        rows = []
        advertised: set[str] = set()
        for rec in records:
            advertised.update(rec.parameters)
            if grid is not None and not grid.contains(rec.lon, rec.lat):
                continue
            if wanted and not rec.measures(wanted):
                continue
            rows.append(
                {
                    "station_id": rec.station_id,
                    "lat": rec.lat,
                    "lon": rec.lon,
                    "height_m": rec.height_m,
                    "name": rec.name,
                    "kind": rec.kind,
                    "parameters": ",".join(rec.parameters),
                }
            )
        # A configured parameter that nothing advertises is how a renamed or
        # retired parameterId shows up: silently, as a column that is simply
        # never requested. Say so rather than letting the store thin out.
        orphans = sorted(set(wanted) - advertised)
        if records and orphans:
            logger.warning("dmi: no station advertises parameter(s) %s", orphans)

        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "height_m": pl.Float64,
            "name": pl.String,
            "kind": pl.String,
            "parameters": pl.String,
        }
        return pl.DataFrame(rows, schema=schema).sort("station_id")

    def _bbox(self, cfg: ProductConfig) -> tuple[float, float, float, float] | None:
        override = cfg.extra.get("bbox")
        if override:
            west, south, east, north = (float(v) for v in override)
            return west, south, east, north
        if cfg.grid_id:
            return grids.get(cfg.grid_id).bbox()
        return None

    @staticmethod
    def _station_limit(cfg: ProductConfig) -> int:
        return int(cfg.extra.get("station_limit", 10_000))

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every (station, parameter, download window) overlapping `window`.

        No observations are downloaded; only the station catalogue is read.
        """
        start, end = self._window_bounds(window)
        # Validate the window against the product cadence in one place.
        Timeline(start, end, cfg.cadence)

        wanted = variables_by_parameter(cfg)
        table = self.stations(cfg)
        parameters = {
            row["station_id"]: sorted(set(str(row["parameters"]).split(",")) & set(wanted))
            for row in table.iter_rows(named=True)
        }
        keep = [sid for sid, pids in parameters.items() if pids]
        only = cfg.extra.get("station_ids")
        if only:
            wanted_ids = {station_id(s) for s in only}
            keep = [s for s in keep if s in wanted_ids]
        limit = cfg.extra.get("max_stations")
        if limit:
            keep = keep[: int(limit)]
        if not keep:
            logger.warning("no in-domain DMI stations for grid %s", cfg.grid_id)
            return []

        blocks = self._windows(start, end, cfg)
        now = self._clock()
        units: list[Unit] = []
        for sid in keep:
            for pid in parameters[sid]:
                for block in blocks:
                    units.append(
                        Unit(
                            key=unit_key(cfg, sid, pid, block.label),
                            url=self.observation_url(
                                sid, pid, start=block.start, end=block.end
                            ),
                            valid_time=block.start,
                            meta={
                                "station_id": sid,
                                "parameter": pid,
                                "label": block.label,
                                "filename": f"{block.label}.json",
                                # A block whose end is still ahead of us keeps
                                # filling upstream under an unchanged key. Without
                                # this the ledger marks a part-harvested window done
                                # and the rest of it is never collected.
                                "volatile": block.end > now,
                            },
                        )
                    )
        logger.info(
            "dmi: %d units across %d station(s) x %d window(s)",
            len(units),
            len(keep),
            len(blocks),
        )
        return units

    def _windows(
        self, start: datetime, end: datetime, cfg: ProductConfig
    ) -> list[DownloadWindow]:
        """Download blocks for this window, refusing any that could overflow a page.

        A too-wide block does not fail: it comes back as a valid, short
        FeatureCollection. Checking the arithmetic here is the only place the
        mistake is cheap to find.
        """
        years = int(cfg.extra.get("window_years", DEFAULT_WINDOW_YEARS))
        anchor = int(cfg.extra.get("window_anchor_year", DEFAULT_ANCHOR_YEAR))
        cap = int(cfg.extra.get("page_limit", PAGE_LIMIT))
        blocks = windows(start, end, years=years, anchor=anchor)
        worst = max(
            (b.end - b.start).total_seconds() / cfg.cadence_seconds for b in blocks
        )
        if worst > cap:
            raise ValueError(
                f"extra.window_years={years} asks for up to {worst:.0f} observations per "
                f"request, above the {cap} feature page cap — the response would be "
                f"silently truncated. Narrow the window."
            )
        return [b for b in blocks if b.end > start and b.start < end]

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Download one station-parameter-window page. None if upstream has nothing.

        A 404 is routine — a station can advertise a parameter it never
        reported — and is a skip, not a failure. Everything else (retries,
        `.part` + atomic rename, checksum, rate limiting) belongs to
        `meteopack.core.http`; the import is local so tests can monkeypatch it.
        """
        from meteopack.core.http import NotFoundError, fetch_file

        meta = unit.meta or {}
        dest = (
            Path(raw_root)
            / meta.get("station_id", "unknown")
            / meta.get("parameter", "unknown")
            / meta.get("filename", f"{unit.key.rsplit('/', 1)[-1]}.json")
        )
        # `fetch_file` is a no-op when `dest` exists, which is exactly right for
        # a closed window and exactly wrong for one that is still filling: the
        # first harvest would freeze the rest of the block out forever.
        if meta.get("volatile"):
            dest.unlink(missing_ok=True)
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
            )
        except NotFoundError:
            logger.info("dmi: no observations for %s (404)", unit.key)
            return None

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One metObs page -> tidy frame of `time`, `station_id`, canonical values.

        Features are pivoted onto `(time, station_id)` with `parameterId` as the
        column, which is what turns metObs' one-record-per-value layout back
        into a grid. Only the parameters `cfg.variables` asks for are kept.

        JSON `null` (plus anything in `extra.missing_sentinels`) becomes null
        *before* scale/offset, so a sentinel can never be scaled into a
        plausible-looking number, and the valid range is applied afterwards, so
        it is expressed in canonical units. The range matters more here than for
        most sources: metObs publishes raw, un-QC'd values.
        """
        import polars as pl

        wanted = variables_by_parameter(cfg)
        document = _read_document(Path(raw.path))
        features = _features(document, Path(raw.path).name)
        self._check_overflow(features, cfg, Path(raw.path).name)
        self._check_fingerprint(features, cfg, Path(raw.path).name)

        fallback = (raw.unit_key or "").split("/")
        default_station = station_id(fallback[-3]) if len(fallback) >= 3 else ""

        times: list[datetime] = []
        sids: list[str] = []
        pids: list[str] = []
        values: list[float | None] = []
        for feature in features:
            props = feature.get("properties") or {}
            pid = str(props.get(_PARAMETER) or "")
            if pid not in wanted:
                continue
            when = _timestamp(props.get(_OBSERVED))
            if when is None:
                logger.warning("skipping feature with unparseable %s: %r", _OBSERVED, props)
                continue
            times.append(when)
            sids.append(station_id(props.get(_STATION) or default_station))
            pids.append(pid)
            values.append(_number(props.get(_VALUE)))

        present = [p for p in dict.fromkeys(pids)]
        columns = [wanted[p] for p in sorted(present, key=lambda p: list(wanted).index(p))]
        if not times:
            return _empty_frame(columns)

        long = pl.DataFrame(
            {
                "time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
                "station_id": pl.Series(sids, dtype=pl.String),
                "parameter": pl.Series(pids, dtype=pl.String),
                "value": pl.Series(values, dtype=pl.Float64),
            }
        )
        # `last` rather than a sum or a mean: metObs restates a value by
        # publishing a second feature for the same instant, and the restatement
        # is the one that is correct.
        wide = long.pivot(
            on="parameter", index=["time", "station_id"], values="value", aggregate_function="last"
        ).rename({p: wanted[p].name for p in present})

        out = wide.select(
            ["time", "station_id", *[v.name for v in columns]]
        )
        out = _apply_variables(out, columns, cfg)
        out = out.sort(["time", "station_id"])
        return self._drop_off_cadence(out, cfg)

    def _check_overflow(self, features: Sequence[dict], cfg: ProductConfig, filename: str) -> None:
        """Refuse a document that came back exactly at the page cap.

        A truncated page is indistinguishable from a complete one — same shape,
        same status, just fewer features — so the only tell is hitting the cap
        on the nose. Treating that as data loses the tail of a window silently.
        """
        cap = int(cfg.extra.get("page_limit", PAGE_LIMIT))
        if len(features) >= cap:
            raise PageOverflowError(
                f"{filename}: {len(features)} features is the {cap} page cap, so the window "
                f"was truncated. Lower extra.window_years and re-fetch."
            )

    def _check_fingerprint(
        self, features: Sequence[dict], cfg: ProductConfig, filename: str
    ) -> None:
        """Refuse to parse a record schema the config was not pinned against.

        Only checked when there are features: an empty page carries no property
        names, and failing on it would turn "this station reported nothing in
        2014" into a job failure.
        """
        pinned = cfg.schema_fingerprint
        if not pinned or not features:
            return
        keys = _property_keys(features)
        seen = schema_fingerprint(keys)
        if seen != pinned:
            raise SchemaDriftError(
                f"{filename}: DMI record schema changed: pinned {pinned}, got {seen} "
                f"(properties {sorted(keys)}). Re-run `scan` and re-pin schema_fingerprint "
                f"after checking what moved."
            )

    def _drop_off_cadence(self, out: pl.DataFrame, cfg: ProductConfig) -> pl.DataFrame:
        """Drop observations that are not on the product's cadence grid.

        Dropped rather than snapped, so a cadence mismatch shows up as missing
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
            logger.warning("dropping %d off-cadence observation(s)", n_bad)
            return out.filter(pl.Series(idx >= 0))
        return out

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile]) -> dict:
        """Report every distinct record schema across `raw_files`.

        Upstream changes the feature properties without announcing it, so the
        pipeline pins the fingerprint this returns and `parse` refuses to run
        against anything else. The report is deliberately a plain dict: it is
        meant to be written to JSON next to the store and diffed between runs.
        """
        variants: dict[tuple[str, ...], dict] = {}
        errors: list[dict[str, str]] = []
        parameters: set[str] = set()
        for raw in raw_files:
            path = Path(raw.path)
            try:
                features = _features(_read_document(path), path.name)
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            if not features:
                continue
            keys = tuple(sorted(_property_keys(features)))
            seen = {str((f.get("properties") or {}).get(_PARAMETER) or "") for f in features}
            parameters.update(p for p in seen if p)
            entry = variants.setdefault(
                keys,
                {
                    "properties": list(keys),
                    "dtypes": _infer_dtypes(features),
                    "fingerprint": schema_fingerprint(keys),
                    "parameters": [],
                    "count": 0,
                    "examples": [],
                },
            )
            entry["count"] += 1
            for pid in sorted(seen):
                if pid and pid not in entry["parameters"]:
                    entry["parameters"].append(pid)
            if len(entry["examples"]) < 3:
                entry["examples"].append(path.name)

        ordered = sorted(variants.values(), key=lambda e: e["fingerprint"])
        fingerprints = [e["fingerprint"] for e in ordered]
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": fingerprints,
            "parameters": sorted(parameters),
            "fingerprint": fingerprints[0] if len(fingerprints) == 1 else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ------------------------------------------------------------ document helpers


def _instant(value: datetime) -> str:
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_document(path: Path) -> Any:
    try:
        return json.loads(path.read_bytes().decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaDriftError(f"{path.name}: not valid JSON ({exc})") from None


def _features(document: Any, filename: str) -> list[dict]:
    if not isinstance(document, Mapping) or "features" not in document:
        raise SchemaDriftError(
            f"{filename}: expected a GeoJSON FeatureCollection with a 'features' list"
        )
    return [f for f in (document.get("features") or []) if isinstance(f, dict)]


def _property_keys(features: Iterable[Mapping[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for feature in features:
        keys.update((feature.get("properties") or {}).keys())
    return keys


def _infer_dtypes(features: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """Best-effort JSON type per property, from the first feature that has it."""
    out: dict[str, str] = {}
    for feature in features:
        for key, value in (feature.get("properties") or {}).items():
            if key not in out or out[key] == "null":
                out[key] = type(value).__name__ if value is not None else "null"
    return out


def _timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        # 3.11+ fromisoformat accepts the trailing 'Z' metObs actually writes.
        when = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return when.replace(tzinfo=UTC) if when.tzinfo is None else when.astimezone(UTC)


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _empty_frame(variables: Sequence[Variable]) -> pl.DataFrame:
    import polars as pl

    schema: dict[str, Any] = {"time": pl.Datetime("us", "UTC"), "station_id": pl.String}
    for var in variables:
        schema[var.name] = _polars_dtype(var.dtype)
    return pl.DataFrame(schema=schema)


def _apply_variables(
    frame: pl.DataFrame, variables: Sequence[Variable], cfg: ProductConfig
) -> pl.DataFrame:
    """Sentinel -> null, then scale/offset, then valid-range mask, then dtype.

    metObs expresses missing as JSON `null`, which `_number` has already turned
    into a null. `extra.missing_sentinels` is for the day a numeric sentinel
    comes through instead; a -999 that gets scaled first is indistinguishable
    from data.
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
        raise ValueError(f"unsupported dtype {name!r} for a DMI variable") from None
