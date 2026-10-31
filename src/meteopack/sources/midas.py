"""Met Office MIDAS Open UK hourly weather observations, via the CEDA archive.

Two hosts, two different jobs::

    https://data.ceda.ac.uk/<path>?json    # catalogue listings, keyless
    https://dap.ceda.ac.uk/<path>          # the files, Bearer token required

and one layout::

    dataset-version-<ver>/<county>/<srcid>_<name>/qc-version-1/
        midas-open_<dataset>_dv-<ver>_<county>_<srcid>_<name>_qcv-1_<year>.csv

The files are BADC-CSV: a long header of `<attribute>,<scope>,<value>...` lines,
a line reading `data`, a column-header row, the rows, and a line reading
`end data`.

Five pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store, the ledger or the credentials if you get it wrong:

* **This is the only source here that needs a credential, and it must never
  enter the repo.** The bearer token is read from a file outside the tree
  (`extra.token_file`, default `~/.ceda_token`, overridable with
  `$CEDA_TOKEN_FILE`), never from config, never logged, never written back. A
  CEDA token expires after about three days, so a stale one is the normal
  failure, not the exotic one.
* **A rejected token does not reliably look like an error.** `dap.ceda.ac.uk`
  answers an expired token two different ways, both seen live on 2026-10-31: a
  plain 401, and a 302 to a Keycloak login page that `requests` follows to a
  *200 with an HTML body*. The second one is the dangerous one — taken at face
  value it is a successful download of a station-year that then fails to
  parse, repeated for every file in the job. So every payload is checked for
  the BADC-CSV signature as well as the status, and both routes raise one
  `CedaAuthError` naming the token file.
* **Units are not canonical and the header says so.** MIDAS stores wind speed
  and gusts in knots, and visibility, cloud base and vertical visibility in
  decametres — each file's own `long_name,<column>,<description>,<units>` line
  declares it. Conversion lives in the per-variable `scale` in config, and
  `extra.expected_units` pins what the header must say, so the day upstream
  switches wind to m/s the job stops instead of quietly halving every wind
  speed.
* **Rows repeat per `ob_time`.** A station can carry several versions of one
  observation, distinguished by `version_num` (0 = as received, 1 = current
  best). `parse` keeps the highest and drops the rest; keeping the first would
  pick the un-QCed message about half the time.
* **A published dataset version is immutable, so the version belongs in the
  unit key.** `extra.dataset_version: latest` is resolved to a concrete
  `dataset-version-<ver>` *before* the key is built. That is what makes a key
  stable (a re-run of the same release is the same key, and skippable) and
  distinct (a new release is new work) without any unit needing to be volatile.

The upstream schema drifts without announcement, so `scan` reports every column
set and every declared unit it sees, and `parse` refuses to run against a
pinned fingerprint it does not match.
"""
from __future__ import annotations

import csv
import hashlib
import io
import logging
import os
import stat
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from meteopack.core import grids
from meteopack.core.config import ProductConfig, RateLimit, SourceConfig, Variable
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, TimeWindow, Unit

if TYPE_CHECKING:  # keep polars off the import path of `meteopack.sources`
    import polars as pl

logger = logging.getLogger(__name__)

#: Catalogue listings (`?json`). Keyless, which is why `list` can enumerate the
#: tree before anybody has a token.
CATALOGUE_BASE = "https://data.ceda.ac.uk"

#: File downloads. Bearer token required.
DOWNLOAD_BASE = "https://dap.ceda.ac.uk"

ARCHIVE_ROOT = "/badc/ukmo-midas-open/data"
DATASET = "uk-hourly-weather-obs"
COLLECTION = "midas-open"

#: Latest release at the time of writing; `extra.dataset_version: latest`
#: resolves against the live listing instead.
DEFAULT_VERSION = "202607"

#: The quality-controlled view. `qc-version-0` is the raw received message.
DEFAULT_QC_VERSION = 1

#: Unit conversions MIDAS needs. Used by config, not by code — they are here so
#: a product YAML's `scale` can be checked against a named constant.
KNOT = 0.514444      # knots -> m s-1
DECAMETRE = 10.0     # decametres -> m

#: BADC-CSV missing-data flag, per each file's own `missing_value,G,NA` line.
MISSING_TOKENS: tuple[str, ...] = ("NA", "na", "N/A", "")

#: Where the token file is named, in precedence order.
TOKEN_ENV = "CEDA_TOKEN_FILE"
DEFAULT_TOKEN_FILE = "~/.ceda_token"

_TIME_COL = "ob_time"
_VERSION_COL = "version_num"
_ID_COL = "src_id"
_STRUCTURAL = frozenset({_TIME_COL, _VERSION_COL, _ID_COL, "id", "id_type", "met_domain_name"})

#: MIDAS writes naive timestamps and documents them as UTC.
_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

_DATA_MARKER = "data"
_END_MARKER = "end data"
_BADC_SIGNATURE = "Conventions"

#: What a Keycloak login page looks like once `requests` has followed the 302.
_LOGIN_MARKERS = ("<!doctype html", "<html", "unauthenticated", "kccontext", "keycloak")

#: Statuses that mean "your token, not your request".
_AUTH_STATUS = frozenset({401, 403})

_VERSION_PREFIX = "dataset-version-"
_QC_PREFIX = "qc-version-"


class SchemaDriftError(RuntimeError):
    """A BADC-CSV does not have the shape, columns or units this reader expects."""


class CedaAuthError(RuntimeError):
    """The CEDA token is missing, empty or rejected. Always names the file."""


# ----------------------------------------------------------------- credentials


def token_file(source: SourceConfig | None = None) -> Path:
    """Where the bearer token lives: `$CEDA_TOKEN_FILE`, config, then the default.

    The environment wins so an operator can point one job at a fresh token
    without editing a shared config, and the token itself is never a config
    value — only the path to it is.
    """
    extra = (source.extra if source else {}) or {}
    for candidate in (os.environ.get(TOKEN_ENV), extra.get("token_file"), DEFAULT_TOKEN_FILE):
        if candidate:
            return Path(str(candidate)).expanduser()
    return Path(DEFAULT_TOKEN_FILE).expanduser()


def read_token(path: Path) -> str:
    """The bearer token, with an actionable message when there isn't one.

    The message names the file and says what to do, because the overwhelmingly
    common case is a token that simply expired: CEDA issues them for about
    three days.
    """
    path = Path(path).expanduser()
    try:
        text = path.read_text()
    except FileNotFoundError:
        raise CedaAuthError(
            f"no CEDA token at {path}. MIDAS Open is the one source here that needs a "
            f"credential: get a token from https://services.ceda.ac.uk/cedasite/myceda/user/ "
            f"and write it to {path} (chmod 600). Tokens expire after about three days, so "
            f"this file needs refreshing, not committing. Set ${TOKEN_ENV} to use another path."
        ) from None
    except OSError as exc:
        raise CedaAuthError(f"cannot read the CEDA token at {path}: {exc}") from None
    token = text.strip()
    if not token:
        raise CedaAuthError(
            f"the CEDA token file {path} is empty; paste a current token into it "
            f"(tokens expire after about three days)"
        )
    try:
        mode = path.stat().st_mode
    except OSError:  # pragma: no cover - stat right after a successful read
        mode = 0
    if mode & (stat.S_IRGRP | stat.S_IROTH):
        logger.warning("CEDA token file %s is readable by others; chmod 600 it", path)
    return token


# ------------------------------------------------------------------ BADC-CSV


def auth_error(where: str, detail: str, path: Path) -> CedaAuthError:
    """One message for every way CEDA says no, always naming the token file."""
    return CedaAuthError(
        f"{where}: {detail}. The CEDA token in {path} is expired or rejected — they last "
        f"about three days. Refresh it from "
        f"https://services.ceda.ac.uk/cedasite/myceda/user/ and write it to that file "
        f"(chmod 600), or point ${TOKEN_ENV} at another one."
    )


def check_payload(text: str, *, where: str, path: Path) -> None:
    """Raise if `text` is not a BADC-CSV, distinguishing a login page from drift.

    This is the defence against the quieter half of the failure in the module
    docstring: a rejected token sometimes answers 401, which is loud, and
    sometimes redirects to Keycloak and answers HTTP 200 with an HTML login
    page, which is indistinguishable from a good download until somebody looks
    at the bytes. Both were observed on 2026-10-31.
    """
    head = text[:4096].lstrip()
    if head.startswith(_BADC_SIGNATURE):
        return
    lowered = head.lower()
    if any(marker in lowered for marker in _LOGIN_MARKERS):
        raise auth_error(
            where,
            "CEDA returned a login page instead of data (dap.ceda.ac.uk redirects to "
            "Keycloak and answers HTTP 200, so this never looks like an error)",
            path,
        )
    raise SchemaDriftError(
        f"{where}: not a BADC-CSV — expected a first line starting {_BADC_SIGNATURE!r}, "
        f"got {head[:80]!r}"
    )


def split_badc(text: str) -> tuple[list[str], list[str]]:
    """`(header lines, table lines)` of a BADC-CSV.

    The table is everything between the line reading `data` and the line
    reading `end data`; its first entry is the column-header row. A file whose
    `end data` is missing (a truncated download) is read to the end rather than
    refused, and the truncation shows up as a short row count.
    """
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().lower() == _DATA_MARKER)
    except StopIteration:
        raise SchemaDriftError(
            f"no {_DATA_MARKER!r} marker: this is not a BADC-CSV table"
        ) from None
    end = next(
        (i for i, line in enumerate(lines) if line.strip().lower() == _END_MARKER), len(lines)
    )
    table = [line for line in lines[start + 1 : end] if line.strip()]
    return lines[:start], table


def badc_columns(table: Sequence[str]) -> list[str]:
    """Column names from the row that follows the `data` marker."""
    if not table:
        return []
    return [c.strip() for c in next(csv.reader([table[0]]), [])]


def declared_units(header: Sequence[str]) -> dict[str, str]:
    """`{column: units}` from the header's `long_name,<column>,<desc>,<units>` lines.

    This is how a unit change announces itself: MIDAS writes the units of every
    column into every file, so `parse` can check them instead of trusting that
    knots are still knots.
    """
    out: dict[str, str] = {}
    for row in csv.reader(header):
        if len(row) >= 4 and row[0].strip() == "long_name":
            out[row[1].strip()] = row[-1].strip()
    return out


def schema_fingerprint(columns: Iterable[str]) -> str:
    """Stable hash of one normalised BADC column set.

    Order-independent and whitespace-insensitive, so it tracks *what* a file
    contains; a genuine addition, removal or rename does move it. Verified
    constant across 400 files of the existing archive.
    """
    normalised = sorted({c.strip() for c in columns} - {""})
    return hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()[:16]


def units_fingerprint(units: Mapping[str, str]) -> str:
    """Stable hash of a declared-units map, for pinning a whole product at once."""
    joined = "|".join(f"{k}={units[k]}" for k in sorted(units))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# -------------------------------------------------------------------- stations


@dataclass(frozen=True, slots=True)
class StationRecord:
    """One row of `..._station-metadata.csv`, plus where its files live."""

    station_id: str
    name: str
    file_name: str
    lon: float
    lat: float
    height_m: float | None = None
    first_year: int | None = None
    last_year: int | None = None
    county: str = ""

    @property
    def src_id(self) -> int:
        return int(self.station_id)

    @property
    def directory(self) -> str:
        """`00160_craibstone` — the per-station directory in the county tree."""
        return f"{self.station_id}_{self.file_name}"

    def years(self, first: int, last: int) -> range:
        """Years to request, clipped to the station's own reporting period."""
        lo = max(first, self.first_year or first)
        hi = min(last, self.last_year or last)
        return range(lo, hi + 1)


def station_id(value: str | int) -> str:
    """Canonical five-digit MIDAS src_id. The archive paths zero-pad; CSVs do not."""
    return f"{int(value):05d}"


def parse_station_metadata(text: str) -> list[StationRecord]:
    """Parse `..._station-metadata.csv`.

    A row without usable coordinates or src_id is skipped with a warning rather
    than raising: one malformed row must not take out a whole network.
    """
    _, table = split_badc(text)
    columns = badc_columns(table)
    records: list[StationRecord] = []
    for row in csv.reader(table[1:]):
        if not row:
            continue
        item = dict(zip(columns, (c.strip() for c in row), strict=False))
        try:
            sid = station_id(item["src_id"])
            lon = float(item["station_longitude"])
            lat = float(item["station_latitude"])
        except (KeyError, TypeError, ValueError):
            logger.warning("skipping unparseable MIDAS station row: %r", str(row)[:120])
            continue
        records.append(
            StationRecord(
                station_id=sid,
                name=str(item.get("station_name", "")),
                file_name=str(item.get("station_file_name", "")),
                lon=lon,
                lat=lat,
                height_m=_number(item.get("station_elevation")),
                first_year=_integer(item.get("first_year")),
                last_year=_integer(item.get("last_year")),
            )
        )
    records.sort(key=lambda r: r.station_id)
    return records


def _number(value: Any) -> float | None:
    text = str(value).strip() if value is not None else ""
    if text in MISSING_TOKENS:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _integer(value: Any) -> int | None:
    found = _number(value)
    return None if found is None else int(found)


def _split_source_key(key: str) -> str:
    """`"air_temperature"` -> `"air_temperature"`, with a useful error otherwise."""
    column = key.strip()
    if not column:
        raise ValueError("source_key must be a MIDAS column name, e.g. 'air_temperature'")
    if column in _STRUCTURAL:
        raise ValueError(
            f"source_key {key!r} names a structural column; MIDAS variables are observation "
            f"columns such as 'air_temperature' or 'wind_speed'"
        )
    return column


def variables_for(cfg: ProductConfig) -> list[tuple[str, Variable]]:
    """`[(MIDAS column, variable), ...]` for everything the config wants."""
    return [(_split_source_key(var.source_key), var) for var in cfg.variables]


def unit_key(cfg: ProductConfig, version: str, sid: str, year: int) -> str:
    """`<source>/<product>/dv-<version>/<srcid>/<year>`.

    The resolved dataset version is in the key because a MIDAS Open release is
    immutable: within `dv-202607` a station-year never changes, and a new
    release is genuinely new work rather than the same unit again.
    """
    return f"{cfg.source}/{cfg.product}/dv-{version}/{sid}/{year}"


class MidasOpen:
    """MIDAS Open UK hourly weather observations as a `meteopack` source.

    Public API::

        src = MidasOpen(source_cfg, session=..., limiter=...)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain stations
        src.list(window, product_cfg)    -> list[Unit]        (one per station-year)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == upstream 404)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        MidasOpen.schema_fingerprint(columns) -> str

    `limiter` is whatever `meteopack.core.http` expects; it is passed straight
    through to `fetch_file`. Catalogue listings use `requests` directly and need
    no token; station metadata and the data files do.
    """

    name = "midas"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="midas", base_url=DOWNLOAD_BASE)
        self.base_url = (self.source.base_url or DOWNLOAD_BASE).rstrip("/")
        self.catalogue_url = str(
            self.source.extra.get("catalogue_url", CATALOGUE_BASE)
        ).rstrip("/")
        self.dataset = str(self.source.extra.get("dataset", DATASET))
        self.token_file = token_file(self.source)
        self._session = session
        self._limiter = limiter
        self._now = now
        self._listing_cache: dict[str, list[dict]] = {}
        self._metadata_cache: dict[str, list[StationRecord]] = {}
        self._county_cache: dict[str, dict[str, tuple[str, str]]] = {}
        self._version_cache: dict[str, str] = {}

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
        """A session carrying the bearer token, created on first use.

        Lazy because listing the catalogue needs no credential at all: a source
        built with no session must still work, and must not demand a token
        before anything has actually asked for data.
        """
        if self._session is None:
            import requests

            self._session = requests.Session()
            agent = self.source.extra.get("user_agent", "meteopack/0.1 (+midas-ceda)")
            self._session.headers["User-Agent"] = str(agent)
            # The token is read here and nowhere else, and never logged.
            self._session.headers["Authorization"] = f"Bearer {read_token(self.token_file)}"
        return self._session

    # ------------------------------------------------------------- remote reads

    def _get_json(self, url: str) -> Any:
        """One keyless GET of a `?json` catalogue listing. Seam for tests."""
        import requests

        resp = requests.get(url, params={"json": ""}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def _get_text(self, url: str) -> str:
        """One authenticated GET, decoded. Seam for tests.

        A 401/403 here is a rejected token, not a transport error, so it is
        translated rather than surfaced as a bare `requests.HTTPError` that
        says nothing about which file to refresh.
        """
        resp = self._requests_session().get(url, timeout=self.timeout)
        if resp.status_code in _AUTH_STATUS:
            raise auth_error(url, f"CEDA answered HTTP {resp.status_code}", self.token_file)
        resp.raise_for_status()
        return resp.text

    def listing(self, path: str) -> list[dict]:
        """One catalogue directory listing (cached per instance)."""
        if path in self._listing_cache:
            return self._listing_cache[path]
        try:
            payload = self._get_json(f"{self.catalogue_url}{path}")
            items = [i for i in (payload or {}).get("items") or [] if isinstance(i, dict)]
        except Exception as exc:  # noqa: BLE001 - a missing subdir is not fatal
            logger.warning("CEDA listing failed for %s: %s", path, exc)
            items = []
        self._listing_cache[path] = items
        return items

    # -------------------------------------------------------------------- paths

    def dataset_version(self, cfg: ProductConfig | None = None) -> str:
        """The release to read, with `latest` resolved against the live listing.

        Resolving before the unit key is built is what keeps the key immutable;
        see `unit_key`.
        """
        wanted = str(
            (cfg.extra.get("dataset_version") if cfg else None)
            or self.source.extra.get("dataset_version")
            or DEFAULT_VERSION
        )
        if wanted != "latest":
            return wanted
        if "latest" in self._version_cache:
            return self._version_cache["latest"]
        names = sorted(
            i["name"][len(_VERSION_PREFIX) :]
            for i in self.listing(f"{ARCHIVE_ROOT}/{self.dataset}")
            if i.get("type") == "dir" and str(i.get("name", "")).startswith(_VERSION_PREFIX)
        )
        if not names:
            raise RuntimeError(
                f"no {_VERSION_PREFIX}* directories under {ARCHIVE_ROOT}/{self.dataset}; "
                f"cannot resolve dataset_version: latest"
            )
        self._version_cache["latest"] = names[-1]
        logger.info("midas: dataset_version latest -> %s", names[-1])
        return names[-1]

    def version_root(self, version: str) -> str:
        return f"{ARCHIVE_ROOT}/{self.dataset}/{_VERSION_PREFIX}{version}"

    def metadata_url(self, version: str) -> str:
        root = self.version_root(version)
        return (
            f"{self.base_url}{root}/{COLLECTION}_{self.dataset}_dv-{version}"
            f"_station-metadata.csv"
        )

    def file_url(self, version: str, rec: StationRecord, year: int, qc: int) -> str:
        root = self.version_root(version)
        # The directory says `qc-version-1`, the filename says `qcv-1`.
        name = (
            f"{COLLECTION}_{self.dataset}_dv-{version}_{rec.county}_{rec.directory}"
            f"_qcv-{qc}_{year}.csv"
        )
        return f"{self.base_url}{root}/{rec.county}/{rec.directory}/{_QC_PREFIX}{qc}/{name}"

    # ----------------------------------------------------------------- stations

    def metadata(self, version: str) -> list[StationRecord]:
        """The whole network from `..._station-metadata.csv` (cached per instance).

        This is the first call that needs a token, so a stale credential is
        reported here rather than 400 downloads later.
        """
        if version in self._metadata_cache:
            return self._metadata_cache[version]
        url = self.metadata_url(version)
        text = self._get_text(url)
        check_payload(text, where=url, path=self.token_file)
        records = parse_station_metadata(text)
        self._metadata_cache[version] = records
        return records

    def county_index(self, version: str, wanted: set[str] | None = None) -> dict[str, tuple[str, str]]:
        """`{station_id: (county, directory)}` from the keyless catalogue listing.

        The directory name is authoritative for both the county and the station
        file name, because it is exactly what the download URL has to contain.
        Listing stops as soon as every wanted station has been placed, which for
        a small domain is a handful of counties rather than all 109.
        """
        cache_key = version
        if cache_key in self._county_cache and wanted is None:
            return self._county_cache[cache_key]
        root = self.version_root(version)
        found: dict[str, tuple[str, str]] = {}
        counties = [
            str(i["name"]) for i in self.listing(root) if i.get("type") == "dir"
        ]
        for county in counties:
            for item in self.listing(f"{root}/{county}"):
                if item.get("type") != "dir":
                    continue
                name = str(item.get("name", ""))
                head = name.partition("_")[0]
                try:
                    sid = station_id(head)
                except ValueError:
                    continue
                if wanted is not None and sid not in wanted:
                    continue
                found[sid] = (county, name)
            if wanted is not None and len(found) == len(wanted):
                break
        if wanted is None:
            self._county_cache[cache_key] = found
        return found

    def stations(self, cfg: ProductConfig) -> pl.DataFrame:
        """In-domain stations, with coordinates, for `pack` to write beside the data.

        Clipping is the grid's job when `cfg.grid_id` is set; the UK sits
        outside every grid in this repo's registry, so the shipped product
        clips with `extra.bbox` instead. With neither, the whole network is
        kept — a deliberate escape hatch rather than a default.
        """
        import polars as pl

        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        bbox = self._bbox(cfg)
        rows = []
        for rec in self.metadata(self.dataset_version(cfg)):
            if grid is not None and not grid.contains(rec.lon, rec.lat):
                continue
            if bbox is not None and not _in_bbox(bbox, rec.lon, rec.lat):
                continue
            rows.append(
                {
                    "station_id": rec.station_id,
                    "lat": rec.lat,
                    "lon": rec.lon,
                    "height_m": rec.height_m,
                    "name": rec.name,
                    "first_year": rec.first_year,
                    "last_year": rec.last_year,
                }
            )
        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "height_m": pl.Float64,
            "name": pl.String,
            "first_year": pl.Int32,
            "last_year": pl.Int32,
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

        A station-year that was never published simply 404s at fetch time, so
        years are bounded by the station's own `first_year`/`last_year` rather
        than by listing every station directory.
        """
        start, end = self._window_bounds(window)
        # Validate the window against the product cadence in one place.
        Timeline(start, end, cfg.cadence)

        version = self.dataset_version(cfg)
        qc = int(cfg.extra.get("qc_version", DEFAULT_QC_VERSION))
        table = self.stations(cfg)
        keep = set(table["station_id"].to_list())
        only = cfg.extra.get("src_ids")
        if only:
            keep &= {station_id(s) for s in only}
        limit = cfg.extra.get("max_stations")
        if limit:
            keep = set(sorted(keep)[: int(limit)])
        if not keep:
            logger.warning("no in-domain MIDAS stations for grid %s", cfg.grid_id)
            return []

        placed = self.county_index(version, keep)
        missing = sorted(keep - set(placed))
        if missing:
            logger.warning(
                "midas: %d in-domain station(s) have no directory in dv-%s (e.g. %s); skipping",
                len(missing), version, ", ".join(missing[:5]),
            )

        # `end` is exclusive, so a window ending exactly at midnight on 1 Jan
        # must not pull in the whole of that year.
        last_year = (end - timedelta(microseconds=1)).year
        units: list[Unit] = []
        for rec in self.metadata(version):
            place = placed.get(rec.station_id)
            if place is None:
                continue
            county, directory = place
            located = StationRecord(
                station_id=rec.station_id,
                name=rec.name,
                file_name=directory.partition("_")[2],
                lon=rec.lon,
                lat=rec.lat,
                height_m=rec.height_m,
                first_year=rec.first_year,
                last_year=rec.last_year,
                county=county,
            )
            for year in located.years(start.year, last_year):
                units.append(
                    Unit(
                        key=unit_key(cfg, version, rec.station_id, year),
                        url=self.file_url(version, located, year, qc),
                        valid_time=datetime(year, 1, 1, tzinfo=UTC),
                        meta={
                            "station_id": rec.station_id,
                            "county": county,
                            "year": str(year),
                            "dataset_version": version,
                            "qc_version": str(qc),
                            "filename": f"{year}.csv",
                            # A published release is frozen, and the key carries
                            # the resolved version, so nothing here is volatile.
                            "volatile": False,
                        },
                    )
                )
        logger.info(
            "midas: %d units across %d station(s) for %d..%d in dv-%s",
            len(units), len(placed), start.year, last_year, version,
        )
        return units

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Download one station-year BADC-CSV. None if that year was not published.

        A rejected token shows up two ways and both are translated into one
        actionable message: an outright 401/403, and a 200 whose body is the
        Keycloak login page — see the module docstring. The downloaded bytes
        are therefore checked for the BADC-CSV signature, and a rejected
        payload is deleted so a later run cannot mistake it for a finished
        download.
        """
        from meteopack.core.http import FetchError, NotFoundError, fetch_file

        meta = unit.meta or {}
        dest = (
            Path(raw_root)
            / meta.get("station_id", unit.key.rsplit("/", 2)[-2])
            / meta.get("filename", f"{unit.key.rsplit('/', 1)[-1]}.csv")
        )
        rl = self.rate_limit
        try:
            raw = fetch_file(
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
            logger.info("midas: no file for %s (404)", unit.key)
            return None
        except FetchError as exc:
            if exc.status in _AUTH_STATUS:
                raise auth_error(
                    unit.url, f"CEDA answered HTTP {exc.status}", self.token_file
                ) from None
            raise
        try:
            check_payload(_head(Path(raw.path)), where=unit.url, path=self.token_file)
        except (CedaAuthError, SchemaDriftError):
            Path(raw.path).unlink(missing_ok=True)
            raise
        return raw

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One station-year BADC-CSV -> tidy frame of `time`, `station_id`, values.

        Only the columns `cfg.variables` asks for are read. `NA` becomes null
        *before* scale/offset, so the missing flag can never be scaled into a
        plausible-looking number, and the valid range is applied afterwards, so
        it is expressed in canonical units (m/s, not knots; m, not decametres).
        Repeated `ob_time` rows are reduced to the highest `version_num`.
        """
        import polars as pl

        wanted = variables_for(cfg)
        text = Path(raw.path).read_text(encoding="utf-8", errors="replace")
        check_payload(text, where=str(raw.path), path=self.token_file)
        header, table = split_badc(text)
        columns = badc_columns(table)
        self._check_fingerprint(columns, cfg, Path(raw.path).name)
        self._check_units(declared_units(header), cfg, Path(raw.path).name)
        if not wanted or len(table) < 2:
            return _empty_frame(wanted)

        present = set(columns)
        missing = [col for col, _ in wanted if col not in present]
        if missing:
            raise SchemaDriftError(
                f"{Path(raw.path).name}: missing column(s) {missing}; present: {sorted(present)}"
            )
        if _TIME_COL not in present:
            raise SchemaDriftError(f"{Path(raw.path).name}: no {_TIME_COL} column")

        select = [c for c in (_TIME_COL, _VERSION_COL, _ID_COL) if c in present]
        select += [col for col, _ in wanted if col not in select]
        body = "\n".join(table).encode("utf-8")
        frame = pl.read_csv(io.BytesIO(body), infer_schema=False, columns=select)

        if _ID_COL in present:
            # The file's own src_id wins, so a CSV found on disk parses
            # correctly whatever it is called.
            sid = (
                pl.col(_ID_COL).str.strip_chars().cast(pl.Int64, strict=False).cast(pl.String)
                .str.zfill(5)
            )
        else:
            sid = pl.lit(self._station_of(raw), dtype=pl.String)
        version = (
            pl.col(_VERSION_COL).str.strip_chars().cast(pl.Int64, strict=False)
            if _VERSION_COL in present
            else pl.lit(0, dtype=pl.Int64)
        )
        out = frame.select(
            pl.col(_TIME_COL)
            .str.strip_chars()
            .str.strptime(pl.Datetime("us"), _TIME_FORMAT, strict=False)
            .dt.replace_time_zone("UTC")
            .alias("time"),
            sid.alias("station_id"),
            version.alias("_version_num"),
            *[
                # Missing first: `NA` must never reach the cast, let alone the
                # knots-to-m/s scale.
                pl.when(pl.col(col).str.strip_chars().is_in(MISSING_TOKENS))
                .then(None)
                .otherwise(pl.col(col).str.strip_chars())
                .cast(pl.Float64, strict=False)
                .alias(var.name)
                for col, var in wanted
            ],
        ).drop_nulls("time")

        # Highest version_num wins: 0 is the message as received, 1 the current
        # best. Keeping the first would pick the un-QCed row about half the time.
        out = (
            out.sort(["time", "station_id", "_version_num"])
            .unique(subset=["time", "station_id"], keep="last", maintain_order=True)
            .drop("_version_num")
        )
        out = _apply_variables(out, [var for _, var in wanted], cfg)
        out = out.sort(["time", "station_id"])
        return self._drop_off_cadence(out, cfg)

    def _station_of(self, raw: RawFile) -> str:
        """Fallback station id from the unit key, for a file with no `src_id`."""
        parts = (raw.unit_key or "").split("/")
        if len(parts) >= 2:
            try:
                return station_id(parts[-2])
            except ValueError:
                pass
        raise SchemaDriftError(
            f"{Path(raw.path).name}: no {_ID_COL} column and the unit key {raw.unit_key!r} "
            f"carries no station id"
        )

    def _check_fingerprint(
        self, columns: Sequence[str], cfg: ProductConfig, filename: str
    ) -> None:
        pinned = cfg.schema_fingerprint
        if not pinned:
            return
        seen = schema_fingerprint(columns)
        if seen != pinned:
            raise SchemaDriftError(
                f"{filename}: MIDAS schema changed: pinned {pinned}, got {seen}. Re-run "
                f"`scan` and re-pin schema_fingerprint after checking what moved."
            )

    def _check_units(
        self, units: Mapping[str, str], cfg: ProductConfig, filename: str
    ) -> None:
        """Refuse to parse a file whose header declares units we do not scale for.

        Every `scale` in the product config encodes an assumption about the
        source units — knots, decametres, hours. The file states those units,
        so the assumption is checkable, and a silent switch upstream becomes a
        stopped job rather than a series that is wrong by a factor of two.
        """
        expected = cfg.extra.get("expected_units") or {}
        wrong = {
            col: (want, units.get(col))
            for col, want in expected.items()
            if col in units and units[col] != want
        }
        if wrong:
            detail = ", ".join(f"{c}: expected {w!r}, file says {g!r}" for c, (w, g) in wrong.items())
            raise SchemaDriftError(
                f"{filename}: MIDAS declared units changed ({detail}). The per-variable `scale` "
                f"in the product config assumes the pinned units; fix both together."
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
            logger.warning("midas: dropping %d off-cadence observation(s)", n_bad)
            return out.filter(pl.Series(idx >= 0))
        return out

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile]) -> dict:
        """Report every distinct column set and declared-unit map across `raw_files`.

        The units are in the report because they are the thing most likely to
        move without the columns moving, and the one that corrupts data rather
        than crashing. The report is deliberately a plain dict: it is meant to
        be written to JSON next to the store and diffed between runs.
        """
        variants: dict[tuple[str, ...], dict] = {}
        errors: list[dict[str, str]] = []
        units_seen: dict[str, set[str]] = {}
        for raw in raw_files:
            path = Path(raw.path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                check_payload(text, where=str(path), path=self.token_file)
                header, table = split_badc(text)
                columns = tuple(badc_columns(table))
                units = declared_units(header)
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            for column, value in units.items():
                units_seen.setdefault(column, set()).add(value)
            entry = variants.setdefault(
                columns,
                {
                    "columns": list(columns),
                    "fingerprint": schema_fingerprint(columns),
                    "n_rows": 0,
                    "count": 0,
                    "examples": [],
                },
            )
            entry["count"] += 1
            entry["n_rows"] += max(len(table) - 1, 0)
            if len(entry["examples"]) < 3:
                entry["examples"].append(path.name)

        ordered = sorted(variants.values(), key=lambda e: e["fingerprint"])
        stable = {c: min(v) for c, v in units_seen.items() if len(v) == 1}
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": [e["fingerprint"] for e in ordered],
            "fingerprint": ordered[0]["fingerprint"] if len(ordered) == 1 else None,
            "declared_units": {c: sorted(v) for c, v in sorted(units_seen.items())},
            "units_fingerprint": units_fingerprint(stable) if stable else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ----------------------------------------------------------------- file helpers


def _head(path: Path, n: int = 4096) -> str:
    with Path(path).open("rb") as f:
        return f.read(n).decode("utf-8", errors="replace")


def _in_bbox(bbox: tuple[float, float, float, float], lon: float, lat: float) -> bool:
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


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

    `NA` is already null by this point. `extra.missing_sentinels` exists for the
    numeric sentinels some MIDAS code columns use; applying them before the
    scale is what keeps a -99 from becoming a believable wind speed.
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
        raise ValueError(f"unsupported dtype {name!r} for a MIDAS variable") from None
