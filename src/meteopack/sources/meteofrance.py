"""Météo-France climatological station observations (BASE, per-département CSV).

The historical archive is keyless and lives as gzipped, `;`-separated CSV in a
public OVH S3 bucket, one file per département per period::

    <base>/data/synchro_ftp/BASE/<subdir>/<prefix>_<dep>_<period>.csv.gz
    <base>/?list-type=2&prefix=data/synchro_ftp/BASE/<subdir>/<prefix>_<dep>_

Only the near-real-time API needs a token; none of this does.

Five pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store, or the ledger, if you get it wrong:

* **The `latest-*` files are rewritten in place.** `H_59_latest-2025-2026.csv.gz`
  keeps its name and grows every morning — verified live on 2026-10-31, where
  the published file was six hours old and 18% larger than a copy taken in
  July. A unit key must stay stable (it is the ledger's identity) so the flag
  goes in `meta["volatile"]` instead, which is what stops the ledger filing the
  first download as the final word. `fetch` additionally drops an existing
  volatile file before re-requesting, because `fetch_file` is a deliberate
  no-op when the destination is already there.
* **"latest" has no end year.** The token reads `latest-2025-2026` today and
  will keep reading that into 2027. Taking the token's last year as the file's
  last year would quietly stop listing the current file the moment a window
  crossed into a year the name has not caught up with, so `latest` is treated
  as open-ended.
* **Station coordinates come free, in the data.** Every row carries
  `NUM_POSTE, NOM_USUEL, LAT, LON, ALTI`, so there is no station catalogue to
  fetch and `stations()` derives from whatever has been parsed. See its
  docstring for what happens on a cold start.
* **Which product this is, is configuration.** The hourly suite (`BASE/HOR`,
  prefix `H`, date column `AAAAMMJJHH`) and the 6-minute precipitation-only
  feed (`BASE/MIN`, prefix `MN`, `AAAAMMJJHHMN`) differ only in those three
  strings, the cadence and the variable map, so the second is a product YAML
  rather than a second module.
* **Roughly thirty northern départements overlap the radar footprint**, but
  only roughly: a département is a listing filter, not a domain test. Rows are
  clipped per station with `grids.get(cfg.grid_id).contains(lon, lat)`.

Station ids are Météo-France's eight-digit `NUM_POSTE`, kept as text so the
leading département digits survive.

The CSV header drifts without announcement — columns are added at the end as
new sensors appear — so `scan` reports every column set seen and `parse`
refuses to run against a pinned fingerprint it does not match.
"""
from __future__ import annotations

import gzip
import hashlib
import logging
import math
import re
import tempfile
import xml.etree.ElementTree as ET
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

BASE_URL = "https://meteofrance.s3.sbg.io.cloud.ovh.net"

#: Everything under BASE is the open climatological archive.
BUCKET_PREFIX = "data/synchro_ftp/BASE"

#: Columns every row carries, whatever the product. They are what makes a
#: station catalogue unnecessary.
ID_COL = "NUM_POSTE"
NAME_COL = "NOM_USUEL"
LAT_COL = "LAT"
LON_COL = "LON"
ALT_COL = "ALTI"
META_COLS: tuple[str, ...] = (ID_COL, NAME_COL, LAT_COL, LON_COL, ALT_COL)

#: Period tokens starting with one of these name a file that is rewritten in
#: place, and are therefore open-ended as well as volatile.
VOLATILE_TOKENS: tuple[str, ...] = ("latest",)

#: Départements whose extent intersects the KNMI radar footprint. A generous
#: list: one that contributes no in-domain station simply costs one listing.
DEPARTEMENTS: tuple[str, ...] = (
    "02", "08", "10", "14", "21", "25", "27", "28", "35", "41", "45", "49",
    "50", "51", "52", "53", "54", "55", "57", "59", "60", "61", "62", "67",
    "68", "70", "72", "76", "77", "78", "80", "88", "89", "90", "91", "92",
    "93", "94", "95",
)

#: `<prefix>_<dep>_<period token>.csv.gz`
_FILE_RE = re.compile(r"^(?P<prefix>[A-Z]+)_(?P<dep>\d{2}[AB]?)_(?P<token>.+)\.csv\.gz$")
_YEAR_RE = re.compile(r"(\d{4})")
_S3_NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


#: A département CSV's columns do not match the fingerprint pinned in config.
SchemaDriftError = _SchemaDriftError


def station_id(value: str | int) -> str:
    """Canonical `NUM_POSTE`. Text, so the leading département digits survive."""
    text = str(value).strip()
    if not text:
        raise ValueError("empty Météo-France station id")
    return text


def schema_fingerprint(columns: Iterable[str]) -> str:
    """Stable hash of one normalised CSV column set.

    Normalisation strips padding only. The `Q*` quality columns are included
    deliberately: a quality flag appearing or vanishing is exactly the kind of
    upstream change that should be looked at before the next pack. It is
    order-independent; an addition, removal or rename does move it.
    """
    normalised = sorted({c.strip() for c in columns} - {""})
    digest = hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()
    return digest[:16]


def product_fingerprint(by_group: Mapping[str, str]) -> str:
    """Stable hash of a whole product's per-group fingerprints."""
    joined = "|".join(f"{k}={by_group[k]}" for k in sorted(by_group))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


def variables_by_column(cfg: ProductConfig) -> dict[str, Variable]:
    """`{CSV column: variable}` for everything the config wants."""
    out: dict[str, Variable] = {}
    for var in cfg.variables:
        column = var.source_key.strip()
        if not column:
            raise ValueError(f"variable {var.name!r} has an empty source_key")
        if column in out:
            raise ValueError(
                f"column {column!r} is claimed by both {out[column].name!r} and {var.name!r}"
            )
        out[column] = var
    return out


# -------------------------------------------------------------------- listings


@dataclass(frozen=True, slots=True)
class ArchiveFile:
    """One per-département CSV in the bucket."""

    key: str
    dep: str
    token: str
    size: int
    last_modified: str

    @property
    def filename(self) -> str:
        return self.key.rsplit("/", 1)[-1]

    @property
    def stem(self) -> str:
        return self.filename[: -len(".csv.gz")]

    @property
    def volatile(self) -> bool:
        return self.token.lower().startswith(VOLATILE_TOKENS)

    @property
    def years(self) -> tuple[int, int | None]:
        """First and last year the file covers; `None` means open-ended.

        A `latest-*` token keeps its printed end year past the end of that
        year, so believing the name would stop listing the live file exactly
        when it matters. See the module docstring.
        """
        found = [int(y) for y in _YEAR_RE.findall(self.token)]
        if not found:
            return (0, None)
        return (min(found), None if self.volatile else max(found))

    def covers(self, start: datetime, end: datetime) -> bool:
        """True if the file's period overlaps the half-open window [start, end)."""
        first, last = self.years
        if last is not None and last < start.year:
            return False
        return first <= (end - timedelta(microseconds=1)).year


def parse_listing(xml: str) -> tuple[list[ArchiveFile], str | None]:
    """One ListObjectsV2 page -> (files, continuation token).

    Keys that are not `<prefix>_<dep>_<period>.csv.gz` are skipped rather than
    raising: the bucket also serves documentation and other products, and a
    neighbour's filename convention is not this reader's problem.
    """
    root = ET.fromstring(xml)
    files: list[ArchiveFile] = []
    for contents in root.findall(f"{_S3_NS}Contents"):
        key = (contents.findtext(f"{_S3_NS}Key") or "").strip()
        m = _FILE_RE.match(key.rsplit("/", 1)[-1])
        if not m:
            continue
        files.append(
            ArchiveFile(
                key=key,
                dep=m.group("dep"),
                token=m.group("token"),
                size=int(contents.findtext(f"{_S3_NS}Size") or 0),
                last_modified=(contents.findtext(f"{_S3_NS}LastModified") or "").strip(),
            )
        )
    truncated = (root.findtext(f"{_S3_NS}IsTruncated") or "").strip().lower() == "true"
    token = (root.findtext(f"{_S3_NS}NextContinuationToken") or "").strip()
    return files, (token or None) if truncated else None


def unit_key(cfg: ProductConfig, stem: str) -> str:
    """`<source>/<product>/<file stem>`.

    The stem, not the ETag: a `latest-*` file changes content under an
    unchanged name, and a key that moved with the content would make every
    rewrite look like a brand new unit and the old one look permanently
    pending. Changing content is what `meta["volatile"]` is for.
    """
    return f"{cfg.source}/{cfg.product}/{stem}"


# ------------------------------------------------------------------ the source


class MeteoFranceStations:
    """Météo-France BASE station observations as a `meteopack` source.

    Public API::

        src = MeteoFranceStations(source_cfg, session=..., limiter=..., raw_root=...)
        src.stations(product_cfg)        -> polars.DataFrame of in-domain stations
        src.list(window, product_cfg)    -> list[Unit]        (one per département file)
        src.fetch(unit, raw_root)        -> RawFile | None    (None == upstream 404)
        src.parse(raw_file, product_cfg) -> polars.DataFrame  (time, station_id, vars)
        src.scan(raw_files)              -> dict              (schema drift report)
        MeteoFranceStations.schema_fingerprint(columns) -> str

    `limiter` is whatever `meteopack.core.http` expects; it is passed straight
    through to `fetch_file`. Listing the bucket uses `requests` directly,
    because that is an XML read rather than an archive download and `core.http`
    only owns the latter.

    `raw_root` is optional and only affects `stations()` on a cold start — see
    its docstring.
    """

    name = "meteofrance"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        session=None,
        limiter=None,
        raw_root: Path | str | None = None,
        now: datetime | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="meteofrance", base_url=BASE_URL)
        self.base_url = (self.source.base_url or BASE_URL).rstrip("/")
        self.raw_root = Path(raw_root) if raw_root else None
        self._session = session
        self._limiter = limiter
        self._now = now
        self._listing_cache: dict[str, list[ArchiveFile]] = {}
        # Filled in by `parse`, so the usual `pack` run (parse everything, then
        # write station metadata) needs no extra request at all.
        self._registry: dict[str, dict] = {}

    # ------------------------------------------------------------------ config

    @property
    def rate_limit(self) -> RateLimit:
        return self.source.rate_limit

    @property
    def timeout(self) -> float:
        return float(self.source.extra.get("timeout_seconds", 300))

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

    @staticmethod
    def _subdir(cfg: ProductConfig) -> str:
        return str(cfg.extra.get("s3_subdir", "HOR"))

    @staticmethod
    def _prefix(cfg: ProductConfig) -> str:
        return str(cfg.extra.get("file_prefix", "H"))

    @staticmethod
    def _date_column(cfg: ProductConfig) -> str:
        return str(cfg.extra.get("date_column", "AAAAMMJJHH"))

    @staticmethod
    def _date_format(cfg: ProductConfig) -> str:
        return str(cfg.extra.get("date_format", "%Y%m%d%H"))

    @staticmethod
    def _departements(cfg: ProductConfig) -> list[str]:
        return [str(d) for d in (cfg.extra.get("departements") or DEPARTEMENTS)]

    # ------------------------------------------------------------- remote reads

    def _get_text(self, url: str) -> str:
        """One plain GET, decoded. Overridable seam for tests."""
        resp = self._requests_session().get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.text

    def file_url(self, cfg: ProductConfig, filename: str) -> str:
        return f"{self.base_url}/{BUCKET_PREFIX}/{self._subdir(cfg)}/{filename}"

    def listing_url(self, cfg: ProductConfig, dep: str, token: str | None = None) -> str:
        query = {
            "list-type": "2",
            "prefix": f"{BUCKET_PREFIX}/{self._subdir(cfg)}/{self._prefix(cfg)}_{dep}_",
            "max-keys": "1000",
        }
        if token:
            query["continuation-token"] = token
        return f"{self.base_url}/?{urlencode(query, safe='/')}"

    def archive_files(self, cfg: ProductConfig, dep: str) -> list[ArchiveFile]:
        """Every published file for one département (cached per instance)."""
        cache_key = f"{self._subdir(cfg)}/{self._prefix(cfg)}/{dep}"
        if cache_key in self._listing_cache:
            return self._listing_cache[cache_key]
        out: list[ArchiveFile] = []
        token: str | None = None
        try:
            while True:
                files, token = parse_listing(self._get_text(self.listing_url(cfg, dep, token)))
                out.extend(f for f in files if f.dep == dep)
                if token is None:
                    break
        except Exception as exc:  # noqa: BLE001 - one missing département is survivable
            logger.warning("bucket listing failed for département %s: %s", dep, exc)
            out = []
        self._listing_cache[cache_key] = out
        return out

    # ----------------------------------------------------------------- stations

    def stations(self, cfg: ProductConfig, *, allow_download: bool = False) -> pl.DataFrame:
        """In-domain stations, derived from the observation files themselves.

        There is no station catalogue to fetch: every row carries its own
        coordinates, so the station list is a by-product of reading the data.
        Two free sources, in order:

        1. What `parse` has already seen this run. This is the normal path —
           `pack` parses everything and *then* asks for metadata — and costs
           nothing, which is why `pack` never needs the third option.
        2. Failing that, the metadata columns of whatever raw files are already
           on disk under `raw_root`.

        `allow_download=True` adds a third: a cold start that pulls the newest
        period file per département for its first five columns. That is tens of
        requests and a few hundred megabytes, so it is opt-in rather than a
        silent fallback — an accidental `stations()` must not start a download.

        Clipping is the grid's job: a station is kept iff
        `grids.get(cfg.grid_id).contains(lon, lat)`. With `grid_id: null`
        nothing is clipped, which is a deliberate escape hatch for a
        France-wide run rather than a default.
        """
        import polars as pl

        rows = list(self._registry.values())
        if not rows:
            rows = self._stations_from_disk(cfg)
        if not rows and allow_download:
            rows = self._stations_from_remote(cfg)
        if not rows:
            # The one failure mode worth shouting about: a points store with
            # observations and no coordinates looks fine and silently drops
            # every station from anything that needs a position.
            logger.warning(
                "meteofrance: no station metadata — nothing parsed this run and no raw files "
                "under raw_root=%s. Parse first, pass raw_root, or call "
                "stations(cfg, allow_download=True).",
                self.raw_root,
            )

        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        kept = [
            r for r in rows if grid is None or grid.contains(r["lon"], r["lat"])
        ]
        schema = {
            "station_id": pl.String,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "height_m": pl.Float64,
            "name": pl.String,
            "dep": pl.String,
        }
        frame = pl.DataFrame(kept, schema=schema)
        return frame.unique(subset=["station_id"], keep="last").sort("station_id")

    def _stations_from_disk(self, cfg: ProductConfig) -> list[dict]:
        if self.raw_root is None or not self.raw_root.exists():
            return []
        paths = sorted(self.raw_root.rglob(f"{self._prefix(cfg)}_*.csv.gz"))
        if not paths:
            return []
        logger.info("meteofrance: deriving stations from %d raw file(s) on disk", len(paths))
        out: dict[str, dict] = {}
        for path in paths:
            for row in _read_station_metadata(path):
                out[row["station_id"]] = row
        return list(out.values())

    def _stations_from_remote(self, cfg: ProductConfig) -> list[dict]:
        """Cold start: one download per département, for five columns."""
        from meteopack.core.http import FetchError, fetch_file

        deps = self._departements(cfg)
        logger.warning(
            "meteofrance: no parsed data and no local archive, so station coordinates have to "
            "be read out of the files — downloading the newest period file for %d "
            "département(s). Run `pack` (or set raw_root) to avoid this.",
            len(deps),
        )
        rl = self.rate_limit
        out: dict[str, dict] = {}
        with tempfile.TemporaryDirectory(prefix="meteopack-mf-") as scratch:
            root = Path(self.raw_root) if self.raw_root else Path(scratch)
            for dep in deps:
                files = [f for f in self.archive_files(cfg, dep) if f.volatile]
                if not files:
                    files = sorted(self.archive_files(cfg, dep), key=lambda f: f.years[0])[-1:]
                for archive in files:
                    dest = root / archive.dep / archive.filename
                    try:
                        fetch_file(
                            self.file_url(cfg, archive.filename),
                            dest,
                            unit_key=unit_key(cfg, archive.stem),
                            session=self._requests_session(),
                            limiter=self._limiter,
                            retries=rl.max_retries,
                            backoff=rl.backoff_seconds,
                            timeout=self.timeout,
                        )
                    except FetchError as exc:
                        logger.warning("meteofrance: %s unavailable: %s", archive.filename, exc)
                        continue
                    for row in _read_station_metadata(dest):
                        out[row["station_id"]] = row
        return list(out.values())

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every per-département file whose period overlaps `window`.

        No archive is downloaded; only the bucket listing is read, one request
        per département.
        """
        start, end = self._window_bounds(window)
        # Validate the window against the product cadence in one place.
        Timeline(start, end, cfg.cadence)

        deps = self._departements(cfg)
        only = cfg.extra.get("period_tokens")
        units: list[Unit] = []
        seen: set[str] = set()
        for dep in deps:
            for archive in self.archive_files(cfg, dep):
                if not archive.covers(start, end):
                    continue
                if only and archive.token not in set(only):
                    continue
                key = unit_key(cfg, archive.stem)
                if key in seen:
                    continue
                seen.add(key)
                first, _ = archive.years
                units.append(
                    Unit(
                        key=key,
                        url=self.file_url(cfg, archive.filename),
                        valid_time=datetime(first, 1, 1, tzinfo=UTC) if first else None,
                        meta={
                            "dep": archive.dep,
                            "token": archive.token,
                            "filename": archive.filename,
                            "size": str(archive.size),
                            "last_modified": archive.last_modified,
                            # `latest-*` is rewritten in place under this exact
                            # name. Without this the ledger marks it done on the
                            # first download and the live tail silently freezes.
                            "volatile": archive.volatile,
                        },
                    )
                )
        logger.info("meteofrance: %d units across %d département(s)", len(units), len(deps))
        return units

    @staticmethod
    def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
        def as_utc(value: datetime) -> datetime:
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

        return as_utc(window.start), as_utc(window.end)

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Download one département CSV. None if upstream has no such file.

        A département with no file for a period 404s, which is routine and a
        skip rather than a failure. Everything else (retries, `.part` + atomic
        rename, checksum, rate limiting) belongs to `meteopack.core.http`; the
        import is local so tests can monkeypatch it.
        """
        from meteopack.core.http import NotFoundError, fetch_file

        meta = unit.meta or {}
        dest = Path(raw_root) / meta.get("dep", "unknown") / meta.get(
            "filename", f"{unit.key.rsplit('/', 1)[-1]}.csv.gz"
        )
        # `fetch_file` is a no-op when `dest` exists, which is right for a
        # closed period and wrong for `latest-*`: that file is rewritten under
        # the same name, so skipping it pins the store to the first download.
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
            logger.info("meteofrance: no upstream file for %s (404)", unit.key)
            return None

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """One département CSV -> tidy frame of `time`, `station_id`, values.

        Only the columns `cfg.variables` asks for are read. An empty field is
        the file's way of saying missing, and becomes null *before* scale and
        offset, so a blank can never be scaled into a plausible-looking number;
        the valid range is applied afterwards, so it is expressed in canonical
        units.

        Station coordinates are harvested on the way past and kept for
        `stations()` — they are in every row, so reading them costs nothing
        here and saves a download later.
        """
        import polars as pl

        path = Path(raw.path)
        wanted = variables_by_column(cfg)
        date_col = self._date_column(cfg)
        header = _header_columns(path)
        self._check_fingerprint(header, cfg, path.name)

        by_stripped = {c.strip(): c for c in header}
        if date_col not in by_stripped or ID_COL not in by_stripped:
            raise SchemaDriftError(
                f"{path.name}: expected columns {date_col!r} and {ID_COL!r}; "
                f"present: {sorted(by_stripped)[:12]}..."
            )
        # Columns genuinely come and go between periods (a 1950s file has no
        # 3-second gust), so an absent one is a null column, not a failure.
        present = [c for c in wanted if c in by_stripped]
        missing = [c for c in wanted if c not in by_stripped]
        if missing:
            logger.info("%s: %d configured column(s) absent: %s", path.name, len(missing), missing)

        select = [by_stripped[c] for c in (*META_COLS, date_col) if c in by_stripped]
        select += [by_stripped[c] for c in present]
        frame = pl.read_csv(
            path,
            separator=";",
            infer_schema=False,
            columns=select,
            encoding="utf8-lossy",
        )
        frame = frame.rename({v: k for k, v in by_stripped.items() if v in select and v != k})

        self._remember_stations(frame, path)

        out = frame.select(
            _time_expr(date_col, self._date_format(cfg)).alias("time"),
            pl.col(ID_COL).str.strip_chars().alias("station_id"),
            pl.col(LON_COL).str.strip_chars().cast(pl.Float64, strict=False).alias("__lon"),
            pl.col(LAT_COL).str.strip_chars().cast(pl.Float64, strict=False).alias("__lat"),
            *[
                pl.col(c).str.strip_chars().cast(pl.Float64, strict=False).alias(wanted[c].name)
                for c in present
            ],
        ).drop_nulls("time")

        out = self._clip(out, cfg)
        out = _apply_variables(out, [wanted[c] for c in present], cfg)
        out = out.unique(subset=["time", "station_id"], keep="last").sort(
            ["time", "station_id"]
        )
        return self._drop_off_cadence(out, cfg)

    def _clip(self, out: pl.DataFrame, cfg: ProductConfig) -> pl.DataFrame:
        """Drop rows for stations outside the grid footprint.

        A département is a listing filter, not a domain: `59` is mostly inside
        the radar footprint and `21` mostly outside, and neither is a reason to
        keep or drop a particular station.
        """
        import polars as pl

        if not cfg.grid_id:
            return out.drop("__lon", "__lat")
        grid = grids.get(cfg.grid_id)
        # One `contains` per *station*, not per row: the test densifies a
        # projected ring and a million-row file has a few hundred stations.
        coords = out.select("station_id", "__lon", "__lat").unique(subset=["station_id"])
        inside = {
            row["station_id"]
            for row in coords.iter_rows(named=True)
            if row["__lon"] is not None
            and row["__lat"] is not None
            and grid.contains(row["__lon"], row["__lat"])
        }
        return out.filter(pl.col("station_id").is_in(inside)).drop("__lon", "__lat")

    def _remember_stations(self, frame: pl.DataFrame, path: Path) -> None:
        """Record every station this file describes, for `stations()`."""
        for row in _station_rows(frame, path):
            self._registry[row["station_id"]] = row

    def _check_fingerprint(
        self, header: Sequence[str], cfg: ProductConfig, filename: str
    ) -> None:
        """Refuse to parse a column set the config was not pinned against.

        Pinned per period token rather than per product: a 1950s file and the
        current one genuinely have different columns, and treating that as
        drift would make every backfill fail. A token with no pin is not
        checked.
        """
        pinned = (cfg.extra.get("schema_fingerprints") or {}).get(_token_of(filename))
        if not pinned:
            return
        seen = schema_fingerprint(header)
        if seen != pinned:
            raise SchemaDriftError(
                f"{filename}: Météo-France column set changed: pinned {pinned}, got {seen} "
                f"({len(header)} columns). Re-run `scan` and re-pin "
                f"extra.schema_fingerprints after checking what moved."
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
        """Report every distinct CSV column set across `raw_files`.

        Upstream appends columns as new sensors appear, so the pipeline pins
        the fingerprints this returns and `parse` refuses to run against
        anything else. The report is deliberately a plain dict: it is meant to
        be written to JSON next to the store and diffed between runs.
        """
        variants: dict[tuple[str, ...], dict] = {}
        errors: list[dict[str, str]] = []
        for raw in raw_files:
            path = Path(raw.path)
            try:
                columns = tuple(c.strip() for c in _header_columns(path))
            except Exception as exc:  # noqa: BLE001 - a bad file is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
                continue
            entry = variants.setdefault(
                columns,
                {
                    "token": _token_of(path.name),
                    "n_columns": len(columns),
                    "columns": list(columns),
                    "fingerprint": schema_fingerprint(columns),
                    "count": 0,
                    "examples": [],
                },
            )
            entry["count"] += 1
            if len(entry["examples"]) < 3:
                entry["examples"].append(path.name)

        ordered = sorted(variants.values(), key=lambda e: (e["token"], e["fingerprint"]))
        by_token: dict[str, list[str]] = {}
        for entry in ordered:
            by_token.setdefault(entry["token"], []).append(entry["fingerprint"])
        single = {t: fps[0] for t, fps in by_token.items() if len(fps) == 1}
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": {t: sorted(fps) for t, fps in by_token.items()},
            "schema_fingerprints": single,
            "fingerprint": product_fingerprint(single) if len(single) == len(by_token) else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ----------------------------------------------------------------- file helpers


def _time_expr(column: str, fmt: str):
    """Parse the product's date column into a UTC `time` column.

    The hourly product stamps rows `AAAAMMJJHH` — no minutes — and chrono, the
    parser behind polars' `strptime`, refuses a format that names an hour
    without a minute. Supplying the implied `00` here keeps the two products'
    date formats honest in config (`%Y%m%d%H` and `%Y%m%d%H%M`) instead of
    making the hourly one carry a lie.
    """
    import polars as pl

    expr = pl.col(column).str.strip_chars()
    if "%H" in fmt and "%M" not in fmt:
        expr, fmt = expr + pl.lit("00"), fmt + "%M"
    return (
        expr.str.strptime(pl.Datetime("us"), fmt, strict=False).dt.replace_time_zone("UTC")
    )


def _token_of(filename: str) -> str:
    """`H_59_latest-2025-2026.csv.gz` -> `latest-2025-2026`."""
    m = _FILE_RE.match(filename)
    return m.group("token") if m else ""


def _header_columns(path: Path) -> list[str]:
    """The CSV header, read without decompressing the rest of the file."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as fh:  # type: ignore[operator]
        line = fh.readline()
    return line.decode("utf-8", errors="replace").rstrip("\r\n").split(";")


def _read_station_metadata(path: Path) -> list[dict]:
    """`station_id, lat, lon, height_m, name, dep` from one file's five id columns."""
    import polars as pl

    try:
        present = {c.strip() for c in _header_columns(path)}
        if not set(META_COLS) <= present:
            return []
        frame = pl.read_csv(
            path,
            separator=";",
            infer_schema=False,
            columns=list(META_COLS),
            encoding="utf8-lossy",
        )
    except Exception as exc:  # noqa: BLE001 - one unreadable file is survivable
        logger.warning("meteofrance: cannot read station metadata from %s: %s", path, exc)
        return []
    return _station_rows(frame, path)


def _station_rows(frame: pl.DataFrame, path: Path) -> list[dict]:
    """Deduplicated station records from a frame that carries the id columns."""
    if not set(META_COLS) <= set(frame.columns):
        return []
    dep = _FILE_RE.match(path.name).group("dep") if _FILE_RE.match(path.name) else ""
    unique = frame.select(META_COLS).unique(subset=[ID_COL], keep="last")
    out: list[dict] = []
    for row in unique.iter_rows(named=True):
        lon, lat = _number(row[LON_COL]), _number(row[LAT_COL])
        if lon is None or lat is None:
            continue
        try:
            sid = station_id(row[ID_COL])
        except ValueError:
            continue
        out.append(
            {
                "station_id": sid,
                "lat": lat,
                "lon": lon,
                "height_m": _number(row[ALT_COL]),
                "name": str(row[NAME_COL] or "").strip(),
                "dep": dep,
            }
        )
    return out


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _apply_variables(
    frame: pl.DataFrame, variables: Sequence[Variable], cfg: ProductConfig
) -> pl.DataFrame:
    """Sentinel -> null, then scale/offset, then valid-range mask, then dtype.

    Météo-France writes missing as an empty field, which the cast to Float64
    has already turned into a null. `extra.missing_sentinels` is for the day a
    numeric sentinel comes through instead; a -999 that gets scaled first is
    indistinguishable from data.
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
        raise ValueError(f"unsupported dtype {name!r} for a Météo-France variable") from None
