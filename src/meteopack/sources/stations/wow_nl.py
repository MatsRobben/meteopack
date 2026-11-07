"""WOW-NL citizen weather reports, from the local archive of yearly zips.

The Met Office WOW platform (KNMI's Dutch instance) collects reports from
private weather stations and a few official ones. The archive is eleven zips,
one per year, each holding **one JSON member per UTC hour**::

    2024.zip:  2024/WOW_NL_2024061512.json  ->  {"NextHashKey": "...", "Total": 0,
                                                  "Object": [{report}, {report}, ...]}

A member holds the reports whose `ReportEndDateTime` falls in [hour, hour+1h),
newest first, as one page of the WOW API - at most 10 000 of them. From 2022 on
almost every hour hits that cap, so the *earliest* minutes of the hour are
simply not in the archive (`hours.at_page_cap` says which). Over the years 82
distinct fields appear, including imperial twins of most measurements
(`*_Fahrenheit`, `*_Inch`, `*_MilePerHour`, `*_Knot`, `*_InchOfMercury`).

This module packs that archive **without making a single analysis choice**:

* every field of every report is kept, with its value exactly as the JSON had
  it - numbers as float64 (or int64 for identifiers and codes), never
  quantised; the imperial twins and the 3-occurrence fields included;
* `ReportEndDateTime` is parsed to UTC microseconds as `time`, and the original
  string is kept beside it (some carry 7 fractional digits, one more than a
  microsecond holds);
* the measurement fields get the CF-style names the other stores use, through
  `variables` in the product config; everything else keeps its WOW name;
* the only rows not stored are *exact* duplicates - identical in every field.
  These come from the 2019 hours that exist in both 2019.zip and 2020.zip.
  The copy with more records is kept (ties: the later zip), every row of the
  other copy that is not an exact duplicate is kept too, and the hours table
  counts what was dropped and why;
* a value that does not fit its pinned type, an explicit JSON `null`, or a
  field not in the pinned catalogue is not coerced and not dropped: it is
  written, as JSON, into the row's `unpinned_fields` column, and `pack` names
  it and fails, so the catalogue is re-pinned deliberately.

No gap filling, no QC, no rain rule: those are choices, and they live in the
parameterised derive step (`meteopack.derive.regularise`).

Two decoding paths, because speed is a goal and exactness is the constraint:

* **typed** - msgspec decodes the member straight into structs whose fields are
  typed from `extra.fields`, with unknown fields forbidden. That is both the
  fastest decoder measured (135 MiB/s of JSON into a typed frame, against 101
  for msgspec + polars inference and 76 for the stdlib; 20 real 2024 members,
  2026-11-06) and an exact type check: msgspec does not turn "1.5" into 1.5 or
  `true` into 1 the way polars' constructor does (measured: it does both,
  silently). It cannot tell an explicit `null` from an absent field, so a
  member containing the bytes `null` anywhere takes the exact path instead.
* **exact** - the generic decoder and a per-key check in Python. Slower, but
  used only for the members the typed path refuses.

A member that is not valid JSON at all (2019.zip has seven, all truncated
uploads) is *salvaged*: every complete report before the damage is recovered
and stored, and the hours table says so. Their hours also exist, intact, in
2020.zip, and the exact-duplicate rule then keeps one copy.
"""
from __future__ import annotations

import json
import logging
import operator
import os
import re
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

from meteopack.core.config import ProductConfig, SourceConfig
from meteopack.sources.base import RawFile, TimeWindow, Unit
from meteopack.sources.common import SchemaDriftError, fingerprint

if TYPE_CHECKING:
    import polars as pl

logger = logging.getLogger(__name__)

#: Where the yearly zips live: $WOW_NL_ARCHIVE (the source config sets
#: `base_url` from it). Read-only: nothing here ever writes into it.
ARCHIVE_DIR = os.environ.get("WOW_NL_ARCHIVE", "")

#: `2019/WOW_NL_2019053023.json`. The folder is matched loosely because
#: 2020.zip keeps its 2019 hours under `2019/`, not `2020/`.
_MEMBER_RE = re.compile(r"^(?:[^/]*/)*WOW_NL_(?P<stamp>\d{10})\.json$")
_ZIP_RE = re.compile(r"^(?P<year>\d{4})\.zip$")

#: Per-row provenance columns, and the column that keeps whatever did not fit.
SOURCE_ZIP = "source_zip"
SOURCE_MEMBER = "source_member"
UNPINNED = "unpinned_fields"

#: The pinned type names `extra.fields` may use.
_KINDS = ("float64", "int64", "bool", "string")

#: Two to the 53rd: the largest integer magnitude float64 holds exactly. An
#: integer beyond it in a float field would be rounded, so it is not "a fit".
_EXACT_INT = 2**53

#: Columns that describe where a row came from rather than what was observed;
#: two rows equal in everything but these are exact duplicates of one report.
PROVENANCE = (SOURCE_ZIP, SOURCE_MEMBER)

#: `%.f` takes any number of fractional digits; polars truncates (never
#: rounds) beyond the requested microsecond unit, verified on
#: "11:59:59.9999999" -> 11:59:59.999999.
TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%.f%:z"


def member_hour(name: str) -> datetime:
    """`2024/WOW_NL_2024061512.json` -> 2024-06-15 12:00 UTC."""
    m = _MEMBER_RE.match(name)
    if not m:
        raise ValueError(f"{name!r} is not a WOW-NL hour member (…/WOW_NL_YYYYMMDDHH.json)")
    s = m.group("stamp")
    return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]), tzinfo=UTC)


def local_path(url: str) -> Path:
    parsed = urlparse(str(url))
    if parsed.scheme in ("", "file"):
        return Path(unquote(parsed.path or str(url)))
    raise ValueError(f"{url!r} is not a local path; wow_nl reads a local archive")


# ------------------------------------------------------------------ catalogue


class Catalogue:
    """The pinned field catalogue of one product config, compiled once.

    `extra.fields` maps every WOW field to its type. Canonical names come from
    `variables` (`source_key` -> `name`); the station id and the report time
    are named by `extra.station_key` / `extra.time_key`. Everything else keeps
    the name WOW gave it, so the store has one column per raw field.
    """

    def __init__(self, cfg: ProductConfig) -> None:
        fields = dict(cfg.extra.get("fields") or {})
        if not fields:
            raise ValueError(f"{cfg.store_id}: extra.fields (the pinned field catalogue) is empty")
        bad = {k: v for k, v in fields.items() if v not in _KINDS}
        if bad:
            raise ValueError(f"{cfg.store_id}: unknown field types {bad}; use one of {_KINDS}")
        if cfg.schema_fingerprint:
            seen = catalogue_fingerprint(fields)
            if seen != cfg.schema_fingerprint:
                raise SchemaDriftError(
                    f"config is inconsistent: extra.fields hashes to {seen} but "
                    f"schema_fingerprint pins {cfg.schema_fingerprint}. Re-pin both together.")
        self.kinds: dict[str, str] = fields
        self.station_key = str(cfg.extra.get("station_key", "SiteId"))
        self.time_key = str(cfg.extra.get("time_key", "ReportEndDateTime"))
        for key in (self.station_key, self.time_key):
            if fields.get(key) != "string":
                raise ValueError(f"{cfg.store_id}: {key} must be pinned as a string field")
        self.rename: dict[str, str] = {}
        for var in cfg.variables:
            if var.source_key not in fields:
                raise ValueError(f"variable {var.name}: source_key {var.source_key!r} is not "
                                 f"in extra.fields")
            if fields[var.source_key] != "float64":
                raise ValueError(f"variable {var.name}: {var.source_key} is not a float64 field")
            self.rename[var.source_key] = var.name
        self.rename[self.station_key] = "station_id"
        self.variables = [v.name for v in cfg.variables]

    @cached_property
    def columns(self) -> dict[str, pl.DataType]:
        """The output schema, in a fixed order, identical for every frame.

        Fixed so frames from different hours concatenate without negotiation
        and every partition file has every column, typed the same, even where
        a rare field never occurs in that month.
        """
        import polars as pl

        dtype = {"float64": pl.Float64, "int64": pl.Int64, "bool": pl.Boolean,
                 "string": pl.String}
        out: dict[str, pl.DataType] = {"station_id": pl.String, "time": pl.Datetime("us")}
        by_name = {self.rename.get(k, k): k for k in self.kinds}
        for name in self.variables:
            out[name] = pl.Float64
        for raw in sorted(self.kinds):
            name = self.rename.get(raw, raw)
            if name not in out:
                out[name] = dtype[self.kinds[raw]]
        out[UNPINNED] = pl.String
        out[SOURCE_ZIP] = pl.String
        out[SOURCE_MEMBER] = pl.String
        assert len(by_name) == len(self.kinds)
        return out

    @cached_property
    def data_columns(self) -> list[str]:
        """Every column except provenance: what "identical in every field" compares."""
        return [c for c in self.columns if c not in PROVENANCE]

    @cached_property
    def decoder(self):
        """A msgspec decoder for a whole member, typed from the catalogue."""
        from typing import Annotated

        import msgspec

        # Bounds route the rare out-of-range number to the exact path rather
        # than letting msgspec round an integer literal beyond 2**53 into a
        # float field, or hand polars an integer it cannot hold in int64.
        py = {"float64": Annotated[float, msgspec.Meta(ge=-_EXACT_INT, le=_EXACT_INT)],
              "int64": Annotated[int, msgspec.Meta(ge=-(2**63), le=2**63 - 1)],
              "bool": bool, "string": str}
        rec = msgspec.defstruct(
            "WowReport", [(k, py[t] | None, None) for k, t in self.kinds.items()],
            forbid_unknown_fields=True, gc=False)
        doc = msgspec.defstruct(
            "WowPage", [("Object", list[rec]), ("NextHashKey", str | None, None),
                        ("Total", int | None, None)],
            forbid_unknown_fields=True)
        return msgspec.json.Decoder(doc)

    @cached_property
    def getters(self) -> dict[str, Any]:
        return {k: operator.attrgetter(k) for k in self.kinds}


def catalogue_fingerprint(fields: dict[str, str]) -> str:
    """What `schema_fingerprint` pins: the field names *and* their types."""
    return fingerprint(f"{k}:{v}" for k, v in fields.items())


# --------------------------------------------------------------- one member


@dataclass(slots=True)
class Member:
    """One hour member, decoded. `frame` is None when nothing was readable."""

    zip: str
    name: str
    hour: datetime
    bytes: int
    compressed_bytes: int
    frame: Any = None
    n_records: int = 0
    status: str = "ok"           # ok | empty_file | salvaged | error
    path: str = "typed"           # typed | exact | salvage | -
    next_hash_key: bool | None = None
    total: int | None = None
    error: str | None = None
    unpinned_top_level: str | None = None
    #: "field|reason" -> how many values of this member went to unpinned_fields.
    unpinned_counts: dict | None = None
    #: Seconds spent reading (and inflating) the member, and decoding it - the
    #: split that says whether a pack is waiting on the filesystem or the CPU.
    read_s: float = 0.0
    decode_s: float = 0.0


def _fits(kind: str, value: Any) -> bool:
    t = type(value)
    if kind == "float64":
        return t is float or (t is int and -_EXACT_INT <= value <= _EXACT_INT)
    if kind == "int64":
        return t is int and -(2**63) <= value < 2**63
    if kind == "bool":
        return t is bool
    return t is str


def _typed_columns(recs: list, cat: Catalogue) -> dict[str, list]:
    """Struct attributes -> one Python list per field (C-level map per field)."""
    return {k: list(map(cat.getters[k], recs)) for k in cat.kinds}


_JSON_TYPE = {type(None): "null", bool: "bool", int: "int", float: "float", str: "string",
              list: "array", dict: "object"}


def unpinned_reason(kind: str | None, value: Any) -> str:
    """Why a value could not go into its column, as a code you can count.

    `unknown_field` for a field the catalogue does not pin (schema drift);
    otherwise `<json type>_for_<pinned type>`: `string_for_float64` is a
    number sent as text (the A1.05 report found wind speeds like "36,612.3"),
    `null_for_float64` an explicit JSON null, `int_inexact_for_float64` an
    integer too large for float64 to hold exactly.
    """
    if kind is None:
        return "unknown_field"
    jt = _JSON_TYPE.get(type(value), type(value).__name__)
    if jt == "int" and kind in ("float64", "int64"):
        jt = "int_inexact" if kind == "float64" else "int_out_of_range"
    return f"{jt}_for_{kind}"


def _exact_columns(objs: list, cat: Catalogue) -> tuple[dict[str, list], list, dict]:
    """The slow, exact path: every key of every record checked in Python.

    A value that fits its pinned type goes to its column. Anything else - an
    unknown field, an explicit null, text where a number was pinned, a float
    where an integer was - goes, unchanged, into that row's `unpinned_fields`
    as `{field: {"raw": <the JSON value>, "reason": <code>}}`, and is counted
    per (field, reason). Nothing is coerced, nulled or dropped: a malformed
    number keeps its original text, and judging it is a later stage's job.
    """
    import msgspec

    n = len(objs)
    cols: dict[str, list] = {k: [None] * n for k in cat.kinds}
    unpinned: list = [None] * n
    counts: dict[str, int] = defaultdict(int)
    kinds = cat.kinds
    for i, obj in enumerate(objs):
        if type(obj) is not dict:
            reason = f"{_JSON_TYPE.get(type(obj), 'other')}_for_record"
            unpinned[i] = msgspec.json.encode(
                {"__record__": {"raw": obj, "reason": reason}}).decode()
            counts[f"__record__|{reason}"] += 1
            continue
        extra = None
        for k, v in obj.items():
            kind = kinds.get(k)
            if kind is not None and v is not None and _fits(kind, v):
                cols[k][i] = v
            else:
                if extra is None:
                    extra = {}
                reason = unpinned_reason(kind, v)
                extra[k] = {"raw": v, "reason": reason}
                counts[f"{k}|{reason}"] += 1
        if extra:
            unpinned[i] = msgspec.json.encode(dict(sorted(extra.items()))).decode()
    return cols, unpinned, dict(counts)


def _salvage(raw: bytes) -> tuple[list, str]:
    """Every complete report before the point where a damaged member breaks.

    Python's decoder reads one value at a time from a position, so the reports
    are taken one by one out of the `Object` array until one does not parse.
    The partial report at the break is not stored - it is not a report.
    """
    text = raw.decode("utf-8", errors="replace")
    m = re.search(r'"Object"\s*:\s*\[', text)
    if not m:
        return [], "no Object array found"
    dec = json.JSONDecoder()
    pos, out = m.end(), []
    while True:
        while pos < len(text) and text[pos] in " \t\r\n,":
            pos += 1
        if pos >= len(text) or text[pos] == "]":
            return out, "array closed"
        try:
            obj, pos = dec.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            return out, f"stopped at char {pos}: {exc.msg}"
        out.append(obj)


def decode_member(raw: bytes, cat: Catalogue, member: Member) -> Member:
    """Bytes of one member -> typed frame in the catalogue's schema."""
    import msgspec
    import polars as pl

    if not raw:
        member.status, member.path = "empty_file", "-"
        return member
    cols = unpinned = None
    if b"null" not in raw:
        try:
            page = cat.decoder.decode(raw)
            recs = page.Object
            cols = _typed_columns(recs, cat)
            member.n_records = len(recs)
            member.next_hash_key = page.NextHashKey is not None
            member.total = page.Total
        except (msgspec.ValidationError, msgspec.DecodeError):
            cols = None
    if cols is None:
        try:
            doc = msgspec.json.decode(raw)
            member.path = "exact"
        except msgspec.DecodeError as exc:
            objs, how = _salvage(raw)
            member.status, member.path = "salvaged", "salvage"
            member.error = f"invalid JSON ({exc}); {len(objs)} complete report(s) recovered, {how}"
            doc = {"Object": objs}
        if not isinstance(doc, dict):
            member.status, member.error = "error", f"top level is {type(doc).__name__}, not an object"
            return member
        objs = doc.get("Object")
        top = {k: v for k, v in doc.items() if k not in ("Object", "NextHashKey", "Total")}
        if not isinstance(objs, list):
            top["Object"] = objs
            objs = []
        if "NextHashKey" in doc:
            member.next_hash_key = doc["NextHashKey"] is not None
        total = doc.get("Total")
        if total is not None and not _fits("int64", total):
            top["Total"] = total
            total = None
        member.total = total
        if top:
            member.unpinned_top_level = msgspec.json.encode(top).decode()
        member.n_records = len(objs)
        cols, unpinned, member.unpinned_counts = _exact_columns(objs, cat)

    n = member.n_records
    if not n:
        return member
    schema = cat.columns
    series = []
    for raw_name, values in cols.items():
        name = cat.rename.get(raw_name, raw_name)
        series.append(pl.Series(name, values, dtype=schema[name], strict=True))
    frame = pl.DataFrame(series)
    when = frame[cat.time_key].str.to_datetime(
        TIME_FORMAT, time_unit="us", time_zone="UTC", strict=False)
    bad = int(when.null_count())
    if bad:
        examples = frame.filter(when.is_null())[cat.time_key].head(3).to_list()
        member.status = "error"
        member.error = (f"{bad} report(s) without a parseable {cat.time_key} "
                        f"(e.g. {examples}); the member is not stored")
        return member
    frame = frame.with_columns(
        when.dt.replace_time_zone(None).alias("time"),
        pl.Series(UNPINNED, unpinned, dtype=pl.String) if unpinned is not None
        else pl.lit(None, pl.String).alias(UNPINNED),
        pl.lit(member.zip).alias(SOURCE_ZIP),
        pl.lit(member.name).alias(SOURCE_MEMBER),
    )
    member.frame = frame.select(list(schema))
    return member


# ------------------------------------------------------------------ one day


#: A zip's index, read once per process: hour -> ZipInfo.
_ZIP_INDEX: dict[str, tuple[zipfile.ZipFile, dict[datetime, zipfile.ZipInfo]]] = {}


def zip_index(path: str | Path) -> tuple[zipfile.ZipFile, dict[datetime, zipfile.ZipInfo]]:
    """Open a zip once per process and index its members by hour.

    Reading the central directory of a 10 GB zip is a few ms, but a worker
    handles hundreds of days and must not repeat it - or reopen the file - for
    each of them.
    """
    key = str(path)
    hit = _ZIP_INDEX.get(key)
    if hit is None:
        zf = zipfile.ZipFile(key)
        index: dict[datetime, zipfile.ZipInfo] = {}
        for info in zf.infolist():
            if info.is_dir():
                continue
            try:
                hour = member_hour(info.filename)
            except ValueError:
                continue
            index[hour] = info
        hit = _ZIP_INDEX[key] = (zf, index)
    return hit


def _rank(member: Member) -> tuple:
    """Which copy of an hour is preferred: intact, then more records, then later zip."""
    return (member.status == "ok", member.n_records, member.zip)


def parse_day(paths: list[str], day: date, cat: Catalogue, *, page_cap: int = 9_900
              ) -> tuple[pl.DataFrame | None, list[dict]]:
    """Every hour of one UTC day, from every zip that has it.

    Returns the day's rows - exact duplicates removed - and one record per
    member for the hours table. A day is the unit of work because it is small
    (at most ~240 000 reports, ~340 MB of JSON) and because both copies of a
    duplicated 2019 hour fall in the same day, so choosing between them never
    needs a second pass.
    """
    import polars as pl

    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    by_hour: dict[datetime, list[Member]] = defaultdict(list)
    for path in paths:
        zf, index = zip_index(path)
        for h in range(24):
            hour = start + timedelta(hours=h)
            info = index.get(hour)
            if info is None:
                continue
            member = Member(zip=Path(path).name, name=info.filename, hour=hour,
                            bytes=info.file_size, compressed_bytes=info.compress_size)
            t0 = time.perf_counter()
            try:
                raw = zf.read(info)
            except (zipfile.BadZipFile, OSError, EOFError) as exc:
                member.status, member.path = "error", "-"
                member.error = f"cannot read from zip: {type(exc).__name__}: {exc}"
            else:
                t1 = time.perf_counter()
                member.read_s = t1 - t0
                decode_member(raw, cat, member)
                member.decode_s = time.perf_counter() - t1
            by_hour[hour].append(member)

    frames: list[pl.DataFrame] = []
    records: list[dict] = []
    data_cols = cat.data_columns
    for hour in sorted(by_hour):
        copies = sorted(by_hour[hour], key=_rank, reverse=True)
        kept = [m for m in copies if m.frame is not None]
        dropped: dict[str, int] = {}
        if len(kept) > 1:
            # Chosen copy first, so `keep="first"` keeps its provenance.
            both = pl.concat([m.frame for m in kept], how="vertical")
            hour_frame = both.unique(subset=data_cols, keep="first", maintain_order=True)
            survived = hour_frame[SOURCE_MEMBER].len()
            counts = dict(hour_frame.group_by(SOURCE_ZIP).len().iter_rows())
            for m in kept:
                dropped[m.zip] = m.frame.height - counts.get(m.zip, 0)
            assert survived == sum(counts.values())
        elif kept:
            hour_frame = kept[0].frame
            # Within one member: duplicates are rare enough that the cheap key
            # check gates the full-row comparison.
            if hour_frame.select("station_id", "time", "Id").is_duplicated().any():
                deduped = hour_frame.unique(subset=data_cols, keep="first", maintain_order=True)
                dropped[kept[0].zip] = hour_frame.height - deduped.height
                hour_frame = deduped
        else:
            hour_frame = None
        if hour_frame is not None:
            frames.append(hour_frame)
        for rank, m in enumerate(copies):
            stored = (m.frame.height - dropped.get(m.zip, 0)) if m.frame is not None else 0
            times = m.frame["time"] if m.frame is not None else None
            outside = 0 if times is None else int(
                ((times < hour.replace(tzinfo=None))
                 | (times >= (hour + timedelta(hours=1)).replace(tzinfo=None))).sum())
            records.append({
                "hour": hour.replace(tzinfo=None), "zip": m.zip, "member": m.name,
                "status": m.status, "decode_path": m.path,
                "bytes": m.bytes, "compressed_bytes": m.compressed_bytes,
                "n_records": m.n_records, "n_stored": stored,
                "n_duplicate_dropped": dropped.get(m.zip, 0),
                "n_unpinned": 0 if m.frame is None else int(m.frame[UNPINNED].is_not_null().sum()),
                "n_outside_hour": outside,
                "at_page_cap": m.n_records >= page_cap,
                "next_hash_key": m.next_hash_key, "total": m.total,
                "first_time": None if times is None else times.min(),
                "last_time": None if times is None else times.max(),
                "n_copies": len(copies), "chosen": rank == 0,
                "error": m.error, "unpinned_top_level": m.unpinned_top_level,
                "unpinned_counts": json.dumps(m.unpinned_counts, sort_keys=True)
                if m.unpinned_counts else None,
                "read_s": round(m.read_s, 4), "decode_s": round(m.decode_s, 4),
            })
    frame = pl.concat(frames, how="vertical") if frames else None
    return frame, records


# -------------------------------------------------------------------- tables

#: The hours table: one row per member of every zip, plus one per hour that no
#: zip has (status "missing"), so a gap in the archive is a row, not an absence.
HOURS_SCHEMA = {
    "hour": "datetime", "zip": "string", "member": "string", "status": "string",
    "decode_path": "string", "bytes": "int64", "compressed_bytes": "int64",
    "n_records": "int64", "n_stored": "int64", "n_duplicate_dropped": "int64",
    "n_unpinned": "int64", "n_outside_hour": "int64", "at_page_cap": "bool",
    "next_hash_key": "bool", "total": "int64", "first_time": "datetime",
    "last_time": "datetime", "n_copies": "int64", "chosen": "bool",
    "error": "string", "unpinned_top_level": "string", "unpinned_counts": "string",
    "read_s": "float64", "decode_s": "float64",
}


def hours_frame(records: list[dict]) -> pl.DataFrame:
    import polars as pl

    dtype = {"datetime": pl.Datetime("us"), "string": pl.String, "int64": pl.Int64,
             "bool": pl.Boolean, "float64": pl.Float64}
    schema = {k: dtype[v] for k, v in HOURS_SCHEMA.items()}
    return pl.DataFrame(records, schema=schema)


def segments(frame: pl.DataFrame) -> pl.DataFrame:
    """Runs of identical (Latitude, Longitude, Height) per station, in time order.

    The position history, exactly: a station that moves, or flips between two
    reported positions, gets a row per run. `frame` must be sorted by
    (station_id, time). Height is the raw value - 0 included; whether 0 means
    "unknown" is a choice for a derived step to make.
    """
    import polars as pl

    return (
        frame.select(*_POSITION, "time", "IsOfficial")
        .with_columns(pl.struct(_POSITION).rle_id().alias("__run"))
        .group_by("__run", maintain_order=True)
        .agg(*(pl.col(k).first() for k in _POSITION),
             pl.col("time").min().alias("first_time"),
             pl.col("time").max().alias("last_time"),
             pl.len().cast(pl.Int64).alias("n_reports"),
             pl.col("IsOfficial").sum().cast(pl.Int64).alias("n_official"))
        .drop("__run")
    )


#: What identifies one position of one station.
_POSITION = ["station_id", "Latitude", "Longitude", "Height"]


def merge_segments(monthly: pl.DataFrame) -> pl.DataFrame:
    """Join per-month runs into the store-wide position history.

    A run that continues across a month boundary is two rows in the per-month
    tables; consecutive rows of one station with the same position are one run.
    """
    import polars as pl

    if monthly.is_empty():
        return monthly
    return (
        monthly.sort("station_id", "first_time", "last_time", nulls_last=True)
        .with_columns(pl.struct(_POSITION).rle_id().alias("__run"))
        .group_by("__run", maintain_order=True)
        .agg(*(pl.col(k).first() for k in _POSITION),
             pl.col("first_time").min(), pl.col("last_time").max(),
             pl.col("n_reports").sum(), pl.col("n_official").sum())
        .drop("__run")
    )


def _weighted_median(values: list, weights: list[int]) -> float | None:
    """The median of the values each repeated `weight` times, without repeating them.

    Same definition as polars' `median`: the middle value, or the mean of the
    two middle values for an even count. Nulls are not values.
    """
    pairs = sorted((v, w) for v, w in zip(values, weights, strict=True)
                   if v is not None and w)
    total = sum(w for _, w in pairs)
    if not total:
        return None

    def at(k: int) -> float:
        acc = 0
        for v, w in pairs:
            acc += w
            if k < acc:
                return v
        return pairs[-1][0]

    if total % 2:
        return float(at(total // 2))
    return (at(total // 2 - 1) + at(total // 2)) / 2


def station_tables(monthly: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(stations, station_positions, station_years) from the per-month runs.

    * `stations` - one row per site in the schema every points store uses
      (`station_id, lat, lon, height, name`), taking position and raw height
      from the site's most recent report: the only single position that is
      not itself an average. Plus when it reported, how often, whether any of
      its reports were official, and how many positions it reported.
    * `station_positions` - the full history: one row per run of identical
      reported position.
    * `station_years` - per site and year, the median position over all its
      reports that year, the raw median height and how many reports gave
      height 0 (WOW's "not set", in most years - deciding that is a derived
      step's job).

    Built from runs split at month boundaries, so a site-year is exact.
    """
    import polars as pl

    m = monthly.filter(pl.col("station_id").is_not_null())
    history = merge_segments(m)
    if history.is_empty():
        return pl.DataFrame(), history, pl.DataFrame()
    stations = history.sort("last_time").group_by("station_id").agg(
        pl.col("Latitude").last().alias("lat"), pl.col("Longitude").last().alias("lon"),
        pl.col("Height").last().alias("height"),
        pl.col("first_time").min(), pl.col("last_time").max(),
        pl.col("n_reports").sum(), pl.col("n_official").sum(),
        pl.len().cast(pl.Int64).alias("n_position_runs"),
        pl.struct("Latitude", "Longitude", "Height").n_unique().cast(pl.Int64)
        .alias("n_positions"),
    ).with_columns(
        (pl.col("n_official") > 0).alias("official"),
        pl.lit(None, pl.String).alias("name"),
    ).sort("station_id")

    rows = []
    by_year = m.with_columns(pl.col("first_time").dt.year().alias("year"))
    for (sid, year), g in by_year.group_by(["station_id", "year"]):
        w = g["n_reports"].to_list()
        heights = g["Height"].to_list()
        rows.append({
            "station_id": sid, "year": int(year),
            "lat_median": _weighted_median(g["Latitude"].to_list(), w),
            "lon_median": _weighted_median(g["Longitude"].to_list(), w),
            "height_median_raw": _weighted_median(heights, w),
            "n_height_zero": int(sum(n for x, n in zip(heights, w, strict=True) if x == 0)),
            "n_reports": int(sum(w)),
            "n_official": int(g["n_official"].sum()),
            "n_positions": int(g.select(pl.struct("Latitude", "Longitude", "Height")
                                        .n_unique()).item()),
            "first_time": g["first_time"].min(), "last_time": g["last_time"].max(),
        })
    years = pl.DataFrame(rows, infer_schema_length=None).sort("station_id", "year")
    return stations, history, years


# ------------------------------------------------------------------- source


class WowNl:
    """WOW-NL hourly JSON archive as a `meteopack` source.

    Public API::

        src.list(window, cfg)           -> one Unit per yearly zip
        src.fetch(unit, raw_root)       -> RawFile (the raw layer links the archive)
        src.parse(raw, cfg)             -> every report of one zip (small zips/tests)
        src.pack_parts(files, cfg, w)   -> one part per UTC day, across all zips
        src.parse_part(paths, day, cfg) -> (rows, hours records) for one day
        src.finish_partition(frame, cfg)-> sorted month + its position runs
        src.finish_pack(...)            -> hours, station tables, pack report

    The last four are the optional capabilities `meteopack pack` uses for a
    source whose raw files are too big to be a unit of work (see
    `meteopack.jobs.packing.run_pipeline`).
    """

    name = "wow_nl"

    def __init__(self, source: SourceConfig | None = None, *, raw_root: Path | None = None,
                 **_ignored) -> None:
        self.source = source or SourceConfig(
            name="wow_nl", base_url=f"file://{ARCHIVE_DIR}" if ARCHIVE_DIR else None)
        self.raw_root = Path(raw_root) if raw_root else None
        self._catalogues: dict[str, Catalogue] = {}

    def catalogue(self, cfg: ProductConfig) -> Catalogue:
        key = cfg.store_id
        if key not in self._catalogues:
            self._catalogues[key] = Catalogue(cfg)
        return self._catalogues[key]

    @property
    def archive_dir(self) -> Path:
        override = self.source.extra.get("archive_dir")
        if override:
            return Path(str(override)).expanduser()
        where = self.source.base_url or ARCHIVE_DIR
        if not where or "${" in where:
            raise ValueError("no WOW-NL archive: set $WOW_NL_ARCHIVE or extra.archive_dir")
        return local_path(where).expanduser()

    # ------------------------------------------------------------- list/fetch

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """One unit per yearly zip whose hours overlap the window.

        The zip's span is read from its central directory, not its name:
        2020.zip runs 2019-01-01 .. 2020-06-29.
        """
        units = []
        for path in sorted(self.archive_dir.glob("*.zip")):
            if not _ZIP_RE.match(path.name):
                continue
            hours = sorted(zip_index(path)[1])
            if not hours:
                continue
            first, last = hours[0], hours[-1] + timedelta(hours=1)
            if last <= window.start or first >= window.end:
                continue
            units.append(Unit(key=f"{cfg.source}/{cfg.product}/{path.name}",
                              url=f"file://{path}", valid_time=first,
                              meta={"filename": path.name, "first_hour": first.isoformat(),
                                    "last_hour": hours[-1].isoformat()}))
        return units

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Make the zip visible in the raw layer - by a link, never a copy.

        The archive is 66 GB, read-only and never rewritten, so the raw layer
        is a symlink to it (`raw/stations/wow_nl/reports -> <archive>`). Fetch creates
        that link when the raw directory does not exist yet, and otherwise only
        checks that the zip is there and that its central directory reads.
        The digest recorded is size and mtime, as for the radar tars: hashing
        66 GB over NFS buys nothing a re-read would not.
        """
        src = local_path(unit.url)
        raw_root = Path(raw_root)
        if not raw_root.exists() and not raw_root.is_symlink():
            raw_root.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(src.parent, raw_root)
        dest = raw_root / src.name
        if not dest.exists():
            if not src.exists():
                return None
            raise FileNotFoundError(
                f"{dest} is missing although {src} exists: {raw_root} is not a link to the "
                f"archive. Remove it and re-run fetch, or link it by hand.")
        zip_index(dest)
        st = dest.stat()
        return RawFile(path=dest, sha256=f"size:{st.st_size}:mtime:{int(st.st_mtime)}",
                       n_bytes=st.st_size, unit_key=unit.key)

    # ------------------------------------------------------------------ parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> pl.DataFrame:
        """Every report of one zip, exact duplicates within an hour removed.

        For tests and small archives: a recent yearly zip is ~85 M reports. The
        pack never calls this; it works a day at a time across all zips.
        """
        import polars as pl

        _, index = zip_index(raw.path)
        days = sorted({h.date() for h in index})
        frames = [f for f, _ in (self.parse_part([str(raw.path)], d.isoformat(), cfg)
                                 for d in days) if f is not None]
        return pl.concat(frames, how="vertical") if frames \
            else pl.DataFrame(schema=self.catalogue(cfg).columns)

    def pack_parts(self, files: list[Path], cfg: ProductConfig,
                   window: TimeWindow | None) -> list[tuple[str, tuple[str, ...], str]]:
        """One part per UTC day: `(day, zips that hold any of its hours, partition)`.

        Chronological, so that months complete in order and the pack can write
        each one as soon as its last day is parsed, instead of holding the
        whole archive in a spill.
        """
        zips = [p for p in files if _ZIP_RE.match(Path(p).name)]
        days: dict[date, list[str]] = defaultdict(list)
        for path in zips:
            _, index = zip_index(path)
            for d in sorted({h.date() for h in index}):
                days[d].append(str(path))
        out = []
        for d in sorted(days):
            when = datetime(d.year, d.month, d.day, tzinfo=UTC)
            if window is not None and not (window.start <= when < window.end):
                continue
            out.append((d.isoformat(), tuple(days[d]), f"{d.year:04d}/{d.month:02d}"))
        return out

    def parse_part(self, paths: list[str], part: str, cfg: ProductConfig
                   ) -> tuple[pl.DataFrame | None, list[dict]]:
        page_cap = int(cfg.extra.get("page_cap_min_records", 9_900))
        return parse_day(list(paths), date.fromisoformat(part), self.catalogue(cfg),
                         page_cap=page_cap)

    # ------------------------------------------------------------- partitions

    def finish_partition(self, frame: pl.DataFrame, cfg: ProductConfig
                         ) -> tuple[pl.DataFrame, dict]:
        """Sort one month by (station, time) and describe it.

        A stable sort on (station, time) alone: the pipeline hands the month's
        days over in day order and a day's rows come out of `parse_day` in a
        fixed order, so two reports a station made in the same second keep
        the order the archive gave them, and the file is the same on every
        run. Breaking that tie with Id and provenance instead cost three more
        string keys in the sort.
        """
        keys = list(cfg.store.sort_by or ["station_id", "time"])
        # Measured on a 7.2 M-row month (5.0 GiB in memory, 8 threads, cn114):
        # `sort` on five keys 18 s at 17.5 GB; reordering by index a column at
        # a time 48 s at 13.2 GB, 16 columns at a time 25 s at 15.3 GB.
        frame = frame.sort(keys, nulls_last=True, maintain_order=True, multithreaded=True)
        runs = segments(frame)
        return frame, {
            "segments_ipc": _to_ipc(runs),
            "time_min": str(frame["time"].min()), "time_max": str(frame["time"].max()),
        }

    # --------------------------------------------------------------- the pack

    def finish_pack(self, root: Path, cfg: ProductConfig, *, result, window: TimeWindow | None,
                    files: list[Path], timing: dict | None = None) -> dict:
        """Write the side tables and the pack report once the partitions are in.

        * `hours/year=Y/month=M/data.parquet` - one row per member of every
          zip whose hour falls in that month, plus one `missing` row per hour
          no zip has (Jul-Dec 2020). Rewritten only for months this pack
          covered and completed, so it always agrees with the partitions.
        * `station_segments/year=Y/month=M/data.parquet` - the month's
          position runs, from which `stations/`, `station_positions.parquet`
          and `station_years.parquet` are rebuilt store-wide on every pack.
        * `pack_report.json` - store-wide counts from the hours tables, plus
          what this run did and how long it took.

        Returns `n_records` for the whole store (not just this run) and the
        errors that must fail the pack.
        """
        import time as _time

        import polars as pl

        from meteopack.core.store import points

        t0 = _time.time()
        store = points.store_path(root, cfg.store_id)
        records = [rec for r in result.parsed for rec in (r.get("records") or ())]
        hours = hours_frame(records)

        # The months this pack is responsible for: every month of the archive
        # span (or of the window), whether or not any zip has an hour in it.
        span = sorted(h for p in files if _ZIP_RE.match(Path(p).name)
                      for h in zip_index(p)[1])
        errors: list[str] = []
        notes: list[str] = []
        if not span:
            return {"n_records": 0, "errors": ["no hour members in the archive"]}
        first = span[0].replace(tzinfo=None)
        last = span[-1].replace(tzinfo=None)
        lo = first if window is None else max(first, window.start.astimezone(UTC).replace(tzinfo=None))
        hi = last + timedelta(hours=1) if window is None else min(
            last + timedelta(hours=1), window.end.astimezone(UTC).replace(tzinfo=None))
        failed = {label for label, why in result.skipped.items() if why != "no rows"}
        written = {r["label"]: r for r in result.written}
        months = []
        m = lo.replace(day=1, hour=0)
        while m < hi:
            months.append(m)
            m = m.replace(year=m.year + (m.month == 12), month=m.month % 12 + 1)

        hour_col = hours.with_columns(
            pl.col("hour").dt.strftime("%Y/%m").alias("__label"))
        for month in months:
            label = f"{month.year:04d}/{month.month:02d}"
            directory = f"year={month.year:04d}/month={month.month:02d}"
            if label in failed:
                notes.append(f"{label}: tables left as they were ({result.skipped[label]})")
                continue
            got = hour_col.filter(pl.col("__label") == label).drop("__label")
            end = month.replace(year=month.year + (month.month == 12), month=month.month % 12 + 1)
            expected = pl.datetime_range(max(month, first), min(end, last + timedelta(hours=1)),
                                         interval="1h", closed="left", eager=True,
                                         time_unit="us")
            missing = pl.DataFrame({"hour": expected}).join(got.select("hour").unique(),
                                                             on="hour", how="anti")
            if missing.height:
                miss = hours_frame([]).vstack(hours_frame([
                    {"hour": h, "status": "missing", "n_records": 0, "n_stored": 0,
                     "n_copies": 0, "chosen": False}
                    for h in missing["hour"].to_list()]))
                got = pl.concat([got, miss], how="vertical")
            _write(got.sort("hour", "zip", nulls_last=True), store / HOURS_DIR / directory
                   / "data.parquet")
            seg_path = store / SEGMENTS_DIR / directory / "data.parquet"
            if label in written:
                _write(_from_ipc(written[label]["info"]["segments_ipc"]), seg_path)
            else:
                # In scope, completed, and no rows at all: whatever an earlier
                # pack left for this month is not this archive's any more.
                seg_path.unlink(missing_ok=True)
                obs = store / points.OBSERVATIONS_DIR / directory
                for f in obs.glob("*.parquet") if obs.is_dir() else ():
                    f.unlink()

        # Store-wide: everything below is rebuilt from all months on disk.
        all_hours = _read_all(store / HOURS_DIR)
        all_segments = _read_all(store / SEGMENTS_DIR)
        stations, history, years = station_tables(all_segments)
        if stations.height:
            points.write_stations(stations.select(
                "station_id", "lat", "lon", "height", "name", "official", "first_time",
                "last_time", "n_reports", "n_official", "n_positions", "n_position_runs"),
                cfg.store_id, root)
            _write(history, store / "station_positions.parquet")
            _write(years, store / "station_years.parquet")

        report = self._report(all_hours, result, window, timing)
        report["finish_s"] = _time.time() - t0
        # Where the parse workers' time went in this run: reading members out
        # of the zips (filesystem + inflate) against decoding them (CPU).
        report["run"]["member_read_s"] = float(hours["read_s"].sum() or 0)
        report["run"]["member_decode_s"] = float(hours["decode_s"].sum() or 0)
        report["run"]["json_bytes"] = int(hours["bytes"].sum() or 0)
        report["run"]["compressed_bytes"] = int(hours["compressed_bytes"].sum() or 0)
        (store / "pack_report.json").write_text(json.dumps(report, indent=2, default=str))

        # What fails the pack. Malformed values (a number sent as text) are
        # data and are kept; a field nobody pinned is drift; an hour whose
        # every copy was unreadable or damaged has lost reports.
        by_reason = report["unpinned_by_field_reason"]
        drift = {k: v for k, v in by_reason.items() if k.endswith("|unknown_field")}
        if drift:
            errors.append(f"{len(drift)} field(s) not in the pinned catalogue (kept in "
                          f"unpinned_fields): {sorted(drift)[:8]}; re-pin extra.fields")
        malformed = {k: v for k, v in by_reason.items() if not k.endswith("|unknown_field")}
        if malformed:
            notes.append(f"values kept verbatim in unpinned_fields (did not fit their pinned "
                         f"type): {dict(sorted(malformed.items(), key=lambda kv: -kv[1])[:10])}")
        if report["hours_lost"]:
            errors.append(f"{len(report['hours_lost'])} hour(s) whose only copies are "
                          f"damaged or unreadable: {report['hours_lost'][:5]}")
        if report["members"].get("salvaged"):
            notes.append(f"{report['members']['salvaged']} damaged member(s) salvaged; every one "
                         f"has an intact copy of its hour in another zip"
                         if not report["hours_lost"] else "")
        if report["rows_outside_their_hour"]:
            notes.append(f"{report['rows_outside_their_hour']} report(s) stamped outside their "
                         f"member's hour (kept, partitioned by their own time)")
        run = report["run"]
        if run.get("parse_busy_s"):
            notes.append(
                f"parse workers: {run['parse_busy_s']:.0f} s busy, of which reading members "
                f"{run['member_read_s']:.0f} s and decoding {run['member_decode_s']:.0f} s; "
                f"{run['json_bytes'] / 2**30:.1f} GiB of JSON from "
                f"{run['compressed_bytes'] / 2**30:.2f} GiB of zip")
        notes.insert(0, (
            f"store: {report['rows_stored']:,} reports from {report['records_raw']:,} raw "
            f"records; {report['exact_duplicates_dropped']:,} exact duplicate(s) dropped "
            f"({report['exact_duplicates_dropped_other_copy']:,} as the second copy of an hour "
            f"present in two zips); {report['hours_missing']:,} hour(s) missing from the "
            f"archive; {report['hours_at_page_cap']:,} at the page cap"))
        return {"n_records": report["rows_stored"], "errors": errors,
                "notes": [n for n in notes if n]}

    @staticmethod
    def _report(hours: pl.DataFrame, result, window, timing) -> dict:
        """Store-wide accounting, read back from every hours table on disk."""
        import collections

        import polars as pl

        from meteopack.jobs.packing import memory_peak_bytes

        present = hours.filter(pl.col("status") != "missing")
        reasons: collections.Counter = collections.Counter()
        for blob in present["unpinned_counts"].drop_nulls().to_list():
            reasons.update(json.loads(blob))
        good = present.filter(pl.col("status") == "ok").select("hour").unique()
        damaged = present.filter(pl.col("status").is_in(["salvaged", "error"]))
        lost = damaged.join(good, on="hour", how="anti")["hour"].unique().sort()
        other_copy = present.filter(~pl.col("chosen"))
        stats = result.stats
        return {
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "window": None if window is None else [str(window.start), str(window.end)],
            "records_raw": int(present["n_records"].sum()),
            "rows_stored": int(present["n_stored"].sum()),
            "exact_duplicates_dropped": int(present["n_duplicate_dropped"].sum()),
            "exact_duplicates_dropped_other_copy": int(other_copy["n_duplicate_dropped"].sum()),
            "exact_duplicates_dropped_within_member": int(
                present.filter(pl.col("n_copies") == 1)["n_duplicate_dropped"].sum()),
            "rows_kept_from_other_copy": int(other_copy["n_stored"].sum()),
            "members": dict(collections.Counter(present["status"].to_list())),
            "decode_paths": dict(collections.Counter(present["decode_path"].to_list())),
            "hours_in_archive": int(present["hour"].n_unique()),
            "hours_missing": int((hours["status"] == "missing").sum()),
            "hours_with_two_copies": int(present.filter(pl.col("n_copies") > 1)["hour"]
                                         .n_unique()),
            "hours_at_page_cap": int(present.filter(pl.col("chosen") & pl.col("at_page_cap"))
                                     .height),
            "hours_lost": [str(h) for h in lost.to_list()],
            "rows_outside_their_hour": int(present["n_outside_hour"].sum()),
            "rows_with_unpinned_fields": int(present["n_unpinned"].sum()),
            "unpinned_by_field_reason": dict(reasons),
            "by_year": [
                {k: v for k, v in r.items()} for r in present.group_by(
                    pl.col("hour").dt.year().alias("year")).agg(
                    pl.col("n_records").sum().alias("records_raw"),
                    pl.col("n_stored").sum().alias("rows_stored"),
                    pl.col("n_duplicate_dropped").sum().alias("exact_duplicates_dropped"),
                    (pl.col("chosen") & pl.col("at_page_cap")).sum().alias("hours_at_page_cap"),
                ).sort("year").iter_rows(named=True)],
            "run": {**{k: v for k, v in stats.items()},
                    "job_memory_peak_bytes": memory_peak_bytes(),
                    "cli_started": (timing or {}).get("cli_started")},
        }

    def validate_store(self, cfg: ProductConfig, root: Path, report, *,
                       raw_root: Path | None = None) -> None:
        """The checks that make this store trustworthy, added to `report`.

        * every member of every zip is in the hours table exactly once, and
          every hour from the archive's first to its last has a row (present
          or `missing`) - nothing in the archive is unaccounted for;
        * per member, the rows stored are the rows the hours table says were
          kept; per store, rows = sum(n_stored);
        * no report is stored twice: no two rows identical in every field;
        * every row's time lies inside its member's hour;
        * nothing went to `unpinned_fields` as an unknown field.

        All of it is DuckDB over the parquet, one partition at a time, so it
        streams; on the full store it takes about a minute.
        """
        import duckdb
        import polars as pl

        from meteopack.core.store import points
        from meteopack.jobs.packing import memory_limit_bytes

        store = points.store_path(root, cfg.store_id)
        hours = _read_all(store / HOURS_DIR)
        report.add("hours table present", hours.height > 0, str(store / HOURS_DIR))
        if not hours.height:
            return
        present = hours.filter(pl.col("status") != "missing")

        # 1. every member of every zip, once; every hour of the span, once or missing.
        raw_root = Path(raw_root) if raw_root else None
        zips = sorted(p for p in (raw_root.iterdir() if raw_root and raw_root.is_dir() else ())
                      if _ZIP_RE.match(p.name))
        if zips:
            listed = pl.DataFrame(
                [{"zip": z.name, "member": info.filename} for z in zips
                 for info in zip_index(z)[1].values()],
                schema={"zip": pl.String, "member": pl.String})
            have = present.select("zip", "member")
            unaccounted = listed.join(have, on=["zip", "member"], how="anti").height
            unknown = have.join(listed, on=["zip", "member"], how="anti").height
            twice = int(have.is_duplicated().sum())
            report.add("every zip member is in the hours table, once",
                       unaccounted == 0 and twice == 0 and unknown == 0,
                       f"{listed.height:,} members in {len(zips)} zips; {unaccounted} not in the "
                       f"table, {twice} listed twice, {unknown} in the table but not in a zip")
        span = pl.datetime_range(hours["hour"].min(), hours["hour"].max(), interval="1h",
                                 eager=True, time_unit="us")
        gaps = pl.DataFrame({"hour": span}).join(hours.select("hour").unique(), on="hour",
                                                 how="anti").height
        missing = hours.filter(pl.col("status") == "missing")
        report.add("every hour of the archive span has a row", gaps == 0,
                   f"{span.len():,} hours {span.min()} .. {span.max()}; "
                   f"{missing.height:,} marked missing (no zip has them)")
        report.stats["hours:missing"] = missing.height
        if missing.height:
            runs = (missing.sort("hour").with_columns(
                (pl.col("hour").diff() != pl.duration(hours=1)).fill_null(True).cum_sum()
                .alias("r")).group_by("r").agg(pl.col("hour").min().alias("a"),
                                                pl.col("hour").max().alias("b"), pl.len())
                .sort("len", descending=True))
            report.stats["hours:missing_longest"] = (
                f"{runs['a'][0]} .. {runs['b'][0]} ({runs['len'][0]} h)")

        # 2/3/4: one DuckDB pass per partition.
        con = duckdb.connect()
        limit = memory_limit_bytes() or 8 << 30
        con.execute(f"SET memory_limit='{max(2, int(0.5 * limit / 2**30))}GB'")
        con.execute(f"SET threads={max(1, len(os.sched_getaffinity(0)))}")
        con.execute("SET preserve_insertion_order=false")
        con.execute("PRAGMA disable_progress_bar")
        files = sorted((store / points.OBSERVATIONS_DIR).rglob("*.parquet"))
        cols = [r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{files[0]}', hive_partitioning=false)"
        ).fetchall()] if files else []
        data_cols = ", ".join(f'"{c}"' for c in cols if c not in PROVENANCE)
        counts, dup_keys, dup_rows, outside, rows = [], 0, 0, 0, 0
        for f in files:
            src = f"read_parquet('{f}', hive_partitioning=false)"
            counts += con.execute(
                f"SELECT source_zip, source_member, count(*) FROM {src} GROUP BY ALL").fetchall()
            row = con.execute(f"""
                SELECT count(*),
                  count(*) FILTER (WHERE time < h OR time >= h + INTERVAL 1 HOUR)
                FROM (SELECT time, strptime(regexp_extract(source_member,
                        'WOW_NL_(\\d{{10}})', 1), '%Y%m%d%H') AS h FROM {src})""").fetchone()
            rows += int(row[0])
            outside += int(row[1])
            k = int(con.execute(f"""SELECT count(*) FROM (SELECT station_id, time, "Id"
                FROM {src} GROUP BY ALL HAVING count(*) > 1)""").fetchone()[0])
            dup_keys += k
            if k:
                # Only when some (station, time, Id) repeats: then compare whole
                # rows, by a 64-bit hash of every data column (a collision
                # would need ~2**32 rows to be likely; a month is 7 M).
                dup_rows += int(con.execute(f"""SELECT count(*) - count(DISTINCT h) FROM (
                    SELECT hash({data_cols}) AS h FROM {src})""").fetchone()[0])
        con.close()
        stored = int(present["n_stored"].sum())
        report.add("rows = sum of n_stored in the hours table", rows == stored,
                   f"{rows:,} rows in {len(files)} partition(s); hours table says {stored:,}")
        report.add("no report stored twice (rows identical in every field)", dup_rows == 0,
                   f"{dup_keys} (station, time, Id) key(s) occur more than once; "
                   f"{dup_rows} identical row(s)")
        report.add("every row lies inside its member's hour", outside == 0,
                   f"{outside} row(s) outside")
        got = pl.DataFrame(counts, schema={"zip": pl.String, "member": pl.String,
                                           "rows": pl.Int64}, orient="row")
        cmp = present.select("zip", "member", "n_stored").join(
            got, on=["zip", "member"], how="full", coalesce=True).with_columns(
            pl.col("n_stored").fill_null(0), pl.col("rows").fill_null(0))
        off = cmp.filter(pl.col("n_stored") != pl.col("rows"))
        report.add("rows per member match the hours table", off.height == 0,
                   f"{off.height} member(s) differ" + (
                       f", e.g. {off.head(3).to_dicts()}" if off.height else ""))
        drift = {}
        for blob in present["unpinned_counts"].drop_nulls().to_list():
            for k, v in json.loads(blob).items():
                drift[k] = drift.get(k, 0) + v
        unknown = {k: v for k, v in drift.items() if k.endswith("|unknown_field")}
        report.add("no field outside the pinned catalogue", not unknown,
                   f"{sum(unknown.values())} value(s) of {len(unknown)} unknown field(s)"
                   if unknown else "every field is pinned")
        report.stats["records_raw"] = int(present["n_records"].sum())
        report.stats["exact_duplicates_dropped"] = int(present["n_duplicate_dropped"].sum())
        report.stats["members"] = dict(present.group_by("status").len().iter_rows())
        report.stats["hours_at_page_cap"] = int(
            present.filter(pl.col("chosen") & pl.col("at_page_cap")).height)
        report.stats["unpinned_values"] = dict(sorted(drift.items(), key=lambda kv: -kv[1])) \
            if drift else 0


#: Side tables of the store, one file per month like the observations.
HOURS_DIR = "hours"
SEGMENTS_DIR = "station_segments"


def _write(frame: pl.DataFrame, path: Path) -> None:
    import uuid

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    frame.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, path)


def _read_all(directory: Path) -> pl.DataFrame:
    import polars as pl

    found = sorted(directory.rglob("*.parquet")) if directory.is_dir() else []
    if not found:
        return pl.DataFrame()
    return pl.concat([pl.read_parquet(f) for f in found], how="diagonal_relaxed")


def _to_ipc(frame: pl.DataFrame) -> bytes:
    import io

    buf = io.BytesIO()
    frame.write_ipc(buf)
    return buf.getvalue()


def _from_ipc(blob: bytes) -> pl.DataFrame:
    import io

    import polars as pl

    return pl.read_ipc(io.BytesIO(blob))
