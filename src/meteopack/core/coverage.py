"""What a source's raw layer is missing, and why.

`fetch` knows, unit by unit, what it was asked to get and what happened; the
ledger keeps that. What it did not keep was the question: the listing the run
worked from. Without it "missing" has no denominator - a unit that was never
attempted leaves no trace at all, and neither does a station-period the
provider describes but publishes no file for. So `fetch` now writes the
listing it planned from beside the ledger, and this module sets three things
against each other:

* the listing (``_bookkeeping/listings/<category>/<source>/<product>/``): every unit
  upstream offered, with the station, component and period it stands for, plus
  - where the source can say - what its station catalogue claims should exist;
* the ledger (`jobs`): the outcome of every attempt, with the reason;
* the `files` table and the disk: what is actually there, and which span of
  time each file holds (from the source's integrity check at fetch time).

The answer comes at two levels: per unit (one file), and per station x
component x year, which is the level at which somebody asks "do we have
station 00044's wind for 2015, and if not, why not".

Nothing here touches the network or writes anything but the listing.
"""
from __future__ import annotations

import os
import uuid
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from meteopack.core import raw_compression
from meteopack.core.config import category_for_source

if TYPE_CHECKING:
    from collections.abc import Iterable

    from meteopack.core.catalog import Catalog
    from meteopack.sources.base import TimeWindow, Unit

LISTINGS_DIRNAME = "listings"

_MICRO = timedelta(microseconds=1)

#: Why a unit or a station-period is in the state it is, best first. A
#: station-period covered by several units takes the best of their states.
COMPLETE = "complete"
PARTIAL = "partial"
FILLING = "still changing upstream"
EMPTY = "empty upstream"
FAILED = "failed"
VANISHED = "missing on disk"
NOT_DOWNLOADED = "not downloaded"
ABSENT = "absent upstream"
NOT_LISTED = "not listed upstream"
REASONS = (COMPLETE, PARTIAL, FILLING, EMPTY, FAILED, VANISHED, NOT_DOWNLOADED, ABSENT,
           NOT_LISTED)

#: What each reason means, printed under the report so it reads on its own.
EXPLAIN = {
    COMPLETE: "on disk and spans its period",
    PARTIAL: "on disk, but the data start later or end earlier than the period",
    FILLING: "fetched, still being rewritten upstream; fetched again next run",
    EMPTY: "upstream answered with no data (e.g. a year before a gauge opened)",
    FAILED: "gave up after retries; retried on the next run (see the error)",
    VANISHED: "the ledger says fetched but the file is gone; the next fetch gets it again",
    NOT_DOWNLOADED: "listed upstream, never attempted (fetch has not reached it)",
    ABSENT: "listed, but upstream answered 404",
    NOT_LISTED: "the station catalogue describes it, but upstream offers no file for it",
}

#: Column headings for the per-year table.
_SHORT = {COMPLETE: "complete", PARTIAL: "partial", FILLING: "changing", EMPTY: "empty",
          FAILED: "failed", VANISHED: "gone", NOT_DOWNLOADED: "not-fetched",
          ABSENT: "absent-404", NOT_LISTED: "not-listed"}

_LISTING_SCHEMA: dict[str, pl.DataType] = {
    "kind": pl.String,              # "unit" or "expected"
    "unit_key": pl.String,          # null for "expected" rows
    "url": pl.String,
    "station_id": pl.String,
    "component": pl.String,
    "period_start": pl.Date,
    "period_end": pl.Date,
    "period_nominal": pl.Boolean,   # true: a nominal span, not checked against content
    "window_start": pl.Datetime("us"),
    "window_end": pl.Datetime("us"),
    "listed_at": pl.Datetime("us"),
}


def listings_dir(catalog_root: Path, source: str, product: str) -> Path:
    root = Path(catalog_root) / LISTINGS_DIRNAME
    category = category_for_source(source)
    if category is None:
        raise ValueError(f"no storage category configured for source {source!r}")
    return root / category / source / product


def _naive(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


def _as_date(text: str | None) -> date | None:
    try:
        return date.fromisoformat(str(text)[:10]) if text else None
    except ValueError:
        return None


def listing_frame(
    units: Iterable[Unit],
    expected: Iterable[dict],
    window: TimeWindow,
    *,
    listed_at: datetime | None = None,
) -> pl.DataFrame:
    """The listing a fetch planned from, as one table.

    A unit's station, component and period come from its `meta`
    (`station_id`, `component`, `period_start`, `period_end`,
    `period_nominal`); a source that sets none of them still gets a unit-level
    report, keyed on `valid_time`.
    """
    when = _naive(listed_at or datetime.now(UTC))
    ws, we = _naive(window.start), _naive(window.end)
    rows = []
    for u in units:
        meta = u.meta or {}
        start = _as_date(meta.get("period_start"))
        if start is None and u.valid_time is not None:
            start = u.valid_time.date()
        rows.append({
            "kind": "unit", "unit_key": u.key, "url": u.url,
            "station_id": meta.get("station_id"), "component": meta.get("component") or "",
            "period_start": start,
            "period_end": _as_date(meta.get("period_end")) or start,
            "period_nominal": str(meta.get("period_nominal", "")) in ("1", "true", "True"),
            "window_start": ws, "window_end": we, "listed_at": when,
        })
    for e in expected:
        rows.append({
            "kind": "expected", "unit_key": None, "url": None,
            "station_id": e["station_id"], "component": e.get("component") or "",
            "period_start": e["start"], "period_end": e["end"], "period_nominal": False,
            "window_start": ws, "window_end": we, "listed_at": when,
        })
    return pl.DataFrame(rows, schema=_LISTING_SCHEMA)


def write_listing(catalog_root: Path, source: str, product: str, frame: pl.DataFrame,
                  writer_id: str) -> Path:
    """Publish a listing atomically, one file per fetch run (newest name last)."""
    base = listings_dir(catalog_root, source, product)
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"{writer_id}.parquet"
    tmp = base / f".{path.name}.{uuid.uuid4().hex}.tmp"
    frame.write_parquet(tmp)
    os.replace(tmp, path)
    return path


def latest_listing(catalog_root: Path, source: str,
                   product: str) -> tuple[pl.DataFrame | None, Path | None]:
    """The newest listing for this product. Writer ids lead with a timestamp,
    so the last name in sort order is the newest run."""
    base = listings_dir(catalog_root, source, product)
    found = sorted(base.glob("*.parquet")) if base.is_dir() else []
    if not found:
        return None, None
    return pl.read_parquet(found[-1]), found[-1]


# --------------------------------------------------------------------- units

def unit_status(catalog: Catalog, listing: pl.DataFrame) -> pl.DataFrame:
    """One row per listed unit: its state, the reason, and the evidence."""
    units = listing.filter(pl.col("kind") == "unit").unique("unit_key", keep="last")
    keys = units.select(pl.col("unit_key"), ("fetch:" + pl.col("unit_key")).alias("job_key"))

    jobs = catalog.query(
        "SELECT job_key, status, outcome, attempts, http_status, error_class, error, "
        "updated_at FROM jobs WHERE stage = 'fetch'")
    failed_runs = catalog.query(
        "SELECT job_key, count(*) AS failed_runs FROM jobs_all "
        "WHERE stage = 'fetch' AND status = 'failed' GROUP BY job_key")
    files = catalog.query(
        "SELECT unit_key, path, n_bytes, n_records, time_min, time_max, fetched_at FROM files")

    frame = (
        units.join(keys, on="unit_key", how="left")
        .join(jobs, on="job_key", how="left")
        .join(failed_runs, on="job_key", how="left")
        .join(files, on="unit_key", how="left")
    )
    paths = frame["path"].to_list()
    on_disk = [bool(p) and raw_compression.stored_path(p).exists() for p in paths]
    frame = frame.with_columns(pl.Series("on_disk", on_disk, dtype=pl.Boolean))
    reasons, details = [], []
    for row in frame.iter_rows(named=True):
        reason, detail = _unit_reason(row)
        reasons.append(reason)
        details.append(detail)
    return frame.with_columns(
        pl.Series("reason", reasons, dtype=pl.String),
        pl.Series("detail", details, dtype=pl.String),
        pl.col("failed_runs").fill_null(0),
    ).drop("job_key")


def _outcome(row: dict) -> str | None:
    """The recorded outcome, or the one implied by a row written before the
    `outcome` column existed (absent units were `done` with that error)."""
    if row.get("outcome"):
        return row["outcome"]
    status = row.get("status")
    if status is None:
        return None
    if status == "failed":
        return "failed"
    if (row.get("error") or "").startswith("absent upstream"):
        return "absent"
    return "done"


def _unit_reason(row: dict) -> tuple[str, str]:
    outcome = _outcome(row)
    status = row.get("status")
    if outcome is None:
        return NOT_DOWNLOADED, ""
    if outcome == "failed":
        bits = [row.get("error_class") or "error"]
        if row.get("http_status"):
            bits.append(f"HTTP {row['http_status']}")
        if row.get("attempts"):
            bits.append(f"after {row['attempts']} attempt(s)")
        when = row.get("updated_at")
        if when is not None:
            bits.append(f"at {when:%Y-%m-%d %H:%M} UTC")
        if row.get("failed_runs", 0) and row["failed_runs"] > 1:
            bits.append(f"failed in {row['failed_runs']} runs")
        if row.get("on_disk"):
            bits.append("an earlier copy is on disk")
        return FAILED, ", ".join(bits) + f": {row.get('error') or ''}".rstrip(": ")
    if outcome == "absent":
        return ABSENT, row.get("error") or "HTTP 404"
    if outcome == "empty" or row.get("n_records") == 0 or row.get("n_bytes") == 0:
        note = "; still changing upstream, fetched again next run" if status == "pending" else ""
        return EMPTY, f"{row.get('n_bytes') or 0} bytes{note}"
    if not row.get("on_disk"):
        return VANISHED, row.get("path") or "no file recorded"
    if status == "pending":
        return FILLING, _span(row)
    start, end = row.get("period_start"), row.get("period_end")
    tmin, tmax = row.get("time_min"), row.get("time_max")
    if row.get("period_nominal") or start is None or tmin is None or tmax is None:
        return COMPLETE, _span(row)
    listed = row.get("listed_at")
    if listed is not None:
        end = min(end, listed.date())
    if tmin.date() > start or tmax.date() < end:
        return PARTIAL, f"data {_span(row)} of period {start}..{end}"
    return COMPLETE, _span(row)


def _span(row: dict) -> str:
    tmin, tmax = row.get("time_min"), row.get("time_max")
    if tmin is None or tmax is None:
        return ""
    return f"{tmin:%Y-%m-%d %H:%M}..{tmax:%Y-%m-%d %H:%M}"


# --------------------------------------------------------- station x period

def period_status(units: pl.DataFrame, listing: pl.DataFrame) -> pl.DataFrame:
    """One row per station x component x year inside the listing's window.

    A unit stands for every year its period touches; a unit with only a
    nominal period (DWD's recent/ and now/) stands for the years its content
    actually spans once it is fetched. Rows from the station catalogue
    (`kind == expected`) that no unit covers are "not listed upstream".
    """
    if not len(listing):
        return pl.DataFrame(schema={"station_id": pl.String, "component": pl.String,
                                    "year": pl.Int32, "units": pl.UInt32,
                                    "reason": pl.String, "reasons": pl.String,
                                    "detail": pl.String})
    first_year = listing["window_start"].min().year
    last_year = (listing["window_end"].max() - _MICRO).year
    rank = {r: i for i, r in enumerate(REASONS)}

    u = units.with_columns(
        pl.when(pl.col("period_nominal") & pl.col("time_min").is_not_null())
        .then(pl.col("time_min").dt.year()).otherwise(pl.col("period_start").dt.year())
        .alias("y0"),
        pl.when(pl.col("period_nominal") & pl.col("time_max").is_not_null())
        .then(pl.col("time_max").dt.year()).otherwise(pl.col("period_end").dt.year())
        .alias("y1"),
    )
    u = _explode_years(u, first_year, last_year).select(
        "station_id", "component", "year", "reason", "detail",
        pl.lit(1, dtype=pl.UInt32).alias("n"))

    exp = listing.filter(pl.col("kind") == "expected").with_columns(
        pl.col("period_start").dt.year().alias("y0"),
        pl.col("period_end").dt.year().alias("y1"))
    exp = _explode_years(exp, first_year, last_year).select(
        "station_id", "component", "year",
        pl.lit(NOT_LISTED).alias("reason"), pl.lit("").alias("detail"),
        pl.lit(0, dtype=pl.UInt32).alias("n"))

    both = pl.concat([u, exp], how="vertical_relaxed").with_columns(
        pl.col("reason").replace_strict(rank, return_dtype=pl.Int32).alias("rank"))
    return (
        both.sort("rank")
        .group_by("station_id", "component", "year", maintain_order=True)
        .agg(
            pl.col("n").sum().alias("units"),
            pl.col("reason").first(),
            pl.col("reason").filter(pl.col("n") > 0).unique(maintain_order=True)
            .str.join(", ").alias("reasons"),
            pl.col("detail").first(),
        )
        .sort("station_id", "component", "year")
    )


def _explode_years(frame: pl.DataFrame, first: int, last: int) -> pl.DataFrame:
    frame = frame.filter(pl.col("y0").is_not_null() & pl.col("y1").is_not_null())
    return (
        frame.with_columns(
            pl.int_ranges(pl.max_horizontal(pl.col("y0"), pl.lit(first)),
                          pl.min_horizontal(pl.col("y1"), pl.lit(last)) + 1,
                          dtype=pl.Int32).alias("year"))
        .explode("year")
        .filter(pl.col("year").is_not_null())
    )


# --------------------------------------------------------------------- text

def report(catalog: Catalog, source: str, product: str, *, window: TimeWindow | None = None,
           out: Path | None = None, show: int = 30) -> tuple[str, int]:
    """The gap report as text, and the number of station-periods not complete.

    With `out`, the full unit and station-period tables are written there as
    CSV, since a terminal holds a summary and not 14,000 rows.
    """
    listing, path = latest_listing(catalog.root, source, product)
    lines = [f"gaps for {source}/{product}"]
    if listing is None:
        lines.append("  no listing recorded yet - `meteopack fetch` writes one each run")
        return "\n".join(lines), 0
    if window is not None:
        ws, we = _naive(window.start).date(), _naive(window.end).date()
        listing = listing.filter(
            (pl.col("period_end") >= ws) & (pl.col("period_start") < we)
        ).with_columns(pl.lit(_naive(window.start)).alias("window_start"),
                       pl.lit(_naive(window.end)).alias("window_end"))
    first = listing.row(0, named=True) if len(listing) else None
    units = unit_status(catalog, listing)
    periods = period_status(units, listing)

    if first is not None:
        lines.append(
            f"  listing of {first['listed_at']:%Y-%m-%d %H:%M} UTC, window "
            f"{first['window_start']:%Y-%m-%d} .. {first['window_end']:%Y-%m-%d} (end exclusive)")
    lines.append(f"  from {path}")

    lines.append(f"\n  units (files): {len(units):,} listed")
    by = units.group_by("reason").agg(pl.len().alias("n"),
                                      pl.col("n_bytes").sum().alias("bytes"))
    counts = {r["reason"]: r for r in by.iter_rows(named=True)}
    for reason in REASONS:
        if reason in counts:
            row = counts[reason]
            size = f"{(row['bytes'] or 0) / 1e9:8.2f} GB" if row["bytes"] else ""
            lines.append(f"    {reason:26s} {row['n']:>8,}  {size}")

    failed = units.filter(pl.col("reason") == FAILED)
    if len(failed):
        lines.append("\n  failures by error")
        grouped = failed.group_by("error_class", "http_status").agg(
            pl.len().alias("n"), pl.col("unit_key").first(), pl.col("error").first()
        ).sort("n", descending=True)
        for r in grouped.iter_rows(named=True):
            status = f" HTTP {r['http_status']}" if r["http_status"] else ""
            lines.append(f"    {r['error_class'] or 'error'}{status}: {r['n']:,}  "
                         f"e.g. {r['unit_key']}: {(r['error'] or '')[:160]}")

    lines.append(f"\n  station x component x year: {len(periods):,}")
    pc = periods.group_by("reason").len()
    pcounts = dict(zip(pc["reason"].to_list(), pc["len"].to_list(), strict=True))
    for reason in REASONS:
        if reason in pcounts:
            lines.append(f"    {reason:26s} {pcounts[reason]:>8,}")

    if len(periods):
        table = periods.pivot(on="reason", index="year", values="station_id",
                              aggregate_function="len").sort("year").fill_null(0)
        cols = [r for r in REASONS if r in table.columns]
        lines.append("\n  by year " + "".join(f"{_SHORT[c]:>12s}" for c in cols))
        for row in table.iter_rows(named=True):
            lines.append(f"    {row['year']:<4d}" + "".join(f"{row[c]:>12,}" for c in cols))

    gaps = periods.filter(pl.col("reason") != COMPLETE)
    if len(gaps):
        lines.append(f"\n  not complete: first {min(show, len(gaps))} of {len(gaps):,}"
                     + ("" if out else " (--out DIR writes them all)"))
        for r in gaps.head(show).iter_rows(named=True):
            comp = f"{r['component']:18s}" if r["component"] else ""
            lines.append(f"    {r['station_id'] or '-':>12s} {comp} {r['year']}  "
                         f"{r['reason']:24s} {(r['detail'] or '')[:90]}")

    lines.append("\n  reasons")
    for reason in REASONS:
        if reason in pcounts or reason in counts:
            lines.append(f"    {reason:26s} {EXPLAIN[reason]}")

    if out is not None:
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        units.with_columns(pl.col(pl.Datetime).dt.strftime("%Y-%m-%dT%H:%M:%S")).write_csv(
            out / f"{source}_{product}_units.csv")
        periods.write_csv(out / f"{source}_{product}_station_years.csv")
        lines.append(f"\n  wrote {out / f'{source}_{product}_units.csv'} and "
                     f"{out / f'{source}_{product}_station_years.csv'}")
    return "\n".join(lines), len(gaps)


__all__ = [
    "EXPLAIN",
    "LISTINGS_DIRNAME",
    "REASONS",
    "latest_listing",
    "listing_frame",
    "period_status",
    "report",
    "unit_status",
    "write_listing",
]
