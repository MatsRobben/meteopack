"""meteopack {scan,list,fetch,parse,pack,derive,validate,catalog} <target>

Every stage is re-runnable. For `fetch` the ledger decides what is left to do,
so running it twice is cheap and a preempted Slurm task costs at most its
in-flight unit; content still changing upstream is downloaded again. `pack`
rebuilds the stores it is given from the raw files, in parallel across one
allocation's cores, and refuses --shard/--shards.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import shutil
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meteopack.sources.base import RawFile, TimeWindow

logger = logging.getLogger("meteopack")

#: When this process got as far as importing the CLI: the zero for METEOPACK_TRACE.
_STARTED = time.time()

#: Stages that write into the data root. Everything else only reads it.
_WRITING_STAGES = frozenset({"fetch", "pack"})

STAGES = ("scan", "list", "fetch", "parse", "pack", "derive", "validate", "catalog",
          "credentials")


# --------------------------------------------------------------------- args

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meteopack", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("target", nargs="?", help="source name (scan/list/fetch/parse/pack/"
                                             "validate/catalog) or dataset name (derive); "
                                             "optional for `credentials`")
    p.add_argument("--product", help="product within the source; required if it has several")
    p.add_argument("--window", help="START/END. Dates and times are half-open: "
                                    "2012-01-01/2013-01-01 is the year 2012. A bare year or "
                                    "year-month is the whole period, both ends included: "
                                    "2012/2013 is two years, 2012-06/2012-06 is June. "
                                    "Default: the last 24 hours")
    p.add_argument("--data-root", help="overrides $METEOPACK_DATA_ROOT, which overrides the config")
    p.add_argument("--configs", help="config directory (default: <repo>/configs)")
    p.add_argument("--workers", type=int, default=None, help="default: all available cores")
    p.add_argument("--shard", type=int, default=0,
                   help="this task's index in a Slurm array (fetch/list/derive; not pack)")
    p.add_argument("--shards", type=int, default=1,
                   help="total tasks in the Slurm array (fetch/list/derive; not pack)")
    p.add_argument("--allow-partial", action="store_true",
                   help="pack: rewrite points partitions even if some raw files failed to "
                        "parse, dropping their rows (default: refuse and change nothing)")
    p.add_argument("--refresh", action="store_true", help="ignore the ledger and redo everything")
    p.add_argument("--limit", type=int, default=None, help="stop after N units (for smoke runs)")
    p.add_argument("--dry-run", action="store_true", help="plan only; touch nothing")
    p.add_argument("--gaps", action="store_true",
                   help="catalog: what the source's raw layer is missing and why, per file "
                        "and per station x year, from the last fetch's listing and the ledger")
    p.add_argument("--out", help="catalog --gaps: directory to write the full tables to (CSV)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="derive (kind: regularise): override one config value for this run, "
                        "e.g. --set step_seconds=300 --set rain.rule=naive (repeatable)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def parse_window(spec: str | None) -> TimeWindow:
    from meteopack.sources.base import TimeWindow

    if not spec:
        now = datetime.now(UTC)
        return TimeWindow(now - timedelta(days=1), now)
    start, _, end = spec.partition("/")
    if not end:
        raise SystemExit(f"--window needs START/END, got {spec!r}")

    def one(text: str, *, upper: bool) -> datetime:
        # A bare year or year-month names a whole period, so as an END it means
        # the end of that period: 2012/2013 is two years. A date or time is an
        # instant, and the window is half-open on it.
        text = text.strip()
        try:
            if len(text) == 4:                      # YYYY
                return datetime(int(text) + (1 if upper else 0), 1, 1, tzinfo=UTC)
            if len(text) == 7 and text[4] == "-":   # YYYY-MM
                year, month = int(text[:4]), int(text[5:])
                if upper:
                    year, month = (year + 1, 1) if month == 12 else (year, month + 1)
                return datetime(year, month, 1, tzinfo=UTC)
            value = datetime.fromisoformat(text)
        except ValueError:
            raise SystemExit(f"--window: cannot read {text!r}; use YYYY, YYYY-MM, a date or "
                             f"an ISO time, e.g. 2012-01-01/2013-01-01") from None
        return value if value.tzinfo else value.replace(tzinfo=UTC)

    return TimeWindow(one(start, upper=False), one(end, upper=True))


# ------------------------------------------------------------------ context

class Context:
    """Everything a stage needs, resolved once."""

    def __init__(self, args: argparse.Namespace) -> None:
        from meteopack.core import readme
        from meteopack.core.catalog import Catalog
        from meteopack.core.config import data_root
        from meteopack.jobs import plan as planning

        self.args = args
        self.started = _STARTED
        self.root = data_root(args.data_root, default=str(Path.cwd() / "data"))
        self.cfg = planning.load_product_config(args.target, args.product, configs=args.configs)
        self.source_cfg = planning.load_source_config(args.target, configs=args.configs)
        self.catalog = Catalog(self.root)
        self.window = parse_window(args.window)
        self._src = None
        # The data root outlives the checkout that made it, so it carries its
        # own explanation of what each directory holds. Only stages that write
        # data refresh it: `validate` or a dry run with no data root set used to
        # create ./data/ in whatever directory they ran in. Best effort.
        if args.stage in _WRITING_STAGES and not getattr(args, "dry_run", False):
            readme.write(self.root)

    @property
    def src(self):
        if self._src is None:
            import requests

            from meteopack.core.http import RateLimiter
            from meteopack.sources import registry

            # raw_root matters to sources that can recover state from the
            # archive on disk. `pack` parses in worker processes, so anything a
            # source accumulates during parse (meteofrance builds its station
            # registry that way) is gone by the time the parent writes station
            # metadata; the disk is the only thing both sides share.
            # registry.build drops the keyword for sources that do not take it.
            self._src = registry.build(
                self.args.target,
                self.source_cfg,
                session=requests.Session(),
                # The configured rate is what upstream should see in total; an
                # array's tasks run at once, so each takes its share. With 30
                # tasks at the full rate DWD would have seen 240 requests/s.
                limiter=RateLimiter(self.source_cfg.rate_limit.requests_per_second
                                    / max(1, self.args.shards)),
                raw_root=self.raw_root,
            )
        return self._src

    @property
    def raw_root(self) -> Path:
        return self.root / "raw" / self.cfg.source / self.cfg.product

    #: Files that sit in a raw directory without being data. `.part` is our own
    #: half-written download; `station_catalog.json` is a sidecar the previous
    #: pipeline wrote beside the observations, and pointing this one at that
    #: archive made every source try to parse it as a data file and report
    #: schema drift. Skipped by name rather than by extension, so a real
    #: `.json` observation file is still packed.
    SIDECARS = frozenset({"station_catalog.json"})

    def raw_files(self) -> list[Path]:
        return sorted(
            p for p in self.raw_root.rglob("*")
            if p.is_file() and p.suffix != ".part" and p.name not in self.SIDECARS
        )

    def plan(self, stage: str):
        from meteopack.jobs.plan import build_plan

        p = build_plan(stage, self.src, self.cfg, self.window, self.catalog,
                       shard=self.args.shard, shards=self.args.shards, refresh=self.args.refresh)
        if self.args.limit:
            p.units = p.units[: self.args.limit]
        return p


# ------------------------------------------------------------------- stages

def stage_list(ctx: Context) -> int:
    p = ctx.plan("fetch")
    print(p.summary())
    for unit in p.units[:10]:
        print(f"  {unit.key}  {unit.url}")
    if p.n > 10:
        print(f"  ... and {p.n - 10} more")
    return 0


#: How often `fetch` writes its ledger shard. Every unit used to rewrite the
#: whole shard - O(n^2) over a 15,000-unit run. A preempted task loses at most
#: this much bookkeeping, and the files it covers are on disk and pass their
#: check, so the next run re-records them without downloading.
_LEDGER_FLUSH_SECONDS = 30.0


def _outcome_of(raw) -> str:
    """`done` or `empty` for a unit `fetch` returned."""
    content = getattr(raw, "content", None)
    if raw.n_bytes == 0 or (content is not None and content.n_records == 0):
        return "empty"
    return "done"


def _failure_fields(exc: BaseException) -> dict:
    """What the ledger records about a failed unit: never just "failed"."""
    from meteopack.core.http import FetchError

    if isinstance(exc, FetchError):
        return {"error_class": exc.error_class, "http_status": exc.status,
                "attempts": exc.attempts, "error": exc.message[:500]}
    return {"error_class": type(exc).__name__, "http_status": None, "attempts": None,
            "error": f"{type(exc).__name__}: {exc}"[:500]}


class _Terminated(Exception):
    """SIGTERM arrived (Slurm preemption or time limit)."""


def _on_sigterm(signum, frame):  # pragma: no cover - exercised by Slurm, not tests
    raise _Terminated()


def stage_fetch(ctx: Context) -> int:
    """Download what the ledger says is not done, and record why each unit ended.

    Every unit leaves a ledger row with an outcome: done, empty (upstream had
    nothing), absent (404) or failed - with the error class, HTTP status,
    message and number of requests. Only `done` units are skipped next time,
    so failures are retried by simply running again. The listing the run
    planned from is written beside the ledger, which is what lets
    `meteopack catalog <source> --gaps` say what is missing rather than only
    what went wrong.
    """
    import signal

    from meteopack.core.catalog import job_key

    try:
        p = ctx.plan("fetch")
    except Exception as exc:                              # noqa: BLE001 - reported, exit 1
        # A listing that cannot be read is not "nothing to do".
        logger.error("listing %s failed: %s: %s", ctx.cfg.store_id, type(exc).__name__, exc)
        print(f"fetch {ctx.cfg.source}/{ctx.cfg.product}: listing failed - "
              f"{type(exc).__name__}: {exc}")
        return 1
    print(p.summary())
    if ctx.args.dry_run:
        return 0
    if ctx.args.shard == 0:
        _record_listing(ctx, p)
        from meteopack.core.http import remove_stale_parts

        stale = remove_stale_parts(ctx.raw_root)
        if stale:
            print(f"  removed {stale} .part file(s) left by killed downloads")
    if not p.units:
        return 0

    from dataclasses import replace as _replace

    from meteopack.jobs.plan import is_volatile

    workers = ctx.args.workers or min(ctx.source_cfg.rate_limit.max_workers, 32)
    counts = {"done": 0, "empty": 0, "absent": 0, "failed": 0}
    still_filling = downloaded = n_bytes = 0
    failures: dict[str, int] = {}
    started = time.monotonic()

    # One decision about re-downloading, made here and handed to every source as
    # meta["volatile"] (read by sources.common.refresh). The planner and the
    # fetchers used to decide separately and disagreed: DWD's recent/ archives
    # were planned as "always refreshed" and then skipped because the file
    # existed. A unit whose last fetch caught it still filling is refreshed too.
    partial = ctx.catalog.partial("fetch")
    units = []
    for u in p.units:
        changing = is_volatile(u, ctx.cfg)
        refresh = changing or job_key("fetch", u.key) in partial
        units.append((_replace(u, meta={**(u.meta or {}), "volatile": refresh}), changing))

    who = {"source": ctx.cfg.source, "product": ctx.cfg.product}
    ctx.catalog.autoflush = False
    last_flush = time.monotonic()
    # Slurm preempts with SIGTERM and kills ~30 s later. Turning it into an
    # exception lets the ledger be written before that, instead of losing the
    # last half-minute of outcomes. (Only the main thread may set a handler.)
    try:
        previous = signal.signal(signal.SIGTERM, _on_sigterm)
    except ValueError:
        previous = None
    pool = cf.ThreadPoolExecutor(max_workers=workers)
    try:
        futures = {pool.submit(ctx.src.fetch, u, ctx.raw_root): (u, changing)
                   for u, changing in units}
        for future in cf.as_completed(futures):
            unit, changing = futures[future]
            key = job_key("fetch", unit.key)
            try:
                raw = future.result()
            except Exception as exc:                      # noqa: BLE001 - recorded, not swallowed
                # Recorded against its own unit, with the reason; `failed` is
                # not `done`, so the next run tries it again.
                counts["failed"] += 1
                why = _failure_fields(exc)
                failures[why["error_class"]] = failures.get(why["error_class"], 0) + 1
                logger.error("fetch failed for %s: %s", unit.key, why["error"])
                ctx.catalog.mark(key, "failed", outcome="failed", **who, **why)
            else:
                if raw is None:
                    counts["absent"] += 1
                    ctx.catalog.mark(key, "done", outcome="absent", http_status=404,
                                     error="absent upstream (HTTP 404)", **who)
                else:
                    outcome = _outcome_of(raw)
                    counts[outcome] += 1
                    downloaded += 1 if raw.attempts else 0
                    n_bytes += raw.n_bytes
                    ctx.catalog.record_file(raw, url=unit.url, valid_time=unit.valid_time,
                                            **who)
                    # Still changing upstream: fetched, not finished. The next
                    # run fetches it again, and only a fetch after it stops
                    # marks it done.
                    if changing:
                        still_filling += 1
                    ctx.catalog.mark(key, "pending" if changing else "done", outcome=outcome,
                                     attempts=raw.attempts, **who)
            if time.monotonic() - last_flush > _LEDGER_FLUSH_SECONDS:
                ctx.catalog.flush()
                last_flush = time.monotonic()
    except (KeyboardInterrupt, _Terminated):
        print("fetch interrupted; the ledger keeps what finished, re-run to continue")
        pool.shutdown(wait=False, cancel_futures=True)
        ctx.catalog.flush()
        return 130
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        ctx.catalog.flush()
        if previous is not None:
            signal.signal(signal.SIGTERM, previous)

    elapsed = time.monotonic() - started
    note = f", {still_filling} still changing upstream (fetched again next run)" \
        if still_filling else ""
    print(f"fetched {counts['done']}, empty upstream {counts['empty']}, absent upstream "
          f"{counts['absent']}, failed {counts['failed']}{note}")
    print(f"  {downloaded} downloaded, {n_bytes / 1e9:.2f} GB on disk for these units, "
          f"{elapsed / 60:.1f} min")
    if failures:
        print("  failed by error: " + ", ".join(f"{k} {v}" for k, v in sorted(failures.items())))
        print(f"  re-run the same command to retry them; `meteopack catalog {ctx.cfg.source} "
              f"--gaps` says what is missing and why")
    return 1 if counts["failed"] else 0


def _record_listing(ctx: Context, plan) -> None:
    """Write what this run was asked to fetch, for the gap report.

    Shard 0 only: every task of an array lists the same thing. Best effort - a
    report that cannot be written must not stop a download.
    """
    from meteopack.core import coverage

    expected_of = getattr(ctx.src, "expected", None)
    try:
        expected = expected_of(ctx.window, ctx.cfg) if expected_of is not None else []
        frame = coverage.listing_frame(plan.listed_units, expected, ctx.window)
        coverage.write_listing(ctx.catalog.root, ctx.cfg.source, ctx.cfg.product, frame,
                               ctx.catalog.writer_id)
    except Exception as exc:                              # noqa: BLE001 - see docstring
        logger.warning("could not record the listing for the gap report: %s: %s",
                       type(exc).__name__, exc)


def _as_raw(files: list[Path]) -> list[RawFile]:
    """Wrap on-disk paths as RawFile.

    Sources take RawFile, not Path, so that parse and scan see the same object
    whether it came straight from fetch or from a later re-run over the raw
    layer. The checksum is left empty here: it is recorded in the catalog at
    fetch time and re-hashing every file to re-parse would be wasteful.
    """
    from meteopack.sources.base import RawFile

    return [
        RawFile(path=p, sha256="", n_bytes=p.stat().st_size, unit_key=p.name) for p in files
    ]


def _parse_gridded(ctx: Context, files: list[Path]) -> int:
    """Parse each file and say what it held; nothing is concatenated or written.

    A gridded source returns one xarray Dataset per file - a day of 765x700
    frames - and these were handed to polars.concat, which crashed. Stacking
    them would not be useful anyway: `parse` is the dry run, and the useful
    answer is per file.
    """
    failed = 0
    for raw in _as_raw(files):
        try:
            ds = ctx.src.parse(raw, ctx.cfg)
        except Exception as exc:                          # noqa: BLE001
            logger.error("parse failed for %s: %s", Path(raw.path).name, exc)
            failed += 1
            continue
        n = int(ds.sizes.get("time", 0)) if ds is not None else 0
        missing = int(ds["is_missing_frame"].sum()) if n and "is_missing_frame" in ds else 0
        print(f"  {Path(raw.path).name}: {n} frame(s), {missing} missing, "
              f"variables {sorted(str(v) for v in ds.data_vars) if n else []}")
    return 1 if failed else 0


def _parse_all(ctx: Context, files: list[Path]):
    import polars as pl

    frames = []
    for raw in _as_raw(files):
        path = raw.path
        try:
            frame = ctx.src.parse(raw, ctx.cfg)
        except Exception as exc:                          # noqa: BLE001
            logger.error("parse failed for %s: %s", path.name, exc)
            continue
        if frame is not None and len(frame):
            frames.append(frame)
    if not frames:
        return None
    return pl.concat(frames, how="diagonal_relaxed")


def _parse_parts(ctx: Context, files: list[Path]) -> int:
    """Parse a few parts of a source packed in parts, and say what they held.

    A WOW-NL zip is a year - tens of millions of reports - so `parse` takes
    parts (days) instead: `--window` picks them, `--limit` caps how many
    (default 3). Nothing is concatenated or written.
    """
    import io
    from collections import Counter

    import polars as pl

    from meteopack.jobs import packing

    window = ctx.window if ctx.args.window else None
    parts = ctx.src.pack_parts(files, ctx.cfg, window)[: ctx.args.limit or 3]
    if not parts:
        print("no parts to parse")
        return 1
    tasks = [packing.PackTask(source=ctx.cfg.source, product=ctx.cfg.product,
                              raw_path=paths[0], also=tuple(paths[1:]), root=str(ctx.root),
                              configs=ctx.args.configs, part=key, partitions=(label,))
             for key, paths, label in parts]
    results = packing.run(tasks, packing.parse_points, workers=ctx.args.workers)
    failed = 0
    for r in sorted(results, key=lambda r: str(r.get("part"))):
        if r.get("error"):
            failed += 1
            print(f"  {r.get('part') or r.get('path')}: ERROR {r['error']}")
            continue
        frame = pl.read_ipc(io.BytesIO(r["ipc"])) if r.get("ipc") else None
        status = Counter(rec.get("status") for rec in r.get("records") or ())
        print(f"  {r['part']}: {r['rows']:,} rows x {frame.width if frame is not None else 0} "
              f"cols; members {dict(status)}")
    return 1 if failed else 0


def stage_parse(ctx: Context) -> int:
    files = ctx.raw_files()
    print(f"parsing {len(files)} raw files under {ctx.raw_root}")
    if ctx.cfg.store.kind == "gridded":
        return _parse_gridded(ctx, files)
    if callable(getattr(ctx.src, "pack_parts", None)):
        return _parse_parts(ctx, files)
    df = _parse_all(ctx, files)
    if df is None:
        print("nothing parsed")
        return 1
    print(f"{df.height} rows x {df.width} cols: {', '.join(df.columns)}")
    return 0


def _write_station_metadata(ctx: Context) -> int:
    """Persist station positions beside the observations.

    Not part of the Source protocol — a gridded source has no stations — so it
    is an optional capability, like scan(). Without it the points store has
    observations but no coordinates, and anything that needs position (a dense
    station matrix, a map, a radar-gauge comparison) silently drops every
    station instead of failing.
    """
    from meteopack.core.store import points

    stations = getattr(ctx.src, "stations", None)
    if stations is None:
        return 0
    import polars as pl

    from meteopack.core.stations import archived_stations

    # The live catalogue and the archived one are each optional. The archive
    # used to be consulted only after the live lookup had succeeded, so a live
    # failure - MIDAS raising CedaAuthError on an expired token - failed the
    # pack after the store was already written, and an empty live answer never
    # reached the archive at all.
    keep = ["station_id", "lat", "lon", "height", "name"]
    schema = {"station_id": pl.String, "lat": pl.Float64, "lon": pl.Float64,
              "height": pl.Float64, "name": pl.String}
    try:
        table = stations(ctx.cfg)
    except Exception as exc:  # noqa: BLE001 - the archive below may still cover it
        logger.warning("%s: live station catalogue unavailable (%s: %s); falling back to "
                       "the archived one", ctx.cfg.source, type(exc).__name__, exc)
        table = None
    # Catalogues do not always spell a station the way the observations do:
    # MIDAS's archived catalogue says 160 where the data say 00160, which left
    # 133 of 185 stations without coordinates. The source says what canonical is.
    canonical = getattr(ctx.src, "canonical_station_id", None)

    def _canon(frame):
        if canonical is None or not len(frame):
            return frame
        return frame.with_columns(pl.col("station_id").cast(pl.String)
                                  .map_elements(canonical, return_dtype=pl.String))

    if table is not None and len(table):
        table = _canon(table)
        # Sources name the elevation column differently; the store has one name.
        renames = {"height_m": "height", "altitude": "height", "elevation": "height"}
        table = table.rename({k: v for k, v in renames.items() if k in table.columns})
        table = table.with_columns(
            [pl.lit(None, dtype=t).alias(c) for c, t in schema.items() if c not in table.columns])
        table = table.select(keep).unique(subset=["station_id"], keep="first")
    else:
        table = pl.DataFrame(schema=schema)
    # A provider's catalogue lists the stations reporting *now*, so a backfill
    # gets nothing for the ones that have since closed - half of DWD's 1,030
    # and three quarters of DMI's 110. A catalogue archived beside the data was
    # written when they were alive, so it fills the gap. Merged *after* the
    # column names are normalised: the live tables call it `height_m` and the
    # archives call it `height`, and concatenating first made the rename
    # collide.
    archived = archived_stations(ctx.raw_root)
    if archived:
        extra = _canon(pl.DataFrame(archived, schema=schema).select(keep)).unique(
            subset=["station_id"], keep="first")
        new_rows = extra.filter(
            ~pl.col("station_id").is_in(table["station_id"].to_list())
        )
        if len(new_rows):
            logger.info(
                "%s: %d station(s) recovered from an archived catalogue that the "
                "live one no longer lists", ctx.cfg.source, len(new_rows),
            )
            table = pl.concat([table, new_rows], how="diagonal_relaxed")

    # Last resort, and needing no network: positions written in the raw files
    # themselves (MIDAS headers). Fills stations still without coordinates.
    from_raw = getattr(ctx.src, "raw_stations", None)
    if from_raw is not None:
        located = table.filter(pl.col("lat").is_not_null() & pl.col("lon").is_not_null())
        try:
            extra = from_raw(ctx.raw_root)
        except Exception as exc:  # noqa: BLE001 - metadata must not fail a pack
            logger.warning("%s: reading station positions from raw files failed: %s: %s",
                           ctx.cfg.source, type(exc).__name__, exc)
            extra = None
        if extra is not None and len(extra):
            extra = extra.select(keep).filter(
                ~pl.col("station_id").is_in(located["station_id"].to_list()))
            if len(extra):
                logger.info("%s: %d station position(s) taken from the raw file headers",
                            ctx.cfg.source, len(extra))
                table = pl.concat(
                    [located, extra,
                     table.filter(~pl.col("station_id").is_in(
                         pl.concat([located["station_id"], extra["station_id"]]).to_list()))],
                    how="diagonal_relaxed")

    if not len(table):
        logger.warning("source %s: no station metadata, live or archived", ctx.cfg.source)
        return 0
    points.write_stations(table, ctx.cfg.store_id, ctx.root)
    return len(table)


def _pack_tasks(ctx: Context, files: list[Path], timeline=None) -> list:
    """One task per raw file, except that a zarr shard may have only one writer.

    Two processes writing different frames of one shard each read-modify-write
    the whole shard file and replace it atomically, so the slower one's frames
    are simply gone - verified here as 7 to 9 of 12 frames lost, every trial.
    Files that land in the same shard are therefore handed to the same task.

    Almost always this changes nothing: a daily tar is exactly one shard. It
    matters for the two KNMI radar tars that start at 08:00 rather than 08:05,
    whose frames straddle a shard boundary and would otherwise race.
    """
    from meteopack.jobs.packing import PackTask

    def task(first: Path, rest: tuple[Path, ...] = ()) -> PackTask:
        return PackTask(source=ctx.cfg.source, product=ctx.cfg.product,
                        raw_path=str(first), root=str(ctx.root),
                        configs=ctx.args.configs, also=tuple(str(p) for p in rest))

    if timeline is None or ctx.cfg.store.kind != "gridded" \
            or not callable(getattr(ctx.src, "span_of", None)):
        return [task(p) for p in files]

    import numpy as np

    from meteopack.core import grids
    from meteopack.core.store.gridded import _geometry

    grid = grids.get(ctx.cfg.grid_id) if ctx.cfg.grid_id else None
    if grid is None:
        return [task(p) for p in files]
    _, shards = _geometry(ctx.cfg, (len(timeline), grid.ny, grid.nx))
    per_shard = int(shards[0]) if shards else 1
    if per_shard <= 1:
        return [task(p) for p in files]

    n_steps = len(timeline)

    def shards_touched(path: Path) -> set[int] | None:
        """Every shard this file writes into, not just the one it starts in.

        Keying on the first frame alone is not enough: a file whose span
        crosses a boundary writes into two shards, and grouping it with only
        the first leaves it racing the neighbour that owns the second. That is
        exactly the case this exists for - the KNMI tars that start at 08:00
        rather than 08:05.

        The span is converted to axis positions arithmetically rather than
        through `index_of`, because a span's endpoints need not land exactly on
        the cadence and a file that starts half a step early still writes into
        the shard it overlaps.
        """
        span = _span_of(ctx, path)
        if span is None:
            return None              # an undatable file gets its own task
        edges = []
        for when in span:
            ns = int(np.datetime64(when.replace(tzinfo=None), "ns").astype("int64"))
            edges.append((ns - timeline.start_ns) // timeline.step_ns)
        first, last = min(edges), max(edges)
        if last < 0 or first >= n_steps:
            return None                      # nothing of this file is on the axis
        first = max(first, 0)
        last = min(last, n_steps - 1)
        return set(range(first // per_shard, last // per_shard + 1))

    # Files that share any shard must share a task. Union them by shard.
    owner: dict[int, int] = {}          # shard -> group id
    members: dict[int, list[Path]] = {}  # group id -> files
    out: list = []
    for path in files:
        touched = shards_touched(path)
        if not touched:
            out.append(task(path))
            continue
        existing = sorted({owner[sh] for sh in touched if sh in owner})
        if not existing:
            gid = len(members)
            members[gid] = []
        else:
            gid = existing[0]
            for other in existing[1:]:          # two groups now joined
                members[gid].extend(members.pop(other))
                for sh, g in list(owner.items()):
                    if g == other:
                        owner[sh] = gid
        members[gid].append(path)
        for sh in touched:
            owner[sh] = gid

    for group in members.values():
        if group:
            out.append(task(group[0], tuple(group[1:])))
    return out


def _report_trace(results: list[dict], workers: int, *, cli_started: float,
                  pool_started: float) -> None:
    """With METEOPACK_TRACE=<file>: where the pack's wall time went.

    Writes every task's timing to the file and prints the split: set-up before
    the pool, the wait for the first worker to start, how busy the workers were
    once going, and the tail after the last task began. A pack that stops
    speeding up with more workers is losing time in one of these, and the
    numbers say which.
    """
    target = os.environ.get("METEOPACK_TRACE")
    timings = [r["timing"] for r in results if r.get("timing")]
    if not target or not timings:
        return
    import json

    end = max(t["end"] for t in timings)
    first = min(t["start"] for t in timings)
    busy = sum(t["end"] - t["start"] for t in timings)
    parse = sum(t["parse_s"] for t in timings)
    write = sum(t["write_s"] for t in timings)
    span = end - pool_started
    used = len({t["pid"] for t in timings})
    Path(target).write_text(json.dumps({
        "workers": workers, "cli_started": cli_started, "pool_started": pool_started,
        "tasks": timings}, indent=1))
    print(f"  trace: set-up {pool_started - cli_started:.1f}s, first worker started after "
          f"{first - pool_started:.1f}s, pool ran {span:.1f}s; workers busy "
          f"{100 * busy / (workers * span):.0f}% (parse {100 * parse / max(busy, 1e-9):.0f}%, "
          f"write {100 * write / max(busy, 1e-9):.0f}% of busy), {used} worker process(es); "
          f"last task started {end - max(t['start'] for t in timings):.1f}s before the end; "
          f"peak worker RSS {max(t.get('rss_mib', 0) for t in timings):,.0f} MiB")


#: More files than this in one task means grouping has chained files together
#: rather than paired a few boundary-straddlers. Legitimate groups here are 1
#: file, or 2-3 at an archive's irregular start.
_LARGE_GROUP = 8


def _report_grouping(files: list, tasks: list, workers: int) -> None:
    """Say how the files became tasks, and warn if parallelism was lost.

    Grouping by shard is never wrong - files that share a shard must share a
    writer - but it can be catastrophically slow, and nothing else shows it:
    6,668 echotop tars once became 16 tasks, the pack ran at a fraction of its
    worker count, and the only symptom was a long wall time.
    """
    if len(tasks) == len(files):
        return
    largest = max(len(t.paths) for t in tasks)
    print(f"  grouped into {len(tasks)} task(s) so no shard has two writers; "
          f"the largest holds {largest} file(s)")
    if largest > _LARGE_GROUP:
        logger.warning(
            "%d file(s) collapsed into %d task(s) for %d worker(s): most of the pack will "
            "run serially. Files that share a shard must share a task, so this is correct "
            "but slow. It usually means the source's span_of over-reports (each file "
            "appears to touch its neighbour's shard), or extra.timeline_origin does not "
            "line up with where the archive's files begin.",
            len(files), len(tasks), workers,
        )


def _span_of(ctx: Context, path: Path):
    """First and last frame one raw file holds, if the source can tell from its name.

    Every caller goes through here so all of them get the exact span. A source
    that takes the product config gets it: without it, KNMI's exclusive end
    tokens and SEVIRI's padded stamps both over-report by a frame, which is
    harmless for skipping files and ruinous for shard grouping - each file
    appears to write its neighbour's shard and the whole archive chains into a
    handful of serial tasks.
    """
    reader = getattr(ctx.src, "span_of", None)
    if not callable(reader):
        return None
    try:
        return reader(path, cfg=ctx.cfg) if _takes_cfg(reader) else reader(path)
    except Exception:  # noqa: BLE001 - an unreadable name is just one less span
        return None


def _takes_cfg(fn) -> bool:
    import inspect

    try:
        return "cfg" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _timeline_for(ctx: Context):
    """The store's time axis: the window if given, else what the catalog holds.

    A gridded store is created whole and then written by region, so the axis has
    to be known before the first frame lands. Guessing it from the first file
    would silently truncate a backfill, so an unknowable axis is an error.
    """
    from meteopack.core.timeline import Timeline
    from meteopack.jobs.plan import timeline_origin

    origin, step = timeline_origin(ctx.cfg), ctx.cfg.cadence
    start, end = ctx.window.start, ctx.window.end
    if not ctx.args.window:
        files = ctx.catalog.files()
        # Source AND product: product names repeat across sources (dwd and dmi
        # both have stations_10min), and one's valid times would size the
        # other's axis.
        rows = files.filter((files["source"] == ctx.cfg.source)
                            & (files["product"] == ctx.cfg.product)) if len(files) else files
        times = [t for t in rows["valid_time"].to_list() if t is not None] if len(rows) else []
        if not times:
            # An archive copied in by hand has no catalog, and that is the
            # normal case for a backfill. Most sources can read a file's span
            # off its own name, which is enough to size the axis and is a
            # filename parse, not a read. Typing the window instead is how the
            # rtcor axis came to start one step after the archive's first
            # frame, dropping it with no way to see which frame was gone.
            spans = [s for s in (_span_of(ctx, f) for f in ctx.raw_files()) if s]
            if not spans:
                raise SystemExit(
                    f"error: cannot size the time axis for {ctx.cfg.store_id}. Pass --window, "
                    f"or fetch first so the catalog records valid times."
                )
            times = [t for span in spans for t in span]
        start, end = min(times), max(times) + step
        start = start.replace(tzinfo=UTC) if start.tzinfo is None else start
        end = end.replace(tzinfo=UTC) if end.tzinfo is None else end

    # Snap outwards onto the SHARD grid, not the cadence grid. A zarr shard is
    # one file, so an axis that starts mid-shard makes every archive unit
    # straddle two shards; two workers then share one and lose whole frames.
    # Measured on three radar tars: serial wrote 268.2M finite pixels, three
    # workers wrote 222.7M - 17% gone, with no error anywhere.
    from meteopack.jobs.plan import shard_span_seconds

    span = float(shard_span_seconds(ctx.cfg)) if ctx.cfg.store.kind == "gridded" \
        else step.total_seconds()
    offset = (start - origin).total_seconds()
    start = origin + timedelta(seconds=offset - (offset % span))
    offset_end = (end - origin).total_seconds()
    if offset_end % span:
        end = origin + timedelta(seconds=offset_end + (span - offset_end % span))
    return Timeline(start, end, step)


def _size(path: str) -> int:
    """A spill file's size, or 0 if it cannot be read - sizing must not fail a pack."""
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def _files_for_window(ctx, files: list[Path]) -> tuple[list[Path], int]:
    """Drop raw files that cannot contribute to `--window`.

    Without this, `pack --window <one day>` still reads the whole archive:
    packing one day of SEVIRI decompressed and reprojected all 576 scenes on
    disk (25 GB) to write 288 frames, and timed out. The window was honoured
    only when sizing the time axis, never when choosing what to open.

    Only gridded stores are filtered, and that restriction is the point. A
    points partition is *replaced* on write, not appended to (see
    `core.store.points`), so packing a subset of a partition's files would
    silently drop the rest of the partition - a windowed pack of one month
    would delete the other eleven. Gridded frames land in their own slots, so a
    subset writes a subset and touches nothing else.

    A file is kept unless it is known to fall outside the window. The source
    says what span a file covers via the optional `span_of`; whatever it cannot
    date is kept, because dropping data that might be in range is far worse
    than reading a file that turns out not to be.
    """
    if not ctx.args.window or ctx.cfg.store.kind != "gridded":
        return files, 0

    if not callable(getattr(ctx.src, "span_of", None)):
        return files, 0

    start, end = ctx.window.start, ctx.window.end
    keep: list[Path] = []
    undated = 0
    for path in files:
        span = _span_of(ctx, path)       # an undatable file is kept, not fatal
        if span is None:
            undated += 1
            keep.append(path)
            continue
        first, last = span
        if last >= start and first < end:
            keep.append(path)
    return keep, undated


#: Filesystems a spill must never land on, whatever the free space says:
#: these are RAM, so spilling there puts the data back in memory and undoes
#: the reason for spilling at all.
_MEMORY_FS = frozenset({"tmpfs", "ramfs", "devtmpfs"})

#: Filesystems where a spill is worth moving off. The whole point of moving it
#: is that reads of the raw archive and writes of the store already share one
#: network filesystem, and spill traffic queues on the same spindles.
_NETWORK_FS = frozenset({
    "nfs", "nfs4", "cifs", "smb3", "lustre", "beegfs", "gpfs", "ceph",
    "glusterfs", "fuse.sshfs", "afs", "9p",
})

#: Node-local places to try, in order, and only when the data root is on a
#: network filesystem. Probed rather than assumed: /scratch is 1.6 TB free on
#: one node of this cluster and 7.4 GB on another.
_LOCAL_SPILL_DIRS: tuple[Path, ...] = (
    Path("/scratch") / (os.environ.get("USER") or "meteopack"),
    Path("/local") / (os.environ.get("USER") or "meteopack"),
)


def _filesystem_type(path: Path) -> str | None:
    """The fstype `path` lives on, or None when it cannot be determined.

    Read from /proc/mounts by longest matching mount point. Unavailable off
    Linux, and None there means "assume ordinary local disk", which is the
    conservative answer: it keeps the spill beside the data.
    """
    try:
        entries = []
        with open("/proc/mounts") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) >= 3:
                    entries.append((parts[1], parts[2]))
    except OSError:
        return None
    try:
        resolved = str(path.resolve())
    except OSError:
        resolved = str(path)
    best = None
    for mount, fstype in entries:
        if (resolved == mount or resolved.startswith(mount.rstrip("/") + "/")) and (
            best is None or len(mount) > len(best[0])
        ):
            best = (mount, fstype)
    return best[1] if best else None


def _spill_root(root: Path, *, ram_ok: bool = False,
                min_free: int | None = None) -> Path:
    """Where a points pack spills its parsed rows between parse and write.

    The spill is written once, read once and deleted; it never reaches the
    store. So moving it is not a copy - it removes that traffic entirely.

    It is only worth moving when the data root is on a *network* filesystem,
    which is the situation it was written for: on this cluster the raw archive
    and the packed store share one NFS export, so a pack already reads and
    writes the same spindles, and ~17 GB of spill traffic queues behind the
    reads it is competing with. Measured here, local disk is 276/442 MB/s
    against NFS at 91/90.

    On an ordinary machine - data root on local disk - there is nothing to
    gain and something to lose, so the spill stays beside the data exactly as
    before. A tmpfs is never used however much space it reports, because it is
    RAM and spilling to it defeats the purpose.
    """
    override = os.environ.get("METEOPACK_SPILL")
    if override:
        base = Path(override)
        base.mkdir(parents=True, exist_ok=True)
        if _filesystem_type(base) in _MEMORY_FS and not ram_ok:
            logger.warning(
                "$METEOPACK_SPILL points at %s, which is RAM; a large pack will "
                "run the machine out of memory rather than spilling to disk", base,
            )
        return base

    fallback = Path(root) / "_spill"
    fallback.mkdir(parents=True, exist_ok=True)
    if _filesystem_type(root) not in _NETWORK_FS:
        # Ordinary machine: the data root is local, so this *is* local disk.
        return fallback

    for base in _LOCAL_SPILL_DIRS:
        try:
            base.mkdir(parents=True, exist_ok=True)
            fstype = _filesystem_type(base)
            if fstype in _MEMORY_FS or fstype in _NETWORK_FS:
                continue
            free = shutil.disk_usage(base).free
        except OSError:
            continue
        if free >= (min_free if min_free is not None else _MIN_SPILL_BYTES):
            logger.info("spilling to local disk %s (%.0f GB free)", base, free / 1e9)
            return base

    logger.info(
        "no node-local disk with %.0f GB free; spilling to %s, which is on the "
        "same filesystem the pack reads from. $METEOPACK_SPILL overrides this.",
        (min_free if min_free is not None else _MIN_SPILL_BYTES) / 1e9, fallback,
    )
    free = shutil.disk_usage(fallback).free
    if free < _MIN_SPILL_BYTES:
        logger.warning("only %.1f GB free at %s", free / 1e9, fallback)
    return fallback


#: Below this, say so before the pack rather than after it has filled the disk.
_MIN_SPILL_BYTES = 20 * 1024**3
#: The pipelined pack's spill peaks at a few months in flight (2.0 GB measured
#: for the full WOW-NL archive), so it needs far less.
_MIN_PIPELINE_SPILL_BYTES = 6 * 1024**3


def _write_pack_scan_report(ctx: Context, store, results: list[dict]) -> None:
    """Write what the pack saw, so pins can be refreshed without a second pass.

    `scan` opens every frame of every file; so does `pack`. On the 127 GB radar
    archive that is two 36-minute reads of the same bytes to learn the same
    thing, and the filesystem is the bottleneck - measured at 66 MiB/s against
    a 76 MiB/s ceiling, so the second pass is pure cost.

    Emitting it here also removes the reason to skip it. Dropping the separate
    scan from a run left the pinned layout set incomplete and cost 5,508
    frames; a report that falls out of the pack cannot be skipped.
    """
    import json

    layouts: dict[str, dict] = {}
    for result in results:
        for entry in result.get("layouts") or ():
            fp = str(entry.get("fingerprint"))
            seen = layouts.get(fp)
            if seen is None:
                layouts[fp] = dict(entry)
            else:
                seen["count"] += int(entry.get("count", 0))
    if not layouts:
        return

    ordered = sorted(layouts.values(), key=lambda e: -int(e.get("count", 0)))
    payload = {
        "store_id": ctx.cfg.store_id,
        "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "pack",
        "n_layouts": len(ordered),
        "layouts": ordered,
        "known_layouts": sorted(layouts),
        "schema_fingerprint": _product_fingerprint(ctx, sorted(layouts)),
    }
    target = Path(store) / "scan_report.json"
    try:
        target.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        logger.debug("could not write %s: %s", target, exc)
        return
    print(f"  saw {len(ordered)} layout(s); report at {target}")


def _product_fingerprint(ctx: Context, fingerprints: list[str]) -> str:
    """The hash `schema_fingerprint` pins, computed the way the source does.

    Asked of the source rather than recomputed here. The generic helper joins
    names on a NUL and knmi_radar joins layouts on a "|", so computing it
    locally produced a pin that the source's own consistency check rejected -
    in a report whose whole purpose is to be pasted into the config.
    """
    from meteopack.sources.common import fingerprint

    own = getattr(ctx.src, "product_fingerprint", None)
    return own(fingerprints) if callable(own) else fingerprint(fingerprints)


def snap_window(cfg, window: TimeWindow) -> TimeWindow:
    """Widen a window outwards to whole partitions of a time-partitioned store.

    A partition is replaced whole, so a window that cut one in half would
    rewrite it with half its rows. Snapping makes the windowed pack replace
    exactly the partitions it names, complete, and touch nothing else.
    """
    from meteopack.sources.base import TimeWindow

    keys = set(cfg.store.partition_by)

    def floor(t: datetime) -> datetime:
        t = t.astimezone(UTC)
        if "hour" in keys:
            return t.replace(minute=0, second=0, microsecond=0)
        if keys & {"day", "date"}:
            return t.replace(hour=0, minute=0, second=0, microsecond=0)
        if "month" in keys:
            return t.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return t.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    def step(t: datetime) -> datetime:
        if "hour" in keys:
            return t + timedelta(hours=1)
        if keys & {"day", "date"}:
            return t + timedelta(days=1)
        if "month" in keys:
            return t.replace(year=t.year + (t.month == 12), month=t.month % 12 + 1)
        return t.replace(year=t.year + 1)

    start = floor(window.start)
    end = floor(window.end)
    if end < window.end:
        end = step(end)
    return TimeWindow(start, end)


def partitions_in(cfg, window: TimeWindow) -> list[str]:
    """Labels of every partition a (snapped) window covers."""
    from meteopack.core.store.points import partition_label_of

    keys = set(cfg.store.partition_by)
    out, t = [], window.start
    while t < window.end:
        out.append(partition_label_of(cfg, t))
        if "hour" in keys:
            t += timedelta(hours=1)
        elif keys & {"day", "date"}:
            t += timedelta(days=1)
        elif "month" in keys:
            t = t.replace(year=t.year + (t.month == 12), month=t.month % 12 + 1)
        else:
            t = t.replace(year=t.year + 1)
    return out


def _pack_in_parts(ctx: Context, found: list[Path]) -> int:
    """Pack a points store from parts of its raw files, writing as it goes.

    For sources whose raw files are too big to be a unit of work (`pack_parts`:
    WOW-NL's yearly zips, parsed a UTC day at a time across all of them).
    Parsing and partition writing run in one pool (`packing.run_pipeline`), so
    a month is written as soon as its last day is parsed.

    `--window` is honoured here, unlike in the per-file points pack: the
    window is snapped outwards to whole partitions, only those are rebuilt and
    every other partition is left alone, which is what makes "re-pack one
    year" safe. A windowed pack into a store built from a different config is
    refused, because it would leave two configs in one store.
    """
    import tempfile

    from meteopack.core.store import manifest, points
    from meteopack.jobs import packing

    cfg = ctx.cfg
    store = points.store_path(ctx.root, cfg.store_id)
    window = allowed = None
    if ctx.args.window:
        if not points.time_partitioned(cfg):
            print(f"error: {cfg.store_id} is partitioned by {cfg.store.partition_by}, not by "
                  f"time alone, so a window cannot name whole partitions; pack without --window")
            return 2
        window = snap_window(cfg, ctx.window)
        allowed = set(partitions_in(cfg, window))
        if (store / manifest.MANIFEST_NAME).exists() and manifest.is_stale(store, cfg):
            print(f"error: {store} was built from a different config; re-packing only "
                  f"{window.start:%Y-%m-%d}..{window.end:%Y-%m-%d} would leave two configs in "
                  f"one store. Pack without --window (or bump the version).")
            return 2
        print(f"window snapped to whole partitions: {window.start:%Y-%m-%dT%H:%M} .. "
              f"{window.end:%Y-%m-%dT%H:%M} ({len(allowed)} partition(s); all others are left "
              f"untouched)")

    parts = ctx.src.pack_parts(found, cfg, window)
    if ctx.args.limit:
        parts = parts[: ctx.args.limit]
    if not parts:
        print(f"no parts of the {len(found)} raw file(s) fall in the window")
        return 1
    workers = ctx.args.workers or len(os.sched_getaffinity(0))
    labels = sorted({label for _, _, label in parts})
    print(f"packing {len(parts)} part(s) of {len(found)} raw file(s) into {len(labels)} "
          f"partition(s) with {workers} worker(s)")
    if ctx.args.dry_run:
        return 0
    if ctx.args.limit and allowed is None:
        # A capped full pack would rewrite its last partition half-built.
        allowed = set(labels)

    # The pipeline only ever holds the months in flight (a zstd month of 2025
    # is 0.45 GB), so node-local disk with a few GB free is enough - cn114's
    # /scratch had 18 GB, below the two-phase pack's 20 GB floor. A RAM spill
    # ($METEOPACK_SPILL=/dev/shm/...) is counted against the memory budget,
    # but beware: this cluster deletes everything a user owns in /dev/shm
    # ~10 s after any job of theirs on that node ends (docs/runbooks/cluster.md).
    # `extra.spill: memory` (the default here) passes each part's rows back
    # through the pool's pipe as compressed IPC bytes; the parent holds them
    # only until their partition is written. Measured on the full WOW-NL
    # archive (2026-11-06): with the spill on the NFS data root (cn117's
    # /scratch had 2.6 GB free) the pack moved ~160 MB/s of which ~60 was its
    # own spill, flushed at every close and read straight back, and was on
    # course for ~35 minutes.
    inline = str(cfg.extra.get("spill", "memory")) == "memory"
    if inline:
        spill, in_memory = None, True
    else:
        spill = Path(tempfile.mkdtemp(prefix="pack-", dir=_spill_root(
            ctx.root, ram_ok=True, min_free=_MIN_PIPELINE_SPILL_BYTES)))
        in_memory = _filesystem_type(spill) in _MEMORY_FS
    compression = str(cfg.extra.get("spill_compression", "lz4"))
    tasks = [packing.PackTask(source=cfg.source, product=cfg.product, raw_path=paths[0],
                              also=tuple(paths[1:]), root=str(ctx.root),
                              configs=ctx.args.configs, spill=str(spill) if spill else None,
                              spill_inline=inline, part=key,
                              partitions=(label,), spill_compression=compression)
             for key, paths, label in parts]

    def make_partition(label: str, files: tuple[str, ...], n_bytes: int):
        return packing.PartitionTask(label=label, files=files, source=cfg.source,
                                     product=cfg.product, root=str(ctx.root),
                                     configs=ctx.args.configs, n_bytes=n_bytes)

    factor = float(cfg.extra.get("partition_rss_factor", packing.PARTITION_RSS_FACTOR))
    limit = packing.memory_limit_bytes()
    print(f"  spill {spill or 'in memory, through the pool'} "
          f"({'RAM' if in_memory else 'disk'}, {compression}); memory limit "
          f"{(limit or 0) / 2**30:.0f} GiB; partition writers sized at {factor:g}x their rows")
    try:
        result = packing.run_pipeline(tasks, workers=workers, make_partition=make_partition,
                                      allowed=allowed, limit=limit, factor=factor,
                                      spill_in_memory=in_memory,
                                      allow_partial=ctx.args.allow_partial,
                                      write_workers=int(cfg.extra.get("write_workers", 3)),
                                      write_threads=int(cfg.extra.get("write_threads", 8)),
                                      max_spill=float(cfg.extra.get("max_spill_gib", 6))
                                      * 2**30 if inline else None)
    finally:
        if spill is not None:
            shutil.rmtree(spill, ignore_errors=True)

    st = result.stats
    rows = sum(int(r["rows"]) for r in result.written)
    print(f"wrote {len(result.written)} partition(s), {rows:,} rows, in {st['wall_s']:.0f} s "
          f"(parsing done at {st['parse_last_end_s']:.0f} s; parse busy {st['parse_busy_s']:.0f} "
          f"s, write busy {st['write_busy_s']:.0f} s over {workers} workers; up to "
          f"{st['max_writers']} writer(s) at once; spill peak "
          f"{st['max_spill_bytes'] / 2**30:.1f} GiB; worker peak "
          f"{max(st['max_parse_worker_rss_mib'], st['max_write_worker_rss_mib']) / 1024:.1f} GiB"
          + (f"; job peak {st['job_memory_peak_bytes'] / 2**30:.1f} GiB"
             if st.get("job_memory_peak_bytes") else "") + ")")
    for label, why in sorted(result.skipped.items()):
        if why != "no rows":
            print(f"  {label}: {why}")
    errors = list(result.errors)

    summary: dict = {}
    finish = getattr(ctx.src, "finish_pack", None)
    if callable(finish):
        summary = finish(ctx.root, cfg, result=result, window=window, files=found,
                         timing={"cli_started": ctx.started}) or {}
        for line in summary.get("notes") or ():
            print(f"  {line}")
        errors += [{"path": str(store), "error": e} for e in summary.get("errors") or ()]

    if any(why.startswith("write failed") for why in result.skipped.values()):
        stale = store / manifest.MANIFEST_NAME
        if stale.exists():
            stale.unlink()
        print("error: a partition failed to write; the manifest was withdrawn and the store "
              "must be packed again")
        for e in errors[:20]:
            print(f"  {e.get('path')}: {e['error']}")
        return 1

    n_records = int(summary.get("n_records", rows))
    points.write_schema(cfg, ctx.root)
    _publish_store(ctx, store, n_records)
    for e in errors[:20]:
        print(f"  ERROR {e.get('part') or Path(str(e.get('path'))).name}: {e['error']}")
    if len(errors) > 20:
        print(f"  ... and {len(errors) - 20} more")
    return 1 if errors else 0


def _publish_store(ctx: Context, store: Path, n_records: int) -> None:
    """Manifest, README and catalog entry: what every finished pack leaves behind."""
    from meteopack.core import readme
    from meteopack.core.store import manifest

    manifest.write(store, manifest.Manifest.new(
        ctx.cfg.store_id, "packed", ctx.cfg.store.kind, ctx.cfg, n_records=n_records))
    readme.store_readme(ctx.cfg, store, None)
    ctx.catalog.record_store(store_id=ctx.cfg.store_id, layer="packed",
                             kind=ctx.cfg.store.kind, path=str(store), version=ctx.cfg.version,
                             n_records=n_records, config_hash=manifest.config_hash(ctx.cfg))
    ctx.catalog.flush()


def stage_pack(ctx: Context) -> int:
    from meteopack.core import readme
    from meteopack.core.store import manifest
    from meteopack.jobs import packing

    if ctx.args.shards > 1:
        # These were silently ignored, so an array of N tasks ran N identical
        # full packs racing on the same files - for a gridded store, exactly
        # the two-writers-one-shard race that loses whole frames. Splitting a
        # pack safely needs the store created once before the array starts;
        # until that exists, refuse rather than pretend.
        print("error: pack does not support --shard/--shards. Run one pack per product, "
              "with --workers for the cores in the allocation.")
        return 2

    found = ctx.raw_files()
    if not found:
        print(f"no raw files under {ctx.raw_root}; run `meteopack fetch {ctx.cfg.source}` first")
        return 1
    if ctx.cfg.store.kind == "points" and callable(getattr(ctx.src, "pack_parts", None)):
        return _pack_in_parts(ctx, found)
    files, undated = _files_for_window(ctx, found)
    if not files:
        print(f"none of the {len(found)} raw file(s) under {ctx.raw_root} fall in the window")
        return 1
    workers = ctx.args.workers or min(len(files), len(os.sched_getaffinity(0)))
    skipped = len(found) - len(files)
    note = f" ({skipped} outside the window)" if skipped else ""
    if undated:
        note += f" ({undated} could not be dated and are packed anyway)"
    print(f"packing {len(files)} raw file(s) with {workers} worker(s){note}")
    if ctx.args.dry_run:
        return 0

    store = None
    if ctx.cfg.store.kind == "gridded":
        from meteopack.core.store import gridded

        timeline = _timeline_for(ctx)
        store = gridded.store_path(ctx.root, ctx.cfg.store_id)
        if not store.exists():
            gridded.initialise(ctx.cfg, ctx.root, timeline)
            print(f"initialised {len(timeline)} time steps "
                  f"{timeline.start:%Y-%m-%dT%H:%M}..{timeline.end:%Y-%m-%dT%H:%M}")
        else:
            # The axis is fixed at creation, so an operational run that packs
            # one day at a time would otherwise write every day after the first
            # into nothing at all.
            gridded.refresh_attrs(ctx.cfg, ctx.root)
            added = gridded.extend(ctx.cfg, ctx.root, timeline)
            if added:
                print(f"extended the time axis by {added} steps "
                      f"to cover {timeline.end:%Y-%m-%dT%H:%M}")
        tasks = _pack_tasks(ctx, files, timeline)
        _report_grouping(files, tasks, workers)
        pool_started = time.time()
        results = packing.run(tasks, packing.write_gridded, workers=workers)
        _report_trace(results, workers, cli_started=ctx.started, pool_started=pool_started)
        frames = sum(r.get("frames", 0) for r in results)
        outside = sum(r.get("outside", 0) for r in results)
        off_cadence = sum(r.get("off_cadence", 0) for r in results)
        errors = [r for r in results if r.get("error")]
        file_errors = [e for r in results for e in (r.get("file_errors") or ())]
        # Count files, not tasks: a task can hold several files now that
        # everything sharing a shard goes to one worker, and "from 1 file(s)"
        # for three tars is the kind of number you stop trusting. A task that
        # failed outright lost all of its files.
        # Results come back in completion order; a failed task reports its
        # first file, which identifies it.
        files_of = {t.raw_path: len(t.paths) for t in tasks}
        lost_files = sum(files_of.get(r.get("path"), 1) for r in errors)
        packed_files = sum(len(t.paths) for t in tasks) - lost_files - len(file_errors)
        for e in file_errors:
            print(f"  ERROR {Path(e['path']).name}: {e['error']}")
        errors += file_errors
        # A damaged raw file still gave what it held, and the pack still fails:
        # its frames are in, but the day is incomplete until it is re-fetched.
        for note in (d for r in results for d in (r.get("damaged") or ())):
            print(f"  ERROR damaged raw file, readable frames packed: {note}")
            errors.append({"path": note.split(":", 1)[0], "error": note})
        print(f"wrote {frames} frames from {packed_files} file(s)")
        if off_cadence:
            print(f"  {off_cadence} frame(s) were not on the "
                  f"{ctx.cfg.cadence_seconds}s cadence and were dropped")
        _write_pack_scan_report(ctx, store, results)
        gridded.refresh_attrs(ctx.cfg, ctx.root,
                              layout_names=getattr(ctx.src, "layout_names", None))
        records = [rec for r in results for rec in (r.get("records") or ())]
        if records:
            gridded.write_provenance_table(ctx.root, ctx.cfg.store_id, records)
        drift = sorted({d for r in results for d in (r.get("drift") or ())})
        if drift:
            print(f"  ERROR {len(drift)} unpinned schema layout(s); the frames were "
                  f"packed, but re-run `scan` and re-pin before trusting the store:")
            for message in drift[:5]:
                print(f"    {message}")
            errors.append({"path": str(store), "error": f"{len(drift)} unpinned layouts"})
        if outside:
            spans = [r["outside_span"] for r in results if r.get("outside_span")]
            where = ""
            if spans:
                lo = min(s[0] for s in spans)
                hi = max(s[1] for s in spans)
                where = f" The dropped frames run {lo}..{hi}." if lo != hi \
                    else f" The dropped frame is {lo}."
            print(f"  ERROR {outside} frame(s) fall outside the store's time axis "
                  f"and were NOT written. The store covers "
                  f"{timeline.start:%Y-%m-%dT%H:%M}..{timeline.end:%Y-%m-%dT%H:%M}.{where}")
            errors.append({"path": str(store), "error": f"{outside} frames outside the axis"})
        n_records = frames
    else:
        import tempfile

        from meteopack.core.store import points

        # Workers spill to per-partition files and the parent handles one
        # partition at a time. Concatenating every parsed frame in the parent
        # is what killed the DWD backfill: 8,495 files, 74 GB peak RSS and a
        # SIGABRT from the allocator.
        spill = Path(tempfile.mkdtemp(prefix="pack-", dir=_spill_root(ctx.root)))
        try:
            tasks = _pack_tasks(ctx, files)
            tasks = [replace(t, spill=str(spill)) for t in tasks]
            # Points tasks are small and short. Recycling a worker every 32
            # of them costs more in respawns - interpreter startup plus the
            # source import - than it saves; dwd's 8,494 files would pay that
            # 265 times. The leak that motivated recycling is on the gridded
            # path, which keeps the tighter number.
            results = packing.run(tasks, packing.parse_points, workers=workers,
                                  max_tasks_per_child=512)
            errors = [r for r in results if r.get("error")]
            if errors and not ctx.args.allow_partial:
                # A partition is rewritten whole from the files that parsed, so
                # one unreadable file would silently delete its rows from the
                # store - 1,287,539 rows became 1,208,339 in a test with one
                # truncated DWD zip, under a manifest and README that said all
                # was well. Change nothing instead.
                print(f"error: {len(errors)} raw file(s) failed to parse; no partition was "
                      f"rewritten and the store is unchanged:")
                for e in errors[:20]:
                    print(f"  {Path(str(e.get('path'))).name}: {e['error']}")
                if len(errors) > 20:
                    print(f"  ... and {len(errors) - 20} more")
                print("Fix or remove them and re-run, or pass --allow-partial to rewrite the "
                      "partitions without their rows.")
                return 1

            by_partition: dict[str, list[str]] = {}
            for result in results:
                for label, path in (result.get("parts") or {}).items():
                    by_partition.setdefault(label, []).append(path)
            if not by_partition:
                print("nothing parsed")
                return 1

            # Partitions are written concurrently: they are independent, and
            # this is the expensive half of a points pack. dwd coalesces ~60
            # million duplicate (time, station) rows per partition - its five
            # variable groups each contribute a row for the same instant - and
            # doing seventeen of those one after another left fifteen cores
            # idle.
            # Largest first, and only as many at once as memory allows: writing
            # all seventeen dwd partitions together reached 127 GB and the OOM
            # killer took the job (2026-11-05).
            part_tasks = sorted(
                (packing.PartitionTask(
                    label=label, files=tuple(by_partition[label]),
                    source=ctx.cfg.source, product=ctx.cfg.product,
                    root=str(ctx.root), configs=ctx.args.configs,
                    n_bytes=sum(_size(f) for f in by_partition[label]),
                ) for label in by_partition),
                key=lambda t: t.n_bytes, reverse=True,
            )
            part_workers = packing.partition_workers(part_tasks, workers)
            limit = packing.memory_limit_bytes()
            print(f"writing {len(part_tasks)} partition(s) with {part_workers} worker(s) "
                  f"({sum(t.n_bytes for t in part_tasks) / 2**30:.1f} GiB spilled, largest "
                  f"{part_tasks[0].n_bytes / 2**30:.2f} GiB"
                  + (f", memory limit {limit / 2**30:.0f} GiB)" if limit else ")"))
            written = packing.run(part_tasks, packing.write_partition,
                                  workers=part_workers, max_tasks_per_child=0)
            n_records = 0
            paths: list[Path] = []
            failed: list[dict] = []
            for result in sorted(written, key=lambda r: str(r.get("label") or r.get("path"))):
                if result.get("error"):
                    failed.append(result)
                    continue
                n_records += int(result["rows"])
                paths.extend(Path(p) for p in result["paths"])
                merged = int(result.get("merged") or 0)
                note = f", {merged:,} duplicate rows merged" if merged else ""
                rss = result.get("peak_rss")
                rss_note = f", worker peak {rss / 2**30:.1f} GiB" if rss else ""
                print(f"  {result['label']}: {int(result['rows']):,} rows{note}{rss_note}")
        finally:
            shutil.rmtree(spill, ignore_errors=True)

        store = points.store_path(ctx.root, ctx.cfg.store_id)
        if failed:
            # Some partitions were rewritten and some were not, so the store
            # no longer matches any single pack. Writing the manifest anyway
            # made `validate` pass on a store whose partitions were all left
            # over from an earlier, failed run, while the catalog recorded 0
            # records (dwd, 2026-11-05). Withdraw the manifest instead, so the
            # store reads as unfinished until a pack succeeds.
            stale = store / "manifest.json"
            if stale.exists():
                stale.unlink()
            print(f"error: {len(failed)} of {len(part_tasks)} partition(s) failed to write; "
                  f"the manifest was withdrawn and the store must be packed again:")
            for e in failed[:20]:
                print(f"  {e.get('path')}: {e['error']}")
            return 1

        points.write_schema(ctx.cfg, ctx.root)
        print(f"packed {n_records} rows into {len(paths)} partition file(s) at {store}")

    manifest.write(store, manifest.Manifest.new(
        ctx.cfg.store_id, "packed", ctx.cfg.store.kind, ctx.cfg, n_records=n_records))
    # A store gets copied elsewhere without its config; the README is what
    # somebody reads then.
    info = None
    if ctx.cfg.store.kind == "gridded":
        try:
            from meteopack.core.store import gridded as _g

            info = _g.info(ctx.cfg.store_id, ctx.root)
        except Exception:  # noqa: BLE001 - a README must never fail a pack
            info = None
    readme.store_readme(ctx.cfg, store, info)
    ctx.catalog.record_store(store_id=ctx.cfg.store_id, layer="packed",
                             kind=ctx.cfg.store.kind, path=str(store), version=ctx.cfg.version,
                             n_records=n_records, config_hash=manifest.config_hash(ctx.cfg))
    ctx.catalog.flush()
    n_stations = _write_station_metadata(ctx)
    if n_stations:
        print(f"wrote metadata for {n_stations} stations")
    for err in errors[:5]:
        print(f"  ERROR {Path(err['path']).name}: {err['error']}")
    return 1 if errors else 0


def _variant_group(variant: dict) -> str | None:
    """The key a scan variant is pinned under.

    Sources name it differently — dwd calls it `group` (a variable group),
    meteofrance `token` (an archive period) — and a mismatch here compares
    nothing while looking like a clean bill of health.
    """
    for key in ("group", "token"):
        value = variant.get(key)
        if value is not None:
            return str(value)
    return None


def _compare_schema(cfg, report: dict) -> tuple[list[str], str]:
    """Compare a scan report against whatever the product pins.

    Sources pin different things, and comparing the wrong two raises an alarm
    that is not drift — which is worse than no alarm, because it teaches you to
    ignore the real one. Four pins are understood, tried in order:

      extra.schema_fingerprints  {group: fp}   per-group  (dwd, per variable group)
      extra.known_layouts        [fp, ...]     allow-list (knmi_radar, generations)
      extra.known_<thing>        names         catalogue  (knmi_insitu)
      schema_fingerprint         fp            whole file (belgium)

    A catalogue or allow-list passes on a SUBSET: a station reporting fewer
    measurements, or a scan covering one generation, is normal. Drift is only
    something observed that is not pinned.
    """
    extra = cfg.extra or {}
    variants = report.get("variants") or []

    groups = extra.get("schema_fingerprints")
    if isinstance(groups, dict) and groups:
        drifted, scanned = [], set()
        for variant in variants:
            group = _variant_group(variant)
            if group is None:
                continue
            scanned.add(group)
            want, got = groups.get(group), variant.get("fingerprint")
            if want and want != got:
                drifted.append(f"{group}: {got} != pinned {want}")
        missing = sorted(set(groups) - scanned)
        unpinned = sorted(scanned - set(groups))
        note = f"{len(scanned & set(groups))} of {len(groups)} pinned group schemas match"
        if missing:
            note += f" (not seen here: {', '.join(missing)})"
        if unpinned:
            # Seen but not pinned: a group no variable reads (dwd's
            # extreme_temperature). Said, so "6 of 5 match" cannot recur.
            note += f"; unpinned groups also in the raw data: {', '.join(unpinned)}"
        if variants and not scanned:
            # Nothing was compared, which reads like "no drift" and is the one
            # outcome a drift check must never report quietly.
            drifted.append(
                f"{len(variants)} variant(s) carry no group key, so none of the "
                f"{len(groups)} pinned group schemas were checked"
            )
        return drifted, note

    layouts = extra.get("known_layouts")
    if isinstance(layouts, (list, tuple)) and layouts:
        allowed = set(layouts)
        seen = {v.get("fingerprint") for v in variants if v.get("fingerprint")}
        unknown = sorted(seen - allowed)
        return ([f"layout not in the pinned list: {fp}" for fp in unknown],
                f"{len(seen)} observed layout(s) all within the {len(allowed)} pinned")

    for key, pinned in extra.items():
        if not key.startswith("known_") or not isinstance(pinned, (dict, list, tuple, set)):
            continue
        seen_names = report.get(key)
        if not isinstance(seen_names, (dict, list, tuple, set)):
            continue
        unknown = sorted(set(seen_names) - set(pinned))
        return ([f"{key[6:]} not in the pinned catalogue: {n}" for n in unknown],
                f"{len(set(seen_names))} observed against a {len(pinned)}-entry {key[6:]} catalogue")

    pin, got = cfg.schema_fingerprint, report.get("fingerprint")
    if pin and got:
        return ([] if pin == got else [f"file schema: {got} != pinned {pin}"],
                "the pinned file schema matches")
    if not got and report.get("compact_files"):
        return [], (f"{report['compact_files']} compact archive file(s) carry no property "
                    f"schema, so there is nothing to compare against the pin")
    return [], f"nothing pinned to compare against; observed {got}"


def _run_scan(ctx: Context, files: list[Path], scan) -> dict:
    """Scan every raw file, across processes when there are enough to be worth it.

    A scan opens every frame of every file, and it was doing that on one core:
    2,680 radar tars at ~3 s each is 2.2 hours of a sixteen-core node sitting
    at 60% of one core in NFS wait. The work is embarrassingly parallel - one
    file is independent of the next - and parallel readers are exactly what a
    network filesystem serves well.
    """
    workers = ctx.args.workers or len(os.sched_getaffinity(0))
    if workers <= 1 or len(files) < workers * 2:
        return scan(_as_raw(files))

    from meteopack.jobs import packing

    # One task per worker-sized batch: a scan report is a dict that has to be
    # merged, so merging a few dozen beats merging a few thousand.
    batches = max(workers * 4, 1)
    size = -(-len(files) // batches)
    chunks = [files[i:i + size] for i in range(0, len(files), size)]
    tasks = [
        packing.PackTask(source=ctx.cfg.source, product=ctx.cfg.product,
                         raw_path=str(chunk[0]), root=str(ctx.root),
                         configs=ctx.args.configs,
                         also=tuple(str(p) for p in chunk[1:]))
        for chunk in chunks
    ]
    print(f"scanning {len(files)} file(s) in {len(tasks)} batch(es) "
          f"across {workers} worker(s)")
    return _merge_scan_reports(packing.run(tasks, packing.scan_files, workers=workers))


def _merge_scan_reports(reports: list[dict]) -> dict:
    """Combine partial scan reports into the one a serial scan would produce.

    Deliberately generic: sources report different keys, and a merge that knew
    about each of them would be another place for the two to drift apart.
    Counts add, lists and dicts union, and variants combine on their
    fingerprint - which is what identifies a variant in every source here.
    """
    merged: dict = {}
    variants: dict[str, dict] = {}
    errors: list = []
    for report in reports:
        if not isinstance(report, dict):
            continue
        for key, value in report.items():
            if key == "variants":
                for variant in value or []:
                    fp = str(variant.get("fingerprint"))
                    seen = variants.get(fp)
                    if seen is None:
                        variants[fp] = dict(variant)
                    else:
                        seen["count"] = int(seen.get("count", 0)) + int(variant.get("count", 0))
                        examples = list(seen.get("examples") or [])
                        for example in variant.get("examples") or []:
                            if len(examples) < 3 and example not in examples:
                                examples.append(example)
                        seen["examples"] = examples
            elif key == "errors":
                errors.extend(value or [])
            elif key in {"n_files", "n_variants", "compact_files"}:
                merged[key] = merged.get(key, 0) + int(value or 0)
            elif isinstance(value, dict):
                combined = dict(merged.get(key) or {})
                for k, v in value.items():
                    if isinstance(v, list) and isinstance(combined.get(k), list):
                        combined[k] = sorted(set(combined[k]) | set(v))
                    else:
                        combined.setdefault(k, v)
                merged[key] = combined
            elif isinstance(value, list):
                existing = merged.get(key) or []
                merged[key] = sorted({*existing, *value}, key=str)
            else:
                merged.setdefault(key, value)

    ordered = sorted(variants.values(), key=lambda e: (-int(e.get("count", 0)),
                                                       str(e.get("fingerprint"))))
    merged["variants"] = ordered
    merged["n_variants"] = len(ordered)
    merged["errors"] = errors
    return merged


def stage_scan(ctx: Context) -> int:
    files = ctx.raw_files()
    if not files:
        print(f"no raw files under {ctx.raw_root}")
        return 1
    scan = getattr(ctx.src, "scan", None)
    if scan is None:
        print(f"source {ctx.cfg.source!r} does not implement scan()")
        return 2
    report = _run_scan(ctx, files, scan)
    print(f"scanned {report.get('n_files', len(files))} files, "
          f"{report.get('n_variants', '?')} schema variant(s)")
    for variant in report.get("variants", []):
        cols = variant.get("columns")
        n_cols = len(cols) if isinstance(cols, (list, tuple, dict, set)) else cols
        print(f"  {_variant_group(variant) or '-'!s:18s} fp={variant.get('fingerprint')} "
              f"n={variant.get('count')}  cols={n_cols if n_cols is not None else '-'}")
    drifted, checked = _compare_schema(ctx.cfg, report)
    if drifted:
        print("SCHEMA DRIFT:")
        for line in drifted:
            print(f"  {line}")
        print("  inspect the variants above, then re-pin in the product config")
        return 3
    print(checked)

    for err in report.get("errors", [])[:5]:
        print(f"  error: {err}")
    return 0


def stage_validate(ctx: Context) -> int:
    from meteopack.core.validate import validate

    report = validate(ctx.cfg, ctx.root)
    # What only the source knows how to check - for WOW-NL, that every hour of
    # the archive is accounted for and no report is stored twice. Optional,
    # like scan(); it adds checks and stats to the same report.
    extra = getattr(ctx.src, "validate_store", None)
    if callable(extra) and report.checks and report.checks[0].ok:
        try:
            extra(ctx.cfg, ctx.root, report, raw_root=ctx.raw_root)
        except Exception as exc:  # noqa: BLE001 - an unreadable store is the finding
            report.add("source checks ran", False, f"{type(exc).__name__}: {exc}")
    print(report.text())
    return 0 if report.ok else 1


def stage_catalog(ctx: Context) -> int:
    """What is in this data root, and what the pipeline thinks it has done.

    The directory names say which layer a store belongs to, not what is in it,
    so the stores are listed with the description from their own manifest.
    """
    import json

    cat = ctx.catalog
    files, stores, jobs = cat.files(), cat.stores(), cat.jobs()
    print(f"data root {ctx.root}")
    print("  (see README.md there for what each directory holds)")

    print("\nstores")
    rows = []
    from meteopack.core.config import layer_path

    for layer in ("packed", "training"):
        base = layer_path(ctx.root, layer)
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("manifest.json")):
            try:
                man = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            rows.append((layer, man.get("store_id", path.parent.name),
                         man.get("kind", "?"), man.get("n_records", 0),
                         man.get("description") or ""))
    if not rows:
        print("  (none yet - run `meteopack pack <source>`)")
    for layer, store_id, kind, n, desc in rows:
        print(f"  {layer:9s} {store_id:34s} {kind:8s} {n:>12,} records")
        if desc:
            print(f"            {desc}")

    print(f"\nbookkeeping at {cat.root}")
    print(f"  files : {len(files):>6}  raw files fetched")
    print(f"  stores: {len(stores):>6}  store builds recorded")
    print(f"  jobs  : {len(jobs):>6}  units of work")
    if len(jobs):
        # `status` says whether a re-run redoes a unit; `outcome` says why it
        # ended where it did (empty upstream, absent, failed). Rows written
        # before outcomes were recorded show "-".
        counts = (jobs.with_columns(jobs["outcome"].fill_null("-"))
                  .group_by("stage", "status", "outcome").len().sort("stage", "status", "outcome"))
        for row in counts.iter_rows(named=True):
            print(f"    {row['stage']:10s} {row['status']:8s} {row['outcome']:8s} {row['len']}")
        if not getattr(ctx.args, "gaps", False):
            print(f"  (`meteopack catalog {ctx.cfg.source} --gaps` for what is missing and why)")
    if getattr(ctx.args, "gaps", False):
        from meteopack.core import coverage

        text, _ = coverage.report(
            cat, ctx.cfg.source, ctx.cfg.product,
            window=ctx.window if ctx.args.window else None,
            out=Path(ctx.args.out) if ctx.args.out else None)
        print()
        print(text)
    return 0


def _derive_timeline(cfg, root: Path, window):
    """The time axis a derived dataset is enumerated over.

    `--window` if given, otherwise what the inputs actually cover: a derived
    dataset over a window its inputs do not hold is all-missing samples, and a
    window typed by hand is the easiest way to get one. Snapped inwards to the
    derive cadence so the axis is on grid.
    """
    import json
    from datetime import timedelta

    from meteopack.core.config import layer_path
    from meteopack.core.timeline import Timeline

    cadence = timedelta(seconds=cfg.cadence_seconds)
    if window is not None:
        return Timeline(start=window.start, end=window.end, cadence=cadence)

    spans = []
    for store_id in cfg.inputs:
        base = layer_path(root, "packed", store_id)
        man = base / "manifest.json"
        if not man.exists():
            continue
        try:
            kind = json.loads(man.read_text()).get("kind")
        except (OSError, ValueError):
            continue
        if kind == "gridded":
            from meteopack.core.store import gridded

            # Through the store module, which knows where the group lives:
            # opening `base` directly broke when the group moved into
            # dataset.zarr, and `derive` could open no store written since.
            meta = gridded.info(store_id, root)
            spans.append((datetime.fromisoformat(meta["time_start"]),
                          datetime.fromisoformat(meta["time_end"])))
        elif kind == "points":
            import duckdb

            glob = str(base / "observations" / "**" / "*.parquet")
            rows = duckdb.sql(
                f"select min(time), max(time) from read_parquet('{glob}')"
            ).fetchall()
            if rows and rows[0][0] is not None:
                lo, hi = rows[0]
                spans.append((lo.replace(tzinfo=UTC), hi.replace(tzinfo=UTC) + cadence))

    if not spans:
        raise SystemExit(
            f"cannot size the time axis for {cfg.name}: none of its inputs "
            f"({', '.join(cfg.inputs)}) is packed. Pack them, or pass --window."
        )
    start = max(s for s, _ in spans)
    end = min(e for _, e in spans)
    # Snap inwards so both ends sit on the cadence grid.
    step = int(cadence.total_seconds())
    start += timedelta(seconds=(-int(start.timestamp())) % step)
    end -= timedelta(seconds=int(end.timestamp()) % step)
    if end <= start:
        raise SystemExit(
            f"the inputs of {cfg.name} do not overlap in time: "
            + "; ".join(f"{a:%Y-%m-%d %H:%M}..{b:%Y-%m-%d %H:%M}" for a, b in spans)
        )
    return Timeline(start=start, end=end, cadence=cadence)


def _derive_kind(cfg, root: Path) -> str:
    """Which builder a derive config wants: gridded patches, or a station matrix.

    Taken from the manifest of the first input store rather than guessed from
    the name, so a config cannot quietly ask for the wrong one. `kind:` in the
    config overrides it.
    """
    import json

    from meteopack.core.config import layer_path

    if getattr(cfg, "kind", None):
        return str(cfg.kind)
    for store_id in cfg.inputs:
        path = layer_path(root, "packed", store_id) / "manifest.json"
        if not path.exists():
            continue
        try:
            kind = json.loads(path.read_text()).get("kind")
        except (OSError, ValueError):
            continue
        if kind == "points":
            return "dense_stations"
        if kind == "gridded":
            return "bricks"
    raise SystemExit(
        f"cannot tell what to build for {cfg.name}: none of its inputs "
        f"({', '.join(cfg.inputs)}) is a packed store under {root / 'canonical'}. "
        f"Pack the input first, or set `kind:` in the derive config."
    )


def stage_derive(ctx_args: argparse.Namespace) -> int:
    from meteopack.core import readme
    from meteopack.core.config import DeriveConfig, data_root
    from meteopack.core.store import manifest as manifest_mod
    from meteopack.jobs.plan import config_dir, load_yaml

    path = config_dir(ctx_args.configs) / "derive" / f"{ctx_args.target}.yaml"
    if not path.exists():
        print(f"no derive config: {path}")
        return 1

    raw = load_yaml(path)
    if raw.get("kind") == "regularise":
        return _derive_regularise(ctx_args, raw)
    if ctx_args.set:
        print("error: --set is only understood by `kind: regularise` derive configs")
        return 2
    cfg = DeriveConfig(**raw)
    root = data_root(ctx_args.data_root, default=str(Path.cwd() / "data"))
    readme.write(root)
    kind = _derive_kind(cfg, root)
    try:
        if kind == "dense_stations":
            from meteopack.derive import dense_stations as builder
        else:
            from meteopack.derive import bricks as builder
    except ImportError as exc:
        print(f"derive is not available: {exc}")
        return 2

    from meteopack.derive.base import DeriveSpec

    timeline = _derive_timeline(cfg, root, parse_window(ctx_args.window)
                                if ctx_args.window else None)
    spec = DeriveSpec(cfg, timeline)
    print(f"building {cfg.name} ({kind}) from {', '.join(cfg.inputs)} over "
          f"{timeline.start:%Y-%m-%dT%H:%M}..{timeline.end:%Y-%m-%dT%H:%M} "
          f"({len(timeline)} steps)")
    if kind == "dense_stations":
        out = builder.build(spec, root)
    else:
        out = builder.build(spec, root, task=ctx_args.shard, n_tasks=ctx_args.shards)
    # A brick store without a split index is a training set with no train/val/
    # test. The config declares the splits, nothing was writing them, and the
    # omission is invisible until a training run reads the store and finds no
    # such column.
    from meteopack.derive import splits as splits_mod

    index = splits_mod.write(spec, root)
    counts = splits_mod.read(spec, root)["split"].value_counts().sort("split")
    summary = ", ".join(f"{r['split']}={r['count']}" for r in counts.iter_rows(named=True))
    print(f"split index -> {index} ({summary})")

    man = out / manifest_mod.MANIFEST_NAME
    note = ""
    if man.exists():
        import json

        try:
            data = json.loads(man.read_text())
            note = f" ({data.get('n_records', 0):,} records)"
        except (OSError, ValueError):
            note = ""
    print(f"derived {cfg.name} -> {out}{note}")
    return 0


def _derive_regularise(ctx_args: argparse.Namespace, raw: dict) -> int:
    """Irregular points store -> regular grid (meteopack.derive.regularise).

    Every parameter is in the config and any can be overridden for one run
    with `--set key=value` (dotted for nested keys; YAML values), e.g.
    `--set step_seconds=300 --set label=start --set rain.rule=naive`. A run
    whose parameters differ from the store already on disk refuses to mix
    into it: give it a new `--set name=...`, or `--refresh` to replace it.
    """
    from meteopack.core.config import data_root
    from meteopack.derive import regularise

    try:
        cfg = regularise.RegulariseConfig(**regularise.apply_overrides(raw, ctx_args.set))
    except (ValueError, TypeError) as exc:
        print(f"error: {exc}")
        return 2
    root = data_root(ctx_args.data_root, default=str(Path.cwd() / "data"))
    window = None
    if ctx_args.window:
        w = parse_window(ctx_args.window)
        window = (w.start, w.end)
    print(f"regularising {cfg.input} -> training/{cfg.store_id}: step {cfg.step_seconds} s, "
          f"label {cfg.label}, origin {cfg.origin.isoformat()}, "
          f"{len(cfg.variables)} variable(s)"
          + (f", rain {cfg.rain.rule}" if cfg.rain else "")
          + (f", window {window[0]:%Y-%m-%d}..{window[1]:%Y-%m-%d}" if window else ""))
    meta = regularise.build(cfg, root, window=window, workers=ctx_args.workers,
                            threads=cfg.threads, refresh=ctx_args.refresh)
    run = meta["run"]
    print(f"wrote {meta['rows']:,} rows ({meta['months_built']} month(s)) in "
          f"{run['wall_s']:.0f} s: {run['input_rows']:,} reports read, {run['workers']} "
          f"worker(s) x {run['polars_threads']} thread(s); read {run['read_s']:.0f} s, "
          f"compute {run['compute_s']:.0f} s, write {run['write_s']:.0f} s summed over "
          f"workers; worker peak {run['max_worker_rss_mib'] / 1024:.1f} GiB")
    if meta["non_causal_columns"]:
        print(f"  non-causal (use reports after T): {', '.join(meta['non_causal_columns'])}")
    for k, v in meta["stage_counts"].items():
        print(f"  {k}: {v:,}")
    return 0


def stage_credentials(ctx_args: argparse.Namespace) -> int:
    """Report, per source, what credential it needs and whether it is usable.

    Written as a command rather than only a document because the document goes
    stale and this cannot: it reads `credentials_env` from the source configs
    and then actually looks at what is on this machine. "Which key do I need
    again" and "is it still valid" are the same question in practice, since the
    one credential here expires every three days.
    """
    import os

    from meteopack.jobs.plan import config_dir, load_source_config
    from meteopack.sources import registry

    # This reads every source's config, including ones whose archive variables
    # this user never sets (KNMI_INSITU_ARCHIVE, ...); warning about each was
    # noise in a command that is about keys, not archives.
    logging.getLogger("meteopack.jobs.plan").setLevel(logging.ERROR)

    names = [p.stem for p in sorted((config_dir(ctx_args.configs) / "sources").glob("*.yaml"))]
    if getattr(ctx_args, "target", None):
        names = [n for n in names if n == ctx_args.target] or names

    needs, open_sources, problems = [], [], 0
    for name in names:
        try:
            cfg = load_source_config(name, configs=ctx_args.configs)
        except (FileNotFoundError, ValueError) as exc:
            print(f"  {name:18s} config unreadable: {exc}")
            continue
        if not cfg.credentials_env:
            open_sources.append(name)
            continue
        needs.append((name, cfg.credentials_env))

    if open_sources:
        print(f"Keyless, nothing to set up ({len(open_sources)}):")
        print(f"  {', '.join(open_sources)}")

    if not needs:
        print("\nNo source in this configuration needs a credential.")
        return 0

    print("\nNeeds a credential:")
    for name, variables in needs:
        print(f"  {name}")
        for variable in variables:
            value = os.environ.get(variable)
            print(f"    ${variable:20s} {'set to ' + value if value else 'not set (a default is used)'}")
        state, detail = _credential_state(name, registry)
        problems += state != "ok"
        print(f"    {'OK  ' if state == 'ok' else 'MISS'} {detail}")

    print("\nHow to obtain each one: docs/runbooks/credentials.md")
    return 1 if problems else 0


def _credential_state(name: str, registry) -> tuple[str, str]:
    """Whether `name`'s credential is present, without spending a request on it.

    Deliberately checks only what is cheap and local. Whether CEDA will still
    accept the token cannot be answered without asking CEDA, and a command that
    reports status should not start downloading.
    """
    if name != "midas":
        return "ok", "no local check for this source"
    try:
        from meteopack.sources import midas
    except ImportError as exc:  # pragma: no cover - optional dependency
        return "miss", f"cannot import the midas source: {exc}"
    path = midas.token_file()
    try:
        token = midas.read_token(path)
    except midas.CedaAuthError as exc:
        return "miss", str(exc).replace("\n", " ")
    age = ""
    try:
        import time

        hours = (time.time() - path.stat().st_mtime) / 3600
        age = f", written {hours:.0f}h ago"
        if hours > 72:
            return "miss", (
                f"the token in {path} is {hours:.0f}h old and CEDA tokens last about 72h; "
                f"run scripts/refresh-ceda-token.sh"
            )
    except OSError:
        pass
    return "ok", f"a token is present in {path} ({len(token)} chars{age})"


# --------------------------------------------------------------------- main

_STAGE_FUNCS = {
    "list": stage_list, "fetch": stage_fetch, "parse": stage_parse, "pack": stage_pack,
    "scan": stage_scan, "validate": stage_validate, "catalog": stage_catalog,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if args.stage == "credentials":
        return stage_credentials(args)
    if args.stage == "derive":
        return stage_derive(args)
    if not args.target:
        print(f"error: `meteopack {args.stage}` needs a source name", file=sys.stderr)
        return 2
    try:
        ctx = Context(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _STAGE_FUNCS[args.stage](ctx)


if __name__ == "__main__":
    raise SystemExit(main())
