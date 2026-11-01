"""meteopack {scan,list,fetch,parse,pack,derive,validate,catalog} <target>

Every stage is re-runnable and shardable. The ledger decides what is left to do,
so running the same command twice is cheap and a preempted Slurm task costs at
most its in-flight unit.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from meteopack.sources.base import RawFile, TimeWindow

logger = logging.getLogger("meteopack")

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
    p.add_argument("--window", help="ISO interval START/END, e.g. 2012-01-01/2013-01-01 "
                                    "(default: the last 24 hours)")
    p.add_argument("--data-root", help="overrides $METEOPACK_DATA_ROOT, which overrides the config")
    p.add_argument("--configs", help="config directory (default: <repo>/configs)")
    p.add_argument("--workers", type=int, default=None, help="default: all available cores")
    p.add_argument("--shard", type=int, default=0, help="this task's index in a Slurm array")
    p.add_argument("--shards", type=int, default=1, help="total tasks in the Slurm array")
    p.add_argument("--refresh", action="store_true", help="ignore the ledger and redo everything")
    p.add_argument("--limit", type=int, default=None, help="stop after N units (for smoke runs)")
    p.add_argument("--dry-run", action="store_true", help="plan only; touch nothing")
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
        text = text.strip()
        if len(text) == 4:                      # a bare year
            year = int(text) + (1 if upper else 0)
            return datetime(year, 1, 1, tzinfo=UTC)
        value = datetime.fromisoformat(text)
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
        self.root = data_root(args.data_root, default=str(Path.cwd() / "data"))
        self.cfg = planning.load_product_config(args.target, args.product, configs=args.configs)
        self.source_cfg = planning.load_source_config(args.target, configs=args.configs)
        self.catalog = Catalog(self.root)
        self.window = parse_window(args.window)
        self._src = None
        # The data root outlives the checkout that made it, so it carries its
        # own explanation of what `raw/`, `canonical/`, `derived/` and
        # `catalog/` hold. Best effort, and never read back.
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
                limiter=RateLimiter.from_config(self.source_cfg.rate_limit),
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


def stage_fetch(ctx: Context) -> int:
    from meteopack.core.catalog import job_key

    p = ctx.plan("fetch")
    print(p.summary())
    if ctx.args.dry_run or not p.units:
        return 0

    workers = ctx.args.workers or min(ctx.source_cfg.rate_limit.max_workers, 32)
    ok = skipped = failed = 0

    def one(unit):
        return unit, ctx.src.fetch(unit, ctx.raw_root)

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for future in cf.as_completed([pool.submit(one, u) for u in p.units]):
            unit, raw = None, None
            try:
                unit, raw = future.result()
            except Exception as exc:                      # noqa: BLE001 - reported, not swallowed
                failed += 1
                logger.error("fetch failed: %s", exc)
                continue
            if raw is None:
                skipped += 1
                ctx.catalog.mark(job_key("fetch", unit.key), "done", error="absent upstream")
                continue
            ok += 1
            ctx.catalog.record_file(raw, source=ctx.cfg.source, product=ctx.cfg.product,
                                    url=unit.url, valid_time=unit.valid_time)
            ctx.catalog.mark(job_key("fetch", unit.key), "done")
    ctx.catalog.flush()
    print(f"fetched {ok}, absent upstream {skipped}, failed {failed}")
    return 1 if failed else 0


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


def stage_parse(ctx: Context) -> int:
    files = ctx.raw_files()
    print(f"parsing {len(files)} raw files under {ctx.raw_root}")
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
    table = stations(ctx.cfg)
    if table is None or not len(table):
        logger.warning("source %s returned no station metadata", ctx.cfg.source)
        return 0

    import polars as pl

    # Sources name the elevation column differently; the store has one name.
    renames = {"height_m": "height", "altitude": "height", "elevation": "height"}
    table = table.rename({k: v for k, v in renames.items() if k in table.columns})
    keep = [c for c in ("station_id", "lat", "lon", "height", "name") if c in table.columns]
    table = table.select(keep).unique(subset=["station_id"], keep="first")
    if "height" not in table.columns:
        table = table.with_columns(pl.lit(None, dtype=pl.Float64).alias("height"))
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

    span_of = getattr(ctx.src, "span_of", None)
    if timeline is None or ctx.cfg.store.kind != "gridded" or span_of is None:
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
        try:
            span = span_of(path)
        except Exception:  # noqa: BLE001 - an undatable file gets its own task
            return None
        if span is None:
            return None
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
        rows = files.filter(files["product"] == ctx.cfg.product) if len(files) else files
        times = [t for t in rows["valid_time"].to_list() if t is not None] if len(rows) else []
        if not times:
            raise SystemExit(
                f"error: cannot size the time axis for {ctx.cfg.store_id}. Pass --window, or "
                f"fetch first so the catalog records valid times."
            )
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


def _coalesce_observations(df):
    """Merge rows that describe the same (time, station) into one.

    Sources differ in what a unit contains. DWD ships a station-year of every
    variable in its group; DMI ships one PARAMETER per station-window, so six
    files for three stations arrive as six rows per timestamp, each carrying a
    different column. Concatenating those stacks them instead of merging, and
    the store then holds a row per parameter rather than a row per observation
    — 711,570 duplicate (time, station) pairs on the first real DMI pack.

    Only pays for itself when duplicates exist, so the common case is a cheap
    check. Where two files genuinely disagree on a value the first non-null
    wins, which is arbitrary but deterministic; the count is logged so a source
    producing real conflicts is visible rather than silently reconciled.
    """
    import polars as pl

    keys = [c for c in ("time", "station_id") if c in df.columns]
    if len(keys) < 2 or df.is_empty():
        return df
    n_dup = int(df.select(keys).is_duplicated().sum())
    if not n_dup:
        return df

    values = [c for c in df.columns if c not in keys]
    merged = (
        df.group_by(keys)
        .agg([pl.col(c).drop_nulls().first().alias(c) for c in values])
        .sort(keys)
    )
    logger.info(
        "coalesced %d duplicate (time, station) rows into %d; %d remain",
        n_dup, merged.height, df.height - merged.height,
    )
    return merged


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

    span_of = getattr(ctx.src, "span_of", None)
    if span_of is None:
        return files, 0

    start, end = ctx.window.start, ctx.window.end
    keep: list[Path] = []
    undated = 0
    for path in files:
        try:
            span = span_of(path)
        except Exception:  # noqa: BLE001 - an undatable file is kept, not fatal
            span = None
        if span is None:
            undated += 1
            keep.append(path)
            continue
        first, last = span
        if last >= start and first < end:
            keep.append(path)
    return keep, undated


def stage_pack(ctx: Context) -> int:
    from meteopack.core import readme
    from meteopack.core.store import manifest
    from meteopack.jobs import packing

    found = ctx.raw_files()
    if not found:
        print(f"no raw files under {ctx.raw_root}; run `meteopack fetch {ctx.cfg.source}` first")
        return 1
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
        results = packing.run(_pack_tasks(ctx, files, timeline),
                              packing.write_gridded, workers=workers)
        frames = sum(r.get("frames", 0) for r in results)
        outside = sum(r.get("outside", 0) for r in results)
        off_cadence = sum(r.get("off_cadence", 0) for r in results)
        errors = [r for r in results if r.get("error")]
        print(f"wrote {frames} frames from {len(results) - len(errors)} file(s)")
        if off_cadence:
            print(f"  {off_cadence} frame(s) were not on the "
                  f"{ctx.cfg.cadence_seconds}s cadence and were dropped")
        if outside:
            print(f"  ERROR {outside} frame(s) fall outside the store's time axis "
                  f"and were NOT written. The store covers "
                  f"{timeline.start:%Y-%m-%dT%H:%M}..{timeline.end:%Y-%m-%dT%H:%M}.")
            errors.append({"path": str(store), "error": f"{outside} frames outside the axis"})
        n_records = frames
    else:
        import io

        import polars as pl

        from meteopack.core.store import points

        results = packing.run(_pack_tasks(ctx, files), packing.parse_points, workers=workers)
        errors = [r for r in results if r.get("error")]
        frames = [pl.read_ipc(io.BytesIO(r["ipc"])) for r in results if r.get("ipc")]
        if not frames:
            print("nothing parsed")
            return 1
        df = pl.concat(frames, how="diagonal_relaxed")
        df = _coalesce_observations(df)
        paths = points.write(df, ctx.cfg, ctx.root)
        points.write_schema(ctx.cfg, ctx.root)
        store = points.store_path(ctx.root, ctx.cfg.store_id)
        n_records = df.height
        print(f"packed {df.height} rows into {len(paths)} partition file(s) at {store}")

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
      extra.known_<thing>        names         catalogue  (esoh, knmi_insitu)
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
        note = f"{len(scanned)} of {len(groups)} pinned group schemas match"
        if missing:
            note += f" (not seen here: {', '.join(missing)})"
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
    return [], f"nothing pinned to compare against; observed {got}"


def stage_scan(ctx: Context) -> int:
    files = ctx.raw_files()
    if not files:
        print(f"no raw files under {ctx.raw_root}")
        return 1
    scan = getattr(ctx.src, "scan", None)
    if scan is None:
        print(f"source {ctx.cfg.source!r} does not implement scan()")
        return 2
    report = scan(_as_raw(files))
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
        counts = jobs.group_by("stage", "status").len().sort("stage")
        for row in counts.iter_rows(named=True):
            print(f"    {row['stage']:10s} {row['status']:8s} {row['len']}")
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
            import zarr

            attrs = zarr.open_group(base, mode="r").attrs
            spans.append((datetime.fromisoformat(str(attrs["time_start"])),
                          datetime.fromisoformat(str(attrs["time_end"]))))
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

    cfg = DeriveConfig(**load_yaml(path))
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
