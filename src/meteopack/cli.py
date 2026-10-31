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

STAGES = ("scan", "list", "fetch", "parse", "pack", "derive", "validate", "catalog")


# --------------------------------------------------------------------- args

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="meteopack", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=STAGES)
    p.add_argument("target", help="source name (scan/list/fetch/parse/pack/validate/catalog) "
                                  "or dataset name (derive)")
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

    @property
    def src(self):
        if self._src is None:
            import requests

            from meteopack.core.http import RateLimiter
            from meteopack.sources import registry

            self._src = registry.build(
                self.args.target,
                self.source_cfg,
                session=requests.Session(),
                limiter=RateLimiter.from_config(self.source_cfg.rate_limit),
            )
        return self._src

    @property
    def raw_root(self) -> Path:
        return self.root / "raw" / self.cfg.source / self.cfg.product

    def raw_files(self) -> list[Path]:
        return sorted(p for p in self.raw_root.rglob("*") if p.is_file() and p.suffix != ".part")

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


def _pack_tasks(ctx: Context, files: list[Path]) -> list:
    from meteopack.jobs.packing import PackTask

    return [PackTask(source=ctx.cfg.source, product=ctx.cfg.product, raw_path=str(p),
                     root=str(ctx.root), configs=ctx.args.configs) for p in files]


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


def stage_pack(ctx: Context) -> int:
    from meteopack.core.store import manifest
    from meteopack.jobs import packing

    files = ctx.raw_files()
    if not files:
        print(f"no raw files under {ctx.raw_root}; run `meteopack fetch {ctx.cfg.source}` first")
        return 1
    workers = ctx.args.workers or min(len(files), len(os.sched_getaffinity(0)))
    print(f"packing {len(files)} raw file(s) with {workers} worker(s)")
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
        results = packing.run(_pack_tasks(ctx, files), packing.write_gridded, workers=workers)
        frames = sum(r.get("frames", 0) for r in results)
        errors = [r for r in results if r.get("error")]
        print(f"wrote {frames} frames from {len(results) - len(errors)} file(s)")
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
        store = points.store_path(ctx.root, ctx.cfg.store_id)
        n_records = df.height
        print(f"packed {df.height} rows into {len(paths)} partition file(s) at {store}")

    manifest.write(store, manifest.Manifest.new(
        ctx.cfg.store_id, "canonical", ctx.cfg.store.kind, ctx.cfg, n_records=n_records))
    ctx.catalog.record_store(store_id=ctx.cfg.store_id, layer="canonical",
                             kind=ctx.cfg.store.kind, path=str(store), version=ctx.cfg.version,
                             n_records=n_records, config_hash=manifest.config_hash(ctx.cfg))
    ctx.catalog.flush()
    n_stations = _write_station_metadata(ctx)
    if n_stations:
        print(f"wrote metadata for {n_stations} stations")
    for err in errors[:5]:
        print(f"  ERROR {Path(err['path']).name}: {err['error']}")
    return 1 if errors else 0


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
            group = variant.get("group")
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
        print(f"  {variant.get('group', '-')!s:18s} fp={variant.get('fingerprint')} "
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
    cat = ctx.catalog
    files, stores, jobs = cat.files(), cat.stores(), cat.jobs()
    print(f"catalog at {cat.root}")
    print(f"  files : {len(files)}")
    print(f"  stores: {len(stores)}")
    print(f"  jobs  : {len(jobs)}")
    if len(jobs):
        counts = jobs.group_by("stage", "status").len().sort("stage")
        for row in counts.iter_rows(named=True):
            print(f"    {row['stage']:10s} {row['status']:8s} {row['len']}")
    return 0


def stage_derive(ctx_args: argparse.Namespace) -> int:
    from meteopack.core.config import DeriveConfig, data_root
    from meteopack.jobs.plan import config_dir

    path = config_dir(ctx_args.configs) / "derive" / f"{ctx_args.target}.yaml"
    if not path.exists():
        print(f"no derive config: {path}")
        return 1
    import yaml

    cfg = DeriveConfig(**yaml.safe_load(path.read_text()))
    root = data_root(ctx_args.data_root, default=str(Path.cwd() / "data"))
    try:
        from meteopack.derive import bricks
    except ImportError as exc:
        print(f"derive is not available: {exc}")
        return 2
    out = bricks.build(cfg, root, shard=ctx_args.shard, shards=ctx_args.shards)
    print(f"derived {cfg.name} -> {out}")
    return 0


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
    if args.stage == "derive":
        return stage_derive(args)
    try:
        ctx = Context(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return _STAGE_FUNCS[args.stage](ctx)


if __name__ == "__main__":
    raise SystemExit(main())
