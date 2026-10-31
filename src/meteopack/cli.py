"""meteopack {scan,list,fetch,parse,pack,derive,validate,catalog} <target>

Every stage is re-runnable and shardable. The ledger decides what is left to do,
so running the same command twice is cheap and a preempted Slurm task costs at
most its in-flight unit.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
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


def stage_pack(ctx: Context) -> int:
    from meteopack.core.store import manifest, points

    files = ctx.raw_files()
    if not files:
        print(f"no raw files under {ctx.raw_root}; run `meteopack fetch {ctx.cfg.source}` first")
        return 1
    df = _parse_all(ctx, files)
    if df is None:
        print("nothing parsed")
        return 1
    if ctx.args.dry_run:
        print(f"would pack {df.height} rows into {ctx.cfg.store_id}")
        return 0

    if ctx.cfg.store.kind != "points":
        print(f"pack for store kind {ctx.cfg.store.kind!r} is not wired yet")
        return 2
    paths = points.write(df, ctx.cfg, ctx.root)
    store = points.store_path(ctx.root, ctx.cfg.store_id)
    manifest.write(store, manifest.Manifest.new(
        ctx.cfg.store_id, "canonical", "points", ctx.cfg, n_records=df.height))
    ctx.catalog.record_store(store_id=ctx.cfg.store_id, layer="canonical", kind="points",
                             path=str(store), version=ctx.cfg.version, n_records=df.height,
                             config_hash=manifest.config_hash(ctx.cfg))
    ctx.catalog.flush()
    print(f"packed {df.height} rows into {len(paths)} partition file(s) at {store}")
    return 0


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
        print(f"  {variant.get('group', '-'):18s} fp={variant.get('fingerprint')} "
              f"n={variant.get('count')}  cols={len(variant.get('columns', []))}")
    # Compare per group first. The product-level fingerprint covers every
    # configured group, so comparing it against a scan of a subset reports a
    # difference that is not drift — a false alarm teaches you to ignore the
    # real one.
    per_group = ctx.cfg.extra.get("schema_fingerprints") or {}
    drifted: list[str] = []
    scanned_groups = set()
    for variant in report.get("variants", []):
        group = variant.get("group")
        if group is None:
            continue
        scanned_groups.add(group)
        want = per_group.get(group)
        got = variant.get("fingerprint")
        if want and want != got:
            drifted.append(f"{group}: {got} != pinned {want}")

    if drifted:
        print("SCHEMA DRIFT:")
        for line in drifted:
            print(f"  {line}")
        print("  inspect the variants above, then re-pin in the product config")
        return 3

    if per_group and scanned_groups:
        covered = scanned_groups >= set(per_group)
        if covered:
            print(f"all {len(per_group)} pinned group schemas match")
        else:
            missing = ", ".join(sorted(set(per_group) - scanned_groups)) or "-"
            print(f"{len(scanned_groups)} of {len(per_group)} groups scanned and matching "
                  f"(not seen here: {missing})")
    elif ctx.cfg.schema_fingerprint:
        seen = report.get("fingerprint")
        if seen != ctx.cfg.schema_fingerprint:
            print(f"SCHEMA DRIFT: {seen} != pinned {ctx.cfg.schema_fingerprint}")
            return 3
        print("fingerprint matches the pin")

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
