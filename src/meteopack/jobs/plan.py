"""Turn (source, product, window) into the units this task should actually do.

Three things happen here, and nowhere else: configs are resolved into objects,
the ledger removes work already done, and what remains is split across Slurm
array tasks. Keeping it in one place is what lets every stage be re-run safely
and lets a preempted job cost at most its in-flight unit.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from meteopack.core.catalog import Catalog, job_key
from meteopack.core.config import ProductConfig, SourceConfig, category_for_source
from meteopack.sources.base import Source, TimeWindow, Unit

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def config_dir(override: str | Path | None = None) -> Path:
    return Path(override) if override else CONFIG_DIR


#: `${NAME}` or `${NAME:-fallback}`, the shell spelling, so a config can give a
#: default without the reader needing one baked in.
_VAR_RE = re.compile(r"\$\{(\w+)(:-([^}]*))?\}")


def _substitute(m: re.Match[str]) -> str:
    value = os.environ.get(m.group(1))
    if value:
        return value
    return m.group(3) if m.group(2) is not None else m.group(0)


def _expand(value):
    """Expand `${VAR}`, `${VAR:-default}` and `~` in every string in a config.

    Configs otherwise carry one machine's absolute paths - the archive roots,
    the scratch directory - and a checkout is useless on any other machine
    until someone finds and edits them. `${VAR}` keeps the config declarative
    while letting the site supply the path.

    An undefined variable is left as written rather than expanded to the empty
    string: `${ARCHIVE}/knmi` silently becoming `/knmi` is how a job ends up
    reading someone else's directory, or an empty one.
    """
    if isinstance(value, str):
        out = _VAR_RE.sub(_substitute, value)
        out = os.path.expanduser(out)
        if "${" in out:
            missing = sorted(set(_VAR_RE.findall(out)))
            logger.warning(
                "config refers to undefined environment variable(s) %s in %r",
                ", ".join(n for n, _, _ in missing), value,
            )
        return out
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


def load_yaml(path: Path) -> dict:
    """Read a config file, with environment variables expanded."""
    return _expand(yaml.safe_load(path.read_text()))


def load_source_config(name: str, *, configs: str | Path | None = None) -> SourceConfig:
    base = config_dir(configs) / "sources"
    category = category_for_source(name)
    path = base / category / f"{name}.yaml" if category else base / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no source config for {name!r}: {path}")
    return SourceConfig(**load_yaml(path))


def load_product_config(
    source: str, product: str | None = None, *, configs: str | Path | None = None
) -> ProductConfig:
    """Load `configs/products/<category>/<source>/<product>.yaml`.

    With no product named, a source with exactly one product resolves to it;
    with several, the caller must choose rather than get an arbitrary one.
    """
    products = config_dir(configs) / "products"
    category = category_for_source(source)
    base = products / category / source if category else products / source
    if not base.is_dir():
        raise FileNotFoundError(f"no product configs for source {source!r}: {base}")
    if product:
        path = base / f"{product}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"no product config {product!r} for {source!r}: {path}")
    else:
        found = sorted(base.glob("*.yaml"))
        if len(found) != 1:
            names = ", ".join(p.stem for p in found) or "(none)"
            raise ValueError(f"source {source!r} has products [{names}]; name one explicitly")
        path = found[0]
    return ProductConfig(**load_yaml(path))


def is_volatile(unit: Unit, cfg: ProductConfig) -> bool:
    """True if this unit's CONTENT changes upstream while its key stays the same.

    DWD's `recent/` and `now/` archives are rewritten daily — the key is stable
    but the bytes are not, so a ledger that skips on key alone would never
    refresh them and the live tail would silently freeze. A source marks such
    units with `meta["volatile"]`, or a product config lists
    `extra.volatile_subdirs`.
    """
    meta = unit.meta or {}
    if meta.get("volatile"):
        return True
    volatile_subdirs = set(cfg.extra.get("volatile_subdirs") or ())
    return bool(volatile_subdirs) and meta.get("subdir") in volatile_subdirs


def shard_span_seconds(cfg: ProductConfig) -> int:
    """How much time one zarr shard file covers."""
    frames = (cfg.store.shard or {}).get("time") or max(1, 86400 // cfg.cadence_seconds)
    return int(frames) * cfg.cadence_seconds


def timeline_origin(cfg: ProductConfig) -> datetime:
    """Where the store's time axis starts, and therefore where shards begin.

    Defaults to the Unix epoch, which is right whenever the archive's natural
    unit aligns with midnight. It does not for KNMI radar: a daily tar runs
    08:05 to 08:00, so on an epoch-origin axis every tar straddles two shards,
    two adjacent tars share the middle one, and the two tasks holding them lose
    whole frames. Setting the origin to an 08:05 instant makes one tar exactly
    one shard.
    """
    raw = cfg.extra.get("timeline_origin")
    if not raw:
        return datetime(1970, 1, 1, tzinfo=UTC)
    value = datetime.fromisoformat(str(raw))
    return value if value.tzinfo else value.replace(tzinfo=UTC)


def shard_bucket(unit: Unit, span_seconds: int, origin: datetime | None = None) -> int:
    """Which shard file this unit's frame lands in.

    Gridded units MUST carry `valid_time`: a zarr shard is one file, and two
    processes writing different frames of the same shard lose whole frames
    rather than corrupting them (measured: 7-9 of 12 frames lost, every trial).
    Without a time there is no way to keep a shard to a single writer.
    """
    if unit.valid_time is None:
        raise ValueError(
            f"unit {unit.key!r} has no valid_time; a gridded source must set it so work "
            f"can be split on shard boundaries (otherwise concurrent tasks lose frames)"
        )
    base = origin or datetime(1970, 1, 1, tzinfo=UTC)
    return int((unit.valid_time - base).total_seconds()) // span_seconds


def shard_units(
    units: Sequence[Unit],
    shard: int,
    shards: int,
    *,
    mode: str = "stride",
    span_seconds: int | None = None,
    origin: datetime | None = None,
) -> list[Unit]:
    """Split the unit list across array tasks.

    `stride` is the default and right for points data: cost correlates with
    station index, so contiguous blocks would hand one task every large file,
    and a points partition is written per task anyway.

    `bucket` is required for gridded data. A zarr shard is a single file, so
    two tasks writing different frames of one shard silently lose frames. This
    mode keeps every frame of a shard with one task.
    """
    if shards < 1 or not (0 <= shard < shards):
        raise ValueError(f"bad shard {shard}/{shards}")
    if mode == "stride":
        return list(units[shard::shards])
    if mode != "bucket":
        raise ValueError(f"unknown shard mode {mode!r}")
    if not span_seconds:
        raise ValueError("bucket mode needs span_seconds")

    groups: dict[int, list[Unit]] = {}
    for unit in units:
        groups.setdefault(shard_bucket(unit, span_seconds, origin), []).append(unit)
    mine = sorted(groups)[shard::shards]
    return [unit for bucket in mine for unit in groups[bucket]]


def _vanished(catalog: Catalog, cfg: ProductConfig, keys: set[str], todo: set[str],
              stage: str) -> set[str]:
    """Job keys the ledger calls done whose recorded file no longer exists.

    The ledger is trusted to skip work, so a raw file deleted after it was
    fetched - by hand, by a clean-up, by a raw directory swapped for a fresh
    one - would otherwise never be fetched again, and the gap would show up
    only as a hole in the packed store. One `stat` per done unit is cheap
    next to the request it saves being forgotten.
    """
    done = [k for k in keys if job_key(stage, k) not in todo]
    if not done:
        return set()
    src = cfg.source.replace("'", "''")
    prod = cfg.product.replace("'", "''")
    rows = catalog.query(
        f"SELECT unit_key, path FROM files WHERE source = '{src}' AND product = '{prod}'")
    paths = dict(zip(rows["unit_key"].to_list(), rows["path"].to_list(), strict=True)) \
        if len(rows) else {}
    wanted = set(done)
    return {job_key(stage, k) for k, p in paths.items()
            if k in wanted and p and not Path(p).exists()}


@dataclass(slots=True)
class Plan:
    stage: str
    source: str
    product: str
    units: list[Unit] = field(default_factory=list)
    listed: int = 0
    skipped_done: int = 0
    volatile: int = 0
    #: Everything `list` returned, before the ledger and the shard took their
    #: share - what the gap report compares the raw layer against.
    listed_units: list[Unit] = field(default_factory=list)
    #: Units the ledger had as done whose file has since disappeared.
    vanished: int = 0

    @property
    def n(self) -> int:
        return len(self.units)

    def summary(self) -> str:
        gone = f", {self.vanished} done but gone from disk" if self.vanished else ""
        return (
            f"{self.stage} {self.source}/{self.product}: {self.n} to do "
            f"({self.listed} listed, {self.skipped_done} already done, "
            f"{self.volatile} volatile and always refreshed{gone})"
        )


def build_plan(
    stage: str,
    src: Source,
    cfg: ProductConfig,
    window: TimeWindow,
    catalog: Catalog | None,
    *,
    shard: int = 0,
    shards: int = 1,
    refresh: bool = False,
) -> Plan:
    """List remotely, drop what the ledger has, then take this task's stride."""
    units = list(src.list(window, cfg))
    plan = Plan(stage=stage, source=cfg.source, product=cfg.product, listed=len(units),
                listed_units=list(units))

    volatile = {u.key for u in units if is_volatile(u, cfg)}
    plan.volatile = len(volatile)

    if catalog is not None and not refresh:
        keys = [job_key(stage, u.key) for u in units]
        todo = set(catalog.pending(stage, keys))
        if stage == "fetch":
            vanished = _vanished(catalog, cfg, {u.key for u in units} - volatile, todo, stage)
            plan.vanished = len(vanished)
            todo |= vanished
        units = [u for u in units if job_key(stage, u.key) in todo or u.key in volatile]
        plan.skipped_done = plan.listed - len(units)

    # Gridded writes are only safe one-shard-per-task; points can stride.
    if cfg.store.kind == "gridded" and shards > 1:
        plan.units = shard_units(units, shard, shards, mode="bucket",
                                 span_seconds=shard_span_seconds(cfg),
                                 origin=timeline_origin(cfg))
    else:
        plan.units = shard_units(units, shard, shards)
    return plan
