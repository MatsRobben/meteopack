"""Turn (source, product, window) into the units this task should actually do.

Three things happen here, and nowhere else: configs are resolved into objects,
the ledger removes work already done, and what remains is split across Slurm
array tasks. Keeping it in one place is what lets every stage be re-run safely
and lets a preempted job cost at most its in-flight unit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from meteopack.core.catalog import Catalog, job_key
from meteopack.core.config import ProductConfig, SourceConfig
from meteopack.sources.base import Source, TimeWindow, Unit

if TYPE_CHECKING:
    from collections.abc import Sequence

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs"


def config_dir(override: str | Path | None = None) -> Path:
    return Path(override) if override else CONFIG_DIR


def load_source_config(name: str, *, configs: str | Path | None = None) -> SourceConfig:
    path = config_dir(configs) / "sources" / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"no source config for {name!r}: {path}")
    return SourceConfig(**yaml.safe_load(path.read_text()))


def load_product_config(
    source: str, product: str | None = None, *, configs: str | Path | None = None
) -> ProductConfig:
    """Load `configs/products/<source>/<product>.yaml`.

    With no product named, a source with exactly one product resolves to it;
    with several, the caller must choose rather than get an arbitrary one.
    """
    base = config_dir(configs) / "products" / source
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
    return ProductConfig(**yaml.safe_load(path.read_text()))


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


def shard_units(units: Sequence[Unit], shard: int, shards: int) -> list[Unit]:
    """Split by stride, not by block.

    Stride keeps the per-task cost even when cost correlates with position —
    historical archives grow with station index, so contiguous blocks would
    give one task all the big files.
    """
    if shards < 1 or not (0 <= shard < shards):
        raise ValueError(f"bad shard {shard}/{shards}")
    return list(units[shard::shards])


@dataclass(slots=True)
class Plan:
    stage: str
    source: str
    product: str
    units: list[Unit] = field(default_factory=list)
    listed: int = 0
    skipped_done: int = 0
    volatile: int = 0

    @property
    def n(self) -> int:
        return len(self.units)

    def summary(self) -> str:
        return (
            f"{self.stage} {self.source}/{self.product}: {self.n} to do "
            f"({self.listed} listed, {self.skipped_done} already done, "
            f"{self.volatile} volatile and always refreshed)"
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
    plan = Plan(stage=stage, source=cfg.source, product=cfg.product, listed=len(units))

    volatile = {u.key for u in units if is_volatile(u, cfg)}
    plan.volatile = len(volatile)

    if catalog is not None and not refresh:
        keys = [job_key(stage, u.key) for u in units]
        todo = set(catalog.pending(stage, keys))
        units = [u for u in units if job_key(stage, u.key) in todo or u.key in volatile]
        plan.skipped_done = plan.listed - len(units)

    plan.units = shard_units(units, shard, shards)
    return plan
