"""The planner decides what work a task does; getting it wrong wastes a cluster."""
from __future__ import annotations

from datetime import timedelta

import pytest

from meteopack.core.catalog import Catalog, job_key
from meteopack.core.config import ProductConfig, StoreConfig
from meteopack.core.timeline import utc
from meteopack.jobs.plan import (
    Plan,
    build_plan,
    is_volatile,
    load_product_config,
    load_source_config,
    shard_units,
)
from meteopack.sources.base import TimeWindow, Unit


def _cfg(**extra) -> ProductConfig:
    return ProductConfig(source="dwd", product="stations_10min",
                         store=StoreConfig(kind="points"), extra=extra)


class FakeSource:
    name = "dwd"

    def __init__(self, units):
        self._units = units

    def list(self, window, cfg):
        return list(self._units)

    def fetch(self, unit, raw_root):
        raise NotImplementedError

    def parse(self, raw, cfg):
        raise NotImplementedError


def test_shards_partition_every_unit_exactly_once():
    units = [Unit(key=str(i), url="u") for i in range(23)]
    for shards in (1, 2, 3, 7):
        seen = [u.key for s in range(shards) for u in shard_units(units, s, shards)]
        assert sorted(seen, key=int) == [u.key for u in units]
        sizes = [len(shard_units(units, s, shards)) for s in range(shards)]
        assert max(sizes) - min(sizes) <= 1, "stride keeps tasks balanced"


def test_bad_shard_arguments_raise():
    with pytest.raises(ValueError):
        shard_units([], 3, 3)
    with pytest.raises(ValueError):
        shard_units([], -1, 2)


def test_ledger_skips_done_units(tmp_path):
    units = [Unit(key=f"u{i}", url="u") for i in range(5)]
    cat = Catalog(tmp_path)
    cat.mark(job_key("fetch", "u1"), "done")
    cat.mark(job_key("fetch", "u3"), "done")
    window = TimeWindow(utc(2012, 1, 1), utc(2012, 1, 2))
    plan = build_plan("fetch", FakeSource(units), _cfg(), window, cat)
    assert [u.key for u in plan.units] == ["u0", "u2", "u4"]
    assert plan.listed == 5 and plan.skipped_done == 2


def test_volatile_units_are_never_skipped(tmp_path):
    """recent/ and now/ are rewritten upstream under the same filename.

    Skipping them on key alone would freeze the live tail — the store would
    look healthy and simply stop advancing.
    """
    units = [
        Unit(key="hist", url="u", meta={"subdir": "historical"}),
        Unit(key="rec", url="u", meta={"subdir": "recent"}),
        Unit(key="flagged", url="u", meta={"volatile": True}),
    ]
    cfg = _cfg(volatile_subdirs=["recent", "now"])
    cat = Catalog(tmp_path)
    for u in units:
        cat.mark(job_key("fetch", u.key), "done")
    plan = build_plan("fetch", FakeSource(units), cfg, TimeWindow(utc(2012), utc(2013)), cat)
    assert [u.key for u in plan.units] == ["rec", "flagged"]
    assert plan.volatile == 2
    assert is_volatile(units[0], cfg) is False


def test_refresh_ignores_the_ledger(tmp_path):
    units = [Unit(key=f"u{i}", url="u") for i in range(3)]
    cat = Catalog(tmp_path)
    for u in units:
        cat.mark(job_key("fetch", u.key), "done")
    plan = build_plan("fetch", FakeSource(units), _cfg(), TimeWindow(utc(2012), utc(2013)),
                      cat, refresh=True)
    assert plan.n == 3


def test_real_configs_load():
    assert load_source_config("dwd").name == "dwd"
    cfg = load_product_config("dwd")
    assert cfg.store_id == "dwd/stations_10min/v1"
    assert cfg.cadence == timedelta(minutes=10)
    assert cfg.extra.get("volatile_subdirs") == ["recent", "now"]


def test_ambiguous_product_must_be_named(tmp_path):
    (tmp_path / "products" / "x").mkdir(parents=True)
    for name in ("a", "b"):
        (tmp_path / "products" / "x" / f"{name}.yaml").write_text(
            f"source: x\nproduct: {name}\nstore: {{kind: points}}\n")
    with pytest.raises(ValueError, match="name one explicitly"):
        load_product_config("x", configs=tmp_path)
    assert load_product_config("x", "a", configs=tmp_path).product == "a"


def test_plan_summary_is_informative():
    p = Plan(stage="fetch", source="dwd", product="p", listed=10, skipped_done=4, volatile=2)
    assert "4 already done" in p.summary() and "2 volatile" in p.summary()


def test_gridded_work_is_split_on_shard_boundaries():
    """Two tasks must never write into the same zarr shard.

    A shard is one file. Concurrent writers to one shard lose whole frames —
    measured at 7-9 of 12 lost, every trial — so stride splitting, which is
    right for points, would silently destroy gridded data.
    """
    from datetime import timedelta

    from meteopack.jobs.plan import shard_bucket, shard_span_seconds, shard_units

    cfg = ProductConfig(source="s", product="p", cadence_seconds=600, grid_id="g",
                        store=StoreConfig(kind="gridded", chunks={"time": 1},
                                          shard={"time": 6}))
    span = shard_span_seconds(cfg)
    assert span == 3600, "6 frames x 600 s"

    start = utc(2012, 1, 1)
    units = [Unit(key=f"f{i}", url="u", valid_time=start + timedelta(minutes=10 * i))
             for i in range(24)]                      # 4 shards' worth

    shards = 3
    assignment: dict[int, int] = {}
    seen: list[str] = []
    for task in range(shards):
        got = shard_units(units, task, shards, mode="bucket", span_seconds=span)
        seen += [u.key for u in got]
        for u in got:
            bucket = shard_bucket(u, span)
            assert assignment.setdefault(bucket, task) == task, "a shard went to two tasks"
    assert sorted(seen, key=lambda k: int(k[1:])) == [u.key for u in units]


def test_gridded_unit_without_a_time_is_refused_not_guessed():
    from meteopack.jobs.plan import shard_units

    units = [Unit(key="x", url="u")]
    with pytest.raises(ValueError, match="valid_time"):
        shard_units(units, 0, 2, mode="bucket", span_seconds=3600)
