"""Irregular points products: an explicit mode, not a weakened regular one."""
from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from meteopack.core.config import ProductConfig, StoreConfig, Variable
from meteopack.core.store import manifest, points
from meteopack.core.validate import validate


def _cfg(cadence, **store) -> ProductConfig:
    return ProductConfig(source="s", product="p", cadence_seconds=cadence,
                         store=StoreConfig(kind="points", **store),
                         variables=[Variable(name="air_temperature", source_key="t",
                                             units="degC")])


def test_irregular_must_be_said_and_only_for_points():
    assert _cfg(600).cadence_seconds == 600 and not _cfg(600).irregular
    default = ProductConfig(source="s", product="p", store=StoreConfig(kind="points"))
    assert default.cadence_seconds == 600, "omitting the field must not make a store irregular"
    assert _cfg(None).irregular
    with pytest.raises(ValueError, match="irregular"):
        _ = _cfg(None).cadence
    with pytest.raises(ValueError, match="points"):
        ProductConfig(source="s", product="p", cadence_seconds=None,
                      store=StoreConfig(kind="gridded"))


def _write(cfg, root, times):
    frame = pl.DataFrame({"time": times, "station_id": ["a"] * len(times),
                          "air_temperature": [1.0] * len(times)}).with_columns(
        pl.col("time").cast(pl.Datetime("us")))
    points.write(frame, cfg, root)
    points.write_stations(pl.DataFrame({"station_id": ["a"], "lat": [52.0], "lon": [5.0]}),
                          cfg.store_id, root)
    manifest.write(points.store_path(root, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "packed", "points", cfg))


def test_off_grid_times_pass_only_when_the_product_is_irregular(tmp_path):
    times = [datetime(2024, 1, 1, 0, 0, 7), datetime(2024, 1, 1, 0, 0, 7),
             datetime(2024, 1, 1, 0, 3, 31)]
    _write(_cfg(None), tmp_path / "irr", times)
    assert validate(_cfg(None), tmp_path / "irr").ok
    _write(_cfg(600), tmp_path / "reg", times)
    report = validate(_cfg(600), tmp_path / "reg")
    failed = {c.name for c in report.checks if not c.ok}
    assert failed == {"no duplicate (time, station)", "all times on cadence"}


def test_sort_order_and_row_groups_come_from_the_config(tmp_path):
    cfg = _cfg(None, sort_by=["station_id", "time"], row_group_rows=2,
               partition_by=["year", "month"])
    frame = pl.DataFrame({
        "time": [datetime(2024, 1, 1, 0, 1), datetime(2024, 1, 1, 0, 0),
                 datetime(2024, 1, 1, 0, 2)],
        "station_id": ["b", "b", "a"], "air_temperature": [1.0, 2.0, 3.0]})
    (path,) = points.write(frame, cfg, tmp_path)
    assert path.parent.name == "month=01"
    back = pl.read_parquet(path)
    assert back["station_id"].to_list() == ["a", "b", "b"]
    assert back["time"].is_sorted() is False and back["air_temperature"].to_list() == [
        3.0, 2.0, 1.0]
    import duckdb
    groups = duckdb.connect().execute(f"select count(*) from parquet_metadata('{path}') "
                                      "where path_in_schema = 'time'").fetchone()[0]
    assert groups == 2


def test_partition_labels_are_unchanged_by_vectorising():
    cfg = _cfg(600, partition_by=["year", "month", "day"])
    frame = pl.DataFrame({"time": [datetime(2024, 3, 5, 1), datetime(1999, 12, 31, 23)],
                          "station_id": ["a", "b"]})
    assert points.partition_labels(frame, cfg).to_list() == ["2024/03/05", "1999/12/31"]
    assert points.partition_label_of(cfg, datetime(2024, 3, 5, 1)) == "2024/03/05"


def test_station_metadata_keeps_extra_columns(tmp_path):
    cfg = _cfg(None)
    points.write_stations(pl.DataFrame({"station_id": ["a"], "lat": [52.0], "lon": [5.0],
                                        "official": [True]}), cfg.store_id, tmp_path)
    meta = points.read_stations(cfg.store_id, tmp_path)
    assert meta["official"].to_list() == [True] and meta["height"].null_count() == 1
