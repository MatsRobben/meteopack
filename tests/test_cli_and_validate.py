"""CLI surface and store validation.

The CLI is what a new user touches first, so its failure messages are part of
the product: an unknown source or an empty store must say what to do next, not
raise a traceback.
"""
from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from meteopack.cli import build_parser, main, parse_window
from meteopack.core.config import ProductConfig, StoreConfig, Variable
from meteopack.core.store import manifest, points
from meteopack.core.validate import validate


def _cfg() -> ProductConfig:
    return ProductConfig(
        source="dwd", product="stations_10min", cadence_seconds=600,
        store=StoreConfig(kind="points"),
        variables=[Variable(name="air_temperature", source_key="TT_10", units="degC",
                            valid_min=-60, valid_max=60)],
    )


def _frame(times, values, station="00044") -> pl.DataFrame:
    return pl.DataFrame({
        "time": times,
        "station_id": [station] * len(times),
        "air_temperature": values,
    }).with_columns(pl.col("time").cast(pl.Datetime("us")))


# ------------------------------------------------------------------ windows

def test_window_accepts_bare_years_and_is_half_open():
    w = parse_window("2012/2013")
    assert w.start == datetime(2012, 1, 1, tzinfo=UTC)
    assert w.end == datetime(2014, 1, 1, tzinfo=UTC), "a bare END year means its whole year"


def test_window_defaults_to_the_last_day():
    w = parse_window(None)
    assert 0 < (w.end - w.start).total_seconds() <= 86401


def test_window_without_a_separator_is_rejected():
    with pytest.raises(SystemExit):
        parse_window("2012")


def test_every_stage_is_reachable_from_the_parser():
    for stage in ("scan", "list", "fetch", "parse", "pack", "derive", "validate", "catalog"):
        assert build_parser().parse_args([stage, "dwd"]).stage == stage


# ------------------------------------------------------------- error paths

def test_unknown_source_explains_itself(capsys, tmp_path):
    code = main(["list", "definitely-not-a-source", "--data-root", str(tmp_path)])
    assert code == 2
    assert "no product configs" in capsys.readouterr().err


def test_pack_with_no_raw_data_tells_you_what_to_run(capsys, tmp_path):
    code = main(["pack", "dwd", "--data-root", str(tmp_path)])
    assert code == 1
    assert "meteopack fetch dwd" in capsys.readouterr().out


# --------------------------------------------------------------- validation

def test_validate_passes_on_a_healthy_store(tmp_path):
    cfg = _cfg()
    times = pl.datetime_range(datetime(2012, 1, 1), datetime(2012, 1, 1, 1),
                              interval="10m", eager=True, closed="left")
    points.write(_frame(times, [5.0] * len(times)), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "canonical", "points", cfg))
    report = validate(cfg, tmp_path)
    assert report.ok, report.text()
    assert report.stats["rows"] == len(times)


def test_validate_catches_off_cadence_rows(tmp_path):
    """A parser that disagrees with the configured cadence must not pass silently."""
    cfg = _cfg()
    times = [datetime(2012, 1, 1, 0, 0), datetime(2012, 1, 1, 0, 7)]   # 7 min: off the 600 s grid
    points.write(_frame(times, [1.0, 2.0]), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "canonical", "points", cfg))
    report = validate(cfg, tmp_path)
    assert not report.ok
    assert any("cadence" in c.name and not c.ok for c in report.checks)


def test_validate_catches_out_of_range_values(tmp_path):
    cfg = _cfg()
    times = [datetime(2012, 1, 1, 0, 0), datetime(2012, 1, 1, 0, 10)]
    points.write(_frame(times, [5.0, 999.0]), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "canonical", "points", cfg))
    report = validate(cfg, tmp_path)
    assert not report.ok
    assert any("valid range" in c.name and not c.ok for c in report.checks)


def test_validate_flags_a_store_built_from_different_config(tmp_path):
    cfg = _cfg()
    times = [datetime(2012, 1, 1, 0, 0)]
    points.write(_frame(times, [5.0]), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "canonical", "points", cfg))
    moved = cfg.model_copy(update={"cadence_seconds": 300})
    report = validate(moved, tmp_path)
    assert any("manifest" in c.name and not c.ok for c in report.checks)


def test_validate_reports_a_missing_store_rather_than_raising(tmp_path):
    report = validate(_cfg(), tmp_path)
    assert not report.ok and "store exists" in report.text()
