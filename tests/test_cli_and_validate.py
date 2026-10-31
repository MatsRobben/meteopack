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




def _stations() -> pl.DataFrame:
    return pl.DataFrame({"station_id": ["00044"], "lat": [52.1], "lon": [5.2],
                         "height": [2.0], "name": ["De Bilt"]})


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
    points.write_stations(_stations(), cfg.store_id, tmp_path)
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


def test_validate_catches_observations_without_station_metadata(tmp_path):
    """Positions missing is a silent failure downstream, so catch it here.

    `pack` once wrote observations but never the station table. Nothing failed:
    the store validated clean, and a dense station matrix simply came back with
    every coordinate NaN and a third of the stations gone.
    """
    cfg = _cfg()
    times = [datetime(2012, 1, 1, 0, 0), datetime(2012, 1, 1, 0, 10)]
    points.write(_frame(times, [5.0, 6.0]), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "canonical", "points", cfg))

    report = validate(cfg, tmp_path)
    assert not report.ok
    check = next(c for c in report.checks if c.name == "stations have metadata")
    assert not check.ok and "station table is empty" in check.detail

    points.write_stations(
        pl.DataFrame({"station_id": ["00044"], "lat": [52.1], "lon": [5.2],
                      "height": [2.0], "name": ["De Bilt"]}),
        cfg.store_id, tmp_path)
    assert validate(cfg, tmp_path).ok


def test_pack_writes_station_metadata_when_the_source_offers_it(tmp_path, monkeypatch, capsys):
    """`stations()` is an optional capability, like `scan()` — use it if present."""
    from meteopack import cli

    class Ctx:
        cfg = _cfg()
        root = tmp_path

        class src:
            @staticmethod
            def stations(cfg):
                return pl.DataFrame({"station_id": ["00044"], "lat": [52.1], "lon": [5.2],
                                     "height_m": [2.0], "name": ["De Bilt"], "state": ["X"]})

    n = cli._write_station_metadata(Ctx())
    assert n == 1
    back = points.read_stations(_cfg().store_id, tmp_path)
    assert back["station_id"].to_list() == ["00044"]
    assert "height" in back.columns, "height_m should be renamed to the store's name"


def test_gridded_timeline_snaps_to_the_shard_grid_not_the_cadence(tmp_path, monkeypatch):
    """An axis starting mid-shard makes every archive unit straddle two shards.

    A zarr shard is one file, so two workers that share one lose whole frames
    rather than corrupting them. Measured on three real radar tars while this
    was wrong: serial wrote 268.2M finite pixels, three workers wrote 222.7M —
    17% gone, silently.
    """
    from meteopack.cli import Context, build_parser
    from meteopack.jobs.plan import shard_span_seconds, timeline_origin

    args = build_parser().parse_args(
        ["pack", "knmi_radar", "--window", "2024-07-02/2024-07-04",
         "--data-root", str(tmp_path)])
    ctx = Context(args)
    from meteopack.cli import _timeline_for

    tl = _timeline_for(ctx)
    origin = timeline_origin(ctx.cfg)
    span = shard_span_seconds(ctx.cfg)

    start_offset = (tl.start - origin).total_seconds()
    assert start_offset % span == 0, (
        f"axis starts {start_offset % span:.0f}s into a shard; every unit would straddle"
    )
    assert len(tl) % (span // ctx.cfg.cadence_seconds) == 0, "axis is a whole number of shards"
    assert tl.start.hour == 8 and tl.start.minute == 5, "a radar day starts 08:05"


def test_gridded_timeline_can_be_derived_from_the_catalog(tmp_path):
    """Packing without --window must work, using the times fetch recorded.

    This path had an undefined name for a while: every test passed --window, so
    nothing executed it. A source whose archive predates today would hit it on
    the first real run.
    """
    from datetime import UTC, datetime

    from meteopack.cli import Context, _timeline_for, build_parser
    from meteopack.core.catalog import Catalog
    from meteopack.sources.base import RawFile

    cat = Catalog(tmp_path)
    for day in (2, 3):
        cat.record_file(RawFile(tmp_path / f"{day}.tar", "0" * 64, 1, f"u{day}"),
                        source="knmi_radar", product="rtcor_5min", url="file:///x",
                        valid_time=datetime(2024, 7, day, 8, 5, tzinfo=UTC))
    cat.flush()

    args = build_parser().parse_args(["pack", "knmi_radar", "--data-root", str(tmp_path)])
    tl = _timeline_for(Context(args))
    assert tl.start <= datetime(2024, 7, 2, 8, 5, tzinfo=UTC)
    assert tl.end >= datetime(2024, 7, 3, 8, 5, tzinfo=UTC)


def test_pack_without_a_window_or_catalog_says_what_to_do(tmp_path):
    from meteopack.cli import Context, _timeline_for, build_parser

    args = build_parser().parse_args(["pack", "knmi_radar", "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="Pass --window"):
        _timeline_for(Context(args))
