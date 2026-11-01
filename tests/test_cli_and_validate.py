"""CLI surface and store validation.

The CLI is what a new user touches first, so its failure messages are part of
the product: an unknown source or an empty store must say what to do next, not
raise a traceback.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

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
                   manifest.Manifest.new(cfg.store_id, "packed", "points", cfg))
    report = validate(cfg, tmp_path)
    assert report.ok, report.text()
    assert report.stats["rows"] == len(times)


def test_validate_catches_off_cadence_rows(tmp_path):
    """A parser that disagrees with the configured cadence must not pass silently."""
    cfg = _cfg()
    times = [datetime(2012, 1, 1, 0, 0), datetime(2012, 1, 1, 0, 7)]   # 7 min: off the 600 s grid
    points.write(_frame(times, [1.0, 2.0]), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "packed", "points", cfg))
    report = validate(cfg, tmp_path)
    assert not report.ok
    assert any("cadence" in c.name and not c.ok for c in report.checks)


def test_validate_catches_out_of_range_values(tmp_path):
    cfg = _cfg()
    times = [datetime(2012, 1, 1, 0, 0), datetime(2012, 1, 1, 0, 10)]
    points.write(_frame(times, [5.0, 999.0]), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "packed", "points", cfg))
    report = validate(cfg, tmp_path)
    assert not report.ok
    assert any("valid range" in c.name and not c.ok for c in report.checks)


def test_validate_flags_a_store_built_from_different_config(tmp_path):
    cfg = _cfg()
    times = [datetime(2012, 1, 1, 0, 0)]
    points.write(_frame(times, [5.0]), cfg, tmp_path)
    manifest.write(points.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "packed", "points", cfg))
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
                   manifest.Manifest.new(cfg.store_id, "packed", "points", cfg))

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
        # The real Context has one, and `_write_station_metadata` reads any
        # archived catalogue under it. A double without it is looser than the
        # thing it stands in for, which is how the attribute went unnoticed.
        raw_root = tmp_path / "raw" / "knmi" / "obs"

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
        ["pack", "knmi_radar", "--product", "rtcor_5min",
         "--window", "2024-07-02/2024-07-04",
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

    args = build_parser().parse_args(
        ["pack", "knmi_radar", "--product", "rtcor_5min", "--data-root", str(tmp_path)])
    tl = _timeline_for(Context(args))
    assert tl.start <= datetime(2024, 7, 2, 8, 5, tzinfo=UTC)
    assert tl.end >= datetime(2024, 7, 3, 8, 5, tzinfo=UTC)


def test_an_uncatalogued_archive_sizes_its_axis_from_the_filenames(tmp_path, monkeypatch):
    """A backfill from an archive copied in by hand has no catalog.

    The axis then has to come from the filenames, which KNMI's carry. Typing
    the window instead is how the real rtcor axis came to begin one step after
    the archive's first frame: the first three 2018 tars run 08:00->07:55 and
    every later one runs 08:05->08:00, so a hand-typed 08:05 start dropped the
    archive's very first frame.
    """
    from datetime import UTC, datetime

    from meteopack.cli import Context, _timeline_for, build_parser

    archive = tmp_path / "raw" / "knmi_radar" / "rtcor_5min"
    archive.mkdir(parents=True)
    for name in (
        "RAD25_OPER_R___TARRRT__L2__20181219T080000_20181220T075500_0001.tar",
        "RAD25_OPER_R___TARRRT__L2__20181220T080500_20181221T080000_0001.tar",
    ):
        (archive / name).write_bytes(b"")
    monkeypatch.setenv("KNMI_RADAR_ARCHIVE", str(archive))

    args = build_parser().parse_args(
        ["pack", "knmi_radar", "--product", "rtcor_5min", "--data-root", str(tmp_path)])
    tl = _timeline_for(Context(args))
    # The 08:00 frame is inside the axis, not one step before it.
    assert tl.start <= datetime(2018, 12, 19, 8, 0, tzinfo=UTC)
    assert tl.end >= datetime(2018, 12, 21, 8, 0, tzinfo=UTC)
    # Still snapped onto the shard grid, so one tar stays one shard.
    assert tl.start.hour == 8 and tl.start.minute == 5


@pytest.mark.parametrize(("product", "env", "token", "prefix_day"), [
    ("echotop_5min", "KNMI_ECHOTOP_ARCHIVE", "TARETH", "T000000"),
    ("refl_5min", "KNMI_REFL_ARCHIVE", "TARPCP", "T000000"),
])
def test_exclusive_end_daily_tars_get_one_task_each(tmp_path, monkeypatch, product, env,
                                                    token, prefix_day):
    """Each daily tar owns exactly one shard, so it must be its own task.

    Their end token is exclusive (`..._20080102T000000` stops at 23:55). Read
    as the last frame, it is the first slot of the next day's shard, every tar
    appeared to write two shards, and union-by-shard chained the archive: 6,668
    echotop tars became 16 tasks.
    """
    from meteopack.cli import Context, _pack_tasks, _timeline_for, build_parser

    archive = tmp_path / "raw" / "knmi_radar" / product
    archive.mkdir(parents=True)
    for day in range(1, 11):
        a, b = f"200801{day:02d}", f"200801{day + 1:02d}"
        (archive / f"RAD25_OPER_R___{token}__L2__{a}{prefix_day}_{b}{prefix_day}_0001.tar"
         ).write_bytes(b"")
    monkeypatch.setenv(env, str(archive))
    args = build_parser().parse_args(
        ["pack", "knmi_radar", "--product", product, "--data-root", str(tmp_path)])
    ctx = Context(args)
    files = ctx.raw_files()
    tasks = _pack_tasks(ctx, files, _timeline_for(ctx))
    assert len(tasks) == len(files) == 10, [len(t.paths) for t in tasks]


def test_a_window_does_not_pull_in_the_day_before(tmp_path, monkeypatch):
    """The day before a window ends at the window's first instant, exclusively.

    Over-reporting its span by one frame made `--window 2015-01-01/...` pack the
    2014-12-31 tar too, and all 288 of its frames were refused as outside the axis.
    """
    from meteopack.cli import Context, _files_for_window, build_parser

    archive = tmp_path / "raw" / "knmi_radar" / "echotop_5min"
    archive.mkdir(parents=True)
    for a, b in (("20141231", "20150101"), ("20150101", "20150102")):
        (archive / f"RAD25_OPER_R___TARETH__L2__{a}T000000_{b}T000000_0001.tar").write_bytes(b"")
    monkeypatch.setenv("KNMI_ECHOTOP_ARCHIVE", str(archive))
    args = build_parser().parse_args(
        ["pack", "knmi_radar", "--product", "echotop_5min", "--data-root", str(tmp_path),
         "--window", "2015-01-01/2015-01-02"])
    ctx = Context(args)
    kept, _ = _files_for_window(ctx, ctx.raw_files())
    assert [p.name[27:35] for p in kept] == ["20150101"]


def test_pack_without_a_window_or_catalog_says_what_to_do(tmp_path):
    from meteopack.cli import Context, _timeline_for, build_parser

    args = build_parser().parse_args(
        ["pack", "knmi_radar", "--product", "rtcor_5min", "--data-root", str(tmp_path)])
    with pytest.raises(SystemExit, match="Pass --window"):
        _timeline_for(Context(args))


def test_raw_files_skips_sidecars_that_are_not_observations(tmp_path):
    """Pointing this pipeline at the previous one's archive made every source
    try to parse `station_catalog.json` as data and report schema drift."""
    import argparse

    from meteopack.cli import Context

    root = tmp_path / "raw" / "belgium" / "aws_10min"
    root.mkdir(parents=True)
    (root / "station_catalog.json").write_text("{}")
    (root / "6414.csv").write_text("a,b\n1,2\n")
    (root / "half.csv.part").write_text("x")

    class FakeCtx:
        raw_root = root
        SIDECARS = Context.SIDECARS
        raw_files = Context.raw_files

    names = [p.name for p in FakeCtx.raw_files(FakeCtx())]
    assert names == ["6414.csv"], names
    assert isinstance(argparse.Namespace(), argparse.Namespace)


def test_an_ordinary_machine_keeps_the_spill_beside_the_data(tmp_path, monkeypatch):
    """The data root is local disk: there is nothing to move it off.

    Relocating the spill only pays when the data root is on a *network*
    filesystem the pack is also reading from. On a laptop it would at best do
    nothing and at worst land on a slower disk or a tmpfs.
    """
    from meteopack import cli

    monkeypatch.delenv("METEOPACK_SPILL", raising=False)
    monkeypatch.setattr(cli, "_filesystem_type", lambda p: "ext4")
    monkeypatch.setattr(cli, "_LOCAL_SPILL_DIRS", (tmp_path / "scratch",))
    assert cli._spill_root(tmp_path / "data") == tmp_path / "data" / "_spill"
    assert not (tmp_path / "scratch").exists(), "must not even look for local disk"


def test_a_network_data_root_moves_the_spill_to_local_disk(tmp_path, monkeypatch):
    """~17 GB of spill traffic otherwise queues on the spindles the reads
    are already waiting for."""
    from meteopack import cli

    monkeypatch.delenv("METEOPACK_SPILL", raising=False)
    local = tmp_path / "scratch"
    types = {str(tmp_path / "data"): "nfs4", str(local): "xfs"}
    monkeypatch.setattr(cli, "_filesystem_type",
                        lambda p: types.get(str(p), "ext4"))
    monkeypatch.setattr(cli, "_LOCAL_SPILL_DIRS", (local,))
    monkeypatch.setattr(cli.shutil, "disk_usage",
                        lambda p: SimpleNamespace(total=1 << 40, used=0, free=1 << 40))
    assert cli._spill_root(tmp_path / "data") == local


def test_a_tmpfs_is_never_chosen_however_much_it_reports(tmp_path, monkeypatch):
    """It is RAM. Spilling there puts the data back in memory, which is the
    whole thing spilling exists to avoid - and /tmp is a tmpfs on many
    systems."""
    from meteopack import cli

    monkeypatch.delenv("METEOPACK_SPILL", raising=False)
    ram = tmp_path / "ram"
    types = {str(tmp_path / "data"): "nfs4", str(ram): "tmpfs"}
    monkeypatch.setattr(cli, "_filesystem_type",
                        lambda p: types.get(str(p), "ext4"))
    monkeypatch.setattr(cli, "_LOCAL_SPILL_DIRS", (ram,))
    monkeypatch.setattr(cli.shutil, "disk_usage",
                        lambda p: SimpleNamespace(total=1 << 40, used=0, free=1 << 40))
    assert cli._spill_root(tmp_path / "data") == tmp_path / "data" / "_spill"


def test_local_disk_without_room_is_skipped(tmp_path, monkeypatch):
    """/scratch is 1.6 TB on one node of this cluster and 7.4 GB on another,
    and a dwd-scale pack spills ~17 GB. Filling it is how a run dies."""
    from meteopack import cli

    monkeypatch.delenv("METEOPACK_SPILL", raising=False)
    small = tmp_path / "small"
    types = {str(tmp_path / "data"): "nfs4", str(small): "xfs"}
    monkeypatch.setattr(cli, "_filesystem_type",
                        lambda p: types.get(str(p), "ext4"))
    monkeypatch.setattr(cli, "_LOCAL_SPILL_DIRS", (small,))
    monkeypatch.setattr(cli.shutil, "disk_usage",
                        lambda p: SimpleNamespace(total=1 << 30, used=0, free=1 << 20))
    assert cli._spill_root(tmp_path / "data") == tmp_path / "data" / "_spill"


def test_an_explicit_spill_location_is_obeyed(tmp_path, monkeypatch):
    from meteopack import cli

    target = tmp_path / "fast"
    monkeypatch.setenv("METEOPACK_SPILL", str(target))
    assert cli._spill_root(tmp_path / "data") == target
    assert target.is_dir()


def test_filesystem_type_reads_the_longest_matching_mount(tmp_path):
    from meteopack.cli import _filesystem_type

    # "/" always exists and is never a network filesystem on a test runner.
    assert _filesystem_type(Path("/")) is not None
    assert _filesystem_type(tmp_path) is not None


def test_pack_recovers_stations_the_live_catalogue_has_forgotten(tmp_path):
    """A provider lists the stations reporting *now*.

    Backfilling 2012 in 2026 therefore gets coordinates for the stations that
    still exist and nothing for the ones that have closed - half of DWD's 1,030
    and three quarters of DMI's 110 - and a station with no position silently
    drops out of anything spatial. A catalogue archived beside the data was
    written while they were alive.
    """
    import json

    from meteopack import cli

    raw = tmp_path / "raw" / "dwd" / "stations_10min"
    raw.mkdir(parents=True)
    (raw / "station_catalog.json").write_text(json.dumps({
        "00044": {"station_id": "00044", "lat": 52.1, "lon": 5.2,
                  "height": 2.0, "name": "De Bilt"},
        "00099": {"station_id": "00099", "lat": 51.0, "lon": 4.0,
                  "height": 10.0, "name": "Closed Long Ago"},
    }))

    class Ctx:
        cfg = _cfg()
        root = tmp_path
        raw_root = raw

        class src:
            @staticmethod
            def stations(cfg):
                # Only the station still operating today.
                return pl.DataFrame({"station_id": ["00044"], "lat": [52.1],
                                     "lon": [5.2], "height_m": [2.0],
                                     "name": ["De Bilt"]})

    n = cli._write_station_metadata(Ctx())
    assert n == 2, "the closed station must be recovered from the archive"
    back = points.read_stations(_cfg().store_id, tmp_path)
    assert set(back["station_id"].to_list()) == {"00044", "00099"}
    closed = back.filter(pl.col("station_id") == "00099")
    assert closed["lat"][0] == 51.0 and closed["lon"][0] == 4.0
    assert closed["name"][0] == "Closed Long Ago"


def test_pack_refuses_shards_rather_than_running_n_racing_copies(capsys, tmp_path):
    """--shard/--shards were accepted and ignored: an array of N tasks ran N full packs."""
    code = main(["pack", "dwd", "--shards", "4", "--shard", "1", "--data-root", str(tmp_path)])
    assert code == 2
    assert "does not support --shard" in capsys.readouterr().out


def test_a_failed_file_leaves_points_partitions_untouched(tmp_path, monkeypatch, capsys):
    """One unreadable file must not silently delete its rows from the store.

    Partitions are rewritten whole from the files that parsed. In review, one
    truncated DWD zip took a store from 1,287,539 rows to 1,208,339 under a
    manifest and README that said nothing was wrong.
    """
    from meteopack.core.store import points
    from meteopack.jobs import packing

    raw = tmp_path / "raw" / "dwd" / "stations_10min"
    raw.mkdir(parents=True)
    for name in ("good.zip", "truncated.zip"):
        (raw / name).write_bytes(b"x")
    calls = []

    def fake_run(tasks, fn, **kw):
        calls.append(fn.__name__)
        return [{"path": str(raw / "good.zip"), "parts": {"2012": "/nonexistent"}},
                {"path": str(raw / "truncated.zip"), "error": "BadZipFile: truncated"}]

    monkeypatch.setattr(packing, "run", fake_run)
    code = main(["pack", "dwd", "--data-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 1, out
    assert "store is unchanged" in out and "truncated.zip" in out
    assert calls == ["parse_points"], "no partition write may be attempted"
    assert not points.store_path(tmp_path, "dwd/stations_10min/v1").exists()
