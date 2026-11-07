

def _suicide(task):
    """A task that kills its own worker, the way the OOM killer does."""
    import os
    import signal

    os.kill(os.getpid(), signal.SIGKILL)


def _ok(task):
    return {"path": task.raw_path, "frames": 1}


def test_a_dead_worker_fails_the_run_instead_of_hanging():
    """Observed for real: a SEVIRI pack lost all six workers to the OOM killer
    and the parent sat in futex_wait for 1 h 53 m with no children, 0% CPU and
    no output. A cluster job that hangs silently is worse than one that fails,
    because nothing notices.
    """
    from meteopack.jobs import packing

    tasks = [packing.PackTask(source="s", product="p", raw_path=f"f{i}", root="/tmp")
             for i in range(4)]
    results = packing.run(tasks, _suicide, workers=2, stall_timeout=60)

    assert len(results) == len(tasks), "every task must be accounted for"
    assert all(r.get("error") for r in results), results
    assert any("worker died" in r["error"] for r in results), results


def test_a_stalled_pool_is_reported_rather_than_waited_on():
    from meteopack.jobs import packing

    tasks = [packing.PackTask(source="s", product="p", raw_path=f"f{i}", root="/tmp")
             for i in range(3)]
    results = packing.run(tasks, _sleep_forever, workers=2, stall_timeout=2)
    assert len(results) == len(tasks)
    assert all("not making progress" in r.get("error", "") for r in results), results


def _sleep_forever(task):
    import time

    time.sleep(600)


def _half_a_second_per_file(task):
    import time

    time.sleep(0.5 * len(task.paths))
    return {"path": task.raw_path, "frames": len(task.paths)}


def test_a_large_task_that_is_working_is_not_killed_as_a_stall():
    """The stall limit is per file in the largest task in flight.

    A flat limit killed two ~2,000-tar echotop tasks part-way: they were
    working, but a task reports nothing until it ends. The store was left
    half-written and those tasks' frames were missing from the total.
    """
    from meteopack.jobs import packing

    big = packing.PackTask(source="s", product="p", raw_path="f0", root="/tmp",
                           also=tuple(f"f0.{k}" for k in range(5)))   # 6 files, ~3 s
    small = packing.PackTask(source="s", product="p", raw_path="f1", root="/tmp")
    results = packing.run([big, small], _half_a_second_per_file, workers=2,
                          stall_timeout=1.0)
    assert not any(r.get("error") for r in results), results
    assert sum(r["frames"] for r in results) == 7


def test_one_bad_file_does_not_abort_the_rest_of_its_task(tmp_path, monkeypatch):
    """A task holds every file that shares a shard; each file fails alone.

    write_gridded parsed a task's files in a loop with no per-file handling, so
    the first unreadable one silently took everything after it with it.
    """
    import numpy as np
    import xarray as xr

    from meteopack.core import grids
    from meteopack.core.store import gridded
    from meteopack.jobs import packing
    from test_gridded_store import FRAMES_PER_DAY, NX, NY, TINY, make_cfg, make_timeline

    monkeypatch.setattr(grids, "get", lambda grid_id, path=None: TINY)
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)

    class Source:
        def parse(self, raw, _cfg):
            day = {"day0": 0, "day2": 2}.get(raw.path.name)
            if day is None:
                raise OSError("truncated tar")
            t = timeline.values()[day * FRAMES_PER_DAY:(day + 1) * FRAMES_PER_DAY]
            return xr.Dataset({"rain": (("time", "y", "x"), np.ones((len(t), NY, NX), "float32"))},
                              coords={"time": t})

    for name in ("day0", "bad", "day2"):
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(packing, "_load", lambda task: (Source(), cfg))
    task = packing.PackTask(source="s", product="p", raw_path=str(tmp_path / "day0"),
                            root=str(tmp_path), also=(str(tmp_path / "bad"), str(tmp_path / "day2")))
    out = packing.write_gridded(task)
    assert out["frames"] == 2 * FRAMES_PER_DAY, out["file_errors"]  # day2 still packed
    assert [e["path"] for e in out["file_errors"]] == [str(tmp_path / "bad")]


def _tiny_pack(tmp_path, monkeypatch, names):
    """write_gridded over `names` (day0, day1, ...) on the tiny test grid."""
    import numpy as np
    import xarray as xr

    from meteopack.core import grids
    from meteopack.core.store import gridded
    from meteopack.jobs import packing
    from test_gridded_store import (
        FRAMES_PER_DAY,
        NX,
        NY,
        TINY,
        make_cfg,
        make_timeline,
    )

    monkeypatch.setattr(grids, "get", lambda grid_id, path=None: TINY)
    cfg, timeline = make_cfg(), make_timeline()
    if not gridded.zarr_path(tmp_path, cfg.store_id).exists():
        gridded.initialise(cfg, tmp_path, timeline)

    class Source:
        def parse(self, raw, _cfg):
            day = int(raw.path.name[3:])
            t = timeline.values()[day * FRAMES_PER_DAY:(day + 1) * FRAMES_PER_DAY]
            return xr.Dataset({"rain": (("time", "y", "x"), np.ones((len(t), NY, NX), "float32"))},
                              coords={"time": t})

    for name in names:
        (tmp_path / name).write_bytes(b"x")
    monkeypatch.setattr(packing, "_load", lambda task: (Source(), cfg))
    task = packing.PackTask(source="s", product="p", raw_path=str(tmp_path / names[0]),
                            root=str(tmp_path), also=tuple(str(tmp_path / n) for n in names[1:]))
    out = packing.write_gridded(task)
    gridded.write_provenance_table(tmp_path, cfg.store_id, out["records"])
    return cfg, out


def test_every_packed_frame_names_its_raw_file_in_the_provenance_table(tmp_path, monkeypatch):
    """source_record_id was -1 on every frame, and the table it pointed at did not exist."""
    import numpy as np
    import polars as pl

    from meteopack.core.store import gridded

    cfg, _ = _tiny_pack(tmp_path, monkeypatch, ["day0", "day1"])
    table = pl.read_parquet(gridded.store_path(tmp_path, cfg.store_id) / gridded.PROVENANCE_TABLE)
    assert sorted(table["file"].to_list()) == ["day0", "day1"]
    ds = gridded.read(cfg.store_id, tmp_path)
    ids = ds["source_record_id"].values[~ds["is_missing_frame"].values]
    assert (ids >= 0).all() and set(np.unique(ids)) == set(table["source_record_id"].to_list())
    assert gridded.record_id("day0") == table.filter(pl.col("file") == "day0")[
        "source_record_id"].item()

    # Re-packing a file replaces its row rather than adding one.
    _tiny_pack(tmp_path, monkeypatch, ["day1", "day2"])
    table = pl.read_parquet(gridded.store_path(tmp_path, cfg.store_id) / gridded.PROVENANCE_TABLE)
    assert sorted(table["file"].to_list()) == ["day0", "day1", "day2"]


def test_record_ids_are_stable_distinct_and_non_negative():
    from meteopack.core.store import gridded

    names = [f"RAD25_OPER_R___TARETH__L2__2008{d:04d}T000000_x.tar" for d in range(5000)]
    ids = [gridded.record_id(n) for n in names]
    assert len(set(ids)) == len(ids) and min(ids) >= 0
    assert gridded.record_id(names[0]) == ids[0]          # same name, same id, every run


def test_validate_flags_a_frame_whose_file_is_not_in_the_table(tmp_path, monkeypatch):
    import polars as pl

    from meteopack.core.store import gridded, manifest
    from meteopack.core.validate import validate

    cfg, _ = _tiny_pack(tmp_path, monkeypatch, ["day0", "day1"])
    manifest.write(gridded.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "packed", "gridded", cfg))
    report = validate(cfg, tmp_path)
    assert report.ok, report.text()

    path = gridded.store_path(tmp_path, cfg.store_id) / gridded.PROVENANCE_TABLE
    pl.read_parquet(path).filter(pl.col("file") == "day0").write_parquet(path)
    report = validate(cfg, tmp_path)
    assert not report.ok and "provenance table" in report.text()


def test_a_dead_partition_writer_is_reported_by_its_label():
    """A dwd pack lost a partition writer to the OOM killer and then crashed in
    the error path itself: PartitionTask has no raw_path (2026-11-05)."""
    from meteopack.jobs import packing

    tasks = [packing.PartitionTask(label=f"year={2020 + i}", files=(), source="s",
                                   product="p", root="/tmp") for i in range(3)]
    results = packing.run(tasks, _suicide, workers=2, stall_timeout=60)
    assert len(results) == len(tasks)
    assert {r["path"] for r in results} == {t.label for t in tasks}
    assert all("worker died" in r["error"] or "not attempted" in r["error"] for r in results)


def test_partitions_are_written_only_as_many_at_once_as_memory_allows():
    """Seventeen dwd partitions written together reached 127 GB in a 110 GB job."""
    from meteopack.jobs import packing

    gib = 1 << 30
    tasks = [packing.PartitionTask(label=str(i), files=(), source="s", product="p",
                                   root="/tmp", n_bytes=2 * gib) for i in range(17)]
    # 7x expansion -> 14 GiB each; 70% of 110 GiB = 77 GiB -> 5 at once
    assert packing.partition_workers(tasks, 24, limit=110 * gib) == 5
    assert packing.partition_workers(tasks, 4, limit=110 * gib) == 4
    # one partition bigger than the budget still gets a worker, and fails on its own
    huge = [packing.PartitionTask(label="x", files=(), source="s", product="p",
                                  root="/tmp", n_bytes=100 * gib)]
    assert packing.partition_workers(huge, 24, limit=110 * gib) == 1
    assert packing.memory_limit_bytes() is None or packing.memory_limit_bytes() > 0


def test_write_partition_merges_real_spill_files(tmp_path):
    """The partition writer had no test of its own: a bad keyword to read_ipc
    failed every partition of the first real pack after it was edited."""
    from datetime import UTC, datetime

    import polars as pl

    from meteopack.core.store import points
    from meteopack.jobs import packing

    times = [datetime(2020, 6, 1, 0, 10 * i, tzinfo=UTC) for i in range(3)]
    frame = pl.DataFrame({"time": times, "station_id": ["6447"] * 3,
                          "air_temperature": [10.0, 11.0, 12.0]})
    frame = frame.with_columns(pl.col("time").dt.cast_time_unit("us"))
    spills = []
    for i, part in enumerate((frame.select("time", "station_id", "air_temperature"),
                              frame.select("time", "station_id"))):
        f = tmp_path / f"{i}.arrow"
        part.write_ipc(f)
        spills.append(str(f))
    task = packing.PartitionTask(label="2020", files=tuple(spills), source="belgium",
                                 product="aws_10min", root=str(tmp_path / "root"))
    out = packing.write_partition(task)
    assert out["rows"] == 3 and out["merged"] == 6, out   # both rows of each pair count
    assert out["peak_rss"] > 0
    back = points.read("belgium/aws_10min/v1", tmp_path / "root")
    assert back.sort("time")["air_temperature"].to_list() == [10.0, 11.0, 12.0]
