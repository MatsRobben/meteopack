

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
    from tests.test_gridded_store import FRAMES_PER_DAY, NX, NY, TINY, make_cfg, make_timeline

    from meteopack.core import grids
    from meteopack.core.store import gridded
    from meteopack.jobs import packing

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
    from tests.test_gridded_store import (
        FRAMES_PER_DAY,
        NX,
        NY,
        TINY,
        make_cfg,
        make_timeline,
    )

    from meteopack.core import grids
    from meteopack.core.store import gridded
    from meteopack.jobs import packing

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
