

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
