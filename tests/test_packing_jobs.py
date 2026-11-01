

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
