"""`run_pipeline`: write each partition as soon as its parts are parsed - and
never write one that is incomplete.

The worker functions are fakes defined here (spawned workers import this
module), so the bookkeeping is tested without any real source.
"""
from __future__ import annotations

from meteopack.jobs import packing


def fake_parse(task):
    if task.part == "bad":
        raise RuntimeError("unreadable")
    label = "2024/09" if task.part == "stray" else task.partitions[0]
    return {"path": task.raw_path, "part": task.part, "rows": 1,
            "parts": {label: task.part.encode()}, "sizes": {label: 10}, "records": []}


def fake_write(task):
    return {"label": task.label, "rows": len(task.files),
            "files": [f.decode() for f in task.files]}


def _task(part, label):
    return packing.PackTask(source="x", product="y", raw_path=f"/{part}", root="/r",
                            part=part, partitions=(label,), spill_inline=True)


def _run(tasks, **kw):
    return packing.run_pipeline(
        tasks, workers=2, make_partition=lambda label, files, n: packing.PartitionTask(
            label=label, files=files, source="x", product="y", root="/r", n_bytes=n),
        parse_fn=fake_parse, write_fn=fake_write, limit=1 << 34, **kw)


def test_a_partition_with_a_failed_part_is_not_rewritten():
    res = _run([_task("a", "2024/06"), _task("bad", "2024/06"), _task("c", "2024/07")])
    written = {r["label"]: r for r in res.written}
    assert set(written) == {"2024/07"}
    assert "not rewritten" in res.skipped["2024/06"]
    assert [e["part"] for e in res.errors] == ["bad"]


def test_pieces_reach_the_writer_in_task_order_not_completion_order():
    parts = [str(i) for i in range(12)]
    res = _run([_task(p, "2024/06") for p in parts])
    assert res.written[0]["files"] == parts


def test_rows_outside_the_window_are_refused_not_written():
    res = _run([_task("a", "2024/06"), _task("stray", "2024/06")], allowed={"2024/06"})
    assert "outside the window" in res.errors[0]["error"]
    # 2024/06 is complete - the stray task had no rows of its own month - and
    # is written; the rows for 2024/09 are an error, never a partial 2024/09.
    assert {r["label"] for r in res.written} == {"2024/06"}


def test_backpressure_still_finishes():
    res = _run([_task(str(i), f"2024/{i % 3 + 1:02d}") for i in range(9)], max_spill=1)
    assert sorted(r["label"] for r in res.written) == ["2024/01", "2024/02", "2024/03"]


def test_partial_is_opt_in():
    res = _run([_task("a", "2024/06"), _task("bad", "2024/06")], allow_partial=True)
    assert [r["label"] for r in res.written] == ["2024/06"]
