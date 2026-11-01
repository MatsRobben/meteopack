"""The catalog's job is to survive many writers and still answer honestly."""
from __future__ import annotations

import multiprocessing as mp
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from meteopack.core import catalog as catalog_mod
from meteopack.core.catalog import Catalog, job_key
from meteopack.sources.base import RawFile


def _raw(key: str, n: int = 10) -> RawFile:
    return RawFile(Path(f"/raw/{key}.zip"), f"{n:064x}", n, key)


def test_empty_catalog_answers_with_typed_empty_tables(tmp_path: Path):
    cat = Catalog(tmp_path)
    for frame in (cat.files(), cat.stores(), cat.jobs()):
        assert len(frame) == 0
    assert cat.jobs().columns[:2] == ["job_key", "source"]
    assert cat.pending("fetch", ["a", "b"]) == ["a", "b"]
    assert cat.is_done("a") is False


def test_is_done_and_pending_round_trip(tmp_path: Path):
    cat = Catalog(tmp_path)
    keys = [job_key("fetch", f"u{i}") for i in range(4)]
    cat.mark(keys[0], "done", stage="fetch")
    cat.mark(keys[1], "failed", stage="fetch", error="http 500")
    cat.mark(keys[2], "pending", stage="fetch")

    assert cat.is_done(keys[0]) is True
    assert cat.is_done(keys[1]) is False
    assert cat.pending("fetch", keys) == keys[1:]
    # a done row for another stage must not satisfy this stage
    cat.mark(job_key("parse", "u1"), "done", stage="parse")
    assert cat.pending("fetch", keys) == keys[1:]
    assert cat.pending("fetch", []) == []


def test_pending_preserves_order_and_drops_repeats(tmp_path: Path):
    cat = Catalog(tmp_path)
    cat.mark("fetch:b", "done", stage="fetch")
    keys = ["fetch:c", "fetch:a", "fetch:b", "fetch:a"]
    assert cat.pending("fetch", keys) == ["fetch:c", "fetch:a"]


def test_latest_row_per_key_wins_but_history_is_kept(tmp_path: Path):
    cat = Catalog(tmp_path)
    key = job_key("fetch", "u0")
    cat.mark(key, "pending", stage="fetch")
    cat.mark(key, "failed", stage="fetch", error="timeout")
    cat.mark(key, "done", stage="fetch")

    jobs = cat.jobs()
    assert len(jobs) == 1
    assert jobs["status"][0] == "done"
    assert jobs["attempts"][0] == 3
    assert len(cat.query("SELECT * FROM jobs_all")) == 3
    assert cat.query("SELECT error FROM jobs_all WHERE error IS NOT NULL")["error"].to_list() == [
        "timeout"
    ]


def test_record_file_keeps_the_checksum_from_fetch(tmp_path: Path):
    cat = Catalog(tmp_path)
    valid = datetime(2012, 1, 1, tzinfo=UTC)
    cat.record_file(_raw("dwd/10min/00044/2012", 4096), source="dwd", product="10min",
                    url="https://example.invalid/a.zip", valid_time=valid)
    files = cat.files()
    assert files["sha256"][0] == f"{4096:064x}"
    assert files["n_bytes"][0] == 4096
    # stored naive UTC, so a reader's session timezone cannot shift it
    assert files["valid_time"][0] == datetime(2012, 1, 1)

    # same unit fetched again -> still one row, the newer one
    cat.record_file(_raw("dwd/10min/00044/2012", 8192), source="dwd", product="10min",
                    url="https://example.invalid/a.zip")
    assert len(cat.files()) == 1
    assert cat.files()["n_bytes"][0] == 8192


def test_record_store_is_what_a_gc_would_read(tmp_path: Path):
    cat = Catalog(tmp_path)
    cat.record_store(store_id="dwd/10min/v1", layer="packed", kind="points",
                     path=str(tmp_path / "packed"), version=1, n_records=1234,
                     time_min=datetime(2012, 1, 1, tzinfo=UTC),
                     time_max=datetime(2013, 1, 1, tzinfo=UTC),
                     config_hash="cafe", code_version="abc1234")
    row = cat.stores().row(0, named=True)
    assert row["n_records"] == 1234
    assert (row["config_hash"], row["code_version"]) == ("cafe", "abc1234")
    assert row["time_min"] == datetime(2012, 1, 1)


def test_query_is_a_real_escape_hatch(tmp_path: Path):
    cat = Catalog(tmp_path)
    for i in range(5):
        cat.record_file(_raw(f"u{i}", i), source="dwd", product="10min", url="x")
    total = cat.query("SELECT sum(n_bytes) AS n FROM files")["n"][0]
    assert total == sum(range(5))


# --------------------------------------------------------------------- concurrency

def _thread_writer(root: Path, worker: int, n: int) -> None:
    cat = Catalog(root)  # each writer owns its own shard
    for i in range(n):
        cat.mark(job_key("fetch", f"w{worker}/u{i}"), "done", stage="fetch", source="dwd")


def test_concurrent_thread_appends_lose_nothing(tmp_path: Path):
    from concurrent.futures import ThreadPoolExecutor

    workers, per_worker = 8, 25
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda w: _thread_writer(tmp_path, w, per_worker), range(workers)))

    cat = Catalog(tmp_path)
    jobs = cat.jobs()
    assert len(jobs) == workers * per_worker
    assert len(cat.query("SELECT * FROM jobs_all")) == workers * per_worker
    expected = {job_key("fetch", f"w{w}/u{i}") for w in range(workers) for i in range(per_worker)}
    assert set(jobs["job_key"].to_list()) == expected
    assert cat.pending("fetch", sorted(expected)) == []
    # one shard per writer, not one file per row
    assert len(list((tmp_path / catalog_mod.CATALOG_DIRNAME / "jobs").glob("*.parquet"))) == workers


def _process_writer(root_str: str, worker: int, n: int) -> None:
    cat = Catalog(Path(root_str))
    for i in range(n):
        cat.record_file(_raw(f"w{worker}/u{i}", worker * 1000 + i), source="dwd",
                        product="10min", url=f"https://example.invalid/w{worker}/u{i}")


def test_concurrent_process_appends_lose_nothing(tmp_path: Path):
    ctx = mp.get_context("spawn")
    workers, per_worker = 4, 20
    procs = [ctx.Process(target=_process_writer, args=(str(tmp_path), w, per_worker))
             for w in range(workers)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    cat = Catalog(tmp_path)
    files = cat.files()
    assert len(files) == workers * per_worker
    assert files["n_bytes"].null_count() == 0
    assert set(files["unit_key"].to_list()) == {
        f"w{w}/u{i}" for w in range(workers) for i in range(per_worker)
    }


def test_a_second_writer_can_overwrite_an_earlier_verdict(tmp_path: Path):
    """A retry on another node must win over the failure recorded earlier."""
    first = Catalog(tmp_path)
    key = job_key("fetch", "u0")
    first.mark(key, "failed", stage="fetch", error="timeout")
    assert first.is_done(key) is False

    second = Catalog(tmp_path)
    second.mark(key, "done", stage="fetch", updated_at=datetime.now(UTC) + timedelta(seconds=1))
    assert Catalog(tmp_path).is_done(key) is True


def test_buffered_writes_are_flushed_before_any_read(tmp_path: Path):
    cat = Catalog(tmp_path, autoflush=False)
    cat.mark("fetch:u0", "done", stage="fetch")
    assert not list((tmp_path / catalog_mod.CATALOG_DIRNAME / "jobs").glob("*.parquet"))
    assert cat.is_done("fetch:u0") is True  # read flushes first
    assert list((tmp_path / catalog_mod.CATALOG_DIRNAME / "jobs").glob("*.parquet"))


def test_writers_never_share_a_shard(tmp_path: Path):
    a, b = Catalog(tmp_path), Catalog(tmp_path)
    assert a.shard_path("jobs") != b.shard_path("jobs")


@pytest.mark.parametrize("table", ["files", "stores", "jobs"])
def test_temp_files_are_invisible_to_readers(tmp_path: Path, table: str):
    cat = Catalog(tmp_path)
    cat.mark("fetch:u0", "done", stage="fetch")
    (cat.root / table / ".half-written.parquet.deadbeef.tmp").write_bytes(b"not parquet")
    cat.jobs()  # must not try to read the temp file


def test_mark_without_stage_still_counts_as_done(tmp_path):
    """A unit marked done must never be re-planned.

    `pending()` filters on the `stage` column, so a caller that passes the key
    but omits `stage=` used to leave it empty and the unit looked pending
    forever — meaning every run re-fetched it against a rate-limited server.
    """
    from meteopack.core.catalog import Catalog, job_key, split_job_key

    cat = Catalog(tmp_path)
    key = job_key("fetch", "dwd/00044/2012")
    cat.mark(key, "done")                      # note: no stage= passed
    assert cat.pending("fetch", [key]) == []
    assert cat.is_done(key)
    row = cat.jobs().row(0, named=True)
    assert row["stage"] == "fetch" and row["unit_key"] == "dwd/00044/2012"
    assert split_job_key(key) == ("fetch", "dwd/00044/2012")


def test_explicit_stage_still_wins(tmp_path):
    from meteopack.core.catalog import Catalog

    cat = Catalog(tmp_path)
    cat.mark("fetch:x", "done", stage="override", unit_key="u")
    assert cat.jobs().row(0, named=True)["stage"] == "override"
