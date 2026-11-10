"""The day-by-day KDP stream, end to end against a fake API serving synthetic files."""
from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from meteopack.core import kdp
from meteopack.core.store import gridded, manifest
from meteopack.jobs import stream_kdp
from meteopack.sources.radar import knmi_radar_3d as r3d
from test_knmi_radar_3d import (  # noqa: F401 - the fixture is used by name
    NZ,
    _tiny_grid,
    expected_codes,
    native_codes,
    tiny_cfg,
    write_file,
)

DAY1 = datetime(2024, 5, 7, tzinfo=UTC)
DAY2 = DAY1 + timedelta(days=1)
STEP = timedelta(minutes=5)


class FakeKey:
    path = Path("/fake/key")


class FakeClient:
    """Serves synthetic files from a directory, counting what a real client would spend."""

    def __init__(self, upstream: Path) -> None:
        self.upstream = upstream
        self.key = FakeKey()
        self.requests_made = 0
        self.downloads: list[str] = []
        self.fail: set[str] = set()
        self.limiter = kdp.QuotaLimiter(1000.0)

    def refresh_key(self) -> bool:
        return False

    def list_files(self, begin=None, end=None):
        self.requests_made += 1
        return [kdp.RemoteFile(p.name, p.stat().st_size)
                for p in sorted(self.upstream.iterdir()) if begin <= p.name <= end]

    def download(self, remote, dest):
        if dest.exists() and dest.stat().st_size == remote.size:
            return kdp.RawFile(path=dest, sha256="", n_bytes=remote.size,
                               unit_key=remote.filename, attempts=0)
        self.requests_made += 1
        if remote.filename in self.fail:
            raise kdp.FetchError(remote.filename, "HTTP 500", 500)
        self.downloads.append(remote.filename)
        shutil.copy(self.upstream / remote.filename, dest)
        return kdp.RawFile(path=dest, sha256="", n_bytes=remote.size,
                           unit_key=remote.filename, attempts=1)


def seed(when: datetime) -> int:
    return int(when.timestamp()) // 300 % 1000


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """Day 1 complete; day 2 lacks 00:10 and 00:20 and has one broken file at 01:00."""
    up = tmp_path / "upstream"
    up.mkdir()
    for day in (DAY1, DAY2):
        for when in r3d.day_slots(day, STEP):
            if day == DAY2 and when.hour == 0 and when.minute in (10, 20):
                continue
            broken = day == DAY2 and when.hour == 1 and when.minute == 0
            write_file(up, when, codes=native_codes(seed(when)),
                       z=0.25 * np.arange(NZ) if broken else None)
    return up


def run(cfg, root, client, **kw):
    lines: list[str] = []
    code = stream_kdp.run(cfg, root, DAY1, DAY2 + timedelta(days=1), client=client,
                          pack_workers=2, download_workers=4, log=lines.append, **kw)
    return code, lines


def test_two_days_stream_check_pack_and_clean_up(tmp_path: Path, upstream: Path):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    client = FakeClient(upstream)
    code, lines = run(cfg, root, client)
    assert code == stream_kdp.EXIT_DONE, lines

    s1 = stream_kdp.read_status(cfg, root, DAY1)
    s2 = stream_kdp.read_status(cfg, root, DAY2)
    assert s1["state"] == "done" and s1["packed"] == 288
    assert s2["state"] == "partial" and s2["packed"] == 288 - 3
    assert s2["missing_upstream"] == ["0010", "0020"]
    assert list(s2["schema_failed"]) == ["3D_RADAR_COMPOSITE_202405080100.nc"]
    assert "z levels" in s2["schema_failed"]["3D_RADAR_COMPOSITE_202405080100.nc"][0]
    assert s2["listed_not_packed"] == ["0100"]
    assert s1["timings"]["pack_s_per_file_mean"] is not None and s1["shard_bytes"] > 0

    # Raw files are gone; the one that failed its check is kept aside.
    raw = root / "raw/radar/knmi_radar/composite3d_5min"
    assert sorted(p.name for p in raw.rglob("*") if p.is_file()) == [
        "3D_RADAR_COMPOSITE_202405080100.nc"]
    assert (raw / "quarantine" / "3D_RADAR_COMPOSITE_202405080100.nc").exists()

    # The store: (time, z, y, x), north-up, every second level, flags right.
    ds = gridded.read(cfg.store_id, root, decode=False)
    assert ds["reflectivity_dbz"].dims == gridded.DIMS_3D
    assert ds.sizes["time"] == 2 * 288
    np.testing.assert_allclose(ds["z"].values, cfg.levels.values)
    when = DAY2 + timedelta(hours=13, minutes=35)
    row = int((when - DAY1) / STEP)
    want = expected_codes(native_codes(seed(when))[1::2], 0.0015, -30.00001)[:, ::-1, :]
    np.testing.assert_array_equal(ds["reflectivity_dbz"].values[row], want)
    missing = ds["is_missing_frame"].values
    corrupt = ds["is_corrupt_frame"].values
    assert missing.sum() == 2 and corrupt.sum() == 1
    assert missing[288 + 2] and missing[288 + 4] and corrupt[288 + 12]
    assert (ds["reflectivity_dbz"].values[288 + 2] == 255).all()   # missing frame code
    assert (ds["reflectivity_dbz"].values[288 + 12] == 255).all()  # corrupt: never packed
    layout = ds["source_layout_id"].values
    assert layout[288 + 12] == r3d.LAYOUT_CORRUPT and layout[0] == r3d.LAYOUT_V1

    m = manifest.read(gridded.store_path(root, cfg.store_id))
    assert m.n_records == 288 + 285
    assert m.extra["days"]["2024-05-08"]["state"] == "partial"
    assert m.extra["encoding"]["formula"] == "dBZ = 0.5 * code - 32"
    assert len(m.extra["schema_fingerprints"]) == 2         # the drift is on record
    assert any(line.startswith("SCHEMA: 2024-05-08") for line in lines)


def test_a_rerun_is_cheap_and_changes_nothing(tmp_path: Path, upstream: Path):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    client = FakeClient(upstream)
    run(cfg, root, client)
    before = gridded.read(cfg.store_id, root, decode=False)["reflectivity_dbz"].values.copy()

    again = FakeClient(upstream)
    code, _ = run(cfg, root, again)
    assert code == stream_kdp.EXIT_DONE
    # Day 1 is done and skipped; day 2 is partial: one listing, and only the
    # one unpackable file is fetched again.
    assert again.downloads == ["3D_RADAR_COMPOSITE_202405080100.nc"]
    assert again.requests_made == 2
    after = gridded.read(cfg.store_id, root, decode=False)["reflectivity_dbz"].values
    np.testing.assert_array_equal(before, after)


def test_a_failed_download_leaves_a_partial_day_that_the_next_run_completes(
        tmp_path: Path, upstream: Path):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    client = FakeClient(upstream)
    client.fail = {"3D_RADAR_COMPOSITE_202405071200.nc"}
    code, _ = run(cfg, root, client)
    assert code == stream_kdp.EXIT_DONE
    s1 = stream_kdp.read_status(cfg, root, DAY1)
    assert s1["state"] == "partial" and list(s1["failed_downloads"]) == [
        "3D_RADAR_COMPOSITE_202405071200.nc"]
    row = 12 * 12
    assert gridded.read(cfg.store_id, root, decode=False)["is_missing_frame"].values[row]

    retry = FakeClient(upstream)
    run(cfg, root, retry)
    assert "3D_RADAR_COMPOSITE_202405071200.nc" in retry.downloads
    assert stream_kdp.read_status(cfg, root, DAY1)["state"] == "done"
    ds = gridded.read(cfg.store_id, root, decode=False)
    assert not ds["is_missing_frame"].values[row]
    when = DAY1 + timedelta(hours=12)
    want = expected_codes(native_codes(seed(when))[1::2], 0.0015, -30.00001)[:, ::-1, :]
    np.testing.assert_array_equal(ds["reflectivity_dbz"].values[row], want)
    # Frames packed in the first run were not touched by the partial re-write.
    other = DAY1 + timedelta(hours=11, minutes=55)
    np.testing.assert_array_equal(
        ds["reflectivity_dbz"].values[row - 1],
        expected_codes(native_codes(seed(other))[1::2], 0.0015, -30.00001)[:, ::-1, :])


def test_a_time_budget_stops_between_days_with_exit_75(tmp_path: Path, upstream: Path):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    code, _ = run(cfg, root, FakeClient(upstream), max_days=1)
    assert stream_kdp.read_status(cfg, root, DAY2) is None
    code, _ = run(cfg, root, FakeClient(upstream), time_budget_s=1e-9)
    assert code == stream_kdp.EXIT_MORE
    code, _ = run(cfg, root, FakeClient(upstream))
    assert code == stream_kdp.EXIT_DONE
    assert json.loads(stream_kdp.status_path(cfg, root, DAY2).read_text())["packed"] == 285


def test_schema_failures_on_consecutive_days_stop_the_stream(tmp_path: Path):
    cfg = tiny_cfg()
    up = tmp_path / "up"
    up.mkdir()
    for day in (DAY1, DAY2):
        for when in r3d.day_slots(day, STEP)[:3]:
            write_file(up, when, z=0.25 * np.arange(NZ))       # every file drifted
    code, lines = run(cfg, tmp_path / "root", FakeClient(up))
    assert code == stream_kdp.EXIT_SCHEMA
    assert any("consecutive days" in line for line in lines)
