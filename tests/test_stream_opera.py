"""The OPERA stream end to end: a fake bucket of synthetic ODIM files across the cutover."""
from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from meteopack.core.s3 import RemoteFile, _Pacer
from meteopack.core.store import gridded, manifest
from meteopack.jobs import stream_kdp
from meteopack.sources.base import RawFile
from meteopack.sources.radar import opera
from test_opera import (  # noqa: F401 - the fixture is used by name
    _tiny_grid,
    expected_codes,
    rates,
    tiny_cfg,
    write_odim,
)

DAY1 = datetime(2024, 7, 5, tzinfo=UTC)      # the cutover day
DAY2 = DAY1 + timedelta(days=1)
STEP = timedelta(minutes=15)
CUTOVER = datetime(2024, 7, 5, 6, 0, tzinfo=UTC)


class FakeBucket:
    """Serves a directory as an anonymous S3 bucket, counting requests."""

    def __init__(self, upstream: Path) -> None:
        self.upstream = upstream
        self.key = type("K", (), {"path": "anonymous"})()
        self.limiter = _Pacer(0)
        self.requests_made = 0
        self.downloads: list[str] = []
        self.fail: set[str] = set()
        self.listed_prefixes: list[str] = []

    def refresh_key(self) -> bool:
        return False

    def list_prefix(self, prefix: str):
        self.requests_made += 1
        self.listed_prefixes.append(prefix)
        day, _, start = prefix.partition("/OPERA/COMP/")
        out = []
        for p in sorted(self.upstream.iterdir()):
            stamp = p.name.split("@")[1] if "@" in p.name else ""
            if p.name.startswith(start) and stamp[:8] == day.replace("/", ""):
                out.append(RemoteFile(p.name, p.stat().st_size, key=f"{day}/OPERA/COMP/{p.name}"))
        return out

    def download(self, remote, dest):
        if dest.exists() and dest.stat().st_size == remote.size:
            return RawFile(path=dest, sha256="", n_bytes=remote.size,
                           unit_key=remote.filename, attempts=0)
        self.requests_made += 1
        if remote.filename in self.fail:
            raise RuntimeError(f"HTTP 500 for {remote.filename}")
        self.downloads.append(remote.filename)
        shutil.copy(self.upstream / remote.filename, dest)
        return RawFile(path=dest, sha256="", n_bytes=remote.size,
                       unit_key=remote.filename, attempts=1)


def seed(when: datetime) -> int:
    return int(when.timestamp()) // 900 % 1000


@pytest.fixture
def upstream(tmp_path: Path) -> Path:
    """Day 1: ODYSSEY to 06:00 and NIMBUS from 05:30 (the overlap), plus an ACRR
    file and a stray. Day 2: NIMBUS, lacking 03:00 and 03:15, one broken file at
    10:00 and one without Den Helder at 12:00."""
    up = tmp_path / "upstream"
    up.mkdir()
    for when in opera.day_slots(DAY1, STEP):
        if when <= CUTOVER:
            write_odim(up, when, opera.ODYSSEY, rates(seed(when)))
        if when >= CUTOVER - timedelta(minutes=30):
            # NIMBUS files in the overlap hold different values: if one were
            # packed instead of the ODYSSEY file, the codes would not match.
            write_odim(up, when, opera.NIMBUS, rates(seed(when) + 500))
    (up / "OPERA@20240705T1200@0@ACRR.h5").write_bytes(b"not a rate")
    for when in opera.day_slots(DAY2, STEP):
        if when.hour == 3 and when.minute in (0, 15):
            continue
        kw = {}
        if when.hour == 10 and when.minute == 0:
            kw = {"gain": 0.5}                                  # drift: never packed
        if when.hour == 12 and when.minute == 0:
            kw = {"nodes": ["nlhrw", "behel"]}
        write_odim(up, when, opera.NIMBUS, rates(seed(when) + 500), **kw)
    return up


def run(cfg, root, client, **kw):
    lines: list[str] = []
    code = stream_kdp.run(cfg, root, DAY1, DAY2 + timedelta(days=1), client=client,
                          pack_workers=2, download_workers=4, log=lines.append, **kw)
    return code, lines


def row_of(when: datetime, cfg) -> int:
    origin = datetime.fromisoformat(cfg.extra["timeline_origin"])
    return int((when - origin) / STEP)


def test_two_days_across_the_cutover(tmp_path: Path, upstream: Path):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    client = FakeBucket(upstream)
    code, lines = run(cfg, root, client)
    assert code == stream_kdp.EXIT_DONE, lines
    assert client.listed_prefixes == ["2024/07/05/OPERA/COMP/OPERA@",
                                      "2024/07/06/OPERA/COMP/OPERA@"]
    # The overlap's NIMBUS files and the ACRR file are never downloaded.
    assert not any(n.endswith("@RATE.h5") and n < "OPERA@20240705T0615"
                   for n in client.downloads)
    assert not any("ACRR" in n for n in client.downloads)

    s1 = stream_kdp.read_status(cfg, root, DAY1)
    s2 = stream_kdp.read_status(cfg, root, DAY2)
    assert s1["state"] == "done" and s1["packed"] == 96 and s1["listed"] == 96
    assert s2["state"] == "partial" and s2["packed"] == 96 - 3
    assert s2["missing_upstream"] == ["0300", "0315"]
    assert list(s2["schema_failed"]) == ["OPERA@20240706T1000@0@RATE.h5"]
    assert s2["listed_not_packed"] == ["1000"]

    raw = root / "raw/radar/opera/rate_15min"
    assert sorted(p.name for p in raw.rglob("*") if p.is_file()) == [
        "OPERA@20240706T1000@0@RATE.h5"]
    assert (raw / "quarantine" / "OPERA@20240706T1000@0@RATE.h5").exists()

    ds = gridded.read(cfg.store_id, root, decode=False,
                      time_range=(DAY1, DAY2 + timedelta(days=1)))
    assert ds.sizes["time"] == 2 * 96
    t = ds["time"].values
    assert np.all(np.diff(t) == np.timedelta64(15, "m")) and len(set(t.tolist())) == 192
    codes = ds["rain_rate_mmh"].values
    chain = ds["chain"].values
    # Cutover: 06:00 is the last ODYSSEY frame, 06:15 the first NIMBUS.
    i0600 = 24
    assert (chain[:i0600 + 1] == opera.CHAIN_CODES[opera.ODYSSEY]).all()
    assert (chain[i0600 + 1:96] == opera.CHAIN_CODES[opera.NIMBUS]).all()
    layout = ds["source_layout_id"].values
    assert layout[i0600] == opera.LAYOUT_ODYSSEY and layout[i0600 + 1] == opera.LAYOUT_NIMBUS
    for i in (0, i0600 - 2, i0600):                 # ODYSSEY, also in the overlap
        when = DAY1 + i * STEP
        np.testing.assert_array_equal(codes[i], expected_codes(rates(seed(when))))
    for i in (i0600 + 1, 95):
        when = DAY1 + i * STEP
        np.testing.assert_array_equal(codes[i], expected_codes(rates(seed(when) + 500)))

    missing = ds["is_missing_frame"].values
    corrupt = ds["is_corrupt_frame"].values
    assert missing.sum() == 2 and missing[96 + 12] and missing[96 + 13]
    assert corrupt.sum() == 1 and corrupt[96 + 40]
    for i in (96 + 12, 96 + 40):
        assert (codes[i] == 65534).all() and chain[i] == 0 and ds["n_radars"].values[i] == -1
        assert ds["has_nldhl"].values[i] == -1
    assert ds["n_radars"].values[96 + 48] == 2
    others = ~missing & ~corrupt
    others[96 + 48] = False
    assert (ds["n_radars"].values[others] == 4).all()
    assert ds["has_nldhl"].values[96 + 48] == 0 and ds["has_nlhrw"].values[96 + 48] == 1
    assert ds["has_nldhl"].values[0] == 1

    prov = pl.read_parquet(gridded.store_path(root, cfg.store_id) / "provenance.parquet")
    assert prov.height == 96 + 94
    assert set(prov["chain"].unique().to_list()) == {"ODYSSEY", "NIMBUS"}
    assert not prov.filter(pl.col("filename") == "OPERA@20240706T1000@0@RATE.h5")[
        "packed"].item()
    assert prov.filter(pl.col("chain") == "ODYSSEY").height == 25
    assert prov.filter(pl.col("filename") == "OPERA@20240705T0000@0@QIND_RATE.h5")[
        "n_clipped"].item() == 1

    m = manifest.read(gridded.store_path(root, cfg.store_id))
    assert m.extra["encoding"]["formula"] == "mm/h = 0.01 * code"
    assert m.extra["window"] == {"row0": 5, "row1": 17, "col0": 4, "col1": 14}
    import zarr

    g = zarr.open_group(gridded.zarr_path(root, cfg.store_id), mode="r")
    assert g.attrs["opera_window"]["row0"] == 5
    assert g.attrs["crs"]["grid_mapping_name"] == "lambert_azimuthal_equal_area"
    assert g["chain"].attrs["flag_meanings"] == "no_frame ODYSSEY NIMBUS"
    # The time axis starts at the configured origin, years before the first day.
    assert str(g.attrs["time_start"]).startswith("2013-01-01")


def test_a_rerun_is_cheap_and_changes_nothing(tmp_path: Path, upstream: Path):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    run(cfg, root, FakeBucket(upstream))
    window = (DAY1, DAY2 + timedelta(days=1))
    before = gridded.read(cfg.store_id, root, decode=False, time_range=window)
    again = FakeBucket(upstream)
    code, _ = run(cfg, root, again)
    assert code == stream_kdp.EXIT_DONE
    assert again.downloads == ["OPERA@20240706T1000@0@RATE.h5"]   # the broken one only
    after = gridded.read(cfg.store_id, root, decode=False, time_range=window)
    for name in ("rain_rate_mmh", "chain", "n_radars", "is_missing_frame"):
        np.testing.assert_array_equal(before[name].values, after[name].values)


def test_a_day_killed_while_writing_is_not_done_and_the_rerun_completes_it(
        tmp_path: Path, upstream: Path, monkeypatch: pytest.MonkeyPatch):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    real = gridded.write
    calls = {"n": 0}

    def dies_on_day_two(ds, *a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("killed")
        return real(ds, *a, **k)

    monkeypatch.setattr(gridded, "write", dies_on_day_two)
    with pytest.raises(KeyboardInterrupt):
        run(cfg, root, FakeBucket(upstream))
    assert stream_kdp.read_status(cfg, root, DAY1)["state"] == "done"
    assert stream_kdp.read_status(cfg, root, DAY2) is None
    monkeypatch.setattr(gridded, "write", real)

    again = FakeBucket(upstream)
    code, _ = run(cfg, root, again)
    assert code == stream_kdp.EXIT_DONE
    # Day 1 is skipped; day 2's files were already on disk and are not
    # downloaded again (except the quarantined one, which never got there).
    assert again.listed_prefixes == ["2024/07/06/OPERA/COMP/OPERA@"]
    assert len(again.downloads) <= 1
    s2 = stream_kdp.read_status(cfg, root, DAY2)
    assert s2["state"] == "partial" and s2["packed"] == 93


def test_a_failed_download_leaves_the_frame_missing_until_a_rerun(tmp_path, upstream):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    client = FakeBucket(upstream)
    client.fail = {"OPERA@20240705T1200@0@RATE.h5"}
    run(cfg, root, client)
    s1 = stream_kdp.read_status(cfg, root, DAY1)
    assert s1["state"] == "partial" and list(s1["failed_downloads"]) == [
        "OPERA@20240705T1200@0@RATE.h5"]
    row = row_of(DAY1 + timedelta(hours=12), cfg)
    import zarr

    g = zarr.open_group(gridded.zarr_path(root, cfg.store_id), mode="r")
    assert g["is_missing_frame"][row] and g["chain"][row] == 0
    run(cfg, root, FakeBucket(upstream))
    assert stream_kdp.read_status(cfg, root, DAY1)["state"] == "done"
    g = zarr.open_group(gridded.zarr_path(root, cfg.store_id), mode="r")
    assert not g["is_missing_frame"][row] and g["chain"][row] == 2


def test_a_configured_fallback_day_is_packed_from_odyssey(tmp_path: Path):
    """A day on extra.opera.odyssey_fallback_days takes the ODYSSEY files (NIMBUS has
    none) and records chain = ODYSSEY; the next day is NIMBUS again."""
    cfg = tiny_cfg()
    extra = dict(cfg.extra)
    extra["opera"] = {**extra["opera"], "odyssey_fallback_days": [f"{DAY2:%Y-%m-%d}"]}
    cfg = cfg.model_copy(update={"extra": extra})
    up = tmp_path / "up"
    up.mkdir()
    day3 = DAY2 + timedelta(days=1)
    for when in opera.day_slots(DAY2, STEP):
        write_odim(up, when, opera.ODYSSEY, rates(seed(when)))
    for when in opera.day_slots(day3, STEP)[:4]:
        write_odim(up, when, opera.NIMBUS, rates(seed(when) + 500))
        write_odim(up, when, opera.ODYSSEY, rates(seed(when)))    # strays: ignored
    root = tmp_path / "root"
    lines: list[str] = []
    code = stream_kdp.run(cfg, root, DAY2, day3 + timedelta(days=1), client=FakeBucket(up),
                          pack_workers=2, download_workers=4, log=lines.append)
    assert code == stream_kdp.EXIT_DONE, lines
    assert stream_kdp.read_status(cfg, root, DAY2)["state"] == "done"
    ds = gridded.read(cfg.store_id, root, decode=False,
                      time_range=(DAY2, day3 + timedelta(days=1)))
    chain = ds["chain"].values
    assert (chain[:96] == opera.CHAIN_CODES[opera.ODYSSEY]).all()
    assert (chain[96:100] == opera.CHAIN_CODES[opera.NIMBUS]).all()
    assert (ds["source_layout_id"].values[:96] == opera.LAYOUT_ODYSSEY).all()
    np.testing.assert_array_equal(ds["rain_rate_mmh"].values[10],
                                  expected_codes(rates(seed(DAY2 + 10 * STEP))))
    np.testing.assert_array_equal(ds["rain_rate_mmh"].values[97],
                                  expected_codes(rates(seed(day3 + STEP) + 500)))
