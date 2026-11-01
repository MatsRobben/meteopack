"""Tests for the EUMETSAT MSG/SEVIRI rapid-scan source.

Nothing below the real-archive guard needs a byte of EUMETSAT data. The fixtures
are synthetic MSG native files built into `tmp_path`: the U-MARF `Key : Value`
product headers verbatim from a real 2026-10-25 scene, NUL-padded the way the
real ones are, followed by binary filler where the level-1.5 header would start.
That is exactly the part of the file this module reads without `satpy` — the
layout fingerprint, the sensing times and the whole of `list`, `fetch` and
`scan` — so those run against files the module actually opens.

The one step that needs `satpy` is `EumetsatSeviri.load_scene`, which exists as
a documented seam for precisely this reason. The parse tests subclass it and
hand back a `SceneData` on a small, honestly-georeferenced source area, and
everything downstream — the real `pyresample` nearest-neighbour lookup, the real
grid registry, the real `gridded.write` — runs unchanged. The regrid is checked
against an independent great-circle nearest-neighbour computed here, not against
`pyresample`'s own answer.

Two things these are defending, because getting either wrong is silent:

* **The filename timestamp is the end of the scan, not the slot.** `...151916Z`
  is the 15:15 repeat cycle. Taking it at face value shifts the whole archive by
  one slot, and nothing crashes.
* **Every unit carries `valid_time`.** Without it a gridded unit cannot be kept
  to one shard, and two tasks writing one shard lose whole frames.
"""
from __future__ import annotations

import math
import os
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from meteopack.core import grids
from meteopack.core.config import ProductConfig, SourceConfig, StoreConfig, Variable
from meteopack.core.store import gridded
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, Source, TimeWindow
from meteopack.sources.eumetsat_seviri import (
    CALIBRATION_UNITS,
    LAYOUT_KEYS,
    EumetsatSeviri,
    SceneData,
    SchemaDriftError,
    apply_variable,
    default_calibration,
    header_time,
    layout_descriptor,
    nominal_slot,
    product_fingerprint,
    read_ascii_headers,
    read_scene_headers,
    scene_time,
    schema_fingerprint,
    target_area,
    wanted_variables,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "eumetsat_seviri.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "eumetsat_seviri" / "msg15_rss.yaml"

#: The two real sample locations. Everything that touches them is skipped when
#: they are not there, so the suite is green on a machine with no archive.
REAL_NAT = Path(
    "/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/eumetsat/msg_sample"
    "/MSG4-SEVI-MSG15-0100-NA-20260625120417.690000000Z-NA.nat"
)
# A real downloaded day, when one is on this machine; the tests that need it skip
# otherwise. Overridable so the suite does not assume one user's directory layout.
REAL_ARCHIVE = Path(os.environ.get(
    "METEOPACK_SEVIRI_ARCHIVE",
    "/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/eumetsat_seviri/msg15_rss",
))

#: The real registry's test grid. Real, loadable, and small enough to eyeball.
GRID_ID = "example_tiny"
CADENCE = 300

#: Verbatim from MSG4-SEVI-MSG15-0100-NA-20261025151916.069000000Z-NA.zip, read
#: on 2026-10-31. Only the times differ between scenes.
REAL_HEADERS: dict[str, str] = {
    "FormatName": "NATIVE",
    "FormatDocumentName": "Distribution Format Detailed Specification",
    "DocumentMajorVersion": "1",
    "DocumentMinorVersion": "0",
    "CreationDateTime": "2000:07:05:10:00:00:000 UTC",
    "CreatingCenter": "EUMETSAT U-MARF, Darmstadt, Germany",
    "Main Product Header": "3674           0",
    "Secondary Product Header": "1440        3674",
    "15Header": "445286        5114",
    "15Data": "101379360      450400",
    "15Trailer": "380363   101829760",
    "TotalFileSize": "102210123",
    "GORT": "GEO",
    "ASTI": "MSG4",
    "LLOS": "009.5000",
    "SNIT": "20261025151916.069000000Z",
    "AIID": "SEVI",
    "SSBT": "20261025151508.826000000Z",
    "SSST": "20261025151916.069000000Z",
    "GPLV": "15",
    "APNM": "MSG15",
    "QQOV": "OK",
    "ABID": "XXXXXXXXXXXX",
    "SMOD": "REDSCN",
    "SelectedBandIDs": "XXXXXXXXXXXX",
    "SouthLineSelectedRectangle": "2321",
    "NorthLineSelectedRectangle": "3712",
    "EastColumnSelectedRectangle": "1",
    "WestColumnSelectedRectangle": "3712",
    "NumberLinesVISIR": "1392",
    "NumberColumnsVISIR": "3712",
    "NumberLinesHRV": "4176",
    "NumberColumnsHRV": "11136",
}


# --------------------------------------------------------------------- fixtures


def header_bytes(headers: dict[str, str], *, record: int = 96) -> bytes:
    """The ASCII product headers, NUL-padded into fixed-width records.

    The real file pads every record with NULs to a fixed width and ends it with
    a newline; the exact width varies by record and is not part of the format
    this module reads, so one width is used here.
    """
    out = bytearray()
    for key, value in headers.items():
        line = f"{key:<28}: {value}".encode("latin-1")
        out += line.ljust(record - 1, b"\x00") + b"\n"
    return bytes(out)


def native_bytes(headers: dict[str, str] | None = None, *, padding: int = 4096) -> bytes:
    """A stand-in native file: real ASCII headers, then binary where the
    level-1.5 header would start. Enough for everything but `load_scene`."""
    blob = header_bytes(headers or REAL_HEADERS)
    # Bytes that look like a record but are not printable, so the parser has
    # something real to stop on rather than running off the end of the string.
    return blob + b"\x00\x01\x02binary: \xff\xfe\x00" + bytes(padding)


def scene_name(when: datetime, *, extension: str = "zip", platform: str = "MSG4") -> str:
    """The name the Data Store gives the scene whose cycle ends at `when`."""
    return (
        f"{platform}-SEVI-MSG15-0100-NA-"
        f"{when:%Y%m%d%H%M%S}.{when.microsecond * 1000:09d}Z-NA.{extension}"
    )


def write_scene(
    root: Path, slot: datetime, *, extension: str = "zip", headers: dict[str, str] | None = None
) -> Path:
    """One synthetic scene, filed the way the archive files them.

    The dissemination stamp is the slot plus 4 min 17.946 s, which is what the
    real feed does; the sensing start inside the file is the slot plus 8.826 s.
    """
    stamp = slot + timedelta(seconds=257, microseconds=946000)
    merged = dict(headers or REAL_HEADERS)
    merged["SNIT"] = f"{stamp:%Y%m%d%H%M%S}.{stamp.microsecond * 1000:09d}Z"
    merged["SSBT"] = f"{slot + timedelta(seconds=8, microseconds=826000):%Y%m%d%H%M%S}.826000000Z"
    merged["SSST"] = merged["SNIT"]

    directory = root / f"{slot:%Y}" / f"{slot:%m}" / f"{slot:%d}"
    directory.mkdir(parents=True, exist_ok=True)
    name = scene_name(stamp, extension=extension)
    path = directory / name
    payload = native_bytes(merged)
    if extension == "zip":
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name.replace(".zip", ".nat"), payload)
            archive.writestr("manifest.xml", "<manifest/>")
    else:
        path.write_bytes(payload)
    return path


def product_config(
    grid_id: str = GRID_ID,
    *,
    variables: list[Variable] | None = None,
    extra: dict | None = None,
) -> ProductConfig:
    return ProductConfig(
        source="eumetsat_seviri",
        product="msg15_rss",
        cadence_seconds=CADENCE,
        grid_id=grid_id,
        variables=[
            Variable(
                name="ir108_brightness_temperature",
                source_key="IR_108",
                units="K",
                valid_min=150.0,
                valid_max=350.0,
            )
        ]
        if variables is None
        else variables,
        store=StoreConfig(kind="gridded", chunks={"time": 1}, shard={"time": 288}),
        extra=extra or {},
    )


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """Six consecutive five-minute scenes on 2026-10-25, nested by day."""
    root = tmp_path / "archive"
    for i in range(6):
        write_scene(root, datetime(2026, 10, 25, 12, 0, tzinfo=UTC) + timedelta(minutes=5 * i))
    return root


def window(start: datetime, hours: float = 1.0) -> TimeWindow:
    return TimeWindow(start, start + timedelta(hours=hours))


# ---------------------------------------------------------------- the contract


def test_source_protocol_is_satisfied() -> None:
    assert isinstance(EumetsatSeviri(), Source)


def test_shipped_configs_load_and_agree_with_the_code() -> None:
    source = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert source.name == EumetsatSeviri.name == cfg.source
    assert cfg.store.kind == "gridded"
    assert cfg.cadence_seconds == 300
    assert cfg.store.chunks == {"time": 1, "y": 765, "x": 700}
    assert cfg.store.shard == {"time": 288, "y": 765, "x": 700}
    # One shard is exactly one UTC day, which is what lets `bucket` mode keep a
    # day of 288 scenes with a single writer.
    assert cfg.store.shard["time"] * cfg.cadence_seconds == 86400
    assert grids.get(cfg.grid_id) is not None
    # Every declared variable resolves to a real channel and a real calibration,
    # and declares the units that calibration actually produces.
    for var, channel, calibration in wanted_variables(cfg):
        assert calibration == default_calibration(channel)
        assert var.units == CALIBRATION_UNITS[calibration]


def test_config_schema_fingerprint_matches_its_known_layouts() -> None:
    """The pinned hash must be the hash of the pinned list, or `parse` lies."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    known = cfg.extra["known_layouts"]
    assert known, "the shipped config must pin at least one layout"
    assert product_fingerprint(known) == cfg.schema_fingerprint


# ------------------------------------------------------- filenames and the slot


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "MSG4-SEVI-MSG15-0100-NA-20261025151916.069000000Z-NA.zip",
            datetime(2026, 10, 25, 15, 15, tzinfo=UTC),
        ),
        (
            "MSG4-SEVI-MSG15-0100-NA-20261026200918.327000000Z-NA.zip",
            datetime(2026, 10, 26, 20, 5, tzinfo=UTC),
        ),
        (
            "MSG4-SEVI-MSG15-0100-NA-20260625120417.690000000Z-NA.nat",
            datetime(2026, 6, 25, 12, 0, tzinfo=UTC),
        ),
        # Midnight, where the dissemination stamp and the slot share a date only
        # because the delay is under five minutes.
        (
            "MSG4-SEVI-MSG15-0100-NA-20261025000417.946000000Z-NA.zip",
            datetime(2026, 10, 25, 0, 0, tzinfo=UTC),
        ),
    ],
)
def test_scene_filename_maps_to_its_repeat_cycle(name: str, expected: datetime) -> None:
    """The stamp in the name is the END of the scan; the slot is it, floored."""
    stamp = scene_time(name)
    assert stamp is not None
    assert stamp > expected, "the filename stamp must be after the slot it belongs to"
    assert stamp - expected < timedelta(seconds=CADENCE)
    assert nominal_slot(stamp, timedelta(seconds=CADENCE)) == expected


@pytest.mark.parametrize(
    "name",
    [
        "MSG4-SEVI-MSG15-0100-NA-20261025151916.069000000Z-NA.tar",
        "RAD_NL25_RAC_RT_202610251515.h5",
        "MSG4-SEVI-MSG15-0100-NA-NOTATIME-NA.zip",
        "EOPMetadata.xml",
        "",
    ],
)
def test_non_scene_names_are_not_scenes(name: str) -> None:
    assert scene_time(name) is None


def test_sensing_start_and_dissemination_stamp_floor_to_the_same_slot() -> None:
    cadence = timedelta(seconds=CADENCE)
    sensing = header_time(REAL_HEADERS["SSBT"])
    disseminated = header_time(REAL_HEADERS["SNIT"])
    assert nominal_slot(sensing, cadence) == nominal_slot(disseminated, cadence)
    assert nominal_slot(sensing, cadence) == datetime(2026, 10, 25, 15, 15, tzinfo=UTC)


def test_nominal_slot_is_a_floor_and_is_idempotent() -> None:
    cadence = timedelta(seconds=CADENCE)
    slot = datetime(2026, 10, 25, 15, 15, tzinfo=UTC)
    assert nominal_slot(slot, cadence) == slot
    assert nominal_slot(slot + timedelta(seconds=299, microseconds=999999), cadence) == slot
    # A naive datetime is read as UTC rather than as local time.
    assert nominal_slot(slot.replace(tzinfo=None), cadence) == slot


def test_header_time_reads_the_fraction_and_rejects_rubbish() -> None:
    assert header_time("20261025151508.826000000Z") == datetime(
        2026, 10, 25, 15, 15, 8, 826000, tzinfo=UTC
    )
    assert header_time("20261025151508Z") == datetime(2026, 10, 25, 15, 15, 8, tzinfo=UTC)
    with pytest.raises(SchemaDriftError):
        header_time("2026-10-25T15:15:08Z")


# -------------------------------------------------------------- product headers


def test_ascii_headers_round_trip_and_stop_at_the_binary_section() -> None:
    headers = read_ascii_headers(native_bytes())
    assert headers["ASTI"] == "MSG4"
    assert headers["SMOD"] == "REDSCN"
    assert headers["SelectedBandIDs"] == "XXXXXXXXXXXX"
    assert headers["NumberColumnsVISIR"] == "3712"
    # The binary filler carries a colon; it must not become a header key.
    assert not any(k.startswith("\x00") for k in headers)
    assert "binary" not in headers


def test_headers_are_read_from_a_zip_and_from_a_loose_nat(tmp_path: Path) -> None:
    root = tmp_path / "a"
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    zipped = write_scene(root, slot, extension="zip")
    loose = write_scene(root, slot + timedelta(minutes=5), extension="nat")
    assert read_scene_headers(zipped)["ASTI"] == "MSG4"
    assert read_scene_headers(loose)["ASTI"] == "MSG4"


def test_a_zip_with_a_traversing_member_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "evil.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("../escape.nat", native_bytes())
    with pytest.raises(SchemaDriftError, match="path component"):
        read_scene_headers(path)


def test_a_file_without_product_headers_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "junk.nat"
    path.write_bytes(bytes(8192))
    with pytest.raises(SchemaDriftError, match="no U-MARF product headers"):
        read_scene_headers(path)


# ------------------------------------------------------------------ fingerprint


def test_layout_descriptor_covers_the_identity_fields_and_nothing_per_scene() -> None:
    descriptor = layout_descriptor(REAL_HEADERS)
    for key in LAYOUT_KEYS:
        assert f"{key}=" in descriptor
    # Times change every five minutes and must not be part of the layout.
    for key in ("SNIT", "SSBT", "SSST", "QQOV"):
        assert f"{key}=" not in descriptor


def test_fingerprint_is_stable_and_moves_only_on_a_real_change() -> None:
    base = layout_descriptor(REAL_HEADERS)
    assert schema_fingerprint([base]) == schema_fingerprint([base, base, "  "])

    later = dict(REAL_HEADERS, SNIT="20270101000000.000000000Z", SSBT="20270101000000.000000000Z")
    assert layout_descriptor(later) == base, "a different scene is not a different layout"

    for key, value in (
        ("ASTI", "MTG1"),
        ("SMOD", "FULSCN"),
        ("LLOS", "000.0000"),
        ("SelectedBandIDs", "XXXXXXXXXXX-"),
        ("SouthLineSelectedRectangle", "2956"),
        ("15Header", "445999        5114"),
    ):
        moved = dict(REAL_HEADERS)
        moved[key] = value
        assert schema_fingerprint([layout_descriptor(moved)]) != schema_fingerprint([base])

    # A field that disappears upstream must move the hash, not be skipped.
    dropped = {k: v for k, v in REAL_HEADERS.items() if k != "SMOD"}
    assert "SMOD=<absent>" in layout_descriptor(dropped)
    assert schema_fingerprint([layout_descriptor(dropped)]) != schema_fingerprint([base])


def test_shipped_fingerprint_matches_the_real_product_layout() -> None:
    """The hash pinned in the config is the hash of the real 2026 scene."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert schema_fingerprint([layout_descriptor(REAL_HEADERS)]) in cfg.extra["known_layouts"]


# ------------------------------------------------------------------------- list


def test_list_enumerates_scenes_and_every_unit_carries_valid_time(archive: Path) -> None:
    src = EumetsatSeviri(archive_root=archive)
    units = src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())
    assert [u.valid_time for u in units] == [
        datetime(2026, 10, 25, 12, 0, tzinfo=UTC) + timedelta(minutes=5 * i) for i in range(6)
    ]
    # The planner refuses a gridded unit without one; see jobs.plan.shard_bucket.
    assert all(u.valid_time is not None for u in units)
    assert [u.key for u in units] == [
        f"eumetsat_seviri/msg15_rss/20261025T{12:02d}{5 * i:02d}" for i in range(6)
    ]
    assert all(u.url.startswith("file://") for u in units)
    assert all((u.meta or {})["raw_subdir"] == "2026/10/25" for u in units)


def test_list_is_bounded_by_the_window(archive: Path) -> None:
    src = EumetsatSeviri(archive_root=archive)
    cfg = product_config()
    got = src.list(
        TimeWindow(
            datetime(2026, 10, 25, 12, 5, tzinfo=UTC), datetime(2026, 10, 25, 12, 15, tzinfo=UTC)
        ),
        cfg,
    )
    assert [u.valid_time.minute for u in got] == [5, 10]
    empty = src.list(window(datetime(2026, 10, 26, 12, 0, tzinfo=UTC)), cfg)
    assert empty == []


def test_list_ignores_strangers_and_keeps_one_unit_per_slot(archive: Path, caplog) -> None:
    day = archive / "2026" / "10" / "25"
    (day / "EOPMetadata.xml").write_text("<x/>")
    (day / "notes.txt").write_text("hello")
    # The same slot, unzipped beside the zip: one slot, one unit.
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    write_scene(archive, slot, extension="nat")

    src = EumetsatSeviri(archive_root=archive)
    units = src.list(window(slot), product_config())
    assert len(units) == 6
    assert len({u.key for u in units}) == 6


def test_list_reads_a_flat_archive(tmp_path: Path) -> None:
    root = tmp_path / "flat"
    root.mkdir()
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    stamp = slot + timedelta(seconds=257, microseconds=946000)
    (root / scene_name(stamp)).write_bytes(native_bytes())

    cfg = product_config(extra={"archive_layout": "flat"})
    units = EumetsatSeviri(archive_root=root).list(window(slot), cfg)
    assert [u.valid_time for u in units] == [slot]


def test_list_refuses_an_unknown_layout_and_a_missing_root(tmp_path: Path) -> None:
    src = EumetsatSeviri(archive_root=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())
    with pytest.raises(ValueError, match="archive_layout"):
        EumetsatSeviri(archive_root=tmp_path).list(
            window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)),
            product_config(extra={"archive_layout": "by-hour"}),
        )


def test_list_without_an_archive_root_says_so() -> None:
    with pytest.raises(ValueError, match="archive root"):
        EumetsatSeviri().list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())


def test_list_validates_the_window_through_timeline(archive: Path) -> None:
    src = EumetsatSeviri(archive_root=archive)
    start = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        src.list(TimeWindow(start, start), product_config())


# ------------------------------------------------------------------------ fetch


def test_fetch_publishes_hashes_and_is_idempotent(archive: Path, tmp_path: Path) -> None:
    src = EumetsatSeviri(archive_root=archive)
    unit = src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())[0]
    raw_root = tmp_path / "raw"

    first = src.fetch(unit, raw_root)
    assert first is not None
    assert first.unit_key == unit.key
    assert first.path == raw_root / "2026" / "10" / "25" / (unit.meta or {})["filename"]
    assert first.n_bytes == first.path.stat().st_size

    import hashlib

    assert first.sha256 == hashlib.sha256(first.path.read_bytes()).hexdigest()

    second = src.fetch(unit, raw_root)
    assert second is not None
    assert (second.path, second.sha256, second.n_bytes) == (
        first.path,
        first.sha256,
        first.n_bytes,
    )
    assert not list(raw_root.rglob("*.part"))


def test_fetch_returns_none_when_the_scene_has_vanished(archive: Path, tmp_path: Path) -> None:
    src = EumetsatSeviri(archive_root=archive)
    unit = src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())[0]
    Path((unit.meta or {})["source_path"]).unlink()
    assert src.fetch(unit, tmp_path / "raw") is None


def test_fetch_copy_mode_makes_a_real_second_copy(archive: Path, tmp_path: Path) -> None:
    source = SourceConfig(name="eumetsat_seviri", extra={"link_mode": "copy"})
    src = EumetsatSeviri(source, archive_root=archive)
    unit = src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())[0]
    raw = src.fetch(unit, tmp_path / "raw")
    assert raw is not None
    assert raw.path.stat().st_ino != Path((unit.meta or {})["source_path"]).stat().st_ino
    assert raw.path.read_bytes() == Path((unit.meta or {})["source_path"]).read_bytes()


def test_fetch_rejects_an_unknown_link_mode(archive: Path, tmp_path: Path) -> None:
    source = SourceConfig(name="eumetsat_seviri", extra={"link_mode": "teleport"})
    src = EumetsatSeviri(source, archive_root=archive)
    unit = src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())[0]
    with pytest.raises(ValueError, match="link_mode"):
        src.fetch(unit, tmp_path / "raw")


# ------------------------------------------------------------------------- scan


def test_scan_reports_one_variant_for_one_product_generation(archive: Path) -> None:
    src = EumetsatSeviri(archive_root=archive)
    units = src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())
    raws = [
        RawFile(path=Path((u.meta or {})["source_path"]), sha256="x", n_bytes=0, unit_key=u.key)
        for u in units
    ]
    report = src.scan(raws)
    assert report["source"] == "eumetsat_seviri"
    assert report["n_files"] == len(raws)
    assert report["n_variants"] == 1
    assert report["errors"] == []
    variant = report["variants"][0]
    assert variant["count"] == len(raws)
    assert variant["headers"]["SMOD"] == "REDSCN"
    assert variant["first_seen"] < variant["last_seen"]
    assert report["fingerprint"] == product_fingerprint(report["fingerprints"])


def test_scan_separates_a_changed_product_and_records_a_bad_file(
    archive: Path, tmp_path: Path
) -> None:
    good = next((archive / "2026" / "10" / "25").glob("*.zip"))
    drifted = tmp_path / "drifted.nat"
    drifted.write_bytes(native_bytes(dict(REAL_HEADERS, ASTI="MTG1", SMOD="FULSCN")))
    broken = tmp_path / "broken.nat"
    broken.write_bytes(bytes(1024))

    report = EumetsatSeviri().scan(
        [
            RawFile(path=p, sha256="x", n_bytes=0, unit_key=str(p))
            for p in (good, drifted, broken)
        ]
    )
    assert report["n_variants"] == 2
    assert len(report["fingerprints"]) == 2
    assert len(report["errors"]) == 1
    assert "broken.nat" in report["errors"][0]["path"]


# ----------------------------------------------------------------- the grid map


@pytest.mark.parametrize("grid_id", ["knmi_1km_stereo", GRID_ID])
def test_target_area_reproduces_the_registry_corners(grid_id: str) -> None:
    """The area must sit exactly on the grid the registry declares.

    The registry's corners are the outer edges of the image, which is what an
    area extent is, so the lon/lat the area reports for a corner *cell* is half
    a pixel inside. The tolerance below is that half pixel and no more.
    """
    grid = grids.get(grid_id)
    area = target_area(grid)
    assert area.shape == tuple(grid.shape)

    lons, lats = area.get_lonlats()
    width = (area.area_extent[2] - area.area_extent[0]) / grid.nx
    height = (area.area_extent[3] - area.area_extent[1]) / grid.ny

    from pyproj import Transformer

    inverse = Transformer.from_crs(grid.proj4, "EPSG:4326", always_xy=True)
    lower_left, upper_left, upper_right, lower_right = grid.corners_lonlat
    expected = {
        (0, 0): (area.area_extent[0] + width / 2, area.area_extent[3] - height / 2),
        (0, grid.nx - 1): (area.area_extent[2] - width / 2, area.area_extent[3] - height / 2),
        (grid.ny - 1, 0): (area.area_extent[0] + width / 2, area.area_extent[1] + height / 2),
    }
    for (row, col), (x, y) in expected.items():
        want_lon, want_lat = inverse.transform(x, y)
        assert lons[row, col] == pytest.approx(want_lon, abs=1e-9)
        assert lats[row, col] == pytest.approx(want_lat, abs=1e-9)

    # And the extent itself is the bounding box of the registry's own corners.
    forward = Transformer.from_crs("EPSG:4326", grid.proj4, always_xy=True)
    xs, ys = zip(*(forward.transform(lon, lat) for lon, lat in grid.corners_lonlat), strict=True)
    assert area.area_extent == pytest.approx((min(xs), min(ys), max(xs), max(ys)))

    # Row 0 is the north edge and column 0 the west edge, the way every other
    # gridded product in this repo is stored.
    assert lats[0, 0] > lats[-1, 0]
    assert lons[0, 0] < lons[0, -1]
    assert upper_left[1] > lower_left[1] and upper_right[0] > upper_left[0]
    assert lower_right[0] > lower_left[0]


def test_target_area_needs_corners() -> None:
    grid = grids.Grid(id="cornerless", proj4="+proj=stere +lat_0=90", shape=(4, 4))
    with pytest.raises(ValueError, match="product_corners_lonlat"):
        target_area(grid)


# ------------------------------------------------------------- value handling


def test_apply_variable_nulls_sentinels_before_it_scales_anything() -> None:
    """Order is the whole point: scale first and a sentinel becomes a number."""
    var = Variable(
        name="t", source_key="IR_108", units="K", scale=2.0, offset=1.0, valid_min=0.0,
        valid_max=100.0,
    )
    values = np.array([[1.0, -999.0, np.inf, 40.0, 60.0]], dtype="float32")
    out = apply_variable(values, var, sentinels=(-999.0,))
    # 1 -> 3; -999 and inf -> NaN *before* scaling; 40 -> 81 which is in range;
    # 60 -> 121 which is not, and the range is checked after the scaling.
    assert out[0, 0] == pytest.approx(3.0)
    assert np.isnan(out[0, 1]) and np.isnan(out[0, 2])
    assert out[0, 3] == pytest.approx(81.0)
    assert np.isnan(out[0, 4])
    assert out.dtype == np.dtype("float32")
    # The caller's array is not modified.
    assert values[0, 1] == pytest.approx(-999.0)


def test_apply_variable_without_scaling_is_a_range_check_only() -> None:
    var = Variable(name="t", source_key="IR_108", units="K", valid_min=150.0, valid_max=350.0)
    out = apply_variable(np.array([[149.0, 150.0, 300.0, 351.0]], dtype="float32"), var)
    assert np.isnan(out[0, 0]) and np.isnan(out[0, 3])
    assert out[0, 1] == pytest.approx(150.0)


@pytest.mark.parametrize(
    ("source_key", "channel", "calibration"),
    [
        ("IR_108", "IR_108", "brightness_temperature"),
        ("VIS006", "VIS006", "reflectance"),
        ("HRV", "HRV", "reflectance"),
        ("IR_108:radiance", "IR_108", "radiance"),
        (" WV_062 : counts ", "WV_062", "counts"),
    ],
)
def test_source_key_resolves_channel_and_calibration(
    source_key: str, channel: str, calibration: str
) -> None:
    cfg = product_config(
        variables=[Variable(name="v", source_key=source_key, units=CALIBRATION_UNITS[calibration])]
    )
    assert wanted_variables(cfg) == [(cfg.variables[0], channel, calibration)]


@pytest.mark.parametrize("source_key", ["IR_999", "", "IR_108:temperature", "ir_108"])
def test_a_source_key_that_is_not_a_channel_is_refused(source_key: str) -> None:
    cfg = product_config(variables=[Variable(name="v", source_key=source_key, units="K")])
    with pytest.raises(ValueError):
        wanted_variables(cfg)


def test_a_product_with_no_variables_is_refused() -> None:
    with pytest.raises(ValueError, match="declares no variables"):
        wanted_variables(product_config(variables=[]))


# ------------------------------------------------------------------------ parse


def source_area(resolution: float = 0.25):
    """A small, honestly-georeferenced stand-in for the SEVIRI grid.

    Plate carree rather than geostationary on purpose: the point of the parse
    tests is the resampling and the bookkeeping around it, and a source whose
    cell centres can be written down in one line is one whose nearest neighbour
    can be recomputed here independently.
    """
    from pyresample.geometry import AreaDefinition

    extent = (-2.0, 48.0, 14.0, 58.0)  # lon/lat bounds, comfortably over the test grid
    nx = round((extent[2] - extent[0]) / resolution)
    ny = round((extent[3] - extent[1]) / resolution)
    return AreaDefinition("fake_seviri", "fake", "fake", "+proj=longlat +datum=WGS84", nx, ny, extent)


def expected_nearest(area, source, values: np.ndarray) -> np.ndarray:
    """Nearest source cell per target cell, by great-circle distance.

    An independent implementation of what `pyresample` is being asked to do --
    it works in 3-D cartesian on a sphere, which orders points the same way --
    so the test compares the regrid against geometry rather than against
    `pyresample`'s own answer.
    """
    src_lon, src_lat = source.get_lonlats()
    tgt_lon, tgt_lat = area.get_lonlats()
    src = np.stack([np.radians(src_lon.ravel()), np.radians(src_lat.ravel())], axis=1)
    out = np.empty(area.shape, dtype="float32")
    flat = values.ravel()
    for row in range(area.shape[0]):
        for col in range(area.shape[1]):
            lon, lat = math.radians(tgt_lon[row, col]), math.radians(tgt_lat[row, col])
            cosine = np.sin(src[:, 1]) * math.sin(lat) + np.cos(src[:, 1]) * math.cos(lat) * np.cos(
                src[:, 0] - lon
            )
            out[row, col] = flat[int(np.argmax(cosine))]
    return out


class FakeSeviri(EumetsatSeviri):
    """The real source with the one `satpy` step replaced.

    Everything else -- the header read, the layout check, the slot check, the
    real `pyresample` lookup, the real grid registry -- runs unchanged.
    """

    def __init__(self, *args, values=None, units="K", area=None, start_time=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._values = values
        self._units = units
        self._area = area if area is not None else source_area()
        self._start_time = start_time
        self.loaded: list[tuple[str, str]] = []

    def load_scene(self, path, wanted, cfg):
        lon, lat = self._area.get_lonlats()
        default = (lat * 100.0 + lon).astype("float32")
        arrays, areas, units = {}, {}, {}
        for _var, channel, calibration in wanted:
            self.loaded.append((channel, calibration))
            arrays[channel] = (self._values if self._values is not None else default).copy()
            areas[channel] = self._area
            units[channel] = self._units
        start = self._start_time
        if start is None:
            start = nominal_slot(header_time(read_scene_headers(path)["SSBT"]), cfg.cadence)
        return SceneData(
            arrays=arrays, areas=areas, units=units, start_time=start, attrs={"platform_name": "x"}
        )


def one_raw(archive: Path, slot: datetime, cfg: ProductConfig) -> RawFile:
    unit = EumetsatSeviri(archive_root=archive).list(window(slot), cfg)[0]
    return RawFile(
        path=Path((unit.meta or {})["source_path"]), sha256="deadbeef", n_bytes=1, unit_key=unit.key
    )


def test_parse_regrids_onto_the_product_grid(archive: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config()
    src = FakeSeviri(archive_root=archive)
    ds = src.parse(one_raw(archive, slot, cfg), cfg)

    grid = grids.get(GRID_ID)
    assert isinstance(ds, xr.Dataset)
    assert ds["ir108_brightness_temperature"].dims == ("time", "y", "x")
    assert ds["ir108_brightness_temperature"].shape == (1, *grid.shape)
    assert ds["ir108_brightness_temperature"].dtype == np.dtype("float32")
    assert ds["time"].values[0] == np.datetime64(slot.replace(tzinfo=None), "ns")
    assert ds.attrs["grid_id"] == GRID_ID
    assert ds.attrs["unit_key"].endswith("20261025T1200")
    assert ds.attrs["raw_sha256"] == "deadbeef"
    assert ds.attrs["platform"] == "MSG4"
    assert ds["ir108_brightness_temperature"].attrs["channel"] == "IR_108"
    assert src.loaded == [("IR_108", "brightness_temperature")]


def test_parse_puts_every_value_where_the_geometry_says_it_goes(archive: Path) -> None:
    """Nearest neighbour, checked against an independent great-circle search.

    This is the test that would fail if the target area were flipped, offset by
    a pixel, or built from the wrong corners: the source field is a known
    function of latitude and longitude, so every target cell has one right
    answer and it is computed here without `pyresample`.
    """
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    # valid_min/max off, so the comparison is against the regrid alone.
    cfg = product_config(
        variables=[Variable(name="field", source_key="IR_108", units="K")],
        extra={"radius_of_influence_m": 50_000},
    )
    src = FakeSeviri(archive_root=archive)
    ds = src.parse(one_raw(archive, slot, cfg), cfg)

    area = target_area(grids.get(GRID_ID))
    lon, lat = src._area.get_lonlats()
    want = expected_nearest(area, src._area, (lat * 100.0 + lon).astype("float32"))
    np.testing.assert_allclose(ds["field"].values[0], want, rtol=0, atol=1e-4)


def test_parse_leaves_cells_with_no_neighbour_missing(archive: Path) -> None:
    """A radius too small to reach any source pixel is NaN, not a nearby guess."""
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config(extra={"radius_of_influence_m": 1.0})
    ds = FakeSeviri(archive_root=archive).parse(one_raw(archive, slot, cfg), cfg)
    assert np.isnan(ds["ir108_brightness_temperature"].values).all()


def test_parse_applies_the_valid_range_after_the_regrid(archive: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config(
        variables=[
            Variable(
                name="t", source_key="IR_108", units="K", valid_min=5300.0, valid_max=5400.0
            )
        ],
        extra={"radius_of_influence_m": 50_000},
    )
    ds = FakeSeviri(archive_root=archive).parse(one_raw(archive, slot, cfg), cfg)
    values = ds["t"].values[0]
    finite = values[np.isfinite(values)]
    assert finite.size, "the fixture should leave some values inside the range"
    assert finite.min() >= 5300.0 and finite.max() <= 5400.0
    assert np.isnan(values).any(), "and some outside it"


def test_parse_refuses_units_that_are_not_the_calibration(archive: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config()
    # The reader starts handing back radiance where brightness temperature used
    # to be: plausible numbers, two orders of magnitude out.
    src = FakeSeviri(archive_root=archive, units="mW m-2 sr-1 (cm-1)-1")
    with pytest.raises(SchemaDriftError, match="should be in 'K'"):
        src.parse(one_raw(archive, slot, cfg), cfg)

    # ...and the same check catches a config that declares the wrong units.
    wrong = product_config(variables=[Variable(name="t", source_key="IR_108", units="degC")])
    with pytest.raises(SchemaDriftError, match="the config"):
        FakeSeviri(archive_root=archive).parse(one_raw(archive, slot, wrong), wrong)


def test_parse_refuses_a_layout_outside_the_pinned_set(archive: Path, tmp_path: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    known = schema_fingerprint([layout_descriptor(REAL_HEADERS)])
    cfg = product_config(
        extra={"known_layouts": [known], "radius_of_influence_m": 50_000},
    )
    cfg = cfg.model_copy(update={"schema_fingerprint": product_fingerprint([known])})
    src = FakeSeviri(archive_root=archive)
    # The pinned layout parses.
    assert src.parse(one_raw(archive, slot, cfg), cfg) is not None

    drifted = tmp_path / "drifted.nat"
    drifted.write_bytes(native_bytes(dict(REAL_HEADERS, SMOD="FULSCN")))
    raw = RawFile(
        path=drifted, sha256="x", n_bytes=0, unit_key="eumetsat_seviri/msg15_rss/20261025T1515"
    )
    with pytest.raises(SchemaDriftError, match="is not in the pinned set"):
        src.parse(raw, cfg)


def test_parse_refuses_a_config_whose_fingerprint_does_not_match(archive: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config(extra={"known_layouts": ["0123456789abcdef"]})
    cfg = cfg.model_copy(update={"schema_fingerprint": "not-the-hash"})
    with pytest.raises(SchemaDriftError, match="config is inconsistent"):
        FakeSeviri(archive_root=archive).parse(one_raw(archive, slot, cfg), cfg)


def test_parse_refuses_a_scene_whose_sensing_time_contradicts_its_unit(tmp_path: Path) -> None:
    """The guard on flooring the filename stamp: if the dissemination delay ever
    grows past one cadence, the slot in the key stops matching the file."""
    path = tmp_path / "late.nat"
    path.write_bytes(native_bytes(dict(REAL_HEADERS, SSBT="20261025152008.826000000Z")))
    raw = RawFile(
        path=path, sha256="x", n_bytes=0, unit_key="eumetsat_seviri/msg15_rss/20261025T1515"
    )
    with pytest.raises(SchemaDriftError, match="dissemination delay"):
        FakeSeviri().parse(raw, product_config())


def test_parse_refuses_a_reader_that_disagrees_about_the_slot(archive: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config()
    src = FakeSeviri(archive_root=archive, start_time=slot + timedelta(minutes=15))
    with pytest.raises(SchemaDriftError, match="the reader reports slot"):
        src.parse(one_raw(archive, slot, cfg), cfg)


def test_parse_needs_a_grid(archive: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config()
    raw = one_raw(archive, slot, cfg)
    cfg = cfg.model_copy(update={"grid_id": None})
    with pytest.raises(ValueError, match="no grid_id"):
        FakeSeviri(archive_root=archive).parse(raw, cfg)


def test_parse_refuses_a_resampler_it_does_not_implement(archive: Path) -> None:
    slot = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    cfg = product_config(extra={"resampler": "bilinear"})
    with pytest.raises(ValueError, match="resampler"):
        FakeSeviri(archive_root=archive).parse(one_raw(archive, slot, cfg), cfg)


# ------------------------------------------------- the store accepts the output


def test_parse_output_is_written_by_the_real_gridded_store(archive: Path, tmp_path: Path) -> None:
    """The contract that matters: `parse` -> `gridded.write`, unmodified.

    Six consecutive scenes, written one at a time the way a packing task writes
    them, then read back and compared frame for frame.
    """
    cfg = product_config(
        variables=[
            Variable(name="t", source_key="IR_108", units="K"),
            Variable(name="w", source_key="WV_062", units="K"),
        ],
        extra={"radius_of_influence_m": 50_000},
    )
    start = datetime(2026, 10, 25, 12, 0, tzinfo=UTC)
    timeline = Timeline(start, start + timedelta(minutes=30), timedelta(seconds=CADENCE))
    root = tmp_path / "data"
    gridded.initialise(cfg, root, timeline)

    src = FakeSeviri(archive_root=archive)
    units = src.list(window(start), cfg)
    assert len(units) == 6
    written = {}
    for unit in units:
        raw = RawFile(
            path=Path((unit.meta or {})["source_path"]),
            sha256="x",
            n_bytes=0,
            unit_key=unit.key,
        )
        ds = src.parse(raw, cfg)
        gridded.write(ds, cfg, root)
        written[unit.valid_time] = ds

    back = gridded.read(cfg.store_id, root)
    assert back["t"].shape == (6, *grids.get(GRID_ID).shape)
    assert not np.isnan(back["t"].values).any()
    for i, slot in enumerate(sorted(written)):
        np.testing.assert_allclose(back["t"].values[i], written[slot]["t"].values[0])
        np.testing.assert_allclose(back["w"].values[i], written[slot]["w"].values[0])

    # Idempotent: writing the same frame again changes nothing.
    gridded.write(written[start], cfg, root)
    np.testing.assert_allclose(gridded.read(cfg.store_id, root)["t"].values, back["t"].values)


def test_units_land_on_shard_boundaries_the_planner_can_split(archive: Path) -> None:
    """One UTC day of scenes is one shard, and the planner can see that."""
    from meteopack.jobs.plan import shard_bucket, shard_span_seconds, timeline_origin

    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    span = shard_span_seconds(cfg)
    assert span == 86400
    origin = timeline_origin(cfg)
    src = EumetsatSeviri(archive_root=archive)
    units = src.list(window(datetime(2026, 10, 25, 12, 0, tzinfo=UTC)), product_config())
    buckets = {shard_bucket(u, span, origin) for u in units}
    assert len(buckets) == 1, "scenes from one UTC day must share one shard"


# ------------------------------------------------- the real archive, if present


real_nat = pytest.mark.skipif(not REAL_NAT.exists(), reason="no local MSG native sample")
real_archive = pytest.mark.skipif(
    not REAL_ARCHIVE.is_dir(), reason="no local MSG15 RSS scene archive"
)


@real_nat
def test_real_sample_headers_match_the_fixtures() -> None:
    """What proves the synthetic fixtures are shaped like the data."""
    headers = read_scene_headers(REAL_NAT)
    for key in LAYOUT_KEYS:
        assert headers[key] == REAL_HEADERS[key], key
    assert layout_descriptor(headers) == layout_descriptor(REAL_HEADERS)


@real_nat
def test_real_sample_is_a_pinned_layout() -> None:
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    report = EumetsatSeviri().scan(
        [RawFile(path=REAL_NAT, sha256="x", n_bytes=0, unit_key="k")]
    )
    assert report["n_variants"] == 1
    assert report["fingerprints"][0] in cfg.extra["known_layouts"]


@real_archive
def test_real_archive_lists_a_full_five_minute_day() -> None:
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    src = EumetsatSeviri(archive_root=REAL_ARCHIVE)
    days = sorted(p for p in REAL_ARCHIVE.rglob("*.zip"))
    if not days:
        pytest.skip("the archive is empty")
    first = scene_time(days[0].name)
    assert first is not None
    start = nominal_slot(first, cfg.cadence).replace(hour=0, minute=0)
    units = src.list(TimeWindow(start, start + timedelta(days=1)), cfg)
    assert units, "a day of the archive should hold scenes"
    assert all(u.valid_time is not None for u in units)
    assert all(u.valid_time.second == 0 and u.valid_time.minute % 5 == 0 for u in units)
    assert len({u.key for u in units}) == len(units)


@real_nat
def test_real_scene_lands_on_the_seviri_pixel_the_projection_names() -> None:
    """Geolocation of the real regrid, checked without `pyresample`'s KD-tree.

    For 2000 random cells of the KNMI grid, the SEVIRI pixel that *contains*
    that cell's lon/lat is found by the source area's own forward projection,
    and compared with what the regrid actually put there. The two definitions
    are not identical -- a point near a pixel edge is contained by one pixel and
    nearest to its neighbour -- so the contract is: the great majority agree
    exactly, and none is ever more than one SEVIRI pixel away. A flip, a
    half-pixel offset or the wrong corners would break the second half at once.

    Measured on the 2026-06-25 sample: 96.0% exact, 100% within one pixel.
    """
    pytest.importorskip("satpy")
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    src = EumetsatSeviri()
    slot = nominal_slot(scene_time(REAL_NAT.name), cfg.cadence)
    raw = RawFile(
        path=REAL_NAT,
        sha256="x",
        n_bytes=0,
        unit_key=f"eumetsat_seviri/msg15_rss/{slot:%Y%m%dT%H%M}",
    )
    scene = src.load_scene(REAL_NAT, wanted_variables(cfg), cfg)
    regridded = src.parse(raw, cfg)["ir108_brightness_temperature"].values[0]

    area = target_area(grids.get(cfg.grid_id))
    source = scene.areas["IR_108"]
    field = scene.arrays["IR_108"]
    lons, lats = area.get_lonlats()
    rng = np.random.default_rng(0)
    rows = rng.integers(0, area.shape[0], 2000)
    cols = rng.integers(0, area.shape[1], 2000)

    col_idx, row_idx = source.get_array_indices_from_lonlat(lons[rows, cols], lats[rows, cols])
    col_idx, row_idx = np.asarray(col_idx), np.asarray(row_idx)
    got = regridded[rows, cols]

    exact = np.isclose(got, field[row_idx, col_idx], atol=1e-5)
    assert exact.mean() > 0.9, f"only {exact.mean():.1%} landed on the containing pixel"

    neighbourhood = np.stack(
        [
            field[
                np.clip(row_idx + dy, 0, field.shape[0] - 1),
                np.clip(col_idx + dx, 0, field.shape[1] - 1),
            ]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    )
    within_one_pixel = np.isclose(got[None, :], neighbourhood, atol=1e-5).any(axis=0)
    assert within_one_pixel.all(), (
        f"{int((~within_one_pixel).sum())} cells came from more than one SEVIRI pixel away"
    )


@real_nat
def test_real_scene_parses_onto_the_radar_grid_and_is_written() -> None:
    """The only test that runs satpy. Slow, and the point of the whole module.

    The physical checks are deliberately crude and cloud-proof: a June midday
    scene over the Netherlands cannot have a 10.8 um brightness temperature
    outside 200-320 K, and the 6.2 um water vapour channel sees the upper
    troposphere, so it must be colder than the window channel everywhere.
    """
    pytest.importorskip("satpy")
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    src = EumetsatSeviri(archive_root=REAL_NAT.parent)
    slot = nominal_slot(scene_time(REAL_NAT.name), cfg.cadence)
    raw = RawFile(
        path=REAL_NAT,
        sha256="x",
        n_bytes=REAL_NAT.stat().st_size,
        unit_key=f"eumetsat_seviri/msg15_rss/{slot:%Y%m%dT%H%M}",
    )
    ds = src.parse(raw, cfg)

    grid = grids.get(cfg.grid_id)
    assert ds["ir108_brightness_temperature"].shape == (1, *grid.shape)
    assert ds["time"].values[0] == np.datetime64(slot.replace(tzinfo=None), "ns")
    assert ds.attrs["platform"] == "MSG4"
    assert ds.attrs["scan_mode"] == "REDSCN"

    window_bt = ds["ir108_brightness_temperature"].values[0]
    vapour_bt = ds["wv062_brightness_temperature"].values[0]
    visible = ds["vis006_reflectance"].values[0]
    # The RSS footprint covers the whole KNMI domain: nothing may be missing.
    for name, values in (("IR_108", window_bt), ("WV_062", vapour_bt), ("VIS006", visible)):
        assert not np.isnan(values).any(), f"{name} has holes over the domain"
    assert 200.0 < np.nanmin(window_bt) and np.nanmax(window_bt) < 320.0
    assert np.nanmax(vapour_bt) < np.nanmin(window_bt) + 60.0
    assert (vapour_bt < window_bt).all(), "6.2 um must be colder than 10.8 um everywhere"
    assert np.nanmean(visible) > 1.0, "a midday June scene is lit"

    # And the real store takes it.
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        timeline = Timeline(slot, slot + timedelta(seconds=CADENCE), cfg.cadence)
        gridded.initialise(cfg, tmp, timeline)
        gridded.write(ds, cfg, tmp)
        back = gridded.read(cfg.store_id, tmp)
        np.testing.assert_allclose(back["ir108_brightness_temperature"].values[0], window_bt)


def test_span_of_with_the_config_is_the_exact_slot() -> None:
    """Padded spans are fine for skipping files and wrong for shard grouping.

    The +/-15 minutes made the scenes either side of midnight share a shard, so
    every day chained into one task and a whole archive packed serially.
    """
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    name = "MSG4-SEVI-MSG15-0100-NA-20261025235918.069000000Z-NA.nat"
    first, last = EumetsatSeviri().span_of(Path(name), cfg=cfg)
    assert first == last == datetime(2026, 10, 25, 23, 55, tzinfo=UTC)
