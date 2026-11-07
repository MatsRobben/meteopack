"""Tests for the KNMI 10-minute in-situ (KMDS) source.

Nothing here touches the network, and nothing here needs the real archive: the
fixtures are tiny netCDF members written with h5netcdf into `tmp_path` and
tarred exactly the way KNMI tars a day, carrying the real variable names, the
real `seconds since 1950-01-01` time axis and the real NaN `_FillValue`. A
handful of tests at the bottom read the actual archive and skip when it is not
mounted; the numbers they assert were measured off it.

The two facts most of this file exists to protect are the two that make this
source different from every other one here: **a day's tar is ONE unit** — the
whole point of `RawLayout(archive="tar", granularity="day")`, since KNMI
publishes one file per ten minutes and meters on request count — and **NaN is
nulled before scale is applied**, because polars treats a float NaN as a value
and this feed uses it as its fill.
"""
from __future__ import annotations

import io
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5netcdf
import h5py
import numpy as np
import polars as pl
import pytest
import yaml

from meteopack.core import http
from meteopack.core.config import ProductConfig, SourceConfig, StoreConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.stations.knmi_insitu import (
    STRUCTURAL,
    KnmiInsitu,
    SchemaDriftError,
    member_time,
    schema_fingerprint,
    station_id,
    tar_date,
    variables_by_key,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "stations" / "knmi_insitu.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "stations" / "knmi_insitu" / "obs_10min.yaml"

#: The real archive. Every test that reads it is skipped when it is absent.
REAL_ARCHIVE = Path(
    "/vol/knmimo-nobackup/users/mrobben/meteopack_data/raw/stations/knmi_insitu/obs_10min"
)

#: `seconds since 1950-01-01`, exactly as the files write it.
_EPOCH = datetime(1950, 1, 1, tzinfo=UTC)

#: Two in-footprint Dutch stations and one that is not (Rome), so the grid
#: clipping test exercises a real polygon rather than a bounding box.
DE_BILT = ("06260", 52.0989, 5.1797, 1.9, "De Bilt", "0-20000-0-06260")
MAASTRICHT = ("06380", 50.9067, 5.7625, 114.3, "Maastricht", "0-20000-0-06380")
ROME = ("16239", 41.9028, 12.4964, 130.0, "Roma Ciampino", "0-380-0-16239")


# --------------------------------------------------------------------- fixtures


def write_member(
    path: Path,
    when: datetime,
    *,
    stations=(DE_BILT, MAASTRICHT),
    values: dict[str, list[float]] | None = None,
) -> Path:
    """One KMDS member: `(station, time=1)`, NaN fill, CF time since 1950.

    Deliberately written with h5netcdf rather than mocked, so the reader is
    exercised against a real netCDF-4 file with the real dtypes.
    """
    n = len(stations)
    if values is None:
        values = {
            "ta": [5.5 + i for i in range(n)],
            "ss": [10.0] + [0.0] * (n - 1),
            "rg": [0.6] + [0.0] * (n - 1),
        }
    offset = (when - _EPOCH).total_seconds()
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"station": n, "time": 1}
        t = f.create_variable("time", ("time",), dtype="f8", data=np.array([offset]))
        t.attrs["units"] = "seconds since 1950-01-01"
        t.attrs["calendar"] = "proleptic_gregorian"
        t.attrs["long_name"] = "Time (in UTC) indicating the end of the 10-minute interval"
        for name, index in (("station", 0), ("stationname", 4), ("wsi", 5)):
            f.create_variable(
                name,
                ("station",),
                dtype=h5py.string_dtype(),
                data=np.array([str(s[index]).encode() for s in stations], dtype=object),
            )
        for name, index in (("lat", 1), ("lon", 2), ("height", 3)):
            f.create_variable(
                name, ("station",), dtype="f8", data=np.array([float(s[index]) for s in stations])
            )
        for name, series in values.items():
            v = f.create_variable(
                name,
                ("station", "time"),
                dtype="f8",
                data=np.asarray(series, dtype="float64").reshape(n, 1),
            )
            v.attrs["_FillValue"] = np.float64("nan")
    return path


def write_day(
    root: Path,
    day: datetime,
    *,
    minutes=(0, 10, 20),
    counter: str = "0001",
    **member_kwargs,
) -> Path:
    """One daily tar holding a few members, named the way the archive names them."""
    root.mkdir(parents=True, exist_ok=True)
    nxt = day + timedelta(days=1)
    tar_path = root / (
        f"KMDS__OPER_P___10M_OBS_L2__{day:%Y%m%d}T000000_"
        f"{nxt:%Y%m%d}T000000_{counter}.tar"
    )
    staging = root / f".staging_{day:%Y%m%d}"
    staging.mkdir(exist_ok=True)
    with tarfile.open(tar_path, "w") as tar:
        for minute in minutes:
            when = day + timedelta(minutes=minute)
            name = f"KMDS__OPER_P___10M_OBS_L2_{when:%Y%m%d%H%M}.nc"
            member = write_member(staging / name, when, **member_kwargs)
            tar.add(member, arcname=name)
    return tar_path


NOW = datetime(2020, 6, 3, 12, 0, tzinfo=UTC)


def source(archive: Path, **kwargs) -> KnmiInsitu:
    kwargs.setdefault("now", NOW)
    return KnmiInsitu(
        SourceConfig(name="knmi_insitu", extra={"archive_dir": str(archive)}), **kwargs
    )


def product(**overrides) -> ProductConfig:
    """A small product config; the real YAML is exercised separately."""
    spec = {
        "source": "knmi_insitu",
        "product": "obs_10min",
        "grid_id": "knmi_1km_stereo",
        "store": StoreConfig(kind="points"),
        "variables": [
            Variable(
                name="air_temperature",
                source_key="ta",
                units="degC",
                valid_min=-60.0,
                valid_max=60.0,
            ),
            Variable(
                name="sunshine_duration",
                source_key="ss",
                units="s",
                scale=60.0,
                valid_min=0.0,
                valid_max=600.0,
            ),
        ],
    }
    spec.update(overrides)
    return ProductConfig(**spec)


def raw_of(path: Path, key: str = "knmi_insitu/obs_10min/20200601") -> RawFile:
    return RawFile(path=path, sha256="0" * 64, n_bytes=path.stat().st_size, unit_key=key)


def window(start: datetime, days: float = 1.0) -> TimeWindow:
    return TimeWindow(start, start + timedelta(days=days))


# ----------------------------------------------------------------------- naming


def test_a_member_name_maps_to_its_ten_minute_slot():
    assert member_time("KMDS__OPER_P___10M_OBS_L2_202006011200.nc") == datetime(
        2020, 6, 1, 12, 0, tzinfo=UTC
    )
    assert member_time("KMDS__OPER_P___10M_OBS_L2_202604112350.nc") == datetime(
        2026, 4, 11, 23, 50, tzinfo=UTC
    )
    # A path works too: members are addressed by their name inside the tar.
    assert member_time("some/dir/KMDS__OPER_P___10M_OBS_L2_201201010000.nc").hour == 0


@pytest.mark.parametrize(
    "bad",
    [
        "KMDS__OPER_P___10M_OBS_L2_2020060112.nc",     # too short
        "KMDS__OPER_P___10M_OBS_L2_202006011200.nc4",  # wrong suffix
        "202006011200.nc",                             # no prefix
        "",
    ],
)
def test_a_name_that_is_not_a_member_says_what_it_should_have_been(bad):
    with pytest.raises(ValueError, match="KMDS member|YYYYMMDDHHMM"):
        member_time(bad)


def test_a_tar_name_maps_to_the_day_it_covers():
    assert tar_date(
        "KMDS__OPER_P___10M_OBS_L2__20200601T000000_20200602T000000_0001.tar"
    ) == datetime(2020, 6, 1).date()


def test_station_ids_keep_the_width_the_file_wrote():
    """"06260" is five characters in the file and in its WIGOS id.

    Re-padding or stripping here would silently split one station in two
    against a WIGOS-keyed feed, which keys on `0-20000-0-06260`.
    """
    assert station_id("06260") == "06260"
    assert station_id(b" 06260 ") == "06260"
    assert station_id("06260") != station_id("6260")


# ------------------------------------------------------------------ variables


def test_a_structural_variable_is_not_an_observation():
    """`lat`/`station`/`time` are read automatically; asking for one is a bug."""
    for key in ("lat", "station", "time"):
        with pytest.raises(ValueError, match="structural"):
            variables_by_key(
                product(variables=[Variable(name="x", source_key=key, units="1")])
            )


def test_the_variable_map_is_the_whole_source_specific_knowledge():
    pairs = variables_by_key(product())
    assert [key for key, _ in pairs] == ["ta", "ss"]
    assert [var.name for _, var in pairs] == ["air_temperature", "sunshine_duration"]


# ---------------------------------------------------------------- fingerprints


def test_the_fingerprint_ignores_order_and_the_structural_variables():
    """Both normalisations are load bearing.

    netCDF variable order is an implementation detail, and the coordinates and
    station table are in every file by construction — including them would make
    the fingerprint say nothing about what is measured.
    """
    a = schema_fingerprint(["ta", "ff", "dd"])
    b = schema_fingerprint(["dd", "ta", "ff"])
    c = schema_fingerprint(["ta", "ff", "dd", *STRUCTURAL])
    assert a == b == c


def test_the_fingerprint_moves_when_a_variable_appears():
    """Exactly the 2021 change: `tg` and `Q24H` arrive and the hash must notice."""
    base = schema_fingerprint(["ta", "ff"])
    assert base != schema_fingerprint(["ta", "ff", "tg"])
    assert base != schema_fingerprint(["ta"])
    assert base != schema_fingerprint(["ta", "ffs"])
    assert len(base) == 16


def test_the_fingerprint_is_stable_across_runs():
    """A literal, so a change to the hashing shows up as a failing test."""
    assert schema_fingerprint(["ta", "ff", "dd"]) == "619915b6c127653b"


# ---------------------------------------------------------------------- list


def test_a_whole_day_is_exactly_one_unit(tmp_path):
    """The reason `RawLayout(archive="tar", granularity="day")` exists.

    KNMI publishes 144 files a day and meters on request count. If a day were
    144 units the ledger would be 750k rows for this archive and the fetch
    would be undownloadable; a day is the tar and the tar is the unit.
    """
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC), minutes=range(0, 60, 10))
    units = source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC)), product())
    assert len(units) == 1
    unit = units[0]
    assert unit.key == "knmi_insitu/obs_10min/20200601"
    assert unit.meta["archive"] == "tar"
    assert unit.meta["granularity"] == "day"
    assert unit.meta["filename"].endswith(".tar")
    assert unit.valid_time == datetime(2020, 6, 1, tzinfo=UTC)


def test_one_unit_per_day_across_a_multi_day_window(tmp_path):
    archive = tmp_path / "archive"
    for day in (1, 2, 3):
        write_day(archive, datetime(2020, 6, day, tzinfo=UTC))
    units = source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC), days=3), product())
    assert [u.key[-8:] for u in units] == ["20200601", "20200602", "20200603"]


def test_a_window_ending_at_midnight_does_not_claim_the_next_day(tmp_path):
    archive = tmp_path / "archive"
    for day in (1, 2):
        write_day(archive, datetime(2020, 6, day, tzinfo=UTC))
    units = source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC), days=1), product())
    assert [u.key[-8:] for u in units] == ["20200601"]


def test_unit_keys_are_stable_across_runs(tmp_path):
    """Re-running a job must not re-copy 25 MB a day, so the key cannot drift."""
    archive = tmp_path / "archive"
    for day in (1, 2):
        write_day(archive, datetime(2020, 6, day, tzinfo=UTC))
    first = [u.key for u in source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC), 2),
                                                 product())]
    second = [u.key for u in source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC), 2),
                                                  product())]
    assert first == second == [
        "knmi_insitu/obs_10min/20200601",
        "knmi_insitu/obs_10min/20200602",
    ]


def test_a_day_the_archive_does_not_have_is_skipped_not_emitted(tmp_path, caplog):
    """The archive has 5211 tars for ~5215 days; the gaps are real."""
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    write_day(archive, datetime(2020, 6, 3, tzinfo=UTC))
    units = source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC), days=3), product())
    assert [u.key[-8:] for u in units] == ["20200601", "20200603"]
    assert "absent from the archive" in caplog.text


def test_today_is_volatile_and_a_closed_day_is_not(tmp_path):
    """A closed day's tar is immutable; today's is still being written.

    The key is deliberately stable per day, so without the flag the ledger would
    mark a half-written day done and freeze the live tail there.
    """
    archive = tmp_path / "archive"
    for day in (1, 3):
        write_day(archive, datetime(2020, 6, day, tzinfo=UTC))
    units = source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC), days=3), product())
    flags = {u.key[-8:]: u.meta["volatile"] for u in units}
    assert flags == {"20200601": False, "20200603": True}   # NOW is 2020-06-03


def test_a_re_issued_day_keeps_the_highest_counter(tmp_path, caplog):
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC), counter="0001")
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC), counter="0002")
    units = source(archive).list(window(datetime(2020, 6, 1, tzinfo=UTC)), product())
    assert len(units) == 1
    assert units[0].meta["filename"].endswith("_0002.tar")
    assert "several tars" in caplog.text


def test_an_absent_archive_directory_is_empty_not_an_exception(tmp_path):
    assert source(tmp_path / "nope").list(window(datetime(2020, 6, 1, tzinfo=UTC)),
                                          product()) == []


# ---------------------------------------------------------------------- fetch


def test_fetch_puts_the_whole_tar_in_the_raw_layer_never_loose_members(tmp_path):
    """The raw layer holds the day's tar verbatim; `parse` reads members out of it."""
    archive = tmp_path / "archive"
    tar_path = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    src = source(archive)
    unit = src.list(window(datetime(2020, 6, 1, tzinfo=UTC)), product())[0]

    raw = src.fetch(unit, tmp_path / "raw")
    assert raw is not None
    assert raw.path.suffix == ".tar"
    assert raw.path.parent.name == "2020"
    assert raw.unit_key == unit.key
    assert raw.n_bytes == tar_path.stat().st_size
    assert list((tmp_path / "raw").rglob("*.nc")) == []
    # It really is the archive's tar, readable as one.
    with tarfile.open(raw.path) as tar:
        assert len(tar.getmembers()) == 3


def test_fetch_is_idempotent_and_re_hashes_rather_than_re_copying(tmp_path):
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    src = source(archive)
    unit = src.list(window(datetime(2020, 6, 1, tzinfo=UTC)), product())[0]
    first = src.fetch(unit, tmp_path / "raw")
    second = src.fetch(unit, tmp_path / "raw")
    assert first.sha256 == second.sha256 == _sha256(first.path)
    assert not list((tmp_path / "raw").rglob("*.part"))


def test_fetch_returns_none_when_the_day_has_gone_from_the_archive(tmp_path):
    """The local equivalent of a 404: a skip, not a failed job."""
    archive = tmp_path / "archive"
    tar_path = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    src = source(archive)
    unit = src.list(window(datetime(2020, 6, 1, tzinfo=UTC)), product())[0]
    tar_path.unlink()
    assert src.fetch(unit, tmp_path / "raw") is None


def test_fetch_never_touches_the_network(tmp_path, monkeypatch):
    """There is no HTTP in this source at all.

    Every other source here is tested against `inspect.signature(fetch_file)`
    because a drifted call crashes on the first real download. This one cannot
    drift, because it never calls it — the archive is already on disk and a unit
    is a hardlink. Asserting that is the equivalent guarantee: the day it starts
    downloading, this test says so.
    """
    def explode(*args, **kwargs):
        raise AssertionError("knmi_insitu must not download; the archive is local")

    monkeypatch.setattr(http, "fetch_file", explode)
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    src = source(archive)
    unit = src.list(window(datetime(2020, 6, 1, tzinfo=UTC)), product())[0]
    assert src.fetch(unit, tmp_path / "raw") is not None


def test_a_source_built_with_no_session_still_works(tmp_path):
    """Constructing without a session must not be a latent AttributeError."""
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    src = KnmiInsitu(SourceConfig(name="knmi_insitu", extra={"archive_dir": str(archive)}))
    unit = src.list(window(datetime(2020, 6, 1, tzinfo=UTC)), product())[0]
    assert src.fetch(unit, tmp_path / "raw") is not None


def test_fetch_tolerates_a_unit_with_no_meta(tmp_path):
    archive = tmp_path / "archive"
    tar_path = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    unit = Unit(key="knmi_insitu/obs_10min/20200601", url=f"file://{tar_path}")
    raw = source(archive).fetch(unit, tmp_path / "raw")
    assert raw is not None and raw.path.name == "20200601.tar"


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------- parse


def test_parse_reads_members_out_of_the_tar(tmp_path):
    """Nothing is extracted to disk; 144 loose files a day is the whole problem."""
    archive = tmp_path / "archive"
    tar_path = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC), minutes=(0, 10, 20))
    frame = source(archive).parse(raw_of(tar_path), product())

    assert frame.columns == ["time", "station_id", "air_temperature", "sunshine_duration"]
    assert frame.height == 6                       # 3 members x 2 stations
    assert frame["time"].dtype == pl.Datetime("us", "UTC")
    assert frame["time"].to_list()[0] == datetime(2020, 6, 1, 0, 0, tzinfo=UTC)
    assert frame["time"].to_list()[-1] == datetime(2020, 6, 1, 0, 20, tzinfo=UTC)
    assert sorted(frame["station_id"].unique().to_list()) == ["06260", "06380"]
    assert list(tmp_path.rglob("*.nc")) == list((archive / ".staging_20200601").glob("*.nc"))


def test_the_timestamp_comes_from_the_file_not_the_member_name(tmp_path):
    """The member name orders members; the `time` variable timestamps rows.

    They agree in the archive, so this fixture deliberately disagrees: if the
    reader ever falls back to the name, this is the only thing that catches it.
    """
    archive = tmp_path / "archive"
    root = archive
    root.mkdir(parents=True)
    staging = tmp_path / "staging"
    staging.mkdir()
    name = "KMDS__OPER_P___10M_OBS_L2_202006010000.nc"
    member = write_member(staging / name, datetime(2020, 6, 1, 7, 30, tzinfo=UTC))
    tar_path = root / (
        "KMDS__OPER_P___10M_OBS_L2__20200601T000000_20200602T000000_0001.tar"
    )
    with tarfile.open(tar_path, "w") as tar:
        tar.add(member, arcname=name)
    frame = source(archive).parse(raw_of(tar_path), product(extra={"drop_off_cadence": False}))
    assert frame["time"].to_list() == [datetime(2020, 6, 1, 7, 30, tzinfo=UTC)] * 2


def test_nan_becomes_null_before_scale_is_applied(tmp_path):
    """Order matters, and NaN is this feed's fill value.

    polars treats a float NaN as a value, not a null, so a reader that lets it
    through writes NaN into the store and poisons every mean taken over it.
    Scaling first would also turn a configured sentinel into a believable number.
    """
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive,
        datetime(2020, 6, 1, tzinfo=UTC),
        minutes=(0,),
        values={"ta": [5.5, np.nan], "ss": [np.nan, 4.0], "rg": [0.0, 0.0]},
    )
    frame = source(archive).parse(raw_of(tar_path), product()).sort("station_id")
    assert frame["air_temperature"].to_list() == pytest.approx([5.5, None], nan_ok=True)
    assert frame["air_temperature"][1] is None
    assert frame["sunshine_duration"][0] is None           # not NaN, and not 0
    assert frame["sunshine_duration"][1] == pytest.approx(240.0)   # 4 min -> 240 s
    assert frame["air_temperature"].is_nan().sum() == 0


def test_a_numeric_sentinel_is_nulled_before_scale_too(tmp_path):
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive,
        datetime(2020, 6, 1, tzinfo=UTC),
        minutes=(0,),
        values={"ta": [5.5, -999.0], "ss": [-999.0, 1.0], "rg": [0.0, 0.0]},
    )
    cfg = product(extra={"missing_sentinels": [-999.0]})
    frame = source(archive).parse(raw_of(tar_path), cfg).sort("station_id")
    assert frame["air_temperature"][1] is None
    # -999 min scaled by 60 would be -59940 s, which the valid range would also
    # have caught -- but only by accident, and only for this variable.
    assert frame["sunshine_duration"][0] is None


def test_the_valid_range_is_applied_after_scale_and_in_canonical_units(tmp_path):
    """`ss` is minutes upstream and seconds here; 0..600 only makes sense after."""
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive,
        datetime(2020, 6, 1, tzinfo=UTC),
        minutes=(0,),
        values={"ta": [5.5, 999.0], "ss": [9.0, 20.0], "rg": [0.0, 0.0]},
    )
    frame = source(archive).parse(raw_of(tar_path), product()).sort("station_id")
    assert frame["sunshine_duration"][0] == pytest.approx(540.0)   # 9 min is fine
    assert frame["sunshine_duration"][1] is None                   # 20 min is not
    assert frame["air_temperature"][1] is None                     # 999 degC is not


def test_stations_outside_the_grid_footprint_are_dropped(tmp_path):
    """Rome is in no sensible bounding box of the Dutch radar domain."""
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive,
        datetime(2020, 6, 1, tzinfo=UTC),
        minutes=(0,),
        stations=(DE_BILT, ROME),
        values={"ta": [5.5, 25.0], "ss": [0.0, 0.0], "rg": [0.0, 0.0]},
    )
    frame = source(archive).parse(raw_of(tar_path), product())
    assert frame["station_id"].to_list() == ["06260"]


def test_a_null_grid_id_is_the_escape_hatch(tmp_path):
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive, datetime(2020, 6, 1, tzinfo=UTC), minutes=(0,), stations=(DE_BILT, ROME)
    )
    frame = source(archive).parse(raw_of(tar_path), product(grid_id=None))
    assert sorted(frame["station_id"].to_list()) == ["06260", "16239"]


def test_off_cadence_observations_are_dropped_rather_than_snapped(tmp_path):
    archive = tmp_path / "archive"
    root = archive
    root.mkdir(parents=True)
    staging = tmp_path / "staging"
    staging.mkdir()
    tar_path = root / (
        "KMDS__OPER_P___10M_OBS_L2__20200601T000000_20200602T000000_0001.tar"
    )
    with tarfile.open(tar_path, "w") as tar:
        for minute, offset in ((0, 0), (10, 3)):      # 00:10:03 is off the grid
            when = datetime(2020, 6, 1, 0, minute, offset, tzinfo=UTC)
            name = f"KMDS__OPER_P___10M_OBS_L2_{when:%Y%m%d%H%M}.nc"
            tar.add(write_member(staging / name, when), arcname=name)
    frame = source(archive).parse(raw_of(tar_path), product())
    assert frame["time"].to_list() == [datetime(2020, 6, 1, tzinfo=UTC)] * 2


def test_a_config_wanting_nothing_yields_a_correctly_typed_empty_frame(tmp_path):
    archive = tmp_path / "archive"
    tar_path = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    frame = source(archive).parse(raw_of(tar_path), product(variables=[]))
    assert frame.is_empty()
    assert frame.columns == ["time", "station_id"]


def test_a_partial_member_is_read_for_what_it_has(tmp_path, caplog):
    """KNMI disseminates members carrying only some of the variables.

    Real: 2018-07-27 09:20 has the wind and pressure variables and none of the
    temperature ones. Refusing it discarded the whole day - 144 members for the
    sake of one. The variables that are present are real observations, so they
    are read and the rest are null, which is what null means.
    """
    import logging

    archive = tmp_path / "archive"
    tar_path = write_day(
        archive, datetime(2020, 6, 1, tzinfo=UTC), values={"ta": [1.0, 2.0]}
    )
    with caplog.at_level(logging.WARNING):
        frame = source(archive).parse(raw_of(tar_path), product())

    # Three members in the tar, two stations each: every partial member is read.
    assert frame.height == 6, "every partial member's observations must survive"
    assert sorted(set(frame["air_temperature"].drop_nulls().to_list())) == [1.0, 2.0]
    missing_column = next(
        v.name for v in product().variables if v.name != "air_temperature"
    )
    assert frame[missing_column].null_count() == frame.height, (
        "a variable the member lacks is null, not zero"
    )
    assert any("reading the" in r.message for r in caplog.records), caplog.text


def test_a_member_with_nothing_we_want_is_still_a_loud_failure(tmp_path):
    """A member with none of the configured variables is not a partial member,
    it is a different kind of file, and reading it would be guessing."""
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive, datetime(2020, 6, 1, tzinfo=UTC), values={}
    )
    with pytest.raises(SchemaDriftError, match="none of the configured variables"):
        source(archive).parse(raw_of(tar_path), product())


# --------------------------------------------------------- the pinned catalogue


def test_a_file_carrying_a_subset_of_the_catalogue_passes(tmp_path):
    """2012 files have 100 variables and 2026 files 102; both must parse."""
    archive = tmp_path / "archive"
    tar_path = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    cfg = product(extra={"known_variables": ["ta", "ss", "rg", "tg", "Q24H"]})
    assert source(archive).parse(raw_of(tar_path), cfg).height == 6


def test_a_variable_outside_the_catalogue_refuses_to_parse(tmp_path):
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive,
        datetime(2020, 6, 1, tzinfo=UTC),
        values={"ta": [1.0, 2.0], "ss": [0.0, 0.0], "rg": [0.0, 0.0], "tg": [3.0, 4.0]},
    )
    cfg = product(extra={"known_variables": ["ta", "ss", "rg"]})
    with pytest.raises(SchemaDriftError, match="tg"):
        source(archive).parse(raw_of(tar_path), cfg)


def test_editing_the_catalogue_without_repinning_its_hash_fails_loudly(tmp_path):
    archive = tmp_path / "archive"
    tar_path = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    cfg = product(
        schema_fingerprint="0000000000000000",
        extra={"known_variables": ["ta", "ss", "rg"]},
    )
    with pytest.raises(SchemaDriftError, match="config is inconsistent"):
        source(archive).parse(raw_of(tar_path), cfg)


# -------------------------------------------------------------------- stations


def test_stations_carry_coordinates_a_height_and_the_wigos_id(tmp_path):
    """`pack` writes this table; without it the store has no coordinates at all."""
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    table = source(archive).stations(product())
    assert table.columns == ["station_id", "lat", "lon", "height", "name", "wigos_id"]
    assert table["station_id"].to_list() == ["06260", "06380"]
    row = table.row(0, named=True)
    assert (row["lat"], row["lon"]) == (52.0989, 5.1797)
    assert row["height"] == pytest.approx(1.9)
    assert row["name"] == "De Bilt"
    # The WIGOS id is what joins this store against WIGOS-keyed feeds without a second lookup.
    assert row["wigos_id"] == "0-20000-0-06260"


def test_the_station_table_is_unioned_across_days_not_taken_from_one(tmp_path):
    """The network grows: 62 stations in 2012, 71 in 2026 in the real archive.

    A table read from a single file silently drops every station commissioned
    after it, and every downstream matrix loses those columns without a word.
    """
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC), stations=(DE_BILT,))
    write_day(archive, datetime(2020, 6, 2, tzinfo=UTC), stations=(DE_BILT, MAASTRICHT))
    table = source(archive).stations(product())
    assert table["station_id"].to_list() == ["06260", "06380"]


def test_stations_are_clipped_to_the_grid_footprint(tmp_path):
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC), stations=(DE_BILT, ROME))
    assert source(archive).stations(product())["station_id"].to_list() == ["06260"]
    assert source(archive).stations(product(grid_id=None))["station_id"].to_list() == [
        "06260",
        "16239",
    ]


def test_an_unreadable_tar_does_not_take_out_the_station_table(tmp_path, caplog):
    archive = tmp_path / "archive"
    write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    broken = archive / (
        "KMDS__OPER_P___10M_OBS_L2__20200602T000000_20200603T000000_0001.tar"
    )
    broken.write_bytes(b"not a tar")
    assert source(archive).stations(product())["station_id"].to_list() == ["06260", "06380"]
    assert "cannot read stations" in caplog.text


# ------------------------------------------------------------------------ scan


def test_scan_reports_a_pinnable_catalogue_and_a_stable_fingerprint(tmp_path):
    archive = tmp_path / "archive"
    old = write_day(
        archive, datetime(2020, 6, 1, tzinfo=UTC), values={"ta": [1.0, 2.0]}
    )
    new = write_day(
        archive,
        datetime(2021, 6, 1, tzinfo=UTC),
        values={"ta": [1.0, 2.0], "tg": [3.0, 4.0]},
    )
    report = KnmiInsitu().scan([raw_of(old), raw_of(new)])
    assert report["source"] == "knmi_insitu"
    assert report["n_files"] == 2
    assert report["n_variants"] == 2                 # the real 2021 change, in miniature
    # The union is what a config pins: 2020 is a strict subset of 2021, so
    # pinning either variant would reject the other half of the archive.
    assert report["known_variables"] == ["ta", "tg"]
    assert report["fingerprint"] == schema_fingerprint(["ta", "tg"])
    assert report["errors"] == []
    assert KnmiInsitu().scan([raw_of(old), raw_of(new)])["fingerprint"] == report["fingerprint"]


def test_scan_records_a_bad_file_as_a_finding_rather_than_crashing(tmp_path):
    archive = tmp_path / "archive"
    good = write_day(archive, datetime(2020, 6, 1, tzinfo=UTC))
    bad = archive / "broken.tar"
    bad.write_bytes(b"not a tar at all")
    report = KnmiInsitu().scan([raw_of(good), raw_of(bad)])
    assert len(report["errors"]) == 1
    assert "broken.tar" in report["errors"][0]["path"]
    assert report["known_variables"]


# --------------------------------------------------------------------- configs


def test_the_source_yaml_loads_into_sourceconfig():
    cfg = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    assert cfg.name == "knmi_insitu"
    assert cfg.base_url.startswith("file://")       # a local archive, not an endpoint
    assert cfg.credentials_env == []
    assert cfg.extra["link_raw"] is True


def test_the_product_yaml_loads_into_productconfig():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert cfg.store_id == "knmi_insitu/obs_10min/v1"
    assert cfg.store.kind == "points"
    assert cfg.grid_id == "knmi_1km_stereo"
    assert cfg.cadence_seconds == 600
    # The motivating case for the tar layout.
    assert (cfg.raw_layout.archive, cfg.raw_layout.granularity) == ("tar", "day")


def test_every_configured_source_key_is_in_the_pinned_catalogue():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    known = set(cfg.extra["known_variables"])
    assert cfg.variables
    for key, var in variables_by_key(cfg):
        assert key in known, f"{var.name} reads {key!r}, which is not in known_variables"


def test_the_pinned_catalogue_hashes_to_the_pinned_fingerprint():
    """Exactly the check `parse` makes, so a half-done re-pin fails here first."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert schema_fingerprint(cfg.extra["known_variables"]) == cfg.schema_fingerprint


def test_the_shipped_config_parses_a_real_shaped_member(tmp_path):
    """End to end on the shipped config, not the trimmed test one."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    archive = tmp_path / "archive"
    tar_path = write_day(
        archive,
        datetime(2020, 6, 1, tzinfo=UTC),
        minutes=(0,),
        values={
            "ta": [5.5, 7.0], "td": [3.0, 4.0], "rh": [84.0, 80.0],
            "p0": [1013.0, 1012.0], "pp": [1020.0, 1019.0],
            "rg": [0.6, 0.0], "R1H": [1.2, 0.0], "dr": [120.0, 0.0],
            "ff": [4.0, 5.0], "dd": [180.0, 200.0], "fx": [9.0, 11.0],
            "vv": [24200.0, 50000.0], "ss": [10.0, 0.0], "qg": [800.0, 750.0],
        },
    )
    frame = source(archive).parse(raw_of(tar_path), cfg)
    assert frame.height == 2
    assert frame["air_temperature"].dtype == pl.Float32
    assert frame["sunshine_duration"][0] == pytest.approx(600.0)      # 10 min -> 600 s
    # rg is a mean RATE in mm/h; the canonical column is the interval total.
    assert frame["precipitation_amount"][0] == pytest.approx(0.1, abs=1e-4)
    assert frame["precipitation_amount_1h"][0] == pytest.approx(1.2)
    assert frame["precipitation_duration"][0] == pytest.approx(120.0)


# -------------------------------------------------------------------- contract


def test_knmi_insitu_satisfies_the_source_protocol():
    assert isinstance(KnmiInsitu(), Source)


# ------------------------------------------------------- against the real archive


@pytest.mark.skipif(not REAL_ARCHIVE.is_dir(), reason="the KNMI tar archive is not mounted")
def test_the_real_archive_is_daily_tars_of_144_members():
    src = KnmiInsitu(SourceConfig(name="knmi_insitu", extra={"archive_dir": str(REAL_ARCHIVE)}))
    archive = src.archive()
    assert len(archive) > 5000
    day = datetime(2020, 6, 1).date()
    with tarfile.open(archive[day]) as tar:
        members = [m.name for m in tar.getmembers() if m.isfile()]
    assert len(members) == 144                     # one per ten minutes
    assert member_time(min(members)).hour == 0
    assert member_time(max(members)) == datetime(2020, 6, 1, 23, 50, tzinfo=UTC)


@pytest.mark.skipif(not REAL_ARCHIVE.is_dir(), reason="the KNMI tar archive is not mounted")
def test_the_pinned_catalogue_covers_both_real_schema_eras():
    """2012 has 100 variables and 2026 has 102; one pinned catalogue holds both."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    src = KnmiInsitu(SourceConfig(name="knmi_insitu", extra={"archive_dir": str(REAL_ARCHIVE)}))
    archive = src.archive()
    known = set(cfg.extra["known_variables"])
    seen: list[set[str]] = []
    for day in (datetime(2012, 1, 1).date(), datetime(2026, 4, 11).date()):
        path = archive.get(day)
        if path is None:
            pytest.skip(f"{day} missing from the archive")
        with tarfile.open(path) as tar:
            name = min(m.name for m in tar.getmembers() if m.isfile())
            data = tar.extractfile(name).read()
        with h5netcdf.File(io.BytesIO(data), "r") as f:
            seen.append(set(f.variables) - set(STRUCTURAL))
    assert seen[0] < seen[1], "2026 should be a strict superset of 2012"
    assert seen[1] - seen[0] == {"tg", "Q24H"}
    for era in seen:
        assert era <= known


@pytest.mark.skipif(not REAL_ARCHIVE.is_dir(), reason="the KNMI tar archive is not mounted")
def test_a_real_days_station_table_is_in_domain_and_carries_wigos_ids():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    src = KnmiInsitu(SourceConfig(name="knmi_insitu", extra={"archive_dir": str(REAL_ARCHIVE)}))
    table = src.stations(
        cfg, window=TimeWindow(datetime(2020, 6, 1, tzinfo=UTC),
                               datetime(2020, 6, 8, tzinfo=UTC))
    )
    assert table.height > 50
    assert table["lat"].min() > 40.0 and table["lat"].max() < 60.0
    assert table["wigos_id"].str.starts_with("0-").all()
    assert "06260" in table["station_id"].to_list()        # De Bilt
