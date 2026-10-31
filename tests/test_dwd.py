"""Tests for the DWD 10-minute station source.

Nothing here touches the network. The fixtures are verbatim excerpts captured
from opendata.dwd.de on 2026-10-31 (a `zehn_min_tu_Beschreibung_Stationen.txt`
catalogue and the `produkt_*.txt` inside two `now` station ZIPs), kept as
latin-1 byte literals so the encoding is genuinely exercised rather than
assumed. They are small enough to live beside the assertions that read them,
which keeps the fixture and the expected values from drifting apart.
"""
from __future__ import annotations

import io
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from meteopack.core.config import ProductConfig, SourceConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.dwd import (
    GROUPS_BY_KEY,
    DwdStations,
    SchemaDriftError,
    parse_station_description,
    product_fingerprint,
    schema_fingerprint,
    station_id,
    variables_by_group,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "dwd.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "dwd" / "stations_10min.yaml"


# --------------------------------------------------------------------- fixtures

# Verbatim rows, latin-1. Note what each one is here to prove:
#   00044 Großenkneten          - a non-ASCII name; inside the KNMI footprint
#   00096 Neuruppin-Alt Ruppin  - a space in the name *and* a two-digit longitude,
#                                 which shifts every field right by one column
#   04189 Altheim, Kreis Bib... - a comma in the name
#   00377 Bergzabern, Bad       - a comma plus a trailing word
# plus a header, a dashes rule misaligned with its own data rows, a blank line
# and a truncated row that must be skipped rather than raise.
STATION_DESC = (
    b"Stations_id von_datum bis_datum Stationshoehe geoBreite geoLaenge "
    b"Stationsname Bundesland Abgabe\r\n"
    b"----------- --------- --------- ------------- --------- --------- "
    b"----------------------------------------- ---------- ------\r\n"
    b"00044 20070208 20261031             44     52.9336    8.2370 "
    b"Gro\xdfenkneten                             Niedersachsen"
    b"                            Frei\r\n"
    b"00096 20190409 20261031             50     52.9437   12.8518 "
    b"Neuruppin-Alt Ruppin                     Brandenburg"
    b"                              Frei\r\n"
    b"00161 20110830 20261031             75     50.4237    7.4202 "
    b"Andernach                                Rheinland-Pfalz"
    b"                          Frei\r\n"
    b"00377 20030325 20261031            210     49.1070    7.9967 "
    b"Bergzabern, Bad                          Rheinland-Pfalz"
    b"                          Frei\r\n"
    b"04189 20030806 20261031            534     48.1479    9.4596 "
    b"Altheim, Kreis Biberach                  Baden-W\xfcrttemberg"
    b"                        Frei\r\n"
    b"\r\n"
    b"05100 2009\r\n"
)

PRODUKT_TU = (
    b"STATIONS_ID;MESS_DATUM;  QN;PP_10;TT_10;TM5_10;RF_10;TD_10;eor\r\n"
    b"         44;202610310950;    3;   -999;   9.6;  10.3;  96.6;   9.1;eor\r\n"
    b"         44;202610311000;    3;   -999;   9.4;  10.4;  94.0;   8.5;eor\r\n"
    b"         44;202610311010;    3;   -999;   9.6;  10.6;  94.6;   8.8;eor\r\n"
    b"         44;202610311020;    3;   -999;   9.7;  10.9;  93.5;   8.7;eor\r\n"
)

PRODUKT_SOLAR = (
    b"STATIONS_ID;MESS_DATUM;  QN;DS_10;GS_10;SD_10;LS_10;eor\r\n"
    b"        183;202610310950;    2;   9.5;  11.0;   0.017;  19.6;eor\r\n"
    b"        183;202610311000;    2;   9.4;  17.7;   0.150;  17.6;eor\r\n"
    b"        183;202610311010;    2;  11.2;  12.5;   0.017;  20.4;eor\r\n"
)

# Same dialect, deliberately out of range / off cadence / pre-2000.
PRODUKT_TU_EDGES = (
    b"STATIONS_ID;MESS_DATUM;  QN;PP_10;TT_10;TM5_10;RF_10;TD_10;eor\r\n"
    b"         44;199906301200;    3;  980.5;   9.6;  10.3; 155.0;   9.1;eor\r\n"
    b"         44;199906301205;    3;  980.4;   9.4;  10.4;  94.0;   8.5;eor\r\n"
    b"         44;200001010000;    3; 1013.2;  -1.5;  -2.0;  94.6;  -2.8;eor\r\n"
)


def _zip(tmp_path: Path, name: str, member: str, payload: bytes) -> RawFile:
    """A station ZIP on disk, as `fetch` would have left it."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(member, payload)
    return RawFile(path=path, sha256="0" * 64, n_bytes=path.stat().st_size, unit_key=name)


@pytest.fixture
def product() -> ProductConfig:
    return ProductConfig.model_validate(yaml.safe_load(PRODUCT_YAML.read_text()))


@pytest.fixture
def source_cfg() -> SourceConfig:
    return SourceConfig.model_validate(yaml.safe_load(SOURCE_YAML.read_text()))


@pytest.fixture
def src(source_cfg: SourceConfig) -> DwdStations:
    # A fixed clock so subdir selection is not a function of the wall time.
    return DwdStations(source_cfg, now=datetime(2026, 10, 31, 12, tzinfo=UTC))


@pytest.fixture
def offline(monkeypatch, src: DwdStations) -> DwdStations:
    """Serve the station catalogue and directory listings from memory."""

    def fake_get(url: str, *, encoding: str = "utf-8") -> str:
        if url.endswith("Beschreibung_Stationen.txt"):
            return STATION_DESC.decode(encoding)
        if url.endswith("/air_temperature/now/"):
            return (
                '<a href="10minutenwerte_TU_00044_now.zip">x</a>'
                '<a href="10minutenwerte_TU_00096_now.zip">x</a>'
                '<a href="10minutenwerte_TU_04189_now.zip">x</a>'
                '<a href="../">up</a>'
                '<a href="zehn_now_tu_Beschreibung_Stationen.txt">desc</a>'
            )
        if url.endswith("/air_temperature/historical/"):
            return (
                '<a href="10minutenwerte_TU_00044_20100101_20191231_hist.zip">x</a>'
                '<a href="10minutenwerte_TU_00044_20200101_20231231_hist.zip">x</a>'
                '<a href="10minutenwerte_TU_00161_20100101_20191231_hist.zip">x</a>'
            )
        return ""

    monkeypatch.setattr(src, "_get_text", fake_get)
    return src


# ----------------------------------------------------- station-description parsing


def test_station_description_handles_latin1_and_names_with_spaces():
    records = {r.station_id: r for r in parse_station_description(STATION_DESC.decode("latin-1"))}

    assert records["00044"].name == "Großenkneten"
    assert records["04189"].state == "Baden-Württemberg"
    # A space in the name survives; splitting on whitespace would not.
    assert records["00096"].name == "Neuruppin-Alt Ruppin"
    # ...and the two-digit longitude in that same row still parses correctly,
    # which is the fixed-width trap: every field after it shifts one column.
    assert records["00096"].lon == pytest.approx(12.8518)
    assert records["00044"].lon == pytest.approx(8.2370)
    # Commas in names must not be mistaken for a field separator.
    assert records["00377"].name == "Bergzabern, Bad"
    assert records["04189"].name == "Altheim, Kreis Biberach"


def test_station_description_skips_header_blank_and_truncated_rows():
    records = parse_station_description(STATION_DESC.decode("latin-1"))
    # Five real rows; the header, the dashes rule, the blank line and the
    # truncated "05100 2009" tail are all dropped without raising.
    assert [r.station_id for r in records] == ["00044", "00096", "00161", "00377", "04189"]


def test_station_description_parses_dates_and_height():
    rec = parse_station_description(STATION_DESC.decode("latin-1"))[0]
    assert rec.from_date == date(2007, 2, 8)
    assert rec.to_date == date(2026, 10, 31)
    assert rec.height_m == pytest.approx(44.0)
    assert rec.lat == pytest.approx(52.9336)


def test_station_description_ignores_an_empty_document():
    assert parse_station_description("") == []


def test_covers_is_a_half_open_overlap_test():
    rec = parse_station_description(STATION_DESC.decode("latin-1"))[0]  # 2007-02-08 .. 2026-10-31
    assert rec.covers(datetime(2010, 1, 1, tzinfo=UTC), datetime(2011, 1, 1, tzinfo=UTC))
    assert not rec.covers(datetime(1990, 1, 1, tzinfo=UTC), datetime(1991, 1, 1, tzinfo=UTC))


def test_station_id_is_five_digits_however_dwd_pads_it():
    assert station_id("44") == station_id(" 44 ") == station_id(44) == "00044"


# ------------------------------------------------------------------- the contract


def test_dwd_stations_satisfies_the_source_protocol(source_cfg):
    assert isinstance(DwdStations(source_cfg), Source)
    assert isinstance(DwdStations(), Source)
    assert DwdStations.name == "dwd"


# ----------------------------------------------------------------- domain clipping


def test_stations_are_clipped_to_the_grid_footprint(offline, product):
    table = offline.stations(product)
    ids = table["station_id"].to_list()

    # Großenkneten (8.24E) and Andernach (7.42E) are inside the KNMI radar
    # footprint; Neuruppin (12.85E) and Altheim (9.46E, 48.15N) are not.
    assert "00044" in ids
    assert "00161" in ids
    assert "00096" not in ids
    assert "04189" not in ids
    assert table["groups"].to_list()[0] == ",".join(sorted(variables_by_group(product)))


def test_stations_without_a_grid_keep_the_whole_network(offline, product):
    unclipped = product.model_copy(update={"grid_id": None})
    assert len(offline.stations(unclipped)) == 5


def test_stations_can_be_restricted_to_a_reporting_window(offline, product):
    window = TimeWindow(datetime(1995, 1, 1, tzinfo=UTC), datetime(1996, 1, 1, tzinfo=UTC))
    # Both in-domain stations started reporting well after 1995.
    assert offline.stations(product, window=window).is_empty()


# ------------------------------------------------------------------------- list


def test_list_emits_one_unit_per_in_domain_station_zip(offline, product):
    window = TimeWindow(datetime(2026, 10, 31, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))
    only_tu = product.model_copy(
        update={
            "variables": [v for v in product.variables if v.source_key.startswith("air_temp")],
            "extra": {**product.extra, "subdirs": ["now"]},
        }
    )
    units = offline.list(window, only_tu)

    assert [u.meta["station_id"] for u in units] == ["00044"]
    assert units[0].url.endswith("/air_temperature/now/10minutenwerte_TU_00044_now.zip")
    # The ledger keys off this, so it must not contain a timestamp or a path.
    assert units[0].key == "dwd/stations_10min/air_temperature/now/10minutenwerte_TU_00044_now"


def test_list_keys_are_stable_across_runs(offline, product):
    window = TimeWindow(datetime(2026, 10, 31, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))
    only_tu = product.model_copy(
        update={
            "variables": [v for v in product.variables if v.source_key.startswith("air_temp")],
            "extra": {**product.extra, "subdirs": ["now"]},
        }
    )
    first = [u.key for u in offline.list(window, only_tu)]
    offline._listing_cache.clear()
    offline._description_cache.clear()
    assert [u.key for u in offline.list(window, only_tu)] == first


def test_list_drops_historical_archives_outside_the_window(offline, product):
    window = TimeWindow(datetime(2012, 1, 1, tzinfo=UTC), datetime(2013, 1, 1, tzinfo=UTC))
    only_tu = product.model_copy(
        update={
            "variables": [v for v in product.variables if v.source_key.startswith("air_temp")],
            "extra": {**product.extra, "subdirs": ["historical"]},
        }
    )
    units = offline.list(window, only_tu)
    names = sorted(u.meta["filename"] for u in units)

    # 00161 is in domain, 00096 is not; the 2020-2023 archive misses the window.
    assert names == [
        "10minutenwerte_TU_00044_20100101_20191231_hist.zip",
        "10minutenwerte_TU_00161_20100101_20191231_hist.zip",
    ]
    assert units[0].valid_time == datetime(2010, 1, 1, tzinfo=UTC)


def test_subdir_selection_follows_the_window(src, product):
    old = TimeWindow(datetime(2012, 1, 1, tzinfo=UTC), datetime(2013, 1, 1, tzinfo=UTC))
    assert src._subdirs(old.start, old.end, product) == ["historical"]

    today = TimeWindow(datetime(2026, 10, 31, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))
    assert src._subdirs(today.start, today.end, product) == ["recent", "now"]

    backfill = TimeWindow(datetime(2012, 1, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))
    assert src._subdirs(backfill.start, backfill.end, product) == ["historical", "recent", "now"]


def test_unknown_subdir_is_rejected(src, product):
    cfg = product.model_copy(update={"extra": {**product.extra, "subdirs": ["yesterday"]}})
    with pytest.raises(ValueError, match="unknown DWD subdir"):
        src._subdirs(datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 2, 1, tzinfo=UTC), cfg)


# ------------------------------------------------------------------------ fetch


def _install_http(monkeypatch, handler):
    """Stand in for the download in `meteopack.core.http`.

    The stub binds the REAL signature before calling the handler, and raises
    the REAL NotFoundError, so this test double cannot drift away from the
    module it stands in for — that drift already shipped a bug once, where
    fetch_file grew a required `unit_key` that dwd.fetch never passed and every
    suite stayed green.
    """
    import inspect

    from meteopack.core import http

    real_sig = inspect.signature(http.fetch_file)

    def fetch_file(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return handler(bound.arguments["url"], Path(bound.arguments["dest"]), http.NotFoundError)

    monkeypatch.setattr(http, "fetch_file", fetch_file)
    return http.NotFoundError


def test_fetch_delegates_to_core_http_and_lays_files_out_by_group(monkeypatch, src, tmp_path):
    seen = {}

    def handler(url, dest, _exc):
        seen["url"] = url
        seen["dest"] = dest
        return RawFile(path=dest, sha256="a" * 64, n_bytes=7, unit_key="k")

    _install_http(monkeypatch, handler)
    unit = Unit(
        key="dwd/stations_10min/wind/now/10minutenwerte_wind_00011_now",
        url="https://example.invalid/wind/now/10minutenwerte_wind_00011_now.zip",
        meta={
            "group": "wind",
            "subdir": "now",
            "station_id": "00011",
            "filename": "10minutenwerte_wind_00011_now.zip",
        },
    )

    raw = src.fetch(unit, tmp_path)
    assert raw.n_bytes == 7
    assert seen["url"] == unit.url
    assert seen["dest"] == tmp_path / "wind" / "now" / "10minutenwerte_wind_00011_now.zip"


def test_fetch_skips_a_missing_station_period(monkeypatch, src, tmp_path):
    def handler(url, dest, exc):
        raise exc(url, "not found", 404)

    _install_http(monkeypatch, handler)
    unit = Unit(
        key="k",
        url="https://example.invalid/solar/now/10minutenwerte_SOLAR_00044_now.zip",
        meta={
            "group": "solar",
            "subdir": "now",
            "station_id": "00044",
            "filename": "10minutenwerte_SOLAR_00044_now.zip",
        },
    )
    # A station that reports temperature but not solar is normal, not an error.
    assert src.fetch(unit, tmp_path) is None


# ------------------------------------------------------------------------ parse


def _tu_raw(tmp_path: Path, payload: bytes = PRODUKT_TU) -> RawFile:
    return _zip(
        tmp_path,
        "10minutenwerte_TU_00044_now.zip",
        "produkt_zehn_now_tu_20261031_20261031_00044.txt",
        payload,
    )


def test_parse_returns_time_station_and_the_configured_variables(tmp_path, src, product):
    df = src.parse(_tu_raw(tmp_path), product)

    assert df.columns == [
        "time",
        "station_id",
        "air_temperature",
        "dew_point_temperature",
        "relative_humidity",
        "air_pressure",
    ]
    assert df.schema["time"] == pl.Datetime("us", "UTC")
    assert df["station_id"].to_list() == ["00044"] * 4
    assert df["time"][0] == datetime(2026, 10, 31, 9, 50, tzinfo=UTC)
    assert df["air_temperature"][0] == pytest.approx(9.6, abs=1e-4)
    # TM5_10 is in the file but not in the config: it must not be read.
    assert "TM5_10" not in df.columns


def test_parse_turns_the_minus_999_sentinel_into_null(tmp_path, src, product):
    df = src.parse(_tu_raw(tmp_path), product)
    # PP_10 is -999 in every row of this station's file.
    assert df["air_pressure"].null_count() == 4
    assert df["air_temperature"].null_count() == 0


def test_parse_applies_scale_before_the_valid_range(tmp_path, src, product):
    raw = _zip(
        tmp_path,
        "10minutenwerte_SOLAR_00183_now.zip",
        "produkt_zehn_now_sd_20261031_20261031_00183.txt",
        PRODUKT_SOLAR,
    )
    df = src.parse(raw, product)

    # SD_10 is published in hours; 0.150 h -> 540 s, inside the 0..600 s range.
    assert df["sunshine_duration"].to_list() == pytest.approx([61.2, 540.0, 61.2], abs=1e-3)
    # GS_10 is J/cm2 per 10 min; 17.7 -> 295 W/m2.
    assert df["surface_downwelling_shortwave_flux"][1] == pytest.approx(295.0, abs=0.1)


def test_parse_applies_offset(tmp_path, src, product):
    kelvin = Variable(
        name="air_temperature_k",
        source_key="air_temperature:TT_10",
        units="K",
        offset=273.15,
        valid_min=180.0,
        valid_max=340.0,
    )
    cfg = product.model_copy(update={"variables": [kelvin]})
    df = src.parse(_tu_raw(tmp_path), cfg)
    assert df["air_temperature_k"][0] == pytest.approx(282.75, abs=1e-3)


def test_parse_masks_values_outside_the_valid_range(tmp_path, src, product):
    raw = _tu_raw(tmp_path, PRODUKT_TU_EDGES)
    cfg = product.model_copy(update={"extra": {**product.extra, "drop_off_cadence": False}})
    df = src.parse(raw, cfg)

    # RF_10 = 155.0 % is above valid_max=100 and must become null, while the
    # other humidity values in the same column survive.
    assert df["relative_humidity"].to_list()[0] is None
    assert df["relative_humidity"].to_list()[1] == pytest.approx(94.0, abs=1e-4)
    # The sentinel is nulled before scaling, so -999 never lands in range.
    assert df["air_pressure"][2] == pytest.approx(1013.2, abs=1e-3)


def test_parse_shifts_pre_2000_historical_timestamps_from_mez_to_utc(tmp_path, src, product):
    raw = _zip(
        tmp_path,
        "10minutenwerte_TU_00044_19990101_20001231_hist.zip",
        "produkt_zehn_min_tu_19990101_20001231_00044.txt",
        PRODUKT_TU_EDGES,
    )
    cfg = product.model_copy(update={"extra": {**product.extra, "drop_off_cadence": False}})
    times = src.parse(raw, cfg)["time"].to_list()

    assert times[0] == datetime(1999, 6, 30, 11, 0, tzinfo=UTC)   # 12:00 MEZ
    assert times[-1] == datetime(2000, 1, 1, 0, 0, tzinfo=UTC)    # already UTC


def test_pre_2000_shift_can_be_switched_off(tmp_path, src, product):
    raw = _zip(
        tmp_path,
        "10minutenwerte_TU_00044_19990101_20001231_hist.zip",
        "produkt_zehn_min_tu_19990101_20001231_00044.txt",
        PRODUKT_TU_EDGES,
    )
    cfg = product.model_copy(
        update={"extra": {**product.extra, "pre_2000_mez": False, "drop_off_cadence": False}}
    )
    assert src.parse(raw, cfg)["time"][0] == datetime(1999, 6, 30, 12, 0, tzinfo=UTC)


def test_parse_drops_observations_off_the_cadence_grid(tmp_path, src, product):
    raw = _tu_raw(tmp_path, PRODUKT_TU_EDGES)
    cfg = product.model_copy(update={"extra": {**product.extra, "pre_2000_mez": False}})
    times = src.parse(raw, cfg)["time"].to_list()
    # 1999-06-30 12:05 is not a multiple of the 600 s cadence.
    assert datetime(1999, 6, 30, 12, 5, tzinfo=UTC) not in times
    assert len(times) == 2


def test_parse_ignores_groups_the_config_does_not_ask_for(tmp_path, src, product):
    raw = _zip(
        tmp_path,
        "10minutenwerte_extrema_temp_00044_now.zip",
        "produkt_zehn_now_tx_20261031_20261031_00044.txt",
        b"STATIONS_ID;MESS_DATUM;  QN;TX_10;TX5_10;TN_10;TN5_10;eor\r\n"
        b"         44;202610310950;    3;   8.1;  -999;   7.9;   5.6;eor\r\n",
    )
    df = src.parse(raw, product)
    assert df.is_empty()
    assert df.columns == ["time", "station_id"]


def test_parse_refuses_a_file_whose_columns_moved(tmp_path, src, product):
    tampered = PRODUKT_TU.replace(b"TD_10", b"TD_NEU")
    with pytest.raises(SchemaDriftError, match="air_temperature schema changed"):
        src.parse(_tu_raw(tmp_path, tampered), product)


def test_parse_reports_a_missing_requested_column(tmp_path, src, product):
    trimmed = (
        b"STATIONS_ID;MESS_DATUM;  QN;PP_10;TT_10;TM5_10;RF_10;eor\r\n"
        b"         44;202610310950;    3;   -999;   9.6;  10.3;  96.6;eor\r\n"
    )
    cfg = product.model_copy(update={"extra": {}})   # no pinned fingerprints
    with pytest.raises(SchemaDriftError, match="missing column"):
        src.parse(_tu_raw(tmp_path, trimmed), cfg)


def test_parse_rejects_a_source_key_without_a_group(tmp_path, src, product):
    cfg = product.model_copy(
        update={
            "variables": [Variable(name="t", source_key="TT_10", units="degC")],
            "extra": {},
        }
    )
    with pytest.raises(ValueError, match="must be '<group>:<DWD column>'"):
        src.parse(_tu_raw(tmp_path), cfg)


# -------------------------------------------------------------------- fingerprint


def test_fingerprint_is_stable_and_ignores_padding_order_and_eor():
    a = schema_fingerprint(["STATIONS_ID", "MESS_DATUM", "QN", "TT_10", "eor"])
    b = schema_fingerprint(["  QN", "TT_10", "MESS_DATUM ", "STATIONS_ID", "eor"])
    assert a == b == schema_fingerprint(["STATIONS_ID", "MESS_DATUM", "QN", "TT_10"])
    assert len(a) == 16


def test_fingerprint_moves_when_a_column_is_added_or_renamed():
    base = ["STATIONS_ID", "MESS_DATUM", "QN", "TT_10", "eor"]
    assert schema_fingerprint(base) != schema_fingerprint([*base, "TD_10"])
    assert schema_fingerprint(base) != schema_fingerprint(
        ["STATIONS_ID", "MESS_DATUM", "QN", "TT_NEU", "eor"]
    )


def test_pinned_fingerprints_match_the_live_schema_captured_in_the_fixtures():
    # These are the values written into configs/products/dwd/stations_10min.yaml.
    assert schema_fingerprint(
        PRODUKT_TU.split(b"\r\n")[0].decode("latin-1").split(";")
    ) == "267791e758f02936"
    assert schema_fingerprint(
        PRODUKT_SOLAR.split(b"\r\n")[0].decode("latin-1").split(";")
    ) == "cf8690abb391903d"


def test_product_fingerprint_is_order_independent():
    a = {"wind": "aaaa", "solar": "bbbb"}
    assert product_fingerprint(a) == product_fingerprint({"solar": "bbbb", "wind": "aaaa"})


# --------------------------------------------------------------------------- scan


def test_scan_reports_one_variant_per_distinct_column_set(tmp_path, src):
    files = [
        _tu_raw(tmp_path / "a"),
        _tu_raw(tmp_path / "b"),
        _zip(
            tmp_path,
            "10minutenwerte_SOLAR_00183_now.zip",
            "produkt_zehn_now_sd_20261031_20261031_00183.txt",
            PRODUKT_SOLAR,
        ),
    ]
    report = src.scan(files)

    assert report["n_files"] == 3
    assert report["n_variants"] == 2
    by_group = {v["group"]: v for v in report["variants"]}
    assert by_group["air_temperature"]["count"] == 2
    assert by_group["air_temperature"]["fingerprint"] == "267791e758f02936"
    assert by_group["solar"]["product_code"] == "zehn_now_sd"
    assert by_group["solar"]["columns"][:3] == ["STATIONS_ID", "MESS_DATUM", "QN"]
    # QN is an integer code, TT_10 a float, eor a literal string.
    assert by_group["air_temperature"]["dtypes"]["QN"] == "i64"
    assert by_group["air_temperature"]["dtypes"]["TT_10"] == "f64"
    assert by_group["air_temperature"]["dtypes"]["eor"] == "str"
    assert report["fingerprint"] == product_fingerprint(
        {"air_temperature": "267791e758f02936", "solar": "cf8690abb391903d"}
    )
    assert report["errors"] == []


def test_scan_separates_a_drifted_variant_of_the_same_group(tmp_path, src):
    # Upstream adds a column: same group, different column set.
    header, *rows = PRODUKT_TU.rstrip(b"\r\n").split(b"\r\n")
    drifted = b"\r\n".join(
        [header.replace(b";eor", b";NEW_10;eor")]
        + [r.replace(b";eor", b";   0.0;eor") for r in rows]
    )
    report = src.scan([_tu_raw(tmp_path / "a"), _tu_raw(tmp_path / "b", drifted)])

    assert report["n_variants"] == 2
    assert sorted(report["fingerprints"]["air_temperature"]) == sorted(
        {v["fingerprint"] for v in report["variants"]}
    )
    # Ambiguous group -> no single product fingerprint to pin.
    assert report["fingerprint"] is None


def test_scan_records_unreadable_files_instead_of_raising(tmp_path, src):
    broken = tmp_path / "10minutenwerte_TU_00044_now.zip"
    broken.write_bytes(b"not a zip")
    raw = RawFile(path=broken, sha256="0" * 64, n_bytes=9, unit_key="k")

    report = src.scan([raw])
    assert report["variants"] == []
    assert len(report["errors"]) == 1
    assert "10minutenwerte_TU_00044_now.zip" in report["errors"][0]["path"]


# ------------------------------------------------------------------------ configs


def test_source_yaml_loads_into_the_model(source_cfg):
    assert source_cfg.name == "dwd"
    assert source_cfg.base_url.endswith("/10_minutes")
    assert source_cfg.credentials_env == []
    assert 0 < source_cfg.rate_limit.requests_per_second <= 20


def test_product_yaml_loads_into_the_model(product):
    assert product.source == "dwd"
    assert product.product == "stations_10min"
    assert product.cadence_seconds == 600
    assert product.grid_id == "knmi_1km_stereo"
    assert product.store.kind == "points"
    assert product.store_id == "dwd/stations_10min/v1"


def test_every_configured_variable_points_at_a_known_group_and_column(product):
    mapping = variables_by_group(product)
    assert set(mapping) <= set(GROUPS_BY_KEY)
    names = {v.name for v in product.variables}
    assert {
        "air_temperature",
        "dew_point_temperature",
        "relative_humidity",
        "air_pressure",
        "precipitation_amount",
        "wind_speed",
        "wind_from_direction",
        "wind_speed_of_gust",
        "sunshine_duration",
        "surface_downwelling_shortwave_flux",
    } <= names
    assert len(names) == len(product.variables)   # no duplicate canonical names


def test_pinned_product_fingerprint_matches_its_per_group_parts(product):
    assert product.schema_fingerprint == product_fingerprint(product.extra["schema_fingerprints"])
    assert set(product.extra["schema_fingerprints"]) == set(variables_by_group(product))


def test_group_urls_match_the_published_layout():
    group = GROUPS_BY_KEY["air_temperature"]
    assert group.description_url("recent").endswith(
        "/air_temperature/recent/zehn_min_tu_Beschreibung_Stationen.txt"
    )
    # The `now` directory names its catalogue differently.
    assert group.description_url("now").endswith(
        "/air_temperature/now/zehn_now_tu_Beschreibung_Stationen.txt"
    )
    assert group.archive_url("historical", "x.zip").endswith(
        "/air_temperature/historical/x.zip"
    )


def test_fixture_round_trips_through_a_real_zip(tmp_path):
    raw = _tu_raw(tmp_path)
    with zipfile.ZipFile(raw.path) as zf:
        payload = zf.read(zf.namelist()[0])
    assert payload == PRODUKT_TU
    assert pl.read_csv(io.BytesIO(payload), separator=";", infer_schema=False).height == 4
