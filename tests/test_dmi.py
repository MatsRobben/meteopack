"""Tests for the DMI metObs 10-minute station source.

Nothing here touches the network. The fixtures are verbatim excerpts captured
from opendataapi.dmi.dk on 2026-10-31 — a `station/items` feature and a slice
of an `observation/items` page for station 06041 — with a handful of rows
added, each labelled, to exercise an edge the live slice happens not to
contain. They are small enough to live beside the assertions that read them,
which keeps the fixture and the expected values from drifting apart.

The local raw archive is only used by the tests guarded on its presence, and
only for its station catalogue: the archive predates this reader and stores
compacted `[observed, value]` pairs rather than the GeoJSON pages `fetch` now
writes, so it cannot exercise `parse`.
"""
from __future__ import annotations

import copy
import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from meteopack.core import grids
from meteopack.core.config import ProductConfig, SourceConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.stations.dmi import (
    PAGE_LIMIT,
    DmiStations,
    PageOverflowError,
    SchemaDriftError,
    parse_stations,
    product_fingerprint,
    schema_fingerprint,
    station_id,
    unit_key,
    variables_by_parameter,
    windows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "stations" / "dmi.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "stations" / "dmi" / "stations_10min.yaml"

LOCAL_ARCHIVE = Path(
    "/vol/knmimo-nobackup/users/mrobben/meteopack_data/raw/stations/dmi/stations_10min"
)
LOCAL_CATALOGUE = LOCAL_ARCHIVE / "station_catalog.json"


# --------------------------------------------------------------------- fixtures


def _station_feature(sid: str, lon: float, lat: float, name: str, **over) -> dict:
    """One `station/items` feature, shaped exactly as the live API writes one."""
    props = {
        "owner": "DMI",
        "country": "DNK",
        "operationFrom": "2018-11-26T00:00:00Z",
        "parameterId": ["temp_dry", "humidity", "precip_past10min", "precip_past1h"],
        "created": "2026-10-14T10:51:09Z",
        "validFrom": "2018-11-26T00:00:00Z",
        "type": "Synop",
        "stationHeight": 14.0,
        "name": name,
        "stationId": sid,
        "validTo": None,
        "status": "Active",
    }
    props.update(over)
    return {
        "type": "Feature",
        "id": f"feature-{sid}",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": props,
    }


# Real coordinates, from the live catalogue:
#   05355 Emmerlev Klev - southern Jutland, INSIDE the KNMI radar footprint
#   06041 Skagen        - the northern tip, OUTSIDE it
#   05408 H. C. Andersen Airport - a Pluvio, listed twice under two validity
#                         periods, each advertising a different parameter set.
#                         The union is what the station measures; the later
#                         feature is where it measures it.
STATIONS_PAGE: dict = {
    "type": "FeatureCollection",
    "features": [
        _station_feature("05355", 8.6652, 54.9872, "Emmerlev Klev", stationHeight=8.0),
        _station_feature("06041", 10.6316, 57.7364, "Skagen Fyr", stationHeight=4.0),
        _station_feature(
            "05408",
            10.3305,
            55.4749,
            "H. C. Andersen Airport (old site)",
            type="Pluvio",
            validFrom="2015-01-01T00:00:00Z",
            parameterId=["precip_past10min"],
        ),
        _station_feature(
            "05408",
            10.3310,
            55.4750,
            "H. C. Andersen Airport",
            type="Pluvio",
            validFrom="2018-11-26T00:00:00Z",
            parameterId=["precip_dur_past10min", "precip_past1min"],
        ),
        # Malformed on purpose: no geometry. Must be skipped, not raise.
        {"type": "Feature", "id": "broken", "properties": {"stationId": "09999"}},
    ],
}


def _observation(pid: str, observed: str, value, *, sid: str = "06041") -> dict:
    return {
        "type": "Feature",
        "id": f"{pid}-{observed}",
        "geometry": {"type": "Point", "coordinates": [10.6316, 57.7364]},
        "properties": {
            "parameterId": pid,
            "created": "2026-10-30T00:01:38.040805Z",
            "value": value,
            "observed": observed,
            "stationId": sid,
        },
    }


# Verbatim values from the live page for 06041 on 2026-10-30, plus four
# labelled additions. Several parameters share each `observed`, which is the
# point: metObs writes one feature per value and `parse` has to put them back
# on a (time, station) grid.
OBSERVATION_PAGE: dict = {
    "type": "FeatureCollection",
    "numberReturned": 15,
    "features": [
        _observation("temp_dry", "2026-10-30T00:00:00Z", 12.0),
        _observation("humidity", "2026-10-30T00:00:00Z", 86.95),
        _observation("precip_past10min", "2026-10-30T00:00:00Z", 0.0),
        _observation("sun_last10min_glob", "2026-10-30T00:00:00Z", 0.0),
        _observation("cloud_cover", "2026-10-30T00:00:00Z", 0.0),
        # An hourly aggregate. metObs returns these alongside the 10-minute
        # parameters; the product config does not ask for them, so they must
        # not reach the frame.
        _observation("temp_mean_past1h", "2026-10-30T00:00:00Z", 12.11),
        _observation("temp_dry", "2026-10-30T00:10:00Z", 11.8),
        _observation("humidity", "2026-10-30T00:10:00Z", 88.05),
        _observation("precip_past10min", "2026-10-30T00:10:00Z", 0.0),
        _observation("sun_last10min_glob", "2026-10-30T00:10:00Z", 0.0),
        _observation("cloud_cover", "2026-10-30T00:10:00Z", 0.0),
        # ADDED: a JSON null, which is how metObs says "sensor reported nothing".
        _observation("temp_dry", "2026-10-30T00:20:00Z", None),
        # ADDED: 9 minutes of sunshine in a 10-minute window. Published in
        # minutes, so 540 s once scaled: inside the range. Its neighbour below
        # is 20 minutes, which is impossible and only looks impossible after
        # scaling - the whole reason the range is checked post-scale.
        _observation("sun_last10min_glob", "2026-10-30T00:20:00Z", 9.0),
        _observation("sun_last10min_glob", "2026-10-30T00:30:00Z", 20.0),
        # ADDED: off the 10-minute grid. Dropped, not snapped.
        _observation("temp_dry", "2026-10-30T00:37:00Z", 11.5),
    ],
}


@pytest.fixture
def product() -> ProductConfig:
    return ProductConfig.model_validate(yaml.safe_load(PRODUCT_YAML.read_text()))


@pytest.fixture
def source_cfg() -> SourceConfig:
    return SourceConfig.model_validate(yaml.safe_load(SOURCE_YAML.read_text()))


@pytest.fixture
def src(source_cfg: SourceConfig) -> DmiStations:
    # A fixed clock so the volatile flag is not a function of the wall time.
    return DmiStations(source_cfg, now=datetime(2026, 10, 31, 12, tzinfo=UTC))


@pytest.fixture
def offline(monkeypatch, src: DmiStations) -> DmiStations:
    """Serve the station catalogue from memory."""
    monkeypatch.setattr(src, "_get_json", lambda url: copy.deepcopy(STATIONS_PAGE))
    return src


def _page(tmp_path: Path, payload: dict, name: str = "2025-2027.json") -> RawFile:
    """A fetched page on disk, as `fetch` would have left it."""
    path = tmp_path / "06041" / "temp_dry" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return RawFile(
        path=path,
        sha256="0" * 64,
        n_bytes=path.stat().st_size,
        unit_key="dmi/stations_10min/06041/temp_dry/2025-2027",
    )


# ------------------------------------------------------------------- the contract


def test_dmi_stations_satisfies_the_source_protocol(source_cfg):
    assert isinstance(DmiStations(source_cfg), Source)
    assert isinstance(DmiStations(), Source)  # also with no config at all


def test_both_yamls_load_into_their_config_models():
    source = SourceConfig.model_validate(yaml.safe_load(SOURCE_YAML.read_text()))
    product = ProductConfig.model_validate(yaml.safe_load(PRODUCT_YAML.read_text()))
    assert source.name == "dmi"
    assert product.source == "dmi" and product.product == "stations_10min"
    assert product.cadence_seconds == 600
    assert product.store.kind == "points"


def test_the_product_asks_only_for_ten_minute_parameters(product):
    """The hourly, 12 h and 24 h aggregates are excluded by omission.

    `list` only requests what the variable map names, so a stray `_past1h`
    here would quietly double the download and pollute the store with a
    different accumulation period under a 10-minute column name.
    """
    keys = set(variables_by_parameter(product))
    assert not [k for k in keys if "past1h" in k or "past12h" in k or "past24h" in k]
    # Advertised by stations but never populated through this endpoint.
    assert "precip_past1min" not in keys
    assert {"temp_dry", "precip_past10min", "wind_speed"} <= keys


def test_a_parameter_claimed_twice_is_rejected(product):
    cfg = product.model_copy(
        update={
            "variables": [
                Variable(name="a", source_key="temp_dry", units="degC"),
                Variable(name="b", source_key="temp_dry", units="degC"),
            ]
        }
    )
    with pytest.raises(ValueError, match="claimed by both"):
        variables_by_parameter(cfg)


# ------------------------------------------------------------------ station ids


def test_station_id_is_five_digits_however_dmi_pads_it():
    assert station_id("6041") == station_id(" 6041 ") == station_id(6041) == "06041"
    assert station_id("06041") == "06041"


# -------------------------------------------------------------- station parsing


def test_station_features_union_their_parameters_across_validity_periods():
    """A station is listed once per validity period, each with its own list.

    Taking only one feature loses whatever the other period measured, and the
    station then never gets asked for it.
    """
    records = {r.station_id: r for r in parse_stations(STATIONS_PAGE)}

    assert set(records["05408"].parameters) == {
        "precip_past10min",
        "precip_dur_past10min",
        "precip_past1min",
    }
    # ...while position and name come from the most recent feature only.
    assert records["05408"].name == "H. C. Andersen Airport"
    assert records["05408"].lon == pytest.approx(10.3310)


def test_station_features_without_a_geometry_are_skipped_not_raised():
    records = parse_stations(STATIONS_PAGE)
    assert "09999" not in {r.station_id for r in records}
    assert [r.station_id for r in records] == ["05355", "05408", "06041"]


def test_station_parsing_tolerates_an_empty_document():
    assert parse_stations({"type": "FeatureCollection", "features": []}) == []
    assert parse_stations(None) == []


# -------------------------------------------------------------------- stations


def test_stations_clips_to_the_grid_footprint(offline, product):
    table = offline.stations(product)

    assert table["station_id"].to_list() == ["05355"]
    # Skagen is real, measures what we want, and sits 300 km north of the
    # radar footprint; 05408's only wanted parameter is precip_past10min and
    # it is also outside.
    assert "06041" not in table["station_id"].to_list()


def test_stations_carries_what_pack_needs_to_place_a_station(offline, product):
    """`pack` writes this beside the observations.

    Without lat/lon the store has values and no coordinates, and every
    downstream station matrix silently drops its columns instead of failing.
    """
    table = offline.stations(product)

    assert {"station_id", "lat", "lon", "height_m", "name"} <= set(table.columns)
    row = table.row(0, named=True)
    assert row["lat"] == pytest.approx(54.9872)
    assert row["lon"] == pytest.approx(8.6652)
    assert row["height_m"] == pytest.approx(8.0)
    assert row["name"] == "Emmerlev Klev"
    assert table.schema["station_id"] == pl.String


def test_stations_keeps_the_whole_network_without_a_grid(offline, product):
    cfg = product.model_copy(update={"grid_id": None})
    assert set(offline.stations(cfg)["station_id"].to_list()) == {"05355", "05408", "06041"}


def test_stations_is_empty_but_typed_when_the_catalogue_fails(monkeypatch, src, product):
    def boom(url):
        raise RuntimeError("DNS is having a day")

    monkeypatch.setattr(src, "_get_json", boom)
    table = src.stations(product)
    assert table.is_empty()
    assert table.schema["lat"] == pl.Float64


# ------------------------------------------------------------------- windows


def test_windows_are_anchored_so_a_unit_key_does_not_depend_on_the_backfill():
    """Two differently sliced backfills must agree on which window holds 2014.

    Blocking from the requested start instead would make the same observations
    two different ledger entries, and they would be fetched twice from a source
    that manages 0.23 files a second.
    """
    a = windows(datetime(2012, 5, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC), years=3)
    b = windows(datetime(2014, 7, 1, tzinfo=UTC), datetime(2015, 1, 1, tzinfo=UTC), years=3)

    labels_a = {w.label for w in a}
    labels_b = {w.label for w in b}
    assert labels_b <= labels_a
    # Blocks run from the 1950 anchor, so 2014 is in 2013-2015 whichever
    # backfill asked for it.
    assert "2013-2015" in labels_a and labels_b == {"2013-2015"}


def test_window_blocks_are_contiguous_and_half_open():
    blocks = windows(datetime(2012, 1, 1, tzinfo=UTC), datetime(2017, 1, 1, tzinfo=UTC), years=3)
    assert [w.label for w in blocks] == ["2010-2012", "2013-2015", "2016-2018"]
    assert blocks[0].end == blocks[1].start == datetime(2013, 1, 1, tzinfo=UTC)


# ---------------------------------------------------------------------- list


def test_list_emits_one_unit_per_station_parameter_window(offline, product):
    window = TimeWindow(datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
    units = offline.list(window, product)

    # One in-domain station (05355), advertising three wanted parameters
    # (temp_dry, humidity, precip_past10min), over the two anchored blocks the
    # window touches.
    assert len(units) == 6
    assert {u.meta["label"] for u in units} == {"2022-2024", "2025-2027"}
    assert {u.meta["parameter"] for u in units} == {
        "temp_dry",
        "humidity",
        "precip_past10min",
    }
    assert {u.meta["station_id"] for u in units} == {"05355"}
    assert all(u.url.startswith(offline.base_url) for u in units)
    assert all("parameterId=" in u.url and "stationId=" in u.url for u in units)


def test_unit_keys_are_stable_across_runs_and_across_windows(offline, product):
    wide = TimeWindow(datetime(2012, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
    narrow = TimeWindow(datetime(2013, 6, 1, tzinfo=UTC), datetime(2013, 7, 1, tzinfo=UTC))

    first = {u.key for u in offline.list(wide, product)}
    again = {u.key for u in offline.list(wide, product)}
    assert first == again

    # The ledger's identity for 2013 must not depend on how the job was sliced.
    assert {u.key for u in offline.list(narrow, product)} <= first


def test_the_unit_key_names_station_parameter_and_window(product):
    assert (
        unit_key(product, "05355", "temp_dry", "2024-2026")
        == "dmi/stations_10min/05355/temp_dry/2024-2026"
    )


def test_the_still_filling_window_is_marked_volatile(offline, product):
    """The current block keeps growing under a key that deliberately never moves.

    Without the flag the ledger files the first partial download as done and
    the rest of the block is never collected.
    """
    window = TimeWindow(datetime(2024, 1, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))
    units = offline.list(window, product)
    current = [u for u in units if u.meta["label"] == "2025-2027"]
    assert current and all(u.meta["volatile"] for u in current)

    closed = [u for u in units if u.meta["label"] == "2022-2024"]
    assert closed and not any(u.meta["volatile"] for u in closed)


def test_a_window_that_could_overflow_a_page_is_refused(offline, product):
    """A too-wide request does not fail upstream; it comes back truncated.

    300000 features at 10 minutes is a bit over five and a half years, so six
    years silently loses the tail of every window. Catching the arithmetic here
    is the only place the mistake is cheap.
    """
    cfg = product.model_copy(update={"extra": {**product.extra, "window_years": 10}})
    window = TimeWindow(datetime(2012, 1, 1, tzinfo=UTC), datetime(2016, 1, 1, tzinfo=UTC))
    with pytest.raises(ValueError, match="page cap"):
        offline.list(window, cfg)


def test_list_is_empty_when_no_station_is_in_domain(monkeypatch, src, product):
    monkeypatch.setattr(
        src,
        "_get_json",
        lambda url: {
            "type": "FeatureCollection",
            "features": [_station_feature("06041", 10.6316, 57.7364, "Skagen Fyr")],
        },
    )
    window = TimeWindow(datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC))
    assert src.list(window, product) == []


# ------------------------------------------------------------------------ fetch


def _install_http(monkeypatch, handler):
    """Stand in for the download in `meteopack.core.http`.

    The stub binds the REAL signature before calling the handler, and raises
    the REAL NotFoundError, so this test double cannot drift away from the
    module it stands in for — that drift already shipped a bug once, where
    fetch_file grew a required `unit_key` that a source never passed and every
    suite stayed green.
    """
    from meteopack.core import http

    real_sig = inspect.signature(http.fetch_file)

    def fetch_file(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return handler(bound.arguments)

    monkeypatch.setattr(http, "fetch_file", fetch_file)
    return http.NotFoundError


def test_fetch_delegates_to_core_http_with_the_unit_key(monkeypatch, src, tmp_path):
    seen = {}

    def handler(args):
        seen.update(args)
        return RawFile(path=args["dest"], sha256="a" * 64, n_bytes=7, unit_key=args["unit_key"])

    _install_http(monkeypatch, handler)
    unit = Unit(
        key="dmi/stations_10min/05355/temp_dry/2024-2026",
        url="https://example.invalid/observation/items?stationId=05355",
        meta={"station_id": "05355", "parameter": "temp_dry", "filename": "2024-2026.json"},
    )

    raw = src.fetch(unit, tmp_path)
    assert raw.n_bytes == 7
    assert seen["url"] == unit.url
    assert seen["unit_key"] == unit.key
    assert Path(seen["dest"]) == tmp_path / "05355" / "temp_dry" / "2024-2026.json"
    # The rate limits come from the source YAML, not from fetch_file's defaults.
    assert seen["retries"] == src.rate_limit.max_retries
    assert seen["backoff"] == src.rate_limit.backoff_seconds
    assert seen["timeout"] == src.timeout


def test_fetch_creates_a_session_when_the_source_was_built_without_one(monkeypatch, tmp_path):
    """A source with no session must still download.

    Passing `self._session` straight through as None crashes deep inside
    `fetch_file` instead of here.
    """
    import requests

    seen = {}

    def handler(args):
        seen.update(args)
        return RawFile(path=args["dest"], sha256="b" * 64, n_bytes=1, unit_key=args["unit_key"])

    _install_http(monkeypatch, handler)
    src = DmiStations()
    src.fetch(Unit(key="k", url="https://example.invalid/x", meta={}), tmp_path)
    assert isinstance(seen["session"], requests.Session)


def test_fetch_survives_a_unit_with_no_meta(monkeypatch, src, tmp_path):
    def handler(args):
        return RawFile(path=args["dest"], sha256="c" * 64, n_bytes=1, unit_key=args["unit_key"])

    _install_http(monkeypatch, handler)
    assert src.fetch(Unit(key="a/b/c", url="https://example.invalid/x"), tmp_path) is not None


def test_fetch_skips_a_404_rather_than_failing_the_job(monkeypatch, src, tmp_path):
    def handler(args):
        raise not_found(args["url"], "not found", 404)

    not_found = _install_http(monkeypatch, handler)
    unit = Unit(
        key="dmi/stations_10min/05355/temp_soil/2024-2026",
        url="https://example.invalid/observation/items",
        meta={"station_id": "05355", "parameter": "temp_soil", "filename": "2024-2026.json"},
    )
    # A station can advertise a parameter it never actually reported.
    assert src.fetch(unit, tmp_path) is None


def test_a_volatile_window_is_re_downloaded_rather_than_skipped(monkeypatch, src, tmp_path):
    """`fetch_file` is a no-op when the destination exists.

    That is right for a closed window and wrong for one still filling: the
    first partial harvest would otherwise be the only one that ever lands.
    """
    seen = {}

    def handler(args):
        seen["existed"] = Path(args["dest"]).exists()
        seen["overwrite"] = args["overwrite"]
        return RawFile(path=args["dest"], sha256="d" * 64, n_bytes=2, unit_key=args["unit_key"])

    _install_http(monkeypatch, handler)
    dest = tmp_path / "05355" / "temp_dry" / "2024-2026.json"
    dest.parent.mkdir(parents=True)
    dest.write_text("stale, half a window old")

    meta = {
        "station_id": "05355",
        "parameter": "temp_dry",
        "filename": "2024-2026.json",
        "volatile": True,
    }
    src.fetch(Unit(key="k", url="https://example.invalid/x", meta=meta), tmp_path)
    # Downloaded again over the old copy, which stays until the new one is
    # complete: this used to delete it first, losing it if the download failed.
    assert seen == {"existed": True, "overwrite": True}

    dest.write_text("a closed window, already correct")
    src.fetch(Unit(key="k", url="https://example.invalid/x", meta={**meta, "volatile": False}),
              tmp_path)
    assert seen == {"existed": True, "overwrite": False}


# ------------------------------------------------------------------------ parse


def test_parse_pivots_one_feature_per_value_onto_a_time_station_grid(tmp_path, src, product):
    """metObs writes one feature per (station, parameter, time, value).

    Reading that as rows would give five one-column rows per instant; the whole
    job of `parse` is to put them back side by side.
    """
    df = src.parse(_page(tmp_path, OBSERVATION_PAGE), product)

    assert df["time"].to_list() == [
        datetime(2026, 10, 30, 0, 0, tzinfo=UTC),
        datetime(2026, 10, 30, 0, 10, tzinfo=UTC),
        datetime(2026, 10, 30, 0, 20, tzinfo=UTC),
        datetime(2026, 10, 30, 0, 30, tzinfo=UTC),
    ]
    assert df["station_id"].to_list() == ["06041"] * 4
    assert df.schema["time"] == pl.Datetime("us", "UTC")
    row = df.row(0, named=True)
    assert row["air_temperature"] == pytest.approx(12.0, abs=1e-4)
    assert row["relative_humidity"] == pytest.approx(86.95, abs=1e-3)
    assert row["precipitation_amount"] == pytest.approx(0.0)
    assert row["cloud_area_fraction"] == pytest.approx(0.0)


def test_parse_ignores_parameters_the_product_did_not_ask_for(tmp_path, src, product):
    df = src.parse(_page(tmp_path, OBSERVATION_PAGE), product)
    # temp_mean_past1h is in the page and in no column.
    assert "temp_mean_past1h" not in df.columns
    assert not [c for c in df.columns if "past1h" in c]


def test_missing_becomes_null_before_anything_is_scaled(tmp_path, src, product):
    df = src.parse(_page(tmp_path, OBSERVATION_PAGE), product)
    at_0020 = df.filter(pl.col("time") == datetime(2026, 10, 30, 0, 20, tzinfo=UTC))

    assert at_0020["air_temperature"][0] is None
    # ...while the scaled neighbour in the same row is a real number: 9 minutes
    # of sunshine, published in minutes, stored in seconds.
    assert at_0020["sunshine_duration"][0] == pytest.approx(540.0)


def test_the_valid_range_is_applied_in_canonical_units(tmp_path, src, product):
    """20 minutes of sunshine in a 10-minute window is impossible.

    It only looks impossible after scaling: 20 is a perfectly ordinary raw
    number, and 1200 s is not. Checking the range before the scale would let it
    through, which is exactly the bug the ordering exists to prevent.
    """
    df = src.parse(_page(tmp_path, OBSERVATION_PAGE), product)
    at_0030 = df.filter(pl.col("time") == datetime(2026, 10, 30, 0, 30, tzinfo=UTC))
    assert at_0030["sunshine_duration"][0] is None


def test_off_cadence_observations_are_dropped_not_snapped(tmp_path, src, product):
    df = src.parse(_page(tmp_path, OBSERVATION_PAGE), product)
    assert datetime(2026, 10, 30, 0, 37, tzinfo=UTC) not in df["time"].to_list()

    cfg = product.model_copy(update={"extra": {**product.extra, "drop_off_cadence": False}})
    kept = src.parse(_page(tmp_path, OBSERVATION_PAGE), cfg)
    assert datetime(2026, 10, 30, 0, 37, tzinfo=UTC) in kept["time"].to_list()


def test_parse_restates_a_duplicated_instant_with_the_later_value(tmp_path, src, product):
    page = copy.deepcopy(OBSERVATION_PAGE)
    page["features"].append(_observation("temp_dry", "2026-10-30T00:00:00Z", 12.4))
    df = src.parse(_page(tmp_path, page), product)
    first = df.filter(pl.col("time") == datetime(2026, 10, 30, 0, 0, tzinfo=UTC))
    assert first.height == 1
    assert first["air_temperature"][0] == pytest.approx(12.4, abs=1e-4)


def test_parse_of_an_empty_page_is_empty_and_typed(tmp_path, src, product):
    df = src.parse(_page(tmp_path, {"type": "FeatureCollection", "features": []}), product)
    assert df.is_empty()
    assert df.columns == ["time", "station_id"]


def test_a_full_page_is_a_truncated_page(tmp_path, src, product):
    """A page at the cap is indistinguishable from a complete one.

    Same shape, same status, just fewer features than exist — so the only tell
    is hitting the cap exactly, and accepting it loses the tail of a window.
    """
    cfg = product.model_copy(update={"extra": {**product.extra, "page_limit": 4}})
    with pytest.raises(PageOverflowError, match="page cap"):
        src.parse(_page(tmp_path, OBSERVATION_PAGE), cfg)


def test_parse_refuses_a_record_schema_it_was_not_pinned_against(tmp_path, src, product):
    page = copy.deepcopy(OBSERVATION_PAGE)
    for feature in page["features"]:
        feature["properties"]["measurementValue"] = feature["properties"].pop("value")
    with pytest.raises(SchemaDriftError, match="record schema changed"):
        src.parse(_page(tmp_path, page), product)


def test_an_empty_page_is_not_schema_drift(tmp_path, src, product):
    """A station that reported nothing in 2014 is normal, not a failure."""
    src.parse(_page(tmp_path, {"type": "FeatureCollection", "features": []}), product)


def test_parse_rejects_something_that_is_not_a_feature_collection(tmp_path, src, product):
    with pytest.raises(SchemaDriftError):
        src.parse(_page(tmp_path, {"type": "Coverage", "ranges": {}}), product)


# ------------------------------------------------------------------ fingerprint


def test_the_fingerprint_is_stable_and_order_independent():
    keys = ["observed", "value", "parameterId", "stationId", "created"]
    assert schema_fingerprint(keys) == schema_fingerprint(keys)
    assert schema_fingerprint(keys) == schema_fingerprint(reversed(keys))
    assert schema_fingerprint(keys) == schema_fingerprint([f" {k} " for k in keys])
    assert schema_fingerprint(keys) != schema_fingerprint([*keys, "qcStatus"])


def test_the_pinned_fingerprint_matches_the_live_record_schema(product):
    """Pinned on 2026-10-31 from opendataapi.dmi.dk.

    If this fails, the API changed its feature properties and `parse` needs
    looking at before the pin is moved.
    """
    live = ["parameterId", "created", "value", "observed", "stationId"]
    assert product.schema_fingerprint == schema_fingerprint(live)


def test_product_fingerprint_rolls_up_a_mapping():
    assert product_fingerprint({"a": "1", "b": "2"}) == product_fingerprint({"b": "2", "a": "1"})
    assert product_fingerprint({"a": "1"}) != product_fingerprint({"a": "2"})


def test_scan_reports_the_record_schema_and_the_parameters_seen(tmp_path, src, product):
    report = src.scan([_page(tmp_path, OBSERVATION_PAGE)])
    assert report["n_variants"] == 1
    assert report["fingerprint"] == product.schema_fingerprint
    assert "temp_dry" in report["parameters"]
    assert report["errors"] == []


def test_scan_reports_a_bad_file_instead_of_crashing(tmp_path, src):
    path = tmp_path / "broken.json"
    path.write_text("{not json")
    raw = RawFile(path=path, sha256="0" * 64, n_bytes=9, unit_key="k")
    report = src.scan([raw])
    assert len(report["errors"]) == 1 and "broken.json" in report["errors"][0]["path"]


# --------------------------------------------------- against the local archive


@pytest.mark.skipif(not LOCAL_CATALOGUE.exists(), reason="local DMI archive not present")
def test_the_real_danish_network_clips_to_a_strict_subset_of_itself(src, product, monkeypatch):
    """Real coordinates for all 113 stations DMI publishes.

    The footprint reaches southern Jutland and Funen and nothing further north,
    so the right answer is "some, but not all". Both degenerate answers — none,
    or all of them — are failures that a bbox-only filter would produce.
    """
    catalogue = json.loads(LOCAL_CATALOGUE.read_text())
    page = {
        "type": "FeatureCollection",
        "features": [
            _station_feature(
                s["stationId"], s["lon"], s["lat"], s["name"], parameterId=s["parameterIds"]
            )
            for s in catalogue
        ],
    }
    monkeypatch.setattr(src, "_get_json", lambda url: page)
    table = src.stations(product)

    assert 0 < table.height < len(catalogue)
    grid = grids.get(product.grid_id)
    assert all(
        grid.contains(row["lon"], row["lat"]) for row in table.iter_rows(named=True)
    )
    # Emmerlev Klev, on the German border, is one of them.
    assert "05355" in table["station_id"].to_list()


@pytest.mark.skipif(not LOCAL_ARCHIVE.exists(), reason="local DMI archive not present")
def test_every_parameter_in_the_real_archive_is_one_the_product_still_wants(product):
    """The archive was built from this parameter list; drift here is a rename.

    It also re-checks the exclusion from the other direction: nothing the
    archive holds is an hourly aggregate, because nothing in the config is.
    """
    wanted = set(variables_by_parameter(product))
    found = {
        pid.name
        for station in LOCAL_ARCHIVE.iterdir()
        if station.is_dir()
        for pid in station.iterdir()
        if pid.is_dir()
    }
    assert found, "archive present but holds no parameter directories"
    assert found <= wanted


def test_the_page_limit_constant_matches_what_the_config_pins(product):
    assert product.extra["page_limit"] == PAGE_LIMIT


def test_the_compact_archive_form_reads_with_the_same_rules(tmp_path, product):
    """An earlier pipeline archived metObs as bare [[time, value], ...] lists,
    three years per file, with the station and parameter in the path rather
    than the payload. 4,775 such files cover 2012-2026 and are worth reading.

    The point of synthesising metObs-shaped features is that everything after
    the read - the pivot, the units, the valid range - is the same code, so the
    two forms cannot drift apart.
    """
    import json

    import polars as pl

    from meteopack.sources.base import RawFile
    from meteopack.sources.stations.dmi import DmiStations

    cfg = product
    param = next(v.source_key for v in cfg.variables)
    path = tmp_path / "06188" / param / "2018.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([
        ["2018-01-01T00:00:00Z", 2.3],
        ["2018-01-01T00:10:00Z", 2.2],
        ["2018-01-01T00:20:00Z", None],
    ]))

    raw = RawFile(path=path, sha256="x", n_bytes=path.stat().st_size,
                  unit_key=f"dmi/stations_10min/06188/{param}/2018")
    frame = DmiStations().parse(raw, cfg)

    assert frame.height == 3
    assert frame["station_id"].to_list() == ["06188"] * 3
    assert frame["time"].dtype == pl.Datetime("us", "UTC")
    column = next(v.name for v in cfg.variables)
    assert frame[column][2] is None, "JSON null stays null, it is not a value"


def test_a_compact_file_outside_its_directory_is_refused(tmp_path, product):
    """Nothing in the payload says what it measures, so a file moved away from
    its <station>/<parameter>/ path must not be guessed at."""
    import json

    import pytest

    from meteopack.sources.base import RawFile
    from meteopack.sources.stations.dmi import DmiStations, SchemaDriftError

    path = tmp_path / "loose.json"
    path.write_text(json.dumps([["2018-01-01T00:00:00Z", 1.0]]))
    raw = RawFile(path=path, sha256="x", n_bytes=1, unit_key="dmi/x")
    with pytest.raises(SchemaDriftError, match="station and parameter"):
        DmiStations().parse(raw, product)


def test_an_unrecognisable_payload_is_still_refused(tmp_path, product):
    import json

    import pytest

    from meteopack.sources.base import RawFile
    from meteopack.sources.stations.dmi import DmiStations, SchemaDriftError

    path = tmp_path / "06188" / "wind_speed" / "2018.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"not": "a feature collection"}))
    raw = RawFile(path=path, sha256="x", n_bytes=1, unit_key="dmi/x")
    with pytest.raises(SchemaDriftError, match="compact"):
        DmiStations().parse(raw, product)


def test_scan_does_not_report_the_compact_archive_form_as_drift(tmp_path, product):
    """`parse` skips the schema pin for the compact form - it has no property
    schema - but `scan` fingerprinted its synthesised features, so a scan of
    the 4,775-file archive failed with drift on every run (2026-11-05)."""
    import json

    from meteopack.cli import _compare_schema
    from meteopack.sources.base import RawFile
    from meteopack.sources.stations.dmi import DmiStations

    param = next(v.source_key for v in product.variables)
    path = tmp_path / "06188" / param / "2018.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps([["2018-01-01T00:00:00Z", 2.3]]))
    raw = RawFile(path=path, sha256="x", n_bytes=path.stat().st_size,
                  unit_key=f"dmi/stations_10min/06188/{param}/2018")
    report = DmiStations().scan([raw])
    assert report["compact_files"] == 1 and report["variants"] == []
    drifted, _ = _compare_schema(product, report)
    assert drifted == []
