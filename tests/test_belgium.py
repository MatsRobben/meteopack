"""Tests for the RMI Belgium AWS source (opendata.meteo.be).

Nothing here touches the network — which for this source is not a style
preference. The RMI WFS fails under concurrency and rate-limits our whole egress
IP after heavy use, so a test suite that hit it would eventually take the
cluster offline for everyone. The fixtures below are verbatim excerpts captured
on 2026-10-31 from `aws:aws_station` (GeoJSON) and `aws:aws_10min` (CSV),
including the empty-result body and the `ows:ExceptionReport` the server really
returns, with two synthetic rows added and labelled where a raw-data outlier was
needed.

The three facts most of this file exists to protect: the config must keep
`requests_per_second: 1` / `max_workers: 1` (anything else has already caused an
outage here), a unit key must be a whole station-year so a narrow backfill
cannot mark a year done, and the valid ranges must mask the nonphysical values
RMI publishes, because this feed is raw and nothing else will.
"""
from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from meteopack.core import http
from meteopack.core.config import ProductConfig, SourceConfig, StoreConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.stations.belgium import (
    DATA_LAYER,
    STATION_LAYER,
    RmiAws,
    SchemaDriftError,
    parse_stations,
    schema_fingerprint,
    station_id,
    unit_key,
    variables_by_column,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "stations" / "belgium.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "stations" / "belgium" / "aws_10min.yaml"


# --------------------------------------------------------------------- fixtures

# `?service=WFS&request=GetFeature&typeNames=aws:aws_station&outputFormat=application/json`.
# The first four features are verbatim; note what each one is here to prove:
#   6447     UCCLE   - the ordinary five-digit WMO code, and `code` is a JSON INT
#   10101002 DE HAAN - an eight-digit code, commissioned 2025: neither the width
#                      nor the lifetime of a station is what you would assume
#   6494     MONT RIGI - the eastern edge of the Belgian network
#   6484     BUZENOL - the southern edge, which the radar footprint still covers
# plus a Rome-ish feature inside any sane bounding box of the domain but outside
# the real stereographic footprint, and two malformed features that must be
# skipped rather than raise.
STATIONS = """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": "aws_station.6447",
      "geometry": {"type": "Point", "coordinates": [4.358, 50.797]},
      "geometry_name": "the_geom",
      "properties": {
        "code": 6447, "name": "UCCLE",
        "date_begin": "2003-12-01T00:00:00Z", "date_end": null,
        "altitude": 100.6, "wigos_id": "0-20000-0-06447"
      }
    },
    {
      "type": "Feature",
      "id": "aws_station.10101002",
      "geometry": {"type": "Point", "coordinates": [3.074, 51.28]},
      "properties": {
        "code": 10101002, "name": "DE HAAN",
        "date_begin": "2025-03-06T00:00:00Z", "date_end": null,
        "altitude": 3.66, "wigos_id": "0-056-101-01002"
      }
    },
    {
      "type": "Feature",
      "id": "aws_station.6494",
      "geometry": {"type": "Point", "coordinates": [6.073, 50.511]},
      "properties": {
        "code": 6494, "name": "MONT RIGI",
        "date_begin": "2001-01-23T00:00:00Z", "date_end": null,
        "altitude": 673.2, "wigos_id": "0-20000-0-06494"
      }
    },
    {
      "type": "Feature",
      "id": "aws_station.6484",
      "geometry": {"type": "Point", "coordinates": [5.587, 49.62]},
      "properties": {
        "code": 6484, "name": "BUZENOL",
        "date_begin": "2003-03-27T12:20:00Z", "date_end": null,
        "altitude": 323.9, "wigos_id": "0-20000-0-06484"
      }
    },
    {
      "type": "Feature",
      "id": "aws_station.99999",
      "geometry": {"type": "Point", "coordinates": [12.4964, 41.9028]},
      "properties": {
        "code": 99999, "name": "ROMA (not a real RMI station)",
        "date_begin": null, "date_end": null, "altitude": 130.0, "wigos_id": ""
      }
    },
    {
      "type": "Feature",
      "id": "aws_station.nocode",
      "geometry": {"type": "Point", "coordinates": [4.0, 51.0]},
      "properties": {"name": "no code at all", "altitude": 1.0}
    },
    {
      "type": "Feature",
      "id": "aws_station.nogeom",
      "geometry": {"type": "Point", "coordinates": []},
      "properties": {"code": 6001, "name": "no coordinates", "altitude": 1.0}
    }
  ]
}
"""

#: The 22 columns of a live `aws:aws_10min` CSV export, in order.
HEADER = (
    "FID,the_geom,code,timestamp,precip_quantity,temp_dry_shelter_avg,"
    "temp_grass_pt100_avg,temp_soil_avg,temp_soil_avg_5cm,temp_soil_avg_10cm,"
    "temp_soil_avg_20cm,temp_soil_avg_50cm,wind_speed_10m,wind_speed_avg_30m,"
    "wind_direction,wind_gusts_speed,humidity_rel_shelter_avg,pressure,"
    "sun_duration,short_wave_from_sky_avg,sun_int_avg,qc_flags"
)

#: `qc_flags` is a quoted JSON blob full of commas and doubled quotes — the one
#: thing in this format that a naive split(",") gets wrong. Shortened from the
#: real 17-key blob; the quoting is exactly as the server writes it.
_QC = (
    '"{""validated"":{""PRECIP_QUANTITY"":true,""WIND_SPEED_10M"":false,'
    '""PRESSURE"":true}}"'
)

# Three verbatim rows from
#   CQL_FILTER=code=6447 AND timestamp DURING 2024-06-01T00:00:00Z/...
# Note `the_geom` reads POINT (lat lon) — the data layer's geometry is SWAPPED
# relative to the station layer's GeoJSON, which is why coordinates never come
# from here. Note also `wind_speed_10m` is empty: that is how RMI writes missing.
SAMPLE_CSV = "\n".join(
    [
        HEADER,
        "aws_10min.fid-1,POINT (50.797 4.358),6447,2024-06-01T00:00:00,0,13.595142,"
        "13.159773,14.034927,14.843172,16.107541,17.081162,16.015902,,3.038952,"
        "324.24916,5.532,92.914603,1004.957453,0,0,0," + _QC,
        "aws_10min.fid-2,POINT (50.797 4.358),6447,2024-06-01T00:10:00,0,13.601369,"
        "13.335115,14.082616,14.830534,16.045455,17.043136,16.021007,,2.990411,"
        "318.738504,4.8344,93.090823,1004.957453,0,0,0," + _QC,
        "aws_10min.fid-3,POINT (50.797 4.358),6447,2024-06-01T00:20:00,0,13.583123,"
        "13.410353,14.092284,14.832049,15.990979,17.004629,16.026058,,3.985653,"
        "322.728308,8.4968,92.959239,1004.837447,0,0,0," + _QC,
        # SYNTHETIC row 4: raw-data outliers of the kind RMI really publishes
        # (a ~504 hPa pressure, a humidity just over 100, a negative night radiation),
        # plus the sunny-day sun_duration of 10 minutes verified live at Uccle
        # on 2022-07-19.
        "aws_10min.fid-4,POINT (50.797 4.358),6447,2024-06-01T00:30:00,0,13.5,"
        "13.4,14.0,14.8,16.0,17.0,16.0,,3.5,"
        "320.0,6.0,100.3,504.1,10,-3.2,0," + _QC,
        "",
    ]
)

#: A station-year with no observations: header only, HTTP 200. Verified live
#: against a nonexistent station code — "nothing there" is NOT a 404 here.
EMPTY_CSV = HEADER + "\n"

#: What GeoServer returns for a bad or overloaded request. Verbatim, HTTP 400.
WFS_ERROR = (
    '<?xml version="1.0" encoding="UTF-8"?><ows:ExceptionReport '
    'xmlns:ows="http://www.opengis.net/ows/1.1" version="2.0.0">\n'
    '  <ows:Exception exceptionCode="InvalidParameterValue" locator="GetFeature">\n'
    "    <ows:ExceptionText>Illegal property name: nosuchcolumn for feature type "
    "aws:aws_10min</ows:ExceptionText>\n"
    "  </ows:Exception>\n</ows:ExceptionReport>\n"
)

NOW = datetime(2026, 10, 31, 20, 0, tzinfo=UTC)


class FakeRmi(RmiAws):
    """The real source with its one network seam replaced."""

    def __init__(self, payload=None, **kwargs):
        kwargs.setdefault("now", NOW)
        super().__init__(SourceConfig(name="belgium"), **kwargs)
        self._payload = payload
        self.urls: list[str] = []

    def _get_json(self, url: str):
        self.urls.append(url)
        return self._payload


def product(**overrides) -> ProductConfig:
    """A small product config; the real YAML is exercised separately."""
    spec = {
        "source": "belgium",
        "product": "aws_10min",
        "grid_id": "knmi_1km_stereo",
        "store": StoreConfig(kind="points"),
        "variables": [
            Variable(
                name="air_temperature",
                source_key="temp_dry_shelter_avg",
                units="degC",
                valid_min=-60.0,
                valid_max=60.0,
            ),
            Variable(
                name="wind_speed",
                source_key="wind_speed_10m",
                units="m s-1",
                valid_min=0.0,
                valid_max=120.0,
            ),
            Variable(
                name="air_pressure",
                source_key="pressure",
                units="hPa",
                valid_min=800.0,
                valid_max=1100.0,
            ),
            Variable(
                name="sunshine_duration",
                source_key="sun_duration",
                units="s",
                scale=60.0,
                valid_min=0.0,
                valid_max=600.0,
            ),
        ],
    }
    spec.update(overrides)
    return ProductConfig(**spec)


def write(tmp_path: Path, text: str, name: str = "2024.csv") -> RawFile:
    path = tmp_path / name
    path.write_text(text)
    return RawFile(
        path=path, sha256="0" * 64, n_bytes=len(text), unit_key="belgium/aws_10min/6447/2024"
    )


def window(start: int, end: int) -> TimeWindow:
    return TimeWindow(datetime(start, 1, 1, tzinfo=UTC), datetime(end, 1, 1, tzinfo=UTC))


def stations_payload():
    return json.loads(STATIONS)


# ------------------------------------------------------------------ station ids


def test_a_station_code_is_a_string_of_whatever_width_rmi_used():
    """De Haan is `10101002`; the rest are five digits. Padding would invent one."""
    assert station_id(6447) == "6447"
    assert station_id("06447") == "06447"
    assert station_id(10101002) == "10101002"
    assert station_id(" 6494 ") == "6494"


@pytest.mark.parametrize("bad", ["", "UCCLE", "64-47", "6447.0", "0x10"])
def test_a_non_numeric_code_says_what_it_should_have_been(bad):
    with pytest.raises(ValueError, match="station code"):
        station_id(bad)


# -------------------------------------------------------------------- stations


def test_parse_stations_reads_codes_coordinates_heights_and_lifetimes():
    records = parse_stations(stations_payload())
    assert [r.station_id for r in records] == ["6447", "10101002", "6494", "6484", "99999"]
    uccle = records[0]
    # GeoJSON is (lon, lat). The DATA layer writes POINT (lat lon); never read it.
    assert (uccle.lon, uccle.lat) == (4.358, 50.797)
    assert uccle.height == pytest.approx(100.6)
    assert uccle.name == "UCCLE"
    assert uccle.wigos_id == "0-20000-0-06447"
    assert uccle.date_begin == datetime(2003, 12, 1).date()
    assert uccle.date_end is None                  # still reporting


def test_parse_stations_skips_malformed_features_rather_than_raising():
    records = parse_stations(stations_payload())
    assert len(records) == 5                       # seven features in, two unusable
    assert all(r.station_id for r in records)


def test_parse_stations_tolerates_an_empty_payload():
    assert parse_stations(None) == []
    assert parse_stations({"type": "FeatureCollection", "features": []}) == []


def test_stations_are_clipped_to_the_grid_footprint():
    """Rome is inside any bounding box of the domain and outside its real shape."""
    table = FakeRmi(stations_payload()).stations(product())
    assert table["station_id"].to_list() == ["10101002", "6447", "6484", "6494"]
    assert "99999" not in table["station_id"].to_list()


def test_stations_carry_everything_pack_needs_to_place_them():
    """`pack` writes this table; without it the store has no coordinates at all."""
    table = FakeRmi(stations_payload()).stations(product())
    assert table.columns == ["station_id", "lat", "lon", "height", "name", "wigos_id"]
    uccle = table.filter(pl.col("station_id") == "6447").row(0, named=True)
    assert (uccle["lat"], uccle["lon"]) == (50.797, 4.358)
    assert uccle["height"] == pytest.approx(100.6)
    assert uccle["name"] == "UCCLE"


def test_a_null_grid_id_is_the_escape_hatch():
    table = FakeRmi(stations_payload()).stations(product(grid_id=None))
    assert "99999" in table["station_id"].to_list()


def test_a_station_commissioned_later_is_absent_from_an_earlier_window():
    """De Haan began in 2025. Asking it for 2024 is ~350 wasted requests at 1 rps."""
    src = FakeRmi(stations_payload())
    assert "10101002" not in src.stations(product(), window=window(2024, 2025))[
        "station_id"
    ].to_list()
    assert "10101002" in src.stations(product(), window=window(2025, 2026))[
        "station_id"
    ].to_list()


def test_the_station_query_names_the_catalogue_layer():
    src = FakeRmi(stations_payload())
    src.stations(product())
    assert f"typeNames={STATION_LAYER}" in src.urls[0]
    assert "outputFormat=application%2Fjson" in src.urls[0] or (
        "outputFormat=application/json" in src.urls[0]
    )


def test_an_unavailable_station_layer_is_zero_stations_not_a_crash():
    assert FakeRmi(None).stations(product()).is_empty()
    assert FakeRmi(None).list(window(2024, 2025), product()) == []


# -------------------------------------------------------------------- unit keys


def test_a_unit_is_one_station_year():
    units = FakeRmi(stations_payload()).list(window(2023, 2025), product())
    assert {u.meta["year"] for u in units} == {"2023", "2024"}
    assert units[0].key == "belgium/aws_10min/6447/2023"
    assert units[0].valid_time == datetime(2023, 1, 1, tzinfo=UTC)
    # Four in-domain stations, of which De Haan does not exist before 2025.
    assert len(units) == 3 * 2


def test_unit_keys_are_stable_across_runs():
    """Re-running a job must not re-download, so the key cannot drift."""
    first = [u.key for u in FakeRmi(stations_payload()).list(window(2024, 2025), product())]
    second = [u.key for u in FakeRmi(stations_payload()).list(window(2024, 2025), product())]
    assert first == second
    assert first


def test_a_narrower_window_produces_the_same_key_and_the_same_url():
    """The unit is a whole year, never the slice a run happened to ask for.

    A key whose content depended on the request window would be filed as done by
    a one-week backfill and then skipped by the full-year run that followed,
    leaving eleven months that nothing will ever fetch again.
    """
    narrow = TimeWindow(datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 8, tzinfo=UTC))
    a = FakeRmi(stations_payload()).list(narrow, product())
    b = FakeRmi(stations_payload()).list(window(2024, 2025), product())
    assert {u.key for u in a} == {u.key for u in b}
    assert {u.url for u in a} == {u.url for u in b}
    assert "DURING+2024-01-01T00:00:00Z/2024-12-31T23:59:59Z" in a[0].url


def test_a_window_ending_at_new_year_does_not_claim_the_next_year():
    units = FakeRmi(stations_payload()).list(window(2024, 2025), product())
    assert {u.meta["year"] for u in units} == {"2024"}


def test_the_current_year_is_volatile_and_a_closed_one_is_not():
    """A finished year is immutable; this year is still being written.

    The key is deliberately stable, so without the flag the ledger would mark
    January's harvest as the whole of 2026 and never collect the rest.
    """
    units = FakeRmi(stations_payload()).list(window(2025, 2027), product())
    flags = {(u.meta["year"], u.meta["volatile"]) for u in units}
    assert ("2025", False) in flags
    assert ("2026", True) in flags                 # NOW is 2026-10-31


def test_unit_key_is_a_pure_function_of_its_inputs():
    cfg = product()
    assert unit_key(cfg, "6447", 2024) == "belgium/aws_10min/6447/2024"
    assert unit_key(cfg, "6447", 2025) != unit_key(cfg, "6447", 2024)


def test_the_cql_filter_names_the_station_and_the_data_layer():
    unit = FakeRmi(stations_payload()).list(window(2024, 2025), product())[0]
    assert f"typeNames={DATA_LAYER}" in unit.url
    assert "outputFormat=csv" in unit.url
    assert "code%3D6447" in unit.url
    # DURING is inclusive at both ends, so the year stops one second short of
    # the next; otherwise New Year midnight lands in two units.
    assert "2024-12-31T23:59:59Z" in unit.url


def test_a_station_filter_narrows_the_unit_list():
    units = FakeRmi(stations_payload()).list(
        window(2024, 2025), product(extra={"station_ids": ["6447"]})
    )
    assert {u.meta["station_id"] for u in units} == {"6447"}


# ------------------------------------------------------------------------ fetch


def test_fetch_binds_the_real_fetch_file_signature(monkeypatch, tmp_path):
    """Bound against the live callable, never a hand-copied parameter list.

    A source that omits `unit_key` passes every mocked test and raises TypeError
    on the first real download; that has happened here before.
    """
    real_sig = inspect.signature(http.fetch_file)
    seen: dict = {}

    def strict_fetch_file(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.update(bound.arguments)
        dest = Path(kwargs["dest"] if "dest" in kwargs else args[1])
        return RawFile(dest, "0" * 64, 0, kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", strict_fetch_file)
    src = FakeRmi(stations_payload())
    unit = src.list(window(2024, 2025), product())[0]
    raw = src.fetch(unit, tmp_path)

    assert seen["unit_key"] == unit.key
    assert seen["url"] == unit.url
    assert seen["session"] is not None             # built lazily, never None
    assert raw is not None
    # The raw tree mirrors the key: station, then year.
    assert raw.path.name == "2024.csv"
    assert raw.path.parent.name == "6447"


def test_fetch_skips_a_404_rather_than_failing_the_job(monkeypatch, tmp_path):
    def not_found(url, dest, **kwargs):
        raise http.NotFoundError(url, "404 not found", 404)

    monkeypatch.setattr(http, "fetch_file", not_found)
    unit = Unit(
        key="belgium/aws_10min/6447/2024",
        url="https://example.invalid/x",
        meta={"station_id": "6447", "year": "2024", "filename": "2024.csv"},
    )
    assert FakeRmi().fetch(unit, tmp_path) is None


def test_a_source_built_with_no_session_still_downloads(monkeypatch, tmp_path):
    """Passing `self._session` straight through when it is None crashes inside
    `fetch_file`; the session has to be built lazily."""
    captured: dict = {}

    def capture(url, dest, **kwargs):
        captured.update(kwargs)
        return RawFile(Path(dest), "0" * 64, 0, kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", capture)
    src = RmiAws(SourceConfig(name="belgium"))     # no session anywhere
    unit = Unit(key="belgium/aws_10min/6447/2024", url="https://example.invalid/x")
    src.fetch(unit, tmp_path)
    assert captured["session"] is not None
    assert "meteopack" in captured["session"].headers["User-Agent"]


def test_fetch_tolerates_a_unit_with_no_meta(monkeypatch, tmp_path):
    captured: dict = {}

    def capture(url, dest, **kwargs):
        captured["dest"] = Path(dest)
        return RawFile(Path(dest), "0" * 64, 0, kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", capture)
    unit = Unit(key="belgium/aws_10min/6447/2024", url="https://example.invalid/x")
    FakeRmi().fetch(unit, tmp_path)
    assert captured["dest"].name == "2024.csv"


# ------------------------------------------------------------------------ parse


def test_a_csv_export_becomes_a_tidy_frame(tmp_path):
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), product())
    assert frame.columns == [
        "time", "station_id", "air_temperature", "wind_speed",
        "air_pressure", "sunshine_duration",
    ]
    assert frame.height == 4
    assert frame["time"].dtype == pl.Datetime("us", "UTC")
    # The export writes a naive timestamp; it is UTC and must be labelled so.
    assert frame["time"][0] == datetime(2024, 6, 1, 0, 0, tzinfo=UTC)
    assert frame["station_id"].unique().to_list() == ["6447"]
    assert frame["air_temperature"][0] == pytest.approx(13.595142, abs=1e-5)


def test_the_quoted_qc_blob_does_not_shift_the_columns(tmp_path):
    """`qc_flags` is JSON full of commas; a naive split would misalign every row."""
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), product())
    assert frame["air_pressure"].to_list()[:3] == pytest.approx(
        [1004.957458, 1004.957458, 1004.837463], abs=1e-4
    )


def test_an_empty_field_becomes_null_before_anything_is_done_to_it(tmp_path):
    """RMI writes missing as an empty field, and `wind_speed_10m` really is empty
    in this export. Null, not zero and not NaN."""
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), product())
    assert frame["wind_speed"].null_count() == 4
    assert frame["wind_speed"].to_list() == [None] * 4


def test_a_sentinel_is_nulled_before_scale_is_applied(tmp_path):
    """Order matters: -999 minutes scaled by 60 is a believable -59940 seconds."""
    text = SAMPLE_CSV.replace(
        "320.0,6.0,100.3,504.1,10,-3.2,0,", "320.0,6.0,100.3,504.1,-999,-3.2,0,"
    )
    cfg = product(extra={"missing_sentinels": [-999.0]})
    frame = FakeRmi().parse(write(tmp_path, text), cfg)
    assert frame["sunshine_duration"][3] is None


def test_scale_is_applied_before_the_valid_range_and_in_canonical_units(tmp_path):
    """`sun_duration` is MINUTES upstream (10 at full sun, verified live at Uccle
    on 2022-07-19) and seconds here, so 0..600 only means anything post-scale."""
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), product())
    assert frame["sunshine_duration"].to_list()[:3] == pytest.approx([0.0, 0.0, 0.0])
    assert frame["sunshine_duration"][3] == pytest.approx(600.0)


def test_the_valid_range_masks_the_nonphysical_values_rmi_really_publishes(tmp_path):
    """This feed is RAW. Nothing else will catch a 504 hPa pressure."""
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), product())
    assert frame["air_pressure"][3] is None                       # 504.1 hPa
    assert frame["air_pressure"][0] == pytest.approx(1004.957458, abs=1e-4)


def test_a_humidity_just_over_one_hundred_is_kept_by_the_shipped_range(tmp_path):
    """Raw saturation reads slightly over 100; nulling it would gap every fog."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), cfg)
    assert frame["relative_humidity"][3] == pytest.approx(100.3, abs=1e-3)


def test_an_empty_export_is_an_empty_frame_not_an_error(tmp_path):
    """Header only with HTTP 200 is how this server says "nothing there"."""
    frame = FakeRmi().parse(write(tmp_path, EMPTY_CSV), product())
    assert frame.is_empty()
    assert frame.columns == [
        "time", "station_id", "air_temperature", "wind_speed",
        "air_pressure", "sunshine_duration",
    ]


def test_a_wfs_error_document_is_a_loud_failure_not_an_empty_store(tmp_path):
    """GeoServer answers a bad or overloaded request with XML.

    Read as a CSV that is a one-column file whose single row happens to be XML,
    which parses cleanly, writes nothing, and looks exactly like a station that
    never reported.
    """
    with pytest.raises(SchemaDriftError, match="WFS error document"):
        FakeRmi().parse(write(tmp_path, WFS_ERROR, "2024.csv"), product())


def test_an_empty_body_is_a_loud_failure(tmp_path):
    with pytest.raises(SchemaDriftError, match="empty body"):
        FakeRmi().parse(write(tmp_path, "   "), product())


def test_a_missing_column_says_what_was_there_instead(tmp_path):
    text = SAMPLE_CSV.replace("pressure,sun_duration", "pressure_xx,sun_duration")
    cfg = product(schema_fingerprint=None)
    with pytest.raises(SchemaDriftError, match="missing column"):
        FakeRmi().parse(write(tmp_path, text), cfg)


def test_off_cadence_observations_are_dropped_rather_than_snapped(tmp_path):
    text = SAMPLE_CSV.replace("2024-06-01T00:20:00", "2024-06-01T00:23:00")
    frame = FakeRmi().parse(write(tmp_path, text), product())
    assert frame.height == 3
    assert datetime(2024, 6, 1, 0, 23, tzinfo=UTC) not in frame["time"].to_list()


def test_a_config_wanting_nothing_yields_a_correctly_typed_empty_frame(tmp_path):
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), product(variables=[]))
    assert frame.is_empty()
    assert frame.columns == ["time", "station_id"]


def test_a_structural_column_is_not_an_observation():
    for column in ("code", "timestamp", "the_geom", "FID"):
        with pytest.raises(ValueError, match="structural"):
            variables_by_column(
                product(variables=[Variable(name="x", source_key=column, units="1")])
            )


# ----------------------------------------------------------------- fingerprints


def test_the_fingerprint_ignores_order_and_the_structural_columns():
    """FID changes between requests and the_geom is the swapped duplicate of the
    station layer; neither says anything about what the export contains."""
    a = schema_fingerprint(["code", "timestamp", "pressure"])
    b = schema_fingerprint(["pressure", "timestamp", "code"])
    c = schema_fingerprint(["code", "timestamp", "pressure", "FID", "the_geom"])
    assert a == b == c


def test_the_fingerprint_moves_when_a_column_changes():
    base = schema_fingerprint(["code", "timestamp", "pressure"])
    assert base != schema_fingerprint(["code", "timestamp", "pressure", "sun_duration"])
    assert base != schema_fingerprint(["code", "timestamp", "pressure_msl"])
    assert len(base) == 16


def test_the_fingerprint_is_stable_across_runs():
    """A literal, so a change to the hashing shows up as a failing test."""
    assert schema_fingerprint(["code", "timestamp", "pressure"]) == "f112d21de39a174b"


def test_the_shipped_fingerprint_matches_a_live_export_header(tmp_path):
    """Pinned from the real 22-column export captured on 2026-10-31."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert schema_fingerprint(HEADER.split(",")) == cfg.schema_fingerprint
    # And the pin is enforced: parse refuses a drifted header.
    drifted = SAMPLE_CSV.replace("sun_int_avg", "sun_int_average")
    with pytest.raises(SchemaDriftError, match="schema changed"):
        FakeRmi().parse(write(tmp_path, drifted), cfg)


# ------------------------------------------------------------------------- scan


def test_scan_reports_every_column_set_and_a_stable_fingerprint(tmp_path):
    files = [
        write(tmp_path, SAMPLE_CSV, "a.csv"),
        write(tmp_path, EMPTY_CSV, "b.csv"),
    ]
    report = FakeRmi().scan(files)
    assert report["source"] == "belgium"
    assert report["n_files"] == 2
    assert report["n_variants"] == 1               # same layer, same columns
    assert report["fingerprint"] == schema_fingerprint(HEADER.split(","))
    assert report["errors"] == []
    assert FakeRmi().scan(files)["fingerprint"] == report["fingerprint"]


def test_scan_records_a_bad_file_as_a_finding_rather_than_crashing(tmp_path):
    files = [
        write(tmp_path, SAMPLE_CSV, "a.csv"),
        write(tmp_path, WFS_ERROR, "bad.csv"),
    ]
    report = FakeRmi().scan(files)
    assert len(report["errors"]) == 1
    assert "bad.csv" in report["errors"][0]["path"]
    assert report["n_variants"] == 1


def test_scan_sees_two_variants_when_the_schema_really_moves(tmp_path):
    files = [
        write(tmp_path, SAMPLE_CSV, "a.csv"),
        write(tmp_path, SAMPLE_CSV.replace("sun_int_avg", "sun_int_average"), "b.csv"),
    ]
    report = FakeRmi().scan(files)
    assert report["n_variants"] == 2
    assert report["fingerprint"] is None           # nothing to pin while it drifts


# ---------------------------------------------------------------------- configs


def test_the_source_yaml_loads_into_sourceconfig():
    cfg = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    assert cfg.name == "belgium"
    assert cfg.base_url == "https://opendata.meteo.be/service/ows"
    assert cfg.credentials_env == []               # keyless by design
    assert cfg.extra["data_layer"] == DATA_LAYER


def test_the_source_yaml_pins_the_rate_limit_that_keeps_the_server_up():
    """DO NOT RELAX THIS TEST.

    This endpoint 500s under concurrency and rate-limits our whole egress IP
    after sustained parallel load, which takes the cluster's downloads out for
    tens of minutes. One request per second and one worker is the only setting
    observed to work, and it is still enough: ~350 requests is a six-minute
    backfill.
    """
    cfg = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    assert cfg.rate_limit.requests_per_second <= 1.0
    assert cfg.rate_limit.max_workers == 1


def test_the_product_yaml_loads_into_productconfig():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert cfg.store_id == "belgium/aws_10min/v1"
    assert cfg.store.kind == "points"
    assert cfg.grid_id == "knmi_1km_stereo"
    assert cfg.cadence_seconds == 600
    assert cfg.raw_layout.archive == "none"        # a station-year is coarse already


def test_every_configured_variable_has_a_valid_range():
    """Not optional here: RMI publishes raw data and this is its only QC."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert cfg.variables
    for var in cfg.variables:
        assert var.valid_min is not None, var.name
        assert var.valid_max is not None, var.name
        assert var.valid_min < var.valid_max, var.name


def test_every_configured_source_key_is_a_column_of_the_live_export():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    columns = set(HEADER.split(","))
    for column, var in variables_by_column(cfg):
        assert column in columns, f"{var.name} reads {column!r}, which the layer does not have"


def test_the_shipped_config_parses_a_real_export(tmp_path):
    """End to end on the shipped config, not the trimmed test one."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    frame = FakeRmi().parse(write(tmp_path, SAMPLE_CSV), cfg)
    assert frame.height == 4
    assert frame["air_temperature"].dtype == pl.Float32
    assert frame["soil_temperature_50cm"][0] == pytest.approx(16.015902, abs=1e-4)
    assert frame["wind_speed"].null_count() == 4              # empty in the export
    assert frame["wind_speed_30m"][0] == pytest.approx(3.038952, abs=1e-5)
    # Raw-data guards, in canonical units.
    assert frame["air_pressure"][3] is None                   # 504.1 hPa
    assert frame["surface_downwelling_shortwave_flux"][3] == pytest.approx(-3.2, abs=1e-3)


# --------------------------------------------------------------------- contract


def test_belgium_satisfies_the_source_protocol():
    assert isinstance(RmiAws(), Source)


def test_a_source_built_with_no_arguments_has_sane_defaults():
    """Constructed bare, with no request made: this suite never calls this server."""
    src = RmiAws()
    assert src.base_url == "https://opendata.meteo.be/service/ows"
    assert src.data_layer == DATA_LAYER
    assert src.station_layer == STATION_LAYER
    assert src.timeout >= 60
    assert f"typeNames={STATION_LAYER}" in src.station_url()
    url = src.data_url("6447", datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC))
    assert "code%3D6447" in url and "2024-12-31T23:59:59Z" in url
