"""Tests for the E-SOH / MeteoGate surface observation source.

Nothing here touches the network. The fixtures are verbatim excerpts captured
from observations.meteogate.eu on 2026-10-31 — a `/locations` GeoJSON
FeatureCollection, a single-cadence `Coverage` for De Bilt, and the
mixed-cadence `CoverageCollection` that Düsseldorf actually returns — kept as
JSON text so the wire format is genuinely exercised rather than assumed. They
are small enough to live beside the assertions that read them, which keeps the
fixture and the expected values from drifting apart.

The two facts most of this file exists to protect are the two that cannot be
recovered once broken: a unit key that does not advance with the harvest window
silently loses data from a feed with no archive, and a reader that understands
only one of CoverageJSON's two shapes silently loses every parameter at a
station that reports more than one cadence.
"""
from __future__ import annotations

import inspect
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import yaml

from meteopack.core import http
from meteopack.core.config import ProductConfig, SourceConfig, StoreConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.esoh import (
    DEFAULT_SLOT_SECONDS,
    EsohObservations,
    ParameterName,
    SchemaDriftError,
    parse_locations,
    product_fingerprint,
    schema_fingerprint,
    unit_key,
    variables_by_standard_name,
    wigos_id,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "esoh.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "esoh" / "observations.yaml"


# --------------------------------------------------------------------- fixtures

# `/collections/observations/locations?bbox=...&standard_name=...&f=GeoJSON`.
# The first three features are verbatim; note what each one is here to prove:
#   0-20000-0-06260 De Bilt      - WMO-number issuer, inside the KNMI footprint
#   0-20000-0-10400 Duesseldorf  - inside the footprint, precipitation only
#   0-276-0-10015                - the ISO-3166-numeric issuer form, which must
#                                  parse identically to the WMO-number one
# plus a Rome-ish feature inside the query bbox but outside the real footprint,
# and two malformed features that must be skipped rather than raise.
LOCATIONS = """
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [5.1797, 52.0989]},
      "properties": {
        "name": "De Bilt",
        "detail": "https://oscar.wmo.int/surface/rest/api/search/station?wigosId=0-20000-0-06260",
        "parameter-name": [
          "air_temperature:2.0:point:PT0S",
          "air_temperature:2.0:point:PT1M"
        ],
        "timeseries-link": "https://observations.meteogate.eu/collections/observations/items?platform=0-20000-0-06260"
      },
      "id": "0-20000-0-06260"
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [6.76863, 51.29597]},
      "properties": {
        "name": "DUESSELDORF",
        "parameter-name": [
          "precipitation_amount:1.0:sum:PT12H",
          "precipitation_amount:1.0:sum:PT24H",
          "precipitation_amount:1.0:sum:PT3H",
          "precipitation_amount:1.0:sum:PT6H"
        ]
      },
      "id": "0-20000-0-10400"
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [8.2370, 52.9336]},
      "properties": {
        "name": "Grossenkneten",
        "parameter-name": ["air_temperature:2.0:point:PT0S"]
      },
      "id": "0-276-0-10015"
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [12.4964, 41.9028]},
      "properties": {
        "name": "Roma Ciampino",
        "parameter-name": ["air_temperature:2.0:point:PT0S"]
      },
      "id": "0-380-0-16239"
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": [1.0, 52.0]},
      "properties": {"name": "no id at all", "parameter-name": []}
    },
    {
      "type": "Feature",
      "geometry": {"type": "Point", "coordinates": []},
      "properties": {"name": "no coordinates", "parameter-name": []},
      "id": "0-20000-0-99999"
    }
  ],
  "parameters": {}
}
"""

# `/collections/observations/locations/0-20000-0-06260?standard_name=air_temperature
#  &datetime=2026-10-31T18:00:00Z/2026-10-31T19:00:00Z&f=CoverageJSON`, verbatim.
# One cadence -> a bare `Coverage` with both parameters sharing one `t` axis.
COVERAGE_SINGLE = """
{
  "type": "Coverage",
  "domain": {
    "type": "Domain",
    "domainType": "PointSeries",
    "axes": {
      "x": {"values": [5.1797]},
      "y": {"values": [52.0989]},
      "t": {"values": [
        "2026-10-31T18:00:00Z", "2026-10-31T18:10:00Z", "2026-10-31T18:20:00Z",
        "2026-10-31T18:30:00Z", "2026-10-31T18:40:00Z", "2026-10-31T18:50:00Z",
        "2026-10-31T19:00:00Z"
      ]}
    },
    "referencing": [
      {"coordinates": ["x", "y"],
       "system": {"type": "GeographicCRS",
                  "id": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
      {"coordinates": ["t"], "system": {"type": "TemporalRS", "calendar": "Gregorian"}}
    ]
  },
  "parameters": {},
  "ranges": {
    "air_temperature:2.0:point:PT0S": {
      "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
      "shape": [7, 1, 1],
      "values": [5.5, 5.0, 4.6, 4.3, 4.0, 3.7, 3.3]
    },
    "air_temperature:2.0:point:PT1M": {
      "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
      "shape": [7, 1, 1],
      "values": [7.4, 6.9, 6.4, 6.2, 5.7, 5.5, 5.3]
    }
  },
  "metocean:wigosId": "0-20000-0-06260"
}
"""

# The same endpoint for 0-20000-0-10400 over 12:00-19:00Z, verbatim apart from
# two injected nulls. Mixed cadence -> a `CoverageCollection` with one coverage
# per parameter, each with its OWN `t` axis: hourly temperature, 3-hourly,
# 6-hourly and 12-hourly precipitation sums. This is the shape that a reader
# written against the single-Coverage response quietly throws away.
COVERAGE_MIXED = """
{
  "type": "CoverageCollection",
  "parameters": {},
  "coverages": [
    {
      "type": "Coverage",
      "domain": {"type": "Domain", "domainType": "PointSeries", "axes": {
        "x": {"values": [6.76863]}, "y": {"values": [51.29597]},
        "t": {"values": ["2026-10-31T12:00:00Z"]}}},
      "ranges": {"precipitation_amount:1.0:sum:PT6H": {
        "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
        "shape": [1, 1, 1], "values": [0.0]}},
      "metocean:wigosId": "0-20000-0-10400"
    },
    {
      "type": "Coverage",
      "domain": {"type": "Domain", "domainType": "PointSeries", "axes": {
        "x": {"values": [6.76863]}, "y": {"values": [51.29597]},
        "t": {"values": [
          "2026-10-31T12:00:00Z", "2026-10-31T13:00:00Z", "2026-10-31T14:00:00Z",
          "2026-10-31T15:00:00Z", "2026-10-31T16:00:00Z", "2026-10-31T17:00:00Z",
          "2026-10-31T18:00:00Z", "2026-10-31T19:00:00Z"]}}},
      "ranges": {"air_temperature:2.0:point:PT0S": {
        "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
        "shape": [8, 1, 1],
        "values": [10.4, 11.7, null, 11.6, 11.1, 10.1, 8.8, 8.2]}},
      "metocean:wigosId": "0-20000-0-10400"
    },
    {
      "type": "Coverage",
      "domain": {"type": "Domain", "domainType": "PointSeries", "axes": {
        "x": {"values": [6.76863]}, "y": {"values": [51.29597]},
        "t": {"values": [
          "2026-10-31T12:00:00Z", "2026-10-31T15:00:00Z", "2026-10-31T18:00:00Z"]}}},
      "ranges": {"precipitation_amount:1.0:sum:PT3H": {
        "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
        "shape": [3, 1, 1], "values": [0.0, null, 1.4]}},
      "metocean:wigosId": "0-20000-0-10400"
    },
    {
      "type": "Coverage",
      "domain": {"type": "Domain", "domainType": "PointSeries", "axes": {
        "x": {"values": [6.76863]}, "y": {"values": [51.29597]},
        "t": {"values": ["2026-10-31T18:00:00Z"]}}},
      "ranges": {"precipitation_amount:1.0:sum:PT12H": {
        "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
        "shape": [1, 1, 1], "values": [1.4]}},
      "metocean:wigosId": "0-20000-0-10400"
    }
  ]
}
"""

NOW = datetime(2026, 10, 31, 20, 0, tzinfo=UTC)


def product(**overrides) -> ProductConfig:
    """A small product config; the real YAML is exercised separately."""
    spec = {
        "source": "esoh",
        "product": "observations",
        "grid_id": "knmi_1km_stereo",
        "store": StoreConfig(kind="points"),
        "variables": [
            Variable(
                name="air_temperature",
                source_key="air_temperature:*:point:PT0S",
                units="degC",
                valid_min=-60.0,
                valid_max=60.0,
            ),
            Variable(
                name="precipitation_amount_3h",
                source_key="precipitation_amount:*:sum:PT3H",
                units="mm",
                valid_min=0.0,
                valid_max=500.0,
            ),
        ],
    }
    spec.update(overrides)
    return ProductConfig(**spec)


class FakeEsoh(EsohObservations):
    """The real source with its one network seam replaced."""

    def __init__(self, payload=None, **kwargs):
        kwargs.setdefault("now", NOW)
        super().__init__(SourceConfig(name="esoh"), **kwargs)
        self._payload = payload
        self.urls: list[str] = []

    def _get_json(self, url: str):
        self.urls.append(url)
        return self._payload


def write(tmp_path: Path, text: str, name: str = "x.covjson") -> RawFile:
    path = tmp_path / name
    path.write_text(text)
    return RawFile(path=path, sha256="0" * 64, n_bytes=len(text), unit_key="esoh/observations/x")


# ------------------------------------------------------------------ WIGOS ids


def test_wigos_id_accepts_both_issuer_forms():
    assert wigos_id("0-20000-0-06260") == "0-20000-0-06260"   # WMO number issuer
    assert wigos_id("0-276-0-10015") == "0-276-0-10015"       # ISO-3166 numeric
    assert wigos_id("  0-56-0-06494 ") == "0-56-0-06494"


def test_wigos_id_does_not_normalise_the_local_identifier():
    """The local id is a string by specification: 06260 is not 6260.

    DWD ids get zero-padded to five digits; doing the same here would merge two
    genuinely different stations, which is exactly the kind of assumption the
    second source exists to catch.
    """
    assert wigos_id("0-20000-0-06260") != wigos_id("0-20000-0-6260")


def test_wigos_id_allows_dashes_inside_the_local_identifier():
    """The local id is alphanumeric plus `-`/`_`, so the id is not four fields
    you can recover by splitting on the separator."""
    assert wigos_id("0-20000-0-ABC-12_3") == "0-20000-0-ABC-12_3"


@pytest.mark.parametrize(
    "bad",
    ["06260", "0-20000-0", "", "a-b-c-d", "0-20000-0-06260 x", "0-20000-0-" + "x" * 17],
)
def test_wigos_id_rejects_anything_else(bad):
    with pytest.raises(ValueError, match="WIGOS"):
        wigos_id(bad)


# ------------------------------------------------------------ parameter names


def test_parameter_name_round_trips_and_exposes_a_level_free_identity():
    p = ParameterName.parse("air_temperature:2.0:point:PT0S")
    assert (p.standard_name, p.level, p.method, p.period) == (
        "air_temperature", 2.0, "point", "PT0S"
    )
    assert p.measurement == "air_temperature:point:PT0S"
    assert ParameterName.parse(p.text) == p


def test_a_wildcard_level_matches_any_level_but_never_a_different_period():
    selector = ParameterName.parse("air_temperature:*:point:PT0S")
    assert selector.level is None
    # The live catalogue carries 1176 distinct levels for air_pressure alone,
    # because the level is often the station elevation, not the sensor height.
    assert selector.matches(ParameterName.parse("air_temperature:2.0:point:PT0S"))
    assert selector.matches(ParameterName.parse("air_temperature:1029.0:point:PT0S"))
    assert not selector.matches(ParameterName.parse("air_temperature:2.0:point:PT1M"))
    assert not selector.matches(ParameterName.parse("air_temperature:2.0:mean:PT0S"))
    assert not selector.matches(ParameterName.parse("dew_point_temperature:2.0:point:PT0S"))


def test_an_explicit_level_matches_only_that_level():
    selector = ParameterName.parse("air_temperature:2.0:point:PT0S")
    assert selector.matches(ParameterName.parse("air_temperature:2.0:point:PT0S"))
    assert not selector.matches(ParameterName.parse("air_temperature:10.0:point:PT0S"))


@pytest.mark.parametrize(
    "bad",
    [
        "air_temperature",
        "air_temperature:2.0:point",
        "air_temperature:2.0:point:PT0S:extra",
        "air_temperature::point:PT0S",
        "air_temperature:two:point:PT0S",
        ":2.0:point:PT0S",
    ],
)
def test_a_malformed_source_key_says_what_it_should_have_been(bad):
    with pytest.raises(ValueError, match="standard_name|empty|non-numeric"):
        ParameterName.parse(bad)


def test_variables_group_by_standard_name():
    """The standard name is the group: it is what `standard_name=` filters on."""
    groups = variables_by_standard_name(product())
    assert sorted(groups) == ["air_temperature", "precipitation_amount"]
    selector, var = groups["precipitation_amount"][0]
    assert selector.period == "PT3H"
    assert var.name == "precipitation_amount_3h"


# --------------------------------------------------------------- locations


def test_parse_locations_reads_ids_coordinates_and_advertised_parameters():
    records = parse_locations(json.loads(LOCATIONS))
    assert [r.station_id for r in records] == [
        "0-20000-0-06260", "0-20000-0-10400", "0-276-0-10015", "0-380-0-16239"
    ]
    de_bilt = records[0]
    assert (de_bilt.lon, de_bilt.lat) == (5.1797, 52.0989)
    assert de_bilt.name == "De Bilt"
    assert de_bilt.parameters == (
        "air_temperature:2.0:point:PT0S",
        "air_temperature:2.0:point:PT1M",
    )


def test_parse_locations_skips_malformed_features_rather_than_raising():
    """A continent's worth of stations must not die for one bad feature."""
    records = parse_locations(json.loads(LOCATIONS))
    assert len(records) == 4           # six features in, two unusable
    assert all(r.station_id for r in records)


def test_parse_locations_tolerates_an_empty_payload():
    assert parse_locations(None) == []
    assert parse_locations({"type": "FeatureCollection", "features": []}) == []


# ---------------------------------------------------------------- clipping


def test_stations_are_clipped_to_the_grid_footprint():
    """The bbox narrows the query; `grid.contains` trims it to the real shape.

    Rome falls inside the grid's bounding rectangle but outside the
    stereographic footprint, which is the whole reason the exact test exists
    rather than just the bbox.
    """
    src = FakeEsoh(json.loads(LOCATIONS))
    table = src.stations(product())
    assert table["station_id"].to_list() == [
        "0-20000-0-06260", "0-20000-0-10400", "0-276-0-10015"
    ]
    assert "0-380-0-16239" not in table["station_id"].to_list()


def test_a_station_advertising_nothing_this_product_wants_is_dropped():
    src = FakeEsoh(json.loads(LOCATIONS))
    cfg = product(
        variables=[
            Variable(name="wind_speed", source_key="wind_speed:*:mean:PT10M", units="m s-1")
        ]
    )
    assert src.stations(cfg).is_empty()


def test_the_query_bbox_comes_from_the_grid():
    src = FakeEsoh(json.loads(LOCATIONS))
    src.stations(product())
    assert "bbox=0.000000,48.895298,10.856429,55.973602" in src.urls[0]
    assert "standard_name=air_temperature,precipitation_amount" in src.urls[0]


def test_a_null_grid_id_is_the_europe_wide_escape_hatch():
    src = FakeEsoh(json.loads(LOCATIONS))
    table = src.stations(product(grid_id=None))
    assert "0-380-0-16239" in table["station_id"].to_list()   # Rome kept
    assert "bbox=" not in src.urls[0]


def test_a_null_grid_id_can_still_be_narrowed_by_an_explicit_bbox():
    src = FakeEsoh(json.loads(LOCATIONS))
    src.stations(product(grid_id=None, extra={"bbox": [3.0, 50.0, 8.0, 54.0]}))
    assert "bbox=3.000000,50.000000,8.000000,54.000000" in src.urls[0]


def test_an_empty_locations_query_is_a_404_and_not_a_failure():
    """A narrow footprint legitimately answers 404; that is zero stations."""
    src = FakeEsoh(None)
    assert src.stations(product()).is_empty()
    assert src.list(TimeWindow(NOW - timedelta(hours=1), NOW), product()) == []


# --------------------------------------------------------------- unit keys


def window(hours_back: float, hours: float = 1.0, *, offset_minutes: float = 0.0) -> TimeWindow:
    start = NOW - timedelta(hours=hours_back) + timedelta(minutes=offset_minutes)
    return TimeWindow(start, start + timedelta(hours=hours))


def test_unit_keys_are_stable_for_the_same_harvest_window():
    """Re-running a job must not re-download, so the key cannot drift."""
    src = FakeEsoh(json.loads(LOCATIONS))
    first = [u.key for u in src.list(window(2), product())]
    second = [u.key for u in FakeEsoh(json.loads(LOCATIONS)).list(window(2), product())]
    assert first == second
    assert first


def test_a_late_run_re_derives_the_same_key_for_the_slot_it_shares():
    """A cron that fires at :00 and a retry at :37 are the same harvest.

    This is why the window is floored onto a fixed slot grid rather than used
    verbatim: without it every retry would look like new work and re-download
    the hour. The later run reaches into the next slot as well, which is new
    work and correctly gets new keys.
    """
    src = FakeEsoh(json.loads(LOCATIONS))
    on_the_hour = {u.key for u in src.list(window(2), product())}
    jittered = {
        u.key
        for u in FakeEsoh(json.loads(LOCATIONS)).list(
            window(2, offset_minutes=37), product()
        )
    }
    assert on_the_hour and on_the_hour < jittered
    assert all("20261031T180000Z_PT1H" in k for k in on_the_hour)
    assert {k for k in jittered - on_the_hour} == {
        k.replace("T180000Z", "T190000Z") for k in on_the_hour
    }


def test_unit_keys_differ_across_harvest_windows():
    """The one property that keeps a rolling feed from being lost.

    A station-only key would be marked done on the first harvest and every
    later harvest of that station would be skipped by the ledger — and with no
    archive upstream, skipped means gone.
    """
    src = FakeEsoh(json.loads(LOCATIONS))
    earlier = {u.key for u in src.list(window(3), product())}
    later = {u.key for u in FakeEsoh(json.loads(LOCATIONS)).list(window(2), product())}
    assert earlier and later
    assert not (earlier & later)


def test_a_unit_key_encodes_the_window_the_station_and_the_parameter_set():
    src = FakeEsoh(json.loads(LOCATIONS))
    unit = src.list(window(2), product())[0]
    assert unit.key.startswith("esoh/observations/")
    assert unit.key.endswith("/20261031T180000Z_PT1H/0-20000-0-06260")
    assert unit.valid_time == datetime(2026, 10, 31, 18, 0, tzinfo=UTC)
    assert unit.meta["slot_start"] == "2026-10-31T18:00:00Z"
    assert unit.meta["standard_names"] == "air_temperature,precipitation_amount"


def test_widening_the_variable_list_changes_the_key():
    """Same URL, different payload: the ledger has to notice."""
    narrow = product()
    wide = product(
        variables=[
            *narrow.variables,
            Variable(
                name="wind_speed",
                source_key="wind_speed:*:mean:PT10M",
                units="m s-1",
            ),
        ]
    )
    a = FakeEsoh(json.loads(LOCATIONS)).list(window(2), narrow)[0]
    b = FakeEsoh(json.loads(LOCATIONS)).list(window(2), wide)[0]
    assert a.key != b.key


def test_unit_key_is_a_pure_function_of_its_inputs():
    cfg = product()
    slot = timedelta(seconds=DEFAULT_SLOT_SECONDS)
    start = datetime(2026, 10, 31, 18, tzinfo=UTC)
    key = unit_key(cfg, "0-276-0-10015", start, slot, "abcd1234")
    assert key == "esoh/observations/abcd1234/20261031T180000Z_PT1H/0-276-0-10015"
    assert unit_key(cfg, "0-276-0-10015", start + slot, slot, "abcd1234") != key


def test_a_slot_requests_a_closed_interval_one_second_short_of_the_next():
    """EDR datetime ranges are closed at both ends, so adjacent slots would
    otherwise both claim the boundary observation."""
    src = FakeEsoh(json.loads(LOCATIONS))
    unit = src.list(window(2), product())[0]
    assert "datetime=2026-10-31T18:00:00Z/2026-10-31T18:59:59Z" in unit.url
    assert unit.url.endswith("f=CoverageJSON") or "f=CoverageJSON" in unit.url


# -------------------------------------------------------- the rolling window


def test_a_backfill_is_clamped_to_what_the_rolling_collection_still_serves(caplog):
    """There is no archive: asking for last week cannot be answered, and must
    not turn into a week of units that are certain to 404."""
    src = FakeEsoh(json.loads(LOCATIONS))
    units = src.list(TimeWindow(NOW - timedelta(days=7), NOW), product())
    slots = {u.meta["slot_start"] for u in units}
    assert len(slots) == 24                       # not 168
    assert min(slots) == "2026-10-30T20:00:00Z"
    assert "clamping" in caplog.text


def test_a_window_entirely_outside_the_rolling_collection_yields_nothing():
    src = FakeEsoh(json.loads(LOCATIONS))
    old = TimeWindow(NOW - timedelta(days=9), NOW - timedelta(days=7))
    assert src.list(old, product()) == []


def test_the_rolling_width_is_configurable():
    src = FakeEsoh(json.loads(LOCATIONS))
    units = src.list(
        TimeWindow(NOW - timedelta(days=7), NOW),
        product(extra={"rolling_window_hours": 3}),
    )
    assert len({u.meta["slot_start"] for u in units}) == 3


def test_the_harvest_slot_is_configurable():
    src = FakeEsoh(json.loads(LOCATIONS))
    units = src.list(window(2), product(extra={"harvest_slot_seconds": 900}))
    assert {u.meta["slot"] for u in units} == {"PT15M"}
    assert len({u.meta["slot_start"] for u in units}) == 4


# ------------------------------------------------------------------- parsing


def test_a_single_coverage_becomes_a_tidy_frame(tmp_path):
    raw = write(tmp_path, COVERAGE_SINGLE)
    frame = FakeEsoh().parse(raw, product())
    assert frame.columns == ["time", "station_id", "air_temperature", "precipitation_amount_3h"]
    assert frame.height == 7
    assert frame["station_id"].unique().to_list() == ["0-20000-0-06260"]
    assert frame["time"].dtype == pl.Datetime("us", "UTC")
    assert frame["time"][0] == datetime(2026, 10, 31, 18, 0, tzinfo=UTC)
    assert frame["air_temperature"].to_list() == pytest.approx(
        [5.5, 5.0, 4.6, 4.3, 4.0, 3.7, 3.3]
    )
    # The product wants PT3H precipitation; this station reports none.
    assert frame["precipitation_amount_3h"].null_count() == 7


def test_the_period_in_the_source_key_picks_between_two_series_of_one_name(tmp_path):
    """De Bilt reports air_temperature twice, as PT0S and as PT1M. They are
    different numbers and the config says which one it means."""
    raw = write(tmp_path, COVERAGE_SINGLE)
    cfg = product(
        variables=[
            Variable(
                name="air_temperature",
                source_key="air_temperature:*:point:PT1M",
                units="degC",
            )
        ]
    )
    frame = FakeEsoh().parse(raw, cfg)
    assert frame["air_temperature"].to_list() == pytest.approx(
        [7.4, 6.9, 6.4, 6.2, 5.7, 5.5, 5.3]
    )


def test_a_coverage_collection_outer_joins_heterogeneous_cadences(tmp_path):
    """The shape a single-Coverage reader silently loses most of.

    Hourly temperature and 3-hourly precipitation share no cadence, so they
    join on time and simply have nulls where the other has nothing to say.
    Nothing is resampled, snapped or forward-filled here.
    """
    raw = write(tmp_path, COVERAGE_MIXED)
    frame = FakeEsoh().parse(raw, product())
    assert frame.height == 8                       # the union of both time axes
    assert frame["time"].to_list()[0] == datetime(2026, 10, 31, 12, tzinfo=UTC)
    at_12 = frame.filter(pl.col("time") == datetime(2026, 10, 31, 12, tzinfo=UTC))
    assert at_12["air_temperature"].item() == pytest.approx(10.4)
    assert at_12["precipitation_amount_3h"].item() == pytest.approx(0.0)
    at_13 = frame.filter(pl.col("time") == datetime(2026, 10, 31, 13, tzinfo=UTC))
    assert at_13["air_temperature"].item() == pytest.approx(11.7)
    assert at_13["precipitation_amount_3h"].item() is None
    at_18 = frame.filter(pl.col("time") == datetime(2026, 10, 31, 18, tzinfo=UTC))
    assert at_18["precipitation_amount_3h"].item() == pytest.approx(1.4)


def test_json_null_becomes_a_null_not_a_zero(tmp_path):
    raw = write(tmp_path, COVERAGE_MIXED)
    frame = FakeEsoh().parse(raw, product())
    at_14 = frame.filter(pl.col("time") == datetime(2026, 10, 31, 14, tzinfo=UTC))
    assert at_14["air_temperature"].item() is None
    at_15 = frame.filter(pl.col("time") == datetime(2026, 10, 31, 15, tzinfo=UTC))
    assert at_15["precipitation_amount_3h"].item() is None


def test_a_sentinel_is_nulled_before_scale_is_applied(tmp_path):
    """Order matters: -99.9 scaled by 0.01 is a believable -0.999 m of snow.

    E-SOH itself uses JSON null, but national feeds have been known to push a
    numeric sentinel through unchanged, so the hook exists and is ordered the
    same way as DWD's.
    """
    document = json.loads(COVERAGE_SINGLE)
    document["ranges"] = {
        "surface_snow_thickness:0.0:point:PT0S": {
            "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
            "shape": [7, 1, 1],
            "values": [12.0, -99.9, 8.0, 4.0, 0.0, 2.0, 6.0],
        }
    }
    raw = write(tmp_path, json.dumps(document))
    cfg = product(
        variables=[
            Variable(
                name="surface_snow_thickness",
                source_key="surface_snow_thickness:*:point:PT0S",
                units="m",
                scale=0.01,
                valid_min=0.0,
                valid_max=10.0,
            )
        ],
        extra={"missing_sentinels": [-99.9]},
    )
    values = FakeEsoh().parse(raw, cfg)["surface_snow_thickness"].to_list()
    assert values[0] == pytest.approx(0.12)        # 12 cm -> 0.12 m
    assert values[1] is None                       # sentinel, not -0.999 m


def test_the_valid_range_is_applied_after_scale_and_in_canonical_units(tmp_path):
    document = json.loads(COVERAGE_SINGLE)
    document["ranges"] = {
        "surface_snow_thickness:0.0:point:PT0S": {
            "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
            "shape": [7, 1, 1],
            "values": [50.0, 2000.0, 8.0, 4.0, 0.0, 2.0, 6.0],
        }
    }
    raw = write(tmp_path, json.dumps(document))
    cfg = product(
        variables=[
            Variable(
                name="surface_snow_thickness",
                source_key="surface_snow_thickness:*:point:PT0S",
                units="m",
                scale=0.01,
                valid_min=0.0,
                valid_max=10.0,
            )
        ]
    )
    values = FakeEsoh().parse(raw, cfg)["surface_snow_thickness"].to_list()
    assert values[0] == pytest.approx(0.5)         # 50 cm is fine
    assert values[1] is None                       # 20 m of snow is not


def test_a_wildcard_level_resolves_to_one_series_deterministically(tmp_path):
    """Two barometers at one station must not make the column's provenance
    depend on dict ordering."""
    document = json.loads(COVERAGE_SINGLE)
    document["ranges"] = {
        "air_temperature:10.0:point:PT0S": {
            "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
            "shape": [7, 1, 1], "values": [1, 1, 1, 1, 1, 1, 1],
        },
        "air_temperature:2.0:point:PT0S": {
            "type": "NdArray", "dataType": "float", "axisNames": ["t", "x", "y"],
            "shape": [7, 1, 1], "values": [2, 2, 2, 2, 2, 2, 2],
        },
    }
    raw = write(tmp_path, json.dumps(document))
    frame = FakeEsoh().parse(raw, product())
    assert frame["air_temperature"].to_list() == pytest.approx([2.0] * 7)

    reversed_doc = json.loads(COVERAGE_SINGLE)
    reversed_doc["ranges"] = dict(reversed(list(document["ranges"].items())))
    other = FakeEsoh().parse(write(tmp_path, json.dumps(reversed_doc), "y.covjson"), product())
    assert other["air_temperature"].to_list() == frame["air_temperature"].to_list()


def test_a_config_wanting_nothing_yields_a_correctly_typed_empty_frame(tmp_path):
    raw = write(tmp_path, COVERAGE_SINGLE)
    frame = FakeEsoh().parse(raw, product(variables=[]))
    assert frame.is_empty()
    assert frame.columns == ["time", "station_id"]


@pytest.mark.parametrize(
    "mangle, match",
    [
        (lambda d: d.update(type="FeatureCollection"), "CoverageCollection"),
        (lambda d: d["domain"]["axes"].pop("t"), "domain.axes.t"),
        (lambda d: d["domain"]["axes"]["x"].update(values=[1.0, 2.0]), "single x coordinate"),
        (
            lambda d: d["ranges"]["air_temperature:2.0:point:PT0S"].update(values=[1.0]),
            "values for 7 times",
        ),
    ],
)
def test_an_unexpected_document_shape_is_a_loud_failure(tmp_path, mangle, match):
    document = json.loads(COVERAGE_SINGLE)
    mangle(document)
    raw = write(tmp_path, json.dumps(document))
    with pytest.raises(SchemaDriftError, match=match):
        FakeEsoh().parse(raw, product())


def test_a_truncated_document_is_a_loud_failure(tmp_path):
    raw = write(tmp_path, COVERAGE_SINGLE[: len(COVERAGE_SINGLE) // 2])
    with pytest.raises(SchemaDriftError, match="not valid JSON"):
        FakeEsoh().parse(raw, product())


# -------------------------------------------------------------- fingerprints


def test_the_fingerprint_ignores_order_and_level():
    """Both normalisations are load bearing.

    Order, because the API is free to reorder its ranges; level, because the
    level is routinely the station elevation and would otherwise make every
    station a schema variant of its own.
    """
    a = schema_fingerprint(["air_temperature:2.0:point:PT0S", "wind_speed:10.0:mean:PT10M"])
    b = schema_fingerprint(["wind_speed:10.0:mean:PT10M", "air_temperature:2.0:point:PT0S"])
    c = schema_fingerprint(["air_temperature:1029.0:point:PT0S", "wind_speed:3.5:mean:PT10M"])
    assert a == b == c


def test_the_fingerprint_moves_when_the_measurement_changes():
    base = schema_fingerprint(["air_temperature:2.0:point:PT0S"])
    assert base != schema_fingerprint(["air_temperature:2.0:point:PT1M"])   # period
    assert base != schema_fingerprint(["air_temperature:2.0:mean:PT0S"])    # method
    assert base != schema_fingerprint(["dew_point_temperature:2.0:point:PT0S"])
    assert base != schema_fingerprint(
        ["air_temperature:2.0:point:PT0S", "wind_speed:10.0:mean:PT10M"]
    )


def test_the_fingerprint_is_pinned_rather_than_recomputed():
    """A literal, so a change to the hashing shows up as a failing test."""
    assert schema_fingerprint(["air_temperature:2.0:point:PT0S"]) == schema_fingerprint(
        ["air_temperature:*:point:PT0S"]
    )
    assert len(schema_fingerprint(["air_temperature:2.0:point:PT0S"])) == 16
    assert len(product_fingerprint({"air_temperature": "deadbeefdeadbeef"})) == 16


def test_a_station_reporting_a_subset_of_the_catalogue_passes(tmp_path):
    """No two stations carry the same parameter set, so a subset is normal."""
    raw = write(tmp_path, COVERAGE_SINGLE)
    known = {
        "air_temperature": [
            "air_temperature:point:PT0S",
            "air_temperature:point:PT1M",
            "air_temperature:mean:PT1H",       # this station does not report it
        ]
    }
    cfg = product(extra={"known_measurements": known})
    assert FakeEsoh().parse(raw, cfg).height == 7


def test_a_measurement_outside_the_catalogue_refuses_to_parse(tmp_path):
    raw = write(tmp_path, COVERAGE_SINGLE)
    cfg = product(
        extra={"known_measurements": {"air_temperature": ["air_temperature:point:PT0S"]}}
    )
    with pytest.raises(SchemaDriftError, match="air_temperature:point:PT1M"):
        FakeEsoh().parse(raw, cfg)


def test_editing_the_catalogue_without_repinning_its_hash_fails_loudly(tmp_path):
    raw = write(tmp_path, COVERAGE_SINGLE)
    cfg = product(
        schema_fingerprint="0000000000000000",
        extra={"known_measurements": {"air_temperature": ["air_temperature:point:PT0S"]}},
    )
    with pytest.raises(SchemaDriftError, match="config is inconsistent"):
        FakeEsoh().parse(raw, cfg)


# --------------------------------------------------------------------- scan


def test_scan_reports_a_pinnable_catalogue_and_a_stable_fingerprint(tmp_path):
    files = [
        write(tmp_path, COVERAGE_SINGLE, "a.covjson"),
        write(tmp_path, COVERAGE_MIXED, "b.covjson"),
    ]
    report = FakeEsoh().scan(files)
    assert report["source"] == "esoh"
    assert report["n_files"] == 2
    assert report["known_measurements"]["air_temperature"] == [
        "air_temperature:point:PT0S",
        "air_temperature:point:PT1M",
    ]
    assert report["known_measurements"]["precipitation_amount"] == [
        "precipitation_amount:sum:PT12H",
        "precipitation_amount:sum:PT3H",
        "precipitation_amount:sum:PT6H",
    ]
    assert report["errors"] == []
    # The union is what a config pins, and it is reproducible.
    assert report["fingerprint"] == product_fingerprint(report["catalogue_fingerprints"])
    assert FakeEsoh().scan(files)["fingerprint"] == report["fingerprint"]


def test_scan_records_a_bad_file_as_a_finding_rather_than_crashing(tmp_path):
    files = [
        write(tmp_path, COVERAGE_SINGLE, "a.covjson"),
        write(tmp_path, "{not json", "bad.covjson"),
    ]
    report = FakeEsoh().scan(files)
    assert len(report["errors"]) == 1
    assert "bad.covjson" in report["errors"][0]["path"]
    assert report["known_measurements"]["air_temperature"]


# -------------------------------------------------------------------- fetch


def test_fetch_binds_the_real_fetch_file_signature(monkeypatch, tmp_path):
    """Bound against the live callable, never a hand-copied parameter list.

    A source that omits `unit_key` passes every mocked test and raises
    TypeError on the first real download; that has happened here before.
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
    src = FakeEsoh(json.loads(LOCATIONS))
    unit = src.list(window(2), product())[0]
    raw = src.fetch(unit, tmp_path)

    assert seen["unit_key"] == unit.key
    assert seen["url"] == unit.url
    assert raw is not None
    # The raw tree mirrors the key: parameter set, then slot, then station.
    assert raw.path.name == "0-20000-0-06260.covjson"
    assert raw.path.parent.name == "2026-10-31T180000Z"


def test_fetch_skips_a_404_rather_than_failing_the_job(monkeypatch, tmp_path):
    """A station that advertised a parameter but reported nothing this hour."""

    def not_found(url, dest, **kwargs):
        raise http.NotFoundError(url, "404 not found", 404)

    monkeypatch.setattr(http, "fetch_file", not_found)
    unit = Unit(
        key="esoh/observations/abcd1234/20261031T180000Z_PT1H/0-276-0-10015",
        url="https://example.invalid/x",
        meta={"digest": "abcd1234", "slot_start": "2026-10-31T18:00:00Z",
              "filename": "0-276-0-10015.covjson"},
    )
    assert FakeEsoh().fetch(unit, tmp_path) is None


def test_fetch_survives_a_unit_with_no_meta(monkeypatch, tmp_path):
    captured: dict = {}

    def capture(url, dest, **kwargs):
        captured["dest"] = Path(dest)
        return RawFile(Path(dest), "0" * 64, 0, kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", capture)
    unit = Unit(key="esoh/observations/a/b/0-276-0-10015", url="https://example.invalid/x")
    FakeEsoh().fetch(unit, tmp_path)
    assert captured["dest"].name == "0-276-0-10015.covjson"


# ------------------------------------------------------------------- configs


def test_the_source_yaml_loads_into_sourceconfig():
    cfg = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    assert cfg.name == "esoh"
    assert cfg.base_url == "https://observations.meteogate.eu"
    assert cfg.credentials_env == []               # keyless by design
    assert cfg.rate_limit.requests_per_second > 0


def test_the_product_yaml_loads_into_productconfig():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert cfg.store_id == "esoh/observations/v1"
    assert cfg.store.kind == "points"
    assert cfg.grid_id == "knmi_1km_stereo"
    assert cfg.retention.raw.keep is True          # upstream has no archive
    assert cfg.extra["harvest_slot_seconds"] == 3600


def test_every_configured_source_key_parses_and_is_in_the_pinned_catalogue():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    known = cfg.extra["known_measurements"]
    assert cfg.variables
    for var in cfg.variables:
        selector = ParameterName.parse(var.source_key)
        assert selector.level is None, f"{var.name} pins a level; the level is usually elevation"
        assert selector.measurement in known[selector.standard_name], var.source_key


def test_the_pinned_catalogue_hashes_to_the_pinned_fingerprint():
    """Exactly the check `parse` makes, so a half-done re-pin fails here first."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    catalogue = {
        group: schema_fingerprint(names)
        for group, names in cfg.extra["known_measurements"].items()
    }
    assert product_fingerprint(catalogue) == cfg.schema_fingerprint


def test_the_real_config_parses_a_real_document(tmp_path):
    """End to end on the shipped config, not the trimmed test one."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    frame = FakeEsoh().parse(write(tmp_path, COVERAGE_MIXED), cfg)
    assert frame.height == 8
    assert "precipitation_amount_1h" in frame.columns
    assert frame["air_temperature"].dtype == pl.Float32
    assert frame["precipitation_amount_1h"].null_count() == 8   # this station is 3/6/12-hourly


# ------------------------------------------------------------------ contract


def test_esoh_satisfies_the_source_protocol():
    assert isinstance(EsohObservations(), Source)


def test_a_half_expired_oldest_slot_is_not_emitted():
    """Never ask for a slot the rolling window has already partly dropped.

    Flooring the clamped start onto the slot grid reaches back behind the
    retention horizon. Stations that report on the hour have already lost their
    observation there, so every unit in that slot 404s and the ledger files
    thousands of them as permanently "absent upstream" — for a feed with no
    archive, permanently is literal. Observed live before this was fixed:
    3 of 3 sampled stations returned 404.
    """
    # The clock must NOT sit on a slot boundary, or flooring is a no-op and the
    # bug cannot appear: the real horizon drifts continuously (observed live at
    # 20:18:34Z) and is almost never aligned.
    ragged_now = datetime(2026, 10, 31, 20, 18, 34, tzinfo=UTC)
    src = FakeEsoh(json.loads(LOCATIONS), now=ragged_now)
    cfg = product(extra={"rolling_window_hours": 24})
    far_back = TimeWindow(ragged_now - timedelta(hours=48), ragged_now)
    units = src.list(far_back, cfg)
    assert units, "a 48 h request should still yield the slots that are served"

    horizon = ragged_now - timedelta(hours=24)
    oldest = min(u.meta["slot_start"] for u in units)
    assert datetime.fromisoformat(oldest) >= horizon, (
        f"emitted slot {oldest} starts before the retention horizon {horizon.isoformat()}"
    )


def test_the_still_filling_slot_is_marked_volatile():
    """The newest slot has not closed yet, so its key must not count as done.

    Unit keys are deliberately stable per slot. Without a volatile flag the
    ledger would mark a half-harvested hour complete and the rest of it would
    never be collected.
    """
    src = FakeEsoh(json.loads(LOCATIONS))
    units = src.list(window(0.5, hours=1.0), product())
    assert units
    assert any(u.meta.get("volatile") for u in units), "current slot should be volatile"

    past = FakeEsoh(json.loads(LOCATIONS)).list(window(6, hours=1.0), product())
    assert past and not any(u.meta.get("volatile") for u in past), (
        "a closed slot is immutable and must stay skippable"
    )
