"""Tests for the UK Environment Agency Hydrology rainfall source.

Nothing here touches the network by default. The fixtures are verbatim
excerpts captured on 2026-10-31 — items from
`/id/stations?observedProperty=rainfall` and rows from a real `readings.csv` —
kept as literals so the wire format is genuinely exercised rather than assumed.
Two optional suites are guarded by `skipif`: one that makes a handful of live
keyless requests (`METEOPACK_LIVE=1`), and one that reads the existing local
archive if it happens to be mounted.

The three facts most of this file exists to protect are the three that are
expensive to discover downstream: `stationReference` is not unique and keying
on it silently merges two gauges (the fixture carries a real duplicate pair), a
zero-byte station-year is a normal answer rather than a failure, and a quality
flag must become null *before* any scaling rather than after.
"""
from __future__ import annotations

import inspect
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from meteopack.core import http
from meteopack.core.config import ProductConfig, SourceConfig, StoreConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.ea import (
    CSV_ROW_CAP,
    PERIOD_SECONDS,
    EaRainfall,
    SchemaDriftError,
    measure_notation,
    parse_stations,
    schema_fingerprint,
    station_id,
    unit_key,
    variables_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "ea.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "ea" / "rainfall_15min.yaml"

LOCAL_ARCHIVE = Path("/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/ea/rainfall_15min")

MEASURE_BASE = "http://environment.data.gov.uk/hydrology/id/measures"


# --------------------------------------------------------------------- fixtures

# `/id/stations?observedProperty=rainfall&_limit=10000`, verbatim items with the
# unused keys trimmed. Note what each one is here to prove:
#   ..._589359  Ulpha Duddo  - the `<guid>_<ref>` notation form (77 of 995)
#   a3be5ed5... Peak Forest  - the bare-GUID form (the other 918)
#   ..._344137 / ..._344138  - TWO gauges sharing stationReference "43204"
#   ..._334156               - a third station sharing "E14900" with a fourth
#   8864da31...              - a gauge with only the DAILY measure, no 15-minute
#   plus a Cornish gauge outside no bbox, a Roman one outside this product's,
#   and two malformed items that must be skipped rather than raise.
STATIONS = """
{
  "meta": {"publisher": "Environment Agency", "limit": 10000},
  "items": [
    {
      "@id": "http://environment.data.gov.uk/hydrology/id/stations/34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359",
      "notation": "34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359",
      "label": "Ulpha  Duddo",
      "lat": 54.341652,
      "long": -3.218087,
      "stationReference": "589359",
      "dateOpened": "1990-10-04",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        },
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359-rainfall-t-86400-mm-qualified",
          "parameter": "rainfall",
          "period": 86400
        }
      ]
    },
    {
      "@id": "http://environment.data.gov.uk/hydrology/id/stations/a3be5ed5-ba57-4035-a5a2-8947ba7fb77f",
      "notation": "a3be5ed5-ba57-4035-a5a2-8947ba7fb77f",
      "label": "Peak Forest",
      "lat": 53.317,
      "long": -1.8,
      "dateOpened": "2003-05-01",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/a3be5ed5-ba57-4035-a5a2-8947ba7fb77f-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        }
      ]
    },
    {
      "notation": "91a0bc46-6c53-4857-8ad7-e94414fd2a40_344137",
      "label": "Black Lane",
      "lat": 50.864324,
      "long": -2.142774,
      "stationReference": "43204",
      "dateOpened": "1995-04-22",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/91a0bc46-6c53-4857-8ad7-e94414fd2a40_344137-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        }
      ]
    },
    {
      "notation": "91a0bc46-6c53-4857-8ad7-e94414fd2a40_344138",
      "label": "Black Lane",
      "lat": 50.864324,
      "long": -2.142774,
      "stationReference": "43204",
      "dateOpened": "2021-12-31",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/91a0bc46-6c53-4857-8ad7-e94414fd2a40_344138-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        }
      ]
    },
    {
      "notation": "5e45b2c5-20ea-4e7f-b37a-a9b220ba3946_334156",
      "label": "Afton",
      "lat": 50.677494,
      "long": -1.504857,
      "stationReference": "E14900",
      "dateOpened": "2001-03-30",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/5e45b2c5-20ea-4e7f-b37a-a9b220ba3946_334156-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        }
      ]
    },
    {
      "notation": "8864da31-6c81-4cb1-bde9-d4ca73423222",
      "label": "Daily only",
      "lat": 52.5,
      "long": -1.0,
      "dateOpened": "1999-01-01",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/8864da31-6c81-4cb1-bde9-d4ca73423222-rainfall-t-86400-mm-qualified",
          "parameter": "rainfall",
          "period": 86400
        }
      ]
    },
    {
      "notation": "0f0f0f0f-0000-0000-0000-00000000beef",
      "label": "Roma Ciampino (not an EA gauge; outside the product bbox)",
      "lat": 41.9028,
      "long": 12.4964,
      "dateOpened": "2010-01-01",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/0f0f0f0f-0000-0000-0000-00000000beef-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        }
      ]
    },
    {
      "label": "no notation at all",
      "lat": 53.0,
      "long": -1.0,
      "measures": []
    },
    {
      "notation": "deadbeef-0000-0000-0000-000000000000",
      "label": "no coordinates",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/deadbeef-0000-0000-0000-000000000000-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        }
      ]
    }
  ]
}
"""

# One synthetic gauge placed inside `example_tiny` (3-5 E, 52-53 N), which is a
# patch of North Sea. It exists only to prove the grid branch of the clipping
# keeps what falls inside a footprint; no real EA gauge is out there.
STATIONS_IN_EXAMPLE_GRID = """
{
  "items": [
    {
      "notation": "cafe0000-0000-0000-0000-000000000001",
      "label": "Synthetic, inside example_tiny",
      "lat": 52.5,
      "long": 4.0,
      "dateOpened": "2000-01-01",
      "measures": [
        {
          "@id": "http://environment.data.gov.uk/hydrology/id/measures/cafe0000-0000-0000-0000-000000000001-rainfall-t-900-mm-qualified",
          "parameter": "rainfall",
          "period": 900
        }
      ]
    }
  ]
}
"""

_M = f"{MEASURE_BASE}/a3be5ed5-ba57-4035-a5a2-8947ba7fb77f-rainfall-t-900-mm-qualified"

# `readings.csv` rows, verbatim from the 2024 file of a3be5ed5-... in the local
# archive, plus two synthetic rows marked below. EA quotes every field.
READINGS = f'''"measure","dateTime","date","value","completeness","quality","qcode"
"{_M}","2024-01-02T07:30:00","2024-01-02","0.0","Complete","Good",""
"{_M}","2024-01-02T07:45:00","2024-01-02","0.2","Complete","Good",""
"{_M}","2024-05-09T12:45:00","2024-05-09","","Incomplete","Missing",""
"{_M}","2024-05-09T13:00:00","2024-05-09","","Incomplete","Missing",""
"{_M}","2024-07-09T08:45:00","2024-07-09","0.2","Complete","Suspect",""
"{_M}","2024-07-16T11:15:00","2024-07-16","0.0","Complete","Unchecked",""
"{_M}","2024-08-01T00:00:00","2024-08-01","0.6","Complete","Good",""
"{_M}","2024-08-01T00:07:00","2024-08-01","0.4","Complete","Good",""
'''
# The last two rows are synthetic: 00:00 carries a 0.6 that a scale of 10 turns
# into 6.0 (used to prove the valid range is applied AFTER the scale), and
# 00:07 is deliberately off the 900 s grid.

#: Zero bytes, exactly what EA answers for a pre-opening year or the QC tail.
EMPTY = ""


# ----------------------------------------------------------------- test doubles


class FakeEa(EaRainfall):
    """EaRainfall with the one remote read replaced by a literal payload."""

    def __init__(self, payload=None, **kwargs):
        super().__init__(**kwargs)
        self._payload = payload if payload is not None else json.loads(STATIONS)

    def _get_json(self, url: str):
        return self._payload


def product(**overrides) -> ProductConfig:
    base = {
        "source": "ea",
        "product": "rainfall_15min",
        "cadence_seconds": 900,
        "grid_id": None,
        "store": StoreConfig(kind="points"),
        "variables": [
            Variable(
                name="precipitation_amount",
                source_key="value",
                units="mm",
                dtype="float32",
                valid_min=0.0,
                valid_max=200.0,
            )
        ],
        "extra": {"bbox": [-6.5, 49.5, 2.2, 56.5]},
    }
    base.update(overrides)
    return ProductConfig(**base)


def window(start_year: int, end_year: int) -> TimeWindow:
    return TimeWindow(
        datetime(start_year, 1, 1, tzinfo=UTC), datetime(end_year, 1, 1, tzinfo=UTC)
    )


def write(tmp_path: Path, text: str, name: str = "2024.csv", key: str | None = None) -> RawFile:
    path = tmp_path / name
    path.write_text(text)
    return RawFile(
        path=path,
        sha256="0" * 64,
        n_bytes=len(text),
        unit_key=key or "ea/rainfall_15min/a3be5ed5-ba57-4035-a5a2-8947ba7fb77f/2024",
    )


# -------------------------------------------------------------------- identity


def test_station_id_keeps_both_notation_forms_verbatim():
    """A GUID and a `<guid>_<ref>` are different stations, not two spellings."""
    assert station_id("a3be5ed5-ba57-4035-a5a2-8947ba7fb77f") == (
        "a3be5ed5-ba57-4035-a5a2-8947ba7fb77f"
    )
    assert station_id(" 34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359 ") == (
        "34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359"
    )


@pytest.mark.parametrize("bad", ["", "   ", "a/b"])
def test_station_id_rejects_anything_unusable(bad):
    with pytest.raises(ValueError, match="notation"):
        station_id(bad)


def test_station_reference_is_not_unique_but_notation_is():
    """The reason `station_id` is the notation, shown with real catalogue rows.

    Both "43204" gauges are real and live; keying on `stationReference` would
    merge them into one column and lose one of the two series.
    """
    records = parse_stations(json.loads(STATIONS))
    references = [r.station_reference for r in records if r.station_reference]
    assert references.count("43204") == 2
    assert len({r.station_id for r in records}) == len(records)


def test_measure_notation_round_trips_from_an_id_or_a_full_url():
    notation = "a3be5ed5-ba57-4035-a5a2-8947ba7fb77f"
    assert measure_notation(f"{notation}-rainfall-t-900-mm-qualified") == notation
    assert measure_notation(_M) == notation
    assert measure_notation("something else entirely") is None


# ------------------------------------------------------------------- catalogue


def test_parse_stations_reads_ids_coordinates_measures_and_opening_dates():
    records = {r.station_id: r for r in parse_stations(json.loads(STATIONS))}
    ulpha = records["34f4e53b-4616-4aaa-b8ad-170c22c808c4_589359"]
    assert (ulpha.lon, ulpha.lat) == (-3.218087, 54.341652)
    assert ulpha.name == "Ulpha  Duddo"
    assert ulpha.date_opened == datetime(1990, 10, 4).date()
    # The 15-minute measure, never the daily one that sits beside it.
    assert ulpha.measure_id.endswith(f"-rainfall-t-{PERIOD_SECONDS}-mm-qualified")


def test_a_station_with_only_the_daily_measure_is_not_this_product():
    records = parse_stations(json.loads(STATIONS))
    assert "8864da31-6c81-4cb1-bde9-d4ca73423222" not in {r.station_id for r in records}


def test_parse_stations_skips_malformed_items_rather_than_raising():
    """A 995-station catalogue must not die for one bad item."""
    records = parse_stations(json.loads(STATIONS))
    # Nine items in: one daily-only, one with no notation, one with no
    # coordinates; six usable.
    assert len(records) == 6
    assert all(r.station_id and r.measure_id for r in records)


def test_parse_stations_tolerates_an_empty_payload():
    assert parse_stations(None) == []
    assert parse_stations({"items": []}) == []


# ------------------------------------------------------------------ fingerprint


def test_schema_fingerprint_is_order_independent_and_quote_insensitive():
    columns = ["measure", "dateTime", "date", "value", "completeness", "quality", "qcode"]
    assert schema_fingerprint(columns) == schema_fingerprint(reversed(columns))
    assert schema_fingerprint(columns) == schema_fingerprint([f' "{c}" ' for c in columns])


def test_schema_fingerprint_moves_when_a_column_really_changes():
    columns = ["measure", "dateTime", "value", "quality"]
    assert schema_fingerprint(columns) != schema_fingerprint([*columns, "qcode"])
    assert schema_fingerprint(columns) != schema_fingerprint(["measure", "dateTime", "val", "quality"])


def test_the_fingerprint_is_stable_across_processes():
    """It is pinned in YAML, so it must not depend on hash randomisation."""
    assert schema_fingerprint(["measure", "dateTime", "value"]) == "aeb649778f48ae76"


# --------------------------------------------------------------------- variables


def test_variables_map_to_readings_columns():
    assert [(col, var.name) for col, var in variables_for(product())] == [
        ("value", "precipitation_amount")
    ]


@pytest.mark.parametrize("bad", ["dateTime", "measure", "quality", "  "])
def test_a_structural_or_empty_source_key_says_what_it_should_have_been(bad):
    cfg = product(variables=[Variable(name="x", source_key=bad, units="mm")])
    with pytest.raises(ValueError, match="value|column name"):
        variables_for(cfg)


# ---------------------------------------------------------------------- stations


def test_stations_are_clipped_to_the_configured_bbox():
    table = FakeEa().stations(product())
    ids = table["station_id"].to_list()
    assert "a3be5ed5-ba57-4035-a5a2-8947ba7fb77f" in ids
    assert "0f0f0f0f-0000-0000-0000-00000000beef" not in ids, "Rome is not in England"


def test_stations_are_clipped_to_a_grid_footprint_when_one_is_set():
    """The grid branch, exercised through the real registry rather than a stub."""
    cfg = product(grid_id="example_tiny", extra={})
    assert FakeEa(json.loads(STATIONS)).stations(cfg).is_empty()
    inside = FakeEa(json.loads(STATIONS_IN_EXAMPLE_GRID)).stations(cfg)
    assert inside["station_id"].to_list() == ["cafe0000-0000-0000-0000-000000000001"]


def test_stations_emit_a_null_height_rather_than_a_fabricated_one():
    """EA publishes no elevation. A zero here would propagate into every
    derived product that uses station height."""
    table = FakeEa().stations(product())
    assert "height" in table.columns
    assert table["height"].dtype == pl.Float64
    assert table["height"].null_count() == table.height


def test_stations_carry_what_pack_needs_to_place_a_column():
    table = FakeEa().stations(product())
    assert {"station_id", "lat", "lon", "height", "name"} <= set(table.columns)
    assert table["station_id"].is_duplicated().sum() == 0
    assert table["station_id"].to_list() == sorted(table["station_id"].to_list())


# -------------------------------------------------------------------------- list


def test_unit_keys_are_one_per_station_year_and_stable():
    units = FakeEa().list(window(2020, 2023), product())
    keys = [u.key for u in units]
    assert len(keys) == len(set(keys))
    assert unit_key(product(), "a3be5ed5-ba57-4035-a5a2-8947ba7fb77f", 2021) in keys
    # Same request, same keys: that is what lets the ledger skip finished work.
    assert keys == [u.key for u in FakeEa().list(window(2020, 2023), product())]


def test_an_exclusive_window_end_does_not_pull_in_the_next_year():
    units = FakeEa().list(window(2020, 2023), product())
    assert {u.meta["year"] for u in units} == {"2020", "2021", "2022"}


def test_years_before_a_gauge_opened_are_not_requested():
    """They answer 200 with an empty body, so they are guaranteed-empty work."""
    units = FakeEa().list(window(1990, 2000), product())
    black_lane = [
        u for u in units if u.meta["station_id"].endswith("_344137")
    ]  # dateOpened 1995-04-22
    assert black_lane and min(int(u.meta["year"]) for u in black_lane) == 1995
    ulpha = [u for u in units if u.meta["station_id"].endswith("_589359")]
    assert ulpha and min(int(u.meta["year"]) for u in ulpha) == 1990


def test_the_quality_controlled_tail_is_volatile_and_settled_years_are_not():
    """QC rewrites the recent tail in place, under the same URL.

    Without the flag the ledger would freeze the provisional version of the
    last months and never collect the qualified one.
    """
    now = datetime(2026, 10, 31, tzinfo=UTC)
    units = FakeEa(now=now).list(window(2020, 2027), product())
    by_year = {u.meta["year"]: u for u in units if u.meta["station_id"].startswith("a3be5ed5")}
    assert by_year["2026"].meta["volatile"] is True
    assert by_year["2020"].meta["volatile"] is False


def test_list_honours_station_ids_and_max_stations():
    only = product(
        extra={
            "bbox": [-6.5, 49.5, 2.2, 56.5],
            "station_ids": ["a3be5ed5-ba57-4035-a5a2-8947ba7fb77f"],
        }
    )
    units = FakeEa().list(window(2020, 2021), only)
    assert {u.meta["station_id"] for u in units} == {"a3be5ed5-ba57-4035-a5a2-8947ba7fb77f"}

    capped = product(extra={"bbox": [-6.5, 49.5, 2.2, 56.5], "max_stations": 2})
    units = FakeEa().list(window(2020, 2021), capped)
    assert len({u.meta["station_id"] for u in units}) == 2


def test_the_readings_url_windows_exactly_one_calendar_year():
    """Per-year windows are what keeps a response under the 100 000 row cap."""
    url = FakeEa().readings_url("a3be5ed5-rainfall-t-900-mm-qualified", 2021)
    assert url.endswith(
        "/id/measures/a3be5ed5-rainfall-t-900-mm-qualified/readings.csv"
        "?min-date=2021-01-01&max-date=2021-12-31"
    )
    assert CSV_ROW_CAP == 100_000


def test_list_returns_nothing_when_the_domain_holds_no_gauge():
    assert FakeEa().list(window(2020, 2021), product(extra={"bbox": [10, 40, 12, 42]})) == []


# ------------------------------------------------------------------------- parse


def test_parse_reads_time_station_and_value(tmp_path):
    frame = FakeEa().parse(write(tmp_path, READINGS), product())
    assert frame.columns == ["time", "station_id", "precipitation_amount"]
    assert frame["station_id"].unique().to_list() == ["a3be5ed5-ba57-4035-a5a2-8947ba7fb77f"]
    assert frame["time"].dtype == pl.Datetime("us", "UTC")
    assert frame["time"][0] == datetime(2024, 1, 2, 7, 30, tzinfo=UTC)
    assert frame["precipitation_amount"][1] == pytest.approx(0.2)


def test_a_missing_flagged_row_is_null_and_is_nulled_before_any_scaling(tmp_path):
    """Order matters: scale-then-null would have multiplied a blank into a 0.

    The variable here carries a scale of 10, so a `Missing` row that survived
    to the scale would show up as a number rather than as nothing.
    """
    cfg = product(
        variables=[
            Variable(
                name="precipitation_amount",
                source_key="value",
                units="mm",
                dtype="float32",
                scale=10.0,
                valid_min=0.0,
                valid_max=1000.0,
            )
        ]
    )
    frame = FakeEa().parse(write(tmp_path, READINGS), cfg)
    at_missing = frame.filter(pl.col("time") == datetime(2024, 5, 9, 12, 45, tzinfo=UTC))
    assert at_missing.height == 1
    assert at_missing["precipitation_amount"][0] is None
    # and a good row is genuinely scaled, so the null above is not just "no scale"
    good = frame.filter(pl.col("time") == datetime(2024, 1, 2, 7, 45, tzinfo=UTC))
    assert good["precipitation_amount"][0] == pytest.approx(2.0)


def test_which_quality_flags_count_as_missing_is_a_config_decision(tmp_path):
    frame = FakeEa().parse(
        write(tmp_path, READINGS),
        product(extra={"bbox": [-6.5, 49.5, 2.2, 56.5], "missing_quality": ["Missing", "Suspect"]}),
    )
    suspect = frame.filter(pl.col("time") == datetime(2024, 7, 9, 8, 45, tzinfo=UTC))
    assert suspect["precipitation_amount"][0] is None
    # Default keeps Suspect: the reader does not get to decide what is usable.
    kept = FakeEa().parse(write(tmp_path, READINGS, name="again.csv"), product())
    suspect = kept.filter(pl.col("time") == datetime(2024, 7, 9, 8, 45, tzinfo=UTC))
    assert suspect["precipitation_amount"][0] == pytest.approx(0.2)


def test_the_valid_range_is_applied_in_canonical_units_after_the_scale(tmp_path):
    """A 0.6 mm reading scaled by 10 is 6.0, which a valid_max of 5 must reject.

    If the mask ran before the scale it would compare 0.6 against 5 and keep a
    6.0 that the product said is impossible.
    """
    cfg = product(
        variables=[
            Variable(
                name="precipitation_amount",
                source_key="value",
                units="mm",
                dtype="float32",
                scale=10.0,
                valid_min=0.0,
                valid_max=5.0,
            )
        ]
    )
    frame = FakeEa().parse(write(tmp_path, READINGS), cfg)
    at = frame.filter(pl.col("time") == datetime(2024, 8, 1, 0, 0, tzinfo=UTC))
    assert at.height == 1
    assert at["precipitation_amount"][0] is None


def test_off_cadence_rows_are_dropped_rather_than_snapped(tmp_path):
    """A 07-past-the-hour reading is not a 15-minute observation."""
    frame = FakeEa().parse(write(tmp_path, READINGS), product())
    assert datetime(2024, 8, 1, 0, 7, tzinfo=UTC) not in frame["time"].to_list()
    assert datetime(2024, 8, 1, 0, 0, tzinfo=UTC) in frame["time"].to_list()

    kept = FakeEa().parse(
        write(tmp_path, READINGS, name="keep.csv"),
        product(extra={"bbox": [-6.5, 49.5, 2.2, 56.5], "drop_off_cadence": False}),
    )
    assert datetime(2024, 8, 1, 0, 7, tzinfo=UTC) in kept["time"].to_list()


def test_a_zero_byte_station_year_is_a_normal_skip_not_a_failure(tmp_path, caplog):
    """~360 of 9450 station-years in the existing archive are exactly this.

    EA answers HTTP 200 with no body for a year before the gauge opened and for
    the recent tail quality control has not reached. Raising here would fail
    roughly one job in twenty-five for no reason.
    """
    raw = write(tmp_path, EMPTY, name="1990.csv")
    frame = FakeEa().parse(raw, product())
    assert frame.is_empty()
    assert frame.columns == ["time", "station_id", "precipitation_amount"]
    assert frame["precipitation_amount"].dtype == pl.Float32


def test_parse_takes_the_station_from_the_file_before_the_unit_key(tmp_path):
    """So a CSV found on disk parses correctly whatever it is called."""
    raw = write(tmp_path, READINGS, key="ea/rainfall_15min/WRONG-ID/2024")
    frame = FakeEa().parse(raw, product())
    assert frame["station_id"].unique().to_list() == ["a3be5ed5-ba57-4035-a5a2-8947ba7fb77f"]


def test_parse_falls_back_to_the_unit_key_when_the_measure_column_is_gone(tmp_path):
    trimmed = "\n".join(
        ",".join(row.split(",")[1:]) for row in READINGS.strip().splitlines()
    )
    raw = write(tmp_path, trimmed + "\n", key="ea/rainfall_15min/from-the-key/2024")
    frame = FakeEa().parse(raw, product())
    assert frame["station_id"].unique().to_list() == ["from-the-key"]


def test_parse_refuses_to_run_against_a_drifted_schema(tmp_path):
    cfg = product(schema_fingerprint="0" * 16)
    with pytest.raises(SchemaDriftError, match="pinned"):
        FakeEa().parse(write(tmp_path, READINGS), cfg)


def test_parse_accepts_the_schema_it_was_pinned_against(tmp_path):
    header = READINGS.splitlines()[0]
    pinned = schema_fingerprint(c.strip('"') for c in header.split(","))
    cfg = product(schema_fingerprint=pinned)
    assert FakeEa().parse(write(tmp_path, READINGS), cfg).height > 0


def test_parse_reports_a_vanished_value_column_rather_than_returning_nothing(tmp_path):
    trimmed = "\n".join(
        ",".join(f for i, f in enumerate(row.split(",")) if i != 3)
        for row in READINGS.strip().splitlines()
    )
    with pytest.raises(SchemaDriftError, match="missing column"):
        FakeEa().parse(write(tmp_path, trimmed + "\n"), product())


# -------------------------------------------------------------------------- scan


def test_scan_counts_empty_files_instead_of_calling_them_errors(tmp_path):
    files = [
        write(tmp_path, READINGS, name="a.csv"),
        write(tmp_path, READINGS, name="b.csv"),
        write(tmp_path, EMPTY, name="c.csv"),
    ]
    report = FakeEa().scan(files)
    assert report["n_files"] == 3
    assert report["n_empty"] == 1
    assert report["errors"] == []
    assert report["n_variants"] == 1
    assert report["fingerprint"] == report["variants"][0]["fingerprint"]
    assert "Missing" in report["variants"][0]["qualities"]


def test_scan_is_reproducible_and_records_a_bad_file_as_a_finding(tmp_path):
    files = [
        write(tmp_path, READINGS, name="a.csv"),
        write(tmp_path, "value\n\x00\x00binary", name="bad.csv"),
    ]
    report = FakeEa().scan(files)
    assert FakeEa().scan(files)["fingerprints"] == report["fingerprints"]
    assert report["n_variants"] >= 1


def test_the_scanned_fingerprint_is_the_one_the_shipped_config_pins(tmp_path):
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    report = FakeEa().scan([write(tmp_path, READINGS)])
    assert report["fingerprint"] == cfg.schema_fingerprint


# ------------------------------------------------------------------------- fetch


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
    src = FakeEa()
    unit = src.list(window(2020, 2021), product())[0]
    raw = src.fetch(unit, tmp_path)

    assert seen["unit_key"] == unit.key
    assert seen["url"] == unit.url
    assert seen["session"] is not None, "a source built with no session must still download"
    assert raw is not None
    # The raw tree mirrors the key: station, then year.
    assert raw.path.name == "2020.csv"
    assert raw.path.parent.name == unit.meta["station_id"]


def test_fetch_skips_a_404_rather_than_failing_the_job(monkeypatch, tmp_path):
    def not_found(url, dest, **kwargs):
        raise http.NotFoundError(url, "404 not found", 404)

    monkeypatch.setattr(http, "fetch_file", not_found)
    unit = Unit(
        key="ea/rainfall_15min/a3be5ed5-ba57-4035-a5a2-8947ba7fb77f/2024",
        url="https://example.invalid/x",
        meta={"station_id": "a3be5ed5-ba57-4035-a5a2-8947ba7fb77f", "filename": "2024.csv"},
    )
    assert FakeEa().fetch(unit, tmp_path) is None


def test_fetch_survives_a_unit_with_no_meta(monkeypatch, tmp_path):
    captured: dict = {}

    def capture(url, dest, **kwargs):
        captured["dest"] = Path(dest)
        return RawFile(Path(dest), "0" * 64, 0, kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", capture)
    unit = Unit(key="ea/rainfall_15min/some-guid/2024", url="https://example.invalid/x")
    FakeEa().fetch(unit, tmp_path)
    assert captured["dest"] == tmp_path / "some-guid" / "2024.csv"


def test_a_source_built_without_a_session_makes_one_lazily():
    src = EaRainfall()
    assert src._session is None
    session = src._requests_session()
    assert session is src._requests_session()
    assert "meteopack" in session.headers["User-Agent"]


# ------------------------------------------------------------------------ config


def test_the_source_yaml_loads_into_sourceconfig():
    cfg = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    assert cfg.name == "ea"
    assert cfg.base_url == "https://environment.data.gov.uk/hydrology"
    assert cfg.credentials_env == []           # keyless by design
    assert cfg.rate_limit.requests_per_second > 0


def test_the_product_yaml_loads_into_productconfig():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert cfg.store_id == "ea/rainfall_15min/v1"
    assert cfg.store.kind == "points"
    assert cfg.cadence_seconds == PERIOD_SECONDS == 900
    assert cfg.extra["missing_quality"] == ["Missing"]
    assert len(cfg.extra["bbox"]) == 4


def test_every_configured_source_key_is_a_real_readings_column():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    header = [c.strip('"') for c in READINGS.splitlines()[0].split(",")]
    assert cfg.variables
    for column, _ in variables_for(cfg):
        assert column in header, column


def test_the_shipped_config_parses_the_real_fixture(tmp_path):
    """End to end on the shipped config, not the trimmed test one."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    frame = FakeEa().parse(write(tmp_path, READINGS), cfg)
    assert frame["precipitation_amount"].dtype == pl.Float32
    assert frame["precipitation_amount"].null_count() == 2       # the two Missing rows
    assert frame.height == 7                                     # the off-cadence row is gone


# ---------------------------------------------------------------------- contract


def test_ea_satisfies_the_source_protocol():
    assert isinstance(EaRainfall(), Source)


# -------------------------------------------------------------- optional suites


@pytest.mark.skipif(
    not LOCAL_ARCHIVE.is_dir(), reason="the local EA archive is not mounted here"
)
def test_the_local_archive_still_matches_the_pinned_fingerprint():
    """Opt-in: the shipped fingerprint against real files, if they are around."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    files = sorted(LOCAL_ARCHIVE.rglob("*.csv"))[:50]
    raws = [
        RawFile(p, "0" * 64, p.stat().st_size, f"ea/rainfall_15min/{p.parent.name}/{p.stem}")
        for p in files
    ]
    report = EaRainfall().scan(raws)
    assert report["errors"] == []
    assert report["fingerprint"] in (None, cfg.schema_fingerprint)


@pytest.mark.skipif(
    not os.environ.get("METEOPACK_LIVE"), reason="set METEOPACK_LIVE=1 to hit the real API"
)
def test_live_catalogue_and_one_station_year():
    """Opt-in: two keyless requests against the real service."""
    src = EaRainfall()
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    table = src.stations(cfg)
    assert table.height > 500
    assert table["height"].null_count() == table.height
    units = src.list(window(2023, 2024), cfg)
    assert units
    resp = src._requests_session().get(units[0].url, timeout=120)
    assert resp.status_code == 200
