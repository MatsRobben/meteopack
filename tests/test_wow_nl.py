"""WOW-NL: a lossless level-1 store from hourly JSON members in yearly zips.

Every test builds its own tiny archive, so the rules are pinned without the
66 GB one: nothing in a report is lost or coerced, only exact duplicates are
dropped, and every hour of the archive is accounted for.
"""
from __future__ import annotations

import json
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from meteopack.cli import main
from meteopack.jobs.plan import load_product_config
from meteopack.sources.stations import wow_nl
from meteopack.sources.stations.wow_nl import Catalogue, WowNl, decode_member, parse_day


@pytest.fixture(scope="module")
def cfg():
    return load_product_config("wow_nl", "reports")


def report(site="901", when="2024-06-15T12:59:59+00:00", **fields) -> dict:
    base = {"SiteId": site, "ReportStartDateTime": when, "ReportEndDateTime": when,
            "Id": f"id-{site}-{when}", "DryBulbTemperature_Celsius": 16.72222222222222,
            "DryBulbTemperature_Fahrenheit": 62.1, "IsOfficial": False, "Latitude": 51.4,
            "Longitude": 5.38, "Height": 0, "Version": 1, "ExternalId": 12425583706,
            # every configured variable, so `validate` finds each one filled
            "DewPointTemperature_Celsius": 9.9, "RelativeHumidity": 64,
            "MeanSeaLevelPressure_Hectopascal": 1013.2, "AirPressure_Hectopascal": 1003.3,
            "WindSpeed_MetrePerSecond": 1.2, "WindGust_MetrePerSecond": 3.4,
            "WindDirection": 157, "WindGustDirection": 160, "WetBulbTemperature_Celsius": 12.5,
            "RainfallRate_MillimetrePerHour": 0, "RainfallAmount_Millimetre": 0.3}
    base.update(fields)
    return {k: v for k, v in base.items() if v is not _ABSENT}


_ABSENT = object()


def page(*reports, **top) -> bytes:
    doc = {"NextHashKey": "abc", "Total": 0, "Object": list(reports), **top}
    return json.dumps(doc, indent=2).encode()


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{path.stem}/", b"")
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def member(cfg, raw: bytes) -> wow_nl.Member:
    m = wow_nl.Member(zip="2024.zip", name="2024/WOW_NL_2024061512.json",
                      hour=datetime(2024, 6, 15, 12, tzinfo=UTC), bytes=len(raw),
                      compressed_bytes=0)
    return decode_member(raw, Catalogue(cfg), m)


# ----------------------------------------------------------------- decoding


def test_values_are_kept_exactly_and_names_follow_the_config(cfg):
    m = member(cfg, page(report(AirPressure_Hectopascal=1003.3870606999999)))
    assert m.status == "ok" and m.path == "typed"
    row = m.frame.row(0, named=True)
    assert row["air_temperature"] == 16.72222222222222          # renamed, not rounded
    assert row["DryBulbTemperature_Fahrenheit"] == 62.1          # the imperial twin is kept
    assert row["air_pressure"] == 1003.3870606999999
    assert row["station_id"] == "901"
    assert row["Height"] == 0.0 and row["ExternalId"] == 12425583706
    assert m.frame.schema["ExternalId"] == pl.Int64
    assert row[wow_nl.UNPINNED] is None


def test_sub_second_times_are_parsed_and_the_string_is_kept(cfg):
    when = "2021-07-14T12:59:59.8364285+00:00"
    m = member(cfg, page(report(when=when)))
    row = m.frame.row(0, named=True)
    assert row["time"] == datetime(2021, 7, 14, 12, 59, 59, 836428)    # truncated to µs
    assert row["ReportEndDateTime"] == when                              # all 7 digits kept


def test_a_number_sent_as_text_is_kept_verbatim_with_a_reason(cfg):
    """KNMI's A1.05 report: WOW stations send e.g. wind "36,612.3" as a string."""
    m = member(cfg, page(report(WindSpeed_MetrePerSecond="36,612.3"), report(site="902")))
    assert m.path == "exact"
    first = m.frame.row(0, named=True)
    assert first["wind_speed"] is None
    kept = json.loads(first[wow_nl.UNPINNED])
    assert kept == {"WindSpeed_MetrePerSecond": {"raw": "36,612.3",
                                                 "reason": "string_for_float64"}}
    assert m.unpinned_counts == {"WindSpeed_MetrePerSecond|string_for_float64": 1}
    assert m.frame.row(1, named=True)[wow_nl.UNPINNED] is None


@pytest.mark.parametrize("field, value, reason", [
    ("DryBulbTemperature_Celsius", None, "null_for_float64"),
    ("DryBulbTemperature_Celsius", True, "bool_for_float64"),
    ("Version", 2.0, "float_for_int64"),
    ("IsOfficial", 1, "int_for_bool"),
    ("SiteId", 901, "int_for_string"),
    ("SomethingNew", 3, "unknown_field"),
    ("DryBulbTemperature_Celsius", 2**60 + 1, "int_inexact_for_float64"),
])
def test_nothing_is_coerced(cfg, field, value, reason):
    """polars' constructor turns True into 1.0, "1.5" into 1.5 and 2.5 into 2 for
    an Int64 column (all measured); none of that may happen here."""
    m = member(cfg, page(report(**{field: value})))
    kept = json.loads(m.frame.row(0, named=True)[wow_nl.UNPINNED])
    assert kept[field] == {"raw": value, "reason": reason}


def test_a_damaged_member_is_salvaged_up_to_the_damage(cfg):
    raw = page(report(site="1"), report(site="2"), report(site="3"))
    cut = raw[: raw.index(b'"3"') + 10]                         # truncated inside report 3
    m = member(cfg, cut)
    assert m.status == "salvaged" and m.n_records == 2
    assert sorted(m.frame["station_id"]) == ["1", "2"]
    assert "2 complete report(s) recovered" in m.error


def test_an_empty_member_is_recorded_not_skipped(cfg):
    m = member(cfg, b"")
    assert m.status == "empty_file" and m.frame is None


def test_an_unparseable_time_fails_the_member_loudly(cfg):
    m = member(cfg, page(report(when="yesterday")))
    assert m.status == "error" and m.frame is None and "ReportEndDateTime" in m.error


def test_the_catalogue_pin_is_checked(cfg):
    bad = cfg.model_copy(update={"schema_fingerprint": "0000000000000000"})
    with pytest.raises(wow_nl.SchemaDriftError, match="(?i)re-pin"):
        Catalogue(bad)


# ------------------------------------------------------- copies of an hour


def test_two_copies_of_an_hour_keep_one_of_each_report(tmp_path, cfg):
    """2019 is in 2019.zip and 2020.zip. Identical reports are kept once, from the
    copy with more records; a report only one copy has is kept as well."""
    same = [report(site=str(i)) for i in range(3)]
    only_small = report(site="9", DryBulbTemperature_Celsius=1.5)
    name = "2019/WOW_NL_2019101910.json"
    a = write_zip(tmp_path / "2019.zip", {name: page(*same[:2], only_small)})
    b = write_zip(tmp_path / "2020.zip", {name: page(*same)})
    wow_nl._ZIP_INDEX.clear()
    frame, records = parse_day([str(a), str(b)], date(2019, 10, 19), Catalogue(cfg))
    assert frame.height == 4
    by_zip = {r["zip"]: r for r in records}
    assert by_zip["2020.zip"]["chosen"] and not by_zip["2019.zip"]["chosen"]
    assert by_zip["2019.zip"]["n_duplicate_dropped"] == 2
    assert by_zip["2019.zip"]["n_stored"] == 1
    assert frame.filter(pl.col("station_id") == "9")[wow_nl.SOURCE_ZIP].item() == "2019.zip"
    assert set(frame.filter(pl.col("station_id") != "9")[wow_nl.SOURCE_ZIP]) == {"2020.zip"}


def test_reports_differing_in_one_field_are_both_kept(tmp_path, cfg):
    r = report(site="5")
    name = "2024/WOW_NL_2024061512.json"
    z = write_zip(tmp_path / "2024.zip", {name: page(r, {**r, "RelativeHumidity": 50})})
    wow_nl._ZIP_INDEX.clear()
    frame, records = parse_day([str(z)], date(2024, 6, 15), Catalogue(cfg))
    assert frame.height == 2 and records[0]["n_duplicate_dropped"] == 0


def test_an_exact_duplicate_inside_one_member_is_dropped_and_counted(tmp_path, cfg):
    r = report(site="5")
    z = write_zip(tmp_path / "2024.zip", {"2024/WOW_NL_2024061512.json": page(r, r)})
    wow_nl._ZIP_INDEX.clear()
    frame, records = parse_day([str(z)], date(2024, 6, 15), Catalogue(cfg))
    assert frame.height == 1 and records[0]["n_duplicate_dropped"] == 1


# --------------------------------------------------------------- the store


def _archive(tmp_path: Path) -> Path:
    """Two months of a tiny archive; one hour of June in two zips; July has a gap."""
    arch = tmp_path / "archive"
    hours = {f"2024/WOW_NL_202406{d:02d}{h:02d}.json": page(
        report(site="1", when=f"2024-06-{d:02d}T{h:02d}:10:00+00:00"),
        report(site="2", when=f"2024-06-{d:02d}T{h:02d}:20:00+00:00", Latitude=51.5 + d))
        for d in (1, 30) for h in (0, 23)}
    hours["2024/WOW_NL_2024070100.json"] = page(
        report(site="1", when="2024-07-01T00:00:00+00:00"))
    hours["2024/WOW_NL_2024070102.json"] = page(
        report(site="1", when="2024-07-01T02:00:00.5+00:00"))
    write_zip(arch / "2024.zip", hours)
    write_zip(arch / "2025.zip", {"2024/WOW_NL_2024060100.json": hours[
        "2024/WOW_NL_2024060100.json"]})
    return arch


def _root(tmp_path: Path, monkeypatch) -> Path:
    arch = _archive(tmp_path)
    root = tmp_path / "data"
    (root / "raw" / "wow_nl").mkdir(parents=True)
    (root / "raw" / "wow_nl" / "reports").symlink_to(arch)
    monkeypatch.setenv("WOW_NL_ARCHIVE", str(arch))
    monkeypatch.setenv("METEOPACK_SPILL", str(tmp_path / "spill"))
    wow_nl._ZIP_INDEX.clear()
    return root


def _pack(root: Path, *extra: str) -> int:
    return main(["pack", "wow_nl", "--data-root", str(root), "--workers", "2", *extra])


def _store(root: Path) -> Path:
    return root / "packed" / "wow_nl" / "reports" / "v1"


def test_pack_validate_and_account_for_every_hour(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path, monkeypatch)
    assert _pack(root) == 0, capsys.readouterr().out
    store = _store(root)
    obs = pl.read_parquet(store / "observations" / "**" / "*.parquet", hive_partitioning=False)
    # 8 June + 2 July; the 2025.zip copy of 06-01 00h is two exact duplicates
    assert obs.height == 10
    june = pl.read_parquet(store / "observations" / "year=2024" / "month=06" / "data.parquet")
    assert june.select("station_id", "time").equals(june.select("station_id", "time")
                                                    .sort("station_id", "time"))
    hours = pl.read_parquet(store / "hours" / "**" / "*.parquet", hive_partitioning=False)
    july = hours.filter(pl.col("hour") >= datetime(2024, 7, 1))
    assert july.filter(pl.col("status") == "missing")["hour"].to_list() == [
        datetime(2024, 7, 1, 1)]
    june = hours.filter(pl.col("hour") < datetime(2024, 7, 1))
    assert june.filter(pl.col("status") == "missing").height == 30 * 24 - 4
    stations = pl.read_parquet(store / "stations" / "data.parquet")
    assert set(stations["station_id"]) == {"1", "2"}
    two = stations.filter(pl.col("station_id") == "2").row(0, named=True)
    assert two["lat"] == 51.5 + 30 and two["n_positions"] == 2
    history = pl.read_parquet(store / "station_positions.parquet")
    assert history.filter(pl.col("station_id") == "1").height == 1   # one run across months
    capsys.readouterr()
    assert main(["validate", "wow_nl", "--data-root", str(root)]) == 0, capsys.readouterr().out


def test_a_windowed_repack_replaces_its_month_and_nothing_else(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path, monkeypatch)
    assert _pack(root) == 0
    june = _store(root) / "observations" / "year=2024" / "month=06" / "data.parquet"
    before = june.read_bytes()
    july = _store(root) / "observations" / "year=2024" / "month=07" / "data.parquet"
    july.unlink()
    assert _pack(root, "--window", "2024-07-01T01:00/2024-07-01T02:00") == 0
    out = capsys.readouterr().out
    assert "snapped to whole partitions" in out and "1 partition(s)" in out
    assert june.read_bytes() == before, "a windowed pack touched another partition"
    assert pl.read_parquet(july).height == 2
    capsys.readouterr()
    assert main(["validate", "wow_nl", "--data-root", str(root)]) == 0


def test_a_windowed_repack_refuses_a_store_built_from_another_config(
        tmp_path, monkeypatch, capsys):
    from meteopack.core.store import manifest

    root = _root(tmp_path, monkeypatch)
    assert _pack(root) == 0
    path = _store(root) / manifest.MANIFEST_NAME
    data = json.loads(path.read_text())
    data["content_hash"] = "somethingelse"
    path.write_text(json.dumps(data))
    assert _pack(root, "--window", "2024-07/2024-07") == 2
    assert "different config" in capsys.readouterr().out


def test_validate_notices_a_report_stored_twice(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path, monkeypatch)
    assert _pack(root) == 0
    july = _store(root) / "observations" / "year=2024" / "month=07" / "data.parquet"
    frame = pl.read_parquet(july)
    pl.concat([frame, frame.head(1)]).write_parquet(july)
    capsys.readouterr()
    assert main(["validate", "wow_nl", "--data-root", str(root)]) == 1
    out = capsys.readouterr().out
    assert "FAIL  no report stored twice" in out, out
    assert "FAIL  rows = sum of n_stored" in out


def test_parse_reads_days_not_whole_zips(tmp_path, monkeypatch, capsys):
    root = _root(tmp_path, monkeypatch)
    assert main(["parse", "wow_nl", "--data-root", str(root), "--limit", "2"]) == 0
    out = capsys.readouterr().out
    assert "2024-06-01: 4 rows" in out


def test_pack_parts_are_days_across_zips_in_time_order(tmp_path, monkeypatch, cfg):
    _root(tmp_path, monkeypatch)
    arch = tmp_path / "archive"
    parts = WowNl().pack_parts(sorted(arch.glob("*.zip")), cfg, None)
    assert [p[0] for p in parts] == ["2024-06-01", "2024-06-30", "2024-07-01"]
    assert len(parts[0][1]) == 2 and parts[0][2] == "2024/06"


def test_packing_twice_writes_the_same_bytes(tmp_path, monkeypatch):
    """The order rows reach a month's writer depends on which worker finishes
    first; the file must not."""
    root = _root(tmp_path, monkeypatch)
    assert _pack(root) == 0
    files = sorted((_store(root) / "observations").rglob("*.parquet"))
    first = [f.read_bytes() for f in files]
    assert _pack(root, "--workers", "3") == 0
    assert [f.read_bytes() for f in files] == first
