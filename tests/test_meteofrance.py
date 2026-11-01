"""Tests for the Météo-France BASE station source.

Nothing here touches the network and nothing requires the local archive. The
CSV fixture is a column subset of the real `H_59`/`H_21` files with verbatim
rows for three real stations, plus two rows added and labelled to exercise an
edge those rows happen not to contain. The bucket listing is a real
ListObjectsV2 response, trimmed, with the sizes and modification times the live
bucket returned on 2026-10-31.

The tests at the end run against the full local archive when it is present and
are skipped when it is not.
"""
from __future__ import annotations

import gzip
import inspect
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from meteopack.core.config import ProductConfig, SourceConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.meteofrance import (
    DEPARTEMENTS,
    MeteoFranceStations,
    SchemaDriftError,
    parse_listing,
    product_fingerprint,
    schema_fingerprint,
    station_id,
    unit_key,
    variables_by_column,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "meteofrance.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "meteofrance" / "hourly_all.yaml"

LOCAL_ARCHIVE = Path(
    "/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/meteofrance/hourly"
)


# --------------------------------------------------------------------- fixtures

# A real ListObjectsV2 page for `prefix=data/synchro_ftp/BASE/HOR/H_59`, trimmed
# to six of its eighteen keys. Sizes and timestamps are the live ones; note that
# `latest-2025-2026` was six hours old when this was captured and 18% larger
# than a copy taken in July, which is the whole reason it is volatile.
LISTING_XML = """<?xml version='1.0' encoding='UTF-8'?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
<Name>meteofrance</Name><Prefix>data/synchro_ftp/BASE/HOR/H_59</Prefix>
<KeyCount>6</KeyCount><MaxKeys>1000</MaxKeys><IsTruncated>false</IsTruncated>
<Contents><Key>data/synchro_ftp/BASE/HOR/H_59_1840-1849.csv.gz</Key>
<LastModified>2026-06-25T03:26:36.000Z</LastModified>
<ETag>"df623b146fb52270e909b4769fbff7f3"</ETag><Size>3944</Size>
<StorageClass>STANDARD</StorageClass></Contents>
<Contents><Key>data/synchro_ftp/BASE/HOR/H_59_1850-1859.csv.gz</Key>
<LastModified>2026-06-25T03:27:44.000Z</LastModified>
<ETag>"e1cd143ee6a76617183614a1305b8cca"</ETag><Size>35449</Size>
<StorageClass>STANDARD</StorageClass></Contents>
<Contents><Key>data/synchro_ftp/BASE/HOR/H_59_2000-2009.csv.gz</Key>
<LastModified>2026-06-25T03:41:02.000Z</LastModified>
<ETag>"0398703c524656442ef3b6bca60d8ceb"</ETag><Size>14882110</Size>
<StorageClass>STANDARD</StorageClass></Contents>
<Contents><Key>data/synchro_ftp/BASE/HOR/H_59_2010-2019.csv.gz</Key>
<LastModified>2026-06-25T03:44:19.000Z</LastModified>
<ETag>"5c2af78f822f13f394a1f58b0820a14b"</ETag><Size>16205112</Size>
<StorageClass>STANDARD</StorageClass></Contents>
<Contents><Key>data/synchro_ftp/BASE/HOR/H_59_previous-2020-2024.csv.gz</Key>
<LastModified>2026-10-02T06:03:12.000Z</LastModified>
<ETag>"901b583b3d639f5d7c262eff0dd59fc5"</ETag><Size>15928750</Size>
<StorageClass>STANDARD</StorageClass></Contents>
<Contents><Key>data/synchro_ftp/BASE/HOR/H_59_latest-2025-2026.csv.gz</Key>
<LastModified>2026-10-31T06:44:18.000Z</LastModified>
<ETag>"a744e878c9f45249a2097f0b58354495"</ETag><Size>5926627</Size>
<StorageClass>STANDARD</StorageClass></Contents>
<Contents><Key>data/synchro_ftp/BASE/HOR/HOR_descriptif_champs.csv</Key>
<LastModified>2026-06-25T03:20:00.000Z</LastModified><ETag>"x"</ETag><Size>9000</Size>
<StorageClass>STANDARD</StorageClass></Contents>
</ListBucketResult>"""

TRUNCATED_XML = LISTING_XML.replace(
    "<IsTruncated>false</IsTruncated>",
    "<IsTruncated>true</IsTruncated><NextContinuationToken>abc123</NextContinuationToken>",
)

# A column subset of the real header. The `Q*` flags are kept because the
# fingerprint deliberately includes them.
CSV_HEADER = (
    "NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJHH;RR1;QRR1;T;QT;U;N;FF;DD;"
    "GLO;NEIGETOT;INS;VV;WW"
)

# Verbatim rows. What each one is here to prove:
#   59343001 LILLE-LESQUIN  - in domain; GLO and INS are EMPTY, which is how the
#                             file says missing, and N is 8 oktas = full cover
#   59606004 VALENCIENNES   - in domain; GLO and INS populated, so the J/cm2 ->
#                             W/m2 and minutes -> seconds scales get exercised
#   21473001 DIJON-LONGVIC  - a real station in a listed département that is
#                             nonetheless outside the radar footprint
CSV_ROWS = [
    "59343001;LILLE-LESQUIN;50.570000;3.097500;47;2025010100;0.0;1;7.8;1;86;8;11.9;200;;0;;26864;0",
    "59343001;LILLE-LESQUIN;50.570000;3.097500;47;2025010101;0.0;1;7.6;1;86;8;11.3;200;;0;;28689;0",
    "59606004;VALENCIENNES;50.328333;3.461000;50;2025061512;0.0;1;21.1;1;53;;6.2;280;268;;40;;",
    "59606004;VALENCIENNES;50.328333;3.461000;50;2025061513;0.0;1;21.3;1;53;;6.1;270;251;;34;;",
    "21473001;DIJON-LONGVIC;47.267833;5.088333;219;2025010100;0.0;1;-1.3;1;100;9;2.2;200;0;0;0;240;49",
    # ADDED: N=9 is the "sky obscured" code, not a ninth okta. At 12.5% per
    # okta that scales to 112.5%, which the range then nulls - and which an
    # unscaled range check would have waved through.
    "59343001;LILLE-LESQUIN;50.570000;3.097500;47;2025010102;0.0;1;7.2;1;86;9;10.9;190;;0;;36361;0",
    # ADDED: 120 minutes of sunshine inside one hour. 120 is an unremarkable
    # number; 7200 seconds is not.
    "59606004;VALENCIENNES;50.328333;3.461000;50;2025061514;0.0;1;21.4;1;52;;6.0;270;240;;120;;",
]


def _csv(tmp_path: Path, name: str = "H_59_2001-2009.csv.gz", rows=None) -> RawFile:
    """A fetched département file on disk, as `fetch` would have left it."""
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join([CSV_HEADER, *(CSV_ROWS if rows is None else rows)]) + "\n"
    path.write_bytes(gzip.compress(body.encode("utf-8")))
    return RawFile(
        path=path,
        sha256="0" * 64,
        n_bytes=path.stat().st_size,
        unit_key=f"meteofrance/hourly_all/{name[: -len('.csv.gz')]}",
    )


@pytest.fixture
def product() -> ProductConfig:
    return ProductConfig.model_validate(yaml.safe_load(PRODUCT_YAML.read_text()))


@pytest.fixture
def source_cfg() -> SourceConfig:
    return SourceConfig.model_validate(yaml.safe_load(SOURCE_YAML.read_text()))


@pytest.fixture
def src(source_cfg: SourceConfig) -> MeteoFranceStations:
    return MeteoFranceStations(source_cfg, now=datetime(2026, 10, 31, 12, tzinfo=UTC))


@pytest.fixture
def offline(monkeypatch, src: MeteoFranceStations) -> MeteoFranceStations:
    """Serve one département's bucket listing from memory; the rest are empty."""

    def fake_get(url: str) -> str:
        if "H_59_" in url:
            return LISTING_XML
        return LISTING_XML.replace("<Contents>", "<!--").replace("</Contents>", "-->")

    monkeypatch.setattr(src, "_get_text", fake_get)
    return src


# ------------------------------------------------------------------- the contract


def test_meteofrance_stations_satisfies_the_source_protocol(source_cfg):
    assert isinstance(MeteoFranceStations(source_cfg), Source)
    assert isinstance(MeteoFranceStations(), Source)  # also with no config at all


def test_both_yamls_load_into_their_config_models():
    source = SourceConfig.model_validate(yaml.safe_load(SOURCE_YAML.read_text()))
    product = ProductConfig.model_validate(yaml.safe_load(PRODUCT_YAML.read_text()))
    assert source.name == "meteofrance"
    assert product.source == "meteofrance" and product.product == "hourly_all"
    assert product.cadence_seconds == 3600
    assert product.store.kind == "points"


def test_the_product_selects_the_hourly_feed_by_configuration_alone(product):
    """The 6-minute feed is the same reader with four strings changed.

    If any of these moved into the module, adding it would mean a second
    reader rather than a second YAML.
    """
    assert product.extra["s3_subdir"] == "HOR"
    assert product.extra["file_prefix"] == "H"
    assert product.extra["date_column"] == "AAAAMMJJHH"
    assert product.extra["date_format"] == "%Y%m%d%H"


def test_the_six_minute_variant_needs_no_code(src, product, tmp_path):
    """Point the same source at BASE/MIN and it reads the 6-minute files."""
    cfg = product.model_copy(
        update={
            "cadence_seconds": 360,
            "variables": [Variable(name="precipitation_amount", source_key="RR", units="mm")],
            "extra": {
                **product.extra,
                "s3_subdir": "MIN",
                "file_prefix": "MN",
                "date_column": "AAAAMMJJHHMN",
                "date_format": "%Y%m%d%H%M",
                "schema_fingerprints": {},
            },
        }
    )
    raw = _csv(
        tmp_path,
        "MN_59_2001-2009.csv.gz",
        rows=["59343001;LILLE-LESQUIN;50.570000;3.097500;47;202501010006;0.2;1"],
    )
    # The header has to match the 6-minute product's own columns.
    path = Path(raw.path)
    path.write_bytes(
        gzip.compress(
            b"NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJHHMN;RR;QRR\n"
            b"59343001;LILLE-LESQUIN;50.570000;3.097500;47;202501010006;0.2;1\n"
        )
    )
    df = src.parse(raw, cfg)
    assert df["time"].to_list() == [datetime(2025, 1, 1, 0, 6, tzinfo=UTC)]
    assert df["precipitation_amount"][0] == pytest.approx(0.2)
    assert "/BASE/MIN/MN_59_" in src.listing_url(cfg, "59")


def test_a_column_claimed_twice_is_rejected(product):
    cfg = product.model_copy(
        update={
            "variables": [
                Variable(name="a", source_key="T", units="degC"),
                Variable(name="b", source_key="T", units="degC"),
            ]
        }
    )
    with pytest.raises(ValueError, match="claimed by both"):
        variables_by_column(cfg)


def test_station_id_keeps_its_leading_departement_digits():
    assert station_id(" 02100001 ") == "02100001"
    with pytest.raises(ValueError):
        station_id("  ")


# ------------------------------------------------------------------- listings


def test_listing_parses_keys_sizes_and_timestamps():
    files, token = parse_listing(LISTING_XML)

    assert token is None
    assert [f.stem for f in files] == [
        "H_59_1840-1849",
        "H_59_1850-1859",
        "H_59_2000-2009",
        "H_59_2010-2019",
        "H_59_previous-2020-2024",
        "H_59_latest-2025-2026",
    ]
    latest = files[-1]
    assert latest.size == 5926627
    assert latest.last_modified == "2026-10-31T06:44:18.000Z"
    assert latest.dep == "59"


def test_listing_skips_keys_that_are_not_departement_files():
    files, _ = parse_listing(LISTING_XML)
    # HOR_descriptif_champs.csv shares the prefix and is not data.
    assert not [f for f in files if "descriptif" in f.key]


def test_listing_reports_a_continuation_token():
    _, token = parse_listing(TRUNCATED_XML)
    assert token == "abc123"


def test_only_the_latest_file_is_volatile():
    files = {f.stem: f for f in parse_listing(LISTING_XML)[0]}
    assert files["H_59_latest-2025-2026"].volatile is True
    assert files["H_59_previous-2020-2024"].volatile is False
    assert files["H_59_2010-2019"].volatile is False


def test_the_latest_file_has_no_end_year():
    """`latest-2025-2026` keeps that name into 2027 and beyond.

    Believing the printed end year would stop listing the live file exactly
    when a window first reaches past it — i.e. for the newest data, which is
    the data that only exists there.
    """
    files = {f.stem: f for f in parse_listing(LISTING_XML)[0]}
    latest = files["H_59_latest-2025-2026"]
    assert latest.years == (2025, None)
    assert latest.covers(
        datetime(2028, 1, 1, tzinfo=UTC), datetime(2028, 2, 1, tzinfo=UTC)
    )
    # A closed period does have an end, and is not listed past it.
    closed = files["H_59_2010-2019"]
    assert closed.years == (2010, 2019)
    assert not closed.covers(
        datetime(2021, 1, 1, tzinfo=UTC), datetime(2022, 1, 1, tzinfo=UTC)
    )


def test_a_period_is_matched_on_overlap_not_containment():
    files = {f.stem: f for f in parse_listing(LISTING_XML)[0]}
    decade = files["H_59_2010-2019"]
    assert decade.covers(datetime(2019, 12, 1, tzinfo=UTC), datetime(2020, 2, 1, tzinfo=UTC))
    assert decade.covers(datetime(2005, 1, 1, tzinfo=UTC), datetime(2011, 1, 1, tzinfo=UTC))
    # The window is half-open: ending exactly at 2010-01-01 does not reach 2010.
    assert not decade.covers(
        datetime(2005, 1, 1, tzinfo=UTC), datetime(2010, 1, 1, tzinfo=UTC)
    )


# ----------------------------------------------------------------------- list


def test_list_emits_the_files_whose_period_overlaps_the_window(offline, product):
    window = TimeWindow(datetime(2015, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
    units = offline.list(window, product)

    assert {u.meta["token"] for u in units} == {
        "2010-2019",
        "previous-2020-2024",
        "latest-2025-2026",
    }
    assert all(u.url.endswith(".csv.gz") for u in units)
    assert all("/BASE/HOR/" in u.url for u in units)


def test_unit_keys_are_stable_across_runs(offline, product):
    window = TimeWindow(datetime(2015, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
    assert {u.key for u in offline.list(window, product)} == {
        u.key for u in offline.list(window, product)
    }


def test_the_unit_key_does_not_move_when_the_file_is_rewritten(product):
    """`latest-*` changes content under an unchanged name.

    A key derived from size or ETag would make every rewrite look like a new
    unit and leave the old one pending forever; the flag that says the content
    moved is `meta["volatile"]`, not the key.
    """
    files, _ = parse_listing(LISTING_XML)
    latest = next(f for f in files if f.volatile)
    rewritten = LISTING_XML.replace("<Size>5926627</Size>", "<Size>6100000</Size>").replace(
        "2026-10-31T06:44:18.000Z", "2026-11-01T06:40:01.000Z"
    )
    after = next(f for f in parse_listing(rewritten)[0] if f.volatile)

    assert after.size != latest.size
    assert unit_key(product, after.stem) == unit_key(product, latest.stem)
    assert unit_key(product, latest.stem) == "meteofrance/hourly_all/H_59_latest-2025-2026"


def test_only_the_rewritten_file_is_flagged_volatile(offline, product):
    window = TimeWindow(datetime(2015, 1, 1, tzinfo=UTC), datetime(2026, 11, 1, tzinfo=UTC))
    volatile = {u.meta["token"] for u in offline.list(window, product) if u.meta["volatile"]}
    assert volatile == {"latest-2025-2026"}


def test_listing_failures_leave_the_other_departements_alone(monkeypatch, src, product):
    def fake_get(url: str) -> str:
        if "H_59_" in url:
            return LISTING_XML
        raise RuntimeError("502 from the bucket")

    monkeypatch.setattr(src, "_get_text", fake_get)
    window = TimeWindow(datetime(2025, 1, 1, tzinfo=UTC), datetime(2026, 1, 1, tzinfo=UTC))
    units = src.list(window, product)
    assert units and {u.meta["dep"] for u in units} == {"59"}


def test_the_departement_list_is_a_listing_filter_only(product):
    """A département narrows what is downloaded; it never decides a station.

    Dijon is in the configured list and 200 km outside the footprint, which is
    exactly why `parse` clips per station.
    """
    assert "21" in product.extra["departements"]
    assert set(product.extra["departements"]) == set(DEPARTEMENTS)


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
        return RawFile(path=args["dest"], sha256="a" * 64, n_bytes=11, unit_key=args["unit_key"])

    _install_http(monkeypatch, handler)
    unit = Unit(
        key="meteofrance/hourly_all/H_59_2010-2019",
        url="https://example.invalid/BASE/HOR/H_59_2010-2019.csv.gz",
        meta={"dep": "59", "token": "2010-2019", "filename": "H_59_2010-2019.csv.gz"},
    )

    raw = src.fetch(unit, tmp_path)
    assert raw.n_bytes == 11
    assert seen["url"] == unit.url
    assert seen["unit_key"] == unit.key
    assert Path(seen["dest"]) == tmp_path / "59" / "H_59_2010-2019.csv.gz"
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
    src = MeteoFranceStations()
    src.fetch(Unit(key="k", url="https://example.invalid/x.csv.gz", meta={}), tmp_path)
    assert isinstance(seen["session"], requests.Session)


def test_fetch_skips_a_404_rather_than_failing_the_job(monkeypatch, src, tmp_path):
    def handler(args):
        raise not_found(args["url"], "not found", 404)

    not_found = _install_http(monkeypatch, handler)
    unit = Unit(
        key="meteofrance/hourly_all/H_99_latest-2025-2026",
        url="https://example.invalid/BASE/HOR/H_99_latest-2025-2026.csv.gz",
        meta={"dep": "99", "filename": "H_99_latest-2025-2026.csv.gz"},
    )
    # A département with no file for a period is routine, not a failure.
    assert src.fetch(unit, tmp_path) is None


def test_a_volatile_file_is_re_downloaded_rather_than_skipped(monkeypatch, src, tmp_path):
    """`fetch_file` is a no-op when the destination exists.

    For `latest-*` that would pin the store to whatever was downloaded first,
    while the file upstream keeps growing under the same name.
    """
    seen = {}

    def handler(args):
        seen["existed"] = Path(args["dest"]).exists()
        seen["overwrite"] = args["overwrite"]
        return RawFile(path=args["dest"], sha256="c" * 64, n_bytes=2, unit_key=args["unit_key"])

    _install_http(monkeypatch, handler)
    dest = tmp_path / "59" / "H_59_latest-2025-2026.csv.gz"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"yesterday's copy")

    meta = {"dep": "59", "filename": "H_59_latest-2025-2026.csv.gz", "volatile": True}
    src.fetch(Unit(key="k", url="https://example.invalid/x.csv.gz", meta=meta), tmp_path)
    # Downloaded again over the old copy, which stays until the new one is
    # complete: this used to delete it first, losing it if the download failed.
    assert seen == {"existed": True, "overwrite": True}

    closed = tmp_path / "59" / "H_59_2010-2019.csv.gz"
    closed.write_bytes(b"a closed decade, already correct")
    src.fetch(
        Unit(
            key="k2",
            url="https://example.invalid/y.csv.gz",
            meta={"dep": "59", "filename": "H_59_2010-2019.csv.gz", "volatile": False},
        ),
        tmp_path,
    )
    assert seen == {"existed": True, "overwrite": False}


# ------------------------------------------------------------------------ parse


@pytest.fixture
def unpinned(product) -> ProductConfig:
    """The product with no fingerprint pinned, for the column-subset fixture."""
    return product.model_copy(
        update={"extra": {**product.extra, "schema_fingerprints": {}}}
    )


def test_parse_returns_time_station_and_the_columns_the_file_has(tmp_path, src, product):
    df = src.parse(_csv(tmp_path), product)

    assert df.columns[:2] == ["time", "station_id"]
    assert df.schema["time"] == pl.Datetime("us", "UTC")
    assert df.schema["station_id"] == pl.String
    # Only the configured columns the file actually carries.
    assert "air_temperature" in df.columns
    assert "soil_temperature_50cm" not in df.columns
    assert df["time"][0] == datetime(2025, 1, 1, 0, 0, tzinfo=UTC)


def test_parse_clips_to_the_grid_footprint(tmp_path, src, product):
    df = src.parse(_csv(tmp_path), product)
    assert set(df["station_id"].to_list()) == {"59343001", "59606004"}
    # Dijon is a real station in a configured département, 200 km outside.
    assert "21473001" not in df["station_id"].to_list()


def test_an_empty_field_becomes_null_before_anything_is_scaled(tmp_path, src, product):
    """Lille reports no radiation and no sunshine in January.

    Treating the empty field as a zero and then scaling it would write a
    confident 0 W/m2 where the right answer is "this station has no
    pyranometer".
    """
    df = src.parse(_csv(tmp_path), product)
    lille = df.filter(pl.col("station_id") == "59343001").sort("time")

    assert lille["surface_downwelling_shortwave_flux"][0] is None
    assert lille["sunshine_duration"][0] is None
    assert lille["air_temperature"][0] == pytest.approx(7.8, abs=1e-4)


def test_scales_bring_the_columns_into_canonical_units(tmp_path, src, product):
    df = src.parse(_csv(tmp_path), product).sort(["station_id", "time"])
    valenciennes = df.filter(pl.col("station_id") == "59606004").sort("time")

    # 268 J/cm2 over an hour = 268 * 1e4 / 3600 W/m2.
    assert valenciennes["surface_downwelling_shortwave_flux"][0] == pytest.approx(
        268 * 1e4 / 3600, rel=1e-5
    )
    # 40 minutes of sunshine -> seconds.
    assert valenciennes["sunshine_duration"][0] == pytest.approx(2400.0)
    # 8 oktas of cloud -> percent.
    lille = df.filter(pl.col("station_id") == "59343001").sort("time")
    assert lille["cloud_area_fraction"][0] == pytest.approx(100.0)


def test_the_valid_range_is_applied_in_canonical_units(tmp_path, src, product):
    """Two values that are ordinary raw and impossible once converted.

    `N=9` is the sky-obscured code, not a ninth okta: 112.5%. `INS=120` is two
    hours of sunshine inside one hour: 7200 s. Checking the range before the
    scale lets both through, which is exactly what the ordering prevents.
    """
    df = src.parse(_csv(tmp_path), product)
    lille = df.filter(pl.col("station_id") == "59343001").sort("time")
    assert lille["cloud_area_fraction"][2] is None

    valenciennes = df.filter(pl.col("station_id") == "59606004").sort("time")
    assert valenciennes["sunshine_duration"][2] is None


def test_parse_harvests_the_station_metadata_it_reads_anyway(tmp_path, src, product):
    """Coordinates are in every row, so `stations()` is a by-product of `parse`.

    `pack` parses everything and then asks for metadata; without this the store
    would have observations and no coordinates, and every downstream station
    matrix would silently drop its columns.
    """
    # Cold, with nothing parsed and no raw_root: empty rather than a download.
    assert src.stations(product).is_empty()
    src.parse(_csv(tmp_path), product)
    table = src.stations(product)

    assert {"station_id", "lat", "lon", "height_m", "name"} <= set(table.columns)
    assert table["station_id"].to_list() == ["59343001", "59606004"]
    row = table.row(0, named=True)
    assert row["lat"] == pytest.approx(50.57)
    assert row["lon"] == pytest.approx(3.0975)
    assert row["height_m"] == pytest.approx(47.0)
    assert row["name"] == "LILLE-LESQUIN"


def test_stations_clips_even_though_parse_already_did(tmp_path, src, product):
    """The registry holds everything the files described, clipping is at read.

    That way `grid_id: null` still yields the whole French network from the
    same parsed data.
    """
    src.parse(_csv(tmp_path), product)
    assert "21473001" not in src.stations(product)["station_id"].to_list()
    everything = src.stations(product.model_copy(update={"grid_id": None}))
    assert "21473001" in everything["station_id"].to_list()


def test_stations_falls_back_to_the_raw_files_on_disk(tmp_path, source_cfg, product):
    raw = _csv(tmp_path / "59")
    src = MeteoFranceStations(source_cfg, raw_root=tmp_path)
    assert raw.path.exists()
    table = src.stations(product)
    assert table["station_id"].to_list() == ["59343001", "59606004"]


def test_a_column_absent_from_an_old_period_is_simply_absent(tmp_path, src, unpinned):
    """A 1950s file has no 3-second gust column. That is not drift."""
    raw = _csv(
        tmp_path,
        "H_59_1950-1959.csv.gz",
        rows=["59343001;LILLE-LESQUIN;50.570000;3.097500;47;1955010100;0.0;1"],
    )
    Path(raw.path).write_bytes(
        gzip.compress(
            b"NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;AAAAMMJJHH;RR1;QRR1\n"
            b"59343001;LILLE-LESQUIN;50.570000;3.097500;47;1955010100;0.0;1\n"
        )
    )
    df = src.parse(raw, unpinned)
    assert df.columns == ["time", "station_id", "precipitation_amount"]


def test_parse_refuses_a_column_set_it_was_not_pinned_against(tmp_path, src, product):
    cfg = product.model_copy(
        update={
            "extra": {
                **product.extra,
                "schema_fingerprints": {"2001-2009": "0000000000000000"},
            }
        }
    )
    with pytest.raises(SchemaDriftError, match="column set changed"):
        src.parse(_csv(tmp_path), cfg)


def test_a_period_with_no_pin_is_not_checked(tmp_path, src, product):
    # The fixture's token, 2001-2009, is deliberately not in the real config.
    assert "2001-2009" not in product.extra["schema_fingerprints"]
    assert not src.parse(_csv(tmp_path), product).is_empty()


def test_parse_raises_when_the_date_column_is_missing(tmp_path, src, unpinned):
    raw = _csv(tmp_path, "H_59_2001-2009.csv.gz")
    Path(raw.path).write_bytes(gzip.compress(b"NUM_POSTE;NOM_USUEL;LAT;LON;ALTI;T\n"))
    with pytest.raises(SchemaDriftError, match="AAAAMMJJHH"):
        src.parse(raw, unpinned)


# ------------------------------------------------------------------ fingerprint


def test_the_fingerprint_is_stable_and_order_independent():
    columns = CSV_HEADER.split(";")
    assert schema_fingerprint(columns) == schema_fingerprint(columns)
    assert schema_fingerprint(columns) == schema_fingerprint(reversed(columns))
    assert schema_fingerprint(columns) == schema_fingerprint([f" {c} " for c in columns])
    assert schema_fingerprint(columns) != schema_fingerprint([*columns, "NEWSENSOR"])


def test_product_fingerprint_rolls_up_a_mapping():
    assert product_fingerprint({"a": "1", "b": "2"}) == product_fingerprint({"b": "2", "a": "1"})
    assert product_fingerprint({"a": "1"}) != product_fingerprint({"a": "2"})


def test_scan_groups_files_by_period_token(tmp_path, src):
    old = _csv(tmp_path, "H_59_2001-2009.csv.gz")
    new = _csv(tmp_path, "H_62_2001-2009.csv.gz")
    report = src.scan([old, new])
    assert report["n_variants"] == 1
    assert report["schema_fingerprints"]["2001-2009"] == schema_fingerprint(
        CSV_HEADER.split(";")
    )
    assert report["errors"] == []


def test_scan_reports_a_bad_file_instead_of_crashing(tmp_path, src):
    path = tmp_path / "H_59_2001-2009.csv.gz"
    path.write_bytes(b"not gzip at all")
    raw = RawFile(path=path, sha256="0" * 64, n_bytes=15, unit_key="k")
    report = src.scan([raw])
    assert len(report["errors"]) == 1


# --------------------------------------------------- against the local archive


@pytest.mark.skipif(not LOCAL_ARCHIVE.exists(), reason="local Météo-France archive not present")
def test_the_pinned_fingerprints_match_the_real_files(product):
    """Pinned on 2026-10-31 against BASE/HOR.

    If this fails, Météo-France changed the header and the variable map needs
    looking at before the pin is moved.
    """
    from meteopack.sources.meteofrance import _header_columns, _token_of

    pinned = product.extra["schema_fingerprints"]
    checked = 0
    for path in sorted(LOCAL_ARCHIVE.glob("H_*.csv.gz")):
        token = _token_of(path.name)
        if token not in pinned:
            continue
        assert schema_fingerprint(_header_columns(path)) == pinned[token], path.name
        checked += 1
    assert checked > 0


@pytest.mark.skipif(not LOCAL_ARCHIVE.exists(), reason="local Météo-France archive not present")
def test_a_real_departement_file_parses_and_clips(src, product):
    path = LOCAL_ARCHIVE / "H_59_latest-2025-2026.csv.gz"
    if not path.exists():
        pytest.skip("H_59 latest not in the local archive")
    raw = RawFile(
        path=path,
        sha256="0" * 64,
        n_bytes=path.stat().st_size,
        unit_key="meteofrance/hourly_all/H_59_latest-2025-2026",
    )
    df = src.parse(raw, product)

    assert df.height > 10_000
    assert df["time"].min() >= datetime(2025, 1, 1, tzinfo=UTC)
    assert df["air_temperature"].drop_nulls().max() < 60.0
    table = src.stations(product)
    assert not table.is_empty()
    assert set(table["station_id"].to_list()) >= set(df["station_id"].unique().to_list())


@pytest.mark.skipif(not LOCAL_ARCHIVE.exists(), reason="local Météo-France archive not present")
def test_a_wholly_out_of_domain_departement_yields_no_rows(src, product):
    """Côte-d'Or is listed and contributes nothing, which is the point.

    The listing filter is coarse on purpose; the domain test is per station.
    """
    path = LOCAL_ARCHIVE / "H_21_latest-2025-2026.csv.gz"
    if not path.exists():
        pytest.skip("H_21 latest not in the local archive")
    raw = RawFile(
        path=path,
        sha256="0" * 64,
        n_bytes=path.stat().st_size,
        unit_key="meteofrance/hourly_all/H_21_latest-2025-2026",
    )
    assert src.parse(raw, product).is_empty()


def test_the_cold_start_download_only_happens_when_it_is_asked_for(
    monkeypatch, tmp_path, source_cfg, product
):
    """Deriving stations from the bucket costs ~150 MB. It must be deliberate.

    `pack` never needs it — it parses first — so the default is the cheap path
    and a caller that genuinely has nothing opts in.
    """
    calls = []

    def fake_fetch_file(url, dest, **kwargs):
        calls.append(url)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_bytes(
            gzip.compress(("\n".join([CSV_HEADER, CSV_ROWS[0]]) + "\n").encode("utf-8"))
        )
        return RawFile(path=Path(dest), sha256="0" * 64, n_bytes=1, unit_key=kwargs["unit_key"])

    from meteopack.core import http

    monkeypatch.setattr(http, "fetch_file", fake_fetch_file)
    src = MeteoFranceStations(source_cfg)
    monkeypatch.setattr(
        src, "_get_text", lambda url: LISTING_XML if "H_59_" in url else "<ListBucketResult/>"
    )
    cfg = product.model_copy(update={"extra": {**product.extra, "departements": ["59", "62"]}})

    assert src.stations(cfg).is_empty()
    assert calls == []

    table = src.stations(cfg, allow_download=True)
    assert [u.rsplit("/", 1)[-1] for u in calls] == ["H_59_latest-2025-2026.csv.gz"]
    assert table["station_id"].to_list() == ["59343001"]
