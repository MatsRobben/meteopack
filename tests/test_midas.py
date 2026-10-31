"""Tests for the Met Office MIDAS Open source, served by CEDA.

Nothing here touches the network and nothing here needs a CEDA token. The
BADC-CSV fixture is a truncated but otherwise verbatim copy of a real
station-year (Topcliffe, src_id 16596, dv-202607): its header block, its full
104-column table row and its first three data rows, plus four rows marked
synthetic below. The login-page fixture is a verbatim excerpt of what
`dap.ceda.ac.uk` actually returned for an expired token on 2026-10-31. Two
optional suites are guarded by `skipif`: one that reads the local archive if it
happens to be mounted, and one that makes live requests only when a token file
already exists and `METEOPACK_LIVE=1` is set.

The four facts most of this file exists to protect are the ones that are
silent rather than loud: an expired token arrives as HTTP 200 with an HTML
login page, knots are not metres per second and decametres are not metres, a
repeated `ob_time` must resolve to the highest `version_num` rather than to
whichever row came last, and the token must never reach the repo or a log.
"""
from __future__ import annotations

import inspect
import os
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
import yaml

from meteopack.core import http
from meteopack.core.config import ProductConfig, SourceConfig, StoreConfig, Variable
from meteopack.sources.base import RawFile, Source, TimeWindow, Unit
from meteopack.sources.midas import (
    ARCHIVE_ROOT,
    DECAMETRE,
    DEFAULT_VERSION,
    DOWNLOAD_BASE,
    KNOT,
    TOKEN_ENV,
    CedaAuthError,
    MidasOpen,
    SchemaDriftError,
    auth_error,
    badc_columns,
    check_payload,
    declared_units,
    parse_station_metadata,
    read_token,
    schema_fingerprint,
    split_badc,
    station_id,
    token_file,
    unit_key,
    variables_for,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_YAML = REPO_ROOT / "configs" / "sources" / "midas.yaml"
PRODUCT_YAML = REPO_ROOT / "configs" / "products" / "midas" / "hourly_weather.yaml"

LOCAL_ARCHIVE = Path("/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/midas/hourly_weather")

DATASET_PATH = f"{ARCHIVE_ROOT}/uk-hourly-weather-obs"
VERSION_PATH = f"{DATASET_PATH}/dataset-version-202607"


# --------------------------------------------------------------------- fixtures

# Verbatim from dv-202607, north-yorkshire/16596_topcliffe, year 2023: the
# header lines this reader actually consults, the real `data` marker, the real
# 104-column row and the first three data rows. Four rows are SYNTHETIC and
# marked where they appear:
#   03:00 twice, version_num 1 then 0  - the version dedupe (file order lies)
#   04:00 air_temperature 999.9, wind_speed NA - range mask and missing flag
#   04:30                              - off the hourly cadence grid
BADC = """Conventions,G,BADC-CSV,1
title,G,uk-hourly-weather-obs
collection_version_number,G,dataset-version-202607
observation_station,G,topcliffe
midas_station_id,G,16596
location,G,54.205,-1.39
height,G,25,m
midas_qc_version_number,G,1
missing_value,G,NA
long_name,ob_time,Date and time of observation,s
long_name,version_num,midas qc version number,1
long_name,src_id,unique source identifier or station site number,1
long_name,wind_direction,wind direction,deg true
long_name,wind_speed,wind speed,knots
long_name,prst_wx_id,present weather code (see documentation),1
long_name,cld_ttl_amt_id,Total cloud amount code,Oktas
long_name,cld_base_ht,cloud base height code,decametres
long_name,visibility,visibility,decametres
long_name,msl_pressure,mean sea level air pressure,hPa
long_name,air_temperature,Air temperature,degC
long_name,dewpoint,Dewpoint temperature,degC
long_name,wetb_temp,Wetbulb temperature,degC
long_name,rltv_hum,Calculated relative humidity,%
long_name,stn_pres,Station air pressure,hPa
long_name,q10mnt_mxgst_spd,10 minute maximum gust speed,knots
long_name,snow_depth,Snow depth,cm
long_name,drv_hr_sun_dur,derived hour sunshine,hours
data
ob_time,id,id_type,met_domain_name,version_num,src_id,rec_st_ind,wind_speed_unit_id,src_opr_type,wind_direction,wind_speed,prst_wx_id,past_wx_id_1,past_wx_id_2,cld_ttl_amt_id,low_cld_type_id,med_cld_type_id,hi_cld_type_id,cld_base_amt_id,cld_base_ht,visibility,msl_pressure,cld_amt_id_1,cloud_type_id_1,cld_base_ht_id_1,cld_amt_id_2,cloud_type_id_2,cld_base_ht_id_2,cld_amt_id_3,cloud_type_id_3,cld_base_ht_id_3,cld_amt_id_4,cloud_type_id_4,cld_base_ht_id_4,vert_vsby,air_temperature,dewpoint,wetb_temp,rltv_hum,stn_pres,alt_pres,ground_state_id,q10mnt_mxgst_spd,cavok_flag,cs_hr_sun_dur,wmo_hr_sun_dur,snow_depth,wind_direction_q,wind_speed_q,prst_wx_id_q,past_wx_id_1_q,past_wx_id_2_q,cld_ttl_amt_id_q,low_cld_type_id_q,med_cld_type_id_q,hi_cld_type_id_q,cld_base_amt_id_q,cld_base_ht_q,visibility_q,msl_pressure_q,air_temperature_q,dewpoint_q,wetb_temp_q,ground_state_id_q,cld_amt_id_1_q,cloud_type_id_1_q,cld_base_ht_id_1_q,cld_amt_id_2_q,cloud_type_id_2_q,cld_base_ht_id_2_q,cld_amt_id_3_q,cloud_type_id_3_q,cld_base_ht_id_3_q,cld_amt_id_4_q,cloud_type_id_4_q,cld_base_ht_id_4_q,vert_vsby_q,stn_pres_q,alt_pres_q,q10mnt_mxgst_spd_q,cs_hr_sun_dur_q,wmo_hr_sun_dur_q,snow_depth_q,wind_direction_j,wind_speed_j,prst_wx_id_j,past_wx_id_1_j,past_wx_id_2_j,cld_amt_id_j,cld_ht_j,visibility_j,msl_pressure_j,air_temperature_j,dewpoint_j,wetb_temp_j,rltv_hum_j,vert_vsby_j,stn_pres_j,alt_pres_j,q10mnt_mxgst_spd_j,meto_stmp_time,midas_stmp_etime,drv_hr_sun_dur,drv_hr_sun_dur_q
2023-01-01 00:00:00,03265,WMO,SYNOP,1,16596,1011,4,7,130.0,7.0,61,NA,NA,8,NA,NA,NA,NA,150.0,1700.0,996.2,8,NA,150.0,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,7.0,6.6,6.8,97.0,993.1,NA,NA,NA,NA,NA,NA,NA,6,6,6,NA,NA,6,NA,NA,NA,NA,6,6,6,6,6,6,NA,6,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2023-01-01 00:06:06,NA,NA,NA
2023-01-01 01:00:00,03265,WMO,SYNOP,1,16596,1011,4,7,140.0,7.0,61,NA,NA,8,NA,NA,NA,NA,100.0,3000.0,995.6,5,NA,93.0,6,NA,150.0,NA,NA,NA,NA,NA,NA,NA,7.3,6.9,7.1,97.0,992.6,NA,NA,NA,NA,NA,NA,NA,6,6,6,NA,NA,6,NA,NA,NA,NA,6,6,6,6,6,6,NA,6,NA,6,6,NA,6,NA,NA,NA,NA,NA,NA,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2023-01-01 01:04:07,NA,NA,NA
2023-01-01 02:00:00,03265,WMO,SYNOP,1,16596,1011,4,7,150.0,10.0,23,NA,NA,8,NA,NA,NA,NA,60.0,4000.0,995.8,8,NA,63.0,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,7.4,6.7,7.1,95.4,992.8,NA,NA,NA,NA,NA,NA,NA,6,6,6,NA,NA,6,NA,NA,NA,NA,6,6,6,6,6,6,NA,6,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2023-01-01 02:04:12,NA,NA,NA
2023-01-01 03:00:00,03265,WMO,SYNOP,1,16596,1011,4,7,150.0,10.0,23,NA,NA,8,NA,NA,NA,NA,60.0,4000.0,995.8,8,NA,63.0,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3.3,6.7,7.1,95.4,992.8,NA,NA,NA,NA,NA,NA,NA,6,6,6,NA,NA,6,NA,NA,NA,NA,6,6,6,6,6,6,NA,6,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2023-01-01 02:04:12,NA,NA,NA
2023-01-01 03:00:00,03265,WMO,SYNOP,0,16596,1011,4,7,150.0,10.0,23,NA,NA,8,NA,NA,NA,NA,60.0,4000.0,995.8,8,NA,63.0,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,1.1,6.7,7.1,95.4,992.8,NA,NA,NA,NA,NA,NA,NA,6,6,6,NA,NA,6,NA,NA,NA,NA,6,6,6,6,6,6,NA,6,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2023-01-01 02:04:12,NA,NA,NA
2023-01-01 04:00:00,03265,WMO,SYNOP,1,16596,1011,4,7,150.0,NA,23,NA,NA,8,NA,NA,NA,NA,60.0,4000.0,995.8,8,NA,63.0,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,999.9,6.7,7.1,95.4,992.8,NA,NA,NA,NA,NA,NA,NA,6,6,6,NA,NA,6,NA,NA,NA,NA,6,6,6,6,6,6,NA,6,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2023-01-01 02:04:12,NA,NA,NA
2023-01-01 04:30:00,03265,WMO,SYNOP,1,16596,1011,4,7,150.0,10.0,23,NA,NA,8,NA,NA,NA,NA,60.0,4000.0,995.8,8,NA,63.0,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,5.5,6.7,7.1,95.4,992.8,NA,NA,NA,NA,NA,NA,NA,6,6,6,NA,NA,6,NA,NA,NA,NA,6,6,6,6,6,6,NA,6,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,6,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2023-01-01 02:04:12,NA,NA,NA
end data
"""

# Reconstructed `..._station-metadata.csv`: the BADC frame is real, and the
# values of the first four rows are the real coordinates, elevations and
# reporting periods of those four stations. The last two rows are synthetic:
# one gauge outside the product bbox, and one with no coordinates at all.
METADATA = """Conventions,G,BADC-CSV,1
title,G,uk-hourly-weather-obs station metadata
collection_version_number,G,dataset-version-202607
missing_value,G,NA
long_name,src_id,unique source identifier or station site number,1
long_name,station_latitude,station latitude,degrees
long_name,station_longitude,station longitude,degrees
long_name,station_elevation,station elevation,m
data
src_id,station_file_name,station_name,historic_county_name,station_latitude,station_longitude,station_elevation,first_year,last_year
160,craibstone,CRAIBSTONE,aberdeenshire,57.187,-2.215,102.0,1972,2021
161,dyce,DYCE,aberdeenshire,57.205,-2.205,65.0,1957,2025
347,carlton-in-cleveland,CARLTON-IN-CLEVELAND,cleveland,54.428,-1.218,103.0,1991,2025
16596,topcliffe,TOPCLIFFE,north-yorkshire,54.205,-1.39,25.0,1997,2025
99999,roma-ciampino,ROMA CIAMPINO (synthetic; outside the product bbox),lazio,41.9028,12.4964,129.0,1950,2025
88888,no-coordinates,BROKEN ROW (synthetic),nowhere,NA,NA,NA,1990,1990
end data
"""

# Keyless `?json` catalogue listings, trimmed to the keys this reader uses.
LISTINGS: dict[str, dict] = {
    DATASET_PATH: {
        "items": [
            {"name": "dataset-version-202507", "type": "dir"},
            {"name": "dataset-version-202607", "type": "dir"},
        ]
    },
    VERSION_PATH: {
        "items": [
            {"name": "00README_catalogue_and_licence.txt", "type": "file"},
            {"name": "aberdeenshire", "type": "dir"},
            {"name": "cleveland", "type": "dir"},
            {"name": "north-yorkshire", "type": "dir"},
        ]
    },
    f"{VERSION_PATH}/aberdeenshire": {
        "items": [
            {"name": "00160_craibstone", "type": "dir"},
            {"name": "00161_dyce", "type": "dir"},
        ]
    },
    f"{VERSION_PATH}/cleveland": {
        "items": [{"name": "00347_carlton-in-cleveland", "type": "dir"}]
    },
    f"{VERSION_PATH}/north-yorkshire": {
        "items": [{"name": "16596_topcliffe", "type": "dir"}]
    },
}

# What dap.ceda.ac.uk really answers for an expired or absent token: a 302 to
# Keycloak that `requests` follows, ending in an HTTP 200 whose body is this.
# Excerpt verbatim from 2026-10-31.
LOGIN_PAGE = """<!DOCTYPE html><html><head><script>


const kcContext = {
    "social": {
        "baseURI": {
            "path": "\\/",
            "query": "client_id=data-service-ceda-ac-uk&tab_id=LifJRAoLWiE"
        }
    }
};
</script></head><body>Sign in to your account</body></html>
"""


# ----------------------------------------------------------------- test doubles


class FakeSession:
    """A stand-in for `requests.Session` that is a session in the ways that matter.

    It used to be a bare `object()`. That passed only because the source
    returned an injected session untouched — the very bug that let `midas` run
    without its bearer token. A double looser than the real thing hides exactly
    the defect it is standing in for, so this one carries `headers`.
    """

    def __init__(self) -> None:
        self.headers: dict[str, str] = {}


class FakeMidas(MidasOpen):
    """MidasOpen with its two remote reads replaced by literal payloads."""

    def __init__(self, *, metadata: str = METADATA, listings=None, **kwargs):
        super().__init__(**kwargs)
        self._metadata_text = metadata
        self._listings = LISTINGS if listings is None else listings

    def _get_json(self, url: str):
        path = url[len(self.catalogue_url) :]
        return self._listings.get(path, {"items": []})

    def _get_text(self, url: str) -> str:
        return self._metadata_text


def product(**overrides) -> ProductConfig:
    base = {
        "source": "midas",
        "product": "hourly_weather",
        "cadence_seconds": 3600,
        "grid_id": None,
        "store": StoreConfig(kind="points"),
        "variables": [
            Variable(
                name="air_temperature",
                source_key="air_temperature",
                units="degC",
                dtype="float32",
                valid_min=-60.0,
                valid_max=60.0,
            ),
            Variable(
                name="wind_speed",
                source_key="wind_speed",
                units="m s-1",
                dtype="float32",
                scale=KNOT,
                valid_min=0.0,
                valid_max=120.0,
            ),
            Variable(
                name="visibility_in_air",
                source_key="visibility",
                units="m",
                dtype="float32",
                scale=DECAMETRE,
                valid_min=0.0,
                valid_max=150000.0,
            ),
        ],
        "extra": {"bbox": [-11.0, 49.5, 2.5, 61.5], "dataset_version": "202607"},
    }
    base.update(overrides)
    return ProductConfig(**base)


def window(start_year: int, end_year: int) -> TimeWindow:
    return TimeWindow(
        datetime(start_year, 1, 1, tzinfo=UTC), datetime(end_year, 1, 1, tzinfo=UTC)
    )


def write(tmp_path: Path, text: str, name: str = "2023.csv", key: str | None = None) -> RawFile:
    path = tmp_path / name
    path.write_text(text)
    return RawFile(
        path=path,
        sha256="0" * 64,
        n_bytes=len(text),
        unit_key=key or "midas/hourly_weather/dv-202607/16596/2023",
    )


def at(frame: pl.DataFrame, hour: int) -> pl.DataFrame:
    return frame.filter(pl.col("time") == datetime(2023, 1, 1, hour, tzinfo=UTC))


# ------------------------------------------------------------------ credentials


def test_token_file_prefers_the_environment_then_config_then_the_default(monkeypatch):
    """The path is configuration; the token itself never is."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    assert token_file(None) == Path("~/.ceda_token").expanduser()
    cfg = SourceConfig(name="midas", extra={"token_file": "/somewhere/else"})
    assert token_file(cfg) == Path("/somewhere/else")
    monkeypatch.setenv(TOKEN_ENV, "/from/the/environment")
    assert token_file(cfg) == Path("/from/the/environment")


def test_a_missing_token_file_says_which_file_and_what_to_do(tmp_path):
    missing = tmp_path / "no-such-token"
    with pytest.raises(CedaAuthError) as excinfo:
        read_token(missing)
    message = str(excinfo.value)
    assert str(missing) in message
    assert "expire" in message
    assert "services.ceda.ac.uk" in message


def test_an_empty_token_file_is_rejected_by_name(tmp_path):
    empty = tmp_path / ".ceda_token"
    empty.write_text("   \n")
    with pytest.raises(CedaAuthError, match=str(empty)):
        read_token(empty)


def test_reading_the_token_never_logs_it(tmp_path, caplog):
    path = tmp_path / ".ceda_token"
    path.write_text("s3cret-bearer-value\n")
    path.chmod(0o600)
    with caplog.at_level("DEBUG"):
        assert read_token(path) == "s3cret-bearer-value"
    assert "s3cret-bearer-value" not in caplog.text


def test_a_token_file_readable_by_others_is_flagged(tmp_path, caplog):
    path = tmp_path / ".ceda_token"
    path.write_text("token\n")
    path.chmod(0o644)
    with caplog.at_level("WARNING"):
        read_token(path)
    assert "chmod 600" in caplog.text


def test_stations_reports_a_missing_token_rather_than_failing_400_files_later(tmp_path, monkeypatch):
    """The first call that needs a credential is the one that complains."""
    monkeypatch.delenv(TOKEN_ENV, raising=False)
    cfg_source = SourceConfig(
        name="midas", base_url=DOWNLOAD_BASE, extra={"token_file": str(tmp_path / "absent")}
    )
    src = MidasOpen(cfg_source)
    with pytest.raises(CedaAuthError, match="absent"):
        src.stations(product())


def test_no_token_value_is_committed_anywhere_in_the_shipped_config():
    """The YAML may name the file; it must never carry the secret."""
    text = SOURCE_YAML.read_text()
    assert "~/.ceda_token" in text
    settings = yaml.safe_load(text)
    assert settings["extra"]["token_file"] == "~/.ceda_token"
    assert "token" not in settings["extra"] and "token" not in settings
    # "Bearer" may be explained in a comment; it must never be a value.
    assert all(line.lstrip().startswith("#") for line in text.splitlines() if "Bearer" in line)


# ------------------------------------------------------------------ login pages


def test_an_expired_token_is_caught_even_though_it_arrives_as_a_200(tmp_path):
    """The whole reason `check_payload` exists.

    dap.ceda.ac.uk answers a stale token with a 302 to Keycloak; requests
    follows it and hands back a perfectly successful HTML page. Without this
    check every station-year in the job "downloads" and then fails to parse.
    """
    token = tmp_path / ".ceda_token"
    with pytest.raises(CedaAuthError) as excinfo:
        check_payload(LOGIN_PAGE, where="https://dap.ceda.ac.uk/x.csv", path=token)
    assert str(token) in str(excinfo.value)
    assert "expired or rejected" in str(excinfo.value)


def test_a_plain_401_is_translated_into_the_same_actionable_message(tmp_path):
    """CEDA also answers a stale token with a bare 401.

    Left alone that surfaces as `requests.HTTPError: 401 Client Error`, which
    says nothing about which file on disk needs refreshing. Observed live on
    2026-10-31 alongside the 200-plus-login-page route.
    """

    class Unauthorised(MidasOpen):
        def _get_text(self, url: str) -> str:
            raise auth_error(url, "CEDA answered HTTP 401", self.token_file)

    src = Unauthorised(
        SourceConfig(name="midas", extra={"token_file": str(tmp_path / "stale")})
    )
    with pytest.raises(CedaAuthError) as excinfo:
        src.metadata("202607")
    assert str(tmp_path / "stale") in str(excinfo.value)
    assert "HTTP 401" in str(excinfo.value)


def test_fetch_translates_a_401_from_fetch_file(monkeypatch, tmp_path):
    def unauthorised(url, dest, **kwargs):
        raise http.FetchError(url, "HTTP 401", 401)

    monkeypatch.setattr(http, "fetch_file", unauthorised)
    unit = Unit(
        key="midas/hourly_weather/dv-202607/16596/2023",
        url="https://dap.ceda.ac.uk/x.csv",
        meta={"station_id": "16596", "filename": "2023.csv"},
    )
    with pytest.raises(CedaAuthError, match="expired or rejected"):
        FakeMidas(session=FakeSession()).fetch(unit, tmp_path)


def test_a_payload_that_is_neither_badc_nor_a_login_page_is_schema_drift(tmp_path):
    with pytest.raises(SchemaDriftError, match="not a BADC-CSV"):
        check_payload("src_id,ob_time\n1,2\n", where="x", path=tmp_path / "t")


def test_a_real_badc_payload_passes(tmp_path):
    check_payload(BADC, where="x", path=tmp_path / "t")


def test_fetch_refuses_and_deletes_a_login_page(monkeypatch, tmp_path):
    """A rejected payload must not be left looking like a finished download."""

    def write_login_page(url, dest, **kwargs):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(LOGIN_PAGE)
        return RawFile(dest, "0" * 64, len(LOGIN_PAGE), kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", write_login_page)
    src = FakeMidas(session=FakeSession())
    unit = Unit(
        key="midas/hourly_weather/dv-202607/16596/2023",
        url="https://dap.ceda.ac.uk/x.csv",
        meta={"station_id": "16596", "filename": "2023.csv"},
    )
    with pytest.raises(CedaAuthError):
        src.fetch(unit, tmp_path)
    assert not (tmp_path / "16596" / "2023.csv").exists()


# -------------------------------------------------------------------- BADC-CSV


def test_split_badc_finds_the_table_between_the_markers():
    header, table = split_badc(BADC)
    assert header[0].startswith("Conventions,G,BADC-CSV")
    assert not any(line.strip().lower() in {"data", "end data"} for line in table)
    assert table[0].startswith("ob_time,")
    assert len(table) == 8          # one column row + seven data rows


def test_split_badc_tolerates_a_truncated_file_with_no_end_marker():
    """A killed download loses `end data`; that is short, not unreadable."""
    truncated = BADC.rsplit("end data", 1)[0]
    _, table = split_badc(truncated)
    assert len(table) == 8


def test_split_badc_refuses_a_file_with_no_data_marker():
    with pytest.raises(SchemaDriftError, match="not a BADC-CSV table"):
        split_badc("Conventions,G,BADC-CSV,1\ntitle,G,nothing\n")


def test_badc_columns_are_the_row_after_the_marker():
    _, table = split_badc(BADC)
    columns = badc_columns(table)
    assert len(columns) == 104
    assert columns[:3] == ["ob_time", "id", "id_type"]
    assert "version_num" in columns and "src_id" in columns


def test_declared_units_come_from_the_files_own_header():
    units = declared_units(split_badc(BADC)[0])
    assert units["wind_speed"] == "knots"
    assert units["q10mnt_mxgst_spd"] == "knots"
    assert units["visibility"] == "decametres"
    assert units["cld_base_ht"] == "decametres"
    assert units["drv_hr_sun_dur"] == "hours"
    assert units["snow_depth"] == "cm"
    assert units["air_temperature"] == "degC"


# ------------------------------------------------------------------ fingerprint


def test_schema_fingerprint_is_order_independent_and_whitespace_insensitive():
    columns = badc_columns(split_badc(BADC)[1])
    assert schema_fingerprint(columns) == schema_fingerprint(reversed(columns))
    assert schema_fingerprint(columns) == schema_fingerprint(f"  {c} " for c in columns)


def test_schema_fingerprint_moves_when_a_column_really_changes():
    columns = badc_columns(split_badc(BADC)[1])
    assert schema_fingerprint(columns) != schema_fingerprint([*columns, "new_column"])
    assert schema_fingerprint(columns) != schema_fingerprint(columns[:-1])


def test_the_fixture_hashes_to_the_fingerprint_the_shipped_config_pins():
    """Pinned from 400 real files, all of which carried this same 104-column table."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert schema_fingerprint(badc_columns(split_badc(BADC)[1])) == cfg.schema_fingerprint


# --------------------------------------------------------------------- stations


def test_station_id_is_the_five_digit_src_id_the_archive_paths_use():
    assert station_id(347) == "00347"
    assert station_id("16596") == "16596"
    assert station_id("00160") == "00160"


def test_parse_station_metadata_reads_coordinates_elevation_and_reporting_years():
    records = {r.station_id: r for r in parse_station_metadata(METADATA)}
    topcliffe = records["16596"]
    assert (topcliffe.lat, topcliffe.lon) == (54.205, -1.39)
    assert topcliffe.height_m == 25.0
    assert (topcliffe.first_year, topcliffe.last_year) == (1997, 2025)
    assert records["00160"].name == "CRAIBSTONE"


def test_a_metadata_row_without_coordinates_is_skipped_not_fatal():
    records = parse_station_metadata(METADATA)
    assert "88888" not in {r.station_id for r in records}
    assert len(records) == 5


def test_stations_are_clipped_to_the_configured_bbox():
    table = FakeMidas().stations(product())
    ids = table["station_id"].to_list()
    assert ids == ["00160", "00161", "00347", "16596"]
    assert "99999" not in ids, "Rome is not in the UK"


def test_stations_are_clipped_to_a_grid_footprint_when_one_is_set():
    """The grid branch, exercised through the real registry rather than a stub."""
    cfg = product(grid_id="example_tiny", extra={"dataset_version": "202607"})
    assert FakeMidas().stations(cfg).is_empty()


def test_stations_carry_the_elevation_midas_actually_publishes():
    table = FakeMidas().stations(product())
    assert {"station_id", "lat", "lon", "height_m", "name"} <= set(table.columns)
    assert table.filter(pl.col("station_id") == "00161")["height_m"][0] == 65.0
    assert table["station_id"].is_duplicated().sum() == 0


# -------------------------------------------------------------------------- list


def test_the_file_url_matches_the_real_archive_layout():
    """Checked against a live catalogue listing of 00160_craibstone on 2026-10-31."""
    units = FakeMidas().list(window(1972, 1973), product())
    craibstone = [u for u in units if u.meta["station_id"] == "00160"]
    assert len(craibstone) == 1
    assert craibstone[0].url == (
        "https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/"
        "dataset-version-202607/aberdeenshire/00160_craibstone/qc-version-1/"
        "midas-open_uk-hourly-weather-obs_dv-202607_aberdeenshire_00160_craibstone"
        "_qcv-1_1972.csv"
    )


def test_the_metadata_url_matches_the_real_archive_layout():
    assert FakeMidas().metadata_url("202607") == (
        "https://dap.ceda.ac.uk/badc/ukmo-midas-open/data/uk-hourly-weather-obs/"
        "dataset-version-202607/midas-open_uk-hourly-weather-obs_dv-202607"
        "_station-metadata.csv"
    )


def test_unit_keys_carry_the_dataset_version_and_are_stable():
    units = FakeMidas().list(window(2020, 2022), product())
    keys = [u.key for u in units]
    assert len(keys) == len(set(keys))
    assert unit_key(product(), "202607", "16596", 2021) in keys
    assert all(k.startswith("midas/hourly_weather/dv-202607/") for k in keys)
    assert keys == [u.key for u in FakeMidas().list(window(2020, 2022), product())]


def test_dataset_version_latest_is_resolved_before_the_key_is_built():
    """Otherwise the key would mean something different on every release day."""
    cfg = product(extra={"bbox": [-11.0, 49.5, 2.5, 61.5], "dataset_version": "latest"})
    src = FakeMidas()
    assert src.dataset_version(cfg) == "202607"
    assert all("dv-202607" in u.key for u in src.list(window(2020, 2021), cfg))


def test_years_are_clipped_to_the_stations_own_reporting_period():
    """Craibstone closed in 2021; asking for 2023 would be a guaranteed 404."""
    units = FakeMidas().list(window(2019, 2024), product())
    craibstone = sorted(int(u.meta["year"]) for u in units if u.meta["station_id"] == "00160")
    assert craibstone == [2019, 2020, 2021]


def test_an_exclusive_window_end_does_not_pull_in_the_next_year():
    units = FakeMidas().list(window(2020, 2023), product())
    assert {u.meta["year"] for u in units} == {"2020", "2021", "2022"}


def test_a_published_release_is_immutable_so_nothing_is_volatile():
    """The version is in the key, so a new release is new work, not a refresh."""
    units = FakeMidas().list(window(2020, 2022), product())
    assert units
    assert not any(u.meta["volatile"] for u in units)


def test_list_honours_src_ids_and_max_stations():
    only = product(
        extra={"bbox": [-11.0, 49.5, 2.5, 61.5], "dataset_version": "202607", "src_ids": [16596]}
    )
    units = FakeMidas().list(window(2020, 2021), only)
    assert {u.meta["station_id"] for u in units} == {"16596"}

    capped = product(
        extra={"bbox": [-11.0, 49.5, 2.5, 61.5], "dataset_version": "202607", "max_stations": 2}
    )
    units = FakeMidas().list(window(2020, 2021), capped)
    assert len({u.meta["station_id"] for u in units}) == 2


def test_list_returns_nothing_when_the_domain_holds_no_station():
    cfg = product(extra={"bbox": [10, 40, 12, 42], "dataset_version": "202607"})
    assert FakeMidas().list(window(2020, 2021), cfg) == []


# ------------------------------------------------------------------------- parse


def test_parse_reads_time_station_and_values(tmp_path):
    frame = FakeMidas().parse(write(tmp_path, BADC), product())
    assert frame.columns == ["time", "station_id", "air_temperature", "wind_speed",
                             "visibility_in_air"]
    assert frame["station_id"].unique().to_list() == ["16596"]
    assert frame["time"].dtype == pl.Datetime("us", "UTC")
    assert at(frame, 0)["air_temperature"][0] == pytest.approx(7.0)


def test_knots_become_metres_per_second(tmp_path):
    """7 knots is 3.60 m/s. Leaving it at 7 would be a 94 % error."""
    frame = FakeMidas().parse(write(tmp_path, BADC), product())
    assert at(frame, 0)["wind_speed"][0] == pytest.approx(7.0 * KNOT, rel=1e-5)
    assert at(frame, 0)["wind_speed"][0] == pytest.approx(3.601, abs=1e-3)


def test_decametres_become_metres(tmp_path):
    """1700 dam is 17 km of visibility, not 1.7 km."""
    frame = FakeMidas().parse(write(tmp_path, BADC), product())
    assert at(frame, 0)["visibility_in_air"][0] == pytest.approx(17000.0)
    assert at(frame, 1)["visibility_in_air"][0] == pytest.approx(30000.0)


def test_cloud_base_height_is_also_decametres(tmp_path):
    cfg = product(
        variables=[
            Variable(
                name="cloud_base_height",
                source_key="cld_base_ht",
                units="m",
                dtype="float32",
                scale=DECAMETRE,
                valid_min=0.0,
                valid_max=30000.0,
            )
        ]
    )
    frame = FakeMidas().parse(write(tmp_path, BADC), cfg)
    assert at(frame, 0)["cloud_base_height"][0] == pytest.approx(1500.0)


def test_repeated_ob_time_keeps_the_highest_version_num(tmp_path):
    """Version 0 is the message as received; version 1 is the current best.

    The fixture lists version 1 *before* version 0 on purpose: keeping whichever
    row came last in the file would pick the un-QCed value here.
    """
    frame = FakeMidas().parse(write(tmp_path, BADC), product())
    three = at(frame, 3)
    assert three.height == 1
    assert three["air_temperature"][0] == pytest.approx(3.3)     # version 1
    assert three["air_temperature"][0] != pytest.approx(1.1)     # version 0


def test_na_becomes_null_before_any_scaling(tmp_path):
    """Order matters: `NA` reaching the knots conversion would become a number."""
    frame = FakeMidas().parse(write(tmp_path, BADC), product())
    assert at(frame, 4)["wind_speed"][0] is None
    # and a real reading in the same column is genuinely converted, so the null
    # above is not just "no scale applied".
    assert at(frame, 2)["wind_speed"][0] == pytest.approx(10.0 * KNOT, rel=1e-5)


def test_the_valid_range_is_applied_in_canonical_units_after_the_scale(tmp_path):
    """A 5 m/s ceiling must keep 7 knots (3.60 m/s) and reject 10 knots (5.14).

    Applied before the scale it would compare knots against a m/s limit and
    throw both away.
    """
    cfg = product(
        variables=[
            Variable(
                name="wind_speed",
                source_key="wind_speed",
                units="m s-1",
                dtype="float32",
                scale=KNOT,
                valid_min=0.0,
                valid_max=5.0,
            )
        ]
    )
    frame = FakeMidas().parse(write(tmp_path, BADC), cfg)
    assert at(frame, 0)["wind_speed"][0] == pytest.approx(3.601, abs=1e-3)
    assert at(frame, 2)["wind_speed"][0] is None


def test_an_out_of_range_temperature_is_masked(tmp_path):
    frame = FakeMidas().parse(write(tmp_path, BADC), product())
    assert at(frame, 4)["air_temperature"][0] is None      # the synthetic 999.9


def test_off_cadence_rows_are_dropped_rather_than_snapped(tmp_path):
    """04:30 is not an hourly observation; a daily station's 09:00 still is."""
    frame = FakeMidas().parse(write(tmp_path, BADC), product())
    assert datetime(2023, 1, 1, 4, 30, tzinfo=UTC) not in frame["time"].to_list()

    kept = FakeMidas().parse(
        write(tmp_path, BADC, name="keep.csv"),
        product(extra={"bbox": [-11.0, 49.5, 2.5, 61.5], "drop_off_cadence": False}),
    )
    assert datetime(2023, 1, 1, 4, 30, tzinfo=UTC) in kept["time"].to_list()


def test_the_station_comes_from_the_src_id_column_before_the_unit_key(tmp_path):
    raw = write(tmp_path, BADC, key="midas/hourly_weather/dv-202607/WRONG/2023")
    assert FakeMidas().parse(raw, product())["station_id"].unique().to_list() == ["16596"]


def test_parse_falls_back_to_the_unit_key_when_src_id_is_gone(tmp_path):
    header, table = split_badc(BADC)
    columns = badc_columns(table)
    drop = columns.index("src_id")
    trimmed = "\n".join(
        [
            *header,
            "data",
            *[",".join(f for i, f in enumerate(row.split(",")) if i != drop) for row in table],
            "end data",
        ]
    )
    raw = write(tmp_path, trimmed, key="midas/hourly_weather/dv-202607/00347/2023")
    assert FakeMidas().parse(raw, product())["station_id"].unique().to_list() == ["00347"]


def test_parse_refuses_to_run_against_a_drifted_schema(tmp_path):
    cfg = product(schema_fingerprint="0" * 16)
    with pytest.raises(SchemaDriftError, match="pinned"):
        FakeMidas().parse(write(tmp_path, BADC), cfg)


def test_parse_refuses_a_file_whose_declared_units_changed(tmp_path):
    """The failure this is here for: upstream switching wind to m/s.

    Nothing about the columns would move, every row would still parse, and
    every wind speed in the store would be wrong by a factor of two.
    """
    tampered = BADC.replace("long_name,wind_speed,wind speed,knots",
                            "long_name,wind_speed,wind speed,m/s")
    cfg = product(
        extra={
            "bbox": [-11.0, 49.5, 2.5, 61.5],
            "expected_units": {"wind_speed": "knots", "visibility": "decametres"},
        }
    )
    with pytest.raises(SchemaDriftError, match="declared units changed"):
        FakeMidas().parse(write(tmp_path, tampered), cfg)
    # unchanged units pass
    assert FakeMidas().parse(write(tmp_path, BADC, name="ok.csv"), cfg).height > 0


def test_parse_reports_a_vanished_column_rather_than_returning_nothing(tmp_path):
    header, table = split_badc(BADC)
    columns = badc_columns(table)
    drop = columns.index("air_temperature")
    trimmed = "\n".join(
        [
            *header,
            "data",
            *[",".join(f for i, f in enumerate(row.split(",")) if i != drop) for row in table],
            "end data",
        ]
    )
    with pytest.raises(SchemaDriftError, match="missing column"):
        FakeMidas().parse(write(tmp_path, trimmed), product())


def test_parse_rejects_a_login_page_found_on_disk(tmp_path):
    with pytest.raises(CedaAuthError):
        FakeMidas().parse(write(tmp_path, LOGIN_PAGE), product())


# -------------------------------------------------------------------------- scan


def test_scan_reports_one_variant_and_the_declared_units(tmp_path):
    files = [write(tmp_path, BADC, name="a.csv"), write(tmp_path, BADC, name="b.csv")]
    report = FakeMidas().scan(files)
    assert report["n_files"] == 2
    assert report["n_variants"] == 1
    assert report["errors"] == []
    assert report["declared_units"]["wind_speed"] == ["knots"]
    assert report["fingerprint"] == report["variants"][0]["fingerprint"]
    assert FakeMidas().scan(files)["fingerprint"] == report["fingerprint"]


def test_scan_shows_a_unit_change_as_two_declared_values(tmp_path):
    tampered = BADC.replace("long_name,wind_speed,wind speed,knots",
                            "long_name,wind_speed,wind speed,m/s")
    files = [write(tmp_path, BADC, name="a.csv"), write(tmp_path, tampered, name="b.csv")]
    report = FakeMidas().scan(files)
    assert report["declared_units"]["wind_speed"] == ["knots", "m/s"]


def test_scan_records_a_bad_file_as_a_finding_rather_than_crashing(tmp_path):
    files = [write(tmp_path, BADC, name="a.csv"), write(tmp_path, "nonsense", name="bad.csv")]
    report = FakeMidas().scan(files)
    assert len(report["errors"]) == 1
    assert "bad.csv" in report["errors"][0]["path"]
    assert report["n_variants"] == 1


# ------------------------------------------------------------------------- fetch


def test_fetch_binds_the_real_fetch_file_signature(monkeypatch, tmp_path):
    """Bound against the live callable, never a hand-copied parameter list.

    A source that omits `unit_key` passes every mocked test and raises
    TypeError on the first real download; that has happened here before. This
    also proves a source built with NO session makes one, and that the session
    it makes carries the bearer token.
    """
    real_sig = inspect.signature(http.fetch_file)
    seen: dict = {}

    def strict_fetch_file(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.update(bound.arguments)
        dest = Path(kwargs["dest"] if "dest" in kwargs else args[1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(BADC)
        return RawFile(dest, "0" * 64, len(BADC), kwargs["unit_key"])

    token = tmp_path / ".ceda_token"
    token.write_text("a-test-token\n")
    token.chmod(0o600)
    monkeypatch.setenv(TOKEN_ENV, str(token))
    monkeypatch.setattr(http, "fetch_file", strict_fetch_file)

    src = FakeMidas()
    unit = src.list(window(2020, 2021), product())[0]
    raw = src.fetch(unit, tmp_path / "raw")

    assert seen["unit_key"] == unit.key
    assert seen["url"] == unit.url
    assert seen["session"].headers["Authorization"] == "Bearer a-test-token"
    assert raw is not None
    # The raw tree mirrors the key: station, then year.
    assert raw.path.name == "2020.csv"
    assert raw.path.parent.name == unit.meta["station_id"]


def test_fetch_skips_a_404_rather_than_failing_the_job(monkeypatch, tmp_path):
    """A station-year that release simply does not carry."""

    def not_found(url, dest, **kwargs):
        raise http.NotFoundError(url, "404 not found", 404)

    monkeypatch.setattr(http, "fetch_file", not_found)
    unit = Unit(
        key="midas/hourly_weather/dv-202607/16596/1801",
        url="https://dap.ceda.ac.uk/x.csv",
        meta={"station_id": "16596", "filename": "1801.csv"},
    )
    assert FakeMidas(session=FakeSession()).fetch(unit, tmp_path) is None


def test_fetch_survives_a_unit_with_no_meta(monkeypatch, tmp_path):
    captured: dict = {}

    def capture(url, dest, **kwargs):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(BADC)
        captured["dest"] = dest
        return RawFile(dest, "0" * 64, len(BADC), kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", capture)
    unit = Unit(key="midas/hourly_weather/dv-202607/00347/2023", url="https://example.invalid/x")
    FakeMidas(session=FakeSession()).fetch(unit, tmp_path)
    assert captured["dest"] == tmp_path / "00347" / "2023.csv"


def test_catalogue_listings_need_no_token(monkeypatch, tmp_path):
    """`list` enumerates the tree before anybody has a credential."""
    monkeypatch.setenv(TOKEN_ENV, str(tmp_path / "definitely-absent"))
    src = FakeMidas()
    assert src.dataset_version(product(extra={"dataset_version": "latest"})) == "202607"
    assert src.county_index("202607")["16596"] == ("north-yorkshire", "16596_topcliffe")


# ------------------------------------------------------------------------ config


def test_the_source_yaml_loads_into_sourceconfig():
    cfg = SourceConfig(**yaml.safe_load(SOURCE_YAML.read_text()))
    assert cfg.name == "midas"
    assert cfg.base_url == DOWNLOAD_BASE
    assert cfg.credentials_env == [TOKEN_ENV]
    assert cfg.extra["token_file"] == "~/.ceda_token"
    assert cfg.extra["catalogue_url"] == "https://data.ceda.ac.uk"


def test_the_product_yaml_loads_into_productconfig():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert cfg.store_id == "midas/hourly_weather/v1"
    assert cfg.store.kind == "points"
    assert cfg.cadence_seconds == 3600
    assert cfg.extra["dataset_version"] == DEFAULT_VERSION
    assert cfg.extra["qc_version"] == 1


def test_every_unit_conversion_in_the_config_uses_the_named_constant():
    """The harmonisation lives in `scale`, never in ad-hoc code."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    by_name = {v.name: v for v in cfg.variables}
    assert by_name["wind_speed"].scale == pytest.approx(KNOT)
    assert by_name["wind_speed_of_gust"].scale == pytest.approx(KNOT)
    assert by_name["visibility_in_air"].scale == pytest.approx(DECAMETRE)
    assert by_name["cloud_base_height"].scale == pytest.approx(DECAMETRE)
    assert by_name["sunshine_duration"].scale == pytest.approx(3600.0)   # hours -> s
    assert by_name["surface_snow_thickness"].scale == pytest.approx(0.01)  # cm -> m
    assert by_name["air_temperature"].scale is None


def test_every_pinned_unit_matches_what_the_real_file_declares():
    """The config's expectation, checked against a real file's own header."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    units = declared_units(split_badc(BADC)[0])
    assert cfg.extra["expected_units"]
    for column, expected in cfg.extra["expected_units"].items():
        assert units[column] == expected, column


def test_every_configured_source_key_is_a_real_midas_column():
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    columns = set(badc_columns(split_badc(BADC)[1]))
    assert cfg.variables
    for column, _ in variables_for(cfg):
        assert column in columns, column


def test_the_shipped_config_parses_the_real_fixture(tmp_path):
    """End to end on the shipped config, not the trimmed test one."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    frame = FakeMidas().parse(write(tmp_path, BADC), cfg)
    assert frame.height == 5                       # 7 rows, one deduped, one off-cadence
    assert frame["air_temperature"].dtype == pl.Float32
    assert frame["present_weather"].dtype == pl.Int16
    assert at(frame, 0)["wind_speed"][0] == pytest.approx(3.601, abs=1e-3)
    assert at(frame, 0)["visibility_in_air"][0] == pytest.approx(17000.0)
    assert at(frame, 0)["cloud_amount"][0] == pytest.approx(8.0)


# ---------------------------------------------------------------------- contract


def test_midas_satisfies_the_source_protocol():
    assert isinstance(MidasOpen(), Source)


def test_constructing_the_source_does_not_touch_the_token(monkeypatch, tmp_path):
    """Importing and building a source must not demand a credential."""
    monkeypatch.setenv(TOKEN_ENV, str(tmp_path / "absent"))
    src = MidasOpen()
    assert src.token_file == tmp_path / "absent"
    assert src._session is None


# -------------------------------------------------------------- optional suites


@pytest.mark.skipif(
    not LOCAL_ARCHIVE.is_dir(), reason="the local MIDAS archive is not mounted here"
)
def test_the_local_archive_still_matches_the_pinned_fingerprint_and_units():
    """Opt-in: the shipped pins against real files, if they are around."""
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    files = sorted(LOCAL_ARCHIVE.rglob("*.csv"))[:25]
    raws = [
        RawFile(p, "0" * 64, p.stat().st_size, f"midas/hourly_weather/dv-202607/{p.parent.name}/{p.stem}")
        for p in files
    ]
    report = MidasOpen().scan(raws)
    assert report["errors"] == []
    assert report["fingerprint"] == cfg.schema_fingerprint
    for column, expected in cfg.extra["expected_units"].items():
        assert report["declared_units"][column] == [expected], column


@pytest.mark.skipif(
    not (os.environ.get("METEOPACK_LIVE") and token_file().exists()),
    reason="set METEOPACK_LIVE=1 and provide a CEDA token file to hit the real archive",
)
def test_live_catalogue_and_station_metadata():
    """Opt-in: the keyless listing plus one token-protected read."""
    src = MidasOpen()
    cfg = ProductConfig(**yaml.safe_load(PRODUCT_YAML.read_text()))
    assert src.county_index(src.dataset_version(cfg))
    table = src.stations(cfg)
    assert table.height > 100
    assert table["height_m"].null_count() < table.height
