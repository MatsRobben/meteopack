"""HARMONIE Cy43 NL: both packagings checked, read, coarsened and streamed into a lead-time store.

Synthetic NetCDF files and P1 GRIB tars on a tiny 12 x 9 source grid, written
to look like the KDP ones, served by a fake KDP client; the product config is
the real factor-2 one shrunk to leads 0-3, so the store grid is 6 x 4 and the
odd eastern source column is dropped.
"""
from __future__ import annotations

import io
import json
import shutil
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from meteopack.core import grids, kdp
from meteopack.core.grids import Grid
from meteopack.core.store import gridded
from meteopack.jobs import stream_kdp
from meteopack.jobs.plan import load_product_config
from meteopack.sources.nwp import knmi_harmonie as hm
from test_grib1 import encode_message

LONLAT = "+proj=longlat +a=6367470 +e=0 +no_defs"
LAT0, DLAT, NLAT, LON0, DLON, NLON = 51.0, 0.25, 12, 2.5, 0.35, 9
#: The source coarsened 2 x 2: 6 x 4 blocks, the 9th (eastern) column dropped.
TINY = Grid(id="tiny_ll_f2", proj4=LONLAT, shape=(6, 4),
            extent=((LON0 - DLON / 2, LON0 - DLON / 2 + 8 * DLON),
                    (LAT0 - DLAT / 2, LAT0 - DLAT / 2 + 12 * DLAT)))
LEADS = [0, 1, 2, 3]
DAY_NC = datetime(2026, 1, 7, tzinfo=UTC)
DAY_P1 = datetime(2026, 1, 8, tzinfo=UTC)
KT = 1852 / 3600

#: field -> (NetCDF file parameter, variable, units, level dim, level, nc = (phys - offset) / factor)
NC = {
    "mslp": ("air-pressure-at-mean-sea-level-hagl", "air-pressure-at-mean-sea-level-hagl",
             "hPa", "mslp_at_hagl", 0, 100.0, 0.0),
    "t2m": ("air-temperature-hagl", "air-temperature-hagl", "C", "temp_at_hagl", 2, 1.0, 273.15),
    "rh2m": ("relative-humidity-hagl", "relative-humidity-hagl", "-", "rh_at_hagl", 2, 1.0, 0.0),
    "u10": ("wind-speed-components-hagl", "x-wind-hagl", "kts", "xwind_at_hagl", 10, KT, 0.0),
    "v10": ("wind-speed-components-hagl", "y-wind-hagl", "kts", "ywind_at_hagl", 10, KT, 0.0),
    "gust10": ("wind-speed-of-gust-01h-hagl", "wind-speed-of-gust-01h-hagl", "kts",
               "gust1h_at_hagl", 10, KT, 0.0),
    "mld": ("mixed-layer-depth-atm", "mixed-layer-depth-atm", "m", "blh_at_atm", 0, 1.0, 0.0),
    "cloud_base": ("cloud-base-altitude-atm", "cloud-base-altitude-atm", "ft", "cldbase_at_atm",
                   0, 0.3048, 0.0),
    "visibility": ("visibility-in-air-hagl", "visibility-in-air-hagl", "m", "vis_at_hagl", 2,
                   1.0, 0.0),
    "tp_1h": ("total-precipitation-accumulation-01h-hagl",
              "total-precipitation-accumulation-01h-hagl", "mm", "precipacc1h_at_hagl", 0,
              1.0, 0.0),
}


@pytest.fixture(autouse=True)
def _tiny_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    real = grids.get
    monkeypatch.setattr(grids, "get",
                        lambda grid_id, path=None: TINY if grid_id == TINY.id else real(grid_id))


def tiny_cfg():
    cfg = load_product_config("knmi_harmonie", "cy43_nl_4km")
    levels = cfg.levels.model_copy(update={"values": [float(v) for v in LEADS]})
    extra = json.loads(json.dumps(cfg.extra))
    extra["harmonie"]["source_grid"] = {"lat0": LAT0, "dlat": DLAT, "nlat": NLAT, "lon0": LON0,
                                        "dlon": DLON, "nlon": NLON, "tolerance_deg": 1e-4}
    extra["timeline_origin"] = extra["stream_start"] = DAY_NC.isoformat()
    store = cfg.store.model_copy(update={
        "chunks": {"time": 1, "lead_time": len(LEADS), "y": 6, "x": 4},
        "shard": {"time": 24, "lead_time": len(LEADS), "y": 6, "x": 4},
        "compression_level": 1})
    return cfg.model_copy(update={"grid_id": TINY.id, "levels": levels, "extra": extra,
                                  "store": store})


def phys(name: str, run: datetime, lead: int) -> np.ndarray:
    """The physical (store-unit) value of a field: smooth in space, distinct per run and lead."""
    r = run.hour
    i = np.arange(NLAT)[:, None]
    j = np.arange(NLON)[None, :]
    base = {
        "mslp": 101000.0 + 100 * lead + 10 * i + r, "t2m": 280.0 + lead + 0.1 * j + 0.01 * r,
        "rh2m": 0.5 + 0.05 * lead + 0.01 * i, "u10": 3.0 + lead + 0.1 * j,
        "v10": -2.0 - 0.5 * lead + 0.1 * i, "gust10": 10.0 + lead + 0.2 * i + 0.1 * j,
        "mld": 500.0 + 50 * lead + 5 * j, "cloud_base": 1000.0 + 100 * lead + 10 * i,
        "visibility": 20000.0 + 1000 * lead + 100 * j, "tp_1h": 0.5 * lead + 0.1 * i + 0.01 * r,
    }[name] + np.zeros((NLAT, NLON))
    if name == "cloud_base":
        base[:, :4] = np.nan  # cloud-free western columns
    return base


def write_netcdf(directory: Path, run: datetime, param: str, *, units_override=None,
                 n_time: int = 4) -> Path:
    import h5netcdf

    specs = [(name, s) for name, s in NC.items() if s[0] == param]
    name = f"uwcw_ha43_nl_2km_{param}_{run:%Y%m%dT%H}.nc"
    path = directory / name
    run_s = int(run.timestamp())
    with h5netcdf.File(path, "w") as f:
        f.attrs["cycle_datetime"] = f"{run:%Y-%m-%dT%H}"
        f.attrs["title"] = "Arcus dataset"
        dims = {"time": n_time, "latitude": NLAT, "longitude": NLON}
        for _, s in specs:
            dims[s[3]] = 1
        f.dimensions = dims
        t = f.create_variable("time", ("time",), "i8")
        t[:] = run_s + 3600 * np.arange(1, n_time + 1)
        t.attrs["units"] = "seconds since 1970-01-01T00:00:00+00:00"
        lat = f.create_variable("latitude", ("latitude",), "f8")
        lat[:] = LAT0 + DLAT * np.arange(NLAT)
        lon = f.create_variable("longitude", ("longitude",), "f8")
        lon[:] = LON0 + DLON * np.arange(NLON)
        ref = f.create_variable("forecast_reference_time", (), "i8")
        ref[()] = run_s
        gm = f.create_variable("latitude_longitude", (), "i8")
        gm.attrs["proj4_params"] = "+proj=longlat +a=6367470 +e=0 +no_defs"
        for field_name, (_, var, units, level_dim, level, factor, offset) in specs:
            lv = f.create_variable(level_dim, (level_dim,), "i8")
            lv[:] = [level]
            v = f.create_variable(var, ("time", level_dim, "latitude", "longitude"), "f4",
                                  fillvalue=np.float32("nan"))
            data = np.full((n_time, 1, NLAT, NLON), np.nan, dtype="float32")
            for k in range(n_time):
                data[k, 0] = (phys(field_name, run, k + 1) - offset) / factor
            v[:] = data
            v.attrs["units"] = units_override or units
    return path


def p1_member(run: datetime, lead: int, *, bad_grid: bool = False) -> bytes:
    """One GRIB member: every needed message (and a distractor), APCP accumulated."""
    kw = {"run": run, "lat0": LAT0, "lon0": LON0 + (1.0 if bad_grid else 0.0), "dlat": DLAT,
          "dlon": DLON}
    out = [encode_message(np.full((NLAT, NLON), 1.0), param=81, **kw)]  # land fraction

    def inst(name, param, lt, level, decimal=0, bits=16):
        out.append(encode_message(phys(name, run, lead), param=param, level_type=lt,
                                  level=level, p1=lead, decimal=decimal, bits=bits, **kw))

    inst("mslp", 1, 103, 0)
    inst("t2m", 11, 105, 2)
    inst("rh2m", 52, 105, 2, decimal=2)
    inst("u10", 33, 105, 10, decimal=1)
    inst("v10", 34, 105, 10, decimal=1)
    gust = phys("gust10", run, lead)
    for param, part in ((162, 0.6), (163, 0.8)):
        out.append(encode_message(gust * part, param=param, level=10, time_range=2,
                                  p1=max(lead - 1, 0), p2=lead, decimal=1, **kw))
    inst("mld", 67, 105, 0)
    inst("cloud_base", 186, 200, 0, bits=24)
    inst("visibility", 20, 105, 0)
    acc = sum((phys("tp_1h", run, k) for k in range(1, lead + 1)),
              np.zeros((NLAT, NLON)))
    out.append(encode_message(acc, param=61, time_range=4, p1=0, p2=lead, decimal=2,
                              bits=16 if lead else 0, **kw))
    return b"".join(out)


def p1_tar(run: datetime, *, bad_lead: int | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for lead in (0, 3, 1, 2, 4):  # not in lead order, like the real tars
            data = p1_member(run, lead, bad_grid=lead == bad_lead)
            info = tarfile.TarInfo(f"HA43_N20_{run:%Y%m%d%H}00_{lead:03d}00_GB")
            info.size = len(data)
            info.pax_headers = {"mtime": "1767830400.5"}
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


class FakeKey:
    path = Path("/fake/key")


class FakeClient:
    """The KDP as the stream sees it: two datasets, listings, URLs and downloads."""

    def __init__(self, upstream: dict[str, Path]) -> None:
        self.upstream = upstream
        self.dataset = "uwcw_extra_lv_ha43_nl_2km"
        self.key = FakeKey()
        self.requests_made = 0
        self.downloads: list[str] = []
        self.limiter = kdp.QuotaLimiter(1000.0)
        self.download_session = None
        self.stats: dict = {}

    def refresh_key(self) -> bool:
        return False

    def list_files(self, begin=None, end=None, *, dataset=None, version=None):
        self.requests_made += 1
        root = self.upstream[dataset or self.dataset]
        return [kdp.RemoteFile(p.name, p.stat().st_size)
                for p in sorted(root.iterdir()) if begin <= p.name <= end]

    def download_url(self, filename, *, dataset=None, version=None):
        self.requests_made += 1
        self.downloads.append(filename)
        return f"fake://{dataset}/{filename}"

    def download(self, remote, dest):
        if dest.exists() and dest.stat().st_size == remote.size:
            return kdp.RawFile(path=dest, sha256="", n_bytes=remote.size,
                               unit_key=remote.filename, attempts=0)
        self.requests_made += 1
        self.downloads.append(remote.filename)
        shutil.copy(self.upstream[self.dataset] / remote.filename, dest)
        return kdp.RawFile(path=dest, sha256="", n_bytes=remote.size,
                           unit_key=remote.filename, attempts=1)


class TarReader:
    """RangeReader over a local tar: `fake://dataset/name` -> upstream file."""

    def __init__(self, upstream, url, session=None, total_size=None):
        _, _, rest = url.partition("fake://")
        dataset, _, name = rest.partition("/")
        self.blob = (upstream[dataset] / name).read_bytes()
        self.requests = 0

    def read(self, start, length):
        self.requests += 1
        return self.blob[start:start + length]


@pytest.fixture
def upstream(tmp_path: Path) -> dict[str, Path]:
    """Day NC: runs 00-02, run 01 lacks its cloud-base file, run 02's T file says K.
    Day P1: P1 runs 00 and 01 (01 has a member on the wrong grid), NetCDF 00-02 (02 has
    no P1, so it falls back)."""
    nc = tmp_path / "up_nc"
    p1 = tmp_path / "up_p1"
    nc.mkdir()
    p1.mkdir()
    params = list(dict.fromkeys(s[0] for s in NC.values()))
    for day in (DAY_NC, DAY_P1):
        for hour in (0, 1, 2):
            run = day + timedelta(hours=hour)
            for param in params:
                if day == DAY_NC and hour == 1 and param == "cloud-base-altitude-atm":
                    continue
                broken = day == DAY_NC and hour == 2 and param == "air-temperature-hagl"
                write_netcdf(nc, run, param, units_override="K" if broken else None)
    for hour, bad in ((0, None), (1, 2)):
        run = DAY_P1 + timedelta(hours=hour)
        (p1 / f"HARM43_V1_P1_{run:%Y%m%d%H}.tar").write_bytes(p1_tar(run, bad_lead=bad))
    # A P1 tar from before p1_from must never be used.
    early = DAY_NC + timedelta(hours=2)
    (p1 / f"HARM43_V1_P1_{early:%Y%m%d%H}.tar").write_bytes(p1_tar(early))
    return {"uwcw_extra_lv_ha43_nl_2km": nc, "harmonie_arome_cy43_p1": p1}


def stream(cfg, root, client, upstream, **kw):
    settings = stream_kdp.Settings.from_config(cfg)
    fmt = hm.HarmonieFormat(cfg, settings)
    fmt.reader_factory = lambda url, session=None, total_size=None: TarReader(
        upstream, url, session, total_size)
    s = stream_kdp.Stream(cfg, root, client=client, fmt=fmt, pack_workers=2,
                          download_workers=3, **kw)
    for day in (DAY_NC, DAY_P1):
        s.pack_day(s.fetch_day(day))
    return s


def expected(cfg, name, run, lead):
    """The physical value the store should hold for one field: 2 x 2 block means
    (gust: maxima) of the 12 x 8 source cells kept, north-up - computed here
    directly, not with the module's own function."""
    v = phys(name, run, lead)[:, :8].reshape(6, 2, 4, 2)
    if name == "gust10":
        out = v.max(axis=(1, 3))
    elif name == "cloud_base":
        ok = np.isfinite(v)
        with np.errstate(all="ignore"):
            out = np.where(ok, v, 0).sum(axis=(1, 3)) / ok.sum(axis=(1, 3))
        out[ok.mean(axis=(1, 3)) < 0.5] = np.nan
    else:
        out = v.mean(axis=(1, 3))
    return out[::-1]


def test_schema_reads_the_real_config():
    schema = hm.Schema.from_config(load_product_config("knmi_harmonie", "cy43_nl_4km"))
    assert schema.leads == tuple(range(13)) and schema.nc_leads == list(range(1, 13))
    assert schema.p1_leads == list(range(13)) and schema.factor == 2
    schema.check_grid()
    two = hm.Schema.from_config(load_product_config("knmi_harmonie", "cy43_nl_2km"))
    assert two.factor == 1 and two.grid_id == "harmonie_nl_2km"
    two.check_grid()
    gust = next(f for f in schema.fields if f.name == "gust10")
    assert gust.aggregate == "max" and not gust.lead0
    assert [f.name for f in schema.fields if not f.lead0] == ["gust10", "tp_1h"]
    assert len(schema.nc_params) == 9  # the wind components share a file
    assert [f.name for f in schema.fields] == ["mslp", "t2m", "rh2m", "u10", "v10", "gust10",
                                               "mld", "cloud_base", "visibility", "tp_1h"]
    assert schema.p1_tar_name(DAY_P1) == "HARM43_V1_P1_2026010800.tar"
    assert schema.p1_member_name(DAY_P1, 7) == "HA43_N20_202601080000_00700_GB"
    assert schema.parse_nc("uwcw_ha43_nl_2km_air-temperature-hagl_20240625T06.nc") == (
        "air-temperature-hagl", datetime(2024, 6, 25, 6, tzinfo=UTC))
    assert schema.parse_nc("uwcw_ha43_bess_air-temperature-hagl_20240625T06.nc") is None


def test_netcdf_and_p1_of_the_same_run_pack_to_the_same_store_values(tmp_path, upstream):
    cfg = tiny_cfg()
    schema = hm.Schema.from_config(cfg)
    run = DAY_P1
    nc_dir = tmp_path / "nc"
    nc_dir.mkdir()
    for param in schema.nc_params:
        shutil.copy(upstream["uwcw_extra_lv_ha43_nl_2km"] /
                    schema.nc_filename(param, run), nc_dir)
    p1_dir = tmp_path / "p1"
    p1_dir.mkdir()
    with tarfile.open(upstream["harmonie_arome_cy43_p1"] / schema.p1_tar_name(run)) as tar:
        tar.extractall(p1_dir, filter="data")
    a = hm.pack_run(nc_dir, hm.RunFiles("a", 0, None, hm.NETCDF, run), schema)
    b = hm.pack_run(p1_dir, hm.RunFiles("b", 0, None, hm.P1, run), schema)
    assert a.check.ok and b.check.ok, (a.check.errors, b.check.errors)
    assert a.check.fields_present == b.check.fields_present == [f.name for f in schema.fields]
    for spec in schema.fields:
        enc = spec.encoding
        # Leads 1-3 agree; GRIB packing may move a rounding by one code.
        diff = np.abs(a.codes[spec.name][1:].astype(int) - b.codes[spec.name][1:].astype(int))
        assert diff.max() <= 1, spec.name
        # Lead 0: never from the NetCDF; from P1 except the past-hour fields.
        assert (a.codes[spec.name][0] == enc.missing).all(), spec.name
        assert (b.codes[spec.name][0] == enc.missing).all() == (not spec.lead0), spec.name
        for k, lead in enumerate(LEADS):
            if lead == 0 and not spec.lead0:
                continue
            want = expected(cfg, spec.name, run, lead)
            got = b.codes[spec.name][k]
            assert got.shape == (6, 4)
            if spec.undefined == "cloud_free":
                assert (got[np.isnan(want)] == hm.cloud_free_code(enc)).all()
                assert np.isnan(want).any() and np.isfinite(want).any()
            ok = np.isfinite(want)
            np.testing.assert_allclose(got[ok] * enc.scale + enc.offset, want[ok], rtol=1e-6,
                                       atol=enc.scale / 2 + 1e-9, err_msg=spec.name)
    assert a.check.fingerprint != b.check.fingerprint


def test_block_reduce_means_maxima_edges_and_north_up():
    v = np.arange(5 * 7, dtype=float).reshape(5, 7)  # rows south to north
    one = hm.block_reduce(v, 1)
    np.testing.assert_array_equal(one, v[::-1])
    two = hm.block_reduce(v, 2)
    assert two.shape == (2, 3)  # the northern row and the eastern column are dropped
    np.testing.assert_allclose(two[-1, 0], v[0:2, 0:2].mean())  # south-west block, last row
    np.testing.assert_allclose(two[0, 2], v[2:4, 4:6].mean())
    np.testing.assert_allclose(hm.block_reduce(v, 2, "max")[-1, 0], v[1, 1])
    w = v.copy()
    w[0, 0] = np.nan
    np.testing.assert_allclose(hm.block_reduce(w, 2)[-1, 0], np.nanmean(w[0:2, 0:2]))
    w[0:2, 0:2] = [[np.nan, np.nan], [np.nan, 1.0]]
    assert np.isnan(hm.block_reduce(w, 2, min_valid=0.5)[-1, 0])
    stack = np.stack([v, v + 1])
    assert hm.block_reduce(stack, 2).shape == (2, 2, 3)


def test_a_grid_that_does_not_match_the_factor_is_refused(monkeypatch):
    cfg = tiny_cfg()
    extra = json.loads(json.dumps(cfg.extra))
    extra["harmonie"]["coarsen_factor"] = 1
    with pytest.raises(ValueError, match="coarsened by 1"):
        hm.Schema.from_config(cfg.model_copy(update={"extra": extra})).check_grid()


def test_a_unit_change_fails_the_file_not_the_run(tmp_path, upstream):
    cfg = tiny_cfg()
    schema = hm.Schema.from_config(cfg)
    run = DAY_NC + timedelta(hours=2)
    d = tmp_path / "run"
    d.mkdir()
    for param in schema.nc_params:
        shutil.copy(upstream["uwcw_extra_lv_ha43_nl_2km"] / schema.nc_filename(param, run), d)
    packed = hm.pack_run(d, hm.RunFiles("r", 0, None, hm.NETCDF, run), schema)
    bad = schema.nc_filename("air-temperature-hagl", run)
    assert packed.check.ok and packed.check.fields_missing == ["t2m"]
    assert "units 'K', expected 'C'" in packed.check.file_errors[bad][0]
    assert packed.check.quarantine == [bad] and "t2m" not in packed.codes


def test_values_out_of_range_fail_the_check(tmp_path):
    cfg = tiny_cfg()
    schema = hm.Schema.from_config(cfg)
    spec = next(f for f in schema.fields if f.name == "rh2m")
    assert hm._range_errors(spec, np.array([0.2, 99.0]))  # percent, not a fraction
    assert not hm._range_errors(spec, np.array([0.2, 1.0001, np.nan]))


def test_stream_two_days_flags_partial_and_corrupt_runs_and_cleans_up(tmp_path, upstream):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    client = FakeClient(upstream)
    stream(cfg, root, client, upstream)
    s_nc = stream_kdp.read_status(cfg, root, DAY_NC)
    s_p1 = stream_kdp.read_status(cfg, root, DAY_P1)

    assert s_nc["listed"] == 3 and s_nc["packed"] == 3 and s_nc["state"] == "partial"
    assert s_nc["partial_runs"] == {"0100": ["cloud_base"], "0200": ["t2m"]}
    assert s_nc["runs_by_source"] == {"netcdf": 3}
    bad_t = "uwcw_ha43_nl_2km_air-temperature-hagl_20260107T02.nc"
    assert bad_t in s_nc["schema_failed"]
    # P1 run 01 failed its check (member on the wrong grid): not packed, no fallback.
    assert s_p1["packed"] == 2 and s_p1["runs_by_source"] == {"p1": 1, "netcdf": 1}
    assert "HARM43_V1_P1_2026010801.tar" in s_p1["schema_failed"]
    assert s_p1["listed_not_packed"] == ["0100"]

    # Listing: a month at a time - 9 NetCDF parameters + 1 P1 listing, once.
    assert sum(1 for d in client.downloads if d.endswith(".tar")) == 2

    ds = gridded.read(cfg.store_id, root, decode=False)
    assert ds["t2m"].dims == ("time", "lead_time", "y", "x")
    np.testing.assert_array_equal(ds["lead_time"].values, LEADS)
    assert ds["lead_time"].attrs["standard_name"] == "forecast_period"
    # The axis starts at the (tiny config's) origin, DAY_NC 00Z: row = hours since.
    assert ds["time"].values[0] == np.datetime64("2026-01-07T00:00", "ns")
    src = ds["source_dataset"].values
    np.testing.assert_array_equal(src[:3], [1, 1, 1])
    np.testing.assert_array_equal(src[24:27], [2, 0, 1])
    np.testing.assert_array_equal(ds["source_layout_id"].values[24:27],
                                  [hm.LAYOUT_P1, hm.LAYOUT_CORRUPT, hm.LAYOUT_NETCDF])
    np.testing.assert_array_equal(ds["is_corrupt_frame"].values[24:27], [False, True, False])
    np.testing.assert_array_equal(ds["is_partial_run"].values[:4], [0, 1, 1, -1])
    assert ds["fields_present"].values[0] == 1023
    assert ds["fields_present"].values[1] == 1023 - (1 << 7)   # no cloud base
    assert ds["is_missing_frame"].values[3] and not ds["is_missing_frame"].values[0]
    # A partial run's missing field reads missing; its other fields are there.
    enc = hm.Schema.from_config(cfg).fields[7].encoding
    assert (ds["cloud_base"].values[1] == enc.missing).all()
    want = expected(cfg, "mslp", DAY_NC + timedelta(hours=1), 3)
    mslp = hm.Schema.from_config(cfg).fields[0].encoding
    got = ds["mslp"].values[1, 3].astype(float) * mslp.scale + mslp.offset
    np.testing.assert_allclose(got, want, atol=mslp.scale / 2)
    # Decoded through xarray's CF rules, missing is NaN: a NetCDF run's lead 0,
    # every field of a run that is not there, a P1 run's past-hour fields at +0.
    dec = gridded.read(cfg.store_id, root)
    assert np.isnan(dec["t2m"].values[0, 0]).all() and np.isfinite(dec["t2m"].values[0, 1]).all()
    assert np.isnan(dec["t2m"].values[3]).all()
    assert np.isfinite(dec["t2m"].values[24, 0]).all()
    assert np.isnan(dec["gust10"].values[24, 0]).all()
    assert np.isnan(dec["tp_1h"].values[24, 0]).all()
    # The grid: HARMONIE's own lat/lon cells (2 x 2 block centres), north-up.
    np.testing.assert_allclose(ds["x"].values, LON0 + DLON / 2 + 2 * DLON * np.arange(4))
    np.testing.assert_allclose(ds["y"].values,
                               (LAT0 + DLAT / 2 + 2 * DLAT * np.arange(6))[::-1])
    assert ds["x"].attrs["units"] == "degrees_east"
    np.testing.assert_allclose(ds["lat"].values[:, 0], ds["y"].values, atol=1e-5)
    np.testing.assert_allclose(ds["lon"].values[0], ds["x"].values, atol=1e-5)

    # Raw data is gone; the failed T file and the failed P1 run are kept aside.
    raw = root / "raw/nwp/knmi_harmonie/cy43_nl_4km"
    left = sorted(str(p.relative_to(raw)) for p in raw.rglob("*") if p.is_file())
    assert f"quarantine/uwcw_ha43_nl_2km_20260107T02/{bad_t}" in left
    assert any(p.startswith("quarantine/HARM43_V1_P1_2026010801/") for p in left)
    assert all(p.startswith("quarantine/") for p in left), left

    # Provenance: one row per run, with its files.
    import polars as pl

    prov = pl.read_parquet(gridded.store_path(root, cfg.store_id) / "provenance.parquet")
    assert prov.height == 6
    row = prov.filter(pl.col("filename") == "HARM43_V1_P1_2026010800.tar").row(0, named=True)
    assert row["source_dataset"] == "harmonie_arome_cy43_p1" and row["n_files"] == 4  # +0..+3

    # Store metadata: the run axis, the cloud-free code.
    import zarr

    group = zarr.open_group(gridded.zarr_path(root, cfg.store_id), mode="r")
    assert group["time"].attrs["standard_name"] == "forecast_reference_time"
    assert group["cloud_base"].attrs["cloud_free_code"] == 65533
    assert group.attrs["dims"] == ["time", "lead_time", "y", "x"]


def test_rerun_resumes_without_downloading_again(tmp_path, upstream):
    cfg = tiny_cfg()
    root = tmp_path / "root"
    client = FakeClient(upstream)
    stream(cfg, root, client, upstream)
    before = list(client.downloads)
    # Day P1 is partial (run 01 corrupt): a rerun re-checks it; files of runs
    # already packed are not fetched again.
    settings = stream_kdp.Settings.from_config(cfg)
    fmt = hm.HarmonieFormat(cfg, settings)
    fmt.reader_factory = lambda url, session=None, total_size=None: TarReader(
        upstream, url, session, total_size)
    again = stream_kdp.Stream(cfg, root, client=client, fmt=fmt, pack_workers=1)
    fetched = again.fetch_day(DAY_P1)
    assert sorted(fetched.listed) and set(fetched.paths) == {DAY_P1 + timedelta(hours=1)}
    new = client.downloads[len(before):]
    assert new == ["HARM43_V1_P1_2026010801.tar"]


def test_a_killed_p1_download_resumes_from_its_member_index(tmp_path, upstream):
    cfg = tiny_cfg()
    schema = hm.Schema.from_config(cfg)
    client = FakeClient(upstream)
    tar = upstream["harmonie_arome_cy43_p1"] / "HARM43_V1_P1_2026010800.tar"
    files = hm.RunFiles(tar.name, tar.stat().st_size, None, hm.P1, DAY_P1,
                        files=(kdp.RemoteFile(tar.name, tar.stat().st_size),))
    readers = []

    def factory(url, session=None, total_size=None):
        readers.append(TarReader(upstream, url, session, total_size))
        return readers[-1]

    d = tmp_path / "run"
    d.mkdir()
    hm.download_p1(client, files, d, schema, factory)
    members = sorted(p.name for p in d.iterdir() if not p.name.startswith("."))
    assert members == [schema.p1_member_name(DAY_P1, k) for k in (0, 1, 2, 3)]
    # Lose one member, as a kill mid-run would: only it is read again, no walk.
    (d / members[1]).unlink()
    _, fresh = hm.download_p1(client, files, d, schema, factory)
    assert fresh == 1 and readers[-1].requests == 1
    # Everything present: no request at all.
    calls = client.requests_made
    assert hm.download_p1(client, files, d, schema, factory) == (0, 0)
    assert client.requests_made == calls
