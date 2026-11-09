"""NOAA GFS: index parsing, message checks, regridding, de-averaging, and the stream end to end."""
from __future__ import annotations

import json
import zlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from meteopack.core import grids, http
from meteopack.core.encoding import Encoding
from meteopack.core.store import gridded
from meteopack.jobs import stream_gfs, stream_kdp
from meteopack.jobs.plan import load_product_config
from meteopack.sources.nwp import gfs

eccodes = pytest.importorskip("eccodes")

IDX = """\
1:0:d=2024060100:PRMSL:mean sea level:1 hour fcst:
2:1014099:d=2024060100:CLMR:1 hybrid level:1 hour fcst:
3:1102641:d=2024060100:PRATE:surface:1 hour fcst:
4:1200000:d=2024060100:PRATE:surface:0-1 hour ave fcst:
5:1300000:d=2024060100:TMP:850 mb:1 hour fcst:
"""

#: A 2 degree global grid keeps synthetic messages small (91 x 180).
SRC = gfs.SourceGrid(ni=180, nj=91, lat_first=90.0, lon_first=0.0, dlat=2.0, dlon=2.0)


# ----------------------------------------------------------------- the index

def test_parse_idx_gives_inclusive_byte_ranges_and_kinds():
    e = gfs.parse_idx(IDX)
    assert [x.start for x in e] == [0, 1014099, 1102641, 1200000, 1300000]
    assert e[0].end == 1014098 and e[-1].end is None
    assert e[2].is_instant and e[2].average_window is None
    assert e[3].average_window == (0, 1) and not e[3].is_instant
    assert gfs.find_entry(e, "PRATE", "surface").number == "3"
    assert gfs.find_entry(e, "PRATE", "surface", average=True).number == "4"
    assert gfs.find_entry(e, "TMP", "500 mb") is None


def test_parse_idx_rejects_garbage_and_decreasing_offsets():
    with pytest.raises(ValueError):
        gfs.parse_idx("<html>not found</html>")
    with pytest.raises(ValueError):
        gfs.parse_idx("1:100:d=2024060100:A:surface:anl:\n2:50:d=2024060100:B:surface:anl:\n")


def test_coalesce_merges_only_adjacent_messages():
    e = gfs.parse_idx(IDX)
    groups = gfs.coalesce([e[4], e[0], e[1], e[3]])
    assert [[x.number for x in g] for g in groups] == [["1", "2"], ["4", "5"]]


def test_archive_paths_switch_to_atmos_with_v16():
    v15 = datetime(2021, 3, 22, 6, tzinfo=UTC)
    v16 = datetime(2021, 3, 22, 12, tzinfo=UTC)
    assert gfs.file_key(v15, 3) == "gfs.20210322/06/gfs.t06z.pgrb2.0p25.f003"
    assert gfs.file_key(v16, 3, "pgrb2b") == "gfs.20210322/12/atmos/gfs.t12z.pgrb2b.0p25.f003"
    assert gfs.floor_run(datetime(2024, 1, 1, 17, 55, tzinfo=UTC)).hour == 12


# ------------------------------------------------------------- de-averaging

def gfs_means(hourly: dict[int, np.ndarray]) -> dict[int, tuple[int, np.ndarray]]:
    """What GFS writes: at lead L the mean since the last 6-hour boundary before L."""
    out = {}
    for lead in hourly:
        start = (lead - 1) // 6 * 6
        out[lead] = (start, np.mean([hourly[k] for k in range(start + 1, lead + 1)], axis=0))
    return out


def test_hourly_means_recover_every_hour_across_window_resets():
    rng = np.random.default_rng(1)
    hourly = {lead: rng.gamma(0.5, 2.0, (4, 5)) for lead in range(1, 16)}
    got = gfs.hourly_means(gfs_means(hourly))
    assert sorted(got) == list(range(1, 16))
    for lead, value in hourly.items():
        np.testing.assert_allclose(got[lead], value, atol=1e-12)


def test_hourly_means_skip_a_lead_whose_predecessor_is_missing():
    hourly = {lead: np.full(3, float(lead)) for lead in range(1, 10)}
    means = gfs_means(hourly)
    del means[3]
    got = gfs.hourly_means(means)
    assert 4 not in got and 3 not in got
    assert {1, 2, 5, 6, 7, 8, 9} <= set(got)     # 5 needs 4, 7 starts a new window
    np.testing.assert_allclose(got[7], 7.0)


def test_three_hourly_means_give_no_hourly_values():
    """The 2021-01/02 archive: f003 = mean 0-3, f006 = mean 0-6. Nothing is one hour."""
    means = {3: (0, np.ones(2)), 6: (0, np.ones(2))}
    assert gfs.hourly_means(means) == {}


# --------------------------------------------------------------- regridding

def test_bilinear_reproduces_a_linear_field_and_wraps_at_greenwich():
    lat = np.array([[52.3, 10.1], [-30.0, 0.0]])
    lon = np.array([[-3.7, 359.0], [5.0, 0.5]])
    r = gfs.Regridder(lon, lat, SRC)
    lats = SRC.lat_first - SRC.dlat * np.arange(SRC.nj)
    field = np.repeat((2.0 * lats)[:, None], SRC.ni, axis=1)
    np.testing.assert_allclose(r.bilinear(field), 2.0 * lat, atol=1e-9)
    lons = SRC.dlon * np.arange(SRC.ni)
    wrapped = np.repeat(np.cos(np.radians(lons))[None, :], SRC.nj, axis=0)
    got = r.bilinear(wrapped)
    c = lambda d: np.cos(np.radians(d))
    assert got[0, 0] == pytest.approx(c(356.0) * 0.85 + c(358.0) * 0.15, abs=1e-9)   # -3.7
    # 359 E lies across the 358 -> 0 seam: half cos(358), half cos(0).
    assert got[0, 1] == pytest.approx(0.5 * c(358.0) + 0.5 * c(0.0), abs=1e-9)


def test_bilinear_drops_masked_neighbours_and_nearest_keeps_codes():
    r = gfs.Regridder(np.array([[1.0]]), np.array([[89.0]]), SRC)   # halfway in both
    field = np.zeros((SRC.nj, SRC.ni))
    field[0, 0], field[0, 1], field[1, 0], field[1, 1] = 1.0, 2.0, 3.0, np.nan
    assert r.bilinear(field)[0, 0] == pytest.approx(2.0)            # mean of the three
    field[0, 1] = field[1, 0] = np.nan                               # 25 % weight left
    assert np.isnan(r.bilinear(field)[0, 0])
    cat = np.zeros((SRC.nj, SRC.ni))
    cat[0, 1] = 1.0
    r2 = gfs.Regridder(np.array([[1.2]]), np.array([[89.5]]), SRC)
    assert r2.nearest(cat)[0, 0] == 1.0 and r2.bilinear(cat)[0, 0] not in (0.0, 1.0)


def test_the_model_frame_is_north_up_and_centred_on_the_netherlands():
    g = grids.get("knmi_stereo_16km_1536")
    assert g.shape == (96, 96)
    lon, lat = g.lonlat()
    assert lat[0, 48] > lat[-1, 48]                      # row 0 is the northern edge
    assert 51.5 < lat[47:49, 47:49].mean() < 53.0 and 4.0 < lon[47:49, 47:49].mean() < 6.5
    y, x = g.coordinates()
    assert x[0] == -418 + 8 and y[0] == -3264 - 8 and x[1] - x[0] == 16


# ----------------------------------------------------------------- encoding

def test_encode_field_clips_counts_and_marks_undefined():
    enc = Encoding(np.dtype("uint16"), 0.5, -10.0, 65534, 65535)
    got = gfs.encode_field(np.array([-20.0, -10.0, 0.0, np.nan, 1e9]), enc)
    assert got.codes.tolist() == [0, 0, 20, 65535, 65533]
    assert (got.n_below, got.n_above, got.n_undefined) == (1, 1, 1)


# ------------------------------------------------------------ GRIB messages

def grib_message(values: np.ndarray, *, run: datetime, lead: int, sig, avg_from=None,
                 bitmap: bool = False, grid: gfs.SourceGrid = SRC) -> bytes:
    """A GRIB2 message on `grid`, float32 IEEE-packed so values decode exactly."""
    h = eccodes.codes_grib_new_from_samples("GRIB2")
    try:
        if avg_from is not None:
            eccodes.codes_set(h, "productDefinitionTemplateNumber", 8)
        for key, value in (("Ni", grid.ni), ("Nj", grid.nj),
                           ("latitudeOfFirstGridPointInDegrees", grid.lat_first),
                           ("longitudeOfFirstGridPointInDegrees", grid.lon_first),
                           ("latitudeOfLastGridPointInDegrees",
                            grid.lat_first - grid.dlat * (grid.nj - 1)),
                           ("longitudeOfLastGridPointInDegrees",
                            grid.lon_first + grid.dlon * (grid.ni - 1)),
                           ("iDirectionIncrementInDegrees", grid.dlon),
                           ("jDirectionIncrementInDegrees", grid.dlat),
                           ("jScansPositively", 0), ("iScansNegatively", 0)):
            eccodes.codes_set(h, key, value)
        d, c, n, s1, v1, s2, v2 = sig
        eccodes.codes_set(h, "discipline", d)
        eccodes.codes_set(h, "parameterCategory", c)
        eccodes.codes_set(h, "parameterNumber", n)
        for prefix, kind, value in (("First", s1, v1), ("Second", s2, v2)):
            eccodes.codes_set(h, f"typeOf{prefix}FixedSurface", kind)
            if value is None:
                eccodes.codes_set_missing(h, f"scaleFactorOf{prefix}FixedSurface")
                eccodes.codes_set_missing(h, f"scaledValueOf{prefix}FixedSurface")
            else:
                eccodes.codes_set(h, f"scaleFactorOf{prefix}FixedSurface", 0)
                eccodes.codes_set(h, f"scaledValueOf{prefix}FixedSurface", int(value))
        eccodes.codes_set(h, "dataDate", int(f"{run:%Y%m%d}"))
        eccodes.codes_set(h, "dataTime", run.hour * 100)
        eccodes.codes_set(h, "stepUnits", 1)
        if avg_from is not None:
            eccodes.codes_set(h, "typeOfStatisticalProcessing", 0)
            eccodes.codes_set(h, "stepRange", f"{avg_from}-{lead}")
        else:
            eccodes.codes_set(h, "forecastTime", lead)
        eccodes.codes_set(h, "packingType", "grid_ieee")
        flat = np.asarray(values, dtype="float64").ravel()
        if bitmap:
            eccodes.codes_set(h, "bitmapPresent", 1)
            eccodes.codes_set(h, "missingValue", 9999.0)
            flat = np.where(np.isnan(flat), 9999.0, flat)
        eccodes.codes_set_values(h, flat)
        return eccodes.codes_get_message(h)
    finally:
        eccodes.codes_release(h)


TMP850 = [0, 0, 0, 100, 85000, 255, None]
PRATE = [0, 1, 7, 1, 0, 255, None]


def spec_of(sig, name="t", factor=1.0) -> gfs.FieldSpec:
    return gfs.FieldSpec(name=name, var="X", level="y", signature=gfs.GribSignature.parse(sig),
                         encoding=Encoding(np.dtype("uint16"), 0.01, 0.0, 65534, 65535),
                         factor=factor)


RUN = datetime(2024, 6, 1, 6, tzinfo=UTC)


def test_check_message_accepts_the_right_message_and_decodes_it():
    v = np.linspace(200, 300, SRC.ni * SRC.nj).reshape(SRC.nj, SRC.ni)
    msg = grib_message(v, run=RUN, lead=3, sig=TMP850)
    check, values, window = gfs.check_message(msg, spec_of(TMP850), run=RUN, lead=3,
                                              average=False, grid=SRC)
    assert check.ok, check.errors
    np.testing.assert_allclose(values, v.astype("float32"))
    assert window == (3, 3) and check.fingerprint


@pytest.mark.parametrize("change, fragment", [
    ({"sig": [0, 0, 0, 100, 50000, 255, None]}, "parameter/level"),
    ({"lead": 4}, "step ends at"),
    ({"run": RUN + timedelta(hours=6)}, "run "),
    ({"grid": gfs.SourceGrid(ni=180, nj=91, lon_first=1.0, dlat=2.0, dlon=2.0)}, "lon_first"),
    ({"grid": gfs.SourceGrid(ni=90, nj=46, dlat=4.0, dlon=4.0)}, "grid regular_ll 90x46"),
    ({"avg_from": 0}, "stepType"),
])
def test_check_message_rejects_every_kind_of_mismatch(change, fragment):
    kw = {"run": RUN, "lead": 3, "sig": TMP850, **change}
    grid = kw.pop("grid", SRC)
    if grid is not SRC:
        v = np.zeros((grid.nj, grid.ni))
        msg = grib_message(v, grid=grid, **kw)
    else:
        msg = grib_message(np.zeros((SRC.nj, SRC.ni)), **kw)
    check, values, _ = gfs.check_message(msg, spec_of(TMP850), run=RUN, lead=3, average=False,
                                         grid=SRC)
    assert values is None and any(fragment in e for e in check.errors), check.errors


def test_check_message_reads_a_time_mean_and_a_bitmap():
    v = np.full((SRC.nj, SRC.ni), 1e-4)
    v[:10] = np.nan
    msg = grib_message(v, run=RUN, lead=8, sig=PRATE, avg_from=6, bitmap=True)
    check, values, window = gfs.check_message(msg, spec_of(PRATE), run=RUN, lead=8,
                                              average=True, grid=SRC)
    assert check.ok, check.errors
    assert window == (6, 8) and np.isnan(values[:10]).all() and np.nanmax(values) > 0
    assert gfs.check_message(b"GRIB nonsense", spec_of(PRATE), run=RUN, lead=8, average=True,
                             grid=SRC)[1] is None


# ---------------------------------------------------------- the stream

DAY = datetime(2024, 6, 2, tzinfo=UTC)
FIELDS = ["tmp_850hpa", "prate_sfc", "crain_sfc", "snod_sfc", "clmr_500hpa"]


def tiny_cfgs():
    state = load_product_config("noaa_gfs", "state_1h_16km")
    prate = load_product_config("noaa_gfs", "prate_16km")
    extra = json.loads(json.dumps(state.extra))
    extra["gfs"]["source_grid"] = {"ni": SRC.ni, "nj": SRC.nj, "lat_first": 90.0,
                                   "lon_first": 0.0, "dlat": 2.0, "dlon": 2.0}
    extra["gfs"]["fields"] = {k: v for k, v in extra["gfs"]["fields"].items() if k in FIELDS}
    store = state.store.model_copy(update={"compression_level": 1})
    state = state.model_copy(update={
        "variables": [v for v in state.variables if v.name in FIELDS], "extra": extra,
        "store": store})
    return state, prate.model_copy(update={"store": prate.store.model_copy(
        update={"compression_level": 1})})


def field_values(name: str, run: datetime, lead: int) -> np.ndarray:
    """A deterministic native field per (field, run, lead)."""
    rng = np.random.default_rng(zlib.crc32(f"{name} {run:%Y%m%d%H} {lead}".encode()))
    base = {"tmp_850hpa": 280.0, "snod_sfc": 0.2, "clmr_500hpa": 1e-4}.get(name, 0.0)
    if name == "crain_sfc":
        return (rng.random((SRC.nj, SRC.ni)) < 0.3).astype("float64")
    if name == "prate_sfc":
        return rng.gamma(0.5, 2e-4, (SRC.nj, SRC.ni))
    v = base * (1 + 0.1 * rng.standard_normal((SRC.nj, SRC.ni)))
    if name == "snod_sfc":
        v[: SRC.nj // 2] = np.nan            # a bitmap
    return v


def hourly_prate(run: datetime, lead: int) -> np.ndarray:
    return field_values("prate_sfc", run, 100 + lead)


class FakeBucket:
    """GFS files with their .idx, built in memory; serves get_text / get_bytes."""

    def __init__(self, cfg, runs, *, absent=(), broken=(), no_clmr_before=None):
        self.objects: dict[str, bytes] = {}
        self.requests_made = 0
        self.stats = {"bytes": 0}
        self.specs = {s.name: s for s in gfs.field_specs(cfg)}
        self.broken = set(broken)
        self.no_clmr_before = no_clmr_before
        for run in runs:
            for lead in range(16):
                if (run, lead) not in absent:
                    self._file(run, lead)

    def _file(self, run: datetime, lead: int) -> None:
        parts: list[bytes] = []
        lines: list[str] = []

        def add(var, level, step, msg):
            off = sum(len(p) for p in parts)
            lines.append(f"{len(lines) + 1}:{off}:d={run:%Y%m%d%H}:{var}:{level}:{step}:")
            parts.append(msg)

        step = "anl" if lead == 0 else f"{lead} hour fcst"
        for name, s in self.specs.items():
            if name == "clmr_500hpa" and self.no_clmr_before and run < self.no_clmr_before:
                continue
            sig = s.signature.as_list()
            if (run, lead, name) in self.broken:
                sig = [0, 0, 0, 100, 70000, 255, None]
            values = hourly_prate(run, lead) / 3600 if name == "prate_sfc" and lead \
                else field_values(name, run, lead)
            add(s.var, s.level, step, grib_message(values, run=run, lead=lead, sig=sig,
                                                   bitmap=name == "snod_sfc"))
        if lead:
            start = (lead - 1) // 6 * 6
            mean = np.mean([hourly_prate(run, k) for k in range(start + 1, lead + 1)],
                           axis=0) / 3600
            add("PRATE", "surface", f"{start}-{lead} hour ave fcst",
                grib_message(mean, run=run, lead=lead, sig=PRATE, avg_from=start))
        key = gfs.file_key(run, lead)
        self.objects[key] = b"".join(parts)
        self.objects[key + ".idx"] = ("\n".join(lines) + "\n").encode()

    def get_text(self, key):
        self.requests_made += 1
        blob = self.objects.get(key)
        return None if blob is None else blob.decode()

    def get_bytes(self, key, start=None, end=None):
        self.requests_made += 1
        if key not in self.objects:
            raise http.NotFoundError(key, "404 not found", 404)
        blob = self.objects[key][start:None if end is None else end + 1]
        self.stats["bytes"] += len(blob)
        return blob


def run_stream(state, prate, root, client, **kw):
    lines: list[str] = []
    code = stream_gfs.run(state, prate, root, DAY, DAY + timedelta(days=1), client=client,
                          download_workers=4, log=lines.append, **kw)
    return code, lines


@pytest.fixture
def bucket():
    state, _ = tiny_cfgs()
    runs = [DAY - timedelta(hours=6)] + gfs.run_times(DAY)
    # 06z f002 is absent upstream -> hour 08 from 00z f008; 12z f004 has a bad message.
    return FakeBucket(state, runs, absent={(DAY + timedelta(hours=6), 2)},
                      broken={(DAY + timedelta(hours=12), 4, "tmp_850hpa")},
                      no_clmr_before=DAY + timedelta(hours=18))


def test_a_day_end_to_end(tmp_path: Path, bucket):
    state, prate = tiny_cfgs()
    root = tmp_path / "root"
    code, lines = run_stream(state, prate, root, bucket)
    assert code == stream_kdp.EXIT_DONE, lines
    status = json.loads((gridded.store_path(root, state.store_id) / "days" / "2024"
                         / "2024-06-02.json").read_text())
    assert status["state"] == "done" and status["state_frames_packed"] == 24
    assert status["hours"]["08"] == {
        "run": "2024060200", "lead": 8, "fallback": True, "state": "packed",
        "fields_absent_upstream": ["clmr_500hpa"], "fields_failed_check": []}
    assert status["hours"]["16"]["fields_failed_check"] == ["tmp_850hpa"]
    assert len(status["schema_failed"]) == 1
    # f002 absent: hour 1-2 needs it, and so does 2-3 (3 m(0,3) - 2 m(0,2)); 3-4 does not.
    assert status["prate_runs"]["2024060206"]["leads_missing"] == [2, 3]

    ds = gridded.read(state.store_id, root)
    day = ds.sel(time=slice("2024-06-02T00", "2024-06-02T23"))
    assert int(day.is_fallback.sum()) == 1 and int(day.source_lead_h[8]) == 8
    assert int(day.source_run_time[13]) == int((DAY + timedelta(hours=12)).timestamp())
    assert bool(day.is_corrupt_frame[16]) and int(day.n_fields_missing[16]) == 2
    assert int(day.n_fields_missing[20]) == 0

    # Packed codes are the encoded regrid of the message values.
    g = grids.get(state.grid_id)
    lon, lat = g.lonlat()
    rg = gfs.Regridder(lon, lat, SRC)
    specs = {s.name: s for s in gfs.field_specs(state)}
    for name, hour in (("tmp_850hpa", 13), ("snod_sfc", 3), ("crain_sfc", 20)):
        run = gfs.floor_run(DAY + timedelta(hours=hour))
        lead = hour % 6
        native = field_values(name, run, lead).astype("float32").astype("float64")  # IEEE32
        want = gfs.encode_field(rg(native, specs[name].regrid),
                                specs[name].encoding).codes
        np.testing.assert_array_equal(_raw(root, state, name, hour), want)
    assert (_raw(root, state, "snod_sfc", 3) == 65535).any()          # bitmap -> undefined
    assert (_raw(root, state, "clmr_500hpa", 2) == 65534).all()       # absent upstream
    assert (_raw(root, state, "tmp_850hpa", 16) == 65534).all()       # failed its check
    quarantine = list((root / "raw" / "nwp" / "noaa_gfs" / "state_1h_16km" / "quarantine")
                      .glob("*.json"))
    assert len(quarantine) == 1

    # The precipitation store: one-hour means de-averaged from the (float32) time means.
    import zarr

    run = DAY + timedelta(hours=12)
    grp = zarr.open_group(gridded.zarr_path(root, prate.store_id), mode="r")
    row = int((run - datetime(2021, 1, 1, tzinfo=UTC)) / timedelta(hours=6))
    enc = Encoding(np.dtype("uint16"), 0.002, 0.0, 65534, 65535)
    for lead in (1, 6, 7, 11, 15):
        start = (lead - 1) // 6 * 6
        want = hourly_prate(run, lead)            # the truth the means were made from
        codes = gfs.encode_field(np.maximum(rg(want), 0.0), enc).codes
        got = np.asarray(grp[f"prate_f{lead:02d}"][row])
        # float32 means x up to 6: a code or two of rounding, never more.
        assert np.abs(got.astype(int) - codes.astype(int)).max() <= 2, (lead, start)
    assert int(np.asarray(grp["n_leads_missing"][row])) == 0


def _raw(root, cfg, name, hour):
    import zarr

    grp = zarr.open_group(gridded.zarr_path(root, cfg.store_id), mode="r")
    row = int((DAY + timedelta(hours=hour) - datetime(2021, 1, 1, tzinfo=UTC))
              / timedelta(hours=1))
    return np.asarray(grp[name][row])


def test_a_done_day_is_skipped_and_redo_rewrites_the_same_codes(tmp_path: Path, bucket):
    state, prate = tiny_cfgs()
    root = tmp_path / "root"
    assert run_stream(state, prate, root, bucket)[0] == stream_kdp.EXIT_DONE
    before = _raw(root, state, "tmp_850hpa", 5).copy()
    n = bucket.requests_made
    code, lines = run_stream(state, prate, root, bucket)
    assert code == stream_kdp.EXIT_DONE and bucket.requests_made == n
    assert any("0 to do" in line for line in lines)
    assert run_stream(state, prate, root, bucket, redo=True)[0] == stream_kdp.EXIT_DONE
    np.testing.assert_array_equal(_raw(root, state, "tmp_850hpa", 5), before)


def test_a_transfer_failure_leaves_the_day_partial_and_a_rerun_completes_it(tmp_path: Path,
                                                                            bucket):
    state, prate = tiny_cfgs()
    root = tmp_path / "root"
    real = bucket.get_bytes
    target = gfs.file_key(DAY + timedelta(hours=18), 2)

    def flaky(key, start=None, end=None):
        if key == target:
            raise http.FetchError(key, "HTTP 503", 503)
        return real(key, start, end)

    bucket.get_bytes = flaky
    assert run_stream(state, prate, root, bucket)[0] == stream_kdp.EXIT_DONE
    path = gridded.store_path(root, state.store_id) / "days" / "2024" / "2024-06-02.json"
    status = json.loads(path.read_text())
    assert status["state"] == "partial" and status["hours"]["20"]["state"] == "transfer_failed"
    assert status["prate_runs"]["2024060218"]["state"] == "transfer_failed"
    bucket.get_bytes = real
    assert run_stream(state, prate, root, bucket)[0] == stream_kdp.EXIT_DONE
    assert json.loads(path.read_text())["state"] == "done"


def test_the_real_configs_hold_the_122_fields():
    state = load_product_config("noaa_gfs", "state_1h_16km")
    specs = gfs.field_specs(state)
    assert len(specs) == 122 and len({s.key for s in specs}) == 122
    assert {s.name for s in specs if s.regrid == "nearest"} == {
        "crain_sfc", "csnow_sfc", "cfrzr_sfc", "cicep_sfc", "cpofp_sfc"}
    prate = load_product_config("noaa_gfs", "prate_16km")
    assert [v.name for v in prate.variables] == [f"prate_f{k:02d}" for k in range(1, 16)]
    assert state.category == "nwp"
    assert str(gridded.store_path(Path("/r"), state.store_id)) == \
        "/r/packed/nwp/noaa_gfs/state_1h_16km/v1"
