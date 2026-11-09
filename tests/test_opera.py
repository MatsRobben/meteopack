"""OPERA rain-rate composite: names, chains, schema check, window read, encoding."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest

from meteopack.core import grids
from meteopack.core.encoding import Encoding
from meteopack.core.grids import Grid
from meteopack.core.s3 import RemoteFile
from meteopack.jobs import stream_kdp
from meteopack.jobs.plan import load_product_config
from meteopack.sources.radar import opera

#: A tiny native grid (rows, cols) and the store's window on it.
NATIVE_Y, NATIVE_X = 30, 26
ROW0, COL0, NY, NX = 5, 4, 12, 10
PROJ = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
TINY = Grid(id="tiny_opera", proj4=PROJ, shape=(NY, NX),
            extent=((COL0 * 2000.0, (COL0 + NX) * 2000.0),
                    (-(ROW0 + NY) * 2000.0, -ROW0 * 2000.0)))
UL = (-39.5357864125034, 67.02283275830867)
LR = (29.421038635578032, 31.98765027794496)
NODATA, UNDETECT = -9999000.0, -8888000.0
CUTOVER = datetime(2024, 7, 5, 6, 0, tzinfo=UTC)
NODES_NL = ["nldhl", "nlhrw", "behel", "deess"]


@pytest.fixture(autouse=True)
def _tiny_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grids, "get", lambda grid_id, path=None: TINY)


def tiny_cfg():
    """The real product config, shrunk to a 30 x 26 native grid and a 12 x 10 window."""
    cfg = load_product_config("opera", "rate_15min")
    extra = dict(cfg.extra)
    spec = dict(extra["opera"])
    spec["native_grid"] = {**spec["native_grid"], "xsize": NATIVE_X, "ysize": NATIVE_Y}
    spec["window"] = {"row0": ROW0, "col0": COL0, "ny": NY, "nx": NX}
    extra["opera"] = spec
    store = cfg.store.model_copy(update={
        "chunks": {"time": 1, "y": NY, "x": NX},
        "shard": {"time": 96, "y": NY, "x": NX},
        "compression_level": 1,
    })
    return cfg.model_copy(update={"grid_id": TINY.id, "extra": extra, "store": store})


def rates(seed: int = 0) -> np.ndarray:
    """A native (NATIVE_Y, NATIVE_X) float64 field with every kind of pixel."""
    rng = np.random.default_rng(seed)
    v = np.round(rng.gamma(0.6, 3.0, size=(NATIVE_Y, NATIVE_X)), 2)
    v[rng.random(v.shape) < 0.4] = UNDETECT
    v[:3, :] = NODATA                       # outside coverage, partly inside the window
    v[ROW0, COL0] = NODATA
    v[ROW0, COL0 + 1] = UNDETECT
    v[ROW0 + 1, COL0] = 0.004               # rounds to code 0
    v[ROW0 + 1, COL0 + 1] = 0.006           # rounds to code 1
    v[ROW0 + 2, COL0] = 812.5               # above the encodable maximum
    return v


def write_odim(directory: Path, when: datetime, chain: str, values: np.ndarray | None = None,
               *, nodes=NODES_NL, name: str | None = None, projdef: str = PROJ,
               gain: float = 1.0, nodata: float = NODATA, quantity: str = "RATE",
               what_time: datetime | None = None, shape=None, shuffle: bool = False,
               chunks=(7, 9)) -> Path:
    """A file laid out like the real ODYSSEY QIND_RATE.h5 or NIMBUS RATE.h5."""
    values = rates(int(when.timestamp()) // 900 % 1000) if values is None else values
    if shape is not None:
        values = np.zeros(shape)
    path = Path(directory) / (name or opera.filename_for(when, chain))
    t = what_time or when
    with h5py.File(path, "w") as f:
        f.attrs["Conventions"] = np.bytes_("ODIM_H5/V2_0" if chain == opera.ODYSSEY
                                           else "ODIM_H5/V2_4")
        w = f.create_group("what")
        w.attrs.update({"date": np.bytes_(f"{t:%Y%m%d}"), "time": np.bytes_(f"{t:%H%M%S}"),
                        "object": np.bytes_("COMP"), "source": np.bytes_("ORG:247"),
                        "version": np.bytes_(opera.VERSIONS[chain])})
        wh = f.create_group("where")
        wh.attrs.update({"projdef": np.bytes_(projdef), "xsize": NATIVE_X, "ysize": NATIVE_Y,
                         "xscale": 2000.0, "yscale": 2000.0, "UL_lon": UL[0], "UL_lat": UL[1],
                         "LR_lon": LR[0], "LR_lat": LR[1]})
        if chain == opera.ODYSSEY:
            text = ",".join(f"NOD:{n}" for n in nodes)
        else:
            text = ",".join(f"'{n}'" for n in nodes)
        f.create_group("how").attrs["nodes"] = np.bytes_(text)
        calib = {"gain": gain, "offset": 0.0, "nodata": nodata, "undetect": UNDETECT,
                 "quantity": np.bytes_(quantity)}
        kw = {"compression": "gzip", "compression_opts": 4, "chunks": chunks, "shuffle": shuffle}
        if chain == opera.ODYSSEY:
            start, end = when - timedelta(minutes=10), when + timedelta(minutes=5)
            times = {"startdate": np.bytes_(f"{start:%Y%m%d}"),
                     "starttime": np.bytes_(f"{start:%H%M%S}"),
                     "enddate": np.bytes_(f"{end:%Y%m%d}"),
                     "endtime": np.bytes_(f"{end:%H%M%S}")}
            d1 = f.create_group("dataset1")
            d1.create_group("what").attrs.update({**calib, **times, "product": np.bytes_("COMP")})
            d1.create_dataset("data1/data", data=values.astype("float64"), **kw)
            d2 = f.create_group("dataset2")
            d2.create_group("what").attrs.update({**calib, **times, "quantity": np.bytes_("QIND"),
                                                 "product": np.bytes_("COMP")})
            d2.create_dataset("data1/data", data=np.ones_like(values), **kw)
        else:
            times = {"startdate": np.bytes_(f"{when:%Y%m%d}"),
                     "starttime": np.bytes_(f"{when:%H%M%S}"),
                     "enddate": np.bytes_(f"{when:%Y%m%d}"),
                     "endtime": np.bytes_(f"{when:%H%M%S}")}
            d1 = f.create_group("dataset1")
            d1.create_group("what").attrs.update({
                **times, "product": np.bytes_("PPI"),
                "prodname": np.bytes_("OPERA NIMBUS instantaneous rain rate composite")})
            d1.create_dataset("data1/data", data=values.astype("float64"), **kw)
            d1.create_group("data1/what").attrs.update(calib)
            d1.create_dataset("data1/quality1/data", data=np.ones_like(values), **kw)
    return path


def expected_codes(native: np.ndarray) -> np.ndarray:
    """What the store must hold for a native field: the window, encoded."""
    v = native[ROW0:ROW0 + NY, COL0:COL0 + NX]
    out = np.clip(np.rint(np.where(v >= 0, v, 0) / 0.01), 0, 65533).astype("uint16")
    out[v == UNDETECT] = 0
    out[v == NODATA] = 65535
    return out


def schema():
    return opera.Schema.from_config(tiny_cfg())


# ------------------------------------------------------------------ names and chains


def test_filenames_and_the_cutover_rule():
    assert opera.parse_filename("OPERA@20240705T0615@0@RATE.h5") == (
        datetime(2024, 7, 5, 6, 15, tzinfo=UTC), "RATE")
    assert opera.parse_filename("OPERA@20240705T0615@0@RATE.tiff") is None
    assert opera.chain_for(CUTOVER, CUTOVER) == opera.ODYSSEY
    assert opera.chain_for(CUTOVER + timedelta(minutes=15), CUTOVER) == opera.NIMBUS
    assert opera.filename_for(CUTOVER, opera.ODYSSEY) == "OPERA@20240705T0600@0@QIND_RATE.h5"


def test_nodes_in_every_chains_spelling():
    assert opera.parse_nodes("NOD:frnan,NOD:nldhl") == ["frnan", "nldhl"]
    assert opera.parse_nodes(b"'behel','nlhrw'") == ["behel", "nlhrw"]
    assert opera.parse_nodes("behel, bewid, behel") == ["behel", "bewid"]
    assert opera.parse_nodes(None) == []


def test_select_takes_one_file_per_step_from_the_right_chain():
    cfg = tiny_cfg()
    fmt = opera.OperaFormat(cfg, stream_kdp.Settings.from_config(cfg))
    day = datetime(2024, 7, 5, tzinfo=UTC)

    def remote(name):
        return RemoteFile(name, 1, key=f"2024/07/05/OPERA/COMP/{name}")

    names = ["OPERA@20240705T0545@0@QIND_RATE.h5", "OPERA@20240705T0545@0@RATE.h5",
             "OPERA@20240705T0600@0@QIND_RATE.h5", "OPERA@20240705T0600@0@RATE.h5",
             "OPERA@20240705T0615@0@QIND_RATE.h5", "OPERA@20240705T0615@0@RATE.h5",
             "OPERA@20240705T0615@0@ACRR.h5", "OPERA@20240705T0615@0@RATE.tiff",
             "OPERA@20240706T0000@0@RATE.h5"]
    picked = fmt.select([remote(n) for n in names], day)
    assert {w.strftime("%H%M"): r.filename for w, r in picked.items()} == {
        "0545": "OPERA@20240705T0545@0@QIND_RATE.h5",
        "0600": "OPERA@20240705T0600@0@QIND_RATE.h5",
        "0615": "OPERA@20240705T0615@0@RATE.h5"}
    # A stray ODYSSEY file long after the cutover is ignored...
    stray = fmt.select([remote("OPERA@20241212T1200@0@QIND_RATE.h5")],
                       datetime(2024, 12, 12, tzinfo=UTC))
    assert stray == {}


def test_configured_fallback_days_take_odyssey_and_only_those():
    cfg = tiny_cfg()
    fmt = opera.OperaFormat(cfg, stream_kdp.Settings.from_config(cfg))
    days = sorted(fmt.schema.odyssey_fallback_days)
    assert [d.isoformat() for d in days] == [f"2024-12-0{k}" for k in range(3, 10)]

    def remote(name):
        return RemoteFile(name, 1, key=name)

    for day, want in ((datetime(2024, 12, 3, tzinfo=UTC), "QIND_RATE"),
                      (datetime(2024, 12, 9, tzinfo=UTC), "QIND_RATE"),
                      (datetime(2024, 12, 2, tzinfo=UTC), "RATE"),
                      (datetime(2024, 12, 10, tzinfo=UTC), "RATE")):
        names = [f"OPERA@{day:%Y%m%d}T2345@0@QIND_RATE.h5", f"OPERA@{day:%Y%m%d}T2345@0@RATE.h5"]
        picked = fmt.select([remote(n) for n in names], day)
        assert [r.filename.split("@")[-1] for r in picked.values()] == [f"{want}.h5"], day
        assert fmt.schema.chain_for(day + timedelta(hours=23, minutes=45)) == (
            opera.ODYSSEY if want == "QIND_RATE" else opera.NIMBUS)


def test_a_fallback_day_still_checks_every_file(tmp_path):
    s = schema()
    when = datetime(2024, 12, 5, 12, 0, tzinfo=UTC)
    good = write_odim(tmp_path, when, opera.ODYSSEY)
    nimbus = write_odim(tmp_path, when, opera.NIMBUS)
    (tmp_path / "bad").mkdir()
    bad = write_odim(tmp_path / "bad", when, opera.ODYSSEY, gain=0.5)
    with h5py.File(good) as f:
        assert opera.check_schema(f, good.name, s, s.chain_for(when)).ok
    with h5py.File(nimbus) as f:
        check = opera.check_schema(f, nimbus.name, s, s.chain_for(when))
        assert any("cutover assigns to ODYSSEY" in e for e in check.errors)
    with h5py.File(bad) as f:
        assert not opera.check_schema(f, bad.name, s, s.chain_for(when)).ok


# ------------------------------------------------------------------ schema


@pytest.mark.parametrize("chain", [opera.ODYSSEY, opera.NIMBUS])
def test_a_well_formed_file_passes_with_a_stable_per_chain_fingerprint(tmp_path, chain):
    when = datetime(2024, 3, 1, 12, 0, tzinfo=UTC) if chain == opera.ODYSSEY \
        else datetime(2025, 3, 1, 12, 0, tzinfo=UTC)
    a = write_odim(tmp_path, when, chain)
    b = write_odim(tmp_path, when + timedelta(minutes=15), chain, nodes=["nldhl"])
    with h5py.File(a) as fa, h5py.File(b) as fb:
        ca = opera.check_schema(fa, a.name, schema(), chain)
        cb = opera.check_schema(fb, b.name, schema(), chain)
    assert ca.ok and cb.ok, (ca.errors, cb.errors)
    assert ca.chain == chain and ca.fingerprint == cb.fingerprint
    assert ca.nodes == NODES_NL
    offsets = ca.descriptor["window_offsets_s"]
    assert offsets == ([-600, 300] if chain == opera.ODYSSEY else [0, 0])


def test_the_two_chains_have_different_fingerprints(tmp_path):
    when = datetime(2024, 7, 5, 6, 0, tzinfo=UTC)
    a = write_odim(tmp_path, when, opera.ODYSSEY)
    b = write_odim(tmp_path, when, opera.NIMBUS)
    with h5py.File(a) as fa, h5py.File(b) as fb:
        assert opera.check_schema(fa, a.name, schema()).fingerprint != \
            opera.check_schema(fb, b.name, schema()).fingerprint


WHEN = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"projdef": PROJ.replace("lat_0=55.0", "lat_0=52.0")}, "projdef"),
    ({"gain": 0.1}, "gain"),
    ({"nodata": -1.0}, "nodata"),
    ({"quantity": "DBZH"}, "quantity"),
    ({"what_time": WHEN + timedelta(minutes=15)}, "/what time"),
    ({"shape": (NATIVE_Y, NATIVE_X - 1)}, "shape"),
    ({"name": "OPERA@20240301T1207@0@QIND_RATE.h5"}, "cadence"),
    ({"nodes": []}, "/how/nodes"),
])
def test_schema_drift_is_caught_and_never_packed(tmp_path, kwargs, message):
    path = write_odim(tmp_path, WHEN, opera.ODYSSEY, **kwargs)
    s = schema()
    with h5py.File(path) as f:
        check = opera.check_schema(f, path.name, s, opera.ODYSSEY)
    assert not check.ok and any(message in e for e in check.errors), check.errors
    packed = opera.pack_file(path, s, Encoding.from_config(
        tiny_cfg().variables[0].encoding), opera.ODYSSEY)
    assert packed.codes is None


def test_a_file_whose_structure_contradicts_its_name_fails(tmp_path):
    # NIMBUS content under an ODYSSEY name.
    path = write_odim(tmp_path, WHEN, opera.NIMBUS,
                      name=opera.filename_for(WHEN, opera.ODYSSEY))
    with h5py.File(path) as f:
        check = opera.check_schema(f, path.name, schema())
    assert any("laid out as NIMBUS" in e for e in check.errors), check.errors


def test_a_file_of_the_wrong_chain_for_its_time_step_fails(tmp_path):
    late = CUTOVER + timedelta(minutes=15)
    path = write_odim(tmp_path, late, opera.ODYSSEY)
    with h5py.File(path) as f:
        check = opera.check_schema(f, path.name, schema(), opera.chain_for(late, CUTOVER))
    assert any("cutover assigns to NIMBUS" in e for e in check.errors), check.errors


def test_an_unreadable_file_is_a_failed_check_not_an_exception(tmp_path):
    path = tmp_path / opera.filename_for(WHEN, opera.ODYSSEY)
    path.write_bytes(b"\x89HDF\r\n\x1a\nnot really")
    packed = opera.pack_file(path, schema(), Encoding.from_config(
        tiny_cfg().variables[0].encoding))
    assert packed.codes is None and "unreadable" in packed.check.errors[0]
    assert packed.check.time == WHEN


# ------------------------------------------------------------------ reading and encoding


@pytest.mark.parametrize("chunks", [(7, 9), (30, 26), (4, 5)])
def test_the_fast_window_read_equals_h5py(tmp_path, chunks):
    values = rates(3)
    path = write_odim(tmp_path, WHEN, opera.ODYSSEY, values, chunks=chunks)
    with h5py.File(path) as f:
        d = f["dataset1/data1/data"]
        for r0, r1, c0, c1 in ((ROW0, ROW0 + NY, COL0, COL0 + NX), (0, 30, 0, 26),
                               (29, 30, 25, 26), (6, 7, 0, 26)):
            np.testing.assert_array_equal(opera.read_window(d, r0, r1, c0, c1),
                                          values[r0:r1, c0:c1])


def test_a_shuffled_file_falls_back_to_h5py(tmp_path):
    values = rates(4)
    path = write_odim(tmp_path, WHEN, opera.ODYSSEY, values, shuffle=True)
    with h5py.File(path) as f:
        got = opera.read_window(f["dataset1/data1/data"], 1, 9, 2, 20)
    np.testing.assert_array_equal(got, values[1:9, 2:20])


def test_encoding_keeps_no_rain_and_no_coverage_apart_and_counts_oddities():
    target = Encoding.from_config(tiny_cfg().variables[0].encoding)
    v = np.array([[NODATA, UNDETECT, 0.0, 0.004, 0.006, 1.234],
                  [655.33, 700.0, -0.5, np.nan, 12.345, 0.01]])
    codes, stats = opera.encode_rate(v, target, nodata=NODATA, undetect=UNDETECT)
    assert codes.dtype == np.uint16
    assert codes.tolist() == [[65535, 0, 0, 0, 1, 123], [65533, 65533, 0, 65535, 1234, 1]]
    assert (stats.n_nodata, stats.n_undetect, stats.n_nan) == (2, 1, 1)
    assert (stats.n_negative, stats.n_clipped, stats.max_rate) == (1, 1, 700.0)
    decoded = target.decode(codes)
    assert np.isnan(decoded[0, 0]) and decoded[0, 1] == 0.0


@pytest.mark.parametrize("chain", [opera.ODYSSEY, opera.NIMBUS])
def test_pack_file_crops_the_window_and_encodes(tmp_path, chain):
    values = rates(5)
    path = write_odim(tmp_path, WHEN, chain, values)
    packed = opera.pack_file(path, schema(), Encoding.from_config(
        tiny_cfg().variables[0].encoding), chain)
    assert packed.check.ok, packed.check.errors
    np.testing.assert_array_equal(packed.codes, expected_codes(values))
    assert packed.stats.n_clipped == 1
    # Decoded values equal the file's to within half a quantisation step.
    v = values[ROW0:ROW0 + NY, COL0:COL0 + NX]
    rain = (v >= 0) & (v <= 655.33)
    assert np.abs(packed.codes[rain] * 0.01 - v[rain]).max() <= 0.005 + 1e-9


# ------------------------------------------------------------------ the real config


def test_the_product_config_window_is_the_grid_and_frames_the_training_square(monkeypatch):
    monkeypatch.undo()                     # the real grid registry
    cfg = load_product_config("opera", "rate_15min")
    s = opera.Schema.from_config(cfg)      # raises if window and grid disagree
    assert s.window == {"row0": 756, "row1": 1636, "col0": 389, "col1": 1269}
    assert s.cutover == CUTOVER
    grid = grids.get(cfg.grid_id)
    assert grid.grid_mapping_name == "lambert_azimuthal_equal_area"
    # The 1536 km square's corners (KNMI stereographic) fall inside the window
    # with at least 16 px to spare.
    from pyproj import Transformer

    stere = "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 +b=6356750 +units=m"
    cx, cy = Transformer.from_crs("EPSG:4326", stere, always_xy=True).transform(5.3, 52.2)
    to_laea = Transformer.from_crs(stere, grid.proj4, always_xy=True)
    for dx in (-768e3, 768e3):
        for dy in (-768e3, 768e3):
            x, y = to_laea.transform(cx + dx, cy + dy)
            col, row = x / 2000.0, -y / 2000.0
            assert s.col0 + 16 <= col <= s.col0 + s.nx - 16
            assert s.row0 + 16 <= row <= s.row0 + s.ny - 16
    lon, lat = grid.lonlat()
    assert lon.shape == (880, 880)
    # 52.2N 5.3E is native col 814.4, row 1200.5 (the exploration's number).
    i, j = np.unravel_index(np.argmin((lon - 5.3) ** 2 + (lat - 52.2) ** 2), lon.shape)
    assert (int(i) + s.row0, int(j) + s.col0) == (1200, 814)


def test_a_window_that_disagrees_with_the_grid_is_refused():
    cfg = tiny_cfg()
    extra = dict(cfg.extra)
    extra["opera"] = {**extra["opera"], "window": {"row0": ROW0 + 1, "col0": COL0,
                                                   "ny": NY, "nx": NX}}
    with pytest.raises(ValueError, match="extent"):
        opera.Schema.from_config(cfg.model_copy(update={"extra": extra}))
