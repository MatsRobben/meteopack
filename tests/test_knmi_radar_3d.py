"""KNMI 3D radar composite: schema check, fast read, encoding, level selection, flip."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import h5py
import numpy as np
import pytest

from meteopack.core import grids
from meteopack.core.encoding import Encoding
from meteopack.core.grids import Grid
from meteopack.jobs.plan import load_product_config
from meteopack.sources.radar import knmi_radar_3d as r3d

NY, NX, NZ = 8, 6, 8
#: South-up row centres, like the file; the tiny grid's coordinates are north-up.
TINY = Grid(
    id="tiny_3d",
    proj4="+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 +b=6356750 +units=km +no_defs",
    shape=(NY, NX),
    corners_lonlat=((3.0, 52.0), (3.0, 53.0), (5.0, 53.0), (5.0, 52.0)),
)
WHEN = datetime(2024, 7, 9, 16, 30, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _tiny_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grids, "get", lambda grid_id, path=None: TINY)


def tiny_cfg():
    """The real product config, shrunk to the tiny grid and 8 native levels (keep 4)."""
    cfg = load_product_config("knmi_radar", "composite3d_5min")
    levels = cfg.levels.model_copy(update={
        "values": [0.375 + 0.5 * k for k in range(NZ // 2)],
        "source_indices": list(range(1, NZ, 2)),
    })
    extra = dict(cfg.extra)
    extra["source_schema"] = {**extra["source_schema"], "n_levels": NZ}
    store = cfg.store.model_copy(update={
        "chunks": {"time": 1, "z": NZ // 2, "y": NY, "x": NX},
        "shard": {"time": 288, "z": NZ // 2, "y": NY, "x": NX},
        "compression_level": 1,
    })
    return cfg.model_copy(update={"grid_id": TINY.id, "levels": levels, "extra": extra,
                                  "store": store, "schema_fingerprint": None})


def native_codes(seed: int = 0) -> np.ndarray:
    """(NZ, NY, NX) uint16 codes, south-up, every level and row distinguishable."""
    rng = np.random.default_rng(seed)
    codes = rng.integers(1, 65535, size=(NZ, NY, NX), dtype=np.uint16)
    codes[:, 0, 0] = 0          # no echo
    codes[:, 0, 1] = 65535      # not observed
    codes[:, 1, :] = 0
    return codes


def write_file(directory: Path, when: datetime = WHEN, *, codes: np.ndarray | None = None,
               scale: float = 0.0015, offset: float = -30.00001, z=None, y=None,
               time_value: float | None = None, fill: int = 65535, shuffle: bool = True,
               proj4: str | None = None, drop: str | None = None,
               chunks: tuple[int, ...] = (1, 1, NY, NX)) -> Path:
    """A NetCDF4 file shaped like the real 3D_RADAR_COMPOSITE_*.nc, on the tiny grid."""
    import h5netcdf

    path = directory / r3d.filename_for(when)
    codes = native_codes() if codes is None else codes
    y_north, x = TINY.coordinates()
    with h5netcdf.File(path, "w") as f:
        f.dimensions = {"time": 1, "z": NZ, "y": NY, "x": NX}
        t = f.create_variable("time", ("time",), "f8")
        t[:] = [when.timestamp() if time_value is None else time_value]
        t.attrs["units"] = "seconds since 1970-01-01 00:00:00"
        zv = f.create_variable("z", ("z",), "f4")
        zv[:] = 0.125 + 0.25 * np.arange(NZ) if z is None else z
        yv = f.create_variable("y", ("y",), "f4")
        yv[:] = np.asarray(y_north)[::-1] if y is None else y
        xv = f.create_variable("x", ("x",), "f4")
        xv[:] = x
        if drop != "projection":
            p = f.create_variable("projection", (), "f8")
            p.attrs["proj4_params"] = proj4 or ("+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 "
                                                "+a=6378140 +b=6356750 +x_0=0 +y_0=0 +units=km")
            p.attrs["semi_major_axis"] = 6378140.0
            p.attrs["semi_minor_axis"] = 6356750.0
        r = f.create_variable("reflectivity", ("time", "z", "y", "x"), "u2",
                              chunks=chunks, compression="gzip",
                              compression_opts=4, shuffle=shuffle, fillvalue=np.uint16(fill))
        r[0] = codes
        r.attrs["units"] = "dBZ"
        r.attrs["add_offset"] = np.float32(offset)
        r.attrs["scale_factor"] = np.float64(scale)
    return path


def expected_codes(native: np.ndarray, scale: float, offset: float) -> np.ndarray:
    dbz = native.astype("float64") * scale + np.float32(offset)
    out = np.clip(np.rint((dbz + 32.0) / 0.5), 0, 253).astype("uint8")
    out[native == 0] = 0
    out[native == 65535] = 254
    return out


# ------------------------------------------------------------------ names


def test_filenames_round_trip_and_reject_other_products():
    assert r3d.filename_for(WHEN) == "3D_RADAR_COMPOSITE_202407091630.nc"
    assert r3d.filename_time("3D_RADAR_COMPOSITE_202407091630.nc") == WHEN
    assert r3d.filename_time("ECHO_TOP_HEIGHT_RADAR_DERIVED_PRODUCTS_202407091630.nc") is None
    slots = r3d.day_slots(WHEN, timedelta(minutes=5))
    assert len(slots) == 288 and slots[0].hour == 0 and slots[-1] == WHEN.replace(
        hour=23, minute=55)


# ------------------------------------------------------------------ reading


def test_fast_chunk_read_equals_h5py(tmp_path: Path):
    path = write_file(tmp_path)
    with h5py.File(path, "r") as h5:
        fast = r3d.read_levels(h5, [1, 3, 5, 7])
        slow = np.stack([h5["reflectivity"][0, k] for k in (1, 3, 5, 7)])
    np.testing.assert_array_equal(fast, slow)
    np.testing.assert_array_equal(fast, native_codes()[[1, 3, 5, 7]])


@pytest.mark.parametrize("chunks", [(1, 4, 3, 4), (1, NZ, NY, NX), (1, 3, 5, 5)])
def test_fast_read_handles_multi_level_and_edge_chunks(tmp_path: Path, chunks):
    # Up to mid-2024 the files hold 32 levels per (383, 350) chunk; 765 and
    # 700 are not multiples of it, so edge chunks are padded.
    path = write_file(tmp_path, chunks=chunks)
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(r3d.read_levels(h5, [1, 3, 5, 7]),
                                      native_codes()[[1, 3, 5, 7]])
        check = r3d.check_schema(h5, path.name, r3d.Schema.from_config(tiny_cfg()))
    assert check.ok and check.storage.startswith(f"chunks={'x'.join(map(str, chunks))}")


def test_unshuffled_files_fall_back_to_h5py(tmp_path: Path):
    path = write_file(tmp_path, shuffle=False)
    with h5py.File(path, "r") as h5:
        np.testing.assert_array_equal(r3d.read_levels(h5, [0, 7]), native_codes()[[0, 7]])


# ------------------------------------------------------------------ encoding


def test_encoding_matches_the_2d_product_calibration_and_keeps_sentinels():
    target = Encoding(dtype=np.dtype("uint8"), scale=0.5, offset=-32.0, missing=255,
                      out_of_image=254)
    native = native_codes()
    out = r3d.encode_levels(native, scale=0.0015, offset=-30.00001, target=target,
                            no_echo=-32.0)
    want = expected_codes(native, 0.0015, -30.00001)[:, ::-1, :]
    np.testing.assert_array_equal(out, want)
    flipped = out[:, ::-1, :]
    assert (flipped[native == 0] == 0).all()          # no echo -> 0, refl_5min's no echo
    assert (flipped[native == 65535] == 254).all()    # not observed
    assert not np.isin(flipped[(native != 0) & (native != 65535)], [0, 1, 2, 3, 254, 255]).any()
    # Decoded, every echo is within half a step of the native value.
    echo = (native != 0) & (native != 65535)
    native_dbz = native[echo] * 0.0015 + np.float32(-30.00001)
    assert np.abs(flipped[echo] * 0.5 - 32.0 - native_dbz).max() <= 0.25 + 1e-9


def test_a_per_file_scale_is_honoured():
    target = Encoding(dtype=np.dtype("uint8"), scale=0.5, offset=-32.0, missing=255,
                      out_of_image=254)
    native = np.full((1, 2, 2), 40000, dtype=np.uint16)
    a = r3d.encode_levels(native, scale=0.0012, offset=-30.0, target=target, no_echo=-32.0)
    b = r3d.encode_levels(native, scale=0.0017, offset=-30.0, target=target, no_echo=-32.0)
    assert int(a[0, 0, 0]) == round((40000 * 0.0012 - 30 + 32) / 0.5)
    assert int(b[0, 0, 0]) == round((40000 * 0.0017 - 30 + 32) / 0.5)


def test_pack_file_selects_every_second_level_and_flips_north_up(tmp_path: Path):
    cfg = tiny_cfg()
    native = native_codes(3)
    path = write_file(tmp_path, codes=native)
    packed = r3d.pack_file(path, cfg, r3d.Schema.from_config(cfg), r3d.target_encoding(cfg))
    assert packed.check.ok, packed.check.errors
    want = expected_codes(native[1::2], 0.0015, -30.00001)[:, ::-1, :]
    np.testing.assert_array_equal(packed.codes, want)
    assert packed.codes.shape == (NZ // 2, NY, NX)
    # Row 0 of the packed frame is the northernmost row, the file's last.
    np.testing.assert_array_equal(packed.codes[0, 0],
                                  expected_codes(native[1:2, -1:], 0.0015, -30.00001)[0, 0])


# ------------------------------------------------------------------ schema


def _check(path: Path, cfg=None):
    cfg = cfg or tiny_cfg()
    with h5py.File(path, "r") as h5:
        return r3d.check_schema(h5, path.name, r3d.Schema.from_config(cfg))


def test_a_well_formed_file_passes_and_has_a_stable_fingerprint(tmp_path: Path):
    a = _check(write_file(tmp_path))
    b = _check(write_file(tmp_path, WHEN + timedelta(minutes=5), scale=0.0012))
    assert a.ok and b.ok, (a.errors, b.errors)
    # The per-file scale and the time are not part of the structure.
    assert a.fingerprint == b.fingerprint
    assert a.scale_factor == pytest.approx(0.0015)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"z": 0.25 * np.arange(NZ)}, "z levels"),
    ({"y": np.asarray(TINY.coordinates()[0])}, "y differs"),     # north-up in the file
    ({"time_value": WHEN.timestamp() + 300}, "differs from the filename"),
    ({"scale": 0.01}, "scale_factor"),
    ({"offset": -32.0}, "add_offset"),
    ({"fill": 0}, "_FillValue"),
    ({"proj4": "+proj=stere +lat_0=90"}, "proj4_params"),
    ({"drop": "projection"}, "missing variable"),
])
def test_schema_drift_is_caught_and_never_packed(tmp_path: Path, kwargs, message):
    cfg = tiny_cfg()
    path = write_file(tmp_path, **kwargs)
    check = _check(path, cfg)
    assert not check.ok
    assert any(message in e for e in check.errors), check.errors
    packed = r3d.pack_file(path, cfg, r3d.Schema.from_config(cfg), r3d.target_encoding(cfg))
    assert packed.codes is None


def test_a_changed_structure_changes_the_fingerprint(tmp_path: Path):
    good = _check(write_file(tmp_path))
    rechunked_dir = tmp_path / "rechunked"
    rechunked_dir.mkdir()
    rechunked = _check(write_file(rechunked_dir, chunks=(1, 4, 4, 3)))
    assert rechunked.fingerprint == good.fingerprint     # storage is not content
    assert rechunked.storage != good.storage
    other = tmp_path / "other"
    other.mkdir()
    drifted = _check(write_file(other, z=0.25 * np.arange(NZ)))
    assert good.fingerprint != drifted.fingerprint


def test_an_unreadable_file_is_a_failed_check_not_an_exception(tmp_path: Path):
    cfg = tiny_cfg()
    path = tmp_path / r3d.filename_for(WHEN)
    path.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\0" * 64)
    packed = r3d.pack_file(path, cfg, r3d.Schema.from_config(cfg), r3d.target_encoding(cfg))
    assert packed.codes is None and "unreadable" in packed.check.errors[0]


def test_the_product_config_keeps_32_levels_at_500_m():
    cfg = load_product_config("knmi_radar", "composite3d_5min")
    assert len(cfg.levels.values) == 32
    assert cfg.levels.source_indices == list(range(1, 64, 2))
    np.testing.assert_allclose(cfg.levels.values,
                               0.125 + 0.25 * np.asarray(cfg.levels.source_indices))
    enc = r3d.target_encoding(cfg)
    assert (enc.dtype, enc.scale, enc.offset, enc.missing, enc.out_of_image) == (
        np.dtype("uint8"), 0.5, -32.0, 255, 254)
    assert cfg.variables[0].no_echo == -32.0
