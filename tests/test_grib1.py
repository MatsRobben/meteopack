"""The minimal GRIB1 reader, against messages written here byte by byte (and ecCodes if present)."""
from __future__ import annotations

import math
from datetime import UTC, datetime

import numpy as np
import pytest

from meteopack.sources.nwp import grib1


def _sm(value: int, n: int) -> bytes:
    """Sign-and-magnitude, as GRIB1 writes signed integers."""
    top = 1 << (8 * n - 1)
    return ((top | -value) if value < 0 else value).to_bytes(n, "big")


def _ibm(value: float, *, floor: bool) -> bytes:
    """An IBM single; rounded down (for a reference value) when `floor`."""
    if value == 0:
        return b"\x00\x00\x00\x00"
    sign = 0x80 if value < 0 else 0
    m, e = abs(value), 64
    while m >= 1:
        m /= 16
        e += 1
    while m < 1 / 16:
        m *= 16
        e -= 1
    mant = m * (1 << 24)
    # A floor of the signed value: truncate positives, round negative magnitudes up.
    mant = int(math.ceil(mant) if (sign and floor) else math.floor(mant))
    if mant >= 1 << 24:
        mant >>= 4
        e += 1
    return bytes([sign | e]) + mant.to_bytes(3, "big")


def encode_message(values: np.ndarray, *, param: int, level_type: int = 105, level: int = 0,
                   time_range: int = 0, p1: int = 0, p2: int = 0,
                   run: datetime = datetime(2026, 1, 8, tzinfo=UTC), table2: int = 253,
                   centre: int = 99, lat0: float = 49.0, lon0: float = 0.0,
                   dlat: float = 0.018, dlon: float = 0.029, decimal: int = 0,
                   bits: int = 16, scan: int = 0x40) -> bytes:
    """A GRIB1 grid-point message, simple packing, regular lat/lon; NaN -> bitmap.

    `values` is ``(nj, ni)`` with rows south to north (scan 0x40), as the
    reader returns them.
    """
    nj, ni = values.shape
    flat = np.asarray(values, dtype="float64").reshape(-1)
    present = np.isfinite(flat)
    has_bitmap = not present.all()
    v = flat[present] * 10.0 ** decimal
    lo = float(v.min()) if v.size else 0.0
    span = float(v.max()) - lo if v.size else 0.0
    if bits and span > 0:
        e = math.ceil(math.log2(span / (2 ** bits - 1)))
    else:
        e = 0
    ref_bytes = _ibm(lo, floor=True)
    ref = grib1.ibm_float(ref_bytes)
    if bits:
        x = np.clip(np.rint((v - ref) / 2.0 ** e), 0, 2 ** bits - 1).astype("uint64")
        packed = np.unpackbits(
            x.astype(">u8").view(np.uint8).reshape(-1, 8), axis=1)[:, 64 - bits:].reshape(-1)
        data = np.packbits(packed).tobytes()
    else:
        data = b""
    pds = bytearray(28)
    pds[0:3] = (28).to_bytes(3, "big")
    pds[3], pds[4], pds[5], pds[6] = table2, centre, 1, 255
    pds[7] = 0x80 | (0x40 if has_bitmap else 0)
    pds[8], pds[9] = param, level_type
    pds[10:12] = level.to_bytes(2, "big")
    pds[12], pds[13], pds[14], pds[15], pds[16] = (run.year % 100 or 100, run.month, run.day,
                                                   run.hour, run.minute)
    pds[17], pds[18], pds[19], pds[20] = 1, p1, p2, time_range
    pds[24] = (run.year - 1) // 100 + 1
    pds[26:28] = _sm(decimal, 2)
    gds = bytearray(32)
    gds[0:3] = (32).to_bytes(3, "big")
    gds[3], gds[4], gds[5] = 0, 255, 0
    gds[6:8] = ni.to_bytes(2, "big")
    gds[8:10] = nj.to_bytes(2, "big")
    gds[10:13] = _sm(round(lat0 * 1000), 3)
    gds[13:16] = _sm(round(lon0 * 1000), 3)
    gds[16] = 0x88
    gds[17:20] = _sm(round((lat0 + (nj - 1) * dlat) * 1000), 3)
    gds[20:23] = _sm(round((lon0 + (ni - 1) * dlon) * 1000), 3)
    gds[23:25] = round(dlon * 1000).to_bytes(2, "big")
    gds[25:27] = round(dlat * 1000).to_bytes(2, "big")
    gds[27] = scan
    bms = b""
    if has_bitmap:
        bitmap = np.packbits(present.astype(np.uint8)).tobytes()
        length = 6 + len(bitmap)
        pad = length % 2
        bms = (length + pad).to_bytes(3, "big") + bytes([(-flat.size) % 8]) + b"\x00\x00" \
            + bitmap + b"\x00" * pad
    length = 11 + len(data)
    pad = length % 2
    unused = (len(data) * 8 - v.size * bits) + 8 * pad
    bds = ((length + pad).to_bytes(3, "big") + bytes([unused & 0x0F]) + _sm(e, 2) + ref_bytes
           + bytes([bits]) + data + b"\x00" * pad)
    body = bytes(pds) + bytes(gds) + bms + bds
    total = 8 + len(body) + 4
    return b"GRIB" + total.to_bytes(3, "big") + b"\x01" + body + b"7777"


def _field(nj=7, ni=5, seed=0, lo=0.0, hi=100.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=(nj, ni))


def test_ibm_float_round_trips_and_known_values():
    assert grib1.ibm_float(b"\x42\x64\x00\x00") == 100.0
    assert grib1.ibm_float(b"\xc2\x76\xa0\x00") == -118.625
    assert grib1.ibm_float(b"\x00\x00\x00\x00") == 0.0
    for value in (1.0, -2.5, 263.15, 101325.0, 1e-6):
        got = grib1.ibm_float(_ibm(value, floor=True))
        assert got <= value and abs(got - value) <= abs(value) * 2 ** -20


def test_simple_packing_decodes_to_the_packing_precision():
    values = _field(lo=263.0, hi=290.0)
    msg = grib1.parse_message(encode_message(values, param=11, level=2))
    assert msg.key == (11, 105, 2, 0)
    assert (msg.grid.ni, msg.grid.nj) == (5, 7)
    assert msg.grid.south_to_north and msg.grid.west_to_east
    assert msg.grid.lat_first == 49.0 and msg.grid.dj == 0.018
    np.testing.assert_allclose(msg.values(), values, atol=27 / 2 ** 16)


def test_negative_values_decimal_scale_and_step_ranges():
    values = _field(lo=-12.0, hi=14.0, seed=1)
    raw = encode_message(values, param=162, level=10, time_range=2, p1=2, p2=3, decimal=2)
    msg = grib1.parse_message(raw)
    assert msg.decimal_scale == 2 and msg.step_range == (2, 3)
    np.testing.assert_allclose(msg.values(), values, atol=1e-3)
    acc = grib1.parse_message(encode_message(values, param=61, time_range=4, p1=0, p2=5))
    assert acc.step_range == (0, 5)
    inst = grib1.parse_message(encode_message(values, param=61, p1=7))
    assert inst.step_range == (7, 7)


def test_bitmap_points_are_nan():
    values = _field(lo=100.0, hi=9000.0, seed=2)
    values[0, :3] = np.nan
    values[4, 2] = np.nan
    msg = grib1.parse_message(encode_message(values, param=186, level_type=200, bits=24))
    assert msg.has_bitmap
    out = msg.values()
    np.testing.assert_array_equal(np.isnan(out), np.isnan(values))
    np.testing.assert_allclose(out[np.isfinite(values)], values[np.isfinite(values)], atol=1e-3)


def test_constant_field_with_zero_bits():
    values = np.full((3, 4), 0.0)
    msg = grib1.parse_message(encode_message(values, param=61, bits=0))
    assert msg.bits == 0
    np.testing.assert_array_equal(msg.values(), values)


def test_north_to_south_scanning_is_flipped_to_south_up():
    values = _field(seed=3)
    north_first = encode_message(values[::-1], param=1, level_type=103, scan=0x00)
    np.testing.assert_allclose(grib1.parse_message(north_first).values(), values,
                               atol=100 / 2 ** 16)


def test_iter_messages_walks_a_file_and_refuses_what_it_cannot_read():
    a = encode_message(_field(seed=4), param=33, level=10)
    b = encode_message(_field(seed=5), param=34, level=10)
    msgs = list(grib1.iter_messages(a + b))
    assert [m.key for m in msgs] == [(33, 105, 10, 0), (34, 105, 10, 0)]
    assert msgs[1].offset == len(a)
    with pytest.raises(grib1.GribError, match="7777"):
        list(grib1.iter_messages(a[:-1] + b"8"))
    with pytest.raises(grib1.GribError, match="no GRIB"):
        list(grib1.iter_messages(b"XXXX" + a[4:]))
    edition2 = bytearray(a)
    edition2[7] = 2
    with pytest.raises(grib1.GribError, match="edition 2"):
        grib1.parse_message(bytes(edition2))
    complex_packing = bytearray(a)
    bds = 8 + 28 + 32  # indicator, product and grid sections, no bitmap
    complex_packing[bds + 3] |= 0x40
    with pytest.raises(grib1.GribError, match="simple"):
        grib1.parse_message(bytes(complex_packing)).values()


def test_agrees_with_eccodes_when_installed(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    values = _field(lo=-5.0, hi=30.0, seed=6)
    values[2, 1] = np.nan
    raw = encode_message(values, param=67, bits=16)
    path = tmp_path / "m.grb"
    path.write_bytes(raw)
    with path.open("rb") as f:
        gid = eccodes.codes_grib_new_from_file(f)
        try:
            theirs = eccodes.codes_get_values(gid).reshape(7, 5)
            missing = eccodes.codes_get(gid, "missingValue")
        finally:
            eccodes.codes_release(gid)
    ours = grib1.parse_message(raw).values()
    theirs = np.where(theirs == missing, np.nan, theirs)
    np.testing.assert_allclose(ours, theirs, atol=1e-9, equal_nan=True)
