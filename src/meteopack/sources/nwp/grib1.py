"""A minimal GRIB edition 1 reader: grid-point fields, simple packing, regular lat/lon.

That is all the KNMI HARMONIE P1 files hold (``table2Version`` 253, 16-bit
``grid_simple`` on a 390 x 390 regular lat/lon grid, no bitmap), and reading it
needs ~150 lines of numpy rather than ecCodes - which is not in the
environment the stream runs from. Some fields (cloud base, snow depth, the
ISBA tiles) carry a bitmap: points outside it are missing and come back NaN
(ecCodes would show its default missing value, 9999). Anything outside that
subset (complex or spherical-harmonic packing, a predefined bitmap, a rotated
or Gaussian grid, a message over 8 MB) is refused with :class:`GribError`,
never guessed at; the tests check this reader against ecCodes where ecCodes is
installed.

Facts each section is decoded with (WMO FM 92 GRIB edition 1):

* Signed integers are sign-and-magnitude (top bit = sign), not two's complement.
* The reference value is an IBM System/360 single-precision float.
* ``value = (R + X * 2**E) / 10**D`` with X the packed unsigned integers.
* Scanning mode bit 2 (0x40) set means rows run south to north.
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np


class GribError(ValueError):
    """A message this reader does not decode (or a broken one)."""


@dataclass(frozen=True, slots=True)
class Grid:
    """Section 2 of a regular lat/lon message, in degrees."""

    ni: int
    nj: int
    lat_first: float
    lon_first: float
    lat_last: float
    lon_last: float
    di: float
    dj: float
    scanning_mode: int
    #: Resolution-and-component flags: bit 0x08 set means u/v are grid-relative.
    component_flags: int

    @property
    def south_to_north(self) -> bool:
        return bool(self.scanning_mode & 0x40)

    @property
    def west_to_east(self) -> bool:
        return not self.scanning_mode & 0x80

    def describe(self) -> list:
        return [self.ni, self.nj, round(self.lat_first, 3), round(self.lon_first, 3),
                round(self.lat_last, 3), round(self.lon_last, 3), round(self.di, 3),
                round(self.dj, 3), self.scanning_mode]


@dataclass(slots=True)
class Message:
    """One decoded GRIB1 message header, with the data section left packed."""

    offset: int
    length: int
    table2_version: int
    centre: int
    parameter: int
    level_type: int
    level: int
    year: int
    month: int
    day: int
    hour: int
    minute: int
    time_unit: int
    p1: int
    p2: int
    time_range: int
    decimal_scale: int
    grid: Grid
    binary_scale: int
    reference: float
    bits: int
    #: Section 4 flags (high nibble of its octet 4): 0 = grid point, simple
    #: packing, floats, no extra flags. Anything else is refused on decode.
    bds_flags: int
    has_bitmap: bool
    _data: memoryview = field(repr=False, default=memoryview(b""))
    #: Section 3's bit map (one bit per grid point, 1 = value present).
    _bitmap: memoryview | None = field(repr=False, default=None)

    @property
    def key(self) -> tuple[int, int, int, int]:
        """``(indicatorOfParameter, indicatorOfTypeOfLevel, level, timeRangeIndicator)``."""
        return (self.parameter, self.level_type, self.level, self.time_range)

    @property
    def step_range(self) -> tuple[int, int]:
        """``(start, end)`` of the product, in `time_unit`s (end == start for an instant)."""
        if self.time_range == 10:
            return (self.p1 * 256 + self.p2,) * 2
        if self.time_range in (2, 3, 4, 5):
            return (self.p1, self.p2)
        return (self.p1, self.p1)

    def values(self) -> np.ndarray:
        """The field as float64 ``(nj, ni)``, rows south to north, columns west to east.

        Points outside the bitmap, if the message has one, are NaN.
        """
        if self.bds_flags & 0xF0:
            raise GribError(f"section 4 flags {self.bds_flags:#x}: only grid-point simple "
                            f"packing of floats is supported")
        n_points = self.grid.ni * self.grid.nj
        present = None
        if self.has_bitmap:
            if self._bitmap is None:
                raise GribError("a predefined (table) bitmap is not supported")
            present = np.unpackbits(np.frombuffer(self._bitmap, dtype=np.uint8))[:n_points]
            if present.size < n_points:
                raise GribError(f"bitmap holds {present.size} bits for {n_points} points")
            present = present.astype(bool)
        n = int(present.sum()) if present is not None else n_points
        scale = 10.0 ** -self.decimal_scale
        if self.bits == 0:
            out = np.full(n, self.reference * scale, dtype="float64")
        elif self.bits in (8, 16, 32):
            packed = np.frombuffer(self._data, dtype=f">u{self.bits // 8}", count=n)
            out = (self.reference + packed.astype("float64") * 2.0 ** self.binary_scale) * scale
        else:
            packed = _unpack_bits(self._data, self.bits, n)
            out = (self.reference + packed * 2.0 ** self.binary_scale) * scale
        if self.grid.scanning_mode & 0x20:
            raise GribError("column-major scanning (scanning mode bit 3) is not supported")
        if present is not None:
            full = np.full(n_points, np.nan)
            full[present] = out
            out = full
        out = out.reshape(self.grid.nj, self.grid.ni)
        if not self.grid.south_to_north:
            out = out[::-1]
        if not self.grid.west_to_east:
            out = out[:, ::-1]
        return out


def _unpack_bits(data: memoryview, bits: int, n: int) -> np.ndarray:
    """Big-endian unsigned integers of any width up to 32 bits."""
    raw = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: n * bits]
    if raw.size < n * bits:
        raise GribError(f"data section holds {raw.size} bits, need {n * bits}")
    weights = 1 << np.arange(bits - 1, -1, -1, dtype=np.uint64)
    return (raw.reshape(n, bits).astype(np.uint64) @ weights).astype("float64")


def _u(b: bytes | memoryview, i: int, n: int) -> int:
    return int.from_bytes(bytes(b[i:i + n]), "big")


def _s(b: bytes | memoryview, i: int, n: int) -> int:
    """Sign-and-magnitude: the top bit is the sign."""
    v = _u(b, i, n)
    top = 1 << (8 * n - 1)
    return -(v & (top - 1)) if v & top else v


def ibm_float(b: bytes | memoryview) -> float:
    """An IBM System/360 single-precision float (GRIB1's reference value)."""
    v = _u(b, 0, 4)
    if v & 0x7FFFFFFF == 0:
        return 0.0
    sign = -1.0 if v >> 31 else 1.0
    exponent = (v >> 24) & 0x7F
    mantissa = v & 0x00FFFFFF
    return sign * mantissa / float(1 << 24) * 16.0 ** (exponent - 64)


def _grid(sec: memoryview) -> Grid:
    if sec[5] != 0:
        raise GribError(f"data representation type {sec[5]}: only regular lat/lon (0)")
    return Grid(
        ni=_u(sec, 6, 2), nj=_u(sec, 8, 2),
        lat_first=_s(sec, 10, 3) / 1000.0, lon_first=_s(sec, 13, 3) / 1000.0,
        component_flags=sec[16],
        lat_last=_s(sec, 17, 3) / 1000.0, lon_last=_s(sec, 20, 3) / 1000.0,
        di=_u(sec, 23, 2) / 1000.0, dj=_u(sec, 25, 2) / 1000.0,
        scanning_mode=sec[27],
    )


def parse_message(buf: bytes | memoryview, offset: int = 0) -> Message:
    """Parse the message starting at `offset` of `buf` (headers only; data stays packed)."""
    mv = memoryview(buf)
    if bytes(mv[offset:offset + 4]) != b"GRIB":
        raise GribError(f"no GRIB indicator at offset {offset}")
    length = _u(mv, offset + 4, 3)
    edition = mv[offset + 7]
    if edition != 1:
        raise GribError(f"GRIB edition {edition}, only 1 is supported")
    if length & 0x800000:
        raise GribError("a 'large' GRIB1 message (over 8 MB) is not supported")
    if offset + length > len(mv):
        raise GribError(f"message at {offset} is {length} bytes, buffer ends at {len(mv)}")
    msg = mv[offset:offset + length]
    if bytes(msg[-4:]) != b"7777":
        raise GribError(f"message at {offset} does not end in 7777")
    pos = 8
    pds = msg[pos:pos + _u(msg, pos, 3)]
    pos += len(pds)
    flags = pds[7]
    if not flags & 0x80:
        raise GribError("no grid description section")
    gds = msg[pos:pos + _u(msg, pos, 3)]
    pos += len(gds)
    has_bitmap = bool(flags & 0x40)
    bitmap = None
    if has_bitmap:
        bms = msg[pos:pos + _u(msg, pos, 3)]
        pos += len(bms)
        if _u(bms, 4, 2) == 0:  # 0: the bit map follows; else a predefined one
            bitmap = bms[6:]
    bds_len = _u(msg, pos, 3)
    bds = msg[pos:pos + bds_len]
    level_type = pds[9]
    # Level types with two one-octet values; every type used here is a single
    # two-octet value (105 height above ground, 103 above sea, 200 entire atmosphere).
    level = _u(pds, 10, 2)
    century = pds[24] if len(pds) > 24 else 21
    return Message(
        offset=offset, length=length,
        table2_version=pds[3], centre=pds[4], parameter=pds[8],
        level_type=level_type, level=level,
        year=(century - 1) * 100 + pds[12], month=pds[13], day=pds[14],
        hour=pds[15], minute=pds[16], time_unit=pds[17], p1=pds[18], p2=pds[19],
        time_range=pds[20],
        decimal_scale=_s(pds, 26, 2) if len(pds) >= 28 else 0,
        grid=_grid(gds),
        binary_scale=_s(bds, 4, 2), reference=ibm_float(bds[6:10]), bits=bds[10],
        bds_flags=bds[3] & 0xF0, has_bitmap=has_bitmap,
        _data=bds[11:], _bitmap=bitmap,
    )


def iter_messages(buf: bytes | memoryview) -> Iterator[Message]:
    """Every message in `buf`, in order. Bytes between messages are not allowed."""
    mv = memoryview(buf)
    offset = 0
    while offset < len(mv):
        if len(mv) - offset < 8:
            if bytes(mv[offset:]).strip(b"\x00"):
                raise GribError(f"{len(mv) - offset} stray bytes after the last message")
            return
        msg = parse_message(mv, offset)
        yield msg
        offset += msg.length


__all__ = ["GribError", "Grid", "Message", "ibm_float", "iter_messages", "parse_message"]
