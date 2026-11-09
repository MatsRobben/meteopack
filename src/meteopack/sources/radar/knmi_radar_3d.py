"""KNMI 3D radar reflectivity composite (KDP ``3d-radar-composite``), one file per 5 minutes.

``3D_RADAR_COMPOSITE_<YYYYMMDDHHMM>.nc`` is NetCDF4/HDF5 with one variable,
``reflectivity`` (uint16, ``(time, z, y, x)`` = ``(1, 64, 765, 700)``, one chunk
per level, gzip 4 + shuffle), on the national 1 km polar-stereographic radar
grid. Facts that each corrupt a store if missed:

* **The scale changes per file.** ``add_offset`` is -30 dBZ (code 0 is the
  floor: no echo) and ``scale_factor`` is chosen per file so that the file's
  highest value lands near the top of the uint16 range (0.0012-0.0017 seen).
  Every file is decoded with its own scale; a scale read once and reused would
  be wrong for every other file.
* **y runs south to north** (first row -4414.5 km), the opposite of the 2D KNMI
  products and of the RTCOR store, which are north-up (-3650.5 km first). The
  packed store flips y so it is pixel-aligned with RTCOR.
* **65535 is "not observed"** (outside the radars' reach at that height, or
  outside the composite) - a different fact from no echo, kept as its own code.

Packing (see `configs/products/radar/knmi_radar/composite3d_5min.yaml`): every
second native level (indices 1, 3, ..., 63 = 0.375 .. 15.875 km) is kept,
subsampled not averaged, and re-encoded to uint8 on the 2D reflectivity
product's calibration ``dBZ = 0.5 * code - 32``. No echo (native code 0) maps
to code 0, the same "no echo" the 2D product uses; 254 is "not observed" and
255 (the store's fill) a frame that is missing or could not be used.

The schema is checked on every file (:func:`check_schema`) and a file that
fails is never packed. What is checked and why is in the docstring there.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from meteopack.core import grids
from meteopack.core.config import ProductConfig
from meteopack.core.encoding import Encoding, encode

logger = logging.getLogger(__name__)

PREFIX = "3D_RADAR_COMPOSITE_"
FILENAME_RE = re.compile(r"^3D_RADAR_COMPOSITE_(\d{12})\.nc$")
VARIABLE = "reflectivity"
#: The variables a file holds besides `reflectivity`, all of which are checked.
COORDINATES = ("time", "z", "y", "x", "projection")

#: `source_layout_id` codes in the packed store.
LAYOUT_MISSING = 0
LAYOUT_CORRUPT = 1
LAYOUT_UNKNOWN = 2
LAYOUT_V1 = 3
LAYOUT_NAMES: dict[int, str] = {
    LAYOUT_MISSING: "missing",
    LAYOUT_CORRUPT: "corrupt_or_schema_failure",
    LAYOUT_UNKNOWN: "unknown",
    LAYOUT_V1: "kdp_3d_radar_composite_v1.0",
}


def filename_time(name: str) -> datetime | None:
    m = FILENAME_RE.match(name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=UTC)


def filename_for(when: datetime) -> str:
    return f"{PREFIX}{when.astimezone(UTC):%Y%m%d%H%M}.nc"


def day_slots(day: datetime, cadence: timedelta) -> list[datetime]:
    """Every timestamp of one UTC day, 00:00 to 23:55 for a 5-minute cadence."""
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    n = int(timedelta(days=1) / cadence)
    return [start + i * cadence for i in range(n)]


# ------------------------------------------------------------------ schema


@dataclass(frozen=True, slots=True)
class Schema:
    """What every file must look like. Built from the product config and the grid."""

    n_levels: int
    z_first: float
    z_step: float
    ny: int
    nx: int
    #: Cell centres the file must carry, *south-up* as the file stores them.
    y_south_up: np.ndarray = field(repr=False)
    x: np.ndarray = field(repr=False)
    proj4: str
    semi_major: float
    semi_minor: float
    fill_value: int
    add_offset: float
    units: str
    scale_range: tuple[float, float]
    coord_tolerance: float = 1e-3
    offset_tolerance: float = 1e-3
    cadence_seconds: int = 300

    @classmethod
    def from_config(cls, cfg: ProductConfig) -> Schema:
        spec = dict(cfg.extra.get("source_schema") or {})
        grid = grids.get(cfg.grid_id)
        y, x = grid.coordinates()  # north-up, as the packed store holds them
        lo, hi = spec.get("scale_factor_range", (0.0004, 0.0025))
        return cls(
            n_levels=int(spec.get("n_levels", 64)),
            z_first=float(spec.get("z_first_km", 0.125)),
            z_step=float(spec.get("z_step_km", 0.25)),
            ny=grid.ny, nx=grid.nx,
            y_south_up=np.asarray(y, dtype="float64")[::-1].copy(),
            x=np.asarray(x, dtype="float64"),
            proj4=str(spec.get("proj4_params",
                               "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 "
                               "+b=6356750 +x_0=0 +y_0=0 +units=km")),
            semi_major=float(spec.get("semi_major_axis", 6378140.0)),
            semi_minor=float(spec.get("semi_minor_axis", 6356750.0)),
            fill_value=int(spec.get("fill_value", 65535)),
            add_offset=float(spec.get("add_offset", -30.0)),
            units=str(spec.get("units", "dBZ")),
            scale_range=(float(lo), float(hi)),
            coord_tolerance=float(spec.get("coord_tolerance_km", 1e-3)),
            offset_tolerance=float(spec.get("add_offset_tolerance", 1e-3)),
            cadence_seconds=int(cfg.cadence_seconds or 300),
        )

    @property
    def z(self) -> np.ndarray:
        return self.z_first + self.z_step * np.arange(self.n_levels)


@dataclass(slots=True)
class FileCheck:
    """One file's schema check: what was found, and everything wrong with it."""

    name: str
    time: datetime | None = None
    scale_factor: float | None = None
    add_offset: float | None = None
    fingerprint: str | None = None
    #: How the file is stored (chunking, filters): recorded, not part of the
    #: fingerprint - it changed in mid-2024 without changing a single value.
    storage: str | None = None
    errors: list[str] = field(default_factory=list)
    #: The structural description the fingerprint hashes; kept for the drift report.
    descriptor: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _attr(obj, name: str, default=None):
    value = obj.attrs.get(name, default)
    if isinstance(value, bytes | np.bytes_):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _dim_names(dataset) -> list[str]:
    names = []
    for dim in dataset.dims:
        try:
            names.append(dim[0].name.rsplit("/", 1)[-1])
        except (IndexError, RuntimeError):
            names.append(dim.label or "?")
    return names


def check_schema(h5, name: str, schema: Schema) -> FileCheck:
    """Check one open file against `schema`. Never raises on content; collects errors.

    Checked, each because getting it wrong corrupts the store silently:
    variable presence, dtype, shape and dimension order; the 64 heights; x and
    y equal to the RTCOR grid's cell centres (y south-up here); the projection
    string and ellipsoid; the fill value, units and add_offset; a scale_factor
    in a sane range (it varies per file, so it is range-checked, not pinned);
    and the file's own time equal to its filename's timestamp, on the cadence.
    """
    check = FileCheck(name=name, time=filename_time(name))
    err = check.errors
    if check.time is None:
        err.append(f"filename {name!r} is not {PREFIX}YYYYMMDDHHMM.nc")
    elif (check.time.minute * 60) % schema.cadence_seconds:
        err.append(f"{name}: timestamp is off the {schema.cadence_seconds}s cadence")

    present = sorted(k for k in h5)
    missing = [k for k in (VARIABLE, *COORDINATES) if k not in h5]
    if missing:
        err.append(f"missing variable(s) {missing}; file holds {present}")
        check.descriptor = {"variables": present}
        check.fingerprint = _fingerprint(check.descriptor)
        return check
    extra = [k for k in present if k not in (VARIABLE, *COORDINATES)]

    refl = h5[VARIABLE]
    dims = _dim_names(refl)
    want_shape = (1, schema.n_levels, schema.ny, schema.nx)
    if refl.dtype != np.dtype("uint16"):
        err.append(f"reflectivity dtype {refl.dtype}, expected uint16")
    if tuple(refl.shape) != want_shape:
        err.append(f"reflectivity shape {tuple(refl.shape)}, expected {want_shape}")
    if dims != ["time", "z", "y", "x"]:
        err.append(f"reflectivity dims {dims}, expected ['time', 'z', 'y', 'x']")

    fill = _attr(refl, "_FillValue")
    offset = _attr(refl, "add_offset")
    scale = _attr(refl, "scale_factor")
    units = _attr(refl, "units")
    if fill is None or int(fill) != schema.fill_value:
        err.append(f"_FillValue {fill}, expected {schema.fill_value}")
    if units != schema.units:
        err.append(f"units {units!r}, expected {schema.units!r}")
    if offset is None or abs(float(offset) - schema.add_offset) > schema.offset_tolerance:
        err.append(f"add_offset {offset}, expected {schema.add_offset}")
    lo, hi = schema.scale_range
    if scale is None or not np.isfinite(float(scale)) or not lo <= float(scale) <= hi:
        err.append(f"scale_factor {scale} outside the sane range [{lo}, {hi}]")
    check.scale_factor = None if scale is None else float(scale)
    check.add_offset = None if offset is None else float(offset)

    z = np.asarray(h5["z"][:], dtype="float64")
    if z.shape != (schema.n_levels,) or not np.allclose(z, schema.z, atol=1e-4):
        err.append(f"z levels differ: {z[:3].tolist()}..{z[-2:].tolist()} (n={z.size}), "
                   f"expected {schema.z[:3].tolist()}..{schema.z[-2:].tolist()}")
    y = np.asarray(h5["y"][:], dtype="float64")
    x = np.asarray(h5["x"][:], dtype="float64")
    if y.shape != schema.y_south_up.shape or not np.allclose(
            y, schema.y_south_up, atol=schema.coord_tolerance):
        err.append(f"y differs from the RTCOR grid (south-up): first {y[:1].tolist()} "
                   f"last {y[-1:].tolist()}, expected {schema.y_south_up[0]} .. "
                   f"{schema.y_south_up[-1]}")
    if x.shape != schema.x.shape or not np.allclose(x, schema.x, atol=schema.coord_tolerance):
        err.append(f"x differs from the RTCOR grid: first {x[:1].tolist()} "
                   f"last {x[-1:].tolist()}")

    proj = h5["projection"]
    proj4 = _attr(proj, "proj4_params")
    a = _attr(proj, "semi_major_axis")
    b = _attr(proj, "semi_minor_axis")
    if proj4 != schema.proj4:
        err.append(f"proj4_params {proj4!r}, expected {schema.proj4!r}")
    if a is None or b is None or abs(float(a) - schema.semi_major) > 1e-3 \
            or abs(float(b) - schema.semi_minor) > 1e-3:
        err.append(f"ellipsoid ({a}, {b}), expected ({schema.semi_major}, {schema.semi_minor})")

    t = np.asarray(h5["time"][:], dtype="float64").reshape(-1)
    t_units = _attr(h5["time"], "units", "")
    if t.size != 1 or not str(t_units).startswith("seconds since 1970-01-01"):
        err.append(f"time {t.tolist()} [{t_units}], expected one value in seconds since 1970")
    elif check.time is not None and round(t[0]) != int(check.time.timestamp()):
        actual = datetime.fromtimestamp(float(t[0]), UTC).isoformat()
        err.append(f"time {actual} differs from the filename's {check.time.isoformat()}")

    check.descriptor = {
        "variables": present,
        "extra_variables": extra,
        "dtype": str(refl.dtype),
        "shape": list(refl.shape),
        "dims": dims,
        "fill": None if fill is None else int(fill),
        "units": units,
        "add_offset": None if offset is None else round(float(offset), 3),
        "z": [round(float(v), 4) for v in z],
        "y_first_last": [round(float(y[0]), 3), round(float(y[-1]), 3)] if y.size else [],
        "x_first_last": [round(float(x[0]), 3), round(float(x[-1]), 3)] if x.size else [],
        "ny_nx": [int(y.size), int(x.size)],
        "proj4": proj4,
        "ellipsoid": [a, b],
    }
    check.fingerprint = _fingerprint(check.descriptor)
    check.storage = (f"chunks={'x'.join(str(c) for c in refl.chunks or ())} "
                     f"{refl.compression}{refl.compression_opts or ''}"
                     f"{' shuffle' if refl.shuffle else ''}")
    return check


def _fingerprint(descriptor: Mapping[str, Any]) -> str:
    blob = json.dumps(descriptor, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ reading


def read_levels(h5, levels: Sequence[int]) -> np.ndarray:
    """The native uint16 codes of `levels`, ``(len(levels), ny, nx)``, as stored (south-up).

    Fast path: the compressed chunks are read directly and inflated with zlib,
    which releases the GIL - h5py's own read holds a global lock for the whole
    decompression and so serialises every thread. Measured on 6 threads: 16.3
    files/s against 7.5 for h5py per-level reads (2024-07, one chunk per
    level). Any chunk shape works: files up to about June 2024 hold 32 levels
    in a (32, 383, 350) chunk, and reading those one level at a time through
    h5py re-inflated each chunk for every level (2.5 s a file). A filter
    pipeline other than shuffle + deflate falls back to one h5py read of the
    whole volume.
    """
    ds = h5[VARIABLE]
    _, _nz, ny, nx = ds.shape
    levels = [int(k) for k in levels]
    fast = (ds.chunks is not None and ds.chunks[0] == 1 and ds.compression == "gzip"
            and ds.shuffle and not ds.fletcher32 and ds.scaleoffset is None
            and ds.dtype == np.dtype("<u2") and set(ds._filters) == {"gzip", "shuffle"})
    if not fast:
        return np.asarray(ds[0])[levels]
    _, cz, cy, cx = ds.chunks
    out = np.empty((len(levels), ny, nx), dtype="<u2")
    wanted: dict[int, list[tuple[int, int]]] = {}
    for i, level in enumerate(levels):
        wanted.setdefault(level // cz * cz, []).append((i, level))
    for z0, picks in wanted.items():
        for y0 in range(0, ny, cy):
            for x0 in range(0, nx, cx):
                mask, blob = ds.id.read_direct_chunk((0, z0, y0, x0))
                if mask != 0:  # a filter was skipped for this chunk: let HDF5 decode it
                    ys, xs = slice(y0, min(y0 + cy, ny)), slice(x0, min(x0 + cx, nx))
                    for i, level in picks:
                        out[i, ys, xs] = ds[0, level, ys, xs]
                    continue
                chunk = _unshuffle_u2(zlib.decompress(blob), cz * cy * cx).reshape(cz, cy, cx)
                hy, hx = min(cy, ny - y0), min(cx, nx - x0)
                for i, level in picks:
                    out[i, y0:y0 + hy, x0:x0 + hx] = chunk[level - z0, :hy, :hx]
    return out


def _unshuffle_u2(data: bytes, n: int) -> np.ndarray:
    """Undo HDF5's byte shuffle for little-endian uint16: all low bytes, then all high."""
    raw = np.frombuffer(data, dtype=np.uint8)
    if raw.size != 2 * n:
        raise ValueError(f"chunk inflated to {raw.size} bytes, expected {2 * n}")
    out = raw[n:].astype("<u2")
    out <<= 8
    out |= raw[:n]
    return out


def target_encoding(cfg: ProductConfig) -> Encoding:
    var = cfg.variables[0]
    enc = Encoding.from_config(var.encoding, name=var.name)
    if enc is None:
        raise ValueError(f"{cfg.store_id}: {var.name} needs an integer encoding")
    return enc


def encode_levels(codes: np.ndarray, *, scale: float, offset: float, target: Encoding,
                  no_echo: float, native_fill: int = 65535,
                  native_no_echo: int = 0) -> np.ndarray:
    """Native uint16 codes (south-up) -> target codes, flipped north-up.

    Native `native_no_echo` (0, the -30 dBZ floor) becomes the target's no-echo
    value; native fill becomes the target's `out_of_image` ("not observed");
    everything else is decoded with this file's own scale and offset, rounded
    to the target step and clipped clear of the sentinels.
    """
    packed = encode(codes, target, source_scale=scale, source_offset=offset,
                    source_out_of_image={native_fill}, source_no_echo={native_no_echo},
                    no_echo=no_echo)
    return packed[:, ::-1, :]


@dataclass(slots=True)
class PackedFile:
    check: FileCheck
    codes: np.ndarray | None  # (nz, ny, nx) target dtype, north-up; None if not packable
    read_s: float = 0.0


def pack_file(path: Path, cfg: ProductConfig, schema: Schema, target: Encoding) -> PackedFile:
    """Check, read and encode one file. A file that fails its check yields no codes."""
    import time

    import h5py

    t0 = time.perf_counter()
    levels = cfg.levels.source_indices if cfg.levels and cfg.levels.source_indices else None
    if levels is None:
        raise ValueError(f"{cfg.store_id}: levels.source_indices must name the kept levels")
    no_echo = cfg.variables[0].no_echo
    if no_echo is None:
        raise ValueError(f"{cfg.store_id}: the variable must declare no_echo")
    try:
        with h5py.File(path, "r") as h5:
            check = check_schema(h5, Path(path).name, schema)
            if not check.ok:
                return PackedFile(check, None, time.perf_counter() - t0)
            native = read_levels(h5, levels)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a corrupt frame
        check = FileCheck(name=Path(path).name, time=filename_time(Path(path).name))
        check.errors.append(f"unreadable: {type(exc).__name__}: {exc}")
        return PackedFile(check, None, time.perf_counter() - t0)
    codes = encode_levels(native, scale=check.scale_factor, offset=check.add_offset,
                          target=target, no_echo=float(no_echo),
                          native_fill=schema.fill_value)
    return PackedFile(check, codes, time.perf_counter() - t0)
