"""EUMETNET OPERA European radar rain-rate composite (ODIM HDF5), 15-minute, 2 km.

Streamed day by day from the public archive bucket
``https://s3.waw3-1.cloudferro.com/openradar-archive/`` (anonymous, CC BY 4.0;
``YYYY/MM/DD/OPERA/COMP/OPERA@YYYYMMDDTHHMM@0@<PRODUCT>.h5``) by
``meteopack stream opera --product rate_15min`` (meteopack.jobs.stream_kdp with
the ``opera_odim`` format below). Raw files are deleted once their day is
packed.

Two production chains make the rain rate, and the series breaks between them:

* **ODYSSEY** ``QIND_RATE.h5`` (ODIM 2.0, "H5rad 2.0"), 2012 to 2024-07-05
  06:00 inclusive: ``dataset1`` = RATE, ``dataset2`` = QIND; the data window is
  the 15 minutes T-10 .. T+5 min.
* **NIMBUS** ``RATE.h5`` (ODIM 2.4, "H5rad 2.4"), from 2024-07-05 05:30:
  ``dataset1/data1`` = RATE plus ``quality1``; instantaneous (start = end = T),
  Z-R a=200, b=1.6.

Both overlap on 2024-07-05 05:30-06:00; the cutover is a configured instant
(``extra.opera.cutover``, the last ODYSSEY frame) and every slot takes exactly
the chain the rule names, so the overlap never produces two frames for one
time step, and stray files of the wrong chain (ODYSSEY files reappear on
2024-12-03..09) are ignored. Which chain made each frame is stored per frame
(``chain``); nothing is calibrated across the break here.

Facts that each corrupt a store if missed:

* Values are **float64 physical mm/h** (gain 1, offset 0), not counts.
* ``nodata`` = -9999000 is *outside radar coverage* and ``undetect`` =
  -8888000 is *observed, no rain*. The first becomes the store's
  ``out_of_image`` code, the second 0 mm/h.
* The grid is LAEA ``+lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000``,
  1900 x 2200 at 2 km, upper-left *corner* at projected (0, 0), rows running
  south. The store keeps a fixed window of it on the native grid (no
  resampling): rows/cols in ``extra.opera.window``, the grid ``grid_id``.

Every file is checked (:func:`check_schema`) before it is packed: name, chain
(by structure, not only by name), ``/what`` time and object, projection, grid
size and corners, quantity, dtype, shape, gain, offset, nodata and undetect. A
file that fails is never packed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from meteopack.core import grids
from meteopack.core.config import ProductConfig
from meteopack.core.encoding import Encoding, encode

logger = logging.getLogger(__name__)

FILENAME_RE = re.compile(r"^OPERA@(\d{8}T\d{4})@0@([A-Z_]+)\.h5$")

ODYSSEY = "ODYSSEY"
NIMBUS = "NIMBUS"
#: The file product of each chain's rain rate.
PRODUCTS = {ODYSSEY: "QIND_RATE", NIMBUS: "RATE"}
#: ``chain`` frame-variable codes (0 = no frame).
CHAIN_CODES = {ODYSSEY: 1, NIMBUS: 2}

#: `source_layout_id` codes: 0 and 1 are the stream's own (missing, corrupt).
LAYOUT_MISSING = 0
LAYOUT_CORRUPT = 1
LAYOUT_ODYSSEY = 2
LAYOUT_NIMBUS = 3
LAYOUT_NAMES: dict[int, str] = {
    LAYOUT_MISSING: "missing",
    LAYOUT_CORRUPT: "corrupt_or_schema_failure",
    LAYOUT_ODYSSEY: "opera_odyssey_qind_rate_odim2.0",
    LAYOUT_NIMBUS: "opera_nimbus_rate_odim2.4",
}
LAYOUT_OF_CHAIN = {ODYSSEY: LAYOUT_ODYSSEY, NIMBUS: LAYOUT_NIMBUS}

#: What each chain's file says about itself: ``/what/version`` and the
#: ``Conventions`` root attribute.
VERSIONS = {ODYSSEY: "H5rad 2.0", NIMBUS: "H5rad 2.4"}


def parse_filename(name: str) -> tuple[datetime, str] | None:
    """``OPERA@20240705T0615@0@RATE.h5`` -> (time, product), or None."""
    m = FILENAME_RE.match(name)
    if not m:
        return None
    when = datetime.strptime(m.group(1), "%Y%m%dT%H%M").replace(tzinfo=UTC)
    return when, m.group(2)


def chain_for(when: datetime, cutover: datetime) -> str:
    """The chain a time step is taken from: ODYSSEY up to and including `cutover`."""
    return ODYSSEY if when <= cutover else NIMBUS


def filename_for(when: datetime, chain: str) -> str:
    return f"OPERA@{when.astimezone(UTC):%Y%m%dT%H%M}@0@{PRODUCTS[chain]}.h5"


def parse_nodes(text: str | bytes | None) -> list[str]:
    """``/how/nodes`` -> radar ids, in each chain's own spelling.

    ODYSSEY writes ``NOD:frnan,NOD:frbou,...``, NIMBUS ``'behel','bejab',...``
    and CIRRUS ``behel, bewid, ...``; all become ``["frnan", "frbou", ...]``.
    """
    if text is None:
        return []
    if isinstance(text, bytes | np.bytes_):
        text = text.decode("utf-8", "replace")
    out = []
    for token in re.split(r"[,\s'\"]+", str(text)):
        token = token.strip()
        if token.upper().startswith("NOD:"):
            token = token[4:]
        if token:
            out.append(token.lower())
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------ schema


@dataclass(frozen=True, slots=True)
class Schema:
    """What every file must look like, from ``extra.opera`` and the store's grid."""

    projdef: str
    xsize: int
    ysize: int
    xscale: float
    yscale: float
    ul_lonlat: tuple[float, float]
    lr_lonlat: tuple[float, float]
    corner_tolerance_deg: float
    gain: float
    offset: float
    nodata: float
    undetect: float
    #: The store's window on the native grid: rows [row0, row0+ny), cols [col0, col0+nx).
    row0: int
    col0: int
    ny: int
    nx: int
    cutover: datetime
    cadence_seconds: int = 900

    @classmethod
    def from_config(cls, cfg: ProductConfig) -> Schema:
        spec = dict(cfg.extra.get("opera") or {})
        native = dict(spec.get("native_grid") or {})
        window = dict(spec["window"])
        cutover = datetime.fromisoformat(str(spec["cutover"]))
        schema = cls(
            projdef=str(native["projdef"]),
            xsize=int(native["xsize"]), ysize=int(native["ysize"]),
            xscale=float(native["xscale"]), yscale=float(native["yscale"]),
            ul_lonlat=tuple(float(v) for v in native["ul_lonlat"]),
            lr_lonlat=tuple(float(v) for v in native["lr_lonlat"]),
            corner_tolerance_deg=float(native.get("corner_tolerance_deg", 1e-4)),
            gain=float(spec.get("gain", 1.0)), offset=float(spec.get("offset", 0.0)),
            nodata=float(spec.get("nodata", -9999000.0)),
            undetect=float(spec.get("undetect", -8888000.0)),
            row0=int(window["row0"]), col0=int(window["col0"]),
            ny=int(window["ny"]), nx=int(window["nx"]),
            cutover=cutover if cutover.tzinfo else cutover.replace(tzinfo=UTC),
            cadence_seconds=int(cfg.cadence_seconds or 900),
        )
        schema.check_grid(cfg)
        return schema

    def check_grid(self, cfg: ProductConfig) -> None:
        """The store's grid must be exactly the window: a config edit that moves one
        without the other would shift every pixel."""
        if not (0 <= self.row0 and self.row0 + self.ny <= self.ysize
                and 0 <= self.col0 and self.col0 + self.nx <= self.xsize):
            raise ValueError(f"{cfg.store_id}: window {self.window} is not inside the "
                             f"{self.ysize}x{self.xsize} native grid")
        if not cfg.grid_id:
            raise ValueError(f"{cfg.store_id}: no grid_id")
        grid = grids.get(cfg.grid_id)
        if grid.shape != (self.ny, self.nx):
            raise ValueError(f"{cfg.store_id}: grid {grid.id} is {grid.shape}, the window "
                             f"is {(self.ny, self.nx)}")
        (x0, x1), (y0, y1) = grid.projected_extent()
        want = ((self.col0 * self.xscale, (self.col0 + self.nx) * self.xscale),
                (-(self.row0 + self.ny) * self.yscale, -self.row0 * self.yscale))
        if not np.allclose([x0, x1, y0, y1], [*want[0], *want[1]], atol=1e-3):
            raise ValueError(f"{cfg.store_id}: grid {grid.id} extent {((x0, x1), (y0, y1))} "
                             f"is not the window's {want}")
        if grid.proj4.split() != self.projdef.split():
            raise ValueError(f"{cfg.store_id}: grid {grid.id} proj4 {grid.proj4!r} is not "
                             f"the files' {self.projdef!r}")

    @property
    def window(self) -> dict[str, int]:
        return {"row0": self.row0, "row1": self.row0 + self.ny,
                "col0": self.col0, "col1": self.col0 + self.nx}


@dataclass(slots=True)
class FileCheck:
    """One file's check. Field names follow `knmi_radar_3d.FileCheck`, which the stream reads."""

    name: str
    time: datetime | None = None
    chain: str | None = None
    nodes: list[str] = field(default_factory=list)
    window_start: datetime | None = None
    window_end: datetime | None = None
    fingerprint: str | None = None
    storage: str | None = None
    errors: list[str] = field(default_factory=list)
    descriptor: dict[str, Any] = field(default_factory=dict)
    #: Neither offered nor used: the values are physical (gain 1, offset 0).
    scale_factor: float | None = None
    add_offset: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _attrs(obj) -> dict[str, Any]:
    out = {}
    for key, value in obj.attrs.items():
        if isinstance(value, bytes | np.bytes_):
            value = value.decode("utf-8", "replace")
        elif isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def _odim_time(date, time_) -> datetime | None:
    try:
        return datetime.strptime(f"{date}{time_}", "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return None


def _chain_by_structure(h5) -> str | None:
    """Which chain wrote this file, from its layout alone (not its name)."""
    if "dataset2" in h5 and "dataset1/data1/quality1" not in h5:
        return ODYSSEY
    if "dataset1/data1/quality1" in h5 and "dataset2" not in h5:
        return NIMBUS
    return None


def _data_paths(h5) -> list[str]:
    paths: list[str] = []

    def visit(name, obj):
        if hasattr(obj, "shape") and hasattr(obj, "dtype"):
            paths.append(name)
    h5.visititems(visit)
    return sorted(paths)


def check_schema(h5, name: str, schema: Schema, expected_chain: str | None = None) -> FileCheck:
    """Check one open file. Never raises on content; collects every error.

    `expected_chain` is the chain the cutover rule names for this time step;
    a file whose name or structure says otherwise fails.
    """
    check = FileCheck(name=name)
    err = check.errors
    parsed = parse_filename(name)
    product = None
    if parsed is None:
        err.append(f"filename {name!r} is not OPERA@YYYYMMDDTHHMM@0@<PRODUCT>.h5")
    else:
        check.time, product = parsed
        if (check.time.hour * 3600 + check.time.minute * 60) % schema.cadence_seconds:
            err.append(f"{name}: timestamp is off the {schema.cadence_seconds} s cadence")
    named = next((c for c, p in PRODUCTS.items() if p == product), None)
    if product is not None and named is None:
        err.append(f"product {product!r} is not a rain-rate product {sorted(PRODUCTS.values())}")
    structural = _chain_by_structure(h5)
    check.chain = named
    if structural is None:
        err.append("layout is neither ODYSSEY (dataset1 + dataset2) nor NIMBUS "
                   "(dataset1/data1 + quality1)")
    elif named is not None and structural != named:
        err.append(f"name says {named} ({product}) but the file is laid out as {structural}")
    if expected_chain is not None and named is not None and named != expected_chain:
        err.append(f"{name}: {named} file for a time step the cutover assigns to "
                   f"{expected_chain}")

    root = _attrs(h5)
    what = _attrs(h5["what"]) if "what" in h5 else {}
    where = _attrs(h5["where"]) if "where" in h5 else {}
    how = _attrs(h5["how"]) if "how" in h5 else {}
    chain = named or structural
    if what.get("object") != "COMP":
        err.append(f"/what/object {what.get('object')!r}, expected 'COMP'")
    stamp = _odim_time(what.get("date"), what.get("time"))
    if stamp is None:
        err.append(f"/what date/time {what.get('date')!r} {what.get('time')!r} unreadable")
    elif check.time is not None and stamp != check.time:
        err.append(f"/what time {stamp.isoformat()} differs from the filename's "
                   f"{check.time.isoformat()}")
    if chain and what.get("version") != VERSIONS[chain]:
        err.append(f"/what/version {what.get('version')!r}, expected {VERSIONS[chain]!r} "
                   f"for {chain}")

    if str(where.get("projdef", "")).split() != schema.projdef.split():
        err.append(f"projdef {where.get('projdef')!r}, expected {schema.projdef!r}")
    for key, want in (("xsize", schema.xsize), ("ysize", schema.ysize),
                      ("xscale", schema.xscale), ("yscale", schema.yscale)):
        got = where.get(key)
        if got is None or abs(float(got) - float(want)) > 1e-6:
            err.append(f"/where/{key} {got}, expected {want}")
    tol = schema.corner_tolerance_deg
    for corner, (lon, lat) in (("UL", schema.ul_lonlat), ("LR", schema.lr_lonlat)):
        got = (where.get(f"{corner}_lon"), where.get(f"{corner}_lat"))
        if None in got or abs(float(got[0]) - lon) > tol or abs(float(got[1]) - lat) > tol:
            err.append(f"/where {corner} corner {got}, expected ({lon}, {lat}) within {tol} deg")

    # The rain rate: dataset1, with its calibration where each chain keeps it.
    meta: dict[str, Any] = {}
    data = None
    if "dataset1/data1/data" in h5:
        data = h5["dataset1/data1/data"]
        if "dataset1/what" in h5:
            meta.update(_attrs(h5["dataset1/what"]))
        if "dataset1/data1/what" in h5:
            meta.update(_attrs(h5["dataset1/data1/what"]))
    else:
        err.append("no dataset1/data1/data")
    if meta.get("quantity") != "RATE":
        err.append(f"dataset1 quantity {meta.get('quantity')!r}, expected 'RATE'")
    for key, want in (("gain", schema.gain), ("offset", schema.offset),
                      ("nodata", schema.nodata), ("undetect", schema.undetect)):
        got = meta.get(key)
        if got is None or float(got) != want:
            err.append(f"dataset1 {key} {got}, expected {want}")
    if data is not None:
        if data.dtype != np.dtype("float64"):
            err.append(f"RATE dtype {data.dtype}, expected float64")
        if tuple(data.shape) != (schema.ysize, schema.xsize):
            err.append(f"RATE shape {tuple(data.shape)}, expected {(schema.ysize, schema.xsize)}")
        check.storage = (f"chunks={'x'.join(str(c) for c in data.chunks or ())} "
                         f"{data.compression}{data.compression_opts or ''}"
                         f"{' shuffle' if data.shuffle else ''}")
    if chain == ODYSSEY and "dataset2/what" in h5:
        q = _attrs(h5["dataset2/what"]).get("quantity")
        if q != "QIND":
            err.append(f"dataset2 quantity {q!r}, expected 'QIND'")

    check.window_start = _odim_time(meta.get("startdate"), meta.get("starttime"))
    check.window_end = _odim_time(meta.get("enddate"), meta.get("endtime"))
    check.nodes = parse_nodes(how.get("nodes"))
    if not check.nodes:
        err.append("/how/nodes is missing or empty: no contributing radar named")

    def offset_s(t):
        return None if t is None or check.time is None else int((t - check.time).total_seconds())

    check.descriptor = {
        "chain": chain,
        "conventions": root.get("Conventions"),
        "version": what.get("version"),
        "object": what.get("object"),
        "product": meta.get("product"),
        "prodname": meta.get("prodname"),
        "quantity": meta.get("quantity"),
        "datasets": _data_paths(h5),
        "dtype": None if data is None else str(data.dtype),
        "shape": None if data is None else list(data.shape),
        "gain": meta.get("gain"), "offset": meta.get("offset"),
        "nodata": meta.get("nodata"), "undetect": meta.get("undetect"),
        "projdef": where.get("projdef"),
        "grid": [where.get(k) for k in ("xsize", "ysize", "xscale", "yscale")],
        "ul_lonlat": [round(float(where[k]), 4) for k in ("UL_lon", "UL_lat") if k in where],
        "window_offsets_s": [offset_s(check.window_start), offset_s(check.window_end)],
    }
    check.fingerprint = _fingerprint(check.descriptor)
    return check


def _fingerprint(descriptor: Mapping[str, Any]) -> str:
    blob = json.dumps(descriptor, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ------------------------------------------------------------------ reading


def read_window(data, row0: int, row1: int, col0: int, col1: int) -> np.ndarray:
    """``data[row0:row1, col0:col1]``, inflating only the chunks it touches.

    Fast path: the gzip chunks are read raw and inflated with zlib, which
    releases the GIL, so a thread pool packs files in parallel (h5py's own read
    holds a global lock through the decompression). Any other filter pipeline,
    or a chunk with a skipped filter, goes through h5py.
    """
    ny, nx = data.shape
    fast = (data.chunks is not None and data.compression == "gzip" and not data.shuffle
            and not data.fletcher32 and data.scaleoffset is None
            and set(data._filters) == {"gzip"})
    if not fast:
        return np.asarray(data[row0:row1, col0:col1])
    cy, cx = data.chunks
    out = np.empty((row1 - row0, col1 - col0), dtype=data.dtype)
    for y0 in range(row0 // cy * cy, row1, cy):
        for x0 in range(col0 // cx * cx, col1, cx):
            ys, ye = max(row0, y0), min(row1, y0 + cy, ny)
            xs, xe = max(col0, x0), min(col1, x0 + cx, nx)
            mask, blob = data.id.read_direct_chunk((y0, x0))
            if mask != 0:
                out[ys - row0:ye - row0, xs - col0:xe - col0] = data[ys:ye, xs:xe]
                continue
            chunk = np.frombuffer(zlib.decompress(blob), dtype=data.dtype).reshape(cy, cx)
            out[ys - row0:ye - row0, xs - col0:xe - col0] = \
                chunk[ys - y0:ye - y0, xs - x0:xe - x0]
    return out


@dataclass(slots=True)
class ValueStats:
    """What a frame held, counted before encoding: kept in provenance.parquet."""

    n_nodata: int = 0
    n_undetect: int = 0
    n_rain: int = 0
    n_nan: int = 0
    n_negative: int = 0
    n_clipped: int = 0
    max_rate: float | None = None


def encode_rate(values: np.ndarray, target: Encoding, *, nodata: float,
                undetect: float) -> tuple[np.ndarray, ValueStats]:
    """mm/h (float64) -> target codes. nodata -> out_of_image, undetect -> 0.

    A NaN pixel is treated as nodata and a negative rate as 0 (neither seen in
    the archive so far); both are counted. A rate above the encodable maximum
    (655.33 mm/h for uint16 at 0.01) is clipped to it and counted.
    """
    raw = np.asarray(values, dtype="float64")
    nan = ~np.isfinite(raw)
    nod = (raw == nodata) | nan
    und = raw == undetect
    rain = ~(nod | und)
    stats = ValueStats(n_nodata=int(nod.sum()), n_undetect=int(und.sum()),
                       n_rain=int(rain.sum()), n_nan=int(nan.sum()))
    if rain.any():
        vals = raw[rain]
        stats.n_negative = int((vals < 0).sum())
        top = (min(target.reserved) - 1) * target.scale + target.offset
        stats.n_clipped = int((vals > top + target.scale / 2).sum())
        stats.max_rate = float(vals.max())
    clean = np.where(nan, nodata, raw)
    codes = encode(clean, target, source_scale=1.0, source_offset=0.0,
                   source_out_of_image={int(nodata)}, source_no_echo={int(undetect)},
                   no_echo=0.0)
    return codes, stats


@dataclass(slots=True)
class PackedFile:
    check: FileCheck
    codes: np.ndarray | None  # (ny, nx) target dtype, the store's window; None if not packable
    stats: ValueStats = field(default_factory=ValueStats)
    read_s: float = 0.0


def pack_file(path: Path, schema: Schema, target: Encoding,
              expected_chain: str | None = None) -> PackedFile:
    """Check, read the window of, and encode one file. A failed check yields no codes."""
    import h5py

    t0 = time.perf_counter()
    name = Path(path).name
    try:
        with h5py.File(path, "r") as h5:
            check = check_schema(h5, name, schema, expected_chain)
            if not check.ok:
                return PackedFile(check, None, read_s=time.perf_counter() - t0)
            w = schema.window
            values = read_window(h5["dataset1/data1/data"], w["row0"], w["row1"],
                                 w["col0"], w["col1"])
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a corrupt frame
        check = FileCheck(name=name)
        parsed = parse_filename(name)
        if parsed:
            check.time = parsed[0]
        check.errors.append(f"unreadable: {type(exc).__name__}: {exc}")
        return PackedFile(check, None, read_s=time.perf_counter() - t0)
    codes, stats = encode_rate(values, target, nodata=schema.nodata, undetect=schema.undetect)
    return PackedFile(check, codes, stats, time.perf_counter() - t0)


def day_slots(day: datetime, cadence: timedelta) -> list[datetime]:
    start = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return [start + i * cadence for i in range(int(timedelta(days=1) / cadence))]


# ------------------------------------------------------------------ stream format


class OperaFormat:
    """`meteopack.jobs.stream_kdp` format ``opera_odim``: see `Knmi3dFormat` there."""

    def __init__(self, cfg: ProductConfig, settings) -> None:
        self.cfg = cfg
        self.settings = settings
        self.schema = Schema.from_config(cfg)
        var = cfg.variables[0]
        enc = Encoding.from_config(var.encoding, name=var.name)
        if enc is None:
            raise ValueError(f"{cfg.store_id}: {var.name} needs an integer encoding")
        self.target = enc
        self.layout_names = LAYOUT_NAMES
        spec = dict(cfg.extra.get("opera") or {})
        pinned = dict(spec.get("pinned_fingerprints") or {})
        self.pinned = frozenset(str(v) for v in pinned.values())
        self.frame_shape = (self.schema.ny, self.schema.nx)
        #: frame variable -> radar id whose presence in /how/nodes it records.
        self.flag_radars = {str(k): str(v).lower()
                            for k, v in (spec.get("flag_radars") or {}).items()}
        declared = {v.name for v in cfg.frame_variables}
        unknown = sorted((set(self.flag_radars) | {"chain", "n_radars"}) - declared)
        if unknown:
            raise ValueError(f"{cfg.store_id}: frame_variables must declare {unknown}")

    def list_day(self, client, day: datetime) -> list:
        return client.list_prefix(self.settings.day_prefix.format(day=day))

    def select(self, files, day: datetime) -> dict[datetime, object]:
        """Exactly one file per time step: the one of the chain the cutover names."""
        out = {}
        for remote in files:
            parsed = parse_filename(remote.filename)
            if parsed is None:
                continue
            when, product = parsed
            if when.date() != day.date():
                continue
            if product != PRODUCTS[chain_for(when, self.schema.cutover)]:
                continue
            out[when] = remote
        return out

    def pack(self, path: Path, when: datetime) -> PackedFile:
        return pack_file(path, self.schema, self.target, chain_for(when, self.schema.cutover))

    def layout(self, packed: PackedFile) -> int:
        return LAYOUT_OF_CHAIN[packed.check.chain]

    def frame_values(self, packed: PackedFile) -> dict:
        nodes = set(packed.check.nodes)
        out = {"chain": CHAIN_CODES[packed.check.chain], "n_radars": len(nodes)}
        out.update({name: radar in nodes for name, radar in self.flag_radars.items()})
        return out

    def provenance(self, packed: PackedFile) -> dict:
        c, s = packed.check, packed.stats
        return {
            "chain": c.chain or "",
            "n_radars": len(c.nodes),
            "nodes": ",".join(c.nodes),
            "window_start": None if c.window_start is None
            else c.window_start.replace(tzinfo=None),
            "window_end": None if c.window_end is None else c.window_end.replace(tzinfo=None),
            "n_nodata": s.n_nodata, "n_undetect": s.n_undetect, "n_rain": s.n_rain,
            "n_nan": s.n_nan, "n_negative": s.n_negative, "n_clipped": s.n_clipped,
            "max_rate_mmh": s.max_rate,
        }

    def store_attrs(self) -> dict:
        s = self.schema
        return {
            "opera_native_grid": {
                "projdef": s.projdef, "xsize": s.xsize, "ysize": s.ysize,
                "xscale": s.xscale, "yscale": s.yscale,
                "ul_corner_xy": [0.0, 0.0], "ul_lonlat": list(s.ul_lonlat),
                "note": "upper-left CORNER of the upper-left pixel at projected (0, 0); "
                        "rows run south",
            },
            "opera_window": {**s.window, "ny": s.ny, "nx": s.nx,
                             "note": "store pixel [i, j] is native pixel "
                                     "[row0 + i, col0 + j]; half-open bounds"},
            "chain_cutover": s.cutover.isoformat(),
        }

    def manifest_extra(self) -> dict:
        enc = self.target
        return {
            "encoding": {"dtype": str(enc.dtype), "scale": enc.scale, "offset": enc.offset,
                         "no_rain_code": 0, "nodata_code": enc.out_of_image,
                         "missing_frame_code": enc.missing,
                         "formula": f"mm/h = {enc.scale:g} * code"},
            "window": self.schema.window,
            "chain_cutover": self.schema.cutover.isoformat(),
            "orientation": "north-up (rows run south), as the source",
            "source": {"endpoint": self.settings.base_url, "bucket": self.settings.bucket,
                       "day_prefix": self.settings.day_prefix,
                       "files": {chain: f"OPERA@YYYYMMDDTHHMM@0@{p}.h5"
                                 for chain, p in PRODUCTS.items()}},
        }


__all__ = [
    "CHAIN_CODES",
    "LAYOUT_NAMES",
    "NIMBUS",
    "ODYSSEY",
    "PRODUCTS",
    "OperaFormat",
    "Schema",
    "chain_for",
    "check_schema",
    "encode_rate",
    "filename_for",
    "pack_file",
    "parse_filename",
    "parse_nodes",
    "read_window",
]
