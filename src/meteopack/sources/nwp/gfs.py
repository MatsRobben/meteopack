"""NOAA GFS 0.25 degree forecasts from the public AWS archive: index, check, decode, regrid.

The archive (``s3://noaa-gfs-bdp-pds``, anonymous, https://noaa-gfs-bdp-pds.s3.amazonaws.com)
holds one GRIB2 file per run and lead, with a text index (``.idx``) of where each
message starts::

    gfs.YYYYMMDD/HH/atmos/gfs.tHHz.pgrb2.0p25.fFFF      from 2021-03-22 12z (GFS v16)
    gfs.YYYYMMDD/HH/gfs.tHHz.pgrb2.0p25.fFFF            before (GFS v15)

plus ``pgrb2b`` files with the less common fields. A file is ~500 MB and only a
few dozen of its ~740 messages are wanted, so nothing here downloads a file:
the index gives each message's byte range, and messages are fetched with HTTP
``Range`` reads (see `meteopack.core.s3.S3Client.get_bytes`) and decoded from
memory. Nothing is written to disk except a message that fails its check
(quarantined by the stream).

An index line is ``n:offset:d=YYYYMMDDHH:VAR:level:step:`` - e.g.
``593:430384572:d=2024060100:PRATE:surface:1 hour fcst:`` - and a message ends
where the next begins. The step says which kind of field it is:

* ``anl`` / ``L hour fcst``: the instantaneous value at the valid time;
* ``A-L hour ave fcst``: the mean over hours A..L. GFS restarts its averaging
  windows every 6 hours, so ``f007`` holds the mean over 6-7 h, ``f008`` over
  6-8 h, ... `hourly_means` turns those into one-hour means.

Every decoded message is checked before it is used (`check_message`): GRIB
edition, the parameter (discipline, category, number) and its level(s) against
the configured signature, the global 0.25 degree grid and its scanning order,
the run's date and time, the step (hours; the lead; instantaneous or average)
and the values. A message that fails is never packed.

Regridding (`Regridder`) is bilinear from the 0.25 degree latitude/longitude
grid (~28 km at 52N) onto the target grid's cell centres, with the weights of
GRIB-masked neighbours (a bitmap: snow fields over sea) dropped and the rest
renormalised; a target cell whose valid neighbours carry less than half the
weight is undefined. Categorical fields - and CPOFP, whose "no precipitation"
value is -50 % - take the nearest neighbour instead, so a 0/1 flag stays 0/1.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np

from meteopack.core.config import ProductConfig
from meteopack.core.encoding import Encoding

ENDPOINT = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
#: The bucket is in the (virtual-hosted) endpoint, so the client is given no bucket.
BUCKET = ""
#: The first run under ``gfs.YYYYMMDD/HH/atmos/`` (GFS v16 went live with it).
V16_FIRST_RUN = datetime(2021, 3, 22, 12, tzinfo=UTC)
#: Run cycle, hours.
CYCLE_HOURS = 6

#: `source_layout_id` codes. 0 and 1 mean what they mean in every stream.
LAYOUT_MISSING = 0
LAYOUT_CORRUPT = 1
LAYOUT_V15 = 2
LAYOUT_V16 = 3
LAYOUT_NAMES: dict[int, str] = {
    LAYOUT_MISSING: "missing",
    LAYOUT_CORRUPT: "corrupt_or_schema_failure",
    LAYOUT_V15: "gfs_v15_flat_path",
    LAYOUT_V16: "gfs_v16_atmos_path",
}

_IDX_LINE = re.compile(r"^(\d+(?:\.\d+)?):(\d+):d=(\d{10}):([^:]+):([^:]*):([^:]*):")
_AVG = re.compile(r"^(\d+)-(\d+) hour ave fcst$")
_INSTANT = re.compile(r"^(anl|\d+ hour fcst)$")


# ------------------------------------------------------------------ archive

def floor_run(when: datetime) -> datetime:
    """The newest run started at or before `when` (00, 06, 12, 18 UTC)."""
    when = when.astimezone(UTC)
    return when.replace(hour=when.hour - when.hour % CYCLE_HOURS, minute=0, second=0,
                        microsecond=0)


def layout_for(run: datetime) -> int:
    return LAYOUT_V16 if run >= V16_FIRST_RUN else LAYOUT_V15


def file_key(run: datetime, lead: int, kind: str = "pgrb2", layout: int | None = None) -> str:
    """The object key of one run's file at `lead` hours; `kind` pgrb2 or pgrb2b."""
    layout = layout_for(run) if layout is None else layout
    sub = "atmos/" if layout == LAYOUT_V16 else ""
    return f"gfs.{run:%Y%m%d}/{run:%H}/{sub}gfs.t{run:%H}z.{kind}.0p25.f{lead:03d}"


def file_name(run: datetime, lead: int, kind: str = "pgrb2") -> str:
    """A run-qualified name for one file, unique across the archive (provenance)."""
    return f"gfs.{run:%Y%m%d%H}.{kind}.0p25.f{lead:03d}"


@dataclass(frozen=True, slots=True)
class IdxEntry:
    """One message: where it is (`end` inclusive; None = to the end of the file)."""

    number: str
    start: int
    end: int | None
    run: str
    var: str
    level: str
    step: str

    @property
    def is_instant(self) -> bool:
        return bool(_INSTANT.match(self.step))

    @property
    def average_window(self) -> tuple[int, int] | None:
        m = _AVG.match(self.step)
        return (int(m.group(1)), int(m.group(2))) if m else None


def parse_idx(text: str) -> list[IdxEntry]:
    """An ``.idx`` file -> its entries in file order. A malformed line raises ValueError."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _IDX_LINE.match(line)
        if not m:
            raise ValueError(f"not a GRIB index line: {line!r}")
        rows.append(m.groups())
    out = []
    for i, (number, start, run, var, level, step) in enumerate(rows):
        nxt = int(rows[i + 1][1]) if i + 1 < len(rows) else None
        if nxt is not None and nxt <= int(start):
            raise ValueError(f"index offsets not increasing at message {number}")
        out.append(IdxEntry(number, int(start), None if nxt is None else nxt - 1,
                            run, var, level, step))
    return out


def find_entry(entries: Iterable[IdxEntry], var: str, level: str, *,
               average: bool = False) -> IdxEntry | None:
    """The first instantaneous (or, with `average`, time-mean) message of var at level."""
    for e in entries:
        if e.var == var and e.level == level and (
                e.average_window is not None if average else e.is_instant):
            return e
    return None


def coalesce(entries: Sequence[IdxEntry], max_gap: int = 0) -> list[list[IdxEntry]]:
    """Group messages into runs of adjacent byte ranges, one ranged GET each.

    Only exactly adjacent messages are merged by default (`max_gap` 0): a gap
    would be bytes downloaded for nothing.
    """
    groups: list[list[IdxEntry]] = []
    for e in sorted(entries, key=lambda e: e.start):
        last = groups[-1][-1] if groups else None
        if last is not None and last.end is not None and e.start - last.end - 1 <= max_gap:
            groups[-1].append(e)
        else:
            groups.append([e])
    return groups


# ------------------------------------------------------------------- fields

@dataclass(frozen=True, slots=True)
class GribSignature:
    """What a message must be: parameter and level(s), as GRIB2 codes.

    ``[discipline, category, number, surface1, value1, surface2, value2]``;
    values are in the surface's own units (Pa for isobaric, m for heights),
    None where a surface has none.
    """

    discipline: int
    category: int
    number: int
    surface1: int
    value1: float | None
    surface2: int
    value2: float | None

    @classmethod
    def parse(cls, spec: Sequence[Any]) -> GribSignature:
        if len(spec) != 7:
            raise ValueError(f"a GRIB signature has 7 entries, got {spec!r}")
        d, c, n, s1, v1, s2, v2 = spec
        return cls(int(d), int(c), int(n), int(s1), None if v1 is None else float(v1),
                   int(s2), None if v2 is None else float(v2))

    def as_list(self) -> list:
        return [self.discipline, self.category, self.number, self.surface1, self.value1,
                self.surface2, self.value2]


@dataclass(frozen=True, slots=True)
class SourceGrid:
    """A regular latitude/longitude grid, first point at (`lat_first`, `lon_first`),
    rows running south (`dlat` > 0 is the step southwards), columns east."""

    ni: int = 1440
    nj: int = 721
    lat_first: float = 90.0
    lon_first: float = 0.0
    dlat: float = 0.25
    dlon: float = 0.25

    @classmethod
    def from_config(cls, spec: Mapping | None) -> SourceGrid:
        return cls(**{k: type(getattr(cls(), k))(v) for k, v in (spec or {}).items()})

    @property
    def is_global(self) -> bool:
        return math.isclose(self.ni * self.dlon, 360.0)


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """One stored field: which message makes it, how it is regridded, scaled and encoded."""

    name: str
    var: str
    level: str
    signature: GribSignature
    encoding: Encoding
    regrid: str = "bilinear"
    #: Multiplies the decoded value (unit conversion), e.g. 3600 for kg m-2 s-1 -> mm/h.
    factor: float = 1.0
    #: Files to look in, in order.
    files: tuple[str, ...] = ("pgrb2", "pgrb2b")

    @property
    def key(self) -> str:
        return f"{self.var}:{self.level}"


def field_specs(cfg: ProductConfig) -> list[FieldSpec]:
    """The fields of a GFS product: `variables` plus ``extra.gfs.fields.<name>``."""
    extra = dict((cfg.extra.get("gfs") or {}).get("fields") or {})
    out = []
    for var in cfg.variables:
        spec = dict(extra.get(var.name) or {})
        if "grib" not in spec:
            raise ValueError(f"{cfg.store_id}: {var.name} has no extra.gfs.fields.{var.name}.grib")
        name, _, level = var.source_key.partition(":")
        enc = Encoding.from_config(var.encoding, name=var.name)
        if enc is None:
            raise ValueError(f"{cfg.store_id}: {var.name} needs an integer encoding")
        regrid = str(spec.get("regrid", "bilinear"))
        if regrid not in ("bilinear", "nearest"):
            raise ValueError(f"{var.name}: regrid must be bilinear or nearest, got {regrid!r}")
        out.append(FieldSpec(
            name=var.name, var=name, level=level, signature=GribSignature.parse(spec["grib"]),
            encoding=enc, regrid=regrid, factor=float(spec.get("factor", 1.0)),
            files=tuple(spec.get("files", ("pgrb2", "pgrb2b")))))
    names = [f.name for f in out]
    if len(set(names)) != len(names):
        raise ValueError(f"{cfg.store_id}: duplicate field names")
    return out


# ------------------------------------------------------------- check, decode

@dataclass(slots=True)
class MessageCheck:
    """The verdict on one message, and what it was."""

    name: str
    errors: list[str] = field(default_factory=list)
    descriptor: dict = field(default_factory=dict)
    fingerprint: str = ""

    @property
    def ok(self) -> bool:
        return not self.errors


def _level(h, prefix: str) -> tuple[int, float | None]:
    import eccodes

    kind = int(eccodes.codes_get(h, f"typeOf{prefix}FixedSurface", int))
    if kind == 255 or eccodes.codes_is_missing(h, f"scaledValueOf{prefix}FixedSurface"):
        return kind, None
    value = eccodes.codes_get(h, f"scaledValueOf{prefix}FixedSurface", int)
    scale = 0 if eccodes.codes_is_missing(h, f"scaleFactorOf{prefix}FixedSurface") else \
        eccodes.codes_get(h, f"scaleFactorOf{prefix}FixedSurface", int)
    return kind, round(value * 10.0 ** -scale, 6)


def check_message(message: bytes, spec: FieldSpec, *, run: datetime, lead: int,
                  average: bool, grid: SourceGrid) -> tuple[MessageCheck, np.ndarray | None,
                                                            tuple[int, int] | None]:
    """Check one GRIB2 message against what it must be; decode it if it passes.

    Returns the check, the values as float64 ``(nj, ni)`` with NaN where the
    message's bitmap masks a point (None if the check failed), and the step
    window ``(startStep, endStep)`` in hours.
    """
    import eccodes

    check = MessageCheck(name=spec.name)
    try:
        h = eccodes.codes_new_from_message(message)
    except Exception as exc:  # noqa: BLE001 - any decoder failure is a failed check
        check.errors.append(f"not a GRIB message: {type(exc).__name__}: {exc}")
        return check, None, None
    try:
        def get(key, ktype=None):
            try:
                return eccodes.codes_get(h, key, ktype) if ktype else eccodes.codes_get(h, key)
            except Exception:  # noqa: BLE001 - absent key
                return None

        try:
            s1, v1 = _level(h, "First")
            s2, v2 = _level(h, "Second")
        except Exception as exc:  # noqa: BLE001 - a message without them is not ours
            check.errors.append(f"no fixed-surface keys: {type(exc).__name__}: {exc}")
            return check, None, None
        d = {
            "edition": get("edition", int),
            "discipline": get("discipline", int),
            "parameterCategory": get("parameterCategory", int),
            "parameterNumber": get("parameterNumber", int),
            "surface1": s1, "value1": v1, "surface2": s2, "value2": v2,
            "stepType": get("stepType", str),
            "stepUnits": get("stepUnits", int),
            "startStep": get("startStep", int),
            "endStep": get("endStep", int),
            "dataDate": get("dataDate", int),
            "dataTime": get("dataTime", int),
            "gridType": get("gridType", str),
            "Ni": get("Ni", int), "Nj": get("Nj", int),
            "lat_first": get("latitudeOfFirstGridPointInDegrees", float),
            "lon_first": get("longitudeOfFirstGridPointInDegrees", float),
            "dlat": get("jDirectionIncrementInDegrees", float),
            "dlon": get("iDirectionIncrementInDegrees", float),
            "jScansPositively": get("jScansPositively", int),
            "iScansNegatively": get("iScansNegatively", int),
            "packingType": get("packingType", str),
            "bitmapPresent": get("bitmapPresent", int),
            "shortName": get("shortName", str),
            "units": get("units", str),
            "decimalScaleFactor": get("decimalScaleFactor", int),
            "binaryScaleFactor": get("binaryScaleFactor", int),
        }
        check.descriptor = d
        structural = {k: d[k] for k in ("edition", "discipline", "parameterCategory",
                                        "parameterNumber", "surface1", "value1", "surface2",
                                        "value2", "stepType", "gridType", "Ni", "Nj",
                                        "lat_first", "lon_first", "dlat", "dlon",
                                        "jScansPositively", "iScansNegatively")}
        check.fingerprint = hashlib.sha1(json.dumps(structural, sort_keys=True).encode(),
                                         usedforsecurity=False).hexdigest()[:16]
        e = check.errors
        sig = spec.signature
        if d["edition"] != 2:
            e.append(f"GRIB edition {d['edition']}, expected 2")
        got = (d["discipline"], d["parameterCategory"], d["parameterNumber"], s1, v1, s2, v2)
        want = (sig.discipline, sig.category, sig.number, sig.surface1, sig.value1,
                sig.surface2, sig.value2)
        if got != want:
            e.append(f"parameter/level {list(got)}, expected {list(want)}")
        if d["gridType"] != "regular_ll" or (d["Ni"], d["Nj"]) != (grid.ni, grid.nj):
            e.append(f"grid {d['gridType']} {d['Ni']}x{d['Nj']}, expected regular_ll "
                     f"{grid.ni}x{grid.nj}")
        for key, want_v in (("lat_first", grid.lat_first), ("lon_first", grid.lon_first),
                            ("dlat", grid.dlat), ("dlon", grid.dlon)):
            if d[key] is None or not math.isclose(d[key], want_v, abs_tol=1e-6):
                e.append(f"{key} {d[key]}, expected {want_v}")
        if d["jScansPositively"] != 0 or d["iScansNegatively"] != 0:
            e.append("scanning mode is not north-to-south, west-to-east")
        if d["dataDate"] != int(f"{run:%Y%m%d}") or d["dataTime"] != run.hour * 100:
            e.append(f"run {d['dataDate']} {d['dataTime']:04d}, expected {run:%Y%m%d %H%M}"
                     if d["dataTime"] is not None else "no run time")
        if d["stepUnits"] != 1:
            e.append(f"step units {d['stepUnits']}, expected hours")
        if d["endStep"] != lead:
            e.append(f"step ends at {d['endStep']} h, expected the lead {lead} h")
        if average:
            if d["stepType"] != "avg" or not (d["startStep"] is not None
                                              and d["startStep"] < lead):
                e.append(f"stepType {d['stepType']} {d['startStep']}-{d['endStep']}, "
                         f"expected a time mean ending at {lead} h")
        elif d["stepType"] != "instant":
            e.append(f"stepType {d['stepType']}, expected instant")
        if e:
            return check, None, None
        values = eccodes.codes_get_values(h).astype("float64", copy=False)
        if values.size != grid.ni * grid.nj:
            e.append(f"{values.size} values, expected {grid.ni * grid.nj}")
            return check, None, None
        if d["bitmapPresent"]:
            missing = eccodes.codes_get(h, "missingValue", float)
            values = np.where(values == missing, np.nan, values)
        elif not np.isfinite(values).all():
            e.append("non-finite values without a bitmap")
            return check, None, None
        if np.isnan(values).all():
            e.append("every value masked")
            return check, None, None
        return check, values.reshape(grid.nj, grid.ni), (d["startStep"], d["endStep"])
    finally:
        eccodes.codes_release(h)


# -------------------------------------------------------------------- regrid

class Regridder:
    """Lat/lon source grid -> target cell centres, bilinear or nearest neighbour.

    The weights are computed once: four source indices and weights per target
    cell. `bilinear` drops neighbours that are NaN (GRIB bitmap) and
    renormalises the rest; a cell whose valid neighbours weigh less than
    `min_weight` is NaN.
    """

    def __init__(self, lon: np.ndarray, lat: np.ndarray, grid: SourceGrid,
                 min_weight: float = 0.5) -> None:
        lon = np.asarray(lon, dtype="float64")
        lat = np.asarray(lat, dtype="float64")
        self.shape = lon.shape
        self.grid = grid
        self.min_weight = min_weight
        fy = (grid.lat_first - lat) / grid.dlat
        dx = (lon - grid.lon_first) % 360.0 if grid.is_global else lon - grid.lon_first
        fx = dx / grid.dlon
        i0 = np.floor(fy).astype("int64")
        j0 = np.floor(fx).astype("int64")
        wy = fy - i0
        wx = fx - j0
        i1 = i0 + 1
        j1 = (j0 + 1) % grid.ni if grid.is_global else j0 + 1
        if (i0.min() < 0 or i1.max() >= grid.nj or j0.min() < 0
                or (not grid.is_global and j1.max() >= grid.ni)):
            raise ValueError("target grid reaches outside the source grid")
        flat = lambda i, j: (i * grid.ni + j).ravel()
        self.index = np.stack([flat(i0, j0), flat(i0, j1), flat(i1, j0), flat(i1, j1)])
        self.weight = np.stack([((1 - wy) * (1 - wx)).ravel(), ((1 - wy) * wx).ravel(),
                                (wy * (1 - wx)).ravel(), (wy * wx).ravel()])
        ni_ = np.rint(fy).astype("int64")
        nj_ = np.rint(fx).astype("int64") % grid.ni if grid.is_global else np.rint(fx)
        self.nearest_index = flat(ni_, nj_.astype("int64"))

    def bilinear(self, values: np.ndarray) -> np.ndarray:
        src = np.asarray(values, dtype="float64").ravel()
        corners = src[self.index]
        valid = np.isfinite(corners)
        w = np.where(valid, self.weight, 0.0)
        total = w.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out = (np.where(valid, corners, 0.0) * w).sum(axis=0) / total
        out[total < self.min_weight - 1e-9] = np.nan
        return out.reshape(self.shape)

    def nearest(self, values: np.ndarray) -> np.ndarray:
        return np.asarray(values, dtype="float64").ravel()[self.nearest_index].reshape(self.shape)

    def __call__(self, values: np.ndarray, method: str = "bilinear") -> np.ndarray:
        return self.nearest(values) if method == "nearest" else self.bilinear(values)


# ------------------------------------------------------------ derive, encode

def hourly_means(averages: Mapping[int, tuple[int, np.ndarray]]) -> dict[int, np.ndarray]:
    """Time means since a window start -> one-hour means.

    `averages` maps lead L -> (A, mean over hours A..L), as GFS writes them:
    the window restarts every 6 h, so f001..f006 average from 0 and f007..f012
    from 6. The mean over L-1..L is

        m(A, L)                                   when A = L - 1
        (L - A) * m(A, L) - (L - 1 - A) * m(A, L - 1)   otherwise,

    which needs the previous lead from the same window. The window start is
    taken from each message (its startStep), not assumed. A lead whose
    previous lead is absent or from another window is left out.
    """
    out: dict[int, np.ndarray] = {}
    for lead, (start, mean) in averages.items():
        if start == lead - 1:
            out[lead] = np.asarray(mean, dtype="float64")
            continue
        prev = averages.get(lead - 1)
        if prev is None or prev[0] != start or start >= lead - 1:
            continue
        out[lead] = (lead - start) * np.asarray(mean, dtype="float64") \
            - (lead - 1 - start) * np.asarray(prev[1], dtype="float64")
    return out


@dataclass(frozen=True, slots=True)
class Encoded:
    codes: np.ndarray
    n_below: int
    n_above: int
    n_undefined: int


def encode_field(values: np.ndarray, enc: Encoding) -> Encoded:
    """Physical values -> integer codes; NaN (undefined here) -> `out_of_image`.

    Values outside the encodable range are clipped to its ends and counted.
    """
    x = np.asarray(values, dtype="float64")
    undefined = ~np.isfinite(x)
    ceiling = min(enc.reserved) - 1
    lowest = int(np.iinfo(enc.dtype).min)
    with np.errstate(invalid="ignore"):
        q = np.rint((x - enc.offset) / enc.scale)
    below = (q < lowest) & ~undefined
    above = (q > ceiling) & ~undefined
    codes = np.clip(np.where(undefined, lowest, q), lowest, ceiling).astype(enc.dtype)
    codes[undefined] = enc.out_of_image
    return Encoded(codes, int(below.sum()), int(above.sum()), int(undefined.sum()))


def run_times(day: datetime) -> list[datetime]:
    """The four runs started on a UTC day."""
    day = day.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return [day + timedelta(hours=h) for h in range(0, 24, CYCLE_HOURS)]


__all__ = ["BUCKET", "ENDPOINT", "LAYOUT_NAMES", "Encoded", "FieldSpec", "GribSignature",
           "IdxEntry", "MessageCheck", "Regridder", "SourceGrid", "check_message", "coalesce",
           "encode_field", "field_specs", "file_key", "file_name", "find_entry", "floor_run",
           "hourly_means", "layout_for", "parse_idx", "run_times"]
