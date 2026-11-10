"""KNMI HARMONIE-AROME Cy43 NL forecasts from the KNMI Data Platform, on their own grid.

Streamed day by day (24 hourly runs) by ``meteopack stream knmi_harmonie
--product cy43_nl_4km`` (meteopack.jobs.stream_kdp with the ``harmonie_cy43``
format below). One model run exists in two packagings on the KDP, and each run
is taken from exactly one of them (``extra.harmonie.p1_from``):

* ``uwcw_extra_lv_ha43_nl_2km`` v1.0 - NetCDF4, one file per parameter per run,
  ``uwcw_ha43_nl_2km_<param>_<YYYYMMDDTHH>.nc``, 60 steps +1..+60 h (no +0),
  variables ``(time, level, latitude, longitude)`` float32 with NaN fill. Runs
  from 2024-06-25T06. Nine files per run here (the wind components share one).
* ``harmonie_arome_cy43_p1`` v1.0 - one tar per run,
  ``HARM43_V1_P1_<YYYYMMDDHH>.tar``, of 61 GRIB1 files
  ``HA43_N20_<YYYYMMDDHH>00_<LLL>00_GB`` (+0..+60 h, not in lead order). Runs
  from 2026-01-07T22. Only the needed members are fetched, by HTTP Range reads
  of the tar (meteopack.core.remote_tar): one API request a run.

Both are on the same 390 x 390 regular lat/lon grid (lat 49.000-56.002 by
0.018, lon 0.000-11.281 by 0.029, rows south to north). Facts that each
corrupt a store if missed (all checked on run 2026-01-08 00Z, where every field
agrees between the two after conversion to within the GRIB packing precision):

* NetCDF units: MSLP hPa, T degC, RH a fraction (units ``-``), winds and gust
  in **knots**, cloud base in **feet**. GRIB: Pa, K, fraction, m/s, m.
* The NetCDF gust is a speed; the GRIB gust is two components
  (162/163, timeRangeIndicator 2 = max over the past hour): speed = hypot.
* GRIB precipitation (61, TRI 4) is accumulated since the run start; the 1 h
  amount is APCP(L) - APCP(L-1), APCP(0) = 0. The NetCDF holds the 1 h amount.
* Cloud base is NaN in the NetCDF and outside the GRIB bitmap where a column
  has no cloud base - "cloud-free", not missing data.
* ecCodes knows none of the GRIB parameters by name (local table 253): fields
  are matched on (indicatorOfParameter, indicatorOfTypeOfLevel, level,
  timeRangeIndicator), and :mod:`meteopack.sources.nwp.grib1` decodes them.

Every file is checked before it is packed (:func:`check_netcdf_field`,
:func:`read_p1_members`): names, run time, step times, grid, units, levels,
GRIB keys and step ranges, and converted values against the variable's
``valid_min``/``valid_max`` (which catches a unit change). A NetCDF file that
fails is quarantined and its field left missing (the run is *partial*); a P1
run with any failing member is not packed at all.

The fields stay on HARMONIE's own regular lat/lon grid - nothing is
reprojected; a consumer regrids at load time - optionally coarsened by an
integer factor with block means (:func:`block_reduce`; the gust takes the
block maximum), turned north-up, and encoded to 16 bits with the variables'
fixed calibrations.

Lead 0: P1 has a +0 member, the NetCDF does not. NetCDF runs therefore hold
the missing code at lead 0 for every field; P1 runs hold +0 for every field
except the two defined over the past hour (gust, 1 h precipitation), which
have no past hour inside the run and are missing at lead 0 everywhere.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import re
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from meteopack.core import grids, remote_tar
from meteopack.core.config import ProductConfig
from meteopack.core.encoding import Encoding, encode
from meteopack.core.http import IntegrityError
from meteopack.sources.base import RawFile
from meteopack.sources.nwp import grib1

logger = logging.getLogger(__name__)

NETCDF = "netcdf"
P1 = "p1"
#: `source_dataset` frame-variable codes (0 = no run).
SOURCE_CODES = {NETCDF: 1, P1: 2}

#: `source_layout_id` codes: 0 and 1 are the stream's own (missing, corrupt).
LAYOUT_MISSING = 0
LAYOUT_CORRUPT = 1
LAYOUT_NETCDF = 2
LAYOUT_P1 = 3
LAYOUT_NAMES: dict[int, str] = {
    LAYOUT_MISSING: "missing",
    LAYOUT_CORRUPT: "corrupt_or_schema_failure",
    LAYOUT_NETCDF: "uwcw_extra_lv_ha43_nl_2km_v1.0_netcdf",
    LAYOUT_P1: "harmonie_arome_cy43_p1_v1.0_grib1",
}
LAYOUT_OF_KIND = {NETCDF: LAYOUT_NETCDF, P1: LAYOUT_P1}

_NC_RUN = re.compile(r"^(?P<param>.+)_(?P<run>\d{8}T\d{2})\.nc$")
_P1_RUN = re.compile(r"^HARM43_V1_P1_(?P<run>\d{10})\.tar$")
_GRIB = b"GRIB"
_HDF5 = b"\x89HDF\r\n\x1a\n"
INDEX_NAME = ".members.json"


# ------------------------------------------------------------------ settings


@dataclass(frozen=True, slots=True)
class FieldSpec:
    """How one store variable is read from each packaging (``extra.harmonie.fields``)."""

    name: str
    nc_param: str
    nc_var: str
    nc_units: str
    nc_level: float
    factor: float
    offset: float
    p1_keys: tuple[tuple[int, int, int, int], ...]
    p1_combine: str
    #: What a NaN source value means: ``missing`` data, or ``cloud_free``.
    undefined: str
    clip_min: float | None
    valid_min: float | None
    valid_max: float | None
    encoding: Encoding
    #: How a coarsening block is reduced: ``mean`` (area mean) or ``max``.
    aggregate: str = "mean"
    #: False for a field defined over the past hour, which has no +0.
    lead0: bool = True


@dataclass(frozen=True, slots=True)
class LatLonGrid:
    """A regular lon/lat grid by its first cell centre, step and size (rows south to north)."""

    lat0: float
    dlat: float
    nlat: int
    lon0: float
    dlon: float
    nlon: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nlat, self.nlon)


@dataclass(frozen=True, slots=True)
class Schema:
    """Everything a run is checked and packed against, from the product config."""

    leads: tuple[int, ...]
    p1_from: datetime
    source: LatLonGrid
    tolerance_deg: float
    #: Block size of the coarsening (1 = the native grid).
    factor: int
    min_valid: float
    nc_prefix: str
    grid_mapping_proj4: str
    p1_dataset: str
    p1_version: str
    p1_tar: str
    p1_member: str
    table2_version: int
    centre: int
    fields: tuple[FieldSpec, ...]
    grid_id: str
    listing_ttl_s: float = 21600.0

    @classmethod
    def from_config(cls, cfg: ProductConfig) -> Schema:
        spec = dict(cfg.extra.get("harmonie") or {})
        if cfg.levels is None or cfg.levels.name != "lead_time":
            raise ValueError(f"{cfg.store_id}: levels must be the lead_time axis")
        leads = tuple(int(v) for v in cfg.levels.values)
        if any(float(v) != int(v) or v < 0 for v in cfg.levels.values):
            raise ValueError(f"{cfg.store_id}: lead times must be whole hours >= 0")
        g = dict(spec["source_grid"])
        nc = dict(spec.get("netcdf") or {})
        p1 = dict(spec.get("p1") or {})
        declared = {v.name: v for v in cfg.variables}
        fields = []
        for name, f in dict(spec["fields"]).items():
            if name not in declared:
                raise ValueError(f"{cfg.store_id}: extra.harmonie.fields names {name!r}, "
                                 f"which is not a variable")
            var = declared[name]
            enc = Encoding.from_config(var.encoding, name=name)
            if enc is None:
                raise ValueError(f"{cfg.store_id}: {name} needs an integer encoding")
            n, q = dict(f["nc"]), dict(f["p1"])
            fields.append(FieldSpec(
                name=name, nc_param=str(n["param"]), nc_var=str(n["var"]),
                nc_units=str(n["units"]), nc_level=float(n.get("level", 0)),
                factor=float(n.get("factor", 1.0)), offset=float(n.get("offset", 0.0)),
                p1_keys=tuple(tuple(int(x) for x in k) for k in q["keys"]),
                p1_combine=str(q.get("combine", "single")),
                undefined=str(f.get("undefined", "missing")),
                clip_min=None if f.get("clip_min") is None else float(f["clip_min"]),
                valid_min=var.valid_min, valid_max=var.valid_max, encoding=enc,
                aggregate=str(f.get("aggregate", "mean")), lead0=bool(f.get("lead0", True))))
            if fields[-1].aggregate not in ("mean", "max"):
                raise ValueError(f"{name}: unknown aggregate {fields[-1].aggregate!r}")
            if fields[-1].p1_combine not in ("single", "hypot", "accumulation"):
                raise ValueError(f"{name}: unknown p1.combine {fields[-1].p1_combine!r}")
            if fields[-1].undefined not in ("missing", "cloud_free"):
                raise ValueError(f"{name}: unknown undefined {fields[-1].undefined!r}")
        if [f.name for f in fields] != list(declared):
            raise ValueError(f"{cfg.store_id}: extra.harmonie.fields must list every variable, "
                             f"in the variables' order")
        p1_from = datetime.fromisoformat(str(spec["p1_from"]))
        return cls(
            leads=leads,
            p1_from=p1_from if p1_from.tzinfo else p1_from.replace(tzinfo=UTC),
            source=LatLonGrid(lat0=float(g["lat0"]), dlat=float(g["dlat"]),
                              nlat=int(g["nlat"]), lon0=float(g["lon0"]),
                              dlon=float(g["dlon"]), nlon=int(g["nlon"])),
            tolerance_deg=float(g.get("tolerance_deg", 1e-4)),
            factor=int(spec.get("coarsen_factor", 1)),
            min_valid=float(spec.get("min_valid", 0.5)),
            nc_prefix=str(nc.get("prefix", "uwcw_ha43_nl_2km_")),
            grid_mapping_proj4=str(nc.get("grid_mapping_proj4", "")),
            p1_dataset=str(p1.get("dataset", "harmonie_arome_cy43_p1")),
            p1_version=str(p1.get("version", "1.0")),
            p1_tar=str(p1.get("tar", "HARM43_V1_P1_{run:%Y%m%d%H}.tar")),
            p1_member=str(p1.get("member", "HA43_N20_{run:%Y%m%d%H}00_{lead:03d}00_GB")),
            table2_version=int(p1.get("table2_version", 253)),
            centre=int(p1.get("centre", 99)),
            fields=tuple(fields),
            grid_id=str(cfg.grid_id),
            listing_ttl_s=float(spec.get("listing_ttl_s", 21600)),
        )

    def check_grid(self) -> None:
        """The store's grid must be exactly the (coarsened) source grid: a config
        edit that changes one without the other would misplace every cell."""
        g, f = self.source, self.factor
        if f < 1:
            raise ValueError(f"coarsen_factor {f} must be >= 1")
        grid = grids.get(self.grid_id)
        ny, nx = g.nlat // f, g.nlon // f
        if grid.shape != (ny, nx):
            raise ValueError(f"grid {grid.id} is {grid.shape}; the source coarsened by {f} is "
                             f"{(ny, nx)}")
        want = ((g.lon0 - g.dlon / 2, g.lon0 - g.dlon / 2 + nx * f * g.dlon),
                (g.lat0 - g.dlat / 2, g.lat0 - g.dlat / 2 + ny * f * g.dlat))
        (x0, x1), (y0, y1) = grid.projected_extent()
        if not np.allclose([x0, x1, y0, y1], [*want[0], *want[1]], atol=1e-6):
            raise ValueError(f"grid {grid.id} extent {((x0, x1), (y0, y1))} is not the "
                             f"coarsened source's {want}")
        if grid.grid_mapping_name != "latitude_longitude":
            raise ValueError(f"grid {grid.id} is not a lat/lon grid")

    @property
    def nc_params(self) -> list[str]:
        """The NetCDF files a run needs, one per parameter (the wind components share one)."""
        return list(dict.fromkeys(f.nc_param for f in self.fields))

    @property
    def nc_leads(self) -> list[int]:
        """The leads the NetCDF holds (+1 onwards)."""
        return [lead for lead in self.leads if lead >= 1]

    @property
    def p1_leads(self) -> list[int]:
        """The P1 members a run needs: every lead, and the one before for an accumulation."""
        need = set(self.leads)
        if any(f.p1_combine == "accumulation" for f in self.fields):
            need |= {lead - 1 for lead in self.leads if lead - 1 >= 1}
        return sorted(need)

    def nc_filename(self, param: str, run: datetime) -> str:
        return f"{self.nc_prefix}{param}_{run:%Y%m%dT%H}.nc"

    def p1_tar_name(self, run: datetime) -> str:
        return self.p1_tar.format(run=run)

    def p1_member_name(self, run: datetime, lead: int) -> str:
        return self.p1_member.format(run=run, lead=lead)

    def parse_nc(self, name: str) -> tuple[str, datetime] | None:
        if not name.startswith(self.nc_prefix):
            return None
        m = _NC_RUN.match(name[len(self.nc_prefix):])
        if not m:
            return None
        return m["param"], datetime.strptime(m["run"], "%Y%m%dT%H").replace(tzinfo=UTC)


def parse_p1_tar(name: str) -> datetime | None:
    m = _P1_RUN.match(name)
    return datetime.strptime(m["run"], "%Y%m%d%H").replace(tzinfo=UTC) if m else None


# ------------------------------------------------------------------ checks


@dataclass(frozen=True, slots=True)
class RunFiles:
    """One run's remote files: what `select` hands the stream for one time step."""

    #: A name for the run as a whole (the provenance row's `filename`).
    filename: str
    size: int
    last_modified: str | None
    kind: str
    run: datetime
    files: tuple = ()
    #: NetCDF: the parameters with a listed file.
    params: tuple[str, ...] = ()


@dataclass(slots=True)
class RunCheck:
    """One run's check. Field names follow `knmi_radar_3d.FileCheck`, which the stream reads."""

    name: str
    time: datetime | None = None
    kind: str = ""
    fingerprint: str | None = None
    storage: str | None = None
    #: Run-level errors: the run is not packed at all.
    errors: list[str] = field(default_factory=list)
    #: File-level errors of a run packed without those files' fields.
    file_errors: dict[str, list[str]] = field(default_factory=dict)
    #: Files of the run to move aside rather than delete.
    quarantine: list[str] = field(default_factory=list)
    descriptor: dict[str, Any] = field(default_factory=dict)
    fields_present: list[str] = field(default_factory=list)
    fields_missing: list[str] = field(default_factory=list)
    scale_factor: float | None = None
    add_offset: float | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def _fingerprint(descriptor: Mapping[str, Any]) -> str:
    blob = json.dumps(descriptor, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _text(value) -> Any:
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


def _range_errors(spec: FieldSpec, values: np.ndarray) -> list[str]:
    finite = values[np.isfinite(values)]
    if not finite.size:
        return []
    lo, hi = float(finite.min()), float(finite.max())
    out = []
    if spec.valid_min is not None and lo < spec.valid_min:
        out.append(f"{spec.name}: minimum {lo:.6g} below {spec.valid_min:g} (units?)")
    if spec.valid_max is not None and hi > spec.valid_max:
        out.append(f"{spec.name}: maximum {hi:.6g} above {spec.valid_max:g} (units?)")
    return out


def _axis_errors(name: str, values: np.ndarray, first: float, step: float, n: int,
                 tol: float) -> list[str]:
    want = first + step * np.arange(n)
    if values.shape != (n,) or not np.allclose(values, want, atol=tol, rtol=0):
        got = (f"{values[:1].tolist()}..{values[-1:].tolist()} (n={values.size})"
               if values.size else "empty")
        return [f"{name} {got}, expected {first}..{want[-1]:.4f} by {step} (n={n})"]
    return []


def check_netcdf_field(h5, filename: str, specs: Sequence[FieldSpec], schema: Schema,
                       run: datetime) -> tuple[dict[str, np.ndarray], dict, list[str]]:
    """Check one NetCDF file and read its fields for the store's leads, converted.

    Returns ``(values by field name, descriptor, errors)``; with errors the
    values are not to be used. Values are float64 ``(n_leads, nlat, nlon)``,
    rows south to north, NaN at a lead the file does not hold (+0).
    """
    err: list[str] = []
    g = schema.source
    n_lead = max(schema.nc_leads)
    attrs = {k: _text(v) for k, v in h5.attrs.items()}
    if attrs.get("cycle_datetime") != f"{run:%Y-%m-%dT%H}":
        err.append(f"cycle_datetime {attrs.get('cycle_datetime')!r}, expected "
                   f"{run:%Y-%m-%dT%H}")
    for key in ("latitude", "longitude", "time", "forecast_reference_time"):
        if key not in h5:
            err.append(f"no {key!r} variable")
    if err:
        return {}, {"variables": sorted(h5.keys())}, err
    lat = np.asarray(h5["latitude"][:], dtype="float64")
    lon = np.asarray(h5["longitude"][:], dtype="float64")
    err += _axis_errors("latitude", lat, g.lat0, g.dlat, g.nlat, schema.tolerance_deg)
    err += _axis_errors("longitude", lon, g.lon0, g.dlon, g.nlon, schema.tolerance_deg)
    t_units = str(_text(h5["time"].attrs.get("units", "")))
    times = np.asarray(h5["time"][:], dtype="int64")
    ref = int(np.asarray(h5["forecast_reference_time"][()]).reshape(-1)[0])
    run_s = int(run.timestamp())
    if not t_units.startswith("seconds since 1970-01-01"):
        err.append(f"time units {t_units!r}")
    if ref != run_s:
        err.append(f"forecast_reference_time {ref} is not the run {run.isoformat()}")
    if times.size < n_lead or not np.array_equal(times[:n_lead],
                                                 run_s + 3600 * np.arange(1, n_lead + 1)):
        err.append(f"time steps are not run + 1..{n_lead} h: first "
                   f"{((times[:3] - run_s) / 3600).tolist()} h of {times.size}")
    gm = h5.get("latitude_longitude")
    proj4 = None if gm is None else _text(gm.attrs.get("proj4_params"))
    if schema.grid_mapping_proj4 and proj4 != schema.grid_mapping_proj4:
        err.append(f"grid mapping {proj4!r}, expected {schema.grid_mapping_proj4!r}")

    descriptor: dict[str, Any] = {
        "lat": [round(float(lat[0]), 4), round(float(lat[-1]), 4), int(lat.size)]
        if lat.size else [],
        "lon": [round(float(lon[0]), 4), round(float(lon[-1]), 4), int(lon.size)]
        if lon.size else [],
        "n_time": int(times.size), "grid_mapping": proj4, "fields": {}}
    out: dict[str, np.ndarray] = {}
    index = [lead - 1 for lead in schema.nc_leads]
    for spec in specs:
        if spec.nc_var not in h5:
            err.append(f"no variable {spec.nc_var!r}")
            continue
        ds = h5[spec.nc_var]
        dims = _dim_names(ds)
        units = _text(ds.attrs.get("units"))
        fill = _text(ds.attrs.get("_FillValue"))
        level_dim = dims[1] if len(dims) == 4 else None
        level = (np.asarray(h5[level_dim][:]).reshape(-1).tolist()
                 if level_dim and level_dim in h5 else None)
        descriptor["fields"][spec.name] = {
            "var": spec.nc_var, "dtype": str(ds.dtype), "shape": list(ds.shape[1:]),
            "dims": [dims[0], "level", *dims[2:]] if len(dims) == 4 else dims,
            "units": units, "standard_name": _text(ds.attrs.get("standard_name")),
            "level": level, "fill_nan": fill is not None and bool(np.isnan(float(fill))),
        }
        bad = []
        if ds.dtype != np.dtype("float32"):
            bad.append(f"{spec.nc_var} dtype {ds.dtype}, expected float32")
        if ds.ndim != 4 or tuple(ds.shape[1:]) != (1, g.nlat, g.nlon):
            bad.append(f"{spec.nc_var} shape {tuple(ds.shape)}, expected (T, 1, {g.nlat}, "
                       f"{g.nlon})")
        elif dims[0] != "time" or dims[2:] != ["latitude", "longitude"]:
            bad.append(f"{spec.nc_var} dims {dims}, expected [time, <level>, latitude, "
                       f"longitude]")
        if units != spec.nc_units:
            bad.append(f"{spec.nc_var} units {units!r}, expected {spec.nc_units!r}")
        if level is None or len(level) != 1 or float(level[0]) != spec.nc_level:
            bad.append(f"{spec.nc_var} level {level}, expected [{spec.nc_level:g}]")
        if bad or err:
            err += bad
            continue
        raw = np.asarray(ds[index[0]:index[-1] + 1, 0], dtype="float64")
        raw = raw[[i - index[0] for i in index]]
        # Every store lead; the ones the NetCDF lacks (+0) stay NaN.
        values = np.full((len(schema.leads), g.nlat, g.nlon), np.nan)
        rows = [schema.leads.index(lead) for lead in schema.nc_leads]
        values[rows] = raw * spec.factor + spec.offset
        rng = _range_errors(spec, values)
        if rng:
            err += rng
            continue
        out[spec.name] = values
    return out, descriptor, err


def read_netcdf_run(directory: Path, files: RunFiles, schema: Schema
                    ) -> tuple[dict[str, np.ndarray], RunCheck]:
    """Check and read every NetCDF file of a run; a failing file costs only its fields."""
    import h5py

    check = RunCheck(name=files.filename, time=files.run, kind=NETCDF)
    values: dict[str, np.ndarray] = {}
    by_param: dict[str, list[FieldSpec]] = {}
    for spec in schema.fields:
        by_param.setdefault(spec.nc_param, []).append(spec)
    descriptors = {}
    storage = set()
    for param, specs in by_param.items():
        name = schema.nc_filename(param, files.run)
        path = Path(directory) / name
        if not path.exists():
            continue  # not listed upstream: the fields stay missing
        try:
            with h5py.File(path, "r") as h5:
                got, descriptor, errors = check_netcdf_field(h5, name, specs, schema, files.run)
                first = h5.get(specs[0].nc_var)
                if first is not None and hasattr(first, "chunks"):
                    storage.add(f"chunks={'x'.join(str(c) for c in first.chunks or ())} "
                                f"{first.compression}{first.compression_opts or ''}")
        except Exception as exc:  # noqa: BLE001 - an unreadable file fails like a bad one
            got, descriptor, errors = {}, {}, [f"unreadable: {type(exc).__name__}: {exc}"]
        descriptors[param] = descriptor
        if errors:
            check.file_errors[name] = errors
            check.quarantine.append(name)
            continue
        values.update(got)
    check.fields_present = [f.name for f in schema.fields if f.name in values]
    check.fields_missing = [f.name for f in schema.fields if f.name not in values]
    check.descriptor = {"kind": NETCDF, "leads": schema.nc_leads, "files": descriptors}
    check.fingerprint = _fingerprint(check.descriptor)
    check.storage = f"netcdf {' / '.join(sorted(storage))}" if storage else "netcdf"
    if not values:
        check.errors.append("no field of the run could be read: " + "; ".join(
            f"{k}: {v}" for k, v in check.file_errors.items()) if check.file_errors
            else "no file of the run is on disk")
    return values, check


def _p1_descriptor(msg: grib1.Message, lead: int) -> dict:
    start, end = msg.step_range
    return {"key": list(msg.key), "step": [start - lead, end - lead],
            "time_unit": msg.time_unit, "grid": msg.grid.describe(),
            "table2": msg.table2_version, "centre": msg.centre}


def read_p1_members(directory: Path, files: RunFiles, schema: Schema
                    ) -> tuple[dict[str, np.ndarray], RunCheck]:
    """Check and decode the P1 members of a run. Any failure fails the whole run."""
    check = RunCheck(name=files.filename, time=files.run, kind=P1)
    run = files.run
    g = schema.source
    needed = {key for spec in schema.fields for key in spec.p1_keys}
    fields: dict[int, dict[tuple, np.ndarray]] = {}
    descriptor: dict[str, Any] = {}
    bits: set[int] = set()
    for lead in schema.p1_leads:
        name = schema.p1_member_name(run, lead)
        path = Path(directory) / name
        if not path.exists():
            check.errors.append(f"member {name} missing")
            continue
        errs: list[str] = []
        try:
            buf = path.read_bytes()
            got: dict[tuple, grib1.Message] = {}
            for msg in grib1.iter_messages(buf):
                if msg.key in needed:
                    if msg.key in got:
                        errs.append(f"{name}: key {msg.key} appears twice")
                    got[msg.key] = msg
            for key in sorted(needed - set(got)):
                errs.append(f"{name}: no message {list(key)}")
            per_key = {}
            for key, msg in sorted(got.items()):
                d = _p1_descriptor(msg, lead)
                per_key[str(list(key))] = d
                bits.add(msg.bits)
                start, end = msg.step_range
                want = {0: (lead, lead), 2: (max(lead - 1, 0), lead),
                        4: (0, lead)}.get(msg.time_range)
                if want is None or (start, end) != want:
                    errs.append(f"{name}: {list(key)} step range {(start, end)}, expected {want}")
                if msg.time_unit != 1:
                    errs.append(f"{name}: {list(key)} time unit {msg.time_unit}, expected hours")
                if (msg.year, msg.month, msg.day, msg.hour, msg.minute) != (
                        run.year, run.month, run.day, run.hour, 0):
                    errs.append(f"{name}: {list(key)} is dated {msg.year}-{msg.month:02d}-"
                                f"{msg.day:02d} {msg.hour:02d}:{msg.minute:02d}, not the run")
                if msg.table2_version != schema.table2_version or msg.centre != schema.centre:
                    errs.append(f"{name}: table2Version/centre {msg.table2_version}/"
                                f"{msg.centre}, expected {schema.table2_version}/"
                                f"{schema.centre}")
                gr = msg.grid
                if (gr.ni, gr.nj) != (g.nlon, g.nlat) or not np.allclose(
                        [gr.lat_first, gr.lon_first, gr.dj, gr.di],
                        [g.lat0, g.lon0, g.dlat, g.dlon], atol=schema.tolerance_deg + 5e-4):
                    errs.append(f"{name}: {list(key)} grid {gr.describe()} is not the source "
                                f"grid")
            if not errs:
                fields[lead] = {key: msg.values() for key, msg in got.items()}
            descriptor[str(lead)] = per_key
        except Exception as exc:  # noqa: BLE001 - unreadable member: the run fails
            errs.append(f"{name}: unreadable: {type(exc).__name__}: {exc}")
        if errs:
            check.errors += errs
            check.quarantine.append(name)
    # The structure, the same for every lead from +1 (steps are relative to the lead).
    structure = descriptor.get("1") or next(iter(descriptor.values()), {})
    check.descriptor = {"kind": P1, "leads": list(schema.leads), "messages": structure}
    check.fingerprint = _fingerprint(check.descriptor)
    check.storage = f"grib1 simple {'/'.join(str(b) for b in sorted(bits))} bit"
    if check.errors:
        return {}, check
    values: dict[str, np.ndarray] = {}
    for spec in schema.fields:
        stack = []
        for lead in schema.leads:
            if lead == 0 and not spec.lead0:
                # Gust and 1 h precipitation are over the past hour: none at +0.
                stack.append(np.full((g.nlat, g.nlon), np.nan))
                continue
            parts = [fields[lead][k] for k in spec.p1_keys]
            if spec.p1_combine == "hypot":
                v = np.hypot(parts[0], parts[1])
            elif spec.p1_combine == "accumulation":
                # APCP is accumulated since the run start, and APCP(0) = 0.
                prev = fields[lead - 1][spec.p1_keys[0]] if lead - 1 >= 1 else 0.0
                v = parts[0] - prev
            else:
                v = parts[0]
            stack.append(v)
        arr = np.stack(stack)
        rng = _range_errors(spec, arr)
        if rng:
            check.errors += rng
            continue
        values[spec.name] = arr
    if check.errors:
        check.quarantine = [schema.p1_member_name(run, lead) for lead in schema.p1_leads]
        return {}, check
    check.fields_present = [f.name for f in schema.fields]
    return values, check


# ------------------------------------------------------------------ packing


@dataclass(slots=True)
class PackedRun:
    check: RunCheck
    #: Field name -> ``(n_leads, ny, nx)`` codes; None if the run is not packable.
    codes: dict[str, np.ndarray] | None
    stats: dict[str, dict] = field(default_factory=dict)
    read_s: float = 0.0


def cloud_free_code(enc: Encoding) -> int:
    """The top ordinary code: "no cloud base" for the cloud-base field."""
    return min(enc.reserved) - 1


def block_reduce(values: np.ndarray, factor: int, how: str = "mean",
                 min_valid: float = 0.5) -> np.ndarray:
    """``(..., nlat, nlon)`` rows south to north -> ``(..., nlat // f, nlon // f)`` north-up.

    Each output cell is the mean (or, ``how="max"``, the maximum) of the
    ``factor x factor`` block of source cells it covers - no interpolation.
    NaN source cells are left out; a block with less than `min_valid` of its
    cells finite is NaN. When `factor` does not divide the grid, the last
    ``nlat % f`` rows (the northern edge) and ``nlon % f`` columns (the
    eastern edge) are dropped rather than averaged over a partial block, so
    every output cell is a full block; the store's grid says so in its extent.
    `factor` 1 only turns the field north-up.
    """
    v = np.asarray(values, dtype="float64")
    f = int(factor)
    if f == 1:
        return v[..., ::-1, :].copy()
    ny, nx = v.shape[-2] // f, v.shape[-1] // f
    blocks = v[..., : ny * f, : nx * f].reshape(*v.shape[:-2], ny, f, nx, f)
    finite = np.isfinite(blocks)
    share = finite.mean(axis=(-3, -1))
    with np.errstate(all="ignore"):
        if how == "max":
            out = np.where(finite, blocks, -np.inf).max(axis=(-3, -1))
        else:
            out = np.where(finite, blocks, 0.0).sum(axis=(-3, -1)) / finite.sum(axis=(-3, -1))
    out[share < min_valid] = np.nan
    return out[..., ::-1, :]


def encode_field(values: np.ndarray, spec: FieldSpec, lead_ok: np.ndarray
                 ) -> tuple[np.ndarray, dict]:
    """Store-grid physical values ``(n_leads, ny, nx)`` -> codes, with what was clipped.

    NaN is `missing`, or "no cloud base" for a field whose undefined value
    means cloud-free; leads the run does not have (`lead_ok` False) are
    `missing` whatever the field.
    """
    enc = spec.encoding
    v = np.asarray(values, dtype="float64")
    if spec.clip_min is not None:
        v = np.where(np.isfinite(v), np.maximum(v, spec.clip_min), v)
    ceiling = cloud_free_code(enc) - (1 if spec.undefined == "cloud_free" else 0)
    info = np.iinfo(enc.dtype)
    q = np.rint((v - enc.offset) / enc.scale)
    finite = np.isfinite(q)
    have = np.broadcast_to(np.asarray(lead_ok, dtype=bool)[:, None, None], v.shape)
    stats = {
        "n_clipped_low": int(((q < info.min) & finite).sum()),
        "n_clipped_high": int(((q > ceiling) & finite).sum()),
        "n_undefined": int((~finite & have).sum()),
        "min": float(np.nanmin(v)) if finite.any() else None,
        "max": float(np.nanmax(v)) if finite.any() else None,
    }
    top = ceiling * enc.scale + enc.offset
    codes = encode(np.where(finite, np.minimum(v, top), np.nan), enc,
                   source_scale=1.0, source_offset=0.0)
    if spec.undefined == "cloud_free":
        codes[~finite & have] = cloud_free_code(enc)
    codes[~have] = enc.missing
    return codes, stats


def lead_available(spec: FieldSpec, kind: str, leads: Sequence[int]) -> np.ndarray:
    """Which of the store's leads a run of this packaging has for this field."""
    return np.array([lead >= 1 or (kind == P1 and spec.lead0) for lead in leads])


def pack_run(directory: Path, files: RunFiles, schema: Schema) -> PackedRun:
    """Check, read, coarsen and encode one run. A run with no usable field yields no codes."""
    t0 = time.perf_counter()
    try:
        if files.kind == P1:
            values, check = read_p1_members(directory, files, schema)
        else:
            values, check = read_netcdf_run(directory, files, schema)
    except Exception as exc:  # noqa: BLE001 - anything unforeseen fails the run, recorded
        check = RunCheck(name=files.filename, time=files.run, kind=files.kind)
        check.errors.append(f"unreadable: {type(exc).__name__}: {exc}")
        return PackedRun(check, None, read_s=time.perf_counter() - t0)
    if not check.ok:
        return PackedRun(check, None, read_s=time.perf_counter() - t0)
    codes: dict[str, np.ndarray] = {}
    stats: dict[str, dict] = {}
    for spec in schema.fields:
        if spec.name not in values:
            continue
        reduced = block_reduce(values[spec.name], schema.factor, spec.aggregate,
                               schema.min_valid)
        codes[spec.name], stats[spec.name] = encode_field(
            reduced, spec, lead_available(spec, files.kind, schema.leads))
    return PackedRun(check, codes, stats, time.perf_counter() - t0)


# ------------------------------------------------------------------ download


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(f".{path.name}.part")
    tmp.write_bytes(data)
    tmp.replace(path)


def download_p1(client, files: RunFiles, directory: Path, schema: Schema,
                reader_factory=remote_tar.RangeReader) -> tuple[int, int]:
    """Fetch the run's needed tar members by Range reads; returns (bytes, fresh members).

    One API request (the tar's temporary URL); the header walk and the members
    go to S3 and cost no quota. The member index is kept beside the members so
    a resumed run does not walk the tar again; members already on disk with
    the indexed size are kept.
    """
    run = files.run
    tar = files.files[0]
    wanted = {schema.p1_member_name(run, lead) for lead in schema.p1_leads}
    index_path = directory / INDEX_NAME
    index: dict[str, list[int]] = {}
    with contextlib.suppress(OSError, ValueError):
        index = {k: list(v) for k, v in json.loads(index_path.read_text()).items()}

    def done(name: str) -> bool:
        p = directory / name
        return name in index and p.exists() and p.stat().st_size == index[name][1]

    if wanted and all(done(n) for n in wanted):
        return 0, 0
    t0 = time.perf_counter()
    url = client.download_url(tar.filename, dataset=schema.p1_dataset, version=schema.p1_version)
    t1 = time.perf_counter()
    reader = reader_factory(url, session=getattr(client, "download_session", None),
                            total_size=tar.size)
    if not wanted <= set(index):
        members = remote_tar.list_members(
            reader.read, tar.size, stop=lambda found: wanted <= {m.name for m in found})
        index = {m.name: [m.offset, m.size] for m in members}
        _write_atomic(index_path, json.dumps(index).encode())
    absent = sorted(wanted - set(index))
    if absent:
        raise IntegrityError(f"{tar.filename}: no member(s) {absent}")
    n_bytes = fresh = 0
    for name in sorted(wanted):
        if done(name):
            continue
        offset, size = index[name]
        data = reader.read(offset, size)
        if data[:4] != _GRIB:
            raise IntegrityError(f"{tar.filename}/{name}: does not start with GRIB")
        _write_atomic(directory / name, data)
        n_bytes += len(data)
        fresh += 1
    stats = getattr(client, "stats", None)
    lock = getattr(client, "_stats_lock", None)
    if isinstance(stats, dict) and lock is not None:
        with lock:
            stats["url_s"] = stats.get("url_s", 0.0) + (t1 - t0)
            stats["transfer_s"] = stats.get("transfer_s", 0.0) + (time.perf_counter() - t1)
            stats["files"] = stats.get("files", 0) + 1
            stats["bytes"] = stats.get("bytes", 0) + n_bytes
    return n_bytes, fresh


# ------------------------------------------------------------------ stream format


class HarmonieFormat:
    """`meteopack.jobs.stream_kdp` format ``harmonie_cy43``: see `Knmi3dFormat` there.

    A time step is a run; its "file" is a bundle (`RunFiles`) fetched into one
    directory by :meth:`download` and packed by :func:`pack_run` into every
    field at once (``variables``).
    """

    def __init__(self, cfg: ProductConfig, settings) -> None:
        self.cfg = cfg
        self.settings = settings
        self.schema = Schema.from_config(cfg)
        self.schema.check_grid()
        grid = grids.get(cfg.grid_id)
        self.variables: dict[str, Encoding] = {f.name: f.encoding for f in self.schema.fields}
        self.target = self.schema.fields[0].encoding
        self.layout_names = LAYOUT_NAMES
        spec = dict(cfg.extra.get("harmonie") or {})
        self.pinned = frozenset(str(v) for v in (spec.get("pinned_fingerprints") or {}).values())
        self.frame_shape = (len(self.schema.leads), grid.ny, grid.nx)
        declared = {v.name for v in cfg.frame_variables}
        missing = {"source_dataset", "fields_present", "is_partial_run"} - declared
        if missing:
            raise ValueError(f"{cfg.store_id}: frame_variables must declare {sorted(missing)}")
        if len(self.schema.fields) > 16:
            raise ValueError("fields_present is a 16-bit mask: at most 16 fields")
        self._lock = threading.Lock()
        self._listings: dict[tuple[str, str], tuple[float, list]] = {}
        self._selected: dict[datetime, RunFiles] = {}
        #: Swapped in tests for a reader over a local tar.
        self.reader_factory = remote_tar.RangeReader

    # -------------------------------------------------------------- listing

    def _month_listing(self, client, kind: str, month: datetime) -> list:
        key = (kind, f"{month:%Y%m}")
        with self._lock:
            hit = self._listings.get(key)
            if hit and time.monotonic() - hit[0] < self.schema.listing_ttl_s:
                return hit[1]
        out: list = []
        if kind == NETCDF:
            for param in self.schema.nc_params:
                stem = f"{self.schema.nc_prefix}{param}_{month:%Y%m}"
                out += client.list_files(begin=stem, end=f"{stem}~")
        else:
            stem = self.schema.p1_tar_name(month).split(f"{month:%Y%m%d%H}")[0] + f"{month:%Y%m}"
            out = client.list_files(begin=stem, end=f"{stem}~", dataset=self.schema.p1_dataset,
                                    version=self.schema.p1_version)
        with self._lock:
            self._listings[key] = (time.monotonic(), out)
        return out

    def list_day(self, client, day: datetime) -> dict[str, list]:
        """The NetCDF (and, from `p1_from`, P1) listings of the day's month: 9-10 requests
        a month, not a day."""
        month = datetime(day.year, day.month, 1, tzinfo=UTC)
        listing = {NETCDF: self._month_listing(client, NETCDF, month), P1: []}
        if day + timedelta(days=1) > self.schema.p1_from:
            listing[P1] = self._month_listing(client, P1, month)
        return listing

    def select(self, files: Mapping[str, list], day: datetime) -> dict[datetime, RunFiles]:
        """One bundle per run of `day`: P1 from `p1_from` when listed, else the NetCDF."""
        params = set(self.schema.nc_params)
        nc: dict[datetime, dict[str, object]] = {}
        for remote in files.get(NETCDF, ()):
            parsed = self.schema.parse_nc(remote.filename)
            if parsed is None or parsed[0] not in params or parsed[1].date() != day.date():
                continue
            nc.setdefault(parsed[1], {})[parsed[0]] = remote
        p1: dict[datetime, object] = {}
        for remote in files.get(P1, ()):
            run = parse_p1_tar(remote.filename)
            if run is not None and run.date() == day.date() and run >= self.schema.p1_from:
                p1[run] = remote
        out: dict[datetime, RunFiles] = {}
        for run in sorted(set(nc) | set(p1)):
            if run in p1:
                tar = p1[run]
                bundle = RunFiles(filename=tar.filename, size=tar.size,
                                  last_modified=tar.last_modified, kind=P1, run=run, files=(tar,))
            else:
                listed = nc[run]
                ordered = [listed[p] for p in self.schema.nc_params if p in listed]
                bundle = RunFiles(
                    filename=f"{self.schema.nc_prefix}{run:%Y%m%dT%H}", kind=NETCDF, run=run,
                    size=sum(f.size for f in ordered),
                    last_modified=max((f.last_modified or "" for f in ordered), default=None),
                    files=tuple(ordered), params=tuple(p for p in self.schema.nc_params
                                                       if p in listed))
            out[run] = bundle
            self._selected[run] = bundle
        return out

    # ------------------------------------------------------------- download

    def download(self, client, remote: RunFiles, target: Path) -> RawFile:
        directory = Path(target) / remote.filename.removesuffix(".tar")
        directory.mkdir(parents=True, exist_ok=True)
        if remote.kind == P1:
            n_bytes, fresh = download_p1(client, remote, directory, self.schema,
                                         self.reader_factory)
        else:
            n_bytes = fresh = 0
            for f in remote.files:
                raw = client.download(f, directory / f.filename)
                if raw.attempts:
                    n_bytes += raw.n_bytes
                    fresh += 1
        return RawFile(path=directory, sha256="", n_bytes=n_bytes, unit_key=remote.filename,
                       attempts=fresh)

    # ----------------------------------------------------------------- pack

    def pack(self, path: Path, when: datetime) -> PackedRun:
        files = self._selected.get(when)
        if files is None:
            raise RuntimeError(f"run {when.isoformat()} was not selected before packing")
        return pack_run(path, files, self.schema)

    def layout(self, packed: PackedRun) -> int:
        return LAYOUT_OF_KIND[packed.check.kind]

    def frame_values(self, packed: PackedRun) -> dict:
        present = set(packed.check.fields_present)
        mask = sum(1 << i for i, f in enumerate(self.schema.fields) if f.name in present)
        return {"source_dataset": SOURCE_CODES[packed.check.kind], "fields_present": mask,
                "is_partial_run": int(len(present) < len(self.schema.fields))}

    def provenance(self, packed: PackedRun) -> dict:
        c = packed.check
        files = self._selected.get(c.time)
        names = [f.filename for f in files.files] if files else []
        if files is not None and files.kind == P1:
            names = [f"{files.filename}/{self.schema.p1_member_name(files.run, lead)}"
                     for lead in self.schema.p1_leads]
        return {
            "source_dataset": {NETCDF: "uwcw_extra_lv_ha43_nl_2km",
                               P1: self.schema.p1_dataset}.get(c.kind, ""),
            "files": ",".join(names),
            "n_files": len(names),
            "fields_missing": ",".join(c.fields_missing),
            "partial": bool(c.fields_missing),
            "n_clipped": sum(s.get("n_clipped_low", 0) + s.get("n_clipped_high", 0)
                             for s in packed.stats.values()),
            "field_stats": json.dumps(packed.stats, sort_keys=True),
        }

    def status_extra(self, checks: Mapping[datetime, RunCheck]) -> dict:
        partial = {f"{w:%H%M}": c.fields_missing for w, c in sorted(checks.items())
                   if c.ok and c.fields_missing}
        kinds: dict[str, int] = {}
        for c in checks.values():
            if c.ok:
                kinds[c.kind] = kinds.get(c.kind, 0) + 1
        return {"partial_runs": partial, "runs_by_source": kinds}

    def store_attrs(self) -> dict:
        s = self.schema
        g = s.source
        return {
            "harmonie_source_grid": {
                "type": "regular lat/lon (south to north)", "lat0": g.lat0, "dlat": g.dlat,
                "nlat": g.nlat, "lon0": g.lon0, "dlon": g.dlon, "nlon": g.nlon,
                "proj4": s.grid_mapping_proj4},
            "coarsening": {
                "factor": s.factor,
                "method": "none: the native grid" if s.factor == 1 else
                          f"block mean of {s.factor} x {s.factor} native cells (gust: block "
                          f"maximum), no interpolation; a block needs min_valid of its cells",
                "min_valid": s.min_valid,
                "edge": "a remainder of nlat % factor rows (north) and nlon % factor columns "
                        "(east) would be dropped; none for factor 1 or 2 (390 = 2 x 195)"},
            "orientation": "north-up: row 0 is the northernmost latitude",
            "p1_from": s.p1_from.isoformat(),
            "lead_times_h": list(s.leads),
            "valid_time": "time + lead_time",
            "wind_components": "earth-relative (eastward, northward), not grid-relative",
        }

    def array_attrs(self) -> dict:
        out: dict[str, dict] = {
            "time": {"standard_name": "forecast_reference_time",
                     "long_name": "model run (forecast reference) time"}}
        for f in self.schema.fields:
            if f.undefined == "cloud_free":
                code = cloud_free_code(f.encoding)
                value = code * f.encoding.scale + f.encoding.offset
                out[f.name] = {"cloud_free_code": code, "cloud_free_value": value,
                               "comment": f"MASK IT: code {code} (decodes to {value:g} m) means "
                                          f"NO CLOUD BASE (a cloud-free column; coarsened: less "
                                          f"than min_valid of the block has a cloud base). It "
                                          f"is not a height and is not NaN after decoding."}
        return out

    def manifest_extra(self) -> dict:
        s = self.schema
        return {
            "encoding": {f.name: {"dtype": str(f.encoding.dtype), "scale": f.encoding.scale,
                                  "offset": f.encoding.offset,
                                  "missing_code": f.encoding.missing,
                                  "out_of_image_code": f.encoding.out_of_image,
                                  **({"cloud_free_code": cloud_free_code(f.encoding)}
                                     if f.undefined == "cloud_free" else {})}
                         for f in s.fields},
            "lead_times_h": list(s.leads),
            "coarsening": self.store_attrs()["coarsening"],
            "orientation": "north-up (y = latitude descending); the source rows run south to "
                           "north and are flipped",
            "source": {"api": self.settings.base_url,
                       "netcdf": {"dataset": self.settings.dataset,
                                  "version": self.settings.version,
                                  "files": f"{s.nc_prefix}<param>_<YYYYMMDDTHH>.nc",
                                  "params": s.nc_params},
                       "p1": {"dataset": s.p1_dataset, "version": s.p1_version,
                              "files": s.p1_tar, "members": s.p1_member,
                              "from": s.p1_from.isoformat()}},
        }


__all__ = [
    "LAYOUT_NAMES",
    "NETCDF",
    "P1",
    "FieldSpec",
    "HarmonieFormat",
    "PackedRun",
    "RunCheck",
    "RunFiles",
    "Schema",
    "check_netcdf_field",
    "cloud_free_code",
    "download_p1",
    "encode_field",
    "pack_run",
    "parse_p1_tar",
    "read_netcdf_run",
    "read_p1_members",
]
