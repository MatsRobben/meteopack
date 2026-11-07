"""Quality-control flags for an irregular points store (WOW-NL): level 2, part 1.

The packed WOW-NL store (`wow_nl/reports/v1`, level 1) keeps every report as
uploaded. This derive judges each report and writes the verdicts to a separate
store, `packed/stations/wow_nl/qc/v<N>`, row for row aligned with level 1: the same
month partitions, the same rows in the same order (`l1_row` is the row's
position in its level-1 month file), so the two are joined by position or by
`(station_id, time)`. Nothing is deleted or changed; a user decides what to
keep by reading the flags.

Per variable and report, three columns:

* `<var>_qc_fail` (uint32): one bit per test the value FAILED (`FAIL_BITS`).
* `<var>_qc_ne` (uint32): one bit per test that could NOT BE EVALUATED (too few
  neighbours, too little data), with the bit number of that test; plus two
  bits that are not tests: `censored_low` (a wind zero: below the detection
  limit, a lower bound rather than an error - Chen et al. 2021) and
  `isolated` (too few stations near this one to check it spatially - an
  isolated station is flagged as isolated, never as failed).
* `<var>_qc_level` (uint8): a summary from the two masks and the report bits
  by a configurable rule (`levels`): 0 no value, then by default 1 bad, 2
  suspect, 3 not spatially confirmed, 4 good.

Per report: `qc_report` (uint32, `REPORT_BITS`): provenance and metadata -
near-duplicate report, position jump, duplicate coordinates (CrowdQC+ m1),
an hour at WOW's 10,000-record page cap, local time inconsistent with UTC.

Tests (literature defaults in configs/derive/stations/wow_nl_qc.yaml, cited there):

* within station - range (climatological) and extreme (physically
  impossible), sentinel values, spike, step, persistence (stuck non-zero /
  long zero), dew point above temperature and dew point inconsistent with
  T and RH, stale repeat after a reporting gap; for the rain counter the
  robust rule's transient drop-outs and one-report spikes, resets away from
  midnight and huge increments.
* between stations, reference-free - every station's last usable value per
  5-min bin is the reference field: CrowdQC+ m2 (robust z against the whole
  network per bin), m3 (station-month fraction failing m2), m4 (station-month
  correlation with the network median), a buddy check (robust z against the
  neighbours' median, asymmetric thresholds), titanlib's SCT (temperature,
  hourly snapshots), isolation; rain: PWSQC's faulty zeros, high influx and
  station outlier on counter increments spread onto 5-min bins (de Vos et al.
  2017, 2019), three-state.

Work is one task per month, each reading its month plus a margin (long enough
for every persistence duration) and, for the rain station-outlier window,
the counter of the weeks before. Neighbour tables are computed once per
station-metadata version (`station_months.parquet`, `neighbours.parquet` in
the store) and reused. `qc.json` records every parameter, their hash, the bit
tables and how the run went; `reliability/` holds the station-month table.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

#: Bits of `<var>_qc_fail`. Never renumber a bit: stores written with it would
#: change meaning. Add new tests at the end.
FAIL_BITS: dict[str, int] = {
    "range": 0, "extreme": 1, "sentinel": 2, "spike": 3, "step": 4, "stuck": 5,
    "stuck_zero": 6, "td_gt_t": 7, "td_mismatch": 8, "stale_repeat": 9,
    "network_z": 10, "month_fraction": 11, "month_corr": 12, "buddy": 13, "sct": 14,
    "rain_transient": 15, "rain_spike": 16, "rain_reset_off_midnight": 17,
    "rain_huge_increment": 18, "rain_faulty_zero": 19, "rain_high_influx": 20,
    "rain_station_outlier": 21,
}
#: Bits of `<var>_qc_ne`: a test's own bit number when it could not be run,
#: plus two states that are not tests.
NE_BITS: dict[str, int] = {
    "network_z": 10, "month_fraction": 11, "month_corr": 12, "buddy": 13, "sct": 14,
    "rain_faulty_zero": 19, "rain_high_influx": 20, "rain_station_outlier": 21,
    "censored_low": 30, "isolated": 31,
}
#: Bits of `qc_report`.
REPORT_BITS: dict[str, int] = {
    "near_duplicate": 0, "position_jump": 1, "dup_coords": 2, "capped_hour": 3,
    "local_time_inconsistent": 4, "no_position": 5,
}

BIT_MEANING = {
    "fail": {
        "range": "outside the configured climatological range",
        "extreme": "outside physically possible values",
        "sentinel": "a known placeholder value (e.g. 0 degF dew point, RH 0, p 0)",
        "spike": "up (down) from both neighbouring reports by more than the threshold",
        "step": "changed by more than the threshold since a recent report",
        "stuck": "part of a run of one identical value lasting at least the configured time",
        "stuck_zero": "part of a run of zeros lasting at least the configured time",
        "td_gt_t": "dew point above air temperature (+ tolerance)",
        "td_mismatch": "dew point differs from the one computed from T and RH",
        "stale_repeat": "identical to the station's previous report across a reporting gap",
        "network_z": "CrowdQC+ m2: robust z against all stations in the 5-min bin out of bounds",
        "month_fraction": "CrowdQC+ m3: too large a share of the station-month failed m2",
        "month_corr": "CrowdQC+ m4: station-month correlation with the network median too low",
        "buddy": "robust z against the neighbours' median out of bounds",
        "sct": "titanlib spatial consistency test failed (hourly snapshot nearest the report)",
        "rain_transient": "counter drop-out the robust rule ignores (climbs back)",
        "rain_spike": "counter up-jump that falls straight back",
        "rain_reset_off_midnight": "counter reset not at a local or UTC day change",
        "rain_huge_increment": "counter increment too large for its interval",
        "rain_faulty_zero": "PWSQC FZ: dry while the neighbours report rain",
        "rain_high_influx": "PWSQC HI: far more rain than the neighbours",
        "rain_station_outlier": "PWSQC SO: rain series uncorrelated with the neighbours",
    },
    "ne": {
        "network_z": "too few stations in the bin", "month_fraction": "too few m2 values",
        "month_corr": "too few bins in common with the network median",
        "buddy": "too few neighbours with a value in the bin",
        "sct": "not in an SCT snapshot, or too few neighbours for the SCT",
        "rain_faulty_zero": "too few neighbours (or no increment for the report)",
        "rain_high_influx": "too few neighbours (or no increment for the report)",
        "rain_station_outlier": "too few neighbours, history or rain for the window",
        "censored_low": "zero: below the detection limit, a lower bound, not an error",
        "isolated": "too few stations with this variable near the station this month",
    },
    "report": {
        "near_duplicate": "less than near_duplicate_s after the station's previous report",
        "position_jump": "position further than position_jump_m from the station-month position",
        "dup_coords": "CrowdQC+ m1: another station has exactly the same coordinates this month",
        "capped_hour": "from an hour file at WOW's 10,000-record page cap (earliest minutes lost)",
        "local_time_inconsistent": "LocalReportEndDateTime is not the report time in local time",
        "no_position": "no latitude/longitude",
    },
}

#: Lattices a value may sit on, by unit: (label, scale, offset, step). A value x
#: in the store's unit is on a lattice when (x*scale + offset)/step is an integer.
LATTICES: dict[str, list[tuple[str, float, float, float]]] = {
    "degC": [("1 degF", 1.8, 32.0, 1.0), ("0.1 degF", 1.8, 32.0, 0.1), ("0.1 degC", 1, 0, 0.1),
             ("0.01 degC", 1, 0, 0.01)],
    "hPa": [("1 hPa", 1, 0, 1.0), ("0.01 inHg", 1 / 33.8639, 0, 0.01), ("0.1 hPa", 1, 0, 0.1)],
    "m s-1": [("1 kn", 1 / 0.514444, 0, 1.0), ("1 mph", 1 / 0.44704, 0, 1.0),
              ("1 km/h", 3.6, 0, 1.0), ("0.1 kn", 1 / 0.514444, 0, 0.1),
              ("0.1 mph", 1 / 0.44704, 0, 0.1), ("0.1 m/s", 1, 0, 0.1)],
    "mm": [("0.3 mm", 1, 0, 0.3), ("0.01 in", 1 / 25.4, 0, 0.01), ("0.2 mm", 1, 0, 0.2),
           ("0.1 mm", 1, 0, 0.1)],
    "mm h-1": [("0.01 in/h", 1 / 25.4, 0, 0.01), ("0.1 mm/h", 1, 0, 0.1)],
    "%": [("1 %", 1, 0, 1.0), ("0.1 %", 1, 0, 0.1)],
    "degree": [("22.5 deg", 1, 0, 22.5), ("1 deg", 1, 0, 1.0)],
}

_R_EARTH = 6_371_000.0
_LAT0, _LON0 = 52.2, 5.3


# -------------------------------------------------------------------- config


class Spike(BaseModel):
    model_config = {"extra": "forbid"}
    threshold: float
    max_dt_s: float = 1200


class Step(BaseModel):
    model_config = {"extra": "forbid"}
    threshold: float
    max_dt_s: float = 600


class Persistence(BaseModel):
    model_config = {"extra": "forbid"}
    #: A run of one identical (non-zero, if zero_is_special) value lasting this long.
    min_duration_s: float | None = None
    #: A run of zeros lasting this long (only with zero_is_special).
    zero_min_duration_s: float | None = None
    zero_is_special: bool = False


class NetworkZ(BaseModel):
    """CrowdQC+ m2: z = (x - median)/scale over all stations in the bin."""
    model_config = {"extra": "forbid"}
    low: float = -2.326          # qnorm(0.01)
    high: float = 1.645          # qnorm(0.95)
    min_stations: int = 20
    #: Floor on the scale (variable's unit), so a calm, uniform bin does not
    #: turn instrument resolution into outliers.
    min_scale: float = 0.0


class MonthFraction(BaseModel):
    """CrowdQC+ m3: the whole station-month when too many of its values fail m2."""
    model_config = {"extra": "forbid"}
    max_fraction: float = 0.2
    min_values: int = 288


class MonthCorr(BaseModel):
    """CrowdQC+ m4: the whole station-month when r with the network median is low."""
    model_config = {"extra": "forbid"}
    min_r: float = 0.9
    min_pairs: int = 288


class Buddy(BaseModel):
    model_config = {"extra": "forbid"}
    radius_m: float = 10_000
    min_buddies: int = 5
    max_buddies: int = 20
    #: Flag when z > pos (value above its buddies) or z < -neg.
    pos: float = 3.0
    neg: float = 3.0
    min_scale: float = 0.0


class Sct(BaseModel):
    """titanlib.sct (Lussana et al. 2010) on hourly snapshots of the reference field."""
    model_config = {"extra": "forbid"}
    step_seconds: int = 3600
    num_min: int = 5
    num_max: int = 30
    inner_radius_m: float = 5_000
    outer_radius_m: float = 20_000
    num_iterations: int = 2
    num_min_prof: int = 20
    min_elev_diff_m: float = 200
    min_horizontal_scale_m: float = 1_000
    vertical_scale_m: float = 200
    pos: float = 4.0
    neg: float = 8.0
    eps2: float = 0.5


class Isolation(BaseModel):
    model_config = {"extra": "forbid"}
    radius_m: float = 15_000
    min_neighbours: int = 5


class VariableQc(BaseModel):
    """The tests of one variable; a test left out (null) is not run."""
    model_config = {"extra": "forbid"}
    #: Unit, for the lattice (native-unit) inference in the reliability table.
    unit: str | None = None
    range: tuple[float | None, float | None] | None = None
    extreme: tuple[float | None, float | None] | None = None
    sentinels: list[float] = Field(default_factory=list)
    sentinel_tol: float = 0.005
    spike: Spike | None = None
    step: Step | None = None
    persistence: Persistence | None = None
    #: Zeros are below the detection limit: NE bit `censored_low`, never a failure.
    censor_zero: bool = False
    stale_repeat: bool = True
    network_z: NetworkZ | None = None
    month_fraction: MonthFraction | None = None
    month_corr: MonthCorr | None = None
    buddy: Buddy | None = None
    sct: Sct | None = None
    isolation: Isolation | None = None

    def network(self) -> bool:
        return any(x is not None for x in (self.network_z, self.month_corr, self.buddy,
                                           self.sct, self.isolation))


class DewPoint(BaseModel):
    model_config = {"extra": "forbid"}
    dew_point: str = "dew_point_temperature"
    temperature: str = "air_temperature"
    humidity: str = "relative_humidity"
    gt_t_tolerance: float | None = 0.5
    mismatch_tolerance: float | None = 2.0


class StaleRepeat(BaseModel):
    model_config = {"extra": "forbid"}
    gap_s: float = 3600
    #: Fields compared; the report must repeat all of them exactly.
    fields: list[str] = Field(default_factory=lambda: [
        "air_temperature", "relative_humidity", "air_pressure_at_sea_level", "wind_speed",
        "wind_from_direction", "rainfall_counter"])


class ReportChecks(BaseModel):
    model_config = {"extra": "forbid"}
    near_duplicate_s: float | None = 30
    position_jump_m: float | None = 1000
    dup_coords: bool = True
    capped_hour: bool = True
    local_time: bool = True
    local_timezone: str = "Europe/Amsterdam"


class Network(BaseModel):
    """The reference field the between-station tests compare against."""
    model_config = {"extra": "forbid"}
    step_seconds: int = 300
    #: A station's value stands for this long after its report (so a 10-min
    #: reporter is in every 5-min bin).
    max_age_s: int = 600
    #: Values failing any of these are not part of the reference field.
    reference_excludes: list[str] = Field(default_factory=lambda: [
        "range", "extreme", "sentinel", "spike", "step", "stuck", "stuck_zero", "td_gt_t",
        "stale_repeat"])
    #: Use reported station heights (elevation difference limits, lapse rate).
    #: Off: WOW heights are 0 for most site-years before 2016 and unreliable after.
    use_height: bool = False
    lapse_rate: float = 0.0065


class SO(BaseModel):
    model_config = {"extra": "forbid"}
    compare_days: float = 14           # PWSQC compareint = 4032 x 5 min
    rainyint: int = 100
    matchint: int = 200
    corthres: float = 0.15
    #: How far back the window may grow to find `rainyint` wet intervals (the
    #: counter of these weeks is read for every month).
    max_window_days: float = 28
    eval_step_s: int = 3600


class Pwsqc(BaseModel):
    model_config = {"extra": "forbid"}
    radius_m: float = 10_000
    max_neighbours: int | None = 20
    nstat: int = 5
    nint: int = 6
    hi_thres_a: float = 0.4
    hi_thres_b: float = 10.0
    so: SO | None = Field(default_factory=SO)


class RainQc(BaseModel):
    """The rain counter: the robust increment rule and the tests built on it."""
    model_config = {"extra": "forbid"}
    counter: str = "rainfall_counter"
    drop_mm: float = 0.25
    lookahead: int = 3
    valid_min: float = 0.0
    valid_max: float = 1000.0
    local_timezone: str = "Europe/Amsterdam"
    huge_increment_mm: float | None = 20.0
    huge_increment_max_dt_s: float = 600
    #: Increments over longer intervals are not spread onto bins (de Vos 2017).
    spread_max_s: float = 1200
    min_coverage: float = 0.5
    pwsqc: Pwsqc | None = Field(default_factory=Pwsqc)


class LevelRule(BaseModel):
    model_config = {"extra": "forbid"}
    level: Annotated[int, Field(ge=1, le=254)]
    name: str
    fail: list[str] = Field(default_factory=list)
    ne: list[str] = Field(default_factory=list)
    report: list[str] = Field(default_factory=list)


class Reliability(BaseModel):
    model_config = {"extra": "forbid"}
    #: Tier 2 needs these variables (those the station reports) at level >= ok_level ...
    core: list[str] = Field(default_factory=lambda: [
        "air_temperature", "relative_humidity", "air_pressure_at_sea_level"])
    ok_level: int = 3
    #: ... for at least this share of their values. Official stations are tier 1.
    tier2_min_fraction: float = 0.9


def _default_levels() -> list[LevelRule]:
    return [
        LevelRule(level=1, name="bad", fail=[
            "range", "extreme", "sentinel", "spike", "step", "stuck", "stuck_zero", "td_gt_t",
            "td_mismatch", "stale_repeat", "rain_transient", "rain_spike",
            "rain_reset_off_midnight", "rain_huge_increment"]),
        LevelRule(level=2, name="suspect", fail=[
            "network_z", "month_fraction", "month_corr", "buddy", "sct", "rain_faulty_zero",
            "rain_high_influx", "rain_station_outlier"], report=["position_jump", "dup_coords",
                                                                "no_position"]),
        LevelRule(level=3, name="not spatially confirmed", ne=[
            "network_z", "month_corr", "buddy", "sct", "rain_faulty_zero", "rain_high_influx",
            "rain_station_outlier", "censored_low", "isolated"]),
    ]


class QcConfig(BaseModel):
    """Quality-control flags for one irregular points store."""

    model_config = {"extra": "forbid"}

    kind: Literal["qc"] = "qc"
    #: Output store `<name>/v<version>` under `layer`.
    name: str = "wow_nl/qc"
    version: int = 1
    layer: Literal["packed", "training"] = "packed"
    description: str | None = None
    input: str = "wow_nl/reports/v1"
    variables: dict[str, VariableQc] = Field(default_factory=dict)
    dew_point: DewPoint | None = Field(default_factory=DewPoint)
    stale_repeat: StaleRepeat = Field(default_factory=StaleRepeat)
    report: ReportChecks = Field(default_factory=ReportChecks)
    network: Network = Field(default_factory=Network)
    rain: RainQc | None = Field(default_factory=RainQc)
    levels: list[LevelRule] = Field(default_factory=_default_levels)
    #: Level of a value no rule matched.
    default_level: int = 4
    reliability: Reliability = Field(default_factory=Reliability)
    #: Also write the scores behind the network tests (z, buddy z, neighbour
    #: count) as float32 columns - for tuning thresholds without re-running.
    write_scores: bool = False
    #: Seconds read either side of a month. None: the longest persistence
    #: duration, stale gap or spike window, plus an hour.
    margin_s: Annotated[int, Field(ge=0)] | None = None
    compression_level: int = 3
    #: polars threads per worker; not part of `params_hash`.
    threads: Annotated[int, Field(ge=1)] = 2

    @model_validator(mode="after")
    def _check(self) -> QcConfig:
        for rule in self.levels:
            for kind, names, table in (("fail", rule.fail, FAIL_BITS), ("ne", rule.ne, NE_BITS),
                                       ("report", rule.report, REPORT_BITS)):
                bad = [n for n in names if n not in table]
                if bad:
                    raise ValueError(f"levels: unknown {kind} bit(s) {bad}; known: {sorted(table)}")
        bad = [n for n in self.network.reference_excludes if n not in FAIL_BITS]
        if bad:
            raise ValueError(f"network.reference_excludes: unknown bit(s) {bad}")
        for name, v in self.variables.items():
            if v.month_fraction is not None and v.network_z is None:
                raise ValueError(f"{name}: month_fraction (m3) needs network_z (m2)")
            if v.sct is not None and v.sct.step_seconds % self.network.step_seconds:
                raise ValueError(f"{name}: sct.step_seconds must be a multiple of the network step")
        if self.rain is not None and self.rain.counter not in self.variables:
            raise ValueError(f"rain.counter {self.rain.counter!r} must be one of the variables")
        if self.rain is not None and self.rain.pwsqc is not None and self.rain.pwsqc.so:
            so = self.rain.pwsqc.so
            if so.eval_step_s % self.network.step_seconds or 3600 % so.eval_step_s:
                raise ValueError("so.eval_step_s must be a multiple of the network step and "
                                 "divide an hour")
        return self

    @property
    def store_id(self) -> str:
        return f"{self.name}/v{self.version}"

    @property
    def category(self) -> str:
        from meteopack.core.config import category_for_source

        source = self.input.split("/", 1)[0]
        return category_for_source(source) or "other"

    @property
    def margin(self) -> int:
        if self.margin_s is not None:
            return self.margin_s
        need = [self.stale_repeat.gap_s]
        for v in self.variables.values():
            if v.persistence:
                need += [v.persistence.min_duration_s or 0, v.persistence.zero_min_duration_s or 0]
            if v.spike:
                need.append(v.spike.max_dt_s)
            if v.step:
                need.append(v.step.max_dt_s)
        if self.rain is not None:
            need.append(2 * 86400)           # the rain rule's day logic and look-ahead
        return int(max(need)) + 3600

    def params_hash(self) -> str:
        payload = self.model_dump(mode="json", exclude={"name", "version", "description",
                                                        "margin_s", "threads", "layer",
                                                        "compression_level"})
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def neighbour_radius(self) -> float:
        """The largest neighbour radius any test needs: what the shared table holds."""
        r = [0.0]
        for v in self.variables.values():
            r += [v.buddy.radius_m if v.buddy else 0, v.isolation.radius_m if v.isolation else 0]
        if self.rain is not None and self.rain.pwsqc is not None:
            r.append(self.rain.pwsqc.radius_m)
        return max(r)


# ----------------------------------------------------- station-month tables


def _xy(lat, lon):
    import numpy as np

    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    x = np.radians(lon - _LON0) * _R_EARTH * np.cos(np.radians(_LAT0))
    y = np.radians(lat - _LAT0) * _R_EARTH
    return np.column_stack([x, y])


def station_months(input_store: Path):
    """One row per (year, month, station): the position it reported most that month.

    From level 1's `station_segments` (runs of identical position per site and
    month): the segment with the most reports, the number of distinct
    positions, whether it reported as official, and whether another station
    shares its exact coordinates (CrowdQC+ m1).
    """
    import polars as pl

    seg = pl.scan_parquet(str(input_store / "station_segments" / "**" / "*.parquet"),
                          hive_partitioning=False)
    seg = seg.filter(pl.col("station_id").is_not_null()).with_columns(
        pl.col("first_time").dt.year().alias("year"),
        pl.col("first_time").dt.month().alias("month")).collect()
    ok = pl.col("Latitude").is_not_null() & pl.col("Longitude").is_not_null()
    best = (seg.sort("n_reports", descending=True)
            .group_by("year", "month", "station_id", maintain_order=True)
            .agg(pl.col("Latitude").filter(ok).first().alias("lat"),
                 pl.col("Longitude").filter(ok).first().alias("lon"),
                 pl.col("Height").filter(ok).first().alias("height"),
                 pl.len().alias("n_positions"), pl.col("n_reports").sum(),
                 (pl.col("n_official").sum() > 0).alias("official")))
    best = best.with_columns(
        (pl.len().over("year", "month", "lat", "lon") > 1).and_(pl.col("lat").is_not_null())
        .alias("dup_coords"))
    return best.sort("year", "month", "station_id")


def neighbour_table(sm, radius: float):
    """All pairs of stations active in the same month within `radius`, with distance."""
    import numpy as np
    import polars as pl

    parts = []
    for (y, m), g in sm.group_by("year", "month", maintain_order=True):
        xy = _xy(g["lat"].to_numpy(), g["lon"].to_numpy())
        ids = g["station_id"]
        hz = g["height"].fill_null(0).to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        i, j = np.nonzero(d <= radius)
        parts.append(pl.DataFrame({"year": np.full(len(i), y, np.int32),
                                   "month": np.full(len(i), m, np.int32),
                                   "station_id": ids.gather(i), "neighbour_id": ids.gather(j),
                                   "dist_m": d[i, j].astype(np.float32),
                                   "dz_m": (hz[j] - hz[i]).astype(np.float32)}))
    return pl.concat(parts).sort("year", "month", "station_id", "dist_m")


def _segments_fingerprint(input_store: Path) -> str:
    h = hashlib.sha256()
    for p in sorted((input_store / "station_segments").rglob("*.parquet")):
        st = p.stat()
        h.update(f"{p.relative_to(input_store)}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


#: Bump when `station_months`/`neighbour_table` change what they compute.
_TABLES_VERSION = 2


def prepare_tables(cfg: QcConfig, input_store: Path, store: Path) -> dict:
    """Station-month positions and neighbour pairs, rebuilt only when they would change."""
    import polars as pl

    key = hashlib.sha256(f"{_TABLES_VERSION}:{_segments_fingerprint(input_store)}:"
                         f"{cfg.neighbour_radius()}"
                         .encode()).hexdigest()[:16]
    meta = store / "tables.json"
    sm_path, nb_path = store / "station_months.parquet", store / "neighbours.parquet"
    if (meta.exists() and sm_path.exists() and nb_path.exists()
            and json.loads(meta.read_text()).get("key") == key):
            return {"key": key, "rebuilt": False, "station_months": str(sm_path),
                    "neighbours": str(nb_path)}
    started = time.time()
    store.mkdir(parents=True, exist_ok=True)
    sm = station_months(input_store)
    nb = neighbour_table(sm.filter(pl.col("lat").is_not_null()), cfg.neighbour_radius())
    for df, p in ((sm, sm_path), (nb, nb_path)):
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        df.write_parquet(tmp)
        os.replace(tmp, p)
    info = {"key": key, "rebuilt": True, "station_months": str(sm_path),
            "neighbours": str(nb_path), "rows_station_months": sm.height,
            "rows_neighbours": nb.height, "radius_m": cfg.neighbour_radius(),
            "seconds": time.time() - started}
    meta.write_text(json.dumps(info, indent=2))
    return info


# ----------------------------------------------------------------- the month


def _month_starts(lo: datetime, hi: datetime) -> list[datetime]:
    m = lo.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while m < hi:
        out.append(m)
        m = m.replace(year=m.year + (m.month == 12), month=m.month % 12 + 1)
    return out


def _month_file(root: Path, store_id: str, m: datetime, sub: str = "observations") -> Path:
    from meteopack.core.store import points

    return (points.store_path(root, store_id) / sub / f"year={m.year:04d}"
            / f"month={m.month:02d}" / "data.parquet")


def _naive(d: datetime) -> datetime:
    return d.astimezone(UTC).replace(tzinfo=None)


def _read(root: Path, cfg: QcConfig, lo: datetime, hi: datetime, a: datetime, b: datetime,
          cols: list[str]):
    """Reports in [a, b), sorted by (station, time); `rid` = file<<32 | row in file."""
    import polars as pl

    scans = []
    for k, m in enumerate(_month_starts(a, b)):
        f = _month_file(root, cfg.input, m)
        if not f.exists():
            continue
        names = pl.read_parquet_schema(f)
        sel = [c for c in cols if c in names]
        target = (m.year, m.month) == (lo.year, lo.month)
        sc = (pl.scan_parquet(f, hive_partitioning=False).with_row_index("l1_row")
              .select(*sel, "l1_row")
              .with_columns(*(pl.lit(None, pl.Float64).alias(c) for c in cols if c not in names),
                            pl.lit(target).alias("in_month"),
                            (pl.lit(k << 32, pl.Int64) + pl.col("l1_row").cast(pl.Int64))
                            .alias("rid")))
        if not target:
            sc = sc.filter((pl.col("time") >= _naive(a)) & (pl.col("time") < _naive(b)))
        scans.append(sc.select(*cols, "l1_row", "in_month", "rid"))
    if not scans:
        return None
    return pl.concat(scans).collect().sort("station_id", "time", "rid")


def _bit(name: str, table: dict[str, int]):
    import numpy as np

    return np.uint32(1 << table[name])


def _within_station(name: str, spec: VariableQc, x_all, t_all, sid_all, fail, ne):
    """Range, extreme, sentinel, spike, step, persistence, censoring for one variable."""
    import numpy as np

    from meteopack.derive import qc_checks as ck

    idx = np.flatnonzero(~np.isnan(x_all))
    x, t, sid = x_all[idx], t_all[idx], sid_all[idx]
    gross = np.zeros(len(x), bool)

    def put(mask, bit, table=FAIL_BITS, target=fail):
        target[idx[mask]] |= _bit(bit, table)

    if spec.range is not None:
        put(ck.outside(x, *spec.range), "range")
        gross |= ck.outside(x, *spec.range)
    if spec.extreme is not None:
        bad = ck.outside(x, *spec.extreme)
        put(bad, "extreme")
        gross |= bad
    if spec.sentinels:
        bad = ck.sentinel(x, spec.sentinels, spec.sentinel_tol)
        put(bad, "sentinel")
        gross |= bad
    if spec.spike is not None or spec.step is not None:
        # judged among values that are not gross errors, so the report after a
        # sentinel is not a "step" back to normal
        keep = np.flatnonzero(~gross)
        xp, xn, dtp, dtn = ck.prev_next(x[keep], t[keep], sid[keep])
        if spec.spike is not None:
            m = ck.spike(x[keep], xp, xn, dtp, dtn, spec.spike.threshold, spec.spike.max_dt_s)
            fail[idx[keep[m]]] |= _bit("spike", FAIL_BITS)
        if spec.step is not None:
            m = ck.step(x[keep], xp, dtp, spec.step.threshold, spec.step.max_dt_s)
            fail[idx[keep[m]]] |= _bit("step", FAIL_BITS)
    if spec.persistence is not None:
        p = spec.persistence
        stuck, stuck0 = ck.persistence(x, t, sid, p.min_duration_s, p.zero_min_duration_s,
                                       p.zero_is_special)
        put(stuck, "stuck")
        put(stuck0, "stuck_zero")
    if spec.censor_zero:
        put(x == 0, "censored_low", NE_BITS, ne)


def _local_time_offset(frame, tz: str):
    """Minutes LocalReportEndDateTime is ahead of `time`, and the offset it should be."""
    import polars as pl

    local = (pl.col("LocalReportEndDateTime").str.slice(0, 19)
             .str.to_datetime("%Y-%m-%dT%H:%M:%S", strict=False))
    zoned = pl.col("time").dt.replace_time_zone("UTC").dt.convert_time_zone(tz)
    expect = (zoned.dt.base_utc_offset() + zoned.dt.dst_offset()).dt.total_minutes()
    got = ((local - pl.col("time").dt.truncate("1s")).dt.total_seconds() / 60).round(0)
    return frame.select(got.alias("got"), expect.alias("expect"))


def _report_bits(cfg: QcConfig, frame, sm_month, hours, arr, rep):
    """qc_report bits from times, positions and the hour table."""
    import numpy as np
    import polars as pl

    from meteopack.derive import qc_checks as ck

    r = cfg.report
    t, sid = arr["t"], arr["sid"]
    if r.near_duplicate_s is not None:
        same = ck.same_as_previous(sid)
        near = np.zeros(len(t), bool)
        near[1:] = (t[1:] - t[:-1]) < r.near_duplicate_s
        rep[same & near] |= _bit("near_duplicate", REPORT_BITS)
    lat = frame["Latitude"].to_numpy().astype(float)
    lon = frame["Longitude"].to_numpy().astype(float)
    nopos = np.isnan(lat) | np.isnan(lon)
    rep[nopos] |= _bit("no_position", REPORT_BITS)
    pos = frame.select("station_id").join(
        sm_month.select("station_id", "lat", "lon", "dup_coords"), on="station_id", how="left",
        maintain_order="left")
    if r.position_jump_m is not None:
        ref = _xy(pos["lat"].to_numpy().astype(float), pos["lon"].to_numpy().astype(float))
        here = _xy(lat, lon)
        dist = np.hypot(*(here - ref).T)
        rep[dist > r.position_jump_m] |= _bit("position_jump", REPORT_BITS)
    if r.dup_coords:
        rep[pos["dup_coords"].fill_null(False).to_numpy()] |= _bit("dup_coords", REPORT_BITS)
    if r.capped_hour and hours is not None and "source_member" in frame.columns:
        capped = hours.filter(pl.col("at_page_cap").fill_null(False))["member"].unique()
        m = frame.select(pl.col("source_member").is_in(capped.to_list())).to_series().to_numpy()
        rep[m] |= _bit("capped_hour", REPORT_BITS)
    if r.local_time and frame.schema.get("LocalReportEndDateTime") == pl.String:
        off = _local_time_offset(frame, r.local_timezone)
        got, exp = off["got"].to_numpy(), off["expect"].to_numpy()
        bad = ~np.isnan(got) & (got != exp)
        rep[bad] |= _bit("local_time_inconsistent", REPORT_BITS)


def _rain_counter(cfg: QcConfig, croot, lo, hi, fail_c, rid_main, result: dict):
    """Robust increments of the counter over [lo - lookback, hi + margin) and their flags.

    Returns the increments frame (all rows of the counter read, with `pos` =
    row in the main frame or -1) for the PWSQC step.
    """
    import numpy as np
    import polars as pl

    from meteopack.derive.regularise import RainSpec, rain_increments

    rq = cfg.rain
    spec = RainSpec(counter=rq.counter, rule="robust", drop_mm=rq.drop_mm, lookahead=rq.lookahead,
                    valid_min=rq.valid_min, valid_max=rq.valid_max,
                    local_timezone=rq.local_timezone)
    inc, _ = rain_increments(croot, spec, keep=("rid",), flags=True)
    pos_of = pl.DataFrame({"rid": rid_main, "pos": np.arange(len(rid_main), dtype=np.int64)})
    inc = inc.join(pos_of, on="rid", how="left").with_columns(pl.col("pos").fill_null(-1))
    p = inc["pos"].to_numpy()
    here = p >= 0
    pi = p[here]

    def put(col_expr, bit):
        m = inc.select(col_expr.fill_null(False)).to_series().to_numpy()[here]
        fail_c[pi[m]] |= _bit(bit, FAIL_BITS)

    put(pl.col("transient"), "rain_transient")
    put(pl.col("spike"), "rain_spike")
    put(pl.col("reset") & ~pl.col("day_change"), "rain_reset_off_midnight")
    if rq.huge_increment_mm is not None:
        put((pl.col("inc") > rq.huge_increment_mm) & (pl.col("dt") <= rq.huge_increment_max_dt_s),
            "rain_huge_increment")
    result["increment"] = np.full(len(rid_main), np.nan, np.float32)
    result["increment"][pi] = inc["inc"].to_numpy()[here].astype(np.float32)
    return inc


def _snap_index(t, lo_s, step):
    import numpy as np

    return np.floor((t - lo_s) / step + 0.5).astype(np.int64)


def _sct(spec: Sct, v, lat, lon, elev, k_step):
    """titanlib SCT on every `k_step`-th column of the reference matrix: (S, Q) three-state."""
    import numpy as np
    import titanlib

    n_s, n_j = v.shape
    snaps = np.arange(0, n_j, k_step)
    out = np.full((n_s, len(snaps)), -1, np.int8)
    has_pos = ~np.isnan(lat) & ~np.isnan(lon)
    for q, j in enumerate(snaps):
        ok = np.flatnonzero(~np.isnan(v[:, j]) & has_pos)
        if len(ok) <= spec.num_min:
            continue
        pts = titanlib.Points(lat[ok], lon[ok], elev[ok])
        vals = v[ok, j].astype(np.float32)
        n = len(ok)
        flags, _, _ = titanlib.sct(
            pts, vals, spec.num_min, spec.num_max, spec.inner_radius_m, spec.outer_radius_m,
            spec.num_iterations, spec.num_min_prof, spec.min_elev_diff_m,
            spec.min_horizontal_scale_m, spec.vertical_scale_m, np.full(n, spec.pos),
            np.full(n, spec.neg), np.full(n, spec.eps2))
        iso = np.asarray(titanlib.isolation_check(pts, spec.num_min, spec.outer_radius_m))
        res = np.asarray(flags).astype(np.int8)
        res[iso == 1] = -1
        out[ok, q] = res
    return out


def _network(cfg: QcConfig, name: str, spec: VariableQc, x_in, s_in, j_in, t_in, lo_s,
             n_s, n_j, xy, lat, lon, elev, fail_in, ne_in, scores, stat_rows: dict):
    """Between-station tests of one variable for the month's reports (arrays: in-month rows)."""
    import numpy as np

    from meteopack.derive import qc_checks as ck

    net = cfg.network
    has = ~np.isnan(x_in)
    excl = np.uint32(0)
    for b in net.reference_excludes:
        excl |= _bit(b, FAIL_BITS)
    ref = has & ((fail_in & excl) == 0)
    v = ck.bin_last(s_in[ref], j_in[ref], x_in[ref], n_s, n_j)
    v = ck.ffill_limited(v, net.max_age_s // net.step_seconds)
    if net.use_height:
        v = v + net.lapse_rate * (elev - np.nanmean(elev))[:, None]
        x_cmp = x_in + net.lapse_rate * (elev - np.nanmean(elev))[s_in]
    else:
        x_cmp = x_in
    station_has = ~np.all(np.isnan(v), axis=1)
    idx = np.flatnonzero(has)
    si, ji, xi = s_in[idx], j_in[idx], x_cmp[idx]
    timing = stat_rows.setdefault("timing", {})

    def setf(mask, bit):
        fail_in[idx[mask]] |= _bit(bit, FAIL_BITS)

    def setn(mask, bit):
        ne_in[idx[mask]] |= _bit(bit, NE_BITS)

    # An isolated station is flagged as isolated, never as failed: the
    # network-wide tests (m2-m4) are not evaluable for it either (an offshore or
    # coastal station need not follow the inland network median).
    isolated = np.zeros(n_s, bool)
    if spec.isolation is not None:
        iso = spec.isolation
        nb = ck.neighbour_lists(xy, iso.radius_m, None, station_has)
        n_nb = (nb >= 0).sum(1) if nb.shape[1] else np.zeros(n_s, int)
        isolated = n_nb < iso.min_neighbours
        setn(isolated[si], "isolated")
        stat_rows["n_neighbours"] = n_nb
        stat_rows["isolated"] = isolated
    iso_r = isolated[si]

    med_net = None
    if spec.network_z is not None or spec.month_corr is not None:
        t0 = time.time()
        med_net, mad_net, n_net = ck.network_stats(v)
        timing["network_stats"] = timing.get("network_stats", 0) + time.time() - t0
    if spec.network_z is not None:
        nz = spec.network_z
        scale = np.maximum(ck.MAD_TO_SD * mad_net, nz.min_scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (xi - med_net[ji]) / scale[ji]
        evaluable = (n_net[ji] >= nz.min_stations) & np.isfinite(z) & ~iso_r
        bad = evaluable & ((z < nz.low) | (z > nz.high))
        setf(bad, "network_z")
        setn(~evaluable, "network_z")
        if cfg.write_scores:
            scores[f"{name}_z_network"] = (idx, z.astype(np.float32))
        if spec.month_fraction is not None:
            mf = spec.month_fraction
            n_eval = np.bincount(si[evaluable], minlength=n_s)
            n_bad = np.bincount(si[bad], minlength=n_s)
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = n_bad / n_eval
            st_bad = (n_eval >= mf.min_values) & (frac > mf.max_fraction) & ~isolated
            st_ne = (n_eval < mf.min_values) | isolated
            setf(st_bad[si], "month_fraction")
            setn(st_ne[si], "month_fraction")
    if spec.month_corr is not None:
        mc = spec.month_corr
        r, n_pairs = ck.pearson_rows(v, med_net)
        st_ne = (n_pairs < mc.min_pairs) | ~np.isfinite(r) | isolated
        st_bad = ~st_ne & (r < mc.min_r)
        setf(st_bad[si], "month_corr")
        setn(st_ne[si], "month_corr")
        stat_rows["r_network"] = np.where((n_pairs < mc.min_pairs), np.nan, r)
        stat_rows["n_pairs_network"] = n_pairs
    if spec.buddy is not None:
        bd = spec.buddy
        t0 = time.time()
        nb = ck.neighbour_lists(xy, bd.radius_m, bd.max_buddies, station_has)
        med, mad, cnt = ck.neighbour_stats(v, nb)
        timing["buddy"] = timing.get("buddy", 0) + time.time() - t0
        scale = np.maximum(ck.MAD_TO_SD * mad[si, ji], bd.min_scale)
        with np.errstate(invalid="ignore", divide="ignore"):
            z = (xi - med[si, ji]) / scale
        evaluable = (cnt[si, ji] >= bd.min_buddies) & np.isfinite(z)
        setf(evaluable & ((z > bd.pos) | (z < -bd.neg)), "buddy")
        setn(~evaluable, "buddy")
        stat_rows["buddy_n"] = cnt[si, ji]
        if cfg.write_scores:
            scores[f"{name}_z_buddy"] = (idx, z.astype(np.float32))
            scores[f"{name}_n_buddy"] = (idx, cnt[si, ji].astype(np.float32))
    if spec.sct is not None:
        t0 = time.time()
        k = spec.sct.step_seconds // net.step_seconds
        res = _sct(spec.sct, v, lat, lon, elev if net.use_height else np.zeros(n_s), k)
        q = _snap_index(t_in[idx], lo_s, spec.sct.step_seconds)
        q = np.clip(q, 0, res.shape[1] - 1)
        verdict = res[si, q]
        setf((verdict == 1) & ~iso_r, "sct")
        setn((verdict == -1) | iso_r, "sct")
        timing["sct"] = timing.get("sct", 0) + time.time() - t0


def _pwsqc(cfg: QcConfig, inc, sm_ids, xy, has_counter, lo, hi, fail_c, ne_c, timing: dict):
    """PWSQC FZ / HI / SO on 5-min amounts from the counter, mapped back to reports."""
    import numpy as np
    import polars as pl

    from meteopack.derive import qc_checks as ck

    pw = cfg.rain.pwsqc
    step = cfg.network.step_seconds
    so = pw.so
    look = so.max_window_days * 86400 if so is not None else 0
    lo_s = lo.timestamp()
    hi_s = hi.timestamp()
    t0 = lo_s - np.ceil(look / 3600) * 3600 if look else lo_s
    n_j = int(np.ceil((hi_s - t0) / 3600) * 3600 / step)
    n_s = len(sm_ids)
    t_start = time.time()
    s_map = pl.DataFrame({"station_id": sm_ids, "s": np.arange(n_s, dtype=np.int32)})
    d = inc.join(s_map, on="station_id", how="inner")
    t_end = d["time"].dt.epoch("us").to_numpy() / 1e6
    r, rep, jj = ck.spread_increments(
        d["s"].to_numpy(), t_end, d["dt"].to_numpy().astype(float),
        d["inc"].to_numpy().astype(float), t0, step, n_s, n_j, cfg.rain.spread_max_s,
        cfg.rain.min_coverage)
    nb = ck.neighbour_lists(xy, pw.radius_m, pw.max_neighbours, has_counter)
    med, _, cnt = ck.neighbour_stats(r.astype(np.float32), nb, mad=False)
    hi_f = ck.pwsqc_high_influx(r, med, cnt, pw.nstat, pw.hi_thres_a, pw.hi_thres_b)
    fz = ck.pwsqc_faulty_zeros(r, med, cnt, pw.nstat, pw.nint)
    flags = {"rain_high_influx": hi_f, "rain_faulty_zero": fz}
    timing["pwsqc_fz_hi"] = time.time() - t_start
    if os.environ.get("METEOPACK_QC_DEBUG"):
        print(f"{lo:%Y-%m} pwsqc_fz_hi {timing['pwsqc_fz_hi']:.1f} s", flush=True)
    if so is not None:
        t1 = time.time()
        rf = np.where((hi_f == 1) | (fz == 1), np.nan, r)
        per = so.eval_step_s // step
        first = max(int((lo_s - t0) // so.eval_step_s) - 1, 0)
        flags["rain_station_outlier"] = ck.pwsqc_station_outlier(
            rf, nb, pw.nstat, per, int(so.compare_days * 86400 // so.eval_step_s), so.rainyint,
            so.matchint, so.corthres, first)
        timing["pwsqc_so"] = time.time() - t1
        if os.environ.get("METEOPACK_QC_DEBUG"):
            print(f"{lo:%Y-%m} pwsqc_so {timing['pwsqc_so']:.1f} s", flush=True)
    pos = d["pos"].to_numpy()
    in_month = pos >= 0
    # every report of the month with a counter value gets a verdict
    keep = np.flatnonzero(in_month)
    remap = np.full(len(pos), -1, np.int64)
    remap[keep] = np.arange(len(keep))
    sel = remap[rep] >= 0
    for bit, f in flags.items():
        fl, ne = ck.three_state_per_report(f, d["s"].to_numpy()[keep], remap[rep[sel]],
                                           jj[sel], len(keep))
        fail_c[pos[keep][fl]] |= _bit(bit, FAIL_BITS)
        ne_c[pos[keep][ne]] |= _bit(bit, NE_BITS)
    return flags


def _levels(cfg: QcConfig, fail, ne, rep, present):
    import numpy as np

    lvl = np.full(len(fail), cfg.default_level, np.uint8)
    done = np.zeros(len(fail), bool)
    for rule in cfg.levels:
        fm = np.uint32(sum(1 << FAIL_BITS[b] for b in rule.fail))
        nm = np.uint32(sum(1 << NE_BITS[b] for b in rule.ne))
        rm = np.uint32(sum(1 << REPORT_BITS[b] for b in rule.report))
        cond = ((fail & fm) != 0) | ((ne & nm) != 0) | ((rep & rm) != 0)
        take = cond & ~done
        lvl[take] = rule.level
        done |= cond
    lvl[~present] = 0
    return lvl


def _lattice_columns(name: str, spec: VariableQc, x, s, n_s):
    """Native unit (coarsest lattice holding >= 90 % of non-zero values) and modal step."""
    import numpy as np

    out_unit = np.full(n_s, None, object)
    out_frac = np.full(n_s, np.nan)
    quantum = np.full(n_s, np.nan)
    ok = ~np.isnan(x)
    if not ok.any():
        return out_unit, out_frac, quantum
    xs, ss = x[ok], s[ok]
    nz = xs != 0
    n_nz = np.bincount(ss[nz], minlength=n_s)
    for label, scale, offset, stp in LATTICES.get(spec.unit or "", []):
        y = (xs[nz] * scale + offset) / stp
        tol = 0.01 if scale == 1 else 0.1
        on = np.abs(y - np.round(y)) < tol
        with np.errstate(invalid="ignore", divide="ignore"):
            frac = np.bincount(ss[nz][on], minlength=n_s) / n_nz
        pick = (frac >= 0.9) & np.equal(out_unit, None)
        out_unit[pick] = label
        out_frac[pick] = frac[pick]
    # modal non-zero change between consecutive values of a station
    same = np.zeros(len(xs), bool)
    same[1:] = ss[1:] == ss[:-1]
    dx = np.zeros(len(xs))
    dx[1:] = np.abs(np.diff(xs))
    m = same & (dx > 1e-9)
    if m.any():
        import polars as pl

        q = (pl.DataFrame({"s": ss[m], "d": np.round(dx[m], 4)})
             .group_by("s", "d").len().sort("len", "d", descending=[True, False])
             .group_by("s", maintain_order=True).first())
        quantum[q["s"].to_numpy()] = q["d"].to_numpy()
    return out_unit, out_frac, quantum


def qc_month(args: tuple) -> dict:
    """One month: read it plus margins, flag every report, write flags and reliability."""
    import resource

    import numpy as np
    import polars as pl

    from meteopack.derive import qc_checks as ck

    cfg_json, root, month_iso, tables = args
    cfg = QcConfig.model_validate_json(cfg_json)
    debug = bool(os.environ.get("METEOPACK_QC_DEBUG"))
    root = Path(root)
    started = time.time()
    timing: dict[str, float] = {}
    lo = datetime.fromisoformat(month_iso)
    hi = lo.replace(year=lo.year + (lo.month == 12), month=lo.month % 12 + 1)
    margin = timedelta(seconds=cfg.margin)
    names = list(cfg.variables)
    cols = sorted({"station_id", "time", *names, "Latitude", "Longitude", "IsOfficial",
                   "LocalReportEndDateTime", "source_member"})
    frame = _read(root, cfg, lo, hi, lo - margin, hi + margin, cols)
    if frame is None or not frame["in_month"].any():
        return {"month": month_iso, "rows": 0}
    timing["read"] = time.time() - started
    if debug:
        print(f"{month_iso} read {timing['read']:.1f} s", flush=True)
    n = frame.height
    t = frame["time"].dt.epoch("us").to_numpy() / 1e6
    sid = (frame["station_id"] != frame["station_id"].shift(1)).fill_null(True).cum_sum() \
        .to_numpy().astype(np.int32)
    xs = {v: frame[v].cast(pl.Float64).fill_null(np.nan).to_numpy() for v in names}
    fail = {v: np.zeros(n, np.uint32) for v in names}
    ne = {v: np.zeros(n, np.uint32) for v in names}
    t0 = time.time()
    for v, spec in cfg.variables.items():
        _within_station(v, spec, xs[v], t, sid, fail[v], ne[v])
    dp = cfg.dew_point
    if dp is not None and {dp.dew_point, dp.temperature} <= set(names):
        td, ta = xs[dp.dew_point], xs[dp.temperature]
        gross = np.uint32(_bit("range", FAIL_BITS) | _bit("extreme", FAIL_BITS)
                          | _bit("sentinel", FAIL_BITS))
        clean_t = (fail[dp.temperature] & gross) == 0
        if dp.gt_t_tolerance is not None:
            fail[dp.dew_point][(td > ta + dp.gt_t_tolerance) & clean_t] |= _bit("td_gt_t",
                                                                               FAIL_BITS)
        if dp.mismatch_tolerance is not None and dp.humidity in names:
            rh = xs[dp.humidity]
            clean = clean_t & ((fail[dp.humidity] & gross) == 0) & (rh > 0)
            calc = ck.dew_point_magnus(ta, rh)
            bad = clean & (np.abs(td - calc) > dp.mismatch_tolerance)
            fail[dp.dew_point][bad] |= _bit("td_mismatch", FAIL_BITS)
    sr = cfg.stale_repeat
    fields = [f for f in sr.fields if f in xs]
    if fields:
        stale = ck.stale_repeat([xs[f] for f in fields], t, sid, sr.gap_s)
        for v, spec in cfg.variables.items():
            if spec.stale_repeat:
                fail[v][stale & ~np.isnan(xs[v])] |= _bit("stale_repeat", FAIL_BITS)
    timing["within_station"] = time.time() - t0
    if debug:
        print(f"{month_iso} within_station {timing['within_station']:.1f} s", flush=True)

    # restrict to the month from here on
    inm = frame["in_month"].to_numpy()
    rows = np.flatnonzero(inm)
    fr = frame.filter(pl.col("in_month")).drop(names)
    rid_in = fr["rid"].to_numpy()
    del frame                                   # the margins are done with
    year, month = lo.year, lo.month
    sm = pl.read_parquet(tables["station_months"]).filter(
        (pl.col("year") == year) & (pl.col("month") == month))
    hours_f = _month_file(root, cfg.input, lo, "hours")
    hours = pl.read_parquet(hours_f) if hours_f.exists() else None
    t0 = time.time()
    rep_in = np.zeros(len(rows), np.uint32)
    _report_bits(cfg, fr, sm, hours, {"t": t[rows], "sid": sid[rows]}, rep_in)
    timing["report"] = time.time() - t0
    if debug:
        print(f"{month_iso} report {timing['report']:.1f} s", flush=True)

    # station index for the month (network tests, rain, reliability)
    # a report without a station id (493 in 2015-2019) is a station of its own kind
    ids = fr["station_id"].fill_null("").unique().sort()
    s_map = pl.DataFrame({"station_id": ids, "s": np.arange(len(ids), dtype=np.int32)})
    s_in = fr.select(pl.col("station_id").fill_null("")).join(
        s_map, on="station_id", how="left", maintain_order="left")["s"].to_numpy()
    pos = pl.DataFrame({"station_id": ids}).join(sm, on="station_id", how="left",
                                                 maintain_order="left")
    lat = pos["lat"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    lon = pos["lon"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    elev = pos["height"].cast(pl.Float64).fill_null(0.0).to_numpy()
    xy = _xy(lat, lon)
    n_s = len(ids)
    step = cfg.network.step_seconds
    lo_s = lo.timestamp()
    t_in = t[rows]
    j_in = np.ceil((t_in - lo_s) / step).astype(np.int64)
    n_j = int((hi.timestamp() - lo_s) // step) + 1
    fail_in = {v: fail[v][rows] for v in names}
    ne_in = {v: ne[v][rows] for v in names}
    scores: dict[str, tuple] = {}
    station_stats: dict[str, dict] = {}

    t0 = time.time()
    for v, spec in cfg.variables.items():
        if spec.network():
            station_stats[v] = {"timing": timing}
            _network(cfg, v, spec, xs[v][rows], s_in, j_in, t_in, lo_s, n_s, n_j, xy, lat, lon,
                     elev, fail_in[v], ne_in[v], scores, station_stats[v])
    timing["network"] = time.time() - t0
    if debug:
        print(f"{month_iso} network {timing['network']:.1f} s", flush=True)

    increment = None
    if cfg.rain is not None:
        t0 = time.time()
        rq = cfg.rain
        look = (rq.pwsqc.so.max_window_days * 86400
                if rq.pwsqc is not None and rq.pwsqc.so is not None else 0)
        a = min(lo - margin, lo - timedelta(seconds=look + 86400))
        croot = _read(root, cfg, lo, hi, a, hi + margin, ["station_id", "time", rq.counter])
        res: dict = {}
        inc = _rain_counter(cfg, croot, lo, hi, fail_in[rq.counter], rid_in, res)
        del croot
        increment = res["increment"]
        timing["rain_counter"] = time.time() - t0
        if debug:
            print(f"{month_iso} rain_counter {timing['rain_counter']:.1f} s", flush=True)
        if rq.pwsqc is not None:
            has_c = np.zeros(n_s, bool)
            has_c[np.unique(s_in[~np.isnan(xs[rq.counter][rows])])] = True
            _pwsqc(cfg, inc, ids.to_numpy(), xy, has_c, lo, hi, fail_in[rq.counter],
                   ne_in[rq.counter], timing)

    # levels, scores, output
    t0 = time.time()
    out_cols: dict[str, Any] = {
        "station_id": fr["station_id"], "time": fr["time"],
        "l1_row": fr["l1_row"].cast(pl.UInt32), "qc_report": pl.Series(rep_in, dtype=pl.UInt32)}
    levels = {}
    for v in names:
        present = ~np.isnan(xs[v][rows])
        levels[v] = _levels(cfg, fail_in[v], ne_in[v], rep_in, present)
        out_cols[f"{v}_qc_fail"] = pl.Series(fail_in[v], dtype=pl.UInt32)
        out_cols[f"{v}_qc_ne"] = pl.Series(ne_in[v], dtype=pl.UInt32)
        out_cols[f"{v}_qc_level"] = pl.Series(levels[v], dtype=pl.UInt8)
    if increment is not None:
        out_cols["rainfall_increment"] = pl.Series(increment, dtype=pl.Float32).fill_nan(None)
    for k, (idx, val) in scores.items():
        col = np.full(len(rows), np.nan, np.float32)
        col[idx] = val
        out_cols[k] = pl.Series(col, dtype=pl.Float32).fill_nan(None)
    out = pl.DataFrame(out_cols).with_columns(
        pl.lit(cfg.params_hash()).cast(pl.Categorical).alias("qc_config_hash")).sort("l1_row")
    timing["levels"] = time.time() - t0
    if debug:
        print(f"{month_iso} levels {timing['levels']:.1f} s", flush=True)

    # reliability (station-month)
    t0 = time.time()
    rel = _reliability(cfg, fr, s_in, n_s, ids, sm, xs, rows, fail_in, ne_in, levels, rep_in,
                       station_stats)
    timing["reliability"] = time.time() - t0
    if debug:
        print(f"{month_iso} reliability {timing['reliability']:.1f} s", flush=True)

    # stats
    official = fr["IsOfficial"].fill_null(False).to_numpy() if "IsOfficial" in fr.columns \
        else np.zeros(len(rows), bool)
    stats = _month_stats(cfg, year, month, xs, rows, fail_in, ne_in, levels, rep_in, official)

    t0 = time.time()
    store = _store_path(root, cfg)
    station_time = fr.select(pl.struct("station_id", "time").is_duplicated().sum()).item()
    run = {"month": month_iso, "rows": out.height, "input_rows": n,
           "duplicate_station_time": int(station_time), "timing": timing,
           "seconds": time.time() - started,
           "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    stats_df = pl.DataFrame(stats)
    for sub, df in (("observations", out), ("reliability", rel), ("stats", stats_df)):
        target = store / sub / f"year={year:04d}" / f"month={month:02d}" / "data.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        df.write_parquet(tmp, compression="zstd", compression_level=cfg.compression_level,
                         statistics=True)
        os.replace(tmp, target)
    timing["write"] = time.time() - t0
    if debug:
        print(f"{month_iso} write {timing['write']:.1f} s", flush=True)
    run["seconds"] = time.time() - started
    target = store / "runs" / f"year={year:04d}" / f"month={month:02d}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(run, indent=1))
    return {**run, "stats": stats}


def _month_stats(cfg, year, month, xs, rows, fail, ne, levels, rep, official):
    """Counts per variable x bit x official/private: the flag-rate tables."""
    import numpy as np

    out = []
    for off in (False, True):
        sel = official == off
        base = {"year": year, "month": month, "official": off}
        out.append({**base, "variable": "*", "kind": "reports", "name": "n",
                    "count": int(sel.sum())})
        for b, k in REPORT_BITS.items():
            out.append({**base, "variable": "*", "kind": "report", "name": b,
                        "count": int(((rep[sel] >> k) & 1).sum())})
        for v in cfg.variables:
            present = ~np.isnan(xs[v][rows]) & sel
            out.append({**base, "variable": v, "kind": "values", "name": "n",
                        "count": int(present.sum())})
            f, e = fail[v][present], ne[v][present]
            for kind, arr, table in (("fail", f, FAIL_BITS), ("ne", e, NE_BITS)):
                seen = int(np.bitwise_or.reduce(arr)) if len(arr) else 0
                for b, k in table.items():
                    if seen >> k & 1:
                        out.append({**base, "variable": v, "kind": kind, "name": b,
                                    "count": int(((arr >> np.uint32(k)) & 1).sum())})
            out.append({**base, "variable": v, "kind": "fail", "name": "any",
                        "count": int((f != 0).sum())})
            lv = np.bincount(levels[v][present], minlength=5)
            for k, c in enumerate(lv):
                out.append({**base, "variable": v, "kind": "level", "name": str(k),
                            "count": int(c)})
    return out


def _reliability(cfg, fr, s_in, n_s, ids, sm, xs, rows, fail, ne, levels, rep, station_stats):
    """Station-month x variable: share of values per bit and level, r, lattice, tier."""
    import numpy as np
    import polars as pl

    n_rep = np.bincount(s_in, minlength=n_s)
    with np.errstate(invalid="ignore", divide="ignore"):
        def frac_rep(bit):
            return np.bincount(s_in, weights=((rep >> REPORT_BITS[bit]) & 1),
                               minlength=n_s) / n_rep
        base = {"station_id": ids.to_numpy(), "n_reports": n_rep,
                "frac_position_jump": frac_rep("position_jump"),
                "frac_near_duplicate": frac_rep("near_duplicate"),
                "frac_capped_hour": frac_rep("capped_hour"),
                "frac_local_time_inconsistent": frac_rep("local_time_inconsistent")}
    meta = pl.DataFrame({"station_id": ids}).join(
        sm.select("station_id", "lat", "lon", "height", "n_positions", "official", "dup_coords"),
        on="station_id", how="left", maintain_order="left")
    parts = []
    ok_share = {}
    for v, spec in cfg.variables.items():
        x = xs[v][rows]
        present = ~np.isnan(x)
        nv = np.bincount(s_in[present], minlength=n_s)
        if not nv.any():
            continue
        sp = s_in[present]
        fv, ev, lv_ = fail[v][present], ne[v][present], levels[v][present]
        cols: dict[str, Any] = {"station_id": ids.to_numpy(), "variable": v, "n_values": nv}
        with np.errstate(invalid="ignore", divide="ignore"):
            for kind, arr, table in (("fail", fv, FAIL_BITS), ("ne", ev, NE_BITS)):
                seen = int(np.bitwise_or.reduce(arr)) if len(arr) else 0
                for b, k in table.items():
                    if seen >> k & 1:
                        hit = sp[((arr >> np.uint32(k)) & 1).astype(bool)]
                        cols[f"{kind}_{b}"] = np.bincount(hit, minlength=n_s) / nv
                    else:
                        cols[f"{kind}_{b}"] = np.where(nv > 0, 0.0, np.nan)
            cols["fail_any"] = np.bincount(sp[fv != 0], minlength=n_s) / nv
            for lv in range(1, 5):
                cols[f"frac_level_{lv}"] = np.bincount(sp[lv_ == lv], minlength=n_s) / nv
            ok = np.bincount(sp[lv_ >= cfg.reliability.ok_level], minlength=n_s) / nv
        cols["frac_ok"] = ok
        ok_share[v] = (nv, ok)
        st = station_stats.get(v, {})
        cols["r_network"] = st.get("r_network", np.full(n_s, np.nan))
        cols["n_pairs_network"] = st.get("n_pairs_network", np.zeros(n_s, int))
        cols["n_neighbours"] = st.get("n_neighbours", np.full(n_s, -1))
        unit, frac_l, quantum = _lattice_columns(v, spec, x, s_in, n_s)
        cols["native_unit"] = pl.Series(unit.tolist(), dtype=pl.String)
        cols["frac_on_native_lattice"] = frac_l
        cols["quantum"] = quantum
        df = pl.DataFrame(cols, strict=False).filter(pl.col("n_values") > 0)
        parts.append(df)
    # trust tier per station-month
    rc = cfg.reliability
    official = meta["official"].fill_null(False).to_numpy()
    tier = np.full(n_s, 3, np.int8)
    good = np.ones(n_s, bool)
    any_core = np.zeros(n_s, bool)
    for v in rc.core:
        if v in ok_share:
            nv, ok = ok_share[v]
            has = nv > 0
            any_core |= has
            good &= ~has | (ok >= rc.tier2_min_fraction)
    if not rc.core or not any(v in ok_share for v in rc.core):
        any_core[:] = False
    for v, (nv, ok) in ok_share.items():          # stations reporting no core variable
        has = (nv > 0) & ~any_core
        good &= ~has | (ok >= rc.tier2_min_fraction)
    tier[good] = 2
    tier[official] = 1
    station = pl.DataFrame({**base, "trust_tier": tier}).join(meta, on="station_id", how="left")
    if not parts:
        return station.with_columns(pl.lit(None, pl.String).alias("variable"))
    rel = pl.concat(parts, how="diagonal_relaxed")
    return rel.join(station, on="station_id", how="left")


def _store_path(root: Path, cfg: QcConfig) -> Path:
    from meteopack.core.config import layer_path

    return layer_path(Path(root), cfg.layer, *cfg.store_id.split("/"),
                      category=cfg.category)


# ---------------------------------------------------------------- the driver


def build(cfg: QcConfig, root: Path, *, window: tuple[datetime, datetime] | None = None,
          workers: int | None = None, threads: int = 2, refresh: bool = False) -> dict:
    """Flag every month of the input (or of `window`) and write the QC store.

    Refuses to write into a store built with other parameters (a store whose
    months disagree about what a bit means is worse than none) unless
    `refresh`, which removes it first.
    """
    import concurrent.futures as cf
    import multiprocessing
    import shutil

    from meteopack.core.store import points

    if any(v.sct is not None for v in cfg.variables.values()):
        try:
            import titanlib  # noqa: F401
        except ImportError as exc:
            raise SystemExit("an `sct` test is configured but titanlib is not installed "
                             "(uv sync --extra qc), or remove the sct entries") from exc
    started = time.time()
    store = _store_path(root, cfg)
    meta_path = store / "qc.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text()).get("params_hash")
        if old != cfg.params_hash() or refresh:
            if old != cfg.params_hash() and not refresh:
                raise SystemExit(
                    f"{store} was built with different parameters ({old} != "
                    f"{cfg.params_hash()}); pass --refresh to replace it, or --set name=... "
                    f"to write a new store")
            for sub in ("observations", "reliability", "stats", "runs"):
                shutil.rmtree(store / sub, ignore_errors=True)
            meta_path.unlink()
    if not meta_path.exists():                 # claim the store for these parameters
        store.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"params_hash": cfg.params_hash()}))
    input_store = points.store_path(root, cfg.input)
    t0 = time.time()
    tables = prepare_tables(cfg, input_store, store)
    t_tables = time.time() - t0

    base = input_store / points.OBSERVATIONS_DIR
    on_disk = sorted(
        datetime(int(p.parent.name.split("=")[1]), int(p.name.split("=")[1]), 1, tzinfo=UTC)
        for p in base.glob("year=*/month=*") if any(p.glob("*.parquet")))
    if not on_disk:
        raise SystemExit(f"no input partitions under {base}")
    months = on_disk if window is None else [m for m in _month_starts(*window) if m in on_disk]
    cfg_json = cfg.model_dump_json()
    tasks = [(cfg_json, str(root), m.isoformat(), tables) for m in months]
    workers = workers or max(1, len(os.sched_getaffinity(0)) // max(1, threads))
    workers = min(workers, len(tasks)) or 1
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    # biggest months first: the pool's tail is then short ones
    tasks.sort(key=lambda a: -_month_file(root, cfg.input, datetime.fromisoformat(a[2]))
               .stat().st_size)
    with cf.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        results = list(ex.map(qc_month, tasks))
    wall = time.time() - started

    meta = finalize(cfg, root, tables=tables, run={
        "wall_s": wall, "tables_s": t_tables, "workers": workers, "polars_threads": threads,
        "months_built": len(months),
        "window": None if window is None else [w.isoformat() for w in window]})
    meta["rows_written_this_run"] = sum(r.get("rows", 0) for r in results)
    return meta


def _atomic_parquet(df, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    df.write_parquet(tmp)
    os.replace(tmp, path)


def _concat_months(store: Path, sub: str):
    import polars as pl

    files = sorted((store / sub).rglob("*.parquet"))
    if not files:
        return None
    return pl.concat([pl.read_parquet(f).with_columns(
        pl.lit(int(f.parent.parent.name.split("=")[1])).alias("year"),
        pl.lit(int(f.parent.name.split("=")[1])).alias("month")) for f in files],
        how="diagonal_relaxed")


def finalize(cfg: QcConfig, root: Path, *, tables: dict | None = None,
             run: dict | None = None) -> dict:
    """Store-wide tables, qc.json, manifest and README from every month on disk.

    Built from the per-month files, so several windowed runs (e.g. one Slurm
    job per few years) end in the same store as one run over everything.
    """

    from meteopack.core.store import manifest, points

    store = _store_path(root, cfg)
    rel = _concat_months(store, "reliability")
    if rel is not None:
        _atomic_parquet(rel, store / "reliability.parquet")
    stats = _concat_months(store, "stats")
    if stats is not None:
        _atomic_parquet(stats.sort("year", "month", "variable", "kind", "name"),
                        store / "stats.parquet")
    runs = [json.loads(p.read_text()) for p in sorted((store / "runs").rglob("*.json"))]
    timing: dict[str, float] = {}
    for r in runs:
        for k, v in (r.get("timing") or {}).items():
            timing[k] = timing.get(k, 0) + v
    n_total = sum(r.get("rows", 0) for r in runs)
    in_manifest = manifest.read(points.store_path(root, cfg.input))
    meta = {
        "store_id": cfg.store_id, "input": cfg.input,
        "input_content_hash": in_manifest.content_hash if in_manifest else None,
        "input_config_hash": in_manifest.config_hash if in_manifest else None,
        "params_hash": cfg.params_hash(), "params": json.loads(cfg.model_dump_json()),
        "margin_s": cfg.margin,
        "bits": {"fail": FAIL_BITS, "ne": NE_BITS, "report": REPORT_BITS},
        "bit_meaning": BIT_MEANING,
        "levels": [r.model_dump() for r in cfg.levels] + [
            {"level": cfg.default_level, "name": "good (no rule matched)"},
            {"level": 0, "name": "no value"}],
        "tables": tables, "rows_total": n_total, "months": len(runs),
        "duplicate_station_time": sum(r.get("duplicate_station_time", 0) for r in runs),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "last_run": run,
        "months_summary": {
            "stage_s_summed": timing,
            "seconds_summed": sum(r.get("seconds", 0) for r in runs),
            "slowest_month_s": max((r.get("seconds", 0) for r in runs), default=0),
            "max_worker_rss_mib": max((r.get("rss_mib", 0) for r in runs), default=0),
            "input_rows": sum(r.get("input_rows", 0) for r in runs)},
    }
    store.mkdir(parents=True, exist_ok=True)
    tmp = store / f".qc.json.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(meta, indent=2, default=str))
    os.replace(tmp, store / "qc.json")
    manifest.write(store, manifest.Manifest.new(
        cfg.store_id, cfg.layer, "points", cfg, n_records=n_total, inputs=[cfg.input],
        extra={"params_hash": cfg.params_hash(), "aligned_to": cfg.input}))
    (store / "README.md").write_text(_readme(cfg, meta, n_total))
    return meta


def _readme(cfg: QcConfig, meta: dict, n_total: int) -> str:
    lines = [f"# {cfg.store_id}", "",
             cfg.description or "Quality-control flags.", "",
             (f"Aligned row for row with `{cfg.input}`: the same `year=/month=` partitions, "
              "the rows in the same order (`l1_row` = position in the level-1 month file). "
              "`(station_id, time)` is unique except for 5,694 reports (2015-2025) sent twice "
              "at one instant; the second of each pair has `near_duplicate`. Nothing in level "
              "1 is changed or dropped."),
             "",
             (f"Parameters hash `{cfg.params_hash()}` (every parameter: `qc.json`); "
              f"{n_total:,} rows."), "",
             ("Per variable: `<var>_qc_fail` (uint32, failed tests), `<var>_qc_ne` (uint32, "
              "tests that could not be evaluated), `<var>_qc_level` (uint8). Per report: "
              "`qc_report` (uint32)."), ""]
    for kind, table in (("fail", FAIL_BITS), ("ne", NE_BITS), ("report", REPORT_BITS)):
        lines += [f"## `{kind}` bits", "", "| bit | name | meaning |", "|---|---|---|"]
        lines += [f"| {k} | `{b}` | {BIT_MEANING[kind][b]} |" for b, k in table.items()]
        lines.append("")
    lines += ["## Levels", "", "| level | name | when |", "|---|---|---|", "| 0 | no value | |"]
    for r in cfg.levels:
        when = "; ".join(f"{k}: {', '.join(getattr(r, k))}" for k in ("fail", "ne", "report")
                         if getattr(r, k))
        lines.append(f"| {r.level} | {r.name} | any of {when} |")
    lines += [f"| {cfg.default_level} | good | no rule matched |", "",
              ("Tables: `reliability.parquet` (station-month x variable: share of values per "
               "bit and level, r with the network median, native unit and quantum, trust "
               "tier), `stats.parquet` (counts per month, variable, bit, official/private), "
               "`station_months.parquet`, `neighbours.parquet`."), ""]
    return "\n".join(lines)


def read_month(root: Path, year: int, month: int, columns: list[str] | None = None,
               qc_store: str = "wow_nl/qc/v1", input_store: str = "wow_nl/reports/v1",
               layer: str = "packed"):
    """Level 1 and its QC flags for one month, side by side (checked to be aligned)."""
    import polars as pl

    from meteopack.core.config import layer_path

    m = datetime(year, month, 1)
    l1 = pl.read_parquet(_month_file(root, input_store, m), columns=columns)
    q = pl.read_parquet(layer_path(Path(root), layer, *qc_store.split("/")) / "observations"
                        / f"year={year:04d}" / f"month={month:02d}" / "data.parquet")
    if q.height != l1.height:
        raise ValueError(f"QC month has {q.height} rows, level 1 {l1.height}: rebuild the QC")
    if columns is None or {"station_id", "time"} <= set(columns):
        same = (l1["station_id"] == q["station_id"]).all() and (l1["time"] == q["time"]).all()
        if not same:
            raise ValueError("QC rows are not aligned with level 1: rebuild the QC store")
    return pl.concat([l1, q.drop([c for c in ("station_id", "time") if c in l1.columns])],
                     how="horizontal")


__all__ = ["FAIL_BITS", "NE_BITS", "REPORT_BITS", "QcConfig", "build", "qc_month",
           "read_month"]
