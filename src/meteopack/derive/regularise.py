"""Irregular points store -> regular grid, with every choice a parameter.

The packed WOW-NL store keeps every report at the time it was sent: seconds to
hours apart, at any phase. Every use of it needs a regular axis, and every way
of getting one is a choice - which bin a report belongs to, what represents a
bin, how far to look for a value, how to turn a running rain counter into
amounts. Those choices are what the group is still deciding (a literature
review is under way), so none is baked in: a derive config names them, the
CLI overrides any of them (`--set step_seconds=300`), and the output records
all of them, so two derived stores can always be told apart.

What a config sets (see `RegulariseConfig`; configs/derive/wow_nl_*.yaml):

* `step_seconds` - the grid step (60, 300, 600, 3600 ...).
* `label` - which instant names a bin. `end` (default): bin T holds
  (T - step, T], KNMI radar's convention - an accumulation is stamped at the
  end of its interval. `start`: [T, T + step). `centre`: [T - step/2, T + step/2).
* `origin` - where the grid is anchored. Default the Unix epoch (bins at :00,
  :05, ...); the radar day starts 08:05 UTC, which for a 5-minute grid is the
  same grid, and for a 24 h step is not.
* per variable, any of these `methods`, each written as `<variable>_<method>`:

  | method | value at T | causal? |
  |---|---|---|
  | `last` | last non-null report in the bin, + `_last_age_s` = T - its time | with `label: end` |
  | `mean`, `min`, `max`, `first` | over the bin's non-null reports, + `_n` | with `label: end` |
  | `nearest` | report closest to T within `tolerance_s` (either side), + `_nearest_dt_s` | no |
  | `linear` | interpolated between the reports either side of T, if they are at most `max_gap_s` apart | no |

  "Causal" means the value uses no report after T; the metadata marks every
  output column, because a nowcasting target built from non-causal inputs
  leaks the future.
* `rain` - a rule (`naive` | `robust`, from wow_nl_analysis/scripts/
  rain_counter.py) turning the running counter into increments at each
  report, summed per bin as `<output>`, with `<output>_span_s` - how many
  seconds of counter history those increments cover - and `<output>_n`.
* `stages` - per-report steps applied in order *before* binning: where QC
  flagging and homogenisation will slot in once the literature review has
  said what they should be. One demonstration stage exists (`range_mask`);
  a new stage is a function registered in `STAGES`.

Always written: `n_reports` per (station, bin).

Work is one task per month of the input, in a process pool. Each task reads
its month plus a margin on either side (`margin_s`, at least one step and
every tolerance, gap and rain look-back), computes everything over that, and
keeps the bins whose label falls in the month - so bins at month boundaries,
nearest/linear across a boundary and the rain rule's look-back are the same
as if the whole archive were one frame. Output layout is the points-store
layout (`observations/year=Y/month=M/data.parquet`, sorted by station, time),
so `meteopack.core.store.points.read` and `to_matrix` read it as they read any
regular store.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator

#: Methods computed from the reports inside a bin.
BIN_METHODS = ("last", "first", "mean", "min", "max")
#: Methods that reach outside the bin.
POINT_METHODS = ("nearest", "linear")
METHODS = BIN_METHODS + POINT_METHODS


class VariableSpec(BaseModel):
    model_config = {"extra": "forbid"}

    methods: list[Literal["last", "first", "mean", "min", "max", "nearest", "linear"]] = Field(
        default_factory=lambda: ["last"])
    #: `nearest`: how far from T a report may be, seconds, either side. At most
    #: half a step, so a report is nearest to at most one grid point.
    tolerance_s: float | None = None
    #: `linear`: the longest gap between two reports that is interpolated across.
    max_gap_s: float | None = None


class RainSpec(BaseModel):
    """A rule for turning a running counter into amounts.

    `naive`: increment = a - a_prev; a drop of more than `drop_mm` is a reset
    (increment = a); a smaller drop is 0. `robust`: a drop of more than
    `drop_mm` is a *transient* - ignored - when one of the next `lookahead`
    reports of the same local and UTC day climbs back to at least
    a_prev - drop_mm; an up-jump that falls straight back is a *spike*, also
    ignored; confirmed resets start a new segment, and inside a segment the
    increment is the rise of the running maximum. Both are exactly the SQL of
    wow_nl_analysis/scripts/rain_counter.py. `robust` looks ahead, so it is
    not causal.
    """

    model_config = {"extra": "forbid"}

    counter: str = "rainfall_counter"
    rule: Literal["naive", "robust"] = "robust"
    output: str = "rainfall_amount"
    drop_mm: float = 0.25
    lookahead: Annotated[int, Field(ge=1)] = 3
    #: Counter values outside this range are not part of the counter's history
    #: (rain_counter.py: BETWEEN 0 AND 1000).
    valid_min: float | None = 0.0
    valid_max: float | None = 1000.0
    #: An increment across a longer gap is not attributed to any bin (and is
    #: counted). None: any gap the margin can see.
    max_gap_s: float | None = None
    #: The day a counter resets on; transients never cross it.
    local_timezone: str = "Europe/Amsterdam"


class StageSpec(BaseModel):
    """One per-report step before binning: `kind` names it in `STAGES`."""

    model_config = {"extra": "allow"}

    kind: str


class RegulariseConfig(BaseModel):
    """A regularised dataset built from one irregular points store."""

    model_config = {"extra": "forbid"}

    kind: Literal["regularise"] = "regularise"
    name: str
    version: int = 1
    description: str | None = None
    #: The input points store, `source/product/vN`.
    input: str
    step_seconds: Annotated[int, Field(gt=0)]
    label: Literal["end", "start", "centre"] = "end"
    origin: datetime = datetime(1970, 1, 1, tzinfo=UTC)
    variables: dict[str, VariableSpec] = Field(default_factory=dict)
    rain: RainSpec | None = None
    stages: list[StageSpec] = Field(default_factory=list)
    #: Only these stations (None: all).
    stations: list[str] | None = None
    #: Read this much input either side of a month. None: the largest of one
    #: step, every tolerance and gap, and two days when rain is derived.
    margin_s: Annotated[int, Field(ge=0)] | None = None
    #: float32 halves the output; float64 keeps every input value exactly.
    dtype: Literal["float32", "float64"] = "float64"
    compression_level: int = 3
    #: polars threads per worker process; workers = CPUs // threads. How it
    #: runs, not what it computes, so not part of `params_hash`.
    threads: Annotated[int, Field(ge=1)] = 2

    @model_validator(mode="after")
    def _check(self) -> RegulariseConfig:
        for name, spec in self.variables.items():
            if "nearest" in spec.methods:
                if spec.tolerance_s is None:
                    raise ValueError(f"{name}: nearest needs tolerance_s")
                if spec.tolerance_s > self.step_seconds / 2:
                    raise ValueError(f"{name}: tolerance_s {spec.tolerance_s} is more than half "
                                     f"a step; a report would be nearest to two grid points")
            if "linear" in spec.methods and spec.max_gap_s is None:
                raise ValueError(f"{name}: linear needs max_gap_s")
        for stage in self.stages:
            if stage.kind not in STAGES:
                raise ValueError(f"unknown stage {stage.kind!r}; known: {sorted(STAGES)}")
        if self.origin.tzinfo is None:
            self.origin = self.origin.replace(tzinfo=UTC)
        return self

    @property
    def store_id(self) -> str:
        return f"{self.name}/v{self.version}"

    @property
    def margin(self) -> int:
        """Seconds of input read either side of a month."""
        if self.margin_s is not None:
            return max(self.margin_s, self.step_seconds)
        need = [self.step_seconds]
        for spec in self.variables.values():
            need += [spec.tolerance_s or 0, spec.max_gap_s or 0]
        if self.rain is not None:
            need += [2 * 86400, self.rain.max_gap_s or 0]
        return int(max(need)) + 1

    def params_hash(self) -> str:
        """What the values depend on - not the name, description or margin."""
        payload = self.model_dump(mode="json", exclude={"name", "version", "description",
                                                        "margin_s", "threads"})
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def outputs(self) -> dict[str, dict[str, Any]]:
        """Every output column: what it is and whether it is causal."""
        binned_causal = self.label == "end"
        cols: dict[str, dict[str, Any]] = {
            "n_reports": {"what": "reports in the bin", "causal": binned_causal}}
        for name, spec in self.variables.items():
            for m in spec.methods:
                if m in BIN_METHODS:
                    cols[f"{name}_{m}"] = {"what": f"{m} of the bin's non-null {name}",
                                           "causal": binned_causal}
                elif m == "nearest":
                    cols[f"{name}_nearest"] = {
                        "what": f"{name} of the report nearest to T within {spec.tolerance_s}"
                                f" s", "causal": False}
                    cols[f"{name}_nearest_dt_s"] = {"what": "that report's time - T",
                                                    "causal": False}
                else:
                    cols[f"{name}_linear"] = {
                        "what": f"{name} linearly interpolated at T between reports at most "
                                f"{spec.max_gap_s} s apart", "causal": False}
            if "last" in spec.methods:
                cols[f"{name}_last_age_s"] = {"what": "T - time of the last report",
                                              "causal": binned_causal}
            if set(spec.methods) & {"mean", "min", "max", "first"}:
                cols[f"{name}_n"] = {"what": f"non-null {name} reports in the bin",
                                     "causal": binned_causal}
        if self.rain is not None:
            r = self.rain
            causal = binned_causal and r.rule == "naive"
            cols[r.output] = {"what": f"sum of {r.rule} counter increments at reports in the "
                                      f"bin, mm", "causal": causal}
            cols[f"{r.output}_span_s"] = {"what": "seconds of counter history those "
                                                  "increments cover", "causal": causal}
            cols[f"{r.output}_n"] = {"what": "increments summed", "causal": causal}
        return cols


# --------------------------------------------------------------------- stages


def stage_range_mask(frame, spec: StageSpec, report: dict):
    """Null `variable` where it lies outside [min, max]; count how many.

    A demonstration of the stage mechanism, not a QC decision: the thresholds
    are whatever the config says, and the default config says none.
    """
    import polars as pl

    params = spec.model_extra or {}
    var = params["variable"]
    lo, hi = params.get("min"), params.get("max")
    bad = pl.lit(False)
    if lo is not None:
        bad = bad | (pl.col(var) < lo)
    if hi is not None:
        bad = bad | (pl.col(var) > hi)
    n = int(frame.select(bad.fill_null(False).sum()).item())
    report[f"range_mask:{var}"] = report.get(f"range_mask:{var}", 0) + n
    return frame.with_columns(pl.when(bad.fill_null(False)).then(None)
                              .otherwise(pl.col(var)).alias(var))


#: kind -> fn(frame, spec, report) -> frame. Each runs on the reports (sorted
#: by station, time, margin included) before anything is binned. QC flagging
#: and homogenisation stages go here, each configured by its own parameters.
STAGES: dict[str, Callable] = {"range_mask": stage_range_mask}


# ----------------------------------------------------------------------- core


def _bin_index(off, step: int, label: str):
    """Integer bin number of each report from its µs offset to the origin."""
    if label == "end":          # (T - step, T]
        return -((-off) // step)
    if label == "start":        # [T, T + step)
        return off // step
    return (off + step // 2) // step   # centre: [T - step/2, T + step/2)


def rain_increments(frame, spec: RainSpec):
    """Per-report counter increments under `spec.rule`, for rows sorted by (station, time).

    A line-for-line port of rain_counter.py's window SQL, including its
    three-valued logic: a condition that is NULL in SQL (a lead past the end of
    a station's series) is null here too and counts as "not a reset", exactly
    as `CASE WHEN` treats it there.
    """
    import polars as pl

    c, thr = spec.counter, spec.drop_mm
    r = frame.select("station_id", "time", pl.col(c).alias("a"))
    r = r.filter(pl.col("a").is_not_null())
    if spec.valid_min is not None:
        r = r.filter(pl.col("a") >= spec.valid_min)
    if spec.valid_max is not None:
        r = r.filter(pl.col("a") <= spec.valid_max)
    st = "station_id"
    r = r.with_columns(
        pl.col("time").dt.replace_time_zone("UTC").dt.convert_time_zone(spec.local_timezone)
        .dt.date().alias("lday"),
        pl.col("time").dt.date().alias("uday"),
    )
    lag = lambda e, n=1: e.shift(n).over(st)
    r = r.with_columns(
        lag(pl.col("a")).alias("ap"),
        *(lag(pl.col("a"), -k).alias(f"a{k}") for k in range(1, spec.lookahead + 1)),
        lag(pl.col("lday")).alias("lday_p"), lag(pl.col("uday")).alias("uday_p"),
        lag(pl.col("lday"), -1).alias("lday_n"), lag(pl.col("uday"), -1).alias("uday_n"),
        ((pl.col("time") - lag(pl.col("time"))).dt.total_microseconds() / 1e6).alias("dt"),
    )
    a, ap = pl.col("a"), pl.col("ap")
    climbs = pl.lit(None, pl.Boolean)
    for k in range(1, spec.lookahead + 1):
        term = pl.col(f"a{k}") >= ap - thr
        climbs = term if k == 1 else (climbs | term)
    drop = a < ap - thr
    r = r.with_columns(
        drop.alias("drop"),
        (drop & (pl.col("lday") == pl.col("lday_p")) & (pl.col("uday") == pl.col("uday_p"))
         & climbs).alias("transient"),
        ((a > ap + thr) & (pl.col("a1") < a - thr) & (pl.col("a1") >= ap - thr)
         & (pl.col("a1") <= ap + thr) & (pl.col("lday_n") == pl.col("lday"))
         & (pl.col("uday_n") == pl.col("uday"))).alias("spike"),
        pl.when(ap.is_null()).then(None).when(a >= ap).then(a - ap)
        .when(a < ap - thr).then(a).otherwise(0.0).alias("inc_naive"),
    )
    if spec.rule == "naive":
        inc = pl.col("inc_naive")
        r = r.with_columns(pl.col("drop").fill_null(False).alias("reset"))
    else:
        r = r.with_columns((pl.col("drop") & ~pl.col("transient")).alias("reset"))
        r = r.with_columns(pl.col("reset").fill_null(False).cast(pl.Int64).cum_sum().over(st)
                           .alias("seg"))
        excluded = (pl.col("transient") | pl.col("spike")).fill_null(False)
        neg = float("-inf")
        r = r.with_columns(
            pl.when(excluded).then(None).otherwise(a).fill_null(neg).cum_max()
            .over([st, "seg"]).alias("cm"))
        r = r.with_columns(pl.when(pl.col("cm") == neg).then(None).otherwise(pl.col("cm"))
                           .alias("cm"))
        cm_prev = pl.col("cm").shift(1).over([st, "seg"])
        inc = (pl.when(ap.is_null()).then(None)
               .when(pl.col("reset").fill_null(False)).then(a)
               .otherwise(pl.col("cm").fill_null(0.0)
                          - pl.coalesce(cm_prev, pl.col("cm"), pl.lit(0.0))))
    r = r.with_columns(inc.alias("inc"))
    gaps = 0
    if spec.max_gap_s is not None:
        long = pl.col("dt") > spec.max_gap_s
        gaps = int(r.select((long & pl.col("inc").is_not_null()).sum()).item())
        r = r.with_columns(pl.when(long).then(None).otherwise(pl.col("inc")).alias("inc"))
    return r.select("station_id", "time", "inc", "dt", pl.col("reset").fill_null(False)), gaps


def regularise_frame(frame, cfg: RegulariseConfig, lo: datetime, hi: datetime,
                     report: dict):
    """Reports (sorted by station, time) -> one row per (station, bin) with T in [lo, hi)."""
    import polars as pl

    step = cfg.step_seconds * 1_000_000
    origin = int(cfg.origin.timestamp() * 1_000_000)
    for stage in cfg.stages:
        frame = STAGES[stage.kind](frame, stage, report)

    off = pl.col("time").dt.epoch("us") - origin
    k = _bin_index(off, step, cfg.label)
    to_time = lambda e: pl.from_epoch(e * step + origin, time_unit="us")
    frame = frame.with_columns(to_time(k).alias("__T"))
    keys = ["station_id", "__T"]

    aggs = [pl.len().cast(pl.Int32).alias("n_reports")]
    for name, spec in cfg.variables.items():
        v = pl.col(name)
        nn = v.is_not_null()
        for m in spec.methods:
            if m == "last":
                aggs += [v.drop_nulls().last().alias(f"{name}_last"),
                         pl.col("time").filter(nn).last().alias(f"__{name}_last_t")]
            elif m == "first":
                aggs.append(v.drop_nulls().first().alias(f"{name}_first"))
            elif m in ("mean", "min", "max"):
                aggs.append(getattr(v, m)().alias(f"{name}_{m}"))
        if set(spec.methods) & {"mean", "min", "max", "first"}:
            aggs.append(v.count().cast(pl.Int32).alias(f"{name}_n"))
    out = frame.group_by(keys).agg(aggs)
    for name, spec in cfg.variables.items():
        if "last" in spec.methods:
            out = out.with_columns(
                ((pl.col("__T") - pl.col(f"__{name}_last_t")).dt.total_microseconds() / 1e6)
                .alias(f"{name}_last_age_s")).drop(f"__{name}_last_t")

    extra = []
    for name, spec in cfg.variables.items():
        src = frame.filter(pl.col(name).is_not_null()).select("station_id", "time", name)
        if "nearest" in spec.methods:
            tol = int(spec.tolerance_s * 1_000_000)
            g = (off + step // 2) // step
            near = (src.with_columns(to_time(g).alias("__T"))
                    .with_columns(((pl.col("time") - pl.col("__T")).dt.total_microseconds())
                                  .alias("__dt"))
                    .filter(pl.col("__dt").abs() <= tol)
                    .sort(pl.col("__dt").abs(), "time")
                    .group_by(keys).first()
                    .select(*keys, pl.col(name).alias(f"{name}_nearest"),
                            (pl.col("__dt") / 1e6).alias(f"{name}_nearest_dt_s")))
            extra.append(near)
        if "linear" in spec.methods:
            gap = int(spec.max_gap_s * 1_000_000)
            t0 = pl.col("time").dt.epoch("us") - origin
            pairs = src.with_columns(
                pl.col("time").shift(-1).over("station_id").alias("__t1"),
                pl.col(name).shift(-1).over("station_id").alias("__v1"),
            ).with_columns(
                (pl.col("__t1").dt.epoch("us") - origin).alias("__o1"), t0.alias("__o0"),
            ).filter((pl.col("__o1") > pl.col("__o0"))
                     & (pl.col("__o1") - pl.col("__o0") <= gap))
            pairs = pairs.with_columns(
                (-((-pl.col("__o0")) // step)).alias("__k0"),
                (pl.col("__o1") // step).alias("__k1"),
            ).filter(pl.col("__k0") <= pl.col("__k1"))
            lin = (pairs.with_columns(pl.int_ranges("__k0", pl.col("__k1") + 1).alias("__k"))
                   .explode("__k")
                   .with_columns(
                       to_time(pl.col("__k")).alias("__T"),
                       (pl.col(name) + (pl.col("__v1") - pl.col(name))
                        * ((pl.col("__k") * step - pl.col("__o0"))
                           / (pl.col("__o1") - pl.col("__o0")))).alias(f"{name}_linear"))
                   .unique(subset=keys, keep="first", maintain_order=True)
                   .select(*keys, f"{name}_linear"))
            extra.append(lin)
    if cfg.rain is not None:
        inc, gaps = rain_increments(frame, cfg.rain)
        report["rain_increments_over_max_gap"] = report.get(
            "rain_increments_over_max_gap", 0) + gaps
        o = cfg.rain.output
        k_inc = _bin_index(pl.col("time").dt.epoch("us") - origin, step, cfg.label)
        rain = (inc.filter(pl.col("inc").is_not_null())
                .with_columns(to_time(k_inc).alias("__T"))
                .group_by(keys).agg(pl.col("inc").sum().alias(o),
                                    pl.col("dt").sum().alias(f"{o}_span_s"),
                                    pl.len().cast(pl.Int32).alias(f"{o}_n")))
        extra.append(rain)
    for e in extra:
        out = out.join(e, on=keys, how="full", coalesce=True)
    lo_n = lo.astimezone(UTC).replace(tzinfo=None)
    hi_n = hi.astimezone(UTC).replace(tzinfo=None)
    out = out.filter((pl.col("__T") >= lo_n) & (pl.col("__T") < hi_n))
    out = out.with_columns(pl.col("n_reports").fill_null(0)).rename({"__T": "time"})
    cols = ["station_id", "time", *[c for c in cfg.outputs() if c in out.columns]]
    out = out.select(cols)
    if cfg.dtype == "float32":
        out = out.with_columns(pl.col(pl.Float64).cast(pl.Float32))
    return out.sort("station_id", "time")


# ---------------------------------------------------------------- the driver


def _month_starts(lo: datetime, hi: datetime) -> list[datetime]:
    m = lo.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    out = []
    while m < hi:
        out.append(m)
        m = m.replace(year=m.year + (m.month == 12), month=m.month % 12 + 1)
    return out


def _input_files(root: Path, store_id: str, months: list[datetime]) -> list[str]:
    from meteopack.core.store import points

    base = points.store_path(root, store_id) / points.OBSERVATIONS_DIR
    out = []
    for m in months:
        out += [str(p) for p in sorted(
            (base / f"year={m.year:04d}" / f"month={m.month:02d}").glob("*.parquet"))]
    return out


def _columns(cfg: RegulariseConfig) -> list[str]:
    cols = {"station_id", "time", *cfg.variables}
    if cfg.rain is not None:
        cols.add(cfg.rain.counter)
    for stage in cfg.stages:
        var = (stage.model_extra or {}).get("variable")
        if var:
            cols.add(var)
    return sorted(cols)


def do_month(args: tuple) -> dict:
    """One output month: read it plus the margin, regularise, write."""
    import resource

    import polars as pl

    cfg_json, root, month_iso, window = args
    cfg = RegulariseConfig.model_validate_json(cfg_json)
    started = time.time()
    lo = datetime.fromisoformat(month_iso)
    hi = lo.replace(year=lo.year + (lo.month == 12), month=lo.month % 12 + 1)
    if window:
        lo, hi = max(lo, datetime.fromisoformat(window[0])), min(hi, datetime.fromisoformat(window[1]))
    margin = timedelta(seconds=cfg.margin)
    files = _input_files(Path(root), cfg.input, _month_starts(lo - margin, hi + margin))
    report: dict[str, Any] = {}
    if not files:
        return {"month": month_iso, "rows": 0, "files": 0}
    a = (lo - margin).astimezone(UTC).replace(tzinfo=None)
    b = (hi + margin).astimezone(UTC).replace(tzinfo=None)
    scan = pl.scan_parquet(files, hive_partitioning=False).select(_columns(cfg))
    scan = scan.filter((pl.col("time") >= a) & (pl.col("time") < b))
    if cfg.stations is not None:
        scan = scan.filter(pl.col("station_id").is_in(cfg.stations))
    frame = scan.collect().sort("station_id", "time")
    t_read = time.time() - started
    out = regularise_frame(frame, cfg, lo, hi, report)
    t_compute = time.time() - started - t_read
    written = None
    if out.height:
        from meteopack.core.config import layer_path

        target = layer_path(Path(root), "training", *cfg.store_id.split("/"),
                            "observations", f"year={lo.year:04d}", f"month={lo.month:02d}",
                            "data.parquet")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
        out.write_parquet(tmp, compression="zstd", compression_level=cfg.compression_level,
                          statistics=True)
        os.replace(tmp, target)
        written = str(target)
    return {"month": month_iso, "rows": out.height, "input_rows": frame.height,
            "files": len(files), "path": written, "report": report,
            "read_s": t_read, "compute_s": t_compute,
            "write_s": time.time() - started - t_read - t_compute,
            "rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}


def apply_overrides(raw: dict, overrides: list[str]) -> dict:
    """`key.sub=value` pairs onto a config dict; values are YAML (`[mean, max]`, `600`)."""
    import copy

    import yaml

    out = copy.deepcopy(raw)
    for item in overrides or ():
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"--set wants key=value, got {item!r}")
        node = out
        parts = key.strip().split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
            if not isinstance(node, dict):
                raise TypeError(f"--set {key}: {p} is not a mapping")
        node[parts[-1]] = yaml.safe_load(value)
    return out


def build(cfg: RegulariseConfig, root: Path, *, window: tuple[datetime, datetime] | None = None,
          workers: int | None = None, threads: int = 2, refresh: bool = False) -> dict:
    """Regularise every month of the input (or of `window`) and write the store.

    Refuses to write into an existing output built with different parameters
    - a store whose months disagree about the grid is worse than none -
    unless `refresh`, which removes it first.
    """
    import concurrent.futures as cf
    import multiprocessing
    import shutil

    import polars as pl

    from meteopack.core.config import layer_path
    from meteopack.core.store import manifest, points

    started = time.time()
    store = layer_path(Path(root), "training", *cfg.store_id.split("/"))
    meta_path = store / "regularise.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text()).get("params_hash")
        if old != cfg.params_hash():
            if not refresh:
                raise SystemExit(
                    f"{store} was built with different parameters ({old} != "
                    f"{cfg.params_hash()}); pass --refresh to replace it, or --set name=... "
                    f"to write a new store")
            shutil.rmtree(store)
        elif refresh:
            shutil.rmtree(store)

    base = points.store_path(root, cfg.input) / points.OBSERVATIONS_DIR
    months_on_disk = sorted(
        datetime(int(p.parent.name.split("=")[1]), int(p.name.split("=")[1]), 1, tzinfo=UTC)
        for p in base.glob("year=*/month=*") if any(p.glob("*.parquet")))
    if not months_on_disk:
        raise SystemExit(f"no input partitions under {base}")
    if window is not None:
        months = [m for m in _month_starts(window[0], window[1])
                  if any(m == d for d in months_on_disk)]
    else:
        months = months_on_disk
    win = None if window is None else (window[0].isoformat(), window[1].isoformat())
    cfg_json = cfg.model_dump_json()
    tasks = [(cfg_json, str(root), m.isoformat(), win) for m in months]
    workers = workers or max(1, len(os.sched_getaffinity(0)) // max(1, threads))
    workers = min(workers, len(tasks)) or 1
    os.environ["POLARS_MAX_THREADS"] = str(threads)
    results = []
    with cf.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        results = list(ex.map(do_month, tasks))
    wall = time.time() - started

    rows = 0
    on_disk = store / points.OBSERVATIONS_DIR
    for f in on_disk.rglob("*.parquet") if on_disk.is_dir() else ():
        rows += pl.scan_parquet(f).select(pl.len()).collect().item()
    totals: dict[str, int] = {}
    for r in results:
        for k, v in (r.get("report") or {}).items():
            totals[k] = totals.get(k, 0) + int(v)
    in_store = manifest.read(points.store_path(root, cfg.input))
    meta = {
        "store_id": cfg.store_id, "input": cfg.input,
        "input_content_hash": in_store.content_hash if in_store else None,
        "params_hash": cfg.params_hash(), "params": json.loads(cfg.model_dump_json()),
        "margin_s": cfg.margin, "outputs": cfg.outputs(),
        "non_causal_columns": [k for k, v in cfg.outputs().items() if not v["causal"]],
        "label_convention": {"end": "bin T holds (T - step, T]",
                             "start": "bin T holds [T, T + step)",
                             "centre": "bin T holds [T - step/2, T + step/2)"}[cfg.label],
        "rows": rows, "months_built": len(months), "stage_counts": totals,
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "run": {"wall_s": wall, "workers": workers, "polars_threads": threads,
                "read_s": sum(r.get("read_s", 0) for r in results),
                "compute_s": sum(r.get("compute_s", 0) for r in results),
                "write_s": sum(r.get("write_s", 0) for r in results),
                "input_rows": sum(r.get("input_rows", 0) for r in results),
                "max_worker_rss_mib": max((r.get("rss_mib", 0) for r in results), default=0),
                "window": win},
    }
    store.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=2, default=str))
    manifest.write(store, manifest.Manifest.new(
        cfg.store_id, "training", "points", cfg, n_records=rows, inputs=[cfg.input],
        extra={"params_hash": cfg.params_hash(), "step_seconds": cfg.step_seconds,
               "label": cfg.label}))
    return meta


__all__ = ["METHODS", "STAGES", "RainSpec", "RegulariseConfig", "StageSpec", "VariableSpec",
           "apply_overrides", "build", "rain_increments", "regularise_frame"]
