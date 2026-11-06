"""Regularising an irregular points store: every choice a parameter, and exact.

The rules pinned here are the ones a downstream user cannot see by looking at
the output: which bin a report on a bin edge lands in, whether a value used
reports after T, and that splitting the work by month changes nothing.
"""
from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

import duckdb
import polars as pl
import pytest

from meteopack.derive import regularise as rg


def _cfg(**kw) -> rg.RegulariseConfig:
    base = {"name": "t", "input": "src/p/v1", "step_seconds": 300,
            "variables": {"x": {"methods": ["last", "mean", "max"]}}}
    base.update(kw)
    return rg.RegulariseConfig(**base)


def _frame(rows) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"station_id": pl.String, "time": pl.Datetime("us"),
                                      "x": pl.Float64}, orient="row").sort("station_id", "time")


T0 = datetime(2024, 1, 1)
REPORTS = _frame([("a", T0, 1.0), ("a", T0 + timedelta(minutes=4), 2.0),
                  ("a", T0 + timedelta(minutes=5), 3.0),
                  ("a", T0 + timedelta(minutes=7, seconds=30), 4.0)])
LO, HI = datetime(2023, 12, 31, tzinfo=UTC), datetime(2024, 1, 2, tzinfo=UTC)


def _run(cfg, frame=REPORTS):
    return rg.regularise_frame(frame, cfg, LO, HI, {}).sort("time")


def test_label_end_puts_a_report_on_the_edge_in_the_bin_it_ends():
    """(T - step, T], KNMI radar's convention: the 00:05 report is in bin 00:05."""
    out = _run(_cfg(label="end"))
    assert out["time"].to_list() == [T0, T0 + timedelta(minutes=5), T0 + timedelta(minutes=10)]
    assert out["x_last"].to_list() == [1.0, 3.0, 4.0]
    assert out["x_mean"].to_list() == [1.0, 2.5, 4.0]
    assert out["n_reports"].to_list() == [1, 2, 1]
    assert out["x_last_age_s"].to_list() == [0.0, 0.0, 150.0]


def test_label_start_and_centre():
    start = _run(_cfg(label="start"))
    assert start["x_last"].to_list() == [2.0, 4.0]                 # [00:00,05), [05,10)
    assert start["x_last_age_s"].to_list() == [-240.0, -150.0]     # after T: not causal
    centre = _run(_cfg(label="centre"))
    # [-2.5, 2.5) min, [2.5, 7.5), [7.5, 12.5): 07:30 is on an edge, in the later bin
    assert centre["x_last"].to_list() == [1.0, 3.0, 4.0]
    assert centre["n_reports"].to_list() == [1, 2, 1]


def test_origin_shifts_the_grid():
    out = _run(_cfg(origin=datetime(2000, 1, 1, 0, 2, tzinfo=UTC)))
    assert out["time"].to_list() == [T0 + timedelta(minutes=2), T0 + timedelta(minutes=7),
                                     T0 + timedelta(minutes=12)]


def test_nearest_respects_its_tolerance_and_says_how_far():
    cfg = _cfg(variables={"x": {"methods": ["nearest"], "tolerance_s": 60}})
    out = _run(cfg).filter(pl.col("x_nearest").is_not_null())
    assert out["time"].to_list() == [T0, T0 + timedelta(minutes=5)]
    assert out["x_nearest_dt_s"].to_list() == [0.0, 0.0]
    with pytest.raises(ValueError, match="half"):
        _cfg(variables={"x": {"methods": ["nearest"], "tolerance_s": 200}})


def test_linear_interpolates_only_across_short_gaps():
    cfg = _cfg(step_seconds=60, variables={"x": {"methods": ["linear"], "max_gap_s": 200}})
    out = _run(cfg).filter(pl.col("x_linear").is_not_null())
    vals = dict(zip(out["time"].to_list(), out["x_linear"].to_list(), strict=True))
    assert vals[T0 + timedelta(minutes=6)] == pytest.approx(3.4)     # 3 -> 4 over 150 s
    assert T0 + timedelta(minutes=2) not in vals                      # 0 -> 4 min: 240 s gap


def test_non_causal_outputs_are_marked():
    cfg = _cfg(variables={"x": {"methods": ["last", "nearest", "linear"], "tolerance_s": 60,
                                "max_gap_s": 600}}, rain={"rule": "robust"})
    out = cfg.outputs()
    assert out["x_last"]["causal"] and out["n_reports"]["causal"]
    assert not out["x_nearest"]["causal"] and not out["x_linear"]["causal"]
    assert not out["rainfall_amount"]["causal"]                       # robust looks ahead
    assert _cfg(rain={"rule": "naive"}).outputs()["rainfall_amount"]["causal"]
    assert not _cfg(label="centre").outputs()["x_last"]["causal"]


def test_overrides_change_one_value_and_the_params_hash():
    raw = {"name": "t", "input": "a/b/v1", "step_seconds": 600,
           "variables": {"x": {"methods": ["last"]}}}
    cfg = rg.RegulariseConfig(**rg.apply_overrides(
        raw, ["step_seconds=300", "variables.x.methods=[mean, max]", "rain.rule=naive"]))
    assert cfg.step_seconds == 300 and cfg.variables["x"].methods == ["mean", "max"]
    assert cfg.rain.rule == "naive"
    assert cfg.params_hash() != rg.RegulariseConfig(**raw).params_hash()
    renamed = rg.RegulariseConfig(**{**raw, "name": "other", "threads": 8})
    assert renamed.params_hash() == rg.RegulariseConfig(**raw).params_hash()


def test_a_range_mask_stage_nulls_and_counts():
    cfg = _cfg(stages=[{"kind": "range_mask", "variable": "x", "min": 1.5, "max": 3.5}])
    report: dict = {}
    out = rg.regularise_frame(REPORTS, cfg, LO, HI, report).sort("time")
    assert report == {"range_mask:x": 2}
    assert out["x_mean"].to_list() == [None, 2.5, None]
    assert out["n_reports"].to_list() == [1, 2, 1]             # reports still counted
    with pytest.raises(ValueError, match="unknown stage"):
        _cfg(stages=[{"kind": "homogenise"}])


# ---------------------------------------------------------------------- rain

_RAIN_SQL = """
CREATE OR REPLACE TEMP TABLE r AS
SELECT station_id AS SiteId, epoch_us(time) AS ts, rainfall_counter AS a,
  CAST(timezone('Europe/Amsterdam', time::TIMESTAMPTZ) AS DATE) AS lday,
  CAST(time AS DATE) AS uday
FROM src WHERE rainfall_counter BETWEEN 0 AND 1000;
CREATE OR REPLACE TEMP TABLE r1 AS
SELECT *, lag(a) OVER s AS ap, lead(a, 1) OVER s AS a1, lead(a, 2) OVER s AS a2,
  lead(a, 3) OVER s AS a3, lag(lday) OVER s AS lday_p, lag(uday) OVER s AS uday_p,
  lead(lday) OVER s AS lday_n, lead(uday) OVER s AS uday_n
FROM r WINDOW s AS (PARTITION BY SiteId ORDER BY ts);
CREATE OR REPLACE TEMP TABLE r1b AS
SELECT *, (a > ap + 0.25 AND a1 < a - 0.25 AND a1 >= ap - 0.25 AND a1 <= ap + 0.25
   AND lday_n = lday AND uday_n = uday) AS spike,
  CASE WHEN ap IS NULL THEN NULL WHEN a >= ap THEN a - ap WHEN a < ap - 0.25 THEN a ELSE 0 END
    AS inc_naive
FROM r1;
-- the fix: drops are judged against the last report that was not a spike
CREATE OR REPLACE TEMP TABLE r1c AS
SELECT *, last_value(CASE WHEN coalesce(spike, false) THEN NULL ELSE a END IGNORE NULLS)
  OVER (PARTITION BY SiteId ORDER BY ts ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS apr
FROM r1b;
CREATE OR REPLACE TEMP TABLE r2 AS
SELECT *, (a < apr - 0.25) AS drop_,
  (a < apr - 0.25 AND lday = lday_p AND uday = uday_p AND (a1 >= apr - 0.25 OR a2 >= apr - 0.25
   OR a3 >= apr - 0.25)) AS transient
FROM r1c;
CREATE OR REPLACE TEMP TABLE r3 AS
SELECT *, (drop_ AND NOT transient) AS reset,
  sum(CASE WHEN drop_ AND NOT transient THEN 1 ELSE 0 END) OVER (PARTITION BY SiteId ORDER BY ts
    ROWS UNBOUNDED PRECEDING) AS seg
FROM r2;
CREATE OR REPLACE TEMP TABLE r4 AS
SELECT *, max(CASE WHEN transient OR spike THEN NULL ELSE a END) OVER (PARTITION BY SiteId, seg
  ORDER BY ts ROWS UNBOUNDED PRECEDING) AS cm
FROM r3;
CREATE OR REPLACE TEMP TABLE r5 AS
SELECT *, CASE WHEN ap IS NULL THEN NULL WHEN reset THEN a
  ELSE coalesce(cm, 0) - coalesce(lag(cm) OVER (PARTITION BY SiteId, seg ORDER BY ts), cm, 0)
  END AS inc_robust
FROM r4;
"""


def _counter_series(seed: int) -> pl.DataFrame:
    """Counters with rain, local-midnight resets, drop-outs, spikes and gaps."""
    rnd = random.Random(seed)
    rows = []
    for s in range(4):
        t, a = datetime(2024, 3, 30, 20), 0.0             # spans the DST change
        for _ in range(400):
            t += timedelta(seconds=rnd.choice([60, 300, 300, 600, 3600]))
            r = rnd.random()
            if r < 0.3:
                a += rnd.choice([0.2, 0.25, 0.3, 1.0])
            if r > 0.97:
                value = rnd.choice([0.0, a / 3, a + 9.0, -5.0, 1500.0])   # glitches
            elif r > 0.95:
                a = 0.0
                value = a                                                 # a reset
            else:
                value = a
            rows.append((str(s), t, None if r > 0.99 else round(value, 2)))
    return pl.DataFrame(rows, schema={"station_id": pl.String, "time": pl.Datetime("us"),
                                      "rainfall_counter": pl.Float64}, orient="row")


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_rain_rules_are_rain_counter_py_exactly(seed):
    """The polars port against the analysis' own SQL (with the spike fix), increment by increment."""
    frame = _counter_series(seed).sort("station_id", "time")
    con = duckdb.connect()
    con.execute("SET TimeZone='UTC'")
    con.execute("CREATE TABLE src (station_id VARCHAR, time TIMESTAMP, rainfall_counter DOUBLE)")
    con.executemany("INSERT INTO src VALUES (?, ?, ?)", frame.rows())
    con.execute(_RAIN_SQL)
    want = con.execute("SELECT SiteId, ts, inc_naive, inc_robust FROM r5 ORDER BY SiteId, ts"
                       ).fetchall()
    for rule, col in (("naive", 2), ("robust", 3)):
        got, _ = rg.rain_increments(frame, rg.RainSpec(rule=rule))
        got = got.sort("station_id", "time")
        assert got.height == len(want)
        mine = got["inc"].to_list()
        theirs = [w[col] for w in want]
        diff = [(i, m, t) for i, (m, t) in enumerate(zip(mine, theirs, strict=True))
                if (m is None) != (t is None) or (m is not None and abs(m - t) > 1e-9)]
        assert not diff, f"{rule}: {len(diff)} increments differ, e.g. {diff[:5]}"


def test_the_robust_rule_ignores_a_transient_drop_out():
    t = [datetime(2024, 6, 1, 10) + timedelta(minutes=5 * i) for i in range(6)]
    frame = pl.DataFrame({"station_id": ["s"] * 6, "time": t,
                          "rainfall_counter": [1.0, 2.0, 6.35, 1.02, 6.35, 7.0]})
    robust, _ = rg.rain_increments(frame, rg.RainSpec(rule="robust"))
    naive, _ = rg.rain_increments(frame, rg.RainSpec(rule="naive"))
    assert robust["inc"].sum() == pytest.approx(6.0)        # 1 + 4.35 + 0 + 0 + 0.65
    assert naive["inc"].sum() == pytest.approx(12.35)       # the drop-out counted as a reset


def test_a_counter_spike_is_not_followed_by_a_false_reset():
    """After a one-report up-jump the counter falls back; judged against the
    spike that fall read as a reset and the whole counter was counted again
    (found by the QC run on the full archive, 2026-11-06)."""
    t = [datetime(2024, 6, 1, 10) + timedelta(minutes=5 * i) for i in range(6)]
    frame = pl.DataFrame({"station_id": ["s"] * 6, "time": t,
                          "rainfall_counter": [1.0, 2.0, 11.0, 2.2, 2.4, 3.0]})
    robust, _ = rg.rain_increments(frame, rg.RainSpec(rule="robust"))
    assert robust["inc"].sum() == pytest.approx(2.0)        # 1 + 0 + 0.2 + 0.2 + 0.6
    assert not robust["reset"].any()


# ------------------------------------------------- months change nothing


def test_work_split_by_month_equals_one_pass(tmp_path):
    """Bins on a month edge, nearest/linear across it and the rain look-back must
    come out the same as if the archive were one frame."""
    from meteopack.core.config import layer_path

    rnd = random.Random(7)
    rows = []
    for s in "abc":
        t = datetime(2024, 1, 31, 20)
        a = 0.0
        while t < datetime(2024, 2, 1, 4):
            t += timedelta(seconds=rnd.randint(20, 900))
            a += rnd.choice([0, 0, 0.2])
            rows.append((s, t, rnd.uniform(-5, 5), a))
    frame = pl.DataFrame(rows, schema={"station_id": pl.String, "time": pl.Datetime("us"),
                                       "x": pl.Float64, "rainfall_counter": pl.Float64},
                         orient="row").sort("station_id", "time")
    base = layer_path(tmp_path, "packed", "src", "p", "v1", "observations")
    for (y, m), g in frame.group_by(pl.col("time").dt.year().alias("y"),
                                    pl.col("time").dt.month().alias("m")):
        d = base / f"year={y:04d}" / f"month={m:02d}"
        d.mkdir(parents=True)
        g.write_parquet(d / "data.parquet")
    cfg = _cfg(step_seconds=600, variables={"x": {
        "methods": ["last", "mean", "nearest", "linear"], "tolerance_s": 300,
        "max_gap_s": 1800}}, rain={"rule": "robust"}, threads=1)
    rg.build(cfg, tmp_path, workers=2)
    split = pl.read_parquet(layer_path(tmp_path, "training", "t", "v1", "observations")
                            / "**" / "*.parquet", hive_partitioning=False)
    whole = rg.regularise_frame(frame, cfg, datetime(2024, 1, 1, tzinfo=UTC),
                                datetime(2024, 3, 1, tzinfo=UTC), {})
    key = ["station_id", "time"]
    assert split.sort(key).equals(whole.sort(key)), "splitting by month changed the result"
    assert split.filter(pl.col("time") == datetime(2024, 2, 1)).height == 3
    meta = (layer_path(tmp_path, "training", "t", "v1") / "regularise.json").read_text()
    assert '"params_hash"' in meta and '"non_causal_columns"' in meta
    with pytest.raises(SystemExit, match="different parameters"):
        rg.build(_cfg(step_seconds=300, threads=1), tmp_path, workers=1)
