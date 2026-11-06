"""WOW-NL quality-control flags: each test fires where an error was put, and nowhere else.

A synthetic level-1 store (a dozen stations a few km apart, 5-min reports,
smooth weather) gets errors injected at known reports; the QC derive must set
exactly the expected bits. The PWSQC rain tests' three states are checked on
hand-made matrices, line for line against PWSQC's R.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from meteopack.derive import qc
from meteopack.derive import qc_checks as ck

T0 = datetime(2024, 6, 1)
STEP = 300
DAYS = 3
N = DAYS * 86400 // STEP
GRID = [(52.0 + 0.02 * (i // 4), 5.0 + 0.03 * (i % 4)) for i in range(12)]
ALL_VARS = ["air_temperature", "dew_point_temperature", "relative_humidity",
            "air_pressure_at_sea_level", "wind_speed", "rainfall_counter"]


def _bit(table, name):
    return 1 << table[name]


def _clean(i: int) -> dict[str, np.ndarray]:
    """Smooth, plausible weather for station i: nothing should fire on it."""
    k = np.arange(N)
    t = k * STEP
    day = 2 * math.pi * t / 86400
    ta = 15 + 2 * np.sin(day) + 0.02 * (i - 6)
    rh = 70 + 10 * np.cos(day) + 0.1 * i
    td = ck.dew_point_magnus(ta, rh)
    p = 1013 + 0.5 * np.sin(day / 3) + 0.03 * i
    ws = 3 + np.sin(day + 0.01 * i)
    # rain: +0.2 mm per 5 min 08:00-10:00 UTC on day 2, counter reset at local midnight
    counter = np.zeros(N)
    rain_on = (t >= 86400 + 8 * 3600) & (t < 86400 + 10 * 3600)
    acc = 0.0
    for j in range(N):
        if (t[j] + 2 * 3600) % 86400 == 0:          # local midnight (CEST)
            acc = 0.0
        if rain_on[j]:
            acc += 0.2
        counter[j] = round(acc, 3)
    return {"air_temperature": ta, "dew_point_temperature": td, "relative_humidity": rh,
            "air_pressure_at_sea_level": p, "wind_speed": ws, "rainfall_counter": counter}


def _store(root, stations: dict[str, dict], positions: dict[str, tuple], extra_rows=(),
           local_utc=(), capped_hours=(), moved=None):
    """Write a level-1-like store: observations, station_segments, hours."""
    rows = []
    for sid, vals in stations.items():
        lat, lon = positions[sid]
        for k in range(N):
            t = T0 + timedelta(seconds=k * STEP)
            r = {"station_id": sid, "time": t, "Latitude": lat, "Longitude": lon,
                 "IsOfficial": False, "source_member": f"m{t:%Y%m%d%H}"}
            if moved and moved[0] == sid and moved[1] == k:
                r["Latitude"] = lat + 0.05
            local = t if sid in local_utc else t + timedelta(hours=2)
            r["LocalReportEndDateTime"] = f"{local:%Y-%m-%dT%H:%M:%S}+00:00"
            for v, arr in vals.items():
                r[v] = None if arr is None or np.isnan(arr[k]) else float(arr[k])
            if all(r[v] is None for v in vals):
                continue                      # no report at all (a gap)
            rows.append(r)
    rows += list(extra_rows)
    schema = {"station_id": pl.String, "time": pl.Datetime("us"), "Latitude": pl.Float64,
              "Longitude": pl.Float64, "IsOfficial": pl.Boolean, "source_member": pl.String,
              "LocalReportEndDateTime": pl.String, **{v: pl.Float64 for v in ALL_VARS}}
    df = pl.DataFrame(rows, schema=schema).sort("station_id", "time")
    base = root / "packed" / "src" / "p" / "v1"
    d = base / "observations" / "year=2024" / "month=06"
    d.mkdir(parents=True)
    df.write_parquet(d / "data.parquet")
    seg = (df.group_by("station_id", "Latitude", "Longitude")
           .agg(pl.col("time").min().alias("first_time"), pl.col("time").max().alias("last_time"),
                pl.len().alias("n_reports"))
           .with_columns(pl.lit(0.0).alias("Height"), pl.lit(0).alias("n_official")))
    d = base / "station_segments" / "year=2024" / "month=06"
    d.mkdir(parents=True)
    seg.write_parquet(d / "data.parquet")
    hours = pl.DataFrame({"member": [f"m{T0 + timedelta(hours=h):%Y%m%d%H}"
                                     for h in range(DAYS * 24)]})
    hours = hours.with_columns(pl.col("member").is_in(list(capped_hours)).alias("at_page_cap"))
    d = base / "hours" / "year=2024" / "month=06"
    d.mkdir(parents=True)
    hours.write_parquet(d / "data.parquet")
    return df


def _cfg(variables: dict, **kw) -> qc.QcConfig:
    base = {"name": "src/qc", "input": "src/p/v1", "variables": variables, "rain": None,
            "dew_point": None, "threads": 1}
    base.update(kw)
    return qc.QcConfig(**base)


def _run(tmp_path, cfg):
    meta = qc.build(cfg, tmp_path, workers=1, threads=1)
    out = pl.read_parquet(tmp_path / "packed" / "src" / "qc" / "v1" / "observations"
                          / "year=2024" / "month=06" / "data.parquet")
    l1 = pl.read_parquet(tmp_path / "packed" / "src" / "p" / "v1" / "observations"
                         / "year=2024" / "month=06" / "data.parquet")
    return meta, out, l1


# ------------------------------------------------------------ within station


WITHIN = {
    "air_temperature": {"range": [-28, 42], "extreme": [-50, 50], "sentinels": [-17.7778],
                        "spike": {"threshold": 3.0}, "step": {"threshold": 5.0},
                        "persistence": {"min_duration_s": 14400}},
    "dew_point_temperature": {"range": [-35, 28]},
    "relative_humidity": {"range": [5, 100], "sentinels": [0.0]},
    "air_pressure_at_sea_level": {"range": [940, 1065]},
    "wind_speed": {"range": [0, 35], "persistence": {
        "min_duration_s": 9600, "zero_min_duration_s": 172800, "zero_is_special": True},
        "censor_zero": True},
    "rainfall_counter": {"range": [0, 1000]},
}


def test_within_station_and_report_bits_fire_exactly_where_injected(tmp_path):
    st = {f"g{i:02d}": _clean(i) for i in range(12)}
    pos = {f"g{i:02d}": GRID[i] for i in range(12)}
    st["dup"] = _clean(3)
    pos["dup"] = GRID[0]                                          # same coordinates as g00
    k_noon = 86400 // STEP + 12 * 12                              # day 2, 12:00 UTC
    st["g01"]["air_temperature"][k_noon] = 60.0                   # extreme + range
    st["g02"]["air_temperature"][k_noon] = -17.7778               # sentinel (0 degF)
    st["g03"]["air_temperature"][k_noon] += 6.0                   # spike
    st["g04"]["air_temperature"][k_noon:k_noon + 12] += 8.0       # step up, an hour later down
    k_min = 86400 // STEP + 18 * 12                               # the diurnal minimum
    run = slice(k_min - 30, k_min + 31)                           # 5 h of one value
    st["g05"]["air_temperature"][run] = 13.0
    st["g11"]["wind_speed"][:] = 0.0                              # calm for 3 days
    for s in ("g01", "g02", "g03", "g04", "g05"):
        st[s]["dew_point_temperature"][:] = np.nan                # keep Td out of it
    st["g06"]["dew_point_temperature"][k_noon] = st["g06"]["air_temperature"][k_noon] + 2
    st["g07"]["relative_humidity"][k_noon] = 0.0                  # sentinel + range
    # g08: a 90-min gap, then the record before the gap repeated exactly
    gap = range(k_noon + 1, k_noon + 19)
    for v in st["g08"]:
        st["g08"][v][list(gap)] = np.nan
        st["g08"][v][k_noon + 19] = st["g08"][v][k_noon]
    dup_row = {"station_id": "g09", "time": T0 + timedelta(seconds=k_noon * STEP + 10),
               "Latitude": GRID[9][0], "Longitude": GRID[9][1], "IsOfficial": False,
               "source_member": "x", "LocalReportEndDateTime": None,
               **{v: float(st["g09"][v][k_noon]) for v in ALL_VARS}}
    df = _store(tmp_path, st, pos, extra_rows=[dup_row], local_utc={"g10"},
                capped_hours={"m2024060203"}, moved=("g10", k_noon))
    # drop the gap rows of g08 (they are not reports at all)
    cfg = _cfg(WITHIN, dew_point={"gt_t_tolerance": 0.5, "mismatch_tolerance": 2.0},
               stale_repeat={"gap_s": 3600, "fields": ALL_VARS},
               network={"reference_excludes": ["range"]})
    _, out, l1 = _run(tmp_path, cfg)

    assert out.height == df.height == l1.height
    assert out["l1_row"].to_list() == list(range(out.height))
    assert (out["station_id"] == l1["station_id"]).all() and (out["time"] == l1["time"]).all()

    F = qc.FAIL_BITS

    def at(sid, k):            # store row of station sid's k-th report (g09 has one extra)
        rows = out.with_row_index("i").filter(pl.col("station_id") == sid)["i"].to_list()
        if sid == "g09" and k > k_noon:
            k += 1
        return rows[k]

    expected = {                                       # (station, variable) -> {bit: rows}
        ("g01", "air_temperature"): {"range": [at("g01", k_noon)],
                                     "extreme": [at("g01", k_noon)]},
        ("g02", "air_temperature"): {"sentinel": [at("g02", k_noon)]},
        ("g03", "air_temperature"): {"spike": [at("g03", k_noon)],
                                     "step": [at("g03", k_noon), at("g03", k_noon + 1)]},
        ("g04", "air_temperature"): {"step": [at("g04", k_noon), at("g04", k_noon + 12)]},
        ("g05", "air_temperature"): {"stuck": [at("g05", k) for k in range(k_min - 30,
                                                                             k_min + 31)]},
        ("g06", "dew_point_temperature"): {"td_gt_t": [at("g06", k_noon)],
                                           "td_mismatch": [at("g06", k_noon)]},
        ("g07", "relative_humidity"): {"sentinel": [at("g07", k_noon)],
                                       "range": [at("g07", k_noon)]},
        ("g11", "wind_speed"): {"stuck_zero": [at("g11", k) for k in range(N)]},
    }
    stale_row = out.with_row_index("i").filter(
        (pl.col("station_id") == "g08")
        & (pl.col("time") == T0 + timedelta(seconds=(k_noon + 19) * STEP)))["i"].item()
    for v in ALL_VARS:
        expected.setdefault(("g08", v), {})["stale_repeat"] = [stale_row]
    for sid in out["station_id"].unique():
        for v in ALL_VARS:
            col = out[f"{v}_qc_fail"].to_numpy()
            idx = np.flatnonzero(out["station_id"].to_numpy() == sid)
            want = expected.get((sid, v), {})
            for b, k in F.items():
                got = idx[(col[idx] & (1 << k)) != 0].tolist()
                assert got == sorted(want.get(b, [])), (sid, v, b, got[:5])
    # censored zeros: every wind zero, and only those
    ne = out["wind_speed_qc_ne"].to_numpy()
    cens = (ne & _bit(qc.NE_BITS, "censored_low")) != 0
    assert np.array_equal(cens, l1["wind_speed"].to_numpy() == 0)
    # report bits
    R = qc.REPORT_BITS
    rep = out["qc_report"].to_numpy()
    sid = out["station_id"].to_numpy()
    near = np.flatnonzero(rep & _bit(R, "near_duplicate"))
    assert near.tolist() == [at("g09", k_noon) + 1]
    assert np.flatnonzero(rep & _bit(R, "position_jump")).tolist() == [at("g10", k_noon)]
    assert set(sid[(rep & _bit(R, "dup_coords")) != 0]) == {"g00", "dup"}
    assert set(sid[(rep & _bit(R, "local_time_inconsistent")) != 0]) == {"g10"}
    capped = (rep & _bit(R, "capped_hour")) != 0
    assert np.array_equal(capped, l1["source_member"].to_numpy() == "m2024060203")
    # levels: the extreme is bad, a clean value is good, a missing one is 0
    lvl = out["air_temperature_qc_level"].to_numpy()
    assert lvl[at("g01", k_noon)] == 1
    assert lvl[at("g07", 5)] == 4
    assert lvl[at("g00", 5)] == 2                               # dup_coords: suspect
    assert out["dew_point_temperature_qc_level"].to_numpy()[at("g01", 5)] == 0


# ---------------------------------------------------------------- network


NETWORK_T = {
    "air_temperature": {
        "range": [-28, 42],
        "network_z": {"low": -2.326, "high": 1.645, "min_stations": 5, "min_scale": 1.0},
        "month_fraction": {"max_fraction": 0.2, "min_values": 100},
        "month_corr": {"min_r": 0.9, "min_pairs": 100},
        "buddy": {"radius_m": 10000, "min_buddies": 5, "max_buddies": 20, "pos": 2.0,
                  "neg": 2.83, "min_scale": 1.0},
        "isolation": {"radius_m": 15000, "min_neighbours": 5}},
}


def _network_store(tmp_path):
    st = {f"g{i:02d}": _clean(i) for i in range(12)}
    pos = {f"g{i:02d}": GRID[i] for i in range(12)}
    k12 = 86400 // STEP + 12 * 12
    st["g10"]["air_temperature"][k12:k12 + 24] += 10.0             # 2 h far too warm
    st["indoor"] = _clean(5)
    st["indoor"]["air_temperature"] = 21 + 0.01 * np.sin(np.arange(N) / 7.0)
    pos["indoor"] = (52.01, 5.04)
    st["iso"] = _clean(2)
    pos["iso"] = (53.2, 6.6)                                       # > 100 km away
    _store(tmp_path, st, pos)
    return k12


def test_network_tests_flag_the_outliers_and_mark_the_isolated_not_evaluable(tmp_path):
    k12 = _network_store(tmp_path)
    _, out, _ = _run(tmp_path, _cfg(NETWORK_T))
    F, E = qc.FAIL_BITS, qc.NE_BITS
    f = out["air_temperature_qc_fail"].to_numpy()
    e = out["air_temperature_qc_ne"].to_numpy()
    sid = out["station_id"].to_numpy()
    idx10 = np.flatnonzero(sid == "g10")
    warm = idx10[k12:k12 + 24]
    for b in ("network_z", "buddy"):
        got = idx10[(f[idx10] & _bit(F, b)) != 0]
        assert got.tolist() == warm.tolist(), b
    ind = sid == "indoor"
    assert ((f[ind] & _bit(F, "month_corr")) != 0).all()          # m4: an indoor sensor
    assert ((f[ind] & _bit(F, "month_fraction")) != 0).all()      # m3: most of it fails m2
    iso = sid == "iso"
    assert ((e[iso] & _bit(E, "isolated")) != 0).all()
    assert ((e[iso] & _bit(E, "buddy")) != 0).all()
    assert (f[iso] == 0).all()                                    # isolated is never failed
    for b in ("network_z", "month_corr", "month_fraction"):       # ... nor judged network-wide
        assert ((e[iso] & _bit(E, b)) != 0).all(), b
    clean = ~np.isin(sid, ["g10", "indoor", "iso"])
    assert (f[clean] == 0).all(), np.unique(f[clean])
    assert (e[clean] & (_bit(E, "isolated") | _bit(E, "buddy")) == 0).all()
    lvl = out["air_temperature_qc_level"].to_numpy()
    assert set(lvl[warm]) == {2} and set(lvl[iso]) == {3} and set(lvl[clean]) == {4}


def test_sct_flags_a_warm_station_at_its_snapshots(tmp_path):
    pytest.importorskip("titanlib")
    k12 = _network_store(tmp_path)
    spec = {"air_temperature": {"sct": {"step_seconds": 3600, "num_min": 5, "num_max": 30,
                                        "inner_radius_m": 20000, "outer_radius_m": 30000}}}
    _, out, _ = _run(tmp_path, _cfg(spec))
    f = out["air_temperature_qc_fail"].to_numpy()
    e = out["air_temperature_qc_ne"].to_numpy()
    sid = out["station_id"].to_numpy()
    bit = _bit(qc.FAIL_BITS, "sct")
    idx10 = np.flatnonzero(sid == "g10")
    # snapshots 12:00 and 13:00 see the warm values; each judges reports within 30 min
    flagged = idx10[(f[idx10] & bit) != 0]
    assert flagged.tolist() == idx10[k12 - 6:k12 + 18].tolist()
    assert ((e[sid == "iso"] & _bit(qc.NE_BITS, "sct")) != 0).all()
    assert (f[sid == "iso"] & bit == 0).all()


# ------------------------------------------------------------------- rain


def test_rain_counter_and_pwsqc_flags(tmp_path):
    st = {f"g{i:02d}": _clean(i) for i in range(12)}
    pos = {f"g{i:02d}": GRID[i] for i in range(12)}
    k8 = 86400 // STEP + 8 * 12                                    # rain starts 08:00 day 2
    st["g01"]["rainfall_counter"][:] = 0.0                         # faulty zeros
    k_mid = 86400 // STEP + 22 * 12                                # local midnight: reset
    st["g02"]["rainfall_counter"][k8 + 5:k_mid] += 15.0            # one 5-min with 15 mm
    st["g03"]["rainfall_counter"][k8 + 5:k_mid] += 25.0            # one 5-min with 25 mm
    k15 = 86400 // STEP + 15 * 12
    st["g04"]["rainfall_counter"][k15] = 0.0                       # transient drop-out
    st["g05"]["rainfall_counter"][k15] = 12.0                      # one-report spike
    k18 = 86400 // STEP + 18 * 12
    reset = (np.arange(N) >= k18) & (np.arange(N) < 86400 // STEP + 22 * 12)
    st["g06"]["rainfall_counter"][reset] = 0.0                     # reset at 18 UTC, not midnight
    st["iso"] = _clean(2)
    pos["iso"] = (53.2, 6.6)
    _store(tmp_path, st, pos)
    rain = {"counter": "rainfall_counter", "pwsqc": {"so": None}}
    _, out, _ = _run(tmp_path, _cfg({"rainfall_counter": {"range": [0, 1000]}}, rain=rain))
    F, E = qc.FAIL_BITS, qc.NE_BITS
    f = out["rainfall_counter_qc_fail"].to_numpy()
    e = out["rainfall_counter_qc_ne"].to_numpy()
    sid = out["station_id"].to_numpy()

    def rows(s, b, kind=f, table=F):
        i = np.flatnonzero(sid == s)
        return (i[(kind[i] & _bit(table, b)) != 0] - i[0]).tolist()

    # FZ: dry from 08:00 while the neighbours are wet: flagged from the 7th wet interval
    # (nint = 6) and then until the station reports rain - which it never does
    assert rows("g01", "rain_faulty_zero") == list(range(k8 + 6, N))
    assert rows("g02", "rain_high_influx") == [k8 + 5]
    assert rows("g03", "rain_high_influx") == [k8 + 5]
    assert rows("g03", "rain_huge_increment") == [k8 + 5]
    assert rows("g04", "rain_transient") == [k15]
    assert rows("g05", "rain_spike") == [k15]
    # the robust rule (rain_counter.py) takes the fall back after a spike for a reset - and
    # counts the whole counter again - so that report is flagged too
    assert rows("g05", "rain_reset_off_midnight") == [k15 + 1]
    assert rows("g06", "rain_reset_off_midnight") == [k18]
    for s in ["g00", "g07", "g08", "g09", "g10", "g11"]:
        i = sid == s
        assert (f[i] == 0).all(), (s, np.unique(f[i]))
    assert ((e[sid == "iso"] & _bit(E, "rain_faulty_zero")) != 0).all()
    assert (f[sid == "iso"] == 0).all()
    inc = out["rainfall_increment"].to_numpy()
    assert np.nansum(inc[sid == "g00"]) == pytest.approx(24 * 0.2)


# ------------------------------------------------------- PWSQC three states


def test_pwsqc_high_influx_three_states():
    r = np.array([[0.1, 11.0, 5.0, 26.0]])
    med = np.array([[0.1, 0.2, 1.0, 1.0]])
    cnt = np.array([[5, 5, 5, 4]])
    got = ck.pwsqc_high_influx(r, med, cnt, nstat=5, thr_a=0.4, thr_b=10)
    # 11 > 10 with a dry median; 5 < 25 x median; 26 > 25 but only 4 neighbours -> -1
    assert got.tolist() == [[0, 1, 0, -1]]


def test_pwsqc_faulty_zeros_like_the_r_code():
    nint, nstat = 6, 5
    x = np.zeros(30)
    x[25] = 0.4                                                   # rain at 25 ends the run
    x[20] = np.nan                                                # missing: breaks the dry run
    med = np.zeros(30)
    med[2:12] = 0.3                                               # neighbours wet 2..11
    cnt = np.full(30, 8)
    cnt[28] = 3
    got = ck.pwsqc_faulty_zeros(x[None, :], med[None, :], cnt[None, :], nstat, nint)[0]
    want = np.zeros(30, np.int8)
    want[2 + nint:20] = 1          # from nint into the wet run to the end of the dry run (19) ...
    want[20:25] = 1                # ... then on through NaN and zeros until the station is wet
    want[28] = -1                  # too few neighbours there
    assert got.tolist() == want.tolist()
    short = ck.pwsqc_faulty_zeros(x[None, :6], med[None, :6], cnt[None, :6], nstat, nint)
    assert (short == 0).all()                                     # a dry run of 6 is not > 6


def test_pwsqc_station_outlier_flags_an_uncorrelated_gauge():
    rng = np.random.default_rng(1)
    per, hours = 12, 24 * 20
    n_j = per * hours
    wet = rng.random(n_j) < 0.15
    field = np.where(wet, rng.gamma(2, 0.3, n_j), 0.0)
    r = np.vstack([field * rng.uniform(0.8, 1.2, n_j) for _ in range(7)])
    r[0] = np.where(rng.random(n_j) < 0.15, rng.gamma(2, 0.3, n_j), 0.0)   # unrelated
    nb = np.array([[j for j in range(7) if j != i] for i in range(7)])
    so = ck.pwsqc_station_outlier(r, nb, nstat=5, per=per, compare=14 * 24, rainy=100,
                                  match=200, corthres=0.15, first_eval=0)
    last = so[:, -1]
    assert last[0] == 1 and (last[1:] == 0).all()
    assert (so[:, :per * (14 * 24 - 1)] == -1).all()              # no full window yet
    few = ck.pwsqc_station_outlier(r[:4], nb[:4, :3], nstat=5, per=per, compare=14 * 24,
                                   rainy=100, match=200, corthres=0.15, first_eval=0)
    assert (few == -1).all()                                      # fewer than nstat neighbours


def test_spreading_puts_increments_where_they_fell():
    # a 10-min report of 1 mm covers two 5-min bins equally; a 30-min one is not spread
    s = np.array([0, 0])
    t_end = np.array([600.0, 2400.0])
    dt = np.array([600.0, 1800.0])
    inc = np.array([1.0, 3.0])
    r, rep, jj = ck.spread_increments(s, t_end, dt, inc, 0.0, 300.0, 1, 8, 1200, 0.5)
    assert r[0, :2].tolist() == [0.5, 0.5]
    assert np.isnan(r[0, 2:]).all()
    assert rep.tolist() == [0, 0] and jj.tolist() == [0, 1]


# ---------------------------------------------------------------- the rest


def test_params_hash_and_refusing_to_mix(tmp_path):
    a = _cfg(WITHIN)
    b = _cfg(WITHIN, threads=4, name="other/qc", description="x")
    c = _cfg({**WITHIN, "air_temperature": {**WITHIN["air_temperature"],
                                            "spike": {"threshold": 2.5}}})
    assert a.params_hash() == b.params_hash()
    assert a.params_hash() != c.params_hash()
    _store(tmp_path, {"g00": _clean(0)}, {"g00": GRID[0]})
    meta = qc.build(a, tmp_path, workers=1, threads=1)
    saved = json.loads((tmp_path / "packed" / "src" / "qc" / "v1" / "qc.json").read_text())
    assert saved["params_hash"] == meta["params_hash"] == a.params_hash()
    assert saved["bits"]["fail"]["buddy"] == 13
    with pytest.raises(SystemExit, match="different parameters"):
        qc.build(c, tmp_path, workers=1, threads=1)
    qc.build(c, tmp_path, workers=1, threads=1, refresh=True)
    out = pl.read_parquet(tmp_path / "packed" / "src" / "qc" / "v1" / "observations"
                          / "**" / "*.parquet")
    assert out["qc_config_hash"].cast(pl.String).unique().to_list() == [c.params_hash()]


def test_unknown_bits_and_bad_combinations_are_refused():
    with pytest.raises(ValueError, match="unknown fail bit"):
        _cfg(WITHIN, levels=[{"level": 1, "name": "x", "fail": ["nonsense"]}])
    with pytest.raises(ValueError, match="needs network_z"):
        _cfg({"air_temperature": {"month_fraction": {}}})


def test_persistence_across_a_month_edge_is_judged_on_the_whole_run():
    sid = np.zeros(10, int)
    t = np.arange(10) * 3600.0
    x = np.array([1, 2, 5, 5, 5, 5, 5, 5, 3, 4], float)
    stuck, _ = ck.persistence(x, t, sid, 5 * 3600, None, False)
    assert stuck.tolist() == [False] * 2 + [True] * 6 + [False] * 2
    stuck, zero = ck.persistence(np.r_[0, 0, 0, 1.0], np.arange(4) * 3600.0, np.zeros(4, int),
                                 3600, 7200, True)
    assert zero.tolist() == [True, True, True, False] and not stuck.any()


def test_reading_level1_with_flags_side_by_side(tmp_path):
    # WOW has reports without a station id: they are flagged and kept like any other
    orphan = {"station_id": None, "time": T0 + timedelta(minutes=7), "Latitude": 52.5,
              "Longitude": 5.5, "IsOfficial": False, "source_member": "x",
              "LocalReportEndDateTime": None, **{v: 1.0 for v in ALL_VARS}}
    _store(tmp_path, {"g00": _clean(0), "g01": _clean(1)}, {"g00": GRID[0], "g01": GRID[1]},
           extra_rows=[orphan])
    qc.build(_cfg(NETWORK_T), tmp_path, workers=1, threads=1)
    both = qc.read_month(tmp_path, 2024, 6, qc_store="src/qc/v1", input_store="src/p/v1")
    assert both.height == 2 * N + 1
    assert both.filter(pl.col("station_id").is_null())["air_temperature_qc_level"].item() > 0
    assert {"air_temperature", "air_temperature_qc_level", "qc_report"} <= set(both.columns)
    rel = pl.read_parquet(tmp_path / "packed" / "src" / "qc" / "v1" / "reliability.parquet")
    t = rel.filter((pl.col("variable") == "air_temperature") & (pl.col("station_id") != ""))
    assert t.height == 2 and (t["frac_ok"] == 1.0).all() and (t["trust_tier"] == 2).all()

