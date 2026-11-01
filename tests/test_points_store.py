"""Station data: partitioned on disk, dense on the way out, never duplicated."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from meteopack.core.config import ProductConfig, StoreConfig
from meteopack.core.store import points
from meteopack.core.timeline import Timeline, utc

CADENCE = timedelta(minutes=10)


def make_cfg(partition_by: list[str] | None = None) -> ProductConfig:
    return ProductConfig(
        source="dwd",
        product="10min",
        cadence_seconds=600,
        store=StoreConfig(kind="points", partition_by=partition_by or ["year"]),
    )


def obs(stations: list[str], start: datetime, n: int, offset: float = 0.0) -> pl.DataFrame:
    """A tidy (time, station_id, t2m) block at the 10-minute cadence."""
    times = [start + i * CADENCE for i in range(n)]
    return pl.DataFrame(
        {
            "time": [t for _ in stations for t in times],
            "station_id": [s for s in stations for _ in times],
            "t2m": [offset + float(i) for s_i, _ in enumerate(stations) for i in range(n)],
        }
    )


def test_write_lays_out_hive_dirs_and_keeps_them_out_of_the_body(tmp_path: Path):
    cfg = make_cfg(["year", "month"])
    frame = pl.concat(
        [obs(["00044"], datetime(2012, 1, 1), 3), obs(["00044"], datetime(2012, 2, 1), 2)]
    )
    written = points.write(frame, cfg, tmp_path)
    rel = sorted(str(p.relative_to(tmp_path)) for p in written)
    assert rel == [
        "packed/dwd/10min/v1/observations/year=2012/month=01/data.parquet",
        "packed/dwd/10min/v1/observations/year=2012/month=02/data.parquet",
    ]
    out = points.read(cfg.store_id, tmp_path)
    assert out.columns == ["time", "station_id", "t2m"]  # no year/month leaked in
    assert len(out) == 5


def test_round_trip_normalises_time_to_naive_utc(tmp_path: Path):
    cfg = make_cfg()
    frame = pl.DataFrame(
        {
            "time": [datetime(2012, 1, 1, 12, tzinfo=UTC)],
            "station_id": ["00044"],
            "t2m": [3.5],
        }
    ).with_columns(pl.col("time").dt.convert_time_zone("Europe/Amsterdam"))
    points.write(frame, cfg, tmp_path)
    out = points.read(cfg.store_id, tmp_path)
    assert out["time"].dtype == pl.Datetime("us")
    assert out["time"][0] == datetime(2012, 1, 1, 12)


def test_rewriting_a_partition_replaces_it_and_never_duplicates(tmp_path: Path):
    cfg = make_cfg()
    first = obs(["00044", "01048"], datetime(2012, 1, 1), 6)
    points.write(first, cfg, tmp_path)
    assert len(points.read(cfg.store_id, tmp_path)) == 12

    # same partition, corrected values: replaced, not appended
    points.write(first.with_columns(pl.col("t2m") + 100), cfg, tmp_path)
    out = points.read(cfg.store_id, tmp_path)
    assert len(out) == 12
    assert out["t2m"].min() == 100.0
    assert len(list((tmp_path / "packed/dwd/10min/v1/observations/year=2012").iterdir())) == 1


def test_part_key_lets_two_tasks_share_a_partition(tmp_path: Path):
    cfg = make_cfg()
    points.write(obs(["00044"], datetime(2012, 1, 1), 4), cfg, tmp_path, part_key="00044")
    points.write(obs(["01048"], datetime(2012, 1, 1), 4), cfg, tmp_path, part_key="01048")
    out = points.read(cfg.store_id, tmp_path)
    assert len(out) == 8
    assert sorted(out["station_id"].unique().to_list()) == ["00044", "01048"]

    # re-running one task replaces only its own shard
    points.write(obs(["00044"], datetime(2012, 1, 1), 4, offset=50.0), cfg, tmp_path,
                 part_key="00044")
    out = points.read(cfg.store_id, tmp_path)
    assert len(out) == 8
    assert out.filter(pl.col("station_id") == "00044")["t2m"].min() == 50.0
    assert out.filter(pl.col("station_id") == "01048")["t2m"].min() == 0.0


def test_read_pushes_down_time_station_and_column_filters(tmp_path: Path):
    cfg = make_cfg()
    points.write(obs(["00044", "01048"], datetime(2012, 1, 1), 6), cfg, tmp_path)

    window = points.read(
        cfg.store_id, tmp_path,
        time_range=(utc(2012, 1, 1, 0, 10), utc(2012, 1, 1, 0, 30)),
    )
    assert window["time"].unique().sort().to_list() == [
        datetime(2012, 1, 1, 0, 10), datetime(2012, 1, 1, 0, 20)
    ]  # half-open: 00:30 excluded

    one = points.read(cfg.store_id, tmp_path, station_ids=["01048"], columns=["time", "t2m"])
    assert one.columns == ["time", "t2m"]
    assert len(one) == 6

    assert len(points.read(cfg.store_id, tmp_path, station_ids=[])) == 0


def test_read_on_a_missing_store_is_an_error_not_an_empty_frame(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        points.read("dwd/10min/v1", tmp_path)


def test_to_matrix_places_values_and_leaves_gaps_nan(tmp_path: Path):
    cfg = make_cfg()
    frame = pl.DataFrame(
        {
            "time": [
                datetime(2012, 1, 1, 0, 0),
                datetime(2012, 1, 1, 0, 20),
                datetime(2012, 1, 1, 0, 50),
                datetime(2012, 1, 1, 0, 20),
            ],
            "station_id": ["00044", "00044", "00044", "01048"],
            "t2m": [1.0, 2.0, 3.0, 4.0],
        }
    )
    points.write(frame, cfg, tmp_path)
    timeline = Timeline(utc(2012, 1, 1), utc(2012, 1, 1, 1), CADENCE)

    values, ids = points.to_matrix(cfg.store_id, tmp_path, "t2m", timeline)
    assert ids == ["00044", "01048"]
    assert values.shape == (6, 2)
    assert values.dtype == np.float32

    expected = np.full((6, 2), np.nan, dtype=np.float32)
    expected[0, 0], expected[2, 0], expected[5, 0] = 1.0, 2.0, 3.0
    expected[2, 1] = 4.0
    np.testing.assert_array_equal(values, expected)


def test_to_matrix_drops_off_cadence_and_out_of_range_rows(tmp_path: Path):
    cfg = make_cfg()
    frame = pl.DataFrame(
        {
            "time": [
                datetime(2012, 1, 1, 0, 10),  # on axis
                datetime(2012, 1, 1, 0, 15),  # off cadence -> dropped, not snapped
                datetime(2012, 1, 1, 2, 0),   # after the timeline -> dropped
                datetime(2011, 12, 31, 23, 50),  # before the timeline -> dropped
            ],
            "station_id": ["00044"] * 4,
            "t2m": [1.0, 99.0, 99.0, 99.0],
        }
    )
    points.write(frame, cfg, tmp_path)
    timeline = Timeline(utc(2012, 1, 1), utc(2012, 1, 1, 1), CADENCE)

    values, ids = points.to_matrix(cfg.store_id, tmp_path, "t2m", timeline)
    assert ids == ["00044"]
    assert values[1, 0] == 1.0
    assert np.isnan(values[[0, 2, 3, 4, 5], 0]).all()
    assert 99.0 not in values


def test_to_matrix_honours_a_requested_station_axis(tmp_path: Path):
    cfg = make_cfg()
    points.write(obs(["00044"], datetime(2012, 1, 1), 2), cfg, tmp_path)
    timeline = Timeline(utc(2012, 1, 1), utc(2012, 1, 1, 0, 20), CADENCE)

    values, ids = points.to_matrix(
        cfg.store_id, tmp_path, "t2m", timeline, station_ids=["01048", "00044"]
    )
    assert ids == ["01048", "00044"]
    assert np.isnan(values[:, 0]).all()            # asked for, never observed
    assert values[:, 1].tolist() == [0.0, 1.0]     # order of ids is respected


def test_to_matrix_on_an_empty_window_is_all_nan(tmp_path: Path):
    cfg = make_cfg()
    points.write(obs(["00044"], datetime(2012, 1, 1), 2), cfg, tmp_path)
    timeline = Timeline(utc(2020, 1, 1), utc(2020, 1, 1, 1), CADENCE)
    values, ids = points.to_matrix(cfg.store_id, tmp_path, "t2m", timeline)
    assert ids == []
    assert values.shape == (6, 0)


def test_station_metadata_lives_apart_and_joins_by_id(tmp_path: Path):
    cfg = make_cfg()
    points.write(obs(["00044", "01048"], datetime(2012, 1, 1), 2), cfg, tmp_path)
    points.write_stations(
        pl.DataFrame(
            {
                "station_id": ["00044", "01048"],
                "lat": [52.1, 53.2],
                "lon": [4.3, 5.4],
                "height": [1.0, 2.0],
                "name": ["De Bilt", "Cabauw"],
            }
        ),
        cfg.store_id,
        tmp_path,
    )
    _, ids = points.to_matrix(cfg.store_id, tmp_path, "t2m",
                              Timeline(utc(2012, 1, 1), utc(2012, 1, 1, 0, 20), CADENCE))
    meta = points.read_stations(cfg.store_id, tmp_path)
    joined = pl.DataFrame({"station_id": ids}).join(meta, on="station_id", how="left")
    assert joined["name"].to_list() == ["De Bilt", "Cabauw"]
    assert joined["lat"].null_count() == 0


def test_station_metadata_tolerates_partial_columns_and_latest_wins(tmp_path: Path):
    cfg = make_cfg()
    points.write_stations(pl.DataFrame({"station_id": ["00044"], "lat": [52.0]}),
                          cfg.store_id, tmp_path, part_key="batch-a")
    meta = points.read_stations(cfg.store_id, tmp_path)
    assert meta["height"].null_count() == 1

    later = datetime.now(UTC).replace(tzinfo=None) + timedelta(hours=1)
    points.write_stations(
        pl.DataFrame({"station_id": ["00044"], "lat": [52.5], "updated_at": [later]}),
        cfg.store_id, tmp_path, part_key="batch-b",
    )
    meta = points.read_stations(cfg.store_id, tmp_path)
    assert len(meta) == 1
    assert meta["lat"][0] == 52.5


def test_read_stations_on_a_store_without_metadata_is_empty_not_an_error(tmp_path: Path):
    assert points.read_stations("dwd/10min/v1", tmp_path).is_empty()


def test_write_rejects_a_gridded_config_and_missing_columns(tmp_path: Path):
    gridded = ProductConfig(source="dwd", product="10min", store=StoreConfig(kind="gridded"))
    with pytest.raises(ValueError, match="points"):
        points.write(obs(["00044"], datetime(2012, 1, 1), 1), gridded, tmp_path)

    cfg = make_cfg()
    with pytest.raises(ValueError, match="station_id"):
        points.write(pl.DataFrame({"time": [datetime(2012, 1, 1)]}), cfg, tmp_path)

    with pytest.raises(ValueError, match="partition key"):
        points.write(obs(["00044"], datetime(2012, 1, 1), 1), make_cfg(["nonsense"]), tmp_path)


def test_concurrent_writers_to_distinct_partitions_do_not_corrupt_the_store(tmp_path: Path):
    cfg = make_cfg(["year", "month"])
    months = list(range(1, 13))

    def write_month(month: int) -> None:
        points.write(obs(["00044", "01048"], datetime(2012, month, 1), 6), cfg, tmp_path)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(write_month, months))

    out = points.read(cfg.store_id, tmp_path)
    assert len(out) == len(months) * 2 * 6
    assert out["time"].dt.month().unique().sort().to_list() == months
    assert out.n_unique(subset=["time", "station_id"]) == len(out)


def test_concurrent_writers_to_one_partition_keep_their_own_shards(tmp_path: Path):
    cfg = make_cfg()
    stations = [f"{i:05d}" for i in range(12)]

    def write_station(station: str) -> None:
        points.write(obs([station], datetime(2012, 1, 1), 6), cfg, tmp_path, part_key=station)

    with ThreadPoolExecutor(6) as pool:
        list(pool.map(write_station, stations))

    out = points.read(cfg.store_id, tmp_path)
    assert len(out) == len(stations) * 6
    assert sorted(out["station_id"].unique().to_list()) == stations
    assert out.n_unique(subset=["time", "station_id"]) == len(out)


def test_partition_keys_may_be_a_real_column_or_absent(tmp_path: Path):
    frame = pl.DataFrame(
        {
            "time": [datetime(2012, 1, 1), datetime(2013, 1, 1)],
            "station_id": ["00044", "01048"],
            "t2m": [1.0, 2.0],
        }
    )
    by_station = make_cfg(["station_id", "year"])
    written = points.write(frame, by_station, tmp_path)
    assert [p.parent.relative_to(tmp_path).as_posix() for p in written] == [
        "packed/dwd/10min/v1/observations/station_id=00044/year=2012",
        "packed/dwd/10min/v1/observations/station_id=01048/year=2013",
    ]
    assert len(points.read(by_station.store_id, tmp_path)) == 2

    flat = ProductConfig(
        source="knmi", product="10min",
        store=StoreConfig(kind="points", partition_by=[]),
    )
    assert points.write(frame, flat, tmp_path) == [
        tmp_path / "packed/knmi/10min/v1/observations/data.parquet"
    ]
    assert len(points.read(flat.store_id, tmp_path)) == 2


def _process_writer(root_str: str, station: str) -> None:
    points.write(
        obs([station], datetime(2012, 1, 1), 6),
        make_cfg(),
        Path(root_str),
        part_key=station,
    )


def test_concurrent_processes_writing_one_partition_keep_every_row(tmp_path: Path):
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    stations = [f"{i:05d}" for i in range(6)]
    procs = [ctx.Process(target=_process_writer, args=(str(tmp_path), s)) for s in stations]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    out = points.read(make_cfg().store_id, tmp_path)
    assert len(out) == len(stations) * 6
    assert sorted(out["station_id"].unique().to_list()) == stations
    assert out["t2m"].null_count() == 0
