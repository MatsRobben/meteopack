"""Gridded fields: one frame per chunk, days written in parallel, nothing shifted."""
from __future__ import annotations

import multiprocessing as mp
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import xarray as xr

from meteopack.core import grids
from meteopack.core.config import ProductConfig, StoreConfig, Variable
from meteopack.core.grids import Grid
from meteopack.core.store import gridded
from meteopack.core.timeline import Timeline, utc

NY, NX = 8, 6
CADENCE_SECONDS = 3600
FRAMES_PER_DAY = 86_400 // CADENCE_SECONDS
DAYS = 4

#: The real registry holds one 765x700 radar grid; tests want frames you can eyeball.
TINY = Grid(
    id="tiny_test",
    proj4="+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 +b=6356750 +units=km +no_defs",
    shape=(NY, NX),
    # Real corners, so initialise exercises the coordinate and lon/lat paths
    # rather than the degraded one. A grid with no geometry is a separate test.
    corners_lonlat=((3.0, 52.0), (3.0, 53.0), (5.0, 53.0), (5.0, 52.0)),
)


@pytest.fixture(autouse=True)
def _tiny_grid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(grids, "get", lambda grid_id, path=None: TINY)


def make_cfg(
    chunks: dict[str, int] | None = None,
    shard: dict[str, int] | None = None,
    variables: list[str] | None = None,
) -> ProductConfig:
    return ProductConfig(
        source="knmi",
        product="test",
        cadence_seconds=CADENCE_SECONDS,
        grid_id=TINY.id,
        variables=[
            Variable(name=name, source_key=name, units="mm", long_name=f"{name} field")
            for name in (variables or ["rain"])
        ],
        store=StoreConfig(kind="gridded", chunks=chunks, shard=shard, compression_level=1),
    )


def make_timeline() -> Timeline:
    return Timeline(
        start=utc(2012, 1, 1),
        end=utc(2012, 1, 1 + DAYS),
        cadence=timedelta(seconds=CADENCE_SECONDS),
    )


def block(rows: np.ndarray, offset: float = 0.0) -> np.ndarray:
    """Frames whose every cell is unique, so a torn or shifted write cannot hide.

    All values are integers below 2**24, hence exact in float32 — equality
    assertions mean what they say.
    """
    cells = np.arange(NY * NX, dtype="float32").reshape(NY, NX)
    return np.asarray(rows, dtype="float32")[:, None, None] * 1000.0 + cells + offset


def dataset(times: np.ndarray, values: np.ndarray, name: str = "rain") -> xr.Dataset:
    return xr.Dataset({name: (gridded.DIMS, values)}, coords={"time": times})


def day_dataset(timeline: Timeline, day: int, offset: float = 0.0) -> xr.Dataset:
    rows = np.arange(day * FRAMES_PER_DAY, (day + 1) * FRAMES_PER_DAY)
    return dataset(timeline.values()[rows], block(rows, offset))


def write_day(root: str, day: int, barrier=None, passes: int = 3) -> int:
    """One worker's job. Top-level so a spawned process can unpickle it.

    The barrier releases every worker at once and the repeated passes keep them
    inside `write` together, so the concurrency tests overlap for real rather
    than finishing one after another.
    """
    cfg, timeline = make_cfg(), make_timeline()
    if barrier is not None:
        barrier.wait(timeout=60)
    for _ in range(passes):
        gridded.write(day_dataset(timeline, day), cfg, root)
    return day


def test_initialise_builds_the_geometry_the_config_asked_for(tmp_path: Path):
    cfg = make_cfg(chunks={"time": 1, "y": NY, "x": NX}, shard={"time": 12, "y": NY, "x": NX})
    gridded.initialise(cfg, tmp_path, make_timeline())
    path = gridded.zarr_path(tmp_path, cfg.store_id)

    assert path == tmp_path / "packed/knmi/test/v1" / gridded.ZARR_NAME
    import zarr

    array = zarr.open_group(path, mode="r")["rain"]
    assert array.shape == (DAYS * FRAMES_PER_DAY, NY, NX)
    assert array.chunks == (1, NY, NX)
    assert array.shards == (12, NY, NX)
    assert array.dtype == np.dtype("float32")
    assert array.attrs["units"] == "mm"
    assert array.attrs["long_name"] == "rain field"


def test_defaults_put_one_frame_per_chunk_and_one_day_per_shard(tmp_path: Path):
    gridded.initialise(make_cfg(), tmp_path, make_timeline())
    path = gridded.zarr_path(tmp_path, make_cfg().store_id)
    reported = gridded.info("knmi/test/v1", tmp_path)
    assert reported["chunks"] == (1, NY, NX)
    assert reported["shards"] == (FRAMES_PER_DAY, NY, NX)

    # One file per shard, not one per frame: a day of frames is a single file.
    gridded.write(day_dataset(make_timeline(), 1), make_cfg(), tmp_path)
    assert [p.name for p in sorted((path / "rain" / "c").rglob("*")) if p.is_file()] == ["0"]


def test_shard_must_be_a_multiple_of_the_chunk(tmp_path: Path):
    cfg = make_cfg(chunks={"time": 5, "y": NY, "x": NX}, shard={"time": 12, "y": NY, "x": NX})
    with pytest.raises(ValueError, match="not a multiple"):
        gridded.initialise(cfg, tmp_path, make_timeline())


def test_round_trip_is_exact_and_untouched_frames_are_nan(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    gridded.write(day_dataset(timeline, 1), cfg, tmp_path)

    out = gridded.read("knmi/test/v1", tmp_path)
    assert out["rain"].dims == gridded.DIMS
    assert out.sizes["time"] == DAYS * FRAMES_PER_DAY
    np.testing.assert_array_equal(out["time"].values, timeline.values())

    values = out["rain"].values
    written = slice(FRAMES_PER_DAY, 2 * FRAMES_PER_DAY)
    np.testing.assert_array_equal(values[written], block(np.arange(*written.indices(10**6))))
    assert np.isnan(values[:FRAMES_PER_DAY]).all()
    assert np.isnan(values[2 * FRAMES_PER_DAY :]).all()
    assert out.attrs["grid_id"] == TINY.id
    assert out.attrs["proj4"] == TINY.proj4
    assert out.attrs["cadence_seconds"] == CADENCE_SECONDS


def test_out_of_order_day_writes_land_at_their_own_indices(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    for day in (3, 0, 2, 1):
        gridded.write(day_dataset(timeline, day), cfg, tmp_path)

    values = gridded.read("knmi/test/v1", tmp_path)["rain"].values
    np.testing.assert_array_equal(values, block(np.arange(DAYS * FRAMES_PER_DAY)))


def test_shuffled_frames_inside_one_dataset_are_scattered_not_appended(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    rows = np.array([30, 2, 17, 3, 71, 0])
    gridded.write(dataset(timeline.values()[rows], block(rows)), cfg, tmp_path)

    values = gridded.read("knmi/test/v1", tmp_path)["rain"].values
    np.testing.assert_array_equal(values[rows], block(rows))
    rest = np.setdiff1d(np.arange(DAYS * FRAMES_PER_DAY), rows)
    assert np.isnan(values[rest]).all()


def test_rewriting_a_day_overwrites_it_and_leaves_the_rest_alone(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    gridded.write(day_dataset(timeline, 1), cfg, tmp_path)
    gridded.write(day_dataset(timeline, 2), cfg, tmp_path)
    gridded.write(day_dataset(timeline, 1, offset=500_000.0), cfg, tmp_path)

    out = gridded.read("knmi/test/v1", tmp_path)
    assert out.sizes["time"] == DAYS * FRAMES_PER_DAY  # no growth, no duplication
    values = out["rain"].values
    day1 = np.arange(FRAMES_PER_DAY, 2 * FRAMES_PER_DAY)
    day2 = np.arange(2 * FRAMES_PER_DAY, 3 * FRAMES_PER_DAY)
    np.testing.assert_array_equal(values[day1], block(day1, offset=500_000.0))
    np.testing.assert_array_equal(values[day2], block(day2))
    assert np.isnan(values[:FRAMES_PER_DAY]).all()


def test_off_cadence_and_out_of_range_frames_are_dropped_not_snapped(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    on_axis = timeline.values()[5]
    times = np.array(
        [
            on_axis,
            on_axis + np.timedelta64(7, "m"),  # off cadence
            timeline.values()[0] - np.timedelta64(1, "h"),  # before the axis
            np.datetime64("2012-01-05T00:00:00", "ns"),  # past the axis
        ]
    )
    gridded.write(dataset(times, block(np.array([5.0, 999.0, 998.0, 997.0]))), cfg, tmp_path)

    values = gridded.read("knmi/test/v1", tmp_path)["rain"].values
    np.testing.assert_array_equal(values[5], block(np.array([5]))[0])
    assert np.isnan(values[np.arange(DAYS * FRAMES_PER_DAY) != 5]).all()


def test_two_writes_into_one_shard_merge_when_serialised(tmp_path: Path):
    """The read-modify-write path works — it is only concurrency it cannot survive."""
    cfg = make_cfg(shard={"time": 2 * FRAMES_PER_DAY, "y": NY, "x": NX})
    timeline = make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    gridded.write(day_dataset(timeline, 1), cfg, tmp_path)
    gridded.write(day_dataset(timeline, 0), cfg, tmp_path)

    values = gridded.read("knmi/test/v1", tmp_path)["rain"].values
    shared = np.arange(2 * FRAMES_PER_DAY)
    np.testing.assert_array_equal(values[shared], block(shared))


def test_threads_write_disjoint_days_without_losing_any(tmp_path: Path):
    gridded.initialise(make_cfg(), tmp_path, make_timeline())
    barrier = threading.Barrier(DAYS)
    with ThreadPoolExecutor(max_workers=DAYS) as pool:
        done = list(pool.map(lambda d: write_day(str(tmp_path), d, barrier), range(DAYS)))

    assert sorted(done) == list(range(DAYS))
    values = gridded.read("knmi/test/v1", tmp_path)["rain"].values
    np.testing.assert_array_equal(values, block(np.arange(DAYS * FRAMES_PER_DAY)))


def test_spawned_processes_write_disjoint_days_without_tearing(tmp_path: Path):
    gridded.initialise(make_cfg(), tmp_path, make_timeline())
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(DAYS)
    workers = [
        ctx.Process(target=write_day, args=(str(tmp_path), day, barrier)) for day in range(DAYS)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=120)
    assert [w.exitcode for w in workers] == [0] * DAYS

    values = gridded.read("knmi/test/v1", tmp_path)["rain"].values
    np.testing.assert_array_equal(values, block(np.arange(DAYS * FRAMES_PER_DAY)))


def test_read_honours_the_time_range_and_the_variable_subset(tmp_path: Path):
    cfg = make_cfg(variables=["rain", "snow"])
    timeline = make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    rows = np.arange(DAYS * FRAMES_PER_DAY)
    times = timeline.values()
    gridded.write(
        xr.Dataset(
            {"rain": (gridded.DIMS, block(rows)), "snow": (gridded.DIMS, block(rows, 1_000_000.0))},
            coords={"time": times},
        ),
        cfg,
        tmp_path,
    )

    out = gridded.read(
        "knmi/test/v1",
        tmp_path,
        time_range=(utc(2012, 1, 2), utc(2012, 1, 3)),
        variables=["snow"],
    )
    assert list(out.data_vars) == ["snow"]
    day1 = np.arange(FRAMES_PER_DAY, 2 * FRAMES_PER_DAY)
    np.testing.assert_array_equal(out["time"].values, times[day1])
    np.testing.assert_array_equal(out["snow"].values, block(day1, 1_000_000.0))

    # Half-open, and a window starting mid-cadence does not reach back for 00:00.
    narrow = gridded.read(
        "knmi/test/v1", tmp_path, time_range=(utc(2012, 1, 2, 0, 30), utc(2012, 1, 2, 2))
    )
    np.testing.assert_array_equal(narrow["time"].values, times[[25]])
    np.testing.assert_array_equal(narrow["rain"].values, block(np.array([25])))


def test_info_reports_shape_chunks_and_time_bounds(tmp_path: Path):
    cfg = make_cfg(chunks={"time": 1, "y": NY, "x": NX}, shard={"time": 12, "y": NY, "x": NX})
    gridded.initialise(cfg, tmp_path, make_timeline())
    reported = gridded.info("knmi/test/v1", tmp_path)

    assert reported["shape"] == (DAYS * FRAMES_PER_DAY, NY, NX)
    assert reported["chunks"] == (1, NY, NX)
    assert reported["shards"] == (12, NY, NX)
    assert reported["dims"] == ("time", "y", "x")
    assert reported["n_times"] == DAYS * FRAMES_PER_DAY
    assert reported["time_start"] == "2012-01-01T00:00:00+00:00"
    assert reported["time_end"] == "2012-01-05T00:00:00+00:00"
    assert reported["cadence_seconds"] == CADENCE_SECONDS
    assert reported["grid_id"] == TINY.id
    assert reported["variables"]["rain"]["units"] == "mm"


def test_initialise_refuses_to_clobber_unless_told_to(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    gridded.write(day_dataset(timeline, 0), cfg, tmp_path)
    with pytest.raises(FileExistsError):
        gridded.initialise(cfg, tmp_path, timeline)

    gridded.initialise(cfg, tmp_path, timeline, overwrite=True)
    assert np.isnan(gridded.read("knmi/test/v1", tmp_path)["rain"].values).all()


def test_write_before_initialise_says_so(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    with pytest.raises(FileNotFoundError, match="initialise"):
        gridded.write(day_dataset(timeline, 0), cfg, tmp_path)


def test_write_rejects_a_variable_the_store_has_no_array_for(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    ds = day_dataset(timeline, 0).rename({"rain": "hail"})
    with pytest.raises(ValueError, match="hail"):
        gridded.write(ds, cfg, tmp_path)


def test_write_rejects_frames_of_the_wrong_grid(tmp_path: Path):
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    ds = xr.Dataset(
        {"rain": (gridded.DIMS, np.zeros((2, NY + 1, NX), "float32"))},
        coords={"time": timeline.values()[:2]},
    )
    with pytest.raises(ValueError, match="store is"):
        gridded.write(ds, cfg, tmp_path)


def test_store_path_follows_the_layer_layout(tmp_path: Path):
    assert gridded.store_path(tmp_path, "knmi/test/v1") == tmp_path / "packed/knmi/test/v1"
    assert gridded.store_path(tmp_path, "knmi/test/v1", layer="training") == (
        tmp_path / "training/knmi/test/v1"
    )
    assert os.fspath(gridded.store_path(str(tmp_path), "a/b/v2")).endswith("packed/a/b/v2")


# --------------------------------------------------------------- growing axis


def test_write_reports_frames_the_store_refused(tmp_path: Path):
    """A frame past the end of the axis is dropped; saying so is the whole point.

    `write` used to return only a path, so callers counted their input and
    reported it as written. Packing four days one day at a time printed "wrote
    576 frames" three times while keeping 288.
    """
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)

    inside = gridded.write(day_dataset(timeline, 1), cfg, tmp_path)
    assert (inside.written, inside.outside, inside.off_cadence) == (FRAMES_PER_DAY, 0, 0)

    beyond = timeline.values()[-1] + np.timedelta64(CADENCE_SECONDS, "s") * np.arange(1, 4)
    result = gridded.write(dataset(beyond, block(np.arange(3))), cfg, tmp_path)
    assert (result.written, result.outside) == (0, 3)

    half = timeline.values()[0] + np.timedelta64(CADENCE_SECONDS // 2, "s")
    ragged = gridded.write(dataset(np.array([half]), block(np.arange(1))), cfg, tmp_path)
    assert (ragged.written, ragged.outside, ragged.off_cadence) == (0, 0, 1)


def test_a_refused_frame_says_which_frame_it_was(tmp_path: Path):
    """A count alone does not say which end of the axis is short.

    One frame out of 770,115 was dropped on the real radar archive and placing
    it meant re-reading tars by hand: the axis had been started one step late,
    so the archive's very first frame fell before it. The timestamps say that
    in the error itself.
    """
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    step = np.timedelta64(CADENCE_SECONDS, "s")

    early = timeline.values()[0] - step
    before = gridded.write(dataset(np.array([early]), block(np.arange(1))), cfg, tmp_path)
    assert before.outside == 1
    assert before.outside_first == before.outside_last == str(early)

    beyond = timeline.values()[-1] + step * np.arange(1, 4)
    after = gridded.write(dataset(beyond, block(np.arange(3))), cfg, tmp_path)
    assert (after.outside_first, after.outside_last) == (str(beyond[0]), str(beyond[-1]))

    kept = gridded.write(day_dataset(timeline, 1), cfg, tmp_path)
    assert (kept.outside, kept.outside_first, kept.outside_last) == (0, None, None)


def test_extending_the_axis_keeps_every_frame_already_written(tmp_path: Path):
    """Day-by-day packing must equal packing the whole window at once.

    This is the operational case: tomorrow does not exist when today is packed.
    Before `extend`, the second day and everything after it was written into a
    store that had no room for it and vanished without a word.
    """
    cfg, timeline = make_cfg(), make_timeline()
    one_day = Timeline(start=timeline.start, end=timeline.start + timedelta(days=1),
                       cadence=timeline.cadence)
    gridded.initialise(cfg, tmp_path, one_day)

    for day in range(DAYS):
        grown = Timeline(start=timeline.start,
                         end=timeline.start + timedelta(days=day + 1),
                         cadence=timeline.cadence)
        gridded.extend(cfg, tmp_path, grown)
        result = gridded.write(day_dataset(timeline, day), cfg, tmp_path)
        assert result.outside == 0, f"day {day} did not fit the extended axis"
        assert result.written == FRAMES_PER_DAY

    whole = tmp_path / "whole"
    gridded.initialise(cfg, whole, timeline)
    for day in range(DAYS):
        gridded.write(day_dataset(timeline, day), cfg, whole)

    grown_arr = gridded._open(cfg.store_id, tmp_path, "packed", "r")["rain"]
    whole_arr = gridded._open(cfg.store_id, whole, "packed", "r")["rain"]
    n = len(timeline)
    np.testing.assert_array_equal(grown_arr[:n], whole_arr[:n])
    assert np.isfinite(np.asarray(grown_arr[:n])).sum() == n * NY * NX


def test_a_store_grown_day_by_day_reads_like_one_packed_at_once(tmp_path: Path):
    """What a consumer sees, not what the arrays hold.

    `extend` rewrote the time axis in seconds under a nanoseconds unit, so every
    timestamp read back as 1970, and it never resized the provenance arrays, so
    `xr.open_zarr` and `gridded.read` refused the store ("conflicting sizes").
    The test above compared one data array and passed throughout.
    """
    cfg, timeline = make_cfg(), make_timeline()
    grown, whole = tmp_path / "grown", tmp_path / "whole"
    first = Timeline(start=timeline.start, end=timeline.start + timedelta(days=1),
                     cadence=timeline.cadence)
    gridded.initialise(cfg, grown, first)
    for day in range(DAYS):
        gridded.extend(cfg, grown, Timeline(start=timeline.start,
                                            end=timeline.start + timedelta(days=day + 1),
                                            cadence=timeline.cadence))
        gridded.write(day_dataset(timeline, day), cfg, grown)
    gridded.initialise(cfg, whole, timeline)
    for day in range(DAYS):
        gridded.write(day_dataset(timeline, day), cfg, whole)

    n = len(timeline)
    a = xr.open_zarr(gridded.zarr_path(grown, cfg.store_id)).isel(time=slice(0, n))
    b = xr.open_zarr(gridded.zarr_path(whole, cfg.store_id))
    np.testing.assert_array_equal(a["time"].values, timeline.values())
    for name in ("time", "rain", *gridded.PROVENANCE):
        np.testing.assert_array_equal(a[name].values, b[name].values, err_msg=name)
    # Past the last written day, the provenance says "nothing here" like fresh space does.
    tail = xr.open_zarr(gridded.zarr_path(grown, cfg.store_id)).isel(time=slice(n, None))
    if tail.sizes["time"]:
        assert bool(tail["is_missing_frame"].all())
    read = gridded.read(cfg.store_id, grown)
    assert read.sizes["time"] >= n


def test_extend_refuses_to_move_frames_that_are_already_written(tmp_path: Path):
    """Growing backwards would reindex every existing frame. Rebuild instead."""
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)

    earlier = Timeline(start=timeline.start - timedelta(days=1), end=timeline.end,
                       cadence=timeline.cadence)
    with pytest.raises(ValueError, match="extending backwards"):
        gridded.extend(cfg, tmp_path, earlier)

    assert gridded.extend(cfg, tmp_path, timeline) == 0

    off_grid = Timeline(start=timeline.start + timedelta(seconds=CADENCE_SECONDS // 2),
                        end=timeline.end + timedelta(days=1), cadence=timeline.cadence)
    with pytest.raises(ValueError, match="not on"):
        gridded.extend(cfg, tmp_path, off_grid)


# ------------------------------------------------- schema: what a store carries


def test_a_store_is_georeferenced_and_self_describing(tmp_path: Path):
    """The store must say what it is without the repository that wrote it.

    radar_v3 carried the projection, the product corners and CF discovery
    metadata; the first version of this store carried a grid id and nothing
    else, so it could not be opened as a georeferenced dataset at all.
    """
    import zarr

    cfg = make_cfg()
    gridded.initialise(cfg, tmp_path, make_timeline())
    group = zarr.open_group(gridded.zarr_path(tmp_path, cfg.store_id), mode="r")

    attrs = dict(group.attrs)
    assert attrs["Conventions"].startswith("CF-")
    assert attrs["grid_id"] == TINY.id
    crs = attrs[gridded.GRID_MAPPING_NAME]
    assert crs["proj4_params"] == TINY.proj4
    assert crs["shape"] == [NY, NX]
    assert len(crs["corners_lonlat"]) == 4

    names = {name for name, _ in group.arrays()}
    assert {"y", "x", "lat", "lon"} <= names, "missing coordinates"
    assert set(gridded.PROVENANCE) <= names, "missing per-frame provenance"

    y, x = np.asarray(group["y"][:]), np.asarray(group["x"][:])
    assert (y.shape, x.shape) == ((NY,), (NX,))
    assert np.all(np.diff(y) < 0), "y must descend: row 0 is the northern edge"
    assert np.all(np.diff(x) > 0)
    assert group["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert group["lat"].attrs["units"] == "degrees_north"
    assert np.asarray(group["lat"][:]).shape == (NY, NX)

    assert group["rain"].attrs["coordinates"] == "lat lon"
    assert group["rain"].attrs["grid_mapping"] == gridded.GRID_MAPPING_NAME


def test_the_time_axis_is_nanoseconds(tmp_path: Path):
    """datetime64[ns] is what every reader here uses; seconds forced a lossy
    conversion at both ends and could not hold a sub-second cadence."""
    import zarr

    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    group = zarr.open_group(gridded.zarr_path(tmp_path, cfg.store_id), mode="r")
    assert "nanoseconds" in group["time"].attrs["units"]
    stored = np.asarray(group["time"][:]).astype("datetime64[ns]")
    np.testing.assert_array_equal(stored, timeline.values())


def test_manifest_lives_beside_the_zarr_group_not_inside_it(tmp_path: Path):
    """Zarr warns about every foreign object in a hierarchy, and is right to."""
    cfg = make_cfg()
    store = gridded.initialise(cfg, tmp_path, make_timeline())
    assert store.name == "v1"
    assert (store / gridded.ZARR_NAME / "zarr.json").exists()
    assert not (store / "zarr.json").exists()


def test_an_integer_variable_is_stored_as_codes_with_its_calibration(tmp_path: Path):
    """float32 doubles a store whose source is already integer, and cannot
    tell `missing` from `out_of_image`."""
    import zarr

    from meteopack.core.config import ProductConfig, StoreConfig, Variable

    cfg = ProductConfig(
        source="knmi", product="test", cadence_seconds=CADENCE_SECONDS, grid_id=TINY.id,
        variables=[Variable(
            name="precipitation_amount", source_key="x", units="mm",
            standard_name="precipitation_amount", legacy_name="accumulation_mm",
            encoding={"dtype": "uint16", "scale": 0.01, "offset": 0.0},
        )],
        store=StoreConfig(kind="gridded", chunks={"time": 1}, compression_level=1),
    )
    timeline = make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    group = zarr.open_group(gridded.zarr_path(tmp_path, cfg.store_id), mode="r")
    array = group["precipitation_amount"]
    assert array.dtype == np.dtype("uint16")
    assert array.attrs["calibration_missing_data"] == 65534
    assert array.attrs["calibration_out_of_image"] == 65535
    assert array.attrs["standard_name"] == "precipitation_amount"
    assert array.attrs["legacy_name"] == "accumulation_mm"

    codes = np.array([[[0, 1, 2, 3, 4, 5]] * NY], dtype="uint16")
    ds = xr.Dataset({"precipitation_amount": (gridded.DIMS, codes)},
                    coords={"time": timeline.values()[:1]})
    result = gridded.write(ds, cfg, tmp_path)
    assert result.written == 1

    back = zarr.open_group(gridded.zarr_path(tmp_path, cfg.store_id), mode="r")
    np.testing.assert_array_equal(np.asarray(back["precipitation_amount"][0]), codes[0])


def test_writing_floats_into_an_integer_store_is_refused(tmp_path: Path):
    """Casting here would round every value without telling anyone."""
    from meteopack.core.config import ProductConfig, StoreConfig, Variable

    cfg = ProductConfig(
        source="knmi", product="test", cadence_seconds=CADENCE_SECONDS, grid_id=TINY.id,
        variables=[Variable(name="rain", source_key="x", units="mm",
                            encoding={"dtype": "uint16", "scale": 0.01})],
        store=StoreConfig(kind="gridded", chunks={"time": 1}, compression_level=1),
    )
    timeline = make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    ds = dataset(timeline.values()[:1], block(np.arange(1)))
    with pytest.raises(ValueError, match="Encode with meteopack.core.encoding"):
        gridded.write(ds, cfg, tmp_path)


def test_per_frame_provenance_is_written_with_the_frames(tmp_path: Path):
    """Measured on the real rtcor archive: 2,013 of 772,129 slots had no file
    and 36 had a file that would not read. Both read as missing imagery, so
    only these arrays can tell them apart."""
    import zarr

    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)

    rows = np.arange(3)
    ds = dataset(timeline.values()[rows], block(rows))
    ds["is_missing_frame"] = ("time", np.array([False, True, False]))
    ds["is_corrupt_frame"] = ("time", np.array([False, False, True]))
    ds["source_layout_id"] = ("time", np.array([5, 0, 1], dtype="int16"))
    ds["source_record_id"] = ("time", np.array([10, 11, 12], dtype="int64"))
    gridded.write(ds, cfg, tmp_path)

    group = zarr.open_group(gridded.zarr_path(tmp_path, cfg.store_id), mode="r")
    np.testing.assert_array_equal(np.asarray(group["is_missing_frame"][:3]),
                                  [False, True, False])
    np.testing.assert_array_equal(np.asarray(group["is_corrupt_frame"][:3]),
                                  [False, False, True])
    np.testing.assert_array_equal(np.asarray(group["source_layout_id"][:3]), [5, 0, 1])
    np.testing.assert_array_equal(np.asarray(group["source_record_id"][:3]), [10, 11, 12])
    # Untouched slots keep "not recorded", not a real layout.
    assert int(group["source_layout_id"][10]) == -1


def test_provenance_is_optional(tmp_path: Path):
    """A source that cannot say leaves the arrays at their fill value."""
    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    result = gridded.write(day_dataset(timeline, 0), cfg, tmp_path)
    assert result.written == FRAMES_PER_DAY and result.outside == 0


def test_files_sharing_a_shard_are_packed_by_one_task(tmp_path: Path):
    """A zarr shard is one file, so it may have exactly one writer.

    Two processes writing different frames of one shard each read-modify-write
    the whole shard and replace it atomically; the slower one's frames vanish.
    Almost every daily tar is exactly one shard, but two of the 2,680 KNMI
    radar tars start at 08:00 rather than 08:05 and straddle a boundary.
    """
    import argparse
    from datetime import datetime, timedelta

    from meteopack.cli import _pack_tasks

    timeline = make_timeline()
    cfg = make_cfg(chunks={"time": 1}, shard={"time": FRAMES_PER_DAY})

    class FakeSource:
        """Dates a file from its name: day0a and day0b fall in the same shard."""

        spans: ClassVar[dict] = {
            "day0a": (0, 11),      # first half of day 0
            "day0b": (12, 23),     # second half of day 0 - same shard
            "day1": (24, 47),      # day 1 - its own shard
        }

        def span_of(self, path):
            lo, hi = self.spans[Path(path).stem]
            step = timedelta(seconds=CADENCE_SECONDS)
            return (timeline.start + lo * step, timeline.start + hi * step)

    class Ctx:
        src = FakeSource()
        root = tmp_path
        args = argparse.Namespace(configs=None)

    Ctx.cfg = cfg
    files = [tmp_path / f"{n}.tar" for n in ("day0a", "day0b", "day1")]
    tasks = _pack_tasks(Ctx(), files, timeline)

    assert len(tasks) == 2, f"expected one task per shard, got {[t.paths for t in tasks]}"
    by_size = sorted(tasks, key=lambda t: -len(t.paths))
    assert len(by_size[0].paths) == 2, "the two same-shard files must share a task"
    assert {Path(p).stem for p in by_size[0].paths} == {"day0a", "day0b"}
    assert {Path(p).stem for p in by_size[1].paths} == {"day1"}
    assert isinstance(datetime.now(), datetime)


def test_a_file_spanning_two_shards_is_grouped_with_both_neighbours(tmp_path: Path):
    """Keying on the first frame alone is not enough.

    A file whose span crosses a shard boundary writes into two shards; grouped
    with only the first, it still races whoever owns the second. This is the
    real KNMI case: the 08:00 tar overlaps the shard the 08:05 tar owns.
    """
    import argparse
    from datetime import timedelta

    from meteopack.cli import _pack_tasks

    timeline = make_timeline()
    cfg = make_cfg(chunks={"time": 1}, shard={"time": FRAMES_PER_DAY})
    step = timedelta(seconds=CADENCE_SECONDS)

    class FakeSource:
        spans: ClassVar[dict] = {
            # straddles the day0/day1 boundary at frame 24
            "straddler": (20, 27),
            "day0": (0, 19),
            "day1": (28, 47),
            "day2": (48, 71),
        }

        def span_of(self, path):
            lo, hi = self.spans[Path(path).stem]
            return (timeline.start + lo * step, timeline.start + hi * step)

    class Ctx:
        src = FakeSource()
        root = tmp_path
        args = argparse.Namespace(configs=None)

    Ctx.cfg = cfg
    files = [tmp_path / f"{n}.tar" for n in ("day0", "straddler", "day1", "day2")]
    tasks = _pack_tasks(Ctx(), files, timeline)

    where = {Path(p).stem: i for i, t in enumerate(tasks) for p in t.paths}
    assert where["day0"] == where["straddler"] == where["day1"], (
        "the straddler joins both neighbours into one task: "
        f"{[sorted(Path(p).stem for p in t.paths) for t in tasks]}"
    )
    assert where["day2"] != where["day0"], "an untouched shard keeps its own task"
    assert sum(len(t.paths) for t in tasks) == len(files), "no file packed twice"


def test_write_batching_does_not_change_a_single_byte(tmp_path: Path, monkeypatch):
    """Batched chunk encoding is a speed setting only.

    Checked on the real radar sample at batch sizes 1, 16, 64 and 288; this
    keeps it true across zarr upgrades, which own the shard assembly.
    """
    cfg, timeline = make_cfg(), make_timeline()
    stores = {}
    for size in (1, gridded.WRITE_BATCH_SIZE):
        root = tmp_path / f"b{size}"
        monkeypatch.setattr(gridded, "WRITE_BATCH_SIZE", size)
        gridded.initialise(cfg, root, timeline)
        gridded.write(day_dataset(timeline, 1), cfg, root)
        stores[size] = gridded.zarr_path(root, cfg.store_id)
    a, b = stores.values()
    files = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file() and p.name != "zarr.json")
    assert files and files == sorted(
        p.relative_to(b) for p in b.rglob("*") if p.is_file() and p.name != "zarr.json")
    for rel in files:
        assert (a / rel).read_bytes() == (b / rel).read_bytes(), rel


def _encoded_cfg() -> ProductConfig:
    cfg = make_cfg()
    cfg.variables[0].encoding = {"dtype": "uint16", "scale": 0.01, "offset": 0.0,
                                 "missing": 65534, "out_of_image": 65535}
    return cfg


def test_gridded_read_and_xarray_agree_on_an_integer_encoded_store(tmp_path: Path):
    """One store, two readers, the same numbers.

    `gridded.read` used to return the stored codes while `xr.open_zarr` decoded
    them, and training bricks were built from the codes: 1234 where the value
    is 12.34 mm, and the 65534/65535 sentinels read as rain.
    """
    cfg, timeline = _encoded_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    codes = np.full((FRAMES_PER_DAY, NY, NX), 1234, dtype="uint16")
    codes[0, 0, 0], codes[0, 0, 1], codes[1, 2, 2] = 65534, 65535, 0
    gridded.write(dataset(timeline.values()[:FRAMES_PER_DAY], codes), cfg, tmp_path)

    ours = gridded.read(cfg.store_id, tmp_path,
                        time_range=(timeline.start, timeline.start + timedelta(days=1)))["rain"]
    theirs = xr.open_zarr(gridded.zarr_path(tmp_path, cfg.store_id))["rain"].isel(
        time=slice(0, FRAMES_PER_DAY)).load()
    assert ours.dtype == theirs.dtype
    np.testing.assert_array_equal(ours.values, theirs.values)       # NaNs in the same places
    assert np.isnan(ours.values[0, 0, 0]) and np.isnan(ours.values[0, 0, 1])
    assert ours.values[1, 2, 2] == 0.0
    assert ours.values[5, 3, 3] == pytest.approx(12.34)

    raw = gridded.read(cfg.store_id, tmp_path, decode=False,
                       time_range=(timeline.start, timeline.start + timedelta(days=1)))["rain"]
    np.testing.assert_array_equal(raw.values, codes)


def test_derive_can_size_its_axis_from_a_store_the_current_code_writes(tmp_path: Path):
    """`derive` opened the store directory, not the dataset.zarr inside it.

    Every gridded store written since the group moved into dataset.zarr raised
    GroupNotFoundError there - reproduced on the real rtcor store.
    """
    from meteopack.cli import _derive_timeline
    from meteopack.core.config import DeriveConfig
    from meteopack.core.store import manifest

    cfg, timeline = make_cfg(), make_timeline()
    gridded.initialise(cfg, tmp_path, timeline)
    manifest.write(gridded.store_path(tmp_path, cfg.store_id),
                   manifest.Manifest.new(cfg.store_id, "packed", "gridded", cfg))
    derive = DeriveConfig(name="d", inputs=[cfg.store_id], grid_id=cfg.grid_id,
                          cadence_seconds=CADENCE_SECONDS, n_timesteps=2, shard_samples=4)
    axis = _derive_timeline(derive, tmp_path, None)
    assert axis.start >= timeline.start and axis.end <= timeline.end


def test_store_metadata_says_what_no_echo_and_the_layout_codes_mean(tmp_path: Path):
    """source_layout_id pointed at a root attribute that did not exist, and
    nothing told a reader that 0 km means "no echo", the commonest pixel."""
    import zarr

    cfg, timeline = make_cfg(), make_timeline()
    cfg.variables[0].no_echo = 0.0
    gridded.initialise(cfg, tmp_path, timeline)
    group = zarr.open_group(gridded.zarr_path(tmp_path, cfg.store_id), mode="r")
    assert group["rain"].attrs["no_echo_value"] == 0.0
    assert "flag_meanings" not in group["source_layout_id"].attrs

    gridded.refresh_attrs(cfg, tmp_path, layout_names={0: "missing", 1: "corrupt",
                                                       6: "echo top height"})
    group = zarr.open_group(gridded.zarr_path(tmp_path, cfg.store_id), mode="r")
    flags = group["source_layout_id"].attrs
    assert flags["flag_values"] == [0, 1, 6]
    assert flags["flag_meanings"] == "missing corrupt echo_top_height"
    assert "flag_meanings" in flags["long_name"] or "flag_values" in flags["long_name"]
    assert "corrupt_layouts" in group["is_corrupt_frame"].attrs["long_name"]
