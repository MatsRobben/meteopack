"""Derived datasets: one tiling code path, a shuffled layout, and honest splits.

The gridded store is stood in for, because these tests are about what `derive`
does with frames rather than about zarr. The stand-in is *strict*: every call it
receives is bound against the real `gridded` function's signature, and the
stand-in's own signature is compared against the real one. That is not
ceremony — this repo has already shipped a bug of exactly that shape, where
`http.fetch_file` grew a required keyword its caller never passed and every
suite stayed green because the double accepted anything.

`dense_stations` is tested against the real points store: it exists, so there is
nothing to fake and no reason to.
"""
from __future__ import annotations

import inspect
import itertools
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xarray as xr

from meteopack.core import grids
from meteopack.core.config import DeriveConfig, ProductConfig, StoreConfig
from meteopack.core.store import gridded as real_gridded
from meteopack.core.store import manifest as manifest_mod
from meteopack.core.store import points
from meteopack.core.timeline import Timeline, utc
from meteopack.derive import base, bricks, dense_stations, splits

CADENCE = timedelta(minutes=5)

#: Exactly representable in float32 for the grids used here, so an assertion
#: about a tile's contents is an assertion about geometry, not about rounding.
def expected(t: int, y: int, x: int) -> float:
    return float(t * 1_000_000 + y * 1024 + x)


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------

REGISTRY = """
tiny:
  proj4: "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +units=km +no_defs"
  shape: [8, 6]
patchy:
  proj4: "+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +units=km +no_defs"
  shape: [512, 384]
"""


@pytest.fixture
def registry(tmp_path: Path) -> str:
    path = tmp_path / "grids.yaml"
    path.write_text(REGISTRY)
    grids.load_registry.cache_clear()
    yield str(path)
    grids.load_registry.cache_clear()


def make_cfg(**kw) -> DeriveConfig:
    defaults = {
        "name": "nowcast",
        "inputs": ["radar/rtcor/v1"],
        "grid_id": "tiny",
        "cadence_seconds": int(CADENCE.total_seconds()),
        "n_timesteps": 2,
        "shard_samples": 4,
    }
    return DeriveConfig(**{**defaults, **kw})


def make_spec(cfg: DeriveConfig, registry: str, *, hours: int = 2) -> base.DeriveSpec:
    timeline = Timeline(utc(2024, 1, 1), utc(2024, 1, 1) + timedelta(hours=hours), CADENCE)
    return base.DeriveSpec(cfg, timeline, grid_registry=registry)


class StrictGridded:
    """A fake `gridded` module that refuses any call the real one would refuse."""

    def __init__(self, spec: base.DeriveSpec, variables=("rain", "dbz"), by_store=None):
        self.spec = spec
        self.variables = tuple(variables)
        self.by_store = {k: tuple(v) for k, v in (by_store or {}).items()}
        self.reads = 0
        self._real = {
            name: getattr(real_gridded, name, None)
            for name in ("store_path", "initialise", "write", "read", "info")
        }

    def _bind(self, name: str, *args, **kwargs) -> None:
        real = self._real[name]
        if real is None:                      # module not written yet
            return
        inspect.signature(real).bind(*args, **kwargs)

    def read(self, store_id, root, *, time_range=None, variables=None, layer="packed"):
        self._bind(
            "read", store_id, root, time_range=time_range, variables=variables, layer=layer
        )
        self.reads += 1
        offered = self.by_store.get(store_id, self.variables)
        names = list(variables) if variables is not None else list(offered)
        unknown = [n for n in names if n not in offered]
        if unknown:
            raise ValueError(f"{store_id} has no array for {unknown}")
        ny, nx = self.spec.grid.shape
        times = self.spec.timeline.values()
        if time_range is not None:
            lo = np.datetime64(time_range[0].replace(tzinfo=None), "ns")
            hi = np.datetime64(time_range[1].replace(tzinfo=None), "ns")
            keep = (times >= lo) & (times < hi)
        else:
            keep = np.ones(len(times), dtype=bool)
        picked = np.flatnonzero(keep)
        t = picked[:, None, None]
        y = np.arange(ny)[None, :, None]
        x = np.arange(nx)[None, None, :]
        block = (t * 1_000_000 + y * 1024 + x).astype("float32")
        return xr.Dataset(
            {name: (("time", "y", "x"), block.copy()) for name in names},
            coords={"time": times[picked]},
        )


@pytest.fixture
def fake_gridded(monkeypatch):
    def install(spec, variables=("rain", "dbz"), by_store=None):
        fake = StrictGridded(spec, variables, by_store)
        monkeypatch.setattr(real_gridded, "read", fake.read, raising=False)
        return fake

    return install


# --------------------------------------------------------------------------
# base: tiling and enumeration
# --------------------------------------------------------------------------

def test_tile_none_means_the_whole_grid(registry):
    spec = make_spec(make_cfg(), registry)
    assert spec.tile == (8, 6) == spec.grid.shape
    assert spec.tiles == ((0, 0),)
    assert spec.sample_shape == (2, 8, 6)


def test_patch_tiling_covers_the_grid_including_a_ragged_edge(registry):
    spec = make_spec(make_cfg(grid_id="patchy", tile=(256, 256)), registry)
    assert spec.tile == (256, 256)
    # y divides exactly; x does not, so the last anchor is pinned to the edge.
    assert base.tile_anchors(512, 256) == (0, 256)
    assert base.tile_anchors(384, 256) == (0, 128)
    assert spec.tiles == ((0, 0), (0, 128), (256, 0), (256, 128))
    for y0, x0 in spec.tiles:
        assert 0 <= y0 <= 512 - 256 and 0 <= x0 <= 384 - 256


def test_a_tile_larger_than_the_grid_is_an_error(registry):
    with pytest.raises(ValueError, match="exceeds grid extent"):
        make_spec(make_cfg(tile=(16, 6)), registry)


def test_sample_enumeration_is_deterministic(registry):
    one = make_spec(make_cfg(grid_id="patchy", tile=(256, 256)), registry)
    two = make_spec(make_cfg(grid_id="patchy", tile=(256, 256)), registry)
    assert one.n_samples == len(one.time_anchors) * 4 == 12 * 4
    assert [s for s in one.samples()] == [s for s in two.samples()]
    assert one.fingerprint == two.fingerprint
    assert np.array_equal(one.order, two.order)
    # ...and the shuffle is a shuffle, not the identity.
    assert not np.array_equal(one.order, np.arange(one.n_samples))


def test_time_windows_do_not_overlap(registry):
    spec = make_spec(make_cfg(n_timesteps=3), registry)
    ends = [a + 3 for a in spec.time_anchors]
    assert all(e <= n for e, n in zip(ends, spec.time_anchors[1:], strict=False))
    assert spec.time_anchors == tuple(range(0, 24 - 2, 3))


def test_shards_partition_the_slots_exactly_once(registry):
    spec = make_spec(make_cfg(grid_id="patchy", tile=(256, 256), shard_samples=5), registry)
    covered: list[int] = []
    for shard in spec.shards():
        covered.extend(shard.slots)
        assert len(shard) == len(shard.samples)
    assert covered == list(range(spec.n_samples))
    seen = sorted(s.index for shard in spec.shards() for s in shard.samples)
    assert seen == list(range(spec.n_samples))


def test_shard_indices_split_the_work_across_array_tasks(registry):
    spec = make_spec(make_cfg(grid_id="patchy", tile=(256, 256), shard_samples=5), registry)
    tasks = [spec.shard_indices(i, 4) for i in range(4)]
    assert sorted(i for t in tasks for i in t) == list(range(spec.n_shards))
    assert max(len(t) for t in tasks) - min(len(t) for t in tasks) <= 1
    assert spec.shard_indices(0, 1) == list(range(spec.n_shards))


def test_slot_and_sample_index_are_inverse(registry):
    spec = make_spec(make_cfg(), registry)
    for index in range(spec.n_samples):
        assert spec.slot_sample(spec.slot_of(index)).index == index


# --------------------------------------------------------------------------
# contract: our call sites match the real gridded signatures
# --------------------------------------------------------------------------

def test_brick_call_sites_bind_against_the_real_gridded_read():
    """bricks calls gridded.read with exactly these arguments."""
    real = getattr(real_gridded, "read", None)
    if real is None:
        pytest.skip("gridded.read is not written yet")
    inspect.signature(real).bind(
        "radar/rtcor/v1",
        Path("/tmp/root"),
        time_range=(utc(2024, 1, 1), utc(2024, 1, 2)),
        variables=["rain"],
        layer="packed",
    )


def test_the_fake_has_the_same_signature_as_the_real_gridded_read():
    real = getattr(real_gridded, "read", None)
    if real is None:
        pytest.skip("gridded.read is not written yet")
    mine = inspect.signature(StrictGridded.read)
    theirs = inspect.signature(real)
    drop_self = list(mine.parameters.values())[1:]
    assert [(p.name, p.kind) for p in drop_self] == [
        (p.name, p.kind) for p in theirs.parameters.values()
    ]


# --------------------------------------------------------------------------
# bricks
# --------------------------------------------------------------------------

def _check_contents(spec, ds):
    ty, tx = spec.tile
    for slot in range(spec.n_samples):
        sample = spec.slot_sample(slot)
        got = ds["rain"].values[slot]
        for k in range(spec.cfg.n_timesteps):
            for dy, dx in ((0, 0), (ty - 1, tx - 1)):
                assert got[k, dy, dx] == expected(
                    sample.t0 + k, sample.y0 + dy, sample.x0 + dx
                ), f"slot {slot} {sample}"


def test_full_field_brick(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(), registry)
    fake_gridded(spec)
    bricks.build(spec, tmp_path)

    assert bricks.is_complete(spec, tmp_path)
    ds = bricks.read(spec, tmp_path)
    assert ds["rain"].shape == (spec.n_samples, 2, 8, 6)
    assert ds["rain"].dims == ("sample", "time", "y", "x")
    _check_contents(spec, ds)


def test_patch_brick_takes_the_same_path(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(grid_id="patchy", tile=(256, 256), shard_samples=16), registry)
    fake_gridded(spec)
    bricks.build(spec, tmp_path)

    ds = bricks.read(spec, tmp_path, slots=slice(0, 8))
    assert ds["rain"].shape == (8, 2, 256, 256)
    assert bricks.info(spec, tmp_path)["sample_shape"] == (2, 256, 256)
    ty, tx = spec.tile
    for slot in range(8):
        sample = spec.slot_sample(slot)
        tile = ds["rain"].values[slot, 0]
        assert tile[0, 0] == expected(sample.t0, sample.y0, sample.x0)
        assert tile[ty - 1, tx - 1] == expected(sample.t0, sample.y0 + ty - 1, sample.x0 + tx - 1)


def test_the_brick_is_written_in_shuffled_order(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(), registry)
    fake_gridded(spec)
    bricks.build(spec, tmp_path)

    ds = bricks.read(spec, tmp_path)
    assert np.array_equal(ds["sample_index"].values, spec.order)
    starts = ds["start_time"].values
    assert not np.array_equal(starts, np.sort(starts))        # genuinely shuffled
    assert sorted(starts.tolist()) == sorted(
        np.repeat(spec.timeline.values()[list(spec.time_anchors)], len(spec.tiles)).tolist()
    )


def test_sample_coords_describe_the_tile_they_sit_next_to(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(grid_id="patchy", tile=(256, 256), shard_samples=16), registry)
    fake_gridded(spec)
    bricks.build(spec, tmp_path)
    ds = bricks.read(spec, tmp_path)
    for slot in range(0, spec.n_samples, 7):
        sample = spec.slot_sample(slot)
        assert int(ds["y0"].values[slot]) == sample.y0
        assert int(ds["x0"].values[slot]) == sample.x0
        assert int(ds["t0"].values[slot]) == sample.t0
        assert ds["rain"].values[slot, 0, 0, 0] == expected(sample.t0, sample.y0, sample.x0)


def test_rerunning_a_shard_overwrites_rather_than_duplicating(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(), registry)
    fake_gridded(spec)
    bricks.initialise(spec, tmp_path)
    assert bricks.write_shard(spec, 0, tmp_path) == len(spec.shard(0))
    first = bricks.read(spec, tmp_path)["rain"].values.copy()

    assert bricks.write_shard(spec, 0, tmp_path) == len(spec.shard(0))
    again = bricks.read(spec, tmp_path)["rain"].values
    assert again.shape == first.shape                       # no growth
    np.testing.assert_array_equal(np.nan_to_num(again, nan=-1), np.nan_to_num(first, nan=-1))
    assert bricks.finished_shards(spec, tmp_path) == [0]
    assert not bricks.is_complete(spec, tmp_path)


def test_a_finished_shard_is_not_redone(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(), registry)
    fake = fake_gridded(spec)
    bricks.build(spec, tmp_path)
    reads = fake.reads
    assert reads > 0

    bricks.build(spec, tmp_path)                            # resume: nothing left
    assert fake.reads == reads
    assert bricks.is_complete(spec, tmp_path)


def test_disjoint_shards_from_several_workers_match_a_single_build(
    tmp_path, registry, fake_gridded
):
    cfg = make_cfg(grid_id="patchy", tile=(256, 256), shard_samples=4)
    spec = make_spec(cfg, registry)
    fake_gridded(spec)

    serial = tmp_path / "serial"
    bricks.build(spec, serial)

    parallel = tmp_path / "parallel"
    bricks.initialise(spec, parallel)
    with ThreadPoolExecutor(max_workers=4) as pool:
        written = list(
            pool.map(lambda i: bricks.write_shard(spec, i, parallel), range(spec.n_shards))
        )
    assert sum(written) == spec.n_samples
    assert bricks.is_complete(spec, parallel)

    a = bricks.read(spec, serial)["rain"].values
    b = bricks.read(spec, parallel)["rain"].values
    np.testing.assert_array_equal(a, b)


def test_a_brick_records_its_provenance(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(), registry)
    fake_gridded(spec)
    bricks.build(spec, tmp_path)

    entry = manifest_mod.read(bricks.store_path(spec, tmp_path))
    assert entry is not None
    assert entry.kind == "bricks"
    assert entry.inputs == ["radar/rtcor/v1"]
    assert entry.n_records == spec.n_samples
    assert entry.config_hash == manifest_mod.config_hash(spec.cfg)
    assert entry.extra["tile"] == [8, 6]
    assert not manifest_mod.is_stale(bricks.store_path(spec, tmp_path), spec.cfg)


def test_a_brick_from_another_spec_is_refused(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(), registry)
    fake_gridded(spec)
    bricks.build(spec, tmp_path)

    other = make_spec(make_cfg(n_timesteps=1), registry)
    fake_gridded(other)
    with pytest.raises(ValueError, match="different spec"):
        bricks.initialise(other, tmp_path)


def test_unknown_variable_is_refused(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(), registry)
    fake_gridded(spec)
    with pytest.raises(ValueError, match="do not provide"):
        bricks.initialise(spec, tmp_path, variables=["snow"])


def test_variables_are_taken_from_whichever_input_carries_them(tmp_path, registry, fake_gridded):
    cfg = make_cfg(inputs=["radar/rtcor/v1", "satellite/msg/v1"])
    spec = make_spec(cfg, registry)
    fake_gridded(
        spec,
        by_store={"radar/rtcor/v1": ("rain",), "satellite/msg/v1": ("dbz",)},
    )
    bricks.build(spec, tmp_path)
    assert bricks.info(spec, tmp_path)["variables"] == {
        "dbz": "satellite/msg/v1",
        "rain": "radar/rtcor/v1",
    }
    ds = bricks.read(spec, tmp_path)
    _check_contents(spec, ds)
    np.testing.assert_array_equal(ds["dbz"].values, ds["rain"].values)


def test_the_same_variable_from_two_inputs_is_refused(tmp_path, registry, fake_gridded):
    spec = make_spec(make_cfg(inputs=["radar/a/v1", "radar/b/v1"]), registry)
    fake_gridded(spec, variables=("rain",))
    with pytest.raises(ValueError, match="offered by both"):
        bricks.initialise(spec, tmp_path)


def test_bricks_against_the_real_gridded_store(tmp_path):
    """End to end on the real canonical writer — no stand-in anywhere.

    The point of the fake is speed, not isolation from reality, so one test pays
    the full price: real zarr in, real brick out, on the real radar grid.
    """
    for name in ("initialise", "write", "read"):
        if getattr(real_gridded, name, None) is None:
            pytest.skip(f"gridded.{name} is not written yet")

    from meteopack.core.config import Variable

    product = ProductConfig(
        source="radar",
        product="rtcor",
        cadence_seconds=int(CADENCE.total_seconds()),
        grid_id="knmi_1km_stereo",
        variables=[Variable(name="rain", source_key="RAD", units="mm")],
        store=StoreConfig(kind="gridded", chunks={"time": 1, "y": 765, "x": 700}),
    )
    timeline = Timeline(utc(2024, 1, 1), utc(2024, 1, 1, 0, 20), CADENCE)
    real_gridded.initialise(product, tmp_path, timeline)

    ny, nx = grids.get("knmi_1km_stereo").shape
    t = np.arange(len(timeline))[:, None, None]
    frames = (t * 1_000_000 + np.arange(ny)[None, :, None] * 1024 + np.arange(nx)[None, None, :])
    real_gridded.write(
        xr.Dataset(
            {"rain": (("time", "y", "x"), frames.astype("float32"))},
            coords={"time": timeline.values()},
        ),
        product,
        tmp_path,
    )

    cfg = DeriveConfig(
        name="nowcast",
        inputs=[product.store_id],
        grid_id="knmi_1km_stereo",
        cadence_seconds=int(CADENCE.total_seconds()),
        n_timesteps=2,
        tile=(256, 256),
        shard_samples=4,
    )
    spec = base.DeriveSpec(cfg, timeline)
    assert spec.tile == (256, 256)
    assert len(spec.tiles) == 9                     # 3 x 3, last row/column pinned to the edge
    bricks.build(spec, tmp_path)

    assert bricks.is_complete(spec, tmp_path)
    ds = bricks.read(spec, tmp_path)
    assert ds["rain"].shape == (spec.n_samples, 2, 256, 256)
    _check_contents(spec, ds)


# --------------------------------------------------------------------------
# dense_stations — against the real points store
# --------------------------------------------------------------------------

STATIONS = {"00044": (52.1, 4.3, 1.0), "00260": (52.1, 5.2, 1.9), "06391": (51.5, 5.7, 19.5)}


def points_cfg() -> ProductConfig:
    return ProductConfig(
        source="knmi",
        product="in_situ",
        cadence_seconds=int(CADENCE.total_seconds()),
        store=StoreConfig(kind="points", partition_by=["year"]),
    )


@pytest.fixture
def points_store(tmp_path: Path) -> Path:
    """A real points store: three stations, 5-minute cadence, one gap."""
    cfg = points_cfg()
    timeline = Timeline(utc(2024, 1, 1), utc(2024, 1, 1, 2), CADENCE)
    times = timeline.values()
    rows = {"time": [], "station_id": [], "t2m": [], "rh": []}
    for s_i, station in enumerate(sorted(STATIONS)):
        for t_i, when in enumerate(times):
            if station == "06391" and t_i == 3:
                continue                                    # a real gap, stays NaN
            rows["time"].append(when.astype("datetime64[us]").item())
            rows["station_id"].append(station)
            rows["t2m"].append(100.0 * s_i + t_i)
            rows["rh"].append(float(s_i))
    points.write(pl.DataFrame(rows), cfg, tmp_path)
    points.write_stations(
        pl.DataFrame(
            {
                "station_id": sorted(STATIONS),
                "lat": [STATIONS[s][0] for s in sorted(STATIONS)],
                "lon": [STATIONS[s][1] for s in sorted(STATIONS)],
                "height": [STATIONS[s][2] for s in sorted(STATIONS)],
                "name": [f"station {s}" for s in sorted(STATIONS)],
            }
        ),
        cfg.store_id,
        tmp_path,
    )
    return tmp_path


def station_spec(registry: str) -> base.DeriveSpec:
    cfg = make_cfg(name="stations", inputs=["knmi/in_situ/v1"])
    return make_spec(cfg, registry)


def test_dense_stations_builds_a_time_station_matrix(points_store, registry):
    spec = station_spec(registry)
    dense_stations.build(spec, points_store)
    ds = dense_stations.read(spec, points_store)

    assert ds["t2m"].dims == ("time", "station")
    assert ds["t2m"].shape == (24, 3)
    assert sorted(str(s) for s in ds["station"].values) == sorted(STATIONS)
    assert set(ds.data_vars) == {"t2m", "rh"}


def test_station_coordinates_line_up_with_the_matrix_columns(points_store, registry):
    spec = station_spec(registry)
    dense_stations.build(spec, points_store)
    ds = dense_stations.read(spec, points_store)

    ids = [str(s) for s in ds["station"].values]
    for column, station in enumerate(ids):
        lat, lon, height = STATIONS[station]
        assert float(ds["lat"].values[column]) == lat
        assert float(ds["lon"].values[column]) == lon
        assert float(ds["height"].values[column]) == height
        assert str(ds["name"].values[column]) == f"station {station}"
        assert str(ds["source"].values[column]) == "knmi/in_situ/v1"
        # the column really does hold this station's series
        s_i = sorted(STATIONS).index(station)
        assert float(ds["t2m"].values[0, column]) == 100.0 * s_i


def test_dense_stations_keeps_gaps_as_nan_on_the_product_axis(points_store, registry):
    spec = station_spec(registry)
    ds = dense_stations.to_dataset(spec, points_store)
    column = [str(s) for s in ds["station"].values].index("06391")
    values = ds["t2m"].values[:, column]
    assert np.isnan(values[3])
    assert not np.isnan(np.delete(values, 3)).any()
    assert np.array_equal(ds["time"].values, spec.timeline.values())


def test_dense_stations_can_be_restricted_to_variables_and_stations(points_store, registry):
    spec = station_spec(registry)
    ds = dense_stations.to_dataset(
        spec, points_store, variables=["t2m"], station_ids=["00260", "00044"]
    )
    assert list(ds.data_vars) == ["t2m"]
    assert [str(s) for s in ds["station"].values] == ["00260", "00044"]
    assert float(ds["lat"].values[0]) == STATIONS["00260"][0]
    assert float(ds["t2m"].values[0, 0]) == 100.0


def test_dense_stations_records_its_provenance(points_store, registry):
    spec = station_spec(registry)
    dense_stations.build(spec, points_store)
    entry = manifest_mod.read(dense_stations.store_path(spec, points_store))
    assert entry is not None and entry.kind == "dense_stations"
    assert entry.extra["n_stations"] == 3
    assert entry.extra["variables"] == ["rh", "t2m"]


def test_available_variables_reads_the_schema_not_the_data(points_store, registry):
    assert dense_stations.available_variables("knmi/in_situ/v1", points_store) == ["t2m", "rh"]


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

def long_spec(registry: str, **kw) -> base.DeriveSpec:
    cfg = make_cfg(n_timesteps=4, shard_samples=8, **kw)
    timeline = Timeline(utc(2024, 1, 1), utc(2024, 4, 1), CADENCE)
    return base.DeriveSpec(cfg, timeline, grid_registry=registry)


def test_blocks_are_contiguous_and_cover_the_timeline(registry):
    spec = long_spec(registry)
    layout = splits.blocks(spec, block=timedelta(days=7))
    assert layout[0].start == spec.timeline.start
    assert layout[-1].end == spec.timeline.end
    for a, b in itertools.pairwise(layout):
        assert a.end == b.start                            # contiguous, no overlap
    assert {b.split for b in layout} <= {"train", "val", "test"}


def test_splits_never_overlap_in_time(registry):
    spec = long_spec(registry)
    frame = splits.assign(spec, block=timedelta(days=7))
    spans = {}
    for name in ("train", "val", "test"):
        part = frame.filter(pl.col("split") == name)
        assert len(part) > 0
        spans[name] = list(
            zip(part["start_time"].to_list(), part["end_time"].to_list(), strict=True)
        )
    for a in ("train", "val", "test"):
        for b in ("train", "val", "test"):
            if a >= b:
                continue
            for s0, e0 in spans[a]:
                for s1, e1 in spans[b]:
                    assert not (s0 < e1 and s1 < e0), f"{a} overlaps {b} at {s0}"


def test_every_sample_appears_exactly_once(registry):
    spec = long_spec(registry)
    frame = splits.assign(spec, block=timedelta(days=7))
    assert len(frame) == spec.n_samples
    assert sorted(frame["sample_index"].to_list()) == list(range(spec.n_samples))
    assert frame["slot"].to_list() == list(range(spec.n_samples))
    assert sum(splits.summary(frame).values()) == spec.n_samples


def test_splits_are_reproducible_and_round_trip(tmp_path, registry):
    spec = long_spec(registry)
    one = splits.assign(spec, block=timedelta(days=7))
    two = splits.assign(make_spec_like(spec, registry), block=timedelta(days=7))
    assert one.equals(two)

    path = splits.write(spec, tmp_path, block=timedelta(days=7))
    assert path.name == "splits.parquet"
    assert path.parent == bricks.store_path(spec, tmp_path)
    assert splits.read(spec, tmp_path).equals(one)


def make_spec_like(spec: base.DeriveSpec, registry: str) -> base.DeriveSpec:
    return base.DeriveSpec(spec.cfg.model_copy(deep=True), spec.timeline, grid_registry=registry)


def test_split_blocks_are_whole_days_not_single_samples(registry):
    """Each block holds many consecutive samples — that is what makes it a block."""
    spec = long_spec(registry)
    frame = splits.assign(spec, block=timedelta(days=7))
    per_block = frame.group_by("block").len()["len"].to_list()
    assert min(per_block) > 100


def test_explicit_intervals_from_the_config_win(registry):
    cfg_splits = {
        "train": ["2024-01-01/2024-02-01"],
        "val": ["2024-02-01/2024-02-15"],
        "test": ["2024-02-15/2024-03-01"],
    }
    spec = long_spec(registry, splits=cfg_splits)
    frame = splits.assign(spec)
    counts = splits.summary(frame)
    assert counts["train"] > counts["val"] > 0
    train_end = frame.filter(pl.col("split") == "train")["end_time"].max()
    val_start = frame.filter(pl.col("split") == "val")["start_time"].min()
    assert train_end <= val_start
    # March is in no interval, so those samples belong to no split.
    assert counts[splits.UNASSIGNED] > 0
    assert splits.slots(frame, "val") == sorted(splits.slots(frame, "val"))


def test_overlapping_configured_splits_are_refused(registry):
    spec = long_spec(
        registry,
        splits={"train": ["2024-01-01/2024-02-10"], "val": ["2024-02-01/2024-02-15"]},
    )
    with pytest.raises(ValueError, match="overlap"):
        splits.assign(spec)


def test_a_sample_straddling_a_split_boundary_is_embargoed(registry):
    """A block edge that falls mid-window must not leak the window into two splits."""
    cfg = make_cfg(n_timesteps=4)
    timeline = Timeline(utc(2024, 1, 1), utc(2024, 1, 3), CADENCE)
    spec = base.DeriveSpec(cfg, timeline, grid_registry=registry)
    # 10 minutes = two steps, so every second window straddles a block edge.
    frame = splits.assign(spec, block=timedelta(minutes=10), cycle=("train", "val"))
    counts = splits.summary(frame)
    assert counts[splits.EMBARGO] > 0
    for name in ("train", "val"):
        part = frame.filter(pl.col("split") == name)
        for start, end in zip(
            part["start_time"].to_list(), part["end_time"].to_list(), strict=True
        ):
            assert (end - start) == timedelta(minutes=20)
    assert set(counts) <= {"train", "val", splits.EMBARGO}
