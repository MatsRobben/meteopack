"""Cross-module contract checks.

Unit tests monkeypatch their collaborators, which is what makes them fast and
offline — and also what lets a signature drift between two modules without
either suite noticing. These tests bind against the REAL callables, so an
incompatible call site fails here instead of on the first live download.
"""
from __future__ import annotations

import inspect
from pathlib import Path

from meteopack.core import http
from meteopack.sources import dwd
from meteopack.sources.base import Source, Unit


def test_dwd_fetch_call_matches_real_fetch_file_signature():
    """dwd.fetch must be able to call the real http.fetch_file.

    This broke once: fetch_file took a keyword-only `unit_key` that dwd.fetch
    never passed, so every real download would have raised TypeError while both
    test suites stayed green.
    """
    sig = inspect.signature(http.fetch_file)
    sig.bind(
        "https://example.invalid/x.zip",
        Path("/tmp/x.zip"),
        unit_key="k",
        session=object(),
        limiter=None,
        retries=5,
        backoff=1.0,
        timeout=60.0,
    )


def test_dwd_fetch_passes_every_required_argument(monkeypatch, tmp_path):
    """Call dwd.fetch for real, with a stand-in that enforces the true signature."""
    real_sig = inspect.signature(http.fetch_file)
    seen = {}

    def strict_fetch_file(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)   # raises if dwd omits anything
        bound.apply_defaults()
        seen.update(bound.arguments)
        from meteopack.sources.base import RawFile
        return RawFile(Path(kwargs["dest"] if "dest" in kwargs else args[1]), "0" * 64, 0,
                       kwargs["unit_key"])

    monkeypatch.setattr(http, "fetch_file", strict_fetch_file)
    src = dwd.DwdStations()
    unit = Unit(
        key="dwd/stations_10min/air_temperature/now/x",
        url="https://example.invalid/x.zip",
        meta={"group": "air_temperature", "subdir": "now", "filename": "x.zip"},
    )
    src.fetch(unit, tmp_path)
    assert seen["unit_key"] == unit.key


def test_dwd_satisfies_the_source_protocol():
    assert isinstance(dwd.DwdStations(), Source)


def test_gridded_round_trip_through_the_real_grid_registry(tmp_path):
    """Exercise initialise/write/read against configs/grids.yaml, not a double.

    The unit tests monkeypatch the registry to keep frames small, which means
    nothing otherwise proves a config's `grid_id` actually resolves and the
    store picks up the grid's shape.
    """
    from datetime import timedelta

    import numpy as np
    import xarray as xr

    from meteopack.core import grids
    from meteopack.core.config import ProductConfig, StoreConfig, Variable
    from meteopack.core.store import gridded
    from meteopack.core.timeline import Timeline, utc

    grid = grids.get("example_tiny")
    ny, nx = grid.shape
    cfg = ProductConfig(
        source="example", product="field", cadence_seconds=600, grid_id="example_tiny",
        variables=[Variable(name="rain", source_key="r", units="mm")],
        store=StoreConfig(kind="gridded", chunks={"time": 1, "y": ny, "x": nx},
                          shard={"time": 6, "y": ny, "x": nx}),
    )
    tl = Timeline(utc(2012, 1, 1), utc(2012, 1, 1, 1), timedelta(minutes=10))
    gridded.initialise(cfg, tmp_path, tl)

    frame = xr.Dataset(
        {"rain": (("time", "y", "x"), np.full((1, ny, nx), 2.5, dtype="float32"))},
        coords={"time": np.array([tl.values()[3]])},
    )
    gridded.write(frame, cfg, tmp_path)

    ds = gridded.read(cfg.store_id, tmp_path)
    assert ds.sizes["time"] == len(tl) and ds.sizes["y"] == ny and ds.sizes["x"] == nx
    assert ds.attrs["grid_id"] == "example_tiny"
    values = ds["rain"].values
    assert np.allclose(values[3], 2.5)
    assert np.isnan(values[0]).all(), "frames nobody wrote read back as NaN"
