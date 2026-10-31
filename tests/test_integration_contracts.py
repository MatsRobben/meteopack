"""Cross-module contract checks.

Unit tests monkeypatch their collaborators, which is what makes them fast and
offline — and also what lets a signature drift between two modules without
either suite noticing. These tests bind against the REAL callables, so an
incompatible call site fails here instead of on the first live download.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import ClassVar

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


def test_sources_constructed_without_a_session_can_still_fetch(monkeypatch, tmp_path):
    """A source built with no session must make one, not pass None downstream.

    `DwdStations()` stored `session=None` and handed it straight to fetch_file,
    which then did `None.get(...)`. The CLI always supplies a session, so this
    only broke scripts and notebooks — the paths least likely to be tested.
    """
    import inspect

    from meteopack.core import http
    from meteopack.sources.base import RawFile
    from meteopack.sources.dwd import DwdStations
    from meteopack.sources.esoh import EsohObservations

    real_sig = inspect.signature(http.fetch_file)
    captured = {}

    def strict(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        captured["session"] = bound.arguments["session"]
        return RawFile(Path(bound.arguments["dest"]), "0" * 64, 0, bound.arguments["unit_key"])

    monkeypatch.setattr(http, "fetch_file", strict)

    unit = Unit(key="k", url="https://example.invalid/x.zip",
                meta={"group": "air_temperature", "subdir": "now", "filename": "x.zip"})
    DwdStations().fetch(unit, tmp_path)
    assert captured["session"] is not None, "DwdStations passed a None session"

    captured.clear()
    esoh_unit = Unit(key="k", url="https://example.invalid/x.json", meta={})
    EsohObservations().fetch(esoh_unit, tmp_path)
    assert captured["session"] is not None, "EsohObservations passed a None session"


def test_volatile_units_are_actually_refetched(monkeypatch, tmp_path):
    """A volatile unit must hit the network again, not just the ledger.

    The ledger re-queues volatile units, but fetch_file skips when the file is
    already there — so the ledger refreshed and the bytes never did, and the
    live tail would freeze while every status looked healthy. Measured before
    the fix: zero HTTP requests on the second fetch of a volatile unit.
    """
    import inspect

    from meteopack.core import http
    from meteopack.sources.base import RawFile
    from meteopack.sources.dwd import DwdStations

    real_sig = inspect.signature(http.fetch_file)
    seen: list[bool] = []

    def strict(*args, **kwargs):
        bound = real_sig.bind(*args, **kwargs)
        bound.apply_defaults()
        seen.append(bool(bound.arguments["overwrite"]))
        dest = Path(bound.arguments["dest"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"x")
        return RawFile(dest, "0" * 64, 1, bound.arguments["unit_key"])

    monkeypatch.setattr(http, "fetch_file", strict)
    src = DwdStations()
    common = {"group": "air_temperature", "subdir": "now", "filename": "x.zip"}

    src.fetch(Unit(key="v", url="https://example.invalid/x.zip",
                   meta={**common, "volatile": True}), tmp_path)
    src.fetch(Unit(key="i", url="https://example.invalid/x.zip", meta=common), tmp_path)

    assert seen == [True, False], "volatile must request overwrite, immutable must not"


def test_fetch_file_honours_overwrite(tmp_path):
    """The mechanism itself: an existing destination is replaced only on request."""
    from meteopack.core import http

    dest = tmp_path / "a.bin"
    dest.write_bytes(b"old")

    class FakeSession:
        def __init__(self):
            self.calls = 0

        def get(self, url, **kw):
            self.calls += 1

            class R:
                status_code = 200
                headers: ClassVar[dict] = {}

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *a):
                    return False

                def iter_content(self_inner, chunk_size=0):
                    yield b"new"

                def raise_for_status(self_inner):
                    pass

            return R()

    s = FakeSession()
    http.fetch_file("https://example.invalid/a.bin", dest, unit_key="k", session=s)
    assert dest.read_bytes() == b"old" and s.calls == 0, "default must skip"

    http.fetch_file("https://example.invalid/a.bin", dest, unit_key="k", session=s, overwrite=True)
    assert dest.read_bytes() == b"new" and s.calls == 1, "overwrite must refetch"


def test_schema_drift_is_one_exception_across_every_source():
    """`except SchemaDriftError` must catch drift from any source.

    Each source originally defined its own class, so catching one caught
    exactly one and silently missed the other nine — a pipeline handling drift
    uniformly would have let nine sources write misaligned stores.
    """
    import importlib

    from meteopack.sources.common import SchemaDriftError, SourceError

    names = ["dwd", "esoh", "belgium", "dmi", "ea", "knmi_insitu",
             "knmi_radar", "meteofrance", "midas", "eumetsat_seviri"]
    for name in names:
        mod = importlib.import_module(f"meteopack.sources.{name}")
        err = getattr(mod, "SchemaDriftError", None)
        assert err is SchemaDriftError, f"{name} defines its own SchemaDriftError"
    assert issubclass(SchemaDriftError, SourceError)


def test_fingerprint_is_order_and_whitespace_independent():
    from meteopack.sources.common import fingerprint

    assert fingerprint(["b", "a"]) == fingerprint([" a ", "b"])
    assert fingerprint(["a"]) != fingerprint(["a", "c"])
    assert fingerprint([]) == fingerprint(["", "  "])


def test_cli_gives_the_source_its_raw_root():
    """`pack` parses in worker processes, so anything the parent source object
    accumulated during parse is gone before it writes station metadata. The
    archive on disk is the only state both sides share, so a source that can
    recover from it must actually be told where it is — meteofrance silently
    wrote an empty station table until it was.
    """
    import inspect

    from meteopack.cli import Context

    assert "raw_root=self.raw_root" in inspect.getsource(Context.src.fget)


def test_scan_group_key_is_read_under_either_name():
    """dwd calls the pin key `group`, meteofrance `token`. Reading only one
    compared nothing for the other while printing a clean bill of health.
    """
    from types import SimpleNamespace

    from meteopack.cli import _compare_schema, _variant_group

    assert _variant_group({"group": "wind"}) == "wind"
    assert _variant_group({"token": "latest-2025-2026"}) == "latest-2025-2026"
    assert _variant_group({"fingerprint": "abc"}) is None

    cfg = SimpleNamespace(
        extra={"schema_fingerprints": {"a": "fp1", "b": "fp2"}},
        schema_fingerprint=None,
    )

    ok, _ = _compare_schema(cfg, {"variants": [{"token": "a", "fingerprint": "fp1"}]})
    assert ok == []

    drift, _ = _compare_schema(cfg, {"variants": [{"token": "a", "fingerprint": "other"}]})
    assert drift and "fp1" in drift[0]

    # A variant with no group key compared nothing at all; that must be loud.
    blind, _ = _compare_schema(cfg, {"variants": [{"fingerprint": "fp1"}]})
    assert blind and "no group key" in blind[0]
