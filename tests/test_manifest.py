from __future__ import annotations

from meteopack.core.config import ProductConfig, StoreConfig
from meteopack.core.store import manifest as mf


def _cfg(**kw) -> ProductConfig:
    return ProductConfig(source="dwd", product="stations_10min",
                         store=StoreConfig(kind="points"), **kw)


def test_hash_is_stable_and_sensitive(tmp_path):
    a, b = _cfg(), _cfg()
    assert mf.config_hash(a) == mf.config_hash(b)
    assert mf.config_hash(a) != mf.config_hash(_cfg(cadence_seconds=300))


def test_roundtrip_and_staleness(tmp_path):
    cfg = _cfg()
    store = tmp_path / "store.parquet"
    mf.write(store, mf.Manifest.new(cfg.store_id, "packed", "points", cfg, n_records=7))
    back = mf.read(store)
    assert back is not None and back.n_records == 7 and back.store_id == "dwd/stations_10min/v1"
    assert not mf.is_stale(store, cfg)
    assert mf.is_stale(store, _cfg(cadence_seconds=300))
    assert mf.is_stale(tmp_path / "nothing", cfg)


def test_write_leaves_no_part_file(tmp_path):
    cfg = _cfg()
    store = tmp_path / "s"
    mf.write(store, mf.Manifest.new(cfg.store_id, "packed", "points", cfg))
    assert not list(store.glob("*.part"))
