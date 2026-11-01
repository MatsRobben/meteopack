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


def test_editing_a_description_does_not_make_a_store_stale(tmp_path):
    """validate said "built from a different config; rebuild" after a prose edit."""
    from meteopack.core.config import ProductConfig, StoreConfig, Variable
    from meteopack.core.store import manifest as m

    def cfg(**kw):
        base = {"source": "s", "product": "p", "store": StoreConfig(kind="points"),
                "variables": [Variable(name="t", source_key="T", units="K", long_name="old")]}
        base.update(kw)
        return ProductConfig(**base)

    m.write(tmp_path, m.Manifest.new("s/p/v1", "packed", "points", cfg()))
    assert not m.is_stale(tmp_path, cfg(description="new words", attrs={"title": "x"}))
    relabelled = cfg()
    relabelled.variables[0].long_name = "new"
    assert not m.is_stale(tmp_path, relabelled)
    assert m.is_stale(tmp_path, cfg(cadence_seconds=300))          # content: still stale


def test_a_manifest_from_before_content_hash_is_judged_by_the_full_hash(tmp_path):
    import json

    from meteopack.core.config import ProductConfig, StoreConfig
    from meteopack.core.store import manifest as m

    c = ProductConfig(source="s", product="p", store=StoreConfig(kind="points"))
    path = m.write(tmp_path, m.Manifest.new("s/p/v1", "packed", "points", c))
    old = json.loads(path.read_text())
    old.pop("content_hash")
    path.write_text(json.dumps(old))
    assert not m.is_stale(tmp_path, c)
