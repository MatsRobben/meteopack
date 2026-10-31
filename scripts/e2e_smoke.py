"""First end-to-end slice: list -> fetch -> parse -> pack -> read -> matrix.

Real network, deliberately tiny: only `now` archives (~1 KB each) for a couple
of stations. Proves the three independently-built modules actually compose.
"""
import shutil, sys, time, yaml
from datetime import timedelta, timezone, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import requests
from meteopack.core.config import ProductConfig, SourceConfig
from meteopack.core.catalog import Catalog, job_key
from meteopack.core.http import RateLimiter
from meteopack.core.store import points, manifest
from meteopack.core.timeline import Timeline
from meteopack.sources.base import TimeWindow
from meteopack.sources.dwd import DwdStations

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/vol/knmimo-nobackup/users/mrobben/.meteopack_e2e")
MAX_FILES = 3

shutil.rmtree(ROOT, ignore_errors=True)
scfg = SourceConfig(**yaml.safe_load((REPO / "configs/sources/dwd.yaml").read_text()))
pcfg = ProductConfig(**yaml.safe_load((REPO / "configs/products/dwd/stations_10min.yaml").read_text()))
print(f"configs loaded: source={scfg.name} product={pcfg.store_id} rps={scfg.rate_limit.requests_per_second}")

src = DwdStations(scfg, session=requests.Session(),
                  limiter=RateLimiter.from_config(scfg.rate_limit))
now = datetime.now(timezone.utc)
window = TimeWindow(now - timedelta(days=1), now)

t0 = time.time()
units = src.list(window, pcfg)
print(f"list()   : {len(units)} units in {time.time()-t0:.1f}s")
units = [u for u in units if (u.meta or {}).get("group") == "air_temperature"][:MAX_FILES]
print(f"           sampling {len(units)}: {[u.meta['station_id'] for u in units]}")

cat = Catalog(ROOT)
raw_root = ROOT / "raw" / "dwd" / "stations_10min"
frames, fetched = [], 0
for u in units:
    rf = src.fetch(u, raw_root)
    if rf is None:
        print(f"  skip (404) {u.key}"); continue
    fetched += 1
    cat.record_file(rf, source="dwd", product="stations_10min", url=u.url)
    cat.mark(job_key("fetch", u.key), "done")
    df = src.parse(rf, pcfg)
    frames.append(df)
    print(f"  {u.meta['station_id']}: {rf.n_bytes:>6} B sha={rf.sha256[:8]} -> {df.height} rows x {df.width} cols")

import polars as pl
obs = pl.concat(frames, how="diagonal_relaxed")
print(f"parse()  : {obs.height} rows, cols={obs.columns}")

paths = points.write(obs, pcfg, ROOT)
manifest.write(points.store_path(ROOT, pcfg.store_id),
               manifest.Manifest.new(pcfg.store_id, "canonical", "points", pcfg, n_records=obs.height))
cat.record_store(store_id=pcfg.store_id, layer="canonical", kind="points",
                 path=str(points.store_path(ROOT, pcfg.store_id)), version=pcfg.version,
                 n_records=obs.height, config_hash=manifest.config_hash(pcfg))
cat.flush()
print(f"pack()   : {len(paths)} partition file(s)")

back = points.read(pcfg.store_id, ROOT)
print(f"read()   : {back.height} rows back (match={back.height == obs.height})")

tmin = obs["time"].min().replace(tzinfo=timezone.utc)
tl = Timeline(tmin, tmin + timedelta(hours=2), pcfg.cadence)
vals, ids = points.to_matrix(pcfg.store_id, ROOT, "air_temperature", tl)
import numpy as np
print(f"matrix() : shape={vals.shape} stations={ids} filled={int(np.isfinite(vals).sum())}/{vals.size}")
print(f"           temps seen: {np.nanmin(vals):.1f} .. {np.nanmax(vals):.1f} degC")
print(f"catalog  : files={len(cat.files())} stores={len(cat.stores())} jobs={len(cat.jobs())}")
print(f"stale?   : {manifest.is_stale(points.store_path(ROOT, pcfg.store_id), pcfg)} (expect False)")
print(f"\nE2E OK in {time.time()-t0:.1f}s  (fetched {fetched} files)")
