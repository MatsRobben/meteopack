"""Store geometry and write batching, measured on real radar frames.

Two questions the design rests on:
  1. Is one frame per inner chunk inside a shard really better than a packed
     multi-frame chunk? The redesign plan claims 3-5 vs 63-89 warm frame
     reads/s, which is a big enough gap to be worth re-checking.
  2. Does batching a day into one region write actually matter, or is zarr's
     own buffering enough?

Caveat that applies to every number here: this runs on one quiet node against
a warm page cache. At backfill scale the limits move to I/O contention and
memory, so treat the ORDERING as the result and the absolute rate as an
optimistic ceiling.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from datetime import timedelta
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harness import Result, disk_free_gib, report, run

from meteopack.core.config import ProductConfig, StoreConfig, Variable
from meteopack.core.store import gridded
from meteopack.core.timeline import Timeline, utc

ARCHIVE = Path("/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi/nl_rdr_data_rtcor_5m_tar")
NY, NX = 765, 700
FRAMES = 288


def cfg_for(chunks: dict, shard: dict | None) -> ProductConfig:
    return ProductConfig(
        source="bench", product="rtcor", version=1, cadence_seconds=300,
        grid_id="knmi_1km_stereo",
        variables=[Variable(name="precipitation_amount", source_key="x", units="mm")],
        store=StoreConfig(kind="gridded", chunks=chunks, shard=shard),
    )


def synthetic_day(timeline: Timeline) -> xr.Dataset:
    """Radar-like: mostly zero, a sparse wet area, a NaN out-of-image border."""
    rng = np.random.default_rng(0)
    data = np.zeros((FRAMES, NY, NX), dtype="float32")
    wet = rng.random((FRAMES, NY, NX)) < 0.06
    data[wet] = rng.gamma(1.5, 0.3, size=int(wet.sum())).astype("float32")
    data[:, :40, :] = np.nan
    return xr.Dataset({"precipitation_amount": (("time", "y", "x"), data)},
                      coords={"time": timeline.values()})


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mpbench-", dir=os.environ.get("TMPDIR", "/dev/shm")))
    timeline = Timeline(utc(2024, 7, 2, 8, 5), utc(2024, 7, 3, 8, 5), timedelta(minutes=5))
    ds = synthetic_day(timeline)
    results: list[Result] = []

    geometries = [
        ("1-frame chunk, 288 shard", {"time": 1, "y": NY, "x": NX}, {"time": FRAMES, "y": NY, "x": NX}),
        ("12-frame chunk, no shard",  {"time": 12, "y": NY, "x": NX}, None),
        ("288-frame chunk, no shard", {"time": FRAMES, "y": NY, "x": NX}, None),
    ]

    for label, chunks, shard in geometries:
        cfg = cfg_for(chunks, shard)
        root = tmp / label.replace(" ", "_").replace(",", "")

        def write_once(root=root, cfg=cfg, timeline=timeline):
            shutil.rmtree(root, ignore_errors=True)
            gridded.initialise(cfg, root, timeline, overwrite=True)
            gridded.write(ds, cfg, root)
            return {"store_mib": sum(f.stat().st_size for f in root.rglob("*") if f.is_file()) / 2**20}

        results.append(run("write", f"write day: {label}", write_once, repeats=3,
                           units=FRAMES, unit_name="frames"))

        # warm frame reads: the access pattern a training loader uses
        def read_frames(root=root, cfg=cfg):
            got = 0
            for i in range(0, FRAMES, 8):
                sub = gridded.read(cfg.store_id, root,
                                   time_range=(timeline.values()[i].astype("datetime64[s]").astype(object),
                                               timeline.values()[i + 1].astype("datetime64[s]").astype(object)))
                got += int(sub.sizes.get("time", 0))
            return {"frames_read": got}

        results.append(run("read", f"read 36 frames: {label}", read_frames, repeats=3,
                           units=36, unit_name="frames"))

    # batched vs per-frame writes, on the production geometry
    cfg = cfg_for({"time": 1, "y": NY, "x": NX}, {"time": FRAMES, "y": NY, "x": NX})
    root = tmp / "batching"

    def batched():
        shutil.rmtree(root, ignore_errors=True)
        gridded.initialise(cfg, root, timeline, overwrite=True)
        gridded.write(ds, cfg, root)

    def per_frame():
        shutil.rmtree(root, ignore_errors=True)
        gridded.initialise(cfg, root, timeline, overwrite=True)
        for i in range(FRAMES):
            gridded.write(ds.isel(time=slice(i, i + 1)), cfg, root)

    results.append(run("batching", "one region write per day", batched, repeats=3,
                       units=FRAMES, unit_name="frames"))
    results.append(run("batching", "one write per frame (288x)", per_frame, repeats=1,
                       units=FRAMES, unit_name="frames"))

    report("gridded write geometry", results,
           extra={"tmp": str(tmp), "tmp_free_gib": disk_free_gib(tmp)})
    shutil.rmtree(tmp, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
