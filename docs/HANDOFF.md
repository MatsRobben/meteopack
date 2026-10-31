# Handoff

Living notes for picking this up cold — or for me after a context reset.
Updated as work lands; newest state at the top of each section.

## Where things are

- **Repo**: `/vol/knmimo-nobackup/users/mrobben/meteopack`, pushed to
  `git@github.com:MatsRobben/meteopack.git` (public). Auth is a **deploy key
  scoped to this repo**: `~/.ssh/id_ed25519_meteopack`, found via
  `core.sshCommand` in `.git/config` — no `~/.ssh/config` entry. `gh` is NOT
  installed on this cluster.
- **Data root**: `/vol/knmimo-nobackup/users/mrobben/data` (setgid, group
  `knmimo`). Set `METEOPACK_DATA_ROOT` or pass `--data-root`.
- **Old repo** (reference only, do not modify):
  `/vol/knmimo-nobackup/users/mrobben/data_raw`, with ~294 GB of raw KNMI and
  ~18 GB of other raw station data under `data_raw/data/raw/`.
- **Env**: `uv sync --extra dev --extra radar --extra satellite`. Python 3.12.

## Cluster facts that cost time to learn

- **cn79 has no AVX2.** `duckdb` and `polars` fault at import there — it looks
  like a mysterious job failure, not a CPU mismatch. Every `slurm/*.slurm`
  carries `--exclude=cn79`. All other das/icis nodes are fine.
- **Scratch is `/dev/shm`, never `/tmp`.** `/tmp` is small, shared and has
  filled mid-job. 126 GB free on `/dev/shm` on cn104.
- Accounts `das` (4 nodes), `icis` (2, busier), `csedu`; use `*-preempt` since
  every stage is resumable from the ledger.

## Measured, not guessed (cn104)

- **Store geometry**, frame reads: 1-frame chunk in a 288-shard **51.3/s**;
  12-frame chunk **23.4/s**; packed 288-frame chunk **1.3/s**.
- **Write batching**: one region write per day **309 frames/s** vs frame-by-frame
  **14 frames/s**.
- **Pack parallelism**: serial 71.9 s (+1328 MiB RSS growth), threads(4) 36.5 s,
  threads(8) **42.0 s — slower than four, GIL contention**, processes(4) 16.6 s,
  processes(8) **12.4 s**. Pack therefore uses a spawn ProcessPool with
  `max_tasks_per_child` to recycle workers.
- Benchmarks live in `benchmarks/`; run with
  `sbatch benchmarks/slurm/bench.slurm bench_gridded_write.py bench_pack_parallel.py`.
  The harness records RSS growth **across repeats** because a leak that appears
  on the third pass survives a benchmark and kills a long backfill.

## Bugs found by running, not reading — keep these fixed

1. `grids.py` — `cached_property` cannot exist on a `slots=True` dataclass, and a
   broad `except` swallowed the `TypeError`, so the footprint test answered
   `False` for every point.
2. `core/http.py` — all three writers used a deterministic `.part` name, so two
   array tasks on one destination interleaved and published a mangled file.
3. `dwd.fetch` never passed the required `unit_key` to `fetch_file`; both suites
   stayed green because the test double had a looser signature.
4. **Volatile units never actually refetched**: the ledger re-queued them and
   `fetch_file` then skipped because the file existed. `fetch_file` now takes
   `overwrite`.
5. **Gridded work must split on shard boundaries.** A zarr shard is one file;
   concurrent writers *lose whole frames* (measured 7-9 of 12, every trial).
   The planner buckets by shard relative to `timeline_origin`.
6. `pack` never wrote station metadata, so stores validated clean with no
   coordinates at all and dense matrices came back all-NaN.
7. `registry.build` passed `session`/`limiter` to every source; local-archive
   sources do not accept them. It now filters kwargs by signature.

## Data facts worth not rediscovering

- **KNMI radar `calibration_missing_data` is 0 before 2023 and 65534 after**, and
  the parameter renames `PRECIP_[MM]` → `ACCUMULATION_[MM]`. Treating 65534 as
  "no echo" turns every missing pixel into 0 mm.
- A rare **uint8** radar frame declares `out_of_image = 65535`, unrepresentable
  in uint8; read literally ~104 000 masked pixels become 2.55 mm of rain. The
  clamp to the dtype max is corroborated three ways (statistics computed over the
  sentinel, a constant 255-count across a tar, and the old implementation doing
  the same).
- **A radar day runs 08:05 → 08:00**, not midnight to midnight.
- **E-SOH is a rolling 24 h window with no archive.** A month of it does not
  exist upstream; it must be harvested continuously.
- **CEDA token (`~/.ceda_token`) is expired** (July 2026; they last ~3 days), so
  MIDAS is fixture-tested only.

## Open / next

- Full-scale runs over the existing `data_raw` raw data (not satellite).
- Simplify and harden: the code is tuned for this cluster and may be fragile
  elsewhere (hard-coded archive roots in configs, `/dev/shm` assumptions).
- `derive` is wired for bricks only in the CLI; `dense_stations` and `splits`
  are API-only.

## Session 2026-10-31 (late): what running everything turned up

Everything below was found by running the pipeline, not by reading it. Each one
is committed with a regression test.

### Silent data loss, now fixed

1. **A gridded store's time axis never grew.** Packing four days of radar one
   day at a time kept 288 of 864 frames and printed "wrote 576 frames" each
   run, exit 0. Two faults compounded: the axis is fixed at `initialise` and
   frames past its end land at index -1, and the count reported was the number
   of frames handed to the writer rather than the number taken. Fixed with
   `gridded.extend` (forward only, rounded to a whole shard) and
   `gridded.WriteResult(written, outside, off_cadence)`; `pack` now fails on a
   non-zero `outside`. Verified: day-by-day == whole-window, element-wise, both
   variables, 0 differing cells.

2. **`pack --window` read the whole archive.** The window sized the time axis
   but never selected files. One day of SEVIRI decompressed and reprojected all
   576 scenes (25 GB) and timed out. Gridded sources now expose `span_of`;
   576 -> 291 files for one day. Points are deliberately *not* filtered: a
   points partition is replaced on write, so packing a subset of its files
   would drop the rest of the partition.

3. **`meteopack derive` had never run.** Wrong keyword (`shard=` vs `task=`),
   wrong type (DeriveConfig vs DeriveSpec), and bricks was called for every
   config so the points path was unreachable. The split index was never written
   either - a brick store with no train/val/test.

4. **meteofrance wrote an empty station table.** Its station registry is a side
   effect of `parse`, which now happens in worker processes, so the parent had
   nothing by the time it wrote metadata. The disk fallback existed but the CLI
   never passed `raw_root`. 0/29 -> 29/29.

5. **The scan drift check compared nothing for meteofrance.** dwd names the pin
   key `group`, meteofrance `token`. "0 of 3 pinned group schemas match" with
   exit 0 reads exactly like a pass. Both names understood; comparing nothing
   is now itself reported as drift.

### Verified working end to end

| what | evidence |
|---|---|
| points incremental download+append | belgium 2 -> 4 stations, 87,526 -> 175,060 rows, existing stations byte-identical, 0 duplicates |
| gridded incremental append | day-by-day == whole-window, 402,196,288 finite precip pixels either way |
| re-pack idempotence | re-packing a day changes nothing |
| dense_stations derive | 134,763 x 58, coordinates and names present, ranges plausible |
| bricks derive | 648 samples of 12x256x256, 87.7% finite, no empty sample |
| splits | 216/216/216 on day boundaries, no sample in two splits |

### Portability

Config values expand `${VAR}` and `${VAR:-default}`. The seven machine-specific
paths in configs and the one in `knmi_insitu.py` are now site settings; see
`.env.example`. An undefined variable is left as written and warned about,
never expanded to empty, because `${ARCHIVE}/knmi` becoming `/knmi` reads the
wrong directory. A test fails if an absolute `/vol`, `/home` or `/dev/shm` path
reappears in a config.

### The data root explains itself

`raw/`, `canonical/`, `derived/` and `catalog/` each carry a README written on
every run: what is in it, what wrote it, whether it can be rebuilt. Catalog
shard filenames lead with a timestamp so `ls` is chronological. Every product
config has a one-line `description` that reaches the store manifest, and
`meteopack catalog` lists the stores with it.

### Still open

- midas: the CEDA token is expired (`CedaAuthError`). Needs a new one.
- eumetsat_seviri: full-day pack (288 scenes) not yet confirmed green.
- The full runs over the existing `data_raw` archives (dwd 7.3G, dmi 2.4G,
  meteofrance 2.5G, ea 4.0G, midas 750M, belgium 1.0G, knmi radar, knmi
  in-situ) - explicitly not satellite.
- Source de-duplication: `_requests_session` x7, `station_id` x7,
  `StationRecord` x7, `unit_key` x6 still live in each source.
