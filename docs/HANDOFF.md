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
