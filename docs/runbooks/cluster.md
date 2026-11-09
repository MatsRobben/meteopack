# Running meteopack on the Science Cluster

Facts here were measured on the cluster in October-November 2026.

## Accounts and partitions

| Account | Partitions | QOS |
|---|---|---|
| `das` | `das`, `das-prio` | `das-{small,normal,large,preempt}` |
| `icis` | `icis`, `icis-prio` | `icis-{small,normal,large,preempt}` |
| `csedu`, `cseduproject` | `csedu`, `csedu-prio` | `csedu-{small,normal,large,preempt}` |

`das` has 4 nodes, `icis` 2 (usually busier).

**Which QOS.** `fetch` is restartable - the ledger skips units already done, so
a preemption costs at most the in-flight unit - and `*-preempt` suits it.
`pack` is not incremental: a preempted pack starts over, and a benchmark that
can be killed half-way measures nothing. Use `*-normal` for those.

| QOS | per-job limit | note |
|---|---|---|
| `das-normal` | 12 CPUs, 46 GB, 12 h | no shared pool, so it schedules |
| `icis-normal` | 48 CPUs, 125 GB, 12 h | |
| `das-large` | none per job, 2 days | a 74-CPU pool shared by everyone; often full (`QOSGrpCpuLimit`) |
| `icis-large` | none per job, 2 days | a 128-CPU shared pool |

(`sacctmgr show qos format=Name,MaxTRES,GrpTRES,MaxWall` shows the current values.)

## Sizing a job (measured 2026-11-01)

* **A Slurm CPU is a hyperthread.** `--cpus-per-task=48` is 24 physical cores;
  12 is 6. One worker per hardware thread is right: running twice as many
  workers as CPUs made a radar pack 1.6x *slower*.
* **Reads come from one NFS server, ~230 MiB/s in total.** One node alone gets
  ~165 MiB/s; three reading at once got ~77 each. Spreading an I/O-bound job
  over more nodes buys 25-40%, not a multiple. Above ~32 concurrent readers per
  node, throughput falls.
* **Per product:** rtcor is read-bound from ~24 workers (and its workers reach
  2.2 GB each); echotop and refl are CPU-bound and want every core, ~1 GB per
  worker. Full archives: rtcor ~21 min, echotop ~10 min, refl ~19 min at 48 CPUs.
* **Check the node before timing anything.** `scontrol show node <n> | grep
  CPULoad` - one node once had a load of 122 on 32 allocated CPUs, which made
  parsing 50% slower. cn84, where interactive sessions run, has a 1 GbE link
  and should not run packs at all.

## Submitting

```bash
# one pack, using every core it is given
sbatch --account=<yours> --exclude=cn79 --cpus-per-task=48 --mem=110G --time=02:00:00 \
  --wrap ".venv/bin/python -m meteopack.cli pack knmi_radar --product echotop_5min"

# a fetch split across an array: pass the task index through yourself
sbatch --account=<yours> --exclude=cn79 --array=0-29 --cpus-per-task=4 --time=04:00:00 \
  --wrap '.venv/bin/python -m meteopack.cli fetch dwd --window 2012/2013 \
          --shard $SLURM_ARRAY_TASK_ID --shards $SLURM_ARRAY_TASK_COUNT'
```

Passing `--shard`/`--shards` from the array variables splits the unit list
without any coordination between tasks (fetch, list and derive; `pack` refuses
them - give one pack job the cores instead). Each task takes `1/shards` of the
source's configured `requests_per_second`, so the array as a whole stays at the
configured rate; more tasks buy parallel checking and disk I/O, not a faster
upstream. For the station networks one job is usually enough (EA's ~14,400
station-years take about an hour at 5 requests/s; DWD's ~3,600 ZIPs minutes).

A fetch that is preempted (SIGTERM) writes its ledger before exiting; files
already on disk pass their check on the next run and are not downloaded again.
That needs the SIGTERM to reach Python. In an `sbatch` script that runs fetch
as a plain foreground command, bash dies on the signal first and Slurm then
SIGKILLs the rest of the step (seen 2026-11-06: the last 16 s of outcomes were
lost - harmless, the files were re-recorded on the next run, but avoidable).
Run it in the background and forward the signal:

```bash
.venv/bin/python -m meteopack.cli fetch ea --window 2012/2026 &
pid=$!; trap 'kill -TERM $pid' TERM
wait $pid; while kill -0 $pid 2>/dev/null; do wait $pid; done
```

## Streaming a product (`meteopack stream`)

`scripts/stream_kdp.slurm` runs `meteopack stream` for any streamed product
(KDP or S3) as a job that resubmits itself while days remain (exit 75), and
stops cleanly on Slurm's USR1 warning. Point `METEOPACK_REPO` at a checkout
nobody edits while it runs.

The OPERA rain rate, 2023-11-01 to the start of today (measured 2026-11-09 on
6 CPUs: ~7 s and ~10 MB a day, ~500 days an hour, 1.2 GB resident; the
download, ~120 MB a day at ~45 MB/s over 8 connections, overlaps the packing):

```bash
mkdir -p /path/to/logs
METEOPACK_DATA_ROOT=/vol/knmimo-nobackup/users/mrobben/meteopack_data \
METEOPACK_REPO=/path/to/pinned/checkout STREAM_LOG_DIR=/path/to/logs \
sbatch --output=/path/to/logs/%x-%j.out scripts/stream_kdp.slurm opera --product rate_15min
```

The script's own `#SBATCH` lines (das-large, 6 CPUs, 24 GB, 2 days) are what
a resubmitted successor gets too, so prefer them to command-line overrides.

Without `--window` it streams from `extra.stream_start` (2023-11-01) to the
start of today; rerunning later adds only the new days. Extend it back with
`--window 2013-01-01/2023-11-01`: the store's axis already starts in 2013.

NOAA GFS (`noaa_gfs/state_1h_0p25`, which also fills `prate_0p25`) needs a
venv synced with `uv sync --extra nwp` (ecCodes) - pass it as `METEOPACK_PY`.
Measured 2026-11-09 on 12 CPUs with 32 download workers: ~2.0 GB fetched a day
(byte ranges of ~2,900 GRIB messages) at 65-85 MB/s, ~33 s a day with the
next day's download overlapping the write, ~20 MB packed a day, 1.7 GB
resident:

```bash
METEOPACK_DATA_ROOT=/vol/knmimo-nobackup/users/mrobben/meteopack_data \
METEOPACK_REPO=/path/to/pinned/checkout METEOPACK_PY=/path/to/nwp/venv/bin/python \
STREAM_LOG_DIR=/path/to/logs \
sbatch --cpus-per-task=12 --output=/path/to/logs/%x-%j.out scripts/stream_kdp.slurm \
    noaa_gfs --product state_1h_0p25 --download-workers 32
```

## /dev/shm is wiped when any of your jobs on the node ends

Measured 2026-11-06 on cn114: about 10 s after **any** job of yours on a node
ends - an `srun true` will do - everything you own in that node's `/dev/shm`
is deleted, including the files of your jobs that are still running there. A
benchmark lost its output directory this way and a test run lost its
`TMPDIR` mid-suite. Python's multiprocessing keeps its named semaphores
there too: a pool worker spawned after a wipe dies at start
(`SemLock._rebuild: FileNotFoundError`) and the pool waits for it - a re-pack
hung this way on cn117 when another job ended. `run_pipeline` therefore starts
all its workers up front and never recycles them. So `/dev/shm` is only safe for a job that is the only one
of yours on its node for its whole run (no array siblings, no `srun` checks,
no other agent's jobs); otherwise use node-local `/scratch/$USER`, or keep the
data in memory. Free space on `/scratch` varies a lot: 18 GB on cn114, 2.6 GB
on cn117 that day.

## Conventions that matter

- **Scratch is `/dev/shm`, never `/tmp`** - but see the section above. `/tmp` is small and shared and has
  filled mid-job before; `/dev/shm` showed 126 GB free on cn79. Point `TMPDIR`
  under `/dev/shm` in your job and remove it on exit. (A points pack's spill is
  different: it goes to node-local `/scratch/$USER` by itself, never to RAM.)
- **`--cpus-per-task` must match the real thread count.** A previous pipeline
  requested 4 CPUs and ran 32 threads, which starved it.
- **Pin BLAS to one thread** (`OMP_NUM_THREADS=1` and friends). Parallelism
  comes from the worker pools, not from BLAS.
- **No heavy compute on the login node.** Submit it.

## Checking a job

```bash
squeue -u $USER
sacct -j <id> --format=JobID,State,Elapsed,MaxRSS,ExitCode
```

## cn79 has no AVX2 — exclude it

Verified 2026-10-31 by importing on every node in `das` and `icis`.

| Node | CPU | AVX2 | duckdb / polars |
|---|---|---|---|
| cn79 | Xeon E5-2670 (Sandy Bridge, 2012) | **no** | **both crash: "Illegal instruction"** |
| cn104, cn105 | Xeon Silver 4214 (Cascade Lake) | yes | OK |
| cn114, cn115 | EPYC 7642 (Zen 2) | yes | OK |
| cn117 | EPYC 7452 (Zen 2) | yes | OK |

The wheels for `duckdb` and `polars` are built with an AVX2 baseline, so they
fault on import on cn79 — a hard crash at import, long before any of our code
runs, and it looks like a mysterious job failure rather than a CPU mismatch.
Every job should therefore pass `--exclude=cn79`.

If cn79 is ever needed, the alternative is `polars-lts-cpu` plus a non-AVX2
duckdb build, which is more trouble than losing one of four `das` nodes.
