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

## Conventions that matter

- **Scratch is `/dev/shm`, never `/tmp`.** `/tmp` is small and shared and has
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
