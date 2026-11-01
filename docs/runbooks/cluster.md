# Running meteopack on the Science Cluster

Verified 2026-10-31 (job 12303351 completed on cn79).

## Accounts and partitions

| Account | Partitions | QOS |
|---|---|---|
| `das` | `das`, `das-prio` | `das-{small,normal,large,preempt}` |
| `icis` | `icis`, `icis-prio` | `icis-{small,normal,large,preempt}` |
| `csedu`, `cseduproject` | `csedu`, `csedu-prio` | `csedu-{small,normal,large,preempt}` |

`das` has 4 nodes, `icis` 2 (usually busier). Use `*-preempt` for anything
restartable — every meteopack stage is, because the job ledger skips units
already marked done, so a preemption costs at most the in-flight unit.

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
them - give one pack job the cores instead).

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
