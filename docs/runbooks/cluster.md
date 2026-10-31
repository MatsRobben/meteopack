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
sbatch slurm/meteopack.slurm <stage> <target> [cli args...]
sbatch --array=0-29 slurm/meteopack.slurm fetch dwd --window 2012/2013
```

The wrapper passes `--shard`/`--shards` from `SLURM_ARRAY_TASK_ID` so an array
job splits the unit list without any coordination between tasks.

## Conventions that matter

- **Scratch is `/dev/shm`, never `/tmp`.** `/tmp` is small and shared and has
  filled mid-job before; `/dev/shm` showed 126 GB free on cn79. The wrapper
  sets `TMPDIR` under `/dev/shm` and removes it on exit.
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
