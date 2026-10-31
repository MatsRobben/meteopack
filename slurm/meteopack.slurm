#!/bin/bash
#SBATCH --job-name=meteopack
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-preempt
#SBATCH --exclude=cn79          # Sandy Bridge, no AVX2: duckdb and polars segfault there
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#
# Thin wrapper. All logic lives in the CLI; this file only sets account,
# partition and environment.
#
#   sbatch slurm/meteopack.slurm <stage> <target> [extra cli args...]
#   sbatch --array=0-29 slurm/meteopack.slurm fetch dwd --window 2012/2013
#
# Every stage is re-runnable: the job ledger skips units already marked done,
# so preemption on das-preempt costs at most the in-flight unit.
set -euo pipefail

REPO="${METEOPACK_REPO:-/vol/knmimo-nobackup/users/mrobben/meteopack}"
cd "${REPO}"

# One thread per worker process; the pools do the parallelism, not BLAS.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
# /dev/shm, never /tmp: /tmp is small and shared, and filled up mid-job before.
export TMPDIR="${TMPDIR:-/dev/shm/$USER/meteopack-${SLURM_JOB_ID:-local}}"
mkdir -p "${TMPDIR}"
trap 'rm -rf "${TMPDIR}"' EXIT

WORKERS="${SLURM_CPUS_PER_TASK:-4}"
STAGE="${1:?usage: sbatch meteopack.slurm <stage> <target> [args...]}"
TARGET="${2:?usage: sbatch meteopack.slurm <stage> <target> [args...]}"
shift 2

echo "[meteopack] stage=${STAGE} target=${TARGET} host=$(hostname)" \
     "cpus=${WORKERS} task=${SLURM_ARRAY_TASK_ID:-none} $(date -u +%FT%TZ)"

ARRAY_ARGS=()
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  ARRAY_ARGS=(--shard "${SLURM_ARRAY_TASK_ID}" --shards "${SLURM_ARRAY_TASK_COUNT:-1}")
fi

t0=$SECONDS
.venv/bin/python -m meteopack.cli "${STAGE}" "${TARGET}" \
    --workers "${WORKERS}" "${ARRAY_ARGS[@]}" "$@"
echo "[meteopack] ${STAGE} finished in $((SECONDS - t0))s"
