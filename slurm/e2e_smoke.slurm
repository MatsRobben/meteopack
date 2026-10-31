#!/bin/bash
#SBATCH --job-name=mp-e2e
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-preempt
#SBATCH --exclude=cn79          # Sandy Bridge, no AVX2: duckdb and polars segfault there
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/e2e-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/e2e-%j.err
set -euo pipefail
REPO="${METEOPACK_REPO:-/vol/knmimo-nobackup/users/mrobben/meteopack}"
cd "${REPO}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} $(date -u +%FT%TZ)"
.venv/bin/python -m pytest -q
.venv/bin/python -I scripts/e2e_smoke.py "/vol/knmimo-nobackup/users/mrobben/.meteopack_e2e_${SLURM_JOB_ID}"
