#!/bin/bash
#SBATCH --job-name=mp-bench
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-preempt
#SBATCH --exclude=cn79
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/bench-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/bench-%j.err
set -euo pipefail
cd /vol/knmimo-nobackup/users/mrobben/meteopack
# Scratch on /dev/shm, never /tmp: extraction filled /tmp and crashed a job before.
export TMPDIR="/dev/shm/$USER/mpbench-${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"; trap 'rm -rf "$TMPDIR"' EXIT
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} TMPDIR=$TMPDIR"
for b in "$@"; do .venv/bin/python "benchmarks/$b"; done
