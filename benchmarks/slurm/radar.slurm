#!/bin/bash
# Run one radar benchmark command on a compute node.
#
#   sbatch [--cpus-per-task=N] [--nodelist=cnXXX] benchmarks/slurm/radar.slurm \
#       benchmarks/radar_profile.py --root "$BENCH/sample" --product echotop_5min
#   sbatch ... benchmarks/slurm/radar.slurm benchmarks/radar_pack_timed.sh \
#       echotop_5min 16 "$BENCH/runs/ref"
#
# Never as a background shell on the interactive node: it took the user's VS
# Code node to load 16 for hours, and nothing bounded a stalled run.
#SBATCH --job-name=mp-radar
#SBATCH --account=das
#SBATCH --partition=das
# das-normal: at most 12 CPUs / 46G per job, but no group-wide pool, so it
# schedules when das-large's shared 74-CPU pool is full. For more cores use
# --account=icis --partition=icis --qos=icis-normal (48 CPUs, 125G). Never
# preempt: a benchmark that can be killed half-way measures nothing.
#SBATCH --qos=das-normal
#SBATCH --exclude=cn79
#SBATCH --cpus-per-task=12
#SBATCH --mem=46G
#SBATCH --time=01:30:00
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/radar-%j.out
set -euo pipefail
cd /vol/knmimo-nobackup/users/mrobben/meteopack
K=/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi
export KNMI_RADAR_ARCHIVE=$K/nl_rdr_data_rtcor_5m_tar
export KNMI_ECHOTOP_ARCHIVE=$K/radar_tar_echotopheight_5min
export KNMI_REFL_ARCHIVE=$K/radar_tar_refl_composites
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$PWD/.venv/bin:$PATH"
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-?} $(date -u +%FT%TZ)"
echo "commit=$(git rev-parse --short HEAD)$(git diff --quiet || echo +dirty)"
echo "cmd: $*"
case "$1" in
  *.py) python "$@" ;;
  *)    bash "$@" ;;
esac
