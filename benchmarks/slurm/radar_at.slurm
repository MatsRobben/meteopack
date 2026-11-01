#!/bin/bash
# Run a radar benchmark command against a PINNED commit, on a compute node.
#
#   sbatch [--nodelist=cnXXX] benchmarks/slurm/radar_at.slurm COMMIT CMD [ARGS...]
#
# The job checks COMMIT out into its own worktree and puts that worktree's src
# first on PYTHONPATH, so editing the main checkout while a job runs cannot
# change the code under test. That has happened: pack workers re-import from
# disk when they respawn, and an edit mid-run gave one pack two versions of
# the code. CMD paths are relative to the worktree, so the benchmark scripts
# are pinned too.
#SBATCH --job-name=mp-radar
#SBATCH --account=das
#SBATCH --partition=das
# das-normal: at most 12 CPUs / 46G per job, but no group-wide pool. For more
# cores: --account=icis --partition=icis --qos=icis-normal (48 CPUs, 125G).
#SBATCH --qos=das-normal
#SBATCH --exclude=cn79
#SBATCH --cpus-per-task=12
#SBATCH --mem=46G
#SBATCH --time=01:30:00
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/radar-%j.out
set -euo pipefail
repo=/vol/knmimo-nobackup/users/mrobben/meteopack
commit=$(git -C "$repo" rev-parse --short "$1"); shift
tree=/vol/knmimo-nobackup/users/mrobben/bench_radar/trees/$commit
if [ ! -d "$tree" ]; then
  git -C "$repo" worktree add --detach "$tree" "$commit" >/dev/null 2>&1 \
    || [ -d "$tree" ]            # another job made it first
fi
cd "$tree"
export PYTHONPATH="$tree/src"
K=/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi
export KNMI_RADAR_ARCHIVE=$K/nl_rdr_data_rtcor_5m_tar
export KNMI_ECHOTOP_ARCHIVE=$K/radar_tar_echotopheight_5min
export KNMI_REFL_ARCHIVE=$K/radar_tar_refl_composites
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH="$repo/.venv/bin:$PATH"
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} job=${SLURM_JOB_ID:-?} $(date -u +%FT%TZ)"
echo "commit=$commit tree=$tree"
python -c "import meteopack, sys; assert meteopack.__file__.startswith('$tree'), meteopack.__file__; print('importing', meteopack.__file__)"
echo "cmd: $*"
case "$1" in
  *.py) python "$@" ;;
  *)    bash "$@" ;;
esac
