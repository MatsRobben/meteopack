#!/bin/bash
#SBATCH --job-name=mp-stream
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-large
#SBATCH --exclude=cn79,cn84     # cn79: no AVX2 (polars dies); cn84: 1 GbE
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G               # a day of 32 levels is 4.9 GB; measured peak 11.2 GB with prefetch
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:USR1@900     # 15 min warning: finish the current day, then stop
#
# `meteopack stream` as a self-resubmitting job: it runs until the window is
# done, the time budget is used, or Slurm warns it; on exit 75 ("more to do")
# it submits its own successor, which resumes from the per-day status records.
#
#   STREAM_LOG_DIR=/path/to/logs sbatch --output=/path/to/logs/%x-%j.out \
#       scripts/stream_kdp.slurm knmi_radar --product composite3d_5min [--window A/B]
#
# Any streamed product works the same way, e.g. the OPERA rain rate from the
# public S3 archive: `... scripts/stream_kdp.slurm opera --product rate_15min`.
#
# Environment:
#   METEOPACK_DATA_ROOT   the data root (required)
#   METEOPACK_REPO        checkout to run from (default: this repo). Point it at
#                         a pinned worktree: editing the code under a running
#                         job breaks it.
#   METEOPACK_PY          the Python to run (default: the main checkout's venv); NOAA GFS
#                         needs one synced with `--extra nwp` (ecCodes)
#   STREAM_LOG_DIR        where successors write their logs (required)
#   STREAM_BUDGET_HOURS   hours after which no new day is started (default 46)
#   STREAM_NO_RESUBMIT=1  do not chain
#
# A new key file (e.g. ~/.knmi_bulk_key) is picked up at the next day boundary
# without resubmitting; the per-key request rate is in the product config.
set -uo pipefail

REPO="${METEOPACK_REPO:-/vol/knmimo-nobackup/users/mrobben/meteopack}"
PY="${METEOPACK_PY:-/vol/knmimo-nobackup/users/mrobben/meteopack/.venv/bin/python}"
: "${METEOPACK_DATA_ROOT:?set METEOPACK_DATA_ROOT}"
: "${STREAM_LOG_DIR:?set STREAM_LOG_DIR}"
export METEOPACK_DATA_ROOT
export PYTHONPATH="${REPO}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/pyc-${SLURM_JOB_ID:-local}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${REPO}"

echo "[stream] job=${SLURM_JOB_ID:-local} host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?}" \
     "commit=$(git -C "${REPO}" rev-parse --short HEAD 2>/dev/null) args=$* $(date -u +%FT%TZ)"

# In the background with the signal forwarded: a foreground child never sees
# SIGTERM/USR1 from sbatch (docs/runbooks/cluster.md).
"${PY}" -m meteopack.cli stream "$@" --workers "${SLURM_CPUS_PER_TASK:-4}" \
    --time-budget "${STREAM_BUDGET_HOURS:-46}" &
pid=$!
trap 'kill -USR1 ${pid} 2>/dev/null' USR1 TERM
wait "${pid}"; rc=$?
while kill -0 "${pid}" 2>/dev/null; do wait "${pid}"; rc=$?; done
echo "[stream] exit ${rc} $(date -u +%FT%TZ)"

if [[ "${rc}" == 75 && -z "${STREAM_NO_RESUBMIT:-}" && -n "${SLURM_JOB_ID:-}" ]]; then
  script="${REPO}/scripts/stream_kdp.slurm"
  # The successor keeps this job's CPUs and memory (sbatch options are not inherited).
  next=$(cd "${STREAM_LOG_DIR}" && sbatch --parsable --dependency="afterany:${SLURM_JOB_ID}" \
         --cpus-per-task="${SLURM_CPUS_PER_TASK:-6}" --mem="${SLURM_MEM_PER_NODE:-24576}M" \
         --output="${STREAM_LOG_DIR}/%x-%j.out" "${script}" "$@")
  echo "[stream] more to do: submitted ${next}"
fi
exit "${rc}"
