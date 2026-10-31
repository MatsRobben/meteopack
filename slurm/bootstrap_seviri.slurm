#!/bin/bash
#SBATCH --job-name=mp-seviri-dl
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-preempt
#SBATCH --exclude=cn79
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/seviri-dl-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/seviri-dl-%j.err
set -euo pipefail
cd /vol/knmimo-nobackup/users/mrobben/meteopack
echo "host=$(hostname) $(date -u +%FT%TZ)"
.venv/bin/python scripts/bootstrap_seviri.py \
  --start 2026-10-25 --end 2026-10-27 \
  --root /vol/knmimo-nobackup/users/mrobben/data/raw/eumetsat_seviri/msg15_rss \
  --workers 8
