#!/bin/bash
#SBATCH --job-name=mp-smoke
#SBATCH --account=das
#SBATCH --partition=das
#SBATCH --qos=das-preempt
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/smoke-%j.out
#SBATCH --error=/vol/knmimo-nobackup/users/mrobben/meteopack/logs/smoke-%j.err
set -euo pipefail
echo "host=$(hostname) cpus=${SLURM_CPUS_PER_TASK} date=$(date -u)"
echo "scratch: /dev/shm free = $(df -h /dev/shm | tail -1 | awk '{print $4}')"
cd /vol/knmimo-nobackup/users/mrobben/meteopack
PYTHONPATH=src /vol/knmimo-nobackup/users/mrobben/data_raw/.venv/bin/python -c "
import meteopack, sys
print('meteopack', meteopack.__version__, 'on', sys.version.split()[0])
from meteopack.sources.base import Source, Unit
print('contract import OK:', Source.__name__, Unit.__name__)
"
echo "SMOKE OK"
