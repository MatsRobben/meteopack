#!/bin/bash
# Rebuild KNMI radar products from their full archives, then check them.
#
#   METEOPACK_DATA_ROOT=/path/to/root scripts/radar_rebuild.sh PRODUCT [PRODUCT ...]
#
# Run it through Slurm, not on an interactive node - for example with
# benchmarks/slurm/radar_at.slurm, which also pins the commit:
#
#   sbatch --account=icis --partition=icis --qos=icis-normal --cpus-per-task=48 \
#     --mem=120G --time=02:00:00 benchmarks/slurm/radar_at.slurm <commit> \
#     scripts/radar_rebuild.sh echotop_5min refl_5min
#
# For each product: a full pack with no --window (the axis comes from the
# archive's own filenames), `validate`, and an independent per-pixel check of
# sampled frames against the raw HDF5. It refuses to write over an existing
# store - move the old one aside first.
set -uo pipefail
root=${METEOPACK_DATA_ROOT:?set METEOPACK_DATA_ROOT}
logs=${REBUILD_LOGS:-$(dirname "$0")/../logs}
mkdir -p "$logs"
status=0
for p in "$@"; do
  case $p in
    rtcor_5min)   arch=${KNMI_RADAR_ARCHIVE:?} ;;
    echotop_5min) arch=${KNMI_ECHOTOP_ARCHIVE:?} ;;
    refl_5min)    arch=${KNMI_REFL_ARCHIVE:?} ;;
    *) echo "unknown product $p"; exit 2 ;;
  esac
  store=$root/packed/knmi_radar/$p
  echo "################ $p  ($(ls "$arch"/*.tar | wc -l) tars, $(du -shL "$arch" | cut -f1)) on $(hostname)"
  if [ -e "$store" ]; then echo "refusing: $store exists; move it aside first"; exit 1; fi
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  log=$logs/rebuild_${p}_$stamp.log
  export METEOPACK_TRACE=$logs/rebuild_${p}_$stamp.trace.json
  /usr/bin/time -f "  PACK %e s wall" python -m meteopack.cli pack knmi_radar --product "$p" \
    > "$log" 2>&1 || status=1
  grep -vE "no image groups|unreadable frame|^INFO" "$log" | tail -12
  python -m meteopack.cli validate knmi_radar --product "$p" > "$log.validate" 2>&1 || status=1
  grep -E "OK$|FAILED$|FAIL |pass |stat " "$log.validate" | head -14
  echo "  -- independent per-pixel check against the raw HDF5 --"
  KNMI_RADAR_ARCHIVE=$arch STORE=$store/v1/dataset.zarr \
    python benchmarks/verify_against_raw.py --frames 40 2>&1 | tail -2
  echo
done
exit $status
