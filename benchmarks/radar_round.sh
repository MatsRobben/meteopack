#!/bin/bash
# One round of the radar optimisation loop: cold-pack every product of the
# sample with the current code, time it, and compare against the reference.
#
#   benchmarks/radar_round.sh NAME WORKERS [PRODUCTS]
#
# NAME=ref writes the reference. Any other NAME is a candidate, and the round
# fails if its store differs from the reference in any way compare_stores.py
# checks - a faster pack that writes something else is not an optimisation.
set -uo pipefail
name=$1 workers=$2
products=${3:-rtcor_5min echotop_5min refl_5min}
bench=/vol/knmimo-nobackup/users/mrobben/bench_radar
here=$(dirname "$0")
fail=0
for p in $products; do
  out="$bench/runs/${name}_$p"
  echo "################ $name  $p  workers=$workers"
  bash "$here/radar_pack_timed.sh" "$p" "$workers" "$out" "$bench/sample" || fail=1
  ref="$bench/runs/ref_$p/packed/knmi_radar/$p/v1/dataset.zarr"
  cand="$out/packed/knmi_radar/$p/v1/dataset.zarr"
  if [ "$name" != ref ] && [ -d "$ref" ]; then
    python "$here/compare_stores.py" "$ref" "$cand" | head -15 || fail=1
  fi
done
exit $fail
