#!/bin/bash
# Traced packs of one product at several worker counts, each on its own sample.
#   benchmarks/radar_trace.sh PRODUCT "12:4 24:5 48:6"     (workers:sample_big<N>)
set -uo pipefail
bench=/vol/knmimo-nobackup/users/mrobben/bench_radar
p=$1
for spec in $2; do
  w=${spec%%:*} k=${spec##*:}
  out=$bench/runs/trace_w${w}_$p
  export METEOPACK_TRACE=$bench/runs/trace_w${w}_$p.json
  echo "== $p workers=$w sample_big$k"
  bash $(dirname "$0")/radar_pack_timed.sh $p $w $out $bench/sample_big$k | grep -E "trace:|PACK_WALL|^wrote"
  rm -rf $out/packed
done
