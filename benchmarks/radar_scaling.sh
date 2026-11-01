#!/bin/bash
# Pack throughput against worker count, each point on a disjoint cold sample.
#
#   benchmarks/radar_scaling.sh "12 24 48" [PRODUCTS]
#
# Point k uses $bench/sample_big<k>, built with radar_sample.py --offset k
# --no-examples: the NFS server caches what an earlier point read, so reusing
# one sample would hand every later point a head start.
set -uo pipefail
bench=/vol/knmimo-nobackup/users/mrobben/bench_radar
here=$(dirname "$0")
products=${2:-rtcor_5min echotop_5min refl_5min}
echo "$(hostname) load $(cut -d' ' -f1 /proc/loadavg); co-tenants: $(squeue -w $(hostname) -h -o '%u:%C' | tr '\n' ' ')"
printf "%-13s %7s %8s %9s %10s\n" product workers wall_s frames/s "MiB/s read"
k=1
for w in $1; do
  for p in $products; do
    sample=$bench/sample_big$k
    mib=$(du -sLm $sample/raw/knmi_radar/$p | cut -f1)
    out=$bench/runs/scale_w${w}_$p
    res=$(bash $here/radar_pack_timed.sh $p $w $out $sample)
    wall=$(sed -n 's/^PACK_WALL \([0-9.]*\) s.*/\1/p' <<< "$res")
    frames=$(sed -n 's/^wrote \([0-9]*\) frames.*/\1/p' <<< "$res")
    printf "%-13s %7s %8s %9.0f %10.0f\n" $p $w ${wall:-FAIL} \
      $(python -c "print(${frames:-0}/${wall:-1})") $(python -c "print($mib/${wall:-1})")
    grep -E "ERROR|Traceback|grouped" <<< "$res" | head -3
    rm -rf $out/packed
  done
  k=$((k+1))
done
