#!/bin/bash
# Do several nodes reading at once get more in total than one node alone?
#
#   benchmarks/nfs_aggregate.sh RUN_ID N_NODES OFFSET
#
# Submit one copy per node. Each waits at a barrier (a file per node under a
# shared directory) until all N have arrived, so the reads genuinely overlap;
# jobs released by a dependency start seconds apart, which on a 20 s read is
# most of the overlap gone.
set -uo pipefail
run=$1 n=$2 offset=$3
bar=/vol/knmimo-nobackup/users/mrobben/bench_radar/barrier/$run
mkdir -p $bar && touch $bar/$(hostname)
for _ in $(seq 1 600); do [ "$(ls $bar | wc -l)" -ge "$n" ] && break; sleep 1; done
echo "$(hostname): released with $(ls $bar | wc -l)/$n at $(date -u +%T.%N | cut -c1-12)"
python benchmarks/node_io_bench.py \
  /vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi/radar_tar_refl_composites \
  16 150 $offset
echo "$(hostname): done at $(date -u +%T.%N | cut -c1-12)"
