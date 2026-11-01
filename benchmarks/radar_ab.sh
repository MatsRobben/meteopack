#!/bin/bash
# Interleaved A/B timing of two commits on the radar sample, in one job.
#
#   benchmarks/radar_ab.sh COMMIT_A COMMIT_B [REPEATS] [WORKERS] [PRODUCTS]
#
# A, B, A, B, ...: the NFS server caches the sample after the first read, so a
# run made later is faster for reasons that have nothing to do with the code.
# Interleaving gives both sides the same cache history. Every round is also
# compared against the reference store, so a fast B that writes something
# else fails here.
set -uo pipefail
repo=/vol/knmimo-nobackup/users/mrobben/meteopack
bench=/vol/knmimo-nobackup/users/mrobben/bench_radar
a=$(git -C $repo rev-parse --short "$1"); b=$(git -C $repo rev-parse --short "$2")
repeats=${3:-2} workers=${4:-12} products=${5:-rtcor_5min echotop_5min refl_5min}
for c in $a $b; do
  [ -d $bench/trees/$c ] || git -C $repo worktree add --detach $bench/trees/$c $c >/dev/null
done
summary=$bench/runs/ab_${a}_${b}_${SLURM_JOB_ID:-local}.tsv
printf "round\tcommit\tproduct\twall_s\tcompare\n" > $summary
for r in $(seq 1 $repeats); do
  for c in $a $b; do
    for p in $products; do
      out=$bench/runs/ab_${c}_r${r}_$p
      wall=$(PYTHONPATH=$bench/trees/$c/src bash $bench/trees/$c/benchmarks/radar_pack_timed.sh \
               $p $workers $out $bench/sample | sed -n 's/^PACK_WALL \([0-9.]*\) s.*/\1/p')
      cmp=$(python $repo/benchmarks/compare_stores.py \
              $bench/runs/ref_$p/packed/knmi_radar/$p/v1/dataset.zarr \
              $out/packed/knmi_radar/$p/v1/dataset.zarr | head -1)
      printf "%s\t%s\t%s\t%s\t%s\n" $r $c $p "${wall:-FAIL}" "$cmp" | tee -a $summary
      rm -rf $out/packed       # keep the log, not 100s of MB per round
    done
  done
done
echo "--- median wall per commit and product ---"
python - "$summary" <<'PY'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
for p in dict.fromkeys(r["product"] for r in rows):
    cell = {}
    for c in dict.fromkeys(r["commit"] for r in rows):
        w = [float(r["wall_s"]) for r in rows if r["product"] == p and r["commit"] == c
             and r["wall_s"] != "FAIL"]
        cell[c] = statistics.median(w) if w else float("nan")
    (ca, wa), (cb, wb) = list(cell.items())
    ok = all(r["compare"] == "IDENTICAL" for r in rows if r["product"] == p)
    print(f"{p:13s} {ca} {wa:6.1f}s   {cb} {wb:6.1f}s   {wa / wb:4.2f}x   "
          f"{'all IDENTICAL' if ok else 'OUTPUT DIFFERS'}")
PY
