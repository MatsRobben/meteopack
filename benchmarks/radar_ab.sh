#!/bin/bash
# Interleaved A/B timing of two commits on the radar sample, in one job.
#
#   benchmarks/radar_ab.sh SIDE_A SIDE_B [REPEATS] [WORKERS] [PRODUCTS]
#
# A side is a commit, optionally with environment settings for that side:
#   4bdd18a                                  the code at that commit
#   4bdd18a@ZARR_THREADING__MAX_WORKERS=1    same code, zarr limited to 1 thread
#   4bdd18a@A=1,B=2                          several settings
#
# A, B, A, B, ...: the NFS server caches the sample after the first read, so a
# run made later is faster for reasons that have nothing to do with the code.
# Interleaving gives both sides the same cache history. Every round is also
# compared against the reference store, so a fast B that writes something
# else fails here.
set -uo pipefail
repo=/vol/knmimo-nobackup/users/mrobben/meteopack
bench=/vol/knmimo-nobackup/users/mrobben/bench_radar
side() {  # "commit@K=V,K2=V2" -> label commit envs
  local spec=$1 commit=${1%%@*} envs=""
  [ "$spec" != "$commit" ] && envs=${spec#*@}
  commit=$(git -C $repo rev-parse --short "$commit")
  local label=$commit; [ -n "$envs" ] && label="$commit@$envs"
  echo "$label $commit ${envs//,/ }"
}
read -r la ca ea <<< "$(side "$1")"; read -r lb cb eb <<< "$(side "$2")"
repeats=${3:-2} workers=${4:-12} products=${5:-rtcor_5min echotop_5min refl_5min}
for c in $ca $cb; do
  [ -d $bench/trees/$c ] || git -C $repo worktree add --detach $bench/trees/$c $c >/dev/null
done
summary=$bench/runs/ab_${SLURM_JOB_ID:-local}.tsv
printf "round\tcommit\tproduct\twall_s\tcompare\n" > $summary
for r in $(seq 1 $repeats); do
  for s in A B; do
    if [ $s = A ]; then label=$la c=$ca envs=$ea; else label=$lb c=$cb envs=$eb; fi
    for p in $products; do
      out=$bench/runs/ab_${s}_r${r}_$p
      # A subshell, so one side's settings never leak into the other's.
      wall=$( (export PYTHONPATH=$bench/trees/$c/src; for kv in $envs; do export "$kv"; done
               bash $bench/trees/$c/benchmarks/radar_pack_timed.sh $p $workers $out $bench/sample) \
             | sed -n 's/^PACK_WALL \([0-9.]*\) s.*/\1/p')
      cmp=$(python $repo/benchmarks/compare_stores.py \
              $bench/runs/ref_$p/packed/knmi_radar/$p/v1/dataset.zarr \
              $out/packed/knmi_radar/$p/v1/dataset.zarr | head -1)
      printf "%s\t%s\t%s\t%s\t%s\n" $r "$label" $p "${wall:-FAIL}" "$cmp" | tee -a $summary
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
