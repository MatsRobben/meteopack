#!/bin/bash
# Pack one product of a radar sample into a fresh data root, cold, and time it.
#
#   benchmarks/radar_pack_timed.sh PRODUCT WORKERS OUT_ROOT [SAMPLE_ROOT]
#
# Every sample tar is evicted from the page cache first, so a repeat run over
# the same sample reads from NFS again rather than from memory.
set -euo pipefail
product=$1 workers=$2 out=$3
sample=${4:-/vol/knmimo-nobackup/users/mrobben/bench_radar/sample}
rm -rf "$out"
mkdir -p "$out/raw/knmi_radar"
ln -s "$(readlink -f "$sample/raw/knmi_radar/$product")" "$out/raw/knmi_radar/$product"
python - "$sample/raw/knmi_radar/$product" <<'PY'
import os, sys
from pathlib import Path
n = 0
for p in Path(sys.argv[1]).iterdir():
    fd = os.open(p.resolve(), os.O_RDONLY)
    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd); n += 1
print(f"evicted {n} tars from the page cache")
PY
# Redirect, then filter: piping a worker pool straight into grep lets an
# orphaned worker hold the pipe open, and the pipeline then never ends.
log="$out/pack.log"
status=0
/usr/bin/time -f "PACK_WALL %e s  PEAK_RSS %M KB" \
  python -m meteopack.cli pack knmi_radar --product "$product" --workers "$workers" \
  --data-root "$out" > "$log" 2>&1 || status=$?
grep -vE "no image groups|unreadable frame" "$log" | tail -12
exit $status
