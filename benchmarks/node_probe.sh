#!/bin/bash
# How fast is THIS node for radar packing, and what is it sharing with?
#
#   benchmarks/node_probe.sh OFFSET
#
# CPU: a warm parse of the same two tars on every node, in ms/frame - parse is
# the CPU-bound part of a pack. I/O: read throughput against concurrency over a
# set of tars chosen by OFFSET, so each node reads tars no other probe touched.
# NFS: per-op round-trip time over the I/O test, from /proc/self/mountstats -
# a busy server shows up as a long RTT whatever this node is doing.
set -uo pipefail
offset=${1:-0}
K=/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi
echo "== $(hostname): load $(cut -d' ' -f1-3 /proc/loadavg), $(nproc) cpus in this job"
echo "== co-tenants: $(squeue -w $(hostname) -h -o '%u:%j:%C' | tr '\n' ' ')"
python - <<'PY'
import os, time
from pathlib import Path
from meteopack import cli
from meteopack.sources.base import RawFile
K = Path("/vol/knmimo-nobackup/users/mrobben/data_raw/data/raw/knmi")
for product, archive, day in (("rtcor_5min", "nl_rdr_data_rtcor_5m_tar", "20210601"),
                              ("echotop_5min", "radar_tar_echotopheight_5min", "20200601")):
    tar = next((K / archive).glob(f"*__{day}T*.tar"))
    ctx = cli.Context(cli.build_parser().parse_args(
        ["pack", "knmi_radar", "--product", product, "--data-root", "/tmp/np"]))
    raw = RawFile(path=tar, sha256="", n_bytes=tar.stat().st_size, unit_key=tar.name)
    ctx.src.parse(raw, ctx.cfg)                          # imports + page cache
    runs = []
    for _ in range(3):
        t0 = time.perf_counter()
        ctx.src.parse(raw, ctx.cfg)
        runs.append(time.perf_counter() - t0)
    print(f"   cpu  {product:13s} warm parse {1000 * min(runs) / 288:6.2f} ms/frame")
PY
# READ: ops trans timeouts bytes_sent bytes_recv queue_ms rtt_ms execute_ms ...
snap() { awk '/mounted on \/vol\/knmimo-nobackup /{f=1} f&&$1=="READ:"{print $2, $7, $8, $6; exit}' /proc/self/mountstats; }
a=$(snap)
python benchmarks/node_io_bench.py $K/radar_tar_refl_composites 4,8,16,32,64 24 $offset | sed 's/^/   io /'
b=$(snap)
python - "$a" "$b" <<'PY'
import sys
(o1, q1, r1, b1), (o2, q2, r2, b2) = (map(float, x.split()) for x in sys.argv[1:3])
n = o2 - o1
print(f"   nfs  {n:.0f} READ ops, {(b2 - b1) / 2**20:.0f} MiB: mean RTT "
      f"{(r2 - r1) / max(n, 1):.1f} ms, mean queue {(q2 - q1) / max(n, 1):.2f} ms")
PY
