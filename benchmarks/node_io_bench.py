"""Read throughput against the archive vs number of concurrent readers.

Latency-bound reads scale with concurrency; bandwidth-bound ones do not. Each
concurrency level uses a DISJOINT set of tars so nothing is served from this
node's page cache by an earlier level.
"""
import os
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARCHIVE = Path(sys.argv[1])
LEVELS = [int(x) for x in sys.argv[2].split(",")]
PER_LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 48
# Different per node, so one node's reads do not leave the next node's
# tars in the NFS server's cache.
OFFSET = int(sys.argv[4]) if len(sys.argv) > 4 else 0

def read_whole(path):
    # Evict first: a re-run on the same node otherwise reads its own page
    # cache - one did, at 10,292 MiB/s. This cannot reach the server's cache.
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    n = 0
    with open(path, "rb", buffering=0) as fh:
        while chunk := fh.read(1 << 20):
            n += len(chunk)
    return n

tars = sorted(ARCHIVE.glob("*.tar"))
# Spread across the years rather than taking a contiguous run.
step = max(1, len(tars) // (PER_LEVEL * len(LEVELS) or 1))
pool = tars[OFFSET::step]
print(f"{socket.gethostname()}: {len(tars)} tars, sampling {len(pool)}", flush=True)
try:
    speed = Path("/sys/class/net/eno1/speed").read_text().strip() + " Mbit/s"
except OSError:
    speed = "?"
print(f"  cores={os.cpu_count()} eno1={speed}", flush=True)

cursor = 0
for n in LEVELS:
    batch = pool[cursor:cursor + PER_LEVEL]
    cursor += PER_LEVEL
    if not batch:
        break
    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=n) as ex:
        total = sum(ex.map(read_whole, batch))
    dt = time.monotonic() - t0
    print(f"  {n:3d} readers: {total/dt/1048576:7.1f} MiB/s "
          f"({len(batch)} tars, {total/1073741824:.1f} GiB in {dt:.1f}s)", flush=True)
