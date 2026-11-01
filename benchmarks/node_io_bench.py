"""Read throughput against the archive vs number of concurrent readers.

Latency-bound reads scale with concurrency; bandwidth-bound ones do not. Each
concurrency level uses a DISJOINT set of tars so nothing is served from this
node's page cache by an earlier level.
"""
import os, socket, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARCHIVE = Path(sys.argv[1])
LEVELS = [int(x) for x in sys.argv[2].split(",")]
PER_LEVEL = int(sys.argv[3]) if len(sys.argv) > 3 else 48

def read_whole(path):
    n = 0
    with open(path, "rb", buffering=0) as fh:
        while chunk := fh.read(1 << 20):
            n += len(chunk)
    return n

tars = sorted(ARCHIVE.glob("*.tar"))
# Spread across the years rather than taking a contiguous run.
step = max(1, len(tars) // (sum(LEVELS and [PER_LEVEL * len(LEVELS)]) or 1))
pool = tars[::step]
print(f"{socket.gethostname()}: {len(tars)} tars, sampling {len(pool)}", flush=True)
try:
    speed = open("/sys/class/net/eno1/speed").read().strip() + " Mbit/s"
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
