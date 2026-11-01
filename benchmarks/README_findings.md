# Where the time goes (measured 2026-11-02, cn84: 16 cores, 251 GB RAM, NFS)

## The pipeline is I/O-bound, and already near the hardware ceiling

Radar parse throughput against the real archive, **cold cache, disjoint files
per run** (re-running over the same tars measures the page cache: the first
sample went 29 MiB/s cold to 483 MiB/s warm, which would have made every
comparison nonsense):

| workers | MiB/s | parallel efficiency |
|---------|-------|---------------------|
| 1       | 26    | —                   |
| 4       | 56    | 0.67                |
| 8       | 57    | 0.39                |
| 16      | 66    | 0.25                |

Raw filesystem ceiling, measured with `dd` independently of this code:

* one cold stream: **29-43 MB/s**
* eight parallel streams: **76 MiB/s**
* a file already in page cache: **2.6 GB/s**

So at 16 workers the pipeline reaches ~87% of what the filesystem will give.
**Adding CPU cannot make this faster.** The 16-core count is not the
constraint; NFS bandwidth is.

## Which makes the number of passes the thing to optimise

Each full pass over the 127 GB radar archive costs ~36 minutes at 60 MiB/s.
The pipeline currently makes three:

1. `fetch` - hardlinks cost nothing, but it reads every byte to sha256 it
2. `scan`  - opens every frame to collect layouts
3. `pack`  - opens every frame again to write them

That is ~108 minutes of pure I/O for 36 minutes of actual work.

## The RAM is the lever, not the cores

251 GB of RAM against a 127 GB archive means **the whole thing fits in page
cache**. A second pass over a file just read runs at 2.6 GB/s - 35x faster
than reading it cold. The win is therefore to do all the work on a file while
it is hot, rather than three cold sweeps over the whole archive.

Concretely, in order of value:

1. **Fuse scan into pack.** `pack` already opens every frame and already
   classifies its layout; emitting the scan report as a by-product removes an
   entire 36-minute pass. This also removes the failure mode that cost 5,508
   frames: pins can be refreshed from the pack itself rather than from a
   separate sweep that is tempting to skip.
2. **Stop re-reading to hash.** `fetch` hardlinks from a local archive, so no
   bytes move and the digest guards against later corruption rather than
   against transfer error - at the price of a full read.
3. **Do not raise the worker count.** 4 workers already reach 85% of the
   throughput that 16 reach, at a quarter of the memory.

## Per-frame CPU, for reference

Measured on 96 frames, serial, with the file hot so the phases are visible:

| phase  | ms/frame | note |
|--------|----------|------|
| decode | 6.7      | zlib, inside h5py |
| encode | 5.5      | lookup-table re-encode onto the canonical calibration |
| open   | 2.5      | h5py metadata parse |
| read   | 0.1      | from page cache |

~15 ms of CPU per frame against ~40 ms of I/O at 60 MiB/s: the CPU is less
than half-busy even at one worker, which is the same conclusion from the other
side.
