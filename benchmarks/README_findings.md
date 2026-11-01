# Radar packing on the compute nodes (measured 2026-11-01)

Everything below was measured as Slurm jobs on das/icis nodes, from pinned
commits (`benchmarks/slurm/radar_at.slurm`), with the page cache evicted
(`posix_fadvise`) and A/B runs interleaved. An earlier version of this file
said the pipeline was I/O-bound at a 76 MiB/s ceiling and that adding CPU
could not help. That was measured on cn84, the interactive node, which has a
1 GbE link; it does not describe the cluster.

## The hardware

| node | CPU | parse, rtcor / echotop | NFS read peak (one node alone) |
|---|---|---|---|
| cn104, cn105 | Xeon Silver 4214, 24 cores | 5.7 / 4.0 ms/frame | ~167 MiB/s at 16-32 readers |
| cn117 | EPYC 7452, 64 cores | 4.9 / 3.5 | 171 |
| cn114, cn115 | EPYC 7642, 96 cores | 4.9 / 3.5 | 157-187 |

* **The NFS server is the ceiling: about 230 MiB/s in total.** One node gets
  ~165; two reading at once got 212 together, three got 231 (each slowing to
  ~77). Spreading I/O-bound work across nodes buys 25-40%, not N times.
  Above ~32 concurrent readers per node, throughput drops.
* **A Slurm CPU is a hyperthread.** `--cpus-per-task=48` is 24 physical cores,
  and a 12-CPU das-normal job is 6. Going from one worker per core to two gave
  1.25x - the usual hyperthreading figure - and from 48 to 96 workers on 48
  hardware threads was 1.6x *slower*.
* **Co-tenants matter for CPU, not for reads.** cn115 (same chip as cn114) parsed
  51% slower under a neighbour running ~120 threads on 32 allocated CPUs.

## What bounds each product

Measured at 12, 24 and 48 workers on cn114 on disjoint cold samples:

| | 12 | 24 | 48 workers |
|---|---|---|---|
| rtcor | 514 frames/s (87 MiB/s) | 810 (139) | 788 (135) |
| echotop | 1,440 | 2,286 | 2,722 |
| refl | 1,100 | 1,763 | 2,040 |

rtcor is read-bound from ~24 workers: 135 MiB/s of reads plus its store
writes to the same server. echotop and refl are CPU-bound and scale with
physical cores. In a traced pack the workers were 91-99% busy at every count,
with start-up and tail a few seconds, so the pool wastes nothing.

## What was changed, and what was tried and dropped

Every change was required to write a store identical to the reference
(`compare_stores.py`: values, dtypes, chunking, codecs, attrs).

| change | effect | output |
|---|---|---|
| exact file spans for shard grouping | echotop/refl packed as 16/15 serial tasks before; now one per tar | (a correctness fix) |
| cache each encode lookup table | rtcor 1.35x, echotop 1.22x, refl 1.24x (pack, A/B) | identical |
| apply tables with `bytes.translate` / `np.take` | parse CPU -14 to -24% (serial profile) | identical |
| zarr `codec_pipeline.batch_size` 64 on write | write -5 to -10% | byte-identical |
| HDF5 open from a file image instead of BytesIO | none | dropped |
| zarr thread-pool size | within noise | dropped |
| read each tar in one sequential pass | none | dropped |
| more workers than hardware threads | 1.6x slower | dropped |
| zstd level 1 instead of 3 | -0.1 to -0.6 ms/frame, same size | not done: changes bytes on disk |

## Full archive, all three products (2026-11-01)

| | tars | frames | pack | where |
|---|---|---|---|---|
| rtcor | 2,680 | 770,116 | 20.7 min | cn115, 24 workers, alongside refl |
| echotop | 6,668 | 1,918,484 | 9.8 min | cn114, 48 workers |
| refl | 6,670 | 1,919,430 | 18.6 min | cn114, 48 workers, after echotop |

The same echotop pack took 85 minutes on cn84 that morning and lost a third
of its frames to the grouping bug.

## Measuring traps met on the way

* A "cold" benchmark that read 148 MiB/s on a 119 MiB/s link was page cache.
  Check any cold number against the link.
* `posix_fadvise(DONTNEED)` empties the node's cache, not the NFS server's: a
  second run over the same files is faster for reasons unrelated to the code.
* The first parse in a process pays ~12 s of imports from the NFS-hosted venv;
  profile after a warm-up.
* A 14-second pack on a small sample cannot resolve a 10% CPU change; use the
  serial profile for CPU changes and the sample pack for equality.

---

# The three radar archives (added 2026-11-02)

There are three, not one, and they are the same reader with different tokens:

| product | tar token | member prefix | tars | raw | day boundary |
|---------|-----------|---------------|------|-----|--------------|
| `rtcor_5min`   | `TARRRT` | `RAD_NL25_RAC_RT_` | 2,680 | 127 G | 08:05 |
| `echotop_5min` | `TARETH` | `RAD_NL25_ETH_NA_` | 6,668 | 38 G  | 00:00 |
| `refl_5min`    | `TARPCP` | `RAD_NL25_PCP_NA_` | 6,670 | 104 G | 00:00 |

Three things differ between them and all three cost data if assumed away.

**The calibration moves within an archive.** Already handled by the re-encoding
machinery, and already documented in the previous pipeline - this is a note,
not a discovery:

    echotop  2008  GEO = 0.062992 * PV + 0.000000
             2017  GEO = 0.062992 * PV + -0.062992
             2026  GEO = 0.102362 * PV + -0.102362
    refl     2008  GEO = 0.500000 * PV + -31.500000
             2017+ GEO = 0.500000 * PV + -32.000000

**The end token means different things.** `rtcor`'s
`..._20240704T080000` names the LAST frame the tar holds, so the half-open end
is one cadence later. `echotop` and `refl` write
`..._20240602T000000`, which is already the exclusive end. Treating them alike
adds a cadence that does not exist and pulls a whole neighbouring day into
every window. `extra.end_token_is_exclusive` carries the difference.

**A stored 0 is an observation, not an absence.** Both declare
`calibration_missing_data = 0` with `calibration_out_of_image = 255`. Measured
on 2024-05-31 12:00:

| | out_of_image (255) | zeros | zeros outside the footprint |
|---|---|---|---|
| echotop | 273,287 | 238,869 | **0** |
| refl    | 104,443 | 357,203 | **0** |

Not one zero falls outside the radar's own coverage mask, and they are 91% and
83% of the in-image pixels. That is the radar looking and seeing no echo,
which for a precipitation dataset is the most common observation there is.

## Is it every schema variant, or just the one sampled?

Every one. A census of 40 tars per archive spread over 2008-2026, counting how
often each *declared* sentinel actually appears in the pixels:

| parameter | dtype | declared missing | how often it occurs |
|---|---|---|---|
| `PRECIP_[MM]` | uint16 | 0 | **74.98%** |
| `PRECIP_[MM]` | uint8 | 0 | **79.50%** |
| `REFLECTIVITY_[DBZ]` | uint8 | 0 | **69.98%** |
| `HEIGHT_[KM]` | uint8 | 0 | **44.72%** |
| `QUALITY_[-]` | uint16 | 0 | **0.00%** |
| `ADJUSTMENT_FACTOR_[DB]` | uint16 | 0 | **0.00%** |
| `ACCUMULATION_[MM]` | uint16 | 65534 | 0.00% |
| `QUALITY_[-]` | uint8 | 254 | **81.57%** |
| `ADJUSTMENT_FACTOR_[DB]` | uint8 | 254 | **81.57%** |

A declared missing code of **0 is only ever populated for the intensity
variables**, where it is the bottom of the scale. For quality and the
adjustment factor - the two where 0 would genuinely mean "no measurement" - it
is declared and never used. The non-zero sentinels (254, 65534) are honoured
normally, and 254 really does cover 81.57% of a quality frame.

So reading a declared 0 as "no echo" is safe across every variant of all
three products, and it is what makes the record continuous. Note the value it
takes comes from the product config (`no_echo:` - 0 mm, 0 km, -32 dBZ), not
from decoding the code with the file's calibration: KNMI shifted the offset by
one count between generations, and decoding put no-echo echo tops at -0.102 km
and split no-echo reflectivity between -31.5 and -32.0 dBZ (fixed 2026-11-01).

Two placeholder calibrations are not data at all: frames declaring
`GEO = 1.0 * PV + 0` (691 echotop and 471 refl frames on ~140 days from 2008
to 2026) or, once, the reflectivity formula on a height image carry an image
with no echo pixel at all - all 1,163 checked - while the neighbours checked
hold tens of thousands. They are failed products, listed in
`extra.corrupt_layouts` and recorded as corrupt frames. rtcor renames
`PRECIP_[MM]` to `ACCUMULATION_[MM]` in 2023 and moves the missing code from 0
to 65534 at the same time. Honouring it literally would make the archive read
~75% missing before the change and ~75% dry after it - a step in the middle of
the record. Measured on the packed store, the dry fraction is flat across the
boundary:

    2019  73.9%   2021  73.7%   2023  74.2%   2025  79.0%
    2020  73.7%   2022  78.0%   2024  74.2%   2026  74.3%

(The `out_of_image` fraction does drop from ~19.5% to ~12.2% around 2023-24.
That is a real change: the composite footprint widened.)

radar_v3 mapped every zero to 255, and never used 254 at all, so
`missing` and `out_of_image` collapsed into a single code there. This store
keeps the zeros and keeps the two sentinels apart - the same judgement the
previous pipeline already made for rtcor's accumulation image, where it
preserved 426,929 dry pixels. It is `extra.missing_data_policy`, one line per
product, if the other reading is wanted.

---

# knmi_insitu was CPU-bound, not I/O-bound (2026-11-02)

Radar is limited by the filesystem. This source was not, and the difference
matters because the fix is the opposite one.

Measured on one tar (23.9 MiB, 144 members, 9,503 rows):

    before   6.6 s    3.6 MiB/s     17x BELOW the 60 MiB/s filesystem floor
    after    1.5 s   15.5 MiB/s

`cProfile` put 65% of the time in `h5netcdf`'s `File()` constructor - 36,864
`h5py` group lookups and 33,264 attribute reads per tar - before a single
value was read. h5netcdf builds a netCDF *view* of the whole hierarchy
eagerly; we want fourteen variables out of ~256 objects.

netCDF-4 is HDF5, so `h5py` reads the same bytes without the walk:

| | before | after |
|---|--------|-------|
| per member | 49 ms | 13.4 ms |
| per tar | 6.6 s | 1.5 s |
| group lookups per tar | 36,864 | 2,736 |

**4.4x.** Over 5,211 tars that is ~70 minutes down to ~22 at six workers, and
it is *still* CPU-bound at 15.5 MiB/s, so unlike radar this one does scale
with workers.

Verified rather than assumed, because a faster reader that returns different
numbers is worthless:

* 432 arrays spread across the archive read byte-identical both ways
* 292 values from six tars spanning 2018-2026 match the store that was built
  with h5netcdf, exactly
* the 55 in-situ unit tests pass

Two h5py differences had to be handled: it returns strings as `bytes` where
h5netcdf returns `str`, and scalars as 1-element arrays. The first comparison
written here reported "every value differs" purely because of that - the data
was identical and the test was wrong.

## The general lesson

Two sources, two opposite bottlenecks. Measure before optimising - and on
the machine that will run it: on cn84's 1 GbE link radar looked I/O-bound
everywhere, while on the compute nodes only rtcor is, and adding workers to
in-situ helps a lot. The per-stage profile is what
distinguishes them, not intuition about "tabular data should be fast".
