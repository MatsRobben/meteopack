# What is in a store, and how to read it

A store is one product, packed: `packed/<category>/<source>/<product>/v<N>/` under the
data root. Everything you need to read it is in the store itself - every store
directory has a `README.md` generated from its config - so this page is the
general picture.

## Gridded stores (radar, satellite, NWP)

```
packed/radar/knmi_radar/rtcor_5min/v1/
├── dataset.zarr/          the data (zarr v3)
├── manifest.json          how and when it was built: config hash, code version
├── provenance.parquet     one row per raw file that went in
├── scan_report.json       every upstream image layout seen, with frame counts
└── README.md
```

One regular time axis with no gaps, one frame per time step, on one grid.
A time step with no data is still there, flagged (see below).

### Reading

```python
import xarray as xr

ds = xr.open_zarr("packed/radar/knmi_radar/rtcor_5min/v1/dataset.zarr", consolidated=False)
usable = ~(ds.is_missing_frame | ds.is_corrupt_frame)
rain = ds.accumulation_mm.sel(time=slice("2024-06-01", "2024-06-02"))
```

or, from code that should not care about paths:

```python
from datetime import UTC, datetime
from meteopack.core.store import gridded

ds = gridded.read("knmi_radar/rtcor_5min/v1", root,
                  time_range=(datetime(2024, 6, 1, tzinfo=UTC), datetime(2024, 6, 2, tzinfo=UTC)))
```

Both give the same numbers: physical values, NaN where there is no value.
`gridded.read(..., decode=False)` returns the stored integer codes instead.

xarray prints `SerializationWarning: variable ... has multiple fill values
... decoding all values to NaN` when it opens an integer field. That is
intended: both reserved codes (below) should read as NaN.

### Arrays

| array | dims | holds |
|---|---|---|
| the fields, e.g. `accumulation_mm` | time, y, x | integer codes; decoded on read |
| `time` | time | UTC, the timestamp in KNMI's file name (for rtcor, the end of the 5-minute accumulation) |
| `x`, `y` | x / y | projected cell centres, km (polar stereographic, `crs`) |
| `lat`, `lon` | y, x | geographic coordinates of every cell |
| `is_missing_frame` | time | no source file covered this time step |
| `is_corrupt_frame` | time | a file covered it, but the frame was unusable |
| `source_layout_id` | time | which upstream layout the frame came from |
| `source_record_id` | time | which raw file it came from |
| product frame variables, e.g. OPERA's `chain` | time | what the source says about each frame (`frame_variables` in the product config); read as coordinates, like the four above |

### Field values: three kinds of "nothing"

Each integer field reserves two codes, named on the array's attributes:

* **`calibration_out_of_image`** - outside the product's footprint. Decodes to NaN.
* **`calibration_missing_data`** - inside the footprint, but no measurement.
  Decodes to NaN.
* **no echo** - the radar looked and saw nothing. This is an observation, and
  it has a value: 0 mm for `accumulation_mm`, 0 km for `echotop_height_km`,
  -32 dBZ for `reflectivity_dbz` (the array's `no_echo_value` attribute). It is
  the commonest pixel in most frames.

The first two can be told apart only in the codes (`mask_and_scale=False`).
KNMI's files mark no echo in different ways in different years; the store
gives it the same value throughout.

### Frame flags

* **Missing** (`is_missing_frame`): no file existed for that time. The archive
  simply has a gap.
* **Corrupt** (`is_corrupt_frame`): a file existed but the frame could not be
  used. Three causes: the HDF5 member is unreadable or has no image; the tar
  is truncated; or the frame carries a calibration the product config lists in
  `extra.corrupt_layouts`. The last are failed products upstream - placeholder
  calibrations over an image with no echo pixel at all - and decoding them
  would claim "no echo anywhere".

Both read as all-NaN frames. Mask on `is_missing_frame | is_corrupt_frame`.

### Provenance

`source_layout_id` codes are listed in the array's `flag_values` and
`flag_meanings`. `source_record_id` is a stable id of the raw file the frame
was packed from; look it up in `provenance.parquet`:

```python
import polars as pl
prov = pl.read_parquet("packed/radar/knmi_radar/rtcor_5min/v1/provenance.parquet")
prov.filter(pl.col("source_record_id") == int(ds.source_record_id.sel(time="2024-06-01T12:00")))
```

Each row gives the file name, size, modification time, the slots it covered,
how many of those were missing, how many it wrote, and when it was packed.

### The radar products

| product | variables | from | frames | notes |
|---|---|---|---|---|
| `rtcor_5min` | `accumulation_mm`, `quality`, `adjustment_factor_db` | 2018-12 | 770,116 | `quality` exists only from 2023; 11 upstream layouts over the years |
| `echotop_5min` | `echotop_height_km` | 2008-01 | 1,918,484 | 692 placeholder frames recorded corrupt |
| `refl_5min` | `reflectivity_dbz` | 2008-01 | 1,919,439 | 471 placeholder frames recorded corrupt; 2014-03-13 is a truncated download upstream of meteopack (8 frames recovered) |

All three were packed from the complete KNMI archives on 2026-11-01, passed
`meteopack validate`, and matched the raw HDF5 pixel for pixel on 40 randomly
sampled frames each (every variable of every sampled frame), checked by a
script that decodes the HDF5 independently of meteopack's own code.

### The 3D radar composite

`packed/radar/knmi_radar/composite3d_5min/v1/` holds KNMI's 3D reflectivity
composite (KNMI Data Platform dataset `3d-radar-composite` v1.0, files
`3D_RADAR_COMPOSITE_YYYYMMDDHHMM.nc`) as **`(time, z, y, x)`**:

| | |
|---|---|
| `reflectivity_dbz` | uint8, `dBZ = 0.5 * code - 32` (the calibration of `refl_5min`) |
| code 0 | **no echo** (the file's -30 dBZ floor, native code 0) - the same "no echo" as `refl_5min`; codes 1-3 never occur, so a weak echo at -30 dBZ (code 4) stays distinct from no echo |
| code 254 | **not observed** (the file's fill 65535: beyond the radars' reach at that height, or outside the composite) |
| code 255 | **no data for this frame** (the fill value); the frame is flagged `is_missing_frame` or `is_corrupt_frame` |
| `z` | 32 levels, 0.375 to 15.875 km every 500 m: the source's levels 1, 3, ..., 63 of 64 (subsampled, not averaged; `z.attrs["source_indices"]`) |
| `y`, `x` | the `knmi_1km_stereo` grid, **north-up** - flipped from the south-up file, so pixel `[i, j]` is the same place as in `rtcor_5min` |
| chunks | one time step, all levels, the whole 765 x 700 frame; one UTC day (288 steps) per shard; zstd 3 |

`xr.open_zarr(...)` or `gridded.read(...)` decode it to dBZ with NaN for both
254 and 255. To keep "no echo" as a value rather than -32 dBZ, read with
`decode=False` and test `code == 0`.

Every file is checked before it is packed (variables, dtype, shape and dims,
the 64 heights, x/y against the RTCOR grid, projection, fill, units,
add_offset, a per-file scale_factor in [0.0004, 0.0025], and the file's time
against its name); a file that fails is never packed, its frame is flagged
corrupt, the raw file is kept in `raw/radar/knmi_radar/composite3d_5min/quarantine/`,
and the day's status says why. Each file is decoded with **its own**
`scale_factor` - KNMI stretches every file's maximum to the top of the uint16
range (0.0012-0.0018 seen). The structural fingerprint is pinned in the
product config (`schema_fingerprint`); the manifest records every fingerprint
and storage layout seen (`extra.schema_fingerprints`, `extra.storage_layouts`).
Up to mid-2024 the files are chunked `(1, 32, 383, 350)`, later
`(1, 1, 765, 700)`; that is storage, not content, and is not drift.

Per-day status: `days/<YYYY>/<YYYY-MM-DD>.json` beside `dataset.zarr` -
`state` (`done`, `partial`, `empty`, `failed`), the timestamps missing
upstream, downloads that failed, files that failed their check, the scale
range, the key file used, and timings. The manifest carries the per-day
counts (`extra.days`). The archive's first file is
`3D_RADAR_COMPOSITE_202405070850.nc` (not the documented 2023-11-01), so the
store's axis starts 2024-05-07 00:00 and that first day is `partial`.

Checked on 2026-11-09 against 146 local sample files that the stream never
touched (2024-05-10 and 2024-07-09): decoded values within 0.25 dBZ of the
file, and the not-observed and no-echo masks identical pixel for pixel.

### The OPERA rain-rate composite

`packed/radar/opera/rate_15min/v1/` holds the EUMETNET OPERA European
rain-rate composite, 15-minute, on its own 2 km grid, cut to a window around
the Netherlands. It is streamed day by day from the public archive bucket
(`https://s3.waw3-1.cloudferro.com/openradar-archive/`, anonymous, CC BY 4.0,
property rights EUMETNET) by `meteopack stream opera --product rate_15min`;
the raw HDF5 never accumulates.

| | |
|---|---|
| `rain_rate_mmh` | uint16, `mm/h = 0.01 * code`, `(time, y, x)` |
| code 0 | **no rain**: the file's `undetect` (-8888000), observed and dry - and any rate below 0.005 mm/h |
| code 65535 | **no radar coverage**: the file's `nodata` (-9999000) |
| code 65534 | **no data for this frame** (the fill); the frame is flagged `is_missing_frame` or `is_corrupt_frame` |
| code 65533 | 655.33 mm/h: anything higher is clipped to it and counted per file (`n_clipped`) |
| grid | `opera_2km_laea_nl`: OPERA's own LAEA grid (`+proj=laea +lat_0=55 +lon_0=10 +x_0=1950000 +y_0=-2100000 +ellps=WGS84`, metres), **not resampled**; store pixel `[i, j]` is native pixel `[756 + i, 389 + j]` of the 2200 x 1900 composite (`attrs["opera_window"]`). 880 x 880 pixels: the 1536 km KNMI training square around 52.2N 5.3E spans native cols 408-1249 and rows 775-1616, plus 19 px on every side. `x`, `y` are cell centres in metres; `lat`, `lon` per cell |
| chunks | one time step in 2 x 2 tiles of 440 x 440; one UTC day (96 steps) per shard; blosc-zstd 5 with byte shuffle |
| time axis | from 2013-01-01 00:00 UTC, so the store can be filled back without a rebuild; steps never streamed read as missing frames |

Two production chains made the rain rate, and **the series breaks between
them**:

| chain | files | until / from | data window | `chain` | `source_layout_id` |
|---|---|---|---|---|---|
| ODYSSEY | `OPERA@YYYYMMDDTHHMM@0@QIND_RATE.h5` (ODIM 2.0) | up to and including 2024-07-05 06:00 | T-10 .. T+5 min | 1 | 2 |
| NIMBUS | `OPERA@YYYYMMDDTHHMM@0@RATE.h5` (ODIM 2.4) | from 2024-07-05 06:15 | instantaneous at T | 2 | 3 |

Both chains publish 05:30-06:00 on 2024-07-05; the cutover rule takes exactly
one file per step, and ODYSSEY's 05:30 is absent upstream, so that step is a
missing frame. Values are stored as the files give them - **no correction
across the break**; calibrate each chain separately (e.g. against
`rtcor_5min`) by selecting on `chain`.

Per frame, beside the four provenance flags:

| array | holds |
|---|---|
| `chain` | 0 no frame, 1 ODYSSEY, 2 NIMBUS |
| `n_radars` | radars listed in the file's `/how/nodes` (-1 no frame); 142-167 in 2023-2026 |
| `has_nldhl`, `has_nlhrw` | Den Helder / Herwijnen listed in `/how/nodes`: 1 yes, 0 no, -1 no frame |

`provenance.parquet` adds, per file: the chain, the full node list, the data
window, and counts of nodata, no-rain, rain, NaN, negative and clipped pixels
in the window, and the window's maximum rate.

```python
import xarray as xr
ds = xr.open_zarr(".../packed/radar/opera/rate_15min/v1/dataset.zarr", consolidated=False)
rate = ds.rain_rate_mmh.sel(time=slice("2024-07-05", "2024-07-06"))   # mm/h, NaN = no data
nimbus = ds.chain == 2
both_nl = (ds.has_nldhl == 1) & (ds.has_nlhrw == 1)
```

Every file is checked before it is packed: its name and quarter-hour; its
chain, by structure (ODYSSEY: `dataset1` RATE + `dataset2` QIND; NIMBUS:
`dataset1/data1` + `quality1`) as well as by name, and against the chain the
cutover assigns to its time step; `/what` object, version and time against
the name; `projdef`, grid size, scale and corners; quantity, dtype, shape,
gain 1, offset 0, nodata and undetect; and a non-empty `/how/nodes`. A file
that fails is never packed: its frame is flagged corrupt, the file is kept in
`raw/radar/opera/rate_15min/quarantine/`, and the day's status says why. The
structural fingerprint of each chain is pinned in the product config
(`extra.opera.pinned_fingerprints`); others are reported as drift.

Known gaps in the archive (no rain-rate file at all): 2024-12-03..09 (only
ODYSSEY files exist those days, which the cutover rule ignores),
2025-05-02..15, 2026-05-24..25, and several partial days (e.g. 2024-02-20,
2025-05-01, 2025-11-05). Per-day status: `days/<YYYY>/<YYYY-MM-DD>.json`
(`done`, `partial`, `empty`), as for the 3D composite.

Upstream artefacts that are packed as they are: ODYSSEY frames carry a few
persistent hot pixels with absurd rates (e.g. up to 63,958 mm/h at the
Feldberg radar site, 47.9N 8.0E), clipped to 655.33 mm/h and counted in
`n_clipped`.

Checked on 2026-11-09 (trial days 2023-11-01..02, 2024-02-18..21,
2024-03-01..06, 2024-07-04..06, 2024-12-05, 2025-05-01, 2025-05-03,
2025-11-01..10, 2026-11-08): nine frames of both chains decoded
independently from re-downloaded files equal the store exactly (largest
error 3e-14 mm/h; the files hold 0.01 mm/h steps), nodata and no-rain
masks identical pixel for pixel; `x`/`y` equal the native cell centres and
`lat`/`lon` agree with PROJ to 2e-6 degrees; against `rtcor_5min` (OPERA sampled at
every RTCOR pixel, integer shifts of +-3 px, correlation of log(1 + rate)
averaged over wet hourly frames) the best shift is (0, 0) for ODYSSEY (51
frames, RTCOR window centred 5 min before the nominal time; corr 0.768) and
(+1, 0) for NIMBUS (64 frames, window centred on the nominal time; corr
0.712 against 0.708 at (0, 0)). Shift and time lag trade off through
advection: compared against a window 5 min later, both chains prefer
(-1, +1) - a single frame's best shift says little.

### The GFS NWP stores

Two stores hold NOAA GFS 0.25 degree forecasts on GFS's **own grid, not
resampled**, cropped to a window around the 1536 km RainPro-NL model frame
(`knmi_stereo_16km_1536`: KNMI polar-stereographic, centred x = 350 km,
y = -4032 km, edges x -418..1118 km, y -4800..-3264 km). Regridding to the
model grid happens at load time. Both are filled by one stream,
`meteopack stream noaa_gfs --product state_1h_0p25`, from NOAA's public AWS
archive (`s3://noaa-gfs-bdp-pds`, anonymous). Nothing raw is kept: the stream
reads each file's `.idx`, fetches only the wanted GRIB messages with HTTP
byte-range requests (adjacent messages in one request), checks, decodes,
crops and encodes them in memory. A message that fails its check is written
to `raw/nwp/noaa_gfs/state_1h_0p25/quarantine/` with the reason, never packed.

**`packed/nwp/noaa_gfs/state_1h_0p25/v1/`** - the 122 input fields of
RainPro-8 (Pablos et al., ICLR 2026, Appendix I, Table 11), hourly, by valid
time.

| | |
|---|---|
| time | valid hour H, 1 h cadence, axis from 2021-01-01 00 UTC |
| source | run `floor6(H)` (00/06/12/18 UTC) at lead `H - floor6(H)` = 0-5 h: the newest run started at or before H, no latency emulated. If that file is absent upstream: the previous run at lead 6-11 (`is_fallback = 1`) |
| fields | 122 variables `(time, y, x)`, one per (GFS variable, level), e.g. `tmp_850hpa`, `cape_255_0hpa_agl`, `rh_atm`, `lftx4_sfc`; always the **instantaneous** GRIB field (`anl` / `L hour fcst`), never a time mean. The list, with units, steps and availability: the product config, or `gfs_122_variables.tsv` in the trial notes |
| codes | uint16, `value = scale_factor * code + add_offset` per field (array attrs); 65534 = field not available for this frame (absent upstream, or failed its check); 65535 = undefined at this pixel (GRIB bitmap: SNOD/WEASD over sea). `xr.open_zarr` decodes both to NaN |
| encoding | per field a fixed physical range mapped onto codes 0..65533, step chosen as a round number >= range / 65533 (e.g. T 0.002 K, RH 0.002 %, MSLP 0.4 Pa, CAPE 0.2 J/kg, specific humidity 2e-9 .. 8e-7 kg/kg by level). Values outside the range are clipped and counted per day (none in the trials) |
| grid | `gfs_0p25_latlon_nl`: GFS 0.25 deg points 8.00W..19.50E x 60.00N..44.50N (63 rows north first x 111 columns); store pixel `[i, j]` = GFS row `120 + i`, column `(1408 + j) mod 1440`. The model frame spans 7.30W-18.91E, 45.07N-59.40N; the window is the 0.25 deg nodes enclosing it plus 2 more on every side, so bilinear interpolation at load time has all its neighbours. `lat`/`lon` per point (and `y`/`x` = lat/lon, degrees) |
| load-time regridding | `meteopack.sources.nwp.gfs.Regridder(lon, lat, SourceGrid(...))` interpolates to any target lon/lat: bilinear with GRIB-masked neighbours dropped and the weights renormalised; the per-field hint `extra.gfs.fields.<name>.regrid` in the product config says **nearest** for CRAIN, CSNOW, CFRZR, CICEP and CPOFP (0/1 flags; CPOFP's "no precipitation" is -50 %) |
| units | GRIB's own (K, Pa, m, kg/kg, %, J/kg ...), except PRATE in mm/h (x 3600) |
| chunks | one hour per inner chunk (63 x 111), 7 days per shard; blosc-zstd 5, byte shuffle |

Per frame, beside the provenance flags (`is_corrupt_frame` = some field of the
frame failed its check; `source_layout_id` 2 = GFS v15 flat path, 3 = v16
`atmos/` path; `source_record_id` -> `provenance.parquet`, one row per run and
lead used):

| array | holds |
|---|---|
| `source_run_time` | run start, seconds since 1970 (-1 no frame) |
| `source_lead_h` | lead in hours (-1 no frame) |
| `is_fallback` | 1 = from the previous run, 0 = newest run, -1 no frame |
| `n_fields_missing` | how many of the 122 fields are not available in this frame |

**`packed/nwp/noaa_gfs/prate_0p25/v1/`** - precipitation rate per run.

| | |
|---|---|
| time | **run start** (00/06/12/18 UTC), 6 h cadence |
| fields | `prate_f01` .. `prate_f15`: mean rate (mm/h) over lead hours `L-1 .. L`, i.e. valid at run + L h. One variable per lead because the gridded store has no lead axis yet; maps 1:1 onto a `(time, lead, y, x)` layout once it has |
| codes | uint16, `mm/h = 0.002 * code`; 65534 = lead not available |
| per frame | `n_leads_missing` (-1 no run) |
| grid | the same native window as the state store |
| chunks | one run per inner chunk, 30 days (120 runs) per shard |

A model at issue time t0 takes run `floor6(t0)` and the leads that cover
t0 + 1 h .. t0 + 8 h (leads 1..14; 15 is kept as margin).

The one-hour means are **de-averaged** from GFS's PRATE time means
(`A-L hour ave fcst`), whose window restarts every 6 h (f001 = mean 0-1,
f002 = mean 0-2, ..., f007 = mean 6-7): `mean(L-1, L) = (L-A) m(A, L) -
(L-1-A) m(A, L-1)`, with the window start A read from each message, not
assumed. GRIB packing makes a few per cent of the de-averaged values slightly
negative (down to -0.014 mm/h in v16 trial days, -0.18 mm/h in v15); they are
clipped to 0 and counted per day in the status record.

#### What the archive holds (scan of every run 2021-01-01 .. 2026-11-08)

| period | what | effect on the stores |
|---|---|---|
| 2021-01-01 .. 2021-02-25 18z | only leads f000, f003, f006, ... (a 3-hourly backfill, uploaded 2024-04-16) | state: hours not divisible by 3 missing (the previous run has the same gap); prate: no one-hour means, every lead missing |
| 7 runs: 2021-02-02 00z, 02-03 12/18z, 02-04 12z, 02-06 12/18z, 02-07 12z | absent | state: their hours missing; prate: run missing |
| 2021-01-01 .. 2021-03-22 06z (GFS v15, path without `atmos/`) | no CAPE/CIN 90-0 hPa, no LCDC/MCDC, TCDC entire atmosphere only as a time mean; SPFH on pressure levels and ICETK only in `pgrb2b` (read from there) | those 5 fields missing (a time mean is not substituted for an instantaneous field) |
| 2021-02-26 .. 2023-02-02, and 2023-05-01 | no CLMR on pressure levels (present in the Jan-Feb 2021 backfill) | 4 CLMR fields missing |
| 2021-02-26 00z .. now | f000-f015 hourly, every run complete | all 122 fields from 2023-02-03 (except 2023-05-01) |

The GRIB signature (discipline, category, number, level) of every one of the
122 fields is the same in v15 and v16 files; no field changes meaning across
the upgrade as far as the metadata says.

```python
import xarray as xr
st = xr.open_zarr(".../packed/nwp/noaa_gfs/state_1h_0p25/v1/dataset.zarr", consolidated=False)
pr = xr.open_zarr(".../packed/nwp/noaa_gfs/prate_0p25/v1/dataset.zarr", consolidated=False)
t0 = "2024-01-02T13:25"
frame = st.sel(time=t0, method="pad")             # the hour at or before t0 (decoded, NaN = n/a)
run = pr.sel(time=t0, method="pad")               # the newest run at or before t0
```

Per day, `days/<YYYY>/<date>.json` in the state store says which run and lead
made each hour, which fields were absent upstream or failed a check, which
precipitation leads are missing, what was clipped, and the transfer and
timings. Day states: `done` (every frame packed or confirmed absent
upstream), `partial` (a transfer failed; a rerun redoes the day), `empty`.

Checked on 2026-11-09. The pipeline first ran with regridding to the 16 km
frame (trial days 2021-01-10, 2021-03-22..23, 2022-08-15..18, 2024-01-02,
2025-07-01..04, 2026-11-07); the native-grid stores were then re-run and
re-checked on 2021-03-22 (v15) and 2024-01-02 (v16):

* all 122 fields at 4 valid hours (v15 lead 0 and 2, v16 lead 1 and 5),
  re-fetched and picked at the store's own `lat`/`lon` (which also checks the
  georeferencing): the store is within half a quantisation step everywhere,
  and fields absent upstream are entirely 65534 (an earlier check of the
  regridded layout covered 7 hours across all eras the same way);
* the de-averaged one-hour rates against hourly differences of the APCP
  accumulations (an independent route) over 40-62N, 12W-25E: correlation
  0.983-0.9998 for the v16 runs 2024-01-02 12z and 2026-11-07 06z, area means
  within 1 % on most leads (worst 7 %), mean absolute difference
  0.005-0.017 mm/h (APCP itself is packed to 0.0625 mm); the v15 run
  2021-03-22 06z correlates 0.93-0.998. The store equals the native
  de-averaged rate to half a step;
* georeferencing: GFS orography's coastline and the sea mask of WEASD sit on
  the Natural Earth coastline, and the model frame lies inside the window
  with its margin;
* the wet day 2024-01-02, regridded to 16 km, against RTCOR averaged onto the
  same cells:
  hourly domain means correlate 0.86 and peak in the same hours; GFS's daily
  total is ~2x RTCOR's (RTCOR is known to underestimate in winter) - a sanity
  check, not a verification;
* a stream killed (SIGKILL) mid-day resumes: done days are skipped, the
  interrupted day is redone whole.

## Points stores (station networks)

```
packed/stations/dwd/stations_10min/v1/
├── observations/year=2024/data.parquet    one file per partition
├── stations/                              station metadata (lat, lon, height, name)
├── manifest.json
└── README.md
```

One row per (time, station). Columns are `time` (UTC), `station_id` (text) and
one column per variable, in the units the product config states. A variable a
station does not report is null.

```python
from meteopack.core.store import points

df = points.read("dwd/stations_10min/v1", root)          # a polars DataFrame
stations = points.read_stations("dwd/stations_10min/v1", root)
```

or straight from DuckDB:

```python
import duckdb
duckdb.sql("select * from read_parquet('observations/**/*.parquet', "
           "union_by_name=true, hive_partitioning=false)")
```

For a dense `(time, station)` matrix on a regular axis, `points.to_matrix`.

## Checking a store yourself

```bash
meteopack validate knmi_radar --product rtcor_5min
```

reads the metadata and samples frames: time axis, flags, provenance, value
ranges against the config. It does not compare against the raw files; that is
a separate, slower check.

## Irregular stores (WOW-NL)

`packed/stations/wow_nl/reports/v1` is a points store with **no cadence**
(`cadence_seconds: null` in its product config): one row per report, at the
time the station made it, nothing snapped to a grid and nothing dropped for
being off one. It is the lossless level 1 of the WOW-NL archive - every field
of every report, exact values, no QC, no gap filling, no rain rule. Regular
grids are *derived* from it (below), with every choice a parameter.

```
packed/stations/wow_nl/reports/v1/
├── observations/year=2024/month=06/data.parquet   ~7 M reports a month, sorted by (station_id, time)
├── hours/year=2024/month=06/data.parquet          one row per zip member, + one per missing hour
├── station_segments/year=2024/month=06/...        runs of identical position per site, per month
├── stations/data.parquet                          one row per site (standard station schema + extras)
├── station_positions.parquet                      full position history (runs), store-wide
├── station_years.parquet                          per site-year medians, raw height, height-0 count
├── pack_report.json                               store-wide counts + what the last pack did
├── schema.json, manifest.json, README.md
```

### Columns

* `station_id` (WOW `SiteId`, text: numeric ids and UUIDs both occur; 493
  reports have none and keep a null) and `time` (`ReportEndDateTime` parsed to
  UTC microseconds; the original string is kept in `ReportEndDateTime` - some
  have 7 fractional digits).
* The measurements other stores carry, under the same names: `air_temperature`,
  `dew_point_temperature`, `relative_humidity`, `air_pressure_at_sea_level`,
  `air_pressure`, `wind_speed`, `wind_speed_of_gust`, `wind_from_direction`,
  `wind_gust_from_direction`, `wet_bulb_temperature`, `rainfall_rate`, and
  **`rainfall_counter`** - the station's running rain total as uploaded. It is
  *not* an amount: it resets (mostly at local midnight) and drops out; turn it
  into amounts with a rain rule in the derive step.
* Every other WOW field under its own name, typed as its values are (float64
  for every measurement, int64 for ids and codes, bool, text): the imperial
  twins (`*_Fahrenheit`, `*_Inch`, `*_MilePerHour`, `*_Knot`,
  `*_InchOfMercury` - for most private stations these are what was uploaded),
  `Latitude`/`Longitude`/`Height` as reported with each report, `Id`,
  `IsOfficial`, `CreatedDateTime`, `SoftwareType`, ... 82 fields in all.
* `unpinned_fields` - JSON, null on almost every row. A value that did not fit
  its field's type is kept here verbatim, never coerced or dropped:
  `{"WindSpeed_MetrePerSecond": {"raw": "36,612.3", "reason": "string_for_float64"}}`.
  Reasons: `<json type>_for_<pinned type>` (`null_for_float64` is an explicit
  JSON null), or `unknown_field`. Counts per field and reason are in the hours
  table (`unpinned_counts`), `pack_report.json` and `validate`.
* `source_zip`, `source_member` - which yearly zip and hour member the row
  came from.

### What was dropped, and what is missing

Only exact duplicates: rows identical in every field. They come from the 2019
hours that are in both 2019.zip and 2020.zip; per hour the copy with more
records is kept (`hours.chosen`; ties: the later zip) and any report that only
the other copy has is kept too. `hours.n_duplicate_dropped` counts them per
member. Seven 2019.zip members are damaged JSON: their complete reports are
salvaged (`status: salvaged`) and then mostly fall away as duplicates of the
intact 2020.zip copy.

Hours no zip has (Jul-Dec 2020, the end of June 2020) are rows with
`status = 'missing'`. From 2022 almost every hour holds exactly 10 000
reports - one API page - and lost its earliest minutes upstream:
`hours.at_page_cap`.

### Reading

```python
import polars as pl
obs = pl.scan_parquet("packed/stations/wow_nl/reports/v1/observations/**/*.parquet",
                      hive_partitioning=False)
june = (obs.filter(pl.col("time").is_between(pl.datetime(2024, 6, 1), pl.datetime(2024, 7, 1),
                                             closed="left"))
           .select("station_id", "time", "air_temperature", "rainfall_counter").collect())
hours = pl.read_parquet("packed/stations/wow_nl/reports/v1/hours/**/*.parquet", hive_partitioning=False)
```

`points.read("wow_nl/reports/v1", root, time_range=..., columns=[...])` works
too. Because each month file is sorted by station, a time filter cannot skip
row groups inside a month - but it skips whole months by path, and a
`station_id` filter skips row groups.

### Re-packing part of it

`meteopack pack wow_nl --window 2024/2024` re-packs one year. The window is
widened to whole months, those months (observations, hours, segments) are
replaced and every other month is left byte-for-byte alone; the store-wide
station tables and report are rebuilt from all months on disk. A windowed
pack into a store built from a different config is refused.

## Regularising an irregular store (`derive`, kind: regularise)

`meteopack derive wow_nl_10min` turns the reports into one row per
(station, bin) on a regular grid. Everything is a parameter in
`configs/derive/stations/wow_nl_10min.yaml`, and any of them can be changed for one run:

```bash
meteopack derive wow_nl_10min                                   # as configured
meteopack derive wow_nl_10min --set step_seconds=300 --set name=wow_nl_5min_try
meteopack derive wow_nl_10min --set label=centre --set rain.rule=naive --set name=...
meteopack derive wow_nl_10min --window 2024/2024 --set name=wow_nl_10min_2024
```

| parameter | meaning |
|---|---|
| `step_seconds` | grid step |
| `label` | `end` (default): bin T holds (T - step, T], KNMI radar's convention. `start`: [T, T + step). `centre`: [T - step/2, T + step/2) |
| `origin` | grid anchor; default the Unix epoch (bins at :00, :05, ...) |
| `variables.<name>.methods` | any of `last` (+ `_last_age_s`), `first`, `mean`, `min`, `max` (+ `_n`), `nearest` (needs `tolerance_s` <= step/2; + `_nearest_dt_s`), `linear` (needs `max_gap_s`) |
| `rain` | `rule: naive` or `robust` (wow_nl_analysis/scripts/rain_counter.py, ported exactly - a test runs both against that SQL), summed per bin as `rainfall_amount`, with `rainfall_amount_span_s` (seconds of counter history the increments cover) and `rainfall_amount_n` |
| `stages` | per-report steps before binning (QC flags, homogenisation). One demonstration stage: `range_mask`. Add one by registering a function in `meteopack.derive.regularise.STAGES` |
| `dtype` | float32 (half the size) or float64 (every value exact) |

`n_reports` is always written. **Causality:** a column is causal when it uses
no report after T - true for the bin methods with `label: end`, false for
`nearest`, `linear`, the `robust` rain rule (it looks three reports ahead) and
any bin method with `label: start`/`centre`. `regularise.json` in the output
lists every parameter, its hash, every output column with its causality, and
how the run went. A run with different parameters will not mix into an
existing output of the same name (give it a new `name`, or `--refresh`).

Output: `training/<category>/<name>/v<version>/observations/year=Y/month=M/data.parquet`
(sorted by station, time), the points-store layout, so `points.read` and
`points.to_matrix` (a dense time x station array) read it like any regular
store. Months are computed independently but read a margin of input on either
side (at least one step, every tolerance and gap, and two days when rain is
derived), so the result equals a single pass over the whole archive - a test
checks exactly that.

## Quality-control flags (`derive`, kind: qc) - `packed/stations/wow_nl/qc/v1`

`meteopack derive wow_nl_qc` judges every WOW-NL report and writes the
verdicts to a store of their own, **row for row aligned with level 1**: the
same `year=/month=` partitions, the same rows in the same order. Level 1 is
never changed and nothing is dropped - a test that fires sets a bit. The
tests are **reference-free**: only the station's own series and other WOW
stations are used (no KNMI AWS, radar or model data, which are model inputs
and targets).

```
packed/stations/wow_nl/qc/v1/
├── observations/year=2024/month=06/data.parquet   one row per level-1 row, same order
├── reliability/year=.../month=.../data.parquet    station-month x variable table
├── reliability.parquet, stats.parquet             all months (stats: counts per bit)
├── station_months.parquet, neighbours.parquet     positions and neighbour pairs, built once
├── runs/year=.../month=...json                    per-month timings
├── qc.json                                        every parameter, its hash, the bit tables
└── manifest.json, README.md
```

### Reading flags with the reports

```python
import polars as pl
from meteopack.derive import qc

june = qc.read_month(root, 2024, 6)            # level 1 + flags side by side, alignment checked
good_t = june.filter(pl.col("air_temperature_qc_level") >= 3)["air_temperature"]
```

or join yourself: the files have the same row order (`l1_row` is the row's
position in its level-1 month file), so
`pl.concat([l1, flags.drop("station_id", "time")], how="horizontal")` is the
join. `(station_id, time)` is unique except for 5,694 reports (2,847 pairs,
mostly Jan-May 2019) that a station sent twice at the same instant with two
WOW `Id`s; the second of each pair carries `near_duplicate`. A regularise derive can use the flags
directly: `stages: [{kind: qc_mask, min_level: 3}]` nulls every variable whose
level is below 3 before binning.

### Columns

Per variable (`air_temperature`, `dew_point_temperature`, `relative_humidity`,
`air_pressure_at_sea_level`, `air_pressure`, `wind_speed`, `wind_speed_of_gust`,
`wind_from_direction`, `rainfall_rate`, `rainfall_counter`):

* `<var>_qc_fail` (uint32) - one bit per test the value **failed**;
* `<var>_qc_ne` (uint32) - one bit per test that **could not be evaluated**
  (too few neighbours or data), with the same bit number as the test, plus
  `censored_low` (30) and `isolated` (31);
* `<var>_qc_level` (uint8) - summary by a configurable rule: 0 no value,
  1 bad, 2 suspect, 3 not spatially confirmed, 4 good.

Per report: `qc_report` (uint32), `rainfall_increment` (the robust-rule
counter increment the rain tests judge, mm), `l1_row`, `qc_config_hash`.

| fail bit | name | test (default) |
|---|---|---|
| 0 | `range` | outside the climatological range (T -28..42 degC, MSLP 940..1065 hPa, wind 0..35 m/s ...) |
| 1 | `extreme` | physically impossible (T -50..50, wind > 75 m/s ...) - flagged, never capped |
| 2 | `sentinel` | placeholder values: 0 degF (-17.78 degC) for T/Td, RH 0, p 0 |
| 3 | `spike` | up/down from both neighbouring reports by > threshold within 20 min (T 3 K) |
| 4 | `step` | change > threshold within 10 min (T 5 K, wind 13.88 m/s - Chen 2021) |
| 5 | `stuck` | one identical (non-zero for wind/rate) value for >= T 4 h (Cornes 2020), wind 160 min (Chen) ... |
| 6 | `stuck_zero` | zeros for >= 2 days (wind, gust, direction; Chen) |
| 7 | `td_gt_t` | dew point > T + 0.5 K |
| 8 | `td_mismatch` | dew point differs > 2 K from Magnus(T, RH) |
| 9 | `stale_repeat` | the first report after a > 1 h gap repeats the previous record exactly |
| 10 | `network_z` | CrowdQC+ m2: robust z against all stations in the 5-min bin (T: z < -2.33 or > 1.64) |
| 11 | `month_fraction` | CrowdQC+ m3: > 20 % of the station-month failed m2 |
| 12 | `month_corr` | CrowdQC+ m4: station-month r with the network median < 0.9 (T, MSLP), 0.8 (Td, RH), 0.5 (wind) |
| 13 | `buddy` | robust z against the median of up to 20 neighbours within 10 km (>= 5); T: warm > 2, cold < -2.83 |
| 14 | `sct` | titanlib SCT (T): pos 4 / neg 8, eps2 0.5, hourly snapshots; a report gets the nearest snapshot's verdict |
| 15-18 | `rain_transient`, `rain_spike`, `rain_reset_off_midnight`, `rain_huge_increment` | counter anomalies from the robust rule; increment > 20 mm in <= 10 min |
| 19-21 | `rain_faulty_zero`, `rain_high_influx`, `rain_station_outlier` | PWSQC FZ / HI / SO on 5-min amounts, 10 km, >= 5 neighbours |

`qc_report` bits: 0 `near_duplicate` (< 30 s after the previous report),
1 `position_jump` (> 1 km from the station-month position), 2 `dup_coords`
(CrowdQC+ m1), 3 `capped_hour` (hour file at the 10,000-record page cap),
4 `local_time_inconsistent`, 5 `no_position`. Bits 3 and 4 are metadata only
and never lower a level.

The default level rule (`levels` in the config): **1 bad** if any of bits
0-9 or 15-18 is set; **2 suspect** if any network test (10-14, 19-21) failed
or the report has `position_jump`, `dup_coords` or `no_position`; **3 not
spatially confirmed** if a network test could not be run, the station is
isolated, or the value is a censored zero; **4 good** otherwise.

### How the network tests work

The reference field is every station's last usable value per 5-min bin
(bins `(T-5min, T]`), carried forward 10 min so 10-min reporters are in every
bin; values failing a within-station test are not part of it. Each report is
compared with the statistics of its own bin, so the flags are per report even
though the field is regular. Neighbours are stations active in the same
month within the radius, nearest first, from `neighbours.parquet` (rebuilt
only when level 1's station segments or the radii change). An isolated
station (< 5 others with the variable within 15 km, TITAN) gets the
`isolated` bit and the not-evaluable bits of the spatial tests, never a
failure. Heights are not used (`network.use_height: false`): WOW heights are
0 for 99 % of 2015 site-years and unreliable after.

Rain: the counter's robust increments (`rain_counter.py`'s rule) are spread
at a constant rate over the 5-min bins they cover (de Vos 2017; intervals
over 20 min are not spread), then PWSQC's filters run on that matrix exactly
as in its R code (FZ, HI), with SO evaluated once per hour over a 14-day
window that grows back to 28 days to find 100 wet intervals. Bins carry
PWSQC's three states; a report fails if any bin it touched failed, and is
not evaluable if none failed but one was -1 (or it touched none).

Deviations from the published methods, all deliberate: MAD instead of Qn as
the robust scale; buddy radius 10 km instead of CrowdQC+'s 3 km (at WOW's
density most stations would have fewer than 5 buddies); CrowdQC+'s m3 code
flags a month when *fewer than 20 % pass* m2 (including missing values), its
paper and Napoly 2018 say *more than 20 % fail* - the latter is implemented;
SCT on hourly snapshots (TITAN's operational cadence) rather than every
5 min; SO hourly rather than every 5 min. Every threshold is a parameter in
`configs/derive/stations/wow_nl_qc.yaml`, cited there.

### Re-running with other settings

```bash
meteopack derive wow_nl_qc                                    # all months
meteopack derive wow_nl_qc --window 2024/2024                 # some months; the rest stay
meteopack derive wow_nl_qc --set variables.air_temperature.buddy.pos=2.5 --set name=wow_nl/qc_t25
```

A run with other parameters refuses to write into an existing store (new
`name`, or `--refresh`). Windowed runs can go in parallel (one Slurm job per
few years): each month writes its own files and every run rebuilds the
store-wide tables from what is on disk. `write_scores: true` also writes the
z-scores behind m2 and the buddy check, for tuning thresholds without
re-running.

### The reliability table

`reliability.parquet`: one row per station-month and variable - number of
values, share of values with each fail and not-evaluable bit and at each
level, r with the network median (m4), number of neighbours, the inferred
native unit (the coarsest lattice holding >= 90 % of the non-zero values, e.g.
`0.1 degF`, `0.01 inHg`, `1 mph`) and the modal step between values
(quantum), position stability (`n_positions`, share of position jumps),
`dup_coords`, shares of capped-hour and local-time-inconsistent reports, and a
`trust_tier`: 1 official, 2 private with >= 90 % of its T/RH/MSLP values at
level >= 3, 3 otherwise.

### What it flags (full archive, built 2026-11-06)

581,564,223 reports, 126 months, parameters hash `3a432451b3403deb`. Share
of private values failing at least one test / at level 4 (good): T 15.6 % /
48 %, RH 17.7 % / 47 %, Td 5.7 % / 51 %, MSLP 14.3 % / 30 % (station offsets:
m2 10.2 %, buddy 6.1 %), wind 11.1 % / 57 %, rain counter 4.9 % / 43 % (FZ
2.7 %, SO 2.3 %). The largest single T flags are CrowdQC+'s m2 (6.5 %, by
design at least ~6 % of a normal distribution) and m3 (5.3 %). 41 % of
private T values could not be buddy-checked (fewer than 5 neighbours with a
value in the bin) and 13 % come from isolated stations. Per test, year and
network: `stats.parquet`; summarised by `wow_nl_analysis/scripts/qc_eval.py`
(out/qc_eval/rates_*.csv).

Checked against KNMI AWS (evaluation only; private WOW stations within 2 km,
27 stations, one mirror of the AWS excluded): T unflagged MAE 0.56 K, level 1
3.7 K, buddy-flagged 3.2 K (74 % off by > 2 K), SCT-flagged 1.5 K (27 %), m2
0.80 K (9.5 %); m3/m4-flagged T is *not* worse than unflagged there (0.49 /
0.56 K) and RH's m4 only slightly (5.2 vs 3.7 % MAE) - those month tests are
the ones to tune first. MSLP flags pick out offsets (flagged MAE 11 hPa,
unflagged 0.6). Rain per station-hour: FZ-flagged hours missed 89 % of the
KNMI-wet hours (unflagged 10 %), r with KNMI 0.85 unflagged vs 0.47 for SO
and 0.08 for FZ. With errors injected into June 2023 (Alerskans 2022 style):
single T errors of +-4 K and +-8 K, sentinels, a 6-h stuck sensor, wind
zeroed for 3 days and counter spikes are caught 98-100 %; a 6-h +4 K offset
93 %, +2 K 70 % (SCT 57 %); single +-2 K 60 % / 32 %; MSLP +5 hPa for a day
52 %; RH -20 % for 6 h 59 %; a frozen counter on a wet day only 16 % (FZ
needs a wet neighbour *median* for half an hour at 5 min).
