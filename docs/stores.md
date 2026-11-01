# What is in a store, and how to read it

A store is one product, packed: `packed/<source>/<product>/v<N>/` under the
data root. Everything you need to read it is in the store itself - every store
directory has a `README.md` generated from its config - so this page is the
general picture.

## Gridded stores (radar, satellite)

```
packed/knmi_radar/rtcor_5min/v1/
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

ds = xr.open_zarr("packed/knmi_radar/rtcor_5min/v1/dataset.zarr", consolidated=False)
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
prov = pl.read_parquet("packed/knmi_radar/rtcor_5min/v1/provenance.parquet")
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

## Points stores (station networks)

```
packed/dwd/stations_10min/v1/
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
