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

## Irregular stores (WOW-NL)

`packed/wow_nl/reports/v1` is a points store with **no cadence**
(`cadence_seconds: null` in its product config): one row per report, at the
time the station made it, nothing snapped to a grid and nothing dropped for
being off one. It is the lossless level 1 of the WOW-NL archive - every field
of every report, exact values, no QC, no gap filling, no rain rule. Regular
grids are *derived* from it (below), with every choice a parameter.

```
packed/wow_nl/reports/v1/
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
obs = pl.scan_parquet("packed/wow_nl/reports/v1/observations/**/*.parquet",
                      hive_partitioning=False)
june = (obs.filter(pl.col("time").is_between(pl.datetime(2024, 6, 1), pl.datetime(2024, 7, 1),
                                             closed="left"))
           .select("station_id", "time", "air_temperature", "rainfall_counter").collect())
hours = pl.read_parquet("packed/wow_nl/reports/v1/hours/**/*.parquet", hive_partitioning=False)
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
`configs/derive/wow_nl_10min.yaml`, and any of them can be changed for one run:

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

Output: `training/<name>/v<version>/observations/year=Y/month=M/data.parquet`
(sorted by station, time), the points-store layout, so `points.read` and
`points.to_matrix` (a dense time x station array) read it like any regular
store. Months are computed independently but read a margin of input on either
side (at least one step, every tolerance and gap, and two days when rain is
derived), so the result equals a single pass over the whole archive - a test
checks exactly that.
