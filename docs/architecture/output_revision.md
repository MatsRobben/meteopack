# Output layout and store schema: a revision

> **Design record, written 2026-11-01.** It explains why the stores look the
> way they do; some details have moved on since. Variable names are KNMI's own
> (`accumulation_mm`, not `precipitation_amount` - see section 6), no-echo
> pixels take a stated value, frames with a broken calibration are recorded
> corrupt, and `source_record_id` now indexes a real `provenance.parquet`.
> For the store as it is, read [docs/stores.md](../stores.md).

Written after reading the radar_v3 and 10min_in_situ_v3 stores in `data_raw`
and the code that produced them. The short version is that the previous
implementation of those stores was right about several things this one got
wrong, and the differences are not cosmetic.

## 1. What the previous radar store does that this one did not

```
radar_v3/rtcor/zarr/dataset.zarr/
  accumulation_mm        uint16  fill 65534   GEO = 0.01 * PV + 0
  quality                uint8   fill 254     GEO = 0.004 * PV + 0
  adjustment_factor_db   uint8   fill 254     GEO = 0.16 * PV - 20.24
  is_missing_frame       bool    (time)
  is_corrupt_frame       bool    (time)
  source_layout_id       int16   (time)
  source_record_id       int64   (time)
  time                   int64   nanoseconds since epoch
  x, y                   int32
  + geographic_attrs, projection_attrs, provenance on the group
  + metadata/ sidecar: source_records.json.gz, layout_lookup.json,
    scan_schema_catalog.json, scan_summary.json
```

against what `meteopack pack knmi_radar` wrote:

```
canonical/knmi_radar/rtcor_5min/v1/
  precipitation_amount   float32  NaN
  radar_quality_index    float32  NaN
  time                   int64    seconds since epoch
  manifest.json          (inside the zarr group, which zarr warns about)
```

Seven real losses:

1. **float32 instead of integer-with-calibration doubles the store.** 772,129
   frames x 765 x 700 is 15 GB as uint16 and 30 GB as float32, for a quantity
   whose source resolution is 0.01 mm and which is *already* integer on disk.
   Decoding to float at pack time throws away exactness and pays for it twice.

2. **`adjustment_factor_db` was not written at all.** One of the three fields
   in the product, silently absent.

3. **`missing` and `out_of_image` were collapsed into one NaN.** They are
   different facts: the radar did not measure here, versus here is outside the
   image. The previous store keeps 65534 and 65535 apart.

4. **No per-frame provenance.** Measured on the real archive: **2,013 missing
   frames** and **36 corrupt frames** out of 772,129. Without
   `is_missing_frame` / `is_corrupt_frame`, a consumer cannot tell a frame that
   was absent from a frame that was present and empty. Both read as NaN.

5. **No layout record.** The archive spans 2018-2026 with **23 schema
   variants**, which the previous pipeline reduced to 7 layout codes and
   recorded per frame. The actual distribution:

   | layout | frames | share |
   |---|---|---|
   | `single_image` (no quality, no adjustment) | 370,100 | 48% |
   | `three_image_mixed_precision` | 215,394 | 28% |
   | `three_image_uint16` | 123,188 | 16% |
   | `legacy_two_image` (**image2 is the adjustment factor**) | 61,398 | 8% |
   | missing | 2,013 | 0.3% |
   | corrupt | 36 | 0.005% |

   That fourth row is the one that bites: for 8% of the archive, `image2` holds
   the gauge adjustment factor and not the quality index. This implementation
   addresses images by `image_geo_parameter` rather than by group index, so it
   would not have mislabelled them - but with no adjustment variable to write
   them to, it discarded them.

6. **No coordinates and no projection metadata.** The store cannot be opened as
   a georeferenced dataset. The previous store carries `geo_product_corners`,
   the proj4 string and the full `/geographic` attribute set.

7. **`manifest.json` lives inside the zarr group**, which is why every read
   prints `Object at manifest.json is not recognized as a component of a Zarr
   hierarchy`.

The previous in-situ store additionally carries CF global attributes straight
from the source - `Conventions`, `title`, `institution`, `source`, `history`,
`references`, `featureType: timeSeries` - and per-variable `standard_name` and
`coordinates`. None of that survived into the points store either.

## 2. What this implementation does that is worth keeping

Not everything should be copied back.

- **Sharded 1-frame chunks.** The previous store uses a `[288, 765, 700]`
  chunk: one chunk is a whole day, 308 MB raw, and reading a single frame
  decompresses all of it. Benchmarked here: 63-89 warm frame reads/s with
  1-frame chunks against 3-5 with a day chunk. Zarr v3 sharding gives the
  1-frame chunk *and* one file per day, so it is not a trade. Keeping it.
- **Content-addressed image lookup.** Addressing by `image_geo_parameter`
  rather than `imageN` is more robust than classifying layouts positionally; a
  new generation that reorders the groups cannot silently swap two variables.
  Layout classification is still recorded, but as provenance, not as the
  mechanism.
- **Points as partitioned parquet, dense matrices as a derived product.** A
  dense `(time, station)` zarr is right for 76 KNMI stations and wrong for the
  ~6,400 European stations this is heading for, where it is mostly NaN. The
  dense matrix stays available as a `derive` output.

## 3. Directory names

`raw/` is clear. The other three were not.

| was | is | why |
|---|---|---|
| `canonical/` | `packed/` | it is what `meteopack pack` writes, and it is the word the previous pipeline used |
| `derived/` | `training/` | says what the datasets are for, which is the only reason they exist |
| `catalog/` | `_bookkeeping/` | it holds no data; the underscore says so and sorts it out of the way |

A store directory also stops being a zarr group:

```
packed/<category>/<source>/<product>/v<n>/
    dataset.zarr/      gridded: the arrays            (or)
    observations/      points: partitioned parquet
    stations/          points: station coordinates
    manifest.json      what built this, from which config
    README.md          what is in it, in words
```

## 4. Gridded store schema

Every gridded product gets the same shape. Encoding is declared per variable in
the product config, not hard-coded:

```yaml
  - name: precipitation_amount
    source_key: "PRECIP_[MM]|ACCUMULATION_[MM]"
    units: mm
    standard_name: precipitation_amount
    legacy_name: accumulation_mm        # what radar_v3 called it
    encoding:
      dtype: uint16
      scale: 0.01                       # value = scale * stored + offset
      offset: 0.0
      missing: 65534                    # measured nothing here
      out_of_image: 65535               # outside the radar image
```

Sentinels default to `iinfo(dtype).max - 1` and `iinfo(dtype).max`, which is
what the previous pipeline used. Re-encoding from each file's own calibration
onto the canonical one goes through a 65536-entry lookup table built once per
frame, as before: it is both faster than per-pixel float arithmetic and exactly
reproducible.

Arrays:

| array | dims | dtype | holds |
|---|---|---|---|
| `<variable>` | time, y, x | per config | the field, integer-encoded |
| `time` | time | int64 | nanoseconds since epoch, UTC |
| `x`, `y` | x / y | float64 | **projected coordinates in km**, not indices |
| `lon`, `lat` | y, x | float32 | geographic coordinates, 4 MB total |
| `is_missing_frame` | time | bool | no file for this slot |
| `is_corrupt_frame` | time | bool | a file that could not be read |
| `source_layout_id` | time | int16 | which generation this frame came from |
| `source_record_id` | time | int64 | row in the provenance table |

`x`/`y` as real projected coordinates rather than `0..n-1` is the one place
this deliberately differs from radar_v3: index arrays carry no information that
the shape does not already carry, and real coordinates make the store open
correctly in xarray and rioxarray without a sidecar.

Group attributes: CF `Conventions`, `title`, `institution`, `source`,
`history`, `references`, `grid_mapping` with the full proj4 and the KNMI
`/geographic` block, plus provenance (config hash, code version, created/updated).

## 5. Points store schema

Unchanged in shape, corrected in metadata: the CF global attributes from the
source, `standard_name` per variable, the source's own short name kept as
`source_name` so KNMI's `ta`/`dd`/`ff` remain greppable, and the provenance
columns.

## 6. Variable naming

Primary names are CF standard names where one exists -
`precipitation_amount`, not `accumulation_mm`. The point of a packed layer that
will hold radar, gauges, satellite and NWP is that the same physical quantity
has the same name in all of them; `accumulation_mm` also puts the unit in the
name, which breaks the day anything is stored in different units. The previous
name is kept in `legacy_name` so existing code can be ported by grep, and the
whole mapping is a config file - if you would rather have the old names, it is
one edit per variable and no code change.

---

## 7. What was built, and what it was checked against

Implemented and verified on real data rather than in fixtures.

### Verified against radar_v3

Same tars, same frames, compared cell by cell.

| generation | frames | result |
|---|---|---|
| 2024-07-02, three images, modern calibration | 576 | `precipitation_amount`, `radar_quality_index`, `gauge_adjustment_factor`: **12,852,000 cells each over 24 frames, 0 differing** |
| 2018-12-20, two images, `PRECIP_[MM]`, missing code 0 | 484 | `source_layout_id` agrees for every frame; precipitation and quality **10,710,000 cells each, 0 differing**; adjustment factor 1,863 of 10,708,137 differ by one count |

Those 1,863 were chased to the end. The 2018 adjustment factor is
`GEO=0.001*PV-32.768`, so re-encoding is real arithmetic, and one input code in
160 is an exact `.5` boundary in exact arithmetic. radar_v3 resolved those in
float32, this does it in float64. **Every differing cell is such a near-tie,
and both answers sit 0.5 counts — 0.08 dB — from the true value.** Neither is
more accurate; float64 is used because it reproduces the intended value 10⁷
times more closely everywhere else.

The dry-pixel rule holds on a real 2018 frame that advertises
`calibration_missing_data = 0`: 426,929 zeros preserved as dry, 104,669
out-of-image, 0 missing.

### Found by opening the store rather than by reading the writer

- A plain `xr.open_zarr` decoded `out_of_image` as data: 65535 × 0.01 = **655.35
  mm of rain** over the whole footprint mask. Fixed with a vector
  `missing_value`; now 66,404 NaN cells, exactly the out-of-image count, and a
  physical maximum of 1.11 mm.
- Root attributes were written once at creation, so adding a title to the
  config changed nothing on disk.
- `read()` returned the provenance arrays as data variables, so `derive` tried
  to build training bricks out of `is_corrupt_frame`.

### Found by reviewing the parallel plan before running it

A zarr shard is one file and may have one writer. `pack` submitted one task per
raw file; 2 of the 2,680 radar tars start at 08:00 rather than 08:05 and
straddle a shard boundary, so they would have raced their neighbours. Files
sharing any shard are now unioned into one task — by the **whole span**, not by
the first frame, which was the first version of the fix and still left a
straddler racing the shard at its far end.

### Storage

28 MB for 576 frames extrapolates to ~37 GB for the 772,129-frame archive,
against 34 GB for radar_v3 and ~68 GB had this stayed float32.
