# Dataset generation

Dataset-specific builders live here; `common.py` holds reusable configuration, indexing and manifest utilities. The RTCOR builder stores the crop as a single timeline, with a Parquet sample index. It is 2D-only and does not claim 3D pairing.

From `meteopack/`:

```bash
.venv/bin/python -m meteopack.datasets.rtcor --config configs/datasets/rtcor.yaml
```

Run full builds on a compute node. Copy the YAML and change `end` and `output` to create a small pilot first; `end` is exclusive. Paths are explicit. The source store is read-only. The output has `data.zarr`, `samples.parquet`, `build.json`, shard markers and a final `manifest.json`. A failed build can be restarted with the same config and source; a completed build refuses overwrite. Run only one writer per output. Choose a new output directory for changes. The manifest records source, crop, time range, code revision, transform, split, counts and actual encoding statistics.

The uint8 encoding sets accumulations <= 0.01 mm to zero, rounds the remainder to the nearest 0.02 mm with ties upward, and clips at 5.08 mm. Code 255 is out-of-image. A separate status array identifies missing, corrupt and pixel-missing frames; none may occur in accepted 22-frame samples. Future new pixel-missing observations are rejected rather than silently treated as dry. This is lossy and must be evaluated on validation data; retain the original packed store for full-resolution evaluation. Validation is the first UTC calendar day of each month; overlapping samples must not cross a split boundary.

Read `samples.parquet` to select a split. For each row use `start_idx` to read 22 frames from `rain_code`, decode valid codes as `code * 0.02` mm, and mask code 255. The first four frames are history. Read encoding and sample lengths from the manifest rather than hard-coding them in a reusable loader. No PyTorch dependency is required to build the data.

The builder manifest records source-read, encoding, output-write and index times. Use `TimelineDataset(path, 'train')` or `TimelineDataset(path, 'validation')` for a small NumPy reader, or adapt its contract to PyTorch.

For a full RTCOR build, submit the generator to a compute node using your cluster's Slurm configuration. This workspace has a local wrapper at `slurm/datasets/rtcor_full.sbatch`; Slurm wrappers are deliberately excluded from the repository because they contain user-specific paths and account settings. The default YAML also contains workspace-specific paths: update them before running elsewhere. The final dataset is `meteopack_data/training/radar/rtcor_central256/v1`. If interrupted, rerun with the same configuration and source to resume completed shards; only one job may write to that output at a time.

## Cleaned RTCOR (clutter removal)

`configs/datasets/rtcor_clean.yaml` builds the same timeline over the whole archive (2018-12 onwards) with `clean: true`. Every frame is first cleaned on the full 765 x 700 grid by `rtcor_clean.py`, using the previous and next source frame as context, and only then cropped and encoded. The cleaner makes the pre-2023 processing chain look like the post-2023 one: sea clutter, anomalous propagation, interference spikes and wind farms are removed. It applies the same procedure to every period; new-chain frames lose about 0.2-1 % of their echo. Nothing is dropped. Clutter becomes dry, spike pixels are refilled from the neighbouring frames and their surroundings, and pixels outside the common radar footprint become code 255. The module docstring lists the nine steps and their parameters. The fixed inputs (tile model, static clutter map, footprint, radar positions) are in `assets/rtcor_clean_v5.npz`, which `scripts/rtcor_clean_assets.py` builds from the clutter study outputs.

Install the optional dependencies with `uv sync --extra clean` (scipy, pySTEPS, OpenCV). Cleaning takes about 1.2 s per frame, which is roughly 260 CPU hours for the archive, so build shards in parallel:

```bash
python -m meteopack.datasets.rtcor --config configs/datasets/rtcor_clean.yaml --prepare
python -m meteopack.datasets.rtcor --config configs/datasets/rtcor_clean.yaml --worker K --workers N   # K = 0..N-1, in parallel
python -m meteopack.datasets.rtcor --config configs/datasets/rtcor_clean.yaml                         # index and manifest
```

Workers can be stopped and restarted; completed shards are kept. Start the final run only after all workers have finished. Frame status 4 (`not_cleaned`) marks the first and last frame of the source archive, which have no context. The manifest's `cleaning` entry records the cleaner version, the hashes of its code and inputs, and totals: frames cleaned, frames with the warm-season scene threshold, spike pixels, and wet pixels and summed codes before and after cleaning. The protection lineage restarts at every shard boundary (once a day).
