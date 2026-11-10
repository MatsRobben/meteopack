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
