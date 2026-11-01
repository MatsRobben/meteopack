"""Write a short README into the data root and into each layer.

The directory names (`raw`, `packed`, `training`, `_bookkeeping`) say what the
pipeline calls a layer, not what is in it, and someone opening the data root in
a file browser has no code to read. A store is also the thing most likely to
outlive the checkout that made it — it gets copied to another machine, handed
to a colleague, or found again two years later — so the explanation belongs
next to the data rather than only in the repo.

These files are regenerated on every write, are never read back by the
pipeline, and describe only the layout. They deliberately say nothing about
which products exist: that would go stale the moment a new one is packed, and
`meteopack catalog` answers it from the data itself.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

README = "README.md"

_ROOT = """# meteopack data root

Four directories, in the order data moves through them.

| directory   | what is in it | written by | safe to delete? |
|-------------|---------------|------------|-----------------|
| `raw/`      | Exactly what the upstream server sent — the bytes, unmodified. | `meteopack fetch` | Yes, if you are willing to download again. Nothing else can be rebuilt without it. |
| `packed/`   | One analysis-ready dataset per product: harmonised names, units and time axis. This is the layer to read. | `meteopack pack` | Yes — rebuildable from `raw/`. |
| `training/` | Datasets built *from* `packed/` and shaped for model training. | `meteopack derive` | Yes — rebuildable from `packed/`. |
| `_bookkeeping/` | Bookkeeping: what has been downloaded, what has been built, what failed. No observations. | every command | Yes, but then the pipeline no longer knows what it already did and will redo it. |

Each directory has its own README with the detail.

Nothing here is edited in place by hand. Every stage rewrites whole files and
moves them into position, so a reader always sees a complete file.
"""

_RAW = """# raw/ — as downloaded, byte for byte

Layout: `raw/<source>/<product>/...`, below which the upstream server's own
layout is kept. Nothing here is renamed, converted or cleaned, so a file can
always be compared against what the provider still serves.

Some sources arrive as one file per observation time. Those are stored as a tar
per day (`raw_layout.archive: tar` in the product config) because hundreds of
thousands of tiny files are what actually breaks a shared filesystem.

This is the only layer that cannot be rebuilt. Everything else is a function of
it.
"""

_CANONICAL = """# packed/ — the datasets you read

Layout: `packed/<source>/<product>/v<n>/`, one store per product, versioned
so a schema change does not overwrite the old one.

Two kinds of store, named by `store.kind` in the product config:

**points** — station observations, as partitioned Parquet:

    observations/year=2024/data.parquet   the observations
    stations/part-*.parquet               station id, lat, lon, height, name
    manifest.json                         what built this, and from which config

Read it with DuckDB or Polars over `observations/**/*.parquet`. A partition is
replaced on write, never appended to, so re-running a job cannot duplicate
rows. Station coordinates live in `stations/`, not on every observation row;
join on `station_id`, which is text.

**gridded** — fields on a fixed grid, as Zarr v3 under `dataset.zarr/`:

    <variable>/        one field, (time, y, x), integer codes plus a calibration
    time/              nanoseconds since 1970-01-01, UTC
    x/ y/              projected cell centres; lat/ lon/ are per-cell degrees
    is_missing_frame/  no source file covered this time step
    is_corrupt_frame/  a file existed but could not be read
    source_layout_id/  which upstream generation produced this frame
    source_record_id/  row in the provenance table
    manifest.json      (beside dataset.zarr, not inside it)

The time axis is regular and grows forward as later windows are packed; every
frame is written into the slot its timestamp belongs to. Frames are chunked one
per chunk inside day-sized shards.

Fields are stored as integers with a fixed calibration, so a reader must decode
them: `value = scale_factor * stored + add_offset`, with two reserved codes.
`calibration_missing_data` means the instrument measured nothing there and
`calibration_out_of_image` means the pixel is outside the product footprint —
different facts, kept apart. Both are on each array's attributes.

`manifest.json` records the config the store was built from. `meteopack
validate <source>` checks a store against it.
"""

_DERIVED = """# training/ — built from packed/, shaped for a model

Layout: `training/<name>/v<n>/`, one per entry in `configs/derive/`.

These are not new observations. Each one is a reshaping of `packed/` for a
specific consumer — a model, a plot, an evaluation — and exists so that work is
done once rather than in every training run. Examples: a dense
station × time matrix, spatial patches cut around a target, train/val/test
splits.

Delete any of these freely; `meteopack derive <name>` rebuilds it. If a derived
dataset disagrees with `packed/`, `packed/` is right.
"""

_CATALOG = """# _bookkeeping/ — no data, just what has been done

Three tables, each a *directory* of Parquet files rather than one file:

| table     | one row per | used for |
|-----------|-------------|----------|
| `jobs/`   | unit of work at one stage | skipping what is already done on a re-run |
| `files/`  | raw file fetched | finding which file covers which time |
| `stores/` | store built | what exists, from which config, how many records |

**Why the odd filenames.** Each writing process owns one file, named
`<when>-<host>-<pid>-t<task>-<nonce>.parquet`: the time it was first written,
the machine, the process, the Slurm array task, and a nonce so two writers in
one process cannot collide. Many processes write at once — a Slurm array is
hundreds — and a process that only ever rewrites its own file can never corrupt
another's or be caught half-written. Reads glob the whole directory and keep
the newest row per key, so duplicates across files are expected and harmless.

Do not edit these by hand. To read them:

    meteopack catalog <source>          # a summary
    duckdb -c "select * from read_parquet('_bookkeeping/jobs/*.parquet')"

Deleting `_bookkeeping/` loses no observations. The pipeline simply forgets what it
has already done, and will redo it.
"""

_LAYERS: dict[str, str] = {
    "raw": _RAW,
    "packed": _CANONICAL,
    "training": _DERIVED,
    "catalog": _CATALOG,
}


def store_readme(cfg, store_path: Path | str, info: dict | None = None) -> Path | None:
    """Write a README into one store directory.

    The data root README explains the layers; this explains one dataset, and it
    is the file somebody reads when the store has been copied somewhere else
    with no config beside it. Regenerated on every pack.
    """
    path = Path(store_path) / README
    kind = cfg.store.kind
    lines = [
        f"# {cfg.store_id}",
        "",
        (cfg.description or "").strip() or f"{cfg.source} {cfg.product}.",
        "",
        f"- source: `{cfg.source}`   product: `{cfg.product}`   version: {cfg.version}",
        f"- cadence: {cfg.cadence_seconds} s",
    ]
    if cfg.grid_id:
        lines.append(f"- grid: `{cfg.grid_id}`")
    if info:
        if info.get("time_start"):
            lines.append(f"- covers: {info['time_start']} .. {info['time_end']}")
        if info.get("shape"):
            lines.append(f"- shape: {tuple(info['shape'])} (time, y, x)")
    lines += ["", "## Variables", ""]
    if kind == "gridded":
        lines += ["| name | units | stored as | decode | was called |",
                  "|---|---|---|---|---|"]
        for var in cfg.variables:
            enc = var.encoding or {}
            stored = enc.get("dtype", "float32")
            decode = (f"`{enc['scale']} * v + {enc.get('offset', 0.0)}`"
                      if enc.get("scale") is not None else "as stored")
            lines.append(
                f"| `{var.name}` | {var.units} | {stored} | {decode} | "
                f"{('`' + var.legacy_name + '`') if var.legacy_name else '-'} |"
            )
        lines += [
            "",
            "Integer fields reserve two codes, on each array's attributes as",
            "`calibration_missing_data` and `calibration_out_of_image`: the first means",
            "the instrument measured nothing there, the second that the pixel is outside",
            "the product footprint. They are different facts; do not merge them.",
            "",
            "## Per-frame provenance",
            "",
            "`is_missing_frame` - no source file covered this time step.",
            "`is_corrupt_frame` - a file existed but could not be read.",
            "`source_layout_id` - which upstream generation produced the frame.",
            "",
            "Both read as missing imagery, so only these arrays tell them apart.",
            "",
            "## Reading it",
            "",
            "```python",
            "import xarray as xr",
            f"ds = xr.open_zarr('{Path(store_path).name}/dataset.zarr', consolidated=False)",
            "```",
        ]
    else:
        lines += ["| name | units | standard name |", "|---|---|---|"]
        for var in cfg.variables:
            lines.append(f"| `{var.name}` | {var.units} | {var.standard_name or '-'} |")
        lines += [
            "",
            "## Reading it",
            "",
            "```python",
            "import duckdb",
            "duckdb.sql(\"select * from read_parquet('observations/**/*.parquet')\")",
            "```",
            "",
            "Station coordinates are in `stations/`, joined on `station_id` (text).",
        ]
    text = "\n".join(lines) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    except OSError as exc:
        logger.debug("could not write %s: %s", path, exc)
        return None
    return path


def write(root: Path | str) -> list[Path]:
    """Refresh the README in `root` and in each layer directory that exists.

    Best effort: a read-only or group-owned data root is a normal thing to hand
    someone, and failing to write a README must never fail a pipeline run.
    """
    root = Path(root)
    written: list[Path] = []
    for path, text in [(root / README, _ROOT)] + [
        (root / name / README, body) for name, body in _LAYERS.items()
    ]:
        if path.parent != root and not path.parent.is_dir():
            continue
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_text() == text:
                continue
            path.write_text(text)
            written.append(path)
        except OSError as exc:
            logger.debug("could not write %s: %s", path, exc)
    return written
