"""Write a short README into the data root and into each layer.

The directory names (`raw`, `canonical`, `derived`, `catalog`) say what the
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
| `canonical/`| One analysis-ready dataset per product: harmonised names, units and time axis. This is the layer to read. | `meteopack pack` | Yes — rebuildable from `raw/`. |
| `derived/`  | Datasets built *from* `canonical/` for a particular use, mostly model training. | `meteopack derive` | Yes — rebuildable from `canonical/`. |
| `catalog/`  | Bookkeeping: what has been downloaded, what has been built, what failed. No observations. | every command | Yes, but then the pipeline no longer knows what it already did and will redo it. |

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

_CANONICAL = """# canonical/ — the datasets you read

Layout: `canonical/<source>/<product>/v<n>/`, one store per product, versioned
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

**gridded** — fields on a fixed grid, as Zarr v3:

    <variable>/        one array, (time, y, x)
    time/              the time axis
    manifest.json

The time axis is regular and fixed when the store is created, and every frame
is written into the slot its timestamp belongs to. Frames are chunked one per
chunk inside day-sized shards.

`manifest.json` records the config the store was built from. `meteopack
validate <source>` checks a store against it.
"""

_DERIVED = """# derived/ — built from canonical/, for a purpose

Layout: `derived/<name>/v<n>/`, one per entry in `configs/derive/`.

These are not new observations. Each one is a reshaping of `canonical/` for a
specific consumer — a model, a plot, an evaluation — and exists so that work is
done once rather than in every training run. Examples: a dense
station × time matrix, spatial patches cut around a target, train/val/test
splits.

Delete any of these freely; `meteopack derive <name>` rebuilds it. If a derived
dataset disagrees with `canonical/`, `canonical/` is right.
"""

_CATALOG = """# catalog/ — bookkeeping, no data

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
    duckdb -c "select * from read_parquet('catalog/jobs/*.parquet')"

Deleting `catalog/` loses no observations. The pipeline simply forgets what it
has already done, and will redo it.
"""

_LAYERS: dict[str, str] = {
    "raw": _RAW,
    "canonical": _CANONICAL,
    "derived": _DERIVED,
    "catalog": _CATALOG,
}


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
