# meteopack

Ingest, canonicalise and pack multi-source meteorological data into
analysis-ready stores: KNMI radar and in-situ, OPERA, EUMETSAT (SEVIRI, FCI,
Lightning Imager), HARMONIE-AROME and ICON, and European station networks.

Replaces the untracked `data_raw` workspace, which grew four ingestion
frameworks and five store conventions. The design is in
`docs/history/` (see the redesign plan); this README is the short version.

## Three layers

| Layer | What it is | Who reads it |
|---|---|---|
| `data/raw/<source>/<product>/YYYY/MM/DD/` | exactly what was fetched, immutable, checksummed | only `parse` |
| `data/canonical/<source>/<product>/vN/` | CF names and units, one named grid, regular time axis — zarr v3 for fields, Parquet for points | analysis, derive |
| `data/derived/<dataset>/vN/` | training-shaped: pre-shuffled bricks, station matrices, splits | models |

Raw is never read by a model. The catalog (`data/catalog/`) holds
`files.parquet`, `stores.parquet` and `jobs.parquet` — the job ledger is what
makes every stage re-runnable and resumable.

## One contract per source

```python
class Source(Protocol):
    name: str
    def list(self, window, cfg) -> list[Unit]: ...    # what exists remotely
    def fetch(self, unit, raw_root) -> RawFile: ...   # idempotent, atomic, sha256
    def parse(self, raw, cfg) -> xr.Dataset | pl.DataFrame: ...  # canonical
```

`pack` is **not** per source: `core/store/gridded.py` and `core/store/points.py`
are shared, driven by config. Adding a source means a listing, a reader, and a
product YAML — not a fourth copy of a packing pipeline.

## Layout

```
src/meteopack/
  core/      grids, timeline, catalog, http, config, validate, store/
  sources/   base.py (the contract), registry.py, one module per source
  derive/    bricks, dense_stations, splits
  jobs/      plan (expand -> units, skip what the ledger has), slurm helpers
configs/     sources/, products/, derive/, grids.yaml
workflow/    Snakefile — pack + derive DAG
slurm/       thin wrappers: account, partition, env. No logic.
```

## Status

Scaffold only. The `Source` contract, the grid registry entry for the KNMI
product grid, and the CLI surface are real; everything else is a documented
stub. Nothing is ported from `data_raw` yet.

## Running

```bash
uv sync
uv run meteopack --help
uv run pytest
```

Cluster conventions (ICIS/DAS, `/dev/shm` for scratch, `--cpus-per-task`
matching real thread counts) are in `docs/runbooks/`.
