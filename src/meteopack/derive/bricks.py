"""Canonical gridded store(s) -> a ``(sample, T, y, x)`` brick for training.

Layout under ``<root>/derived/<name>/v<n>/``::

    brick.zarr/                 # one array per variable, (sample, time, y, x)
    _shards/shard-000007.json   # "this slot range is written" markers
    manifest.json

Why it is shaped like this:

*One code path for full-field and patches.* The writer only ever copies
``spec.tile`` out of a frame. ``cfg.tile is None`` makes that tile the whole
grid, so a full-field dataset is a patch dataset with one patch — there is no
branch anywhere in this module on "is this full-field".

*Pre-shuffled on disk.* A training loader that reads slot 0, 1, 2 … is already
reading a shuffled stream, so it needs no random access into a compressed
store. Slot ``j`` holds ``spec.order[j]``; the permutation is derived from the
spec fingerprint, so every process computes the same one.

*Shard = unit of work = unit of resume = zarr shard.* ``cfg.shard_samples``
sets all three. A task owns a contiguous slot range, fills it in memory and
writes it in one region write, which is why disjoint tasks never touch the same
zarr shard and why re-running a task overwrites its slots instead of appending.
Peak memory is therefore ``shard_samples x n_variables x T x tile_y x tile_x x
4 bytes`` — that is the knob to turn when a full-field dataset runs out of RAM.

``gridded`` is imported inside the functions, not at module scope: the canonical
writer is a separate module with its own release cadence, and a lazy import
keeps a plan/split run from depending on it at all.
"""
from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from meteopack.core.config import Layer, layer_path
from meteopack.core.store import manifest as manifest_mod
from meteopack.derive.base import DeriveSpec, Sample, Shard, sample_table

DEFAULT_LAYER: Layer = "training"
SOURCE_LAYER: Layer = "packed"

BRICK_DIRNAME = "brick.zarr"
SHARD_DIRNAME = "_shards"

#: Sample geometry, written once by :func:`initialise` and never by a shard.
INDEX_ARRAYS = ("sample_index", "t0", "y0", "x0", "start_time")

_SAMPLE_DIM = "sample"
_DIMS = (_SAMPLE_DIM, "time", "y", "x")


def store_path(spec: DeriveSpec, root: str | os.PathLike[str], *, layer: Layer = DEFAULT_LAYER
               ) -> Path:
    """Directory holding the brick, its manifest and its split index."""
    return layer_path(Path(root), layer, *spec.store_id.split("/"))


def brick_path(spec: DeriveSpec, root: str | os.PathLike[str], *, layer: Layer = DEFAULT_LAYER
               ) -> Path:
    return store_path(spec, root, layer=layer) / BRICK_DIRNAME


def _marker(spec: DeriveSpec, root, index: int, *, layer: Layer) -> Path:
    return store_path(spec, root, layer=layer) / SHARD_DIRNAME / f"shard-{index:06d}.json"


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _probe_variables(
    spec: DeriveSpec,
    source_root: Path,
    *,
    source_layer: Layer,
    variables: Sequence[str] | None,
) -> dict[str, str]:
    """Map variable -> input store id, reading one window per input to find out.

    A variable offered by two inputs is an error rather than a silent
    precedence rule: which radar's reflectivity ended up in the brick is not
    something a training run should have to guess.

    The probe reads an *empty* time range: that returns the store's variables
    without touching a chunk, and asking for a named variable instead would make
    an input that simply does not carry it an error rather than a non-match.
    """
    from meteopack.core.store import gridded

    wanted = set(variables) if variables is not None else None
    start, _ = spec.window(spec.sample(0))
    owner: dict[str, str] = {}
    for store_id in spec.input_store_ids:
        ds = gridded.read(
            store_id,
            source_root,
            time_range=(start, start),
            variables=None,
            layer=source_layer,
        )
        for name in ds.data_vars:
            name = str(name)
            if wanted is not None and name not in wanted:
                continue
            if name in owner:
                raise ValueError(
                    f"variable {name!r} is offered by both {owner[name]!r} and {store_id!r}"
                )
            owner[name] = store_id
    if wanted is not None:
        missing = sorted(wanted - set(owner))
        if missing:
            raise ValueError(f"inputs {list(spec.input_store_ids)} do not provide {missing}")
    if not owner:
        raise ValueError(f"no variables found in {list(spec.input_store_ids)}")
    return dict(sorted(owner.items()))


def initialise(
    spec: DeriveSpec,
    root: str | os.PathLike[str],
    *,
    variables: Sequence[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
    source_layer: Layer = SOURCE_LAYER,
    layer: Layer = DEFAULT_LAYER,
    overwrite: bool = False,
) -> Path:
    """Create the brick and write the sample index. Run once, before the shards.

    Idempotent: re-running without ``overwrite`` leaves existing data alone, so
    an array task that finds the store already there can carry on. Shard markers
    are cleared on ``overwrite``, which is what makes a rebuild a rebuild.
    """
    store = store_path(spec, root, layer=layer)
    path = store / BRICK_DIRNAME
    source_root = Path(source_root) if source_root is not None else Path(root)

    if overwrite:
        for marker in sorted((store / SHARD_DIRNAME).glob("shard-*.json")):
            marker.unlink()
    store.mkdir(parents=True, exist_ok=True)

    existing = _open_group(path, mode="a") if path.exists() and not overwrite else None
    if existing is not None and existing.attrs.get("fingerprint") != spec.fingerprint:
        raise ValueError(
            f"{path} was built from a different spec "
            f"({existing.attrs.get('fingerprint')} != {spec.fingerprint}); pass overwrite=True"
        )
    if existing is not None and "variables" in existing.attrs:
        owner = dict(existing.attrs["variables"])
    else:
        owner = _probe_variables(
            spec, source_root, source_layer=source_layer, variables=variables
        )

    group = zarr.open_group(path, mode="w" if overwrite or existing is None else "a",
                            zarr_format=3)
    n, (t, ty, tx) = spec.n_samples, spec.sample_shape
    chunks = (1, t, ty, tx)
    shards = (min(spec.shard_samples, n), t, ty, tx)
    for name in owner:
        if name in group:
            array = group[name]
            if tuple(array.shape) != (n, t, ty, tx):
                raise ValueError(f"{name} exists with shape {array.shape}, expected {(n,t,ty,tx)}")
            continue
        group.create_array(
            name,
            shape=(n, t, ty, tx),
            chunks=chunks,
            shards=shards,
            dtype="float32",
            fill_value=float("nan"),
            dimension_names=_DIMS,
        )

    table = sample_table(spec)
    for name in INDEX_ARRAYS:
        values = table[name]
        array = group.create_array(
            name,
            shape=(n,),
            chunks=(n,),
            dtype="int64",
            dimension_names=(_SAMPLE_DIM,),
            overwrite=True,
        )
        array[:] = values.astype("int64")

    group.attrs.update(
        {
            "store_id": spec.store_id,
            "fingerprint": spec.fingerprint,
            "variables": owner,
            "inputs": list(spec.input_store_ids),
            "grid_id": spec.cfg.grid_id,
            "tile": list(spec.tile),
            "n_timesteps": spec.cfg.n_timesteps,
            "cadence_seconds": spec.cfg.cadence_seconds,
            "shard_samples": spec.shard_samples,
            "timeline_start": spec.timeline.start.isoformat(),
            "timeline_end": spec.timeline.end.isoformat(),
        }
    )
    _write_manifest(spec, store, layer=layer, variables=list(owner))
    return path


def _write_manifest(spec: DeriveSpec, store: Path, *, layer: Layer, variables: list[str]) -> None:
    existing = manifest_mod.read(store)
    entry = manifest_mod.Manifest.new(
        spec.store_id,
        layer,
        "bricks",
        spec.cfg,
        n_records=spec.n_samples,
        time_min=spec.timeline.start.isoformat(),
        time_max=spec.timeline.end.isoformat(),
        inputs=list(spec.input_store_ids),
        extra={
            "fingerprint": spec.fingerprint,
            "tile": list(spec.tile),
            "n_timesteps": spec.cfg.n_timesteps,
            "shard_samples": spec.shard_samples,
            "n_shards": spec.n_shards,
            "variables": variables,
        },
    )
    if existing is not None:
        entry.created_at = existing.created_at
    manifest_mod.write(store, entry)


def _open_group(path: Path, *, mode: str = "r") -> zarr.Group | None:
    try:
        return zarr.open_group(path, mode=mode)
    except (FileNotFoundError, KeyError, ValueError):
        return None


def _frames(ds: xr.Dataset, name: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """``(times, values)`` for one variable, as ``(n,)`` and ``(n, ny, nx)``."""
    if name not in ds.variables:
        raise KeyError(f"input did not return variable {name!r}; got {list(ds.data_vars)}")
    da = ds[name]
    dims = [str(d) for d in da.dims]
    if "time" not in dims:
        raise ValueError(f"{name} has dims {dims}, expected a time dimension")
    spatial = [d for d in dims if d != "time"]
    if len(spatial) != 2:
        raise ValueError(f"{name} has dims {dims}, expected (time, y, x)")
    values = np.asarray(da.transpose("time", *spatial).values, dtype="float32")
    if values.shape[1:] != shape:
        raise ValueError(f"{name} frames are {values.shape[1:]}, expected {shape}")
    return np.asarray(ds["time"].values), values


def write_shard(
    spec: DeriveSpec,
    shard: Shard | int,
    root: str | os.PathLike[str],
    *,
    source_root: str | os.PathLike[str] | None = None,
    source_layer: Layer = SOURCE_LAYER,
    layer: Layer = DEFAULT_LAYER,
    overwrite: bool = True,
) -> int:
    """Fill one shard's slots. Returns the number of slots written (0 if skipped).

    Safe to run concurrently with other shards of the same brick, and safe to
    re-run: the slot range is written as a whole, so a repeat is an overwrite,
    never an append.
    """
    from meteopack.core.store import gridded

    shard = spec.shard(shard) if isinstance(shard, int) else shard
    store = store_path(spec, root, layer=layer)
    path = store / BRICK_DIRNAME
    marker = _marker(spec, root, shard.index, layer=layer)
    if marker.exists() and not overwrite:
        return 0

    if not path.exists():
        raise FileNotFoundError(f"no brick at {path}; call initialise() first")
    group = _open_group(path, mode="a")
    if group is None:
        raise FileNotFoundError(f"no brick at {path}; call initialise() first")
    if group.attrs.get("fingerprint") != spec.fingerprint:
        raise ValueError(f"{path} was built from a different spec; refusing to write")
    owner: dict[str, str] = dict(group.attrs["variables"])
    source_root = Path(source_root) if source_root is not None else Path(root)

    t, ty, tx = spec.sample_shape
    buffers = {
        name: np.full((len(shard), t, ty, tx), np.nan, dtype="float32") for name in owner
    }
    by_window: dict[int, list[tuple[int, Sample]]] = {}
    for offset, sample in enumerate(shard.samples):
        by_window.setdefault(sample.t0, []).append((offset, sample))

    for store_id in sorted(set(owner.values())):
        names = [n for n, s in owner.items() if s == store_id]
        for t0, members in sorted(by_window.items()):
            start, end = spec.window(members[0][1])
            ds = gridded.read(
                store_id,
                source_root,
                time_range=(start, end),
                variables=names,
                layer=source_layer,
            )
            for name in names:
                times, values = _frames(ds, name, spec.grid.shape)
                rows = spec.timeline.index_of(times) - t0
                keep = (rows >= 0) & (rows < t)
                if not keep.any():
                    continue
                rows, values = rows[keep], values[keep]
                for offset, sample in members:
                    buffers[name][offset, rows] = values[
                        :, sample.y0 : sample.y0 + ty, sample.x0 : sample.x0 + tx
                    ]

    for name, buffer in buffers.items():
        group[name][shard.start : shard.stop] = buffer

    _write_atomic(
        marker,
        {
            "shard": shard.index,
            "slots": [shard.start, shard.stop],
            "fingerprint": spec.fingerprint,
            "written_at": datetime.now(UTC).isoformat(timespec="seconds"),
        },
    )
    return len(shard)


def build(
    spec: DeriveSpec,
    root: str | os.PathLike[str],
    *,
    shards: Iterable[int] | None = None,
    task: int = 0,
    n_tasks: int = 1,
    variables: Sequence[str] | None = None,
    source_root: str | os.PathLike[str] | None = None,
    source_layer: Layer = SOURCE_LAYER,
    layer: Layer = DEFAULT_LAYER,
    overwrite: bool = False,
) -> Path:
    """Initialise if needed, then write this task's shards, skipping finished ones.

    ``overwrite`` rebuilds, so it is refused for an array job: every task would
    wipe the store the others are writing into. Rebuild by calling
    :func:`initialise` once with ``overwrite=True``, then the array.
    """
    if overwrite and n_tasks > 1:
        raise ValueError(
            "overwrite=True rebuilds the whole store; call initialise(overwrite=True) once "
            "from the planner, then run the array with overwrite=False"
        )
    path = brick_path(spec, root, layer=layer)
    if overwrite or not path.exists():
        initialise(
            spec,
            root,
            variables=variables,
            source_root=source_root,
            source_layer=source_layer,
            layer=layer,
            overwrite=overwrite,
        )
    indices = list(shards) if shards is not None else spec.shard_indices(task, n_tasks)
    for index in indices:
        write_shard(
            spec,
            index,
            root,
            source_root=source_root,
            source_layer=source_layer,
            layer=layer,
            overwrite=overwrite,
        )
    return path


def finished_shards(spec: DeriveSpec, root: str | os.PathLike[str], *,
                    layer: Layer = DEFAULT_LAYER) -> list[int]:
    directory = store_path(spec, root, layer=layer) / SHARD_DIRNAME
    if not directory.is_dir():
        return []
    out = []
    for marker in sorted(directory.glob("shard-*.json")):
        payload = json.loads(marker.read_text())
        if payload.get("fingerprint") == spec.fingerprint:
            out.append(int(payload["shard"]))
    return sorted(out)


def is_complete(spec: DeriveSpec, root: str | os.PathLike[str], *,
                layer: Layer = DEFAULT_LAYER) -> bool:
    return finished_shards(spec, root, layer=layer) == list(range(spec.n_shards))


def read(
    spec: DeriveSpec,
    root: str | os.PathLike[str],
    *,
    slots: slice | Sequence[int] | None = None,
    variables: Sequence[str] | None = None,
    layer: Layer = DEFAULT_LAYER,
) -> xr.Dataset:
    """Read slots back as an ``(sample, time, y, x)`` dataset.

    Eager on purpose: this is for inspection and tests. A training loader reads
    the zarr arrays directly, in slot order, which is the point of the layout.
    """
    path = brick_path(spec, root, layer=layer)
    group = _open_group(path, mode="r")
    if group is None:
        raise FileNotFoundError(f"no brick at {path}")
    index = np.arange(spec.n_samples)[slots] if slots is not None else np.arange(spec.n_samples)
    index = np.asarray(index, dtype="int64")
    names = list(variables) if variables is not None else list(dict(group.attrs["variables"]))
    data = {name: (_DIMS, np.asarray(group[name])[index]) for name in names}
    coords = {
        name: (_SAMPLE_DIM, np.asarray(group[name])[index]) for name in INDEX_ARRAYS
    }
    coords["start_time"] = (_SAMPLE_DIM, coords["start_time"][1].astype("datetime64[ns]"))
    ds = xr.Dataset(data, coords=coords)
    ds.attrs.update(dict(group.attrs))
    return ds


def info(spec: DeriveSpec, root: str | os.PathLike[str], *, layer: Layer = DEFAULT_LAYER) -> dict:
    """What is in this brick and how much of it is written."""
    path = brick_path(spec, root, layer=layer)
    group = _open_group(path, mode="r")
    if group is None:
        raise FileNotFoundError(f"no brick at {path}")
    done = finished_shards(spec, root, layer=layer)
    return {
        "store_id": spec.store_id,
        "path": str(path),
        "n_samples": spec.n_samples,
        "sample_shape": spec.sample_shape,
        "tile": spec.tile,
        "variables": dict(group.attrs["variables"]),
        "shards_done": len(done),
        "n_shards": spec.n_shards,
        "complete": done == list(range(spec.n_shards)),
        "fingerprint": spec.fingerprint,
    }


__all__ = [
    "BRICK_DIRNAME",
    "DEFAULT_LAYER",
    "INDEX_ARRAYS",
    "brick_path",
    "build",
    "finished_shards",
    "info",
    "initialise",
    "is_complete",
    "read",
    "store_path",
    "write_shard",
]
