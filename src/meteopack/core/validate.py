"""Store-level checks, run after packing and before anything trusts a store.

These are the questions that distinguish "the job exited 0" from "the data is
usable": is the time axis what the config asked for, did anything actually land,
and are the values physically possible. Cheap enough to run every time.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meteopack.core.config import Layer, ProductConfig
from meteopack.core.store import manifest


@dataclass(slots=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


@dataclass(slots=True)
class Report:
    store_id: str
    kind: str
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(c.ok for c in self.checks)

    def add(self, name: str, ok: bool, detail: str = "") -> None:
        self.checks.append(Check(name, bool(ok), detail))

    def text(self) -> str:
        lines = [f"{self.store_id} [{self.kind}] {'OK' if self.ok else 'FAILED'}"]
        lines += [f"  {'pass' if c.ok else 'FAIL'}  {c.name}: {c.detail}" for c in self.checks]
        lines += [f"  stat  {k} = {v}" for k, v in self.stats.items()]
        return "\n".join(lines)


def _check_manifest(report: Report, path: Path, cfg: ProductConfig) -> None:
    m = manifest.read(path)
    report.add("manifest present", m is not None, str(path / manifest.MANIFEST_NAME))
    if m is not None:
        fresh = not manifest.is_stale(path, cfg)
        report.add(
            "config matches manifest",
            fresh,
            "store was built from the current config" if fresh
            else f"built from a different config ({m.config_hash}); rebuild or bump version",
        )


def validate_points(cfg: ProductConfig, root: Path, *, layer: Layer = "packed") -> Report:
    """Check a points store with aggregates, never by loading it.

    This used to read the whole store into one frame and then call
    `.is_duplicated()` on it. On the DWD archive - 762 million rows - that
    reached 102 GB resident and had to be killed, and because this node uses
    strict overcommit it took the machine with it: nothing could fork, not even
    a shell. The same mistake as the gridded validator, left in place because
    that one was fixed as an instance rather than as a pattern.

    Every number below is a streaming SQL aggregate. 762 million rows now
    summarise in 93 s at 3.2 GB.
    """
    from meteopack.core.store import points

    report = Report(store_id=cfg.store_id, kind="points")
    path = points.store_path(root, cfg.store_id, layer=layer)
    report.add("store exists", path.exists(), str(path))
    if not path.exists():
        return report
    _check_manifest(report, path, cfg)

    try:
        summary = points.summarise(cfg.store_id, root, cfg, layer=layer)
    except Exception as exc:  # noqa: BLE001 - an unreadable store is the finding
        report.add("store is readable", False, f"{type(exc).__name__}: {exc}")
        return report

    rows = int(summary["rows"])
    report.stats["rows"] = rows
    report.add("not empty", rows > 0, f"{rows} rows")
    if rows == 0:
        return report

    report.stats["partitions"] = summary.get("partitions")
    report.stats["stations"] = int(summary["stations"])
    report.stats["time_min"] = str(summary["time_min"])
    report.stats["time_max"] = str(summary["time_max"])

    columns = summary["columns"]
    report.add("has time column", "time" in columns, ", ".join(columns))
    report.add("has station_id", "station_id" in columns)

    duplicates = int(summary["duplicates"])
    report.add("no duplicate (time, station)", duplicates == 0, f"{duplicates} duplicates")

    off_grid = int(summary.get("off_grid") or 0)
    report.add("all times on cadence", off_grid == 0,
               f"{off_grid} rows off the {cfg.cadence_seconds}s grid")

    # Observations without positions are unusable for anything spatial, and the
    # failure is silent downstream: a dense matrix just drops those stations.
    _check_station_metadata(report, cfg, root, layer, summary.get("station_ids") or [])

    # Every configured variable must be in the store and have at least one
    # value. A selector that matches nothing upstream produces a column of
    # nulls, or no column, and passed silently: air_pressure was 0 of 237,600
    # rows in one quickstart.
    for var in cfg.variables:
        if var.name not in columns:
            report.add(f"{var.name} has values", False,
                       "configured but not in the store - no raw file that carries it was "
                       "packed (was its group fetched?)")
        elif not int(summary.get(f"filled__{var.name}") or 0):
            report.add(f"{var.name} has values", False, f"0/{rows} rows filled")

    for name in summary["value_columns"]:
        filled = int(summary.get(f"filled__{name}") or 0)
        report.stats[f"{name}:filled"] = f"{filled}/{rows}"
        lo, hi = summary.get(f"min__{name}"), summary.get(f"max__{name}")
        var = next((v for v in cfg.variables if v.name == name), None)
        if var is None or lo is None or hi is None:
            continue
        if var.valid_min is None and var.valid_max is None:
            continue
        ok = ((var.valid_min is None or float(lo) >= var.valid_min)
              and (var.valid_max is None or float(hi) <= var.valid_max))
        report.add(f"{name} within valid range", ok,
                   f"[{float(lo):g}, {float(hi):g}] vs [{var.valid_min}, {var.valid_max}] {var.units}")
    return report


def _check_station_metadata(report: Report, cfg: ProductConfig, root: Path,
                            layer: Layer, observed: list[str]) -> None:
    """Whether every station that reported has a position - by id, not by count.

    This compared counts, so a table with the right number of rows for the
    wrong stations passed. The reporting ids come from a streaming DISTINCT,
    so the check is still two small reads.
    """
    from meteopack.core.store import points

    try:
        meta = points.read_stations(cfg.store_id, root, layer=layer)
    except Exception:  # noqa: BLE001 - absence is the thing being reported
        meta = None
    placed = set() if meta is None else set(
        meta.filter(meta["lat"].is_not_null() & meta["lon"].is_not_null())["station_id"]
        .cast(str).to_list()
    )
    unplaced = sorted(set(observed) - placed)
    detail = f"{len(observed) - len(unplaced)} of {len(observed)} reporting stations have lat/lon"
    if unplaced:
        detail += f"; without: {', '.join(unplaced[:8])}" + (" ..." if len(unplaced) > 8 else "")
    if not placed:
        detail += " (station table is empty - did pack write it?)"
    report.add("stations have metadata", bool(observed) and not unplaced, detail)


def validate_gridded(cfg: ProductConfig, root: Path, *, layer: Layer = "packed",
                     sample_frames: int = 64) -> Report:
    """Check a gridded store without reading it all.

    The first version called `gridded.read()` with no time range. On the real
    rtcor archive that is 772,129 x 765 x 700 across three variables - over a
    terabyte - and validation died of memory before saying anything. The
    store's own docstring says a whole-store read is not meant to fit; the
    validator was the one caller ignoring it.

    So: metadata and the per-frame provenance arrays are read in full, because
    they are one value per time step and that is the point of them. The imagery
    is sampled - `sample_frames` frames spread evenly across the axis - which
    is enough to catch an all-missing variable or a range violation, and is
    honest about being a sample rather than claiming to have checked every
    pixel of 15 GB.
    """
    import numpy as np
    import zarr

    from meteopack.core.encoding import Encoding
    from meteopack.core.store import gridded

    report = Report(store_id=cfg.store_id, kind="gridded")
    path = gridded.store_path(root, cfg.store_id, layer=layer)
    report.add("store exists", path.exists(), str(path))
    if not path.exists():
        return report
    _check_manifest(report, path, cfg)

    group = zarr.open_group(gridded.zarr_path(root, cfg.store_id, layer=layer), mode="r")
    attrs = dict(group.attrs)
    names = [n for n, _ in group.arrays() if n not in gridded._NON_FIELD]
    if not names:
        report.add("has data arrays", False, "the store holds no fields")
        return report

    shape = tuple(int(n) for n in group[names[0]].shape)
    report.stats["dims"] = dict(zip(gridded.DIMS, shape, strict=True))
    report.add("has a time axis", shape[0] > 0, f"{shape[0]} steps")

    stamps = np.asarray(group[gridded.TIME_NAME][:]).astype("datetime64[ns]")
    if stamps.size > 1:
        deltas = np.unique(np.diff(stamps).astype("timedelta64[s]").astype(int))
        uniform = bool(np.all(deltas == cfg.cadence_seconds))
        report.add("time axis is regular at the configured cadence", uniform,
                   f"steps seen: {sorted(deltas.tolist())[:4]} vs {cfg.cadence_seconds}s")
    report.add("grid_id recorded", bool(attrs.get("grid_id")), str(attrs.get("grid_id")))
    mapping = attrs.get(gridded.GRID_MAPPING_NAME)
    report.add(
        "georeferenced", bool(mapping),
        # The detail is printed whether the check passes or fails, so it has to
        # read correctly both ways; "no grid mapping" next to a `pass` is the
        # kind of output that teaches you to stop reading it.
        str((mapping or {}).get("proj4_params", "")) or "no grid mapping on the root group",
    )
    report.stats["time_start"] = str(stamps[0]) if stamps.size else "-"
    report.stats["time_end"] = str(stamps[-1]) if stamps.size else "-"

    _check_frame_provenance_arrays(report, group, shape[0])

    encodings = {v.name: Encoding.from_config(v.encoding, name=v.name) for v in cfg.variables}
    ranges = {v.name: (v.valid_min, v.valid_max) for v in cfg.variables}
    picks = _sample_positions(group, shape[0], sample_frames)
    report.stats["frames:sampled"] = len(picks)
    any_data = False
    for name in names:
        array = group[name]
        encoding = encodings.get(name)
        filled = total = missing = out_of_image = 0
        lo_seen = hi_seen = None
        for i in picks:
            frame = np.asarray(array[int(i)])
            total += frame.size
            if encoding is not None:
                real = ~np.isin(frame, np.asarray(sorted(encoding.reserved), dtype=frame.dtype))
                missing += int((frame == encoding.missing).sum())
                out_of_image += int((frame == encoding.out_of_image).sum())
                values = encoding.decode(frame[real]) if real.any() else None
            else:
                real = np.isfinite(frame)
                values = frame[real] if real.any() else None
            filled += int(real.sum())
            if values is not None and values.size:
                lo = float(np.nanmin(values))
                hi = float(np.nanmax(values))
                lo_seen = lo if lo_seen is None else min(lo_seen, lo)
                hi_seen = hi if hi_seen is None else max(hi_seen, hi)
        any_data = any_data or filled > 0
        report.stats[f"{name}:filled"] = (
            f"{filled}/{total} ({100 * filled / max(total, 1):.1f}% of sampled frames)"
        )
        if encoding is not None:
            report.stats[f"{name}:missing"] = missing
            report.stats[f"{name}:out_of_image"] = out_of_image
        want_lo, want_hi = ranges.get(name, (None, None))
        if lo_seen is not None and (want_lo is not None or want_hi is not None):
            ok = ((want_lo is None or lo_seen >= want_lo)
                  and (want_hi is None or hi_seen <= want_hi))
            report.add(f"{name} within valid range", ok,
                       f"[{lo_seen:g}, {hi_seen:g}] vs [{want_lo}, {want_hi}]")

    report.add("some data present", any_data)
    return report


def _sample_positions(group, n_times: int, wanted: int) -> list[int]:
    """Frame indices to inspect: spread across the axis, skipping known gaps.

    Sampling only written frames keeps the fill statistics meaningful - an
    archive with a 0.3% gap would otherwise report that gap as missing data in
    the variable rather than as an absent frame, which the provenance arrays
    already say.
    """
    import numpy as np

    if n_times <= wanted:
        candidates = np.arange(n_times)
    else:
        candidates = np.linspace(0, n_times - 1, wanted).astype(int)
    if "is_missing_frame" in {n for n, _ in group.arrays()}:
        flags = np.asarray(group["is_missing_frame"][:])
        kept = [int(i) for i in candidates if not flags[i]]
        if kept:
            return kept
    return [int(i) for i in candidates]


def _check_frame_provenance_arrays(report: Report, group, n_times: int) -> None:
    """Per-frame provenance, read in full: one value per time step is cheap."""
    import numpy as np

    from meteopack.core.store import gridded

    present = {n for n, _ in group.arrays()}
    if not ({"is_missing_frame", "is_corrupt_frame"} & present):
        report.add("per-frame provenance recorded", False,
                   "no is_missing_frame / is_corrupt_frame arrays")
        return
    missing = np.asarray(group["is_missing_frame"][:]) if "is_missing_frame" in present else None
    corrupt = np.asarray(group["is_corrupt_frame"][:]) if "is_corrupt_frame" in present else None
    if missing is not None:
        report.stats["frames:missing"] = int(missing.sum())
        report.stats["frames:written"] = int(n_times - missing.sum())
    if corrupt is not None:
        report.stats["frames:corrupt"] = int(corrupt.sum())
    if missing is not None and corrupt is not None:
        both = int((missing & corrupt).sum())
        report.add("no frame is both missing and corrupt", both == 0, f"{both} frames")
    if "source_record_id" in present and missing is not None:
        _check_record_ids(report, group, np.asarray(group["source_record_id"][:]), missing)
    if "source_layout_id" in present:
        layouts = np.asarray(group["source_layout_id"][:])
        seen = layouts[~missing] if missing is not None else layouts
        counts = {int(v): int(c) for v, c in zip(*np.unique(seen, return_counts=True),
                                                 strict=True)}
        report.stats["frames:layouts"] = counts
    assert gridded.PROVENANCE


def _check_record_ids(report: Report, group, ids, missing) -> None:
    """Every written frame names a raw file, and that file is in the table.

    `source_record_id` was declared, documented as "row in the provenance
    table", and -1 on every frame of every store, with no table anywhere.
    """
    import numpy as np
    import polars as pl

    from meteopack.core.store import gridded

    written = ids[~missing]
    report.add("every written frame names its raw file", bool((written >= 0).all()),
               f"{int((written < 0).sum())} written frame(s) with no source_record_id")
    table = Path(str(group.store.root)).parent / gridded.PROVENANCE_TABLE
    if not table.exists():
        report.add("provenance table present", False, f"no {table.name} beside the store")
        return
    known = pl.read_parquet(table, columns=["source_record_id"])["source_record_id"].to_numpy()
    named = np.unique(written[written >= 0])
    unknown = np.setdiff1d(named, known)
    report.add("every source_record_id is in the provenance table", unknown.size == 0,
               f"{unknown.size} id(s) missing from {table.name}")
    report.stats["provenance:files"] = int(known.size)


def validate(cfg: ProductConfig, root: Path, *, layer: Layer = "packed") -> Report:
    """Dispatch on the store kind the config declares."""
    if cfg.store.kind == "points":
        return validate_points(cfg, root, layer=layer)
    return validate_gridded(cfg, root, layer=layer)
