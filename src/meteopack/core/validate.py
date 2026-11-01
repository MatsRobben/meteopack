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
    import polars as pl

    from meteopack.core.store import points

    report = Report(store_id=cfg.store_id, kind="points")
    path = points.store_path(root, cfg.store_id, layer=layer)
    report.add("store exists", path.exists(), str(path))
    if not path.exists():
        return report
    _check_manifest(report, path, cfg)

    df = points.read(cfg.store_id, root, layer=layer)
    report.stats["rows"] = df.height
    report.add("not empty", df.height > 0, f"{df.height} rows")
    if df.height == 0:
        return report

    report.add("has time column", "time" in df.columns, ", ".join(df.columns))
    report.add("has station_id", "station_id" in df.columns)

    n_dup = df.select(["time", "station_id"]).is_duplicated().sum()
    report.add("no duplicate (time, station)", n_dup == 0, f"{n_dup} duplicates")

    # Observations without positions are unusable for anything spatial, and the
    # failure is silent downstream: a dense matrix just drops those stations.
    try:
        meta = points.read_stations(cfg.store_id, root, layer=layer)
    except Exception:  # noqa: BLE001 - absence is the thing being reported
        meta = None
    observed = set(df["station_id"].unique().to_list())
    known = set(meta["station_id"].to_list()) if meta is not None and len(meta) else set()
    covered = observed & known
    report.add(
        "stations have metadata",
        bool(observed) and len(covered) == len(observed),
        f"{len(covered)}/{len(observed)} observed stations have a lat/lon row"
        + ("" if known else " (station table is empty - did pack write it?)"),
    )

    t0, t1 = df["time"].min(), df["time"].max()
    report.stats["time_min"], report.stats["time_max"] = str(t0), str(t1)
    report.stats["stations"] = df["station_id"].n_unique()

    # Off-cadence rows mean the parser and the configured cadence disagree.
    step_us = cfg.cadence_seconds * 1_000_000
    offset = df["time"].cast(pl.Int64) % step_us
    off = int((offset != 0).sum())
    report.add("all times on cadence", off == 0, f"{off} rows off the {cfg.cadence_seconds}s grid")

    for var in cfg.variables:
        if var.name not in df.columns:
            continue
        col = df[var.name]
        n_ok = int(col.is_not_null().sum())
        report.stats[f"{var.name}:filled"] = f"{n_ok}/{df.height}"
        if n_ok and (var.valid_min is not None or var.valid_max is not None):
            lo = col.min()
            hi = col.max()
            within = (var.valid_min is None or lo >= var.valid_min) and (
                var.valid_max is None or hi <= var.valid_max
            )
            report.add(
                f"{var.name} within valid range",
                within,
                f"[{lo}, {hi}] vs [{var.valid_min}, {var.valid_max}] {var.units}",
            )
    return report


def validate_gridded(cfg: ProductConfig, root: Path, *, layer: Layer = "packed") -> Report:
    import numpy as np

    from meteopack.core.store import gridded

    report = Report(store_id=cfg.store_id, kind="gridded")
    path = gridded.store_path(root, cfg.store_id, layer=layer)
    report.add("store exists", path.exists(), str(path))
    if not path.exists():
        return report
    _check_manifest(report, path, cfg)

    ds = gridded.read(cfg.store_id, root, layer=layer)
    report.stats["dims"] = dict(ds.sizes)
    report.add("has time dim", "time" in ds.sizes)
    if "time" in ds.sizes and ds.sizes["time"] > 1:
        deltas = np.diff(ds["time"].values).astype("timedelta64[s]").astype(int)
        uniform = bool(np.all(deltas == cfg.cadence_seconds))
        report.add(
            "time axis is regular at the configured cadence",
            uniform,
            f"steps seen: {sorted(set(deltas.tolist()))[:4]} vs {cfg.cadence_seconds}s",
        )
    report.add("grid_id recorded", bool(ds.attrs.get("grid_id")), str(ds.attrs.get("grid_id")))

    from meteopack.core.encoding import Encoding

    encodings = {v.name: Encoding.from_config(v.encoding, name=v.name) for v in cfg.variables}
    ranges = {v.name: (v.valid_min, v.valid_max) for v in cfg.variables}
    any_data = False
    for name, arr in ds.data_vars.items():
        if name in gridded.PROVENANCE:
            continue
        values = arr.values
        encoding = encodings.get(str(name))
        if encoding is not None:
            # An integer array has no NaN, so counting non-NaN would report
            # every sentinel as data and a store of pure fill as 100% full.
            codes = values
            real = ~np.isin(codes, np.asarray(sorted(encoding.reserved), dtype=codes.dtype))
            report.stats[f"{name}:missing"] = int((codes == encoding.missing).sum())
            report.stats[f"{name}:out_of_image"] = int((codes == encoding.out_of_image).sum())
            physical = encoding.decode(codes[real]) if real.any() else np.empty(0, "float32")
        else:
            real = np.isfinite(values)
            physical = values[real]
        filled, total = int(real.sum()), int(arr.size)
        any_data = any_data or filled > 0
        report.stats[f"{name}:filled"] = f"{filled}/{total} ({100 * filled / max(total, 1):.1f}%)"
        lo, hi = ranges.get(str(name), (None, None))
        if physical.size and (lo is not None or hi is not None):
            seen = (float(physical.min()), float(physical.max()))
            ok = (lo is None or seen[0] >= lo) and (hi is None or seen[1] <= hi)
            report.add(
                f"{name} within valid range",
                ok,
                f"[{seen[0]:g}, {seen[1]:g}] vs [{lo}, {hi}]",
            )

    report.add("some data present", any_data)
    _check_frame_provenance(report, ds)
    return report


def _check_frame_provenance(report: Report, ds) -> None:
    """Count the frames the store says it never got, and sanity-check them.

    A missing frame and a corrupt one are different upstream failures and a
    frame cannot be both: it either had no file or had one that would not read.
    """
    import numpy as np

    from meteopack.core.store import gridded

    present = [n for n in gridded.PROVENANCE if n in ds.variables]
    if not present:
        report.add("per-frame provenance recorded", False,
                   "no is_missing_frame / is_corrupt_frame arrays")
        return
    missing = np.asarray(ds["is_missing_frame"].values) if "is_missing_frame" in ds.variables else None
    corrupt = np.asarray(ds["is_corrupt_frame"].values) if "is_corrupt_frame" in ds.variables else None
    if missing is not None:
        report.stats["frames:missing"] = int(missing.sum())
    if corrupt is not None:
        report.stats["frames:corrupt"] = int(corrupt.sum())
    if missing is not None and corrupt is not None:
        both = int((missing & corrupt).sum())
        report.add("no frame is both missing and corrupt", both == 0, f"{both} frames")
    if "source_layout_id" in ds.variables:
        layouts = np.asarray(ds["source_layout_id"].values)
        seen = np.unique(layouts[~missing]) if missing is not None else np.unique(layouts)
        report.stats["frames:layouts"] = sorted(int(v) for v in seen)


def validate(cfg: ProductConfig, root: Path, *, layer: Layer = "packed") -> Report:
    """Dispatch on the store kind the config declares."""
    if cfg.store.kind == "points":
        return validate_points(cfg, root, layer=layer)
    return validate_gridded(cfg, root, layer=layer)
