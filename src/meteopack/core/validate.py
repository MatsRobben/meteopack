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

    for name, arr in ds.data_vars.items():
        finite = int(np.isfinite(arr.values).sum())
        total = int(arr.size)
        report.stats[f"{name}:filled"] = f"{finite}/{total} ({100 * finite / max(total, 1):.1f}%)"
    report.add("some data present", any("0/" not in v for v in report.stats.values() if isinstance(v, str)))
    return report


def validate(cfg: ProductConfig, root: Path, *, layer: Layer = "packed") -> Report:
    """Dispatch on the store kind the config declares."""
    if cfg.store.kind == "points":
        return validate_points(cfg, root, layer=layer)
    return validate_gridded(cfg, root, layer=layer)
