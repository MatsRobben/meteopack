"""Small shared helpers for dataset builders."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import yaml


def read_config(path: Path) -> dict:
    with path.open() as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a YAML mapping")
    return config


def fingerprint(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sample_index(times: np.ndarray, status: np.ndarray, *, history: int, forecast: int,
                 stride: int, origin: str, cadence_seconds: int):
    """Return starts and split, plus auditable rejection counts; all times are UTC ns."""
    import polars as pl

    n = history + forecast
    if history < 1 or forecast < 1 or stride < 1:
        raise ValueError("history, forecast and stride must be positive")
    if len(times) != len(status):
        raise ValueError("time and status lengths differ")
    ns = times.astype('datetime64[ns]').astype('int64')
    step = cadence_seconds * 1_000_000_000
    if np.any(np.diff(ns) != step):
        raise ValueError("source timeline is not regular at the configured cadence")
    origin_ns = np.datetime64(origin.replace('Z', ''), 'ns').astype('int64')
    starts = np.arange(max(0, len(times) - n + 1), dtype=np.int64)
    starts = starts[(ns[starts] - origin_ns) % (stride * step) == 0]
    day = times.astype('datetime64[D]')
    is_val = day == times.astype('datetime64[M]').astype('datetime64[D]')
    bad = np.concatenate(([0], np.cumsum(status != 0)))
    changed = np.concatenate(([0], np.cumsum(is_val[1:] != is_val[:-1])))
    frame_ok = (bad[starts + n] - bad[starts]) == 0
    split_ok = (changed[starts + n - 1] - changed[starts]) == 0
    keep = frame_ok & split_ok
    selected = starts[keep]
    split = np.where(is_val[selected], 'validation', 'train')
    result = pl.DataFrame({
        'sample_id': selected,
        'start_idx': selected,
        'source_start_idx': selected,  # caller adjusts for source range
        'issue_time': times[selected + history - 1].astype('datetime64[us]'),
        'split': split,
    })
    counts = {'candidates': int(len(starts)), 'accepted': int(keep.sum()),
              'invalid_frame': int((~frame_ok).sum()),
              'split_crossing': int((~split_ok).sum()),
              'train': int((split == 'train').sum()),
              'validation': int((split == 'validation').sum())}
    return result, counts
