"""Per-store manifest: what built this store, from what, and when.

The point is answering "is this store still valid?" without reading it. A store
whose config hash no longer matches its config was built from different
settings, so `pack` can refuse to append to it rather than silently mixing two
conventions into one array — the failure mode that made the previous generation
of stores untrustworthy.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel

MANIFEST_NAME = "manifest.json"


def config_hash(cfg: BaseModel | dict[str, Any]) -> str:
    """Stable hash of a config.

    Serialised with sorted keys so field order and model version churn do not
    change the digest; only actual setting changes do.
    """
    payload = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else cfg
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def code_version(repo_root: Path | None = None) -> str:
    """Short git describe of the building code, or 'unknown' outside a checkout.

    Recorded so a store can be traced to the code that wrote it; never fails the
    build, because jobs must run from exported trees too.
    """
    root = repo_root or Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "describe", "--always", "--dirty"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or "unknown"
    except (subprocess.SubprocessError, OSError):
        return "unknown"


@dataclass(slots=True)
class Manifest:
    store_id: str
    layer: str
    kind: str
    config_hash: str
    code_version: str
    created_at: str
    updated_at: str
    n_records: int = 0
    time_min: str | None = None
    time_max: str | None = None
    inputs: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, store_id: str, layer: str, kind: str, cfg: BaseModel | dict, **kw) -> Manifest:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return cls(
            store_id=store_id, layer=layer, kind=kind,
            config_hash=config_hash(cfg), code_version=code_version(),
            created_at=now, updated_at=now, **kw,
        )


def write(store_path: Path, manifest: Manifest) -> Path:
    """Atomically write the manifest beside the store."""
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    dest = store_path / MANIFEST_NAME
    manifest.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tmp = dest.with_suffix(".json.part")
    tmp.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    os.replace(tmp, dest)
    return dest


def read(store_path: Path) -> Manifest | None:
    dest = Path(store_path) / MANIFEST_NAME
    if not dest.exists():
        return None
    return Manifest(**json.loads(dest.read_text()))


def is_stale(store_path: Path, cfg: BaseModel | dict) -> bool:
    """True if the store was built from a different config (or has no manifest)."""
    m = read(store_path)
    return m is None or m.config_hash != config_hash(cfg)
