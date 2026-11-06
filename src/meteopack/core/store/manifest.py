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
from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
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


#: Config fields that describe a store without changing a single stored value:
#: prose, CF discovery metadata, and retention (which nothing enforces). Editing
#: one of these made `validate` call the store stale and ask for a rebuild,
#: although `gridded.refresh_attrs` exists to republish exactly these.
_DESCRIPTIVE = frozenset({"description", "attrs", "retention"})
_DESCRIPTIVE_VARIABLE = frozenset({"long_name", "legacy_name", "standard_name"})


#: `extra` keys that change how a pack runs, never what it stores.
_RUNTIME_EXTRA = frozenset({"readme", "spill", "spill_compression", "partition_rss_factor",
                            "write_workers", "write_threads", "max_spill_gib"})


def content_hash(cfg: BaseModel | dict[str, Any]) -> str:
    """`config_hash` over only the settings that change what the store holds."""
    payload = cfg.model_dump(mode="json") if isinstance(cfg, BaseModel) else dict(cfg)
    payload = {k: v for k, v in payload.items() if k not in _DESCRIPTIVE}
    if isinstance(payload.get("extra"), dict):
        # README prose a config may carry for its store, and knobs for how a
        # pack runs rather than what it writes: tuning them must not make a
        # store "stale" and block a windowed re-pack.
        payload["extra"] = {k: v for k, v in payload["extra"].items()
                            if k not in _RUNTIME_EXTRA}
    if isinstance(payload.get("variables"), list):
        payload["variables"] = [
            {k: v for k, v in var.items() if k not in _DESCRIPTIVE_VARIABLE}
            if isinstance(var, dict) else var
            for var in payload["variables"]
        ]
    return config_hash(payload)


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
    #: One line about what the store holds, from the config. Optional and
    #: defaulted: manifests written before this field existed must still read.
    description: str | None = None
    n_records: int = 0
    time_min: str | None = None
    time_max: str | None = None
    inputs: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)
    #: `content_hash` of the config at pack time. Optional: manifests written
    #: before it existed are judged by `config_hash` instead.
    content_hash: str | None = None

    @classmethod
    def new(cls, store_id: str, layer: str, kind: str, cfg: BaseModel | dict, **kw) -> Manifest:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        described = cfg.model_dump() if isinstance(cfg, BaseModel) else cfg
        return cls(
            store_id=store_id, layer=layer, kind=kind,
            description=kw.pop("description", None) or described.get("description"),
            config_hash=config_hash(cfg), content_hash=content_hash(cfg),
            code_version=code_version(),
            created_at=now, updated_at=now, **kw,
        )


def write(store_path: Path, manifest: Manifest) -> Path:
    """Atomically write the manifest beside the store."""
    store_path = Path(store_path)
    store_path.mkdir(parents=True, exist_ok=True)
    dest = store_path / MANIFEST_NAME
    manifest.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
    tmp = dest.with_suffix(".json.part")
    tmp.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True))
    os.replace(tmp, dest)
    return dest


def read(store_path: Path) -> Manifest | None:
    dest = Path(store_path) / MANIFEST_NAME
    if not dest.exists():
        return None
    payload = json.loads(dest.read_text())
    # A manifest written by a newer version may carry fields this one does not
    # know. Dropping them beats refusing to open the store.
    known = {f.name for f in fields(Manifest)}
    return Manifest(**{k: v for k, v in payload.items() if k in known})


def is_stale(store_path: Path, cfg: BaseModel | dict) -> bool:
    """True if the store was built from a config that changes what it holds.

    Edits to descriptions, CF metadata or retention do not count. A manifest
    from before `content_hash` existed is judged by the full config hash.
    """
    m = read(store_path)
    if m is None:
        return True
    if m.content_hash:
        return m.content_hash != content_hash(cfg)
    return m.config_hash != config_hash(cfg)
