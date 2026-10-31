"""Configuration models.

Everything that varies between sources lives here as data, not code: endpoints,
rate limits, how the raw layer is laid out, what gets kept, and how a store is
chunked. A new source should be a YAML file plus a listing and a reader.
"""
from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator

Layer = Literal["raw", "canonical", "derived"]

_ENV_DATA_ROOT = "METEOPACK_DATA_ROOT"


class RetentionRule(BaseModel):
    """What happens to one layer of one product.

    Storage is the binding constraint, so this is deliberately per layer and per
    product rather than a global policy. `after` names a stage that must have
    completed before the layer may be dropped.
    """

    model_config = {"extra": "forbid"}

    keep: bool = True
    after: Literal["parse", "pack", "derive"] | None = None
    days: int | None = Field(default=None, ge=0)
    tier: Literal["warm", "cold"] = "warm"

    @field_validator("after")
    @classmethod
    def _after_needs_drop(cls, v: str | None, info) -> str | None:
        if v is not None and info.data.get("keep", True):
            raise ValueError("`after` only applies when keep=false")
        return v


class RetentionConfig(BaseModel):
    model_config = {"extra": "forbid"}

    raw: RetentionRule = RetentionRule()
    canonical: RetentionRule = RetentionRule()
    derived: RetentionRule = RetentionRule()

    def for_layer(self, layer: Layer) -> RetentionRule:
        return getattr(self, layer)


class RawLayout(BaseModel):
    """How fetched units are stored.

    `archive: tar` exists because some feeds publish one file per observation —
    the KNMI 10-minute in-situ feed is one file per 10 minutes, which overwhelms
    the filesystem and is near-undownloadable since KNMI rate-limits on requests
    rather than bytes. Such sources write one tar per granularity bucket.
    """

    model_config = {"extra": "forbid"}

    archive: Literal["none", "tar"] = "none"
    granularity: Literal["day", "month", "year"] = "day"


class RateLimit(BaseModel):
    """Per-host politeness. Expressed in requests/second, because that is what
    upstream services actually meter."""

    model_config = {"extra": "forbid"}

    requests_per_second: float = Field(default=5.0, gt=0)
    max_workers: int = Field(default=8, ge=1)
    max_retries: int = Field(default=5, ge=0)
    backoff_seconds: float = Field(default=1.0, ge=0)


class Variable(BaseModel):
    """One canonical variable: the name we use, and how to get there."""

    model_config = {"extra": "forbid"}

    name: str
    source_key: str
    units: str
    long_name: str | None = None
    scale: float | None = None
    offset: float | None = None
    dtype: str = "float32"
    valid_min: float | None = None
    valid_max: float | None = None


class StoreConfig(BaseModel):
    model_config = {"extra": "forbid"}

    kind: Literal["gridded", "points"]
    chunks: dict[str, int] | None = None
    shard: dict[str, int] | None = None
    compressor: str = "zstd"
    compression_level: int = 3
    partition_by: list[str] = Field(default_factory=lambda: ["year"])


class ProductConfig(BaseModel):
    """One (source, product) pair: what to pull, how to name it, where it lands."""

    model_config = {"extra": "forbid"}

    source: str
    product: str
    version: int = 1
    cadence_seconds: Annotated[int, Field(gt=0)] = 600
    grid_id: str | None = None
    variables: list[Variable] = Field(default_factory=list)
    store: StoreConfig
    retention: RetentionConfig = RetentionConfig()
    raw_layout: RawLayout = RawLayout()
    schema_fingerprint: str | None = None
    extra: dict = Field(default_factory=dict)

    @property
    def cadence(self) -> timedelta:
        return timedelta(seconds=self.cadence_seconds)

    @property
    def store_id(self) -> str:
        return f"{self.source}/{self.product}/v{self.version}"


class SourceConfig(BaseModel):
    model_config = {"extra": "forbid"}

    name: str
    base_url: str | None = None
    rate_limit: RateLimit = RateLimit()
    credentials_env: list[str] = Field(default_factory=list)
    extra: dict = Field(default_factory=dict)


class DeriveConfig(BaseModel):
    """A training-shaped dataset built from one or more canonical stores.

    `tile` is what makes full-field and patch-based the same code path: a full
    frame is simply tile == the grid shape.
    """

    model_config = {"extra": "forbid"}

    name: str
    version: int = 1
    inputs: list[str]
    grid_id: str
    cadence_seconds: Annotated[int, Field(gt=0)] = 300
    n_timesteps: Annotated[int, Field(gt=0)] = 1
    tile: tuple[int, int] | None = None
    shard_samples: Annotated[int, Field(gt=0)] = 64
    splits: dict[str, list[str]] = Field(default_factory=dict)


def data_root(override: str | os.PathLike[str] | None = None, default: str | None = None) -> Path:
    """Resolve the data root: --data-root, then $METEOPACK_DATA_ROOT, then config.

    Deliberately not a constant — where the data lives is a deployment question,
    and the same code runs against a scratch copy in tests.
    """
    for candidate in (override, os.environ.get(_ENV_DATA_ROOT), default):
        if candidate:
            return Path(candidate).expanduser().resolve()
    raise RuntimeError(
        f"no data root: pass --data-root, set ${_ENV_DATA_ROOT}, or set it in the config"
    )


def layer_path(root: Path, layer: Layer, *parts: str) -> Path:
    return root.joinpath(layer, *parts)
