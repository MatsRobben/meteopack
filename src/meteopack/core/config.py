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

#: The three layers data moves through, named after the command that fills
#: them. "canonical" and "derived" said nothing about what was in them, which
#: is the complaint that prompted the rename; "packed" is what `meteopack pack`
#: writes and the word the previous pipeline used, and "training" says what the
#: derived datasets are for.
Layer = Literal["raw", "packed", "training"]

#: What each layer used to be called on disk. A data root written before the
#: rename keeps working: `layer_path` falls back to the old directory when it
#: exists and the new one does not, so nobody has to migrate a 600 GB tree to
#: read it. New stores are always written under the new names.
LEGACY_LAYER_DIRS: dict[str, str] = {"packed": "canonical", "training": "derived"}

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


#: Canonical names that are also CF standard names, verified individually
#: against the CF standard-name table rather than assumed from the spelling.
#: A variable whose name is in here gets `standard_name` for free; anything
#: else must set it explicitly, because claiming CF compliance for a name the
#: table does not define is worse than claiming nothing. Names ending in a
#: qualifier (`_min`, `_1h`, `_10cm`) are deliberately absent: CF expresses
#: those with cell methods and coordinates, not by inventing a name.
CF_STANDARD_NAMES: frozenset[str] = frozenset({
    "air_pressure",
    "air_pressure_at_sea_level",
    "air_temperature",
    "cloud_area_fraction",
    "cloud_base_height",
    "dew_point_temperature",
    "precipitation_amount",
    "relative_humidity",
    "soil_temperature",
    "surface_diffuse_downwelling_shortwave_flux_in_air",
    "surface_downwelling_longwave_flux_in_air",
    "surface_downwelling_shortwave_flux_in_air",
    "surface_snow_thickness",
    "visibility_in_air",
    "water_vapor_pressure",
    "wet_bulb_temperature",
    "wind_from_direction",
    "wind_speed",
    "wind_speed_of_gust",
})

#: Where our canonical name and the CF one differ, usually because CF appends
#: `_in_air` to a flux. Keeping our shorter name as the column and the CF one
#: as the attribute is the point of having both.
CF_ALIASES: dict[str, str] = {
    "surface_downwelling_shortwave_flux": "surface_downwelling_shortwave_flux_in_air",
    "surface_downwelling_longwave_flux": "surface_downwelling_longwave_flux_in_air",
    "surface_diffuse_downwelling_shortwave_flux":
        "surface_diffuse_downwelling_shortwave_flux_in_air",
    "sunshine_duration": "duration_of_sunshine",
    "precipitation_duration": "duration_of_precipitation",
}


class Variable(BaseModel):
    """One canonical variable: the name we use, and how to get there."""

    model_config = {"extra": "forbid"}

    name: str
    source_key: str
    units: str
    long_name: str | None = None
    #: The CF standard name, when the quantity has one. This is what makes a
    #: radar field and a gauge field recognisably the same quantity, which is
    #: the only reason a packed layer spanning several sources is worth having.
    standard_name: str | None = None
    #: What an earlier generation of this store called the variable. Carried in
    #: the array attrs so code written against that store can be ported by
    #: grep rather than by reading two schemas side by side.
    legacy_name: str | None = None
    scale: float | None = None
    offset: float | None = None
    dtype: str = "float32"
    valid_min: float | None = None
    valid_max: float | None = None
    #: Store this variable as integers with a fixed calibration instead of
    #: float32. See meteopack.core.encoding: halves the store for imagery that
    #: is already integer on disk, and keeps `missing` and `out_of_image`
    #: apart, which NaN cannot.
    encoding: dict | None = None

    @property
    def cf_standard_name(self) -> str | None:
        """The CF standard name, explicit or inferred from a verified list."""
        if self.standard_name:
            return self.standard_name
        alias = CF_ALIASES.get(self.name, self.name)
        return alias if alias in CF_STANDARD_NAMES or alias in CF_ALIASES.values() else None

    @property
    def cf_attrs(self) -> dict[str, str]:
        """The CF attributes this variable contributes to its array."""
        out = {"units": self.units, "long_name": self.long_name or self.name}
        standard = self.cf_standard_name
        if standard:
            out["standard_name"] = standard
        if self.legacy_name:
            out["legacy_name"] = self.legacy_name
        return out


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
    #: One line, for a human opening the data root. Copied into the store's
    #: manifest so the store says what it is without the config beside it.
    description: str | None = None
    cadence_seconds: Annotated[int, Field(gt=0)] = 600
    grid_id: str | None = None
    variables: list[Variable] = Field(default_factory=list)
    store: StoreConfig
    retention: RetentionConfig = RetentionConfig()
    raw_layout: RawLayout = RawLayout()
    schema_fingerprint: str | None = None
    #: CF discovery metadata written onto the store's root group. Separate from
    #: `extra` because it is a published part of the dataset rather than a knob:
    #: title, institution, source, history, references, comment.
    attrs: dict[str, str] = Field(default_factory=dict)
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
    description: str | None = None
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
    """`<root>/<layer>/<parts...>`, honouring a pre-rename directory if present."""
    current = root.joinpath(layer, *parts)
    legacy_name = LEGACY_LAYER_DIRS.get(str(layer))
    if legacy_name and not root.joinpath(layer).is_dir():
        legacy = root.joinpath(legacy_name)
        if legacy.is_dir():
            return legacy.joinpath(*parts)
    return current
