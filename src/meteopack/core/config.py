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

from pydantic import BaseModel, Field, field_validator, model_validator

#: The three layers data moves through, named after the command that fills
#: them. "canonical" and "derived" said nothing about what was in them, which
#: is the complaint that prompted the rename; "packed" is what `meteopack pack`
#: writes and the word the previous pipeline used, and "training" says what the
#: derived datasets are for.
Layer = Literal["raw", "packed", "training"]

# One taxonomy for source code, raw/packed stores, and bookkeeping. Keep this
# here rather than in each product config: every product from a source belongs
# to the same data family.
SOURCE_CATEGORIES: dict[str, str] = {
    "belgium": "stations",
    "dmi": "stations",
    "dwd": "stations",
    "ea": "stations",
    "knmi_insitu": "stations",
    "meteofrance": "stations",
    "midas": "stations",
    "wow_nl": "stations",
    "knmi_radar": "radar",
    "eumetsat_seviri": "satellite",
}


def category_for_source(source: str) -> str | None:
    """Return the enforced data family for a built-in source, if known."""
    return SOURCE_CATEGORIES.get(source)

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
    #: Extra HTTP statuses this upstream uses to mean "slow down" rather than
    #: "no". 408, 429 and 5xx are always retried; EA's server also answers a
    #: share of requests with 403 under load, and the same URL succeeds a
    #: moment later (measured 2026-11-06: 1,084 of 6,580 at 8 workers).
    retry_status: list[int] = Field(default_factory=list)


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
    #: The value a "no echo" pixel takes. KNMI radar declares its missing code
    #: as 0 in several generations while 0 is the commonest pixel in the frame
    #: (a dry country, no echo top); `missing_data_policy` decides that such a
    #: code means no echo, and this is the value it then gets. It has to be
    #: stated, not derived: decoding the code with the file's own calibration
    #: gives 0 only where the offset is 0, and KNMI shifted the offset by one
    #: count between generations - which put no-echo echo tops at -0.102 km
    #: and split no-echo reflectivity across -31.5 and -32.0 dBZ. Unset, a
    #: zero sentinel is honoured as missing data.
    no_echo: float | None = None
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
    #: Row order inside each points partition file. Unset keeps the historical
    #: (time, station_id). An irregular store sorts by (station_id, time) so a
    #: station's reports are contiguous - what every per-station lag, run or
    #: regularisation reads.
    sort_by: list[str] | None = None
    #: Rows per parquet row group in a points store; unset is polars' default.
    #: Smaller groups let a reader skip more by their min/max statistics.
    row_group_rows: Annotated[int, Field(gt=0)] | None = None


class ProductConfig(BaseModel):
    """One (source, product) pair: what to pull, how to name it, where it lands."""

    model_config = {"extra": "forbid"}

    source: str
    product: str
    version: int = 1
    #: One line, for a human opening the data root. Copied into the store's
    #: manifest so the store says what it is without the config beside it.
    description: str | None = None
    #: The product's fixed time step. `null` declares an IRREGULAR points
    #: product: reports at whatever time each station sent them, nothing
    #: snapped, nothing dropped for being off a grid. It has to be said
    #: explicitly - leaving the field out still means the 600 s default - so a
    #: regular product can never lose its cadence check by omission.
    cadence_seconds: Annotated[int, Field(gt=0)] | None = 600
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

    @model_validator(mode="after")
    def _irregular_needs_points(self) -> ProductConfig:
        if self.cadence_seconds is None and self.store.kind != "points":
            raise ValueError("cadence_seconds: null (irregular) is only meaningful for a "
                             "points store; a gridded store is a regular time axis")
        return self

    @property
    def irregular(self) -> bool:
        """True for a product with no time step (`cadence_seconds: null`)."""
        return self.cadence_seconds is None

    @property
    def cadence(self) -> timedelta:
        if self.cadence_seconds is None:
            raise ValueError(f"{self.store_id} is irregular (cadence_seconds: null) and has "
                             f"no time step; derive a regular product from it instead")
        return timedelta(seconds=self.cadence_seconds)

    @property
    def store_id(self) -> str:
        return f"{self.source}/{self.product}/v{self.version}"

    @property
    def category(self) -> str:
        """Storage family; source/product IDs stay stable for the CLI/API."""
        return category_for_source(self.source) or "other"


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

    @property
    def category(self) -> str:
        source = self.inputs[0].split("/", 1)[0] if self.inputs else ""
        return category_for_source(source) or "other"


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


def layer_path(root: Path, layer: Layer, *parts: str,
               category: str | None = None) -> Path:
    """Resolve the canonical categorized path for one layer.

    Source store IDs remain ``source/product/vN``. Their physical path is
    ``layer/category/source/product/vN``. Derived stores pass their category
    explicitly because their public name is not necessarily a source name.
    """
    parts = tuple(str(part) for part in parts)
    if len(parts) == 1 and "/" in parts[0]:
        parts = tuple(p for p in parts[0].split("/") if p)
    if category is None and parts:
        category = category_for_source(parts[0])
    if category:
        return root.joinpath(layer, category, *parts)
    return root.joinpath(layer, *parts)
