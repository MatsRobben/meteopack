"""KNMI real-time corrected radar composite (`RAD_NL25_RAC_RT`), 5-minute fields.

The first *gridded* source here, and the first one whose archive is already on
local disk: there is nothing to download, so `fetch` publishes the day's tar
into the raw layer by hardlink (falling back to a copy) and records a real
sha256, which keeps the ledger and the manifest honest without a second copy of
~35 GB.

Layout. One tar per day under `extra.archive_root`::

    RAD25_OPER_R___TARRRT__L2__<YYYYMMDD>T080500_<YYYYMMDD+1>T080000_0001.tar
      RAD_NL25_RAC_RT_<YYYYMMDDHHMM>.h5   x 288, 5 minutes apart

Six pieces of domain knowledge are worth stating explicitly, because each one
silently corrupts the store if you get it wrong:

* **A KNMI radar "day" runs 08:05 UTC to 08:00 UTC the next day**, not midnight
  to midnight — 288 frames, the first one five minutes after the nominal start.
  `extra.day_start` carries that offset, and `valid_time` on every unit is the
  *first frame* the tar holds, not a midnight. Two 2018 tars still use the older
  `T080000_...T075500` convention and are read from their own filenames rather
  than assumed.
* **Unit boundaries must be shard boundaries.** `store.shard.time` is 288 — one
  tar — so a store whose timeline starts at an 08:05 UTC instant puts each tar
  in exactly one shard and lets one Slurm task own it. A timeline starting at
  midnight splits every tar across two shards, and `core.store.gridded` is
  explicit that two tasks writing one shard lose frames. `list` refuses to
  pretend otherwise: every unit carries `valid_time`.
* **The product has drifted four times since 2018, including a rename of the
  variable itself.** Sampled across the archive: `image1` is `PRECIP_[MM]` until
  ~2019-06, `PRECIP_[MM]` alone (no second image) until ~2023-01, then
  `ACCUMULATION_[MM]` with a uint16 `QUALITY_[-]` beside it, and from ~2024-05
  the quality and adjustment images are uint8 on different scales. Image groups
  are therefore addressed by `image_geo_parameter`, never by `image1`/`image2` —
  the group index moves, the parameter name does not — and `source_key` lists
  every accepted spelling. `scan` reports every layout it sees and `parse`
  refuses to run against one that is not in `extra.known_layouts`, because a
  rename that goes unnoticed does not crash: it writes an all-NaN archive.
* **The layout also changes *within* a day.** KNMI narrows an individual frame
  to uint8 when the whole field fits — ten frames of 2019-06-01 are uint8 inside
  a day of uint16 — and those frames still quote `calibration_out_of_image` as
  65535, which no uint8 pixel can hold. :func:`representable_sentinel` clamps a
  sentinel to the dtype maximum for exactly this reason; read literally, the
  footprint mask of such a frame goes unmasked and ~104 000 out-of-image pixels
  become 2.55 mm of rain. It also means a layout `scan` has to sample *frames*,
  not days.
* **Calibration is read from the file, never hard-coded.** `calibration_formulas`
  reads `GEO = 0.010000 * PV + 0.000000` in some generations and
  `GEO=0.001*PV+-32.768` in others — same grammar, different spacing, and a
  signed offset written as `+-`. Both are parsed by :func:`parse_calibration`.
  The precipitation scale happens to be 0.01 throughout, but the quality scale
  moved from 1.52593e-05 to 0.004, so a hard-coded factor would be wrong by two
  orders of magnitude the day you reused this reader for another image.
* **Missing is not the same as no echo, and which sentinel means which moved.**
  See :func:`calibrate`. `calibration_out_of_image` (65535) is always
  `_FillValue`/NaN. `calibration_missing_data` is 0 in the pre-2023 generations,
  where it collides with a genuine dry pixel and therefore means *no echo* — a
  real 0 mm, not missing — and 65534 afterwards, where it is a sentinel outside
  the data range and genuinely means no data. The default policy
  (`extra.missing_data_policy: no_echo_if_zero`) reads that distinction off the
  file. Forcing either answer turns "no rain" into "no data", or the reverse,
  across the whole archive, so the policy is config rather than a constant.
* **Quality exists only from ~2023.** The pre-2023 generations ship no per-pixel
  quality array at all — `image2` there is `ADJUSTMENT_FACTOR_[DB]`, a gauge
  adjustment field, which is not a quality index and is not treated as one. A
  variable that resolves in no frame of a tar is dropped from the output rather
  than written as NaN, so backfilling 2019 leaves the quality array untouched at
  its fill value instead of overwriting it.

Verified against the real archive on 2026-10-31: `(765, 700)` uint16 throughout,
`geo_number_rows`/`geo_number_columns` and `geo_product_corners` matching
`knmi_1km_stereo` in every generation, 288 members per tar (200 in the first,
partial day).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tarfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import numpy as np

from meteopack.core import grids
from meteopack.core.config import ProductConfig, RateLimit, SourceConfig, Variable
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, TimeWindow, Unit

if TYPE_CHECKING:  # xarray is a caller-facing type here, not an import-time cost
    import xarray as xr

logger = logging.getLogger(__name__)

#: `RAD25_OPER_R___TARRRT__L2__<from>_<to>_<version>.tar`. The time tokens are
#: `\d+` rather than `\d{6}` on purpose: at least one archived name has a stray
#: digit (`20181220T0080000`), and a day that cannot be read from its own
#: filename is still a day — see `_time_of_day`.
ARCHIVE_RE = re.compile(
    r"^RAD25_OPER_R___TARRRT__L2__"
    r"(?P<from_date>\d{8})T(?P<from_time>\d+)_"
    r"(?P<to_date>\d{8})T(?P<to_time>\d+)"
    r"_(?P<version>\d+)\.tar$"
)

#: `RAD_NL25_RAC_RT_<YYYYMMDDHHMM>.h5`, one frame.
MEMBER_RE = re.compile(r"^RAD_NL25_RAC_RT_(?P<stamp>\d{12})\.h5$")

#: `GEO = <scale> * PV + <offset>`, with or without spaces, and with the offset
#: sign written either once (`+ 0.0`) or twice (`+-32.768`).
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
CALIBRATION_RE = re.compile(
    rf"^\s*GEO\s*=\s*(?P<scale>{_NUMBER})\s*\*\s*PV"
    rf"\s*(?P<sign>[+-])\s*(?P<offset>{_NUMBER})\s*$"
)

#: Offset of the first frame of a KNMI radar day from midnight UTC.
DEFAULT_DAY_START = "08:05"

#: What `calibration_missing_data` means. `no_echo_if_zero` reads it off the
#: file: a sentinel of 0 collides with a genuine dry pixel and means no echo,
#: anything else is a real missing-data sentinel.
MISSING_POLICIES = ("no_echo_if_zero", "no_echo", "missing")
DEFAULT_MISSING_POLICY = "no_echo_if_zero"

#: How far the product corners may drift from the registry before it is a
#: different grid. The float32 corners wobble in the 5th decimal between
#: generations; 1e-3 degrees is ~100 m, well under one pixel.
DEFAULT_CORNER_TOLERANCE = 1e-3

_IMAGE_PREFIX = "image"
_GEO_GROUP = "geographic"
_GEO_PARAMETER = "image_geo_parameter"
_CALIBRATION = "calibration"
_IMAGE_DATA = "image_data"
_SOURCE_KEY_SEP = "|"


class SchemaDriftError(RuntimeError):
    """An HDF5 frame's image layout is not one the config was pinned against."""


# ------------------------------------------------------------- HDF5 attribute IO


def _text(value: Any) -> str:
    """One HDF5 attribute as text. KNMI writes these as fixed-length bytes."""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace").strip()
    arr = np.asarray(value)
    if arr.dtype.kind == "S":
        return bytes(arr.ravel()[0]).decode("utf-8", "replace").strip()
    if arr.shape:
        return str(arr.ravel()[0]).strip()
    return str(value).strip()


def _scalar(value: Any) -> float:
    """One numeric HDF5 attribute. KNMI writes most of these as length-1 arrays."""
    return float(np.ravel(np.asarray(value))[0])


def parse_calibration(text: str) -> tuple[float, float]:
    """`"GEO = 0.01 * PV + 0.0"` -> `(0.01, 0.0)`.

    The scale is read from the file rather than pinned, because it differs
    between images and has changed between product generations. A formula this
    cannot read is an error, not a default: silently assuming 0.01 would write a
    plausible-looking archive in the wrong units.
    """
    m = CALIBRATION_RE.match(text)
    if not m:
        raise SchemaDriftError(
            f"cannot read calibration formula {text!r}; expected 'GEO = <scale> * PV + <offset>'"
        )
    offset = float(m.group("offset"))
    return float(m.group("scale")), offset if m.group("sign") == "+" else -offset


@dataclass(frozen=True, slots=True)
class ImageSpec:
    """One `imageN` group: what it holds and how to turn counts into units."""

    group: str
    geo_parameter: str
    scale: float
    offset: float
    missing: float | None
    out_of_image: float | None
    dtype: str
    shape: tuple[int, int]

    @property
    def descriptor(self) -> str:
        """Normalised identity of this image, for fingerprinting.

        Deliberately excludes the group name: `image2` is the quality field in
        one generation and the adjustment factor in another, so including it
        would report a variant every time KNMI inserted an image, while
        excluding it reports a variant exactly when the *content* changes.
        """
        return (
            f"{self.geo_parameter}|{self.dtype}|{self.shape[0]}x{self.shape[1]}"
            f"|scale={self.scale:.12g}|offset={self.offset:.12g}"
            f"|missing={self.missing}|out_of_image={self.out_of_image}"
        )


def _image_specs(frame: Any) -> dict[str, ImageSpec]:
    """`{geo_parameter: ImageSpec}` for one open HDF5 frame."""
    specs: dict[str, ImageSpec] = {}
    for group in sorted(k for k in frame if k.startswith(_IMAGE_PREFIX)):
        node = frame[group]
        if _IMAGE_DATA not in node or _CALIBRATION not in node:
            continue
        attrs = node[_CALIBRATION].attrs
        scale, offset = parse_calibration(_text(attrs["calibration_formulas"]))
        data = node[_IMAGE_DATA]
        parameter = _text(node.attrs[_GEO_PARAMETER])
        specs[parameter] = ImageSpec(
            group=group,
            geo_parameter=parameter,
            scale=scale,
            offset=offset,
            missing=_optional_scalar(attrs, "calibration_missing_data"),
            out_of_image=_optional_scalar(attrs, "calibration_out_of_image"),
            dtype=str(data.dtype),
            shape=(int(data.shape[0]), int(data.shape[1])),
        )
    return specs


def _optional_scalar(attrs: Any, name: str) -> float | None:
    return _scalar(attrs[name]) if name in attrs else None


# ----------------------------------------------------------------- fingerprints


def schema_fingerprint(descriptors: Iterable[str]) -> str:
    """Stable hash of one frame's image layout.

    Order-independent, because the group order is an implementation detail of
    the writer; a genuine addition, removal, rename, retyping or recalibration
    of an image does move it.
    """
    normalised = sorted({d.strip() for d in descriptors} - {""})
    return hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()[:16]


def product_fingerprint(layouts: Iterable[str]) -> str:
    """Stable hash of the whole set of layouts a product accepts."""
    joined = "|".join(sorted(set(layouts)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- config access


def _accepted_parameters(var: Variable) -> list[str]:
    """`source_key` is a `|`-separated list of accepted `image_geo_parameter`s.

    A list rather than one name because KNMI renamed the precipitation image
    from `PRECIP_[MM]` to `ACCUMULATION_[MM]` mid-archive without changing what
    it holds, and one canonical variable has to span both.
    """
    names = [p.strip() for p in str(var.source_key).split(_SOURCE_KEY_SEP)]
    names = [n for n in names if n]
    if not names:
        raise ValueError(
            f"variable {var.name!r} has an empty source_key; expected one or more "
            f"image_geo_parameter values, e.g. 'PRECIP_[MM]|ACCUMULATION_[MM]'"
        )
    return names


def wanted_variables(cfg: ProductConfig) -> list[tuple[Variable, list[str]]]:
    """`[(variable, accepted image_geo_parameters), ...]` for this product."""
    return [(var, _accepted_parameters(var)) for var in cfg.variables]


# -------------------------------------------------------------------- calibrate


def representable_sentinel(sentinel: float | None, dtype: np.dtype) -> float | None:
    """Clamp a sentinel the image dtype cannot hold down to that dtype's maximum.

    KNMI narrows individual frames to uint8 when the whole field fits — ten
    frames of 2019-06-01 are uint8 inside a day of uint16 — but leaves
    `calibration_out_of_image` at the uint16 value 65535, which no uint8 pixel
    can ever equal. Taken literally, the footprint mask of such a frame goes
    unmasked and ~104 000 out-of-image pixels read back as 2.55 mm of rain.

    Clamping to `iinfo(dtype).max` is what the product means: the sentinel is
    the top of the range in every generation, and the 2024 generation writes 255
    explicitly for its uint8 images. It costs the top code point of a uint8
    frame, which is unambiguous in practice — the value histogram of such a
    frame runs 0..223 and then jumps straight to the 255 mask.
    """
    if sentinel is None or dtype.kind not in "ui":
        return sentinel
    info = np.iinfo(dtype)
    if info.min <= sentinel <= info.max:
        return sentinel
    if sentinel < info.min:
        logger.debug("sentinel %s is below %s; no pixel can carry it", sentinel, dtype)
        return None
    logger.debug("sentinel %s does not fit %s; reading it as %s", sentinel, dtype, info.max)
    return float(info.max)


def calibrate(
    counts: np.ndarray, spec: ImageSpec, *, policy: str = DEFAULT_MISSING_POLICY
) -> np.ndarray:
    """Pixel counts -> geophysical float32, with the two sentinels resolved.

    The decided rule, in order:

    1. `calibration_out_of_image` (65535 in every generation here) is outside
       the radar footprint or otherwise unusable: NaN, the array fill value.
    2. `calibration_missing_data` is ambiguous by generation, so `policy`
       decides. `no_echo_if_zero` (the default) treats a sentinel of 0 as a
       real **0 mm — no echo, not missing**, because the pre-2023 product has no
       separate missing value and a dry pixel is simply 0; any other sentinel
       (65534 from 2023) sits outside the data range and is genuinely missing,
       so NaN. `no_echo` and `missing` force one answer for a feed that needs it.
    3. Everything else goes through the file's own `scale`/`offset`.

    Getting step 2 backwards converts every dry pixel in the archive into
    missing data, or every genuine gap into a confident zero. Both sentinels go
    through :func:`representable_sentinel` first, for the uint8 frames whose
    metadata still quotes the uint16 value.
    """
    if policy not in MISSING_POLICIES:
        raise ValueError(f"unknown missing_data_policy {policy!r}; use one of {MISSING_POLICIES}")
    # float64 first: 0.01 is not exact in binary, and rounding once to float32
    # at the end is closer to the true value than accumulating in float32.
    out = (counts.astype("float64") * spec.scale + spec.offset).astype("float32")
    out_of_image = representable_sentinel(spec.out_of_image, counts.dtype)
    missing = representable_sentinel(spec.missing, counts.dtype)
    if out_of_image is not None:
        out[counts == out_of_image] = np.float32("nan")
    if missing is not None and missing != out_of_image:
        # The policy reads the DECLARED sentinel, not the clamped one: whether a
        # frame means "no echo" is a property of the product generation, not of
        # how wide that frame happened to be written.
        no_echo = policy == "no_echo" or (policy == "no_echo_if_zero" and spec.missing == 0)
        out[counts == missing] = np.float32(0.0) if no_echo else np.float32("nan")
    return out


def _apply_variable(values: np.ndarray, var: Variable) -> np.ndarray:
    """Optional unit conversion, then the valid range — in canonical units.

    Mirrors the points sources: sentinels are already resolved, so scaling can
    never turn a sentinel into a plausible number, and `valid_min`/`valid_max`
    are expressed post-scale.
    """
    if var.scale is not None:
        values = values * np.float32(var.scale)
    if var.offset is not None:
        values = values + np.float32(var.offset)
    if var.valid_min is not None:
        values[values < var.valid_min] = np.float32("nan")
    if var.valid_max is not None:
        values[values > var.valid_max] = np.float32("nan")
    return values


# ------------------------------------------------------------------ time naming


def _hhmm(token: str) -> timedelta:
    """`"08:05"` -> a timedelta. Used for `extra.day_start`."""
    try:
        hours, _, minutes = token.partition(":")
        return timedelta(hours=int(hours), minutes=int(minutes or 0))
    except ValueError:
        raise ValueError(f"day_start {token!r} must be 'HH:MM'") from None


def _yyyymmdd(token: str) -> datetime:
    return datetime(int(token[:4]), int(token[4:6]), int(token[6:8]), tzinfo=UTC)


def _time_of_day(token: str, filename: str) -> timedelta | None:
    """`"080500"` -> 8h05m; `None` for anything else, with a warning.

    At least one archived name carries a stray digit (`20181220T0080000`). The
    *date* part of these names has never been wrong, so a bad time token means
    falling back to the configured day start rather than dropping the day.
    """
    if len(token) == 6 and token.isdigit():
        hour, minute, second = int(token[:2]), int(token[2:4]), int(token[4:6])
        if hour < 24 and minute < 60 and second < 60:
            return timedelta(hours=hour, minutes=minute, seconds=second)
    logger.warning(
        "knmi_radar: malformed time %r in %s; falling back to the configured day start",
        token,
        filename,
    )
    return None


def member_time(name: str) -> datetime | None:
    """`RAD_NL25_RAC_RT_201906010805.h5` -> that instant in UTC, else `None`."""
    m = MEMBER_RE.match(Path(name).name)
    if not m:
        return None
    stamp = m.group("stamp")
    try:
        return datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=UTC)
    except ValueError:
        logger.warning("knmi_radar: member %s has an impossible timestamp", name)
        return None


# ------------------------------------------------------------------- the source


class KnmiRadar:
    """KNMI 5-minute real-time corrected radar composite as a `meteopack` source.

    Public API::

        src = KnmiRadar(source_cfg, archive_root=...)
        src.list(window, product_cfg)    -> list[Unit]      (one per daily tar)
        src.fetch(unit, raw_root)        -> RawFile | None  (None == tar vanished)
        src.parse(raw_file, product_cfg) -> xarray.Dataset  (time, y, x) float32
        src.scan(raw_files)              -> dict            (schema drift report)
        KnmiRadar.schema_fingerprint(descriptors) -> str

    The archive is local, so there is no session, no limiter and no rate limit
    that matters; `rate_limit` is kept because the planner reads `max_workers`
    off it to size the parse fan-out.
    """

    name = "knmi_radar"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        archive_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="knmi_radar")
        self._archive_root = Path(archive_root) if archive_root else None
        self._listing_cache: dict[Path, list[str]] = {}

    # ------------------------------------------------------------------ config

    @property
    def rate_limit(self) -> RateLimit:
        return self.source.rate_limit

    def archive_root(self, cfg: ProductConfig | None = None) -> Path:
        """Where the daily tars live: constructor, then product, then source config.

        Deliberately not a constant — the archive is a mount point that differs
        between the cluster and a scratch copy, and the tests point it at a
        tmp_path.
        """
        for candidate in (
            self._archive_root,
            (cfg.extra.get("archive_root") if cfg else None),
            self.source.extra.get("archive_root"),
            self.source.base_url,
        ):
            if candidate:
                return Path(str(candidate)).expanduser()
        raise ValueError(
            "no KNMI radar archive root: pass archive_root=, or set extra.archive_root "
            "in the product or source config"
        )

    def _day_start(self, cfg: ProductConfig) -> timedelta:
        return _hhmm(str(cfg.extra.get("day_start", DEFAULT_DAY_START)))

    def _missing_policy(self, cfg: ProductConfig) -> str:
        policy = str(cfg.extra.get("missing_data_policy", DEFAULT_MISSING_POLICY))
        if policy not in MISSING_POLICIES:
            raise ValueError(
                f"extra.missing_data_policy {policy!r} must be one of {MISSING_POLICIES}"
            )
        return policy

    def _listing(self, root: Path) -> list[str]:
        if root not in self._listing_cache:
            try:
                self._listing_cache[root] = sorted(
                    p.name for p in root.iterdir() if p.is_file()
                )
            except OSError as exc:
                raise FileNotFoundError(f"cannot read KNMI radar archive {root}: {exc}") from exc
        return self._listing_cache[root]

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every daily tar whose frames overlap `window`. No tar is opened.

        Each unit is one tar, one shard and one Slurm task, and carries
        `valid_time` = the first frame it holds. The planner needs that: with
        `store.shard.time` = 288 a gridded unit without a valid time cannot be
        placed on a shard, and two tasks sharing a shard lose frames.
        """
        start, end = _window_bounds(window)
        # Timeline owns cadence and UTC validation for every store here; borrow
        # it rather than inventing a second rule for the window.
        Timeline(start, end, cfg.cadence)

        root = self.archive_root(cfg)
        day_start = self._day_start(cfg)
        units: dict[str, Unit] = {}
        skipped = 0
        for filename in self._listing(root):
            m = ARCHIVE_RE.match(filename)
            if not m:
                skipped += 1
                continue
            unit = self._unit_for(m, root / filename, cfg, day_start)
            covers_to = datetime.fromisoformat(str((unit.meta or {})["covers_to"]))
            if unit.valid_time >= end or covers_to <= start:
                continue
            if unit.key in units:
                logger.warning(
                    "knmi_radar: %s duplicates the day already covered by %s; keeping the first",
                    filename,
                    (units[unit.key].meta or {}).get("filename"),
                )
                continue
            units[unit.key] = unit
        if skipped:
            logger.info("knmi_radar: ignored %d non-archive file(s) in %s", skipped, root)
        out = sorted(units.values(), key=lambda u: u.valid_time or start)
        logger.info("knmi_radar: %d daily tar(s) overlap %s..%s", len(out), start, end)
        return out

    def _unit_for(
        self, m: re.Match[str], path: Path, cfg: ProductConfig, day_start: timedelta
    ) -> Unit:
        """One tar -> one Unit, reading its coverage off its own filename.

        The first frame is the start token when the name is well formed, and the
        configured day start when it is not, so a typo costs a five-minute
        uncertainty rather than a missing day.
        """
        filename = path.name
        day = _yyyymmdd(m.group("from_date"))
        offset = _time_of_day(m.group("from_time"), filename)
        first = day + (day_start if offset is None else offset)

        end_offset = _time_of_day(m.group("to_time"), filename)
        if end_offset is None:
            last_exclusive = first + timedelta(days=1)
        else:
            # The end token names the LAST frame, so the half-open end is one
            # cadence later.
            last_exclusive = _yyyymmdd(m.group("to_date")) + end_offset + cfg.cadence
        if last_exclusive <= first:
            last_exclusive = first + cfg.cadence

        return Unit(
            key=f"{cfg.source}/{cfg.product}/{m.group('from_date')}",
            url=path.as_uri(),
            valid_time=first,
            meta={
                "filename": filename,
                "source_path": str(path),
                "archive_date": m.group("from_date"),
                "covers_from": first.isoformat(),
                "covers_to": last_exclusive.isoformat(),
            },
        )

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Publish one daily tar into the raw layer. `None` if it has vanished.

        There is no HTTP here, so `meteopack.core.http` cannot do the download —
        but its rules still apply, and its helpers are reused rather than
        reimplemented: a private `.part` name, `os.replace` to publish, and the
        same streaming sha256 the downloaders record. A tar already present is
        re-hashed and returned, which is what makes a re-run free.

        The default `extra.link_mode: hardlink` costs no bytes, because the raw
        layer and the archive are the same inode. That is exactly what you want
        for a 35 GB archive that is never rewritten in place, and exactly what
        you do not want if it is — set `copy` then.
        """
        # Private, but these are the house implementations of "name a temp file
        # safely" and "hash a file"; a second copy of either is how the two
        # drift apart.
        from meteopack.core.http import _part_path, _sha256_file

        meta = unit.meta or {}
        src = Path(meta.get("source_path") or _path_from_uri(unit.url))
        # One directory per year: 2680 tars in one directory is already enough.
        dest = Path(raw_root) / _archive_year(meta, src.name) / src.name
        if dest.exists():
            digest, n = _sha256_file(dest)
            return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit.key)
        if not src.exists():
            logger.warning("knmi_radar: %s is gone from the archive; skipping", src)
            return None

        dest.parent.mkdir(parents=True, exist_ok=True)
        part = _part_path(dest)
        mode = str(self.source.extra.get("link_mode", "hardlink"))
        try:
            _publish(src, part, mode)
            os.replace(part, dest)
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        digest, n = _sha256_file(dest)
        return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit.key)

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> xr.Dataset:
        """One daily tar -> a `(time, y, x)` float32 dataset on the product grid.

        Frames are read in timestamp order into a preallocated NaN array, so a
        member that is missing or unreadable simply stays at the fill value
        instead of shifting the rest of the day. A variable that resolved in no
        frame is dropped rather than written, so backfilling a generation that
        has no quality image leaves the quality array alone.
        """
        import h5py
        import xarray as xr

        path = Path(raw.path)
        grid = grids.get(cfg.grid_id) if cfg.grid_id else None
        policy = self._missing_policy(cfg)
        wanted = wanted_variables(cfg)
        if not wanted:
            raise ValueError(f"{cfg.store_id} declares no variables")

        stamped = _members(path)
        if not stamped:
            logger.warning("knmi_radar: no radar members in %s", path.name)
            return xr.Dataset(coords={"time": np.array([], dtype="datetime64[ns]")})

        ny, nx = grid.shape if grid else _probe_shape(path, stamped[0][1])
        frames = {var.name: np.full((len(stamped), ny, nx), np.nan, "float32") for var, _ in wanted}
        hits = dict.fromkeys(frames, 0)
        seen_layout: tuple[str, ...] | None = None

        with tarfile.open(path, "r") as tar:
            for index, (_when, member) in enumerate(stamped):
                data = _read_member(tar, member, path)
                if data is None:
                    continue
                try:
                    with h5py.File(BytesIO(data), "r") as frame:
                        specs = _image_specs(frame)
                        if grid is not None:
                            _check_grid(frame, specs, grid, member.name, cfg)
                        layout = tuple(sorted(s.descriptor for s in specs.values()))
                        if layout != seen_layout:
                            self._check_layout(layout, cfg, member.name)
                            seen_layout = layout
                        matched = _fill_frame(
                            frame, member.name, specs, wanted, frames, hits, index, policy
                        )
                except SchemaDriftError:
                    raise
                except Exception as exc:  # noqa: BLE001 - one bad frame is not a bad day
                    logger.warning("knmi_radar: unreadable frame %s: %s", member.name, exc)
                    continue
                if not matched:
                    raise SchemaDriftError(
                        f"{path.name}: {member.name} has none of the configured images "
                        f"{sorted({p for _, ps in wanted for p in ps})}; it holds "
                        f"{sorted(specs)}. Re-run `scan` and re-pin extra.known_layouts."
                    )

        data_vars = {}
        for var, _ in wanted:
            if not hits[var.name]:
                logger.info(
                    "knmi_radar: %s not present in %s; leaving it unwritten", var.name, path.name
                )
                continue
            data_vars[var.name] = (
                ("time", "y", "x"),
                frames[var.name],
                {"units": var.units, "long_name": var.long_name or var.name},
            )
        if not data_vars:
            raise SchemaDriftError(
                f"{path.name}: none of {[v.name for v, _ in wanted]} resolved to an image"
            )
        times = np.array([np.datetime64(w.replace(tzinfo=None), "ns") for w, _ in stamped])
        return xr.Dataset(
            data_vars,
            coords={"time": times},
            attrs={
                "source": cfg.source,
                "product": cfg.product,
                "grid_id": cfg.grid_id or "",
                "unit_key": raw.unit_key,
                "raw_sha256": raw.sha256,
            },
        )

    def _check_layout(self, layout: Sequence[str], cfg: ProductConfig, name: str) -> None:
        """Refuse a frame whose image layout the config has never seen.

        What is pinned is the *set* of layouts, not one fingerprint, because the
        archive genuinely contains several product generations and all of them
        are legitimate. Drift means a layout outside the set — which, when it is
        a rename of the precipitation image, does not crash anything: it writes
        an all-NaN archive. `cfg.schema_fingerprint` hashes the set, so editing
        the list without re-pinning the hash fails loudly rather than quietly
        widening what counts as known.
        """
        known = [str(k) for k in (cfg.extra.get("known_layouts") or [])]
        if not known:
            return
        pinned = cfg.schema_fingerprint
        if pinned:
            seen = product_fingerprint(known)
            if seen != pinned:
                raise SchemaDriftError(
                    f"config is inconsistent: extra.known_layouts hashes to {seen} but "
                    f"schema_fingerprint pins {pinned}. Re-run `scan` and re-pin both together."
                )
        fingerprint = schema_fingerprint(layout)
        if fingerprint not in set(known):
            raise SchemaDriftError(
                f"{name}: image layout {fingerprint} is not in the pinned set {sorted(known)}; "
                f"images are {list(layout)}. Re-run `scan` and re-pin extra.known_layouts "
                f"and schema_fingerprint after checking what moved."
            )

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile], *, members_per_file: int | None = None) -> dict:
        """Report every distinct image layout across `raw_files`.

        KNMI has changed this product four times since 2018 — including renaming
        the precipitation image — so the pipeline pins the fingerprints this
        returns and `parse` refuses to run against anything else.

        Every frame is opened by default, which is slow (~3 s per tar, attrs
        only) and is still the right default: the layout changes *within* a day,
        not only between days, and the uint8 variant that occasioned this note
        is ten frames out of 288. `members_per_file` samples evenly instead,
        which is useful for a first look and cannot be trusted for a pinned set.

        The report is deliberately a plain dict: it is meant to be written to
        JSON next to the store and diffed between runs.
        """
        import h5py

        variants: dict[tuple[str, ...], dict] = {}
        errors: list[dict[str, str]] = []
        for raw in raw_files:
            path = Path(raw.path)
            try:
                stamped = _members(path)
                if not stamped:
                    raise ValueError("no RAD_NL25_RAC_RT_*.h5 members")
                wanted = len(stamped) if members_per_file is None else max(1, members_per_file)
                step = max(1, len(stamped) // wanted)
                with tarfile.open(path, "r") as tar:
                    for when, member in stamped[::step][:wanted]:
                        blob = _read_member(tar, member, path)
                        if blob is None:
                            raise ValueError(f"unreadable member {member.name}")
                        with h5py.File(BytesIO(blob), "r") as frame:
                            specs = _image_specs(frame)
                            geo = dict(frame[_GEO_GROUP].attrs) if _GEO_GROUP in frame else {}
                        layout = tuple(sorted(s.descriptor for s in specs.values()))
                        entry = variants.setdefault(
                            layout,
                            {
                                "images": list(layout),
                                "parameters": sorted(specs),
                                "rows": int(_scalar(geo["geo_number_rows"]))
                                if "geo_number_rows" in geo
                                else None,
                                "columns": int(_scalar(geo["geo_number_columns"]))
                                if "geo_number_columns" in geo
                                else None,
                                "corners": [
                                    round(float(v), 6)
                                    for v in np.ravel(geo.get("geo_product_corners", []))
                                ],
                                "fingerprint": schema_fingerprint(layout),
                                "count": 0,
                                "first_seen": when.isoformat(),
                                "last_seen": when.isoformat(),
                                "examples": [],
                            },
                        )
                        entry["count"] += 1
                        entry["first_seen"] = min(entry["first_seen"], when.isoformat())
                        entry["last_seen"] = max(entry["last_seen"], when.isoformat())
                        if path.name not in entry["examples"] and len(entry["examples"]) < 3:
                            entry["examples"].append(path.name)
            except Exception as exc:  # noqa: BLE001 - a bad tar is a finding, not a crash
                errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})

        ordered = sorted(variants.values(), key=lambda e: e["first_seen"])
        fingerprints = [e["fingerprint"] for e in ordered]
        return {
            "source": self.name,
            "n_files": len(raw_files),
            "n_variants": len(ordered),
            "variants": ordered,
            "fingerprints": sorted(set(fingerprints)),
            "fingerprint": product_fingerprint(fingerprints) if fingerprints else None,
            "errors": errors,
        }

    schema_fingerprint = staticmethod(schema_fingerprint)


# ----------------------------------------------------------------- file helpers


def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
    def as_utc(value: datetime) -> datetime:
        return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)

    return as_utc(window.start), as_utc(window.end)


def _path_from_uri(url: str) -> str:
    parsed = urlparse(url)
    return unquote(parsed.path) if parsed.scheme in ("file", "") else url


def _publish(src: Path, part: Path, mode: str) -> None:
    """Materialise `src` at `part`: hardlink where possible, copy where not."""
    if mode == "symlink":
        part.symlink_to(src.resolve())
        return
    if mode == "hardlink":
        try:
            os.link(src, part)
            return
        except OSError as exc:  # a different filesystem, or a backend without links
            logger.info("knmi_radar: hardlink failed (%s); copying %s", exc, src.name)
    elif mode != "copy":
        raise ValueError(f"unknown link_mode {mode!r}; use hardlink, symlink or copy")
    shutil.copyfile(src, part)
    with part.open("rb") as f:
        os.fsync(f.fileno())


def _archive_year(meta: Mapping[str, str], filename: str) -> str:
    """Year the tar belongs to, from the unit meta or the filename itself."""
    date = str(meta.get("archive_date") or "")
    if len(date) >= 4 and date[:4].isdigit():
        return date[:4]
    m = ARCHIVE_RE.match(filename)
    return m.group("from_date")[:4] if m else "unknown"


def _fill_frame(
    frame: Any,
    name: str,
    specs: Mapping[str, ImageSpec],
    wanted: Sequence[tuple[Variable, list[str]]],
    frames: dict[str, np.ndarray],
    hits: dict[str, int],
    index: int,
    policy: str,
) -> bool:
    """Calibrate every configured image in one open frame. False if none matched.

    The first accepted `image_geo_parameter` present wins, which is how one
    canonical variable spans a mid-archive rename.
    """
    matched = False
    for var, parameters in wanted:
        spec = next((specs[p] for p in parameters if p in specs), None)
        if spec is None:
            continue
        counts = np.asarray(frame[f"{spec.group}/{_IMAGE_DATA}"][:])
        if counts.shape != frames[var.name].shape[1:]:
            raise SchemaDriftError(
                f"{name}: {spec.geo_parameter} is {counts.shape}, "
                f"the grid is {frames[var.name].shape[1:]}"
            )
        frames[var.name][index] = _apply_variable(calibrate(counts, spec, policy=policy), var)
        hits[var.name] += 1
        matched = True
    return matched


def _members(path: Path) -> list[tuple[datetime, tarfile.TarInfo]]:
    """Radar members of one tar, in timestamp order. Anything else is ignored."""
    with tarfile.open(path, "r") as tar:
        stamped = []
        for member in tar:
            if not member.isfile():
                continue
            when = member_time(member.name)
            if when is not None:
                stamped.append((when, member))
    return sorted(stamped, key=lambda item: item[0])


def _read_member(tar: tarfile.TarFile, member: tarfile.TarInfo, path: Path) -> bytes | None:
    """Bytes of one member; `None` (with a warning) if the tar is damaged there."""
    try:
        handle = tar.extractfile(member)
        return None if handle is None else handle.read()
    except (OSError, tarfile.TarError) as exc:
        logger.warning("knmi_radar: cannot read %s from %s: %s", member.name, path.name, exc)
        return None


def _probe_shape(path: Path, member: tarfile.TarInfo) -> tuple[int, int]:
    """Frame shape taken from the file, for a product with no `grid_id`."""
    import h5py

    with tarfile.open(path, "r") as tar:
        blob = _read_member(tar, member, path)
    if blob is None:
        raise ValueError(f"{path.name}: cannot read {member.name} to determine the frame shape")
    with h5py.File(BytesIO(blob), "r") as frame:
        specs = _image_specs(frame)
    if not specs:
        raise SchemaDriftError(f"{path.name}: {member.name} holds no image groups")
    return next(iter(specs.values())).shape


def _check_grid(
    frame: Any, specs: Mapping[str, ImageSpec], grid: Any, name: str, cfg: ProductConfig
) -> None:
    """Assert the frame really is on the product grid, before anything is read.

    Shape, the file's own row/column counts and the product corners all have to
    agree with the registry: a composite silently reprojected or re-cornered
    upstream would otherwise be written pixel-for-pixel into the wrong place.
    """
    for spec in specs.values():
        if spec.shape != tuple(grid.shape):
            raise SchemaDriftError(
                f"{name}: {spec.geo_parameter} is {spec.shape}, grid {grid.id} is "
                f"{tuple(grid.shape)}"
            )
    if _GEO_GROUP not in frame:
        raise SchemaDriftError(f"{name}: no /{_GEO_GROUP} group to check against {grid.id}")
    geo = frame[_GEO_GROUP].attrs
    rows, columns = int(_scalar(geo["geo_number_rows"])), int(_scalar(geo["geo_number_columns"]))
    if (rows, columns) != tuple(grid.shape):
        raise SchemaDriftError(
            f"{name}: declares {rows}x{columns}, grid {grid.id} is {tuple(grid.shape)}"
        )
    if not grid.corners_lonlat or "geo_product_corners" not in geo:
        return
    tolerance = float(cfg.extra.get("corner_tolerance_deg", DEFAULT_CORNER_TOLERANCE))
    seen = np.ravel(np.asarray(geo["geo_product_corners"], dtype="float64"))
    expected = np.ravel(np.asarray(grid.corners_lonlat, dtype="float64"))
    if seen.shape != expected.shape or not np.allclose(seen, expected, atol=tolerance, rtol=0):
        raise SchemaDriftError(
            f"{name}: geo_product_corners {seen.tolist()} differ from grid {grid.id} "
            f"{expected.tolist()} by more than {tolerance} degrees"
        )


__all__ = [
    "ARCHIVE_RE",
    "DEFAULT_DAY_START",
    "MEMBER_RE",
    "MISSING_POLICIES",
    "ImageSpec",
    "KnmiRadar",
    "SchemaDriftError",
    "calibrate",
    "member_time",
    "parse_calibration",
    "product_fingerprint",
    "representable_sentinel",
    "schema_fingerprint",
    "wanted_variables",
]
