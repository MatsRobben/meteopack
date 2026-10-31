"""EUMETSAT MSG/SEVIRI Level 1.5 rapid-scan imagery (`MSG15 RSS`), 5-minute scenes.

One unit is one scene: a single ~98 MB MSG Native (`.nat`) image, usually still
inside the `.zip` the EUMETSAT Data Store ships it in. `parse` decodes it with
`satpy`'s `seviri_l1b_native` reader and resamples the configured channels onto
the product's own grid with `pyresample`, so what reaches the store is already
co-registered with every other gridded product on that grid.

Layout. One directory per day under `extra.archive_root`::

    <root>/<YYYY>/<MM>/<DD>/MSG4-SEVI-MSG15-0100-NA-<YYYYMMDDhhmmss.fffffffff>Z-NA.zip
      MSG4-SEVI-MSG15-0100-NA-<...>Z-NA.nat     the image
      EOPMetadata.xml, manifest.xml             ignored here

`extra.archive_layout: flat` reads a single flat directory instead, which is
what a scratch copy or a test fixture usually looks like. Loose `.nat` files are
accepted everywhere a `.zip` is.

Seven pieces of domain knowledge, each of which silently corrupts the store if
you get it wrong:

* **The timestamp in the filename is the end of the scan, not the slot.** A
  scene named `...20261025151916.069Z...` is the 15:15 repeat cycle: SEVIRI
  starts the RSS scan ~10 s after the nominal slot and the product is written
  ~4 min 18 s later. `list` therefore floors the filename stamp to the cadence,
  which is only valid while the dissemination delay stays below one cadence —
  so `parse` re-derives the slot from the file's own `SSBT` (sensing start) and
  refuses a scene whose slot disagrees with the unit's `valid_time`. Taking the
  filename stamp as the valid time would shift the whole archive by one slot.
* **Every unit carries `valid_time`.** A gridded unit without one cannot be
  placed on a shard, and `meteopack.core.store.gridded` is explicit that two
  tasks writing different frames of one shard lose frames. One scene is one
  frame, so a day of 288 scenes shares one 288-frame shard; the planner's
  `bucket` mode keeps that day with a single task, and it can only do that
  because `valid_time` is set.
* **Pixel counts are not the product.** `satpy` applies the file's own
  `Cal_Slope`/`Cal_Offset` and the per-platform Planck coefficients, giving
  reflectance in `%` for the solar channels and brightness temperature in `K`
  for the thermal ones. Nothing here hard-codes a calibration constant. The
  units `satpy` reports are checked against the ones the config declares, so a
  channel that silently starts arriving as radiance fails instead of writing
  plausible-looking numbers two orders of magnitude out.
* **The resample must not reduce the source data.** `pyresample`'s
  `reduce_data` pre-crop mis-handles a target CRS whose units are kilometres —
  `knmi_1km_stereo` is one — and quietly drops a quarter of the target cells to
  NaN (measured: 24.0% NaN with it on, 0.0% with it off, same scene, same
  radius). `reduce_data` is therefore off and not configurable.
* **`radius_of_influence` is a real parameter, not a formality.** A SEVIRI
  pixel over the Netherlands is ~3 km across but ~5.5 km along the meridian,
  because the sub-satellite point is on the equator. At 3000 m, 11.8% of the
  KNMI domain finds no neighbour; at 6000 m, none of it does. The default is
  6000 m and the config says why.
* **The scene is a reduced scan, not a full disc.** `SMOD: REDSCN` and
  `SouthLineSelectedRectangle: 2321` mean the image covers SEVIRI grid lines
  2321-3712 only. That is the whole of Europe, so it covers any grid you would
  sensibly use it with — but it is part of the pinned layout, and a scene that
  arrives as a full disc (or a narrower RSS) is reported by `scan` rather than
  being silently reprojected into a different footprint.
* **HRV has its own grid.** The 1 km channel arrives on a separate, narrower
  area definition (5568 of 11136 columns in the RSS product). `parse` handles
  it correctly — the channels are grouped by source area and each group gets
  its own neighbour lookup — but the default product config leaves it out,
  because the RSS HRV window does not necessarily cover a given domain and a
  partly-covered channel is worse than an absent one.

Downloading is deliberately out of scope. `fetch` publishes a scene that is
already on local disk into the raw layer, by hardlink, exactly the way
`knmi_radar` does. A live feed would slot in as a `list` that queries the
EUMETSAT Data Store (`eumdac.DataStore(...).get_collection("EO:EUM:DAT:MSG:MSG15-RSS")`)
for the same scene names and a `fetch` that calls
`meteopack.core.http.fetch_file` with the product's download URL; neither
touches `parse`, and neither is imported here, so nothing in this module needs
credentials to load.

Verified against the real archive on 2026-10-31: `MSG4`, `REDSCN`, SSP 9.5 E,
all 12 bands selected, lines 2321-3712 x columns 1-3712, `(1392, 3712)` per
VIS/IR channel, in both a June 2026 full sample and the October 2026 feed.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import zipfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import numpy as np

from meteopack.core import grids
from meteopack.core.config import ProductConfig, RateLimit, SourceConfig, Variable
from meteopack.core.timeline import Timeline
from meteopack.sources.base import RawFile, TimeWindow, Unit

if TYPE_CHECKING:  # heavy, caller-facing types only
    import xarray as xr
    from pyresample.geometry import AreaDefinition

logger = logging.getLogger(__name__)

#: `MSG4-SEVI-MSG15-0100-NA-20261025151916.069000000Z-NA.zip`. The platform and
#: the fractional seconds are both captured because both move: `MSG4` will not
#: be the last Meteosat, and the fraction is 9 digits today and need not stay so.
SCENE_RE = re.compile(
    r"^(?P<platform>MSG\d)-(?P<instrument>[A-Z]+)-(?P<product>MSG\d+)-(?P<subtype>\d{4})"
    r"-(?P<coverage>[A-Z]{2})-(?P<stamp>\d{14})(?:\.(?P<fraction>\d+))?Z"
    r"-(?P<disposition>[A-Z]{2})(?P<tail>[-\w.]*?)\.(?P<extension>zip|nat)$"
)

#: The ASCII U-MARF main + secondary product headers at the very front of every
#: native file: 3674 + 1440 bytes of `Key : Value` records, NUL-padded. Reading
#: this is how `scan` fingerprints a scene and how `parse` recovers the sensing
#: times without decoding a single image line.
ASCII_HEADER_BYTES = 5114

#: What a layout is, for drift detection. Deliberately only the fields that
#: describe the *shape and identity* of the product — never a time, a size that
#: follows from the rectangle, or a quality flag, all of which change per scene.
LAYOUT_KEYS: tuple[str, ...] = (
    "FormatName",
    "DocumentMajorVersion",
    "DocumentMinorVersion",
    "GORT",
    "ASTI",
    "AIID",
    "APNM",
    "GPLV",
    "SMOD",
    "LLOS",
    "SelectedBandIDs",
    "SouthLineSelectedRectangle",
    "NorthLineSelectedRectangle",
    "EastColumnSelectedRectangle",
    "WestColumnSelectedRectangle",
    "NumberLinesVISIR",
    "NumberColumnsVISIR",
    "NumberLinesHRV",
    "NumberColumnsHRV",
)

#: SEVIRI's twelve channels. The three solar ones calibrate to reflectance, the
#: eight thermal ones to brightness temperature, and HRV is solar.
SOLAR_CHANNELS = frozenset({"VIS006", "VIS008", "IR_016", "HRV"})
THERMAL_CHANNELS = frozenset(
    {"IR_039", "WV_062", "WV_073", "IR_087", "IR_097", "IR_108", "IR_120", "IR_134"}
)
CHANNELS = frozenset(SOLAR_CHANNELS | THERMAL_CHANNELS)

#: What each calibration is measured in, as `satpy` reports it. Used to check
#: the config rather than to label the output: the label comes from the file.
CALIBRATION_UNITS = {
    "reflectance": "%",
    "brightness_temperature": "K",
    "radiance": "mW m-2 sr-1 (cm-1)-1",
    "counts": "1",
}

#: `source_key: "IR_108:radiance"` pins a non-default calibration.
_CALIBRATION_SEP = ":"

#: Metres. See the module docstring: 3000 leaves 11.8% of the KNMI domain with
#: no neighbour, because SEVIRI pixels are ~5.5 km long north-south at 52 N.
DEFAULT_RADIUS_M = 6000.0

#: satpy's flip to the conventional north-up, east-right orientation. The area
#: definition is flipped with the data, so the resample is correct either way;
#: 'NE' is kept because every other reader in the house is north-up.
DEFAULT_CORNER = "NE"

_SECTION_RECORD_KEYS = ("15Header", "15Data", "15Trailer")
_ARCHIVE_LAYOUTS = ("year/month/day", "flat")
_NAT_SUFFIX = ".nat"
_ZIP_SUFFIX = ".zip"


class SchemaDriftError(RuntimeError):
    """A scene's product layout is not one the config was pinned against."""


# ------------------------------------------------------- ASCII product headers


def read_ascii_headers(blob: bytes) -> dict[str, str]:
    """The U-MARF `Key : Value` records at the front of a native file.

    Records are fixed width and NUL-padded, so the NULs come out first and the
    rest splits on newlines. Parsing stops at the first line that is not a
    printable `Key : Value` record, which is where the ASCII headers end and the
    binary level-1.5 header begins — a binary blob can contain a colon, and a
    key invented out of one would poison the layout fingerprint.
    """
    text = blob.replace(b"\x00", b"").decode("latin-1")
    out: dict[str, str] = {}
    for line in text.split("\n"):
        if not line:
            continue
        key, sep, value = line.partition(":")
        key = key.strip()
        if not sep or not key or not _is_header_key(key) or not value.isprintable():
            break
        out[key] = value.strip()
    return out


def _is_header_key(key: str) -> bool:
    return key.isascii() and all(c.isalnum() or c in " _-" for c in key)


def read_scene_headers(path: str | os.PathLike[str]) -> dict[str, str]:
    """Product headers of one scene, from a `.zip` or a loose `.nat`.

    Only the first few kilobytes are read. For a zip that still means inflating
    from the start of the member, which is cheap and avoids extracting 98 MB to
    answer "what product is this".
    """
    path = Path(path)
    if path.suffix.lower() == _ZIP_SUFFIX:
        with zipfile.ZipFile(path) as archive, archive.open(
            _native_member(archive, path)
        ) as handle:
            blob = handle.read(ASCII_HEADER_BYTES)
    else:
        with path.open("rb") as handle:
            blob = handle.read(ASCII_HEADER_BYTES)
    headers = read_ascii_headers(blob)
    if not headers:
        raise SchemaDriftError(f"{path.name}: no U-MARF product headers at the start of the file")
    return headers


def _native_member(archive: zipfile.ZipFile, path: Path) -> str:
    """The single `.nat` member of a scene zip.

    Names are checked rather than trusted: a member called `../x.nat` is a
    perfectly valid zip entry and a perfectly good way to write outside the
    extraction directory.
    """
    names = [n for n in archive.namelist() if n.lower().endswith(_NAT_SUFFIX)]
    if len(names) != 1:
        raise SchemaDriftError(
            f"{path.name}: expected exactly one *.nat member, found {sorted(names)}"
        )
    name = names[0]
    if Path(name).name != name or name.startswith(("/", "\\")):
        raise SchemaDriftError(f"{path.name}: refusing member with a path component: {name!r}")
    return name


def header_time(value: str) -> datetime:
    """`"20261025151508.826000000Z"` -> that instant in UTC.

    The fraction is truncated to microseconds rather than rounded: these stamps
    are only ever used to derive a slot, and a nanosecond is not a slot.
    """
    token = value.strip().rstrip("Zz")
    stamp, _, fraction = token.partition(".")
    if len(stamp) != 14 or not stamp.isdigit():
        raise SchemaDriftError(f"cannot read product header time {value!r}; expected YYYYMMDDhhmmss")
    when = datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    if fraction.isdigit():
        when = when.replace(microsecond=int(fraction[:6].ljust(6, "0")))
    return when


# ------------------------------------------------------------------ time naming


def scene_time(name: str) -> datetime | None:
    """`...-20261025151916.069000000Z-NA.zip` -> that instant, else `None`.

    This is the **end** of the repeat cycle, not the slot: see `nominal_slot`.
    """
    m = SCENE_RE.match(Path(name).name)
    if not m:
        return None
    try:
        when = datetime.strptime(m.group("stamp"), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        logger.warning("eumetsat_seviri: scene %s has an impossible timestamp", name)
        return None
    fraction = m.group("fraction")
    if fraction:
        when = when.replace(microsecond=int(fraction[:6].ljust(6, "0")))
    return when


def nominal_slot(when: datetime, cadence: timedelta) -> datetime:
    """The repeat cycle `when` belongs to: floor to the cadence, in UTC.

    Correct for the dissemination stamp (slot + ~4 min 18 s for 5-minute RSS)
    and for the sensing start (slot + ~10 s) alike, as long as the delay stays
    inside one cadence. `parse` checks that the two agree, which is what turns
    "as long as" into something the pipeline notices.
    """
    step = int(cadence.total_seconds())
    if step <= 0:
        raise ValueError("cadence must be positive")
    anchor = datetime(1970, 1, 1, tzinfo=UTC)
    seconds = int((_as_utc(when) - anchor).total_seconds())
    return anchor + timedelta(seconds=seconds - seconds % step)


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _window_bounds(window: TimeWindow) -> tuple[datetime, datetime]:
    return _as_utc(window.start), _as_utc(window.end)


# ----------------------------------------------------------------- fingerprints


def layout_descriptor(headers: Mapping[str, str]) -> str:
    """Normalised identity of one product layout, for fingerprinting.

    Only `LAYOUT_KEYS` and the three section-size records take part. A key that
    is absent is recorded as absent rather than skipped, so a field disappearing
    upstream moves the fingerprint instead of leaving it unchanged.
    """
    parts = [f"{key}={headers.get(key, '<absent>').strip()}" for key in LAYOUT_KEYS]
    parts += [f"{key}={_section_size(headers, key)}" for key in _SECTION_RECORD_KEYS]
    return "|".join(parts)


def _section_size(headers: Mapping[str, str], key: str) -> str:
    """`"445286        5114"` -> `"445286"`. The offset follows from the sizes."""
    value = headers.get(key)
    if not value:
        return "<absent>"
    first = value.split()
    return first[0] if first else "<absent>"


def schema_fingerprint(descriptors: Iterable[str]) -> str:
    """Stable hash of one scene's layout. Order-independent over `descriptors`."""
    normalised = sorted({d.strip() for d in descriptors} - {""})
    return hashlib.sha256("|".join(normalised).encode("utf-8")).hexdigest()[:16]


def product_fingerprint(layouts: Iterable[str]) -> str:
    """Stable hash of the whole set of layouts a product accepts."""
    joined = "|".join(sorted(set(layouts)))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------- config access


def default_calibration(channel: str) -> str:
    """Reflectance for the solar channels, brightness temperature for the rest."""
    return "reflectance" if channel in SOLAR_CHANNELS else "brightness_temperature"


def _channel_and_calibration(var: Variable) -> tuple[str, str]:
    """`"IR_108"` or `"IR_108:radiance"` -> `("IR_108", "<calibration>")`."""
    channel, sep, calibration = str(var.source_key).strip().partition(_CALIBRATION_SEP)
    channel = channel.strip()
    if channel not in CHANNELS:
        raise ValueError(
            f"variable {var.name!r} has source_key {var.source_key!r}; the channel must be "
            f"one of {sorted(CHANNELS)}"
        )
    calibration = calibration.strip() if sep else default_calibration(channel)
    if calibration not in CALIBRATION_UNITS:
        raise ValueError(
            f"variable {var.name!r} asks for calibration {calibration!r}; "
            f"use one of {sorted(CALIBRATION_UNITS)}"
        )
    return channel, calibration


def wanted_variables(cfg: ProductConfig) -> list[tuple[Variable, str, str]]:
    """`[(variable, channel, calibration), ...]` for this product."""
    wanted = [(var, *_channel_and_calibration(var)) for var in cfg.variables]
    if not wanted:
        raise ValueError(f"{cfg.store_id} declares no variables")
    return wanted


# -------------------------------------------------------------------- the grid


def target_area(grid: grids.Grid, *, area_id: str | None = None) -> AreaDefinition:
    """The product grid as a `pyresample` area, from the registry alone.

    The registry stores corners, not a transform, so the extent is the bounding
    box of the four product corners in the grid's own projection — which is
    exact for an axis-aligned grid, and every grid in the registry is one. Those
    corners are the *outer edges* of the corner pixels, which is what an area
    extent is, so the lon/lat the area reports for cell (0, 0) sits half a pixel
    inside the registry's corner rather than on it.
    """
    from pyresample.geometry import AreaDefinition

    if not grid.corners_lonlat:
        raise ValueError(f"grid {grid.id} has no product_corners_lonlat; cannot build an area")
    from pyproj import Transformer

    forward = Transformer.from_crs("EPSG:4326", grid.proj4, always_xy=True)
    xs, ys = zip(*(forward.transform(lon, lat) for lon, lat in grid.corners_lonlat), strict=True)
    if not all(np.isfinite(v) for v in (*xs, *ys)):
        raise ValueError(f"grid {grid.id}: product corners do not project finitely")
    return AreaDefinition(
        grid.id,
        grid.description or grid.id,
        grid.id,
        grid.proj4,
        grid.nx,
        grid.ny,
        (min(xs), min(ys), max(xs), max(ys)),
    )


# --------------------------------------------------------------- value handling


def apply_variable(
    values: np.ndarray, var: Variable, *, sentinels: Sequence[float] = ()
) -> np.ndarray:
    """Sentinels to NaN, then scale and offset, then the valid range.

    The order is the whole point. `satpy` already masks the product's own
    no-data count, so `sentinels` is normally empty and exists for a feed that
    smuggles one through; anything non-finite is masked here too, so an `inf`
    cannot be scaled into a plausible number. `valid_min`/`valid_max` are
    expressed in canonical units, i.e. after the scaling, which is the same
    contract the points sources use.
    """
    out = np.asarray(values, dtype="float32").copy()
    out[~np.isfinite(out)] = np.float32("nan")
    for sentinel in sentinels:
        out[out == np.float32(sentinel)] = np.float32("nan")
    if var.scale is not None:
        out = out * np.float32(var.scale)
    if var.offset is not None:
        out = out + np.float32(var.offset)
    if var.valid_min is not None:
        out[out < var.valid_min] = np.float32("nan")
    if var.valid_max is not None:
        out[out > var.valid_max] = np.float32("nan")
    return out


@dataclass(frozen=True, slots=True)
class SceneData:
    """One decoded scene: channel arrays, their source areas and their units.

    The one thing in this module that `satpy` produces, and therefore the seam
    a test substitutes. `areas` is per channel because HRV does not share the
    VIS/IR grid.
    """

    arrays: dict[str, np.ndarray]
    areas: dict[str, Any]
    units: dict[str, str]
    start_time: datetime | None = None
    end_time: datetime | None = None
    attrs: dict[str, str] = field(default_factory=dict)


# ------------------------------------------------------------------- the source


class EumetsatSeviri:
    """MSG/SEVIRI level 1.5 rapid-scan imagery as a `meteopack` source.

    Public API::

        src = EumetsatSeviri(source_cfg, archive_root=...)
        src.list(window, product_cfg)    -> list[Unit]      (one per 5-minute scene)
        src.fetch(unit, raw_root)        -> RawFile | None  (None == scene vanished)
        src.parse(raw_file, product_cfg) -> xarray.Dataset  (time, y, x) float32
        src.scan(raw_files)              -> dict            (layout drift report)
        src.load_scene(path, wanted, cfg)-> SceneData       (the only satpy step)
        EumetsatSeviri.schema_fingerprint(descriptors) -> str

    The archive is local, so there is no session and no limiter; `rate_limit` is
    kept because the planner reads `max_workers` off it to size the parse
    fan-out, and a SEVIRI parse is memory-hungry rather than network-bound.
    """

    name = "eumetsat_seviri"

    def __init__(
        self,
        source: SourceConfig | None = None,
        *,
        archive_root: str | os.PathLike[str] | None = None,
    ) -> None:
        self.source = source or SourceConfig(name="eumetsat_seviri")
        self._archive_root = Path(archive_root) if archive_root else None
        self._listing_cache: dict[Path, list[str]] = {}
        #: (source area hash, target area hash, radius) -> pyresample neighbour info.
        #: Building it over the 5.2 M source pixels costs ~8 s; a day of 288
        #: scenes shares one source area, so it is built once per task.
        self._neighbours: dict[tuple[Any, ...], tuple[np.ndarray, ...]] = {}

    # ------------------------------------------------------------------ config

    @property
    def rate_limit(self) -> RateLimit:
        return self.source.rate_limit

    def archive_root(self, cfg: ProductConfig | None = None) -> Path:
        """Where the scenes live: constructor, then product, then source config."""
        for candidate in (
            self._archive_root,
            (cfg.extra.get("archive_root") if cfg else None),
            self.source.extra.get("archive_root"),
            self.source.base_url,
        ):
            if candidate:
                return Path(str(candidate)).expanduser()
        raise ValueError(
            "no EUMETSAT SEVIRI archive root: pass archive_root=, or set extra.archive_root "
            "in the product or source config"
        )

    def _archive_layout(self, cfg: ProductConfig | None) -> str:
        layout = str((cfg.extra if cfg else {}).get("archive_layout") or _ARCHIVE_LAYOUTS[0])
        if layout not in _ARCHIVE_LAYOUTS:
            raise ValueError(
                f"extra.archive_layout {layout!r} must be one of {list(_ARCHIVE_LAYOUTS)}"
            )
        return layout

    def _listing(self, directory: Path) -> list[str]:
        """Filenames in one directory, cached. A directory that is not there is
        an empty day, not an error — `list` has already checked the root."""
        if directory not in self._listing_cache:
            try:
                names = sorted(p.name for p in directory.iterdir() if p.is_file())
            except FileNotFoundError:
                names = []
            except OSError as exc:
                raise FileNotFoundError(f"cannot read {directory}: {exc}") from exc
            self._listing_cache[directory] = names
        return self._listing_cache[directory]

    def _directories(self, root: Path, start: datetime, end: datetime, layout: str) -> list[Path]:
        """Which directories can hold a scene whose slot falls in the window.

        One day either side of the window, because a scene is filed under the
        date in its *filename* — the end of the scan — and nothing guarantees
        that a slot and its dissemination stamp fall on the same UTC day.
        """
        if layout == "flat":
            return [root]
        day = (start - timedelta(days=1)).date()
        last = (end + timedelta(days=1)).date()
        out = []
        while day <= last:
            out.append(root / f"{day:%Y}" / f"{day:%m}" / f"{day:%d}")
            day += timedelta(days=1)
        return out

    # --------------------------------------------------------------------- list

    def list(self, window: TimeWindow, cfg: ProductConfig) -> list[Unit]:
        """Every scene whose repeat cycle starts in `window`. No scene is opened.

        Each unit is one 5-minute frame and carries `valid_time` = its nominal
        slot, which is what lets the planner keep a shard to one writer.
        """
        start, end = _window_bounds(window)
        # Timeline owns cadence and UTC validation for every store here; borrow
        # it rather than inventing a second rule for the window.
        Timeline(start, end, cfg.cadence)

        root = self.archive_root(cfg)
        layout = self._archive_layout(cfg)
        if not root.is_dir():
            # A missing day directory is normal in a sparse archive; a missing
            # root is a misconfiguration, and answering "no scenes" to one is
            # how a backfill silently does nothing.
            raise FileNotFoundError(f"cannot read EUMETSAT SEVIRI archive {root}")
        units: dict[str, Unit] = {}
        skipped = 0
        for directory in self._directories(root, start, end, layout):
            for filename in self._listing(directory):
                m = SCENE_RE.match(filename)
                if not m:
                    skipped += 1
                    continue
                stamp = scene_time(filename)
                if stamp is None:
                    skipped += 1
                    continue
                slot = nominal_slot(stamp, cfg.cadence)
                if not (start <= slot < end):
                    continue
                unit = self._unit_for(m, directory / filename, cfg, slot, stamp)
                self._keep_newest(units, unit)
        if skipped:
            logger.info("eumetsat_seviri: ignored %d non-scene file(s) under %s", skipped, root)
        out = sorted(units.values(), key=lambda u: (u.valid_time or start, u.key))
        logger.info("eumetsat_seviri: %d scene(s) in %s..%s", len(out), start, end)
        return out

    @staticmethod
    def _keep_newest(units: dict[str, Unit], unit: Unit) -> None:
        """One slot, one unit. A reprocessed scene wins over the one it replaces.

        Two files can legitimately claim the same slot: the `.zip` and a `.nat`
        extracted beside it, or an original and a reprocessing. Keeping the
        later dissemination stamp is the only choice that is stable under a
        re-run, which the ledger needs.
        """
        previous = units.get(unit.key)
        if previous is None:
            units[unit.key] = unit
            return
        old = (previous.meta or {}).get("disseminated", ""), (previous.meta or {}).get("filename", "")
        new = (unit.meta or {}).get("disseminated", ""), (unit.meta or {}).get("filename", "")
        logger.warning(
            "eumetsat_seviri: %s and %s both claim slot %s; keeping the later one",
            old[1],
            new[1],
            unit.valid_time,
        )
        if new > old:
            units[unit.key] = unit

    def _unit_for(
        self,
        m: re.Match[str],
        path: Path,
        cfg: ProductConfig,
        slot: datetime,
        stamp: datetime,
    ) -> Unit:
        return Unit(
            key=f"{cfg.source}/{cfg.product}/{slot:%Y%m%dT%H%M}",
            url=path.as_uri(),
            valid_time=slot,
            meta={
                "filename": path.name,
                "source_path": str(path),
                "platform": m.group("platform"),
                "disseminated": stamp.isoformat(),
                "raw_subdir": f"{slot:%Y/%m/%d}",
            },
        )

    # -------------------------------------------------------------------- fetch

    def fetch(self, unit: Unit, raw_root: Path) -> RawFile | None:
        """Publish one scene into the raw layer. `None` if it has vanished.

        There is no HTTP here, so `meteopack.core.http` cannot do the download —
        but its rules still apply and its helpers are reused rather than
        reimplemented: a private `.part` name, `os.replace` to publish, and the
        same streaming sha256 the downloaders record. A scene already present is
        re-hashed and returned, which is what makes a re-run free.

        `extra.link_mode: hardlink` (the default) costs no bytes, because the
        raw layer and the archive end up as the same inode — right for an
        archive that is never rewritten in place, wrong for one that is.
        """
        # Private, but these are the house implementations of "name a temp file
        # safely" and "hash a file"; a second copy of either is how the two drift.
        from meteopack.core.http import _part_path, _sha256_file

        meta = unit.meta or {}
        src = Path(meta.get("source_path") or _path_from_uri(unit.url))
        dest = Path(raw_root) / meta.get("raw_subdir", _subdir_of(unit)) / src.name
        if dest.exists():
            digest, n = _sha256_file(dest)
            return RawFile(path=dest, sha256=digest, n_bytes=n, unit_key=unit.key)
        if not src.exists():
            logger.warning("eumetsat_seviri: %s is gone from the archive; skipping", src)
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

    # --------------------------------------------------------------- satpy seam

    def load_scene(
        self, path: Path, wanted: Sequence[tuple[Variable, str, str]], cfg: ProductConfig
    ) -> SceneData:
        """Decode one scene with `satpy`. The only step that needs the reader.

        A `.zip` is extracted to `extra.extract_dir` first, because
        `seviri_l1b_native` wants a path to a `.nat`. That directory wants to be
        local and fast — `/dev/shm` is the right answer on a cluster node whose
        `/tmp` is small and shared; the default is whatever `tempfile` picks,
        and the extraction is always removed, including on failure.

        Overriding this method is the supported way to feed `parse` something
        other than a real native file; everything downstream of it is plain
        numpy and `pyresample`.
        """
        from satpy import DataQuery, Scene

        queries = {
            channel: DataQuery(name=channel, calibration=calibration)
            for _, channel, calibration in wanted
        }
        corner = str(cfg.extra.get("upper_right_corner", DEFAULT_CORNER))
        with _native_path(path, cfg.extra.get("extract_dir")) as native:
            scene = Scene(filenames=[str(native)], reader="seviri_l1b_native")
            scene.load(list(queries.values()), upper_right_corner=corner)
            arrays, areas, units = {}, {}, {}
            for channel, query in queries.items():
                try:
                    dataset = scene[query]
                except KeyError:
                    raise SchemaDriftError(
                        f"{path.name}: channel {channel} is not in this scene; it holds "
                        f"{sorted(str(n) for n in scene.available_dataset_names())}"
                    ) from None
                arrays[channel] = np.asarray(dataset.values, dtype="float32")
                areas[channel] = dataset.attrs.get("area")
                units[channel] = str(dataset.attrs.get("units", ""))
            reference = scene[next(iter(queries.values()))]
        return SceneData(
            arrays=arrays,
            areas=areas,
            units=units,
            start_time=_maybe_utc(reference.attrs.get("start_time")),
            end_time=_maybe_utc(reference.attrs.get("end_time")),
            attrs={
                "platform_name": str(reference.attrs.get("platform_name", "")),
                "sensor": str(reference.attrs.get("sensor", "")),
            },
        )

    # -------------------------------------------------------------------- parse

    def parse(self, raw: RawFile, cfg: ProductConfig) -> xr.Dataset:
        """One scene -> a `(1, y, x)` float32 dataset on the product grid.

        The scene is decoded once, checked against the pinned layout and the
        unit's slot, then every configured channel is resampled onto
        `cfg.grid_id` by nearest neighbour. Channels are grouped by their source
        area so HRV, which has its own, costs one extra neighbour lookup rather
        than being silently put on the VIS/IR one.
        """
        import xarray as xr

        path = Path(raw.path)
        if not cfg.grid_id:
            raise ValueError(f"{cfg.store_id} has no grid_id; a gridded product needs one")
        grid = grids.get(cfg.grid_id)
        wanted = wanted_variables(cfg)

        headers = read_scene_headers(path)
        self._check_layout(layout_descriptor(headers), cfg, path.name)
        slot = self._scene_slot(headers, cfg, raw, path)

        scene = self.load_scene(path, wanted, cfg)
        self._check_times(scene, slot, cfg, path.name)
        area = target_area(grid)

        data_vars = {}
        for var, channel, calibration in wanted:
            self._check_units(scene.units.get(channel, ""), var, channel, calibration, path.name)
            values = self._resample(scene.areas[channel], area, scene.arrays[channel], cfg, channel)
            sentinels = _sentinels(cfg, channel)
            data_vars[var.name] = (
                ("time", "y", "x"),
                apply_variable(values, var, sentinels=sentinels)[np.newaxis],
                {
                    "units": var.units,
                    "long_name": var.long_name or var.name,
                    "channel": channel,
                    "calibration": calibration,
                },
            )

        times = np.array([np.datetime64(slot.replace(tzinfo=None), "ns")])
        return xr.Dataset(
            data_vars,
            coords={"time": times},
            attrs={
                "source": cfg.source,
                "product": cfg.product,
                "grid_id": grid.id,
                "unit_key": raw.unit_key,
                "raw_sha256": raw.sha256,
                "platform": headers.get("ASTI", scene.attrs.get("platform_name", "")),
                "scan_mode": headers.get("SMOD", ""),
                "subsatellite_longitude": headers.get("LLOS", ""),
            },
        )

    def _scene_slot(
        self, headers: Mapping[str, str], cfg: ProductConfig, raw: RawFile, path: Path
    ) -> datetime:
        """The repeat cycle this scene belongs to, from the file's own `SSBT`.

        `list` floors the *filename* stamp; this floors the sensing start. They
        can only disagree if the dissemination delay has grown past one cadence,
        which would shift the whole archive by a slot without any error — so
        they are compared, and a disagreement is a drift error rather than a
        silent choice between them.
        """
        sensing = headers.get("SSBT") or headers.get("SNIT")
        if not sensing:
            raise SchemaDriftError(f"{path.name}: product headers carry no SSBT or SNIT")
        slot = nominal_slot(header_time(sensing), cfg.cadence)
        from_key = _slot_from_key(raw.unit_key)
        if from_key is not None and from_key != slot:
            raise SchemaDriftError(
                f"{path.name}: sensing start {sensing} floors to {slot.isoformat()}, but the "
                f"unit key {raw.unit_key!r} says {from_key.isoformat()}. The dissemination delay "
                f"has moved past one cadence; `list` must stop flooring the filename stamp."
            )
        return slot

    def _check_times(
        self, scene: SceneData, slot: datetime, cfg: ProductConfig, name: str
    ) -> None:
        """`satpy` reports the nominal slot too; make the two agree."""
        if scene.start_time is None:
            return
        reported = nominal_slot(scene.start_time, cfg.cadence)
        if reported != slot:
            raise SchemaDriftError(
                f"{name}: the reader reports slot {reported.isoformat()} but the product "
                f"headers say {slot.isoformat()}"
            )

    def _check_units(
        self, reported: str, var: Variable, channel: str, calibration: str, name: str
    ) -> None:
        """Refuse a channel whose units are not what the config declares.

        A channel that quietly starts arriving as radiance instead of brightness
        temperature does not crash: it writes an archive of plausible numbers
        two orders of magnitude out, which is the failure this is for.
        """
        expected = CALIBRATION_UNITS[calibration]
        for seen, what in ((reported, "the reader"), (var.units, "the config")):
            if _normalise_units(seen) != _normalise_units(expected):
                raise SchemaDriftError(
                    f"{name}: {channel} as {calibration} should be in {expected!r}, but "
                    f"{what} says {seen!r}"
                )

    def _resample(
        self,
        source_area: Any,
        area: AreaDefinition,
        values: np.ndarray,
        cfg: ProductConfig,
        channel: str,
    ) -> np.ndarray:
        """Nearest neighbour from the scene's area onto the product grid.

        Nearest and nothing else, on purpose: the target is finer than the
        source everywhere this is used (1 km against 3 km), and an interpolator
        asked to upsample invents gradients between SEVIRI pixels that the
        instrument never measured. `reduce_data` is off — see the module
        docstring; with a kilometre-unit target CRS it loses a quarter of the
        domain to NaN.
        """
        from pyresample.kd_tree import get_neighbour_info, get_sample_from_neighbour_info

        if source_area is None:
            raise SchemaDriftError(f"channel {channel} carries no area definition to resample from")
        resampler = str(cfg.extra.get("resampler", "nearest"))
        if resampler != "nearest":
            raise ValueError(
                f"extra.resampler {resampler!r} is not supported; this source resamples by "
                f"nearest neighbour only"
            )
        radius = float(cfg.extra.get("radius_of_influence_m", DEFAULT_RADIUS_M))
        key = (str(source_area), str(area), radius)
        info = self._neighbours.get(key)
        if info is None:
            info = get_neighbour_info(
                source_area, area, radius, neighbours=1, reduce_data=False, segments=1
            )
            self._neighbours[key] = info
        valid_in, valid_out, index_array, _distance = info
        out = get_sample_from_neighbour_info(
            "nn", area.shape, values, valid_in, valid_out, index_array, fill_value=np.float32("nan")
        )
        return np.asarray(out, dtype="float32")

    def _check_layout(self, descriptor: str, cfg: ProductConfig, name: str) -> None:
        """Refuse a scene whose product layout the config has never seen.

        What is pinned is the *set* of layouts, because a long archive spans
        more than one legitimately — a platform change, a different rapid-scan
        rectangle. `cfg.schema_fingerprint` hashes that set, so widening the
        list without re-pinning the hash fails loudly rather than quietly.
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
        fingerprint = schema_fingerprint([descriptor])
        if fingerprint not in set(known):
            raise SchemaDriftError(
                f"{name}: product layout {fingerprint} is not in the pinned set {sorted(known)}; "
                f"it is {descriptor}. Re-run `scan` and re-pin extra.known_layouts and "
                f"schema_fingerprint after checking what moved."
            )

    # --------------------------------------------------------------------- scan

    def scan(self, raw_files: Sequence[RawFile]) -> dict:
        """Report every distinct product layout across `raw_files`.

        Only the ASCII product headers are read — five kilobytes per scene, no
        image data, no `satpy` — so this is cheap enough to run over a whole
        backfill before packing it. The report is a plain dict on purpose: it is
        meant to be written to JSON next to the store and diffed between runs.
        """
        variants: dict[str, dict] = {}
        errors: list[dict[str, str]] = []
        for raw in raw_files:
            path = Path(raw.path)
            try:
                headers = read_scene_headers(path)
                descriptor = layout_descriptor(headers)
                when = header_time(headers.get("SSBT") or headers["SNIT"]).isoformat()
                entry = variants.setdefault(
                    descriptor,
                    {
                        "descriptor": descriptor,
                        "headers": {k: headers.get(k) for k in LAYOUT_KEYS},
                        "fingerprint": schema_fingerprint([descriptor]),
                        "count": 0,
                        "first_seen": when,
                        "last_seen": when,
                        "examples": [],
                    },
                )
                entry["count"] += 1
                entry["first_seen"] = min(entry["first_seen"], when)
                entry["last_seen"] = max(entry["last_seen"], when)
                if path.name not in entry["examples"] and len(entry["examples"]) < 3:
                    entry["examples"].append(path.name)
            except Exception as exc:  # noqa: BLE001 - a bad scene is a finding, not a crash
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


@contextmanager
def _native_path(
    path: Path, extract_dir: str | os.PathLike[str] | None
) -> Iterator[Path]:
    """Yield a path to the scene's `.nat`, extracting it from a zip if need be.

    A loose `.nat` is yielded as it stands and never copied. A zip is extracted
    into a temporary directory that is removed on the way out, failure included,
    because a half-written 98 MB extraction left behind on `/dev/shm` is how the
    next task on that node runs out of space.
    """
    path = Path(path)
    if path.suffix.lower() != _ZIP_SUFFIX:
        yield path
        return
    directory = Path(extract_dir).expanduser() if extract_dir else None
    if directory is not None:
        directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="seviri-", dir=directory) as temp:
        yield Path(temp) / _extract_native(path, Path(temp))


def _extract_native(path: Path, into: Path) -> str:
    """Extract the scene's single `.nat` member into `into`; return its name."""
    with zipfile.ZipFile(path) as archive:
        member = _native_member(archive, path)
        archive.extract(member, into)
    return member


def _path_from_uri(url: str) -> str:
    parsed = urlparse(url)
    return unquote(parsed.path) if parsed.scheme in ("file", "") else url


def _subdir_of(unit: Unit) -> str:
    """`YYYY/MM/DD` for a unit whose meta has lost its `raw_subdir`."""
    return f"{unit.valid_time:%Y/%m/%d}" if unit.valid_time else "unknown"


def _slot_from_key(unit_key: str) -> datetime | None:
    """`"eumetsat_seviri/msg15_rss/20261025T1515"` -> that instant, else `None`."""
    tail = unit_key.rsplit("/", 1)[-1]
    try:
        return datetime.strptime(tail, "%Y%m%dT%H%M").replace(tzinfo=UTC)
    except ValueError:
        return None


def _sentinels(cfg: ProductConfig, channel: str) -> list[float]:
    """Extra values that mean "no data" for one channel, from the config.

    Normally empty: `satpy` masks the product's own no-data count. This is the
    escape hatch for a feed that smuggles a sentinel through as a number.
    """
    table = cfg.extra.get("sentinel_values") or {}
    return [float(v) for v in (table.get(channel) or ())]


def _normalise_units(value: str) -> str:
    return " ".join(str(value).split()).casefold()


def _maybe_utc(value: Any) -> datetime | None:
    return _as_utc(value) if isinstance(value, datetime) else None


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
            logger.info("eumetsat_seviri: hardlink failed (%s); copying %s", exc, src.name)
    elif mode != "copy":
        raise ValueError(f"unknown link_mode {mode!r}; use hardlink, symlink or copy")
    shutil.copyfile(src, part)
    with part.open("rb") as f:
        os.fsync(f.fileno())


__all__ = [
    "ASCII_HEADER_BYTES",
    "CALIBRATION_UNITS",
    "CHANNELS",
    "DEFAULT_RADIUS_M",
    "LAYOUT_KEYS",
    "SCENE_RE",
    "SOLAR_CHANNELS",
    "THERMAL_CHANNELS",
    "EumetsatSeviri",
    "SceneData",
    "SchemaDriftError",
    "apply_variable",
    "default_calibration",
    "header_time",
    "layout_descriptor",
    "nominal_slot",
    "product_fingerprint",
    "read_ascii_headers",
    "read_scene_headers",
    "scene_time",
    "schema_fingerprint",
    "target_area",
    "wanted_variables",
]
