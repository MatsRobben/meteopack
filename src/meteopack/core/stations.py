"""Station coordinates recovered from a catalogue archived beside the data.

A station catalogue is a live thing: every one of these providers serves the
stations that are *currently* reporting. Backfilling 2012 in 2026 therefore
gets coordinates for the stations that still exist and nothing for the ones
that have since closed, and a station with no position silently drops out of
anything spatial - which, for a dataset whose point is dense maps, is the whole
value of those rows.

Measured on the archives in `data_raw`:

===========  ====================  ==========  =====================
source       archived catalogue    reported    resolvable live today
===========  ====================  ==========  =====================
dwd          1115                  1030        513
dmi          113                   110         31
ea           630                   628         628
===========  ====================  ==========  =====================

So half of DWD's stations and three quarters of DMI's have no coordinates from
the live API, and the archived catalogue has every one of them. EA is whole
because the Environment Agency still lists its historical gauges - the problem
is the provider's policy, not ours, which is why this reads whatever is on
disk rather than trying to be clever.

The five catalogues came from a previous pipeline and differ only in what they
call the id. Everything else - lon, lat, a name, usually a height - is already
common, so this is one reader with a table of aliases rather than five readers.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CATALOGUE_NAME = "station_catalog.json"

#: Field aliases, in the order they are tried. Drawn from the five catalogues
#: actually present, not invented: dmi `stationId`, dwd `station_id`, ea
#: `notation`, midas `src_id`, belgium `code`.
_ALIASES: dict[str, tuple[str, ...]] = {
    "station_id": ("station_id", "stationId", "notation", "src_id", "code", "id"),
    "name": ("name", "label", "NOM_USUEL"),
    "lat": ("lat", "latitude"),
    "lon": ("lon", "longitude", "long"),
    "height": ("height", "altitude", "elevation", "alt"),
}


def _first(record: dict, names: tuple[str, ...]) -> Any:
    for name in names:
        if record.get(name) is not None:
            return record[name]
    return None


def _number(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    # NaN is not a coordinate; math.isnan says so more clearly than out == out.
    return None if math.isnan(out) else out


def read_catalogue(path: Path | str) -> list[dict]:
    """Normalise one archived catalogue into station rows.

    Accepts either a list of records or a mapping keyed by station id, because
    both shapes are present. A record with no usable position is dropped rather
    than stored as a station without coordinates, which is the condition this
    exists to remove.
    """
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("cannot read the archived station catalogue %s: %s", path, exc)
        return []

    if isinstance(payload, dict):
        records = []
        for key, value in payload.items():
            if isinstance(value, dict):
                records.append({"station_id": key, **value})
    elif isinstance(payload, list):
        records = [r for r in payload if isinstance(r, dict)]
    else:
        logger.warning("%s is neither a list nor a mapping of stations", path)
        return []

    out: list[dict] = []
    for record in records:
        sid = _first(record, _ALIASES["station_id"])
        lat = _number(_first(record, _ALIASES["lat"]))
        lon = _number(_first(record, _ALIASES["lon"]))
        if sid is None or lat is None or lon is None:
            continue
        name = _first(record, _ALIASES["name"])
        out.append({
            "station_id": str(sid),
            "lat": lat,
            "lon": lon,
            "height": _number(_first(record, _ALIASES["height"])),
            "name": None if name is None else str(name),
        })
    return out


def find_catalogues(raw_root: Path | str) -> list[Path]:
    """Every archived catalogue under a source's raw directory."""
    root = Path(raw_root)
    if not root.exists():
        return []
    return sorted(root.rglob(CATALOGUE_NAME))


def archived_stations(raw_root: Path | str) -> list[dict]:
    """Station rows from whatever catalogues are archived under `raw_root`."""
    rows: dict[str, dict] = {}
    for path in find_catalogues(raw_root):
        for row in read_catalogue(path):
            rows.setdefault(row["station_id"], row)
    return list(rows.values())
