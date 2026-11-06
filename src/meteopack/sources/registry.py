"""name -> Source class.

A built-in table plus the `meteopack.sources` entry points. The table exists so
a source works from a plain checkout without reinstalling the package, which is
how jobs actually run on the cluster; entry points let a separate package add a
source without editing this file. Imports are lazy so one broken or
dependency-heavy source cannot stop the CLI from starting.
"""
from __future__ import annotations

import importlib
import inspect
import logging
from importlib.metadata import entry_points

from meteopack.sources.base import Source

logger = logging.getLogger(__name__)

_GROUP = "meteopack.sources"

#: name -> "module:attribute"
BUILTIN: dict[str, str] = {
    "dwd": "meteopack.sources.dwd:DwdStations",
    "knmi_radar": "meteopack.sources.knmi_radar:KnmiRadar",
    "belgium": "meteopack.sources.belgium:RmiAws",
    "dmi": "meteopack.sources.dmi:DmiStations",
    "ea": "meteopack.sources.ea:EaRainfall",
    "eumetsat_seviri": "meteopack.sources.eumetsat_seviri:EumetsatSeviri",
    "knmi_insitu": "meteopack.sources.knmi_insitu:KnmiInsitu",
    "meteofrance": "meteopack.sources.meteofrance:MeteoFranceStations",
    "midas": "meteopack.sources.midas:MidasOpen",
    "wow_nl": "meteopack.sources.wow_nl:WowNl",
}


def _load(spec: str) -> type[Source]:
    module_name, _, attr = spec.partition(":")
    return getattr(importlib.import_module(module_name), attr)


def available() -> list[str]:
    names = set(BUILTIN) | {ep.name for ep in entry_points(group=_GROUP)}
    return sorted(names)


def get(name: str) -> type[Source]:
    for ep in entry_points(group=_GROUP):
        if ep.name == name:
            return ep.load()
    if name in BUILTIN:
        return _load(BUILTIN[name])
    raise KeyError(f"unknown source {name!r}; available: {', '.join(available())}")


def build(name: str, *args, **kwargs) -> Source:
    """Instantiate a source, passing only the keywords it accepts.

    The `Source` protocol constrains the three methods, not the constructor,
    and sources legitimately differ: one that reads a local archive has no use
    for an HTTP session or a rate limiter. Filtering here keeps the caller from
    having to know which is which — without it, every local-archive source
    raises TypeError the moment the CLI builds it, which no unit test catches
    because tests construct their source directly.
    """
    try:
        cls = get(name)
    except ImportError as exc:
        raise ImportError(f"source {name!r} could not be imported: {exc}") from exc

    try:
        params = inspect.signature(cls).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins and C types
        return cls(*args, **kwargs)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        accepted = kwargs
    else:
        accepted = {k: v for k, v in kwargs.items() if k in params}
        dropped = set(kwargs) - set(accepted)
        if dropped:
            logger.debug("source %s ignores %s", name, ", ".join(sorted(dropped)))
    return cls(*args, **accepted)
