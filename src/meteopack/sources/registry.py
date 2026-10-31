"""name -> Source class.

A built-in table plus the `meteopack.sources` entry points. The table exists so
a source works from a plain checkout without reinstalling the package, which is
how jobs actually run on the cluster; entry points let a separate package add a
source without editing this file. Imports are lazy so one broken or
dependency-heavy source cannot stop the CLI from starting.
"""
from __future__ import annotations

import importlib
from importlib.metadata import entry_points

from meteopack.sources.base import Source

_GROUP = "meteopack.sources"

#: name -> "module:attribute"
BUILTIN: dict[str, str] = {
    "dwd": "meteopack.sources.dwd:DwdStations",
    "esoh": "meteopack.sources.esoh:EsohObservations",
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
    """Instantiate a source. Keeps the import failure attached to the name."""
    try:
        cls = get(name)
    except ImportError as exc:
        raise ImportError(f"source {name!r} could not be imported: {exc}") from exc
    return cls(*args, **kwargs)
