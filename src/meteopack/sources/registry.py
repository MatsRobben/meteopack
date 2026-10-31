"""name -> Source class, resolved from the `meteopack.sources` entry points."""
from __future__ import annotations

from importlib.metadata import entry_points

from meteopack.sources.base import Source

_GROUP = "meteopack.sources"


def available() -> list[str]:
    return sorted(ep.name for ep in entry_points(group=_GROUP))


def get(name: str) -> type[Source]:
    for ep in entry_points(group=_GROUP):
        if ep.name == name:
            return ep.load()
    raise KeyError(f"unknown source {name!r}; available: {', '.join(available()) or '(none)'}")
