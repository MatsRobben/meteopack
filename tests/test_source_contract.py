"""The contract is the one thing worth testing before anything implements it."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from meteopack.sources.base import RawFile, Source, TimeWindow, Unit


def test_timewindow_and_unit_are_hashable():
    w = TimeWindow(datetime(2012, 1, 1, tzinfo=UTC), datetime(2012, 1, 2, tzinfo=UTC))
    u = Unit(key="dwd/10min/00044/2012", url="https://example.invalid/x.zip")
    assert {w, u}


def test_a_minimal_implementation_satisfies_the_protocol():
    class Dummy:
        name = "dummy"

        def list(self, window, cfg):
            return []

        def fetch(self, unit, raw_root):
            return RawFile(Path("x"), "0" * 64, 0, unit.key)

        def parse(self, raw, cfg):
            raise NotImplementedError

    assert isinstance(Dummy(), Source)
