"""Grid registry: id -> CRS, shape, transform, plus footprint tests.

Grids are data (configs/grids.yaml), not constants, so a new product declares
`grid_id` rather than hard-coding corners. The footprint test densifies along
the *projected* edges: stereographic edges are curved in lon/lat, so a
straight-line quadrilateral through the corners under-covers the domain.
"""
from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_DEFAULT_REGISTRY = Path(__file__).resolve().parents[3] / "configs" / "grids.yaml"


@dataclass(frozen=True, slots=True)
class Grid:
    id: str
    proj4: str
    shape: tuple[int, int]            # (ny, nx)
    corners_lonlat: tuple[tuple[float, float], ...] | None = None
    description: str = ""

    @property
    def ny(self) -> int:
        return self.shape[0]

    @property
    def nx(self) -> int:
        return self.shape[1]

    def _ring(self, n: int = 64) -> list[tuple[float, float]]:
        from pyproj import Transformer

        if not self.corners_lonlat:
            raise ValueError(f"grid {self.id} has no corners_lonlat")
        fwd = Transformer.from_crs("EPSG:4326", self.proj4, always_xy=True)
        inv = Transformer.from_crs(self.proj4, "EPSG:4326", always_xy=True)
        xy = [fwd.transform(lon, lat) for lon, lat in self.corners_lonlat]
        ring: list[tuple[float, float]] = []
        for i, (x0, y0) in enumerate(xy):
            x1, y1 = xy[(i + 1) % len(xy)]
            ring.extend(
                inv.transform(x0 + k / n * (x1 - x0), y0 + k / n * (y1 - y0)) for k in range(n)
            )
        return ring

    def contains(self, lon: float, lat: float) -> bool:
        """True if (lon, lat) falls inside the grid footprint.

        Only a non-finite or non-numeric coordinate yields False; anything else
        (a missing registry entry, a bad proj4) raises, because a footprint test
        that silently answers False hides the error in an empty station list.
        """
        try:
            lon_f, lat_f = float(lon), float(lat)
        except (TypeError, ValueError):
            return False
        if not (math.isfinite(lon_f) and math.isfinite(lat_f)):
            return False
        return _contains(_polygon(self), lon_f, lat_f)

    def bbox(self) -> tuple[float, float, float, float]:
        ring = self._ring()
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        return min(lons), min(lats), max(lons), max(lats)


@functools.lru_cache(maxsize=64)
def _polygon(grid: Grid) -> np.ndarray:
    """Densified footprint ring as an (N, 2) array.

    Cached here rather than on the instance because Grid is a frozen slots
    dataclass and so has no __dict__.
    """
    return np.asarray(grid._ring(), dtype="float64")


def _contains(ring: np.ndarray, lon: float, lat: float) -> bool:
    """Crossing-number point-in-polygon. Avoids a matplotlib dependency for
    what is fifteen lines of ray casting."""
    x, y = ring[:, 0], ring[:, 1]
    x2, y2 = np.roll(x, -1), np.roll(y, -1)
    straddles = (y > lat) != (y2 > lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        x_cross = x + (lat - y) / (y2 - y) * (x2 - x)
    return bool(np.count_nonzero(straddles & (lon < x_cross)) % 2)


@functools.lru_cache(maxsize=4)
def load_registry(path: str | None = None) -> dict[str, Grid]:
    src = Path(path) if path else _DEFAULT_REGISTRY
    raw = yaml.safe_load(src.read_text()) or {}
    out: dict[str, Grid] = {}
    for gid, spec in raw.items():
        corners = spec.get("product_corners_lonlat")
        out[gid] = Grid(
            id=gid,
            proj4=spec["proj4"],
            shape=tuple(spec["shape"]),
            corners_lonlat=tuple(tuple(c) for c in corners) if corners else None,
            description=spec.get("description", ""),
        )
    return out


def get(grid_id: str, path: str | None = None) -> Grid:
    reg = load_registry(path)
    try:
        return reg[grid_id]
    except KeyError:
        raise KeyError(f"unknown grid {grid_id!r}; known: {', '.join(sorted(reg))}") from None
