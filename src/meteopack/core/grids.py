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
    #: Projected extent in the projection's own units, as
    #: ``{"x": (min, max), "y": (min, max)}``. Optional: without it the extent
    #: is projected from `corners_lonlat`, which is the same answer to within
    #: the float32 wobble in the corners KNMI writes.
    extent: tuple[tuple[float, float], tuple[float, float]] | None = None

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

    def projected_extent(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """``((x_min, x_max), (y_min, y_max))`` in projection units.

        From the registry when given, otherwise projected from the corners.
        """
        if self.extent is not None:
            return self.extent
        if not self.corners_lonlat:
            raise ValueError(f"grid {self.id} has neither extent nor corners_lonlat")
        from pyproj import Transformer

        fwd = Transformer.from_crs("EPSG:4326", self.proj4, always_xy=True)
        xy = [fwd.transform(lon, lat) for lon, lat in self.corners_lonlat]
        xs = [p[0] for p in xy]
        ys = [p[1] for p in xy]
        return ((min(xs), max(xs)), (min(ys), max(ys)))

    def coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre ``(y, x)`` coordinates in projection units.

        Centres, not edges: a pixel's coordinate is the middle of the pixel,
        which is what CF means by a coordinate variable and what every
        regridder assumes. `y` descends, because row 0 of a radar composite is
        the northern edge — the same orientation as `geo_row_offset` and a
        negative `geo_pixel_size_y` in the KNMI files.
        """
        (x0, x1), (y0, y1) = self.projected_extent()
        dx = (x1 - x0) / self.nx
        dy = (y1 - y0) / self.ny
        x = x0 + (np.arange(self.nx, dtype="float64") + 0.5) * dx
        y = y1 - (np.arange(self.ny, dtype="float64") + 0.5) * dy
        return y, x

    def lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-cell ``(lon, lat)`` as float32 ``(ny, nx)`` arrays.

        Written into the store so it is self-describing: a consumer can plot or
        co-locate without reconstructing the projection, and 4 MB for the KNMI
        grid is nothing against 15 GB of imagery.
        """
        from pyproj import Transformer

        y, x = self.coordinates()
        xx, yy = np.meshgrid(x, y)
        inv = Transformer.from_crs(self.proj4, "EPSG:4326", always_xy=True)
        lon, lat = inv.transform(xx, yy)
        return np.asarray(lon, dtype="float32"), np.asarray(lat, dtype="float32")

    @property
    def grid_mapping_name(self) -> str:
        """The CF grid_mapping_name of this grid's projection."""
        proj = next((p.split("=", 1)[1] for p in self.proj4.split()
                     if p.startswith("+proj=")), "")
        return _CF_GRID_MAPPINGS.get(proj, proj or "unknown")

    def bbox(self) -> tuple[float, float, float, float]:
        ring = self._ring()
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        return min(lons), min(lats), max(lons), max(lats)


#: PROJ projection name -> CF grid_mapping_name, for the ones in grids.yaml.
_CF_GRID_MAPPINGS = {
    "stere": "polar_stereographic",
    "laea": "lambert_azimuthal_equal_area",
    "lcc": "lambert_conformal_conic",
    "longlat": "latitude_longitude",
}


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
        # `extent_km` predates grids in metres; `extent` is in the
        # projection's own units, whatever +units says.
        extent = spec.get("extent") or spec.get("extent_km")
        out[gid] = Grid(
            id=gid,
            proj4=spec["proj4"],
            shape=tuple(spec["shape"]),
            corners_lonlat=tuple(tuple(c) for c in corners) if corners else None,
            description=spec.get("description", ""),
            extent=(
                (tuple(extent["x"]), tuple(extent["y"])) if extent else None
            ),
        )
    return out


def get(grid_id: str, path: str | None = None) -> Grid:
    reg = load_registry(path)
    try:
        return reg[grid_id]
    except KeyError:
        raise KeyError(f"unknown grid {grid_id!r}; known: {', '.join(sorted(reg))}") from None
