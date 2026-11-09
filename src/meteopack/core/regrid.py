"""Area-weighted (conservative) regridding from a regular lat/lon grid onto a registry grid.

The weights are built by *supersampling*: every target cell is split into
``oversample x oversample`` equal sub-cells in its own projected plane, each
sub-cell centre is mapped back to lon/lat and assigned to the source cell that
contains it, and a source cell's weight in the target cell is the fraction of
sub-cells it received. That is the first-order conservative remap with the
cell-overlap areas estimated on a fine raster instead of computed from
polygons: for a 4 km target over a ~2 km source at ``oversample=10`` (0.4 km
sub-cells) each overlap fraction is off by at most about one sub-cell row,
and the weights of a fully covered cell sum to exactly 1, so the area mean of a
constant is that constant and a field's total over fully covered cells is
kept. Weights are in target-plane area, which for a conformal projection at
these scales is the true area to within the scale-factor change across one
cell (< 0.1 %).

Applying them is ``bincount`` over the (target, source, weight) triplets - no
sparse-matrix library needed - with NaN source values left out and the
remaining weights renormalised:

* ``coverage`` - the fraction of a target cell inside the source domain. Below
  `min_coverage` the cell is *outside* (the caller's out-of-image code).
* ``valid_fraction`` - the share of the covered part with a finite source
  value. Below `min_valid` the cell is *undefined* (NaN): the caller decides
  what that means (missing data, or e.g. "no cloud base").
"""
from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from meteopack.core.grids import Grid


@dataclass(frozen=True, slots=True)
class LatLonGrid:
    """A regular lon/lat source grid by its first cell centre, step and size.

    Rows run south to north (``lat0`` is the southernmost row centre), columns
    west to east - the orientation the readers hand fields over in.
    """

    lat0: float
    dlat: float
    nlat: int
    lon0: float
    dlon: float
    nlon: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.nlat, self.nlon)


@dataclass(frozen=True, slots=True)
class AreaWeights:
    target_shape: tuple[int, int]
    source_shape: tuple[int, int]
    #: Flat target index, flat source index, weight (fraction of the target cell).
    target: np.ndarray
    source: np.ndarray
    weight: np.ndarray
    #: Per target cell: the fraction of its area inside the source domain.
    coverage: np.ndarray
    oversample: int

    def apply(self, field: np.ndarray, *, min_coverage: float = 0.5,
              min_valid: float = 0.5) -> np.ndarray:
        """Area mean of `field` ``(..., nlat, nlon)`` per target cell -> ``(..., ny, nx)``.

        Outside (coverage < `min_coverage`) and undefined (valid share <
        `min_valid`) cells are NaN; :meth:`outside` tells the two apart.
        """
        values = np.asarray(field, dtype="float64")
        lead = values.shape[:-2]
        if values.shape[-2:] != self.source_shape:
            raise ValueError(f"field is {values.shape[-2:]}, weights expect {self.source_shape}")
        flat = values.reshape(-1, self.source_shape[0] * self.source_shape[1])
        n_target = self.target_shape[0] * self.target_shape[1]
        out = np.empty((flat.shape[0], n_target), dtype="float64")
        outside = self.coverage < min_coverage
        for k, row in enumerate(flat):
            picked = row[self.source]
            ok = np.isfinite(picked)
            w_ok = np.where(ok, self.weight, 0.0)
            num = np.bincount(self.target, weights=np.where(ok, picked, 0.0) * w_ok,
                              minlength=n_target)
            den = np.bincount(self.target, weights=w_ok, minlength=n_target)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = num / den
                share = den / self.coverage
            mean[outside | ~(share >= min_valid) | (den <= 0)] = np.nan
            out[k] = mean
        return out.reshape(*lead, *self.target_shape)

    def valid_share(self, field: np.ndarray) -> np.ndarray:
        """Per target cell, the share of its covered area with a finite value (2-D field)."""
        values = np.asarray(field, dtype="float64").reshape(-1)
        ok = np.isfinite(values[self.source]).astype("float64")
        n_target = self.target_shape[0] * self.target_shape[1]
        den = np.bincount(self.target, weights=ok * self.weight, minlength=n_target)
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(self.coverage > 0, den / self.coverage, 0.0)
        return share.reshape(self.target_shape)

    def outside(self, min_coverage: float = 0.5) -> np.ndarray:
        """``(ny, nx)`` True where the target cell lies (mostly) outside the source domain."""
        return (self.coverage < min_coverage).reshape(self.target_shape)


def latlon_to_grid(source: LatLonGrid, grid: Grid, oversample: int = 10) -> AreaWeights:
    """Supersampled area weights from `source` onto `grid` (cached per arguments)."""
    return _weights(source, grid, int(oversample))


@functools.lru_cache(maxsize=8)
def _weights(source: LatLonGrid, grid: Grid, oversample: int) -> AreaWeights:
    from pyproj import Transformer

    (x0, x1), (y0, y1) = grid.projected_extent()
    ny, nx = grid.shape
    dx, dy = (x1 - x0) / nx, (y1 - y0) / ny
    # Sub-cell centres, row-major from the north-west corner (row 0 is north).
    sub = (np.arange(oversample) + 0.5) / oversample
    xs = x0 + (np.arange(nx)[:, None] + sub[None, :]).reshape(-1) * dx
    ys = y1 - (np.arange(ny)[:, None] + sub[None, :]).reshape(-1) * dy
    inv = Transformer.from_crs(grid.proj4, "EPSG:4326", always_xy=True)
    n_sub = oversample * oversample
    targets, sources = [], []
    # One target row at a time keeps the transform's arrays small.
    for i in range(ny):
        yy, xx = np.meshgrid(ys[i * oversample:(i + 1) * oversample], xs, indexing="ij")
        lon, lat = inv.transform(xx.reshape(-1), yy.reshape(-1))
        col = np.rint((np.asarray(lon) - source.lon0) / source.dlon).astype("int64")
        row = np.rint((np.asarray(lat) - source.lat0) / source.dlat).astype("int64")
        inside = (col >= 0) & (col < source.nlon) & (row >= 0) & (row < source.nlat)
        # Which target cell each sub-cell belongs to: xx varies fastest.
        j = np.broadcast_to((np.arange(nx * oversample) // oversample)[None, :],
                            (oversample, nx * oversample)).reshape(-1)
        targets.append((i * nx + j)[inside])
        sources.append((row * source.nlon + col)[inside])
    t = np.concatenate(targets)
    s = np.concatenate(sources)
    key = t * (source.nlat * source.nlon) + s
    uniq, counts = np.unique(key, return_counts=True)
    t_u = uniq // (source.nlat * source.nlon)
    s_u = uniq % (source.nlat * source.nlon)
    w = counts.astype("float64") / n_sub
    coverage = np.bincount(t_u, weights=w, minlength=ny * nx)
    return AreaWeights(target_shape=(ny, nx), source_shape=source.shape, target=t_u,
                       source=s_u, weight=w, coverage=coverage, oversample=oversample)


__all__ = ["AreaWeights", "LatLonGrid", "latlon_to_grid"]
