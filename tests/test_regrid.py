"""Supersampled area-mean regridding from a regular lat/lon grid onto a registry grid."""
from __future__ import annotations

import numpy as np
import pytest

from meteopack.core import grids, regrid

STEREO = ("+proj=stere +lat_0=90 +lon_0=0 +lat_ts=60 +a=6378140 +b=6356750 +x_0=0 +y_0=0 "
          "+units=km +no_defs")


def target(extent=((200.0, 360.0), (-4100.0, -3940.0)), shape=(8, 8)) -> grids.Grid:
    """A 20 km grid over the Netherlands (inside the source below) unless told otherwise."""
    return grids.Grid(id="t", proj4=STEREO, shape=shape, extent=extent)


SOURCE = regrid.LatLonGrid(lat0=49.0, dlat=0.09, nlat=80, lon0=0.0, dlon=0.145, nlon=79)


def test_weights_of_a_covered_cell_sum_to_one_and_constants_survive():
    w = regrid.latlon_to_grid(SOURCE, target(), oversample=8)
    np.testing.assert_allclose(w.coverage, 1.0)
    out = w.apply(np.full(SOURCE.shape, 7.25))
    np.testing.assert_allclose(out, 7.25)


def test_area_mean_matches_a_brute_force_supersample():
    from pyproj import Transformer

    rng = np.random.default_rng(0)
    field = rng.uniform(0, 10, size=SOURCE.shape)
    grid = target()
    w = regrid.latlon_to_grid(SOURCE, grid, oversample=6)
    got = w.apply(field)
    # Independent: 6 x 6 sample points per cell, nearest source cell, plain mean.
    (x0, x1), (y0, y1) = grid.projected_extent()
    inv = Transformer.from_crs(grid.proj4, "EPSG:4326", always_xy=True)
    want = np.empty(grid.shape)
    for i in range(grid.ny):
        for j in range(grid.nx):
            sub = (np.arange(6) + 0.5) / 6
            xs = x0 + (j + sub) * (x1 - x0) / grid.nx
            ys = y1 - (i + sub) * (y1 - y0) / grid.ny
            xx, yy = np.meshgrid(xs, ys)
            lon, lat = inv.transform(xx.ravel(), yy.ravel())
            c = np.rint((lon - SOURCE.lon0) / SOURCE.dlon).astype(int)
            r = np.rint((lat - SOURCE.lat0) / SOURCE.dlat).astype(int)
            want[i, j] = field[r, c].mean()
    np.testing.assert_allclose(got, want, rtol=1e-12)


def test_cells_outside_the_domain_and_the_coverage_threshold():
    # Straddles the source's western edge (lon -0.0725) near 52N.
    edge = target(extent=((-200.0, 200.0), (-4100.0, -3940.0)), shape=(4, 10))
    w = regrid.latlon_to_grid(SOURCE, edge, oversample=10)
    cov = w.coverage.reshape(edge.shape)
    assert (cov[:, 0] == 0).all() and np.allclose(cov[:, -1], 1.0)
    partial = (cov > 0) & (cov < 1)
    assert partial.any()
    out = w.apply(np.ones(SOURCE.shape), min_coverage=0.5)
    np.testing.assert_array_equal(np.isnan(out), cov < 0.5)
    np.testing.assert_array_equal(w.outside(0.5), cov < 0.5)
    # A cell half inside is the mean of its inside part, not diluted.
    np.testing.assert_allclose(out[cov >= 0.5], 1.0)


def test_nan_source_values_are_left_out_and_counted():
    field = np.full(SOURCE.shape, 2.0)
    w = regrid.latlon_to_grid(SOURCE, target(), oversample=8)
    all_nan = np.full(SOURCE.shape, np.nan)
    assert np.isnan(w.apply(all_nan)).all()
    np.testing.assert_array_equal(w.valid_share(all_nan), 0.0)
    # A third of the domain NaN: cells keep the mean of what is left, unless
    # less than min_valid of them has a value.
    field[:, : SOURCE.nlon // 3] = np.nan
    share = w.valid_share(field)
    out = w.apply(field, min_valid=0.5)
    np.testing.assert_allclose(out[share >= 0.5], 2.0)
    assert np.isnan(out[share < 0.5]).all()


def test_shape_mismatch_and_leading_axes():
    w = regrid.latlon_to_grid(SOURCE, target(), oversample=4)
    stack = np.stack([np.full(SOURCE.shape, k, dtype="float64") for k in range(3)])
    out = w.apply(stack)
    assert out.shape == (3, 8, 8)
    np.testing.assert_allclose(out[:, 0, 0], [0, 1, 2])
    with pytest.raises(ValueError, match="weights expect"):
        w.apply(np.zeros((5, 5)))


def test_the_model_grid_is_pixel_aligned_with_rtcor():
    """knmi_4km_stereo_1024: every 4 km edge is a whole km, so each cell is 4 x 4 RTCOR pixels."""
    g = grids.get("knmi_4km_stereo_1024")
    (x0, x1), (y0, y1) = g.projected_extent()
    assert g.shape == (256, 256) and (x1 - x0, y1 - y0) == (1024.0, 1024.0)
    assert ((x0 + x1) / 2, (y0 + y1) / 2) == (350.0, -4032.0)
    rtcor = grids.get("knmi_1km_stereo")
    (rx0, rx1), (ry0, ry1) = rtcor.projected_extent()
    # Whole-km edges, like RTCOR's: a 4 km cell is exactly 4 x 4 RTCOR pixels.
    assert all(float(v).is_integer() for v in (x0, x1, y0, y1, rx0, rx1, ry0, ry1))
    # And every edge of the 2 km stage (512 km square, x 94..606, y -4288..-3776)
    # is a 4 km edge too.
    assert (94 - x0) % 4 == 0 and (y1 - -3776) % 4 == 0
    y, x = g.coordinates()
    assert x[0] == -160.0 and y[0] == -3522.0 and y[1] < y[0]
