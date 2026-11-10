"""Pack the fixed inputs of the RTCOR clutter cleaner into src/meteopack/datasets/assets/rtcor_clean_v5.npz.

Inputs are the outputs of the clutter study (research_nowcasting/analysis/rtcor_clutter/exp_v4/out):
  tile_model_E.npz  logistic tile model (fit_v5.py: gauge, satellite and curated labels)
  static_map.npz    old-chain static clutter (static_map.py)
  coverage.npz      common footprint (coverage.py)
and the RTCOR store, for the grid coordinates of the radar sites.

Run: python scripts/rtcor_clean_assets.py --study DIR --store PATH/dataset.zarr
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr

OUT = Path(__file__).resolve().parents[1] / 'src/meteopack/datasets/assets/rtcor_clean_v5.npz'
# Radars whose interference spikes reach the RTCOR grid (lat, lon).
SITES = {'Den Helder': (52.9528, 4.7906), 'Herwijnen': (51.8369, 5.1381),
         'Jabbeke': (51.1917, 3.0642), 'Wideumont': (49.9135, 5.5044),
         'Borkum': (53.5640, 6.7483), 'Essen': (51.4056, 6.9670),
         'Neuheilenbach': (50.1097, 6.5485), 'Helchteren': (51.0692, 5.4058)}


def ll_to_rc(lat, lon, xs, ys):
    """Fractional (row, col) on the KNMI 1 km polar stereographic grid (lat_ts 60, lon_0 0,
    ellipsoid a=6378.140, b=6356.750 km; Snyder 1987), given the store's x/y cell centres."""
    a, b = 6378.140, 6356.750
    e = np.sqrt(1 - (b / a) ** 2)

    def t_(phi):
        s = np.sin(phi)
        return np.tan(np.pi / 4 - phi / 2) / ((1 - e * s) / (1 + e * s)) ** (e / 2)
    phic = np.radians(60.0)
    mc = np.cos(phic) / np.sqrt(1 - e ** 2 * np.sin(phic) ** 2)
    rho = a * mc * t_(np.radians(lat)) / t_(phic)
    x, y = rho * np.sin(np.radians(lon)), -rho * np.cos(np.radians(lon))
    return (y - ys[0]) / (ys[1] - ys[0]), (x - xs[0]) / (xs[1] - xs[0])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True, help='directory with the three .npz inputs')
    p.add_argument('--store', type=Path, required=True, help='RTCOR dataset.zarr')
    p.add_argument('--out', type=Path, default=OUT)
    args = p.parse_args()
    model = np.load(args.study / 'tile_model_E.npz')
    static = np.load(args.study / 'static_map.npz')['mask'].astype(bool)
    cover = np.load(args.study / 'coverage.npz')['mask'].astype(bool)
    g = zarr.open_group(args.store, mode='r')
    xs, ys = np.asarray(g['x'][:]), np.asarray(g['y'][:])
    if static.shape != cover.shape or cover.shape != (len(ys), len(xs)):
        raise ValueError('mask shapes do not match the store grid')
    site_rc = np.array([ll_to_rc(la, lo, xs, ys) for la, lo in SITES.values()], dtype=float)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, coef=model['b'], mu=model['mu'], sd=model['sd'],
                        feats=np.array([str(f) for f in model['feats']]), static=static, cover=cover,
                        site_names=np.array(list(SITES)), site_rc=site_rc)
    print(f'{args.out}: {static.sum()} static clutter px, footprint {cover.sum()} km2,',
          'sites', dict(zip(SITES, np.round(site_rc, 1).tolist())))


if __name__ == '__main__':
    main()
