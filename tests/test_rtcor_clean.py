"""Behaviour checks for the RTCOR clutter cleaner and the cleaned dataset build, on small grids."""
import json

import numpy as np
import pytest
import zarr

pytest.importorskip('pysteps')
pytest.importorskip('cv2')
from meteopack.datasets import rtcor, rtcor_clean as rc  # noqa: E402

if not rc.ASSETS.exists():
    pytest.skip('cleaner assets not built (scripts/rtcor_clean_assets.py)', allow_module_level=True)

SHAPE = (160, 160)
SITE = (80.0, 80.0)
STEP = np.timedelta64(5, 'm')


def small_assets(static=None, cover=None):
    real = rc.load_assets()
    return rc.Assets(coef=real.coef, mu=real.mu, sd=real.sd, feats=real.feats,
                     static=np.zeros(SHAPE, bool) if static is None else static,
                     cover=np.ones(SHAPE, bool) if cover is None else cover, sites=(SITE,))


def times(n, start='2024-01-15T12:00'):
    return np.datetime64(start, 'ns') + np.arange(n) * STEP


def rain_blob(n, peak_code=40, radius=25.0, speed=2.0, row=40, col=40):
    """A smooth Gaussian rain area moving east by `speed` km per 5 min."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    out = np.zeros((n,) + SHAPE, 'uint16')
    for t in range(n):
        d2 = (yy - row) ** 2 + (xx - col - speed * t) ** 2
        v = peak_code * np.exp(-d2 / (2 * (radius / 2) ** 2))
        out[t] = np.where(d2 <= radius ** 2, np.maximum(np.rint(v), 1), 0)
    return out


def test_dry_frames_stay_dry_and_footprint_is_applied():
    cover = np.ones(SHAPE, bool); cover[:, :5] = False
    codes = np.zeros((5,) + SHAPE, 'uint16')
    out, info = rc.clean(codes, times(5), small_assets(cover=cover))
    assert (out[1:-1][:, :, :5] == rc.OUTSIDE).all()
    assert (out[1:-1][:, :, 5:] == 0).all()
    assert (out[[0, -1]] == 0).all()                    # context frames are returned unchanged
    assert info['cleaned'].tolist() == [False, True, True, True, False]


def test_moving_smooth_rain_is_kept():
    codes = rain_blob(7)
    out, info = rc.clean(codes, times(7), small_assets())
    raw, cleaned = codes[1:-1].astype(int).sum(), out[1:-1].astype(int).sum()
    assert cleaned >= 0.99 * raw
    assert info['spike_pixels'].sum() == 0


def test_spike_in_clear_air_is_removed():
    codes = np.zeros((5,) + SHAPE, 'uint16')
    codes[:, 80:82, 100:160] = 50                       # 2 km wide radial line, 20-80 km east of the radar
    out, info = rc.clean(codes, times(5), small_assets())
    assert (info['spike_pixels'][1:-1] > 0).all()
    assert (out[1:-1, 80:82, 100:160] == 0).mean() >= 0.95


def test_static_clutter_inside_rain_is_refilled_with_rain():
    codes = rain_blob(5, peak_code=30, radius=30.0, row=60, col=60, speed=0.0)
    static = np.zeros(SHAPE, bool); static[59:62, 59:62] = True
    hot = codes.copy(); hot[:, 59:62, 59:62] = 200
    out, _ = rc.clean(hot, times(5, '2021-03-01T00:00'), small_assets(static=static))
    filled, truth = out[2, 59:62, 59:62].astype(int), codes[2, 59:62, 59:62].astype(int)
    assert (np.abs(filled - truth) <= 0.2 * truth).all()   # rain under the clutter, not 0 or 200


def test_static_clutter_map_applies_to_old_chain_only():
    static = np.zeros(SHAPE, bool); static[30:33, 120:123] = True
    codes = np.zeros((5,) + SHAPE, 'uint16'); codes[:, 30:33, 120:123] = 80
    a = small_assets(static=static)
    out_old, info_old = rc.clean(codes, times(5, '2021-03-01T00:00'), a)
    _, info_new = rc.clean(codes, times(5, '2024-03-01T00:00'), a)
    assert (out_old[1:-1, 30:33, 120:123] == 0).all()
    assert info_old['spike_pixels'][1:-1].sum() > 0 and info_new['spike_pixels'].sum() == 0


def _source(path, n=30, start='2018-12-01T00:00'):
    g = zarr.open_group(path, mode='w', zarr_format=3)
    g.create_array('time', data=times(n, start).astype('int64'))
    g.create_array('is_missing_frame', data=np.zeros(n, bool))
    g.create_array('is_corrupt_frame', data=np.zeros(n, bool))
    a = g.create_array('accumulation_mm', data=rain_blob(n, speed=1.5), chunks=(1,) + SHAPE,
                       shards=(10,) + SHAPE)
    a.attrs.update({'scale_factor': .01, 'add_offset': 0., 'calibration_missing_data': 65534,
                    'calibration_out_of_image': 65535})


def _cfg(source, output):
    return {'source': str(source), 'output': str(output), 'start': None, 'end': None, 'clean': True,
            'cadence_seconds': 300, 'crop': [10, 74, 10, 74],
            'encoding': {'dry_threshold_mm': .01, 'step_mm': .02},
            'storage': {'chunk_frames': 4, 'shard_frames': 8, 'zstd_level': 3},
            'samples': {'history': 4, 'forecast': 18, 'stride': 4, 'origin': '2018-12-01T00:00:00Z'}}


def test_cleaned_build_parallel_workers_match_serial(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, 'load_assets', lambda path=rc.ASSETS: small_assets())
    source = tmp_path / 'src' / 'dataset.zarr'; source.parent.mkdir()
    _source(source)
    serial = rtcor.build(_cfg(source, tmp_path / 'serial'))
    cfg = _cfg(source, tmp_path / 'parallel')
    with pytest.raises(ValueError, match='--prepare'):
        rtcor.build(cfg, worker=(0, 2))
    rtcor.build(cfg, prepare=True)
    for k in range(2):
        rtcor.build(cfg, worker=(k, 2))
    assert not (tmp_path / 'parallel' / 'manifest.json').exists()
    parallel = rtcor.build(cfg)
    a = zarr.open_group(serial / 'data.zarr', mode='r'); b = zarr.open_group(parallel / 'data.zarr', mode='r')
    assert np.array_equal(a['rain_code'][:], b['rain_code'][:])
    status = np.asarray(b['frame_status'][:])
    assert status[0] == 4 and status[-1] == 4 and (status[1:-1] == 0).all()   # archive ends: no context
    m = json.loads((parallel / 'manifest.json').read_text())
    assert m['cleaning']['version'] == rc.VERSION
    assert m['cleaning']['statistics']['frames_cleaned'] == 28
    assert m['frame_status_counts']['4'] == 2
