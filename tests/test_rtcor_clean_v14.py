"""Contract checks for the RTCOR clutter cleaner v14 on synthetic full-grid windows (no stored fixtures).

Bit-for-bit equivalence with the research cleaner on real windows: scripts/rtcor_clean_v14_equivalence.py.
"""
import json

import numpy as np
import pytest
import zarr

pytest.importorskip('cv2')
pytest.importorskip('scipy')
from meteopack.datasets import rtcor  # noqa: E402
from meteopack.datasets import rtcor_clean_v14 as v14  # noqa: E402
from meteopack.datasets.rtcor_clean_v14 import assets as v14_assets  # noqa: E402

if not v14.ASSETS.exists():
    pytest.skip('cleaner assets not built (scripts/rtcor_clean_v14_assets.py)', allow_module_level=True)

H, W = 765, 700
STEP = np.timedelta64(5, 'm')
DEN_HELDER = (331.5, 333.2)


def times(n, start='2022-07-19T22:00'):
    return np.datetime64(start, 'ns') + np.arange(n) * STEP


def scene(n=12, seed=0):
    """Moving rain, one-frame specks in clear air, one interference ray, missing data and an outside band."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    out = np.zeros((n, H, W), 'uint16')
    for t in range(n):
        d2 = (yy - 450) ** 2 + (xx - 250 - 3 * t) ** 2
        rain = np.where(d2 <= 45 ** 2, np.maximum(np.rint(60 * np.exp(-d2 / (2 * 20.0 ** 2))), 1), 0)
        out[t] = rain
        ys, xs = rng.integers(150, 300, 60), rng.integers(450, 650, 60)
        out[t, ys, xs] = rng.integers(3, 30, 60)
    az = np.radians(60.0)                       # a ray from Den Helder in frame 6, 40-200 km out
    r = np.arange(40, 200, 0.5)
    ry = np.rint(DEN_HELDER[0] - r * np.cos(az)).astype(int)
    rx = np.rint(DEN_HELDER[1] + r * np.sin(az)).astype(int)
    out[6, ry, rx] = 40
    out[:, :30, :] = 65535                      # outside the image
    out[4, 600:650, 100:200] = 65534            # missing data
    return out


@pytest.fixture(scope='module')
def old_run():
    codes = scene()
    return codes, v14.clean_window(codes, times(len(codes)))


def test_contract_shape_dtype_mask_never_raised(old_run):
    codes, out = old_run
    assert out.shape == codes.shape and out.dtype == np.uint16
    invalid = codes >= 65534
    assert np.array_equal(out[invalid], codes[invalid])         # radar mask unchanged
    assert ((out >= 65534) == invalid).all()
    assert (out <= codes).all()                                 # never raised
    assert (out[codes == 0] == 0).all()                         # no echo added


def test_clutter_is_removed_and_rain_is_kept(old_run):
    codes, out = old_run
    core = slice(2, -2)
    specks = codes[core, 150:300, 450:650].astype(np.int64)
    kept = out[core, 150:300, 450:650].astype(np.int64)
    assert kept.sum() <= 0.2 * specks.sum()                     # one-frame specks in clear air
    rain = codes[core, 380:520, 180:340].astype(np.int64)
    assert out[core, 380:520, 180:340].astype(np.int64).sum() >= 0.95 * rain.sum()


def test_new_chain_passes_unchanged():
    codes = scene(8)
    out = v14.clean_window(codes, times(8, '2024-07-19T22:00'))
    assert np.array_equal(out, codes)
    cleaned, info = v14.clean(codes, times(8, '2024-07-19T22:00'))
    assert np.array_equal(cleaned, codes) and not info['old_chain'].any()


def test_deterministic(old_run):
    codes, out = old_run
    assert np.array_equal(v14.clean_window(codes, times(len(codes))), out)


def test_clean_blocks_and_context(old_run):
    codes, out = old_run
    T = len(codes)
    one, info = v14.clean(codes, times(T), context=1)          # one block, pad covers the window
    assert np.array_equal(one[1:-1], out[1:-1])
    assert np.array_equal(one[[0, -1]], codes[[0, -1]])         # context frames unchanged
    assert info['cleaned'].tolist() == [False] + [True] * (T - 2) + [False]
    assert (info['clean_sum_code'] <= info['raw_sum_code']).all()
    assert info['changed_pixels'][[0, -1]].sum() == 0
    blk, _ = v14.clean(codes, times(T), context=(2, 3), block=4, pad=3)
    assert np.array_equal(blk[:2], codes[:2]) and np.array_equal(blk[-3:], codes[-3:])
    assert (blk <= codes).all() and np.array_equal(blk >= 65534, codes >= 65534)
    with pytest.raises(ValueError):
        v14.clean(codes[:, :100], times(T))


def test_registered_post_and_final_stages(old_run):
    codes, out = old_run
    calls = []

    def halve(ctx, cur, factor=2):
        calls.append(factor)
        return (cur // factor).astype(cur.dtype), None
    v14.register_stage('test_halve', halve)
    res = v14.clean_window(codes, times(len(codes)), post=('test_halve',), test_halve_kw={'factor': 2})
    assert calls == [2]
    valid = codes < 65534
    assert np.array_equal(res[~valid], codes[~valid]) and (res <= codes).all()
    assert res[valid].astype(np.int64).sum() < out[valid].astype(np.int64).sum()

    def raise_all(ctx, cur):                                    # a final stage may not raise or touch the mask
        return np.full_like(cur, 60000), None
    v14.register_stage('test_raise', raise_all)
    res = v14.clean_window(codes, times(len(codes)), final=('test_raise',))
    assert np.array_equal(res, codes)                           # clipped back to the raw codes


@pytest.fixture(scope='module')
def base_run(old_run):
    """The window cleaned without the post and final stages (aggr 1): reference for the single-stage runs."""
    codes, _ = old_run
    return v14.clean_window(codes, times(len(codes)), 1.0, post=(), final=())


@pytest.mark.parametrize('name', v14.POST_STAGES)
def test_each_post_stage_keeps_the_contract(old_run, base_run, name):
    codes, _ = old_run
    res = v14.clean_window(codes, times(len(codes)), 1.0, post=(name,), final=())
    invalid = codes >= 65534
    assert res.dtype == np.uint16 and np.array_equal(res[invalid], codes[invalid])
    assert (res <= base_run).all() and (res <= codes).all()     # only lowers what the base kept
    new = v14.clean_window(codes, times(len(codes), '2024-07-19T22:00'), 1.0, post=(name,), final=())
    assert np.array_equal(new, codes)


def test_residue_final_stage_keeps_the_contract(old_run):
    codes, _ = old_run
    T = len(codes)
    assert v14.FINAL_STAGES == ('residue',)
    base = v14.clean_window(codes, times(T), 2.0, post=(), final=())     # aggr 2: the residue gate is open
    res = v14.clean_window(codes, times(T), 2.0, post=(), final=('residue',))
    invalid = codes >= 65534
    assert res.dtype == np.uint16 and np.array_equal(res[invalid], codes[invalid])
    assert (res <= base).all() and (res <= codes).all()
    big = slice(2, -2), slice(380, 520), slice(180, 340)                 # the coherent rain area is untouched
    assert np.array_equal(res[big], base[big])
    new = v14.clean_window(codes, times(T, '2024-07-19T22:00'), 2.0, post=(), final=('residue',))
    assert np.array_equal(new, codes)


def ray_scene(n=10, az_deg=75.0, frames=range(2, 9)):
    """A straight interference ray from Den Helder (40-250 km, 1 px wide) in clear air in several frames, plus a
    small moving shower so the scene has motion."""
    codes = np.zeros((n, H, W), 'uint16')
    yy, xx = np.mgrid[0:H, 0:W]
    for t in range(n):
        d2 = (yy - 600) ** 2 + (xx - 200 - 3 * t) ** 2
        codes[t] = np.where(d2 <= 20 ** 2, 20, 0)
    az = np.radians(az_deg)
    r = np.arange(40, 250, 0.5)
    ry = np.rint(DEN_HELDER[0] - r * np.cos(az)).astype(int)
    rx = np.rint(DEN_HELDER[1] + r * np.sin(az)).astype(int)
    ok = (ry >= 0) & (ry < H) & (rx >= 0) & (rx < W)
    ry, rx = ry[ok], rx[ok]
    ray = np.zeros((n, H, W), bool)
    for t in frames:
        codes[t, ry, rx] = 25
        ray[t, ry, rx] = True
    return codes, ray


def test_spike_ray_in_clear_air_is_removed():
    codes, ray = ray_scene()
    out = v14.clean_window(codes, times(len(codes)))
    assert (out[ray] == 0).mean() >= 0.95
    from meteopack.datasets.rtcor_clean_v14 import spikes2
    alone = spikes2.remove_spikes(codes, None, raw=codes)       # the spikes2 stage by itself
    assert (alone[ray] == 0).mean() >= 0.9
    assert (alone <= codes).all() and alone[~ray].astype(np.int64).sum() >= 0.99 * codes[~ray].astype(np.int64).sum()


def test_tree_tables_evaluate_like_a_hand_made_tree():
    # one tree: x0 <= 0.5 -> leaf -1, else x1 <= 2 (NaN right) -> leaf 0.5 / leaf 1.0; base 0.25
    t = {'leaf': np.array([0, 1, 0, 1, 1], bool), 'feat': np.array([0, 0, 1, 0, 0]),
         'thr': np.array([0.5, 0, 2.0, 0, 0]), 'mleft': np.array([1, 0, 0, 0, 0], bool),
         'left': np.array([1, 1, 3, 3, 4]), 'right': np.array([2, 1, 4, 3, 4]),
         'value': np.array([0, -1.0, 0, 0.5, 1.0]), 'roots': np.array([0]), 'depth': 2, 'base': 0.25}
    X = np.array([[0.0, 9.0], [1.0, 1.0], [1.0, 3.0], [np.nan, 0.0], [1.0, np.nan]])
    raw = 0.25 + np.array([-1.0, 0.5, 1.0, -1.0, 1.0])
    assert np.allclose(v14_assets.HGB(t).proba(X), 1 / (1 + np.exp(-raw)))


def test_fingerprint_names_version_and_hashes():
    fp = v14.fingerprint()
    assert fp['version'] == v14.VERSION and len(fp['code_sha256']) == 64 and len(fp['assets_sha256']) == 64


def _source(path, n=14, start='2021-07-14T11:00'):
    g = zarr.open_group(path, mode='w', zarr_format=3)
    g.create_array('time', data=times(n, start).astype('int64'))
    g.create_array('is_missing_frame', data=np.zeros(n, bool))
    g.create_array('is_corrupt_frame', data=np.zeros(n, bool))
    a = g.create_array('accumulation_mm', data=scene(n, seed=1), chunks=(1, H, W), shards=(7, H, W))
    a.attrs.update({'scale_factor': .01, 'add_offset': 0., 'calibration_missing_data': 65534,
                    'calibration_out_of_image': 65535})


def _cfg(source, output):
    return {'source': str(source), 'output': str(output), 'start': None, 'end': None, 'clean': True,
            'cleaner': 'v14', 'cadence_seconds': 300, 'crop': [380, 540, 160, 360],
            'encoding': {'dry_threshold_mm': .01, 'step_mm': .02},
            'storage': {'chunk_frames': 4, 'shard_frames': 8, 'zstd_level': 3},
            'samples': {'history': 4, 'forecast': 4, 'stride': 4, 'origin': '2021-07-14T00:00:00Z'}}


def test_cleaned_build_v14_workers_match_serial_and_direct_clean(tmp_path):
    source = tmp_path / 'src' / 'dataset.zarr'; source.parent.mkdir()
    _source(source)
    serial = rtcor.build(_cfg(source, tmp_path / 'serial'))
    cfg = _cfg(source, tmp_path / 'parallel')
    rtcor.build(cfg, prepare=True)
    for k in range(2):
        rtcor.build(cfg, worker=(k, 2))
    parallel = rtcor.build(cfg)
    a = zarr.open_group(serial / 'data.zarr', mode='r'); b = zarr.open_group(parallel / 'data.zarr', mode='r')
    assert np.array_equal(a['rain_code'][:], b['rain_code'][:])
    status = np.asarray(b['frame_status'][:])
    assert status[0] == 4 and status[-1] == 4 and (status[1:-1] == 0).all()
    # shard 0 = frames 0..7 cleaned from frames 0..14 (pad 7), as clean() would do directly
    raw = scene(14, seed=1)
    direct, _ = v14.clean(raw[:14], times(14, '2021-07-14T11:00'), context=(1, 6))
    codes, _, _ = rtcor.encode_rain(direct[1:8, 380:540, 160:360], missing=65534, outside=65535,
                                    source_scale=.01, threshold=.01, step=.02)
    assert np.array_equal(np.asarray(a['rain_code'][1:8]), codes)
    m = json.loads((parallel / 'manifest.json').read_text())
    assert m['cleaning']['version'] == v14.VERSION
    assert m['cleaning']['statistics']['frames_cleaned'] == 12
    assert set(v14.STATS) <= set(m['cleaning']['statistics'])
