"""Contract checks for RTCOR dataset generation."""
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import zarr

from meteopack.datasets.common import sample_index
from meteopack.datasets.rtcor import build, encode_rain
from meteopack.datasets.reader import TimelineDataset


def test_encoding_threshold_rounding_masks_and_cap():
    raw = np.array([0, 1, 2, 3, 4, 508, 509, 510, 65534, 65535], dtype='uint16')
    codes, missing, counts = encode_rain(raw, missing=65534, outside=65535,
        source_scale=.01, threshold=.01, step=.02)
    assert codes.tolist() == [0, 0, 1, 2, 2, 254, 254, 254, 255, 255]
    assert missing.tolist() == [False]*8 + [True, False]
    assert counts == {'thresholded_wet_pixels': 1, 'clipped_pixels': 2,
                      'source_missing_pixels': 1, 'outside_pixels': 1}


def test_index_rejects_missing_and_split_boundary():
    times = np.datetime64('2024-01-31T22:00', 'ns') + np.arange(72) * np.timedelta64(5, 'm')
    status = np.zeros(72, dtype='uint8'); status[4] = 1
    index, counts = sample_index(times, status, history=4, forecast=18, stride=4,
                                 origin='2024-01-31T22:00:00Z', cadence_seconds=300)
    assert counts['invalid_frame'] == 2
    assert counts['split_crossing'] > 0
    for row in index.to_dicts():
        i = row['start_idx']; used = times[i:i+22]
        assert not (i <= 4 < i+22)
        split = used.astype('datetime64[D]') == used.astype('datetime64[M]').astype('datetime64[D]')
        assert np.all(split == split[0])
        assert row['split'] == ('validation' if split[0] else 'train')
        assert row['issue_time'] == used[3].astype('datetime64[us]').astype(object)


def test_build_real_zarr_and_resume(tmp_path):
    source = tmp_path/'source'/'dataset.zarr'; source.parent.mkdir()
    (source.parent/'manifest.json').write_text('{"version":1}')
    g = zarr.open_group(source, mode='w', zarr_format=3)
    t0 = np.datetime64('2024-01-31T22:00', 'ns')
    times=(t0 + np.arange(72)*np.timedelta64(5,'m')).astype('int64')
    g.create_array('time', data=times)
    missing=np.zeros(72,dtype=bool); missing[4]=True
    g.create_array('is_missing_frame',data=missing)
    g.create_array('is_corrupt_frame',data=np.zeros(72,dtype=bool))
    data=np.full((72,4,4), 2,dtype='uint16'); data[25,1,1]=65534; data[50,2,2]=65535
    a=g.create_array('accumulation_mm',data=data,chunks=(1,4,4),shards=(24,4,4))
    a.attrs.update({'scale_factor':.01,'add_offset':0.,'calibration_missing_data':65534,'calibration_out_of_image':65535})
    cfg={'source':str(source),'output':str(tmp_path/'output'),'start':str(t0)+'Z','end':None,
         'cadence_seconds':300,'crop':[1,3,1,3],
         'encoding':{'dry_threshold_mm':.01,'step_mm':.02},
         'storage':{'chunk_frames':4,'shard_frames':24,'zstd_level':3},
         'samples':{'history':4,'forecast':18,'stride':4,'origin':str(t0)+'Z'}}
    out=build(cfg); manifest=json.loads((out/'manifest.json').read_text())
    output=zarr.open_group(out/'data.zarr',mode='r'); status=np.asarray(output['frame_status'][:])
    assert status[4]==1 and status[25]==3 and status[50]==0
    assert output['rain_code'][50,1,1]==255
    assert manifest['sample_counts']['accepted']==len(pl.read_parquet(out/'samples.parquet'))
    assert manifest['encoding_statistics']['source_missing_pixels']==1
    for split in ('train', 'validation'):
        loader = TimelineDataset(out, split)
        if len(loader):
            sample = loader[0]
            assert sample['history_rain_mm'].shape == (4, 1, 2, 2)
            assert sample['future_valid'].shape == (18, 1, 2, 2)
            assert sample['split'] == split
    with pytest.raises(FileExistsError): build(cfg)
    # Interruption after all shards: rebuild from markers and regenerate metadata/index.
    (out/'manifest.json').unlink()
    assert build(cfg)==out
    assert np.array_equal(zarr.open_group(out/'data.zarr',mode='r')['frame_status'][:],status)
    cfg['encoding']['step_mm']=.04
    (out/'manifest.json').unlink()
    with pytest.raises(ValueError,match='different source/config'): build(cfg)
