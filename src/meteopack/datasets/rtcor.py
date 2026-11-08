"""Build an indexed uint8 RTCOR timeline, one output shard at a time.

Run: python -m meteopack.datasets.rtcor --config configs/datasets/rtcor.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import zarr
from zarr.codecs import ZstdCodec

from .common import fingerprint, read_config, sample_index, utc_now, write_json


def encode_rain(raw: np.ndarray, *, missing: int, outside: int,
                source_scale: float, threshold: float, step: float,
                max_code: int = 254) -> tuple[np.ndarray, np.ndarray, dict]:
    """Threshold then round half up to uint8; return codes, pixel-missing and counts."""
    if source_scale <= 0 or step <= 0 or threshold < 0 or max_code != 254:
        raise ValueError('invalid encoding parameters')
    invalid = (raw == missing) | (raw == outside)
    # Float64 and a small tolerance make decimal ties deterministic for source integer codes.
    values = raw.astype(np.float64) * source_scale
    dry = (~invalid) & (values <= threshold + 1e-10)
    rounded = np.floor(values / step + 0.5 + 1e-10)
    clipped = (~invalid) & (rounded > max_code)
    codes = np.minimum(rounded, max_code).astype('uint8')
    codes[dry] = 0
    codes[invalid] = 255
    return codes, raw == missing, {
        'thresholded_wet_pixels': int(((raw > 0) & dry).sum()),
        'clipped_pixels': int(clipped.sum()),
        'source_missing_pixels': int((raw == missing).sum()),
        'outside_pixels': int((raw == outside).sum()),
    }


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'],
            cwd=Path(__file__).resolve().parents[3], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build(cfg: dict, config_path: Path | None = None) -> Path:
    source = Path(cfg['source']).expanduser().resolve()
    output = Path(cfg['output']).expanduser().resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('source and output must be separate directories')
    started = time.perf_counter()
    src = zarr.open_group(source, mode='r')
    rain = src['accumulation_mm']
    attrs = dict(rain.attrs)
    missing, outside = int(attrs['calibration_missing_data']), int(attrs['calibration_out_of_image'])
    scale, offset = float(attrs['scale_factor']), float(attrs.get('add_offset', 0))
    if offset != 0 or missing == outside:
        raise ValueError('unsupported source encoding')
    times = np.asarray(src['time'][:], dtype=np.int64).astype('datetime64[ns]')
    start = np.datetime64(str(cfg['start']).replace('Z', ''), 'ns')
    end_text = cfg.get('end')
    end = np.datetime64(str(end_text).replace('Z', ''), 'ns') if end_text else None
    lo = int(np.searchsorted(times, start)); hi = int(np.searchsorted(times, end)) if end is not None else len(times)
    if lo >= hi or (lo and times[lo] != start) or (lo == 0 and times[0] > start):
        raise ValueError('start must be an existing source timestamp and range must be nonempty')
    times = times[lo:hi]; n = len(times)
    source_missing = np.asarray(src['is_missing_frame'][lo:hi], dtype=bool)
    source_corrupt = np.asarray(src['is_corrupt_frame'][lo:hi], dtype=bool)
    status = np.where(source_missing, 1, np.where(source_corrupt, 2, 0)).astype('uint8')
    y0, y1, x0, x1 = map(int, cfg['crop'])
    if not (0 <= y0 < y1 <= rain.shape[1] and 0 <= x0 < x1 <= rain.shape[2]):
        raise ValueError('crop outside source grid')
    enc = cfg['encoding']; chunk = int(cfg['storage']['chunk_frames']); shard = int(cfg['storage']['shard_frames'])
    if chunk < 1 or shard < chunk or shard % chunk:
        raise ValueError('shard_frames must be a positive multiple of chunk_frames')
    manifest_path = source.parent / 'manifest.json'
    source_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None
    identity = {'config': cfg, 'source_manifest_sha256': source_hash, 'source_shape': list(rain.shape),
                'source_attrs': attrs, 'source_first_time_ns': int(times[0].astype('int64')),
                'source_last_time_ns': int(times[-1].astype('int64')),
                'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'common_sha256': hashlib.sha256(Path(__file__).with_name('common.py').read_bytes()).hexdigest()}
    fp = fingerprint(identity)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'build.json'
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['fingerprint'] != fp:
            raise ValueError('output exists with different source/config; choose a new output directory')
    else:
        if any(output.iterdir()):
            raise ValueError('output directory is not empty and has no matching build state')
        write_json(state_path, {'fingerprint': fp, 'created_at': utc_now(), 'identity': identity})
    if (output / 'manifest.json').exists():
        raise FileExistsError('dataset already completed; choose a new output directory')
    group = zarr.open_group(output / 'data.zarr', mode='a', zarr_format=3)
    shape = (n, y1-y0, x1-x0)
    if 'rain_code' not in group:
        arr = group.create_array('rain_code', shape=shape, dtype='uint8',
            chunks=(chunk,shape[1],shape[2]), shards=(shard,shape[1],shape[2]),
            fill_value=255, compressors=[ZstdCodec(level=int(cfg['storage']['zstd_level']))],
            dimension_names=('time','y','x'))
        arr.attrs.update({'units':'mm per 5 minutes', 'scale_factor':float(enc['step_mm']),
            'offset':0.0, 'out_of_image':255, 'rain_codes':'0..254',
            'threshold_mm_inclusive':float(enc['dry_threshold_mm']), 'rounding':'nearest, ties up',
            'clipped_at_mm':254*float(enc['step_mm'])})
    else:
        arr = group['rain_code']
        if tuple(arr.shape) != shape:
            raise ValueError('existing output shape mismatch')
    markers = output / '_shards'; markers.mkdir(exist_ok=True)
    counts = {'thresholded_wet_pixels':0, 'clipped_pixels':0, 'source_missing_pixels':0, 'outside_pixels':0}
    timings = {'source_read_seconds': 0.0, 'encoding_seconds': 0.0, 'output_write_seconds': 0.0}
    resumed_shards = 0
    for s, i in enumerate(range(0,n,shard)):
        stop=min(i+shard,n); marker=markers / f'{s:06d}.json'
        if marker.exists():
            info=json.loads(marker.read_text())
            if info['fingerprint'] != fp or info['start'] != i or info['stop'] != stop:
                raise ValueError('invalid shard completion marker')
            status[i:stop]=np.array(info['status'],dtype='uint8')
            resumed_shards += 1
        else:
            t = time.perf_counter()
            raw=np.asarray(rain[lo+i:lo+stop,y0:y1,x0:x1])
            read_s = time.perf_counter() - t
            t = time.perf_counter()
            codes, pixel_missing, stats=encode_rain(raw, missing=missing, outside=outside,
                source_scale=scale, threshold=float(enc['dry_threshold_mm']), step=float(enc['step_mm']))
            usable=status[i:stop] == 0
            # Flag any otherwise usable frame with missing pixels; no fabricated dry data.
            has_missing=pixel_missing.reshape(stop-i,-1).any(axis=1) & usable
            status[i:stop][has_missing]=3
            codes[~usable] = 255
            # Encoding statistics describe usable source frames only.
            if not usable.all():
                _, _, stats=encode_rain(raw[usable], missing=missing, outside=outside,
                    source_scale=scale, threshold=float(enc['dry_threshold_mm']), step=float(enc['step_mm']))
            encode_s = time.perf_counter() - t
            t = time.perf_counter()
            arr[i:stop]=codes
            write_s = time.perf_counter() - t
            info={'timings': {'source_read_seconds': read_s, 'encoding_seconds': encode_s,
                              'output_write_seconds': write_s}, 'fingerprint':fp,'start':i,'stop':stop,'status':status[i:stop].tolist(),'statistics':stats}
            write_json(marker,info)
        for key in counts: counts[key]+=int(info['statistics'][key])
        for key in timings: timings[key]+=float(info.get('timings', {}).get(key, 0.0))
        print(f'shard {s+1}/{(n+shard-1)//shard}: frames {i}:{stop}',flush=True)
    for name, data, dims in [('time', times.astype('int64'), ('time',)), ('frame_status',status,('time',))]:
        if name in group: del group[name]
        a=group.create_array(name, data=data, chunks=(min(n,65536),), dimension_names=dims)
        if name == 'time': a.attrs.update({'units':'nanoseconds since 1970-01-01T00:00:00Z','timezone':'UTC'})
        else: a.attrs.update({'flag_values':[0,1,2,3], 'flag_meanings':'usable missing_frame corrupt_frame missing_pixels'})
    index_started = time.perf_counter()
    settings=cfg['samples']
    index, index_counts=sample_index(times,status,history=int(settings['history']),
        forecast=int(settings['forecast']),stride=int(settings['stride']),
        origin=str(settings['origin']),cadence_seconds=int(cfg['cadence_seconds']))
    import polars as pl
    index=index.with_columns((pl.col('source_start_idx')+lo).alias('source_start_idx'))
    temp=output / 'samples.parquet.tmp'; index.write_parquet(temp); os.replace(temp, output / 'samples.parquet')
    timings['index_seconds'] = time.perf_counter() - index_started
    timings['invocation_seconds'] = time.perf_counter() - started
    timings['resumed_shards'] = resumed_shards
    timings['shards'] = (n + shard - 1) // shard
    manifest={'schema_version':1,'created_at':utc_now(),'fingerprint':fp,'config':cfg,
        'source':str(source),'source_manifest_sha256':source_hash,'source_frame_start_index':lo,
        'source_frame_stop_index_exclusive':hi,'source_attributes':attrs,'code_revision':_git_revision(),
        'config_path':str(config_path) if config_path else None,'shape':list(shape),
        'start_time_utc':str(times[0])+'Z','last_time_utc':str(times[-1])+'Z',
        'frame_status_counts':{str(i):int((status==i).sum()) for i in range(4)},
        'encoding_statistics':counts,'sample_counts':index_counts, 'timings':timings,
        'split':'first UTC calendar day of each month is validation; remaining days train',
        'paired_3d':False,'completed':True}
    write_json(output / 'manifest.json',manifest)
    return output


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args()
    print(f'Completed: {build(read_config(args.config),args.config)}')


if __name__ == '__main__':
    main()
