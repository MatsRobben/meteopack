"""Simple NumPy reader for a completed timeline dataset; no ML dependency."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import polars as pl
import zarr


class TimelineDataset:
    """Lazy random-access samples for one train/validation split.

    Create one instance in each worker. Values are in mm, invalid pixels are
    zero with a separate mask. Metadata comes from the immutable manifest.
    """
    def __init__(self, path: str | Path, split: str):
        self.path = Path(path)
        self.manifest = json.loads((self.path/'manifest.json').read_text())
        if not self.manifest.get('completed'):
            raise ValueError('dataset is incomplete')
        if split not in ('train', 'validation'):
            raise ValueError('split must be train or validation')
        self.index = pl.read_parquet(self.path/'samples.parquet').filter(pl.col('split') == split)
        self.history = int(self.manifest['config']['samples']['history'])
        self.forecast = int(self.manifest['config']['samples']['forecast'])
        self.step = float(self.manifest['config']['encoding']['step_mm'])
        self._group = None

    def __len__(self):
        return self.index.height

    def __getitem__(self, idx):
        if self._group is None:
            self._group = zarr.open_group(self.path/'data.zarr', mode='r')
        row = self.index.row(idx, named=True)
        start = int(row['start_idx']); n = self.history + self.forecast
        codes = np.asarray(self._group['rain_code'][start:start+n])
        mask = codes != 255
        values = np.where(mask, codes, 0).astype('float32') * self.step
        return {'history_rain_mm': values[:self.history, None],
                'future_rain_mm': values[self.history:, None],
                'history_valid': mask[:self.history, None],
                'future_valid': mask[self.history:, None],
                'sample_id': row['sample_id'], 'issue_time': row['issue_time'],
                'split': row['split']}

    def __getstate__(self):
        state = self.__dict__.copy(); state['_group'] = None
        return state
