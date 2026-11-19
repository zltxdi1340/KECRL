"""Source-restricted causal windows plus the official cached obs/ee_pos field."""
from __future__ import annotations

import numpy as np
import torch

from .libero_causal_t1_dataset import CausalT1LiberoSequenceDataset


class CachedEEFLiberoSequenceDataset(CausalT1LiberoSequenceDataset):
    def _index_windows(self):
        windows = super()._index_windows()
        self.eef_schema = []
        data = self._open()['data']
        for index in self.access.demonstration_indices:
            demo = data[f'demo_{index}']
            eef = demo['obs']['ee_pos']  # Missing field is an error; no fallback.
            length = int(demo['actions'].shape[0])
            if eef.shape != (length,3) or eef.dtype not in (np.dtype('float32'),np.dtype('float64')):
                raise ValueError('official cached ee_pos must be floating [actions,3]')
            self.eef_schema.append(dict(demonstration_index=index,shape=list(eef.shape),dtype=str(eef.dtype)))
        return windows

    def __getitem__(self, index):
        row = super().__getitem__(index)
        window = self.windows[index]
        demo = self._open()['data'][f'demo_{window.demonstration_index}']
        xyz = np.asarray(self._observation_slice(demo,'ee_pos',window),dtype=np.float32)
        if xyz.shape != (window.valid_length,3) or not np.isfinite(xyz).all():
            raise ValueError('finite causal cached XYZ required')
        row['eef_pos'] = torch.from_numpy(self._padded(xyz,self.sequence_length,window.valid_length))
        return row

    @property
    def access_record(self):
        return {**super().access_record, 'eef_schema':self.eef_schema,
                'eef_field':'obs/ee_pos', 'eef_sampling':'cached_robot0_eef_pos',
                'eef_coordinates':'world_xyz_metres_float32_no_normalization',
                'private_object_state_access':False}
