"""Keep v1 access gates and explicitly budget each fit normalization/reuse."""
from __future__ import annotations
import numpy as np
from experiments import libero_task7_endpoint_gradient_access as old
from experiments import libero_task7_endpoint_gradient_protocol_v2 as d

base, runtime, load, Guard, Jobs = old.base, old.runtime, old.load, old.Guard, old.Jobs


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    return d.validate(config), d.require_binding(config, binding)


class RepresentationWork:
    def __init__(self, cfg):
        self.expected = {(m['seed'], m['arm'], f['fold']): len(f['fit_rows'])
                         for m in cfg['inputs']['models'] for f in cfg['inputs']['folds']}
        self.builds = {}; self.reuses = set(); self.active = None
        self.rows = self.coordinates = self.reuse_rows = 0

    def begin(self, key, n):
        d.p.require(self.active is None and key in self.expected and key not in self.builds
                    and n == self.expected[key], 'duplicate or invalid normalization work')
        self.active = key

    def built(self, key, z):
        d.p.require(self.active == key and z.shape == (self.expected[key], 256)
                    and z.dtype == np.float64 and np.isfinite(z).all(), 'normalization result differs')
        z.flags.writeable = False
        self.builds[key] = id(z); self.rows += len(z); self.coordinates += z.size; self.active = None

    def reuse(self, key, z):
        d.p.require(self.active is None and key in self.builds and key not in self.reuses
                    and id(z) == self.builds[key] and not z.flags.writeable, 'invalid paired representation reuse')
        self.reuses.add(key); self.reuse_rows += len(z)

    def complete(self):
        d.p.require(self.active is None and set(self.builds) == self.reuses == set(self.expected)
                    and self.rows == self.reuse_rows == 50256 and self.coordinates == 12865536,
                    'preprocessing budget differs')

    def record(self):
        return {'scaler_reconstructions': len(self.builds), 'fit_standardization_row_visits': self.rows,
                'fit_standardization_feature_coordinates': self.coordinates,
                'paired_reuses': len(self.reuses), 'paired_reuse_rows': self.reuse_rows,
                'head_admissions': len(self.builds) + len(self.reuses),
                'build_keys': [list(k) for k in sorted(self.builds)],
                'reuse_keys': [list(k) for k in sorted(self.reuses)]}
