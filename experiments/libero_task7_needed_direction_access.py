"""Seven-cache access budget and fit/probe gate for needed-direction readout."""
from __future__ import annotations

from pathlib import Path
import sys

from experiments import libero_task7_needed_direction_protocol as d
from experiments import libero_task7_current_residual_correctability_access as inherited

base = inherited.base


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg = d.validate(config); bound = d.require_binding(config, binding)
    d.p.require('torch' not in sys.modules, 'NumPy-only fresh interpreter required')
    return cfg, bound


class Guard(inherited.Guard):
    """Reuse the pinned source-only mode; heads, raw/mmap, duplicates denied."""
    def __init__(self, inputs, outputs=(), limits=None):
        super().__init__(inputs, outputs, limits, heads=(), roles=None)
        d.p.require(d.MEMBERS == inherited.d.MEMBERS, 'reused member scope changed')


def load_sources(inputs):
    import numpy as np
    def load(ref, kind):
        with np.load(ref['path'], allow_pickle=False) as z:
            return {key: z[key] for key in d.MEMBERS[kind]}
    shared = load(inputs['shared'], 'shared')
    models = {(m['seed'], m['arm']): load(m['current_archive'], 'current') for m in inputs['models']}
    return shared, models


def correction_labels(y, p0):
    import numpy as np
    delta = y.astype(np.float64) - p0.astype(np.float64)
    return np.where(delta <= -.025, -1, np.where(delta >= .025, 1, 0)).astype(np.int8)


class Roles:
    def __init__(self, cfg):
        self.cfg = cfg; self.folds = d.fold_plan(cfg['inputs']['action_lengths'])
        self.expected = {(m['seed'], m['arm'], f) for m in cfg['inputs']['models'] for f in range(3)}
        self.banks = {}; self.probe_open = False; self.scored = set()
        self.counts = {'fit_banks': 0, 'fit_rows': 0, 'scaler_coordinates': 0,
                       'pair_distances_per_block': 0, 'coordinate_squared_differences': 0,
                       'neighbor_classifications': 0, 'reference_classifications': 0}

    def prepare(self, model, fold, feature, p0, target):
        import numpy as np
        key = model['seed'], model['arm'], fold
        d.p.require(key in self.expected and key not in self.banks and not self.probe_open, 'fit role or repeated bank')
        rows = np.asarray(self.folds[fold]['fit_rows'], np.int32)
        raw = np.column_stack((feature[rows].astype(np.float64), p0[rows].astype(np.float64)))
        mean = raw.mean(0); scale = raw.std(0, ddof=0); scale[scale <= 1e-6] = 1.
        labels = correction_labels(target[rows], p0[rows]); counts = [int((labels == c).sum()) for c in d.CLASSES]
        d.p.require(raw.shape == (len(rows), 257) and np.isfinite(raw).all() and
                    np.isfinite(mean).all() and np.isfinite(scale).all() and min(counts) >= 5, 'fit classes/scaler unsupported')
        majority = next(c for c in d.TIE_ORDER if counts[d.CLASSES.index(c)] == max(counts))
        bank = {'rows': rows, 'labels': labels, 'mean': mean, 'scale': scale,
                'values': (raw - mean) / scale, 'counts': counts, 'majority': majority}
        for v in bank.values():
            if isinstance(v, np.ndarray): v.flags.writeable = False
        self.banks[key] = bank; self.counts['fit_banks'] += 1; self.counts['fit_rows'] += len(rows)
        self.counts['scaler_coordinates'] += raw.size
        return bank

    def open_probe(self):
        d.p.require(not self.probe_open and set(self.banks) == self.expected and
                    self.counts['fit_banks'] == 18 and self.counts['fit_rows'] == 50256,
                    'all18 fit banks required before probe')
        self.probe_open = True

    def query(self, key, rows):
        d.p.require(self.probe_open and key in self.banks and key not in self.scored and
                    list(rows) == self.folds[key[2]]['probe_rows'], 'probe rows/gate/duplicate differ')
        self.scored.add(key)

    def add_distances(self, queries, bank):
        n = queries * bank
        self.counts['pair_distances_per_block'] += n
        self.counts['coordinate_squared_differences'] += n * 257
        self.counts['neighbor_classifications'] += queries * 3
        self.counts['reference_classifications'] += queries * 2
        b = self.cfg['budget']
        d.p.require(self.counts['pair_distances_per_block'] <= b['pair_distances_per_block_per_stage'] and
                    self.counts['neighbor_classifications'] <= 73224, 'distance/classification budget exceeded')

    def complete(self):
        b = self.cfg['budget']
        d.p.require(self.probe_open and self.scored == self.expected and self.counts == {
            'fit_banks': 18, 'fit_rows': 50256, 'scaler_coordinates': 12915792,
            'pair_distances_per_block': b['pair_distances_per_block_per_stage'],
            'coordinate_squared_differences': b['coordinate_squared_differences_per_stage'],
            'neighbor_classifications': 73224, 'reference_classifications': 48816}, 'computation/role coverage incomplete')


def check_tables(cfg, tables):
    d.p.require({k: v['rows'] for k, v in tables.items()} == cfg['budget']['tables'], 'output row budget differs')
    d.p.require(sum(Path(v['path']).stat().st_size for v in tables.values()) <= cfg['execution']['output_CSV_bytes_max'], 'output size budget exceeded')
