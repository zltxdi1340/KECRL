"""Two-table fixed projection guard; IO only is shared with the audit."""
from __future__ import annotations
import csv
import math
from pathlib import Path
import sys
from experiments import libero_task7_direction_harm_alignment_protocol as d
from experiments import libero_task7_protection_residual_decomposition_access as inherited

base = inherited.base


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg = d.validate(config); bound = d.require_binding(config, binding)
    d.p.require('torch' not in sys.modules, 'fresh NumPy-only process required')
    return cfg, bound


def typed(kind, raw):
    d.p.require(list(raw) == d.COLUMNS[kind], 'unlisted projection')
    integers = {'task', 'seed', 'fold', 'source_row', 'demo', 'action_step', 'truth', 'predicted', 'probe_fold'}
    row = {}
    for field, text in raw.items():
        if field in integers:
            value = int(text); d.p.require(str(value) == text, 'noncanonical integer'); row[field] = value
        elif field == 'tail20':
            d.p.require(text in ('True', 'False'), 'noncanonical boolean'); row[field] = text == 'True'
        elif field in ('arm', 'view'):
            d.p.require(text in (d.p.ARMS if field == 'arm' else d.VIEWS), 'unknown enum'); row[field] = text
        else:
            value = float(text); d.p.require(math.isfinite(value), 'nonfinite projected value'); row[field] = value
    d.p.require(row['task'] == 7 and row['seed'] in d.p.SEEDS and 0 <= row['source_row'] < 4188 and
        0 <= row['demo'] < 30 and row['action_step'] >= 5 and row.get('fold', row.get('probe_fold')) in range(3), 'common source role differs')
    if kind == 'direction': d.p.require(row['truth'] in (-1, 0, 1) and row['predicted'] in (-1, 0, 1), 'direction class differs')
    return row


class Guard(inherited.Guard):
    def __init__(self, inputs, outputs=(), limits=None):
        self.refs = {name: {**inputs[name], 'path': str(Path(inputs[name]['path']).resolve())} for name in d.COLUMNS}
        self.paths = {r['path'] for r in self.refs.values()}; self.outputs = {str(Path(p).resolve()) for p in outputs}
        d.p.require(len(self.paths) == 2 and not self.paths.intersection(self.outputs) and all(Path(p).suffix == '.csv' for p in self.paths | self.outputs), 'two disjoint CSV inputs and CSV outputs required')
        self.telemetry_path = str(Path('/proc/self/status').resolve()); self.limits = limits or base.Limits(180)
        self.saved = []; self.denied = []; self.attempted = []; self.completed = []
        self.rows = {}; self.cells = {}; self.hash_checks = []; self.lexical_cells = 0
        self.hash_path = self.active_path = None; self.reader = csv.reader

    def hash_inputs(self):
        for name in self.refs: self.hash_one(name)

    def project(self, name, columns=None):
        if name not in self.refs: return self.reject('unlisted_CSV', name)
        if columns is not None and list(columns) != d.COLUMNS[name]: return self.reject('projection_columns', columns)
        if name in self.attempted: return self.reject('repeated_CSV_pass', name)
        self.attempted.append(name); self.rows[name] = self.cells[name] = 0
        self.hash_one(name); ref = self.refs[name]; self.active_path = ref['path']
        try: handle = Path(ref['path']).open(newline='')
        finally: self.active_path = None
        with handle:
            reader = self.reader(handle); header = next(reader, None)
            d.p.require(header == ref['columns'] and len(header) == len(set(header)), 'source header/duplicate columns differ')
            positions = {key: header.index(key) for key in d.COLUMNS[name]}
            for raw in reader:
                d.p.require(len(raw) == len(header), 'source row width differs')
                self.rows[name] += 1; self.lexical_cells += len(raw)
                projected = {key: raw[i] for key, i in positions.items()}; self.cells[name] += len(projected)
                if self.rows[name] % 256 == 0: self.limits.check()
                yield typed(name, projected)
        d.p.require(self.rows[name] == ref['rows'], 'source row count differs'); self.completed.append(name)

    def complete(self):
        b = d.budget()
        d.p.require(self.attempted == self.completed == list(d.COLUMNS) and not self.denied and self.rows == b['rows_per_stage'] and
                    self.cells == b['projected_cells_per_stage'] and self.lexical_cells == b['lexical_cells_per_stage'], 'projection/pass budget differs')
        self.hash_inputs(); self.limits.check()

    def record(self):
        return {'CSV_data_passes': {'attempted': len(self.attempted), 'completed': len(self.completed)},
            'rows': self.rows.copy(), 'projected_cells': self.cells.copy(), 'lexical_cells': self.lexical_cells,
            'byte_SHA_checks': self.hash_checks.copy(), 'projected_columns': d.COLUMNS, 'denied_operations': self.denied.copy(),
            'unselected_columns_numeric_converted': 0, 'archive_head_checkpoint_reads_including_SHA': 0,
            'new_training_forward_environment_neighbor_or_sealed_calls': 0}
