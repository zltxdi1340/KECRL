"""One derived CSV, 28 projected fields, one pass per process; no archives.

Reuses the existing scoped file/array/runtime denial contract and IO limits.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

from experiments import libero_task7_target_magnitude_matching_protocol as d
from experiments import libero_task7_residual_sign_transition_access as previous_access

base = previous_access.base
SCIENCE_SHA = 'bb7ddb50c714332ac6e7518e028207f3c0ae0330a34efffd13e7b245cb1ae775'


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    d.p.require(d.p.sha(config) == SCIENCE_SHA, 'frozen science changed')
    cfg = d.validate(config); bound = d.require_binding(config, binding)
    d.p.require(bound['CPU_environment'] == cfg['execution']['environment'] and bound['future_output'] == cfg['output']
                and bound['future_artifacts'] == cfg['artifacts'], 'execution scope differs')
    refs = bound['dependencies']; paths = [d.p.resolve(r['path']).resolve() for r in refs]
    required = {*d.BINDING_FILES.values(), Path(__file__).resolve(), Path(previous_access.__file__).resolve(),
                Path(base.__file__).resolve(), d.p.resolve(bound['execution_contract']['path'])}
    d.p.require(len(paths) == len(set(paths)) and required.issubset(paths) and bound['execution_contract'] in refs, 'dependency coverage differs')
    for ref in refs:
        path = d.p.resolve(ref['path'])
        d.p.require(path.suffix.lower() not in {'.npz','.npy','.pt','.pth','.ckpt','.h5','.hdf5'}, 'archive byte hash forbidden')
        d.p.require(d.p.sha(path) == ref['sha256'], 'bound dependency changed')
    return cfg, bound


def typed(columns):
    d.p.require(set(columns) == set(d.COLUMNS), 'unlisted projected fields')
    integers = {'task','seed','source_row','demo','action_step','target_sign'}
    booleans = {'tail20','reference_clip','candidate_clip','reference_error','candidate_error'}
    enums = {'arm': d.p.ARMS, 'candidate': d.previous.MODES, 'reference': d.previous.MODES,
             'reference_category': d.previous.CLASSES, 'candidate_category': d.previous.CLASSES,
             'reference_band': d.previous.BANDS, 'candidate_band': d.previous.BANDS, 'transition': d.TRANSITIONS,
             'comparison': [c+'-minus-'+r for c,r in d.COMPARISONS]}
    row = {}
    for key, text in columns.items():
        if key in integers:
            value = int(text); d.p.require(str(value) == text, 'noncanonical integer'); row[key] = value
        elif key in booleans:
            d.p.require(text in ('True','False'), 'noncanonical boolean'); row[key] = text == 'True'
        elif key in enums:
            d.p.require(text in enums[key], 'unlisted enum: '+key); row[key] = text
        else:
            value = float(text); d.p.require(math.isfinite(value), 'nonfinite scalar'); row[key] = value
    d.p.require(row['task'] == 7 and row['seed'] in d.p.SEEDS and 0 <= row['source_row'] < 4188 and 0 <= row['demo'] < 30
                and row['action_step'] >= 5 and row['target_sign'] in (-1,1)
                and [row['candidate'],row['reference']] in d.COMPARISONS
                and row['comparison'] == row['candidate']+'-minus-'+row['reference'], 'frame identity/role differs')
    return row


class Guard(previous_access.Guard):
    def __init__(self, inputs, outputs=(), limits=None):
        ref = inputs['paired_frames']; self.refs = {'paired_frames': {**ref,'path':str(Path(ref['path']).resolve())}}
        self.paths = {self.refs['paired_frames']['path']}; self.outputs = {str(Path(path).resolve()) for path in outputs}
        d.p.require(all(Path(p).suffix == '.csv' for p in self.paths | self.outputs) and not self.paths.intersection(self.outputs), 'CSV allowlists invalid')
        self.telemetry_path = str(Path('/proc/self/status').resolve()); self.limits = limits or base.Limits(seconds=180)
        self.saved = []; self.denied = []; self.attempted = []; self.completed = []; self.rows = {}; self.cells = {}; self.hash_checks = []
        self.hash_path = self.active_path = None; self.reader = csv.reader

    def hash_inputs(self): self.hash_one('paired_frames')

    def project(self, columns=None):
        if columns is not None and list(columns) != d.COLUMNS: return self.reject('projection_columns',columns)
        name = 'paired_frames'
        if self.attempted: return self.reject('repeated_CSV_pass',name)
        self.attempted.append(name); self.rows[name] = self.cells[name] = 0; self.hash_one(name)
        ref = self.refs[name]; self.active_path = ref['path']
        try: handle = Path(ref['path']).open(newline='')
        finally: self.active_path = None
        with handle:
            reader = self.reader(handle); header = next(reader,None)
            d.p.require(header == ref['columns'] and len(header) == len(set(header)), 'CSV header changed/duplicate')
            indices = [header.index(key) for key in d.COLUMNS]
            for raw in reader:
                self.rows[name] += 1
                d.p.require(self.rows[name] <= ref['rows'] and len(raw) == len(header), 'CSV row budget/width differs')
                row = typed(dict(zip(d.COLUMNS,(raw[i] for i in indices)))); self.cells[name] += len(d.COLUMNS)
                if self.rows[name] % 256 == 0: self.limits.check()
                yield row
        d.p.require(self.rows[name] == ref['rows'], 'CSV row coverage differs'); self.completed.append(name)

    def complete(self):
        b = d.budget()['per_stage']
        d.p.require(self.attempted == self.completed == ['paired_frames'] and not self.denied
                    and self.rows == {'paired_frames':b['input_CSV_lexical_rows']} and self.cells == {'paired_frames':b['projected_cells']}, 'CSV projection budget differs')
        self.hash_inputs(); self.limits.check()

    def record(self):
        return {'CSV_data_passes':{'attempted':len(self.attempted),'completed':len(self.completed)},
                'CSV_lexical_rows':self.rows.copy(),'CSV_projected_cells':self.cells.copy(),'projected_columns':d.COLUMNS,
                'byte_SHA_checks':self.hash_checks.copy(),'denied_operations':self.denied.copy(),
                'original_CSV_array_archive_checkpoint_reads_including_hashes':0,'research_training_forward_optimizer_environment_sealed_calls':0}
