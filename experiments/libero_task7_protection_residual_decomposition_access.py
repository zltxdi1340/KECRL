"""Single-CSV fixed seventeen-column projection contract; no archives, weights or new readouts.

Scoped research access checks, not an operating-system security sandbox.
Shared with independent audit only for IO, syntax, limits and provenance.
"""
from __future__ import annotations

import builtins
import csv
import io
import math
import os
from pathlib import Path
import sys
import zipfile

from experiments import libero_task7_protection_residual_decomposition_protocol as d
from experiments import libero_task7_component3_history_transition_access as base

SCIENCE_SHA = 'aee1484ace7c86fe33b9245d80f9ef2e9497208375d8bcb72d87f06885417299'


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    d.p.require(d.p.sha(config) == SCIENCE_SHA, 'frozen science changed')
    cfg = d.validate(config); bound = d.require_binding(config, binding)
    d.p.require(bound['CPU_environment'] == cfg['execution']['environment']
                and bound['future_output'] == cfg['output'] and bound['future_artifacts'] == cfg['artifacts'], 'execution scope differs')
    refs = bound['dependencies']; paths = [d.p.resolve(ref['path']).resolve() for ref in refs]
    required = {*d.BINDING_FILES.values(), Path(__file__).resolve(), Path(base.__file__).resolve(), d.p.resolve(bound['execution_contract']['path'])}
    d.p.require(len(paths) == len(set(paths)) and required.issubset(paths)
                and bound['execution_contract'] in refs, 'dependency coverage differs')
    for ref in refs:
        path = d.p.resolve(ref['path'])
        d.p.require(path.suffix.lower() in {'.py','.md','.json','.toml'}, 'archive hash forbidden')
        d.p.require(d.p.pin(path) == {'path': str(path), 'sha256': ref['sha256']}, 'bound dependency changed')
    return cfg, bound


def typed(columns):
    d.p.require(set(columns)==set(d.COLUMNS), 'unlisted projected fields')
    integers = {'task', 'seed', 'update', 'source_row', 'probe_fold', 'demo', 'action_step', 'target_direction', 'fold', 'n', 'material_n', 'material_sign_error_n', 'clip_n'}
    booleans = {'tail20', 'clip'}
    enums = {'arm': d.p.ARMS, 'mode': d.MODES, 'role': ['fit', 'probe'],
             'phase': ['all_source', 'all_common', 'tail20', 'complement'],
             'direction': ['all', 'negative', 'positive', 'neutral'],
             'component3_category': d.layout.CLASSES}
    row = {}
    for key, raw in columns.items():
        if key in integers:
            value = int(raw); d.p.require(str(value) == raw, 'noncanonical integer'); row[key] = value
        elif key in booleans:
            d.p.require(raw in ('True', 'False'), 'noncanonical boolean'); row[key] = raw == 'True'
        elif key in enums:
            d.p.require(raw in enums[key], 'unlisted enum: ' + key); row[key] = raw
        else:
            value = float(raw); d.p.require(math.isfinite(value), 'nonfinite scalar'); row[key] = value
    d.p.require(row['task']==7 and row['seed'] in d.p.SEEDS and row['update']==500
                and 0<=row['source_row']<4188 and 0<=row['demo']<30 and row['probe_fold'] in (0,1,2)
                and row['action_step']>=1 and row['target_direction'] in (-1,0,1),'source frame role differs')
    return row


class Guard:
    def __init__(self, inputs, outputs=(), limits=None):
        self.refs = {name: {**ref, 'path': str(Path(ref['path']).resolve())} for name, ref in {'frames':inputs['frames']}.items()}
        d.p.require(set(self.refs)=={'frames'} and len(self.refs)==1, 'one CSV identity required')
        self.paths = {r['path'] for r in self.refs.values()}; self.outputs = {str(Path(p).resolve()) for p in outputs}
        d.p.require(all(Path(path).suffix == '.csv' for path in self.paths), 'CSV paths required')
        d.p.require(all(Path(path).suffix == '.csv' for path in self.outputs), 'only output CSV reads permitted')
        self.telemetry_path = str(Path('/proc/self/status').resolve())
        d.p.require(not self.paths.intersection(self.outputs), 'input/output overlap')
        self.limits = limits or base.Limits(seconds=180)
        self.saved = []; self.denied = []; self.attempted = []; self.completed = []
        self.rows = {}; self.cells = {}; self.hash_checks = []
        self.hash_path = self.active_path = None; self.reader = csv.reader

    def patch(self, obj, name, value):
        self.saved.append((obj, name, getattr(obj, name))); setattr(obj, name, value)

    def reject(self, kind, detail=''):
        self.denied.append({'kind': kind, 'detail': str(detail)})
        raise PermissionError('forbidden residual CSV access: ' + kind + ': ' + str(detail))

    def hash_one(self, name):
        ref = self.refs[name]; self.hash_path = ref['path']
        try: d.p.require(d.p.sha(ref['path']) == ref['sha256'], 'input CSV SHA changed')
        finally: self.hash_path = None
        self.hash_checks.append(name)

    def hash_inputs(self):
        self.hash_one('frames')

    def project(self, name, columns=None):
        if name not in self.refs: return self.reject('unlisted_CSV', name)
        expected = d.COLUMNS
        if columns is not None and list(columns) != expected: return self.reject('projection_columns', columns)
        if name in self.attempted: return self.reject('repeated_CSV_pass', name)
        self.attempted.append(name); self.rows[name] = self.cells[name] = 0
        self.hash_one(name); ref = self.refs[name]; self.active_path = ref['path']
        try: handle = Path(ref['path']).open(newline='')
        finally: self.active_path = None
        with handle:
            reader = self.reader(handle); header = next(reader, None)
            d.p.require(header == ref['columns'] and len(header) == len(set(header)), 'CSV header changed/duplicate')
            positions = {key: header.index(key) for key in expected}
            for raw in reader:
                self.rows[name] += 1; d.p.require(len(raw) == len(header), 'CSV row width differs')
                projected = {key: raw[i] for key, i in positions.items()}; row = typed(projected)
                self.cells[name] += len(projected)
                if self.rows[name] % 256 == 0: self.limits.check()
                yield row
        d.p.require(self.rows[name] == ref['rows'], 'CSV row coverage differs')
        self.completed.append(name)

    def __enter__(self):
        import numpy as np
        native_open, native_io, native_path, native_import = builtins.open, io.open, Path.open, builtins.__import__
        native_reader, native_os_open, native_fileio = csv.reader, os.open, io.FileIO
        def check(path, mode):
            if isinstance(path, int): return self.reject('raw_file_descriptor', path)
            if not isinstance(path, (str, bytes, os.PathLike)): return self.reject('nonpath_file', type(path))
            value = str(Path(os.fsdecode(path)).resolve()); read = 'r' in mode or '+' in mode
            if value in self.paths:
                if any(c in mode for c in 'wax+'): return self.reject('source_write', value)
                if value not in (self.active_path, self.hash_path): return self.reject('raw_CSV_bypass', value)
            elif read and value not in self.outputs and value != self.telemetry_path: return self.reject('unlisted_read_or_archive_hash', value)
            elif not read and value not in self.outputs: return self.reject('unlisted_write', value)
        def opened(path, mode='r', *args, **kw): check(path, mode); return native_open(path, mode, *args, **kw)
        def io_opened(path, mode='r', *args, **kw): check(path, mode); return native_io(path, mode, *args, **kw)
        def path_opened(path, mode='r', *args, **kw): check(path, mode); return native_path(path, mode, *args, **kw)
        def os_opened(path, flags, *args, **kw): return self.reject('os_open_bypass', path)
        def fileio(path, mode='r', *args, **kw): check(path, mode); return native_fileio(path, mode, *args, **kw)
        def imported(name, *args, **kw):
            if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco', 'pickle', 'pandas', 'mmap'} or name.startswith(('src.environments', 'src.algorithms', 'src.skills')):
                return self.reject('runtime_import', name)
            return native_import(name, *args, **kw)
        def reader(handle, *args, **kw):
            name = getattr(handle, 'name', None)
            if isinstance(name, (str, Path)) and str(Path(name).resolve()) in self.paths: return self.reject('direct_CSV_reader', name)
            return native_reader(handle, *args, **kw)
        deny = lambda *args, **kw: self.reject('array_archive_or_alternate_reader')
        for obj, key, replacement in [(builtins, 'open', opened), (io, 'open', io_opened), (Path, 'open', path_opened),
                                     (io, 'FileIO', fileio), (os, 'open', os_opened), (builtins, '__import__', imported), (csv, 'reader', reader)]:
            self.patch(obj, key, replacement)
        for key in ('load', 'fromfile', 'loadtxt', 'genfromtxt', 'fromregex'):
            self.patch(np, key, deny)
        for obj, key in [(np.lib.npyio, 'load'), (np.lib.npyio.NpzFile, '__getitem__'), (np.lib.format, 'read_array'),
                         (np.memmap, '__new__'), (zipfile.ZipFile, 'open')]: self.patch(obj, key, deny)
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            for obj, key in [(torch, 'load'), (torch.autograd, 'backward'), (torch.autograd, 'grad'),
                             (torch.optim.Optimizer, '__init__'), (torch.nn.Module, '__call__')]: self.patch(obj, key, deny)
        if 'h5py' in sys.modules: self.patch(sys.modules['h5py'], 'File', deny)
        return self

    def __exit__(self, *args):
        for obj, key, original in reversed(self.saved): setattr(obj, key, original)

    def complete(self):
        expected=d.budget()['per_stage']
        d.p.require(self.attempted==self.completed==['frames'] and not self.denied
                    and self.rows=={'frames':expected['lexical_rows']}
                    and self.cells=={'frames':expected['projected_cells']},'CSV projection budget differs')
        self.hash_inputs();self.limits.check()

    def record(self):
        return {'CSV_data_passes':{'attempted':len(self.attempted),'completed':len(self.completed)},
                'CSV_lexical_rows':self.rows.copy(),'CSV_projected_cells':self.cells.copy(),
                'lexically_tokenized_cells':sum(self.rows.values())*35,'byte_SHA_checks':self.hash_checks.copy(),
                'projected_columns':d.COLUMNS.copy(),'unselected_columns_numeric_converted':0,
                'denied_operations':self.denied.copy(),'archive_array_head_checkpoint_reads_including_hashes':0,
                'training_forward_environment_or_sealed_calls':0}
