"""Seven-archive/eleven-member and two-CSV projection contract for task7.

This scoped contract prevents accidental research reads; it is not an OS sandbox.
Numerical implementations live in separate analysis and audit modules.
"""
from __future__ import annotations

import builtins
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

from experiments import libero_task7_policy_error_focus_alignment_protocol as d
from experiments import libero_task7_component3_history_transition_access as base

SCIENCE_SHA = '39fd9bbe7c578ddd80191602762a68ceeebc15e1731f6117ce531c5453690f30'


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    d.p.require(d.p.sha(config) == SCIENCE_SHA, 'scientific config changed')
    cfg = d.validate(config); bound = d.require_binding(config, binding)
    d.p.require(bound['before_research_payload'] is True
                and bound['CPU_environment'] == cfg['execution']['environment']
                and bound['future_output'] == cfg['output'] and bound['future_artifacts'] == cfg['artifacts'],
                'execution contract differs')
    refs = bound['dependencies']
    paths = [d.p.resolve(ref['path']).resolve() for ref in refs]
    required = {Path(__file__).resolve(), *d.BINDING_FILES.values(), Path(base.__file__).resolve()}
    contract = bound['execution_contract']
    d.p.require(d.p.pin(d.p.resolve(contract['path'])) == contract and contract in refs,
                'execution contract changed or unbound')
    d.p.require(len(paths) == len(set(paths)) and required.issubset(paths), 'dependency coverage differs')
    for ref in refs:
        d.p.require(d.p.pin(d.p.resolve(ref['path'])) == ref, 'bound dependency changed')
    return cfg, bound


def load(ref, kind):
    import numpy as np
    with np.load(ref['path'], allow_pickle=False) as archive:
        return {key: archive[key] for key in d.MEMBERS[kind]}


def typed_projection(name, row):
    """Only the authorized twelve lexical values enter typed computation."""
    columns = d.CSV_COLUMNS[name]
    d.p.require(set(row) == set(columns), 'projection columns differ')
    values = {}
    for key in columns:
        raw = row[key]
        if key == 'tail20':
            d.p.require(raw in ('True', 'False'), 'noncanonical tail boolean')
            values[key] = raw == 'True'
        elif key == 'arm':
            d.p.require(raw in d.p.ARMS, 'unknown arm tag'); values[key] = raw
        elif key in ('transition', 'original_transition'):
            d.p.require(raw in d.ORIGINAL_GROUPS, 'unknown transition tag'); values[key] = raw
        else:
            value = int(raw)
            d.p.require(str(value) == raw, 'noncanonical integer tag'); values[key] = value
    d.p.require(values['task'] == 7 and values['seed'] in d.p.SEEDS
                and 0 <= values['demo'] < 30 and values['action_step'] >= 5
                and 0 <= values['source_row'] < 4188 and values['target_sign'] in (-1, 1), 'tag role/ID differs')
    signs = ('current_prediction', 'history5_prediction') if name == 'transition_tags' else ('CC_prediction', 'HC_prediction', 'HH_prediction')
    d.p.require(all(values[key] in (-1, 1) for key in signs), 'binary readout tag required')
    if name == 'transition_tags': d.p.require(values['policy_sign'] in (-1, 0, 1), 'Policy sign tag invalid')
    return values


class Guard(base.ReadGuard):
    def __init__(self, inputs, outputs=(), limits=None):
        pairs = [(inputs['shared'], 'shared')] + [(row['prediction_archive'], 'prediction') for row in inputs['models']]
        self.refs = {str(Path(ref['path']).resolve()): {**ref, 'kind': kind} for ref, kind in pairs}
        d.p.require(len(pairs) == len(self.refs) == 7, 'seven unique archives required')
        self.tags = {key: {**ref, 'path': str(Path(ref['path']).resolve())} for key, ref in inputs['tags'].items()}
        d.p.require(set(self.tags) == set(d.CSV_COLUMNS) and len({r['path'] for r in self.tags.values()}) == 2,
                    'two unique tag CSVs required')
        self.tag_paths = {ref['path'] for ref in self.tags.values()}
        self.read_outputs = {str(Path(path).resolve()) for path in outputs}
        d.p.require(not self.tag_paths.intersection(self.refs)
                    and not self.read_outputs.intersection(set(self.refs) | self.tag_paths), 'allowlists overlap')
        self.telemetry_path = str(Path('/proc/self/status').resolve()); self.limits = limits or base.Limits()
        self.attempts, self.completed, self.member_attempts, self.member_completed, self.denied = [], [], [], [], []
        self.saved, self.registry, self.zip_registry = [], {}, {}
        self.load_depth = self.hash_depth = self.member_depth = 0; self.active_member = None
        self.csv_attempts, self.csv_completed = [], []; self.csv_rows = {}; self.csv_cells = {}; self.active_tag = None
        self.hash_checks = []; self.reader = csv.reader

    def hash_one(self, path, ref):
        self.hash_depth += 1
        try: d.p.require(d.p.sha(path) == ref['sha256'], 'pinned input bytes changed')
        finally: self.hash_depth -= 1
        self.hash_checks.append(path)

    def hash_inputs(self):
        for path, ref in self.refs.items(): self.hash_one(path, ref)
        for ref in self.tags.values(): self.hash_one(ref['path'], ref)

    def project(self, name, columns=None):
        d.p.require(name in self.tags, 'unlisted tag CSV')
        if columns is not None and list(columns) != d.CSV_COLUMNS[name]: return self.reject('CSV_projection', columns)
        if name in self.csv_attempts: return self.reject('repeated_CSV_pass', name)
        self.csv_attempts.append(name); self.csv_rows[name] = self.csv_cells[name] = 0
        ref = self.tags[name]; self.hash_one(ref['path'], ref); self.active_tag = ref['path']
        try:
            handle = Path(ref['path']).open(newline='')
        finally: self.active_tag = None
        try:
            with handle:
                reader = self.reader(handle); header = next(reader, None)
                d.p.require(header == ref['columns'] and len(set(header)) == len(header), 'tag CSV header differs/duplicates')
                positions = [header.index(key) for key in d.CSV_COLUMNS[name]]
                for row in reader:
                    self.csv_rows[name] += 1
                    d.p.require(len(row) == len(header), 'tag CSV row width differs')
                    projected = dict(zip(d.CSV_COLUMNS[name], (row[i] for i in positions)))
                    self.csv_cells[name] += len(projected)
                    # Never expose the remaining lexical fields to callers.
                    result = typed_projection(name, projected)
                    if self.csv_rows[name] % 256 == 0: self.limits.check()
                    yield result
            d.p.require(self.csv_rows[name] == ref['rows'] == 21588, 'tag CSV coverage differs')
            self.csv_completed.append(name)
        finally: self.active_tag = None

    def __enter__(self):
        import numpy as np
        load0, get0 = np.load, np.lib.npyio.NpzFile.__getitem__
        zip0, read0, from0 = zipfile.ZipFile.open, np.lib.format.read_array, np.fromfile
        open0, io0, path0, import0 = builtins.open, io.open, Path.open, builtins.__import__
        reader0 = csv.reader

        def load(path, *args, **kwargs):
            if not isinstance(path, (str, Path)): return self.reject('nonpath', type(path))
            name = str(Path(path).resolve())
            if name not in self.refs or name in self.attempts: return self.reject('unlisted_or_repeated_load', name)
            if args or kwargs != {'allow_pickle': False}: return self.reject('pickle_mmap_or_arguments', name)
            self.attempts.append(name); self.limits.check(); self.hash_one(name, self.refs[name]); self.load_depth += 1
            try: archive = load0(path, allow_pickle=False)
            finally: self.load_depth -= 1
            d.p.require(isinstance(archive, np.lib.npyio.NpzFile), 'NPZ required')
            self.registry[id(archive)] = name; self.zip_registry[id(archive.zip)] = name; self.completed.append(name)
            return archive

        def get(archive, key):
            name = self.registry.get(id(archive))
            if name is None or key not in d.MEMBERS[self.refs[name]['kind']]: return self.reject('member', key)
            identity = name, key
            if identity in self.member_attempts: return self.reject('repeated_member', key)
            self.member_attempts.append(identity); self.member_depth += 1; self.active_member = identity
            try: value = get0(archive, key)
            finally: self.member_depth -= 1; self.active_member = None
            d.p.require(isinstance(value, np.ndarray) and not value.dtype.hasobject, 'object/nonarray denied')
            value.flags.writeable = False; self.member_completed.append(identity); self.limits.check(); return value

        def zip_open(archive, name, mode='r', *args, **kwargs):
            source = self.zip_registry.get(id(archive)); key = name.filename if isinstance(name, zipfile.ZipInfo) else name
            if source is not None and mode == 'r' and (not self.member_depth or self.active_member != (source, str(key).removesuffix('.npy'))):
                return self.reject('zip_bypass', key)
            return zip0(archive, name, mode, *args, **kwargs)

        def check(path, mode):
            if not isinstance(path, (str, Path)): return
            value = str(Path(path).resolve()); source = value in self.refs or value in self.tag_paths
            if source and any(flag in mode for flag in ('w', 'a', 'x', '+')): return self.reject('source_write', value)
            if 'r' not in mode and '+' not in mode: return
            if value in self.refs:
                if not (self.load_depth or self.hash_depth): return self.reject('raw_archive_bypass', value)
            elif value in self.tag_paths:
                if not (self.hash_depth or self.active_tag == value): return self.reject('raw_CSV_bypass', value)
            elif value not in self.read_outputs and value != self.telemetry_path: return self.reject('unlisted_read', value)

        def opened(path, mode='r', *args, **kwargs): check(path, mode); return open0(path, mode, *args, **kwargs)
        def io_opened(path, mode='r', *args, **kwargs): check(path, mode); return io0(path, mode, *args, **kwargs)
        def path_opened(path, mode='r', *args, **kwargs): check(path, mode); return path0(path, mode, *args, **kwargs)
        def imported(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco', 'pickle'}: return self.reject('runtime_import', name)
            return import0(name, *args, **kwargs)
        def fromfile(*args, **kwargs):
            if not self.member_depth: return self.reject('fromfile', '')
            return from0(*args, **kwargs)
        def read_array(*args, **kwargs):
            if not self.member_depth: return self.reject('read_array', '')
            return read0(*args, **kwargs)
        def reader(handle, *args, **kwargs):
            if isinstance(getattr(handle, 'name', None), (str, Path)) and str(Path(handle.name).resolve()) in self.tag_paths:
                return self.reject('direct_CSV_reader', handle.name)
            return reader0(handle, *args, **kwargs)
        for obj, key, value in [(np, 'load', load), (np.lib.npyio, 'load', load), (np.lib.npyio.NpzFile, '__getitem__', get),
             (zipfile.ZipFile, 'open', zip_open), (np.lib.format, 'read_array', read_array), (np, 'fromfile', fromfile),
             (np.memmap, '__new__', lambda *a, **k: self.reject('mmap', '')), (builtins, 'open', opened),
             (io, 'open', io_opened), (Path, 'open', path_opened), (builtins, '__import__', imported), (csv, 'reader', reader)]:
            self.patch(obj, key, value)
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            for obj, key in [(torch, 'load'), (torch.autograd, 'backward'), (torch.autograd, 'grad'), (torch.optim.Optimizer, '__init__')]:
                self.patch(obj, key, lambda *a, **k: self.reject('model_gradient_optimizer', ''))
        if 'h5py' in sys.modules: self.patch(sys.modules['h5py'], 'File', lambda *a, **k: self.reject('HDF5', ''))
        return self

    def complete(self):
        expected = {(path, member) for path, ref in self.refs.items() for member in d.MEMBERS[ref['kind']]}
        d.p.require(len(self.attempts) == len(self.completed) == 7 and set(self.completed) == set(self.refs)
                    and len(self.member_attempts) == len(self.member_completed) == len(expected) == 11
                    and set(self.member_completed) == expected and not self.denied, 'file/member budget differs')
        d.p.require(len(self.csv_attempts) == len(self.csv_completed) == 2 and set(self.csv_completed) == set(self.tags)
                    and sum(self.csv_rows.values()) == 43176 and sum(self.csv_cells.values()) == 518112, 'CSV projection budget differs')
        self.hash_inputs(); self.limits.check()

    def record(self):
        return {**super().record(), 'tag_CSV_passes': {'attempted': len(self.csv_attempts), 'completed': len(self.csv_completed)},
                'tag_CSV_rows': self.csv_rows.copy(), 'projected_CSV_cells': self.csv_cells.copy(),
                'byte_SHA_checks': list(self.hash_checks), 'projected_columns': d.CSV_COLUMNS}
