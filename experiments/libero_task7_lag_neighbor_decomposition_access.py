"""Thirteen-file/member guard and bound preflight; reuses pinned IO/resource helpers."""
from __future__ import annotations
import builtins
import io
import json
from pathlib import Path
import sys
import zipfile
from experiments import libero_task7_lag_neighbor_decomposition_protocol as d
from experiments import libero_task7_component3_history_transition_access as base

SCIENCE_SHA = 'ae5e70c20b0835067f06468ee1d076554043c04c8152a72433ce61c7ae107a58'


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    d.p.require(d.p.sha(config) == SCIENCE_SHA, 'scientific config changed')
    cfg = d.validate(config); bound = d.require_binding(config, binding)
    d.p.require(bound['before_research_payload'] is True and bound['CPU_environment'] == cfg['execution']['environment'], 'execution environment differs')
    readout = d.p.load_pin(d.PARENTS['readout'])
    parents = {(r['seed'], r['arm']): r for r in readout['primary_results'] if r['task'] == 7}
    return cfg, bound, parents


class Guard(base.ReadGuard):
    def __init__(self, inputs, outputs, limits):
        refs = [(inputs['shared'], 'shared')]
        refs += [(model[key], kind) for model in inputs['models'] for key, kind in [('archive', 'probe'), ('feature', 'feature')]]
        self.refs = {str(Path(ref['path']).resolve()): {**ref, 'kind': kind} for ref, kind in refs}
        d.p.require(len(self.refs) == len(refs) == 13, '13 unique source archives required')
        self.read_outputs = {str(Path(path).resolve()) for path in outputs}
        self.telemetry_path = str(Path('/proc/self/status').resolve()); self.limits = limits
        self.attempts, self.completed, self.denied = [], [], []
        self.member_attempts, self.member_completed, self.saved = [], [], []
        self.registry, self.zip_registry = {}, {}
        self.load_depth = self.hash_depth = self.member_depth = 0; self.active_member = None

    def __enter__(self):
        import numpy as np
        load0, get0 = np.load, np.lib.npyio.NpzFile.__getitem__
        zip0, read0, from0 = zipfile.ZipFile.open, np.lib.format.read_array, np.fromfile
        open0, io0, path0, import0 = builtins.open, io.open, Path.open, builtins.__import__

        def load(path, *args, **kwargs):
            if not isinstance(path, (str, Path)): return self.reject('nonpath', type(path))
            name = str(Path(path).resolve())
            if name not in self.refs or name in self.attempts: return self.reject('unlisted_or_repeated_load', name)
            if args or kwargs != {'allow_pickle': False}: return self.reject('pickle_mmap_or_arguments', name)
            self.attempts.append(name); self.limits.check(); self.hash_depth += 1
            try: d.p.require(d.p.sha(name) == self.refs[name]['sha256'], 'source bytes changed')
            finally: self.hash_depth -= 1
            self.load_depth += 1
            try: archive = load0(path, allow_pickle=False)
            finally: self.load_depth -= 1
            d.p.require(isinstance(archive, np.lib.npyio.NpzFile), 'NPZ required')
            self.registry[id(archive)] = name; self.zip_registry[id(archive.zip)] = name
            self.completed.append(name); return archive

        def get(archive, key):
            name = self.registry.get(id(archive))
            if name is None or key not in d.MEMBERS[self.refs[name]['kind']]: return self.reject('member', key)
            identity = name, key
            if identity in self.member_attempts: return self.reject('repeated_member', key)
            self.member_attempts.append(identity); self.member_depth += 1; self.active_member = identity
            try: value = get0(archive, key)
            finally: self.member_depth -= 1; self.active_member = None
            d.p.require(isinstance(value, np.ndarray) and not value.dtype.hasobject, 'object/nonarray denied')
            value.flags.writeable = False; self.member_completed.append(identity); return value

        def zip_open(archive, name, mode='r', *args, **kwargs):
            source = self.zip_registry.get(id(archive)); key = name.filename if isinstance(name, zipfile.ZipInfo) else name
            if source is not None and mode == 'r' and (not self.member_depth or self.active_member != (source, str(key).removesuffix('.npy'))):
                return self.reject('zip_bypass', key)
            return zip0(archive, name, mode, *args, **kwargs)

        def check(path, mode):
            if not isinstance(path, (str, Path)): return
            value = str(Path(path).resolve())
            if value in self.refs and any(flag in mode for flag in ('w', 'a', 'x', '+')): return self.reject('source_write', value)
            if 'r' not in mode and '+' not in mode: return
            if value in self.refs:
                if not (self.load_depth or self.hash_depth): return self.reject('raw_bytes_bypass', value)
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
        for obj, key, value in [(np, 'load', load), (np.lib.npyio, 'load', load), (np.lib.npyio.NpzFile, '__getitem__', get),
             (zipfile.ZipFile, 'open', zip_open), (np.lib.format, 'read_array', read_array), (np, 'fromfile', fromfile),
             (np.memmap, '__new__', lambda *a, **k: self.reject('mmap', '')), (builtins, 'open', opened),
             (io, 'open', io_opened), (Path, 'open', path_opened), (builtins, '__import__', imported)]: self.patch(obj, key, value)
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            for obj, key in [(torch, 'load'), (torch.autograd, 'backward'), (torch.autograd, 'grad'), (torch.optim.Optimizer, '__init__')]:
                self.patch(obj, key, lambda *a, **k: self.reject('model_gradient_optimizer', ''))
        if 'h5py' in sys.modules: self.patch(sys.modules['h5py'], 'File', lambda *a, **k: self.reject('HDF5', ''))
        return self

    def complete(self):
        expected = {(path, member) for path, ref in self.refs.items() for member in d.MEMBERS[ref['kind']]}
        d.p.require(len(self.attempts) == len(self.completed) == 13 and set(self.completed) == set(self.refs)
                    and len(self.member_attempts) == len(self.member_completed) == len(expected) == 81
                    and set(self.member_completed) == expected and not self.denied, 'file/member access budget differs')
        self.hash_inputs(); self.limits.check()


def load(ref, kind):
    import numpy as np
    with np.load(ref['path'], allow_pickle=False) as archive: return {key: archive[key] for key in d.MEMBERS[kind]}
