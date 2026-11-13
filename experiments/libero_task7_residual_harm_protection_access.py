"""Scoped archive/member, auxiliary-gradient and whole-demo role guards."""
from __future__ import annotations

import builtins
import ctypes
import gc
from contextlib import contextmanager
import importlib.metadata
import io
from pathlib import Path
import platform
import sys
import zipfile

from experiments import libero_task7_residual_harm_protection_protocol as d
from experiments import libero_task7_component3_history_transition_access as base

SCIENCE_SHA = 'aabfd4a198a0e59e96394ca788dc1bf0ab9c4612afb8e7d57eac2880531955b5'


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    d.p.require(d.p.sha(config) == SCIENCE_SHA, 'scientific configuration changed')
    cfg = d.validate(config); bound = d.require_binding(config, binding)
    d.p.require(bound['CPU_environment'] == cfg['execution']['environment']
                and bound['future_output'] == cfg['output'] and bound['future_artifacts'] == cfg['artifacts'], 'execution environment/paths differ')
    refs = bound['dependencies']; paths = [d.p.resolve(ref['path']).resolve() for ref in refs]
    required = {*d.BINDING_FILES.values(), Path(base.__file__).resolve()}
    d.p.require(len(paths) == len(set(paths)) and required.issubset(paths), 'execution dependency coverage differs')
    d.p.require(bound['execution_contract'] in refs, 'execution contract unbound')
    for ref in refs: d.p.require(d.p.pin(d.p.resolve(ref['path'])) == ref, 'bound dependency changed')
    return cfg, bound


def runtime(cfg, *, training=False):
    import numpy as np
    import os
    versions = {'python': platform.python_version(), 'numpy': np.__version__, 'torch': importlib.metadata.version('torch')}
    d.p.require(versions == cfg['execution']['runtime'], 'frozen runtime differs')
    d.p.require(all(os.environ.get(key) == value for key, value in cfg['execution']['environment'].items()), 'CPU environment differs')
    if training:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
        d.p.require(torch.get_num_threads() == 1 and torch.are_deterministic_algorithms_enabled(), 'Torch determinism differs')
    return {**versions, 'device': 'CPU', 'pid': os.getpid(), 'environment': dict(cfg['execution']['environment']),
            'environment_seed': None, 'initial_state_indices': []}


def load(ref, kind):
    import numpy as np
    fields = d.HEAD_MEMBERS if kind == 'head' else d.HISTORICAL_MEMBERS if kind == 'historical' else d.MEMBERS[kind]
    with np.load(ref['path'], allow_pickle=False) as archive: return {key: archive[key] for key in fields}


def head_key(model, fold, mode): return model['seed'], model['arm'], fold, mode


def head_path(directory, model, fold, mode):
    d.p.require(mode in tuple(d.CASES) and fold in range(3), 'unknown head identity')
    return Path(directory) / f"head-task7-seed{model['seed']}-{model['arm']}-fold{fold}-{mode}.npz"


def release_temporaries():
    """Return unused CPU heap pages; VmHWM remains the true process peak."""
    gc.collect()
    libc = ctypes.CDLL(None)
    if hasattr(libc, 'malloc_trim'): libc.malloc_trim(0)


class Roles:
    """Workers receive only fit slices; gradients can reach registered heads only."""
    def __init__(self, cfg):
        self.cfg = cfg; self.plans = cfg['inputs']['folds']; self.expected = {
            head_key(m, f, mode) for m in cfg['inputs']['models'] for f in range(3) for mode in tuple(d.CASES)}
        d.p.require(self.plans == d.training.folds(cfg['inputs']['action_lengths']) and len(self.expected) == 36, 'canonical whole-demo roles required')
        self.active = None; self.parameter_ids = set(); self.registered = set(); self.done = {}
        self.probe_open = False; self.batch_pending = False; self.forward_pending = False; self.step_pending = False
        self.optimizer_heads = set()
        self.counts = {'optimizer_constructions': 0, 'optimizer_steps': 0, 'backward_calls': 0,
                       'train_forward_batches': 0, 'train_row_visits': 0, 'fit_standardization_rows': 0,
                       'scaler_fits': 0, 'scaler_fit_rows': 0, 'full_standardization_rows': 0,
                       'snapshot_batches': 0, 'snapshot_rows': 0}
        self.scalers, self.full_transforms, self.snapshots = set(), set(), set(); self.trace = []

    def fit_transform(self, model, fold, feature):
        import numpy as np
        key = model['seed'], model['arm'], fold
        d.p.require(not self.probe_open and key not in self.scalers, 'repeated or post-probe scaler')
        rows = np.asarray(self.plans[fold]['fit_rows'], np.int32)
        values = feature[rows].astype(np.float64); mean = values.mean(0)
        # NumPy population std arithmetic with its full-size variance buffer reused.
        values -= mean; np.square(values,out=values); scale=values.mean(0); np.sqrt(scale,out=scale)
        scale[scale <= 1e-6] = 1.
        values[:] = feature[rows]; values -= mean; values /= scale; x = values.astype(np.float32)
        d.p.require(np.isfinite(x).all() and np.isfinite(mean).all() and np.isfinite(scale).all(), 'nonfinite fit transform')
        self.scalers.add(key); self.counts['scaler_fits'] += 1; self.counts['scaler_fit_rows'] += len(rows)
        self.counts['fit_standardization_rows'] += len(rows)
        return x, mean, scale

    @contextmanager
    def head(self, key, parameters):
        d.p.require(not self.probe_open and self.active is None and key in self.expected and key not in self.registered, 'head role/identity differs')
        params = list(parameters); n = sum(p.numel() for p in params)
        d.p.require(n == 257 and [p.numel() for p in params] == [256, 1] and all(p.device.type == 'cpu' and str(p.dtype) == 'torch.float32' and p.requires_grad for p in params), 'auxiliary parameters/device differ')
        d.p.require(all(p.detach().count_nonzero().item() == 0 for p in params), 'zero residual initialization required')
        self.registered.add(key); self.active = key; self.parameter_ids = {id(p) for p in params}; self.done[key] = 0
        self.trace.append({'event': 'fit_begin', 'head': list(key), 'completed_heads': sum(n == 500 for n in self.done.values())})
        try:
            yield
            d.p.require(self.done[key] == 500 and not self.batch_pending and not self.forward_pending and not self.step_pending, 'head incomplete/no retry')
        finally: self.active = None; self.parameter_ids = set()

    def batch(self, absolute_rows):
        import numpy as np
        d.p.require(self.active is not None and not self.probe_open and not self.batch_pending and not self.forward_pending and not self.step_pending, 'fit batch outside head')
        plan = self.plans[self.active[2]]; fit = np.asarray(plan['fit_rows'], np.int32)
        d.p.require(absolute_rows.shape == (64,) and absolute_rows.dtype == np.int32
                    and np.isin(absolute_rows, fit).all() and self.done[self.active] < 500, 'probe or invalid rows in gradients')
        self.batch_pending = True

    def forward(self):
        d.p.require(self.active is not None and self.batch_pending and not self.forward_pending, 'unregistered fit forward')
        self.batch_pending = False; self.forward_pending = True
        self.counts['train_forward_batches'] += 1; self.counts['train_row_visits'] += 64

    def backward(self):
        d.p.require(self.active is not None and self.forward_pending and not self.probe_open, 'backward outside fit head')
        self.forward_pending = False; self.step_pending = True; self.counts['backward_calls'] += 1

    def optimizer(self, groups):
        d.p.require(self.active is not None and {id(p) for group in groups for p in group['params']} == self.parameter_ids, 'original/unregistered optimizer parameters')

    def step(self, groups):
        self.optimizer(groups)
        d.p.require(not self.batch_pending and not self.forward_pending and self.step_pending and self.done[self.active] < 500, 'step budget/order differs')
        self.step_pending = False; self.done[self.active] += 1; self.counts['optimizer_steps'] += 1

    def open_probe(self):
        d.p.require(not self.probe_open and self.active is None and set(self.done) == self.expected
                    and all(n == 500 for n in self.done.values()) and self.counts['backward_calls'] == 18000
                    and self.counts['train_forward_batches'] == 18000 and self.counts['optimizer_steps'] == 18000,
                    'all36 heads must finish500 before probe')
        self.probe_open = True; self.trace.append({'event': 'probe_open', 'heads': 36, 'optimizer_steps': 18000, 'backward_calls': 18000})

    def full_transform(self, model, fold, feature, mean, scale):
        import numpy as np
        key = model['seed'], model['arm'], fold
        d.p.require(self.probe_open and key in self.scalers and key not in self.full_transforms, 'probe transform before gate or repeated')
        self.full_transforms.add(key); self.counts['full_standardization_rows'] += 4188
        values = feature.astype(np.float64); values -= mean; values /= scale
        return values.astype(np.float32)

    def snapshot(self, key, update):
        identity = (*key, update)
        d.p.require(self.probe_open and key in self.expected and update in d.SNAPSHOTS and identity not in self.snapshots, 'snapshot before gate/repeated')
        self.snapshots.add(identity); self.counts['snapshot_batches'] += 1; self.counts['snapshot_rows'] += 4188

    def complete(self):
        expected = {'optimizer_constructions': 36, 'optimizer_steps': 18000, 'backward_calls': 18000,
                    'train_forward_batches': 18000, 'train_row_visits': 1152000,
                    'fit_standardization_rows': 50256, 'scaler_fits': 18, 'scaler_fit_rows': 50256,
                    'full_standardization_rows': 75384, 'snapshot_batches': 144, 'snapshot_rows': 603072}
        d.p.require(self.probe_open and self.counts == expected, 'role/computation budget differs')

    def record(self):
        return {'counts': self.counts.copy(), 'heads': [{'key': list(k), 'updates': v} for k, v in self.done.items()],
                'trace': self.trace, 'probe_open': self.probe_open}


class Guard(base.ReadGuard):
    def __init__(self, inputs, outputs=(), limits=None, *, heads=(), historical=(), roles=None):
        pairs = [(inputs['shared'], 'shared')] + [(m['current_archive'], 'current') for m in inputs['models']]
        d.p.require(len({ref['path'] for ref, _ in pairs}) == len(pairs) == 7, 'seven source archives required')
        self.source_paths = {str(Path(ref['path']).resolve()) for ref, _ in pairs}
        pairs += [(ref, 'head') for ref in heads] + [(ref, 'historical') for ref in historical]
        self.refs = {str(Path(ref['path']).resolve()): {**ref, 'kind': kind} for ref, kind in pairs}
        d.p.require(len(pairs) == len(self.refs) and len(heads) in (0, 36) and len(historical) in (0, 18) and (not historical or len(heads)==36), 'head/source identities differ')
        self.hash_refs = dict(self.refs)
        if roles is not None:
            self.hash_refs.update({str(Path(row['archive']['path']).resolve()): {**row['archive'], 'kind': 'historical_hash_only'} for row in inputs.get('historical_uniform_heads', [])})
        self.roles = roles; self.expected_files = 7 + len(heads) + len(historical); self.expected_members = 17 + len(heads) * 27 + len(historical) * 14
        self.read_outputs = {str(Path(path).resolve()) for path in outputs}
        d.p.require(not self.read_outputs.intersection(self.hash_refs), 'source/output allowlist overlaps')
        self.telemetry_path = str(Path('/proc/self/status').resolve()); self.limits = limits or base.Limits(180)
        self.attempts, self.completed, self.member_attempts, self.member_completed, self.denied = [], [], [], [], []
        self.registry, self.zip_registry, self.saved = {}, {}, []
        self.load_depth = self.member_depth = self.hash_depth = 0; self.active_member = None; self.hash_checks = []

    def hash_one(self, path, ref):
        self.hash_depth += 1
        try: d.p.require(d.p.sha(path) == ref['sha256'], 'pinned archive bytes changed')
        finally: self.hash_depth -= 1
        self.hash_checks.append(path)

    def hash_inputs(self):
        for path, ref in self.hash_refs.items(): self.hash_one(path, ref)

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
            self.attempts.append(name); self.limits.check(); self.hash_one(name, self.refs[name]); self.load_depth += 1
            try: archive = load0(path, allow_pickle=False)
            finally: self.load_depth -= 1
            d.p.require(isinstance(archive, np.lib.npyio.NpzFile), 'NPZ required')
            self.registry[id(archive)] = name; self.zip_registry[id(archive.zip)] = name; self.completed.append(name); return archive
        def get(archive, key):
            name = self.registry.get(id(archive)); kind = self.refs[name]['kind'] if name in self.refs else None
            allowed = d.HEAD_MEMBERS if kind == 'head' else d.HISTORICAL_MEMBERS if kind == 'historical' else d.MEMBERS.get(kind, [])
            if key not in allowed: return self.reject('member', key)
            pair = name, key
            if pair in self.member_attempts: return self.reject('repeated_member', key)
            self.member_attempts.append(pair); self.member_depth += 1; self.active_member = pair
            try: value = get0(archive, key)
            finally: self.member_depth -= 1; self.active_member = None
            d.p.require(isinstance(value, np.ndarray) and value.dtype.kind in 'biuf' and not value.dtype.hasobject, 'numeric array required')
            value.flags.writeable = False; self.member_completed.append(pair); self.limits.check(); return value
        def zip_open(archive, name, mode='r', *args, **kwargs):
            source = self.zip_registry.get(id(archive)); key = name.filename if isinstance(name, zipfile.ZipInfo) else name
            if source is not None and mode == 'r' and (not self.member_depth or self.active_member != (source, str(key).removesuffix('.npy'))):
                return self.reject('zip_bypass', key)
            return zip0(archive, name, mode, *args, **kwargs)
        def check(path, mode):
            if not isinstance(path, (str, Path)): return
            name = str(Path(path).resolve())
            if name in self.hash_refs and any(flag in mode for flag in ('w', 'a', 'x', '+')): return self.reject('source_write', name)
            if 'r' not in mode and '+' not in mode: return
            if name in self.hash_refs:
                if not (self.hash_depth or (self.load_depth and name in self.refs)): return self.reject('raw_archive_bypass', name)
            elif name not in self.read_outputs and name != self.telemetry_path: return self.reject('unlisted_read', name)
        def opened(path, mode='r', *args, **kwargs): check(path, mode); return open0(path, mode, *args, **kwargs)
        def io_opened(path, mode='r', *args, **kwargs): check(path, mode); return io0(path, mode, *args, **kwargs)
        def path_opened(path, mode='r', *args, **kwargs): check(path, mode); return path0(path, mode, *args, **kwargs)
        def imported(name, *args, **kwargs):
            forbidden = {'h5py', 'libero', 'robosuite', 'mujoco', 'pickle', 'src'}
            if self.roles is None: forbidden.add('torch')
            if name.split('.')[0] in forbidden: return self.reject('runtime_import', name)
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
            torch = sys.modules['torch']; self.patch(torch, 'load', lambda *a, **k: self.reject('checkpoint', ''))
            if self.roles is None:
                for obj, key in [(torch.autograd, 'backward'), (torch.autograd, 'grad'), (torch.optim.Optimizer, '__init__')]:
                    self.patch(obj, key, lambda *a, **k: self.reject('gradient_optimizer', ''))
            else:
                roles = self.roles; backward0 = torch.autograd.backward; adam0 = torch.optim.Adam.__init__
                # Pinned 2.4.1 eager functions: omit compiler graph-break wrappers,
                # retain the original Adam constructor, state and numerical kernel.
                step0 = torch.optim.Adam.step
                while hasattr(step0, '__wrapped__'): step0 = step0.__wrapped__
                for method in ('add_param_group', 'zero_grad'):
                    fn = getattr(torch.optim.Optimizer, method)
                    d.p.require(hasattr(fn, '__wrapped__'), 'pinned eager optimizer wrapper missing')
                    self.patch(torch.optim.Optimizer, method, fn.__wrapped__)
                def backward(*args, **kwargs): roles.backward(); return backward0(*args, **kwargs)
                def adam(optimizer, params, *args, **kwargs):
                    params = list(params)
                    d.p.require(roles.active is not None and {id(p) for p in params} == roles.parameter_ids, 'unregistered auxiliary optimizer')
                    d.p.require(roles.active not in roles.optimizer_heads, 'repeated optimizer construction')
                    result = adam0(optimizer, params, *args, **kwargs); roles.optimizer(optimizer.param_groups)
                    roles.optimizer_heads.add(roles.active)
                    roles.counts['optimizer_constructions'] += 1; return result
                def step(optimizer, *args, **kwargs):
                    roles.step(optimizer.param_groups)
                    with torch.no_grad(): return step0(optimizer, *args, **kwargs)
                self.patch(torch.autograd, 'backward', backward); self.patch(torch.autograd, 'grad', lambda *a, **k: self.reject('unregistered_grad', ''))
                self.patch(torch.optim.Adam, '__init__', adam); self.patch(torch.optim.Adam, 'step', step)
        if 'h5py' in sys.modules: self.patch(sys.modules['h5py'], 'File', lambda *a, **k: self.reject('HDF5', ''))
        return self

    def complete(self):
        expected = {(path, key) for path, ref in self.refs.items() for key in (d.HEAD_MEMBERS if ref['kind'] == 'head' else d.HISTORICAL_MEMBERS if ref['kind'] == 'historical' else d.MEMBERS[ref['kind']])}
        d.p.require(len(self.attempts) == len(self.completed) == self.expected_files and set(self.completed) == set(self.refs)
                    and len(self.member_attempts) == len(self.member_completed) == len(expected) == self.expected_members
                    and set(self.member_completed) == expected and not self.denied, 'archive/member budget differs')
        self.hash_inputs(); self.limits.check()

    def record(self):
        return {**super().record(), 'byte_SHA_checks': self.hash_checks.copy(), 'source_files': sorted(self.source_paths),
                'source_loads': sum(path in self.source_paths for path in self.completed),
                'head_loads': sum(self.refs[path]['kind']=='head' for path in self.completed),
                'historical_head_loads': sum(self.refs[path]['kind']=='historical' for path in self.completed),
                'historical_byte_hash_only_paths': [path for path,ref in self.hash_refs.items() if ref['kind']=='historical_hash_only'], 'input_CSV_passes': 0}
