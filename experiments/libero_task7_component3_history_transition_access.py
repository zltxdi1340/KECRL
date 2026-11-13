"""Pinned file/member access and execution bookkeeping for cached task7 analysis."""
from __future__ import annotations

import builtins
import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import platform
import resource
import sys
import time
import zipfile

from experiments import libero_task7_component3_history_transition_protocol as d

SCIENCE_SHA = '3731985fc975ce88ace5dad55bbf5c5f53304463a6c7d5ff6b6375a5aadf4962'


def now():
    return datetime.now(timezone.utc).isoformat()


def write_new(path, record):
    with Path(path).open('x') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def preflight(config_path=d.DEFAULT_CONFIG, binding_path=d.DEFAULT_BINDING):
    d.p.require(d.p.sha(config_path) == SCIENCE_SHA, 'frozen task7 scientific config changed')
    cfg = d.validate(config_path)
    binding = d.require_execution_binding(config_path, binding_path)
    d.p.require(binding.get('before_research_payload') is True
                and binding.get('CPU_environment') == cfg['execution']['environment']
                and binding.get('future_output') == cfg['output']
                and binding.get('future_artifacts') == cfg['artifacts'], 'execution contract differs')
    for ref in binding['code_references'] + [binding['execution_contract']]:
        d.p.require(d.p.sha(d.p.resolve(ref['path'])) == ref['sha256'], 'bound code/contract changed')
    parent = d.p.load_pin(cfg['parents']['result'])
    models = {(r['task'], r['seed'], r['arm']): r for r in parent['primary_results'] if r['task'] == 7}
    return cfg, binding, models


def runtime(cfg):
    import numpy as np
    import os
    d.p.require(platform.python_version() == cfg['execution']['runtime']['python']
                and np.__version__ == cfg['execution']['runtime']['numpy'], 'frozen CPU runtime differs')
    d.p.require(all(os.environ.get(key) == value for key, value in cfg['execution']['environment'].items()),
                'CPU environment differs; CUDA must be hidden and BLAS single threaded')
    return {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'CPU',
            'environment': {key: os.environ.get(key) for key in cfg['execution']['environment']},
            'pid': os.getpid(), 'environment_seed': None, 'initial_state_indices': []}


class Limits:
    def __init__(self, seconds=120, memory_bytes=536870912):
        self.started = time.monotonic()
        self.seconds, self.memory_bytes = seconds, memory_bytes
        self.peak_RSS_bytes = 0

    def check(self):
        # Linux rusage can retain the pre-exec address-space peak.  VmHWM is
        # attached to the current mm and measures this interpreter's peak.
        status = Path('/proc/self/status').read_text()
        memory = {line.split(':')[0]: int(line.split()[1]) * 1024
                  for line in status.splitlines() if line.startswith(('VmHWM:', 'VmRSS:'))}
        d.p.require(set(memory) == {'VmHWM', 'VmRSS'}, 'current address-space RSS telemetry missing')
        self.peak_RSS_bytes = max(self.peak_RSS_bytes, memory['VmHWM'], memory['VmRSS'])
        d.p.require(time.monotonic() - self.started <= self.seconds, 'active time budget exceeded')
        d.p.require(self.peak_RSS_bytes <= self.memory_bytes, '512MiB working memory/RSS ceiling exceeded')

    def record(self):
        self.check()
        return {'elapsed_seconds': time.monotonic() - self.started, 'peak_RSS_bytes': self.peak_RSS_bytes,
                'active_seconds_max': self.seconds, 'memory_bytes_max': self.memory_bytes,
                'memory_monitor': '/proc/self/status VmHWM/VmRSS current address space',
                'rusage_peak_RSS_bytes_including_pre_exec': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


class ReadGuard:
    """One np.load and one decode per permitted member, with no alternate readers.

    SHA reads use an explicit byte-only scope.  Lazy NpzFile indexing and its
    underlying ZipFile.open are both guarded, including the archive.f interface.
    This is a scoped research access contract, not an OS security sandbox.
    """
    def __init__(self, inputs, *, read_outputs=(), limits=None):
        shared = {**inputs['shared'], 'kind': 'shared'}
        refs = [shared] + [{**row['archive'], 'kind': 'probe'} for row in inputs['models']]
        self.refs = {str(Path(row['path']).resolve()): row for row in refs}
        d.p.require(len(refs) == len(self.refs) == 7, 'exactly seven unique source archives required')
        self.read_outputs = {str(Path(path).resolve()) for path in read_outputs}
        self.telemetry_path = str(Path('/proc/self/status').resolve())
        d.p.require(not self.read_outputs.intersection(self.refs), 'output/read allowlist overlaps source')
        self.limits = limits or Limits()
        self.attempts, self.completed, self.denied, self.member_attempts, self.member_completed = [], [], [], [], []
        self.saved = []; self.registry = {}; self.zip_registry = {}
        self.load_depth = self.hash_depth = self.member_depth = 0
        self.active_member = None

    def patch(self, obj, name, value):
        self.saved.append((obj, name, getattr(obj, name))); setattr(obj, name, value)

    def reject(self, kind, detail):
        self.denied.append({'kind': kind, 'detail': str(detail)})
        raise PermissionError('forbidden task7 cached access: ' + kind + ': ' + str(detail))

    def hash_inputs(self):
        for path, ref in self.refs.items():
            self.hash_depth += 1
            try:
                d.p.require(d.p.sha(path) == ref['sha256'], 'pinned source archive bytes changed')
            finally:
                self.hash_depth -= 1

    def __enter__(self):
        import numpy as np
        native_load, native_fromfile = np.load, np.fromfile
        native_get = np.lib.npyio.NpzFile.__getitem__
        native_zip_open, native_read_array = zipfile.ZipFile.open, np.lib.format.read_array
        native_open, native_io_open, native_path_open = builtins.open, io.open, Path.open
        native_import = builtins.__import__

        def load(path, *args, **kwargs):
            if not isinstance(path, (str, Path)): return self.reject('np.load_nonpath', repr(type(path)))
            name = str(Path(path).resolve())
            if name not in self.refs or name in self.attempts: return self.reject('unlisted_or_repeated_np_load', name)
            if args or kwargs != {'allow_pickle': False}: return self.reject('pickle_mmap_or_extra_arguments', name)
            self.attempts.append(name); self.limits.check(); self.hash_depth += 1
            try: d.p.require(d.p.sha(name) == self.refs[name]['sha256'], 'source archive hash differs')
            finally: self.hash_depth -= 1
            self.load_depth += 1
            try: value = native_load(path, allow_pickle=False)
            finally: self.load_depth -= 1
            d.p.require(isinstance(value, np.lib.npyio.NpzFile), 'only NPZ archives permitted')
            self.registry[id(value)] = name; self.zip_registry[id(value.zip)] = name
            self.completed.append(name)
            return value

        def get(archive, key):
            name = self.registry.get(id(archive))
            if name is None: return self.reject('unregistered_archive', key)
            if key not in d.MEMBERS[self.refs[name]['kind']]: return self.reject('forbidden_member', key)
            identity = (name, key)
            if identity in self.member_attempts: return self.reject('repeated_member_decode', key)
            self.member_attempts.append(identity); self.active_member = identity; self.member_depth += 1
            try: value = native_get(archive, key)
            finally:
                self.member_depth -= 1; self.active_member = None
            d.p.require(isinstance(value, np.ndarray) and not value.dtype.hasobject, 'non-numeric/object array denied')
            value.flags.writeable = False; self.member_completed.append(identity); self.limits.check()
            return value

        def zip_open(archive, name, mode='r', *args, **kwargs):
            source = self.zip_registry.get(id(archive))
            if source is not None and mode == 'r':
                value = name.filename if isinstance(name, zipfile.ZipInfo) else name
                if not self.member_depth or self.active_member != (source, str(value).removesuffix('.npy')):
                    return self.reject('zip_member_bypass', value)
            return native_zip_open(archive, name, mode, *args, **kwargs)

        def fromfile(*args, **kwargs):
            if not self.member_depth: return self.reject('direct_numpy_fromfile', '')
            return native_fromfile(*args, **kwargs)

        def read_array(*args, **kwargs):
            if not self.member_depth: return self.reject('direct_read_array', '')
            return native_read_array(*args, **kwargs)

        def check(path, mode):
            if not isinstance(path, (str, Path)): return
            if 'r' not in mode and '+' not in mode: return
            value = str(Path(path).resolve())
            if value in self.refs:
                if not (self.load_depth or self.hash_depth): return self.reject('raw_archive_bypass', value)
            elif value not in self.read_outputs and value != self.telemetry_path:
                return self.reject('unlisted_file_read', value)

        def opened(path, mode='r', *args, **kwargs):
            check(path, mode); return native_open(path, mode, *args, **kwargs)

        def io_opened(path, mode='r', *args, **kwargs):
            check(path, mode); return native_io_open(path, mode, *args, **kwargs)

        def path_opened(path, mode='r', *args, **kwargs):
            check(path, mode); return native_path_open(path, mode, *args, **kwargs)

        def imported(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'h5py', 'libero', 'mujoco', 'robosuite', 'pickle'}:
                return self.reject('model_raw_data_or_environment_import', name)
            return native_import(name, *args, **kwargs)

        self.patch(np, 'load', load); self.patch(np.lib.npyio, 'load', load)
        self.patch(np.lib.npyio.NpzFile, '__getitem__', get)
        self.patch(zipfile.ZipFile, 'open', zip_open); self.patch(np.lib.format, 'read_array', read_array)
        self.patch(np, 'fromfile', fromfile)
        self.patch(np.memmap, '__new__', lambda *a, **k: self.reject('mmap', 'numpy.memmap'))
        self.patch(builtins, 'open', opened); self.patch(io, 'open', io_opened); self.patch(Path, 'open', path_opened)
        self.patch(builtins, '__import__', imported)
        # Unit-test processes may already contain these optional modules.
        if 'torch' in sys.modules:
            torch = sys.modules['torch']
            self.patch(torch, 'load', lambda *a, **k: self.reject('checkpoint', 'torch.load'))
            self.patch(torch.autograd, 'backward', lambda *a, **k: self.reject('backward', 'torch.autograd'))
            self.patch(torch.autograd, 'grad', lambda *a, **k: self.reject('gradient', 'torch.autograd'))
            self.patch(torch.optim.Optimizer, '__init__', lambda *a, **k: self.reject('optimizer', 'constructor'))
        if 'h5py' in sys.modules:
            self.patch(sys.modules['h5py'], 'File', lambda *a, **k: self.reject('HDF5', 'File'))
        return self

    def __exit__(self, *args):
        for obj, name, original in reversed(self.saved): setattr(obj, name, original)

    def complete(self):
        expected = {(path, member) for path, ref in self.refs.items() for member in d.MEMBERS[ref['kind']]}
        d.p.require(len(self.attempts) == len(self.completed) == 7 and set(self.completed) == set(self.refs)
                    and len(self.member_attempts) == len(self.member_completed) == len(expected) == 75
                    and set(self.member_completed) == expected and not self.denied,
                    'file/member attempted/completed budgets differ or denied operation occurred')
        self.hash_inputs(); self.limits.check()

    def record(self):
        return {'array_file_loads': {'attempted': len(self.attempts), 'completed': len(self.completed)},
                'array_member_decodes': {'attempted': len(self.member_attempts), 'completed': len(self.member_completed)},
                'files': self.completed,
                'members': [{'path': path, 'member': member} for path, member in self.member_completed],
                'denied_operations': self.denied}


def load_members(ref, kind):
    import numpy as np
    with np.load(ref['path'], allow_pickle=False) as archive:
        return {key: archive[key] for key in d.MEMBERS[kind]}


class CsvSink:
    def __init__(self, path, limits):
        self.path, self.limits = Path(path), limits
        self.handle = self.path.open('x', newline=''); self.writer = None; self.rows = 0

    def add(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row)); self.writer.writeheader()
        d.p.require(set(row) == set(self.writer.fieldnames), 'CSV field schema changed')
        self.writer.writerow(row); self.rows += 1
        if self.rows % 128 == 0: self.limits.check()

    def close(self):
        self.handle.close()

    def pin(self):
        d.p.require(self.handle.closed and self.writer is not None, 'CSV must be complete before pin')
        return {**d.p.pin(self.path), 'rows': self.rows, 'columns': self.writer.fieldnames}
