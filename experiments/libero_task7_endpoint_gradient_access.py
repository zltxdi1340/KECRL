"""Pinned eight-member heads, fit-only jobs and diagnostic-only autograd."""
from __future__ import annotations
import builtins
import csv
import importlib.metadata
import io
import os
from pathlib import Path
import sys
import zipfile
from experiments import libero_task7_endpoint_gradient_protocol as d
from experiments import libero_task7_component3_history_transition_access as base


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    cfg = d.validate(config); return cfg, d.require_binding(config, binding)


def runtime(cfg, audit=False):
    value = base.runtime(cfg); value['torch'] = importlib.metadata.version('torch')
    d.p.require({k: value[k] for k in ('python', 'numpy', 'torch')} == cfg['execution']['runtime'], 'runtime differs')
    if audit:
        import torch
        torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.use_deterministic_algorithms(True)
    else: d.p.require('torch' not in sys.modules, 'primary must be NumPy only')
    return value


def load(ref, kind):
    import numpy as np
    fields = d.HEAD_MEMBERS if kind == 'head' else d.SOURCE_MEMBERS[kind]
    with np.load(ref['path'], allow_pickle=False) as archive: return {k: archive[k] for k in fields}


class Jobs:
    def __init__(self, cfg, audit=False):
        self.expected = {(h['seed'], h['arm'], h['fold'], h['mode'], point) for h in cfg['inputs']['heads'] for point in d.POINTS}
        self.done = set(); self.active = None; self.audit = audit; self.params = (); self.n = None
        self.counts = {'point_jobs': 0, 'fit_row_visits': 0, 'linear_forms': 0, 'gradient_vectors': 0, 'autograd_calls': 0}
    def begin(self, key, n, params=()):
        d.p.require(self.active is None and key in self.expected and key not in self.done and n in (2782,2817,2777), 'job identity/fit count differs')
        self.active = key; self.params = tuple(params); self.n = n; self.forms = self.grads = 0
        self.before = [p.detach().numpy().tobytes() for p in self.params]
        if self.audit: d.p.require(len(params) == 2 and [p.numel() for p in params] == [256,1] and all(str(p.dtype) == 'torch.float64' and p.requires_grad and p.device.type == 'cpu' for p in params), 'registered auxiliary parameters differ')
    def linear(self, n):
        d.p.require(self.active is not None and n == self.n and self.forms < 2, 'linear form outside fit job')
        self.forms += 1; self.counts['linear_forms'] += 1
    def gradient(self, params=()):
        d.p.require(self.active is not None and self.grads < 3 and self.forms == 1, 'gradient outside fitted point')
        if self.audit: d.p.require(tuple(id(p) for p in params) == tuple(id(p) for p in self.params), 'unregistered gradient parameters')
        self.grads += 1; self.counts['gradient_vectors'] += 1
        if self.audit: self.counts['autograd_calls'] += 1
    def end(self):
        d.p.require(self.forms == 2 and self.grads == 3 and self.before == [p.detach().numpy().tobytes() for p in self.params], 'incomplete job or parameter mutation')
        self.done.add(self.active); self.counts['point_jobs'] += 1; self.counts['fit_row_visits'] += self.n
        self.active = None; self.params = (); self.n = None
    def complete(self):
        d.p.require(self.done == self.expected and self.active is None and self.counts == {'point_jobs':72,'fit_row_visits':201024,'linear_forms':144,'gradient_vectors':216,'autograd_calls':216 if self.audit else 0}, 'job computation budget differs')


class Guard(base.ReadGuard):
    def __init__(self, cfg, outputs=(), limits=None, jobs=None):
        pairs = [(cfg['inputs']['shared'], 'shared')] + [(m['current_archive'], 'current') for m in cfg['inputs']['models']] + [(h['archive'],'head') for h in cfg['inputs']['heads']]
        self.refs = {str(Path(ref['path']).resolve()): {**ref, 'kind': kind} for ref, kind in pairs}
        d.p.require(len(self.refs) == 43, '43 archives required')
        self.read_outputs = {str(Path(p).resolve()) for p in outputs}; self.telemetry_path = str(Path('/proc/self/status').resolve())
        self.limits = limits or base.Limits(180); self.jobs = jobs; self.saved = []; self.denied = []
        self.attempts = []; self.completed = []; self.member_attempts = []; self.member_completed = []; self.hash_checks = []
        self.registry = {}; self.zip_registry = {}; self.load_depth = self.member_depth = self.hash_depth = 0; self.active_member = None
    def reject(self, kind, detail=''):
        return super().reject(kind, detail)
    def hash_one(self, name):
        self.hash_depth += 1
        try: d.p.require(d.p.sha(name) == self.refs[name]['sha256'], 'archive bytes changed')
        finally: self.hash_depth -= 1
        self.hash_checks.append(name)
    def __enter__(self):
        import numpy as np
        load0, get0, zip0 = np.load, np.lib.npyio.NpzFile.__getitem__, zipfile.ZipFile.open
        open0, io0, path0, import0 = builtins.open, io.open, Path.open, builtins.__import__; file0=io.FileIO
        read0, from0 = np.lib.format.read_array, np.fromfile
        def loaded(path, *args, **kw):
            if not isinstance(path,(str,Path)): return self.reject('nonpath')
            name = str(Path(path).resolve())
            if name not in self.refs or name in self.attempts: return self.reject('unlisted_or_repeated_archive',name)
            if args or kw != {'allow_pickle':False}: return self.reject('pickle_mmap_arguments')
            self.attempts.append(name); self.hash_one(name); self.load_depth += 1
            try: z = load0(path,allow_pickle=False)
            finally: self.load_depth -= 1
            d.p.require(isinstance(z,np.lib.npyio.NpzFile),'NPZ required'); self.registry[id(z)] = name; self.zip_registry[id(z.zip)] = name; self.completed.append(name); self.limits.check(); return z
        def get(z, key):
            name = self.registry.get(id(z)); kind = self.refs[name]['kind'] if name in self.refs else None
            fields = d.HEAD_MEMBERS if kind == 'head' else d.SOURCE_MEMBERS.get(kind,[])
            if key not in fields or (name,key) in self.member_attempts: return self.reject('unlisted_or_repeated_member',key)
            self.member_attempts.append((name,key)); self.member_depth += 1; self.active_member = name,key
            try: value = get0(z,key)
            finally: self.member_depth -= 1; self.active_member = None
            d.p.require(isinstance(value,np.ndarray) and value.dtype.kind in 'biuf' and not value.dtype.hasobject,'numeric array required')
            value.flags.writeable = False; self.member_completed.append((name,key)); self.limits.check(); return value
        def zip_open(z, key, mode='r', *args, **kw):
            name = self.zip_registry.get(id(z)); field = key.filename if isinstance(key,zipfile.ZipInfo) else key
            if name is not None and (not self.member_depth or self.active_member != (name,str(field).removesuffix('.npy'))): return self.reject('zip_bypass',field)
            return zip0(z,key,mode,*args,**kw)
        def check(path, mode):
            if not isinstance(path,(str,Path)): return self.reject('raw_descriptor')
            name = str(Path(path).resolve()); read = 'r' in mode or '+' in mode
            if name in self.refs:
                if any(c in mode for c in 'wax+'): return self.reject('source_write')
                if not (self.load_depth or self.hash_depth): return self.reject('raw_archive')
            elif read and name not in self.read_outputs and name != self.telemetry_path: return self.reject('unlisted_read',name)
            elif not read and name not in self.read_outputs: return self.reject('unlisted_write',name)
        def opened(path, mode='r', *args, **kw): check(path,mode); return open0(path,mode,*args,**kw)
        def io_opened(path, mode='r', *args, **kw): check(path,mode); return io0(path,mode,*args,**kw)
        def path_opened(path, mode='r', *args, **kw): check(path,mode); return path0(path,mode,*args,**kw)
        class ScopedFileIO(file0):
            def __init__(self,path,mode='r',*args,**kw):
                check(path,mode);super().__init__(path,mode,*args,**kw)
        def imported(name,*args,**kw):
            if name.split('.')[0] in {'h5py','libero','mujoco','robosuite','pickle','pandas','src'} or (name.split('.')[0]=='torch' and not (self.jobs and self.jobs.audit)): return self.reject('runtime_import',name)
            return import0(name,*args,**kw)
        def fromfile(*args,**kw):
            if not self.member_depth: return self.reject('fromfile')
            return from0(*args,**kw)
        def read_array(*args,**kw):
            if not self.member_depth: return self.reject('read_array')
            return read0(*args,**kw)
        deny = lambda *args,**kw: self.reject('optimizer_model_or_alternate_reader')
        for obj,key,value in [(np,'load',loaded),(np.lib.npyio,'load',loaded),(np.lib.npyio.NpzFile,'__getitem__',get),(zipfile.ZipFile,'open',zip_open),
            (builtins,'open',opened),(io,'open',io_opened),(Path,'open',path_opened),(builtins,'__import__',imported),
            (np,'fromfile',fromfile),(np.lib.format,'read_array',read_array),(np.memmap,'__new__',deny),(np,'loadtxt',deny),(np,'genfromtxt',deny),
            (os,'open',deny),(io,'FileIO',ScopedFileIO)]: self.patch(obj,key,value)
        if 'torch' in sys.modules:
            torch = sys.modules['torch']; grad0 = torch.autograd.grad; linear0 = torch.nn.functional.linear
            for obj,key in [(torch,'load'),(torch.optim.Optimizer,'__init__'),(torch.nn.Module,'__call__'),(torch.autograd,'backward')]: self.patch(obj,key,deny)
            def grad(outputs,inputs,*args,**kw):
                if not self.jobs or not self.jobs.audit: return deny()
                self.jobs.gradient(inputs); return grad0(outputs,inputs,*args,**kw)
            def linear(x,w,b=None):
                if not self.jobs or not self.jobs.audit: return deny()
                d.p.require(tuple(x.shape)==(self.jobs.n,256) and tuple(w.shape)==(1,256) and tuple(b.shape)==(1,) and str(x.dtype)=='torch.float64','only fit auxiliary linear forms allowed')
                self.jobs.linear(len(x)); return linear0(x,w,b)
            self.patch(torch.autograd,'grad',grad); self.patch(torch.nn.functional,'linear',linear)
        return self
    def complete(self):
        expected = {(path,k) for path,ref in self.refs.items() for k in (d.HEAD_MEMBERS if ref['kind']=='head' else d.SOURCE_MEMBERS[ref['kind']])}
        d.p.require(len(self.completed)==len(self.attempts)==43 and set(self.completed)==set(self.refs) and set(self.member_completed)==expected and len(self.member_attempts)==len(self.member_completed)==305 and not self.denied,'archive/member budget differs')
        for path in self.refs: self.hash_one(path)
        self.limits.check()
    def record(self):
        return {'NPZ_loads':{'attempted':len(self.attempts),'completed':len(self.completed)},'member_decodes':{'attempted':len(self.member_attempts),'completed':len(self.member_completed)},
            'files':self.completed,'members':list(self.member_completed),'byte_SHA_checks':self.hash_checks,'denied_operations':self.denied,
            'input_CSV_reads':0,'optimizer_steps_parameter_updates_original_Policy_CNN_environment_or_sealed_calls':0}
