"""One-pass JSON scalar projection and fail-closed source access budget."""
from __future__ import annotations
import builtins
import io
import json
import math
import os
from pathlib import Path
import platform
import resource
import sys
import time
from experiments import libero_task7_direction_penalty_tradeoff_protocol as d


def scalar(row, columns):
    out = {k: row[k] for k in columns}
    d.require(all(type(v) in (int,float,str,bool) and (not isinstance(v,float) or math.isfinite(v)) for v in out.values()), 'nonfinite or nonscalar projection')
    return out


class Guard:
    def __init__(self, cfg):
        self.cfg = cfg; self.saved = []; self.allowed = None; self.seen = []; self.sha_checks = 0
        self.counts = dict(states=0,trial_flags=0,accepted_trials=0); self.denied = []
        self.start = time.monotonic()

    def reject(self, detail):
        self.denied.append(str(detail))
        raise PermissionError('saved-scalar access denied: '+str(detail))

    def limits(self):
        d.require(time.monotonic()-self.start <= self.cfg['execution']['seconds_per_stage']
                  and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= self.cfg['execution']['RSS_bytes'], 'scalar stage resource budget exceeded')

    def hash(self, ref):
        self.allowed = ref['path']
        try:
            d.require(d.sha(ref['path']) == ref['sha256'], 'source JSON changed')
        finally:
            self.allowed = None
        self.sha_checks += 1

    def project(self, backend):
        if backend not in self.cfg['inputs'] or backend in self.seen:
            return self.reject('unlisted/repeated JSON pass')
        self.seen.append(backend); ref = self.cfg['inputs'][backend]; self.hash(ref)
        d.require(Path(ref['path']).stat().st_size <= self.cfg['execution']['input_JSON_bytes_max'], 'source JSON too large')
        self.allowed = ref['path']
        try:
            raw = json.loads(Path(ref['path']).read_text())
        finally:
            self.allowed = None
        expected = 'complete_joint_direction_v2_pending_same_parameter_audit' if backend == 'P' else 'passed_independent_same_parameter_joint_direction_v2_audit'
        d.require(raw['status'] == expected and raw['science_config'] == d.PARENTS['science'] and raw['binding'] == d.PARENTS['binding'], 'source JSON role/provenance mismatch')
        if backend == 'U':
            d.require(raw['primary'] == self.cfg['inputs']['P'], 'U did not audit this P')
        states = [scalar(r,d.STATE) for r in raw['states']]
        trials = []
        for row in raw['trials']:
            d.require(type(row['accepted']) is bool, 'trial acceptance must be bool')
            self.counts['trial_flags'] += 1
            if row['accepted']:
                trials.append(scalar(row,d.TRIAL))
        d.require(len(states) == 1334 and len(raw['trials']) == 10920 and len(trials) == 1298, 'source coverage changed')
        self.counts['states'] += len(states); self.counts['accepted_trials'] += len(trials)
        del raw
        self.limits()
        return states, trials

    def __enter__(self):
        originals = [(builtins,'open'),(io,'open'),(Path,'open'),(io,'FileIO'),(os,'open'),(builtins,'__import__')]
        self.saved = [(o,n,getattr(o,n)) for o,n in originals]
        open0,io0,path0,file0,_,import0 = [x[2] for x in self.saved]
        def check(path, mode):
            if not isinstance(path,(str,bytes,os.PathLike)) or any(c in mode for c in 'wax+'):
                return self.reject('write/nonpath IO')
            if str(Path(os.fsdecode(path)).resolve()) != self.allowed:
                return self.reject(path)
        def opened(path,mode='r',*args,**kw):
            check(path,mode); return open0(path,mode,*args,**kw)
        def io_opened(path,mode='r',*args,**kw):
            check(path,mode); return io0(path,mode,*args,**kw)
        def path_opened(path,mode='r',*args,**kw):
            check(path,mode); return path0(path,mode,*args,**kw)
        class CheckedFileIO(file0):
            def __init__(self,path,mode='r',*args,**kw):
                check(path,mode); super().__init__(path,mode,*args,**kw)
        def imported(name,*args,**kw):
            if name.split('.')[0] in {'numpy','torch','h5py','pandas','pickle','mmap','libero','mujoco','robosuite','src'}:
                return self.reject('numerical/model/runtime import '+name)
            return import0(name,*args,**kw)
        for (obj,name,_),value in zip(self.saved,[opened,io_opened,path_opened,CheckedFileIO,lambda *a,**k:self.reject('os.open'),imported]):
            setattr(obj,name,value)
        return self

    def __exit__(self,*args):
        for obj,name,value in reversed(self.saved):
            setattr(obj,name,value)

    def complete(self):
        d.require(self.seen == ['P','U'] and self.counts == {'states':2668,'trial_flags':21840,'accepted_trials':2596} and not self.denied, 'scalar projection budget differs')
        for ref in self.cfg['inputs'].values():
            self.hash(ref)
        d.require(self.sha_checks == 4, 'source SHA budget differs'); self.limits()

    def record(self):
        return dict(source_JSON_passes=len(self.seen),source_SHA_checks=self.sha_checks,**self.counts,
                    projections=self.cfg['projections'],denied=self.denied,
                    physical_decode='whole_JSON_including_unprojected_fields;semantic_scalar_whitelist_only',
                    arrays_CSV_checkpoints_private_or_sealed_reads=0,new_training_forward_gradient_environment=0)


def runtime(cfg):
    d.require(platform.python_version() == cfg['execution']['python'] and 'numpy' not in sys.modules and 'torch' not in sys.modules
              and all(os.environ.get(k) == v for k,v in cfg['execution']['environment'].items()), 'fresh bounded CPU runtime required')
    return dict(python=platform.python_version(),device='CPU',pid=os.getpid(),environment=cfg['execution']['environment'],environment_seed=None,initial_states=[])
