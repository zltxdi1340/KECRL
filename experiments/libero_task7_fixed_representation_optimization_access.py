"""Scoped archive IO and fit-only finite optimization work accounting."""
from __future__ import annotations
import builtins
import hashlib
import io
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
from experiments import libero_task7_fixed_representation_optimization_protocol as d
from experiments import libero_task7_endpoint_gradient_access as old
from experiments.libero_task7_endpoint_gradient_access_v2 import RepresentationWork

base, runtime = old.base, old.runtime


def preflight(config=d.DEFAULT_CONFIG, binding=d.DEFAULT_BINDING):
    return d.validate(config), d.require_binding(config, binding)


def validate_call(cfg,science_pin,binding_pin):
    frozen,bound=preflight(Path(science_pin['path']),Path(binding_pin['path']))
    d.p.require(cfg==frozen and d.p.pin(Path(science_pin['path']))==science_pin
        and d.p.pin(Path(binding_pin['path']))==binding_pin and bound['before_research_payload'] is True,'payload API requires unchanged bound science')


def identity(head): return {k: head[k] for k in ('task','seed','arm','fold','mode')}
def key(head): return tuple(head[k] for k in ('seed','arm','fold','mode'))
def digest(array): return hashlib.sha256(np.asarray(array,dtype='<f8').tobytes()).hexdigest()
def readonly(array): array.flags.writeable = False; return array


def members(kind):
    return d.NEW_HEAD_MEMBERS if kind == 'newhead' else d.HEAD_MEMBERS if kind == 'head' else d.SOURCE_MEMBERS[kind]


def load(ref, kind):
    with np.load(ref['path'],allow_pickle=False) as data: return {k:data[k] for k in members(kind)}


class Jobs:
    """Only registered fit slices and accepted trial copies can advance a head."""
    def __init__(self,cfg,limits,audit=False):
        self.cfg,self.limits,self.audit=cfg,limits,audit
        self.expected={key(h) for h in cfg['inputs']['heads']};self.done={};self.active=None;self.phase=None;self.array_sources={};self.fit_registrations=set()
        self.counts=dict(heads=0,updates=0,accepted_parameter_coordinates=0,gradient_states=0,
            gradient_vectors=0,autograd_calls=0,trial_forwards=0,linear_forms=0,
            state_fit_rows=0,trial_fit_rows=0,linear_form_rows=0)

    def register_fit(self,model,plan,z,targets,predictions,rows,work):
        k=(model['seed'],model['arm'],plan['fold'])
        d.p.require(self.active is None and k not in self.fit_registrations and work.builds.get(k)==id(z)
            and np.array_equal(rows,np.asarray(plan['fit_rows'],np.int32))
            and self.array_sources.get(id(targets))==(str(Path(self.cfg['inputs']['shared']['path']).resolve()),'targets')
            and self.array_sources.get(id(predictions))==(str(Path(model['current_archive']['path']).resolve()),'predictions'),'fit context origin differs')
        context=(z,readonly(targets[rows].astype(np.float64)),readonly(predictions[rows].astype(np.float64)))
        self.allowed_key=k;self.allowed_context=tuple(map(id,context));self.fit_registrations.add(k);return context

    def head(self,k,z,y,p0,rows,theta):
        d.p.require(self.active is None and k in self.expected and k not in self.done,'head identity repeated or unregistered')
        plan=self.cfg['inputs']['folds'][k[2]];n=len(plan['fit_rows'])
        d.p.require(k[:3]==self.allowed_key and tuple(map(id,(z,y,p0)))==self.allowed_context
            and np.array_equal(rows,np.asarray(plan['fit_rows'],np.int32)) and z.shape==(n,256) and y.shape==p0.shape==(n,7)
            and all(v.dtype==np.float64 and not v.flags.writeable and np.isfinite(v).all() for v in (z,y,p0)),'worker inputs must be readonly own-fit slices')
        d.p.require(theta.shape==(257,) and theta.dtype==np.float64 and np.isfinite(theta).all(),'invalid500 parameter copy')
        self.active=k;self.n=n;self.context=(z,y,p0);self.context_hash=[digest(a) for a in self.context]
        self.start=readonly(theta.copy());self.current=readonly(theta.copy());self.control_hash=digest(self.start)
        self.updates=0;self.next_trial=0;self.measured=-1;self.pending=None;self.phase=None;self.local_trials=0

    def check_context(self,z,y,p0):
        d.p.require(self.active is not None and all(x is v for x,v in zip((z,y,p0),self.context)),'unregistered observation/target batch')

    def state_begin(self,theta,params=()):
        d.p.require(self.active is not None and self.phase is None and self.measured!=self.updates and np.array_equal(theta,self.current),'unexpected or repeated gradient state')
        self.phase='state';self.forms=self.grads=0;self.params=tuple(params)
        self.theta_hash=digest(self.current)
        self.before=[v.detach().numpy().tobytes() for v in self.params]
        if self.audit:
            d.p.require(len(params)==2 and [p.numel() for p in params]==[256,1] and all(str(p.dtype)=='torch.float64' and p.device.type=='cpu' and p.requires_grad for p in params),'unregistered autograd parameters')
            d.p.require(np.array_equal(np.r_[params[0].detach().numpy(),params[1].detach().numpy()],theta),'autograd parameter copy differs')

    def linear(self,n):
        d.p.require(n==self.n and self.phase in ('state','trial') and self.forms<(2 if self.phase=='state' else 1),'linear form outside bounded job')
        self.forms+=1;self.counts['linear_forms']+=1;self.counts['linear_form_rows']+=n;self.limits.check()

    def gradient(self,params=()):
        d.p.require(self.phase=='state' and self.forms==1 and self.grads<3,'gradient outside current state')
        if self.audit: d.p.require(tuple(map(id,params))==tuple(map(id,self.params)),'gradient of unregistered parameters')
        self.grads+=1;self.counts['gradient_vectors']+=1;self.counts['autograd_calls']+=int(self.audit)

    def state_end(self,row):
        d.p.require(self.phase=='state' and self.forms==2 and self.grads==3 and self.before==[v.detach().numpy().tobytes() for v in self.params]
            and digest(self.current)==self.theta_hash and not self.current.flags.writeable,'incomplete state or parameter mutation')
        self.last=row;self.measured=self.updates;self.phase=None;self.params=();self.next_trial=0
        self.counts['gradient_states']+=1;self.counts['state_fit_rows']+=self.n

    def trial_begin(self,index,theta):
        d.p.require(self.phase is None and self.measured==self.updates and self.updates<50 and self.pending is None
            and index==self.next_trial and index<12 and self.last['total_l2']>0 and self.last['directional_derivative']<0,'trial outside frozen round')
        expected=self.current+d.trial_alpha(index)*np.asarray(self.last['direction'])
        d.p.require(np.array_equal(theta,expected) and not theta.flags.writeable,'trial is not current minus alpha full gradient')
        self.phase='trial';self.forms=0;self.trial_theta=theta;self.trial_index=index
        self.trial_hash=digest(theta)

    def trial_end(self,trainable):
        d.p.require(self.phase=='trial' and self.forms==1 and digest(self.current)==self.theta_hash
            and digest(self.trial_theta)==self.trial_hash and not self.trial_theta.flags.writeable,'incomplete trial or trial/control mutation')
        accepted=d.armijo_accept(self.last['trainable_objective'],trainable,self.last['directional_derivative'],self.trial_index)
        self.phase=None;self.local_trials+=1;self.counts['trial_forwards']+=1;self.counts['trial_fit_rows']+=self.n
        self.next_trial+=1
        if accepted:self.pending=self.trial_theta
        self.trial_theta=None;return accepted

    def commit(self,theta):
        d.p.require(self.phase is None and self.pending is theta and self.pending is not None and self.updates<50,'unaccepted parameter update')
        self.current=self.pending;self.pending=None;self.updates+=1
        self.counts['updates']+=1;self.counts['accepted_parameter_coordinates']+=257

    def finish(self,reason):
        d.p.require(self.phase is None and self.pending is None and self.measured==self.updates
            and reason==d.stopping_reason(self.updates,self.last['total_l2'],self.last['directional_derivative'],self.next_trial)
            and reason is not None,'invalid stop or checkpoint selection')
        d.p.require(digest(self.start)==self.control_hash and self.context_hash==[digest(v) for v in self.context]
            and digest(self.current)==self.theta_hash and all(not v.flags.writeable for v in self.context+(self.start,self.current)),'source inputs or hold mutated')
        self.last_finished=(self.active,self.start,self.current,self.updates)
        self.last_finished_hashes=(digest(self.start),digest(self.current))
        self.done[self.active]={'updates':self.updates,'trials':self.local_trials,'reason':reason}
        self.counts['heads']+=1;self.active=None;self.context=();self.limits.check()

    def complete(self):
        c=self.counts;b=self.cfg['budget']
        d.p.require(set(self.done)==self.expected and len(self.fit_registrations)==18 and self.active is None and c['gradient_states']==36+c['updates']
            and c['gradient_vectors']==3*c['gradient_states'] and c['autograd_calls']==(c['gradient_vectors'] if self.audit else 0)
            and c['linear_forms']==2*c['gradient_states']+c['trial_forwards'] and c['updates']<=1800
            and c['accepted_parameter_coordinates']==c['updates']*257,'work ledger mismatch')
        for field,cap in [('gradient_states','gradient_states_max_per_stage'),('trial_forwards','trial_objective_forms_max_per_stage'),
            ('linear_forms','all_linear_forms_max_per_stage'),('state_fit_rows','gradient_state_fit_row_visits_max_per_stage'),
            ('trial_fit_rows','trial_component3_fit_row_visits_max_per_stage'),('linear_form_rows','all_linear_form_row_visits_max_per_stage')]:
            d.p.require(c[field]<=b[cap],'work exceeds '+field)


class Guard(base.ReadGuard):
    def __init__(self,cfg,table_paths,new_paths,limits,jobs,new_refs=()):
        pairs=[(cfg['inputs']['shared'],'shared')]+[(m['current_archive'],'current') for m in cfg['inputs']['models']]+[(h['archive'],'head') for h in cfg['inputs']['heads']]
        self.cfg=cfg;self.refs={str(Path(ref['path']).resolve()):{**ref,'kind':kind} for ref,kind in pairs}
        self.original=set(self.refs);d.p.require(len(self.original)==43,'43 original archives required')
        self.new_paths={str(Path(path).resolve()):key(h) for path,h in zip(new_paths,cfg['inputs']['heads'])}
        d.p.require(len(self.new_paths)==36 and not self.original.intersection(self.new_paths),'36 distinct new outputs required')
        for ref in new_refs:
            name=str(Path(ref['path']).resolve());d.p.require(name in self.new_paths,'unlisted new archive')
            self.refs[name]={**ref,'kind':'newhead'}
        d.p.require(len(self.refs)==(79 if jobs.audit else 43),'new archive audit manifest differs')
        self.read_outputs={str(Path(path).resolve()) for path in table_paths};self.telemetry_path=str(Path('/proc/self/status').resolve())
        self.limits,self.jobs=limits,jobs;self.saved=[];self.denied=[]
        self.attempts=[];self.completed=[];self.member_attempts=[];self.member_completed=[];self.hash_checks=[];self.writes=[];self.write_members=[]
        self.registry={};self.zip_registry={};self.load_depth=self.member_depth=self.hash_depth=self.save_depth=0;self.active_member=None

    def reject(self,kind,detail=''):return super().reject(kind,detail)

    def hash_one(self,name,expected=None):
        self.hash_depth+=1
        try:value=d.p.sha(name)
        finally:self.hash_depth-=1
        if expected is not None:d.p.require(value==expected,'archive bytes changed')
        self.hash_checks.append(name);return value

    def __enter__(self):
        load0,get0,zip0=np.load,np.lib.npyio.NpzFile.__getitem__,zipfile.ZipFile.open
        save0=np.savez;open0,io0,path0,import0=builtins.open,io.open,Path.open,builtins.__import__;file0=io.FileIO
        read0,from0=np.lib.format.read_array,np.fromfile
        def loaded(path,*args,**kw):
            if not isinstance(path,(str,Path)):return self.reject('nonpath')
            name=str(Path(path).resolve())
            if name not in self.refs or name in self.attempts:return self.reject('unlisted_or_repeated_archive',name)
            if args or kw!={'allow_pickle':False}:return self.reject('pickle_mmap_arguments')
            self.attempts.append(name);self.hash_one(name,self.refs[name]['sha256']);self.load_depth+=1
            try:z=load0(path,allow_pickle=False)
            finally:self.load_depth-=1
            d.p.require(isinstance(z,np.lib.npyio.NpzFile),'NPZ required');self.registry[id(z)]=name;self.zip_registry[id(z.zip)]=name
            if self.refs[name]['kind']=='newhead':d.p.require(set(z.files)==set(d.NEW_HEAD_MEMBERS),'new archive has undeclared fields')
            self.completed.append(name);self.limits.check();return z
        def get(z,field):
            name=self.registry.get(id(z));kind=self.refs[name]['kind'] if name in self.refs else None
            if kind is None or field not in members(kind) or (name,field) in self.member_attempts:return self.reject('unlisted_or_repeated_member',field)
            self.member_attempts.append((name,field));self.member_depth+=1;self.active_member=name,field
            try:value=get0(z,field)
            finally:self.member_depth-=1;self.active_member=None
            d.p.require(isinstance(value,np.ndarray) and value.dtype.kind in 'biuf' and not value.dtype.hasobject,'numeric arrays only')
            readonly(value);self.jobs.array_sources[id(value)]=(name,field);self.member_completed.append((name,field));self.limits.check();return value
        def saved(path,*args,**arrays):
            if not isinstance(path,(str,Path)):return self.reject('nonpath_save')
            name=str(Path(path).resolve())
            if self.jobs.audit or name not in self.new_paths or name in self.writes or Path(name).exists():return self.reject('unlisted_or_repeated_save',name)
            d.p.require(not args and set(arrays)==set(d.NEW_HEAD_MEMBERS),'new head fields differ')
            d.p.require(hasattr(self.jobs,'last_finished'),'new head save before completion')
            k,start,final,updates=self.jobs.last_finished
            d.p.require((digest(start),digest(final))==self.jobs.last_finished_hashes,'completed parameters mutated before save')
            plan=self.cfg['inputs']['folds'][k[2]]
            d.p.require(arrays['fit_rows'].dtype==arrays['fit_demo_indices'].dtype==np.int32
                and arrays['fit_rows'].tolist()==plan['fit_rows'] and arrays['fit_demo_indices'].tolist()==plan['fit_demos']
                and all(arrays[f].shape==(256,) and arrays[f].dtype==np.float64 and np.isfinite(arrays[f]).all() for f in ('scaler_mean','scaler_scale'))
                and (arrays['scaler_scale']>0).all(),'new head scaler/fit roles differ')
            d.p.require(k==self.new_paths[name] and arrays['weights'].dtype==arrays['bias'].dtype==np.float64
                and arrays['weights'].shape==(2,256) and arrays['bias'].shape==(2,)
                and np.array_equal(np.c_[arrays['weights'],arrays['bias']],np.stack([start,final]))
                and arrays['additional_updates'].dtype==np.int32 and arrays['additional_updates'].tolist()==[0,updates]
                and all(isinstance(v,np.ndarray) and v.dtype.kind in 'biuf' for v in arrays.values()),'new head endpoint schema differs')
            self.save_depth+=1
            try:save0(name,**arrays)
            finally:self.save_depth-=1
            self.writes.append(name);pin={'path':name,'sha256':self.hash_one(name)};self.limits.check();return pin
        def zip_open(z,field,mode='r',*args,**kw):
            name=self.zip_registry.get(id(z));member=field.filename if isinstance(field,zipfile.ZipInfo) else field
            if mode=='w':
                target=str(Path(z.filename).resolve()) if isinstance(z.filename,(str,Path)) else None
                entry=(target,str(member).removesuffix('.npy'))
                if self.jobs.audit or not self.save_depth or target not in self.new_paths or entry[1] not in d.NEW_HEAD_MEMBERS or entry in self.write_members:return self.reject('zip_write_bypass',member)
                self.write_members.append(entry);return zip0(z,field,mode,*args,**kw)
            if name is not None and (not self.member_depth or self.active_member!=(name,str(member).removesuffix('.npy'))):return self.reject('zip_bypass',member)
            return zip0(z,field,mode,*args,**kw)
        def check(path,mode):
            if not isinstance(path,(str,Path)):return self.reject('raw_descriptor')
            name=str(Path(path).resolve());write=any(c in mode for c in 'wax+')
            if name in self.new_paths and not self.jobs.audit:
                if write and self.save_depth:return
                if not write and self.hash_depth:return
                return self.reject('raw_new_head',name)
            if name in self.refs:
                if write:return self.reject('source_write')
                if not (self.load_depth or self.hash_depth):return self.reject('raw_archive')
            elif name==self.telemetry_path and not write:return
            elif name in self.read_outputs:
                if write and self.jobs.audit:return self.reject('audit_table_write')
            else:return self.reject('unlisted_IO',name)
        def opened(path,mode='r',*args,**kw):check(path,mode);return open0(path,mode,*args,**kw)
        def io_opened(path,mode='r',*args,**kw):check(path,mode);return io0(path,mode,*args,**kw)
        def path_opened(path,mode='r',*args,**kw):check(path,mode);return path0(path,mode,*args,**kw)
        class ScopedFileIO(file0):
            def __init__(self,path,mode='r',*args,**kw):check(path,mode);super().__init__(path,mode,*args,**kw)
        def imported(name,*args,**kw):
            if name.split('.')[0] in {'h5py','libero','mujoco','robosuite','pickle','pandas','src'} or (name.split('.')[0]=='torch' and not self.jobs.audit):return self.reject('runtime_import',name)
            return import0(name,*args,**kw)
        def fromfile(*args,**kw):
            if not self.member_depth:return self.reject('fromfile')
            return from0(*args,**kw)
        def read_array(*args,**kw):
            if not self.member_depth:return self.reject('read_array')
            return read0(*args,**kw)
        deny=lambda *args,**kw:self.reject('optimizer_model_or_alternate_reader')
        for obj,name,value in [(np,'load',loaded),(np.lib.npyio,'load',loaded),(np.lib.npyio.NpzFile,'__getitem__',get),(np,'savez',saved),
            (np,'savez_compressed',deny),(np,'save',deny),(zipfile.ZipFile,'open',zip_open),(builtins,'open',opened),(io,'open',io_opened),
            (Path,'open',path_opened),(builtins,'__import__',imported),(np,'fromfile',fromfile),(np.lib.format,'read_array',read_array),
            (np.memmap,'__new__',deny),(np,'loadtxt',deny),(np,'genfromtxt',deny),(os,'open',deny),(io,'FileIO',ScopedFileIO)]:self.patch(obj,name,value)
        if 'torch' in sys.modules:
            torch=sys.modules['torch'];grad0=torch.autograd.grad;linear0=torch.nn.functional.linear
            for obj,name in [(torch,'load'),(torch,'save'),(torch.optim.Optimizer,'__init__'),(torch.nn.Module,'__call__'),(torch.autograd,'backward')]:self.patch(obj,name,deny)
            def grad(outputs,inputs,*args,**kw):self.jobs.gradient(inputs);return grad0(outputs,inputs,*args,**kw)
            def linear(x,w,b=None):
                d.p.require(self.jobs.audit and id(x)==self.jobs.tensor_input_id and tuple(x.shape)==(self.jobs.n,256) and tuple(w.shape)==(1,256) and tuple(b.shape)==(1,) and str(x.dtype)=='torch.float64','only own-fit auxiliary forms allowed')
                self.jobs.linear(len(x));return linear0(x,w,b)
            self.patch(torch.autograd,'grad',grad);self.patch(torch.nn.functional,'linear',linear)
        return self

    def complete(self):
        expected={(name,k) for name,ref in self.refs.items() for k in members(ref['kind'])}
        d.p.require(set(self.completed)==set(self.refs) and len(self.attempts)==len(self.completed)==(79 if self.jobs.audit else 43)
            and set(self.member_completed)==expected and len(self.member_attempts)==len(self.member_completed)==(557 if self.jobs.audit else 305)
            and len(self.writes)==(0 if self.jobs.audit else 36) and len(self.write_members)==(0 if self.jobs.audit else 252) and not self.denied,'archive IO budget differs')
        for name,ref in self.refs.items():self.hash_one(name,ref['sha256'])
        d.p.require(len(self.hash_checks)==(158 if self.jobs.audit else 122),'archive SHA budget differs');self.limits.check()

    def record(self):
        return {'NPZ_loads':{'attempted':len(self.attempts),'completed':len(self.completed)},
            'member_decodes':{'attempted':len(self.member_attempts),'completed':len(self.member_completed)},
            'files':self.completed,'members':list(self.member_completed),'new_NPZ_writes':self.writes,'new_member_writes':self.write_members,'byte_SHA_checks':self.hash_checks,
            'denied_operations':self.denied,'input_research_CSV_reads':0,'original_Policy_CNN_environment_or_sealed_calls':0}
