"""Bound source-only archives and paired finite optimization work guard."""
from __future__ import annotations
import builtins
import copy
import hashlib
import io
import math
import os
from pathlib import Path
import sys
import zipfile
import numpy as np
from experiments import libero_task7_joint_direction_protocol as d
from experiments import libero_task7_fixed_representation_optimization_access as legacy

base,runtime=legacy.base,legacy.runtime
identity,key,digest,readonly=legacy.identity,legacy.key,legacy.digest,legacy.readonly
rules=d.rules


def preflight(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    return d.validate(config),d.require_binding(config,binding)


def validate_call(cfg,science_pin,binding_pin):
    frozen,bound=preflight(Path(science_pin['path']),Path(binding_pin['path']))
    d.p.require(cfg==frozen and d.p.pin(Path(science_pin['path']))==science_pin
        and d.p.pin(Path(binding_pin['path']))==binding_pin and bound['before_research_payload'] is True,'payload API requires unchanged bound science')


def members(kind):
    return d.finer.NEW_HEAD_MEMBERS if kind=='newhead' else d.finer.HEAD_MEMBERS if kind=='head' else d.finer.SOURCE_MEMBERS[kind]


def load(ref,kind):
    with np.load(ref['path'],allow_pickle=False) as data:return {f:data[f] for f in members(kind)}


class RepresentationWork(legacy.RepresentationWork):
    def record(self):
        value=super().record()
        value['source_head_admissions']=len(self.builds)
        value['optimization_path_admissions']=value.pop('head_admissions')
        return value


class DirectionWork:
    """Count actual direction operations; one construction per measured state."""
    def __init__(self,jobs=None):
        self.jobs=jobs;self.active=False;self.finished=None
        self.counts=dict(constructions=0,norm_reductions=0,dot_reductions=0,vector_forms=0,coordinate_writes=0)

    def begin(self,arm,total,base):
        d.p.require(not self.active and arm in d.ARMS,'repeated direction construction')
        if self.jobs:
            j=self.jobs
            d.p.require(j.phase=='state' and j.forms==1 and j.grads==3 and j.active[-1]==arm
                and self.finished!=(j.active,j.updates),'direction outside registered state')
        self.active=True;self.arm=arm;self.local=dict(norm_reductions=0,dot_reductions=0,vector_forms=0)
        d.p.require(len(total)==len(base)==257,'direction inputs must have257 coordinates')
        self.input_hashes=(digest(total),digest(base))

    def norm(self,values,backend=None):
        d.p.require(self.active and self.local['norm_reductions']<4 and len(values)==257,'extra direction norm')
        self.local['norm_reductions']+=1;self.counts['norm_reductions']+=1
        value=backend(values) if backend else math.hypot(*values)
        d.finite(value);return value

    def dot(self,left,right,backend=None):
        d.p.require(self.active and self.local['dot_reductions']<2 and len(left)==len(right)==257,'extra direction dot')
        self.local['dot_reductions']+=1;self.counts['dot_reductions']+=1
        value=backend(left,right) if backend else math.fsum(x*y for x,y in zip(left,right))
        d.finite(value);return value

    def vector(self,values):
        d.p.require(self.active and self.local['vector_forms']<4 and len(values)==257,'extra direction vector')
        d.finite(*values);self.local['vector_forms']+=1;self.counts['vector_forms']+=1;self.counts['coordinate_writes']+=257
        return values

    def end(self,meta):
        d.p.require(self.active and self.local['dot_reductions']==2 and self.local['vector_forms'] in (1,4)
            and self.local['norm_reductions'] in (3,4),'incomplete direction work')
        d.p.require(meta['status'] in ('zero_total_gradient','negative_total_gradient','base_zero_use_total','joint_bisector','unresolved_joint_direction'),
            'unlisted direction status')
        if meta['status']!='unresolved_joint_direction':
            d.p.require(abs(meta['direction_norm']-meta['total_norm'])<=1e-10,'direction norm not matched')
        if self.jobs:
            self.finished=(self.jobs.active,self.jobs.updates)
            self.jobs.direction_record=(self.finished,self.input_hashes,dict(meta,direction=list(meta['direction'])))
        self.active=False;self.counts['constructions']+=1

    def complete(self):
        d.p.require(self.jobs is not None and not self.active,'direction work not complete')
        b=self.jobs.cfg['budget'];c=self.counts
        d.p.require(c['constructions']==self.jobs.counts['gradient_states'] and c['dot_reductions']==2*c['constructions']
            and c['coordinate_writes']==257*c['vector_forms'],'direction ledger does not conserve')
        for field,cap in [('constructions','direction_constructions_max_per_stage'),('norm_reductions','direction_norm_reductions_max_per_stage'),
            ('dot_reductions','direction_dot_reductions_max_per_stage'),('vector_forms','direction_vector_forms_max_per_stage'),
            ('coordinate_writes','direction_vector_coordinate_writes_max_per_stage')]:
            d.p.require(c[field]<=b[cap],'direction budget exceeded: '+field)


def stop(arm,k,row,rejected=0):
    return d.stopping_reason(arm,k,row['total_l2'],row['direction_status'],row['selected_joint'],
        row['directional_derivative'],row['exact_MSE_slope'],rejected)


def direction_fields(meta):
    return dict(direction_status=meta['status'],direction_total_reference_norm=meta['total_norm'],
        direction_base_reference_norm=meta['base_norm'],direction_unit_sum_norm=meta['unit_sum_norm'],
        chosen_direction_norm=meta['direction_norm'],selected_T_slope=meta['selected_T_slope'],
        selected_MSE_slope=meta['selected_MSE_slope'],selected_joint=meta['selected_joint'])


class Jobs(legacy.Jobs):
    def __init__(self,cfg,limits,audit=False):
        super().__init__(cfg,limits,audit)
        self.expected={key(h)+(arm,) for h in cfg['inputs']['heads'] for arm in d.ARMS}
        self.sources={};self.start_mse_values={};self.pair_finals={};self.open_source=None
        self.trials_by_arm={arm:0 for arm in d.ARMS};self.direction_work=DirectionWork(self)
        self.counts.update(trial_MSE_penalty_pairs=0,mse_vetoes_of_armijo_passes=0)

    def source(self,spec,head,start):
        k=key(spec);name=str(Path(spec['archive']['path']).resolve())
        d.p.require(self.active is None and self.open_source is None and k not in self.sources and k+(d.ARMS[0],) in self.expected
            and all(self.array_sources.get(id(head[f]))==(name,f) for f in d.finer.HEAD_MEMBERS),'unregistered source500 fields')
        d.p.require(start.dtype==np.float64 and start.shape==(257,) and np.array_equal(start,
            np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])]),'not original auxiliary500 start')
        self.sources[k]={'start':readonly(start.copy()),'scaler_mean':digest(head['scaler_mean']),'scaler_scale':digest(head['scaler_scale'])}
        self.open_source=k

    def head(self,k,z,y,p0,rows,theta):
        source=k[:-1]
        d.p.require(source==self.open_source and source in self.sources and np.array_equal(theta,self.sources[source]['start'])
            and (k[-1]==d.ARMS[0] or source+(d.ARMS[0],) in self.done),'wrong source start or path order')
        super().head(k,z,y,p0,rows,theta)
        self.round_rows=[];self.local_vetoes=0;self.round_summaries=[];self.mse_deltas=[];self.accepted_delta=None

    @staticmethod
    def _terms(row):
        rules.finite_nonnegative(*(row[f] for f in ('component3_mse','penalty','trainable_objective','total_objective')))
        d.p.require(abs(row['trainable_objective']-row['component3_mse']/7-row['penalty'])<=1e-10,'lambda1 terms do not conserve')

    def state_end(self,row):
        registered,hashes,meta=self.direction_record
        d.p.require(registered==(self.active,self.updates) and hashes==(digest(row['total_gradient']),digest(row['base_gradient']))
            and row['direction']==meta['direction'] and all(row[f]==v for f,v in direction_fields(meta).items()),'direction construction/state differs')
        if self.active[-1]==d.ARMS[0]:
            d.p.require(all(x==-g for x,g in zip(row['direction'],row['total_gradient'])),'control direction changed')
        d.finite(row['exact_MSE_slope'],row['exact_penalty_slope'])
        d.p.require(abs(row['directional_derivative']-row['exact_MSE_slope']/7-row['exact_penalty_slope'])<=1e-10,'exact directional terms do not conserve')
        self._terms(row)
        if self.updates:
            d.p.require(row['trainable_objective']<self.last['trainable_objective'] and len(self.mse_deltas)==self.updates
                and abs(row['component3_mse']-self.last['component3_mse']-self.mse_deltas[-1])<=1e-10,'accepted state/step ledger differs')
        super().state_end(row)
        source=self.active[:-1]
        if self.updates==0:
            if self.active[-1]==d.ARMS[0]:self.start_mse_values[source]=row['component3_mse']
            else:d.p.require(row['component3_mse']==self.start_mse_values[source],'paired500 MSE changed')
        self.fixed_mse=self.start_mse_values[source]
        d.p.require(row['component3_mse']<=self.fixed_mse,'accepted state exceeds fixed500 envelope')
        self.round_rows=[]

    def trial_begin(self,index,theta):
        d.p.require(self.phase is None and self.measured==self.updates and self.pending is None
            and type(index) is int and index==self.next_trial and 0<=index<20 and stop(self.active[-1],self.updates,self.last) is None
            and self.fixed_mse==self.start_mse_values[self.active[:-1]],'trial outside frozen direction round')
        expected=self.current+d.trial_alpha(self.active[-1],index)*np.asarray(self.last['direction'])
        d.p.require(np.array_equal(theta,expected) and not theta.flags.writeable,'trial does not follow chosen direction')
        self.phase='trial';self.forms=0;self.trial_theta=theta;self.trial_index=index;self.trial_hash=digest(theta)

    def trial_end(self,row):
        d.p.require(self.phase=='trial' and self.forms==1 and digest(self.current)==self.theta_hash
            and digest(self.trial_theta)==self.trial_hash and not self.trial_theta.flags.writeable,'incomplete trial or mutation')
        self._terms(row)
        d.p.require(abs(row['total_objective']-row['trainable_objective']-self.last['fixed_other6_floor'])<=1e-10,'trial other6 floor differs')
        arm=self.active[-1];decision=d.trial_decision(arm,self.last['trainable_objective'],row['trainable_objective'],
            self.last['directional_derivative'],self.trial_index,row['component3_mse'],self.fixed_mse)
        entry=dict(row,trial_index=self.trial_index,alpha=d.trial_alpha(arm,self.trial_index),**decision)
        self.round_rows.append(entry);self.phase=None;self.local_trials+=1;self.next_trial+=1
        self.counts['trial_forwards']+=1;self.counts['trial_fit_rows']+=self.n;self.counts['trial_MSE_penalty_pairs']+=1;self.trials_by_arm[arm]+=1
        veto=decision['armijo_pass'] and not decision['mse_envelope_pass']
        self.local_vetoes+=int(veto);self.counts['mse_vetoes_of_armijo_passes']+=int(veto)
        if decision['accepted']:
            self.pending=self.trial_theta;self.accepted_delta=row['component3_mse']-self.last['component3_mse']
        self.trial_theta=None
        if decision['accepted'] or self.next_trial==20:
            summary=d.finer.validate_round('fine20',self.last['trainable_objective'],self.last['directional_derivative'],self.fixed_mse,self.round_rows)
            summary['armijo_passes']=sum(r['armijo_pass'] for r in self.round_rows);self.round_summaries.append(summary)
        return decision

    def commit(self,theta):
        super().commit(theta);d.p.require(self.accepted_delta is not None,'missing accepted MSE increment')
        self.mse_deltas.append(self.accepted_delta);self.accepted_delta=None

    def finish(self,reason):
        d.p.require(self.phase is None and self.pending is None and self.measured==self.updates and reason is not None
            and reason==stop(self.active[-1],self.updates,self.last,self.next_trial),'invalid direction stop or endpoint selection')
        d.p.require(digest(self.start)==self.control_hash and self.context_hash==[digest(v) for v in self.context]
            and digest(self.current)==self.theta_hash and all(not v.flags.writeable for v in self.context+(self.start,self.current)),'source/parameters mutated')
        k=self.active;source=k[:-1];self.last_finished=(k,self.start,self.current,self.updates)
        self.last_finished_hashes=(digest(self.start),digest(self.current))
        self.done[k]=dict(updates=self.updates,trials=self.local_trials,reason=reason,mse_vetoes=self.local_vetoes,
            round_summaries=list(self.round_summaries),mse_deltas=list(self.mse_deltas))
        self.pair_finals[k]=(self.current,self.updates);self.counts['heads']+=1;self.active=None;self.context=();self.limits.check()
        if k[-1]==d.ARMS[1]:
            baseline,kb=self.pair_finals[source+(d.ARMS[0],)]
            self.completed_pair=(source,(self.sources[source]['start'],baseline,self.current),(kb,self.updates))
            self.completed_pair_hashes=tuple(digest(v) for v in self.completed_pair[1]);self.open_source=None

    def complete(self):
        super().complete();self.direction_work.complete()
        d.p.require(len(self.sources)==18 and self.open_source is None and self.counts['trial_MSE_penalty_pairs']==self.counts['trial_forwards']
            and all(self.trials_by_arm[arm]<=18000 for arm in d.ARMS),'direction/source/arm ledger differs')


class History:
    """One allowed scalar projection; no old vectors or CSVs enter numerical work."""
    def __init__(self,cfg):
        spec=cfg['historical_control'];raw=d.p.load_pin(spec['source_JSON']);self.arm=spec['optimization_arm']
        self.data={name:[{f:row[f] for f in fields} for row in raw[name]
            if (row['condition'] in spec['endpoint_conditions'] if name=='endpoints' else row['optimization_arm']==self.arm)]
            for name,fields in spec['projections'].items()}
        self.counts={name:0 for name in self.data};self.checks=0
        d.p.require({k:len(v) for k,v in self.data.items()}==cfg['budget']['historical_control_JSON_projected_records_per_stage'],'historical projection rows differ')
        self.expected_counts=cfg['budget']['historical_control_JSON_projected_records_per_stage']

    def row(self,table,actual):
        i=self.counts[table];d.p.require(i<len(self.data[table]),'extra historical parity row')
        expected=self.data[table][i]
        for field,wanted in expected.items():
            value=actual[field]
            if field=='condition':value={d.ARMS[0]+'_final':self.arm+'_final'}.get(value,value)
            if field=='optimization_arm':value=self.arm if value==d.ARMS[0] else value
            if type(wanted) is float:
                d.p.require(type(value) in (int,float) and math.isfinite(value) and abs(value-wanted)<=1e-10,'historical '+table+'.'+field+' differs')
            else:d.p.require(type(value) is type(wanted) and value==wanted,'historical '+table+'.'+field+' differs')
            self.checks+=1
        self.counts[table]+=1

    def complete(self):
        d.p.require(self.counts==self.expected_counts,'historical fine20 parity incomplete')
        return {'projected_records':dict(self.counts),'scalar_checks':self.checks,'all18_paths_reproduced':True}


class Guard(base.ReadGuard):
    def __init__(self,cfg,table_paths,new_paths,limits,jobs,new_refs=()):
        pairs=[(cfg['inputs']['shared'],'shared')]+[(m['current_archive'],'current') for m in cfg['inputs']['models']]+[(h['archive'],'head') for h in cfg['inputs']['heads']]
        self.cfg=cfg;self.refs={str(Path(ref['path']).resolve()):{**ref,'kind':kind} for ref,kind in pairs}
        self.original=set(self.refs);d.p.require(len(self.original)==25,'25 original archives required')
        self.new_paths={str(Path(path).resolve()):key(h) for path,h in zip(new_paths,cfg['inputs']['heads'])}
        d.p.require(len(self.new_paths)==18 and not self.original.intersection(self.new_paths),'18 distinct new outputs required')
        for ref in new_refs:
            name=str(Path(ref['path']).resolve());d.p.require(name in self.new_paths,'unlisted new archive')
            self.refs[name]={**ref,'kind':'newhead'}
        d.p.require(len(self.refs)==(43 if jobs.audit else 25),'new archive audit manifest differs')
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
            if self.refs[name]['kind']=='newhead':d.p.require(set(z.files)==set(d.finer.NEW_HEAD_MEMBERS),'new archive has undeclared fields')
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
            d.p.require(not args and set(arrays)==set(d.finer.NEW_HEAD_MEMBERS),'new head fields differ')
            d.p.require(hasattr(self.jobs,'completed_pair'),'new head save before both paths complete')
            k,points,updates=self.jobs.completed_pair
            d.p.require(tuple(digest(v) for v in points)==self.jobs.completed_pair_hashes,'completed pair mutated before save')
            plan=self.cfg['inputs']['folds'][k[2]]; source=self.jobs.sources[k]
            d.p.require(k==self.new_paths[name] and arrays['fit_rows'].dtype==arrays['fit_demo_indices'].dtype==np.int32
                and arrays['fit_rows'].tolist()==plan['fit_rows'] and arrays['fit_demo_indices'].tolist()==plan['fit_demos']
                and all(arrays[f].dtype==np.float64 and arrays[f].shape==(256,) and digest(arrays[f])==source[f] for f in ('scaler_mean','scaler_scale')),
                'new head fit/scalers do not match source')
            d.p.require(arrays['weights'].dtype==arrays['bias'].dtype==np.float64 and arrays['weights'].shape==(3,256)
                and arrays['bias'].shape==(3,) and np.array_equal(np.c_[arrays['weights'],arrays['bias']],np.stack(points))
                and arrays['additional_updates'].dtype==np.int32 and arrays['additional_updates'].tolist()==[0,*updates]
                and all(isinstance(v,np.ndarray) and v.dtype.kind in 'biuf' for v in arrays.values()),'new paired endpoint schema differs')
            self.save_depth+=1
            try:save0(name,**arrays)
            finally:self.save_depth-=1
            self.writes.append(name);pin={'path':name,'sha256':self.hash_one(name)};self.limits.check();return pin
        def zip_open(z,field,mode='r',*args,**kw):
            name=self.zip_registry.get(id(z));member=field.filename if isinstance(field,zipfile.ZipInfo) else field
            if mode=='w':
                target=str(Path(z.filename).resolve()) if isinstance(z.filename,(str,Path)) else None
                entry=(target,str(member).removesuffix('.npy'))
                if self.jobs.audit or not self.save_depth or target not in self.new_paths or entry[1] not in d.finer.NEW_HEAD_MEMBERS or entry in self.write_members:return self.reject('zip_write_bypass',member)
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
        d.p.require(set(self.completed)==set(self.refs) and len(self.attempts)==len(self.completed)==(43 if self.jobs.audit else 25)
            and set(self.member_completed)==expected and len(self.member_attempts)==len(self.member_completed)==(287 if self.jobs.audit else 161)
            and len(self.writes)==(0 if self.jobs.audit else 18) and len(self.write_members)==(0 if self.jobs.audit else 126) and not self.denied,'archive IO budget differs')
        for name,ref in self.refs.items():self.hash_one(name,ref['sha256'])
        d.p.require(len(self.hash_checks)==(86 if self.jobs.audit else 68),'archive SHA budget differs');self.limits.check()

    def record(self):
        return {'NPZ_loads':{'attempted':len(self.attempts),'completed':len(self.completed)},
            'member_decodes':{'attempted':len(self.member_attempts),'completed':len(self.member_completed)},
            'files':self.completed,'members':list(self.member_completed),'new_NPZ_writes':self.writes,'new_member_writes':self.write_members,'byte_SHA_checks':self.hash_checks,
            'denied_operations':self.denied,'input_research_CSV_reads':0,'original_Policy_CNN_environment_or_sealed_calls':0}
