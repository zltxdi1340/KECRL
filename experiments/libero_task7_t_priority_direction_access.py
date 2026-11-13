"""Bound T-priority source access, exact directions and separate U/S budgets."""
from __future__ import annotations
import builtins,copy,hashlib,io,json,math,os,sys,zipfile
from pathlib import Path
import numpy as np
from experiments import libero_task7_t_priority_direction_protocol as d
from experiments import libero_task7_fixed_representation_optimization_access as legacy
from experiments import libero_task7_joint_direction_access as old
from experiments import libero_task7_joint_direction_prefix_review_access as bounded
base,identity,key,digest,readonly=old.base,old.identity,old.key,old.digest,old.readonly
RepresentationWork=old.RepresentationWork
finer=d.parent.finer
rules=finer.rules
SOURCE_ARMS=['A_original','B_fomaml']
SEEDS=[20261106,20261107,20261108]
VECTORS=['base_gradient','penalty_gradient','total_gradient','direction']
require=d.p.require

class History:
    """One pinned JSON parse and scalar-only projection; hashes before/after."""
    def __init__(self,cfg,audit=False):
        self.cfg=cfg;self.spec=cfg['historical_control'];self.ref=self.spec['audit_source_JSON' if audit else 'primary_source_JSON']
        self.hashes=0;self.checks=0;self.arm=self.spec['optimization_arm'];self.data={};self.counts={}
        path=Path(self.ref['path']).resolve();saved=[]
        def check(p,mode):
            require(isinstance(p,(str,Path)) and Path(p).resolve()==path and not any(c in mode for c in 'wax+'),'unlisted historical JSON IO')
        open0,io0,path0=builtins.open,io.open,Path.open
        def op(p,mode='r',*a,**kw):check(p,mode);return open0(p,mode,*a,**kw)
        def ip(p,mode='r',*a,**kw):check(p,mode);return io0(p,mode,*a,**kw)
        def pp(p,mode='r',*a,**kw):check(p,mode);return path0(p,mode,*a,**kw)
        try:
            builtins.open,io.open,Path.open=op,ip,pp
            require(d.p.sha(path)==self.ref['sha256'],'historical source JSON changed');self.hashes+=1
            raw=json.loads(path.read_text())
        finally:builtins.open,io.open,Path.open=open0,io0,path0
        require(raw['science_config']==self.spec['source_science'],'historical science differs')
        for table,fields in self.spec['projections'].items():
            chosen=[]
            for row in raw[table]:
                if table=='endpoints':
                    if row['condition'] not in self.spec['endpoint_conditions']:continue
                    selected=self.spec['saved500_endpoint_projection'] if row['condition']=='saved500' else fields
                else:
                    if row['optimization_arm']!=self.arm:continue
                    selected=fields
                item={f:row[f] for f in selected}
                require(all(v is None or type(v) in (str,bool,int,float) and (type(v) is not float or math.isfinite(v)) for v in item.values()),'historical nonscalar/nonfinite projection')
                chosen.append(item)
            self.data[table]=chosen;self.counts[table]=0
        del raw
        self.expected=cfg['budget']['historical_control_JSON_projected_records_per_stage']
        require({k:len(v) for k,v in self.data.items()}==self.expected,'historical scalar coverage differs')
    def row(self,table,actual):
        k=self.counts[table];require(k<len(self.data[table]),'extra historical row')
        for field,value in self.data[table][k].items():
            got=actual[field]
            if type(value) is float:require(type(got) in (int,float) and math.isfinite(got) and abs(got-value)<=1e-10,'historical '+table+'.'+field+' differs')
            else:require(type(got) is type(value) and got==value,'historical '+table+'.'+field+' differs')
            self.checks+=1
        self.counts[table]+=1
    def complete(self):
        require(self.counts==self.expected,'historical path reproduction incomplete')
        require(self.hashes==1,'repeated historical completion')
        require(d.p.sha(self.ref['path'])==self.ref['sha256'],'historical JSON changed during run');self.hashes+=1
        require(self.hashes==2,'historical SHA budget differs')
        return dict(projected_records=self.counts,scalar_checks=self.checks,JSON_parses=1,JSON_SHA_checks=self.hashes,
                    all18_paths_reproduced=True,source=self.ref,unselected_vectors_or_parameters_used=False)
def members(kind):return d.parent.NEW_HEAD_MEMBERS if kind=='newhead' else finer.HEAD_MEMBERS if kind=='head' else finer.SOURCE_MEMBERS[kind]
def load(ref,kind):
 with np.load(ref['path'],allow_pickle=False) as z:return {f:z[f] for f in members(kind)}

def preflight(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
 b=d.require_binding(config,binding);return json.loads(Path(config).read_text()),b
def validate_call(cfg,sp,bp):
 actual,b=preflight(Path(sp['path']),Path(bp['path']))
 require(cfg==actual and d.p.pin(Path(sp['path']))==sp and d.p.pin(Path(bp['path']))==bp,'T-priority bound API required')

def runtime(cfg,audit=False):
 result=old.runtime(cfg,audit=audit)
 result['numpy_backend']={k:np.__config__.get_info(k) for k in ('blas_ilp64_opt_info','lapack_ilp64_opt_info')}
 if audit:
  import torch
  result.update(torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),torch_backend=torch.__config__.show())
 return result

class Limits(bounded.Limits):pass

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
        d.p.require(self.active and self.local['dot_reductions']<(2 if self.arm==d.ARMS[0] else 3) and len(left)==len(right)==257,'extra direction dot')
        self.local['dot_reductions']+=1;self.counts['dot_reductions']+=1
        value=backend(left,right) if backend else math.fsum(x*y for x,y in zip(left,right))
        d.finite(value);return value

    def vector(self,values):
        d.p.require(self.active and self.local['vector_forms']<4 and len(values)==257,'extra direction vector')
        d.finite(*values);self.local['vector_forms']+=1;self.counts['vector_forms']+=1;self.counts['coordinate_writes']+=257
        return values

    def end(self,meta):
        d.p.require(self.active and self.local['dot_reductions'] in (2,3) and self.local['vector_forms'] in (1,3,4)
            and self.local['norm_reductions'] in (3,4),'incomplete direction work')
        d.p.require(meta['status'] in d.STATUSES,
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
        d.p.require(c['constructions']==self.jobs.counts['gradient_states'] and 2*c['constructions']<=c['dot_reductions']<=3*c['constructions']
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


def stop(arm,k,row,rejected=0):
    return d.stopping_reason(arm,k,row['total_l2'],row['direction_status'],row['selected_joint'],row['directional_derivative'],row['exact_MSE_slope'],rejected)

def direction_fields(meta):
    return dict(direction_status=meta['status'],direction_construction_status=meta['construction_status'],direction_total_reference_norm=meta['total_norm'],
      direction_base_reference_norm=meta['base_norm'],direction_unit_sum_norm=meta['unit_sum_norm'],direction_unit_cosine=meta['unit_cosine'],
      direction_projection_norm=meta['projection_norm'],chosen_direction_norm=meta['direction_norm'],selected_T_slope=meta['selected_T_slope'],
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
            and all(self.array_sources.get(id(head[f]))==(name,f) for f in finer.HEAD_MEMBERS),'unregistered source500 fields')
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
            summary=finer.validate_round('fine20',self.last['trainable_objective'],self.last['directional_derivative'],self.fixed_mse,self.round_rows)
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


class TraceJobs(Jobs):
 def __init__(self,cfg,limits,audit=False):
  super().__init__(cfg,limits,audit);self.traces={};self.trace_coordinate_copies=0
 def state_end(self,row):
  super().state_end(row)
  spec=self.active[:-1];record=(d.ARMS.index(self.active[-1]),self.updates,self.current.copy(),*[np.asarray(row[f],np.float64) for f in VECTORS])
  self.traces.setdefault(spec,[]).append(record);self.trace_coordinate_copies+=5*257
  require(self.trace_coordinate_copies<=self.cfg['budget']['primary_trace_coordinate_copies_max'],'trace copy budget exceeded')

def trace_arrays(records,prefix='trace'):
 require(records and len(records)<=102,'nonempty bounded two-path trace required')
 return {prefix+'_arm_indices':np.asarray([r[0] for r in records],np.int32),prefix+'_extra_updates':np.asarray([r[1] for r in records],np.int32),
  **{prefix+'_'+name:np.stack([r[j+2] for r in records]) for j,name in enumerate(['parameters',*VECTORS])}}

def validate_trace_arrays(arrays,jobs):
 worker=jobs.u if hasattr(jobs,'u') else jobs
 spec=worker.completed_pair[0]
 expected=trace_arrays(worker.traces[spec])
 if hasattr(jobs,'local'):expected.update(trace_arrays(jobs.local.traces[spec],'local'))
 for f,v in expected.items():require(arrays[f].dtype==v.dtype and arrays[f].shape==v.shape and arrays[f].tobytes()==v.tobytes(),'trace identity/coordinate changed: '+f)

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
        self.read_outputs={str(Path(path).resolve()) for path in table_paths};self.write_outputs=set();self.audit_paths={str((Path(cfg['artifacts'])/f'audit-head-{i:03d}.npz').resolve()):key(h) for i,h in enumerate(cfg['inputs']['heads'])};self.telemetry_path=str(Path('/proc/self/status').resolve())
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
            if self.refs[name]['kind']=='newhead':d.p.require(set(z.files)==set(d.parent.NEW_HEAD_MEMBERS),'new archive has undeclared fields')
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
            if name not in (self.audit_paths if self.jobs.audit else self.new_paths) or name in self.writes or Path(name).exists():return self.reject('unlisted_or_repeated_save',name)
            d.p.require(not args and set(arrays)==set(d.parent.AUDIT_HEAD_MEMBERS if self.jobs.audit else d.parent.NEW_HEAD_MEMBERS),'new head fields differ')
            d.p.require(hasattr(self.jobs,'completed_pair'),'new head save before both paths complete')
            k,points,updates=self.jobs.completed_pair
            d.p.require(tuple(digest(v) for v in points)==self.jobs.completed_pair_hashes,'completed pair mutated before save')
            plan=self.cfg['inputs']['folds'][k[2]]; source=self.jobs.sources[k]
            d.p.require(k==(self.audit_paths if self.jobs.audit else self.new_paths)[name] and arrays['fit_rows'].dtype==arrays['fit_demo_indices'].dtype==np.int32
                and arrays['fit_rows'].tolist()==plan['fit_rows'] and arrays['fit_demo_indices'].tolist()==plan['fit_demos']
                and all(arrays[f].dtype==np.float64 and arrays[f].shape==(256,) and digest(arrays[f])==source[f] for f in ('scaler_mean','scaler_scale')),
                'new head fit/scalers do not match source')
            d.p.require(arrays['weights'].dtype==arrays['bias'].dtype==np.float64 and arrays['weights'].shape==(3,256)
                and arrays['bias'].shape==(3,) and np.array_equal(np.c_[arrays['weights'],arrays['bias']],np.stack(points))
                and arrays['additional_updates'].dtype==np.int32 and arrays['additional_updates'].tolist()==[0,*updates]
                and all(isinstance(v,np.ndarray) and v.dtype.kind in 'biuf' for v in arrays.values()),'new paired endpoint schema differs')
            validate_trace_arrays(arrays,self.jobs)
            self.save_depth+=1
            try:save0(name,**arrays)
            finally:self.save_depth-=1
            self.writes.append(name);pin={'path':name,'sha256':self.hash_one(name)};self.limits.check();return pin
        def zip_open(z,field,mode='r',*args,**kw):
            name=self.zip_registry.get(id(z));member=field.filename if isinstance(field,zipfile.ZipInfo) else field
            if mode=='w':
                target=str(Path(z.filename).resolve()) if isinstance(z.filename,(str,Path)) else None
                entry=(target,str(member).removesuffix('.npy'))
                if not self.save_depth or target not in (self.audit_paths if self.jobs.audit else self.new_paths) or entry[1] not in (d.parent.AUDIT_HEAD_MEMBERS if self.jobs.audit else d.parent.NEW_HEAD_MEMBERS) or entry in self.write_members:return self.reject('zip_write_bypass',member)
                self.write_members.append(entry);return zip0(z,field,mode,*args,**kw)
            if name is not None and (not self.member_depth or self.active_member!=(name,str(member).removesuffix('.npy'))):return self.reject('zip_bypass',member)
            return zip0(z,field,mode,*args,**kw)
        def check(path,mode):
            if not isinstance(path,(str,Path)):return self.reject('raw_descriptor')
            name=str(Path(path).resolve());write=any(c in mode for c in 'wax+')
            if name in (self.audit_paths if self.jobs.audit else self.new_paths):
                if write and self.save_depth:return
                if not write and self.hash_depth:return
                return self.reject('raw_new_head',name)
            if name in self.refs:
                if write:return self.reject('source_write')
                if not (self.load_depth or self.hash_depth):return self.reject('raw_archive')
            elif name==self.telemetry_path and not write:return
            elif name in self.read_outputs or name in self.write_outputs:
                if write and name not in self.write_outputs and self.jobs.audit:return self.reject('audit_table_write')
                if not write and name in self.write_outputs:return self.reject('read_own_audit_output')
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
                self.jobs.linear(len(x));result=linear0(x,w,b)
                if getattr(self.jobs,'local_active',False):self.jobs.local.kernel_result(result,w,b)
                return result
            self.patch(torch.autograd,'grad',grad);self.patch(torch.nn.functional,'linear',linear)
        return self

    def complete(self):
        expected={(name,k) for name,ref in self.refs.items() for k in members(ref['kind'])}
        d.p.require(set(self.completed)==set(self.refs) and len(self.attempts)==len(self.completed)==(43 if self.jobs.audit else 25)
            and set(self.member_completed)==expected and len(self.member_attempts)==len(self.member_completed)==(413 if self.jobs.audit else 161)
            and len(self.writes)==18 and len(self.write_members)==(378 if self.jobs.audit else 252) and not self.denied,'archive IO budget differs')
        for name,ref in self.refs.items():self.hash_one(name,ref['sha256'])
        d.p.require(len(self.hash_checks)==(104 if self.jobs.audit else 68),'archive SHA budget differs');self.limits.check()

    def record(self):
        return {'NPZ_loads':{'attempted':len(self.attempts),'completed':len(self.completed)},
            'member_decodes':{'attempted':len(self.member_attempts),'completed':len(self.member_completed)},
            'files':self.completed,'members':list(self.member_completed),'new_NPZ_writes':self.writes,'new_member_writes':self.write_members,'byte_SHA_checks':self.hash_checks,
            'denied_operations':self.denied,'input_research_CSV_reads':0,'original_Policy_CNN_environment_or_sealed_calls':0}


class ShadowJobs:
    """Local Torch readouts never own an optimizer or an accepted parameter."""
    audit=True
    def __init__(self,cfg,limits):
        self.cfg,self.limits=cfg,limits;self.direction_work=DirectionWork(self);self.phase=None;self.active=None;self.updates=0;self.n=None
        self.traces={};self.done=set();self.counts=dict(gradient_states=0,gradient_vectors=0,autograd_calls=0,linear_forms=0,linear_form_rows=0,trial_forwards=0,
            state_fit_rows=0,trial_fit_rows=0,fixed_tangents=0,vector_maxabs_checks=0,vector_coordinates=0,parameter_coordinate_checks=0)
        self.latest={};self.last_actual=None
    def register(self,spec,context,rows):
        require(self.phase is None,'unfinished local role');self.spec=spec;self.context=context;self.rows=rows;self.n=len(rows)
        require(self.n in (2782,2817,2777) and all(v.dtype==np.float64 and not v.flags.writeable for v in context),'local fit must reuse registered own-fit context')
    def begin(self,opt,k,theta,expected_direction):
        require(self.phase is None and (self.spec,opt,k) not in self.done and opt in d.ARMS and type(k) is int and 0<=k<=50,'repeated or invalid local state')
        self.active=self.spec+(opt,);self.updates=k;self.current=readonly(theta);self.P_direction=readonly(expected_direction);self.next_trial=0
        self.latest=dict(path=list(self.active),step=k,parameter_digest=digest(theta));self.last_actual=None
    def check_context(self,*values):require(all(x is y for x,y in zip(values,self.context)),'foreign local context')
    def state_begin(self,theta,params=()):
        require(self.phase is None and (self.spec,self.active[-1],self.updates) not in self.done
                and np.array_equal(theta,self.current) and len(params)==2,'foreign or repeated local parameter')
        require(np.array_equal(np.r_[params[0].detach().numpy(),params[1].detach().numpy()],theta)
                and all(str(v.dtype)=='torch.float64' and v.device.type=='cpu' and v.requires_grad for v in params),'local Float64 parameter mismatch')
        self.phase='state';self.params=tuple(params);self.parameter_hash=digest(theta);self.forms=self.grads=0
    def linear(self,n):
        require(n==self.n and self.phase in ('state','fixed_tangent','trial'),'local form outside declared job')
        require(self.forms<(2 if self.phase=='state' else 3 if self.phase=='fixed_tangent' else 1),'extra local form')
        self.forms+=1;self.counts['linear_forms']+=1;self.counts['linear_form_rows']+=n;self.check()
    def kernel_result(self,result,w,b):
        if self.phase=='state' and self.forms==1:self.raw_linear=result.detach().squeeze(1)
        if self.phase in ('fixed_tangent','trial'):
            expected=self.P_direction if self.phase=='fixed_tangent' else self.trial_theta
            require(np.array_equal(np.r_[w.detach().numpy().ravel(),b.detach().numpy()],expected),'foreign local tangent or trial parameters')
    def gradient(self,params=()):
        require(self.phase=='state' and self.forms==1 and self.grads<3 and tuple(map(id,params))==tuple(map(id,self.params)),'extra or foreign local autograd')
        self.grads+=1;self.counts['gradient_vectors']+=1;self.counts['autograd_calls']+=1;self.check()
    def state_end(self,row):
        self.last_actual=row
        require(self.phase=='state' and self.forms==2 and self.grads==3 and digest(self.current)==self.parameter_hash
                and digest(np.r_[self.params[0].detach().numpy(),self.params[1].detach().numpy()])==self.parameter_hash,'local state incomplete or mutated')
        self.last=row;self.phase='fixed_tangent';self.counts['gradient_states']+=1;self.counts['state_fit_rows']+=self.n
        self.traces.setdefault(self.spec,[]).append((d.ARMS.index(self.active[-1]),self.updates,self.current.copy(),*[np.asarray(row[f],np.float64) for f in VECTORS]))
    def finish_tangent(self):
        require(self.phase=='fixed_tangent' and self.forms==3 and self.grads==3,'missing fixed P tangent')
        self.phase=None;self.params=();self.done.add((self.spec,self.active[-1],self.updates));self.counts['fixed_tangents']+=1;self.check()
    def trial_begin(self,j,theta):
        require(self.phase is None and self.active is not None and (self.spec,self.active[-1],self.updates) in self.done
                and type(j) is int and j==self.next_trial and 0<=j<20 and self.updates<50,'unbudgeted shadow trial')
        require(np.array_equal(theta,self.current+2.**(-j)*self.P_direction),'shadow trial not exact P parameters')
        self.phase='trial';self.forms=0;self.trial_theta=readonly(theta)
    def trial_end(self):
        require(self.phase=='trial' and self.forms==1,'shadow trial incomplete');self.phase=None
        self.counts['trial_forwards']+=1;self.counts['trial_fit_rows']+=self.n;self.next_trial+=1;self.check()
    def check(self):
        b=self.cfg['budget'];c=self.counts
        for field,cap in [('gradient_states','local_S_gradient_states_max'),('autograd_calls','local_S_gradient_vectors_max'),
          ('trial_forwards','local_shadow_trial_evaluations_max'),('linear_forms','local_S_linear_forms_max'),('linear_form_rows','local_S_linear_rows_max'),
          ('vector_maxabs_checks','local_gradient_vector_comparisons_max'),('vector_coordinates','local_vector_coordinate_checks_max'),
          ('parameter_coordinate_checks','P_path_parameter_coordinate_checks_max')]:
            require(c[field]<=b[cap],'local budget exceeded: '+field)
        self.limits.check()
    def complete(self,Pstates,Ptrials):
        c=self.counts;require(c['gradient_states']==len(Pstates) and c['gradient_vectors']==c['autograd_calls']==3*len(Pstates)
            and c['fixed_tangents']==len(Pstates) and c['trial_forwards']==len(Ptrials)
            and c['linear_forms']==3*len(Pstates)+len(Ptrials) and c['vector_maxabs_checks']==4*len(Pstates)
            and c['vector_coordinates']==4*257*len(Pstates),'local complete ledger differs')
        self.direction_work.complete();self.check()


class Dispatch:
    """Archive guard routes every actual Torch operation to one declared role."""
    audit=True
    def __init__(self,u,local):self.u=u;self.local=local;self.worker=u;self.local_active=False
    def use_local(self,value):self.local_active=value;self.worker=self.local if value else self.u
    @property
    def array_sources(self):return self.u.array_sources
    def __getattr__(self,name):return getattr(self.worker,name)
    def gradient(self,params=()):self.worker.gradient(params);self.check()
    def linear(self,n):self.worker.linear(n);self.check()
    def check(self):
        b=self.u.cfg['budget'];s=self.local.counts;u=self.u.counts
        require(u['autograd_calls']+s['autograd_calls']<=b['audit_total_autograd_max']
            and u['linear_forms']+s['linear_forms']<=b['audit_total_linear_forms_max']
            and u['linear_form_rows']+s['linear_form_rows']<=b['audit_total_linear_rows_max'],'combined audit budget exceeded')


def fixed_P_slope(backend,job):
    import torch
    require(job.phase=='fixed_tangent' and job.forms==2,'fixed tangent must follow local measure')
    raw=backend.p0[:,2]+job.raw_linear;pred=raw.clamp(-1,1)
    difference=(pred-backend.y[:,2]).square()-(backend.p0[:,2]-backend.y[:,2]).square()
    tangent=torch.nn.functional.linear(backend.x,torch.tensor(job.P_direction[:-1][None]),torch.tensor(job.P_direction[-1:])).squeeze(1)
    dp=torch.where((raw>-1)&(raw<1),tangent,torch.zeros_like(tangent));dp=torch.where(raw==-1,torch.clamp_min(tangent,0),dp);dp=torch.where(raw==1,torch.clamp_max(tangent,0),dp)
    de=2*(pred-backend.y[:,2])*dp;dh=torch.where(difference>0,de,torch.where(difference==0,torch.clamp_min(de,0),torch.zeros_like(de)))
    values={'exact_MSE_slope':de.mean().item(),'exact_penalty_slope':(dh.mean()/7).item(),'directional_derivative':((de+dh).mean()/7).item()}
    require(all(math.isfinite(v) for v in values.values()),'nonfinite fixed P slope');job.finish_tangent();return values


def local_numeric_gate(primary,point,job,checks):
    observed={'direction_status','direction_construction_status','selected_joint'};maximum=0.
    job.latest.update(P=primary,S=point)
    for field,wanted in primary.items():
        if field in d.IDS+['optimization_arm','extra_updates','parameter_digest'] or field in observed:continue
        actual=point[field]
        if type(wanted) is float:
            require(type(actual) in (int,float) and math.isfinite(actual),'nonfinite local scalar')
            error=abs(actual-wanted);maximum=max(maximum,error);job.latest.update(field=field,expected=wanted,actual=actual,error=error)
            require(error<=1e-10,'same-parameter scalar mismatch: '+field);checks.n+=1
        else:checks.equal(actual,wanted,'same-parameter '+field)
    return maximum


def fail_record(path,error,jobs,local,guard,limits,checks=None):
    worker=jobs.u if hasattr(jobs,'u') else jobs
    record=dict(status='failed_no_retry',error=repr(error),computation=dict(worker.counts),direction_work=dict(worker.direction_work.counts),
     access=guard.record(),cost=limits.snapshot(),trace_coordinate_copies=worker.trace_coordinate_copies,
     local_computation=dict(local.counts) if local else None,local_direction_work=dict(local.direction_work.counts) if local else None,
     latest=local.latest if local else None,actual_local=local.last_actual if local else None,scalar_checks=checks.n if checks else None)
    if local is None:
        record['last_own_actual']=getattr(worker,'last',None)
        record['last_direction_record']=getattr(worker,'direction_record',None)
    raw=json.dumps(record,allow_nan=False);require(len(raw.encode())<=1048576,'failure record exceeds bound')
    with Path(path).open('x') as f:f.write(raw+'\n')


Sink=bounded.Sink
class NullSink:
    def __init__(self):self.rows=0
    def add(self,row):self.rows+=1
