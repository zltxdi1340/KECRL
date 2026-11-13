"""Single-head prefix IO, role and arithmetic budgets; stdlib until numeric entry."""
from __future__ import annotations
import builtins
import copy
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile
from experiments import libero_task7_joint_direction_prefix_review_protocol as d
from experiments import libero_task7_component3_history_transition_access as base

require=d.p.require
VECTORS=['base_gradient','penalty_gradient','total_gradient','direction']
IDENT={**d.FOCUS,'optimization_arm':d.OPTIMIZATION_ARM}
ERROR_EDGES={'theta_U_P':('theta_P','theta_U'),
 **{g+'_'+name:(g+'_'+left,g+'_'+right) for g in ('Gb','Gp','GT') for name,left,right in [('S_P','P','S'),('U_S','S','U'),('U_P','P','U')]},
 'd_N_P':('d_P','d_N'),'d_S_N':('d_N','d_S'),'d_U_S':('d_S','d_U'),'d_U_P':('d_P','d_U')}


def preflight(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    return d.validate(config),d.require_binding(config,binding)


def validate_call(cfg,sp,bp):
    frozen,bound=preflight(Path(sp['path']),Path(bp['path']))
    require(cfg==frozen and d.p.pin(Path(sp['path']))==sp and d.p.pin(Path(bp['path']))==bp,'unchanged bound API required')
    return bound


def runtime(cfg,stage):
    require(platform.python_version()==cfg['execution']['runtime']['python'],'Python version changed')
    require(all(os.environ.get(k)==v for k,v in cfg['execution']['environment'].items()),'CPU environment changed')
    versions={k:importlib.metadata.version(k) for k in ('numpy','torch')}
    require(all(versions[k]==cfg['execution']['runtime'][k] for k in versions),'numeric runtime changed')
    out=dict(python=platform.python_version(),**versions,pid=os.getpid(),device='CPU',stage=stage,
             environment={k:os.environ.get(k) for k in cfg['execution']['environment']},
             platform=platform.platform(),machine=platform.machine(),CPU=platform.processor(),
             environment_seed=None,initial_state_indices=[])
    if stage!='audit':
        import numpy as np
        out['numpy_backend']={k:np.__config__.get_info(k) for k in ('blas_ilp64_opt_info','openblas64__info','lapack_ilp64_opt_info')}
        if stage=='cross':
            import torch
            torch.set_num_threads(1)
            try:torch.set_num_interop_threads(1)
            except RuntimeError:require(torch.get_num_interop_threads()==1,'interop threads changed')
            torch.use_deterministic_algorithms(True)
            out.update(torch_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),torch_backend=torch.__config__.show())
        else:require('torch' not in sys.modules,'primary must be NumPy only')
    else:require('numpy' not in sys.modules and 'torch' not in sys.modules,'arithmetic auditor must remain stdlib')
    return out


def digest(values):
    import struct
    flat=values.ravel().tolist() if hasattr(values,'ravel') else values
    return hashlib.sha256(b''.join(struct.pack('<d',float(x)) for x in flat)).hexdigest()


def readonly(array):
    array.flags.writeable=False
    return array


class Limits(base.Limits):
    def snapshot(self):
        return dict(elapsed_seconds=time.monotonic()-self.started,peak_RSS_bytes=self.peak_RSS_bytes,
                    active_seconds_max=self.seconds,memory_bytes_max=self.memory_bytes)


class Sink:
    def __init__(self,path,columns,cap):
        self.path=Path(path);self.columns=columns;self.cap=cap;self.rows=0
        self.handle=self.path.open('x',newline='');self.writer=csv.DictWriter(self.handle,fieldnames=columns);self.writer.writeheader()
    def add(self,row):
        require(set(row)==set(self.columns) and self.rows<self.cap,'CSV schema/row budget exceeded')
        self.writer.writerow(row);self.rows+=1
    def close(self):self.handle.close()
    def pin(self):
        require(self.handle.closed,'close before CSV digest')
        return dict(d.p.pin(self.path),rows=self.rows,columns=self.columns)


class Events:
    def __init__(self,path,stage,limits,cap):
        self.path=Path(path);self.stage=stage;self.limits=limits;self.cap=cap;self.rows=0
        self.handle=self.path.open('x')
    def add(self,role,step,event,jobs,theta=None,meta=None,error=None):
        require(self.rows<self.cap,'event budget exceeded')
        row=dict(stage=self.stage,role=role,step=step,event=event,status=None,
                 parameter_digest=digest(theta) if theta is not None else None,Gb_digest=None,GT_digest=None,
                 q_value=None,q_hex=None,counters=jobs.record(),last_RSS_bytes=self.limits.peak_RSS_bytes,
                 elapsed_seconds=time.monotonic()-self.limits.started,error=error)
        if role=='N':row.update(Gb_digest=jobs.last_direction_inputs[1],GT_digest=jobs.last_direction_inputs[0])
        if meta:
            q=meta.get('unit_sum_norm');row.update(status=meta.get('status'),q_value=q,q_hex=float(q).hex() if q is not None else None,
                 Gb_digest=jobs.last_direction_inputs[1],GT_digest=jobs.last_direction_inputs[0])
        require(set(row)==set(d.EVENT_COLUMNS),'event schema changed')
        self.handle.write(json.dumps(row,allow_nan=False)+'\n');self.handle.flush();self.rows+=1
    def close(self):self.handle.close()
    def pin(self):return dict(d.p.pin(self.path),rows=self.rows,columns=d.EVENT_COLUMNS)


class DirectionWork:
    def __init__(self,jobs):
        from experiments.libero_task7_joint_direction_access import DirectionWork as Legacy
        self.inner=Legacy();self.owner=jobs
    @property
    def counts(self):return self.inner.counts
    def begin(self,arm,total,base):
        j=self.owner
        require(j.phase in ('state','normalization') and arm==d.OPTIMIZATION_ARM and
                (j.phase=='normalization' or (j.forms==1 and j.grads==3)),'direction outside registered role')
        require((j.role,j.updates) not in j.direction_done,'repeated direction point')
        self.inner.begin(arm,total,base);j.last_direction_inputs=(digest(total),digest(base))
    def norm(self,*args):return self.inner.norm(*args)
    def dot(self,*args):return self.inner.dot(*args)
    def vector(self,*args):return self.inner.vector(*args)
    def end(self,meta):
        j=self.owner;j.last_direction=copy.deepcopy(meta)  # retain actual q before any validation
        self.inner.end(meta);j.direction_done.add((j.role,j.updates));j.check_caps()


class Jobs:
    def __init__(self,cfg,stage,limits):
        self.cfg,self.stage,self.limits=cfg,stage,limits;self.cap=cfg['budget'][stage]
        self.context=None;self.phase=None;self.role=None;self.active=None;self.array_sources={}
        self.direction_done=set();self.direction_work=DirectionWork(self);self.last_direction=None;self.last_direction_inputs=(None,None)
        self.counts=dict(gradient_states=0,gradient_vectors=0,gradient_coordinates=0,autograd_calls=0,
                         updates=0,parameter_coordinate_writes=0,trials=0,linear_forms=0,linear_form_rows=0,
                         state_fit_rows=0,state_seven_action_coordinates=0,trial_fit_rows=0,
                         scaler_reconstructions=0,standardization_feature_coordinates=0)
        self.measured={r:[] for r in d.ROLES};self.finished_U=False;self.next_trial=0;self.pending=None
    def register(self,context,start):
        import numpy as np
        n=len(self.cfg['fold']['fit_rows'])
        require(n==d.N and all(v.dtype==np.float64 and not v.flags.writeable and np.isfinite(v).all() for v in context)
                and [v.shape for v in context]==[(n,256),(n,7),(n,7)] and start.dtype==np.float64 and start.shape==(257,), 'invalid own-fit context')
        self.context=context;self.context_digests=[digest(v) for v in context];self.start=readonly(start.copy())
        self.counts['scaler_reconstructions']=1;self.counts['standardization_feature_coordinates']=n*256
    def check_context(self,*context):require(self.context is not None and all(x is y for x,y in zip(context,self.context)),'unregistered fit context')
    def select(self,role,k,theta):
        import numpy as np
        require(self.phase is None and type(k) is int and 0<=k<=35,'point outside prefix')
        if self.stage=='primary':require(role=='P','wrong primary role')
        else:require(role in ('U','S','N') and (role=='U' or self.finished_U),'cross jobs before U completed')
        require(k==len(self.measured[role]),'skipped/repeated role step')
        if role in ('P','U'):
            expected=self.start if k==0 else self.pending
            require(expected is not None and np.array_equal(theta,expected),'free path cross-fed or unaccepted update')
        elif role=='S':require(digest(theta)==digest(self.P_trace['parameters'][k]),'S parameters differ from P')
        else:require(theta is None,'N must not use parameters')
        self.role=role;self.active=(role,d.OPTIMIZATION_ARM);self.updates=k
        self.current=theta;self.params=();self.forms=self.grads=self.kernels=0;self.next_trial=0;self.pending=None
    def state_begin(self,theta,params=()):
        import numpy as np
        require(self.role in ('P','U','S') and self.phase is None and np.array_equal(theta,self.current),'unregistered gradient state')
        self.phase='state';self.forms=self.grads=self.kernels=0;self.params=tuple(params);self.theta_digest=digest(theta)
        if self.stage=='cross':
            require(len(params)==2 and [v.numel() for v in params]==[256,1]
                    and all(v.device.type=='cpu' and str(v.dtype)=='torch.float64' and v.requires_grad for v in params)
                    and np.array_equal(np.r_[params[0].detach().numpy(),params[1].detach().numpy()],theta),'Torch parameters differ')
        self.counts['gradient_states']+=1;self.counts['state_fit_rows']+=d.N;self.counts['state_seven_action_coordinates']+=d.N*7;self.check_caps()
    def linear(self,n):
        require(n==d.N and self.role!='N' and self.phase in ('state','trial') and self.forms<(2 if self.phase=='state' else 1),'unbudgeted linear form')
        self.forms+=1;self.counts['linear_forms']+=1;self.counts['linear_form_rows']+=n;self.check_caps()
    def gradient(self,params=()):
        require(self.phase=='state' and self.forms==1 and self.grads<3,'unbudgeted gradient')
        if self.stage=='cross':require(tuple(map(id,params))==tuple(map(id,self.params)),'unregistered autograd parameters')
        self.grads+=1;self.counts['gradient_vectors']+=1;self.counts['gradient_coordinates']+=257
        self.counts['autograd_calls']+=int(self.stage=='cross');self.check_caps()
    def state_end(self,row):
        require(self.phase=='state' and self.forms==self.kernels==2 and self.grads==3
                and digest(self.current)==self.theta_digest and (self.role,self.updates) in self.direction_done,'incomplete state')
        require(all(math.isfinite(v) for f in VECTORS for v in row[f]),'nonfinite recorded vectors')
        if self.params:
            require(digest([*self.params[0].detach().tolist(),*self.params[1].detach().tolist()])==self.theta_digest,'autograd mutated parameters')
        self.last=row;self.phase=None;self.params=();self.measured[self.role].append(self.updates)
        if self.updates==0 and self.role in ('P','U'):self.fixed_mse=row['component3_mse']
        if self.role in ('P','U'):require(row['component3_mse']<=self.fixed_mse,'free path violated fixed M500')
    def normalize(self,k,gb,gt):
        self.select('N',k,None)
        require(digest(gb)==digest(self.P_trace['base_gradient'][k]) and digest(gt)==digest(self.P_trace['total_gradient'][k]),'N received wrong gradients')
        self.last_direction_inputs=(digest(gt),digest(gb))
        self.phase='normalization'
    def normalized(self):
        require(self.phase=='normalization' and ('N',self.updates) in self.direction_done,'incomplete N job')
        self.measured['N'].append(self.updates);self.phase=None
    def trial_begin(self,j,theta):
        import numpy as np
        require(self.role in ('P','U') and self.updates<35 and self.phase is None and self.next_trial==j and j<20
                and self.pending is None and stop(self.updates,self.last) is None,'trial outside free path or cap')
        require(np.array_equal(theta,self.current+2.**(-j)*np.asarray(self.last['direction'])) and not theta.flags.writeable,'trial direction or alpha changed')
        self.phase='trial';self.forms=self.kernels=0;self.trial_theta=theta;self.trial_index=j;self.trial_digest=digest(theta)
    def trial_end(self,values):
        require(self.phase=='trial' and self.forms==self.kernels==1 and digest(self.trial_theta)==self.trial_digest,'incomplete trial')
        row=decision(self.last,values,self.trial_index,self.fixed_mse)
        self.counts['trials']+=1;self.counts['trial_fit_rows']+=d.N;self.phase=None;self.next_trial+=1
        if row['accepted']:self.pending=self.trial_theta
        self.check_caps();return row
    def commit(self,theta):
        require(self.pending is theta and self.phase is None and self.role in ('P','U') and self.updates<35,'unaccepted or forbidden update')
        self.counts['updates']+=1;self.counts['parameter_coordinate_writes']+=257;self.check_caps()
    def check_caps(self):
        mapping={'gradient_states':'gradient_states_max','gradient_vectors':'gradient_vectors_max','gradient_coordinates':'gradient_coordinates_max',
         'updates':'updates_max','parameter_coordinate_writes':'parameter_coordinate_writes_max','trials':'trials_max','linear_forms':'linear_forms_max',
         'linear_form_rows':'linear_form_rows_max','state_fit_rows':'state_fit_rows_max','state_seven_action_coordinates':'state_seven_action_coordinates_max','trial_fit_rows':'trial_fit_rows_max'}
        for k,cap in mapping.items():require(self.counts[k]<=self.cap[cap],'compute budget exceeded: '+k)
        if self.stage=='cross':require(self.counts['autograd_calls']<=self.cfg['budget']['cross_autograd_max'],'autograd budget exceeded')
        dc=self.direction_work.counts
        for k,cap in [('constructions','direction_constructions_max'),('norm_reductions','direction_norm_reductions_max'),('dot_reductions','direction_dot_reductions_max'),('vector_forms','direction_vector_forms_max'),('coordinate_writes','direction_coordinate_writes_max')]:
            require(dc[k]<=self.cap[cap],'direction budget exceeded: '+k)
        self.limits.check()
    def record(self):
        return dict(computation=dict(self.counts),direction=dict(self.direction_work.counts),measured={k:list(v) for k,v in self.measured.items()})
    def complete(self):
        require(self.phase is None and self.counts['linear_forms']==2*self.counts['gradient_states']+self.counts['trials']
                and self.counts['gradient_vectors']==3*self.counts['gradient_states'],'computation ledger inconsistent')
        if self.stage=='primary':require(self.measured['P']==list(range(36)) and self.counts['updates']==35
                                        and self.counts['trials']==self.cfg['budget']['historical_P_trial_rows'],'P prefix incomplete')
        else:d.validate_coverage(list(range(36)),self.measured['U'],self.measured['S'],self.measured['N'])
        require(self.context_digests==[digest(v) for v in self.context],'fit context mutated');self.check_caps()


def stop(k,row):
    if k==35:return 'diagnostic_prefix_cap'
    if row['total_l2']==0:return 'zero_gradient'
    if row['direction_status']=='unresolved_joint_direction' or not row['selected_joint']:return 'unresolved_joint_direction'
    if row['exact_MSE_slope']>0:return 'exact_MSE_direction_increase'
    if row['directional_derivative']>=0:return 'non_descent_selected_direction'
    return None


def decision(point,values,j,m500):
    bound=point['trainable_objective']+.0001*2.**(-j)*point['directional_derivative']
    armijo=bool(values['trainable_objective']<point['trainable_objective'] and values['trainable_objective']<=bound)
    mse=bool(values['component3_mse']<=m500);accepted=armijo and mse
    reason='accepted' if accepted else 'armijo_and_mse_rejected' if not armijo and not mse else 'armijo_rejected' if not armijo else 'mse_envelope_rejected'
    return dict(armijo_pass=armijo,mse_envelope_pass=mse,mse_gate_applied=True,accepted=accepted,reason=reason)


def anchor(expected,actual):
    for field,wanted in expected.items():
        value=actual[field]
        if type(wanted) is float:require(type(value) in (float,int) and math.isfinite(value) and abs(value-wanted)<=d.TOL,'P historical scalar differs: '+field)
        else:require(type(value) is type(wanted) and value==wanted,'P historical discrete/digest differs: '+field)


class NumericGuard(base.ReadGuard):
    """Scoped contract, not an OS sandbox; exactly listed archives/members/jobs."""
    def __init__(self,cfg,stage,jobs,limits,write_paths,trace_ref=None):
        self.cfg,self.stage,self.jobs,self.limits=cfg,stage,jobs,limits
        self.refs={str(Path(r['path']).resolve()):dict(r,kind=k) for k,r in cfg['inputs'].items()}
        if trace_ref:self.refs[str(Path(trace_ref['path']).resolve())]=dict(trace_ref,kind='trace')
        self.trace_path=str((Path(cfg['artifacts'])/'primary-trace.npz').resolve())
        self.telemetry_path=str(Path('/proc/self/status').resolve())
        self.write_paths={str(Path(p).resolve()) for p in write_paths};self.saved=[];self.denied=[]
        self.attempts=[];self.completed=[];self.member_attempts=[];self.member_completed=[];self.hash_checks=[];self.hash_attempts=[];self.writes=[];self.write_members=[]
        self.registry={};self.zip_registry={};self.load_depth=self.hash_depth=self.member_depth=self.save_depth=0;self.active_member=None
    def fields(self,kind):return self.cfg['trace_members'] if kind=='trace' else self.cfg['head_members'] if kind=='head' else self.cfg['source_members'][kind]
    def hash_one(self,path,expected=None):
        self.hash_attempts.append(str(path))
        self.hash_depth+=1
        try:sha=d.p.sha(path)
        finally:self.hash_depth-=1
        self.hash_checks.append(str(path))
        if expected is not None:require(sha==expected,'archive hash changed')
        return sha
    def __enter__(self):
        import numpy as np
        load0,get0,save0,zip0=np.load,np.lib.npyio.NpzFile.__getitem__,np.savez,zipfile.ZipFile.open
        open0,io0,path0,file0=builtins.open,io.open,Path.open,io.FileIO
        read0,from0,import0=np.lib.format.read_array,np.fromfile,builtins.__import__;einsum0=np.einsum
        def loaded(path,*args,**kw):
            name=str(Path(path).resolve()) if isinstance(path,(str,Path)) else ''
            if name not in self.refs or name in self.attempts:return self.reject('unlisted_or_repeated_archive',name)
            if args or kw!={'allow_pickle':False}:return self.reject('pickle_mmap_arguments','')
            self.attempts.append(name);self.hash_one(name,self.refs[name]['sha256']);self.load_depth+=1
            try:z=load0(path,allow_pickle=False)
            finally:self.load_depth-=1
            require(isinstance(z,np.lib.npyio.NpzFile),'NPZ only');self.registry[id(z)]=name;self.zip_registry[id(z.zip)]=name
            allowed=set(self.fields(self.refs[name]['kind']))
            require((set(z.files)==allowed if self.refs[name]['kind']=='trace' else allowed.issubset(z.files)),
                    'required archive fields missing or trace fields changed')
            self.completed.append(name);return z
        def get(z,field):
            name=self.registry.get(id(z));kind=self.refs[name]['kind'] if name in self.refs else None
            if kind is None or field not in self.fields(kind) or (name,field) in self.member_attempts:return self.reject('unlisted_or_repeated_member',field)
            self.member_attempts.append((name,field));self.member_depth+=1;self.active_member=name,field
            try:value=get0(z,field)
            finally:self.member_depth-=1;self.active_member=None
            require(isinstance(value,np.ndarray) and value.dtype.kind in 'biuf' and not value.dtype.hasobject,'numeric member required')
            readonly(value);self.jobs.array_sources[id(value)]=(name,field);self.member_completed.append((name,field));self.limits.check();return value
        def saved(path,*args,**arrays):
            name=str(Path(path).resolve()) if isinstance(path,(str,Path)) else ''
            if self.stage!='primary' or name!=self.trace_path or self.writes:return self.reject('unlisted_or_repeated_archive_write',name)
            require(not args and set(arrays)==set(self.cfg['trace_members']),'trace fields changed')
            require(self.jobs.measured['P']==list(range(36)),'trace before complete P')
            for k,v in arrays.items():
                dtype=np.int32 if k in ('steps','fit_rows','fit_demo_indices') else np.float64
                require(v.dtype==dtype and list(v.shape)==self.cfg['trace_layout'][k] and np.isfinite(v).all(),'trace schema changed')
            require(arrays['steps'].tolist()==list(range(36)) and arrays['fit_rows'].tolist()==self.cfg['fold']['fit_rows'] and arrays['fit_demo_indices'].tolist()==self.cfg['fold']['fit_demos'],'trace role changed')
            self.save_depth+=1
            try:save0(name,**arrays)
            finally:self.save_depth-=1
            require(Path(name).stat().st_size<=self.cfg['execution']['new_trace_NPZ_bytes_max'],'trace size exceeded')
            self.writes.append(name);return dict(path=name,sha256=self.hash_one(name))
        def zipped(z,field,mode='r',*args,**kw):
            member=field.filename if isinstance(field,zipfile.ZipInfo) else field
            if mode=='w':
                target=str(Path(z.filename).resolve()) if isinstance(z.filename,(str,Path)) else ''
                key=(target,str(member).removesuffix('.npy'))
                if not self.save_depth or target!=self.trace_path or key in self.write_members or key[1] not in self.cfg['trace_members']:return self.reject('zip_write_bypass',member)
                self.write_members.append(key)
            elif id(z) in self.zip_registry:
                if not self.member_depth or self.active_member!=(self.zip_registry[id(z)],str(member).removesuffix('.npy')):return self.reject('zip_read_bypass',member)
            else:return self.reject('unregistered_zip',member)
            return zip0(z,field,mode,*args,**kw)
        def check(path,mode):
            if not isinstance(path,(str,Path)):return self.reject('raw_descriptor','')
            name=str(Path(path).resolve());write=any(c in mode for c in 'wax+')
            if name in self.refs:
                if write or not (self.load_depth or self.hash_depth):return self.reject('raw_archive',name)
            elif name==self.trace_path:
                if not ((write and self.save_depth) or (not write and self.hash_depth)):return self.reject('raw_trace',name)
            elif name in self.write_paths:
                if not write:return self.reject('numeric_output_read',name)
            elif name==self.telemetry_path and not write:return
            else:return self.reject('unlisted_IO',name)
        def opened(path,mode='r',*args,**kw):check(path,mode);return open0(path,mode,*args,**kw)
        def io_opened(path,mode='r',*args,**kw):check(path,mode);return io0(path,mode,*args,**kw)
        def path_opened(path,mode='r',*args,**kw):check(path,mode);return path0(path,mode,*args,**kw)
        class File(file0):
            def __init__(self,path,mode='r',*args,**kw):check(path,mode);super().__init__(path,mode,*args,**kw)
        def imported(name,*args,**kw):
            if name.split('.')[0] in {'h5py','libero','robosuite','mujoco','pandas','pickle','src'} or (name.split('.')[0]=='torch' and self.stage=='primary'):return self.reject('forbidden_import',name)
            return import0(name,*args,**kw)
        def fromfile(*args,**kw):
            if not self.member_depth:return self.reject('fromfile','')
            return from0(*args,**kw)
        def read_array(*args,**kw):
            if not self.member_depth:return self.reject('read_array','')
            return read0(*args,**kw)
        def einsum(spec,x,w,*args,**kw):
            j=self.jobs
            require(self.stage=='primary' and spec=='ij,j->i' and x is j.context[0] and w.shape==(256,) and j.phase in ('state','trial')
                    and j.forms==j.kernels+1 and not args and kw=={'optimize':False},'unregistered NumPy linear kernel')
            j.kernels+=1;return einsum0(spec,x,w,**kw)
        deny=lambda *args,**kw:self.reject('alternate_reader_model_optimizer','')
        for obj,name,value in [(np,'load',loaded),(np.lib.npyio,'load',loaded),(np.lib.npyio.NpzFile,'__getitem__',get),(np,'savez',saved),
         (zipfile.ZipFile,'open',zipped),(builtins,'open',opened),(io,'open',io_opened),(Path,'open',path_opened),(io,'FileIO',File),
         (builtins,'__import__',imported),(np,'fromfile',fromfile),(np.lib.format,'read_array',read_array),(np,'einsum',einsum),
         (np,'loadtxt',deny),(np,'genfromtxt',deny),(np,'save',deny),(np,'savez_compressed',deny),(np.memmap,'__new__',deny),(os,'open',deny)]:self.patch(obj,name,value)
        if 'torch' in sys.modules:
            t=sys.modules['torch'];grad0=t.autograd.grad;linear0=t.nn.functional.linear
            def gradient(out,params,*args,**kw):self.jobs.gradient(params);return grad0(out,params,*args,**kw)
            def linear(x,w,b=None):
                j=self.jobs
                require(self.stage=='cross' and j.phase in ('state','trial') and id(x)==j.tensor_input_id and tuple(x.shape)==(d.N,256)
                        and tuple(w.shape)==(1,256) and tuple(b.shape)==(1,) and str(x.dtype)=='torch.float64','unregistered Torch linear kernel')
                j.linear(len(x));j.kernels+=1;return linear0(x,w,b)
            for obj,name in [(t,'load'),(t,'save'),(t.optim.Optimizer,'__init__'),(t.nn.Module,'__call__'),(t.autograd,'backward')]:self.patch(obj,name,deny)
            self.patch(t.autograd,'grad',gradient);self.patch(t.nn.functional,'linear',linear)
        return self
    def complete(self):
        expected={(p,f) for p,r in self.refs.items() for f in self.fields(r['kind'])}
        require(set(self.completed)==set(self.refs) and set(self.member_completed)==expected and not self.denied,'archive/member access incomplete')
        for name,ref in self.refs.items():self.hash_one(name,ref['sha256'])
        require(len(self.attempts)==(3 if self.stage=='primary' else 4) and len(self.member_completed)==(15 if self.stage=='primary' else 25)
                and len(self.hash_checks)==(7 if self.stage=='primary' else 8),'archive budget changed')
    def record(self):
        return dict(NPZ_loads=len(self.attempts),member_decodes=len(self.member_completed),archive_SHA=len(self.hash_checks),
                    completed_NPZ_loads=len(self.completed),member_decode_attempts=len(self.member_attempts),archive_SHA_attempts=len(self.hash_attempts),
                    files=self.completed,members=self.member_completed,writes=self.writes,write_members=self.write_members,
                    denied=self.denied,original_CSV_reads=0,original_Policy_CNN_environment_or_sealed_calls=0)


def load(guard,ref,kind):
    import numpy as np
    with np.load(ref['path'],allow_pickle=False) as z:return {f:z[f] for f in guard.fields(kind)}


def admit_inputs(cfg,guard,jobs,independent=False):
    import numpy as np
    shared=load(guard,cfg['inputs']['shared'],'shared');cache=load(guard,cfg['inputs']['current'],'current');head=load(guard,cfg['inputs']['head'],'head')
    lengths=shared['demo_lengths'].tolist();pairs=[(i,t) for i,n in enumerate(lengths) for t in range(1,n)]
    require(len(lengths)==30 and len(pairs)==4188 and shared['targets'].dtype==np.float32 and shared['targets'].shape==(4188,7)
            and list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==pairs
            and all(shared[f].dtype==np.int32 for f in ('demo_lengths','demo_indices','action_steps'))
            and shared['tail20'].dtype==np.bool_ and shared['tail20'].tolist()==[t>=lengths[i]-20 for i,t in pairs],'source schema changed')
    rows=np.asarray(cfg['fold']['fit_rows'],np.int32)
    require(rows.tolist()==[k for k,(i,t) in enumerate(pairs) if i in cfg['fold']['fit_demos'] and t>=1]
            and len(rows)==2777 and len(cfg['fold']['fit_demos'])==20,'fit role changed')
    require(cache['features'].dtype==cache['predictions'].dtype==np.float32 and cache['features'].shape==(4188,256)
            and cache['predictions'].shape==(4188,7),'cache schema changed')
    if independent:
        from experiments.audit_libero_task7_endpoint_gradient_diagnostic import prepare
        checked,z=prepare(head,cache['features'],cfg['fold'],lengths);require(np.array_equal(checked,rows),'independent fit changed')
    else:
        from experiments.run_libero_task7_endpoint_gradient_diagnostic import admit
        z=admit(head,cache['features'],cfg['fold'])
    context=tuple(readonly(v) for v in (z,shared['targets'][rows].astype(np.float64),cache['predictions'][rows].astype(np.float64)))
    require(all(np.isfinite(v).all() for v in context) and (abs(context[1])<=1).all() and (abs(context[2])<=1).all(),'invalid fit values')
    start=readonly(np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])]);jobs.register(context,start)
    return context,start,rows,head


def write_failure(path,stage,error,jobs,guard,limits,latest):
    # No fresh read, hash, norm, gradient, forward or telemetry in this handler.
    record=dict(status='failed_no_retry',stage=stage,error=repr(error),ledger=jobs.record(),access=guard.record(),cost=limits.snapshot(),latest=latest,
                last_direction=jobs.last_direction,last_direction_input_digests=jobs.last_direction_inputs)
    raw=json.dumps(record,allow_nan=False)
    require(len(raw.encode())<=1048576,'failure JSON budget exceeded')
    with Path(path).open('x') as f:f.write(raw+'\n')


class DerivedGuard(base.ReadGuard):
    """V reads each of five new CSVs and two new JSONLs once; SHA before/after."""
    def __init__(self,refs,limits):
        self.refs={str(Path(r['path']).resolve()):r for r in refs};require(len(self.refs)==7,'five CSV and two JSONL required')
        self.telemetry_path=str(Path('/proc/self/status').resolve())
        self.limits=limits;self.saved=[];self.denied=[];self.reads=[];self.hashes=[];self.hash_depth=self.parse_depth=0
    def __enter__(self):
        open0,io0,path0,file0,import0=builtins.open,io.open,Path.open,io.FileIO,builtins.__import__
        def check(path,mode):
            if not isinstance(path,(str,Path)):return self.reject('descriptor','')
            name=str(Path(path).resolve())
            if name==self.telemetry_path and mode=='r':return
            if name not in self.refs or any(c in mode for c in 'wax+') or not (self.hash_depth or self.parse_depth):return self.reject('unlisted_audit_IO',name)
        def opened(path,mode='r',*args,**kw):check(path,mode);return open0(path,mode,*args,**kw)
        def io_opened(path,mode='r',*args,**kw):check(path,mode);return io0(path,mode,*args,**kw)
        def path_opened(path,mode='r',*args,**kw):check(path,mode);return path0(path,mode,*args,**kw)
        class File(file0):
            def __init__(self,path,mode='r',*args,**kw):check(path,mode);super().__init__(path,mode,*args,**kw)
        def imported(name,*args,**kw):
            if name.split('.')[0] in {'numpy','torch','pandas','pickle','h5py','libero','mujoco','robosuite','src'}:return self.reject('audit_numeric_or_environment_import',name)
            return import0(name,*args,**kw)
        deny=lambda *args,**kw:self.reject('audit_alternate_IO','')
        for obj,name,value in [(builtins,'open',opened),(io,'open',io_opened),(Path,'open',path_opened),(io,'FileIO',File),(builtins,'__import__',imported),
                               (os,'open',deny),(zipfile.ZipFile,'open',deny)]:self.patch(obj,name,value)
        return self
    def read(self,ref,jsonl=False):
        name=str(Path(ref['path']).resolve());require(name in self.refs and name not in self.reads,'repeated or unlisted derived read')
        self._hash(name,ref['sha256'])
        self.reads.append(name);self.parse_depth+=1
        try:
            with Path(name).open(newline='') as f:
                if jsonl:rows=[json.loads(line) for line in f]
                else:
                    reader=csv.DictReader(f);require(reader.fieldnames==ref['columns'],'CSV columns differ');rows=list(reader)
        finally:self.parse_depth-=1
        require(len(rows)==ref['rows'],'derived rows differ');self._hash(name,ref['sha256']);self.limits.check();return rows
    def _hash(self,name,expected):
        self.hash_depth+=1
        try:actual=d.p.sha(name)
        finally:self.hash_depth-=1
        self.hashes.append(name);require(actual==expected,'derived bytes changed')
    def record(self):return dict(read_passes=self.reads,byte_SHA_checks=self.hashes,denied=self.denied)
    def complete(self):require(set(self.reads)==set(self.refs) and len(self.hashes)==14 and not self.denied,'arithmetic read budget incomplete')
