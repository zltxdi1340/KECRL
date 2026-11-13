"""Synthetic-only role, payload, failure logging and stdlib audit checks."""
import builtins
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import numpy as np
import pytest
from experiments import libero_task7_joint_direction_prefix_review_protocol as d
from experiments import libero_task7_joint_direction_prefix_review_access as a
from experiments import run_libero_task7_joint_direction_prefix_numerical_review as P
from experiments import run_libero_task7_joint_direction_prefix_cross_review as C
from experiments import audit_libero_task7_joint_direction_prefix_numerical_review as V


@pytest.fixture
def cfg():return json.loads(d.DEFAULT_CONFIG.read_text())


def test_unbound_API_fails_before_directory_or_archive(cfg,tmp_path,monkeypatch):
    cfg['artifacts']=str(tmp_path/'must-not-exist');cfg['output']=str(tmp_path/'no-result.json')
    def no_binding(*args):raise FileNotFoundError('unbound')
    def forbidden(*args,**kwargs):raise AssertionError('archive or job before binding')
    monkeypatch.setattr(a,'preflight',no_binding);monkeypatch.setattr(np,'load',forbidden);monkeypatch.setattr(a,'Jobs',forbidden)
    for fn in (P.execute_payload,C.execute_payload,V.execute_payload):
        with pytest.raises(FileNotFoundError):fn(cfg,{'path':str(tmp_path/'science.json')},{'path':str(tmp_path/'binding.json')})
    assert not Path(cfg['artifacts']).exists()


def setup_jobs(cfg,tmp_path,stage='primary'):
    limits=a.Limits(180,536870912);jobs=a.Jobs(cfg,stage,limits)
    z=a.readonly(np.zeros((2777,256),np.float64));y=a.readonly(np.zeros((2777,7),np.float64));p0=a.readonly(np.zeros((2777,7),np.float64))
    jobs.register((z,y,p0),a.readonly(np.zeros(257,np.float64)))
    return jobs,limits


@pytest.mark.parametrize('role,step',[('P',36),('P',-1),('P',True),('U',0),('S',0),('N',0)])
def test_primary_rejects_other_roles_and_unfrozen_steps(cfg,tmp_path,role,step):
    jobs,_=setup_jobs(cfg,tmp_path)
    with pytest.raises(ValueError):jobs.select(role,step,jobs.start)


def test_cross_jobs_wait_for_U_and_never_cross_feed_free_path(cfg,tmp_path):
    jobs,_=setup_jobs(cfg,tmp_path,'cross')
    with pytest.raises(ValueError):jobs.select('S',0,jobs.start)
    with pytest.raises(ValueError):jobs.select('U',0,a.readonly(np.ones(257)))
    jobs.select('U',0,jobs.start)
    with pytest.raises(ValueError):jobs.gradient(())
    with pytest.raises(ValueError):jobs.linear(2777)


@pytest.mark.parametrize('fault',['state','gradient','trial','linear','update','direction'])
def test_each_compute_budget_has_fail_closed_cap(cfg,tmp_path,fault):
    jobs,_=setup_jobs(cfg,tmp_path)
    target={'state':('gradient_states','gradient_states_max'),'gradient':('gradient_vectors','gradient_vectors_max'),
            'trial':('trials','trials_max'),'linear':('linear_forms','linear_forms_max'),'update':('updates','updates_max')}
    if fault=='direction':jobs.direction_work.counts['constructions']=37
    else:
        field,cap=target[fault];jobs.counts[field]=jobs.cap[cap]+1
    with pytest.raises(ValueError):jobs.check_caps()


def test_step35_cannot_create_trial_even_with_descent(cfg,tmp_path):
    jobs,_=setup_jobs(cfg,tmp_path);jobs.role='P';jobs.updates=35;jobs.phase=None;jobs.next_trial=0
    jobs.last=dict(total_l2=1.,direction_status='joint_bisector',selected_joint=True,exact_MSE_slope=-1.,directional_derivative=-1.)
    with pytest.raises(ValueError):jobs.trial_begin(0,jobs.start)
    assert a.stop(35,jobs.last)=='diagnostic_prefix_cap'


def test_failure_keeps_actual_values_and_ledgers_without_fresh_read(cfg,tmp_path,monkeypatch):
    jobs,limits=setup_jobs(cfg,tmp_path);jobs.last_direction={'unit_sum_norm':1.014,'status':'joint_bisector'}
    class Guard:
        def record(self):return {'NPZ_loads':2,'denied':[]}
    def no_read(*args):raise AssertionError('failure handler must not read telemetry')
    monkeypatch.setattr(limits,'check',no_read)
    path=tmp_path/'failed.json';a.write_failure(path,'cross',ValueError('test'),jobs,Guard(),limits,{'actual_q':1.015,'gradient':[.2]})
    value=json.loads(path.read_text())
    assert value['latest']['actual_q']==1.015 and value['last_direction']['unit_sum_norm']==1.014
    assert value['cost']['peak_RSS_bytes']==limits.peak_RSS_bytes and value['ledger']['computation']['gradient_vectors']==0


def toy_guard(cfg,tmp_path):
    cfg=copy.deepcopy(cfg);refs={}
    for kind,fields in [('shared',cfg['source_members']['shared']),('current',cfg['source_members']['current']),('head',cfg['head_members'])]:
        path=tmp_path/(kind+'.npz');np.savez(path,**{f:np.asarray([1.],np.float64) for f in fields},unread_extra=np.asarray([17.]))
        refs[kind]=d.p.pin(path)
    cfg['inputs']=refs;cfg['artifacts']=str(tmp_path/'out');jobs,limits=setup_jobs(cfg,tmp_path)
    return cfg,jobs,limits,a.NumericGuard(cfg,'primary',jobs,limits,[])


@pytest.mark.parametrize('fault',['unlisted','repeat_load','repeat_member','extra_member','raw','zip','pickle','mmap','fromfile','import','linear'])
def test_numeric_access_rejects_bypasses(cfg,tmp_path,fault):
    cfg,jobs,limits,guard=toy_guard(cfg,tmp_path)
    with guard:
        with pytest.raises((PermissionError,ValueError)):
            if fault=='unlisted':np.load(tmp_path/'other.npz',allow_pickle=False)
            elif fault=='raw':Path(cfg['inputs']['head']['path']).read_bytes()
            elif fault=='fromfile':np.fromfile(cfg['inputs']['head']['path'])
            elif fault=='import':__import__('h5py')
            elif fault=='linear':np.einsum('ij,j->i',np.zeros((2777,256)),np.zeros(256),optimize=False)
            elif fault=='pickle':np.load(cfg['inputs']['head']['path'],allow_pickle=True)
            elif fault=='mmap':np.load(cfg['inputs']['head']['path'],allow_pickle=False,mmap_mode='r')
            else:
                with np.load(cfg['inputs']['head']['path'],allow_pickle=False) as data:
                    if fault=='repeat_load':np.load(cfg['inputs']['head']['path'],allow_pickle=False)
                    elif fault=='extra_member':data['unread_extra']
                    elif fault=='repeat_member':data['weights'];data['weights']
                    else:data.zip.open('weights.npy')


def test_allowed_members_decode_once_without_decoding_extra(cfg,tmp_path):
    cfg,jobs,limits,guard=toy_guard(cfg,tmp_path)
    # Production stages use fresh processes. A full pytest host has unrelated
    # Torch models in its high-water RSS; keep the real 512MiB cap in a child.
    path=tmp_path/'toy-guard.json';path.write_text(json.dumps(cfg))
    code=textwrap.dedent('''
        import json,sys,numpy as np
        from pathlib import Path
        from experiments import libero_task7_joint_direction_prefix_review_access as a
        cfg=json.loads(Path(sys.argv[1]).read_text());limits=a.Limits(180,536870912)
        jobs=a.Jobs(cfg,'primary',limits);guard=a.NumericGuard(cfg,'primary',jobs,limits,[])
        with guard:
            with np.load(cfg['inputs']['current']['path'],allow_pickle=False) as archive:
                for f in cfg['source_members']['current']:assert not archive[f].flags.writeable
        assert guard.record()['member_decodes']==2 and not guard.denied
    ''')
    env=os.environ.copy();env.update(d.EXECUTION['environment'])
    result=subprocess.run([sys.executable,'-c',code,str(path)],capture_output=True,text=True,env=env,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr


def test_scalar_audit_preserves_observed_threshold_failure():
    r=V.q_decomposition(1.,1.,1.,1.+2e-10)
    assert r['total_over_tolerance'] and r['total_abs']>1e-10
    assert V.q_decomposition(1.,1.,1.,None)['total'] is None


def make_synthetic(root,cfg):
    """Create all archives locally; no study archive is opened or hashed."""
    rng=np.random.RandomState(731)
    lengths=[140]*19+[137]+[142]*9+[143];pairs=[(i,t) for i,n in enumerate(lengths) for t in range(1,n)]
    assert len(pairs)==4188
    rows=np.asarray([j for j,(i,t) in enumerate(pairs) if i<20],np.int32);assert len(rows)==2777
    f=rng.normal(size=(4188,256)).astype(np.float32)*.06
    y=np.zeros((4188,7),np.float32);y[:,2]=np.clip(.2+.8*f[:,0]-.3*f[:,1],-1,1)
    p0=np.zeros_like(y);p0[:,2]=-.07
    mean=f[rows].astype(np.float64).mean(0);scale=f[rows].astype(np.float64).std(0);scale[scale<=1e-6]=1.
    weights=np.zeros((4,256),np.float32);weights[3,:2]=[.003,-.002];bias=np.zeros(4,np.float32);bias[3]=.005
    root.mkdir();inputs={}
    arrays={'shared':dict(targets=y,demo_indices=np.asarray([i for i,t in pairs],np.int32),action_steps=np.asarray([t for i,t in pairs],np.int32),demo_lengths=np.asarray(lengths,np.int32),tail20=np.asarray([t>=lengths[i]-20 for i,t in pairs],bool)),
     'current':dict(features=f,predictions=p0,logits=np.zeros((4188,2),np.float32)),
     'head':dict(fit_rows=rows,fit_demo_indices=np.arange(20,dtype=np.int32),scaler_mean=mean,scaler_scale=scale,updates=np.asarray([0,100,250,500],np.int32),weights=weights,bias=bias,optimizer_steps=np.arange(1,501,dtype=np.int32))}
    for kind,arr in arrays.items():
        path=root/(kind+'.npz');np.savez(path,**arr);inputs[kind]=d.p.pin(path)
    cfg=copy.deepcopy(cfg);cfg['inputs']=inputs;cfg['artifacts']=str(root/'artifacts');cfg['output']=str(root/'result.json')
    cfg['fold']=dict(fold=2,fit_demos=list(range(20)),fit_rows=rows.tolist(),fit_tge1_rows=2777)
    from experiments.run_libero_task7_endpoint_gradient_diagnostic import admit
    z=admit(arrays['head'],f,cfg['fold']);context=(z,y[rows].astype(np.float64),p0[rows].astype(np.float64));theta=np.r_[weights[3].astype(np.float64),float(bias[3])]
    from experiments import run_libero_task7_joint_direction_diagnostic as num
    states=[];trials=[]
    for k in range(36):
        point=num.measure(context,theta,1.,rows,arm=d.OPTIMIZATION_ARM)
        states.append({**a.IDENT,'extra_updates':k,'parameter_digest':a.digest(theta),**{f:v for f,v in point.items() if f not in a.VECTORS}})
        if k==35:break
        for j in range(20):
            trial=theta+2.**(-j)*np.asarray(point['direction']);values=num.trial_objective(context,trial,1.,point['fixed_other6_floor'])
            decision=a.decision(point,values,j,states[0]['component3_mse'])
            row={**a.IDENT,'from_extra_updates':k,'trial_index':j,'alpha':2.**(-j),**values,'start500_component3_mse':states[0]['component3_mse'],
                 'slope':point['directional_derivative'],'current_component3_mse':point['component3_mse'],'delta_component3_mse':values['component3_mse']-point['component3_mse'],
                 'armijo_bound':point['trainable_objective']+.0001*2.**(-j)*point['directional_derivative'],**decision}
            trials.append(row)
            if decision['accepted']:theta=trial;break
        else:raise AssertionError('synthetic prefix must complete35')
    cfg['historical_prefix']['states']=states;cfg['historical_prefix']['trials']=trials
    cfg['budget']['historical_P_trial_rows']=len(trials);cfg['budget']['CSV_rows']['primary_trials']=len(trials)
    cfg['budget']['primary']['trials_max']=max(258,len(trials))
    cfg['budget']['primary']['trial_fit_rows_max']=cfg['budget']['primary']['trials_max']*2777
    cfg['budget']['primary']['linear_forms_max']=72+cfg['budget']['primary']['trials_max']
    cfg['budget']['primary']['linear_form_rows_max']=cfg['budget']['primary']['linear_forms_max']*2777
    path=root/'synthetic-science.json';path.write_text(json.dumps(cfg));binding=root/'synthetic-binding.json';binding.write_text('{}')
    return cfg,path,binding


def child(stage,path,binding,root):
    modules={'primary':P.__name__,'cross':C.__name__,'audit':V.__name__}
    source=textwrap.dedent('''
        import json,sys
        from pathlib import Path
        from experiments import libero_task7_joint_direction_prefix_review_access as a
        from experiments import libero_task7_joint_direction_prefix_review_protocol as d
        import importlib
        cfg=json.loads(Path(sys.argv[1]).read_text())
        root=Path(sys.argv[3]).resolve()
        assert all(Path(v['path']).resolve().is_relative_to(root) for v in cfg['inputs'].values())
        assert Path(cfg['artifacts']).resolve().is_relative_to(root)
        a.validate_call=lambda *args: None  # isolated synthetic fixture only
        module=importlib.import_module(sys.argv[4])
        module.execute_payload(cfg,d.p.pin(Path(sys.argv[1])),d.p.pin(Path(sys.argv[2])))
    ''')
    env=os.environ.copy();env.update(d.EXECUTION['environment'])
    return subprocess.run([sys.executable,'-c',source,str(path),str(binding),str(root),modules[stage]],text=True,capture_output=True,env=env,timeout=120)


def test_complete_synthetic_P_C_V_flow_and_target_failure_retention(cfg,tmp_path):
    cfg,path,binding=make_synthetic(tmp_path/'synthetic',cfg)
    for stage in ('primary','cross','audit'):
        result=child(stage,path,binding,path.parent)
        assert result.returncode==0,result.stdout+result.stderr
    result=json.loads(Path(cfg['output']).read_text());assert result['old_v1_certified'] is False
    audit=json.loads((Path(cfg['artifacts'])/'independent-audit.json').read_text())
    assert audit['runtime']['stage']=='audit' and audit['original_archive_or_gradient_calls']==0
    assert audit['arithmetic']['q_reference_jobs']==108 and audit['arithmetic']['coordinate_telescope_checks']==37008
    assert len(audit['access']['read_passes'])==7 and len(audit['access']['byte_SHA_checks'])==14
    # Strict q mismatch reporting is independently checked in scalar tests; no favorable outcome is required here.


@pytest.mark.parametrize('fault',['header','rows','bytes'])
def test_arithmetic_guard_rejects_corruption_without_array_access(tmp_path,fault):
    refs=[]
    for i in range(7):
        path=tmp_path/f'derived{i}.csv';path.write_text('step,value\n0,1\n');refs.append(dict(d.p.pin(path),rows=1,columns=['step','value']))
    if fault=='header':refs[0]['columns']=['wrong']
    elif fault=='rows':refs[0]['rows']=2
    else:Path(refs[0]['path']).write_text('changed')
    guard=a.DerivedGuard(refs,a.Limits())
    with guard:
        with pytest.raises(ValueError):guard.read(refs[0])


def test_maximum_numeric_resource_envelope(tmp_path):
    code=textwrap.dedent('''
        import json,sys,numpy as np
        from experiments import libero_task7_joint_direction_prefix_review_protocol as d
        from experiments import libero_task7_joint_direction_prefix_review_access as a
        stage=sys.argv[1];a.runtime({'execution':d.EXECUTION},stage)
        rng=np.random.RandomState(17);z=rng.normal(size=(2777,256))*.03
        y=np.zeros((2777,7));y[:,2]=.2+.05*z[:,0];p0=np.zeros_like(y);p0[:,2]=-.05
        context=(z,y,p0);theta=np.r_[np.zeros(256),.005];rows=np.arange(2777,dtype=np.int32)
        jobs,tries=(36,258) if stage=='primary' else (72,700)
        cap=d.EXECUTION['stages'][stage];limits=a.Limits(cap['seconds'],cap['RSS_bytes'])
        from experiments.libero_task7_joint_direction_access import DirectionWork
        work=DirectionWork()
        if stage=='primary':
            from experiments import run_libero_task7_joint_direction_diagnostic as num
            backend=context
        else:
            from experiments import audit_libero_task7_joint_direction_diagnostic as num
            from experiments.audit_libero_task7_fixed_representation_optimization import TorchFit
            backend=TorchFit(context)
        for i in range(jobs):point=num.measure(backend,theta,1.,rows,arm=d.OPTIMIZATION_ARM,direction_work=work);limits.check()
        for i in range(tries):num.trial_objective(backend,theta,1.,point['fixed_other6_floor']);limits.check()
        if stage=='cross':
            for i in range(36):num.construct_direction(d.OPTIMIZATION_ARM,point['total_gradient'],point['base_gradient'],work);limits.check()
        assert work.counts['constructions']==(36 if stage=='primary' else 108)
        print(json.dumps({'stage':stage,'gradient_states':jobs,'trials':tries,'autograd_calls':jobs*3 if stage=='cross' else 0,'directions':work.counts,'cost':limits.record()}))
    ''')
    env=os.environ.copy();env.update(d.EXECUTION['environment'])
    for stage in ('primary','cross'):
        proc=subprocess.run([sys.executable,'-c',code,stage],text=True,capture_output=True,env=env,timeout=90)
        assert proc.returncode==0,proc.stdout+proc.stderr
        (tmp_path/(stage+'-envelope.json')).write_text(proc.stdout)


@pytest.mark.parametrize('role',['S','N'])
def test_cross_readout_cannot_trial_commit_or_import_training(cfg,tmp_path,role):
    jobs,_=setup_jobs(cfg,tmp_path,'cross');jobs.finished_U=True
    jobs.P_trace=dict(parameters=np.zeros((36,257)),base_gradient=np.zeros((36,257)),total_gradient=np.ones((36,257)))
    if role=='S':jobs.select('S',0,jobs.start)
    else:jobs.normalize(0,jobs.P_trace['base_gradient'][0],jobs.P_trace['total_gradient'][0])
    with pytest.raises(ValueError):jobs.trial_begin(0,jobs.start)
    with pytest.raises(ValueError):jobs.commit(jobs.start)
    with pytest.raises(ValueError):jobs.gradient(())


def test_N_cannot_substitute_newly_measured_gradient(cfg,tmp_path):
    jobs,_=setup_jobs(cfg,tmp_path,'cross');jobs.finished_U=True
    jobs.P_trace=dict(parameters=np.zeros((36,257)),base_gradient=np.zeros((36,257)),total_gradient=np.ones((36,257)))
    with pytest.raises(ValueError):jobs.normalize(0,np.ones(257),np.ones(257))
    assert jobs.counts['gradient_vectors']==jobs.counts['updates']==0


def test_comparison_records_large_q_gap_and_null_U_without_asserting(cfg):
    C.comparison.point_columns=[f for f in cfg['output_columns']['primary_states'] if f not in d.IDS+['optimization_arm','extra_updates']]
    point={f:cfg['historical_prefix']['states'][0][f] for f in C.comparison.point_columns}
    point.update({f:[.01]*257 for f in a.VECTORS});point['direction_unit_sum_norm']=1.
    other=dict(point,direction_unit_sum_norm=1.+2e-10)
    meta=dict(status='joint_bisector',total_norm=.2,base_norm=.2,unit_sum_norm=1.,direction_norm=.2,selected_T_slope=-.1,selected_MSE_slope=-.1,
              selected_joint=True,input_Gb_digest=a.digest(point['base_gradient']),input_GT_digest=a.digest(point['total_gradient']),direction=[.01]*257)
    counters=dict(error_norms=0,error_maxabs=0,error_coordinates=0,reference_norms=0)
    r,_=C.comparison(0,point,point,other,meta,np.zeros(257),np.ones(257)*1e-12,'diagnostic_prefix_cap',counters)
    assert r['q_total_over_tolerance'] and r['q_path_over_tolerance']
    r,raw=C.comparison(1,point,point,None,meta,np.zeros(257),None,'line_search_exhausted',counters)
    assert r['U_available'] is False and r['q_total'] is None and raw['theta_U'] is None
    assert r['theta_U_P_l2'] is None and r['theta_U_P_relative_l2'] is None


@pytest.mark.parametrize('fault',['numeric_import','unlisted_read','raw_read'])
def test_arithmetic_guard_denies_read_bypasses(tmp_path,fault):
    refs=[]
    for i in range(7):
        p=tmp_path/f'r{i}.csv';p.write_text('x\n1\n');refs.append(dict(d.p.pin(p),rows=1,columns=['x']))
    with a.DerivedGuard(refs,a.Limits()):
        with pytest.raises(PermissionError):
            if fault=='numeric_import':__import__('numpy')
            elif fault=='unlisted_read':Path(d.DEFAULT_CONFIG).read_text()
            else:Path(refs[0]['path']).read_text()


def test_state_evidence_is_flushed_before_P_anchor_failure(cfg,tmp_path,monkeypatch):
    jobs,limits=setup_jobs(cfg,tmp_path)
    point={f:v for f,v in cfg['historical_prefix']['states'][0].items() if f not in d.IDS+['optimization_arm','extra_updates','parameter_digest']}
    point.update({f:[0.]*257 for f in a.VECTORS})
    monkeypatch.setattr(P.numerical,'measure',lambda *args,**kwargs:point)
    class Sink:
        def __init__(self):self.rows=[];self.handle=io_handle
        def add(self,row):self.rows.append(row)
    import io
    io_handle=io.StringIO();s=Sink();events=[]
    class Events:
        def add(self,*args,**kwargs):events.append(args)
    latest={}
    with pytest.raises(ValueError,match='parameter_digest'):
        P.reconstruct(cfg,jobs,jobs.context,jobs.start,np.arange(2777),s,Sink(),Events(),latest)
    assert s.rows and latest['state']['parameter_digest']==a.digest(jobs.start)
    assert len(latest['parameters'])==257 and 'vectors' in latest and len(events)==2


@pytest.mark.parametrize('states,updates',[(37,35),(0,0),(True,0),(36,34)])
def test_arithmetic_manifest_rejects_unrecorded_U_states_before_read(cfg,tmp_path,states,updates):
    with pytest.raises(ValueError,match='cardinality'):
        V.manifest_checks(cfg,{},dict(U_states=states,U_updates=updates),tmp_path)


def test_arithmetic_manifest_enforces_frozen_header_before_read(cfg,tmp_path):
    P=dict(states=[{}]*36,trials=[{}]*258,tables={});C=dict(U_states=36,U_updates=35,comparisons=[{}]*36,tables={})
    for name in ('primary-states','primary-trials','comparisons','coordinates','torch_trials'):
        owner=P if name.startswith('primary') else C
        owner['tables'][name]={'path':str(tmp_path/(name+'.csv')),'columns':['wrong'],'rows':36}
    with pytest.raises(ValueError,match='frozen columns'):V.manifest_checks(cfg,P,C,tmp_path)
