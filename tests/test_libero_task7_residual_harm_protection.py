"""Synthetic Torch, access, paired-endpoint and independent-audit checks."""
from __future__ import annotations

import copy
import inspect
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import libero_task7_residual_harm_protection_protocol as d
from experiments import libero_task7_residual_harm_protection_access as access
from experiments import run_libero_task7_residual_harm_protection as trainer
from experiments import audit_libero_task7_residual_harm_protection as audit


def make_synthetic(directory, *, dummy_history=True):
    previous=runpy.run_path('tests/test_libero_task7_current_residual_correctability.py')
    cfg=previous['make_synthetic'](directory)
    cfg.update(scope=copy.deepcopy(d.SCOPE),execution=copy.deepcopy(d.EXECUTION),budget=d.budget(),parents={'synthetic':True})
    heads=[]
    for model in cfg['inputs']['models']:
        for fold in range(3):
            path=Path(directory)/f"history-{model['seed']}-{model['arm']}-{fold}.npz"
            if dummy_history:
                np.savez(path,**{key:np.zeros(1,np.float32) for key in d.HISTORICAL_MEMBERS},forbidden=np.zeros(1))
            heads.append({**{k:model[k] for k in ('task','seed','arm','checkpoint')},'fold':fold,
                          'archive':d.p.pin(path) if dummy_history else {'path':str(path),'sha256':'0'*64}})
    cfg['inputs']['historical_uniform_heads']=heads
    (Path(directory)/'config.json').write_text(json.dumps(cfg));return cfg


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):return make_synthetic(tmp_path_factory.mktemp('harm-protection'))


class NoLimits:
    def check(self):pass


class Parameter:
    device=SimpleNamespace(type='cpu');dtype='torch.float32';requires_grad=True
    def __init__(self,n):self.n=n
    def numel(self):return self.n
    def detach(self):return self
    def count_nonzero(self):return SimpleNamespace(item=lambda:0)


def test_fit_scaler_is_fit_only_and_probe_gate_requires_all36(synthetic):
    roles=access.Roles(synthetic);model=synthetic['inputs']['models'][0]
    f=np.zeros((4188,256),np.float32);f[synthetic['inputs']['folds'][0]['probe_rows']]=100
    x,mean,scale=roles.fit_transform(model,0,f)
    assert not x.any() and not mean.any() and np.array_equal(scale,np.ones(256))
    with pytest.raises(ValueError,match='before gate'):roles.full_transform(model,0,f,mean,scale)
    with pytest.raises(ValueError,match='all36'):roles.open_probe()


def test_memory_reused_scaler_is_bit_exact_to_historical_population_std(synthetic):
    rng=np.random.default_rng(111);feature=rng.normal(size=(4188,256)).astype(np.float32)
    feature[:,0]=.001;feature[:,1]=np.linspace(0,1e-7,4188,dtype=np.float32)
    roles=access.Roles(synthetic);model=synthetic['inputs']['models'][0]
    for fold,plan in enumerate(synthetic['inputs']['folds']):
        x,mean,scale=roles.fit_transform(model,fold,feature)
        old=feature[plan['fit_rows']].astype(np.float64);m=old.mean(0);s=old.std(0,ddof=0);s[s<=1e-6]=1
        old-=m;old/=s
        assert np.array_equal(mean.view(np.uint64),m.view(np.uint64))
        assert np.array_equal(scale.view(np.uint64),s.view(np.uint64))
        assert np.array_equal(x.view(np.uint32),old.astype(np.float32).view(np.uint32))


@pytest.mark.parametrize('mode',list(d.CASES))
def test_each_case_requires_exactly_weight256_bias1_and_blocks_probe_gradients(synthetic,mode):
    roles=access.Roles(synthetic);model=synthetic['inputs']['models'][0];key=access.head_key(model,0,mode)
    with pytest.raises(ValueError,match='parameters'): 
        with roles.head(key,[Parameter(257)]):pass
    manager=roles.head(key,[Parameter(256),Parameter(1)]);manager.__enter__()
    try:
        probe=np.repeat(synthetic['inputs']['folds'][0]['probe_rows'][0],64).astype(np.int32)
        with pytest.raises(ValueError,match='probe'):roles.batch(probe)
        with pytest.raises(ValueError,match='backward'):roles.backward()
        with pytest.raises(ValueError,match='original/unregistered'):roles.optimizer([{'params':[Parameter(256),Parameter(1)]}])
    finally:manager.__exit__(RuntimeError,RuntimeError('synthetic cleanup'),None)


@pytest.mark.parametrize('mode',list(d.CASES))
def test_sampler_is_same_frozen500x64_for_both_cases_and_A_B(synthetic,mode):
    plan=synthetic['inputs']['folds'][1]
    actual=trainer.sampled_rows(d.p.SEEDS[0],1,plan)
    assert np.array_equal(actual,audit.expected_samples(d.p.SEEDS[0],1,plan['fit_rows']))
    assert actual.shape==(500,64) and actual.dtype==np.int32


def test_main_may_hash_but_never_decode_or_raw_read_historical_heads(synthetic):
    guard=access.Guard(synthetic['inputs'],limits=NoLimits(),roles=access.Roles(synthetic))
    ref=synthetic['inputs']['historical_uniform_heads'][0]['archive']
    with guard:
        guard.hash_inputs()
        with pytest.raises(PermissionError):np.load(ref['path'],allow_pickle=False)
        with pytest.raises(PermissionError):Path(ref['path']).read_bytes()
    assert guard.record()['historical_head_loads']==0 and len(guard.hash_checks)==25


@pytest.mark.parametrize('field',['predictions','fit_rows','forbidden'])
def test_historical_projection_denies_every_non14_member(synthetic,field):
    refs=[r['archive'] for r in synthetic['inputs']['historical_uniform_heads']]
    heads=[{'path':'/never/open/new'+str(i)+'.npz','sha256':'0'*64} for i in range(36)]
    with access.Guard(synthetic['inputs'],limits=NoLimits(),heads=heads,historical=refs):
        with np.load(refs[0]['path'],allow_pickle=False) as archive:
            with pytest.raises(PermissionError):archive[field]


@pytest.mark.parametrize('operation',['member','repeat','unlisted','checkpoint','mmap','raw','CSV'])
def test_input_access_bypasses_fail_closed(synthetic,operation,tmp_path):
    ref=synthetic['inputs']['models'][0]['current_archive']
    with access.Guard(synthetic['inputs'],limits=NoLimits()):
        with pytest.raises(PermissionError):
            if operation=='member':
                with np.load(ref['path'],allow_pickle=False) as archive:archive['logits']
            elif operation=='repeat':
                with np.load(ref['path'],allow_pickle=False) as archive:archive['features'];archive['features']
            elif operation=='unlisted':np.load('/never/open/not.npz',allow_pickle=False)
            elif operation=='checkpoint':Path('/never/open/original.pt').read_bytes()
            elif operation=='mmap':np.load(ref['path'],allow_pickle=False,mmap_mode='r')
            elif operation=='raw':Path(ref['path']).read_bytes()
            else:Path('/never/open/paired-frames.csv').read_text()


@pytest.mark.parametrize('original,target,current', [(-.25,.25,-.25),(-.25,.25,.125),(.125,.25,.5),(.125,.25,-.0625)])
def test_real_Torch_hinge_subgradient_scale_and_improvement_rule(original,target,current):
    import torch
    gradients=[]
    for coefficient in (0.,1.):
        p=torch.zeros((64,7),dtype=torch.float32);p[:,2]=current;p.requires_grad_()
        y=torch.zeros_like(p);y[:,2]=target;b=torch.zeros_like(p);b[:,2]=original
        loss,telemetry=trainer.objective(p,y,b,coefficient);loss.backward()
        analytic=d.loss_terms([0,0,original,0,0,0,0],[0,0,target,0,0,0,0],current,coefficient)
        assert abs(loss.item()-analytic['total_loss'])<1e-7
        assert abs(p.grad[0,2].item()-analytic['gradient_wrt_clipped_prediction2']/64)<1e-8
        assert telemetry['step_active_harm_n']==(64 if analytic['active_harm'] else 0)
        gradients.append(p.grad.clone())
    if current==original:assert torch.equal(gradients[0],gradients[1])


def test_lambda0_returns_original_base_tensor_path_and_first_updates_are_equal():
    import torch
    torch.set_num_threads(1)
    rng=np.random.default_rng(17);x=torch.from_numpy(rng.normal(size=(64,256)).astype(np.float32))
    y=torch.from_numpy(rng.uniform(-.7,.7,(64,7)).astype(np.float32));p0=torch.from_numpy(rng.uniform(-.7,.7,(64,7)).astype(np.float32))
    outcomes=[]
    for coefficient in (0.,1.):
        w=torch.nn.Parameter(torch.zeros(256));b=torch.nn.Parameter(torch.zeros(1))
        p=p0.clone();p[:,2]=torch.clamp(p0[:,2]+torch.nn.functional.linear(x,w[None],b).squeeze(1),-1,1)
        loss,t=trainer.objective(p,y,p0,coefficient)
        assert t['step_penalty_loss']==t['step_active_harm_n']==0
        loss.backward();torch.nn.utils.clip_grad_norm_([w,b],1.,error_if_nonfinite=True)
        outcomes.append((w.grad.clone(),b.grad.clone()))
    assert all(torch.equal(a,b) for a,b in zip(*outcomes))
    # Equal initial gradients imply equal first Adam update with identical zero state.
    for (gw,gb) in outcomes:
        assert torch.isfinite(gw).all() and torch.isfinite(gb).all()


def test_independent_rules_match_frozen63_flags_for_pass_and_loss_of_same_frame():
    ns=runpy.run_path('tests/test_libero_task7_residual_harm_protection_protocol.py')
    for lost in (False,True):
        cube,good,retention=ns['cue_inputs']()
        if lost:retention['tail20']['positive'].update(retained_n=19,lost_n=1,candidate_only_n=6)
        actual=audit.independent_flags(cube,good,retention,True,True,True)
        expected=d.candidate_flags(cube,good,retention,numerical_sign_stable=True,numerical_harm_stable=True,other6_bit_exact=True,historical_baseline_bit_exact=True)
        assert actual==expected and actual['cue'] is (not lost)


@pytest.mark.parametrize('field',d.HISTORICAL_MEMBERS)
def test_historical_parity_rejects_any_changed_field(field):
    data={k:np.zeros(2,np.float32) for k in d.HISTORICAL_MEMBERS};legacy={k:v.copy() for k,v in data.items()}
    legacy[field][0]=1
    with pytest.raises(ValueError,match='not bit exact'):audit.check_historical(data,legacy,{'task':7,'seed':1,'arm':'synthetic','checkpoint':{'weight_hash':'x'}},0,SimpleNamespace(n=0))


@pytest.mark.parametrize('problem',['linear','loss','active_count','snapshot'])
def test_step_trace_audit_rejects_corrupt_numeric_records(problem):
    x=np.ones((4188,256),np.float32);p0=np.zeros((4188,7),np.float32);y=p0.copy();y[:,2]=.25
    zero=np.zeros(500,np.float64);data={'weights':np.zeros((4,256),np.float32),'bias':np.zeros(4,np.float32),
        'sample_rows':np.zeros((500,64),np.int32),'step_weights_before':np.zeros((500,256),np.float32),
        'step_bias_before':np.zeros(500,np.float32),'step_train_prediction2':np.zeros((500,64),np.float32),
        'step_base_loss':np.full(500,np.float32(.25**2/7),np.float64),'step_loss':np.full(500,np.float32(.25**2/7),np.float64),
        'step_penalty_loss':zero,'step_active_harm_n':np.zeros(500,np.int32),
        'step_original_component3_mse':np.full(500,.25**2),'step_candidate_component3_mse':np.full(500,.25**2)}
    if problem=='linear':data['step_weights_before'][7,0]=.1
    elif problem=='loss':data['step_base_loss'][7]+=.1
    elif problem=='active_count':data['step_active_harm_n'][7]=1
    else:data['step_bias_before'][100]=.1
    counter=dict(training_trace_batches=0,training_trace_rows=0,training_trace_active_harm_linear_disagreements=0)
    with pytest.raises(ValueError):audit.audit_training_trace(data,x,p0,y,'uniform_affine',SimpleNamespace(n=0),counter)


def test_independent_audit_fresh_import_has_no_primary_numerics_or_Torch():
    script="from experiments import audit_libero_task7_residual_harm_protection; import sys; assert 'torch' not in sys.modules; assert 'experiments.run_libero_task7_residual_harm_protection' not in sys.modules; print('independent')"
    r=subprocess.run([sys.executable,'-c',script],text=True,capture_output=True,timeout=10)
    assert r.returncode==0 and r.stdout.strip()=='independent',r.stderr
    assert 'd.candidate_flags(' not in inspect.getsource(audit)


def test_wrong_science_or_unbound_refuses_without_payload(monkeypatch,tmp_path):
    p=tmp_path/'wrong.json';p.write_text('{}')
    with pytest.raises(ValueError,match='scientific configuration'):access.preflight(p)
    monkeypatch.setattr(access.d,'validate',lambda *a: {})
    monkeypatch.setattr(access.d.p,'sha',lambda *a:access.SCIENCE_SHA)
    def unbound(*a):raise ValueError('execution binding required')
    monkeypatch.setattr(access.d,'require_binding',unbound)
    with pytest.raises(ValueError,match='execution binding'):access.preflight(p)


def test_full_paired500_pipeline_historical_parity_and_fresh_audit(tmp_path):
    directory=tmp_path/'full';cfg=make_synthetic(directory,dummy_history=False)
    config=directory/'config.json'
    prepare='''import json
from pathlib import Path
import numpy as np
from experiments import run_libero_task7_current_residual_correctability as m
from experiments import libero_task7_current_residual_correctability_access as a
cfg=json.loads(Path(CONFIG).read_text());a.runtime(cfg,training=True);roles=a.Roles(cfg)
with np.load(cfg['inputs']['shared']['path'],allow_pickle=False) as data:y=data['targets']
caches={}
for model in cfg['inputs']['models']:
 with np.load(model['current_archive']['path'],allow_pickle=False) as z:caches[model['seed'],model['arm']]={k:z[k] for k in ('features','predictions')}
class Limits:
 def check(self):pass
with a.Guard(cfg['inputs'],[r['archive']['path'] for r in cfg['inputs']['historical_uniform_heads']],Limits(),roles=roles):
 for model in cfg['inputs']['models']:
  cache=caches[model['seed'],model['arm']]
  for fold,plan in enumerate(cfg['inputs']['folds']):
   x,mean,scale=roles.fit_transform(model,fold,cache['features']);fit=np.asarray(plan['fit_rows'])
   arrays,meta=m.fit_head(x,y[fit],cache['predictions'][fit],m.sampled_rows(model['seed'],fold,plan),plan,(model['seed'],model['arm'],fold,'current_affine'),roles,Limits())
   arrays.update(scaler_mean=mean,scaler_scale=scale)
   ref=next(r for r in cfg['inputs']['historical_uniform_heads'] if (r['seed'],r['arm'],r['fold'])==(model['seed'],model['arm'],fold))
   with Path(ref['archive']['path']).open('xb') as f:np.savez(f,**arrays)
for ref in cfg['inputs']['historical_uniform_heads']:ref['archive']=a.d.p.pin(ref['archive']['path'])
Path(CONFIG).write_text(json.dumps(cfg))
'''
    main='''import json
from pathlib import Path
from experiments import run_libero_task7_residual_harm_protection as m
cfg=json.loads(Path(CONFIG).read_text());p={'path':'synthetic-only','sha256':'a'*64}
r=m.train_payload(cfg,p,p)
assert r['roles']['counts']['optimizer_steps']==18000
assert r['access']['array_file_loads']=={'attempted':7,'completed':7}
assert sum(t['rows'] for t in r['tables'].values())==191994
try:m.train_payload(cfg,p,p)
except ValueError:pass
else:raise AssertionError('retry allowed')
'''
    independent='''import json,sys
from pathlib import Path
from experiments import audit_libero_task7_residual_harm_protection as a
cfg=json.loads(Path(CONFIG).read_text());p={'path':'synthetic-only','sha256':'a'*64}
assert 'torch' not in sys.modules
r=a.audit_payload(cfg,p,p)
assert r['access']['array_file_loads']=={'attempted':61,'completed':61}
assert r['access']['array_member_decodes']=={'attempted':1241,'completed':1241}
assert len(r['historical_parity'])==18 and all(z['bit_exact'] for z in r['historical_parity'])
assert r['computation']['training_trace_batches']==18000
assert len(r['model_flags'])==6 and len(r['arm_flags'])==2
assert 'torch' not in sys.modules
Path(PROOF).write_text(json.dumps({'checks':r['scalar_checks'],'cost':r['cost'],'cue':[z['joint_cue'] for z in r['arm_flags']]}))
'''
    env=os.environ.copy();env.update(d.EXECUTION['environment'])
    for name,code in [('prepare_history',prepare),('training',main),('audit',independent)]:
        command=code.replace('CONFIG',repr(str(config))).replace('PROOF',repr(str(directory/'synthetic-proof.json')))
        r=subprocess.run([sys.executable,'-c',command],cwd=d.ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=195)
        (tmp_path/(name+'.log')).write_text(r.stdout)
        assert r.returncode==0,name+'\n'+r.stdout
    proof=json.loads((directory/'synthetic-proof.json').read_text());assert proof['checks']>8000000
