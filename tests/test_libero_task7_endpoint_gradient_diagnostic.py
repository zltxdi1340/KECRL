"""Only synthetic arrays/scalars; real archives remain closed before binding."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments import libero_task7_endpoint_gradient_protocol as d
from experiments import libero_task7_endpoint_gradient_access as a
from experiments import run_libero_task7_endpoint_gradient_diagnostic as main
from experiments import audit_libero_task7_endpoint_gradient_diagnostic as audit


@pytest.fixture
def cfg():return d.create()


@pytest.mark.parametrize('coefficient',[0.,1.])
def test_analytic_gradient_against_independent_autograd(coefficient):
    z=np.array([[1.,0.],[0.,1.],[1.,1.],[-1.,0.]],np.float64);y=np.zeros((4,7));y[:,2]=[.5,-.25,.25,.5];p=np.zeros_like(y)
    for w,b in [(np.zeros(2),0.),(np.array([.0625,-.03125]),.03125),(np.array([2.,-2.]),.5)]:
        left=main.measure(z,y,p,w,b,coefficient,np.arange(4));right=audit.autograd_measure(z,y,p,w,b,coefficient,np.arange(4))
        audit.deep_check(audit.Checks(),left,right,'synthetic point')
        assert len(left['total_gradient'])==3 and abs(left['fixed_other6_floor'])==0


def test_job_gate_and_exact_zero_updates(cfg):
    jobs=a.Jobs(cfg);key=next(iter(jobs.expected))
    with pytest.raises(ValueError):jobs.gradient()
    jobs.begin(key,2782)
    with pytest.raises(ValueError):jobs.linear(1)
    jobs.linear(2782)
    for _ in range(3):jobs.gradient()
    jobs.linear(2782);jobs.end()
    with pytest.raises(ValueError):jobs.begin(key,2782)
    with pytest.raises(ValueError):jobs.complete()


def small_archives(cfg,tmp_path):
    cfg=copy.deepcopy(cfg)
    pairs=[(cfg['inputs'],'shared','shared')]+[(m,'current_archive','current') for m in cfg['inputs']['models']]+[(h,'archive','head') for h in cfg['inputs']['heads']]
    for i,(owner,key,kind) in enumerate(pairs):
        fields=d.HEAD_MEMBERS if kind=='head' else d.SOURCE_MEMBERS[kind]
        path=tmp_path/f'archive-{i}.npz';arrays={k:np.zeros(1,np.float32) for k in fields};arrays['forbidden_extra']=np.ones(1)
        np.savez(path,**arrays);owner[key]=d.p.pin(path)
    return cfg


def test_full_member_ledger_without_extra_fields(cfg,tmp_path):
    cfg=small_archives(cfg,tmp_path);guard=a.Guard(cfg,limits=a.base.Limits(180,2**40))
    with guard:
        a.load(cfg['inputs']['shared'],'shared')
        for m in cfg['inputs']['models']:a.load(m['current_archive'],'current')
        for h in cfg['inputs']['heads']:a.load(h['archive'],'head')
        guard.complete()
    assert guard.record()['member_decodes']=={'attempted':305,'completed':305}
    assert len(guard.record()['byte_SHA_checks'])==86


@pytest.mark.parametrize('operation',['raw','write','extra','repeat','mmap','os_open','fileio','torch','optimizer'])
def test_guard_denies_unlisted_access(cfg,tmp_path,operation):
    cfg=small_archives(cfg,tmp_path);ref=cfg['inputs']['heads'][0]['archive'];guard=a.Guard(cfg,limits=a.base.Limits(180,2**40))
    with pytest.raises(PermissionError):
        with guard:
            if operation=='raw':Path(ref['path']).read_bytes()
            elif operation=='write':Path(ref['path']).open('w')
            elif operation=='mmap':np.load(ref['path'],allow_pickle=False,mmap_mode='r')
            elif operation=='os_open':os.open(ref['path'],os.O_RDONLY)
            elif operation=='fileio':__import__('io').FileIO(ref['path'])
            elif operation=='torch':__import__('torch')
            elif operation=='optimizer':
                # The import itself is denied in primary mode.
                __import__('torch').optim.Adam([])
            else:
                with np.load(ref['path'],allow_pickle=False) as z:
                    if operation=='extra':z['forbidden_extra']
                    else:z['weights'];z['weights']
    assert guard.denied


def test_fit_rows_and_scaler_cannot_include_probe(cfg):
    plan=cfg['inputs']['folds'][0];f=np.zeros((4188,256),np.float32)
    head={'fit_rows':np.array(plan['fit_rows'],np.int32),'fit_demo_indices':np.array(plan['fit_demos'],np.int32),'updates':np.array([0,100,250,500],np.int32),
          'optimizer_steps':np.arange(1,501,dtype=np.int32),'scaler_mean':np.zeros(256),'scaler_scale':np.ones(256),'weights':np.zeros((4,256),np.float32),'bias':np.zeros(4,np.float32)}
    x=main.admit(head,f,plan);f[plan['probe_rows']]=1e5
    assert np.array_equal(x,main.admit(head,f,plan))
    head['fit_rows'][0]=plan['probe_rows'][0]
    with pytest.raises(ValueError):main.admit(head,f,plan)


def synthetic_inputs(cfg,tmp_path):
    cfg=copy.deepcopy(cfg);lengths=cfg['inputs']['action_lengths'];pairs=[(demo,t) for demo,n in enumerate(lengths) for t in range(1,n)]
    y=np.zeros((4188,7),np.float32);y[:,2]=np.resize(np.array([.5,-.25,.25],np.float32),4188)
    demos=np.array([d for d,t in pairs],np.int32);steps=np.array([t for d,t in pairs],np.int32)
    shared={'targets':y,'demo_indices':demos,'action_steps':steps,'demo_lengths':np.array(lengths,np.int32),'tail20':steps>=np.asarray(lengths)[demos]-20}
    p=tmp_path/'shared.npz';np.savez(p,**shared);cfg['inputs']['shared']=d.p.pin(p)
    features_by_model={}
    for i,m in enumerate(cfg['inputs']['models']):
        p=tmp_path/f'current-{i}.npz';pred=np.zeros((4188,7),np.float32);pred[:,0]=.125
        feature=np.random.default_rng(20261109+i).normal(0,.2,(4188,256)).astype(np.float32);features_by_model[m['seed'],m['arm']]=feature
        np.savez(p,features=feature,predictions=pred);m['current_archive']=d.p.pin(p)
    for i,h in enumerate(cfg['inputs']['heads']):
        plan=cfg['inputs']['folds'][h['fold']];p=tmp_path/f'head-{i}.npz'
        raw=features_by_model[h['seed'],h['arm']][plan['fit_rows']].astype(np.float64);mean=raw.mean(0);scale=raw.std(0);scale[scale<=1e-6]=1
        weight=np.zeros((4,256),np.float32);weight[3]=np.linspace(-.0005,.0005,256,dtype=np.float32)
        np.savez(p,fit_rows=np.array(plan['fit_rows'],np.int32),fit_demo_indices=np.array(plan['fit_demos'],np.int32),scaler_mean=mean,scaler_scale=scale,updates=np.array([0,100,250,500],np.int32),
            weights=weight,bias=np.array([0,10,-10,.0625 if h['mode']=='uniform_affine' else .03125],np.float32),optimizer_steps=np.arange(1,501,dtype=np.int32))
        h['archive']=d.p.pin(p)
    cfg['output']=str(tmp_path/'result.json');cfg['artifacts']=str(tmp_path/'artifacts');return cfg


def test_full_synthetic_fixed72points_and_fresh_autograd_audit(cfg,tmp_path):
    cfg=synthetic_inputs(cfg,tmp_path);path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
    snippets=['from experiments.run_libero_task7_endpoint_gradient_diagnostic import execute_payload;execute_payload(cfg,pin,pin)',
        'from experiments.audit_libero_task7_endpoint_gradient_diagnostic import audit_payload;audit_payload(cfg,pin,pin)']
    for i,snippet in enumerate(snippets):
        code='import json;cfg=json.load(open('+repr(str(path))+'));pin={"path":"synthetic-only","sha256":"'+'a'*64+'"};'+snippet
        proc=subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=195)
        (tmp_path/f'child-{i}.log').write_text(proc.stdout);assert proc.returncode==0,proc.stdout
    r=json.loads(Path(cfg['output']).read_text());done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
    assert done['total_NPZ_loads']==86 and done['total_member_decodes']==610 and len(r['points'])==72
    assert r['computation']=={'point_jobs':72,'fit_row_visits':201024,'linear_forms':144,'gradient_vectors':216,'autograd_calls':0}
    assert len(r['pairs'])==36 and len(r['seed_summaries'])==12 and len(r['arm_summaries'])==4
