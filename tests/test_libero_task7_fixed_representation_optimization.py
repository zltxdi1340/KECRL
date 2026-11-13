"""Synthetic numerical paths, scoped access and upper-envelope resources."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments import libero_task7_fixed_representation_optimization_protocol as d
from experiments import libero_task7_fixed_representation_optimization_access as a
from experiments import run_libero_task7_fixed_representation_optimization as main
from experiments import audit_libero_task7_fixed_representation_optimization as audit


@pytest.fixture
def cfg():return d.create()


@pytest.mark.parametrize('coefficient',[0.,1.])
@pytest.mark.parametrize('theta',[[0.,0.,0.],[.0625,-.03125,.03125],[2.,-2.,.5]])
def test_analytic_state_matches_independent_autograd(coefficient,theta):
    z=np.array([[1.,0.],[0.,1.],[1.,1.],[-1.,0.]],np.float64);y=np.zeros((4,7));y[:,2]=[.5,-.25,.25,.5];p0=np.zeros_like(y)
    context=tuple(a.readonly(x) for x in (z,y,p0));theta=np.asarray(theta);rows=np.arange(4,dtype=np.int32)
    left=main.measure(context,theta,coefficient,rows);right=audit.measure(audit.TorchFit(context),theta,coefficient,rows)
    audit.deep_check(audit.Checks(),left,right,'synthetic state')
    lt=main.trial_objective(context,theta,coefficient,left['fixed_other6_floor']);rt=audit.trial_objective(audit.TorchFit(context),theta,coefficient,right['fixed_other6_floor'])
    assert np.allclose(lt,rt,atol=1e-10,rtol=0)


def test_hinge_selected_subgradient_ascent_stops_before_trial():
    z=np.zeros((2,1));y=np.zeros((2,7));y[:,2]=[1.,-.75];p0=np.zeros_like(y)
    row=main.measure((z,y,p0),np.zeros(2),1.,np.arange(2,dtype=np.int32))
    assert row['total_l2']>0 and row['directional_derivative']>0
    assert d.stopping_reason(0,row['total_l2'],row['directional_derivative'])=='non_descent_selected_direction'


def registered_job(cfg,feature=0.):
    model=cfg['inputs']['models'][0];plan=cfg['inputs']['folds'][0];rows=np.asarray(plan['fit_rows'],np.int32)
    jobs=a.Jobs(cfg,a.base.Limits(300,2**40));work=a.RepresentationWork(cfg)
    z=np.full((len(rows),256),feature,np.float64);y=np.zeros((4188,7),np.float32);y[:,2]=.5;p0=np.zeros_like(y)
    jobs.array_sources[id(y)]=(str(Path(cfg['inputs']['shared']['path']).resolve()),'targets')
    jobs.array_sources[id(p0)]=(str(Path(model['current_archive']['path']).resolve()),'predictions')
    rkey=model['seed'],model['arm'],plan['fold'];work.begin(rkey,len(rows));work.built(rkey,z)
    context=jobs.register_fit(model,plan,z,y,p0,rows,work);key=rkey+('uniform_affine',);start=np.zeros(257)
    jobs.head(key,*context,rows,start);return jobs,context,rows,start


def test_first_acceptable_trial_and_fixed_final_endpoint(cfg):
    jobs,context,rows,start=registered_job(cfg,feature=1.);states=[];trials=[]
    initial,final,theta,reason=main.optimize(context,start,0.,rows,jobs,lambda k,t,r:states.append((k,r['trainable_objective'])),trials.append)
    first=[r for r in trials if r['from_extra_updates']==0]
    assert [r['trial_index'] for r in first]==list(range(7))
    assert not any(r['accepted'] for r in first[:-1]) and first[-1]['accepted']
    assert all(b[1]<aa[1] for aa,b in zip(states,states[1:]))
    assert final['trainable_objective']<=initial['trainable_objective'] and np.array_equal(start,np.zeros(257))
    assert len(states)==jobs.counts['updates']+1 and jobs.counts['updates']<=50
    assert np.array_equal(theta,jobs.last_finished[2]) and reason in d.OPTIMIZATION['stop_priority']


@pytest.mark.parametrize('fault',['commit_without_accept','wrong_trial','extra_trial','gradient_outside','duplicate_state','foreign_context','mutated_context'])
def test_work_guard_rejects_illegal_path(cfg,fault):
    jobs,context,rows,start=registered_job(cfg);row=main.measure(context,jobs.current,0.,rows,jobs)
    with pytest.raises(ValueError):
        if fault=='commit_without_accept':jobs.commit(start)
        elif fault=='wrong_trial':jobs.trial_begin(0,a.readonly(start+.1))
        elif fault=='extra_trial':jobs.trial_begin(12,a.readonly(start-np.asarray(row['total_gradient'])))
        elif fault=='gradient_outside':jobs.gradient()
        elif fault=='duplicate_state':main.measure(context,jobs.current,0.,rows,jobs)
        elif fault=='foreign_context':main.measure(tuple(x.copy() for x in context),jobs.current,0.,rows,jobs)
        else:
            context[0].flags.writeable=True;context[0][0,0]=2
            # A valid terminal state is required before its source digest check.
            jobs.last['total_l2']=0.;jobs.last['directional_derivative']=0.;jobs.finish('zero_gradient')


def test_unbound_payload_apis_fail_before_directory_or_source_io(cfg,tmp_path):
    science=tmp_path/'science.json';science.write_text(json.dumps(cfg));sp=d.p.pin(science);bp={'path':str(tmp_path/'missing.json'),'sha256':'0'*64}
    before=Path(cfg['output']).exists(),Path(cfg['artifacts']).exists()
    with pytest.raises(FileNotFoundError):main.execute_payload(cfg,sp,bp)
    with pytest.raises(FileNotFoundError):audit.audit_payload(cfg,sp,bp)
    assert (Path(cfg['output']).exists(),Path(cfg['artifacts']).exists())==before


@pytest.mark.parametrize('fault',['current_during_state','current_during_trial','trial_during_forward'])
def test_readout_parameters_cannot_mutate_silently(cfg,fault):
    jobs,context,rows,start=registered_job(cfg)
    if fault=='current_during_state':
        jobs.state_begin(jobs.current);jobs.linear(len(rows))
        for _ in range(3):jobs.gradient()
        jobs.linear(len(rows));jobs.current.flags.writeable=True;jobs.current[0]=1;jobs.current.flags.writeable=False
        with pytest.raises(ValueError):jobs.state_end({})
        return
    row=main.measure(context,jobs.current,0.,rows,jobs);trial=a.readonly(jobs.current+np.asarray(row['direction']))
    jobs.trial_begin(0,trial);jobs.linear(len(rows))
    target=jobs.current if fault=='current_during_trial' else trial
    target.flags.writeable=True;target[0]+=1;target.flags.writeable=False
    with pytest.raises(ValueError):jobs.trial_end(row['trainable_objective']/2)


def test_source_role_registration_rejects_wrong_model_predictions(cfg):
    jobs,context,rows,start=registered_job(cfg);model=cfg['inputs']['models'][1];plan=cfg['inputs']['folds'][0]
    jobs.active=None;y=np.zeros((4188,7),np.float32);p0=np.zeros_like(y)
    jobs.array_sources[id(y)]=(str(Path(cfg['inputs']['shared']['path']).resolve()),'targets')
    jobs.array_sources[id(p0)]=(str(Path(cfg['inputs']['models'][0]['current_archive']['path']).resolve()),'predictions')
    work=a.RepresentationWork(cfg);rkey=(model['seed'],model['arm'],plan['fold']);z=np.zeros((len(rows),256));work.begin(rkey,len(rows));work.built(rkey,z)
    with pytest.raises(ValueError):jobs.register_fit(model,plan,z,y,p0,rows,work)


def small_archives(cfg,tmp_path):
    cfg=copy.deepcopy(cfg)
    owners=[(cfg['inputs'],'shared','shared')]+[(m,'current_archive','current') for m in cfg['inputs']['models']]+[(h,'archive','head') for h in cfg['inputs']['heads']]
    for i,(owner,field,kind) in enumerate(owners):
        path=tmp_path/f'source-{i}.npz';arrays={k:np.zeros(1,np.float32) for k in a.members(kind)};arrays['forbidden_extra']=np.zeros(1)
        np.savez(path,**arrays);owner[field]=d.p.pin(path)
    return cfg


@pytest.mark.parametrize('operation',['raw','write','extra','repeat','mmap','zip','zip_write','os_open','fileio','unlisted','torch','optimizer','unlisted_save','compressed_save'])
def test_access_guard_rejects_alternate_reads_or_writes(cfg,tmp_path,operation):
    cfg=small_archives(cfg,tmp_path);jobs=a.Jobs(cfg,a.base.Limits(300,2**40));paths=[tmp_path/f'new-{i}.npz' for i in range(36)]
    guard=a.Guard(cfg,[],paths,jobs.limits,jobs);ref=cfg['inputs']['heads'][0]['archive']
    with pytest.raises(PermissionError):
        with guard:
            if operation=='raw':Path(ref['path']).read_bytes()
            elif operation=='write':Path(ref['path']).open('w')
            elif operation=='mmap':np.load(ref['path'],allow_pickle=False,mmap_mode='r')
            elif operation=='os_open':os.open(ref['path'],os.O_RDONLY)
            elif operation=='fileio':__import__('io').FileIO(ref['path'])
            elif operation=='unlisted':Path('/tmp/nonexistent-forbidden-array.npz').read_bytes()
            elif operation=='torch':__import__('torch')
            elif operation=='optimizer':__import__('torch').optim.Adam([])
            elif operation=='unlisted_save':np.savez(tmp_path/'unlisted.npz',x=np.zeros(1))
            elif operation=='compressed_save':np.savez_compressed(paths[0],x=np.zeros(1))
            elif operation=='zip':
                with np.load(ref['path'],allow_pickle=False) as z:z.zip.open('weights.npy')
            elif operation=='zip_write':
                with np.load(ref['path'],allow_pickle=False) as z:z.zip.open('weights.npy','w')
            else:
                with np.load(ref['path'],allow_pickle=False) as z:
                    if operation=='extra':z['forbidden_extra']
                    else:z['weights'];z['weights']
    assert guard.denied


def synthetic_config(cfg,tmp_path):
    path=Path(__file__).with_name('test_libero_task7_endpoint_gradient_diagnostic.py')
    spec=importlib.util.spec_from_file_location('endpoint_synthetic_helpers',path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cfg=helper.synthetic_inputs(cfg,tmp_path)
    with np.load(cfg['inputs']['shared']['path'],allow_pickle=False) as shared:y=shared['targets']
    references=[]
    for model in cfg['inputs']['models']:
        with np.load(model['current_archive']['path'],allow_pickle=False) as cache:f,p0=cache['features'],cache['predictions']
        for plan in cfg['inputs']['folds']:
            rows=np.asarray(plan['fit_rows'],np.int32)
            for mode,coefficient in d.CASES.items():
                h=next(h for h in cfg['inputs']['heads'] if a.key(h)==(model['seed'],model['arm'],plan['fold'],mode))
                with np.load(h['archive']['path'],allow_pickle=False) as archive:head={k:archive[k] for k in d.HEAD_MEMBERS}
                z=main.admit(head,f,plan);context=z,y[rows].astype(np.float64),p0[rows].astype(np.float64)
                theta=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];point=main.measure(context,theta,coefficient,rows)
                references.append({**a.identity(h),'n':len(rows),'start500_total_objective':point['total_objective'],'start500_trainable_objective':point['trainable_objective'],
                    'start500_gradient_l2':point['total_l2'],'zero_residual_gradient_l2':point['total_l2'],'start500_branch_digest':point['branch_digest']})
    cfg['reference_points']=references;return cfg


def test_full_synthetic_finite_optimization_and_independent_replay(cfg,tmp_path):
    cfg=synthetic_config(cfg,tmp_path);path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
    for i,(module,entry,operation) in enumerate([
        ('run_libero_task7_fixed_representation_optimization','execute_payload','admit'),
        ('audit_libero_task7_fixed_representation_optimization','audit_payload','prepare')]):
        code=f'''import json
from pathlib import Path
from experiments import libero_task7_fixed_representation_optimization_access as a
from experiments import {module} as module
cfg=json.load(open({str(path)!r}))
def synthetic_only(cfg,sp,bp):
    refs=[cfg['inputs']['shared']]+[m['current_archive'] for m in cfg['inputs']['models']]+[h['archive'] for h in cfg['inputs']['heads']]
    assert all(Path(r['path']).is_relative_to(Path({str(tmp_path)!r})) for r in refs)
a.validate_call=synthetic_only
calls=[]
original=module.{operation}
def counted(*args,**kwargs):
    calls.append(len(args[2]['fit_rows']))
    return original(*args,**kwargs)
module.{operation}=counted
pin={{'path':'synthetic-only','sha256':{'a'*64!r}}}
r=module.{entry}(cfg,pin,pin)
assert len(calls)==18 and sum(calls)==50256
assert r['representation_work']['scaler_reconstructions']==18
'''
        proc=subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=630)
        (tmp_path/f'child-{i}.log').write_text(proc.stdout);assert proc.returncode==0,proc.stdout
    record=json.loads(Path(cfg['output']).read_text());done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
    assert done['total_NPZ_loads']==122 and done['total_member_decodes']==862 and done['total_archive_SHA_checks']==280
    assert len(record['new_heads'])==36 and len(record['endpoints'])==72
    assert record['computation']['gradient_states']==36+record['computation']['updates']
    assert record['computation']['updates']<=1800 and record['computation']['trial_forwards']<=21600


def test_upper_envelope_synthetic_resource_budget(cfg,tmp_path):
    """Worst-volume numerical envelope, separate from algorithm-path evidence."""
    path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
    for stage in ['primary','audit']:
        proof=tmp_path/(stage+'-envelope.json')
        code=f'''import json
from pathlib import Path
import numpy as np
from experiments import libero_task7_fixed_representation_optimization_access as a
cfg=json.load(open({str(path)!r}))
stage={stage!r}
if stage=='primary':
    from experiments import run_libero_task7_fixed_representation_optimization as module
else:
    from experiments import audit_libero_task7_fixed_representation_optimization as module
runtime=a.runtime(cfg,audit=stage=='audit')
limits=a.base.Limits(300 if stage=='primary' else 600,536870912 if stage=='primary' else 805306368)
calls=[0]
if stage=='audit':
    import torch
    original=torch.autograd.grad
    def counted(*args,**kwargs):
        calls[0]+=1
        return original(*args,**kwargs)
    torch.autograd.grad=counted
states=trials=state_rows=trial_rows=0
for i in range(36):
    n=[2782,2817,2777][(i//2)%3]
    rng=np.random.default_rng(20261109+i)
    z=rng.normal(0,.2,(n,256));y=np.zeros((n,7));y[:,2]=np.resize(np.array([.5,-.25,.25]),n);p0=np.zeros_like(y)
    context=tuple(a.readonly(v) for v in (z,y,p0));backend=context if stage=='primary' else module.TorchFit(context)
    theta=np.linspace(-.0005,.0005,257);rows=np.arange(n,dtype=np.int32);coefficient=float(i%2)
    for k in range(51):
        row=module.measure(backend,theta,coefficient,rows);states+=1;state_rows+=n
        assert len(row['total_gradient'])==257
        if k<50:
            for j in range(12):
                trial=theta+2.**(-j)*np.asarray(row['direction'])
                module.trial_objective(backend,trial,coefficient,row['fixed_other6_floor']);trials+=1;trial_rows+=n
    limits.check()
    print(stage,i,flush=True)
assert states==1836 and trials==21600 and state_rows==5126112 and trial_rows==60307200
assert calls[0]==(5508 if stage=='audit' else 0)
proof={{'synthetic_only':True,'resource_envelope_not_algorithm_result':True,'runtime':runtime,'cost':limits.record(),'gradient_states':states,'trial_forwards':trials,'state_fit_rows':state_rows,'trial_fit_rows':trial_rows,'actual_autograd_calls':calls[0],'real_archive_SHA_or_decode_or_real_updates':0}}
Path({str(proof)!r}).write_text(json.dumps(proof,indent=2))
'''
        proc=subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=630)
        (tmp_path/(stage+'-envelope.log')).write_text(proc.stdout);assert proc.returncode==0,proc.stdout
        result=json.loads(proof.read_text());assert result['gradient_states']==1836 and result['trial_forwards']==21600
