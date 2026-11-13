"""Synthetic-only paired paths, archive isolation, parity and resource envelope."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments import libero_task7_joint_direction_protocol as d
from experiments import libero_task7_joint_direction_access as a
from experiments import run_libero_task7_joint_direction_diagnostic as main
from experiments import audit_libero_task7_joint_direction_diagnostic as audit


@pytest.fixture(scope='module')
def canonical():return d.create()

@pytest.fixture
def cfg(canonical):return copy.deepcopy(canonical)


def registered_job(cfg,opt=d.ARMS[0],bias=.1,feature=0.,balanced=True):
    model=cfg['inputs']['models'][0];plan=cfg['inputs']['folds'][0];spec=cfg['inputs']['heads'][0]
    rows=np.asarray(plan['fit_rows'],np.int32);jobs=a.Jobs(cfg,a.base.Limits(300,2**40));work=a.RepresentationWork(cfg)
    z=np.zeros((len(rows),256));y=np.zeros((4188,7),np.float32);p0=np.zeros_like(y)
    z[:,0]=feature
    # Balanced own-fit batch creates a real MSE/penalty gradient tradeoff.
    if balanced:y[rows,2]=np.resize(np.array([1.,-.75],np.float32),len(rows))
    jobs.array_sources[id(y)]=(str(Path(cfg['inputs']['shared']['path']).resolve()),'targets')
    jobs.array_sources[id(p0)]=(str(Path(model['current_archive']['path']).resolve()),'predictions')
    rkey=(model['seed'],model['arm'],plan['fold']);work.begin(rkey,len(rows));work.built(rkey,z)
    context=jobs.register_fit(model,plan,z,y,p0,rows,work)
    weights=np.zeros((4,256),np.float32);biases=np.array([0,9,-9,bias],np.float32)
    head={f:np.zeros(1,np.float64) for f in d.finer.HEAD_MEMBERS}
    head.update(weights=weights,bias=biases,scaler_mean=np.zeros(256),scaler_scale=np.ones(256))
    for f,v in head.items():jobs.array_sources[id(v)]=(str(Path(spec['archive']['path']).resolve()),f)
    start=np.r_[weights[3].astype(np.float64),float(biases[3])];jobs.source(spec,head,start)
    if opt==d.ARMS[1]:
        jobs.head(a.key(spec)+(d.ARMS[0],),*context,rows,start)
        main.optimize(context,start,rows,jobs,lambda *args:None,lambda row:None)
    jobs.head(a.key(spec)+(opt,),*context,rows,start)
    return jobs,context,rows,start


@pytest.mark.parametrize('theta',[[0.,0.,0.],[.0625,-.03125,.1],[2.,-2.,.5]])
def test_independent_gradient_and_trial_MSE_penalty_agree(theta):
    z=np.zeros((4,256));z[:,:2]=[[1.,0.],[0.,1.],[1.,1.],[-1.,0.]]
    y=np.zeros((4,7));y[:,2]=[.5,-.25,.25,.5];p0=np.zeros_like(y)
    context=tuple(a.readonly(v) for v in (z,y,p0));rows=np.arange(4,dtype=np.int32);theta=np.r_[np.asarray(theta)[:-1],np.zeros(254),theta[-1]]
    l=main.measure(context,theta,1.,rows);r=audit.measure(audit.TorchFit(context),theta,1.,rows)
    audit.deep_check(audit.Checks(),l,r,'synthetic state')
    audit.deep_check(audit.Checks(),main.trial_objective(context,theta,1.,l['fixed_other6_floor']),
        audit.trial_objective(audit.TorchFit(context),theta,1.,r['fixed_other6_floor']),'synthetic trial')


@pytest.mark.parametrize('fault',['commit_without_accept','wrong_trial','extra_trial','extra_gradient','duplicate_state','foreign_context','source_start','candidate_first','reference','terms'])
def test_paired_work_guard_rejects_invalid_jobs(cfg,fault):
    jobs,context,rows,start=registered_job(cfg)
    row=main.measure(context,jobs.current,1.,rows,jobs)
    with pytest.raises(ValueError):
        if fault=='commit_without_accept':jobs.commit(start)
        elif fault=='wrong_trial':jobs.trial_begin(0,a.readonly(start+.1))
        elif fault=='extra_trial':jobs.trial_begin(20,a.readonly(start-np.asarray(row['total_gradient'])))
        elif fault=='extra_gradient':jobs.gradient()
        elif fault=='duplicate_state':main.measure(context,jobs.current,1.,rows,jobs)
        elif fault=='foreign_context':main.measure(tuple(v.copy() for v in context),jobs.current,1.,rows,jobs)
        elif fault in ('source_start','candidate_first'):
            jobs.active=None
            jobs.head(a.key(cfg['inputs']['heads'][0])+((d.ARMS[1] if fault=='candidate_first' else d.ARMS[0]),),*context,rows,start+.1 if fault=='source_start' else start)
        elif fault=='reference':
            jobs.fixed_mse+=.001;jobs.trial_begin(0,a.readonly(start+np.asarray(row['direction'])))
        else:
            trial=a.readonly(start+np.asarray(row['direction']));jobs.trial_begin(0,trial);jobs.linear(len(rows))
            jobs.trial_end(dict(component3_mse=.1,penalty=0.,trainable_objective=1.,total_objective=1.))


def test_source_origin_cannot_be_forged(cfg):
    jobs,context,rows,start=registered_job(cfg);jobs.active=None
    h=cfg['inputs']['heads'][1];head={f:np.zeros(1) for f in d.finer.HEAD_MEMBERS}
    with pytest.raises(ValueError):jobs.source(h,head,start)


def test_unbound_payload_APIs_reject_before_source_or_directory(cfg,tmp_path):
    path=tmp_path/'science.json';path.write_text(json.dumps(cfg));sp=d.p.pin(path);bp={'path':str(tmp_path/'absent.json'),'sha256':'0'*64}
    before=Path(cfg['output']).exists(),Path(cfg['artifacts']).exists()
    with pytest.raises(FileNotFoundError):main.execute_payload(cfg,sp,bp)
    with pytest.raises(FileNotFoundError):audit.audit_payload(cfg,sp,bp)
    assert before==(Path(cfg['output']).exists(),Path(cfg['artifacts']).exists())


def small_archives(cfg,tmp_path):
    cfg=copy.deepcopy(cfg)
    owners=[(cfg['inputs'],'shared','shared')]+[(m,'current_archive','current') for m in cfg['inputs']['models']]+[(h,'archive','head') for h in cfg['inputs']['heads']]
    for i,(owner,field,kind) in enumerate(owners):
        path=tmp_path/f'input-{i}.npz';np.savez(path,**{f:np.zeros(1) for f in a.members(kind)},forbidden_extra=np.zeros(1))
        owner[field]=d.p.pin(path)
    return cfg


@pytest.mark.parametrize('operation',['raw','write','member','repeat','mmap','zip','zip_write','os_open','fileio','unlisted','torch','optimizer','save','compressed_save','lambda0','old_optimized'])
def test_archive_guard_denies_bypass_and_unlisted_roles(cfg,tmp_path,operation):
    cfg=small_archives(cfg,tmp_path);jobs=a.Jobs(cfg,a.base.Limits(300,2**40));newpaths=[tmp_path/f'new-{i}.npz' for i in range(18)]
    guard=a.Guard(cfg,[],newpaths,jobs.limits,jobs);ref=cfg['inputs']['heads'][0]['archive']
    with pytest.raises(PermissionError):
        with guard:
            if operation=='raw':Path(ref['path']).read_bytes()
            elif operation=='write':Path(ref['path']).open('w')
            elif operation=='mmap':np.load(ref['path'],allow_pickle=False,mmap_mode='r')
            elif operation=='os_open':os.open(ref['path'],os.O_RDONLY)
            elif operation=='fileio':__import__('io').FileIO(ref['path'])
            elif operation in ('unlisted','lambda0','old_optimized'):Path(tmp_path/(operation+'.npz')).read_bytes()
            elif operation=='torch':__import__('torch')
            elif operation=='optimizer':__import__('torch').optim.Adam([])
            elif operation=='save':np.savez(tmp_path/'unlisted.npz',x=np.zeros(1))
            elif operation=='compressed_save':np.savez_compressed(newpaths[0],x=np.zeros(1))
            elif operation in ('zip','zip_write'):
                with np.load(ref['path'],allow_pickle=False) as z:z.zip.open('weights.npy','w' if operation=='zip_write' else 'r')
            else:
                with np.load(ref['path'],allow_pickle=False) as z:
                    if operation=='member':z['forbidden_extra']
                    else:z['weights'];z['weights']
    assert guard.denied


def test_history_projects_only_guarded_scalars_and_normalizes_role(cfg,tmp_path):
    projections=cfg['historical_control']['projections']
    raw={name:[] for name in projections}
    row={f:0. for f in projections['states']}
    row.update(a.identity(cfg['inputs']['heads'][0]),optimization_arm='fine20',extra_updates=0)
    row['parameters']=['must-not-enter-projection'];row['parameter_digest']='unread'
    raw['states']=[row,dict(row,optimization_arm='coarse12')]
    path=tmp_path/'history.json';path.write_text(json.dumps(raw))
    cfg['historical_control']['source_JSON']=d.p.pin(path)
    cfg['budget']['historical_control_JSON_projected_records_per_stage']={k:int(k=='states') for k in raw}
    history=a.History(cfg)
    assert len(history.data['states'])==1 and 'parameters' not in history.data['states'][0]
    actual=dict(row,optimization_arm=d.ARMS[0]);bad=dict(actual,component3_mse=1.)
    with pytest.raises(ValueError):history.row('states',bad)
    history.row('states',actual);history.complete()


def synthetic_config(cfg,tmp_path):
    # Non-collinear fit gradients, rather than a zero-target fixture where both
    # arms are mathematically identical and strict endpoint ordering is a tie.
    helper_path=Path(__file__).with_name('test_libero_task7_endpoint_gradient_diagnostic.py')
    spec=importlib.util.spec_from_file_location('endpoint_synthetic_helpers',helper_path);helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    from experiments import libero_task7_mse_guard_finer_grid_access as old_access
    from experiments import run_libero_task7_mse_guard_finer_grid as old_main
    cfg=helper.synthetic_inputs(cfg,tmp_path);raw={k:[] for k in cfg['historical_control']['projections']};refs=[]
    with np.load(cfg['inputs']['shared']['path'],allow_pickle=False) as archive:y=archive['targets']
    for model in cfg['inputs']['models']:
        with np.load(model['current_archive']['path'],allow_pickle=False) as cache:f,p0=cache['features'],cache['predictions']
        for plan in cfg['inputs']['folds']:
            h=next(h for h in cfg['inputs']['heads'] if a.key(h)==(model['seed'],model['arm'],plan['fold'],d.finer.source.MODE))
            with np.load(h['archive']['path'],allow_pickle=False) as archive:head={k:archive[k] for k in d.finer.HEAD_MEMBERS}
            rows=np.asarray(plan['fit_rows'],np.int32);jobs=old_access.Jobs(cfg,a.base.Limits(300,2**40));work=old_access.RepresentationWork(cfg)
            rkey=(model['seed'],model['arm'],plan['fold']);z=old_main.admit(head,f,plan);work.begin(rkey,len(rows));work.built(rkey,z)
            jobs.array_sources[id(y)]=(str(Path(cfg['inputs']['shared']['path']).resolve()),'targets')
            jobs.array_sources[id(p0)]=(str(Path(model['current_archive']['path']).resolve()),'predictions')
            context=jobs.register_fit(model,plan,z,y,p0,rows,work)
            for field,value in head.items():jobs.array_sources[id(value)]=(str(Path(h['archive']['path']).resolve()),field)
            start=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];jobs.source(h,head,start);ident=a.identity(h)
            for opt in old_main.d.ARMS:
                pathkey=a.key(h)+(opt,);jobs.head(pathkey,*context,rows,start)
                own_trials=[]
                def state(k,theta,point):
                    if opt=='fine20':raw['states'].append({**ident,'optimization_arm':'fine20','extra_updates':k,**point})
                def trial(row):
                    if opt=='fine20':
                        own_trials.append(row);raw['trials'].append({**ident,'optimization_arm':'fine20',**row})
                first,final,theta,reason=old_main.optimize(context,start,rows,jobs,state,trial)
                if opt!='fine20':continue
                updates=jobs.last_finished[3];done=jobs.done[pathkey]
                raw['endpoints'] += [{**ident,'condition':condition,'extra_updates':k,**point} for condition,k,point in [('saved500',0,first),('fine20_final',updates,final)]]
                accepted=[r['alpha'] for r in own_trials if r['accepted']]
                raw['heads'].append(dict(ident,optimization_arm='fine20',extra_updates=updates,trials=done['trials'],terminal_reason=reason,
                    initial_objective=first['total_objective'],final_objective=final['total_objective'],initial_trainable_objective=first['trainable_objective'],final_trainable_objective=final['trainable_objective'],
                    initial_gradient_l2=first['total_l2'],final_gradient_l2=final['total_l2'],initial_component3_mse=first['component3_mse'],final_component3_mse=final['component3_mse'],
                    initial_penalty=first['penalty'],final_penalty=final['penalty'],mse_vetoes_of_armijo_passes=done['mse_vetoes'],
                    alpha_min=min(accepted) if accepted else None,alpha_median=float(np.median(accepted)) if accepted else None,alpha_max=max(accepted) if accepted else None,
                    exhaustion_class=('armijo_feasible_but_mse_vetoed' if done['round_summaries'][-1]['armijo_passes'] else 'no_original_armijo_trial') if reason=='line_search_exhausted' else 'not_exhausted',
                    **d.finer.source.endpoint_evidence(first['trainable_objective'],final['trainable_objective'],first['total_l2'],final['total_l2'],first['component3_mse'],final['component3_mse'],updates,d.finer.source.ARMS[1])))
                refs.append({**ident,'n':len(rows),'start500':{k:first[k] for k in ['component3_mse','penalty','trainable_objective','total_objective','total_l2','branch_digest']}})
                cohort=next(r for r in cfg['cohorts'] if a.key(r)==a.key(h));cohort.update(prior_control_updates=updates,prior_control_stop=reason)
    path=tmp_path/'fine20-history.json';path.write_text(json.dumps(raw));cfg['historical_control']['source_JSON']=d.p.pin(path)
    cfg['reference_points']=refs;cfg['budget']['historical_control_JSON_projected_records_per_stage']={k:len(v) for k,v in raw.items()}
    return cfg


def test_full_synthetic_paired_run_and_fresh_independent_audit(cfg,tmp_path):
    cfg=synthetic_config(cfg,tmp_path);path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
    for index,(module,entry,operation) in enumerate([
        ('run_libero_task7_joint_direction_diagnostic','execute_payload','admit'),
        ('audit_libero_task7_joint_direction_diagnostic','audit_payload','prepare')]):
        code=f'''import json
from pathlib import Path
from experiments import libero_task7_joint_direction_access as a
from experiments import {module} as module
cfg=json.load(open({str(path)!r}))
def synthetic_only(cfg,sp,bp):
    refs=[cfg['inputs']['shared']]+[m['current_archive'] for m in cfg['inputs']['models']]+[h['archive'] for h in cfg['inputs']['heads']]+[cfg['historical_control']['source_JSON']]
    assert all(Path(r['path']).is_relative_to(Path({str(tmp_path)!r})) for r in refs)
a.validate_call=synthetic_only
def forbidden_reference(*args,**kwargs):raise AssertionError('numeric reference must not be called')
module.d.direction=forbidden_reference
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
        (tmp_path/f'child-{index}.log').write_text(proc.stdout);assert proc.returncode==0,proc.stdout
    record=json.loads(Path(cfg['output']).read_text());done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
    assert done['total_NPZ_loads']==68 and done['total_member_decodes']==448 and done['total_archive_SHA_checks']==154
    assert len(record['heads'])==36 and len(record['pairs'])==18 and len(record['endpoints'])==54
    assert record['historical_control_parity']['all18_paths_reproduced']
    assert all(h['final_component3_mse']<=h['initial_component3_mse'] for h in record['heads'] if h['optimization_arm']==d.ARMS[1])


def test_upper_envelope_synthetic_resource_budget(cfg,tmp_path):
    """Worst-volume numerical envelope, separate from algorithm-path evidence."""
    path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
    for stage in ['primary','audit']:
        proof=tmp_path/(stage+'-envelope.json')
        code=f'''import json
from pathlib import Path
import numpy as np
from experiments import libero_task7_joint_direction_access as a
cfg=json.load(open({str(path)!r}))
stage={stage!r}
if stage=='primary':
    from experiments import run_libero_task7_joint_direction_diagnostic as module
else:
    from experiments import audit_libero_task7_joint_direction_diagnostic as module
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
direction_work=a.DirectionWork()
for i in range(36):
    n=[2782,2817,2777][(i//2)%3]
    rng=np.random.default_rng(20261109+i)
    z=rng.normal(0,.2,(n,256));y=np.zeros((n,7));y[:,2]=np.resize(np.array([.5,-.25,.25]),n);p0=np.zeros_like(y)
    context=tuple(a.readonly(v) for v in (z,y,p0));backend=context if stage=='primary' else module.TorchFit(context)
    theta=np.linspace(-.0005,.0005,257);rows=np.arange(n,dtype=np.int32);coefficient=1.
    for k in range(51):
        row=module.measure(backend,theta,coefficient,rows,arm=module.d.ARMS[i%2],direction_work=direction_work);states+=1;state_rows+=n
        assert len(row['total_gradient'])==257
        if k<50:
            for j in range(20):
                trial=theta+2.**(-j)*np.asarray(row['direction'])
                values=module.trial_objective(backend,trial,coefficient,row['fixed_other6_floor']);trials+=1;trial_rows+=n
                assert abs(values['trainable_objective']-values['component3_mse']/7-values['penalty'])<=1e-10
    limits.check()
    print(stage,i,flush=True)
assert states==1836 and trials==36000 and state_rows==5126112 and trial_rows==100512000
assert calls[0]==(5508 if stage=='audit' else 0)
proof={{'synthetic_only':True,'resource_envelope_not_algorithm_result':True,'runtime':runtime,'cost':limits.record(),'gradient_states':states,'trial_forwards':trials,'state_fit_rows':state_rows,'trial_fit_rows':trial_rows,'actual_autograd_calls':calls[0],'direction_work':direction_work.counts,'real_archive_SHA_or_decode_or_real_updates':0}}
Path({str(proof)!r}).write_text(json.dumps(proof,indent=2))
'''
        proc=subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=630)
        (tmp_path/(stage+'-envelope.log')).write_text(proc.stdout);assert proc.returncode==0,proc.stdout
        result=json.loads(proof.read_text());assert result['gradient_states']==1836 and result['trial_forwards']==36000




@pytest.mark.parametrize('total,base',[
 ([1.,0.],[0.,1.]),([2.,0.],[1.,0.]),([1.,0.],[-1.,0.]),([1.,0.],[-1.,1e-13]),([1.,0.],[-1.,1e-5]),
 ([0.,0.],[1.,0.]),([1.,2.],[0.,0.])])
@pytest.mark.parametrize('arm',d.ARMS)
def test_independent_directions_match_frozen_rule_and_count_work(total,base,arm):
    total=total+[0.]*255;base=base+[0.]*255
    wanted=d.direction(arm,total,base)
    for constructor in (main.construct_direction,audit.construct_direction):
        work=a.DirectionWork();actual=constructor(arm,total,base,work)
        audit.deep_check(audit.Checks(),actual,wanted,'direction scalar reference')
        assert work.counts['constructions']==1 and work.counts['dot_reductions']==2
        assert work.counts['norm_reductions']<=4 and work.counts['vector_forms']<=4
        assert work.counts['coordinate_writes']==257*work.counts['vector_forms']


def test_actual_antiparallel_state_is_retained_as_unresolved(cfg):
    jobs,context,rows,start=registered_job(cfg,d.ARMS[1]);states=[];trials=[]
    first,final,theta,reason=main.optimize(context,start,rows,jobs,lambda k,t,r:states.append(r),trials.append)
    assert reason=='unresolved_joint_direction' and jobs.updates==0 and np.array_equal(theta,start) and not trials
    assert states[0]['direction_status']=='unresolved_joint_direction' and states[0]['total_l2']>0


@pytest.mark.parametrize('fault',['norm','dot','vector','outside_state'])
def test_direction_ledger_rejects_excess_or_outside_work(fault):
    if fault=='outside_state':
        cfg=d.create();jobs,context,rows,start=registered_job(cfg)
        with pytest.raises(ValueError):jobs.direction_work.begin(d.ARMS[0],[0.]*257,[0.]*257)
        return
    work=a.DirectionWork();work.begin(d.ARMS[1],[1.]*257,[0.]*257)
    with pytest.raises(ValueError):
        if fault=='norm':
            for _ in range(5):work.norm([0.]*257)
        elif fault=='dot':
            for _ in range(3):work.dot([0.]*257,[0.]*257)
        else:
            for _ in range(5):work.vector([0.]*257)


@pytest.mark.parametrize('theta',[[0.,0.,0.],[.0625,-.03125,.1],[2.,-2.,.5]])
def test_candidate_exact_slopes_agree_at_saved_clip_and_hinge_cases(theta):
    z=np.zeros((4,256));z[:,:2]=[[1.,0.],[0.,1.],[1.,1.],[-1.,0.]]
    y=np.zeros((4,7));y[:,2]=[.5,-.25,.25,.5];p0=np.zeros_like(y)
    context=tuple(a.readonly(v) for v in (z,y,p0));rows=np.arange(4,dtype=np.int32)
    params=np.r_[theta[:-1],np.zeros(254),theta[-1]]
    left=main.measure(context,params,1.,rows,arm=d.ARMS[1]);right=audit.measure(audit.TorchFit(context),params,1.,rows,arm=d.ARMS[1])
    audit.deep_check(audit.Checks(),left,right,'candidate exact derivatives')
    assert abs(left['directional_derivative']-left['exact_MSE_slope']/7-left['exact_penalty_slope'])<=1e-10


def test_finite_MSE_increase_below_fixed_start_is_not_a_new_acceptance_gate(cfg):
    jobs,context,rows,start=registered_job(cfg)
    row=main.measure(context,jobs.current,1.,rows,jobs)
    # Construct a current interior state and its strict-Armijo trial at the scalar
    # gate, distinct from the first-order direction guard.
    decision=d.trial_decision(d.ARMS[1],.02,.019,-.01,0,.095,.1)
    assert decision['accepted']
    first=dict(trainable_objective=.02,component3_mse=.1)
    control=dict(final_trainable_objective=.018,final_component3_mse=.09,extra_updates=2)
    candidate=dict(final_trainable_objective=.017,final_component3_mse=.085,extra_updates=2)
    one=main.paired_reference(first,control,candidate,[-.02,.005]);two=audit.paired_reference(first,control,candidate,[-.02,.005])
    assert one==two and one['candidate_stepwise_MSE_increases']==1 and not one['paired_joint_cue']
