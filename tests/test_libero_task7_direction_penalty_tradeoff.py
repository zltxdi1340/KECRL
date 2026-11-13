"""Reference/hinge accounting and forbidden-data regression checks."""
import copy
import json
from pathlib import Path
import sys
import pytest
from experiments import libero_task7_direction_penalty_tradeoff_protocol as d
from experiments import libero_task7_direction_penalty_tradeoff_access as access
from experiments import analyze_libero_task7_direction_penalty_tradeoff as main
from experiments import audit_libero_task7_direction_penalty_tradeoff as audit


def synthetic():
    ident=dict(task=7,seed=20261106,arm='A_original',fold=0,mode='harm_protected_affine')
    spec=dict(ident,cohort='full50',prior_control_updates=50)
    states=[]; trials=[]
    for opt in d.ARMS:
        dm,dp=(-.001,-.0001) if opt==d.ARMS[0] else (-.002,.00005)
        for k in range(51):
            m,p=1+k*dm,.1+k*dp
            states.append(dict(ident,optimization_arm=opt,extra_updates=k,n=1000,component3_mse=m,penalty=p,trainable_objective=m/7+p,
                               exact_MSE_slope=dm,exact_penalty_slope=dp,directional_derivative=dm/7+dp,hinge_positive_n=100,branch_digest=str(k%2)))
            if k<50:
                trials.append(dict(ident,optimization_arm=opt,from_extra_updates=k,trial_index=0,alpha=1.,accepted=True))
    return [spec],states,trials


def test_paired_counterexample_is_not_failure_to_reduce_own_objective():
    specs,states,trials=synthetic()
    out=main.analyze_backend('P',states,trials,specs)
    row=out['pair_endpoints'][0]
    assert row['dM']<0 and row['dP']>0 and row['dT']>0 and not row['paired_cue']
    assert all(r['dT']<0 for r in out['own_endpoints'])
    expected=audit.reconstruct('P',states,trials,specs)
    assert audit.compare(out,expected)>1000


def test_active_symmetric_product_handles_changed_denominator_share():
    left=dict(M=1.,P=.04,T=1/7+.04,n=100,h=20,active_fraction=.2,positive_excess_mean=1.4)
    right=dict(M=.9,P=.06,T=.9/7+.06,n=100,h=40,active_fraction=.4,positive_excess_mean=1.05)
    result=main.factors(left,right)
    assert result['count_component']>0 and result['severity_component']<0
    assert result['count_component']+result['severity_component']==pytest.approx(.02)
    audit.compare(result,audit.active(left,right))


def test_zero_active_does_not_invent_conditional_mean_or_contributions():
    s=dict(n=100,hinge_positive_n=0,component3_mse=.1,penalty=0.,trainable_objective=.1/7)
    p=main.point(s)
    assert p['positive_excess_mean'] is None
    assert main.factors(p,p)['defined'] is False
    assert main.factors(p,p)['count_component'] is None
    s['penalty']=.01
    with pytest.raises(ValueError):main.point(s)


def test_strict_small_positive_gap_is_not_epsilon_reclassified():
    s=dict(M=1.,P=.1,T=1/7+.1)
    t=dict(s,T=s['T']+1e-12,P=s['P']+1e-12)
    assert main.change(s,t)['dT']>0


def test_arithmetic_audit_rejects_tampered_penalty():
    specs,states,trials=synthetic(); out=main.analyze_backend('P',states,trials,specs)
    out['pair_endpoints'][0]['dP']+=1e-4
    with pytest.raises(ValueError):audit.compare(out,audit.reconstruct('P',states,trials,specs))


@pytest.mark.parametrize('corruption',['missing_state','duplicate_trial','non_grid','acceptance','nans'])
def test_malformed_path_fails_closed(corruption):
    specs,states,trials=synthetic()
    if corruption=='missing_state':states.pop(10)
    elif corruption=='duplicate_trial':trials.append(dict(trials[0]))
    elif corruption=='non_grid':trials[0]['alpha']=.3
    elif corruption=='acceptance':states[1]['component3_mse']=2.
    else:states[1]['penalty']=float('nan')
    with pytest.raises(ValueError):main.analyze_backend('P',states,trials,specs)
    with pytest.raises(ValueError):audit.reconstruct('P',states,trials,specs)


def test_semantic_projection_never_exports_unselected_arrays():
    data={'n':3,'penalty':.2,'secret_vector':[1,2,3]}
    assert access.scalar(data,['n','penalty'])=={'n':3,'penalty':.2}
    with pytest.raises(ValueError):access.scalar(data,['secret_vector'])
    with pytest.raises(ValueError):access.scalar({'penalty':float('nan')},['penalty'])


@pytest.mark.parametrize('operation',['checkpoint_SHA','source_CSV','write','os_open','numpy_import','raw_JSON_bypass'])
def test_guard_blocks_unlisted_io_and_numerical_runtime(tmp_path,operation):
    cfg={'execution':{'seconds_per_stage':120,'RSS_bytes':2**31}}
    guard=access.Guard(cfg); path=tmp_path/'payload.json'; path.write_text('{}')
    with pytest.raises(PermissionError):
        with guard:
            if operation=='checkpoint_SHA':d.sha(tmp_path/'head.npz')
            elif operation=='source_CSV':(tmp_path/'fit.csv').read_text()
            elif operation=='write':path.write_text('changed')
            elif operation=='os_open':__import__('os').open(path,0)
            elif operation=='numpy_import':__import__('numpy')
            else:path.read_text()
    assert guard.denied and path.read_text()=='{}'


def test_guard_rejects_extra_projection_pass():
    guard=access.Guard({'inputs':{'P':{}},'execution':{}}); guard.seen=['P']
    with pytest.raises(PermissionError):guard.project('P')


def test_runtime_requires_bounded_fresh_process(monkeypatch):
    monkeypatch.setitem(sys.modules,'torch',object())
    with pytest.raises(ValueError):access.runtime({'execution':{'python':'3.10.22','environment':{}}})


def test_full50_matched_windows_telescope_and_audit_uses_own_implementation():
    specs,states,trials=synthetic(); out=main.analyze_backend('P',states,trials,specs)
    path=out['full50_paths'][0]
    assert path['first_positive_k']==path['persistent_positive_from']==1
    for field in ('dM','dP','dT'):
        assert sum(w['change_'+field] for w in path['windows'])==pytest.approx(out['pair_endpoints'][0][field])
    assert 'analyze_libero_task7_direction_penalty_tradeoff' not in Path(audit.__file__).read_text().split('def run():')[0]


def test_full18_projection_and_independent_arithmetic_with_exact_real_row_budget(tmp_path):
    full=set(d.FOCUS)|{('A_original',20261107,0),('A_original',20261107,2),('A_original',20261108,1)}
    short={('A_original',20261106,1):7,('B_fomaml',20261106,0):7,('A_original',20261107,1):8,('A_original',20261108,0):16,('B_fomaml',20261108,0):10}
    specs=[]; states=[]; trials=[]
    for seed in (20261106,20261107,20261108):
        for arm in ('A_original','B_fomaml'):
            for fold in range(3):
                ident=dict(task=7,seed=seed,arm=arm,fold=fold,mode='harm_protected_affine'); key=(arm,seed,fold)
                count=50 if key in full else short.get(key,0)
                specs.append(dict(ident,prior_control_updates=count,cohort='full50' if count==50 else 'short' if count else 'zero'))
                for opt in d.ARMS:
                    last=count if opt==d.ARMS[0] else 50
                    dm=-.001 if opt==d.ARMS[0] else -.002
                    dp=-.0001 if opt==d.ARMS[0] else .00005 if key in d.FOCUS else -.00015
                    for k in range(last+1):
                        m,p=1+k*dm,.1+k*dp
                        states.append(dict(ident,optimization_arm=opt,extra_updates=k,n=1000,component3_mse=m,penalty=p,trainable_objective=m/7+p,
                                           exact_MSE_slope=dm,exact_penalty_slope=dp,directional_derivative=dm/7+dp,hinge_positive_n=100,branch_digest=str(k%2)))
                        if k<last:trials.append(dict(ident,optimization_arm=opt,from_extra_updates=k,trial_index=0,alpha=1.,accepted=True))
    assert len(states)==1334 and len(trials)==1298
    alltrials=trials+[{'accepted':False,'unselected_vector':[99]} for _ in range(10920-len(trials))]
    cfg={'inputs':{},'execution':{'seconds_per_stage':120,'RSS_bytes':2**31,'input_JSON_bytes_max':100663296},'projections':{'states':d.STATE,'accepted_trials':d.TRIAL}}
    for backend in ('P','U'):
        path=tmp_path/(backend+'.json')
        status='complete_joint_direction_v2_pending_same_parameter_audit' if backend=='P' else 'passed_independent_same_parameter_joint_direction_v2_audit'
        obj=dict(status=status,science_config=d.PARENTS['science'],binding=d.PARENTS['binding'],states=states,trials=alltrials,unselected_parameters=[0,1])
        if backend=='U':obj['primary']=cfg['inputs']['P']
        path.write_text(json.dumps(obj)); cfg['inputs'][backend]=d.pin(path)
    guard=access.Guard(cfg)
    with guard:
        for backend in ('P','U'):
            ss,tt=guard.project(backend)
            assert all(set(s)==set(d.STATE) for s in ss)
            actual=main.analyze_backend(backend,ss,tt,specs)
            assert audit.compare(actual,audit.reconstruct(backend,ss,tt,specs))>40000
        guard.complete()
    assert guard.record()['states']==2668 and guard.record()['source_SHA_checks']==4
