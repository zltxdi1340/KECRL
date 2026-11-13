"""Human-made geometry and metadata gates; no research source numerics."""
import builtins
import copy
import io
import json
import math
from pathlib import Path
import pytest
from experiments import libero_task7_t_priority_direction_protocol as d


def vec(*xs):return list(xs)+[0.]*(257-len(xs))


@pytest.fixture(scope='module')
def canonical():return d.create()


def test_keeps_all18_original500_loss_and_role_boundaries(canonical):
    old=d.p.load(d.meta.PARENTS['science'])
    for field in ('inputs','reference_points','cohorts','source_members','head_members','new_head_members','audit_head_members'):
        assert canonical[field]==old[field]
    assert len(canonical['inputs']['heads'])==18
    assert canonical['scope']['original_Policy_updates']==canonical['scope']['source_auxiliary_updates']==500
    assert canonical['scope']['future_stabilization_steps']==5
    assert canonical['scope']['source_meta_query_validation_heldout_qualification_reserved_final_access'] is False
    assert canonical['scope']['new_loss_lambda_representation_or_data_weight_change'] is False
    for field in ('same_loss','same_precision','extra_updates_max','alpha_initial','backtrack_factor','armijo_c1','acceptance','gate','mse_monotonicity','stop_priority'):
        assert canonical['optimization'][field]==old['optimization'][field]
    assert canonical['optimization']['arms']==d.ARMS
    assert canonical['optimization']['baseline_MSE_direction_guard'] is True


def test_control_is_old_bisector900_not_old398_or_a_final_start(canonical):
    history=canonical['historical_control']; b=canonical['budget']
    assert history['optimization_arm']=='norm_matched_joint_bisector'
    assert history['primary_source_JSON']==d.meta.INPUTS['P'] and history['audit_source_JSON']==d.meta.INPUTS['U']
    assert b['historical_control_updates']==900 and b['historical_control_JSON_projected_records_per_stage']==dict(states=918,trials=7411,endpoints=36,heads=18)
    assert sum(h['prior_control_updates'] for h in canonical['cohorts'])==398
    assert [sum(h['cohort']==c for h in canonical['cohorts']) for c in ('zero','short','full50')]==[6,5,7]
    assert 'parameter_digest' in history['projections']['states']
    assert 'directional_derivative' not in history['saved500_endpoint_projection']
    assert 'directional_derivative' in history['projections']['endpoints']
    assert 'v2_bisector_state_k0' in history['saved500_direction_reference']
    assert canonical['new_head_layout']['order']==['saved500',*[a+'_final' for a in d.ARMS]]


def test_budget_counts_projection_dot_and_fixes_inherited_archive_ledger(canonical):
    b=canonical['budget']; s=1836
    assert b['main_auxiliary_updates_max']==b['audit_replay_updates_max']==1800
    assert b['total_auxiliary_updates_max']==3600 and b['total_gradient_states_max']==5508
    assert b['total_trial_evaluations_max']==108000 and b['total_linear_forms_max']==120852
    assert b['total_linear_rows_max']==337418784 and b['audit_total_autograd_max']==11016
    assert b['direction_dot_reductions_max_per_stage']==s*3 and b['total_direction_dots_max']==5508*3
    assert b['direction_branch_dot_caps']==dict(zip(d.ARMS,(2,3)))
    assert b['total_direction_norms_max']==22032 and b['total_direction_coordinates_max']==5662224
    assert (b['total_NPZ_loads'],b['total_member_decodes'],b['total_archive_SHA_checks'])==(68,574,172)
    assert b['new_member_writes_primary']==b['primary_new_member_writes']==b['new_member_decodes_audit']==252
    assert b['audit_new_member_writes']==378 and b['new_member_writes']==630
    assert b['total_historical_JSON_passes']==2 and b['total_historical_JSON_SHA']==4
    assert canonical['execution']['audit_seconds']==canonical['execution']['audit_seconds_max']==900
    assert canonical['execution']['audit_RSS_bytes']==canonical['execution']['audit_RSS_bytes_max']==1073741824
    assert canonical['execution']['output_NPZ_bytes_max']==canonical['execution']['trace_NPZ_bytes_per_stage_max']==83886080


@pytest.mark.parametrize('total,base',[(vec(1.,0.),vec(0.,1.)),(vec(2.,0.),vec(1.,1.)),(vec(3.,0.),vec(2.,0.))])
def test_unconstrained_minus_total_is_kept_when_already_feasible(total,base):
    row=d.direction(d.ARMS[1],total,base)
    assert row['direction']==[-v for v in total]
    assert row['status']=='negative_total_feasible' and row['selected_joint']
    assert row['projection_norm'] is None and row['unit_cosine']>=0


@pytest.mark.parametrize('base',[vec(-1.,1.),vec(-2.,1.),vec(-3.,4.),vec(-4.,3.)])
def test_conflicting_projection_matches_independent_two_dimensional_solution(base):
    row=d.direction(d.ARMS[1],vec(1.,0.),base); direction=row['direction']
    b=math.hypot(base[0],base[1]); expected=vec(-base[1]/b,base[0]/b)
    assert direction==pytest.approx(expected,abs=1e-14)
    assert math.hypot(*direction)==pytest.approx(1.,abs=1e-14)
    assert abs(math.fsum(x*y for x,y in zip(base,direction)))<1e-14
    bisector=d.direction(d.ARMS[0],vec(1.,0.),base)
    assert row['selected_T_slope']<bisector['selected_T_slope']
    # Positive feasibility residuals must be retained, never epsilon repaired.
    assert row['status']==('projected_T_priority' if row['selected_joint'] else 'unresolved_joint_direction')
    assert row['construction_status']=='projected_T_priority'


def test_projection_is_optimal_against_a_dense_synthetic_feasible_circle():
    total,base=vec(2.,0.),vec(-3.,4.); row=d.direction(d.ARMS[1],total,base)
    optimum=row['selected_T_slope']
    for index in range(7201):
        angle=2*math.pi*index/7200; x,y=2*math.cos(angle),2*math.sin(angle)
        if base[0]*x+base[1]*y<=0:
            assert total[0]*x>=optimum-1e-12


@pytest.mark.parametrize('base',[vec(-1.),vec(-1.,1e-13)])
def test_zero_or_near_zero_projection_has_no_fallback(base):
    row=d.direction(d.ARMS[1],vec(1.),base)
    assert row['projection_norm']<=1e-12 and row['direction']==vec()
    assert row['status']=='unresolved_joint_direction' and not row['selected_joint']
    assert d.stopping_reason(d.ARMS[1],0,1.,row['status'],False,0.,0.)=='unresolved_joint_direction'


def test_zero_total_and_zero_base_retain_exact_slope_guards():
    zero=d.direction(d.ARMS[1],vec(),vec(1.))
    assert zero['status']=='zero_total_gradient' and zero['direction']==vec()
    assert d.stopping_reason(d.ARMS[1],0,0.,zero['status'],False,0.,0.)=='zero_gradient'
    row=d.direction(d.ARMS[1],vec(1.,2.),vec())
    assert row['direction']==vec(-1.,-2.) and row['selected_MSE_slope']==0
    assert d.stopping_reason(d.ARMS[1],0,row['total_norm'],row['status'],True,-1.,.01)=='exact_MSE_direction_increase'


@pytest.mark.parametrize('total,base',[(vec(1.),vec(2.)),(vec(1.,2.),vec(0.,1.)),(vec(1.),vec(-1.)),(vec(),vec(1.)),(vec(1.),vec())])
def test_control_reference_is_unchanged_v2_formula(total,base):
    old=d.parent.old.direction(d.parent.old.ARMS[1],total,base)
    new=d.direction(d.ARMS[0],total,base)
    for key in old:
        assert new[key]==old[key]


@pytest.mark.parametrize('arm',d.ARMS)
def test_no_extra_M_or_P_acceptance_gate(arm):
    # T decreases, but M may exceed a hypothetical previous M=.08 below M500=.1.
    assert d.trial_decision(arm,.02,.019,-.01,0,.095,.1)['accepted']
    for index in range(20):
        assert d.trial_alpha(arm,index)==2.**(-index)
        assert d.trial_decision(arm,.02,.019,-.01,index,.09,.1)==d.parent.old.trial_decision(d.parent.old.ARMS[1],.02,.019,-.01,index,.09,.1)
    assert not d.trial_decision(arm,.02,.019,-.01,0,math.nextafter(.1,math.inf),.1)['accepted']


@pytest.mark.parametrize('arm',d.ARMS)
def test_stop_priority_and_strict_exact_boundary_predicates(arm):
    args=(arm,0,1.,'joint_bisector' if arm==d.ARMS[0] else 'projected_T_priority',True)
    assert d.stopping_reason(*args,-.1,math.nextafter(0.,1.))=='exact_MSE_direction_increase'
    assert d.stopping_reason(*args,0.,-.1)=='non_descent_selected_direction'
    assert d.stopping_reason(*args,-.1,0.) is None
    assert d.stopping_reason(*args,-.1,0.,20)=='line_search_exhausted'
    assert d.stopping_reason(arm,50,1.,'unresolved_joint_direction',False,0.,0.)=='update_cap'


def test_fixed_M500_and_zero_progress_do_not_create_scientific_cue():
    row=d.paired_evidence(.02,.1,.018,.09,.017,.085,2,2,[-.02,.005])
    assert row['candidate_progress_from500'] and not row['paired_joint_cue']
    row=d.paired_evidence(.02,.1,.018,.09,.02,.1,2,0,[])
    assert not row['candidate_progress_from500'] and not row['paired_joint_cue']


def test_T_better_but_M_worse_than_bisector_is_still_failed_paired_cue():
    row=d.paired_evidence(.02,.1,.018,.08,.017,.085,2,2,[-.01,-.005])
    assert row['candidate_path_joint_cue'] and not row['paired_joint_cue']


@pytest.mark.parametrize('arm,status',[(d.ARMS[0],'projected_T_priority'),(d.ARMS[1],'joint_bisector'),(d.ARMS[1],'zero_total_gradient')])
def test_wrong_direction_status_or_zero_norm_claim_is_rejected(arm,status):
    with pytest.raises(ValueError):d.stopping_reason(arm,0,1.,status,True,-1.,0.)


@pytest.mark.parametrize('fault',['shape','nan','inf','bool','overflow','arm'])
def test_bad_direction_inputs_fail_closed(fault):
    t,b=vec(1.),vec(2.);arm=d.ARMS[1]
    if fault=='shape':b.pop()
    elif fault=='nan':t[0]=float('nan')
    elif fault=='inf':b[0]=float('inf')
    elif fault=='bool':b[0]=True
    elif fault=='overflow':t[0]=b[0]=1e308
    else:arm='other'
    with pytest.raises((ValueError,OverflowError)):d.direction(arm,t,b)


@pytest.mark.parametrize('index',[-1,20,True,1.])
def test_no_unfrozen_grid_point(index):
    with pytest.raises(ValueError):d.trial_alpha(d.ARMS[1],index)


@pytest.mark.parametrize('section,key,value',[
    ('scope','original_Policy_updates',1000),('scope','future_stabilization_steps',0),
    ('optimization','extra_updates_max',100),('optimization','baseline_MSE_direction_guard',False),
    ('execution','local_absolute_tolerance',1e-8),('execution','technical_retries',1),
    ('budget','total_direction_dots_max',11016),('budget','historical_control_updates',398),
])
def test_changed_science_is_rejected(canonical,tmp_path,monkeypatch,section,key,value):
    bad=copy.deepcopy(canonical);bad[section][key]=value;path=tmp_path/'bad.json';path.write_text(json.dumps(bad))
    monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(ValueError):d.validate(path)


def test_metadata_factory_never_opens_archives_or_historical_numerics(canonical,monkeypatch):
    original=(builtins.open,io.open); forbidden={r['path'] for r in d.meta.INPUTS.values()}
    def check(path):
        if isinstance(path,(str,Path)):
            assert str(Path(path).resolve()) not in forbidden
            assert Path(path).suffix not in {'.npz','.npy','.csv','.pt','.h5','.hdf5'}
    def op(path,*args,**kw):check(path);return original[0](path,*args,**kw)
    def ip(path,*args,**kw):check(path);return original[1](path,*args,**kw)
    monkeypatch.setattr(builtins,'open',op);monkeypatch.setattr(io,'open',ip)
    monkeypatch.setattr(d,'direction',lambda *a:(_ for _ in ()).throw(AssertionError('no real directions')))
    assert d.create()==canonical


def test_science_freeze_does_not_substitute_for_execution_binding(canonical,tmp_path,monkeypatch):
    path=tmp_path/'science.json';path.write_text(json.dumps(canonical));monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(FileNotFoundError):d.require_binding(path,tmp_path/'missing-execution-binding.json')
    assert not Path(d.OUTPUT).exists() and not Path(d.ARTIFACTS).exists()


def test_all18_source_gate_preserves_every_failure(canonical):
    rows=[{**{k:r[k] for k in d.IDS},'paired_joint_cue':True} for r in canonical['cohorts']]
    assert d.source_evidence(rows)['population_direction_fit_cue']
    rows[0]['paired_joint_cue']=False
    assert not d.source_evidence(rows)['population_direction_fit_cue']
    with pytest.raises(ValueError):d.source_evidence(rows[1:])
