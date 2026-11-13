"""Synthetic margin geometry, dual references and metadata isolation."""
import builtins
import copy
import io
import json
import math
from pathlib import Path
import pytest
from experiments import libero_task7_m_descent_margin_direction_protocol as d


def vec(*xs):return list(xs)+[0.]*(257-len(xs))


def row_of(point,exact_m=None,exact_t=None):
    return dict(total_l2=point['total_norm'],direction_status=point['status'],direction_construction_status=point['construction_status'],
                direction_total_reference_norm=point['total_norm'],direction_base_reference_norm=point['base_norm'],
                selected_joint=point['selected_joint'],selected_T_slope=point['selected_T_slope'],selected_MSE_slope=point['selected_MSE_slope'],
                required_M_slope_bound=point['required_M_slope_bound'],
                required_margin_fraction=point['required_margin_fraction'],construction_margin_fraction=point['construction_margin_fraction'],selected_margin_met=point['selected_margin_met'],
                exact_MSE_slope=point['selected_MSE_slope'] if exact_m is None else exact_m,
                directional_derivative=point['selected_T_slope'] if exact_t is None else exact_t)


@pytest.fixture(scope='module')
def canonical():return d.create()


def test_same18_same500_loss_roles_finite_acceptance_and_no_grid_change(canonical):
    old=d.p.load(d.PARENTS['science'])
    for field in ('inputs','reference_points','cohorts','source_members','head_members','new_head_members','audit_head_members'):
        assert canonical[field]==old[field]
    for field in ('same_loss','same_precision','extra_updates_max','alpha_initial','backtrack_factor','armijo_c1','acceptance','gate','mse_monotonicity','final'):
        assert canonical['optimization'][field]==old['optimization'][field]
    assert len(canonical['inputs']['heads'])==18
    assert canonical['scope']['original_Policy_updates']==canonical['scope']['source_auxiliary_updates']==500
    assert canonical['scope']['future_stabilization_steps']==5
    assert canonical['scope']['source_meta_query_validation_heldout_qualification_reserved_final_access'] is False
    assert canonical['scope']['new_loss_lambda_representation_or_data_weight_change'] is False
    assert canonical['optimization']['required_margin_fraction_by_arm']==dict(zip(d.ARMS,(0.,.01)))
    assert canonical['optimization']['construction_margin_multiplier']==2.
    assert canonical['optimization']['trial_caps']=={arm:20 for arm in d.ARMS}


def test_zero_control_counts_are_engine_specific_and_anchor_is_not_training_arm(canonical):
    b=canonical['budget'];h=canonical['historical_control']
    assert (b['historical_control_counts_by_engine']['P']['updates'],b['historical_control_counts_by_engine']['U']['updates'])==(81,74)
    assert (h['counts_by_engine']['P']['states'],h['counts_by_engine']['U']['states'])==(99,92)
    assert (h['counts_by_engine']['P']['trials'],h['counts_by_engine']['U']['trials'])==(673,624)
    assert h['optimization_arm']==d.ARMS[0] and 'required_total_updates' not in h
    assert 'directional_derivative' not in h['saved500_endpoint_projection']
    assert 'zero_margin_state_k0' in h['saved500_direction_reference']
    assert canonical['bisector_reference']['rows_per_engine']==18
    assert canonical['bisector_reference']['source']=='same_single_historical_JSON_parse'
    assert canonical['new_head_layout']['order']==['saved500',*[a+'_final' for a in d.ARMS]]
    assert sum(c['prior_control_updates'] for c in canonical['cohorts'])==398


def test_budget_conservation_and_tight_limits(canonical):
    b=canonical['budget'];P,U,S=b['per_engine']['P'],b['per_engine']['U'],b['S']
    assert (P['updates_max'],U['updates_max'],S['updates_max'])==(981,974,0)
    assert (P['gradient_states_max'],U['gradient_states_max'],S['gradient_states_max'])==(1017,1010,1017)
    assert (P['trials_max'],U['trials_max'],S['trials_max'])==(18673,18624,18673)
    assert b['total_auxiliary_updates_max']==1955 and b['total_gradient_states_max']==3044
    assert b['total_gradient_vectors_max']==9132 and b['audit_total_autograd_max']==6081
    assert b['total_trial_evaluations_max']==55970 and b['total_linear_forms_max']==63075
    assert b['total_linear_rows_max']==176171625
    assert b['total_linear_rows_max']==P['linear_rows_max']+U['linear_rows_max']+S['linear_rows_max']
    assert b['total_direction_dots_max']==9132 and b['total_direction_norms_max']==12176
    assert (b['output_CSV_rows_max'],b['audit_local_CSV_rows_max'],b['total_CSV_rows_max'])==(33694,19690,53384)
    assert (b['total_NPZ_loads'],b['total_member_decodes'],b['total_archive_SHA_checks'])==(68,574,172)
    assert b['primary_new_member_writes']+b['audit_new_member_writes']==b['new_member_writes']==630
    assert P['linear_forms_max']==2*P['gradient_states_max']+P['trials_max']
    assert S['linear_forms_max']==3*S['gradient_states_max']+S['trials_max']
    assert b['primary_trace_coordinate_copies_max']==5*1017*257


@pytest.mark.parametrize('total,base',[(vec(1.),vec(2.)),(vec(1.,2.),vec(-2.,1.)),(vec(1.),vec()),(vec(),vec(1.)),(vec(1.),vec(-1.))])
def test_control_exactly_preserves_previous_zero_margin_reference(total,base):
    old=d.parent.direction(d.parent.ARMS[1],total,base);new=d.direction(d.ARMS[0],total,base)
    for key,value in old.items():assert new[key]==value
    row=row_of(new)
    for k in (0,50):
        assert d.stopping_reason(d.ARMS[0],k,row)==d.parent.stopping_reason(d.parent.ARMS[1],k,row['total_l2'],row['direction_status'],row['selected_joint'],row['directional_derivative'],row['exact_MSE_slope'])


@pytest.mark.parametrize('base',[vec(0.,1.),vec(-1.,1.),vec(-3.,4.),vec(-4.,3.)])
def test_margin_geometry_matches_independent_two_dimensional_solution(base):
    point=d.direction(d.ARMS[1],vec(1.,0.),base)
    norm=math.hypot(base[0],base[1]);nx,ny=base[0]/norm,base[1]/norm
    kappa=.02;tangent=math.sqrt(1-kappa*kappa)
    expected=vec(-kappa*nx-tangent*ny,-kappa*ny+tangent*nx)
    assert point['direction']==pytest.approx(expected,abs=1e-14)
    assert point['direction_norm']==pytest.approx(1.,abs=1e-14)
    assert point['selected_MSE_slope']==pytest.approx(-7*.02*norm,abs=1e-14)
    assert point['selected_MSE_slope']<point['required_M_slope_bound']<0
    assert point['selected_joint'] and d.stopping_reason(d.ARMS[1],0,row_of(point)) is None


def test_construction_is_T_optimal_for_kappa_not_claimed_optimal_for_rho():
    base=vec(-3.,4.);point=d.direction(d.ARMS[1],vec(1.),base)
    for index in range(14401):
        angle=2*math.pi*index/14400;x,y=math.cos(angle),math.sin(angle)
        if base[0]*x+base[1]*y<=-.02*5:
            assert x>=point['selected_T_slope']-1e-12


@pytest.mark.parametrize('base',[vec(1.),vec(1.,1.)])
def test_already_margin_feasible_minus_G_is_kept(base):
    point=d.direction(d.ARMS[1],vec(1.),base)
    assert point['status']=='negative_total_margin_feasible'
    assert point['direction']==vec(-1.)


def test_positive_margin_not_a_sign_epsilon_and_exact_right_guard_is_strong():
    point=d.direction(d.ARMS[1],vec(1.),vec(-1.,1.));row=row_of(point);bound=point['required_M_slope_bound']
    assert bound<0
    assert d.stopping_reason(d.ARMS[1],0,dict(row,exact_MSE_slope=math.nextafter(bound,math.inf)))=='exact_M_margin_violation'
    assert d.stopping_reason(d.ARMS[1],0,dict(row,exact_MSE_slope=-1e-18))=='exact_M_margin_violation'
    assert d.stopping_reason(d.ARMS[1],0,dict(row,selected_MSE_slope=1e-21))=='selected_M_margin_violation'
    assert d.stopping_reason(d.ARMS[1],0,dict(row,exact_MSE_slope=bound)) is None


@pytest.mark.parametrize('field,value',[('required_M_slope_bound',None),('required_margin_fraction',0.),('construction_margin_fraction',.01),('selected_margin_met',False)])
def test_forged_margin_fields_fail_before_trial_reuse(field,value):
    row=row_of(d.direction(d.ARMS[1],vec(1.),vec(-1.,1.)))
    assert d.validate_margin_record(d.ARMS[1],row)
    row[field]=value
    with pytest.raises(ValueError):d.validate_margin_record(d.ARMS[1],row)


def test_zero_gradient_update_cap_and_exact_T_stop_priority():
    point=d.direction(d.ARMS[1],vec(),vec(1.));row=row_of(point)
    assert d.validate_margin_record(d.ARMS[1],row)
    assert d.stopping_reason(d.ARMS[1],0,row)=='zero_gradient'
    assert d.stopping_reason(d.ARMS[1],50,row)=='update_cap'
    row=row_of(d.direction(d.ARMS[1],vec(1.),vec(-1.,1.)),exact_t=0.)
    assert d.stopping_reason(d.ARMS[1],0,row)=='non_descent_selected_direction'
    assert d.stopping_reason(d.ARMS[1],0,dict(row,directional_derivative=-.1),20)=='line_search_exhausted'


def test_zero_base_underflow_and_opposite_geometry_stop_without_fallback():
    point=d.direction(d.ARMS[1],vec(1.),vec())
    assert point['status']=='zero_base_margin_unavailable' and point['direction']==vec()
    assert d.stopping_reason(d.ARMS[1],0,row_of(point))=='zero_base_margin_unavailable'
    point=d.direction(d.ARMS[1],vec(1e-300),vec(1e-300))
    assert point['status']=='margin_unrepresentable' and point['required_M_slope_bound'] is None
    assert d.stopping_reason(d.ARMS[1],0,row_of(point))=='margin_unrepresentable'
    point=d.direction(d.ARMS[1],vec(1.),vec(-1.,1e-13))
    assert point['status']=='unresolved_margin_direction'
    assert d.stopping_reason(d.ARMS[1],0,row_of(point))=='unresolved_margin_direction'


def test_non_descent_T_under_margin_is_not_repaired_or_smaller_margin():
    point=d.direction(d.ARMS[1],vec(1.),vec(-1.,.001))
    assert point['selected_margin_met'] and point['selected_T_slope']>0
    assert d.stopping_reason(d.ARMS[1],0,row_of(point))=='non_descent_selected_direction'
    assert point['construction_margin_fraction']==.02


@pytest.mark.parametrize('arm',d.ARMS)
def test_finite_M500_gate_and_grid_are_unchanged(arm):
    assert d.trial_decision(arm,.02,.019,-.01,0,.095,.1)['accepted'] # previous M could be .08
    for i in range(20):
        assert d.trial_alpha(arm,i)==2.**(-i)
        assert d.trial_decision(arm,.02,.019,-.01,i,.09,.1)==d.parent.trial_decision(d.parent.ARMS[1],.02,.019,-.01,i,.09,.1)
    assert not d.trial_decision(arm,.02,.019,-.01,0,math.nextafter(.1,math.inf),.1)['accepted']


def test_weak_control_improvement_does_not_replace_old_bisector_acceptance():
    weak=d.paired_evidence(.02,.1,.019,.095,.017,.08,.018,.09,1,1,[-.01])
    assert weak['margin_pair_joint_cue'] and not weak['bisector_anchor_joint_cue'] and not weak['paired_joint_cue']
    good=d.paired_evidence(.02,.1,.019,.095,.017,.08,.016,.075,1,1,[-.025])
    assert good['paired_joint_cue']
    finite_up=d.paired_evidence(.02,.1,.019,.095,.017,.08,.016,.075,1,2,[-.03,.005])
    assert not finite_up['paired_joint_cue']


@pytest.mark.parametrize('fault',['shape','nan','inf','bool','overflow','arm'])
def test_invalid_vectors_fail_closed(fault):
    t,b=vec(1.),vec(2.);arm=d.ARMS[1]
    if fault=='shape':t.pop()
    elif fault=='nan':b[0]=float('nan')
    elif fault=='inf':t[0]=float('inf')
    elif fault=='bool':b[0]=True
    elif fault=='overflow':t[0]=b[0]=1e308
    else:arm='other'
    with pytest.raises((ValueError,OverflowError)):d.direction(arm,t,b)


@pytest.mark.parametrize('index',[-1,20,True,1.])
def test_grid_cannot_expand(index):
    with pytest.raises(ValueError):d.trial_alpha(d.ARMS[1],index)


@pytest.mark.parametrize('section,key,value',[
    ('scope','original_Policy_updates',1000),('scope','future_stabilization_steps',0),
    ('optimization','extra_updates_max',100),('optimization','construction_margin_multiplier',1.),
    ('budget','total_auxiliary_updates_max',3600),('decision','primary_head_cue','zero_control_only'),
    ('execution','technical_retries',1),('execution','local_absolute_tolerance',1e-8),
])
def test_changed_science_rejected(canonical,tmp_path,monkeypatch,section,key,value):
    changed=copy.deepcopy(canonical);changed[section][key]=value;path=tmp_path/'bad.json';path.write_text(json.dumps(changed))
    monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(ValueError):d.validate(path)


def test_factory_denies_all_research_archives_and_old_numeric_JSON(canonical,monkeypatch):
    originals=(builtins.open,io.open);forbidden={r['path'] for r in d.HISTORY_INPUTS.values()}
    def check(path):
        if isinstance(path,(str,Path)):
            assert str(Path(path).resolve()) not in forbidden
            assert Path(path).suffix not in {'.npz','.npy','.csv','.pt','.h5','.hdf5'}
    def op(path,*args,**kw):check(path);return originals[0](path,*args,**kw)
    def ip(path,*args,**kw):check(path);return originals[1](path,*args,**kw)
    monkeypatch.setattr(builtins,'open',op);monkeypatch.setattr(io,'open',ip)
    monkeypatch.setattr(d,'direction',lambda *a:(_ for _ in ()).throw(AssertionError('no real directions')))
    assert d.create()==canonical


def test_scientific_freeze_does_not_authorize_unbound_execution(canonical,tmp_path,monkeypatch):
    path=tmp_path/'science.json';path.write_text(json.dumps(canonical));monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(FileNotFoundError):d.require_binding(path,tmp_path/'missing-execution-binding.json')
    assert not Path(d.OUTPUT).exists() and not Path(d.ARTIFACTS).exists()


def test_all18_and_both_sources_are_required(canonical):
    rows=[{**{k:h[k] for k in d.IDS},'paired_joint_cue':True} for h in canonical['inputs']['heads']]
    assert d.source_evidence(rows)['population_direction_fit_cue']
    rows[-1]['paired_joint_cue']=False
    assert not d.source_evidence(rows)['population_direction_fit_cue']
    with pytest.raises(ValueError):d.source_evidence(rows[:-1])
