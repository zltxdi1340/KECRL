"""Metadata and human-made references for the one-extra-comparison intervention."""
import builtins
import copy
import io
import json
import math
from pathlib import Path

import pytest

from experiments import libero_task7_stepwise_m_nonincrease_acceptance_protocol as d


@pytest.fixture(scope='module')
def canonical():
    return d.create()


def row(arm, j, mse, total, *, current_m=.08, m500=.1, current_t=.02, slope=-.01):
    return dict(trial_index=j, alpha=d.trial_alpha(arm, j), component3_mse=mse,
                penalty=total-mse/7, trainable_objective=total,
                **d.trial_decision(arm, current_t, total, slope, j, mse, m500, current_m))


def test_all18_500_same_loss_direction_roles_and_no_endpoint_continuation(canonical):
    old=d.parent.validate()
    for field in ('inputs','reference_points','cohorts','source_members','head_members','new_head_members','audit_head_members'):
        assert canonical[field]==old[field]
    assert len(canonical['inputs']['heads'])==18
    assert canonical['scope']['original_Policy_updates']==canonical['scope']['source_auxiliary_updates']==500
    assert canonical['scope']['future_stabilization_steps']==5
    assert not canonical['scope']['new_loss_lambda_representation_or_data_weight_change']
    assert not canonical['scope']['source_meta_query_validation_heldout_qualification_reserved_final_access']
    for field in ('same_loss','same_precision','start','extra_updates_max','alpha_restart','final','slope','angle_resolution'):
        assert canonical['optimization'][field]==old['optimization'][field]
    assert canonical['optimization']['required_margin_fraction_by_arm']=={arm:.01 for arm in d.ARMS}
    assert canonical['optimization']['construction_margin_multiplier']==2.


def test_control_is_900_negative_margin_not_earlier_zero_or_bisector(canonical):
    hist=canonical['historical_control']
    assert hist['optimization_arm']==d.parent.ARMS[1]==d.ARMS[0]
    for engine in ('P','U'):
        c=hist['counts_by_engine'][engine]
        assert (c['updates'],c['states'],c['trials'],c['endpoints'],c['heads'],c['accepted_M_up_steps'])==(900,918,7017,36,18,407)
    assert hist['primary_source_JSON']==d.HISTORY_INPUTS['P']
    assert hist['audit_source_JSON']==d.HISTORY_INPUTS['U']
    assert 'directional_derivative' not in hist['saved500_endpoint_projection']
    assert 'negative_margin_state_k0' in hist['saved500_direction_reference']
    assert all(f in hist['projections']['states'] for f in ('required_M_slope_bound','selected_margin_met','exact_margin_met'))
    refs=canonical['fixed_references']
    assert refs['zero']['rows_per_engine']==refs['bisector']['rows_per_engine']==18
    assert refs['bisector']['table']=='historical_control_parity.bisector_anchors'
    assert refs['zero']['condition']==d.parent.ARMS[0]+'_final'


def test_tight_P_U_S_budgets_and_no_original_Policy_updates(canonical):
    b=canonical['budget'];P,U,S=[b['per_engine']['P'],b['per_engine']['U'],b['S']]
    assert (P['updates_max'],U['updates_max'],S['updates_max'])==(1800,1800,0)
    assert b['total_auxiliary_updates_max']==3600
    assert b['total_gradient_states_max']==5508 and b['total_gradient_vectors_max']==16524
    assert b['audit_total_autograd_max']==11016
    assert b['total_trial_evaluations_max']==75051
    assert b['total_linear_forms_max']==87903 and b['total_linear_rows_max']==245951451
    assert b['total_linear_rows_max']==sum(c['linear_rows_max'] for c in (P,U,S))
    assert (P['state_fit_rows_max'],P['trial_fit_rows_max'])==(5126112,70022889)
    assert P['trials_max']==U['trials_max']==S['trials_max']==25017
    assert b['trial_objective_forms_max_by_arm']==dict(zip(d.ARMS,(7017,18000)))
    assert (b['output_CSV_rows_max'],b['audit_local_CSV_rows_max'],b['total_CSV_rows_max'])==(40857,26853,67710)
    assert b['total_NPZ_loads']==68 and b['total_member_decodes']==574 and b['total_archive_SHA_checks']==172
    assert b['primary_new_member_writes']+b['audit_new_member_writes']==630
    assert b['step_M_gate_evaluations_max']==75051 and b['step_M_record_rechecks_max']==100068
    assert b['step_M_total_scalar_comparisons_max']==175119
    assert b['total_shared_prefix_round_checks_max']==1800 and b['total_shared_prefix_trial_checks_max']==14034


@pytest.mark.parametrize('total,base',[([1.,0.],[-1.,1.]),([1.,2.],[2.,1.]),([0.,0.],[1.,0.]),([1.,0.],[0.,0.]),([1.,0.],[-1.,0.])])
def test_both_arms_preserve_identical_negative_margin_formula(total,base):
    total+= [0.]*255; base += [0.]*255
    old=d.parent.direction(d.parent.ARMS[1],total,base)
    for arm in d.ARMS:
        assert d.direction(arm,total,base)==old


def margin_state(meta):
    return dict(total_l2=meta['total_norm'],direction_status=meta['status'],
        direction_construction_status=meta['construction_status'],
        direction_total_reference_norm=meta['total_norm'],direction_base_reference_norm=meta['base_norm'],
        selected_T_slope=meta['selected_T_slope'],selected_MSE_slope=meta['selected_MSE_slope'],
        directional_derivative=meta['selected_T_slope'],exact_MSE_slope=meta['selected_MSE_slope'],
        required_M_slope_bound=meta['required_M_slope_bound'],required_margin_fraction=meta['required_margin_fraction'],
        construction_margin_fraction=meta['construction_margin_fraction'],selected_joint=meta['selected_joint'],
        selected_margin_met=meta['selected_margin_met'])


@pytest.mark.parametrize('arm',d.ARMS)
def test_both_arms_keep_margin_certificate_and_stop_order(arm):
    state=margin_state(d.direction(arm,[1.]+[0.]*256,[1.]+[0.]*256))
    assert d.validate_margin_record(arm,state)
    assert d.stopping_reason(arm,0,state) is None
    assert d.stopping_reason(arm,0,state,20)=='line_search_exhausted'
    assert d.stopping_reason(arm,50,state,20)=='update_cap'
    forged={**state,'required_margin_fraction':0.}
    with pytest.raises(ValueError):d.validate_margin_record(arm,forged)


@pytest.mark.parametrize('arm',d.ARMS)
def test_zero_direction_remains_zero_stop_without_extra_steps(arm):
    state=margin_state(d.direction(arm,[0.]*257,[1.]+[0.]*256))
    assert d.stopping_reason(arm,0,state)=='zero_gradient'


@pytest.mark.parametrize('mse',[.07,.08,.09,.11])
def test_control_original_flags_and_reasons_remain_exact(mse):
    old=d.parent.trial_decision(d.parent.ARMS[1],.02,.019,-.01,0,mse,.1)
    new=d.trial_decision(d.ARMS[0],.02,.019,-.01,0,mse,.1,.08)
    assert {k:new[k] for k in old}==old
    assert not new['step_mse_gate_applied'] and new['step_mse_nonincrease_pass']==(mse<=.08)


def test_single_gate_uses_current_own_state_not_M500():
    control=d.trial_decision(d.ARMS[0],.02,.019,-.01,0,.09,.1,.08)
    candidate=d.trial_decision(d.ARMS[1],.02,.019,-.01,0,.09,.1,.08)
    assert control['accepted'] and not candidate['accepted'] and candidate['mse_envelope_pass']
    assert candidate['reason']=='step_mse_nonincrease_rejected'
    # An equal-M step remains valid when T improves through P.
    assert d.trial_decision(d.ARMS[1],.02,.019,-.01,0,.08,.1,.08)['accepted']


def test_raw_step_boundary_no_epsilon_and_no_P_monotonicity_gate():
    up=math.nextafter(.08,math.inf)
    assert up-.08<1e-10
    assert not d.trial_decision(d.ARMS[1],.02,.019,-.01,0,up,.1,.08)['accepted']
    # Large M saving can accompany a larger P while still lowering T.
    assert d.trial_decision(d.ARMS[1],.02,.0195,-.01,0,.01,.1,.08)['accepted']


@pytest.mark.parametrize('j',[-1,20,True,1.])
def test_no_grid_extension_or_index_coercion(j):
    with pytest.raises(ValueError):d.trial_alpha(d.ARMS[1],j)


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),True,-.1])
def test_invalid_M_scalar_fail_closed(bad):
    with pytest.raises(ValueError):d.trial_decision(d.ARMS[1],.02,.019,-.01,0,bad,.1,.08)


def test_current_state_above_original500_is_technical_failure():
    with pytest.raises(ValueError):d.trial_decision(d.ARMS[1],.02,.019,-.01,0,.09,.1,.11)


def test_first_acceptance_and_complete_twenty_rejections():
    candidate=[row(d.ARMS[1],j,.09,.019) for j in range(20)]
    result=d.validate_round(d.ARMS[1],.02,-.01,.1,.08,candidate)
    assert result['exhausted'] and result['accepted_index'] is None and result['step_vetoes_of_original_acceptances']==20
    with pytest.raises(ValueError):d.validate_round(d.ARMS[1],.02,-.01,.1,.08,candidate[:-1])
    with pytest.raises(ValueError):d.validate_round(d.ARMS[0],.02,-.01,.1,.08,[row(d.ARMS[0],j,.09,.019) for j in range(2)])


def test_raw_candidate_remeasured_state_invariant_not_repaired_by_tolerance():
    assert d.validate_state_transition(d.ARMS[0],.08,.09,.1)['step_M_nonincrease'] is False
    assert d.validate_state_transition(d.ARMS[1],.08,.08,.1)['enforced']
    with pytest.raises(ValueError):d.validate_state_transition(d.ARMS[1],.08,math.nextafter(.08,math.inf),.1)


def test_shared_prefix_identical_until_first_actual_step_veto():
    control=[row(d.ARMS[0],0,.09,.019)]
    candidate=[row(d.ARMS[1],0,.09,.019),row(d.ARMS[1],1,.079,.0199)]
    result=d.validate_shared_round(.02,-.01,.1,.08,control,candidate)
    assert result['first_gate_veto'] and result['prefix_trials_checked']==1 and result['no_new_path_prediction']
    assert candidate[-1]['trainable_objective']>control[-1]['trainable_objective']


def test_no_shared_prefix_divergence_when_old_acceptance_is_step_feasible():
    control=[row(d.ARMS[0],0,.09,.021),row(d.ARMS[0],1,.079,.019)]
    candidate=[row(d.ARMS[1],0,.09,.021),row(d.ARMS[1],1,.079,.019)]
    assert not d.validate_shared_round(.02,-.01,.1,.08,control,candidate)['first_gate_veto']
    changed=copy.deepcopy(candidate);changed[0]['penalty']+=1e-18
    with pytest.raises(ValueError):d.validate_shared_round(.02,-.01,.1,.08,control,changed)


def test_monotonic_gate_alone_cannot_be_claimed_as_joint_improvement():
    control={'T':.02,'M':.09};candidate={'T':.021,'M':.08}
    anchors={'zero':{'T':.025,'M':.095},'bisector':{'T':.024,'M':.085}}
    result=d.paired_evidence(.03,.12,control,candidate,anchors,1,[-.04])
    assert result['step_M_invariant_met'] and result['step_M_invariant_is_not_evidence']
    assert not result['margin_control_pair_cue'] and not result['paired_joint_cue']


@pytest.mark.parametrize('bad_anchor',['zero','bisector'])
def test_no_weaker_reference_replaces_prior_scientific_conditions(bad_anchor):
    control={'T':.02,'M':.1};candidate={'T':.019,'M':.09}
    anchors={'zero':{'T':.021,'M':.095},'bisector':{'T':.020,'M':.093}}
    assert d.paired_evidence(.03,.12,control,candidate,anchors,1,[-.03])['paired_joint_cue']
    anchors[bad_anchor]['M']=.08
    result=d.paired_evidence(.03,.12,control,candidate,anchors,1,[-.03])
    assert result['margin_control_pair_cue'] and not result['paired_joint_cue']


def test_zero_update_or_raw_positive_step_never_becomes_success():
    anchors={k:{'T':.04,'M':.2} for k in ('zero','bisector')}
    assert not d.paired_evidence(.03,.12,{'T':.04,'M':.2},{'T':.03,'M':.12},anchors,0,[])['paired_joint_cue']
    with pytest.raises(ValueError):d.paired_evidence(.03,.12,{'T':.04,'M':.2},{'T':.02,'M':.12},anchors,1,[1e-18])


@pytest.mark.parametrize('section,field,value',[
    ('scope','original_Policy_updates',1000),('scope','future_stabilization_steps',0),
    ('scope','new_loss_lambda_representation_or_data_weight_change',True),
    ('scope','source_meta_query_validation_heldout_qualification_reserved_final_access',True),
    ('optimization','step_gate','Mtrial<=M500'),('optimization','extra_updates_max',100),
    ('budget','total_auxiliary_updates_max',7200),('budget','total_trial_evaluations_max',100000),
    ('decision','primary_head_cue','enforced_monotonicity_only'),('execution','technical_retries',1),
])
def test_changed_science_rejected(canonical,tmp_path,monkeypatch,section,field,value):
    changed=copy.deepcopy(canonical);changed[section][field]=value;path=tmp_path/'bad.json';path.write_text(json.dumps(changed))
    monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(ValueError):d.validate(path)


def test_factory_denies_archives_and_all_numerical_source_JSON_even_SHA(canonical,monkeypatch):
    forbidden={r['path'] for r in d.HISTORY_INPUTS.values()}|{str(d.evidence.OUTPUT),str(d.evidence.AUDIT)}
    original=builtins.open,io.open
    def check(path):
        if isinstance(path,(str,Path)):
            assert str(Path(path).resolve()) not in forbidden
            assert Path(path).suffix not in {'.npz','.npy','.csv','.pt','.h5','.hdf5'}
    def op(path,*args,**kwargs):check(path);return original[0](path,*args,**kwargs)
    def ip(path,*args,**kwargs):check(path);return original[1](path,*args,**kwargs)
    monkeypatch.setattr(builtins,'open',op);monkeypatch.setattr(io,'open',ip)
    monkeypatch.setattr(d,'direction',lambda *args:(_ for _ in ()).throw(AssertionError('no actual direction')))
    assert d.create()==canonical


def test_scientific_freeze_cannot_authorize_unbound_execution(canonical,tmp_path,monkeypatch):
    path=tmp_path/'science.json';path.write_text(json.dumps(canonical));monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(FileNotFoundError):d.require_binding(path,tmp_path/'missing-execution-binding.json')


def test_all18_and_each_source_nine_required(canonical):
    rows=[{**{k:h[k] for k in d.IDS},'paired_joint_cue':True} for h in canonical['inputs']['heads']]
    assert d.source_evidence(rows)['population_direction_fit_cue']
    rows[0]['paired_joint_cue']=False
    assert not d.source_evidence(rows)['population_direction_fit_cue']
    with pytest.raises(ValueError):d.source_evidence(rows[:-1])
