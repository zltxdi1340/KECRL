"""Human vectors and metadata only; research numerical payload access is forbidden."""
import builtins
import copy
import io
import json
import math
from pathlib import Path
import pytest
from experiments import libero_task7_m_priority_direction_protocol as d


@pytest.fixture(scope='module')
def canonical():return d.create()


def pad(values):return list(values)+[0.]*(257-len(values))


def state(meta):
    return dict(total_l2=meta['total_norm'],direction_status=meta['status'],direction_construction_status=meta['construction_status'],
      direction_total_reference_norm=meta['total_norm'],direction_base_reference_norm=meta['base_norm'],
      selected_T_slope=meta['selected_T_slope'],selected_MSE_slope=meta['selected_MSE_slope'],
      directional_derivative=meta['selected_T_slope'],exact_MSE_slope=meta['selected_MSE_slope'],
      required_M_slope_bound=meta['required_M_slope_bound'],required_margin_fraction=meta['required_margin_fraction'],
      construction_margin_fraction=meta['construction_margin_fraction'],selected_joint=meta['selected_joint'],selected_margin_met=meta['selected_margin_met'])


def test_all18_original500_same_loss_data_roles_and_no_endpoint_continuation(canonical):
    prior=d.parent.validate()
    for field in ('inputs','reference_points','cohorts','source_members','head_members','new_head_members','audit_head_members'):
        assert canonical[field]==prior[field]
    scope=canonical['scope'];assert scope['original_Policy_updates']==scope['source_auxiliary_updates']==500
    assert scope['future_stabilization_steps']==5 and not scope['new_loss_lambda_representation_or_data_weight_change']
    assert not scope['source_meta_query_validation_heldout_qualification_reserved_final_access']
    assert len(canonical['inputs']['heads'])==18
    for field in ('same_loss','same_precision','start','extra_updates_max','alpha_restart','final','slope'):
        assert canonical['optimization'][field]==prior['optimization'][field]


def test_control_is_old_guarded_candidate_with7691_trials_and_no_M_up(canonical):
    hist=canonical['historical_control'];assert hist['optimization_arm']==d.parent.ARMS[1]==d.ARMS[0]
    assert canonical['optimization_parent_science']==d.evidence.PARENTS['science']
    assert canonical['optimization_parent_binding']==d.evidence.PARENTS['binding']
    assert hist['primary_source_JSON']==d.HISTORY_INPUTS['P'] and hist['audit_source_JSON']==d.HISTORY_INPUTS['U']
    for c in hist['counts_by_engine'].values():
        assert (c['updates'],c['states'],c['trials'],c['endpoints'],c['heads'],c['accepted_M_up_steps'])==(900,918,7691,36,18,0)
    assert all(f in hist['projections']['trials'] for f in ('step_mse_nonincrease_pass','step_mse_gate_applied'))
    assert not canonical['budget']['after_start_pair_prefix_required']
    assert not any('shared_prefix' in key for key in canonical['budget'])


def test_four_references_preserve_old_margin_and_only_same_JSON_parse(canonical):
    refs=canonical['fixed_references'];assert all(refs[k]['rows_per_engine']==18 for k in ('margin','zero','bisector'))
    assert refs['margin']['condition']==d.parent.ARMS[0]+'_final'
    assert refs['zero']['table']=='historical_control_parity.fixed_anchors.zero'
    assert refs['bisector']['table']=='historical_control_parity.fixed_anchors.bisector'
    assert 'all4' in canonical['decision']['primary_head_cue']


def test_budget_conservation_and_tight_7691_control(canonical):
    b=canonical['budget'];P,U,S=b['per_engine']['P'],b['per_engine']['U'],b['S']
    assert (P['updates_max'],U['updates_max'],S['updates_max'])==(1800,1800,0)
    assert b['total_auxiliary_updates_max']==3600 and b['total_gradient_states_max']==5508
    assert b['total_gradient_vectors_max']==16524 and b['audit_total_autograd_max']==11016
    assert b['total_trial_evaluations_max']==77073 and b['total_linear_forms_max']==89925
    assert b['total_linear_rows_max']==251647425==sum(v['linear_rows_max'] for v in (P,U,S))
    assert P['trials_max']==U['trials_max']==S['trials_max']==25691
    assert (P['state_fit_rows_max'],P['trial_fit_rows_max'])==(5126112,71921547)
    assert b['trial_objective_forms_max_by_arm']==dict(zip(d.ARMS,(7691,18000)))
    assert (b['output_CSV_rows_max'],b['audit_local_CSV_rows_max'],b['total_CSV_rows_max'])==(41531,27527,69058)
    assert b['step_M_gate_evaluations_max']==77073 and b['step_M_record_rechecks_max']==102764 and b['step_M_total_scalar_comparisons_max']==179837
    assert (b['total_NPZ_loads'],b['total_member_decodes'],b['total_archive_SHA_checks'])==(68,574,172)
    assert b['primary_new_member_writes']+b['audit_new_member_writes']==630


@pytest.mark.parametrize('total,base',[([1.,0.],[1.,0.]),([1.,0.],[-1.,1.]),([0.,0.],[1.,0.]),([1.,0.],[0.,0.]),([1.,2.],[2.,1.])])
def test_control_direction_exactly_keeps_previous_arithmetic(total,base):
    assert d.direction(d.ARMS[0],pad(total),pad(base))==d.parent.direction(d.parent.ARMS[1],pad(total),pad(base))


@pytest.mark.parametrize('total,base',[([3.,4.],[1.,2.]),([1.,0.],[.4,1.]),([1.,0.],[-.5,1.]),([1e-100,2e-100],[3.,4.]),([1.,0.],[1e-100,2e-100])])
def test_M_priority_matches_independent_vector_formula_and_norm_ball_bound(total,base):
    r=d.direction(d.ARMS[1],pad(total),pad(base));g=math.sqrt(sum(x*x for x in total));b=math.sqrt(sum(x*x for x in base))
    expected=[-x/b*g for x in base]
    assert r['direction'][:2]==pytest.approx(expected,rel=1e-13,abs=0)
    assert r['direction_norm']==pytest.approx(g,rel=1e-13,abs=0)
    assert r['selected_MSE_slope']==pytest.approx(-7*g*b,rel=1e-13,abs=0)
    assert r['selected_margin_met'] and r['construction_margin_fraction'] is None
    # A fixed human circle checks the claimed norm-ball linear optimum only.
    for j in range(72):
        theta=2*math.pi*j/72;sample=g*(base[0]*math.cos(theta)+base[1]*math.sin(theta))
        assert r['selected_MSE_slope']/7<=sample+1e-13*g*b


@pytest.mark.parametrize('base',[[-.5,1.],[-1.,0.],[0.,1.]])
def test_conflicting_or_orthogonal_M_direction_stops_without_projection(base):
    meta=d.direction(d.ARMS[1],pad([1.,0.]),pad(base));row=state(meta)
    assert meta['selected_margin_met'] and not meta['selected_joint']
    assert d.stopping_reason(d.ARMS[1],0,row)=='non_descent_selected_direction'
    assert meta['projection_norm'] is None and meta['unit_cosine'] is None
    if base==[-.5,1.]:
        control=d.direction(d.ARMS[0],pad([1.,0.]),pad(base))
        assert control['selected_joint'] # Stopping M-steepest is not global infeasibility.


@pytest.mark.parametrize('total,base,reason',[([0.,0.],[1.,0.],'zero_gradient'),([1.,0.],[0.,0.],'zero_base_margin_unavailable'),([1e-200,0.],[1e-200,0.],'margin_unrepresentable')])
def test_degenerate_and_underflow_cases_keep_zero_placeholder(total,base,reason):
    meta=d.direction(d.ARMS[1],pad(total),pad(base));assert meta['direction']==[0.]*257
    assert d.stopping_reason(d.ARMS[1],0,state(meta))==reason


@pytest.mark.parametrize('field,value,reason',[
    ('selected_MSE_slope',1e-30,'selected_M_margin_violation'),
    ('selected_T_slope',1e-30,'non_descent_selected_direction'),
    ('exact_MSE_slope',1e-30,'exact_M_margin_violation'),
    ('directional_derivative',0.,'non_descent_selected_direction'),
])
def test_selected_and_exact_raw_sign_guards_and_stop_priority(field,value,reason):
    row=state(d.direction(d.ARMS[1],pad([1.,0.]),pad([1.,0.])));row[field]=value
    assert d.stopping_reason(d.ARMS[1],0,row)==reason
    assert d.stopping_reason(d.ARMS[1],50,row,20)=='update_cap'


def test_exact_nextafter_margin_violation_is_not_repaired_by_tolerance():
    row=state(d.direction(d.ARMS[1],pad([1.,0.]),pad([1.,0.])))
    row['exact_MSE_slope']=math.nextafter(row['required_M_slope_bound'],math.inf)
    assert row['exact_MSE_slope']-row['required_M_slope_bound']<1e-10
    assert d.stopping_reason(d.ARMS[1],0,row)=='exact_M_margin_violation'


@pytest.mark.parametrize('fault',['short','tuple','bool','nan','inf','nonscalar','overflow'])
def test_invalid_vectors_and_unrepresentable_arithmetic_fail_closed(fault):
    total,base=pad([1.,0.]),pad([1.,0.])
    if fault=='short':base.pop()
    elif fault=='tuple':base=tuple(base)
    elif fault=='bool':base[0]=True
    elif fault=='nan':base[0]=float('nan')
    elif fault=='inf':base[0]=float('inf')
    elif fault=='nonscalar':base[0]=[]
    else:total[0]=base[0]=1e200
    with pytest.raises(ValueError):d.direction(d.ARMS[1],total,base)


@pytest.mark.parametrize('field,value',[('required_margin_fraction',0.),('construction_margin_fraction',.02),('required_M_slope_bound',0.),('selected_margin_met',False),('selected_joint',False)])
def test_forged_M_priority_certificate_fails(field,value):
    row=state(d.direction(d.ARMS[1],pad([1.,0.]),pad([1.,0.])));assert d.validate_margin_record(d.ARMS[1],row)
    row[field]=value
    with pytest.raises(ValueError):d.validate_margin_record(d.ARMS[1],row)


@pytest.mark.parametrize('arm',d.ARMS)
@pytest.mark.parametrize('mse',[.07,.08,math.nextafter(.08,math.inf),.09,.11])
def test_both_arms_keep_exact_T_Armijo_M500_and_current_M_gates(arm,mse):
    expected=d.parent.trial_decision(d.parent.ARMS[1],.02,.019,-.01,0,mse,.1,.08)
    assert d.trial_decision(arm,.02,.019,-.01,0,mse,.1,.08)==expected
    assert expected['step_mse_gate_applied'] is True


@pytest.mark.parametrize('arm',d.ARMS)
def test_complete_twenty_rejections_and_raw_remeasured_M_gate(arm):
    rows=[]
    for j in range(20):
        decision=d.trial_decision(arm,.02,.021,-.01,j,.08,.1,.08)
        rows.append(dict(trial_index=j,alpha=d.trial_alpha(arm,j),trainable_objective=.021,component3_mse=.08,**decision))
    assert d.validate_round(arm,.02,-.01,.1,.08,rows)['exhausted']
    with pytest.raises(ValueError):d.validate_round(arm,.02,-.01,.1,.08,rows[:-1])
    with pytest.raises(ValueError):d.validate_state_transition(arm,.08,math.nextafter(.08,math.inf),.1)


@pytest.mark.parametrize('index',[-1,20,True,1.])
def test_grid_cannot_expand_or_coerce(index):
    with pytest.raises(ValueError):d.trial_alpha(d.ARMS[1],index)


@pytest.mark.parametrize('bad_anchor',['margin','zero','bisector'])
def test_retained_reference_cannot_be_replaced_by_weaker_new_control(bad_anchor):
    control={'T':.02,'M':.1};candidate={'T':.019,'M':.09};anchors={k:{'T':.021,'M':.095} for k in ('margin','zero','bisector')}
    assert d.paired_evidence(.03,.12,control,candidate,anchors,1,[-.03])['paired_joint_cue']
    anchors[bad_anchor]['M']=.08
    result=d.paired_evidence(.03,.12,control,candidate,anchors,1,[-.03])
    assert result['guarded_T_control_pair_cue'] and not result['paired_joint_cue']


def test_zero_update_and_monotonicity_alone_never_become_improvement():
    anchors={k:{'T':.04,'M':.2} for k in ('margin','zero','bisector')}
    result=d.paired_evidence(.03,.12,{'T':.04,'M':.2},{'T':.03,'M':.12},anchors,0,[])
    assert result['step_M_invariant_is_not_evidence'] and not result['paired_joint_cue']
    result=d.paired_evidence(.03,.12,{'T':.02,'M':.1},{'T':.021,'M':.09},anchors,1,[-.03])
    assert not result['guarded_T_control_pair_cue']


@pytest.mark.parametrize('section,field,value',[
    ('scope','original_Policy_updates',1000),('scope','future_stabilization_steps',0),
    ('scope','new_loss_lambda_representation_or_data_weight_change',True),
    ('scope','source_meta_query_validation_heldout_qualification_reserved_final_access',True),
    ('optimization','extra_updates_max',100),('optimization','step_gate','Mtrial<=M500'),
    ('budget','total_trial_evaluations_max',100000),('decision','primary_head_cue','monotonicity_only'),
    ('execution','technical_retries',1),
])
def test_changed_science_is_rejected(canonical,tmp_path,monkeypatch,section,field,value):
    changed=copy.deepcopy(canonical);changed[section][field]=value;path=tmp_path/'changed.json';path.write_text(json.dumps(changed))
    monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(ValueError):d.validate(path)


def test_factory_forbids_all_current_numeric_payload_reads_even_SHA(canonical,monkeypatch):
    forbidden={r['path'] for r in d.HISTORY_INPUTS.values()}|{str(d.evidence.OUTPUT),str(d.evidence.AUDIT)}
    open0,io0=builtins.open,io.open
    def check(path):
        if isinstance(path,(str,Path)):
            assert str(Path(path).resolve()) not in forbidden
            assert Path(path).suffix not in {'.npz','.npy','.csv','.pt','.h5','.hdf5'}
    def op(path,*args,**kw):check(path);return open0(path,*args,**kw)
    def ip(path,*args,**kw):check(path);return io0(path,*args,**kw)
    monkeypatch.setattr(builtins,'open',op);monkeypatch.setattr(io,'open',ip)
    monkeypatch.setattr(d,'direction',lambda *args:(_ for _ in ()).throw(AssertionError('no research direction')))
    assert d.create()==canonical


def test_scientific_freeze_does_not_authorize_execution(canonical,tmp_path,monkeypatch):
    path=tmp_path/'science.json';path.write_text(json.dumps(canonical));monkeypatch.setattr(d,'create',lambda:canonical)
    with pytest.raises(FileNotFoundError):d.require_binding(path,tmp_path/'unbound.json')
    assert not Path(d.OUTPUT).exists() and not Path(d.ARTIFACTS).exists()


def test_all18_both_sources_each_nine_required(canonical):
    rows=[{**{f:h[f] for f in d.IDS},'paired_joint_cue':True} for h in canonical['inputs']['heads']]
    assert d.source_evidence(rows)['population_direction_fit_cue']
    rows[0]['paired_joint_cue']=False;assert not d.source_evidence(rows)['population_direction_fit_cue']
    with pytest.raises(ValueError):d.source_evidence(rows[:-1])
