"""Scientific freeze only: norm-matched negative-M direction, unchanged T/step-M gates."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_stepwise_m_nonincrease_acceptance_protocol as parent
from experiments import libero_task7_stepwise_m_reference_tradeoff_protocol as evidence

p, ROOT, IDS = parent.p, parent.ROOT, parent.IDS
margin = parent.parent
ARMS = [parent.ARMS[1], 'norm_matched_M_priority_with_step_M_gate']
STEM = 'libero-task7-m-priority-direction'
CONFIG = ROOT / f'configs/{STEM}-diagnostic_v1.json'
FREEZE_BINDING = ROOT / f'configs/{STEM}-freeze-binding_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT / f'docs/{STEM}-plan-2026-11-11.md'
TESTS = ROOT / 'tests/test_libero_task7_m_priority_direction_protocol.py'
OUTPUT = f'/tmp/{STEM}-20261111-v1.json'
ARTIFACTS = f'/tmp/{STEM}-artifacts-20261111-v1'
PARENTS = {
 'science': {'path': str(evidence.CONFIG), 'sha256': 'e22ec131b5c283c037ed04a34bd4840a025529920908da79ea6dc80b809990de'},
 'binding': {'path': str(evidence.BINDING), 'sha256': '924617a59298bc856eb3eb98afb67b858dcdeb0516edf7b5c62d38806aa04f1f'},
 'completion': {'path': str(evidence.ARTIFACTS/'completion.json'), 'sha256': '290fa88ed90d95d65276ec403b4771f918ad0a2e2ae5b95a87b5425f52909d9e'},
 'process': {'path': str(evidence.ARTIFACTS.with_name(evidence.ARTIFACTS.name+'-process.json')), 'sha256': '617c3cd948cf10f5c563c2c7c74f51827146df4219a0171aa6a43339bc0f5106'},
 'report': {'path': str(ROOT/'docs/libero-task7-stepwise-m-reference-tradeoff-results-2026-11-11.md'), 'sha256': 'f3a8d0567428f19404d96c0a6e25bb1db6b4bb43965d67a9c287982b895068aa'},
}
HISTORY_INPUTS = copy.deepcopy(evidence.INPUTS)
CONTROL_COUNTS = {engine: dict(updates=900,states=918,trials=7691,endpoints=36,heads=18,
                              accepted_M_up_steps=0,margin_anchor_endpoints=18,zero_anchor_endpoints=18,bisector_anchor_endpoints=18)
                  for engine in ('P','U')}
FUTURE_FILES = {
 'primary': ROOT/'experiments/run_libero_task7_m_priority_direction_diagnostic.py',
 'audit': ROOT/'experiments/audit_libero_task7_m_priority_direction_diagnostic.py',
 'guard': ROOT/'experiments/libero_task7_m_priority_direction_access.py',
 'runtime_tests': ROOT/'tests/test_libero_task7_m_priority_direction_diagnostic.py',
}
FUTURE_CONTRACT = ROOT/f'docs/{STEM}-execution-contract-2026-11-11.md'


def direction(arm,total,base):
    """Human-made vector reference; future source execution needs new bound roles."""
    p.require(arm in ARMS,'unregistered M-priority arm')
    if arm==ARMS[0]:return parent.direction(parent.ARMS[1],total,base)
    p.require(all(type(v) is list and len(v)==257 for v in (total,base)),'two257 vectors required')
    margin.parent.finite(*total,*base)
    g,b=math.hypot(*total),math.hypot(*base);margin.parent.finite(g,b)
    bound=margin.required_bound(g,b,margin.REQUIRED_MARGIN)
    if g==0:values=[0.]*257;status='zero_total_gradient'
    elif b==0:values=[0.]*257;status='zero_base_margin_unavailable'
    elif bound is None:values=[0.]*257;status='margin_unrepresentable'
    else:values=[-g*(x/b) for x in base];status='negative_base_norm_matched'
    margin.parent.finite(*values)
    norm=math.hypot(*values);st=math.fsum(x*y for x,y in zip(total,values));sm=7.*math.fsum(x*y for x,y in zip(base,values))
    margin.parent.finite(norm,st,sm);construction=status;met=bound is not None and sm<=bound
    if status=='negative_base_norm_matched':
        if not met:status='selected_M_margin_violation'
        elif st>=0:status='selected_T_non_descent'
    return dict(direction=values,status=status,construction_status=construction,total_norm=g,base_norm=b,
      unit_sum_norm=None,unit_cosine=None,projection_norm=None,direction_norm=norm,selected_T_slope=st,selected_MSE_slope=sm,
      selected_joint=met and st<0,required_margin_fraction=margin.REQUIRED_MARGIN,construction_margin_fraction=None,
      required_M_slope_bound=bound,selected_margin_met=met)


def validate_margin_record(arm,row):
    p.require(arm in ARMS,'unregistered M-priority certificate')
    if arm==ARMS[0]:return parent.validate_margin_record(parent.ARMS[1],row)
    bound=margin.required_bound(row['direction_total_reference_norm'],row['direction_base_reference_norm'],margin.REQUIRED_MARGIN)
    sm,st=row['selected_MSE_slope'],row['selected_T_slope'];margin.parent.finite(sm,st)
    met=bound is not None and sm<=bound
    p.require(row['required_margin_fraction']==margin.REQUIRED_MARGIN and row['construction_margin_fraction'] is None
      and row['required_M_slope_bound']==bound and type(row['selected_margin_met']) is type(row['selected_joint']) is bool
      and row['selected_margin_met']==met and row['selected_joint']==(met and st<0),'forged M-priority certificate')
    return True


def stopping_reason(arm,updates,row,rejected_trials=0):
    p.require(arm in ARMS,'unregistered M-priority stop arm')
    if arm==ARMS[0]:return parent.stopping_reason(parent.ARMS[1],updates,row,rejected_trials)
    p.require(type(updates) is int and 0<=updates<=50 and type(rejected_trials) is int and 0<=rejected_trials<=20,'invalid M-priority stop index')
    g,b=row['direction_total_reference_norm'],row['direction_base_reference_norm']
    st,sm,et,em=[row[f] for f in ('selected_T_slope','selected_MSE_slope','directional_derivative','exact_MSE_slope')]
    margin.parent.finite(g,b,st,sm,et,em);p.require(g>=0 and b>=0,'negative norm')
    p.require(row['direction_construction_status'] in {'zero_total_gradient','zero_base_margin_unavailable','margin_unrepresentable','negative_base_norm_matched'},'unlisted M-priority construction')
    if updates==50:return 'update_cap'
    if g==0:
        p.require(st==sm==et==em==0,'nonzero zero-direction slope');return 'zero_gradient'
    if b==0:return 'zero_base_margin_unavailable'
    bound=row['required_M_slope_bound']
    if bound is None:return 'margin_unrepresentable'
    margin.parent.finite(bound);p.require(bound<0,'M bound must be negative')
    if sm>bound:return 'selected_M_margin_violation'
    if st>=0:return 'non_descent_selected_direction'
    if em>bound:return 'exact_M_margin_violation'
    if et>=0:return 'non_descent_selected_direction'
    if rejected_trials==20:return 'line_search_exhausted'
    return None


def trial_alpha(arm,index):
    p.require(arm in ARMS,'unregistered M-priority grid');return parent.trial_alpha(parent.ARMS[1],index)


def trial_decision(arm,current,trial,slope,index,trial_mse,start_mse,current_mse):
    p.require(arm in ARMS,'unregistered M-priority trial')
    return parent.trial_decision(parent.ARMS[1],current,trial,slope,index,trial_mse,start_mse,current_mse)


def validate_round(arm,current,slope,start_mse,current_mse,rows):
    p.require(arm in ARMS,'unregistered M-priority round')
    return parent.validate_round(parent.ARMS[1],current,slope,start_mse,current_mse,rows)


def validate_state_transition(arm,previous_mse,measured_mse,start_mse):
    p.require(arm in ARMS,'unregistered M-priority state')
    return parent.validate_state_transition(parent.ARMS[1],previous_mse,measured_mse,start_mse)


def paired_evidence(t0,m0,control,candidate,anchors,candidate_updates,step_mse_deltas):
    p.require(set(anchors)=={'margin','zero','bisector'} and set(anchors['margin'])=={'T','M'},'all three retained anchors required')
    record=parent.paired_evidence(t0,m0,control,candidate,{k:anchors[k] for k in ('zero','bisector')},candidate_updates,step_mse_deltas)
    parent.finite_nonnegative(*anchors['margin'].values())
    margin_cue=record['candidate_progress_from500'] and candidate['T']<anchors['margin']['T'] and candidate['M']<=anchors['margin']['M']
    control_cue=record.pop('margin_control_pair_cue')
    return dict(record,guarded_T_control_pair_cue=control_cue,margin_anchor_cue=margin_cue,
                paired_joint_cue=record['paired_joint_cue'] and margin_cue)


source_evidence=parent.source_evidence


def budget():
    b=copy.deepcopy(parent.budget());s,sr=1836,2*50256*51;t=7691+18000;tr=7691*2817+50256*50*20
    for engine,caps in b['per_engine'].items():
        caps.update(trials_max=t,linear_forms_max=2*s+t,linear_rows_max=2*sr+tr,trial_fit_rows_max=tr)
    b['S'].update(trials_max=t,linear_forms_max=3*s+t,linear_rows_max=3*sr+tr)
    b.update(historical_control_counts_by_engine=copy.deepcopy(CONTROL_COUNTS),trial_caps_per_round={a:20 for a in ARMS},
      primary_trials_max=t,independent_U_trials_max=t,local_shadow_trial_evaluations_max=t,total_trial_evaluations_max=3*t,
      total_linear_forms_max=7*s+3*t,total_linear_rows_max=7*sr+3*tr,shadow_trial_parameter_coordinates_max=t*257,
      trial_objective_forms_max_per_stage=t,trial_MSE_and_penalty_pair_evaluations_max_per_stage=t,
      trial_objective_forms_max_by_arm={ARMS[0]:7691,ARMS[1]:18000},trial_component3_fit_row_visits_max_per_stage=tr,
      all_linear_forms_max_per_stage=2*s+t,all_linear_form_row_visits_max_per_stage=2*sr+tr,all_linear_form_feature_multiplications_max_per_stage=256*(2*sr+tr),
      local_S_linear_forms_max=3*s+t,local_S_linear_rows_max=3*sr+tr,audit_total_linear_forms_max=5*s+2*t,
      audit_total_linear_rows_max=5*sr+2*tr,step_M_gate_evaluations_max=3*t,step_M_record_rechecks_max=4*t,
      step_M_total_scalar_comparisons_max=7*t,after_start_pair_prefix_required=False,
      same_start_gradient_and_branch_comparisons_per_engine=18,margin_anchor_projection_rows_by_engine={'P':18,'U':18})
    for field in list(b):
        if 'shared_prefix' in field:b.pop(field)
    b['tables']['trials_max']=t;b['output_CSV_rows_max']=sum(b['tables'].values());b['audit_local_CSV_rows_max']=s+t
    b['total_CSV_rows_max']=b['output_CSV_rows_max']+b['audit_local_CSV_rows_max']
    b['total_linear_feature_multiplications_max']=256*b['total_linear_rows_max']
    b['direction_operation_caps_by_arm']={ARMS[0]:dict(norms=4,dots=3,vectors=4),ARMS[1]:dict(norms=3,dots=2,vectors=1)}
    return b


def create():
    evidence_cfg,evidence_binding=evidence.preflight()
    meta={k:p.load(v) for k,v in PARENTS.items() if k!='report'}
    p.require(p.pin(PARENTS['report']['path'])==PARENTS['report'],'completed endpoint tradeoff report changed')
    done,proc=meta['completion'],meta['process']
    p.require(meta['science']==evidence_cfg and meta['binding']==evidence_binding and len(evidence_binding['dependencies'])==873
      and done['status']==proc['status']=='complete_independently_audited_stepwise_reference_accounting'
      and done['science']==proc['science']==PARENTS['science'] and done['binding']==proc['binding']==PARENTS['binding']
      and done['checks']==6100 and done['source_JSON_passes']==4 and done['source_SHA_checks']==8
      and len(proc['processes'])==2 and all(r['exit_code']==0 for r in proc['processes']),'completed independently audited endpoint lineage required')
    base=parent.validate();cfg=copy.deepcopy(base)
    cfg.update(kind='task7_M_priority_direction_with_unchanged_T_and_step_M_acceptance',version=1,formal_result=False,
      status='scientific_freeze_only_execution_binding_required',parents=copy.deepcopy(PARENTS),budget=budget(),output=OUTPUT,artifacts=ARTIFACTS,
      optimization_parent_science=copy.deepcopy(evidence.PARENTS['science']),optimization_parent_binding=copy.deepcopy(evidence.PARENTS['binding']),
      code={'metadata':p.pin(Path(__file__).resolve()),'tests':p.pin(TESTS)},plan=p.pin(PLAN),
      future_files={k:str(path) for k,path in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT))
    cfg['scope'].update(changed_variable='only_direction_construction;candidate_normalized_negative_Gb_at_own_normG',
      head_coverage='all18_original_auxiliary500;not_eight_outcome_stratified_counterexamples',
      candidate_norm_matching='own_current_normG;not_control_path_norm;no_new_T_margin_or_direction_fallback')
    opt=cfg['optimization'];control_stop=copy.deepcopy(base['optimization']['stop_priority_by_arm'][parent.ARMS[1]])
    candidate_stop=[r for r in control_stop if r!='unresolved_margin_direction']
    opt.update(arms=ARMS,direction='control=previous_step_M_guarded_T_priority;candidate=norm_matched_negative_Gb',
      direction_definition=dict(G='gradient_T',Gb='gradient_M_over7',g='norm_G',b='norm_Gb',
        problem='min_Gb_dot_d_subject_norm_d<=g;T_and_M_certificates_are_admission_guards_not_new_objective',
        candidate='d_i=-g*(Gb_i/b);do_not_form_g_over_b_first',control='unchanged_negative_margin_T_priority_rho0.01_kappa0.02',
        optimality='Cauchy_Schwarz_Gb_dot_d>=-b*g;candidate_attains_bound_in_exact_arithmetic',
        zero_total='zero_placeholder_then_zero_gradient_stop',zero_base='zero_placeholder_then_zero_base_margin_unavailable',
        selected_and_exact='selected_and_exact_T<0;selected_and_exact_M<=-7*0.01*g*b',
        rounding='Float64;no_epsilon_or_feasibility_repair_or_reprojection;no_fallback'),
      required_margin_fraction_by_arm={a:.01 for a in ARMS},construction_margin_fraction_by_arm={ARMS[0]:.02,ARMS[1]:None},
      angle_resolution_applies_to='control_only;candidate_has_no_projection_or_angle_branch',
      acceptance={a:'original_strict_T_Armijo_and_Mtrial<=own_M500_and_Mtrial<=own_current_M' for a in ARMS},
      step_gate='raw_trial_MSE3<=own_measured_current_state_MSE3;both_arms;no_epsilon',
      mse_monotonicity='both_arms_enforced_trial_and_remeasured_state;invariant_not_success',
      stop_priority_by_arm={ARMS[0]:control_stop,ARMS[1]:candidate_stop},trial_caps={a:20 for a in ARMS},
      trial_index_range={a:[0,19] for a in ARMS},alpha_minima={a:2.**-19 for a in ARMS},
      control='same_engine_previous_guarded_T_priority900updates918states7691trials0Mup',
      path_order='seed_source_fold_then_guarded_T_control_then_M_priority_candidate',
      prefix='historical_control_full_path_only;paired_initial_parameter_3gradients_branch_same;directions_may_differ_at_k0',
      per_trial='one_linear_form;both_arms_original_T_Armijo_M500_and_step_M_gate',
      new_head_order=['saved500',*[a+'_final' for a in ARMS]])
    opt.pop('construction_margin_multiplier',None);cfg['new_head_layout']['order']=opt['new_head_order']
    hist=cfg['historical_control'];hist.update(source_science=evidence.PARENTS['science'],primary_source_JSON=HISTORY_INPUTS['P'],audit_source_JSON=HISTORY_INPUTS['U'],
      optimization_arm=ARMS[0],endpoint_conditions=['saved500',ARMS[0]+'_final'],counts_by_engine=copy.deepcopy(CONTROL_COUNTS),
      saved500_direction_reference='guarded_T_priority_k0;source_saved500_same_negative_margin_arithmetic',
      old_step_flags='present_in_source;original_and_step_flags_reasons_exact;both_arms_step_gate_applied')
    for f in ('step_mse_nonincrease_pass','step_mse_gate_applied'):
        if f not in hist['projections']['trials']:hist['projections']['trials'].append(f)
    fields=copy.deepcopy(base['fixed_references']['zero']['fields'])
    cfg['fixed_references']=dict(source='same_single_historical_JSON_parse;no_extra_file_or_training_arm',
      margin=dict(table='endpoints',condition=parent.ARMS[0]+'_final',rows_per_engine=18,fields=fields),
      zero=dict(table='historical_control_parity.fixed_anchors.zero',rows_per_engine=18,fields=fields),
      bisector=dict(table='historical_control_parity.fixed_anchors.bisector',rows_per_engine=18,fields=fields),
      role='preserve_all_three_prior_references_in_addition_to_new_guarded_T_control')
    cfg['decision'].update(primary_comparison='all18_vs_new_guarded_T_control_and_retained_margin_zero_bisector',
      primary_head_cue='updates>0_Tprogress_step_M_invariant_and_Tstrict_Mnonworse_vs_all4_references',
      aggregate='each_source_all9_all4_cues;both_sources_P_U_and_no_local_stop_acceptance_disagreements',
      baseline_parity='each_P_U_control900updates918states7691trials18heads0Mup;exact_original_bits_branches_flags_reasons',
      structural_invariants='same_original500_parameters_3gradients_and_branch;no_cross_arm_direction_or_later_prefix_assumption',
      next='report_all18_direction_conflicts_stops_M_P_T_and_four_reference_cues_before_any_Policy_or_loss_change')
    cfg['execution'].update(binding='new_roles_contract_science_scientific_binding_and_parent873closure_before_source_access',
      future_envelope='full18_synthetic_control7691_both_step_gates_and_maximum_new_P_U_S_budgets_before_execution_binding')
    cfg['audit_contract'].update(only_changed_variable='candidate_direction_only;both_arms_same_T_Armijo_M500_step_M',
      P='all18_original500_guarded_T_control_then_M_priority;historical_control900',
      U='independent_Torch_objective_3gradients_directions_exact_slopes_acceptance_stops_four_refs',
      prefix='historical_control_full_path;paired_k0_parameters_3gradients_branch_only;no_later_cross_arm_prefix',
      robust_cue='P_U_all4_reference_cues_and_no_same_parameter_stop_or_acceptance_disagreement')
    cfg['freeze_access']=dict(source_archive_or_historical_numerical_JSON_reads_or_SHA=0,
      real_direction_gradient_trial_update_environment=0,metadata_and_human_made_vectors_only=True,execution_binding_present=False)
    return cfg


def validate(path=CONFIG):
    cfg=json.loads(Path(path).read_text());p.require(cfg==create(),'M-priority scientific freeze changed');return cfg


def require_binding(config=CONFIG,binding=BINDING):
    cfg=validate(config);bound=json.loads(Path(binding).read_text())
    p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True
      and bound['science_config']==p.pin(config) and bound['scientific_freeze_binding']==p.pin(FREEZE_BINDING)
      and bound['budget']==cfg['budget'] and bound['CPU_environment']==cfg['execution']['environment']
      and bound['future_output']==OUTPUT and bound['future_artifacts']==ARTIFACTS,'new M-priority execution binding required')
    refs=bound['dependencies'];paths=[r['path'] for r in refs];freeze=p.load(p.pin(FREEZE_BINDING))
    p.require(freeze['status']=='scientific_frozen_execution_not_bound' and freeze['science_config']==p.pin(config)
      and len(paths)==len(set(paths)) and all(r in refs for r in freeze['dependencies'])
      and all(p.pin(path) in refs for path in [*FUTURE_FILES.values(),FUTURE_CONTRACT,Path(config),FREEZE_BINDING]),'execution roles/scientific closure missing')
    p.require(all(r in refs for r in p.load(PARENTS['binding'])['dependencies']) and PARENTS['binding'] in refs,'parent873 closure missing')
    for ref in refs:p.require(Path(ref['path']).suffix in {'.py','.md','.json','.toml'} and p.pin(ref['path'])==ref,'bound dependency changed')
    return bound


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
    if args.write_freeze:p.write(CONFIG,create())
    else:validate()
    print(json.dumps({'status':'M_priority_scientific_freeze_no_research_payload'}))
