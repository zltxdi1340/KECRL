"""Metadata/synthetic freeze: matched-norm joint direction at fixed lambda1."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_fine_stop_direction_review_protocol as parent

finer=parent.parent
rules=finer.rules
p,ROOT=parent.p,parent.ROOT
IDS=parent.IDS
ARMS=['negative_total_gradient','norm_matched_joint_bisector']
ANGLE_RESOLUTION=1e-12
DEFAULT_CONFIG=ROOT/'configs/libero-task7-joint-direction-diagnostic_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-joint-direction-execution-binding_v1.json'
PLAN=ROOT/'docs/libero-task7-joint-direction-plan-2026-11-10.md'
TESTS=ROOT/'tests/test_libero_task7_joint_direction_protocol.py'
OUTPUT='/tmp/libero-task7-joint-direction-20261110-v1.json'
ARTIFACTS='/tmp/libero-task7-joint-direction-artifacts-20261110-v1'
PARENTS={
 'science':{'path':str(parent.DEFAULT_CONFIG),'sha256':'15939ffb2abe778dd7ad834059252d0f669d264d12656e57ec22b656fc78dff7'},
 'binding':{'path':str(parent.DEFAULT_BINDING),'sha256':'e0479dae36f82e4a20fe478ba14a2466a4d574d19908160cabae99686b6124a0'},
 'result':{'path':parent.OUTPUT,'sha256':'e5c5b3fcfed0051c9a55dbfd1d53adad958ff15c93ce1e0c037dfaf35007431a'},
 'audit':{'path':parent.ARTIFACTS+'/independent-audit.json','sha256':'01bbf04dc0be4f6d42f6eb41d998c0604dc1dba9fe6b45f1c6c6aa5e36eda5fe'},
 'completion':{'path':parent.ARTIFACTS+'/completion.json','sha256':'d87c7017ff38e3a806251a8b6147c310afff58fb27c95c2fe8f7d0d3e25c5b18'},
 'process':{'path':'/tmp/libero-task7-fine-stop-direction-review-20261110-v1-process.json','sha256':'1745282b6d9edfa7278e6e5f461c059974829f3b74e00661b975665549841eb9'},
 'report':{'path':str(ROOT/'docs/libero-task7-fine-stop-direction-review-results-2026-11-10.md'),'sha256':'bf72366c2a305929fc717aa3b5e6ea3e02e24f3e87d075f44aedbe1a454bb62a'},
}
FUTURE_FILES={
 'primary':ROOT/'experiments/run_libero_task7_joint_direction_diagnostic.py',
 'audit':ROOT/'experiments/audit_libero_task7_joint_direction_diagnostic.py',
 'guard':ROOT/'experiments/libero_task7_joint_direction_access.py',
 'runtime_tests':ROOT/'tests/test_libero_task7_joint_direction_diagnostic.py',
 'metadata':Path(__file__).resolve(),'metadata_tests':TESTS,'plan':PLAN,
}
FUTURE_CONTRACT=ROOT/'docs/libero-task7-joint-direction-execution-contract-2026-11-10.md'
SCOPE={**copy.deepcopy(finer.SCOPE),
 'changed_variable':'only_search_direction_rule;match_current_total_gradient_norm;unchanged_trial_acceptance',
 'head_coverage':'all18_lambda1_auxiliary500;not_selected_short5_stop_points',
 'candidate_norm_matching':'at_each_own_state_to_own_total_gradient;not_across_different_states',
 'new_auxiliary_parameter_updates':'two_copies_each_500_plus_k_max50;not_original_Policy_training',
 'old_cues_G_failed_H_unrun_unchanged':True}
OPTIMIZATION={**copy.deepcopy(finer.OPTIMIZATION),
 'arms':ARMS,'direction':'control=-G_T;candidate=norm_matched_angle_bisector_of_G_T_and_Gb',
 'start':'both_from_same_own_original_saved_lambda1_auxiliary500;not_previous_fine_stop_or_final',
 'trial_caps':{arm:20 for arm in ARMS},'trial_index_range':{arm:[0,19] for arm in ARMS},
 'alpha_minima':{arm:2.**-19 for arm in ARMS},
 'acceptance':'both_first_Ttrial<Tcurrent_and_Armijo_exact_sT_and_MSEtrial<=own_auxiliary500_MSE;no_epsilon',
 'mse_monotonicity':'fixed_start_gate_unchanged;stepwise_MSE_nonincrease_is_measured_outcome_not_acceptance_gate',
 'direction_definition':dict(Gb='gradient_of_MSE3/7',G_T='gradient_of_same_T=MSE3/7+P',
  v='G_T/norm(G_T)+Gb/norm(Gb)',candidate='-norm(G_T)*(v/norm(v))',
  norm='stable_Float64_Euclidean_norm;no_learning_rate_or_loss_lambda_change',
  total_zero='zero_direction_then_zero_gradient_stop',base_zero='exact_minus_G_T;selected_MSE_slope_zero;finite_behavior_measured',
  near_opposite='norm(v)<=1e-12_returns_zero_placeholder_and_unresolved_stop;no_fallback',
  numeric_selected_check='G_T.dot(d)<0_and_Gb.dot(d)<=0;failure_is_unresolved_direction_not_optimality'),
 'candidate_direction_guard':'exact_right_sT<0_and_exact_right_sM<=0_before_trials;compute_from_current_tangent_even_at_clip_hinge_ties',
 'baseline_MSE_direction_guard':False,
 'slope':'exact_right_MSE_P_T_derivatives_along_chosen_direction;never_replace_at_clip_or_hinge_kinks',
 'stop_priority':['update_cap','zero_gradient','unresolved_joint_direction','exact_MSE_direction_increase','non_descent_selected_direction','line_search_exhausted'],
 'angle_resolution':ANGLE_RESOLUTION,
 'control':'rerun_all18_historical_fine20_paths;398updates416states3509trials36endpoints18heads',
 'prefix':'same_source500_parameters_and_readonly_fit_only;no_shared_after_start_path_or_final_ordering_required',
 'grid_change':'none;both20levels;alpha1_restart_and_first_acceptable_only',
 'grid_rationale':'unchanged_fine20_control_grid;no_new_grid_selection',
 'path_order':'seed_then_source_then_fold;negative_total_gradient_then_joint_bisector',
 'exhaustion':'all20_rejected;no_smaller_alpha_extra_updates_or_alternate_direction',
 'new_head_output':'18_NPZ_each_saved500_control_final_candidate_final;no_intermediate_checkpoint',
 'new_head_order':['saved500',ARMS[0]+'_final',ARMS[1]+'_final']}
DECISION={'trainable_ratio_max':.99,'gradient_ratio_max':.1,
 'primary_comparison':'all18_paired_final_T_MSE_P_total_gradient_updates_trials_and_cost',
 'gate_validity':'all_accepted_and_final_MSE<=own_auxiliary500_is_invariant_not_success',
 'primary_head_cue':'candidate_updates>0_and_Tcandidate<Tcontrol_and_MSEcandidate<=MSEcontrol_and_every_candidate_accepted_step_MSE<=previous',
 'path_joint_cue':'candidate_updates>0_and_Tfinal<T500_and_MSEfinal<=MSE500_and_every_accepted_step_MSE<=previous',
 'relative_reference':'report_start500_control_final_and_previous_accepted_state_separately;do_not_equate_them',
 'allowed_scientific_failure':'MSE_increase_below_start_is_valid_execution_but_fails_path_joint_cue;do_not_stop_or_relax_rule',
 'aggregate':'source_direction_fit_support_cue_requires_all9_paired_head_cues;both_sources_required_for_population_cue;no_auto_promotion',
 'cohorts':'fixed_old_zero6_short5_full7;include_all18_and_empty_source_seed_cohort_cells',
 'cohort_reporting':'all18_pairs;seed_cohort18_source_cohort6_include_empty',
 'zero6':'report_candidate_started_and_joint_cues;not_a_forced_opening_gate',
 'short5':'all5_secondary_mechanism_stratum;do_not_run_from_their_old_stops',
 'full7':'candidate_paths_and_endpoints_may_change;retain_any_regression;no_old_grid_prefix_invariant',
 'structural_invariants':'same_start_and_roles;strict_T_descent_and_fixed_MSE_envelope_within_each_path;no_between_arm_update_or_T_order_constraint',
 'baseline_parity':'prior_fine20_scalar_projection_abs1e-10;ID_count_branch_and_acceptance_exact',
 'original_optimization_cue':'unchanged_Tfinal/T500<=0.99_and_normGfinal/normG500<=0.1;report_separately_from_direction_fit_cue',
 'no_promotion':'diagnostic_only;old_cues_G_H_and_sealed_roles_unchanged;no_query_or_closed_loop_automatic_access',
 'statistics':'descriptive_pairs_and_all3_fold_values_equal_fold;folds_not_independent_seeds;no_pvalues',
 'next':'inspect_all18_tradeoffs_and_stops_before_any_new_training_direction_loss_or_formal_stage'}
EXECUTION={**copy.deepcopy(finer.EXECUTION),
 'binding':'seven_roles_contract_science_and_parent774_dependency_closure_before_archive_SHA_or_decode',
 'technical_stop':'lineage_source_historical_control_direction_norm_role_budget_nonfinite_branch_acceptance_or_independent_audit_mismatch;retain_partial_no_retry',
 'future_envelope':'1836states36000trials_and_direction_construction_ledger_both_engines_before_research_payload'}


def finite(*values):
 p.require(all(type(x) in (int,float) and math.isfinite(x) for x in values),'finite real scalars required')


def direction(arm,total,base):
 """Scalar reference, exercised only on synthetic vectors during this freeze."""
 p.require(arm in ARMS and all(type(v) is list and len(v)==257 for v in (total,base)),'two257 vectors and registered arm required')
 finite(*total,*base);g=math.hypot(*total);b=math.hypot(*base);finite(g,b)
 q=None
 if g==0:chosen=[0.]*257;status='zero_total_gradient'
 elif arm==ARMS[0]:chosen=[-x for x in total];status='negative_total_gradient'
 elif b==0:chosen=[-x for x in total];status='base_zero_use_total'
 else:
  v=[x/g+y/b for x,y in zip(total,base)];q=math.hypot(*v);finite(q)
  if q<=ANGLE_RESOLUTION:chosen=[0.]*257;status='unresolved_joint_direction'
  else:chosen=[-g*(x/q) for x in v];status='joint_bisector'
 finite(*chosen)
 norm=math.hypot(*chosen);dot_t=math.fsum(x*y for x,y in zip(total,chosen));dot_b=math.fsum(x*y for x,y in zip(base,chosen));finite(norm,dot_t,dot_b)
 joint=dot_t<0 and dot_b<=0
 if arm==ARMS[1] and status in ('base_zero_use_total','joint_bisector') and not joint:status='unresolved_joint_direction'
 slope_m=7*dot_b;finite(slope_m)
 return dict(direction=chosen,status=status,total_norm=g,base_norm=b,unit_sum_norm=q,direction_norm=norm,
  selected_T_slope=dot_t,selected_MSE_slope=slope_m,selected_joint=joint)


def trial_alpha(arm,index):
 p.require(arm in ARMS and type(index) is int and 0<=index<20,'trial outside fixed20 grid')
 return 2.**(-index)


def trial_decision(arm,current,trial,slope,index,trial_mse,start_mse):
 trial_alpha(arm,index)
 return finer.trial_decision('fine20',current,trial,slope,index,trial_mse,start_mse)


def stopping_reason(arm,updates,gradient_norm,status,selected_joint,slope_T,slope_M,rejected_trials=0):
 p.require(arm in ARMS and type(updates) is int and 0<=updates<=50 and type(rejected_trials) is int and 0<=rejected_trials<=20
  and type(selected_joint) is bool and status in ('zero_total_gradient','negative_total_gradient','base_zero_use_total','joint_bisector','unresolved_joint_direction'), 'invalid direction stop fields')
 finite(gradient_norm,slope_T,slope_M);p.require(gradient_norm>=0,'negative gradient norm')
 if arm==ARMS[0]:p.require(status in ('negative_total_gradient','zero_total_gradient'),'control direction changed')
 if updates==50:return 'update_cap'
 if gradient_norm==0:
  p.require(slope_T==slope_M==0,'nonzero slope of zero total direction');return 'zero_gradient'
 if arm==ARMS[1] and (status=='unresolved_joint_direction' or not selected_joint):return 'unresolved_joint_direction'
 if arm==ARMS[1] and slope_M>0:return 'exact_MSE_direction_increase'
 if slope_T>=0:return 'non_descent_selected_direction'
 if rejected_trials==20:return 'line_search_exhausted'
 return None


def paired_evidence(t0,m0,t_control,m_control,t_candidate,m_candidate,k_control,k_candidate,step_mse_deltas):
 p.require(type(step_mse_deltas) is list,'candidate increments must be a list')
 finite(t0,m0,t_control,m_control,t_candidate,m_candidate,*step_mse_deltas)
 p.require(min(t0,m0,t_control,m_control,t_candidate,m_candidate)>=0 and t_control<=t0 and t_candidate<=t0
  and m_control<=m0 and m_candidate<=m0 and type(k_control) is type(k_candidate) is int
  and 0<=k_control<=50 and 0<=k_candidate<=50 and type(step_mse_deltas) is list and len(step_mse_deltas)==k_candidate,
  'invalid paired fixed500 endpoints or candidate path')
 for k,t,m in [(k_control,t_control,m_control),(k_candidate,t_candidate,m_candidate)]:
  p.require((k==0 and t==t0 and m==m0) or (k>0 and t<t0),'unique accepted endpoint/count differs')
 p.require(abs(math.fsum(step_mse_deltas)-(m_candidate-m0))<=1e-10,'candidate MSE increments do not conserve')
 increases=sum(dm>0 for dm in step_mse_deltas);progress=k_candidate>0 and t_candidate<t0
 path=progress and increases==0
 paired=path and t_candidate<t_control and m_candidate<=m_control
 return dict(candidate_progress_from500=progress,candidate_stepwise_MSE_increases=increases,
  candidate_path_joint_cue=path,paired_joint_cue=paired,candidate_minus_start_MSE=m_candidate-m0,
  candidate_minus_control_MSE=m_candidate-m_control,candidate_minus_control_T=t_candidate-t_control,
  automatic_Policy_promotion=False)


def source_evidence(rows):
 expected={(7,s,a,f,finer.source.MODE) for s in p.SEEDS for a in p.ARMS for f in range(3)}
 p.require(len(rows)==18 and {tuple(r[k] for k in IDS) for r in rows}==expected
  and all(type(r['paired_joint_cue']) is bool for r in rows),'all18 paired head cues required')
 flags={a:all(r['paired_joint_cue'] for r in rows if r['arm']==a) for a in p.ARMS}
 return dict(source_all9_cues=flags,population_direction_fit_cue=all(flags.values()),automatic_Policy_promotion=False)


def budget():
 n=50256;states=36*51;trials=36*50*20;state_rows=n*2*51;trial_rows=n*2*50*20
 tables=dict(endpoints=54,parameters=54*257,states_max=states,trials_max=trials,heads=36,pairs=18,**{'scaler-parity':18})
 b=copy.deepcopy(finer.budget())
 b.pop('historical_coarse_JSON_projected_records_per_stage');b.pop('pair_prefix_checks')
 b.update(trial_caps_per_round={arm:20 for arm in ARMS},trial_objective_forms_max_by_arm={arm:18000 for arm in ARMS},
  trial_objective_forms_max_per_stage=trials,trial_MSE_and_penalty_pair_evaluations_max_per_stage=trials,
  trial_component3_fit_row_visits_max_per_stage=trial_rows,all_linear_forms_max_per_stage=2*states+trials,
  all_linear_form_row_visits_max_per_stage=2*state_rows+trial_rows,
  all_linear_form_feature_multiplications_max_per_stage=(2*state_rows+trial_rows)*256,
  direction_constructions_max_per_stage=states,direction_norm_reductions_max_per_stage=4*states,
  direction_dot_reductions_max_per_stage=2*states,direction_vector_forms_max_per_stage=4*states,
  direction_vector_coordinate_writes_max_per_stage=4*states*257,
  historical_control_JSON_projected_records_per_stage=dict(states=416,trials=3509,endpoints=36,heads=18),
  historical_control_updates=398,same_start_checks=18,after_start_pair_prefix_required=False,
  tables=tables,output_CSV_rows_max=sum(tables.values()))
 return b


def create():
 lineage={k:p.load_pin(v) for k,v in PARENTS.items() if k!='report'}
 p.require(p.pin(Path(PARENTS['report']['path']))==PARENTS['report'],'completed review report changed')
 parent_cfg=parent.validate();parent.require_binding()
 r,a,done,process=(lineage[k] for k in ('result','audit','completion','process'))
 p.require(lineage['science']==parent_cfg and r['science_config']==done['science_config']==process['science_config']==PARENTS['science']
  and r['binding']==done['binding']==process['binding']==PARENTS['binding'] and a['result']==done['result']==PARENTS['result']
  and done['audit']==PARENTS['audit'] and a['status']=='passed_independent_fine_stop_direction_review_audit'
  and done['status']=='complete_independently_audited_fine_stop_direction_review' and process['status']=='complete_independently_audited'
  and len(process['processes'])==2 and all(x['exit_code']==0 for x in process['processes'])
  and len(lineage['binding']['dependencies'])==774,'completed audited review lineage required')
 training=finer.validate();prior=p.load_pin(parent.PARENTS['result'])
 p.require(prior['science_config']==parent.PARENTS['science'] and prior['inputs']==training['inputs'],'original auxiliary500 inputs changed')
 control=[h for h in prior['heads'] if h['optimization_arm']=='fine20']
 p.require(len(control)==18 and sum(h['extra_updates'] for h in control)==398,'all18 prior fine20 heads required')
 cohorts=[]
 for h in control:
  ident={k:h[k] for k in IDS};old=next(c for c in training['cohorts'] if all(c[k]==v for k,v in ident.items()))
  cohorts.append(dict(ident,cohort=old['cohort'],prior_control_updates=h['extra_updates'],prior_control_stop=h['terminal_reason']))
 p.require([sum(h['cohort']==c for h in cohorts) for c in ('zero','short','full50')]==[6,5,7],'fixed6/5/7 strata changed')
 projections={name:[f for f in prior['tables'][name]['columns'] if f!='parameter_digest'] for name in ('states','trials','endpoints','heads')}
 p.require(all(not set(fields).intersection({'parameter','parameters','weights','bias',*parent.VECTORS}) for fields in projections.values()),'historical projection must be scalar')
 selected={name:[row for row in prior[name] if (row['condition'] in ('saved500','fine20_final') if name=='endpoints' else row['optimization_arm']=='fine20')]
  for name in projections}
 p.require({k:len(v) for k,v in selected.items()}==budget()['historical_control_JSON_projected_records_per_stage'],'control scalar projection coverage differs')
 return copy.deepcopy(dict(kind='task7_same_loss_fixed_MSE_bound_joint_direction_diagnostic',version=1,formal_result=False,
  status='scientific_freeze_only_execution_binding_required',parents=PARENTS,scope=SCOPE,optimization=OPTIMIZATION,decision=DECISION,
  execution=EXECUTION,budget=budget(),inputs=training['inputs'],reference_points=training['reference_points'],cohorts=cohorts,
  source_members=finer.SOURCE_MEMBERS,head_members=finer.HEAD_MEMBERS,new_head_members=finer.NEW_HEAD_MEMBERS,
  new_head_layout=dict(weights=[3,256],bias=[3],additional_updates=[3],parameter_dtype='float64',additional_updates_dtype='int32',order=OPTIMIZATION['new_head_order']),
  historical_control=dict(source_JSON=parent.PARENTS['result'],source_science=parent.PARENTS['science'],optimization_arm='fine20',
   endpoint_conditions=['saved500','fine20_final'],projections=projections,no_old_CSV_or_optimized_NPZ_read=True),
  code=dict(metadata=p.pin(Path(__file__).resolve()),tests=p.pin(TESTS)),plan=p.pin(PLAN),future_files={k:str(v) for k,v in FUTURE_FILES.items()},
  future_contract=str(FUTURE_CONTRACT),output=OUTPUT,artifacts=ARTIFACTS))


def validate(path=DEFAULT_CONFIG):
 cfg=json.loads(Path(path).read_text())
 p.require(json.dumps(cfg,sort_keys=True,allow_nan=False)==json.dumps(create(),sort_keys=True,allow_nan=False),'joint-direction scientific freeze differs')
 return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
 cfg=validate(config);bound=json.loads(Path(binding).read_text())
 p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True and bound['science_config']==p.pin(config)
  and bound['budget']==cfg['budget'] and bound['CPU_environment']==EXECUTION['environment']
  and bound['future_output']==OUTPUT and bound['future_artifacts']==ARTIFACTS,'joint-direction execution binding required')
 refs=bound['dependencies'];paths=[p.resolve(ref['path']).resolve() for ref in refs]
 p.require(len(paths)==len(set(paths)) and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT,DEFAULT_CONFIG}).issubset(paths)
  and all(ref in refs for ref in p.load_pin(PARENTS['binding'])['dependencies']),'implementation or parent774 closure missing')
 for ref in refs:
  path=p.resolve(ref['path']);p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path)==ref,'bound code changed or research archive in closure')
 return bound


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');parser.add_argument('--require-binding',action='store_true');args=parser.parse_args()
 if args.write_freeze:
  with DEFAULT_CONFIG.open('x') as f:json.dump(create(),f,ensure_ascii=False,indent=2);f.write('\n')
 else:
  validate()
  if args.require_binding:require_binding()
 print(json.dumps({'status':'scientific_frozen_no_new_research_directions_readouts_or_updates'}))
