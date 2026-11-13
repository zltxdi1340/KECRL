"""Metadata/synthetic freeze for one failed head's numerical prefix review."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
from experiments import libero_task7_joint_direction_protocol as parent

p,ROOT=parent.p,parent.ROOT
IDS=parent.IDS
FOCUS=dict(task=7,seed=20261107,arm='B_fomaml',fold=2,mode=parent.finer.source.MODE)
OPTIMIZATION_ARM='norm_matched_joint_bisector'
LAST_STEP=35
N=2777
D=257
TOL=1e-10
ROLES=['P','U','S','N']
Q_FIELDS=['available','total','normalization','same_parameter_gradient','path','residual',
 'total_abs','normalization_abs','same_parameter_gradient_abs','path_abs',
 'total_over_tolerance','normalization_over_tolerance','same_parameter_gradient_over_tolerance','path_over_tolerance']
DEFAULT_CONFIG=ROOT/'configs/libero-task7-joint-direction-prefix-review-diagnostic_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-joint-direction-prefix-review-execution-binding_v1.json'
PLAN=ROOT/'docs/libero-task7-joint-direction-prefix-review-plan-2026-11-10.md'
TESTS=ROOT/'tests/test_libero_task7_joint_direction_prefix_review_protocol.py'
OUTPUT='/tmp/libero-task7-joint-direction-prefix-review-20261110-v1.json'
ARTIFACTS='/tmp/libero-task7-joint-direction-prefix-review-artifacts-20261110-v1'
PARENTS={
 'science':{'path':str(parent.DEFAULT_CONFIG),'sha256':'3a586c3270b9834889848e3dd175aa54ff8fd4d693f83ba7005fa55f7ec09251'},
 'binding':{'path':str(parent.DEFAULT_BINDING),'sha256':'4061a0134a5f6000459977b987ffcbd23baaeb308b7fac2b5f06e1c1d12d8400'},
 'pending_result':{'path':parent.OUTPUT,'sha256':'a920d27a4f0fb61840c9adc33c96262d97e28e7d47ef75c3ea55f8d004c0ddbe'},
 'audit_failed_partial':{'path':parent.ARTIFACTS+'/audit-failed-partial.json','sha256':'bcefd39f8a4b5af6b6a47a81976ab88019c441ecb8e2ae0d6e64d88a70b5eba9'},
 'process':{'path':'/tmp/libero-task7-joint-direction-20261110-v1-process.json','sha256':'ceb94cbb2b521de5eeeac812f70aedc136ce1ef541c5291d88c2c13f6d121bb6'},
 'report':{'path':str(ROOT/'docs/libero-task7-joint-direction-results-2026-11-10.md'),'sha256':'dc70fc7bfaf64e9303ec2bab4fdd53a5e939528f980f079ab8116e14a4e633a2'},
}
FUTURE_FILES={
 'primary':ROOT/'experiments/run_libero_task7_joint_direction_prefix_numerical_review.py',
 'cross':ROOT/'experiments/run_libero_task7_joint_direction_prefix_cross_review.py',
 'audit':ROOT/'experiments/audit_libero_task7_joint_direction_prefix_numerical_review.py',
 'guard':ROOT/'experiments/libero_task7_joint_direction_prefix_review_access.py',
 'runtime_tests':ROOT/'tests/test_libero_task7_joint_direction_prefix_numerical_review.py',
 'metadata':Path(__file__).resolve(),'metadata_tests':TESTS,'plan':PLAN,
}
FUTURE_CONTRACT=ROOT/'docs/libero-task7-joint-direction-prefix-review-execution-contract-2026-11-10.md'
TRACE_MEMBERS=['steps','fit_rows','fit_demo_indices','scaler_mean','scaler_scale','parameters',
 'base_gradient','penalty_gradient','total_gradient','direction']
RAW_COORDINATES=['theta_P','theta_U','Gb_P','Gp_P','GT_P','d_P','Gb_S','Gp_S','GT_S','d_S',
 'Gb_U','Gp_U','GT_U','d_U','d_N']
COORDINATE_COLUMNS=IDS+['step','coordinate','U_available']+RAW_COORDINATES
EVENT_COLUMNS=['stage','role','step','event','status','parameter_digest','Gb_digest','GT_digest',
 'q_value','q_hex','counters','last_RSS_bytes','elapsed_seconds','error']
SCOPE=dict(focus=FOCUS,selection='one_head_selected_by_technical_failure_not_favorable_fit',
 original_Policy_updates=500,future_stabilization_steps=5,same_representation_loss_M500_direction_and_grid=True,
 source_role='source_support0..29_own20fit_tge1;encoder_has_seen_all30',
 probe_source_query_validation_heldout_qualification_reserved_final_access=False,
 original_Policy_CNN_environment_calls=0,SPT_Knowledge_Module_changes=False,
 parent_failed_run_remains_failed=True,replay_not_checkpoint_selection_or_new_full_direction_experiment=True)
DESIGN=dict(
 prefix='k0..35_inclusive_36_states;only35_accepted_updates;never_evaluate_k35_trial_or_update36',
 P='NumPy_reconstruct_original_candidate500_prefix;exact_parameter_digests_and_original_scalar_branch_trial_parity',
 U='Torch_own_free_candidate_path_from_original500;up_to35_updates;no_P_gradients_or_P_updates',
 S='Torch_gradient_and_direction_readout_at_exact_P_parameters_k0..35;no_optimizer_or_trials',
 N='Torch_direction_normalization_only_at_exact_P_Gb_and_GT;no_parameter_feature_linear_or_gradient_job',
 cross_order='finish_U_own_path_first_then_S_and_N;do_not_interleave_cross_jobs_into_U_trajectory',
 observation_chain='qP->qN->qS->qU;normalization_on_same_gradients_then_gradient_on_same_parameters_then_path_difference',
 gradient_chain='GU-GP=(GS-GP)+(GU-GS);Gb_Gp_GT_each257coords',
 direction_chain='dU-dP=(dN-dP)+(dS-dN)+(dU-dS);257coords',
 decomposition='signed_scalar_and_coordinate_telescoping;L2_Linf_not_additive;no_causal_percentages',
 point_values='save_actual_q_values_float_hex_statuses_digests_all_raw_parameter_and_gradient_coordinates_before_comparison',
 old_failure_numeric_tolerance=TOL,old_angle_resolution=parent.ANGLE_RESOLUTION,
 mismatch='numeric_threshold_or_cross_branch_difference_is_recorded_observation_not_technical_abort_in_this_diagnostic',
 primary_anchor='P_must_reproduce_historical36_parameter_digests258_trials_discrete_flags_exact;scalar_abs1e-10;failure_stops_no_retry',
 early_U_stop='keep_own_mathematical_stop_or_trial_divergence;U_unavailable_after_stop_masked_null;never_force35_or_zero_fill',
 gradient_zero_and_clip_hinge='retain_original_selected_and_exact_derivative_rules;no_fallback_new_direction_or_epsilon',
 old_missing_value='new_U_actual_value_is_new_isolated_replay_not_recovery_of_unlogged_old_audit_value',
 final_arithmetic_audit='stdlib_only_new_CSV_JSON_JSONL;no_original_archives_no_model_or_gradient',
 q_reference='recomputed_q_from_saved_vectors_is_a_logged_observation_not_a_gate_that_discards_target_mismatches',
 final_status='new_numerical_observations_checked_only;never_write_old_completion_or_accept_parent14of18',
 statistics='descriptive_one_failure_head_no_seed_generalization_pvalues_or_closed_loop_claims')
EXECUTION=dict(runtime=copy.deepcopy(parent.EXECUTION['runtime']),environment=copy.deepcopy(parent.EXECUTION['environment']),
 stages={'primary':dict(seconds=180,RSS_bytes=536870912),'cross':dict(seconds=300,RSS_bytes=805306368),
         'audit':dict(seconds=120,RSS_bytes=268435456)},overall_seconds=720,
 new_CSV_bytes_max=67108864,new_trace_NPZ_bytes_max=16777216,events_bytes_max=8388608,error_JSON_bytes_max=1048576,
 technical_retries=0,overwrite=False,binding='eight_roles_contract_science_parent785_closure_before_new_archive_access',
 numerical_comparison_abs=TOL,numerical_comparison_rel=0,
 technical_stop='source_lineage_role_shape_nonfinite_budget_primary_anchor_or_derived_arithmetic_corruption;not_observed_cross_numeric_mismatch',
 failure_log='only_already_available_values;capture_before_assert;no_new_read_forward_or_gradient_in_handler;save_resource_direction_access_ledgers',
 loss_direction_acceptance='original_v1_unchanged;prefix_cap_is_diagnostic_scope_not_optimizer_convergence')


def finite(*values):
 p.require(all(type(x) in (int,float) and math.isfinite(x) for x in values),'finite scalar required')


def q_chain(qp,qn,qs,qu):
 values=(qp,qn,qs,qu)
 for x in values:
  if x is not None:finite(x);p.require(x>=0,'negative unit-sum norm')
 if None in values:
  return dict(available=False,total=None,normalization=None,same_parameter_gradient=None,path=None,
   residual=None,total_abs=None,normalization_abs=None,same_parameter_gradient_abs=None,path_abs=None,
   total_over_tolerance=None,normalization_over_tolerance=None,
   same_parameter_gradient_over_tolerance=None,path_over_tolerance=None)
 norm=qn-qp;grad=qs-qn;path=qu-qs;total=qu-qp
 residual=math.fsum([norm,grad,path,-total]);finite(norm,grad,path,total,residual)
 return dict(available=True,total=total,normalization=norm,same_parameter_gradient=grad,path=path,residual=residual,
  total_abs=abs(total),normalization_abs=abs(norm),same_parameter_gradient_abs=abs(grad),path_abs=abs(path),
  total_over_tolerance=abs(total)>TOL,normalization_over_tolerance=abs(norm)>TOL,
  same_parameter_gradient_over_tolerance=abs(grad)>TOL,path_over_tolerance=abs(path)>TOL)


def vector_chain(*vectors):
 p.require(len(vectors) in (2,3,4) and all(type(v) is list and len(v)==D for v in vectors),'two_to_four257_vectors_required')
 for v in vectors:finite(*v)
 edges=[[b-a for a,b in zip(left,right)] for left,right in zip(vectors,vectors[1:])]
 total=[b-a for a,b in zip(vectors[0],vectors[-1])]
 residual=[math.fsum([*(edge[j] for edge in edges),-total[j]]) for j in range(D)]
 def metrics(v):
  norm=math.hypot(*v);linf=max(abs(x) for x in v);finite(norm,linf)
  return dict(l2=norm,linf=linf,over_tolerance=linf>TOL)
 return dict(edges=[metrics(v) for v in edges],total=metrics(total),residual_linf=max(abs(x) for x in residual),
  additive_norm_claim=False)


def parameter_digest(vector):
 p.require(type(vector) is list and len(vector)==D,'257 parameters required');finite(*vector)
 return hashlib.sha256(b''.join(struct.pack('<d',x) for x in vector)).hexdigest()


def prefix_action(step,mathematical_stop=None):
 p.require(type(step) is int and 0<=step<=LAST_STEP,'outside diagnostic prefix')
 p.require(mathematical_stop is None or mathematical_stop in parent.OPTIMIZATION['stop_priority'][1:],
  'unlisted or unreachable mathematical stop inside diagnostic prefix')
 return dict(diagnostic_cap=step==LAST_STEP,mathematical_stop=mathematical_stop,
  may_try_action=step<LAST_STEP and mathematical_stop is None,
  stop_reason='diagnostic_prefix_cap' if step==LAST_STEP else mathematical_stop)


def validate_round(step,trials,start_mse,current_t,slope):
 p.require(prefix_action(step)['may_try_action'] and type(trials) is list and 1<=len(trials)<=20,'trial outside prefix or round')
 finite(start_mse,current_t,slope)
 accepted=False
 for j,row in enumerate(trials):
  p.require(not accepted and type(row['trial_index']) is int and row['trial_index']==j and row['alpha']==2.**(-j),'skipped/repeated grid point')
  expected=parent.trial_decision(OPTIMIZATION_ARM,current_t,row['trainable_objective'],slope,j,row['component3_mse'],start_mse)
  p.require(all(type(row[k]) is type(v) and row[k]==v for k,v in expected.items()),'original acceptance changed')
  accepted=expected['accepted']
 p.require(accepted or len(trials)==20,'incomplete exhausted round')
 return dict(accepted=accepted,trial_count=len(trials))


def validate_coverage(P_steps,U_steps,S_steps,N_steps):
 expected=list(range(36))
 p.require(P_steps==S_steps==N_steps==expected and type(U_steps) is list and 1<=len(U_steps)<=36
  and U_steps==list(range(len(U_steps))) and all(type(k) is int for seq in (P_steps,U_steps,S_steps,N_steps) for k in seq),
  'noncanonical step coverage or forced/missing U path')
 return dict(U_complete=len(U_steps)==36,U_last_step=U_steps[-1],missing_U_steps=list(range(len(U_steps),36)),
  missing_U_values='masked_null_not_zero')


def role_inputs(role,parameter_origin=None,gradient_origin=None,updates=0,trial_index=None):
 p.require(role in ROLES and type(updates) is int and 0<=updates<=35,'unlisted role/update count')
 if role in ('P','U'):
  p.require(parameter_origin=='own_original500_or_own_accepted_path' and gradient_origin in (None,'own_measured_gradient'),'cross-fed optimizer')
 else:
  p.require(updates==0 and trial_index is None,'cross readout must not update or trial')
  p.require((role=='S' and parameter_origin=='exact_P_parameters' and gradient_origin in (None,'own_measured_gradient'))
   or (role=='N' and parameter_origin is None and gradient_origin=='exact_P_saved_Gb_GT'),'wrong cross-input role')
 if trial_index is not None:p.require(type(trial_index) is int and 0<=trial_index<20 and updates<35,'unbudgeted trial')
 return True


def budget():
 def compute(states,trials,updates,directions):
  return dict(gradient_states_max=states,gradient_vectors_max=3*states,gradient_coordinates_max=3*states*D,
   updates_max=updates,parameter_coordinate_writes_max=updates*D,trials_max=trials,
   state_fit_rows_max=states*N,state_seven_action_coordinates_max=states*N*7,trial_fit_rows_max=trials*N,
   linear_forms_max=2*states+trials,linear_form_rows_max=(2*states+trials)*N,
   linear_feature_multiplications_max=(2*states+trials)*N*256,
   direction_constructions_max=directions,direction_norm_reductions_max=4*directions,
   direction_dot_reductions_max=2*directions,direction_vector_forms_max=4*directions,
   direction_coordinate_writes_max=4*directions*D)
 primary=compute(36,258,35,36);cross=compute(72,700,35,108)
 return dict(original_heads=1,fit_rows=2777,feature_dimension=256,parameter_dimension=D,
  original_NPZ_loads_per_numeric_stage=3,original_members_per_numeric_stage=15,original_archive_SHA_per_numeric_stage=6,
  new_trace_NPZ_writes=1,new_trace_member_writes=10,new_trace_NPZ_loads_cross=1,new_trace_member_decodes_cross=10,
  new_trace_archive_SHA_primary=1,new_trace_archive_SHA_cross=2,total_NPZ_loads=7,total_member_decodes=40,total_archive_SHA=15,
  primary=primary,cross=cross,primary_autograd=0,cross_autograd_max=216,
  total_replay_updates_max=70,total_gradient_states_max=108,total_gradient_vectors_max=324,
  total_trials_max=958,total_linear_forms_max=1174,total_linear_form_rows_max=3260198,
  total_linear_feature_multiplications_max=834610688,
  scaler_reconstructions_per_numeric_stage=1,total_scaler_reconstructions=2,standardization_rows_per_stage=N,
  total_standardization_feature_coordinates=2*N*256,
  historical_P_scalar_states=36,historical_P_parameter_digests=36,historical_P_trial_rows=258,
  same_parameter_Torch_jobs=36,same_gradient_Torch_normalization_jobs=36,
  CSV_rows=dict(primary_states=36,primary_trials=258,comparisons=36,coordinates=36*D,torch_trials_max=700),
  all_CSV_rows_max=10282,CSV_files=5,new_derived_CSV_SHA_total=15,arithmetic_audit_CSV_passes=5,
  primary_events_max=74,cross_events_max=218,total_event_records_max=292,event_JSONL_files=2,total_event_JSONL_SHA=6,
  arithmetic_audit_event_passes=2,arithmetic_audit_original_archive_or_gradient_calls=0,
  arithmetic_error_vector_norms_max=14*36,arithmetic_error_vector_maxabs_max=14*36,
  arithmetic_error_vector_coordinate_writes_max=14*36*D,arithmetic_coordinate_telescoping_checks_max=4*36*D,
  arithmetic_q_reference_jobs_max=3*36,arithmetic_q_reference_norms_max=9*36,
  arithmetic_q_reference_vector_forms_max=9*36,
  new_original_Policy_CNN_environment_or_sealed_calls=0)


def create():
 lineage={k:p.load_pin(ref) for k,ref in PARENTS.items() if k!='report'}
 p.require(p.pin(Path(PARENTS['report']['path']))==PARENTS['report'],'failed direction report changed')
 old=parent.validate();parent.require_binding()
 r,a,process=(lineage[k] for k in ('pending_result','audit_failed_partial','process'))
 p.require(lineage['science']==old and r['science_config']==process['science_config']==PARENTS['science']
  and r['binding']==process['binding']==PARENTS['binding'] and r['status']=='complete_joint_direction_pending_audit'
  and a['status']==process['status']=='failed_no_retry' and [v['exit_code'] for v in process['processes']]==[0,1]
  and 'direction_unit_sum_norm' in a['error'] and a['scalar_checks']==319010
  and a['computation']['gradient_states']==881 and len(lineage['binding']['dependencies'])==785
  and not (Path(parent.ARTIFACTS)/'completion.json').exists() and not (Path(parent.ARTIFACTS)/'independent-audit.json').exists(),
  'unqualified failed parent lineage required')
 def match(row):return all(row.get(k)==v for k,v in FOCUS.items()) and row.get('optimization_arm')==OPTIMIZATION_ARM
 states=[{f:row[f] for f in r['tables']['states']['columns']} for row in r['states'] if match(row) and 0<=row['extra_updates']<=35]
 trials=[{f:row[f] for f in r['tables']['trials']['columns']} for row in r['trials'] if match(row) and 0<=row['from_extra_updates']<35]
 p.require([row['extra_updates'] for row in states]==list(range(36)) and len(trials)==258
  and sum(row['accepted'] for row in trials)==35 and r['states'][880]==states[-1], 'exact failed-head prefix coverage differs')
 for k in range(35):
  part=[row for row in trials if row['from_extra_updates']==k]
  p.require(part and [row['trial_index'] for row in part]==list(range(len(part))) and len(part)<=20
   and part[-1]['accepted'] is True and all(row['accepted'] is False for row in part[:-1]),'historical prefix round changed')
 head=next(h for h in old['inputs']['heads'] if all(h[k]==v for k,v in FOCUS.items()))
 model=next(m for m in old['inputs']['models'] if (m['seed'],m['arm'])==(FOCUS['seed'],FOCUS['arm']))
 old_fold=old['inputs']['folds'][2]
 fold={k:old_fold[k] for k in ('fold','fit_demos','fit_rows','fit_tge1_rows')}
 p.require(len(fold['fit_rows'])==N and type(fold['fit_tge1_rows']) is int and fold['fit_tge1_rows']==N
  and len(fold['fit_demos'])==20 and head['training']['updates']==500,'fixed own-fit role changed')
 point_columns=[f for f in r['tables']['states']['columns'] if f not in IDS+['optimization_arm','extra_updates']]
 comparison_columns=IDS+['step','U_available','U_stop']+[role+'.'+f for role in ('P','S','U') for f in point_columns]
 comparison_columns += ['N.'+f for f in ['status','total_norm','base_norm','unit_sum_norm','direction_norm','selected_T_slope','selected_MSE_slope','selected_joint','input_Gb_digest','input_GT_digest']]
 comparison_columns += [role+'_q_hex' for role in ROLES]+['q_'+f for f in Q_FIELDS]
 errors=['theta_U_P']+[g+'_'+e for g in ('Gb','Gp','GT') for e in ('S_P','U_S','U_P')]+['d_N_P','d_S_N','d_U_S','d_U_P']
 comparison_columns += [name+'_'+metric for name in errors for metric in ('l2','linf','relative_l2')]
 comparison_columns += ['branch_P_S_equal','branch_S_U_equal','branch_P_U_equal']
 return copy.deepcopy(dict(kind='task7_failed_head35_numerical_prefix_forensics',version=1,formal_result=False,
  status='scientific_freeze_only_execution_binding_required',parents=PARENTS,scope=SCOPE,design=DESIGN,execution=EXECUTION,budget=budget(),
  inputs=dict(shared=old['inputs']['shared'],current=model['current_archive'],head=head['archive']),head=head,fold=fold,
  source_members=copy.deepcopy(parent.finer.SOURCE_MEMBERS),head_members=copy.deepcopy(parent.finer.HEAD_MEMBERS),
  trace_members=TRACE_MEMBERS,trace_layout=dict(steps=[36],fit_rows=[N],fit_demo_indices=[20],scaler_mean=[256],scaler_scale=[256],
   **{f:[36,D] for f in ('parameters','base_gradient','penalty_gradient','total_gradient','direction')}),
  trace_dtypes={'steps':'int32','fit_rows':'int32','fit_demo_indices':'int32','other_members':'float64'},
  historical_prefix=dict(source_JSON=PARENTS['pending_result'],states=states,trials=trials,
   state_columns=r['tables']['states']['columns'],trial_columns=r['tables']['trials']['columns'],
   metadata_JSON_physically_parsed_vectors_not_used_for_new_direction_or_gradient=True),
  output_columns=dict(primary_states=r['tables']['states']['columns'],primary_trials=r['tables']['trials']['columns'],
   torch_trials=r['tables']['trials']['columns'],coordinates=COORDINATE_COLUMNS,comparisons=comparison_columns,events=EVENT_COLUMNS),
  code=dict(metadata=p.pin(Path(__file__).resolve()),tests=p.pin(TESTS)),plan=p.pin(PLAN),
  future_files={k:str(v) for k,v in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT),output=OUTPUT,artifacts=ARTIFACTS))


def validate(path=DEFAULT_CONFIG):
 cfg=json.loads(Path(path).read_text())
 p.require(json.dumps(cfg,sort_keys=True,allow_nan=False)==json.dumps(create(),sort_keys=True,allow_nan=False),'prefix-review scientific freeze differs')
 return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
 cfg=validate(config);b=json.loads(Path(binding).read_text())
 p.require(b['status']=='bound_before_payload' and b['before_research_payload'] is True and b['science_config']==p.pin(config)
  and b['budget']==cfg['budget'] and b['CPU_environment']==EXECUTION['environment'] and b['future_output']==OUTPUT
  and b['future_artifacts']==ARTIFACTS,'new prefix-review binding required')
 refs=b['dependencies'];paths=[p.resolve(r['path']).resolve() for r in refs]
 p.require(len(paths)==len(set(paths)) and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT,DEFAULT_CONFIG}).issubset(paths)
  and all(r in refs for r in p.load_pin(PARENTS['binding'])['dependencies']),'eight roles or parent785 closure missing')
 for ref in refs:
  path=p.resolve(ref['path']);p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path)==ref,'bound code changed or archive included')
 return b


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');parser.add_argument('--require-binding',action='store_true');args=parser.parse_args()
 if args.write_freeze:
  with DEFAULT_CONFIG.open('x') as f:json.dump(create(),f,ensure_ascii=False,indent=2);f.write('\n')
 else:
  validate()
  if args.require_binding:require_binding()
 print(json.dumps({'status':'scientific_frozen_no_new_prefix_replay_or_research_numeric_readout'}))
