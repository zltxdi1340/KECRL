"""Read-only saved-fit decomposition: metadata and fixed projection contract."""
from __future__ import annotations
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_fixed_representation_optimization_protocol as parent

p,ROOT=parent.p,parent.ROOT
DEFAULT_CONFIG=ROOT/'configs/libero-task7-optimization-path-decomposition-readonly_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-optimization-path-decomposition-execution-binding_v1.json'
PLAN=ROOT/'docs/libero-task7-optimization-path-decomposition-plan-2026-11-09.md'
TESTS=ROOT/'tests/test_libero_task7_optimization_path_decomposition.py'
OUTPUT='/tmp/libero-task7-optimization-path-decomposition-20261109-v1.json'
ARTIFACTS='/tmp/libero-task7-optimization-path-decomposition-artifacts-20261109-v1'
PARENTS={
 'science':{'path':str(parent.DEFAULT_CONFIG),'sha256':'95c476741cabf0ef79b4a34848e634c8d72eeee4d80fec5687241dbc89f576fa'},
 'binding':{'path':str(parent.DEFAULT_BINDING),'sha256':'936c7becdcc476c54ec39d986d4811f5d3982c28c9ea0822779badefd954d583'},
 'result':{'path':'/tmp/libero-task7-fixed-representation-optimization-20261109-v1.json','sha256':'33021ef69f0040396d33923b5606c62248815c55d0c3655d76cfcc94515b1132'},
 'audit':{'path':'/tmp/libero-task7-fixed-representation-optimization-artifacts-20261109-v1/independent-audit.json','sha256':'88acec5c4276bec92778237448abdf7da71eab2fb189c5c14509e500458ad6d2'},
 'completion':{'path':'/tmp/libero-task7-fixed-representation-optimization-artifacts-20261109-v1/completion.json','sha256':'4c79178ede41348124177ccd6c53caf6e1e567f3b4fcf6a0c7fc1e2c63fc1832'},
 'report':{'path':str(ROOT/'docs/libero-task7-fixed-representation-optimization-results-2026-11-09.md'),'sha256':'da2e76656df6dc91e15e2910727d6952c2824a3e5f4b646085fdfc552e6c8577'},
}
IDS=['task','seed','arm','fold','mode']
NUMBERS=['fixed_other6_floor','component3_mse','penalty','trainable_objective','total_objective','base_l2','penalty_l2','total_l2','directional_derivative']
COUNTS=['clip_'+n+'_n' for n in ['below','low_tie','inside','high_tie','above']]+['hinge_'+n+'_n' for n in ['negative','zero','positive']]
COLUMNS={
 'states':IDS+['extra_updates','n']+NUMBERS+COUNTS,
 'trials':IDS+['from_extra_updates','trial_index','alpha','trainable_objective','total_objective','slope','armijo_bound','accepted'],
 'endpoints':IDS+['condition','extra_updates','n']+NUMBERS+COUNTS,
 'heads':IDS+['extra_updates','trials','terminal_reason','initial_objective','final_objective','initial_trainable_objective','final_trainable_objective','initial_gradient_l2','final_gradient_l2','trainable_ratio','gradient_ratio','objective_target_met','gradient_target_met','joint_cue','status'],
}
CASES=parent.CASES
ANCHORS=[0,10,20,30,40,50]
CLASSES=['both','gradient_only','objective_only','neither']
FILES={
 'analysis':ROOT/'experiments/analyze_libero_task7_optimization_path_decomposition.py',
 'audit':ROOT/'experiments/audit_libero_task7_optimization_path_decomposition.py',
 'guard':ROOT/'experiments/libero_task7_optimization_path_decomposition_access.py',
 'metadata':Path(__file__).resolve(),'tests':TESTS,'plan':PLAN,
}
SCOPE={'task_ids':[7],'seeds':p.SEEDS,'source_arms':p.ARMS,'original_Policy_updates':500,'saved_auxiliary_extra_updates':50,'future_stabilization_steps':5,
 'source':'own20_fit_demos_of_support0..29_all_tge1;encoder_seen_all30;no_new_unseen_data',
 'loss_representation_thresholds_or_checkpoints_changed':False,'archive_head_checkpoint_HDF5_RGB_proprio_reads_including_SHA':0,
 'new_training_forward_gradient_environment_or_sealed_calls':0,'probe_source_query_validation_heldout_qualification_reserved_final_access':False,
 'Policy_SPT_Knowledge_Module_changes':False,'old_cues_G_failed_H_unrun_unchanged':True}
DESIGN={'cohorts':'all36;outcome_stratified_lambda1_MSEup11_and_nonjoint21;not_blind_or_causal',
 'terms':'DeltaT=DeltaMSE3/7+lambda*DeltaP;DeltaF=DeltaT;C6_fixed;endpoint_equals_50step_sum',
 'step_classes':['mse_up_penalty_down','both_down','mse_down_penalty_up','other'],
 'compensation':'positive_DeltaMSE3/7_over_minus_DeltaP_only;zero_denominator_null_no_epsilon',
 'anchors':ANCHORS,'windows':list(zip(ANCHORS[:-1],ANCHORS[1:])),
 'path':'all50_updates;first_Armijo_acceptance;accepted_trial_equals_next_state;alpha_min_median_max;reject_counts;gradient_increases_and_nonzero_change_sign_switches',
 'normalizations':'window_T_drop_over_head_T0;window_T_drop_over_total_Tdrop;Gend_over_Gstart;zero_null',
 'groups':'12_source_loss_seed_x(all+4_goal_classes)=60;6_lambda1_source_seed_x2_MSEcohorts=12;retain_empty',
 'numeric_signs':'strict_on_saved_float64_no_epsilon_relabel','statistics':'descriptive_equal_head_fold_values;no_pvalues_or_independent_fold_claim',
 'limit':'aggregate_fit_only_no_identification_of_specific_frames_directions_or_physical_phase;no_checkpoint_or_training_decision_from_validation'}
EXECUTION={'status':'binding_required_before_source_CSV_SHA_or_data','device':'CPU','python':'3.10.22',
 'environment':parent.EXECUTION['environment'],'seconds_per_stage':120,'RSS_bytes_per_stage':268435456,'overall_seconds':300,
 'CSV_bytes_max':16777216,'absolute_tolerance':1e-10,'relative_tolerance':0,'technical_retries':0,'overwrite':False,
 'stop':'any_lineage_projection_schema_role_coverage_conservation_path_or_audit_mismatch_keep_partial_no_relaxation'}


def budget():
 rows={'states':1836,'trials':12285,'endpoints':72,'heads':36}
 tables={'heads':36,'transitions':1800,'windows':180,'anchors':216,'strata':72}
 return {'source_CSV_passes_per_stage':4,'rows_per_stage':rows,'projected_cells_per_stage':{k:n*len(COLUMNS[k]) for k,n in rows.items()},
  'lexical_cells_per_stage':219609,'total_source_CSV_passes':8,'total_source_rows':28458,'total_projected_cells':412578,
  'total_lexical_cells':439218,'source_SHA_per_stage':8,'total_source_SHA':16,'tables':tables,'output_rows':sum(tables.values()),
  'seed_summaries':12,'source_loss_summaries':4,'new_training_forward_gradient_array_or_sealed_reads':0}


def safe_ratio(a,b):
 p.require(all(type(x) in (float,int) and math.isfinite(x) for x in (a,b)),'finite ratio inputs required')
 if b==0:return None
 value=a/b;p.require(math.isfinite(value),'nonfinite ratio');return value


def goal_class(objective,gradient):
 p.require(type(objective) is type(gradient) is bool,'boolean targets required')
 return 'both' if objective and gradient else 'gradient_only' if gradient else 'objective_only' if objective else 'neither'


def terms(m0,p0,t0,m1,p1,t1,coefficient):
 p.require(coefficient in (0.,1.) and all(type(x) in (float,int) and math.isfinite(x) for x in (m0,p0,t0,m1,p1,t1)),'invalid decomposition inputs')
 dm=m1-m0;dp=p1-p0;dt=t1-t0
 p.require(abs(dt-(dm/7+coefficient*dp))<=1e-10,'loss decomposition does not conserve')
 kind='mse_up_penalty_down' if dm>0 and dp<0 else 'both_down' if dm<0 and dp<0 else 'mse_down_penalty_up' if dm<0 and dp>0 else 'other'
 return dict(delta_mse3=dm,delta_mse3_scaled=dm/7,delta_penalty=dp,weighted_delta_penalty=coefficient*dp,delta_trainable=dt,
  conservation_residual=dt-dm/7-coefficient*dp,step_class=kind)


def create():
 parents={k:p.load_pin(v) for k,v in PARENTS.items() if k!='report'}
 p.require(p.pin(Path(PARENTS['report']['path']))==PARENTS['report'],'parent report changed')
 science=parent.validate();r=parents['result'];audit=parents['audit'];done=parents['completion']
 p.require(parents['science']==science and r['science_config']==done['science_config']==PARENTS['science']
  and r['binding']==done['binding']==PARENTS['binding'] and audit['result']==done['result']==PARENTS['result']
  and done['audit']==PARENTS['audit'] and done['status']=='complete_independently_audited_finite_optimization'
  and audit['status']=='passed_independent_finite_optimization_audit','completed parent lineage required')
 inputs={name:r['tables'][name] for name in COLUMNS}
 p.require(all(inputs[k]['rows']==n for k,n in budget()['rows_per_stage'].items())
  and all(set(COLUMNS[k]).issubset(inputs[k]['columns']) for k in inputs),'source layout differs')
 p.require(sum(inputs[k]['rows']*len(inputs[k]['columns']) for k in inputs)==219609,'source lexical budget differs')
 identities=[{k:h[k] for k in IDS} for h in r['heads']]
 p.require(len(identities)==36 and all(h['extra_updates']==50 and h['terminal_reason']=='update_cap' for h in r['heads']),'all36 capped heads required')
 return copy.deepcopy(dict(kind='task7_saved_optimization_path_decomposition',version=1,formal_result=False,status='scientific_frozen_execution_binding_required',
  parents=PARENTS,scope=SCOPE,design=DESIGN,execution=EXECUTION,budget=budget(),inputs=inputs,projections=COLUMNS,
  head_identities=identities,fit_rows={str(f['fold']):f['fit_tge1_rows'] for f in science['inputs']['folds']},
  expected_counts={'heads':36,'lambda1_mse_up':11,'nonjoint':21,'goal_classes':{'both':15,'gradient_only':10,'objective_only':2,'neither':9}},
  code={'metadata':p.pin(Path(__file__).resolve()),'tests':p.pin(TESTS)},plan=p.pin(PLAN),future_files={k:str(v) for k,v in FILES.items()},output=OUTPUT,artifacts=ARTIFACTS))


def validate(path=DEFAULT_CONFIG):
 cfg=json.loads(Path(path).read_text());p.require(json.dumps(cfg,sort_keys=True,allow_nan=False)==json.dumps(create(),sort_keys=True,allow_nan=False),'frozen path-decomposition protocol differs');return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
 cfg=validate(config);b=json.loads(Path(binding).read_text())
 p.require(b['status']=='bound_before_payload' and b['before_research_payload'] is True and b['science_config']==p.pin(config)
  and b['budget']==cfg['budget'] and b['CPU_environment']==EXECUTION['environment'] and b['future_output']==OUTPUT and b['future_artifacts']==ARTIFACTS,'new analysis binding required')
 refs=b['dependencies'];paths=[p.resolve(r['path']).resolve() for r in refs]
 p.require(len(paths)==len(set(paths)) and set(FILES.values()).union({DEFAULT_CONFIG}).issubset(paths),'analysis roles not bound')
 p.require(all(r in refs for r in p.load_pin(PARENTS['binding'])['dependencies']),'parent724 closure missing')
 for ref in refs:
  path=p.resolve(ref['path']);p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path)==ref,'bound dependency changed or source payload in closure')
 return b


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
 if args.write_freeze:
  with DEFAULT_CONFIG.open('x') as f:json.dump(create(),f,ensure_ascii=False,indent=2);f.write('\n')
  result={'status':'scientific_frozen_no_source_CSV_or_array_read'}
 else:validate();result={'status':'metadata_valid_no_source_data'}
 print(json.dumps(result))
