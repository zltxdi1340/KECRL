"""Frozen stdlib-only review of five saved fine20 stops; no new readout."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from experiments import libero_task7_mse_guard_finer_grid_protocol as parent

p,ROOT=parent.p,parent.ROOT
geometry=parent.parent
IDS,VECTORS,COUNTS,COLUMNS=geometry.IDS,geometry.VECTORS,geometry.COUNTS,copy.deepcopy(geometry.COLUMNS)
TOL=geometry.TOL
gradient_geometry,trial_terms=geometry.gradient_geometry,geometry.trial_terms
DEFAULT_CONFIG=ROOT/'configs/libero-task7-fine-stop-direction-review-readonly_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-task7-fine-stop-direction-review-execution-binding_v1.json'
PLAN=ROOT/'docs/libero-task7-fine-stop-direction-review-plan-2026-11-10.md'
OUTPUT='/tmp/libero-task7-fine-stop-direction-review-20261110-v1.json'
ARTIFACTS='/tmp/libero-task7-fine-stop-direction-review-artifacts-20261110-v1'
PARENTS={
 'science':{'path':str(parent.DEFAULT_CONFIG),'sha256':'03e8d5bd337a40da181100f8fdf1e177edb12b49ec2f152149d3513b16e666e2'},
 'binding':{'path':str(parent.DEFAULT_BINDING),'sha256':'50c66596de57a1a83251f260a1b414688ebbbcd764d3b0c51077356e40388339'},
 'result':{'path':parent.OUTPUT,'sha256':'006090f077ea86a55bb254b3d86f760ac2e2433b4eaa3777eddc1fae13d78c26'},
 'audit':{'path':parent.ARTIFACTS+'/independent-audit.json','sha256':'15330ab3c900c119113f03b832c3262177c0bb0733ab92719d5d6a02a3b3694e'},
 'completion':{'path':parent.ARTIFACTS+'/completion.json','sha256':'562b16310f1ce1f62eae455c844e059165c7ed0226cabf3396a9d745fed7ff27'},
 'process':{'path':'/tmp/libero-task7-mse-guard-finer-grid-20261110-v1-process.json','sha256':'864e1867c73877c154afdaf71eacd7cb4e0f1b85f278a239550ec0e6ad383718'},
 'report':{'path':str(ROOT/'docs/libero-task7-mse-guard-finer-grid-results-2026-11-10.md'),'sha256':'3b3bf3bd16ce29739d41e2da2d3e10cbd427a73bb3096dbf7813b50f6959f1ee'},
}
FUTURE_FILES={
 'analysis':ROOT/'experiments/analyze_libero_task7_fine_stop_direction_review.py',
 'audit':ROOT/'experiments/audit_libero_task7_fine_stop_direction_review.py',
 'guard':ROOT/'experiments/libero_task7_fine_stop_direction_review_access.py',
 'metadata':Path(__file__).resolve(),
 'tests':ROOT/'tests/test_libero_task7_fine_stop_direction_review.py','plan':PLAN,
}
HEADS=[dict(task=7,seed=s,arm=a,fold=f,mode=parent.source.MODE,cohort='short',stop_extra_updates=k)
 for s,a,f,k in [(20261106,'A_original',1,7),(20261106,'B_fomaml',0,7),
 (20261107,'A_original',1,8),(20261108,'A_original',0,16),(20261108,'B_fomaml',0,10)]]
EXECUTION=dict(python='3.10.22',environment=copy.deepcopy(parent.EXECUTION['environment']),
 seconds_per_stage=120,RSS_bytes_per_stage=268435456,overall_seconds=300,
 CSV_bytes_max=16777216,absolute_tolerance=TOL,relative_tolerance=0,technical_retries=0,overwrite=False)
DESIGN=dict(points=['saved500','fine20_final'],trial_indices=list(range(20)),minimum_alpha=2.**-19,
 selection='fixed_old_short5_of18;all5_new_fine20_stops;remaining13_metadata_only;outcome_selected_not_causal',
 exact_slope='MSE:7*dot(Gb,d)_only_without_clip_ties;P:sT-dot(Gb,d);sT_from_cached_exact_right_derivative',
 direction='cached_full_unnormalized_minus_G;no_new_direction_or_gradient',
 inference='strict_slack_and_slope_signs;abs1e-10_descriptive_band_only;linear_slack_over_slope_is_not_safe_step',
 trial_check='all20_exact_grid_flags_and_reasons;fixed_auxiliary500_envelope_no_epsilon;all_rejected',
 trial_decomposition='DeltaM/7+DeltaP=DeltaT;gate_excess;Armijo_margin;secant_and_first_order_residual',
 finite_only='no_untried_alpha_or_array_model_forward;no_global_infeasibility_or_closed_loop_root_cause',
 physical_JSON_parse='whole_parent_JSON_physically_decoded;nonselected_records_and_parameter_values_excluded_from_numeric_projection',
 metadata_discovery='one_whole_parent_JSON_parse_for_keys_only_before_freeze;no_new_direction_or_trial_calculation',
 statistics='descriptive_only;all3seeds_source_cells_including_empty;folds_not_independent_seeds')


def feasibility_evidence(*args):
 value=geometry.feasibility_evidence(*args)
 scale=value['linear_alpha_scale']
 value['linear_scale_over_smallest_grid_alpha']=scale/(2.**-19) if scale is not None else None
 return value


def create():
 cfg=parent.validate();parent.require_binding()
 lineage={k:p.load_pin(PARENTS[k]) for k in ('audit','completion','process')}
 a,done,process=(lineage[k] for k in ('audit','completion','process'))
 p.require(p.pin(Path(PARENTS['report']['path']))==PARENTS['report']
  and a['result']==done['result']==PARENTS['result'] and done['audit']==PARENTS['audit']
  and done['science_config']==process['science_config']==PARENTS['science']
  and done['binding']==process['binding']==PARENTS['binding']
  and a['status']=='passed_independent_mse_guard_finer_grid_audit'
  and done['status']=='complete_independently_audited_mse_guard_finer_grid'
  and process['status']=='complete_independently_audited' and len(process['processes'])==2
  and all(row['exit_code']==0 for row in process['processes']),'completed fine20 lineage required')
 return dict(kind='task7_five_fine20_stop_saved_direction_review',version=1,formal_result=False,
  parents=PARENTS,inputs={'source_JSON':PARENTS['result']},heads=copy.deepcopy(HEADS),
  scope=dict(original_Policy_updates=500,future_stabilization_steps=5,loss_representation_unchanged=True,
   new_training_forward_gradient_or_environment_calls=0,array_CSV_checkpoint_HDF5_RGB_reads_including_SHA=0,
   probe_source_query_validation_heldout_qualification_reserved_final_access=False,SPT_Knowledge_Module_changes=False),
  fit_rows={str(f['fold']):len(f['fit_rows']) for f in cfg['inputs']['folds']},design=DESIGN,execution=EXECUTION,
  budget=dict(parent_JSON_parses_per_stage=1,parent_JSON_SHA_per_stage=2,total_parent_JSON_parses=2,total_parent_JSON_SHA=4,
   source_rows_per_stage=dict(endpoints=54,trials=6569,heads=36,pairs=18),
   selected_rows_per_stage=dict(endpoints=10,parameters=2570,trials=100),
   projected_cells_per_stage={name:n*len(COLUMNS[name]) for name,n in [('endpoints',10),('parameters',2570),('trials',100)]},
   cached_gradient_coordinates_per_stage=10*4*257,all18_heads_metadata_checked=18,excluded_head_count=13,
   tables=dict(points=10,trials=100,heads=5,strata=6),metadata_discovery_JSON_parses_before_freeze=1,
   new_training_forward_gradient_array_or_sealed_calls=0),
  code={k:p.pin(v) for k,v in FUTURE_FILES.items()},output=OUTPUT,artifacts=ARTIFACTS)


def validate(path=DEFAULT_CONFIG):
 cfg=json.loads(Path(path).read_text())
 p.require(json.dumps(cfg,sort_keys=True,allow_nan=False)==json.dumps(create(),sort_keys=True,allow_nan=False),'fine-stop frozen configuration differs')
 return cfg


def require_binding(config=DEFAULT_CONFIG,binding=DEFAULT_BINDING):
 cfg=validate(config);b=json.loads(Path(binding).read_text())
 p.require(b['status']=='bound_before_payload' and b['before_research_payload'] is True
  and b['science_config']==p.pin(config) and b['budget']==cfg['budget']
  and b['CPU_environment']==EXECUTION['environment'] and b['future_output']==OUTPUT and b['future_artifacts']==ARTIFACTS,'fine-stop binding differs')
 refs=b['dependencies'];paths=[p.resolve(r['path']).resolve() for r in refs]
 p.require(len(paths)==len(set(paths)) and set(FUTURE_FILES.values()).union({DEFAULT_CONFIG}).issubset(paths)
  and all(r in refs for r in p.load_pin(PARENTS['binding'])['dependencies']),'fine-stop dependency closure missing')
 for ref in refs:
  path=p.resolve(ref['path']);p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path)==ref,'bound code changed or archive in closure')
 return b


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
 if args.write_freeze:
  with DEFAULT_CONFIG.open('x') as f:json.dump(create(),f,ensure_ascii=False,indent=2);f.write('\n')
 else:validate()
 print(json.dumps({'status':'frozen_metadata_only_no_new_analysis'}))
