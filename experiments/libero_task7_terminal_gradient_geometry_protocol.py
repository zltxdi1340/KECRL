"""Metadata/human-vector freeze for saved-gradient linear feasibility and branches."""
from __future__ import annotations
import argparse
from fractions import Fraction
import json
import math
from pathlib import Path
from experiments import libero_task7_m_priority_direction_protocol as parent

p, ROOT, IDS = parent.p, parent.ROOT, parent.IDS
STEM = 'libero-task7-terminal-gradient-geometry'
CONFIG = ROOT/f'configs/{STEM}-readonly_v1.json'
FREEZE_BINDING = ROOT/f'configs/{STEM}-freeze-binding_v1.json'
BINDING = ROOT/f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT/f'docs/{STEM}-plan-2026-11-11.md'
TESTS = ROOT/'tests/test_libero_task7_terminal_gradient_geometry_protocol.py'
ARTIFACTS = Path(f'/tmp/{STEM}-20261111-v1')
OUTPUT, AUDIT = ARTIFACTS/'analysis.json', ARTIFACTS/'independent-audit.json'
PARENTS = {
 'science': {'path': str(parent.CONFIG), 'sha256': '6f13044f54f06e19fcd745625f368daf6104dd1d2a88d6bab62a82e4b35b402f'},
 'binding': {'path': str(parent.BINDING), 'sha256': '33ccca6b67b3eb569aafdf1308707d010bf2a8b420af0f70cdca7d0f467647fb'},
 'result': {'path': parent.OUTPUT, 'sha256': '8d869c68ed5543ce6794f5eb0e5bf6024a948aa18781f9be8cbd4c3202d930fc'},
 'completion': {'path': str(Path(parent.ARTIFACTS)/'completion.json'), 'sha256': '9e2180fb5b2755e2f1541b4da5710a8df944c233d52bcfea04893459a1cc965b'},
 'process': {'path': str(Path(parent.OUTPUT).with_name(Path(parent.OUTPUT).stem+'-process.json')), 'sha256': '6bedaabb41f0bcaf3437080127e51023079acf7a42faecd3bba1be28c4607aab'},
 'report': {'path': str(ROOT/'docs/libero-task7-m-priority-direction-results-2026-11-11.md'), 'sha256': '21a9d17f60fff49ee2c51d791c651c11bde27fcd28ee0284aea8ec10f73c1ac1'},
}
INPUTS = {
 'P': {'path': str(Path(parent.ARTIFACTS)/'primary-result.json'), 'sha256': 'e3e1554cde2128fa07afa7723f244c5c32f79cea9aacbfe82d4b138ee7e651b7'},
 'U': {'path': str(Path(parent.ARTIFACTS)/'independent-audit.json'), 'sha256': 'f70838eec9c2498b20b585c74bae4201114ecb83b0a26b8f1214b53d21a59db5'},
}
ZERO_IDS = [('A_original',20261106,2),('B_fomaml',20261106,1),('B_fomaml',20261106,2),
            ('B_fomaml',20261107,1),('A_original',20261108,2),('B_fomaml',20261108,2)]
HEAD_HEADER = IDS+['optimization_arm']
ENDPOINT_HEADER = IDS+['condition']
HEAD_FIELDS = IDS+['optimization_arm','extra_updates','trials','terminal_reason','accepted_MSE_step_increases']
CLIP = ['clip_'+name+'_n' for name in ('below','low_tie','inside','high_tie','above')]
HINGE = ['hinge_'+name+'_n' for name in ('negative','zero','positive')]
POINT_FIELDS = IDS+['condition','extra_updates','n','component3_mse','penalty','trainable_objective',
 'total_l2','base_l2','penalty_l2','direction_total_reference_norm','direction_base_reference_norm','chosen_direction_norm',
 'selected_T_slope','selected_MSE_slope','directional_derivative','exact_MSE_slope','exact_penalty_slope',
 'required_M_slope_bound','required_margin_fraction','selected_joint','selected_margin_met','exact_margin_met',
 'direction_status','direction_construction_status','branch_digest']+CLIP+HINGE
VECTORS = ['total_gradient','base_gradient','penalty_gradient','direction']
FUTURE_FILES = {
 'primary': ROOT/'experiments/analyze_libero_task7_terminal_gradient_geometry.py',
 'guard': ROOT/'experiments/libero_task7_terminal_gradient_geometry_access.py',
 'audit': ROOT/'experiments/audit_libero_task7_terminal_gradient_geometry.py',
 'runtime_tests': ROOT/'tests/test_libero_task7_terminal_gradient_geometry.py',
}
FUTURE_CONTRACT = ROOT/f'docs/{STEM}-execution-contract-2026-11-11.md'


def _binary64_vector(values):
    p.require(type(values) is list and len(values)==257
              and all(type(x) is float and math.isfinite(x) for x in values),'finite binary64 vector257 required')
    return [Fraction.from_float(x) for x in values]


def sign(value):
    return int(value>0)-int(value<0)


def linear_certificate(total,base,radius,M_bound):
    """Exact real model of cached binary64 coordinates; creates no direction."""
    G,Gb=_binary64_vector(total),_binary64_vector(base)
    p.require(type(radius) is float and math.isfinite(radius) and radius>=0
              and type(M_bound) is float and math.isfinite(M_bound) and M_bound<0,'finite saved radius/nonzero negative M bound required')
    r=Fraction.from_float(radius);m=Fraction.from_float(M_bound)/7
    A=sum(x*x for x in G);B=sum(x*x for x in Gb);z=sum(x*y for x,y in zip(G,Gb))
    gram=A*B-z*z;p.require(gram>=0,'exact Gram determinant cannot be negative')
    capacity=r*r*B-m*m;gap=r*r*gram-m*m*A
    cap_nonempty=capacity>=0
    if not cap_nonempty:feasible=False;tmin=None;reason='M_halfspace_empty_in_saved_radius'
    elif A==0:feasible=False;tmin=0;reason='zero_total_gradient_no_strict_T'
    elif z>0:feasible=True;tmin=-1;reason='positive_G_dot_Gb_linear_witness'
    elif gap>0:feasible=True;tmin=-1;reason='strict_joint_linear_feasible'
    elif gap==0:feasible=False;tmin=0;reason='T_linear_minimum_zero'
    else:feasible=False;tmin=1;reason='T_linear_minimum_positive'
    return dict(M_halfspace_nonempty=cap_nonempty,saved_gradient_linear_feasible=feasible,
      total_gradient_zero=A==0,base_gradient_zero=B==0,dot_G_Gb_sign=sign(z),Gram_sign=sign(gram),
      M_capacity_sign=sign(capacity),T_feasibility_gap_sign=sign(gap),minimum_linear_T_sign=tmin,reason=reason,
      proof_model='cached_binary64_coordinates_as_exact_reals;radius=saved_g;7Gb_dot_d<=saved_M_bound;G_dot_d<0',
      alternative_direction_exact_gate_checked=False,finite_trial_checked=False)


def branch_status(n,counts):
    p.require(type(n) is int and n>0 and set(counts)==set(CLIP+HINGE)
      and all(type(x) is int and 0<=x<=n for x in counts.values())
      and sum(counts[k] for k in CLIP)==sum(counts[k] for k in HINGE)==n,'complete clip/hinge counts required')
    clip_ties=counts['clip_low_tie_n']+counts['clip_high_tie_n'];hinge_ties=counts['hinge_zero_n']
    return dict(clip_tie_n=clip_ties,hinge_zero_n=hinge_ties,smooth_by_cached_counts=clip_ties==hinge_ties==0,
      alternative_direction_exact_gate_checked=False,finite_trial_checked=False)


def budget():
    # Per JSON: inspect all headers/heads; only18 terminal +6 saved500 point bodies.
    headers=36*len(HEAD_HEADER)+54*len(ENDPOINT_HEADER)
    body=36*len(HEAD_FIELDS)+24*len(POINT_FIELDS);coordinates=24*4*257
    return dict(stages=2,per_stage_source_JSON_passes=2,per_stage_source_SHA=4,total_source_JSON_passes=4,total_source_SHA=8,
      per_JSON_rows=dict(head_headers=36,endpoint_headers=54,heads=36,terminal_points=18,zero_saved500_points=6),
      per_stage_header_values=2*headers,per_stage_scalar_body_values=2*body,per_stage_vector_coordinates=2*coordinates,
      per_stage_projection_values=2*(headers+body+coordinates),total_projection_values=4*(headers+body+coordinates),
      points_per_stage=48,total_points=96,vector_dimension=257,per_stage_float_norms=4*48,per_stage_float_dots=6*48,
      per_stage_float_dot_coordinate_products=6*48*257,per_stage_gradient_sum_coordinate_checks=48*257,
      per_stage_zero_gradient_coordinate_comparisons=12*3*257,per_stage_exact_inner_product_reductions=3*48,
      per_stage_exact_coordinate_products=3*48*257,total_float_norms=384,total_float_dots=576,
      total_exact_inner_product_reductions=288,total_exact_coordinate_products=74016,
      primary_result_audit_JSON_reads=1,primary_result_audit_SHA=2,technical_retries=0,
      output_rows=dict(terminal_geometry=36,zero_same_point_control=12,source_seed=12,source=4,cross_engine=18),
      source_NPZ_CSV_HDF5_checkpoint_private_or_sealed_reads_including_SHA=0,
      new_forward_gradient_autograd_optimizer_trial_environment_calls=0,new_direction_vectors=0)


def create():
    bound=parent.require_binding()
    docs={k:p.load(ref) for k,ref in PARENTS.items() if k!='report'}
    p.require(p.pin(PARENTS['report']['path'])==PARENTS['report'],'completed M-priority report changed')
    r,done,proc=docs['result'],docs['completion'],docs['process']
    p.require(docs['binding']==bound and len(bound['dependencies'])==887
      and r['status']==done['status']==proc['status']=='complete_independently_audited_M_priority_direction'
      and r['primary']==INPUTS['P'] and r['audit']==done['audit']==INPUTS['U'] and done['result']==PARENTS['result']
      and r['science_config']==done['science_config']==proc['science_config']==PARENTS['science']
      and r['binding']==done['binding']==proc['binding']==PARENTS['binding']
      and r['primary_updates']==r['independent_U_updates']==933 and r['local_S_updates']==0
      and r['local_numeric_states']==969 and r['local_trial_targets']==7957
      and r['local_stop_disagreements']==r['local_accept_disagreements']==0
      and r['original_Policy_CNN_environment_or_sealed_calls']==0 and r['formal_result'] is False
      and len(proc['processes'])==2 and all(row['exit_code']==0 for row in proc['processes']),'completed M-priority lineage required')
    p.require(len(docs['science']['cohorts'])==18,'all18 terminal identities required')
    return dict(kind='task7_saved_terminal_gradient_linear_geometry',version=1,formal_result=False,
      status='scientific_freeze_only_execution_binding_required',parents=PARENTS,inputs=INPUTS,
      identities=docs['science']['cohorts'],source_optimization_arms=parent.ARMS,zero_ids=[list(k) for k in ZERO_IDS],
      selection=dict(terminal_condition=parent.ARMS[1]+'_final',control_zero_condition='saved500',
        expected_heads=36,expected_endpoints=54,candidate_updates=33,candidate_trials=266,control_updates=900,control_trials=7691,
        terminal_stop='non_descent_selected_direction',zero_subset='fixed_outcome_stratum_6;not_new_generalization'),
      fields=dict(head_headers=HEAD_HEADER,endpoint_headers=ENDPOINT_HEADER,heads=HEAD_FIELDS,point_scalars=POINT_FIELDS,point_vectors=VECTORS),
      scope=dict(original_Policy_updates=500,future_stabilization_steps=5,loss_lambda=1,
        source='own20_fit_support0..29_tge1_encoder_seen_all30',original_control_stopping_and_four_reference_cues_unchanged=True,
        extra_original_archives_parameters_inputs_features_targets_or_trials_semantic_reads=0,SPT_Knowledge_Module_changes=False),
      geometry=dict(radius='saved_direction_total_reference_norm_not_renormalized',M_bound='saved_required_M_slope_bound_not_replaced_by_ideal_rho',
        linear_model='G=stored_total_gradient;Gb=stored_base_gradient;norm_d<=r;7Gb_dot_d<=B;G_dot_d<0',
        exact_arithmetic='binary64_as_exact_rationals;no_sqrt_or_epsilon_or_cosine_clamp_in_feasibility',
        proof='A=G.G;B2=Gb.Gb;z=G.Gb;m=B/7;C=r^2*B2-m^2;D=r^2*(A*B2-z^2)-m^2*A;C>=0_and_A>0_and_(z>0_or_D>0)',
        raw_readout='four_norms_and_six_dots_per_point;cosines_G_Gb_G_Gp_Gb_Gp;raw_out_of_range_flag_no_clamp',
        conservation='G~Gb+Gp_per_coordinate_abs1e-10;keep_stored_G',
        branch='counts_no_clip_ties_and_no_hinge_zero_is_sufficient_smooth_label;otherwise_alternative_exact_unknown',
        never_claim='linear_feasible_does_not_verify_new_direction_exact_or_fixed_grid_Armijo_or_qualification'),
      audit=dict(primary='stdlib_Fraction_exact_capacity_Gram_and_gap_signs',independent='own_common_power_of_two_integer_coordinates;no_primary_math_import',
        local_float_absolute_tolerance=1e-10,relative_tolerance=0,exact_signs_and_IDs_tolerance=0,
        P_U='classifications_and_branch_disagreements_reported;not_same_parameter_gate_across_independent_paths',
        zero_receipt='3gradient_bits_and_branch_equal;zero_update_same_start_from_parent_validated_lineage;cached_control_selected_exact_certificates_only'),
      budget=budget(),execution=dict(device='CPU',python='3.10.22',seconds_per_stage=120,overall_seconds=300,RSS_bytes=536870912,
        input_JSON_bytes_max=201326592,output_JSON_bytes_max=4194304,
        environment={'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
        technical_stop='source_provenance_fields_shapes_counts_nonfinite_or_conservation_or_independent_audit_mismatch;retain_partial_no_retry'),
      code=dict(metadata=p.pin(Path(__file__).resolve()),tests=p.pin(TESTS)),plan=p.pin(PLAN),
      future_files={k:str(v) for k,v in FUTURE_FILES.items()},future_contract=str(FUTURE_CONTRACT),output=str(OUTPUT),audit_output=str(AUDIT),
      freeze_access=dict(source_numeric_JSON_reads_or_SHA=0,source_NPZ_CSV_checkpoint_HDF5_reads_or_SHA=0,
        research_gradient_geometry_or_directions=0,metadata_and_human_vectors_only=True,execution_binding_present=False))


def validate(path=CONFIG):
    cfg=json.loads(Path(path).read_text());p.require(cfg==create(),'terminal geometry scientific freeze differs');return cfg


def require_binding(config=CONFIG,binding=BINDING):
    cfg=validate(config);bound=json.loads(Path(binding).read_text())
    p.require(bound['status']=='bound_before_payload' and bound['before_research_payload'] is True
      and bound['science_config']==p.pin(config) and bound['scientific_freeze_binding']==p.pin(FREEZE_BINDING)
      and bound['budget']==cfg['budget'] and bound['CPU_environment']==cfg['execution']['environment']
      and bound['future_output']==str(OUTPUT) and bound['future_artifacts']==str(ARTIFACTS),'new geometry execution binding required')
    refs=bound['dependencies'];paths=[r['path'] for r in refs];freeze=p.load(p.pin(FREEZE_BINDING))
    p.require(freeze['status']=='scientific_frozen_execution_not_bound' and freeze['science_config']==p.pin(config)
      and len(paths)==len(set(paths)) and all(r in refs for r in freeze['dependencies'])
      and all(p.pin(path) in refs for path in [*FUTURE_FILES.values(),FUTURE_CONTRACT,Path(config),FREEZE_BINDING]),'new geometry roles/scientific closure missing')
    p.require(all(r in refs for r in p.load(PARENTS['binding'])['dependencies']) and PARENTS['binding'] in refs,'parent887 closure missing')
    p.require(not(set(paths)&{r['path'] for r in INPUTS.values()}),'source numeric JSON cannot be dependency-hashed before binding')
    for ref in refs:p.require(Path(ref['path']).suffix in {'.py','.md','.json','.toml'} and p.pin(ref['path'])==ref,'bound dependency changed')
    return bound


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write-freeze',action='store_true');args=parser.parse_args()
    if args.write_freeze:p.write(CONFIG,create())
    else:validate()
    print(json.dumps({'status':'terminal_geometry_scientific_freeze_no_research_payload'}))
