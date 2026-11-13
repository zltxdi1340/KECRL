"""Metadata-only scientific freeze of finite fit-only auxiliary optimization."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_endpoint_gradient_protocol_v2 as parent

p, ROOT = parent.p, parent.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-fixed-representation-optimization-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-fixed-representation-optimization-execution-binding_v1.json'
PLAN = ROOT / 'docs/libero-task7-fixed-representation-optimization-plan-2026-11-09.md'
TESTS = ROOT / 'tests/test_libero_task7_fixed_representation_optimization_protocol.py'
OUTPUT = '/tmp/libero-task7-fixed-representation-optimization-20261109-v1.json'
ARTIFACTS = '/tmp/libero-task7-fixed-representation-optimization-artifacts-20261109-v1'
PARENTS = {
    'gradient_science': {'path': str(parent.DEFAULT_CONFIG), 'sha256': '5231b172352871ea0d1f760b8bfa637c85f0a1b3d595a49c43ff9ed38ef2831c'},
    'gradient_binding': {'path': str(parent.DEFAULT_BINDING), 'sha256': 'bc6f25bcac14cbcd307addbb8873d815560864f63cfb8b546f86411b501bf414'},
    'gradient_result': {'path': '/tmp/libero-task7-endpoint-gradient-20261109-v2.json', 'sha256': '43ace1221276fdfd68aaaaed9fda6366e0120d608856762d7f956dbee2886601'},
    'gradient_audit': {'path': '/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v2/independent-audit.json', 'sha256': '698fe898fc7525416556e1175a368831bcf827a7261f2cd9d304383289c1908e'},
    'gradient_completion': {'path': '/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v2/completion.json', 'sha256': 'f5c2ee4ed8b51e545c92d40043ca881774d8d475c63d873049bf550f074a6b29'},
    'gradient_report': {'path': str(ROOT / 'docs/libero-task7-endpoint-gradient-results-2026-11-09.md'), 'sha256': '88d4619971a5dbf4db22323af126873c78dd144de6ca9c4d0340aa1cccd1b4b5'},
}
CASES = parent.CASES
SOURCE_MEMBERS, HEAD_MEMBERS = parent.SOURCE_MEMBERS, parent.HEAD_MEMBERS
NEW_HEAD_MEMBERS = ['fit_rows', 'fit_demo_indices', 'scaler_mean', 'scaler_scale', 'additional_updates', 'weights', 'bias']
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_fixed_representation_optimization.py',
    'audit': ROOT / 'experiments/audit_libero_task7_fixed_representation_optimization.py',
    'guard': ROOT / 'experiments/libero_task7_fixed_representation_optimization_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_fixed_representation_optimization.py',
    'protocol': PLAN, 'metadata': Path(__file__).resolve(), 'metadata_tests': TESTS,
}
FUTURE_CONTRACT = ROOT / 'docs/libero-task7-fixed-representation-optimization-execution-contract-2026-11-09.md'
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'original_Policy_and_source_auxiliary_updates': 500, 'future_stabilization_steps': 5,
    'source_role': 'meta_support_demos0..29', 'fit_role': 'own20_whole_demos_all_tge1_including_neutral',
    'auxiliary_probe_scoring': 0, 'encoder_has_seen_all30_support_demos': True,
    'source_meta_query_validation_heldout_qualification_reserved_final_access': False,
    'new_original_Policy_CNN_environment_calls': 0, 'new_loss_lambda_representation_or_data_weight_change': False,
    'new_auxiliary_parameter_updates': 'explicit_finite_budget_not_zero;copies_only;not_Policy550',
    'SPT_Knowledge_Module_or_original_Policy_changes': False,
    'prior_joint_cues_G_failed_H_unrun_unchanged': True,
}
OPTIMIZATION = {
    'arms': ['hold_500', 'fullbatch_armijo'], 'start': 'each_own_saved500_W_b_copy_not_zero_or_retrained',
    'same_loss': 'F=C6+mean(e)/7+lambda*mean(relu(e-e0))/7;lambda0_base_direct;lambda1_unchanged',
    'same_precision': 'both_Float64_on_saved_f32_values_and_training_normalized_f32_z;not_Float32_Adam_replay',
    'direction': 'negative_full_unclipped_total_gradient;no_momentum_projection_weight_decay_or_Adam_state',
    'extra_updates_max': 50, 'trials_per_round_max': 12, 'alpha_initial': 1., 'backtrack_factor': .5,
    'armijo_c1': .0001, 'trial_index_range': [0, 11],
    'slope': 'exact_right_directional_derivative_along_d=-G;not_autograd_JVP_or_minus_norm_squared_at_kinks',
    'acceptance': 'first_trial_Ttrial<Tcurrent_and_Ttrial<=Tcurrent+c1*alpha*slope;no_epsilon',
    'alpha_restart': '1_at_every_round', 'final': 'last_accepted_after50_or_prespecified_stop;never_best_checkpoint',
    'stop_priority': ['update_cap', 'zero_gradient', 'non_descent_selected_direction', 'line_search_exhausted'],
    'no_early_success_stop_or_adaptive_budget': True,
    'control': 'same_initial500_readout_reused;input_and_parameter_immutable;no50_duplicate_control_evaluations',
    'fit_preprocessing': '18_rebuilds;paired_loss_scalers_bit_equal;reuse_same_readonly_z;36_admissions',
    'per_state': 'one_parameter_linear_form_three_gradient_vectors_one_direction_form;no_retained_graph_history',
    'per_trial': 'one_full_fit_component3_target_objective_form;no_gradient;rejected_trial_counted',
    'new_parameters': 'Float64_diagnostic_copy;source500_and_hold_unchanged;no_optimizer_state_inheritance',
    'saved_points': 'control_and_unique_final_only;intermediate_state_norms_and_trial_decisions_not_selectable_checkpoints',
}
DECISION = {
    'trainable_ratio_max': .99, 'gradient_ratio_max': .1,
    'joint': 'audited_and_Tstart>0_and_Gstart>0_and_Tend/Tstart<=.99_and_Gend/Gstart<=.1',
    'aggregate': 'all3fold_values_per_seed;equal_fold_descriptions;source_loss_cue_requires_all9heads',
    'zero_denominator': 'null_with_reason_no_epsilon', 'statistics': 'descriptive_only_folds_not_independent_seeds_no_pvalues',
    'meaning': 'finite_fit_mathematical_optimization_reference_not_global_optimality_or_fair_compute_algorithm_causality',
    'no_automatic_training_or_Policy_promotion': True,
    'next': 'joint_fit_improvement_only_motivates_separately_frozen_probe_check;failure_does_not_identify_representation_or_loss_root_cause',
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU',
    'runtime': {'python': '3.10.22', 'numpy': '1.22.4', 'torch': '2.4.1+cu124'},
    'environment': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    'primary_seconds_max': 300, 'audit_seconds_max': 600, 'overall_seconds_max': 960,
    'primary_RSS_bytes_max': 536870912, 'audit_RSS_bytes_max': 805306368,
    'output_CSV_bytes_max': 67108864, 'output_NPZ_bytes_max': 67108864,
    'float_absolute_tolerance': 1e-10, 'relative_tolerance': 0,
    'IDs_counts_shapes_members_clip_hinge_and_acceptance_decisions_tolerance': 0,
    'new_parameter_bytes': 'within_process_immutability_only;cross_implementation_coordinate_check_abs1e-10_not_bit_exact',
    'anchor_action_rounding_absolute_tolerance': 1e-7, 'anchor_objective_absolute_tolerance': 1e-6,
    'audit': 'independent_Torch_Float64_autograd_all_states_and_replay_all_tried_steps;no_primary_numerical_import',
    'binding': 'seven_roles_contract_science_parent710closure_before_archive_SHA_or_decode',
    'technical_retries': 0, 'overwrite': False,
    'technical_stop': 'any_provenance_role_counter_nonfinite_resource_coordinate_branch_or_Armijo_decision_mismatch;keep_partial_no_relaxation',
}


def budget():
    heads, nsum, rounds, trials = 36, 100512, 50, 12
    states = heads * (rounds + 1); candidate_forms = heads * rounds * trials
    state_rows = nsum * (rounds + 1); trial_rows = nsum * rounds * trials
    tables = {'endpoints': 72, 'parameters': 72 * 257, 'states_max': states,
              'trials_max': candidate_forms, 'heads': heads, 'scaler-parity': 18}
    return {'original_endpoints': 6, 'folds': 3, 'heads': heads, 'parameters_per_head': 257,
        'main_auxiliary_updates_max': heads * rounds, 'audit_replay_updates_max': heads * rounds,
        'parameter_coordinate_writes_max_per_stage': heads * rounds * 257,
        'scaler_reconstructions_per_stage': 18, 'fit_standardization_row_visits_per_stage': 50256,
        'fit_standardization_feature_coordinates_per_stage': 12865536, 'paired_reuses_per_stage': 18,
        'gradient_states_max_per_stage': states, 'gradient_vectors_max_per_stage': states * 3,
        'gradient_coordinates_max_per_stage': states * 3 * 257, 'direction_coordinates_max_per_stage': states * 257,
        'audit_autograd_calls_max': states * 3, 'main_autograd_calls': 0,
        'trial_objective_forms_max_per_stage': candidate_forms,
        'gradient_state_fit_row_visits_max_per_stage': state_rows,
        'gradient_state_seven_action_coordinates_max_per_stage': state_rows * 7,
        'trial_component3_fit_row_visits_max_per_stage': trial_rows,
        'all_linear_forms_max_per_stage': states * 2 + candidate_forms,
        'all_linear_form_row_visits_max_per_stage': state_rows * 2 + trial_rows,
        'all_linear_form_feature_multiplications_max_per_stage': (state_rows * 2 + trial_rows) * 256,
        'original_NPZ_loads_per_stage': 43, 'original_member_decodes_per_stage': 305,
        'new_NPZ_writes_primary': 36, 'new_NPZ_loads_audit': 36, 'new_member_decodes_audit': 252,
        'total_NPZ_loads': 122, 'total_member_decodes': 862,
        'original_archive_SHA_per_stage': 86, 'new_archive_SHA_primary': 36, 'new_archive_SHA_audit': 72,
        'total_archive_SHA_checks': 280, 'input_research_CSV_passes': 0,
        'tables': tables, 'output_CSV_rows_max': sum(tables.values()),
        'seed_summaries': 12, 'arm_loss_summaries': 4,
        'original_Policy_CNN_environment_or_sealed_calls': 0}


def finite_nonnegative(*values):
    p.require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in values), 'finite nonnegative scalar required')


def trial_alpha(index):
    p.require(type(index) is int and 0 <= index < 12, 'trial outside frozen grid')
    return 2. ** (-index)


def armijo_accept(current, trial, slope, index):
    finite_nonnegative(current, trial)
    p.require(type(slope) in (int, float) and math.isfinite(slope) and slope < 0, 'negative exact directional slope required')
    rhs = current + .0001 * trial_alpha(index) * slope
    p.require(math.isfinite(rhs), 'nonfinite Armijo bound')
    return trial < current and trial <= rhs


def stopping_reason(updates, gradient_norm, slope, rejected_trials=0):
    p.require(type(updates) is int and 0 <= updates <= 50 and type(rejected_trials) is int and 0 <= rejected_trials <= 12, 'iteration budget differs')
    finite_nonnegative(gradient_norm)
    p.require(type(slope) in (int, float) and math.isfinite(slope) and (gradient_norm != 0 or slope == 0), 'invalid zero-gradient slope')
    if updates == 50: return 'update_cap'
    if gradient_norm == 0: return 'zero_gradient'
    if slope >= 0: return 'non_descent_selected_direction'
    if rejected_trials == 12: return 'line_search_exhausted'
    return None


def endpoint_evidence(t0, tend, g0, gend, *, numeric_valid=True):
    finite_nonnegative(t0, tend, g0, gend)
    p.require(numeric_valid is True and tend <= t0, 'invalid audit or monotone path')
    tr = tend / t0 if t0 else None; gr = gend / g0 if g0 else None
    p.require(all(v is None or math.isfinite(v) for v in (tr, gr)), 'nonfinite ratio')
    loss_ok = tr is not None and tr <= .99; gradient_ok = gr is not None and gr <= .1
    cue = t0 > 0 and g0 > 0 and loss_ok and gradient_ok
    status = 'reference_zero' if not t0 or not g0 else 'joint_optimization_cue' if cue else 'partial_target_reduction' if tend < t0 else 'no_target_reduction'
    return {'trainable_ratio': tr, 'gradient_ratio': gr, 'objective_target_met': loss_ok,
            'gradient_target_met': gradient_ok, 'joint_cue': cue, 'status': status}


def joint_evidence(rows):
    p.require(len(rows) == 9 and {(r['seed'], r['fold']) for r in rows} == {(s,f) for s in p.SEEDS for f in range(3)}
              and len({(r['arm'],r['mode']) for r in rows}) == 1
              and all(r['arm'] in p.ARMS and r['mode'] in CASES and type(r['joint_cue']) is bool for r in rows), 'complete paired nine-head evidence required')
    return all(r['joint_cue'] for r in rows)


def create():
    parents = {name: p.load_pin(ref) for name, ref in PARENTS.items() if name != 'gradient_report'}
    p.require(p.pin(Path(PARENTS['gradient_report']['path'])) == PARENTS['gradient_report'], 'prior result report changed')
    cfg = parent.validate(); record = parents['gradient_result']; audit = parents['gradient_audit']; done = parents['gradient_completion']
    p.require(parents['gradient_science'] == cfg and done['status'] == 'complete_independently_audited_fit_endpoint_gradient'
        and audit['status'] == 'passed_independent_fit_endpoint_gradient_audit'
        and record['science_config'] == done['science_config'] == PARENTS['gradient_science']
        and record['binding'] == done['binding'] == PARENTS['gradient_binding']
        and audit['result'] == done['result'] == PARENTS['gradient_result'] and done['audit'] == PARENTS['gradient_audit']
        and record['inputs'] == cfg['inputs'] and record['representation_work']['scaler_reconstructions'] == 18,
        'completed compliant v2 lineage required')
    references = []
    for head in cfg['inputs']['heads']:
        identity = {k: head[k] for k in ('task','seed','arm','fold','mode')}
        matches = [r for r in record['points'] if all(r[k] == v for k,v in identity.items())]
        p.require(len(matches) == 2 and {r['point'] for r in matches} == {0,500} and head['training']['updates'] == 500, 'fixed two-point parent head missing')
        zero = next(r for r in matches if r['point'] == 0); start = next(r for r in matches if r['point'] == 500)
        references.append({**identity, 'n': start['n'], 'start500_total_objective': start['total_objective'],
            'start500_trainable_objective': start['trainable_objective'], 'start500_gradient_l2': start['total_l2'],
            'zero_residual_gradient_l2': zero['total_l2'], 'start500_branch_digest': start['branch_digest']})
    p.require(len(references) == 36, 'all36 source heads required')
    return copy.deepcopy({'kind': 'task7_fixed_representation_finite_optimization_diagnostic', 'version': 1,
        'status': 'scientific_freeze_only_execution_binding_required', 'formal_result': False,
        'parents': PARENTS, 'scope': SCOPE, 'optimization': OPTIMIZATION, 'decision': DECISION,
        'execution': EXECUTION, 'budget': budget(), 'inputs': cfg['inputs'], 'reference_points': references,
        'source_members': SOURCE_MEMBERS, 'head_members': HEAD_MEMBERS, 'new_head_members': NEW_HEAD_MEMBERS,
        'code': {'metadata': p.pin(Path(__file__).resolve()), 'tests': p.pin(TESTS)}, 'plan': p.pin(PLAN),
        'future_files': {k: str(v) for k,v in FUTURE_FILES.items()}, 'future_contract': str(FUTURE_CONTRACT),
        'output': OUTPUT, 'artifacts': ARTIFACTS})


def validate(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(json.dumps(cfg, sort_keys=True, allow_nan=False) == json.dumps(create(), sort_keys=True, allow_nan=False), 'finite optimization scientific freeze differs')
    return cfg


def require_binding(config=DEFAULT_CONFIG, binding=DEFAULT_BINDING):
    cfg = validate(config); b = json.loads(Path(binding).read_text())
    p.require(b['status'] == 'bound_before_payload' and b['before_research_payload'] is True
        and b['science_config'] == p.pin(config) and b['budget'] == cfg['budget']
        and b['CPU_environment'] == EXECUTION['environment'] and b['future_output'] == OUTPUT
        and b['future_artifacts'] == ARTIFACTS, 'execution binding missing or changed')
    refs = b['dependencies']; paths = [p.resolve(r['path']).resolve() for r in refs]
    p.require(len(paths) == len(set(paths)) and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT, DEFAULT_CONFIG}).issubset(paths), 'future implementation roles not bound')
    p.require(all(ref in refs for ref in p.load_pin(PARENTS['gradient_binding'])['dependencies']), 'parent710 closure missing')
    for ref in refs:
        path = p.resolve(ref['path'])
        p.require(path.suffix in ('.py','.md','.json','.toml') and p.pin(path) == ref, 'bound code changed or payload in closure')
    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true'); parser.add_argument('--require-binding', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        with DEFAULT_CONFIG.open('x') as f: json.dump(create(), f, ensure_ascii=False, indent=2); f.write('\n')
        result = {'status': 'scientific_frozen_no_payload_or_updates'}
    else:
        validate(); result = {'status': 'metadata_valid_no_payload_or_updates'}
        if args.require_binding: require_binding(); result['status'] = 'bound_no_payload'
    print(json.dumps(result))
