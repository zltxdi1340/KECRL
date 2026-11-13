"""Scientific freeze only: fit-only stationarity of existing 500-update heads.

Stdlib metadata and scalar specifications. No research NPZ/CSV read, numerical
endpoint evaluation, optimizer construction, or execution binding here.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_source_expert_late_translation_protocol as p

ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-endpoint-gradient-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-endpoint-gradient-execution-binding_v1.json'
OUTPUT = '/tmp/libero-task7-endpoint-gradient-20261109-v1.json'
ARTIFACTS = '/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v1'
RATIONALE = ROOT / 'docs/libero-task7-direction-harm-alignment-results-2026-11-09.md'
PLAN = ROOT / 'docs/libero-task7-endpoint-gradient-plan-2026-11-09.md'
TESTS = ROOT / 'tests/test_libero_task7_endpoint_gradient_protocol.py'
PARENTS = {
    'alignment_result': {'path': '/tmp/libero-task7-direction-harm-alignment-20261109-v1.json', 'sha256': '7c943714948dec6cc35908b52a08a31facaf9676e84995d9210e611e563ec6f9'},
    'alignment_audit': {'path': '/tmp/libero-task7-direction-harm-alignment-artifacts-20261109-v1/independent-audit.json', 'sha256': '1fdee08a097a248d291244428f568d7ee84d7dfc8ffd2143b7edd486e9396534'},
    'alignment_completion': {'path': '/tmp/libero-task7-direction-harm-alignment-artifacts-20261109-v1/completion.json', 'sha256': '27f061fe0b89de1d3e22290bc2f42b6e375fc71e1db9af22fc4867fab8c91c82'},
    'previous_binding': {'path': str(ROOT / 'configs/libero-task7-direction-harm-alignment-execution-binding_v1.json'), 'sha256': '7d5442063141051c2fecae5f5021d1bfd457a17ac1cf83df129c9fd0da7d7d21'},
    'training_science': {'path': str(ROOT / 'configs/libero-task7-residual-harm-protection-diagnostic_v1.json'), 'sha256': 'aabfd4a198a0e59e96394ca788dc1bf0ab9c4612afb8e7d57eac2880531955b5'},
    'training_result': {'path': '/tmp/libero-task7-residual-harm-protection-20261108-v1.json', 'sha256': 'd08adbdfb31ea391ee84bf3756d675dc6c42763db284c2409308f7fb791d02a7'},
    'training_audit': {'path': '/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/independent-audit.json', 'sha256': '779fc47038da00e7b49f6fd27cad807f1d94cfeba6a023659e176ce8020deacb'},
    'training_completion': {'path': '/tmp/libero-task7-residual-harm-protection-artifacts-20261108-v1/completion.json', 'sha256': '00fd556419f1095b34daad1889f76f5f4eba72c1e0abb915b4693a2cf7c4fa46'},
}
CASES = {'uniform_affine': 0., 'harm_protected_affine': 1.}
POINTS = [0, 500]
SOURCE_MEMBERS = {'shared': ['targets', 'demo_indices', 'action_steps', 'demo_lengths', 'tail20'],
                  'current': ['features', 'predictions']}
HEAD_MEMBERS = ['fit_rows', 'fit_demo_indices', 'scaler_mean', 'scaler_scale', 'updates', 'weights', 'bias', 'optimizer_steps']
OPTIMIZER_PROVENANCE = {'lr': .0003, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': 0.,
    'amsgrad': False, 'maximize': False, 'foreach': False, 'fused': False, 'capturable': False, 'differentiable': False}
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_endpoint_gradient_diagnostic.py',
    'audit': ROOT / 'experiments/audit_libero_task7_endpoint_gradient_diagnostic.py',
    'guard': ROOT / 'experiments/libero_task7_endpoint_gradient_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_endpoint_gradient_diagnostic.py',
    'metadata': Path(__file__).resolve(), 'metadata_tests': TESTS, 'protocol': PLAN,
}
FUTURE_CONTRACT = ROOT / 'docs/libero-task7-endpoint-gradient-execution-contract-2026-11-09.md'
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'original_Policy_and_auxiliary_endpoints': 500, 'future_stabilization_steps': 5,
    'source_role': 'meta_support_demos0..29', 'gradient_role': 'readout_fit_only_all_tge1_of20_whole_demos',
    'encoder_has_seen_all30_support_demos': True, 'encoder_unseen_generalization': False,
    'same_loss_within_each_existing_head': True, 'lambda_or_loss_or_observation_change': False,
    'new_optimizer_steps_parameter_updates_or_trial_steps': 0,
    'new_original_Policy_CNN_generator_or_environment_calls': 0,
    'query_or_probe_numerical_scoring': 0, 'source_meta_query_or_sealed_role_access': False,
    'CSV_HDF5_RGB_proprio_private_state_original_checkpoint_or_historical18head_reads_including_SHA': 0,
    'SPT_Knowledge_Module_or_Policy_changes': False, 'G_failed_H_unrun_and_prior_joint_cues_unchanged': True,
}
DESIGN = {
    'question': 'whether_frozen500_auxiliary_heads_retain_consistent_fit_only_first_order_descent_under_their_own_unchanged_objective',
    'heads': 'all6endpoints_x3whole_demo_folds_x2existing_losses;36heads;257parameters_each',
    'model': 'z=float32((features_fit-float64_saved_mean)/float64_saved_scale);delta=Wz+b;p2=clip(p0_2+delta,-1,1);other6=p0',
    'scaler': 'reconstruct18_fit_only_population_std_scalers;std<=1e-6_scale1;verify_saved_means_scales_abs1e-10;paired_lambda_scalers_bit_exact;transform_using_saved_scaler',
    'points': POINTS, 'point0': 'actual_saved_zero_auxiliary_W_b_reference_only;original_Policy_remains500;not_candidate_checkpoint',
    'snapshot_access': 'weights_and_bias_members_contain0_100_250_500;single_decode_all4;only_positions0_and3_evaluated;no100_250_readout_or_selection',
    'whole_cache_decode': 'shared_current_members_physically_include_aux_probe_rows;only_fit_indices_enter_transform_forward_objective_or_gradient;no_unread_probe_bytes_claim',
    'fit_data': 'all_fit_tge1_rows_including_neutral;no_direction_phase_good_or_event_masks_or_weights;probe_rows_excluded_per_head',
    'fullbatch_measure': 'uniform_empirical_fit_objective_under_original_with_replacement_sampler;not_historical_sample_frequency_weighting_or_Adam_replay',
    'loss': 'F_lambda=C6+mean(e)/7+lambda*mean(relu(e-e0))/7;e=(p2-y2)^2;e0=(p0_2-y2)^2;C6=sum_other6_errors/(7N)',
    'same_loss': 'lambda0_uses_base_directly;lambda1_base_plus_same_excess_penalty;compare0_vs500_within_same_lambda;do_not_rank_cross_lambda_raw_objective_values',
    'loss_floor': 'report_full7D_objective_C6_trainable_part_component3_MSE_penalty_and_trainable_ratio;C6_fixed_across_points_and_losses',
    'precision': 'diagnostic_mathematical_objective_float64_on_saved_float32_weights_and_training_normalized_float32_z;not_replay_of_Float32_Adam',
    'primary': 'NumPy_float64_independent_analytic_full_fit_gradients_base_penalty_total;sorted_fit_rows;no_Torch_or_optimizer',
    'audit': 'fresh_Torch_float64_autograd_of_same_objective;3_grad_calls_per_point;independent_piecewise_one_sided_derivative;no_primary_numerical_import',
    'subgradient': 'clamp_derivative1_including_raw=-1_or1_otherwise0_outside;ReLU_derivative0_at_exact0;diagnostic_selected_subgradient_not_minimum_norm_subgradient',
    'full_gradient': 'Gbase=mean(2*(p2-y2)/7*clampmask*[z,1]);Gpen=mean(same*1[e>e0]*[z,1]);Gtotal=Gbase_if_lambda0_else_Gbase+Gpen',
    'gradient_metrics': 'all257_coordinates_L2_RMS_Linf;G500_L2/G0_L2;Gbase_Gpen_cosine_and_dot;mean_individual_gradient_L2_and_cancellation_ratio;no_epsilon_denominator',
    'direction': 'v=-Gtotal/L2_if_nonzero_else_zero;Xv_is_linear_directional_form_only;no_theta_plus_step_v_prediction_or_parameter_update',
    'clip_directional': 'raw_inside:draw;raw_outside:0;raw=-1:max(draw,0);raw=1:min(draw,0)',
    'hinge_directional': 'de=2*(p2-y2)*dp;e>e0:de;e<e0:0;e=e0:max(de,0);Dplus=mean(de+lambda*dH)/7',
    'non_smooth_limit': 'nonzero_autograd_subgradient_can_have_nonnegative_Dplus_at_hinge;small_gradient_or_nonnegative_selected_direction_does_not_prove_adequate_optimization',
    'normalization': 'report_Dplus500_and_minus_Dplus500/G0_L2;zero_G0_or_zero_initial_trainable_objective_yields_null_reason_not_epsilon_substitution',
    'remaining_cue': 'numeric_valid_and_G0_L2>0_and_Dplus500<-1e-10_and_minus_Dplus500/G0_L2>=.1;fixed10percent_initial_scale_screen_not_universal_optimality_threshold',
    'head_status': 'remaining_first_order_cue;weak_descent_in_direction;no_descent_in_direction;zero_gradient;reference_zero;never_label_optimized',
    'aggregation': 'all3fold_raw_values_per_seed_arm_loss;equal_fold_descriptive_mean;all3seed_values_per_arm_loss;cue_requires_all9head_points_pass;folds_not_independent_seeds',
    'anchor': 'canonical_float64_readout_rounded_to_float32_actions_without_differentiating_rounding;report_loss_and_active_harm_disagreements_vs_float64;not_original_Float32_training_graph',
    'branch_evidence': 'clip5way_and_hinge3way_counts_and_digest_on_sorted_fit_row_IDs;primary_vs_audit_exact;anchor_disagreements_descriptive_not_silent_relabel',
    'invariants': '18paired_initial_zero_penalty_active_count_and_total_grad_identity;other6_floor_invariant;lambda0_Gtotal_equals_Gbase;Gtotal_decomposition;one_head_only_257parameters',
    'meaning': 'fit_only_first_order_space_in_a_frozen_auxiliary_head;not_global_optimality_Float32_Adam_convergence_query_improvement_or_closed_loop_gain',
    'decision': 'remaining_cue_only_motivates_separately_frozen_optimization_check;absence_keeps_question_open;no_new_training_or_original_cue_change',
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU',
    'runtime': {'python': '3.10.22', 'numpy': '1.22.4', 'torch': '2.4.1+cu124'},
    'environment': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    'torch_deterministic_algorithms': True, 'streaming': 'one_model_current_cache_and_one_head_at_a_time;no36_graph_retention',
    'main_seconds_max': 180, 'audit_seconds_max': 180, 'overall_seconds_max': 420,
    'RSS_bytes_max': 536870912, 'output_CSV_bytes_max': 67108864, 'new_output_NPZ': 0,
    'future_reads': 'each_stage7sourceNPZ17members_plus36heads8members_each_once;no_predictions_probe_rows_samples_traces_or_Adam_state_member',
    'byte_SHA': 'all43archives_once_before_load_and_once_after_stage;no_source_archive_SHA_at_scientific_freeze',
    'role_guard': 'fit_only_transform_and_registered_head_forward_gradient;deny_parameter_writes_optimizer_Module_calls_and_original_model_imports',
    'audit_scope': 'source_roles_scalers_weights_own_loss_all_gradient_coordinates_Dplus_ratios_branches_cues_and_all_output_tables',
    'float_absolute_tolerance': 1e-10, 'relative_tolerance': 0, 'IDs_shapes_members_and_branch_digest_tolerance': 0,
    'anchor_action_rounding_absolute_tolerance': 1e-7, 'anchor_objective_absolute_tolerance': 1e-6,
    'threshold_disagreement': 'technical_stop_no_relabeling_or_threshold_relaxation',
    'technical_retries': 0, 'overwrite': False,
    'stop': 'any_provenance_endpoint_field_role_budget_nonfinite_loss_gradient_directional_or_branch_audit_mismatch_keep_partial_stop;otherwise_end_fixed_diagnostic',
}


def budget():
    return {'original_endpoints': 6, 'folds': 3, 'heads': 36, 'parameters_per_head': 257, 'points_per_head': 2,
        'fit_rows_across18_model_folds': 50256, 'scaler_reconstructions_per_stage': 18,
        'fit_standardization_feature_coordinates_per_stage': 12865536,
        'point_jobs_per_stage': 72, 'fit_row_point_visits_per_stage': 201024,
        'seven_action_coordinates_per_stage': 1407168,
        'parameter_and_direction_linear_forms_per_stage': 144, 'linear_form_row_visits_per_stage': 402048,
        'linear_form_feature_multiplications_per_stage': 102924288,
        'gradient_vectors_per_stage': 216, 'gradient_coordinate_values_per_stage': 55512,
        'direction_coordinate_values_per_stage': 18504, 'audit_autograd_calls': 216,
        'paired_initial_parity_records_per_stage': 18,
        'source_NPZ_per_stage': 7, 'source_members_per_stage': 17,
        'head_NPZ_per_stage': 36, 'head_members_per_stage': 288,
        'total_NPZ_loads': 86, 'total_member_decodes': 610, 'archive_SHA_checks_per_stage': 86,
        'total_archive_SHA_checks': 172, 'input_CSV_passes': 0,
        'tables': {'points': 72, 'parameters': 18504, 'pairs': 36, 'initial-parity': 18},
        'total_output_CSV_rows': 18630, 'seed_summary_records': 12, 'arm_loss_summary_records': 4,
        'new_optimizer_steps_parameter_updates_trial_steps_Policy_CNN_environment_or_sealed_calls': 0}


def scalar_terms(y, p0, raw, coefficient, other6_sse=0.):
    """Per-frame mathematical objective; only used for specification/toy tests."""
    p.require(all(type(v) in (int, float) and math.isfinite(v) for v in (y, p0, raw, coefficient, other6_sse))
              and abs(y) <= 1 and abs(p0) <= 1 and coefficient in (0., 1.) and 0 <= other6_sse <= 24, 'invalid scalar objective')
    pred = min(1., max(-1., raw)); e = (pred - y) ** 2; e0 = (p0 - y) ** 2
    penalty = max(e - e0, 0.) / 7; floor = other6_sse / 7; trainable = e / 7 + coefficient * penalty
    return {'prediction': pred, 'component3_mse': e, 'original_component3_mse': e0, 'fixed_other6_floor': floor,
        'base_seven_dimensional_mse': floor + e / 7, 'penalty': penalty,
        'trainable_objective': trainable, 'full_objective': floor + trainable}


def scalar_subgradient(y, p0, raw, coefficient):
    terms = scalar_terms(y, p0, raw, coefficient)
    base = 2 * (terms['prediction'] - y) / 7 if -1 <= raw <= 1 else 0.
    penalty = base if terms['component3_mse'] > terms['original_component3_mse'] else 0.
    return {'base': base, 'penalty': penalty, 'total': base if coefficient == 0 else base + penalty}


def scalar_directional_derivative(y, p0, raw, draw, coefficient):
    terms = scalar_terms(y, p0, raw, coefficient)
    p.require(type(draw) in (int, float) and math.isfinite(draw), 'finite tangent required')
    tangent = draw if -1 < raw < 1 else max(draw, 0.) if raw == -1 else min(draw, 0.) if raw == 1 else 0.
    de = 2 * (terms['prediction'] - y) * tangent
    diff = terms['component3_mse'] - terms['original_component3_mse']
    dh = de if diff > 0 else max(de, 0.) if diff == 0 else 0.
    result = (de + coefficient * dh) / 7
    p.require(math.isfinite(result), 'nonfinite directional derivative')
    return result


def remaining_evidence(initial_norm, endpoint_norm, endpoint_directional, *, numeric_valid=True):
    p.require(all(type(v) in (int, float) and math.isfinite(v) for v in (initial_norm, endpoint_norm, endpoint_directional))
        and initial_norm >= 0 and endpoint_norm >= 0 and type(numeric_valid) is bool, 'invalid descent evidence')
    p.require(numeric_valid, 'numerical gate failed')
    p.require(endpoint_norm != 0 or endpoint_directional == 0, 'zero gradient direction must be zero')
    if initial_norm == 0: return {'status': 'reference_zero', 'relative_gradient': None, 'normalized_descent': None, 'remaining_cue': False}
    g = endpoint_norm / initial_norm; slope = -endpoint_directional / initial_norm
    p.require(math.isfinite(g) and math.isfinite(slope), 'nonfinite normalized evidence')
    if endpoint_norm == 0:
        p.require(endpoint_directional == 0, 'zero gradient direction must be zero'); status = 'zero_gradient'
    elif endpoint_directional < -1e-10: status = 'remaining_first_order_cue' if slope >= .1 else 'weak_descent_in_direction'
    else: status = 'no_descent_in_direction'
    return {'status': status, 'relative_gradient': g, 'normalized_descent': slope, 'remaining_cue': status == 'remaining_first_order_cue'}


def joint_remaining_evidence(records):
    """Conservative all-nine coverage; this is not a Policy acceptance gate."""
    expected = {(seed, fold) for seed in p.SEEDS for fold in range(3)}
    p.require(len(records) == 9 and {(r['seed'], r['fold']) for r in records} == expected
              and all(type(r['remaining_cue']) is bool for r in records), 'complete nine-head boolean evidence required')
    return all(r['remaining_cue'] for r in records)


def select_inputs(parents):
    source = parents['training_science']['inputs']; recorded = parents['training_result']
    models = [{k: m[k] for k in ('task', 'seed', 'arm', 'checkpoint', 'current_archive')} for m in source['models']]
    endpoints = [{k: m[k] for k in ('task', 'seed', 'arm', 'checkpoint')} for m in models]
    p.require(endpoints == parents['alignment_result']['inputs']['endpoints'], 'current original endpoint lineage differs')
    for key in ('alignment_audit', 'training_audit'): p.require(parents[key]['status'].startswith('passed_'), 'parent audit incomplete')
    for key in ('alignment_completion', 'training_completion'): p.require(parents[key]['status'].startswith('complete_independently_audited'), 'parent run incomplete')
    p.require(recorded['science_config'] == PARENTS['training_science'] and parents['training_completion']['result'] == PARENTS['training_result']
        and parents['training_completion']['audit'] == PARENTS['training_audit'], 'training provenance differs')
    heads = recorded['heads']; expected = [(7, m['seed'], m['arm'], f, mode) for m in models for f in range(3) for mode in CASES]
    p.require([(r['task'], r['seed'], r['arm'], r['fold'], r['mode']) for r in heads] == expected, 'all36 fixed heads/order required')
    for h in heads:
        model = next(m for m in models if (m['seed'], m['arm']) == (h['seed'], h['arm'])); t = h['training']
        p.require(h['checkpoint'] == model['checkpoint'] and t['key'] == [h['seed'], h['arm'], h['fold'], h['mode']]
            and type(t['updates']) is int and t['updates'] == 500 and t['parameters'] == 257 and t['actual_state_steps'] == [500, 500]
            and t['actual_optimizer'] == OPTIMIZER_PROVENANCE and t['excess_error_penalty_lambda'] == CASES[h['mode']]
            and t['readout_seed'] == h['seed'] + 100000 * h['fold'] and t['all_finite'] is True, 'saved training endpoint/settings differ')
    refs = [source['shared']] + [m['current_archive'] for m in models] + [h['archive'] for h in heads]
    p.require(len(refs) == len({r['path'] for r in refs}) == 43 and all(Path(r['path']).suffix == '.npz' and len(r['sha256']) == 64 for r in refs), '43 distinct pinned archives required')
    return {'shared': source['shared'], 'models': models, 'heads': heads,
            'action_lengths': source['action_lengths'], 'folds': source['folds']}


def create():
    parents = {name: p.load_pin(ref) for name, ref in PARENTS.items()}; inputs = select_inputs(parents)
    p.require(sum(f['fit_tge1_rows'] for f in inputs['folds']) == 8376 and len(inputs['folds']) == 3, 'canonical fit roles differ')
    return copy.deepcopy({'kind': 'task7_frozen500_fit_endpoint_gradient_diagnostic', 'version': 1, 'formal_result': False,
        'status': 'scientific_freeze_only_execution_binding_required', 'parents': PARENTS, 'scope': SCOPE, 'design': DESIGN,
        'execution': EXECUTION, 'source_members': SOURCE_MEMBERS, 'head_members': HEAD_MEMBERS,
        'budget': budget(), 'inputs': inputs, 'code': {'metadata': p.pin(Path(__file__).resolve()), 'tests': p.pin(TESTS)},
        'plan': p.pin(PLAN), 'rationale': p.pin(RATIONALE), 'future_files': {k: str(v) for k, v in FUTURE_FILES.items()},
        'future_contract': str(FUTURE_CONTRACT), 'output': OUTPUT, 'artifacts': ARTIFACTS})


def validate(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(json.dumps(cfg, sort_keys=True, allow_nan=False) == json.dumps(create(), sort_keys=True, allow_nan=False), 'frozen endpoint-gradient protocol/provenance differs')
    return cfg


def require_binding(config=DEFAULT_CONFIG, binding=DEFAULT_BINDING):
    cfg = validate(config); b = json.loads(Path(binding).read_text())
    p.require(b['status'] == 'bound_before_payload' and b['before_research_payload'] is True and b['science_config'] == p.pin(config)
        and b['budget'] == cfg['budget'] and b['CPU_environment'] == EXECUTION['environment'] and b['future_output'] == OUTPUT
        and b['future_artifacts'] == ARTIFACTS, 'execution binding required or changed')
    refs = b['dependencies']; paths = [p.resolve(r['path']).resolve() for r in refs]
    p.require(len(paths) == len(set(paths)) and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT}).issubset(paths), 'future implementation/contract not bound')
    inherited = p.load_pin(PARENTS['previous_binding'])['dependencies']; p.require(all(r in refs for r in inherited), 'parent code closure missing')
    for ref in refs:
        path = p.resolve(ref['path']); p.require(path.suffix in ('.py', '.md', '.json', '.toml'), 'payload in code closure')
        p.require(p.pin(path) == ref, 'bound code changed')
    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--write-freeze', action='store_true'); parser.add_argument('--require-binding', action='store_true'); args = parser.parse_args()
    if args.write_freeze:
        with DEFAULT_CONFIG.open('x') as f: json.dump(create(), f, ensure_ascii=False, indent=2); f.write('\n')
        result = {'status': 'scientific_frozen_no_research_payload'}
    else:
        validate(); result = {'status': 'metadata_valid_no_research_payload'}
        if args.require_binding: require_binding(); result['status'] = 'bound_no_payload'
    print(json.dumps(result))
