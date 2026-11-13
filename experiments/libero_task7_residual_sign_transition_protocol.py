"""Metadata-only freeze for final500 residual action sign/magnitude decomposition.

Real input CSV data passes require a future execution binding. No array/archive,
Torch, Policy, optimizer or environment access belongs to this protocol.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

from experiments import libero_task7_current_residual_correctability_protocol as previous

p = previous.p
exact = previous.previous.previous.old.exact
ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-residual-sign-transition-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-residual-sign-transition-execution-binding_v1.json'
OUTPUT = '/tmp/libero-task7-residual-sign-transition-20261108-v1.json'
ARTIFACTS = '/tmp/libero-task7-residual-sign-transition-artifacts-20261108-v1'
PARENTS = {
    'science': {'path': str(previous.DEFAULT_CONFIG), 'sha256': '722b626c34f7faff67356b3d7e385784536aad4b95abf80084f685079855c58e'},
    'binding': {'path': str(previous.DEFAULT_BINDING), 'sha256': '634889184ab14dcea00fbf7f104b7e0026e90dc9b5db3c13c6ea51130fd3c18c'},
    'result': {'path': previous.OUTPUT, 'sha256': '3dede2f27f67135c2ac6a54e1402b79869b330c562f7bc10b35aab0e7ffc5973'},
    'audit': {'path': str(Path(previous.ARTIFACTS) / 'independent-audit.json'), 'sha256': 'c932f93555f518b194732e3ac2052456454a120a5334007aa1d51841b13b09c1'},
    'completion': {'path': str(Path(previous.ARTIFACTS) / 'completion.json'), 'sha256': '042a426495aebc53496bf9ebcb2d4e3500222c650a1caa5eb0f247f7eb9affd0'},
}
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'readout_modes': ['original', 'bias_only', 'current_affine'], 'original_Policy_updates': 500, 'readout_endpoint_updates': 500,
    'closed_loop_default_stabilization_steps': 5, 'loss': 'uniform_seven_dimensional_valid_step_masked_action_MSE',
    'role': 'source_meta_support_exposed_representation_OOF_readout_diagnostic', 'demonstration_indices': list(range(30)),
    'encoder_has_seen_all30_support_demos': True, 'causal_alignment': 'obs[t-1] -> action[t],t>=1',
    'pair_first_action_step': 5, 'component_index': 2, 'target_material_margin_inclusive': .025,
    'new_training_or_optimizer_updates': 0, 'new_forward_readout_CNN_Policy_or_environment_calls': 0,
    'NPZ_NPY_checkpoint_HDF5_RGB_feature_public_proprio_or_private_state_reads_including_byte_hashes': 0,
    'old_NN_or_history_tag_CSV_reads': 0, 'checkpoint_fold_or_seed_selection': False,
    'source_meta_query_accessed': False, 'sealed_roles_accessed': False,
    'parent_all4_joint_cues_false_unchanged': True, 'G_failed_H_unrun_unchanged': True,
    'SPT_Knowledge_Module_changes': False, 'automatic_training_or_rollout_promotion': False,
}
MODES = SCOPE['readout_modes']
COMPARISONS = [['bias_only', 'original'], ['current_affine', 'original'], ['current_affine', 'bias_only']]
PHASES = ['all_common', 'tail20', 'complement']
DIRECTIONS = ['negative', 'positive']
TRANSITIONS = ['stays_correct', 'corrected', 'new_error', 'stays_wrong']
BANDS = ['zero', 'nonzero_abs_lt.025', 'abs_ge.025']
CLASSES = ['wrong_sign', 'prediction_zero', 'same_sign_under', 'same_sign_over', 'exact_match', 'zero_target_nonzero']
FRAME_COLUMNS = ['task', 'seed', 'arm', 'mode', 'update', 'source_row', 'probe_fold', 'demo', 'action_step',
                 'target_direction', 'tail20', 'component3_category', 'raw_residual', 'clip', 'target_2', 'prediction_2', 'error_2']
CURVE_ID_COLUMNS = ['task', 'seed', 'arm', 'fold', 'mode', 'update', 'role', 'phase', 'direction']
CURVE_METRICS = ['n', 'material_n', 'component3_mse', 'component3_sse', 'component3_mae', 'component3_bias',
                 'material_sign_error_n', 'material_sign_error_rate', 'clip_n', 'clip_fraction',
                 'raw_residual_mean', 'raw_residual_abs_mean', 'raw_residual_min', 'raw_residual_max']
DESIGN = {
    'question': 'net_sign_error_change_corrected_vs_new_error;reference_output_margin_and_raw_vs_effective_residual;cross_seed_and_bias_affine_overlap',
    'primary_contrast_focus': 'positive_corrected_and_negative_new_error;keep_both_directions_all4_cells_and_all3_comparisons',
    'action_transition_not_NN': 'strict_sign_of_saved_continuous_outputs_vs_expert_material_target;not_previous_NN_tag_or_episode_success_transition',
    'comparisons_candidate_reference': COMPARISONS, 'phases': PHASES, 'directions': DIRECTIONS,
    'transitions': TRANSITIONS, 'reference_and_candidate_output_bands': BANDS, 'strict_error_classes': CLASSES,
    'input_frames': 'one_lexical_pass75384rows_project17columns;all_raw_tge1_IDs_and_three_modes_required',
    'input_curves': 'one_lexical_pass4608rows_project9IDcolumns_allrows;project14metrics_only_update500_AND_roleprobe_576rows',
    'curve_filter': 'no_fit_numeric_metrics_no_update0_100_250_numeric_metrics;ID_coverage_all4608_checked',
    'join': 'one_to_one_task_seed_arm_mode_source_row;rebuild_demo_step_probe_fold_tail_from30lengths;target_and_IDs_match_all_modes_and_seeds',
    'scalar_parity': 'target_prediction_finite_range_float32_exact;error=prediction-target;category_target_direction_clip_match;original_residual0',
    'prediction_reconstruction': 'float32(clip(original_prediction_2+saved_raw_residual,-1,1))equals_saved_prediction;no_weight_or_feature_forward',
    'bias_only_parity': 'raw_residual_is_one_float32_constant_for_each_endpoint_probe_fold;no_inference_of_current_affine_bias',
    'pair_material': 'tge5_abs_target2_ge.025_3598shared;neutral_and_first4_frames_read_for_ID_and_probe_parity_only',
    'transition_rule': 'correct_iff_target_sign*prediction>0;zero_wrong_for_material;four_cells_complete_and_exclusive',
    'conservation': 'candidate_error_n-reference_error_n=new_error_n-corrected_n;four_counts_sum_class_n;SSE_delta_sum_conserves',
    'signed_zero_cells': 'report_ref_and_candidate_signs_minus_zero_plus_9counts_and_separate_zero_counts_in_each_transition',
    'margins': 'signed_margin=target_sign*prediction;reference_and_candidate_abs0_distance;fixed.025_bands_no_search',
    'raw_vs_effective': 'raw_shift=residual_candidate-residual_reference;effective_shift=prediction_candidate-prediction_reference;clip_or_rounding_gap=effective-raw',
    'gap_limit': 'gap_includes_clamp_AND_saved_float32_rounding;report_ref_candidate_clip_flags_separately;nonzero_gap_does_not_prove_clipping',
    'ratio': 'abs(effective_shift)/abs(reference_prediction);raw_shift_ratio_also;denominator0_null_with_count_no_epsilon',
    'crossing': 'new_error_refmargin>0_candmargin_le0;corrected_refmargin_le0_candmargin>0;ratio_ge1_alone_does_not_define_harm',
    'same_frame_errors': 'ref_candidate_component2_SSE_MSE_MAE_bias_target_energy_relative_zero_error;delta_SSE_lower_tie_higher_per_transition',
    'descriptions': 'target_abs_ref_abs_candidate_abs_raw_residuals_raw_shift_effective_shift_clip_or_rounding_gap_signed_margins_and_ratios;n_mean_median_q25_q75_min_max_float64_linear',
    'fixed_transition_references': [['corrected', 'stays_wrong'], ['new_error', 'stays_correct']],
    'references_limit': 'match_reference_correctness_and_target_sign;report_target_scale_and_group_n;not_causal_or_same_sample_group_effect',
    'demo': 'retain_all30_empty_groups;paired_demo_delta_only_both_nonempty;equal_demo_mean_and_higher_tie_lower_null_counts',
    'cross_seed': 'per_arm_comparison_same3598frames;keep_all3_transitions_and_ref_error_candidate_error_and4transition_votes0to3;baseline_common_errors_separate',
    'bias_affine_overlap': 'same_endpoint_frame_original_to_bias_transition_by_original_to_affine_transition4x4;not_affine_Wz_vs_bias_parameter_decomposition',
    'parity': 'recompute14metrics_for_all576_final500probe_curve_rows_from_mode_OOF_frames;no_fit_or_earlier_checkpoint_prediction_reconstruction',
    'empties': 'n_counts_SSE0;rates_MSE_quantiles_null;zero_energy_ratios_null;never_drop_seed_demo_or_cell',
    'aggregation': 'all3_seeds_per_A_B_source_arm_raw_values_and_complete_mean;no_pvalues_significance_or_independent_frame_assumption',
    'interpretation': 'offline_error_association_and_output_shift_only;not_missing_history_visual_head_root_cause_physical_phase_worldz_or_closed_loop_gain',
    'data_limit': 'CSV_has_no_W_b_features_or_observations;cannot_separate_current_affine_internal_Wz_and_b_or_retrain_or_restore_G_H',
    'decision': 'all_parent_cues_false_stay_false;no_new_acceptance_threshold_or_candidate_selection;describe_evidence_and_counterexamples_only',
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU',
    'runtime': {'python': '3.10.22', 'numpy': '1.22.4'}, 'environment': previous.EXECUTION['environment'],
    'analysis_active_seconds_max': 180, 'audit_active_seconds_max': 180, 'overall_watchdog_seconds': 420,
    'working_memory_bytes_max': 536870912, 'memory_monitor': 'current_/proc/self/status_VmHWM_VmRSS',
    'output_CSV_bytes_max': 268435456, 'technical_retries': 0, 'output_overwrite': False,
    'access': 'two_fixed_CSV_only_one_data_pass_each_per_stage;conditional_projected_columns;deny_raw_unlisted_CSV_or_all_archives_and_alternate_readers',
    'metadata': 'five_fixed_parent_JSON_for_provenance_only;no_parent_metric_payload_used_as_new_analysis_input',
    'byte_SHA': 'two_input_CSV_before_after_separate_from_data_passes;never_hash_NPZ_or_checkpoint',
    'audit': 'separate_process_independent_numerical_reconstruction_no_primary_analysis_import;repeat_same2CSV_passes_projection_and_all_output_checks',
    'float_absolute_tolerance': 1e-10, 'float_relative_tolerance': 0, 'integer_ID_category_and_f32_prediction_tolerance': 0,
    'stop': 'any_hash_provenance_role_ID_header_duplicate_missing_projection_scalar_curve_parity_budget_or_audit_mismatch_keep_partial_stop_no_retry',
}
BINDING_FILES = {
    'analysis': ROOT / 'experiments/analyze_libero_task7_residual_sign_transition.py',
    'audit': ROOT / 'experiments/audit_libero_task7_residual_sign_transition.py',
    'guard': ROOT / 'experiments/libero_task7_residual_sign_transition_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_residual_sign_transition.py',
    'metadata': Path(__file__).resolve(),
    'metadata_tests': ROOT / 'tests/test_libero_task7_residual_sign_transition_protocol.py',
    'protocol': ROOT / 'docs/libero-task7-residual-sign-transition-plan-2026-11-08.md',
}
CONTEXT_FILES = [BINDING_FILES['metadata'], BINDING_FILES['metadata_tests'], Path(previous.__file__).resolve(),
                 Path(p.__file__).resolve(), ROOT / 'experiments/run_libero_task7_current_residual_correctability.py']


def budget():
    return {'source_endpoints': 6, 'modes': 3, 'comparisons': 3, 'raw_source_rows_shared': 4188,
            'raw_mode_frame_rows': 75384, 'material_common_rows_shared': 3598, 'pair_frame_rows': 64764,
            'cross_seed_frame_rows': 21588, 'phases': 3, 'directions': 2, 'demos': 30,
            'per_stage': {'CSV_data_passes': 2, 'frames_lexical_rows': 75384, 'curve_lexical_rows': 4608,
                          'CSV_lexical_rows': 79992, 'frame_projected_columns': 17, 'frame_projected_cells': 1281528,
                          'curve_ID_columns': 9, 'curve_ID_projected_cells': 41472,
                          'selected_curve_numeric_rows': 576, 'selected_curve_metric_columns': 14,
                          'selected_curve_metric_cells': 8064, 'projected_cells': 1331064},
            'total_CSV_data_passes': 4, 'total_CSV_lexical_rows': 159984, 'total_projected_cells': 2662128,
            'probe_curve_scalar_parity_checks_per_stage': 8064,
            'tables': {'paired-frames': 64764, 'model-transitions': 432, 'demo-transitions': 12960,
                       'margin-cells': 3888, 'error-cells': 3888, 'contrasts': 216, 'demo-contrasts': 6480,
                       'demo-standardized': 216, 'crossseed-frames': 21588, 'crossseed-summary': 144,
                       'head-overlap': 576, 'probe-parity': 576},
            'seed_summary_JSON_rows': 72, 'NPZ_NPY_checkpoint_feature_or_other_array_reads_including_hashes': 0,
            'new_training_or_forward_environment_neighbor_or_sealed_calls': 0}


def source_layout(lengths):
    p.require(len(lengths) == 30 and all(type(n) is int and n >= 25 for n in lengths)
              and sum(n - 1 for n in lengths) == 4188 and sum(n - 5 for n in lengths) == 4068, 'source causal lengths differ')
    return [{'source_row': i, 'demo': demo, 'action_step': step, 'probe_fold': demo // 10,
             'tail20': step >= lengths[demo] - 20}
            for i, (demo, step) in enumerate((d, t) for d, n in enumerate(lengths) for t in range(1, n))]


def finite_number(value): return type(value) in (int, float) and math.isfinite(value)


def float32(value):
    p.require(finite_number(value), 'finite numerical scalar required')
    return struct.unpack('<f', struct.pack('<f', value))[0]


def error_category(target, prediction):
    p.require(all(finite_number(v) and abs(v) <= 1 for v in (target, prediction)), 'target/action scalar invalid')
    if prediction == target: return 'exact_match'
    if target == 0: return 'zero_target_nonzero'
    if prediction == 0: return 'prediction_zero'
    if target * prediction < 0: return 'wrong_sign'
    return 'same_sign_under' if abs(prediction) < abs(target) else 'same_sign_over'


def output_band(value):
    p.require(finite_number(value) and abs(value) <= 1, 'invalid action margin')
    return 'zero' if value == 0 else ('nonzero_abs_lt.025' if abs(value) < .025 else 'abs_ge.025')


def pair_quantities(target, reference, candidate, raw_reference, raw_candidate):
    p.require(all(finite_number(v) for v in (target, reference, candidate, raw_reference, raw_candidate))
              and abs(target) >= .025 and all(abs(v) <= 1 for v in (target, reference, candidate)), 'material pair scalars required')
    sign = 1 if target > 0 else -1; good_ref = sign * reference > 0; good_cand = sign * candidate > 0
    transition = {(True, True): 'stays_correct', (False, True): 'corrected', (True, False): 'new_error', (False, False): 'stays_wrong'}[good_ref, good_cand]
    raw = raw_candidate - raw_reference; actual = candidate - reference
    return {'transition': transition, 'reference_error': not good_ref, 'candidate_error': not good_cand,
            'reference_zero': reference == 0, 'candidate_zero': candidate == 0,
            'reference_band': output_band(reference), 'candidate_band': output_band(candidate),
            'signed_reference_margin': sign * reference, 'signed_candidate_margin': sign * candidate,
            'raw_shift': raw, 'effective_shift': actual, 'signed_effective_shift': sign * actual,
            'clip_or_rounding_gap': actual - raw, 'effective_shift_over_abs_reference': abs(actual) / abs(reference) if reference else None,
            'raw_shift_over_abs_reference': abs(raw) / abs(reference) if reference else None,
            'delta_sse': (candidate - target) ** 2 - (reference - target) ** 2}


def validate_action_scalar(target, prediction, raw_residual, original, saved_error, clipped, mode):
    p.require(mode in MODES and type(clipped) is bool and all(finite_number(v) for v in (target, prediction, raw_residual, original, saved_error)), 'scalar role/dtype invalid')
    p.require(all(abs(v) <= 1 and float32(v) == v for v in (target, prediction, original)), 'saved float32 target/action required')
    p.require(abs(saved_error - (prediction - target)) <= 1e-10, 'saved target error mismatch')
    value = original + raw_residual
    p.require(float32(min(1., max(-1., value))) == prediction and clipped == (abs(value) > 1), 'saved residual/clip reconstruction mismatch')
    if mode == 'original': p.require(raw_residual == 0 and prediction == original and clipped is False, 'original identity mismatch')
    return error_category(target, prediction)


def curve_metric_allowed(update, role):
    p.require(type(update) is int and update in (0, 100, 250, 500) and role in ('fit', 'probe'), 'curve ID role invalid')
    return update == 500 and role == 'probe'


def select_inputs(parents):
    science, result = parents['science'], parents['result']; endpoints = []
    for model in science['inputs']['models']:
        endpoints.append({key: model[key] for key in ('task', 'seed', 'arm', 'checkpoint')})
    expected = {(7, seed, arm) for seed in p.SEEDS for arm in p.ARMS}
    p.require(len(endpoints) == 6 and {(m['task'], m['seed'], m['arm']) for m in endpoints} == expected, 'all six fixed endpoints required')
    refs = {key: result['tables'][name] for key, name in [('frames', 'oof-frames'), ('curves', 'curves')]}
    for key, filename, rows, columns in [('frames', 'oof-frames.csv', 75384, FRAME_COLUMNS),
                                         ('curves', 'curves.csv', 4608, CURVE_ID_COLUMNS + CURVE_METRICS)]:
        ref = refs[key]
        p.require(ref['path'] == str(Path(previous.ARTIFACTS) / filename) and type(ref['rows']) is int and ref['rows'] == rows
                  and len(set(ref['columns'])) == len(ref['columns']) and set(columns).issubset(ref['columns']), 'approved CSV identity/schema differs')
    lengths = science['inputs']['action_lengths']; source_layout(lengths)
    return {'tables': refs, 'action_lengths': lengths, 'endpoints': endpoints}


def validate_design(cfg, parents):
    p.require(cfg['kind'] == 'task7_cached_residual_action_sign_transition_readonly' and type(cfg['version']) is int and cfg['version'] == 1
              and cfg['formal_result'] is False and cfg['status'] == 'scientific_frozen_execution_unbound'
              and exact(cfg['parents'], PARENTS), 'scientific identity/parents differ')
    for key, value in [('scope', SCOPE), ('design', DESIGN), ('execution', EXECUTION), ('budget', budget()),
                       ('frame_columns', FRAME_COLUMNS), ('curve_ID_columns', CURVE_ID_COLUMNS), ('curve_metric_columns', CURVE_METRICS)]:
        p.require(exact(cfg[key], value), 'frozen ' + key + ' differs')
    p.require(all(row['formal_result'] is False for row in parents.values()), 'diagnostic parents required')
    result, audit, done, science = (parents[key] for key in ('result', 'audit', 'completion', 'science'))
    p.require(result['status'] == 'complete_task7_current_residual_pending_audit'
              and audit['status'] == 'passed_independent_task7_current_residual_audit'
              and done['status'] == 'complete_independently_audited_task7_current_residual'
              and result['science_config'] == audit['science_config'] == done['science_config'] == PARENTS['science']
              and result['binding'] == audit['binding'] == done['binding'] == PARENTS['binding']
              and audit['result'] == done['result'] == PARENTS['result'] and done['audit'] == PARENTS['audit']
              and result['inputs'] == science['inputs'], 'audited current-residual parent required')
    p.require(len(audit['model_flags']) == 18 and len(audit['arm_flags']) == 4
              and all(f['cue'] is False and f['flags']['numerical_sign_stable'] is True for f in audit['model_flags'])
              and all(f['joint_cue'] is False and f['automatic_promotion'] is False for f in audit['arm_flags'])
              and done['arm_flags'] == audit['arm_flags'] and result['G_failed_H_unrun_unchanged'] is True
              and result['SPT_Knowledge_Module_changes'] is False, 'parent stopped-candidate/numerical audit boundary differs')
    p.require(science['scope']['source_role'] == 'meta_support' and science['scope']['source_meta_query_accessed'] is False
              and science['scope']['sealed_roles_accessed'] is False and science['scope']['encoder_has_seen_all30_support_demos'] is True
              and science['scope']['original_Policy_checkpoint_updates'] == science['design']['endpoint'] == 500, 'source role/endpoint boundary differs')
    p.require(exact(cfg['inputs'], select_inputs(parents)) and cfg['output'] == OUTPUT and cfg['artifacts'] == ARTIFACTS, 'CSV input/projection output identities differ')
    return True


def validate(path=DEFAULT_CONFIG, *, check_csv_byte_hashes=False):
    cfg = json.loads(Path(path).read_text())
    p.require(exact(cfg['parents'], PARENTS), 'immutable metadata parent pins differ')
    parents = {key: p.load_pin(ref) for key, ref in PARENTS.items()}; validate_design(cfg, parents)
    p.require(len(cfg['code']) == len(CONTEXT_FILES) and {p.resolve(ref['path']).resolve() for ref in cfg['code']} == set(CONTEXT_FILES)
              and p.resolve(cfg['protocol']['path']).resolve() == BINDING_FILES['protocol'], 'required context/protocol identity differs')
    for ref in cfg['code'] + [cfg['protocol']]: p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'frozen metadata/protocol changed')
    if check_csv_byte_hashes:
        for ref in cfg['inputs']['tables'].values(): p.require(p.sha(ref['path']) == ref['sha256'], 'approved CSV bytes changed')
    return cfg


def require_binding(path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    cfg = validate(path)
    p.require(Path(binding_path).is_file(), 'execution binding required before real CSV projection; scientific freeze only')
    bound = json.loads(Path(binding_path).read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['formal_result'] is False and bound['before_research_payload'] is True
              and bound['science_config'] == p.pin(path) and exact(bound['budget'], budget())
              and set(bound['files']) == set(BINDING_FILES), 'execution identity/budget differs')
    for role, file in BINDING_FILES.items(): p.require(bound['files'][role] == p.pin(file), 'execution role/code changed')
    return bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-csv-byte-hashes', action='store_true')
    parser.add_argument('--require-executable', action='store_true'); args = parser.parse_args()
    cfg = validate(check_csv_byte_hashes=args.check_csv_byte_hashes)
    if args.require_executable: require_binding()
    print(json.dumps({'status': cfg['status'], 'future_total_CSV_data_passes': 4, 'future_projected_cells': 2662128,
                      'actual_real_CSV_numeric_projection_rows': 0, 'actual_archive_reads_including_byte_hashes': 0,
                      'actual_training_forward_environment_or_sealed_calls': 0}, ensure_ascii=False))
