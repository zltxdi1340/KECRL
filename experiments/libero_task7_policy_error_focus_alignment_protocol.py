"""Metadata-only freeze: align fixed Policy action errors with saved readout tags."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments import libero_task7_lag_neighbor_decomposition_protocol as previous

p = previous.p
ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-policy-error-focus-alignment-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-policy-error-focus-alignment-execution-binding_v1.json'
LAG_DIRECTORY = Path('/tmp/libero-task7-lag-neighbor-decomposition-artifacts-20261108-v1')
PARENTS = {
    'lag_science': {'path': str(previous.DEFAULT_CONFIG), 'sha256': 'ae5e70c20b0835067f06468ee1d076554043c04c8152a72433ce61c7ae107a58'},
    'lag_binding': {'path': str(previous.DEFAULT_BINDING), 'sha256': 'db4f4da78fb6269752ea1e4aa03426c66cbdcc0a013237e0af3f0034fb7dcc43'},
    'lag_result': {'path': previous.OUTPUT, 'sha256': '6dc248edeaf664df7da555a574a14637ee4f99f1ff6a441c674aed54a52c8644'},
    'lag_audit': {'path': str(LAG_DIRECTORY / 'independent-audit.json'), 'sha256': 'e2ed4ce689b921ad88c3e819b73917927e921ab1785751e1e81fb0a9f8c86e1c'},
    'lag_completion': {'path': str(LAG_DIRECTORY / 'completion.json'), 'sha256': '9b4d83ed8b9ff4f0a752da322834a1b7edf84e5598adbcc69fc43c9f41f016fb'},
    'transition_result': previous.PARENTS['transition_result'],
    'transition_audit': previous.PARENTS['transition_audit'],
    'transition_completion': previous.PARENTS['transition_completion'],
    'capture': previous.PARENTS['capture'],
    'legacy_result': {'path': '/tmp/libero-source-expert-late-translation-readout-20261107-v1.json',
                      'sha256': 'b44f498a82d50e6a19a289dfcdcf3bdf308f514ab92f770b3be458890c879b67'},
}
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'arms': p.ARMS, 'checkpoint_updates': 500,
    'role': 'source_meta_support_training_fit_only', 'demonstration_indices': list(range(30)),
    'causal_alignment': 'obs[t-1] -> action[t], t>=1', 'alignment_first_action_step': 5,
    'component_index': 2, 'target_absolute_margin_inclusive': .025, 'tail_valid_actions_per_demo': 20,
    'loss': 'uniform_seven_dimensional_valid_step_masked_action_MSE',
    'closed_loop_default_stabilization_steps': 5, 'offline_extra_stabilization_steps': 0,
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_CNN_Policy_or_environment_calls': 0,
    'new_neighbor_search_feature_distance_or_lag_sweeps': 0, 'neighbor_scan_closed': True,
    'checkpoint_HDF5_RGB_initial_private_state_reads': 0, 'source_meta_query_accessed': False,
    'sealed_roles_accessed': False, 'checkpoint_or_seed_selection': False,
    'automatic_training_promotion': False, 'G_failed_H_unrun_unchanged': True,
    'SPT_Knowledge_Module_changes': False,
}
MEMBERS = {'shared': ['targets', 'demo_indices', 'action_steps', 'demo_lengths', 'tail20'],
           'prediction': ['predictions']}
CSV_COLUMNS = {
    'transition_tags': ['task', 'seed', 'arm', 'source_row', 'demo', 'action_step', 'target_sign', 'tail20',
                        'current_prediction', 'history5_prediction', 'transition', 'policy_sign'],
    'lag_tags': ['task', 'seed', 'arm', 'source_row', 'demo', 'action_step', 'target_sign', 'tail20',
                 'original_transition', 'CC_prediction', 'HC_prediction', 'HH_prediction'],
}
ORIGINAL_GROUPS = ['stays_correct', 'corrected', 'new_error', 'stays_wrong']
GROUPS = ['all_same_sign'] + ORIGINAL_GROUPS + ['cancelled_internal_change', 'cancelled_internal_control']
PHASES = ['all_common', 'tail20', 'complement']
DESIGN = {
    'primary_focus': 'positive_corrected_and_negative_new_error_in_saved_CC_to_HH_readout',
    'primary_reference': 'positive_stays_wrong_and_negative_stays_correct; match_target_sign_and_old_CC_correctness',
    'secondary_focus': 'CC=HH=-1_HC=+1; negative_endpoints_correct_or_positive_endpoints_wrong',
    'secondary_reference': 'same_original_transition_and_target_sign_CC=HH=HC=-1',
    'groups_per_target_sign': GROUPS, 'phases': PHASES, 'directions': ['negative', 'positive'],
    'contrast_types': ['readout_change', 'cancelled_internal_change'],
    'join': 'one_to_one_task_seed_arm_source_row; verify_demo_step_target_sign_tail20_and_all3598material_IDs',
    'readout_parity': 'CC=old_current_HH=old_history_and_reconstruct_original_transition; no_new_readout',
    'Policy_sign_parity': 'sign(saved_Policy_prediction_component2)=old_transition_tags_policy_sign_including_zero',
    'legacy_parity': 'all_tge1_tail20_complement_n_MSE_translation_rotation_gripper_sign_and_translation_axes',
    'prediction_dtype': 'saved_float32_cast_float64_before_subtraction_sign_and_magnitude_comparison',
    'component3_classes': ['wrong_sign', 'prediction_zero', 'same_sign_under', 'same_sign_over', 'exact_match', 'zero_target_nonzero'],
    'class_definition': 'strict_sign_and_absolute_magnitude_no_epsilon; exact_match_includes_zero_zero',
    'primary_sign_metric': 'wrong_sign_or_prediction_zero_on_nonzero_material_target',
    'wrong_sign_output_margin': 'abs(prediction_component2)<0.025_versus_ge0.025; symmetric_no_threshold_search',
    'primary_continuous_metrics': ['component3_SSE', 'component3_MSE', 'component3_MAE', 'component3_bias',
                                   'component3_target_energy', 'component3_SSE_over_target_energy'],
    'same_frame_action_groups': {'all7': list(range(7)), 'translation': [0, 1, 2], 'other_translation': [0, 1],
                                'rotation': [3, 4, 5], 'gripper': [6]},
    'same_frame_metrics': 'each_of7_axes_MSE_MAE_bias_target_energy_relative_zero_error; action_group_dimension_means',
    'gripper_sign': 'strict_numpy_sign_equality; target_zero_and_prediction_zero_counts_separate',
    'magnitude_descriptions': 'target_abs_prediction_abs_and_abs_error_n_mean_median_q25_q75_linear_float64',
    'empty_groups': 'retain_all_groups_demos_seeds; counts_SSE0_rates_and_MSE_null; zero_energy_ratio_null',
    'contrasts': 'focus_minus_fixed_reference_same_model_phase_sign; report_group_n_and_target_energy; no_after_Policy',
    'demo_standardization': 'all30demo_rows; paired_delta_only_if_both_groups_nonempty; equal_demo_mean_of_paired_deltas',
    'demo_missing': 'retain_null_pair_and_reason; report_paired_demo_count_both_group_rows_and_positive_zero_negative_delta_counts',
    'seed_aggregation': 'report_all3seeds_and_complete_mean; higher_lower_tie_null_directions_not_significance_or_admission',
    'cross_seed': 'per_arm_same3598source_frames; group_membership_Policy_error_and_joint_votes0to3; baseline_errors_separate',
    'interpretation': 'overlap_with_expert_action_error_not_Policy_improvement_or_closed_loop_failure_cause; no_unique_expert_control_claim',
    'no_error_size_acceptance_threshold_or_pvalues': True,
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU',
    'runtime': {'python': '3.10.22', 'numpy': '1.22.4'},
    'environment': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    'analysis_active_seconds_max': 120, 'audit_active_seconds_max': 120, 'overall_watchdog_seconds': 300,
    'working_memory_bytes_max': 536870912, 'memory_monitor': 'current_/proc/self/status_VmHWM_VmRSS',
    'output_CSV_bytes_max': 268435456, 'technical_retries': 0, 'output_overwrite': False,
    'input_checksums': 'seven_fixed_NPZ_and_two_fixed_tag_CSV_byte_SHA_before_after_separate_from_numeric_decode',
    'field_guard': 'one_np_load_per_archive_and_one_decode_per_allowed_member; deny_features_logits_public_proprio_mmap_pickle_fromfile',
    'tag_guard': 'two_lexical_CSV_passes_per_stage_only_project_named_columns_into_typed_values_no_gap_distance_or_lag_columns',
    'audit': 'separate_process_independent_numerical_implementation_no_primary_analysis_import',
    'float_absolute_tolerance': 1e-10, 'float_relative_tolerance': 0, 'integer_or_tag_tolerance': 0,
    'stop': 'any_hash_endpoint_role_ID_projection_duplicate_missing_frame_metric_or_budget_mismatch_keep_partial_no_retry',
}
OUTPUT = '/tmp/libero-task7-policy-error-focus-alignment-20261108-v1.json'
ARTIFACTS = '/tmp/libero-task7-policy-error-focus-alignment-artifacts-20261108-v1'
BINDING_FILES = {
    'analysis': ROOT / 'experiments/analyze_libero_task7_policy_error_focus_alignment.py',
    'audit': ROOT / 'experiments/audit_libero_task7_policy_error_focus_alignment.py',
    'guard': ROOT / 'experiments/libero_task7_policy_error_focus_alignment_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_policy_error_focus_alignment.py',
    'metadata': Path(__file__).resolve(),
    'metadata_tests': ROOT / 'tests/test_libero_task7_policy_error_focus_alignment_protocol.py',
    'protocol': ROOT / 'docs/libero-task7-policy-error-focus-alignment-plan-2026-11-08.md',
}
CONTEXT_FILES = [BINDING_FILES['metadata'], BINDING_FILES['metadata_tests'],
                 Path(previous.__file__).resolve(), Path(p.__file__).resolve()]


def budget():
    return {'models': 6, 'shared_tge1_target_rows': 4188, 'raw_Policy_prediction_rows': 25128,
            'raw_seven_axis_squared_error_coordinates': 175896, 'shared_target_energy_coordinates': 29316,
            'common_rows_including_neutral_shared': 4068, 'aligned_material_rows_shared': 3598,
            'aligned_model_rows': 21588, 'tail20_material_positive': 75, 'tail20_material_negative': 402,
            'per_stage': {'source_NPZ_files': 7, 'source_NPZ_members': 11, 'prediction_files': 6, 'shared_files': 1,
                          'tag_CSV_passes': 2, 'tag_CSV_rows': 43176, 'projected_CSV_cells': 518112},
            'total_NPZ_loads': 14, 'total_NPZ_member_decodes': 22, 'total_tag_CSV_passes': 4,
            'tables': {'frames': 21588, 'model-groups': 252, 'demo-groups': 7560, 'contrasts': 72,
                       'demo-contrasts': 2160, 'demo-standardized': 72, 'crossseed-frames': 7196,
                       'crossseed-summary': 72, 'legacy-parity': 18},
            'legacy_scalar_parity_checks': 216,
            'new_training_forward_environment_neighbor_search_or_private_query_sealed_calls': 0}


def group_memberships(sign, current, history, hc):
    """Pure tag specification, testable without numerical libraries or payload."""
    p.require(all(type(value) is int and value in (-1, 1) for value in (sign, current, history, hc)), 'binary tag integers required')
    original = {(True, True): 'stays_correct', (False, True): 'corrected',
                (True, False): 'new_error', (False, False): 'stays_wrong'}[current == sign, history == sign]
    groups = ['all_same_sign', original]
    if current == history == -1:
        groups.append('cancelled_internal_change' if hc == 1 else 'cancelled_internal_control')
    return groups


def contrast_groups(direction, kind):
    p.require(direction in ('positive', 'negative') and kind in DESIGN['contrast_types'], 'contrast identity differs')
    if kind == 'cancelled_internal_change': return 'cancelled_internal_change', 'cancelled_internal_control'
    return ('corrected', 'stays_wrong') if direction == 'positive' else ('new_error', 'stays_correct')


def select_inputs(parents):
    source = parents['lag_science']['inputs']; captures = {(r['task'], r['seed'], r['arm']): r for r in parents['capture']['models'] if r['task'] == 7}
    legacy = {(r['task'], r['seed'], r['arm']): r for r in parents['legacy_result']['models'] if r['task'] == 7}
    models = []
    lengths = source['action_lengths']
    p.require(len(lengths) == 30 and all(type(n) is int and n >= 25 for n in lengths)
              and sum(n - 1 for n in lengths) == 4188 and sum(n - 5 for n in lengths) == 4068,
              'causal source lengths differ')
    for row in source['models']:
        key = row['task'], row['seed'], row['arm']; captured, prior = captures[key], legacy[key]
        p.require(previous.old.exact(row['checkpoint'], captured['checkpoint']) and previous.old.exact(row['checkpoint'], prior['checkpoint'])
                  and captured['features'] == row['feature'] and captured['legacy_prediction_bit_exact'] is True
                  and prior['G500_parity_passed'] is True and previous.old.exact(captured['G500_parity'], prior['strata']['all']),
                  'Policy cache endpoint/legacy parity lineage differs')
        p.require(prior['strata']['all']['valid_steps'] == 4188 and prior['strata']['tail20']['valid_steps'] == 600
                  and prior['strata']['complement']['valid_steps'] == 3588, 'legacy causal support counts differ')
        metric_keys = {'valid_steps', 'mse', 'translation_mse', 'rotation_mse', 'gripper_mse', 'gripper_sign_accuracy',
                       'translation_axis_mse', 'translation_axis_bias'}
        p.require(set(prior['strata']) == {'all', 'tail20', 'complement'}, 'legacy strata differ')
        for metrics in prior['strata'].values():
            p.require(set(metrics) == metric_keys and type(metrics['valid_steps']) is int,
                      'legacy metric schema differs')
            for key in ('mse', 'translation_mse', 'rotation_mse', 'gripper_mse', 'gripper_sign_accuracy'):
                p.require(type(metrics[key]) in (int, float) and math.isfinite(metrics[key]) and metrics[key] >= 0,
                          'legacy scalar metric invalid')
            p.require(metrics['gripper_sign_accuracy'] <= 1, 'legacy sign metric invalid')
            for key in ('translation_axis_mse', 'translation_axis_bias'):
                p.require(len(metrics[key]) == 3 and all(type(v) in (int, float) and math.isfinite(v) for v in metrics[key]),
                          'legacy axis metrics invalid')
            p.require(all(v >= 0 for v in metrics['translation_axis_mse'])
                      and abs(sum(metrics['translation_axis_mse']) / 3 - metrics['translation_mse']) <= 1e-10
                      and abs((3 * metrics['translation_mse'] + 3 * metrics['rotation_mse'] + metrics['gripper_mse']) / 7 - metrics['mse']) <= 1e-10,
                      'legacy seven-dimensional metric identity differs')
        models.append({key: row[key] for key in ('task', 'seed', 'arm', 'checkpoint')}
                      | {'prediction_archive': row['feature'], 'legacy_strata': prior['strata']})
    expected = {(7, seed, arm) for seed in p.SEEDS for arm in p.ARMS}
    p.require(len(models) == 6 and {(r['task'], r['seed'], r['arm']) for r in models} == expected, 'all six fixed source endpoints required')
    tags = {'transition_tags': parents['transition_result']['tables']['frame-transitions'],
            'lag_tags': parents['lag_result']['tables']['frames']}
    for name, ref in tags.items():
        p.require(ref['rows'] == 21588 and set(CSV_COLUMNS[name]).issubset(ref['columns']), 'saved tag schema or count differs')
    return {'shared': source['shared'], 'models': models, 'action_lengths': source['action_lengths'], 'tags': tags}


def validate_design(cfg, parents):
    p.require(cfg['kind'] == 'task7_original_Policy_action_error_focus_alignment' and type(cfg['version']) is int and cfg['version'] == 1
              and cfg['formal_result'] is False and cfg['status'] == 'scientific_frozen_execution_unbound'
              and previous.old.exact(cfg['parents'], PARENTS), 'frozen identity/parents differ')
    for key, value in [('scope', SCOPE), ('design', DESIGN), ('members', MEMBERS), ('CSV_columns', CSV_COLUMNS),
                       ('execution', EXECUTION), ('budget', budget())]:
        p.require(previous.old.exact(cfg[key], value), 'frozen ' + key + ' differs')
    p.require(all(parents[key]['formal_result'] is False for key in PARENTS), 'diagnostic parent roles required')
    audit, done = parents['lag_audit'], parents['lag_completion']
    p.require(audit['status'] == 'passed_independent_task7_lag_neighbor_audit'
              and done['status'] == 'complete_independently_audited_task7_lag_neighbor'
              and audit['result'] == done['result'] == PARENTS['lag_result'] and done['audit'] == PARENTS['lag_audit']
              and done['science_config'] == audit['science_config'] == PARENTS['lag_science']
              and done['binding'] == audit['binding'] == PARENTS['lag_binding'], 'audited lag parent required')
    audit, done = parents['transition_audit'], parents['transition_completion']
    p.require(audit['status'] == 'passed_independent_task7_history_transition_audit'
              and done['status'] == 'complete_independently_audited_task7_history_transitions'
              and audit['result'] == done['result'] == PARENTS['transition_result'] and done['audit'] == PARENTS['transition_audit'],
              'audited original tag parent required')
    p.require(previous.old.exact(cfg['inputs'], select_inputs(parents)) and cfg['output'] == OUTPUT and cfg['artifacts'] == ARTIFACTS,
              'fixed Policy input/output identities differ')
    refs = [cfg['inputs']['shared']] + [r['prediction_archive'] for r in cfg['inputs']['models']]
    p.require(len({ref['path'] for ref in refs}) == 7, 'seven unique source archives required')
    return True


def validate(path=DEFAULT_CONFIG, *, check_input_byte_hashes=False):
    cfg = json.loads(Path(path).read_text())
    p.require(previous.old.exact(cfg['parents'], PARENTS), 'immutable parent pins differ')
    parents = {key: p.load_pin(ref) for key, ref in PARENTS.items()}
    validate_design(cfg, parents); previous.validate(PARENTS['lag_science']['path'])
    p.require(len(cfg['code']) == len(CONTEXT_FILES) and {p.resolve(ref['path']).resolve() for ref in cfg['code']} == set(CONTEXT_FILES)
              and p.resolve(cfg['protocol']['path']).resolve() == BINDING_FILES['protocol'], 'required context/protocol identity differs')
    for ref in cfg['code'] + [cfg['protocol']]: p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'frozen context/protocol changed')
    if check_input_byte_hashes:
        refs = [cfg['inputs']['shared']] + [r['prediction_archive'] for r in cfg['inputs']['models']] + list(cfg['inputs']['tags'].values())
        for ref in refs: p.require(p.sha(ref['path']) == ref['sha256'], 'fixed archive/tag bytes changed')
    return cfg


def require_binding(path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    cfg = validate(path)
    p.require(Path(binding_path).is_file(), 'execution binding required before numeric payload; scientific freeze only')
    binding = json.loads(Path(binding_path).read_text())
    p.require(binding['status'] == 'bound_before_payload' and binding['formal_result'] is False
              and binding['science_config'] == p.pin(path) and previous.old.exact(binding['budget'], cfg['budget'])
              and set(binding['files']) == set(BINDING_FILES), 'execution binding identity/budget differs')
    for role, file in BINDING_FILES.items():
        ref = binding['files'][role]
        p.require(p.resolve(ref['path']).resolve() == file and p.pin(file) == ref, 'execution role/code differs')
    return binding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-executable', action='store_true'); parser.add_argument('--check-input-byte-hashes', action='store_true')
    args = parser.parse_args(); cfg = validate(check_input_byte_hashes=args.check_input_byte_hashes)
    if args.require_executable: require_binding()
    print(json.dumps({'status': cfg['status'], 'models': 6, 'future_NPZ_loads': 14, 'future_tag_CSV_passes': 4,
                      'actual_research_array_numeric_loads': 0, 'actual_tag_row_numeric_decodes': 0}, ensure_ascii=False))
