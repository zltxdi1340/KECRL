"""Metadata-only freeze for task7 cached current/history direction transitions.

This module never imports a numerical runtime or decodes a research array.  The
analysis, payload guard and independent audit require a separate execution bind.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments import libero_source_expert_late_translation_protocol as p

ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-component3-history-transition-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-component3-history-transition-execution-binding_v1.json'
PARENT_DIRECTORY = Path('/tmp/libero-source-expert-component3-observability-artifacts-20261107-v1')
PARENTS = {
    'science': {'path': str(ROOT / 'configs/libero-source-expert-component3-observability-diagnostic_v1.json'),
                'sha256': 'acac57118851969ece67cd2fe606927678e34a50647db78343aaf7280f513550'},
    'binding': {'path': str(ROOT / 'configs/libero-source-expert-component3-observability-execution-binding_v1.json'),
                'sha256': '8dd3aef8bd6aba28dbe120b205fd1347167590671907d256558bae0fe37d7051'},
    'capture': {'path': str(PARENT_DIRECTORY / 'capture.json'),
                'sha256': 'd0d87e754e82f1cf86a2d2ae5a2c0a88e95c46621b548c53a17ab964d39dd0ab'},
    'result': {'path': '/tmp/libero-source-expert-component3-observability-20261107-v1.json',
               'sha256': '8c10b1537ac15d97e4a8226831f97eed018145bee95fb6cc92c718d824d694e7'},
    'audit': {'path': str(PARENT_DIRECTORY / 'independent-audit.json'),
              'sha256': '1fefdd95d6caa405d8349ccf9ac850831bd10a9468c979dfb59823a1eb368261'},
    'completion': {'path': str(PARENT_DIRECTORY / 'completion.json'),
                   'sha256': '054f067d77f295ac6df970709ff473970af53bff9f7ac47466719db1903fc015'},
}
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'arms': p.ARMS, 'checkpoint_updates': 500,
    'role': 'source_meta_support_training_fit_only', 'demonstration_indices': list(range(30)),
    'component_index': 2, 'material_absolute_margin_inclusive': .025,
    'causal_alignment': 'obs[t-1] -> action[t], t>=1', 'common_first_action_step': 5,
    'history_steps': 5, 'tail_valid_actions_per_demo': 20,
    'primary_space': 'pre_action_head_256', 'primary_rule': 'balanced',
    'modes': ['current', 'history5'], 'original_policy_sign': 'secondary_zero_is_error',
    'offline_extra_stabilization_steps': 0, 'closed_loop_default_stabilization_steps': 5,
    'loss': 'uniform_seven_dimensional_valid_step_masked_action_MSE',
    'new_training_updates': 0, 'new_backward_calls': 0, 'new_CNN_or_Policy_forward_calls': 0,
    'new_nearest_neighbor_searches': 0, 'new_feature_or_full_library_distance_computations': 0,
    'checkpoint_HDF5_RGB_environment_initial_private_state_reads': 0,
    'source_meta_query_accessed': False, 'sealed_roles_accessed': False,
    'checkpoint_or_seed_selection': False, 'automatic_training_promotion': False,
    'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False,
}
ANALYSIS = {
    'primary_stratum': 'tail20_both_target_signs',
    'strata': ['all_common', 'tail20', 'complement'], 'directions': ['negative', 'positive'],
    'transitions': ['stays_correct', 'corrected', 'new_error', 'stays_wrong'],
    'transition_definition': 'compare_current_and_history5_predictions_to_same_nonzero_target_sign',
    'conservation': 'history_correct-current_correct=corrected-new_error_for_each_model_demo_phase_sign',
    'denominators': ['class_n', 'current_wrong_n_for_correction_rate', 'current_correct_n_for_new_error_rate'],
    'empty_groups': 'retain_zero_counts_null_rates_no_demo_or_seed_drop',
    'seed_aggregation': 'all_three_seed_mean_and_each_seed_counts_never_pool_as_independent_frames',
    'seed_agreement': 'per_arm_same_source_row_all_three_codes_and_vote_counts_0_1_2_3_for_each_transition',
    'time_coordinates': ['action_step', 'demo_length', 'steps_to_last_action', 'offset_from_tail20_start'],
    'label_history': 'target_direction_labels_t_through_t_minus4_descriptive_only_neutral_zero_retained',
    'material_run': 'maximal_consecutive_original_action_steps_with_same_material_sign_neutral_breaks_run',
    'run_age': 'action_step-minus-observed_run_start; start_at_t1_left_censored_no_action0_read',
    'run_age_summary': 'observed_age_0_to3_vs_ge4_and_left_censor_flag_reported_separately',
    'target_magnitude': 'abs_saved_float32_component2_cast_float64_no_bins_or_threshold_search',
    'distance_gap': 'mean_negative5_distance-minus-mean_positive5_distance; positive_iff_gap>0_tie_negative',
    'normalized_gap': 'gap/(mean_negative5_distance+mean_positive5_distance); zero_denominator_null',
    'gap_comparison': 'within_same_model_only_no_calibration_confidence_or_cross_model_distance_claim',
    'neighbor_analysis': 'saved_balanced_10_per_mode_only_negative5_then_positive5_no_new_search',
    'neighbor_fields': ['source_row', 'demo', 'action_step', 'target_sign', 'tail20', 'distance', 'rank_within_class'],
    'neighbor_composition': ['per_class_current_history_ID_overlap_0_to5', 'per_class_distinct_reference_demos',
                             'per_class_reference_tail20_count', 'reference_demo_and_step_identity'],
    'neighbor_long_table': 'all_tail20_rows_all_four_transitions_both_signs_not_only_changed_frames',
    'descriptive_summaries': 'n_mean_median_q25_q75_float64_linear_interpolation_no_significance_tests',
    'no_geometric_or_success_phase_inference': True,
    'no_secondary_readout_or_public9_rescue': True,
}
MEMBERS = {
    'shared': ['targets', 'demo_indices', 'action_steps', 'demo_lengths', 'tail20',
               'common_rows', 'history_rows', 'selected_rows', 'direction_labels'],
    'probe': ['source_rows', 'labels', 'policy_sign'] +
             [mode + '_balanced_' + key for mode in ('current', 'history5')
              for key in ('prediction', 'neighbors', 'distances', 'ties')],
    'denied': ['public_proprio', 'features', 'logits', 'predictions', '*ordinary*', '*public9*'],
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU',
    'runtime': {'python': '3.10.22', 'numpy': '1.22.4'},
    'environment': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1',
                    'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    'analysis_active_seconds_max': 120, 'independent_audit_active_seconds_max': 120,
    'overall_watchdog_seconds': 300, 'working_memory_bytes_max': 536870912,
    'technical_retries': 0, 'output_overwrite': False, 'pickle_or_mmap_allowed': False,
    'array_loads': 'exact_seven_whole_archives_once_per_stage_only_allowed_members_no_np_fromfile',
    'input_hash_checks': 'seven_pinned_files_byte_SHA_before_and_after_not_array_numeric_decodes',
    'audit': 'separate_process_independent_implementation_no_import_of_primary_analysis',
    'audit_checks': ['same_seven_file_hashes_and_member_allowlist', 'full_causal_ID_history_labels_and_all_six_endpoints',
                     'all_21588_frame_transition_rows_and_parent_confusions', 'all_431760_saved_neighbor_records',
                     'all_group_conservation_seed_votes_time_fields_gaps_and_summary_tables'],
    'audit_integer_tolerance': 0, 'audit_float_absolute_tolerance': 1e-10, 'audit_float_relative_tolerance': 0,
    'stop': 'any_provenance_role_ID_dtype_shape_nonfinite_neighbor_or_budget_mismatch_keep_partial_no_retry',
}
BINDING_FILES = {
    'analysis': ROOT / 'experiments/analyze_libero_task7_component3_history_transition.py',
    'independent_audit': ROOT / 'experiments/audit_libero_task7_component3_history_transition.py',
    'payload_guard': ROOT / 'experiments/libero_task7_component3_history_transition_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_component3_history_transition.py',
    'metadata_protocol': Path(__file__).resolve(),
    'protocol_tests': ROOT / 'tests/test_libero_task7_component3_history_transition_protocol.py',
    'protocol': ROOT / 'docs/libero-task7-component3-history-transition-plan-2026-11-08.md',
}
CONTEXT_FILES = [
    Path(p.__file__).resolve(),
    ROOT / 'experiments/run_libero_source_expert_component3_observability.py',
    ROOT / 'experiments/libero_component3_direction_neighbors.py',
    ROOT / 'experiments/audit_libero_source_expert_component3_observability.py',
    ROOT / 'experiments/libero_source_expert_component3_observability_access.py',
    BINDING_FILES['metadata_protocol'], BINDING_FILES['protocol_tests'],
]
OUTPUT = '/tmp/libero-task7-component3-history-transition-20261108-v1.json'
ARTIFACTS = '/tmp/libero-task7-component3-history-transition-artifacts-20261108-v1'


def exact(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def derive_budget():
    # Exact counts already audited in the parent run, not inferred from net gains.
    return {
        'shared_t_ge1_rows': 4188, 'shared_common_rows_including_neutral': 4068,
        'shared_material_common_rows': 3598, 'shared_neutral_common_rows_excluded': 470,
        'shared_material_negative_rows': 2273, 'shared_material_positive_rows': 1325,
        'shared_tail20_material_rows': 477, 'shared_tail20_negative_rows': 402,
        'shared_tail20_positive_rows': 75, 'fixed_models': 6,
        'per_stage': {'shared_npz_loads': 1, 'probe_npz_loads': 6, 'total_array_file_loads': 7,
                      'model_frame_records': 3598 * 6, 'tail20_model_frames': 477 * 6,
                      'saved_neighbor_records_checked': 3598 * 6 * 2 * 10},
        'analysis_and_audit_array_file_loads': 14,
        'expected_table_rows': {'frame-transitions': 3598 * 6, 'model-transitions': 6 * 3 * 2 * 4,
                                'demo-transitions': 6 * 30 * 3 * 2 * 4,
                                'seed-agreement-frames': 3598 * 2,
                                'seed-agreement-counts': 2 * 3 * 2 * 4 * 4,
                                'tail20-neighbors': 477 * 6 * 2 * 10},
        'new_training_updates': 0, 'new_CNN_Policy_environment_or_nearest_neighbor_search_calls': 0,
        'checkpoint_HDF5_initial_private_query_or_sealed_reads': 0,
    }


def source_selection(parents):
    source, captured, result = [parents[key] for key in ('science', 'capture', 'result')]
    expected = {(7, seed, arm) for seed in p.SEEDS for arm in p.ARMS}
    models = [row for row in result['primary_results'] if row['task'] == 7]
    endpoints = [row for row in source['endpoints'] if row['task'] == 7]
    identity = lambda row: (row['task'], row['seed'], row['arm'])
    p.require(len(models) == len(endpoints) == 6 and {identity(r) for r in models} ==
              {identity(r) for r in endpoints} == expected, 'all six fixed task7 endpoints required')
    endpoint_map = {identity(row): row for row in endpoints}
    selected = []
    for row in sorted(models, key=identity):
        endpoint = endpoint_map[identity(row)]
        p.require(endpoint['checkpoint_update'] == 500 and endpoint['support_role'] == 'meta_support'
                  and exact(endpoint['checkpoint'], row['checkpoint']), 'fixed500 source endpoint lineage differs')
        p.require(row['archive']['path'] == str(PARENT_DIRECTORY / f"task7-seed{row['seed']}-{row['arm']}-probe.npz"),
                  'task7 probe archive identity differs')
        for mode in ('current_balanced', 'history5_balanced'):
            for phase, expected_counts in {'all_common': (3598, 2273, 1325), 'tail20': (477, 402, 75),
                                           'complement': (3121, 1871, 1250)}.items():
                metrics = row['metrics'][mode]['strata'][phase]
                p.require(exact([metrics[k] for k in ('n', 'negative_n', 'positive_n')], list(expected_counts))
                          and metrics['zero_prediction_n'] == 0, 'audited shared class denominators differ')
        selected.append({k: row[k] for k in ('task', 'seed', 'arm', 'checkpoint', 'archive')})
    shared = [r for r in captured['shared'] if r['task'] == 7]
    p.require(len(shared) == 1 and shared[0]['path'] == str(PARENT_DIRECTORY / 'task7-shared.npz'),
              'single task7 shared archive required')
    lengths = source['action_lengths']['7']
    p.require(len(lengths) == 30 and all(type(n) is int and n >= 25 for n in lengths)
              and sum(n - 1 for n in lengths) == 4188 and sum(n - 5 for n in lengths) == 4068,
              'task7 causal source lengths differ')
    return {'shared': {k: shared[0][k] for k in ('path', 'sha256')}, 'models': selected,
            'action_lengths': lengths}


def validate_parent_lineage(parents):
    science, binding, capture, result, audit, completion = [parents[k] for k in PARENTS]
    p.require(all(row['formal_result'] is False for row in (science, binding, capture, result, audit, completion)),
              'diagnostic parent roles required')
    p.require(binding['status'] == 'bound_before_payload' and binding['science_config'] == PARENTS['science']
              and capture['status'] == 'complete_component3_feature_capture'
              and result['status'] == 'complete_component3_observability_pending_independent_audit'
              and audit['status'] == 'passed_independent_component3_observability_audit'
              and completion['status'] == 'complete_independently_audited_component3_observability_diagnostic',
              'completed independently audited parents required')
    p.require(all(row['science_config'] == PARENTS['science'] and row['binding'] == PARENTS['binding']
                  for row in (capture, result, audit, completion))
              and result['capture'] == audit['capture'] == PARENTS['capture']
              and audit['result'] == completion['result'] == PARENTS['result']
              and completion['audit'] == PARENTS['audit'], 'parent provenance chain differs')


def validate_design(config, parents):
    p.require(config['kind'] == 'task7_component3_cached_history_transitions_readonly'
              and type(config['version']) is int and config['version'] == 1
              and config['formal_result'] is False
              and config['status'] == 'scientific_protocol_frozen_execution_unbound'
              and exact(config['parents'], PARENTS), 'frozen identity or parent pins differ')
    for key, expected in [('scope', SCOPE), ('analysis', ANALYSIS), ('allowed_members', MEMBERS),
                          ('execution', EXECUTION), ('budget', derive_budget())]:
        p.require(exact(config[key], expected), 'frozen scientific ' + key + ' differs')
    validate_parent_lineage(parents)
    p.require(exact(config['inputs'], source_selection(parents)), 'all fixed input identities must match parents')
    p.require(config['output'] == OUTPUT and config['artifacts'] == ARTIFACTS, 'new output identities differ')
    return True


def validate(config_path=DEFAULT_CONFIG, *, verify_array_hashes=False):
    config = json.loads(Path(config_path).read_text())
    p.require(exact(config['parents'], PARENTS), 'immutable audited parent pins differ')
    parents = {key: p.load_pin(ref) for key, ref in PARENTS.items()}
    validate_design(config, parents)
    p.require(len(config['context_code']) == len(CONTEXT_FILES)
              and {p.resolve(ref['path']).resolve() for ref in config['context_code']} == set(CONTEXT_FILES)
              and p.resolve(config['protocol']['path']).resolve() == BINDING_FILES['protocol'],
              'required frozen context or protocol identities differ')
    for ref in config['context_code'] + [config['protocol']]:
        p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'frozen context or protocol changed')
    refs = [config['inputs']['shared']] + [r['archive'] for r in config['inputs']['models']]
    p.require(len({r['path'] for r in refs}) == 7, 'exactly seven unique arrays required')
    if verify_array_hashes:
        for ref in refs:
            p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'cached array byte hash changed')
    return config


def require_execution_binding(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    config = validate(config_path)
    p.require(Path(binding_path).is_file(), 'execution binding required before array payload; freeze is not executable')
    binding = json.loads(Path(binding_path).read_text())
    p.require(binding.get('formal_result') is False and binding.get('status') == 'bound_before_payload'
              and binding.get('science_config') == p.pin(config_path)
              and exact(binding.get('budget'), config['budget'])
              and set(binding.get('files', {})) == set(BINDING_FILES), 'incomplete execution binding')
    for role, path in BINDING_FILES.items():
        ref = binding['files'][role]
        p.require(p.resolve(ref['path']).resolve() == path and path.is_file()
                  and p.sha(path) == ref['sha256'], 'bound execution role or code differs: ' + role)
    return binding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--verify-array-byte-hashes', action='store_true')
    parser.add_argument('--require-executable', action='store_true')
    args = parser.parse_args()
    config = validate(args.config, verify_array_hashes=args.verify_array_byte_hashes)
    if args.require_executable:
        require_execution_binding(args.config)
    print(json.dumps({'status': config['status'], 'fixed_models': 6,
                      'future_analysis_and_audit_array_loads': 14,
                      'actual_research_array_numeric_decodes': 0,
                      'actual_new_training_or_Policy_environment_calls': 0,
                      'array_bytes_hashed_only': 7 if args.verify_array_byte_hashes else 0}, ensure_ascii=False))
