"""Scientific design/budget validation only; no numerical runtime or payload access."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from experiments import libero_source_expert_late_translation_protocol as p

DEFAULT_CONFIG = p.ROOT / 'configs/libero-source-expert-component3-observability-diagnostic_v1.json'
DEFAULT_BINDING = p.ROOT / 'configs/libero-source-expert-component3-observability-execution-binding_v1.json'
BINDING_FILES = {
    'runner': p.ROOT / 'experiments/run_libero_source_expert_component3_observability.py',
    'direction_neighbors': p.ROOT / 'experiments/libero_component3_direction_neighbors.py',
    'audit_runner': p.ROOT / 'experiments/audit_libero_source_expert_component3_observability.py',
    'payload_guard': p.ROOT / 'experiments/libero_source_expert_component3_observability_access.py',
    'runtime_tests': p.ROOT / 'tests/test_libero_source_expert_component3_observability.py',
    'metadata_protocol': Path(__file__).resolve(),
    'protocol_tests': p.ROOT / 'tests/test_libero_source_expert_component3_observability_protocol.py',
    'protocol': p.ROOT / 'docs/libero-source-expert-component3-observability-protocol-2026-11-07.md',
}
SCOPE = {
    'task_ids': p.TASKS, 'training_seeds': p.SEEDS, 'arms': p.ARMS, 'checkpoint_updates': 500,
    'source_role': 'meta_support', 'demonstration_indices': list(range(30)), 'component_index': 2,
    'target_absolute_margin_inclusive': .025, 'causal_alignment': 'obs[t-1] -> action[t], t>=1',
    'common_first_action_step': 5, 'history_steps': 5, 'tail_valid_actions_per_demo': 20,
    'sequence_length': 10, 'stride': 10, 'batch_size': 32,
    'primary_space': 'pre_action_head_256', 'secondary_space': 'public_joint7_gripper2',
    'modes': ['current', 'history5'], 'repeated_current5_alias': 'current',
    'primary_readout': 'mean_distance_to_five_nearest_per_class', 'reference_readout': 'ordinary_majority_5NN',
    'closed_loop_default_stabilization_steps': 5, 'offline_extra_stabilization_steps': 0,
    'loss': 'uniform_seven_dimensional_valid_step_masked_action_MSE',
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
    'new_initial_private_state_reads': 0, 'source_query_accessed': False, 'sealed_roles_accessed': False,
    'checkpoint_selected': False, 'automatic_training_promotion': False,
    'SPT_Knowledge_Module_changes': False, 'G_failed_H_unrun_unchanged': True,
}
READOUT = {
    'neighbors_per_class': 5, 'ordinary_neighbors': 5, 'query_chunk': 64, 'coordinate_block': 64,
    'distance_dtype': 'float64', 'distance': 'direct_squared_euclidean_no_Gram',
    'history_distance': 'mean_of_five_corresponding_lag_squared_distances',
    'reference_isolation': 'entire_query_demo_excluded_from_bank_and_scaler',
    'scaler_fit_rows': 'other_demo_all_t_ge_5_current_frames_including_neutral_targets',
    'scaler': 'population_mean_std; std<=1e-6 replaced_by_1; same_scaler_for_all_lags_and_modes',
    'neighbor_bank_rows': 'other_demo_t_ge_5_abs_target_ge_0.025_only',
    'query_rows': 'query_demo_t_ge_5_abs_target_ge_0.025_only',
    'history': 'current_and_previous_four_legal_features_same_demo_no_padding_future_or_expert_actions',
    'stable_order': 'demo_ascending_then_action_step_ascending',
    'balanced_rule': 'predict_positive_if_mean_positive5_distance<mean_negative5_distance_else_negative',
    'balanced_distance_tie': -1, 'ordinary_rule': 'positive_if_at_least_three_of_five_labels_positive_else_negative',
    'minimum_reference_rows_per_class': 5, 'missing_reference_class': 'stop_no_k_change_or_fold_drop',
    'empty_or_single_class_score_stratum': 'retain_counts_recalls_error_rates_BA_null_if_either_class_absent',
    'absolute_easy_accuracy_threshold': None, 'significance_tests': False,
    'primary_metrics': ['balanced_accuracy', 'negative_recall', 'positive_recall', 'confusion_counts', 'error_count'],
    'metric_aggregation': 'per_model_sum_fold_confusions_then_compute_BA; all_three_seed_mean_no_missing_seed_drop',
    'paired_comparison_arithmetic': 'integer_confusion_cross_products_on_identical_class_denominators',
    'strata': ['all_common', 'tail20', 'complement', 'negative', 'positive', 'original_policy_wrong', 'per_demo'],
    'original_policy_wrong_stratum': 'secondary_outcome_stratified_not_independent_test',
    'original_policy_prediction': 'sign_of_saved_component3_zero_counted_as_error',
    'paired_history_cue': 'all_three_seeds_strict_BA_gain_on_all_common_and_tail20_with_neither_tail_recall_lower',
    'paired_current_over_policy_cue': 'same_joint_rule_current_balanced_readout_versus_original_policy',
    'secondary_cannot_rescue_primary_cue': True,
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required',
    'physical_gpu': 1, 'GPU_uuid': 'GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff', 'logical_device': 'cuda:0',
    'GPU_max_memory_MiB': 512, 'GPU_max_utilization_percent': 5, 'other_compute_processes_allowed': 0,
    'GPU_wait_seconds': 3600, 'GPU_wait_poll_seconds': 30,
    'capture_active_seconds': 600, 'CPU_probe_active_seconds': 2400, 'independent_audit_active_seconds': 1200,
    'overall_watchdog_seconds': 8100, 'technical_retries': 0, 'device_or_cpu_capture_fallback': False,
    'host_observation_cache_bytes_max': 2147483648, 'CPU_neighbor_working_bytes_max': 1073741824,
    'CUDA_VISIBLE_DEVICES': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'PYTHONHASHSEED': '0',
    'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
    'deterministic': True, 'TF32': False, 'augmentation': False, 'capture_dtype': 'float32',
    'runtime': {'python': '3.10.22', 'torch': '2.4.1+cu124', 'numpy': '1.22.4', 'h5py': '3.11.0', 'cuda': '12.4'},
    'old_prediction_parity': 'bit_exact_all111888_rows', 'first_batch_repeat': 'bit_exact_features_logits_predictions_including_padding',
    'logit_reconstruction': 'torch_tanh_logits_bit_exact_predictions',
    'G500_aggregate_absolute_tolerance': 1e-10, 'G500_aggregate_relative_tolerance': 0,
    'audit_distance_absolute_tolerance': 1e-8, 'audit_metric_absolute_tolerance': 1e-10,
    'raw_RGB_or_private_archive': False, 'output_overwrite': False,
}


def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(b, sort_keys=True, allow_nan=False)


def endpoint_identity(row):
    return {key: row[key] for key in ('task', 'seed', 'arm', 'checkpoint_update', 'support_role', 'checkpoint')}


def read_shared_counts(path):
    """Only n fields of an existing audited CSV; never load its original arrays."""
    counts, seen, selected = {}, set(), {}
    with Path(path).open(newline='') as handle:
        for row in csv.DictReader(handle):
            task, seed, demo = (int(row[key]) for key in ('task', 'seed', 'demo'))
            arm, phase, direction, band = (row[key] for key in ('arm', 'phase', 'direction', 'band'))
            p.require(task in p.TASKS and seed in p.SEEDS and arm in p.ARMS and demo in range(30)
                      and phase in ('tail20', 'complement') and direction in ('all', 'negative', 'zero', 'positive')
                      and band in ('all', 'material'), 'out-of-scope direction count')
            key = task, seed, arm, demo, phase, direction, band
            p.require(key not in seen and row['n'].isdigit(), 'duplicate or malformed count metadata')
            seen.add(key); n = int(row['n'])
            if band == 'material' and direction in ('negative', 'positive'):
                shared = task, demo, phase, direction
                p.require(shared not in selected or selected[shared] == n, 'shared labels differ across models')
                selected[shared] = n
    p.require(len(seen) == 14400 and len(selected) == 600, 'incomplete source-demo count metadata')
    for task in p.TASKS:
        counts[str(task)] = [{phase: {direction: selected[task, d, phase, direction]
                                    for direction in ('negative', 'positive')}
                             for phase in ('tail20', 'complement')} for d in range(30)]
    return counts


def derive_budget(lengths_by_task, counts):
    """Exact geometry budgets and conservative label-dependent upper/lower bounds."""
    task_rows = []; valid = windows = batches = common = upper = pairs = repeats = 0
    for task in p.TASKS:
        lengths, rows = lengths_by_task[str(task)], counts[str(task)]
        p.require(len(lengths) == len(rows) == 30 and all(type(n) is int and n >= 25 for n in lengths), 'invalid source length metadata')
        plan = p.window_plan(lengths); n = sum(x - 1 for x in lengths); nc = sum(x - 5 for x in lengths)
        material_lower, material_upper = [], []; by_class = {k: [] for k in ('negative', 'positive')}
        tail_counts = {k: 0 for k in by_class}
        for length, row in zip(lengths, rows):
            tail = row['tail20']; comp = row['complement']
            p.require(set(tail) == set(comp) == set(by_class) and
                      all(type(v) is int and v >= 0 for v in list(tail.values()) + list(comp.values())) and
                      sum(tail.values()) <= 20 and sum(comp.values()) <= length - 21,
                      'inconsistent material count metadata')
            material_lower.append(sum(tail.values()) + max(0, sum(comp.values()) - 4))
            material_upper.append(sum(tail.values()) + min(sum(comp.values()), length - 25))
            for sign in by_class:
                by_class[sign].append(tail[sign] + max(0, comp[sign] - 4))
                tail_counts[sign] += tail[sign]
        lower_bank = {sign: min(sum(ns) - n for n in ns) for sign, ns in by_class.items()}
        p.require(min(lower_bank.values()) >= 5, 'metadata cannot guarantee five references per class')
        u = sum(material_upper); pair = u * u - sum(x * x for x in material_upper)
        b = (len(plan) + 31) // 32; r = sum(x['valid'] for x in plan[:32])
        task_rows.append({'task': task, 'source_t_ge_1_rows': n, 'common_t_ge_5_rows': nc,
                          'windows': len(plan), 'main_batches_per_model': b,
                          'material_common_rows_lower': sum(material_lower), 'material_common_rows_upper': u,
                          'reference_class_rows_lower_in_any_fold': lower_bank,
                          'tail20_material_class_rows_exact': tail_counts, 'distance_pair_rows_upper_per_model': pair,
                          'repeat_valid_rows_per_model': r})
        valid += n; windows += len(plan); batches += b; common += nc; upper += u; pairs += pair; repeats += r
    exact = {'models': 30, 'checkpoint_loads': 30, 'HDF5_files': 5, 'HDF5_File_opens': 10,
             'source_windows': windows, 'source_rows_per_modality_and_actions': valid,
             'main_forward_batches': batches * 6, 'repeat_forward_batches': 30,
             'main_valid_rows': valid * 6, 'main_slots': windows * 10 * 6,
             'repeat_valid_rows': repeats * 6, 'repeat_slots': 30 * 32 * 10,
             'common_rows_before_neutral_exclusion_shared': common, 'common_rows_before_neutral_exclusion_models': common * 6,
             'primary_fold_mode_rule_records': 30 * 30 * 2 * 2, 'secondary_fold_mode_rule_records': 5 * 30 * 2 * 2,
             'capture_legacy_reference_array_loads': 35, 'CPU_probe_new_array_loads': 35,
             'independent_audit_new_array_loads': 100, 'independent_audit_legacy_reference_array_loads': 35,
             'total_capture_probe_audit_array_file_loads': 205,
             'capture_output_array_files': 65, 'probe_output_array_files': 35,
             'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
             'new_initial_private_state_reads': 0, 'action0_or_source_query_or_sealed_payload_reads': 0}
    caps = {'material_common_rows_shared': upper, 'material_common_rows_models': upper * 6,
            'primary_scored_output_rows': upper * 6 * 4, 'secondary_scored_output_rows': upper * 4,
            'primary_lag_pair_distances': pairs * 6 * 5, 'secondary_lag_pair_distances': pairs * 5,
            'primary_coordinate_squared_differences': pairs * 6 * 5 * 256,
            'secondary_coordinate_squared_differences': pairs * 5 * 9,
            'audit_selected_neighbor_distances': upper * 7 * 2 * (5 + 10),
            'audit_selected_neighbor_lag_distances': upper * 7 * (1 + 5) * (5 + 10),
            'audit_full_bank_sentinel_folds': 35 * 30}
    return {'exact': exact, 'label_dependent_upper_bounds': caps, 'per_task': task_rows,
            'unknown_until_execution': 'exact material t>=5 counts and their subsets; derive once before first CNN, no budget/mask changes'}


def load_inputs(inputs):
    json_names = ('readout_config', 'readout_result', 'readout_audit', 'readout_completion',
                  'error_config', 'error_result', 'error_audit', 'old_feature_config')
    loaded = {name: p.load_pin(inputs[name]) for name in json_names}
    base, result, audit, completion = [loaded[x] for x in ('readout_config', 'readout_result', 'readout_audit', 'readout_completion')]
    error, error_audit = loaded['error_result'], loaded['error_audit']
    p.require(base['formal_result'] is False and result['formal_result'] is False and error['formal_result'] is False
              and result['config'] == inputs['readout_config'] and audit['result'] == completion['result'] == inputs['readout_result']
              and completion['independent_audit'] == inputs['readout_audit']
              and error['config'] == inputs['error_config'] and error_audit['result'] == inputs['error_result']
              and error['cache_arrays_verified_before_after'] == result['labels'] + [x['predictions'] for x in result['models']],
              'current source-cache lineage differs')
    p.require(audit['status'] == 'passed_independent_source_expert_late_translation_audit'
              and completion['status'] == 'complete_independently_audited_source_expert_late_translation_diagnostic'
              and error_audit['status'] == 'passed_independent_readonly_component3_error_types_audit'
              and error['status'] == 'complete_readonly_source_component3_error_types', 'completed independent parents required')
    ref = error['tables']['demo-decomposition']
    p.require({k: ref[k] for k in ('path', 'sha256')} == inputs['demo_counts'], 'direction count table lineage differs')
    p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'audited count metadata changed')
    counts = read_shared_counts(p.resolve(ref['path']))
    return loaded, counts


def validate_design(config, source, counts):
    p.require(config['formal_result'] is False and config['status'] == 'scientific_protocol_frozen_execution_unbound'
              and exact(config['scope'], SCOPE) and exact(config['readout'], READOUT) and exact(config['execution'], EXECUTION),
              'frozen scientific scope/readout/execution differs')
    expected = {(t, s, a) for t in p.TASKS for s in p.SEEDS for a in p.ARMS}
    endpoints = [endpoint_identity(row) for row in source['endpoints']]
    p.require(len(endpoints) == 30 and {(r['task'], r['seed'], r['arm']) for r in endpoints} == expected
              and all(r['checkpoint_update'] == 500 and r['support_role'] == 'meta_support' for r in endpoints)
              and config['endpoints'] == endpoints and config['action_lengths'] == source['action_lengths']
              and config['data_files'] == source['data_files'] and config['data_root'] == source['data_root']
              and config['policy_config'] == source['G_reference_selection']['effective_policy_config'],
              'all thirty fixed source endpoints/data/config required')
    p.require(config['policy_config']['memory_mode'] == 'cnn_only'
              and config['policy_config']['recurrent_hidden_dim'] == 256, 'pre-head space differs')
    p.require(config['budget'] == derive_budget(source['action_lengths'], counts), 'derived scientific budgets differ')
    return True


def validate(config_path=DEFAULT_CONFIG):
    config = json.loads(Path(config_path).read_text())
    p.require(exact(config['scope'], SCOPE), 'frozen scientific scope differs')
    loaded, counts = load_inputs(config['inputs'])
    validate_design(config, loaded['readout_config'], counts)
    for ref in config['context_code'] + [config['protocol']]:
        p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'frozen context/protocol changed')
    old_weights = {r['weight_hash'] for r in loaded['old_feature_config']['endpoints']}
    matches = sum(r['checkpoint']['weight_hash'] in old_weights for r in config['endpoints'])
    p.require(all(not any(k in row for k in ('features', 'prehead', 'representation', 'projection'))
                  for row in loaded['readout_result']['models']), 'current canonical manifest now contains features')
    p.require(config['cache_inventory']['old18_weight_matches'] == matches == 0
              and config['cache_inventory']['current_manifest_models_with_prehead_features'] == 0,
              'declared cache inventory differs')
    return config


def require_execution_binding(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    config = validate(config_path)
    p.require(Path(binding_path).is_file(), 'execution binding required before any payload; scientific freeze is not executable')
    binding = json.loads(Path(binding_path).read_text())
    p.require(binding.get('formal_result') is False and binding.get('science_config') == p.pin(config_path)
              and binding.get('status') == 'bound_before_payload' and set(binding.get('files', {})) == set(BINDING_FILES)
              and exact(binding.get('budget'), config['budget']),
              'incomplete or mismatched execution binding')
    for role, ref in binding['files'].items():
        p.require(p.resolve(ref['path']).resolve() == BINDING_FILES[role], 'bound role points to wrong implementation')
        p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'bound execution code changed')
    return binding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--require-executable', action='store_true')
    args = parser.parse_args(); cfg = validate(args.config)
    if args.require_executable:
        require_execution_binding(args.config)
    print(json.dumps({'status': cfg['status'], 'models': len(cfg['endpoints']), 'budget': cfg['budget'],
                      'actual_new_payload_or_numeric_runtime_calls': 0}, ensure_ascii=False))
