"""Metadata-only scientific freeze for fixed-current task7 residual readouts.

No NumPy/Torch import, cache decoding, auxiliary fitting or execution binding here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from experiments import libero_task7_policy_error_focus_alignment_protocol as previous

p = previous.p
ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-current-residual-correctability-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-current-residual-correctability-execution-binding_v1.json'
OUTPUT = '/tmp/libero-task7-current-residual-correctability-20261108-v1.json'
ARTIFACTS = '/tmp/libero-task7-current-residual-correctability-artifacts-20261108-v1'
PARENTS = {
    'alignment_science': {'path': str(previous.DEFAULT_CONFIG), 'sha256': '39fd9bbe7c578ddd80191602762a68ceeebc15e1731f6117ce531c5453690f30'},
    'alignment_binding': {'path': str(previous.DEFAULT_BINDING), 'sha256': 'df3f7dc79676b47d4c73e76dab60ab17962f4c8f6c915171566db21c6bd4c8e2'},
    'alignment_result': {'path': previous.OUTPUT, 'sha256': '186b49379f9c9d2aa7b5ee5a903a095e8ccaa7a1e3f93ae4faa0476447cc78a1'},
    'alignment_audit': {'path': str(Path(previous.ARTIFACTS) / 'independent-audit.json'), 'sha256': 'ce94a8d1d45a647b53c15e82a07ca1bfbdc922ce223fedfc67713d8275886a14'},
    'alignment_completion': {'path': str(Path(previous.ARTIFACTS) / 'completion.json'), 'sha256': 'f9208f5b5aae55b7f27d7577b9060e694c486a4f1ec5bc86e1d451a8df9c5ccb'},
    'capture': previous.PARENTS['capture'],
}
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'original_Policy_checkpoint_updates': 500, 'closed_loop_default_stabilization_steps': 5,
    'loss': 'uniform_seven_dimensional_valid_step_masked_action_MSE',
    'source_role': 'meta_support', 'demonstration_indices': list(range(30)),
    'auxiliary_roles': ['readout_fit', 'readout_probe'],
    'encoder_has_seen_all30_support_demos': True, 'encoder_unseen_generalization': False,
    'new_original_Policy_or_generator_optimizer_updates': 0, 'new_original_CNN_Policy_or_environment_calls': 0,
    'checkpoint_HDF5_RGB_public_proprio_initial_or_private_state_reads': 0,
    'source_meta_query_accessed': False, 'sealed_roles_accessed': False,
    'history_NN_tags_or_neighbor_search_used': False, 'checkpoint_fold_or_seed_selection': False,
    'automatic_Policy_or_training_promotion': False, 'G_failed_H_unrun_unchanged': True,
    'SPT_Knowledge_Module_changes': False,
}
PHASES = ['all_source', 'all_common', 'tail20', 'complement']
CUE_PHASES = ['all_common', 'tail20', 'complement']
DIRECTIONS = ['all', 'negative', 'positive', 'neutral']
MODES = ['original', 'bias_only', 'current_affine']
LEARNED = MODES[1:]
CONTRASTS = [['bias_only', 'original'], ['current_affine', 'original'], ['current_affine', 'bias_only']]
SNAPSHOTS = [0, 100, 250, 500]
MEMBERS = {'shared': previous.MEMBERS['shared'], 'current': ['features', 'predictions']}
HEAD_MEMBERS = ['fit_rows', 'probe_rows', 'fit_demo_indices', 'probe_demo_indices', 'sample_rows',
                'scaler_mean', 'scaler_scale', 'updates', 'weights', 'bias', 'predictions', 'step_loss',
                'gradient_norm_before_clip', 'gradient_norm_after_clip', 'optimizer_steps',
                'final_adam_m_weight', 'final_adam_v_weight', 'final_adam_m_bias', 'final_adam_v_bias']
OPTIMIZER = {'name': 'Adam', 'updates_per_readout': 500, 'batch_size': 64, 'learning_rate': .0003,
             'betas': [.9, .999], 'epsilon': 1e-8, 'weight_decay': 0., 'gradient_clip_norm': 1.,
             'amsgrad': False, 'maximize': False, 'foreach': False, 'fused': False,
             'capturable': False, 'differentiable': False,
             'loss': 'mean_over_all64_valid_rows_and_all7_action_dimensions_after_clip',
             'phase_direction_gripper_weights_regularization_or_target_normalization': False}
DESIGN = {
    'hypothesis': 'current256_representation_can_read_out_component2_residual_beyond_constant_bias_without_opposite_direction_tradeoff',
    'feature_input': 'saved_CNN_only_fused_current256_feature; no_new_or_historical_observation',
    'features_dtype_shape': ['float32', 4188, 256], 'predictions_targets_dtype_shape': ['float32', 4188, 7],
    'causal_alignment': 'obs[t-1] -> action[t], all_tge1_fit; common_tge5_scoring',
    'fold_demo_indices': [list(range(begin, begin + 10)) for begin in (0, 10, 20)],
    'fold_rule': 'one_whole10demo_probe_fold_other20demo_fit; each_raw_source_row_probe_once',
    'fit_population': 'all_tge1_rows_of20fit_demos; no_direction_phase_event_or_NN_focus_sampling',
    'baseline': 'same_saved_original_Policy_prediction_p0; no_new_original_forward_or_fit',
    'bias_only': 'delta=b; one_trainable_parameter; no_feature_dependency',
    'current_affine': 'delta=W@z+b; W256_plus_bias;257_trainable_parameters',
    'initialization': 'W=0_b=0; both_auxiliary_full7D_predictions_numerically_equal_original_at_update0',
    'action': 'p2=clip(p0_2+delta,-1,1); other6_components_bit_exact_copies_of_p0',
    'parameter_control_limit': '1_vs257_effective_parameters; matched_data_seed_loss_updates_and_initial_outputs_do_not_match_capacity_or_optimization_geometry',
    'scaler': 'per_endpoint_fold_float64_mean_population_std_of_fit_current_features_only; std_le1e-6_scale1; transform_all_to_float32',
    'transform_schedule': 'normalize_fit_rows_only_during_fit_and_release_per_job;normalize_full_cache_per_fold_after_all_fit_gate;no_probe_transform_or_forward_before_gate',
    'sampling': 'PCG64(seed+100000*fold); ascending_fit_source_rows; replacement500x64; identical_across_heads_and_A_B',
    'checkpoints': SNAPSHOTS, 'endpoint': 500,
    'probe_gate': 'all36_auxiliary_heads_complete500_updates_and_fit_only_checks_before_any_probe_scoring_or_full_cache_snapshot_readout',
    'whole_cache_decode_limit': 'cache_decode_contains_probe_rows; only_role_indexed_fit_rows_enter_scaler_gradients_or_sampling; no_claim_probe_bytes_unread',
    'post_fit_readout': 'retrospective4_snapshots_after_gate;144_full4188row_batches; no_curve_selection_or_early_stop',
    'scoring_phases': PHASES, 'directions': DIRECTIONS, 'target_material_margin_inclusive': .025,
    'direction_rule': 'negative_y2_le_minus.025_positive_y2_ge.025_other_neutral; all_includes_neutral',
    'time_rule': 'all_source_tge1;all_common_tge5;tail20_last20_per_demo;complement_common_without_tail20',
    'OOF': 'for_each_model_mode_update_stitch3_probe_predictions_by_raw_source_row; original_baseline_reused_not3_independent_samples',
    'curves': 'fit_and_probe_per_fold_at4_fixed_updates_after_all_fit; source_cache_legacy_all_tge1_parity_separate',
    'primary_metrics': ['component3_mse', 'material_sign_error_rate'],
    'material_sign_metric': 'wrong_sign_or_prediction_zero_on_abs_y2_ge.025; neutral_denominator0_rate_null',
    'descriptive_metrics': 'component3_SSE_MAE_bias_target_energy_relative_zero_error_six_strict_error_categories_and_margins;7axis_and_action_group_MSE;gripper_sign;target_prediction_error_magnitudes;raw_residual_and_clip_saturation',
    'strict_categories': previous.DESIGN['component3_classes'],
    'same_frame_action_groups': previous.DESIGN['same_frame_action_groups'],
    'comparisons': CONTRASTS, 'delta': 'candidate_minus_fixed_reference_same_OOF_source_IDs',
    'demo': 'retain_all30_with_empty_nulls; report_per_demo_and_equal_nonempty_demo_delta_means_and_direction_counts',
    'seed': 'all3_seeds_per_source_arm_raw_values_complete_mean_higher_lower_tie_null; folds_not_independent_seeds',
    'cue_only_update': 500, 'cue_positive': 'all_common_and_tail20_both_MSE_and_sign_error_strictly_lower;complement_both_nonincreasing',
    'cue_negative': 'all_common_tail20_complement_both_MSE_and_sign_error_nonincreasing',
    'cue_global': 'all_common_all_direction_sevenD_MSE_nonincreasing;other6bit_exact_invariant',
    'cue_weighting': 'pooled_OOF_frames; equal_demo_results_descriptive_and_reversals_explicit; no_weighting_switch',
    'cue_nulls': 'missing_empty_nonfinite_or_numerically_unstable_final_sign_gives_false_with_reason',
    'bias_cue': 'all3seed_bias_vs_original_flags_for_each_arm',
    'feature_extra_cue': 'all3seed_current_vs_original_AND_current_vs_bias_flags_for_each_arm',
    'inference': 'bounded_auxiliary_readout_correctability_of_exposed_support_representation;not_head_root_cause_or_unique_expert_control_or_closed_loop_improvement',
    'promotion': 'no_pvalues_formal_acceptance_automatic_rollout_or_Policy_SPT_Knowledge_Module_update_even_if_cue_true',
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU',
    'runtime': {'python': '3.10.22', 'numpy': '1.22.4', 'torch': '2.4.1+cu124'},
    'environment': previous.EXECUTION['environment'], 'torch_deterministic_algorithms': True,
    'main_active_seconds_max': 180, 'audit_active_seconds_max': 180, 'overall_watchdog_seconds': 420,
    'working_memory_bytes_max': 536870912, 'memory_monitor': 'current_/proc/self/status_VmHWM_VmRSS',
    'output_CSV_bytes_max': 268435456, 'head_NPZ_bytes_max': 134217728,
    'technical_retries': 0, 'output_overwrite': False,
    'source_access': 'each_stage7NPZ17members_exactly_once;allow_current_features_predictions_only;deny_logits_public_proprio_raw_HDF5_checkpoint_or_other_arrays',
    'head_access': 'main_write36_new_NPZ;independent_audit_read36_once19members_each;no_old_readout_or_NN_archives',
    'no_input_CSV_reads': True, 'byte_SHA': 'seven_source_NPZ_before_after_separate_from_numeric_decodes;output_head_and_table_SHA_before_after_audit',
    'role_guard': 'fit_only_scaler_sampling_and_gradients; all_heads500_gate_for_probe; canonical_complete_OOF_IDs;all_code_bound_before_payload',
    'original_weight_or_generator_access': 'metadata_SHA_only; no_torch_load_no_original_optimizer_no_original_forward',
    'audit': 'separate_process_numpy_no_Torch_no_primary_numerical_import; reconstruct_roles_scaler_batches_all_snapshots_OOF_metrics_decisions_and_invariants',
    'audit_training_limit': 'validate_actual_optimizer_settings_steps_state_finite_loss_norms_and_role_indices;not_independent_replay_of18000_optimizer_steps',
    'readout_reconstruction_absolute_tolerance': 1e-5, 'metric_absolute_tolerance': 1e-10, 'relative_tolerance': 0,
    'gradient_clip_after_norm_absolute_tolerance': 1e-6,
    'integer_ID_tag_and_other6_component_tolerance': 0,
    'numeric_sign': 'score_saved_float32_strictly; record_NumPy_reconstruction_sign_disagreements_and_abs_saved_p2_lt1e-5_without_relabeling',
    'cue_numerical_gate': 'final500_material_common_OOF_sign_disagreements_must_be0_for_candidate_and_learned_reference;else_cue_false_no_promotion',
    'stop': 'any_role_hash_endpoint_schema_budget_optimizer_nonfinite_or_audit_mismatch_stop_keep_partial_no_retry;negative_or_no_cue_end_this_candidate_no_budget_or_checkpoint_change',
}
BINDING_FILES = {
    'training': ROOT / 'experiments/run_libero_task7_current_residual_correctability.py',
    'audit': ROOT / 'experiments/audit_libero_task7_current_residual_correctability.py',
    'guard': ROOT / 'experiments/libero_task7_current_residual_correctability_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_current_residual_correctability.py',
    'metadata': Path(__file__).resolve(),
    'metadata_tests': ROOT / 'tests/test_libero_task7_current_residual_correctability_protocol.py',
    'protocol': ROOT / 'docs/libero-task7-current-residual-correctability-plan-2026-11-08.md',
}
CONTEXT_FILES = [BINDING_FILES['metadata'], BINDING_FILES['metadata_tests'], Path(previous.__file__).resolve(),
                 Path(p.__file__).resolve(), ROOT / 'experiments/run_libero_learned_phase_history_readout.py']


def folds(lengths):
    p.require(len(lengths) == 30 and all(type(n) is int and n >= 25 for n in lengths)
              and sum(n - 1 for n in lengths) == 4188, 'fixed source lengths differ')
    pairs = [(demo, step) for demo, length in enumerate(lengths) for step in range(1, length)]
    result = []
    for fold, demos in enumerate(DESIGN['fold_demo_indices']):
        probe = [i for i, (demo, _) in enumerate(pairs) if demo in demos]
        fit = [i for i, (demo, _) in enumerate(pairs) if demo not in demos]
        result.append({'fold': fold, 'probe_demos': demos.copy(), 'fit_demos': [i for i in range(30) if i not in demos],
                       'fit_rows': fit, 'probe_rows': probe, 'fit_tge1_rows': len(fit), 'probe_tge1_rows': len(probe),
                       'probe_tge5_rows': sum(pairs[i][1] >= 5 for i in probe)})
    p.require(sorted(i for row in result for i in row['probe_rows']) == list(range(4188)), 'exact OOF coverage required')
    return result


def budget():
    return {'source_endpoints': 6, 'source_rows_shared': 4188, 'feature_dimensions': 256,
            'common_rows_shared': 4068, 'common_material_rows_shared': 3598, 'tail_material_positive': 75, 'tail_material_negative': 402,
            'folds': 3, 'paired_fit_jobs': 18, 'auxiliary_heads': 36, 'parameters_by_mode': {'bias_only': 1, 'current_affine': 257},
            'updates_per_head': 500, 'auxiliary_optimizer_updates': 18000, 'auxiliary_backward_calls': 18000,
            'auxiliary_train_forward_batches': 18000, 'auxiliary_train_row_visits': 1152000,
            'unique_sampling_tables': 9, 'sample_RNG_row_draws': 288000, 'sampler_rows_stored_across36_heads': 1152000,
            'scaler_fits': 18, 'scaler_fit_row_visits': 50256, 'scaler_fit_feature_coordinates': 12865536,
            'main_fit_standardization_rows': 50256, 'main_fit_standardization_coordinates': 12865536,
            'full_cache_standardization_rows': 75384, 'full_cache_standardization_coordinates': 19298304,
            'main_total_standardization_rows': 125640, 'main_total_standardization_coordinates': 32163840,
            'snapshot_readout_batches_main': 144, 'snapshot_readout_batches_audit': 144,
            'snapshot_readout_rows_per_stage': 603072, 'snapshot_sevenD_prediction_coordinates_per_stage': 4221504,
            'OOF_all_modes_all_snapshots_logical_rows': 301536, 'final500_OOF_all_modes_rows': 75384,
            'source_NPZ_per_stage': 7, 'source_members_per_stage': 17, 'total_source_NPZ_loads': 14, 'total_source_member_decodes': 34,
            'head_NPZ_written_main': 36, 'head_NPZ_loads_audit': 36, 'members_per_head_NPZ': 19,
            'head_member_decodes_audit': 684, 'total_NPZ_loads_including_new_heads': 50, 'total_NPZ_member_decodes_including_new_heads': 718,
            'input_tag_CSV_passes': 0, 'legacy_scalar_parity_checks_per_stage': 216,
            'tables': {'curves': 4608, 'oof-groups': 1152, 'oof-demo-groups': 34560, 'contrasts': 1152,
                       'demo-contrasts': 34560, 'oof-frames': 75384, 'legacy-parity': 18},
            'seed_summary_JSON_rows': 384, 'model_comparison_flags': 18, 'arm_joint_flags': 4,
            'new_original_Policy_CNN_environment_neighbor_private_query_or_sealed_calls': 0}


def select_inputs(parents):
    source = parents['alignment_science']['inputs']; models = []
    captures = {(row['task'], row['seed'], row['arm']): row for row in parents['capture']['models'] if row['task'] == 7}
    for row in source['models']:
        key = row['task'], row['seed'], row['arm']; captured = captures[key]
        p.require(captured['features'] == row['prediction_archive'] and captured['checkpoint'] == row['checkpoint']
                  and captured['legacy_prediction_bit_exact'] is True and captured['G500_parity'] == row['legacy_strata']['all'],
                  'cached current representation and original endpoint lineage differ')
        models.append({k: row[k] for k in ('task', 'seed', 'arm', 'checkpoint', 'legacy_strata')}
                      | {'current_archive': row['prediction_archive']})
    expected = {(7, seed, arm) for seed in p.SEEDS for arm in p.ARMS}
    p.require(len(models) == 6 and {(m['task'], m['seed'], m['arm']) for m in models} == expected, 'all six current endpoints required')
    p.require(len({row['current_archive']['path'] for row in models} | {source['shared']['path']}) == 7, 'seven unique source archives required')
    return {'shared': source['shared'], 'models': models, 'action_lengths': source['action_lengths'],
            'folds': folds(source['action_lengths'])}


def comparison_flags(metrics, candidate, reference, *, numerical_sign_stable, other6_bit_exact):
    """Pure final500 decision specification. No payload, training or p-values."""
    p.require([candidate, reference] in CONTRASTS, 'unregistered comparison')
    p.require(type(numerical_sign_stable) is type(other6_bit_exact) is bool, 'strict boolean evidence required')
    def value(mode, phase, direction, key):
        node = metrics
        for field in (mode, phase, direction, key):
            if not isinstance(node, dict): return None
            node = node.get(field)
        return node
    flags = {}; missing = []
    for phase in CUE_PHASES:
        for direction in ('positive', 'negative'):
            strict = direction == 'positive' and phase in ('all_common', 'tail20')
            for key in ('component3_mse', 'material_sign_error_rate'):
                name = phase + '/' + direction + '/' + key
                left = value(candidate, phase, direction, key); right = value(reference, phase, direction, key)
                valid = all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (left, right))
                if valid and key == 'material_sign_error_rate': valid = left <= 1 and right <= 1
                if not valid: missing.append(name)
                flags[name] = bool((left < right if strict else left <= right) if valid else False)
    name = 'all_common/all/all7_mse'; left = value(candidate, 'all_common', 'all', 'all7_mse'); right = value(reference, 'all_common', 'all', 'all7_mse')
    valid = all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in (left, right))
    if not valid: missing.append(name)
    flags[name] = bool(left <= right) if valid else False
    flags['numerical_sign_stable'] = numerical_sign_stable; flags['other6_bit_exact'] = other6_bit_exact
    return {'candidate': candidate, 'reference': reference, 'flags': flags, 'missing_or_invalid': missing,
            'cue': all(flags.values()), 'automatic_promotion': False}


def joint_flags(records):
    expected = {(seed, arm, c, r) for seed in p.SEEDS for arm in p.ARMS for c, r in CONTRASTS}
    keys = [(row['seed'], row['arm'], row['candidate'], row['reference']) for row in records]
    p.require(len(records) == len(set(keys)) == 18 and set(keys) == expected, 'complete all-seed paired flags required')
    p.require(all(type(row['cue']) is bool for row in records), 'strict cue booleans required')
    expected_flags = {phase + '/' + direction + '/' + key for phase in CUE_PHASES
                      for direction in ('positive', 'negative') for key in ('component3_mse', 'material_sign_error_rate')}
    expected_flags.update(['all_common/all/all7_mse', 'numerical_sign_stable', 'other6_bit_exact'])
    p.require(all(set(row['flags']) == expected_flags and all(type(value) is bool for value in row['flags'].values())
                  and row['cue'] == all(row['flags'].values()) and row['automatic_promotion'] is False for row in records),
              'inconsistent comparison flags or promotion requested')
    result = []
    for arm in p.ARMS:
        for mode, pairs in [('bias_only', [CONTRASTS[0]]), ('current_affine_extra', CONTRASTS[1:])]:
            selected = [row for row in records if row['arm'] == arm and [row['candidate'], row['reference']] in pairs]
            result.append({'arm': arm, 'cue_type': mode, 'source_seeds': p.SEEDS.copy(),
                           'per_seed': [{'seed': seed, 'cue': all(row['cue'] for row in selected if row['seed'] == seed)} for seed in p.SEEDS],
                           'joint_cue': all(row['cue'] for row in selected), 'automatic_promotion': False})
    return result


def validate_design(cfg, parents):
    p.require(cfg['kind'] == 'fixed_current_representation_continuous_component3_correctability'
              and type(cfg['version']) is int and cfg['version'] == 1 and cfg['formal_result'] is False
              and cfg['status'] == 'scientific_frozen_execution_unbound'
              and previous.previous.old.exact(cfg['parents'], PARENTS), 'scientific identity/parents differ')
    for key, expected in [('scope', SCOPE), ('design', DESIGN), ('optimizer', OPTIMIZER), ('source_members', MEMBERS),
                          ('head_members', HEAD_MEMBERS), ('execution', EXECUTION), ('budget', budget())]:
        p.require(previous.previous.old.exact(cfg[key], expected), 'frozen ' + key + ' differs')
    p.require(all(row['formal_result'] is False for row in parents.values()), 'diagnostic parents required')
    result, audit, done = (parents[key] for key in ('alignment_result', 'alignment_audit', 'alignment_completion'))
    p.require(result['status'] == 'complete_task7_policy_error_focus_pending_audit'
              and audit['status'] == 'passed_independent_task7_policy_error_focus_audit'
              and done['status'] == 'complete_independently_audited_task7_policy_error_focus'
              and result['science_config'] == audit['science_config'] == done['science_config'] == PARENTS['alignment_science']
              and result['binding'] == audit['binding'] == done['binding'] == PARENTS['alignment_binding']
              and audit['result'] == done['result'] == PARENTS['alignment_result'] and done['audit'] == PARENTS['alignment_audit']
              and result['inputs'] == parents['alignment_science']['inputs'], 'audited fixed-current parent required')
    p.require(previous.previous.old.exact(cfg['inputs'], select_inputs(parents))
              and cfg['output'] == OUTPUT and cfg['artifacts'] == ARTIFACTS, 'source roles/folds/output differ')
    return True


def validate(path=DEFAULT_CONFIG, *, check_source_byte_hashes=False):
    cfg = json.loads(Path(path).read_text())
    p.require(previous.previous.old.exact(cfg['parents'], PARENTS), 'immutable parent pins differ')
    parents = {key: p.load_pin(ref) for key, ref in PARENTS.items()}
    validate_design(cfg, parents); previous.validate(PARENTS['alignment_science']['path'])
    p.require(len(cfg['code']) == len(CONTEXT_FILES) and {p.resolve(ref['path']).resolve() for ref in cfg['code']} == set(CONTEXT_FILES)
              and p.resolve(cfg['protocol']['path']).resolve() == BINDING_FILES['protocol'], 'context/protocol identity differs')
    for ref in cfg['code'] + [cfg['protocol']]: p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'frozen metadata/protocol changed')
    if check_source_byte_hashes:
        refs = [cfg['inputs']['shared']] + [row['current_archive'] for row in cfg['inputs']['models']]
        for ref in refs: p.require(p.sha(ref['path']) == ref['sha256'], 'cached source bytes changed')
    return cfg


def require_binding(path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    cfg = validate(path)
    p.require(Path(binding_path).is_file(), 'execution binding required before real decoding or auxiliary fit; freeze only')
    bound = json.loads(Path(binding_path).read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['formal_result'] is False
              and bound['before_research_payload'] is True and bound['science_config'] == p.pin(path)
              and previous.previous.old.exact(bound['budget'], budget()) and set(bound['files']) == set(BINDING_FILES), 'execution identity/budget differs')
    for key, file in BINDING_FILES.items(): p.require(bound['files'][key] == p.pin(file), 'execution code/role differs')
    return bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--require-executable', action='store_true')
    parser.add_argument('--check-source-byte-hashes', action='store_true'); args = parser.parse_args()
    cfg = validate(check_source_byte_hashes=args.check_source_byte_hashes)
    if args.require_executable: require_binding()
    print(json.dumps({'status': cfg['status'], 'future_auxiliary_heads': 36, 'future_auxiliary_updates': 18000,
                      'actual_cache_decodes': 0, 'actual_auxiliary_updates': 0, 'new_original_Policy_updates': 0}, ensure_ascii=False))
