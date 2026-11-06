"""Frozen scientific design and pure operators; validates metadata without running a Policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.environments.libero_dataset import resolve_demonstration_access

DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-expert-camera-target-error-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-expert-camera-target-error-protocol-validation-20261102-v1.json')
TASKS, SEEDS = (4, 7), (20261022, 20261023, 20261024)
VIEWS = ('agentview_rgb', 'eye_in_hand_rgb')
OBS_KEYS = (*VIEWS, 'joint_states', 'gripper_states')
CONDITIONS = ['identity_uint8', 'identity_float', 'eye_in_hand_photo_matched', 'eye_in_hand_spatial_matched',
              'agentview_photo_matched', 'agentview_spatial_matched', 'eye_in_hand_spatial_full', 'agentview_spatial_full']
COMPONENTS = {'translation': [0, 3], 'rotation': [3, 6], 'gripper': [6, 7], 'all7': [0, 7]}
PINS = {
    'expert_result': '11bed14d3624144553d11e9395b66ee4621e69b5284e740c75117a60e7d7273e',
    'expert_audit': '8a2e6ed375680c88b2fabcc0d7b165096f87e806ef20d7ac42b188f39cbf3bf5',
    'expert_config': '9051738cf37d7cc6f3b1fd09a01e93755ff314cda2cb719e83db98f50ff95cef',
    'expert_helper': 'bd44981057bbb09ecc485379ca4ad13563b3f770d892dbd9c73ce3e1af557425',
    'expert_preflight': 'd3c4abc420be3e1370d56e27cc6b26f834516abbc343be1c836f93d7ba225f3e',
    'input_decomposition_result': '2cc2ab1f14cf6bfdc2ede73d01a2bcd3df1f3d62ed72adc6c722f739df287ecc',
    'input_decomposition_config': 'efcdcb8a090923ce5b6726eb746bf5cfa7e1012dd0c16de16d6c68d9d8874613',
    'input_decomposition_audit': '88688b21a7edd74ccf49634129e2addd9c59463eb9a32ece1489a810802d02c5',
    'transfer': '6cdc8c3eeaa90886ad0e15f0f4ebbe260aa972b6e69b1f122e95f65ab63dd7d6',
    'dataset_manifest': '866429693febf966a38b48e3b89faa54623acace5d3c50cd67c5c7978010eaf6',
    'dataset_access_code': 'a35ce74905655a1e1f3f0ffb597329eb486e000a5f9dee925d8231885a7f7599',
    'policy_config': 'f5e57c895e9249ef7a3361ebee2119bf06c6cbb48da01c31980b1eb0af8feb1b',
    'policy_code': 'a3098a68b5ceab21495207856c9d94fd4a6d26c1668737625ecb7b5d46977735',
    'offline_helpers': '9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4',
    'old_camera_sensitivity_result': 'dab39a916d43f2eb2d8dbfad6690504367131983efbdb4afc88f9e71bb892ec7',
    'old_augmentation_result': '06e309d7f22bb04dd0b18a5759388989fa56591af015fece3ad4de11eb9f1ecc',
}
SCOPE = {
    'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arm': 'source_baseline',
    'checkpoint_updates': 500, 'memory_mode': 'cnn_only', 'default_closed_loop_stabilization_steps': 5,
    'offline_extra_stabilization_steps': 0, 'source_role': 'meta_support', 'demo_indices': list(range(30)),
    'target_alignment': 'stored_HDF5_obs[t-1]_to_original_expert_action[t]; t>=1',
    'target_origin': 'original_source_expert_action_at_unchanged_reference_state',
    'unique_source_demonstrations': 60, 'excluded_t0_rows': 60,
    'source_action_payload_rows': 8690, 'scored_unique_expert_rows': 8630,
    'expert_rows_by_task': {'4': 4442, '7': 4188}, 'shared_batches_by_task': {'4': 154, '7': 146},
    'batch_size': 32, 'checkpoint_loads': 6, 'conditions': CONDITIONS,
    'planned_main_forward_calls': 7200, 'planned_main_prediction_rows': 207120,
    'planned_repeat_forward_calls': 48, 'planned_repeat_prediction_rows': 1536,
    'planned_original_batch_parity_checks': 900, 'planned_uint8_float_identity_checks': 900,
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_resets_or_steps': 0,
    'new_policy_control_steps': 0, 'new_initial_state_or_private_state_reads': 0,
    'source_query_accessed': False, 'qualification_accessed': False, 'reserved_accessed': False,
    'validation_accessed': False, 'heldout_accessed': False, 'final_evaluation_accessed': False,
    'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False,
    'device': 'cpu', 'torch_threads': 1, 'analysis_seed': 20261022,
}
TRANSFORMS = {
    'input': 'original_uint8_N1281283_RGB_only; other modalities and reference targets unchanged',
    'normalization': 'torch.from_numpy(uint8).to(float32).div(255); then float64 arithmetic and float32 output',
    'photo_base': 'per_image_per_channel_spatial_mean_broadcast; no spatial resampling',
    'spatial_base': 'nonoverlapping_8x8_block_means_repeated_to_128x128; no crop_or_translation',
    'matched_budget': 'm=0.1*min(d_photo_agent,d_spatial_agent,d_photo_hand,d_spatial_hand) per source frame',
    'distance': 'mean_absolute_pixel_delta_on_normalized_RGB',
    'mixture': 'U+(m/d)*(base-U); d=0 gives alpha0; all four alpha<=0.1',
    'same_m_for_both_cameras_and_both_matched_controls': True,
    'zero_budget_rows': 'both matched controls unchanged; retain rows and record active flag; no exclusion',
    'full_spatial_probe': 'one camera replaced by its entire spatial_base; unmatched secondary information-loss probe',
    'normalised_realized_mae_tolerance': 1e-7, 'no_clipping_or_uint8_requantization': True,
    'no_strength_direction_roi_or_seed_sweep': True, 'inputs_only_not_targets_or_predictions_determine_parameters': True,
}
ANALYSIS = {
    'primary': 'step_weighted_translation_MSE_deltas_on_all_support_t_ge1; task_seed_condition_separate',
    'secondary': 'all7_rotation_gripper_MSE; demo_equal_means_medians; command_strata_and_t1',
    'components': COMPONENTS,
    'strata': ['all', 'first_scored_t1', 'expert_open', 'expert_close', 'expert_neutral',
               'around_close_transition', 'around_open_transition'],
    'transition_radius': 5, 'command_masks': 'reuse_frozen_expert_helper; source_action0_for_masks_only_not_scored',
    'paired_error_identity': 'DeltaMSE_D=(2*e_D.dot(delta_D)+delta_D.dot(delta_D))/dim_D; e=clean-target',
    'identity_max_abs_tolerance': 1e-12, 'arithmetic': 'float64_after_float32_predictions_and_targets',
    'per_row_terms': ['clean_MSE', 'changed_MSE', 'DeltaMSE', 'error_alignment_cross_term', 'response_energy_term'],
    'mse_identity': 'all7=(3*translation+3*rotation+gripper)/7',
    'empty_strata_or_zero_denominators': 'null_with_valid_missing_counts; no_infinity_or_zero_imputation',
    'appearance_cue': 'photo_matched_Delta_translation_MSE>0_in_all_three_seeds_per_camera',
    'spatial_detail_cue': '(spatial_matched_Delta_translation_MSE-photo_matched_Delta_translation_MSE)>0_AND_spatial_full_Delta_translation_MSE>0_in_all_three_seeds_per_camera',
    'secondary_all7_flags': 'same_rules_separately; cannot_override_translation_primary',
    'flags_can_coexist': True, 'zero_or_nonfinite_does_not_satisfy_strict_positive': True,
    'camera_contrast': 'hand_minus_agent_paired_DeltaMSE_at_identical_per_frame_pixel_MAE; both cameras reported',
    'no_significance_independence_or_population_claim': True,
    'spatial_probe_removes_texture_and_may_be_off_distribution': True,
    'no_proof_of_geometric_necessity_or_closed_loop_cause': True,
    'corrupted_observations_not_new_expert_demonstrations_or_training_labels': True,
    'no_training_rollout_checkpoint_or_qualification_promotion': True,
}
GATES = [
    'all_pinned_source_roles_code_configs_checkpoints_and_support_references_match',
    'authorize_meta_support_before_opening_HDF5; never_read_query_groups_or_init_attrs',
    'original_targets_indices_command_masks_and_every_shared_batch_digest_exact_to_old_support_archive',
    'original_uint8_predictions_exact_to_old_baseline_predictions_in_same_batch_partition',
    'float_identity_predictions_exact_to_uint8_predictions_for_every_batch',
    'repeat_all_eight_conditions_for_first_batch_of_each_task_and_seed_exact',
    'matched_per_frame_pixel_MAE_agrees_with_common_budget_within1e-7_and_all_coefficients<=0.1',
    'input_target_weight_hashes_unchanged_grad_none_finite_outputs_within_action_bounds',
    'signed_error_and_component_identities_residual<=1e-12',
    'actual_calls_rows_source_demo_order_and_t0_exclusions_match_frozen_budget',
    'on_any_failure_stop_preserve_events_partial_predictions_and_failed_result; never_relax_or_append_budget',
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def normalized(image):
    require(image.dtype == np.uint8 and image.ndim == 4 and image.shape[1:] == (128, 128, 3), 'original RGB must be uint8 N1281283')
    return torch.from_numpy(image).to(torch.float32).div(255).numpy().copy()


def prepare_interventions(observations):
    require(set(observations) == set(OBS_KEYS), 'expert public observation interface differs')
    n = len(observations[VIEWS[0]])
    require(n > 0 and all(len(observations[k]) == n for k in OBS_KEYS), 'unaligned expert observation batch')
    for key, width in (('joint_states', 7), ('gripper_states', 2)):
        require(observations[key].shape == (n, width) and observations[key].dtype == np.float32 and
                np.isfinite(observations[key]).all(), 'invalid public proprioception')
    u = {v: normalized(observations[v]) for v in VIEWS}
    bases, distances = {}, {}
    for view in VIEWS:
        values = u[view].astype(np.float64)
        photo = np.broadcast_to(values.mean(axis=(1, 2), keepdims=True), values.shape)
        spatial = values.reshape(n, 16, 8, 16, 8, 3).mean(axis=(2, 4)).repeat(8, axis=1).repeat(8, axis=2)
        for kind, base in (('photo', photo), ('spatial', spatial)):
            bases[view, kind] = base
            distances[view, kind] = np.abs(base - values).mean(axis=(1, 2, 3))
    budget = .1 * np.minimum.reduce(list(distances.values()))
    identity = {**observations, **u}
    result = {'identity_uint8': observations, 'identity_float': identity}
    diagnostics = {'common_pixel_mae': budget.tolist(), 'active_rows': (budget > 0).tolist(),
                   'base_distances': {f'{v}_{kind}': value.tolist() for (v, kind), value in distances.items()}, 'controls': {}}
    for view, prefix in ((VIEWS[1], 'eye_in_hand'), (VIEWS[0], 'agentview')):
        for kind in ('photo', 'spatial'):
            distance = distances[view, kind]
            alpha = np.divide(budget, distance, out=np.zeros_like(budget), where=distance > 0)
            require(np.all((alpha >= 0) & (alpha <= .1 + 1e-12)), 'matched mixture coefficient differs')
            changed = (u[view].astype(np.float64) + alpha[:, None, None, None] *
                       (bases[view, kind] - u[view])).astype(np.float32)
            realized = np.abs(changed.astype(np.float64) - u[view]).mean(axis=(1, 2, 3))
            require(np.max(np.abs(realized - budget)) <= 1e-7, 'matched pixel MAE differs')
            name = f'{prefix}_{kind}_matched'
            result[name] = {**identity, view: changed}
            diagnostics['controls'][name] = {'alpha': alpha.tolist(), 'realized_pixel_mae': realized.tolist()}
        result[f'{prefix}_spatial_full'] = {**identity, view: bases[view, 'spatial'].astype(np.float32)}
    require(set(result) == set(CONDITIONS), 'intervention condition set differs')
    return {k: result[k] for k in CONDITIONS}, diagnostics


def error_terms(clean, changed, target, component='translation'):
    require(component in COMPONENTS, 'unknown action component')
    require(clean.shape == changed.shape == target.shape and clean.ndim == 2 and clean.shape[1] == 7,
            'prediction/target shape differs')
    require(all(np.isfinite(v).all() and np.all(np.abs(v) <= 1) for v in (clean, changed, target)), 'invalid action values')
    start, stop = COMPONENTS[component]
    error = clean.astype(np.float64)[:, start:stop] - target.astype(np.float64)[:, start:stop]
    delta = changed.astype(np.float64)[:, start:stop] - clean.astype(np.float64)[:, start:stop]
    cross = 2 * (error * delta).mean(axis=1)
    energy = (delta * delta).mean(axis=1)
    before, after = (error * error).mean(axis=1), ((error + delta) ** 2).mean(axis=1)
    require(not len(before) or np.max(np.abs((after - before) - cross - energy)) <= 1e-12, 'target error identity differs')
    return {'clean_MSE': before, 'changed_MSE': after, 'DeltaMSE': after - before,
            'error_alignment_cross_term': cross, 'response_energy_term': energy}


def verify_design(config):
    require(config.get('schema_version') == 1 and config.get('check_type') == 'libero_expert_camera_target_error_protocol' and
            config.get('formal_result') is False and config.get('research_run_executed') is False and
            config.get('protocol_status') == 'scientific_design_frozen_execution_binding_pending', 'protocol design status differs')
    require(config.get('scope') == SCOPE and config.get('transforms') == TRANSFORMS and config.get('analysis') == ANALYSIS and
            config.get('stop_rules') == GATES, 'frozen roles/budget/intervention/analysis differs')
    require(config.get('execution_binding') == {'status': 'pending_runner_implementation_and_pre_run_validation',
            'must_pin': ['this_design_sha256', 'execution_runner_sha256', 'operator_validator_sha256', 'launch_environment'],
            'scientific_design_must_not_change': True, 'actual_experiment_result_is_separate': True}, 'execution binding differs')


def validate(config_path):
    cfg = load_json(config_path)
    verify_design(cfg)
    require(cfg['operator_validator_sha256'] == sha(Path(__file__)), 'operator/validator code differs')
    require(set(cfg['inputs']) == set(PINS), 'frozen input roles differ')
    paths = {}
    for name, row in cfg['inputs'].items():
        require(row['sha256'] == PINS[name], f'frozen reference pin differs: {name}')
        p = ROOT / row['path']
        require(p.is_file() and sha(p) == row['sha256'], f'input pin differs: {name}')
        paths[name] = p
    expert = load_json(paths['expert_result'])
    require(expert['status'] == 'complete' and expert['formal_result'] is False and
            load_json(paths['expert_audit'])['result_sha256'] == cfg['inputs']['expert_result']['sha256'], 'expert reference incomplete')
    source_cfg = load_json(paths['input_decomposition_config'])
    expected_models = [{'task_id': row['task_id'], 'training_seed': row['training_seed'], 'checkpoint': row['checkpoint']}
                       for row in source_cfg['model_inputs']]
    require(cfg['models'] == expected_models and {(r['task_id'], r['training_seed']) for r in cfg['models']} ==
            {(t, s) for t in TASKS for s in SEEDS} and len(cfg['models']) == 6, 'baseline500 matrix differs')
    references = [{'task_id': t, 'role': 'meta_support', 'dataset': expert['task_results'][str(t)]['dataset'],
        'archive': expert['task_results'][str(t)]['roles']['meta_support']['archive'],
        'source_action_lengths': [r['source_action_count'] for r in expert['task_results'][str(t)]['roles']['meta_support']['coverage']],
        'shared_input_batch_count': len(expert['task_results'][str(t)]['roles']['meta_support']['shared_input_batches']),
        'shared_input_batch_manifest_sha256': expert['task_results'][str(t)]['roles']['meta_support']['shared_input_batch_manifest_sha256']} for t in TASKS]
    require(cfg['support_references'] == references, 'support-only target/batch reference differs')
    transfer, manifest = load_json(paths['transfer']), load_json(paths['dataset_manifest'])
    access = []
    for row in references:
        task, lengths = row['task_id'], row['source_action_lengths']
        batches = expert['task_results'][str(task)]['roles']['meta_support']['shared_input_batches']
        batch_hash = hashlib.sha256(json.dumps(batches, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        require(batch_hash == row['shared_input_batch_manifest_sha256'], 'original batch manifest digest differs')
        role = resolve_demonstration_access(transfer, manifest, task_id=task, demonstration_role='meta_support', access_phase='development')
        require(role.task_role == 'source_spi' and not role.locked_before_method_freeze and role.demonstration_indices == tuple(range(30)), 'support authorization differs')
        require(len(lengths) == 30 and sum(v - 1 for v in lengths) == SCOPE['expert_rows_by_task'][str(task)] and
                sum((v - 2) // 32 + 1 for v in lengths) == len(batches) == SCOPE['shared_batches_by_task'][str(task)], 'support row/batch budget differs')
        for batch in batches:
            require(batch['observation_start'] == batch['target_start'] - 1 and
                    batch['observation_stop_exclusive'] == batch['target_stop_exclusive'] - 1 and
                    batch['target_start'] >= 1 and batch['demonstration_index'] in range(30), 'causal batch alignment differs')
        record = row['archive']
        require(sha(Path(record['path'])) == record['sha256'], 'support archive hash differs')
        access.append(role.to_dict())
    for row in cfg['models']:
        cp = row['checkpoint']
        require(cp['optimizer_updates'] == 500 and cp['arm'] == 'source_baseline' and cp['task_id'] == row['task_id'] and
                cp['training_seed'] == row['training_seed'] and sha(Path(cp['path'])) == cp['file_sha256'], 'checkpoint provenance differs')
    return {'status': 'passed_protocol_validation', 'formal_result': False, 'research_run_executed': False,
        'config_path': str(config_path), 'config_sha256': sha(config_path), 'operator_validator_sha256': sha(Path(__file__)),
        'role_authorization': access, 'support_batch_metadata_verified': 300, 'support_target_archives_hash_verified': 2,
        'checkpoint_file_hashes_verified': 6, 'new_model_forward_or_backward_calls': 0, 'new_optimizer_updates': 0,
        'new_HDF5_payload_or_initial_state_reads': 0, 'new_environment_calls': 0, 'query_payload_read': False,
        'execution_binding_status': cfg['execution_binding']['status'], 'validated_at_utc': datetime.now(timezone.utc).isoformat()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    require(not args.output.exists(), 'protocol validation output already exists')
    result = validate(args.config.resolve())
    with args.output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
