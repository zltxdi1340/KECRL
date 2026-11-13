"""Freeze-only metadata checks for a bounded source-candidate geometry capture.

This module has no capture mode, numeric payload loader, Policy, or simulator.
An execution entry, observer and independent audit must be bound separately.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from experiments import libero_formal_transfer_protocol_v2 as prior

ROOT = prior.ROOT
require, sha = prior.require, prior.sha
TASKS = [0, 1, 3, 4, 7]
SEEDS = [20261106, 20261107, 20261108]
ARMS = ['A_original', 'B_fomaml']
INDICES = [0, 1, 2, 3]
ARCHIVE = Path('/tmp/libero-formal-G-artifacts-20261105-v2')
ENV_ROOT = Path('/home/zl202621/.conda/envs/libero-kecrl/lib/python3.10/site-packages/robosuite')
LIBERO_ROOT = Path('/home/zl202621/third_party/LIBERO/libero/libero/envs')
METADATA_PATHS = {
    'result': Path('/tmp/libero-formal-G-result-20261105-v2.json'),
    'training': ARCHIVE/'training.json',
    'training_audit': ARCHIVE/'training-audit.json',
    'completion': ARCHIVE/'completion.json',
    'complete_audit': ARCHIVE/'complete-audit.json',
    'manifest': ROOT/'configs/libero-formal-G-execution-manifest_v2.json',
    'family': ROOT/'configs/libero_spatial_skill_family_candidate_v1.json',
}
SOURCE_PATHS = {
    'prior_science': ROOT/'configs/libero-formal-transfer-protocol_v2.json',
    'prior_execution_binding': ROOT/'configs/libero-formal-G-execution-binding_v2.json',
    'prior_protocol': ROOT/'experiments/libero_formal_transfer_protocol_v2.py',
    'public_adapter': ROOT/'src/environments/libero_adapter.py',
    'selective_initial_reader': ROOT/'src/environments/libero_initial_state_reader.py',
    'policy': ROOT/'src/algorithms/libero_bc_policy.py',
    'libero_wrapper': LIBERO_ROOT/'env_wrapper.py',
    'libero_domain': LIBERO_ROOT/'bddl_base_domain.py',
    'libero_object_states': LIBERO_ROOT/'object_states/base_object_states.py',
    'robosuite_environment': ENV_ROOT/'environments/base.py',
    'robosuite_binding': ENV_ROOT/'utils/binding_utils.py',
    'robosuite_single_arm': ENV_ROOT/'robots/single_arm.py',
    'robosuite_gripper': ENV_ROOT/'models/grippers/panda_gripper.py',
    'robosuite_gripper_model': ENV_ROOT/'models/grippers/gripper_model.py',
    'robosuite_macros': ENV_ROOT/'macros.py',
}


def canonical(value):
    return json.dumps(value, sort_keys=True, allow_nan=False)


def design():
    return {
        'scope': {
            'tasks': TASKS, 'seeds': SEEDS, 'arms': ARMS,
            'initial_role': 'candidate_nonregression', 'global_initial_indices': INDICES,
            'selection_rule': 'all_source_tasks; first_three_G_seed_ids; first_four_candidate_indices',
            'known_outcomes': True, 'selection_uses_performance_ranking': False,
            'cohort': 'new_fresh_adapter_prefixes; not a replay of G or validation36',
        },
        'fixed_policy': {
            'checkpoint_update': 500, 'new_training_updates': 0,
            'loss': 'uniform_seven_dimensional_valid_step_masked_action_MSE',
            'policy_observation': ['agentview_rgb', 'eye_in_hand_rgb', 'joint_states', 'gripper_states'],
            'family': 'CNN_only', 'deterministic': True, 'eval_mode': True,
            'dtype': 'float32', 'TF32': False, 'new_loss_augmentation_or_history': False,
        },
        'capture': {
            'environment_seed': 20261022, 'stabilization_steps': 5,
            'stabilization_action': [0.0]*7, 'maximum_policy_actions_per_prefix': 150,
            'native_policy_horizon': 300, 'native_horizon_including_stabilization': 305,
            'control_frequency_hz': 20, 'physics_timestep_seconds': 0.002,
            'physics_substeps_per_control': 25, 'boundary_dt_tolerance_seconds': 1e-9,
            'lifecycle': 'fresh_adapter_per_episode; no prior_scene_history',
            'adapter_construction': 'inject_native_environment_and_single_authorized_row; never_default_bank_loader',
            'episode_python_numpy_torch_seed_before_adapter': 20261022,
            'order': 'task_then_seed_then_arm_then_ascending_global_initial_index',
            'native_reset_attempts_per_episode': 1, 'automatic_retries': 0,
            'cutoff_reason': 'diagnostic_prefix_limit_not_environment_timeout',
            'stop_at_native_success': True, 'hidden_state_reset_per_episode': True,
        },
        'observer': {
            'classification': 'privileged_evaluation_only_geometry_sidecar',
            'native_simulation_writes_or_extra_forward_or_step': 0,
            'shadow_method': 'one_independent_MjData_per_episode; mj_fwdPosition_once_per_saved_boundary',
            'shadow_copied_fields': ['qpos', 'qvel', 'time', 'mocap_pos', 'mocap_quat', 'eq_active'],
            'raw_private_state_serialization': False,
            'positions': 'world_metres; gripper_grip_site and bowl/plate_root_body_origins',
            'rotations': 'world_3x3_row_major_matrices',
            'source_object': 'akita_black_bowl_1', 'target_object': 'plate_1',
            'eef_resolution': 'robot0.gripper.important_sites[grip_site]',
            'finger_resolution': 'important_geoms[left_finger] and [right_finger], including pads',
            'object_resolution': 'objects_dict[name].root_body and contact_geoms; validate against obj_body_id',
            'contact_records': 'shadow_collision_contacts_involving_bowl_only; geom_names_ids,dist_m,pos_world',
            'contacts_per_snapshot_cap': 256, 'contact_overflow': 'stop_not_truncate',
            'touch_proxy': 'shadow_contact_distance_le_zero; retain_positive_margin_records_separately',
            'bilateral_touch_proxy': 'both_finger_groups_touch_bowl_at_same_saved_boundary',
            'force_or_stable_grasp_claim': False, 'site_check_contact_used': False,
            'policy_receives_geometry': False, 'original_RGB_or_observation_recomputed': False,
            'RGB_archived': False, 'all_native_buffers_unchanged_after_observer': True,
        },
        'analysis': {
            'state_zero': 'post_five_step_stabilization',
            'action_k_pre_state': 'k_minus_1', 'action_k_post_state': 'k',
            'close_onset_time': 'state[k_minus_1].sim_time',
            'touch_time': 'state[j].sim_time; state0_touch_is_preexisting',
            'close_to_touch_seconds': 'touch_time_minus_close_onset_time; not_j_minus_k',
            'pre_close_distance_deltas': 'd[t]-d[t-1], t=max(1,k-5)..k-1',
            'first_close': 'action[6]>0', 'neutral': 'separate_sign_zero',
            'first_multi_close': 'first_observed_run_of_at_least_two_positive_actions; retrospective_start',
            'first_open': 'first_negative_after_first_close',
            'event_window_offsets': list(range(-5, 6)),
            'primary': ['pre_close_eef_to_bowl_xyz_xy_signed_z',
                        'first_close_to_first_bilateral_touch_proxy_time_order_with_censoring'],
            'secondary': ['pre_close_five_transition_distance_changes',
                          'bowl_height_change_from_reset', 'bowl_to_plate_xyz_xy',
                          'close_runs_and_open_after_close_with_geometry'],
            'paired_alignment': 'same_absolute_state/action_index_within_each_event_anchor',
            'retention': 'all_60_pairs; all_15_task_seed_groups; missing_and_prefix_terminal_censoring',
            'distance_phase_threshold_or_grasp_qualification': None,
            'forward_fill_or_adaptive_window_extension': False,
            'significance_causal_or_unseen_generalization_claims': False,
            'full_300_step_success_rate_estimated': False,
            'loss_checkpoint_seed_or_method_selection': False,
        },
        'budget': {
            'models': 30, 'paired_scenes': 60, 'episodes': 120,
            'checkpoint_deserializations_max': 30, 'adapter_creations_max': 120,
            'native_reset_attempts_max': 120, 'policy_forwards_and_actions_max': 18000,
            'zero_actions_max': 600, 'native_control_steps_max': 18600,
            'native_physics_steps_max': 465000, 'public_rows_max': 18120,
            'shadow_allocations_max': 120, 'shadow_position_forwards_max': 18120,
            'geometry_rows_max': 18120, 'saved_bowl_contact_records_max': 4638720,
            'initial_containers_max': 5, 'numeric_initial_rows_max': 20,
            'active_wall_seconds_max': 3600, 'new_optimizer_updates': 0,
            'HDF5_reads': 0, 'old_raw_trajectory_reads': 0, 'new_RGB_archives': 0,
            'validation_heldout_qualification_reserved_or_final_payload_reads': 0,
            'technical_rerun_or_replacement_episodes': 0,
        },
        'gates': {
            'technical': ['separate_execution_code_and_independent_audit_binding_before_payload',
                          'all30_checkpoint_file_and_weight_hashes_match_before_first_environment',
                          'bounded_reset_without_legacy_unlimited_retry',
                          'paired_reset_public_digest_and_geometry_exact',
                          'native_state_and_policy_inputs_unchanged_by_shadow_observer',
                          'resolved_site_body_and_disjoint_finger_object_geom_groups',
                          'finite_coordinates_rotations_and_time_alignment',
                          'complete_pair_accounting_and_attempt_completed_resource_ledger'],
            'technical_failure': 'retain_partial_and_failure; stop; no_retry_or_new_budget',
            'no_close_or_touch_before_prefix_end': 'right_censored_not_invalid_or_full_episode_failure',
            'performance_stop_or_accept_threshold': None,
            'G_decision': 'failed_unchanged; H_sealed',
            'SPT_Knowledge_or_Module_updates': False,
        },
        'runtime': {
            'conda_environment': 'libero-kecrl', 'python': '3.10.22',
            'torch': '2.4.1+cu124', 'robosuite': '1.4.0', 'mujoco': '3.2.3',
            'libero_revision': '8f1084e3132a39270c3a13ebe37270a43ece2a01',
            'device': 'CUDA_float32; physical_GPU_UUID_and_occupancy_bound_before_execution',
            'silent_device_fallback': False, 'archive_outside_repository': True,
        },
        'freeze_boundary': {
            'scientific_design_frozen': True, 'execution_binding_complete': False,
            'capture_implementation_present_at_freeze': False, 'capture_authorized_by_this_preflight': False,
            'new_policy_optimizer_environment_initial_private_or_shadow_calls_this_freeze': 0,
        },
    }


def model_endpoints(training):
    expected = {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS}
    found = {}
    for model in training['models']:
        key = model['task'], model['seed'], model['arm']
        if key not in expected:
            continue
        require(key not in found and model['support_role'] == 'meta_support' and
                model['query_role'] is None and model['initial_state_role'] == 'candidate_nonregression' and
                model['initial_state_indices'] == list(range(20)) and
                sorted(map(int, model['curve'])) == prior.CURVE, 'source endpoint identity/roles differ')
        endpoint = {k: model['curve']['500'][k] for k in ('path', 'sha256', 'weight_hash')}
        require(endpoint['path'] == str(ARCHIVE/f'G-task{key[0]}-seed{key[1]}-{key[2]}-u500.pt') and
                all(re.fullmatch('[0-9a-f]{64}', endpoint[k]) for k in ('sha256', 'weight_hash')),
                'fixed500 endpoint path/hash differs')
        found[key] = {'task': key[0], 'seed': key[1], 'arm': key[2], 'checkpoint_update': 500,
                      'support_role': 'meta_support', 'initial_state_role': 'candidate_nonregression',
                      'checkpoint': endpoint}
    require(set(found) == expected, 'all30 existing source endpoints must be retained')
    return [found[t, s, a] for t in TASKS for s in SEEDS for a in ARMS]


def task_bindings(family):
    tasks = {r['task_id']: r for r in family['tasks']}
    return [{'task': t, 'task_name': tasks[t]['task_name'],
             'source_object': 'akita_black_bowl_1', 'target_object': 'plate_1',
             'source_geometry': tasks[t]['source_geometry'],
             'source_initial_predicate': tasks[t]['source_initial_predicate']} for t in TASKS]


def validate_config(config):
    require(set(config) == {'schema_version', 'status', 'formal_result', 'diagnostic_only', 'design',
                           'metadata_inputs', 'source_references', 'model_endpoints', 'initial_banks', 'task_bindings'},
            'unexpected freeze config fields')
    require(config['schema_version'] == 1 and config['status'] == 'science_frozen_not_execution_bound' and
            config['formal_result'] is False and config['diagnostic_only'] is True and
            canonical(config['design']) == canonical(design()), 'frozen source geometry design differs')
    for label, paths in [('metadata_inputs', METADATA_PATHS), ('source_references', SOURCE_PATHS)]:
        require(set(config[label]) == set(paths), 'metadata/source reference scope differs')
        for name, path in paths.items():
            pin = config[label][name]
            require(set(pin) == {'path', 'sha256'} and pin['path'] == str(path) and
                    re.fullmatch('[0-9a-f]{64}', pin['sha256']), 'unapproved input path/hash')
    require(all(p.suffix == '.json' for p in METADATA_PATHS.values()), 'freeze metadata must be JSON')
    require(set(config['initial_banks']) == set(map(str, TASKS)) and
            len(config['model_endpoints']) == 30, 'source-only inventory differs')


def preflight(config_path, binding_path):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)  # Reject scope/budget/payload paths before provenance I/O.
    binding = json.loads(Path(binding_path).read_text())
    require(binding.get('status') == 'source_candidate_geometry_science_frozen' and
            binding.get('config_sha256') == sha(config_path) and
            binding.get('execution_binding_complete') is False and
            binding.get('capture_authorized') is False, 'science binding is not an execution authorization')
    require(set(binding['files']) == {'protocol_validator', 'protocol_report', 'tests'}, 'freeze code pins incomplete')
    for pin in [*config['metadata_inputs'].values(), *config['source_references'].values(), *binding['files'].values()]:
        path = Path(pin['path']); path = path if path.is_absolute() else ROOT/path
        require(path.suffix in ('.json', '.py', '.md') and sha(path) == pin['sha256'], 'freeze metadata/code checksum differs')
    require((ROOT/binding['files']['protocol_validator']['path']).resolve() == Path(__file__).resolve(),
            'wrong geometry protocol validator')
    docs = {name: json.loads(path.read_text()) for name, path in METADATA_PATHS.items()}
    training, train_audit, raw, done, audited = (docs[n] for n in
        ('training', 'training_audit', 'result', 'completion', 'complete_audit'))
    pins = config['metadata_inputs']
    require(training['status'] == 'complete_training' and training['stage'] == raw['stage'] == 'G' and
            train_audit['status'] == 'passed_training_audit' and
            done['status'] == 'complete_audited_G' and audited['status'] == 'passed_complete_audit' and
            done['decision'] == audited['decision'] and done['decision']['passed'] is False,
            'audited negative G lineage missing')
    require(train_audit['training_sha256'] == raw['training']['sha256'] == pins['training']['sha256'] and
            raw['training_audit']['sha256'] == pins['training_audit']['sha256'] and
            audited['result_sha256'] == done['result']['sha256'] == pins['result']['sha256'] and
            done['audit']['sha256'] == pins['complete_audit']['sha256'], 'G input lineage changed')
    require(canonical(config['model_endpoints']) == canonical(model_endpoints(training)), 'fixed source500 endpoints changed')
    require(config['initial_banks'] == {str(t): docs['manifest']['initial_banks'][str(t)] for t in TASKS} and
            config['task_bindings'] == task_bindings(docs['family']), 'source bank/task metadata differs')
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists() and not args.output.resolve().is_relative_to(ROOT), 'output must be new and outside repository')
    config = preflight(args.config, args.binding)
    record = {'status': 'source_candidate_geometry_science_preflight_passed_no_capture',
              'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'formal_result': False,
              'config_sha256': sha(args.config), 'binding_sha256': sha(args.binding),
              'models': len(config['model_endpoints']), 'planned_budget': config['design']['budget'],
              'capture_authorized': False, 'execution_binding_complete': False,
              'actual_policy_optimizer_environment_initial_private_shadow_calls': 0}
    with args.output.open('x') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2); handle.write('\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
