"""Metadata-only freeze: legal-input discrimination of needed action correction."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from experiments import libero_source_expert_late_translation_protocol as p

ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-needed-direction-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-needed-direction-execution-binding_v1.json'
OUTPUT = '/tmp/libero-task7-needed-direction-20261109-v1.json'
ARTIFACTS = '/tmp/libero-task7-needed-direction-artifacts-20261109-v1'
PARENTS = {
    'source_science': {'path': str(ROOT / 'configs/libero-task7-current-residual-correctability-diagnostic_v1.json'), 'sha256': '722b626c34f7faff67356b3d7e385784536aad4b95abf80084f685079855c58e'},
    'geometry_result': {'path': '/tmp/libero-task7-protection-residual-decomposition-20261108-v1.json', 'sha256': '74f1315025fcd789e64b9bc70a727cb2235826d2240adffda3ceb72d1b2db3ee'},
    'geometry_audit': {'path': '/tmp/libero-task7-protection-residual-decomposition-artifacts-20261108-v1/independent-audit.json', 'sha256': '8aae93307aab1e65d810044de46bb10add96258a08ac15114cb27771d24f2c88'},
    'geometry_completion': {'path': '/tmp/libero-task7-protection-residual-decomposition-artifacts-20261108-v1/completion.json', 'sha256': 'c5787ae0f9f24f3e27ccf928a2db9adba2f7286700133cd22600e830bee5d4ba'},
    'previous_binding': {'path': str(ROOT / 'configs/libero-task7-protection-residual-decomposition-execution-binding_v1.json'), 'sha256': '72fe68e5fd5121c882b41f6be76016801ac220f9e8993c0e017a5ba45885d3c1'},
}
MEMBERS = {'shared': ['targets', 'demo_indices', 'action_steps', 'demo_lengths', 'tail20'],
           'current': ['features', 'predictions']}
VIEWS = ['p0_only', 'current_only', 'current_plus_p0']
READOUTS = ['fit_majority', 'always_hold'] + VIEWS
CLASSES = [-1, 0, 1]
CLASS_NAMES = ['negative', 'hold', 'positive']
TIE_ORDER = [0, -1, 1]
PHASES = ['all_common', 'tail20', 'complement']
COHORTS = ['all', 'expert_negative', 'expert_positive', 'expert_neutral', 'original_good', 'original_wrong']
SCOPE = {
    'task_ids': [7], 'seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'original_Policy_endpoint_updates': 500, 'future_stabilization_steps': 5,
    'original_loss_unchanged': True, 'source_role': 'meta_support', 'demonstrations': list(range(30)),
    'encoder_has_seen_all30': True, 'encoder_unseen_generalization': False,
    'new_optimizer_updates': 0, 'new_Policy_CNN_environment_initial_state_calls': 0,
    'checkpoint_head_CSV_HDF5_RGB_proprio_history_or_private_reads': 0,
    'source_meta_query_validation_heldout_qualification_reserved_final_reads': 0,
    'G_failed_H_unrun_unchanged': True, 'automatic_training_or_Policy_promotion': False,
    'SPT_Knowledge_Module_changes': False,
}
DESIGN = {
    'hypothesis': 'current_features_add_needed_correction_direction_discrimination_beyond_saved_p0_only',
    'target': 'r=float64(y[:,2])-float64(p0[:,2]);NOT_expert_action_sign',
    'labels': 'negative_if_r<=-.025;positive_if_r>=.025;otherwise_hold;inclusive_material_threshold',
    'threshold': .025, 'threshold_search': False, 'component_index': 2,
    'features': 'frozen_current256_float32; saved_p0_component2_float32; no_new_forward',
    'causal_alignment': 'obs[t-1]->action[t];fit_tge1;common_probe_tge5',
    'fold_probe_demos': [list(range(n, n + 10)) for n in (0, 10, 20)],
    'bank': 'all_tge1_rows_in_other20_demos;all3_classes_min5_each_or_fail_closed',
    'scaler': 'float64_fit_only_mean_population_std;std<=1e-6_set1;features_and_p0_separately',
    'gate': 'all18_fit_banks_and_scalers_complete_before_any_probe_distance_or_truth_scoring',
    'full_cache_decode': 'contains_probe_bytes;only_fit_rows_enter_bank_labels_or_scalers;no_unread_probe_claim',
    'readouts': READOUTS, 'views': VIEWS,
    'rule': 'for_each_class_take5_nearest;minimum_mean_squared_distance_class_wins',
    'distances': 'p0_only:scalar_squared;current_only:mean256_squared;combined:(p0_only+current_only)/2',
    'feature_distance': 'direct_float64_coordinate_differences;main_sum64feature_blocks;audit_sum32feature_blocks;no_Gram_cancellation',
    'k': 5, 'query_chunk': 16,
    'neighbor_ties': 'stable_ascending_raw_source_row',
    'class_ties': 'exact_float64_equality;hold_then_negative_then_positive;no_epsilon_relabel',
    'references': 'fit_majority_class;exact_count_tie_hold_negative_positive;always_hold',
    'leakage': 'query_inference_receives_only_features_and_p0;no_y_residual_label_phase_or_good_mask',
    'information_limit': 'p0_is_deterministic_readout_of_current_features;different_metric_geometry_not_independent_new_information_or_capacity_cause',
    'scoring': 'all_common_tge5;tail20_last20;complement;each_raw_query_scored_once_per_readout',
    'cohorts': COHORTS, 'cohort_rule': 'expert_material_.025;good:material_and_(p0-y)^2<y^2;wrong:material_and_sign(p0)!=sign(y)',
    'metrics': '3x3_confusion;class_counts_recalls;all3_macro_recall_else_null;accuracy;opposite_correction_direction_errors;hold_false_activation;material_as_hold',
    'summaries': 'all6_endpoints_all30demos_all3folds;frame_pooled_primary;demo_equal_descriptive;all3raw_seed_values_mean_median_delta_counts',
    'cue': 'per_arm_all3seeds:combined_vs_p0_macro_strictly_up_common_and_tail;nondecrease_complement;all3_class_recalls_nondecrease_all3phases;allclasses_present;independent_prediction_and_neighbor_ID_agreement_exact',
    'stop': 'no_class_support_or_hash_role_schema_budget_nonfinite_numeric_or_audit_mismatch_stop_no_retry;cue_false_end_this_probe_no_k_threshold_metric_search',
    'interpretation': 'bounded_support_auxiliary_discriminability_only;not_continuous_correction_accuracy_closed_loop_gain_unique_root_cause_or_SPISPT_acceptance',
}
EXECUTION = {
    'device': 'CPU', 'runtime': {'python': '3.10.22', 'numpy': '1.22.4'},
    'environment': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    'seconds_per_stage_max': 180, 'overall_seconds_max': 420,
    'working_memory_bytes_max': 536870912, 'output_CSV_bytes_max': 134217728,
    'source_access': 'each_stage7NPZ17members_once;SHA_before_load_and_after_stage;main_plus_audit14loads34members',
    'output_access': 'audit_each_new_CSV_once;JSON_metadata_not_payload;no_output_NPZ',
    'audit': 'separate_process_independent_distance_scaler_selection_labels_metrics_cue;no_primary_numeric_import_or_Torch',
    'distance_and_metric_absolute_tolerance': 1e-10, 'relative_tolerance': 0,
    'predictions_neighbors_counts_roles_and_tie_flags_tolerance': 0,
    'technical_retries': 0, 'overwrite': False, 'binding_required_before_source_byte_hash_or_decode': True,
}
FILES = {
    'runner': ROOT / 'experiments/run_libero_task7_needed_direction.py',
    'audit': ROOT / 'experiments/audit_libero_task7_needed_direction.py',
    'guard': ROOT / 'experiments/libero_task7_needed_direction_access.py',
    'metadata': Path(__file__).resolve(),
    'tests': ROOT / 'tests/test_libero_task7_needed_direction.py',
    'plan': ROOT / 'docs/libero-task7-needed-direction-plan-2026-11-09.md',
    'contract': ROOT / 'docs/libero-task7-needed-direction-execution-contract-2026-11-09.md',
}


def select_inputs(source):
    models = [{k: m[k] for k in ('task', 'seed', 'arm', 'checkpoint', 'current_archive')}
              for m in source['inputs']['models']]
    p.require([(m['task'], m['seed'], m['arm']) for m in models] ==
              [(7, s, a) for s in p.SEEDS for a in p.ARMS], 'endpoint order/coverage differs')
    return {'shared': source['inputs']['shared'], 'models': models,
            'action_lengths': source['inputs']['action_lengths']}


def fold_plan(lengths):
    p.require(len(lengths) == 30 and all(type(n) is int and n >= 25 for n in lengths)
              and sum(n - 1 for n in lengths) == 4188, 'source lengths differ')
    ids = [(d, s) for d, n in enumerate(lengths) for s in range(1, n)]
    return [{'fold': f, 'fit_rows': [i for i, (d, _) in enumerate(ids) if d not in demos],
             'probe_rows': [i for i, (d, s) in enumerate(ids) if d in demos and s >= 5]}
            for f, demos in enumerate(DESIGN['fold_probe_demos'])]


def budget(lengths):
    pairs = sum(len(f['fit_rows']) * len(f['probe_rows']) for f in fold_plan(lengths)) * 6
    return {'endpoints': 6, 'rows_per_endpoint': 4188, 'common_per_endpoint': 4068,
            'fit_banks_per_stage': 18, 'fit_rows_per_stage': 50256, 'scalar_scaler_coordinates_per_stage': 12915792,
            'pair_distances_per_block_per_stage': pairs, 'coordinate_squared_differences_per_stage': pairs * 257,
            'neighbor_classifications_per_stage': 73224, 'reference_classifications_per_stage': 48816,
            'source_NPZ_per_stage': 7, 'source_members_per_stage': 17, 'total_source_NPZ_loads': 14,
            'total_source_member_decodes': 34, 'source_byte_SHA_checks_per_stage': 14,
            'new_optimizer_Policy_CNN_environment_sealed_calls': 0,
            'tables': {'frames': 73224, 'groups': 540, 'demo-groups': 16200, 'fold-groups': 1620},
            'seed_comparisons': 36, 'arm_cues': 2}


def create():
    parents = {name: p.load_pin(ref) for name, ref in PARENTS.items()}
    p.require(parents['geometry_audit']['status'].startswith('passed') and
              parents['geometry_completion']['status'].startswith('complete'), 'parent audit incomplete')
    inputs = select_inputs(parents['source_science'])
    return copy.deepcopy({'kind': 'task7_needed_direction_diagnostic', 'version': 1, 'formal_result': False,
            'status': 'frozen_execution_binding_required', 'parents': PARENTS, 'scope': SCOPE,
            'design': DESIGN, 'execution': EXECUTION, 'members': MEMBERS,
            'inputs': inputs, 'budget': budget(inputs['action_lengths']),
            'metadata': p.pin(FILES['metadata']), 'plan': p.pin(FILES['plan']),
            'output': OUTPUT, 'artifacts': ARTIFACTS})


def validate(path=DEFAULT_CONFIG):
    value = json.loads(Path(path).read_text())
    p.require(json.dumps(value, sort_keys=True, allow_nan=False) == json.dumps(create(), sort_keys=True, allow_nan=False),
              'frozen design/provenance/inputs/budget differ')
    p.require(len({value['inputs']['shared']['path']} | {m['current_archive']['path'] for m in value['inputs']['models']}) == 7, 'source overlap')
    return value


def require_binding(config=DEFAULT_CONFIG, binding=DEFAULT_BINDING):
    bound = json.loads(Path(binding).read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['before_research_payload'] is True
              and bound['science_config'] == p.pin(config), 'unbound or changed scientific configuration')
    p.require(bound['budget'] == budget(validate(config)['inputs']['action_lengths']) and
              bound['CPU_environment'] == EXECUTION['environment'] and
              bound['future_output'] == OUTPUT and bound['future_artifacts'] == ARTIFACTS, 'binding contract differs')
    refs = bound['dependencies']; paths = [p.resolve(r['path']).resolve() for r in refs]
    p.require(len(set(paths)) == len(paths) and set(FILES.values()).issubset(paths), 'code role coverage differs')
    inherited = p.load_pin(PARENTS['previous_binding'])['dependencies']
    p.require(all(ref in refs for ref in inherited), 'transitive parent code coverage differs')
    for ref in refs:
        p.require(p.resolve(ref['path']).suffix not in ('.npz', '.npy', '.csv', '.pt', '.pth', '.h5', '.hdf5'), 'payload in code closure')
        p.require(p.pin(p.resolve(ref['path'])) == ref, 'bound dependency changed')
    return bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--write-freeze', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        with DEFAULT_CONFIG.open('x') as f: json.dump(create(), f, ensure_ascii=False, indent=2); f.write('\n')
    else:
        validate(DEFAULT_CONFIG)
    print(json.dumps({'status': 'metadata_valid_no_payload', 'config': p.pin(DEFAULT_CONFIG)} if not args.write_freeze else {'status': 'metadata_frozen_no_payload'}))
