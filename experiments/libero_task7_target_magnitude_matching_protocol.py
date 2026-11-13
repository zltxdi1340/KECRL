"""Metadata-only freeze: target-scale strata and within-demo output comparisons.

No real CSV projection or research numerical payload without a future binding.
Only standard-library synthetic helpers are evaluated during this freeze.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import struct

from experiments import libero_task7_residual_sign_transition_protocol as previous

p = previous.p
exact = previous.exact
ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-target-magnitude-matching-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-target-magnitude-matching-execution-binding_v1.json'
OUTPUT = '/tmp/libero-task7-target-magnitude-matching-20261108-v1.json'
ARTIFACTS = '/tmp/libero-task7-target-magnitude-matching-artifacts-20261108-v1'
PARENTS = {
    'science': {'path': str(previous.DEFAULT_CONFIG), 'sha256': '7de0233952c2913c50b59071d7af79706dc26477f4db635d1bd6606a7aac0c0a'},
    'binding': {'path': str(previous.DEFAULT_BINDING), 'sha256': '7804869d9df1c025e7ae1ff1a3da15f968468431141f51d16c1dbef885ceaa95'},
    'result': {'path': previous.OUTPUT, 'sha256': 'f702c18e6a1f61a66d88d53805dfcba95c5dacf1267b05c8d0b40974f4a11184'},
    'audit': {'path': str(Path(previous.ARTIFACTS)/'independent-audit.json'), 'sha256': '6fa7968780569c827a09f15dfd4c1db1c20fa30f6a0919761195ad652fe91bff'},
    'completion': {'path': str(Path(previous.ARTIFACTS)/'completion.json'), 'sha256': '56b83176cc33656611312f0a34f352ea793e9b2a96863f4567be6f8b528d4845'},
}
COMPARISONS = previous.COMPARISONS
PHASES = previous.PHASES
DIRECTIONS = previous.DIRECTIONS
TRANSITIONS = previous.TRANSITIONS
CONTRASTS = [['corrected', 'stays_wrong'], ['new_error', 'stays_correct']]
SCHEMES = ['exact_target', 'coarse_bin']
BIN_EDGES = [.025, .05, .1, .2, .4, .6, .8, 1.]
COLUMNS = ['task', 'seed', 'arm', 'comparison', 'candidate', 'reference', 'source_row', 'demo', 'action_step', 'target_sign', 'tail20',
           'target_2', 'reference_prediction_2', 'candidate_prediction_2', 'reference_category', 'candidate_category',
           'reference_clip', 'candidate_clip', 'transition', 'reference_error', 'candidate_error', 'reference_band', 'candidate_band',
           'reference_sse', 'candidate_sse', 'delta_sse', 'reference_raw_residual', 'candidate_raw_residual']
METRICS = ['target_abs', 'reference_abs', 'reference_abs_over_target_abs', 'reference_squared_error', 'candidate_squared_error',
           'delta_squared_error', 'reference_relative_squared_error', 'candidate_relative_squared_error', 'delta_relative_squared_error',
           'raw_shift', 'signed_effective_shift', 'absolute_effective_shift_over_target_abs']
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'feature_source_arms': p.ARMS, 'readout_modes': previous.MODES,
    'original_Policy_updates': 500, 'readout_endpoint_updates': 500, 'closed_loop_default_stabilization_steps': 5,
    'loss': previous.SCOPE['loss'], 'component_index': 2, 'source_role': 'meta_support_exposed_representation_OOF_readout',
    'encoder_has_seen_all30_support_demos': True, 'demonstration_indices': list(range(30)),
    'causal_alignment': 'obs[t-1] -> action[t]', 'action_step_min': 5, 'target_material_margin_inclusive': .025,
    'source_common_material_frames': 3598, 'new_research_training_optimizer_or_forward_calls': 0,
    'array_archive_checkpoint_HDF5_observation_feature_state_reads_including_byte_hashes': 0,
    'original_oof_frames_curves_NN_tags_or_other_source_CSV_reads': 0,
    'source_meta_query_accessed': False, 'sealed_roles_accessed': False, 'checkpoint_fold_seed_or_threshold_selection': False,
    'all_parent_cues_false_unchanged': True, 'G_failed_H_unrun_unchanged': True, 'SPT_Knowledge_Module_changes': False,
    'automatic_candidate_training_rollout_or_protocol_promotion': False,
}
DESIGN = {
    'question': 'reference_output_magnitude_association_after_controlling_observed_target_scale_within_demo_common_support',
    'comparisons_candidate_reference': COMPARISONS, 'phases': PHASES, 'directions': DIRECTIONS, 'all_transitions': TRANSITIONS,
    'contrasts_focus_control': CONTRASTS, 'schemes': SCHEMES,
    'primary': 'exact_target_all_common;preserve_tail20_complement_and_coarse_bin_as_secondary_diagnostics',
    'bin_edges': BIN_EDGES, 'bin_interval': 'left_closed_right_open;last_right_closed_at1;float64_decimal_edges_no_rounding_or_epsilon',
    'primary_key': 'within_endpoint_comparison_phase_direction_contrast:demo_and_exact_float32_abs_target_bits',
    'coarse_key': 'same_endpoint_comparison_phase_direction_contrast_and_demo_and_one_of7_fixed_bins',
    'matching': 'all_observed_focus_control_frames_in_cell;paired_iff_both_n_positive;no_one_to_one_or_nearest_neighbor_cartesian_expansion',
    'exact_cells': 'union_of_observed_focus_or_control_keys_only;retain_one_sided;no_empty_empty_exact_cell',
    'coarse_cells': 'all30_demos_by7_bins_including_empty_empty;within_bin_target_imbalance_is_not_exact_control',
    'weights': 'equal_matched_target_cells_within_each_demo_then_equal_eligible_demos;identical_weights_for_focus_and_control',
    'decomposition': 'full_frame_weighted_raw;common_support_frame_weighted;cell_equal_demo_equal;coverage_alongside_all3',
    'estimand_limit': 'standardization_changes_support_and_weights;raw_to_matched_attenuation_not_uniquely_target_scale_causation',
    'metrics': METRICS, 'relative_denominator': 'abs_target_ge.025_or_target_squared;no_zero_denominator_no_epsilon',
    'cell_checks': 'exact_key_target_imbalance0;coarse_report_focus_control_target_mean_min_max_and_delta;all_metrics_finite',
    'demo_missing': 'retain_all30;both_empty_focus_empty_control_empty_no_common_exact_target_or_no_common_target_bin',
    'coverage': 'full_and_matched_focus_control_frame_n;matched_cells_and_demos;unmatched_counts_rates_null_when_denominator0',
    'no_fallback': 'empty_common_support_means_null;no_caliper_relaxation_bin_merging_or_min_sample_threshold_search',
    'bin_transition_checks': 'all7bins_all4transitions_both_directions;count_SSE_and_net_error_conservation',
    'join_checks': 'source_row_demo_step_tail_from30lengths;all6models_by3comparisons_by3598_IDs_unique_complete;same_targets_and3mode_scalars_cross_comparison',
    'primitive_checks': 'saved_target_and_predictions_float32_range;strict_categories_signs_error_bands_SSE_delta;raw_residual_clip_reconstruction_using_original_mode',
    'frame_assignment': 'one_row_per_input_pair_fixed_bin_exact_target_bits_reference_correctness_and_exact_coarse_match_flags_all_common_and_own_phase',
    'cross_seed': 'per_arm_all3_raw_values_and_complete_mean;common3_matched_cell_intersection_secondary_recomputed_with_same_hierarchical_weights',
    'cross_seed_limit': 'same_demo_target_key_does_not_force_same_focus_control_frame_membership;A_B_not6_independent_seeds',
    'interpretation': 'descriptive_outcome_defined_groups_only;matching_does_not_control_unobserved_geometry_history_or_establish_cause_or_missing_information',
    'decision': 'no_new_acceptance_threshold_or_training_selection;no_pvalues_significance_bootstrap_or_frame_independence;parent_stop_stays',
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU', 'runtime': previous.EXECUTION['runtime'],
    'environment': previous.EXECUTION['environment'], 'analysis_active_seconds_max': 180, 'audit_active_seconds_max': 180,
    'overall_watchdog_seconds': 420, 'working_memory_bytes_max': 536870912, 'output_CSV_bytes_max': 268435456,
    'technical_retries': 0, 'output_overwrite': False,
    'metadata': 'five_pinned_parent_JSON_for_provenance_only;no_metric_payload_as_new_analysis_source',
    'access': 'one_pinned_derived_paired_frames_CSV_only_one_pass28_projected_fields_per_stage;deny_original_CSV_arrays_and_alternate_readers',
    'byte_SHA': 'only_this_input_CSV_before_after_separate_from_data_pass;never_hash_original_arrays_or_old_input_CSV',
    'audit': 'fresh_process_independent_numeric_ID_group_matching_weights_coverage_and_all_outputs;no_primary_numerics_import',
    'float_absolute_tolerance': 1e-10, 'float_relative_tolerance': 0, 'ID_counts_bits_saved_float32_and_categories_tolerance': 0,
    'stop': 'any_metadata_SHA_role_ID_projection_scalar_key_conservation_budget_or_audit_mismatch_keep_partial_stop_no_retry',
}
BINDING_FILES = {
    'analysis': ROOT/'experiments/analyze_libero_task7_target_magnitude_matching.py',
    'audit': ROOT/'experiments/audit_libero_task7_target_magnitude_matching.py',
    'guard': ROOT/'experiments/libero_task7_target_magnitude_matching_access.py',
    'runtime_tests': ROOT/'tests/test_libero_task7_target_magnitude_matching.py',
    'metadata': Path(__file__).resolve(), 'metadata_tests': ROOT/'tests/test_libero_task7_target_magnitude_matching_protocol.py',
    'protocol': ROOT/'docs/libero-task7-target-magnitude-matching-plan-2026-11-08.md',
}
CONTEXT_FILES = [BINDING_FILES['metadata'], BINDING_FILES['metadata_tests'], Path(previous.__file__).resolve(),
                 previous.BINDING_FILES['analysis'], previous.BINDING_FILES['audit'], Path(p.__file__).resolve()]


def budget():
    return {'models': 6, 'comparisons': 3, 'shared_material_frames': 3598, 'phases': 3, 'directions': 2,
            'contrasts': 2, 'schemes': 2, 'demos': 30, 'fixed_target_bins': 7,
            'per_stage': {'input_CSV_data_passes': 1, 'input_CSV_lexical_rows': 64764, 'projected_columns': 28, 'projected_cells': 1813392},
            'total_input_CSV_data_passes': 2, 'total_input_CSV_lexical_rows': 129528, 'total_projected_cells': 3626784,
            'fixed_table_rows': {'frame-strata': 64764, 'bin-transitions': 3024, 'coarse-bin-cells': 45360,
                                 'demo-matched': 12960, 'model-matched': 432, 'common3-summary': 144},
            'variable_table_rows_max': {'exact-target-cells': 129528}, 'total_output_table_rows_max': 256212,
            'exact_cell_bound': 'sum_union_focus_control_target_keys_le_material_frame_visits;6models*3comparisons*2phase_visits*3598',
            'seed_summary_JSON_rows': 144, 'new_original_CSV_array_archive_weight_or_observation_reads_including_hashes': 0,
            'new_training_forward_optimizer_environment_neighbor_or_sealed_calls': 0}


def finite(value): return type(value) in (int, float) and math.isfinite(value)


def target_bin(target):
    p.require(finite(target) and .025 <= abs(target) <= 1, 'material finite target required')
    value = abs(target)
    return next(i for i, (left, right) in enumerate(zip(BIN_EDGES, BIN_EDGES[1:])) if left <= value and (value < right or i == 6 and value <= right))


def target_bits(target):
    p.require(finite(target) and .025 <= abs(target) <= 1 and previous.float32(target) == target, 'exact saved float32 material target required')
    return struct.unpack('<I', struct.pack('<f', abs(target)))[0]


def match_key(demo, target, scheme):
    p.require(type(demo) is int and 0 <= demo < 30 and scheme in SCHEMES, 'matching role invalid')
    bits = target_bits(target)
    return (demo, 1 if target > 0 else -1, bits if scheme == 'exact_target' else target_bin(target))


def frame_metrics(target, reference, candidate, raw_reference, raw_candidate):
    target_bits(target)
    p.require(all(finite(v) for v in (reference, candidate, raw_reference, raw_candidate))
              and all(abs(v) <= 1 and previous.float32(v) == v for v in (reference, candidate)), 'finite saved float32 actions required')
    y = abs(target); r = (reference-target)**2; c = (candidate-target)**2; shift = candidate-reference
    values = [y, abs(reference), abs(reference)/y, r, c, c-r, r/(y*y), c/(y*y), (c-r)/(y*y),
              raw_candidate-raw_reference, (1 if target>0 else -1)*shift, abs(shift)/y]
    p.require(all(finite(v) for v in values), 'nonfinite derived metric')
    return dict(zip(METRICS, values))


def compare_cell(focus, control, scheme, cell_key):
    p.require(scheme in SCHEMES and type(cell_key) is int, 'cell scheme/key invalid')
    if scheme == 'coarse_bin': p.require(0 <= cell_key < 7, 'coarse cell key invalid')
    rows = focus + control
    p.require(rows or scheme == 'coarse_bin', 'exact cell requires observed target')
    for row in rows:
        p.require(set(row) == set(METRICS) and all(finite(v) for v in row.values()), 'cell metrics invalid')
        value = row['target_abs']; actual = target_bits(value) if scheme == 'exact_target' else target_bin(value)
        p.require(actual == cell_key, 'cell target magnitude key mismatch')
    nf, nc = len(focus), len(control); paired = bool(nf and nc)
    reason = '' if paired else 'both_empty' if nf == nc == 0 else 'focus_empty' if nf == 0 else 'control_empty'
    row = {'scheme': scheme, 'cell_key': cell_key, 'focus_n': nf, 'control_n': nc, 'matched': paired, 'missing_reason': reason}
    for name, values in [('focus', focus), ('control', control)]:
        row[name] = {key: math.fsum(v[key] for v in values)/len(values) if values else None for key in METRICS}
        row[name+'_target_min'] = min(v['target_abs'] for v in values) if values else None
        row[name+'_target_max'] = max(v['target_abs'] for v in values) if values else None
    row['delta'] = {key: row['focus'][key]-row['control'][key] if paired else None for key in METRICS}
    if paired and scheme == 'exact_target': p.require(row['delta']['target_abs'] == 0, 'exact target imbalance')
    return row


def pooled(cells, group):
    n = sum(row[group+'_n'] for row in cells)
    return {key: math.fsum(row[group+'_n'] * row[group][key] for row in cells if row[group+'_n'])/n if n else None for key in METRICS}


def summarize_cells(cells, scheme):
    p.require(scheme in SCHEMES and all(row['scheme'] == scheme for row in cells)
              and len({row['cell_key'] for row in cells}) == len(cells), 'mixed or duplicated cell keys')
    matched = [row for row in cells if row['matched']]; nf = sum(row['focus_n'] for row in cells); nc = sum(row['control_n'] for row in cells)
    reason = '' if matched else 'both_empty' if nf == nc == 0 else 'focus_empty' if nf == 0 else 'control_empty' if nc == 0 else ('no_common_exact_target' if scheme == 'exact_target' else 'no_common_target_bin')
    row = {'scheme': scheme, 'cell_n': len(cells), 'matched_cell_n': len(matched), 'focus_total_n': nf, 'control_total_n': nc,
           'focus_matched_n': sum(r['focus_n'] for r in matched), 'control_matched_n': sum(r['control_n'] for r in matched), 'missing_reason': reason}
    for name, n in [('focus', nf), ('control', nc)]:
        row[name+'_unmatched_n'] = n-row[name+'_matched_n']; row[name+'_coverage'] = row[name+'_matched_n']/n if n else None
    for name, use in [('pooled_full', cells), ('pooled_common', matched)]:
        f, c = pooled(use, 'focus'), pooled(use, 'control')
        row[name] = {'focus': f, 'control': c, 'delta': {key: f[key]-c[key] if f[key] is not None and c[key] is not None else None for key in METRICS}}
    row['equal_cell'] = {group: {key: math.fsum(r[group][key] for r in matched)/len(matched) if matched else None for key in METRICS}
                         for group in ('focus', 'control', 'delta')}
    return row


def summarize_demos(demos):
    p.require(len(demos) == 30 and {row['demo'] for row in demos} == set(range(30))
              and all(type(row['demo']) is int for row in demos) and len({row['scheme'] for row in demos}) == 1, 'all30 unique same-scheme demos required')
    use = [row for row in demos if row['matched_cell_n'] > 0]
    output = {'eligible_demo_n': len(use), 'ineligible_demo_n': 30-len(use), 'matched_cell_n': sum(row['matched_cell_n'] for row in use)}
    for group in ('focus', 'control'):
        total = sum(row[group+'_total_n'] for row in demos); matched = sum(row[group+'_matched_n'] for row in demos)
        output[group+'_total_n'] = total; output[group+'_matched_n'] = matched; output[group+'_unmatched_n'] = total-matched
        output[group+'_coverage'] = matched/total if total else None
    for scope, suffix in [('pooled_full', '_total_n'), ('pooled_common', '_matched_n')]:
        output[scope] = {}
        for group in ('focus', 'control'):
            n = output[group+suffix]
            output[scope][group] = {key: math.fsum(row[group+suffix] * row[scope][group][key] for row in demos if row[group+suffix])/n if n else None for key in METRICS}
        output[scope]['delta'] = {key: output[scope]['focus'][key]-output[scope]['control'][key]
                                 if output[scope]['focus'][key] is not None and output[scope]['control'][key] is not None else None for key in METRICS}
    output['standardized'] = {group: {key: math.fsum(r['equal_cell'][group][key] for r in use)/len(use) if use else None for key in METRICS}
                              for group in ('focus', 'control', 'delta')}
    return output


def common3_keys(sets):
    p.require(len(sets) == 3 and all(type(value) is set for value in sets), 'all3 seed key sets required')
    return set.intersection(*sets)


def select_inputs(parents):
    source = parents['science']['inputs']; ref = parents['result']['tables']['paired-frames']
    p.require(ref['path'] == str(Path(previous.ARTIFACTS)/'paired-frames.csv') and type(ref['rows']) is int and ref['rows'] == 64764
              and len(ref['columns']) == len(set(ref['columns'])) and set(COLUMNS).issubset(ref['columns']), 'derived input CSV identity/schema differs')
    endpoints = [{key: row[key] for key in ('task', 'seed', 'arm', 'checkpoint')} for row in source['endpoints']]
    p.require(len(endpoints) == 6 and {(r['task'],r['seed'],r['arm']) for r in endpoints} == {(7,seed,arm) for seed in p.SEEDS for arm in p.ARMS}, 'all six fixed endpoints required')
    previous.source_layout(source['action_lengths'])
    return {'paired_frames': ref, 'endpoints': endpoints, 'action_lengths': source['action_lengths']}


def validate_design(cfg, parents):
    p.require(cfg['kind'] == 'task7_derived_CSV_target_magnitude_within_demo_matching' and type(cfg['version']) is int and cfg['version'] == 1
              and cfg['formal_result'] is False and cfg['status'] == 'scientific_frozen_execution_unbound' and exact(cfg['parents'], PARENTS), 'science identity differs')
    for key, value in [('scope', SCOPE), ('design', DESIGN), ('execution', EXECUTION), ('budget', budget()), ('columns', COLUMNS)]:
        p.require(exact(cfg[key], value), 'frozen '+key+' differs')
    p.require(all(row['formal_result'] is False for row in parents.values()), 'diagnostic parents required')
    science, result, audit, done = (parents[k] for k in ('science', 'result', 'audit', 'completion'))
    bound = parents['binding']
    p.require(bound['status'] == 'bound_before_payload' and bound['before_research_payload'] is True
              and bound['science_config'] == PARENTS['science'] and exact(bound['budget'], previous.budget())
              and bound['future_output'] == previous.OUTPUT and bound['future_artifacts'] == previous.ARTIFACTS, 'parent execution identity differs')
    p.require(result['status'] == 'complete_task7_residual_sign_transition_pending_audit'
              and audit['status'] == 'passed_independent_task7_residual_sign_transition_audit'
              and done['status'] == 'complete_independently_audited_task7_residual_sign_transition'
              and result['science_config'] == audit['science_config'] == done['science_config'] == PARENTS['science']
              and result['binding'] == audit['binding'] == done['binding'] == PARENTS['binding']
              and audit['result'] == done['result'] == PARENTS['result'] and done['audit'] == PARENTS['audit']
              and result['inputs'] == science['inputs'] and audit['tables_verified'] == result['tables'], 'independently audited parent required')
    p.require(all(result[key] is True and done[key] is True for key in ('all_parent_joint_cues_false_unchanged','G_failed_H_unrun_unchanged'))
              and result['SPT_Knowledge_Module_changes'] is done['SPT_Knowledge_Module_changes'] is False
              and audit['primary_numerical_implementation_imported'] is False and audit['scalar_checks'] == 6300492, 'parent stop/independence boundary differs')
    p.require(all(type(row[key]) is int and row[key] == 0 for row in (result,audit,done)
                  for key in ('archive_array_checkpoint_reads_including_byte_hashes','new_training_forward_environment_neighbor_query_sealed_calls'))
              and done['total_input_CSV_data_passes'] == 4 and done['total_input_CSV_projected_cells'] == 2662128
              and done['output_rows_verified'] == 115728 and done['seed_summary_rows_verified'] == 72, 'parent access/completion counters differ')
    p.require(science['scope']['source_meta_query_accessed'] is False and science['scope']['sealed_roles_accessed'] is False
              and science['scope']['encoder_has_seen_all30_support_demos'] is True
              and science['scope']['original_Policy_updates'] == science['scope']['readout_endpoint_updates'] == 500, 'source role or fixed endpoint differs')
    p.require(exact(cfg['inputs'], select_inputs(parents)) and cfg['output'] == OUTPUT and cfg['artifacts'] == ARTIFACTS, 'derived input/output identities differ')
    return True


def validate(path=DEFAULT_CONFIG, *, check_csv_byte_hash=False):
    cfg = json.loads(Path(path).read_text()); p.require(exact(cfg['parents'], PARENTS), 'immutable parent pins differ')
    parents = {key: p.load_pin(ref) for key, ref in PARENTS.items()}; validate_design(cfg, parents)
    p.require(len(cfg['code']) == len(CONTEXT_FILES) and {p.resolve(ref['path']).resolve() for ref in cfg['code']} == set(CONTEXT_FILES)
              and p.resolve(cfg['protocol']['path']).resolve() == BINDING_FILES['protocol'], 'context/protocol identity differs')
    for ref in cfg['code'] + [cfg['protocol']]: p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'frozen code/protocol changed')
    if check_csv_byte_hash:
        ref = cfg['inputs']['paired_frames']; p.require(p.sha(ref['path']) == ref['sha256'], 'derived input CSV bytes changed')
    return cfg


def require_binding(path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    cfg = validate(path); p.require(Path(binding_path).is_file(), 'execution binding required before real CSV projection; scientific freeze only')
    bound = json.loads(Path(binding_path).read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['formal_result'] is False and bound['before_research_payload'] is True
              and bound['science_config'] == p.pin(path) and exact(bound['budget'], budget()) and set(bound['files']) == set(BINDING_FILES), 'execution identity/budget differs')
    for role, file in BINDING_FILES.items(): p.require(bound['files'][role] == p.pin(file), 'execution role/code changed')
    return bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--check-csv-byte-hash', action='store_true'); parser.add_argument('--require-executable', action='store_true'); args = parser.parse_args()
    cfg = validate(check_csv_byte_hash=args.check_csv_byte_hash)
    if args.require_executable: require_binding()
    print(json.dumps({'status': cfg['status'], 'future_input_CSV_passes': 2, 'future_projected_cells': 3626784,
                      'actual_real_CSV_projection_rows': 0, 'actual_original_CSV_array_or_checkpoint_reads_including_hashes': 0,
                      'actual_research_training_forward_environment_or_sealed_calls': 0}, ensure_ascii=False))
