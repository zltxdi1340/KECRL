"""Frozen metadata and access budget for cached task7 lag/neighbor decomposition."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from experiments import libero_task7_component3_history_transition_protocol as old

p = old.p
ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-lag-neighbor-decomposition-readonly_v2.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-lag-neighbor-decomposition-execution-binding_v1.json'
BASE = Path('/tmp/libero-task7-component3-history-transition-artifacts-20261108-v1')
PARENTS = {
    'transition_science': {'path': str(old.DEFAULT_CONFIG), 'sha256': '3731985fc975ce88ace5dad55bbf5c5f53304463a6c7d5ff6b6375a5aadf4962'},
    'transition_binding': {'path': str(old.DEFAULT_BINDING), 'sha256': 'cdcb6ee254ed67641203a5086a9eb015aee90be8cad59b66a295e639d3e6055f'},
    'transition_result': {'path': old.OUTPUT, 'sha256': '723fa85815f0b95d7096af5d222ef5a57d6a59e7ea89d5a5c56babf4176de741'},
    'transition_audit': {'path': str(BASE / 'independent-audit.json'), 'sha256': '5b26df6f8befdce9297ea2d842c256fcfb23b9a35acd9d81b1d4b85938165c73'},
    'transition_completion': {'path': str(BASE / 'completion.json'), 'sha256': 'b0b2358fcfd5b4f559fbef3b582cf092949659191dcd9d76a0cbfd98096cb0b0'},
    'capture': old.PARENTS['capture'], 'readout': old.PARENTS['result'],
}
SCOPE = copy.deepcopy(old.SCOPE)
del SCOPE['new_feature_or_full_library_distance_computations']
SCOPE.update({'feature_width': 256, 'new_full_library_distances': 0, 'selected_pair_lag_distances_per_stage': 2158800,
              'new_optimizer_updates': 0, 'new_reference_or_scaler_selection': False})
DESIGN = {
    'cells': ['CC', 'HC', 'CH', 'HH'], 'cell_axes': 'first_distance_C_lag0_H_mean5; second_neighbor_set_C_oldcurrent_H_oldhistory',
    'sets': ['C', 'H'], 'lags': [0, 1, 2, 3, 4], 'neighbors_per_set': 10,
    'distance': 'float64_direct_squared_differences_64_coordinate_blocks; no_Gram',
    'normalization': 'same_original_leave_one_demo_population_mean_std_for_every_lag; inactive_std_le1e-6_replaced_by1',
    'scaler_check': 'recompute_other29demo_all_tge5_including_neutral; compare_all30folds_to_parent',
    'history': 'query_and_reference_same_demo_local_past_lag0to4; no_padding_future_actions_as_features',
    'prediction': 'gap=mean_negative5-minus-mean_positive5; positive_if_gap_gt0_else_negative',
    'endpoint_check': 'CC_matches_old_current_HH_matches_old_history; distances_abs1e-8; labels_and_ties_exact',
    'paths': ['temporal_on_C=HC-CC', 'neighbors_under_H=HH-HC',
              'neighbors_under_C=CH-CC', 'temporal_on_H=HH-CH'],
    'symmetric_decomposition': 'T=((HC-CC)+(HH-CH))/2; N=((CH-CC)+(HH-HC))/2; T+N=HH-CC',
    'interaction': 'HH-HC-CH+CC; algebra_not_environment_causal_effect',
    'lag_temporal_terms': '(gap_lag_l-gap_lag0)/5_for_l1to4; sum_equals_fixed_set_temporal_delta',
    'attribution': ['endpoint_same', 'temporal_only', 'neighbors_only', 'either_alone', 'joint_only'],
    'attribution_rule': 'for_CC_ne_HH: HC_and_CH_equal_HH_flags=(10,01,11,00); same_endpoints_keep_cross_flags',
    'original_transitions': old.ANALYSIS['transitions'], 'strata': ['all_common', 'tail20', 'complement'],
    'directions': ['negative', 'positive'], 'demonstrations': list(range(30)),
    'summaries': 'all_seed_models_counts_mean_median_q25_q75; empty_groups_counts0_rates_null',
    'score_comparison': 'within_same_model_fold_only_no_cross_model_absolute_gap_mean',
    'audit': 'independent_distance_recompute_abs1e-8_then_reconstruct_strict_decisions_from_verified_published_distances',
    'significance_or_automatic_training_promotion': False,
}
MEMBERS = {'shared': old.MEMBERS['shared'], 'probe': old.MEMBERS['probe'], 'feature': ['features']}
EXECUTION = copy.deepcopy(old.EXECUTION)
EXECUTION.update({'array_loads': 'exact13NPZ_once_each_per_stage_81_members; no_pickle_mmap_fromfile',
                  'input_hash_checks': 'thirteen_pinned_files_byte_SHA_before_and_after_not_array_numeric_decodes',
                  'audit_checks': ['all13files_and81members', 'all_selected_pair_lag_distances', 'all_six_CSV_tables',
                                   'all_original_diagonal_distances_predictions_ties_and_scalers'],
                  'audit_float_absolute_tolerance': 1e-8, 'scaler_absolute_tolerance': 1e-10,
                  'CSV_bytes_max': 268435456, 'query_chunk': 64, 'coordinate_block': 64})
OUTPUT = '/tmp/libero-task7-lag-neighbor-decomposition-20261108-v1.json'
ARTIFACTS = '/tmp/libero-task7-lag-neighbor-decomposition-artifacts-20261108-v1'
BINDING_FILES = {
    'analysis': ROOT / 'experiments/analyze_libero_task7_lag_neighbor_decomposition.py',
    'audit': ROOT / 'experiments/audit_libero_task7_lag_neighbor_decomposition.py',
    'guard': ROOT / 'experiments/libero_task7_lag_neighbor_decomposition_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_lag_neighbor_decomposition.py',
    'metadata': Path(__file__).resolve(), 'metadata_tests': ROOT / 'tests/test_libero_task7_lag_neighbor_decomposition_protocol.py',
    'protocol': ROOT / 'docs/libero-task7-lag-neighbor-decomposition-plan-2026-11-08.md',
}


def budget():
    return {'models': 6, 'shared_rows': 4188, 'common_material_rows': 3598, 'tail20_material_rows': 477,
            'per_stage': {'array_files': 13, 'array_members': 81, 'feature_files': 6, 'shared_files': 1, 'probe_files': 6,
                          'paired_lag_distances': 2158800, 'coordinate_differences': 552652800,
                          'scaler_fit_rows': 707832, 'scaler_fit_coordinates': 181204992,
                          'fold_normalized_coordinates': 192983040},
            'total_array_loads': 26, 'total_member_decodes': 162,
            'tables': {'distances': 21588, 'frames': 21588, 'scores': 144, 'demo-scores': 4320,
                       'attribution': 720, 'lag-summary': 1440},
            'distance_CSV_scalars': 2158800, 'new_training_CNN_environment_nearest_neighbor_search_calls': 0}


def source_inputs(parents):
    base = parents['transition_science']['inputs']
    captured = {(r['task'], r['seed'], r['arm']): r for r in parents['capture']['models'] if r['task'] == 7}
    rows = []
    for model in base['models']:
        key = model['task'], model['seed'], model['arm']; capture = captured[key]
        p.require(old.exact(capture['checkpoint'], model['checkpoint']), 'feature endpoint lineage differs')
        rows.append({**model, 'feature': capture['features']})
    p.require(len(rows) == 6 and {(r['task'], r['seed'], r['arm']) for r in rows} ==
              {(7, s, a) for s in p.SEEDS for a in p.ARMS}, 'all six endpoints required')
    return {**base, 'models': rows}


def validate_design(config, parents):
    p.require(config['kind'] == 'task7_cached_lag_fixed_neighbor_decomposition' and config['formal_result'] is False
              and type(config.get('version')) is int and config['version'] == 2
              and config['status'] == 'scientific_frozen_execution_unbound' and old.exact(config['parents'], PARENTS), 'frozen identity differs')
    for key, expected in [('scope', SCOPE), ('design', DESIGN), ('members', MEMBERS), ('execution', EXECUTION), ('budget', budget())]:
        p.require(old.exact(config[key], expected), 'frozen ' + key + ' differs')
    p.require(parents['transition_audit']['status'] == 'passed_independent_task7_history_transition_audit'
              and parents['transition_completion']['status'] == 'complete_independently_audited_task7_history_transitions'
              and parents['transition_audit']['result'] == parents['transition_completion']['result'] == PARENTS['transition_result']
              and parents['transition_completion']['audit'] == PARENTS['transition_audit'], 'audited transition parent required')
    p.require(old.exact(config['inputs'], source_inputs(parents)) and config['output'] == OUTPUT and config['artifacts'] == ARTIFACTS,
              'fixed input/output identities differ')
    refs = [config['inputs']['shared']] + [r[k] for r in config['inputs']['models'] for k in ('archive', 'feature')]
    p.require(len({r['path'] for r in refs}) == 13, 'exactly13 distinct source arrays required')
    return True


def validate(path=DEFAULT_CONFIG):
    config = json.loads(Path(path).read_text())
    p.require(old.exact(config['parents'], PARENTS), 'parent pins differ')
    parents = {key: p.load_pin(ref) for key, ref in PARENTS.items()}
    validate_design(config, parents)
    old.validate(PARENTS['transition_science']['path'])
    for ref in config['code'] + [config['protocol']]: p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'frozen metadata/protocol changed')
    return config


def require_binding(path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING):
    config = validate(path)
    p.require(Path(binding_path).is_file(), 'execution binding required before array payload')
    binding = json.loads(Path(binding_path).read_text())
    p.require(binding['status'] == 'bound_before_payload' and binding['formal_result'] is False
              and binding['science_config'] == p.pin(path) and old.exact(binding['budget'], config['budget'])
              and set(binding['files']) == set(BINDING_FILES), 'binding identity/budget differs')
    for role, code in BINDING_FILES.items():
        p.require(p.resolve(binding['files'][role]['path']).resolve() == code and p.pin(code) == binding['files'][role], 'bound code role differs')
    for ref in binding['dependencies']: p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'bound dependency changed')
    return binding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--require-executable', action='store_true')
    args = parser.parse_args(); cfg = validate()
    if args.require_executable: require_binding()
    print(json.dumps({'status': cfg['status'], 'budget': cfg['budget'], 'actual_research_array_loads': 0}))
