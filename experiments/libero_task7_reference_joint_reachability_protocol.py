"""Metadata/human freeze of finite, fixed-representation four-reference witnesses.

No optimizer, research payload loader, numerical runner, or execution authority.
A negative grid result never establishes global infeasibility.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments import libero_task7_joint_t_tradeoff_gradient_protocol as evidence

p, ROOT, IDS = evidence.p, evidence.ROOT, evidence.IDS
STEM = 'libero-task7-reference-joint-reachability'
CONFIG = ROOT / f'configs/{STEM}-diagnostic_v1.json'
FREEZE_BINDING = ROOT / f'configs/{STEM}-freeze-binding_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT / f'docs/{STEM}-plan-2026-11-11.md'
TESTS = ROOT / 'tests/test_libero_task7_reference_joint_reachability_protocol.py'
ARTIFACTS = Path(f'/tmp/{STEM}-20261111-v1')
REFERENCES = ['control', 'margin', 'zero', 'bisector']
ROLES = ['P_numpy', 'P_torch', 'U_torch', 'U_numpy']
TOL = 1e-10
PARENTS = {
    'science': {'path': str(evidence.CONFIG), 'sha256': '4323f4bd3fc3542a359e96c1309e576d164c4159917955da47820e7a4a2a93de'},
    'binding': {'path': str(evidence.BINDING), 'sha256': 'c83e6ff1e2775f462986451c59f037540fa756685d197aaf4db40bda368a390e'},
    'completion': {'path': str(evidence.ARTIFACTS / 'completion.json'), 'sha256': 'fd5f972a4092ea74eb026aed4ba374ada4886b81e9488381b0f7cd4622eebf48'},
    'process': {'path': '/tmp/libero-task7-joint-t-tradeoff-gradient-20261111-v1-process.json', 'sha256': '7cff0ca7892c4876357bc782527d8daa1e24879d5cd63e2b5ea68d58f38c42e2'},
    'report': {'path': str(ROOT / 'docs/libero-task7-joint-t-tradeoff-gradient-results-2026-11-11.md'), 'sha256': 'de4782eed48073aed3321128c11dcf51b72a5ecb4824dcccf060bee7d0b118ed'},
}
SOURCE_JSON = {
    'latest': copy.deepcopy(evidence.INPUTS),
    'stepwise': {
        'P': {'path': '/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/primary-result.json', 'sha256': '3de285874988df32ef5e7114c3ed4cd6c604a5bb042f45cb39f9596b074b7cde'},
        'U': {'path': '/tmp/libero-task7-stepwise-m-nonincrease-acceptance-artifacts-20261111-v1/independent-audit.json', 'sha256': 'cf9af192c284d13bda471357e6f0d065b5ba71034c8f0fc7cb29e786bdd3295c'},
    },
    'zero_T': {
        'P': {'path': '/tmp/libero-task7-t-priority-direction-artifacts-20261111-v1/primary-result.json', 'sha256': 'bc1997a63248460bfcbdeb7b5be1db55ae038200900544fbc0d1f5507c030cf9'},
        'U': {'path': '/tmp/libero-task7-t-priority-direction-artifacts-20261111-v1/independent-audit.json', 'sha256': 'bcd3fcaa30b69d1604cdb8a377b8cd0df7fb828580300c4acf663ec3896f8e4d'},
    },
}
SOURCE_ROLES = {
    'latest': dict(science=evidence.PARENTS['science'], binding=evidence.PARENTS['binding'],
        statuses=['complete_interior_joint_T_pending_same_parameter_audit', 'passed_independent_interior_joint_T_direction_audit'],
        conditions=['saved500', 'stepwise_M_nonincrease_guarded_margin_final', 'interior_joint_T_under_original_M_bound_final'],
        select={'control': 'stepwise_M_nonincrease_guarded_margin_final'}),
    'stepwise': dict(
        science={'path': str(ROOT / 'configs/libero-task7-stepwise-m-nonincrease-acceptance-diagnostic_v1.json'), 'sha256': 'a37d3433274f0d76a4db22d283a0059878a343b648f2c1256071fdf7a5afee18'},
        binding={'path': str(ROOT / 'configs/libero-task7-stepwise-m-nonincrease-acceptance-execution-binding_v1.json'), 'sha256': 'd7639a498b1a3b4b5ff9f69ed304456aaf7da2384c70e793a09790284227e04c'},
        statuses=['complete_stepwise_M_gate_pending_same_parameter_audit', 'passed_independent_stepwise_M_gate_audit'],
        conditions=['saved500', 'norm_matched_T_priority_M_margin_final', 'stepwise_M_nonincrease_guarded_margin_final'],
        select={'margin': 'norm_matched_T_priority_M_margin_final'}),
    'zero_T': dict(
        science={'path': str(ROOT / 'configs/libero-task7-t-priority-direction-diagnostic_v1.json'), 'sha256': '335d7f2e310af5b97ad2e5fbaf406f74beecfe1fd7b0a14ed96c9e5f450b0c97'},
        binding={'path': str(ROOT / 'configs/libero-task7-t-priority-direction-execution-binding_v1.json'), 'sha256': 'd7cf2888b0e516212dce953055d6986973973a52275aa704d4ec76a32ae71d9d'},
        statuses=['complete_T_priority_pending_same_parameter_audit', 'passed_independent_T_priority_direction_audit'],
        conditions=['saved500', 'norm_matched_joint_bisector_final', 'norm_matched_T_priority_projection_final'],
        select={'zero': 'norm_matched_T_priority_projection_final', 'bisector': 'norm_matched_joint_bisector_final'}),
}
HEADER = IDS + ['condition']
ENDPOINT = HEADER + ['extra_updates', 'n', 'component3_mse', 'penalty', 'trainable_objective']
PAIR = IDS + ['cohort', 'source500_parameters_same'] + [
    ref + '_anchor_' + field for ref in REFERENCES[1:]
    for field in ['component3_mse', 'penalty', 'trainable_objective', 'extra_updates']]
HEAD_SCHEMA = ['fit_rows', 'fit_demo_indices', 'scaler_mean', 'scaler_scale', 'updates', 'weights', 'bias', 'optimizer_steps']
HEAD_READ = [field for field in HEAD_SCHEMA if field not in ('weights', 'bias')]
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_reference_joint_reachability_diagnostic.py',
    'audit': ROOT / 'experiments/audit_libero_task7_reference_joint_reachability_diagnostic.py',
    'guard': ROOT / 'experiments/libero_task7_reference_joint_reachability_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_reference_joint_reachability_diagnostic.py',
}
FUTURE_CONTRACT = ROOT / f'docs/{STEM}-execution-contract-2026-11-11.md'


def point(m, penalty):
    """Human scalar objective; real payload computation awaits new bound roles."""
    p.require(all(type(v) is float and math.isfinite(v) and v >= 0 for v in (m, penalty)), 'finite nonnegative binary64 M/P required')
    return validate_point(dict(M=m, P=penalty, T=m / 7. + penalty))


def validate_point(row):
    p.require(type(row) is dict and set(row) == {'M', 'P', 'T'}
              and all(type(v) is float and math.isfinite(v) and v >= 0 for v in row.values()), 'finite nonnegative M/P/T required')
    p.require(abs(row['T'] - (row['M'] / 7. + row['P'])) <= TOL, 'M/P/T conservation failed')
    return dict(row)


def reference_points(refs):
    p.require(type(refs) is dict and set(refs) == set(REFERENCES), 'all original four reference roles required')
    return {name: validate_point(refs[name]) for name in REFERENCES}


def caps(refs):
    values = reference_points(refs)
    mc, tc = min(v['M'] for v in values.values()), min(v['T'] for v in values.values())
    return dict(M=mc, T=tc, M_active=[r for r in REFERENCES if values[r]['M'] == mc],
                T_active=[r for r in REFERENCES if values[r]['T'] == tc])


def gates(value, refs):
    """Original raw four-reference conditions, no sign/strictness tolerance."""
    value, refs = validate_point(value), reference_points(refs)
    cap = caps(refs)
    per = {r: dict(M=value['M'] <= refs[r]['M'], T=value['T'] < refs[r]['T']) for r in REFERENCES}
    compressed = dict(M=value['M'] <= cap['M'], T=value['T'] < cap['T'])
    raw = all(v['M'] and v['T'] for v in per.values())
    p.require(raw == (compressed['M'] and compressed['T']), 'four-reference/cap equivalence failed')
    return dict(per_reference=per, caps=cap, cap_pass=compressed, all_four=raw)


def replay_gates(value, cached, replayed):
    cached, replayed = reference_points(cached), reference_points(replayed)
    p.require(all(abs(cached[r][f] - replayed[r][f]) <= TOL for r in REFERENCES for f in ('M', 'P', 'T')),
              'same reference replay exceeds original absolute1e-10 tolerance')
    raw, fresh = gates(value, cached), gates(value, replayed)
    return dict(cached=raw, replayed=fresh, raw_cached_pass=raw['all_four'],
                replay_confirmed=raw['all_four'] and fresh['all_four'])


def same_parameter_witness(theta, other_theta, value, other_value, cached, replayed, other_replayed):
    """Both implementations must pass at exactly the same coordinates, not means."""
    vector(theta)
    vector(other_theta)
    p.require(all(x.hex() == y.hex() for x, y in zip(theta, other_theta)), 'same-parameter replay changed binary64 coordinates')
    value, other_value = validate_point(value), validate_point(other_value)
    p.require(all(abs(value[f] - other_value[f]) <= TOL for f in ('M', 'P', 'T')), 'same-parameter numerical tolerance exceeded')
    first, second = replay_gates(value, cached, replayed), replay_gates(other_value, cached, other_replayed)
    return dict(first=first, second=second, raw_flag_disagreement=first['raw_cached_pass'] != second['raw_cached_pass'],
                confirmed_flag_disagreement=first['replay_confirmed'] != second['replay_confirmed'],
                verified=first['replay_confirmed'] and second['replay_confirmed'])


def vector(values):
    p.require(type(values) is list and len(values) == 257 and all(type(v) is float and math.isfinite(v) for v in values),
              'finite binary64 parameter list257 required')
    return values


def recipes():
    records = [dict(recipe_id=f'ref:{name}', kind='reference', left=name, right=name, numerator=0, denominator=32)
               for name in REFERENCES]
    for i, left in enumerate(REFERENCES):
        for right in REFERENCES[i + 1:]:
            records.extend(dict(recipe_id=f'edge:{left}:{right}:{j:02d}/32', kind='edge', left=left, right=right,
                                numerator=j, denominator=32) for j in range(1, 32))
    return records


def interpolate(refs, recipe):
    p.require(type(refs) is dict and set(refs) == set(REFERENCES), 'four parameter references required')
    for theta in refs.values():
        vector(theta)
    p.require(type(recipe) is dict and type(recipe.get('numerator')) is int and type(recipe.get('denominator')) is int
              and recipe in recipes(), 'recipe outside frozen 190 slots')
    left, right = refs[recipe['left']], refs[recipe['right']]
    if recipe['kind'] == 'reference':
        out = list(left)
    else:
        alpha = recipe['numerator'] / recipe['denominator']
        out = [x + alpha * (y - x) for x, y in zip(left, right)]
    return vector(out)


def result_label(verified):
    p.require(type(verified) is list and len(verified) == 190 and all(type(v) is bool for v in verified),
              'complete190 grid required before classifying a head')
    return 'verified_endpoint_witness' if any(verified) else 'not_found_on_frozen_grid'


def budget():
    n = 6 * (2782 + 2817 + 2777)
    per = 18 * 190
    return dict(heads=18, parameter_families=['P', 'U'], computation_roles=ROLES,
        reference_slots_per_head=4, unordered_pairs=6, interior_slots_per_pair=31, recipe_slots_per_head=190,
        complete_head_grids=72, core_forwards_per_role=per, core_forwards_total=4 * per,
        reference_calibration_forwards_per_role=72, reference_calibration_forwards_total=288,
        fit_rows_sum_per_role=n, forward_rows_per_role=n * 190, forward_rows_total=4 * n * 190,
        feature_multiplies_per_role=n * 190 * 256, feature_multiplies_total=4 * n * 190 * 256,
        parameter_forms_by_role=dict(P_numpy=per, P_torch=per, U_torch=per, U_numpy=0),
        parameter_forms_total=3 * per, parameter_coordinate_writes_total=3 * per * 257,
        original_source_JSON_parses=6, original_source_JSON_SHA=12,
        original_source_endpoint_header_rows=6 * 54, original_source_endpoint_body_rows=144, original_source_pair_rows=36,
        original_source_scalar_projections=4212, original_source_parameter_coordinates=37008, original_source_total_projection_values=41220,
        new_primary_JSON_audit_parses=1, new_primary_JSON_audit_SHA=2, new_primary_reference_parameter_coordinates=18504,
        parameter_coordinates_read_total=55512,
        NPZ_loads_per_stage=25, NPZ_loads_total=50, NPZ_decodes_per_stage=125, NPZ_decodes_total=250, NPZ_SHA_total=100,
        logical_gate_comparisons_per_set=20, logical_gate_sets_total=17100, logical_gate_comparisons_total=342000,
        logical_cap_reductions_total=360, logical_cap_input_scalars_total=1440,
        probe_CSV_rows_by_role={role: per for role in ROLES}, probe_CSV_rows_total=4 * per,
        original_Policy_CNN_environment_forwards=0, optimizer_constructions_updates_backward_autograd_gradients_trials=0,
        checkpoint_reads_or_writes=0, NPZ_writes=0, extra_source_CSV_HDF5_observation_or_sealed_reads_including_SHA=0,
        retries_overwrites_grid_refinements=0)


def create():
    """Only already-bound metadata/code and the completed readonly envelope."""
    previous, bound = evidence.preflight()
    meta = {name: p.load(ref) for name, ref in PARENTS.items() if name != 'report'}
    p.require(p.pin(PARENTS['report']['path']) == PARENTS['report'], 'completed readonly report changed')
    done, proc = meta['completion'], meta['process']
    p.require(meta['science'] == previous and meta['binding'] == bound and len(bound['dependencies']) == 929
              and done['status'] == proc['status'] == 'complete_independently_audited_joint_T_tradeoff_gradient'
              and done['science'] == proc['science'] == PARENTS['science'] and done['binding'] == proc['binding'] == PARENTS['binding']
              and done['analysis']['path'] == str(evidence.OUTPUT) and done['audit']['path'] == str(evidence.AUDIT)
              and done['checks'] == 36351 and done['source_JSON_passes'] == 4 and done['source_SHA_checks'] == 8
              and done['total_projected_scalar_fields'] == 116064
              and done['new_forward_gradient_autograd_optimizer_trial_environment_calls'] == 0
              and done['formal_result'] is False and len(proc['processes']) == 2
              and all(row['exit_code'] == 0 for row in proc['processes']), 'completed readonly provenance required')
    source = p.load(evidence.PARENTS['science'])
    heads = source['cohorts']
    inputs = source['inputs']
    p.require(len(heads) == 18 and len(inputs['models']) == 6 and len(inputs['heads']) == 18
              and [(f['fold'], len(f['fit_rows']), f['fit_tge1_rows']) for f in inputs['folds']]
              == [(0, 2782, 2782), (1, 2817, 2817), (2, 2777, 2777)], 'original18 fixed fit roles changed')
    return dict(kind='fixed_representation_four_reference_joint_M_T_finite_witness_grid', version=1, formal_result=False,
        status='scientific_freeze_only_execution_binding_required', parents=copy.deepcopy(PARENTS),
        source_JSON=copy.deepcopy(SOURCE_JSON), source_roles=copy.deepcopy(SOURCE_ROLES), inputs=copy.deepcopy(inputs),
        identities=[{f: h[f] for f in IDS} for h in heads], references=REFERENCES.copy(), computation_roles=ROLES.copy(),
        scope=dict(task=7, original_Policy_updates=500, future_stabilization_steps=5, loss_lambda=1,
            parameter_mode='harm_protected_affine_current256_plus_bias;other6_actions_fixed_originalPolicy',
            source='own20_fit_support0..29_tge1_including_neutral;encoder_seen_all30;not_unseen_generalization',
            zero_new_training_gradients_directions_or_Policy_environment_calls=True,
            all18_all4refs_both_histories_all190_slots_retained=True, query_validation_heldout_qualification_reserved_final_sealed=True,
            original_main_cue='0/18_unchanged;G_stopped_H_unrun', SPT_Knowledge_Module_changes=False),
        projections=dict(endpoint_header=HEADER, selected_endpoint=ENDPOINT, latest_pairs=PAIR,
            reference_parameters='only_selected_4x257_per_head_per_history;no_other_endpoint_state_trial_gradient_vectors',
            source_JSON_physical_decode='whole_file_including_unprojected_fields;semantic_whitelist_not_physical_redaction',
            shared=['targets', 'demo_indices', 'action_steps', 'demo_lengths', 'tail20'],
            current=['features', 'predictions'], head_schema=HEAD_SCHEMA, head_members_decoded=HEAD_READ,
            source_archives='1shared+6current+18head_per_stage;all_source_file_bytes_hashed_before_after;no_checkpoints'),
        objective=dict(M='mean((clip(p0_3+(sum(z_j*theta_j)+theta_bias),-1,1)-expert3)^2)',
            P='mean(max(per_frame_expert3_squared_error-originalPolicy3_squared_error,0))/7',
            T='M/7+P;lambda1_unchanged;Float64_no_extra_float32_action_rounding',
            normalization='own_fit20_mean_std64_scale_le1e-6_to1;saved_scalers_allclose_abs1e-10_rel0;normalize_then_cast_float32_then_float64',
            row_order='original_fixed_fit_rows;no_probe_scoring;features_targets_p0_archives_float32_promoted_to_float64',
            source_join='strict_identity_condition_n_updates;old_endpoint_M_P_T_match_latest_cached_four_refs_abs1e-10;P_U_anchors_separate'),
        conditions=dict(M='M(theta)<=min(original_cached_M_control_margin_zero_bisector)',
            T='T(theta)<min(original_cached_T_control_margin_zero_bisector)',
            P='no_separate_P_bound', same_single_theta_for_both=True,
            cached_caps_never_replaced=True, raw_sign_tolerance=0, numerical_absolute_tolerance=TOL, numerical_relative_tolerance=0,
            replay='evaluate_all4_refs_first;cached_vs_replay_abs1e-10_or_technical_stop;witness_must_also_pass_same_role_replayed_all4_raw_gates',
            independent='P_numpy_vs_P_torch_same_exact_theta_and_U_torch_vs_U_numpy_same_exact_theta;both_raw_and_replay_gates_pass',
            boolean_disagreement='retain_even_if_metrics_within_tolerance;not_verified;never_force_P_U_history_agreement'),
        grid=dict(reference_order=REFERENCES.copy(), recipes=recipes(), denominator=32,
            construction='endpoint_copy_or_coordinate_left+alpha*(right-left);binary64_no_reassociation_FMA_or_rescale',
            coverage='4_vertices_plus31_interior_points_on_each_of6_edges;not_full_convex_hull_or_all_parameters',
            no_metric_based_dedup_or_early_success_stop=True, normalization_frame='same_own_fit20_current256_frame_for_all4refs',
            retain_all_recipes_and_exact_parameter_digests=True, checkpoint_or_best_recipe_selection=False),
        decision=dict(positive='verified_endpoint_witness_after_same_parameter_independent_confirmation',
            negative='not_found_on_frozen_grid;global_fixed_representation_reachability_unresolved',
            no_global_infeasibility_or_global_optimality_claim=True,
            path='endpoint_witness_does_not_establish_original500_50step_direction_Armijo_M500_stepM_path_reachability',
            promotion='does_not_replace_progress_and_path_gates_or_change_original_four_reference_main0/18',
            reporting='all18_and_each_history_separately;A/B_source_seed_fold_counts_means_medians;no_pvalues_or_head_selection'),
        budget=budget(), execution=dict(python='3.10.22', numpy='1.22.4', torch='2.4.1+cu124', device='CPU',
            seconds_primary=300, seconds_audit=900, overall_seconds=1500, RSS_bytes_primary=536870912, RSS_bytes_audit=1073741824,
            source_JSON_bytes_max=201326592, output_JSON_bytes_max=67108864, output_CSV_bytes_max=67108864,
            environment={'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            torch_threads=1, torch_interop_threads=1, torch_autograd=False,
            stop='role_identity_fields_coverage_scaler_frame_nonfinite_overflow_budget_replay_or_binding_mismatch_technical_failed_no_retry',
            no_new_research_access_including_SHA_before_execution_binding=True,
            required_before_execution='new_roles_guard_independent_formulas_contract_full18_synthetic_maximum_budget_envelope_and_execution_binding'),
        code=dict(metadata=p.pin(Path(__file__).resolve()), metadata_tests=p.pin(TESTS)), plan=p.pin(PLAN),
        future_files={k: str(v) for k, v in FUTURE_FILES.items()}, future_contract=str(FUTURE_CONTRACT),
        artifacts=str(ARTIFACTS), output=str(ARTIFACTS / 'primary-result.json'), audit=str(ARTIFACTS / 'independent-audit.json'))


def forbidden_payload_paths(cfg):
    refs = [r for family in SOURCE_JSON.values() for r in family.values()]
    refs += [cfg['inputs']['shared']] + [m['current_archive'] for m in cfg['inputs']['models']]
    refs += [h['archive'] for h in cfg['inputs']['heads']]
    refs += [h['checkpoint'] for h in cfg['inputs']['heads']]
    return {r['path'] for r in refs} | {str(evidence.OUTPUT), str(evidence.AUDIT)}


def freeze_dependencies(cfg):
    old = p.load(PARENTS['binding'])['dependencies']
    refs = old + list(PARENTS.values()) + [cfg['code']['metadata'], cfg['code']['metadata_tests'], cfg['plan'], p.pin(CONFIG)]
    bypath = {}
    for ref in refs:
        p.require(ref['path'] not in bypath or bypath[ref['path']] == ref, 'conflicting dependency hash')
        bypath[ref['path']] = ref
    p.require(not set(bypath) & forbidden_payload_paths(cfg), 'payload must never enter metadata binding hashes')
    return [bypath[k] for k in sorted(bypath)]


def validate():
    cfg = json.loads(CONFIG.read_text())
    p.require(cfg == create(), 'frozen four-reference reachability design changed')
    return cfg


def validate_freeze():
    cfg = validate()
    frozen = json.loads(FREEZE_BINDING.read_text())
    p.require(frozen['status'] == 'scientific_frozen_execution_not_bound' and frozen['science'] == p.pin(CONFIG)
              and frozen['dependencies'] == freeze_dependencies(cfg) and frozen['budget'] == cfg['budget']
              and frozen['research_payload_access_including_SHA'] == 0
              and frozen['new_research_forward_gradient_or_update_calls'] == 0, 'scientific freeze binding required')
    for ref in frozen['dependencies']:
        p.require(p.pin(ref['path']) == ref, 'scientific dependency changed')
    return cfg, frozen


def require_binding():
    cfg, frozen = validate_freeze()
    p.require(BINDING.is_file(), 'no execution binding; science freeze alone denies research payload access')
    bound = json.loads(BINDING.read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['before_payload'] is True
              and bound['science'] == p.pin(CONFIG) and bound['scientific_binding'] == p.pin(FREEZE_BINDING)
              and bound['budget'] == cfg['budget'] and bound['CPU_environment'] == cfg['execution']['environment'], 'invalid future execution binding')
    refs = bound['dependencies']
    paths = [r['path'] for r in refs]
    p.require(len(paths) == len(set(paths)) and not set(paths) & forbidden_payload_paths(cfg)
              and all(ref in refs for ref in frozen['dependencies']) and p.pin(FREEZE_BINDING) in refs,
              'full scientific closure required before payload')
    for file in [*FUTURE_FILES.values(), FUTURE_CONTRACT]:
        p.require(file.is_file() and p.pin(file) in refs, 'new execution role/contract missing from binding')
    p.require(bound['full18_synthetic_passed'] is True and bound['maximum_budget_envelope_passed'] is True,
              'full synthetic and maximum envelope required')
    for ref in refs:
        p.require(Path(ref['path']).suffix in {'.py', '.md', '.json', '.toml'} and p.pin(ref['path']) == ref, 'execution dependency changed')
    return cfg, bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        p.write(CONFIG, create())
    else:
        validate_freeze()
    print(json.dumps({'status': 'metadata_human_only_four_reference_reachability_freeze', 'research_payload_access_including_SHA': 0}))
