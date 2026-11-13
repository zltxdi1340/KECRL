"""Metadata/scalar freeze for saved-gradient and last-grid read-only alignment."""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path
from experiments import libero_task7_mse_nonregression_acceptance_protocol as parent

p, ROOT = parent.p, parent.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-mse-gate-stop-direction-grid-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-mse-gate-stop-direction-grid-execution-binding_v1.json'
PLAN = ROOT / 'docs/libero-task7-mse-gate-stop-direction-grid-plan-2026-11-09.md'
TESTS = ROOT / 'tests/test_libero_task7_mse_gate_stop_direction_grid_protocol.py'
OUTPUT = '/tmp/libero-task7-mse-gate-stop-direction-grid-20261109-v1.json'
ARTIFACTS = '/tmp/libero-task7-mse-gate-stop-direction-grid-artifacts-20261109-v1'
PARENTS = {
    'science': {'path': str(parent.DEFAULT_CONFIG), 'sha256': 'b470d22c51b11848e2aa920cd2969dbf5cfe6520e8a36f481ce3be55b433515b'},
    'binding': {'path': str(parent.DEFAULT_BINDING), 'sha256': 'c4ca468bd54ea62e102154741dad21f9b9381b529f374a4e1a0122679eea497c'},
    'result': {'path': parent.OUTPUT, 'sha256': '5ff96041da1af933f2054e85816bda921a5c54581c77bc918a5e04041f52bfa6'},
    'audit': {'path': parent.ARTIFACTS + '/independent-audit.json', 'sha256': 'd5ccf67d596e7a1b5abba789f7f0b2f6f808508858e8b3cb0f1c8aeffea173c6'},
    'completion': {'path': parent.ARTIFACTS + '/completion.json', 'sha256': 'b5f18a8b13c470d4e98a42b8f65035f9299c6a2baa04d9195fea2afb4c88ce44'},
    'process': {'path': '/tmp/libero-task7-mse-nonregression-acceptance-20261109-v1-process.json', 'sha256': '3c6f0dd513a7d5a141eb6a55c82fa827a662ef29c4b60a0a23bb48691231b96d'},
    'report': {'path': str(ROOT / 'docs/libero-task7-mse-nonregression-acceptance-results-2026-11-09.md'), 'sha256': 'a9177ae9996e5bc054bc17fee67a540f8539a15be9920bc527df84943cfac136'},
}
IDS = ['task', 'seed', 'arm', 'fold', 'mode']
VECTORS = ['base_gradient', 'penalty_gradient', 'total_gradient', 'direction']
COUNTS = ['clip_' + n + '_n' for n in ['below', 'low_tie', 'inside', 'high_tie', 'above']] + ['hinge_' + n + '_n' for n in ['negative', 'zero', 'positive']]
COLUMNS = {
    'endpoints': IDS + ['condition', 'extra_updates', 'n', 'fixed_other6_floor', 'component3_mse', 'penalty',
                        'trainable_objective', 'total_objective', 'base_l2', 'penalty_l2', 'total_l2',
                        'directional_derivative', 'branch_digest'] + COUNTS,
    'parameters': IDS + ['condition', 'coordinate'] + VECTORS,
    'trials': IDS + ['optimization_arm', 'from_extra_updates', 'trial_index', 'alpha', 'component3_mse', 'penalty',
                     'trainable_objective', 'total_objective', 'start500_component3_mse', 'slope', 'armijo_bound',
                     'armijo_pass', 'mse_envelope_pass', 'mse_gate_applied', 'accepted', 'reason'],
}
FUTURE_FILES = {
    'analysis': ROOT / 'experiments/analyze_libero_task7_mse_gate_stop_direction_grid.py',
    'audit': ROOT / 'experiments/audit_libero_task7_mse_gate_stop_direction_grid.py',
    'guard': ROOT / 'experiments/libero_task7_mse_gate_stop_direction_grid_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_mse_gate_stop_direction_grid.py',
    'metadata': Path(__file__).resolve(), 'metadata_tests': TESTS, 'plan': PLAN,
}
FUTURE_CONTRACT = ROOT / 'docs/libero-task7-mse-gate-stop-direction-grid-execution-contract-2026-11-09.md'
TOL = 1e-10
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'original_Policy_updates': 500, 'source_auxiliary_updates': 500, 'future_stabilization_steps': 5,
    'analysis_cohort': 'all11_candidate_line_search_exhausted_of18;6zero5short;outcome_selected_not_causal',
    'full50_heads': '7_metadata_only_denominator_not_gradient_controls',
    'source_role': 'own20_fit_demos_of_support0..29_all_tge1;encoder_seen_all30',
    'new_optimization_gradient_or_model_forward_environment_calls': 0,
    'array_checkpoint_HDF5_RGB_proprio_reads_including_SHA': 0,
    'probe_source_query_validation_heldout_qualification_reserved_final_access': False,
    'loss_representation_search_direction_grid_budget_or_checkpoint_rule_changes': False,
    'Policy_SPT_Knowledge_Module_changes': False, 'old_cues_G_failed_H_unrun_unchanged': True,
}
DESIGN = {
    'points': ['saved500', parent.ARMS[1] + '_final'], 'dimension': 257,
    'vectors': 'Gb=selected_base_derivative_MSE3/7,Gp=selected_penalty_subgradient,G=Gb+Gp,d=-G;coordinate_and_cached_norm_checks',
    'selection': '11_head_IDs_from_audited_metadata;two_points_each_even_zero_update;last12_candidate_trials_at_saved_stop_k',
    'gradient_outputs': 'norms_dot_Gb_Gp_cosine;dot_Gb_d_dot_Gp_d_dot_G_d;start_stop_vector_changes;zero_norm_cosine_null',
    'exact_total_slope': 'reuse_cached_exact_right_directional_derivative_sT;never_replace_by_minus_normG_squared_at_kinks',
    'exact_MSE_slope': 'only_if_clip_low_tie_plus_high_tie_is0:7*dot_Gb_d;else_null_clip_kink_unresolved',
    'exact_penalty_slope': 'if_exact_MSE_available:sT-dot_Gb_d;valid_even_hinge_ties;else_null',
    'hinge_gap': 'when_no_clip_ties:sT-dot_G_d>=-1e-10;hinge0_requires_abs_gap<=1e-10;gap_is_not_new_forward',
    'point_pair': 'all4vector_change_norms;zero_update_start_stop_vectors_and_selected_scalars_exact_equal',
    'slack': 'M500-Mstop>=0_strict;fixed_auxiliary500_not_original_Policy_or_previous_state',
    'trial_outputs': 'actual_DeltaM_step_and_gate_excess_DeltaP_DeltaT;Armijo_margin;secants;eligible_first_order_prediction_and_residual',
    'alpha_linear': 'slack/exact_MSE_slope_only_if_both_positive;scale_only_not_safe_step_or_trial',
    'classification': 'positive_robust_slack_plus_negative_sT_implies_local_feasibility_by_continuity;boundary_exact_positive_MSE_slope_implies_near0_conflict;boundary_negative_both_implies_local_feasibility;clip_or_nearzero_unresolved',
    'numeric_band': 'abs1e-10_descriptive_inference_only;retain_strict_signs;never_relax_or_relabel_trial_acceptance_or_slack',
    'no_new_alpha': 'no_untried_alpha_evaluation_no_feature_read_or_gradient_recomputation_no_direction_search',
    'grouping': '6source_seed_cells_each_zero_and_short_including_empty=12;equal_head_values_folds_not_independent_seeds',
    'statistics': 'descriptive_only_no_pvalues_no_global_feasibility_loss_root_cause_or_closed_loop_claim',
}
EXECUTION = {
    'status': 'scientific_freeze_only_execution_binding_required', 'device': 'CPU', 'python': '3.10.22',
    'environment': copy.deepcopy(parent.EXECUTION['environment']), 'pure_stdlib_no_numpy_torch_import': True,
    'seconds_per_stage': 120, 'RSS_bytes_per_stage': 268435456, 'overall_seconds': 300,
    'CSV_bytes_max': 16777216, 'absolute_tolerance': TOL, 'relative_tolerance': 0,
    'ID_count_shape_role_flags_and_decisions_tolerance': 0, 'technical_retries': 0, 'overwrite': False,
    'stop': 'any_lineage_projection_coverage_coordinate_identity_norm_slope_conservation_grid_or_audit_mismatch;retain_partial_no_relaxation',
}


def budget():
    rows = {'endpoints': 54, 'parameters': 13878, 'trials': 10090}
    selected = {'endpoints': 22, 'parameters': 5654, 'trials': 132}
    cells = {k: selected[k] * len(COLUMNS[k]) for k in rows}
    tables = {'points': 22, 'trials': 132, 'heads': 11, 'strata': 12}
    return {
        'head_cohort': {'candidate_total': 18, 'exhausted': 11, 'zero': 6, 'short': 5, 'full50_metadata_only': 7},
        'source_CSV_passes_per_stage': 3, 'source_rows_per_stage': rows, 'selected_rows_per_stage': selected,
        'projected_cells_per_stage': cells, 'lexical_cells_per_stage': 380100,
        'total_source_CSV_passes': 6, 'total_source_rows': 48044,
        'total_selected_rows': sum(selected.values()) * 2, 'total_projected_cells': sum(cells.values()) * 2,
        'total_lexical_cells': 760200, 'source_SHA_per_stage': 6, 'total_source_SHA': 12,
        'cached_gradient_vector_coordinates_per_stage': 22 * 4 * 257,
        'gradient_coordinate_rows_checked_per_stage': 22 * 257,
        'new_model_forward_gradient_update_array_or_sealed_calls': 0,
        'tables': tables, 'output_rows': sum(tables.values()),
    }


def finite(*values):
    p.require(all(type(v) in (int, float) and math.isfinite(v) for v in values), 'finite real scalars required')


def sign(value):
    finite(value)
    return 'positive' if value > 0 else 'negative' if value < 0 else 'zero'


def safe_ratio(numerator, denominator):
    finite(numerator, denominator)
    if denominator == 0: return None
    value = numerator / denominator
    finite(value)
    return value


def gradient_geometry(gb, gp, total, direction, clip_ties, hinge_ties, exact_total_slope):
    vectors = (gb, gp, total, direction)
    p.require(all(type(v) is list and len(v) == 257 for v in vectors)
              and type(clip_ties) is type(hinge_ties) is int and clip_ties >= 0 and hinge_ties >= 0,
              'four257 vectors and nonnegative integer tie counts required')
    for vector in vectors: finite(*vector)
    finite(exact_total_slope)
    p.require(all(abs(g - a - b) <= TOL and abs(d + g) <= TOL for a, b, g, d in zip(*vectors)),
              'saved total gradient or direction identity differs')
    dot = lambda a, b: math.fsum(x * y for x, y in zip(a, b))
    norms = [math.sqrt(dot(v, v)) for v in vectors]
    ab, ad, pd, td = dot(gb, gp), dot(gb, direction), dot(gp, direction), dot(total, direction)
    finite(*norms, ab, ad, pd, td)
    p.require(abs(td + norms[2] ** 2) <= TOL and abs(td - ad - pd) <= TOL, 'gradient directional conservation differs')
    cosine = safe_ratio(ab, norms[0] * norms[1])
    if cosine is not None:
        p.require(-1 - TOL <= cosine <= 1 + TOL, 'gradient cosine outside range')
        cosine = max(-1., min(1., cosine))
    gap = exact_total_slope - td
    if clip_ties == 0:
        p.require(gap >= -TOL and (hinge_ties != 0 or abs(gap) <= TOL), 'cached exact slope inconsistent with smooth/hinge geometry')
    exact_mse = 7 * ad if clip_ties == 0 else None
    exact_penalty = exact_total_slope - ad if clip_ties == 0 else None
    if exact_mse is not None: finite(exact_mse, exact_penalty)
    return dict(base_l2=norms[0], penalty_l2=norms[1], total_l2=norms[2], direction_l2=norms[3],
                base_penalty_dot=ab, base_penalty_cosine=cosine, dot_base_direction=ad,
                dot_penalty_direction=pd, dot_total_direction=td, cached_exact_total_slope=exact_total_slope,
                kink_gap=gap, exact_mse_slope=exact_mse, exact_penalty_slope=exact_penalty,
                mse_slope_status='exact_from_saved_gradient_no_clip_tie' if clip_ties == 0 else 'unresolved_clip_tie')


def feasibility_evidence(start_mse, stop_mse, exact_total_slope, exact_mse_slope):
    finite(start_mse, stop_mse, exact_total_slope)
    p.require(start_mse >= 0 and 0 <= stop_mse <= start_mse and exact_total_slope < 0,
              'valid fixed-envelope exhausted point with negative exact slope required')
    if exact_mse_slope is not None: finite(exact_mse_slope)
    slack = start_mse - stop_mse
    if exact_total_slope >= -TOL: label = 'near_zero_total_slope_unresolved'
    elif slack > TOL: label = 'positive_slack_local_feasible_by_continuity'
    elif slack > 0: label = 'positive_slack_near_numeric_band'
    elif exact_mse_slope is None: label = 'boundary_clip_kink_unresolved'
    elif exact_mse_slope > TOL: label = 'boundary_direction_conflict_near_zero_alpha'
    elif exact_mse_slope < -TOL: label = 'boundary_descent_local_feasible'
    else: label = 'boundary_near_zero_mse_slope_unresolved'
    scale = safe_ratio(slack, exact_mse_slope) if slack > 0 and exact_mse_slope is not None and exact_mse_slope > 0 else None
    return dict(slack=slack, strict_slack_sign=sign(slack), strict_total_slope_sign=sign(exact_total_slope),
                strict_mse_slope_sign=sign(exact_mse_slope) if exact_mse_slope is not None else None,
                inference_class=label, linear_alpha_scale=scale,
                linear_scale_over_smallest_grid_alpha=safe_ratio(scale, 1 / 2048) if scale is not None else None,
                untried_step_evaluated=False, global_infeasibility_claim=False)


def trial_terms(start_mse, stop_mse, stop_penalty, stop_trainable, trial_mse, trial_penalty, trial_trainable, alpha, exact_mse_slope):
    finite(start_mse, stop_mse, stop_penalty, stop_trainable, trial_mse, trial_penalty, trial_trainable, alpha)
    p.require(alpha > 0 and min(start_mse, stop_mse, stop_penalty, stop_trainable, trial_mse, trial_penalty, trial_trainable) >= 0
              and stop_mse <= start_mse, 'invalid cached trial quantities')
    dm, dp, dt = trial_mse - stop_mse, trial_penalty - stop_penalty, trial_trainable - stop_trainable
    p.require(abs(dt - dm / 7 - dp) <= TOL, 'trial loss contributions do not conserve')
    if exact_mse_slope is not None: finite(exact_mse_slope)
    prediction = alpha * exact_mse_slope if exact_mse_slope is not None else None
    return dict(delta_mse_step=dm, mse_gate_excess=trial_mse - start_mse, delta_penalty=dp, delta_trainable=dt,
                mse_secant=safe_ratio(dm, alpha), mse_first_order_prediction=prediction,
                mse_first_order_residual=dm - prediction if prediction is not None else None,
                conservation_residual=dt - dm / 7 - dp)


def create():
    lineage = {k: p.load_pin(v) for k, v in PARENTS.items() if k != 'report'}
    p.require(p.pin(Path(PARENTS['report']['path'])) == PARENTS['report'], 'completed parent report changed')
    cfg = parent.validate(); r, a, done, process = (lineage[k] for k in ('result', 'audit', 'completion', 'process'))
    p.require(lineage['science'] == cfg and r['science_config'] == done['science_config'] == process['science_config'] == PARENTS['science']
              and r['binding'] == done['binding'] == process['binding'] == PARENTS['binding']
              and a['result'] == done['result'] == PARENTS['result'] and done['audit'] == PARENTS['audit']
              and done['status'] == 'complete_independently_audited_mse_acceptance'
              and a['status'] == 'passed_independent_mse_acceptance_audit'
              and process['status'] == 'complete_independently_audited' and len(process['processes']) == 2
              and all(row['exit_code'] == 0 for row in process['processes']) and len(lineage['binding']['dependencies']) == 745,
              'completed audited paired-run lineage required')
    candidates = [h for h in r['heads'] if h['optimization_arm'] == parent.ARMS[1]]
    all_ids = {(7, seed, arm, fold, parent.MODE) for seed in p.SEEDS for arm in p.ARMS for fold in range(3)}
    p.require(len(candidates) == 18 and {tuple(h[k] for k in IDS) for h in candidates} == all_ids, 'all18 parent candidates required')
    heads, reference_full = [], []
    for h in candidates:
        identity = {k: h[k] for k in IDS}
        if h['terminal_reason'] == 'line_search_exhausted':
            p.require(type(h['extra_updates']) is int and 0 <= h['extra_updates'] < 50, 'invalid exhausted update index')
            heads.append({**identity, 'stop_extra_updates': h['extra_updates'], 'cohort': 'zero' if h['extra_updates'] == 0 else 'short'})
        else:
            p.require(h['terminal_reason'] == 'update_cap' and h['extra_updates'] == 50, 'unknown parent stop')
            reference_full.append(identity)
    p.require(len(heads) == 11 and sum(h['cohort'] == 'zero' for h in heads) == 6
              and sum(h['cohort'] == 'short' for h in heads) == 5 and len(reference_full) == 7, 'fixed6zero5short7full coverage differs')
    inputs = {k: r['tables'][k] for k in COLUMNS}
    p.require(all(inputs[k]['rows'] == n and set(COLUMNS[k]).issubset(inputs[k]['columns'])
                  for k, n in budget()['source_rows_per_stage'].items())
              and sum(v['rows'] * len(v['columns']) for v in inputs.values()) == 380100, 'source CSV layout differs')
    return copy.deepcopy(dict(kind='task7_mse_gate_stop_direction_grid_readonly', version=1, formal_result=False,
        status='scientific_freeze_only_execution_binding_required', parents=PARENTS, scope=SCOPE, design=DESIGN,
        execution=EXECUTION, budget=budget(), inputs=inputs, projections=COLUMNS, heads=heads,
        full50_metadata_only=reference_full, fit_rows={str(f['fold']): f['fit_tge1_rows'] for f in cfg['inputs']['folds']},
        code={'metadata': p.pin(Path(__file__).resolve()), 'tests': p.pin(TESTS)}, plan=p.pin(PLAN),
        future_files={k: str(v) for k, v in FUTURE_FILES.items()}, future_contract=str(FUTURE_CONTRACT), output=OUTPUT, artifacts=ARTIFACTS))


def validate(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(json.dumps(cfg, sort_keys=True, allow_nan=False) == json.dumps(create(), sort_keys=True, allow_nan=False),
              'saved direction/grid scientific freeze differs')
    return cfg


def require_binding(config=DEFAULT_CONFIG, binding=DEFAULT_BINDING):
    cfg = validate(config); bound = json.loads(Path(binding).read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['before_research_payload'] is True
              and bound['science_config'] == p.pin(config) and bound['budget'] == cfg['budget']
              and bound['CPU_environment'] == EXECUTION['environment'] and bound['future_output'] == OUTPUT
              and bound['future_artifacts'] == ARTIFACTS, 'new read-only binding required')
    refs = bound['dependencies']; paths = [p.resolve(r['path']).resolve() for r in refs]
    p.require(len(paths) == len(set(paths)) and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT, DEFAULT_CONFIG}).issubset(paths),
              'read-only implementation roles not bound')
    p.require(all(r in refs for r in p.load_pin(PARENTS['binding'])['dependencies']), 'parent745 dependency closure missing')
    for ref in refs:
        path = p.resolve(ref['path'])
        p.require(path.suffix in ('.py', '.md', '.json', '.toml') and p.pin(path) == ref, 'bound code changed or payload in closure')
    return bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true'); parser.add_argument('--require-binding', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        cfg = create()
        with DEFAULT_CONFIG.open('x') as stream: json.dump(cfg, stream, ensure_ascii=False, indent=2); stream.write('\n')
        result = {'status': 'scientific_frozen_no_source_CSV_gradient_analysis_or_new_optimization'}
    else:
        validate(); result = {'status': 'metadata_valid_no_source_data_analysis'}
        if args.require_binding: require_binding(); result['status'] = 'bound_no_payload'
    print(json.dumps(result))
