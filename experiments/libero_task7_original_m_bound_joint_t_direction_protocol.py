"""Metadata/human freeze: interior T direction, original M certificate and step gates."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments import libero_task7_m_priority_direction_protocol as parent
from experiments import libero_task7_terminal_gradient_geometry_protocol as geometry

p, ROOT, IDS = parent.p, parent.ROOT, parent.IDS
margin = parent.margin
RHO = .01
INTERIOR_DELTA = 2. ** -26
RESOLUTION = margin.RESOLUTION
ARMS = [parent.ARMS[0], 'interior_joint_T_under_original_M_bound']
STEM = 'libero-task7-original-m-bound-joint-t-direction'
CONFIG = ROOT / f'configs/{STEM}-diagnostic_v1.json'
FREEZE_BINDING = ROOT / f'configs/{STEM}-freeze-binding_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT / f'docs/{STEM}-plan-2026-11-11.md'
TESTS = ROOT / 'tests/test_libero_task7_original_m_bound_joint_t_direction_protocol.py'
OUTPUT = f'/tmp/{STEM}-20261111-v1.json'
ARTIFACTS = f'/tmp/{STEM}-artifacts-20261111-v1'
PARENTS = {
    'science': {'path': str(geometry.CONFIG), 'sha256': 'a4759c468a1bf10cac7941e83ddd01c138c6a9e85a083f57407fe95ac138eadc'},
    'binding': {'path': str(geometry.BINDING), 'sha256': 'cc547d0cdef606c8712dca9183d9c46b37ab6db330456c2256977a552233418e'},
    'completion': {'path': str(geometry.ARTIFACTS / 'completion.json'), 'sha256': 'ca83a93c1575bb0517ab95c0c5bf3b49b1488bcd7b045da9db77970fd213ab38'},
    'process': {'path': str(geometry.ARTIFACTS.with_name(geometry.ARTIFACTS.name + '-process.json')), 'sha256': '65ac86fce94fea579db218bf3445d84c2d91b38c9f619099d6ae12b1a488deae'},
    'report': {'path': str(ROOT / 'docs/libero-task7-terminal-gradient-geometry-results-2026-11-11.md'), 'sha256': '299b667b46dd4380db52c0b2b4ac79606e81a97bd3b74bd3b5b2cc07c1013c71'},
}
# The guarded-T control is already fully reproduced in the M-priority run.
# Reuse its original, frozen step-M history/anchors, not the weak M-priority arm.
HISTORY_INPUTS = copy.deepcopy(parent.HISTORY_INPUTS)
CONTROL_COUNTS = copy.deepcopy(parent.CONTROL_COUNTS)
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_original_m_bound_joint_t_direction.py',
    'audit': ROOT / 'experiments/audit_libero_task7_original_m_bound_joint_t_direction.py',
    'guard': ROOT / 'experiments/libero_task7_original_m_bound_joint_t_direction_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_original_m_bound_joint_t_direction.py',
}
FUTURE_CONTRACT = ROOT / f'docs/{STEM}-execution-contract-2026-11-11.md'
NEW_STATE_FIELDS = ['direction_radius_fraction', 'direction_radius_limit', 'direction_interior_delta',
                    'direction_normalized_M_bound', 'selected_radius_met']
CONSTRUCTIONS = {'zero_total_gradient', 'zero_base_margin_unavailable', 'margin_unrepresentable',
                'interior_radius_unrepresentable', 'interior_cap_unrepresentable', 'unresolved_interior_direction',
                'interior_negative_total_feasible', 'interior_cap_T_priority'}


def vector(values):
    p.require(type(values) is list and len(values) == 257
              and all(type(v) is float and math.isfinite(v) for v in values), 'finite binary64 list257 required')


def interior(g, b, bound):
    """Scalar construction only; saved required bound is never replaced or relaxed."""
    R = g * (1. - INTERIOR_DELTA)
    if not 0 < R < g:
        return R, None, None, 'interior_radius_unrepresentable'
    eta = ((-bound / 7.) / b) / R
    kappa = eta + INTERIOR_DELTA
    p.require(math.isfinite(eta) and math.isfinite(kappa), 'nonfinite interior geometry')
    if not 0 < eta < kappa < 1:
        return R, eta, kappa, 'interior_cap_unrepresentable'
    return R, eta, kappa, None


def direction(arm, total, base):
    """Human vector reference only; execution requires new independent bound roles."""
    p.require(arm in ARMS, 'unregistered original-M-bound direction arm')
    if arm == ARMS[0]:
        return parent.direction(parent.ARMS[0], total, base)
    vector(total)
    vector(base)
    g, b = math.hypot(*total), math.hypot(*base)
    p.require(math.isfinite(g) and math.isfinite(b), 'nonfinite direction norms')
    bound = margin.required_bound(g, b, RHO)
    R, eta, kappa, c, q = None, None, None, None, None
    if g == 0:
        chosen, status = [0.] * 257, 'zero_total_gradient'
    elif b == 0:
        chosen, status = [0.] * 257, 'zero_base_margin_unavailable'
    elif bound is None:
        chosen, status = [0.] * 257, 'margin_unrepresentable'
    else:
        R, eta, kappa, error = interior(g, b, bound)
        if error:
            chosen, status = [0.] * 257, error
        else:
            u = [x / g for x in total]
            v = [x / b for x in base]
            c = math.fsum(x * y for x, y in zip(u, v))
            p.require(math.isfinite(c), 'nonfinite direction cosine')
            if c >= kappa:
                chosen = [-R * x for x in u]
                status = 'interior_negative_total_feasible'
            else:
                w = [-x + c * y for x, y in zip(u, v)]
                q = math.hypot(*w)
                p.require(math.isfinite(q), 'nonfinite tangent norm')
                if q <= RESOLUTION:
                    chosen, status = [0.] * 257, 'unresolved_interior_direction'
                else:
                    tangent = math.sqrt(1. - kappa * kappa)
                    chosen = [R * (-kappa * y + tangent * (x / q)) for x, y in zip(w, v)]
                    status = 'interior_cap_T_priority'
    vector(chosen)
    norm = math.hypot(*chosen)
    st = math.fsum(x * y for x, y in zip(total, chosen))
    sm = 7. * math.fsum(x * y for x, y in zip(base, chosen))
    p.require(all(math.isfinite(x) for x in (norm, st, sm)), 'nonfinite selected direction readout')
    construction = status
    met = bound is not None and sm <= bound
    radius_met = norm <= g
    if construction in {'interior_negative_total_feasible', 'interior_cap_T_priority'}:
        if not radius_met:
            status = 'selected_radius_violation'
        elif not met:
            status = 'selected_M_margin_violation'
        elif st >= 0:
            status = 'selected_T_non_descent'
    return dict(direction=chosen, status=status, construction_status=construction, total_norm=g, base_norm=b,
                unit_sum_norm=None, unit_cosine=c, projection_norm=q, direction_norm=norm,
                selected_T_slope=st, selected_MSE_slope=sm, selected_joint=met and st < 0,
                required_margin_fraction=RHO, construction_margin_fraction=kappa, required_M_slope_bound=bound,
                selected_margin_met=met, radius_fraction=1. - INTERIOR_DELTA, radius_limit=R,
                interior_delta=INTERIOR_DELTA, normalized_M_bound=eta, selected_radius_met=radius_met)


def validate_margin_record(arm, row):
    p.require(arm in ARMS, 'unregistered original-M-bound record')
    if arm == ARMS[0]:
        return parent.validate_margin_record(parent.ARMS[0], row)
    g, b = row['direction_total_reference_norm'], row['direction_base_reference_norm']
    bound = margin.required_bound(g, b, RHO)
    p.require(row['required_margin_fraction'] == RHO and row['required_M_slope_bound'] == bound
              and row['direction_radius_fraction'] == 1. - INTERIOR_DELTA
              and row['direction_interior_delta'] == INTERIOR_DELTA, 'original bound or fixed interior delta changed')
    R, eta, kappa = None, None, None
    if g != 0 and b != 0 and bound is not None:
        R, eta, kappa, _ = interior(g, b, bound)
    p.require(row['direction_radius_limit'] == R and row['direction_normalized_M_bound'] == eta
              and row['construction_margin_fraction'] == kappa, 'interior scalar geometry changed')
    st, sm, norm = row['selected_T_slope'], row['selected_MSE_slope'], row['chosen_direction_norm']
    p.require(all(type(x) is float and math.isfinite(x) for x in (st, sm, norm)) and norm >= 0,
              'invalid selected scalar readout')
    met = bound is not None and sm <= bound
    p.require(all(type(row[f]) is bool for f in ('selected_joint', 'selected_margin_met', 'selected_radius_met'))
              and row['selected_joint'] == (met and st < 0) and row['selected_margin_met'] == met
              and row['selected_radius_met'] == (norm <= g), 'forged original bound/radius/selected flags')
    return True


def stopping_reason(arm, updates, row, rejected_trials=0):
    p.require(arm in ARMS, 'unregistered original-M-bound stop arm')
    if arm == ARMS[0]:
        return parent.stopping_reason(parent.ARMS[0], updates, row, rejected_trials)
    p.require(type(updates) is int and 0 <= updates <= 50 and type(rejected_trials) is int and 0 <= rejected_trials <= 20,
              'stop budget differs')
    g, b = row['direction_total_reference_norm'], row['direction_base_reference_norm']
    st, sm, et, em = [row[k] for k in ('selected_T_slope', 'selected_MSE_slope', 'directional_derivative', 'exact_MSE_slope')]
    p.require(all(type(x) is float and math.isfinite(x) for x in (g, b, st, sm, et, em)) and g >= 0 and b >= 0,
              'invalid stop slopes/norms')
    construction = row['direction_construction_status']
    p.require(construction in CONSTRUCTIONS, 'unlisted construction stop')
    if updates == 50:
        return 'update_cap'
    if g == 0:
        p.require(st == sm == et == em == 0, 'nonzero slope for zero direction')
        return 'zero_gradient'
    if b == 0:
        return 'zero_base_margin_unavailable'
    bound = row['required_M_slope_bound']
    if bound is None:
        return 'margin_unrepresentable'
    p.require(type(bound) is float and math.isfinite(bound) and bound < 0, 'original M bound must be strictly negative')
    if construction in {'interior_radius_unrepresentable', 'interior_cap_unrepresentable', 'unresolved_interior_direction'}:
        return construction
    p.require(type(row['chosen_direction_norm']) is float and math.isfinite(row['chosen_direction_norm']), 'invalid direction norm')
    if row['chosen_direction_norm'] > g:
        return 'selected_radius_violation'
    if sm > bound:
        return 'selected_M_margin_violation'
    if st >= 0:
        return 'non_descent_selected_direction'
    if em > bound:
        return 'exact_M_margin_violation'
    if et >= 0:
        return 'non_descent_selected_direction'
    if rejected_trials == 20:
        return 'line_search_exhausted'
    return None


def trial_alpha(arm, index):
    p.require(arm in ARMS, 'unregistered original-M-bound grid')
    return parent.trial_alpha(parent.ARMS[0], index)


def trial_decision(arm, current, trial, slope, index, trial_mse, start_mse, current_mse):
    p.require(arm in ARMS, 'unregistered original-M-bound acceptance')
    return parent.trial_decision(parent.ARMS[0], current, trial, slope, index, trial_mse, start_mse, current_mse)


def validate_round(arm, current, slope, start_mse, current_mse, rows):
    p.require(arm in ARMS, 'unregistered original-M-bound round')
    return parent.validate_round(parent.ARMS[0], current, slope, start_mse, current_mse, rows)


def validate_state_transition(arm, previous_mse, measured_mse, start_mse):
    p.require(arm in ARMS, 'unregistered original-M-bound measured state')
    return parent.validate_state_transition(parent.ARMS[0], previous_mse, measured_mse, start_mse)


paired_evidence = parent.paired_evidence
source_evidence = parent.source_evidence


def budget():
    b = copy.deepcopy(parent.budget())
    old = parent.ARMS[1]
    for f in ('trial_caps_per_round', 'trial_objective_forms_max_by_arm'):
        b[f][ARMS[1]] = b[f].pop(old)
    b['direction_operation_caps_by_arm'] = {arm: dict(norms=4, dots=3, vectors=4) for arm in ARMS}
    b.update(candidate_scalar_geometry_records_max_by_role={role: 918 for role in ('P', 'U', 'S')},
             candidate_radius_checks_max_by_role={role: 918 for role in ('P', 'U', 'S', 'audit_P_record')},
             total_candidate_scalar_geometry_records_max=2754, total_candidate_radius_checks_max=3672,
             candidate_formula_coordinate_checks_max_by_role={role: 918 * 257 for role in ('P', 'U', 'S')},
             total_candidate_formula_coordinate_checks_max=3 * 918 * 257,
             extra_gradient_or_trial_for_geometry=0, construction_direction_attempts_per_state=1,
             interior_delta_candidates=1, postconstruction_repair_or_fallback_attempts=0)
    return b


def create():
    previous_binding = geometry.require_binding()
    meta = {k: p.load(ref) for k, ref in PARENTS.items() if k != 'report'}
    p.require(p.pin(PARENTS['report']['path']) == PARENTS['report'], 'completed terminal geometry report changed')
    done, proc = meta['completion'], meta['process']
    p.require(meta['binding'] == previous_binding and len(previous_binding['dependencies']) == 902
              and done['status'] == proc['status'] == 'complete_independently_integer_audited_terminal_geometry'
              and done['science'] == proc['science'] == PARENTS['science']
              and done['binding'] == proc['binding'] == PARENTS['binding'] and done['checks'] == 6235
              and done['source_JSON_passes'] == 4 and done['source_SHA_checks'] == 8 and done['total_projection_values'] == 105936
              and done['new_direction_vectors'] == done['new_forward_gradient_autograd_optimizer_trial_environment_calls'] == 0
              and done['formal_result'] is False and len(proc['processes']) == 2 and all(r['exit_code'] == 0 for r in proc['processes']),
              'completed integer-audited geometry provenance required')
    # Do not open done.analysis/done.audit: they are numerical research payloads.
    p.require(done['analysis']['path'] == str(geometry.OUTPUT) and done['audit']['path'] == str(geometry.AUDIT), 'geometry completion role mismatch')
    cfg = copy.deepcopy(parent.validate())
    cfg.update(kind='task7_joint_T_direction_under_original_M_bound', version=1, formal_result=False,
               status='scientific_freeze_only_execution_binding_required', parents=copy.deepcopy(PARENTS), budget=budget(),
               output=OUTPUT, artifacts=ARTIFACTS, code=dict(metadata=p.pin(Path(__file__).resolve()), tests=p.pin(TESTS)),
               plan=p.pin(PLAN), future_files={k: str(path) for k, path in FUTURE_FILES.items()}, future_contract=str(FUTURE_CONTRACT))
    cfg['scope'].update(changed_variable='candidate_direction_construction_only;near_original_M_cap_with_fixed_interior_reserve',
                        head_coverage='all18_original_auxiliary500;not_current_terminal_or_zero6_restart',
                        candidate_norm_matching='norm_d<=own_current_normG;interior_radius_R=(1-delta)*g;control_unchanged',
                        no_current_geometry_terminal_or_direction_vector_reuse=True)
    opt = cfg['optimization']
    opt.update(arms=ARMS, direction='control=old_guarded_T_kappa0.02;candidate=interior_joint_T_at_original_M_cap',
               direction_definition=dict(G='own_current_gradient_T', Gb='own_current_gradient_M_over7',
                   original_problem='min_G_dot_d_subject_norm_d<=g_and_7Gb_dot_d<=saved_B',
                   saved_B='-7.*0.01*g*b;same_original_expression_and_admission_bound;not_historical_terminal_B',
                   fixed_delta=INTERIOR_DELTA, delta_definition='sqrt_binary64_machine_epsilon=2^-26;not_data_selected_or_scanned',
                   R='g*(1.-delta);strict_0<R<g_or_stop', eta='((-B/7.)/b)/R;exact_frozen_Float64_order',
                   kappa='eta+delta;strict_0<eta<kappa<1_or_stop',
                   u='G/g', v='Gb/b', c='fsum(u_i*v_i);no_clamp',
                   inactive='if_c>=kappa:d_i=-R*u_i',
                   active='w_i=-u_i+c*v_i;q=norm_w;if_q>1e-12:d_i=R*(-kappa*v_i+sqrt(1-kappa*kappa)*(w_i/q))',
                   optimality='ideal_norm_ball_R_and_cap_Gb_dot_d<=-kappa*b*R;interior_subset_not_exact_original_cap_optimum',
                   ideal_M_slack='7Gb_dot_d<=B-7*delta*b*R;raw_original_B_still_verified',
                   selected='raw_norm_d<=g_and_selected_M<=B_and_selected_T<0',
                   exact='own_current_direction_exact_right_M<=B_and_T<0;original_gate_unchanged',
                   rounding='Float64;no_epsilon_in_predicates;no_cosine_clamp_reprojection_fallback_or_delta_retry'),
               required_margin_fraction_by_arm={arm: RHO for arm in ARMS},
               construction_margin_fraction_by_arm={ARMS[0]: .02, ARMS[1]: 'own_eta_plus_fixed_delta'},
               direction_radius_fraction_by_arm={ARMS[0]: 1., ARMS[1]: 1. - INTERIOR_DELTA},
               angle_resolution_applies_to='both_arms_construction_q_only;not_linear_feasibility_or_sign_tolerance',
               stop_priority_by_arm={ARMS[0]: copy.deepcopy(opt['stop_priority_by_arm'][ARMS[0]]),
                   ARMS[1]: ['update_cap', 'zero_gradient', 'zero_base_margin_unavailable', 'margin_unrepresentable',
                             'interior_radius_unrepresentable', 'interior_cap_unrepresentable', 'unresolved_interior_direction',
                             'selected_radius_violation', 'selected_M_margin_violation', 'selected_T_non_descent',
                             'exact_M_margin_violation', 'exact_T_non_descent', 'line_search_exhausted']},
               acceptance={arm: 'original_strict_T_Armijo_M500_and_step_M_nonincrease' for arm in ARMS},
               trial_caps={arm: 20 for arm in ARMS}, trial_index_range={arm: [0, 19] for arm in ARMS},
               alpha_minima={arm: 2. ** -19 for arm in ARMS}, path_order='seed_source_fold_then_guarded_T_control_then_interior_joint_T',
               new_head_order=['saved500', *(arm + '_final' for arm in ARMS)],
               candidate_direction_guard='raw_norm_d<=g;original_selected_and_exact_M<=B_T<0;no_relaxation',
               new_state_fields=NEW_STATE_FIELDS,
               radius_gate='candidate_only_direction_construction_admission;do_not_change_historical_control_norm_or_stops')
    cfg['new_head_layout']['order'] = opt['new_head_order']
    cfg['decision'].update(primary_comparison='all18_vs_current_guarded_T_control_and_retained_margin_zero_bisector',
                           primary_head_cue='progress_step_M_invariant_and_Tstrict_Mnonworse_vs_all4;direction_gate_pass_not_success',
                           next='report_all18_certificate_finite_step_stop_and_four_reference_results_before_new_Policy_or_loss_change')
    cfg['execution'].update(binding='new_roles_contract_science_scientific_binding_and_parent902closure_before_source_access',
                            future_envelope='full18_synthetic_independent_P_U_S_and_original_control7691_maximum_envelope_before_binding')
    cfg['audit_contract'].update(P='all18_original500_guarded_T_control_then_interior_joint_T;control_full_history',
                                only_changed_variable='candidate_direction_with_frozen_delta;original_loss_M_bound_and_all_trial_gates',
                                radius='own_norms_and_raw_candidate_radius_gate;S_never_copies_P_bound_or_radius',
                                geometry='no_cached_terminal_gradient_or_parameters_or_data_role_substitution;no_primary_numerical_import')
    cfg['freeze_access'] = dict(source_archive_or_historical_or_geometry_numerical_JSON_reads_or_SHA=0,
                               real_direction_gradient_trial_update_environment=0, metadata_and_human_vectors_only=True,
                               execution_binding_present=False)
    return cfg


def validate(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(cfg == create(), 'original-M-bound joint-T scientific freeze changed')
    return cfg


def require_binding(config=CONFIG, binding=BINDING):
    cfg = validate(config)
    bound = json.loads(Path(binding).read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['before_research_payload'] is True
              and bound['science_config'] == p.pin(config) and bound['scientific_freeze_binding'] == p.pin(FREEZE_BINDING)
              and bound['budget'] == cfg['budget'] and bound['CPU_environment'] == cfg['execution']['environment']
              and bound['future_output'] == OUTPUT and bound['future_artifacts'] == ARTIFACTS, 'new joint-T execution binding required')
    refs = bound['dependencies']
    paths = [ref['path'] for ref in refs]
    freeze = p.load(p.pin(FREEZE_BINDING))
    p.require(freeze['status'] == 'scientific_frozen_execution_not_bound' and freeze['science_config'] == p.pin(config)
              and len(paths) == len(set(paths)) and all(ref in refs for ref in freeze['dependencies'])
              and all(p.pin(path) in refs for path in [*FUTURE_FILES.values(), FUTURE_CONTRACT, Path(config), FREEZE_BINDING]),
              'new joint-T roles/scientific closure missing')
    p.require(all(ref in refs for ref in p.load(PARENTS['binding'])['dependencies']) and PARENTS['binding'] in refs,
              'parent902 closure missing')
    forbidden = {ref['path'] for ref in [*HISTORY_INPUTS.values(), *geometry.INPUTS.values()]}
    forbidden.update({str(geometry.OUTPUT), str(geometry.AUDIT)})
    p.require(not set(paths) & forbidden, 'research numerical payload cannot be dependency-hashed before binding')
    for ref in refs:
        p.require(Path(ref['path']).suffix in {'.py', '.md', '.json', '.toml'} and p.pin(ref['path']) == ref, 'bound joint-T dependency changed')
    return bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        p.write(CONFIG, create())
    else:
        validate()
    print(json.dumps({'status': 'joint_T_original_M_bound_scientific_freeze_no_research_payload'}))
