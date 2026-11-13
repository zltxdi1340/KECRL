"""Freeze a bounded257-coordinate local feasibility search; metadata/human only.

No research solver, payload loader, or execution authority is implemented here.
Solver failure/infeasibility/status0 is never a global mathematical certificate.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import math
from pathlib import Path

from experiments import libero_task7_reference_joint_reachability_protocol as parent

p, ROOT, IDS = parent.p, parent.ROOT, parent.IDS
STEM = 'libero-task7-bounded-constrained-feasibility'
CONFIG = ROOT / f'configs/{STEM}-diagnostic_v1.json'
FREEZE_BINDING = ROOT / f'configs/{STEM}-freeze-binding_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT / f'docs/{STEM}-plan-2026-11-12.md'
TESTS = ROOT / 'tests/test_libero_task7_bounded_constrained_feasibility_protocol.py'
ARTIFACTS = Path(f'/tmp/{STEM}-20261112-v1')
REFERENCES = parent.REFERENCES.copy()
ROLES = ['P_numpy_search', 'U_torch_search', 'S_P_torch', 'S_U_numpy']
RADIUS_FLOOR = 2. ** -20
UNIQUE_CAP, CALLBACK_CAP, NATIVE_CAP, MAJOR_CAP = 256, 4096, 1024, 100
FTOL, TOL = 1e-12, 1e-10
PARENTS = {
    'science': {'path': str(parent.CONFIG), 'sha256': 'f66a06736891529a45fa2cd5b791ca6841163ab99760e1e273c11f1f5880d145'},
    'scientific_binding': {'path': str(parent.FREEZE_BINDING), 'sha256': '613dbf8a471f600ce30433aa4598be0666fee9848b5e4957c3fac79fe994136d'},
    'binding': {'path': str(parent.BINDING), 'sha256': '251869eabe25c43b06a73c537fe76bb0816bba8e3479112038492195eed37550'},
    'completion': {'path': str(parent.ARTIFACTS / 'completion.json'), 'sha256': 'c6f24db92786c8dc3583ae4b32d5624e64e3bdd7137cca1e874cc856c1c0a0ee'},
    'process': {'path': '/tmp/libero-task7-reference-joint-reachability-20261111-v1-process.json', 'sha256': 'e437b0ea25ee6e55557562af2d2cdc28262fd3ea66266bff8a297db6ec55db8b'},
    'report': {'path': str(ROOT / 'docs/libero-task7-reference-joint-reachability-results-2026-11-11.md'), 'sha256': 'fa5dd7b3deaee984297a7af50eb56afcab0b2e2d588a034562b076c27860afb1'},
}
RUNTIME_CODE = {
    'scipy_wrapper': {'path': '/home/zl202621/.conda/envs/libero-kecrl/lib/python3.10/site-packages/scipy/optimize/_slsqp_py.py', 'sha256': '54f43317113e2f6943fa08454ef55c9bac243d1e782e21f87dbfbc35435682f2'},
    'scipy_native': {'path': '/home/zl202621/.conda/envs/libero-kecrl/lib/python3.10/site-packages/scipy/optimize/_slsqp.cpython-310-x86_64-linux-gnu.so', 'sha256': 'f9c29410611bb9db6d6f732b58e2b3ddf0b94a5e20368e1cf7d22218c8cf9987'},
}
STATE = IDS + ['family', 'evaluation_index', 'first_callback_index', 'first_native_call_index',
    'M', 'P', 'T', 'M_constraint', 'M_old_pass', 'T_old_pass', 'M_replayed_pass', 'T_replayed_pass',
    'primal_old', 'primal_replayed', 'clip_below_n', 'clip_low_tie_n', 'clip_inside_n', 'clip_high_tie_n', 'clip_above_n',
    'hinge_negative_n', 'hinge_zero_n', 'hinge_positive_n', 'context_sha256', 'u_sha256', 'parameter_sha256', 'branch_digest']
STATE_VECTORS = ['u', 'theta', 'gradient_M_u', 'gradient_T_u']
CALLBACK = IDS + ['family', 'callback_index', 'callback_kind', 'native_call_index', 'major_iteration_index',
    'raw_u_sha256', 'canonical_u_sha256', 'projected_coordinates', 'maximum_abs_projection',
    'evaluation_index', 'cache_hit', 'returned_scalar', 'returned_gradient_sha256']
NATIVE = IDS + ['family', 'native_call_index', 'mode_before', 'mode_after', 'major_before', 'major_after',
    'raw_u_sha256', 'canonical_u_sha256']
HEAD = IDS + ['family', 'n', 'solver_status', 'stop_reason', 'actual_major_iterations', 'actual_native_calls',
    'actual_callback_requests', 'actual_unique_evaluations', 'initial_M_constraint', 'terminal_u_sha256',
    'terminal_evaluation_index', 'terminal_unscored']
REFERENCE_FIELDS = ['M', 'P', 'T', 'extra_updates', 'n']
CAP_FIELDS = ['cached_M', 'cached_T', 'replayed_M', 'replayed_T', 'M_search_bound', 'T_witness_bound',
              'cached_M_active', 'cached_T_active', 'replayed_M_active', 'replayed_T_active']
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_bounded_constrained_feasibility.py',
    'audit': ROOT / 'experiments/audit_libero_task7_bounded_constrained_feasibility.py',
    'guard': ROOT / 'experiments/libero_task7_bounded_constrained_feasibility_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_bounded_constrained_feasibility.py',
}
FUTURE_CONTRACT = ROOT / f'docs/{STEM}-execution-contract-2026-11-12.md'
SCIENTIFIC_STOPS = {'solver_status_' + str(i) for i in (0, 2, 3, 4, 5, 6, 7, 8, 9)} | {
    'unique_evaluation_cap', 'callback_cap', 'native_call_cap', 'major_iteration_cap'}


def vector(values):
    parent.vector(values)
    return values


def domain(references):
    """Human vectors only; no reference payloads are opened by this function."""
    p.require(type(references) is dict and set(references) == set(REFERENCES), 'all four reference parameters required')
    for values in references.values():
        vector(values)
    center = list(references['bisector'])
    radii, lower, upper = [], [], []
    for j, c in enumerate(center):
        gaps = [math.nextafter(abs(values[j] - c), math.inf) for values in references.values()]
        radius = max(RADIUS_FLOOR, *gaps)
        lo, hi = c - radius, c + radius
        p.require(all(math.isfinite(v) for v in (*gaps, radius, lo, hi)) and 0 < radius and lo < c < hi,
                  'nonfinite or unrepresentable frozen domain; no radius repair')
        p.require(all(lo <= values[j] <= hi for values in references.values()), 'frozen domain fails to contain reference coordinate')
        radii.append(radius)
        lower.append(lo)
        upper.append(hi)
    return dict(center=center, radius=radii, lower=lower, upper=upper)


def canonicalize(raw):
    vector(raw)
    projected = [-1. if x < -1. else 1. if x > 1. else x for x in raw]
    return dict(u=projected, projected_coordinates=sum(x != y for x, y in zip(raw, projected)),
                maximum_abs_projection=max(abs(x - y) for x, y in zip(raw, projected)))


def theta_from_u(box, u):
    vector(u)
    p.require(set(box) == {'center', 'radius', 'lower', 'upper'} and all(-1. <= v <= 1. for v in u), 'canonical box coordinates required')
    for values in box.values():
        vector(values)
    values = []
    for c, r, lo, hi, x in zip(box['center'], box['radius'], box['lower'], box['upper'], u):
        p.require(r > 0 and lo < c < hi, 'invalid domain coordinate')
        value = c + r * x
        p.require(math.isfinite(value) and lo <= value <= hi, 'reconstructed parameter outside fixed physical domain')
        values.append(value)
    return values


def effective_caps(cached, replayed):
    p.require(all(set(v) == set(REFERENCES) for v in (cached, replayed)), 'four cached and replayed references required')
    for name in REFERENCES:
        parent.validate_point(cached[name])
        parent.validate_point(replayed[name])
        p.require(all(abs(cached[name][field] - replayed[name][field]) <= TOL for field in ('M', 'P', 'T')), 'reference calibration mismatch')
    old, fresh = parent.caps(cached), parent.caps(replayed)
    return dict(cached=old, replayed=fresh, M_search_bound=min(old['M'], fresh['M']),
                T_witness_bound=min(old['T'], fresh['T']))


def human_readout(features, targets, original, theta, box):
    """At most8 invented rows; definition/gradient oracle, never a research runner."""
    vector(theta)
    p.require(1 <= len(features) <= 8 and len(features) == len(targets) == len(original), 'human oracle row budget exceeded')
    p.require(all(type(row) is list and len(row) == 256 for row in features)
              and all(type(v) is float and math.isfinite(v) for row in features for v in row)
              and all(type(v) is float and math.isfinite(v) and abs(v) <= 1 for v in [*targets, *original]), 'human observation/target schema differs')
    errors, excess, qm, qt = [], [], [], []
    for row, y, p0 in zip(features, targets, original):
        raw = p0 + (math.fsum(x * w for x, w in zip(row, theta[:256])) + theta[256])
        p.require(math.isfinite(raw), 'human linear form nonfinite')
        pred = max(-1., min(1., raw))
        error, old = (pred - y) ** 2, (p0 - y) ** 2
        q = 2. * (pred - y) * float(-1. <= raw <= 1.)
        errors.append(error)
        excess.append(max(error - old, 0.))
        qm.append(q)
        qt.append((q / 7.) * (1. + float(error - old > 0.)))
    n = len(features)
    m, penalty = math.fsum(errors) / n, (math.fsum(excess) / n) / 7.
    grads = []
    for coefficients in (qm, qt):
        g = [math.fsum(coefficients[i] * features[i][j] for i in range(n)) / n for j in range(256)]
        g.append(math.fsum(coefficients) / n)
        grads.append([r * v for r, v in zip(box['radius'], g)])
    return dict(M=m, P=penalty, T=m / 7. + penalty, gradient_M_u=grads[0], gradient_T_u=grads[1])


def witnessed(box, u, theta, first, second, cached, first_replay, second_replay):
    expected = theta_from_u(box, u)
    p.require(all(a.hex() == b.hex() for a, b in zip(vector(theta), expected)), 'saved theta not canonical same-parameter reconstruction')
    for field in ('gradient_M_u', 'gradient_T_u'):
        vector(first[field])
        vector(second[field])
        p.require(all(abs(a - b) <= TOL for a, b in zip(first[field], second[field])), 'same-parameter analytic gradient mismatch')
    return parent.same_parameter_witness(theta, list(theta), {f: first[f] for f in ('M', 'P', 'T')},
        {f: second[f] for f in ('M', 'P', 'T')}, cached, first_replay, second_replay)


def classification(verified, stop):
    p.require(type(verified) is list and 1 <= len(verified) <= UNIQUE_CAP and all(type(v) is bool for v in verified), 'bounded complete visited-state audit required')
    p.require(stop in SCIENTIFIC_STOPS, 'technical failure is not scientific search completion')
    return 'verified_primal_witness_in_bounded_domain' if any(verified) else 'not_found_within_frozen_domain_and_budget'


def can_admit(kind, completed, increment=1):
    caps = dict(unique=UNIQUE_CAP, callback=CALLBACK_CAP, native=NATIVE_CAP, major=MAJOR_CAP)
    p.require(kind in caps and type(completed) is int and type(increment) is int
              and 0 <= completed <= caps[kind] and increment >= 0, 'invalid per-problem budget request')
    return completed + increment <= caps[kind]


def budget():
    n, head_count, roles = 50256, 18, 4
    states = head_count * UNIQUE_CAP
    forward = head_count * (4 + UNIQUE_CAP)
    scalar_projection = states * len(STATE) + head_count * CALLBACK_CAP * len(CALLBACK) + head_count * NATIVE_CAP * len(NATIVE)
    head_scalar_projection = head_count * (len(HEAD) + 4 * len(REFERENCE_FIELDS) + 4 * 3 + len(CAP_FIELDS))
    return dict(heads=18, parameter_histories=['P', 'U'], computation_roles=ROLES,
        solver_problems_total=36, starting_points_per_problem=1, starts_or_box_factors_scanned=1,
        major_iterations_max_per_problem=MAJOR_CAP, major_iterations_max_total=36 * MAJOR_CAP,
        native_calls_max_per_problem=NATIVE_CAP, native_calls_max_total=36 * NATIVE_CAP,
        callback_requests_max_per_problem=CALLBACK_CAP, callback_requests_max_total=36 * CALLBACK_CAP,
        canonicalization_coordinate_visits_max=36 * CALLBACK_CAP * 257,
        unique_states_max_per_problem=UNIQUE_CAP, unique_states_max_per_role=states, total_unique_state_readouts_max=roles * states,
        source_search_states_max=2 * states, same_parameter_audit_states_max=2 * states,
        reference_calibration_forwards_total=4 * head_count * roles, forward_calls_max_per_role=forward,
        forward_calls_max_total=roles * forward, forward_fit_rows_max_per_role=n * (4 + UNIQUE_CAP),
        forward_fit_rows_max_total=roles * n * (4 + UNIQUE_CAP),
        gradient_vectors_max_total=roles * states * 2, transpose_linear_forms_max_total=roles * states * 2,
        gradient_fit_rows_max_total=roles * n * UNIQUE_CAP * 2,
        all_linear_forms_max_total=roles * forward + roles * states * 2,
        all_linear_feature_multiplies_max_total=256 * roles * (n * (4 + UNIQUE_CAP) + n * UNIQUE_CAP * 2),
        parameter_forms_including_calibration_max=roles * forward,
        parameter_form_coordinate_writes_max=roles * forward * 257,
        scaled_gradient_coordinate_writes_max=roles * states * 2 * 257,
        callback_gradient_delivery_coordinates_max=36 * CALLBACK_CAP * 257,
        domain_constructions_total=72, domain_vector_coordinates_total=72 * 4 * 257,
        source_JSON_parses=6, source_JSON_SHA=12, source_NPZ_loads=50, source_NPZ_decodes=250, source_NPZ_SHA=100,
        source_reference_scalar_projections=4212, source_reference_parameter_coordinates=37008,
        new_primary_audit_JSON_parses=1, new_primary_audit_JSON_SHA=2,
        new_primary_audit_reference_and_domain_coordinates=37008,
        new_primary_audit_state_coordinates_max=states * 4 * 257,
        new_primary_audit_vector_coordinates_max=37008 + states * 4 * 257,
        new_primary_audit_state_callback_native_scalar_projections_max=scalar_projection,
        new_primary_audit_head_scalar_projections=head_scalar_projection,
        new_primary_audit_total_scalar_projections_max=scalar_projection + head_scalar_projection,
        state_rows_max_per_search=states, callback_rows_max_per_search=head_count * CALLBACK_CAP,
        native_rows_max_per_search=head_count * NATIVE_CAP, same_parameter_audit_rows_max=2 * states,
        original_Policy_CNN_optimizer_environment_calls=0, autograd_backward_finite_difference_calls=0,
        original_checkpoint_CSV_HDF5_RGB_initial_private_or_sealed_reads_including_SHA=0,
        new_NPZ_checkpoint_or_deployable_Policy_writes=0, extra_starts_restarts_radii_ftols_or_loss_scans=0,
        technical_retries_overwrites=0, unused_budget_transferred_between_heads_or_histories=False)


def create():
    previous, bound = parent.require_binding()
    meta = {name: p.load(ref) for name, ref in PARENTS.items() if name != 'report'}
    p.require(p.pin(PARENTS['report']['path']) == PARENTS['report'], 'completed bounded-grid report changed')
    done, process = meta['completion'], meta['process']
    p.require(meta['science'] == previous and meta['binding'] == bound and len(bound['dependencies']) == 943
              and done['status'] == process['status'] == 'complete_independently_audited_reference_joint_reachability'
              and done['science'] == process['science'] == PARENTS['science']
              and done['binding'] == process['binding'] == PARENTS['binding']
              and done['scientific_binding'] == PARENTS['scientific_binding']
              and done['checks'] == 208967 and done['core_forwards_total'] == 13680
              and done['verified_heads'] == {'P': 0, 'U': 0} and done['both_histories_heads'] == 0
              and done['optimizer_gradient_Policy_CNN_environment_calls'] == 0 and done['formal_result'] is False
              and len(process['processes']) == 2 and all(r['exit_code'] == 0 for r in process['processes'])
              and all(v == previous['budget'][k] for k, v in done['combined_budget_envelope'].items()), 'completed paired-grid provenance required')
    p.require(importlib.metadata.version('scipy') == '1.10.1', 'installed SciPy version differs')
    for ref in RUNTIME_CODE.values():
        p.require(p.pin(ref['path']) == ref, 'frozen installed SLSQP code differs')
    return dict(kind='task7_fixed_representation_bounded_local_constrained_feasibility', version=1, formal_result=False,
        status='scientific_freeze_only_execution_binding_required', parents=copy.deepcopy(PARENTS), runtime_code=copy.deepcopy(RUNTIME_CODE),
        inputs=copy.deepcopy(previous['inputs']), source_JSON=copy.deepcopy(previous['source_JSON']), source_roles=copy.deepcopy(previous['source_roles']),
        reference_projections=copy.deepcopy(previous['projections']), identities=copy.deepcopy(previous['identities']), references=REFERENCES.copy(),
        scope=dict(original_Policy_updates=500, future_stabilization_steps=5, loss_lambda=1,
            source='own20_fit_source_support0..29_tge1_including_neutral;encoder_seen_all30;not_unseen_generalization',
            original_main_cue='0/18_unchanged;G_stopped_H_unrun', formal_evaluation_data_sealed=True,
            no_new_original_Policy_CNN_training_or_environment=True, SPT_Knowledge_Module_changes=False),
        domain=dict(dimension=257, center='own_history_bisector_parameters',
            radius='max(2^-20,nextafter(abs(reference_j-center_j),+inf)_over_all4_refs)',
            radius_floor=RADIUS_FLOOR, radius_candidates=1, lower='center-radius', upper='center+radius',
            variable='u_in_closed[-1,1]^257;theta_j=center_j+(radius_j*u_j);Float64_no_FMA_reassociation',
            canonicalization='all_callback_raw_u_exact_component_clip_to[-1,1];log_raw_and_canonical_hash_projection_count_and_max_change',
            seed='one_u0_zero257_per_history_head;initial_M_feasibility_not_assumed',
            unusable='nonfinite_or_unrepresentable_center_radius_bounds_fails_closed;no_enlarge_shrink_rescale_or_restart',
            meaning='bounded_full_coordinate_local_search_beyond_six_edges;not_all_parameter_space_or_guaranteed_full_box_coverage'),
        objective=copy.deepcopy(previous['objective']),
        optimization=dict(method='scipy.optimize.minimize_SLSQP_1.10.1', fun='original_T=M/7+P;no_new_weight_or_penalty',
            constraint='one_inequality_c(u)=min(original_Mcap,replayed_Mcap)-M(theta(u))>=0',
            target='strict_T_below_both_original_and_replayed_all4_caps;no_T_margin_epsilon',
            bounds=[-1., 1.], initial_u='zero257', options=dict(maxiter=100, ftol=FTOL, disp=False, iprint=0),
            analytic_jacobians=True, finite_difference=False, random_seed=None,
            cache='one_paired_M_P_T_and_gradientM_gradientT_eval_per_distinct_canonical_u_binary64_bytes;exact_cache_hits_only',
            cache_immutability='copy_canonical_u_theta_and_gradients;never_store_mutable_native_views;returned_jacobian_delivery_must_not_mutate_cache',
            gradients='qM=2*(clip(raw)-y)*1[-1<=raw<=1];qT=(qM/7)*(1+1[error-original_error>0]);mean(q*Zbar);grad_u=radius*grad_theta',
            nonsmooth='clip_ties_use_inclusive1;hinge_ties_use0;chosen_analytic_convention_not_smoothness_KKT_or_global_certificate',
            callbacks='T,gradient_T,c,gradient_c,major_monitor;all_counted;major_monitor_no_extra_fit_eval',
            native='instrument_scipy.optimize._slsqp_py.slsqp_reverse_communication_calls_and_major_iteration_counters',
            finite_difference_guard='deny_approx_derivative_and_epsilon_probes;native_BFGS_QP_not_counted_as_fit_gradients',
            caps='stop_before_work_exceeding_unique256_callback4096_native1024_major100;unused_budget_not_transferred',
            after_stop='no_uncached_final_rescore_or_extra_trial;return_all_visited_states;terminal_unscored_if_not_in_cache',
            witness='finding_a_witness_does_not_early_stop_solver_or_drop_other_heads;no_checkpoint_selection'),
        audit=dict(searches=['P_numpy_search', 'U_torch_search'], same_parameter=['S_P_torch', 'S_U_numpy'],
            S_solver_calls=0, all_visited_states=True, independently_rebuild_domains_and_theta=True,
            parameter_comparison='saved_u_theta_reconstruction_coordinate_hex_exact;same_actual_theta_for_M_T_and_gradients',
            numeric_absolute_tolerance=TOL, relative_tolerance=0, raw_sign_tolerance=0,
            reference_calibration='each_role4_forwards;original_vs_fresh_M_P_T_abs1e-10;caps_never_replaced_or_weakened',
            primal='both_implementations_inside_domain_and_all_original_and_fresh4refs_M<=_T<;no_P_bound',
            cached_and_fresh_scalar_gate_disagreement='retain_raw_even_within_numeric_tolerance;not_verified',
            callback_checks='all_scalar_and_gradient_returns_cache_mapping_projection_and_state_identity;constraint_jac=-gradient_M_u',
            independence='independent_numeric_formula_and_domain_reconstruction;SLSQP_vendor_shared_by_P_U_not_independent_global_optimization',
            negative_certificate='none;solver_status0_4_9_KKT_or_budget_exhaustion_cannot_certify_box_or_global_infeasibility'),
        projections=dict(state_scalars=STATE, state_vectors=STATE_VECTORS, callback_scalars=CALLBACK, native_scalars=NATIVE,
            head_scalars=HEAD, reference_scalars=REFERENCE_FIELDS, head_cap_scalars=CAP_FIELDS,
            head_references_and_domain='4x257_reference_theta_plus4x257_center_radius_lower_upper_per_head',
            S_output='scalar_same_parameter_M_P_T_and_gradient_comparison_summaries;no_duplicate_S_full_vectors',
            callback_vectors='raw_u_hash_and_projection_summary_only;no_extra_full_raw_vectors',
            whole_file_decode_includes_unprojected_bytes=True),
        decision=dict(positive='verified_primal_witness_in_bounded_domain',
            negative='not_found_within_frozen_domain_and_budget;bounded_and_global_reachability_unresolved',
            technical_failure='failed_no_retry;retain_actual_partial_costs;do_not_call_search_negative',
            scientific_stops=sorted(SCIENTIFIC_STOPS), not_proven='global_or_box_infeasibility_optimality_original500_to_endpoint_path_closedloop_or_Module_qualification',
            report='all18_each_history_source_seed_fold;raw_solver_status_actual_iterations_callback_native_unique_counts_all_witnesses_and_counterexamples;no_pvalues'),
        budget=budget(), execution=dict(python='3.10.22', numpy='1.22.4', scipy='1.10.1', torch='2.4.1+cu124', device='CPU',
            seconds_primary=900, seconds_audit=1800, overall_seconds=3000, RSS_primary=1073741824, RSS_audit=2147483648,
            input_original_JSON_bytes_max=201326592, new_primary_JSON_bytes_max=268435456, output_JSON_bytes_max=268435456,
            output_CSV_bytes_max=67108864, environment=copy.deepcopy(previous['execution']['environment']),
            torch_threads=1, torch_interop_threads=1, autograd=False,
            required_before_execution='new_main_guard_independent_U_S_contract_full18_synthetic_maximum_envelope_and_execution_hash_binding',
            no_research_payload_reads_or_SHA_at_scientific_freeze=True),
        code=dict(metadata=p.pin(Path(__file__).resolve()), tests=p.pin(TESTS)), plan=p.pin(PLAN),
        future_files={k: str(v) for k, v in FUTURE_FILES.items()}, future_contract=str(FUTURE_CONTRACT), artifacts=str(ARTIFACTS))


def forbidden_payloads(cfg):
    return parent.forbidden_payload_paths(cfg) | {str(parent.ARTIFACTS / 'primary-result.json'), str(parent.ARTIFACTS / 'independent-audit.json')}


def freeze_dependencies(cfg):
    old = p.load(PARENTS['binding'])['dependencies']
    refs = old + list(PARENTS.values()) + list(RUNTIME_CODE.values()) + list(cfg['code'].values()) + [cfg['plan'], p.pin(CONFIG)]
    by_path = {}
    for ref in refs:
        p.require(ref['path'] not in by_path or by_path[ref['path']] == ref, 'conflicting metadata dependency')
        by_path[ref['path']] = ref
    p.require(not set(by_path) & forbidden_payloads(cfg), 'research payload cannot enter freeze dependency SHA list')
    return [by_path[name] for name in sorted(by_path)]


def validate():
    cfg = json.loads(CONFIG.read_text())
    p.require(cfg == create(), 'bounded constrained-feasibility freeze changed')
    return cfg


def validate_freeze():
    cfg = validate()
    bound = json.loads(FREEZE_BINDING.read_text())
    p.require(bound['status'] == 'scientific_frozen_execution_not_bound' and bound['science'] == p.pin(CONFIG)
              and bound['dependencies'] == freeze_dependencies(cfg) and bound['budget'] == cfg['budget']
              and bound['research_payload_reads_including_SHA'] == bound['real_search_or_gradient_calls'] == 0, 'scientific binding required')
    for ref in bound['dependencies']:
        p.require(p.pin(ref['path']) == ref, 'bounded-freeze dependency changed')
    return cfg, bound


def require_binding():
    cfg, frozen = validate_freeze()
    p.require(BINDING.is_file(), 'execution binding absent; scientific freeze denies all research search/payload access')
    bound = json.loads(BINDING.read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['before_payload'] is True
              and bound['science'] == p.pin(CONFIG) and bound['scientific_binding'] == p.pin(FREEZE_BINDING)
              and bound['budget'] == cfg['budget'] and bound['CPU_environment'] == cfg['execution']['environment']
              and bound['full18_synthetic_passed'] is True and bound['maximum_budget_envelope_passed'] is True,
              'complete independent-search execution binding required')
    refs = bound['dependencies']
    paths = [r['path'] for r in refs]
    exact_paths = {r['path'] for r in frozen['dependencies']} | {str(FREEZE_BINDING), str(FUTURE_CONTRACT)} | {str(file) for file in FUTURE_FILES.values()}
    p.require(len(paths) == len(set(paths)) and not set(paths) & forbidden_payloads(cfg)
              and set(paths) == exact_paths
              and all(r in refs for r in frozen['dependencies']) and p.pin(FREEZE_BINDING) in refs, 'execution closure incomplete or includes payload')
    for file in [*FUTURE_FILES.values(), FUTURE_CONTRACT]:
        p.require(file.is_file() and p.pin(file) in refs, 'new constrained-search role/contract not bound')
    for ref in refs:
        p.require(p.pin(ref['path']) == ref, 'execution dependency changed')
    return cfg, bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        p.write(CONFIG, create())
    else:
        validate_freeze()
    print(json.dumps(dict(status='metadata_human_only_bounded_local_feasibility_freeze', research_payload_reads_including_SHA=0)))
