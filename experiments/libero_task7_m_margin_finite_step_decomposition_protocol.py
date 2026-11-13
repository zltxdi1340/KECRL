"""Metadata/synthetic freeze for saved negative-margin finite-step accounting."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments import libero_task7_m_descent_margin_direction_protocol as parent

p, ROOT = parent.p, parent.ROOT
IDS = ['task', 'seed', 'arm', 'fold', 'mode']
SOURCES = ['A_original', 'B_fomaml']
SEEDS = [20261106, 20261107, 20261108]
CANDIDATE = 'norm_matched_T_priority_M_margin'
STEM = 'libero-task7-m-margin-finite-step-decomposition'
CONFIG = ROOT / f'configs/{STEM}-readonly_v1.json'
FREEZE_BINDING = ROOT / f'configs/{STEM}-freeze-binding_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT / f'docs/{STEM}-plan-2026-11-11.md'
TESTS = ROOT / 'tests/test_libero_task7_m_margin_finite_step_decomposition_protocol.py'
ARTIFACTS = Path('/tmp/libero-task7-m-margin-finite-step-decomposition-20261111-v1')
OUTPUT, AUDIT = ARTIFACTS / 'analysis.json', ARTIFACTS / 'independent-audit.json'
TOLERANCE = 1e-10
WINDOWS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
ALPHA_INDEX_BINS = [(0, 4), (5, 9), (10, 14), (15, 19)]
CLIP = ['clip_below_n', 'clip_low_tie_n', 'clip_inside_n', 'clip_high_tie_n', 'clip_above_n']
HINGE = ['hinge_negative_n', 'hinge_zero_n', 'hinge_positive_n']
STATE = IDS + ['optimization_arm', 'extra_updates', 'n', 'component3_mse', 'penalty',
               'trainable_objective', 'exact_MSE_slope', 'exact_penalty_slope',
               'directional_derivative', 'required_M_slope_bound', 'selected_margin_met',
               'exact_margin_met', 'branch_digest'] + CLIP + HINGE
TRIAL = IDS + ['optimization_arm', 'from_extra_updates', 'trial_index', 'alpha',
               'component3_mse', 'penalty', 'trainable_objective', 'current_component3_mse',
               'delta_component3_mse', 'start500_component3_mse', 'slope', 'armijo_bound',
               'armijo_pass', 'mse_envelope_pass', 'mse_gate_applied', 'accepted', 'reason']
PARENTS = {
    'science': {'path': str(parent.DEFAULT_CONFIG), 'sha256': '0101f655813b94f0036048eefc2f3e2c922fb4db20da06e7ccbb637cfacc6600'},
    'binding': {'path': str(parent.DEFAULT_BINDING), 'sha256': '18002a8ad84979eb25ef889f58b1ee7099a264ab2e10e579d4658b39b6315c3f'},
    'result': {'path': parent.OUTPUT, 'sha256': 'e38929b55b41a2c8db404bc02d8abe96745e562f624cfd1316d8133b8569efad'},
    'completion': {'path': parent.ARTIFACTS + '/completion.json', 'sha256': '0a682ed0b215e5d652a1d93c2b7900fcfac4e606c8165cd2fce0a03aba726d21'},
    'process': {'path': '/tmp/libero-task7-m-descent-margin-direction-20261111-v1-process.json', 'sha256': 'fff3d8127a338bd14b4989a2d8493477a35753fdcaeb7f817f99b7712ffad572'},
    'report': {'path': str(ROOT / 'docs/libero-task7-m-descent-margin-direction-results-2026-11-11.md'), 'sha256': '4ddee803b70b62c22f2c960bc6ca5bb3a726aec82fbabd690e42e121d1fd8345'},
}
INPUTS = {
    'P': {'path': parent.ARTIFACTS + '/primary-result.json', 'sha256': '4c5a316f2d1824cb517b5edeb2d54dcc463146a45c7f48c078e679e3b319e1a4'},
    'U': {'path': parent.ARTIFACTS + '/independent-audit.json', 'sha256': '95afa7fab0b5f3749106f99858fefaa68aa3883a260b1a950b600f1f361ecba1'},
}
FUTURE_FILES = {
    'analysis': ROOT / 'experiments/analyze_libero_task7_m_margin_finite_step_decomposition.py',
    'audit': ROOT / 'experiments/audit_libero_task7_m_margin_finite_step_decomposition.py',
    'guard': ROOT / 'experiments/libero_task7_m_margin_finite_step_decomposition_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_m_margin_finite_step_decomposition.py',
}
FUTURE_CONTRACT = ROOT / f'docs/{STEM}-execution-contract-2026-11-11.md'


def finite(*values):
    p.require(all(type(v) in (int, float) and math.isfinite(v) for v in values),
              'finite real scalars required; bool is not a measurement')


def close(left, right):
    finite(left, right)
    p.require(abs(left - right) <= TOLERANCE, 'saved scalar conservation differs')


def alpha_group(index):
    p.require(type(index) is int and 0 <= index < 20, 'only saved twenty-level grid')
    return index // 5


def update_window(from_update):
    p.require(type(from_update) is int and 0 <= from_update < 50, 'only fifty saved rounds')
    return from_update // 10


def ratio(numerator, denominator):
    """No small-value epsilon; retain undefined/overflow cases explicitly."""
    finite(numerator, denominator)
    if denominator <= 0:
        return {'value': None, 'status': 'nonpositive_denominator'}
    value = numerator / denominator
    return {'value': value if math.isfinite(value) else None,
            'status': 'finite' if math.isfinite(value) else 'nonfinite_ratio'}


def trial_reference(before, trial, m500):
    """Human-made scalars only until an independent execution is bound."""
    keys = ['component3_mse', 'penalty', 'trainable_objective']
    finite(m500, *(before[k] for k in keys), *(trial[k] for k in keys),
           *(before[k] for k in ('exact_MSE_slope', 'exact_penalty_slope',
                                'directional_derivative', 'required_M_slope_bound')))
    p.require(m500 >= 0 and all(row[k] >= 0 for row in (before, trial) for k in keys),
              'nonnegative loss terms required')
    for row in (before, trial):
        close(row['trainable_objective'], row['component3_mse'] / 7 + row['penalty'])
    sm, sp, st = (before[k] for k in ('exact_MSE_slope', 'exact_penalty_slope', 'directional_derivative'))
    close(st, sm / 7 + sp)
    p.require(sm <= before['required_M_slope_bound'] < 0 and st < 0,
              'saved candidate must have strict exact negative M/T certificate')
    index, alpha = trial['trial_index'], trial['alpha']
    alpha_group(index); finite(alpha)
    p.require(alpha == 2. ** (-index), 'alpha outside the saved fixed grid')
    for field in ('armijo_pass', 'mse_envelope_pass', 'mse_gate_applied', 'accepted'):
        p.require(type(trial[field]) is bool, 'trial decisions must be raw bool')
    armijo_bound = before['trainable_objective'] + .0001 * alpha * st
    armijo = trial['trainable_objective'] < before['trainable_objective'] and trial['trainable_objective'] <= armijo_bound
    envelope = trial['component3_mse'] <= m500
    p.require(trial['armijo_pass'] == armijo and trial['mse_envelope_pass'] == envelope
              and trial['mse_gate_applied'] and trial['accepted'] == (armijo and envelope),
              'saved finite-step decision differs; no sign tolerance')
    reason = ('accepted' if armijo and envelope else 'armijo_and_mse_rejected' if not armijo and not envelope
              else 'armijo_rejected' if not armijo else 'mse_envelope_rejected')
    p.require(trial['reason'] == reason, 'saved rejection reason differs')
    close(trial['armijo_bound'], armijo_bound)
    close(trial['slope'], st); close(trial['start500_component3_mse'], m500)
    close(trial['current_component3_mse'], before['component3_mse'])
    dm = trial['component3_mse'] - before['component3_mse']
    dp = trial['penalty'] - before['penalty']
    dt = trial['trainable_objective'] - before['trainable_objective']
    close(trial['delta_component3_mse'], dm); close(dt, dm / 7 + dp)
    predicted_m, predicted_p, predicted_t = alpha * sm, alpha * sp, alpha * st
    rm, rp, rt = dm - predicted_m, dp - predicted_p, dt - predicted_t
    finite(dm, dp, dt, predicted_m, predicted_p, predicted_t, rm, rp, rt)
    close(rt, rm / 7 + rp)
    p.require(before['component3_mse'] <= m500, 'current state exceeds original M500')
    slack = m500 - before['component3_mse']
    return dict(dM=dm, dP=dp, dT=dt, predicted_M=predicted_m, predicted_P=predicted_p,
                predicted_T=predicted_t, remainder_M=rm, remainder_P=rp, remainder_T=rt,
                M_class='up' if dm > 0 else 'nonincrease', M_raw_sign='up' if dm > 0 else 'down' if dm < 0 else 'zero',
                previous_M_gate_pass=trial['component3_mse'] <= before['component3_mse'],
                M500_slack_before=slack, M500_slack_after=m500 - trial['component3_mse'],
                penalty_decreased=dp < 0, penalty_exceeds_M_cost=dm > 0 and -dp > dm / 7,
                penalty_surplus=-dp - dm / 7,
                compensation_ratio=ratio(-dp, dm / 7) if dm > 0 else {'value': None, 'status': 'not_M_up'},
                remainder_over_predicted_descent=ratio(rm, -predicted_m),
                fraction_of_slack_spent=ratio(dm, slack) if dm > 0 else {'value': None, 'status': 'not_M_up'})


def step_reference(before, after, trial, m500):
    p.require(trial['accepted'] is True, 'step must be an actually accepted trial')
    result = trial_reference(before, trial, m500)
    for field in ('component3_mse', 'penalty', 'trainable_objective'):
        close(after[field], trial[field])
    p.require(after['component3_mse'] <= m500 and after['trainable_objective'] < before['trainable_objective'],
              'accepted next state violates saved raw gates')
    state_dm = after['component3_mse'] - before['component3_mse']
    result.update(state_dM=state_dm,
                  state_trial_M_sign_equal=(state_dm > 0) == (result['dM'] > 0))
    return result


def budget():
    counts = {'P': {'all_states': 1017, 'all_trials': 7690}, 'U': {'all_states': 1010, 'all_trials': 7641}}
    rows = dict(steps=1800, saved_trials=14034, rounds=1800, heads=36,
                head_window_outcome=360, alpha_source_outcome=160, source_seed_outcome=24,
                source_outcome=8, engine_outcome=4, matched_head_window_alpha=720, P_U_round_comparison=900)
    label_reads = sum(c['all_states'] + c['all_trials'] for c in counts.values())
    fields = label_reads + 1836 * (len(STATE) - 1) + 14034 * (len(TRIAL) - 1)
    return dict(stages=2, source_JSON_passes_per_stage=2, source_SHA_per_stage=4,
                total_source_JSON_passes=4, total_source_SHA=8, primary_result_JSON_reads_by_audit=1,
                primary_result_SHA_by_audit=2, physical_table_rows_by_engine=counts,
                per_stage_arm_label_reads=label_reads, per_stage_candidate_states=1836,
                per_stage_candidate_trials=14034, per_stage_accepted_trials=1800, per_stage_rejected_trials=12234,
                per_engine_candidate_states=918, per_engine_candidate_trials=7017,
                per_engine_accepted=900, per_engine_M_up=407, per_engine_M_nonincrease=493,
                per_stage_scalar_field_projections=fields, total_scalar_field_projections=2 * fields,
                steps_per_head=50, states_per_head=51, heads_per_engine=18,
                trial_indices=[0, 19], output_rows_per_stage=rows,
                total_output_rows=2 * sum(rows.values()), source_CSV_NPZ_checkpoint_reads_or_SHA=0,
                extra_forward_gradient_trial_update_environment_calls=0, technical_retries=0, overwrite=0)


def create():
    bound = parent.require_binding()
    docs = {k: p.load(v) for k, v in PARENTS.items() if k != 'report'}
    p.require(p.pin(PARENTS['report']['path']) == PARENTS['report'], 'completed report changed')
    science, binding, result, done, process = [docs[k] for k in ('science', 'binding', 'result', 'completion', 'process')]
    status = 'complete_independently_audited_M_margin_direction'
    p.require(binding == bound and len(bound['dependencies']) == 837
              and result['status'] == done['status'] == process['status'] == status
              and result['primary'] == INPUTS['P'] and result['audit'] == done['audit'] == INPUTS['U']
              and done['result'] == PARENTS['result']
              and all(obj['science_config'] == PARENTS['science'] and obj['binding'] == PARENTS['binding']
                      for obj in (result, done, process))
              and len(process['processes']) == 2 and all(row['exit_code'] == 0 for row in process['processes']),
              'completed independently audited margin provenance required')
    p.require((result['primary_updates'], result['independent_U_updates'], result['local_S_updates']) == (981, 974, 0)
              and len(result['robust_pairs']) == 18 and all(not row['robust_paired_cue'] for row in result['robust_pairs']),
              'parent completion coverage/cues changed')
    identities = [{k: h[k] for k in IDS} for h in science['cohorts']]
    p.require(len(identities) == 18 and {tuple(h[k] for k in IDS) for h in identities} == {
        (7, seed, source, fold, 'harm_protected_affine') for seed in SEEDS for source in SOURCES for fold in range(3)},
        'all original eighteen identities required')
    return dict(kind='task7_saved_M_margin_finite_step_decomposition', version=1, formal_result=False,
                status='scientific_freeze_only_execution_binding_required', parents=copy.deepcopy(PARENTS),
                inputs=copy.deepcopy(INPUTS), identities=identities, candidate=CANDIDATE,
                input_contract=dict(status_by_engine={'P': 'complete_M_margin_pending_same_parameter_audit',
                                                      'U': 'passed_independent_M_margin_direction_audit'},
                                    science_config=PARENTS['science'], binding=PARENTS['binding'],
                                    U_primary=INPUTS['P'], formal_result=False,
                                    order='validate_status_pins_identity_then_candidate_scalar_projection'),
                scope=dict(original_Policy_updates=500, future_stabilization_steps=5, loss_lambda=1,
                           fixed_rho=.01, construction_kappa=.02, current_M500_and_last_accepted_rules_unchanged=True,
                           source='own20_fit_meta_support0..29_tge1_encoder_seen_all30',
                           all18_no_focus_or_checkpoint_selection=True, outcome_conditioned=True,
                           original_training_forward_gradient_environment_or_sealed_calls=0,
                           source_meta_query_validation_heldout_qualification_reserved_final_access=False),
                projections=dict(states=STATE, candidate_trials=TRIAL,
                                 other_rows='optimization_arm_label_only;control_numeric_fields_not_used',
                                 unselected='vectors_parameters_endpoints_heads_local_tables_not_used',
                                 physical_decode='whole_pinned_JSON;semantic_scalar_whitelist_only'),
                design=dict(reference='407_actual_trial_M_up_vs493_nonincrease_each_engine;all18_heads',
                            classification='raw_accepted_trial_M_minus_current_state_M;save_next_state_sign_disagreement',
                            accounting='delta_T=delta_M/7+delta_P;R_T=R_M/7+R_P',
                            finite_remainder='actual_saved_trial_change_minus_alpha_saved_exact_right_derivative',
                            M500_slack='original_own500_M_minus_current_or_trial_M',
                            ratios='null_with_reason_on_nonpositive_denominator_or_nonfinite_result;no_epsilon',
                            step_reference='all_accepted_and_all_preceding_actual_rejections;first_acceptance_only',
                            hypothetical_previous_M_gate='flag_saved_trials_only;no_counterfactual_path_or_budget_prediction',
                            windows=[list(w) for w in WINDOWS], alpha_grid_indices=list(range(20)),
                            alpha_index_bins=[list(w) for w in ALPHA_INDEX_BINS],
                            matched='within_engine_head_10update_window_5index_bin;both_outcomes_or_null',
                            summaries='counts_mean_median_signed_sums_ranges;equal_head_fold_then_seed;empty_null',
                            branches='endpoint_digest_and_clip_hinge_counts;unchanged_does_not_exclude_between_point_crossing',
                            P_U='compare_same_ID_round_descriptively;no_cross_path_local_tolerance_or_forced_equality',
                            numeric_absolute_tolerance=TOLERANCE, numeric_relative_tolerance=0,
                            raw_decisions='no_epsilon_for_delta_signs_or_original_Armijo_M500_acceptance',
                            no_pvalues_causal_threshold_optimization_or_safe_alpha_estimation=True),
                decision=dict(validation='every_identity_k0to50;all_rounds_first_accepted;900_steps407up493nonincrease_per_engine;conservation',
                              interpretation='descriptive_support_and_counterevidence_for_finite_step_remainder_P_compensation_and_slack',
                              stopping='any_provenance_projection_budget_or_conservation_mismatch_fail_closed_keep_partial_no_retry',
                              training='none;0of18_parent_condition_G_H_and_formal_sealing_unchanged',
                              next='only_after_independent_scalar_audit_decide_whether_to_freeze_one_new_training_or_acceptance_variable'),
                budget=budget(),
                execution=dict(python='3.10.22', device='CPU', seconds_per_stage=120, overall_seconds=300,
                               RSS_bytes=536870912, input_JSON_bytes_max=100663296, output_JSON_bytes_max=33554432,
                               environment={'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1',
                                            'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                               independent_audit='new_process_stdlib_only;no_import_of_primary_numeric_or_summary_helpers',
                               source_order=['P', 'U'], audit_reads_primary_output_once=True),
                code={'metadata': p.pin(Path(__file__).resolve()), 'tests': p.pin(TESTS)}, plan=p.pin(PLAN),
                future_files={k: str(v) for k, v in FUTURE_FILES.items()}, future_contract=str(FUTURE_CONTRACT),
                artifacts=str(ARTIFACTS), output=str(OUTPUT), audit=str(AUDIT),
                freeze_access=dict(source_payload_JSON_reads_or_SHA=0, real_scalar_decomposition=0,
                                   CSV_NPZ_checkpoint_forward_gradient_trial_update_environment=0,
                                   metadata_and_human_made_scalars_only=True, execution_binding_present=False))


def validate(path=CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(cfg == create(), 'finite-step scientific freeze differs')
    return cfg


def require_binding(config=CONFIG, binding=BINDING):
    cfg = validate(config)
    bound = json.loads(Path(binding).read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['before_research_payload'] is True
              and bound['science_config'] == p.pin(config) and bound['scientific_freeze_binding'] == p.pin(FREEZE_BINDING)
              and bound['budget'] == cfg['budget'] and bound['CPU_environment'] == cfg['execution']['environment']
              and bound['future_output'] == str(OUTPUT) and bound['future_audit'] == str(AUDIT)
              and bound['future_artifacts'] == str(ARTIFACTS),
              'new finite-step execution binding required')
    refs = bound['dependencies']; paths = [r['path'] for r in refs]
    required = [*FUTURE_FILES.values(), FUTURE_CONTRACT, Path(config), FREEZE_BINDING]
    p.require(len(paths) == len(set(paths)) and all(p.pin(path) in refs for path in required), 'new execution roles missing')
    freeze = json.loads(FREEZE_BINDING.read_text())
    p.require(freeze['science_config'] == p.pin(config) and freeze['status'] == 'scientific_frozen_execution_not_bound'
              and all(r in refs for r in freeze['dependencies'])
              and all(r in refs for r in p.load(PARENTS['binding'])['dependencies']) and PARENTS['binding'] in refs,
              'scientific and parent837 closure missing')
    for ref in refs:
        p.require(Path(ref['path']).suffix in {'.py', '.md', '.json', '.toml'} and p.pin(ref['path']) == ref,
                  'bound dependency changed')
    return bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        p.write(CONFIG, create())
    else:
        validate()
    print(json.dumps({'status': 'finite_step_scientific_freeze_no_research_payload'}))
