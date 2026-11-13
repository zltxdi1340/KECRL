"""Metadata-only freeze: add a fixed-start fit-MSE gate to lambda1 Armijo."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from experiments import libero_task7_optimization_path_decomposition_protocol as parent

optimization_parent = parent.parent
p, ROOT = parent.p, parent.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-mse-nonregression-acceptance-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-mse-nonregression-acceptance-execution-binding_v1.json'
PLAN = ROOT / 'docs/libero-task7-mse-nonregression-acceptance-plan-2026-11-09.md'
TESTS = ROOT / 'tests/test_libero_task7_mse_nonregression_acceptance_protocol.py'
OUTPUT = '/tmp/libero-task7-mse-nonregression-acceptance-20261109-v1.json'
ARTIFACTS = '/tmp/libero-task7-mse-nonregression-acceptance-artifacts-20261109-v1'
PARENTS = {
    'science': {'path': str(parent.DEFAULT_CONFIG), 'sha256': '6d01cf826d9fd1c65f5c3cbed937118ea89afcbde8c8a2f153bb5fc8b7621eb3'},
    'binding': {'path': str(parent.DEFAULT_BINDING), 'sha256': '7604960266a5f5b738260377b297a931cfb50f07be0f1e6c4f7d7ca43c2bfde3'},
    'result': {'path': '/tmp/libero-task7-optimization-path-decomposition-20261109-v1.json', 'sha256': '1e757799bfa39879e0a349fde2cd199c39d0e293a39195fbebd189489fc893ba'},
    'audit': {'path': '/tmp/libero-task7-optimization-path-decomposition-artifacts-20261109-v1/independent-audit.json', 'sha256': '478e1723f71178949b5e78902ab274e4a2ebd91daeaa900f15e589d78a543b76'},
    'completion': {'path': '/tmp/libero-task7-optimization-path-decomposition-artifacts-20261109-v1/completion.json', 'sha256': '034f784c4e171bfc789ce3c1b217b59d79dc858577d7a19d28cbc0fdcaffa509'},
    'report': {'path': str(ROOT / 'docs/libero-task7-optimization-path-decomposition-results-2026-11-09.md'), 'sha256': 'cace190aa0a09c21c6ecc82c1119bf5604afe7b89a83630a9cfe662c36e84704'},
}
IDS = ['task', 'seed', 'arm', 'fold', 'mode']
MODE = 'harm_protected_affine'
ARMS = ['original_armijo', 'start500_mse_guarded_armijo']
SOURCE_MEMBERS = optimization_parent.SOURCE_MEMBERS
HEAD_MEMBERS = optimization_parent.HEAD_MEMBERS
NEW_HEAD_MEMBERS = optimization_parent.NEW_HEAD_MEMBERS
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_mse_nonregression_acceptance.py',
    'audit': ROOT / 'experiments/audit_libero_task7_mse_nonregression_acceptance.py',
    'guard': ROOT / 'experiments/libero_task7_mse_nonregression_acceptance_access.py',
    'runtime_tests': ROOT / 'tests/test_libero_task7_mse_nonregression_acceptance.py',
    'protocol': PLAN, 'metadata': Path(__file__).resolve(), 'metadata_tests': TESTS,
}
FUTURE_CONTRACT = ROOT / 'docs/libero-task7-mse-nonregression-acceptance-execution-contract-2026-11-09.md'
SCOPE = {
    'task_ids': [7], 'training_seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'original_Policy_updates': 500, 'source_auxiliary_updates': 500, 'future_stabilization_steps': 5,
    'head_coverage': 'all18_lambda1_heads_2sources_x3seeds_x3folds;not_selected_MSEup11',
    'source_role': 'meta_support_demos0..29',
    'fit_role': 'own20_whole_demos_all_tge1_including_neutral;same_rows_for_both_arms',
    'encoder_has_seen_all30_support_demos': True, 'auxiliary_probe_scoring': 0,
    'source_meta_query_validation_heldout_qualification_reserved_final_access': False,
    'new_original_Policy_CNN_environment_calls': 0,
    'new_loss_lambda_representation_or_data_weight_change': False,
    'new_auxiliary_parameter_updates': 'copies_only_500_plus_k;explicit_budget;not_Policy550',
    'SPT_Knowledge_Module_or_original_Policy_changes': False,
    'prior_joint_cues_G_failed_H_unrun_unchanged': True,
}
OPTIMIZATION = {
    **copy.deepcopy(optimization_parent.OPTIMIZATION),
    'arms': ARMS,
    'start': 'both_arms_from_each_own_saved_lambda1_auxiliary500_W_b_copy;not_previous_plus50',
    'same_loss': 'F=C6+T;T=MSE3/7+P;P=mean(relu(e-e0))/7;lambda1_unchanged',
    'gate': 'trial_MSE3<=this_head_own_auxiliary500_MSE3;fixed_for_all_rounds;no_epsilon',
    'gate_reference': 'auxiliary500_expert_MSE;distinct_from_per_frame_original_Policy_error_e0_in_loss',
    'gate_equivalence': 'other6_actions_fixed_so_MSE3_gate_equivalent_to_uniform7_action_MSE_gate',
    'acceptance': {
        ARMS[0]: 'first_trial_Ttrial<Tcurrent_and_Ttrial<=Tcurrent+c1*alpha*slope',
        ARMS[1]: 'first_trial_original_Armijo_and_trial_MSE3<=own_auxiliary500_MSE3',
    },
    'mse_monotonicity': 'fixed_start_envelope;may_rise_from_previous_accepted_state_if_below_start',
    'control': 'rerun_all18_original_Armijo_from_same500;check_historical_all_states_trials_endpoints',
    'fit_preprocessing': '18_rebuilds;reuse_identical_readonly_z_between_two_arms;18_source_heads_36_paths',
    'per_trial': 'one_linear_form_then_MSE3_P_T_F;both_gate_flags_and_reasons_logged_even_if_rejected',
    'path_order': 'seed_then_source_then_fold;original_armijo_then_start500_mse_guarded_armijo',
    'exhaustion': 'same_line_search_exhausted_stop;record_Armijo_and_MSE_vetoes;no_extra_direction_or_smaller_alpha',
    'new_head_output': '18_archives_each_start_baseline_final_candidate_final;no_intermediate_checkpoint',
}
DECISION = {
    **copy.deepcopy(optimization_parent.DECISION),
    'primary_comparison': 'paired_candidate_vs_original_final_MSE3_P_T_G_updates_trials_alpha;all18_heads_retained',
    'gate_validity': 'all_candidate_accepted_states_and_final_MSE3<=own500;invariant_not_evidence_of_improvement',
    'finite_fit_progress': 'audited_candidate_updates>0_and_Tfinal<T500_and_MSE3final<=MSE3500',
    'no_update': 'zero_updates_is_no_feasible_accepted_step_not_success;no_budget_transfer',
    'source_cue': 'separate_new_fit_only_cue_requires_all9_existing_joint_criteria_and_MSE_gate_and_updates>0',
    'baseline_parity': 'all18_historical_lambda1_paths_match;abs1e-10_scalars_exact_ID_count_branch_and_decisions',
    'no_promotion': 'no_automatic_loss_optimizer_Policy_or_formal_protocol_change;probe_requires_new_freeze',
    'failure_meaning': 'no_acceptable_trial_only_for_selected_direction_and_12point_grid;not_global_infeasibility',
    'next': 'inspect_all18_paired_results_and_stops_before_any_new_probe_or_training_change',
}
EXECUTION = {
    **copy.deepcopy(optimization_parent.EXECUTION),
    'binding': 'seven_roles_contract_science_parent736closure_before_archive_SHA_or_decode',
    'technical_stop': 'any_parent_baseline_parity_gate_role_counter_nonfinite_resource_branch_or_acceptance_mismatch;keep_partial_no_relaxation',
}
# The old JSON already contains these scalar records. No old CSV/NPZ is read for parity.
HISTORICAL_PROJECTIONS = {
    'states': parent.COLUMNS['states'] + ['branch_digest'],
    'trials': parent.COLUMNS['trials'],
    'endpoints': parent.COLUMNS['endpoints'] + ['branch_digest'],
    'heads': parent.COLUMNS['heads'],
}


def budget():
    heads, paths, nsum, rounds, trials = 18, 36, 100512, 50, 12
    states = paths * (rounds + 1)
    candidate_forms = paths * rounds * trials
    state_rows = nsum * (rounds + 1)
    trial_rows = nsum * rounds * trials
    tables = {'endpoints': 54, 'parameters': 54 * 257, 'states_max': states,
              'trials_max': candidate_forms, 'heads': paths, 'pairs': heads, 'scaler-parity': heads}
    return {
        'original_endpoints': 6, 'folds': 3, 'source_heads': heads, 'optimization_paths': paths,
        'parameters_per_head': 257,
        'updates_max_per_head_per_arm': rounds, 'trials_per_round_max': trials,
        'main_auxiliary_updates_max': paths * rounds, 'audit_replay_updates_max': paths * rounds,
        'parameter_coordinate_writes_max_per_stage': paths * rounds * 257,
        'scaler_reconstructions_per_stage': heads, 'fit_standardization_row_visits_per_stage': 50256,
        'fit_standardization_feature_coordinates_per_stage': 50256 * 256,
        'paired_representation_reuses_per_stage': heads, 'source_head_admissions_per_stage': heads,
        'optimization_path_admissions_per_stage': paths,
        'gradient_states_max_per_stage': states, 'gradient_vectors_max_per_stage': states * 3,
        'gradient_coordinates_max_per_stage': states * 3 * 257,
        'direction_coordinates_max_per_stage': states * 257,
        'audit_autograd_calls_max': states * 3, 'main_autograd_calls': 0,
        'trial_objective_forms_max_per_stage': candidate_forms,
        'trial_MSE_and_penalty_pair_evaluations_max_per_stage': candidate_forms,
        'gradient_state_fit_row_visits_max_per_stage': state_rows,
        'gradient_state_seven_action_coordinates_max_per_stage': state_rows * 7,
        'trial_component3_fit_row_visits_max_per_stage': trial_rows,
        'all_linear_forms_max_per_stage': states * 2 + candidate_forms,
        'all_linear_form_row_visits_max_per_stage': state_rows * 2 + trial_rows,
        'all_linear_form_feature_multiplications_max_per_stage': (state_rows * 2 + trial_rows) * 256,
        'original_NPZ_loads_per_stage': 25, 'original_member_decodes_per_stage': 161,
        'new_NPZ_writes_primary': heads, 'new_member_writes_primary': heads * 7,
        'new_NPZ_loads_audit': heads, 'new_member_decodes_audit': heads * 7,
        'total_NPZ_loads': 25 + 25 + heads, 'total_member_decodes': 161 * 2 + heads * 7,
        'original_archive_SHA_per_stage': 50, 'new_archive_SHA_primary': heads,
        'new_archive_SHA_audit': heads * 2, 'total_archive_SHA_checks': 100 + heads * 3,
        'historical_baseline_paths': heads,
        'historical_JSON_projected_records_per_stage': {'states': 918, 'trials': 7030, 'endpoints': 36, 'heads': 18},
        'input_research_CSV_passes': 0,
        'tables': tables, 'output_CSV_rows_max': sum(tables.values()),
        'seed_arm_summaries': 12, 'source_arm_summaries': 4,
        'seed_paired_summaries': 6, 'source_paired_summaries': 2,
        'original_Policy_CNN_environment_or_sealed_calls': 0,
    }


def trial_decision(current, trial, slope, index, trial_mse, start_mse, arm):
    """Strict scalar contract. The MSE envelope has no audit-tolerance slack."""
    p.require(arm in ARMS, 'unregistered optimization arm')
    optimization_parent.finite_nonnegative(trial_mse, start_mse)
    armijo = optimization_parent.armijo_accept(current, trial, slope, index)
    mse = trial_mse <= start_mse
    guarded = arm == ARMS[1]
    accepted = armijo and (mse or not guarded)
    reason = ('accepted' if accepted else 'armijo_and_mse_rejected' if guarded and not armijo and not mse
              else 'armijo_rejected' if not armijo else 'mse_envelope_rejected')
    return {'armijo_pass': armijo, 'mse_envelope_pass': mse, 'mse_gate_applied': guarded,
            'accepted': accepted, 'reason': reason}


def validate_round(current, slope, start_mse, arm, trials):
    """Require the first acceptable trial, or all 12 rejected grid points."""
    p.require(arm in ARMS and type(trials) is list and 1 <= len(trials) <= 12, 'invalid complete round')
    accepted_index = None
    for index, row in enumerate(trials):
        p.require(accepted_index is None and type(row['trial_index']) is int and row['trial_index'] == index,
                  'skipped trial or continued after first acceptance')
        p.require(type(row['alpha']) is float and row['alpha'] == optimization_parent.trial_alpha(index),
                  'trial alpha differs from frozen grid')
        expected = trial_decision(current, row['trainable_objective'], slope, index,
                                  row['component3_mse'], start_mse, arm)
        p.require(all(type(row[k]) is type(v) and row[k] == v for k, v in expected.items()),
                  'trial gate or rejection reason differs')
        if expected['accepted']:
            accepted_index = index
    p.require(accepted_index is not None or len(trials) == 12, 'incomplete rejected grid')
    return {'accepted_index': accepted_index, 'exhausted': accepted_index is None,
            'armijo_passes': sum(r['armijo_pass'] for r in trials),
            'mse_vetoes_of_armijo_passes': sum(r['armijo_pass'] and not r['mse_envelope_pass'] for r in trials)
            if arm == ARMS[1] else 0}


def endpoint_evidence(t0, tend, g0, gend, m0, mend, updates, arm, *, numeric_valid=True):
    optimization_parent.finite_nonnegative(m0, mend)
    p.require(arm in ARMS and type(updates) is int and 0 <= updates <= 50, 'invalid endpoint identity or budget')
    old = optimization_parent.endpoint_evidence(t0, tend, g0, gend, numeric_valid=numeric_valid)
    gate = mend <= m0
    p.require(arm != ARMS[1] or gate, 'candidate violated fixed500 MSE envelope')
    p.require(updates != 0 or (tend == t0 and gend == g0 and mend == m0), 'zero-update endpoint changed')
    progress = updates > 0 and tend < t0 and gate
    return {**old, 'start500_mse_envelope_met': gate,
            'finite_fit_progress': progress, 'protected_joint_fit_cue': progress and old['joint_cue'],
            'update_status': 'no_parameter_update' if updates == 0 else 'updated'}


def _identity(row):
    return tuple(row[k] for k in IDS)


def create():
    lineage = {name: p.load_pin(ref) for name, ref in PARENTS.items() if name != 'report'}
    p.require(p.pin(Path(PARENTS['report']['path'])) == PARENTS['report'], 'prior decomposition report changed')
    history_science = parent.validate()
    r, audit, done = (lineage[k] for k in ('result', 'audit', 'completion'))
    p.require(lineage['science'] == history_science
              and r['science_config'] == done['science_config'] == PARENTS['science']
              and r['binding'] == done['binding'] == PARENTS['binding']
              and audit['result'] == done['result'] == PARENTS['result']
              and done['audit'] == PARENTS['audit']
              and done['status'] == 'complete_independently_audited_saved_path_decomposition'
              and audit['status'] == 'passed_independent_saved_path_decomposition_audit',
              'completed independently audited decomposition lineage required')
    old = optimization_parent.validate()
    source = copy.deepcopy(old['inputs'])
    source['heads'] = [h for h in source['heads'] if h['mode'] == MODE]
    expected = {(7, seed, arm, fold, MODE) for seed in p.SEEDS for arm in p.ARMS for fold in range(3)}
    p.require(len(source['heads']) == 18 and {_identity(h) for h in source['heads']} == expected
              and all(h['training']['updates'] == 500 and h['training']['excess_error_penalty_lambda'] == 1.
                      for h in source['heads']), 'all18 original lambda1 auxiliary500 heads required')
    history = p.load_pin(parent.PARENTS['result'])
    references = []
    historical_counts = {}
    for table, fields in HISTORICAL_PROJECTIONS.items():
        selected = [h for h in history[table] if h['mode'] == MODE]
        p.require(all(set(fields).issubset(h) for h in selected), 'historical scalar projection missing')
        historical_counts[table] = len(selected)
    p.require(historical_counts == budget()['historical_JSON_projected_records_per_stage'],
              'historical lambda1 path count changed')
    for head in source['heads']:
        rows = [h for h in history['endpoints'] if _identity(h) == _identity(head)]
        p.require(len(rows) == 2 and {h['condition'] for h in rows} == {'hold_500', 'fullbatch_armijo'},
                  'historical paired endpoints missing')
        start = next(h for h in rows if h['condition'] == 'hold_500')
        final = next(h for h in rows if h['condition'] == 'fullbatch_armijo')
        summary = next(h for h in history['heads'] if _identity(h) == _identity(head))
        p.require(start['extra_updates'] == 0 and final['extra_updates'] == summary['extra_updates'] == 50
                  and summary['terminal_reason'] == 'update_cap', 'historical original Armijo path changed')
        metrics = ['component3_mse', 'penalty', 'trainable_objective', 'total_objective', 'total_l2', 'branch_digest']
        references.append({**{k: head[k] for k in IDS}, 'n': start['n'],
                           'start500': {k: start[k] for k in metrics},
                           'historical_original_final': {k: final[k] for k in metrics},
                           'historical_trials': summary['trials']})
    return copy.deepcopy({
        'kind': 'task7_lambda1_fixed_start_mse_acceptance_diagnostic', 'version': 1,
        'status': 'scientific_freeze_only_execution_binding_required', 'formal_result': False,
        'parents': PARENTS, 'scope': SCOPE, 'optimization': OPTIMIZATION, 'decision': DECISION,
        'execution': EXECUTION, 'budget': budget(), 'inputs': source, 'reference_points': references,
        'source_members': SOURCE_MEMBERS, 'head_members': HEAD_MEMBERS, 'new_head_members': NEW_HEAD_MEMBERS,
        'new_head_layout': {'weights': [3, 256], 'bias': [3], 'additional_updates': [3],
                            'order': ['saved500', ARMS[0] + '_final', ARMS[1] + '_final'],
                            'parameter_dtype': 'float64', 'additional_updates_dtype': 'int32'},
        'historical_baseline': {'source_JSON': parent.PARENTS['result'],
                                'projections': HISTORICAL_PROJECTIONS, 'mode': MODE,
                                'no_historical_CSV_or_optimized_NPZ_read': True},
        'code': {'metadata': p.pin(Path(__file__).resolve()), 'tests': p.pin(TESTS)},
        'plan': p.pin(PLAN), 'future_files': {k: str(v) for k, v in FUTURE_FILES.items()},
        'future_contract': str(FUTURE_CONTRACT), 'output': OUTPUT, 'artifacts': ARTIFACTS,
    })


def validate(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(json.dumps(cfg, sort_keys=True, allow_nan=False) == json.dumps(create(), sort_keys=True, allow_nan=False),
              'fixed-start MSE acceptance scientific freeze differs')
    return cfg


def require_binding(config=DEFAULT_CONFIG, binding=DEFAULT_BINDING):
    cfg = validate(config)
    b = json.loads(Path(binding).read_text())
    p.require(b['status'] == 'bound_before_payload' and b['before_research_payload'] is True
              and b['science_config'] == p.pin(config) and b['budget'] == cfg['budget']
              and b['CPU_environment'] == EXECUTION['environment']
              and b['future_output'] == OUTPUT and b['future_artifacts'] == ARTIFACTS,
              'new execution binding missing or changed')
    refs = b['dependencies']
    paths = [p.resolve(r['path']).resolve() for r in refs]
    p.require(len(paths) == len(set(paths))
              and set(FUTURE_FILES.values()).union({FUTURE_CONTRACT, DEFAULT_CONFIG}).issubset(paths),
              'future implementation roles not bound')
    p.require(all(ref in refs for ref in p.load_pin(PARENTS['binding'])['dependencies']),
              'parent736 dependency closure missing')
    for ref in refs:
        path = p.resolve(ref['path'])
        p.require(path.suffix in ('.py', '.md', '.json', '.toml') and p.pin(path) == ref,
                  'bound dependency changed or payload in closure')
    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    parser.add_argument('--require-binding', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        with DEFAULT_CONFIG.open('x') as stream:
            json.dump(create(), stream, ensure_ascii=False, indent=2)
            stream.write('\n')
        result = {'status': 'scientific_frozen_no_research_archive_or_new_optimization'}
    else:
        validate()
        result = {'status': 'metadata_valid_no_research_archive_or_new_optimization'}
        if args.require_binding:
            require_binding()
            result['status'] = 'bound_no_payload'
    print(json.dumps(result))
