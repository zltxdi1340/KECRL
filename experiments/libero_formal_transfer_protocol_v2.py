"""Version 2: independent G/H training cohorts; metadata and count decisions only.

This module has no training, simulator, checkpoint, HDF5 or initial-bank loader.
The registered design requires a separately bound execution implementation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
GATE_SEEDS = list(range(20261106, 20261126))
FINAL_SEEDS = list(range(20261126, 20261146))
ARMS = ['A_original', 'B_fomaml', 'C_pooled_bc']
SOURCE = [0, 1, 3, 4, 7]
VALIDATION = [2, 6]
HELDOUT = [5, 8, 9]
CURVE = [0, 50, 150, 300, 500]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def budgets():
    n = len(GATE_SEEDS)
    final_n = len(FINAL_SEEDS)
    nonreg = len(SOURCE) * 2 * n
    validation = len(VALIDATION) * len(ARMS) * n
    final = len(HELDOUT) * len(ARMS) * final_n
    gate_episodes = (nonreg + validation) * 20
    final_episodes = final * 30
    episodes = gate_episodes + final_episodes
    return {
        'gate_source_generator_adam_updates': 2*n*50,
        'final_source_generator_adam_updates': 2*final_n*50,
        'source_generator_adam_updates': 2*(n+final_n)*50,
        'gate_source_disposable_sgd_updates': n*50*len(SOURCE),
        'final_source_disposable_sgd_updates': final_n*50*len(SOURCE),
        'source_disposable_sgd_updates': (n+final_n)*50*len(SOURCE),
        'source_data_gradient_calls': 2 * (n+final_n) * 50 * len(SOURCE) * 2,
        'source_generator_vjp_calls': 2 * (n+final_n) * 50 * len(SOURCE),
        'gate_adaptation_models': nonreg + validation,
        'final_adaptation_models': final,
        'maximum_adaptation_models': nonreg + validation + final,
        'gate_adaptation_adam_updates': (nonreg + validation) * 500,
        'final_adaptation_adam_updates': final * 500,
        'maximum_adaptation_adam_updates': (nonreg + validation + final) * 500,
        'maximum_total_optimizer_updates': (nonreg + validation + final) * 500 + 2*(n+final_n)*50 + (n+final_n)*50*len(SOURCE),
        'maximum_policy_curve_checkpoint_files': (nonreg + validation + final) * len(CURVE),
        'generator_endpoint_files': (n+final_n) * len(ARMS),
        'maximum_offline_dataset_sweeps': nonreg*len(CURVE) + (validation+final)*len(CURVE)*2,
        'maximum_offline_evaluation_forward_batches': 120000,
        'gate_nonregression_episodes': nonreg * 20,
        'gate_validation_episodes': validation * 20,
        'gate_episodes': gate_episodes,
        'final_evaluation_episodes': final_episodes,
        'maximum_episodes': episodes,
        'maximum_policy_actions': episodes * 300,
        'fixed_zero_actions_if_all_stages_complete': episodes * 5,
        'maximum_native_steps': episodes * 305,
        'maximum_adapters': nonreg + validation + final,
        'maximum_numeric_initial_rows': len(SOURCE)*20 + len(VALIDATION)*20 + len(HELDOUT)*30,
        'maximum_reset_references': len(SOURCE)*20 + len(VALIDATION)*20 + len(HELDOUT)*30,
        'maximum_exact_reset_comparisons': episodes - (len(SOURCE)*20 + len(VALIDATION)*20 + len(HELDOUT)*30),
        'qualification_episodes': 0,
        'source_or_validation_reserved_rows': 0,
    }


def scientific_design():
    return {
        'experiment_type': 'fixed_budget_same_family_spt_spi_transfer',
        'task_roles': {'source_spi': SOURCE, 'spt_validation_spi': VALIDATION, 'heldout_query_spi': HELDOUT},
        'training_seed_cohorts': {'gate': GATE_SEEDS, 'final': FINAL_SEEDS}, 'arms': ARMS,
        'cohort_separation': {'no_shared_source_generator_weights': True,
                              'no_G_seed_or_accepted_subset_in_H': True,
                              'H_source_training_only_after_G_passes': True,
                              'gate_accepts_method_configuration_not_H_weight_instances': True},
        'primary_hypothesis': {
            'metric': '500_update_closed_loop_success_rate',
            'tasks': HELDOUT, 'paired_independent_unit': 'training_seed',
            'within_seed_task_aggregation': 'equal_weight_macro_of_three_fixed_heldout_tasks',
            'comparisons': ['B_minus_C', 'B_minus_A'],
            'minimum_observed_mean_gain': 0.05,
            'inferential_null': 'probability_of_strict_positive_seed_macro_difference_at_most_one_half',
            'not_a_population_mean_or_learning_speed_test': True,
        },
        'algorithm': {
            'generator_source_tasks': SOURCE, 'source_outer_updates_per_seed_per_B_C': 50,
            'source_spis_per_update': 5, 'source_inner_steps_B_only': 1,
            'source_inner_optimizer': 'SGD', 'source_inner_lr': 0.001,
            'generator_optimizer': 'Adam', 'generator_lr': 0.0003,
            'adaptation_updates_per_policy': 500, 'adaptation_lr': 0.0003,
            'adam_betas': [0.9, 0.999], 'adam_eps': 1e-8, 'weight_decay': 0.0,
            'clip_norm': 1.0, 'batch_size': 8, 'sequence_length': 10, 'stride': 10,
            'offline_eval_batch_size': 32, 'num_workers': 0,
            'rng_streams': {'generator_initialization': 'seed',
                            'source_batch_sampler': 'seed_plus_1000003',
                            'target_batch_sampler': 'seed_plus_task_id_times10000_plus777',
                            'online_every_adapter': 'environment_seed20261022',
                            'source_task_order': SOURCE, 'source_batch_order': 'support_then_query_shared_B_C',
                            'pythonhashseed': '0'},
            'memory_mode': 'cnn_only', 'action_dimension': 7,
            'loss': 'uniform_seven_dimensional_valid_step_mse',
            'alignment': 'causal_pre_action_t_ge_1', 'rgb_normalization': 'uint8_div_255',
            'curve_updates': CURVE, 'selected_update': 500,
            'source_update_objectives': {
                'A_original': 'no_source_update',
                'B_fomaml': 'query_gradient_after_one_support_sgd_step_via_original_generator_jacobian',
                'C_pooled_bc': 'mean_of_unadapted_support_and_query_gradients_via_same_generator_jacobian'},
            'paired_initialization_and_batches': True, 'no_exact_total_flop_matching_claim': True,
            'augmentation_or_new_loss': False, 'historical_checkpoints_reused': False,
        },
        'data': {
            'source_update': {'tasks': SOURCE, 'meta_support': list(range(30)), 'meta_query': list(range(30,50)),
                              'query_is_training_feedback': True},
            'nonregression_adaptation': {'tasks': SOURCE, 'arms': ARMS[:2], 'meta_support': list(range(30))},
            'validation_adaptation': {'tasks': VALIDATION, 'adaptation_support': list(range(30))},
            'validation_secondary': {'tasks': VALIDATION, 'selection_query': list(range(30,50)),
                                     'gate': 'all60_support_fits_per_task_complete', 'used_in_gradients': False},
            'heldout_adaptation': {'tasks': HELDOUT, 'adaptation_support': list(range(30)),
                                   'gate': 'validation_and_nonregression_pass_and_execution_bound'},
            'heldout_secondary': {'tasks': HELDOUT, 'sealed_query': list(range(30,50)),
                                  'gate': 'all180_fits_and_all5400_final_episodes_complete',
                                  'used_in_gradients_or_selection': False},
            'demo_to_official_state_pairing': 'forbidden', 'read_or_reconstruct_t0': False,
        },
        'online': {
            'source_spi': {'tasks': SOURCE, 'arms': ARMS[:2], 'role': 'candidate_nonregression', 'indices': list(range(20))},
            'spt_validation_spi': {'tasks': VALIDATION, 'arms': ARMS, 'role': 'candidate_acceptance', 'indices': list(range(10,30))},
            'heldout_query_spi': {'tasks': HELDOUT, 'arms': ARMS, 'role': 'final_evaluation', 'indices': list(range(20,50))},
            'checkpoint_selection_states_used': False, 'closed_loop_only_at_update': 500,
            'environment_seed': 20261022, 'stabilization_steps': 5,
            'policy_action_limit': 300, 'native_horizon': 305, 'image_size': 128,
            'adapter_lifecycle': 'fresh_per_task_seed_arm; ascending_local_indices_of_only_authorized_rows',
            'task_order': 'source_nonregression_then_validation_then_conditional_heldout',
            'model_order': 'task_then_training_seed_then_arm',
            'global_and_local_initial_indices_recorded': True,
            'read_other_rows_to_recreate_history': False,
            'reset_gate': 'bit_exact_post_stabilization_digest_per_task_global_state_across_all_models',
            'private_live_simulator_state_recorded': False,
        },
        'validation_gate': {
            'B_macro_success_at_least': 0.20,
            'B_minus_A_and_C_mean_at_least': 0.05,
            'strict_positive_macro_seeds_each_comparison_at_least': 16,
            'source_B_minus_A_each_task_seed_mean_at_least': -0.05,
            'source_B_minus_A_every_individual_seed_at_least': -0.10,
            'all_integrity_and_public_interface_checks_pass': True,
            'validation_query_mse_selects_nothing': True,
            'failure_action': 'stop_no_heldout_access_no_additional_training',
            'pass_action': 'freeze_method_admission_then_retrain_H_sources_with_independent_seed_cohort',
            'active_spt_mutation': False,
        },
        'statistics': {
            'test': 'exact_binomial_strict_seed_win_probability', 'null_probability': 0.5,
            'alternative': 'greater', 'ties': 'count_as_nonwins_keep_n20',
            'multiplicity': 'Holm_two_comparisons', 'familywise_alpha': 0.05,
            'effect_sizes': 'exact_count_fractions_then_descriptive_mean_median_sd_and_all_seed_values',
            'win_probability_intervals': 'one_sided_Clopper_Pearson_lower97.5_each_Bonferroni_family95',
            'per_heldout_task_mean_delta_guard_A_and_C': -0.05,
            'final_uses_no_efficacy_stopping_or_model_selection': True,
            'final_seed_cohort_independent_of_validation_screening': True,
            'source_nonregression_is_G_only_not_a_claim_for_each_H_generator': True,
            'three_fixed_tasks_and_states_not_iid_population_draws': True,
            'no_episode_binomial_pvalue_or_cross_environment_pooling': True,
            'secondary': ['support_query_MSE_curves', 'translation_rotation_gripper_MSE',
                          'gripper_sign_accuracy', 'runtime_actual_update_and_interaction_costs',
                          'per_task_seed_success_four_cells'],
            'secondary_cannot_rescue_failed_primary': True,
        },
        'module_qualification': {
            'enabled_in_this_study': False, 'episodes': 0, 'numeric_state_rows': 0,
            'module_registration': False, 'later_minimum_valid_episodes': 20,
            'later_empirical_success_threshold': 0.80,
            'later_scope': 'explicit_fixed_qualification_bank_only_no_population_confidence_claim',
            'later_required': ['known_start_conditions', 'known_hold_conditions', 'detectable_goal',
                               'contract_and_resource_checks', 'applicability_scope', 'execution_budget'],
            'unknown_means_pending_hard_violation_means_failed': True,
            'later_enable_requires_separate_frozen_qualification_execution_contract': True,
        },
        'runtime': {'physical_gpu': 1, 'cuda_visible_devices': '1', 'egl_device_id': '1',
                    'threads': 1, 'cublas_workspace_config': ':4096:8', 'deterministic_algorithms': True,
                    'tf32': False, 'gpu_idle_memory_mib_max': 512, 'gpu_idle_utilization_percent_max': 5,
                    'gpu_compute_processes_before_launch': 0, 'max_host_tensor_cache_bytes': 8*1024**3,
                    'maximum_active_wall_hours_per_gate_or_final_stage': 16,
                    'python': '3.10.22', 'torch': '2.4.1+cu124', 'numpy': '1.22.4',
                    'h5py': '3.11.0', 'robosuite': '1.4.0', 'mujoco': '3.2.3'},
        'budget_totals': budgets(),
        'stop_rules': {
            'on_hash_role_scope_reset_nonfinite_budget_resource_error': 'technical_abort_preserve_partial_no_primary_result',
            'invalid_or_unknown_episode': 'technical_abort_not_silent_failure_or_denominator_exclusion',
            'validation_failure': 'valid_screening_failure_primary_heldout_not_performed',
            'failed_primary': 'hypothesis_not_supported_no_retraining_or_endpoint_seed_budget_search',
            'all_final_arms_run_regardless_of_low_interim_success': True,
            'no_result_based_retry_or_sample_replacement': True,
        },
        'freeze_boundary': {'method_design_frozen': True, 'execution_binding_complete': False,
                            'execution_authorized_by_this_manifest': False,
                            'sealed_payload_access_allowed_now': False,
                            'new_training_forward_environment_initial_state_calls_this_freeze': 0,
                            'knowledge_update': False, 'active_spt_update': False, 'module_registration': False},
    }


def validate_config(config):
    require(config.get('formal_result') is False and config.get('status') == 'method_design_frozen_not_executed',
            'expected an unexecuted frozen design')
    require(json.dumps(config.get('design'), sort_keys=True, allow_nan=False) ==
            json.dumps(scientific_design(), sort_keys=True, allow_nan=False), 'frozen scientific design differs')
    require(set(config.get('references', {})) >= {'transfer', 'family', 'learning', 'theory', 'diagnostic_summary'},
            'required provenance references missing')


def preflight(config_path, binding_path):
    config = json.loads(Path(config_path).read_text())
    validate_config(config)
    binding = json.loads(Path(binding_path).read_text())
    require(binding.get('config_sha256') == sha(config_path) and binding.get('formal_result') is False,
            'freeze binding/config mismatch')
    require(binding.get('execution_binding_complete') is False, 'freeze metadata is not a runtime binding')
    require(set(binding.get('files', {})) >= {'protocol_validator_and_statistics', 'protocol', 'tests'}, 'freeze code pins missing')
    for name, pin in {**config['references'], **binding['files']}.items():
        path = Path(pin['path'])
        require(sha(path if path.is_absolute() else ROOT/path) == pin['sha256'], f'code/input pin changed: {name}')
    validator = Path(binding['files']['protocol_validator_and_statistics']['path'])
    require((validator if validator.is_absolute() else ROOT/validator).resolve() == Path(__file__).resolve(),
            'validator pin points elsewhere')
    transfer = json.loads((ROOT/config['references']['transfer']['path']).read_text())
    require(transfer['task_roles'] == scientific_design()['task_roles'], 'transfer task roles changed')
    for role, key in [('source_spi','source_spi'), ('spt_validation_spi','spt_validation_spi'), ('heldout_query_spi','heldout_query_spi')]:
        plan = scientific_design()['online'][key]
        require(transfer['evaluation_initial_state_indices_by_task_role'][role][plan['role']] == plan['indices'],
                'official-state namespace changed')
    return config, binding


def indexed_counts(rows, tasks, arms, trials, *, seeds):
    expected = {(task, seed, arm) for task in tasks for seed in seeds for arm in arms}
    result = {}
    for row in rows:
        require(type(row.get('task')) is int and type(row.get('seed')) is int, 'invalid task/seed type')
        key = (row['task'], row['seed'], row['arm'])
        require(key in expected and key not in result, 'duplicate or out-of-scope count row')
        require(type(row.get('episodes')) is int and row['episodes'] == trials and
                type(row.get('successes')) is int and 0 <= row['successes'] <= trials,
                'missing, invalid or excluded trial count')
        result[key] = Fraction(row['successes'], trials)
    require(set(result) == expected, 'incomplete count table')
    return result


def fraction_record(value):
    return {'numerator': value.numerator, 'denominator': value.denominator, 'value': float(value)}


def macro_deltas(index, tasks, reference, *, seeds):
    return [sum((index[t,s,ARMS[1]] - index[t,s,reference] for t in tasks), Fraction()) / len(tasks) for s in seeds]


def validation_decision(source_rows, validation_rows, *, integrity_passed):
    require(integrity_passed is True, 'integrity failure cannot be scored as an efficacy rejection')
    source = indexed_counts(source_rows, SOURCE, ARMS[:2], 20, seeds=GATE_SEEDS)
    validation = indexed_counts(validation_rows, VALIDATION, ARMS, 20, seeds=GATE_SEEDS)
    rules = {'B_validation_macro_at_least_20_percent':
             sum((validation[t,s,ARMS[1]] for t in VALIDATION for s in GATE_SEEDS), Fraction()) / (len(VALIDATION)*len(GATE_SEEDS)) >= Fraction(1,5)}
    for reference in (ARMS[0], ARMS[2]):
        deltas = macro_deltas(validation, VALIDATION, reference, seeds=GATE_SEEDS)
        rules['validation_mean_gain_vs_'+reference] = sum(deltas, Fraction())/len(GATE_SEEDS) >= Fraction(1,20)
        rules['validation_at_least16_seed_wins_vs_'+reference] = sum(d > 0 for d in deltas) >= 16
    for task in SOURCE:
        deltas = [source[task,s,ARMS[1]] - source[task,s,ARMS[0]] for s in GATE_SEEDS]
        rules[f'source_task{task}_mean_nonregression'] = sum(deltas, Fraction())/len(GATE_SEEDS) >= -Fraction(1,20)
        rules[f'source_task{task}_every_seed_nonregression'] = all(d >= -Fraction(1,10) for d in deltas)
    passed = all(rules.values())
    return {'status': 'valid_validation_screening_decision', 'passed': passed, 'rules': rules,
            'next_stage': 'fresh_H_source_training_after_runtime_binding' if passed else 'stop_keep_heldout_sealed',
            'active_spt_updated': False, 'module_registered': False}


def binomial_win_tail(wins, n):
    require(type(wins) is int and type(n) is int and 0 <= wins <= n and n > 0, 'invalid binomial counts')
    return Fraction(sum(math.comb(n, k) for k in range(wins, n+1)), 2**n)


def holm_adjust(pvalues):
    require(bool(pvalues), 'empty p-value family')
    require(all(isinstance(p, Fraction) and 0 <= p <= 1 for p in pvalues.values()), 'invalid exact p-value')
    ordered = sorted(pvalues, key=lambda key: (pvalues[key], key))
    result, previous = {}, Fraction()
    for i, key in enumerate(ordered):
        previous = min(Fraction(1), max(previous, (len(ordered)-i)*pvalues[key]))
        result[key] = previous
    return result


def binomial_win_interval(wins, n, alpha=0.025):
    """Exact one-sided lower confidence interval for strict seed-win probability."""
    require(type(wins) is int and type(n) is int and 0 <= wins <= n and n > 0 and 0 < alpha < 1,
            'invalid confidence interval inputs')
    def quantile_tail(k, target):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            p = (lo+hi)/2
            tail = sum(math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(k,n+1))
            if tail < target:
                lo = p
            else:
                hi = p
        return (lo+hi)/2
    return [0.0 if wins == 0 else quantile_tail(wins, alpha), 1.0]


def final_decision(rows, *, validation_passed, integrity_passed):
    require(validation_passed is True and integrity_passed is True, 'final data cannot bypass validation/integrity gates')
    index = indexed_counts(rows, HELDOUT, ARMS, 30, seeds=FINAL_SEEDS)
    reports, pvalues = {}, {}
    for reference in (ARMS[2], ARMS[0]):
        name = 'B_minus_'+reference[0]
        deltas = macro_deltas(index, HELDOUT, reference, seeds=FINAL_SEEDS)
        wins = sum(d > 0 for d in deltas)
        pvalues[name] = binomial_win_tail(wins, len(FINAL_SEEDS))
        task_means = {str(t): sum((index[t,s,ARMS[1]]-index[t,s,reference] for s in FINAL_SEEDS), Fraction())/len(FINAL_SEEDS) for t in HELDOUT}
        reports[name] = {'seed_macro_deltas': [fraction_record(d) for d in deltas], 'strict_wins': wins,
                         'ties': sum(d == 0 for d in deltas), 'n': len(FINAL_SEEDS),
                         'observed_mean_delta': fraction_record(sum(deltas,Fraction())/len(FINAL_SEEDS)),
                         'observed_median_delta': fraction_record(statistics.median(deltas)),
                         'sample_sd': statistics.stdev(float(d) for d in deltas),
                         'pvalue': fraction_record(pvalues[name]),
                         'strict_win_probability_lower_interval_97_5_percent': binomial_win_interval(wins,len(FINAL_SEEDS)),
                         'per_task_mean_deltas': {t: fraction_record(d) for t,d in task_means.items()},
                         'observed_practical_gain_pass': sum(deltas,Fraction())/len(FINAL_SEEDS) >= Fraction(1,20),
                         'per_task_guard_pass': all(d >= -Fraction(1,20) for d in task_means.values())}
    adjusted = holm_adjust(pvalues)
    for name, report in reports.items():
        report['holm_adjusted_pvalue'] = fraction_record(adjusted[name])
        report['statistical_pass'] = adjusted[name] <= Fraction(1,20)
    supported = all(r['statistical_pass'] and r['observed_practical_gain_pass'] and r['per_task_guard_pass']
                    for r in reports.values())
    return {'status': 'complete_frozen_primary_decision', 'hypothesis_supported': supported,
            'comparisons': reports, 'independent_units': '20_training_seeds_conditional_on_fixed_tasks_and_states',
            'inferential_estimand': 'strict_seed_win_probability_not_population_mean_success_gain',
            'checkpoint_or_budget_or_active_version_selected_from_final': False,
            'module_registered': False, 'knowledge_updated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.output is not None:
        require(not args.output.exists(), 'preflight output already exists')
    config, binding = preflight(args.config,args.binding)
    record = {'status': 'method_design_freeze_preflight_passed', 'formal_result': False,
              'verified_at_utc': datetime.now(timezone.utc).isoformat(),
              'config_sha256': sha(args.config), 'binding_sha256': sha(args.binding),
              'budget_totals': config['design']['budget_totals'],
              'execution_binding_complete': False, 'sealed_access_authorized': False,
              'new_training_model_forward_environment_dataset_or_initial_bank_calls': 0}
    if args.output is not None:
        with args.output.open('x') as handle:
            json.dump(record,handle,ensure_ascii=False,indent=2)
            handle.write('\n')
    print(json.dumps(record,ensure_ascii=False))


if __name__ == '__main__':
    main()
