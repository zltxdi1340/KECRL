"""Freeze saved scalar accounting: M/P/T references and residual gradient norms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments import libero_task7_original_m_bound_joint_t_direction_protocol as parent

p, ROOT, IDS, ARMS = parent.p, parent.ROOT, parent.IDS, parent.ARMS
STEM = 'libero-task7-joint-t-tradeoff-gradient'
CONFIG = ROOT / f'configs/{STEM}-readonly_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
ARTIFACTS = Path(f'/tmp/{STEM}-20261111-v1')
OUTPUT, AUDIT = ARTIFACTS / 'analysis.json', ARTIFACTS / 'independent-audit.json'
PLAN = ROOT / f'docs/{STEM}-plan-2026-11-11.md'
PARENTS = {
    'science': {'path': str(parent.CONFIG), 'sha256': 'd5849f6e6a7fa54bf767251464d3107396fa4c393121d44574643f1523f57b71'},
    'binding': {'path': str(parent.BINDING), 'sha256': 'd6372c0347be110c6892862596b0194f1c412690666f622ddf5e121e037afd29'},
    'result': {'path': parent.OUTPUT, 'sha256': '35be3fa0440ea272cf8b8a95b7aa5a6ffbe3a6b4515a226afde1fbebded32ae9'},
    'completion': {'path': str(Path(parent.ARTIFACTS) / 'completion.json'), 'sha256': 'ea45029c864275de2368abde9bf7539af77354116dcfd5d6db0824454817ad33'},
    'process': {'path': str(Path(parent.OUTPUT).with_name(Path(parent.OUTPUT).stem + '-process.json')), 'sha256': '5c432f70c000700eaf1356d695fab9c51b400805f8b8cb90882b0a72e82053bb'},
    'report': {'path': str(ROOT / 'docs/libero-task7-original-m-bound-joint-t-direction-results-2026-11-11.md'), 'sha256': 'a5154a83a2d0308512c6b1093d91186e8d33fcdc2c73aabbc725679b50446a74'},
}
INPUTS = {
    'P': {'path': str(Path(parent.ARTIFACTS) / 'primary-result.json'), 'sha256': '942078ca737ad5779b3104a29d1557e89226d4163e6bfaecbcebf83bf4935dde'},
    'U': {'path': str(Path(parent.ARTIFACTS) / 'independent-audit.json'), 'sha256': 'ab7772504153076722b9a66775b32106a46e2fb030ba66dd3346db4cd5cca9ef'},
}
HEAD = IDS + ['optimization_arm', 'extra_updates', 'trials', 'terminal_reason',
    'initial_component3_mse', 'final_component3_mse', 'initial_penalty', 'final_penalty',
    'initial_trainable_objective', 'final_trainable_objective', 'initial_gradient_l2', 'final_gradient_l2',
    'accepted_MSE_step_increases', 'trainable_ratio', 'gradient_ratio', 'objective_target_met', 'gradient_target_met', 'joint_cue']
PAIR = IDS + ['cohort', 'candidate_progress_from500', 'step_M_invariant_met', 'guarded_T_control_pair_cue',
    'margin_anchor_cue', 'zero_anchor_cue', 'bisector_anchor_cue', 'paired_joint_cue'] + [
    'candidate_minus_' + ref + '_' + metric for ref in ('control', 'margin', 'zero', 'bisector') for metric in ('MSE', 'T')] + [
    ref + '_anchor_' + f for ref in ('margin', 'zero', 'bisector')
    for f in ('component3_mse', 'penalty', 'trainable_objective', 'total_l2', 'extra_updates')]
STATE = IDS + ['optimization_arm', 'extra_updates', 'n', 'component3_mse', 'penalty', 'trainable_objective',
               'total_l2', 'base_l2', 'penalty_l2', 'branch_digest']
PROJECTIONS = dict(heads=HEAD, pairs=PAIR, states=STATE)
REFERENCES = ['own500', 'control', 'margin', 'zero', 'bisector']
WINDOWS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
FILES = {name: ROOT / path for name, path in {
    'protocol': 'experiments/libero_task7_joint_t_tradeoff_gradient_protocol.py',
    'guard': 'experiments/libero_task7_joint_t_tradeoff_gradient_access.py',
    'analysis': 'experiments/analyze_libero_task7_joint_t_tradeoff_gradient.py',
    'audit': 'experiments/audit_libero_task7_joint_t_tradeoff_gradient.py',
    'tests': 'tests/test_libero_task7_joint_t_tradeoff_gradient.py',
    'plan': f'docs/{STEM}-plan-2026-11-11.md',
}.items()}


def key(row):
    return tuple(row[f] for f in IDS)


def budget():
    counts = dict(heads=36, pairs=18, states=1836)
    values = sum(counts[k] * len(v) for k, v in PROJECTIONS.items())
    return dict(stages=2, source_JSON_passes_per_stage=2, source_SHA_per_stage=4,
                total_source_JSON_passes=4, total_source_SHA=8, rows_per_JSON=counts,
                projected_rows_per_stage=2 * sum(counts.values()), projected_scalar_fields_per_stage=2 * values,
                total_projected_scalar_fields=4 * values, saved_paths_per_stage=72, saved_transitions_per_stage=3600,
                windows_per_stage=360, endpoint_gradient_summaries_per_stage=72,
                primary_result_audit_reads=1, primary_result_audit_SHA=2, technical_retries=0,
                output_rows=dict(comparisons=180, gradient_endpoints=72, path_windows=360,
                    comparison_source_seed=60, comparison_source=20, gradient_source_seed=24, gradient_source=8,
                    window_source_seed=120, window_source=40, cross_engine=18),
                new_forward_gradient_autograd_optimizer_trial_environment_calls=0,
                extra_NPZ_CSV_HDF5_checkpoint_vectors_parameters_inputs_private_or_sealed_reads_including_SHA=0)


def create():
    previous = parent.require_binding()
    meta = {k: p.load(ref) for k, ref in PARENTS.items() if k != 'report'}
    p.require(p.pin(PARENTS['report']['path']) == PARENTS['report'], 'joint-T report changed')
    r, done, proc = meta['result'], meta['completion'], meta['process']
    p.require(meta['binding'] == previous and len(previous['dependencies']) == 917
              and r['status'] == done['status'] == proc['status'] == 'complete_independently_audited_interior_joint_T_direction'
              and r['primary'] == INPUTS['P'] and r['audit'] == done['audit'] == INPUTS['U'] and done['result'] == PARENTS['result']
              and r['science_config'] == done['science_config'] == proc['science_config'] == PARENTS['science']
              and r['binding'] == done['binding'] == proc['binding'] == PARENTS['binding']
              and r['primary_updates'] == r['independent_U_updates'] == 1800 and r['local_S_updates'] == 0
              and r['local_numeric_states'] == 1836 and r['local_trial_targets'] == 16186
              and r['local_stop_disagreements'] == r['local_accept_disagreements'] == 0
              and r['original_Policy_CNN_environment_or_sealed_calls'] == 0 and r['formal_result'] is False
              and len(proc['processes']) == 2 and all(row['exit_code'] == 0 for row in proc['processes']), 'completed joint-T provenance required')
    return dict(kind='joint_T_saved_M_P_T_and_residual_gradient_accounting', version=1, formal_result=False,
                parents=PARENTS, inputs=INPUTS, identities=meta['science']['cohorts'], optimization_arms=ARMS,
                fit_n_by_fold={str(row['fold']): len(row['fit_rows']) for row in meta['science']['inputs']['folds']},
                projections=PROJECTIONS, references=REFERENCES, windows=[list(w) for w in WINDOWS],
                scope=dict(original_Policy_updates=500, future_stabilization_steps=5, loss_lambda=1,
                    source='own20_fit_support0..29_tge1_encoder_seen_all30;not_unseen_generalization',
                    no_new_training_or_loss_direction_checkpoint_choice=True, SPT_Knowledge_Module_changes=False,
                    all18_both_arms_retained=True, formal_evaluation_data_stays_sealed=True),
                design=dict(point='M=component3_expert_MSE;P=positive_excess_vs_originalPolicy_error/7;T=M/7+P',
                    comparison='candidate_final_minus_own500_and_all4_preserved_references;raw_sign_categories',
                    gradients='saved_G_Gb_Gp_norms_only;no_new_gradient_vector_or_geometric_certificate',
                    original_targets='saved_t50/t0<=.99_and_saved_g50/g0<=.1;None_for_zero_denominator;no_new_threshold',
                    gradient_focus='all18_retained;candidate_original_gradient_target_false_is_known_outcome_stratum',
                    late='fixed40_to50_window;fractions_of_0_to50_total_T_saving_only_when_defined',
                    norm_ratio='saved_normG/(saved_normGb+saved_normGp);descriptive_not_angle_or_causal_cancellation_proof',
                    path='all51_saved_states;raw_M_nonincrease_T_strict_lower;gradient_not_assumed_monotone',
                    branch='raw_digest_changes_only;cannot_identify_clip_vs_hinge_or_establish_causation',
                    statistics='equal_fold_within_source_seed;equal_head_source;means_medians_counts_no_pvalues',
                    numerical_absolute_tolerance=1e-10, relative_tolerance=0, raw_classification_and_flags_tolerance=0,
                    cannot_claim='residual_gradient_does_not_prove_more_updates_help_or_convergence;T_lower_not_M_lower;no_formal_promotion'),
                budget=budget(), execution=dict(python='3.10.22', device='CPU', seconds_per_stage=120, overall_seconds=300,
                    RSS_bytes=536870912, input_JSON_bytes_max=201326592, output_JSON_bytes_max=8388608,
                    environment={'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                    stop='metadata_fields_roles_identity_coverage_nonfinite_conservation_or_audit_mismatch_fail_closed;no_retry_no_overwrite'),
                roles={k: p.pin(path) for k, path in FILES.items()}, output=str(OUTPUT), audit=str(AUDIT))


def preflight():
    cfg = json.loads(CONFIG.read_text())
    p.require(cfg == create(), 'frozen joint-T accounting changed')
    bound = json.loads(BINDING.read_text())
    p.require(bound['status'] == 'bound_before_payload' and bound['before_payload'] is True
              and bound['science'] == p.pin(CONFIG) and bound['budget'] == cfg['budget']
              and bound['CPU_environment'] == cfg['execution']['environment'], 'new accounting execution binding required')
    refs = bound['dependencies']
    paths = [r['path'] for r in refs]
    previous = p.load(PARENTS['binding'])
    p.require(len(paths) == len(set(paths)) and all(p.pin(path) in refs for path in [*FILES.values(), CONFIG])
              and all(r in refs for r in previous['dependencies']) and all(r in refs for r in PARENTS.values()), 'parent917/new role closure missing')
    p.require(not set(paths) & {r['path'] for r in INPUTS.values()}, 'source numerical JSON cannot be hashed before binding')
    for ref in refs:
        p.require(Path(ref['path']).suffix in {'.py', '.md', '.json', '.toml'} and p.pin(ref['path']) == ref, 'bound accounting dependency changed')
    return cfg, bound


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        p.write(CONFIG, create())
    else:
        preflight()
    print(json.dumps({'status': 'joint_T_accounting_metadata_no_numeric_payload'}))
