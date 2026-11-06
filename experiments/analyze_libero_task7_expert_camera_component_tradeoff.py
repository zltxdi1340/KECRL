"""Read-only localization of cached task7 translation/gripper error changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-expert-camera-component-tradeoff-readonly-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task7-expert-camera-component-tradeoff-20261103-v1.json')
DEFAULT_MEMBERSHIPS = Path('/tmp/libero-task7-expert-camera-component-tradeoff-memberships-20261103-v1.npz')
SEEDS = (20261022, 20261023, 20261024)
CONDITIONS = ('identity_uint8', 'identity_float', 'eye_in_hand_photo_matched', 'eye_in_hand_spatial_matched',
    'agentview_photo_matched', 'agentview_spatial_matched', 'eye_in_hand_spatial_full', 'agentview_spatial_full')
LOCAL = CONDITIONS[2:6]
STRATA = ('all', 'first_scored_t1', 'expert_open', 'expert_close', 'expert_neutral', 'around_close_transition', 'around_open_transition')
COMPONENTS = {'translation': (0, 3), 'rotation': (3, 6), 'gripper': (6, 7), 'all7': (0, 7)}
WINDOWS = ('neither', 'open_only', 'close_only', 'open_close_overlap')
FATES = ('both_correct', 'correct_to_wrong', 'wrong_to_correct', 'both_wrong', 'expert_neutral')
PINS = {'source_result': '787bf42f4b66d472d4a73e09fb9f9d09713ad87f671474c762678a7de26c6b13',
    'source_audit': 'dc2f01442e5bbd9a2f1df8804b8feaa21376bb236617cb20d740cd1b915bc7e4',
    'source_design': '67654a94e8e41497fa24b0929849f2d9535d9b5ae518bd56c39a863182acfdb4',
    'source_binding': '16ebf9946b39abb83f24a87d397b9f1db90acf00459317a27e7e22b652d7aa42',
    'source_task7_archive': 'e42844cf1a43154fe7842cacf6ac1efcaea8d95b8302443def7fc7e7e7697d10',
    'source_runner': '5a3ed5c2bc611120521a71f9ae821e46b09e4a607c8d1aae6163e6e784793067',
    'source_operator': '681a3390bc894e753e945fa0bdab6b79b5dd4ee2a60cc2fccb9e18fb10b7d834'}
SCOPE = {'task_ids': [7], 'training_seeds': list(SEEDS), 'checkpoint_updates': 500,
    'default_closed_loop_stabilization_steps': 5, 'source_role': 'meta_support', 'demo_indices': list(range(30)),
    'unique_cached_expert_rows': 4188, 'cached_prediction_rows': 100512, 'cached_archive_files': 1,
    'new_policy_forwards': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_resets_or_steps': 0,
    'new_HDF5_RGB_initial_state_or_checkpoint_reads': 0, 'source_query_accessed': False,
    'qualification_accessed': False, 'reserved_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False,
    'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
ANALYSIS = {'focus_condition': 'eye_in_hand_spatial_matched', 'prior_identified_focus_seed': 20261024,
    'all_reference_conditions': list(CONDITIONS), 'localization_conditions': list(LOCAL), 'existing_strata': list(STRATA),
    'window_partition': list(WINDOWS), 'window_radius_unchanged': 5, 'expert_command_groups': [-1, 0, 1],
    'direction_code': '3*(sign(Delta_translation_MSE)+1)+sign(Delta_gripper_MSE)+1; nine exact-sign cells',
    'same_row_tradeoff_code': 2, 'gripper_sign_fates': list(FATES),
    'global_contribution': 'sum(group_DeltaMSE)/all4188; component contributions weighted3/7,3/7,1/7',
    'positive_gripper_harm_share': 'sum(max(DeltaG,0) in group)/sum(max(DeltaG,0) all); null if denominator0',
    'translation_benefit_share': 'sum(max(-DeltaT,0) in group)/sum(max(-DeltaT,0) all); null if denominator0',
    'overlapping_existing_strata_not_added': True, 'all30_demos_and_demo_x_existing_strata': True,
    'concentration_top_k': 5, 'concentration_ties': 'demo_index_ascending; descriptive_only_not_training_selection',
    'cross_seed': 'same_row_intersection_union_pairwise_Jaccard; all_three_demo_mean_tradeoff; no majority override',
    'effect_stratified_rows_not_independent_validation': True, 'no_epsilon_or_threshold_sweep': True,
    'numerical_recompute_tolerance': 1e-12, 'empty_group': 'n0_mean_median_null; additive_sum0; zero_denominator_null',
    'no_significance_causal_optimization_conflict_or_training_promotion': True,
    'parent_primary_flags_and_stop_decisions_preserved': True}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def load_protocol(path):
    c = load_json(path)
    require(c.get('check_type') == 'libero_task7_cached_component_tradeoff' and c.get('formal_result') is False and
        c.get('scope') == SCOPE and c.get('analysis') == ANALYSIS, 'frozen read-only scope or definitions differ')
    require(c['runner_sha256'] == sha(Path(__file__)), 'read-only analyzer code differs')
    require(set(c['inputs']) == set(PINS), 'cached input roles differ')
    paths = {}
    for name, pin in PINS.items():
        row = c['inputs'][name]
        require(row['sha256'] == pin and sha(ROOT / row['path']) == pin, f'cached input hash differs: {name}')
        paths[name] = ROOT / row['path']
    return c, paths


def validate_source(source, audit, design, binding):
    require(source['status'] == audit['status'] == 'complete' and source['failure'] is None and
        source['formal_result'] is False and source['training_run'] is False and
        audit['result_sha256'] == PINS['source_result'], 'source diagnostic incomplete or mismatched')
    require(source['scope'] == design['scope'] and source['analysis'] == design['analysis'] and
        source['runtime']['actual_calls'] == binding['expected_actual_calls'], 'source design or actual budget differs')
    require(source['scope']['source_role'] == 'meta_support' and source['scope']['demo_indices'] == list(range(30)) and
        source['scope']['checkpoint_updates'] == 500 and source['scope']['default_closed_loop_stabilization_steps'] == 5,
        'source role or fixed Policy differs')
    for key in ('source_query_accessed', 'qualification_accessed', 'reserved_accessed', 'validation_accessed', 'heldout_accessed', 'final_evaluation_accessed'):
        require(source['scope'][key] is False, f'sealed source boundary differs: {key}')
    require(source['task_results']['7']['archive']['sha256'] == PINS['source_task7_archive'], 'not the pinned task7 archive')


def validate_labels(a):
    target, demo, steps, masks = (a[k] for k in ('target_actions', 'demonstration_indices', 'action_indices', 'command_masks'))
    n = len(target)
    require(n == 4188 and target.shape == (n, 7) and target.dtype == np.float32 and np.isfinite(target).all() and
        np.abs(target).max() <= 1 and demo.dtype == steps.dtype == np.int64 and masks.dtype == np.bool_ and masks.shape == (n, 7), 'cached target/index/mask interface differs')
    require(a['mask_names'].tolist() == list(STRATA) and a['condition_names'].tolist() == list(CONDITIONS), 'condition or command label order differs')
    require(np.array_equal(demo, np.sort(demo)) and set(demo.tolist()) == set(range(30)), 'support demonstration role/order differs')
    signs = np.sign(target[:, 6])
    expected = np.column_stack((np.ones(n, bool), steps == 1, signs < 0, signs > 0, signs == 0,
                               np.zeros(n, bool), np.zeros(n, bool)))
    previous = np.sign(a['previous_expert_gripper'])
    for d in range(30):
        rows = np.flatnonzero(demo == d)
        require(np.array_equal(steps[rows], np.arange(1, len(rows) + 1)), 't0 exclusion or causal action indices differ')
        for col, transitions in ((5, (signs[rows] > 0) & (previous[rows] <= 0)), (6, (signs[rows] < 0) & (previous[rows] >= 0))):
            centers = steps[rows][transitions]
            expected[rows, col] = np.any(np.abs(steps[rows, None] - centers[None]) <= 5, axis=1) if len(centers) else False
    require(np.array_equal(masks, expected), 'cached expert command masks differ')
    return signs.astype(np.int8), masks[:, 6].astype(np.int8) + 2 * masks[:, 5].astype(np.int8)


def stat(values):
    v = np.asarray(values, np.float64)
    require(v.ndim == 1 and np.isfinite(v).all(), 'invalid cached metric')
    return {'n': len(v), 'sum': float(v.sum()), 'mean': float(v.mean()) if len(v) else None,
            'median': float(np.median(v)) if len(v) else None}


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def direction_codes(t, g):
    require(t.shape == g.shape and np.isfinite(t).all() and np.isfinite(g).all(), 'invalid direction rows')
    return (3 * (np.sign(t).astype(np.int8) + 1) + np.sign(g).astype(np.int8) + 1).astype(np.int8)


def sign_fates(base, changed, targets):
    b, c, y = np.sign(base), np.sign(changed), np.sign(targets)
    return np.where(y == 0, 4, np.where(b == y, np.where(c == y, 0, 1), np.where(c == y, 2, 3))).astype(np.int8)


def group_summary(mask, delta, cross, energy, base_g, changed_g, target_g, fates):
    n, total = int(mask.sum()), len(mask)
    positive_g = np.maximum(delta['gripper'], 0)
    benefit_t = np.maximum(-delta['translation'], 0)
    contribution = {k: float(v[mask].sum() / total) for k, v in delta.items()}
    weighted = {k: contribution[k] * weight / 7 for k, weight in (('translation', 3), ('rotation', 3), ('gripper', 1))}
    require(abs(sum(weighted.values()) - contribution['all7']) <= 1e-12, 'weighted component contribution identity failed')
    return {'n': n, 'fraction_of_all_rows': n / total,
        'delta_mse': {k: stat(v[mask]) for k, v in delta.items()}, 'global_signed_contribution': contribution,
        'weighted7_signed_contribution': weighted,
        'positive_gripper_harm_sum': float(positive_g[mask].sum()),
        'positive_gripper_harm_share': fraction(positive_g[mask].sum(), positive_g.sum()),
        'translation_benefit_sum': float(benefit_t[mask].sum()),
        'translation_benefit_share': fraction(benefit_t[mask].sum(), benefit_t.sum()),
        'cross_mean': {k: stat(v[mask])['mean'] for k, v in cross.items()},
        'energy_mean': {k: stat(v[mask])['mean'] for k, v in energy.items()},
        'gripper_sign_fates': {name: int(np.sum(mask & (fates == i))) for i, name in enumerate(FATES)},
        'gripper_sign_flips': int(np.sum(mask & (np.sign(base_g) != np.sign(changed_g)))),
        'gripper_values': {k: stat(v[mask])['mean'] for k, v in (('original', base_g), ('changed', changed_g), ('target', target_g))}}


def validate_partition(groups, overall):
    require(sum(g['n'] for g in groups.values()) == overall['n'], 'partition row counts differ')
    for component in COMPONENTS:
        require(abs(sum(g['global_signed_contribution'][component] for g in groups.values()) -
                    overall['global_signed_contribution'][component]) <= 1e-12, 'partition contribution reconstruction failed')


def case_summary(delta, cross, energy, base_g, changed_g, a, commands, windows):
    n, demo = len(base_g), a['demonstration_indices']
    codes, fates = direction_codes(delta['translation'], delta['gripper']), sign_fates(base_g, changed_g, a['target_actions'][:, 6])
    def group(mask):
        return group_summary(mask, delta, cross, energy, base_g, changed_g, a['target_actions'][:, 6], fates)
    overall = group(np.ones(n, bool))
    partitions = {'commands': {str(sign): group(commands == sign) for sign in (-1, 0, 1)},
        'windows': {label: group(windows == i) for i, label in enumerate(WINDOWS)},
        'directions': {str(i): group(codes == i) for i in range(9)},
        'gripper_sign_fates': {label: group(fates == i) for i, label in enumerate(FATES)},
        'command_x_window': {f'{sign}:{label}': group((commands == sign) & (windows == i))
                             for sign in (-1, 0, 1) for i, label in enumerate(WINDOWS)}}
    for p in partitions.values():
        validate_partition(p, overall)
    demos = [{'demo_index': d, 'all': group(demo == d), 'existing_strata':
        {s: group((demo == d) & a['command_masks'][:, i]) for i, s in enumerate(STRATA)}} for d in range(30)]
    validate_partition({str(r['demo_index']): r['all'] for r in demos}, overall)
    ranked = sorted(demos, key=lambda r: (-r['all']['positive_gripper_harm_sum'], r['demo_index']))
    return {'overall': overall, 'existing_strata': {s: group(a['command_masks'][:, i]) for i, s in enumerate(STRATA)},
        'partitions': partitions, 'demos': demos,
        'demo_mean_direction_counts': {str(i): sum(int(direction_codes(np.array([r['all']['delta_mse']['translation']['mean']]),
            np.array([r['all']['delta_mse']['gripper']['mean']]))[0]) == i for r in demos) for i in range(9)},
        'top5_demo_positive_gripper_harm': {'demo_indices': [r['demo_index'] for r in ranked[:5]],
            'share': fraction(sum(r['all']['positive_gripper_harm_sum'] for r in ranked[:5]), overall['positive_gripper_harm_sum'])}}, codes, fates


def analyze(source, archive):
    keys = ['target_actions', 'previous_expert_gripper', 'demonstration_indices', 'action_indices', 'command_masks', 'mask_names', 'condition_names']
    a = {k: archive[k] for k in keys}
    commands, windows = validate_labels(a)
    membership = {'demonstration_indices': a['demonstration_indices'], 'action_indices': a['action_indices'],
                  'expert_command_sign': commands, 'window_partition': windows}
    cases, global_reference, old = {}, {}, source['task_results']['7']['models']
    for seed in SEEDS:
        predictions = archive[f'predictions_seed{seed}']
        require(predictions.shape == (8, 4188, 7) and predictions.dtype == np.float32 and np.isfinite(predictions).all() and
            archive[f'prediction_valid_seed{seed}'].all() and np.max(np.abs(predictions)) <= 1, 'cached predictions incomplete/invalid')
        require(np.array_equal(predictions[0], predictions[1]), 'cached identity outputs differ')
        for i in range(8):
            require(np.array_equal(archive[f'repeat_seed{seed}_condition{i}'], predictions[i, :32]), 'cached repeat outputs differ')
        require(old[str(seed)]['checkpoint']['optimizer_updates'] == 500 and old[str(seed)]['checkpoint']['arm'] == 'source_baseline', 'not baseline500')
        cases[str(seed)], global_reference[str(seed)] = {}, {}
        for index, condition in enumerate(CONDITIONS):
            delta, cross, energy = {}, {}, {}
            for component, (start, stop) in COMPONENTS.items():
                b = predictions[0, :, start:stop].astype(np.float64)
                c = predictions[index, :, start:stop].astype(np.float64)
                y = a['target_actions'][:, start:stop].astype(np.float64)
                expected = {'clean_MSE': np.mean((b - y) ** 2, axis=1), 'changed_MSE': np.mean((c - y) ** 2, axis=1),
                    'DeltaMSE': np.mean((c - y) ** 2 - (b - y) ** 2, axis=1),
                    'error_alignment_cross_term': 2 * np.mean((b - y) * (c - b), axis=1), 'response_energy_term': np.mean((c - b) ** 2, axis=1)}
                for measure, expected_values in expected.items():
                    v = archive[f'terms_seed{seed}_{condition}_{component}_{measure}']
                    require(v.shape == (4188,) and v.dtype == np.float64 and np.isfinite(v).all() and
                        np.max(np.abs(v - expected_values)) <= 1e-12, 'cached signed error terms differ')
                delta[component] = archive[f'terms_seed{seed}_{condition}_{component}_DeltaMSE']
                cross[component] = expected['error_alignment_cross_term']
                energy[component] = expected['response_energy_term']
            global_reference[str(seed)][condition] = {k: stat(v) for k, v in delta.items()}
            for i, label in enumerate(STRATA):
                for component, v in delta.items():
                    actual = stat(v[a['command_masks'][:, i]])
                    previous = old[str(seed)]['statistics']['by_condition_stratum'][condition][label]['terms'][component]['DeltaMSE']
                    require(actual['n'] == previous['n'] and all(actual[k] == previous[k] if actual[k] is None else
                        abs(actual[k] - previous[k]) <= 1e-12 for k in ('mean', 'median')), 'parent stratum statistic differs')
            if condition in LOCAL:
                result, codes, fates = case_summary(delta, cross, energy, predictions[0, :, 6], predictions[index, :, 6], a, commands, windows)
                cases[str(seed)][condition] = result
                membership[f'direction_seed{seed}_{condition}'] = codes
                membership[f'sign_fate_seed{seed}_{condition}'] = fates
    common = {}
    for condition in LOCAL:
        masks = {seed: membership[f'direction_seed{seed}_{condition}'] == 2 for seed in SEEDS}
        shared, union = np.logical_and.reduce(list(masks.values())), np.logical_or.reduce(list(masks.values()))
        membership[f'all_three_same_row_tradeoff_{condition}'] = shared
        pairwise = {}
        for i, seed in enumerate(SEEDS):
            for other in SEEDS[i + 1:]:
                intersection, combined = masks[seed] & masks[other], masks[seed] | masks[other]
                pairwise[f'{seed}:{other}'] = {'intersection': int(intersection.sum()), 'union': int(combined.sum()),
                    'jaccard': fraction(intersection.sum(), combined.sum())}
        demos = [d for d in range(30) if all(direction_codes(
            np.array([cases[str(seed)][condition]['demos'][d]['all']['delta_mse']['translation']['mean']]),
            np.array([cases[str(seed)][condition]['demos'][d]['all']['delta_mse']['gripper']['mean']]))[0] == 2 for seed in SEEDS)]
        common[condition] = {'per_seed_same_row_tradeoff_counts': {str(s): int(masks[s].sum()) for s in SEEDS},
            'all_three_same_row_count': int(shared.sum()), 'any_seed_same_row_count': int(union.sum()),
            'shared_rows_by_expert_command': {str(s): int(np.sum(shared & (commands == s))) for s in (-1, 0, 1)},
            'shared_rows_by_window_partition': {label: int(np.sum(shared & (windows == i))) for i, label in enumerate(WINDOWS)},
            'pairwise': pairwise, 'all_three_demo_mean_tradeoff_indices': demos}
    return {'cases': cases, 'global_all8_references': global_reference, 'cross_seed': common,
        'parent_task7_descriptive_flags_preserved': source['task_results']['7']['descriptive_flags']}, membership


def run(config_path, output, membership_output):
    require(not output.exists() and not membership_output.exists(), 'analysis output/memberships already exists')
    cfg, paths = load_protocol(config_path)
    started, begin = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    source, audit, design, binding = [load_json(paths[n]) for n in ('source_result', 'source_audit', 'source_design', 'source_binding')]
    validate_source(source, audit, design, binding)
    with np.load(paths['source_task7_archive'], allow_pickle=False) as archive:
        analysis, memberships = analyze(source, archive)
    with membership_output.open('xb') as f:
        np.savez_compressed(f, **memberships)
    result = {'check_type': 'libero_task7_cached_component_tradeoff', 'status': 'complete', 'formal_result': False,
        'scope': SCOPE, 'analysis_definition': ANALYSIS, **analysis,
        'memberships': {'path': str(membership_output), 'sha256': sha(membership_output)},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'inputs': cfg['inputs'], 'started_at_utc': begin, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'device': 'cpu',
            'elapsed_seconds': time.perf_counter() - started, 'new_model_environment_optimizer_HDF5_checkpoint_RGB_calls': 0},
        'interpretation': {'diagnostic_only': True, 'effect_stratification_not_independent_validation': True,
            'overlapping_source_strata_not_additive': True, 'no_training_causal_conflict_or_module_claim': True}}
    with output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--memberships', type=Path, default=DEFAULT_MEMBERSHIPS)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve(), args.memberships.resolve())
    print(json.dumps({'status': result['status'], 'runtime': result['runtime']}))
