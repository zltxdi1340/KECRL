"""Read-only validation B/C public-trajectory diagnostic, all800 fixed pairs.

Reuse the established censor-aware command/prefix metrics and frozen public
episode validator. Never deserialize weights, construct Policy/environment,
read images/HDF5/initial banks, or change the completed negative G decision.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

from experiments import analyze_libero_source_internal_six_group_paired_traces as metrics
from experiments import audit_libero_formal_transfer_v2 as public_audit
from experiments import libero_formal_transfer_protocol_v2 as protocol

ROOT = protocol.ROOT
require, sha = protocol.require, protocol.sha
TASKS, SEEDS, ARMS, INDICES = [2, 6], protocol.GATE_SEEDS, protocol.ARMS[1:], list(range(10, 30))
TICKS = [1, 5, 10, 20]
OUTCOMES = ['both_success', 'B_only', 'C_only', 'both_failure']
TRACE_ROOT = Path('/tmp/libero-formal-G-artifacts-20261105-v2')
EPISODE_KEYS = set('status task_id training_seed arm initial_state_role initial_state_index local_initial_state_index '
    'stabilization_steps environment_seed checkpoint_sha256 checkpoint_weight_hash stepwise_public_state_trace '
    'reset_observation_sha256 last_attempted_action success steps terminal_reason reward_sum'.split())
STATE0_KEYS = set('step joint_state gripper_state joint_offset_l2_from_start gripper_offset_l2_from_start'.split())
STATE_KEYS = STATE0_KEYS | {'action', 'joint_offset_from_start', 'gripper_offset_from_start'}
PAIR_METRICS = ['first_action_gap_l2', 'first_translation_gap_l2', 'first_rotation_gap_l2', 'first_gripper_abs_gap',
    'B_first_close_step', 'C_first_close_step', 'B_first_open_after_close_step', 'C_first_open_after_close_step',
    'close_step_delta_B_minus_C', 'open_step_delta_B_minus_C', 'first_motion_action_difference_step',
    'first_translation_action_difference_step', 'first_gripper_sign_difference_step',
    'first_joint_state_difference_step', 'first_gripper_state_difference_step',
    'common_prefix_steps', 'common_prefix_translation_action_gap_mean', 'first20_translation_action_gap_mean',
    'first20_rotation_action_gap_mean', 'first20_motion_action_gap_mean', 'first20_gripper_action_abs_gap_mean',
    'first20_gripper_sign_disagreement_steps', 'first20_mean_joint_gap', 'first20_mean_gripper_gap',
    'first20_B_joint_path', 'first20_C_joint_path', 'first20_joint_path_delta_B_minus_C',
    'first20_B_gripper_path', 'first20_C_gripper_path', 'first20_gripper_path_delta_B_minus_C',
    'winner_minus_loser_joint_path', 'winner_minus_loser_gripper_path'] + [
    f'step{t}_{name}' for t in TICKS for name in ('pre_joint_gap', 'post_joint_gap', 'post_gripper_gap',
                                               'joint_displacement_delta', 'gripper_displacement_delta')]
BALANCED_METRICS = ['first_translation_gap_l2', 'first20_translation_action_gap_mean',
    'first20_joint_path_delta_B_minus_C', 'first20_gripper_path_delta_B_minus_C', 'close_step_delta_B_minus_C']


def expected_keys():
    return {(t, s, a, i) for t in TASKS for s in SEEDS for a in ARMS for i in INDICES}


def preflight(config_path, binding_path):
    config, binding = (json.loads(Path(p).read_text()) for p in (config_path, binding_path))
    require(binding['config_sha256'] == sha(config_path), 'readonly definition changed')
    require(config['diagnostic_only'] is True and config['G_outcome_already_observed'] is True and
            config['tasks'] == TASKS and config['seeds'] == SEEDS and config['arms'] == ARMS and
            config['global_initial_state_indices'] == INDICES and config['initial_state_role'] == 'candidate_acceptance',
            'validation-only task/seed/arm/state scope differs')
    require(config['checkpoint_update'] == 500 and config['stabilization_steps'] == 5 and
            config['environment_seed'] == 20261022 and config['maximum_episode_steps'] == 300 and
            config['common_prefix_steps'] == 20 and config['snapshot_steps'] == TICKS and
            config['outcome_strata'] == OUTCOMES and Path(config['trace_root']) == TRACE_ROOT,
            'fixed endpoint/prefix/command scope differs')
    names = {'result', 'training', 'completion', 'complete_audit', 'training_audit', 'prior_analysis'}
    require(set(config['inputs']) == names and all(Path(p['path']).suffix == '.json' for p in config['inputs'].values()),
            'metadata input scope differs')
    require({'runner', 'tests', 'command_metrics', 'public_validator', 'protocol'}.issubset(binding['files']),
            'readonly execution dependencies incomplete')
    for name, pin in {**config['inputs'], **binding['files']}.items():
        p = Path(pin['path']); p = p if p.is_absolute() else ROOT/p
        require(sha(p) == pin['sha256'], f'input/dependency hash differs: {name}')
    require((ROOT/binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'wrong readonly entry')
    docs = {k: json.loads(Path(v['path']).read_text()) for k, v in config['inputs'].items()}
    done, raw, audit, train_audit = (docs[n] for n in ('completion', 'result', 'complete_audit', 'training_audit'))
    require(done['status'] == 'complete_audited_G' and done['decision']['passed'] is False and
            audit['status'] == 'passed_complete_audit' and audit['stage'] == raw['stage'] == 'G' and
            done['decision'] == audit['decision'] == docs['prior_analysis']['G_decision_unchanged'],
            'complete negative-G lineage missing')
    require(done['result']['sha256'] == audit['result_sha256'] == config['inputs']['result']['sha256'] and
            done['audit']['sha256'] == config['inputs']['complete_audit']['sha256'] and
            train_audit['status'] == 'passed_training_audit' and train_audit['training_sha256'] ==
            raw['training']['sha256'] == config['inputs']['training']['sha256'] and
            raw['training_audit']['sha256'] == config['inputs']['training_audit']['sha256'], 'G audit hash lineage differs')
    selected, models = select_metadata(raw, docs['training'])
    budget = config['read_budget']
    expected_budget = {'metadata_JSON_inputs': 6, 'unique_episode_trace_files': 1600, 'pairs': 800,
        'policy_action_rows': sum(r['steps'] for r in selected.values()),
        'public_state_trace_rows': sum(r['steps']+1 for r in selected.values()),
        'maximum_common_prefix_pair_steps': 16000, 'checkpoint_files': 0, 'HDF5_files': 0,
        'RGB_files': 0, 'numeric_initial_rows': 0, 'new_policy_forwards': 0,
        'new_optimizer_updates': 0, 'new_environment_steps': 0, 'heldout_payload_reads': 0}
    require(budget == expected_budget, 'saved trace read budget differs')
    return config, docs, selected, models


def select_metadata(raw, training):
    selected = {}; models = {}; allowed = expected_keys()
    require(raw['stage'] == training['stage'] == 'G' and len(raw['rollout']['episode_files']) == 6400,
            'incomplete or wrong-stage raw G artifact')
    for model in training['models']:
        task, seed, arm = model['task'], model['seed'], model['arm']
        if task not in TASKS or arm not in ARMS:
            continue
        key = task, seed, arm
        require(seed in SEEDS and key not in models and model['initial_state_role'] == 'candidate_acceptance' and
                model['initial_state_indices'] == INDICES and model['support_role'] == 'adaptation_support' and
                model['query_role'] == 'selection_query' and sorted(map(int, model['curve'])) == protocol.CURVE,
                'selected model scope differs')
        models[key] = model
    for row in raw['rollout']['episode_files']:
        task, seed, arm, index = (row[k] for k in ('task_id', 'training_seed', 'arm', 'initial_state_index'))
        if task not in TASKS or arm not in ARMS:
            continue
        key = task, seed, arm, index
        require(key in allowed and key not in selected and row['initial_state_role'] == 'candidate_acceptance' and
                row['local_initial_state_index'] == index-10, 'selected episode role/index differs')
        path = TRACE_ROOT/f'task{task}-seed{seed}-{arm}-global{index:02d}.json'
        require(Path(row['path']).resolve() == path.resolve(), 'selected episode path is outside its fixed G key')
        require(type(row['success']) is bool and type(row['steps']) is int and 1 <= row['steps'] <= 300,
                'invalid saved terminal metadata')
        selected[key] = row
    require(set(selected) == allowed and set(models) == {k[:3] for k in allowed},
            'missing/duplicate selected model or episode')
    return selected, models


def validate_episode(ep, model):
    require(set(ep) == EPISODE_KEYS, 'unexpected public episode schema/private field')
    require(ep['task_id'] in TASKS and ep['training_seed'] in SEEDS and ep['arm'] in ARMS and
            ep['task_id'] == model['task'] and ep['training_seed'] == model['seed'] and ep['arm'] == model['arm'],
            'episode identity differs from selected validation model')
    for i, row in enumerate(ep['stepwise_public_state_trace']):
        require(set(row) == (STATE0_KEYS if i == 0 else STATE_KEYS), 'unexpected public trace field')
    return public_audit.check_episode(ep, model)


def episode_row(ep, feature):
    row = {k: ep[k] for k in ('task_id', 'training_seed', 'arm', 'initial_state_index', 'success', 'steps')}
    for name in ('first_action_gripper_sign', 'first_action_translation_norm', 'first_action_rotation_norm',
                 'first_close_step', 'first_close_run_length', 'first_close_run_right_censored',
                 'first_release_step', 'first_release_run_length', 'first_release_run_right_censored',
                 'full20_available', 'full20_joint_path', 'full20_gripper_path', 'full20_close_fraction', 'full20_sign_switches'):
        row[name] = feature[name]
    return row


def pair_row(left, right, lf=None, rf=None):
    require(left['task_id'] in TASKS and left['task_id'] == right['task_id'] and
            left['training_seed'] in SEEDS and left['training_seed'] == right['training_seed'] and
            left['arm'] == ARMS[0] and right['arm'] == ARMS[1] and
            left['initial_state_role'] == right['initial_state_role'] == 'candidate_acceptance' and
            left['initial_state_index'] in INDICES and left['initial_state_index'] == right['initial_state_index'],
            'pair must use same validation task/seed/global index and B/C order')
    lf = metrics.episode_features(left) if lf is None else lf
    rf = metrics.episode_features(right) if rf is None else rf
    old = metrics.pair_features(left, right, lf, rf)
    outcome = {'left_only_success': 'B_only', 'right_only_success': 'C_only'}.get(old['outcome'], old['outcome'])
    h = min(20, left['steps'], right['steps'])
    la, ra = (np.asarray([r['action'] for r in ep['stepwise_public_state_trace'][1:h+1]], np.float64) for ep in (left, right))
    lj, rj = (np.asarray([r['joint_state'] for r in ep['stepwise_public_state_trace'][1:h+1]], np.float64) for ep in (left, right))
    lg, rg = (np.asarray([r['gripper_state'] for r in ep['stepwise_public_state_trace'][1:h+1]], np.float64) for ep in (left, right))
    def first(mask):
        indices = np.flatnonzero(mask)
        return int(indices[0])+1 if len(indices) else None
    row = {'task': left['task_id'], 'seed': left['training_seed'], 'global_initial_index': left['initial_state_index'],
           'outcome': outcome, 'B_steps': left['steps'], 'C_steps': right['steps'],
           'same_reset_digest_and_public_state': True, 'full20_available': h == 20,
           'first_gripper_sign_differs': old['first_action_sign_differs'],
           'first_gripper_sign_pair': old['first_action_sign_pair'],
           'first_close_order': old['first_close_order'], 'first_open_after_close_order': old['first_release_order'],
           'winner_close_order': None, 'winner_open_after_close_order': None,
           **{k: old[k] for k in PAIR_METRICS[:4]},
           'B_first_close_step': lf['first_close_step'], 'C_first_close_step': rf['first_close_step'],
           'B_first_open_after_close_step': lf['first_release_step'], 'C_first_open_after_close_step': rf['first_release_step'],
           'close_step_delta_B_minus_C': metrics.delta(lf['first_close_step'], rf['first_close_step']),
           'open_step_delta_B_minus_C': metrics.delta(lf['first_release_step'], rf['first_release_step']),
           'first_motion_action_difference_step': first(np.any(la[:, :6] != ra[:, :6], axis=1)),
           'first_translation_action_difference_step': first(np.any(la[:, :3] != ra[:, :3], axis=1)),
           'first_gripper_sign_difference_step': first(np.sign(la[:, 6]) != np.sign(ra[:, 6])),
           'first_joint_state_difference_step': first(np.any(lj != rj, axis=1)),
           'first_gripper_state_difference_step': first(np.any(lg != rg, axis=1)),
           'common_prefix_steps': h, 'common_prefix_translation_action_gap_mean': float(np.linalg.norm(la[:, :3]-ra[:, :3], axis=1).mean())}
    prefix = old['full20_common_prefix']
    row.update(first20_translation_action_gap_mean=row['common_prefix_translation_action_gap_mean'] if h == 20 else None,
        first20_rotation_action_gap_mean=float(np.linalg.norm(la[:, 3:6]-ra[:, 3:6], axis=1).mean()) if h == 20 else None,
        first20_motion_action_gap_mean=float(np.linalg.norm(la[:, :6]-ra[:, :6], axis=1).mean()) if h == 20 else None,
        first20_gripper_action_abs_gap_mean=float(np.abs(la[:, 6]-ra[:, 6]).mean()) if h == 20 else None,
        first20_gripper_sign_disagreement_steps=int(np.sum(np.sign(la[:, 6]) != np.sign(ra[:, 6]))) if h == 20 else None,
        first20_mean_joint_gap=prefix['mean_joint_gap_l2'], first20_mean_gripper_gap=prefix['mean_gripper_gap_l2'],
        first20_B_joint_path=prefix['left_joint_path'], first20_C_joint_path=prefix['right_joint_path'],
        first20_joint_path_delta_B_minus_C=prefix['joint_path_delta'], first20_B_gripper_path=prefix['left_gripper_path'],
        first20_C_gripper_path=prefix['right_gripper_path'], first20_gripper_path_delta_B_minus_C=prefix['gripper_path_delta'])
    winner = 1 if outcome == 'B_only' else -1 if outcome == 'C_only' else None
    row['winner_minus_loser_joint_path'] = None if winner is None or h < 20 else winner*prefix['joint_path_delta']
    row['winner_minus_loser_gripper_path'] = None if winner is None or h < 20 else winner*prefix['gripper_path_delta']
    if winner is not None:
        wf, losef, we, losee = (lf, rf, left['steps'], right['steps']) if winner == 1 else (rf, lf, right['steps'], left['steps'])
        row['winner_close_order'] = metrics.event_order(wf['first_close_step'], losef['first_close_step'], we, losee)
        row['winner_open_after_close_order'] = metrics.event_order(wf['first_release_step'], losef['first_release_step'], we, losee)
    for tick in TICKS:
        snap = old['snapshots'][str(tick)]
        row.update({f'step{tick}_{name}': snap[old_name] for name, old_name in [
            ('pre_joint_gap', 'pre_joint_gap_l2'), ('post_joint_gap', 'post_joint_gap_l2'),
            ('post_gripper_gap', 'post_gripper_gap_l2'), ('joint_displacement_delta', 'joint_offset_delta'),
            ('gripper_displacement_delta', 'gripper_offset_delta')]})
    return row


def summarize(rows):
    return {'n': len(rows), 'seeds_present': len({r['seed'] for r in rows}),
        'outcomes': {k: sum(r['outcome'] == k for r in rows) for k in OUTCOMES},
        'full20_pairs': sum(r['full20_available'] for r in rows),
        'first_gripper_sign_differences': sum(r['first_gripper_sign_differs'] for r in rows),
        'first_gripper_sign_pairs': dict(Counter(r['first_gripper_sign_pair'] for r in rows)),
        'first_close_orders': dict(Counter(r['first_close_order'] for r in rows)),
        'first_open_after_close_orders': dict(Counter(r['first_open_after_close_order'] for r in rows)),
        'winner_close_orders': dict(Counter(r['winner_close_order'] for r in rows if r['winner_close_order'] is not None)),
        'winner_open_after_close_orders': dict(Counter(r['winner_open_after_close_order'] for r in rows if r['winner_open_after_close_order'] is not None)),
        'metrics': {k: metrics.stats(r[k] for r in rows) for k in PAIR_METRICS}}


def group_results(rows):
    task_groups = []; seed_groups = []; balanced = []
    for task in TASKS:
        for stratum in ['all']+OUTCOMES:
            group = [r for r in rows if r['task'] == task and (stratum == 'all' or r['outcome'] == stratum)]
            item = {'task': task, 'stratum': stratum, 'pooled': summarize(group), 'equal_seed_means': {}}
            for seed in SEEDS:
                subset = [r for r in group if r['seed'] == seed]
                seed_groups.append({'task': task, 'seed': seed, 'stratum': stratum, **summarize(subset)})
            for metric in PAIR_METRICS:
                means = [metrics.stats(r[metric] for r in group if r['seed'] == seed)['mean'] for seed in SEEDS]
                item['equal_seed_means'][metric] = metrics.stats(means)
            task_groups.append(item)
        for seed in SEEDS:
            bs = [r for r in rows if r['task'] == task and r['seed'] == seed and r['outcome'] == 'B_only']
            cs = [r for r in rows if r['task'] == task and r['seed'] == seed and r['outcome'] == 'C_only']
            item = {'task': task, 'seed': seed, 'B_only_pairs': len(bs), 'C_only_pairs': len(cs), 'both_strata_present': bool(bs and cs)}
            for metric in BALANCED_METRICS:
                b = metrics.stats(r[metric] for r in bs); c = metrics.stats(r[metric] for r in cs)
                item[metric] = {'B_only_observed': b['n'], 'C_only_observed': c['n'],
                    'B_only_median': b['median'], 'C_only_median': c['median'],
                    'median_difference_B_only_minus_C_only': metrics.delta(b['median'], c['median'])}
            balanced.append(item)
    return task_groups, seed_groups, balanced


def run(config_path, binding_path, output, artifacts):
    require(not output.exists() and not artifacts.exists(), 'readonly outputs must be new')
    require(not output.resolve().is_relative_to(ROOT) and not artifacts.resolve().is_relative_to(ROOT),
            'analysis artifacts must remain outside repository')
    config, docs, selected, models = preflight(config_path, binding_path)
    started = time.monotonic(); artifacts.mkdir(parents=True, exist_ok=False)
    counts = {'trace_read_attempts': 0, 'trace_files_validated': 0, 'public_state_rows_validated': 0,
              'policy_action_rows_validated': 0, 'pairs_completed': 0, 'common_prefix_pair_steps': 0}
    pair_rows = []; episode_rows = []; pins = []
    def load_episode(key):
        descriptor = selected[key]; counts['trace_read_attempts'] += 1
        raw = Path(descriptor['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'saved trace checksum differs')
        ep = json.loads(raw)
        require(all(ep[k] == descriptor[k] for k in ('task_id', 'training_seed', 'arm', 'initial_state_index',
            'local_initial_state_index', 'initial_state_role', 'success', 'steps', 'terminal_reason', 'reset_observation_sha256')),
            'trace/descriptor lineage differs')
        n = validate_episode(ep, models[key[:3]])
        counts['trace_files_validated'] += 1; counts['public_state_rows_validated'] += n+1
        counts['policy_action_rows_validated'] += n
        pins.append({'path': descriptor['path'], 'sha256': descriptor['sha256']})
        return ep
    try:
        for task in TASKS:
            for seed in SEEDS:
                for index in INDICES:
                    left, right = (load_episode((task, seed, arm, index)) for arm in ARMS)
                    lf, rf = metrics.episode_features(left), metrics.episode_features(right)
                    pair_rows.append(pair_row(left, right, lf, rf))
                    episode_rows.extend([episode_row(left, lf), episode_row(right, rf)])
                    counts['pairs_completed'] += 1; counts['common_prefix_pair_steps'] += pair_rows[-1]['common_prefix_steps']
        budget = config['read_budget']
        require(counts['trace_read_attempts'] == counts['trace_files_validated'] == budget['unique_episode_trace_files'] and
                counts['pairs_completed'] == budget['pairs'] and counts['public_state_rows_validated'] == budget['public_state_trace_rows'] and
                counts['policy_action_rows_validated'] == budget['policy_action_rows'] and
                counts['common_prefix_pair_steps'] <= budget['maximum_common_prefix_pair_steps'], 'actual read budget differs')
        task_groups, seed_groups, balanced = group_results(pair_rows)
        previous = {(r['task'], r['reference']): r for r in docs['prior_analysis']['task_comparisons']}
        for group in task_groups:
            if group['stratum'] != 'all': continue
            old = previous[group['task'], 'C_pooled_bc']['paired_outcomes']
            require(group['pooled']['outcomes'] == {'both_success': old['both_success'], 'B_only': old['B_only'],
                'C_only': old['reference_only'], 'both_failure': old['both_failure']}, 'prior outcome counts changed')
        csv_pins = {}
        seed_csv = [{'task': r['task'], 'seed': r['seed'], 'stratum': r['stratum'], 'pairs': r['n'],
            'full20_pairs': r['full20_pairs'], **{f'{k}_median': r['metrics'][k]['median'] for k in PAIR_METRICS},
            **{f'{k}_observed': r['metrics'][k]['n'] for k in PAIR_METRICS}} for r in seed_groups]
        for name, rows in [('pairs', pair_rows), ('episodes', episode_rows), ('seed_strata', seed_csv)]:
            p = artifacts/f'{name}.csv'
            with p.open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            csv_pins[name] = {'path': str(p), 'sha256': sha(p), 'rows': len(rows)}
        result = {'status': 'complete_posthoc_readonly_validation_BC_trace_diagnostic', 'formal_result': False,
            'task_groups': task_groups, 'seed_groups': seed_groups, 'within_seed_outcome_comparisons': balanced,
            'G_decision_unchanged': docs['completion']['decision'], 'actual_access': counts,
            'forbidden_access': {k: 0 for k in ('checkpoint_files', 'HDF5_files', 'RGB_files', 'numeric_initial_rows',
                'new_policy_forwards', 'new_optimizer_updates', 'new_environment_steps', 'heldout_payload_reads')},
            'csv_artifacts': csv_pins, 'episode_file_pins': pins,
            'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'binding_path': str(binding_path),
                'binding_sha256': sha(binding_path), 'runner_sha256': sha(Path(__file__)), 'metadata_inputs': config['inputs'],
                'python': platform.python_version(), 'numpy': np.__version__, 'elapsed_seconds': time.monotonic()-started,
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
            'limits': ['command close/open are not verified grasp/release events',
                'public displacement is not target error; outcome strata are post-hoc and shared-seed',
                'exact saved-value divergence is not expert-action error or component causality',
                'no retraining, new rollout, prediction, checkpoint selection or G admission change']}
        with output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
        return result
    except BaseException as error:
        with (artifacts/'failure.json').open('x') as handle:
            json.dump({'status': 'readonly_analysis_failed', 'error': repr(error), 'actual_access': counts,
                       'config_sha256': sha(config_path), 'G_decision_unchanged': True}, handle, indent=2); handle.write('\n')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--binding', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); p.add_argument('--artifacts', type=Path, required=True)
    a = p.parse_args(); result = run(a.config, a.binding, a.output, a.artifacts)
    print(json.dumps({'status': result['status'], 'elapsed_seconds': result['provenance']['elapsed_seconds'],
                      'actual_access': result['actual_access'], 'forbidden_access': result['forbidden_access']}))


if __name__ == '__main__':
    main()
