"""Read-only action/event/public-state decomposition of existing360 rollouts.

No torch, Policy, simulator, checkpoint, HDF5, RGB or initial-state access.
Event and state comparisons retain censoring and matched-horizon denominators.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_full_policy_closed_loop_diagnostic as legacy
from experiments import audit_libero_source_internal_spt_spi_closed_loop_diagnostic as source_audit

DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-six-group-paired-trace-readonly-analysis_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-six-group-paired-trace-analysis-execution-binding_v1.json'
CONFIG_SHA = '5f4d08db8535612c3ff2b1c31bc3865be6725c2300eda1bb6bb68c6b505b393d'
TASKS, SEEDS, ARMS = [0, 1], [20261022, 20261023, 20261024], ['A_original', 'B_fomaml', 'C_pooled_bc']
TICKS = [1, 5, 10, 20, 50, 100, 200, 300]
OUTCOMES = ['both_success', 'left_only_success', 'right_only_success', 'both_failure']
SCALARS = ['steps', 'first_action_norm', 'first_action_translation_norm', 'first_action_rotation_norm',
    'first_action_gripper', 'first_close_step', 'first_release_step', 'close_to_release_gap',
    'first_close_run_length', 'longest_contiguous_close_run', 'first_release_run_length',
    'close_step_count', 'close_fraction', 'gripper_sign_switches', 'positive_run_count', 'negative_run_count',
    'action_norm_mean', 'arm_action_norm_mean', 'first_close_run_joint_path', 'first_close_run_gripper_path',
    'first_release_run_joint_path', 'first_release_run_gripper_path',
    'full20_action_norm_mean', 'full20_translation_norm_mean', 'full20_rotation_norm_mean',
    'full20_joint_path', 'full20_gripper_path', 'full20_close_fraction', 'full20_sign_switches',
    'max_joint_offset_l2', 'max_gripper_offset_l2', 'final_joint_offset_l2', 'final_gripper_offset_l2']
SNAPSHOT = ['joint_offset_l2', 'gripper_offset_l2', 'action_norm', 'translation_norm', 'rotation_norm', 'gripper_action']
GAPS = ['action_gap_l2', 'translation_gap_l2', 'rotation_gap_l2', 'gripper_action_abs_gap',
    'pre_joint_gap_l2', 'pre_gripper_gap_l2', 'post_joint_gap_l2', 'post_gripper_gap_l2',
    'joint_offset_delta', 'gripper_offset_delta']
PREFIX = ['mean_action_gap_l2', 'mean_joint_gap_l2', 'mean_gripper_gap_l2', 'max_action_gap_l2',
    'left_joint_path', 'right_joint_path', 'joint_path_delta', 'left_gripper_path', 'right_gripper_path', 'gripper_path_delta']


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    return legacy.sha256_file(Path(path))


def stats(values):
    values = list(values)
    present = [float(v) for v in values if v is not None]
    require(all(math.isfinite(v) for v in present), 'nonfinite analysis summary')
    return {'total': len(values), 'n': len(present), 'missing': len(values)-len(present),
            'mean': statistics.fmean(present) if present else None,
            'median': statistics.median(present) if present else None,
            'minimum': min(present) if present else None, 'maximum': max(present) if present else None}


def norm(value): return float(np.linalg.norm(value))


def delta(left, right): return None if left is None or right is None else left-right


def key(ep): return ep['task_id'], ep['training_seed'], ep['arm'], ep['initial_state_index']


def command_runs(actions):
    runs = []
    for i, action in enumerate(actions, 1):
        sign = legacy._sign(float(action[6]))
        if not runs or runs[-1]['sign'] != sign:
            runs.append({'sign': sign, 'start': i, 'end': i, 'length': 1})
        else: runs[-1]['end'] = i; runs[-1]['length'] += 1
    return runs


def episode_features(ep):
    trace, n = ep['stepwise_public_state_trace'], ep['steps']
    actions = np.asarray([r['action'] for r in trace[1:]], np.float64)
    joints = np.asarray([r['joint_state'] for r in trace], np.float64)
    grip = np.asarray([r['gripper_state'] for r in trace], np.float64)
    require(actions.shape == (n, 7) and joints.shape == (n+1, 7) and grip.shape == (n+1, 2), 'episode interface differs')
    require(np.isfinite(actions).all() and np.isfinite(joints).all() and np.isfinite(grip).all(), 'nonfinite public trace')
    old = legacy._action_summary(actions.tolist())
    # Reuse numerical command metrics, never the legacy physical-phase heuristic.
    old.pop('command_phase_at_end')
    runs = command_runs(actions)
    positives = [r for r in runs if r['sign'] > 0]
    first = positives[0] if positives else None
    release = next((r for r in runs if first is not None and r['sign'] < 0 and r['start'] > first['start']), None)
    jpath = np.linalg.norm(np.diff(joints, axis=0), axis=1)
    gpath = np.linalg.norm(np.diff(grip, axis=0), axis=1)
    joffset = np.linalg.norm(joints-joints[0], axis=1)
    goffset = np.linalg.norm(grip-grip[0], axis=1)
    phase_value = lambda r, field: None if r is None else r[field]
    path = lambda r, values: None if r is None else float(values[r['start']-1:r['end']].sum())
    row = {**old, 'episode_id': list(key(ep)), 'task_id': ep['task_id'], 'training_seed': ep['training_seed'],
        'arm': ep['arm'], 'initial_state_index': ep['initial_state_index'], 'success': ep['success'],
        'outcome': ep['terminal_reason'], 'steps': n, 'reset_observation_sha256': ep['reset_observation_sha256'],
        'first_action_translation_norm': norm(actions[0, :3]), 'first_action_rotation_norm': norm(actions[0, 3:6]),
        'first_close_run_length': phase_value(first, 'length'),
        'first_close_run_right_censored': None if first is None else first['end'] == n,
        'first_release_run_length': phase_value(release, 'length'),
        'first_release_run_right_censored': None if release is None else release['end'] == n,
        'close_to_release_gap': delta(old['first_release_step'], old['first_close_step']),
        'first_close_run_joint_path': path(first, jpath), 'first_close_run_gripper_path': path(first, gpath),
        'first_release_run_joint_path': path(release, jpath), 'first_release_run_gripper_path': path(release, gpath),
        'positive_run_count': len(positives), 'negative_run_count': sum(r['sign'] < 0 for r in runs),
        'close_fraction': old['close_step_count']/n,
        'command_stage_at_end': 'no_close_observed' if first is None else 'close_no_later_open' if release is None else 'open_after_close_observed',
        'max_joint_offset_l2': float(joffset.max()), 'max_gripper_offset_l2': float(goffset.max()),
        'final_joint_offset_l2': float(joffset[-1]), 'final_gripper_offset_l2': float(goffset[-1]),
        'full20_available': n >= 20,
        'full20_action_norm_mean': float(np.linalg.norm(actions[:20], axis=1).mean()) if n >= 20 else None,
        'full20_translation_norm_mean': float(np.linalg.norm(actions[:20, :3], axis=1).mean()) if n >= 20 else None,
        'full20_rotation_norm_mean': float(np.linalg.norm(actions[:20, 3:6], axis=1).mean()) if n >= 20 else None,
        'full20_joint_path': float(jpath[:20].sum()) if n >= 20 else None,
        'full20_gripper_path': float(gpath[:20].sum()) if n >= 20 else None,
        'full20_close_fraction': float(np.mean(actions[:20, 6] > 0)) if n >= 20 else None,
        'full20_sign_switches': int(np.sum(np.sign(actions[1:20, 6]) != np.sign(actions[:19, 6]))) if n >= 20 else None,
        'snapshots': {}}
    require(old['first_close_step'] == (None if first is None else first['start']) and
            old['first_release_step'] == (None if release is None else release['start']), 'command helper disagreement')
    for t in TICKS:
        row['snapshots'][str(t)] = {m: None for m in SNAPSHOT}
        if t <= n:
            a = actions[t-1]
            row['snapshots'][str(t)] = dict(zip(SNAPSHOT, [float(joffset[t]), float(goffset[t]), norm(a), norm(a[:3]), norm(a[3:6]), float(a[6])]))
    return row


def event_order(left, right, left_end, right_end):
    if left is not None and right is not None:
        return 'observed_left_earlier' if left < right else 'observed_left_later' if left > right else 'observed_equal'
    if left is None and right is None: return 'both_not_observed'
    if left is not None: return 'left_observed_reference_later_bound' if left <= right_end else 'indeterminate_reference_ended_before_left_event'
    return 'right_observed_left_later_bound' if right <= left_end else 'indeterminate_left_ended_before_reference_event'


def pair_features(left, right, lf, rf):
    require(left['task_id'] == right['task_id'] and left['initial_state_index'] == right['initial_state_index'] and
            left['reset_observation_sha256'] == right['reset_observation_sha256'], 'paired reset provenance differs')
    ltrace, rtrace = left['stepwise_public_state_trace'], right['stepwise_public_state_trace']
    for field in ['joint_state', 'gripper_state']:
        require(np.array_equal(np.asarray(ltrace[0][field]), np.asarray(rtrace[0][field])), 'paired public reset state differs')
    la = np.asarray([r['action'] for r in ltrace[1:]], np.float64)
    ra = np.asarray([r['action'] for r in rtrace[1:]], np.float64)
    lj = np.asarray([r['joint_state'] for r in ltrace], np.float64); rj = np.asarray([r['joint_state'] for r in rtrace], np.float64)
    lg = np.asarray([r['gripper_state'] for r in ltrace], np.float64); rg = np.asarray([r['gripper_state'] for r in rtrace], np.float64)
    lsuccess, rsuccess = left['success'], right['success']
    outcome = 'both_success' if lsuccess and rsuccess else 'left_only_success' if lsuccess else 'right_only_success' if rsuccess else 'both_failure'
    row = {'left_id': list(key(left)), 'right_id': list(key(right)), 'task_id': left['task_id'],
        'initial_state_index': left['initial_state_index'], 'outcome': outcome, 'same_reset': True,
        'first_action_sign_differs': legacy._sign(float(la[0, 6])) != legacy._sign(float(ra[0, 6])),
        'first_action_sign_pair': f"{legacy._sign(float(la[0,6]))}:{legacy._sign(float(ra[0,6]))}",
        'first_action_gap_l2': norm(la[0]-ra[0]), 'first_translation_gap_l2': norm(la[0,:3]-ra[0,:3]),
        'first_rotation_gap_l2': norm(la[0,3:6]-ra[0,3:6]), 'first_gripper_abs_gap': abs(float(la[0,6]-ra[0,6])),
        'scalar_deltas': {k: delta(lf[k], rf[k]) for k in SCALARS},
        'first_close_order': event_order(lf['first_close_step'], rf['first_close_step'], left['steps'], right['steps']),
        'first_release_order': event_order(lf['first_release_step'], rf['first_release_step'], left['steps'], right['steps']),
        'snapshots': {}}
    for t in TICKS:
        both = min(left['steps'], right['steps']) >= t
        snap = {'left_available': left['steps'] >= t, 'right_available': right['steps'] >= t, 'both_available': both,
                **{k: None for k in GAPS}}
        if both:
            snap.update(action_gap_l2=norm(la[t-1]-ra[t-1]), translation_gap_l2=norm(la[t-1,:3]-ra[t-1,:3]),
                rotation_gap_l2=norm(la[t-1,3:6]-ra[t-1,3:6]), gripper_action_abs_gap=abs(float(la[t-1,6]-ra[t-1,6])),
                pre_joint_gap_l2=norm(lj[t-1]-rj[t-1]), pre_gripper_gap_l2=norm(lg[t-1]-rg[t-1]),
                post_joint_gap_l2=norm(lj[t]-rj[t]), post_gripper_gap_l2=norm(lg[t]-rg[t]),
                joint_offset_delta=norm(lj[t]-lj[0])-norm(rj[t]-rj[0]),
                gripper_offset_delta=norm(lg[t]-lg[0])-norm(rg[t]-rg[0]))
        row['snapshots'][str(t)] = snap
    h = min(20, left['steps'], right['steps'])
    gap_a = np.linalg.norm(la[:h]-ra[:h], axis=1)
    ljpath, rjpath = float(np.linalg.norm(np.diff(lj[:h+1], axis=0), axis=1).sum()), float(np.linalg.norm(np.diff(rj[:h+1], axis=0), axis=1).sum())
    lgpath, rgpath = float(np.linalg.norm(np.diff(lg[:h+1], axis=0), axis=1).sum()), float(np.linalg.norm(np.diff(rg[:h+1], axis=0), axis=1).sum())
    prefix = dict(zip(PREFIX, [float(gap_a.mean()), float(np.linalg.norm(lj[1:h+1]-rj[1:h+1], axis=1).mean()),
        float(np.linalg.norm(lg[1:h+1]-rg[1:h+1], axis=1).mean()), float(gap_a.max()),
        ljpath, rjpath, ljpath-rjpath, lgpath, rgpath, lgpath-rgpath]))
    row['observed_common_prefix'] = {'steps': h, **prefix}
    row['full20_common_prefix'] = {'available': h == 20, **(prefix if h == 20 else {k: None for k in PREFIX})}
    return row


def summarize_episodes(rows):
    return {'n': len(rows), 'successes': sum(r['success'] for r in rows),
        'no_close': sum(r['first_close_step'] is None for r in rows),
        'close_without_later_open': sum(r['first_close_step'] is not None and r['first_release_step'] is None for r in rows),
        'first_close_run_right_censored': sum(r['first_close_run_right_censored'] is True for r in rows),
        'full20_available': sum(r['full20_available'] for r in rows),
        'first_command_signs': dict(Counter(str(r['first_action_gripper_sign']) for r in rows)),
        'command_stage_at_end': dict(Counter(r['command_stage_at_end'] for r in rows)),
        'metrics': {k: stats(r[k] for r in rows) for k in SCALARS},
        'snapshots': {str(t): {k: stats(r['snapshots'][str(t)][k] for r in rows) for k in SNAPSHOT} for t in TICKS}}


def summarize_pairs(rows):
    first = ['first_action_gap_l2', 'first_translation_gap_l2', 'first_rotation_gap_l2', 'first_gripper_abs_gap']
    return {'n': len(rows), 'outcomes': {k: sum(r['outcome'] == k for r in rows) for k in OUTCOMES},
        'state_indices': [r['initial_state_index'] for r in rows],
        'first_action_sign_differences': sum(r['first_action_sign_differs'] for r in rows),
        'first_action_sign_pairs': dict(Counter(r['first_action_sign_pair'] for r in rows)),
        'first_close_orders': dict(Counter(r['first_close_order'] for r in rows)),
        'first_release_orders': dict(Counter(r['first_release_order'] for r in rows)),
        'first_action_metrics': {k: stats(r[k] for r in rows) for k in first},
        'scalar_deltas': {k: stats(r['scalar_deltas'][k] for r in rows) for k in SCALARS},
        'snapshots': {str(t): {'both_available': sum(r['snapshots'][str(t)]['both_available'] for r in rows),
            'metrics': {k: stats(r['snapshots'][str(t)][k] for r in rows) for k in GAPS}} for t in TICKS},
        'full20': {'n': sum(r['full20_common_prefix']['available'] for r in rows),
            'metrics': {k: stats(r['full20_common_prefix'][k] for r in rows) for k in PREFIX}},
        'observed_common_prefix': {'lengths': stats(r['observed_common_prefix']['steps'] for r in rows),
            'metrics': {k: stats(r['observed_common_prefix'][k] for r in rows) for k in PREFIX}}}


def load_inputs(config_path, binding_path):
    require(sha(config_path) == CONFIG_SHA, 'frozen readonly analysis definition differs')
    config = json.loads(config_path.read_text())
    binding = json.loads(binding_path.read_text())
    require(binding.get('formal_result') is False and binding.get('config_sha256') == CONFIG_SHA and
            {'analyzer', 'tests', 'legacy_action_metrics', 'source_public_validator'}.issubset(binding.get('files', {})),
            'analysis execution binding missing or mismatched')
    for name, pin in {**config['inputs'], **binding['files']}.items():
        p = Path(pin['path']); p = p if p.is_absolute() else ROOT/p
        require(sha(p) == pin['sha256'], f'analysis input or code pin differs: {name}')
    analyzer = Path(binding['files']['analyzer']['path'])
    analyzer = analyzer if analyzer.is_absolute() else ROOT/analyzer
    require(analyzer.resolve() == Path(__file__).resolve(), 'analyzer binding points to another entry')
    loaded = {}
    for name in ['source', 'source_audit', 'source_verification', 'source_config', 'source_binding', 'gripper_mapping_audit']:
        p = Path(config['inputs'][name]['path']); p = p if p.is_absolute() else ROOT/p
        loaded[name] = json.loads(p.read_text())
    source, audited, verified, definition = [loaded[k] for k in ['source', 'source_audit', 'source_verification', 'source_config']]
    require(source['status'] == verified['status'] == 'complete' and source['formal_result'] is False and
            audited['status'] == 'passed' and audited['experiment_status_preserved'] == 'complete' and audited['primary_summary_valid'] is True,
            'source is not the complete independently audited diagnostic')
    require(audited['result_sha256'] == verified['experiment_result_sha256'] == config['inputs']['source']['sha256'] and
            verified['independent_audit_sha256'] == config['inputs']['source_audit']['sha256'], 'source audit/verification lineage differs')
    require(source['scope'] == definition['scope'] and source['analysis'] == definition['analysis'] and
            source['provenance']['config_sha256'] == config['inputs']['source_config']['sha256'] and
            source['provenance']['binding_sha256'] == config['inputs']['source_binding']['sha256'], 'frozen source scope changed')
    require(audited['episodes_and_files_verified'] == 360 and audited['public_policy_actions_verified'] == 99457 and
            audited['public_trace_rows_verified'] == 99817 and audited['reset_equalities_verified'] == 320,
            'source recorded budget differs')
    require(sha(Path(source_audit.__file__)) == audited['audit_script_sha256'], 'public trace validator differs from source audit')
    mapping = loaded['gripper_mapping_audit']
    require(mapping['status'] == 'complete' and mapping['configuration']['inputs']['formatter']['sha256'] ==
            loaded['source_binding']['files']['robosuite_panda_gripper']['sha256'], 'gripper mapping evidence uses a different runtime')
    return config, binding, loaded


def index_source(source, config):
    require(len(source['episodes']) == 360, 'all360 source episodes required')
    endpoint_index = {(p['task_id'], p['training_seed'], p['arm']): p for p in config['endpoints']}
    index = {}
    for ep in source['episodes']:
        k = key(ep)
        require(all(type(v) is int for v in [k[0], k[1], k[3]]) and k not in index, 'duplicate/noninteger episode key')
        require(k[:3] in endpoint_index and k[3] in range(20), 'episode is outside source roles')
        source_audit.audit_public_episode(ep, endpoint_index[k[:3]])
        index[k] = ep
    require(set(index) == {(t, s, a, i) for t in TASKS for s in SEEDS for a in ARMS for i in range(20)}, 'source matrix differs')
    require(sum(e['steps'] for e in index.values()) == 99457, 'source public action budget differs')
    for task in TASKS:
        for state in range(20):
            require(len({index[task, seed, arm, state]['reset_observation_sha256'] for seed in SEEDS for arm in ARMS}) == 1,
                    'cross-arm/cross-seed reset digests differ')
    return index


def analyze(index):
    features = {k: episode_features(ep) for k, ep in index.items()}
    episode_groups = []
    primary_pairs, primary_groups = [], []
    secondary_pairs, secondary_groups = [], []
    for task in TASKS:
        for seed in SEEDS:
            for arm in ARMS:
                rows = [features[task, seed, arm, i] for i in range(20)]
                episode_groups.append({'task_id': task, 'training_seed': seed, 'arm': arm,
                    'all': summarize_episodes(rows),
                    'success': summarize_episodes([r for r in rows if r['success']]),
                    'time_limit': summarize_episodes([r for r in rows if not r['success']])})
            for reference in ['A_original', 'C_pooled_bc']:
                pairs = []
                for i in range(20):
                    lk, rk = (task, seed, 'B_fomaml', i), (task, seed, reference, i)
                    row = pair_features(index[lk], index[rk], features[lk], features[rk])
                    row.update(comparison=f'B_fomaml_minus_{reference}', training_seed=seed)
                    pairs.append(row)
                primary_pairs.extend(pairs)
                primary_groups.append({'task_id': task, 'training_seed': seed, 'comparison': f'B_fomaml_minus_{reference}',
                    'all': summarize_pairs(pairs),
                    'outcome_strata': {o: summarize_pairs([r for r in pairs if r['outcome'] == o]) for o in OUTCOMES}})
        for arm in ARMS:
            for left_seed, right_seed in [(SEEDS[0], SEEDS[1]), (SEEDS[0], SEEDS[2]), (SEEDS[1], SEEDS[2])]:
                pairs = []
                for i in range(20):
                    lk, rk = (task, left_seed, arm, i), (task, right_seed, arm, i)
                    row = pair_features(index[lk], index[rk], features[lk], features[rk])
                    row.update(arm=arm, left_seed=left_seed, right_seed=right_seed)
                    pairs.append(row)
                secondary_pairs.extend(pairs)
                secondary_groups.append({'task_id': task, 'arm': arm, 'left_seed': left_seed, 'right_seed': right_seed,
                    'all': summarize_pairs(pairs),
                    'outcome_strata': {o: summarize_pairs([r for r in pairs if r['outcome'] == o]) for o in OUTCOMES}})
    require(len(features) == 360 and len(primary_pairs) == 240 and len(secondary_pairs) == 360, 'comparison budget differs')
    for group in primary_groups + secondary_groups:
        require(sum(g['n'] for g in group['outcome_strata'].values()) == group['all']['n'] == 20, 'outcome partition does not reconstruct all pairs')
    return {'episode_rows': list(features.values()), 'episode_groups': episode_groups,
            'between_arm_pairs': primary_pairs, 'between_arm_groups': primary_groups,
            'cross_seed_pairs': secondary_pairs, 'cross_seed_groups': secondary_groups}


def run(config_path, binding_path, output):
    require(not output.exists(), 'analysis output already exists')
    started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    config, binding, loaded = load_inputs(config_path, binding_path)
    index = index_source(loaded['source'], loaded['source_config'])
    values = analyze(index)
    for group in values['between_arm_groups']:
        expected = loaded['source']['task_results'][str(group['task_id'])]['seed_results'][str(group['training_seed'])]['comparisons'][group['comparison']]['paired_outcomes']
        require(group['all']['outcomes'] == expected, 'analysis outcomes disagree with original complete result')
    result = {'status': 'complete', 'formal_result': False, 'read_only_analysis': True,
        'scope': config['scope'], 'definition': config['definition'], **values,
        'counts': {'unique_episodes': 360, 'public_trace_rows': 99817, 'public_actions': 99457,
            'between_arm_pairs': 240, 'cross_seed_pairs': 360, 'new_model_environment_training_HDF5_RGB_initial_checkpoint_calls': 0},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path),
            'binding_path': str(binding_path), 'binding_sha256': sha(binding_path), 'inputs': config['inputs'],
            'source_success_counts_preserved': {t: g['success_counts'] for t, g in loaded['source']['task_results'].items()},
            'started_at_utc': utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'runtime': {'elapsed_seconds': time.perf_counter()-started, 'python': sys.version.split()[0],
            'numpy': np.__version__, 'device': 'CPU; cached public traces only'},
        'interpretation': {'outcome_strata_are_descriptive_not_independent_validation': True,
            'no_expert_action_or_target_geometry_or_physical_grasp_labels': True,
            'public_motion_is_not_expert_state_error': True, 'later_action_gap_conflates_mapping_and_observation_drift': True,
            'no_automatic_training_change': True, 'sealed_data_accessed': False,
            'active_spt_updated': False, 'knowledge_updated': False, 'module_registered': False}}
    with output.open('x') as f: json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.binding, args.output)
    print(json.dumps({'status': result['status'], 'counts': result['counts'], 'elapsed_seconds': result['runtime']['elapsed_seconds']}))


if __name__ == '__main__': main()
