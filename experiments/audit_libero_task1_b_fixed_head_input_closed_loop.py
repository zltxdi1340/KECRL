"""Independent NumPy audit of archived fixed-head trajectories; no project imports."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'libero-task1-b-fixed-head-input-closed-loop'
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('candidate_real_history', 'candidate_repeat_current')
HEADS = ('current_all7', 'history_motion_current_gripper')
CONFIG_SHA = 'b5ff7ec57bd75aa27a5c3eef37fec36105bca763a8a33614633213ef8a48f5c4'
BINDING_SHA = '532440c6bbe0d5454d305a143d8568134800ba1b64ea2c3c874785acd564c160'
SNAPSHOTS = (1, 5, 10, 20, 50, 100, 200, 300)
PARTS = {'translation': slice(0, 3), 'rotation': slice(3, 6), 'motion': slice(0, 6),
         'gripper': slice(6, 7), 'all7': slice(0, 7)}
DEFAULT_RESULT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-independent-audit-20261104-v1.json')
READOUT_TOLERANCE = 1e-5  # Previously frozen CPU runtime bridge tolerance.


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path(value):
    p = Path(value)
    return p if p.is_absolute() else ROOT / p


def sha(value):
    return hashlib.sha256(path(value).read_bytes()).hexdigest()


def load(value):
    return json.loads(path(value).read_text())


def array_digest(arrays):
    digest = hashlib.sha256()
    for key, value in arrays.items():
        value = np.ascontiguousarray(value)
        digest.update(key.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def stats(values):
    values = list(values)
    x = np.asarray([v for v in values if v is not None], np.float64)
    require(np.isfinite(x).all(), 'nonfinite descriptive statistic')
    return {'total': len(values), 'n': len(x), 'missing': len(values)-len(x),
        'mean': float(x.mean()) if len(x) else None, 'median': float(np.median(x)) if len(x) else None,
        'min': float(x.min()) if len(x) else None, 'max': float(x.max()) if len(x) else None}


def first_difference(left, right):
    require(left.shape == right.shape and left.ndim == 2, 'unaligned comparison')
    indices = np.flatnonzero(np.any(left != right, axis=1))
    return int(indices[0]+1) if len(indices) else None


def command_events(actions):
    g = actions[:, 6]
    closed = np.flatnonzero(g > 0)
    first_close = int(closed[0]+1) if len(closed) else None
    release = np.flatnonzero(g[first_close:] < 0) if first_close is not None else []
    return {'first_close': first_close,
        'first_open_after_close': int(first_close+release[0]+1) if len(release) else None,
        'close_steps': int(np.count_nonzero(g > 0)), 'open_steps': int(np.count_nonzero(g < 0)),
        'zero_steps': int(np.count_nonzero(g == 0)),
        'adjacent_nonzero_sign_switches': int(np.count_nonzero(g[:-1]*g[1:] < 0))}


def expected_route(arm, step):
    startup, real = step <= 4, step >= 5 and arm == ARMS[0]
    history = list(range(step, max(0, step-5), -1))
    return {'policy_step': step, 'history_length': min(step, 5), 'startup': startup,
        'history_policy_steps': history, 'weight_route': 'shared_baseline_500' if startup else 'fixed_candidate_500',
        'motion_input_policy_steps': history if real else [step]*5, 'gripper_input_policy_steps': [step]*5,
        'primitive_linear_calls': 2 if real else 1,
        'route': 'shared_baseline_startup' if startup else 'candidate_motion_real_history_current_gripper' if real else
            'candidate_motion_repeat_current_current_gripper'}


def reconstruct_actions(features, heads, arm):
    """Independent batched float64 arithmetic over the float32 runtime inputs."""
    require(arm in ARMS and features.ndim == 2 and features.shape[1] == 256 and
        features.dtype == np.float32 and np.isfinite(features).all(), 'invalid feature matrix')
    n = len(features)
    base, candidate = [heads[a] for a in HEADS]
    z = (features.astype(np.float64)-base['mean'])/base['scale']
    current = (np.tile(z, (1, 5))/np.sqrt(5)).astype(np.float32).astype(np.float64)
    output, batches, rows = np.empty((n, 7), np.float64), 0, 0
    def linear(x, head):
        nonlocal batches, rows
        batches += 1
        rows += len(x)
        return x @ head['weight'].astype(np.float64).T + head['bias'].astype(np.float64)
    startup = min(4, n)
    if startup:
        output[:startup] = linear(current[:startup], base)
    if n > 4:
        output[4:] = linear(current[4:], candidate)
        if arm == ARMS[0]:
            indices = np.arange(4, n)[:, None]-np.arange(5)[None, :]
            history = (z[indices].reshape(n-4, 1280)/np.sqrt(5)).astype(np.float32).astype(np.float64)
            output[4:, :6] = linear(history, candidate)[:, :6]
    return np.tanh(output), batches, rows


def compare_pair(real, repeat):
    n = min(len(real['actions']), len(repeat['actions']))
    a, b = real['actions'][:n], repeat['actions'][:n]
    delta = b.astype(np.float64)-a.astype(np.float64)
    result = {'common_action_frames': n, 'success': {ARMS[0]: real['episode']['success'], ARMS[1]: repeat['episode']['success']},
        'steps': {ARMS[0]: len(real['actions']), ARMS[1]: len(repeat['actions'])},
        'first_action_difference': {k: first_difference(a[:, sl], b[:, sl]) for k, sl in PARTS.items()},
        'first_gripper_sign_difference': first_difference(np.sign(a[:, 6:]), np.sign(b[:, 6:])),
        'first_post_joint_difference': first_difference(real['joints'][1:n+1], repeat['joints'][1:n+1]),
        'first_post_gripper_difference': first_difference(real['grippers'][1:n+1], repeat['grippers'][1:n+1]),
        'action_delta_l2': {k: stats(np.linalg.norm(delta[:, sl], axis=1)) for k, sl in PARTS.items()},
        'commands': {ARMS[0]: command_events(real['actions']), ARMS[1]: command_events(repeat['actions'])},
        'snapshots': {}}
    for step in SNAPSHOTS:
        if step > n:
            result['snapshots'][str(step)] = None
            continue
        result['snapshots'][str(step)] = {
            'action_delta_l2': {k: float(np.linalg.norm(delta[step-1, sl])) for k, sl in PARTS.items()},
            'post_joint_delta_l2': float(np.linalg.norm(repeat['joints'][step]-real['joints'][step])),
            'post_gripper_delta_l2': float(np.linalg.norm(repeat['grippers'][step]-real['grippers'][step])),
            'reset_relative_joint_offset_l2': {arm: float(np.linalg.norm(v['joints'][step]-v['joints'][0]))
                for arm, v in zip(ARMS, (real, repeat))},
            'reset_relative_gripper_offset_l2': {arm: float(np.linalg.norm(v['grippers'][step]-v['grippers'][0]))
                for arm, v in zip(ARMS, (real, repeat))}}
    return result


class Audit:
    def __init__(self):
        self.checks, self.max_readout_error, self.max_offset_error = 0, 0., 0.
        self.replay_batches = self.replay_rows = 0

    def check(self, condition, message, entries=1):
        require(condition, message)
        self.checks += entries

    def close(self, actual, expected, tolerance, message):
        a, b = np.asarray(actual), np.asarray(expected)
        self.check(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), message+' shape/finite')
        error = float(np.max(np.abs(a-b))) if a.size else 0.
        self.check(error <= tolerance, f'{message}: max_error={error}', max(1, a.size))
        return error

    def pin(self, pin):
        self.check(sha(pin['path']) == pin['sha256'], 'file hash differs: '+pin['path'])

    def heads(self, c, seed):
        result = {}
        for name in HEADS:
            pin = c['inputs'][f'{seed}_{name}']
            self.pin(pin)
            with np.load(path(pin['path']), allow_pickle=False) as archive:
                self.check(archive['updates'].tolist() == [0, 100, 250, 500], 'endpoint schedule differs')
                self.check(archive['weights'].shape == (4, 7, 1280) and archive['biases'].shape == (4, 7), 'head endpoint shape differs')
                result[name] = {'weight': archive['weights'][-1], 'bias': archive['biases'][-1],
                    'mean': archive['scaler_mean'], 'scale': archive['scaler_scale']}
            h = result[name]
            self.check(h['weight'].dtype == h['bias'].dtype == np.float32 and
                h['mean'].shape == h['scale'].shape == (256,) and (h['scale'] > 0).all() and
                all(np.isfinite(v).all() for v in h.values()), 'invalid frozen head/scaler')
        for key in ('mean', 'scale'):
            self.check(np.array_equal(result[HEADS[0]][key], result[HEADS[1]][key]), 'support scalers differ', 256)
        arrays = {f'{name}_{key}': result[name][key] for name in HEADS for key in ('weight', 'bias')}
        arrays.update(scaler_mean=result[HEADS[0]]['mean'], scaler_scale=result[HEADS[0]]['scale'])
        return result, array_digest(arrays)

    def episode(self, ep, heads, expected_identity):
        n = ep['steps']
        self.check(ep['status'] == 'complete' and ep['task_id'] == 1 and ep['training_seed'] in SEEDS and
            ep['arm'] in ARMS and ep['initial_state_index'] in range(20) and ep['initial_state_role'] == 'candidate_nonregression'
            and ep['environment_seed'] == 20261022 and ep['stabilization_steps'] == 5 and
            type(n) is int and 1 <= n <= 300 and type(ep['success']) is bool, 'episode role/boundary differs')
        self.check(ep['terminal_reason'] == ('success' if ep['success'] else 'time_limit') and
            (ep['success'] or n == 300), 'terminal reason/length differs')
        self.check(ep['controller_identity'] == expected_identity, 'fixed same-head identity differs')
        pin = ep['public_derived_feature_archive']
        self.pin(pin)
        with np.load(path(pin['path']), allow_pickle=False) as z:
            self.check(set(z.files) == {'projection', 'actions', 'policy_steps'}, 'feature archive payload keys differ')
            features, actions, steps = z['projection'], z['actions'], z['policy_steps']
        self.check(features.shape == (n, 256) and features.dtype == np.float32 and actions.shape == (n, 7) and
            actions.dtype == np.float32 and steps.dtype == np.int32 and np.array_equal(steps, np.arange(1, n+1)) and
            np.isfinite(features).all() and np.isfinite(actions).all() and (np.abs(actions) <= 1).all(), 'derived archive interface differs')
        self.check(array_digest({'actions': actions}) == ep['action_trace_sha256'], 'action digest differs')
        trace = ep['stepwise_public_state_trace']
        self.check(len(trace) == n+1 and [r['step'] for r in trace] == list(range(n+1)), 'public trace clock differs')
        joints = np.asarray([r['joint_state'] for r in trace], np.float64)
        grippers = np.asarray([r['gripper_state'] for r in trace], np.float64)
        self.check(joints.shape == (n+1, 7) and grippers.shape == (n+1, 2) and
            np.isfinite(joints).all() and np.isfinite(grippers).all(), 'invalid public state')
        self.check(np.array_equal(np.asarray([r['action'] for r in trace[1:]], np.float64), actions.astype(np.float64)),
            'JSON and NPZ actions differ', n*7)
        self.check(ep['last_attempted_action'] == actions[-1].tolist(), 'last action differs')
        for i, row in enumerate(trace):
            for stem, states in (('joint', joints), ('gripper', grippers)):
                offset = states[i]-states[0]
                if i:
                    self.max_offset_error = max(self.max_offset_error,
                        self.close(row[stem+'_offset_from_start'], offset, 1e-12, 'public offset'))
                self.max_offset_error = max(self.max_offset_error,
                    self.close(row[stem+'_offset_l2_from_start'], np.linalg.norm(offset), 1e-12, 'public offset norm'))
            if i:
                self.check(row['history'] == expected_route(ep['arm'], i), 'causal history/input/weight routing differs')
        prefix = ep['shared_prefix_pre_observation_digests']
        self.check(len(prefix) == min(n, 5) and prefix[0] == ep['reset_observation_sha256'] and
            all(isinstance(d, str) and len(d) == 64 for d in prefix), 'observation prefix differs')
        replay, batches, rows = reconstruct_actions(features, heads, ep['arm'])
        self.replay_batches += batches
        self.replay_rows += rows
        self.max_readout_error = max(self.max_readout_error,
            self.close(actions.astype(np.float64), replay, READOUT_TOLERANCE, 'fixed-head NumPy replay'))
        return {'episode': ep, 'actions': actions, 'features': features, 'joints': joints, 'grippers': grippers}


def analyze(result_path, audit):
    d = load(result_path)
    audit.check(d['status'] == 'complete_fixed_head_input_closed_loop' and d['formal_result'] is False, 'complete diagnostic required')
    config_path, binding_path = [ROOT/'configs'/f'{PREFIX}-{s}_v1.json' for s in ('diagnostic', 'binding')]
    audit.check(sha(config_path) == d['config_sha256'] == CONFIG_SHA and
        sha(binding_path) == d['binding_sha256'] == BINDING_SHA, 'frozen config/binding hash differs')
    c, b = load(config_path), load(binding_path)
    audit.check(d['scope'] == c['scope'] and d['analysis'] == c['analysis'] and d['inputs'] == c['inputs'], 'result protocol differs')
    audit.check(b['config_sha256'] == CONFIG_SHA, 'binding config differs')
    for pin in [*c['inputs'].values(), *b['files'].values()]:
        audit.pin(pin)
    scope = c['scope']
    audit.check(scope['same_candidate_weights_between_inputs'] is True and scope['initial_state_indices'] == list(range(20))
        and scope['encoder_updates'] == scope['readout_updates'] == 500 and scope['stabilization_steps'] == 5, 'fixed scope differs')
    audit.check(all(scope[k] is False for k in ('qualification_accessed', 'reserved_accessed', 'validation_accessed',
        'heldout_accessed', 'final_evaluation_accessed', 'spt_updated', 'knowledge_updated', 'module_registered')), 'sealed roles/promotion differ')
    previous = load(c['inputs']['parent_loop_result']['path'])
    audit.check(d['initial_state_access'] == previous['initial_state_access'] and
        d['initial_state_access']['candidate_rows_numerically_decoded'] == list(range(20)) and
        d['initial_state_access']['qualification_or_reserved_rows_decoded'] == 0 and
        d['initial_state_access']['used_torch_or_pickle_deserialization'] is False, 'candidate-only access metadata differs')
    heads, identities = {}, {}
    for seed in SEEDS:
        heads[seed], head_hash = audit.heads(c, seed)
        ck = next(r['checkpoint'] for r in c['fixed_encoders'] if r['checkpoint']['training_seed'] == seed)
        audit.pin(ck)
        identities[seed] = {'encoder_tensor_sha256': ck['weight_hash'], 'head_and_scaler_sha256': head_hash}
    audit.check(len(d['episodes']) == len(d['episode_files']) == 120, 'complete120 required')
    expected = [(s, a, i) for s in SEEDS for a in ARMS for i in range(20)]
    records, resets = {}, {}
    for key, summary, pin in zip(expected, d['episodes'], d['episode_files']):
        audit.pin(pin)
        ep = load(pin['path'])
        audit.check((ep['training_seed'], ep['arm'], ep['initial_state_index']) == key, 'episode execution order differs')
        audit.check(summary == {k: v for k, v in ep.items() if k not in ('stepwise_public_state_trace', 'last_attempted_action')}, 'summary differs from episode')
        ck = next(r['checkpoint'] for r in c['fixed_encoders'] if r['checkpoint']['training_seed'] == key[0])
        audit.check(ep['encoder'] == ck and ep['readout_archives'] ==
            {a: c['inputs'][f'{key[0]}_{a}'] for a in HEADS}, 'episode endpoint provenance differs')
        if key[2] in resets:
            audit.check(resets[key[2]] == ep['reset_observation_sha256'], 'cross-model reset digest differs')
        else:
            resets[key[2]] = ep['reset_observation_sha256']
        records[key] = audit.episode(ep, heads[key[0]], identities[key[0]])

    paired, primary_rows = [], []
    outcome_names = ('both_success', 'repeat_current_only_success', 'real_history_only_success', 'both_failure')
    for seed in SEEDS:
        counts, outcomes = dict.fromkeys(ARMS, 0), dict.fromkeys(outcome_names, 0)
        for state in range(20):
            real, repeat = [records[seed, a, state] for a in ARMS]
            ra, rb = real['episode'], repeat['episode']
            x, y = ra['success'], rb['success']
            label = 'both_success' if x and y else 'real_history_only_success' if x else 'repeat_current_only_success' if y else 'both_failure'
            outcomes[label] += 1
            counts[ARMS[0]] += int(x)
            counts[ARMS[1]] += int(y)
            n, k = min(ra['steps'], rb['steps']), min(4, ra['steps'], rb['steps'])
            audit.check(ra['shared_prefix_pre_observation_digests'] == rb['shared_prefix_pre_observation_digests'], 'pair observation prefix differs')
            audit.check(np.array_equal(real['features'][:5], repeat['features'][:5]), 'pair feature prefix differs', min(n, 5)*256)
            audit.check(np.array_equal(real['actions'][:k], repeat['actions'][:k]), 'pair startup actions differ', k*7)
            for field in ('joints', 'grippers'):
                audit.check(np.array_equal(real[field][:k+1], repeat[field][:k+1]), 'startup public states differ')
            if n >= 5:
                audit.check(real['actions'][4, 6] == repeat['actions'][4, 6], 'pair step5 gripper differs')
            pair = compare_pair(real, repeat)
            pair.update(training_seed=seed, initial_state_index=state, outcome=label)
            paired.append(pair)
        primary_rows.append({'training_seed': seed, 'episodes_per_arm': 20, 'success_counts': counts,
            'success_rates': {a: v/20 for a, v in counts.items()}, 'paired_outcomes': outcomes,
            'repeat_current_minus_real_history_success_n': counts[ARMS[1]]-counts[ARMS[0]]})
    totals = {a: sum(r['success_counts'][a] for r in primary_rows) for a in ARMS}
    process = []
    for p in paired:
        real, repeat = [records[p['training_seed'], a, p['initial_state_index']]['episode'] for a in ARMS]
        process.append({'training_seed': p['training_seed'], 'initial_state_index': p['initial_state_index'], 'outcome': p['outcome'],
            'real_history_steps': real['steps'], 'repeat_current_steps': repeat['steps'],
            'action_traces_identical': real['action_trace_sha256'] == repeat['action_trace_sha256'],
            'terminal_lengths_identical': real['steps'] == repeat['steps'],
            'terminal_results_identical': real['steps'] == repeat['steps'] and real['terminal_reason'] == repeat['terminal_reason']})
    summary = {'seed_results': primary_rows, 'episodes_per_arm': 60, 'success_counts': totals,
        'success_rates': {a: v/60 for a, v in totals.items()},
        'paired_outcomes': {k: sum(r['paired_outcomes'][k] for r in primary_rows) for k in outcome_names},
        'strict_repeat_current_improvement_all_three_seeds': all(r['repeat_current_minus_real_history_success_n'] > 0 for r in primary_rows),
        'paired_action_and_terminal_comparison': process,
        'action_trace_different_pairs': sum(not p['action_traces_identical'] for p in process),
        'terminal_result_different_pairs': sum(not p['terminal_results_identical'] for p in process),
        'automatic_training_or_spt_promotion': False}
    audit.check(summary == d['summary'], 'independent primary/process summary differs')
    action_n = sum(len(r['actions']) for r in records.values())
    real_post = sum(max(0, len(records[s, ARMS[0], i]['actions'])-4) for s in SEEDS for i in range(20))
    startup = sum(min(4, len(records[s, ARMS[0], i]['actions'])) for s in SEEDS for i in range(20))
    prefix = sum(min(5, len(records[s, ARMS[0], i]['actions'])) for s in SEEDS for i in range(20))
    step5 = sum(len(records[s, ARMS[0], i]['actions']) >= 5 for s in SEEDS for i in range(20))
    expected_counts = {'adapters_built': 6, 'episodes_attempted': 120, 'episodes_completed': 120,
        'resets_attempted': 120, 'resets_completed': 120, 'encoder_forward_attempts': action_n,
        'encoder_forwards_completed': action_n, 'readout_forward_attempts': action_n, 'readout_forwards_completed': action_n,
        'policy_action_attempts': action_n, 'policy_actions_completed': action_n,
        'native_step_attempts': action_n+600, 'native_steps_completed': action_n+600,
        'zero_step_attempts': 600, 'zero_steps_completed': 600, 'reset_digest_references': 20, 'reset_digest_comparisons': 100,
        'startup_action_references': startup, 'startup_action_comparisons': startup,
        'prefix_observation_references': prefix, 'prefix_observation_comparisons': prefix,
        'prefix_feature_references': prefix, 'prefix_feature_comparisons': prefix,
        'step5_gripper_references': step5, 'step5_gripper_comparisons': step5,
        'readout_primitive_linear_calls': action_n+real_post, 'readout_primitive_linear_attempts': action_n+real_post,
        'checkpoint_load_attempts': 6, 'checkpoint_loads': 6, 'readout_bundle_load_attempts': 6, 'readout_bundle_loads': 6,
        'model_load_attempts': 6, 'models_loaded': 6, 'controller_identity_references': 3, 'controller_identity_comparisons': 3,
        'post_startup_real_history_actions': real_post, 'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_HDF5_payload_reads': 0}
    audit.check(d['actual_counts'] == expected_counts, 'reconstructed operation budgets differ')
    audit.check(action_n <= 36000 and action_n+real_post <= 53760 and action_n+600 <= 36600, 'operation cap exceeded')
    audit.check(audit.replay_rows == action_n+real_post, 'audit arithmetic rows differ')
    audit.pin(d['event_log'])
    events = [json.loads(line) for line in path(d['event_log']['path']).read_text().splitlines()]
    audit.check(len(events) == 122 and events[0]['event'] == 'runtime_ready' and events[-1]['event'] == 'complete', 'event log lifecycle differs')
    audit.check(events[-1]['actual_counts'] == expected_counts and events[-1]['summary'] == summary, 'final event differs')
    for i, (key, event) in enumerate(zip(expected, events[1:-1]), 1):
        ep = records[key]['episode']
        audit.check(event['event'] == 'episode_complete' and (event['seed'], event['arm'], event['state']) == key and
            event['episodes'] == i and event['success'] == ep['success'] and event['steps'] == ep['steps'], 'episode event differs')
    audit.check([e['utc'] for e in events] == sorted(e['utc'] for e in events), 'event clock is not monotonic')
    audit.check(d['runtime']['versions'] == c['runtime_versions'] and d['runtime']['inference_device'] == 'cpu' and
        d['runtime']['physical_renderer_gpu'] == 1 and d['runtime']['threads'] == 1 and d['runtime']['deterministic'] is True,
        'runtime provenance differs')
    audit.check(d['runtime']['launch_environment'] == {'CUDA_VISIBLE_DEVICES': '', 'MUJOCO_GL': 'egl',
        'MUJOCO_EGL_DEVICE_ID': '1', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}, 'launch environment differs')
    for pin in [*c['inputs'].values(), *b['files'].values()]:
        audit.pin(pin)
    groups = {}
    for label, group in [('all', paired), *[(str(seed), [p for p in paired if p['training_seed'] == seed]) for seed in SEEDS]]:
        groups[label] = {'pairs': len(group), 'common_action_frames': sum(p['common_action_frames'] for p in group),
            'first_motion_difference': stats(p['first_action_difference']['motion'] for p in group),
            'first_gripper_value_difference': stats(p['first_action_difference']['gripper'] for p in group),
            'first_gripper_sign_difference': stats(p['first_gripper_sign_difference'] for p in group),
            'pair_mean_action_delta_l2': {k: stats(p['action_delta_l2'][k]['mean'] for p in group) for k in PARTS},
            'commands': {arm: {k: stats(p['commands'][arm][k] for p in group) for k in group[0]['commands'][arm]} for arm in ARMS},
            'snapshots': {}}
        for step in SNAPSHOTS:
            visible = [p['snapshots'][str(step)] for p in group if p['snapshots'][str(step)] is not None]
            groups[label]['snapshots'][str(step)] = {'paired_n': len(visible),
                'action_delta_l2': {k: stats(v['action_delta_l2'][k] for v in visible) for k in PARTS},
                'post_joint_delta_l2': stats(v['post_joint_delta_l2'] for v in visible),
                'post_gripper_delta_l2': stats(v['post_gripper_delta_l2'] for v in visible)}
    return {'primary': summary, 'actual_counts': expected_counts, 'paired_trajectories': paired,
        'descriptive_groups': groups, 'terminal_reasons': dict(Counter(v['episode']['terminal_reason'] for v in records.values())),
        'audit_readout_arithmetic_batches': audit.replay_batches, 'audit_readout_arithmetic_rows': audit.replay_rows,
        'independent_checks': audit.checks, 'max_abs_readout_action_error': audit.max_readout_error,
        'max_abs_public_offset_error': audit.max_offset_error,
        'audit_decoder_counts': {'readout_archives': 6, 'feature_archives': 120, 'encoder_checkpoint_payload': 0,
            'new_CNN_or_environment_training_HDF5_initial_state_calls': 0},
        'limits': 'Stored features audit the readout; RGB/encoder/private environment state are not replayed. Same used source scenes; descriptive, no significance/qualification or unique-cause inference.'}


def run(result=DEFAULT_RESULT, output=DEFAULT_OUTPUT):
    output = Path(output)
    failure = output.with_name(output.stem+'-failed.json')
    require(not output.exists() and not failure.exists(), 'audit outputs already exist')
    audit, started = Audit(), time.perf_counter()
    provenance = {'result_path': str(result), 'script_path': str(Path(__file__).resolve()),
        'script_sha256': sha(__file__), 'python': sys.version.split()[0], 'numpy': np.__version__,
        'created_at_utc': datetime.now(timezone.utc).isoformat()}
    try:
        provenance['result_sha256'] = sha(result)
        analysis = analyze(result, audit)
        require(sha(result) == provenance['result_sha256'], 'result changed during audit')
        record = {'status': 'complete_independent_fixed_head_input_closed_loop_audit', 'formal_result': False,
            **provenance, **analysis, 'elapsed_seconds': time.perf_counter()-started}
        with output.open('x') as handle:
            json.dump(record, handle, indent=2, allow_nan=False)
            handle.write('\n')
        return record
    except Exception as exc:
        with failure.open('x') as handle:
            json.dump({'status': 'failed_independent_fixed_head_input_closed_loop_audit', 'formal_result': False,
                **provenance, 'error_type': type(exc).__name__, 'error': str(exc), 'checks_completed': audit.checks,
                'primary_validated': False}, handle, indent=2, allow_nan=False)
            handle.write('\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    record = run(args.result, args.output)
    print(json.dumps({k: record[k] for k in ('status', 'independent_checks', 'max_abs_readout_action_error',
        'max_abs_public_offset_error', 'actual_counts', 'terminal_reasons', 'elapsed_seconds')}))
