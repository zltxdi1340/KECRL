"""Read-only action/command timing analysis of all 120 stabilization pairs.

No Policy, environment, checkpoint, HDF5 or initial-state reader is imported.
Commands and public joint/gripper states do not label physical grasp phases.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_full_policy_closed_loop_diagnostic as legacy

TASKS, SEEDS, ARMS = (4, 7), (20261022, 20261023, 20261024), (0, 5)
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-stabilization-trace-alignment-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-stabilization-trace-alignment-20261102-v1.json')
PINS = {'source': '0f17ad53f86dce6aa7d38332297174702725bc5388092a4507a0214aadc1a49e',
        'source_config': 'e3654bf195aa4ea8bdb5b71f74f8920526548c177473ad3afbf946e4cd52fc06',
        'source_audit': 'ac16e8b359bb6c79cb3fbf3660c6a037025cdf34679d852a2689ac625dc3bb4f',
        'pure_action_helper': 'c26b1bfe7266dd332f0e5c55dfdb6f668d570478a790e01e5f8676766013f181'}
DISCORDANT = [(4, 20261023, 8, 'five_only_success'), (4, 20261024, 5, 'zero_only_success'),
              (4, 20261024, 17, 'five_only_success'), (7, 20261023, 3, 'five_only_success'),
              (7, 20261024, 12, 'zero_only_success'), (7, 20261024, 14, 'zero_only_success')]
OUTCOMES = ('both_success', 'zero_only_success', 'five_only_success', 'both_failure')
SCOPE = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'checkpoint_updates': 500,
    'initial_state_role': 'candidate_nonregression', 'initial_state_indices': list(range(20)),
    'source_episodes': 240, 'source_pairs': 120, 'discordant_pairs': 6,
    'source_policy_action_rows': 70188, 'source_public_state_rows': 70428,
    'analysis_device': 'cpu_scalar', 'analysis_is_deterministic_without_rng': True,
    'new_policy_forwards': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
    'new_environment_resets_or_steps': 0, 'new_checkpoint_or_hdf5_or_initial_state_reads': 0,
    'new_rgb_reads': 0, 'private_simulator_state_read': False,
    'qualification_accessed': False, 'reserved_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False,
    'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
ANALYSIS = {'first_action': 'first Policy command in each arm; its pre-clock is0 or5, not a same-time comparison',
    'policy_action_step': 'one_based_n; trace[n] contains action[n] and state_after_action[n]',
    'command_pre_clock': 'stabilization+n-1', 'state_post_clock': 'stabilization+n',
    'close_command': 'action6>0', 'open_command': 'action6<0', 'neutral': 'action6==0',
    'first_open_after_close': 'first strictly negative command after first positive command',
    'close_runs': 'strictly positive contiguous Policy commands; neutral interrupts run',
    'sign_switches': 'adjacent Policy signs, including neutral; first command is not a switch',
    'missing_event': 'null; report missing denominator and trace-end censoring; no infinity substitution',
    'first_close_run_right_censored': 'first positive run ends at the observed episode boundary',
    'common_physical_ticks': [5, 6, 10, 20, 50, 100, 200, 300], 'common_reference_tick': 5,
    'common_state_lookup': 'trace[T-stabilization]; unavailable before reset observation or after termination',
    'action_at_common_tick': 'trace[T-stabilization+1].action; null after terminal state',
    'common_tick_state_gaps': 'direct cross-arm public joint/gripper L2, not target error',
    'common_reference_motion': 'within_arm state[T]-state[5]; null if either state unavailable',
    'no_stabilization_intermediate_state_reconstruction': True,
    'paired_deltas': 'zero_minus_five; close timing difference only if both observed',
    'all_pair_reference': True, 'discordant_roster': [list(r) for r in DISCORDANT],
    'outcome_stratification_not_general_failure_distribution': True,
    'no_action_correctness_physical_grasp_phase_visual_or_causal_label': True,
    'no_significance_test_or_training_setting_selection': True}
EP_METRICS = ('policy_steps', 'first_action_norm', 'first_action_translation_norm', 'first_action_rotation_norm',
    'first_action_gripper', 'first_close_policy_step', 'first_close_pre_clock',
    'first_open_after_close_policy_step', 'first_open_after_close_pre_clock',
    'first_close_run_length', 'longest_close_run', 'close_step_count', 'close_fraction', 'gripper_sign_switches')
PAIR_METRICS = ('first_action_l2_gap', 'first_translation_l2_gap', 'first_rotation_l2_gap',
    'first_gripper_abs_gap', 'first_gripper_zero_minus_five', 'reset_joint_gap_different_clocks',
    'reset_gripper_gap_different_clocks', 'first_close_policy_delta', 'first_close_pre_clock_delta',
    'first_open_pre_clock_delta', 'first_close_run_length_delta', 'close_fraction_delta', 'sign_switch_count_delta')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return legacy.sha256_file(Path(path))


def load_protocol(path):
    cfg = legacy._load_json(path)
    require(cfg.get('check_type') == 'libero_stabilization_public_trace_alignment' and
            cfg.get('formal_result') is False and cfg.get('scope') == SCOPE and cfg.get('analysis') == ANALYSIS,
            'trace alignment frozen scope or definitions differ')
    require(cfg['runner_sha256'] == sha(Path(__file__)), 'trace alignment runner differs')
    require(set(cfg['inputs']) == set(PINS), 'trace alignment input roles differ')
    paths = {}
    for name, pin in PINS.items():
        row = cfg['inputs'][name]; p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(row['sha256'] == pin and sha(p) == pin, f'trace input hash differs: {name}')
        paths[name] = p
    return cfg, paths


def norm(values):
    return math.sqrt(sum(float(x) ** 2 for x in values))


def gap(left, right):
    return norm([a - b for a, b in zip(left, right)])


def stats(values):
    values = list(values); present = [float(x) for x in values if x is not None]
    require(all(math.isfinite(x) for x in present), 'nonfinite summary value')
    return {'total': len(values), 'n': len(present), 'missing': len(values) - len(present),
            'mean': statistics.fmean(present) if present else None,
            'median': statistics.median(present) if present else None,
            'minimum': min(present) if present else None, 'maximum': max(present) if present else None}


def validate_episode(ep):
    n, arm = ep['steps'], ep['stabilization_steps']
    require(type(n) is int and 1 <= n <= 300 and type(arm) is int and arm in ARMS, 'invalid episode length or arm')
    require(type(ep['success']) is bool and ep['terminal_reason'] == ('success' if ep['success'] else 'time_limit') and
            (ep['success'] or n == 300), 'terminal result differs')
    require(ep['initial_state_role'] == 'candidate_nonregression', 'episode is not candidate data')
    trace = ep['stepwise_public_state_trace']
    require(len(trace) == n + 1 and all(type(r['step']) is int for r in trace) and
            [r['step'] for r in trace] == list(range(n + 1)), 'public trace steps are incomplete')
    for i, row in enumerate(trace):
        legacy._finite_vector(row['joint_state'], 7, 'joint_state')
        legacy._finite_vector(row['gripper_state'], 2, 'gripper_state')
        if i:
            action = legacy._finite_vector(row['action'], 7, 'action')
            require(all(abs(x) <= 1 for x in action), 'action outside interface')


def index_source(source, audit, config):
    require(source['status'] == audit['status'] == 'complete' and source['failure'] is None and
            source['formal_result'] is False and source['check_type'] == 'libero_task4_task7_stabilization_contemporary_closed_loop',
            'source is not the completed current paired diagnostic')
    require(source['scope'] == config['scope'] and source['provenance']['config_sha256'] == PINS['source_config'] and
            audit['result_sha256'] == PINS['source'] and audit['all_primary_episodes_fresh'] and
            audit['both_failed_historical_experiments_preserved'], 'source provenance or fresh boundary differs')
    scope = source['scope']
    require(scope['checkpoint_updates'] == 500 and scope['initial_state_indices'] == list(range(20)) and
            scope['initial_state_role'] == 'candidate_nonregression' and scope['reused_primary_episodes'] == 0,
            'source Policy or initial-state role differs')
    for key in ('qualification_accessed', 'reserved_accessed', 'validation_accessed', 'heldout_accessed', 'final_evaluation_accessed'):
        require(scope[key] is False, f'source sealed boundary differs: {key}')
    require(len(source['episodes']) == len(source['episode_files']) == 240, 'source episode count differs')
    indexed = {}
    for ep in source['episodes']:
        key = (ep['task_id'], ep['training_seed'], ep['initial_state_index'], ep['stabilization_steps'])
        require(all(type(x) is int for x in key) and key not in indexed, 'duplicate or noninteger episode key')
        validate_episode(ep); indexed[key] = ep
    require(set(indexed) == {(t, s, i, a) for t in TASKS for s in SEEDS for i in range(20) for a in ARMS}, 'source paired matrix differs')
    require(sum(ep['steps'] for ep in indexed.values()) == 70188 and
            sum(len(ep['stepwise_public_state_trace']) for ep in indexed.values()) == 70428, 'source public row budget differs')
    require(all(sum(int(e['success']) for e in indexed.values() if e['stabilization_steps'] == a) == 7 for a in ARMS), 'source outcome count differs')
    return indexed


def command_runs(actions):
    runs = []
    for n, action in enumerate(actions, 1):
        sign = legacy._sign(action[6])
        if not runs or runs[-1]['sign'] != sign:
            runs.append({'sign': sign, 'start_policy_step': n, 'end_policy_step': n, 'length': 1})
        else:
            runs[-1]['end_policy_step'] = n; runs[-1]['length'] += 1
    return runs


def summarize_episode(ep):
    actions = [r['action'] for r in ep['stepwise_public_state_trace'][1:]]
    old = legacy._action_summary(actions)
    positive = [r for r in command_runs(actions) if r['sign'] > 0]
    first = positive[0] if positive else None
    close, opened, arm = old['first_close_step'], old['first_release_step'], ep['stabilization_steps']
    return {'task_id': ep['task_id'], 'training_seed': ep['training_seed'], 'initial_state_index': ep['initial_state_index'],
        'stabilization_steps': arm, 'success': ep['success'], 'outcome': ep['terminal_reason'], 'policy_steps': ep['steps'],
        'first_command_pre_clock': arm, 'observed_end_clock': arm + ep['steps'], 'first_action': old['first_action'],
        'first_action_norm': old['first_action_norm'], 'first_action_translation_norm': norm(actions[0][:3]),
        'first_action_rotation_norm': norm(actions[0][3:6]), 'first_action_gripper': old['first_action_gripper'],
        'first_action_gripper_sign': old['first_action_gripper_sign'], 'first_close_policy_step': close,
        'first_close_pre_clock': arm + close - 1 if close is not None else None,
        'first_open_after_close_policy_step': opened,
        'first_open_after_close_pre_clock': arm + opened - 1 if opened is not None else None,
        'first_close_not_observed_censored': close is None, 'first_open_after_close_not_observed': opened is None,
        'first_close_run_length': first['length'] if first else None,
        'first_close_run_right_censored': first['end_policy_step'] == ep['steps'] if first else None,
        'longest_close_run': old['longest_contiguous_close_run'], 'close_step_count': old['close_step_count'],
        'close_fraction': old['close_step_count'] / ep['steps'], 'gripper_sign_switches': old['gripper_sign_switches']}


def state_at(ep, tick):
    index = tick - ep['stabilization_steps']
    return ep['stepwise_public_state_trace'][index] if 0 <= index <= ep['steps'] else None


def action_at(ep, tick):
    index = tick - ep['stabilization_steps']
    return ep['stepwise_public_state_trace'][index + 1]['action'] if 0 <= index < ep['steps'] else None


def common_point(zero, five, tick):
    z, f, z5, f5 = state_at(zero, tick), state_at(five, tick), state_at(zero, 5), state_at(five, 5)
    za, fa = action_at(zero, tick), action_at(five, tick)
    row = {'physical_tick': tick, 'zero_state_available': z is not None, 'five_state_available': f is not None,
           'joint_gap_l2': gap(z['joint_state'], f['joint_state']) if z is not None and f is not None else None,
           'gripper_gap_l2': gap(z['gripper_state'], f['gripper_state']) if z is not None and f is not None else None,
           'action_gap_l2_at_tick': gap(za, fa) if za is not None and fa is not None else None,
           'zero_gripper_command_at_tick': za[6] if za is not None else None,
           'five_gripper_command_at_tick': fa[6] if fa is not None else None}
    for name, current, initial in (('zero', z, z5), ('five', f, f5)):
        for key in ('joint', 'gripper'):
            row[f'{name}_{key}_motion_l2_from_common5'] = gap(current[f'{key}_state'], initial[f'{key}_state']) if current is not None and initial is not None else None
    return row


def delta(zero, five):
    return zero - five if zero is not None and five is not None else None


def pair_outcome(zero, five):
    return 'both_success' if zero and five else 'zero_only_success' if zero else 'five_only_success' if five else 'both_failure'


def pair_row(zero, five, z, f):
    za, fa = z['first_action'], f['first_action']
    cz, cf = z['first_close_pre_clock'], f['first_close_pre_clock']
    relation = ('both_not_observed' if cz is None and cf is None else 'only_five_observed' if cz is None else
                'only_zero_observed' if cf is None else 'zero_earlier' if cz < cf else 'five_earlier' if cz > cf else 'same_clock')
    return {'task_id': z['task_id'], 'training_seed': z['training_seed'], 'initial_state_index': z['initial_state_index'],
        'pair_outcome': pair_outcome(z['success'], f['success']), 'zero': z, 'five': f,
        'first_action_l2_gap': gap(za, fa), 'first_translation_l2_gap': gap(za[:3], fa[:3]),
        'first_rotation_l2_gap': gap(za[3:6], fa[3:6]), 'first_gripper_abs_gap': abs(za[6] - fa[6]),
        'first_gripper_zero_minus_five': za[6] - fa[6], 'first_gripper_sign_same': z['first_action_gripper_sign'] == f['first_action_gripper_sign'],
        'reset_joint_gap_different_clocks': gap(zero['stepwise_public_state_trace'][0]['joint_state'], five['stepwise_public_state_trace'][0]['joint_state']),
        'reset_gripper_gap_different_clocks': gap(zero['stepwise_public_state_trace'][0]['gripper_state'], five['stepwise_public_state_trace'][0]['gripper_state']),
        'first_close_policy_delta': delta(z['first_close_policy_step'], f['first_close_policy_step']),
        'first_close_pre_clock_delta': delta(cz, cf), 'first_close_relation_physical_clock': relation,
        'first_open_pre_clock_delta': delta(z['first_open_after_close_pre_clock'], f['first_open_after_close_pre_clock']),
        'first_close_run_length_delta': delta(z['first_close_run_length'], f['first_close_run_length']),
        'close_fraction_delta': z['close_fraction'] - f['close_fraction'],
        'sign_switch_count_delta': z['gripper_sign_switches'] - f['gripper_sign_switches'],
        'common_clock': {str(t): common_point(zero, five, t) for t in ANALYSIS['common_physical_ticks']}}


def episode_group(rows, dimensions):
    return {**dimensions, 'episodes': len(rows), 'successes': sum(int(r['success']) for r in rows),
            'no_close': sum(r['first_close_policy_step'] is None for r in rows),
            'no_open_after_close': sum(r['first_open_after_close_policy_step'] is None for r in rows),
            'no_open_after_observed_close': sum(r['first_close_policy_step'] is not None and
                r['first_open_after_close_policy_step'] is None for r in rows),
            'first_close_run_right_censored': sum(r['first_close_run_right_censored'] is True for r in rows),
            'first_action_sign_counts': dict(Counter(str(r['first_action_gripper_sign']) for r in rows)),
            'metrics': {m: stats(r[m] for r in rows) for m in EP_METRICS}}


def pair_group(rows, dimensions):
    clocks = {}
    for tick in ANALYSIS['common_physical_ticks']:
        keys = ('joint_gap_l2', 'gripper_gap_l2', 'action_gap_l2_at_tick', 'zero_joint_motion_l2_from_common5',
                'five_joint_motion_l2_from_common5', 'zero_gripper_motion_l2_from_common5', 'five_gripper_motion_l2_from_common5')
        clocks[str(tick)] = {k: stats(r['common_clock'][str(tick)][k] for r in rows) for k in keys}
    return {**dimensions, 'pairs': len(rows), 'outcome_counts': dict(Counter(r['pair_outcome'] for r in rows)),
        'first_gripper_sign_same_pairs': sum(r['first_gripper_sign_same'] for r in rows),
        'first_close_physical_relation_counts': dict(Counter(r['first_close_relation_physical_clock'] for r in rows)),
        'metrics': {m: stats(r[m] for r in rows) for m in PAIR_METRICS}, 'common_clock': clocks}


def analyze(indexed):
    episodes = {key: summarize_episode(ep) for key, ep in indexed.items()}
    pairs = [pair_row(indexed[(t, s, i, 0)], indexed[(t, s, i, 5)], episodes[(t, s, i, 0)], episodes[(t, s, i, 5)])
             for t in TASKS for s in SEEDS for i in range(20)]
    discordant = [r for r in pairs if r['pair_outcome'] in ('zero_only_success', 'five_only_success')]
    require([(r['task_id'], r['training_seed'], r['initial_state_index'], r['pair_outcome']) for r in discordant] == DISCORDANT,
            'six discordant source pairs differ')
    require(dict(Counter(r['pair_outcome'] for r in pairs)) ==
            {'both_success': 4, 'zero_only_success': 3, 'five_only_success': 3, 'both_failure': 110}, 'paired source outcomes differ')
    groups = [pair_group(pairs, {'group': 'all'})]
    for task in TASKS:
        rows = [r for r in pairs if r['task_id'] == task]; groups.append(pair_group(rows, {'group': 'task', 'task_id': task}))
        for seed in SEEDS:
            groups.append(pair_group([r for r in rows if r['training_seed'] == seed], {'group': 'task_seed', 'task_id': task, 'training_seed': seed}))
        for outcome in OUTCOMES:
            groups.append(pair_group([r for r in rows if r['pair_outcome'] == outcome], {'group': 'task_outcome', 'task_id': task, 'pair_outcome': outcome}))
    for outcome in OUTCOMES:
        groups.append(pair_group([r for r in pairs if r['pair_outcome'] == outcome], {'group': 'outcome', 'pair_outcome': outcome}))
    ep_groups = []
    values = list(episodes.values())
    for arm in ARMS:
        rows = [r for r in values if r['stabilization_steps'] == arm]
        ep_groups.append(episode_group(rows, {'group': 'arm', 'stabilization_steps': arm}))
        for outcome in ('success', 'time_limit'):
            ep_groups.append(episode_group([r for r in rows if r['outcome'] == outcome],
                {'group': 'arm_outcome', 'stabilization_steps': arm, 'outcome': outcome}))
        for task in TASKS:
            tr = [r for r in rows if r['task_id'] == task]
            ep_groups.append(episode_group(tr, {'group': 'arm_task', 'stabilization_steps': arm, 'task_id': task}))
            for seed in SEEDS:
                sr = [r for r in tr if r['training_seed'] == seed]
                ep_groups.append(episode_group(sr, {'group': 'arm_task_seed', 'stabilization_steps': arm, 'task_id': task, 'training_seed': seed}))
                for outcome in ('success', 'time_limit'):
                    ep_groups.append(episode_group([r for r in sr if r['outcome'] == outcome],
                        {'group': 'arm_task_seed_outcome', 'stabilization_steps': arm, 'task_id': task, 'training_seed': seed, 'outcome': outcome}))
            for outcome in ('success', 'time_limit'):
                ep_groups.append(episode_group([r for r in tr if r['outcome'] == outcome],
                    {'group': 'arm_task_outcome', 'stabilization_steps': arm, 'task_id': task, 'outcome': outcome}))
    return {'episode_rows': values, 'paired_rows': pairs, 'discordant_rows': discordant,
            'paired_groups': groups, 'episode_groups': ep_groups}


def run(config_path, output):
    require(not output.exists(), 'analysis output already exists')
    cfg, paths = load_protocol(config_path)
    started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    source, audit, source_cfg = [legacy._load_json(paths[n]) for n in ('source', 'source_audit', 'source_config')]
    indexed = index_source(source, audit, source_cfg)
    result = {'check_type': 'libero_stabilization_public_trace_alignment', 'status': 'complete', 'formal_result': False,
        'read_only_analysis': True, 'scope': SCOPE, 'analysis': ANALYSIS, **analyze(indexed),
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'inputs': cfg['inputs'], 'started_at_utc': utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'runtime': {'python': platform.python_version(), 'device': 'cpu_scalar', 'elapsed_seconds': time.perf_counter() - started,
            'public_rows_read': 70428, 'policy_action_rows_read': 70188, 'pairs_analyzed': 120,
            'new_policy_or_environment_or_optimizer_calls': 0, 'new_checkpoint_or_hdf5_or_initial_state_reads': 0},
        'interpretation': {'diagnostic_only': True, 'outcome_selected_six_not_population': True,
            'actions_have_no_expert_targets': True, 'commands_not_physical_grasp_or_release': True,
            'state_gaps_not_target_error': True, 'no_visual_or_causal_or_module_qualification_claim': True}}
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG); parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = run(args.config.resolve(), args.output.resolve())
    print(json.dumps({'status': result['status'], 'runtime': result['runtime'], 'discordant_pairs': len(result['discordant_rows'])}))


if __name__ == '__main__': main()
