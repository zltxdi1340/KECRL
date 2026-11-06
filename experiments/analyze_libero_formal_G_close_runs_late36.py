"""Read only the72 saved trajectories belonging to all36 late-close winners.

Enumerate signs and observed run lengths; multi-step means at least two saved
positive commands, not a grasp qualification. Preserve terminal censoring and
compare both arms at the same absolute steps in fixed event neighborhoods.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import time

from experiments import analyze_libero_formal_G_validation_BC_traces as trace

ROOT = trace.ROOT
require, sha = trace.require, trace.sha
ARMS, TASKS, SEEDS = trace.ARMS, trace.TASKS, trace.SEEDS
ANCHORS = ['B_first_positive', 'C_first_positive', 'B_first_multi_step', 'C_first_multi_step']
WINDOW_METRICS = ['translation_gap_l2', 'rotation_gap_l2', 'gripper_action_abs_gap',
    'joint_gap_pre_l2', 'joint_gap_post_l2', 'gripper_gap_pre_l2', 'gripper_gap_post_l2',
    'B_gripper_command', 'C_gripper_command', 'B_command_sign', 'C_command_sign',
    'B_positive_run_age', 'C_positive_run_age', 'B_translation_norm', 'C_translation_norm',
    'B_joint_step_l2', 'C_joint_step_l2', 'B_gripper_step_l2', 'C_gripper_step_l2']


def norm(values):
    return math.sqrt(math.fsum(x*x for x in values))


def distance(a, b):
    return norm([x-y for x, y in zip(a, b)])


def load_csv(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def case_key(row):
    return int(row['task']), int(row['seed']), int(row['global_initial_index'])


def select_cases(config, docs):
    require(len(docs['prior_pairs']) == 800 and len(docs['prior_episodes']) == 1600, 'incomplete cached context')
    pairs = {case_key(r): r for r in docs['prior_pairs']}
    require(len(pairs) == 800, 'duplicate cached pair')
    selected = {k: r for k, r in pairs.items() if r['winner_close_order'] == 'observed_left_later'}
    counters = [r for r in docs['counterexamples'] if r['category'] == 'winner_closed_later']
    ids = [case_key(r) for r in counters]
    require(len(ids) == len(set(ids)) == len(selected) == 36 and
            set(ids) == set(selected) == {tuple(k) for k in config['case_ids']}, 'all36 late-close cases must be preserved')
    for row in counters:
        p = selected[case_key(row)]
        require(p['outcome'] in ('B_only', 'C_only') and p['outcome'] == row['outcome'] and
                row['winner'] == ('B' if p['outcome'] == 'B_only' else 'C') and
                float(row['B_first_close_step']) == float(p['B_first_close_step']) and
                float(row['C_first_close_step']) == float(p['C_first_close_step']), 'counterexample identity/timing differs')
    return selected


def preflight(config_path, binding_path):
    config, binding = (json.loads(Path(p).read_text()) for p in (config_path, binding_path))
    require(binding['config_sha256'] == sha(config_path), 'close-run definition changed')
    require(config['diagnostic_only'] is True and config['prior_outcomes_and_first_close_order_known'] is True and
            config['tasks'] == TASKS and config['seeds'] == SEEDS and config['arms'] == ARMS and
            config['initial_state_role'] == 'candidate_acceptance' and config['global_initial_indices'] == trace.INDICES and
            Path(config['trace_root']) == trace.TRACE_ROOT and config['neighborhood_half_window'] == 5 and
            config['anchor_kinds'] == ANCHORS, 'selected validation/event scope differs')
    json_names = {'result', 'training', 'completion', 'complete_audit', 'prior_trace_analysis', 'prior_trace_summary'}
    csv_names = {'counterexamples', 'prior_pairs', 'prior_episodes'}
    require(set(config['inputs']) == json_names | csv_names and
            {'runner', 'tests', 'trace_helpers', 'command_metrics', 'public_validator'}.issubset(binding['files']),
            'metadata or dependency scope differs')
    for name, pin in {**config['inputs'], **binding['files']}.items():
        p = Path(pin['path']); p = p if p.is_absolute() else ROOT/p
        require(sha(p) == pin['sha256'], f'input/dependency hash differs: {name}')
    require((ROOT/binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'wrong close-run entry')
    docs = {}
    for name, pin in config['inputs'].items():
        p = Path(pin['path']); require(p.suffix == ('.csv' if name in csv_names else '.json'), 'unexpected input format')
        docs[name] = load_csv(p) if name in csv_names else json.loads(p.read_text())
    done, raw, audited, old = (docs[k] for k in ('completion', 'result', 'complete_audit', 'prior_trace_analysis'))
    require(done['status'] == 'complete_audited_G' and done['decision']['passed'] is False and
            audited['status'] == 'passed_complete_audit' and audited['stage'] == 'G' and
            done['decision'] == audited['decision'] == old['G_decision_unchanged'], 'negative G lineage missing')
    require(done['result']['sha256'] == audited['result_sha256'] == config['inputs']['result']['sha256'] and
            done['audit']['sha256'] == config['inputs']['complete_audit']['sha256'] and
            raw['training']['sha256'] == config['inputs']['training']['sha256'], 'G input lineage differs')
    for name, prior_name in [('prior_pairs', 'pairs'), ('prior_episodes', 'episodes')]:
        require(config['inputs'][name]['sha256'] == old['csv_artifacts'][prior_name]['sha256'], 'cached context changed')
    require(config['inputs']['counterexamples']['sha256'] == docs['prior_trace_summary']['counterexamples']['sha256'],
            'counterexample table changed')
    cases = select_cases(config, docs)
    selected, models = trace.select_metadata(raw, docs['training'])
    subset = {key: value for key, value in selected.items() if (key[0], key[1], key[3]) in cases}
    require(len(subset) == 72, 'selected trace set incomplete')
    expected = {'metadata_JSON_inputs': 6, 'metadata_CSV_inputs': 3, 'selected_pairs': 36,
        'unique_episode_trace_files': 72, 'policy_action_rows': sum(r['steps'] for r in subset.values()),
        'public_state_rows': sum(r['steps']+1 for r in subset.values()),
        'maximum_sign_run_rows': sum(r['steps'] for r in subset.values()), 'maximum_neighborhood_anchors': 144,
        'maximum_neighborhood_rows': 1584, 'checkpoint_files': 0, 'HDF5_files': 0, 'RGB_files': 0,
        'official_initial_container_reads': 0, 'new_policy_forwards': 0, 'new_optimizer_updates': 0,
        'new_environment_steps': 0, 'heldout_payload_reads': 0}
    require(config['read_budget'] == expected, 'selected read budget differs')
    return config, docs, cases, subset, models


def command_segments(ep):
    states = ep['stepwise_public_state_trace']; actions = [r['action'] for r in states[1:]]
    runs = trace.metrics.command_runs(actions); output = []; close_ordinal = 0
    for ordinal, item in enumerate(runs, 1):
        start, end, sign = item['start'], item['end'], item['sign']
        if sign > 0: close_ordinal += 1
        values = [a[6] for a in actions[start-1:end]]
        output.append({**item, 'run_ordinal': ordinal, 'positive_run_ordinal': close_ordinal if sign > 0 else None,
            'right_censored': end == ep['steps'], 'previous_sign': runs[ordinal-2]['sign'] if ordinal > 1 else None,
            'next_sign': runs[ordinal]['sign'] if ordinal < len(runs) else None,
            'gripper_command_min': min(values), 'gripper_command_max': max(values), 'gripper_command_mean': math.fsum(values)/len(values),
            'joint_path': math.fsum(distance(states[t]['joint_state'], states[t-1]['joint_state']) for t in range(start, end+1)),
            'gripper_path': math.fsum(distance(states[t]['gripper_state'], states[t-1]['gripper_state']) for t in range(start, end+1)),
            'gripper_displacement': distance(states[end]['gripper_state'], states[start-1]['gripper_state'])})
    return output


def first_close_kind(first):
    if first is None: return 'not_observed'
    return ('one_step' if first['length'] == 1 else 'multi_step') + (
        '_right_censored' if first['right_censored'] else '_completed')


def episode_features(ep, runs):
    positive = [r for r in runs if r['sign'] > 0]
    first = positive[0] if positive else None
    multi = next((r for r in positive if r['length'] >= 2), None)
    return {'steps': ep['steps'], 'success': ep['success'], 'positive_runs': len(positive),
        'first_positive_start': None if first is None else first['start'],
        'first_positive_observed_length': None if first is None else first['length'],
        'first_positive_right_censored': None if first is None else first['right_censored'],
        'first_positive_kind': first_close_kind(first),
        'first_multi_step_start': None if multi is None else multi['start'],
        'first_multi_step_observed_length': None if multi is None else multi['length'],
        'first_multi_step_right_censored': None if multi is None else multi['right_censored'],
        'positive_command_steps': sum(r['length'] for r in positive)}


def command_at(ep, runs, step):
    if not 1 <= step <= ep['steps']: return None
    run = next(r for r in runs if r['start'] <= step <= r['end'])
    return {'value': ep['stepwise_public_state_trace'][step]['action'][6], 'sign': run['sign'],
            'positive_age': step-run['start']+1 if run['sign'] > 0 else None,
            'positive_ordinal': run['positive_run_ordinal']}


def event_windows(left, right, lf, rf, lruns, rruns):
    anchors = {}
    for name, value in zip(ANCHORS, [lf['first_positive_start'], rf['first_positive_start'],
                                   lf['first_multi_step_start'], rf['first_multi_step_start']]):
        if value is not None: anchors.setdefault(value, []).append(name)
    output = []
    for anchor, names in sorted(anchors.items()):
        for offset in range(-5, 6):
            step = anchor+offset; available = [1 <= step <= ep['steps'] for ep in (left, right)]
            row = {'anchor_step': anchor, 'anchor_aliases': ';'.join(names), 'offset': offset, 'absolute_step': step,
                'window_phase': 'pre' if offset < 0 else 'post' if offset > 0 else 'event',
                'B_available': available[0], 'C_available': available[1], 'both_available': all(available),
                'reset_side_truncated': step < 1, 'B_terminal_truncated': step > left['steps'],
                'C_terminal_truncated': step > right['steps'], **dict.fromkeys(WINDOW_METRICS)}
            for name, ep, runs, ok in zip(('B', 'C'), (left, right), (lruns, rruns), available):
                if not ok: continue
                previous, current = ep['stepwise_public_state_trace'][step-1:step+1]
                action = current['action']; command = command_at(ep, runs, step)
                row.update({name+'_gripper_command': action[6], name+'_command_sign': command['sign'],
                    name+'_positive_run_age': command['positive_age'], name+'_translation_norm': norm(action[:3]),
                    name+'_joint_step_l2': distance(previous['joint_state'], current['joint_state']),
                    name+'_gripper_step_l2': distance(previous['gripper_state'], current['gripper_state'])})
            if all(available):
                lp, l = left['stepwise_public_state_trace'][step-1:step+1]
                rp, r = right['stepwise_public_state_trace'][step-1:step+1]
                row.update(translation_gap_l2=distance(l['action'][:3], r['action'][:3]),
                    rotation_gap_l2=distance(l['action'][3:6], r['action'][3:6]), gripper_action_abs_gap=abs(l['action'][6]-r['action'][6]),
                    joint_gap_pre_l2=distance(lp['joint_state'], rp['joint_state']), joint_gap_post_l2=distance(l['joint_state'], r['joint_state']),
                    gripper_gap_pre_l2=distance(lp['gripper_state'], rp['gripper_state']), gripper_gap_post_l2=distance(l['gripper_state'], r['gripper_state']))
            output.append(row)
    return output, len(anchors)


def analyze_case(left, right, prior):
    require(left['task_id'] in TASKS and left['task_id'] == right['task_id'] and left['training_seed'] in SEEDS and left['training_seed'] == right['training_seed'] and
            left['initial_state_index'] == right['initial_state_index'] and left['arm'] == ARMS[0] and right['arm'] == ARMS[1],
            'case pairing differs')
    require(left['initial_state_index'] in trace.INDICES and
            left['initial_state_role'] == right['initial_state_role'] == 'candidate_acceptance', 'case role/index differs')
    require(left['success'] != right['success'], 'case must have exactly one winner')
    require(left['reset_observation_sha256'] == right['reset_observation_sha256'] and
            all(left['stepwise_public_state_trace'][0][k] == right['stepwise_public_state_trace'][0][k]
                for k in ('joint_state', 'gripper_state')), 'paired saved reset differs')
    lruns, rruns = command_segments(left), command_segments(right)
    lf, rf = episode_features(left, lruns), episode_features(right, rruns)
    require(lf['first_positive_start'] is not None and rf['first_positive_start'] is not None, 'original case must have both close events')
    require(float(prior['B_first_close_step']) == lf['first_positive_start'] and
            float(prior['C_first_close_step']) == rf['first_positive_start'], 'original first-positive event changed')
    winner = 'B' if left['success'] else 'C'; wf, losef, wruns, loseruns = (
        (lf, rf, lruns, rruns) if winner == 'B' else (rf, lf, rruns, lruns))
    require(wf['first_positive_start'] > losef['first_positive_start'], 'original late-close counterexample lost')
    first_loser = next(r for r in loseruns if r['sign'] > 0)
    loser_ep = right if winner == 'B' else left
    loser_at_winner = command_at(loser_ep, loseruns, wf['first_positive_start'])
    first_run_ongoing = first_loser['start'] <= wf['first_positive_start'] <= first_loser['end']
    row = {'task': left['task_id'], 'seed': left['training_seed'], 'global_initial_index': left['initial_state_index'],
        'winner': winner, 'outcome': 'B_only' if winner == 'B' else 'C_only',
        'original_first_positive_winner_later_preserved': True,
        **{f'{side}_{k}': v for side, f in [('B', lf), ('C', rf)] for k, v in f.items()},
        'winner_first_positive_kind': wf['first_positive_kind'], 'loser_first_positive_kind': losef['first_positive_kind'],
        'loser_first_positive_run_still_active_at_winner_first_positive': first_run_ongoing,
        'loser_first_run_relation': 'ongoing_at_winner_first_positive' if first_run_ongoing else
            'completed_one_step_before_winner' if first_loser['length'] == 1 else 'completed_multi_step_before_winner',
        'loser_sign_at_winner_first_positive': None if loser_at_winner is None else loser_at_winner['sign'],
        'loser_positive_age_at_winner_first_positive': None if loser_at_winner is None else loser_at_winner['positive_age'],
        'winner_vs_loser_first_multi_order': trace.metrics.event_order(wf['first_multi_step_start'], losef['first_multi_step_start'], wf['steps'], losef['steps'])}
    windows, anchors = event_windows(left, right, lf, rf, lruns, rruns)
    run_rows = [{**{k: row[k] for k in ('task', 'seed', 'global_initial_index', 'winner', 'outcome')}, 'arm': ep['arm'], **run}
                for ep, runs in [(left, lruns), (right, rruns)] for run in runs]
    window_rows = [{**{k: row[k] for k in ('task', 'seed', 'global_initial_index', 'winner', 'outcome')}, **window} for window in windows]
    return row, run_rows, window_rows, anchors


def summarize(cases):
    groups = []
    for task in [None]+TASKS:
        group = [r for r in cases if task is None or r['task'] == task]
        groups.append({'task': task, 'pairs': len(group), 'seeds_present': len({r['seed'] for r in group}),
            'winners': dict(Counter(r['winner'] for r in group)),
            'winner_first_positive_kinds': dict(Counter(r['winner_first_positive_kind'] for r in group)),
            'loser_first_positive_kinds': dict(Counter(r['loser_first_positive_kind'] for r in group)),
            'loser_first_run_relations': dict(Counter(r['loser_first_run_relation'] for r in group)),
            'first_multi_orders': dict(Counter(r['winner_vs_loser_first_multi_order'] for r in group)),
            'loser_sign_at_winner_first_positive': dict(Counter(str(r['loser_sign_at_winner_first_positive']) for r in group))})
    return groups


def run(config_path, binding_path, output, artifacts):
    require(not output.exists() and not artifacts.exists(), 'close-run outputs must be new')
    require(not output.resolve().is_relative_to(ROOT) and not artifacts.resolve().is_relative_to(ROOT), 'artifacts must remain outside repository')
    config, docs, cases, subset, models = preflight(config_path, binding_path)
    started = time.monotonic(); artifacts.mkdir(parents=True, exist_ok=False)
    access = {'trace_read_attempts': 0, 'trace_files_validated': 0, 'policy_action_rows': 0, 'public_state_rows': 0,
              'pairs_completed': 0, 'sign_run_rows': 0, 'neighborhood_anchors': 0, 'neighborhood_rows': 0}
    case_rows = []; run_rows = []; window_rows = []; pins = []
    def load(key):
        descriptor = subset[key]; access['trace_read_attempts'] += 1
        raw = Path(descriptor['path']).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == descriptor['sha256'], 'selected trace checksum differs')
        ep = json.loads(raw)
        require(all(ep[k] == descriptor[k] for k in ('task_id', 'training_seed', 'arm', 'initial_state_index',
                'local_initial_state_index', 'success', 'steps', 'terminal_reason', 'initial_state_role', 'reset_observation_sha256')),
                'selected descriptor/trace identity differs')
        n = trace.validate_episode(ep, models[key[:3]])
        access['trace_files_validated'] += 1; access['policy_action_rows'] += n; access['public_state_rows'] += n+1
        pins.append({'path': descriptor['path'], 'sha256': descriptor['sha256']})
        return ep
    try:
        for task, seed, index in sorted(cases):
            b, c = (load((task, seed, arm, index)) for arm in ARMS)
            row, runs, windows, anchors = analyze_case(b, c, cases[task, seed, index])
            case_rows.append(row); run_rows.extend(runs); window_rows.extend(windows)
            access['pairs_completed'] += 1; access['sign_run_rows'] += len(runs)
            access['neighborhood_anchors'] += anchors; access['neighborhood_rows'] += len(windows)
        budget = config['read_budget']
        require(access['trace_read_attempts'] == access['trace_files_validated'] == 72 and access['pairs_completed'] == 36 and
            access['policy_action_rows'] == budget['policy_action_rows'] and access['public_state_rows'] == budget['public_state_rows'] and
            access['sign_run_rows'] <= budget['maximum_sign_run_rows'] and access['neighborhood_anchors'] <= 144 and
            access['neighborhood_rows'] <= 1584, 'close-run read budget differs')
        csv_pins = {}
        for name, rows in [('cases', case_rows), ('command_runs', run_rows), ('neighborhoods', window_rows)]:
            p = artifacts/f'{name}.csv'
            with p.open('x', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
            csv_pins[name] = {'path': str(p), 'sha256': sha(p), 'rows': len(rows)}
        context = docs['prior_episodes']
        result = {'status': 'complete_readonly_close_runs_late36_diagnostic', 'formal_result': False,
            'case_rows': case_rows, 'groups': summarize(case_rows), 'actual_access': access, 'csv_artifacts': csv_pins,
            'cached_context': {'rows': len(context), 'observed_completed_one_step_first_runs': sum(
                r['first_close_run_length'] == '1' and r['first_close_run_right_censored'] == 'False' for r in context),
                'selected_raw_episodes_only': 72, 'other_raw_episodes_opened': 0},
            'event_window_availability': {'both_available_rows': sum(r['both_available'] for r in window_rows),
                'reset_side_truncated_rows': sum(r['reset_side_truncated'] for r in window_rows),
                'B_terminal_truncated_rows': sum(r['B_terminal_truncated'] for r in window_rows),
                'C_terminal_truncated_rows': sum(r['C_terminal_truncated'] for r in window_rows)},
            'G_decision_unchanged': docs['completion']['decision'], 'episode_file_pins': pins,
            'forbidden_access': {k: 0 for k in ('checkpoint_files', 'HDF5_files', 'RGB_files', 'official_initial_container_reads',
                'new_policy_forwards', 'new_optimizer_updates', 'new_environment_steps', 'heldout_payload_reads')},
            'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'binding_path': str(binding_path),
                'binding_sha256': sha(binding_path), 'runner_sha256': sha(Path(__file__)), 'input_pins': config['inputs'],
                'python': platform.python_version(), 'elapsed_seconds': time.monotonic()-started,
                'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
            'limits': ['selected36 known-outcome/timing counterexamples, not independent causal evidence',
                'one/multi-step commands are not effective grasp labels; preserve all run lengths and censoring',
                'first-multi start is retrospective after observing its second positive action',
                'event relabeling does not erase the36 original first-positive counterexamples or change G admission']}
        with output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')
        return result
    except BaseException as error:
        with (artifacts/'failure.json').open('x') as handle:
            json.dump({'status': 'readonly_close_run_analysis_failed', 'error': repr(error), 'actual_access': access,
                       'G_decision_unchanged': True}, handle, indent=2); handle.write('\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True); parser.add_argument('--binding', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args(); result = run(args.config, args.binding, args.output, args.artifacts)
    print(json.dumps({'status': result['status'], 'actual_access': result['actual_access'], 'groups': result['groups'],
                      'elapsed_seconds': result['provenance']['elapsed_seconds']}))


if __name__ == '__main__':
    main()
