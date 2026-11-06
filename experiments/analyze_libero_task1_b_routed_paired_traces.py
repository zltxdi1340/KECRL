"""Read existing paired public traces: shared startup, commands and state paths."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_source_internal_six_group_paired_traces as prior

require, sha, stats = prior.require, prior.sha, prior.stats
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('current_all7', 'history_motion_current_gripper')
OUTCOMES = ('baseline_only_success', 'candidate_only_success', 'both_success', 'both_failure')
TICKS = (1, 4, 5, 6, 10, 20, 50, 100, 200, 300)
PREFIX = 'libero-task1-b-routed-paired-traces'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-readonly_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
CONFIG_SHA = '8f32a921bc2c3bc5724df8cf2828600ef370d2291f5356df577fe7a2a4c2eb23'
FRAME_METRICS = ('action_gap_l2', 'translation_gap_l2', 'rotation_gap_l2', 'gripper_action_abs_gap',
    'baseline_action_norm', 'candidate_action_norm', 'baseline_translation_norm', 'candidate_translation_norm',
    'baseline_rotation_norm', 'candidate_rotation_norm', 'baseline_g', 'candidate_g',
    'pre_joint_gap_l2', 'pre_gripper_gap_l2', 'post_joint_gap_l2', 'post_gripper_gap_l2',
    'baseline_joint_offset_l2', 'candidate_joint_offset_l2', 'joint_offset_delta',
    'baseline_gripper_offset_l2', 'candidate_gripper_offset_l2', 'gripper_offset_delta')
WINDOW_METRICS = ('action_norm_mean', 'translation_norm_mean', 'rotation_norm_mean', 'close_steps', 'open_steps',
    'zero_steps', 'close_fraction', 'sign_switches', 'joint_path_l2', 'gripper_path_l2',
    'joint_offset_at_end_l2', 'gripper_offset_at_end_l2')
FIRST_DIFFERENCES = ('action', 'translation', 'rotation', 'gripper_value', 'gripper_sign', 'post_joint', 'post_gripper')
EVENT_ORDER_MAP = {'observed_left_earlier': 'candidate_earlier', 'observed_left_later': 'candidate_later',
    'observed_equal': 'observed_equal', 'both_not_observed': 'both_not_observed',
    'left_observed_reference_later_bound': 'candidate_observed_baseline_not_yet_within_baseline_horizon',
    'right_observed_left_later_bound': 'baseline_observed_candidate_not_yet_within_candidate_horizon',
    'indeterminate_reference_ended_before_left_event': 'baseline_ended_before_candidate_event_order_unknown',
    'indeterminate_left_ended_before_reference_event': 'candidate_ended_before_baseline_event_order_unknown'}


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load(path):
    value = json.loads(resolve(path).read_text())
    require(isinstance(value, dict), 'expected JSON object')
    return value


def write(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def vectors(ep):
    trace, n = ep['stepwise_public_state_trace'], ep['steps']
    require([r['step'] for r in trace] == list(range(n+1)), 'noncontiguous trace')
    a = np.asarray([r['action'] for r in trace[1:]], np.float64)
    j = np.asarray([r['joint_state'] for r in trace], np.float64)
    g = np.asarray([r['gripper_state'] for r in trace], np.float64)
    require(a.shape == (n, 7) and j.shape == (n+1, 7) and g.shape == (n+1, 2) and
            all(np.isfinite(v).all() for v in (a, j, g)) and (np.abs(a) <= 1).all(), 'invalid public trace')
    return a, j, g


def window_metrics(values, start, end):
    a, j, g = values
    require(1 <= start <= end <= len(a), 'invalid observed window')
    command = a[start-1:end]
    return {'action_norm_mean': float(np.linalg.norm(command, axis=1).mean()),
        'translation_norm_mean': float(np.linalg.norm(command[:, :3], axis=1).mean()),
        'rotation_norm_mean': float(np.linalg.norm(command[:, 3:6], axis=1).mean()),
        'close_steps': int((command[:, 6] > 0).sum()), 'open_steps': int((command[:, 6] < 0).sum()),
        'zero_steps': int((command[:, 6] == 0).sum()), 'close_fraction': float((command[:, 6] > 0).mean()),
        'sign_switches': int((np.sign(command[1:, 6]) != np.sign(command[:-1, 6])).sum()),
        'joint_path_l2': float(np.linalg.norm(np.diff(j[start-1:end+1], axis=0), axis=1).sum()),
        'gripper_path_l2': float(np.linalg.norm(np.diff(g[start-1:end+1], axis=0), axis=1).sum()),
        'joint_offset_at_end_l2': prior.norm(j[end]-j[0]), 'gripper_offset_at_end_l2': prior.norm(g[end]-g[0])}


def first_nonzero(mask):
    positions = np.flatnonzero(mask)
    return int(positions[0])+1 if len(positions) else None


def pair_features(baseline, candidate, bf=None, cf=None):
    require(baseline['task_id'] == candidate['task_id'] == 1 and
            baseline['arm'] == ARMS[0] and candidate['arm'] == ARMS[1] and
            baseline['training_seed'] == candidate['training_seed'] and
            baseline['initial_state_index'] == candidate['initial_state_index'] and
            baseline['reset_observation_sha256'] == candidate['reset_observation_sha256'], 'paired provenance differs')
    ba, bj, bg = bv = vectors(baseline)
    ca, cj, cg = cv = vectors(candidate)
    h = min(len(ba), len(ca))
    require(np.array_equal(ba[:min(4, h)], ca[:min(4, h)]) and
            np.array_equal(bj[:min(4, h)+1], cj[:min(4, h)+1]) and
            np.array_equal(bg[:min(4, h)+1], cg[:min(4, h)+1]), 'shared startup public state/action differs')
    bp, cp = baseline['shared_prefix_pre_observation_digests'], candidate['shared_prefix_pre_observation_digests']
    require(len(bp) == min(5, len(ba)) and len(cp) == min(5, len(ca)) and bp[:min(5, h)] == cp[:min(5, h)],
            'shared pre-action observation digests differ')
    bf = prior.episode_features(baseline) if bf is None else bf
    cf = prior.episode_features(candidate) if cf is None else cf
    series = []
    for i in range(h):
        t = i+1
        bo, co = prior.norm(bj[t]-bj[0]), prior.norm(cj[t]-cj[0])
        bgo, cgo = prior.norm(bg[t]-bg[0]), prior.norm(cg[t]-cg[0])
        row = {'step': t, 'certified_same_full_observation': True if t <= 5 else None,
            'action_gap_l2': prior.norm(ca[i]-ba[i]), 'translation_gap_l2': prior.norm(ca[i, :3]-ba[i, :3]),
            'rotation_gap_l2': prior.norm(ca[i, 3:6]-ba[i, 3:6]), 'gripper_action_abs_gap': abs(float(ca[i, 6]-ba[i, 6])),
            'baseline_action_norm': prior.norm(ba[i]), 'candidate_action_norm': prior.norm(ca[i]),
            'baseline_translation_norm': prior.norm(ba[i, :3]), 'candidate_translation_norm': prior.norm(ca[i, :3]),
            'baseline_rotation_norm': prior.norm(ba[i, 3:6]), 'candidate_rotation_norm': prior.norm(ca[i, 3:6]),
            'baseline_g': float(ba[i, 6]), 'candidate_g': float(ca[i, 6]),
            'gripper_sign_differs': bool(np.sign(ba[i, 6]) != np.sign(ca[i, 6])),
            'pre_joint_gap_l2': prior.norm(cj[i]-bj[i]), 'pre_gripper_gap_l2': prior.norm(cg[i]-bg[i]),
            'post_joint_gap_l2': prior.norm(cj[t]-bj[t]), 'post_gripper_gap_l2': prior.norm(cg[t]-bg[t]),
            'baseline_joint_offset_l2': bo, 'candidate_joint_offset_l2': co, 'joint_offset_delta': co-bo,
            'baseline_gripper_offset_l2': bgo, 'candidate_gripper_offset_l2': cgo, 'gripper_offset_delta': cgo-bgo}
        series.append(row)
    first = {'action': first_nonzero(np.any(ca[:h] != ba[:h], axis=1)),
        'translation': first_nonzero(np.any(ca[:h, :3] != ba[:h, :3], axis=1)),
        'rotation': first_nonzero(np.any(ca[:h, 3:6] != ba[:h, 3:6], axis=1)),
        'gripper_value': first_nonzero(ca[:h, 6] != ba[:h, 6]),
        'gripper_sign': first_nonzero(np.sign(ca[:h, 6]) != np.sign(ba[:h, 6])),
        'post_joint': first_nonzero(np.any(cj[1:h+1] != bj[1:h+1], axis=1)),
        'post_gripper': first_nonzero(np.any(cg[1:h+1] != bg[1:h+1], axis=1))}
    windows = {}
    for name, start, end in [('first20', 1, 20), ('after_shared_startup', 5, 20), ('common_horizon', 1, h)]:
        available = h >= end
        bm, cm = (window_metrics(bv, start, end), window_metrics(cv, start, end)) if available else (None, None)
        windows[name] = {'available': available, 'start': start, 'end': end, 'baseline': bm, 'candidate': cm,
                         'candidate_minus_baseline': {k: cm[k]-bm[k] for k in WINDOW_METRICS} if available else None}
    snapshots = {str(t): {'baseline_available': len(ba) >= t, 'candidate_available': len(ca) >= t,
        'both_available': h >= t, 'frame': series[t-1] if h >= t else None} for t in TICKS}
    event_comparisons = {}
    for name in ('first_close_step', 'first_release_step'):
        b, c = bf[name], cf[name]
        event_comparisons[name] = {'baseline': b, 'candidate': c, 'candidate_minus_baseline': None if b is None or c is None else c-b,
            'order': EVENT_ORDER_MAP[prior.event_order(c, b, len(ca), len(ba))]}
    bs, cs = baseline['success'], candidate['success']
    outcome = 'both_success' if bs and cs else 'baseline_only_success' if bs else 'candidate_only_success' if cs else 'both_failure'
    return {'pair_id': [baseline['training_seed'], baseline['initial_state_index']], 'outcome': outcome,
        'baseline_steps': len(ba), 'candidate_steps': len(ca), 'common_steps': h, 'first_different_steps': first,
        'step5': series[4] if h >= 5 else None, 'event_comparisons': event_comparisons, 'windows': windows,
        'snapshots': snapshots, 'common_stepwise_comparison': series,
        'baseline_episode_features': bf, 'candidate_episode_features': cf}


def frame_summary(frames):
    return {'n': len(frames), 'gripper_sign_differences': sum(r['gripper_sign_differs'] for r in frames),
            'metrics': {k: stats(r[k] for r in frames) for k in FRAME_METRICS}}


def summarize_pairs(rows):
    output = {'n': len(rows), 'outcomes': {k: sum(r['outcome'] == k for r in rows) for k in OUTCOMES},
        'pair_ids': [r['pair_id'] for r in rows], 'common_steps': stats(r['common_steps'] for r in rows),
        'first_different_steps': {k: stats(r['first_different_steps'][k] for r in rows) for k in FIRST_DIFFERENCES},
        'first_different_step_counts': {k: dict(Counter(str(r['first_different_steps'][k]) for r in rows)) for k in FIRST_DIFFERENCES},
        'step5': frame_summary([r['step5'] for r in rows if r['step5'] is not None]),
        'event_orders': {k: dict(Counter(r['event_comparisons'][k]['order'] for r in rows)) for k in ('first_close_step', 'first_release_step')},
        'event_time_deltas': {k: stats(r['event_comparisons'][k]['candidate_minus_baseline'] for r in rows) for k in ('first_close_step', 'first_release_step')},
        'windows': {}, 'snapshots': {}}
    for name in ('first20', 'after_shared_startup', 'common_horizon'):
        available = [r['windows'][name] for r in rows if r['windows'][name]['available']]
        output['windows'][name] = {'total': len(rows), 'available': len(available),
            'baseline': {k: stats(r['baseline'][k] for r in available) for k in WINDOW_METRICS},
            'candidate': {k: stats(r['candidate'][k] for r in available) for k in WINDOW_METRICS},
            'candidate_minus_baseline': {k: stats(r['candidate_minus_baseline'][k] for r in available) for k in WINDOW_METRICS}}
    for t in TICKS:
        frames = [r['snapshots'][str(t)]['frame'] for r in rows if r['snapshots'][str(t)]['both_available']]
        output['snapshots'][str(t)] = {'total': len(rows), 'missing': len(rows)-len(frames), **frame_summary(frames)}
    return output


def grouped_summary(rows):
    return {name: summarize_pairs(rows if name == 'all' else [r for r in rows if r['outcome'] == name])
            for name in ('all', *OUTCOMES)}


def validate_episode(ep, summary):
    require(all(ep.get(k) == v for k, v in summary.items()), 'parent episode summary differs')
    require(ep['status'] == 'complete' and type(ep['success']) is bool and ep['task_id'] == 1 and
            ep['training_seed'] in SEEDS and ep['arm'] in ARMS and ep['initial_state_index'] in range(20) and
            ep['initial_state_role'] == 'candidate_nonregression' and ep['stabilization_steps'] == 5 and
            ep['environment_seed'] == 20261022, 'episode role/provenance differs')
    require(type(ep['steps']) is int and 1 <= ep['steps'] <= 300 and
            ep['terminal_reason'] == ('success' if ep['success'] else 'time_limit') and
            (ep['success'] or ep['steps'] == 300), 'terminal boundary differs')
    a, j, g = vectors(ep)
    for t, row in enumerate(ep['stepwise_public_state_trace'][1:], 1):
        require(abs(prior.norm(j[t]-j[0])-row['joint_offset_l2_from_start']) <= 1e-10 and
                abs(prior.norm(g[t]-g[0])-row['gripper_offset_l2_from_start']) <= 1e-10 and
                np.allclose(j[t]-j[0], row['joint_offset_from_start'], atol=1e-10, rtol=0) and
                np.allclose(g[t]-g[0], row['gripper_offset_from_start'], atol=1e-10, rtol=0), 'public offset record differs')
        meta = row['history']
        require(meta['policy_step'] == t and meta['history_length'] == min(5, t) and meta['startup'] == (t <= 4) and
                meta['history_policy_steps'] == list(range(t, max(0, t-5), -1)) and
                meta['route'] == ('shared_baseline_startup' if t <= 4 else 'baseline_current' if ep['arm'] == ARMS[0]
                                  else 'candidate_motion_history_current_gripper'), 'recorded runtime route differs')


def validate_parent_summary(parent, index):
    all_outcomes = {k: 0 for k in OUTCOMES}
    for seed, old in zip(SEEDS, parent['summary']['seed_results']):
        require(old['training_seed'] == seed, 'parent seed order differs')
        totals = {arm: sum(index[seed, arm, i]['success'] for i in range(20)) for arm in ARMS}
        require(totals == old['success_counts'], 'parent success counts differ')
        outcomes = {k: 0 for k in OUTCOMES}
        for i in range(20):
            b, c = [index[seed, a, i]['success'] for a in ARMS]
            name = 'both_success' if b and c else 'baseline_only_success' if b else 'candidate_only_success' if c else 'both_failure'
            outcomes[name] += 1
        require(outcomes == old['paired_outcomes'], 'parent paired outcomes differ')
        for k, v in outcomes.items():
            all_outcomes[k] += v
    require(all_outcomes == parent['summary']['paired_outcomes'], 'parent pooled outcomes differ')
    require(parent['summary']['success_counts'] == {a: sum(index[s, a, i]['success'] for s in SEEDS for i in range(20)) for a in ARMS},
            'parent pooled success counts differ')


def preflight(config_path, binding_path):
    c = load(config_path)
    expected = {'task_id': 1, 'training_seeds': list(SEEDS), 'arms': list(ARMS), 'encoder_updates': 500,
        'readout_updates': 500, 'stabilization_steps': 5, 'environment_seed': 20261022,
        'initial_state_role': 'candidate_nonregression', 'initial_state_indices': list(range(20)),
        'unique_episodes': 120, 'public_trace_rows': 35609, 'saved_action_rows': 35489, 'paired_episodes': 60,
        'baseline_only_pairs': 3, 'both_failure_pairs': 57, 'candidate_only_pairs': 0, 'both_success_pairs': 0,
        'common_action_frame_pairs': 17489, 'first20_frame_pairs': 1200, 'step5_to20_frame_pairs': 960,
        'snapshot_policy_steps': list(TICKS), 'snapshot_records': 600, 'new_policy_encoder_readout_calls': 0,
        'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
        'new_HDF5_RGB_initial_private_checkpoint_reads': 0, 'new_feature_payload_reads': 0,
        'sealed_roles_accessed': False, 'checkpoint_selected': False, 'training_changed': False,
        'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c['formal_result'] is False, 'diagnostic only')
    for k, v in expected.items():
        require(c['scope'].get(k) == v and type(c['scope'].get(k)) is type(v), f'frozen scope differs: {k}')
    require(sha(resolve(config_path)) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'input pin differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
            set(b['files']) == {'analyzer', 'tests', 'protocol'}, 'binding differs')
    for pin in b['files'].values():
        require(sha(resolve(pin['path'])) == pin['sha256'], 'bound execution code differs')
    require(resolve(b['files']['analyzer']['path']).resolve() == Path(__file__).resolve(), 'analyzer binding path differs')
    for pin in load(c['inputs']['parent_binding']['path'])['files'].values():
        require(sha(resolve(pin['path'])) == pin['sha256'], 'parent bound code differs')
    return c, b


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT):
    output = Path(output)
    failure = output.with_name(output.stem+'-failed.json')
    require(not output.exists() and not failure.exists(), 'outputs already exist')
    counts = {k: 0 for k in ('episode_jsons_loaded', 'public_trace_rows', 'saved_action_rows', 'paired_episodes',
        'common_action_frame_pairs', 'first20_frame_pairs', 'step5_to20_frame_pairs', 'snapshot_records',
        'new_policy_encoder_readout_calls', 'new_optimizer_updates', 'new_backward_calls', 'new_environment_calls',
        'new_HDF5_RGB_initial_private_checkpoint_reads', 'new_feature_payload_reads')}
    start, stage, index, features, pairs = time.perf_counter(), 'preflight', {}, {}, []
    try:
        c, _b = preflight(config_path, binding_path)
        require(sys.version.split()[0] == c['execution']['python'] and np.__version__ == c['execution']['numpy'], 'runtime differs')
        parent, audit, parent_config, verification = [load(c['inputs'][k]['path']) for k in
            ('parent_result', 'parent_audit', 'parent_config', 'parent_verification')]
        require(parent['status'] == 'complete_routed_history_closed_loop' and parent['formal_result'] is False and
                audit['status'] == 'complete_independent_routed_history_closed_loop_audit' and
                audit['result_sha256'] == c['inputs']['parent_result']['sha256'] and
                parent['config_sha256'] == c['inputs']['parent_config']['sha256'] and
                parent['binding_sha256'] == c['inputs']['parent_binding']['sha256'] and
                verification['status'] == 'complete_routed_runtime_and_closed_loop_verification', 'parent provenance differs')
        require(parent['scope'] == parent_config['scope'] and parent_config['scope']['encoder_updates'] == 500 and
                parent_config['scope']['readout_updates'] == 500 and parent_config['scope']['stabilization_steps'] == 5 and
                all(parent_config['scope'][k] is False for k in ('qualification_accessed', 'reserved_accessed', 'validation_accessed',
                    'heldout_accessed', 'final_evaluation_accessed', 'checkpoint_selected', 'spt_updated', 'knowledge_updated', 'module_registered')),
                'parent roles or frozen endpoints differ')
        require(len(parent['episodes']) == len(parent['episode_files']) == 120, 'parent episode count differs')
        files = parent['episode_files']
        require(len({pin['path'] for pin in files}) == 120, 'duplicate episode paths')
        for pin in files:
            require(sha(resolve(pin['path'])) == pin['sha256'], 'episode hash differs before payload decode')
        for summary, pin in zip(parent['episodes'], files):
            stage = 'load_and_validate_public_episode'
            ep = load(pin['path'])
            counts['episode_jsons_loaded'] += 1
            validate_episode(ep, summary)
            key = (ep['training_seed'], ep['arm'], ep['initial_state_index'])
            require(key not in index, 'duplicate episode key')
            index[key], features[key] = ep, prior.episode_features(ep)
            features[key]['command_runs'] = prior.command_runs(np.asarray([r['action'] for r in ep['stepwise_public_state_trace'][1:]]))
            counts['public_trace_rows'] += ep['steps']+1
            counts['saved_action_rows'] += ep['steps']
        require(set(index) == {(s, a, i) for s in SEEDS for a in ARMS for i in range(20)}, 'missing episode key')
        validate_parent_summary(parent, index)
        for seed in SEEDS:
            for state in range(20):
                stage = f'pair_seed{seed}_state{state}'
                bkey, ckey = [(seed, arm, state) for arm in ARMS]
                pair = pair_features(index[bkey], index[ckey], features[bkey], features[ckey])
                pairs.append(pair)
                counts['paired_episodes'] += 1
                counts['common_action_frame_pairs'] += pair['common_steps']
                counts['first20_frame_pairs'] += 20 if pair['windows']['first20']['available'] else 0
                counts['step5_to20_frame_pairs'] += 16 if pair['windows']['after_shared_startup']['available'] else 0
                counts['snapshot_records'] += len(TICKS)
        require(all(counts[k] == c['scope'][k] for k in counts if k != 'episode_jsons_loaded') and counts['episode_jsons_loaded'] == 120,
                'analysis count budget differs')
        baseline_only = [r for r in pairs if r['outcome'] == 'baseline_only_success']
        require([r['pair_id'] for r in baseline_only] == c['definition']['baseline_only_cases'] and
                {name: sum(r['outcome'] == name for r in pairs) for name in OUTCOMES} == parent['summary']['paired_outcomes'],
                'outcome cases/denominators differ')
        seed_groups = [{'training_seed': seed, 'paired_groups': grouped_summary([p for p in pairs if p['pair_id'][0] == seed]),
            'episode_groups': {arm: {name: prior.summarize_episodes([f for k, f in features.items() if
                k[0] == seed and k[1] == arm and (name == 'all' or f['success'] == (name == 'success'))])
                for name in ('all', 'success', 'time_limit')} for arm in ARMS}} for seed in SEEDS]
        pooled = grouped_summary(pairs)
        for pin in c['inputs'].values():
            require(sha(resolve(pin['path'])) == pin['sha256'], 'input mutated')
        for pin in files:
            require(sha(resolve(pin['path'])) == pin['sha256'], 'episode source mutated')
        result = {'status': 'complete_readonly_routed_paired_traces', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(resolve(config_path)),
            'binding_sha256': sha(resolve(binding_path)), 'scope': c['scope'], 'definition': c['definition'],
            'input_provenance': c['inputs'], 'episode_input_pins': files, 'actual_counts': counts,
            'parent_summary_reproduced': parent['summary'], 'episode_features': list(features.values()), 'pair_results': pairs,
            'baseline_only_cases': [{k: v for k, v in p.items() if k != 'common_stepwise_comparison'} for p in baseline_only],
            'seed_results': seed_groups, 'pooled_results': pooled,
            'routed_candidate_stop_retained': True, 'automatic_training_change': False,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu',
                'entry_elapsed_seconds': time.perf_counter()-start, 'argv': sys.argv}}
        write(output, result)
        return result
    except Exception as exc:
        write(failure, {'status': 'failed_readonly_routed_paired_traces', 'formal_result': False, 'stage': stage,
            'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts,
            'completed_episode_keys': [list(k) for k in index], 'completed_pair_ids': [p['pair_id'] for p in pairs]})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = run(args.config, args.binding, args.output)
        print(json.dumps({'status': result['status'], 'actual_counts': result['actual_counts']}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'scope': c['scope']}))


if __name__ == '__main__':
    main()
