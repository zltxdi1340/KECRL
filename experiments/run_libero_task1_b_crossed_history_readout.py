"""Cross saved heads and cached features without fitting or running encoders."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import run_libero_task1_b_support_query_history_readout as cached
from experiments import analyze_libero_task1_b_query_event_margin_timing as event

require, sha, load, resolve = cached.require, cached.sha, cached.load, cached.resolve
SEEDS, MODES = cached.SEEDS, cached.MODES
PREFIX = 'libero-task1-b-crossed-history-readout'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_ARTIFACTS = Path(f'/tmp/{PREFIX}-artifacts-20261104-v1')
CONFIG_SHA = '2e596ba0bd3233f8e4b2d41fccc27f74c45438ad505556b630dfcfcd9e2496cc'


def linear_readout(x, weight, bias):
    require(x.ndim == 2 and x.shape[1] == 1280 and weight.shape == (7, 1280) and bias.shape == (7,) and
            all(np.isfinite(v).all() for v in (x, weight, bias)), 'readout interface differs')
    logits = x.astype(np.float64) @ weight.astype(np.float64).T + bias.astype(np.float64)
    return np.tanh(logits), logits


def decompose(current, crossed, history, tolerance):
    require(current.shape == crossed.shape == history.shape and np.isfinite(crossed).all(), 'invalid decomposition')
    weights, content, total = crossed - current, history - crossed, history - current
    error = float(np.max(np.abs(total - weights - content)))
    require(error <= tolerance, 'action decomposition differs')
    return weights, content, total, error


def preflight(config_path, binding_path):
    c = load(config_path)
    expected = {'task_id': 1, 'feature_arm': 'B_fomaml', 'training_seeds': list(SEEDS),
        'original_policy_updates': 500, 'readout_updates': 500, 'closed_loop_default_stabilization_steps': 5,
        'demonstration_role': 'meta_query', 'demonstration_indices': list(range(30, 50)), 'history_steps': 5,
        'common_first_action_step': 5, 'target_rows': 2598, 'common_rows_per_seed': 2518,
        'new_readout_batches': 3, 'new_readout_rows': 7554, 'arithmetic_replay_batches': 6,
        'arithmetic_replay_rows': 15108, 'model_event_pairs': 69, 'full_post_pairs': 696,
        'original_encoder_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
        'new_HDF5_RGB_initial_private_reads': 0, 'new_environment_calls': 0, 'sealed_roles_accessed': False,
        'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c['formal_result'] is False, 'diagnostic only')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
            set(b['files']) == {'runner', 'tests', 'protocol'}, 'binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound code differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner binding differs')
    for key in ('parent_binding', 'event_binding'):
        for pin in load(c['inputs'][key]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'historical bound code differs')
    return c, b


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failure = output.with_name(output.stem + '-failed.json')
    require(not any(p.exists() for p in (output, failure, artifacts)), 'outputs already exist')
    start = time.perf_counter()
    counts = {k: 0 for k in ('new_readout_batches', 'new_readout_rows', 'arithmetic_replay_batches',
        'arithmetic_replay_rows', 'model_event_pairs', 'full_post_pairs', 'original_encoder_calls',
        'new_optimizer_updates', 'new_backward_calls', 'new_HDF5_RGB_initial_private_reads', 'new_environment_calls')}
    results, events = [], []
    stage = 'preflight'
    try:
        c, _ = preflight(config_path, binding_path)
        require(sys.version.split()[0] == c['execution']['python'] and np.__version__ == c['execution']['numpy'], 'runtime differs')
        parent = load(c['inputs']['parent_result']['path'])
        previous = load(c['inputs']['event_result']['path'])
        require(parent['status'] == 'complete_task1_b_support_query_history_readout' and
                previous['status'] == 'complete_readonly_query_event_margin_timing' and
                parent['primary_query_comparison']['joint_query_cue'] is False, 'parent status/gate differs')
        with np.load(resolve(c['inputs']['query_targets']['path']), allow_pickle=False) as z:
            targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
        plan = cached.make_plan(targets, ids, steps)
        source = plan['eligible']
        require(len(targets) == 2598 and set(ids.tolist()) == set(range(30, 50)) and len(source) == 2518, 'query role/count differs')
        masks = {'all': np.ones(len(source), bool), **{k: v[source] for k, v in plan['masks'].items()}}
        require(int(masks['known_open_event'].sum()) == 23 and int(masks['known_open_segment'].sum()) == 232, 'event coverage differs')
        artifacts.mkdir(parents=True, exist_ok=False)
        for seed in SEEDS:
            stage = f'seed{seed}'
            with np.load(resolve(c['inputs'][f'{seed}_features']['path']), allow_pickle=False) as z:
                features = z['projection']
            saved, heads, scalers = {}, {}, []
            for mode in MODES:
                with np.load(resolve(c['inputs'][f'{seed}_{mode}_weights']['path']), allow_pickle=False) as z:
                    require(z['updates'].tolist() == [0, 100, 250, 500], 'fixed endpoint differs')
                    heads[mode] = (z['weights'][-1], z['biases'][-1])
                    scalers.append((z['scaler_mean'], z['scaler_scale']))
                with np.load(resolve(c['inputs'][f'{seed}_{mode}_predictions']['path']), allow_pickle=False) as z:
                    require(np.array_equal(z['source_rows'], source), 'prediction row identity differs')
                    saved[mode] = z['predictions']
            require(all(np.array_equal(a, b) for a, b in zip(*scalers)), 'support scalers differ')
            inputs = cached.transform(features, plan['history_rows'], *scalers[0])
            replay_errors, logits = {}, {}
            for mode in MODES:
                replay, logits[mode] = linear_readout(inputs[mode], *heads[mode])
                counts['arithmetic_replay_batches'] += 1
                counts['arithmetic_replay_rows'] += len(source)
                replay_errors[mode] = float(np.max(np.abs(replay - saved[mode])))
                require(replay_errors[mode] <= c['analysis']['existing_prediction_atol'], 'saved prediction replay differs')
                require(np.array_equal(np.sign(replay[:, 6]), np.sign(saved[mode][:, 6])), 'numerical replay changed gripper signs')
                expected = next(x for x in parent['seed_results'] if x['training_seed'] == seed)['query_metrics'][mode]
                event.validate_parent_metrics(saved[mode], targets[source], masks, expected)
            crossed, cross_logits = linear_readout(inputs[MODES[0]], *heads[MODES[1]])
            counts['new_readout_batches'] += 1
            counts['new_readout_rows'] += len(source)
            a, b = saved[MODES[0]].astype(np.float64), saved[MODES[1]].astype(np.float64)
            weights, content, total, action_error = decompose(a, crossed, b, c['analysis']['algebra_atol'])
            lw, lc, lt, logit_error = decompose(logits[MODES[0]], cross_logits, logits[MODES[1]], c['analysis']['algebra_atol'])
            archive = cached.write_npz(artifacts / f'seed{seed}-crossed-query.npz', {
                'source_rows': source, 'crossed_predictions': crossed, 'crossed_logits': cross_logits,
                'weight_action_term': weights, 'content_action_term': content, 'total_action_delta': total,
                'weight_logit_term': lw, 'content_logit_term': lc, 'total_logit_delta': lt})
            cells = {'current_head_current': a, 'history_head_current': crossed, 'history_head_history': b}
            metrics = {name: {k: cached.parent.metrics(value, targets[source], mask) for k, mask in masks.items()}
                       for name, value in cells.items()}
            for i in np.flatnonzero(masks['known_open_event']):
                row = int(source[i])
                cg, xg, hg = [float(v[i, 6]) for v in (a, crossed, b)]
                y = float(targets[row, 6])
                correct = {name: bool(np.sign(v[i, 6]) == np.sign(y)) for name, v in cells.items()}
                original = next(e for e in previous['all_event_results'] if
                    (e['training_seed'], e['demonstration_index'], e['event_step']) == (seed, int(ids[row]), int(steps[row])))
                require(cg == original['event_frame']['current']['g'] and hg == original['event_frame']['history']['g'], 'event identity differs')
                events.append({'training_seed': seed, 'demonstration_index': int(ids[row]), 'event_step': int(steps[row]),
                    'target_g': y, 'current_g': cg, 'crossed_g': xg, 'history_g': hg, 'correct': correct,
                    'original_case': original['event_frame']['case'], 'weight_action_term': float(weights[i, 6]),
                    'content_action_term': float(content[i, 6]), 'weight_logit_term': float(lw[i, 6]),
                    'content_logit_term': float(lc[i, 6]),
                    'history_content_harmed_at_same_head': bool(correct['history_head_current'] and not correct['history_head_history']),
                    'history_content_repaired_at_same_head': bool(not correct['history_head_current'] and correct['history_head_history'])})
            counts['model_event_pairs'] += int(masks['known_open_event'].sum())
            counts['full_post_pairs'] += int(masks['known_open_segment'].sum())
            results.append({'training_seed': seed, 'metrics': metrics, 'archive': archive,
                'replay_max_abs_errors': replay_errors, 'action_decomposition_error': action_error, 'logit_decomposition_error': logit_error})
        require(all(counts[k] == c['scope'][k] for k in counts), 'actual budget differs')
        for pin in c['inputs'].values():
            require(sha(pin['path']) == pin['sha256'], 'input mutated')
        harmed = [e for e in events if e['original_case'] == 'harmed']
        result = {'status': 'complete_fixed_head_crossed_history_readout', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path),
            'binding_sha256': sha(binding_path), 'scope': c['scope'], 'analysis_definition': c['analysis'],
            'input_provenance': c['inputs'], 'actual_counts': counts, 'seed_results': results, 'event_results': events,
            'original_harmed_events': harmed,
            'original_harmed_corrected_by_current_reference_n': sum(e['history_content_harmed_at_same_head'] for e in harmed),
            'prior_failed_history_gate_retained': True, 'automatic_training_or_rollout': False,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu',
                'entry_elapsed_seconds': time.perf_counter() - start, 'argv': sys.argv}}
        cached.write_json(output, result)
        return result
    except Exception as exc:
        cached.write_json(failure, {'status': 'failed_crossed_history_readout', 'stage': stage,
            'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts,
            'completed_seed_results': results, 'completed_events': events})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({k: result[k] for k in ('status', 'actual_counts', 'original_harmed_corrected_by_current_reference_n')}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'scope': c['scope']}))


if __name__ == '__main__':
    main()
