"""Read paired query predictions: exact MSE accounting and complete open timing."""
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
from experiments import analyze_libero_source_internal_full_open_segments as full

require, sha, load, resolve, stats = full.require, full.sha, full.load, full.resolve, full.stats
SEEDS = (20261022, 20261023, 20261024)
MODES = ('current_repeat5', 'causal_history5')
CASES = ('stable_correct', 'harmed', 'repaired', 'wrong_both')
LAGS = tuple(range(-4, 6))
DEFAULT_CONFIG = ROOT / 'configs/libero-task1-b-query-event-margin-timing-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task1-b-query-event-margin-timing-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task1-b-query-event-margin-timing-20261104-v1.json')
CONFIG_SHA = '9add68417dbb586dd8121ba2e3043d830af0449c574c725b1872be11c4d8e1be'


def frame_pair(current, history, target):
    require(current.shape == history.shape == target.shape == (7,) and
            current.dtype == history.dtype == target.dtype == np.float32 and
            all(np.isfinite(x).all() and np.max(np.abs(x)) <= 1 for x in (current, history, target)), 'invalid saved action pair')
    y = float(target[6])
    result = {}
    for name, value in [('current', current), ('history', history)]:
        g = float(value[6])
        squared = (value.astype(np.float64) - target.astype(np.float64)) ** 2
        result[name] = {'g': g, 'abs_g': abs(g), 'sign': int(np.sign(g)), 'correct': bool(np.sign(g) == np.sign(y)),
                        'gripper_squared_error': float(squared[6]), 'translation_MSE': float(squared[:3].mean()),
                        'rotation_MSE': float(squared[3:6].mean()), 'seven_dimensional_MSE': float(squared.mean())}
    a, b = result['current'], result['history']
    case = CASES[0] if a['correct'] and b['correct'] else CASES[1] if a['correct'] else CASES[2] if b['correct'] else CASES[3]
    delta = b['gripper_squared_error'] - a['gripper_squared_error']
    identity = (b['g'] - a['g']) * (b['g'] + a['g'] - 2 * y)
    require(abs(delta - identity) <= 1e-10, 'gripper squared error identity differs')
    result.update(case=case, target_g=y, target_sign=int(np.sign(y)), delta_g=b['g'] - a['g'],
                  delta_gripper_squared_error=delta, gripper_delta_identity=identity,
                  delta_translation_MSE=b['translation_MSE'] - a['translation_MSE'],
                  delta_seven_dimensional_MSE=b['seven_dimensional_MSE'] - a['seven_dimensional_MSE'])
    return result


def cohort_accounting(records):
    n = len(records)
    output = {}
    for case in CASES:
        selected = [r for r in records if r['case'] == case]
        output[case] = {'n': len(selected), 'current_g': stats([r['current']['g'] for r in selected]),
            'history_g': stats([r['history']['g'] for r in selected]),
            'current_abs_g': stats([r['current']['abs_g'] for r in selected]),
            'history_abs_g': stats([r['history']['abs_g'] for r in selected]),
            'delta_g': stats([r['delta_g'] for r in selected]),
            'gripper_squared_error_delta': stats([r['delta_gripper_squared_error'] for r in selected]),
            'translation_MSE_delta': stats([r['delta_translation_MSE'] for r in selected]),
            'current_zero_n': sum(r['current']['sign'] == 0 for r in selected),
            'history_zero_n': sum(r['history']['sign'] == 0 for r in selected),
            'gripper_MSE_delta_contribution': sum(r['delta_gripper_squared_error'] for r in selected) / n if n else None,
            'translation_MSE_delta_contribution': sum(r['delta_translation_MSE'] for r in selected) / n if n else None,
            'seven_dimensional_MSE_delta_contribution': sum(r['delta_seven_dimensional_MSE'] for r in selected) / n if n else None}
    current_error = sum(not r['current']['correct'] for r in records)
    history_error = sum(not r['history']['correct'] for r in records)
    require(history_error - current_error == output['harmed']['n'] - output['repaired']['n'], 'paired error count identity differs')
    means = {name: {field: float(np.mean([r[name][field] for r in records])) if n else None
                   for field in ('gripper_squared_error', 'translation_MSE', 'seven_dimensional_MSE')} for name in ('current', 'history')}
    if n:
        require(abs(sum(v['gripper_MSE_delta_contribution'] for v in output.values()) -
                    (means['history']['gripper_squared_error'] - means['current']['gripper_squared_error'])) <= 1e-10, 'cohort MSE contributions do not add up')
    return {'n': n, 'current_error_n': current_error, 'history_error_n': history_error, 'error_count_delta': history_error - current_error,
            'means': means, 'cases': output}


def prefix_length(mask):
    require(mask.ndim == 1 and mask.dtype == bool, 'prefix requires boolean sequence')
    missing = np.flatnonzero(~mask)
    return int(missing[0]) if len(missing) else len(mask)


def timing_pair(current_g, history_g):
    require(current_g.shape == history_g.shape and len(current_g) > 0, 'empty or mismatched command')
    a, b = full.describe(current_g), full.describe(history_g)
    first_a, first_b = a['first_correct_lag'], b['first_correct_lag']
    tail_a, tail_b = a['tail_correct_start_lag'], b['tail_correct_start_lag']
    def observed_status(x, y):
        return 'both_observed' if x is not None and y is not None else 'only_current_observed' if x is not None else 'only_history_observed' if y is not None else 'neither_observed'
    return {'current': a, 'history': b,
            'first_correct_status': observed_status(first_a, first_b),
            'first_correct_lag_delta_if_both_observed': first_b - first_a if first_a is not None and first_b is not None else None,
            'tail_correct_status': observed_status(tail_a, tail_b),
            'tail_correct_start_delta_if_both_observed': tail_b - tail_a if tail_a is not None and tail_b is not None else None,
            'initial_additional_wrong_prefix': prefix_length((current_g < 0) & (history_g >= 0)),
            'initial_additional_correct_prefix': prefix_length((current_g >= 0) & (history_g < 0)),
            'current_initial_correct_prefix': prefix_length(current_g < 0),
            'history_initial_correct_prefix': prefix_length(history_g < 0)}


def summarize_events(events):
    output = {}
    for case in ('all', *CASES):
        selected = events if case == 'all' else [e for e in events if e['event_frame']['case'] == case]
        output[case] = {'n': len(selected),
            'timing': {name: full.summarize([{'segment': e['segment'], 'metrics': e['timing'][name]} for e in selected]) for name in ('current', 'history')},
            'first_correct_status': {name: sum(e['timing']['first_correct_status'] == name for e in selected) for name in
                ('both_observed', 'only_current_observed', 'only_history_observed', 'neither_observed')},
            'first_correct_lag_delta_if_both_observed': stats([e['timing']['first_correct_lag_delta_if_both_observed'] for e in selected if e['timing']['first_correct_lag_delta_if_both_observed'] is not None]),
            'tail_correct_status': {name: sum(e['timing']['tail_correct_status'] == name for e in selected) for name in
                ('both_observed', 'only_current_observed', 'only_history_observed', 'neither_observed')},
            'initial_additional_wrong_prefix': stats([e['timing']['initial_additional_wrong_prefix'] for e in selected]),
            'initial_additional_correct_prefix': stats([e['timing']['initial_additional_correct_prefix'] for e in selected])}
    return output


def preflight(config_path, binding_path):
    c = load(config_path)
    require(c['formal_result'] is False, 'must remain diagnostic')
    expected = {'task_id': 1, 'feature_arm': 'B_fomaml', 'training_seeds': list(SEEDS), 'readout_modes': list(MODES),
        'readout_updates': 500, 'original_policy_updates': 500, 'closed_loop_default_stabilization_steps': 5,
        'demonstration_role': 'meta_query', 'demonstration_indices': list(range(30, 50)), 'cached_target_rows': 2598,
        'cached_prediction_rows': 15108, 'common_rows_per_model': 2518, 'prediction_archives': 6,
        'known_open_events': 23, 'model_event_pairs': 69, 'full_post_model_frame_pairs': 696, 'relative_window_records': 690,
        'relative_lags': list(LAGS), 'new_model_forward_calls': 0, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
        'new_environment_calls': 0, 'new_HDF5_RGB_initial_private_checkpoint_payload_reads': 0, 'new_feature_payload_reads': 0,
        'loss_budget_or_policy_changed': False, 'checkpoint_selected': False, 'sealed_roles_accessed': False,
        'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and set(b['files']) == {'analyzer', 'tests', 'protocol'}, 'execution binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'execution code differs')
    require(resolve(b['files']['analyzer']['path']).resolve() == Path(__file__).resolve(), 'analyzer binding path differs')
    for pin in load(c['inputs']['parent_binding']['path'])['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'parent execution changed')
    return c, b


def validate_parent_metrics(predictions, targets, masks, expected):
    for name, mask in masks.items():
        pred, target = predictions[mask].astype(np.float64), targets[mask].astype(np.float64)
        n = len(target)
        square = (pred - target) ** 2
        got = {'n': n, 'translation_MSE': float(square[:, :3].mean()) if n else None,
               'rotation_MSE': float(square[:, 3:6].mean()) if n else None,
               'gripper_MSE': float(square[:, 6].mean()) if n else None,
               'seven_dimensional_MSE': float(square.mean()) if n else None,
               'gripper_sign_error_n': int((np.sign(pred[:, 6]) != np.sign(target[:, 6])).sum()),
               'gripper_sign_error_rate': float((np.sign(pred[:, 6]) != np.sign(target[:, 6])).mean()) if n else None,
               'gripper_zero_n': int((pred[:, 6] == 0).sum())}
        for key, value in got.items():
            old = expected[name][key]
            require(abs(value - old) <= 1e-10 if isinstance(value, float) else value == old, f'parent metric differs: {name}/{key}')


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT):
    output = Path(output)
    failed = output.with_name(output.stem + '-failed.json')
    require(not output.exists() and not failed.exists(), 'output/failure already exists; preserve history')
    started = time.monotonic()
    events, seed_summaries, saved_files = [], [], []
    counts = dict.fromkeys(('cached_target_rows', 'cached_prediction_rows', 'prediction_archives', 'model_event_pairs',
        'full_post_model_frame_pairs', 'relative_window_records', 'new_model_forward_calls', 'new_optimizer_updates',
        'new_backward_calls', 'new_environment_calls', 'new_HDF5_RGB_initial_private_checkpoint_payload_reads', 'new_feature_payload_reads'), 0)
    try:
        c, binding = preflight(config_path, binding_path)
        parent = load(c['inputs']['parent_result']['path'])
        audit = load(c['inputs']['parent_audit']['path'])
        require(parent['status'] == 'complete_task1_b_support_query_history_readout' and parent['formal_result'] is False and
                audit['status'] == 'complete_independent_task1_support_query_readout_audit' and
                audit['result']['sha256'] == sha(c['inputs']['parent_result']['path']) and
                parent['primary_query_comparison']['joint_query_cue'] is False, 'parent completion/audit/stop decision differs')
        require(parent['config_sha256'] == sha(c['inputs']['parent_config']['path']) and
                parent['binding_sha256'] == sha(c['inputs']['parent_binding']['path']), 'parent provenance differs')
        heads = parent['readout_results']
        require(len(heads) == 6 and {(h['training_seed'], h['mode']) for h in heads} == {(s, m) for s in SEEDS for m in MODES}, 'fixed readout matrix differs')
        pins = [parent['query_shared_archive']] + [h['query_prediction_archive'] for h in heads]
        require(len({p['path'] for p in pins}) == 7, 'duplicate query archive paths')
        for pin in pins:
            require(sha(pin['path']) == pin['sha256'], 'saved query hash differs')
        saved_files = pins
        with np.load(pins[0]['path'], allow_pickle=False) as z:
            targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
        require(targets.dtype == np.float32 and targets.shape == (2598, 7) and set(ids.tolist()) == set(range(30, 50)), 'cached query target roles/counts differ')
        counts['cached_target_rows'] = len(targets)
        known, prefixes = full.target_segments(targets, ids, steps)
        source = np.flatnonzero(steps >= 5)
        require(len(source) == 2518 and len(known) == 23 and all(s['event_step'] >= 5 for s in known), 'query coverage differs')
        lookup = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(ids, steps))}
        require(len(lookup) == len(targets), 'duplicate query frame IDs')
        masks = {'known_open_event': np.zeros(len(targets), bool), 'known_open_segment': np.zeros(len(targets), bool),
                 'initial_open': np.zeros(len(targets), bool), 'expert_close': targets[:, 6] > 0}
        for segment in known:
            masks['known_open_event'][segment['source_rows'][0]] = True
            masks['known_open_segment'][segment['source_rows']] = True
        for segment in prefixes:
            masks['initial_open'][segment['source_rows']] = True
        masks = {'all': np.ones(len(source), bool), **{k: v[source] for k, v in masks.items()}}
        positions = {int(row): i for i, row in enumerate(source)}
        for seed in SEEDS:
            predictions = {}
            for mode in MODES:
                head = next(h for h in heads if h['training_seed'] == seed and h['mode'] == mode)
                with np.load(head['query_prediction_archive']['path'], allow_pickle=False) as z:
                    cached_rows, prediction = z['source_rows'], z['predictions']
                require(np.array_equal(cached_rows, source) and prediction.shape == (2518, 7) and prediction.dtype == np.float32 and
                        np.isfinite(prediction).all() and np.max(np.abs(prediction)) <= 1, 'cached prediction interface differs')
                parent_metric = next(x for x in parent['seed_results'] if x['training_seed'] == seed)['query_metrics'][mode]
                validate_parent_metrics(prediction, targets[source], masks, parent_metric)
                predictions[mode] = prediction
                counts['prediction_archives'] += 1
                counts['cached_prediction_rows'] += len(prediction)
            seed_events = []
            for segment in known:
                demo, step = segment['demonstration_index'], segment['event_step']
                rows = segment['source_rows']
                require(all(r in positions for r in rows) and np.all(targets[rows, 6] < 0), 'post command contains unavailable or changed-label rows')
                current, history = [predictions[mode][[positions[r] for r in rows]] for mode in MODES]
                post = [{'source_row': int(row), 'action_step': int(steps[row]), 'lag': lag,
                         **frame_pair(current[lag], history[lag], targets[row])} for lag, row in enumerate(rows)]
                relative = []
                for lag in LAGS:
                    if lag >= 0:
                        if lag < len(post):
                            relative.append({'lag': lag, 'available': True, 'frame': post[lag]})
                        else:
                            relative.append({'lag': lag, 'available': False, 'reason': 'expert_open_command_ended:' + segment['end_reason']})
                    else:
                        row = lookup.get((demo, step + lag))
                        if row is None or row not in positions:
                            relative.append({'lag': lag, 'available': False,
                                             'reason': 'outside_observed_demo' if row is None else 'prediction_cache_excludes_t_lt5'})
                        else:
                            j = positions[row]
                            relative.append({'lag': lag, 'available': True, 'frame': {'source_row': row, 'action_step': int(steps[row]),
                                **frame_pair(predictions[MODES[0]][j], predictions[MODES[1]][j], targets[row])}})
                event = {'training_seed': seed, 'demonstration_index': demo, 'event_step': step, 'segment': segment,
                         'event_frame': post[0], 'full_post_frames': post, 'relative_window': relative,
                         'timing': timing_pair(current[:, 6], history[:, 6])}
                seed_events.append(event)
                events.append(event)
                counts['model_event_pairs'] += 1
                counts['full_post_model_frame_pairs'] += len(post)
                counts['relative_window_records'] += len(relative)
            accounting = cohort_accounting([e['event_frame'] for e in seed_events])
            parent_seed = next(x for x in parent['seed_results'] if x['training_seed'] == seed)
            for name, mode in [('current', MODES[0]), ('history', MODES[1])]:
                old = parent_seed['query_metrics'][mode]['known_open_event']
                require(accounting[name + '_error_n'] == old['gripper_sign_error_n'] and
                        abs(accounting['means'][name]['gripper_squared_error'] - old['gripper_MSE']) <= 1e-10, 'event accounting does not reproduce parent')
            seed_summaries.append({'training_seed': seed, 'event_accounting': accounting,
                                   'full_post_accounting': cohort_accounting([f for e in seed_events for f in e['full_post_frames']]),
                                   'event_timing_summaries': summarize_events(seed_events)})
        for key, value in counts.items():
            require(value == c['scope'][key], f'actual count differs: {key}')
        relative_summary = {}
        for lag in LAGS:
            values = [next(w for w in e['relative_window'] if w['lag'] == lag) for e in events]
            frames = [w['frame'] for w in values if w['available']]
            relative_summary[str(lag)] = {'requested_model_event_pairs': len(values), 'available_pairs': len(frames),
                'unavailable': {reason: sum(not w['available'] and w.get('reason') == reason for w in values)
                                for reason in sorted({w['reason'] for w in values if not w['available']})},
                'target_sign_counts': {str(sign): sum(f['target_sign'] == sign for f in frames) for sign in (-1, 0, 1)},
                'paired_frame_accounting': cohort_accounting(frames)}
        result = {'status': 'complete_readonly_query_event_margin_timing', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path),
            'scope': c['scope'], 'analysis_definition': c['analysis'], 'input_provenance': c['inputs'], 'saved_prediction_inputs': saved_files,
            'actual_counts': counts, 'seed_summaries': seed_summaries, 'all_event_results': events,
            'pooled_event_accounting': cohort_accounting([e['event_frame'] for e in events]),
            'pooled_full_post_accounting': cohort_accounting([f for e in events for f in e['full_post_frames']]),
            'pooled_event_timing_summaries': summarize_events(events), 'relative_window_summaries': relative_summary,
            'expert_event_target_values': sorted({e['event_frame']['target_g'] for e in events}),
            'prior_joint_query_cue': False, 'prior_no_rollout_decision_retained': True,
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu',
                        'entry_elapsed_seconds': time.monotonic() - started, 'argv': sys.argv},
            'interpretation': 'Exact descriptive accounting of saved query readouts; no root-cause, physical-phase, training or closed-loop result.'}
        with output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        return result
    except Exception as exc:
        with failed.open('x') as handle:
            json.dump({'status': 'failed_readonly_query_event_margin_timing', 'formal_result': False, 'error_type': type(exc).__name__,
                       'error': str(exc), 'actual_counts_so_far': counts, 'completed_events': events,
                       'completed_seed_summaries': seed_summaries}, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
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
        print(json.dumps({'status': result['status'], 'output': str(args.output), 'actual_counts': result['actual_counts']}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'model_event_pairs': c['scope']['model_event_pairs'], 'new_numeric_analysis_rows': 0}))


if __name__ == '__main__':
    main()
