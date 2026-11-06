"""Complete observed expert-open segments from saved predictions; no new models.

Separates first response, maintained open tail, error reentry and censoring.
The saved intervention remains limited to its original306 selected frames.
"""
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
from experiments import analyze_libero_source_internal_gripper_sign_margin_temporal as base

require, sha, load, resolve, stats = base.require, base.sha, base.load, base.resolve, base.stats
TASKS, SEEDS, ARMS = base.TASKS, base.SEEDS, base.ARMS
CONDITIONS = ('identity', *base.CONDITIONS)
LANDMARKS = (0, 1, 2, 3, 4, 5, 10, 20)
DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-full-open-segments-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-full-open-segments-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-full-open-segments-20261104-v1.json')
CONFIG_SHA = 'df860a26504ef23ba712028234bc165fce09f6cc246e474104c4c09e1eb4a2dc'


def target_segments(targets, demo_ids, steps):
    require(targets.shape == (len(steps), 7) and len(demo_ids) == len(steps) and targets.dtype == np.float32 and
            np.isfinite(targets).all() and np.max(np.abs(targets)) <= 1 and
            np.issubdtype(demo_ids.dtype, np.integer) and np.issubdtype(steps.dtype, np.integer), 'cached target interface differs')
    known, prefixes = [], []
    for demo in sorted(set(demo_ids.tolist())):
        rows = np.flatnonzero(demo_ids == demo)
        require(np.array_equal(steps[rows], np.arange(1, len(rows) + 1)), 'demo steps must be contiguous from1')
        g = targets[rows, 6]
        def segment(start, kind):
            stop = start
            while stop < len(rows) and g[stop] < 0:
                stop += 1
            selected = rows[start:stop]
            return {'kind': kind, 'demonstration_index': int(demo),
                'event_step': int(steps[rows[start]]) if kind == 'known_open_event' else None,
                'first_observed_step': int(steps[selected[0]]) if len(selected) else None,
                'last_observed_step': int(steps[selected[-1]]) if len(selected) else None,
                'source_rows': selected.astype(int).tolist(),
                'end_reason': 'expert_command_change' if stop < len(rows) else 'demo_end',
                'next_expert_step': int(steps[rows[stop]]) if stop < len(rows) else None,
                'start_known': kind == 'known_open_event'}
        prefixes.append(segment(0, 'initial_observed_open_prefix'))
        starts = np.flatnonzero((g[1:] < 0) & (g[:-1] >= 0)) + 1
        known.extend(segment(int(start), 'known_open_event') for start in starts)
    return known, prefixes


def true_runs(mask):
    require(mask.ndim == 1 and mask.dtype == bool, 'run mask must be1D boolean')
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) != 1) + 1
    return [{'start_lag': int(part[0]), 'end_lag': int(part[-1]), 'length': len(part)}
            for part in np.split(indices, splits)]


def describe(g):
    require(g.ndim == 1 and g.dtype == np.float32 and np.isfinite(g).all() and
            (not len(g) or np.max(np.abs(g)) <= 1), 'saved gripper output interface differs')
    n = len(g)
    correct, wrong = g < 0, g >= 0
    first = int(np.flatnonzero(correct)[0]) if correct.any() else None
    last_wrong = int(np.flatnonzero(wrong)[-1]) if wrong.any() else None
    tail = (0 if last_wrong is None else last_wrong + 1) if n else None
    if tail == n:
        tail = None
    runs = true_runs(wrong)
    reentries = [r for r in runs if first is not None and r['start_lag'] > first]
    return {'n': n, 'wrong_frames': int(wrong.sum()), 'wrong_fraction': float(wrong.mean()) if n else None,
            'positive_wrong_frames': int((g > 0).sum()), 'neutral_error_frames': int((g == 0).sum()),
            'first_correct_lag': first, 'last_wrong_lag': last_wrong, 'tail_correct_start_lag': tail,
            'tail_correct_length': n - tail if tail is not None else 0,
            'recovery_not_observed_before_end': tail is None if n else None,
            'initial_wrong_observed_length': (first if first is not None else n) if n else None,
            'first_correct_missing': first is None if n else None,
            'wrong_reentry_count_after_first_correct': len(reentries) if first is not None else None,
            'first_reentry_lag': reentries[0]['start_lag'] if reentries else None,
            'longest_wrong_run': max((r['length'] for r in runs), default=0) if n else None,
            'all_correct': bool(correct.all()) if n else None, 'all_wrong': bool(wrong.all()) if n else None,
            'wrong_runs': runs, 'g': g.tolist()}


def summarize(rows):
    observed = [r for r in rows if r['metrics']['n']]
    m = [r['metrics'] for r in observed]
    n = sum(v['n'] for v in m)
    wrong = sum(v['wrong_frames'] for v in m)
    output = {'segments': len(rows), 'nonempty_segments': len(m), 'empty_segments': len(rows) - len(m),
        'scored_model_frames': n, 'wrong_model_frames': wrong, 'frame_weighted_wrong_fraction': wrong / n if n else None,
        'equal_segment_wrong_fraction': stats([v['wrong_fraction'] for v in m]),
        'first_correct_observed': stats([v['first_correct_lag'] for v in m if v['first_correct_lag'] is not None]),
        'first_correct_missing_segments': sum(v['first_correct_lag'] is None for v in m),
        'tail_correct_start_observed': stats([v['tail_correct_start_lag'] for v in m if v['tail_correct_start_lag'] is not None]),
        'tail_recovery_missing_segments': sum(v['recovery_not_observed_before_end'] for v in m),
        'initial_wrong_observed_length': stats([v['initial_wrong_observed_length'] for v in m]),
        'longest_wrong_run': stats([v['longest_wrong_run'] for v in m]),
        'all_correct_segments': sum(v['all_correct'] for v in m), 'all_wrong_segments': sum(v['all_wrong'] for v in m),
        'reentry_eligible_segments': sum(v['first_correct_lag'] is not None for v in m),
        'segments_with_wrong_reentry': sum((v['wrong_reentry_count_after_first_correct'] or 0) > 0 for v in m),
        'positive_wrong_model_frames': sum(v['positive_wrong_frames'] for v in m),
        'neutral_error_model_frames': sum(v['neutral_error_frames'] for v in m),
        'end_reasons': {k: sum(r['segment']['end_reason'] == k for r in rows) for k in ('expert_command_change', 'demo_end')}}
    output['landmarks'] = {}
    for lag in LANDMARKS:
        eligible = [v for v in m if v['n'] > lag]
        nw = sum(v['g'][lag] >= 0 for v in eligible)
        output['landmarks'][str(lag)] = {'eligible_segments': len(eligible), 'wrong_segments': nw,
            'first_correct_seen_segments': sum(v['first_correct_lag'] is not None and v['first_correct_lag'] <= lag for v in eligible),
            'wrong_fraction_among_eligible': nw / len(eligible) if eligible else None}
    return output


def harmed_context(segment, original, changed, source_row):
    rows = segment['source_rows']
    require(source_row in rows, 'old harmed frame not in its expert-open segment')
    lag = rows.index(source_row)
    require(original[lag] < 0 and changed[lag] >= 0, 'old harmed frame no longer crosses sign')
    before_wrong = bool(original[lag - 1] >= 0) if lag > 0 else None
    after_wrong = bool(original[lag + 1] >= 0) if lag + 1 < len(rows) else None
    if before_wrong is None or after_wrong is None:
        category = 'command_boundary_neighbor_unobserved'
    elif before_wrong and after_wrong:
        category = 'bridges_two_original_wrong_runs'
    elif before_wrong:
        category = 'extends_previous_original_wrong_run'
    elif after_wrong:
        category = 'advances_next_original_wrong_run'
    else:
        category = 'isolated_new_wrong_run'
    runs = true_runs(changed >= 0)
    containing = next(r for r in runs if r['start_lag'] <= lag <= r['end_lag'])
    return {'lag': lag, 'segment_length': len(rows), 'previous_original_wrong': before_wrong,
        'next_original_wrong': after_wrong, 'category': category, 'next_step_observed_in_command': lag + 1 < len(rows),
        'original_next_g': float(original[lag + 1]) if lag + 1 < len(rows) else None,
        'changed_next_g': float(changed[lag + 1]) if lag + 1 < len(rows) else None,
        'containing_changed_wrong_run': containing,
        'changed_wrong_run_reaches_command_end': containing['end_lag'] == len(rows) - 1,
        'end_reason': segment['end_reason']}


def load_inputs(config_path, binding_path):
    c = load(config_path)
    expected = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arms': list(ARMS), 'conditions': list(CONDITIONS),
        'checkpoint_updates': 500, 'closed_loop_stabilization_steps': 5, 'source_role': 'meta_support',
        'demonstration_indices': list(range(30)), 'known_open_events_by_task': {'0': 31, '1': 34},
        'models': 18, 'model_event_segments_per_condition': 585, 'new_policy_forward_batches': 0,
        'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
        'new_HDF5_RGB_initial_private_checkpoint_payload_reads': 0, 'loss_observation_changed': False,
        'sealed_roles_accessed': False, 'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c['formal_result'] is False, 'analysis is diagnostic')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'full-segment config hash differs')
    for key, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {key}')
    b = load(binding_path)
    require(b['config_sha256'] == CONFIG_SHA and b['formal_result'] is False and set(b['files']) == {'analyzer', 'tests'}, 'code binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound code differs')
    require(resolve(b['files']['analyzer']['path']).resolve() == Path(__file__).resolve(), 'analyzer path differs')
    parent = load(c['inputs']['control_result']['path'])
    audit = load(c['inputs']['control_audit']['path'])
    margin = load(c['inputs']['margin_result']['path'])
    require(parent['status'] == 'complete' and audit['status'] == 'complete_independent_saved_array_audit' and
            margin['status'] == 'complete_readonly_margin_temporal_analysis' and
            parent['actual_counts'] == audit['actual_counts_verified'], 'prior run/audit incomplete')
    for name in ('control_binding', 'margin_binding'):
        for pin in load(c['inputs'][name]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'historical implementation changed')
    require(len(parent['model_results']) == 18 and
        {(r['task_id'], r['training_seed'], r['arm']) for r in parent['model_results']} ==
        {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS}, 'fixed model matrix differs')
    for row in parent['model_results']:
        pin = row['prediction_file']
        require(sha(pin['path']) == pin['sha256'], 'saved prediction hash differs')
    return c, b, parent, margin, load(c['inputs']['samples']['path'])


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT):
    output = Path(output)
    failure = output.with_name(output.stem + '-failed.json')
    require(not output.exists() and not failure.exists(), 'output already exists; preserve historical artifacts')
    started = time.monotonic()
    segment_rows, prefix_rows, harm_rows, task_plans = [], [], [], []
    try:
        c, binding, parent, margin, samples = load_inputs(config_path, binding_path)
        for task in TASKS:
            sample = next(s for s in samples['tasks'] if s['task_id'] == task)
            with np.load(resolve(c['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
                targets, ids, times, old_predictions, model_ids = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps', 'predictions', 'model_ids')]
            require(np.array_equal(np.column_stack((ids, times)), sample['source_frame_ids']) and set(ids.tolist()) == set(range(30)), 'cached source IDs/roles differ')
            known, prefixes = target_segments(targets, ids, times)
            require([(s['demonstration_index'], s['event_step']) for s in known] ==
                [tuple(sample['source_frame_ids'][i]) for i in sample['at_event_sham_rows']] and
                len(known) == c['scope']['known_open_events_by_task'][str(task)], 'frozen event set differs')
            task_plans.append({'task_id': task, 'known_segments': known, 'initial_observed_prefixes': prefixes})
            selected = np.zeros(len(targets), bool)
            selected[[r['source_row'] for r in sample['selected']]] = True
            for seed in SEEDS:
                for arm in ARMS:
                    mi = SEEDS.index(seed) * 3 + ARMS.index(arm)
                    require(str(model_ids[mi]) == f'{seed}:{arm}', 'original model ordering differs')
                    row = next(r for r in parent['model_results'] if (r['task_id'], r['training_seed'], r['arm']) == (task, seed, arm))
                    with np.load(row['prediction_file']['path'], allow_pickle=False) as z:
                        pred = z['predictions']
                        require(z['conditions'].tolist() == list(CONDITIONS), 'condition ordering differs')
                    require(pred.dtype == np.float32 and pred.shape == (3, len(targets), 7) and np.isfinite(pred).all() and
                        np.max(np.abs(pred)) <= 1 and np.array_equal(pred[0], old_predictions[mi]) and
                        np.array_equal(pred[1:, ~selected], np.broadcast_to(pred[0, ~selected], pred[1:, ~selected].shape)),
                        'raw prediction identity/outside-intervention gate differs')
                    context = {'task_id': task, 'training_seed': seed, 'arm': arm}
                    for segment in known:
                        indices = segment['source_rows']
                        for ci, condition in enumerate(CONDITIONS):
                            segment_rows.append({**context, 'condition': condition, 'segment': segment,
                                                 'metrics': describe(pred[ci, indices, 6])})
                        for ci, condition in enumerate(CONDITIONS[1:], 1):
                            actual_harm_rows = [i for i in indices if pred[0, i, 6] < 0 and pred[ci, i, 6] >= 0]
                            prior_cases = [h for h in margin['all_harmed_cases'][condition] if
                                (h['task_id'], h['training_seed'], h['arm'], h['demonstration_index'], h['event_step']) ==
                                (task, seed, arm, segment['demonstration_index'], segment['event_step'])]
                            require(actual_harm_rows == [h['source_row'] for h in prior_cases], 'old harmed set changed under full command')
                            for h in prior_cases:
                                require(float(pred[0, h['source_row'], 6]) == h['original_g'] and
                                        float(pred[ci, h['source_row'], 6]) == h['changed_g'], 'old harmed raw values changed')
                                harm_rows.append({**h, 'full_command_context': harmed_context(segment, pred[0, indices, 6], pred[ci, indices, 6], h['source_row'])})
                    for segment in prefixes:
                        prefix_rows.append({**context, 'condition': 'identity', 'segment': segment,
                                            'metrics': describe(pred[0, segment['source_rows'], 6])})
        summaries = []
        for task in TASKS:
            for condition in CONDITIONS:
                for arm in (None, *ARMS):
                    rows = [r for r in segment_rows if r['task_id'] == task and r['condition'] == condition and (arm is None or r['arm'] == arm)]
                    summaries.append({'task_id': task, 'condition': condition, 'arm': arm, **summarize(rows),
                        'per_demo': [{'demonstration_index': d, **summarize([r for r in rows if r['segment']['demonstration_index'] == d])} for d in range(30)]})
        paired = []
        for task in TASKS:
            for seed in SEEDS:
                for control in ('A_original', 'C_pooled_bc'):
                    left = [r for r in segment_rows if (r['task_id'], r['training_seed'], r['arm'], r['condition']) == (task, seed, 'B_fomaml', 'identity')]
                    right = [r for r in segment_rows if (r['task_id'], r['training_seed'], r['arm'], r['condition']) == (task, seed, control, 'identity')]
                    require([r['segment'] for r in left] == [r['segment'] for r in right], 'unpaired expert command segments')
                    differences = {k: [] for k in ('wrong_fraction', 'first_correct_lag', 'tail_correct_start_lag', 'longest_wrong_run')}
                    for b, a in zip(left, right):
                        for k in differences:
                            bv, av = b['metrics'][k], a['metrics'][k]
                            differences[k].append(bv - av if bv is not None and av is not None else None)
                    paired.append({'task_id': task, 'training_seed': seed, 'comparison': 'B_fomaml-' + control,
                        'segments': len(left), 'differences': {k: {**stats([v for v in values if v is not None]),
                                                                  'missing_pairs': sum(v is None for v in values)} for k, values in differences.items()}})
        require(len(segment_rows) == 1755 and len(prefix_rows) == 540 and
                sum(h['condition'] == CONDITIONS[1] for h in harm_rows) == 17 and
                sum(h['condition'] == CONDITIONS[2] for h in harm_rows) == 5, 'segment/model/harmed budget differs')
        result = {'status': 'complete_full_expert_open_segment_analysis', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path),
            'scope': c['scope'], 'analysis_definition': c['analysis'], 'input_provenance': c['inputs'], 'task_segment_plans': task_plans,
            'model_segment_results': segment_rows, 'initial_prefix_results': prefix_rows, 'group_summaries': summaries,
            'initial_prefix_summaries': [{'task_id': t, **summarize([r for r in prefix_rows if r['task_id'] == t])} for t in TASKS],
            'paired_arm_identity_summaries': paired, 'prior_harmed_full_context': harm_rows,
            'actual_new_research_counts': {k: 0 for k in ('policy_forward_batches', 'optimizer_updates', 'backward_calls',
                'environment_calls', 'HDF5_RGB_initial_private_checkpoint_payload_reads')},
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu', 'elapsed_seconds': time.monotonic() - started},
            'interpretation': 'Complete observed expert command segments, not physical release or closed-loop duration. Prefix starts unknown; unobserved recovery stays censored. No training promotion.'}
        with output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        return result
    except Exception as error:
        with failure.open('x') as handle:
            json.dump({'status': 'failed_readonly_full_segment_analysis', 'error_type': type(error).__name__, 'error': str(error),
                'completed_segment_rows': len(segment_rows), 'actual_new_research_calls': 0}, handle, indent=2)
            handle.write('\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.binding, args.output)
    print(json.dumps({'status': result['status'], 'model_event_segments': len(result['model_segment_results']),
                      'initial_prefix_segments': len(result['initial_prefix_results']), 'actual_new_research_counts': result['actual_new_research_counts']}))
