"""Read saved gripper outputs: strict sign margin and observed event-local runs.

No model, source HDF5, RGB, initial/private state or checkpoint payload is read.
No near-zero cutoff is selected; every original selected frame is retained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-gripper-sign-margin-temporal-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-gripper-sign-margin-temporal-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-gripper-sign-margin-temporal-20261104-v1.json')
CONFIG_SHA = '808cf3ff5bd4ec4da7e64eb4392d3aed4cd65f722a938a7fa6b2fe545f83159f'
TASKS, SEEDS = (0, 1), (20261022, 20261023, 20261024)
ARMS = ('A_original', 'B_fomaml', 'C_pooled_bc')
CONDITIONS = ('hold_event_preaction_gripper', 'lag1_gripper')
CASES = ('stable_correct', 'harmed', 'repaired', 'wrong_both')
SIGN_NAMES = {-1: 'neg', 0: 'zero', 1: 'pos'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    value = json.loads(resolve(path).read_text())
    require(isinstance(value, dict), 'expected JSON object')
    return value


def stats(values):
    values = [float(v) for v in values]
    require(all(np.isfinite(values)), 'nonfinite summary value')
    return {'n': len(values), 'mean': float(np.mean(values)) if values else None,
            'median': float(np.median(values)) if values else None,
            'min': min(values) if values else None, 'max': max(values) if values else None}


def frame_cohorts(records):
    output = {}
    for case in ('all', *CASES):
        rows = records if case == 'all' else [r for r in records if r['case'] == case]
        output[case] = {'n': len(rows), 'original_margin': stats([r['original_margin'] for r in rows]),
            'original_g': stats([r['original_g'] for r in rows]), 'changed_g': stats([r['changed_g'] for r in rows]),
            'delta_g': stats([r['delta_g'] for r in rows]),
            'lag_counts': {str(lag): sum(r['lag'] == lag for r in rows) for lag in range(1, 6)},
            'changed_positive_close_n': sum(r['changed_sign'] > 0 for r in rows),
            'changed_exact_zero_n': sum(r['changed_sign'] == 0 for r in rows)}
    require(sum(output[k]['n'] for k in CASES) == len(records), 'frame cohort partition differs')
    return output


def contiguous_runs(records, key):
    """Actual t adjacency within one explicit model/condition/demo/event only."""
    require(key in ('original_wrong', 'changed_wrong', 'is_harmed', 'persistent_at_frame'), 'unknown run predicate')
    if not records:
        return []
    scope_keys = ('task_id', 'training_seed', 'arm', 'condition', 'demonstration_index', 'event_step')
    require(len({tuple(r[k] for k in scope_keys) for r in records}) == 1, 'run cannot join different model/event scopes')
    ordered = sorted(records, key=lambda r: r['action_step'])
    times = [r['action_step'] for r in ordered]
    require(len(set(times)) == len(times), 'duplicate event action step')
    available = set(times)
    groups, current = [], []
    for row in ordered:
        if not row[key] or (current and row['action_step'] != current[-1]['action_step'] + 1):
            if current:
                groups.append(current)
                current = []
        if row[key]:
            current.append(row)
    if current:
        groups.append(current)
    return [{'start_step': rows[0]['action_step'], 'end_step': rows[-1]['action_step'],
             'start_lag': rows[0]['lag'], 'end_lag': rows[-1]['lag'], 'length': len(rows),
             'source_rows': [r['source_row'] for r in rows],
             'left_scope_censored': rows[0]['action_step'] - 1 not in available,
             'right_scope_censored': rows[-1]['action_step'] + 1 not in available,
             'original_wrong_n': sum(r['original_wrong'] for r in rows),
             'harmed_n': sum(r['is_harmed'] for r in rows)} for rows in groups]


def event_summary(records, demo, event):
    require(all(r['demonstration_index'] == demo and r['event_step'] == event for r in records), 'event membership differs')
    runs = {k: contiguous_runs(records, k) for k in ('original_wrong', 'changed_wrong', 'is_harmed', 'persistent_at_frame')}
    for run in runs['is_harmed']:
        parent = next(p for p in runs['changed_wrong'] if set(run['source_rows']).issubset(p['source_rows']))
        run['containing_changed_wrong_run'] = {k: parent[k] for k in
            ('start_step', 'end_step', 'start_lag', 'end_lag', 'length', 'original_wrong_n', 'harmed_n',
             'left_scope_censored', 'right_scope_censored')}
        run['joins_existing_wrong_run'] = parent['original_wrong_n'] > 0
    lags = sorted(r['lag'] for r in records)
    return {'demonstration_index': demo, 'event_step': event, 'n': len(records), 'available_lags': lags,
            'missing_lags': [lag for lag in range(1, 6) if lag not in lags],
            'complete_selected_lag1_to5': lags == list(range(1, 6)), 'cohorts': frame_cohorts(records), 'runs': runs,
            'observed_sign_pattern': [{'lag': r['lag'], 'original_sign': r['original_sign'],
                                      'changed_sign': r['changed_sign'], 'case': r['case']} for r in sorted(records, key=lambda r: r['action_step'])]}


def summarize(records, events):
    run_stats = {}
    for kind in ('original_wrong', 'changed_wrong', 'is_harmed', 'persistent_at_frame'):
        runs = [r for event in events for r in event['runs'][kind]]
        run_stats[kind] = {'runs': len(runs), 'length': stats([r['length'] for r in runs]),
            'single_step_runs': sum(r['length'] == 1 for r in runs),
            'two_or_more_observed_step_runs': sum(r['length'] >= 2 for r in runs),
            'left_scope_censored_runs': sum(r['left_scope_censored'] for r in runs),
            'right_scope_censored_runs': sum(r['right_scope_censored'] for r in runs)}
    harmed_runs = [r for event in events for r in event['runs']['is_harmed']]
    return {'cohorts': frame_cohorts(records), 'run_summary': run_stats,
            'harmed_runs_joining_existing_wrong_run': sum(r['joins_existing_wrong_run'] for r in harmed_runs),
            'harmed_runs_with_only_new_wrong_frames': sum(not r['joins_existing_wrong_run'] for r in harmed_runs),
            'event_windows': len(events), 'zero_selected_event_windows': sum(e['n'] == 0 for e in events),
            'complete_selected_event_windows': sum(e['complete_selected_lag1_to5'] for e in events),
            'event_windows_with_harm': sum(e['cohorts']['harmed']['n'] > 0 for e in events),
            'event_windows_with_original_error': sum(e['cohorts']['wrong_both']['n'] + e['cohorts']['repaired']['n'] > 0 for e in events),
            'unique_selected_source_frames': len({(r['task_id'], r['demonstration_index'], r['action_step']) for r in records}),
            'unique_harmed_source_frames': len({(r['task_id'], r['demonstration_index'], r['action_step']) for r in records if r['is_harmed']}),
            'unique_harmed_source_events': len({(r['task_id'], r['demonstration_index'], r['event_step']) for r in records if r['is_harmed']})}


def analyze_model(task, seed, arm, condition, targets, original, changed, sample):
    require(task in TASKS and seed in SEEDS and arm in ARMS and condition in CONDITIONS, 'unknown model/condition')
    require(original.shape == changed.shape == targets.shape == (sample['source_frames'], 7) and
            original.dtype == changed.dtype == targets.dtype == np.float32 and
            all(np.isfinite(x).all() and np.max(np.abs(x)) <= 1 for x in (original, changed, targets)), 'prediction/target interface differs')
    context = {'task_id': task, 'training_seed': seed, 'arm': arm, 'condition': condition}
    index = SEEDS.index(seed) * 3 + ARMS.index(arm)
    selected = sample['selected']
    require(len({r['source_row'] for r in selected}) == len(selected), 'duplicate selected source row')
    correct_margins = sorted(abs(float(original[r['source_row'], 6])) for r in selected if original[r['source_row'], 6] < 0)
    records = []
    for selected_row in selected:
        i = selected_row['source_row']
        require(0 <= i < len(targets) and targets[i, 6] < 0 and
                sample['source_frame_ids'][i] == [selected_row['demonstration_index'], selected_row['action_step']] and
                selected_row['action_step'] - selected_row['event_step'] == selected_row['lag'] and 1 <= selected_row['lag'] <= 5,
                'frozen expert-open frame/lag differs')
        before, after = float(original[i, 6]), float(changed[i, 6])
        old_sign, new_sign = int(np.sign(before)), int(np.sign(after))
        require(old_sign == selected_row['original_predicted_gripper_signs'][index], 'frozen original sign label differs')
        old_wrong, new_wrong = old_sign >= 0, new_sign >= 0
        case = ('wrong_both' if new_wrong else 'repaired') if old_wrong else ('harmed' if new_wrong else 'stable_correct')
        margin, delta = abs(before), after - before
        if case == 'harmed':
            require(delta > 0 and abs(delta - (margin + after)) <= 1e-15, 'sign-crossing identity differs')
        records.append({**context, **{k: selected_row[k] for k in ('source_row', 'demonstration_index', 'action_step', 'event_step', 'lag')},
            'original_g': before, 'changed_g': after, 'delta_g': delta, 'original_margin': margin,
            'changed_margin': abs(after), 'original_sign': old_sign, 'changed_sign': new_sign,
            'case': case, 'original_wrong': old_wrong, 'changed_wrong': new_wrong,
            'is_harmed': case == 'harmed', 'persistent_at_frame': case == 'wrong_both',
            'delta_over_original_margin': delta / margin if margin else None,
            'original_correct_reference_n': len(correct_margins),
            'original_correct_margin_fraction_le': sum(v <= margin for v in correct_margins) / len(correct_margins) if correct_margins else None})
    events = []
    event_keys = [tuple(sample['source_frame_ids'][i]) for i in sample['at_event_sham_rows']]
    require(len(set(event_keys)) == len(event_keys) and all((r['demonstration_index'], r['event_step']) in event_keys for r in records), 'event reference set differs')
    for demo, event in event_keys:
        rows = [r for r in records if r['demonstration_index'] == demo and r['event_step'] == event]
        events.append(event_summary(rows, demo, event))
    confusion = {f'{SIGN_NAMES[a]}_to_{SIGN_NAMES[b]}': sum(r['original_sign'] == a and r['changed_sign'] == b for r in records)
                 for a in (-1, 0, 1) for b in (-1, 0, 1)}
    return {**context, 'summary': summarize(records, events), 'sign_transition9': confusion,
            'frames': records, 'events': events,
            'per_demo': [{'demonstration_index': demo,
                **summarize([r for r in records if r['demonstration_index'] == demo],
                            [e for e in events if e['demonstration_index'] == demo])} for demo in range(30)]}


def load_inputs(config_path, binding_path):
    c = load(config_path)
    expected = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arms': list(ARMS),
        'conditions': list(CONDITIONS), 'checkpoint_updates': 500, 'closed_loop_stabilization_steps': 5,
        'source_role': 'meta_support', 'demonstration_indices': list(range(30)), 'reused_models': 18,
        'selected_model_frame_rows_per_condition': 2754, 'new_policy_forward_batches': 0,
        'new_training_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
        'new_HDF5_RGB_initial_private_checkpoint_payload_reads': 0, 'sealed_roles_accessed': False,
        'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c['formal_result'] is False, 'diagnostic cannot be a formal result')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'analysis config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {name}')
    binding = load(binding_path)
    require(binding['config_sha256'] == CONFIG_SHA and binding['formal_result'] is False and
            set(binding['files']) == {'analyzer', 'tests'}, 'analysis code binding differs')
    for name, pin in binding['files'].items():
        require(sha(pin['path']) == pin['sha256'], f'analysis implementation differs: {name}')
    require(resolve(binding['files']['analyzer']['path']).resolve() == Path(__file__).resolve(), 'analyzer path differs')
    original = load(c['inputs']['control_result']['path'])
    audit = load(c['inputs']['control_audit']['path'])
    require(original['status'] == 'complete' and audit['status'] == 'complete_independent_saved_array_audit' and
            original['actual_counts'] == audit['actual_counts_verified'] and
            audit['result']['sha256'] == c['inputs']['control_result']['sha256'], 'parent completion/audit provenance differs')
    require(original['config_sha256'] == c['inputs']['control_design']['sha256'] and
            original['execution_binding_sha256'] == c['inputs']['control_execution_binding']['sha256'], 'parent design provenance differs')
    for name in ('control_design_binding', 'control_execution_binding'):
        for pin in load(c['inputs'][name]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'parent bound implementation differs')
    for record in original['access_records']:
        require(record['task_id'] in TASKS and record['task_role'] == 'source_spi' and
                record['demonstration_role'] == 'meta_support' and record['demonstration_indices'] == list(range(30)) and
                record['source_action0_rows_read'] == 0 and record['simulator_state_access'] is False, 'parent source access differs')
    require(len(original['model_results']) == 18 and
            {(r['task_id'], r['training_seed'], r['arm']) for r in original['model_results']} ==
            {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS}, 'fixed model matrix differs')
    for row in original['model_results']:
        require(row['checkpoint']['optimizer_updates'] == 500, 'endpoint update count differs')
        pin = row['prediction_file']
        require(sha(pin['path']) == pin['sha256'], 'saved readout pin differs')
    return c, original, load(c['inputs']['samples']['path']), binding


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT):
    output = Path(output)
    failed = output.with_name(output.stem + '-failed.json')
    require(not output.exists() and not failed.exists(), 'output already exists; preserve historical artifacts')
    started = time.monotonic()
    rows = []
    try:
        c, original, samples, binding = load_inputs(config_path, binding_path)
        for task in TASKS:
            sample = next(s for s in samples['tasks'] if s['task_id'] == task)
            with np.load(resolve(c['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
                old = {k: z[k] for k in ('targets', 'predictions', 'model_ids', 'demonstration_indices', 'action_steps')}
            require(np.array_equal(np.column_stack((old['demonstration_indices'], old['action_steps'])), sample['source_frame_ids']), 'source frame IDs differ')
            for seed in SEEDS:
                for arm in ARMS:
                    mi = SEEDS.index(seed) * 3 + ARMS.index(arm)
                    require(str(old['model_ids'][mi]) == f'{seed}:{arm}', 'source model order differs')
                    parent = next(r for r in original['model_results'] if (r['task_id'], r['training_seed'], r['arm']) == (task, seed, arm))
                    with np.load(parent['prediction_file']['path'], allow_pickle=False) as z:
                        pred, names = z['predictions'], z['conditions'].tolist()
                    require(names == ['identity', *CONDITIONS] and pred.shape == (3, len(old['targets']), 7) and
                            np.array_equal(pred[0], old['predictions'][mi]), 'saved identity/condition layout differs')
                    for ci, condition in enumerate(CONDITIONS, 1):
                        row = analyze_model(task, seed, arm, condition, old['targets'], pred[0], pred[ci], sample)
                        prior = parent['conditions'][condition]['selected_all']
                        require(row['summary']['cohorts']['all']['n'] == prior['n'] and
                                row['summary']['cohorts']['harmed']['n'] == prior['correct_to_wrong_n'] and
                                row['summary']['cohorts']['repaired']['n'] == prior['wrong_to_correct_n'], 'prior sign-transition totals differ')
                        rows.append(row)
        grouped = []
        for task in TASKS:
            for condition in CONDITIONS:
                for arm in (None, *ARMS):
                    selected = [r for r in rows if r['task_id'] == task and r['condition'] == condition and (arm is None or r['arm'] == arm)]
                    frames = [f for r in selected for f in r['frames']]
                    events = [e for r in selected for e in r['events']]
                    grouped.append({'task_id': task, 'condition': condition, 'arm': arm,
                                    'models': len(selected), **summarize(frames, events)})
        harmed = {condition: [f for row in rows if row['condition'] == condition for f in row['frames'] if f['is_harmed']] for condition in CONDITIONS}
        for condition in CONDITIONS:
            require(len(harmed[condition]) == c['scope']['expected_correct_to_wrong_by_condition'][condition], 'harmed-case count differs')
        keys = lambda case_rows: {(r['task_id'], r['training_seed'], r['arm'], r['demonstration_index'], r['action_step']) for r in case_rows}
        primary, secondary = map(lambda condition: keys(harmed[condition]), CONDITIONS)
        result = {'status': 'complete_readonly_margin_temporal_analysis', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path),
            'binding_sha256': sha(binding_path), 'scope': c['scope'], 'analysis_definition': c['analysis'],
            'input_provenance': c['inputs'], 'code_binding': binding, 'model_condition_results': rows,
            'group_summaries': grouped, 'all_harmed_cases': harmed,
            'condition_overlap': {'primary_harmed_model_frames': len(primary), 'secondary_harmed_model_frames': len(secondary),
                'intersection_model_frames': len(primary & secondary), 'union_model_frames': len(primary | secondary)},
            'actual_new_research_counts': {k: 0 for k in ('policy_forward_batches', 'training_updates', 'backward_calls',
                'environment_calls', 'HDF5_RGB_initial_private_checkpoint_payload_reads')},
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'cpu',
                'elapsed_seconds': time.monotonic() - started, 'launch_argv': sys.argv,
                'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
            'interpretation': 'Raw continuous boundary distances and runs on frozen expert-open selected lags only; no chosen proximity threshold, physical cause, new expert labels or training gate.'}
        with output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        return result
    except Exception as error:
        with failed.open('x') as handle:
            json.dump({'status': 'failed_readonly_analysis', 'error_type': type(error).__name__, 'error': str(error),
                       'completed_model_condition_rows': len(rows), 'actual_new_research_calls': 0}, handle, indent=2)
            handle.write('\n')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.binding, args.output)
    print(json.dumps({'status': result['status'], 'condition_overlap': result['condition_overlap'],
                      'actual_new_research_counts': result['actual_new_research_counts']}))
