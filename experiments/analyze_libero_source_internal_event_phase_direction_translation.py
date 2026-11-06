"""Read saved expert predictions: event side, sign direction and translation error.

No model, HDF5, RGB, simulator or checkpoint is opened. Event1 remains unknown
because action0 was not archived; boundary rows are retained without guessing.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import validate_libero_source_internal_fixed500_expert_frame_protocol as parent

DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-event-phase-direction-translation-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-event-phase-direction-translation-execution-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-event-phase-direction-translation-20261103-v1.json')
CONFIG_SHA = '162a3248118b94d3c83bb9d898c8bdddbae5210a929795dfd02002966c30d84f'
TASKS, SEEDS, ARMS = parent.TASKS, parent.SEEDS, parent.ARMS
PHASES = ('pre', 'at', 'post', 'unresolved_boundary')
CONTEXTS = ('all', 'without_opposite_transition_overlap')
DIRECTIONS = ('expert_open_pred_close', 'expert_close_pred_open')
ALL_CASES = (*DIRECTIONS, 'neutral_involved_mismatch', 'sign_correct')
require, sha, load, resolve = parent.require, parent.sha, parent.load, parent.resolve


def stats(values):
    total = len(values)
    valid = [float(v) for v in values if v is not None]
    require(all(np.isfinite(valid)), 'nonfinite summary value')
    return {'total': total, 'n': len(valid), 'missing': total - len(valid),
            'mean': float(np.mean(valid)) if valid else None, 'median': float(np.median(valid)) if valid else None,
            'minimum': min(valid) if valid else None, 'maximum': max(valid) if valid else None,
            'positive_count': sum(v > 0 for v in valid)}


def assign_events(targets, demo_ids, times, saved_mask, kind):
    require(kind in ('open', 'close'), 'unknown event kind')
    require(len(targets) == len(demo_ids) == len(times) == len(saved_mask) and saved_mask.dtype == bool,
            'event frame interface differs')
    event_step = np.full(len(times), -1, np.int32)
    lag = np.full(len(times), 99, np.int8)
    known_events = []
    for demo in sorted(set(demo_ids.tolist())):
        rows = np.flatnonzero(demo_ids == demo)
        t = times[rows]
        require(np.array_equal(t, np.arange(1, len(rows) + 1)), 'demo action IDs must be contiguous from1')
        signs = np.sign(targets[rows, 6])
        transition = ((signs[1:] < 0) & (signs[:-1] >= 0)) if kind == 'open' else ((signs[1:] > 0) & (signs[:-1] <= 0))
        centers = t[np.flatnonzero(transition) + 1]
        known_events.extend({'demonstration_index': int(demo), 'event_step': int(c)} for c in centers)
        for row in rows:
            if len(centers):
                nearest = int(centers[np.argmin(np.abs(centers - times[row]))])
                if abs(int(times[row]) - nearest) <= 5:
                    require(saved_mask[row], 'known event window missing from saved mask')
                    event_step[row] = nearest
                    lag[row] = int(times[row]) - nearest
            if saved_mask[row] and event_step[row] < 0:
                require(times[row] <= 6, 'nonboundary window frame has no reconstructible event')
    phases = {'pre': saved_mask & (event_step >= 2) & (lag < 0),
              'at': saved_mask & (event_step >= 2) & (lag == 0),
              'post': saved_mask & (event_step >= 2) & (lag > 0),
              'unresolved_boundary': saved_mask & (event_step < 0)}
    require(np.array_equal(sum(m.astype(int) for m in phases.values()), saved_mask.astype(int)), 'event phase partition differs')
    return {'event_step': event_step, 'lag': lag, 'phases': phases, 'known_events': known_events}


def direction_masks(truth, guessed):
    result = {'expert_open_pred_close': (truth < 0) & (guessed > 0),
              'expert_close_pred_open': (truth > 0) & (guessed < 0),
              'neutral_involved_mismatch': (truth != guessed) & ((truth == 0) | (guessed == 0)),
              'sign_correct': truth == guessed}
    require(np.all(sum(m.astype(int) for m in result.values()) == 1), 'sign-direction partition differs')
    return result


def brief_summary(translation, truth, cases, selection):
    n = int(selection.sum())
    return {'n': n, 'translation_mse': float(translation[selection].mean()) if n else None,
            'sign_error_rate': float((selection & ~cases['sign_correct']).sum() / n) if n else None,
            'expert_sign_counts': {str(s): int((selection & (truth == s)).sum()) for s in (-1, 0, 1)},
            'direction_counts': {k: int((selection & v).sum()) for k, v in cases.items()},
            'direction_translation_mse': {k: float(translation[selection & v].mean()) if (selection & v).any() else None
                                          for k, v in cases.items()}}


def phase_summary(translation, truth, cases, selection, demo_ids):
    result = brief_summary(translation, truth, cases, selection)
    details = {}
    for direction, sign in zip(DIRECTIONS, (-1, 1)):
        wrong = selection & cases[direction]
        correct = selection & cases['sign_correct'] & (truth == sign)
        wrong_stats, correct_stats = stats(translation[wrong].tolist()), stats(translation[correct].tolist())
        denominator = int((selection & (truth == sign)).sum())
        by_demo = []
        for demo in sorted(set(demo_ids.tolist())):
            w, c = wrong & (demo_ids == demo), correct & (demo_ids == demo)
            by_demo.append({'demonstration_index': int(demo), 'wrong_n': int(w.sum()), 'correct_n': int(c.sum()),
                            'translation_difference': float(translation[w].mean() - translation[c].mean()) if w.any() and c.any() else None})
        details[direction] = {'wrong_n': int(wrong.sum()), 'expert_class_n': denominator,
                              'phase_fraction': int(wrong.sum()) / result['n'] if result['n'] else None,
                              'conditional_error_rate': int(wrong.sum()) / denominator if denominator else None,
                              'wrong_translation': wrong_stats, 'correct_same_command_translation': correct_stats,
                              'translation_difference': wrong_stats['mean'] - correct_stats['mean'] if wrong.any() and correct.any() else None,
                              'per_demo': by_demo, 'per_demo_difference': stats([r['translation_difference'] for r in by_demo])}
    result['directions'] = details
    return result


def cross_seed_summary(translation, truth, cases_by_seed, selection, ids, times, assignment, direction):
    wrong = np.stack([cases[direction] & selection for cases in cases_by_seed])
    sign = -1 if direction == DIRECTIONS[0] else 1
    correct = np.stack([cases['sign_correct'] & (truth == sign) & selection for cases in cases_by_seed])
    union, intersection = wrong.any(0), wrong.all(0)
    mixed = union & correct.any(0)
    contrasts = []
    def frame_info(i):
        return {'demonstration_index': int(ids[i]), 'action_step': int(times[i]),
                'event_step': int(assignment['event_step'][i]) if assignment['event_step'][i] >= 2 else None,
                'lag': int(assignment['lag'][i]) if assignment['event_step'][i] >= 2 else None}
    for i in np.flatnonzero(mixed):
        ew, ec = float(translation[wrong[:, i], i].mean()), float(translation[correct[:, i], i].mean())
        contrasts.append({**frame_info(i), 'wrong_seeds': [s for s, v in zip(SEEDS, wrong[:, i]) if v],
                          'correct_seeds': [s for s, v in zip(SEEDS, correct[:, i]) if v],
                          'wrong_translation_mse': ew, 'correct_translation_mse': ec, 'difference': ew - ec})
    pairwise = []
    for a, b in itertools.combinations(range(3), 2):
        n_union = int((wrong[a] | wrong[b]).sum())
        n_intersection = int((wrong[a] & wrong[b]).sum())
        pairwise.append({'seeds': [SEEDS[a], SEEDS[b]], 'intersection_n': n_intersection, 'union_n': n_union,
                         'jaccard': n_intersection / n_union if n_union else None})
    common = [{**frame_info(i), 'translation_mse_by_seed': [float(v) for v in translation[:, i]]} for i in np.flatnonzero(intersection)]
    return {'wrong_n_by_seed': wrong.sum(1).astype(int).tolist(), 'all_three_intersection_n': int(intersection.sum()),
            'any_seed_union_n': int(union.sum()), 'all_three_jaccard': int(intersection.sum()) / int(union.sum()) if union.any() else None,
            'pairwise': pairwise, 'common_frames': common, 'same_frame_mixed_n': len(contrasts),
            'same_frame_translation_contrasts': contrasts, 'same_frame_translation_difference': stats([r['difference'] for r in contrasts])}


def analyze_task(task, data):
    targets, ids, times = data['targets'], data['demonstration_indices'], data['action_steps']
    model_ids = [str(v) for v in data['model_ids']]
    expected_ids = [f'{s}:{a}' for s in SEEDS for a in ARMS]
    require(model_ids == expected_ids and data['predictions'].shape == (9, len(targets), 7), 'model matrix differs')
    truth = np.sign(targets[:, 6])
    model_info = {}
    for i, model_id in enumerate(model_ids):
        seed, arm = model_id.split(':')
        errors, _ = parent.row_errors(data['predictions'][i], targets)
        model_info[int(seed), arm] = (errors['translation_mse'], direction_masks(truth, np.sign(data['predictions'][i, :, 6])))
    phase_rows, lag_rows, cross_rows, event_info = [], [], [], []
    for event in ('open', 'close'):
        opposite = 'close' if event == 'open' else 'open'
        saved, overlap = data['mask_near_' + event], data['mask_near_' + opposite]
        assigned = assign_events(targets, ids, times, saved, event)
        event_info.append({'task_id': task, 'event': event, 'known_events': assigned['known_events'],
                           'saved_window_n': int(saved.sum()), 'opposite_overlap_n': int((saved & overlap).sum()),
                           'phase_counts': {k: int(v.sum()) for k, v in assigned['phases'].items()}})
        for context in CONTEXTS:
            eligible = np.ones(len(targets), bool) if context == 'all' else ~overlap
            for phase in PHASES:
                selection = assigned['phases'][phase] & eligible
                for seed in SEEDS:
                    for arm in ARMS:
                        translation, cases = model_info[seed, arm]
                        phase_rows.append({'task_id': task, 'training_seed': seed, 'arm': arm, 'event': event,
                                           'context': context, 'phase': phase,
                                           **phase_summary(translation, truth, cases, selection, ids)})
                for arm in ARMS:
                    translation = np.stack([model_info[s, arm][0] for s in SEEDS])
                    cases = [model_info[s, arm][1] for s in SEEDS]
                    for direction in DIRECTIONS:
                        row = {'task_id': task, 'arm': arm, 'event': event, 'context': context, 'phase': phase, 'direction': direction,
                               **cross_seed_summary(translation, truth, cases, selection, ids, times, assigned, direction)}
                        values = [next(r for r in reversed(phase_rows) if r['training_seed'] == s and r['arm'] == arm and
                                       r['event'] == event and r['context'] == context and r['phase'] == phase)
                                  ['directions'][direction]['translation_difference'] for s in SEEDS]
                        row['within_seed_translation_differences'] = values
                        row['all_three_positive_translation_association'] = None if any(v is None for v in values) else all(v > 0 for v in values)
                        cross_rows.append(row)
            for lag in range(-5, 6):
                selection = saved & eligible & (assigned['lag'] == lag) & (assigned['event_step'] >= 2)
                for seed in SEEDS:
                    for arm in ARMS:
                        translation, cases = model_info[seed, arm]
                        lag_rows.append({'task_id': task, 'training_seed': seed, 'arm': arm, 'event': event,
                                         'context': context, 'lag': lag, **brief_summary(translation, truth, cases, selection)})
    return {'phase_rows': phase_rows, 'lag_rows': lag_rows, 'cross_seed_rows': cross_rows, 'event_info': event_info}


def paired_arms(rows):
    index = {(r['task_id'], r['training_seed'], r['arm'], r['event'], r['context'], r['phase']): r for r in rows}
    output = []
    for task, seed, event, context, phase in itertools.product(TASKS, SEEDS, ('open', 'close'), CONTEXTS, PHASES):
        for reference in (ARMS[0], ARMS[2]):
            a = index[task, seed, ARMS[1], event, context, phase]
            b = index[task, seed, reference, event, context, phase]
            require(a['n'] == b['n'] and a['expert_sign_counts'] == b['expert_sign_counts'], 'paired phase frames differ')
            output.append({'task_id': task, 'training_seed': seed, 'left_arm': ARMS[1], 'right_arm': reference,
                           'event': event, 'context': context, 'phase': phase, 'n': a['n'],
                           'translation_mse_delta': a['translation_mse'] - b['translation_mse'] if a['n'] else None,
                           'sign_error_rate_delta': a['sign_error_rate'] - b['sign_error_rate'] if a['n'] else None})
    return output


def load_inputs(config_path, binding_path):
    config = load(config_path)
    require(config['formal_result'] is False and config['analysis']['phases'] == list(PHASES) and
            config['analysis']['radius'] == 5 and config['scope']['new_policy_forward_batches'] == 0 and
            config['scope']['checkpoint_updates'] == 500, 'readonly definition differs')
    require(sha(config_path) == CONFIG_SHA, 'frozen config hash differs')
    binding = load(binding_path)
    require(binding['formal_result'] is False and binding['config_sha256'] == CONFIG_SHA and
            set(binding['files']) == {'analyzer', 'tests', 'protocol'}, 'execution binding differs')
    require(resolve(binding['files']['analyzer']['path']).resolve() == Path(__file__).resolve(), 'bound analyzer differs')
    for name, pin in {**config['inputs'], **binding['files']}.items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'input/binding hash differs: {name}')
    source = load(resolve(config['inputs']['source_result']['path']))
    audit = load(resolve(config['inputs']['source_audit']['path']))
    require(source['status'] == 'complete' and audit['status'] == 'passed' and
            audit['result_sha256'] == config['inputs']['source_result']['sha256'], 'source run/audit differs')
    require(source['scope']['source_role'] == 'meta_support' and source['scope']['checkpoint_updates'] == 500 and
            source['actual_counts']['main_valid_prediction_rows'] == 62586, 'source role/budget differs')
    old_binding = load(resolve(config['inputs']['source_binding']['path']))
    for name, pin in old_binding['files'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'parent code differs: {name}')
    return config, binding, source


def run(config_path, binding_path, output):
    output = Path(output)
    require(not output.exists(), 'output already exists; preserve old artifacts')
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    config, binding, source = load_inputs(config_path, binding_path)
    result = {'phase_rows': [], 'lag_rows': [], 'cross_seed_rows': [], 'event_info': []}
    for task in TASKS:
        pin = config['inputs'][f'task{task}_archive']
        require(any(a['task_id'] == task and a['sha256'] == pin['sha256'] for a in source['archives']), 'archive source identity differs')
        with np.load(resolve(pin['path']), allow_pickle=False) as archive:
            data = {k: archive[k] for k in ('targets', 'predictions', 'model_ids', 'demonstration_indices', 'action_steps',
                                            'mask_near_open', 'mask_near_close')}
        require(data['targets'].shape == ({0: 2895, 1: 4059}[task], 7) and data['targets'].dtype == np.float32 and
                data['predictions'].dtype == np.float32 and sorted(set(data['demonstration_indices'].tolist())) == list(range(30)),
                'archive frame roles/shape/dtype differ')
        part = analyze_task(task, data)
        for k in result:
            result[k].extend(part[k])
    result['paired_arm_rows'] = paired_arms(result['phase_rows'])
    result.update({'status': 'complete', 'formal_result': False, 'read_only_analysis': True, 'scope': config['scope'],
                   'analysis_definition': config['analysis'],
                   'counts': {'phase_rows': len(result['phase_rows']), 'lag_rows': len(result['lag_rows']),
                              'cross_seed_rows': len(result['cross_seed_rows']), 'paired_arm_rows': len(result['paired_arm_rows']),
                              'existing_unique_expert_frames': 6954, 'existing_model_prediction_rows': 62586,
                              'new_policy_training_environment_HDF5_RGB_initial_checkpoint_calls': 0},
                   'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path),
                                  'binding_path': str(binding_path), 'binding_sha256': sha(binding_path), 'inputs': config['inputs'],
                                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
                   'runtime': {'started_at_utc': started_utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                               'elapsed_seconds': time.monotonic() - started, 'python': sys.version.split()[0],
                               'numpy': np.__version__, 'device': 'CPU numerical analysis; no model device'}})
    with output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config, args.binding, args.output)
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'sha256': sha(args.output),
                      'counts': result['counts'], 'runtime': result['runtime']}))
