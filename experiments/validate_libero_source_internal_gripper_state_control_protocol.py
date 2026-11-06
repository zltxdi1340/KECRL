"""Freeze/validate causal gripper-state input controls; no Policy or HDF5 imports.

The sample manifest is derived only from existing target/prediction archives.
Operators below are exercised on synthetic public-state arrays during freezing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_source_internal_event_phase_direction_translation as phase

DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-gripper-state-local-control-diagnostic_v1.json'
CONFIG_SHA = 'bf38389d06d2d43c74991365d4763cc0cbbe94ee8517ef1c9cf51ca4d4d40206'
CONDITIONS = ('identity', 'hold_event_preaction_gripper', 'lag1_gripper')
TASKS, SEEDS, ARMS = phase.TASKS, phase.SEEDS, phase.ARMS
require, sha, load, resolve = phase.require, phase.sha, phase.load, phase.resolve


def build_task_samples(task, data):
    target, ids, times = data['targets'], data['demonstration_indices'], data['action_steps']
    require([str(v) for v in data['model_ids']] == [f'{s}:{a}' for s in SEEDS for a in ARMS], 'model matrix differs')
    require(target.shape == (len(ids), 7) and target.dtype == np.float32 and data['predictions'].dtype == np.float32 and
            data['predictions'].shape == (9, len(target), 7) and np.isfinite(target).all() and np.isfinite(data['predictions']).all() and
            np.max(np.abs(target)) <= 1 and np.max(np.abs(data['predictions'])) <= 1,
            'cached predictions/targets differ')
    assigned = phase.assign_events(target, ids, times, data['mask_near_open'], 'open')
    selection = assigned['phases']['post'] & (target[:, 6] < 0)
    lookup = {(int(d), int(t)): i for i, (d, t) in enumerate(zip(ids, times))}
    require(len(lookup) == len(target), 'duplicate source frame IDs')
    windows = []
    for demo in sorted(set(ids.tolist())):
        n = int((ids == demo).sum())
        for start in range(1, n + 1, 10):
            windows.append({'demonstration_index': int(demo), 'start': start, 'valid_length': min(10, n + 1 - start)})
    window_lookup = {(w['demonstration_index'], w['start']): i for i, w in enumerate(windows)}
    selected = []
    for i in np.flatnonzero(selection):
        demo, t, event = int(ids[i]), int(times[i]), int(assigned['event_step'][i])
        require(2 <= event < t and 1 <= t - event <= 5 and (demo, event) in lookup and (demo, t - 1) in lookup,
                'anchor must be observed causal past in same demo')
        window_index = window_lookup[demo, ((t - 1) // 10) * 10 + 1]
        selected.append({'demonstration_index': demo, 'action_step': t, 'source_row': int(i),
                         'event_step': event, 'lag': t - event, 'event_anchor_row': int(lookup[demo, event]),
                         'lag1_anchor_row': int(lookup[demo, t - 1]), 'canonical_batch': window_index // 8,
                         'original_predicted_gripper_signs': np.sign(data['predictions'][:, i, 6]).astype(int).tolist()})
    require(bool(selected), 'frozen local control requires post-open frames')
    repeat_batch = min(r['canonical_batch'] for r in selected)
    repeat_windows = windows[repeat_batch * 8:(repeat_batch + 1) * 8]
    at_rows = np.flatnonzero(assigned['phases']['at']).astype(int).tolist()
    model_counts = []
    for model, name in enumerate(data['model_ids']):
        signs = np.sign(data['predictions'][model, selection, 6])
        model_counts.append({'model_id': str(name), 'original_correct_n': int((signs < 0).sum()),
                             'original_wrong_close_n': int((signs > 0).sum()), 'original_neutral_n': int((signs == 0).sum())})
    return {'task_id': task, 'source_frames': len(target), 'selected_count': len(selected),
            'source_frame_ids': [[int(d), int(t)] for d, t in zip(ids, times)],
            'selected': selected, 'at_event_sham_rows': at_rows, 'unchanged_valid_rows': len(target) - len(selected),
            'canonical_windows': windows, 'canonical_batches': (len(windows) + 7) // 8,
            'repeat_batch': repeat_batch, 'repeat_valid_rows': sum(w['valid_length'] for w in repeat_windows),
            'repeat_network_slots': len(repeat_windows) * 10, 'model_labels': model_counts}


def replace_gripper_states(original, sample, condition):
    values = np.asarray(original)
    require(condition in CONDITIONS and values.shape == (sample['source_frames'], 2) and
            values.dtype == np.float32 and np.isfinite(values).all(), 'public gripper state interface differs')
    result = values.copy()
    frame_ids = sample['source_frame_ids']
    require(len(frame_ids) == len(values), 'source frame metadata differs')
    seen = set()
    for row in sample['selected']:
        i, event, lag1 = row['source_row'], row['event_anchor_row'], row['lag1_anchor_row']
        require(i not in seen and 0 <= i < len(values) and 0 <= event < i and 0 <= lag1 < i and
                2 <= row['event_step'] < row['action_step'] and
                1 <= row['lag'] <= 5 and row['action_step'] - row['event_step'] == row['lag'],
                'invalid or future gripper anchor')
        require(frame_ids[i] == [row['demonstration_index'], row['action_step']] and
                frame_ids[event] == [row['demonstration_index'], row['event_step']] and
                frame_ids[lag1] == [row['demonstration_index'], row['action_step'] - 1],
                'gripper anchor must refer to exact same-demo source history')
        seen.add(i)
        if condition != 'identity':
            anchor = event if condition == CONDITIONS[1] else lag1
            result[i] = values[anchor]  # Always immutable original history, never a previously replaced row.
    return result


def condition_metrics(original, changed, target, selected_mask):
    original = np.asarray(original)
    changed = np.asarray(changed)
    target = np.asarray(target)
    require(selected_mask.dtype == bool and selected_mask.shape == (len(target),), 'metric selection differs')
    before, _ = phase.parent.row_errors(original, target)
    after, _ = phase.parent.row_errors(changed, target)
    n = int(selected_mask.sum())
    if not n:
        return {'n': 0, 'sign_error_delta': None, 'translation_mse_delta': None,
                'gripper_mse_delta': None, 'correct_to_wrong_n': 0, 'wrong_to_correct_n': 0}
    old_wrong = np.sign(original[:, 6]) != np.sign(target[:, 6])
    new_wrong = np.sign(changed[:, 6]) != np.sign(target[:, 6])
    return {'n': n,
            'sign_error_delta': float(after['gripper_sign_error'][selected_mask].mean() - before['gripper_sign_error'][selected_mask].mean()),
            'translation_mse_delta': float(after['translation_mse'][selected_mask].mean() - before['translation_mse'][selected_mask].mean()),
            'gripper_mse_delta': float(after['gripper_mse'][selected_mask].mean() - before['gripper_mse'][selected_mask].mean()),
            'correct_to_wrong_n': int((selected_mask & ~old_wrong & new_wrong).sum()),
            'wrong_to_correct_n': int((selected_mask & old_wrong & ~new_wrong).sum())}


def descriptive_joint_flag(seed_metrics):
    require(len(seed_metrics) == 3, 'retain all3 seeds')
    if any(r['sign_error_delta'] is None or r['translation_mse_delta'] is None for r in seed_metrics):
        return None
    return bool(all(r['sign_error_delta'] < 0 and r['translation_mse_delta'] <= 0 for r in seed_metrics))


def validate(config_path=DEFAULT_CONFIG, binding_path=None):
    c = load(config_path)
    expected = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arms': list(ARMS),
                'checkpoint_updates': 500, 'closed_loop_stabilization_steps': 5, 'component': 'gripper_states',
                'conditions': list(CONDITIONS), 'source_role': 'meta_support', 'demonstration_indices': list(range(30)),
                'memory_mode': 'cnn_only', 'sequence_length': 10, 'stride': 10, 'batch_size': 8,
                'planned_main_forward_batches': 2457, 'planned_repeat_forward_batches': 54,
                'planned_main_valid_prediction_rows': 187758, 'planned_main_network_prediction_slots': 196020,
                'selected_frames_by_task': {'0': 149, '1': 157}, 'new_optimizer_updates': 0,
                'new_backward_calls': 0, 'new_environment_calls': 0, 'new_initial_or_private_state_reads': 0,
                'qualification_validation_heldout_final_access': False, 'checkpoint_selected': False,
                'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c.get('formal_result') is False and c.get('research_run_executed') is False, 'design must not claim an executed result')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scientific scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'frozen scientific design hash differs')
    for name, pin in c['inputs'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'input pin differs: {name}')
    review = load(resolve(c['inputs']['input_sensitivity_review']['path']))
    for name, pin in review['inputs'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'historical review pin differs: {name}')
    sample = load(resolve(c['inputs']['sample_manifest']['path']))
    require(len(sample['tasks']) == 2 and [s['task_id'] for s in sample['tasks']] == list(TASKS), 'sample matrix differs')
    original = load(resolve(c['inputs']['original_frame_result']['path']))
    require(original['status'] == 'complete' and original['scope']['source_role'] == 'meta_support' and
            original['scope']['checkpoint_updates'] == 500 and original['actual_counts']['checkpoint_loads'] == 18,
            'original model/input provenance differs')
    parent_binding = load(resolve(c['inputs']['original_frame_binding']['path']))
    for name, pin in parent_binding['files'].items():
        require(sha(resolve(pin['path'])) == pin['sha256'], f'parent implementation differs: {name}')
    endpoints = load(resolve(c['inputs']['original_frame_design']['path']))['endpoints']
    require(c['endpoints'] == endpoints and len(endpoints) == 18, 'all18 fixed endpoints must be retained')
    repeat_valid = repeat_slots = 0
    for task in TASKS:
        with np.load(resolve(c['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
            data = {k: z[k] for k in ('targets', 'predictions', 'model_ids', 'demonstration_indices', 'action_steps', 'mask_near_open')}
        require(data['targets'].shape == ({0: 2895, 1: 4059}[task], 7), 'source expert coverage differs')
        rebuilt = build_task_samples(task, data)
        declared = next(s for s in sample['tasks'] if s['task_id'] == task)
        require(rebuilt == declared, 'sample/anchor plan changed')
        require(rebuilt['selected_count'] == c['scope']['selected_frames_by_task'][str(task)], 'selected target coverage differs')
        repeat_valid += rebuilt['repeat_valid_rows'] * 9 * len(CONDITIONS)
        repeat_slots += rebuilt['repeat_network_slots'] * 9 * len(CONDITIONS)
    require(repeat_valid == c['scope']['planned_repeat_valid_prediction_rows'] and
            repeat_slots == c['scope']['planned_repeat_network_prediction_slots'], 'repeat budget differs')
    binding = None
    if binding_path is not None:
        binding = load(binding_path)
        require(binding['config_sha256'] == CONFIG_SHA and binding['formal_result'] is False and
                set(binding['files']) == {'validator', 'tests', 'protocol'}, 'design/operator binding differs')
        for name, pin in binding['files'].items():
            require(sha(resolve(pin['path'])) == pin['sha256'], f'design/operator code differs: {name}')
    return c, sample, review, binding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path)
    args = parser.parse_args()
    c, sample, review, binding = validate(args.config, args.binding)
    print(json.dumps({'status': 'design_and_samples_validated_not_executed',
                      'planned_main_batches': 2457, 'planned_repeat_batches': 54,
                      'selected_frames': {str(s['task_id']): s['selected_count'] for s in sample['tasks']},
                      'design_operator_binding_verified': binding is not None,
                      'actual_policy_calls': 0, 'actual_HDF5_RGB_initial_checkpoint_payload_reads': 0}))
