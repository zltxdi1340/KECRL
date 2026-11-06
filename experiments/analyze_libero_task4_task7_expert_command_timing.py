"""Read expert-state command timing from a frozen feature archive, without models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-expert-command-timing-diagnostic_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-expert-command-timing-20261102-v1.json")
SCOPE = {"task_ids": [4, 7], "training_seeds": [20261022, 20261023, 20261024],
    "arm": "source_baseline", "checkpoint_updates": 500, "role": "meta_support", "demo_ids": list(range(30)),
    "cached_source_rows": 8630, "common_lag_rows_per_seed_across_tasks": 8030,
    "gripper_lag_scalar_comparisons": 168630, "new_model_forward_or_backward_calls": 0,
    "new_optimizer_updates": 0, "new_environment_interactions": 0, "new_hdf5_reads": 0,
    "qualification_accessed": False, "validation_accessed": False, "heldout_accessed": False, "final_evaluation_accessed": False}
ANALYSIS = {"lags": [-5, -2, -1, 0, 1, 2, 5], "lag_definition": "prediction_at_t_against_expert_at_t_plus_lag",
    "common_interior": "exclude_first_and_last5_cached_rows_per_demo_for_every_lag",
    "lag_ties": "lowest_mse_then_smallest_absolute_lag_then_negative",
    "event_match_radius": 10, "event_matching": "chronological_expert_events; nearest_unused_same_direction_prediction; earlier_tie",
    "event_offset": "predicted_switch_index_minus_expert_switch_index; positive_is_late",
    "first_cached_row_not_an_event": True, "censored_windows_reported": True,
    "expert_labels_are_commands_not_physical_grasp_phases": True,
    "no_causal_dataset_alignment_change_or_future_label_training": True, "no_significance_test": True}


def require(condition, message):
    if not condition: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()


def load_protocol(path):
    config = json.loads(path.read_text())
    require(config.get('check_type') == 'libero_task4_task7_expert_command_timing_diagnostic' and config.get('formal_result') is False,
            'timing protocol type differs')
    require(config.get('scope') == SCOPE and config.get('analysis') == ANALYSIS, 'timing frozen scope or analysis differs')
    require(config['runner_sha256'] == sha(Path(__file__)), 'timing runner hash differs')
    require(set(config['inputs']) == {'parent_result', 'parent_config', 'parent_code', 'support4', 'support7', 'overnight_plan'}, 'timing input roles differ')
    for row in config['inputs'].values(): require(sha(Path(row['path'])) == row['sha256'], 'timing input hash differs')
    return config


def events(values, action_ids, direction):
    require(direction in ('open', 'close'), 'invalid timing direction')
    require(values.ndim == 1 and values.shape == action_ids.shape and np.isfinite(values).all(), 'invalid timing commands')
    require(np.array_equal(action_ids, np.arange(1, len(values) + 1)), 'timing action indices missing or shifted')
    signs = np.sign(values)
    mask = (signs[1:] < 0) & (signs[:-1] >= 0) if direction == 'open' else (signs[1:] > 0) & (signs[:-1] <= 0)
    return action_ids[1:][mask].tolist()


def match_events(expert, predicted, last_action, direction):
    available, rows = set(predicted), []
    for event in expert:
        eligible = [value for value in available if abs(value - event) <= 10]
        match = min(eligible, key=lambda value: (abs(value - event), value)) if eligible else None
        if match is not None: available.remove(match)
        rows.append({'direction': direction, 'expert_step': event, 'predicted_step': match,
                     'offset': match - event if match is not None else None,
                     'window_censored': event - 10 < 2 or event + 10 > last_action})
    return rows, sorted(available)


def lag_arrays(prediction, truth, open_mask):
    require(prediction.shape == truth.shape == open_mask.shape and prediction.ndim == 1, 'lag arrays differ')
    require(open_mask.dtype == np.bool_, 'lag mask must be boolean')
    require(len(truth) > 10 and np.isfinite(prediction).all() and np.isfinite(truth).all(), 'lag sequence too short or nonfinite')
    common = np.arange(5, len(truth) - 5)
    result = {}
    for lag in ANALYSIS['lags']:
        delta = prediction[common].astype(np.float64) - truth[common + lag].astype(np.float64)
        wrong = np.sign(prediction[common]) != np.sign(truth[common + lag])
        result[str(lag)] = {'squared_errors': delta ** 2, 'wrong_sign': wrong, 'open_mask_at_original_t': open_mask[common]}
    return result


def summarize_events(rows):
    valid = [row['offset'] for row in rows if row['offset'] is not None]
    return {'expert_events': len(rows), 'matched': len(valid), 'unmatched_within_radius': len(rows) - len(valid),
            'censored_event_windows': sum(row['window_censored'] for row in rows),
            'offset_mean': float(np.mean(valid)) if valid else None, 'offset_median': float(np.median(valid)) if valid else None,
            'early': sum(value < 0 for value in valid), 'on_time': sum(value == 0 for value in valid), 'late': sum(value > 0 for value in valid)}


def analyze(config_path, output):
    require(not output.exists(), 'timing output already exists')
    config = load_protocol(config_path)
    started = time.perf_counter()
    parent = json.loads(Path(config['inputs']['parent_result']['path']).read_text())
    require(parent['status'] == 'complete' and parent['scope']['demonstration_role'] == 'meta_support', 'timing parent not complete support diagnostic')
    require(parent['scope']['checkpoint_optimizer_updates'] == 500 and parent['scope']['task_ids'] == [4, 7] and
            parent['scope']['training_seeds'] == SCOPE['training_seeds'], 'timing parent baseline matrix differs')
    task_results, total_comparisons = {}, 0
    for task in SCOPE['task_ids']:
        require(config['inputs'][f'support{task}']['sha256'] == parent['task_results'][str(task)]['archive']['sha256'], 'timing support archive provenance differs')
        with np.load(config['inputs'][f'support{task}']['path'], allow_pickle=False) as cache:
            demo_ids, ids, targets = cache['demo_ids'], cache['action_ids'], cache['target_actions'][:, 6]
            require(set(demo_ids.tolist()) == set(range(30)), 'timing demonstration role differs')
            open_mask = cache['command_masks'][:, cache['stratum_names'].tolist().index('around_open_transition')]
            seed_results = {}
            for seed in SCOPE['training_seeds']:
                prediction = cache[f'seed{seed}__predictions'][:, 6]
                lag_parts = {str(lag): [] for lag in ANALYSIS['lags']}
                event_rows, extra_events = [], []
                for demo in range(30):
                    selected = demo_ids == demo
                    p, y, positions = prediction[selected], targets[selected], ids[selected]
                    scores = lag_arrays(p, y, open_mask[selected])
                    for lag, values in scores.items(): lag_parts[lag].append(values)
                    for direction in ('open', 'close'):
                        matched, extra = match_events(events(y, positions, direction), events(p, positions, direction), int(positions[-1]), direction)
                        event_rows.extend({'demo': demo, **row} for row in matched)
                        extra_events.extend({'demo': demo, 'direction': direction, 'predicted_step': value} for value in extra)
                lag_metrics = {}
                for lag, parts in lag_parts.items():
                    errors = np.concatenate([part['squared_errors'] for part in parts])
                    wrong = np.concatenate([part['wrong_sign'] for part in parts])
                    local = np.concatenate([part['open_mask_at_original_t'] for part in parts])
                    lag_metrics[lag] = {'common_rows': len(errors), 'gripper_mse': float(errors.mean()), 'sign_error': float(wrong.mean()),
                        'open_window_common_rows': int(local.sum()), 'open_window_gripper_mse': float(errors[local].mean()) if local.any() else None,
                        'open_window_sign_error': float(wrong[local].mean()) if local.any() else None}
                    total_comparisons += len(errors)
                best = min(ANALYSIS['lags'], key=lambda lag: (lag_metrics[str(lag)]['gripper_mse'], abs(lag), lag))
                seed_results[str(seed)] = {'lag_metrics': lag_metrics, 'descriptive_minimum_mse_lag': best,
                    'events': event_rows, 'unmatched_predicted_events': extra_events,
                    'event_summary': {direction: summarize_events([row for row in event_rows if row['direction'] == direction]) for direction in ('open', 'close')},
                    'uncensored_event_summary': {direction: summarize_events([row for row in event_rows if row['direction'] == direction and not row['window_censored']]) for direction in ('open', 'close')}}
            task_results[str(task)] = {'seed_results': seed_results}
    require(total_comparisons == 168630, 'timing actual lag budget differs')
    return {'check_type': config['check_type'], 'status': 'complete', 'formal_result': False, 'scope': SCOPE, 'analysis': ANALYSIS,
        'runtime': {'elapsed_seconds': time.perf_counter() - started, 'lag_scalar_comparisons': total_comparisons,
                    'new_model_calls': 0, 'new_optimizer_or_environment_steps': 0, 'python': sys.version.split()[0], 'numpy': np.__version__},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
                       'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': config['inputs']},
        'task_results': task_results, 'interpretation': {'lag_minimum_not_new_alignment_rule': True, 'no_future_label_training': True,
            'expert_forced_command_timing_not_closed_loop_grasp_phase': True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = analyze(args.config, args.output)
    with args.output.open('x') as f: json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__': main()
