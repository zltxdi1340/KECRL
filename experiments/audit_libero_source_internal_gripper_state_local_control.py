"""Independent saved-array audit: no Policy, HDF5 or simulator imports/calls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ('identity', 'hold_event_preaction_gripper', 'lag1_gripper')
SEEDS = (20261022, 20261023, 20261024)
ARMS = ('A_original', 'B_fomaml', 'C_pooled_bc')
DEFAULT_RESULT = Path('/tmp/libero-source-internal-gripper-state-local-control-20261104-v1.json')
DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-gripper-state-local-control-independent-audit-20261104-v1.json')


def require(value, message):
    if not value:
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
    return json.loads(resolve(path).read_text())


def scalar_summary(before, after, target, mask):
    n = int(mask.sum())
    old_error = (before.astype(np.float64) - target.astype(np.float64)) ** 2
    new_error = (after.astype(np.float64) - target.astype(np.float64)) ** 2
    old_wrong = np.sign(before[:, 6]) != np.sign(target[:, 6])
    new_wrong = np.sign(after[:, 6]) != np.sign(target[:, 6])
    original, changed = {}, {}
    for name, columns in [('translation_mse', slice(0, 3)), ('rotation_mse', slice(3, 6)),
                          ('gripper_mse', slice(6, 7)), ('mse', slice(0, 7))]:
        original[name] = float(old_error[mask, columns].mean()) if n else None
        changed[name] = float(new_error[mask, columns].mean()) if n else None
    original['gripper_sign_error'] = float(old_wrong[mask].mean()) if n else None
    changed['gripper_sign_error'] = float(new_wrong[mask].mean()) if n else None
    difference = {k: changed[k] - original[k] if n else None for k in original}
    response = after.astype(np.float64) - before.astype(np.float64)
    norms = np.sqrt((response[mask] ** 2).sum(1))
    return {'n': n, 'sign_error_delta': difference['gripper_sign_error'],
            'translation_mse_delta': difference['translation_mse'], 'gripper_mse_delta': difference['gripper_mse'],
            'correct_to_wrong_n': int((mask & ~old_wrong & new_wrong).sum()),
            'wrong_to_correct_n': int((mask & old_wrong & ~new_wrong).sum()),
            'original_means': original, 'changed_means': changed, 'metric_deltas': difference,
            'action_response': {'mean_l2': float(norms.mean()) if n else None,
                'median_l2': float(np.median(norms)) if n else None, 'maximum_l2': float(norms.max()) if n else None,
                'signed_mean7': response[mask].mean(0).tolist() if n else None}}


def compare(expected, observed):
    """1e-12 applies only to recomputed descriptive arithmetic, never NN parity."""
    if isinstance(expected, dict):
        require(set(expected).issubset(observed), 'missing summary keys')
        return sum(compare(v, observed[k]) for k, v in expected.items())
    if isinstance(expected, list):
        require(len(expected) == len(observed), 'summary vector length differs')
        return sum(compare(a, b) for a, b in zip(expected, observed))
    if isinstance(expected, float):
        require(isinstance(observed, (int, float)) and np.isfinite(observed) and abs(expected - observed) <= 1e-12,
                'descriptive arithmetic differs')
    else:
        require(expected == observed and type(expected) is type(observed), 'summary count/null/label differs')
    return 1


def equal_demo_summary(values):
    valid = [v for v in values if v is not None]
    return {'total': len(values), 'n': len(valid), 'missing': len(values) - len(valid),
            'mean': float(np.mean(valid)) if valid else None, 'median': float(np.median(valid)) if valid else None,
            'minimum': min(valid) if valid else None, 'maximum': max(valid) if valid else None,
            'positive_count': sum(v > 0 for v in valid)}


def audit(result_path=DEFAULT_RESULT, output_path=DEFAULT_OUTPUT):
    output_path = Path(output_path)
    require(not output_path.exists(), 'audit output already exists')
    result = load(result_path)
    require(result['status'] == 'complete' and result['formal_result'] is False, 'need complete diagnostic result')
    config_path = ROOT / 'configs/libero-source-internal-gripper-state-local-control-diagnostic_v1.json'
    config = load(config_path)
    require(sha(config_path) == result['config_sha256'] == 'bf38389d06d2d43c74991365d4763cc0cbbe94ee8517ef1c9cf51ca4d4d40206', 'scientific config hash differs')
    for pin in config['inputs'].values():
        require(sha(pin['path']) == pin['sha256'], 'input provenance differs')
    binding_path = ROOT / 'configs/libero-source-internal-gripper-state-local-control-execution-binding_v1.json'
    require(sha(binding_path) == result['execution_binding_sha256'], 'execution binding hash differs')
    binding = load(binding_path)
    require(binding == result['execution_binding'], 'execution binding record differs')
    for pin in binding['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound implementation differs')
    samples = load(config['inputs']['sample_manifest']['path'])
    require(result['scope'] == config['scope'] and len(result['model_results']) == 18 and
            len(result['predictions']) == 18 and len(result['input_doses']) == 2, 'model/scope matrix differs')
    expected_counts = {'main_forward_batches': 2457, 'main_completed_batches': 2457, 'repeat_forward_batches': 54,
        'repeat_completed_batches': 54, 'main_valid_prediction_rows': 187758, 'repeat_valid_prediction_rows': 3996,
        'main_network_prediction_slots': 196020, 'repeat_network_prediction_slots': 4320, 'checkpoint_load_attempts': 18,
        'checkpoint_loads': 18, 'repeat_exact_checks': 54, 'historical_prediction_exact_rows': 62586,
        'historical_batch_digest_checks': 819, 'untreated_modified_condition_exact_rows': 119664,
        'padding_modified_condition_exact_rows': 5508, 'at_event_sham_exact_rows': 1170,
        'source_observation_rows_per_modality_attempted': 6954, 'source_observation_rows_per_modality': 6954,
        'source_action_rows_attempted': 6954, 'source_action_rows': 6954, 'source_action0_rows': 0,
        'selected_prediction_rows_in_main': 8262,
        'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0, 'new_policy_control_steps': 0,
        'new_initial_or_private_state_reads': 0}
    require(result['actual_counts'] == expected_counts, 'actual budget differs')
    checked = historical_rows = untreated_rows = padding_rows = repeat_slots = sham_rows = 0
    group_metrics, dose_summaries = {}, []
    for task in (0, 1):
        sample = next(s for s in samples['tasks'] if s['task_id'] == task)
        with np.load(resolve(config['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
            targets, originals, ids = [z[k] for k in ('targets', 'predictions', 'demonstration_indices')]
        selected = np.zeros(len(targets), bool)
        selected[[r['source_row'] for r in sample['selected']]] = True
        require(int(selected.sum()) == {0: 149, 1: 157}[task], 'selected coverage differs')
        dp = next(d for d in result['input_doses'] if d['task_id'] == task)
        require(sha(dp['path']) == dp['sha256'], 'public gripper inputs/doses hash differs')
        with np.load(dp['path'], allow_pickle=False) as z:
            doses = {k: z[k] for k in z.files}
        require(np.array_equal(doses['selected_mask'], selected), 'dose selection differs')
        original_gripper = doses['original_gripper']
        for condition in CONDITIONS:
            expected = original_gripper.copy()
            if condition != 'identity':
                anchor_key = 'event_anchor_row' if condition == CONDITIONS[1] else 'lag1_anchor_row'
                for row in sample['selected']:
                    expected[row['source_row']] = original_gripper[row[anchor_key]]
            require(np.array_equal(expected, doses['gripper_' + condition]), 'gripper replacement operator differs')
            delta = np.sqrt(((expected.astype(np.float64) - original_gripper.astype(np.float64)) ** 2).sum(1))
            require(np.array_equal(delta, doses['delta_l2_' + condition]), 'gripper dose differs')
            measured = {'n': int(selected.sum()), 'no_change_n': int((delta[selected] == 0).sum()),
                'mean_l2': float(delta[selected].mean()), 'median_l2': float(np.median(delta[selected])), 'max_l2': float(delta[selected].max())}
            checked += compare(measured, dp['selected_dose'][condition])
        dose_summaries.append({'task_id': task, 'selected_dose': dp['selected_dose']})
        for mi in range(9):
            seed, arm = SEEDS[mi // 3], ARMS[mi % 3]
            saved = next(m for m in result['model_results'] if (m['task_id'], m['training_seed'], m['arm']) == (task, seed, arm))
            pin = saved['prediction_file']
            require(sha(pin['path']) == pin['sha256'] and pin in result['predictions'], 'prediction file provenance differs')
            require(saved['checkpoint'] == next(e for e in config['endpoints'] if
                    (e['task_id'], e['training_seed'], e['arm']) == (task, seed, arm)), 'fixed endpoint changed')
            require(sha(saved['checkpoint']['path']) == saved['checkpoint']['sha256'], 'checkpoint file changed')
            require(saved['weights_unchanged'] is True and saved['gradients_none'] is True, 'weights/gradient gate differs')
            with np.load(pin['path'], allow_pickle=False) as z:
                pred, full, repeated, full_rows, repeated_rows = [z[k] for k in
                    ('predictions', 'full_predictions', 'repeat_predictions', 'full_source_rows', 'repeat_source_rows')]
                require(z['conditions'].tolist() == list(CONDITIONS), 'condition order differs')
            require(pred.dtype == full.dtype == repeated.dtype == np.float32 and
                    pred.shape == (3, len(targets), 7) and np.isfinite(full).all() and np.max(np.abs(full)) <= 1,
                    'prediction interface differs')
            require(np.array_equal(pred[0], originals[mi]), 'historical identity not bitwise exact')
            historical_rows += len(targets)
            valid = full_rows >= 0
            require(np.array_equal(full_rows[valid], np.arange(len(targets))) and
                    np.array_equal(full[:, valid], pred[:, full_rows[valid]]), 'raw valid/padding layout differs')
            require(np.array_equal(repeated_rows, full_rows[80:160]) and
                    np.array_equal(repeated, full[:, 80:160]), 'repeated batch not exact')
            repeat_slots += repeated.shape[0] * repeated.shape[1]
            at_rows = sample['at_event_sham_rows']
            for ci in (1, 2):
                require(np.array_equal(pred[ci, ~selected], pred[0, ~selected]), 'untreated output differs')
                require(np.array_equal(full[ci, ~valid], full[0, ~valid]), 'padding output differs')
                require(np.array_equal(pred[ci, at_rows], pred[0, at_rows]), 'sham output differs')
                untreated_rows += int((~selected).sum())
                padding_rows += int((~valid).sum())
                sham_rows += len(at_rows)
            for left, right in ((0, 1), (0, 2), (1, 2)):
                equal = (doses['gripper_' + CONDITIONS[left]] == doses['gripper_' + CONDITIONS[right]]).all(1)
                require(np.array_equal(pred[left, equal], pred[right, equal]), 'equal complete inputs differ')
            signs = np.sign(originals[:, :, 6])
            arm_signs = signs[mi % 3::3]
            masks = {'selected_all': selected, 'original_correct': selected & (signs[mi] < 0),
                'original_wrong_close': selected & (signs[mi] > 0), 'original_neutral': selected & (signs[mi] == 0),
                'all_three_wrong_close': selected & np.all(arm_signs > 0, axis=0),
                'same_frame_mixed_seeds': selected & np.any(arm_signs > 0, axis=0) & np.any(arm_signs < 0, axis=0),
                'all_three_correct': selected & np.all(arm_signs < 0, axis=0)}
            for lag in range(1, 6):
                mask = np.zeros(len(targets), bool)
                mask[[row['source_row'] for row in sample['selected'] if row['lag'] == lag]] = True
                masks[f'lag_{lag}'] = mask
            for ci, condition in enumerate(CONDITIONS):
                require(set(saved['conditions'][condition]) == set(masks), 'strata missing')
                for name, mask in masks.items():
                    row = saved['conditions'][condition][name]
                    metrics = scalar_summary(originals[mi], pred[ci], targets, mask)
                    checked += compare(metrics, row)
                    demos = []
                    for demo in range(30):
                        dm = scalar_summary(originals[mi], pred[ci], targets, mask & (ids == demo))
                        checked += compare(dm, row['per_demo'][str(demo)])
                        demos.append(dm)
                    for key in ('sign_error_delta', 'translation_mse_delta', 'gripper_mse_delta'):
                        checked += compare(equal_demo_summary([d[key] for d in demos]), row['equal_demo'][key])
                group_metrics[task, arm, seed, condition] = scalar_summary(originals[mi], pred[ci], targets, selected)
        record = next(a for a in result['access_records'] if a['task_id'] == task)
        require(record['demonstration_indices'] == list(range(30)) and record['demonstration_role'] == 'meta_support' and
                record['task_role'] == 'source_spi' and record['source_action0_rows_read'] == 0 and
                record['t0_excluded'] is True and record['simulator_state_access'] is False, 'source role access differs')
    recomputed_flags = []
    for task in (0, 1):
        for arm in ARMS:
            observed = next(s for s in result['paired_seed_summaries'] if (s['task_id'], s['arm']) == (task, arm))
            flags = {}
            for condition in CONDITIONS[1:]:
                metrics = [group_metrics[task, arm, seed, condition] for seed in SEEDS]
                flag = all(m['sign_error_delta'] < 0 and m['translation_mse_delta'] <= 0 for m in metrics)
                require(observed['conditions'][condition]['joint_cue'] is flag, 'joint cue differs')
                checked += compare({'mean_sign_error_delta': float(np.mean([m['sign_error_delta'] for m in metrics])),
                                    'mean_translation_mse_delta': float(np.mean([m['translation_mse_delta'] for m in metrics]))}, observed['conditions'][condition])
                flags[condition] = flag
            require(observed['primary_joint_cue'] is flags[CONDITIONS[1]] and observed['training_promotion'] is False, 'primary cue or promotion differs')
            recomputed_flags.append({'task_id': task, 'arm': arm, 'flags': flags})
    require((historical_rows, untreated_rows, padding_rows, repeat_slots, sham_rows) == (62586, 119664, 5508, 4320, 1170), 'raw parity denominators differ')
    ep = result['events']
    require(sha(ep['path']) == ep['sha256'], 'events hash differs')
    events = [json.loads(line) for line in resolve(ep['path']).read_text().splitlines()]
    for kind, expected in [('main_forward_attempt', 2457), ('main_forward_complete', 2457), ('repeat_forward_attempt', 54)]:
        entries = [e for e in events if e['event'] == kind]
        require(len(entries) == expected and len({(e['task'], e['seed'], e['arm'], e['condition'], e['batch']) for e in entries}) == expected, 'forward attempt/completion events differ')
        if kind == 'repeat_forward_attempt':
            require(all(e['batch'] == 1 for e in entries), 'repeat selection differs')
    source_events = [e for e in events if e['event'] == 'source_window_read_attempt']
    require(len(source_events) == 726 and all(e['action_slice'][0] >= 1 and 0 <= e['demo'] < 30 and e['task'] in (0, 1) for e in source_events), 'source read events differ')
    require(sum(e['action_slice'][1] - e['action_slice'][0] for e in source_events) == 6954, 'source action read budget differs')
    require(events[-1]['event'] == 'complete', 'run completion missing')
    report = {'status': 'complete_independent_saved_array_audit', 'formal_result': False,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'result': {'path': str(result_path), 'sha256': sha(result_path)},
        'auditor_sha256': sha(__file__), 'scalar_checks': checked, 'arithmetic_absolute_tolerance': 1e-12,
        'neural_parity_tolerance': 0, 'historical_exact_rows': historical_rows,
        'untreated_exact_rows': untreated_rows, 'padding_exact_rows': padding_rows, 'sham_exact_rows': sham_rows,
        'repeat_full_network_slots': repeat_slots, 'actual_counts_verified': expected_counts,
        'recomputed_joint_cues': recomputed_flags, 'input_doses_verified': dose_summaries,
        'new_policy_forward_batches': 0, 'new_training_updates': 0, 'new_environment_calls': 0,
        'new_HDF5_RGB_initial_private_payload_reads': 0}
    with output_path.open('x') as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, default=DEFAULT_RESULT)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = audit(args.result, args.output)
    print(json.dumps({'status': report['status'], 'scalar_checks': report['scalar_checks'],
                      'recomputed_joint_cues': report['recomputed_joint_cues']}))
