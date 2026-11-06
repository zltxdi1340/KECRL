"""Frozen full-Policy gripper-state readout:2457 main+54 repeat batches.

Default mode validates bound JSON/code and cached labels only. --execute reads
source-support causal observations on idle physicalGPU1. No training/rollout.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import validate_libero_source_internal_gripper_state_control_protocol as p

f = p.phase.parent
DEFAULT_DESIGN_BINDING = ROOT / 'configs/libero-source-internal-gripper-state-local-control-design-binding_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-gripper-state-local-control-execution-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-gripper-state-local-control-20261104-v1.json')
DEFAULT_ARTIFACTS = Path('/tmp/libero-source-internal-gripper-state-local-control-artifacts-20261104-v1')
DESIGN_BINDING_SHA = 'e81809ebff82d4adebf39aafb2b13cbd963db9d69a4c30aceed155ddca8257d8'
LAUNCH = {'CUDA_VISIBLE_DEVICES': '1', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
          'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
BATCH_KEYS = (*f.OBS_SHAPES, 'actions', 'valid_mask', 'demonstration_index', 'start_index')


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def preflight(config_path, design_binding_path, binding_path):
    c, sample, review, design_binding = p.validate(config_path, design_binding_path)
    p.require(p.sha(design_binding_path) == DESIGN_BINDING_SHA, 'original design/operator binding differs')
    binding = p.load(binding_path)
    p.require(binding['formal_result'] is False and binding['config_sha256'] == p.CONFIG_SHA and
              binding['design_binding_sha256'] == DESIGN_BINDING_SHA and
              binding['launch_environment'] == LAUNCH and binding['runtime'] == c['execution']['runtime'],
              'execution binding specification differs')
    p.require(set(binding['files']) == {'runner', 'tests', 'auditor', 'resource_gate', 'execution_protocol'},
              'execution file set differs')
    for name, pin in binding['files'].items():
        p.require(p.sha(p.resolve(pin['path'])) == pin['sha256'], f'execution code differs: {name}')
    p.require(p.resolve(binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner path differs')
    return c, sample, review, binding


def launch_gate():
    for name, value in LAUNCH.items():
        p.require(os.environ.get(name) == value, f'frozen launch environment differs: {name}')


def batch_frame_rows(sample):
    """Original Bx10 layout; -1 denotes padding, never a source array index."""
    lookup = {tuple(v): i for i, v in enumerate(sample['source_frame_ids'])}
    windows = sample['canonical_windows']
    result = []
    for start in range(0, len(windows), 8):
        part = windows[start:start + 8]
        rows = np.full((len(part), 10), -1, np.int32)
        for j, w in enumerate(part):
            for lag in range(w['valid_length']):
                rows[j, lag] = lookup[w['demonstration_index'], w['start'] + lag]
        result.append(rows)
    scored = np.concatenate([r[r >= 0] for r in result])
    p.require(np.array_equal(scored, np.arange(sample['source_frames'])), 'batch layout changed source ordering')
    return result


def condition_batch_gripper(original, row_ids, changed):
    """Replace valid rows only, retaining original padded states."""
    p.require(original.shape == (*row_ids.shape, 2) and original.dtype == np.float32 and
              changed.ndim == 2 and changed.shape[1] == 2 and changed.dtype == np.float32,
              'batch gripper interface differs')
    valid = row_ids >= 0
    p.require(np.isfinite(original).all() and np.isfinite(changed).all() and
              np.all(row_ids >= -1) and np.all(row_ids[valid] < len(changed)), 'invalid batch source row')
    result = original.copy()
    result[valid] = changed[row_ids[valid]]
    return result


def fixed_masks(original_predictions, model_index, target, sample):
    selected = np.zeros(len(target), bool)
    selected[[r['source_row'] for r in sample['selected']]] = True
    signs = np.sign(original_predictions[:, :, 6])
    model_sign = signs[model_index]
    arm_signs = signs[model_index % 3::3]
    result = {'selected_all': selected, 'original_correct': selected & (model_sign < 0),
              'original_wrong_close': selected & (model_sign > 0), 'original_neutral': selected & (model_sign == 0),
              'all_three_wrong_close': selected & (arm_signs > 0).all(0),
              'same_frame_mixed_seeds': selected & (arm_signs > 0).any(0) & (arm_signs < 0).any(0),
              'all_three_correct': selected & (arm_signs < 0).all(0)}
    for lag in range(1, 6):
        m = np.zeros(len(target), bool)
        m[[r['source_row'] for r in sample['selected'] if r['lag'] == lag]] = True
        result[f'lag_{lag}'] = m
    p.require(np.array_equal(sum(result[k].astype(int) for k in
              ('original_correct', 'original_wrong_close', 'original_neutral')), selected.astype(int)), 'original labels do not partition selection')
    return result


def metric_summary(original, changed, target, mask):
    result = p.condition_metrics(original, changed, target, mask)
    before, _ = f.row_errors(original, target)
    after, _ = f.row_errors(changed, target)
    n = result['n']
    result['original_means'] = {k: float(v[mask].mean()) if n else None for k, v in before.items()}
    result['changed_means'] = {k: float(v[mask].mean()) if n else None for k, v in after.items()}
    result['metric_deltas'] = {k: float(after[k][mask].mean() - v[mask].mean()) if n else None for k, v in before.items()}
    response = changed.astype(np.float64) - original.astype(np.float64)
    norms = np.linalg.norm(response[mask], axis=1)
    result['action_response'] = {'mean_l2': float(norms.mean()) if n else None,
                                 'median_l2': float(np.median(norms)) if n else None,
                                 'maximum_l2': float(norms.max()) if n else None,
                                 'signed_mean7': response[mask].mean(0).tolist() if n else None}
    return result


def analyze_model(original_predictions, model_index, predictions, target, ids, sample):
    masks = fixed_masks(original_predictions, model_index, target, sample)
    original = original_predictions[model_index]
    result = {}
    for ci, condition in enumerate(p.CONDITIONS):
        rows = {}
        for name, mask in masks.items():
            summary = metric_summary(original, predictions[ci], target, mask)
            per_demo = {str(d): metric_summary(original, predictions[ci], target, mask & (ids == d)) for d in range(30)}
            summary['per_demo'] = per_demo
            summary['equal_demo'] = {k: p.phase.stats([r[k] for r in per_demo.values()])
                                     for k in ('sign_error_delta', 'translation_mse_delta', 'gripper_mse_delta')}
            rows[name] = summary
        result[condition] = rows
    return result


def summarize_models(rows):
    output = []
    for task in p.TASKS:
        for arm in p.ARMS:
            group = sorted((r for r in rows if r['task_id'] == task and r['arm'] == arm), key=lambda r: r['training_seed'])
            p.require([r['training_seed'] for r in group] == list(p.SEEDS), 'incomplete paired seeds')
            conditions = {}
            for condition in p.CONDITIONS[1:]:
                metrics = [r['conditions'][condition]['selected_all'] for r in group]
                conditions[condition] = {'per_seed': [{'training_seed': r['training_seed'],
                                        **{k: m[k] for k in ('n', 'sign_error_delta', 'translation_mse_delta',
                                                             'gripper_mse_delta', 'correct_to_wrong_n', 'wrong_to_correct_n')}}
                                        for r, m in zip(group, metrics)],
                                        'joint_cue': p.descriptive_joint_flag(metrics),
                                        'mean_sign_error_delta': float(np.mean([m['sign_error_delta'] for m in metrics])),
                                        'mean_translation_mse_delta': float(np.mean([m['translation_mse_delta'] for m in metrics]))}
            output.append({'task_id': task, 'arm': arm, 'primary_joint_cue': conditions[p.CONDITIONS[1]]['joint_cue'],
                           'conditions': conditions, 'training_promotion': False})
    return output


def new_counts():
    return {k: 0 for k in ('main_forward_batches', 'main_completed_batches', 'repeat_forward_batches',
        'repeat_completed_batches', 'main_valid_prediction_rows', 'repeat_valid_prediction_rows',
        'main_network_prediction_slots', 'repeat_network_prediction_slots', 'checkpoint_load_attempts', 'checkpoint_loads',
        'repeat_exact_checks', 'historical_prediction_exact_rows', 'historical_batch_digest_checks',
        'untreated_modified_condition_exact_rows', 'padding_modified_condition_exact_rows', 'at_event_sham_exact_rows',
        'source_observation_rows_per_modality_attempted', 'source_observation_rows_per_modality',
        'source_action_rows_attempted', 'source_action_rows', 'source_action0_rows', 'selected_prediction_rows_in_main', 'new_optimizer_updates',
        'new_backward_calls', 'new_environment_calls', 'new_policy_control_steps', 'new_initial_or_private_state_reads')}


def execute_forward(policy, observations, counts, repeat, event, context):
    """All actual forward calls pass here, counted before invocation."""
    key = 'repeat' if repeat else 'main'
    maximum = 54 if repeat else 2457
    p.require(counts[key + '_forward_batches'] < maximum, f'{key} forward budget exhausted')
    counts[key + '_forward_batches'] += 1
    event(key + '_forward_attempt', **context, attempted=counts[key + '_forward_batches'])
    prediction, _ = policy.forward_sequence(*observations)
    counts[key + '_completed_batches'] += 1
    return prediction


def run(config_path, design_binding_path, binding_path, output, artifacts):
    output, artifacts = Path(output), Path(artifacts)
    failed = output.with_name(output.stem + '-failed-partial.json')
    p.require(not output.exists() and not failed.exists() and not artifacts.exists(), 'output/artifacts already exist; preserve old runs')
    p.require(output.resolve().is_relative_to(Path('/tmp')) and artifacts.resolve().is_relative_to(Path('/tmp')),
              'experiment artifacts must stay under /tmp')
    entry_started = time.monotonic()
    requested_utc = datetime.now(timezone.utc).isoformat()
    artifacts.mkdir(parents=True, exist_ok=False)
    events_path = artifacts / 'events.jsonl'
    events = events_path.open('x')
    counts = new_counts()
    completed, dose_files, access_records, model_results, task_parity = [], [], [], [], []
    active, last_prediction = None, None
    config = binding = None
    runtime_metadata = {}
    stage = 'binding_preflight'
    def event(kind, **fields):
        events.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}, allow_nan=False) + '\n')
        events.flush()
    def digest(tensors):
        h = hashlib.sha256()
        for name, tensor in tensors.items():
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    timings = {'cache_seconds': 0., 'model_readout_and_checks_seconds': 0., 'numerical_analysis_seconds': 0.}
    peak_cache_bytes = 0
    try:
        event('requested', research_calls=0)
        config, samples, review, binding = preflight(config_path, design_binding_path, binding_path)
        launch_gate()
        from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
        stage = 'resource_gate'
        gpu = idle_gpu(1)
        runtime_metadata['gpu'] = gpu
        event('resource_gate_passed', gpu=gpu, research_calls=0)
        import torch
        import h5py
        from torch.utils.data import DataLoader
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
        from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
        from src.environments.libero_dataset import resolve_demonstration_access
        runtime = {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'numpy': np.__version__, 'h5py': h5py.__version__}
        runtime_metadata.update(runtime)
        p.require(runtime == config['execution']['runtime'], 'frozen runtime differs')
        p.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one selected CUDA device required')
        torch.set_num_threads(1)
        torch.manual_seed(20261022)
        torch.cuda.manual_seed_all(20261022)
        np.random.seed(20261022)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device('cuda:0')
        readout_started_utc = datetime.now(timezone.utc).isoformat()
        source_root = Path('/home/zl202621/third_party/LIBERO/datasets').resolve()
        transfer, manifest, learning = [p.load(p.resolve(config['inputs'][k]['path'])) for k in ('transfer', 'dataset_manifest', 'learning')]
        original_result = p.load(p.resolve(config['inputs']['original_frame_result']['path']))
        manifest_rows = {r['task_id']: r for r in manifest['files']}
        for task in p.TASKS:
            stage = f'task{task}_source_cache'
            started = time.monotonic()
            sample = next(r for r in samples['tasks'] if r['task_id'] == task)
            authorized = resolve_demonstration_access(transfer, manifest, task_id=task, demonstration_role='meta_support')
            p.require(authorized.task_role == 'source_spi' and list(authorized.demonstration_indices) == list(range(30)), 'source role authorization differs')
            m = manifest_rows[task]
            source_path = (source_root / m['path']).resolve()
            p.require(source_path.is_relative_to(source_root) and source_path.stat().st_size == m['size_bytes'] and
                      p.sha(source_path) == m['sha256'], 'source file provenance differs')
            ds = CausalT1LiberoSequenceDataset(transfer, manifest, source_root, task_id=task,
                 demonstration_role='meta_support', sequence_length=10, stride=10, verify_sha256=False)
            try:
                p.require([asdict(w) for w in ds.windows] == sample['canonical_windows'], 'source window plan differs')
                items = []
                for i, w in enumerate(ds.windows):
                    counts['source_observation_rows_per_modality_attempted'] += w.valid_length
                    counts['source_action_rows_attempted'] += w.valid_length
                    event('source_window_read_attempt', task=task, demo=w.demonstration_index,
                          observation_slice=[w.start - 1, w.start + w.valid_length - 1], action_slice=[w.start, w.start + w.valid_length])
                    items.append(ds[i])
                    counts['source_observation_rows_per_modality'] += w.valid_length
                    counts['source_action_rows'] += w.valid_length
                record = {**ds.access_record, 'file_sha256_verified_before_payload': True, 'source_action0_rows_read': 0}
            finally:
                ds.close()
            cache_bytes = sum(v.numel() * v.element_size() for item in items for v in item.values() if torch.is_tensor(v))
            peak_cache_bytes = max(peak_cache_bytes, cache_bytes)
            p.require(cache_bytes <= config['scope']['maximum_host_cache_bytes'], 'host cache budget exceeded')
            record['cache_bytes'] = cache_bytes
            access_records.append(record)
            target = np.concatenate([item['actions'][:w['valid_length']].numpy() for item, w in zip(items, sample['canonical_windows'])])
            original_gripper = np.concatenate([item['gripper_states'][:w['valid_length']].numpy() for item, w in zip(items, sample['canonical_windows'])])
            with np.load(p.resolve(config['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
                old = {k: z[k] for k in ('targets', 'predictions', 'model_ids', 'demonstration_indices', 'action_steps')}
            p.require(np.array_equal(target, old['targets']) and
                      np.array_equal(np.column_stack((old['demonstration_indices'], old['action_steps'])), sample['source_frame_ids']),
                      'original target/frame parity differs')
            ids = old['demonstration_indices']
            grip = {condition: p.replace_gripper_states(original_gripper, sample, condition) for condition in p.CONDITIONS}
            selected = np.zeros(len(target), bool)
            selected[[r['source_row'] for r in sample['selected']]] = True
            sham = np.zeros(len(target), bool)
            sham[sample['at_event_sham_rows']] = True
            doses = {condition: np.linalg.norm(grip[condition].astype(np.float64) - original_gripper.astype(np.float64), axis=1) for condition in p.CONDITIONS}
            dose_path = artifacts / f'task{task}-gripper-inputs-and-doses.npz'
            with dose_path.open('xb') as handle:
                np.savez_compressed(handle, original_gripper=original_gripper, selected_mask=selected,
                    **{'gripper_' + k: v for k, v in grip.items()}, **{'delta_l2_' + k: v for k, v in doses.items()})
            dose_files.append({'task_id': task, 'path': str(dose_path), 'sha256': p.sha(dose_path),
                'selected_dose': {k: {'n': int(selected.sum()), 'no_change_n': int((v[selected] == 0).sum()),
                    'mean_l2': float(v[selected].mean()), 'median_l2': float(np.median(v[selected])), 'max_l2': float(v[selected].max())} for k, v in doses.items()}})
            frame_batches = batch_frame_rows(sample)
            old_batches = next(a['shared_batch_digests'] for a in original_result['archives'] if a['task_id'] == task)
            task_parity.append({'task_id': task, 'shared_batch_digests': old_batches, 'targets_exact': True,
                                'selected_frames': int(selected.sum()), 'at_event_sham_frames': int(sham.sum())})
            timings['cache_seconds'] += time.monotonic() - started
            event('task_cached', task=task, valid_rows=len(target), cache_bytes=cache_bytes, action0_rows_read=0)
            endpoints = [e for e in config['endpoints'] if e['task_id'] == task]
            for model_index, endpoint in enumerate(endpoints):
                seed, arm = endpoint['training_seed'], endpoint['arm']
                stage = f'task{task}_seed{seed}_{arm}_readout'
                started = time.monotonic()
                p.require(f'{seed}:{arm}' == str(old['model_ids'][model_index]), 'model ordering differs')
                p.require(p.sha(endpoint['path']) == endpoint['sha256'], 'checkpoint file SHA differs')
                spec = {**learning['policy'], 'memory_mode': 'cnn_only', 'policy_family': 'dual_view_shared_cnn_only_uniform_bc_diagnostic_v1'}
                policy = make_deterministic_libero_policy(LiberoBCPolicyConfig.from_mapping(spec)).to(device)
                counts['checkpoint_load_attempts'] += 1
                state = torch.load(endpoint['path'], map_location=device, weights_only=True)
                counts['checkpoint_loads'] += 1
                p.require(digest(state) == endpoint['weight_hash'], 'checkpoint tensor SHA differs')
                policy.load_state_dict(state, strict=True)
                policy.requires_grad_(False)
                policy.eval()
                p.require(digest(policy.state_dict()) == endpoint['weight_hash'], 'loaded Policy hash differs')
                network_slots = sum(r.size for r in frame_batches)
                active = {'predictions': np.full((3, len(target), 7), np.nan, np.float32),
                          'full_predictions': np.full((3, network_slots, 7), np.nan, np.float32),
                          'completed_rows': np.zeros((3, len(target)), bool)}
                repeats = {}
                slot_start = 0
                with torch.inference_mode():
                    for bi, batch in enumerate(DataLoader(items, batch_size=8, shuffle=False, num_workers=0)):
                        row_ids = frame_batches[bi]
                        valid = row_ids >= 0
                        p.require(np.array_equal(batch['valid_mask'].numpy(), valid), 'valid mask differs from frozen rows')
                        base_hash = digest({k: batch[k] for k in BATCH_KEYS})
                        p.require(base_hash == old_batches[bi], 'historical original batch digest differs')
                        counts['historical_batch_digest_checks'] += 1
                        b = {k: v.to(device) for k, v in batch.items()}
                        cpu_grippers = {k: condition_batch_gripper(batch['gripper_states'].numpy(), row_ids, v) for k, v in grip.items()}
                        outputs = {}
                        untreated = np.ones(row_ids.shape, bool)
                        untreated[valid] = ~selected[row_ids[valid]]
                        sham_rows = np.zeros(row_ids.shape, bool)
                        sham_rows[valid] = sham[row_ids[valid]]
                        for ci, condition in enumerate(p.CONDITIONS):
                            ctx = {'task': task, 'seed': seed, 'arm': arm, 'condition': condition, 'batch': bi}
                            changed_gripper = torch.from_numpy(cpu_grippers[condition]).to(device)
                            observations = [b[k] if k != 'gripper_states' else changed_gripper for k in f.OBS_SHAPES]
                            prediction = execute_forward(policy, observations, counts, False, event, ctx)
                            last_prediction = prediction.detach().cpu().numpy().copy()
                            p.require(last_prediction.shape == (*row_ids.shape, 7) and np.isfinite(last_prediction).all() and
                                      np.max(np.abs(last_prediction)) <= 1, 'invalid Policy output')
                            counts['main_valid_prediction_rows'] += int(valid.sum())
                            counts['main_network_prediction_slots'] += row_ids.size
                            counts['selected_prediction_rows_in_main'] += int(selected[row_ids[valid]].sum())
                            active['predictions'][ci, row_ids[valid]] = last_prediction[valid]
                            active['full_predictions'][ci, slot_start:slot_start + row_ids.size] = last_prediction.reshape(-1, 7)
                            active['completed_rows'][ci, row_ids[valid]] = True
                            outputs[condition] = last_prediction
                            if condition == 'identity':
                                p.require(np.array_equal(last_prediction[valid], old['predictions'][model_index, row_ids[valid]]),
                                          'historical raw predictions not exact')
                                counts['historical_prediction_exact_rows'] += int(valid.sum())
                            else:
                                p.require(np.array_equal(last_prediction[untreated], outputs['identity'][untreated]), 'untreated or padding output differs')
                                counts['untreated_modified_condition_exact_rows'] += int((untreated & valid).sum())
                                counts['padding_modified_condition_exact_rows'] += int((~valid).sum())
                                p.require(np.array_equal(last_prediction[sham_rows], outputs['identity'][sham_rows]), 'at-event sham output differs')
                                counts['at_event_sham_exact_rows'] += int(sham_rows.sum())
                            if bi == sample['repeat_batch']:
                                repeated = execute_forward(policy, observations, counts, True, event, ctx)
                                repeated_np = repeated.detach().cpu().numpy().copy()
                                repeats[condition] = repeated_np.reshape(-1, 7)
                                counts['repeat_valid_prediction_rows'] += int(valid.sum())
                                counts['repeat_network_prediction_slots'] += row_ids.size
                                p.require(torch.equal(prediction, repeated), 'repeat output not exact')
                                counts['repeat_exact_checks'] += 1
                            p.require(np.array_equal(changed_gripper.cpu().numpy(), cpu_grippers[condition]) and
                                      digest({k: b[k] for k in BATCH_KEYS}) == base_hash, 'Policy mutated inputs/targets')
                            event('main_forward_complete', **ctx, valid_rows=int(valid.sum()), network_slots=row_ids.size,
                                  output_sha256=hashlib.sha256(last_prediction.tobytes()).hexdigest(), source_input_sha256=base_hash)
                        for left, right in ((p.CONDITIONS[0], p.CONDITIONS[1]), (p.CONDITIONS[0], p.CONDITIONS[2]), (p.CONDITIONS[1], p.CONDITIONS[2])):
                            equal_input = np.all(cpu_grippers[left] == cpu_grippers[right], axis=-1)
                            p.require(np.array_equal(outputs[left][equal_input], outputs[right][equal_input]), 'same complete input output differs')
                        slot_start += row_ids.size
                p.require(active['completed_rows'].all() and digest(policy.state_dict()) == endpoint['weight_hash'] and
                          all(param.grad is None for param in policy.parameters()), 'coverage/weights/gradients changed')
                pred_path = artifacts / f'task{task}-seed{seed}-{arm}-readouts.npz'
                full_rows = np.concatenate([r.reshape(-1) for r in frame_batches])
                repeat_rows = frame_batches[sample['repeat_batch']].reshape(-1)
                with pred_path.open('xb') as handle:
                    np.savez_compressed(handle, predictions=active['predictions'], full_predictions=active['full_predictions'],
                        repeat_predictions=np.stack([repeats[k] for k in p.CONDITIONS]), conditions=np.array(p.CONDITIONS),
                        full_source_rows=full_rows, repeat_source_rows=repeat_rows)
                completed.append({'task_id': task, 'training_seed': seed, 'arm': arm, 'path': str(pred_path), 'sha256': p.sha(pred_path)})
                timings['model_readout_and_checks_seconds'] += time.monotonic() - started
                started = time.monotonic()
                conditions = analyze_model(old['predictions'], model_index, active['predictions'], target, ids, sample)
                model_results.append({'task_id': task, 'training_seed': seed, 'arm': arm, 'checkpoint': endpoint,
                                      'prediction_file': completed[-1], 'conditions': conditions,
                                      'weights_unchanged': True, 'gradients_none': True})
                timings['numerical_analysis_seconds'] += time.monotonic() - started
                active = last_prediction = None
                event('model_complete', task=task, seed=seed, arm=arm, actual_counts=counts)
                print(json.dumps({'progress': 'model_complete', 'task': task, 'seed': seed, 'arm': arm,
                                  'main_batches': counts['main_forward_batches'], 'repeat_batches': counts['repeat_forward_batches']}), flush=True)
                del policy, state
            del items, grip, old
        expected = {'main_forward_batches': 2457, 'main_completed_batches': 2457, 'repeat_forward_batches': 54,
                    'repeat_completed_batches': 54, 'main_valid_prediction_rows': 187758, 'repeat_valid_prediction_rows': 3996,
                    'main_network_prediction_slots': 196020, 'repeat_network_prediction_slots': 4320,
                    'checkpoint_load_attempts': 18, 'checkpoint_loads': 18, 'repeat_exact_checks': 54,
                    'historical_prediction_exact_rows': 62586, 'historical_batch_digest_checks': 819,
                    'untreated_modified_condition_exact_rows': 119664, 'padding_modified_condition_exact_rows': 5508,
                    'at_event_sham_exact_rows': 1170, 'source_observation_rows_per_modality_attempted': 6954,
                    'source_observation_rows_per_modality': 6954, 'source_action_rows_attempted': 6954, 'source_action_rows': 6954,
                    'source_action0_rows': 0, 'selected_prediction_rows_in_main': 8262,
                    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
                    'new_policy_control_steps': 0, 'new_initial_or_private_state_reads': 0}
        p.require(counts == expected, 'actual forward/access budget differs')
        result = {'status': 'complete', 'formal_result': False, 'config_sha256': p.sha(config_path),
                  'design_binding_sha256': p.sha(design_binding_path), 'execution_binding_sha256': p.sha(binding_path),
                  'scope': config['scope'], 'analysis_definition': config['analysis'], 'input_provenance': config['inputs'],
                  'execution_binding': binding, 'actual_counts': counts, 'model_results': model_results,
                  'paired_seed_summaries': summarize_models(model_results), 'predictions': completed, 'input_doses': dose_files,
                  'access_records': access_records, 'historical_task_parity': task_parity,
                  'runtime': {**runtime, 'requested_at_utc': requested_utc, 'started_at_utc': readout_started_utc,
                      'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                      'entry_elapsed_seconds': time.monotonic() - entry_started, 'stage_seconds': timings, 'gpu': gpu,
                      'device': 'cuda:0', 'torch_threads': 1, 'deterministic_algorithms': True, 'tf32': False,
                      'peak_host_cache_bytes': peak_cache_bytes, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'launch_argv': sys.argv, 'launch_environment': {k: os.environ.get(k) for k in LAUNCH}},
                  'interpretation': 'Fixed Policy response to local hybrid observations; reference error uses original expert action, no physical/training/SPT/qualification result.'}
        event('complete', actual_counts=counts)
        events.close()
        result['events'] = {'path': str(events_path), 'sha256': p.sha(events_path)}
        write_json(output, result)
        return result
    except Exception as error:
        partial = None
        if active is not None or last_prediction is not None:
            partial = artifacts / 'failed-active-readouts.npz'
            with partial.open('xb') as handle:
                np.savez_compressed(handle, **(active or {}), **({'last_prediction': last_prediction} if last_prediction is not None else {}))
        event('failed_partial', stage=stage, error_type=type(error).__name__, error=str(error), actual_counts=counts)
        events.close()
        write_json(failed, {'status': 'failed_partial', 'formal_result': False, 'stage': stage,
            'error_type': type(error).__name__, 'error': str(error), 'actual_counts': counts,
            'config_sha256': p.sha(config_path), 'execution_binding_sha256': p.sha(binding_path) if Path(binding_path).exists() else None,
            'scope': config['scope'] if config else None,
            'completed_predictions': completed, 'input_doses': dose_files, 'access_records': access_records,
            'partial_predictions': {'path': str(partial), 'sha256': p.sha(partial)} if partial is not None else None,
            'events': {'path': str(events_path), 'sha256': p.sha(events_path)},
            'runtime': {**runtime_metadata, 'requested_at_utc': requested_utc, 'failed_at_utc': datetime.now(timezone.utc).isoformat(),
                        'entry_elapsed_seconds': time.monotonic() - entry_started, 'launch_environment': {k: os.environ.get(k) for k in LAUNCH}}})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=p.DEFAULT_CONFIG)
    parser.add_argument('--design-binding', type=Path, default=DEFAULT_DESIGN_BINDING)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = run(args.config, args.design_binding, args.binding, args.output, args.artifacts)
        print(json.dumps({'status': result['status'], 'actual_counts': result['actual_counts']}), flush=True)
    else:
        c, samples, _, _ = preflight(args.config, args.design_binding, args.binding)
        print(json.dumps({'status': 'execution_binding_validated_not_executed', 'planned_main_batches': 2457,
                          'planned_repeat_batches': 54, 'actual_new_policy_calls': 0, 'actual_new_observation_payload_reads': 0}))


if __name__ == '__main__':
    main()
