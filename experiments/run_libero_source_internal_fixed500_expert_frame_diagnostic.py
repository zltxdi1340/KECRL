"""Bounded expert-input readout at fixed500 endpoints; no training or rollout.

Default mode validates JSON/code only. --execute performs the frozen819+18
forward batches on idle physicalGPU1 and preserves all predictions and failures.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import validate_libero_source_internal_fixed500_expert_frame_protocol as p

DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-fixed500-expert-frame-20261103-v1.json')
DEFAULT_ARTIFACTS = Path('/tmp/libero-source-internal-fixed500-expert-frame-artifacts-20261103-v1')
BATCH_KEYS = (*p.OBS_SHAPES, 'actions', 'valid_mask', 'demonstration_index', 'start_index')


def historical_parity(summary, old):
    p.require(summary['n'] == old['valid_steps'], 'historical valid-step parity failed')
    for metric in ('mse', 'translation_mse', 'rotation_mse', 'gripper_mse'):
        p.require(abs(summary['metric_means'][metric] - old[metric]) <= 1e-10, f'historical MSE parity failed: {metric}')
    correct = sum(summary['gripper_sign_confusion'][i][i] for i in range(3))
    p.require(correct / summary['n'] == old['gripper_sign_accuracy'],
              'historical gripper accuracy parity failed')


def run(config_path, binding_path, output, artifacts):
    output, artifacts = Path(output), Path(artifacts)
    failed_path = output.with_name(output.stem + '-failed-partial.json')
    p.require(not output.exists() and not artifacts.exists() and not failed_path.exists(),
              'output/artifacts already exist; preserve old runs')
    entry_started = time.monotonic()
    requested_utc = datetime.now(timezone.utc).isoformat()
    config, inputs, budget, binding = p.validate(config_path, binding_path)
    p.require(binding is not None and p.resolve(binding['files']['runner']['path']).resolve() == Path(__file__).resolve(),
              'execution runner binding differs')
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu = idle_gpu(1)  # Original, pinned resource gate; never calls its training entry.
    p.require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8', 'deterministic cuBLAS launch setting differs')
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
    from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
    expected_runtime = config['execution']['runtime']
    p.require({'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
               'numpy': np.__version__} == expected_runtime, 'frozen runtime differs')
    p.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one selected CUDA device required')
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    np.random.seed(20261022)
    torch.cuda.manual_seed_all(20261022)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    artifacts.mkdir(parents=True, exist_ok=False)
    event_path = artifacts / 'events.jsonl'
    events = event_path.open('x')
    counts = {'main_forward_batches': 0, 'main_completed_batches': 0, 'repeat_forward_batches': 0,
              'repeat_completed_batches': 0, 'main_valid_prediction_rows': 0, 'repeat_valid_prediction_rows': 0,
              'main_network_prediction_slots': 0, 'repeat_network_prediction_slots': 0,
              'checkpoint_loads': 0, 'repeat_exact_checks': 0, 'historical_model_parity_checks': 0,
              'source_observation_rows_per_modality': 0, 'source_action_rows_including_label_only_t0': 0,
              'source_observation_rows_per_modality_attempted': 0, 'source_action_rows_including_label_only_t0_attempted': 0,
              'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
              'new_policy_control_steps': 0, 'new_initial_or_private_state_reads': 0}
    rows, archives, access_records = [], [], []
    completed_predictions = []
    source = inputs['source_result']
    old = {(f['probe'], s['seed'], a): entry['curve']['500']['meta_support']
           for f in source['fold_results'] for s in f['seed_results'] for a, entry in s['arms'].items()}
    stage_seconds = {'cache': 0., 'inference_and_batch_checks': 0., 'numerical_analysis': 0.}
    peak_cache_bytes = 0
    def event(kind, **fields):
        events.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}, allow_nan=False) + '\n')
        events.flush()
    def digest(tensors):
        import hashlib
        h = hashlib.sha256()
        for name, tensor in tensors.items():
            h.update(name.encode())
            h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def write_result(path, value):
        with path.open('x') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
    started_utc = datetime.now(timezone.utc).isoformat()
    try:
        root = Path('/home/zl202621/third_party/LIBERO/datasets').resolve()
        manifest_rows = {r['task_id']: r for r in inputs['dataset_manifest']['files']}
        for task in p.TASKS:
            cache_started = time.monotonic()
            m = manifest_rows[task]
            dataset_path = (root / m['path']).resolve()
            p.require(dataset_path.is_relative_to(root) and dataset_path.stat().st_size == m['size_bytes'] and
                      p.sha(dataset_path) == m['sha256'], 'source dataset file provenance differs')
            ds = CausalT1LiberoSequenceDataset(inputs['transfer'], inputs['dataset_manifest'], root, task_id=task,
                                              demonstration_role='meta_support', sequence_length=10, stride=10,
                                              verify_sha256=False)
            meta = next(r for r in inputs['source_metadata']['tasks'] if r['task_id'] == task)
            plan = p.window_plan([d['actions_shape'][0] for d in meta['demonstrations']])
            p.require([asdict(w) for w in ds.windows] == plan, 'actual source window plan differs')
            items = []
            for i, w in enumerate(ds.windows):
                counts['source_observation_rows_per_modality_attempted'] += w.valid_length
                counts['source_action_rows_including_label_only_t0_attempted'] += w.valid_length
                items.append(ds[i])
                counts['source_observation_rows_per_modality'] += w.valid_length
                counts['source_action_rows_including_label_only_t0'] += w.valid_length
            cache_bytes = sum(t.numel() * t.element_size() for item in items for t in item.values() if torch.is_tensor(t))
            peak_cache_bytes = max(peak_cache_bytes, cache_bytes)
            p.require(cache_bytes <= config['scope']['maximum_host_cache_bytes'], 'host cache budget exceeded')
            targets = np.concatenate([item['actions'][:w.valid_length].numpy() for item, w in zip(items, ds.windows)])
            ids = np.concatenate([np.full(w.valid_length, w.demonstration_index, np.int16) for w in ds.windows])
            times = np.concatenate([np.arange(w.start, w.start + w.valid_length, dtype=np.int32) for w in ds.windows])
            mask_parts = {name: [] for name in p.STRATA + p.ATOMS}
            atom_parts = []
            for demo in range(30):
                counts['source_action_rows_including_label_only_t0_attempted'] += 1
                a0 = np.asarray(ds._open()[f'data/demo_{demo}/actions'][0], dtype=np.float32)
                counts['source_action_rows_including_label_only_t0'] += 1
                commands = np.concatenate([a0[None], targets[ids == demo]])
                masks, atoms = p.expert_masks(commands)
                for name in mask_parts:
                    mask_parts[name].append(masks[name][1:])
                atom_parts.append(atoms[1:])
            masks = {name: np.concatenate(parts) for name, parts in mask_parts.items()}
            atomic_ids = np.concatenate(atom_parts)
            access_records.append({**ds.access_record, 'file_sha256_verified_before_payload': True,
                                   'source_action0_rows_for_labels_only': 30, 'cache_bytes': cache_bytes})
            ds.close()
            stage_seconds['cache'] += time.monotonic() - cache_started
            event('task_source_cached', task=task, valid_rows=len(targets), windows=len(items), cache_bytes=cache_bytes)
            labels_path = artifacts / f'task{task}-expert-labels.npz'
            with labels_path.open('xb') as handle:
                np.savez_compressed(handle, targets=targets, demonstration_indices=ids, action_steps=times,
                                    atomic_ids=atomic_ids, **{'mask_' + name: value for name, value in masks.items()})
            shared_batch_digests = []
            task_predictions, task_model_ids = [], []
            for endpoint in [e for e in config['endpoints'] if e['task_id'] == task]:
                seed, arm = endpoint['training_seed'], endpoint['arm']
                model_started = time.monotonic()
                p.require(p.sha(endpoint['path']) == endpoint['sha256'], 'checkpoint file hash differs')
                policy_spec = {**inputs['learning']['policy'], 'memory_mode': 'cnn_only',
                               'policy_family': 'dual_view_shared_cnn_only_uniform_bc_diagnostic_v1'}
                policy = make_deterministic_libero_policy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
                state = torch.load(endpoint['path'], map_location=device, weights_only=True)
                counts['checkpoint_loads'] += 1
                p.require(digest(state) == endpoint['weight_hash'], 'checkpoint tensor hash differs')
                policy.load_state_dict(state, strict=True)
                policy.requires_grad_(False)
                policy.eval()
                p.require(digest(policy.state_dict()) == endpoint['weight_hash'], 'loaded Policy hash differs')
                predictions = []
                with torch.inference_mode():
                    for batch_index, batch in enumerate(DataLoader(items, batch_size=8, shuffle=False, num_workers=0)):
                        h = digest({k: batch[k] for k in BATCH_KEYS})
                        if len(task_model_ids) == 0:
                            shared_batch_digests.append(h)
                        else:
                            p.require(h == shared_batch_digests[batch_index], 'unpaired expert input batch')
                        b = {k: v.to(device) for k, v in batch.items()}
                        mask = b['valid_mask'].bool()
                        counts['main_forward_batches'] += 1
                        event('main_forward_attempt', task=task, seed=seed, arm=arm, batch=batch_index, input_sha256=h)
                        p.require(counts['main_forward_batches'] <= 819, 'main forward budget exceeded')
                        prediction, _ = policy.forward_sequence(*(b[k] for k in p.OBS_SHAPES))
                        p.require(torch.isfinite(prediction).all().item() and (prediction.abs() <= 1).all().item(),
                                  'invalid inference output')
                        counts['main_completed_batches'] += 1
                        counts['main_valid_prediction_rows'] += int(mask.sum())
                        counts['main_network_prediction_slots'] += int(prediction.shape[0] * prediction.shape[1])
                        predictions.append(prediction[mask].cpu().numpy().copy())
                        if batch_index == 0:
                            counts['repeat_forward_batches'] += 1
                            event('repeat_forward_attempt', task=task, seed=seed, arm=arm, batch=0)
                            p.require(counts['repeat_forward_batches'] <= 18, 'repeat budget exceeded')
                            repeat, _ = policy.forward_sequence(*(b[k] for k in p.OBS_SHAPES))
                            counts['repeat_completed_batches'] += 1
                            counts['repeat_valid_prediction_rows'] += int(mask.sum())
                            counts['repeat_network_prediction_slots'] += int(repeat.shape[0] * repeat.shape[1])
                            p.require(torch.equal(repeat, prediction), 'repeat output is not exact')
                            counts['repeat_exact_checks'] += 1
                        p.require(digest({k: b[k] for k in BATCH_KEYS}) == h, 'Policy mutated shared input/target batch')
                        event('main_forward_complete', task=task, seed=seed, arm=arm, batch=batch_index,
                              valid_rows=int(mask.sum()), output_sha256=digest({'prediction': prediction}))
                p.require(digest(policy.state_dict()) == endpoint['weight_hash'] and
                          all(param.grad is None for param in policy.parameters()), 'Policy parameters/gradients changed')
                pred = np.concatenate(predictions)
                stage_seconds['inference_and_batch_checks'] += time.monotonic() - model_started
                pred_path = artifacts / f'task{task}-seed{seed}-{arm}-predictions.npy'
                with pred_path.open('xb') as handle:
                    np.save(handle, pred, allow_pickle=False)
                completed_predictions.append({'path': str(pred_path), 'sha256': p.sha(pred_path)})
                analysis_started = time.monotonic()
                analysis = p.analyze_predictions(pred, targets, ids, masks)
                historical_parity(analysis['strata']['all'], old[task, seed, arm])
                counts['historical_model_parity_checks'] += 1
                rows.append({'task_id': task, 'training_seed': seed, 'arm': arm, 'checkpoint': endpoint,
                             'prediction_file': completed_predictions[-1], **analysis})
                stage_seconds['numerical_analysis'] += time.monotonic() - analysis_started
                task_predictions.append(pred)
                task_model_ids.append(f'{seed}:{arm}')
                event('model_complete', task=task, seed=seed, arm=arm, all_support=analysis['strata']['all'])
                print(json.dumps({'progress': 'model_complete', 'task': task, 'seed': seed, 'arm': arm,
                                  'main_batches': counts['main_forward_batches'], 'repeat_batches': counts['repeat_forward_batches']}), flush=True)
                del policy, state
            archive_path = artifacts / f'task{task}-expert-targets-and-predictions.npz'
            with archive_path.open('xb') as handle:
                np.savez_compressed(handle, targets=targets, predictions=np.stack(task_predictions),
                                    model_ids=np.array(task_model_ids), demonstration_indices=ids, action_steps=times,
                                    atomic_ids=atomic_ids, **{'mask_' + name: value for name, value in masks.items()})
            archives.append({'task_id': task, 'path': str(archive_path), 'sha256': p.sha(archive_path),
                             'shared_batch_digests': shared_batch_digests})
            del items, task_predictions
        expected = {'main_forward_batches': 819, 'main_completed_batches': 819, 'repeat_forward_batches': 18,
                    'repeat_completed_batches': 18, 'main_valid_prediction_rows': 62586, 'repeat_valid_prediction_rows': 1440,
                    'main_network_prediction_slots': 65340, 'repeat_network_prediction_slots': 1440, 'checkpoint_loads': 18,
                    'repeat_exact_checks': 18, 'historical_model_parity_checks': 18,
                    'source_observation_rows_per_modality': 6954, 'source_action_rows_including_label_only_t0': 7014,
                    'source_observation_rows_per_modality_attempted': 6954,
                    'source_action_rows_including_label_only_t0_attempted': 7014,
                    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
                    'new_policy_control_steps': 0, 'new_initial_or_private_state_reads': 0}
        p.require(counts == expected, 'actual forward/row/access budgets differ')
        result = {'status': 'complete', 'formal_result': False, 'config_sha256': p.sha(config_path),
                  'binding_sha256': p.sha(binding_path), 'scope': config['scope'], 'analysis_definition': config['analysis'],
                  'input_provenance': config['inputs'], 'execution_binding': binding, 'model_results': rows,
                  **p.across_models(rows), 'actual_counts': counts, 'archives': archives, 'access_records': access_records,
                  'runtime': {'requested_at_utc': requested_utc, 'started_at_utc': started_utc,
                              'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                              'entry_elapsed_seconds': time.monotonic() - entry_started, 'stage_seconds': stage_seconds,
                              **expected_runtime, 'gpu': gpu, 'device': 'cuda:0', 'deterministic_algorithms': True,
                              'tf32': False, 'peak_host_cache_bytes': peak_cache_bytes,
                              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                              'launch_argv': sys.argv, 'launch_environment': {k: os.environ.get(k) for k in
                                  ('CUDA_VISIBLE_DEVICES', 'CUBLAS_WORKSPACE_CONFIG', 'OMP_NUM_THREADS')}},
                  'interpretation': 'Source-support descriptive frame-error localization; no causal/training/qualification gate.'}
        event('complete', actual_counts=counts)
        events.close()
        result['events'] = {'path': str(event_path), 'sha256': p.sha(event_path)}
        write_result(output, result)
        return result
    except Exception as error:
        event('failed_partial', error_type=type(error).__name__, error=str(error), actual_counts=counts)
        events.close()
        failed = output.with_name(output.stem + '-failed-partial.json')
        write_result(failed, {'status': 'failed_partial', 'formal_result': False, 'error_type': type(error).__name__,
                             'error': str(error), 'config_sha256': p.sha(config_path), 'binding_sha256': p.sha(binding_path),
                             'actual_counts': counts, 'completed_predictions': completed_predictions, 'archives': archives,
                             'events': {'path': str(event_path), 'sha256': p.sha(event_path)}})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=p.DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=p.DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({'status': result['status'], 'output': str(args.output), 'sha256': p.sha(args.output),
                          'actual_counts': result['actual_counts']}))
    else:
        _, _, budget, _ = p.validate(args.config, args.binding)
        print(json.dumps({'status': 'bound_preflight_passed_no_payload_or_policy_calls', 'budget_by_task': budget}))


if __name__ == '__main__':
    main()
