"""Fixed500 Policy features plus causal current/history5 cross-demo probes.

Default validates metadata/code only. --execute consumes819+18 full forwards
on idleGPU1; all neighbor computations are fixed NumPy readouts, no optimizer.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import libero_phase_history_neighbors as n
from experiments import run_libero_source_internal_gripper_state_local_control as control

p, f = control.p, control.f
DEFAULT_CONFIG = ROOT / 'configs/libero-source-internal-phase-history-observability-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-internal-phase-history-observability-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-source-internal-phase-history-observability-20261104-v1.json')
DEFAULT_ARTIFACTS = Path('/tmp/libero-source-internal-phase-history-observability-artifacts-20261104-v1')
CONFIG_SHA = 'ab53c4c73f4df493749d1476cec50fd60bea79670a777f2817fd6df8450762f3'


def preflight(config_path, binding_path):
    c = p.load(config_path)
    expected = {'task_ids': list(p.TASKS), 'training_seeds': list(p.SEEDS), 'arms': list(p.ARMS),
        'checkpoint_updates': 500, 'closed_loop_default_stabilization_steps': 5, 'history_steps': 5,
        'source_role': 'meta_support', 'demonstration_indices': list(range(30)), 'common_first_action_step': 5,
        'neighbors': 5, 'query_chunk': 64, 'main_forward_batches': 819, 'repeat_forward_batches': 18,
        'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0,
        'new_initial_private_state_reads': 0, 'source_query_accessed': False, 'sealed_roles_accessed': False,
        'physical_gpu': 1, 'device': 'cuda:0', 'loss_observation_or_training_budget_changed': False,
        'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    p.require(c['formal_result'] is False, 'diagnostic cannot be formal')
    for k, v in expected.items():
        p.require(c['scope'].get(k) == v and type(c['scope'].get(k)) is type(v), f'frozen scope differs: {k}')
    p.require(p.sha(config_path) == CONFIG_SHA, 'scientific configuration hash differs')
    for k, pin in c['inputs'].items():
        p.require(p.sha(p.resolve(pin['path'])) == pin['sha256'], f'input pin differs: {k}')
    ctrl, samples, _, _ = p.validate(p.resolve(c['inputs']['control_design']['path']), p.resolve(c['inputs']['control_design_binding']['path']))
    p.require(c['endpoints'] == ctrl['endpoints'], 'all18 fixed500 endpoints required')
    old_binding = p.load(p.resolve(c['inputs']['control_binding']['path']))
    for pin in old_binding['files'].values():
        p.require(p.sha(p.resolve(pin['path'])) == pin['sha256'], 'historical execution code differs')
    binding = p.load(binding_path)
    p.require(binding['config_sha256'] == CONFIG_SHA and binding['formal_result'] is False and
        set(binding['files']) == {'runner', 'neighbors', 'tests', 'protocol'}, 'execution binding differs')
    for k, pin in binding['files'].items():
        p.require(p.sha(p.resolve(pin['path'])) == pin['sha256'], f'bound implementation differs: {k}')
    p.require(p.resolve(binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner path differs')
    return c, ctrl, samples, binding


def capture(policy, observations):
    import torch
    hidden, logits = [], []
    def prehook(_module, args):
        hidden.append(args[0].detach().clone())
    def posthook(_module, _args, output):
        logits.append(output.detach().clone())
    handles = [policy.action_head.register_forward_pre_hook(prehook), policy.action_head.register_forward_hook(posthook)]
    try:
        prediction, state = policy.forward_sequence(*observations)
        p.require(state is None and policy.config.memory_mode == 'cnn_only' and len(hidden) == len(logits) == 1,
                  'unexpected history or hook count in stateless Policy')
        p.require(hidden[0].shape == (*prediction.shape[:2], 256) and hidden[0].dtype == logits[0].dtype == torch.float32 and
                  torch.equal(torch.tanh(logits[0]), prediction),
                  'pre-head representation/logit reconstruction differs')
        p.require(torch.isfinite(hidden[0]).all().item() and torch.isfinite(prediction).all().item(), 'nonfinite capture')
        return prediction, hidden[0]
    finally:
        for handle in handles:
            handle.remove()


def save_probe(path, results):
    arrays = {label + '_' + mode + '_' + key: value for (label, mode), row in results.items() for key, value in row.items()}
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'path': str(path), 'sha256': p.sha(path)}


def run(config_path, binding_path, output, artifacts):
    output, artifacts = Path(output), Path(artifacts)
    failed = output.with_name(output.stem + '-failed-partial.json')
    p.require(not output.exists() and not failed.exists() and not artifacts.exists(), 'output/artifacts already exist')
    started = time.monotonic()
    artifacts.mkdir(parents=True, exist_ok=False)
    events_path = artifacts / 'events.jsonl'
    events = events_path.open('x')
    def event(kind, **fields):
        events.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}, allow_nan=False) + '\n')
        events.flush()
    def digest(values):
        h = hashlib.sha256()
        for k, v in values.items():
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    counts = {k: 0 for k in ('main_attempts', 'main_completed', 'repeat_attempts', 'repeat_completed',
        'main_valid_rows', 'repeat_valid_rows', 'main_network_slots', 'repeat_network_slots',
        'checkpoint_loads', 'historical_exact_prediction_rows', 'historical_batch_checks', 'repeat_exact_checks',
        'source_observation_rows_per_modality', 'source_action_rows', 'source_action0_rows',
        'primary_neighbor_folds', 'primary_neighbor_rows', 'secondary_neighbor_folds', 'secondary_neighbor_rows',
        'new_optimizer_updates', 'new_backward_calls', 'new_environment_calls', 'new_initial_private_state_reads')}
    features_done, secondary, primary = [], [], []
    active = None
    stage = 'preflight'
    try:
        c, ctrl, samples, binding = preflight(config_path, binding_path)
        full = p.load(p.resolve(c['inputs']['full_open_result']['path']))
        old_result = p.load(p.resolve(ctrl['inputs']['original_frame_result']['path']))
        cached = {}
        for task in p.TASKS:
            with np.load(p.resolve(ctrl['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
                data = {k: z[k] for k in ('targets', 'predictions', 'model_ids', 'demonstration_indices', 'action_steps')}
            segments = next(t for t in full['task_segment_plans'] if t['task_id'] == task)
            plan = n.frame_plan(data['targets'], data['demonstration_indices'], data['action_steps'], segments)
            for label in n.LABELS:
                p.require(int((plan[label][plan['eligible']] != 0).sum()) == c['scope']['eligible_by_task'][str(task)][label], 'eligible label coverage differs')
            with np.load(p.resolve(c['inputs'][f'gripper_task{task}']['path']), allow_pickle=False) as z:
                gripper = z['original_gripper']
            cached[task] = (data, plan, gripper)
            stage = f'task{task}_cached_public_gripper_neighbors'
            results, folds, metrics = n.probe(gripper, plan)
            pin = save_probe(artifacts / f'task{task}-public-gripper-neighbors.npz', results)
            secondary.append({'task_id': task, 'space': 'public_gripper2', 'metrics': metrics, 'folds': folds, 'archive': pin})
            counts['secondary_neighbor_folds'] += len(folds) * 4
            counts['secondary_neighbor_rows'] += sum(len(v['source_rows']) for v in results.values())
            event('secondary_complete', task=task, archive=pin)
        stage = 'resource_gate'
        control.launch_gate()
        from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
        gpu = idle_gpu(1)
        event('resource_gate_passed', gpu=gpu)
        import torch
        import h5py
        from torch.utils.data import DataLoader
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
        from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
        from src.environments.libero_dataset import resolve_demonstration_access
        runtime = {'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
                   'numpy': np.__version__, 'h5py': h5py.__version__}
        p.require(runtime == c['execution']['runtime'] and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'runtime/device differs')
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
        transfer, manifest, learning = [p.load(p.resolve(ctrl['inputs'][k]['path'])) for k in ('transfer', 'dataset_manifest', 'learning')]
        root = Path('/home/zl202621/third_party/LIBERO/datasets').resolve()
        access_records, peak_cache = [], 0
        for task in p.TASKS:
            stage = f'task{task}_source_cache'
            sample = next(s for s in samples['tasks'] if s['task_id'] == task)
            data, plan, old_gripper = cached[task]
            access = resolve_demonstration_access(transfer, manifest, task_id=task, demonstration_role='meta_support')
            p.require(access.task_role == 'source_spi' and list(access.demonstration_indices) == list(range(30)), 'source role differs')
            m = next(r for r in manifest['files'] if r['task_id'] == task)
            path = (root / m['path']).resolve()
            p.require(path.is_relative_to(root) and path.stat().st_size == m['size_bytes'] and p.sha(path) == m['sha256'], 'source file provenance differs')
            ds = CausalT1LiberoSequenceDataset(transfer, manifest, root, task_id=task, demonstration_role='meta_support', sequence_length=10, stride=10, verify_sha256=False)
            try:
                p.require([asdict(w) for w in ds.windows] == sample['canonical_windows'], 'source window plan differs')
                items = []
                for wi, w in enumerate(ds.windows):
                    event('source_window_attempt', task=task, demo=w.demonstration_index, observation_slice=[w.start - 1, w.start + w.valid_length - 1], action_slice=[w.start, w.start + w.valid_length])
                    items.append(ds[wi])
                    counts['source_observation_rows_per_modality'] += w.valid_length
                    counts['source_action_rows'] += w.valid_length
                access_records.append({**ds.access_record, 'file_sha256_verified_before_payload': True, 'source_action0_rows_read': 0})
            finally:
                ds.close()
            cache_bytes = sum(v.numel() * v.element_size() for item in items for v in item.values() if torch.is_tensor(v))
            peak_cache = max(peak_cache, cache_bytes)
            p.require(cache_bytes <= c['scope']['maximum_host_cache_bytes'], 'host cache budget exceeded')
            target = np.concatenate([item['actions'][:w['valid_length']].numpy() for item, w in zip(items, sample['canonical_windows'])])
            gripper = np.concatenate([item['gripper_states'][:w['valid_length']].numpy() for item, w in zip(items, sample['canonical_windows'])])
            p.require(np.array_equal(target, data['targets']) and np.array_equal(gripper, old_gripper), 'source target/gripper parity differs')
            row_batches = control.batch_frame_rows(sample)
            old_hashes = next(t['shared_batch_digests'] for t in old_result['archives'] if t['task_id'] == task)
            for mi, endpoint in enumerate(e for e in c['endpoints'] if e['task_id'] == task):
                seed, arm = endpoint['training_seed'], endpoint['arm']
                stage = f'task{task}_seed{seed}_{arm}_features'
                p.require(p.sha(endpoint['path']) == endpoint['sha256'] and str(data['model_ids'][mi]) == f'{seed}:{arm}', 'fixed endpoint/order differs')
                spec = {**learning['policy'], 'memory_mode': 'cnn_only', 'policy_family': 'dual_view_shared_cnn_only_uniform_bc_diagnostic_v1'}
                policy = make_deterministic_libero_policy(LiberoBCPolicyConfig.from_mapping(spec)).to(device)
                state = torch.load(endpoint['path'], map_location=device, weights_only=True)
                counts['checkpoint_loads'] += 1
                p.require(digest(state) == endpoint['weight_hash'], 'checkpoint tensor hash differs')
                policy.load_state_dict(state, strict=True)
                policy.requires_grad_(False)
                policy.eval()
                active = {'projection': np.full((len(target), 256), np.nan, np.float32), 'predictions': np.full((len(target), 7), np.nan, np.float32)}
                with torch.inference_mode():
                    for bi, batch in enumerate(DataLoader(items, batch_size=8, shuffle=False, num_workers=0)):
                        rows = row_batches[bi]
                        valid = rows >= 0
                        h = digest({k: batch[k] for k in control.BATCH_KEYS})
                        p.require(h == old_hashes[bi] and np.array_equal(batch['valid_mask'].numpy(), valid), 'original batch digest/mask differs')
                        counts['historical_batch_checks'] += 1
                        b = {k: v.to(device) for k, v in batch.items()}
                        obs = [b[k] for k in f.OBS_SHAPES]
                        p.require(counts['main_attempts'] < 819, 'main budget exhausted')
                        counts['main_attempts'] += 1
                        event('main_forward_attempt', task=task, seed=seed, arm=arm, batch=bi)
                        prediction, hidden = capture(policy, obs)
                        counts['main_completed'] += 1
                        pred, feat = prediction.cpu().numpy().copy(), hidden.cpu().numpy().copy()
                        active['projection'][rows[valid]] = feat[valid]
                        active['predictions'][rows[valid]] = pred[valid]
                        p.require(np.array_equal(pred[valid], data['predictions'][mi, rows[valid]]), 'historical predictions not exact')
                        counts['historical_exact_prediction_rows'] += int(valid.sum())
                        counts['main_valid_rows'] += int(valid.sum())
                        counts['main_network_slots'] += rows.size
                        if bi == 0:
                            p.require(counts['repeat_attempts'] < 18, 'repeat budget exhausted')
                            counts['repeat_attempts'] += 1
                            event('repeat_forward_attempt', task=task, seed=seed, arm=arm, batch=bi)
                            rp, rh = capture(policy, obs)
                            counts['repeat_completed'] += 1
                            counts['repeat_valid_rows'] += int(valid.sum())
                            counts['repeat_network_slots'] += rows.size
                            p.require(torch.equal(rp, prediction) and torch.equal(rh, hidden), 'feature/action repeat differs')
                            counts['repeat_exact_checks'] += 1
                        p.require(digest({k: b[k] for k in control.BATCH_KEYS}) == h, 'capture mutated input/targets')
                        event('main_forward_complete', task=task, seed=seed, arm=arm, batch=bi, valid_rows=int(valid.sum()), input_sha256=h)
                p.require(all(np.isfinite(v).all() for v in active.values()) and digest(policy.state_dict()) == endpoint['weight_hash'] and
                          all(v.grad is None for v in policy.parameters()), 'feature coverage/weights/gradients differ')
                path = artifacts / f'task{task}-seed{seed}-{arm}-features.npz'
                with path.open('xb') as handle:
                    np.savez_compressed(handle, **active)
                pin = {'path': str(path), 'sha256': p.sha(path)}
                features_done.append({'task_id': task, 'training_seed': seed, 'arm': arm, 'features': pin, 'checkpoint': endpoint})
                event('features_complete', task=task, seed=seed, arm=arm, features=pin)
                print(json.dumps({'progress': 'features_complete', 'task': task, 'seed': seed, 'arm': arm, 'main_batches': counts['main_attempts']}), flush=True)
                active = None
                del policy, state
            del items
        torch.cuda.empty_cache()
        feature_finished_utc = datetime.now(timezone.utc).isoformat()
        for model in features_done:
            task, seed, arm = model['task_id'], model['training_seed'], model['arm']
            stage = f'task{task}_seed{seed}_{arm}_neighbors'
            with np.load(model['features']['path'], allow_pickle=False) as z:
                features, predictions = z['projection'], z['predictions']
            results, folds, metrics = n.probe(features, cached[task][1], predictions[:, 6])
            pin = save_probe(artifacts / f'task{task}-seed{seed}-{arm}-neighbors.npz', results)
            primary.append({**model, 'metrics': metrics, 'folds': folds, 'neighbors': pin})
            counts['primary_neighbor_folds'] += len(folds) * 4
            counts['primary_neighbor_rows'] += sum(len(v['source_rows']) for v in results.values())
            event('primary_neighbors_complete', task=task, seed=seed, arm=arm, neighbors=pin)
            print(json.dumps({'progress': 'neighbors_complete', 'task': task, 'seed': seed, 'arm': arm}), flush=True)
        expected = {'main_attempts': 819, 'main_completed': 819, 'repeat_attempts': 18, 'repeat_completed': 18,
            'main_valid_rows': 62586, 'repeat_valid_rows': 1440, 'main_network_slots': 65340, 'repeat_network_slots': 1440,
            'checkpoint_loads': 18, 'historical_exact_prediction_rows': 62586, 'historical_batch_checks': 819, 'repeat_exact_checks': 18,
            'source_observation_rows_per_modality': 6954, 'source_action_rows': 6954, 'source_action0_rows': 0,
            'primary_neighbor_folds': 2160, 'primary_neighbor_rows': 179280, 'secondary_neighbor_folds': 240, 'secondary_neighbor_rows': 19920,
            'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0, 'new_initial_private_state_reads': 0}
        p.require(counts == expected, 'actual forward/readout/source budget differs')
        result = {'status': 'complete_phase_history_observability_diagnostic', 'formal_result': False,
            'config_sha256': p.sha(config_path), 'binding_sha256': p.sha(binding_path), 'scope': c['scope'],
            'analysis_definition': c['analysis'], 'input_provenance': c['inputs'], 'actual_counts': counts,
            'primary_model_results': primary, 'public_gripper_results': secondary, 'paired_seed_flags': n.across_models(primary),
            'access_records': access_records,
            'runtime': {**runtime, 'device': 'cuda:0 feature capture / cpu float64 neighbors', 'gpu': gpu,
                'feature_finished_at_utc': feature_finished_utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                'entry_elapsed_seconds': time.monotonic() - started, 'peak_host_source_cache_bytes': peak_cache,
                'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'deterministic_algorithms': True, 'tf32': False,
                'launch_environment': c['execution']['launch_environment'], 'launch_argv': sys.argv},
            'interpretation': 'Metric-specific geometry on already exposed source-support representations. No unseen generalization, necessary-memory claim, physical cause, training or qualification result.'}
        event('complete', actual_counts=counts)
        events.close()
        result['events'] = {'path': str(events_path), 'sha256': p.sha(events_path)}
        control.write_json(output, result)
        return result
    except Exception as error:
        partial = None
        if active is not None:
            partial = artifacts / 'failed-active-features.npz'
            with partial.open('xb') as handle:
                np.savez_compressed(handle, **active)
        event('failed_partial', stage=stage, error_type=type(error).__name__, error=str(error), actual_counts=counts)
        events.close()
        control.write_json(failed, {'status': 'failed_partial', 'formal_result': False, 'stage': stage,
            'error_type': type(error).__name__, 'error': str(error), 'actual_counts': counts,
            'features_completed': features_done, 'secondary_completed': secondary, 'primary_completed': primary,
            'partial_features': str(partial) if partial else None, 'events': str(events_path),
            'config_sha256': p.sha(config_path), 'binding_sha256': p.sha(binding_path)})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({'status': result['status'], 'actual_counts': result['actual_counts']}))
    else:
        preflight(args.config, args.binding)
        print(json.dumps({'status': 'binding_and_inputs_validated_not_executed', 'new_policy_calls': 0}))
