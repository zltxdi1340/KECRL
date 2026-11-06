"""Full-support paired readouts, then source-query features on frozen CPU encoders."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import run_libero_learned_phase_history_readout as parent
from experiments import libero_phase_history_neighbors as prior
from experiments.analyze_libero_source_internal_full_open_segments import target_segments

require, sha, load, resolve = parent.require, parent.sha, parent.load, parent.resolve
MODES, SEEDS = parent.MODES, parent.SEEDS
DEFAULT_CONFIG = ROOT / 'configs/libero-task1-b-support-query-history-readout-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task1-b-support-query-history-readout-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task1-b-support-query-history-readout-20261104-v1.json')
DEFAULT_ARTIFACTS = Path('/tmp/libero-task1-b-support-query-history-readout-artifacts-20261104-v1')
CONFIG_SHA = 'e0e717f1c8eadf6447f38dec0b1e6a8d69ea5b92a71054e8265387b3f4174b72'


def write_json(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def write_npz(path, data):
    with Path(path).open('xb') as handle:
        np.savez_compressed(handle, **data)
    return {'path': str(path), 'sha256': sha(path)}


def tensor_digest(values):
    h = hashlib.sha256()
    for key, value in values.items():
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def make_plan(targets, ids, steps):
    known, prefixes = target_segments(targets, ids, steps)
    return prior.frame_plan(targets, ids, steps, {'known_segments': known, 'initial_observed_prefixes': prefixes})


def transform(features, histories, mean, scale):
    require(features.dtype == np.float32 and features.ndim == 2 and features.shape[1] == 256 and
            np.isfinite(features).all() and histories.ndim == 2 and histories.shape[1] == 5 and
            (histories >= 0).all() and (histories < len(features)).all() and
            mean.shape == scale.shape == (256,) and np.isfinite(mean).all() and np.isfinite(scale).all() and
            (scale > 0).all(), 'readout feature/scaler/history interface differs')
    z = (features[histories].astype(np.float64) - mean) / scale
    raw = {'current_repeat5': np.repeat(z[:, :1], 5, axis=1), 'causal_history5': z}
    return {k: np.ascontiguousarray((v.reshape(len(histories), 1280) / np.sqrt(5)).astype(np.float32)) for k, v in raw.items()}


def full_support_inputs(features, plan):
    values = features[plan['eligible']].astype(np.float64)
    mean, scale = values.mean(0), values.std(0)
    scale[scale <= 1e-6] = 1.
    return transform(features, plan['history_rows'], mean, scale), mean, scale


def query_gate(heads, counts):
    require(len(heads) == 6 and {(h['training_seed'], h['mode']) for h in heads} ==
            {(s, m) for s in SEEDS for m in MODES} and all(h['actual']['optimizer_updates'] == 500 for h in heads),
            'query gate: all six fixed support fits must finish')
    require(counts['readout_models'] == 6 and counts['optimizer_updates'] == counts['backward_calls'] == 3000,
            'query gate: actual support update counts differ')


def open_query(factory, heads, counts, bridges, *args, **kwargs):
    query_gate(heads, counts)
    require(len(bridges) == 3 and {b['training_seed'] for b in bridges} == set(SEEDS) and
            all(b['passed'] is True for b in bridges), 'query gate: numerical bridges incomplete')
    require(kwargs.get('task_id') == 1 and kwargs.get('demonstration_role') == 'meta_query', 'query task/role differs')
    return factory(*args, **kwargs)


def canonical_rows(windows):
    rows, ids, steps, offset = [], [], [], 0
    for start in range(0, len(windows), 8):
        batch = np.full((min(8, len(windows) - start), 10), -1, np.int32)
        for j, w in enumerate(windows[start:start + 8]):
            require(w['start'] >= 1 and 1 <= w['valid_length'] <= 10, 'invalid causal window')
            length = w['valid_length']
            batch[j, :length] = np.arange(offset, offset + length)
            ids.extend([w['demonstration_index']] * length)
            steps.extend(range(w['start'], w['start'] + length))
            offset += length
        rows.append(batch)
    require(np.array_equal(np.concatenate([x[x >= 0] for x in rows]), np.arange(offset)), 'window row coverage differs')
    return rows, np.array(ids, np.int16), np.array(steps, np.int32)


def paired_cue(seed_metrics):
    require([r['training_seed'] for r in seed_metrics] == list(SEEDS), 'retain all three seeds')
    changes = []
    for row in seed_metrics:
        a, b = [row['query_metrics'][m] for m in MODES]
        values = {'open_event_sign_error': (a['known_open_event']['gripper_sign_error_rate'], b['known_open_event']['gripper_sign_error_rate']),
                  'known_open_segment_sign_error': (a['known_open_segment']['gripper_sign_error_rate'], b['known_open_segment']['gripper_sign_error_rate']),
                  'overall_7D_MSE': (a['all']['seven_dimensional_MSE'], b['all']['seven_dimensional_MSE']),
                  'overall_translation_MSE': (a['all']['translation_MSE'], b['all']['translation_MSE'])}
        changes.append({'training_seed': row['training_seed'], 'deltas': {k: b - a if a is not None and b is not None else None for k, (a, b) in values.items()}})
    flags = {}
    for key in changes[0]['deltas']:
        values = [c['deltas'][key] for c in changes]
        flags[key] = bool(all(x < 0 if key == 'open_event_sign_error' else x <= 0 for x in values)) if all(x is not None for x in values) else None
    return {'per_seed': changes, 'flags': flags, 'joint_query_cue': all(flags.values()) if all(x is not None for x in flags.values()) else None,
            'automatic_rollout_or_spt_promotion': False}


def preflight(config_path, binding_path):
    c = load(config_path)
    require(c['formal_result'] is False, 'run must remain diagnostic')
    expected = {'task_id': 1, 'feature_arm': 'B_fomaml', 'training_seeds': list(SEEDS), 'original_policy_updates': 500,
        'readout_modes': list(MODES), 'support_demos': list(range(30)), 'query_demos': list(range(30, 50)), 'history_steps': 5,
        'common_first_action_step': 5, 'readout_models': 6, 'updates_per_readout': 500, 'optimizer_updates': 3000,
        'backward_calls': 3000, 'train_forward_batches': 3000, 'train_valid_rows': 192000, 'metric_forward_batches': 24,
        'metric_valid_rows': 94536, 'query_readout_forwards': 6, 'query_readout_rows': 15108,
        'support_t_ge_1_rows': 4059, 'support_common_rows': 3939, 'query_t_ge_1_rows': 2598, 'query_common_rows': 2518,
        'query_windows': 269, 'encoder_sequence_length': 10, 'encoder_batch_size': 8, 'encoder_stride': 10,
        'query_encoder_main_attempts': 102, 'query_encoder_main_completed': 102, 'query_encoder_main_valid_rows': 7794,
        'query_encoder_main_slots': 8070, 'query_encoder_repeat_attempts': 3, 'query_encoder_repeat_completed': 3,
        'query_encoder_repeat_valid_rows': 240, 'query_encoder_repeat_slots': 240, 'support_bridge_attempts': 3,
        'support_bridge_completed': 3, 'support_bridge_valid_rows': 240, 'checkpoint_loads': 3,
        'new_support_observation_rows_per_modality': 80, 'new_support_action_rows': 80,
        'new_query_observation_rows_per_modality': 2598, 'new_query_action_rows': 2598,
        'new_original_policy_optimizer_updates': 0, 'new_environment_calls': 0, 'new_initial_private_state_reads': 0,
        'action0_rows': 0, 'new_nearest_neighbor_searches': 0, 'closed_loop_default_stabilization_steps': 5,
        'query_opened_after_all_six_fits': True, 'sealed_roles_accessed': False, 'checkpoint_selected': False,
        'original_policy_modified': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'scientific config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {name}')
    for name in ('learned_binding', 'phase_history_binding', 'source_original_binding'):
        for pin in load(c['inputs'][name]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'historical bound implementation differs')
    b = load(binding_path)
    require(b['config_sha256'] == CONFIG_SHA and b['formal_result'] is False and
            set(b['files']) == {'runner', 'tests', 'protocol'}, 'execution binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound execution file differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner path differs')
    return c, b


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failed = output.with_name(output.stem + '-failed.json')
    require(not output.exists() and not failed.exists() and not artifacts.exists(), 'outputs/artifacts already exist; preserve them')
    started = time.monotonic()
    counters = ('readout_models', 'optimizer_updates', 'backward_calls', 'train_forward_batches', 'train_valid_rows',
        'metric_forward_batches', 'metric_valid_rows', 'query_readout_forwards', 'query_readout_rows',
        'query_encoder_main_attempts', 'query_encoder_main_completed', 'query_encoder_main_valid_rows', 'query_encoder_main_slots',
        'query_encoder_repeat_attempts', 'query_encoder_repeat_completed', 'query_encoder_repeat_valid_rows', 'query_encoder_repeat_slots',
        'support_bridge_attempts', 'support_bridge_completed', 'support_bridge_valid_rows', 'checkpoint_loads',
        'new_support_observation_rows_per_modality', 'new_support_action_rows', 'new_query_observation_rows_per_modality',
        'new_query_action_rows', 'new_original_policy_optimizer_updates', 'new_environment_calls', 'new_initial_private_state_reads',
        'action0_rows', 'new_nearest_neighbor_searches')
    counts = dict.fromkeys(counters, 0)
    heads, bridges, query_features, seed_results, access_records, active = [], [], [], [], [], {}
    event_handle, partial_query = None, None
    stage = 'preflight'
    try:
        c, binding = preflight(config_path, binding_path)
        learned = load(c['inputs']['learned_result']['path'])
        learned_audit = load(c['inputs']['learned_audit']['path'])
        require(learned['status'] == 'complete_learned_phase_history_readout_diagnostic' and
                learned_audit['status'] == 'complete_independent_learned_readout_audit' and
                learned_audit['result']['sha256'] == sha(c['inputs']['learned_result']['path']), 'learned parent/audit incomplete')
        require(next(x for x in learned['primary_flags'] if x['task_id'] == 1 and x['feature_arm'] == 'B_fomaml')['joint_history_cue'] is True,
                'task1 B was not the retained support candidate')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1' for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'fixed CPU launch environment differs')
        gpu_occupancy = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
        import torch
        import h5py
        require({'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__, 'h5py': h5py.__version__} == c['execution']['runtime'] and
                torch.cuda.device_count() == 0, 'runtime/device differs')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        artifacts.mkdir(parents=True, exist_ok=False)
        event_path = artifacts / 'events.jsonl'
        event_handle = event_path.open('x')
        def event(kind, **fields):
            event_handle.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}, allow_nan=False) + '\n')
            event_handle.flush()
        event('runtime_ready', device='cpu', gpu_occupancy=gpu_occupancy)
        with np.load(resolve(c['inputs']['task1_archive']['path']), allow_pickle=False) as z:
            targets, ids, steps, old_predictions, model_ids = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps', 'predictions', 'model_ids')]
        plan = make_plan(targets, ids, steps)
        source = plan['eligible']
        require(set(ids.tolist()) == set(range(30)) and len(steps) == 4059 and len(source) == 3939, 'full support role/count differs')
        y = np.ascontiguousarray(targets[source])
        saved_by_seed = {}
        for fixed in c['fixed_encoders']:
            seed = fixed['checkpoint']['training_seed']
            pin = fixed['support_features']
            require(sha(pin['path']) == pin['sha256'], 'cached support feature hash differs')
            with np.load(pin['path'], allow_pickle=False) as z:
                features, predictions = z['projection'], z['predictions']
            mi = list(SEEDS).index(seed) * 3 + 1
            require(str(model_ids[mi]) == f'{seed}:B_fomaml' and np.array_equal(predictions, old_predictions[mi]), 'cached original Policy identity differs')
            before = parent.array_hash({'features': features})
            inputs, mean, scale = full_support_inputs(features, plan)
            rng = np.random.default_rng(seed)
            batches = rng.integers(0, len(source), size=(500, 64))
            batch_hash = parent.array_hash({'batch_source_rows': source[batches]})
            state = {'features': features, 'original_predictions': predictions, 'mean': mean, 'scale': scale, 'heads': {}}
            initial_hash = None
            for mode in MODES:
                stage = f'support_seed{seed}_{mode}'
                active.clear()
                active['context'] = {'training_seed': seed, 'mode': mode}
                torch.manual_seed(seed)
                arrays, actual = parent.train_readout(inputs[mode], y, batches, parent.CURVE, c['analysis']['optimizer'], counts, active)
                require(actual['parameters'] == 8967 and actual['optimizer_updates'] == 500 and
                        np.count_nonzero(arrays['predictions'][0]) == 0, 'paired architecture/init/budget differs')
                if initial_hash is None:
                    initial_hash = actual['initial_weight_hash']
                require(actual['initial_weight_hash'] == initial_hash, 'paired initialization differs')
                curves = {str(u): parent.stratified(p, y, plan, np.ones(len(source), bool)) for u, p in zip(parent.CURVE, arrays['predictions'])}
                arrays.update(source_rows=source, batch_source_rows=source[batches], scaler_mean=mean, scaler_scale=scale)
                pin = write_npz(artifacts / f'seed{seed}-{mode}-support-readout.npz', arrays)
                heads.append({'training_seed': seed, 'mode': mode, 'actual': actual, 'batch_source_rows_hash': batch_hash,
                              'curve_metrics': curves, 'archive': pin})
                state['heads'][mode] = arrays
                counts['readout_models'] += 1
                event('support_fit_complete', seed=seed, mode=mode, updates=counts['optimizer_updates'], archive=pin)
            require(parent.array_hash({'features': features}) == before, 'support encoder cache mutated')
            saved_by_seed[seed] = state
        query_gate(heads, counts)
        event('all_support_fits_complete', readout_models=6, optimizer_updates=3000)
        # Dataset modules are imported only after the complete support fit gate.
        from torch.utils.data import DataLoader, default_collate
        from src.environments.libero_dataset import resolve_demonstration_access
        from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
        from experiments import run_libero_source_internal_phase_history_observability as capture
        transfer, manifest, learning = [load(c['inputs'][k]['path']) for k in ('transfer', 'dataset_manifest', 'learning')]
        for role, expected_ids in [('meta_support', range(30)), ('meta_query', range(30, 50))]:
            # This resolves metadata only after all fits; no query dataset exists yet.
            access = resolve_demonstration_access(transfer, manifest, task_id=1, demonstration_role=role)
            require(access.task_role == 'source_spi' and list(access.demonstration_indices) == list(expected_ids), 'source demo access differs')
        root = Path(c['execution']['dataset_root']).resolve()
        file_spec = next(x for x in manifest['files'] if x['task_id'] == 1)
        dataset_path = (root / file_spec['path']).resolve()
        require(dataset_path.is_relative_to(root) and dataset_path.stat().st_size == file_spec['size_bytes'] and
                sha(dataset_path) == file_spec['sha256'], 'source HDF5 provenance differs')
        sample = next(x for x in load(c['inputs']['sample_manifest']['path'])['tasks'] if x['task_id'] == 1)
        old_input_digest = next(x for x in load(c['inputs']['original_frame_result']['path'])['archives'] if x['task_id'] == 1)['shared_batch_digests'][0]
        ds = CausalT1LiberoSequenceDataset(transfer, manifest, root, task_id=1, demonstration_role='meta_support', sequence_length=10, stride=10, verify_sha256=False)
        try:
            require([asdict(w) for w in ds.windows[:8]] == sample['canonical_windows'][:8], 'source bridge windows differ')
            bridge_items = [ds[i] for i in range(8)]
            bridge_batch = default_collate(bridge_items)
            access_records.append({**ds.access_record, 'payload_windows': sample['canonical_windows'][:8], 'file_sha256_verified_before_payload': True})
        finally:
            ds.close()
        require(tensor_digest({k: bridge_batch[k] for k in capture.control.BATCH_KEYS}) == old_input_digest, 'source bridge input is not exact')
        bridge_rows = capture.control.batch_frame_rows(sample)[0]
        valid = bridge_rows >= 0
        require(valid.sum() == 80 and np.array_equal(bridge_rows[valid], np.arange(80)), 'bridge row layout differs')
        counts['new_support_observation_rows_per_modality'] = counts['new_support_action_rows'] = 80
        policies = {}
        for fixed in c['fixed_encoders']:
            stage = 'support_cpu_numerical_bridge'
            ck = fixed['checkpoint']
            seed = ck['training_seed']
            require(sha(ck['path']) == ck['sha256'], 'frozen encoder file hash differs')
            spec = {**learning['policy'], 'memory_mode': 'cnn_only', 'policy_family': 'dual_view_shared_cnn_only_uniform_bc_diagnostic_v1'}
            policy = make_deterministic_libero_policy(LiberoBCPolicyConfig.from_mapping(spec))
            state = torch.load(ck['path'], map_location='cpu', weights_only=True)
            counts['checkpoint_loads'] += 1
            require(tensor_digest(state) == ck['weight_hash'], 'frozen encoder tensor hash differs')
            policy.load_state_dict(state, strict=True)
            policy.requires_grad_(False)
            policy.eval()
            counts['support_bridge_attempts'] += 1
            with torch.inference_mode():
                pred, feat = capture.capture(policy, [bridge_batch[k] for k in capture.f.OBS_SHAPES])
            counts['support_bridge_completed'] += 1
            counts['support_bridge_valid_rows'] += 80
            pp, hh = pred.numpy()[valid].copy(), feat.numpy()[valid].copy()
            saved = saved_by_seed[seed]
            feature_delta = float(np.max(np.abs(hh - saved['features'][:80])))
            action_delta = float(np.max(np.abs(pp - saved['original_predictions'][:80])))
            require(feature_delta <= 1e-5 and action_delta <= 1e-5, 'CPU/cache feature or action bridge differs')
            local_histories = plan['history_rows'][:76]
            require(local_histories.max() < 80, 'support bridge history exceeds captured rows')
            inputs = transform(hh, local_histories, saved['mean'], saved['scale'])
            readout_delta = {}
            for mode in MODES:
                arrays = saved['heads'][mode]
                rebuilt = np.tanh(inputs[mode].astype(np.float64) @ arrays['weights'][-1].astype(np.float64).T + arrays['biases'][-1].astype(np.float64))
                readout_delta[mode] = float(np.max(np.abs(rebuilt - arrays['predictions'][-1, :76])))
                require(readout_delta[mode] <= 1e-5, 'CPU/cache new readout bridge differs')
            require(tensor_digest(policy.state_dict()) == ck['weight_hash'] and all(p.grad is None for p in policy.parameters()) and
                    tensor_digest({k: bridge_batch[k] for k in capture.control.BATCH_KEYS}) == old_input_digest, 'bridge mutated weights/input')
            pin = write_npz(artifacts / f'seed{seed}-support-cpu-bridge.npz', {'projection': hh, 'predictions': pp, 'source_rows': np.arange(80, dtype=np.int32)})
            bridges.append({'training_seed': seed, 'passed': True, 'feature_max_abs_delta': feature_delta,
                            'action_max_abs_delta': action_delta, 'new_readout_max_abs_delta': readout_delta,
                            'input_digest': old_input_digest, 'archive': pin})
            policies[seed] = policy
            event('support_numerical_bridge_passed', seed=seed, bridge=bridges[-1])
        event('all_bridges_passed_before_query_dataset')
        stage = 'query_dataset_creation_after_fits_and_bridges'
        ds = open_query(CausalT1LiberoSequenceDataset, heads, counts, bridges, transfer, manifest, root,
                        task_id=1, demonstration_role='meta_query', sequence_length=10, stride=10, verify_sha256=False)
        query_items = []
        try:
            windows = [asdict(w) for w in ds.windows]
            require(len(windows) == 269 and sum(w['valid_length'] for w in windows) == 2598 and
                    set(w['demonstration_index'] for w in windows) == set(range(30, 50)), 'query role/window budget differs')
            event('query_dataset_created', support_models=counts['readout_models'], support_updates=counts['optimizer_updates'], windows=len(windows))
            for wi, window in enumerate(windows):
                event('query_window_read_attempt', window=window, observation_slice=[window['start'] - 1, window['start'] + window['valid_length'] - 1],
                      action_slice=[window['start'], window['start'] + window['valid_length']])
                query_items.append(ds[wi])
                counts['new_query_observation_rows_per_modality'] += window['valid_length']
                counts['new_query_action_rows'] += window['valid_length']
            access_records.append({**ds.access_record, 'file_sha256_verified_before_payload': True, 'support_fits_complete_before_creation': True,
                                   'support_models_before_creation': 6, 'support_updates_before_creation': 3000, 'action0_rows_read': 0})
        finally:
            ds.close()
        cache_bytes = sum(v.numel() * v.element_size() for item in query_items for v in item.values() if torch.is_tensor(v))
        require(cache_bytes <= c['execution']['maximum_query_tensor_cache_bytes'], 'query tensor cache budget exceeded')
        row_batches, qids, qsteps = canonical_rows(windows)
        qtarget = np.concatenate([item['actions'][:w['valid_length']].numpy() for item, w in zip(query_items, windows)])
        require(qtarget.shape == (2598, 7), 'query target count differs')
        qplan = make_plan(qtarget, qids, qsteps)
        qsource = qplan['eligible']
        require(len(qsource) == 2518, 'query common frame count differs')
        shared_query = write_npz(artifacts / 'query-targets-and-frame-ids.npz', {'targets': qtarget, 'demonstration_indices': qids, 'action_steps': qsteps})
        original_run = load(c['inputs']['source_original_run']['path'])
        original_fold = next(f for f in original_run['fold_results'] if f['probe'] == 1)
        query_batch_hashes = []
        for fixed in c['fixed_encoders']:
            seed = fixed['checkpoint']['training_seed']
            policy = policies[seed]
            stage = f'query_features_seed{seed}'
            partial_query = {'projection': np.full((2598, 256), np.nan, np.float32), 'predictions': np.full((2598, 7), np.nan, np.float32)}
            with torch.inference_mode():
                for bi, batch in enumerate(DataLoader(query_items, batch_size=8, shuffle=False, num_workers=0)):
                    rows = row_batches[bi]
                    valid = rows >= 0
                    digest = tensor_digest({k: batch[k] for k in capture.control.BATCH_KEYS})
                    if len(query_batch_hashes) <= bi:
                        query_batch_hashes.append(digest)
                    require(digest == query_batch_hashes[bi] and np.array_equal(valid, batch['valid_mask'].numpy()), 'query batch identity differs')
                    require(counts['query_encoder_main_attempts'] < 102, 'query encoder main budget exhausted')
                    counts['query_encoder_main_attempts'] += 1
                    event('query_main_forward_attempt', seed=seed, batch=bi, input_digest=digest)
                    pred, feat = capture.capture(policy, [batch[k] for k in capture.f.OBS_SHAPES])
                    counts['query_encoder_main_completed'] += 1
                    counts['query_encoder_main_valid_rows'] += int(valid.sum())
                    counts['query_encoder_main_slots'] += rows.size
                    partial_query['projection'][rows[valid]] = feat.numpy()[valid]
                    partial_query['predictions'][rows[valid]] = pred.numpy()[valid]
                    if bi == 0:
                        counts['query_encoder_repeat_attempts'] += 1
                        event('query_repeat_forward_attempt', seed=seed, batch=bi)
                        rp, rh = capture.capture(policy, [batch[k] for k in capture.f.OBS_SHAPES])
                        counts['query_encoder_repeat_completed'] += 1
                        counts['query_encoder_repeat_valid_rows'] += int(valid.sum())
                        counts['query_encoder_repeat_slots'] += rows.size
                        require(torch.equal(rp, pred) and torch.equal(rh, feat), 'query CPU repetition not bitwise exact')
                    require(tensor_digest({k: batch[k] for k in capture.control.BATCH_KEYS}) == digest, 'query capture mutated input/target')
            require(all(np.isfinite(v).all() for v in partial_query.values()) and
                    tensor_digest(policy.state_dict()) == fixed['checkpoint']['weight_hash'] and
                    sha(fixed['checkpoint']['path']) == fixed['checkpoint']['sha256'] and
                    all(p.grad is None for p in policy.parameters()), 'query feature coverage/weights/gradients differ')
            feature_pin = write_npz(artifacts / f'seed{seed}-query-features.npz', partial_query)
            query_features.append({'training_seed': seed, 'archive': feature_pin})
            full_metrics = parent.metrics(partial_query['predictions'], qtarget, np.ones(2598, bool))
            old_metrics = next(x for x in original_fold['seed_results'] if x['seed'] == seed)['arms']['B_fomaml']['curve']['500']['meta_query']
            check_pairs = {'mse': 'seven_dimensional_MSE', 'translation_mse': 'translation_MSE', 'rotation_mse': 'rotation_MSE', 'gripper_mse': 'gripper_MSE'}
            metric_delta = {old: full_metrics[new] - old_metrics[old] for old, new in check_pairs.items()}
            require(old_metrics['valid_steps'] == 2598 and all(abs(v) <= 1e-5 for v in metric_delta.values()), 'original query continuous metric parity differs')
            saved = saved_by_seed[seed]
            x = transform(partial_query['projection'], qplan['history_rows'], saved['mean'], saved['scale'])
            qm = {}
            for mode in MODES:
                arrays = saved['heads'][mode]
                prediction = torch.tanh(torch.nn.functional.linear(torch.from_numpy(x[mode]),
                    torch.from_numpy(arrays['weights'][-1]), torch.from_numpy(arrays['biases'][-1]))).detach().numpy().copy()
                counts['query_readout_forwards'] += 1
                counts['query_readout_rows'] += len(qsource)
                qm[mode] = parent.stratified(prediction, qtarget[qsource], qplan, np.ones(len(qsource), bool))
                qp = write_npz(artifacts / f'seed{seed}-{mode}-query-predictions.npz', {'source_rows': qsource, 'predictions': prediction})
                next(h for h in heads if h['training_seed'] == seed and h['mode'] == mode)['query_prediction_archive'] = qp
            seed_results.append({'training_seed': seed, 'query_metrics': qm,
                'support_metrics': {m: next(h for h in heads if h['training_seed'] == seed and h['mode'] == m)['curve_metrics']['500'] for m in MODES},
                'original_query_context': {'cpu_metrics_t_ge_1': full_metrics, 'historical_gpu_metrics_t_ge_1': old_metrics,
                                           'continuous_metric_deltas': metric_delta, 'gripper_sign_accuracy_delta': 1 - full_metrics['gripper_sign_error_rate'] - old_metrics['gripper_sign_accuracy']}})
            event('query_seed_complete', seed=seed, features=feature_pin, original_metric_deltas=metric_delta)
            print(json.dumps({'progress': 'query_seed_complete', 'seed': seed, 'main_batches': counts['query_encoder_main_completed'], 'support_updates': counts['optimizer_updates']}), flush=True)
            partial_query = None
        for key, value in counts.items():
            require(value == c['scope'][key], f'actual count differs: {key}')
        event('complete', actual_counts=counts)
        event_handle.close()
        event_handle = None
        result = {'status': 'complete_task1_b_support_query_history_readout', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path),
            'scope': c['scope'], 'analysis_definition': c['analysis'], 'input_provenance': c['inputs'], 'fixed_encoders': c['fixed_encoders'],
            'actual_counts': counts, 'readout_results': heads, 'numerical_bridges': bridges,
            'query_shared_archive': shared_query, 'query_features': query_features, 'query_canonical_windows': windows,
            'query_batch_digests': query_batch_hashes, 'seed_results': seed_results, 'primary_query_comparison': paired_cue(seed_results),
            'query_event_coverage': {'known_events_total': int(qplan['masks']['known_open_event'].sum()),
                                     'eligible_events': int(qplan['masks']['known_open_event'][qsource].sum()),
                                     'early_events_excluded': int(qplan['masks']['known_open_event'][qsteps < 5].sum())},
            'access_records': access_records, 'events': {'path': str(event_path), 'sha256': sha(event_path)},
            'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__, 'h5py': h5py.__version__,
                        'device': 'cpu', 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                        'deterministic': torch.are_deterministic_algorithms_enabled(), 'gpu_occupancy_before_run': gpu_occupancy,
                        'launch_environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                        'query_tensor_cache_bytes': cache_bytes, 'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        'entry_elapsed_seconds': time.monotonic() - started, 'argv': sys.argv},
            'interpretation': 'Previously exposed source meta-query endpoint readout diagnostic; no rollout, necessary-memory proof, SPT acceptance or qualification.'}
        write_json(output, result)
        return result
    except Exception as exc:
        partial = {}
        if artifacts.exists() and 'model' in active:
            partial['readout'] = write_npz(artifacts / 'failed-active-readout.npz', {'weight': active['model'].weight.detach().numpy(),
                'bias': active['model'].bias.detach().numpy(), 'completed_updates': np.array(active.get('update', 0))})
        if artifacts.exists() and partial_query is not None:
            partial['query_features'] = write_npz(artifacts / 'failed-active-query-features.npz', partial_query)
        if event_handle is not None:
            event_handle.close()
        write_json(failed, {'status': 'failed_task1_b_support_query_history_readout', 'formal_result': False,
            'stage': stage, 'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts,
            'completed_readouts': heads, 'completed_bridges': bridges, 'completed_query_features': query_features,
            'completed_query_seed_results': seed_results, 'partial': partial, 'access_records': access_records})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        result = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({'status': result['status'], 'output': str(args.output), 'actual_counts': result['actual_counts']}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'task_id': c['scope']['task_id'], 'new_updates_or_dataset_payloads': 0}))


if __name__ == '__main__':
    main()
