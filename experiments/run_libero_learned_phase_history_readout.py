"""Paired supervised readouts of cached features; never run original Policies."""
from __future__ import annotations

import argparse
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
from experiments import libero_phase_history_neighbors as prior
from experiments import analyze_libero_source_internal_gripper_sign_margin_temporal as base

require, sha, load, resolve, stats = base.require, base.sha, base.load, base.resolve, base.stats
TASKS, SEEDS, ARMS = base.TASKS, base.SEEDS, base.ARMS
MODES = ('current_repeat5', 'causal_history5')
CURVE = (0, 100, 250, 500)
DEFAULT_CONFIG = ROOT / 'configs/libero-learned-phase-history-readout-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-learned-phase-history-readout-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-learned-phase-history-readout-20261104-v1.json')
DEFAULT_ARTIFACTS = Path('/tmp/libero-learned-phase-history-readout-artifacts-20261104-v1')
CONFIG_SHA = '0b3a3d73a67e78223a822d8539d648b68ea8e98ac4b05306b80fbc9fb8196cba'


def array_hash(arrays):
    h = hashlib.sha256()
    for key, value in arrays.items():
        value = np.ascontiguousarray(value)
        h.update(key.encode())
        h.update(str((value.shape, value.dtype.str)).encode())
        h.update(value.tobytes())
    return h.hexdigest()


def fold_inputs(features, plan, probe_demos):
    require(features.dtype == np.float32 and features.shape == (len(plan['steps']), 256) and
            np.isfinite(features).all(), 'cached feature interface differs')
    source = plan['eligible']
    probe = np.isin(plan['demo_ids'][source], probe_demos)
    fit = ~probe
    require(fit.any() and probe.any() and len(set(probe_demos)) == len(probe_demos), 'invalid whole-demo split')
    # Only fit demos enter the scaler, including all its history positions.
    fit_features = features[source[fit]].astype(np.float64)
    mean, scale = fit_features.mean(0), fit_features.std(0)
    scale[scale <= 1e-6] = 1.
    history = plan['history_rows']
    require(np.all(plan['demo_ids'][history] == plan['demo_ids'][source, None]) and
            np.array_equal(plan['steps'][history], plan['steps'][source, None] - np.arange(5)), 'noncausal history')
    normalized = (features[history].astype(np.float64) - mean) / scale
    values = {'current_repeat5': np.repeat(normalized[:, :1], 5, axis=1), 'causal_history5': normalized}
    values = {k: np.ascontiguousarray((v.reshape(len(source), 1280) / np.sqrt(5)).astype(np.float32)) for k, v in values.items()}
    require(all(np.isfinite(v).all() for v in values.values()), 'nonfinite readout inputs')
    return values, np.flatnonzero(fit), np.flatnonzero(probe), mean, scale


def metrics(prediction, targets, mask):
    require(prediction.shape == targets.shape and prediction.shape[1] == 7 and np.isfinite(prediction).all(), 'invalid saved readout prediction')
    n = int(mask.sum())
    residual = prediction[mask].astype(np.float64) - targets[mask].astype(np.float64)
    truth, guess = np.sign(targets[mask, 6]), np.sign(prediction[mask, 6])
    return {'n': n, 'translation_MSE': float(np.mean(residual[:, :3] ** 2)) if n else None,
            'rotation_MSE': float(np.mean(residual[:, 3:6] ** 2)) if n else None,
            'gripper_MSE': float(np.mean(residual[:, 6] ** 2)) if n else None,
            'seven_dimensional_MSE': float(np.mean(residual ** 2)) if n else None,
            'gripper_sign_error_n': int((truth != guess).sum()),
            'gripper_sign_error_rate': float(np.mean(truth != guess)) if n else None,
            'gripper_zero_n': int((guess == 0).sum())}


def stratified(prediction, targets, plan, selected):
    source = plan['eligible']
    return {name: metrics(prediction, targets, selected & mask) for name, mask in
            {'all': np.ones(len(source), bool), **{k: v[source] for k, v in plan['masks'].items()}}.items()}


def train_readout(x, targets, batches, curve, optimizer_spec, counts, active):
    import torch
    require(x.dtype == targets.dtype == np.float32 and x.ndim == targets.ndim == 2 and
            len(x) == len(targets) and targets.shape[1] == 7 and batches.ndim == 2, 'readout training interface differs')
    require(tuple(sorted(set(curve))) == tuple(curve) and curve[0] == 0 and curve[-1] == len(batches), 'fixed curve budget differs')
    model = torch.nn.Linear(x.shape[1], 7, device='cpu', dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_spec['learning_rate'], betas=tuple(optimizer_spec['betas']),
                                 eps=optimizer_spec['eps'], weight_decay=optimizer_spec['weight_decay'])
    xt, yt = torch.from_numpy(x), torch.from_numpy(targets)
    active.update(model=model, optimizer=optimizer, update=0)
    weights, biases, predictions, losses, norms = [], [], [], [], []
    def evaluate():
        model.eval()
        with torch.inference_mode():
            pred = torch.tanh(model(xt))
        counts['metric_forward_batches'] += 1
        counts['metric_valid_rows'] += len(x)
        require(torch.isfinite(pred).all().item(), 'nonfinite metric readout')
        weights.append(model.weight.detach().numpy().copy())
        biases.append(model.bias.detach().numpy().copy())
        predictions.append(pred.numpy().copy())
    evaluate()
    initial_hash = array_hash({'weight': weights[0], 'bias': biases[0]})
    for step, batch in enumerate(batches, 1):
        require(np.issubdtype(batch.dtype, np.integer) and (batch >= 0).all() and (batch < len(x)).all(), 'invalid sampled row')
        index = torch.from_numpy(batch.astype(np.int64, copy=False))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = torch.tanh(model(xt.index_select(0, index)))
        counts['train_forward_batches'] += 1
        counts['train_valid_rows'] += len(batch)
        loss = (prediction - yt.index_select(0, index)).square().mean()
        require(torch.isfinite(loss).item(), 'nonfinite training loss')
        loss.backward()
        counts['backward_calls'] += 1
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_spec['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        counts['optimizer_updates'] += 1
        active['update'] = step
        losses.append(float(loss.item()))
        norms.append(float(norm.item()))
        if step in curve:
            evaluate()
    require(all(np.isfinite(v).all() for v in weights + biases), 'nonfinite readout weights')
    states = [optimizer.state[p] for p in (model.weight, model.bias)]
    require(all(int(s['step'].item()) == len(batches) for s in states), 'actual Adam update count differs')
    buffers = {f'adam_{name}_{field}': state[field].detach().numpy().copy() for name, state in zip(('weight', 'bias'), states) for field in ('exp_avg', 'exp_avg_sq')}
    arrays = {'weights': np.stack(weights), 'biases': np.stack(biases), 'predictions': np.stack(predictions),
              'updates': np.array(curve, np.int32), 'step_losses': np.array(losses), 'pre_clip_gradient_norms': np.array(norms), **buffers}
    parameter_group = optimizer.param_groups[0]
    actual_optimizer = {k: parameter_group[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused')}
    return arrays, {'initial_weight_hash': initial_hash, 'final_weight_hash': array_hash({'weight': weights[-1], 'bias': biases[-1]}),
                    'optimizer_updates': len(batches), 'parameters': sum(p.numel() for p in model.parameters()),
                    'clipped_update_n': int((np.array(norms) > optimizer_spec['gradient_clip_norm']).sum()),
                    'actual_optimizer': actual_optimizer, 'gradient_clip_norm': optimizer_spec['gradient_clip_norm']}


def primary_flags(endpoints):
    output = []
    for task in TASKS:
        for arm in ARMS:
            rows = sorted([r for r in endpoints if r['task_id'] == task and r['feature_arm'] == arm], key=lambda x: x['training_seed'])
            require([r['training_seed'] for r in rows] == list(SEEDS), 'retain all three feature seeds')
            changes = []
            for row in rows:
                a, b = [row['oof_metrics'][mode] for mode in MODES]
                values = {'open_event_sign_error': (a['known_open_event']['gripper_sign_error_rate'], b['known_open_event']['gripper_sign_error_rate']),
                          'known_open_segment_sign_error': (a['known_open_segment']['gripper_sign_error_rate'], b['known_open_segment']['gripper_sign_error_rate']),
                          'overall_7D_MSE': (a['all']['seven_dimensional_MSE'], b['all']['seven_dimensional_MSE']),
                          'overall_translation_MSE': (a['all']['translation_MSE'], b['all']['translation_MSE'])}
                changes.append({'training_seed': row['training_seed'], 'deltas': {k: v[1] - v[0] if all(x is not None for x in v) else None for k, v in values.items()}})
            flags = {}
            for key in changes[0]['deltas']:
                delta = [r['deltas'][key] for r in changes]
                flags[key] = bool(all(v < 0 if key == 'open_event_sign_error' else v <= 0 for v in delta)) if all(v is not None for v in delta) else None
            output.append({'task_id': task, 'feature_arm': arm, 'per_seed': changes, 'flags': flags,
                           'joint_history_cue': all(flags.values()) if all(v is not None for v in flags.values()) else None,
                           'automatic_policy_or_spt_promotion': False})
    return output


def preflight(config_path, binding_path):
    c = load(config_path)
    require(c['formal_result'] is False, 'readout run is diagnostic')
    expected = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'feature_arms': list(ARMS), 'feature_endpoints': 18,
        'original_policy_updates': 500, 'closed_loop_default_stabilization_steps': 5, 'source_role': 'meta_support',
        'demonstration_indices': list(range(30)), 'common_first_action_step': 5, 'history_steps': 5, 'readout_modes': list(MODES),
        'folds': 3, 'readout_models': 108, 'updates_per_readout': 500, 'optimizer_updates': 54000, 'backward_calls': 54000,
        'train_forward_batches': 54000, 'metric_forward_batches': 432, 'train_valid_rows': 3456000,
        'metric_valid_rows': 1450224, 'oof_rows_per_mode': 60426, 'event_pairs': 585,
        'new_full_policy_forward_batches': 0, 'new_original_policy_optimizer_updates': 0, 'new_environment_calls': 0,
        'new_HDF5_RGB_initial_private_checkpoint_payload_reads': 0, 'new_nearest_neighbor_searches': 0,
        'source_query_accessed': False, 'sealed_roles_accessed': False, 'checkpoint_selected': False,
        'original_policy_modified': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'scientific config hash differs')
    for key, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {key}')
    binding = load(binding_path)
    require(binding['config_sha256'] == CONFIG_SHA and binding['formal_result'] is False and
            set(binding['files']) == {'runner', 'tests', 'protocol'}, 'execution binding differs')
    for pin in binding['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound implementation differs')
    require(resolve(binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner binding path differs')
    old_binding = load(c['inputs']['phase_history_binding']['path'])
    for pin in old_binding['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'historical representation code differs')
    return c, binding


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failed = output.with_name(output.stem + '-failed.json')
    require(not output.exists() and not failed.exists() and not artifacts.exists(), 'outputs/artifacts already exist; preserve them')
    started = time.monotonic()
    counts = {k: 0 for k in ('readout_models', 'optimizer_updates', 'backward_calls', 'train_forward_batches',
        'metric_forward_batches', 'train_valid_rows', 'metric_valid_rows', 'oof_rows_per_mode', 'event_pairs',
        'new_full_policy_forward_batches', 'new_original_policy_optimizer_updates', 'new_environment_calls',
        'new_HDF5_RGB_initial_private_checkpoint_payload_reads', 'new_nearest_neighbor_searches')}
    heads, endpoints, used_features, active = [], [], [], {}
    event_handle = None
    stage = 'metadata_preflight'
    try:
        c, binding = preflight(config_path, binding_path)
        parent = load(c['inputs']['phase_history_result']['path'])
        audit = load(c['inputs']['phase_history_audit']['path'])
        require(parent['status'] == 'complete_phase_history_observability_diagnostic' and parent['formal_result'] is False and
                audit['status'] == 'complete_independent_phase_history_audit' and
                audit['result']['sha256'] == sha(c['inputs']['phase_history_result']['path']), 'parent completed status/hash differs')
        require(parent['config_sha256'] == sha(c['inputs']['phase_history_config']['path']) and
                parent['binding_sha256'] == sha(c['inputs']['phase_history_binding']['path']), 'parent provenance differs')
        require(len(parent['primary_model_results']) == 18 and
                {(r['task_id'], r['training_seed'], r['arm']) for r in parent['primary_model_results']} ==
                {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS}, 'all fixed feature endpoints required')
        neighbor = load(c['inputs']['neighbor_result']['path'])
        neighbor_audit = load(c['inputs']['neighbor_audit']['path'])
        require(neighbor['status'] == 'complete_readonly_transition_neighbor_analysis' and
                neighbor_audit['status'] == 'complete_independent_transition_neighbor_audit' and
                neighbor_audit['result']['sha256'] == sha(c['inputs']['neighbor_result']['path']), 'neighbor parent/audit differs')
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1' for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'CPU launch environment differs')
        # Read-only occupancy check is recorded even though no GPU is requested.
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                             check=True, capture_output=True, text=True).stdout
        import torch
        require({'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__} == c['execution']['runtime'] and
                torch.cuda.device_count() == 0, 'runtime/visible-device mismatch')
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
        event('runtime_ready', gpu_occupancy=gpu, device='cpu')
        plans = load(c['inputs']['full_open_result']['path'])['task_segment_plans']
        for task in TASKS:
            with np.load(resolve(c['inputs'][f'task{task}_archive']['path']), allow_pickle=False) as z:
                targets, ids, steps, old_predictions, model_ids = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps', 'predictions', 'model_ids')]
            require(set(ids.tolist()) == set(range(30)), 'source-support demo roles differ')
            plan = prior.frame_plan(targets, ids, steps, next(p for p in plans if p['task_id'] == task))
            source = plan['eligible']
            y = np.ascontiguousarray(targets[source])
            require(np.isin(np.sign(y[:, 6]), (-1, 1)).all(), 'unexpected neutral expert label')
            for seed in SEEDS:
                for arm in ARMS:
                    context = {'task_id': task, 'training_seed': seed, 'feature_arm': arm}
                    original = next(r for r in parent['primary_model_results'] if (r['task_id'], r['training_seed'], r['arm']) == (task, seed, arm))
                    pin = original['features']
                    require(sha(pin['path']) == pin['sha256'], 'fixed feature hash differs')
                    used_features.append(pin)
                    with np.load(pin['path'], allow_pickle=False) as z:
                        features, original_predictions = z['projection'], z['predictions']
                    feature_hash_before = array_hash({'features': features})
                    mi = SEEDS.index(seed) * 3 + ARMS.index(arm)
                    require(str(model_ids[mi]) == f'{seed}:{arm}' and np.array_equal(original_predictions, old_predictions[mi]), 'original prediction identity differs')
                    oof = {mode: np.full(y.shape, np.nan, np.float32) for mode in MODES}
                    covered = {mode: np.zeros(len(source), bool) for mode in MODES}
                    for fold, probe_demos in enumerate(c['analysis']['fold_demo_indices']):
                        x, fit_positions, probe_positions, mean, scale = fold_inputs(features, plan, probe_demos)
                        require(set(ids[source[probe_positions]].tolist()) == set(probe_demos) and
                                not np.isin(ids[source[fit_positions]], probe_demos).any(), 'readout fit/probe overlap')
                        readout_seed = seed + 100000 * fold
                        rng = np.random.default_rng(readout_seed)
                        batches = fit_positions[rng.integers(0, len(fit_positions), size=(500, 64))]
                        batch_hash = array_hash({'batch_source_rows': source[batches]})
                        pair_initial_hash = None
                        for mode in MODES:
                            stage = f't{task}-s{seed}-{arm}-f{fold}-{mode}'
                            active.clear()
                            active.update(context={**context, 'fold': fold, 'mode': mode})
                            torch.manual_seed(readout_seed)
                            arrays, actual = train_readout(x[mode], y, batches, CURVE, c['analysis']['optimizer'], counts, active)
                            require(actual['parameters'] == 8967 and actual['optimizer_updates'] == 500 and
                                    np.array_equal(arrays['predictions'][0], np.zeros_like(y)), 'readout architecture/init/budget differs')
                            if pair_initial_hash is None:
                                pair_initial_hash = actual['initial_weight_hash']
                            require(actual['initial_weight_hash'] == pair_initial_hash, 'paired initial weights differ')
                            fit_mask = np.zeros(len(source), bool)
                            fit_mask[fit_positions] = True
                            probe_mask = ~fit_mask
                            curve_metrics = {str(u): {'fit': stratified(pred, y, plan, fit_mask),
                                                       'probe': stratified(pred, y, plan, probe_mask)} for u, pred in zip(CURVE, arrays['predictions'])}
                            path = artifacts / f'{stage}.npz'
                            arrays.update(source_rows=source, fit_source_rows=source[fit_positions], probe_source_rows=source[probe_positions],
                                          batch_source_rows=source[batches], scaler_mean=mean, scaler_scale=scale)
                            with path.open('xb') as handle:
                                np.savez_compressed(handle, **arrays)
                            file_pin = {'path': str(path), 'sha256': sha(path)}
                            heads.append({**context, 'fold': fold, 'mode': mode, 'readout_seed': readout_seed,
                                          'fit_demonstrations': sorted(set(ids[source[fit_positions]].tolist())), 'probe_demonstrations': probe_demos,
                                          'batch_source_rows_hash': batch_hash, 'actual': actual, 'curve_metrics': curve_metrics, 'archive': file_pin})
                            require(not covered[mode][probe_positions].any(), 'OOF rows scored twice')
                            oof[mode][probe_positions] = arrays['predictions'][-1, probe_positions]
                            covered[mode][probe_positions] = True
                            counts['readout_models'] += 1
                            event('readout_complete', **context, fold=fold, mode=mode, optimizer_updates=counts['optimizer_updates'], archive=file_pin)
                        print(json.dumps({'progress': 'paired_fold_complete', **context, 'fold': fold, 'models': counts['readout_models'], 'updates': counts['optimizer_updates']}), flush=True)
                    require(all(v.all() for v in covered.values()) and all(np.isfinite(v).all() for v in oof.values()), 'missing OOF rows')
                    require(array_hash({'features': features}) == feature_hash_before, 'fixed encoder feature array modified')
                    endpoints.append({**context, 'oof_metrics': {mode: stratified(pred, y, plan, np.ones(len(source), bool)) for mode, pred in oof.items()},
                                      'original_policy_context_metrics': stratified(original_predictions[source], y, plan, np.ones(len(source), bool)),
                                      'feature_hash_unchanged': True})
                    counts['oof_rows_per_mode'] += len(source)
                    counts['event_pairs'] += int(plan['masks']['known_open_event'][source].sum())
        for key, value in counts.items():
            require(value == c['scope'][key], f'actual budget differs: {key}')
        event('complete', actual_counts=counts)
        event_handle.close()
        event_handle = None
        result = {'status': 'complete_learned_phase_history_readout_diagnostic', 'formal_result': False,
                  'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path),
                  'scope': c['scope'], 'analysis_definition': c['analysis'], 'input_provenance': c['inputs'], 'used_features': used_features,
                  'actual_counts': counts, 'readout_results': heads, 'primary_endpoint_results': endpoints, 'primary_flags': primary_flags(endpoints),
                  'events': {'path': str(event_path), 'sha256': sha(event_path)},
                  'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu',
                              'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                              'deterministic': torch.are_deterministic_algorithms_enabled(), 'gpu_occupancy_before_training': gpu,
                              'launch_environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                              'entry_elapsed_seconds': time.monotonic() - started, 'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                              'argv': sys.argv},
                  'interpretation': 'Cross-fitted readout learning on already-exposed support encoders; no original Policy training, rollout, unseen-transfer or necessary-memory result.'}
        with output.open('x') as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
        return result
    except Exception as exc:
        partial = None
        if 'model' in active and artifacts.exists():
            partial = artifacts / 'failed-active-readout.npz'
            with partial.open('xb') as handle:
                np.savez_compressed(handle, weight=active['model'].weight.detach().numpy(), bias=active['model'].bias.detach().numpy(),
                                    completed_updates=np.array(active.get('update', 0)))
        if event_handle is not None:
            event_handle.close()
        with failed.open('x') as handle:
            json.dump({'status': 'failed_learned_phase_history_readout_diagnostic', 'formal_result': False,
                       'stage': stage, 'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts,
                       'completed_readouts': heads, 'completed_endpoints': endpoints, 'partial': str(partial) if partial else None,
                       'active_context': active.get('context'), 'active_completed_updates': active.get('update', 0)}, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
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
        print(json.dumps({'status': 'metadata_preflight_passed', 'readout_models': c['scope']['readout_models'], 'research_updates': 0}))


if __name__ == '__main__':
    main()
