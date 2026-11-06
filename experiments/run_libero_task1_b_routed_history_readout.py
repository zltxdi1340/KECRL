"""Paired fixed-feature heads: current actions versus history motion/current grip."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from experiments import run_libero_task1_b_support_query_history_readout as cached

parent = cached.parent
require, sha, load, resolve = cached.require, cached.sha, cached.load, cached.resolve
SEEDS, CURVE = cached.SEEDS, parent.CURVE
ARMS = ('current_all7', 'history_motion_current_gripper')
PREFIX = 'libero-task1-b-routed-history-readout'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_ARTIFACTS = Path(f'/tmp/{PREFIX}-artifacts-20261104-v1')
CONFIG_SHA = 'd8f601227ae3052fee78ecfbcff2106477e74d2893e4f0b4fcd1fbfaedd99b81'


def prediction(model, current, history, arm):
    import torch
    require(arm in ARMS, 'unknown input route')
    if arm == ARMS[0]:
        logits = model(current)
    else:
        logits = torch.cat((model(history)[:, :6], model(current)[:, 6:]), dim=1)
    return torch.tanh(logits)


def train(current, history, targets, batches, arm, optimizer_spec, counts, active):
    import torch
    require(current.shape == history.shape == (len(targets), 1280) and targets.shape[1] == 7 and
            all(v.dtype == np.float32 and np.isfinite(v).all() for v in (current, history, targets)), 'training interface differs')
    require(batches.shape == (500, 64) and np.issubdtype(batches.dtype, np.integer) and
            (batches >= 0).all() and (batches < len(targets)).all(), 'sampling budget differs')
    model = torch.nn.Linear(1280, 7, device='cpu', dtype=torch.float32)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_spec['learning_rate'], betas=tuple(optimizer_spec['betas']),
                                 eps=optimizer_spec['eps'], weight_decay=optimizer_spec['weight_decay'])
    ct, ht, yt = [torch.from_numpy(v) for v in (current, history, targets)]
    active.update(model=model, optimizer=optimizer, update=0)
    weights, biases, predictions, losses, norms = [], [], [], [], []
    primitive = 1 if arm == ARMS[0] else 2
    start = time.perf_counter()

    def evaluate():
        model.eval()
        with torch.inference_mode():
            pred = prediction(model, ct, ht, arm)
        counts['metric_forward_batches'] += 1
        counts['metric_linear_calls'] += primitive
        counts['metric_valid_rows'] += len(current)
        require(torch.isfinite(pred).all().item(), 'nonfinite metric output')
        weights.append(model.weight.detach().numpy().copy())
        biases.append(model.bias.detach().numpy().copy())
        predictions.append(pred.numpy().copy())

    evaluate()
    initial_hash = parent.array_hash({'weight': weights[0], 'bias': biases[0]})
    for step, batch in enumerate(batches, 1):
        index = torch.from_numpy(batch.astype(np.int64, copy=False))
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = prediction(model, ct.index_select(0, index), ht.index_select(0, index), arm)
        counts['train_forward_batches'] += 1
        counts['train_linear_calls'] += primitive
        counts['train_valid_rows'] += len(batch)
        loss = (pred - yt.index_select(0, index)).square().mean()
        require(torch.isfinite(loss).item(), 'nonfinite training loss')
        loss.backward()
        counts['backward_calls'] += 1
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), optimizer_spec['gradient_clip_norm'], error_if_nonfinite=True)
        optimizer.step()
        counts['optimizer_updates'] += 1
        active['update'] = step
        losses.append(float(loss.item()))
        norms.append(float(norm.item()))
        if step in CURVE:
            evaluate()
    states = [optimizer.state[p] for p in (model.weight, model.bias)]
    require(all(int(s['step'].item()) == 500 for s in states), 'actual Adam steps differ')
    require(all(np.isfinite(v).all() for v in weights + biases), 'nonfinite weights')
    buffers = {f'adam_{name}_{field}': state[field].detach().numpy().copy()
               for name, state in zip(('weight', 'bias'), states) for field in ('exp_avg', 'exp_avg_sq')}
    arrays = {'weights': np.stack(weights), 'biases': np.stack(biases), 'predictions': np.stack(predictions),
              'updates': np.array(CURVE, np.int32), 'step_losses': np.array(losses),
              'pre_clip_gradient_norms': np.array(norms), **buffers}
    pg = optimizer.param_groups[0]
    actual = {'initial_weight_hash': initial_hash, 'final_weight_hash': parent.array_hash({'weight': weights[-1], 'bias': biases[-1]}),
        'optimizer_updates': 500, 'parameters': sum(p.numel() for p in model.parameters()),
        'clipped_update_n': int((np.array(norms) > optimizer_spec['gradient_clip_norm']).sum()),
        'actual_optimizer': {k: pg[k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad', 'maximize', 'foreach', 'capturable', 'differentiable', 'fused')},
        'gradient_clip_norm': optimizer_spec['gradient_clip_norm'], 'elapsed_seconds': time.perf_counter() - start,
        'linear_calls_per_forward': primitive}
    return arrays, actual


def comparison(rows, metric_key):
    require([r['training_seed'] for r in rows] == list(SEEDS), 'retain all three seeds')
    per_seed = []
    for row in rows:
        a, b = [row[metric_key][arm] for arm in ARMS]
        pairs = {'event_sign_errors': (a['known_open_event']['gripper_sign_error_n'], b['known_open_event']['gripper_sign_error_n']),
            'full_open_sign_errors': (a['known_open_segment']['gripper_sign_error_n'], b['known_open_segment']['gripper_sign_error_n']),
            'translation_MSE': (a['all']['translation_MSE'], b['all']['translation_MSE']),
            'seven_dimensional_MSE': (a['all']['seven_dimensional_MSE'], b['all']['seven_dimensional_MSE'])}
        require(a['known_open_event']['n'] == b['known_open_event']['n'] > 0 and
                a['known_open_segment']['n'] == b['known_open_segment']['n'] > 0, 'empty or mismatched gate denominator')
        per_seed.append({'training_seed': row['training_seed'], 'deltas': {k: y - x for k, (x, y) in pairs.items()}})
    flags = {k: bool(all(r['deltas'][k] < 0 if k.endswith('MSE') else r['deltas'][k] <= 0 for r in per_seed))
             for k in per_seed[0]['deltas']}
    return {'per_seed': per_seed, 'flags': flags, 'joint_cue': all(flags.values()),
            'old_strict_history_gate_reinterpreted': False, 'automatic_rollout_or_spt_promotion': False}


def assert_query_gate(oof_comparison, heads, counts):
    require(oof_comparison['joint_cue'] is True and len(heads) == 24 and
            {(h['training_seed'], h['arm']) for h in heads if h['stage'] == 'full_support'} == {(s, a) for s in SEEDS for a in ARMS} and
            sum(h['stage'] == 'full_support' for h in heads) == 6 and all(h['actual']['optimizer_updates'] == 500 for h in heads),
            'query gate: support cue and all fixed fits required')
    require(counts['readout_models'] == 24 and counts['optimizer_updates'] == counts['backward_calls'] == 12000,
            'query gate: actual training budget differs')


def assert_baseline(arrays, pin):
    with np.load(resolve(pin['path']), allow_pickle=False) as old:
        for key in old.files:
            require(key in arrays and np.array_equal(arrays[key], old[key]), f'baseline reproduction differs: {key}')
    return {'archive': pin, 'all_saved_arrays_bitwise_equal': True}


def preflight(config_path, binding_path):
    c = load(config_path)
    expected = {'task_id': 1, 'feature_arm': 'B_fomaml', 'training_seeds': list(SEEDS),
        'original_policy_updates': 500, 'closed_loop_default_stabilization_steps': 5, 'arms': list(ARMS),
        'support_demos': list(range(30)), 'query_demos': list(range(30, 50)), 'history_steps': 5,
        'common_first_action_step': 5, 'support_target_rows': 4059, 'support_common_rows': 3939,
        'query_target_rows': 2598, 'query_common_rows': 2518, 'folds': 3, 'updates_per_readout': 500,
        'parameters_per_readout': 8967, 'support_stage_models': 18, 'support_stage_optimizer_updates': 9000,
        'full_stage_models_if_support_pass': 6, 'full_stage_optimizer_updates_if_support_pass': 3000,
        'maximum_readout_models': 24, 'maximum_optimizer_updates': 12000, 'maximum_backward_calls': 12000,
        'maximum_train_forward_batches': 12000, 'maximum_train_valid_rows': 768000, 'maximum_train_linear_calls': 18000,
        'maximum_metric_forward_batches': 96, 'maximum_metric_valid_rows': 378144, 'maximum_metric_linear_calls': 144,
        'query_readout_batches_if_support_pass': 6, 'query_readout_rows_if_support_pass': 15108, 'query_linear_calls_if_support_pass': 9,
        'new_encoder_calls': 0, 'new_HDF5_RGB_initial_private_checkpoint_payload_reads': 0, 'new_environment_calls': 0,
        'original_policy_updates_new': 0, 'sealed_roles_accessed': False, 'checkpoint_selected': False,
        'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    require(c['formal_result'] is False, 'diagnostic only')
    for key, value in expected.items():
        require(c['scope'].get(key) == value and type(c['scope'].get(key)) is type(value), f'frozen scope differs: {key}')
    require(sha(config_path) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input pin differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
            set(b['files']) == {'runner', 'tests', 'protocol'}, 'binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'execution code differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner binding differs')
    for key in ('parent_binding', 'crossed_binding', 'learned_binding'):
        for pin in load(c['inputs'][key]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'historical bound code differs')
    return c, b


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failure = output.with_name(output.stem + '-failed.json')
    require(not any(p.exists() for p in (output, failure, artifacts)), 'outputs already exist')
    counts = {k: 0 for k in ('readout_models', 'optimizer_updates', 'backward_calls', 'train_forward_batches',
        'train_valid_rows', 'train_linear_calls', 'metric_forward_batches', 'metric_valid_rows', 'metric_linear_calls',
        'query_readout_batches', 'query_readout_rows', 'query_linear_calls', 'support_feature_archives_loaded',
        'query_feature_archives_loaded', 'query_target_archives_loaded', 'new_encoder_calls',
        'new_HDF5_RGB_initial_private_checkpoint_payload_reads', 'new_environment_calls', 'original_policy_updates_new')}
    heads, oof_rows, full_rows, query_rows, active = [], [], [], [], {}
    start, stage, event_handle = time.perf_counter(), 'preflight', None
    try:
        c, _ = preflight(config_path, binding_path)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and
                all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'launch environment differs')
        occupancy = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader'],
                                   capture_output=True, text=True, check=True).stdout
        import torch
        require(sys.version.split()[0] == c['execution']['python'] and np.__version__ == c['execution']['numpy'] and
                torch.__version__ == c['execution']['torch'] and torch.cuda.device_count() == 0, 'runtime/device differs')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        artifacts.mkdir(parents=True, exist_ok=False)
        event_handle = (artifacts / 'events.jsonl').open('x')

        def record(kind, **fields):
            event_handle.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}, allow_nan=False) + '\n')
            event_handle.flush()

        record('runtime_ready', gpu_occupancy_before_training=occupancy, device='cpu')
        with np.load(resolve(c['inputs']['support_targets']['path']), allow_pickle=False) as z:
            targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
        plan = cached.make_plan(targets, ids, steps)
        source = plan['eligible']
        require(len(targets) == 4059 and len(source) == 3939 and set(ids.tolist()) == set(range(30)), 'support role/count differs')
        y, selected = np.ascontiguousarray(targets[source]), np.ones(len(source), bool)
        support, fixed_full = {}, {}

        def fit_pair(seed, fold, inputs, fit, probe, mean, scale, fit_stage):
            readout_seed = seed if fold is None else seed + 100000 * fold
            rng = np.random.default_rng(readout_seed)
            batches = fit[rng.integers(0, len(fit), size=(500, 64))]
            initial_hash = None
            paired = {}
            for arm in ARMS:
                active.clear()
                active['context'] = {'training_seed': seed, 'fold': fold, 'stage': fit_stage, 'arm': arm}
                torch.manual_seed(readout_seed)
                arrays, actual = train(inputs['current_repeat5'], inputs['causal_history5'], y, batches, arm,
                                       c['analysis']['optimizer'], counts, active)
                require(actual['parameters'] == 8967 and actual['optimizer_updates'] == 500 and not np.count_nonzero(arrays['predictions'][0]),
                        'paired architecture/init/budget differs')
                if initial_hash is None:
                    initial_hash = actual['initial_weight_hash']
                require(actual['initial_weight_hash'] == initial_hash, 'paired initialization differs')
                arrays.update(source_rows=source, batch_source_rows=source[batches], scaler_mean=mean, scaler_scale=scale)
                if fold is not None:
                    arrays.update(fit_source_rows=source[fit], probe_source_rows=source[probe])
                reproduced = None
                if arm == ARMS[0]:
                    key = f'{seed}_full_baseline' if fold is None else f'{seed}_fold{fold}_baseline'
                    reproduced = assert_baseline(arrays, c['inputs'][key])
                fit_mask = np.zeros(len(source), bool)
                fit_mask[fit] = True
                curve = {str(u): {'fit': parent.stratified(pred, y, plan, fit_mask),
                                  'probe': parent.stratified(pred, y, plan, ~fit_mask)}
                         for u, pred in zip(CURVE, arrays['predictions'])}
                name = f'seed{seed}-{fit_stage}-fold{fold}-{arm}'
                pin = cached.write_npz(artifacts / f'{name}.npz', arrays)
                heads.append({'training_seed': seed, 'fold': fold, 'stage': fit_stage, 'arm': arm,
                    'actual': actual, 'batch_source_rows_hash': parent.array_hash({'batch_source_rows': source[batches]}),
                    'curve_metrics': curve, 'archive': pin, 'baseline_reproduction': reproduced,
                    'motion_input_hash': parent.array_hash({'input': inputs['current_repeat5'] if arm == ARMS[0] else inputs['causal_history5']}),
                    'gripper_input_hash': parent.array_hash({'input': inputs['current_repeat5']})})
                counts['readout_models'] += 1
                paired[arm] = arrays
                record('readout_complete', training_seed=seed, fold=fold, stage=fit_stage, arm=arm,
                       optimizer_updates=counts['optimizer_updates'], archive=pin)
            print(json.dumps({'progress': 'paired_fit_complete', 'seed': seed, 'fold': fold,
                              'stage': fit_stage, 'updates': counts['optimizer_updates']}), flush=True)
            return paired

        for seed in SEEDS:
            stage = f'support_oof_seed{seed}'
            with np.load(resolve(c['inputs'][f'{seed}_support_features']['path']), allow_pickle=False) as z:
                features = z['projection']
            counts['support_feature_archives_loaded'] += 1
            support[seed] = features
            oof = {arm: np.full(y.shape, np.nan, np.float32) for arm in ARMS}
            covered = np.zeros(len(source), bool)
            for fold, demos in enumerate(c['analysis']['fold_demo_indices']):
                inputs, fit, probe, mean, scale = parent.fold_inputs(features, plan, demos)
                pair = fit_pair(seed, fold, inputs, fit, probe, mean, scale, 'support_oof')
                require(not covered[probe].any(), 'duplicate OOF rows')
                for arm in ARMS:
                    oof[arm][probe] = pair[arm]['predictions'][-1, probe]
                covered[probe] = True
            require(covered.all(), 'missing OOF rows')
            oof_rows.append({'training_seed': seed, 'oof_metrics': {arm: parent.stratified(v, y, plan, selected) for arm, v in oof.items()},
                'archive': cached.write_npz(artifacts / f'seed{seed}-oof-predictions.npz', {'source_rows': source, **oof})})
        oof_cue = comparison(oof_rows, 'oof_metrics')
        record('support_gate_decision', comparison=oof_cue)
        query_shared = None
        if oof_cue['joint_cue']:
            for seed in SEEDS:
                stage = f'full_support_seed{seed}'
                inputs, mean, scale = cached.full_support_inputs(support[seed], plan)
                pair = fit_pair(seed, None, inputs, np.arange(len(source)), np.array([], np.int64), mean, scale, 'full_support')
                fixed_full[seed] = {'heads': pair, 'mean': mean, 'scale': scale}
                full_rows.append({'training_seed': seed, 'support_metrics': {arm: parent.stratified(v['predictions'][-1], y, plan, selected) for arm, v in pair.items()}})
            assert_query_gate(oof_cue, heads, counts)
            stage = 'cached_query_open'
            record('all_support_fits_complete_before_query_payload', optimizer_updates=counts['optimizer_updates'])
            with np.load(resolve(c['inputs']['query_targets']['path']), allow_pickle=False) as z:
                qt, qi, qs = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
            counts['query_target_archives_loaded'] += 1
            qp = cached.make_plan(qt, qi, qs)
            qr = qp['eligible']
            require(len(qt) == 2598 and len(qr) == 2518 and set(qi.tolist()) == set(range(30, 50)), 'query role/count differs')
            query_shared = c['inputs']['query_targets']
            for seed in SEEDS:
                stage = f'query_seed{seed}'
                assert_query_gate(oof_cue, heads, counts)
                with np.load(resolve(c['inputs'][f'{seed}_query_features']['path']), allow_pickle=False) as z:
                    features = z['projection']
                counts['query_feature_archives_loaded'] += 1
                full_fit = fixed_full[seed]
                inputs = cached.transform(features, qp['history_rows'], full_fit['mean'], full_fit['scale'])
                metrics, archives, reproductions = {}, {}, {}
                for arm in ARMS:
                    arrays = full_fit['heads'][arm]
                    model = torch.nn.Linear(1280, 7, device='cpu', dtype=torch.float32)
                    with torch.no_grad():
                        model.weight.copy_(torch.from_numpy(arrays['weights'][-1]))
                        model.bias.copy_(torch.from_numpy(arrays['biases'][-1]))
                    model.eval()
                    with torch.inference_mode():
                        pred = prediction(model, torch.from_numpy(inputs['current_repeat5']), torch.from_numpy(inputs['causal_history5']), arm).numpy().copy()
                    counts['query_readout_batches'] += 1
                    counts['query_readout_rows'] += len(qr)
                    counts['query_linear_calls'] += 1 if arm == ARMS[0] else 2
                    require(np.isfinite(pred).all(), 'nonfinite query output')
                    if arm == ARMS[0]:
                        with np.load(resolve(c['inputs'][f'{seed}_query_baseline']['path']), allow_pickle=False) as z:
                            require(np.array_equal(z['source_rows'], qr), 'old query row identity differs')
                            error = float(np.max(np.abs(pred - z['predictions'])))
                            signs_equal = bool(np.array_equal(np.sign(pred[:, 6]), np.sign(z['predictions'][:, 6])))
                        require(error <= 1e-5 and signs_equal, 'baseline query reproduction differs')
                        reproductions[arm] = {'max_abs_error': error, 'gripper_signs_equal': signs_equal}
                    archives[arm] = cached.write_npz(artifacts / f'seed{seed}-{arm}-query-predictions.npz', {'source_rows': qr, 'predictions': pred})
                    metrics[arm] = parent.stratified(pred, qt[qr], qp, np.ones(len(qr), bool))
                query_rows.append({'training_seed': seed, 'query_metrics': metrics, 'prediction_archives': archives,
                    'baseline_reproduction': reproductions})
            query_cue = comparison(query_rows, 'query_metrics')
        else:
            query_cue = None
        expected = {'readout_models': 24 if oof_cue['joint_cue'] else 18,
            'optimizer_updates': 12000 if oof_cue['joint_cue'] else 9000,
            'backward_calls': 12000 if oof_cue['joint_cue'] else 9000,
            'train_forward_batches': 12000 if oof_cue['joint_cue'] else 9000,
            'train_valid_rows': 768000 if oof_cue['joint_cue'] else 576000,
            'train_linear_calls': 18000 if oof_cue['joint_cue'] else 13500,
            'metric_forward_batches': 96 if oof_cue['joint_cue'] else 72,
            'metric_valid_rows': 378144 if oof_cue['joint_cue'] else 283608,
            'metric_linear_calls': 144 if oof_cue['joint_cue'] else 108,
            'query_readout_batches': 6 if oof_cue['joint_cue'] else 0,
            'query_readout_rows': 15108 if oof_cue['joint_cue'] else 0,
            'query_linear_calls': 9 if oof_cue['joint_cue'] else 0,
            'support_feature_archives_loaded': 3, 'query_feature_archives_loaded': 3 if oof_cue['joint_cue'] else 0,
            'query_target_archives_loaded': 1 if oof_cue['joint_cue'] else 0,
            **{k: 0 for k in ('new_encoder_calls', 'new_HDF5_RGB_initial_private_checkpoint_payload_reads', 'new_environment_calls', 'original_policy_updates_new')}}
        require(counts == expected, 'actual budget differs')
        for pin in c['inputs'].values():
            require(sha(pin['path']) == pin['sha256'], 'input mutated')
        record('complete', actual_counts=counts, query_comparison=query_cue)
        result = {'status': 'complete_routed_history_support_query' if oof_cue['joint_cue'] else 'complete_routed_history_stopped_at_support',
            'formal_result': False, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path), 'scope': c['scope'],
            'analysis_definition': c['analysis'], 'input_provenance': c['inputs'], 'actual_counts': counts,
            'readout_results': heads, 'oof_results': oof_rows, 'support_comparison': oof_cue, 'full_support_results': full_rows,
            'query_shared_archive': query_shared, 'query_results': query_rows, 'query_comparison': query_cue,
            'old_fully_historical_candidate_still_failed': True, 'automatic_rollout_or_spt_promotion': False,
            'events_archive': {'path': str(artifacts / 'events.jsonl'), 'sha256': sha(artifacts / 'events.jsonl')},
            'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__,
                'device': 'cpu', 'threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(),
                'deterministic': torch.are_deterministic_algorithms_enabled(), 'gpu_occupancy_before_training': occupancy,
                'launch_environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
                'peak_process_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                'entry_elapsed_seconds': time.perf_counter() - start, 'argv': sys.argv}}
        cached.write_json(output, result)
        return result
    except Exception as exc:
        partial = None
        if artifacts.exists() and 'model' in active:
            model = active['model']
            partial = cached.write_npz(artifacts / 'failed-active-readout.npz', {
                'weight': model.weight.detach().numpy().copy(), 'bias': model.bias.detach().numpy().copy(),
                'completed_updates': np.array([active['update']], np.int32)})
        cached.write_json(failure, {'status': 'failed_routed_history_readout', 'stage': stage,
            'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts,
            'completed_readouts': heads, 'oof_results': oof_rows, 'query_results': query_rows,
            'active_context': active.get('context'), 'active_partial_archive': partial})
        raise
    finally:
        if event_handle:
            event_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.execute:
        r = run(args.config, args.binding, args.output, args.artifacts)
        print(json.dumps({k: r[k] for k in ('status', 'actual_counts', 'support_comparison', 'query_comparison')}))
    else:
        c, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'scope': c['scope']}))


if __name__ == '__main__':
    main()
