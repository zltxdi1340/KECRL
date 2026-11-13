"""Bound fixed500 feature capture, isolated CPU direction probes and independent audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from experiments import libero_source_expert_component3_observability_protocol as d
from experiments import run_libero_source_expert_late_translation_readout as old
from experiments.libero_formal_execution import Ledger
from experiments.libero_source_expert_component3_observability_access import (
    ArrayOnlyGuard, CaptureGuard, capture_caps, load_array, save_array, same_array_bits,
)

p = d.p
SCIENCE_SHA = 'acac57118851969ece67cd2fe606927678e34a50647db78343aaf7280f513550'
MODULE = 'experiments.run_libero_source_expert_component3_observability'


def preflight(config_path=d.DEFAULT_CONFIG, binding_path=d.DEFAULT_BINDING):
    p.require(p.sha(config_path) == SCIENCE_SHA, 'scientific config changed')
    binding = d.require_execution_binding(config_path, binding_path)
    config = d.validate(config_path)
    for ref in binding['code_references'] + [binding['execution_contract']]:
        p.require(p.sha(p.resolve(ref['path'])) == ref['sha256'], 'bound dependency/contract changed')
    source = p.load_pin(config['inputs']['readout_config']); parent = p.load_pin(config['inputs']['readout_result'])
    p.require(binding['capture_caps'] == capture_caps(source), 'capture accounting changed')
    return config, binding, source, parent


def check_environment(binding, phase):
    environment = binding['capture_environment'] if phase == 'capture' else binding['CPU_environment']
    for key, value in environment.items(): p.require(os.environ.get(key) == value, 'launch environment differs: ' + key)
    p.require(sys.version.split()[0] == '3.10.22', 'Python version changed')
    for name, expected in {'torch': '2.4.1+cu124', 'numpy': '1.22.4', 'h5py': '3.11.0'}.items():
        p.require(importlib.metadata.version(name) == expected, 'runtime version changed: ' + name)


def capture_forward(policy, observations):
    import torch
    hidden, logits = [], []
    handles = [policy.action_head.register_forward_pre_hook(lambda _m, args: hidden.append(args[0].detach().clone())),
               policy.action_head.register_forward_hook(lambda _m, _args, output: logits.append(output.detach().clone()))]
    try:
        prediction, state = policy.forward_sequence(*observations)
        p.require(state is None and policy.config.memory_mode == 'cnn_only' and len(hidden) == len(logits) == 1,
                  'unexpected capture state/hook count')
        h, z = hidden[0], logits[0]
        p.require(h.shape == (*prediction.shape[:2], 256) and z.shape == prediction.shape
                  and h.dtype == z.dtype == prediction.dtype == torch.float32
                  and torch.isfinite(h).all() and torch.isfinite(z).all() and torch.isfinite(prediction).all()
                  and same_tensor_bits(torch.tanh(z), prediction), 'feature/logit/action reconstruction differs')
        return prediction, h, z
    finally:
        for handle in handles: handle.remove()


def same_tensor_bits(a, b):
    import torch
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def actual_plan_counts(plans):
    selected = sum(len(plan['selected']) for plan in plans.values())
    pairs = sum(sum(f['material_query_rows'] * sum(f['reference_class_n'].values()) for f in plan['fold_counts'])
                for plan in plans.values())
    return {'primary_scored_output_rows': selected * 24, 'secondary_scored_output_rows': selected * 4,
            'primary_lag_pair_distances': pairs * 30, 'secondary_lag_pair_distances': pairs * 5,
            'primary_coordinate_squared_differences': pairs * 30 * 256,
            'secondary_coordinate_squared_differences': pairs * 5 * 9,
            'primary_fold_mode_rule_records': 3600, 'secondary_fold_mode_rule_records': 600}


def capture(config_path, binding_path):
    config, binding, source, parent = preflight(config_path, binding_path)
    check_environment(binding, 'capture')
    directory = Path(config['artifacts']); output = Path(config['output'])
    p.require(not directory.exists() and not output.exists()
              and not output.with_name(output.stem + '-capture-failed.json').exists(), 'existing output/partial; no retry')
    try: waiting = old.wait_gpu(config['execution']['GPU_wait_seconds'])
    except BaseException as exc:
        old.write_new(output.with_name(output.stem + '-capture-failed.json'),
                      {'status': 'capture_stopped_before_payload', 'error': repr(exc), 'research_payload_calls': 0})
        raise
    preflight(config_path, binding_path)
    directory.mkdir(parents=True, exist_ok=False); events = (directory / 'capture-events.jsonl').open('x')
    ledger = Ledger(binding['capture_caps'], seconds=600); started = datetime.now(timezone.utc).isoformat()
    guard = None; rows = []; shared = []; after = []; accesses = []; hashes = {}; peak = 0
    def event(kind, **values):
        events.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}) + '\n'); events.flush()
    try:
        import numpy as np
        import torch
        from torch.utils.data import DataLoader
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
        from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
        from experiments import libero_component3_direction_neighbors as n
        p.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'single selected CUDA device required')
        torch.set_num_threads(1); random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022); torch.cuda.manual_seed_all(20261022)
        torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        p.require(torch.version.cuda == '12.4', 'CUDA version differs')
        refs = parent['labels'] + [r['predictions'] for r in parent['models']]
        guard = CaptureGuard(source, ledger, refs)
        policy_config = LiberoBCPolicyConfig.from_mapping(config['policy_config']); models = {}
        legacy_models = {(r['task'], r['seed'], r['arm']): r for r in parent['models']}
        legacy_labels = {r['task']: r for r in parent['labels']}
        legacy_hashes = {(r['task'], r['batch']): r['sha256'] for r in parent['shared_batch_input_hashes']}
        references = {(r['task'], r['seed'], r['arm']): r['metrics'] for r in source['G_reference_selection']['source500_references']}
        transfer = p.load_pin(source['inputs']['transfer']); manifest = p.load_pin(source['inputs']['dataset_manifest'])
        with guard:
            for endpoint in config['endpoints']:
                checkpoint = endpoint['checkpoint']; key = endpoint['task'], endpoint['seed'], endpoint['arm']
                p.require(p.sha(checkpoint['path']) == checkpoint['sha256'], 'checkpoint file changed')
                state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
                p.require(old.tensor_digest(state) == checkpoint['weight_hash'] and all(torch.isfinite(v).all() for v in state.values()), 'weight hash/values differ')
                policy = ledger.call('policy_object', make_deterministic_libero_policy, policy_config)
                policy.load_state_dict(state, strict=True); policy.requires_grad_(False); policy.eval()
                p.require(old.tensor_digest(policy.state_dict()) == checkpoint['weight_hash'], 'loaded weight hash differs')
                models[key] = policy
            event('all30_models_verified_before_data')
            labels = {task: load_array(legacy_labels[task]) for task in p.TASKS}
            plans = {task: n.frame_plan(labels[task]) for task in p.TASKS}
            for task in p.TASKS:
                p.require(len(labels[task]['demo_lengths']) == 30 and
                          labels[task]['demo_lengths'].tolist() == config['action_lengths'][str(task)], 'source length metadata differs')
                bounds = next(r for r in config['budget']['per_task'] if r['task'] == task)
                p.require(bounds['material_common_rows_lower'] <= len(plans[task]['selected']) <= bounds['material_common_rows_upper'], 'selected count outside frozen bounds')
            counts = actual_plan_counts(plans)
            for name, value in counts.items():
                cap = config['budget']['exact'].get(name, config['budget']['label_dependent_upper_bounds'].get(name))
                p.require(value <= cap, 'actual source-label plan exceeds budget: ' + name)
            label_plan = {'counts': counts, 'per_task': {str(t): {k: plans[t][k] for k in ('fold_counts', 'excluded_early_rows', 'excluded_neutral_common_rows')}
                                                        for t in p.TASKS}}
            old.write_new(directory / 'label-plan-before-first-CNN.json', label_plan)
            event('all_source_label_plans_fixed_before_first_CNN', counts=counts)
            for task in p.TASKS:
                file = config['data_files'][str(task)]; path = Path(config['data_root']) / file['path']
                p.require(path.stat().st_size == file['size_bytes'] and p.sha(path) == file['sha256'], 'source HDF5 changed')
                dataset = CausalT1LiberoSequenceDataset(transfer, manifest, Path(config['data_root']), task_id=task,
                    demonstration_role='meta_support', sequence_length=10, stride=10, access_phase='post_method_freeze', verify_sha256=False)
                windows = p.window_plan(config['action_lengths'][str(task)])
                p.require([{'demo': w.demonstration_index, 'start': w.start, 'valid': w.valid_length} for w in dataset.windows] == windows, 'source window plan differs')
                items = []
                try:
                    for i in range(len(dataset)): items.append(ledger.call('source_window', dataset.__getitem__, i))
                    access = {**dataset.access_record, 'file_sha256_verified_before_payload': True, 'action0_payload_reads': 0}
                finally: dataset.close()
                cache_bytes = sum(v.numel() * v.element_size() for item in items for v in item.values() if torch.is_tensor(v))
                p.require(cache_bytes <= config['execution']['host_observation_cache_bytes_max'], 'observation cache cap exceeded'); peak = max(peak, cache_bytes)
                accesses.append({**access, 'cache_bytes': cache_bytes})
                targets = np.concatenate([item['actions'][:w['valid']].numpy() for item, w in zip(items, windows)])
                demos = np.concatenate([np.full(w['valid'], w['demo'], np.int32) for w in windows])
                steps = np.concatenate([np.arange(w['start'], w['start'] + w['valid'], dtype=np.int32) for w in windows])
                old_label = labels[task]
                p.require(same_array_bits(targets, old_label['targets']) and same_array_bits(demos, old_label['demo_indices'])
                          and same_array_bits(steps, old_label['action_steps']), 'legacy label/frame parity differs')
                ledger.begin('legacy_label_parity'); ledger.finish('legacy_label_parity')
                public = np.concatenate([torch.cat((item['joint_states'][:w['valid']], item['gripper_states'][:w['valid']]), -1).numpy()
                                         for item, w in zip(items, windows)])
                shared_pin = save_array(directory / f'task{task}-shared.npz', **old_label, public_proprio=public,
                    common_rows=plans[task]['common'], history_rows=plans[task]['history'], selected_rows=plans[task]['selected'], direction_labels=plans[task]['labels'])
                shared.append({'task': task, **shared_pin}); event('source_task_cached', task=task, access=accesses[-1])
                for endpoint in [r for r in config['endpoints'] if r['task'] == task]:
                    key = task, endpoint['seed'], endpoint['arm']; policy = models[key].to('cuda:0').eval()
                    legacy_prediction = load_array(legacy_models[key]['predictions'])
                    predictions = []; features = []; logits = []; repeat_pin = None
                    for batch_number, batch in enumerate(DataLoader(items, batch_size=32, shuffle=False, num_workers=0)):
                        digest = old.tensor_digest({k: batch[k] for k in old.BATCH_KEYS})
                        p.require(digest == legacy_hashes[task, batch_number], 'legacy batch input hash differs')
                        ledger.begin('legacy_batch_hash_checks'); ledger.finish('legacy_batch_hash_checks')
                        if (task, batch_number) not in hashes: hashes[task, batch_number] = digest
                        else:
                            p.require(hashes[task, batch_number] == digest, 'cross-model expert input differs')
                            ledger.begin('input_paired'); ledger.finish('input_paired')
                        gpu = {k: v.to('cuda:0') for k, v in batch.items()}; mask = gpu['valid_mask'].bool()
                        with torch.inference_mode():
                            event('main_forward_attempt', task=task, seed=key[1], arm=key[2], batch=batch_number, input_sha256=digest)
                            pred, hidden, raw_logits = ledger.call('main_forward', capture_forward, policy, [gpu[k] for k in p.OBS])
                            ledger.begin('logit_reconstruction'); ledger.finish('logit_reconstruction')
                            valid, slots = int(mask.sum()), mask.numel()
                            for counter, amount in [('main_valid_rows', valid), ('main_slots', slots)]: ledger.begin(counter, amount); ledger.finish(counter, amount)
                            predictions.append(pred[mask].cpu().numpy().copy()); features.append(hidden[mask].cpu().numpy().copy()); logits.append(raw_logits[mask].cpu().numpy().copy())
                            if batch_number == 0:
                                event('repeat_forward_attempt', task=task, seed=key[1], arm=key[2], input_sha256=digest)
                                again, h_again, z_again = ledger.call('repeat_forward', capture_forward, policy, [gpu[k] for k in p.OBS])
                                ledger.begin('logit_reconstruction'); ledger.finish('logit_reconstruction')
                                for counter, amount in [('repeat_valid_rows', valid), ('repeat_slots', slots)]: ledger.begin(counter, amount); ledger.finish(counter, amount)
                                repeat_pin = save_array(directory / f'task{task}-seed{key[1]}-{key[2]}-repeat.npz',
                                    main_predictions=pred.cpu().numpy(), repeat_predictions=again.cpu().numpy(),
                                    main_features=hidden.cpu().numpy(), repeat_features=h_again.cpu().numpy(),
                                    main_logits=raw_logits.cpu().numpy(), repeat_logits=z_again.cpu().numpy(), valid_mask=mask.cpu().numpy())
                                p.require(same_tensor_bits(pred, again) and same_tensor_bits(hidden, h_again) and same_tensor_bits(raw_logits, z_again), 'full-slot first-batch repeat differs')
                                ledger.begin('repeat_exact'); ledger.finish('repeat_exact')
                        p.require(old.tensor_digest({k: gpu[k] for k in old.BATCH_KEYS}) == digest, 'input mutated')
                        ledger.begin('input_unchanged'); ledger.finish('input_unchanged')
                        event('main_forward_complete', task=task, seed=key[1], arm=key[2], batch=batch_number, input_sha256=digest, valid_rows=valid, slots=slots)
                    prediction = np.concatenate(predictions); feature = np.concatenate(features); logit = np.concatenate(logits)
                    p.require(prediction.dtype == legacy_prediction.dtype == np.float32 and same_array_bits(prediction, legacy_prediction), 'all-seven-dimensional legacy prediction parity differs')
                    ledger.begin('legacy_prediction_parity_rows', len(prediction)); ledger.finish('legacy_prediction_parity_rows', len(prediction))
                    measured = old.frame_metrics(prediction, targets, old_label['tail20']); old.check_parity(measured['all'], references[key])
                    ledger.begin('G_parity'); ledger.finish('G_parity')
                    p.require(old.tensor_digest(policy.state_dict()) == endpoint['checkpoint']['weight_hash']
                              and all(v.grad is None and not v.requires_grad for v in policy.parameters()), 'weights/gradients changed')
                    ledger.begin('weight_after'); ledger.finish('weight_after')
                    after.append({'task': task, 'seed': key[1], 'arm': key[2], 'weight_hash': endpoint['checkpoint']['weight_hash'], 'no_gradients': True})
                    feature_pin = save_array(directory / f'task{task}-seed{key[1]}-{key[2]}-features.npz', features=feature, logits=logit, predictions=prediction)
                    rows.append({'task': task, 'seed': key[1], 'arm': key[2], 'checkpoint': endpoint['checkpoint'], 'features': feature_pin, 'repeat': repeat_pin,
                                 'G500_parity': measured['all'], 'legacy_prediction_bit_exact': True})
                    models[key] = policy.cpu(); event('model_capture_complete', task=task, seed=key[1], arm=key[2])
                    print(json.dumps({'stage': 'capture', 'models': len(rows), 'main_batches': ledger.counts['main_forward']['completed'],
                                      'repeat_batches': ledger.counts['repeat_forward']['completed']}), flush=True)
                del items, targets, public
            p.require(set(guard.array_completed) == set(guard.references) and len(guard.array_attempts) == 35 and not guard.denied, 'capture reference guard incomplete')
            p.require(len(guard.seen_models) == 30 and len(guard.pointers) == 25 and len(guard.opens) == 5
                      and all(guard.opens[k] == 2 for k in guard.opens)
                      and all(index == len(guard.plans[task]) for (task, _field), index in guard.pointers.items()), 'capture guard coverage differs')
        for name, value in ledger.counts.items(): p.require(value['attempted'] == value['completed'] == ledger.caps[name], 'capture accounting mismatch: ' + name)
        record = {'status': 'complete_component3_feature_capture', 'formal_result': False, 'started_at_utc': started,
                  'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'science_config': p.pin(config_path), 'binding': p.pin(binding_path),
                  'shared': shared, 'models': rows, 'weight_after': after, 'label_plan': p.pin(directory / 'label-plan-before-first-CNN.json'),
                  'planned_probe_counts_before_first_CNN': counts, 'ledger': ledger.snapshot(), 'data_access': accesses,
                  'HDF5_file_opens': guard.opens, 'payload_window_counts': [{'task': t, 'field': f, 'windows': v} for (t, f), v in guard.pointers.items()],
                  'reference_arrays': refs, 'reference_array_loads': guard.array_completed, 'denied_operations': guard.denied,
                  'shared_batch_input_hashes': [{'task': t, 'batch': b, 'sha256': h} for (t, b), h in hashes.items()],
                  'peak_host_cache_bytes': peak, 'gpu_wait': waiting, 'events': p.pin(directory / 'capture-events.jsonl'),
                  'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'torch': torch.__version__, 'cuda': torch.version.cuda,
                              'device': 'cuda:0', 'GPU_name': torch.cuda.get_device_name(0), 'physical_GPU_uuid': config['execution']['GPU_uuid'],
                              'readout_seed': 20261022, 'deterministic': True, 'TF32': False},
                  'new_training_updates': 0, 'new_environment_calls': 0, 'G_failed_H_unrun_sealed_roles_unchanged': True,
                  'SPT_Knowledge_Module_changes': False}
        old.write_new(directory / 'capture.json', record)
    except BaseException as exc:
        old.write_new(directory / 'capture-failed-partial.json', {'status': 'capture_failed_no_retry', 'error': repr(exc), 'ledger': ledger.snapshot(),
            'completed_models': rows, 'completed_shared': shared, 'denied_operations': guard.denied if guard else []})
        raise
    finally: events.close()
    return record


def probe(config_path, binding_path):
    config, binding, _, _ = preflight(config_path, binding_path); check_environment(binding, 'probe')
    from experiments import libero_component3_direction_neighbors as n
    import numpy as np
    directory = Path(config['artifacts']); output = Path(config['output']); captured = json.loads((directory / 'capture.json').read_text())
    probe_paths = [directory / f'task{task}-public9-probe.npz' for task in p.TASKS]
    probe_paths += [directory / f"task{r['task']}-seed{r['seed']}-{r['arm']}-probe.npz" for r in captured['models']]
    p.require(not output.exists() and captured['status'] == 'complete_component3_feature_capture' and captured.get('formal_result') is False
              and not (directory / 'probe-failed-partial.json').exists() and not any(path.exists() for path in probe_paths)
              and captured['science_config'] == p.pin(config_path) and captured['binding'] == p.pin(binding_path), 'capture not complete or output/partial exists')
    caps = {key: config['budget']['exact'][key] for key in ('primary_fold_mode_rule_records', 'secondary_fold_mode_rule_records')}
    caps.update({key: value for key, value in config['budget']['label_dependent_upper_bounds'].items()
                 if key.startswith(('primary_', 'secondary_')) and key != 'material_common_rows_models'})
    caps['CPU_probe_new_array_loads'] = 35
    ledger = Ledger(caps, seconds=2400); refs = captured['shared'] + [r['features'] for r in captured['models']]
    guard = ArrayOnlyGuard(refs, ledger, 'CPU_probe_new_array_loads', hash_only_files=probe_paths); primary = []; secondary = []; peak = 0
    try:
        with guard:
            shared = {r['task']: load_array(r) for r in captured['shared']}
            plans = {task: n.frame_plan(data) for task, data in shared.items()}
            p.require(actual_plan_counts(plans) == captured['planned_probe_counts_before_first_CNN'], 'planned source label counts changed')
            for task in p.TASKS:
                arrays, folds, metrics, working = n.probe(shared[task]['public_proprio'], plans[task], ledger, 'secondary')
                peak = max(peak, working)
                pin = save_array(directory / f'task{task}-public9-probe.npz', **arrays)
                secondary.append({'task': task, 'space': 'public_joint7_gripper2', 'archive': pin, 'folds': folds, 'metrics': metrics})
                print(json.dumps({'stage': 'public9_probe', 'task': task}), flush=True)
            for model in captured['models']:
                task, seed, arm = (model[k] for k in ('task', 'seed', 'arm')); feature = load_array(model['features'])
                p.require(set(feature) == {'features', 'logits', 'predictions'}, 'unexpected capture feature fields')
                arrays, folds, metrics, working = n.probe(feature['features'], plans[task], ledger, 'primary', policy_values=feature['predictions'][:, 2])
                peak = max(peak, working)
                pin = save_array(directory / f'task{task}-seed{seed}-{arm}-probe.npz', **arrays)
                primary.append({'task': task, 'seed': seed, 'arm': arm, 'checkpoint': model['checkpoint'], 'archive': pin, 'folds': folds, 'metrics': metrics})
                print(json.dumps({'stage': 'primary_probe', 'task': task, 'seed': seed, 'arm': arm, 'models': len(primary),
                                  'elapsed_seconds': ledger.snapshot()['elapsed_seconds']}), flush=True)
                del feature, arrays
            guard.complete()
        expected = {**captured['planned_probe_counts_before_first_CNN'], 'CPU_probe_new_array_loads': 35}
        for key, value in expected.items(): p.require(ledger.counts[key] == {'attempted': value, 'completed': value}, 'probe operation counts differ: ' + key)
        result = {'status': 'complete_component3_observability_pending_independent_audit', 'formal_result': False,
                  'science_config': p.pin(config_path), 'binding': p.pin(binding_path), 'capture': p.pin(directory / 'capture.json'),
                  'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'primary_results': primary, 'secondary_results': secondary,
                  'across_models': n.across_models(primary), 'ledger': ledger.snapshot(), 'array_loads': guard.completed,
                  'denied_operations': guard.denied, 'peak_neighbor_working_bytes': peak,
                  'runtime': {'python': sys.version.split()[0], 'numpy': np.__version__, 'device': 'CPU'},
                  'new_training_updates': 0, 'new_environment_calls': 0, 'G_failed_H_unrun_sealed_roles_unchanged': True,
                  'SPT_Knowledge_Module_changes': False}
        old.write_new(output, result)
    except BaseException as exc:
        old.write_new(directory / 'probe-failed-partial.json', {'status': 'probe_failed_no_retry', 'error': repr(exc), 'ledger': ledger.snapshot(),
            'primary_results': primary, 'secondary_results': secondary, 'array_attempts': guard.attempts, 'denied': guard.denied})
        raise
    return result


def orchestrate(config_path, binding_path):
    config, binding, _, _ = preflight(config_path, binding_path)
    output, directory = Path(config['output']), Path(config['artifacts'])
    p.require(not output.exists() and not directory.exists(), 'existing scientific output/partial; no retry')
    launch = output.with_name(output.stem + '-launch.json'); process_path = output.with_name(output.stem + '-process.json')
    record = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'science_config': p.pin(config_path),
              'binding': p.pin(binding_path), 'processes': []}; old.write_new(launch, record)
    stages = [('capture', MODULE, 4320), ('probe', MODULE, 2460), ('audit', 'experiments.audit_libero_source_expert_component3_observability', 1260)]
    try:
        for phase, module, seconds in stages:
            env = os.environ.copy(); env.update(binding['capture_environment'] if phase == 'capture' else binding['CPU_environment'])
            command = [sys.executable, '-m', module, '--config', str(config_path), '--binding', str(binding_path)]
            if phase != 'audit': command += ['--mode', phase]
            logfile = output.with_name(output.stem + '-' + phase + '.log'); started = time.monotonic()
            with logfile.open('x') as handle:
                child = subprocess.Popen(command, cwd=p.ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                timed_out = False; interrupted = None
                try: code = child.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGINT)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    code = child.returncode
                except BaseException as exc:
                    interrupted = repr(exc)
                    os.killpg(child.pid, signal.SIGINT)
                    try: child.wait(timeout=5)
                    except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
                    code = child.returncode
            item = {'stage': phase, 'command': command, 'environment': binding['capture_environment'] if phase == 'capture' else binding['CPU_environment'],
                    'pid': child.pid, 'returncode': code, 'watchdog_expired': timed_out, 'interrupted': interrupted,
                    'wall_seconds': time.monotonic() - started, 'log': p.pin(logfile)}
            record['processes'].append(item); print(json.dumps(item), flush=True)
            p.require(code == 0 and not timed_out and interrupted is None, 'scientific stage failed; no retry: ' + phase)
        audit = directory / 'independent-audit.json'; checked = json.loads(audit.read_text())
        p.require(checked['status'] == 'passed_independent_component3_observability_audit' and checked['result'] == p.pin(output), 'independent audit did not pass')
        preflight(config_path, binding_path)
        old.write_new(directory / 'completion.json', {'status': 'complete_independently_audited_component3_observability_diagnostic',
            'formal_result': False, 'completed_at_utc': datetime.now(timezone.utc).isoformat(), 'result': p.pin(output), 'audit': p.pin(audit),
            'science_config': p.pin(config_path), 'binding': p.pin(binding_path), 'new_training_updates': 0, 'new_environment_calls': 0,
            'G_failed_H_unrun_sealed_roles_unchanged': True, 'SPT_Knowledge_Module_changes': False})
    finally:
        record['finished_at_utc'] = datetime.now(timezone.utc).isoformat(); old.write_new(process_path, record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=d.DEFAULT_CONFIG); parser.add_argument('--binding', type=Path, default=d.DEFAULT_BINDING)
    parser.add_argument('--mode', choices=('preflight', 'run', 'capture', 'probe'), default='preflight')
    args = parser.parse_args()
    if args.mode == 'preflight':
        config, binding, _, _ = preflight(args.config, args.binding)
        print(json.dumps({'status': 'bound_component3_metadata_preflight_passed', 'science_config': p.pin(args.config), 'budget': config['budget']}))
    elif args.mode == 'run': orchestrate(args.config, args.binding)
    elif args.mode == 'capture': capture(args.config, args.binding)
    else: probe(args.config, args.binding)
