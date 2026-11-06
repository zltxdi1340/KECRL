"""Freeze/check sequential readout and pre-action encoder alignment on source data."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import libero_routed_history_runtime as runtime

saved = runtime.saved
require, sha, load, resolve = saved.require, saved.sha, saved.load, saved.resolve
SEEDS, ARMS = saved.SEEDS, runtime.ARMS
PREFIX = 'libero-task1-b-routed-runtime-interface'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_ARTIFACTS = Path(f'/tmp/{PREFIX}-artifacts-20261104-v1')
CONFIG_SHA = '18f178f15c77585464e93f8c002e8dd764fd919d85da225a50f3f9f44f0669f9'


def preflight(config_path, binding_path):
    c = load(config_path)
    require(c['formal_result'] is False, 'diagnostic only')
    expected = {'task_id': 1, 'feature_arm': 'B_fomaml', 'training_seeds': list(SEEDS), 'arms': list(ARMS),
        'encoder_updates': 500, 'readout_updates': 500, 'closed_loop_default_stabilization_steps': 5,
        'source_role': 'meta_support', 'demonstration_indices': list(range(30)), 'history_steps': 5, 'startup_actions': 4,
        'cached_source_rows': 4059, 'cached_common_rows': 3939, 'cached_stream_readout_calls': 24354,
        'cached_stream_common_comparisons': 23634, 'cached_stream_startup_calls': 720,
        'runtime_rgb_support_windows': 8, 'runtime_rgb_support_observation_rows_per_modality': 80,
        'runtime_rgb_support_action_rows': 80, 'encoder_batch_calls': 3, 'encoder_step_calls': 240,
        'encoder_repeat_calls': 3, 'encoder_total_calls': 246, 'encoder_total_valid_rows': 483,
        'online_bridge_readout_calls': 480, 'total_readout_calls': 24834, 'checkpoint_loads': 3,
        'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_calls': 0, 'new_query_payload_reads': 0,
        'new_initial_private_state_reads': 0, 'sealed_roles_accessed': False, 'checkpoint_selected': False,
        'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
    for k, v in expected.items():
        require(c['scope'].get(k) == v and type(c['scope'].get(k)) is type(v), f'frozen scope differs: {k}')
    require(sha(config_path) == CONFIG_SHA, 'scientific config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input hash differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and
            set(b['files']) == {'runner', 'runtime', 'tests', 'protocol'}, 'binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'bound code differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner path differs')
    for name in ('routed_binding', 'parent_binding'):
        for pin in load(c['inputs'][name]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'parent bound code differs')
    return c, b


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failure = output.with_name(output.stem + '-failed.json')
    require(not any(p.exists() for p in (output, failure, artifacts)), 'outputs already exist')
    counts = {k: 0 for k in ('cached_stream_readout_calls', 'cached_stream_common_comparisons', 'cached_stream_startup_calls',
        'runtime_rgb_support_windows', 'runtime_rgb_support_observation_rows_per_modality', 'runtime_rgb_support_action_rows',
        'encoder_batch_calls', 'encoder_step_calls', 'encoder_repeat_calls', 'encoder_total_calls', 'encoder_total_valid_rows',
        'online_bridge_readout_calls', 'total_readout_calls', 'checkpoint_loads', 'new_optimizer_updates', 'new_backward_calls',
        'new_environment_calls', 'new_query_payload_reads', 'new_initial_private_state_reads')}
    start, stage, rows = time.perf_counter(), 'preflight', []
    try:
        c, _ = preflight(config_path, binding_path)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and all(os.environ.get(k) == '1' for k in
            ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'fixed CPU launch differs')
        import torch
        import h5py
        require(sys.version.split()[0] == '3.10.22' and torch.__version__ == '2.4.1+cu124' and
                np.__version__ == '1.22.4' and h5py.__version__ == '3.11.0' and torch.cuda.device_count() == 0, 'runtime differs')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(load(c['inputs']['routed_result']['path'])['query_comparison']['joint_cue'] is True, 'routed candidate gate failed')
        artifacts.mkdir(parents=True, exist_ok=False)
        with np.load(resolve(c['inputs']['task1_archive']['path']), allow_pickle=False) as z:
            targets, ids, steps = [z[k] for k in ('targets', 'demonstration_indices', 'action_steps')]
        plan = saved.make_plan(targets, ids, steps)
        source = plan['eligible']
        require(len(targets) == 4059 and len(source) == 3939 and set(ids.tolist()) == set(range(30)), 'support role/count differs')
        all_cache, bundles = {}, {}
        for seed in SEEDS:
            stage = f'cached_stream_seed{seed}'
            heads = runtime.load_heads(c['inputs'], seed)
            bundles[seed] = heads
            with np.load(resolve(c['inputs'][f'{seed}_support_features']['path']), allow_pickle=False) as z:
                features, original = z['projection'], z['predictions']
            stream, diagnostics = {}, {}
            for arm in ARMS:
                reader = runtime.FrozenHistoryReadout(heads, arm)
                before = reader.state_hash()
                values = np.empty((4059, 7), np.float32)
                for demo in range(30):
                    reader.reset()
                    positions = np.flatnonzero(ids == demo)
                    for i in positions:
                        values[i], meta = reader.act(features[i], int(steps[i]))
                        require(meta['history_length'] == min(5, int(steps[i])) and
                                meta['history_policy_steps'] == list(range(int(steps[i]), max(0, int(steps[i])-5), -1)), 'history sequence differs')
                        counts['cached_stream_readout_calls'] += 1
                        counts['total_readout_calls'] += 1
                        counts['cached_stream_startup_calls'] += int(meta['startup'])
                with np.load(resolve(c['inputs'][f'{seed}_{arm}']['path']), allow_pickle=False) as z:
                    require(np.array_equal(z['source_rows'], source), 'saved head source rows differ')
                    error = float(np.max(np.abs(values[source] - z['predictions'][-1])))
                    equal_signs = bool(np.array_equal(np.sign(values[source, 6]), np.sign(z['predictions'][-1, :, 6])))
                require(error <= 1e-5 and equal_signs and reader.state_hash() == before, 'sequential readout alignment differs')
                counts['cached_stream_common_comparisons'] += len(source)
                stream[arm] = values
                diagnostics[arm] = {'max_abs_error': error, 'gripper_signs_equal': equal_signs, 'weights_unchanged': True}
            require(np.array_equal(stream[ARMS[0]][steps < 5], stream[ARMS[1]][steps < 5]), 'shared startup differs')
            pin = saved.write_npz(artifacts / f'seed{seed}-cached-sequential-actions.npz', {'source_rows': np.arange(4059), **stream})
            all_cache[seed] = {'features': features, 'original': original, 'stream': stream}
            rows.append({'training_seed': seed, 'cached_stream_comparisons': diagnostics, 'cached_stream_archive': pin})
        stage = 'source_rgb_payload'
        from torch.utils.data import default_collate
        from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
        from experiments import run_libero_source_internal_phase_history_observability as capture
        transfer, manifest, learning = [load(c['inputs'][k]['path']) for k in ('transfer', 'dataset_manifest', 'learning')]
        file_spec = next(x for x in manifest['files'] if x['task_id'] == 1)
        dataset_root = Path(c['execution']['dataset_root'])
        dataset_path = dataset_root / file_spec['path']
        require(dataset_path.stat().st_size == file_spec['size_bytes'] and sha(dataset_path) == file_spec['sha256'], 'HDF5 source provenance differs')
        sample = next(x for x in load(c['inputs']['sample_manifest']['path'])['tasks'] if x['task_id'] == 1)
        digest = next(x for x in load(c['inputs']['original_frame_result']['path'])['archives'] if x['task_id'] == 1)['shared_batch_digests'][0]
        ds = CausalT1LiberoSequenceDataset(transfer, manifest, dataset_root, task_id=1, demonstration_role='meta_support',
                                         sequence_length=10, stride=10, verify_sha256=False)
        try:
            require([asdict(w) for w in ds.windows[:8]] == sample['canonical_windows'][:8], 'source RGB window metadata differs')
            batch = default_collate([ds[i] for i in range(8)])
            access = {**ds.access_record, 'payload_windows': sample['canonical_windows'][:8], 'action0_read': False}
        finally:
            ds.close()
        require(saved.tensor_digest({k: batch[k] for k in capture.control.BATCH_KEYS}) == digest, 'source batch input digest differs')
        counts['runtime_rgb_support_windows'] = 8
        counts['runtime_rgb_support_observation_rows_per_modality'] = counts['runtime_rgb_support_action_rows'] = 80
        for fixed, row in zip(c['fixed_encoders'], rows):
            seed, stage = fixed['checkpoint']['training_seed'], 'encoder_single_step_bridge'
            encoder = runtime.load_encoder(fixed['checkpoint'], learning)
            counts['checkpoint_loads'] += 1
            with torch.inference_mode():
                pred, feat = capture.capture(encoder, [batch[k] for k in capture.f.OBS_SHAPES])
            counts['encoder_batch_calls'] += 1
            counts['encoder_total_calls'] += 1
            counts['encoder_total_valid_rows'] += 80
            bp, bf = pred.numpy().reshape(80, 7), feat.numpy().reshape(80, 256)
            require(float(np.max(np.abs(bf-all_cache[seed]['features'][:80]))) <= 1e-5 and
                    float(np.max(np.abs(bp-all_cache[seed]['original'][:80]))) <= 1e-5, 'batch/cache encoder bridge differs')
            readers = {a: runtime.FrozenHistoryReadout(bundles[seed], a) for a in ARMS}
            for reader in readers.values():
                reader.reset()
            sf, sp = np.empty_like(bf), np.empty_like(bp)
            native_actions = {a: np.empty((80, 7), np.float32) for a in ARMS}
            encoder_start = time.perf_counter()
            first_obs = None
            for i in range(80):
                b, t = divmod(i, 10)
                obs = {k: batch[k][b, t].permute(1, 2, 0).contiguous().numpy().copy() if k.endswith('rgb') else
                       batch[k][b, t].numpy().copy() for k in capture.f.OBS_SHAPES}
                if i == 0:
                    first_obs = obs
                sf[i], sp[i] = runtime.encode_step(encoder, obs)
                counts['encoder_step_calls'] += 1
                counts['encoder_total_calls'] += 1
                counts['encoder_total_valid_rows'] += 1
                for arm, reader in readers.items():
                    native_actions[arm][i], _meta = reader.act(sf[i], i+1)
                    counts['online_bridge_readout_calls'] += 1
                    counts['total_readout_calls'] += 1
            per_step_seconds = (time.perf_counter() - encoder_start) / 80
            repeat_f, repeat_p = runtime.encode_step(encoder, first_obs)
            counts['encoder_repeat_calls'] += 1
            counts['encoder_total_calls'] += 1
            counts['encoder_total_valid_rows'] += 1
            require(np.array_equal(repeat_f, sf[0]) and np.array_equal(repeat_p, sp[0]), 'encoder repeat not bitwise')
            deltas = {'step_vs_batch_feature': float(np.max(np.abs(sf-bf))), 'step_vs_batch_original_action': float(np.max(np.abs(sp-bp))),
                      'step_vs_cache_feature': float(np.max(np.abs(sf-all_cache[seed]['features'][:80])))}
            deltas.update({a: float(np.max(np.abs(v-all_cache[seed]['stream'][a][:80]))) for a, v in native_actions.items()})
            require(all(v <= 1e-5 for v in deltas.values()), 'runtime source input/readout alignment differs')
            require(saved.tensor_digest(encoder.state_dict()) == fixed['checkpoint']['weight_hash'] and
                    all(p.grad is None and not p.requires_grad for p in encoder.parameters()), 'fixed encoder mutated')
            row.update(online_bridge_errors=deltas, encoder_repeat_bitwise=True, mean_online_call_seconds=per_step_seconds,
                bridge_archive=saved.write_npz(artifacts / f'seed{seed}-online-bridge.npz', {'projection': sf,
                    'original_predictions': sp, 'batch_projection': bf, 'batch_original_predictions': bp, **native_actions}))
        require(all(counts[k] == c['scope'][k] for k in counts), 'actual interface budget differs')
        require(saved.tensor_digest({k: batch[k] for k in capture.control.BATCH_KEYS}) == digest, 'source tensors mutated')
        for pin in c['inputs'].values():
            require(sha(pin['path']) == pin['sha256'], 'input mutated')
        result = {'status': 'complete_routed_runtime_interface', 'formal_result': False, 'passed': True,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path),
            'binding_sha256': sha(binding_path), 'scope': c['scope'], 'analysis': c['analysis'], 'inputs': c['inputs'],
            'fixed_encoders': c['fixed_encoders'], 'actual_counts': counts, 'seed_results': rows,
            'source_access': access, 'source_batch_digest': digest, 'automatic_closed_loop': False,
            'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
                'device': 'cpu', 'threads': 1, 'entry_elapsed_seconds': time.perf_counter()-start, 'argv': sys.argv}}
        saved.write_json(output, result)
        return result
    except Exception as exc:
        saved.write_json(failure, {'status': 'failed_routed_runtime_interface', 'stage': stage, 'error_type': type(exc).__name__,
            'error': str(exc), 'actual_counts_so_far': counts, 'completed_seed_results': rows})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in [('config', DEFAULT_CONFIG), ('binding', DEFAULT_BINDING), ('output', DEFAULT_OUTPUT), ('artifacts', DEFAULT_ARTIFACTS)]:
        parser.add_argument('--'+key, type=Path, default=value)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    if a.execute:
        r = run(a.config, a.binding, a.output, a.artifacts)
        print(json.dumps({'status': r['status'], 'actual_counts': r['actual_counts']}))
    else:
        c, _b = preflight(a.config, a.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'scope': c['scope']}))


if __name__ == '__main__':
    main()
