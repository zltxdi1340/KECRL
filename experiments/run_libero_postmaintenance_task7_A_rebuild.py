"""Real support-only A rerun after storage loss; never fabricates old receipts.

This new lineage does not call or weaken historical execution preflights.
The explicitly bound source file is hashed in full; only demos 0..29 and
the five public BC arrays may be numerically decoded.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/libero-postmaintenance-task7-A-rebuild_v1.json'
BINDING = ROOT / 'configs/libero-postmaintenance-task7-A-rebuild-execution-binding_v1.json'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def preflight(config=CONFIG, binding=BINDING):
    cfg = json.loads(Path(config).read_text())
    bound = json.loads(Path(binding).read_text())
    require(cfg['kind'] == 'postmaintenance_A_support_only_rerun'
            and cfg['formal_result'] is False and cfg['historical_restoration'] is False,
            'this must be a new non-formal rerun lineage')
    require(cfg['seeds'] == [20261106, 20261107, 20261108]
            and cfg['arm'] == 'A_original' and cfg['task'] == 7
            and cfg['demonstrations'] == list(range(30))
            and cfg['policy_updates'] == 500 and cfg['source_outer_updates'] == 0
            and cfg['source_meta_query_allowed'] is False,
            'support-only scope changed')
    require(bound['status'] == 'bound_before_payload'
            and bound['science'] == pin(config)
            and bound['synthetic_validation_passed'] is True,
            'new execution binding required before any research read')
    for ref in bound['code_dependencies']:
        require(pin(ref['path']) == ref, 'bound rerun code changed: ' + ref['path'])
    require(str(Path(__file__).resolve()) in {r['path'] for r in bound['code_dependencies']},
            'rerun entry is not bound')
    require(Path(cfg['artifacts']).is_relative_to(ROOT / 'results'), 'persistent result directory required')
    for key, expected in cfg['environment'].items():
        require(os.environ.get(key) == expected, 'launch environment differs: ' + key)
    require(sys.version.split()[0] == '3.10.22', 'Python differs')
    for name, version in cfg['versions'].items():
        require(importlib.metadata.version(name) == version, 'package version differs: ' + name)
    return cfg, bound


def admit_hdf5_path(name):
    if name in ('/', '/data'):
        return
    match = re.fullmatch(r'/data/demo_(\d+)(?:/(obs)(?:/(agentview_rgb|eye_in_hand_rgb|joint_states|gripper_states))?|/(actions))?', name)
    if match is None or not 0 <= int(match.group(1)) < 30:
        raise PermissionError('HDF5 path outside task7 support public BC fields: ' + name)


@contextmanager
def support_guard(source, counters):
    import h5py
    file_call = h5py.File
    group_get = h5py.Group.__getitem__
    array_get = h5py.Dataset.__getitem__
    attribute_get = h5py.AttributeManager.__getitem__
    allowed = Path(source).resolve()

    def open_file(name, mode='r', *args, **kwargs):
        if Path(name).resolve() != allowed or mode != 'r':
            raise PermissionError('unbound HDF5 file or write attempt')
        counters['HDF5_open_attempted'] += 1
        require(counters['HDF5_open_attempted'] <= 2, 'HDF5 open cap')
        result = file_call(name, mode, *args, **kwargs)
        counters['HDF5_open_completed'] += 1
        return result

    def group_read(group, name):
        result = group_get(group, name)
        admit_hdf5_path(result.name)
        return result

    def array_read(array, key, *args, **kwargs):
        admit_hdf5_path(array.name)
        require(isinstance(key, slice) and key.step in (None, 1), 'only contiguous causal slices allowed')
        floor = 1 if array.name.endswith('/actions') else 0
        require(key.start is not None and key.stop is not None and key.start >= floor
                and key.stop > key.start, 't0 or malformed numeric slice')
        counters['array_slice_attempted'] += 1
        require(counters['array_slice_attempted'] <= 2160, 'support array decode cap')
        result = array_get(array, key, *args, **kwargs)
        counters['array_slice_completed'] += 1
        return result

    def attribute_read(*args, **kwargs):
        raise PermissionError('simulator/private HDF5 attributes remain sealed')

    h5py.File = open_file
    h5py.Group.__getitem__ = group_read
    h5py.Dataset.__getitem__ = array_read
    h5py.AttributeManager.__getitem__ = attribute_read
    try:
        yield
    finally:
        h5py.File = file_call
        h5py.Group.__getitem__ = group_get
        h5py.Dataset.__getitem__ = array_get
        h5py.AttributeManager.__getitem__ = attribute_get


def fit(policy, items, seed, directory, emit, limits):
    import torch
    from torch.utils.data import default_collate
    from src.algorithms.libero_bc_fomaml import bc_loss
    from experiments.run_libero_formal_transfer import tensor_digest, optimizer_settings
    optimizer = torch.optim.Adam(policy.parameters(), lr=.0003, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    rng = torch.Generator().manual_seed(seed + 7 * 10000 + 777)
    sampling = hashlib.sha256()
    snapshots = {}
    losses = []
    for update in range(501):
        if update in (0, 50, 150, 300, 500):
            state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
            path = directory / f'task7-seed{seed}-A_original-u{update}.pt'
            with path.open('xb') as handle:
                torch.save(state, handle)
            snapshots[str(update)] = {**pin(path), 'weight_hash': tensor_digest(state)}
        if update == 500:
            break
        limits()
        indices = torch.randint(len(items), (8,), generator=rng).tolist()
        chosen = [items[index] for index in indices]
        ids = [(int(item['demonstration_index']), int(item['start_index'])) for item in chosen]
        sampling.update(json.dumps(ids).encode())
        batch = {key: value.to('cuda:0') if torch.is_tensor(value) else value
                 for key, value in default_collate(chosen).items()}
        optimizer.zero_grad(set_to_none=True)
        loss = bc_loss(policy, batch)
        require(torch.isfinite(loss).item(), 'nonfinite Policy loss')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()
        record = {'seed': seed, 'update': update + 1, 'loss': float(loss.detach()),
                  'gradient_norm': float(norm), 'valid_rows': int(batch['valid_mask'].sum())}
        losses.append(record)
        emit('policy_update', **record)
    steps = [int(state['step'].item()) for state in optimizer.state.values()]
    require(steps and set(steps) == {500}, 'actual Adam states not at500')
    policy.eval().requires_grad_(False)
    policy.zero_grad(set_to_none=True)
    return {'curve': snapshots, 'steps': losses, 'sampling_sha256': sampling.hexdigest(),
            'actual_optimizer': optimizer_settings(optimizer), 'actual_state_steps': steps}


def run(config=CONFIG, binding=BINDING):
    cfg, bound = preflight(config, binding)
    directory = Path(cfg['artifacts'])
    require(not directory.exists(), 'output already exists; no overwrite or automatic retry')
    gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True)
    line = next(line for line in gpu.splitlines() if line.startswith('1,'))
    fields = [value.strip() for value in line.split(',')]
    require(fields[1] == cfg['GPU_uuid'] and int(fields[2]) <= 512 and int(fields[3]) <= 5,
            'GPU1 unavailable; stop before payload')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    require(cfg['GPU_uuid'] not in processes, 'another GPU1 compute process is active')
    directory.mkdir(parents=True)
    started = time.monotonic()
    events = (directory / 'events.jsonl').open('x')
    def emit(kind, **values):
        events.write(json.dumps({'event': kind, 'at_UTC': datetime.now(timezone.utc).isoformat(), **values}, allow_nan=False) + '\n')
        events.flush()
    def limits():
        require(time.monotonic() - started <= 900, 'A rerun wall budget exhausted')
    counters = dict(HDF5_open_attempted=0, HDF5_open_completed=0, array_slice_attempted=0, array_slice_completed=0)
    try:
        import numpy as np
        import torch
        from dataclasses import replace
        from torch.utils.data import DataLoader
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy, cuda_pool_equivalence_gate
        from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
        from src.skills.libero_pick_place_spt import LearnableLiberoPickPlaceSPT, build_libero_pick_place_spt, instantiate_libero_pick_place_spi
        from experiments.run_libero_formal_transfer import tensor_digest
        from experiments.run_libero_source_expert_component3_observability import capture_forward, same_tensor_bits
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        require(torch.cuda.device_count() == 1, 'only selected GPU1 may be visible')
        pool_gate = cuda_pool_equivalence_gate()
        inputs = {key: json.loads(Path(ref['path']).read_text()) for key, ref in cfg['metadata_inputs'].items()}
        for ref in cfg['metadata_inputs'].values():
            require(pin(ref['path']) == ref, 'metadata input changed')
        require(sha(cfg['source']['path']) == cfg['source']['sha256'], 'raw source file SHA differs')
        emit('raw_file_integrity_checked', full_file_hash_includes_unprojected_bytes=True)
        with support_guard(cfg['source']['path'], counters):
            dataset = CausalT1LiberoSequenceDataset(inputs['transfer'], inputs['dataset_manifest'],
                Path(cfg['data_root']), task_id=7, demonstration_role='meta_support', sequence_length=10,
                stride=10, access_phase='post_method_freeze', verify_sha256=False)
            try:
                require(len(dataset.windows) == 432, 'support window metadata changed')
                items = [dataset[index] for index in range(len(dataset))]
                access = dataset.access_record
            finally:
                dataset.close()
        require(counters['array_slice_completed'] == 2160 and counters['HDF5_open_completed'] == 2,
                'support access envelope differs')
        cache_bytes = sum(value.numel() * value.element_size() for item in items for value in item.values() if torch.is_tensor(value))
        require(cache_bytes <= 1024**3, 'host tensor cache cap')
        targets = np.concatenate([item['actions'][item['valid_mask']].numpy() for item in items])
        demos = np.concatenate([np.full(int(item['valid_mask'].sum()), int(item['demonstration_index']), np.int32) for item in items])
        steps = np.concatenate([np.arange(int(item['start_index']), int(item['start_index']) + int(item['valid_mask'].sum()), dtype=np.int32) for item in items])
        require(targets.shape == (4188, 7) and set(demos.tolist()) == set(range(30)) and int(steps.min()) == 1,
                'source fit frame identities differ')
        shared = directory / 'task7-shared.npz'
        with shared.open('xb') as handle:
            np.savez(handle, targets=targets, demo_indices=demos, action_steps=steps,
                     demo_lengths=np.asarray(cfg['action_lengths'], np.int32))
        learning = inputs['learning']
        learning['policy'] = dict(learning['policy'], memory_mode='cnn_only', policy_family='dual_view_shared_cnn_only_uniform_bc_formal_v2')
        policy_cfg = LiberoBCPolicyConfig.from_mapping(learning['policy'])
        models = []
        for seed in cfg['seeds']:
            limits()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            template = make_deterministic_libero_policy(policy_cfg)
            generator = LearnableLiberoPickPlaceSPT(build_libero_pick_place_spt(inputs['family'], learning), template).to('cuda:0')
            generator.record = replace(generator.record, version=f'formal-v2-G-{seed}-A_original')
            spi, _ = instantiate_libero_pick_place_spi(generator.record, inputs['family'], learning, 7)
            policy = generator.instantiate(spi).policy.to('cuda:0').train()
            before = tensor_digest(generator.state_dict())
            trained = fit(policy, items, seed, directory, emit, limits)
            require(tensor_digest(generator.state_dict()) == before, 'A generator mutated')
            expected = next(row for row in cfg['historical_endpoints'] if row['seed'] == seed)
            parity = trained['curve']['500']['weight_hash'] == expected['checkpoint']['weight_hash']
            emit('historical_tensor_hash_comparison', seed=seed, matches=parity,
                 observed=trained['curve']['500']['weight_hash'], expected=expected['checkpoint']['weight_hash'])
            features = []; predictions = []; logits = []; main_count = 0
            for number, batch in enumerate(DataLoader(items, batch_size=32, shuffle=False, num_workers=0)):
                limits()
                gpu_batch = {key: value.to('cuda:0') for key, value in batch.items()}
                with torch.inference_mode():
                    pred, hidden, raw = capture_forward(policy, [gpu_batch[key] for key in ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states')])
                    main_count += 1
                    mask = gpu_batch['valid_mask'].bool()
                    features.append(hidden[mask].cpu().numpy().copy())
                    predictions.append(pred[mask].cpu().numpy().copy())
                    logits.append(raw[mask].cpu().numpy().copy())
                    if number == 0:
                        again, h_again, z_again = capture_forward(policy, [gpu_batch[key] for key in ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states')])
                        require(same_tensor_bits(pred, again) and same_tensor_bits(hidden, h_again)
                                and same_tensor_bits(raw, z_again), 'first-batch repeat differs')
            require(main_count == 14, 'feature forward budget differs')
            require(tensor_digest(policy.state_dict()) == trained['curve']['500']['weight_hash'], 'capture mutated Policy')
            feature_path = directory / f'task7-seed{seed}-A_original-features.npz'
            arrays = dict(features=np.concatenate(features), predictions=np.concatenate(predictions), logits=np.concatenate(logits))
            require(arrays['features'].shape == (4188,256), 'feature shape differs')
            with feature_path.open('xb') as handle:
                np.savez(handle, **arrays)
            models.append({'task': 7, 'seed': seed, 'arm': 'A_original', 'training': trained,
                           'historical_tensor_hash_matches': parity, 'historical_checkpoint': expected['checkpoint'],
                           'features': pin(feature_path), 'main_capture_forwards': main_count, 'repeat_forwards': 1})
            emit('model_complete', seed=seed, historical_tensor_hash_matches=parity)
            del policy, generator, template
            torch.cuda.empty_cache()
        record = {'status': 'complete_new_A_rerun', 'formal_result': False, 'historical_restoration': False,
                  'science': pin(config), 'binding': pin(binding), 'models': models, 'shared': pin(shared),
                  'actual_policy_updates': sum(len(row['training']['steps']) for row in models),
                  'main_capture_forwards': sum(row['main_capture_forwards'] for row in models),
                  'repeat_forwards': len(models), 'source_outer_updates': 0, 'source_meta_query_numeric_reads': 0,
                  'sealed_evaluation_reads': 0, 'environment_calls': 0, 'HDF5_access': counters, 'dataset_access': access,
                  'runtime': {'seconds': time.monotonic()-started, 'device': 'cuda:0 (physical GPU1)',
                              'gpu_uuid': cfg['GPU_uuid'], 'versions': cfg['versions'], 'cache_bytes': cache_bytes,
                              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'pool_gate': pool_gate},
                  'old_missing_files_remain_missing': True}
        require(record['actual_policy_updates'] == 1500 and record['main_capture_forwards'] == 42,
                'completed A rerun budget mismatch')
        write_new(directory / 'rerun-result.json', record)
        return record
    except BaseException as error:
        write_new(directory / 'failure.json', {'status': 'technical_abort_new_A_rerun', 'formal_result': False,
                  'error': repr(error), 'HDF5_access': counters, 'seconds': time.monotonic()-started})
        raise
    finally:
        events.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--binding', type=Path, default=BINDING)
    parser.add_argument('--mode', choices=('preflight', 'run'), default='preflight')
    args = parser.parse_args()
    if args.mode == 'preflight':
        preflight(args.config, args.binding)
        print(json.dumps({'status': 'new_A_binding_preflight_passed', 'payload_reads': 0}))
    else:
        result = run(args.config, args.binding)
        print(json.dumps({'status': result['status'], 'policy_updates': result['actual_policy_updates'],
                          'historical_tensor_hash_matches': [row['historical_tensor_hash_matches'] for row in result['models']]}))
