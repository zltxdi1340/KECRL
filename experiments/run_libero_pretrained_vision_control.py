"""New support-only paired control trial; historical runners remain unchanged.

Research access requires a new scientific configuration and execution binding.
Evaluation additionally requires an explicit source-initial authorization pin.
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
import subprocess
import sys
import time

from experiments.libero_formal_execution import Ledger, require

ROOT = Path(__file__).resolve().parents[1]
TASKS = [4, 7]
SEEDS = [20261106, 20261107, 20261108]
ARMS = ['scratch_cnn', 'frozen_imagenet_resnet18']
POLICY = dict(cnn_channels=[32,64,128],spatial_pool_size=4,visual_embedding_dim=128,
              fusion_dim=256,recurrent_hidden_dim=256,action_dimension=7,memory_mode='cnn_only')
CAPS = dict(train_forward=6000, backward=6000, optimizer=6000, checkpoint=12,
            frozen_encoder=278, parity_forward=6, dataset_items=1100,
            source_sha=2, model_create=14,
            adapter=12, reset=240, episode=240, policy_forward=72000,
            policy_action=72000, zero_step=1200, native_step=73200, numeric_rows=40)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def pin(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, allow_nan=False, indent=2)
        handle.write('\n')


def validate_scope(cfg):
    require(cfg['kind'] == 'libero_pretrained_vision_source_control_development' and cfg['formal_result'] is False,
            'this is a new non-formal development trial')
    require(cfg['tasks'] == TASKS and cfg['seeds'] == SEEDS and cfg['arms'] == ARMS,
            'paired task/seed/arm scope changed')
    require(cfg['updates_per_policy'] == 500 and cfg['stabilization_steps'] == 5 and
            cfg['demonstrations'] == list(range(30)) and cfg['initial_indices'] == list(range(20)) and
            cfg['loss'] == 'uniform_masked_seven_dimensional_mse' and cfg['caps'] == CAPS,
            'fixed budget or data roles changed')
    require(cfg['source_meta_query_allowed'] is False and cfg['formal_evaluation_allowed'] is False,
            'query and formal roles remain sealed')
    require(cfg['runtime_environment'] == dict(CUDA_VISIBLE_DEVICES='1', MUJOCO_EGL_DEVICE_ID='1', MUJOCO_GL='egl',
            OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONHASHSEED='0',
            CUBLAS_WORKSPACE_CONFIG=':4096:8'), 'runtime declaration changed')
    require(cfg['source_role'] == 'meta_support' and cfg['initial_role'] == 'candidate_nonregression',
            'source roles changed')
    require(cfg['policy'] == POLICY, 'base Policy architecture changed')
    require(Path(cfg['artifacts']).resolve().is_relative_to(ROOT / 'results'), 'persistent new output required')
    require(set(cfg['sources']) == {'4', '7'} and set(cfg['initial_banks']) == {'4', '7'}, 'input tasks changed')


def preflight(config, binding, stage):
    cfg = json.loads(Path(config).read_text())
    bound = json.loads(Path(binding).read_text())
    validate_scope(cfg)
    require(cfg['status'] == 'scientific_frozen' and bound['status'] == 'bound_before_research_payload'
            and bound['science'] == pin(config) and bound['synthetic_validation_passed'] is True,
            'new scientific and execution binding required')
    for item in bound['code_dependencies']:
        require(pin(item['path']) == item, 'execution dependency changed: ' + item['path'])
    for item in bound.get('validation',[]):
        require(pin(item['path']) == item, 'bound validation evidence changed')
    require(str(Path(__file__).resolve()) in {item['path'] for item in bound['code_dependencies']}, 'runner not bound')
    if stage == 'evaluate':
        approval = bound.get('source_initial_authorization')
        require(isinstance(approval, dict) and pin(approval['path']) == approval,
                'source initial states remain sealed without explicit authorization')
        permitted = json.loads(Path(approval['path']).read_text())
        require(permitted['user_approved'] is True and permitted['tasks'] == TASKS and
                permitted['indices'] == list(range(20)) and permitted['role'] == 'candidate_nonregression',
                'source initial authorization scope differs')
    for key, value in cfg['runtime_environment'].items():
        require(os.environ.get(key) == value, 'runtime environment differs: ' + key)
    for key, value in cfg['versions'].items():
        require(importlib.metadata.version(key) == value, 'runtime package differs: ' + key)
    require(sys.version.split()[0] == '3.10.22', 'Python runtime differs')
    revision=subprocess.check_output(['git','-C',str(ROOT/'third_party/LIBERO'),'rev-parse','HEAD'],text=True).strip()
    require(revision == cfg['libero_revision'] == '8f1084e3132a39270c3a13ebe37270a43ece2a01', 'LIBERO revision differs')
    return cfg, bound


@contextmanager
def support_guard(paths):
    """Logical HDF5 decode scope; whole-file SHA still covers all bytes."""
    import h5py
    from experiments.run_libero_postmaintenance_task7_A_rebuild import admit_hdf5_path
    old_file, old_get, old_slice, old_attr = h5py.File, h5py.Group.__getitem__, h5py.Dataset.__getitem__, h5py.AttributeManager.__getitem__
    allowed = {str(Path(path).resolve()) for path in paths}
    counts = dict(opens=0, opens_completed=0, array_slices=0, array_slices_completed=0)
    def opening(name, mode='r', *args, **kwargs):
        require(str(Path(name).resolve()) in allowed and mode == 'r', 'unbound HDF5 file or write')
        require(counts['opens'] < 4, 'source HDF5 open cap')
        counts['opens'] += 1
        result=old_file(name, mode, *args, **kwargs)
        counts['opens_completed']+=1
        return result
    def getting(group, name):
        result = old_get(group, name)
        admit_hdf5_path(result.name)
        return result
    def slicing(array, key, *args, **kwargs):
        admit_hdf5_path(array.name)
        require(isinstance(key, slice) and key.step in (None, 1) and key.start is not None
                and key.stop is not None and key.start >= (1 if array.name.endswith('/actions') else 0)
                and key.stop > key.start, 'noncausal or malformed source decode')
        require(counts['array_slices'] < 5500, 'source array decode cap')
        counts['array_slices'] += 1
        result=old_slice(array, key, *args, **kwargs)
        counts['array_slices_completed']+=1
        return result
    def forbidden(*args, **kwargs):
        raise PermissionError('HDF5 attributes and private states remain sealed')
    h5py.File, h5py.Group.__getitem__, h5py.Dataset.__getitem__, h5py.AttributeManager.__getitem__ = opening, getting, slicing, forbidden
    try:
        yield counts
    finally:
        h5py.File, h5py.Group.__getitem__, h5py.Dataset.__getitem__, h5py.AttributeManager.__getitem__ = old_file, old_get, old_slice, old_attr


def model_factory(arm, seed, cfg, device):
    import numpy as np
    import torch
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
    from src.algorithms.libero_frozen_resnet18_policy import make_frozen_resnet18_policy
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    policy_cfg = LiberoBCPolicyConfig.from_mapping(cfg['policy'])
    if arm == ARMS[0]:
        model = make_deterministic_libero_policy(policy_cfg)
    else:
        require(arm == ARMS[1], 'unknown backend')
        model = make_frozen_resnet18_policy(policy_cfg, cfg['public_weights']['path'], cfg['public_weights']['sha256'])
    return model.to(device)


def tensor_hash(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def fit_one(model, items, *, task, seed, arm, device, ledger, event, feature_cache=None):
    """Exactly 500 actual Adam updates; all minibatch IDs are retained."""
    import torch
    from torch.utils.data import default_collate
    from src.algorithms.libero_bc_policy import masked_behavior_cloning_loss
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=.0003, betas=(.9, .999), eps=1e-8, weight_decay=0.)
    rng = torch.Generator().manual_seed(seed + task * 10000 + 777)
    sampling = hashlib.sha256(); trace = []
    moving = lambda batch: {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}
    for update in range(500):
        indices = torch.randint(len(items), (8,), generator=rng).tolist()
        batch = moving(default_collate([items[index] for index in indices]))
        ids = [(int(items[index]['demonstration_index']), int(items[index]['start_index'])) for index in indices]
        sampling.update(json.dumps(ids).encode())
        optimizer.zero_grad(set_to_none=True)
        if feature_cache is None:
            prediction, _ = ledger.call('train_forward', model.forward_sequence, *(batch[key] for key in
                ('agentview_rgb', 'eye_in_hand_rgb', 'joint_states', 'gripper_states')))
        else:
            selected = moving(default_collate([feature_cache[index] for index in indices]))
            prediction, _ = ledger.call('train_forward', model.forward_feature_sequence,
                selected['agent'], selected['hand'], batch['joint_states'], batch['gripper_states'])
            if update == 0:
                with torch.no_grad():
                    direct, _ = ledger.call('parity_forward', model.forward_sequence, *(batch[key] for key in
                        ('agentview_rgb', 'eye_in_hand_rgb', 'joint_states', 'gripper_states')))
                require(float((direct - prediction).abs().max()) <= 1e-5, 'pixel/cache action parity failed')
        loss = masked_behavior_cloning_loss(prediction, batch['actions'], batch['valid_mask'])
        require(torch.isfinite(loss).item(), 'nonfinite loss')
        ledger.call('backward', loss.backward)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        ledger.call('optimizer', optimizer.step)
        row = dict(update=update+1, loss=float(loss.detach()), gradient_norm=float(norm),
                   valid_rows=int(batch['valid_mask'].sum()), minibatch_ids=ids)
        trace.append(row); event('optimizer_update', task=task, seed=seed, arm=arm, **row)
    steps = [int(value['step'].item()) for value in optimizer.state.values()]
    require(len(steps) == len(parameters) and set(steps) == {500}, 'actual optimizer states not at500')
    model.eval().requires_grad_(False); model.zero_grad(set_to_none=True)
    return dict(trace=trace, sampling_sha256=sampling.hexdigest(), optimizer_defaults=optimizer.defaults,
                actual_optimizer_steps=steps, trainable_parameters=sum(parameter.numel() for parameter in parameters))


def runtime():
    import torch
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    idle = idle_gpu(1)
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'selected GPU1 required')
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    return torch.device('cuda:0'), idle


def train(config, binding):
    cfg, bound = preflight(config, binding, 'train')
    import torch
    from torch.utils.data import default_collate
    from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
    archive = Path(cfg['artifacts']); require(not archive.exists(), 'new output required; no overwrite')
    require(sha(cfg['public_weights']['path']) == cfg['public_weights']['sha256'], 'public weights changed')
    device, gpu = runtime(); archive.mkdir(parents=True)
    ledger = Ledger(CAPS, seconds=1800); models = []; accesses = []; started_utc=datetime.now(timezone.utc).isoformat()
    with (archive/'training-events.jsonl').open('x') as events:
        def event(kind, **values):
            events.write(json.dumps(dict(event=kind, **values), allow_nan=False)+'\n'); events.flush()
        try:
            metadata = {key: json.loads(Path(item['path']).read_text()) for key, item in cfg['metadata'].items()}
            for item in cfg['metadata'].values():
                require(pin(item['path']) == item, 'input metadata changed')
            with support_guard([row['path'] for row in cfg['sources'].values()]) as decode_counts:
                for task in TASKS:
                    source = cfg['sources'][str(task)]
                    require(ledger.call('source_sha',sha,source['path']) == source['sha256'], 'source file changed')
                    dataset = CausalT1LiberoSequenceDataset(metadata['transfer'], metadata['dataset_manifest'],
                        Path(cfg['data_root']), task_id=task, demonstration_role='meta_support', sequence_length=10,
                        stride=10, access_phase='post_method_freeze', verify_sha256=False)
                    try:
                        items = [ledger.call('dataset_items', dataset.__getitem__, index) for index in range(len(dataset))]
                        accesses.append({**dataset.access_record,'entry_file_sha256_verified':True,'input_pin':source})
                    finally:
                        dataset.close()
                    raw_bytes = sum(value.numel()*value.element_size() for item in items for value in item.values() if torch.is_tensor(value))
                    require(raw_bytes <= 2*1024**3, 'raw support cache exceeds2GiB')
                    cache_model = ledger.call('model_create',model_factory,ARMS[1],SEEDS[0],cfg,device)
                    cache = []; cache_bytes = 0
                    for start in range(0, len(items), 8):
                        batch = default_collate(items[start:start+8]); features = {}
                        for field, label in [('agentview_rgb','agent'), ('eye_in_hand_rgb','hand')]:
                            images = batch[field]; batch_size, length = images.shape[:2]
                            value = ledger.call('frozen_encoder', cache_model.visual_encoder.encode_frozen,
                                                images.reshape(-1,3,128,128).to(device)).cpu()
                            features[label] = value.reshape(batch_size,length,512,4,4).clone()
                        for index in range(batch_size):
                            row = {key:value[index].clone() for key,value in features.items()}
                            cache_bytes += sum(value.numel()*value.element_size() for value in row.values())
                            require(cache_bytes <= 1024**3, 'frozen support feature cache exceeds1GiB')
                            cache.append(row)
                    del cache_model
                    for seed in SEEDS:
                        paired = []
                        for arm in ARMS:
                            model = ledger.call('model_create',model_factory,arm,seed,cfg,device)
                            initial_nonvisual = tensor_hash({name:value for name,value in model.state_dict().items() if not name.startswith('visual_encoder.')})
                            frozen_before = tensor_hash(model.visual_encoder.backbone.state_dict()) if arm == ARMS[1] else None
                            fitted = fit_one(model, items, task=task, seed=seed, arm=arm, device=device,
                                             ledger=ledger, event=event, feature_cache=cache if arm == ARMS[1] else None)
                            frozen_after = tensor_hash(model.visual_encoder.backbone.state_dict()) if arm == ARMS[1] else None
                            require(frozen_before == frozen_after, 'frozen backbone or BN mutated')
                            path = archive/f'task{task}-seed{seed}-{arm}-u500.pt'
                            ledger.begin('checkpoint')
                            state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
                            with path.open('xb') as handle: torch.save(state, handle)
                            ledger.finish('checkpoint')
                            row = dict(task=task,seed=seed,arm=arm,initial_state_role='candidate_nonregression',initial_state_indices=list(range(20)),
                                initial_nonvisual_sha256=initial_nonvisual,frozen_before=frozen_before,frozen_after=frozen_after,
                                checkpoint={**pin(path), 'weight_hash':tensor_hash(state)}, **fitted)
                            paired.append(row); models.append(row); event('model_complete', task=task,seed=seed,arm=arm)
                            del model; torch.cuda.empty_cache()
                        require(paired[0]['initial_nonvisual_sha256'] == paired[1]['initial_nonvisual_sha256'] and
                                paired[0]['sampling_sha256'] == paired[1]['sampling_sha256'], 'initialization or sampling unpaired')
                    del cache,items
            require(len(models) == 12 and ledger.counts['optimizer']['completed'] == 6000, 'incomplete training')
            result = dict(status='training_complete_pending_audit',formal_result=False,science=pin(config),binding=pin(binding),
                models=models,source_access=accesses,hdf5_decode_counts=decode_counts,ledger=ledger.snapshot(),gpu=gpu,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),started_at_UTC=started_utc,
                finished_at_UTC=datetime.now(timezone.utc).isoformat(),versions=cfg.get('versions',{}),
                python=sys.version.split()[0],cuda=torch.version.cuda,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))
            write_new(archive/'training.json', result)
            return result
        except BaseException as error:
            write_new(archive/'training-failure.json',dict(status='technical_failure',error=repr(error),ledger=ledger.snapshot(),completed_models=len(models)))
            raise


def evaluate(config, binding):
    cfg, bound = preflight(config, binding, 'evaluate')
    import numpy as np
    import torch
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    from experiments.libero_formal_rollout import execute
    from experiments.audit_libero_pretrained_vision_control import audit_training
    archive = Path(cfg['artifacts']); training_path = archive/'training.json'
    training = audit_training(training_path, cfg, config=config)
    destination = archive/'episodes'; require(not destination.exists(), 'new evaluation directory required')
    device, gpu = runtime(); ledger = Ledger(CAPS, seconds=7200); states = {}; access = {}; started_utc=datetime.now(timezone.utc).isoformat()
    destination.mkdir()
    with (archive/'evaluation-events.jsonl').open('x') as events:
        def event(kind, **values):
            events.write(json.dumps(dict(event=kind, **values),allow_nan=False)+'\n');events.flush()
        try:
            for task in TASKS:
                item = cfg['initial_banks'][str(task)]
                def decoded(kind, indices):
                    (ledger.begin if kind == 'attempted' else ledger.finish)('numeric_rows',len(indices))
                states[task], access[str(task)] = load_initial_state_rows(item['path'],item,list(range(20)),
                    authorized_indices=list(range(20)),on_decode=decoded)
            def loader(row):
                require(pin(row['checkpoint']['path']) == {key:row['checkpoint'][key] for key in ('path','sha256')}, 'endpoint changed')
                model = ledger.call('model_create',model_factory,row['arm'],row['seed'],cfg,device)
                model.load_state_dict(torch.load(row['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
                return model.eval().requires_grad_(False)
            def adapter(task):
                random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022)
                return _build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
                    max_episode_steps=300,stabilization_steps=5,initial_states=states[task].copy())
            def predict(model, observation):
                image=lambda name:torch.from_numpy(np.ascontiguousarray(observation[name])).permute(2,0,1).unsqueeze(0).to(device)
                prop=lambda name:torch.from_numpy(np.ascontiguousarray(observation[name])).unsqueeze(0).to(device)
                with torch.inference_mode():
                    output,hidden=model.forward_step(image('agentview_rgb'),image('eye_in_hand_rgb'),prop('joint_states'),prop('gripper_states'))
                require(hidden is None,'unexpected recurrent hidden');return output[0].cpu().numpy().copy()
            def weights(model,row):
                require(tensor_hash(model.state_dict()) == row['checkpoint']['weight_hash'] and
                        all(not p.requires_grad and p.grad is None for p in model.parameters()), 'inference changed model')
            rollout = execute(training['models'],model_loader=loader,adapter_factory=adapter,predictor=predict,
                weight_checker=weights,archive=destination,ledger=ledger,event=event,heartbeat=lambda *args,**kwargs:None)
            require(ledger.counts['episode']['completed'] == 240,'incomplete paired evaluation')
            result = dict(status='evaluation_complete_pending_audit',formal_result=False,science=pin(config),binding=pin(binding),
                          training=pin(training_path),initial_access=access,rollout=rollout,ledger=ledger.snapshot(),gpu=gpu,
                          started_at_UTC=started_utc,finished_at_UTC=datetime.now(timezone.utc).isoformat(),
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0))
            write_new(archive/'evaluation.json', result)
            return result
        except BaseException as error:
            write_new(archive/'evaluation-failure.json',dict(status='technical_failure',error=repr(error),ledger=ledger.snapshot()))
            raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--stage',choices=['train','evaluate'],required=True)
    args=parser.parse_args(); result=(train if args.stage=='train' else evaluate)(args.config,args.binding)
    print(json.dumps(dict(status=result['status'])))


if __name__ == '__main__':
    main()
