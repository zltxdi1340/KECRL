"""Versioned support-only training/evaluation; requires new execution bindings.

This research access contract is not an operating-system sandbox. Preparing or
scientifically freezing this code does not open data or authorize execution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

from experiments import libero_restricted_vision_finetune_protocol as p
from experiments.libero_formal_execution import Ledger
from experiments.run_libero_pretrained_vision_control import (
    fit_one, pin, runtime, sha, support_guard, tensor_hash, write_new,
)

ROOT, TASKS, SEEDS, ARMS = p.ROOT, p.TASKS, p.SEEDS, p.ARMS
require = p.require


def preflight(config, binding, stage):
    metadata = json.loads(Path(config).read_text())
    cfg, frozen = p.validate_science(config, metadata['scientific_freeze_path'])
    bound = json.loads(Path(binding).read_text())

    def authorization_loader(item):
        require(pin(item['path']) == item, 'authorization metadata changed')
        return json.loads(Path(item['path']).read_text())

    p.authorize_stage(bound, cfg, pin(config), stage, authorization_loader=authorization_loader)
    require(bound.get('scientific_freeze') == pin(cfg['scientific_freeze_path']), 'scientific binding differs')
    dependencies = bound.get('code_dependencies', [])
    require(len(dependencies) == len({row['path'] for row in dependencies}), 'duplicate execution dependencies')
    require({row['path']:row['sha256'] for row in frozen['metadata_dependencies']}.items()
            <= {row['path']:row['sha256'] for row in dependencies}.items(), 'execution dependency closure incomplete')
    for row in dependencies + bound.get('validation', []):
        require(pin(row['path']) == row, 'execution dependency changed: ' + row['path'])
    require(str(Path(__file__).resolve()) in {row['path'] for row in dependencies}, 'runner not bound')
    for key, value in cfg['runtime_environment'].items():
        require(os.environ.get(key) == value, 'runtime environment differs: ' + key)
    for key, value in cfg['versions'].items():
        require(importlib.metadata.version(key) == value, 'runtime package differs: ' + key)
    require(sys.version.split()[0] == '3.10.22', 'Python runtime differs')
    revision = subprocess.check_output(['git','-C',str(ROOT/'third_party/LIBERO'),'rev-parse','HEAD'],text=True).strip()
    require(revision == cfg['libero_revision'] == '8f1084e3132a39270c3a13ebe37270a43ece2a01', 'LIBERO revision differs')
    return cfg, bound


def model_factory(arm, seed, cfg, device):
    import numpy as np
    import torch
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_restricted_resnet18_policy import make_restricted_resnet18_policy
    require(arm in ARMS, 'unknown backend')
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    return make_restricted_resnet18_policy(LiberoBCPolicyConfig.from_mapping(cfg['policy']),
        cfg['public_weights']['path'], cfg['public_weights']['sha256'], train_layer4=arm == ARMS[1]).to(device)


def describe_mask(model, arm):
    from src.algorithms.libero_restricted_resnet18_policy import parameter_manifest, CONVOLUTION_PARAMETER_NAMES
    record = parameter_manifest(model)
    expected = set(CONVOLUTION_PARAMETER_NAMES) if arm == ARMS[1] else set()
    require(set(record['backbone_names']) == expected, 'backbone trainable mask differs')
    require(record['total'] == p.MASK['total_parameters'] and record['trainable'] ==
            p.MASK['candidate_trainable' if arm == ARMS[1] else 'control_trainable'], 'parameter capacity differs')
    return record


def immutable_hash(model, trainable_names):
    trainable = set(trainable_names)
    return tensor_hash({name:value for name,value in model.state_dict().items() if name not in trainable})


def cache_hash(cache):
    import hashlib
    value = hashlib.sha256()
    for row in cache:
        value.update(tensor_hash(row).encode())
    return value.hexdigest()


def train(config, binding):
    cfg, bound = preflight(config, binding, 'train')
    import torch
    from torch.utils.data import default_collate
    from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
    archive = Path(cfg['artifacts'])
    require(not archive.exists(), 'new output required; no overwrite or resume')
    require(sha(cfg['public_weights']['path']) == cfg['public_weights']['sha256'], 'public weights changed')
    device, gpu = runtime()
    archive.mkdir(parents=True)
    ledger = Ledger(p.TRAIN_CAPS, seconds=cfg['training_wall_seconds_max'])
    models, accesses, caches = [], [], []
    started = datetime.now(timezone.utc).isoformat()
    with (archive/'training-events.jsonl').open('x') as events:
        def event(kind, **values):
            events.write(json.dumps(dict(event=kind, **values),allow_nan=False)+'\n'); events.flush()
        try:
            for item in cfg['metadata'].values():
                require(pin(item['path']) == item, 'source metadata changed')
            metadata = {key:json.loads(Path(item['path']).read_text()) for key,item in cfg['metadata'].items()}
            with support_guard([row['path'] for row in cfg['sources'].values()]) as decode_counts:
                for task in TASKS:
                    source = cfg['sources'][str(task)]
                    require(ledger.call('source_sha',sha,source['path']) == source['sha256'], 'source changed')
                    dataset = CausalT1LiberoSequenceDataset(metadata['transfer'],metadata['dataset_manifest'],
                        Path(cfg['data_root']),task_id=task,demonstration_role='meta_support',sequence_length=10,
                        stride=10,access_phase='post_method_freeze',verify_sha256=False)
                    try:
                        items = [ledger.call('dataset_items',dataset.__getitem__,index) for index in range(len(dataset))]
                        accesses.append({**dataset.access_record,'entry_file_sha256_verified':True,'input_pin':source})
                    finally:
                        dataset.close()
                    require(items, 'empty support task')
                    raw_bytes = sum(value.numel()*value.element_size() for item in items for value in item.values() if torch.is_tensor(value))
                    require(raw_bytes <= p.CACHE['raw_bytes_per_task_max'], 'raw cache exceeds2GiB')
                    cache_model = ledger.call('model_create',model_factory,ARMS[0],SEEDS[0],cfg,device)
                    cache_model.visual_encoder.compute_guard = ledger.call
                    cache, cache_bytes = [], 0
                    for start in range(0,len(items),8):
                        batch = default_collate(items[start:start+8]); features = {}
                        for field,label in [('agentview_rgb','agent'),('eye_in_hand_rgb','hand')]:
                            images = batch[field]; count,length = images.shape[:2]
                            value = cache_model.visual_encoder.encode_prefix(images.reshape(-1,3,128,128).to(device)).cpu()
                            features[label] = value.reshape(count,length,256,8,8).clone()
                        for index in range(count):
                            row = {key:value[index].clone() for key,value in features.items()}
                            cache_bytes += sum(value.numel()*value.element_size() for value in row.values())
                            require(cache_bytes <= p.CACHE['prefix_bytes_per_task_max'], 'prefix cache exceeds2GiB')
                            cache.append(row)
                    digest = cache_hash(cache)
                    caches.append(dict(task=task,items=len(items),raw_bytes=raw_bytes,prefix_bytes=cache_bytes,
                        stage='layer3',shape=[256,8,8],before_sha256=digest))
                    del cache_model
                    for seed in SEEDS:
                        paired = []
                        for arm in ARMS:
                            model = ledger.call('model_create',model_factory,arm,seed,cfg,device)
                            model.visual_encoder.compute_guard = ledger.call
                            mask = describe_mask(model,arm)
                            initial = tensor_hash(model.state_dict())
                            immutable_before = immutable_hash(model,mask['names'])
                            fitted = fit_one(model,items,task=task,seed=seed,arm=arm,device=device,
                                ledger=ledger,event=event,feature_cache=cache)
                            immutable_after = immutable_hash(model,mask['names'])
                            require(immutable_before == immutable_after, 'frozen parameters, RGB or BN mutated')
                            path = archive/f'task{task}-seed{seed}-{arm}-u500.pt'
                            state = {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}
                            ledger.begin('checkpoint')
                            with path.open('xb') as handle: torch.save(state,handle)
                            ledger.finish('checkpoint')
                            row = dict(task=task,seed=seed,arm=arm,initial_state_role='candidate_nonregression',
                                initial_state_indices=list(range(20)),initial_full_state_sha256=initial,parameter_mask=mask,
                                immutable_before=immutable_before,immutable_after=immutable_after,
                                clipped_updates=sum(event['gradient_norm'] > 1. for event in fitted['trace']),
                                checkpoint={**pin(path),'weight_hash':tensor_hash(state)},**fitted)
                            paired.append(row); models.append(row)
                            event('model_complete',task=task,seed=seed,arm=arm)
                            del model; torch.cuda.empty_cache()
                        require(paired[0]['initial_full_state_sha256'] == paired[1]['initial_full_state_sha256']
                                and paired[0]['sampling_sha256'] == paired[1]['sampling_sha256'], 'unpaired initial state or batches')
                    caches[-1]['after_sha256'] = cache_hash(cache)
                    require(caches[-1]['after_sha256'] == digest, 'prefix cache mutated')
                    del cache,items
            require(len(models) == 12 and ledger.counts['optimizer']['completed'] == 6000, 'incomplete training')
            ledger.check_time()
            result = dict(status='training_complete_pending_audit',formal_result=False,science=pin(config),binding=pin(binding),
                models=models,source_access=accesses,hdf5_decode_counts=decode_counts,caches=caches,ledger=ledger.snapshot(),
                gpu=gpu,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0),
                started_at_UTC=started,finished_at_UTC=datetime.now(timezone.utc).isoformat(),versions=cfg['versions'],
                python=sys.version.split()[0],cuda=torch.version.cuda)
            write_new(archive/'training.json',result)
            return result
        except BaseException as error:
            write_new(archive/'training-failure.json',dict(status='technical_failure',error=repr(error),
                ledger=ledger.snapshot(),completed_models=len(models)))
            raise


def evaluate(config, binding):
    cfg, bound = preflight(config,binding,'evaluate')
    import numpy as np
    import torch
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    from experiments.libero_formal_rollout import execute
    from experiments.audit_libero_restricted_vision_finetune import audit_training
    archive = Path(cfg['artifacts']); training_path = archive/'training.json'
    training = audit_training(training_path,cfg,config=config)
    destination = archive/'episodes'
    require(not destination.exists(), 'new evaluation directory required')
    device,gpu = runtime()
    ledger = Ledger(p.EVAL_CAPS,seconds=cfg['evaluation_wall_seconds_max'])
    states,access = {},{}
    started = datetime.now(timezone.utc).isoformat(); destination.mkdir()
    with (archive/'evaluation-events.jsonl').open('x') as events:
        def event(kind,**values):
            events.write(json.dumps(dict(event=kind,**values),allow_nan=False)+'\n'); events.flush()
        try:
            for task in TASKS:
                item = cfg['initial_banks'][str(task)]
                def decoded(kind,indices):
                    (ledger.begin if kind == 'attempted' else ledger.finish)('numeric_rows',len(indices))
                states[task],access[str(task)] = load_initial_state_rows(item['path'],item,list(range(20)),
                    authorized_indices=list(range(20)),on_decode=decoded)
            def loader(row):
                require(pin(row['checkpoint']['path']) == {key:row['checkpoint'][key] for key in ('path','sha256')}, 'endpoint changed')
                model = ledger.call('model_create',model_factory,row['arm'],row['seed'],cfg,device)
                model.load_state_dict(torch.load(row['checkpoint']['path'],map_location='cpu',weights_only=True),strict=True)
                model.visual_encoder.compute_guard = ledger.call
                return model.eval().requires_grad_(False)
            def adapter(task):
                random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022)
                return _build_injected_adapter(suite_name='libero_spatial',task_id=task,seed=20261022,image_size=128,
                    max_episode_steps=300,stabilization_steps=5,initial_states=states[task].copy())
            def predict(model,observation):
                image = lambda name:torch.from_numpy(np.ascontiguousarray(observation[name])).permute(2,0,1).unsqueeze(0).to(device)
                prop = lambda name:torch.from_numpy(np.ascontiguousarray(observation[name])).unsqueeze(0).to(device)
                with torch.inference_mode():
                    output,hidden = model.forward_step(image('agentview_rgb'),image('eye_in_hand_rgb'),prop('joint_states'),prop('gripper_states'))
                require(hidden is None,'unexpected recurrent hidden')
                return output[0].cpu().numpy().copy()
            def weights(model,row):
                require(tensor_hash(model.state_dict()) == row['checkpoint']['weight_hash'] and
                        all(not value.requires_grad and value.grad is None for value in model.parameters()), 'inference changed model')
            rollout = execute(training['models'],model_loader=loader,adapter_factory=adapter,predictor=predict,
                weight_checker=weights,archive=destination,ledger=ledger,event=event,heartbeat=lambda *args,**kwargs:None)
            require(ledger.counts['episode']['completed'] == 240, 'incomplete paired evaluation')
            ledger.check_time()
            result = dict(status='evaluation_complete_pending_audit',formal_result=False,science=pin(config),binding=pin(binding),
                training=pin(training_path),initial_access=access,rollout=rollout,ledger=ledger.snapshot(),gpu=gpu,
                started_at_UTC=started,finished_at_UTC=datetime.now(timezone.utc).isoformat(),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(0))
            write_new(archive/'evaluation.json',result)
            return result
        except BaseException as error:
            write_new(archive/'evaluation-failure.json',dict(status='technical_failure',error=repr(error),ledger=ledger.snapshot()))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--stage',choices=['train','evaluate'],required=True)
    args = parser.parse_args()
    result = (train if args.stage == 'train' else evaluate)(args.config,args.binding)
    print(json.dumps(dict(status=result['status'])))


if __name__ == '__main__':
    main()
