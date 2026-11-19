"""Independent state/mask, trajectory, pairing and budget audit; no simulator.

This checks recorded work and immutable tensors, not an independent retraining
or replay of native contact predicates. It never changes an old receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments import libero_restricted_vision_finetune_protocol as p
from experiments.audit_libero_pretrained_vision_control import audit_counts, file_sha, state_sha

check = p.require
CONVOLUTIONS = {
    'visual_encoder.backbone.7.0.conv1.weight', 'visual_encoder.backbone.7.0.conv2.weight',
    'visual_encoder.backbone.7.0.downsample.0.weight', 'visual_encoder.backbone.7.1.conv1.weight',
    'visual_encoder.backbone.7.1.conv2.weight',
}


def verify_backbone(state, public, row):
    import torch
    names = {'0':'conv1','1':'bn1','4':'layer1','5':'layer2','6':'layer3','7':'layer4'}
    prefix = 'visual_encoder.backbone.'
    candidate = row['arm'] == p.ARMS[1]
    permitted = CONVOLUTIONS if candidate else set()
    mask = row['parameter_mask']
    expected_trainable = {name for name in state if not name.startswith('visual_encoder.backbone.')
                          and name not in ('visual_encoder.rgb_mean','visual_encoder.rgb_std')} | permitted
    check(set(mask['names']) == expected_trainable and len(mask['names']) == len(expected_trainable)
          and set(mask['backbone_names']) == permitted, 'actual parameter mask differs')
    trainable = sum(state[name].numel() for name in expected_trainable)
    expected_count = p.MASK['candidate_trainable' if candidate else 'control_trainable']
    check(trainable == mask['trainable'] == row['trainable_parameters'] == expected_count
          and mask['total'] == p.MASK['total_parameters'], 'trainable parameter count differs')
    immutable = {name:value for name,value in state.items() if name not in expected_trainable}
    check(state_sha(immutable) == row['immutable_before'] == row['immutable_after'], 'immutable state changed')
    present = set()
    for name,value in state.items():
        check(not value.is_floating_point() or torch.isfinite(value).all(), 'nonfinite endpoint')
        if not name.startswith(prefix):
            continue
        number,suffix = name[len(prefix):].split('.',1)
        check(number in names, 'unexpected backbone module')
        original = names[number]+'.'+suffix
        if original not in public and original.endswith('.num_batches_tracked'):
            check(value.dtype == torch.int64 and value.numel() == 1 and int(value) == 0, 'default BN counter changed')
            continue
        check(original in public and value.shape == public[original].shape and value.dtype == public[original].dtype,
              'backbone shape/dtype differs')
        if name not in permitted:
            check(torch.equal(value,public[original]), 'frozen convolution or BN differs from public weights')
        present.add(original)
    check(present == {name for name in public if not name.startswith('fc.')}, 'incomplete public backbone')
    check(torch.equal(state['visual_encoder.rgb_mean'],torch.tensor([.485,.456,.406]).view(1,3,1,1))
          and torch.equal(state['visual_encoder.rgb_std'],torch.tensor([.229,.224,.225]).view(1,3,1,1)), 'RGB normalization differs')


def audit_ledger(snapshot, caps):
    check(snapshot['caps'] == caps and set(snapshot['counts']) == set(caps), 'ledger scope differs')
    audit_counts(snapshot)
    check(math.isfinite(snapshot['elapsed_seconds']) and snapshot['elapsed_seconds'] >= 0, 'invalid elapsed time')


def verify_initial_access(access, cfg):
    check(set(access) == {'4','7'}, 'initial access task set differs')
    for task,row in access.items():
        bank = cfg['initial_banks'][task]
        check(row['numeric_rows_decoded'] == list(range(20)) and row['skipped_rows_numerically_decoded'] == []
              and row['skipped_rows_buffered'] is False and row['used_pickle_or_torch_deserialization'] is False
              and row['global_to_local_index'] == {str(i):i for i in range(20)}
              and row['decoded_selected_bytes'] == 20*92*8, 'initial numeric decode scope differs')
        check(row['source_path'] == bank['path'] and row['container_sha256'] == bank['sha256']
              and row['pickle_header_sha256'] == bank['pickle_header_sha256']
              and row['checksum_reads_entire_container_bytes'] is True, 'initial provenance differs')
        check(set(row['selected_row_sha256']) == set(map(str,range(20)))
              and all(isinstance(value,str) and len(value) == 64 for value in row['selected_row_sha256'].values()),
              'initial selected row identity missing')


def audit_training(path, cfg, *, config):
    import torch
    path = Path(path)
    check(path.resolve() == (Path(cfg['artifacts'])/'training.json').resolve(), 'training path differs')
    result = json.loads(path.read_text())
    check(result['status'] == 'training_complete_pending_audit' and result['formal_result'] is False, 'incomplete training')
    check(result['science'] == dict(path=str(Path(config).resolve()),sha256=file_sha(config)), 'training science differs')
    expected = {(t,s,a) for t in p.TASKS for s in p.SEEDS for a in p.ARMS}
    rows = result['models']; keys = [(row['task'],row['seed'],row['arm']) for row in rows]
    check(len(keys) == len(set(keys)) == 12 and set(keys) == expected, 'missing/duplicate model identities')
    check(file_sha(cfg['public_weights']['path']) == cfg['public_weights']['sha256'], 'public weights changed')
    public = torch.load(cfg['public_weights']['path'],map_location='cpu',weights_only=True)
    for row in rows:
        endpoint = row['checkpoint']
        check(Path(endpoint['path']).resolve().parent == path.parent.resolve() and file_sha(endpoint['path']) == endpoint['sha256'], 'endpoint path/SHA differs')
        state = torch.load(endpoint['path'],map_location='cpu',weights_only=True)
        check(state_sha(state) == endpoint['weight_hash'], 'endpoint tensor hash differs')
        verify_backbone(state,public,row)
        check(len(row['actual_optimizer_steps']) == len(row['parameter_mask']['names'])
              and set(row['actual_optimizer_steps']) == {500}, 'Adam states not exactly at500')
        opt = row['optimizer_defaults']
        check(opt['lr'] == .0003 and opt['betas'] == [.9,.999] and opt['eps'] == 1e-8
              and opt['weight_decay'] == 0 and opt['amsgrad'] is False, 'actual Adam differs')
        trace = row['trace']; check(len(trace) == 500, 'missing update records')
        sampling = hashlib.sha256(); clipped = 0
        for index,event in enumerate(trace,1):
            check(event['update'] == index and math.isfinite(event['loss']) and event['loss'] >= 0
                  and math.isfinite(event['gradient_norm']) and event['gradient_norm'] >= 0, 'invalid update trace')
            check(len(event['minibatch_ids']) == 8 and type(event['valid_rows']) is int and 0 < event['valid_rows'] <= 80,
                  'invalid batch size or mask')
            for demo,start in event['minibatch_ids']:
                check(type(demo) is int and 0 <= demo < 30 and type(start) is int and start >= 1 and (start-1)%10 == 0,
                      'source batch outside causal support stride')
            sampling.update(json.dumps(event['minibatch_ids']).encode()); clipped += event['gradient_norm'] > 1.
        check(row['sampling_sha256'] == sampling.hexdigest() and row['clipped_updates'] == clipped, 'batch digest or clip count differs')
        check(row['initial_state_role'] == 'candidate_nonregression' and row['initial_state_indices'] == list(range(20)), 'future state scope differs')
    for task in p.TASKS:
        for seed in p.SEEDS:
            pair = [row for row in rows if row['task'] == task and row['seed'] == seed]
            check(pair[0]['initial_full_state_sha256'] == pair[1]['initial_full_state_sha256']
                  and pair[0]['sampling_sha256'] == pair[1]['sampling_sha256'], 'unpaired initial states or batches')
    accesses = result['source_access']; decodes = result['hdf5_decode_counts']
    check(decodes['opens'] == decodes['opens_completed'] <= 4 and decodes['array_slices'] == decodes['array_slices_completed'] <= 5500,
          'HDF5 operation incomplete or over cap')
    check(len(accesses) == 2 and {row['task_id'] for row in accesses} == set(p.TASKS), 'source tasks differ')
    for row in accesses:
        check(row['demonstration_role'] == 'meta_support' and row['demonstration_indices'] == list(range(30))
              and row['t0_excluded'] is True and row['simulator_state_access'] is False
              and row['observation_action_alignment'] == 'causal_pre_action_t_ge_1'
              and row['entry_file_sha256_verified'] is True and row['input_pin'] == cfg['sources'][str(row['task_id'])], 'support scope differs')
    caches = result['caches']
    check(len(caches) == 2 and {row['task'] for row in caches} == set(p.TASKS), 'cache task set differs')
    for row in caches:
        check(row['stage'] == 'layer3' and row['shape'] == [256,8,8] and 0 < row['items'] <= 1100
              and 0 < row['raw_bytes'] <= p.CACHE['raw_bytes_per_task_max']
              and 0 < row['prefix_bytes'] <= p.CACHE['prefix_bytes_per_task_max']
              and row['before_sha256'] == row['after_sha256'], 'cache scope/immutability differs')
    audit_ledger(result['ledger'],p.TRAIN_CAPS); actual = result['ledger']['counts']
    for name,count in dict(train_forward=6000,backward=6000,optimizer=6000,checkpoint=12,
                           parity_forward=12,source_sha=2,model_create=14,layer4_forward=12024,
                           dataset_items=sum(row['items'] for row in caches)).items():
        check(actual[name]['completed'] == count, 'training count differs: '+name)
    expected_prefix = sum(2*((row['items']+7)//8) for row in caches)+24
    check(actual['prefix_forward']['completed'] == expected_prefix, 'prefix count differs')
    check(result['ledger']['elapsed_seconds'] <= cfg['training_wall_seconds_max'], 'training wall budget exceeded')
    return result


def audit_evaluation(config):
    cfg = json.loads(Path(config).read_text()); archive = Path(cfg['artifacts'])
    training = audit_training(archive/'training.json',cfg,config=config)
    result = json.loads((archive/'evaluation.json').read_text())
    check(result['status'] == 'evaluation_complete_pending_audit' and result['formal_result'] is False
          and result['science'] == dict(path=str(Path(config).resolve()),sha256=file_sha(config)), 'evaluation incomplete/science differs')
    check(result['training'] == dict(path=str((archive/'training.json').resolve()),sha256=file_sha(archive/'training.json')), 'training SHA differs')
    verify_initial_access(result['initial_access'],cfg)
    descriptors = result['rollout']['episode_files']; decision = p.paired_decision(descriptors)
    reset = {}; actions = 0
    endpoints = {(row['task'],row['seed'],row['arm']):row['checkpoint'] for row in training['models']}
    for item in descriptors:
        path = Path(item['path'])
        check(path.resolve().parent == (archive/'episodes').resolve() and file_sha(path) == item['sha256'], 'episode path/SHA differs')
        episode = json.loads(path.read_text()); steps = episode['steps']; trace = episode['stepwise_public_state_trace']
        check(episode['status'] == 'complete' and 1 <= steps <= 300 and len(trace) == steps+1
              and episode['stabilization_steps'] == 5 and episode['environment_seed'] == 20261022
              and episode['initial_state_role'] == 'candidate_nonregression', 'episode scope/length differs')
        for field in ('task_id','training_seed','arm','initial_state_index','success','steps','terminal_reason','reset_observation_sha256'):
            check(episode[field] == item[field], 'descriptor differs: '+field)
        key = (item['task_id'],item['initial_state_index']); digest = episode['reset_observation_sha256']
        check(reset.setdefault(key,digest) == digest, 'unpaired reset observations')
        check(episode['success'] == (episode['terminal_reason'] == 'success') and (episode['success'] or steps == 300), 'terminal boundary differs')
        checkpoint = endpoints[episode['task_id'],episode['training_seed'],episode['arm']]
        check(episode['checkpoint_sha256'] == checkpoint['sha256'] and episode['checkpoint_weight_hash'] == checkpoint['weight_hash'], 'wrong model')
        for index,row in enumerate(trace):
            check(row['step'] == index and len(row['joint_state']) == 7 and len(row['gripper_state']) == 2
                  and all(math.isfinite(v) for v in row['joint_state']+row['gripper_state']), 'invalid public trace')
            if index:
                check(len(row['action']) == 7 and all(math.isfinite(v) and abs(v) <= 1 for v in row['action']), 'invalid action')
        actions += steps
    audit_ledger(result['ledger'],p.EVAL_CAPS); actual = result['ledger']['counts']
    expected = dict(episode=240,reset=240,adapter=12,model_create=12,policy_forward=actions,
        policy_action=actions,native_step=actions+1200,zero_step=1200,numeric_rows=40,
        prefix_forward=2*actions,layer4_forward=2*actions)
    for name,count in expected.items():
        check(actual[name]['completed'] == count, 'evaluation count differs: '+name)
    check(result['rollout']['reset_references'] == 40 and result['rollout']['reset_comparisons'] == 200, 'reset count differs')
    check(result['ledger']['elapsed_seconds'] <= cfg['evaluation_wall_seconds_max'], 'evaluation wall budget exceeded')
    receipt = dict(status='complete_audited_restricted_visual_source_development',formal_result=False,decision=decision,
        training_sha256=file_sha(archive/'training.json'),evaluation_sha256=file_sha(archive/'evaluation.json'))
    from experiments.run_libero_pretrained_vision_control import write_new
    write_new(archive/'completion.json',receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True); parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--stage',choices=['training','complete'],required=True)
    args = parser.parse_args()
    from experiments.run_libero_restricted_vision_finetune import preflight
    cfg,bound = preflight(args.config,args.binding,'train' if args.stage == 'training' else 'evaluate')
    if args.stage == 'training':
        result = audit_training(Path(cfg['artifacts'])/'training.json',cfg,config=args.config)
        receipt = dict(status='passed_independent_restricted_visual_training_audit',formal_result=False,
            training_sha256=file_sha(Path(cfg['artifacts'])/'training.json'),models=len(result['models']))
        from experiments.run_libero_pretrained_vision_control import write_new
        write_new(Path(cfg['artifacts'])/'training-audit.json',receipt)
    else:
        receipt = audit_evaluation(args.config)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
