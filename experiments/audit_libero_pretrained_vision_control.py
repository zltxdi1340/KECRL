"""Independent bookkeeping, checkpoint and paired-outcome audit; no environment."""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

TASKS = (4, 7)
SEEDS = (20261106, 20261107, 20261108)
ARMS = ('scratch_cnn', 'frozen_imagenet_resnet18')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def state_sha(state):
    digest = hashlib.sha256()
    for name, value in state.items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def audit_counts(snapshot):
    for name, values in snapshot['counts'].items():
        check(type(values['attempted']) is int and values['attempted'] == values['completed']
              and 0 <= values['completed'] <= snapshot['caps'][name], 'counter incomplete or over cap: '+name)


def verify_public_backbone(state, public, row):
    import torch
    names = {'0':'conv1','1':'bn1','4':'layer1','5':'layer2','6':'layer3','7':'layer4'}
    prefix = 'visual_encoder.backbone.'
    frozen = {name[len(prefix):]:value for name,value in state.items() if name.startswith(prefix)}
    check(frozen and state_sha(frozen) == row['frozen_before'] == row['frozen_after'], 'frozen backbone/BN changed')
    present = set()
    for name, value in frozen.items():
        number, suffix = name.split('.',1)
        check(number in names, 'unexpected backbone module')
        original = names[number]+'.'+suffix
        if original not in public and original.endswith('.num_batches_tracked'):
            # The fixed official f37072fd file predates these serialized BN
            # counters. PyTorch fills them with zero; they must remain zero.
            check(value.dtype == torch.int64 and value.numel() == 1 and int(value) == 0,
                  'default frozen BN counter changed')
            continue
        check(original in public and torch.equal(value,public[original]), 'backbone differs from public weights')
        present.add(original)
    check(present == {name for name in public if not name.startswith('fc.')}, 'incomplete public convolution state')
    check(torch.equal(state['visual_encoder.rgb_mean'],torch.tensor([.485,.456,.406]).view(1,3,1,1)) and
          torch.equal(state['visual_encoder.rgb_std'],torch.tensor([.229,.224,.225]).view(1,3,1,1)), 'RGB normalization changed')


def audit_training(path, cfg, *, config):
    import torch
    path = Path(path)
    check(path.resolve() == (Path(cfg['artifacts'])/'training.json').resolve(), 'training path outside new output')
    result = json.loads(path.read_text())
    check(result['status'] == 'training_complete_pending_audit' and result['formal_result'] is False,
          'incomplete training result')
    check(result['science'] == dict(path=str(Path(config).resolve()),sha256=file_sha(config)), 'training science differs')
    expected = {(task,seed,arm) for task in TASKS for seed in SEEDS for arm in ARMS}
    rows = result['models']; keys = [(row['task'],row['seed'],row['arm']) for row in rows]
    check(len(keys) == len(set(keys)) == 12 and set(keys) == expected, 'missing/duplicate model identities')
    check(file_sha(cfg['public_weights']['path']) == cfg['public_weights']['sha256'], 'public weights changed')
    public = torch.load(cfg['public_weights']['path'], map_location='cpu', weights_only=True)
    for row in rows:
        endpoint = row['checkpoint']
        check(Path(endpoint['path']).resolve().parent == path.parent.resolve() and file_sha(endpoint['path']) == endpoint['sha256'],
              'endpoint path or SHA differs')
        state = torch.load(endpoint['path'],map_location='cpu',weights_only=True)
        check(state_sha(state) == endpoint['weight_hash'], 'endpoint tensor hash differs')
        check(row['actual_optimizer_steps'] and set(row['actual_optimizer_steps']) == {500}, 'Adam state not at500')
        optimizer = row['optimizer_defaults']
        check(optimizer['lr'] == .0003 and optimizer['betas'] == [.9,.999] and optimizer['eps'] == 1e-8
              and optimizer['weight_decay'] == 0 and optimizer['amsgrad'] is False, 'actual Adam settings differ')
        trace = row['trace']; check(len(trace) == 500, 'missing actual update records')
        sampling = hashlib.sha256()
        for index, event in enumerate(trace, 1):
            check(event['update'] == index and math.isfinite(event['loss']) and math.isfinite(event['gradient_norm']), 'invalid optimizer trace')
            check(len(event['minibatch_ids']) == 8 and event['valid_rows'] > 0, 'invalid source batch')
            for demo, start in event['minibatch_ids']:
                check(type(demo) is int and 0 <= demo < 30 and type(start) is int and start >= 1, 'source identity outside support')
            sampling.update(json.dumps(event['minibatch_ids']).encode())
        check(sampling.hexdigest() == row['sampling_sha256'], 'sampling trace digest differs')
        if row['arm'] == ARMS[1]:
            verify_public_backbone(state,public,row)
    for task in TASKS:
        for seed in SEEDS:
            pair = [row for row in rows if row['task'] == task and row['seed'] == seed]
            check(pair[0]['initial_nonvisual_sha256'] == pair[1]['initial_nonvisual_sha256'] and
                  pair[0]['sampling_sha256'] == pair[1]['sampling_sha256'], 'unpaired initialization or batches')
    audit_counts(result['ledger'])
    accesses = result['source_access']
    decodes=result['hdf5_decode_counts']
    check(decodes['opens'] == decodes['opens_completed'] <= 4 and
          decodes['array_slices'] == decodes['array_slices_completed'] <= 5500, 'HDF5 operation incomplete or over cap')
    check(len(accesses) == 2 and {row['task_id'] for row in accesses} == set(TASKS), 'source access tasks differ')
    for row in accesses:
        check(row['demonstration_role'] == 'meta_support' and row['demonstration_indices'] == list(range(30)) and
              row['t0_excluded'] is True and row['simulator_state_access'] is False and
              row['observation_action_alignment'] == 'causal_pre_action_t_ge_1' and
              row['entry_file_sha256_verified'] is True and row['input_pin'] == cfg['sources'][str(row['task_id'])],
              'source numeric scope or integrity record differs')
    for name in ('train_forward','backward','optimizer'):
        check(result['ledger']['counts'][name]['completed'] == 6000, 'wrong full training count: '+name)
    check(result['ledger']['counts']['checkpoint']['completed'] == 12 and
          result['ledger']['counts']['parity_forward']['completed'] == 6, 'checkpoint/parity count differs')
    check(result['ledger']['counts']['source_sha']['completed'] == 2 and
          result['ledger']['counts']['model_create']['completed'] == 14, 'source SHA/model count differs')
    return result


def paired_decision(descriptors):
    expected = {(task,seed,arm,index) for task in TASKS for seed in SEEDS for arm in ARMS for index in range(20)}
    by_key = {}
    for item in descriptors:
        key = (item['task_id'],item['training_seed'],item['arm'],item['initial_state_index'])
        check(key in expected and key not in by_key and type(item['success']) is bool, 'invalid/duplicate episode identity')
        by_key[key] = int(item['success'])
    check(set(by_key) == expected, 'complete240 paired episodes required')
    counts = {(task,seed,arm):sum(by_key[task,seed,arm,index] for index in range(20))
              for task in TASKS for seed in SEEDS for arm in ARMS}
    task_delta = {str(task):Fraction(sum(counts[task,seed,ARMS[1]]-counts[task,seed,ARMS[0]] for seed in SEEDS),60) for task in TASKS}
    seed_delta = {str(seed):Fraction(sum(counts[task,seed,ARMS[1]]-counts[task,seed,ARMS[0]] for task in TASKS),40) for seed in SEEDS}
    delta = sum(task_delta.values())/2
    gates = dict(mean_gain_at_least_ten_percentage_points=delta >= Fraction(1,10),
                 both_task_means_nonregressing=all(value >= 0 for value in task_delta.values()),
                 all_seed_means_nonregressing=all(value >= 0 for value in seed_delta.values()))
    return dict(continue_to_all_source_development=all(gates.values()), gates=gates,
                mean_delta_fraction=str(delta),task_delta_fraction={key:str(value) for key,value in task_delta.items()},
                seed_delta_fraction={key:str(value) for key,value in seed_delta.items()},
                counts=[dict(task=task,seed=seed,arm=arm,successes=value,episodes=20) for (task,seed,arm),value in counts.items()],
                formal_transfer_or_module_admitted=False,stay_in_LIBERO=True)


def audit_evaluation(config):
    cfg = json.loads(Path(config).read_text()); archive = Path(cfg['artifacts'])
    training = audit_training(archive/'training.json',cfg,config=config)
    result = json.loads((archive/'evaluation.json').read_text())
    check(result['status'] == 'evaluation_complete_pending_audit' and result['formal_result'] is False, 'incomplete evaluation')
    check(result['training'] == dict(path=str((archive/'training.json').resolve()),sha256=file_sha(archive/'training.json')), 'evaluation training differs')
    descriptors = result['rollout']['episode_files']; decision = paired_decision(descriptors)
    reset = {}; actions = 0
    model_pins = {(row['task'],row['seed'],row['arm']):row['checkpoint'] for row in training['models']}
    for item in descriptors:
        path = Path(item['path'])
        check(path.resolve().parent == (archive/'episodes').resolve() and file_sha(path) == item['sha256'], 'episode path/SHA differs')
        episode = json.loads(path.read_text()); steps = episode['steps']; trace = episode['stepwise_public_state_trace']
        check(episode['status'] == 'complete' and 1 <= steps <= 300 and len(trace) == steps+1 and
              episode['stabilization_steps'] == 5 and episode['environment_seed'] == 20261022 and
              episode['initial_state_role'] == 'candidate_nonregression', 'episode scope/length differs')
        for field in ('task_id','training_seed','arm','initial_state_index','success','steps','terminal_reason','reset_observation_sha256'):
            check(episode[field] == item[field], 'descriptor differs: '+field)
        key = (item['task_id'],item['initial_state_index']); digest = episode['reset_observation_sha256']
        check(reset.setdefault(key,digest) == digest, 'unpaired reset observations')
        check(episode['success'] == (episode['terminal_reason'] == 'success') and
              (episode['success'] or steps == 300), 'invalid terminal boundary')
        checkpoint = model_pins[episode['task_id'],episode['training_seed'],episode['arm']]
        check(episode['checkpoint_sha256'] == checkpoint['sha256'] and episode['checkpoint_weight_hash'] == checkpoint['weight_hash'], 'wrong rollout model')
        for index, row in enumerate(trace):
            check(row['step'] == index and len(row['joint_state']) == 7 and len(row['gripper_state']) == 2, 'invalid public trace')
            if index:
                check(len(row['action']) == 7 and all(math.isfinite(value) and abs(value) <= 1 for value in row['action']), 'invalid action')
        actions += steps
    audit_counts(result['ledger']); actual = result['ledger']['counts']
    check(actual['episode']['completed'] == actual['reset']['completed'] == 240 and actual['adapter']['completed'] == 12,
          'evaluation identity counts differ')
    check(actual['policy_forward']['completed'] == actual['policy_action']['completed'] == actions and
          actual['native_step']['completed'] == actions+1200 and actual['zero_step']['completed'] == 1200 and
          actual['numeric_rows']['completed'] == 40, 'evaluation work counts differ')
    check(result['rollout']['reset_references'] == 40 and result['rollout']['reset_comparisons'] == 200, 'reset comparison counts differ')
    receipt = dict(status='complete_audited_source_control_development',formal_result=False,decision=decision,
                   training_sha256=file_sha(archive/'training.json'),evaluation_sha256=file_sha(archive/'evaluation.json'))
    with (archive/'completion.json').open('x') as handle:
        json.dump(receipt,handle,indent=2);handle.write('\n')
    return receipt


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--stage',choices=['training','complete'],default='complete')
    parser.add_argument('--binding',type=Path,required=True)
    arguments=parser.parse_args()
    cfg=json.loads(arguments.config.read_text());bound=json.loads(arguments.binding.read_text())
    check(bound['science'] == dict(path=str(arguments.config.resolve()),sha256=file_sha(arguments.config)), 'audit science binding differs')
    for dependency in bound['code_dependencies']:
        check(file_sha(dependency['path']) == dependency['sha256'], 'audit execution dependency changed')
    check(str(Path(__file__).resolve()) in {item['path'] for item in bound['code_dependencies']}, 'auditor is not bound')
    if arguments.stage=='training':
        result=audit_training(Path(cfg['artifacts'])/'training.json',cfg,config=arguments.config)
        receipt=dict(status='passed_independent_source_control_training_audit',formal_result=False,
                     training_sha256=file_sha(Path(cfg['artifacts'])/'training.json'),models=len(result['models']))
        with (Path(cfg['artifacts'])/'training-audit.json').open('x') as handle:
            json.dump(receipt,handle,indent=2);handle.write('\n')
    else:
        receipt=audit_evaluation(arguments.config)
    print(json.dumps(receipt))
