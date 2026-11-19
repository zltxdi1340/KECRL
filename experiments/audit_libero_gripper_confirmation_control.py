"""Independent scalar filter, baseline replay, paired outcomes and cost audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from experiments import libero_gripper_confirmation_protocol as p


def audit_result(cfg,result,*,source_loader,episode_loader):
    p.require(result['status']=='confirmation_rollouts_complete_pending_audit' and result['training_updates']==0,
              'complete untrained control required')
    sources={};bytes_read=0
    for item in cfg['references']:
        sources[item['training_seed'],item['arm'],item['initial_state_index']]=source_loader(item)
        bytes_read+=Path(item['path']).stat().st_size
    item=result['initial_access'];p.require(p.pin(item['path'])==item,'initial receipt changed')
    access=json.loads(Path(item['path']).read_text());p.check_access(access,cfg)
    expected=[(m['seed'],m['arm'],c,i) for m in cfg['models'] for c in p.CONDITIONS for i in range(20)]
    rows=result['episodes'];p.require([(r['seed'],r['policy_arm'],r['condition'],r['initial_state_index']) for r in rows]==expected,
                                   'paired control coverage/order differs')
    checkpoints={(r['seed'],r['arm']):r['checkpoint'] for r in cfg['models']}
    steps=0;checks=0;masked_total=0;outcomes={};cell_counts={};paired=dict(both_success=0,unchanged_only=0,confirm5_only=0,both_failure=0)
    base=Path(cfg['output']);expected_names=set()
    for descriptor in rows:
        seed,arm,condition,index=(descriptor[k] for k in ('seed','policy_arm','condition','initial_state_index'))
        name=f'{seed}-{arm}-{condition}-{index:02d}.json';expected_names.add(name)
        p.require(Path(descriptor['path'])==base/'episodes'/name and p.pin(descriptor['path'])=={k:descriptor[k] for k in ('path','sha256')}, 'episode identity/hash differs')
        episode=episode_loader(descriptor);source=sources[seed,arm,index]
        p.require(episode['status']=='complete' and episode['task']==7 and episode['seed']==seed
            and episode['policy_arm']==arm and episode['condition']==condition and episode['initial_state_index']==index
            and episode['initial_role']==cfg['initial_role'] and episode['stabilization_steps']==5
            and episode['checkpoint']==checkpoints[seed,arm] and type(episode['success']) is bool
            and 1<=episode['steps']<=300 and episode['terminal_reason']==('success' if episode['success'] else 'time_limit')
            and (episode['success'] or episode['steps']==300), 'episode protocol/terminal differs')
        p.require(episode['reset_observation_sha256']==source['reset_observation_sha256'], 'paired reset differs')
        trace=episode['trace'];p.require(len(trace)==episode['steps']+1 and [r['step'] for r in trace]==list(range(len(trace))), 'trace coverage differs')
        for field,n in [('joint_state',7),('gripper_state',2)]:
            p.require(len(trace[0][field])==n and trace[0][field]==source['stepwise_public_state_trace'][0][field], 'reset public state differs')
        run=0;masked=0
        for i,row in enumerate(trace[1:],1):
            raw=row['raw_action'];applied=row['applied_action']
            p.require(len(raw)==len(applied)==7 and all(math.isfinite(x) and abs(x)<=1 for x in raw+applied), 'invalid action values')
            run=min(6,run+1) if raw[6]>0 else 0
            should_hold=condition=='confirm5' and 0<run<=5
            desired=raw[:6]+[0. if should_hold else raw[6]]
            p.require(applied==desired and row['positive_run']==run and type(row['masked']) is bool
                and row['masked']==should_hold, 'independent confirmation rule differs')
            masked+=int(should_hold);checks+=7
            for field,n in [('joint_state',7),('gripper_state',2)]:
                p.require(len(row[field])==n and all(math.isfinite(x) for x in row[field]), 'invalid public state')
            if condition=='unchanged':
                original=source['stepwise_public_state_trace'][i]
                p.require(raw==original['action'] and row['joint_state']==original['joint_state']
                    and row['gripper_state']==original['gripper_state'], 'baseline original trace differs');checks+=16
        p.require(episode['masked_steps']==masked and descriptor['masked_steps']==masked
            and descriptor['success']==episode['success'] and descriptor['steps']==episode['steps'], 'episode scalar summary differs')
        if condition=='unchanged':p.require(episode['steps']==source['steps'] and episode['success']==source['success'], 'original baseline outcome differs')
        steps+=episode['steps'];masked_total+=masked
        outcomes[seed,arm,condition,index]=episode['success']
    p.require({path.name for path in (base/'episodes').glob('*.json')}==expected_names,'unexpected episode files')
    cells=[]
    for seed in p.SEEDS:
        for arm in p.ARMS:
            u=sum(outcomes[seed,arm,'unchanged',i] for i in range(20));c=sum(outcomes[seed,arm,'confirm5',i] for i in range(20))
            cells.append(dict(seed=seed,policy_arm=arm,n=20,unchanged=u,confirm5=c))
            for i in range(20):
                a,b=outcomes[seed,arm,'unchanged',i],outcomes[seed,arm,'confirm5',i]
                paired['both_success' if a and b else 'unchanged_only' if a else 'confirm5_only' if b else 'both_failure']+=1
    unchanged=sum(r['unchanged'] for r in cells);confirmed=sum(r['confirm5'] for r in cells)
    gates=dict(overall_gain_at_least10pp=confirmed-unchanged>=12,
        each_policy_arm_nonregressing=all(sum(r['confirm5']-r['unchanged'] for r in cells if r['policy_arm']==a)>=0 for a in p.ARMS),
        each_seed_nonregressing=all(sum(r['confirm5']-r['unchanged'] for r in cells if r['seed']==s)>=0 for s in p.SEEDS))
    decision=dict(cells=cells,n_per_condition=120,unchanged_successes=unchanged,confirm5_successes=confirmed,
        gain_fraction=(confirmed-unchanged)/120,paired=paired,gates=gates,continue_this_candidate=all(gates.values()),
        formal_transfer_admitted=False,gripper_timing_universal_cause_proven=False)
    p.require(result['decision']==decision,'independent paired decision differs')
    ledger=result['ledger'];p.require(ledger['caps']==p.CAPS and ledger['elapsed_seconds']<=3600,'control cost envelope differs')
    fixed=dict(model_create=6,checkpoint_sha=6,checkpoint_load=6,model_weight_checks=18,adapters=12,resets=240,episodes=240,
        policy_forward=steps,policy_actions=steps,prefix_forward=2*steps,layer4_forward=2*steps,
        zero_steps=1200,native_steps=steps+1200,numeric_rows=20,initial_row_loads=1,
        source_episode_parses=120,source_episode_bytes=bytes_read,backwards=0,optimizer_updates=0,dataset_reads=0,
        public_initialization_weight_reads=0,checkpoint_writes=0,extra_render_calls=0,failure_observations=0)
    for key,count in fixed.items():
        p.require(ledger['counts'][key]==dict(attempted=count,completed=count) and count<=p.CAPS[key], 'actual control counter differs: '+key);checks+=1
    expected_weights=[dict(seed=m['seed'],policy_arm=m['arm'],weight_hash=m['checkpoint']['weight_hash']) for m in cfg['models'] for _ in range(3)]
    p.require(result['weight_checks']==expected_weights,'before/after model weight checks differ')
    return dict(status='passed_independent_gripper_confirmation_audit',formal_result=False,
        episodes=240,checks=checks,decision=decision,total_policy_steps=steps,masked_steps=masked_total,
        baseline_original_reproductions=120,models_unchanged=6,new_policy_forwards=0,new_native_steps=0,
        checkpoint_reads=0,initial_bank_reads=0,source_episode_parses=120,source_episode_bytes=bytes_read,
        result_episode_parses=240,sole_failure_cause_identified=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();cfg,_=p.preflight(args.config,args.binding);base=Path(cfg['output'])
    execution=json.loads((base/'execution.json').read_text())
    p.require(execution['science']==p.pin(args.config) and execution['binding']==p.pin(args.binding)
        and execution['result']==p.pin(base/'result.json'),'control execution provenance differs')
    result=json.loads((base/'result.json').read_text())
    def loader(item):
        raw=Path(item['path']).read_bytes();p.require(hashlib.sha256(raw).hexdigest()==item['sha256'],'audit JSON source changed')
        return json.loads(raw)
    try:
        receipt=audit_result(cfg,result,source_loader=loader,episode_loader=loader)
        receipt.update(science=p.pin(args.config),binding=p.pin(args.binding),result=p.pin(base/'result.json'))
        p.write_new(base/'independent-audit.json',receipt)
        p.write_new(base/'completion.json',dict(status='complete_audited_gripper_confirmation_control',formal_result=False,
            result=p.pin(base/'result.json'),audit=p.pin(base/'independent-audit.json'),decision=receipt['decision']))
        print(json.dumps(dict(status=receipt['status'],checks=receipt['checks'],decision=receipt['decision'])))
    except BaseException as error:
        p.write_new(base/'audit-failure.json',dict(status='technical_confirmation_audit_failure',error=repr(error)));raise


if __name__=='__main__':main()
