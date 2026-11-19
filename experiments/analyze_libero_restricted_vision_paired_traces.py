"""Explore all saved pairs without training, inference, RGB or simulator access."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

from experiments import libero_restricted_vision_paired_trace_protocol as p
from experiments.analyze_libero_source_internal_six_group_paired_traces import stats,event_order


def arrays(ep):
    n=ep['steps'];trace=ep['stepwise_public_state_trace']
    p.require(type(n) is int and 1<=n<=300 and len(trace)==n+1,'invalid trace length')
    p.require([row['step'] for row in trace]==list(range(n+1)),'nonconsecutive public steps')
    a=np.asarray([row['action'] for row in trace[1:]],np.float64)
    q=np.asarray([row['joint_state'] for row in trace],np.float64)
    g=np.asarray([row['gripper_state'] for row in trace],np.float64)
    p.require(a.shape==(n,7) and q.shape==(n+1,7) and g.shape==(n+1,2),'public vector dimensions differ')
    p.require(np.isfinite(a).all() and np.isfinite(q).all() and np.isfinite(g).all() and np.abs(a).max()<=1,
              'invalid public numeric values')
    return a,q,g


def measures(a,q,g):
    """Command differences are not physical acceleration, contact or task phase."""
    n=len(a);norm=lambda x:np.linalg.norm(x,axis=-1)
    successive=lambda x:float(np.mean(norm(np.diff(x,axis=0)))) if n>=2 else None
    second=lambda x:float(np.mean(norm(np.diff(x,n=2,axis=0)))) if n>=3 else None
    sign=np.where(a[:,6]>p.DEADBAND,1,np.where(a[:,6]<-p.DEADBAND,-1,0))
    active=sign[sign!=0];switches=int(np.sum(active[1:]!=active[:-1]))
    close=next((i+1 for i,value in enumerate(sign) if value==1),None)
    opened=next((i+1 for i,value in enumerate(sign) if close is not None and i+1>close and value==-1),None)
    qstep=norm(np.diff(q,axis=0));gstep=norm(np.diff(g,axis=0))
    return dict(translation_norm_mean=float(norm(a[:,:3]).mean()),rotation_norm_mean=float(norm(a[:,3:6]).mean()),
        gripper_abs_mean=float(np.abs(a[:,6]).mean()),joint_step_mean=float(qstep.mean()),gripper_step_mean=float(gstep.mean()),
        joint_path=float(qstep.sum()),gripper_path=float(gstep.sum()),
        translation_saturation_fraction=float(np.mean(np.abs(a[:,:3])>=p.SATURATION)),
        rotation_saturation_fraction=float(np.mean(np.abs(a[:,3:6])>=p.SATURATION)),
        gripper_saturation_fraction=float(np.mean(np.abs(a[:,6])>=p.SATURATION)),
        translation_step_difference=successive(a[:,:3]),rotation_step_difference=successive(a[:,3:6]),
        gripper_step_difference=successive(a[:,6:7]),translation_second_difference=second(a[:,:3]),
        rotation_second_difference=second(a[:,3:6]),gripper_second_difference=second(a[:,6:7]),
        gripper_switches=switches,gripper_non_neutral_steps=int(len(active)),
        gripper_switch_rate=float(switches/(len(active)-1)) if len(active)>1 else None,
        positive_command_fraction=float(np.mean(sign==1)),first_close=close,first_open_after_close=opened)


def pair(left,right):
    p.require(left['arm']==p.ARMS[0] and right['arm']==p.ARMS[1],'pair arm orientation differs')
    for key in ('task_id','training_seed','initial_state_index','reset_observation_sha256'):
        p.require(left[key]==right[key],'pair identity or reset differs: '+key)
    la,lq,lg=arrays(left);ra,rq,rg=arrays(right)
    p.require(np.array_equal(lq[0],rq[0]) and np.array_equal(lg[0],rg[0]),'public reset state differs')
    lf,rf=measures(la,lq,lg),measures(ra,rq,rg)
    label=('both_success' if right['success'] else 'frozen_only') if left['success'] else ('finetune_only' if right['success'] else 'both_failure')
    row=dict(task=left['task_id'],seed=left['training_seed'],initial_state_index=left['initial_state_index'],outcome=label,
        frozen_steps=left['steps'],finetune_steps=right['steps'],common_observed_steps=min(left['steps'],right['steps']),
        first_action_gap=float(np.linalg.norm(la[0]-ra[0])),first_translation_gap=float(np.linalg.norm(la[0,:3]-ra[0,:3])),
        first_rotation_gap=float(np.linalg.norm(la[0,3:6]-ra[0,3:6])),first_gripper_gap=float(abs(la[0,6]-ra[0,6])),
        first_pre_joint_gap=0.,first_pre_gripper_gap=0.,frozen_full=lf,finetune_full=rf,
        first_close_order=event_order(lf['first_close'],rf['first_close'],len(la),len(ra)),
        first_open_after_close_order=event_order(lf['first_open_after_close'],rf['first_open_after_close'],len(la),len(ra)))
    prefixes=[]
    for h in p.HORIZONS:
        item=dict(task=row['task'],seed=row['seed'],initial_state_index=row['initial_state_index'],outcome=label,horizon=h,
                  eligible=len(la)>=h and len(ra)>=h,frozen_available=len(la)>=h,finetune_available=len(ra)>=h,
                  frozen=None,finetune=None,delta=None,gaps=None)
        if item['eligible']:
            frozen,tuned=measures(la[:h],lq[:h+1],lg[:h+1]),measures(ra[:h],rq[:h+1],rg[:h+1])
            item.update(frozen=frozen,finetune=tuned,delta={key:tuned[key]-frozen[key] if frozen[key] is not None and tuned[key] is not None else None for key in frozen},
                gaps=dict(action_gap_mean=float(np.linalg.norm(la[:h]-ra[:h],axis=1).mean()),
                    translation_gap_mean=float(np.linalg.norm(la[:h,:3]-ra[:h,:3],axis=1).mean()),
                    rotation_gap_mean=float(np.linalg.norm(la[:h,3:6]-ra[:h,3:6],axis=1).mean()),
                    gripper_action_gap_mean=float(np.abs(la[:h,6]-ra[:h,6]).mean()),
                    pre_joint_gap=float(np.linalg.norm(lq[h-1]-rq[h-1])),post_joint_gap=float(np.linalg.norm(lq[h]-rq[h])),
                    pre_gripper_gap=float(np.linalg.norm(lg[h-1]-rg[h-1])),post_gripper_gap=float(np.linalg.norm(lg[h]-rg[h])),
                    joint_gap_mean=float(np.linalg.norm(lq[1:h+1]-rq[1:h+1],axis=1).mean()),
                    gripper_gap_mean=float(np.linalg.norm(lg[1:h+1]-rg[1:h+1],axis=1).mean())))
        prefixes.append(item)
    return row,prefixes


def summaries(pairs,prefixes):
    selectors=[{}]+[dict(outcome=o) for o in p.OUTCOMES]+[dict(task=t) for t in p.TASKS]+[dict(seed=s) for s in p.SEEDS]
    selectors += [dict(task=t,seed=s) for t in p.TASKS for s in p.SEEDS]
    selectors += [dict(task=t,outcome=o) for t in p.TASKS for o in p.OUTCOMES]
    selectors += [dict(task=t,seed=s,outcome=o) for t in p.TASKS for s in p.SEEDS for o in p.OUTCOMES]
    groups=[]
    for selector in selectors:
        select=lambda row:all(row[key]==value for key,value in selector.items())
        rows=[row for row in pairs if select(row)]
        group=dict(selector=selector,pairs=len(rows),outcomes={o:sum(row['outcome']==o for row in rows) for o in p.OUTCOMES},
            first_actions={key:stats(row[key] for row in rows) for key in ('first_action_gap','first_translation_gap','first_rotation_gap','first_gripper_gap')},
            close_order=dict(Counter(row['first_close_order'] for row in rows)),prefixes=[])
        for h in p.HORIZONS:
            all_rows=[row for row in prefixes if row['horizon']==h and select(row)]
            eligible=[row for row in all_rows if row['eligible']]
            item=dict(horizon=h,pairs=len(all_rows),eligible=len(eligible),censored=len(all_rows)-len(eligible),frozen={},finetune={},delta={},gaps={})
            if eligible:
                for category in ('frozen','finetune','delta','gaps'):
                    for key in eligible[0][category]:
                        values=[row[category][key] for row in eligible]
                        item[category][key]=stats(values)
                        if category=='delta':
                            item[category][key].update(positive=sum(v is not None and v>p.SIGN_EPS for v in values),negative=sum(v is not None and v<-p.SIGN_EPS for v in values),zero=sum(abs(v)<=p.SIGN_EPS for v in values if v is not None))
            group['prefixes'].append(item)
        groups.append(group)
    return groups


def training_features(training,pairs):
    rows=[];bins=[];indexed={}
    p.require(len(training['models'])==12,'complete12 training models required')
    for row in training['models']:
        key=(row['task'],row['seed'],row['arm'])
        p.require(key not in indexed and key[0] in p.TASKS and key[1] in p.SEEDS and key[2] in p.ARMS,'training identity differs')
        trace=row['trace'];p.require(len(trace)==500 and [v['update'] for v in trace]==list(range(1,501)),'actual update trace differs')
        for event in trace:
            p.require(np.isfinite(event['loss']) and event['loss']>=0 and np.isfinite(event['gradient_norm']) and event['gradient_norm']>=0,'nonfinite training event')
        p.require(set(row['actual_optimizer_steps'])=={500} and row['clipped_updates']==sum(v['gradient_norm']>1 for v in trace),'actual training steps/clip differs')
        item=dict(task=key[0],seed=key[1],arm=key[2],trainable_parameters=row['trainable_parameters'],
            first50_loss=float(np.mean([v['loss'] for v in trace[:50]])),last50_loss=float(np.mean([v['loss'] for v in trace[-50:]])),
            clipped_updates=row['clipped_updates'],sampling_sha256=row['sampling_sha256'],initial_full_state_sha256=row['initial_full_state_sha256'])
        rows.append(item);indexed[key]=(row,item)
        for start in range(0,500,50):
            part=trace[start:start+50]
            bins.append(dict(task=key[0],seed=key[1],arm=key[2],update_first=start+1,update_last=start+50,
                loss_mean=float(np.mean([e['loss'] for e in part])),gradient_norm_mean=float(np.mean([e['gradient_norm'] for e in part])),clipped=sum(e['gradient_norm']>1 for e in part)))
    comparisons=[]
    for t in p.TASKS:
        for s in p.SEEDS:
            frozen,lf=indexed[t,s,p.ARMS[0]];tuned,rf=indexed[t,s,p.ARMS[1]]
            p.require(frozen['initial_full_state_sha256']==tuned['initial_full_state_sha256'] and frozen['sampling_sha256']==tuned['sampling_sha256']
                and all(a['minibatch_ids']==b['minibatch_ids'] and a['valid_rows']==b['valid_rows'] for a,b in zip(frozen['trace'],tuned['trace'])),'training pairing differs')
            selected=[row for row in pairs if row['task']==t and row['seed']==s]
            fs=sum(row['outcome'] in ('both_success','frozen_only') for row in selected)
            ts=sum(row['outcome'] in ('both_success','finetune_only') for row in selected)
            comparisons.append(dict(task=t,seed=s,frozen_last50_loss=lf['last50_loss'],finetune_last50_loss=rf['last50_loss'],
                last50_loss_delta=rf['last50_loss']-lf['last50_loss'],frozen_successes=fs,finetune_successes=ts,
                success_delta=ts-fs,frozen_clipped_updates=lf['clipped_updates'],finetune_clipped_updates=rf['clipped_updates']))
    return rows,bins,comparisons


def load_inputs(cfg):
    reader=p.Sources(cfg)
    values={key:reader.read(item) for key,item in cfg['sources'].items()}
    training,evaluation,completion,audit=(values[key] for key in ('training','evaluation','completion','training_audit'))
    p.require(completion['status']=='complete_audited_restricted_visual_source_development'
        and completion['training_sha256']==cfg['sources']['training']['sha256'] and completion['evaluation_sha256']==cfg['sources']['evaluation']['sha256']
        and audit['status']=='passed_independent_restricted_visual_training_audit' and audit['training_sha256']==cfg['sources']['training']['sha256'], 'completed source audits/provenance required')
    p.require(training['science']==evaluation['science']==cfg['trial_science'],'source scientific identity differs')
    expected={(t,s,a,i) for t in p.TASKS for s in p.SEEDS for a in p.ARMS for i in range(20)}
    endpoints={(row['task'],row['seed'],row['arm']):row['checkpoint'] for row in training['models']}
    index={}
    for descriptor in evaluation['rollout']['episode_files']:
        key=(descriptor['task_id'],descriptor['training_seed'],descriptor['arm'],descriptor['initial_state_index'])
        p.require(all(type(value) is int for value in (key[0],key[1],key[3])) and key in expected and key not in index,
                  'episode identity missing/duplicate/outside scope')
        ep=reader.read({k:descriptor[k] for k in ('path','sha256')},episode=True)
        for field in ('task_id','training_seed','arm','initial_state_index','success','steps','terminal_reason','reset_observation_sha256'):
            p.require(ep[field]==descriptor[field],'episode descriptor differs: '+field)
        p.require(ep['status']=='complete' and type(ep['success']) is bool and ep['stabilization_steps']==5
            and ep['initial_state_role']=='candidate_nonregression' and ep['environment_seed']==20261022
            and ep['terminal_reason'] in ('success','time_limit') and ep['success']==(ep['terminal_reason']=='success')
            and (ep['success'] or ep['steps']==300) and ep['local_initial_state_index']==key[3], 'invalid episode scope/terminal')
        endpoint=endpoints[key[:3]]
        p.require(ep['checkpoint_sha256']==endpoint['sha256'] and ep['checkpoint_weight_hash']==endpoint['weight_hash'],
                  'episode belongs to a different trained model')
        index[key]=ep
    p.require(set(index)==expected and reader.parses==244,'complete240 public episodes required')
    return reader,training,completion,index


def compute(cfg):
    reader,training,completion,index=load_inputs(cfg);pairs=[];prefixes=[]
    for t in p.TASKS:
        for s in p.SEEDS:
            for i in range(20):
                row,windows=pair(index[t,s,p.ARMS[0],i],index[t,s,p.ARMS[1],i]);pairs.append(row);prefixes.extend(windows)
    counts={o:sum(row['outcome']==o for row in pairs) for o in p.OUTCOMES}
    p.require(counts==completion['decision']['paired'],'source paired outcomes differ')
    models,bins,training_pairs=training_features(training,pairs)
    return dict(formal_result=False,exploratory_after_outcomes_known=True,causal_identification=False,
        pairs=pairs,prefixes=prefixes,groups=summaries(pairs,prefixes),training_models=models,training_bins=bins,training_pairs=training_pairs,
        outcomes=counts,reads=reader.snapshot(),limitations=dict(no_rgb=True,no_object_or_contact_state=True,
            no_expert_actions_on_policy_states=True,no_hidden_feature_distribution=True,
            no_lower_learning_rate_intervention=True,no_temporal_model_intervention=True))


def flatten(row,prefix=''):
    result={}
    for key,value in row.items():
        label=prefix+key
        if isinstance(value,dict):result.update(flatten(value,label+'.'))
        else:result[label]=value
    return result


def export_csv(path,rows):
    flat=[flatten(row) for row in rows];fields=list(dict.fromkeys(key for row in flat for key in row))
    with path.open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(flat)
    return p.pin(path)


def run(config,binding):
    cfg,bound=p.preflight(config,binding);destination=Path(cfg['output'])
    p.require(not destination.exists(),'new analysis directory required')
    started=time.monotonic();result=compute(cfg)
    destination.mkdir(parents=True)
    result.update(status='complete_existing_trace_analysis_pending_audit',science=p.pin(config),binding=p.pin(binding),
        at_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started,
        python=sys.version.split()[0],numpy=np.__version__,source_pins=cfg['sources'])
    result['tables']=[export_csv(destination/name,result[key]) for name,key in (
        ('pairs.csv','pairs'),('matched-prefixes.csv','prefixes'),('training-models.csv','training_models'),
        ('training-50-update-bins.csv','training_bins'),('training-pairs.csv','training_pairs'))]
    p.write_new(destination/'analysis.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();result=run(args.config,args.binding)
    print(json.dumps(dict(status=result['status'],pairs=len(result['pairs']),prefix_rows=len(result['prefixes']),reads=result['reads'])))
