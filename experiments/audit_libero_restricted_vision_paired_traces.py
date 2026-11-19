"""Independent Python scalar formulas for the existing-output-only analysis."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import statistics

from experiments import libero_restricted_vision_paired_trace_protocol as p


def norm(values):return math.sqrt(math.fsum(value*value for value in values))


def independent_measures(a,q,g):
    n=len(a)
    diffs=lambda rows,order:[[rows[i+2][j]-2*rows[i+1][j]+rows[i][j] for j in range(len(rows[0]))] for i in range(n-2)] if order==2 else [[rows[i+1][j]-rows[i][j] for j in range(len(rows[0]))] for i in range(n-1)]
    mean_norm=lambda rows:statistics.fmean(norm(row) for row in rows)
    translation=[row[:3] for row in a];rotation=[row[3:6] for row in a];grip=[[row[6]] for row in a]
    signs=[1 if row[6]>p.DEADBAND else -1 if row[6]<-p.DEADBAND else 0 for row in a]
    active=[value for value in signs if value];switch=sum(x!=y for x,y in zip(active,active[1:]))
    close=next((i for i,value in enumerate(signs,1) if value==1),None)
    opened=next((i for i,value in enumerate(signs,1) if close is not None and i>close and value==-1),None)
    qdist=[math.dist(left,right) for left,right in zip(q,q[1:])];gdist=[math.dist(left,right) for left,right in zip(g,g[1:])]
    return dict(translation_norm_mean=mean_norm(translation),rotation_norm_mean=mean_norm(rotation),
        gripper_abs_mean=statistics.fmean(abs(row[6]) for row in a),joint_step_mean=statistics.fmean(qdist),gripper_step_mean=statistics.fmean(gdist),
        joint_path=math.fsum(qdist),gripper_path=math.fsum(gdist),
        translation_saturation_fraction=sum(abs(v)>=p.SATURATION for row in translation for v in row)/(3*n),
        rotation_saturation_fraction=sum(abs(v)>=p.SATURATION for row in rotation for v in row)/(3*n),
        gripper_saturation_fraction=sum(abs(row[6])>=p.SATURATION for row in a)/n,
        translation_step_difference=mean_norm(diffs(translation,1)) if n>1 else None,
        rotation_step_difference=mean_norm(diffs(rotation,1)) if n>1 else None,
        gripper_step_difference=mean_norm(diffs(grip,1)) if n>1 else None,
        translation_second_difference=mean_norm(diffs(translation,2)) if n>2 else None,
        rotation_second_difference=mean_norm(diffs(rotation,2)) if n>2 else None,
        gripper_second_difference=mean_norm(diffs(grip,2)) if n>2 else None,
        gripper_switches=switch,gripper_non_neutral_steps=len(active),gripper_switch_rate=switch/(len(active)-1) if len(active)>1 else None,
        positive_command_fraction=signs.count(1)/n,first_close=close,first_open_after_close=opened)


def independent_order(left,right,le,re):
    if left is None:
        if right is None:return 'both_not_observed'
        return 'right_observed_left_later_bound' if right<=le else 'indeterminate_left_ended_before_reference_event'
    if right is None:return 'left_observed_reference_later_bound' if left<=re else 'indeterminate_reference_ended_before_left_event'
    return 'observed_equal' if left==right else 'observed_left_earlier' if left<right else 'observed_left_later'


def independent_pair(left,right):
    la=[row['action'] for row in left['stepwise_public_state_trace'][1:]];ra=[row['action'] for row in right['stepwise_public_state_trace'][1:]]
    lq=[row['joint_state'] for row in left['stepwise_public_state_trace']];rq=[row['joint_state'] for row in right['stepwise_public_state_trace']]
    lg=[row['gripper_state'] for row in left['stepwise_public_state_trace']];rg=[row['gripper_state'] for row in right['stepwise_public_state_trace']]
    p.require(lq[0]==rq[0] and lg[0]==rg[0] and left['reset_observation_sha256']==right['reset_observation_sha256'],'audit reset differs')
    lf,rf=independent_measures(la,lq,lg),independent_measures(ra,rq,rg)
    outcome=p.OUTCOMES[0 if left['success'] and right['success'] else 1 if left['success'] else 2 if right['success'] else 3]
    row=dict(task=left['task_id'],seed=left['training_seed'],initial_state_index=left['initial_state_index'],outcome=outcome,
        frozen_steps=len(la),finetune_steps=len(ra),common_observed_steps=min(len(la),len(ra)),
        first_action_gap=math.dist(la[0],ra[0]),first_translation_gap=math.dist(la[0][:3],ra[0][:3]),
        first_rotation_gap=math.dist(la[0][3:6],ra[0][3:6]),first_gripper_gap=float(abs(la[0][6]-ra[0][6])),
        first_pre_joint_gap=0.,first_pre_gripper_gap=0.,frozen_full=lf,finetune_full=rf,
        first_close_order=independent_order(lf['first_close'],rf['first_close'],len(la),len(ra)),
        first_open_after_close_order=independent_order(lf['first_open_after_close'],rf['first_open_after_close'],len(la),len(ra)))
    prefixes=[]
    for h in p.HORIZONS:
        item=dict(task=row['task'],seed=row['seed'],initial_state_index=row['initial_state_index'],outcome=outcome,horizon=h,
            eligible=min(len(la),len(ra))>=h,frozen_available=len(la)>=h,finetune_available=len(ra)>=h,
            frozen=None,finetune=None,delta=None,gaps=None)
        if item['eligible']:
            a,b=independent_measures(la[:h],lq[:h+1],lg[:h+1]),independent_measures(ra[:h],rq[:h+1],rg[:h+1])
            item.update(frozen=a,finetune=b,delta={key:b[key]-a[key] if a[key] is not None and b[key] is not None else None for key in a},
                gaps=dict(action_gap_mean=statistics.fmean(math.dist(x,y) for x,y in zip(la[:h],ra[:h])),
                    translation_gap_mean=statistics.fmean(math.dist(x[:3],y[:3]) for x,y in zip(la[:h],ra[:h])),
                    rotation_gap_mean=statistics.fmean(math.dist(x[3:6],y[3:6]) for x,y in zip(la[:h],ra[:h])),
                    gripper_action_gap_mean=statistics.fmean(abs(x[6]-y[6]) for x,y in zip(la[:h],ra[:h])),
                    pre_joint_gap=math.dist(lq[h-1],rq[h-1]),post_joint_gap=math.dist(lq[h],rq[h]),
                    pre_gripper_gap=math.dist(lg[h-1],rg[h-1]),post_gripper_gap=math.dist(lg[h],rg[h]),
                    joint_gap_mean=statistics.fmean(math.dist(x,y) for x,y in zip(lq[1:h+1],rq[1:h+1])),
                    gripper_gap_mean=statistics.fmean(math.dist(x,y) for x,y in zip(lg[1:h+1],rg[1:h+1]))))
        prefixes.append(item)
    return row,prefixes


class Comparison:
    def __init__(self):self.checks=0;self.maximum_abs_difference=0.
    def check(self,expected,actual,label='value'):
        self.checks+=1
        if isinstance(expected,dict):
            p.require(isinstance(actual,dict) and set(expected)==set(actual),'audit schema differs: '+label)
            for key,value in expected.items():self.check(value,actual[key],label+'.'+key)
        elif isinstance(expected,list):
            p.require(isinstance(actual,list) and len(expected)==len(actual),'audit length differs: '+label)
            for i,(a,b) in enumerate(zip(expected,actual)):self.check(a,b,label+f'[{i}]')
        elif type(expected) is float:
            p.require(type(actual) in (float,int) and math.isfinite(actual) and math.isclose(expected,actual,rel_tol=1e-11,abs_tol=1e-12),'audit numeric difference: '+label)
            self.maximum_abs_difference=max(self.maximum_abs_difference,abs(expected-actual))
        else:p.require(type(actual) is type(expected) and actual==expected,'audit exact difference: '+label)


def independent_stats(values):
    values=list(values);present=[float(value) for value in values if value is not None]
    return dict(total=len(values),n=len(present),missing=len(values)-len(present),mean=statistics.fmean(present) if present else None,
        median=statistics.median(present) if present else None,minimum=min(present) if present else None,maximum=max(present) if present else None)


def audit(config,binding):
    cfg,bound=p.preflight(config,binding);destination=Path(cfg['output']);raw=json.loads((destination/'analysis.json').read_text())
    p.require(raw['status']=='complete_existing_trace_analysis_pending_audit' and raw['formal_result'] is False and raw['science']==p.pin(config)
        and raw['binding']==p.pin(binding) and raw['causal_identification'] is False and raw['exploratory_after_outcomes_known'] is True,'analysis provenance differs')
    reader=p.Sources(cfg);source={key:reader.read(item) for key,item in cfg['sources'].items()}
    index={}
    for descriptor in source['evaluation']['rollout']['episode_files']:
        ep=reader.read({k:descriptor[k] for k in ('path','sha256')},episode=True)
        key=(ep['task_id'],ep['training_seed'],ep['arm'],ep['initial_state_index'])
        p.require(key not in index,'duplicate audit episode');index[key]=ep
    expected={(t,s,a,i) for t in p.TASKS for s in p.SEEDS for a in p.ARMS for i in range(20)}
    p.require(set(index)==expected and reader.parses==244,'audit requires all240 episodes')
    pairs=[];prefixes=[]
    for t in p.TASKS:
        for s in p.SEEDS:
            for i in range(20):
                a,b=independent_pair(index[t,s,p.ARMS[0],i],index[t,s,p.ARMS[1],i]);pairs.append(a);prefixes.extend(b)
    compare=Comparison();compare.check(pairs,raw['pairs'],'pairs');compare.check(prefixes,raw['prefixes'],'prefixes')
    compare.check({o:sum(row['outcome']==o for row in pairs) for o in p.OUTCOMES},raw['outcomes'],'outcomes')
    compare.check(source['completion']['decision']['paired'],raw['outcomes'],'original_outcomes')
    compare.check(reader.snapshot(),raw['reads'],'read_counts')
    selectors=[{}]+[dict(outcome=o) for o in p.OUTCOMES]+[dict(task=t) for t in p.TASKS]+[dict(seed=s) for s in p.SEEDS]
    selectors += [dict(task=t,seed=s) for t in p.TASKS for s in p.SEEDS]
    selectors += [dict(task=t,outcome=o) for t in p.TASKS for o in p.OUTCOMES]
    selectors += [dict(task=t,seed=s,outcome=o) for t in p.TASKS for s in p.SEEDS for o in p.OUTCOMES]
    compare.check([row['selector'] for row in raw['groups']],selectors,'complete_strata')
    for group in raw['groups']:
        selector=group['selector'];matches=lambda row:all(row[k]==v for k,v in selector.items())
        selected=[row for row in pairs if matches(row)]
        compare.check(len(selected),group['pairs'],'group_pairs')
        compare.check({o:sum(row['outcome']==o for row in selected) for o in p.OUTCOMES},group['outcomes'],'group_outcomes')
        compare.check({key:independent_stats(row[key] for row in selected) for key in group['first_actions']},group['first_actions'],'first_actions')
        compare.check(dict(Counter(row['first_close_order'] for row in selected)),group['close_order'],'close_order')
        compare.check([row['horizon'] for row in group['prefixes']],p.HORIZONS,'prefix_scope')
        for item in group['prefixes']:
            candidates=[row for row in prefixes if row['horizon']==item['horizon'] and matches(row)]
            eligible=[row for row in candidates if row['eligible']]
            compare.check(len(candidates),item['pairs']);compare.check(len(eligible),item['eligible']);compare.check(len(candidates)-len(eligible),item['censored'])
            for category in ('frozen','finetune','delta','gaps'):
                values={}
                if eligible:
                    for key in eligible[0][category]:
                        data=[row[category][key] for row in eligible];summary=independent_stats(data)
                        if category=='delta':summary.update(positive=sum(v is not None and v>p.SIGN_EPS for v in data),negative=sum(v is not None and v<-p.SIGN_EPS for v in data),zero=sum(abs(v)<=p.SIGN_EPS for v in data if v is not None))
                        values[key]=summary
                compare.check(values,item[category],'group_'+category)
    models=source['training']['models'];model_index={(row['task'],row['seed'],row['arm']):row for row in models}
    compare.check(len(models),12,'training_models')
    expected_models=[];bins=[];training_pairs=[]
    for row in models:
        trace=row['trace'];p.require(len(trace)==500 and [event['update'] for event in trace]==list(range(1,501)),'audit incomplete training')
        expected_models.append(dict(task=row['task'],seed=row['seed'],arm=row['arm'],trainable_parameters=row['trainable_parameters'],
            first50_loss=statistics.fmean(event['loss'] for event in trace[:50]),last50_loss=statistics.fmean(event['loss'] for event in trace[-50:]),
            clipped_updates=sum(event['gradient_norm']>1 for event in trace),sampling_sha256=row['sampling_sha256'],initial_full_state_sha256=row['initial_full_state_sha256']))
        for start in range(0,500,50):
            block=trace[start:start+50]
            bins.append(dict(task=row['task'],seed=row['seed'],arm=row['arm'],update_first=start+1,update_last=start+50,
                loss_mean=statistics.fmean(event['loss'] for event in block),gradient_norm_mean=statistics.fmean(event['gradient_norm'] for event in block),clipped=sum(event['gradient_norm']>1 for event in block)))
    for t in p.TASKS:
        for s in p.SEEDS:
            left,right=(model_index[t,s,a] for a in p.ARMS)
            p.require(all(a['minibatch_ids']==b['minibatch_ids'] and a['valid_rows']==b['valid_rows'] for a,b in zip(left['trace'],right['trace'])),'audit unpaired training')
            a,b=(statistics.fmean(event['loss'] for event in row['trace'][-50:]) for row in (left,right))
            fs=sum(index[t,s,p.ARMS[0],i]['success'] for i in range(20));ts=sum(index[t,s,p.ARMS[1],i]['success'] for i in range(20))
            training_pairs.append(dict(task=t,seed=s,frozen_last50_loss=a,finetune_last50_loss=b,last50_loss_delta=b-a,
                frozen_successes=fs,finetune_successes=ts,success_delta=ts-fs,
                frozen_clipped_updates=left['clipped_updates'],finetune_clipped_updates=right['clipped_updates']))
    compare.check(expected_models,raw['training_models'],'training_models');compare.check(bins,raw['training_bins'],'training_bins');compare.check(training_pairs,raw['training_pairs'],'training_pairs')
    p.require(len(raw['tables'])==5,'complete five analysis tables required')
    def flatten(row,prefix=''):
        out={}
        for name,value in row.items():
            if isinstance(value,dict):out.update(flatten(value,prefix+name+'.'))
            else:out[prefix+name]=value
        return out
    for item,key in zip(raw['tables'],('pairs','prefixes','training_models','training_bins','training_pairs')):
        p.require(p.pin(item['path'])==item and Path(item['path']).resolve().parent==destination.resolve(),'table pin differs')
        with Path(item['path']).open() as handle:
            parser=csv.DictReader(handle);rows=list(parser);p.require(len(rows)==len(raw[key]),'table row count differs')
            expected_rows=[flatten(row) for row in raw[key]]
            expected_fields={name for row in expected_rows for name in row}
            p.require(len(parser.fieldnames)==len(expected_fields) and set(parser.fieldnames)==expected_fields,'CSV fields differ')
            for actual,expected_row in zip(rows,expected_rows):
                for field in expected_fields:
                    value=expected_row.get(field)
                    p.require(actual[field]==('' if value is None else str(value)),'CSV cell differs: '+field)
                    compare.checks+=1
    receipt=dict(status='passed_independent_existing_paired_trace_audit',formal_result=False,causal_identification=False,
        science=p.pin(config),binding=p.pin(binding),analysis=p.pin(destination/'analysis.json'),
        at_UTC=datetime.now(timezone.utc).isoformat(),numeric_and_schema_checks=compare.checks,
        maximum_absolute_numeric_difference=compare.maximum_abs_difference,source_reads=reader.snapshot(),
        pairs=120,prefix_rows=600,training_models=12,all_original_outcomes_retained=True)
    p.write_new(destination/'audit.json',receipt)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--binding',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(audit(args.config,args.binding)))
