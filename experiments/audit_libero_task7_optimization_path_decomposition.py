"""Independent scalar reconstruction of saved loss tradeoffs and paths."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import json
import math
from pathlib import Path
import sys
from experiments import libero_task7_optimization_path_decomposition_protocol as d
from experiments import libero_task7_optimization_path_decomposition_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks,CheckedTable
from experiments.audit_libero_task7_direction_harm_alignment import deep_check


def near(x,y):d.p.require(abs(x-y)<=1e-10,'independent source identity/conservation mismatch')
def divide(x,y):
 if y==0:return None
 z=x/y;d.p.require(math.isfinite(z),'independent ratio invalid');return z
def median(values):
 v=sorted(values);n=len(v)
 return float(v[n//2]) if n%2 else math.fsum([v[n//2-1],v[n//2]])/2
def turns(values):
 previous=0;result=0
 for i in range(1,len(values)):
  sign=1 if values[i]>values[i-1] else -1 if values[i]<values[i-1] else 0
  if sign:
   if previous and previous!=sign:result+=1
   previous=sign
 return result


def change(before,after,coefficient):
 dm=math.fsum([after['component3_mse'],-before['component3_mse']]);dp=math.fsum([after['penalty'],-before['penalty']]);dt=math.fsum([after['trainable_objective'],-before['trainable_objective']])
 residual=dt-dm/7-coefficient*dp;near(residual,0.)
 label='other'
 if dm>0 and dp<0:label='mse_up_penalty_down'
 elif dm<0 and dp<0:label='both_down'
 elif dm<0 and dp>0:label='mse_down_penalty_up'
 return dict(delta_mse3=dm,delta_mse3_scaled=dm/7,delta_penalty=dp,weighted_delta_penalty=coefficient*dp,delta_trainable=dt,conservation_residual=residual,step_class=label)


def reconstruct_head(ident,s,trial_groups,e,h,n):
 coefficient=d.CASES[ident['mode']]
 d.p.require(len(s)==51 and [r['extra_updates'] for r in s]==list(range(51)) and len(trial_groups)==50,'independent path coverage')
 for point in s:
  d.p.require(point['n']==n and all(point[k]>=0 for k in d.NUMBERS if k!='directional_derivative') and all(point[k]>=0 for k in d.COUNTS),'independent nonnegative/source counts')
  near(point['component3_mse']/7+coefficient*point['penalty'],point['trainable_objective']);near(point['total_objective']-point['fixed_other6_floor'],point['trainable_objective']);near(point['fixed_other6_floor'],s[0]['fixed_other6_floor'])
  d.p.require(sum(point[k] for k in d.COUNTS[:5])==n and sum(point[k] for k in d.COUNTS[5:])==n and point['penalty']<=point['component3_mse']/7+1e-10,'independent branch/penalty bound')
  if coefficient==0:near(point['base_l2'],point['total_l2'])
  else:d.p.require(abs(point['base_l2']-point['penalty_l2'])-1e-10<=point['total_l2']<=point['base_l2']+point['penalty_l2']+1e-10,'independent norm triangle')
 for cond,index in [('hold_500',0),('fullbatch_armijo',50)]:
  d.p.require(e[cond]['extra_updates']==index,'independent endpoint ordinal')
  for k in ['n']+d.NUMBERS+d.COUNTS:
   if isinstance(s[index][k],int):d.p.require(e[cond][k]==s[index][k],'independent endpoint count')
   else:near(e[cond][k],s[index][k])
 initial,final=s[0],s[-1];tr=divide(final['trainable_objective'],initial['trainable_objective']);gr=divide(final['total_l2'],initial['total_l2'])
 obj=tr is not None and tr<=.99;grad=gr is not None and gr<=.1
 label={(True,True):'both',(False,True):'gradient_only',(True,False):'objective_only',(False,False):'neither'}[obj,grad]
 d.p.require(h['extra_updates']==50 and h['terminal_reason']=='update_cap' and h['objective_target_met']==obj and h['gradient_target_met']==grad and h['joint_cue']==(obj and grad)
  and h['status']==('joint_optimization_cue' if obj and grad else 'partial_target_reduction'),'independent parent flags')
 wanted=[initial['total_objective'],final['total_objective'],initial['trainable_objective'],final['trainable_objective'],initial['total_l2'],final['total_l2'],tr,gr]
 for field,value in zip(['initial_objective','final_objective','initial_trainable_objective','final_trainable_objective','initial_gradient_l2','final_gradient_l2','trainable_ratio','gradient_ratio'],wanted):near(h[field],value)
 steps=[];total_trials=0
 for update in range(50):
  b,c=s[update],s[update+1];trials=trial_groups[update];total_trials+=len(trials)
  d.p.require(1<=len(trials)<=12 and b['directional_derivative']<0,'independent trial coverage')
  for index,t in enumerate(trials):
   d.p.require(t['trial_index']==index and t['from_extra_updates']==update and t['alpha']==2**(-index),'independent trial grid')
   near(t['slope'],b['directional_derivative']);bound=b['trainable_objective']+1e-4*t['alpha']*b['directional_derivative'];near(t['armijo_bound'],bound)
   decision=bool(t['trainable_objective']<b['trainable_objective'] and t['trainable_objective']<=bound)
   d.p.require(t['accepted']==decision and decision==(index==len(trials)-1),'independent first acceptance')
   near(t['total_objective']-b['fixed_other6_floor'],t['trainable_objective'])
  near(trials[-1]['trainable_objective'],c['trainable_objective']);near(trials[-1]['total_objective'],c['total_objective'])
  d.p.require(c['trainable_objective']<b['trainable_objective'],'independent accepted path monotonicity')
  delta=change(b,c,coefficient);near(c['total_objective']-b['total_objective'],delta['delta_trainable'])
  steps.append({**ident,'from_update':update,'to_update':update+1,**delta,'delta_gradient':c['total_l2']-b['total_l2'],
   'gradient_step_ratio':divide(c['total_l2'],b['total_l2']),'alpha':trials[-1]['alpha'],'accepted_index':len(trials)-1,'rejected_trials':len(trials)-1})
 d.p.require(total_trials==h['trials'],'independent head trial total')
 whole=change(initial,final,coefficient)
 for name in ['delta_mse3','delta_mse3_scaled','delta_penalty','weighted_delta_penalty','delta_trainable']:near(math.fsum(t[name] for t in steps),whole[name])
 up=coefficient==1 and whole['delta_mse3']>0;comp=divide(whole['delta_mse3_scaled'],-whole['delta_penalty']) if up and whole['delta_penalty']<0 else None
 windows=[]
 for block in range(5):
  lo=block*10;hi=lo+10;interval=steps[lo:hi];b,c=s[lo],s[hi];drop=b['trainable_objective']-c['trainable_objective'];delta=change(b,c,coefficient)
  for field in ['delta_mse3','delta_mse3_scaled','delta_penalty','weighted_delta_penalty','delta_trainable']:near(math.fsum(t[field] for t in interval),delta[field])
  gs=[v['total_l2'] for v in s[lo:hi+1]];alphas=[v['alpha'] for v in interval]
  windows.append({**ident,'goal_class':label,'lambda1_mse_up':up,'start_update':lo,'end_update':hi,**delta,'target_drop':drop,
   'target_drop_over_initial':divide(drop,initial['trainable_objective']),'fraction_of_total_drop':divide(drop,initial['trainable_objective']-final['trainable_objective']),
   'gradient_ratio':divide(c['total_l2'],b['total_l2']),'gradient_increases':sum(gs[j]>gs[j-1] for j in range(1,11)),
   'gradient_change_switches':turns(gs),'alpha_min':min(alphas),'alpha_median':median(alphas),'alpha_max':max(alphas),'rejected_trials':sum(t['rejected_trials'] for t in interval)})
 anchors=[]
 for index in [0,10,20,30,40,50]:
  row=s[index];anchors.append({**ident,'extra_updates':index,'trainable_objective':row['trainable_objective'],'component3_mse':row['component3_mse'],'penalty':row['penalty'],
   'gradient_l2':row['total_l2'],'base_l2':row['base_l2'],'penalty_l2':row['penalty_l2'],'trainable_ratio':divide(row['trainable_objective'],initial['trainable_objective']),
   'gradient_ratio':divide(row['total_l2'],initial['total_l2']),'active_harm_n':row['hinge_positive_n']})
 alphas=[t['alpha'] for t in steps];counts=Counter(t['step_class'] for t in steps);gs=[v['total_l2'] for v in s]
 summary={**ident,'goal_class':label,'nonjoint':label!='both','lambda1_mse_up':up,**whole,'compensation_ratio':comp,'compensation_reason':'defined' if comp is not None else 'not_lambda1_MSEup_penaltydown',
  'initial_trainable':initial['trainable_objective'],'final_trainable':final['trainable_objective'],'trainable_ratio':tr,'initial_gradient':initial['total_l2'],'final_gradient':final['total_l2'],'gradient_ratio':gr,
  'initial_active_harm_n':initial['hinge_positive_n'],'final_active_harm_n':final['hinge_positive_n'],
  **{kind+'_steps':counts[kind] for kind in d.DESIGN['step_classes']},'gradient_increases':sum(gs[j]>gs[j-1] for j in range(1,51)),
  'gradient_change_switches':turns(gs),'alpha_min':min(alphas),'alpha_median':median(alphas),'alpha_max':max(alphas),'rejected_trials':total_trials-50,
  'last10_drop_over_initial':windows[4]['target_drop_over_initial'],'last10_fraction_total_drop':windows[4]['fraction_of_total_drop'],
  'last10_gradient_ratio':windows[4]['gradient_ratio'],'last10_gradient_increases':windows[4]['gradient_increases']}
 return summary,steps,windows,anchors


def strata_rows(heads):
 result=[]
 for arm in d.p.ARMS:
  for mode in d.CASES:
   for seed in d.p.SEEDS:
    source=[h for h in heads if h['arm']==arm and h['mode']==mode and h['seed']==seed]
    for cohort in ['all','both','gradient_only','objective_only','neither']+(['mse_up','not_mse_up'] if mode=='harm_protected_affine' else []):
     group=[]
     for h in source:
      if cohort=='all' or cohort in d.CLASSES and h['goal_class']==cohort or cohort=='mse_up' and h['lambda1_mse_up'] or cohort=='not_mse_up' and not h['lambda1_mse_up']:group.append(h)
     row=dict(task=7,arm=arm,mode=mode,seed=seed,cohort=cohort,n=len(group),mse_up_n=sum(h['lambda1_mse_up'] for h in group),nonjoint_n=sum(h['nonjoint'] for h in group))
     for field in ['delta_mse3_scaled','weighted_delta_penalty','delta_trainable','gradient_ratio','last10_fraction_total_drop','last10_gradient_ratio','alpha_median']:
      values=[h[field] for h in group];row[field+'_mean']=math.fsum(values)/len(values) if values and all(v is not None for v in values) else None
     result.append(row)
 return result


def reconstruct(cfg,data):
 expected={tuple(h[k] for k in d.IDS) for h in cfg['head_identities']};indexed={name:defaultdict(dict) for name in data}
 for name,rows in data.items():
  for row in rows:
   k=tuple(row[field] for field in d.IDS);d.p.require(k in expected,'independent unknown source head')
   sub=row['extra_updates'] if name=='states' else (row['from_extra_updates'],row['trial_index']) if name=='trials' else row['condition'] if name=='endpoints' else 'head'
   d.p.require(sub not in indexed[name][k],'independent duplicate record');indexed[name][k][sub]=row
  d.p.require(set(indexed[name])==expected,'independent table head coverage')
 output={name:[] for name in cfg['budget']['tables']}
 for ident in cfg['head_identities']:
  k=tuple(ident[f] for f in d.IDS);d.p.require(set(indexed['states'][k])==set(range(51)) and set(indexed['endpoints'][k])=={'hold_500','fullbatch_armijo'},'independent endpoint/state coverage')
  states=[indexed['states'][k][j] for j in range(51)];groups=[[] for _ in range(50)]
  for (update,index),t in sorted(indexed['trials'][k].items()):
   d.p.require(update in range(50) and index in range(12),'independent unlisted iteration');groups[update].append(t)
  result=reconstruct_head(ident,states,groups,indexed['endpoints'][k],indexed['heads'][k]['head'],cfg['fit_rows'][str(ident['fold'])])
  output['heads'].append(result[0]);output['transitions'].extend(result[1]);output['windows'].extend(result[2]);output['anchors'].extend(result[3])
 output['strata']=strata_rows(output['heads']);known=cfg['expected_counts'];counts=dict(Counter(h['goal_class'] for h in output['heads']))
 d.p.require(len(output['heads'])==known['heads'] and sum(h['nonjoint'] for h in output['heads'])==known['nonjoint'] and sum(h['lambda1_mse_up'] for h in output['heads'])==known['lambda1_mse_up'] and counts==known['goal_classes'],'independent parent stratum counts')
 return output


def summaries(tables):
 seeds=list(filter(lambda r:r['cohort']=='all',tables['strata']));sources=[]
 for arm in d.p.ARMS:
  for mode in d.CASES:
   group=[h for h in tables['heads'] if h['arm']==arm and h['mode']==mode]
   sources.append(dict(arm=arm,mode=mode,n=len(group),goal_counts={c:len([h for h in group if h['goal_class']==c]) for c in d.CLASSES},lambda1_mse_up_n=sum(h['lambda1_mse_up'] for h in group),
    seed_rows=[s for s in seeds if s['arm']==arm and s['mode']==mode],
    delta_mse3_scaled_mean=math.fsum(h['delta_mse3_scaled'] for h in group)/len(group) if group else None,
    delta_penalty_mean=math.fsum(h['delta_penalty'] for h in group)/len(group) if group else None,
    delta_trainable_mean=math.fsum(h['delta_trainable'] for h in group)/len(group) if group else None))
 return seeds,sources


def audit_payload(cfg,sp,bp):
 a.validate_call(cfg,sp,bp);d.p.require('experiments.analyze_libero_task7_optimization_path_decomposition' not in sys.modules,'primary numerical implementation imported')
 directory=Path(cfg['artifacts']);d.p.require(not any((directory/n).exists() for n in ['independent-audit.json','completion.json','audit-failed-partial.json']),'existing audit; no retry')
 limits=a.base.Limits(120,268435456);runtime=a.runtime(cfg);started=a.base.now();output=Path(cfg['output']);record=json.loads(output.read_text());result_pin=d.p.pin(output)
 d.p.require(record['status']=='complete_saved_path_decomposition_pending_audit' and record['science_config']==sp and record['binding']==bp and record['parents']==cfg['parents'] and record['inputs']==cfg['inputs'] and record['budget']==cfg['budget']
  and record['formal_result'] is False and record['new_training_forward_gradient_or_sealed_calls']==0 and record['old_cues_G_failed_H_unrun_unchanged'] is True,'analysis provenance differs')
 d.p.require(record['runtime']['environment']==cfg['execution']['environment'] and record['runtime']['python']==cfg['execution']['python'] and record['cost']['elapsed_seconds']<=120 and record['cost']['peak_RSS_bytes']<=268435456,'analysis runtime budget differs')
 d.p.require(set(record['tables'])==set(cfg['budget']['tables']) and {k:v['rows'] for k,v in record['tables'].items()}==cfg['budget']['tables'],'analysis output rows differ')
 for name,ref in record['tables'].items():d.p.require(Path(ref['path'])==directory/(name+'.csv') and d.p.sha(ref['path'])==ref['sha256'],'output table bytes/path differ')
 d.p.require(sum(Path(r['path']).stat().st_size for r in record['tables'].values())<=16777216,'output size exceeded')
 guard=a.Guard(cfg,[r['path'] for r in record['tables'].values()],limits,audit=True);checks=Checks();readers={}
 try:
  with guard:
   data={name:list(guard.project(name)) for name in d.COLUMNS};tables=reconstruct(cfg,data);seeds,sources=summaries(tables)
   readers={name:CheckedTable(ref,checks,limits) for name,ref in record['tables'].items()}
   for name,rows in tables.items():
    for row in rows:readers[name].row(row)
    readers[name].finish()
   deep_check(checks,record['records'],tables,'saved decomposition');deep_check(checks,record['seed_summaries'],seeds,'seed descriptions');deep_check(checks,record['source_summaries'],sources,'source descriptions')
   guard.complete()
  deep_check(checks,record['access'],guard.record(),'source projection ledger')
  for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output changed during audit')
  d.p.require(d.p.pin(output)==result_pin,'analysis JSON changed')
  audit=dict(status='passed_independent_saved_path_decomposition_audit',formal_result=False,science_config=sp,binding=bp,result=result_pin,runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),scalar_checks=checks.n,primary_numerical_implementation_imported=False,new_training_forward_gradient_array_or_sealed_calls=0)
  path=directory/'independent-audit.json';a.base.write_new(path,audit)
  a.base.write_new(directory/'completion.json',dict(status='complete_independently_audited_saved_path_decomposition',formal_result=False,science_config=sp,binding=bp,result=result_pin,audit=d.p.pin(path),completed_at_utc=a.base.now(),total_source_CSV_passes=8,total_projected_cells=412578,total_source_SHA=16,new_training_forward_gradient_array_or_sealed_calls=0));return audit
 except BaseException as error:
  for reader in readers.values():
   if not reader.handle.closed:reader.handle.close()
  a.base.write_new(directory/'audit-failed-partial.json',dict(status='failed_no_retry',error=repr(error),access=guard.record(),scalar_checks=checks.n));raise


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);args=parser.parse_args()
 cfg,_=a.preflight(args.config,args.binding);r=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':r['status']}))
