"""Saved CSV target accounting and optimization paths; no new readouts."""
from __future__ import annotations
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from experiments import libero_task7_optimization_path_decomposition_protocol as d
from experiments import libero_task7_optimization_path_decomposition_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child


def close(x,y):d.p.require(abs(x-y)<=1e-10,'saved numeric parity/conservation differs')
def switches(values):
 signs=[1 if b>a else -1 for a,b in zip(values,values[1:]) if b!=a]
 return sum(x!=y for x,y in zip(signs,signs[1:]))


def describe_head(ident,states,rounds,endpoints,head,n):
 d.p.require([r['extra_updates'] for r in states]==list(range(51)) and len(rounds)==50,'complete50-state path required')
 coefficient=d.CASES[ident['mode']]
 for row in states:
  d.p.require(row['n']==n and all(row[k]>=0 for k in d.NUMBERS if k!='directional_derivative') and all(row[k]>=0 for k in d.COUNTS),'negative/count/schema differs')
  close(row['trainable_objective'],row['component3_mse']/7+coefficient*row['penalty']);close(row['total_objective'],row['fixed_other6_floor']+row['trainable_objective'])
  close(row['fixed_other6_floor'],states[0]['fixed_other6_floor'])
  d.p.require(sum(row[k] for k in d.COUNTS[:5])==sum(row[k] for k in d.COUNTS[5:])==n,'branch counts do not cover fit')
  d.p.require(row['penalty']<=row['component3_mse']/7+1e-10,'excess penalty exceeds squared error')
  if coefficient==0:close(row['total_l2'],row['base_l2'])
  else:d.p.require(abs(row['base_l2']-row['penalty_l2'])-1e-10<=row['total_l2']<=row['base_l2']+row['penalty_l2']+1e-10,'gradient norm decomposition invalid')
 for condition,k in [('hold_500',0),('fullbatch_armijo',50)]:
  endpoint=endpoints[condition];d.p.require(endpoint['extra_updates']==k,'endpoint ordinal differs')
  for field in ['n']+d.NUMBERS+d.COUNTS:
   if type(states[k][field]) is int:d.p.require(endpoint[field]==states[k][field],'endpoint count differs')
   else:close(endpoint[field],states[k][field])
 first,last=states[0],states[-1];tr=d.safe_ratio(last['trainable_objective'],first['trainable_objective']);gr=d.safe_ratio(last['total_l2'],first['total_l2'])
 of=tr is not None and tr<=.99;gf=gr is not None and gr<=.1;category=d.goal_class(of,gf)
 d.p.require(head['extra_updates']==50 and head['terminal_reason']=='update_cap' and head['objective_target_met']==of and head['gradient_target_met']==gf and head['joint_cue']==(of and gf)
  and head['status']==('joint_optimization_cue' if of and gf else 'partial_target_reduction'),'source goal/stop flags differ')
 for field,wanted in [('initial_objective',first['total_objective']),('final_objective',last['total_objective']),('initial_trainable_objective',first['trainable_objective']),('final_trainable_objective',last['trainable_objective']),('initial_gradient_l2',first['total_l2']),('final_gradient_l2',last['total_l2']),('trainable_ratio',tr),('gradient_ratio',gr)]:close(head[field],wanted)
 transitions=[]
 for k,(before,after) in enumerate(zip(states,states[1:])):
  trials=rounds[k];d.p.require([t['trial_index'] for t in trials]==list(range(len(trials))) and 1<=len(trials)<=12 and before['directional_derivative']<0,'trial round/grid differs')
  for j,t in enumerate(trials):
   d.p.require(t['from_extra_updates']==k and t['alpha']==2.**(-j),'trial step differs');close(t['slope'],before['directional_derivative'])
   bound=before['trainable_objective']+1e-4*t['alpha']*before['directional_derivative'];close(bound,t['armijo_bound'])
   accepted=t['trainable_objective']<before['trainable_objective'] and t['trainable_objective']<=bound
   d.p.require(t['accepted']==accepted==(j==len(trials)-1),'not first Armijo acceptance');close(t['total_objective'],t['trainable_objective']+before['fixed_other6_floor'])
  accepted=trials[-1];close(accepted['trainable_objective'],after['trainable_objective']);close(accepted['total_objective'],after['total_objective'])
  d.p.require(after['trainable_objective']<before['trainable_objective'],'accepted path not strictly decreasing')
  terms=d.terms(before['component3_mse'],before['penalty'],before['trainable_objective'],after['component3_mse'],after['penalty'],after['trainable_objective'],coefficient)
  close(after['total_objective']-before['total_objective'],terms['delta_trainable'])
  transitions.append({**ident,'from_update':k,'to_update':k+1,**terms,'delta_gradient':after['total_l2']-before['total_l2'],
   'gradient_step_ratio':d.safe_ratio(after['total_l2'],before['total_l2']),'alpha':accepted['alpha'],'accepted_index':accepted['trial_index'],'rejected_trials':len(trials)-1})
 d.p.require(sum(len(r) for r in rounds)==head['trials'],'head trial total differs')
 terms=d.terms(first['component3_mse'],first['penalty'],first['trainable_objective'],last['component3_mse'],last['penalty'],last['trainable_objective'],coefficient)
 for field in ['delta_mse3','delta_mse3_scaled','delta_penalty','weighted_delta_penalty','delta_trainable']:close(math.fsum(t[field] for t in transitions),terms[field])
 up=coefficient==1 and terms['delta_mse3']>0
 ratio=d.safe_ratio(terms['delta_mse3_scaled'],-terms['delta_penalty']) if up and terms['delta_penalty']<0 else None
 reason='defined' if ratio is not None else 'not_lambda1_MSEup_penaltydown'
 windows=[]
 for lo,hi in zip(d.ANCHORS[:-1],d.ANCHORS[1:]):
  segment=transitions[lo:hi];left,right=states[lo],states[hi];drop=left['trainable_objective']-right['trainable_objective']
  wterms=d.terms(left['component3_mse'],left['penalty'],left['trainable_objective'],right['component3_mse'],right['penalty'],right['trainable_objective'],coefficient)
  for field in ['delta_mse3','delta_mse3_scaled','delta_penalty','weighted_delta_penalty','delta_trainable']:close(math.fsum(t[field] for t in segment),wterms[field])
  alphas=[s['alpha'] for s in segment];gs=[s['total_l2'] for s in states[lo:hi+1]]
  windows.append({**ident,'goal_class':category,'lambda1_mse_up':up,'start_update':lo,'end_update':hi,**wterms,'target_drop':drop,
   'target_drop_over_initial':d.safe_ratio(drop,first['trainable_objective']),'fraction_of_total_drop':d.safe_ratio(drop,first['trainable_objective']-last['trainable_objective']),
   'gradient_ratio':d.safe_ratio(right['total_l2'],left['total_l2']),'gradient_increases':sum(b>a for a,b in zip(gs,gs[1:])),
   'gradient_change_switches':switches(gs),'alpha_min':min(alphas),'alpha_median':float(statistics.median(alphas)),'alpha_max':max(alphas),'rejected_trials':sum(t['rejected_trials'] for t in segment)})
 anchors=[{**ident,'extra_updates':k,'trainable_objective':states[k]['trainable_objective'],'component3_mse':states[k]['component3_mse'],'penalty':states[k]['penalty'],
  'gradient_l2':states[k]['total_l2'],'base_l2':states[k]['base_l2'],'penalty_l2':states[k]['penalty_l2'],
  'trainable_ratio':d.safe_ratio(states[k]['trainable_objective'],first['trainable_objective']),'gradient_ratio':d.safe_ratio(states[k]['total_l2'],first['total_l2']),
  'active_harm_n':states[k]['hinge_positive_n']} for k in d.ANCHORS]
 counts=Counter(t['step_class'] for t in transitions);alphas=[t['alpha'] for t in transitions]
 summary={**ident,'goal_class':category,'nonjoint':category!='both','lambda1_mse_up':up,**terms,'compensation_ratio':ratio,'compensation_reason':reason,
  'initial_trainable':first['trainable_objective'],'final_trainable':last['trainable_objective'],'trainable_ratio':tr,
  'initial_gradient':first['total_l2'],'final_gradient':last['total_l2'],'gradient_ratio':gr,
  'initial_active_harm_n':first['hinge_positive_n'],'final_active_harm_n':last['hinge_positive_n'],
  **{kind+'_steps':counts[kind] for kind in d.DESIGN['step_classes']},'gradient_increases':sum(t['delta_gradient']>0 for t in transitions),
  'gradient_change_switches':switches([s['total_l2'] for s in states]),'alpha_min':min(alphas),'alpha_median':float(statistics.median(alphas)),'alpha_max':max(alphas),
  'rejected_trials':head['trials']-50,'last10_drop_over_initial':windows[-1]['target_drop_over_initial'],'last10_fraction_total_drop':windows[-1]['fraction_of_total_drop'],
  'last10_gradient_ratio':windows[-1]['gradient_ratio'],'last10_gradient_increases':windows[-1]['gradient_increases']}
 return summary,transitions,windows,anchors


def strata_rows(heads):
 rows=[]
 for arm in d.p.ARMS:
  for mode in d.CASES:
   for seed in d.p.SEEDS:
    base=[h for h in heads if (h['arm'],h['mode'],h['seed'])==(arm,mode,seed)]
    cohorts=['all']+d.CLASSES+(['mse_up','not_mse_up'] if mode=='harm_protected_affine' else [])
    for cohort in cohorts:
     selected=base if cohort=='all' else [h for h in base if h['lambda1_mse_up']==(cohort=='mse_up')] if cohort in ('mse_up','not_mse_up') else [h for h in base if h['goal_class']==cohort]
     row=dict(task=7,arm=arm,mode=mode,seed=seed,cohort=cohort,n=len(selected),mse_up_n=sum(h['lambda1_mse_up'] for h in selected),nonjoint_n=sum(h['nonjoint'] for h in selected))
     for field in ['delta_mse3_scaled','weighted_delta_penalty','delta_trainable','gradient_ratio','last10_fraction_total_drop','last10_gradient_ratio','alpha_median']:
      values=[h[field] for h in selected];row[field+'_mean']=math.fsum(values)/len(values) if values and None not in values else None
     rows.append(row)
 return rows


def analyze_rows(cfg,data):
 allowed={a.key(h) for h in cfg['head_identities']};stores={name:{} for name in data}
 for name,rows in data.items():
  for row in rows:
   key=a.key(row);d.p.require(key in allowed,'unlisted head identity')
   sub=row['extra_updates'] if name=='states' else (row['from_extra_updates'],row['trial_index']) if name=='trials' else row['condition'] if name=='endpoints' else 0
   group=stores[name].setdefault(key,{});d.p.require(sub not in group,'duplicate saved record');group[sub]=row
  d.p.require(set(stores[name])==allowed,'incomplete table head coverage')
 outputs={name:[] for name in cfg['budget']['tables']}
 for ident in cfg['head_identities']:
  key=a.key(ident);d.p.require(set(stores['states'][key])==set(range(51)) and set(stores['endpoints'][key])=={'hold_500','fullbatch_armijo'},'saved state coverage differs')
  states=[stores['states'][key][k] for k in range(51)];raw=stores['trials'][key]
  d.p.require(all(0<=k<50 and 0<=j<12 for k,j in raw),'unlisted round')
  rounds=[[row for (update,_),row in sorted(raw.items()) if update==k] for k in range(50)]
  result=describe_head(ident,states,rounds,stores['endpoints'][key],stores['heads'][key][0],cfg['fit_rows'][str(ident['fold'])])
  for name,records in zip(['heads','transitions','windows','anchors'],[[result[0]],result[1],result[2],result[3]]):outputs[name].extend(records)
 outputs['strata']=strata_rows(outputs['heads'])
 counts=Counter(h['goal_class'] for h in outputs['heads']);expected=cfg['expected_counts']
 d.p.require(len(outputs['heads'])==expected['heads'] and sum(h['nonjoint'] for h in outputs['heads'])==expected['nonjoint'] and sum(h['lambda1_mse_up'] for h in outputs['heads'])==expected['lambda1_mse_up'] and dict(counts)==expected['goal_classes'],'parent endpoint strata differ')
 return outputs


def summaries(tables):
 seeds=[r for r in tables['strata'] if r['cohort']=='all'];sources=[]
 for arm in d.p.ARMS:
  for mode in d.CASES:
   selected=[r for r in tables['heads'] if (r['arm'],r['mode'])==(arm,mode)]
   sources.append(dict(arm=arm,mode=mode,n=len(selected),goal_counts={c:sum(r['goal_class']==c for r in selected) for c in d.CLASSES},
    lambda1_mse_up_n=sum(r['lambda1_mse_up'] for r in selected),seed_rows=[r for r in seeds if (r['arm'],r['mode'])==(arm,mode)],
    delta_mse3_scaled_mean=math.fsum(r['delta_mse3_scaled'] for r in selected)/len(selected) if selected else None,
    delta_penalty_mean=math.fsum(r['delta_penalty'] for r in selected)/len(selected) if selected else None,
    delta_trainable_mean=math.fsum(r['delta_trainable'] for r in selected)/len(selected) if selected else None))
 return seeds,sources


def execute_payload(cfg,sp,bp):
 a.validate_call(cfg,sp,bp);directory=Path(cfg['artifacts']);output=Path(cfg['output']);d.p.require(not directory.exists() and not output.exists(),'existing run; no retry');directory.mkdir(parents=True)
 limits=a.base.Limits(120,268435456);runtime=a.runtime(cfg);started=a.base.now();paths={k:directory/(k+'.csv') for k in cfg['budget']['tables']};guard=a.Guard(cfg,paths.values(),limits);sinks={}
 try:
  with guard:
   data={name:list(guard.project(name)) for name in d.COLUMNS};tables=analyze_rows(cfg,data)
   sinks={name:a.base.CsvSink(path,limits) for name,path in paths.items()}
   for name,rows in tables.items():
    for row in rows:sinks[name].add(row)
   for sink in sinks.values():sink.close()
   d.p.require({k:s.rows for k,s in sinks.items()}==cfg['budget']['tables'] and sum(p.stat().st_size for p in paths.values())<=16777216,'output budget differs');guard.complete()
  seeds,sources=summaries(tables)
  result=dict(status='complete_saved_path_decomposition_pending_audit',formal_result=False,science_config=sp,binding=bp,parents=cfg['parents'],inputs=cfg['inputs'],budget=cfg['budget'],runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),tables={k:s.pin() for k,s in sinks.items()},records=tables,seed_summaries=seeds,source_summaries=sources,new_training_forward_gradient_or_sealed_calls=0,old_cues_G_failed_H_unrun_unchanged=True)
  a.base.write_new(output,result);return result
 except BaseException as error:
  for s in sinks.values():
   if not s.handle.closed:s.close()
  a.base.write_new(directory/'analysis-failed-partial.json',dict(status='failed_no_retry',error=repr(error),access=guard.record()));raise


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
 cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
 d.p.require(not any(p.exists() for p in (output,launch,final,Path(cfg['artifacts']))),'existing run; no retry')
 record=dict(started_at_utc=a.base.now(),science_config=d.p.pin(config),binding=d.p.pin(binding),processes=[]);a.base.write_new(launch,record);started=time.monotonic();child=None
 try:
  for stage,module in [('analysis','experiments.analyze_libero_task7_optimization_path_decomposition'),('audit','experiments.audit_libero_task7_optimization_path_decomposition')]:
   env=os.environ.copy();env.update(bound['CPU_environment']);cmd=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
   if stage=='analysis':cmd+=['--mode','evaluate']
   row=dict(stage=stage,command=cmd,started_at_utc=a.base.now());record['processes'].append(row)
   with output.with_name(output.stem+'-'+stage+'.log').open('x') as log:
    child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
    try:row['exit_code']=child.wait(timeout=min(135,max(1,295-(time.monotonic()-started))))
    except BaseException:stop_child(child);raise
   row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
  d.p.require(time.monotonic()-started<=300,'overall watchdog exceeded');record['status']='complete_independently_audited'
 except BaseException as error:
  if child is not None:stop_child(child)
  record['status']='failed_no_retry';record['error']=repr(error);raise
 finally:record['wall_seconds']=time.monotonic()-started;record['completed_at_utc']=a.base.now();a.base.write_new(final,record)
 return record


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);parser.add_argument('--mode',choices=['preflight','evaluate','run'],default='preflight');args=parser.parse_args()
 if args.mode=='run':result=run(args.config,args.binding)
 else:
  cfg,_=a.preflight(args.config,args.binding);result=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding)) if args.mode=='evaluate' else {'status':'bound_no_source_data'}
 print(json.dumps({'status':result['status']}))
