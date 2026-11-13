"""Read-only saved endpoints/gradients/last trials, with no new candidate steps."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from experiments import libero_task7_mse_gate_stop_direction_grid_protocol as d
from experiments import libero_task7_mse_gate_stop_direction_grid_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child


def close(x,y):d.p.require(type(x) in (int,float) and math.isfinite(x) and abs(x-y)<=d.TOL,'saved scalar/norm/conservation mismatch')


def index_rows(cfg,data):
 expected={a.key(h) for h in cfg['heads']};indexed={name:{} for name in d.COLUMNS}
 d.p.require(set(data)==set(d.COLUMNS),'unlisted source table')
 for name,rows in data.items():
  d.p.require(len(rows)==cfg['budget']['selected_rows_per_stage'][name],'selected data count differs')
  for row in rows:
   d.p.require(list(row)==d.COLUMNS[name] and a.key(row) in expected,'projected source identity/fields differ')
   sub=row['condition'] if name=='endpoints' else (row['condition'],row['coordinate']) if name=='parameters' else row['trial_index']
   group=indexed[name].setdefault(a.key(row),{})
   d.p.require(sub not in group,'duplicate selected source record');group[sub]=row
  d.p.require(set(indexed[name])==expected,'source head coverage differs')
 for h in cfg['heads']:
  k=a.key(h)
  d.p.require(set(indexed['endpoints'][k])==set(cfg['design']['points'])
   and set(indexed['parameters'][k])=={(c,j) for c in cfg['design']['points'] for j in range(257)}
   and set(indexed['trials'][k])==set(range(12)),'point/coordinate/last12 coverage differs')
 return indexed


def point_record(cfg,h,row,vectors,role,m500):
 k=0 if role=='start' else h['stop_extra_updates']
 d.p.require(row['extra_updates']==k and row['n']==cfg['fit_rows'][str(h['fold'])]
  and all(row[f]>=0 for f in ['fixed_other6_floor','component3_mse','penalty','trainable_objective','total_objective','base_l2','penalty_l2','total_l2',*d.COUNTS]),'point role/nonnegative quantities differ')
 close(row['trainable_objective'],row['component3_mse']/7+row['penalty']);close(row['total_objective'],row['fixed_other6_floor']+row['trainable_objective'])
 d.p.require(sum(row[f] for f in d.COUNTS[:5])==sum(row[f] for f in d.COUNTS[5:])==row['n']
  and row['penalty']<=row['component3_mse']/7+d.TOL,'branch/penalty bounds differ')
 geometry=d.gradient_geometry(*(vectors[f] for f in d.VECTORS),row['clip_low_tie_n']+row['clip_high_tie_n'],row['hinge_zero_n'],row['directional_derivative'])
 for f in ['base_l2','penalty_l2','total_l2']:close(geometry[f],row[f])
 inference=d.feasibility_evidence(m500,row['component3_mse'],row['directional_derivative'],geometry['exact_mse_slope'])
 return {**a.identity(h),'cohort':h['cohort'],'point_role':role,'condition':row['condition'],'extra_updates':k,'n':row['n'],
  **{f:row[f] for f in ['fixed_other6_floor','component3_mse','penalty','trainable_objective','total_objective','branch_digest',*d.COUNTS]},
  'cached_base_l2':row['base_l2'],'cached_penalty_l2':row['penalty_l2'],'cached_total_l2':row['total_l2'],**geometry,**inference}


def describe_head(cfg,h,endpoints,coordinates,trials):
 points=[];vectors=[];m500=endpoints['saved500']['component3_mse']
 for role,condition in zip(['start','stop'],cfg['design']['points']):
  v={name:[coordinates[condition,j][name] for j in range(257)] for name in d.VECTORS}
  vectors.append(v);points.append(point_record(cfg,h,endpoints[condition],v,role,m500))
 first,last=points;source_first,source_last=(endpoints[c] for c in cfg['design']['points'])
 close(first['fixed_other6_floor'],last['fixed_other6_floor'])
 if h['cohort']=='zero':
  d.p.require(h['stop_extra_updates']==0 and vectors[0]==vectors[1]
   and all(source_first[f]==source_last[f] for f in d.COLUMNS['endpoints'] if f!='condition'),'zero-update points changed')
 else:d.p.require(h['stop_extra_updates']>0 and last['trainable_objective']<first['trainable_objective'],'short path has no target descent')
 out_trials=[]
 for j in range(12):
  t=trials[j];d.p.require(t['optimization_arm']==d.parent.ARMS[1] and t['from_extra_updates']==h['stop_extra_updates']
   and t['trial_index']==j and t['alpha']==2.**(-j),'last round role/grid differs')
  close(t['slope'],last['cached_exact_total_slope']);d.p.require(t['start500_component3_mse']==m500,'fixed500 trial envelope differs')
  close(t['trainable_objective'],t['component3_mse']/7+t['penalty']);close(t['total_objective'],last['fixed_other6_floor']+t['trainable_objective'])
  bound=last['trainable_objective']+.0001*t['alpha']*last['cached_exact_total_slope'];close(t['armijo_bound'],bound)
  expected=d.parent.trial_decision(last['trainable_objective'],t['trainable_objective'],last['cached_exact_total_slope'],j,t['component3_mse'],m500,d.parent.ARMS[1])
  d.p.require(all(type(t[f]) is type(v) and t[f]==v for f,v in expected.items()) and not t['accepted'],'cached trial decision differs or non-exhausted round')
  terms=d.trial_terms(m500,last['component3_mse'],last['penalty'],last['trainable_objective'],t['component3_mse'],t['penalty'],t['trainable_objective'],t['alpha'],last['exact_mse_slope'])
  out_trials.append({**t,'cohort':h['cohort'],'stop_mse':last['component3_mse'],'fixed500_mse':m500,
   'armijo_margin':bound-t['trainable_objective'],'parameter_step_l2':t['alpha']*last['direction_l2'],**terms,'untried_step_evaluated':False})
 changes={f+'_change_l2':math.sqrt(math.fsum((x-y)**2 for x,y in zip(vectors[1][f],vectors[0][f]))) for f in d.VECTORS}
 summary={**a.identity(h),'cohort':h['cohort'],'stop_extra_updates':h['stop_extra_updates'],
  'start_mse':m500,'stop_mse':last['component3_mse'],'start_trainable':first['trainable_objective'],'stop_trainable':last['trainable_objective'],
  'start_exact_mse_slope':first['exact_mse_slope'],'stop_exact_mse_slope':last['exact_mse_slope'],
  'start_exact_total_slope':first['cached_exact_total_slope'],'stop_exact_total_slope':last['cached_exact_total_slope'],
  'start_gradient_cosine':first['base_penalty_cosine'],'stop_gradient_cosine':last['base_penalty_cosine'],
  'start_clip_ties':first['clip_low_tie_n']+first['clip_high_tie_n'],'stop_clip_ties':last['clip_low_tie_n']+last['clip_high_tie_n'],
  'start_hinge_ties':first['hinge_zero_n'],'stop_hinge_ties':last['hinge_zero_n'],
  **{k:last[k] for k in ['slack','strict_slack_sign','strict_total_slope_sign','strict_mse_slope_sign','inference_class','linear_alpha_scale','linear_scale_over_smallest_grid_alpha']},
  **changes,'last_trials':12,'last_armijo_passes':sum(t['armijo_pass'] for t in out_trials),
  'last_mse_passes':sum(t['mse_envelope_pass'] for t in out_trials),'last_accepted':sum(t['accepted'] for t in out_trials),
  **{'reason_'+reason+'_n':sum(t['reason']==reason for t in out_trials) for reason in ['armijo_rejected','mse_envelope_rejected','armijo_and_mse_rejected','accepted']},
  'min_gate_excess':min(t['mse_gate_excess'] for t in out_trials),'max_gate_excess':max(t['mse_gate_excess'] for t in out_trials),
  'min_mse_secant':min(t['mse_secant'] for t in out_trials),'max_mse_secant':max(t['mse_secant'] for t in out_trials),
  'min_first_order_residual':min(t['mse_first_order_residual'] for t in out_trials) if last['exact_mse_slope'] is not None else None,
  'max_first_order_residual':max(t['mse_first_order_residual'] for t in out_trials) if last['exact_mse_slope'] is not None else None,
  'untried_step_evaluated':False,'global_infeasibility_claim':False}
 d.p.require(summary['last_armijo_passes']>0,'published Armijo-feasible exhausted head differs')
 return points,out_trials,summary


def strata_rows(heads):
 rows=[]
 for arm in d.p.ARMS:
  for seed in d.p.SEEDS:
   for cohort in ['zero','short']:
    selected=[h for h in heads if (h['arm'],h['seed'],h['cohort'])==(arm,seed,cohort)]
    row=dict(task=7,arm=arm,seed=seed,cohort=cohort,n=len(selected),exact_mse_slope_n=sum(h['stop_exact_mse_slope'] is not None for h in selected))
    row.update({'class_'+c+'_n':sum(h['inference_class']==c for h in selected) for c in a.CLASSES})
    for field in ['stop_extra_updates','slack','stop_exact_mse_slope','stop_exact_total_slope','stop_gradient_cosine','last_armijo_passes','min_gate_excess','linear_scale_over_smallest_grid_alpha']:
     values=[h[field] for h in selected];row[field+'_mean']=math.fsum(values)/len(values) if values and None not in values else None
    rows.append(row)
 return rows


def analyze_rows(cfg,data):
 indexed=index_rows(cfg,data);result={name:[] for name in cfg['budget']['tables']}
 for h in cfg['heads']:
  k=a.key(h);points,trials,summary=describe_head(cfg,h,indexed['endpoints'][k],indexed['parameters'][k],indexed['trials'][k])
  result['points'].extend(points);result['trials'].extend(trials);result['heads'].append(summary)
 result['strata']=strata_rows(result['heads']);d.p.require({k:len(v) for k,v in result.items()}==cfg['budget']['tables'],'result row counts differ')
 return result


def execute_payload(cfg,sp,bp):
 a.validate_call(cfg,sp,bp);directory=Path(cfg['artifacts']);output=Path(cfg['output'])
 d.p.require(not directory.exists() and not output.exists(),'existing run; no retry');directory.mkdir(parents=True)
 limits=a.base.Limits(120,268435456);runtime=a.runtime(cfg);started=a.base.now();paths={k:directory/(k+'.csv') for k in cfg['budget']['tables']};guard=a.Guard(cfg,paths.values(),limits);sinks={}
 try:
  with guard:
   data={name:list(guard.project(name)) for name in d.COLUMNS};tables=analyze_rows(cfg,data)
   sinks={name:a.base.CsvSink(path,limits) for name,path in paths.items()}
   for name,rows in tables.items():
    for row in rows:sinks[name].add(row)
   for sink in sinks.values():sink.close()
   d.p.require({k:s.rows for k,s in sinks.items()}==cfg['budget']['tables'] and sum(p.stat().st_size for p in paths.values())<=16777216,'output budget differs');guard.complete()
  result=dict(status='complete_mse_gate_direction_grid_pending_audit',formal_result=False,science_config=sp,binding=bp,parents=cfg['parents'],inputs=cfg['inputs'],budget=cfg['budget'],runtime=runtime,cost=limits.record(),
   started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),tables={k:s.pin() for k,s in sinks.items()},records=tables,
   new_training_forward_gradient_or_sealed_calls=0,old_cues_G_failed_H_unrun_unchanged=True)
  a.base.write_new(output,result);return result
 except BaseException as error:
  for sink in sinks.values():
   if not sink.handle.closed:sink.close()
  a.base.write_new(directory/'analysis-failed-partial.json',dict(status='failed_no_retry',error=repr(error),access=guard.record()));raise


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
 cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
 d.p.require(not any(p.exists() for p in (output,launch,final,Path(cfg['artifacts']))),'existing run; no retry')
 record=dict(started_at_utc=a.base.now(),science_config=d.p.pin(config),binding=d.p.pin(binding),processes=[]);a.base.write_new(launch,record);started=time.monotonic();child=None
 try:
  for stage,module in [('analysis','experiments.analyze_libero_task7_mse_gate_stop_direction_grid'),('audit','experiments.audit_libero_task7_mse_gate_stop_direction_grid')]:
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
