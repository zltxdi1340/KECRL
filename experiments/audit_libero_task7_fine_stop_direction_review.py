"""Independent stdlib reconstruction of saved geometry, inference and trials."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
from experiments import libero_task7_fine_stop_direction_review_protocol as d
from experiments import libero_task7_fine_stop_direction_review_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks,CheckedTable
from experiments.audit_libero_task7_direction_harm_alignment import deep_check


def close(x,y):
 d.p.require(type(x) in (int,float) and math.isfinite(x) and abs(x-y)<=1e-10,'independent scalar/norm/conservation mismatch')


def divide(x,y):
 if y==0:return None
 value=x/y;d.p.require(math.isfinite(value),'independent nonfinite ratio');return value


def geometry_reference(gb,gp,g,direction,clip_ties,hinge_ties,slope):
 vectors=[gb,gp,g,direction]
 d.p.require(all(len(v)==257 and all(type(x) in (int,float) and math.isfinite(x) for x in v) for v in vectors),'independent257 finite coordinates required')
 for i in range(257):close(g[i],math.fsum([gb[i],gp[i]]));close(direction[i],-g[i])
 inner=lambda x,y:math.fsum(x[i]*y[i] for i in range(257))
 lengths=[math.sqrt(math.fsum(x*x for x in v)) for v in vectors]
 bp,bd,pd,gd=[inner(x,y) for x,y in [(gb,gp),(gb,direction),(gp,direction),(g,direction)]]
 close(gd,-lengths[2]*lengths[2]);close(gd,bd+pd)
 cosine=divide(bp,lengths[0]*lengths[1])
 if cosine is not None:
  d.p.require(-1-1e-10<=cosine<=1+1e-10,'independent cosine outside range');cosine=min(1.,max(-1.,cosine))
 gap=slope-gd
 if clip_ties==0:
  d.p.require(gap>=-1e-10,'independent invalid hinge correction')
  if hinge_ties==0:close(slope,gd)
 sm=7*bd if clip_ties==0 else None;sp=slope-bd if clip_ties==0 else None
 return dict(base_l2=lengths[0],penalty_l2=lengths[1],total_l2=lengths[2],direction_l2=lengths[3],
  base_penalty_dot=bp,base_penalty_cosine=cosine,dot_base_direction=bd,dot_penalty_direction=pd,dot_total_direction=gd,
  cached_exact_total_slope=slope,kink_gap=gap,exact_mse_slope=sm,exact_penalty_slope=sp,
  mse_slope_status='exact_from_saved_gradient_no_clip_tie' if clip_ties==0 else 'unresolved_clip_tie')


def infer(m0,m,st,sm):
 d.p.require(0<=m<=m0 and st<0,'independent fixed envelope/exact descent invalid')
 slack=m0-m
 if st>=-1e-10:label='near_zero_total_slope_unresolved'
 elif slack>1e-10:label='positive_slack_local_feasible_by_continuity'
 elif slack!=0:label='positive_slack_near_numeric_band'
 elif sm is None:label='boundary_clip_kink_unresolved'
 elif sm>1e-10:label='boundary_direction_conflict_near_zero_alpha'
 elif sm< -1e-10:label='boundary_descent_local_feasible'
 else:label='boundary_near_zero_mse_slope_unresolved'
 scale=divide(slack,sm) if slack>0 and sm is not None and sm>0 else None
 sgn=lambda x:'positive' if x>0 else 'negative' if x<0 else 'zero'
 return dict(slack=slack,strict_slack_sign=sgn(slack),strict_total_slope_sign=sgn(st),strict_mse_slope_sign=sgn(sm) if sm is not None else None,
  inference_class=label,linear_alpha_scale=scale,linear_scale_over_smallest_grid_alpha=divide(scale,2.**-19) if scale is not None else None,
  untried_step_evaluated=False,global_infeasibility_claim=False)


def terms(m0,m,p,t,mt,pt,tt,alpha,sm):
 dm=mt-m;dp=pt-p;dt=tt-t;residual=dt-dm/7-dp;close(residual,0.)
 prediction=sm*alpha if sm is not None else None
 return dict(delta_mse_step=dm,mse_gate_excess=mt-m0,delta_penalty=dp,delta_trainable=dt,mse_secant=divide(dm,alpha),
  mse_first_order_prediction=prediction,mse_first_order_residual=dm-prediction if prediction is not None else None,conservation_residual=residual)


def decision_reference(current,trial,slope,index,trial_mse,start_mse,opt):
 d.p.require(opt==d.parent.ARMS[1] and slope<0,'independent candidate descent required')
 original=trial<current and trial<=current+1e-4*(2.**(-index))*slope
 mse=trial_mse<=start_mse;accept=original and mse
 reason='accepted' if accept else 'armijo_and_mse_rejected' if not original and not mse else 'armijo_rejected' if not original else 'mse_envelope_rejected'
 return dict(armijo_pass=original,mse_envelope_pass=mse,mse_gate_applied=True,accepted=accept,reason=reason)


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
   and set(indexed['trials'][k])==set(range(20)),'point/coordinate/last20 coverage differs')
 return indexed


def point_record(cfg,h,row,vectors,role,m500):
 k=0 if role=='start' else h['stop_extra_updates']
 d.p.require(row['extra_updates']==k and row['n']==cfg['fit_rows'][str(h['fold'])]
  and all(row[f]>=0 for f in ['fixed_other6_floor','component3_mse','penalty','trainable_objective','total_objective','base_l2','penalty_l2','total_l2',*d.COUNTS]),'point role/nonnegative quantities differ')
 close(row['trainable_objective'],row['component3_mse']/7+row['penalty']);close(row['total_objective'],row['fixed_other6_floor']+row['trainable_objective'])
 d.p.require(sum(row[f] for f in d.COUNTS[:5])==sum(row[f] for f in d.COUNTS[5:])==row['n']
  and row['penalty']<=row['component3_mse']/7+d.TOL,'branch/penalty bounds differ')
 geometry=geometry_reference(*(vectors[f] for f in d.VECTORS),row['clip_low_tie_n']+row['clip_high_tie_n'],row['hinge_zero_n'],row['directional_derivative'])
 for f in ['base_l2','penalty_l2','total_l2']:close(geometry[f],row[f])
 inference=infer(m500,row['component3_mse'],row['directional_derivative'],geometry['exact_mse_slope'])
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
 for j in range(20):
  t=trials[j];d.p.require(t['optimization_arm']==d.parent.ARMS[1] and t['from_extra_updates']==h['stop_extra_updates']
   and t['trial_index']==j and t['alpha']==2.**(-j),'last round role/grid differs')
  close(t['slope'],last['cached_exact_total_slope']);d.p.require(t['start500_component3_mse']==m500,'fixed500 trial envelope differs')
  close(t['trainable_objective'],t['component3_mse']/7+t['penalty']);close(t['total_objective'],last['fixed_other6_floor']+t['trainable_objective'])
  bound=last['trainable_objective']+.0001*t['alpha']*last['cached_exact_total_slope'];close(t['armijo_bound'],bound)
  expected=decision_reference(last['trainable_objective'],t['trainable_objective'],last['cached_exact_total_slope'],j,t['component3_mse'],m500,d.parent.ARMS[1])
  d.p.require(all(type(t[f]) is type(v) and t[f]==v for f,v in expected.items()) and not t['accepted'],'cached trial decision differs or non-exhausted round')
  contribution=terms(m500,last['component3_mse'],last['penalty'],last['trainable_objective'],t['component3_mse'],t['penalty'],t['trainable_objective'],t['alpha'],last['exact_mse_slope'])
  out_trials.append({**t,'cohort':h['cohort'],'stop_mse':last['component3_mse'],'fixed500_mse':m500,
   'armijo_margin':bound-t['trainable_objective'],'parameter_step_l2':t['alpha']*last['direction_l2'],**contribution,'untried_step_evaluated':False})
 changes={f+'_change_l2':math.sqrt(math.fsum((x-y)**2 for x,y in zip(vectors[1][f],vectors[0][f]))) for f in d.VECTORS}
 summary={**a.identity(h),'cohort':h['cohort'],'stop_extra_updates':h['stop_extra_updates'],
  'start_mse':m500,'stop_mse':last['component3_mse'],'start_trainable':first['trainable_objective'],'stop_trainable':last['trainable_objective'],
  'start_exact_mse_slope':first['exact_mse_slope'],'stop_exact_mse_slope':last['exact_mse_slope'],
  'start_exact_total_slope':first['cached_exact_total_slope'],'stop_exact_total_slope':last['cached_exact_total_slope'],
  'start_gradient_cosine':first['base_penalty_cosine'],'stop_gradient_cosine':last['base_penalty_cosine'],
  'start_clip_ties':first['clip_low_tie_n']+first['clip_high_tie_n'],'stop_clip_ties':last['clip_low_tie_n']+last['clip_high_tie_n'],
  'start_hinge_ties':first['hinge_zero_n'],'stop_hinge_ties':last['hinge_zero_n'],
  **{k:last[k] for k in ['slack','strict_slack_sign','strict_total_slope_sign','strict_mse_slope_sign','inference_class','linear_alpha_scale','linear_scale_over_smallest_grid_alpha']},
  **changes,'last_trials':20,'last_armijo_passes':sum(t['armijo_pass'] for t in out_trials),
  'last_mse_passes':sum(t['mse_envelope_pass'] for t in out_trials),'last_accepted':sum(t['accepted'] for t in out_trials),
  **{'reason_'+reason+'_n':sum(t['reason']==reason for t in out_trials) for reason in ['armijo_rejected','mse_envelope_rejected','armijo_and_mse_rejected','accepted']},
  'min_gate_excess':min(t['mse_gate_excess'] for t in out_trials),'max_gate_excess':max(t['mse_gate_excess'] for t in out_trials),
  'min_mse_secant':min(t['mse_secant'] for t in out_trials),'max_mse_secant':max(t['mse_secant'] for t in out_trials),
  'min_first_order_residual':min(t['mse_first_order_residual'] for t in out_trials) if last['exact_mse_slope'] is not None else None,
  'max_first_order_residual':max(t['mse_first_order_residual'] for t in out_trials) if last['exact_mse_slope'] is not None else None,
  'untried_step_evaluated':False,'global_infeasibility_claim':False}
 d.p.require(summary['last_armijo_passes']>=0,'published Armijo-feasible exhausted head differs')
 return points,out_trials,summary


def strata_rows(heads):
 rows=[]
 for arm in d.p.ARMS:
  for seed in d.p.SEEDS:
   for cohort in ['short']:
    selected=[h for h in heads if (h['arm'],h['seed'],h['cohort'])==(arm,seed,cohort)]
    row=dict(task=7,arm=arm,seed=seed,cohort=cohort,n=len(selected),exact_mse_slope_n=sum(h['stop_exact_mse_slope'] is not None for h in selected))
    row.update({'class_'+c+'_n':sum(h['inference_class']==c for h in selected) for c in a.CLASSES})
    for field in ['stop_extra_updates','slack','stop_exact_mse_slope','stop_exact_total_slope','stop_gradient_cosine','last_armijo_passes','min_gate_excess','linear_scale_over_smallest_grid_alpha']:
     values=[h[field] for h in selected];row[field+'_mean']=math.fsum(values)/len(values) if values and None not in values else None
    rows.append(row)
 return rows


def reconstruct(cfg,data):
 indexed=index_rows(cfg,data);result={name:[] for name in cfg['budget']['tables']}
 for h in cfg['heads']:
  k=a.key(h);points,trials,summary=describe_head(cfg,h,indexed['endpoints'][k],indexed['parameters'][k],indexed['trials'][k])
  result['points'].extend(points);result['trials'].extend(trials);result['heads'].append(summary)
 result['strata']=strata_rows(result['heads']);d.p.require({k:len(v) for k,v in result.items()}==cfg['budget']['tables'],'result row counts differ')
 return result




def audit_payload(cfg,sp,bp):
 a.validate_call(cfg,sp,bp);d.p.require('experiments.analyze_libero_task7_fine_stop_direction_review' not in sys.modules,'primary numerical implementation imported')
 directory=Path(cfg['artifacts']);d.p.require(not any((directory/n).exists() for n in ['independent-audit.json','completion.json','audit-failed-partial.json']),'existing audit; no retry')
 limits=a.base.Limits(120,268435456);runtime=a.runtime(cfg);started=a.base.now();output=Path(cfg['output']);record=json.loads(output.read_text());result_pin=d.p.pin(output)
 d.p.require(record['status']=='complete_fine_stop_direction_review_pending_audit' and record['science_config']==sp and record['binding']==bp and record['parents']==cfg['parents'] and record['inputs']==cfg['inputs'] and record['budget']==cfg['budget']
  and record['formal_result'] is False and record['new_training_forward_gradient_or_sealed_calls']==0 and record['old_cues_G_failed_H_unrun_unchanged'] is True,'analysis provenance differs')
 d.p.require(record['runtime']['environment']==cfg['execution']['environment'] and record['runtime']['python']==cfg['execution']['python'] and record['cost']['elapsed_seconds']<=120 and record['cost']['peak_RSS_bytes']<=268435456,'analysis runtime budget differs')
 d.p.require(set(record['tables'])==set(cfg['budget']['tables']) and {k:v['rows'] for k,v in record['tables'].items()}==cfg['budget']['tables'],'analysis output rows differ')
 for name,ref in record['tables'].items():d.p.require(Path(ref['path'])==directory/(name+'.csv') and d.p.sha(ref['path'])==ref['sha256'],'output table bytes/path differ')
 d.p.require(sum(Path(r['path']).stat().st_size for r in record['tables'].values())<=16777216,'output size exceeded')
 guard=a.Guard(cfg,[r['path'] for r in record['tables'].values()],limits,audit=True);checks=Checks();readers={}
 try:
  with guard:
   data=guard.project();tables=reconstruct(cfg,data)
   readers={name:CheckedTable(ref,checks,limits) for name,ref in record['tables'].items()}
   for name,rows in tables.items():
    for row in rows:readers[name].row(row)
    readers[name].finish()
   deep_check(checks,record['records'],tables,'saved direction grid')
   guard.complete()
  deep_check(checks,record['access'],guard.record(),'source projection ledger')
  for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output changed during audit')
  d.p.require(d.p.pin(output)==result_pin,'analysis JSON changed')
  audit=dict(status='passed_independent_fine_stop_direction_review_audit',formal_result=False,science_config=sp,binding=bp,result=result_pin,runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),scalar_checks=checks.n,primary_numerical_implementation_imported=False,new_training_forward_gradient_array_or_sealed_calls=0)
  path=directory/'independent-audit.json';a.base.write_new(path,audit)
  a.base.write_new(directory/'completion.json',dict(status='complete_independently_audited_fine_stop_direction_review',formal_result=False,science_config=sp,binding=bp,result=result_pin,audit=d.p.pin(path),completed_at_utc=a.base.now(),total_parent_JSON_parses=2,total_projected_cells=2*sum(cfg['budget']['projected_cells_per_stage'].values()),total_parent_JSON_SHA=4,source_array_or_CSV_reads_including_SHA=0,new_training_forward_gradient_array_or_sealed_calls=0));return audit
 except BaseException as error:
  for reader in readers.values():
   if not reader.handle.closed:reader.handle.close()
  a.base.write_new(directory/'audit-failed-partial.json',dict(status='failed_no_retry',error=repr(error),access=guard.record(),scalar_checks=checks.n));raise


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);args=parser.parse_args()
 cfg,_=a.preflight(args.config,args.binding);r=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':r['status']}))
