"""Analytic full-fit gradients at fixed zero/500 residual points; no updates."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments import libero_task7_endpoint_gradient_protocol as d
from experiments import libero_task7_endpoint_gradient_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child

VECTOR_FIELDS = ['base_gradient','penalty_gradient','total_gradient','direction']


def measure(z,y,p0,w,b,coefficient,ids,jobs=None):
    n=len(z)
    if jobs: jobs.linear(n)
    raw=p0[:,2]+(np.einsum('ij,j->i',z,w,optimize=False)+b); pred=np.clip(raw,-1,1)
    e=(pred-y[:,2])**2; e0=(p0[:,2]-y[:,2])**2; diff=e-e0; inside=(raw>=-1)&(raw<=1)
    base=2*(pred-y[:,2])/7*inside; pen=base*(diff>0); total=base if coefficient==0 else base+pen
    gradients=[]
    for factor in (base,pen,total):
        if jobs: jobs.gradient()
        gradients.append(np.r_[np.mean(z*factor[:,None],axis=0),factor.mean()])
    gb,gp,g=gradients; norms=[float(np.linalg.norm(v)) for v in gradients]; norm=norms[2]
    v=-g/norm if norm else np.zeros_like(g)
    if jobs: jobs.linear(n)
    draw=np.einsum('ij,j->i',z,v[:-1],optimize=False)+v[-1]
    d.p.require(np.isfinite(raw).all() and np.isfinite(draw).all(),'nonfinite diagnostic linear form')
    dp=np.where((raw>-1)&(raw<1),draw,0.);dp=np.where(raw==-1,np.maximum(draw,0),dp);dp=np.where(raw==1,np.minimum(draw,0),dp)
    de=2*(pred-y[:,2])*dp; dh=np.where(diff>0,de,np.where(diff==0,np.maximum(de,0),0.))
    floor=float(np.mean(np.sum((p0[:,[0,1,3,4,5,6]]-y[:,[0,1,3,4,5,6]])**2,axis=1))/7)
    penalty=float(np.maximum(diff,0).mean()/7); mse=float(e.mean()); objective=floor+mse/7+coefficient*penalty
    anchor=pred.astype(np.float32).astype(np.float64); ae=(anchor-y[:,2])**2
    ao=floor+float(ae.mean())/7+coefficient*float(np.maximum(ae-e0,0).mean()/7)
    cliptag=np.where(raw<-1,0,np.where(raw==-1,1,np.where(raw<1,2,np.where(raw==1,3,4)))).astype(np.uint8)
    hingetag=np.where(diff<0,0,np.where(diff==0,1,2)).astype(np.uint8)
    digest=hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()+cliptag.tobytes()+hingetag.tobytes()).hexdigest()
    sample=float(np.mean(abs(total)*np.sqrt(np.sum(z*z,axis=1)+1)));dot=float(np.dot(gb,gp))
    result={'n':n,'fixed_other6_floor':floor,'component3_mse':mse,'base_objective':floor+mse/7,'penalty':penalty,
        'trainable_objective':mse/7+coefficient*penalty,'total_objective':objective,'anchor_objective':ao,
        'anchor_max_action_delta':float(np.max(abs(anchor-pred))),'anchor_harm_disagreements':int(np.count_nonzero((ae>e0)!=(e>e0))),
        'base_l2':norms[0],'penalty_l2':norms[1],'total_l2':norm,'total_rms':norm/math.sqrt(len(g)),
        'total_linf':float(np.max(abs(g))),'base_penalty_dot':dot,'base_penalty_cosine':dot/(norms[0]*norms[1]) if norms[0] and norms[1] else None,
        'mean_individual_total_gradient_l2':sample,'cancellation_ratio':norm/sample if sample else None,
        'directional_derivative':float(np.mean(de+coefficient*dh)/7),'branch_digest':digest}
    for i,name in enumerate(['below','low_tie','inside','high_tie','above']): result['clip_'+name+'_n']=int((cliptag==i).sum())
    for i,name in enumerate(['negative','zero','positive']):result['hinge_'+name+'_n']=int((hingetag==i).sum())
    result.update({name:values.tolist() for name,values in zip(VECTOR_FIELDS,[gb,gp,g,v])})
    d.p.require(result['anchor_max_action_delta']<=1e-7 and abs(objective-ao)<=1e-6,'anchor precision gate failed')
    d.p.require(all(np.isfinite(x).all() for x in gradients+[v]),'nonfinite gradient')
    return result


def admit(head,feature,plan):
    rows=np.asarray(plan['fit_rows'],np.int32)
    for field,values in [('fit_rows',rows),('fit_demo_indices',np.asarray(plan['fit_demos'],np.int32)),('updates',np.asarray([0,100,250,500],np.int32)),('optimizer_steps',np.arange(1,501,dtype=np.int32))]:
        d.p.require(head[field].dtype==np.int32 and np.array_equal(head[field],values),'head fit/endpoint role differs')
    values=feature[rows].astype(np.float64); mean=values.mean(0);scale=values.std(0);scale[scale<=1e-6]=1.
    for key,reference in [('scaler_mean',mean),('scaler_scale',scale)]:d.p.require(head[key].shape==(256,) and head[key].dtype==np.float64 and np.allclose(head[key],reference,atol=1e-10,rtol=0),'fit-only scaler differs')
    d.p.require(head['weights'].shape==(4,256) and head['weights'].dtype==np.float32 and head['bias'].shape==(4,) and head['bias'].dtype==np.float32 and (head['weights'][0]==0).all() and head['bias'][0]==0 and np.isfinite(head['weights'][[0,3]]).all() and np.isfinite(head['bias'][[0,3]]).all(),'head weights differ')
    return ((values-head['scaler_mean'])/head['scaler_scale']).astype(np.float32).astype(np.float64)


def pairs_and_summaries(points):
    pairs=[];initial=[];lookup={(r['seed'],r['arm'],r['fold'],r['mode'],r['point']):r for r in points}
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            for fold in range(3):
                zeros=[lookup[seed,arm,fold,mode,0] for mode in d.CASES]
                diff=max(abs(x-y) for x,y in zip(zeros[0]['total_gradient'],zeros[1]['total_gradient']))
                d.p.require(diff==0 and zeros[0]['total_objective']==zeros[1]['total_objective'] and all(r['penalty']==r['penalty_l2']==0 and r['hinge_positive_n']==0 for r in zeros),'initial lambda invariants failed')
                initial.append({'task':7,'seed':seed,'arm':arm,'fold':fold,'loss_difference':zeros[1]['total_objective']-zeros[0]['total_objective'],'gradient_max_difference':diff,'penalty_zero':True})
                for mode in d.CASES:
                    x,y=(lookup[seed,arm,fold,mode,t] for t in (0,500));d.p.require(x['fixed_other6_floor']==y['fixed_other6_floor'],'fixed6 floor changed')
                    pairs.append({'task':7,'seed':seed,'arm':arm,'fold':fold,'mode':mode,
                        'initial_objective':x['total_objective'],'endpoint_objective':y['total_objective'],
                        'initial_trainable_objective':x['trainable_objective'],'endpoint_trainable_objective':y['trainable_objective'],
                        'relative_trainable_objective':y['trainable_objective']/x['trainable_objective'] if x['trainable_objective'] else None,
                        'initial_gradient_l2':x['total_l2'],'endpoint_gradient_l2':y['total_l2'],'endpoint_directional_derivative':y['directional_derivative'],
                        **d.remaining_evidence(x['total_l2'],y['total_l2'],y['directional_derivative'])})
    seeds=[];arms=[]
    for arm in d.p.ARMS:
        for mode in d.CASES:
            selected=[r for r in pairs if r['arm']==arm and r['mode']==mode]
            for seed in d.p.SEEDS:
                part=sorted((r for r in selected if r['seed']==seed),key=lambda r:r['fold'])
                metrics={}
                for field in ('relative_trainable_objective','relative_gradient','normalized_descent'):
                    vals=[r[field] for r in part];metrics[field]={'fold_values':vals,'equal_fold_mean':math.fsum(vals)/3 if None not in vals else None}
                seeds.append({'arm':arm,'mode':mode,'seed':seed,'metrics':metrics,'all3fold_cue':all(r['remaining_cue'] for r in part)})
            arms.append({'arm':arm,'mode':mode,'seeds':d.p.SEEDS,'seed_all3fold_cues':[r['all3fold_cue'] for r in seeds if r['arm']==arm and r['mode']==mode],
                'all9head_remaining_cue':d.joint_remaining_evidence(selected),'automatic_training_promotion':False})
    return pairs,initial,seeds,arms


def execute_payload(cfg,science_pin,binding_pin):
    output=Path(cfg['output']);directory=Path(cfg['artifacts']);d.p.require(not output.exists() and not directory.exists(),'existing run; no retry');directory.mkdir(parents=True)
    limits=a.base.Limits(180);runtime=a.runtime(cfg);started=a.base.now();jobs=a.Jobs(cfg)
    paths={name:directory/(name+'.csv') for name in cfg['budget']['tables']};guard=a.Guard(cfg,paths.values(),limits,jobs);sinks={};points=[]
    try:
        with guard:
            shared=a.load(cfg['inputs']['shared'],'shared');y=shared['targets'];lengths=cfg['inputs']['action_lengths']
            expected=[(demo,step) for demo,n in enumerate(lengths) for step in range(1,n)]
            d.p.require(y.shape==(4188,7) and y.dtype==np.float32 and np.isfinite(y).all() and (abs(y)<=1).all() and list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==expected
                and shared['demo_indices'].dtype==shared['action_steps'].dtype==shared['demo_lengths'].dtype==np.int32 and shared['demo_lengths'].tolist()==lengths
                and shared['tail20'].dtype==np.bool_ and shared['tail20'].tolist()==[t>=lengths[demo]-20 for demo,t in expected],'source causal target schema differs')
            sinks={name:a.base.CsvSink(path,limits) for name,path in paths.items()}
            for model in cfg['inputs']['models']:
                cache=a.load(model['current_archive'],'current');f,p0=cache['features'],cache['predictions']
                d.p.require(f.shape==(4188,256) and p0.shape==(4188,7) and f.dtype==p0.dtype==np.float32 and np.isfinite(f).all() and np.isfinite(p0).all() and (abs(p0)<=1).all(),'current schema differs')
                for plan in cfg['inputs']['folds']:
                    rows=np.asarray(plan['fit_rows'],np.int32);previous_scaler=None
                    for mode,lam in d.CASES.items():
                        manifest=next(h for h in cfg['inputs']['heads'] if (h['seed'],h['arm'],h['fold'],h['mode'])==(model['seed'],model['arm'],plan['fold'],mode))
                        head=a.load(manifest['archive'],'head');z=admit(head,f,plan)
                        scaler=(head['scaler_mean'].tobytes(),head['scaler_scale'].tobytes())
                        d.p.require(previous_scaler is None or previous_scaler==scaler,'paired scaler bits differ');previous_scaler=scaler
                        for pos,point in ((0,0),(3,500)):
                            key=(model['seed'],model['arm'],plan['fold'],mode,point);jobs.begin(key,len(rows))
                            record={'task':7,'seed':model['seed'],'arm':model['arm'],'fold':plan['fold'],'mode':mode,'point':point,
                                **measure(z,y[rows].astype(np.float64),p0[rows].astype(np.float64),head['weights'][pos].astype(np.float64),float(head['bias'][pos]),lam,rows,jobs)}
                            jobs.end();points.append(record);sinks['points'].add({k:v for k,v in record.items() if k not in VECTOR_FIELDS})
                            param=np.r_[head['weights'][pos].astype(np.float64),float(head['bias'][pos])]
                            for j in range(257):sinks['parameters'].add({**{k:record[k] for k in ('task','seed','arm','fold','mode','point')},'coordinate':j,'parameter':float(param[j]),**{k:record[k][j] for k in VECTOR_FIELDS}})
                        del z,head
                del cache,f,p0;limits.check();print(json.dumps({'stage':'primary','seed':model['seed'],'arm':model['arm']}),flush=True)
            pairs,initial,seeds,arms=pairs_and_summaries(points)
            for row in pairs:sinks['pairs'].add(row)
            for row in initial:sinks['initial-parity'].add(row)
            for sink in sinks.values():sink.close()
            d.p.require({k:v.rows for k,v in sinks.items()}==cfg['budget']['tables'] and sum(p.stat().st_size for p in paths.values())<=67108864,'output budget differs');jobs.complete();guard.complete()
        result={'status':'complete_fit_endpoint_gradient_pending_audit','formal_result':False,'science_config':science_pin,'binding':binding_pin,'parents':cfg['parents'],'inputs':cfg['inputs'],'budget':cfg['budget'],
            'runtime':runtime,'cost':limits.record(),'started_at_utc':started,'completed_at_utc':a.base.now(),'access':guard.record(),'computation':jobs.counts,
            'tables':{k:v.pin() for k,v in sinks.items()},'points':points,'pairs':pairs,'initial_parity':initial,'seed_summaries':seeds,'arm_summaries':arms,
            'parameter_updates_original_Policy_CNN_environment_or_sealed_calls':0,'G_failed_H_unrun_and_prior_joint_cues_unchanged':True}
        a.base.write_new(output,result);return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        a.base.write_new(directory/'primary-failed-partial.json',{'status':'failed_no_retry','error':repr(error),'access':guard.record(),'computation':jobs.counts});raise


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(p.exists() for p in (output,launch,final,Path(cfg['artifacts']))),'existing run; no retry')
    record={'started_at_utc':a.base.now(),'science_config':d.p.pin(config),'binding':d.p.pin(binding),'processes':[]};a.base.write_new(launch,record);started=time.monotonic();child=None
    try:
        for stage,module in [('primary','experiments.run_libero_task7_endpoint_gradient_diagnostic'),('audit','experiments.audit_libero_task7_endpoint_gradient_diagnostic')]:
            env=os.environ.copy();env.update(bound['CPU_environment']);cmd=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            if stage=='primary':cmd+=['--mode','evaluate']
            row={'stage':stage,'command':cmd,'started_at_utc':a.base.now()};record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(195,max(1,415-(time.monotonic()-started))))
                except BaseException:stop_child(child);raise
            row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-started<=420,'overall watchdog exceeded');record['status']='complete_independently_audited'
    except BaseException as error:
        if child is not None:stop_child(child)
        record['status']='failed_no_retry';record['error']=repr(error);raise
    finally:
        record['wall_seconds']=time.monotonic()-started;record['completed_at_utc']=a.base.now();a.base.write_new(final,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);parser.add_argument('--mode',choices=['preflight','evaluate','run'],default='preflight');args=parser.parse_args()
    if args.mode=='run':result=run(args.config,args.binding)
    else:
        cfg,_=a.preflight(args.config,args.binding);result=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding)) if args.mode=='evaluate' else {'status':'bound_no_payload'}
    print(json.dumps({'status':result['status']}))
