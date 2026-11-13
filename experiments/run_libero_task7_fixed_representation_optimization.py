"""Finite full-fit NumPy optimization of fixed500 auxiliary copies only."""
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
from experiments import libero_task7_fixed_representation_optimization_protocol as d
from experiments import libero_task7_fixed_representation_optimization_access as a
from experiments.run_libero_task7_endpoint_gradient_diagnostic import admit
from experiments.run_libero_task7_endpoint_gradient_diagnostic_v2 import admit_reused
from experiments.analyze_libero_task7_component3_history_transition import stop_child

VECTORS=['base_gradient','penalty_gradient','total_gradient','direction']
TABLES=['endpoints','parameters','states','trials','heads','scaler-parity']


def measure(context,theta,coefficient,rows,jobs=None):
    z,y,p0=context;n=len(z)
    if jobs:jobs.check_context(z,y,p0);jobs.state_begin(theta);jobs.linear(n)
    raw=p0[:,2]+(np.einsum('ij,j->i',z,theta[:-1],optimize=False)+theta[-1]);pred=np.clip(raw,-1,1)
    error=(pred-y[:,2])**2;old_error=(p0[:,2]-y[:,2])**2;diff=error-old_error
    factor=2*(pred-y[:,2])/7*((raw>=-1)&(raw<=1));penfactor=factor*(diff>0)
    gradients=[]
    for f in (factor,penfactor,factor if coefficient==0 else factor+penfactor):
        if jobs:jobs.gradient()
        gradients.append(np.r_[np.mean(z*f[:,None],axis=0),f.mean()])
    gb,gp,g=gradients;norms=[float(np.linalg.norm(v)) for v in gradients];direction=-g
    if jobs:jobs.linear(n)
    tangent=np.einsum('ij,j->i',z,direction[:-1],optimize=False)+direction[-1]
    dp=np.where((raw>-1)&(raw<1),tangent,0.);dp=np.where(raw==-1,np.maximum(tangent,0),dp);dp=np.where(raw==1,np.minimum(tangent,0),dp)
    de=2*(pred-y[:,2])*dp;dh=np.where(diff>0,de,np.where(diff==0,np.maximum(de,0),0.))
    floor=float(np.mean(np.sum((p0[:,[0,1,3,4,5,6]]-y[:,[0,1,3,4,5,6]])**2,axis=1))/7)
    mse=float(error.mean());penalty=float(np.maximum(diff,0).mean()/7);trainable=mse/7 if coefficient==0 else mse/7+penalty
    rounded=pred.astype(np.float32).astype(np.float64);rounded_error=(rounded-y[:,2])**2
    anchor=floor+float(rounded_error.mean())/7+coefficient*float(np.maximum(rounded_error-old_error,0).mean()/7)
    clip=np.where(raw<-1,0,np.where(raw==-1,1,np.where(raw<1,2,np.where(raw==1,3,4)))).astype(np.uint8)
    hinge=np.where(diff<0,0,np.where(diff==0,1,2)).astype(np.uint8)
    row={'n':n,'fixed_other6_floor':floor,'component3_mse':mse,'penalty':penalty,'trainable_objective':trainable,
        'total_objective':floor+trainable,'base_l2':norms[0],'penalty_l2':norms[1],'total_l2':norms[2],
        'total_rms':norms[2]/math.sqrt(len(g)),'total_linf':float(np.max(abs(g))),
        'directional_derivative':float(np.mean(de+coefficient*dh)/7),
        'anchor_objective':anchor,'anchor_max_action_delta':float(np.max(abs(pred-rounded))),
        'anchor_harm_disagreements':int(np.count_nonzero((error>old_error)!=(rounded_error>old_error))),
        'branch_digest':hashlib.sha256(np.asarray(rows,dtype='<i4').tobytes()+clip.tobytes()+hinge.tobytes()).hexdigest()}
    for i,name in enumerate(['below','low_tie','inside','high_tie','above']):row['clip_'+name+'_n']=int((clip==i).sum())
    for i,name in enumerate(['negative','zero','positive']):row['hinge_'+name+'_n']=int((hinge==i).sum())
    row.update({k:v.tolist() for k,v in zip(VECTORS,[gb,gp,g,direction])})
    d.p.require(np.isfinite(theta).all() and np.isfinite(raw).all() and np.isfinite(tangent).all()
        and all(np.isfinite(v).all() for v in gradients) and math.isfinite(row['directional_derivative'])
        and row['anchor_max_action_delta']<=1e-7 and abs(anchor-row['total_objective'])<=1e-6,'invalid current-state numerics')
    if jobs:jobs.state_end(row)
    return row


def trial_objective(context,theta,coefficient,floor,jobs=None):
    z,y,p0=context
    if jobs:jobs.check_context(z,y,p0);jobs.linear(len(z))
    raw=p0[:,2]+(np.einsum('ij,j->i',z,theta[:-1],optimize=False)+theta[-1]);error=(np.clip(raw,-1,1)-y[:,2])**2
    old_error=(p0[:,2]-y[:,2])**2;trainable=float(error.mean())/7
    if coefficient!=0:trainable+=coefficient*float(np.maximum(error-old_error,0).mean()/7)
    d.p.require(np.isfinite(theta).all() and np.isfinite(raw).all() and math.isfinite(trainable),'nonfinite trial')
    return trainable,floor+trainable


def optimize(context,start,coefficient,rows,jobs,emit_state,emit_trial):
    first=None
    while True:
        theta=jobs.current;updates=jobs.updates;row=measure(context,theta,coefficient,rows,jobs)
        if first is None:first=row
        emit_state(updates,theta,row)
        reason=d.stopping_reason(updates,row['total_l2'],row['directional_derivative'])
        if reason:break
        for index in range(12):
            alpha=d.trial_alpha(index);trial=a.readonly(theta+alpha*np.asarray(row['direction']))
            jobs.trial_begin(index,trial);t,f=trial_objective(context,trial,coefficient,row['fixed_other6_floor'],jobs)
            accepted=jobs.trial_end(t)
            emit_trial(dict(from_extra_updates=updates,trial_index=index,alpha=alpha,trainable_objective=t,total_objective=f,
                slope=row['directional_derivative'],armijo_bound=row['trainable_objective']+.0001*alpha*row['directional_derivative'],accepted=accepted))
            if accepted:jobs.commit(trial);break
        else:reason='line_search_exhausted';break
    final=jobs.current;jobs.finish(reason);return first,row,final,reason


def summaries(heads):
    seeds=[];arms=[]
    for arm in d.p.ARMS:
        for mode in d.CASES:
            selected=[h for h in heads if h['arm']==arm and h['mode']==mode]
            for seed in d.p.SEEDS:
                part=sorted((h for h in selected if h['seed']==seed),key=lambda h:h['fold']);metrics={}
                for field in ('trainable_ratio','gradient_ratio'):
                    values=[h[field] for h in part];metrics[field]={'fold_values':values,'equal_fold_mean':math.fsum(values)/3 if None not in values else None}
                seeds.append(dict(arm=arm,mode=mode,seed=seed,metrics=metrics,updates=[h['extra_updates'] for h in part],all3fold_cue=all(h['joint_cue'] for h in part)))
            arms.append(dict(arm=arm,mode=mode,seed_all3fold_cues=[h['all3fold_cue'] for h in seeds if h['arm']==arm and h['mode']==mode],all9head_cue=d.joint_evidence(selected),automatic_Policy_promotion=False))
    return seeds,arms


def execute_payload(cfg,science_pin,binding_pin):
    a.validate_call(cfg,science_pin,binding_pin)
    directory=Path(cfg['artifacts']);output=Path(cfg['output']);d.p.require(not directory.exists() and not output.exists(),'existing run; no retry');directory.mkdir(parents=True)
    limits=a.base.Limits(300,536870912);runtime=a.runtime(cfg);started=a.base.now();jobs=a.Jobs(cfg,limits);work=a.RepresentationWork(cfg)
    paths={name:directory/(name+'.csv') for name in TABLES};newpaths=[directory/f'head-{i:03d}.npz' for i in range(36)]
    guard=a.Guard(cfg,paths.values(),newpaths,limits,jobs);sinks={};states=[];trials=[];endpoints=[];heads=[];parities=[];archives=[]
    try:
        with guard:
            shared=a.load(cfg['inputs']['shared'],'shared');y=shared['targets'];lengths=cfg['inputs']['action_lengths'];expected=[(i,t) for i,n in enumerate(lengths) for t in range(1,n)]
            d.p.require(y.dtype==np.float32 and y.shape==(4188,7) and np.isfinite(y).all() and (abs(y)<=1).all()
                and shared['demo_indices'].dtype==shared['action_steps'].dtype==shared['demo_lengths'].dtype==np.int32
                and list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==expected and shared['demo_lengths'].tolist()==lengths
                and shared['tail20'].dtype==np.bool_ and shared['tail20'].tolist()==[t>=lengths[i]-20 for i,t in expected],'source causal target schema differs')
            sinks={name:a.base.CsvSink(path,limits) for name,path in paths.items()}
            for model in cfg['inputs']['models']:
                cache=a.load(model['current_archive'],'current');f,p0=cache['features'],cache['predictions']
                d.p.require(f.dtype==p0.dtype==np.float32 and f.shape==(4188,256) and p0.shape==(4188,7) and np.isfinite(f).all() and np.isfinite(p0).all() and (abs(p0)<=1).all(),'source current schema differs')
                for plan in cfg['inputs']['folds']:
                    z=None;previous=None;rows=np.asarray(plan['fit_rows'],np.int32);rkey=(model['seed'],model['arm'],plan['fold'])
                    for mode,coefficient in d.CASES.items():
                        h=next(h for h in cfg['inputs']['heads'] if a.key(h)==rkey+(mode,));head=a.load(h['archive'],'head')
                        if z is None:
                            work.begin(rkey,len(rows));z=admit(head,f,plan);work.built(rkey,z);context=jobs.register_fit(model,plan,z,y,p0,rows,work)
                        else:admit_reused(head,plan,previous);work.reuse(rkey,z)
                        previous=(head['scaler_mean'].tobytes(),head['scaler_scale'].tobytes());ident=a.identity(h)
                        start=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];jobs.head(a.key(h),*context,rows,start)
                        def state_sink(k,theta,row):
                            entry={**ident,'extra_updates':k,'parameter_digest':a.digest(theta),**{k:v for k,v in row.items() if k not in VECTORS}}
                            states.append(entry);sinks['states'].add(entry)
                        def trial_sink(row):
                            entry={**ident,**row};trials.append(entry);sinks['trials'].add(entry)
                        initial,final,theta,reason=optimize(context,start,coefficient,rows,jobs,state_sink,trial_sink)
                        ref=next(r for r in cfg['reference_points'] if all(r[k]==v for k,v in ident.items()))
                        for left,right in [('total_objective','start500_total_objective'),('trainable_objective','start500_trainable_objective'),('total_l2','start500_gradient_l2')]:d.p.require(abs(initial[left]-ref[right])<=1e-10,'fixed500 parent readout differs')
                        d.p.require(initial['n']==ref['n'] and initial['branch_digest']==ref['start500_branch_digest'],'fixed500 branch differs')
                        updates=jobs.last_finished[3]
                        for condition,k,params,point in [('hold_500',0,start,initial),('fullbatch_armijo',updates,theta,final)]:
                            ep={**ident,'condition':condition,'extra_updates':k,**point,'parameters':params.tolist()};endpoints.append(ep)
                            sinks['endpoints'].add({k:v for k,v in ep.items() if k not in VECTORS+['parameters']})
                            for j in range(257):sinks['parameters'].add({**ident,'condition':condition,'coordinate':j,'parameter':float(params[j]),**{k:point[k][j] for k in VECTORS}})
                        evidence=d.endpoint_evidence(initial['trainable_objective'],final['trainable_objective'],initial['total_l2'],final['total_l2'])
                        summary={**ident,'extra_updates':updates,'trials':jobs.done[a.key(h)]['trials'],'terminal_reason':reason,
                            'initial_objective':initial['total_objective'],'final_objective':final['total_objective'],
                            'initial_trainable_objective':initial['trainable_objective'],'final_trainable_objective':final['trainable_objective'],
                            'initial_gradient_l2':initial['total_l2'],'final_gradient_l2':final['total_l2'],
                            'final_normalized_remaining_descent':-final['directional_derivative']/final['total_l2']/ref['zero_residual_gradient_l2'] if final['total_l2'] and ref['zero_residual_gradient_l2'] else 0. if ref['zero_residual_gradient_l2'] else None,**evidence}
                        heads.append(summary);sinks['heads'].add(summary)
                        index=next(i for i,item in enumerate(cfg['inputs']['heads']) if a.key(item)==a.key(h))
                        archive=np.savez(newpaths[index],fit_rows=rows,fit_demo_indices=head['fit_demo_indices'],scaler_mean=head['scaler_mean'],scaler_scale=head['scaler_scale'],
                            additional_updates=np.asarray([0,updates],np.int32),weights=np.stack([start[:-1],theta[:-1]]),bias=np.asarray([start[-1],theta[-1]],np.float64))
                        archives.append({**ident,**archive});del head
                    parity=dict(seed=model['seed'],arm=model['arm'],fold=plan['fold'],fit_rows=len(rows),saved_scalers_bit_equal=True)
                    parities.append(parity);sinks['scaler-parity'].add(parity);del z,context
                del cache,f,p0;print(json.dumps({'stage':'primary','seed':model['seed'],'arm':model['arm'],'updates':jobs.counts['updates']}),flush=True)
            jobs.complete();work.complete();guard.complete()
            for sink in sinks.values():sink.close()
        d.p.require(sum(p.stat().st_size for p in paths.values())<=67108864 and sum(p.stat().st_size for p in newpaths)<=67108864,'output size exceeded')
        expected_rows={'endpoints':72,'parameters':18504,'states':jobs.counts['gradient_states'],'trials':jobs.counts['trial_forwards'],'heads':36,'scaler-parity':18}
        d.p.require({k:s.rows for k,s in sinks.items()}==expected_rows and sum(expected_rows.values())<=42066,'table budget differs')
        seeds,arms=summaries(heads)
        result=dict(status='complete_finite_optimization_pending_audit',formal_result=False,science_config=science_pin,binding=binding_pin,parents=cfg['parents'],inputs=cfg['inputs'],budget=cfg['budget'],
            runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),computation=jobs.counts,representation_work=work.record(),
            tables={k:s.pin() for k,s in sinks.items()},new_heads=archives,states=states,trials=trials,endpoints=endpoints,heads=heads,scaler_parity=parities,seed_summaries=seeds,arm_summaries=arms,
            original_Policy_CNN_environment_or_sealed_calls=0,prior_joint_cues_G_failed_H_unrun_unchanged=True)
        a.base.write_new(output,result);return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        a.base.write_new(directory/'primary-failed-partial.json',dict(status='failed_no_retry',error=repr(error),access=guard.record(),computation=jobs.counts,representation_work=work.record()));raise


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(p.exists() for p in (output,launch,final,Path(cfg['artifacts']))),'existing run; no retry')
    record=dict(started_at_utc=a.base.now(),science_config=d.p.pin(config),binding=d.p.pin(binding),processes=[]);a.base.write_new(launch,record);started=time.monotonic();child=None
    try:
        for stage,module,seconds in [('primary','experiments.run_libero_task7_fixed_representation_optimization',315),('audit','experiments.audit_libero_task7_fixed_representation_optimization',615)]:
            env=os.environ.copy();env.update(bound['CPU_environment']);cmd=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            if stage=='primary':cmd+=['--mode','evaluate']
            row=dict(stage=stage,command=cmd,started_at_utc=a.base.now());record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(seconds,max(1,955-(time.monotonic()-started))))
                except BaseException:stop_child(child);raise
            row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-started<=960,'overall watchdog exceeded');record['status']='complete_independently_audited'
    except BaseException as error:
        if child is not None:stop_child(child)
        record['status']='failed_no_retry';record['error']=repr(error);raise
    finally:record['wall_seconds']=time.monotonic()-started;record['completed_at_utc']=a.base.now();a.base.write_new(final,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);parser.add_argument('--mode',choices=['preflight','evaluate','run'],default='preflight');args=parser.parse_args()
    if args.mode=='run':result=run(args.config,args.binding)
    else:
        cfg,_=a.preflight(args.config,args.binding);result=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding)) if args.mode=='evaluate' else {'status':'bound_no_payload'}
    print(json.dumps({'status':result['status']}))
