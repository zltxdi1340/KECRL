"""Paired guarded lambda1 copies with frozen coarse12 and fine20 trial grids."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments import libero_task7_mse_guard_finer_grid_protocol as d
from experiments import libero_task7_mse_guard_finer_grid_access as a
from experiments.run_libero_task7_fixed_representation_optimization import measure,admit,VECTORS
from experiments.analyze_libero_task7_component3_history_transition import stop_child

TABLES=['endpoints','parameters','states','trials','heads','pairs','scaler-parity']
METRICS=['component3_mse','penalty','trainable_objective','total_objective','total_l2']


def trial_objective(context,theta,coefficient,floor,jobs=None):
    d.p.require(coefficient==1.,'only frozen lambda1 allowed')
    z,y,p0=context
    if jobs:jobs.check_context(z,y,p0);jobs.linear(len(z))
    raw=p0[:,2]+(np.einsum('ij,j->i',z,theta[:-1],optimize=False)+theta[-1])
    error=(np.clip(raw,-1,1)-y[:,2])**2;old_error=(p0[:,2]-y[:,2])**2
    mse=float(error.mean());penalty=float(np.maximum(error-old_error,0).mean()/7)
    t=mse/7+penalty
    d.p.require(np.isfinite(theta).all() and np.isfinite(raw).all() and math.isfinite(t),'nonfinite trial')
    return dict(component3_mse=mse,penalty=penalty,trainable_objective=t,total_objective=floor+t)


def optimize(context,start,rows,jobs,emit_state,emit_trial):
    first=None
    while True:
        theta=jobs.current;updates=jobs.updates;row=measure(context,theta,1.,rows,jobs)
        if first is None:first=row
        emit_state(updates,theta,row)
        reason=d.stopping_reason(jobs.active[-1],updates,row['total_l2'],row['directional_derivative'])
        if reason:break
        for index in range(d.TRIAL_CAPS[jobs.active[-1]]):
            alpha=d.trial_alpha(jobs.active[-1],index);trial=a.readonly(theta+alpha*np.asarray(row['direction']))
            jobs.trial_begin(index,trial);values=trial_objective(context,trial,1.,row['fixed_other6_floor'],jobs)
            decision=jobs.trial_end(values)
            emit_trial(dict(from_extra_updates=updates,trial_index=index,alpha=alpha,**values,
                start500_component3_mse=jobs.fixed_mse,slope=row['directional_derivative'],
                armijo_bound=row['trainable_objective']+.0001*alpha*row['directional_derivative'],**decision))
            if decision['accepted']:jobs.commit(trial);break
        else:reason='line_search_exhausted';break
    final=jobs.current;jobs.finish(reason);return first,row,final,reason


def summaries(heads,pairs):
    seeds=[];sources=[];seed_pairs=[];source_pairs=[]
    fields=['final_component3_mse','final_penalty','final_trainable_objective','final_gradient_l2','trainable_ratio','gradient_ratio']
    for source in d.p.ARMS:
        for opt in d.ARMS:
            chosen=[h for h in heads if h['arm']==source and h['optimization_arm']==opt]
            d.p.require(len(chosen)==9,'incomplete source optimization group')
            for seed in d.p.SEEDS:
                part=sorted([h for h in chosen if h['seed']==seed],key=lambda h:h['fold']);metrics={}
                for field in fields:
                    values=[h[field] for h in part]
                    metrics[field]=dict(fold_values=values,equal_fold_mean=math.fsum(values)/3 if None not in values else None)
                seeds.append(dict(arm=source,optimization_arm=opt,seed=seed,metrics=metrics,
                    updates=[h['extra_updates'] for h in part],stops=[h['terminal_reason'] for h in part],
                    all3fold_joint_cue=all(h['joint_cue'] for h in part),all3fold_protected_cue=all(h['protected_joint_fit_cue'] for h in part)))
            sources.append(dict(arm=source,optimization_arm=opt,all9head_joint_cue=all(h['joint_cue'] for h in chosen),
                all9head_protected_cue=all(h['protected_joint_fit_cue'] for h in chosen),automatic_Policy_promotion=False))
        chosen=[p for p in pairs if p['arm']==source]
        pf=[k for k in chosen[0] if k.startswith('candidate_minus_baseline_')]
        for seed in d.p.SEEDS:
            part=sorted([r for r in chosen if r['seed']==seed],key=lambda r:r['fold'])
            seed_pairs.append(dict(arm=source,seed=seed,metrics={f:dict(fold_values=[r[f] for r in part],equal_fold_mean=math.fsum(r[f] for r in part)/3) for f in pf}))
        source_pairs.append(dict(arm=source,metrics={f:math.fsum(r[f] for r in chosen)/9 for f in pf}))
    seed_cohorts=[];source_cohorts=[]
    for source in d.p.ARMS:
        for cohort in ('zero','short','full50'):
            chosen=[r for r in pairs if r['arm']==source and r['cohort']==cohort]
            def group(rows):
                return dict(n=len(rows),grid_opened=sum(r['grid_opened'] for r in rows),
                    fold_values=[dict(fold=r['fold'],seed=r['seed'],coarse_updates=r['baseline_updates'],fine_updates=r['candidate_updates'],
                        mse_delta=r['candidate_minus_baseline_component3_mse'],T_delta=r['candidate_minus_baseline_trainable_objective']) for r in rows],
                    mean_MSE_delta=math.fsum(r['candidate_minus_baseline_component3_mse'] for r in rows)/len(rows) if rows else None,
                    mean_T_delta=math.fsum(r['candidate_minus_baseline_trainable_objective'] for r in rows)/len(rows) if rows else None)
            source_cohorts.append(dict(arm=source,cohort=cohort,**group(chosen)))
            for seed in d.p.SEEDS:
                seed_cohorts.append(dict(arm=source,seed=seed,cohort=cohort,**group([r for r in chosen if r['seed']==seed])))
    return dict(seed_arm_summaries=seeds,source_arm_summaries=sources,seed_paired_summaries=seed_pairs,
        source_paired_summaries=source_pairs,seed_cohort_summaries=seed_cohorts,source_cohort_summaries=source_cohorts,
        mechanism_opening_summary=d.opening_summary(pairs))


def execute_payload(cfg,science_pin,binding_pin):
    a.validate_call(cfg,science_pin,binding_pin)
    directory=Path(cfg['artifacts']);output=Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(),'existing run; no retry')
    history=a.History(cfg)
    directory.mkdir(parents=True);limits=a.base.Limits(300,536870912);runtime=a.runtime(cfg);started=a.base.now()
    jobs=a.Jobs(cfg,limits);work=a.RepresentationWork(cfg)
    paths={name:directory/(name+'.csv') for name in TABLES};newpaths=[directory/f'head-{i:03d}.npz' for i in range(18)]
    guard=a.Guard(cfg,paths.values(),newpaths,limits,jobs);sinks={};states=[];trials=[];endpoints=[];heads=[];pairs=[];parities=[];archives=[]
    try:
        with guard:
            shared=a.load(cfg['inputs']['shared'],'shared');y=shared['targets'];lengths=cfg['inputs']['action_lengths']
            expected=[(i,t) for i,n in enumerate(lengths) for t in range(1,n)]
            d.p.require(y.dtype==np.float32 and y.shape==(4188,7) and np.isfinite(y).all() and (abs(y)<=1).all()
                and shared['demo_indices'].dtype==shared['action_steps'].dtype==shared['demo_lengths'].dtype==np.int32
                and list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==expected
                and shared['demo_lengths'].tolist()==lengths and shared['tail20'].dtype==np.bool_
                and shared['tail20'].tolist()==[t>=lengths[i]-20 for i,t in expected],'source causal schema differs')
            sinks={name:a.base.CsvSink(path,limits) for name,path in paths.items()}
            for model in cfg['inputs']['models']:
                cache=a.load(model['current_archive'],'current');f,p0=cache['features'],cache['predictions']
                d.p.require(f.dtype==p0.dtype==np.float32 and f.shape==(4188,256) and p0.shape==(4188,7)
                    and np.isfinite(f).all() and np.isfinite(p0).all() and (abs(p0)<=1).all(),'source cache schema differs')
                for plan in cfg['inputs']['folds']:
                    rkey=(model['seed'],model['arm'],plan['fold']);h=next(h for h in cfg['inputs']['heads'] if a.key(h)==rkey+(d.source.MODE,))
                    head=a.load(h['archive'],'head');rows=np.asarray(plan['fit_rows'],np.int32)
                    work.begin(rkey,len(rows));z=admit(head,f,plan);work.built(rkey,z)
                    context=jobs.register_fit(model,plan,z,y,p0,rows,work)
                    start=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];jobs.source(h,head,start)
                    ident=a.identity(h);results={};points=[start];update_values=[];initials=[]
                    for opt in d.ARMS:
                        if opt==d.ARMS[1]:work.reuse(rkey,z)
                        pathkey=a.key(h)+(opt,);jobs.head(pathkey,*context,rows,start);pid={**ident,'optimization_arm':opt}
                        def state_sink(k,theta,row):
                            entry={**pid,'extra_updates':k,'parameter_digest':a.digest(theta),**{f:v for f,v in row.items() if f not in VECTORS}}
                            states.append(entry);sinks['states'].add(entry)
                            if opt==d.ARMS[0]:history.row('states',entry)
                        def trial_sink(row):
                            entry={**pid,**row};trials.append(entry);sinks['trials'].add(entry)
                            if opt==d.ARMS[0]:history.row('trials',entry)
                        first,final,theta,reason=optimize(context,start,rows,jobs,state_sink,trial_sink)
                        ref=next(r for r in cfg['reference_points'] if all(r[k]==v for k,v in ident.items()))
                        for field,value in ref['start500'].items():
                            d.p.require(first[field]==value if isinstance(value,str) else abs(first[field]-value)<=1e-10,'fixed500 reference '+field+' differs')
                        d.p.require(first['n']==ref['n'],'fixed500 fit rows differ')
                        initials.append(first);updates=jobs.last_finished[3];points.append(theta);update_values.append(updates)
                        ep_specs=([('saved500',0,start,first)] if opt==d.ARMS[0] else [])+[(opt+'_final',updates,theta,final)]
                        for condition,k,params,point in ep_specs:
                            ep={**ident,'condition':condition,'extra_updates':k,**point,'parameters':params.tolist()};endpoints.append(ep)
                            sinks['endpoints'].add({f:v for f,v in ep.items() if f not in VECTORS+['parameters']})
                            for j in range(257):sinks['parameters'].add({**ident,'condition':condition,'coordinate':j,'parameter':float(params[j]),**{f:point[f][j] for f in VECTORS}})
                            if opt==d.ARMS[0]:history.row('endpoints',ep)
                        evidence=d.source.endpoint_evidence(first['trainable_objective'],final['trainable_objective'],first['total_l2'],final['total_l2'],
                            first['component3_mse'],final['component3_mse'],updates,d.source.ARMS[1])
                        done=jobs.done[pathkey];accepted=[r['alpha'] for r in trials if all(r[k]==v for k,v in pid.items()) and r['accepted']]
                        summary={**pid,'extra_updates':updates,'trials':done['trials'],'terminal_reason':reason,
                            'initial_objective':first['total_objective'],'final_objective':final['total_objective'],
                            'initial_trainable_objective':first['trainable_objective'],'final_trainable_objective':final['trainable_objective'],
                            'initial_gradient_l2':first['total_l2'],'final_gradient_l2':final['total_l2'],
                            'initial_component3_mse':first['component3_mse'],'final_component3_mse':final['component3_mse'],
                            'initial_penalty':first['penalty'],'final_penalty':final['penalty'],
                            'mse_vetoes_of_armijo_passes':done['mse_vetoes'],
                            'alpha_min':min(accepted) if accepted else None,'alpha_median':float(np.median(accepted)) if accepted else None,'alpha_max':max(accepted) if accepted else None,
                            'exhaustion_class':('armijo_feasible_but_mse_vetoed' if done['round_summaries'][-1]['armijo_passes'] else 'no_original_armijo_trial') if reason=='line_search_exhausted' else 'not_exhausted',**evidence}
                        heads.append(summary);results[opt]=summary;sinks['heads'].add(summary)
                        if opt==d.ARMS[0]:history.row('heads',summary)
                    d.p.require(initials[0]==initials[1],'paired initial readout differs')
                    baseline,candidate=(results[opt] for opt in d.ARMS)
                    paired={**ident,**{'candidate_minus_baseline_'+field:candidate['final_'+field]-baseline['final_'+field]
                        for field in ['component3_mse','penalty','trainable_objective','gradient_l2']},
                        'baseline_updates':baseline['extra_updates'],'candidate_updates':candidate['extra_updates'],
                        'baseline_stop':baseline['terminal_reason'],'candidate_stop':candidate['terminal_reason'],
                        'candidate_finite_fit_progress':candidate['finite_fit_progress'],'candidate_protected_joint_fit_cue':candidate['protected_joint_fit_cue']}
                    cohort=next(r for r in cfg['cohorts'] if all(r[k]==v for k,v in ident.items()))
                    d.p.require(baseline['extra_updates']==cohort['prior_coarse_updates'] and baseline['terminal_reason']==cohort['prior_coarse_stop'],
                        'coarse count/stop differs from fixed cohort source')
                    paired.update({k:cohort[k] for k in ('cohort','prior_coarse_updates','prior_coarse_stop')})
                    paired.update(jobs.prefixes[a.key(h)])
                    paired.update(fine_minus_start_component3_mse=candidate['final_component3_mse']-candidate['initial_component3_mse'],
                        coarse_updates=baseline['extra_updates'],fine_updates=candidate['extra_updates'],
                        fine_minus_coarse_trials=candidate['trials']-baseline['trials'])
                    pairs.append(paired);sinks['pairs'].add(paired)
                    index=next(i for i,x in enumerate(cfg['inputs']['heads']) if a.key(x)==a.key(h))
                    pin=np.savez(newpaths[index],fit_rows=rows,fit_demo_indices=head['fit_demo_indices'],scaler_mean=head['scaler_mean'],scaler_scale=head['scaler_scale'],
                        additional_updates=np.asarray([0,*update_values],np.int32),weights=np.stack([v[:-1] for v in points]),bias=np.asarray([v[-1] for v in points],np.float64))
                    archives.append({**ident,**pin})
                    parity=dict(seed=model['seed'],arm=model['arm'],fold=plan['fold'],fit_rows=len(rows),paired_representation_same_object=True,source500_start_same_parameters=True)
                    parities.append(parity);sinks['scaler-parity'].add(parity);del z,context,head
                del cache,f,p0;print(json.dumps({'stage':'primary','seed':model['seed'],'arm':model['arm'],'updates':jobs.counts['updates']}),flush=True)
            jobs.complete();work.complete();guard.complete();historical=history.complete()
            for sink in sinks.values():sink.close()
        d.p.require(sum(p.stat().st_size for p in paths.values())<=67108864 and sum(p.stat().st_size for p in newpaths)<=67108864,'output size exceeded')
        expected_rows={'endpoints':54,'parameters':13878,'states':jobs.counts['gradient_states'],'trials':jobs.counts['trial_forwards'],'heads':36,'pairs':18,'scaler-parity':18}
        d.p.require({k:s.rows for k,s in sinks.items()}==expected_rows and sum(expected_rows.values())<=cfg['budget']['output_CSV_rows_max'],'table budget differs')
        result=dict(status='complete_mse_guard_finer_grid_pending_audit',formal_result=False,science_config=science_pin,binding=binding_pin,parents=cfg['parents'],inputs=cfg['inputs'],budget=cfg['budget'],
            runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),computation=jobs.counts,representation_work=work.record(),historical_coarse_parity=historical,trial_counts_by_arm=jobs.trials_by_arm,
            tables={k:s.pin() for k,s in sinks.items()},new_heads=archives,states=states,trials=trials,endpoints=endpoints,heads=heads,pairs=pairs,scaler_parity=parities,**summaries(heads,pairs),
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
        for stage,module,seconds in [('primary','experiments.run_libero_task7_mse_guard_finer_grid',315),('audit','experiments.audit_libero_task7_mse_guard_finer_grid',615)]:
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
