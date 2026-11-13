"""Full18 NumPy direction comparison v2; scientific optimization unchanged."""
from __future__ import annotations
import argparse,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
from experiments import libero_task7_joint_direction_protocol_v2 as d
from experiments import libero_task7_joint_direction_access_v2 as a
from experiments.run_libero_task7_joint_direction_diagnostic import measure,trial_objective,optimize,summaries,paired_reference,TABLES,VECTORS,admit
from experiments.analyze_libero_task7_component3_history_transition import stop_child

def execute_payload(cfg,science_pin,binding_pin):
    a.validate_call(cfg,science_pin,binding_pin)
    directory=Path(cfg['artifacts']);output=directory/'primary-result.json'
    d.p.require(not directory.exists() and not output.exists(),'existing run; no retry')
    history=a.History(cfg)
    directory.mkdir(parents=True);limits=a.Limits(cfg['execution']['primary_seconds'],cfg['execution']['primary_RSS_bytes']);runtime=a.runtime(cfg);started=a.base.now()
    jobs=a.TraceJobs(cfg,limits);work=a.RepresentationWork(cfg)
    paths={name:directory/(name+'.csv') for name in TABLES};newpaths=[directory/f'head-{i:03d}.npz' for i in range(18)]
    guard=a.Guard(cfg,paths.values(),newpaths,limits,jobs);sinks={};states=[];trials=[];endpoints=[];heads=[];pairs=[];parities=[];archives=[];contexts=[]
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
                    rkey=(model['seed'],model['arm'],plan['fold']);h=next(h for h in cfg['inputs']['heads'] if a.key(h)==rkey+(d.finer.source.MODE,))
                    head=a.load(h['archive'],'head');rows=np.asarray(plan['fit_rows'],np.int32)
                    work.begin(rkey,len(rows));z=admit(head,f,plan);work.built(rkey,z)
                    context=jobs.register_fit(model,plan,z,y,p0,rows,work)
                    contexts.append({**a.identity(h),"digests":[a.digest(v) for v in context]})
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
                        evidence=d.finer.source.endpoint_evidence(first['trainable_objective'],final['trainable_objective'],first['total_l2'],final['total_l2'],
                            first['component3_mse'],final['component3_mse'],updates,d.finer.source.ARMS[1])
                        done=jobs.done[pathkey];accepted=[r['alpha'] for r in trials if all(r[k]==v for k,v in pid.items()) and r['accepted']]
                        summary={**pid,'extra_updates':updates,'trials':done['trials'],'terminal_reason':reason,
                            'initial_objective':first['total_objective'],'final_objective':final['total_objective'],
                            'initial_trainable_objective':first['trainable_objective'],'final_trainable_objective':final['trainable_objective'],
                            'initial_gradient_l2':first['total_l2'],'final_gradient_l2':final['total_l2'],
                            'initial_component3_mse':first['component3_mse'],'final_component3_mse':final['component3_mse'],
                            'initial_penalty':first['penalty'],'final_penalty':final['penalty'],
                            'accepted_MSE_step_increases':sum(v>0 for v in done['mse_deltas']),
                            'accepted_positive_MSE_sum':math.fsum(v for v in done['mse_deltas'] if v>0),
                            'final_direction_status':final['direction_status'],'final_exact_MSE_slope':final['exact_MSE_slope'],
                            'mse_vetoes_of_armijo_passes':done['mse_vetoes'],
                            'alpha_min':min(accepted) if accepted else None,'alpha_median':float(np.median(accepted)) if accepted else None,'alpha_max':max(accepted) if accepted else None,
                            'exhaustion_class':('armijo_feasible_but_mse_vetoed' if done['round_summaries'][-1]['armijo_passes'] else 'no_original_armijo_trial') if reason=='line_search_exhausted' else 'not_exhausted',**evidence}
                        heads.append(summary);results[opt]=summary;sinks['heads'].add(summary)
                        if opt==d.ARMS[0]:history.row('heads',summary)
                    d.p.require(all(initials[0][f]==initials[1][f] for f in [*VECTORS[:3],'component3_mse','penalty','trainable_objective','total_objective','total_l2','branch_digest']),'paired500 fit/gradient readout differs')
                    baseline,candidate=(results[opt] for opt in d.ARMS)
                    paired={**ident,**{'candidate_minus_baseline_'+field:candidate['final_'+field]-baseline['final_'+field]
                        for field in ['component3_mse','penalty','trainable_objective','gradient_l2']},
                        'baseline_updates':baseline['extra_updates'],'candidate_updates':candidate['extra_updates'],
                        'baseline_stop':baseline['terminal_reason'],'candidate_stop':candidate['terminal_reason'],
                        'candidate_finite_fit_progress':candidate['finite_fit_progress'],'candidate_protected_joint_fit_cue':candidate['protected_joint_fit_cue']}
                    cohort=next(r for r in cfg['cohorts'] if all(r[k]==v for k,v in ident.items()))
                    d.p.require(baseline['extra_updates']==cohort['prior_control_updates'] and baseline['terminal_reason']==cohort['prior_control_stop'],
                        'control count/stop differs from fixed cohort source')
                    paired.update({k:cohort[k] for k in ('cohort','prior_control_updates','prior_control_stop')})
                    paired.update(paired_reference(initials[0],baseline,candidate,jobs.done[a.key(h)+(d.ARMS[1],)]['mse_deltas']))
                    paired.update(candidate_minus_control_trials=candidate['trials']-baseline['trials'],source500_parameters_same=True)
                    pairs.append(paired);sinks['pairs'].add(paired)
                    index=next(i for i,x in enumerate(cfg['inputs']['heads']) if a.key(x)==a.key(h))
                    pin=np.savez(newpaths[index],fit_rows=rows,fit_demo_indices=head['fit_demo_indices'],scaler_mean=head['scaler_mean'],scaler_scale=head['scaler_scale'],
                        additional_updates=np.asarray([0,*update_values],np.int32),weights=np.stack([v[:-1] for v in points]),bias=np.asarray([v[-1] for v in points],np.float64),**a.trace_arrays(jobs.traces[a.key(h)]))
                    archives.append({**ident,**pin})
                    parity=dict(seed=model['seed'],arm=model['arm'],fold=plan['fold'],fit_rows=len(rows),paired_representation_same_object=True,source500_start_same_parameters=True)
                    parities.append(parity);sinks['scaler-parity'].add(parity);del z,context,head
                del cache,f,p0;print(json.dumps({'stage':'primary','seed':model['seed'],'arm':model['arm'],'updates':jobs.counts['updates']}),flush=True)
            jobs.complete();work.complete();guard.complete();historical=history.complete()
            for sink in sinks.values():sink.close()
        d.p.require(sum(p.stat().st_size for p in paths.values())<=67108864 and sum(p.stat().st_size for p in newpaths)<=cfg['execution']['trace_NPZ_bytes_per_stage_max'],'output size exceeded')
        expected_rows={'endpoints':54,'parameters':13878,'states':jobs.counts['gradient_states'],'trials':jobs.counts['trial_forwards'],'heads':36,'pairs':18,'scaler-parity':18}
        d.p.require({k:s.rows for k,s in sinks.items()}==expected_rows and sum(expected_rows.values())<=cfg['budget']['output_CSV_rows_max'],'table budget differs')
        result=dict(status='complete_joint_direction_v2_pending_same_parameter_audit',formal_result=False,science_config=science_pin,binding=binding_pin,parents=cfg['parents'],inputs=cfg['inputs'],budget=cfg['budget'],
            runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),computation=jobs.counts,representation_work=work.record(),direction_work=jobs.direction_work.counts,historical_control_parity=historical,trial_counts_by_arm=jobs.trials_by_arm,
            tables={k:s.pin() for k,s in sinks.items()},new_heads=archives,context_digests=contexts,trace_coordinate_copies=jobs.trace_coordinate_copies,states=states,trials=trials,endpoints=endpoints,heads=heads,pairs=pairs,scaler_parity=parities,**summaries(heads,pairs),
            original_Policy_CNN_environment_or_sealed_calls=0,prior_joint_cues_G_failed_H_unrun_unchanged=True)
        a.base.write_new(output,result);return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        a.fail_record(directory/'primary-failed-partial.json',error,jobs,None,guard,limits);raise


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    started=time.monotonic();cfg,bound=a.preflight(config,binding);output=Path(cfg['output'])
    launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(p.exists() for p in (output,launch,final,Path(cfg['artifacts']))),'existing v2 run; no retry')
    record=dict(started_at_utc=a.base.now(),science_config=d.p.pin(config),binding=d.p.pin(binding),processes=[]);a.base.write_new(launch,record);child=None
    try:
        for stage,module,cap in [('primary','experiments.run_libero_task7_joint_direction_diagnostic_v2',cfg['execution']['primary_seconds']),('audit','experiments.audit_libero_task7_joint_direction_diagnostic_v2',cfg['execution']['audit_seconds'])]:
            env=os.environ.copy();env.update(bound['CPU_environment']);cmd=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            if stage=='primary':cmd+=['--mode','evaluate']
            row=dict(stage=stage,command=cmd,started_at_utc=a.base.now());record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(cap+60,max(1,cfg['execution']['overall_seconds']-5-(time.monotonic()-started))))
                except BaseException:stop_child(child);raise
            row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-started<=cfg['execution']['overall_seconds'],'v2 overall watchdog exceeded');record['status']='complete_independently_same_parameter_audited_direction_v2'
    except BaseException as error:
        if child is not None:stop_child(child)
        record.update(status='failed_no_retry',error=repr(error));raise
    finally:
        record.update(wall_seconds=time.monotonic()-started,completed_at_utc=a.base.now());a.base.write_new(final,record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);p.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    p.add_argument('--mode',choices=['preflight','evaluate','run'],default='preflight');args=p.parse_args()
    if args.mode=='run':result=run(args.config,args.binding)
    elif args.mode=='evaluate':
        cfg=json.loads(args.config.read_text());result=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding))
    else:a.preflight(args.config,args.binding);result={'status':'bound_no_payload'}
    print(json.dumps({'status':result['status']}))
