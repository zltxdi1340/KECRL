"""Bound P prefix reconstruction and P/C/V watchdog. No research default run."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from experiments import libero_task7_joint_direction_prefix_review_protocol as d
from experiments import libero_task7_joint_direction_prefix_review_access as a
from experiments import run_libero_task7_joint_direction_diagnostic as numerical
from experiments.run_libero_task7_endpoint_gradient_diagnostic import admit  # preload before IO guard
from experiments.analyze_libero_task7_component3_history_transition import stop_child


def reconstruct(cfg,jobs,context,start,rows,state_sink,trial_sink,events,latest):
    states=[];trials=[];parameters=[];points=[];theta=start;trial_cursor=0;reason=None
    for k in range(36):
        jobs.select('P',k,theta);events.add('P',k,'before',jobs,theta)
        point=numerical.measure(context,theta,1.,rows,jobs)
        entry={**a.IDENT,'extra_updates':k,'parameter_digest':a.digest(theta),**{f:v for f,v in point.items() if f not in a.VECTORS}}
        # Evidence is written before historical comparison can throw.
        latest.clear();latest.update(role='P',step=k,state=entry,parameters=theta.tolist(),vectors={f:point[f] for f in a.VECTORS})
        parameters.append(theta.copy());points.append(point);states.append(entry);state_sink.add(entry)
        state_sink.handle.flush();events.add('P',k,'after',jobs,theta,jobs.last_direction)
        a.anchor(cfg['historical_prefix']['states'][k],entry)
        reason=a.stop(k,point)
        if reason:break
        for j in range(20):
            trial=a.readonly(theta+2.**(-j)*np.asarray(point['direction']))
            jobs.trial_begin(j,trial);values=numerical.trial_objective(context,trial,1.,point['fixed_other6_floor'],jobs)
            decision=jobs.trial_end(values)
            row={**a.IDENT,'from_extra_updates':k,'trial_index':j,'alpha':2.**(-j),**values,
                 'start500_component3_mse':jobs.fixed_mse,'slope':point['directional_derivative'],
                 'current_component3_mse':point['component3_mse'],'delta_component3_mse':values['component3_mse']-point['component3_mse'],
                 'armijo_bound':point['trainable_objective']+.0001*2.**(-j)*point['directional_derivative'],**decision}
            trials.append(row);trial_sink.add(row);latest['last_trial']=row;latest['last_trial_parameters']=trial.tolist()
            a.require(trial_cursor<len(cfg['historical_prefix']['trials']),'P exceeded historical trial prefix')
            a.anchor(cfg['historical_prefix']['trials'][trial_cursor],row);trial_cursor+=1
            if decision['accepted']:jobs.commit(trial);theta=trial;break
        else:reason='line_search_exhausted';break
    a.require(reason=='diagnostic_prefix_cap' and len(states)==36 and trial_cursor==len(cfg['historical_prefix']['trials']),
              'P historical prefix stopped early')
    return states,trials,parameters,points,reason


def execute_payload(cfg,sp,bp):
    a.validate_call(cfg,sp,bp)
    directory=Path(cfg['artifacts']);output=Path(cfg['output'])
    a.require(not directory.exists() and not output.exists(),'existing run; no retry')
    runtime=a.runtime(cfg,'primary');directory.mkdir(parents=True)
    cap=cfg['execution']['stages']['primary'];limits=a.Limits(cap['seconds'],cap['RSS_bytes'])
    jobs=a.Jobs(cfg,'primary',limits);paths={name:directory/(name+'.csv') for name in ('primary-states','primary-trials')}
    event_path=directory/'primary-events.jsonl'
    guard=a.NumericGuard(cfg,'primary',jobs,limits,[*paths.values(),event_path]);sinks={};events=None;latest={}
    started=a.base.now()
    try:
        sinks={name:a.Sink(path,cfg['output_columns'][name.replace('-','_')],cfg['budget']['CSV_rows'][name.replace('-','_')]) for name,path in paths.items()}
        events=a.Events(event_path,'primary',limits,74);events.add(None,None,'start',jobs)
        with guard:
            context,start,rows,head=a.admit_inputs(cfg,guard,jobs)
            states,trials,parameters,points,reason=reconstruct(cfg,jobs,context,start,rows,sinks['primary-states'],sinks['primary-trials'],events,latest)
            jobs.complete()
            arrays=dict(steps=np.arange(36,dtype=np.int32),fit_rows=rows,fit_demo_indices=head['fit_demo_indices'],scaler_mean=head['scaler_mean'],scaler_scale=head['scaler_scale'],
                        parameters=np.stack(parameters),**{f:np.asarray([p[f] for p in points],np.float64) for f in a.VECTORS})
            trace=np.savez(directory/'primary-trace.npz',**arrays)
            vector_digests=[{f:a.digest(p[f]) for f in a.VECTORS} for p in points]
            guard.complete();events.add(None,None,'end',jobs)
            for sink in sinks.values():sink.close()
            events.close()
        tables={name:s.pin() for name,s in sinks.items()}
        a.require(sum(Path(p['path']).stat().st_size for p in tables.values())<=cfg['execution']['new_CSV_bytes_max'],'CSV size exceeded')
        result=dict(status='complete_primary_prefix_pending_cross_and_audit',formal_result=False,science_config=sp,binding=bp,
                    started_at_utc=started,completed_at_utc=a.base.now(),runtime=runtime,cost=limits.record(),ledger=jobs.record(),access=guard.record(),
                    states=states,trials=trials,vector_digests=vector_digests,context_digests=jobs.context_digests,
                    trace=trace,tables=tables,events=events.pin(),stop=reason,parent_failed_run_remains_failed=True)
        a.base.write_new(directory/'primary-result.json',result);return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        if events and not events.handle.closed:events.close()
        a.write_failure(directory/'primary-failed-partial.json','primary',error,jobs,guard,limits,latest);raise


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);directory=Path(cfg['artifacts'])
    launch=output.with_name(output.stem+'-launch.json');process=output.with_name(output.stem+'-process.json')
    a.require(not any(p.exists() for p in (output,directory,launch,process)),'existing run; no retry')
    record=dict(started_at_utc=a.base.now(),science_config=d.p.pin(config),binding=d.p.pin(binding),processes=[])
    a.base.write_new(launch,record);started=time.monotonic();child=None
    modules=[('primary','experiments.run_libero_task7_joint_direction_prefix_numerical_review'),
             ('cross','experiments.run_libero_task7_joint_direction_prefix_cross_review'),
             ('audit','experiments.audit_libero_task7_joint_direction_prefix_numerical_review')]
    try:
        for stage,module in modules:
            env=os.environ.copy();env.update(bound['CPU_environment'])
            cmd=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding),'--mode','evaluate']
            row=dict(stage=stage,command=cmd,started_at_utc=a.base.now());record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(cfg['execution']['stages'][stage]['seconds']+30,max(1,715-(time.monotonic()-started))))
                except BaseException:stop_child(child);raise
            row['completed_at_utc']=a.base.now();a.require(row['exit_code']==0,stage+' failed; no retry')
        a.require(time.monotonic()-started<=720,'overall watchdog exceeded');record['status']='complete_numerical_observations_independently_checked'
    except BaseException as error:
        if child is not None:stop_child(child)
        record.update(status='failed_no_retry',error=repr(error));raise
    finally:
        record.update(wall_seconds=time.monotonic()-started,completed_at_utc=a.base.now());a.base.write_new(process,record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);p.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    p.add_argument('--mode',choices=['preflight','evaluate','run'],default='preflight');args=p.parse_args()
    if args.mode=='run':out=run(args.config,args.binding)
    else:
        cfg,bound=a.preflight(args.config,args.binding)
        out=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding)) if args.mode=='evaluate' else {'status':'bound_no_payload'}
    print(json.dumps({'status':out['status']}))
