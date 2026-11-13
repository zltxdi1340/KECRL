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
from experiments import libero_task7_endpoint_gradient_protocol_v2 as d
from experiments import libero_task7_endpoint_gradient_access_v2 as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child

from experiments.run_libero_task7_endpoint_gradient_diagnostic import measure, admit, pairs_and_summaries, VECTOR_FIELDS


def admit_reused(head, plan, previous_scaler):
    for field, values in [('fit_rows', np.asarray(plan['fit_rows'], np.int32)), ('fit_demo_indices', np.asarray(plan['fit_demos'], np.int32)), ('updates', np.asarray([0,100,250,500], np.int32)), ('optimizer_steps', np.arange(1,501,dtype=np.int32))]:
        d.p.require(head[field].dtype == np.int32 and np.array_equal(head[field], values), 'paired head fit/update role differs')
    d.p.require(head['weights'].shape == (4,256) and head['weights'].dtype == np.float32 and head['bias'].shape == (4,) and head['bias'].dtype == np.float32
        and (head['weights'][0] == 0).all() and head['bias'][0] == 0 and np.isfinite(head['weights'][[0,3]]).all() and np.isfinite(head['bias'][[0,3]]).all(), 'paired head endpoint differs')
    d.p.require(all(head[k].shape == (256,) and head[k].dtype == np.float64 for k in ('scaler_mean','scaler_scale'))
        and (head['scaler_mean'].tobytes(), head['scaler_scale'].tobytes()) == previous_scaler, 'paired saved scaler differs')


def execute_payload(cfg,science_pin,binding_pin):
    output=Path(cfg['output']);directory=Path(cfg['artifacts']);d.p.require(not output.exists() and not directory.exists(),'existing run; no retry');directory.mkdir(parents=True)
    limits=a.base.Limits(180);runtime=a.runtime(cfg);started=a.base.now();jobs=a.Jobs(cfg);work=a.RepresentationWork(cfg)
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
                    rows=np.asarray(plan['fit_rows'],np.int32);previous_scaler=None;z=None
                    for mode,lam in d.CASES.items():
                        manifest=next(h for h in cfg['inputs']['heads'] if (h['seed'],h['arm'],h['fold'],h['mode'])==(model['seed'],model['arm'],plan['fold'],mode))
                        head=a.load(manifest['archive'],'head');representation_key=(model['seed'],model['arm'],plan['fold'])
                        if z is None:
                            work.begin(representation_key,len(rows));z=admit(head,f,plan);work.built(representation_key,z)
                        else:
                            admit_reused(head,plan,previous_scaler);work.reuse(representation_key,z)
                        scaler=(head['scaler_mean'].tobytes(),head['scaler_scale'].tobytes())
                        d.p.require(previous_scaler is None or previous_scaler==scaler,'paired scaler bits differ');previous_scaler=scaler
                        for pos,point in ((0,0),(3,500)):
                            key=(model['seed'],model['arm'],plan['fold'],mode,point);jobs.begin(key,len(rows))
                            record={'task':7,'seed':model['seed'],'arm':model['arm'],'fold':plan['fold'],'mode':mode,'point':point,
                                **measure(z,y[rows].astype(np.float64),p0[rows].astype(np.float64),head['weights'][pos].astype(np.float64),float(head['bias'][pos]),lam,rows,jobs)}
                            jobs.end();points.append(record);sinks['points'].add({k:v for k,v in record.items() if k not in VECTOR_FIELDS})
                            param=np.r_[head['weights'][pos].astype(np.float64),float(head['bias'][pos])]
                            for j in range(257):sinks['parameters'].add({**{k:record[k] for k in ('task','seed','arm','fold','mode','point')},'coordinate':j,'parameter':float(param[j]),**{k:record[k][j] for k in VECTOR_FIELDS}})
                        del head
                    del z
                del cache,f,p0;limits.check();print(json.dumps({'stage':'primary','seed':model['seed'],'arm':model['arm']}),flush=True)
            pairs,initial,seeds,arms=pairs_and_summaries(points)
            for row in pairs:sinks['pairs'].add(row)
            for row in initial:sinks['initial-parity'].add(row)
            for sink in sinks.values():sink.close()
            d.p.require({k:v.rows for k,v in sinks.items()}==cfg['budget']['tables'] and sum(p.stat().st_size for p in paths.values())<=67108864,'output budget differs');jobs.complete();work.complete();guard.complete()
        result={'status':'complete_fit_endpoint_gradient_pending_audit','formal_result':False,'science_config':science_pin,'binding':binding_pin,'parents':cfg['parents'],'inputs':cfg['inputs'],'budget':cfg['budget'],
            'runtime':runtime,'cost':limits.record(),'started_at_utc':started,'completed_at_utc':a.base.now(),'access':guard.record(),'computation':jobs.counts,'representation_work':work.record(),
            'tables':{k:v.pin() for k,v in sinks.items()},'points':points,'pairs':pairs,'initial_parity':initial,'seed_summaries':seeds,'arm_summaries':arms,
            'parameter_updates_original_Policy_CNN_environment_or_sealed_calls':0,'G_failed_H_unrun_and_prior_joint_cues_unchanged':True}
        a.base.write_new(output,result);return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        a.base.write_new(directory/'primary-failed-partial.json',{'status':'failed_no_retry','error':repr(error),'access':guard.record(),'computation':jobs.counts,'representation_work':work.record()});raise


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(p.exists() for p in (output,launch,final,Path(cfg['artifacts']))),'existing run; no retry')
    record={'started_at_utc':a.base.now(),'science_config':d.p.pin(config),'binding':d.p.pin(binding),'processes':[]};a.base.write_new(launch,record);started=time.monotonic();child=None
    try:
        for stage,module in [('primary','experiments.run_libero_task7_endpoint_gradient_diagnostic_v2'),('audit','experiments.audit_libero_task7_endpoint_gradient_diagnostic_v2')]:
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
