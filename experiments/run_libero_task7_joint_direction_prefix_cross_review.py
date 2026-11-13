"""U independent Torch path, S same-parameter gradients, N same-gradient norms."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
from experiments import libero_task7_joint_direction_prefix_review_protocol as d
from experiments import libero_task7_joint_direction_prefix_review_access as a
from experiments import audit_libero_task7_joint_direction_diagnostic as numerical
from experiments.audit_libero_task7_fixed_representation_optimization import TorchFit
from experiments.audit_libero_task7_endpoint_gradient_diagnostic import prepare  # preload


def independent_path(cfg,jobs,backend,start,rows,sink,events,latest):
    points=[];parameters=[];trials=[];theta=start;reason=None
    for k in range(36):
        jobs.select('U',k,theta);events.add('U',k,'before',jobs,theta)
        point=numerical.measure(backend,theta,1.,rows,jobs)
        points.append(point);parameters.append(theta.copy())
        latest.clear();latest.update(role='U',step=k,parameters=theta.tolist(),point=point)
        events.add('U',k,'after',jobs,theta,jobs.last_direction)
        reason=a.stop(k,point)
        if reason:break
        for j in range(20):
            trial=a.readonly(theta+2.**(-j)*np.asarray(point['direction']))
            jobs.trial_begin(j,trial);values=numerical.trial_objective(backend,trial,1.,point['fixed_other6_floor'],jobs)
            decision=jobs.trial_end(values)
            row={**a.IDENT,'from_extra_updates':k,'trial_index':j,'alpha':2.**(-j),**values,
                 'start500_component3_mse':jobs.fixed_mse,'slope':point['directional_derivative'],
                 'current_component3_mse':point['component3_mse'],'delta_component3_mse':values['component3_mse']-point['component3_mse'],
                 'armijo_bound':point['trainable_objective']+.0001*2.**(-j)*point['directional_derivative'],**decision}
            sink.add(row);trials.append(row);latest['last_trial']=row;latest['last_trial_parameters']=trial.tolist()
            if decision['accepted']:jobs.commit(trial);theta=trial;break
        else:reason='line_search_exhausted';break
    jobs.finished_U=True
    return points,parameters,trials,reason


def comparison(k,P,S,U,N,thetaP,thetaU,stop_U,counters):
    u=U is not None
    row={**d.FOCUS,'step':k,'U_available':u,'U_stop':stop_U}
    for role,point,theta in [('P',P,thetaP),('S',S,thetaP),('U',U,thetaU)]:
        # Scalar columns are projected by the caller's frozen schema, not old vectors.
        values={} if point is None else dict(point,parameter_digest=a.digest(theta))
        for field in comparison.point_columns:row[role+'.'+field]=values.get(field)
    nm=['status','total_norm','base_norm','unit_sum_norm','direction_norm','selected_T_slope','selected_MSE_slope','selected_joint','input_Gb_digest','input_GT_digest']
    row.update({'N.'+f:N[f] for f in nm})
    qp=P['direction_unit_sum_norm'];qn=N['unit_sum_norm'];qs=S['direction_unit_sum_norm'];qu=U['direction_unit_sum_norm'] if u else None
    row.update({role+'_q_hex':float(q).hex() if q is not None else None for role,q in [('P',qp),('N',qn),('S',qs),('U',qu)]})
    row.update({'q_'+f:v for f,v in d.q_chain(qp,qn,qs,qu).items()})
    raw={'theta_P':thetaP.tolist(),'theta_U':thetaU.tolist() if u else None,'d_N':N['direction']}
    for role,point in [('P',P),('S',S),('U',U)]:
        for name,field in zip(('Gb','Gp','GT','d'),a.VECTORS):raw[name+'_'+role]=point[field] if point is not None else None
    references={}
    for name,(left,right) in a.ERROR_EDGES.items():
        if raw[left] is None or raw[right] is None:
            row.update({name+'_'+f:None for f in ('l2','linf','relative_l2')});continue
        delta=[b-a0 for a0,b in zip(raw[left],raw[right])]
        l2=math.hypot(*delta);linf=max(abs(x) for x in delta)
        if left not in references:references[left]=math.hypot(*raw[left]);counters['reference_norms']+=1
        denom=references[left]
        row.update({name+'_l2':l2,name+'_linf':linf,name+'_relative_l2':l2/denom if denom else None})
        counters['error_norms']+=1;counters['error_maxabs']+=1;counters['error_coordinates']+=257
    row.update(branch_P_S_equal=P['branch_digest']==S['branch_digest'],branch_S_U_equal=S['branch_digest']==U['branch_digest'] if u else None,
               branch_P_U_equal=P['branch_digest']==U['branch_digest'] if u else None)
    return row,raw


def execute_payload(cfg,sp,bp):
    a.validate_call(cfg,sp,bp)
    directory=Path(cfg['artifacts']);primary_path=directory/'primary-result.json';cross_path=directory/'cross-result.json'
    a.require(directory.is_dir() and not cross_path.exists() and not (directory/'cross-failed-partial.json').exists(),'missing P or existing C; no retry')
    primary_pin=d.p.pin(primary_path);primary=json.loads(primary_path.read_text())
    a.require(primary['science_config']==sp and primary['binding']==bp and primary['status']=='complete_primary_prefix_pending_cross_and_audit'
              and len(primary['states'])==36 and len(primary['trials'])==cfg['budget']['historical_P_trial_rows'],'invalid primary lineage')
    for expected,actual in zip(cfg['historical_prefix']['states'],primary['states']):a.anchor(expected,actual)
    runtime=a.runtime(cfg,'cross');cap=cfg['execution']['stages']['cross'];limits=a.Limits(cap['seconds'],cap['RSS_bytes'])
    jobs=a.Jobs(cfg,'cross',limits);paths={k:directory/(k.replace('_','-')+'.csv') for k in ('comparisons','coordinates','torch_trials')}
    event_path=directory/'cross-events.jsonl';guard=a.NumericGuard(cfg,'cross',jobs,limits,[*paths.values(),event_path],primary['trace'])
    sinks={};events=None;latest={};started=a.base.now()
    try:
        sinks={k:a.Sink(path,cfg['output_columns'][k],cfg['budget']['CSV_rows'].get(k,cfg['budget']['CSV_rows']['torch_trials_max'])) for k,path in paths.items()}
        events=a.Events(event_path,'cross',limits,218);events.add(None,None,'start',jobs)
        with guard:
            context,start,rows,head=a.admit_inputs(cfg,guard,jobs,independent=True)
            trace=a.load(guard,primary['trace'],'trace');jobs.P_trace=trace
            for key,value in trace.items():
                dtype=np.int32 if key in ('steps','fit_rows','fit_demo_indices') else np.float64
                a.require(value.dtype==dtype and list(value.shape)==cfg['trace_layout'][key] and np.isfinite(value).all(),'trace dtype/shape changed')
            a.require(trace['steps'].tolist()==list(range(36)) and np.array_equal(trace['fit_rows'],rows)
                    and trace['fit_demo_indices'].tolist()==cfg['fold']['fit_demos'] and a.digest(trace['parameters'][0])==a.digest(start)
                    and jobs.context_digests==primary['context_digests']
                    and all(np.array_equal(trace[f],head[f]) for f in ('scaler_mean','scaler_scale')),'P/C context or start changed')
            for k in range(36):
                a.require(a.digest(trace['parameters'][k])==primary['states'][k]['parameter_digest'],'trace parameter hash differs')
                for f in a.VECTORS:a.require(a.digest(trace[f][k])==primary['vector_digests'][k][f],'trace vector hash differs')
            backend=TorchFit(context)
            U,thetaU,trials,reason=independent_path(cfg,jobs,backend,start,rows,sinks['torch_trials'],events,latest)
            S=[];N=[]
            for k in range(36):
                theta=a.readonly(trace['parameters'][k]);jobs.select('S',k,theta);events.add('S',k,'before',jobs,theta)
                point=numerical.measure(backend,theta,1.,rows,jobs);S.append(point)
                latest.clear();latest.update(role='S',step=k,parameters=theta.tolist(),point=point)
                events.add('S',k,'after',jobs,theta,jobs.last_direction)
            for k in range(36):
                gb,gt=(trace[f][k] for f in ('base_gradient','total_gradient'));jobs.normalize(k,gb,gt)
                events.add('N',k,'before',jobs)
                meta=numerical.construct_direction(d.OPTIMIZATION_ARM,gt.tolist(),gb.tolist(),jobs.direction_work)
                meta.update(input_Gb_digest=a.digest(gb),input_GT_digest=a.digest(gt));N.append(meta);jobs.normalized()
                latest.clear();latest.update(role='N',step=k,meta=meta)
                events.add('N',k,'after',jobs,meta=meta)
            comparison.point_columns=[f for f in cfg['output_columns']['primary_states'] if f not in d.IDS+['optimization_arm','extra_updates']]
            derived=dict(error_norms=0,error_maxabs=0,error_coordinates=0,reference_norms=0)
            comparisons=[]
            for k in range(36):
                P=dict(primary['states'][k],**{f:trace[f][k].tolist() for f in a.VECTORS})
                point,raw=comparison(k,P,S[k],U[k] if k<len(U) else None,N[k],trace['parameters'][k],thetaU[k] if k<len(U) else None,reason,derived)
                sinks['comparisons'].add(point);comparisons.append(point)
                for j in range(257):sinks['coordinates'].add({**d.FOCUS,'step':k,'coordinate':j,'U_available':k<len(U),**{f:values[j] if values is not None else None for f,values in raw.items()}})
            a.require(derived['error_norms']<=504 and derived['error_coordinates']<=129528 and derived['reference_norms']<=360,'derived metric budget exceeded')
            jobs.complete();guard.complete();events.add(None,None,'end',jobs)
            for s in sinks.values():s.close()
            events.close()
        tables={k:s.pin() for k,s in sinks.items()}
        a.require(sum(Path(p['path']).stat().st_size for p in [*primary['tables'].values(),*tables.values()])<=cfg['execution']['new_CSV_bytes_max'],'CSV byte budget exceeded')
        a.require(Path(event_path).stat().st_size+Path(primary['events']['path']).stat().st_size<=cfg['execution']['events_bytes_max'],'event byte budget exceeded')
        summary={f:dict(first_over_tolerance=next((r['step'] for r in comparisons if r[f] is True),None),over_count=sum(r[f] is True for r in comparisons))
                 for f in ('q_total_over_tolerance','q_normalization_over_tolerance','q_same_parameter_gradient_over_tolerance','q_path_over_tolerance')}
        result=dict(status='complete_cross_review_pending_arithmetic_audit',formal_result=False,science_config=sp,binding=bp,primary=primary_pin,
                    started_at_utc=started,completed_at_utc=a.base.now(),runtime=runtime,cost=limits.record(),ledger=jobs.record(),access=guard.record(),
                    context_digests=jobs.context_digests,tables=tables,events=events.pin(),derived_arithmetic=derived,
                    U_stop=reason,U_updates=len(U)-1,U_states=len(U),comparisons=comparisons,mismatch_summary=summary,
                    parent_failed_run_remains_failed=True,old_missing_audit_value_recovered=False)
        a.base.write_new(cross_path,result);return result
    except BaseException as error:
        for s in sinks.values():
            if not s.handle.closed:s.close()
        if events and not events.handle.closed:events.close()
        a.write_failure(directory/'cross-failed-partial.json','cross',error,jobs,guard,limits,latest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);p.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    p.add_argument('--mode',choices=['preflight','evaluate'],default='preflight');args=p.parse_args();cfg,bound=a.preflight(args.config,args.binding)
    result=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding)) if args.mode=='evaluate' else {'status':'bound_no_payload'}
    print(json.dumps({'status':result['status']}))
