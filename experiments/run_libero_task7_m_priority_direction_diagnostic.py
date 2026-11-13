"""Bound negative-M direction versus guarded T priority at original auxiliary500."""
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
from experiments import libero_task7_m_priority_direction_protocol as d
from experiments import libero_task7_m_priority_direction_access as a
from experiments.run_libero_task7_fixed_representation_optimization import admit,VECTORS
from experiments.analyze_libero_task7_component3_history_transition import stop_child

TABLES=['endpoints','parameters','states','trials','heads','pairs','scaler-parity']
METRICS=['component3_mse','penalty','trainable_objective','total_objective','total_l2']


def construct_direction(arm,total,base,work=None):
    work=work or a.DirectionWork();work.begin(arm,total,base)
    g=work.norm(total);b=work.norm(base);c=q=None
    rho=.01;kappa=2*rho if arm==d.ARMS[0] else None
    bound=0. if rho==0 else None if g==0 or b==0 else -7.*rho*g*b
    if bound is not None:a.finite(bound)
    if rho and bound==0:bound=None
    if g==0:chosen=work.vector([0.]*257);status='zero_total_gradient'
    elif b==0:
        chosen=work.vector([-x for x in total] if rho==0 else [0.]*257)
        status='base_zero_use_total' if rho==0 else 'zero_base_margin_unavailable'
    elif rho and bound is None:chosen=work.vector([0.]*257);status='margin_unrepresentable'
    elif arm==d.ARMS[1]:chosen=work.vector([-g*(x/b) for x in base]);status='negative_base_norm_matched'
    else:
        u=work.vector([x/g for x in total]);v=work.vector([x/b for x in base]);c=work.dot(u,v)
        if c>=kappa:
            chosen=work.vector([-x for x in total]);status='negative_total_feasible' if rho==0 else 'negative_total_margin_feasible'
        else:
            w=work.vector([-x+c*y for x,y in zip(u,v)]);q=work.norm(w)
            if q<=1e-12:
                chosen=work.vector([0.]*257);status='unresolved_joint_direction' if rho==0 else 'unresolved_margin_direction'
            elif rho==0:chosen=work.vector([g*(x/q) for x in w]);status='projected_T_priority'
            else:
                tangent=math.sqrt(1-kappa*kappa)
                chosen=work.vector([g*(-kappa*y+tangent*(x/q)) for x,y in zip(w,v)]);status='margin_projected_T_priority'
    norm=work.norm(chosen);st=work.dot(total,chosen);sm=7*work.dot(base,chosen);a.finite(sm)
    met=bool(bound is not None and sm<=bound);joint=met and st<0;construction=status
    if rho==0:
        if status not in ('zero_total_gradient','unresolved_joint_direction') and not joint:status='unresolved_joint_direction'
    elif status in ('negative_total_margin_feasible','margin_projected_T_priority','negative_base_norm_matched'):
        if not met:status='selected_M_margin_violation'
        elif st>=0:status='selected_T_non_descent'
    meta=dict(direction=chosen,status=status,construction_status=construction,total_norm=g,base_norm=b,unit_sum_norm=None,unit_cosine=c,projection_norm=q,
      direction_norm=norm,selected_T_slope=st,selected_MSE_slope=sm,selected_joint=joint,required_margin_fraction=rho,
      construction_margin_fraction=kappa,required_M_slope_bound=bound,selected_margin_met=met)
    work.end(meta);return meta


def paired_reference(first,control,candidate,deltas,anchors):
    t0,m0=first['trainable_objective'],first['component3_mse']
    c={'T':control['final_trainable_objective'],'M':control['final_component3_mse']}
    v={'T':candidate['final_trainable_objective'],'M':candidate['final_component3_mse']}
    refs={kind:{'T':point['trainable_objective'],'M':point['component3_mse']} for kind,point in anchors.items()}
    cue=d.paired_evidence(t0,m0,c,v,refs,candidate['extra_updates'],deltas)
    return dict(cue,candidate_stepwise_MSE_increases=sum(dm>0 for dm in deltas),
        candidate_path_joint_cue=cue['candidate_progress_from500'],candidate_minus_start_MSE=v['M']-m0,
        candidate_minus_control_MSE=v['M']-c['M'],candidate_minus_control_T=v['T']-c['T'],
        **{'candidate_minus_'+kind+'_'+metric:v[key]-refs[kind][key] for kind in refs for metric,key in [('MSE','M'),('T','T')]})


def source_reference(pairs):return d.source_evidence(pairs)


def measure(context,theta,coefficient,rows,jobs=None,*,arm=None,direction_work=None):
    d.p.require(coefficient==1.,'only lambda1 allowed')
    arm=arm or d.ARMS[0]
    z,y,p0=context;n=len(z)
    if jobs:jobs.check_context(z,y,p0);jobs.state_begin(theta);jobs.linear(n)
    raw=p0[:,2]+(np.einsum('ij,j->i',z,theta[:-1],optimize=False)+theta[-1]);pred=np.clip(raw,-1,1)
    error=(pred-y[:,2])**2;old_error=(p0[:,2]-y[:,2])**2;diff=error-old_error
    factor=2*(pred-y[:,2])/7*((raw>=-1)&(raw<=1));penfactor=factor*(diff>0)
    gradients=[]
    for f in (factor,penfactor,factor if coefficient==0 else factor+penfactor):
        if jobs:jobs.gradient()
        gradients.append(np.r_[np.mean(z*f[:,None],axis=0),f.mean()])
    gb,gp,g=gradients;norms=[float(np.linalg.norm(v)) for v in gradients];meta=construct_direction(jobs.active[-1] if jobs else arm,g.tolist(),gb.tolist(),jobs.direction_work if jobs else direction_work)
    direction=np.asarray(meta['direction'],np.float64)
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
    row.update(a.direction_fields(meta),exact_MSE_slope=float(de.mean()),exact_penalty_slope=float(dh.mean()/7))
    row['exact_margin_met']=bool(row['required_M_slope_bound'] is not None and row['exact_MSE_slope']<=row['required_M_slope_bound'])
    if jobs:jobs.state_end(row)
    return row


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
        reason=a.stop(jobs.active[-1],updates,row)
        if reason:break
        for index in range(20):
            alpha=d.trial_alpha(jobs.active[-1],index);trial=a.readonly(theta+alpha*np.asarray(row['direction']))
            jobs.trial_begin(index,trial);values=trial_objective(context,trial,1.,row['fixed_other6_floor'],jobs)
            decision=jobs.trial_end(values)
            emit_trial(dict(from_extra_updates=updates,trial_index=index,alpha=alpha,**values,
                start500_component3_mse=jobs.fixed_mse,slope=row['directional_derivative'],current_component3_mse=row['component3_mse'],delta_component3_mse=values['component3_mse']-row['component3_mse'],
                armijo_bound=row['trainable_objective']+.0001*alpha*row['directional_derivative'],**decision))
            if decision['accepted']:jobs.commit(trial);break
        else:reason='line_search_exhausted';break
    final=jobs.current;jobs.finish(reason);return first,row,final,reason


def summaries(heads,pairs):
    seeds=[];sources=[];seed_pairs=[];source_pairs=[]
    fields=['final_component3_mse','final_penalty','final_trainable_objective','final_gradient_l2','trainable_ratio','gradient_ratio']
    for source in a.SOURCE_ARMS:
        for opt in d.ARMS:
            chosen=[h for h in heads if h['arm']==source and h['optimization_arm']==opt]
            d.p.require(len(chosen)==9,'incomplete source optimization group')
            for seed in a.SEEDS:
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
        pf=[k for k in chosen[0] if k.startswith(('candidate_minus_baseline_','candidate_minus_bisector_','candidate_minus_zero_','candidate_minus_margin_'))]
        for seed in a.SEEDS:
            part=sorted([r for r in chosen if r['seed']==seed],key=lambda r:r['fold'])
            seed_pairs.append(dict(arm=source,seed=seed,metrics={f:dict(fold_values=[r[f] for r in part],equal_fold_mean=math.fsum(r[f] for r in part)/3) for f in pf}))
        source_pairs.append(dict(arm=source,metrics={f:math.fsum(r[f] for r in chosen)/9 for f in pf}))
    seed_cohorts=[];source_cohorts=[]
    for source in a.SOURCE_ARMS:
        for cohort in ('zero','short','full50'):
            chosen=[r for r in pairs if r['arm']==source and r['cohort']==cohort]
            def group(rows):
                return dict(n=len(rows),paired_joint_count=sum(r['paired_joint_cue'] for r in rows),
                    fold_values=[dict(fold=r['fold'],seed=r['seed'],control_updates=r['baseline_updates'],candidate_updates=r['candidate_updates'],
                        mse_delta=r['candidate_minus_baseline_component3_mse'],T_delta=r['candidate_minus_baseline_trainable_objective']) for r in rows],
                    mean_MSE_delta=math.fsum(r['candidate_minus_baseline_component3_mse'] for r in rows)/len(rows) if rows else None,
                    mean_T_delta=math.fsum(r['candidate_minus_baseline_trainable_objective'] for r in rows)/len(rows) if rows else None)
            source_cohorts.append(dict(arm=source,cohort=cohort,**group(chosen)))
            for seed in a.SEEDS:
                seed_cohorts.append(dict(arm=source,seed=seed,cohort=cohort,**group([r for r in chosen if r['seed']==seed])))
    return dict(seed_arm_summaries=seeds,source_arm_summaries=sources,seed_paired_summaries=seed_pairs,
        source_paired_summaries=source_pairs,seed_cohort_summaries=seed_cohorts,source_cohort_summaries=source_cohorts,
        direction_support_summary=source_reference(pairs))


def execute_payload(cfg,science_pin,binding_pin):
    a.validate_call(cfg,science_pin,binding_pin)
    directory=Path(cfg['artifacts']);output=directory/'primary-result.json'
    d.p.require(not directory.exists() and not output.exists(),'existing run; no retry')
    runtime=a.runtime(cfg);history=a.History(cfg)
    directory.mkdir(parents=True);limits=a.Limits(cfg['execution']['primary_seconds'],cfg['execution']['primary_RSS_bytes']);started=a.base.now()
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
                    rkey=(model['seed'],model['arm'],plan['fold']);h=next(h for h in cfg['inputs']['heads'] if a.key(h)==rkey+(a.finer.source.MODE,))
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
                        evidence=a.finer.source.endpoint_evidence(first['trainable_objective'],final['trainable_objective'],first['total_l2'],final['total_l2'],
                            first['component3_mse'],final['component3_mse'],updates,a.finer.source.ARMS[1])
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
                    # History validates each actual control head and the frozen total coverage.
                    paired.update({k:cohort[k] for k in ('cohort','prior_control_updates','prior_control_stop')})
                    anchor=history.anchor(h)
                    paired.update({kind+'_anchor_'+f:point[f] for kind,point in anchor.items() for f in ['component3_mse','penalty','trainable_objective','total_l2','extra_updates']})
                    paired.update(paired_reference(initials[0],baseline,candidate,jobs.done[a.key(h)+(d.ARMS[1],)]['mse_deltas'],anchor))
                    paired.update(candidate_minus_control_trials=candidate['trials']-baseline['trials'],source500_parameters_same=True)
                    pairs.append(paired);sinks['pairs'].add(paired)
                    index=next(i for i,x in enumerate(cfg['inputs']['heads']) if a.key(x)==a.key(h))
                    pin=np.savez(newpaths[index],fit_rows=rows,fit_demo_indices=head['fit_demo_indices'],scaler_mean=head['scaler_mean'],scaler_scale=head['scaler_scale'],
                        additional_updates=np.asarray([0,*update_values],np.int32),weights=np.stack([v[:-1] for v in points]),bias=np.asarray([v[-1] for v in points],np.float64),**a.trace_arrays(jobs.traces[a.key(h)]))
                    archives.append({**ident,**pin})
                    parity=dict(seed=model['seed'],arm=model['arm'],fold=plan['fold'],fit_rows=len(rows),paired_representation_same_object=True,source500_start_same_parameters=True)
                    parities.append(parity);sinks['scaler-parity'].add(parity);del z,context,head
                del cache,f,p0;print(json.dumps({'stage':'primary','seed':model['seed'],'arm':model['arm'],'updates':jobs.counts['updates']}),flush=True)
            jobs.complete();work.complete();guard.complete()
            for sink in sinks.values():sink.close()
        historical=history.complete()
        d.p.require(sum(p.stat().st_size for p in paths.values())<=67108864 and sum(p.stat().st_size for p in newpaths)<=cfg['execution']['trace_NPZ_bytes_per_stage_max'],'output size exceeded')
        expected_rows={'endpoints':54,'parameters':13878,'states':jobs.counts['gradient_states'],'trials':jobs.counts['trial_forwards'],'heads':36,'pairs':18,'scaler-parity':18}
        d.p.require({k:s.rows for k,s in sinks.items()}==expected_rows and sum(expected_rows.values())<=cfg['budget']['output_CSV_rows_max'],'table budget differs')
        result=dict(status='complete_M_priority_pending_same_parameter_audit',formal_result=False,science_config=science_pin,binding=binding_pin,parents=cfg['parents'],inputs=cfg['inputs'],budget=cfg['budget'],
            runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),computation=jobs.counts,representation_work=work.record(),direction_work=jobs.direction_work.counts,historical_control_parity=historical,shared_start=jobs.starts.record(),trial_counts_by_arm=jobs.trials_by_arm,
            tables={k:s.pin() for k,s in sinks.items()},new_heads=archives,context_digests=contexts,trace_coordinate_copies=jobs.trace_coordinate_copies,states=states,trials=trials,endpoints=endpoints,heads=heads,pairs=pairs,scaler_parity=parities,**summaries(heads,pairs),
            original_Policy_CNN_environment_or_sealed_calls=0,prior_joint_cues_G_failed_H_unrun_unchanged=True)
        a.base.write_new(output,result);return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        a.fail_record(directory/'primary-failed-partial.json',error,jobs,None,guard,limits);raise


def run(config=d.CONFIG,binding=d.BINDING):
    started=time.monotonic();cfg,bound=a.preflight(config,binding);output=Path(cfg['output'])
    launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(p.exists() for p in (output,launch,final,Path(cfg['artifacts']))),'existing T-priority run; no retry')
    record=dict(started_at_utc=a.base.now(),science_config=d.p.pin(config),binding=d.p.pin(binding),processes=[]);a.base.write_new(launch,record);child=None
    try:
        for stage,module,cap in [('primary','experiments.run_libero_task7_m_priority_direction_diagnostic',cfg['execution']['primary_seconds']),('audit','experiments.audit_libero_task7_m_priority_direction_diagnostic',cfg['execution']['audit_seconds'])]:
            env=os.environ.copy();env.update(bound['CPU_environment']);cmd=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            if stage=='primary':cmd+=['--mode','evaluate']
            row=dict(stage=stage,command=cmd,started_at_utc=a.base.now());record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as log:
                child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(cap+60,max(1,cfg['execution']['overall_seconds']-5-(time.monotonic()-started))))
                except BaseException:stop_child(child);raise
            row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-started<=cfg['execution']['overall_seconds'],'T-priority overall watchdog exceeded');record['status']='complete_independently_audited_M_priority_direction'
    except BaseException as error:
        if child is not None:stop_child(child)
        record.update(status='failed_no_retry',error=repr(error));raise
    finally:
        record.update(wall_seconds=time.monotonic()-started,completed_at_utc=a.base.now());a.base.write_new(final,record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=d.CONFIG);p.add_argument('--binding',type=Path,default=d.BINDING)
    p.add_argument('--mode',choices=['preflight','evaluate','run'],default='preflight');args=p.parse_args()
    if args.mode=='run':result=run(args.config,args.binding)
    elif args.mode=='evaluate':
        cfg=json.loads(args.config.read_text());result=execute_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding))
    else:a.preflight(args.config,args.binding);result={'status':'bound_no_payload'}
    print(json.dumps({'status':result['status']}))
