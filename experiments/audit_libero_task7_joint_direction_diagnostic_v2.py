"""Independent full18 Torch U paths and same-parameter S audit; no P import."""
from __future__ import annotations
import argparse,json,math,statistics,sys
from pathlib import Path
import numpy as np
from experiments import libero_task7_joint_direction_protocol_v2 as d
from experiments import libero_task7_joint_direction_access_v2 as a
from experiments import audit_libero_task7_joint_direction_diagnostic as numerical
from experiments.audit_libero_task7_fixed_representation_optimization import TorchFit,prepare,Checks,CheckedTable,deep_check,VECTORS
replay,paired_reference,summaries=numerical.replay,numerical.paired_reference,numerical.summaries
TABLES=['endpoints','parameters','states','trials','heads','pairs','scaler-parity']


class SerializedChecks(Checks):
    def equal(self,actual,wanted,label):
        if type(wanted) is float:
            d.p.require(type(actual) in (float,int) and float(actual).hex()==wanted.hex(),'CSV/JSON exact float '+label);self.n+=1
        else:super().equal(actual,wanted,label)


def predicate(current,values,j,m500,slope):
    bound=current+.0001*2.**(-j)*slope
    armijo=bool(values['trainable_objective']<current and values['trainable_objective']<=bound)
    mse=bool(values['component3_mse']<=m500);ok=armijo and mse
    return dict(armijo_pass=armijo,mse_envelope_pass=mse,mse_gate_applied=True,accepted=ok,
        reason='accepted' if ok else 'armijo_and_mse_rejected' if not armijo and not mse else 'armijo_rejected' if not armijo else 'mse_envelope_rejected',armijo_bound=bound)


def own_stop(opt,k,point,trials):
    reason=numerical.stop_reference(opt,k,point)
    if reason:return reason
    return 'line_search_exhausted' if len(trials)==20 and not trials[-1]['accepted'] else None


def check_primary_tables(record,limits):
    checks=SerializedChecks();n=0
    for name,ref in record['tables'].items():
        table=CheckedTable(ref,checks,limits)
        try:
            if name=='parameters':
                for point in record['endpoints']:
                    ident={k:point[k] for k in d.IDS}
                    for j in range(257):table.row({**ident,'condition':point['condition'],'coordinate':j,'parameter':point['parameters'][j],**{f:point[f][j] for f in VECTORS}})
            else:
                key='scaler_parity' if name=='scaler-parity' else name
                for row in record[key]:table.row({f:row[f] for f in ref['columns']})
            table.finish();n+=table.index if hasattr(table,'index') else ref['rows']
        finally:
            if not table.handle.closed:table.handle.close()
    return checks.n


def local_head(cfg,record,h,saved,head,start,context,backend,local,sinks,checks,local_points,shadow_trials):
    ident=a.identity(h);spec=a.key(h)
    for f in ('fit_rows','fit_demo_indices','scaler_mean','scaler_scale'):
        d.p.require(saved[f].dtype==head[f].dtype and saved[f].shape==head[f].shape and saved[f].tobytes()==head[f].tobytes(),'P fit/scaler bits changed')
    coords=np.c_[saved['weights'],saved['bias']]
    d.p.require(saved['weights'].dtype==saved['bias'].dtype==np.float64 and coords.shape==(3,257) and a.digest(coords[0])==a.digest(start),'P saved500 schema')
    states=[r for r in record['states'] if all(r[k]==v for k,v in ident.items())]
    trials=[r for r in record['trials'] if all(r[k]==v for k,v in ident.items())]
    n=len(states);d.p.require(2<=n<=102 and saved['trace_arm_indices'].dtype==saved['trace_extra_updates'].dtype==np.int32
        and saved['trace_arm_indices'].shape==saved['trace_extra_updates'].shape==(n,),'P trace index shape')
    for f in ['trace_parameters',*['trace_'+v for v in VECTORS]]:
        d.p.require(saved[f].dtype==np.float64 and saved[f].shape==(n,257) and np.isfinite(saved[f]).all(),'P trace coordinate schema')
    context_ref=next(r for r in record['context_digests'] if all(r[k]==v for k,v in ident.items()))
    d.p.require(context_ref['digests']==[a.digest(v) for v in context],'P/S fit contexts differ')
    local.register(spec,context,saved['fit_rows']);cursor=0
    for arm_index,opt in enumerate(d.ARMS):
        part=[r for r in states if r['optimization_arm']==opt]
        d.p.require(part and [r['extra_updates'] for r in part]==list(range(len(part))) and len(part)<=51,'P state prefix')
        own_head=next(r for r in record['heads'] if all(r[k]==v for k,v in ident.items()) and r['optimization_arm']==opt)
        previous=None;previous_trials=None;m500=None
        for k,Ppoint in enumerate(part):
            index=cursor;cursor+=1;theta=saved['trace_parameters'][index];Pdirection=saved['trace_direction'][index]
            d.p.require(int(saved['trace_arm_indices'][index])==arm_index and int(saved['trace_extra_updates'][index])==k
                and a.digest(theta)==Ppoint['parameter_digest'],'P parameter identity/hash')
            if k==0:d.p.require(a.digest(theta)==a.digest(start),'P path did not start at original500')
            else:
                accepted=previous_trials[-1]
                d.p.require(accepted['accepted'] and a.digest(theta)==a.digest(previous['theta']+accepted['alpha']*previous['direction']),'P accepted update bits')
                local.counts['parameter_coordinate_checks']+=257
                d.p.require(Ppoint['trainable_objective']<previous['point']['trainable_objective'],'P accepted T not strict')
            d.p.require(Ppoint['component3_mse']<=part[0]['component3_mse'],'P M500 envelope')
            local.begin(opt,k,theta,Pdirection)
            local.latest.update(parameters=theta.tolist(),P_vectors={f:saved['trace_'+f][index].tolist() for f in VECTORS})
            S=numerical.measure(backend,theta,1.,saved['fit_rows'],local)
            fixed=a.fixed_P_slope(backend,local)
            max_scalar=a.local_numeric_gate(Ppoint,S,local,checks);errors={}
            for f in VECTORS:
                Pvector=saved['trace_'+f][index];Svector=np.asarray(S[f],np.float64)
                error=float(np.max(np.abs(Svector-Pvector)));errors[f]=error
                local.counts['vector_maxabs_checks']+=1;local.counts['vector_coordinates']+=257
                local.latest.update(field=f,expected=Pvector.tolist(),actual=Svector.tolist(),max_abs=error)
                d.p.require(error<=1e-10,'same-parameter vector mismatch '+f);checks.n+=257
            for f,v in fixed.items():
                local.latest.update(field='fixed_P_'+f,expected=Ppoint[f],actual=v)
                checks.equal(v,Ppoint[f],'fixed P direction '+f)
            own_trials=[r for r in trials if r['optimization_arm']==opt and r['from_extra_updates']==k]
            reasonP=own_stop(opt,k,Ppoint,own_trials)
            reasonS=numerical.stop_reference(opt,k,S)
            if k==len(part)-1:checks.equal(reasonP,own_head['terminal_reason'],'P exact stop')
            else:d.p.require(reasonP is None,'P continued after stop')
            if m500 is None:m500=S['component3_mse']
            row={**ident,'optimization_arm':opt,'extra_updates':k,'parameter_digest':a.digest(theta),
                'local_max_abs':max_scalar,'max_gradient_abs':max(errors[f] for f in VECTORS[:3]),'direction_max_abs':errors['direction'],
                'q_P':Ppoint['direction_unit_sum_norm'],'q_S':S['direction_unit_sum_norm'],
                'q_delta':S['direction_unit_sum_norm']-Ppoint['direction_unit_sum_norm'] if Ppoint['direction_unit_sum_norm'] is not None and S['direction_unit_sum_norm'] is not None else None,
                'same_branch':Ppoint['branch_digest']==S['branch_digest'],'direction_status_equal':Ppoint['direction_status']==S['direction_status'],
                'selected_joint_equal':Ppoint['selected_joint']==S['selected_joint'],'stop_P':reasonP,'stop_S':reasonS,'stop_equal':reasonP==reasonS,
                'fixed_P_MSE_slope':fixed['exact_MSE_slope'],'fixed_P_T_slope':fixed['directional_derivative'],'numeric_gate_passed':True}
            shadow_round=[]
            d.p.require(len(own_trials)<=20 and [r['trial_index'] for r in own_trials]==list(range(len(own_trials))),'P trial grid order')
            if numerical.stop_reference(opt,k,Ppoint) is not None:d.p.require(not own_trials,'P trials after stop')
            else:d.p.require(bool(own_trials),'P missing trial round')
            for j,trial in enumerate(own_trials):
                d.p.require(k<50 and trial['alpha']==2.**(-j) and not any(r['accepted'] for r in own_trials[:j]),'P extra trial or first acceptance')
                wanted=predicate(Ppoint['trainable_objective'],trial,j,part[0]['component3_mse'],Ppoint['directional_derivative'])
                for field,value in wanted.items():checks.equal(trial[field],value,'P own predicate '+field)
                checks.equal(trial['start500_component3_mse'],part[0]['component3_mse'],'P M500 reference')
                checks.equal(trial['trainable_objective'],trial['component3_mse']/7+trial['penalty'],'P loss terms')
                checks.equal(trial['total_objective'],trial['trainable_objective']+Ppoint['fixed_other6_floor'],'P loss floor')
                local.trial_begin(j,a.readonly(theta+trial['alpha']*Pdirection))
                values=numerical.trial_objective(backend,local.trial_theta,1.,S['fixed_other6_floor'],local);local.trial_end()
                error=max(abs(values[f]-trial[f]) for f in values)
                local.latest.update(field='trial_target',P=trial,S=values,max_abs=error)
                d.p.require(error<=1e-10,'same-parameter P trial target mismatch');checks.n+=len(values)
                decision=predicate(S['trainable_objective'],values,j,m500,fixed['directional_derivative'])
                out={**ident,'optimization_arm':opt,'from_extra_updates':k,'trial_index':j,'alpha':trial['alpha'],'target_max_abs':error,
                    'P_accepted':trial['accepted'],'shadow_accepted':decision['accepted'],'accept_equal':trial['accepted']==decision['accepted'],
                    'shadow_armijo_pass':decision['armijo_pass'],'shadow_mse_pass':decision['mse_envelope_pass'],
                    'shadow_T':values['trainable_objective'],'shadow_MSE':values['component3_mse'],'shadow_slope':fixed['directional_derivative'],
                    'shadow_M500':m500,'numeric_gate_passed':True}
                sinks['local-trials'].add(out);shadow_trials.append(out)
                shadow_round.append(out)
            if own_trials and not own_trials[-1]['accepted']:d.p.require(len(own_trials)==20 and k==len(part)-1,'P partial exhaustion')
            if reasonS is None and len(shadow_round)==20 and not any(r['shadow_accepted'] for r in shadow_round):reasonS='line_search_exhausted'
            row.update(stop_S=reasonS,stop_equal=reasonP==reasonS)
            sinks['local-states'].add(row);local_points.append(row)
            previous={'theta':theta,'direction':Pdirection,'point':Ppoint};previous_trials=own_trials
            local.check()
        d.p.require(a.digest(coords[arm_index+1])==a.digest(previous['theta'])
            and int(saved['additional_updates'][arm_index+1])==len(part)-1,'P last accepted endpoint selection')
        for prefix_name,point in [('initial',part[0]),('final',part[-1])]:
            for suffix,field in [('objective','total_objective'),('trainable_objective','trainable_objective'),
                                 ('gradient_l2','total_l2'),('component3_mse','component3_mse'),('penalty','penalty')]:
                checks.equal(own_head[prefix_name+'_'+suffix],point[field],'P head actual endpoint '+suffix)
        checks.equal(own_head['extra_updates'],len(part)-1,'P head update count')
        checks.equal(own_head['trials'],sum(r['optimization_arm']==opt for r in trials),'P head trial count')
        for field,value in numerical.evidence(part[0],part[-1],len(part)-1,opt).items():
            checks.equal(own_head[field],value,'P head scientific evidence '+field)
    d.p.require(cursor==n,'P trace states missing')


def audit_payload(cfg,science_pin,binding_pin):
    a.validate_call(cfg,science_pin,binding_pin)
    d.p.require('experiments.run_libero_task7_joint_direction_diagnostic' not in sys.modules
        and 'experiments.run_libero_task7_joint_direction_diagnostic_v2' not in sys.modules,'primary implementation imported by audit')
    directory=Path(cfg['artifacts']);output=Path(cfg['output']);primary=directory/'primary-result.json'
    d.p.require(directory.is_dir() and not output.exists() and not any((directory/f).exists() for f in ('independent-audit.json','completion.json','audit-failed-partial.json')),'existing audit; no retry')
    primary_pin=d.p.pin(primary);record=json.loads(primary.read_text())
    d.p.require(record['science_config']==science_pin and record['binding']==binding_pin and record['inputs']==cfg['inputs']
        and record['status']=='complete_joint_direction_v2_pending_same_parameter_audit' and len(record['new_heads'])==18 and len(record['heads'])==36,'P provenance or coverage invalid')
    runtime=a.runtime(cfg,audit=True);limits=a.Limits(cfg['execution']['audit_seconds'],cfg['execution']['audit_RSS_bytes']);started=a.base.now()
    jobs=a.TraceJobs(cfg,limits,audit=True);local=a.ShadowJobs(cfg,limits);dispatch=a.Dispatch(jobs,local);work=a.RepresentationWork(cfg);checks=Checks()
    paths={name:directory/(name+'.csv') for name in ('local-states','local-trials')}
    for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'P CSV changed')
    Ppaths=[directory/f'head-{i:03d}.npz' for i in range(18)];newpaths=[directory/f'audit-head-{i:03d}.npz' for i in range(18)]
    d.p.require(set(record['tables'])==set(TABLES) and all(Path(ref['path'])==directory/(name+'.csv') for name,ref in record['tables'].items()),'P table paths')
    for i,ref in enumerate(record['new_heads']):d.p.require(Path(ref['path'])==Ppaths[i] and a.identity(ref)==a.identity(cfg['inputs']['heads'][i]),'P archive identity')
    guard=a.Guard(cfg,[r['path'] for r in record['tables'].values()],Ppaths,limits,dispatch,record['new_heads'])
    guard.write_outputs=set(map(lambda p:str(p.resolve()),paths.values()))
    local_sinks={};local_points=[];shadow_trials=[];archives=[]
    states=[];trials=[];endpoints=[];heads=[];pairs=[];parities=[]
    try:
        local_sinks={'local-states':a.Sink(paths['local-states'],cfg['local_state_columns'],1836),
                     'local-trials':a.Sink(paths['local-trials'],cfg['local_trial_columns'],36000)}
        with guard:
            shared=a.load(cfg['inputs']['shared'],'shared');y=shared['targets'];lengths=cfg['inputs']['action_lengths']
            expected=[(i,t) for i,n in enumerate(lengths) for t in range(1,n)]
            d.p.require(y.dtype==np.float32 and y.shape==(4188,7) and np.isfinite(y).all() and (abs(y)<=1).all()
                and shared['demo_indices'].dtype==shared['action_steps'].dtype==shared['demo_lengths'].dtype==np.int32
                and list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==expected
                and shared['demo_lengths'].tolist()==lengths and shared['tail20'].dtype==np.bool_
                and shared['tail20'].tolist()==[t>=lengths[i]-20 for i,t in expected],'source causal schema differs')
            csv_checks=check_primary_tables(record,limits)
            sinks={name:a.NullSink() for name in TABLES}
            for model in cfg['inputs']['models']:
                cache=a.load(model['current_archive'],'current');f,p0=cache['features'],cache['predictions']
                d.p.require(f.dtype==p0.dtype==np.float32 and f.shape==(4188,256) and p0.shape==(4188,7)
                    and np.isfinite(f).all() and np.isfinite(p0).all() and (abs(p0)<=1).all(),'source cache schema differs')
                for plan in cfg['inputs']['folds']:
                    rkey=(model['seed'],model['arm'],plan['fold']);h=next(h for h in cfg['inputs']['heads'] if a.key(h)==rkey+(d.finer.source.MODE,))
                    head=a.load(h['archive'],'head');rows=np.asarray(plan['fit_rows'],np.int32)
                    work.begin(rkey,len(rows));rows,z=prepare(head,f,plan,lengths);work.built(rkey,z)
                    context=jobs.register_fit(model,plan,z,y,p0,rows,work);backend=TorchFit(context)
                    start=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];jobs.source(h,head,start)
                    ident=a.identity(h);results={};points=[start];update_values=[];initials=[]
                    for opt in d.ARMS:
                        if opt==d.ARMS[1]:work.reuse(rkey,z)
                        pathkey=a.key(h)+(opt,);jobs.head(pathkey,*context,rows,start);pid={**ident,'optimization_arm':opt}
                        def state_sink(k,theta,row):
                            entry={**pid,'extra_updates':k,'parameter_digest':a.digest(theta),**{f:v for f,v in row.items() if f not in VECTORS}}
                            states.append(entry);sinks['states'].add(entry)
                        def trial_sink(row):
                            entry={**pid,**row};trials.append(entry);sinks['trials'].add(entry)
                        first,final,theta,reason=replay(backend,rows,jobs,state_sink,trial_sink)
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
                        evidence=numerical.evidence(first,final,updates,opt)
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
                    d.p.require(all(initials[0][f]==initials[1][f] for f in [*VECTORS[:3],'component3_mse','penalty','trainable_objective','total_objective','total_l2','branch_digest']),'paired500 fit/gradient readout differs')
                    baseline,candidate=(results[opt] for opt in d.ARMS)
                    paired={**ident,**{'candidate_minus_baseline_'+field:candidate['final_'+field]-baseline['final_'+field]
                        for field in ['component3_mse','penalty','trainable_objective','gradient_l2']},
                        'baseline_updates':baseline['extra_updates'],'candidate_updates':candidate['extra_updates'],
                        'baseline_stop':baseline['terminal_reason'],'candidate_stop':candidate['terminal_reason'],
                        'candidate_finite_fit_progress':candidate['finite_fit_progress'],'candidate_protected_joint_fit_cue':candidate['protected_joint_fit_cue']}
                    cohort=next(r for r in cfg['cohorts'] if all(r[k]==v for k,v in ident.items()))
                    paired.update({k:cohort[k] for k in ('cohort','prior_control_updates','prior_control_stop')})
                    paired.update(paired_reference(initials[0],baseline,candidate,jobs.done[a.key(h)+(d.ARMS[1],)]['mse_deltas']))
                    paired.update(candidate_minus_control_trials=candidate['trials']-baseline['trials'],source500_parameters_same=True)
                    pairs.append(paired);sinks['pairs'].add(paired)
                    index=next(i for i,x in enumerate(cfg['inputs']['heads']) if a.key(x)==a.key(h))
                    saved=a.load(record['new_heads'][index],'newhead')
                    dispatch.use_local(True)
                    local_head(cfg,record,h,saved,head,start,context,backend,local,local_sinks,checks,local_points,shadow_trials)
                    dispatch.use_local(False)
                    pin=np.savez(newpaths[index],fit_rows=rows,fit_demo_indices=head['fit_demo_indices'],scaler_mean=head['scaler_mean'],scaler_scale=head['scaler_scale'],
                        additional_updates=np.asarray([0,*update_values],np.int32),weights=np.stack([v[:-1] for v in points]),bias=np.asarray([v[-1] for v in points],np.float64),**a.trace_arrays(jobs.traces[a.key(h)]),**a.trace_arrays(local.traces[a.key(h)],'local'))
                    archives.append({**ident,**pin})
                    parity=dict(seed=model['seed'],arm=model['arm'],fold=plan['fold'],fit_rows=len(rows),paired_representation_same_object=True,source500_start_same_parameters=True)
                    parities.append(parity);sinks['scaler-parity'].add(parity);del z,context,head,saved,backend
                del cache,f,p0;print(json.dumps({'stage':'audit','seed':model['seed'],'arm':model['arm'],'updates':jobs.counts['updates']}),flush=True)

            jobs.complete();work.complete();local.complete(record['states'],record['trials']);dispatch.check();guard.complete()
            for s in local_sinks.values():s.close()
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'P CSV changed during audit')
        d.p.require(d.p.pin(primary)==primary_pin,'P result changed during audit')
        # Independently reconstruct P's scientific cues from its actual accepted states.
        for pair in record['pairs']:
            spec={k:pair[k] for k in d.IDS}
            H={opt:next(h for h in record['heads'] if all(h[k]==v for k,v in spec.items()) and h['optimization_arm']==opt) for opt in d.ARMS}
            first=next(r for r in record['states'] if all(r[k]==v for k,v in spec.items()) and r['optimization_arm']==d.ARMS[0] and r['extra_updates']==0)
            part=[r for r in record['states'] if all(r[k]==v for k,v in spec.items()) and r['optimization_arm']==d.ARMS[1]]
            deltas=[b['component3_mse']-a0['component3_mse'] for a0,b in zip(part,part[1:])]
            for f,v in paired_reference(first,H[d.ARMS[0]],H[d.ARMS[1]],deltas).items():checks.equal(pair[f],v,'P scientific pair '+f)
        deep_check(checks,record['direction_support_summary'],numerical.source_reference(record['pairs']),'P source cues')
        robust=[]
        for Ppair,Upair in zip(record['pairs'],pairs):
            ident=a.identity(Ppair);d.p.require(a.identity(Upair)==ident,'P/U head coverage order')
            localrows=[r for r in local_points if all(r[k]==v for k,v in ident.items())]
            localtrials=[r for r in shadow_trials if all(r[k]==v for k,v in ident.items())]
            stopdiff=sum(not r['stop_equal'] for r in localrows);acceptdiff=sum(not r['accept_equal'] for r in localtrials)
            robust.append({**ident,'P_paired_cue':Ppair['paired_joint_cue'],'U_paired_cue':Upair['paired_joint_cue'],
                'local_stop_disagreements':stopdiff,'local_accept_disagreements':acceptdiff,
                'robust_paired_cue':Ppair['paired_joint_cue'] and Upair['paired_joint_cue'] and stopdiff==acceptdiff==0})
        robust_sources={arm:all(r['robust_paired_cue'] for r in robust if r['arm']==arm) for arm in d.p.ARMS}
        tables={k:s.pin() for k,s in local_sinks.items()}
        d.p.require(sum(Path(r['path']).stat().st_size for r in tables.values())<=cfg['execution']['audit_CSV_bytes_max']
            and sum(p.stat().st_size for p in newpaths)<=cfg['execution']['trace_NPZ_bytes_per_stage_max'],'audit output byte budget')
        audit=dict(status='passed_independent_same_parameter_joint_direction_v2_audit',formal_result=False,science_config=science_pin,binding=binding_pin,primary=primary_pin,
            runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),computation=jobs.counts,
            direction_work=jobs.direction_work.counts,local_computation=local.counts,local_direction_work=local.direction_work.counts,
            representation_work=work.record(),trace_coordinate_copies=jobs.trace_coordinate_copies+5*257*local.counts['gradient_states'],
            scalar_checks=checks.n,primary_CSV_exact_checks=csv_checks,new_heads=archives,tables=tables,
            states=states,trials=trials,endpoints=endpoints,heads=heads,pairs=pairs,local_states=local_points,local_trials=shadow_trials,
            robust_pairs=robust,robust_source_cues=robust_sources,robust_population_cue=all(robust_sources.values()),**summaries(heads,pairs),
            old_failed_v1_certified=False,primary_numerical_implementation_imported=False,original_Policy_CNN_environment_or_sealed_calls=0)
        path=directory/'independent-audit.json';a.base.write_new(path,audit)
        d.p.require(path.stat().st_size<=cfg['execution']['audit_JSON_bytes_max'],'audit JSON size budget')
        result=dict(status='complete_independently_same_parameter_audited_direction_v2',formal_result=False,science_config=science_pin,binding=binding_pin,
            primary=primary_pin,audit=d.p.pin(path),P_direction_support_summary=record['direction_support_summary'],U_direction_support_summary=audit['direction_support_summary'],
            robust_pairs=robust,robust_source_cues=robust_sources,robust_population_cue=audit['robust_population_cue'],
            local_numeric_states=local.counts['gradient_states'],local_trial_targets=local.counts['trial_forwards'],
            primary_updates=record['computation']['updates'],independent_U_updates=jobs.counts['updates'],local_S_updates=0,
            local_stop_disagreements=sum(r['local_stop_disagreements'] for r in robust),local_accept_disagreements=sum(r['local_accept_disagreements'] for r in robust),
            old_failed_v1_certified=False,original_Policy_CNN_environment_or_sealed_calls=0)
        a.base.write_new(output,result)
        a.base.write_new(directory/'completion.json',dict(status=result['status'],result=d.p.pin(output),audit=result['audit'],science_config=science_pin,binding=binding_pin,
            total_NPZ_loads=68,total_member_decodes=574,total_archive_SHA_checks=172,old_failed_v1_certified=False))
        return audit
    except BaseException as error:
        for s in local_sinks.values():
            if not s.handle.closed:s.close()
        a.fail_record(directory/'audit-failed-partial.json',error,dispatch,local,guard,limits,checks);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);p.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    args=p.parse_args();cfg=json.loads(args.config.read_text());out=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':out['status']}))
