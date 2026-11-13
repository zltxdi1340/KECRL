"""Independent Float64 autograd replay of both trial grids; no main import."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import numpy as np
from experiments import libero_task7_mse_guard_finer_grid_protocol as d
from experiments import libero_task7_mse_guard_finer_grid_access as a
from experiments.audit_libero_task7_fixed_representation_optimization import TorchFit,measure,prepare,Checks,CheckedTable,deep_check,VECTORS


def trial_objective(backend,theta,coefficient,floor,jobs=None):
    import torch
    d.p.require(coefficient==1.,'independent lambda1 only')
    if jobs:jobs.check_context(*backend.context);jobs.tensor_input_id=id(backend.x)
    with torch.no_grad():
        raw=backend.p0[:,2]+torch.nn.functional.linear(backend.x,torch.tensor(theta[:-1][None]),torch.tensor(theta[-1:])).squeeze(1)
        error=(raw.clamp(-1,1)-backend.y[:,2]).square();prior=(backend.p0[:,2]-backend.y[:,2]).square()
        mse=error.mean().item();penalty=(torch.relu(error-prior).mean()/7).item()
    t=mse/7+penalty
    d.p.require(torch.isfinite(raw).all().item() and np.isfinite(theta).all() and math.isfinite(t),'independent invalid trial')
    return dict(component3_mse=mse,penalty=penalty,trainable_objective=t,total_objective=floor+t)


def replay(backend,rows,jobs,emit_state,emit_trial):
    first=None
    while True:
        theta=jobs.current;k=jobs.updates;point=measure(backend,theta,1.,rows,jobs)
        if first is None:first=point
        emit_state(k,theta,point)
        if k==50:reason='update_cap';break
        if point['total_l2']==0:reason='zero_gradient';break
        if point['directional_derivative']>=0:reason='non_descent_selected_direction';break
        accepted_any=False
        for j in range(d.TRIAL_CAPS[jobs.active[-1]]):
            alpha=2.**(-j);trial=a.readonly(theta-alpha*np.asarray(point['total_gradient']))
            jobs.trial_begin(j,trial);values=trial_objective(backend,trial,1.,point['fixed_other6_floor'],jobs)
            bound=point['trainable_objective']+1e-4*alpha*point['directional_derivative']
            original=bool(values['trainable_objective']<point['trainable_objective'] and values['trainable_objective']<=bound)
            mse_ok=bool(values['component3_mse']<=first['component3_mse']);gated=True
            accepted=original and (not gated or mse_ok)
            reason_trial='accepted' if accepted else 'armijo_and_mse_rejected' if gated and not original and not mse_ok else 'armijo_rejected' if not original else 'mse_envelope_rejected'
            decision=dict(armijo_pass=original,mse_envelope_pass=mse_ok,mse_gate_applied=gated,accepted=accepted,reason=reason_trial)
            d.p.require(jobs.trial_end(values)==decision,'independent acceptance predicate differs')
            emit_trial(dict(from_extra_updates=k,trial_index=j,alpha=alpha,**values,start500_component3_mse=first['component3_mse'],
                slope=point['directional_derivative'],armijo_bound=bound,**decision))
            if accepted:jobs.commit(trial);accepted_any=True;break
        if not accepted_any:reason='line_search_exhausted';break
    jobs.finish(reason);return first,point,jobs.current,reason


def evidence(first,final,updates,opt):
    t0,t,g0,g,m0,m=(row[field] for row,field in [(first,'trainable_objective'),(final,'trainable_objective'),(first,'total_l2'),(final,'total_l2'),(first,'component3_mse'),(final,'component3_mse')])
    d.p.require(t<=t0 and m<=m0,'independent objective/gate invalid')
    tr=t/t0 if t0 else None;gr=g/g0 if g0 else None
    lo=tr is not None and tr<=.99;go=gr is not None and gr<=.1;cue=bool(t0>0 and g0>0 and lo and go)
    status='reference_zero' if not t0 or not g0 else 'joint_optimization_cue' if cue else 'partial_target_reduction' if t<t0 else 'no_target_reduction'
    progress=updates>0 and t<t0 and m<=m0
    return dict(trainable_ratio=tr,gradient_ratio=gr,objective_target_met=lo,gradient_target_met=go,joint_cue=cue,status=status,
        start500_mse_envelope_met=m<=m0,finite_fit_progress=progress,protected_joint_fit_cue=progress and cue,
        update_status='no_parameter_update' if updates==0 else 'updated')


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



def audit_payload(cfg,science_pin,binding_pin):
    a.validate_call(cfg,science_pin,binding_pin)
    forbidden=['experiments.run_libero_task7_mse_guard_finer_grid','experiments.run_libero_task7_fixed_representation_optimization',
        'experiments.run_libero_task7_endpoint_gradient_diagnostic','experiments.run_libero_task7_endpoint_gradient_diagnostic_v2']
    d.p.require(all(name not in sys.modules for name in forbidden),'primary numerical implementation imported')
    directory=Path(cfg['artifacts']);output=Path(cfg['output'])
    d.p.require(not any((directory/n).exists() for n in ['independent-audit.json','completion.json','audit-failed-partial.json']),'existing audit; no retry')
    limits=a.base.Limits(600,805306368);runtime=a.runtime(cfg,audit=True);started=a.base.now();history=a.History(cfg)
    record=json.loads(output.read_text());result_pin=d.p.pin(output)
    d.p.require(record['science_config']==science_pin and record['binding']==binding_pin and record['parents']==cfg['parents']
        and record['inputs']==cfg['inputs'] and record['budget']==cfg['budget'] and record['status']=='complete_mse_guard_finer_grid_pending_audit'
        and record['formal_result'] is False and record['original_Policy_CNN_environment_or_sealed_calls']==0
        and record['prior_joint_cues_G_failed_H_unrun_unchanged'] is True,'primary provenance differs')
    d.p.require({k:record['runtime'][k] for k in ('python','numpy','torch')}==cfg['execution']['runtime']
        and record['runtime']['environment']==cfg['execution']['environment'] and record['cost']['elapsed_seconds']<=300
        and record['cost']['peak_RSS_bytes']<=536870912,'primary runtime budget differs')
    names=['endpoints','parameters','states','trials','heads','pairs','scaler-parity'];newpaths=[directory/f'head-{i:03d}.npz' for i in range(18)]
    d.p.require(set(record['tables'])==set(names) and len(record['new_heads'])==18,'primary outputs missing')
    for i,head in enumerate(record['new_heads']):d.p.require(a.identity(head)==a.identity(cfg['inputs']['heads'][i]) and Path(head['path'])==newpaths[i],'new archive lineage differs')
    for name in names:d.p.require(Path(record['tables'][name]['path'])==directory/(name+'.csv'),'table path differs')
    for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'table bytes changed')
    d.p.require(sum(Path(r['path']).stat().st_size for r in record['tables'].values())<=67108864 and sum(p.stat().st_size for p in newpaths)<=67108864,'output size exceeded')
    jobs=a.Jobs(cfg,limits,audit=True);work=a.RepresentationWork(cfg)
    guard=a.Guard(cfg,[r['path'] for r in record['tables'].values()],newpaths,limits,jobs,record['new_heads'])
    checks=Checks();tables={};states=[];trials=[];endpoints=[];heads=[];pairs=[];parities=[];own_digests=[];state_index=0
    try:
        with guard:
            shared=a.load(cfg['inputs']['shared'],'shared');y=shared['targets'];lengths=cfg['inputs']['action_lengths']
            expected=[(i,t) for i,n in enumerate(lengths) for t in range(1,n)]
            d.p.require(y.dtype==np.float32 and y.shape==(4188,7) and np.isfinite(y).all() and (abs(y)<=1).all()
                and shared['demo_indices'].dtype==shared['action_steps'].dtype==shared['demo_lengths'].dtype==np.int32
                and list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==expected
                and shared['demo_lengths'].tolist()==lengths and shared['tail20'].dtype==np.bool_
                and shared['tail20'].tolist()==[t>=lengths[i]-20 for i,t in expected],'independent causal source invalid')
            tables={name:CheckedTable(ref,checks,limits) for name,ref in record['tables'].items()}
            for model in cfg['inputs']['models']:
                cache=a.load(model['current_archive'],'current');f,p0=cache['features'],cache['predictions']
                d.p.require(f.dtype==p0.dtype==np.float32 and f.shape==(4188,256) and p0.shape==(4188,7)
                    and np.isfinite(f).all() and np.isfinite(p0).all() and (abs(p0)<=1).all(),'independent cache invalid')
                for plan in cfg['inputs']['folds']:
                    rkey=(model['seed'],model['arm'],plan['fold']);h=next(h for h in cfg['inputs']['heads'] if a.key(h)==rkey+(d.source.MODE,))
                    head=a.load(h['archive'],'head');work.begin(rkey,len(plan['fit_rows']));rows,z=prepare(head,f,plan,lengths);work.built(rkey,z)
                    context=jobs.register_fit(model,plan,z,y,p0,rows,work);backend=TorchFit(context)
                    start=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];jobs.source(h,head,start)
                    ident=a.identity(h);results={};points=[start];update_values=[];initials=[]
                    for opt in d.ARMS:
                        if opt==d.ARMS[1]:work.reuse(rkey,z)
                        pathkey=a.key(h)+(opt,);jobs.head(pathkey,*context,rows,start);pid={**ident,'optimization_arm':opt}
                        def state_sink(k,theta,row):
                            nonlocal state_index
                            d.p.require(state_index<len(record['states']),'extra independent state')
                            primary_digest=record['states'][state_index]['parameter_digest']
                            d.p.require(type(primary_digest) is str and len(primary_digest)==64 and all(c in '0123456789abcdef' for c in primary_digest),'invalid primary within-process digest')
                            own_digests.append(a.digest(theta))
                            entry={**pid,'extra_updates':k,'parameter_digest':primary_digest,**{f:v for f,v in row.items() if f not in VECTORS}}
                            states.append(entry);tables['states'].row(entry);state_index+=1
                            if opt==d.ARMS[0]:history.row('states',entry)
                        def trial_sink(row):
                            entry={**pid,**row};trials.append(entry);tables['trials'].row(entry)
                            if opt==d.ARMS[0]:history.row('trials',entry)
                        first,final,theta,reason=replay(backend,rows,jobs,state_sink,trial_sink)
                        ref=next(r for r in cfg['reference_points'] if all(r[k]==v for k,v in ident.items()))
                        for field,value in ref['start500'].items():checks.equal(first[field],value,'parent500 '+field)
                        checks.equal(first['n'],ref['n'],'parent rows')
                        initials.append(first);updates=jobs.last_finished[3];points.append(theta);update_values.append(updates)
                        ep_specs=([('saved500',0,start,first)] if opt==d.ARMS[0] else [])+[(opt+'_final',updates,theta,final)]
                        for condition,k,params,point in ep_specs:
                            ep={**ident,'condition':condition,'extra_updates':k,**point,'parameters':params.tolist()};endpoints.append(ep)
                            tables['endpoints'].row({f:v for f,v in ep.items() if f not in VECTORS+['parameters']})
                            for j in range(257):tables['parameters'].row({**ident,'condition':condition,'coordinate':j,'parameter':float(params[j]),**{f:point[f][j] for f in VECTORS}})
                            if opt==d.ARMS[0]:history.row('endpoints',ep)
                        done=jobs.done[pathkey];accepted=[r['alpha'] for r in trials if all(r[k]==v for k,v in pid.items()) and r['accepted']]
                        summary={**pid,'extra_updates':updates,'trials':done['trials'],'terminal_reason':reason,
                            'initial_objective':first['total_objective'],'final_objective':final['total_objective'],
                            'initial_trainable_objective':first['trainable_objective'],'final_trainable_objective':final['trainable_objective'],
                            'initial_gradient_l2':first['total_l2'],'final_gradient_l2':final['total_l2'],
                            'initial_component3_mse':first['component3_mse'],'final_component3_mse':final['component3_mse'],
                            'initial_penalty':first['penalty'],'final_penalty':final['penalty'],
                            'mse_vetoes_of_armijo_passes':done['mse_vetoes'],
                            'alpha_min':min(accepted) if accepted else None,'alpha_median':float(statistics.median(accepted)) if accepted else None,'alpha_max':max(accepted) if accepted else None,
                            'exhaustion_class':('armijo_feasible_but_mse_vetoed' if done['round_summaries'][-1]['armijo_passes'] else 'no_original_armijo_trial') if reason=='line_search_exhausted' else 'not_exhausted',
                            **evidence(first,final,updates,opt)}
                        heads.append(summary);results[opt]=summary;tables['heads'].row(summary)
                        if opt==d.ARMS[0]:history.row('heads',summary)
                    d.p.require(initials[0]==initials[1],'independent paired initial readout differs')
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
                    pairs.append(paired);tables['pairs'].row(paired)
                    newref=next(ref for ref in record['new_heads'] if a.key(ref)==a.key(h));saved=a.load(newref,'newhead')
                    for field in ('fit_rows','fit_demo_indices','scaler_mean','scaler_scale'):
                        d.p.require(saved[field].dtype==head[field].dtype and np.array_equal(saved[field],head[field]),'saved source fit/scaler differs')
                    d.p.require(saved['additional_updates'].dtype==np.int32 and saved['additional_updates'].tolist()==[0,*update_values]
                        and saved['weights'].dtype==saved['bias'].dtype==np.float64 and saved['weights'].shape==(3,256) and saved['bias'].shape==(3,),'saved paired endpoint schema differs')
                    coords=np.c_[saved['weights'],saved['bias']];d.p.require(np.array_equal(coords[0],start),'saved500 start changed')
                    for actual,wanted in zip(coords[1:].ravel(),np.stack(points[1:]).ravel()):checks.equal(float(actual),float(wanted),'saved final coordinate')
                    parity=dict(seed=model['seed'],arm=model['arm'],fold=plan['fold'],fit_rows=len(rows),paired_representation_same_object=True,source500_start_same_parameters=True)
                    parities.append(parity);tables['scaler-parity'].row(parity);del z,context,backend,head,saved
                del cache,f,p0;print(json.dumps({'stage':'audit','seed':model['seed'],'arm':model['arm'],'checks':checks.n,'updates':jobs.counts['updates']}),flush=True)
            jobs.complete();work.complete();guard.complete();historical=history.complete()
            for table in tables.values():table.finish()
            for name,value in [('states',states),('trials',trials),('endpoints',endpoints),('heads',heads),('pairs',pairs),('scaler_parity',parities),('historical_coarse_parity',historical),*summaries(heads,pairs).items()]:deep_check(checks,record[name],value,name)
            deep_check(checks,record['trial_counts_by_arm'],jobs.trials_by_arm,'per-arm trial counts')
            counts=dict(jobs.counts);counts['autograd_calls']=0
            deep_check(checks,record['computation'],counts,'primary computation');deep_check(checks,record['representation_work'],work.record(),'primary representation')
        access=record['access']
        d.p.require(access['NPZ_loads']=={'attempted':25,'completed':25} and access['member_decodes']=={'attempted':161,'completed':161}
            and len(access['byte_SHA_checks'])==68 and set(access['new_NPZ_writes'])==set(map(str,newpaths)) and set(access['files'])==guard.original
            and len(access['new_member_writes'])==126 and {tuple(v) for v in access['new_member_writes']}=={(str(path),field) for path in newpaths for field in d.NEW_HEAD_MEMBERS}
            and {tuple(v) for v in access['members']}=={v for v in guard.member_completed if v[0] in guard.original}
            and not access['denied_operations'] and access['input_research_CSV_reads']==access['original_Policy_CNN_environment_or_sealed_calls']==0,'primary IO ledger differs')
        expected_rows={'endpoints':54,'parameters':13878,'states':jobs.counts['gradient_states'],'trials':jobs.counts['trial_forwards'],'heads':36,'pairs':18,'scaler-parity':18}
        d.p.require({k:r['rows'] for k,r in record['tables'].items()}==expected_rows and sum(expected_rows.values())<=cfg['budget']['output_CSV_rows_max'],'primary CSV budget differs')
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'table changed during audit')
        d.p.require(d.p.pin(output)==result_pin,'main JSON changed during audit')
        audit=dict(status='passed_independent_mse_guard_finer_grid_audit',formal_result=False,science_config=science_pin,binding=binding_pin,result=result_pin,
            runtime=runtime,cost=limits.record(),started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),computation=jobs.counts,
            representation_work=work.record(),historical_coarse_parity=historical,trial_counts_by_arm=jobs.trials_by_arm,scalar_checks=checks.n,replay_state_parameter_digests=own_digests,
            parameter_digest_policy='per_process_immutability;cross_implementation_coordinates_abs1e-10',primary_numerical_implementation_imported=False,
            original_Policy_CNN_environment_or_sealed_calls=0)
        path=directory/'independent-audit.json';a.base.write_new(path,audit)
        a.base.write_new(directory/'completion.json',dict(status='complete_independently_audited_mse_guard_finer_grid',formal_result=False,science_config=science_pin,binding=binding_pin,
            result=result_pin,audit=d.p.pin(path),completed_at_utc=a.base.now(),main_auxiliary_updates=jobs.counts['updates'],audit_replay_updates=jobs.counts['updates'],
            total_NPZ_loads=68,total_member_decodes=448,total_archive_SHA_checks=154,original_Policy_CNN_environment_or_sealed_calls=0))
        return audit
    except BaseException as error:
        for table in tables.values():
            if not table.handle.closed:table.handle.close()
        a.base.write_new(directory/'audit-failed-partial.json',dict(status='failed_no_retry',error=repr(error),access=guard.record(),computation=jobs.counts,
            representation_work=work.record(),scalar_checks=checks.n));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);args=parser.parse_args()
    cfg,_=a.preflight(args.config,args.binding);result=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':result['status']}))
