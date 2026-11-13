"""Independent Float64 autograd and finite-path replay; no primary import."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from experiments import libero_task7_fixed_representation_optimization_protocol as d
from experiments import libero_task7_fixed_representation_optimization_access as a
from experiments.audit_libero_task7_endpoint_gradient_diagnostic import prepare
from experiments.audit_libero_task7_endpoint_gradient_diagnostic_v2 import admit_reused
from experiments.audit_libero_task7_direction_harm_alignment import deep_check
from experiments.audit_libero_task7_component3_history_transition import Checks,CheckedTable

VECTORS=['base_gradient','penalty_gradient','total_gradient','direction']


class TorchFit:
    def __init__(self,context):
        import torch
        self.context=context
        self.x,self.y,self.p0=(torch.tensor(v,dtype=torch.float64) for v in context)


def measure(backend,theta,coefficient,rows,jobs=None):
    import torch
    x,y,p0=backend.x,backend.y,backend.p0
    w=torch.tensor(theta[:-1],dtype=torch.float64,requires_grad=True);b=torch.tensor(theta[-1:],dtype=torch.float64,requires_grad=True)
    if jobs:jobs.check_context(*backend.context);jobs.tensor_input_id=id(x);jobs.state_begin(theta,(w,b))
    raw=p0[:,2]+torch.nn.functional.linear(x,w[None],b).squeeze(1);pred=raw.clamp(-1,1)
    e=(pred-y[:,2]).square();e0=(p0[:,2]-y[:,2]).square();diff=e-e0
    others=[0,1,3,4,5,6];floor=(p0[:,others]-y[:,others]).square().sum(1).mean()/7
    base=floor+e.mean()/7;penalty=torch.relu(diff).mean()/7;objective=base if coefficient==0 else base+penalty
    gradients=[]
    for term in (base,penalty,objective):
        pieces=torch.autograd.grad(term,(w,b),retain_graph=True);gradients.append(np.concatenate([v.detach().numpy() for v in pieces]))
    gb,gp,g=gradients;norms=[math.sqrt(math.fsum(float(t)*float(t) for t in v)) for v in gradients];direction=-g
    tangent=torch.nn.functional.linear(x,torch.tensor(direction[:-1][None]),torch.tensor(direction[-1:])).squeeze(1)
    rv=raw.detach();pv=pred.detach();dv=diff.detach()
    dp=torch.where((rv>-1)&(rv<1),tangent,torch.zeros_like(tangent))
    dp=torch.where(rv==-1,torch.clamp_min(tangent,0),dp);dp=torch.where(rv==1,torch.clamp_max(tangent,0),dp)
    de=2*(pv-y[:,2])*dp;dh=torch.where(dv>0,de,torch.where(dv==0,torch.clamp_min(de,0),torch.zeros_like(de)))
    slope=(de.mean()/7 if coefficient==0 else (de+dh).mean()/7).item()
    raw_np=rv.numpy();error=e.detach().numpy();old_error=e0.numpy();difference=dv.numpy()
    clip=np.select([raw_np<-1,raw_np==-1,raw_np<1,raw_np==1],[0,1,2,3],default=4).astype(np.uint8)
    hinge=np.select([difference<0,difference==0],[0,1],default=2).astype(np.uint8)
    rounded=pv.numpy().astype(np.float32).astype(np.float64);rounded_error=(rounded-backend.context[1][:,2])**2
    fc=float(floor.item());trainable=float((e.mean()/7 if coefficient==0 else e.mean()/7+penalty).item())
    anchor=fc+float(rounded_error.mean())/7+coefficient*float(np.maximum(rounded_error-old_error,0).mean()/7)
    row={'n':len(x),'fixed_other6_floor':fc,'component3_mse':float(e.mean().item()),'penalty':float(penalty.item()),
        'trainable_objective':trainable,'total_objective':float(objective.item()),'base_l2':norms[0],'penalty_l2':norms[1],'total_l2':norms[2],
        'total_rms':norms[2]/math.sqrt(len(g)),'total_linf':float(max(abs(g))),'directional_derivative':float(slope),
        'anchor_objective':anchor,'anchor_max_action_delta':float(max(abs(rounded-pv.numpy()))),
        'anchor_harm_disagreements':int(np.count_nonzero((error>old_error)!=(rounded_error>old_error))),
        'branch_digest':hashlib.sha256(np.asarray(rows,dtype='<i4').tobytes()+clip.tobytes()+hinge.tobytes()).hexdigest()}
    for i,name in enumerate(['below','low_tie','inside','high_tie','above']):row['clip_'+name+'_n']=int(np.count_nonzero(clip==i))
    for i,name in enumerate(['negative','zero','positive']):row['hinge_'+name+'_n']=int(np.count_nonzero(hinge==i))
    row.update({name:v.tolist() for name,v in zip(VECTORS,[gb,gp,g,direction])})
    d.p.require(np.isfinite(raw_np).all() and np.isfinite(tangent.numpy()).all() and all(np.isfinite(v).all() for v in gradients)
        and math.isfinite(slope) and row['anchor_max_action_delta']<=1e-7 and abs(anchor-row['total_objective'])<=1e-6,'independent invalid current-state numerics')
    if jobs:jobs.state_end(row)
    return row


def trial_objective(backend,theta,coefficient,floor,jobs=None):
    import torch
    if jobs:jobs.check_context(*backend.context);jobs.tensor_input_id=id(backend.x)
    with torch.no_grad():
        raw=backend.p0[:,2]+torch.nn.functional.linear(backend.x,torch.tensor(theta[:-1][None]),torch.tensor(theta[-1:])).squeeze(1)
        e=(raw.clamp(-1,1)-backend.y[:,2]).square();e0=(backend.p0[:,2]-backend.y[:,2]).square()
        value=e.mean()/7 if coefficient==0 else (e.mean()+torch.relu(e-e0).mean())/7
    result=float(value.item());d.p.require(torch.isfinite(raw).all().item() and np.isfinite(theta).all() and math.isfinite(result),'independent invalid trial')
    return result,floor+result


def replay(backend,coefficient,rows,jobs,emit_state,emit_trial):
    first=None
    while True:
        theta=jobs.current;k=jobs.updates;point=measure(backend,theta,coefficient,rows,jobs)
        if first is None:first=point
        emit_state(k,theta,point)
        if k==50:reason='update_cap';break
        if point['total_l2']==0:reason='zero_gradient';break
        if point['directional_derivative']>=0:reason='non_descent_selected_direction';break
        accepted_any=False
        for j in range(12):
            alpha=2.**(-j);trial=a.readonly(theta-alpha*np.asarray(point['total_gradient']))
            jobs.trial_begin(j,trial);t,f=trial_objective(backend,trial,coefficient,point['fixed_other6_floor'],jobs)
            bound=point['trainable_objective']+1e-4*alpha*point['directional_derivative']
            accepted=bool(t<point['trainable_objective'] and t<=bound)
            d.p.require(jobs.trial_end(t)==accepted,'independent acceptance predicate differs')
            emit_trial(dict(from_extra_updates=k,trial_index=j,alpha=alpha,trainable_objective=t,total_objective=f,slope=point['directional_derivative'],armijo_bound=bound,accepted=accepted))
            if accepted:jobs.commit(trial);accepted_any=True;break
        if not accepted_any:reason='line_search_exhausted';break
    final=jobs.current;jobs.finish(reason);return first,point,final,reason


def evidence(t0,t,g0,g):
    d.p.require(t<=t0,'independent nonmonotone path')
    tr=t/t0 if t0 else None;gr=g/g0 if g0 else None
    lo=tr is not None and tr<=.99;go=gr is not None and gr<=.1;cue=bool(t0>0 and g0>0 and lo and go)
    status='reference_zero' if not t0 or not g0 else 'joint_optimization_cue' if cue else 'partial_target_reduction' if t<t0 else 'no_target_reduction'
    return dict(trainable_ratio=tr,gradient_ratio=gr,objective_target_met=lo,gradient_target_met=go,joint_cue=cue,status=status)


def summaries(heads):
    seeds=[];arms=[]
    for arm in d.p.ARMS:
        for mode in d.CASES:
            chosen=[h for h in heads if h['arm']==arm and h['mode']==mode]
            for seed in d.p.SEEDS:
                part=sorted([h for h in chosen if h['seed']==seed],key=lambda h:h['fold']);metrics={}
                for field in ('trainable_ratio','gradient_ratio'):
                    values=[h[field] for h in part];metrics[field]=dict(fold_values=values,equal_fold_mean=math.fsum(values)/3 if None not in values else None)
                seeds.append(dict(arm=arm,mode=mode,seed=seed,metrics=metrics,updates=[h['extra_updates'] for h in part],all3fold_cue=all(h['joint_cue'] for h in part)))
            d.p.require(len(chosen)==9 and {(h['seed'],h['fold']) for h in chosen}=={(s,f) for s in d.p.SEEDS for f in range(3)},'independent group coverage differs')
            arms.append(dict(arm=arm,mode=mode,seed_all3fold_cues=[h['all3fold_cue'] for h in seeds if h['arm']==arm and h['mode']==mode],all9head_cue=all(h['joint_cue'] for h in chosen),automatic_Policy_promotion=False))
    return seeds,arms


def audit_payload(cfg,science_pin,binding_pin):
    a.validate_call(cfg,science_pin,binding_pin)
    d.p.require(all(name not in sys.modules for name in ('experiments.run_libero_task7_fixed_representation_optimization','experiments.run_libero_task7_endpoint_gradient_diagnostic','experiments.run_libero_task7_endpoint_gradient_diagnostic_v2')),'primary numerical implementation imported')
    directory=Path(cfg['artifacts']);d.p.require(not any((directory/name).exists() for name in ['independent-audit.json','completion.json','audit-failed-partial.json']),'existing audit; no retry')
    limits=a.base.Limits(600,805306368);runtime=a.runtime(cfg,audit=True);started=a.base.now();output=Path(cfg['output']);record=json.loads(output.read_text());result_pin=d.p.pin(output)
    d.p.require(record['science_config']==science_pin and record['binding']==binding_pin and record['parents']==cfg['parents'] and record['inputs']==cfg['inputs'] and record['budget']==cfg['budget']
        and record['status']=='complete_finite_optimization_pending_audit' and record['formal_result'] is False and record['original_Policy_CNN_environment_or_sealed_calls']==0
        and record['prior_joint_cues_G_failed_H_unrun_unchanged'] is True,'primary provenance differs')
    d.p.require({k:record['runtime'][k] for k in ('python','numpy','torch')}==cfg['execution']['runtime'] and record['runtime']['environment']==cfg['execution']['environment']
        and record['cost']['elapsed_seconds']<=300 and record['cost']['peak_RSS_bytes']<=536870912,'primary runtime budget differs')
    names=['endpoints','parameters','states','trials','heads','scaler-parity'];newpaths=[directory/f'head-{i:03d}.npz' for i in range(36)]
    d.p.require(set(record['tables'])==set(names) and len(record['new_heads'])==36,'primary outputs missing')
    for i,head in enumerate(record['new_heads']):d.p.require(a.identity(head)==a.identity(cfg['inputs']['heads'][i]) and Path(head['path'])==newpaths[i],'new archive lineage/order differs')
    for name in names:d.p.require(Path(record['tables'][name]['path'])==directory/(name+'.csv'),'table path differs')
    for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'table bytes changed')
    d.p.require(sum(Path(r['path']).stat().st_size for r in record['tables'].values())<=67108864 and sum(p.stat().st_size for p in newpaths)<=67108864,'primary output size exceeded')
    jobs=a.Jobs(cfg,limits,audit=True);work=a.RepresentationWork(cfg);guard=a.Guard(cfg,[r['path'] for r in record['tables'].values()],newpaths,limits,jobs,record['new_heads'])
    checks=Checks();tables={};states=[];trials=[];endpoints=[];heads=[];parities=[];own_digests=[];state_index=0
    try:
        with guard:
            shared=a.load(cfg['inputs']['shared'],'shared');y=shared['targets'];lengths=cfg['inputs']['action_lengths'];expected=[(i,t) for i,n in enumerate(lengths) for t in range(1,n)]
            d.p.require(y.dtype==np.float32 and y.shape==(4188,7) and np.isfinite(y).all() and (abs(y)<=1).all()
                and shared['demo_indices'].dtype==shared['action_steps'].dtype==shared['demo_lengths'].dtype==np.int32
                and list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==expected and shared['demo_lengths'].tolist()==lengths
                and shared['tail20'].dtype==np.bool_ and shared['tail20'].tolist()==[t>=lengths[i]-20 for i,t in expected],'independent causal source schema differs')
            tables={name:CheckedTable(ref,checks,limits) for name,ref in record['tables'].items()}
            for model in cfg['inputs']['models']:
                cache=a.load(model['current_archive'],'current');f,p0=cache['features'],cache['predictions']
                d.p.require(f.dtype==p0.dtype==np.float32 and f.shape==(4188,256) and p0.shape==(4188,7) and np.isfinite(f).all() and np.isfinite(p0).all() and (abs(p0)<=1).all(),'independent cache schema differs')
                for plan in cfg['inputs']['folds']:
                    z=None;previous=None;rkey=(model['seed'],model['arm'],plan['fold'])
                    for mode,coefficient in d.CASES.items():
                        h=next(h for h in cfg['inputs']['heads'] if a.key(h)==rkey+(mode,));head=a.load(h['archive'],'head')
                        if z is None:
                            work.begin(rkey,len(plan['fit_rows']));rows,z=prepare(head,f,plan,lengths);work.built(rkey,z)
                            context=jobs.register_fit(model,plan,z,y,p0,rows,work);backend=TorchFit(context)
                        else:rows=admit_reused(head,plan,lengths,previous);work.reuse(rkey,z)
                        previous=head['scaler_mean'].tobytes()+head['scaler_scale'].tobytes();ident=a.identity(h)
                        start=np.r_[head['weights'][3].astype(np.float64),float(head['bias'][3])];jobs.head(a.key(h),*context,rows,start)
                        def state_sink(k,theta,row):
                            nonlocal state_index
                            d.p.require(state_index<len(record['states']),'extra independent state')
                            primary_digest=record['states'][state_index]['parameter_digest']
                            d.p.require(type(primary_digest) is str and len(primary_digest)==64 and all(c in '0123456789abcdef' for c in primary_digest),'invalid within-process parameter digest')
                            own_digests.append(a.digest(theta))
                            entry={**ident,'extra_updates':k,'parameter_digest':primary_digest,**{k:v for k,v in row.items() if k not in VECTORS}}
                            states.append(entry);tables['states'].row(entry);state_index+=1
                        def trial_sink(row):
                            entry={**ident,**row};trials.append(entry);tables['trials'].row(entry)
                        initial,final,theta,reason=replay(backend,coefficient,rows,jobs,state_sink,trial_sink)
                        ref=next(r for r in cfg['reference_points'] if all(r[k]==v for k,v in ident.items()))
                        for left,right in [('total_objective','start500_total_objective'),('trainable_objective','start500_trainable_objective'),('total_l2','start500_gradient_l2')]:checks.equal(initial[left],ref[right],'parent500 '+left)
                        checks.equal(initial['n'],ref['n'],'parent rows');checks.equal(initial['branch_digest'],ref['start500_branch_digest'],'parent branch')
                        updates=jobs.last_finished[3]
                        for condition,k,params,point in [('hold_500',0,start,initial),('fullbatch_armijo',updates,theta,final)]:
                            ep={**ident,'condition':condition,'extra_updates':k,**point,'parameters':params.tolist()};endpoints.append(ep)
                            tables['endpoints'].row({k:v for k,v in ep.items() if k not in VECTORS+['parameters']})
                            for j in range(257):tables['parameters'].row({**ident,'condition':condition,'coordinate':j,'parameter':float(params[j]),**{k:point[k][j] for k in VECTORS}})
                        summary={**ident,'extra_updates':updates,'trials':jobs.done[a.key(h)]['trials'],'terminal_reason':reason,
                            'initial_objective':initial['total_objective'],'final_objective':final['total_objective'],
                            'initial_trainable_objective':initial['trainable_objective'],'final_trainable_objective':final['trainable_objective'],
                            'initial_gradient_l2':initial['total_l2'],'final_gradient_l2':final['total_l2'],
                            'final_normalized_remaining_descent':-final['directional_derivative']/final['total_l2']/ref['zero_residual_gradient_l2'] if final['total_l2'] and ref['zero_residual_gradient_l2'] else 0. if ref['zero_residual_gradient_l2'] else None,
                            **evidence(initial['trainable_objective'],final['trainable_objective'],initial['total_l2'],final['total_l2'])}
                        heads.append(summary);tables['heads'].row(summary)
                        newref=next(item for item in record['new_heads'] if a.key(item)==a.key(h));saved=a.load(newref,'newhead')
                        for field in ('fit_rows','fit_demo_indices','scaler_mean','scaler_scale'):d.p.require(saved[field].dtype==head[field].dtype and np.array_equal(saved[field],head[field]),'saved fit/scaler differs')
                        d.p.require(saved['additional_updates'].dtype==np.int32 and saved['additional_updates'].tolist()==[0,updates]
                            and saved['weights'].dtype==saved['bias'].dtype==np.float64 and saved['weights'].shape==(2,256) and saved['bias'].shape==(2,),'saved final schema differs')
                        coords=np.c_[saved['weights'],saved['bias']]
                        d.p.require(np.array_equal(coords[0],start),'saved hold changed')
                        for actual,wanted in zip(coords[1],theta):checks.equal(float(actual),float(wanted),'saved final parameter')
                        del head,saved
                    parity=dict(seed=model['seed'],arm=model['arm'],fold=plan['fold'],fit_rows=len(rows),saved_scalers_bit_equal=True)
                    parities.append(parity);tables['scaler-parity'].row(parity);del z,context,backend
                del cache,f,p0;print(json.dumps({'stage':'audit','seed':model['seed'],'arm':model['arm'],'checks':checks.n,'updates':jobs.counts['updates']}),flush=True)
            jobs.complete();work.complete();guard.complete()
            for table in tables.values():table.finish()
            seeds,arms=summaries(heads)
            for name,value in [('states',states),('trials',trials),('endpoints',endpoints),('heads',heads),('scaler_parity',parities),('seed_summaries',seeds),('arm_summaries',arms)]:deep_check(checks,record[name],value,name)
            expected_counts=dict(jobs.counts);expected_counts['autograd_calls']=0;deep_check(checks,record['computation'],expected_counts,'primary work')
            deep_check(checks,record['representation_work'],work.record(),'primary preprocessing')
        access=record['access']
        d.p.require(access['NPZ_loads']=={'attempted':43,'completed':43} and access['member_decodes']=={'attempted':305,'completed':305}
            and len(access['byte_SHA_checks'])==122 and set(access['new_NPZ_writes'])==set(map(str,newpaths)) and set(access['files'])==guard.original
            and len(access['new_member_writes'])==252 and {tuple(v) for v in access['new_member_writes']}=={(str(path),field) for path in newpaths for field in d.NEW_HEAD_MEMBERS}
            and {tuple(v) for v in access['members']}=={v for v in guard.member_completed if v[0] in guard.original}
            and not access['denied_operations'] and access['input_research_CSV_reads']==access['original_Policy_CNN_environment_or_sealed_calls']==0,'primary IO ledger differs')
        expected_rows={'endpoints':72,'parameters':18504,'states':jobs.counts['gradient_states'],'trials':jobs.counts['trial_forwards'],'heads':36,'scaler-parity':18}
        d.p.require({k:r['rows'] for k,r in record['tables'].items()}==expected_rows and sum(expected_rows.values())<=42066,'primary table budget differs')
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'table changed during audit')
        d.p.require(d.p.pin(output)==result_pin,'main JSON changed during audit')
        audit=dict(status='passed_independent_finite_optimization_audit',formal_result=False,science_config=science_pin,binding=binding_pin,result=result_pin,runtime=runtime,cost=limits.record(),
            started_at_utc=started,completed_at_utc=a.base.now(),access=guard.record(),computation=jobs.counts,representation_work=work.record(),scalar_checks=checks.n,
            replay_state_parameter_digests=own_digests,parameter_digest_policy='per_process_immutability;new_cross_implementation_parameters_checked_numerically_not_by_bytes',primary_numerical_implementation_imported=False,original_Policy_CNN_environment_or_sealed_calls=0)
        path=directory/'independent-audit.json';a.base.write_new(path,audit)
        a.base.write_new(directory/'completion.json',dict(status='complete_independently_audited_finite_optimization',formal_result=False,science_config=science_pin,binding=binding_pin,result=result_pin,audit=d.p.pin(path),completed_at_utc=a.base.now(),
            main_auxiliary_updates=jobs.counts['updates'],audit_replay_updates=jobs.counts['updates'],total_NPZ_loads=122,total_member_decodes=862,total_archive_SHA_checks=280,original_Policy_CNN_environment_or_sealed_calls=0))
        return audit
    except BaseException as error:
        for table in tables.values():
            if not table.handle.closed:table.handle.close()
        a.base.write_new(directory/'audit-failed-partial.json',dict(status='failed_no_retry',error=repr(error),access=guard.record(),computation=jobs.counts,representation_work=work.record(),scalar_checks=checks.n));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);args=parser.parse_args()
    cfg,_=a.preflight(args.config,args.binding);result=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':result['status']}))
