"""Independent Torch Float64 autograd audit of fit-only frozen endpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from experiments import libero_task7_endpoint_gradient_protocol as d
from experiments import libero_task7_endpoint_gradient_access as a
from experiments.audit_libero_task7_direction_harm_alignment import deep_check
from experiments.audit_libero_task7_component3_history_transition import Checks,CheckedTable

VECTORS=['base_gradient','penalty_gradient','total_gradient','direction']


def autograd_measure(z,y,p0,w,b,coefficient,ids,jobs=None):
    import torch
    x=torch.tensor(z,dtype=torch.float64);target=torch.tensor(y,dtype=torch.float64);original=torch.tensor(p0,dtype=torch.float64)
    wt=torch.tensor(w,dtype=torch.float64,requires_grad=True);bt=torch.tensor([b],dtype=torch.float64,requires_grad=True)
    if jobs:jobs.begin(jobs.pending,len(z),(wt,bt))
    raw=original[:,2]+torch.nn.functional.linear(x,wt[None],bt).squeeze(1);pred=torch.clamp(raw,-1.,1.)
    e=(pred-target[:,2]).square();e0=(original[:,2]-target[:,2]).square();diff=e-e0
    other=[0,1,3,4,5,6];floor=(original[:,other]-target[:,other]).square().sum(1).mean()/7
    base=floor+e.mean()/7;penalty=torch.relu(diff).mean()/7;total=base if coefficient==0 else base+penalty
    grads=[]
    for objective in (base,penalty,total):
        parts=torch.autograd.grad(objective,(wt,bt),retain_graph=True);grads.append(np.concatenate([p.detach().numpy() for p in parts]))
    gb,gp,g=grads;norms=[math.sqrt(math.fsum(float(x)*float(x) for x in vec)) for vec in grads];norm=norms[2]
    v=-g/norm if norm else np.zeros_like(g)
    tangent=torch.nn.functional.linear(x,torch.tensor(v[:-1][None]),torch.tensor(v[-1:])).squeeze(1).detach().numpy()
    rv,pv,ev,eov=(t.detach().numpy() for t in (raw,pred,e,e0));deltas=[];clips=[];hinges=[]
    d.p.require(np.isfinite(rv).all() and np.isfinite(tangent).all(),'independent nonfinite linear form')
    for r,p,yy,eo,ee,draw in zip(rv,pv,y[:,2],eov,ev,tangent):
        clips.append(0 if r<-1 else 1 if r==-1 else 2 if r<1 else 3 if r==1 else 4)
        hinges.append(0 if ee<eo else 1 if ee==eo else 2)
        dp=draw if -1<r<1 else max(draw,0) if r==-1 else min(draw,0) if r==1 else 0.
        de=2*(p-yy)*dp;dh=de if ee>eo else max(de,0.) if ee==eo else 0.
        deltas.append((de+coefficient*dh)/7)
    rawdiff=ev-eov;factor=2*(pv-y[:,2])/7*((rv>=-1)&(rv<=1))*(1+coefficient*(rawdiff>0))
    sample=math.fsum(float(abs(f)*math.sqrt(math.fsum(float(t)*float(t) for t in row)+1)) for f,row in zip(factor,z))/len(z)
    dot=math.fsum(float(x)*float(y) for x,y in zip(gb,gp));anchor=pv.astype(np.float32).astype(np.float64);ae=(anchor-y[:,2])**2
    fc=float(floor.item());ao=fc+float(ae.mean())/7+coefficient*float(np.maximum(ae-eov,0).mean()/7)
    digest=hashlib.sha256(np.asarray(ids,dtype='<i4').tobytes()+bytes(clips)+bytes(hinges)).hexdigest()
    result={'n':len(z),'fixed_other6_floor':fc,'component3_mse':float(e.mean().item()),'base_objective':float(base.item()),'penalty':float(penalty.item()),
        'trainable_objective':float((e.mean()/7+coefficient*penalty).item()),'total_objective':float(total.item()),'anchor_objective':ao,
        'anchor_max_action_delta':float(max(abs(anchor-pv))),'anchor_harm_disagreements':int(np.count_nonzero((ae>eov)!=(ev>eov))),
        'base_l2':norms[0],'penalty_l2':norms[1],'total_l2':norm,'total_rms':norm/math.sqrt(len(g)),'total_linf':float(max(abs(g))),
        'base_penalty_dot':dot,'base_penalty_cosine':dot/(norms[0]*norms[1]) if norms[0] and norms[1] else None,
        'mean_individual_total_gradient_l2':sample,'cancellation_ratio':norm/sample if sample else None,
        'directional_derivative':math.fsum(deltas)/len(deltas),'branch_digest':digest}
    for i,name in enumerate(['below','low_tie','inside','high_tie','above']):result['clip_'+name+'_n']=clips.count(i)
    for i,name in enumerate(['negative','zero','positive']):result['hinge_'+name+'_n']=hinges.count(i)
    result.update({name:value.tolist() for name,value in zip(VECTORS,[gb,gp,g,v])})
    if jobs:jobs.end()
    return result


def prepare(head,features,plan,lengths):
    pairs=[(demo,t) for demo,n in enumerate(lengths) for t in range(1,n)]
    fit=np.asarray([i for i,(demo,_) in enumerate(pairs) if demo//10!=plan['fold']],np.int32)
    demos=np.asarray([i for i in range(30) if i//10!=plan['fold']],np.int32)
    for field,expected in [('fit_rows',fit),('fit_demo_indices',demos),('updates',np.asarray([0,100,250,500],np.int32)),('optimizer_steps',np.arange(1,501,dtype=np.int32))]:
        d.p.require(head[field].dtype==np.int32 and np.array_equal(head[field],expected),'independent fit/update role mismatch')
    d.p.require(fit.tolist()==plan['fit_rows'],'independent fold mismatch')
    raw=features[fit].astype(np.float64);mean=np.sum(raw,axis=0)/len(fit);scale=np.sqrt(np.sum((raw-mean)**2,axis=0)/len(fit));scale[scale<=1e-6]=1
    for key,expected in [('scaler_mean',mean),('scaler_scale',scale)]:d.p.require(head[key].dtype==np.float64 and head[key].shape==(256,) and np.allclose(head[key],expected,atol=1e-10,rtol=0),'independent scaler mismatch')
    d.p.require(head['weights'].dtype==head['bias'].dtype==np.float32 and head['weights'].shape==(4,256) and head['bias'].shape==(4,) and not head['weights'][0].any() and head['bias'][0]==0 and np.isfinite(head['weights'][[0,3]]).all() and np.isfinite(head['bias'][[0,3]]).all(),'independent head schema mismatch')
    return fit,((raw-head['scaler_mean'])/head['scaler_scale']).astype(np.float32).astype(np.float64)


def summaries(points):
    lookup={(r['seed'],r['arm'],r['fold'],r['mode'],r['point']):r for r in points};pairs=[];initial=[]
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            for fold in range(3):
                zeros=[lookup[seed,arm,fold,mode,0] for mode in d.CASES];difference=max(abs(x-y) for x,y in zip(zeros[0]['total_gradient'],zeros[1]['total_gradient']))
                d.p.require(difference==0 and all(z['penalty']==z['penalty_l2']==0 for z in zeros) and zeros[0]['total_objective']==zeros[1]['total_objective'],'initial invariant mismatch')
                initial.append(dict(task=7,seed=seed,arm=arm,fold=fold,loss_difference=zeros[1]['total_objective']-zeros[0]['total_objective'],gradient_max_difference=difference,penalty_zero=True))
                for mode in d.CASES:
                    x,y=(lookup[seed,arm,fold,mode,t] for t in (0,500));g0,gf,dd=x['total_l2'],y['total_l2'],y['directional_derivative']
                    ratio=gf/g0 if g0 else None;descent=-dd/g0 if g0 else None
                    status='reference_zero' if not g0 else 'zero_gradient' if not gf else 'remaining_first_order_cue' if dd<-1e-10 and descent>=.1 else 'weak_descent_in_direction' if dd<-1e-10 else 'no_descent_in_direction'
                    d.p.require(x['fixed_other6_floor']==y['fixed_other6_floor'],'floor changed')
                    pairs.append(dict(task=7,seed=seed,arm=arm,fold=fold,mode=mode,initial_objective=x['total_objective'],endpoint_objective=y['total_objective'],
                        initial_trainable_objective=x['trainable_objective'],endpoint_trainable_objective=y['trainable_objective'],relative_trainable_objective=y['trainable_objective']/x['trainable_objective'] if x['trainable_objective'] else None,
                        initial_gradient_l2=g0,endpoint_gradient_l2=gf,endpoint_directional_derivative=dd,status=status,relative_gradient=ratio,normalized_descent=descent,remaining_cue=status=='remaining_first_order_cue'))
    seeds=[];arms=[]
    for arm in d.p.ARMS:
        for mode in d.CASES:
            chosen=[r for r in pairs if r['arm']==arm and r['mode']==mode]
            for seed in d.p.SEEDS:
                part=sorted((r for r in chosen if r['seed']==seed),key=lambda r:r['fold']);metrics={}
                for field in ('relative_trainable_objective','relative_gradient','normalized_descent'):
                    values=[r[field] for r in part];metrics[field]={'fold_values':values,'equal_fold_mean':math.fsum(values)/3 if None not in values else None}
                seeds.append(dict(arm=arm,mode=mode,seed=seed,metrics=metrics,all3fold_cue=all(r['remaining_cue'] for r in part)))
            arms.append(dict(arm=arm,mode=mode,seeds=d.p.SEEDS,seed_all3fold_cues=[r['all3fold_cue'] for r in seeds if r['arm']==arm and r['mode']==mode],all9head_remaining_cue=all(r['remaining_cue'] for r in chosen),automatic_training_promotion=False))
    return pairs,initial,seeds,arms


def audit_payload(cfg,science_pin,binding_pin):
    d.p.require('experiments.run_libero_task7_endpoint_gradient_diagnostic' not in sys.modules,'primary numerics imported')
    directory=Path(cfg['artifacts']);d.p.require(not any((directory/x).exists() for x in ['independent-audit.json','completion.json','audit-failed-partial.json']),'existing audit; no retry')
    limits=a.base.Limits(180);runtime=a.runtime(cfg,audit=True);started=a.base.now();output=Path(cfg['output']);record=json.loads(output.read_text());result_pin=d.p.pin(output)
    d.p.require(record['science_config']==science_pin and record['binding']==binding_pin and record['inputs']==cfg['inputs'] and record['budget']==cfg['budget'] and record['status']=='complete_fit_endpoint_gradient_pending_audit','main provenance differs')
    d.p.require(record['formal_result'] is False and record['parents']==cfg['parents'] and record['parameter_updates_original_Policy_CNN_environment_or_sealed_calls']==0 and record['G_failed_H_unrun_and_prior_joint_cues_unchanged'] is True,'main scope differs')
    d.p.require({k:record['runtime'][k] for k in ('python','numpy','torch')}==cfg['execution']['runtime'] and record['runtime']['environment']==cfg['execution']['environment']
        and record['cost']['elapsed_seconds']<=180 and record['cost']['peak_RSS_bytes']<=536870912,'main runtime budget differs')
    d.p.require({k:r['rows'] for k,r in record['tables'].items()}==cfg['budget']['tables'] and sum(Path(r['path']).stat().st_size for r in record['tables'].values())<=67108864,'main output budget differs')
    for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output bytes changed')
    jobs=a.Jobs(cfg,audit=True);guard=a.Guard(cfg,[r['path'] for r in record['tables'].values()],limits,jobs);checks=Checks();tables={};points=[]
    try:
        with guard:
            shared=a.load(cfg['inputs']['shared'],'shared');y=shared['targets'];lengths=cfg['inputs']['action_lengths'];expected=[(demo,t) for demo,n in enumerate(lengths) for t in range(1,n)]
            d.p.require(list(zip(shared['demo_indices'].tolist(),shared['action_steps'].tolist()))==expected and y.dtype==np.float32 and y.shape==(4188,7) and np.isfinite(y).all() and (abs(y)<=1).all()
                and shared['demo_indices'].dtype==shared['action_steps'].dtype==shared['demo_lengths'].dtype==np.int32 and shared['demo_lengths'].tolist()==lengths
                and shared['tail20'].dtype==np.bool_ and shared['tail20'].tolist()==[t>=lengths[demo]-20 for demo,t in expected],'independent source target/schema mismatch')
            tables={name:CheckedTable(ref,checks,limits) for name,ref in record['tables'].items()}
            for model in cfg['inputs']['models']:
                cache=a.load(model['current_archive'],'current');f,p0=cache['features'],cache['predictions']
                d.p.require(f.shape==(4188,256) and p0.shape==(4188,7) and f.dtype==p0.dtype==np.float32 and np.isfinite(f).all() and np.isfinite(p0).all() and (abs(p0)<=1).all(),'independent cache mismatch')
                for plan in cfg['inputs']['folds']:
                    previous=None
                    for mode,coefficient in d.CASES.items():
                        h=next(h for h in cfg['inputs']['heads'] if (h['seed'],h['arm'],h['fold'],h['mode'])==(model['seed'],model['arm'],plan['fold'],mode));head=a.load(h['archive'],'head');rows,z=prepare(head,f,plan,lengths)
                        bits=head['scaler_mean'].tobytes()+head['scaler_scale'].tobytes();d.p.require(previous is None or previous==bits,'paired scalers differ');previous=bits
                        for pos,point in ((0,0),(3,500)):
                            jobs.pending=(model['seed'],model['arm'],plan['fold'],mode,point)
                            rec=dict(task=7,seed=model['seed'],arm=model['arm'],fold=plan['fold'],mode=mode,point=point,
                                **autograd_measure(z,y[rows].astype(np.float64),p0[rows].astype(np.float64),head['weights'][pos].astype(np.float64),float(head['bias'][pos]),coefficient,rows,jobs))
                            points.append(rec);tables['points'].row({k:v for k,v in rec.items() if k not in VECTORS});param=np.r_[head['weights'][pos],head['bias'][pos]]
                            for j in range(257):tables['parameters'].row({**{k:rec[k] for k in ('task','seed','arm','fold','mode','point')},'coordinate':j,'parameter':float(param[j]),**{k:rec[k][j] for k in VECTORS}})
                        del z,head
                del cache,f,p0;limits.check();print(json.dumps({'stage':'audit','seed':model['seed'],'arm':model['arm'],'checks':checks.n}),flush=True)
            pairs,initial,seeds,arms=summaries(points)
            for row in pairs:tables['pairs'].row(row)
            for row in initial:tables['initial-parity'].row(row)
            for t in tables.values():t.finish()
            for key,value in [('points',points),('pairs',pairs),('initial_parity',initial),('seed_summaries',seeds),('arm_summaries',arms)]:deep_check(checks,record[key],value,key)
            jobs.complete();guard.complete()
        expected_counts=dict(jobs.counts);expected_counts['autograd_calls']=0;deep_check(checks,record['computation'],expected_counts,'primary computation')
        d.p.require(record['access']['NPZ_loads']=={'attempted':43,'completed':43} and record['access']['member_decodes']=={'attempted':305,'completed':305} and len(record['access']['byte_SHA_checks'])==86 and not record['access']['denied_operations'],'primary access budget mismatch')
        d.p.require(set(record['access']['files'])==set(guard.refs) and {tuple(v) for v in record['access']['members']}==set(guard.member_completed),'primary source identity coverage differs')
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output changed during audit')
        d.p.require(d.p.pin(output)==result_pin,'main JSON changed')
        audit={'status':'passed_independent_fit_endpoint_gradient_audit','formal_result':False,'science_config':science_pin,'binding':binding_pin,'result':result_pin,'runtime':runtime,'cost':limits.record(),
            'started_at_utc':started,'completed_at_utc':a.base.now(),'access':guard.record(),'computation':jobs.counts,'scalar_checks':checks.n,'primary_numerical_implementation_imported':False,'parameter_updates_original_Policy_CNN_environment_or_sealed_calls':0}
        path=directory/'independent-audit.json';a.base.write_new(path,audit);a.base.write_new(directory/'completion.json',{'status':'complete_independently_audited_fit_endpoint_gradient','formal_result':False,'science_config':science_pin,'binding':binding_pin,'result':result_pin,'audit':d.p.pin(path),'completed_at_utc':a.base.now(),'total_NPZ_loads':86,'total_member_decodes':610,'parameter_updates_original_Policy_CNN_environment_or_sealed_calls':0});return audit
    except BaseException as error:
        for t in tables.values():
            if not t.handle.closed:t.handle.close()
        a.base.write_new(directory/'audit-failed-partial.json',{'status':'failed_no_retry','error':repr(error),'access':guard.record(),'computation':jobs.counts,'checks':checks.n});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);args=parser.parse_args()
    cfg,_=a.preflight(args.config,args.binding);r=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':r['status'],'checks':r['scalar_checks']}))
