"""Independent Torch Float64 autograd audit of fit-only frozen endpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import numpy as np
from experiments import libero_task7_endpoint_gradient_protocol_v2 as d
from experiments import libero_task7_endpoint_gradient_access_v2 as a
from experiments.audit_libero_task7_direction_harm_alignment import deep_check
from experiments.audit_libero_task7_component3_history_transition import Checks,CheckedTable

from experiments.audit_libero_task7_endpoint_gradient_diagnostic import autograd_measure, prepare, summaries, VECTORS


def admit_reused(head, plan, lengths, previous_bits):
    pairs = [(demo,t) for demo,n in enumerate(lengths) for t in range(1,n)]
    fit = np.asarray([i for i,(demo,_) in enumerate(pairs) if demo//10 != plan['fold']], np.int32)
    demos = np.asarray([demo for demo in range(30) if demo//10 != plan['fold']], np.int32)
    for field, value in [('fit_rows',fit), ('fit_demo_indices',demos), ('updates',np.asarray([0,100,250,500],np.int32)), ('optimizer_steps',np.arange(1,501,dtype=np.int32))]:
        d.p.require(head[field].dtype == np.int32 and np.array_equal(head[field],value), 'independent paired role mismatch')
    d.p.require(fit.tolist() == plan['fit_rows'], 'independent paired fold mismatch')
    d.p.require(head['weights'].dtype == head['bias'].dtype == np.float32 and head['weights'].shape == (4,256) and head['bias'].shape == (4,)
        and not head['weights'][0].any() and head['bias'][0] == 0 and np.isfinite(head['weights'][[0,3]]).all() and np.isfinite(head['bias'][[0,3]]).all(), 'independent paired endpoint mismatch')
    d.p.require(all(head[k].dtype == np.float64 and head[k].shape == (256,) for k in ('scaler_mean','scaler_scale'))
        and head['scaler_mean'].tobytes() + head['scaler_scale'].tobytes() == previous_bits, 'independent paired scaler mismatch')
    return fit


def audit_payload(cfg,science_pin,binding_pin):
    d.p.require(all(name not in sys.modules for name in ('experiments.run_libero_task7_endpoint_gradient_diagnostic','experiments.run_libero_task7_endpoint_gradient_diagnostic_v2')),'primary numerics imported')
    directory=Path(cfg['artifacts']);d.p.require(not any((directory/x).exists() for x in ['independent-audit.json','completion.json','audit-failed-partial.json']),'existing audit; no retry')
    limits=a.base.Limits(180);runtime=a.runtime(cfg,audit=True);started=a.base.now();output=Path(cfg['output']);record=json.loads(output.read_text());result_pin=d.p.pin(output)
    d.p.require(record['science_config']==science_pin and record['binding']==binding_pin and record['inputs']==cfg['inputs'] and record['budget']==cfg['budget'] and record['status']=='complete_fit_endpoint_gradient_pending_audit','main provenance differs')
    d.p.require(record['formal_result'] is False and record['parents']==cfg['parents'] and record['parameter_updates_original_Policy_CNN_environment_or_sealed_calls']==0 and record['G_failed_H_unrun_and_prior_joint_cues_unchanged'] is True,'main scope differs')
    d.p.require({k:record['runtime'][k] for k in ('python','numpy','torch')}==cfg['execution']['runtime'] and record['runtime']['environment']==cfg['execution']['environment']
        and record['cost']['elapsed_seconds']<=180 and record['cost']['peak_RSS_bytes']<=536870912,'main runtime budget differs')
    d.p.require({k:r['rows'] for k,r in record['tables'].items()}==cfg['budget']['tables'] and sum(Path(r['path']).stat().st_size for r in record['tables'].values())<=67108864,'main output budget differs')
    for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output bytes changed')
    jobs=a.Jobs(cfg,audit=True);work=a.RepresentationWork(cfg);guard=a.Guard(cfg,[r['path'] for r in record['tables'].values()],limits,jobs);checks=Checks();tables={};points=[]
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
                    previous=None;z=None
                    for mode,coefficient in d.CASES.items():
                        h=next(h for h in cfg['inputs']['heads'] if (h['seed'],h['arm'],h['fold'],h['mode'])==(model['seed'],model['arm'],plan['fold'],mode));head=a.load(h['archive'],'head');representation_key=(model['seed'],model['arm'],plan['fold'])
                        if z is None:
                            work.begin(representation_key,len(plan['fit_rows']));rows,z=prepare(head,f,plan,lengths);work.built(representation_key,z)
                        else:
                            rows=admit_reused(head,plan,lengths,previous);work.reuse(representation_key,z)
                        bits=head['scaler_mean'].tobytes()+head['scaler_scale'].tobytes();d.p.require(previous is None or previous==bits,'paired scalers differ');previous=bits
                        for pos,point in ((0,0),(3,500)):
                            jobs.pending=(model['seed'],model['arm'],plan['fold'],mode,point)
                            rec=dict(task=7,seed=model['seed'],arm=model['arm'],fold=plan['fold'],mode=mode,point=point,
                                **autograd_measure(z,y[rows].astype(np.float64),p0[rows].astype(np.float64),head['weights'][pos].astype(np.float64),float(head['bias'][pos]),coefficient,rows,jobs))
                            points.append(rec);tables['points'].row({k:v for k,v in rec.items() if k not in VECTORS});param=np.r_[head['weights'][pos],head['bias'][pos]]
                            for j in range(257):tables['parameters'].row({**{k:rec[k] for k in ('task','seed','arm','fold','mode','point')},'coordinate':j,'parameter':float(param[j]),**{k:rec[k][j] for k in VECTORS}})
                        del head
                    del z
                del cache,f,p0;limits.check();print(json.dumps({'stage':'audit','seed':model['seed'],'arm':model['arm'],'checks':checks.n}),flush=True)
            pairs,initial,seeds,arms=summaries(points)
            for row in pairs:tables['pairs'].row(row)
            for row in initial:tables['initial-parity'].row(row)
            for t in tables.values():t.finish()
            for key,value in [('points',points),('pairs',pairs),('initial_parity',initial),('seed_summaries',seeds),('arm_summaries',arms)]:deep_check(checks,record[key],value,key)
            jobs.complete();work.complete();guard.complete()
        deep_check(checks,record['representation_work'],work.record(),'preprocessing actual work');expected_counts=dict(jobs.counts);expected_counts['autograd_calls']=0;deep_check(checks,record['computation'],expected_counts,'primary computation')
        d.p.require(record['access']['NPZ_loads']=={'attempted':43,'completed':43} and record['access']['member_decodes']=={'attempted':305,'completed':305} and len(record['access']['byte_SHA_checks'])==86 and not record['access']['denied_operations'],'primary access budget mismatch')
        d.p.require(set(record['access']['files'])==set(guard.refs) and {tuple(v) for v in record['access']['members']}==set(guard.member_completed),'primary source identity coverage differs')
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output changed during audit')
        d.p.require(d.p.pin(output)==result_pin,'main JSON changed')
        audit={'status':'passed_independent_fit_endpoint_gradient_audit','formal_result':False,'science_config':science_pin,'binding':binding_pin,'result':result_pin,'runtime':runtime,'cost':limits.record(),
            'started_at_utc':started,'completed_at_utc':a.base.now(),'access':guard.record(),'computation':jobs.counts,'representation_work':work.record(),'scalar_checks':checks.n,'primary_numerical_implementation_imported':False,'parameter_updates_original_Policy_CNN_environment_or_sealed_calls':0}
        path=directory/'independent-audit.json';a.base.write_new(path,audit);a.base.write_new(directory/'completion.json',{'status':'complete_independently_audited_fit_endpoint_gradient','formal_result':False,'science_config':science_pin,'binding':binding_pin,'result':result_pin,'audit':d.p.pin(path),'completed_at_utc':a.base.now(),'representation_work_per_stage':work.record(),'total_NPZ_loads':86,'total_member_decodes':610,'parameter_updates_original_Policy_CNN_environment_or_sealed_calls':0});return audit
    except BaseException as error:
        for t in tables.values():
            if not t.handle.closed:t.handle.close()
        a.base.write_new(directory/'audit-failed-partial.json',{'status':'failed_no_retry','error':repr(error),'access':guard.record(),'computation':jobs.counts,'representation_work':work.record(),'checks':checks.n});raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);args=parser.parse_args()
    cfg,_=a.preflight(args.config,args.binding);r=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':r['status'],'checks':r['scalar_checks']}))
