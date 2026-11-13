"""Account for endpoint M/7 and penalty gaps without any model or trajectory read."""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time
import os
from experiments import libero_task7_stepwise_m_reference_tradeoff_protocol as d
from experiments import libero_task7_stepwise_m_reference_tradeoff_access as a


def point(m,p,t):
    d.p.require(all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in (m,p,t))
                and abs(t-m/7-p)<=1e-10,'invalid endpoint terms')
    return dict(M=m,P=p,T=t)


def delta(left,right):
    dm,dp,dt=(right[f]-left[f] for f in ('M','P','T'))
    d.p.require(abs(dt-dm/7-dp)<=1e-10,'endpoint gap does not conserve')
    if dm<=0 and dt<0:category='joint_T_lower_M_nonhigher'
    elif dm<0:category='M_lower_T_notlower'
    elif dm>0 and dt<0:category='M_higher_T_lower'
    elif dm>0:category='M_higher_T_notlower'
    else:category='M_equal_T_notlower'
    return dict(dM=dm,dM_scaled=dm/7,dP=dp,dT=dt,category=category,joint=dt<0 and dm<=0,
      penalty_covers_M_cost=(-dp)/(dm/7) if dm>0 and dp<0 else None,
      penalty_consumes_M_saving=dp/(-dm/7) if dm<0 and dp>0 else None,
      M_sign='up' if dm>0 else 'down' if dm<0 else 'zero',P_sign='up' if dp>0 else 'down' if dp<0 else 'zero')


def analyze(engine,projected,cfg):
    lookup={table:{d.key(r)+(r['optimization_arm'],) if table=='heads' else d.key(r):r for r in rows} for table,rows in projected.items()}
    rows=[];focus=[]
    for spec in cfg['identities']:
        ident={f:spec[f] for f in d.IDS};key=d.key(spec);pair=lookup['pairs'][key]
        control,candidate=[lookup['heads'][key+(opt,)] for opt in d.ARMS]
        starts=[point(h['initial_component3_mse'],h['initial_penalty'],h['initial_trainable_objective']) for h in (control,candidate)]
        ends=[point(h['final_component3_mse'],h['final_penalty'],h['final_trainable_objective']) for h in (control,candidate)]
        d.p.require(starts[0]==starts[1],'paired auxiliary500 start differs')
        for h,end in zip((control,candidate),ends):
            d.p.require(type(h['extra_updates']) is int and h['extra_updates']==50 and h['terminal_reason']=='update_cap'
              and type(h['accepted_MSE_step_increases']) is int and 0<=h['accepted_MSE_step_increases']<=50
              and type(h['joint_cue']) is bool and end['M']<=starts[0]['M'] and end['T']<starts[0]['T'],'saved endpoint invariants differ')
        d.p.require(candidate['accepted_MSE_step_increases']==0 and pair['candidate_progress_from500'] is True and pair['step_M_invariant_met'] is True,
                    'candidate step-M invariant/progress provenance differs')
        references={'own500':starts[0],'control':ends[0]}
        for name in ('zero','bisector'):
            r=lookup[name][key];references[name]=point(r['component3_mse'],r['penalty'],r['trainable_objective'])
        cues={}
        is_focus=(spec['arm'],spec['seed'],spec['fold']) in d.FOCUS
        for name in d.REFERENCES:
            change=delta(references[name],ends[1]);cues[name]=change['joint']
            row=dict(backend=engine,**ident,cohort=pair['cohort'],focus_current=is_focus,reference=name,
              candidate=ends[1],fixed_reference=references[name],**change)
            if name!='own500':
                for value,saved in [('dM','MSE'),('dT','T')]:
                    d.p.require(abs(change[value]-pair[f'candidate_minus_{name}_{saved}'])<=1e-10,'saved pair gap differs')
                flag='margin_control_pair_cue' if name=='control' else name+'_anchor_cue'
                d.p.require(type(pair[flag]) is bool and pair[flag]==change['joint'],'saved raw reference cue differs')
            rows.append(row)
            if name=='control':
                d.p.require(is_focus is (not change['joint']),'frozen eight counterexample identities differ')
                if is_focus:focus.append(dict(row))
        d.p.require(type(pair['paired_joint_cue']) is bool and pair['paired_joint_cue']==all(cues.values()),'saved all-reference cue differs')
    d.p.require(len(rows)==72 and len(focus)==8 and sum(r['accepted_MSE_step_increases'] for r in projected['heads'] if r['optimization_arm']==d.ARMS[0])==407,
                'all18/focus/control accounting differs')
    return rows,focus


def aggregate(rows,fields):
    groups={}
    for row in rows:groups.setdefault(tuple(row[f] for f in fields),[]).append(row)
    out=[]
    for key,items in groups.items():
        summary=dict(zip(fields,key));summary.update(n=len(items),joint=sum(r['joint'] for r in items),
          M_lower=sum(r['dM']<0 for r in items),M_equal=sum(r['dM']==0 for r in items),M_higher=sum(r['dM']>0 for r in items),
          T_lower=sum(r['dT']<0 for r in items),categories=dict(Counter(r['category'] for r in items)))
        for f in ('dM','dM_scaled','dP','dT'):
            summary['mean_'+f]=math.fsum(r[f] for r in items)/len(items);summary['median_'+f]=statistics.median(r[f] for r in items)
        d.p.require(abs(summary['mean_dT']-summary['mean_dM_scaled']-summary['mean_dP'])<=1e-10,'group terms do not conserve')
        out.append(summary)
    return out


def finish_tables(comparisons,focus,cfg):
    lookup={(r['backend'],d.key(r),r['reference']):r for r in comparisons};cross=[]
    for spec in cfg['identities']:
        k=d.key(spec);P,U=lookup['P',k,'control'],lookup['U',k,'control']
        equal={ref:lookup['P',k,ref]['category']==lookup['U',k,ref]['category'] for ref in d.REFERENCES}
        cross.append(dict(**{f:spec[f] for f in d.IDS},P_control_category=P['category'],U_control_category=U['category'],
          reference_categories_equal=equal,both_all_reference_joint=all(lookup[engine,k,ref]['joint'] for engine in ('P','U') for ref in d.REFERENCES)))
    return dict(comparisons=comparisons,focus_current=focus,source_seed=aggregate(comparisons,['backend','arm','seed','reference']),
                source=aggregate(comparisons,['backend','arm','reference']),cross_engine=cross)


def execute():
    cfg,bound=d.preflight();runtime=a.runtime(cfg);start=time.monotonic();started=datetime.now(timezone.utc).isoformat()
    d.ARTIFACTS.mkdir(exist_ok=False);guard=a.Guard(cfg);comparisons=[];focus=[]
    try:
        with guard:
            for engine in ('P','U'):
                values,selected=analyze(engine,guard.project(engine),cfg);comparisons.extend(values);focus.extend(selected)
            tables=finish_tables(comparisons,focus,cfg);guard.complete()
        d.p.require({k:len(v) for k,v in tables.items()}==cfg['budget']['output_rows'],'output row budget differs')
        result=dict(status='complete_stepwise_reference_accounting_pending_independent_audit',formal_result=False,science=d.p.pin(d.CONFIG),binding=d.p.pin(d.BINDING),
          inputs=cfg['inputs'],runtime=runtime,started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
          cost=dict(seconds=time.monotonic()-start,RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),access=guard.record(),tables=tables)
        d.p.write(d.OUTPUT,result);d.p.require(d.OUTPUT.stat().st_size<=cfg['execution']['output_JSON_bytes_max'],'analysis JSON too large')
        return result
    except BaseException as error:
        d.p.write(d.ARTIFACTS/'primary-failed-partial.json',dict(error=repr(error),access=guard.record(),completed_comparisons=len(comparisons)));raise


def run():
    cfg,bound=d.preflight();start=time.monotonic();launch=d.ARTIFACTS.with_name(d.ARTIFACTS.name+'-launch.json');process=d.ARTIFACTS.with_name(d.ARTIFACTS.name+'-process.json')
    d.p.require(not any(path.exists() for path in (launch,process,d.ARTIFACTS)),'existing run; no retry or overwrite')
    result=dict(started_at_utc=datetime.now(timezone.utc).isoformat(),science=d.p.pin(d.CONFIG),binding=d.p.pin(d.BINDING),processes=[])
    d.p.write(launch,result);child=None
    try:
        for stage,module in [('primary','experiments.analyze_libero_task7_stepwise_m_reference_tradeoff'),('audit','experiments.audit_libero_task7_stepwise_m_reference_tradeoff')]:
            cmd=[sys.executable,'-m',module]+(['--mode','analyze'] if stage=='primary' else [])
            env=os.environ.copy();env.update(bound['CPU_environment']);entry=dict(stage=stage,command=cmd,started_at_utc=datetime.now(timezone.utc).isoformat());result['processes'].append(entry)
            with d.ARTIFACTS.with_name(d.ARTIFACTS.name+'-'+stage+'.log').open('x') as handle:
                child=subprocess.Popen(cmd,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);entry['pid']=child.pid
                entry['exit_code']=child.wait(timeout=min(cfg['execution']['seconds_per_stage']+10,cfg['execution']['overall_seconds']-(time.monotonic()-start)))
            entry['completed_at_utc']=datetime.now(timezone.utc).isoformat();d.p.require(entry['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-start<=cfg['execution']['overall_seconds'],'overall time exceeded')
        result['status']='complete_independently_audited_stepwise_reference_accounting'
    except BaseException as error:
        if child is not None and child.poll() is None:child.kill();child.wait()
        result.update(status='failed_no_retry',error=repr(error));raise
    finally:
        result.update(wall_seconds=time.monotonic()-start,completed_at_utc=datetime.now(timezone.utc).isoformat());d.p.write(process,result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',choices=['preflight','analyze','run'],default='preflight');args=parser.parse_args()
    if args.mode=='analyze':result=execute()
    elif args.mode=='run':result=run()
    else:d.preflight();result={'status':'bound_no_numeric_payload'}
    print(json.dumps({'status':result['status']}))
