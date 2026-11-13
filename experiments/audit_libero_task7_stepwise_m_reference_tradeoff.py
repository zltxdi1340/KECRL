"""Independent scalar reconstruction; never import the new analyzer."""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
import json
import math
import resource
import statistics
import sys
import time
from experiments import libero_task7_stepwise_m_reference_tradeoff_protocol as d
from experiments import libero_task7_stepwise_m_reference_tradeoff_access as a
from experiments.audit_libero_task7_direction_penalty_tradeoff import compare


def terms(m,penalty,total):
    d.p.require(all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in (m,penalty,total))
                and abs(total-(m/7+penalty))<=1e-10,'independent endpoint terms invalid')
    return {'M':m,'P':penalty,'T':total}


def difference(reference,candidate):
    mse=candidate['M']-reference['M'];penalty=candidate['P']-reference['P'];total=candidate['T']-reference['T']
    d.p.require(abs(total-(mse/7+penalty))<=1e-10,'independent gap conservation')
    joint=bool(total<0 and mse<=0)
    if joint:label='joint_T_lower_M_nonhigher'
    elif mse<0:label='M_lower_T_notlower'
    elif mse==0:label='M_equal_T_notlower'
    elif total<0:label='M_higher_T_lower'
    else:label='M_higher_T_notlower'
    return dict(dM=mse,dM_scaled=mse/7,dP=penalty,dT=total,category=label,joint=joint,
      penalty_covers_M_cost=(-penalty)/(mse/7) if mse>0 and penalty<0 else None,
      penalty_consumes_M_saving=penalty/(-mse/7) if mse<0 and penalty>0 else None,
      M_sign='up' if mse>0 else 'zero' if mse==0 else 'down',P_sign='up' if penalty>0 else 'zero' if penalty==0 else 'down')


def reconstruct(engine,projected,cfg):
    groups={table:{} for table in projected}
    for table,values in projected.items():
        for value in values:
            k=d.key(value)+(value['optimization_arm'],) if table=='heads' else d.key(value)
            d.p.require(k not in groups[table],'independent duplicate endpoint');groups[table][k]=value
    output=[];selected=[]
    for spec in cfg['identities']:
        k=d.key(spec);ident={f:spec[f] for f in d.IDS};pair=groups['pairs'].pop(k)
        hs=[groups['heads'].pop(k+(opt,)) for opt in d.ARMS]
        initial=[terms(h['initial_component3_mse'],h['initial_penalty'],h['initial_trainable_objective']) for h in hs]
        end=[terms(h['final_component3_mse'],h['final_penalty'],h['final_trainable_objective']) for h in hs]
        d.p.require(initial[0]==initial[1],'independent starts differ')
        for h,value in zip(hs,end):
            d.p.require(type(h['extra_updates']) is int and h['extra_updates']==50 and h['terminal_reason']=='update_cap'
              and type(h['accepted_MSE_step_increases']) is int and h['accepted_MSE_step_increases'] in range(51)
              and type(h['joint_cue']) is bool and value['M']<=initial[0]['M'] and value['T']<initial[0]['T'],'independent endpoint invariants differ')
        d.p.require(hs[1]['accepted_MSE_step_increases']==0 and pair['candidate_progress_from500'] is True and pair['step_M_invariant_met'] is True,'independent step-M provenance differs')
        refs={'own500':initial[0],'control':end[0]}
        for name in ('zero','bisector'):
            r=groups[name].pop(k);refs[name]=terms(r['component3_mse'],r['penalty'],r['trainable_objective'])
        outcomes=[];focus=(spec['arm'],spec['seed'],spec['fold']) in d.FOCUS
        for name in d.REFERENCES:
            gap=difference(refs[name],end[1]);outcomes.append(gap['joint'])
            row={'backend':engine,**ident,'cohort':pair['cohort'],'focus_current':focus,'reference':name,
                 'candidate':end[1],'fixed_reference':refs[name],**gap}
            if name!='own500':
                d.p.require(abs(gap['dM']-pair['candidate_minus_'+name+'_MSE'])<=1e-10
                  and abs(gap['dT']-pair['candidate_minus_'+name+'_T'])<=1e-10,'independent saved gap differs')
                field='margin_control_pair_cue' if name=='control' else name+'_anchor_cue'
                d.p.require(type(pair[field]) is bool and pair[field]==gap['joint'],'independent saved cue differs')
            if name=='control':
                d.p.require(focus==(not gap['joint']),'independent fixed focus mismatch')
                if focus:selected.append(dict(row))
            output.append(row)
        d.p.require(type(pair['paired_joint_cue']) is bool and pair['paired_joint_cue']==all(outcomes),'independent main cue differs')
    d.p.require(not any(groups.values()) and len(output)==72 and len(selected)==8
       and sum(h['accepted_MSE_step_increases'] for h in projected['heads'] if h['optimization_arm']==d.ARMS[0])==407,'independent input coverage/control differs')
    return output,selected


def regroup(values,fields):
    order=[];groups={}
    for r in values:
        k=tuple(r[f] for f in fields)
        if k not in groups:order.append(k);groups[k]=[]
        groups[k].append(r)
    output=[]
    for k in order:
        rows=groups[k];result=dict(zip(fields,k))
        result.update(n=len(rows),joint=len([r for r in rows if r['joint']]),M_lower=len([r for r in rows if r['dM']<0]),
          M_equal=len([r for r in rows if r['dM']==0]),M_higher=len([r for r in rows if r['dM']>0]),
          T_lower=len([r for r in rows if r['dT']<0]),categories=dict(Counter(r['category'] for r in rows)))
        for f in ('dM','dM_scaled','dP','dT'):
            result['mean_'+f]=float(statistics.mean(r[f] for r in rows))
            items=sorted(r[f] for r in rows);n=len(items);result['median_'+f]=items[n//2] if n%2 else (items[n//2-1]+items[n//2])/2
        d.p.require(abs(result['mean_dT']-result['mean_dM_scaled']-result['mean_dP'])<=1e-10,'independent group conservation')
        output.append(result)
    return output


def reconstruct_tables(rows,focus,cfg):
    by_identity={}
    for r in rows:by_identity.setdefault(d.key(r),{}).setdefault(r['backend'],{})[r['reference']]=r
    paired=[]
    for spec in cfg['identities']:
        own=by_identity[d.key(spec)];P,U=own['P'],own['U']
        paired.append(dict(**{f:spec[f] for f in d.IDS},P_control_category=P['control']['category'],U_control_category=U['control']['category'],
          reference_categories_equal={ref:P[ref]['category']==U[ref]['category'] for ref in d.REFERENCES},
          both_all_reference_joint=all(P[ref]['joint'] and U[ref]['joint'] for ref in d.REFERENCES)))
    return dict(comparisons=rows,focus_current=focus,source_seed=regroup(rows,['backend','arm','seed','reference']),
                source=regroup(rows,['backend','arm','reference']),cross_engine=paired)


def run():
    cfg,bound=d.preflight();runtime=a.runtime(cfg);started=datetime.now(timezone.utc).isoformat();start=time.monotonic()
    d.p.require('experiments.analyze_libero_task7_stepwise_m_reference_tradeoff' not in sys.modules,'independent audit imported analyzer')
    pin=d.p.pin(d.OUTPUT);primary=json.loads(d.OUTPUT.read_text())
    d.p.require(primary['status']=='complete_stepwise_reference_accounting_pending_independent_audit' and primary['formal_result'] is False
      and primary['science']==d.p.pin(d.CONFIG) and primary['binding']==d.p.pin(d.BINDING) and primary['inputs']==cfg['inputs'],'new analysis provenance differs')
    guard=a.Guard(cfg);rows=[];focus=[];checks=0
    try:
        with guard:
            for engine in ('P','U'):
                values,selected=reconstruct(engine,guard.project(engine),cfg);rows.extend(values);focus.extend(selected)
            expected=reconstruct_tables(rows,focus,cfg)
            d.p.require({k:len(v) for k,v in expected.items()}==cfg['budget']['output_rows'],'independent output row budget differs')
            checks=compare(primary['tables'],expected);guard.complete()
        d.p.require(d.p.pin(d.OUTPUT)==pin,'new analysis changed during audit')
        result=dict(status='passed_independent_stepwise_reference_accounting_audit',formal_result=False,science=d.p.pin(d.CONFIG),binding=d.p.pin(d.BINDING),
          analysis=pin,inputs=cfg['inputs'],runtime=runtime,started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
          cost=dict(seconds=time.monotonic()-start,RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),access=guard.record(),checks=checks,
          primary_numerical_implementation_imported=False,tables=expected)
        d.p.write(d.AUDIT,result);d.p.require(d.AUDIT.stat().st_size<=cfg['execution']['output_JSON_bytes_max'],'audit JSON too large')
        d.p.write(d.ARTIFACTS/'completion.json',dict(status='complete_independently_audited_stepwise_reference_accounting',science=result['science'],binding=result['binding'],
          analysis=pin,audit=d.p.pin(d.AUDIT),checks=checks,source_JSON_passes=4,source_SHA_checks=8))
        return result
    except BaseException as error:
        d.p.write(d.ARTIFACTS/'audit-failed-partial.json',dict(error=repr(error),access=guard.record(),checks=checks));raise


if __name__=='__main__':print(json.dumps({'status':run()['status']}))
