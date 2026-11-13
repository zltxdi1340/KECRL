"""Independent stdlib arithmetic reconstruction; never imports the analyzer."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import json
import math
import resource
import sys
import time
from experiments import libero_task7_direction_penalty_tradeoff_protocol as d
from experiments import libero_task7_direction_penalty_tradeoff_access as a


def pstate(s):
    h,n = s['hinge_positive_n'],s['n']
    d.require(type(h) is type(n) is int and n>0 and 0<=h<=n, 'independent active count schema')
    m,p,t = s['component3_mse'],s['penalty'],s['trainable_objective']
    d.require(m>=0 and p>=0 and abs(t-m/7-p)<=1e-10 and (h>0 or p==0), 'independent state terms')
    return {'M':m,'P':p,'T':t,'n':n,'h':h,'active_fraction':h/n,'positive_excess_mean':None if h==0 else p/(h/n)*7}


def delta(x,y):
    dm=y['M']-x['M']; dp=y['P']-x['P']; dt=y['T']-x['T']
    d.require(abs(dt-dm/7-dp)<=1e-10, 'independent conservation')
    return {'dM':dm,'dM_scaled':dm/7,'dP':dp,'dT':dt,'penalty_over_M_saving':-7*dp/dm if dm<0 and dp>0 else None}


def active(x,y):
    d.require(x['n']==y['n'], 'independent fit denominator')
    valid=bool(x['h'] and y['h']); count=severity=extra=None
    if valid:
        u,v=x['h']/x['n'],y['h']/y['n']
        r,s=x['P']/u,y['P']/v
        count=(r+s)*(v-u)/2; severity=(u+v)*(s-r)/2; extra=(s-r)*7
        d.require(abs(count+severity-(y['P']-x['P']))<=1e-10, 'independent active accounting')
    return dict(count_component=count,severity_component=severity,defined=valid,delta_active_n=y['h']-x['h'],delta_positive_excess_mean=extra)


def summarize(rows):
    bins=Counter()
    for r in rows:
        category=lambda v:'up' if v>0 else 'down' if v<0 else 'zero'
        bins[category(r['sP'])+'/'+category(r['dP'])]+=1
    return dict(steps=len(rows),M_up_steps=[r['to_update'] for r in rows if r['dM']>0],
                P_up_steps=len([r for r in rows if r['dP']>0]),P_positive_slopes=len([r for r in rows if r['sP']>0]),P_slope_observed=dict(bins),
                total_dM=math.fsum(r['dM'] for r in rows),total_dP=math.fsum(r['dP'] for r in rows),total_dT=math.fsum(r['dT'] for r in rows),
                total_predicted_P=math.fsum(r['predicted_P'] for r in rows),total_P_remainder=math.fsum(r['remainder_P'] for r in rows),
                branch_changed_steps=len([r for r in rows if r['branch_changed']]))


def reconstruct(backend,states,trials,specs):
    out={k:[] for k in ('own_endpoints','pair_endpoints','steps','matched_full50','full50_paths')}
    groups={}; trial_groups={}
    key=lambda r:tuple(r[f] for f in d.IDS+['optimization_arm'])
    for s in states: groups.setdefault(key(s),[]).append(s)
    for t in trials: trial_groups.setdefault(key(t),[]).append(t)
    d.require(len(groups)==2*len(specs), 'independent complete specified pairs')
    for spec in specs:
        base={f:spec[f] for f in d.IDS}; ident={'backend':backend,**base}; paths={}; heads={}
        for opt in d.ARMS:
            k=tuple(base[f] for f in d.IDS)+(opt,); ss=sorted(groups.pop(k),key=lambda s:s['extra_updates']); tt=trial_groups.pop(k,[])
            expected=spec['prior_control_updates'] if opt==d.ARMS[0] else 50
            d.require([s['extra_updates'] for s in ss]==list(range(expected+1)), 'independent path ordinals')
            ts={t['from_extra_updates']:t for t in tt}
            d.require(len(tt)==len(ts)==expected and set(ts)==set(range(expected)), 'independent accepted coverage')
            pp=[pstate(s) for s in ss]; steps=[]
            for j in range(expected):
                t=ts[j]; left,right=ss[j],ss[j+1]; x,y=pp[j:j+2]; index=t['trial_index']; alpha=t['alpha']
                d.require(type(index) is int and 0<=index<20 and alpha==2.**(-index) and t['accepted'] is True, 'independent grid')
                d.require(y['T']<x['T'] and y['M']<=pp[0]['M'], 'independent acceptance')
                sm,sp,st=left['exact_MSE_slope'],left['exact_penalty_slope'],left['directional_derivative']
                d.require(abs(st-sm/7-sp)<=1e-10,'independent exact slope terms')
                diff=delta(x,y); bm,bp,bt=alpha*sm/7,alpha*sp,alpha*st
                steps.append(dict(**ident,optimization_arm=opt,from_update=j,to_update=j+1,alpha=alpha,trial_index=index,**diff,
                                  sM=sm,sP=sp,sT=st,predicted_M_scaled=bm,predicted_P=bp,predicted_T=bt,
                                  remainder_M_scaled=diff['dM_scaled']-bm,remainder_P=diff['dP']-bp,remainder_T=diff['dT']-bt,
                                  branch_changed=left['branch_digest']!=right['branch_digest']))
            head=dict(**ident,optimization_arm=opt,updates=expected,start=pp[0],end=pp[-1],**delta(pp[0],pp[-1]),active_accounting=active(pp[0],pp[-1]),step_summary=summarize(steps))
            for field in ('dM','dP','dT'):
                d.require(abs(head[field]-math.fsum(r[field] for r in steps))<=1e-10,'independent endpoint telescope')
            paths[opt]=pp; heads[opt]=head; out['own_endpoints'].append(head); out['steps']+=steps
        b,c=[paths[opt] for opt in d.ARMS]
        d.require(b[0]==c[0], 'independent paired500')
        cue=len(c)>1 and c[-1]['T']<c[0]['T'] and len(heads[d.ARMS[1]]['step_summary']['M_up_steps'])==0 and c[-1]['T']<b[-1]['T'] and c[-1]['M']<=b[-1]['M']
        focus=tuple(base[f] for f in ('arm','seed','fold')) in d.FOCUS
        d.require(focus==(not cue), 'independent known failures')
        out['pair_endpoints'].append(dict(**ident,cohort=spec['cohort'],focus=focus,paired_cue=cue,control=b[-1],candidate=c[-1],**delta(b[-1],c[-1]),active_accounting=active(b[-1],c[-1])))
        if spec['cohort']=='full50':
            matched=[dict(**ident,k=j,**delta(x,y),active_accounting=active(x,y)) for j,(x,y) in enumerate(zip(b,c))]
            d.require(len(matched)==51, 'independent full50 coverage'); out['matched_full50']+=matched
            positive=[r['k'] for r in matched if r['dT']>0]; suffix=None
            for j in range(50,-1,-1):
                if matched[j]['dT']<=0:break
                suffix=j
            signs=[r['dT']>0 for r in matched if r['dT']!=0]
            windows=[dict(lo=lo,hi=hi,**{'change_'+f:matched[hi][f]-matched[lo][f] for f in ('dM','dM_scaled','dP','dT')}) for lo,hi in d.WINDOWS]
            out['full50_paths'].append(dict(**ident,focus=focus,first_positive_k=positive[0] if positive else None,positive_T_gap_steps=len(positive),persistent_positive_from=suffix,
                 sign_switches=sum(x!=y for x,y in zip(signs,signs[1:])),anchors=[matched[j] for j in d.ANCHORS],windows=windows,
                 control_steps=heads[d.ARMS[0]]['step_summary'],candidate_steps=heads[d.ARMS[1]]['step_summary']))
    d.require(not groups and not trial_groups, 'independent unconsumed projected rows')
    return out


def compare(actual,expected,where='root'):
    if isinstance(expected,dict):
        d.require(type(actual) is dict and actual.keys()==expected.keys(), 'audit keys differ at '+where)
        return 1+sum(compare(actual[k],expected[k],where+'.'+str(k)) for k in expected)
    if isinstance(expected,list):
        d.require(type(actual) is list and len(actual)==len(expected), 'audit list differs at '+where)
        return 1+sum(compare(x,y,where+f'[{i}]') for i,(x,y) in enumerate(zip(actual,expected)))
    if type(expected) is float:
        d.require(type(actual) is float and math.isfinite(actual) and abs(actual-expected)<=1e-10, 'audit scalar differs at '+where)
    else:
        d.require(type(actual) is type(expected) and actual==expected, 'audit exact value differs at '+where)
    return 1


def run():
    cfg,binding=d.preflight(); runtime=a.runtime(cfg); start=time.monotonic(); started=datetime.now(timezone.utc).isoformat()
    d.require('experiments.analyze_libero_task7_direction_penalty_tradeoff' not in sys.modules,'audit imported primary numerical implementation')
    primary_pin=d.pin(d.OUTPUT); primary=json.loads(d.OUTPUT.read_text())
    d.require(primary['status']=='complete_saved_scalar_accounting_pending_independent_audit' and primary['science']==d.pin(d.CONFIG) and primary['binding']==d.pin(d.BINDING) and primary['inputs']==cfg['inputs'], 'new primary provenance differs')
    guard=a.Guard(cfg); checks=0; counts={k:0 for k in cfg['budget']['output_rows']}
    try:
        with guard:
            for backend in ('P','U'):
                states,trials=guard.project(backend); expected=reconstruct(backend,states,trials,cfg['identities'])
                for name,rows in expected.items():
                    actual=[r for r in primary['tables'][name] if r['backend']==backend]
                    checks+=compare(actual,rows,backend+'.'+name); counts[name]+=len(rows)
                del states,trials,expected
            guard.complete()
        d.require(counts==cfg['budget']['output_rows'] and {k:len(v) for k,v in primary['tables'].items()}==counts,'independent output budget differs')
        d.require(d.pin(d.OUTPUT)==primary_pin, 'new primary changed during audit')
        audit=dict(status='passed_independent_saved_scalar_penalty_accounting_audit',formal_result=False,
                   science=d.pin(d.CONFIG),binding=d.pin(d.BINDING),analysis=primary_pin,runtime=runtime,
                   started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),checks=checks,rows=counts,access=guard.record(),
                   primary_numerical_implementation_imported=False,
                   cost=dict(seconds=time.monotonic()-start,RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024))
        d.write(d.AUDIT,audit); d.require(d.AUDIT.stat().st_size<=cfg['execution']['output_JSON_bytes_max'],'audit byte budget')
        d.write(d.ARTIFACTS/'completion.json',dict(status='complete_independently_audited_readonly_penalty_accounting',science=audit['science'],binding=audit['binding'],analysis=primary_pin,audit=d.pin(d.AUDIT),checks=checks))
        return {'status':audit['status'],'checks':checks}
    except BaseException as error:
        d.write(d.ARTIFACTS/'audit-failed-partial.json',dict(error=repr(error),access=guard.record(),checks=checks,rows=counts))
        raise


if __name__ == '__main__':
    print(json.dumps(run()))
