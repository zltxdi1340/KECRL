"""Read saved v2 scalars once; separate reference, active count and step effects."""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import resource
import time
from experiments import libero_task7_direction_penalty_tradeoff_protocol as d
from experiments import libero_task7_direction_penalty_tradeoff_access as a


def close(x,y):
    d.require(abs(x-y) <= 1e-10, 'penalty accounting conservation differs')


def point(row):
    n,h = row['n'],row['hinge_positive_n']; m,p,t = (row[k] for k in ('component3_mse','penalty','trainable_objective'))
    d.require(type(n) is type(h) is int and 0 <= h <= n and n > 0 and m >= 0 and p >= 0, 'invalid saved state')
    close(t,m/7+p)
    d.require(h > 0 or p == 0, 'zero active count with nonzero penalty')
    return dict(M=m,P=p,T=t,n=n,h=h,active_fraction=h/n,positive_excess_mean=7*p*n/h if h else None)


def change(left,right):
    dm,dp,dt = (right[k]-left[k] for k in ('M','P','T'))
    close(dt,dm/7+dp)
    return dict(dM=dm,dM_scaled=dm/7,dP=dp,dT=dt,
                penalty_over_M_saving=dp/(-dm/7) if dm < 0 and dp > 0 else None)


def factors(left,right):
    d.require(left['n'] == right['n'], 'active fractions from different fit sets')
    if left['h'] == 0 or right['h'] == 0:
        return dict(count_component=None,severity_component=None,defined=False,delta_active_n=right['h']-left['h'],delta_positive_excess_mean=None)
    count = (left['positive_excess_mean']+right['positive_excess_mean'])/2*(right['active_fraction']-left['active_fraction'])/7
    severity = (left['active_fraction']+right['active_fraction'])/2*(right['positive_excess_mean']-left['positive_excess_mean'])/7
    close(count+severity,right['P']-left['P'])
    return dict(count_component=count,severity_component=severity,defined=True,delta_active_n=right['h']-left['h'],delta_positive_excess_mean=right['positive_excess_mean']-left['positive_excess_mean'])


def step_summary(rows):
    def sign(v): return 'up' if v > 0 else 'down' if v < 0 else 'zero'
    return dict(steps=len(rows),M_up_steps=[r['to_update'] for r in rows if r['dM'] > 0],
                P_up_steps=sum(r['dP'] > 0 for r in rows),P_positive_slopes=sum(r['sP'] > 0 for r in rows),
                P_slope_observed=dict(Counter(sign(r['sP'])+'/'+sign(r['dP']) for r in rows)),
                total_dM=math.fsum(r['dM'] for r in rows),total_dP=math.fsum(r['dP'] for r in rows),total_dT=math.fsum(r['dT'] for r in rows),
                total_predicted_P=math.fsum(r['predicted_P'] for r in rows),total_P_remainder=math.fsum(r['remainder_P'] for r in rows),
                branch_changed_steps=sum(r['branch_changed'] for r in rows))


def analyze_backend(backend, states, trials, identities):
    result = {k:[] for k in ('own_endpoints','pair_endpoints','steps','matched_full50','full50_paths')}
    for spec in identities:
        ident = {k:spec[k] for k in d.IDS}; output_id = dict(backend=backend,**ident)
        chosen = lambda row: all(row[k] == v for k,v in ident.items())
        own = {}; paths = {}; pathsteps = {}
        for opt in d.ARMS:
            rows = sorted((r for r in states if chosen(r) and r['optimization_arm'] == opt),key=lambda r:r['extra_updates'])
            d.require([r['extra_updates'] for r in rows] == list(range(len(rows))) and 1 <= len(rows) <= 51, 'path coverage differs')
            wanted = spec['prior_control_updates'] if opt == d.ARMS[0] else 50
            d.require(len(rows) == wanted+1, 'fixed cohort path length differs')
            points = [point(r) for r in rows]; accepted = [r for r in trials if chosen(r) and r['optimization_arm'] == opt]
            lookup = {r['from_extra_updates']:r for r in accepted}
            d.require(len(accepted) == len(lookup) == len(rows)-1 and set(lookup) == set(range(len(rows)-1)), 'accepted trial path differs')
            steps = []
            for k,(left,right) in enumerate(zip(points,points[1:])):
                before,after = rows[k:k+2]; trial = lookup[k]; alpha = trial['alpha']; slope_m,slope_p,slope_t = (before[f] for f in ('exact_MSE_slope','exact_penalty_slope','directional_derivative'))
                d.require(trial['accepted'] is True and type(trial['trial_index']) is int and 0 <= trial['trial_index'] < 20 and alpha == 2.**(-trial['trial_index']), 'accepted alpha grid differs')
                d.require(right['T'] < left['T'] and right['M'] <= points[0]['M'], 'saved acceptance invariants differ')
                close(slope_t,slope_m/7+slope_p); delta = change(left,right)
                predicted_m,predicted_p,predicted_t = alpha*slope_m/7,alpha*slope_p,alpha*slope_t
                step = dict(**output_id,optimization_arm=opt,from_update=k,to_update=k+1,alpha=alpha,trial_index=trial['trial_index'],**delta,
                            sM=slope_m,sP=slope_p,sT=slope_t,predicted_M_scaled=predicted_m,predicted_P=predicted_p,predicted_T=predicted_t,
                            remainder_M_scaled=delta['dM_scaled']-predicted_m,remainder_P=delta['dP']-predicted_p,remainder_T=delta['dT']-predicted_t,
                            branch_changed=before['branch_digest'] != after['branch_digest'])
                close(step['remainder_T'],step['remainder_M_scaled']+step['remainder_P']); steps.append(step)
            paths[opt] = points; pathsteps[opt] = steps
            own[opt] = dict(**output_id,optimization_arm=opt,updates=len(rows)-1,start=points[0],end=points[-1],
                            **change(points[0],points[-1]),active_accounting=factors(points[0],points[-1]),step_summary=step_summary(steps))
            for field in ('dM','dP','dT'):
                close(own[opt][field],math.fsum(r[field] for r in steps))
            result['own_endpoints'].append(own[opt]); result['steps'].extend(steps)
        control,candidate = (paths[o] for o in d.ARMS)
        d.require(control[0] == candidate[0], 'paired own500 must match')
        cm,mm = control[-1],candidate[-1]
        cue = len(candidate)>1 and mm['T']<candidate[0]['T'] and not own[d.ARMS[1]]['step_summary']['M_up_steps'] and mm['T']<cm['T'] and mm['M']<=cm['M']
        focus = (ident['arm'],ident['seed'],ident['fold']) in d.FOCUS
        paired = dict(**output_id,cohort=spec['cohort'],focus=focus,paired_cue=cue,control=cm,candidate=mm,
                      **change(cm,mm),active_accounting=factors(cm,mm))
        d.require(focus == (not cue), 'known v2 counterexample identities changed')
        result['pair_endpoints'].append(paired)
        if spec['cohort'] != 'full50':
            continue
        d.require(len(control) == len(candidate) == 51, 'full50 matching differs')
        matched = [dict(**output_id,k=k,**change(c,m),active_accounting=factors(c,m)) for k,(c,m) in enumerate(zip(control,candidate))]
        result['matched_full50'].extend(matched)
        positives = [r['k'] for r in matched if r['dT'] > 0]
        tail = None
        if matched[-1]['dT'] > 0:
            tail = 50
            while tail>0 and matched[tail-1]['dT']>0:
                tail -= 1
        nonzero_signs = [1 if r['dT'] > 0 else -1 for r in matched if r['dT'] != 0]
        windows = []
        for lo,hi in d.WINDOWS:
            windows.append(dict(lo=lo,hi=hi,**{'change_'+f:matched[hi][f]-matched[lo][f] for f in ('dM','dM_scaled','dP','dT')}))
        result['full50_paths'].append(dict(**output_id,focus=focus,first_positive_k=min(positives) if positives else None,
               positive_T_gap_steps=len(positives),persistent_positive_from=tail,sign_switches=sum(x!=y for x,y in zip(nonzero_signs,nonzero_signs[1:])),
               anchors=[r for r in matched if r['k'] in d.ANCHORS],windows=windows,
               control_steps=own[d.ARMS[0]]['step_summary'],candidate_steps=own[d.ARMS[1]]['step_summary']))
    d.require(len(states) == sum(r['updates']+1 for r in result['own_endpoints']) and len(trials) == len(result['steps']), 'unconsumed scalar input rows')
    return result


def run():
    cfg,binding = d.preflight(); runtime = a.runtime(cfg); started = datetime.now(timezone.utc).isoformat(); start = time.monotonic()
    d.ARTIFACTS.mkdir(exist_ok=False); result = {k:[] for k in cfg['budget']['output_rows']}; guard = a.Guard(cfg)
    try:
        with guard:
            for backend in ('P','U'):
                states,trials = guard.project(backend)
                rows = analyze_backend(backend,states,trials,cfg['identities'])
                for key in result:
                    result[key].extend(rows[key])
                del states,trials
            guard.complete()
        d.require({k:len(v) for k,v in result.items()} == cfg['budget']['output_rows'], 'analysis output budget differs')
        out = dict(status='complete_saved_scalar_accounting_pending_independent_audit',formal_result=False,
                   science=d.pin(d.CONFIG),binding=d.pin(d.BINDING),inputs=cfg['inputs'],runtime=runtime,
                   started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                   cost=dict(seconds=time.monotonic()-start,RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024),access=guard.record(),tables=result)
        d.write(d.OUTPUT,out); d.require(d.OUTPUT.stat().st_size <= cfg['execution']['output_JSON_bytes_max'], 'output byte budget exceeded')
        return {'status':out['status'],'output':str(d.OUTPUT),'rows':{k:len(v) for k,v in result.items()}}
    except BaseException as error:
        d.write(d.ARTIFACTS/'primary-failed-partial.json',dict(error=repr(error),access=guard.record(),completed_rows={k:len(v) for k,v in result.items()}))
        raise


if __name__ == '__main__':
    print(json.dumps(run()))
