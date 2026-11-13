"""Independent CSV reconstruction; never imports primary numerical implementation."""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np

from experiments import libero_task7_residual_sign_transition_protocol as d
from experiments import libero_task7_residual_sign_transition_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks, CheckedTable

MEASURES = ('target_abs', 'reference_abs', 'candidate_abs', 'reference_raw_residual', 'candidate_raw_residual',
            'raw_shift', 'effective_shift', 'signed_effective_shift', 'clip_or_rounding_gap',
            'signed_reference_margin', 'signed_candidate_margin', 'effective_shift_over_abs_reference', 'raw_shift_over_abs_reference')
METRICS = ('reference_mse', 'candidate_mse', 'delta_mse', 'reference_mae', 'candidate_mae', 'reference_bias', 'candidate_bias',
           'reference_sign_error_rate', 'candidate_sign_error_rate', 'target_energy_mean', 'reference_relative_zero_error',
           'candidate_relative_zero_error', *(name+'_mean' for name in MEASURES))
FOCUS = [('corrected', 'stays_wrong'), ('new_error', 'stays_correct')]
CLASSES = ('wrong_sign', 'prediction_zero', 'same_sign_under', 'same_sign_over', 'exact_match', 'zero_target_nonzero')
KINDS = ('stays_correct', 'corrected', 'new_error', 'stays_wrong')
BANDS = ('zero', 'nonzero_abs_lt.025', 'abs_ge.025')


class StrictTable(CheckedTable):
    def row(self, expected):
        actual = next(self.reader, None)
        d.p.require(actual is not None and set(actual) == set(expected), 'independent CSV row/schema differs')
        for key, value in expected.items():
            text = actual[key]
            if key in ('target_2', 'reference_prediction_2', 'candidate_prediction_2'):
                self.checks.equal(float(text) == value, True, 'exact saved float32/'+key)
            elif value is None: self.checks.equal(text, '', key)
            elif isinstance(value, float): self.checks.equal(float(text), value, key)
            else: self.checks.equal(text, str(value), key)
        self.n += 1
        if self.n % 128 == 0: self.limits.check()


def classify(prediction, target):
    if prediction == target: return 4
    if target == 0: return 5
    if prediction == 0: return 1
    if (prediction > 0) != (target > 0): return 0
    return 2 if abs(prediction) < abs(target) else 3


def reconstruct_frames(guard, cfg):
    pairs = [(demo, t) for demo, length in enumerate(cfg['inputs']['action_lengths']) for t in range(1, length)]
    d.p.require(len(pairs) == 4188 and sum(t>=5 for _,t in pairs) == 4068, 'independent source length differs')
    arrays = {}; present = set(); truth = [None]*4188
    for endpoint in cfg['inputs']['endpoints']:
        arrays[endpoint['seed'], endpoint['arm']] = {name: np.zeros((3,4188), dtype=kind) for name,kind in
                                                    [('p',np.float64),('d',np.float64),('clip',bool),('category',np.int8),('saved_error',np.float64)]}
    for row in guard.project('frames'):
        seed, arm, mode, i = row['seed'],row['arm'],row['mode'],row['source_row']; key=(seed,arm,mode,i)
        d.p.require((seed,arm) in arrays and key not in present, 'independent duplicate/unlisted endpoint/frame')
        demo,t=pairs[i]; tail=t>=cfg['inputs']['action_lengths'][demo]-20
        d.p.require((row['demo'],row['action_step'],row['probe_fold'],row['tail20']) == (demo,t,demo//10,tail), 'independent causal ID differs')
        target, pred = row['target_2'], row['prediction_2']
        d.p.require(-1<=target<=1 and -1<=pred<=1 and float(np.float32(target))==target and float(np.float32(pred))==pred, 'independent saved float32/range differs')
        if truth[i] is None: truth[i]=target
        d.p.require(truth[i]==target and row['target_direction']==(1 if target>=.025 else -1 if target<=-.025 else 0), 'independent shared target/label differs')
        category=classify(pred,target)
        d.p.require(CLASSES[category]==row['component3_category'] and abs(row['error_2']-(pred-target))<=1e-10, 'independent category/error differs')
        m=('original','bias_only','current_affine').index(mode); data=arrays[seed,arm]
        for field,val in [('p',pred),('d',row['raw_residual']),('clip',row['clip']),('category',category),('saved_error',row['error_2'])]: data[field][m,i]=val
        present.add(key)
    expected={(seed,arm,mode,i) for seed,arm in arrays for mode in ('original','bias_only','current_affine') for i in range(4188)}
    d.p.require(present==expected,'independent missing frame/mode')
    for data in arrays.values():
        for m in range(3):
            for i in range(4188):
                baseline=float(data['p'][0,i]); delta=float(data['d'][m,i]); before=baseline+delta
                expected_p=float(np.float32(min(1.,max(-1.,before))))
                d.p.require(expected_p==data['p'][m,i] and bool(data['clip'][m,i])==(before < -1 or before > 1), 'independent residual/clip/float32 reconstruction differs')
                if m==0: d.p.require(delta==0 and not data['clip'][m,i], 'independent original identity differs')
        for fold in range(3):
            values=[float(data['d'][1,i]) for i,(demo,_) in enumerate(pairs) if demo//10==fold]
            d.p.require(all(x==values[0] for x in values) and float(np.float32(values[0]))==values[0], 'independent bias-only fold constant differs')
    y=np.asarray(truth,np.float64); signs=np.asarray([1 if t>=.025 else -1 if t<=-.025 else 0 for t in truth],np.int8)
    tail=np.asarray([t>=cfg['inputs']['action_lengths'][demo]-20 for demo,t in pairs],bool)
    chosen=[i for i,(_,t) in enumerate(pairs) if t>=5 and signs[i]!=0]
    d.p.require(len(chosen)==3598 and sum(signs[i]<0 for i in chosen)==2273 and sum(signs[i]>0 for i in chosen)==1325
                and sum(tail[i] and signs[i]<0 for i in chosen)==402 and sum(tail[i] and signs[i]>0 for i in chosen)==75,'independent material counts differ')
    return {'pairs':pairs,'y':y,'sign':signs,'tail':tail,'selected':chosen}, arrays


def reconstruct_probe(shared,data,mode,ids):
    n=len(ids); residual=data['d'][mode,ids]; pred=data['p'][mode,ids]; target=shared['y'][ids]
    errors=pred-target; m=sum(shared['sign'][i]!=0 for i in ids)
    wrong=sum(shared['sign'][i]!=0 and shared['sign'][i]*float(data['p'][mode,i])<=0 for i in ids)
    clip=int(np.count_nonzero(data['clip'][mode,ids])); sse=math.fsum(float(x)*float(x) for x in errors)
    return {'n':n,'material_n':int(m),'component3_mse':sse/n if n else None,'component3_sse':sse,
            'component3_mae':math.fsum(abs(float(x)) for x in errors)/n if n else None,
            'component3_bias':math.fsum(float(x) for x in errors)/n if n else None,
            'material_sign_error_n':int(wrong),'material_sign_error_rate':int(wrong)/int(m) if m else None,
            'clip_n':clip,'clip_fraction':clip/n if n else None,'raw_residual_mean':float(np.mean(residual)) if n else None,
            'raw_residual_abs_mean':float(np.mean(np.abs(residual))) if n else None,
            'raw_residual_min':float(min(residual)) if n else None,'raw_residual_max':float(max(residual)) if n else None}


def audit_curves(guard,shared,arrays,table,checks):
    missing={(7,seed,arm,fold,mode,update,role,phase,direction)
             for seed,arm,fold,mode,update,role,phase,direction in itertools.product(
                 d.p.SEEDS,d.p.ARMS,range(3),('bias_only','current_affine'),(0,100,250,500),('fit','probe'),
                 ('all_source','all_common','tail20','complement'),('all','negative','positive','neutral'))}
    parity=0
    for row in guard.project('curves'):
        key=tuple(row[k] for k in d.CURVE_ID_COLUMNS)
        d.p.require(key in missing,'independent curve duplicate/ID coverage');missing.remove(key)
        if row['update']!=500 or row['role']!='probe':
            d.p.require(set(row)==set(d.CURVE_ID_COLUMNS),'independent unselected metric leaked');continue
        selected=[]
        for i,(demo,t) in enumerate(shared['pairs']):
            if demo//10!=row['fold']:continue
            if row['phase']!='all_source' and t<5:continue
            if row['phase']=='tail20' and not shared['tail'][i]:continue
            if row['phase']=='complement' and shared['tail'][i]:continue
            if row['direction']!='all' and shared['sign'][i]!={'negative':-1,'positive':1,'neutral':0}[row['direction']]:continue
            selected.append(i)
        m=('original','bias_only','current_affine').index(row['mode']); metrics=reconstruct_probe(shared,arrays[row['seed'],row['arm']],m,selected)
        expected={k:row[k] for k in d.CURVE_ID_COLUMNS}
        for metric in d.CURVE_METRICS:
            value=metrics[metric];checks.equal(row[metric],value,'input_probe/'+metric);parity+=1
            expected['saved_'+metric]=row[metric];expected['computed_'+metric]=value
        table.row({**expected,'parity_passed':True})
    d.p.require(not missing and parity==8064,'independent probe coverage differs');return parity


def build_pairs(shared,data,candidate,reference):
    ci=('original','bias_only','current_affine').index(candidate);ri=('original','bias_only','current_affine').index(reference)
    ref=data['p'][ri];cand=data['p'][ci];dr=data['d'][ri];dc=data['d'][ci];y=shared['y'];sgn=shared['sign']
    correct_r=(sgn!=0)&(np.sign(ref)==sgn);correct_c=(sgn!=0)&(np.sign(cand)==sgn)
    transitions=np.asarray([0 if r and c else 1 if c else 2 if r else 3 for r,c in zip(correct_r,correct_c)],np.int8)
    effective=cand-ref;raw=dc-dr
    ratios=np.full((len(y),2),np.nan)
    for i in range(len(y)):
        if ref[i]!=0:ratios[i]=(abs(effective[i])/abs(ref[i]),abs(raw[i])/abs(ref[i]))
    values=np.column_stack([np.abs(y),np.abs(ref),np.abs(cand),dr,dc,raw,effective,sgn*effective,effective-raw,sgn*ref,sgn*cand,ratios[:,0],ratios[:,1]])
    band=lambda xs:np.asarray([0 if x==0 else 1 if abs(x)<.025 else 2 for x in xs],np.int8)
    refdiff=ref-y;canddiff=cand-y;refsq=np.square(refdiff);candsq=np.square(canddiff)
    return {'ref':ref,'cand':cand,'refclip':data['clip'][ri],'candclip':data['clip'][ci],'rcat':data['category'][ri],'ccat':data['category'][ci],
            'rwrong':~correct_r,'cwrong':~correct_c,'kind':transitions,'rband':band(ref),'cband':band(cand),
            'rdiff':refdiff,'cdiff':canddiff,'rsq':refsq,'csq':candsq,'delta':candsq-refsq,'values':values,'y':y}


def statistics(data,ids):
    n=len(ids);sr=math.fsum(float(data['rsq'][i]) for i in ids);sc=math.fsum(float(data['csq'][i]) for i in ids)
    delta=data['delta'][ids];energy=math.fsum(float(data['y'][i])**2 for i in ids)
    re=sum(bool(data['rwrong'][i]) for i in ids);ce=sum(bool(data['cwrong'][i]) for i in ids)
    row={'n':n,'reference_error_n':re,'candidate_error_n':ce,'net_error_n':ce-re,
         'reference_sign_error_rate':re/n if n else None,'candidate_sign_error_rate':ce/n if n else None,
         'reference_sse':sr,'candidate_sse':sc,'delta_sse':math.fsum(float(x) for x in delta),
         'reference_mse':sr/n if n else None,'candidate_mse':sc/n if n else None,'delta_mse':math.fsum(float(x) for x in delta)/n if n else None,
         'reference_mae':math.fsum(abs(float(data['rdiff'][i])) for i in ids)/n if n else None,
         'candidate_mae':math.fsum(abs(float(data['cdiff'][i])) for i in ids)/n if n else None,
         'reference_bias':math.fsum(float(data['rdiff'][i]) for i in ids)/n if n else None,
         'candidate_bias':math.fsum(float(data['cdiff'][i]) for i in ids)/n if n else None,
         'target_energy':energy,'target_energy_mean':energy/n if n else None,
         'reference_relative_zero_error':sr/energy if energy else None,'candidate_relative_zero_error':sc/energy if energy else None,
         'reference_zero_n':sum(data['ref'][i]==0 for i in ids),'candidate_zero_n':sum(data['cand'][i]==0 for i in ids)}
    # Python sum of NumPy bools is a NumPy integer; emit canonical Python ints.
    row['reference_zero_n']=int(row['reference_zero_n']);row['candidate_zero_n']=int(row['candidate_zero_n'])
    for label,predicate in [('lower',lambda x:x<0),('tie',lambda x:x==0),('higher',lambda x:x>0)]:
        selected=[float(x) for x in delta if predicate(x)];row['delta_sse_'+label+'_n']=len(selected);row['delta_sse_'+label+'_sum']=math.fsum(selected)
    for r,c in itertools.product((-1,0,1),repeat=2):row[f'sign_ref{r}_cand{c}_n']=sum(int(np.sign(data['ref'][i])==r and np.sign(data['cand'][i])==c) for i in ids)
    for label,r,c in [('none',False,False),('reference_only',True,False),('candidate_only',False,True),('both',True,True)]:
        row['clip_'+label+'_n']=sum(int(data['refclip'][i]==r and data['candclip'][i]==c) for i in ids)
    for j,name in enumerate(MEASURES):
        values=data['values'][ids,j];values=values[np.isfinite(values)];size=len(values)
        quantiles=np.quantile(values,[.25,.5,.75],method='linear') if size else [None]*3
        row.update({name+'_n':size,name+'_null_n':n-size,name+'_mean':float(np.mean(values)) if size else None,
                    name+'_median':float(quantiles[1]) if size else None,name+'_q25':float(quantiles[0]) if size else None,
                    name+'_q75':float(quantiles[2]) if size else None,name+'_min':float(min(values)) if size else None,name+'_max':float(max(values)) if size else None})
    d.p.require(sum(row[f'sign_ref{r}_cand{c}_n'] for r,c in itertools.product((-1,0,1),repeat=2))==n
                and sum(row['clip_'+k+'_n'] for k in ('none','reference_only','candidate_only','both'))==n
                and abs(sc-sr-row['delta_sse'])<=1e-10,'independent count/SSE conservation differs')
    return row


def small_cells(data,ids):
    return {'n':len(ids),'reference_sse':math.fsum(float(data['rsq'][i]) for i in ids),
            'candidate_sse':math.fsum(float(data['csq'][i]) for i in ids),'delta_sse':math.fsum(float(data['delta'][i]) for i in ids)}


def difference(focus,control):
    nf,nr=focus['n'],control['n'];paired=bool(nf and nr)
    reason='' if paired else 'both_empty' if nf==nr==0 else 'focus_empty' if nf==0 else 'reference_empty'
    out={'focus_n':nf,'reference_n':nr,'paired':paired,'missing_reason':reason}
    for metric in METRICS:
        f,r=focus[metric],control[metric];out['focus_'+metric]=f;out['reference_'+metric]=r
        out['delta_'+metric]=f-r if paired and f is not None and r is not None else None
    return out


def standardize(rows):
    common=[r for r in rows if r['paired']];n=len(common)
    out={'paired_demo_n':n,'unpaired_demo_n':30-n,'focus_rows_in_paired_demos':sum(r['focus_n'] for r in common),
         'reference_rows_in_paired_demos':sum(r['reference_n'] for r in common),'missing_reason':'' if n else 'no_common_demo'}
    for metric in METRICS:
        values=[r['delta_'+metric] for r in common if r['delta_'+metric] is not None]
        out['delta_'+metric]=math.fsum(values)/len(values) if values else None
        out[metric+'_higher_demo_n']=sum(v>0 for v in values);out[metric+'_tie_demo_n']=sum(v==0 for v in values)
        out[metric+'_lower_demo_n']=sum(v<0 for v in values);out[metric+'_null_demo_n']=30-len(values)
    return out


def selected_ids(shared,phase,direction):
    truth=-1 if direction=='negative' else 1
    return [i for i in shared['selected'] if shared['sign'][i]==truth and (phase=='all_common' or shared['tail'][i]==(phase=='tail20'))]


def audit_model(endpoint,shared,store,tables,cache,summaries):
    for candidate,reference in d.COMPARISONS:
        name=candidate+'-minus-'+reference;data=build_pairs(shared,store,candidate,reference);cache[endpoint['seed'],endpoint['arm'],name]=data
        prefix={k:endpoint[k] for k in ('task','seed','arm')};prefix.update(comparison=name,candidate=candidate,reference=reference)
        for i in shared['selected']:
            demo,t=shared['pairs'][i]
            row={**prefix,'source_row':i,'demo':demo,'action_step':t,'target_sign':int(shared['sign'][i]),'tail20':bool(shared['tail'][i]),
                 'target_2':float(shared['y'][i]),'reference_prediction_2':float(data['ref'][i]),'candidate_prediction_2':float(data['cand'][i]),
                 'reference_sign':int(np.sign(data['ref'][i])),'candidate_sign':int(np.sign(data['cand'][i])),
                 'reference_category':CLASSES[data['rcat'][i]],'candidate_category':CLASSES[data['ccat'][i]],
                 'reference_clip':bool(data['refclip'][i]),'candidate_clip':bool(data['candclip'][i]),'transition':KINDS[data['kind'][i]],
                 'reference_error':bool(data['rwrong'][i]),'candidate_error':bool(data['cwrong'][i]),
                 'reference_band':BANDS[data['rband'][i]],'candidate_band':BANDS[data['cband'][i]],
                 'reference_sse':float(data['rsq'][i]),'candidate_sse':float(data['csq'][i]),'delta_sse':float(data['delta'][i])}
            row.update({key:float(data['values'][i,j]) if math.isfinite(data['values'][i,j]) else None for j,key in enumerate(MEASURES)})
            tables['paired-frames'].row(row)
        for phase,direction in itertools.product(d.PHASES,d.DIRECTIONS):
            ids=selected_ids(shared,phase,direction);base={**prefix,'phase':phase,'direction':direction};groups={};demos={}
            for k,transition in enumerate(KINDS):
                use=[i for i in ids if data['kind'][i]==k];groups[transition]=statistics(data,use)
                tables['model-transitions'].row({**base,'transition':transition,'class_n':len(ids),**groups[transition]})
                for demo in range(30):
                    sub=[i for i in use if shared['pairs'][i][0]==demo];value=statistics(data,sub);demos[transition,demo]=value
                    tables['demo-transitions'].row({**base,'transition':transition,'demo':demo,'class_n':sum(shared['pairs'][i][0]==demo for i in ids),**value})
                for rb,cb in itertools.product(range(3),repeat=2):
                    sub=[i for i in use if data['rband'][i]==rb and data['cband'][i]==cb]
                    tables['margin-cells'].row({**base,'transition':transition,'reference_band':BANDS[rb],'candidate_band':BANDS[cb],**small_cells(data,sub)})
            d.p.require(sum(v['n'] for v in groups.values())==len(ids)
                        and sum(v['net_error_n'] for v in groups.values())==groups['new_error']['n']-groups['corrected']['n']
                        and abs(math.fsum(v['delta_sse'] for v in groups.values())-math.fsum(float(data['delta'][i]) for i in ids))<=1e-10,
                        'independent four-cell conservation differs')
            for rc,cc in itertools.product(range(6),repeat=2):
                sub=[i for i in ids if data['rcat'][i]==rc and data['ccat'][i]==cc]
                tables['error-cells'].row({**base,'reference_category':CLASSES[rc],'candidate_category':CLASSES[cc],**small_cells(data,sub)})
            for focus,control in FOCUS:
                label=focus+'-minus-'+control;ident={**base,'contrast':label,'focus':focus,'control':control};raw=difference(groups[focus],groups[control])
                tables['contrasts'].row({**ident,**raw});pairs=[]
                for demo in range(30):
                    value=difference(demos[focus,demo],demos[control,demo]);pairs.append(value)
                    tables['demo-contrasts'].row({**ident,'demo':demo,**value})
                std=standardize(pairs);tables['demo-standardized'].row({**ident,**std})
                summaries[endpoint['seed'],endpoint['arm'],name,phase,direction,label]={'raw':raw,'demo_standardized':std}


def audit_cross(shared,cache,tables):
    for arm,(candidate,reference) in itertools.product(d.p.ARMS,d.COMPARISONS):
        name=candidate+'-minus-'+reference;data=[cache[seed,arm,name] for seed in d.p.SEEDS]
        base={'task':7,'arm':arm,'comparison':name,'candidate':candidate,'reference':reference}
        for i in shared['selected']:
            demo,t=shared['pairs'][i]
            row={**base,'source_row':i,'demo':demo,'action_step':t,'target_sign':int(shared['sign'][i]),'tail20':bool(shared['tail'][i]),
                 'reference_error_votes':sum(bool(x['rwrong'][i]) for x in data),'candidate_error_votes':sum(bool(x['cwrong'][i]) for x in data)}
            for seed,ctx in zip(d.p.SEEDS,data):
                tag='seed'+str(seed)+'_';delta=float(ctx['delta'][i])
                row[tag+'transition']=KINDS[ctx['kind'][i]];row[tag+'reference_error']=bool(ctx['rwrong'][i]);row[tag+'candidate_error']=bool(ctx['cwrong'][i])
                row[tag+'reference_band']=BANDS[ctx['rband'][i]];row[tag+'candidate_band']=BANDS[ctx['cband'][i]]
                row[tag+'delta_sse_direction']='lower' if delta<0 else 'higher' if delta>0 else 'tie'
            for k,kind in enumerate(KINDS):row[kind+'_votes']=sum(int(x['kind'][i]==k) for x in data)
            d.p.require(sum(row[kind+'_votes'] for kind in KINDS)==3,'independent crossseed vote partition differs')
            tables['crossseed-frames'].row(row)
        for phase,direction,k in itertools.product(d.PHASES,d.DIRECTIONS,range(4)):
            ids=selected_ids(shared,phase,direction);row={**base,'phase':phase,'direction':direction,'transition':KINDS[k],'n':len(ids)}
            sequences={'transition':[sum(int(x['kind'][i]==k) for x in data) for i in ids],
                       'reference_error':[sum(bool(x['rwrong'][i]) for x in data) for i in ids],
                       'candidate_error':[sum(bool(x['cwrong'][i]) for x in data) for i in ids]}
            for name2,values in sequences.items():
                for votes in range(4):row[name2+f'_votes{votes}_n']=values.count(votes)
                row[name2+'_any_seed_n']=sum(v>0 for v in values);row[name2+'_all3_seed_n']=values.count(3)
            tables['crossseed-summary'].row(row)
    for seed,arm in itertools.product(d.p.SEEDS,d.p.ARMS):
        bias=cache[seed,arm,'bias_only-minus-original'];affine=cache[seed,arm,'current_affine-minus-original']
        for phase,direction,kb,ka in itertools.product(d.PHASES,d.DIRECTIONS,range(4),range(4)):
            ids=selected_ids(shared,phase,direction);use=[i for i in ids if bias['kind'][i]==kb and affine['kind'][i]==ka]
            tables['head-overlap'].row({'task':7,'seed':seed,'arm':arm,'phase':phase,'direction':direction,
                                       'bias_transition':KINDS[kb],'affine_transition':KINDS[ka],'class_n':len(ids),'n':len(use),
                                       'bias_delta_sse':math.fsum(float(bias['delta'][i]) for i in use),
                                       'affine_delta_sse':math.fsum(float(affine['delta'][i]) for i in use)})


def reconstructed_summary(summaries):
    output=[]
    for arm,(candidate,reference),phase,direction,(focus,control) in itertools.product(d.p.ARMS,d.COMPARISONS,d.PHASES,d.DIRECTIONS,FOCUS):
        name=candidate+'-minus-'+reference;label=focus+'-minus-'+control
        sources=[summaries[seed,arm,name,phase,direction,label] for seed in d.p.SEEDS]
        item={'task':7,'arm':arm,'comparison':name,'candidate':candidate,'reference':reference,'phase':phase,'direction':direction,'contrast':label,
              'focus':focus,'control':control,'seeds':d.p.SEEDS,'paired_demo_n_values':[r['demo_standardized']['paired_demo_n'] for r in sources]}
        for scope in ('raw','demo_standardized'):
            item[scope]={}
            for measure in METRICS:
                values=[r[scope]['delta_'+measure] for r in sources];nonempty=[x for x in values if x is not None]
                item[scope][measure]={'values':values,'mean':math.fsum(nonempty)/3 if len(nonempty)==3 else None,
                                      'higher_seed_n':sum(x>0 for x in nonempty),'tie_seed_n':sum(x==0 for x in nonempty),
                                      'lower_seed_n':sum(x<0 for x in nonempty),'null_seed_n':3-len(nonempty)}
        output.append(item)
    d.p.require(len(output)==72,'independent seed summary budget differs');return output


def compare_tree(actual,expected,checks,name='seed_summary'):
    if isinstance(expected,dict):
        d.p.require(isinstance(actual,dict) and set(actual)==set(expected),'independent JSON schema differs: '+name)
        for key in expected:compare_tree(actual[key],expected[key],checks,name+'/'+key)
    elif isinstance(expected,list):
        d.p.require(isinstance(actual,list) and len(actual)==len(expected),'independent JSON length differs: '+name)
        for i,(a0,e0) in enumerate(zip(actual,expected)):compare_tree(a0,e0,checks,name+'/'+str(i))
    else:checks.equal(actual,expected,name)


def validate_record(cfg,record,science_pin,binding_pin):
    d.p.require(record['status']=='complete_task7_residual_sign_transition_pending_audit' and record['formal_result'] is False
                and record['science_config']==science_pin and record['binding']==binding_pin
                and record['inputs']==cfg['inputs'] and record['parents']==cfg['parents'] and record['budget']==cfg['budget']
                and record['probe_scalar_parity_checks']==8064 and record['archive_array_checkpoint_reads_including_byte_hashes']==0
                and record['new_training_forward_environment_neighbor_query_sealed_calls']==0
                and record['all_parent_joint_cues_false_unchanged'] is True and record['G_failed_H_unrun_unchanged'] is True
                and record['SPT_Knowledge_Module_changes'] is False,'independent result provenance differs')
    expected=d.budget()['per_stage'];access=record['access']
    d.p.require(access['CSV_data_passes']=={'attempted':2,'completed':2}
                and access['CSV_lexical_rows']=={'frames':75384,'curves':4608}
                and access['CSV_projected_cells']=={'frames':expected['frame_projected_cells'],'curves':expected['curve_ID_projected_cells']+expected['selected_curve_metric_cells']}
                and access['selected_curve_numeric_rows']==576 and not access['denied_operations']
                and access['projected_frame_columns']==d.FRAME_COLUMNS and access['projected_curve_ID_columns']==d.CURVE_ID_COLUMNS
                and access['conditional_curve_metrics']==d.CURVE_METRICS
                and access['byte_SHA_checks']==['frames','curves','frames','curves','frames','curves']
                and access['archive_array_checkpoint_reads_including_hashes']==access['training_forward_environment_sealed_calls']==0,'independent input access counters differ')
    runtime,cost=record['runtime'],record['cost']
    d.p.require(runtime['device']=='CPU' and runtime['environment']==cfg['execution']['environment']
                and {k:runtime[k] for k in ('python','numpy')}==cfg['execution']['runtime']
                and runtime['environment_seed'] is None and runtime['initial_state_indices']==[]
                and cost['elapsed_seconds']<=180 and cost['peak_RSS_bytes']<=536870912,'independent runtime/cost differs')
    d.p.require(set(record['tables'])==set(cfg['budget']['tables']),'independent table set differs')
    for name,ref in record['tables'].items():
        d.p.require(ref['path']==str(Path(cfg['artifacts'])/(name+'.csv')) and ref['rows']==cfg['budget']['tables'][name]
                    and d.p.sha(ref['path'])==ref['sha256'],'independent table path/count/SHA differs')
    d.p.require(sum(Path(ref['path']).stat().st_size for ref in record['tables'].values())<=cfg['execution']['output_CSV_bytes_max'],'independent CSV byte ceiling differs')


def audit_payload(cfg,science_pin,binding_pin):
    directory=Path(cfg['artifacts']);output=Path(cfg['output']);audit_path=directory/'independent-audit.json'
    d.p.require(not any((directory/name).exists() for name in ('independent-audit.json','completion.json','audit-failed-partial.json')),'existing audit/partial; no retry')
    record=json.loads(output.read_text());result_pin=d.p.pin(output);validate_record(cfg,record,science_pin,binding_pin)
    limits=a.base.Limits(cfg['execution']['audit_active_seconds_max'],cfg['execution']['working_memory_bytes_max']);checks=Checks()
    guard=a.Guard(cfg['inputs'],[ref['path'] for ref in record['tables'].values()],limits)
    tables={};cache={};summaries={};started=a.base.now();parity=0
    try:
        with guard:
            guard.hash_inputs();tables={name:StrictTable(ref,checks,limits) for name,ref in record['tables'].items()}
            shared,arrays=reconstruct_frames(guard,cfg);parity=audit_curves(guard,shared,arrays,tables['probe-parity'],checks)
            for endpoint in cfg['inputs']['endpoints']:
                audit_model(endpoint,shared,arrays[endpoint['seed'],endpoint['arm']],tables,cache,summaries)
                print(json.dumps({'stage':'audit','seed':endpoint['seed'],'arm':endpoint['arm'],'scalar_checks':checks.n}),flush=True);limits.check()
            audit_cross(shared,cache,tables);compare_tree(record['seed_summary'],reconstructed_summary(summaries),checks)
            for table in tables.values():table.finish()
            guard.complete()
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output changed during audit')
        d.p.require(d.p.pin(output)==result_pin,'pending result mutated during audit')
        result={'status':'passed_independent_task7_residual_sign_transition_audit','formal_result':False,'result':result_pin,'science_config':science_pin,'binding':binding_pin,
                'started_at_utc':started,'completed_at_utc':a.base.now(),'access':guard.record(),'runtime':a.base.runtime(cfg),'cost':limits.record(),
                'probe_scalar_parity_checks':parity,'scalar_checks':checks.n,'tables_verified':record['tables'],
                'archive_array_checkpoint_reads_including_byte_hashes':0,'new_training_forward_environment_neighbor_query_sealed_calls':0,
                'primary_numerical_implementation_imported': 'experiments.analyze_libero_task7_residual_sign_transition' in sys.modules}
        d.p.require(result['primary_numerical_implementation_imported'] is False,'independent audit imported primary analysis')
        a.base.write_new(audit_path,result)
        a.base.write_new(directory/'completion.json',{'status':'complete_independently_audited_task7_residual_sign_transition','formal_result':False,
            'science_config':science_pin,'binding':binding_pin,'result':result_pin,'audit':d.p.pin(audit_path),'completed_at_utc':a.base.now(),
            'total_input_CSV_data_passes':4,'total_input_CSV_lexical_rows':159984,'total_input_CSV_projected_cells':2662128,
            'total_probe_scalar_parity_checks':16128,'output_rows_verified':115728,'seed_summary_rows_verified':72,
            'archive_array_checkpoint_reads_including_byte_hashes':0,'new_training_forward_environment_neighbor_query_sealed_calls':0,
            'all_parent_joint_cues_false_unchanged':True,'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False})
        return result
    except BaseException as exc:
        for table in tables.values():
            if not table.handle.closed:table.handle.close()
        a.base.write_new(directory/'audit-failed-partial.json',{'status':'failed_no_retry','error':repr(exc),'access':guard.record(),'scalar_checks':checks.n,'probe_scalar_parity_checks':parity})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    args=parser.parse_args();cfg,_=a.preflight(args.config,args.binding);a.base.runtime(cfg)
    result=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':result['status'],'scalar_checks':result['scalar_checks']}))
