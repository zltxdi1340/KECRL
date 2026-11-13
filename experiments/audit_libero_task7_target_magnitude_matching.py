"""Independent target-scale matching reconstruction; never imports primary numerics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import struct
import sys

from experiments import libero_task7_target_magnitude_matching_protocol as d
from experiments import libero_task7_target_magnitude_matching_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks, CheckedTable


class StrictTable(CheckedTable):
    def row(self,expected):
        actual=next(self.reader,None);d.p.require(actual is not None and set(actual)==set(expected),'independent row/schema differs')
        for key,value in expected.items():
            text=actual[key]
            if key in ('target_2','reference_prediction_2','candidate_prediction_2','exact_target_value') and value is not None:
                self.checks.equal(float(text)==value,True,'exact float32/'+key)
            elif value is None:self.checks.equal(text,'',key)
            elif isinstance(value,float):self.checks.equal(float(text),value,key)
            else:self.checks.equal(text,str(value),key)
        self.n+=1
        if self.n%128==0:self.limits.check()


def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]


def bits(value):return struct.unpack('<I',struct.pack('<f',abs(value)))[0]


def bin_id(value):
    magnitude=abs(value)
    for i in range(6,-1,-1):
        if magnitude>=d.BIN_EDGES[i]:return i
    raise ValueError('nonmaterial target')


def category(pred,target):
    if pred==target:return 'exact_match'
    if target==0:return 'zero_target_nonzero'
    if pred==0:return 'prediction_zero'
    if (pred>0)!=(target>0):return 'wrong_sign'
    return 'same_sign_under' if abs(pred)<abs(target) else 'same_sign_over'


def band(pred):return 'zero' if pred==0 else 'nonzero_abs_lt.025' if abs(pred)<.025 else 'abs_ge.025'


def transition(target,reference,candidate):
    r=reference>0 if target>0 else reference<0;c=candidate>0 if target>0 else candidate<0
    return 'stays_correct' if r and c else 'new_error' if r else 'corrected' if c else 'stays_wrong'


def reconstruct_source(guard,cfg):
    pairs=[(demo,t) for demo,length in enumerate(cfg['inputs']['action_lengths']) for t in range(1,length)]
    d.p.require(len(pairs)==4188,'independent source lengths differ')
    truth={};seen=set();stores={(r['seed'],r['arm']):{m:{} for m in ('original','bias_only','current_affine')} for r in cfg['inputs']['endpoints']}
    for row in guard.project():
        seed,arm,cmp,i=row['seed'],row['arm'],row['comparison'],row['source_row'];key=(seed,arm,cmp,i)
        d.p.require(key not in seen and (seed,arm) in stores,'independent duplicate/extra frame')
        demo,t=pairs[i];tail=t>=cfg['inputs']['action_lengths'][demo]-20
        d.p.require((row['demo'],row['action_step'],row['tail20'])==(demo,t,tail) and t>=5,'independent causal ID differs')
        target=row['target_2'];d.p.require(.025<=abs(target)<=1 and f32(target)==target,'independent target scalar differs')
        if i not in truth:truth[i]=target
        d.p.require(truth[i]==target and row['target_sign']==(1 if target>0 else -1),'independent target/label join differs')
        pr,pc=row['reference_prediction_2'],row['candidate_prediction_2'];label=transition(target,pr,pc)
        re=not (pr>0 if target>0 else pr<0);ce=not (pc>0 if target>0 else pc<0)
        d.p.require(row['transition']==label and row['reference_error']==re and row['candidate_error']==ce
                    and row['reference_band']==band(pr) and row['candidate_band']==band(pc),'independent saved transition differs')
        er,ec=pr-target,pc-target
        d.p.require(abs(row['reference_sse']-er*er)<=1e-10 and abs(row['candidate_sse']-ec*ec)<=1e-10
                    and abs(row['delta_sse']-(ec*ec-er*er))<=1e-10,'independent SSE differs')
        for role,pred in [('reference',pr),('candidate',pc)]:
            d.p.require(abs(pred)<=1 and f32(pred)==pred and row[role+'_category']==category(pred,target),'independent prediction/category differs')
            saved=(pred,row[role+'_raw_residual'],row[role+'_clip'],row[role+'_category']);mode=row[role]
            existing=stores[seed,arm][mode]
            if i in existing:d.p.require(existing[i]==saved,'independent mode scalar disagreement')
            existing[i]=saved
        seen.add(key)
    ids=sorted(truth);sign={i:1 if truth[i]>0 else -1 for i in ids};tails={i:pairs[i][1]>=cfg['inputs']['action_lengths'][pairs[i][0]]-20 for i in ids}
    expected={(seed,arm,c+'-minus-'+r,i) for seed,arm in stores for c,r in d.COMPARISONS for i in ids}
    d.p.require(seen==expected and len(ids)==3598 and sum(sign[i]<0 for i in ids)==2273
                and sum(sign[i]>0 for i in ids)==1325 and sum(tails[i] and sign[i]<0 for i in ids)==402
                and sum(tails[i] and sign[i]>0 for i in ids)==75,'independent rectangle/material cohort differs')
    for data in stores.values():
        for mode in data:
            d.p.require(set(data[mode])==set(ids),'independent mode missing source')
            for i,(pred,delta,clip,_) in data[mode].items():
                original=data['original'][i][0];value=original+delta
                d.p.require(f32(max(-1.,min(1.,value)))==pred and clip==(value<-1 or value>1),'independent residual/clip reconstruction differs')
                if mode=='original':d.p.require(delta==0 and not clip,'independent original identity differs')
        for fold in range(3):
            values=[data['bias_only'][i][1] for i in ids if pairs[i][0]//10==fold]
            if values:d.p.require(min(values)==max(values) and f32(values[0])==values[0],'independent bias-only fold constant differs')
    return {'pairs':pairs,'truth':truth,'ids':ids,'sign':sign,'tail':tails,'bits':{i:bits(truth[i]) for i in ids},'bins':{i:bin_id(truth[i]) for i in ids}},stores


def vectors(shared,data,candidate,reference):
    out={}
    for i in shared['ids']:
        y=shared['truth'][i];pr,dr,_,_=data[reference][i];pc,dc,_,_=data[candidate][i];er=pr-y;ec=pc-y
        rs,cs=er*er,ec*ec;scale=abs(y);energy=scale*scale;delta=cs-rs;step=pc-pr
        values=[scale,abs(pr),abs(pr)/scale,rs,cs,delta,rs/energy,cs/energy,delta/energy,dc-dr,shared['sign'][i]*step,abs(step)/scale]
        d.p.require(all(math.isfinite(x) for x in values),'independent nonfinite metric')
        out[i]={'values':values,'transition':transition(y,pr,pc),'reference':pr,'candidate':pc}
    return out


def select(shared,phase,direction):
    s=-1 if direction=='negative' else 1
    return [i for i in shared['ids'] if shared['sign'][i]==s and (phase=='all_common' or shared['tail'][i]==(phase=='tail20'))]


def bin_stats(data,ids,class_n,bin_n):
    n=len(ids);re=sum(data[i]['transition'] in ('corrected','stays_wrong') for i in ids);ce=sum(data[i]['transition'] in ('new_error','stays_wrong') for i in ids)
    row={'class_n':class_n,'bin_n':bin_n,'n':n,'fraction_class':n/class_n if class_n else None,'fraction_bin':n/bin_n if bin_n else None,
         'reference_error_n':re,'candidate_error_n':ce,'reference_sse':math.fsum(data[i]['values'][3] for i in ids),
         'candidate_sse':math.fsum(data[i]['values'][4] for i in ids),'delta_sse':math.fsum(data[i]['values'][5] for i in ids),
         'target_min':min(data[i]['values'][0] for i in ids) if n else None,'target_max':max(data[i]['values'][0] for i in ids) if n else None,'net_error_n':ce-re}
    for j,metric in enumerate(d.METRICS):row[metric+'_mean']=math.fsum(data[i]['values'][j] for i in ids)/n if n else None
    return row


def independent_cell(focus,control,scheme,key):
    nf,nc=len(focus),len(control);matched=nf>0 and nc>0
    if scheme=='exact_target':
        d.p.require(nf+nc>0 and all(bits(row[0])==key for row in focus+control),'independent exact key differs')
    else:d.p.require(all(bin_id(row[0])==key for row in focus+control),'independent bin key differs')
    result={'scheme':scheme,'cell_key':key,'focus_n':nf,'control_n':nc,'matched':matched,
            'missing_reason':'' if matched else 'both_empty' if nf==nc==0 else 'focus_empty' if nf==0 else 'control_empty'}
    for label,rows in [('focus',focus),('control',control)]:
        result[label]={metric:math.fsum(row[j] for row in rows)/len(rows) if rows else None for j,metric in enumerate(d.METRICS)}
        result[label+'_target_min']=min(row[0] for row in rows) if rows else None;result[label+'_target_max']=max(row[0] for row in rows) if rows else None
    result['delta']={metric:result['focus'][metric]-result['control'][metric] if matched else None for metric in d.METRICS}
    if matched and scheme=='exact_target':d.p.require(result['delta']['target_abs']==0,'independent exact imbalance')
    return result


def encode_cell(cell):
    scheme=cell['scheme'];key=cell['cell_key'];target=struct.unpack('<f',struct.pack('<I',key))[0] if scheme=='exact_target' else None
    b=bin_id(target) if target is not None else key
    row={k:cell[k] for k in ('scheme','cell_key','focus_n','control_n','matched','missing_reason','focus_target_min','focus_target_max','control_target_min','control_target_max')}
    row.update(exact_target_value=target,bin_index=b,bin_left=d.BIN_EDGES[b],bin_right=d.BIN_EDGES[b+1],right_closed=b==6)
    for group in ('focus','control','delta'):
        for metric in d.METRICS:row[group+'_'+metric]=cell[group][metric]
    return row


def independent_demo(cells,scheme):
    d.p.require(len({c['cell_key'] for c in cells})==len(cells),'independent duplicate cell')
    common=[c for c in cells if c['focus_n']>0 and c['control_n']>0];nf=sum(c['focus_n'] for c in cells);nc=sum(c['control_n'] for c in cells)
    reason='' if common else 'both_empty' if nf==nc==0 else 'focus_empty' if nf==0 else 'control_empty' if nc==0 else 'no_common_exact_target' if scheme=='exact_target' else 'no_common_target_bin'
    row={'scheme':scheme,'cell_n':len(cells),'matched_cell_n':len(common),'focus_total_n':nf,'control_total_n':nc,
         'focus_matched_n':sum(c['focus_n'] for c in common),'control_matched_n':sum(c['control_n'] for c in common),'missing_reason':reason}
    for group,total in [('focus',nf),('control',nc)]:
        row[group+'_unmatched_n']=total-row[group+'_matched_n'];row[group+'_coverage']=row[group+'_matched_n']/total if total else None
    for scope,cohort in [('pooled_full',cells),('pooled_common',common)]:
        row[scope]={}
        for group in ('focus','control'):
            total=sum(c[group+'_n'] for c in cohort)
            row[scope][group]={metric:math.fsum(c[group+'_n']*c[group][metric] for c in cohort if c[group+'_n'])/total if total else None for metric in d.METRICS}
        row[scope]['delta']={metric:row[scope]['focus'][metric]-row[scope]['control'][metric]
                             if row[scope]['focus'][metric] is not None and row[scope]['control'][metric] is not None else None for metric in d.METRICS}
    row['equal_cell']={group:{metric:math.fsum(c[group][metric] for c in common)/len(common) if common else None for metric in d.METRICS} for group in ('focus','control','delta')}
    return row


def independent_model(demos):
    d.p.require(len(demos)==30 and {r['demo'] for r in demos}==set(range(30)),'independent all30 demos required')
    common=[r for r in demos if r['matched_cell_n']>0]
    out={'eligible_demo_n':len(common),'ineligible_demo_n':30-len(common),'matched_cell_n':sum(r['matched_cell_n'] for r in common)}
    for group in ('focus','control'):
        total=sum(r[group+'_total_n'] for r in demos);n=sum(r[group+'_matched_n'] for r in demos)
        out[group+'_total_n']=total;out[group+'_matched_n']=n;out[group+'_unmatched_n']=total-n;out[group+'_coverage']=n/total if total else None
    for scope,suffix in [('pooled_full','_total_n'),('pooled_common','_matched_n')]:
        out[scope]={}
        for group in ('focus','control'):
            n=out[group+suffix]
            out[scope][group]={metric:math.fsum(r[group+suffix]*r[scope][group][metric] for r in demos if r[group+suffix])/n if n else None for metric in d.METRICS}
        out[scope]['delta']={metric:out[scope]['focus'][metric]-out[scope]['control'][metric]
                            if out[scope]['focus'][metric] is not None and out[scope]['control'][metric] is not None else None for metric in d.METRICS}
    out['standardized']={group:{metric:math.fsum(r['equal_cell'][group][metric] for r in common)/len(common) if common else None for metric in d.METRICS} for group in ('focus','control','delta')}
    return out


def flatten(summary):
    out={key:value for key,value in summary.items() if not isinstance(value,dict)}
    for scope in ('pooled_full','pooled_common','equal_cell','standardized'):
        if scope in summary:
            for group in ('focus','control','delta'):
                for metric in d.METRICS:out[scope+'_'+group+'_'+metric]=summary[scope][group][metric]
    return out


def compressed(cell):return (cell['focus_n'],cell['control_n'],tuple(cell['focus'][m] for m in d.METRICS),tuple(cell['control'][m] for m in d.METRICS))


def uncompressed(value,scheme,key):
    nf,nc,f,c=value
    return {'scheme':scheme,'cell_key':key,'focus_n':nf,'control_n':nc,'focus':dict(zip(d.METRICS,f)),'control':dict(zip(d.METRICS,c)),
            'delta':{metric:x-y for metric,x,y in zip(d.METRICS,f,c)},'matched':True}


def audit_model(model,shared,store,tables,cache,model_summaries):
    for candidate,reference in d.COMPARISONS:
        cmp=candidate+'-minus-'+reference;data=vectors(shared,store,candidate,reference)
        base={k:model[k] for k in ('task','seed','arm')};base.update(comparison=cmp,candidate=candidate,reference=reference)
        for phase,direction in itertools.product(d.PHASES,d.DIRECTIONS):
            ids=select(shared,phase,direction);prefix={**base,'phase':phase,'direction':direction}
            for b in range(7):
                within=[i for i in ids if shared['bins'][i]==b];parts=[]
                for label in d.TRANSITIONS:
                    use=[i for i in within if data[i]['transition']==label];stats=bin_stats(data,use,len(ids),len(within));parts.append(stats)
                    tables['bin-transitions'].row({**prefix,'bin_index':b,'bin_left':d.BIN_EDGES[b],'bin_right':d.BIN_EDGES[b+1],'right_closed':b==6,'transition':label,**stats})
                d.p.require(sum(x['n'] for x in parts)==len(within) and sum(x['net_error_n'] for x in parts)==parts[2]['n']-parts[1]['n']
                            and abs(math.fsum(x['delta_sse'] for x in parts)-math.fsum(data[i]['values'][5] for i in within))<=1e-10,'independent bin conservation differs')
            for focus,control in d.CONTRASTS:
                label=focus+'-minus-'+control;ident={**prefix,'contrast':label,'focus':focus,'control':control}
                for scheme in d.SCHEMES:
                    groups={focus:defaultdict(list),control:defaultdict(list)}
                    for i in ids:
                        transition_label=data[i]['transition']
                        if transition_label in groups:
                            key=(shared['pairs'][i][0],shared['bits'][i] if scheme=='exact_target' else shared['bins'][i])
                            groups[transition_label][key].append(data[i]['values'])
                    universe=set(groups[focus])|set(groups[control]);paired={};demos=[]
                    for demo in range(30):
                        keys=sorted(k for dm,k in universe if dm==demo) if scheme=='exact_target' else list(range(7));cells=[]
                        for key in keys:
                            cell=independent_cell(groups[focus].get((demo,key),[]),groups[control].get((demo,key),[]),scheme,key);cells.append(cell)
                            tables['exact-target-cells' if scheme=='exact_target' else 'coarse-bin-cells'].row({**ident,'demo':demo,**encode_cell(cell)})
                            if cell['matched']:paired[demo,key]=compressed(cell)
                        result={'demo':demo,**independent_demo(cells,scheme)};demos.append(result)
                        tables['demo-matched'].row({**ident,**flatten(result)})
                    model_result=independent_model(demos);nf,nc=model_result['focus_total_n'],model_result['control_total_n']
                    model_result['missing_reason']='' if model_result['eligible_demo_n']>0 else 'both_empty' if nf==nc==0 else 'focus_empty' if nf==0 else 'control_empty' if nc==0 else 'no_common_exact_target' if scheme=='exact_target' else 'no_common_target_bin'
                    tables['model-matched'].row({**ident,'scheme':scheme,**flatten(model_result)})
                    key=(model['seed'],model['arm'],cmp,phase,direction,label,scheme);cache[key]=paired;model_summaries[key]=model_result
        for i in shared['ids']:
            tr=data[i]['transition'];focus,control=d.CONTRASTS[0 if tr in ('corrected','stays_wrong') else 1];label=focus+'-minus-'+control
            demo,t=shared['pairs'][i];direction='positive' if shared['sign'][i]>0 else 'negative';own='tail20' if shared['tail'][i] else 'complement'
            row={**base,'source_row':i,'demo':demo,'action_step':t,'target_sign':shared['sign'][i],'tail20':shared['tail'][i],
                 'target_2':shared['truth'][i],'reference_prediction_2':data[i]['reference'],'candidate_prediction_2':data[i]['candidate'],
                 'target_bin':shared['bins'][i],'exact_target_bits':shared['bits'][i],'transition':tr,'contrast':label,'group_role':'focus' if tr==focus else 'control'}
            for scheme,tag in [('exact_target','exact'),('coarse_bin','coarse')]:
                cell=(demo,shared['bits'][i] if scheme=='exact_target' else shared['bins'][i])
                for phase,phase_tag in [('all_common','all_common'),(own,'own_phase')]:
                    row[tag+'_matched_'+phase_tag]=cell in cache[model['seed'],model['arm'],cmp,phase,direction,label,scheme]
            tables['frame-strata'].row(row)


def three_values(values):
    actual=[v for v in values if v is not None]
    return {'values':values,'mean':math.fsum(actual)/3 if len(actual)==3 else None,'higher_seed_n':sum(v>0 for v in actual),
            'tie_seed_n':sum(v==0 for v in actual),'lower_seed_n':sum(v<0 for v in actual),'null_seed_n':3-len(actual)}


def audit_common(cache,models,table):
    summaries=[]
    for arm,(candidate,reference),phase,direction,(focus,control),scheme in itertools.product(d.p.ARMS,d.COMPARISONS,d.PHASES,d.DIRECTIONS,d.CONTRASTS,d.SCHEMES):
        cmp=candidate+'-minus-'+reference;label=focus+'-minus-'+control;keys=[(seed,arm,cmp,phase,direction,label,scheme) for seed in d.p.SEEDS]
        common=set(cache[keys[0]]).intersection(cache[keys[1]],cache[keys[2]]);restricted=[]
        for key in keys:
            demos=[]
            for demo in range(30):
                cells=[uncompressed(cache[key][dm,k],scheme,k) for dm,k in sorted(common) if dm==demo]
                demos.append({'demo':demo,**independent_demo(cells,scheme)})
            restricted.append(independent_model(demos))
        individual=[models[key] for key in keys]
        prefix={'task':7,'arm':arm,'comparison':cmp,'candidate':candidate,'reference':reference,'phase':phase,'direction':direction,'contrast':label,'focus':focus,'control':control,'scheme':scheme}
        out={**prefix,'common_cell_n':len(common),'common_demo_n':len({dm for dm,_ in common})}
        for seed,full,subset in zip(d.p.SEEDS,individual,restricted):
            tag='seed'+str(seed)+'_';out[tag+'individual_matched_cell_n']=full['matched_cell_n'];out[tag+'individual_eligible_demo_n']=full['eligible_demo_n']
            for group in ('focus','control'):
                total=full[group+'_total_n'];matched=subset[group+'_matched_n'];out[tag+group+'_total_n']=total
                out[tag+'individual_'+group+'_matched_n']=full[group+'_matched_n'];out[tag+'common_'+group+'_matched_n']=matched
                out[tag+'common_'+group+'_coverage']=matched/total if total else None
            for metric in d.METRICS:
                out[tag+'individual_delta_'+metric]=full['standardized']['delta'][metric]
                out[tag+'common_delta_'+metric]=subset['standardized']['delta'][metric]
        summary={**prefix,'seeds':d.p.SEEDS,'common_cell_n':len(common),'common_demo_n':len({dm for dm,_ in common}),
                 'individual_eligible_demo_n_values':[r['eligible_demo_n'] for r in individual],
                 'individual_support':{scope:{metric:three_values([r[scope]['delta'][metric] for r in individual]) for metric in d.METRICS} for scope in ('pooled_full','pooled_common','standardized')},
                 'common3_support':{metric:three_values([r['standardized']['delta'][metric] for r in restricted]) for metric in d.METRICS}}
        for metric,values in summary['common3_support'].items():
            for key in ('mean','higher_seed_n','tie_seed_n','lower_seed_n','null_seed_n'):out['common_'+metric+'_'+key]=values[key]
        table.row(out);summaries.append(summary)
    d.p.require(len(summaries)==144,'independent seed summary count differs');return summaries


def compare_tree(actual,expected,checks,path='seed_summary'):
    if isinstance(expected,dict):
        d.p.require(isinstance(actual,dict) and set(actual)==set(expected),'independent JSON keys differ: '+path)
        for key in expected:compare_tree(actual[key],expected[key],checks,path+'/'+key)
    elif isinstance(expected,list):
        d.p.require(isinstance(actual,list) and len(actual)==len(expected),'independent JSON length differs: '+path)
        for i,(a0,e0) in enumerate(zip(actual,expected)):compare_tree(a0,e0,checks,path+'/'+str(i))
    else:checks.equal(actual,expected,path)


def validate_record(cfg,record,science_pin,binding_pin):
    d.p.require(record['status']=='complete_task7_target_magnitude_matching_pending_audit' and record['formal_result'] is False
                and record['science_config']==science_pin and record['binding']==binding_pin and d.exact(record['budget'],cfg['budget'])
                and record['inputs']==cfg['inputs'] and record['parents']==cfg['parents']
                and record['new_original_CSV_array_archive_checkpoint_reads_including_hashes']==0
                and record['new_research_training_forward_optimizer_environment_neighbor_query_sealed_calls']==0
                and record['all_parent_cues_false_unchanged'] is True and record['G_failed_H_unrun_unchanged'] is True
                and record['SPT_Knowledge_Module_changes'] is False,'independent provenance differs')
    access=record['access']
    d.p.require(access['CSV_data_passes']=={'attempted':1,'completed':1} and access['CSV_lexical_rows']=={'paired_frames':64764}
                and access['CSV_projected_cells']=={'paired_frames':1813392} and access['projected_columns']==d.COLUMNS
                and access['byte_SHA_checks']==['paired_frames']*3 and not access['denied_operations']
                and access['original_CSV_array_archive_checkpoint_reads_including_hashes']==access['research_training_forward_optimizer_environment_sealed_calls']==0,'independent field budget differs')
    runtime,cost=record['runtime'],record['cost']
    d.p.require(runtime['device']=='CPU' and runtime['environment']==cfg['execution']['environment']
                and {k:runtime[k] for k in ('python','numpy')}==cfg['execution']['runtime']
                and runtime['environment_seed'] is None and runtime['initial_state_indices']==[] and cost['elapsed_seconds']<=180
                and cost['peak_RSS_bytes']<=536870912,'independent runtime/cost differs')
    expected=set(cfg['budget']['fixed_table_rows'])|set(cfg['budget']['variable_table_rows_max']);d.p.require(set(record['tables'])==expected,'independent table set differs')
    for name,ref in record['tables'].items():
        d.p.require(ref['path']==str(Path(cfg['artifacts'])/(name+'.csv')) and d.p.sha(ref['path'])==ref['sha256'],'independent table path/SHA differs')
        if name in cfg['budget']['fixed_table_rows']:d.p.require(ref['rows']==cfg['budget']['fixed_table_rows'][name],'independent fixed row count differs')
        else:d.p.require(0<=ref['rows']<=cfg['budget']['variable_table_rows_max'][name],'independent variable cell cap differs')
    d.p.require(sum(ref['rows'] for ref in record['tables'].values())<=256212 and sum(Path(ref['path']).stat().st_size for ref in record['tables'].values())<=268435456,'independent output caps differ')


def audit_payload(cfg,science_pin,binding_pin):
    directory=Path(cfg['artifacts']);output=Path(cfg['output']);audit_path=directory/'independent-audit.json'
    d.p.require(not any((directory/name).exists() for name in ('independent-audit.json','completion.json','audit-failed-partial.json')),'existing audit/partial; no retry')
    record=json.loads(output.read_text());result_pin=d.p.pin(output);validate_record(cfg,record,science_pin,binding_pin)
    limits=a.base.Limits(cfg['execution']['audit_active_seconds_max'],cfg['execution']['working_memory_bytes_max']);checks=Checks()
    guard=a.Guard(cfg['inputs'],[ref['path'] for ref in record['tables'].values()],limits);tables={};cache={};models={};started=a.base.now()
    try:
        with guard:
            guard.hash_inputs();tables={key:StrictTable(ref,checks,limits) for key,ref in record['tables'].items()}
            shared,stores=reconstruct_source(guard,cfg)
            for model in cfg['inputs']['endpoints']:
                audit_model(model,shared,stores[model['seed'],model['arm']],tables,cache,models)
                print(json.dumps({'stage':'audit','seed':model['seed'],'arm':model['arm'],'scalar_checks':checks.n}),flush=True);limits.check()
            summaries=audit_common(cache,models,tables['common3-summary']);compare_tree(record['seed_summary'],summaries,checks)
            for table in tables.values():table.finish()
            guard.complete()
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output changed during audit')
        d.p.require(d.p.pin(output)==result_pin,'pending result mutated')
        result={'status':'passed_independent_task7_target_magnitude_matching_audit','formal_result':False,'science_config':science_pin,'binding':binding_pin,'result':result_pin,
                'started_at_utc':started,'completed_at_utc':a.base.now(),'runtime':a.base.runtime(cfg),'cost':limits.record(),'access':guard.record(),
                'scalar_checks':checks.n,'tables_verified':record['tables'],'seed_summary_rows_verified':144,
                'new_original_CSV_array_archive_checkpoint_reads_including_hashes':0,'new_research_training_forward_optimizer_environment_neighbor_query_sealed_calls':0,
                'primary_numerical_implementation_imported':'experiments.analyze_libero_task7_target_magnitude_matching' in sys.modules}
        d.p.require(result['primary_numerical_implementation_imported'] is False,'independent audit imported primary')
        a.base.write_new(audit_path,result)
        a.base.write_new(directory/'completion.json',{'status':'complete_independently_audited_task7_target_magnitude_matching','formal_result':False,
            'science_config':science_pin,'binding':binding_pin,'result':result_pin,'audit':d.p.pin(audit_path),'completed_at_utc':a.base.now(),
            'total_source_CSV_data_passes':2,'total_source_CSV_lexical_rows':129528,'total_source_projected_cells':3626784,
            'output_table_rows_verified':sum(ref['rows'] for ref in record['tables'].values()),'exact_target_cells_verified':record['tables']['exact-target-cells']['rows'],
            'seed_summary_rows_verified':144,'new_original_CSV_array_archive_checkpoint_reads_including_hashes':0,
            'new_research_training_forward_optimizer_environment_neighbor_query_sealed_calls':0,'all_parent_cues_false_unchanged':True,
            'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False})
        return result
    except BaseException as exc:
        for table in tables.values():
            if not table.handle.closed:table.handle.close()
        a.base.write_new(directory/'audit-failed-partial.json',{'status':'failed_no_retry','error':repr(exc),'access':guard.record(),'scalar_checks':checks.n})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    args=parser.parse_args();cfg,_=a.preflight(args.config,args.binding);a.base.runtime(cfg)
    result=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding));print(json.dumps({'status':result['status'],'scalar_checks':result['scalar_checks']}))
