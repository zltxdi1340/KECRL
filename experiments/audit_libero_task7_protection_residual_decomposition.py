"""Independent single-pass reconstruction; no primary numerical implementation."""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
import json
import math
from pathlib import Path
import struct
import sys

from experiments import libero_task7_protection_residual_decomposition_protocol as d
from experiments import libero_task7_protection_residual_decomposition_access as a
from experiments.audit_libero_task7_component3_history_transition import Checks,CheckedTable

SUMMARY_FIELDS=('n','cohort_n','baseline_abs_mean','candidate_abs_mean','delta_abs_mean','ratio_median',
                'baseline_mse','candidate_mse','delta_mse','baseline_harm_mean','candidate_harm_mean',
                'delta_harm_sum','delta_harm_cohort_contribution','baseline_wrong_n','candidate_wrong_n',
                'baseline_clip_n','candidate_clip_n','closed_segment_n','strict_toward_original_n')


class StrictTable(CheckedTable):
    def row(self,expected):
        actual=next(self.reader,None)
        d.p.require(actual is not None and set(actual)==set(expected),'independent row/schema differs')
        exact_frame=Path(self.ref['path']).name=='paired-frames.csv'
        for key,value in expected.items():
            text=actual[key]
            if value is None:self.checks.equal(text,'',key)
            elif type(value) is float:
                if exact_frame:self.checks.equal(float(text)==value,True,'exact frame/'+key)
                else:self.checks.equal(float(text),value,key)
            else:self.checks.equal(text,str(value),key)
        self.n+=1
        if self.n%128==0:self.limits.check()


def same_tree(actual,expected,checks,name='record'):
    if isinstance(expected,dict):
        d.p.require(type(actual) is dict and set(actual)==set(expected),'independent dictionary schema differs: '+name)
        for key in expected:same_tree(actual[key],expected[key],checks,name+'/'+key)
    elif isinstance(expected,list):
        d.p.require(type(actual) is list and len(actual)==len(expected),'independent list schema differs: '+name)
        for i,value in enumerate(expected):same_tree(actual[i],value,checks,name+'/'+str(i))
    else:
        d.p.require(type(actual) is type(expected),'independent scalar type differs: '+name)
        checks.equal(actual,expected,name)


def movement(b,c):
    if b==0:return 'zero_stays' if c==0 else 'zero_to_nonzero'
    if c==0:return 'to_zero'
    if math.copysign(1,b)!=math.copysign(1,c):return 'sign_reversal'
    order=(abs(c)>abs(b))-(abs(c)<abs(b))
    return {-1:'same_sign_smaller',0:'same_sign_equal',1:'same_sign_larger'}[order]


def classify(target,pred):
    if target==pred:return 'exact_match'
    if target==0:return 'zero_target_nonzero'
    if pred==0:return 'prediction_zero'
    if (target>0)!=(pred>0):return 'wrong_sign'
    return 'same_sign_over' if abs(pred)>abs(target) else 'same_sign_under'


def frame(row0,rowb,rowc):
    y,p,b,c=row0['target_2'],row0['prediction_2'],rowb['prediction_2'],rowc['prediction_2']
    e0=(p-y)**2;eb=(b-y)**2;ec=(c-y)**2;hb=max(eb-e0,0.);hc=max(ec-e0,0.)
    material=abs(y)>=.025;sign=1 if y>0 else -1 if y<0 else 0
    original_wrong=material and sign*p<=0;bc=sign*b>0;cc=sign*c>0
    harm='stays_harmed' if eb>e0 and ec>e0 else 'resolved_harm' if eb>e0 else 'new_harm' if ec>e0 else 'unharmed_stays'
    correction=('retained' if bc and cc else 'lost' if bc else 'candidate_only' if cc else 'neither_corrected') if original_wrong else None
    db=b-p;dc=c-p;effective_move=movement(db,dc);inside=not((c<p and c<b) or (c>p and c>b))
    d.p.require(not inside or hc<=hb,'independent convexity mismatch')
    result={key:row0[key] for key in ('source_row','probe_fold','demo','action_step','target_direction','tail20')}
    result.update(original_good=material and e0<y*y,original_bad=material and not(e0<y*y),original_wrong=original_wrong,
        material=material,effective_baseline=db,effective_candidate=dc,effective_movement=effective_move,
        closed_segment=inside,strict_toward_original=effective_move in ('to_zero','same_sign_smaller'),
        e0=e0,eB=eb,eC=ec,hB=hb,hC=hc,deltaE=ec-eb,deltaH=hc-hb,harm_transition=harm,correction_transition=correction,
        baseline_corrected=bc if original_wrong else None,candidate_corrected=cc if original_wrong else None,
        target=y,original=p,baseline=b,candidate=c,raw_baseline=rowb['raw_residual'],raw_candidate=rowc['raw_residual'],
        baseline_clip=rowb['clip'],candidate_clip=rowc['clip'])
    result['raw_movement']=movement(rowb['raw_residual'],rowc['raw_residual'])
    for rep in ('raw','effective'):
        baseline=result[rep+'_baseline'];candidate=result[rep+'_candidate'];q=candidate/baseline if baseline!=0 else None
        finite=q is not None and math.isfinite(q)
        result[rep+'_ratio']=q if finite else None
        result[rep+'_ratio_reason']=None if finite else 'baseline_zero' if baseline==0 else 'nonfinite_ratio'
    return result


def reconstruct(guard,cfg):
    pairs=[(demo,t) for demo,length in enumerate(cfg['inputs']['action_lengths']) for t in range(1,length)]
    d.p.require(len(pairs)==4188,'independent layout length differs')
    ids={(r['seed'],r['arm']) for r in cfg['inputs']['endpoints']};data={key:{mode:{} for mode in d.MODES} for key in ids};truth={}
    for row in guard.project('frames'):
        key=row['seed'],row['arm'];mode=row['mode'];i=row['source_row'];demo,step=pairs[i]
        d.p.require(key in ids and i not in data[key][mode],'independent duplicate/unlisted frame')
        expected=(demo,step,demo//10,step>=cfg['inputs']['action_lengths'][demo]-20)
        d.p.require(tuple(row[name] for name in ('demo','action_step','probe_fold','tail20'))==expected,'independent causal metadata differs')
        y,pred=row['target_2'],row['prediction_2']
        d.p.require(-1<=y<=1 and -1<=pred<=1 and struct.unpack('<f',struct.pack('<f',y))[0]==y
                    and struct.unpack('<f',struct.pack('<f',pred))[0]==pred,'independent f32/range differs')
        truth.setdefault(i,struct.pack('<f',y))
        d.p.require(truth[i]==struct.pack('<f',y) and row['target_direction']==(1 if y>=.025 else -1 if y<=-.025 else 0),'independent shared truth differs')
        d.p.require(row['error_2']==pred-y and row['component3_category']==classify(y,pred),'independent error/category differs')
        data[key][mode][i]=row
    for modes in data.values():
        d.p.require(all(set(rows)==set(range(4188)) for rows in modes.values()),'independent missing source frame/mode')
        for i in range(4188):
            origin=modes['original'][i]
            d.p.require(origin['raw_residual']==0 and origin['clip'] is False,'independent original identity differs')
            for mode in d.MODES:
                r=modes[mode][i];before=origin['prediction_2']+r['raw_residual'];value=struct.pack('<f',max(-1.,min(1.,before)))
                d.p.require(value==struct.pack('<f',r['prediction_2']) and r['clip']==(before < -1 or before > 1),'independent clip identity differs')
    selected=[i for i,(_,step) in enumerate(pairs) if step>=5]
    labels=[1 if struct.unpack('<f',truth[i])[0]>=.025 else -1 if struct.unpack('<f',truth[i])[0]<=-.025 else 0 for i in selected]
    tails=[pairs[i][1]>=cfg['inputs']['action_lengths'][pairs[i][0]]-20 for i in selected]
    d.p.require(len(selected)==4068 and sum(v!=0 for v in labels)==3598 and labels.count(-1)==2273
                and sum(t and s<0 for t,s in zip(tails,labels))==402 and sum(t and s>0 for t,s in zip(tails,labels))==75,'independent material coverage differs')
    return data


def selected(frames,phase,direction):
    answer=[]
    for row in frames:
        if phase=='tail20' and not row['tail20']:continue
        if phase=='complement' and row['tail20']:continue
        if direction!='all' and row['target_direction']!=dict(negative=-1,positive=1,neutral=0)[direction]:continue
        answer.append(row)
    return answer


def distribution(values,prefix):
    ordered=sorted(values);n=len(ordered);out={prefix+'_mean':math.fsum(ordered)/n if n else None}
    for name,q in [('median',.5),('q25',.25),('q75',.75)]:
        if n==0:out[prefix+'_'+name]=None
        else:
            index=(n-1)*q;first=int(index);fraction=index-first
            out[prefix+'_'+name]=math.fsum([ordered[first]*(1-fraction),ordered[min(first+1,n-1)]*fraction])
    return out


def statistics(rows,cohort_n,representation='effective'):
    n=len(rows);out={'n':n,'cohort_n':cohort_n}
    base=[r[representation+'_baseline'] for r in rows];candidate=[r[representation+'_candidate'] for r in rows]
    measurements={'baseline_abs':[abs(v) for v in base],'candidate_abs':[abs(v) for v in candidate],
        'delta_abs':[abs(c)-abs(b) for b,c in zip(base,candidate)],'baseline_delta':base,'candidate_delta':candidate,
        'delta_signed':[c-b for b,c in zip(base,candidate)]}
    for prefix,values in measurements.items():out.update(distribution(values,prefix))
    ratios=[r[representation+'_ratio'] for r in rows if r[representation+'_ratio_reason'] is None]
    counts=Counter(r[representation+'_ratio_reason'] for r in rows)
    out.update(ratio_n=len(ratios),ratio_baseline_zero_n=counts['baseline_zero'],ratio_nonfinite_n=counts['nonfinite_ratio'])
    out.update(distribution(ratios,'ratio'))
    for prefix,error in [('original','e0'),('baseline','eB'),('candidate','eC')]:out[prefix+'_sse']=math.fsum(r[error] for r in rows)
    for prefix,error in [('baseline','eB'),('candidate','eC')]:
        out[prefix+'_mse']=out[prefix+'_sse']/n if n else None
        out[prefix+'_mae']=math.fsum(math.sqrt(r[error]) for r in rows)/n if n else None
    de=math.fsum(r['eC']-r['eB'] for r in rows);dh=math.fsum(r['hC']-r['hB'] for r in rows)
    out['delta_sse']=de;out['delta_mse']=de/n if n else None
    for prefix,harm in [('baseline','hB'),('candidate','hC')]:
        total=math.fsum(r[harm] for r in rows);out[prefix+'_harm_sum']=total;out[prefix+'_harm_mean']=total/n if n else None
        out[prefix+'_harmed_n']=sum(r[harm]>0 for r in rows)
    out.update(delta_harm_sum=dh,delta_harm_mean=dh/n if n else None,
        delta_harm_cohort_contribution=dh/cohort_n if cohort_n else None,delta_mse_cohort_contribution=de/cohort_n if cohort_n else None)
    material=[r for r in rows if abs(r['target'])>=.025];out['material_n']=len(material)
    for prefix in ('original','baseline','candidate'):
        margins=[math.copysign(1,r['target'])*r[prefix] for r in material]
        out[prefix+'_wrong_n']=sum(v<=0 for v in margins);out[prefix+'_margin_mean']=math.fsum(margins)/len(margins) if margins else None
    for prefix in ('baseline','candidate'):out[prefix+'_clip_n']=sum(r[prefix+'_clip'] for r in rows)
    out['closed_segment_n']=sum(r['closed_segment'] for r in rows);out['strict_toward_original_n']=sum(r['strict_toward_original'] for r in rows)
    return out


def demo_description(records):
    cell=[r for r in records if r['n']];cohort=[r for r in records if r['cohort_n']]
    result={'cell_nonempty_demo_n':len(cell),'cohort_nonempty_demo_n':len(cohort)}
    for key in ['delta_abs_mean','delta_mse','delta_harm_mean']:
        result[key+'_equal_demo']=math.fsum(r[key] for r in cell)/len(cell) if cell else None
    result['delta_harm_contribution_equal_demo']=math.fsum(r['delta_harm_cohort_contribution'] for r in cohort)/len(cohort) if cohort else None
    return result


def parity_projection(frames):
    result={'good':{},'retention':{},'metrics':{mode:{} for mode in d.MODES}}
    for phase in d.PHASES:
        result['good'][phase]={};result['retention'][phase]={}
        for direction in d.DIRECTIONS:
            pop=selected(frames,phase,direction);material=[r for r in pop if r['material']];n=len(pop)
            for mode,field in [('original','original'),('uniform_affine','baseline'),('harm_protected_affine','candidate')]:
                mse=math.fsum((r[field]-r['target'])**2 for r in pop)/n if n else None
                rate=sum(math.copysign(1,r['target'])*r[field]<=0 for r in material)/len(material) if material else None
                result['metrics'][mode].setdefault(phase,{})[direction]=dict(n=n,material_n=len(material),component3_mse=mse,material_sign_error_rate=rate)
            if direction in ('all','neutral'):continue
            good=[r for r in pop if r['original_good']];wrong=[r for r in pop if r['original_wrong']]
            goodout={'n':len(good),'new_harm_n':sum(r['hB']==0 and r['hC']>0 for r in good),'resolved_harm_n':sum(r['hB']>0 and r['hC']==0 for r in good)}
            for mode,error,harm in [('uniform_affine','eB','hB'),('harm_protected_affine','eC','hC')]:
                goodout[mode]={'harm_mean':math.fsum(r[harm] for r in good)/len(good) if good else None,
                    'harmed_n':sum(r[harm]>0 for r in good),'component3_mse':math.fsum(r[error] for r in good)/len(good) if good else None}
            result['good'][phase][direction]=goodout
            base={r['source_row'] for r in wrong if r['baseline_corrected']};candidate={r['source_row'] for r in wrong if r['candidate_corrected']}
            result['retention'][phase][direction]={'n':n,'original_wrong_n':len(wrong),'baseline_corrected_n':len(base),'candidate_corrected_n':len(candidate),
                'retained_n':len(base&candidate),'lost_n':len(base-candidate),'candidate_only_n':len(candidate-base),
                'neither_n':len(wrong)-len(base|candidate),'retention_rate':len(base&candidate)/len(base) if base else None,'retention_evidence':bool(base)}
    return result


def ident(model):return {key:model[key] for key in ('task','seed','arm')}


def check_group(model,panel,phase,direction,representation,label,transition,rows,pop,tables,records):
    table={'movement':'movement-summary','harm':'harm-decomposition','correction':'correction-decomposition'}[panel]
    demo_table={'movement':'movement-demo-summary','harm':'harm-demo-decomposition','correction':'correction-demo-decomposition'}[panel]
    prefix={**ident(model),'phase':phase,'direction':direction,'representation':representation,'movement':label}
    if panel!='movement':prefix['transition']=transition
    value=statistics(rows,len(pop),representation);by_demo=defaultdict(list);sizes=Counter(r['demo'] for r in pop)
    for row in rows:by_demo[row['demo']].append(row)
    demos=[]
    for demo in range(30):
        stats=statistics(by_demo[demo],sizes[demo],representation);demos.append(stats)
        tables[demo_table].row({**prefix,'demo':demo,**stats})
    tables[table].row({**prefix,**value,**demo_description(demos)})
    key=(panel,phase,direction,representation,label,transition)
    records.setdefault((model['seed'],model['arm']),{})[key]={name:value[name] for name in SUMMARY_FIELDS}
    return value


def check_model(model,frames,reference,tables,records,checks):
    expected=parity_projection(frames);before=checks.n;same_tree(expected,{key:reference[key] for key in ('good','retention','metrics')},checks,'parent_parity')
    parent_fields=checks.n-before;d.p.require(parent_fields==258,'independent parent parity field count differs')
    for row in frames:tables['paired-frames'].row({**ident(model),**row})
    for phase in ['all_common','tail20','complement']:
        full=selected(frames,phase,'all');moves_checked=0;focal_checked=0
        for direction in ['all','negative','positive','neutral']:
            pop=selected(frames,phase,direction)
            for rep in ['raw','effective']:
                buckets=defaultdict(list)
                for row in pop:buckets[row[rep+'_movement']].append(row)
                for label in d.MOVEMENT_ROWS:check_group(model,'movement',phase,direction,rep,label,'',pop if label=='all_movements' else buckets[label],pop,tables,records)
                d.p.require(sum(len(buckets[label]) for label in d.MOVEMENTS)==len(pop),'independent movement partition mismatch');moves_checked+=1
        for panel,attribute,events in [('harm','original_good',d.HARM_TRANSITIONS),('correction','original_wrong',d.CORRECTION_TRANSITIONS)]:
            for direction in ['negative','positive']:
                cohort=[r for r in selected(frames,phase,direction) if r[attribute]];buckets=defaultdict(list)
                for row in cohort:buckets[row[panel+'_transition'],row['effective_movement']].append(row)
                parts=[]
                for event in ['all_transitions',*events]:
                    for label in d.MOVEMENT_ROWS:
                        rows=[r for r in cohort if (event=='all_transitions' or r[panel+'_transition']==event) and (label=='all_movements' or r['effective_movement']==label)]
                        value=check_group(model,panel,phase,direction,'effective',label,event,rows,cohort,tables,records)
                        if event!='all_transitions' and label!='all_movements':parts.append(value)
                full_stats=statistics(cohort,len(cohort));d.p.require(sum(r['n'] for r in parts)==len(cohort),'independent focal partition mismatch')
                for field in ['original_sse','baseline_sse','candidate_sse','delta_sse','delta_harm_sum']:
                    checks.equal(math.fsum(r[field] for r in parts),full_stats[field],'independent contribution conservation/'+field)
                focal_checked+=1
        violations=sum(r['closed_segment'] and r['hC']>r['hB'] for r in full)
        new_inside=sum(r['closed_segment'] and r['hB']==0 and r['hC']>0 for r in full)
        d.p.require(violations==new_inside==0,'independent geometry invariant failed')
        tables['invariants'].row({**ident(model),'phase':phase,'n':len(full),'movement_partition_checks':moves_checked,
            'focal_partition_checks':focal_checked,'parent_parity_fields':86,'closed_segment_harm_violations':violations,
            'new_harm_inside_segment_violations':new_inside,'raw_effective_class_disagreement_n':sum(r['raw_movement']!=r['effective_movement'] for r in full),
            'clip_changed_n':sum(r['baseline_clip']!=r['candidate_clip'] for r in full),
            'new_harm_original_good_n':sum(r['original_good'] and r['hB']==0 and r['hC']>0 for r in full),
            'lost_original_wrong_n':sum(r['correction_transition']=='lost' for r in full),
            'strict_toward_original_n':sum(r['strict_toward_original'] for r in full),'outside_segment_n':sum(not r['closed_segment'] for r in full)})
    return parent_fields


def summary(records):
    answer=[]
    for arm in d.p.ARMS:
        for key in records[d.p.SEEDS[0],arm]:
            panel,phase,direction,rep,label,event=key;parts=[records[seed,arm][key] for seed in d.p.SEEDS]
            supported=all(row['n']!=0 for row in parts);metrics={}
            for field in SUMMARY_FIELDS:
                vals=[row[field] for row in parts];valid=supported and None not in vals
                metrics[field]={'values':vals,'complete_mean':math.fsum(vals)/len(vals) if valid else None,
                    'complete_median':sorted(vals)[1] if valid else None,
                    'directions':['null' if value is None else 'positive' if value>0 else 'negative' if value<0 else 'zero' for value in vals]}
            answer.append(dict(arm=arm,panel=panel,phase=phase,direction=direction,representation=rep,movement=label,transition=event,
                seeds=list(d.p.SEEDS),cell_supported_in_all3=supported,support_seed_n=sum(r['n']>0 for r in parts),metrics=metrics))
    d.p.require(len(answer)==1344,'independent seed summary count differs');return answer


def validate_record(cfg,record,science_pin,binding_pin):
    d.p.require(record['status']=='complete_task7_protection_residual_decomposition_pending_audit' and record['formal_result'] is False
                and record['science_config']==science_pin and record['binding']==binding_pin and d.exact(record['inputs'],cfg['inputs'])
                and d.exact(record['parents'],cfg['parents']) and d.exact(record['budget'],cfg['budget']),'main provenance differs')
    d.p.require(d.exact(record['computation'],dict(raw_triplets_checked=25128,common_frames_analyzed=24408,movement_classifications=48816,parent_parity_fields=1548,output_CSV_rows=149418)),'main logical computation budget differs')
    expected_access={'CSV_data_passes':{'attempted':1,'completed':1},'CSV_lexical_rows':{'frames':75384},'CSV_projected_cells':{'frames':1281528},
        'lexically_tokenized_cells':2638440,'byte_SHA_checks':['frames']*3,'projected_columns':d.COLUMNS.copy(),'unselected_columns_numeric_converted':0,
        'denied_operations':[],'archive_array_head_checkpoint_reads_including_hashes':0,'training_forward_environment_or_sealed_calls':0}
    d.p.require(d.exact(record['access'],expected_access),'main access/projection counts differ')
    d.p.require(type(record['parent_parity_checks']) is int and record['parent_parity_checks']==1548
                and all(type(record[key]) is int and record[key]==0 for key in ('archive_array_head_checkpoint_reads_including_byte_hashes','new_training_forward_environment_neighbor_query_sealed_calls'))
                and record['all_parent_joint_cues_false_unchanged'] is True and record['G_failed_H_unrun_unchanged'] is True
                and record['SPT_Knowledge_Module_changes'] is False,'main role/stop/parity boundary differs')
    runtime=record['runtime'];cost=record['cost']
    d.p.require({key:runtime[key] for key in ('python','numpy')}==cfg['execution']['runtime'] and runtime['device']=='CPU'
                and runtime['environment']==cfg['execution']['environment'] and runtime['environment_seed'] is None and runtime['initial_state_indices']==[]
                and cost['elapsed_seconds']<=180 and cost['peak_RSS_bytes']<=536870912,'main runtime/cost differs')
    d.p.require(set(record['tables'])==set(cfg['budget']['tables']) and len(record['seed_summary'])==1344,'main table/summary identity differs')
    for name,ref in record['tables'].items():
        d.p.require(ref['path']==str(Path(cfg['artifacts'])/(name+'.csv')) and type(ref['rows']) is int and ref['rows']==cfg['budget']['tables'][name]
                    and d.p.sha(ref['path'])==ref['sha256'],'main output identity/hash differs')
    d.p.require(sum(Path(ref['path']).stat().st_size for ref in record['tables'].values())<=268435456,'main CSV bytes exceeded')


def audit_payload(cfg,science_pin,binding_pin):
    primary='experiments.analyze_libero_task7_protection_residual_decomposition'
    d.p.require(primary not in sys.modules and 'torch' not in sys.modules,'independent audit imported primary numerics/Torch')
    directory=Path(cfg['artifacts']);output=Path(cfg['output']);audit_path=directory/'independent-audit.json'
    d.p.require(not any((directory/name).exists() for name in ('independent-audit.json','completion.json','audit-failed-partial.json')),'existing audit/partial; no retry')
    record=json.loads(output.read_text());result_pin=d.p.pin(output);validate_record(cfg,record,science_pin,binding_pin)
    limits=a.base.Limits(180,536870912);runtime=a.base.runtime(cfg);checks=Checks();tables={};cache={};parity=0;common_frames=0;started=a.base.now()
    guard=a.Guard(cfg['inputs'],[ref['path'] for ref in record['tables'].values()],limits)
    try:
        with guard:
            guard.hash_inputs();tables={name:StrictTable(ref,checks,limits) for name,ref in record['tables'].items()}
            source=reconstruct(guard,cfg);raw_triplets=sum(len(modes['original']) for modes in source.values())
            for model in cfg['inputs']['endpoints']:
                key=model['seed'],model['arm'];modes=source.pop(key)
                frames=[frame(*(modes[mode][i] for mode in d.MODES)) for i in range(4188) if modes['original'][i]['action_step']>=5]
                common_frames+=len(frames)
                reference=next(row for row in cfg['inputs']['parent_parity'] if (row['seed'],row['arm'])==key)
                parity+=check_model(model,frames,reference,tables,cache,checks)
                print(json.dumps({'stage':'audit','seed':key[0],'arm':key[1],'checks':checks.n,'parity_fields':parity}),flush=True);limits.check()
            same_tree(record['seed_summary'],summary(cache),checks,'seed_summary')
            for table in tables.values():table.finish()
            d.p.require(parity==1548 and primary not in sys.modules and 'torch' not in sys.modules,'audit parity/independence differs')
            guard.complete()
        for ref in record['tables'].values():d.p.require(d.p.sha(ref['path'])==ref['sha256'],'output changed during audit')
        d.p.require(d.p.pin(output)==result_pin,'main pending result changed')
        result={'status':'passed_independent_task7_protection_residual_decomposition_audit','formal_result':False,'science_config':science_pin,'binding':binding_pin,
            'result':result_pin,'started_at_utc':started,'completed_at_utc':a.base.now(),'access':guard.record(),'scalar_checks':checks.n,
            'computation':{'raw_triplets_checked':raw_triplets,'common_frames_analyzed':common_frames,'movement_classifications':common_frames*2,'parent_parity_fields':parity,'output_CSV_rows':sum(table.n for table in tables.values())},
            'parent_parity_checks':parity,'tables_verified':record['tables'],'seed_summary_rows_verified':1344,'runtime':runtime,'cost':limits.record(),
            'primary_numerical_implementation_imported':False,'archive_array_head_checkpoint_reads_including_byte_hashes':0,
            'new_training_forward_environment_neighbor_query_sealed_calls':0,'all_parent_joint_cues_false_unchanged':True,
            'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False}
        a.base.write_new(audit_path,result)
        a.base.write_new(directory/'completion.json',{'status':'complete_independently_audited_task7_protection_residual_decomposition','formal_result':False,
            'science_config':science_pin,'binding':binding_pin,'result':result_pin,'audit':d.p.pin(audit_path),'completed_at_utc':a.base.now(),
            'total_input_CSV_data_passes':2,'total_projected_cells':2563056,'total_lexically_tokenized_cells':5276880,
            'output_rows_verified':149418,'seed_summary_rows_verified':1344,'parent_parity_fields_per_stage':1548,
            'archive_array_head_checkpoint_reads_including_byte_hashes':0,'new_training_forward_environment_neighbor_query_sealed_calls':0,
            'all_parent_joint_cues_false_unchanged':True,'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False})
        return result
    except BaseException as error:
        for table in tables.values():
            if not table.handle.closed:table.handle.close()
        a.base.write_new(directory/'audit-failed-partial.json',{'status':'failed_no_retry','error':repr(error),'access':guard.record(),'scalar_checks':checks.n,'parent_parity_checks':parity})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING);args=parser.parse_args()
    cfg,_=a.preflight(args.config,args.binding);result=audit_payload(cfg,d.p.pin(args.config),d.p.pin(args.binding))
    print(json.dumps({'status':result['status'],'scalar_checks':result['scalar_checks']}))
