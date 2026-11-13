"""Single-pass saved-output geometry/accounting; no training or new predictions."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

from experiments import libero_task7_protection_residual_decomposition_protocol as d
from experiments import libero_task7_protection_residual_decomposition_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child

SUMMARY_FIELDS=('n','cohort_n','baseline_abs_mean','candidate_abs_mean','delta_abs_mean','ratio_median',
                'baseline_mse','candidate_mse','delta_mse','baseline_harm_mean','candidate_harm_mean',
                'delta_harm_sum','delta_harm_cohort_contribution','baseline_wrong_n','candidate_wrong_n',
                'baseline_clip_n','candidate_clip_n','closed_segment_n','strict_toward_original_n')


def identity(model):return {key:model[key] for key in ('task','seed','arm')}


def load_frames(guard,cfg):
    plan=d.layout.source_layout(cfg['inputs']['action_lengths']);keys={(r['seed'],r['arm']) for r in cfg['inputs']['endpoints']}
    stores={key:{mode:[None]*4188 for mode in d.MODES} for key in keys};truth=[None]*4188
    for row in guard.project('frames'):
        key=row['seed'],row['arm'];i=row['source_row'];mode=row['mode']
        d.p.require(key in stores and stores[key][mode][i] is None,'duplicate or unlisted source identity')
        d.p.require(all(row[name]==value for name,value in plan[i].items()),'source causal ID differs')
        y,pred=row['target_2'],row['prediction_2']
        d.p.require(abs(y)<=1 and abs(pred)<=1 and d.layout.float32(y)==y and d.layout.float32(pred)==pred,'saved float32/range differs')
        bits=struct.pack('<f',y)
        if truth[i] is None:truth[i]=bits
        d.p.require(truth[i]==bits and row['target_direction']==(1 if y>=.025 else -1 if y<=-.025 else 0),'shared target/direction differs')
        d.p.require(row['error_2']==pred-y and row['component3_category']==d.layout.error_category(y,pred),'stored error/category differs')
        stores[key][mode][i]=row
    d.p.require(all(value is not None for value in truth),'missing shared target')
    for store in stores.values():
        d.p.require(all(all(row is not None for row in store[mode]) for mode in d.MODES),'missing source mode/frame')
        for i,original in enumerate(store['original']):
            p0=original['prediction_2']
            d.p.require(original['raw_residual']==0 and original['clip'] is False,'original identity differs')
            for mode in d.MODES:
                row=store[mode][i];before=p0+row['raw_residual'];rebuilt=d.layout.float32(min(1.,max(-1.,before)))
                d.p.require(struct.pack('<f',rebuilt)==struct.pack('<f',row['prediction_2']) and row['clip']==(abs(before)>1),'float32 clip identity differs')
    chosen=[row for row in plan if row['action_step']>=5]
    direction=[(1 if struct.unpack('<f',v)[0]>=.025 else -1 if struct.unpack('<f',v)[0]<=-.025 else 0) for v in truth]
    d.p.require(len(chosen)==4068 and sum(direction[r['source_row']]!=0 for r in chosen)==3598
                and sum(direction[r['source_row']]<0 for r in chosen)==2273
                and sum(r['tail20'] and direction[r['source_row']]<0 for r in chosen)==402
                and sum(r['tail20'] and direction[r['source_row']]>0 for r in chosen)==75,'source material coverage differs')
    return stores


def model_frames(store):
    rows=[]
    for i in range(4188):
        o,b,c=(store[mode][i] for mode in d.MODES)
        if o['action_step']<5:continue
        value=d.frame_geometry(o['target_2'],o['prediction_2'],b['prediction_2'],c['prediction_2'])
        row={key:o[key] for key in ('source_row','probe_fold','demo','action_step','target_direction','tail20')}
        row.update(value);row.update(target=o['target_2'],original=o['prediction_2'],baseline=b['prediction_2'],candidate=c['prediction_2'],
            raw_baseline=b['raw_residual'],raw_candidate=c['raw_residual'],baseline_clip=b['clip'],candidate_clip=c['clip'])
        row['raw_movement']=d.movement(row['raw_baseline'],row['raw_candidate'])
        for rep in d.REPRESENTATIONS:
            q=d.ratio(row[rep+'_baseline'],row[rep+'_candidate']);row[rep+'_ratio']=q['value'];row[rep+'_ratio_reason']=q['reason']
        rows.append(row)
    d.p.require(len(rows)==4068,'common frame coverage differs');return rows


def selected(frames,phase,direction):
    return [r for r in frames if (phase=='all_common' or r['tail20']==(phase=='tail20'))
            and (direction=='all' or r['target_direction']=={'negative':-1,'positive':1,'neutral':0}[direction])]


def describe(values,prefix):
    values=list(values);ordered=sorted(values);n=len(values)
    out={prefix+'_mean':math.fsum(values)/n if n else None}
    for key,q in [('median',.5),('q25',.25),('q75',.75)]:
        if not n:out[prefix+'_'+key]=None;continue
        pos=(n-1)*q;lo=math.floor(pos);fraction=pos-lo
        out[prefix+'_'+key]=math.fsum([(1-fraction)*ordered[lo],fraction*ordered[min(lo+1,n-1)]])
    return out


def statistics(rows,cohort_n,representation='effective'):
    n=len(rows);out={'n':n,'cohort_n':cohort_n}
    b=[r[representation+'_baseline'] for r in rows];c=[r[representation+'_candidate'] for r in rows]
    for prefix,values in [('baseline_abs',map(abs,b)),('candidate_abs',map(abs,c)),('delta_abs',(abs(y)-abs(x) for x,y in zip(b,c))),
                          ('baseline_delta',b),('candidate_delta',c),('delta_signed',(y-x for x,y in zip(b,c)))]:out.update(describe(values,prefix))
    ratios=[r[representation+'_ratio'] for r in rows if r[representation+'_ratio'] is not None]
    out.update(ratio_n=len(ratios),ratio_baseline_zero_n=sum(r[representation+'_ratio_reason']=='baseline_zero' for r in rows),
               ratio_nonfinite_n=sum(r[representation+'_ratio_reason']=='nonfinite_ratio' for r in rows))
    out.update(describe(ratios,'ratio'))
    for prefix,key in [('original','e0'),('baseline','eB'),('candidate','eC')]:out[prefix+'_sse']=math.fsum(r[key] for r in rows)
    for prefix,key in [('baseline','eB'),('candidate','eC')]:
        out[prefix+'_mse']=out[prefix+'_sse']/n if n else None
        out[prefix+'_mae']=math.fsum(math.sqrt(r[key]) for r in rows)/n if n else None
    de=math.fsum(r['deltaE'] for r in rows);dh=math.fsum(r['deltaH'] for r in rows)
    out['delta_sse']=de;out['delta_mse']=de/n if n else None
    for prefix,key in [('baseline','hB'),('candidate','hC')]:
        total=math.fsum(r[key] for r in rows);out[prefix+'_harm_sum']=total;out[prefix+'_harm_mean']=total/n if n else None
        out[prefix+'_harmed_n']=sum(r[key]>0 for r in rows)
    out.update(delta_harm_sum=dh,delta_harm_mean=dh/n if n else None,
               delta_harm_cohort_contribution=dh/cohort_n if cohort_n else None,delta_mse_cohort_contribution=de/cohort_n if cohort_n else None)
    material=[r for r in rows if r['material']];out['material_n']=len(material)
    for prefix in ('original','baseline','candidate'):
        margins=[(1 if r['target']>0 else -1)*r[prefix] for r in material]
        out[prefix+'_wrong_n']=sum(v<=0 for v in margins)
        out[prefix+'_margin_mean']=math.fsum(margins)/len(margins) if margins else None
    out.update(baseline_clip_n=sum(r['baseline_clip'] for r in rows),candidate_clip_n=sum(r['candidate_clip'] for r in rows),
               closed_segment_n=sum(r['closed_segment'] for r in rows),strict_toward_original_n=sum(r['strict_toward_original'] for r in rows))
    return out


def demo_description(records):
    out={'cell_nonempty_demo_n':sum(r['n']>0 for r in records),'cohort_nonempty_demo_n':sum(r['cohort_n']>0 for r in records)}
    for key in ('delta_abs_mean','delta_mse','delta_harm_mean'):
        values=[r[key] for r in records if r['n']>0]
        out[key+'_equal_demo']=math.fsum(values)/len(values) if values else None
    values=[r['delta_harm_cohort_contribution'] for r in records if r['cohort_n']>0]
    out['delta_harm_contribution_equal_demo']=math.fsum(values)/len(values) if values else None
    return out


def compare_parent(frames,reference):
    computed={'good':{},'retention':{},'metrics':{mode:{} for mode in d.MODES}}
    for phase in d.PHASES:
        computed['good'][phase]={};computed['retention'][phase]={}
        for direction in d.DIRECTIONS:
            rows=selected(frames,phase,direction);n=len(rows);mat=[r for r in rows if r['material']]
            for mode,key in zip(d.MODES,('original','baseline','candidate')):
                computed['metrics'][mode].setdefault(phase,{})[direction]={'n':n,'material_n':len(mat),
                    'component3_mse':math.fsum((r[key]-r['target'])**2 for r in rows)/n if n else None,
                    'material_sign_error_rate':sum((1 if r['target']>0 else -1)*r[key]<=0 for r in mat)/len(mat) if mat else None}
            if direction not in ('negative','positive'):continue
            good=[r for r in rows if r['original_good']];wrong=[r for r in rows if r['original_wrong']]
            computed['good'][phase][direction]={'n':len(good),'new_harm_n':sum(r['harm_transition']=='new_harm' for r in good),
                'resolved_harm_n':sum(r['harm_transition']=='resolved_harm' for r in good)}
            for mode,h,e in [('uniform_affine','hB','eB'),('harm_protected_affine','hC','eC')]:
                computed['good'][phase][direction][mode]={'harm_mean':math.fsum(r[h] for r in good)/len(good) if good else None,
                    'harmed_n':sum(r[h]>0 for r in good),'component3_mse':math.fsum(r[e] for r in good)/len(good) if good else None}
            counts={label:sum(r['correction_transition']==label for r in wrong) for label in d.CORRECTION_TRANSITIONS}
            bn=counts['lost']+counts['retained'];cn=counts['candidate_only']+counts['retained']
            computed['retention'][phase][direction]={'n':n,'original_wrong_n':len(wrong),'baseline_corrected_n':bn,'candidate_corrected_n':cn,
                'retained_n':counts['retained'],'lost_n':counts['lost'],'candidate_only_n':counts['candidate_only'],'neither_n':counts['neither_corrected'],
                'retention_rate':counts['retained']/bn if bn else None,'retention_evidence':bn>0}
    count=0
    def walk(value,expected):
        nonlocal count
        if isinstance(expected,dict):
            d.p.require(set(value)==set(expected),'parent parity schema differs')
            for key in expected:walk(value[key],expected[key])
        else:
            d.p.require(type(value) is type(expected) and (abs(value-expected)<=1e-10 if type(expected) is float else value==expected),'parent scalar parity differs')
            count+=1
    walk(computed,{key:reference[key] for key in ('good','retention','metrics')})
    d.p.require(count==258,'parent parity count differs');return count


def emit_group(model,panel,phase,direction,representation,label,transition,rows,cohort,sinks,summaries):
    name={'movement':'movement-summary','harm':'harm-decomposition','correction':'correction-decomposition'}[panel]
    demo_name={'movement':'movement-demo-summary','harm':'harm-demo-decomposition','correction':'correction-demo-decomposition'}[panel]
    prefix={**identity(model),'phase':phase,'direction':direction,'representation':representation,'movement':label}
    if panel!='movement':prefix['transition']=transition
    stat=statistics(rows,len(cohort),representation);demo_records=[]
    for demo in range(30):
        part=[r for r in rows if r['demo']==demo];denom=sum(r['demo']==demo for r in cohort)
        value=statistics(part,denom,representation);demo_records.append(value)
        sinks[demo_name].add({**prefix,'demo':demo,**value})
    sinks[name].add({**prefix,**stat,**demo_description(demo_records)})
    key=(panel,phase,direction,representation,label,transition)
    summaries.setdefault((model['seed'],model['arm']),{})[key]={field:stat[field] for field in SUMMARY_FIELDS}
    return stat


def model_tables(model,frames,reference,sinks,summaries):
    checks=compare_parent(frames,reference)
    for row in frames:sinks['paired-frames'].add({**identity(model),**row})
    for phase in d.PHASES:
        full=selected(frames,phase,'all');partition_checks=0;focal_checks=0
        for direction in d.DIRECTIONS:
            population=selected(frames,phase,direction)
            for representation in d.REPRESENTATIONS:
                groups=[]
                for label in d.MOVEMENT_ROWS:
                    rows=population if label=='all_movements' else [r for r in population if r[representation+'_movement']==label]
                    stat=emit_group(model,'movement',phase,direction,representation,label,'',rows,population,sinks,summaries)
                    if label!='all_movements':groups.append(stat)
                d.p.require(sum(r['n'] for r in groups)==len(population),'movement partition count differs');partition_checks+=1
        for panel,predicate,transitions in [('harm','original_good',d.HARM_TRANSITIONS),('correction','original_wrong',d.CORRECTION_TRANSITIONS)]:
            for direction in ('negative','positive'):
                cohort=[r for r in selected(frames,phase,direction) if r[predicate]];parts=[]
                whole=statistics(cohort,len(cohort))
                for transition in ['all_transitions',*transitions]:
                    for label in d.MOVEMENT_ROWS:
                        rows=[r for r in cohort if (transition=='all_transitions' or r[panel+'_transition']==transition)
                              and (label=='all_movements' or r['effective_movement']==label)]
                        value=emit_group(model,panel,phase,direction,'effective',label,transition,rows,cohort,sinks,summaries)
                        if transition!='all_transitions' and label!='all_movements':parts.append(value)
                d.p.require(sum(r['n'] for r in parts)==len(cohort),'focal transition partition differs')
                for key in ('original_sse','baseline_sse','candidate_sse','delta_sse','delta_harm_sum'):
                    d.p.require(abs(math.fsum(r[key] for r in parts)-whole[key])<=1e-10,'focal additive accounting differs')
                focal_checks+=1
        cv=sum(r['closed_segment'] and r['hC']>r['hB'] for r in full)
        nh=sum(r['closed_segment'] and r['harm_transition']=='new_harm' for r in full)
        d.p.require(cv==nh==0,'convexity/new-harm segment violation')
        sinks['invariants'].add({**identity(model),'phase':phase,'n':len(full),'movement_partition_checks':partition_checks,
            'focal_partition_checks':focal_checks,'parent_parity_fields':checks//3,'closed_segment_harm_violations':cv,
            'new_harm_inside_segment_violations':nh,'raw_effective_class_disagreement_n':sum(r['raw_movement']!=r['effective_movement'] for r in full),
            'clip_changed_n':sum(r['baseline_clip']!=r['candidate_clip'] for r in full),
            'new_harm_original_good_n':sum(r['original_good'] and r['harm_transition']=='new_harm' for r in full),
            'lost_original_wrong_n':sum(r['correction_transition']=='lost' for r in full),
            'strict_toward_original_n':sum(r['strict_toward_original'] for r in full),'outside_segment_n':sum(not r['closed_segment'] for r in full)})
    return checks


def seed_summary(records):
    result=[]
    for arm in d.p.ARMS:
        keys=list(records[d.p.SEEDS[0],arm])
        for key in keys:
            panel,phase,direction,rep,movement,transition=key
            per_seed=[{'seed':seed,**records[seed,arm][key]} for seed in d.p.SEEDS]
            complete=all(row['n']>0 for row in per_seed);summary={}
            for field in SUMMARY_FIELDS:
                values=[row[field] for row in per_seed];valid=complete and all(value is not None for value in values)
                summary[field]={'values':values,'complete_mean':math.fsum(values)/3 if valid else None,
                    'complete_median':sorted(values)[1] if valid else None,
                    'directions':['null' if value is None else 'positive' if value>0 else 'negative' if value<0 else 'zero' for value in values]}
            result.append({'arm':arm,'panel':panel,'phase':phase,'direction':direction,'representation':rep,'movement':movement,'transition':transition,
                'seeds':list(d.p.SEEDS),'cell_supported_in_all3':complete,'support_seed_n':sum(row['n']>0 for row in per_seed),'metrics':summary})
    d.p.require(len(result)==1344,'seed summary coverage differs');return result


def analyze_payload(cfg,science_pin,binding_pin):
    directory=Path(cfg['artifacts']);output=Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(),'existing run/partial; no retry')
    directory.mkdir(parents=True);limits=a.base.Limits(180,536870912);runtime=a.base.runtime(cfg)
    paths={name:directory/(name+'.csv') for name in cfg['budget']['tables']};guard=a.Guard(cfg['inputs'],paths.values(),limits)
    sinks={};summaries={};parity=0;common_frames=0;started=a.base.now()
    try:
        with guard:
            guard.hash_inputs();stores=load_frames(guard,cfg)
            raw_triplets=sum(len(modes['original']) for modes in stores.values())
            sinks={name:a.base.CsvSink(path,limits) for name,path in paths.items()}
            for model in cfg['inputs']['endpoints']:
                key=model['seed'],model['arm'];frames=model_frames(stores.pop(key));common_frames+=len(frames)
                ref=next(r for r in cfg['inputs']['parent_parity'] if (r['seed'],r['arm'])==key)
                parity+=model_tables(model,frames,ref,sinks,summaries)
                print(json.dumps({'stage':'analysis','seed':key[0],'arm':key[1],'parity_fields':parity}),flush=True);limits.check()
            summary=seed_summary(summaries)
            for sink in sinks.values():sink.close()
            d.p.require({name:sink.rows for name,sink in sinks.items()}==cfg['budget']['tables'] and parity==1548,'output/parity budget differs')
            d.p.require(sum(path.stat().st_size for path in paths.values())<=268435456,'CSV byte ceiling exceeded');guard.complete()
        result={'status':'complete_task7_protection_residual_decomposition_pending_audit','formal_result':False,'science_config':science_pin,'binding':binding_pin,
            'started_at_utc':started,'completed_at_utc':a.base.now(),'inputs':cfg['inputs'],'parents':cfg['parents'],'budget':cfg['budget'],
            'tables':{name:sink.pin() for name,sink in sinks.items()},'access':guard.record(),'runtime':runtime,'cost':limits.record(),
            'computation':{'raw_triplets_checked':raw_triplets,'common_frames_analyzed':common_frames,'movement_classifications':common_frames*2,'parent_parity_fields':parity,'output_CSV_rows':sum(sink.rows for sink in sinks.values())},
            'parent_parity_checks':parity,'seed_summary':summary,'archive_array_head_checkpoint_reads_including_byte_hashes':0,
            'new_training_forward_environment_neighbor_query_sealed_calls':0,'all_parent_joint_cues_false_unchanged':True,
            'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False}
        a.base.write_new(output,result);return result
    except BaseException as error:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        a.base.write_new(directory/'analysis-failed-partial.json',{'status':'failed_no_retry','error':repr(error),'access':guard.record(),'parent_parity_checks':parity})
        raise


def analyze(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,_=a.preflight(config,binding);return analyze_payload(cfg,d.p.pin(config),d.p.pin(binding))


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(path.exists() for path in (launch,final,output,Path(cfg['artifacts']))),'existing run/partial; no retry')
    record={'started_at_utc':a.base.now(),'science_config':d.p.pin(config),'binding':d.p.pin(binding),'processes':[]};a.base.write_new(launch,record)
    start=time.monotonic();child=None
    try:
        for stage,module in [('analysis','experiments.analyze_libero_task7_protection_residual_decomposition'),('audit','experiments.audit_libero_task7_protection_residual_decomposition')]:
            env=os.environ.copy();env.update(bound['CPU_environment']);command=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            if stage=='analysis':command+=['--mode','analyze']
            row={'stage':stage,'command':command,'started_at_utc':a.base.now()};record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as handle:
                child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(195,max(1,415-(time.monotonic()-start))))
                except BaseException:stop_child(child);row['interrupted_or_watchdog']=True;raise
            row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-start<=420,'overall watchdog exceeded');record['status']='complete_independently_audited'
    except BaseException as error:
        if child is not None:stop_child(child)
        record['status']='failed_no_retry';record['error']=repr(error);raise
    finally:
        record['wall_seconds']=time.monotonic()-start;record['completed_at_utc']=a.base.now();a.base.write_new(final,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    parser.add_argument('--mode',choices=['preflight','analyze','run'],default='preflight');args=parser.parse_args()
    if args.mode=='preflight':a.preflight(args.config,args.binding);result={'status':'bound_no_payload'}
    elif args.mode=='analyze':result=analyze(args.config,args.binding)
    else:result=run(args.config,args.binding)
    print(json.dumps({'status':result['status']}))
