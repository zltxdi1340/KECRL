"""Final500 derived-CSV target strata and within-demo common-support comparisons."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

from experiments import libero_task7_target_magnitude_matching_protocol as d
from experiments import libero_task7_target_magnitude_matching_access as a
from experiments.analyze_libero_task7_component3_history_transition import stop_child


def name(candidate, reference): return candidate+'-minus-'+reference


def contrast_name(focus, control): return focus+'-minus-'+control


def load_source(guard, cfg):
    layout=d.previous.source_layout(cfg['inputs']['action_lengths']);n=len(layout)
    shared={key:np.asarray([row[key] for row in layout]) for key in ('demo','action_step','tail20')}
    y=np.full(n,np.nan);stores={}
    for model in cfg['inputs']['endpoints']:
        stores[model['seed'],model['arm']]={key:np.zeros((3,n),dtype=dtype) for key,dtype in
            [('p',np.float64),('raw',np.float64),('clip',bool),('cat',np.int8),('mode_seen',bool),('comparison_seen',bool)]}
    for row in guard.project():
        i=row['source_row'];store=stores[row['seed'],row['arm']];cmp=d.COMPARISONS.index([row['candidate'],row['reference']])
        d.p.require(not store['comparison_seen'][cmp,i] and all(row[k]==layout[i][k] for k in ('source_row','demo','action_step','tail20')), 'duplicate or mismatched frame ID')
        target=row['target_2'];d.target_bits(target);sign=1 if target>0 else -1
        d.p.require(row['target_sign']==sign,'target direction differs')
        if np.isnan(y[i]):y[i]=target
        d.p.require(y[i]==target,'shared target differs across endpoints/comparisons')
        pr,pc=row['reference_prediction_2'],row['candidate_prediction_2']
        quantity=d.previous.pair_quantities(target,pr,pc,row['reference_raw_residual'],row['candidate_raw_residual'])
        d.p.require(row['transition']==quantity['transition'] and row['reference_error']==quantity['reference_error']
                    and row['candidate_error']==quantity['candidate_error'] and row['reference_band']==quantity['reference_band']
                    and row['candidate_band']==quantity['candidate_band'],'saved transition/error/band differs')
        d.p.require(abs(row['reference_sse']-(pr-target)**2)<=1e-10 and abs(row['candidate_sse']-(pc-target)**2)<=1e-10
                    and abs(row['delta_sse']-quantity['delta_sse'])<=1e-10,'saved SSE differs')
        for role in ('reference','candidate'):
            m=d.previous.MODES.index(row[role]);pred=row[role+'_prediction_2'];raw=row[role+'_raw_residual'];clip=row[role+'_clip']
            category=d.previous.error_category(target,pred)
            d.p.require(category==row[role+'_category'] and d.previous.float32(pred)==pred,'saved category/float32 differs')
            code=d.previous.CLASSES.index(category)
            if store['mode_seen'][m,i]:
                d.p.require((store['p'][m,i],store['raw'][m,i],bool(store['clip'][m,i]),int(store['cat'][m,i]))==(pred,raw,clip,code),'mode scalars differ across comparisons')
            else:
                store['p'][m,i]=pred;store['raw'][m,i]=raw;store['clip'][m,i]=clip;store['cat'][m,i]=code;store['mode_seen'][m,i]=True
        store['comparison_seen'][cmp,i]=True
    ids=np.flatnonzero(np.isfinite(y));sign=np.zeros(n,np.int8);sign[ids]=np.where(y[ids]>0,1,-1)
    d.p.require(len(ids)==3598 and np.all(shared['action_step'][ids]>=5) and sum(sign[ids]<0)==2273 and sum(sign[ids]>0)==1325
                and sum(shared['tail20'][ids]&(sign[ids]<0))==402 and sum(shared['tail20'][ids]&(sign[ids]>0))==75,'material cohort differs')
    for store in stores.values():
        d.p.require(store['mode_seen'][:,ids].all() and store['comparison_seen'][:,ids].all()
                    and int(store['comparison_seen'].sum())==3*3598,'missing endpoint/comparison/frame')
        for m,mode in enumerate(d.previous.MODES):
            for i in ids:d.previous.validate_action_scalar(float(y[i]),float(store['p'][m,i]),float(store['raw'][m,i]),float(store['p'][0,i]),
                                                          float(store['p'][m,i]-y[i]),bool(store['clip'][m,i]),mode)
        for fold in range(3):
            values=store['raw'][1,ids[(shared['demo'][ids]//10)==fold]]
            if len(values):d.p.require(np.all(values==values[0]) and d.previous.float32(float(values[0]))==values[0],'bias fold constant differs')
    bins=np.full(n,-1,np.int8);bits=np.zeros(n,np.uint32)
    for i in ids:bins[i]=d.target_bin(float(y[i]));bits[i]=d.target_bits(float(y[i]))
    return {**shared,'y':y,'sign':sign,'ids':ids,'bin':bins,'bits':bits},stores


def context(shared,store,candidate,reference):
    c,r=d.previous.MODES.index(candidate),d.previous.MODES.index(reference);ids=shared['ids']
    pr,pc=store['p'][r],store['p'][c];goodr=shared['sign']*pr>0;goodc=shared['sign']*pc>0
    kind=np.where(goodr,np.where(goodc,0,2),np.where(goodc,1,3));values=np.full((4188,len(d.METRICS)),np.nan)
    for i in ids:
        row=d.frame_metrics(float(shared['y'][i]),float(pr[i]),float(pc[i]),float(store['raw'][r,i]),float(store['raw'][c,i]))
        values[i]=[row[k] for k in d.METRICS]
    return {'p_ref':pr,'p_cand':pc,'kind':kind,'values':values}


def phase_ids(shared,phase,direction):
    ids=shared['ids'];mask=shared['sign'][ids]==(-1 if direction=='negative' else 1)
    if phase!='all_common':mask&=shared['tail20'][ids]==(phase=='tail20')
    return ids[mask]


def bin_row(ctx,ids,class_n,bin_n):
    n=len(ids);values=ctx['values'][ids]
    row={'class_n':class_n,'bin_n':bin_n,'n':n,'fraction_class':n/class_n if class_n else None,'fraction_bin':n/bin_n if bin_n else None,
         'reference_error_n':int(np.count_nonzero(np.isin(ctx['kind'][ids],[1,3]))),
         'candidate_error_n':int(np.count_nonzero(np.isin(ctx['kind'][ids],[2,3]))),
         'reference_sse':float(np.sum(values[:,3])),'candidate_sse':float(np.sum(values[:,4])),'delta_sse':float(np.sum(values[:,5])),
         'target_min':float(np.min(values[:,0])) if n else None,'target_max':float(np.max(values[:,0])) if n else None}
    row['net_error_n']=row['candidate_error_n']-row['reference_error_n']
    row.update({key+'_mean':float(np.mean(values[:,j])) if n else None for j,key in enumerate(d.METRICS)})
    return row


def cell_row(cell):
    scheme=cell['scheme'];key=cell['cell_key']
    target=struct.unpack('<f',struct.pack('<I',key))[0] if scheme=='exact_target' else None
    b=d.target_bin(target) if scheme=='exact_target' else key
    row={k:cell[k] for k in ('scheme','cell_key','focus_n','control_n','matched','missing_reason','focus_target_min','focus_target_max','control_target_min','control_target_max')}
    row.update(exact_target_value=target,bin_index=b,bin_left=d.BIN_EDGES[b],bin_right=d.BIN_EDGES[b+1],right_closed=b==6)
    for group in ('focus','control','delta'):
        row.update({group+'_'+metric:cell[group][metric] for metric in d.METRICS})
    return row


def summary_row(summary):
    row={k:v for k,v in summary.items() if not isinstance(v,dict)}
    for scope in ('pooled_full','pooled_common','equal_cell','standardized'):
        if scope in summary:
            for group in ('focus','control','delta'):
                row.update({scope+'_'+group+'_'+metric:summary[scope][group][metric] for metric in d.METRICS})
    return row


def compact(cell):
    return (cell['focus_n'],cell['control_n'],tuple(cell['focus'][k] for k in d.METRICS),tuple(cell['control'][k] for k in d.METRICS))


def restore(value,scheme,key):
    nf,nc,f,c=value;return {'scheme':scheme,'cell_key':key,'focus_n':nf,'control_n':nc,'matched':True,
                           'focus':dict(zip(d.METRICS,f)),'control':dict(zip(d.METRICS,c)),
                           'delta':{k:a0-b0 for k,a0,b0 in zip(d.METRICS,f,c)}}


def matched_tables(model,shared,store,sinks,matched_cache,models):
    for candidate,reference in d.COMPARISONS:
        cmp=name(candidate,reference);ctx=context(shared,store,candidate,reference)
        base={k:model[k] for k in ('task','seed','arm')};base.update(comparison=cmp,candidate=candidate,reference=reference)
        for phase,direction in itertools.product(d.PHASES,d.DIRECTIONS):
            ids=phase_ids(shared,phase,direction);prefix={**base,'phase':phase,'direction':direction}
            for b in range(7):
                use=ids[shared['bin'][ids]==b];partition=[]
                for k,transition in enumerate(d.TRANSITIONS):
                    sub=use[ctx['kind'][use]==k];stats=bin_row(ctx,sub,len(ids),len(use));partition.append(stats)
                    sinks['bin-transitions'].add({**prefix,'bin_index':b,'bin_left':d.BIN_EDGES[b],'bin_right':d.BIN_EDGES[b+1],'right_closed':b==6,'transition':transition,**stats})
                d.p.require(sum(x['n'] for x in partition)==len(use)
                            and sum(x['net_error_n'] for x in partition)==partition[2]['n']-partition[1]['n']
                            and abs(sum(x['delta_sse'] for x in partition)-float(np.sum(ctx['values'][use,5])))<=1e-10,'bin/transition conservation differs')
            for focus,control in d.CONTRASTS:
                label=contrast_name(focus,control);ident={**prefix,'contrast':label,'focus':focus,'control':control}
                fk,ck=d.TRANSITIONS.index(focus),d.TRANSITIONS.index(control)
                for scheme in d.SCHEMES:
                    buckets=defaultdict(lambda:[[],[]])
                    for i in ids:
                        k=int(ctx['kind'][i])
                        if k not in (fk,ck):continue
                        key=(int(shared['demo'][i]),int(shared['bits'][i]) if scheme=='exact_target' else int(shared['bin'][i]))
                        buckets[key][0 if k==fk else 1].append(int(i))
                    demos=[];paired={}
                    for demo in range(30):
                        keys=sorted(k for dm,k in buckets if dm==demo) if scheme=='exact_target' else list(range(7));cells=[]
                        for key in keys:
                            fi,ci=buckets[demo,key]
                            frames=lambda use:[dict(zip(d.METRICS,ctx['values'][i].tolist())) for i in use]
                            cell=d.compare_cell(frames(fi),frames(ci),scheme,key);cells.append(cell)
                            sinks['exact-target-cells' if scheme=='exact_target' else 'coarse-bin-cells'].add({**ident,'demo':demo,**cell_row(cell)})
                            if cell['matched']:paired[demo,key]=compact(cell)
                        demo_summary={'demo':demo,**d.summarize_cells(cells,scheme)};demos.append(demo_summary)
                        sinks['demo-matched'].add({**ident,**summary_row(demo_summary)})
                    model_summary=d.summarize_demos(demos)
                    nf,nc=model_summary['focus_total_n'],model_summary['control_total_n']
                    model_summary['missing_reason']='' if model_summary['eligible_demo_n'] else 'both_empty' if nf==nc==0 else 'focus_empty' if nf==0 else 'control_empty' if nc==0 else 'no_common_exact_target' if scheme=='exact_target' else 'no_common_target_bin'
                    sinks['model-matched'].add({**ident,'scheme':scheme,**summary_row(model_summary)})
                    cache_key=(model['seed'],model['arm'],cmp,phase,direction,label,scheme)
                    matched_cache[cache_key]=paired;models[cache_key]=model_summary
        for i in shared['ids']:
            k=int(ctx['kind'][i]);focus,control=d.CONTRASTS[0 if k in (1,3) else 1];label=contrast_name(focus,control)
            direction='positive' if shared['sign'][i]>0 else 'negative';own='tail20' if shared['tail20'][i] else 'complement'
            row={**base,'source_row':int(i),'demo':int(shared['demo'][i]),'action_step':int(shared['action_step'][i]),'target_sign':int(shared['sign'][i]),'tail20':bool(shared['tail20'][i]),
                 'target_2':float(shared['y'][i]),'reference_prediction_2':float(ctx['p_ref'][i]),'candidate_prediction_2':float(ctx['p_cand'][i]),
                 'target_bin':int(shared['bin'][i]),'exact_target_bits':int(shared['bits'][i]),'transition':d.TRANSITIONS[k],
                 'contrast':label,'group_role':'focus' if d.TRANSITIONS[k]==focus else 'control'}
            for scheme,tag in [('exact_target','exact'),('coarse_bin','coarse')]:
                key=(int(shared['demo'][i]),int(shared['bits'][i]) if scheme=='exact_target' else int(shared['bin'][i]))
                for phase,phase_tag in [('all_common','all_common'),(own,'own_phase')]:
                    row[tag+'_matched_'+phase_tag]=key in matched_cache[model['seed'],model['arm'],cmp,phase,direction,label,scheme]
            sinks['frame-strata'].add(row)


def triplet(values):
    available=[x for x in values if x is not None]
    return {'values':values,'mean':math.fsum(available)/3 if len(available)==3 else None,'higher_seed_n':sum(x>0 for x in available),
            'tie_seed_n':sum(x==0 for x in available),'lower_seed_n':sum(x<0 for x in available),'null_seed_n':3-len(available)}


def common3_tables(cache,models,sink):
    summaries=[]
    for arm,(candidate,reference),phase,direction,(focus,control),scheme in itertools.product(d.p.ARMS,d.COMPARISONS,d.PHASES,d.DIRECTIONS,d.CONTRASTS,d.SCHEMES):
        cmp=name(candidate,reference);label=contrast_name(focus,control)
        keys=[(seed,arm,cmp,phase,direction,label,scheme) for seed in d.p.SEEDS];sets=[set(cache[key]) for key in keys]
        common=d.common3_keys(sets);common_models=[]
        for key in keys:
            demos=[]
            for demo in range(30):
                cells=[restore(cache[key][dm,k],scheme,k) for dm,k in sorted(common) if dm==demo]
                demos.append({'demo':demo,**d.summarize_cells(cells,scheme)})
            common_models.append(d.summarize_demos(demos))
        prefix={'task':7,'arm':arm,'comparison':cmp,'candidate':candidate,'reference':reference,'phase':phase,'direction':direction,'contrast':label,'focus':focus,'control':control,'scheme':scheme}
        row={**prefix,'common_cell_n':len(common),'common_demo_n':len({dm for dm,_ in common})}
        individuals=[models[key] for key in keys]
        for seed,individual,restricted in zip(d.p.SEEDS,individuals,common_models):
            tag='seed'+str(seed)+'_'
            row[tag+'individual_matched_cell_n']=individual['matched_cell_n'];row[tag+'individual_eligible_demo_n']=individual['eligible_demo_n']
            for group in ('focus','control'):
                total=individual[group+'_total_n'];matched=restricted[group+'_matched_n']
                row[tag+group+'_total_n']=total;row[tag+'individual_'+group+'_matched_n']=individual[group+'_matched_n']
                row[tag+'common_'+group+'_matched_n']=matched;row[tag+'common_'+group+'_coverage']=matched/total if total else None
            for metric in d.METRICS:
                row[tag+'individual_delta_'+metric]=individual['standardized']['delta'][metric]
                row[tag+'common_delta_'+metric]=restricted['standardized']['delta'][metric]
        summary={**prefix,'seeds':d.p.SEEDS,'common_cell_n':len(common),'common_demo_n':len({dm for dm,_ in common}),
                 'individual_eligible_demo_n_values':[r['eligible_demo_n'] for r in individuals],
                 'individual_support':{scope:{metric:triplet([r[scope]['delta'][metric] for r in individuals]) for metric in d.METRICS}
                                       for scope in ('pooled_full','pooled_common','standardized')},
                 'common3_support':{metric:triplet([r['standardized']['delta'][metric] for r in common_models]) for metric in d.METRICS}}
        for metric,item in summary['common3_support'].items():
            for key in ('mean','higher_seed_n','tie_seed_n','lower_seed_n','null_seed_n'):row['common_'+metric+'_'+key]=item[key]
        sink.add(row);summaries.append(summary)
    d.p.require(len(summaries)==144,'seed summary budget differs');return summaries


def analyze_payload(cfg,science_pin,binding_pin):
    directory,output=Path(cfg['artifacts']),Path(cfg['output'])
    d.p.require(not directory.exists() and not output.exists(),'existing run/partial; no retry');directory.mkdir(parents=True)
    limits=a.base.Limits(cfg['execution']['analysis_active_seconds_max'],cfg['execution']['working_memory_bytes_max'])
    names=list(cfg['budget']['fixed_table_rows'])+list(cfg['budget']['variable_table_rows_max'])
    paths={key:directory/(key+'.csv') for key in names};guard=a.Guard(cfg['inputs'],paths.values(),limits)
    sinks={};cache={};models={};started=a.base.now()
    try:
        with guard:
            guard.hash_inputs();shared,stores=load_source(guard,cfg);sinks={key:a.base.CsvSink(path,limits) for key,path in paths.items()}
            for model in cfg['inputs']['endpoints']:
                matched_tables(model,shared,stores[model['seed'],model['arm']],sinks,cache,models)
                print(json.dumps({'stage':'analysis','seed':model['seed'],'arm':model['arm']}),flush=True);limits.check()
            summaries=common3_tables(cache,models,sinks['common3-summary'])
            for sink in sinks.values():sink.close()
            for key,n in cfg['budget']['fixed_table_rows'].items():d.p.require(sinks[key].rows==n,'fixed output row count differs: '+key)
            for key,n in cfg['budget']['variable_table_rows_max'].items():d.p.require(sinks[key].rows<=n,'variable cell row cap exceeded')
            d.p.require(sum(s.rows for s in sinks.values())<=cfg['budget']['total_output_table_rows_max']
                        and sum(path.stat().st_size for path in paths.values())<=cfg['execution']['output_CSV_bytes_max'],'output row/byte cap exceeded')
            guard.complete()
        result={'status':'complete_task7_target_magnitude_matching_pending_audit','formal_result':False,'started_at_utc':started,'completed_at_utc':a.base.now(),
                'science_config':science_pin,'binding':binding_pin,'inputs':cfg['inputs'],'parents':cfg['parents'],'budget':cfg['budget'],
                'tables':{key:sink.pin() for key,sink in sinks.items()},'seed_summary':summaries,'access':guard.record(),
                'runtime':a.base.runtime(cfg),'cost':limits.record(),'new_original_CSV_array_archive_checkpoint_reads_including_hashes':0,
                'new_research_training_forward_optimizer_environment_neighbor_query_sealed_calls':0,
                'all_parent_cues_false_unchanged':True,'G_failed_H_unrun_unchanged':True,'SPT_Knowledge_Module_changes':False}
        a.base.write_new(output,result);return result
    except BaseException as exc:
        for sink in sinks.values():
            if not sink.handle.closed:sink.close()
        a.base.write_new(directory/'analysis-failed-partial.json',{'status':'failed_no_retry','error':repr(exc),'access':guard.record()})
        raise


def analyze(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,_=a.preflight(config,binding);a.base.runtime(cfg);return analyze_payload(cfg,d.p.pin(config),d.p.pin(binding))


def run(config=d.DEFAULT_CONFIG,binding=d.DEFAULT_BINDING):
    cfg,bound=a.preflight(config,binding);output=Path(cfg['output']);launch=output.with_name(output.stem+'-launch.json');final=output.with_name(output.stem+'-process.json')
    d.p.require(not any(p.exists() for p in (output,Path(cfg['artifacts']),launch,final)),'existing run/partial; no retry')
    record={'started_at_utc':a.base.now(),'science_config':d.p.pin(config),'binding':d.p.pin(binding),'processes':[]};a.base.write_new(launch,record)
    begun=time.monotonic();child=None
    try:
        for stage,module in [('analysis','experiments.analyze_libero_task7_target_magnitude_matching'),('audit','experiments.audit_libero_task7_target_magnitude_matching')]:
            command=[sys.executable,'-m',module,'--config',str(config),'--binding',str(binding)]
            if stage=='analysis':command+=['--mode','analyze']
            env=os.environ.copy();env.update(bound['CPU_environment']);row={'stage':stage,'command':command,'started_at_utc':a.base.now()};record['processes'].append(row)
            with output.with_name(output.stem+'-'+stage+'.log').open('x') as handle:
                child=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True);row['pid']=child.pid
                try:row['exit_code']=child.wait(timeout=min(195,max(1,420-(time.monotonic()-begun))))
                except BaseException:stop_child(child);row['interrupted_or_watchdog']=True;raise
            row['completed_at_utc']=a.base.now();d.p.require(row['exit_code']==0,stage+' failed; no retry')
        d.p.require(time.monotonic()-begun<=420,'overall watchdog exceeded');record['status']='complete_independently_audited'
    except BaseException as exc:
        if child is not None:stop_child(child)
        record['status']='failed_no_retry';record['error']=repr(exc);raise
    finally:
        record['wall_seconds']=time.monotonic()-begun;record['completed_at_utc']=a.base.now();a.base.write_new(final,record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=d.DEFAULT_CONFIG);parser.add_argument('--binding',type=Path,default=d.DEFAULT_BINDING)
    parser.add_argument('--mode',choices=['preflight','analyze','run'],default='preflight');args=parser.parse_args()
    if args.mode=='preflight':a.preflight(args.config,args.binding);r={'status':'bound_no_payload'}
    elif args.mode=='analyze':r=analyze(args.config,args.binding)
    else:r=run(args.config,args.binding)
    print(json.dumps({'status':r['status']}))
