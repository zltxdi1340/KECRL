"""Independent standard-library audit; never import the analysis or simulator."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]
FIELDS=['eef_to_bowl_xyz','eef_to_bowl_xy','eef_minus_bowl_signed_z',
        'bowl_height_change','bowl_to_plate_xyz','bowl_to_plate_xy','bilateral_touch_proxy']
ARMS=['A_original','B_fomaml']


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(item):
    p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
    raw=p.read_bytes()
    require(hashlib.sha256(raw).hexdigest()==item['sha256'],f'hash mismatch: {p}')
    return json.loads(raw)


def read_table(item):
    require(sha(item['path'])==item['sha256'],'table checksum differs')
    with Path(item['path']).open(newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==item['rows'],'table row count differs')
    return rows


def cell(actual,expected):
    if expected is None:require(actual=='','missing/censored value filled')
    elif type(expected) is bool:require(actual==str(expected),'boolean differs')
    elif type(expected) is int:require(actual==str(expected),'index/count differs')
    elif type(expected) is float:
        number=float(actual)
        require(math.isfinite(number) and abs(number-expected)<=1e-12,'derived scalar differs')
    else:require(actual==str(expected),'identity/category differs')


def compare_rows(actual,expected):
    require(len(actual)==len(expected),'independent row count differs')
    checks=0
    for a,e in zip(actual,expected):
        require(set(a)==set(e),'unexpected/missing analysis field')
        for key,value in e.items():cell(a[key],value);checks+=1
    return checks


def scan_runs(public):
    runs=[]
    for k in range(1,public['steps']+1):
        value=public['rows'][k]['action'][6]
        s=1 if value>0 else -1 if value<0 else 0
        if not runs or runs[-1]['sign']!=s:
            runs.append({'run_ordinal':len(runs)+1,'sign':s,'start':k,'end':k,'observed_length':1,'right_censored':False})
        else:
            runs[-1]['end']=k;runs[-1]['observed_length']+=1
    if runs:runs[-1]['right_censored']=True
    return runs


def expected_close(public,geometry,run,identity):
    g=geometry['rows'];lo=run['start']-1;hi=run['end']
    touches=[k for k in range(lo,hi+1) if g[k]['bilateral_touch_proxy']]
    first=touches[0] if touches else None
    first_end=first
    if first is not None:
        while first_end<hi and g[first_end+1]['bilateral_touch_proxy']:first_end+=1
    blocks=sum(k==lo or not g[k-1]['bilateral_touch_proxy'] for k in touches)
    following=None if hi==public['steps'] else (1 if public['rows'][hi+1]['action'][6]>0 else -1 if public['rows'][hi+1]['action'][6]<0 else 0)
    opening=hi+1 if following==-1 else None
    row={**identity,'prefix_success':public['prefix_success'],**run,'pre_state':lo,'post_state':hi,
         'duration_observed_seconds':g[hi]['sim_time']-g[lo]['sim_time'],
         'complete_one_step_pulse':run['observed_length']==1 and hi<public['steps'],
         'touch_at_pre':g[lo]['bilateral_touch_proxy'],'touch_at_end':g[hi]['bilateral_touch_proxy'],
         'touch_blocks_in_close':blocks,'touch_boundary_count_in_close':len(touches),
         'first_touch_state_in_close':first,
         'first_touch_seconds_from_close_onset':None if first is None else g[first]['sim_time']-g[lo]['sim_time'],
         'first_touch_block_end_state':first_end,
         'first_touch_block_ends_inside_close':None if first is None else first_end<hi,
         'following_command_sign':following,'immediate_open_action':opening,
         'first_touch_to_close_end_seconds':None if first is None else g[hi]['sim_time']-g[first]['sim_time']}
    for field in FIELDS[:-1]:
        series=[g[k][field] for k in range(lo,hi+1)]
        row.update({'pre_'+field:series[0],'end_'+field:series[-1],
                    'delta_'+field:series[-1]-series[0],'min_'+field:min(series),'max_'+field:max(series)})
    for field in ['bowl_height_change','bowl_to_plate_xyz','bowl_to_plate_xy']:
        series=[] if first is None else [g[k][field] for k in range(first,hi+1)]
        row['touch_'+field]=series[0] if series else None
        row['touch_to_end_delta_'+field]=series[-1]-series[0] if series else None
        row['touch_to_end_min_delta_'+field]=min(series)-series[0] if series else None
        row['touch_to_end_max_delta_'+field]=max(series)-series[0] if series else None
    for field in FIELDS:row['immediate_open_post_'+field]=None if opening is None else g[opening][field]
    return row


def expected_episode(public,geometry,closes,identity):
    g=geometry['rows'];touches=[k for k in range(public['steps']+1) if g[k]['bilateral_touch_proxy']]
    first=touches[0] if touches else None
    multi=next((r for r in closes if r['observed_length']>=2),None)
    row={**identity,'steps':public['steps'],'prefix_success':public['prefix_success'],
         'capture_stop_reason':public['capture_stop_reason'],'close_runs':len(closes),
         'multi_close_runs':sum(r['observed_length']>=2 for r in closes),
         'close_runs_with_touch':sum(r['touch_boundary_count_in_close']>0 for r in closes),
         'close_runs_without_touch':sum(r['touch_boundary_count_in_close']==0 for r in closes),
         'first_touch_state':first,'last_touch_state':touches[-1] if touches else None,
         'touch_boundary_count':len(touches),
         'touch_blocks_in_prefix':sum(k==0 or not g[k-1]['bilateral_touch_proxy'] for k in touches),
         'touch_at_terminal':g[-1]['bilateral_touch_proxy'],
         'first_multi_pre_eef_distance':None if multi is None else multi['pre_eef_to_bowl_xyz'],
         'first_multi_pre_plate_distance_xy':None if multi is None else multi['pre_bowl_to_plate_xy'],
         'first_close_step':closes[0]['start'] if closes else None,
         'first_multi_close_step':None if multi is None else multi['start'],
         'complete_first_close_pulse':bool(closes and closes[0]['complete_one_step_pulse'])}
    for field in ['bowl_height_change','bowl_to_plate_xyz','bowl_to_plate_xy','eef_to_bowl_xyz']:
        series=[v[field] for v in g]
        row['reset_'+field]=series[0];row['terminal_'+field]=series[-1]
        row['terminal_minus_reset_'+field]=series[-1]-series[0]
        row['minimum_'+field]=min(series);row['maximum_'+field]=max(series)
        row['at_first_touch_'+field]=None if first is None else series[first]
        row['first_touch_to_terminal_delta_'+field]=None if first is None else series[-1]-series[first]
        row['first_touch_to_terminal_min_delta_'+field]=None if first is None else min(series[first:])-series[first]
        row['first_touch_to_terminal_max_delta_'+field]=None if first is None else max(series[first:])-series[first]
    return row


def audit(output_path,audit_output):
    started=time.monotonic();audit_output=Path(audit_output)
    require(not audit_output.exists(),'refuse existing independent audit')
    result=json.loads(Path(output_path).read_text())
    config,binding=read_json(result['config']),read_json(result['binding'])
    require(binding['config_sha256']==result['config']['sha256'],'config lineage differs')
    for item in binding['files'].values():
        p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
        require(sha(p)==item['sha256'],'analysis code binding changed')
    require(result['inputs']==config['inputs'] and result['formal_result'] is False and
            result['status']=='complete_readonly_close_geometry_alignment_pending_independent_audit','wrong analysis lineage')
    parents={k:read_json(v) for k,v in config['inputs'].items() if k!='prior_runs'}
    require(parents['audit']['status']=='passed_independent_geometry_capture_audit' and
            parents['audit']['result_sha256']==config['inputs']['capture']['sha256'] and
            parents['completion']['independent_audit']['sha256']==config['inputs']['audit']['sha256'], 'parent audit differs')
    require(config['tasks']==[0,1,3,4,7] and config['seeds']==[20261106,20261107,20261108] and
            config['arms']==ARMS and config['global_initial_indices']==[0,1,2,3] and
            config['candidate_role']=='candidate_nonregression' and config['neighborhood_half_window']==5,
            'sealed/out-of-scope role')
    require(result['resource_new_calls']==config['new_research_calls'] and all(v==0 for v in result['resource_new_calls'].values()),'new research calls')
    expected_keys={(t,s,a,i) for t in [0,1,3,4,7] for s in [20261106,20261107,20261108] for a in ARMS for i in range(4)}
    indexed={};closes=[];episodes=[];all_runs=[];row_count=0
    for item in sorted(parents['capture']['rollout']['episode_files'],key=lambda r:(r['task'],r['seed'],r['arm'],r['global_initial_index'])):
        key=tuple(item[k] for k in ['task','seed','arm','global_initial_index'])
        require(key in expected_keys and key not in indexed,'pair/prefix identity differs')
        require(item['initial_state_role']=='candidate_nonregression' and item['checkpoint_update']==500,'wrong role/endpoint')
        docs=[]
        for field in ['public','geometry']:
            require(Path(item[field]['path']).resolve().parent==Path(config['capture_root']).resolve(),'outside captured paths')
            docs.append(read_json(item[field]))
        public,geo=docs;indexed[key]=docs
        require(public['identity']==geo['identity'] and public['steps']==geo['steps']==item['steps'],'public/geometry join differs')
        row_count+=public['steps']+1
        identity=dict(zip(['task','seed','arm','global_initial_index'],key))
        runs=scan_runs(public);cr=[expected_close(public,geo,r,identity) for r in runs if r['sign']==1]
        closes+=cr;episodes.append(expected_episode(public,geo,cr,identity))
        all_runs += [{**identity,**r} for r in runs]
    require(set(indexed)==expected_keys and len(closes)==170 and row_count==17247,'incomplete saved cohort')
    require(sha(config['inputs']['prior_runs']['path'])==config['inputs']['prior_runs']['sha256'],'prior command table differs')
    with Path(config['inputs']['prior_runs']['path']).open(newline='') as f:prior=list(csv.DictReader(f))
    fields=['task','seed','arm','global_initial_index','run_ordinal','sign','start','end','observed_length','right_censored']
    sorted_prior=sorted(prior,key=lambda r:(int(r['task']),int(r['seed']),r['arm'],int(r['global_initial_index']),int(r['run_ordinal'])))
    checks=compare_rows([{k:r[k] for k in fields} for r in sorted_prior],[{k:r[k] for k in fields} for r in all_runs])
    tables={k:read_table(v) for k,v in result['tables'].items()}
    require(set(tables)=={'close_runs','episodes','pairs','aligned_states','event_windows'},'table schema differs')
    checks+=compare_rows(tables['close_runs'],closes)+compare_rows(tables['episodes'],episodes)

    # Verify every state/window against the same absolute raw boundary, including
    # explicit empty cells beyond either terminal; no analysis helper is used.
    expected_frames=[];expected_pairs=[]
    ep_lookup={(e['task'],e['seed'],e['arm'],e['global_initial_index']):e for e in episodes}
    for t,s,_,i in sorted(k for k in indexed if k[2]==ARMS[0]):
        docs=[indexed[(t,s,a,i)] for a in ARMS];lengths=[p['steps'] for p,g in docs]
        common=min(lengths)
        for state in range(max(lengths)+1):
            row={'task':t,'seed':s,'global_initial_index':i,'state_index':state,'jointly_available':state<=common}
            for a,(p,g) in zip(('A','B'),docs):
                available=state<=p['steps'];row[a+'_available']=available
                row[a+'_sim_time']=g['rows'][state]['sim_time'] if available else None
                row[a+'_command_to_state']=p['rows'][state]['action'][6] if available and state else None
                for f in FIELDS:row[a+'_'+f]=g['rows'][state][f] if available else None
            for f in FIELDS[:-1]:row['B_minus_A_'+f]=None if state>common else row['B_'+f]-row['A_'+f]
            expected_frames.append(row)
        ea,eb=[ep_lookup[(t,s,a,i)] for a in ARMS]
        row={'task':t,'seed':s,'global_initial_index':i,'common_last_state':common,'common_state_count':common+1,
             'A_prefix_success':ea['prefix_success'],'B_prefix_success':eb['prefix_success'],
             'outcome':'both' if ea['prefix_success'] and eb['prefix_success'] else 'A_only' if ea['prefix_success'] else 'B_only' if eb['prefix_success'] else 'neither'}
        for a,arm,(p,g) in zip(('A','B'),ARMS,docs):
            ep=ep_lookup[(t,s,arm,i)];series=g['rows'][:common+1]
            row[a+'_first_multi_close_step']=ep['first_multi_close_step']
            row[a+'_first_multi_pre_distance']=ep['first_multi_pre_eef_distance']
            row[a+'_common_max_height']=max(v['bowl_height_change'] for v in series)
            row[a+'_common_terminal_plate_xy']=series[-1]['bowl_to_plate_xy']
            row[a+'_common_touch_states']=sum(v['bilateral_touch_proxy'] for v in series)
        row['common_B_minus_A_max_height']=row['B_common_max_height']-row['A_common_max_height']
        row['common_B_minus_A_terminal_plate_xy']=row['B_common_terminal_plate_xy']-row['A_common_terminal_plate_xy']
        expected_pairs.append(row)
    checks+=compare_rows(tables['aligned_states'],expected_frames)+compare_rows(tables['pairs'],expected_pairs)
    anchors={}
    for c in closes:
        case=c['task'],c['seed'],c['global_initial_index']
        for label,k in [('close_onset_pre',c['pre_state']),('close_end_post',c['end']),('first_touch_in_close',c['first_touch_state_in_close']),('immediate_open_post',c['immediate_open_action'])]:
            if k is not None:anchors.setdefault((*case,k),[]).append(f"{c['arm']}:run{c['run_ordinal']}:{label}")
    expected_windows=[]
    for (t,s,i,k),labels in sorted(anchors.items()):
        for offset in range(-5,6):
            state=k+offset;row={'task':t,'seed':s,'global_initial_index':i,'anchor_state':k,
                'anchor_aliases':'|'.join(sorted(labels)),'offset':offset,'absolute_state_index':state,'reset_side_truncated':state<0}
            for a,arm in zip(('A','B'),ARMS):
                p,g=indexed[(t,s,arm,i)];available=0<=state<=p['steps']
                row[a+'_available']=available;row[a+'_terminal_truncated']=state>p['steps']
                row[a+'_command_to_state']=p['rows'][state]['action'][6] if available and state else None
                for f in FIELDS:row[a+'_'+f]=g['rows'][state][f] if available else None
            expected_windows.append(row)
    checks+=compare_rows(tables['event_windows'],expected_windows)

    # JSON aggregates are checked independently with their observed denominators.
    mapping={'multi_close_pre_distance':('e','first_multi_pre_eef_distance'),
             'maximum_height_from_reset':('e','maximum_bowl_height_change'),
             'terminal_plate_xy_from_reset':('e','terminal_minus_reset_bowl_to_plate_xy'),
             'touch_to_terminal_height_delta':('e','first_touch_to_terminal_delta_bowl_height_change'),
             'touch_to_terminal_max_height_delta':('e','first_touch_to_terminal_max_delta_bowl_height_change'),
             'touch_to_terminal_plate_xy_delta':('e','first_touch_to_terminal_delta_bowl_to_plate_xy'),
             'close_touch_to_end_height_delta':('c','touch_to_end_delta_bowl_height_change'),
             'close_touch_to_end_plate_xy_delta':('c','touch_to_end_delta_bowl_to_plate_xy')}
    require(len(result['groups'])==43,'missing aggregate groups')
    for group in result['groups']:
        match=lambda r:all(group[q] is None or group[q]==r[q] for q in ['task','seed','arm'])
        er,cr=list(filter(match,episodes)),list(filter(match,closes))
        counts={'prefixes':len(er),'prefix_native_goals':sum(r['prefix_success'] for r in er),
                'no_close_prefixes':sum(r['close_runs']==0 for r in er),'touch_prefixes':sum(r['first_touch_state'] is not None for r in er),
                'close_runs':len(cr),'close_runs_with_touch':sum(r['first_touch_state_in_close'] is not None for r in cr),
                'close_runs_touch_block_ends_inside':sum(r['first_touch_block_ends_inside_close'] is True for r in cr),
                'close_runs_touch_at_end':sum(r['touch_at_end'] for r in cr),'close_runs_right_censored':sum(r['right_censored'] for r in cr)}
        require(all(group[k]==v for k,v in counts.items()),'aggregate counts differ');checks+=len(counts)
        for label,(kind,field) in mapping.items():
            values=[r[field] for r in (er if kind=='e' else cr) if r[field] is not None]
            expected={'observed':len(values),'mean':sum(values)/len(values) if values else None,
                      'median':statistics.median(values) if values else None,'minimum':min(values) if values else None,'maximum':max(values) if values else None}
            require(set(group[label])==set(expected),'aggregate schema differs')
            for key,value in expected.items():
                actual=group[label][key]
                require(actual==value if value is None or isinstance(value,int) else abs(actual-value)<1e-12,'aggregate statistic differs');checks+=1
    require(result['prefix_outcomes']==dict(Counter(r['outcome'] for r in expected_pairs)),'prefix four-cell differs')
    record={'status':'passed_independent_readonly_close_geometry_alignment_audit','formal_result':False,
            'verified_at_utc':datetime.now(timezone.utc).isoformat(),'result':{'path':str(output_path),'sha256':sha(output_path)},
            'config_sha256':result['config']['sha256'],'binding_sha256':result['binding']['sha256'],
            'table_rows':{k:len(v) for k,v in tables.items()},'independent_scalar_checks':checks,
            'source_prefixes':len(indexed),'paired_scenes':len(expected_pairs),'close_runs':len(closes),
            'capture_json_files_read':240,'geometry_state_rows_read':row_count,'new_policy_optimizer_environment_checkpoint_initial_shadow_calls':0,
            'elapsed_seconds':time.monotonic()-started,'G_decision_unchanged':True,
            'limits':'Independent saved-field arithmetic audit; no physics replay or force/grasp validation.'}
    with audit_output.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,default=Path('/tmp/libero-source-candidate-close-geometry-alignment-20261107-v1.json'))
    p.add_argument('--output',type=Path,default=Path('/tmp/libero-source-candidate-close-geometry-alignment-audit-20261107-v1.json'))
    a=p.parse_args();print(json.dumps(audit(a.result,a.output),ensure_ascii=False))


if __name__=='__main__':main()
