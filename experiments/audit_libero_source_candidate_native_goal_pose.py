"""Independent source/CSV arithmetic audit; no analysis or environment imports."""
from __future__ import annotations
import argparse
import ast
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

ROOT=Path(__file__).resolve().parents[1]


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(item):
    p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
    raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==item['sha256'],'input hash mismatch')
    return raw


def table(item):
    require(sha(item['path'])==item['sha256'],'output table hash mismatch')
    with Path(item['path']).open(newline='') as f:rows=list(csv.DictReader(f))
    require(len(rows)==item['rows'],'output table rows mismatch')
    return rows


def compare(actual,expected):
    require(len(actual)==len(expected),'independent row count mismatch');checks=0
    for row,want in zip(actual,expected):
        require(set(row)==set(want),'unexpected/missing output columns')
        for key,value in want.items():
            got=row[key]
            if value is None:require(got=='','missing native/saved field filled')
            elif type(value) is bool:require(got==str(value),'gate/label mismatch')
            elif type(value) is int:require(got==str(value),'index/count mismatch')
            elif type(value) is float:require(math.isfinite(float(got)) and abs(float(got)-value)<=1e-12,'pose/margin mismatch')
            else:require(got==value,'identity/failed-terms mismatch')
            checks+=1
    return checks


def expected_pose(g):
    x,y,z=[g['bowl_position_world'][j]-g['plate_position_world'][j] for j in range(3)]
    xy=math.sqrt(x*x+y*y)
    records=[r for r in g['contacts'] if r['other_group']=='plate']
    nonpositive=sum(r['distance_m']<=0 for r in records)
    gates=[xy<0.03,z>=0,len(records)>0]
    missing=[name for name,gate in zip(['xy','height','contact_record'],gates) if not gate]
    dot=sum(g['bowl_rotation_world'][j][2]*g['plate_rotation_world'][j][2] for j in range(3))
    row={'state_index':g['state_index'],'sim_time':g['sim_time']}
    for name in ['bowl','plate']:
        for axis,value in zip('xyz',g[name+'_position_world']):row[name+'_'+axis]=value
    row.update(bowl_to_plate_xy=xy,bowl_to_plate_xyz=math.sqrt(x*x+y*y+z*z),
        xy_margin_to_native_radius=.03-xy,bowl_minus_plate_z=z,
        proxy_xy_gate=gates[0],proxy_height_gate=gates[1],plate_contact_records=len(records),
        plate_nonpositive_contact_records=nonpositive,plate_positive_margin_contact_records=len(records)-nonpositive,
        minimum_plate_record_distance=min((r['distance_m'] for r in records),default=None),
        proxy_contact_record_gate=gates[2],shadow_native_semantics_conjunction=all(gates),
        shadow_nonpositive_contact_conjunction=gates[0] and gates[1] and nonpositive>0,
        shadow_failed_terms='|'.join(missing) if missing else 'none',
        finger_bilateral_proxy=g['bilateral_touch_proxy'],bowl_relative_reset_height=g['bowl_height_change'],
        bowl_plate_up_axis_angle_deg=math.acos(min(1,max(-1,dot)))*180/math.pi,
        exact_native_goal_operands_available=False)
    return row


def audit(path,output):
    started=time.monotonic();output=Path(output);require(not output.exists(),'refuse existing native pose audit')
    result=json.loads(Path(path).read_text());config=json.loads(read(result['config']));binding=json.loads(read(result['binding']))
    require(binding['config_sha256']==result['config']['sha256'],'analysis binding mismatch')
    for item in binding['files'].values():read(item)
    require(config['tasks']==[0,1,3,4,7] and config['seeds']==[20261106,20261107,20261108] and
            config['arms']==['A_original','B_fomaml'] and config['global_initial_indices']==[0,1,2,3] and
            config['initial_role']=='candidate_nonregression' and config['last_saved_boundaries']==6 and
            config['new_research_calls']==result['new_policy_optimizer_environment_shadow_checkpoint_initial_HDF5_RGB_sealed_payload_calls']==0,
            'analysis role/window/budget mismatch')
    sources={k:read(v).decode() for k,v in config['native_sources'].items()}
    for item in config['source_bddls'].values():
        text=read(item).decode().lower()
        require('(and (on akita_black_bowl_1 plate_1))' in text,'source BDDL goal changed')
    state=ast.parse(sources['object_states'])
    fn=next(n for c in state.body if isinstance(c,ast.ClassDef) and c.name=='ObjectState'
            for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='check_ontop')
    ret=next(n.value for n in fn.body if isinstance(n,ast.Return))
    require(isinstance(ret,ast.BoolOp) and isinstance(ret.op,ast.And) and len(ret.values)==3,'goal conjunction structure changed')
    require(isinstance(ret.values[0].ops[0],ast.LtE) and isinstance(ret.values[2].ops[0],ast.Lt) and
            ret.values[2].comparators[0].value==.03,'native strict XY/inclusive height mismatch')
    require(result['source_semantics']['horizontal_radius_m']==.03 and
            result['source_semantics']['explicit_gripper_open_release_rotation_or_dwell_condition'] is False,
            'reported goal semantics mismatch')
    parents={k:json.loads(read(v)) for k,v in config['inputs'].items()}
    require(parents['capture']['G_decision_unchanged']=='failed_keep_H_sealed' and
            parents['capture_audit']['result_sha256']==config['inputs']['capture']['sha256'] and
            parents['alignment_audit']['result']['sha256']==config['inputs']['alignment']['sha256'], 'parent lineage mismatch')
    expected_keys={(t,s,a,i) for t in [0,1,3,4,7] for s in [20261106,20261107,20261108]
                   for a in ['A_original','B_fomaml'] for i in range(4)}
    seen=set();terminals=[];windows=[];geometry_rows=0
    for item in sorted(parents['capture']['rollout']['episode_files'],key=lambda r:(r['task'],r['seed'],r['arm'],r['global_initial_index'])):
        key=tuple(item[k] for k in ['task','seed','arm','global_initial_index'])
        require(key in expected_keys and key not in seen and item['initial_state_role']=='candidate_nonregression','source identity mismatch')
        seen.add(key)
        for field in ['public','geometry']:
            require(Path(item[field]['path']).resolve().parent==Path(config['capture_root']).resolve(),'outside captured scope')
        public,geometry=[json.loads(read(item[f])) for f in ['public','geometry']]
        require(public['identity']==geometry['identity'] and public['steps']==geometry['steps']==item['steps'],'public/geometry join mismatch')
        n=public['steps'];geometry_rows+=len(geometry['rows'])
        identity=dict(zip(['task','seed','arm','global_initial_index'],key))
        identity.update(steps=n,recorded_native_goal_at_end=public['prefix_success'])
        for k in range(max(0,n-5),n+1):
            row={**identity,'offset_from_terminal':k-n,
                 'gripper_command_to_state':None if k==0 else public['rows'][k]['action'][6],
                 **expected_pose(geometry['rows'][k])}
            windows.append(row)
        terminals.append({**row,'shadow_proxy_matches_recorded_native_goal':row['shadow_native_semantics_conjunction']==public['prefix_success']})
    require(seen==expected_keys and geometry_rows==17247 and len(windows)==720,'incomplete saved terminal cohort')
    expected={'terminals':terminals,'terminal_last6':windows,
              'proxy_native_disagreements':[r for r in terminals if not r['shadow_proxy_matches_recorded_native_goal']]}
    require(set(result['tables'])==set(expected),'output tables mismatch')
    checks=0
    for name,rows in expected.items():checks+=compare(table(result['tables'][name]),rows)
    scopes={(None,None,None)}|{(None,None,a) for a in config['arms']}|{(t,None,a) for t in config['tasks'] for a in config['arms']}|{(t,s,a) for t in config['tasks'] for s in config['seeds'] for a in config['arms']}
    require({(g['task'],g['seed'],g['arm']) for g in result['groups']}==scopes and len(result['groups'])==43,'aggregate group scope mismatch')
    for group in result['groups']:
        rows=[r for r in terminals if all(group[k] is None or group[k]==r[k] for k in ['task','seed','arm'])]
        counts={'prefixes':len(rows),'recorded_goals':sum(r['recorded_native_goal_at_end'] for r in rows),
                'shadow_conjunction_true':sum(r['shadow_native_semantics_conjunction'] for r in rows),
                'proxy_native_disagreements':sum(not r['shadow_proxy_matches_recorded_native_goal'] for r in rows),
                'all_masks':dict(Counter(r['shadow_failed_terms'] for r in rows)),
                'native_not_goal_masks':dict(Counter(r['shadow_failed_terms'] for r in rows if not r['recorded_native_goal_at_end'])),
                'native_goal_masks':dict(Counter(r['shadow_failed_terms'] for r in rows if r['recorded_native_goal_at_end']))}
        require(all(group[k]==v for k,v in counts.items()),'aggregate native goal counts mismatch');checks+=len(counts)
        for field in ['xy_margin_to_native_radius','bowl_minus_plate_z','bowl_to_plate_xyz','bowl_plate_up_axis_angle_deg']:
            v=[r[field] for r in rows];s={'mean':sum(v)/len(v),'median':statistics.median(v),'min':min(v),'max':max(v)}
            require(all(abs(group[field][k]-value)<=1e-12 for k,value in s.items()),'aggregate pose statistic mismatch');checks+=4
    record={'status':'passed_independent_native_goal_saved_pose_audit','formal_result':False,
            'verified_at_utc':datetime.now(timezone.utc).isoformat(),'result':{'path':str(path),'sha256':sha(path)},
            'independent_scalar_checks':checks,'table_rows':{k:len(v) for k,v in expected.items()},
            'source_prefixes':len(seen),'capture_JSON_files_read':240,'geometry_rows_loaded':geometry_rows,
            'new_policy_optimizer_environment_shadow_checkpoint_initial_calls':0,
            'native_goal_exact_reconstruction':False,'elapsed_seconds':time.monotonic()-started,
            'limits':'Source semantics and saved shadow operands verified; native cache/contact buffers not reconstructed.'}
    with output.open('x') as f:json.dump(record,f,ensure_ascii=False,indent=2);f.write('\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--result',type=Path,default=Path('/tmp/libero-source-candidate-native-goal-pose-20261107-v1.json'))
    p.add_argument('--output',type=Path,default=Path('/tmp/libero-source-candidate-native-goal-pose-independent-audit-20261107-v1.json'))
    a=p.parse_args();print(json.dumps(audit(a.result,a.output),ensure_ascii=False))
