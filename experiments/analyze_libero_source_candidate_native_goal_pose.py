"""Read native goal SOURCE and saved shadow poses, without importing LIBERO.

The source-derived condition is a diagnostic proxy on shadow data.  Original
native body_xpos/contact operands were not archived and cannot be reconstructed
exactly from these sidecars.  Never replace the recorded native success label.
"""
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
import re
import statistics
import time

from experiments import analyze_libero_source_candidate_close_geometry_alignment as prior

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_CONFIG=ROOT/'configs/libero-source-candidate-native-goal-pose-readonly_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-source-candidate-native-goal-pose-binding_v1.json'
TASKS,SEEDS,ARMS=prior.TASKS,prior.SEEDS,prior.ARMS
require=prior.require


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):return {'path':str(path),'sha256':sha(path)}


def read_pin(item):
    p=Path(item['path']);p=p if p.is_absolute() else ROOT/p
    raw=p.read_bytes();require(hashlib.sha256(raw).hexdigest()==item['sha256'],f'input changed: {p}')
    return raw


def method(tree,cls,name):
    found=[n for c in tree.body if isinstance(c,ast.ClassDef) and c.name==cls
           for n in c.body if isinstance(n,ast.FunctionDef) and n.name==name]
    require(len(found)==1,f'missing/duplicate native method {cls}.{name}')
    return found[0]


def return_expression(node):
    returns=[n.value for n in node.body if isinstance(n,ast.Return)]
    require(len(returns)==1,'native predicate return layout changed')
    return returns[0]


def expression_equal(node,text):
    return ast.dump(node,include_attributes=False)==ast.dump(ast.parse(text,mode='eval').body,include_attributes=False)


def sexpr(text):
    tokens=re.findall(r'\(|\)|[^\s()]+',re.sub(r';[^\n]*','',text).lower())
    roots=[];stack=[]
    for token in tokens:
        if token=='(':
            item=[]
            if stack:stack[-1].append(item)
            else:roots.append(item)
            stack.append(item)
        elif token==')':
            require(bool(stack),'unbalanced BDDL');stack.pop()
        else:
            require(bool(stack),'unwrapped BDDL token');stack[-1].append(token)
    require(not stack and len(roots)==1,'unbalanced BDDL root')
    return roots[0]


def goal_from_bddl(text):
    tree=sexpr(text)
    require(tree[0]=='define','BDDL define missing')
    problem=[r for r in tree[1:] if isinstance(r,list) and r and r[0]=='problem']
    goals=[r for r in tree[1:] if isinstance(r,list) and r and r[0]==':goal']
    require(problem==[['problem','libero_tabletop_manipulation']] and
            goals==[[':goal',['and',['on','akita_black_bowl_1','plate_1']]]], 'source task/goal changed')
    objects=[r for r in tree[1:] if isinstance(r,list) and r and r[0]==':objects']
    require(len(objects)==1 and 'plate_1' in objects[0] and 'akita_black_bowl_1' in objects[0], 'goal objects are not physical objects')
    return goals[0][1][1]


def inspect_sources(config):
    texts={k:read_pin(v).decode() for k,v in config['native_sources'].items()}
    states=ast.parse(texts['object_states']);predicates=ast.parse(texts['predicates'])
    on=return_expression(method(predicates,'On','__call__'))
    require(expression_equal(on,'arg2.check_ontop(arg1)'),'On argument order changed')
    body=return_expression(method(states,'ObjectState','check_ontop'))
    expected='(this_object_position[2] <= other_object_position[2]) and self.check_contact(other) and (np.linalg.norm(this_object_position[:2] - other_object_position[:2]) < 0.03)'
    require(expression_equal(body,expected),'ObjectState native goal expression changed')
    delegate=return_expression(method(states,'ObjectState','check_contact'))
    require(expression_equal(delegate,'self.env.check_contact(object_1, object_2)'),'object contact delegation changed')
    registry=ast.parse(texts['predicate_registry'])
    found=[n for n in ast.walk(registry) if isinstance(n,ast.Dict) and any(isinstance(k,ast.Constant) and k.value=='on' for k in n.keys)]
    require(len(found)==1,'predicate registry changed')
    on_entry=[v for k,v in zip(found[0].keys,found[0].values) if isinstance(k,ast.Constant) and k.value=='on']
    require(len(on_entry)==1 and expression_equal(on_entry[0],'On()'),'On registry changed')
    base=ast.parse(texts['robosuite_environment'])
    contact=method(base,'MujocoEnv','check_contact')
    attrs=[n.attr for n in ast.walk(contact) if isinstance(n,ast.Attribute)]
    require('contact' in attrs and 'ncon' in attrs and 'contact_geoms' in attrs and
            not any(x in attrs for x in ['dist','distance','force','efc_force']),'native contact semantics changed')
    require('self.sim.data.contact[: self.sim.data.ncon]' in ast.get_source_segment(texts['robosuite_environment'],contact),
            'native contact buffer slice changed')
    # Pin the rest of the call chain as source; never execute its imports.
    for key in ['problem','domain','wrapper','binding','adapter']:
        ast.parse(texts[key])
    goals={}
    for task,item in config['source_bddls'].items():goals[task]=goal_from_bddl(read_pin(item).decode())
    require(set(goals)=={'0','1','3','4','7'},'unexpected BDDL source scope')
    return {'goal':goals,'horizontal_radius_m':.03,'horizontal_comparison':'strict_less_than',
            'height_condition':'bowl_root_z >= plate_root_z',
            'contact_condition':'any_native_contact_record_between_object_contact_geom_groups; no dist or force threshold',
            'explicit_gripper_open_release_rotation_or_dwell_condition':False,
            'source_AST_checks_passed':True,
            'native_input_reconstruction':'unavailable: native cached body_xpos/contact not serialized; shadow differs in pipeline'}


def pose_features(g):
    bowl,plate=g['bowl_position_world'],g['plate_position_world']
    xy=math.hypot(bowl[0]-plate[0],bowl[1]-plate[1])
    z=bowl[2]-plate[2]
    plate_records=[c for c in g['contacts'] if c['other_group']=='plate']
    nonpositive=[c for c in plate_records if c['distance_m']<=0]
    require(len(nonpositive)==g['touch_counts']['plate'] and len(plate_records)-len(nonpositive)==g['positive_margin_counts']['plate'],
            'saved plate contact counts changed')
    tests={'xy':xy<.03,'height':z>=0,'contact_record':bool(plate_records)}
    failed=[name for name,value in tests.items() if not value]
    a=[g['bowl_rotation_world'][j][2] for j in range(3)]
    b=[g['plate_rotation_world'][j][2] for j in range(3)]
    cosine=max(-1.,min(1.,sum(x*y for x,y in zip(a,b))))
    return {'state_index':g['state_index'],'sim_time':g['sim_time'],
            'bowl_x':bowl[0],'bowl_y':bowl[1],'bowl_z':bowl[2],
            'plate_x':plate[0],'plate_y':plate[1],'plate_z':plate[2],
            'bowl_to_plate_xy':xy,'bowl_to_plate_xyz':math.dist(bowl,plate),
            'xy_margin_to_native_radius':.03-xy,'bowl_minus_plate_z':z,
            'proxy_xy_gate':tests['xy'],'proxy_height_gate':tests['height'],
            'plate_contact_records':len(plate_records),'plate_nonpositive_contact_records':len(nonpositive),
            'plate_positive_margin_contact_records':len(plate_records)-len(nonpositive),
            'minimum_plate_record_distance':min((c['distance_m'] for c in plate_records),default=None),
            'proxy_contact_record_gate':tests['contact_record'],
            'shadow_native_semantics_conjunction':all(tests.values()),
            'shadow_nonpositive_contact_conjunction':tests['xy'] and tests['height'] and bool(nonpositive),
            'shadow_failed_terms':'|'.join(failed) if failed else 'none',
            'finger_bilateral_proxy':g['bilateral_touch_proxy'],
            'bowl_relative_reset_height':g['bowl_height_change'],
            'bowl_plate_up_axis_angle_deg':math.degrees(math.acos(cosine)),
            'exact_native_goal_operands_available':False}


def build_tables(indexed):
    terminals=[];windows=[]
    for (task,seed,arm,index),(public,geometry) in sorted(indexed.items()):
        identity={'task':task,'seed':seed,'arm':arm,'global_initial_index':index,
                  'steps':public['steps'],'recorded_native_goal_at_end':public['prefix_success']}
        n=public['steps'];row=None
        for k in range(max(0,n-5),n+1):
            row={**identity,'offset_from_terminal':k-n,
                 'gripper_command_to_state':None if not k else public['rows'][k]['action'][6],
                 **pose_features(geometry['rows'][k])}
            windows.append(row)
        require(row is not None,'empty saved prefix')
        terminals.append({**row,'shadow_proxy_matches_recorded_native_goal':row['shadow_native_semantics_conjunction']==public['prefix_success']})
    return {'terminals':terminals,'terminal_last6':windows,
            'proxy_native_disagreements':[r for r in terminals if not r['shadow_proxy_matches_recorded_native_goal']]}


def summarize(terminals):
    groups=[]
    scopes=[(None,None,None)]+[(None,None,a) for a in ARMS]
    scopes += [(t,None,a) for t in TASKS for a in ARMS]+[(t,s,a) for t in TASKS for s in SEEDS for a in ARMS]
    for task,seed,arm in scopes:
        rows=[r for r in terminals if (task is None or task==r['task']) and (seed is None or seed==r['seed']) and (arm is None or arm==r['arm'])]
        row={'task':task,'seed':seed,'arm':arm,'prefixes':len(rows),
             'recorded_goals':sum(r['recorded_native_goal_at_end'] for r in rows),
             'shadow_conjunction_true':sum(r['shadow_native_semantics_conjunction'] for r in rows),
             'proxy_native_disagreements':sum(not r['shadow_proxy_matches_recorded_native_goal'] for r in rows),
             'all_masks':dict(Counter(r['shadow_failed_terms'] for r in rows)),
             'native_not_goal_masks':dict(Counter(r['shadow_failed_terms'] for r in rows if not r['recorded_native_goal_at_end'])),
             'native_goal_masks':dict(Counter(r['shadow_failed_terms'] for r in rows if r['recorded_native_goal_at_end']))}
        for name in ['xy_margin_to_native_radius','bowl_minus_plate_z','bowl_to_plate_xyz','bowl_plate_up_axis_angle_deg']:
            values=[r[name] for r in rows]
            row[name]={'mean':statistics.fmean(values),'median':statistics.median(values),'min':min(values),'max':max(values)}
        groups.append(row)
    return groups


def preflight(config_path,binding_path):
    config,binding=[json.loads(Path(p).read_text()) for p in [config_path,binding_path]]
    require(binding['config_sha256']==sha(config_path),'native goal analysis definition changed')
    require({'runner','independent_auditor','tests','analysis_plan','prior_alignment','saved_geometry_validator','science_metadata','package'}<=set(binding['files']),
            'incomplete native goal code binding')
    for item in binding['files'].values():read_pin(item)
    require((ROOT/binding['files']['runner']['path']).resolve()==Path(__file__).resolve(),'wrong native goal entry')
    require(config['tasks']==TASKS and config['seeds']==SEEDS and config['arms']==ARMS and
            config['global_initial_indices']==[0,1,2,3] and config['initial_role']=='candidate_nonregression' and
            config['last_saved_boundaries']==6 and config['new_research_calls']==0 and config['formal_result'] is False,
            'native goal role/window/budget changed')
    metadata={k:json.loads(read_pin(v)) for k,v in config['inputs'].items()}
    require(set(metadata)=={'capture','capture_audit','alignment','alignment_audit','alignment_completion'},'unexpected metadata')
    require(metadata['capture']['G_decision_unchanged']=='failed_keep_H_sealed' and metadata['capture']['new_training_updates']==0 and
            metadata['capture_audit']['status']=='passed_independent_geometry_capture_audit' and
            metadata['capture_audit']['result_sha256']==config['inputs']['capture']['sha256'] and
            metadata['alignment_audit']['status']=='passed_independent_readonly_close_geometry_alignment_audit' and
            metadata['alignment_audit']['result']['sha256']==config['inputs']['alignment']['sha256'] and
            metadata['alignment_completion']['G_admission_remains_failed_H_not_run_and_roles_sealed'] is True,
            'parent roles/audit/negative G decision changed')
    semantics=inspect_sources(config)
    return config,metadata,semantics


def run(config_path,binding_path,output,directory):
    started=time.monotonic();output,directory=Path(output),Path(directory)
    require(not output.exists() and not directory.exists(),'refuse existing native goal artifacts')
    config,metadata,semantics=preflight(config_path,binding_path)
    indexed,row_count=prior.read_cohort(config,{'capture':metadata['capture']})
    tables=build_tables(indexed);groups=summarize(tables['terminals'])
    directory.mkdir(parents=True,exist_ok=False);pins={}
    for name,rows in tables.items():
        path=directory/(name+'.csv')
        fields=list(tables['terminals'][0]) if not rows else list(rows[0])
        with path.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        pins[name]={**pin(path),'rows':len(rows)}
    result={'status':'complete_readonly_native_goal_shadow_pose_audit_pending_independent_check','formal_result':False,
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'config':pin(config_path),'binding':pin(binding_path),
            'source_semantics':semantics,'tables':pins,'groups':groups,
            'capture_JSON_files_read':240,'saved_geometry_rows_loaded':row_count,
            'terminal_states_analyzed':len(tables['terminals']),'terminal_window_rows':len(tables['terminal_last6']),
            'new_policy_optimizer_environment_shadow_checkpoint_initial_HDF5_RGB_sealed_payload_calls':0,
            'runtime':{'device':'CPU_stdlib_only','elapsed_seconds':time.monotonic()-started},
            'G_decision_unchanged':True,'SPT_Knowledge_Module_changes':False,
            'limits':['Native goal operands were not saved; source-semantics on shadow poses is diagnostic only.',
                      'Source 3cm is not a tuned diagnostic threshold; native contact accepts positive margin records.',
                      'Orientation/open/force/gripper disengagement are not explicit goal conjunction terms.',
                      'Last6 control boundaries cannot identify the native physics-substep cache.']}
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--binding',type=Path,default=DEFAULT_BINDING)
    p.add_argument('--output',type=Path,default=Path('/tmp/libero-source-candidate-native-goal-pose-20261107-v1.json'))
    p.add_argument('--tables',type=Path,default=Path('/tmp/libero-source-candidate-native-goal-pose-tables-20261107-v1'))
    p.add_argument('--preflight',action='store_true');args=p.parse_args()
    if args.preflight:
        preflight(args.config,args.binding);print('passed_native_goal_metadata_and_source_preflight')
    else:
        result=run(args.config,args.binding,args.output,args.tables)
        print(json.dumps({'status':result['status'],'groups':result['groups'][0],'tables':result['tables']},ensure_ascii=False))
