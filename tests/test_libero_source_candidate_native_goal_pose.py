"""Native goal source semantics and saved-pose arithmetic, with no simulator."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import analyze_libero_source_candidate_native_goal_pose as analysis
from experiments import audit_libero_source_candidate_native_goal_pose as independent


def pose(x=.02,z=.01,dist=-.001):
    contacts=[] if dist is None else [{'other_group':'plate','distance_m':dist}]
    return {'state_index':0,'sim_time':.25,'bowl_position_world':[x,0,z],'plate_position_world':[0,0,0],
            'contacts':contacts,'touch_counts':{'plate':int(dist is not None and dist<=0)},
            'positive_margin_counts':{'plate':int(dist is not None and dist>0)},
            'bowl_rotation_world':[[1,0,0],[0,1,0],[0,0,1]],'plate_rotation_world':[[1,0,0],[0,1,0],[0,0,1]],
            'bilateral_touch_proxy':False,'bowl_height_change':0.}


BDDL='(define (problem LIBERO_Tabletop_Manipulation) (:objects akita_black_bowl_1 - akita_black_bowl plate_1 - plate) (:goal (And (On akita_black_bowl_1 plate_1))))'


def test_source_bddl_has_exact_physical_object_goal():
    assert analysis.goal_from_bddl(BDDL)==['on','akita_black_bowl_1','plate_1']


@pytest.mark.parametrize('text',[BDDL.replace('plate_1))','plate_region))'),BDDL.replace('(On ','(In '),BDDL.replace('LIBERO_Tabletop','LIBERO_Floor'),BDDL[:-1]])
def test_changed_bddl_goal_or_problem_rejected(text):
    with pytest.raises(ValueError):analysis.goal_from_bddl(text)


def test_exact_three_cm_boundary_fails_and_equal_height_passes():
    at=analysis.pose_features(pose(x=.03,z=0))
    assert not at['proxy_xy_gate'] and at['proxy_height_gate']
    assert at['xy_margin_to_native_radius']==0
    inside=analysis.pose_features(pose(x=.03-1e-8,z=0))
    assert inside['shadow_native_semantics_conjunction']


def test_positive_margin_is_native_style_contact_record_not_nonpositive_proxy():
    row=analysis.pose_features(pose(dist=1e-6))
    assert row['proxy_contact_record_gate'] and row['shadow_native_semantics_conjunction']
    assert row['plate_positive_margin_contact_records']==1
    assert not row['shadow_nonpositive_contact_conjunction']


def test_horizontal_alignment_alone_does_not_satisfy_height_or_contact():
    row=analysis.pose_features(pose(x=.001,z=-.001,dist=None))
    assert row['proxy_xy_gate'] and not row['proxy_height_gate']
    assert row['shadow_failed_terms']=='height|contact_record'
    assert row['minimum_plate_record_distance'] is None


def test_finger_contact_is_not_plate_contact():
    g=pose(dist=None);g['contacts']=[{'other_group':'left','distance_m':-.01}];g['bilateral_touch_proxy']=True
    row=analysis.pose_features(g)
    assert row['finger_bilateral_proxy'] and not row['proxy_contact_record_gate']


def test_orientation_is_descriptive_not_an_explicit_goal_gate():
    g=pose();g['bowl_rotation_world']=[[1,0,0],[0,-1,0],[0,0,-1]]
    row=analysis.pose_features(g)
    assert row['bowl_plate_up_axis_angle_deg']==180 and row['shadow_native_semantics_conjunction']
    assert row['exact_native_goal_operands_available'] is False


def test_inconsistent_positive_margin_count_fails_closed():
    g=pose(dist=1e-6);g['touch_counts']['plate']=1
    with pytest.raises(ValueError,match='contact counts'):analysis.pose_features(g)


def test_recorded_native_goal_kept_when_shadow_proxy_disagrees():
    rows=[]
    for k in range(8):
        g=pose();g.update(state_index=k,sim_time=.25+.05*k);rows.append(g)
    public={'steps':7,'prefix_success':False,'rows':[{'action':None}]+[{'action':[0]*6+[1.]} for _ in range(7)]}
    data=analysis.build_tables({(7,20261107,'A_original',0):(public,{'rows':rows})})
    assert len(data['terminal_last6'])==6
    assert [r['offset_from_terminal'] for r in data['terminal_last6']]==[-5,-4,-3,-2,-1,0]
    terminal=data['terminals'][0]
    assert terminal['shadow_native_semantics_conjunction'] and not terminal['recorded_native_goal_at_end']
    assert len(data['proxy_native_disagreements'])==1


@pytest.mark.parametrize('x,z,dist',[(.02,.01,-.001),(.03,0,.001),(.04,-.01,None),(.001,.1,1e-6)])
def test_independent_operand_derivations_cover_native_condition_combinations(x,z,dist):
    g=pose(x,z,dist)
    assert analysis.pose_features(g)==independent.expected_pose(g)


def fake_sources(tmp_path):
    texts={'object_states':'''class ObjectState:
 def check_ontop(self,other):
  return (this_object_position[2] <= other_object_position[2]) and self.check_contact(other) and (np.linalg.norm(this_object_position[:2]-other_object_position[:2]) < 0.03)
 def check_contact(self,other):
  return self.env.check_contact(object_1,object_2)
''', 'predicates':'class On:\n def __call__(self,arg1,arg2):\n  return arg2.check_ontop(arg1)\n',
    'predicate_registry':"VALIDATE_PREDICATE_FN_DICT={'on':On()}\n",
    'robosuite_environment':'''class MujocoEnv:
 def check_contact(self,geoms_1,geoms_2):
  names=geoms_1.contact_geoms
  for contact in self.sim.data.contact[: self.sim.data.ncon]:
   if contact.geom1 in names:return True
  return False
'''}
    texts.update({k:'x=1\n' for k in ['problem','domain','wrapper','binding','adapter']})
    pins={}
    for k,text in texts.items():
        p=tmp_path/(k+'.py');p.write_text(text);pins[k]=analysis.pin(p)
    bddl=tmp_path/'source.bddl';bddl.write_text(BDDL)
    return {'native_sources':pins,'source_bddls':{str(t):analysis.pin(bddl) for t in [0,1,3,4,7]}}


def test_source_ast_is_read_without_importing_or_executing_native_code(tmp_path):
    config=fake_sources(tmp_path)
    semantics=analysis.inspect_sources(config)
    assert semantics['horizontal_radius_m']==.03 and semantics['source_AST_checks_passed']
    assert not semantics['explicit_gripper_open_release_rotation_or_dwell_condition']


@pytest.mark.parametrize('kind',['inclusive_xy','contact_dist_filter','wrong_On_order'])
def test_source_semantic_change_fails_closed(tmp_path,kind):
    config=fake_sources(tmp_path)
    name='object_states' if kind=='inclusive_xy' else 'robosuite_environment' if kind=='contact_dist_filter' else 'predicates'
    p=Path(config['native_sources'][name]['path']);text=p.read_text()
    if kind=='inclusive_xy':text=text.replace('< 0.03','<= 0.03')
    elif kind=='contact_dist_filter':text=text.replace('if contact.geom1','if contact.dist<=0 and contact.geom1')
    else:text=text.replace('arg2.check_ontop(arg1)','arg1.check_ontop(arg2)')
    p.write_text(text);config['native_sources'][name]=analysis.pin(p)
    with pytest.raises(ValueError):analysis.inspect_sources(config)


def test_old_result_never_overwritten_or_used_to_trigger_new_reads(tmp_path,monkeypatch):
    p=tmp_path/'old.json';p.write_text('preserve')
    monkeypatch.setattr(analysis,'preflight',lambda *a:pytest.fail('preflight called'))
    with pytest.raises(ValueError,match='existing'):analysis.run(None,None,p,tmp_path/'tables')
    assert p.read_text()=='preserve'


def test_imports_remain_standard_library_only_and_independent_audit_stays_independent():
    code="""
import builtins
original=builtins.__import__
def guard(name,*a,**kw):
 if name.split('.')[0] in {'torch','numpy','mujoco','robosuite','libero','h5py','PIL','src'}:
  raise AssertionError(name)
 return original(name,*a,**kw)
builtins.__import__=guard
from experiments import analyze_libero_source_candidate_native_goal_pose
from experiments import audit_libero_source_candidate_native_goal_pose
print('no_environment_imports')
"""
    r=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
    assert r.stdout.strip()=='no_environment_imports'
