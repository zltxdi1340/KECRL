"""Synthetic vectors/scalars and metadata only; no research directions or IO."""
import builtins
import copy
import io
import json
import math
from pathlib import Path
import pytest
from experiments import libero_task7_joint_direction_protocol as d


def vector(*coordinates):return list(coordinates)+[0.]*(257-len(coordinates))


@pytest.fixture(scope='module')
def canonical():return d.create()


@pytest.fixture
def cfg(canonical):return copy.deepcopy(canonical)


def test_fixed18_same500_same_loss_same_gate_and_grid(cfg):
 old=d.finer.validate()
 assert cfg['inputs']==old['inputs'] and len(cfg['inputs']['heads'])==18
 assert cfg['reference_points']==old['reference_points']
 for field in ('same_loss','same_precision','alpha_initial','backtrack_factor','armijo_c1','alpha_restart','extra_updates_max','gate'):
  assert cfg['optimization'][field]==old['optimization'][field]
 assert cfg['scope']['original_Policy_updates']==500 and cfg['scope']['future_stabilization_steps']==5
 assert cfg['scope']['new_loss_lambda_representation_or_data_weight_change'] is False
 assert cfg['optimization']['trial_caps']=={arm:20 for arm in d.ARMS}
 assert cfg['historical_control']['optimization_arm']=='fine20'
 assert [sum(h['cohort']==c for h in cfg['cohorts']) for c in ('zero','short','full50')]==[6,5,7]
 assert sum(h['prior_control_updates'] for h in cfg['cohorts'])==398
 assert cfg['new_head_layout']['order']==['saved500',*[a+'_final' for a in d.ARMS]]
 assert 'all18' in cfg['decision']['primary_comparison']
 assert 'grid_opening' not in cfg['decision'] and cfg['budget']['after_start_pair_prefix_required'] is False


def test_budget_includes_full20_both_arms_and_direction_algebra(cfg):
 b=cfg['budget']
 assert b['main_auxiliary_updates_max']==b['audit_replay_updates_max']==1800
 assert b['gradient_states_max_per_stage']==1836 and b['audit_autograd_calls_max']==5508
 assert b['trial_objective_forms_max_per_stage']==36000 and b['trial_objective_forms_max_by_arm']=={a:18000 for a in d.ARMS}
 assert b['trial_component3_fit_row_visits_max_per_stage']==100512000
 assert b['all_linear_forms_max_per_stage']==39672 and b['all_linear_form_row_visits_max_per_stage']==110764224
 assert b['all_linear_form_feature_multiplications_max_per_stage']==28355641344
 assert b['direction_constructions_max_per_stage']==1836 and b['direction_norm_reductions_max_per_stage']==7344
 assert b['direction_dot_reductions_max_per_stage']==3672 and b['direction_vector_coordinate_writes_max_per_stage']==1887408
 assert b['fit_standardization_row_visits_per_stage']==50256 and b['scaler_reconstructions_per_stage']==b['paired_representation_reuses_per_stage']==18
 assert (b['total_NPZ_loads'],b['total_member_decodes'],b['total_archive_SHA_checks'])==(68,448,154)
 assert b['historical_control_JSON_projected_records_per_stage']=={'states':416,'trials':3509,'endpoints':36,'heads':18}
 assert b['output_CSV_rows_max']==sum(b['tables'].values())==51840


@pytest.mark.parametrize('total,base',[(vector(1.,0.),vector(0.,1.)),(vector(2.,0.),vector(1.,0.)),
 (vector(1.,2.),vector(-.5,1.)),(vector(.1,-.3,.2),vector(-.2,.1,.4)),(vector(1.,0.),vector(-1.,1e-5))])
def test_joint_direction_matches_norm_and_descends_two_selected_gradients(total,base):
 r=d.direction(d.ARMS[1],total,base)
 assert r['status']=='joint_bisector' and r['selected_joint']
 assert math.isclose(math.hypot(*r['direction']),math.hypot(*total),rel_tol=0,abs_tol=1e-12)
 assert r['selected_T_slope']<0 and r['selected_MSE_slope']<0
 # Independent closed-form angle identities for nonzero resolved gradients.
 g,b=math.hypot(*total),math.hypot(*base);c=math.fsum(x*y for x,y in zip(total,base))/(g*b)
 assert abs(r['selected_T_slope']-(-g*g*(1+c)/r['unit_sum_norm']))<1e-10
 assert abs(r['selected_MSE_slope']/7-(-g*b*(1+c)/r['unit_sum_norm']))<1e-10


def test_control_keeps_full_negative_gradient_even_if_MSE_direction_increases():
 total,base=vector(-1.,.5),vector(1.,0.)
 r=d.direction(d.ARMS[0],total,base)
 assert r['direction']==[-x for x in total] and r['selected_MSE_slope']>0
 assert d.stopping_reason(d.ARMS[0],0,r['total_norm'],r['status'],r['selected_joint'],-1.,.5) is None


def test_orthogonal_bisector_and_same_direction():
 r=d.direction(d.ARMS[1],vector(1.,0.),vector(0.,1.))
 assert abs(r['direction'][0]+1/math.sqrt(2))<1e-15 and abs(r['direction'][1]+1/math.sqrt(2))<1e-15
 same=d.direction(d.ARMS[1],vector(3.,0.),vector(2.,0.))
 assert same['direction']==vector(-3.,0.)


@pytest.mark.parametrize('base',[vector(-1.,0.),vector(-1.,1e-13)])
def test_opposite_or_unresolved_near_opposite_has_no_fallback(base):
 r=d.direction(d.ARMS[1],vector(1.,0.),base)
 assert r['status']=='unresolved_joint_direction' and r['direction']==[0.]*257
 assert d.stopping_reason(d.ARMS[1],0,r['total_norm'],r['status'],r['selected_joint'],0.,0.)=='unresolved_joint_direction'


def test_zero_total_and_zero_base_are_distinct():
 total_zero=d.direction(d.ARMS[1],vector(),vector(1.))
 assert total_zero['status']=='zero_total_gradient'
 assert d.stopping_reason(d.ARMS[1],0,0.,total_zero['status'],False,0.,0.)=='zero_gradient'
 base_zero=d.direction(d.ARMS[1],vector(1.,2.),vector())
 assert base_zero['status']=='base_zero_use_total' and base_zero['direction']==vector(-1.,-2.)
 assert base_zero['selected_MSE_slope']==0 and base_zero['selected_T_slope']<0
 assert d.stopping_reason(d.ARMS[1],0,base_zero['total_norm'],base_zero['status'],True,-1.,0.) is None


@pytest.mark.parametrize('fault',['shape','nan','infinity','bool','arm','overflow'])
def test_invalid_direction_inputs_fail_closed(fault):
 t,b=vector(1.),vector(2.);arm=d.ARMS[1]
 if fault=='shape':b.pop()
 elif fault=='nan':t[0]=float('nan')
 elif fault=='infinity':b[0]=float('inf')
 elif fault=='bool':t[0]=True
 elif fault=='arm':arm='other'
 else:t[0]=b[0]=1e308
 with pytest.raises((ValueError,OverflowError)):d.direction(arm,t,b)


def test_candidate_exact_slope_gate_handles_kink_counterexamples():
 # Selected subgradients alone do not license a trial at clip/hinge boundaries.
 args=(d.ARMS[1],0,1.,'joint_bisector',True)
 assert d.stopping_reason(*args,-.1,.01)=='exact_MSE_direction_increase'
 assert d.stopping_reason(*args,.01,-.1)=='non_descent_selected_direction'
 assert d.stopping_reason(*args,-.1,-.1) is None
 assert d.stopping_reason(*args,-.1,-.1,20)=='line_search_exhausted'
 assert d.stopping_reason(d.ARMS[1],50,1.,'unresolved_joint_direction',False,0.,0.)=='update_cap'


@pytest.mark.parametrize('arm',d.ARMS)
def test_same20_first_acceptance_matches_old_fine20(arm):
 for j in range(20):
  assert d.trial_alpha(arm,j)==2.**(-j)
  for mse in (.09,.1,math.nextafter(.1,math.inf)):
   assert d.trial_decision(arm,.02,.019,-.01,j,mse,.1)==d.finer.trial_decision('fine20',.02,.019,-.01,j,mse,.1)


@pytest.mark.parametrize('index',[20,-1,True,1.])
def test_no_unfrozen_grid_point(index):
 with pytest.raises(ValueError):d.trial_alpha(d.ARMS[1],index)


def test_stepwise_MSE_increase_is_valid_acceptance_and_failed_scientific_cue():
 for arm in d.ARMS:
  assert d.trial_decision(arm,.02,.019,-.01,0,.095,.1)['accepted'] # may exceed prior M=.08
 result=d.paired_evidence(.02,.1,.018,.09,.017,.085,2,2,[-.02,.005])
 assert result['candidate_stepwise_MSE_increases']==1 and result['candidate_progress_from500']
 assert not result['candidate_path_joint_cue'] and not result['paired_joint_cue']


def test_joint_path_and_paired_endpoint_require_separate_references():
 good=d.paired_evidence(.02,.1,.018,.09,.017,.085,2,2,[-.01,-.005])
 assert good['candidate_path_joint_cue'] and good['paired_joint_cue']
 tradeoff=d.paired_evidence(.02,.1,.018,.09,.019,.085,2,2,[-.01,-.005])
 assert tradeoff['candidate_path_joint_cue'] and not tradeoff['paired_joint_cue']
 worseM=d.paired_evidence(.02,.1,.018,.09,.017,.095,2,2,[-.003,-.002])
 assert worseM['candidate_path_joint_cue'] and not worseM['paired_joint_cue']
 zero=d.paired_evidence(.02,.1,.018,.09,.02,.1,2,0,[])
 assert not zero['candidate_path_joint_cue'] and not zero['paired_joint_cue']


@pytest.mark.parametrize('fault',['envelope','zero_changed','sum','count','T_increase'])
def test_invalid_endpoints_and_increment_ledger_rejected(fault):
 args=[.02,.1,.018,.09,.017,.085,2,2,[-.01,-.005]]
 if fault=='envelope':args[5]=.11
 elif fault=='zero_changed':args[7]=0;args[8]=[]
 elif fault=='sum':args[8]=[-.01,-.01]
 elif fault=='count':args[7]=1
 else:args[4]=.021
 with pytest.raises(ValueError):d.paired_evidence(*args)


def test_all18_source_conjunction_is_not_best_seed_or_best_five(cfg):
 rows=[{**{k:h[k] for k in d.IDS},'paired_joint_cue':True} for h in cfg['cohorts']]
 assert d.source_evidence(rows)['population_direction_fit_cue']
 rows[0]['paired_joint_cue']=False
 r=d.source_evidence(rows);assert not r['population_direction_fit_cue'] and not r['source_all9_cues'][rows[0]['arm']]
 with pytest.raises(ValueError):d.source_evidence(rows[:-1])


def test_factory_never_reads_research_archive_or_evaluates_real_direction(cfg,monkeypatch):
 sha0,open0,io0=d.p.sha,builtins.open,io.open
 def check(path):
  if isinstance(path,(str,Path)):assert Path(path).suffix not in ('.npz','.npy','.pt','.csv','.h5','.hdf5')
 def sha(path):check(path);return sha0(path)
 def opened(path,*args,**kw):check(path);return open0(path,*args,**kw)
 def io_opened(path,*args,**kw):check(path);return io0(path,*args,**kw)
 def no_direction(*args,**kw):raise AssertionError('real directions must not run during freeze')
 monkeypatch.setattr(d.p,'sha',sha);monkeypatch.setattr(builtins,'open',opened);monkeypatch.setattr(io,'open',io_opened);monkeypatch.setattr(d,'direction',no_direction)
 cfg['inputs']['heads'][0]['training']['updates']=550;cfg['optimization']['direction_definition']['near_opposite']='fallback'
 new=d.create()
 assert new['inputs']['heads'][0]['training']['updates']==500 and 'no_fallback' in new['optimization']['direction_definition']['near_opposite']
 assert all(not {'parameter','parameters','weights','bias',*d.parent.VECTORS}.intersection(fields) for fields in new['historical_control']['projections'].values())


@pytest.mark.parametrize('section,key,value',[
 ('scope','original_Policy_updates',550),('scope','future_stabilization_steps',0),('scope','auxiliary_probe_scoring',1),
 ('optimization','direction','new_loss'),('optimization','gate','previous_MSE'),('optimization','same_loss','lambda0'),
 ('optimization','angle_resolution',1e-6),('optimization','extra_updates_max',100),
 ('optimization','trial_caps',{a:24 for a in d.ARMS}),('optimization','mse_monotonicity','new_per_step_acceptance_gate'),
 ('decision','trainable_ratio_max',1.),('decision','gradient_ratio_max',.5),
 ('budget','trial_objective_forms_max_per_stage',28800),('budget','direction_vector_coordinate_writes_max_per_stage',0),
 ('budget','scaler_reconstructions_per_stage',36),('budget','total_archive_SHA_checks',0),('execution','technical_retries',1),
])
def test_protocol_changes_need_new_version(cfg,tmp_path,section,key,value):
 cfg[section][key]=value;path=tmp_path/'changed.json';path.write_text(json.dumps(cfg))
 with pytest.raises(ValueError):d.validate(path)


@pytest.mark.parametrize('fault',['head','fit_role','cohort','stop','source_hash','projection'])
def test_data_role_and_historical_control_mutations_fail(cfg,tmp_path,fault):
 if fault=='head':cfg['inputs']['heads'].pop()
 elif fault=='fit_role':cfg['inputs']['folds'][0]['fit_rows'][0]=cfg['inputs']['folds'][0]['probe_rows'][0]
 elif fault=='cohort':cfg['cohorts'][0]['cohort']='short' if cfg['cohorts'][0]['cohort']!='short' else 'zero'
 elif fault=='stop':cfg['cohorts'][0]['prior_control_updates']+=1
 elif fault=='source_hash':cfg['inputs']['heads'][0]['archive']['sha256']='0'*64
 else:cfg['historical_control']['projections']['endpoints'].append('parameters')
 path=tmp_path/'changed.json';path.write_text(json.dumps(cfg))
 with pytest.raises(ValueError):d.validate(path)


def test_absent_binding_rejects_without_archive_SHA(cfg,tmp_path,monkeypatch):
 path=tmp_path/'science.json';path.write_text(json.dumps(cfg));sha0=d.p.sha
 def sha(path):
  assert Path(path).suffix in ('.py','.md','.json','.toml');return sha0(path)
 monkeypatch.setattr(d.p,'sha',sha)
 with pytest.raises(FileNotFoundError):d.require_binding(path,tmp_path/'absent.json')


@pytest.mark.parametrize('fault',['empty','budget','environment','status'])
def test_forged_binding_cannot_authorize_execution(cfg,tmp_path,fault):
 path=tmp_path/'science.json';path.write_text(json.dumps(cfg))
 b=dict(status='bound_before_payload',before_research_payload=True,science_config=d.p.pin(path),budget=copy.deepcopy(cfg['budget']),
  CPU_environment=copy.deepcopy(cfg['execution']['environment']),future_output=cfg['output'],future_artifacts=cfg['artifacts'],dependencies=[])
 if fault=='budget':b['budget']['main_auxiliary_updates_max']=9999
 elif fault=='environment':b['CPU_environment']['CUDA_VISIBLE_DEVICES']='0'
 elif fault=='status':b['status']='scientific_freeze_only'
 bound=tmp_path/'binding.json';bound.write_text(json.dumps(b))
 with pytest.raises(ValueError):d.require_binding(path,bound)


def test_historical_unbound_APIs_preserve_existing_completed_artifacts(tmp_path,monkeypatch):
 """Historical test asserts global absence after completion; preserve its hash.

 Check the actual gate and unchanged artifacts with the completed run present.
 No CSV/array contents or their SHA enter this regression check.
 """
 from experiments import libero_task7_fine_stop_direction_review_access as access
 from experiments import analyze_libero_task7_fine_stop_direction_review as analysis
 from experiments import audit_libero_task7_fine_stop_direction_review as audit
 cfg=d.parent.create()
 def snapshot():
  roots=[Path(cfg['output']),Path(cfg['artifacts'])];paths=[]
  for root in roots:
   if root.exists():paths.extend([root,*root.rglob('*')] if root.is_dir() else [root])
  return {str(path):(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns) for path in paths}
 before=snapshot();science=tmp_path/'science.json';science.write_text(json.dumps(cfg))
 sp=d.p.pin(science);bp={'path':str(tmp_path/'absent.json'),'sha256':'0'*64}
 def no_payload(*args,**kw):raise AssertionError('unbound API must not access payload')
 sha0=d.p.sha
 def sha(path):
  assert Path(path).suffix not in ('.npz','.npy','.pt','.csv','.h5','.hdf5');return sha0(path)
 monkeypatch.setattr(access.Guard,'project',no_payload);monkeypatch.setattr(d.p,'sha',sha)
 for fn in (analysis.execute_payload,audit.audit_payload):
  with pytest.raises(FileNotFoundError):fn(cfg,sp,bp)
 assert snapshot()==before
