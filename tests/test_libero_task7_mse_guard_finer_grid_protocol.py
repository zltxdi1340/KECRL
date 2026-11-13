"""Metadata and scalar grid/prefix contracts; no real archives or updates."""
import builtins
import copy
import io
import json
import math
from pathlib import Path
import pytest
from experiments import libero_task7_mse_guard_finer_grid_protocol as d


@pytest.fixture
def cfg():return d.create()


def test_only_grid_tail_changes_and_control_is_guarded12(cfg):
 old=d.source.validate()
 assert cfg['inputs']==old['inputs'] and len(cfg['inputs']['heads'])==18
 assert cfg['scope']['original_Policy_updates']==500 and cfg['scope']['future_stabilization_steps']==5
 assert cfg['optimization']['trial_caps']=={'coarse12':12,'fine20':20}
 assert cfg['optimization']['trial_index_range']=={'coarse12':[0,11],'fine20':[0,19]}
 assert cfg['optimization']['alpha_minima']=={'coarse12':1/2048,'fine20':1/524288}
 for field in ['direction','same_precision','slope','alpha_initial','backtrack_factor','armijo_c1','alpha_restart','extra_updates_max','gate','mse_monotonicity']:
  assert cfg['optimization'][field]==old['optimization'][field]
 assert cfg['historical_coarse']['optimization_arm']=='start500_mse_guarded_armijo'
 assert cfg['historical_coarse']['endpoint_conditions']==['saved500','start500_mse_guarded_armijo_final']
 assert cfg['new_head_layout']['order']==['saved500','coarse12_final','fine20_final']
 assert cfg['head_members']==old['head_members'] and len(cfg['reference_points'])==18
 assert [sum(h['cohort']==c for h in cfg['cohorts']) for c in ['zero','short','full50']]==[6,5,7]
 assert sum(h['prior_coarse_updates'] for h in cfg['cohorts'])==376
 assert cfg['decision']['trainable_ratio_max']==.99 and cfg['decision']['gradient_ratio_max']==.1


def test_updated_compute_and_unchanged_archive_budget(cfg):
 b=cfg['budget']
 assert b['main_auxiliary_updates_max']==b['audit_replay_updates_max']==1800
 assert b['trial_objective_forms_max_by_arm']=={'coarse12':10800,'fine20':18000}
 assert b['trial_objective_forms_max_per_stage']==b['trial_MSE_and_penalty_pair_evaluations_max_per_stage']==28800
 assert b['gradient_states_max_per_stage']==1836 and b['gradient_vectors_max_per_stage']==5508
 assert b['gradient_coordinates_max_per_stage']==1415556 and b['direction_coordinates_max_per_stage']==471852
 assert b['parameter_coordinate_writes_max_per_stage']==462600
 assert b['scaler_reconstructions_per_stage']==b['paired_representation_reuses_per_stage']==18
 assert b['fit_standardization_row_visits_per_stage']==50256
 assert b['trial_component3_fit_row_visits_max_per_stage']==80409600
 assert b['all_linear_forms_max_per_stage']==32472 and b['all_linear_form_row_visits_max_per_stage']==90661824
 assert b['all_linear_form_feature_multiplications_max_per_stage']==23209426944
 assert b['total_NPZ_loads']==68 and b['total_member_decodes']==448 and b['total_archive_SHA_checks']==154
 assert b['new_member_writes_primary']==b['new_member_decodes_audit']==126
 assert b['historical_coarse_JSON_projected_records_per_stage']=={'states':394,'trials':3060,'endpoints':36,'heads':18}
 assert b['output_CSV_rows_max']==sum(b['tables'].values())==44640
 assert b['input_research_CSV_passes']==b['original_Policy_CNN_environment_or_sealed_calls']==0


def test_metadata_factory_reads_no_payload_and_copies_nested_values(cfg,monkeypatch):
 sha0,open0,io0=d.p.sha,builtins.open,io.open;seen=[]
 def check(path):
  if isinstance(path,(str,Path)):assert Path(path).suffix not in ('.npz','.npy','.pt','.csv','.h5','.hdf5')
 def sha(path):check(path);seen.append(str(path));return sha0(path)
 def opened(path,*args,**kw):check(path);return open0(path,*args,**kw)
 def io_opened(path,*args,**kw):check(path);return io0(path,*args,**kw)
 monkeypatch.setattr(d.p,'sha',sha);monkeypatch.setattr(builtins,'open',opened);monkeypatch.setattr(io,'open',io_opened)
 cfg['scope']['training_seeds'].append(42);cfg['inputs']['heads'][0]['training']['updates']=999
 new=d.create()
 assert new['scope']['training_seeds']==[20261106,20261107,20261108] and new['inputs']['heads'][0]['training']['updates']==500 and seen
 assert all(not {'parameters','parameter','weights','bias',*d.parent.VECTORS}.intersection(fields) for fields in new['historical_coarse']['projections'].values())


@pytest.mark.parametrize('section,key,value',[
 ('scope','original_Policy_updates',550),('scope','future_stabilization_steps',0),
 ('scope','auxiliary_probe_scoring',1),('scope','new_loss_lambda_representation_or_data_weight_change',True),
 ('optimization','extra_updates_max',100),('optimization','trial_caps',{'coarse12':12,'fine20':24}),
 ('optimization','gate','previous_state_MSE_gate'),('optimization','direction','projected_gradient'),
 ('optimization','same_loss','new_lambda_or_phase_loss'),('optimization','alpha_restart','adaptive_from_previous'),
 ('optimization','trial_index_range',{'coarse12':[0,11],'fine20':[0,15]}),
 ('decision','trainable_ratio_max',1.),('decision','gradient_ratio_max',.5),
 ('budget','trial_objective_forms_max_per_stage',21600),('budget','main_auxiliary_updates_max',900),
 ('budget','scaler_reconstructions_per_stage',36),('budget','total_archive_SHA_checks',0),
 ('execution','technical_retries',1),('execution','float_absolute_tolerance',1e-6),
])
def test_changed_protocol_rejects_without_new_version(cfg,tmp_path,section,key,value):
 cfg[section][key]=value;p=tmp_path/'changed.json';p.write_text(json.dumps(cfg))
 with pytest.raises(ValueError):d.validate(p)


@pytest.mark.parametrize('fault',['missing','cohort','checkpoint','head_updates','probe','reference','projection'])
def test_role_lineage_and_fixed_cohorts_cannot_change(cfg,tmp_path,fault):
 if fault=='missing':cfg['inputs']['heads'].pop()
 elif fault=='cohort':cfg['cohorts'][0]['cohort']='short' if cfg['cohorts'][0]['cohort']=='full50' else 'full50'
 elif fault=='checkpoint':cfg['inputs']['heads'][0]['archive']['sha256']='0'*64
 elif fault=='head_updates':cfg['inputs']['heads'][0]['training']['updates']=550
 elif fault=='probe':cfg['inputs']['folds'][0]['fit_rows'][0]=cfg['inputs']['folds'][0]['probe_rows'][0]
 elif fault=='reference':cfg['reference_points'][0]['start500']['component3_mse']=0.
 else:cfg['historical_coarse']['projections']['endpoints'].append('parameters')
 p=tmp_path/'changed.json';p.write_text(json.dumps(cfg))
 with pytest.raises(ValueError):d.validate(p)


def test_unbound_rejects_before_research_SHA(cfg,tmp_path,monkeypatch):
 p=tmp_path/'science.json';p.write_text(json.dumps(cfg));sha0=d.p.sha
 def sha(path):
  assert Path(path).suffix in ('.py','.md','.json','.toml');return sha0(path)
 monkeypatch.setattr(d.p,'sha',sha)
 with pytest.raises(FileNotFoundError):d.require_binding(p,tmp_path/'absent.json')


@pytest.mark.parametrize('fault',['empty','status','budget','environment'])
def test_forged_binding_cannot_enable_readout(cfg,tmp_path,fault):
 config=tmp_path/'science.json';config.write_text(json.dumps(cfg))
 b={'status':'bound_before_payload','before_research_payload':True,'science_config':d.p.pin(config),
  'budget':copy.deepcopy(cfg['budget']),'CPU_environment':copy.deepcopy(cfg['execution']['environment']),
  'future_output':cfg['output'],'future_artifacts':cfg['artifacts'],'dependencies':[]}
 if fault=='status':b['status']='scientific_freeze_only'
 elif fault=='budget':b['budget']['trial_objective_forms_max_per_stage']=0
 elif fault=='environment':b['CPU_environment']['CUDA_VISIBLE_DEVICES']='0'
 p=tmp_path/'binding.json';p.write_text(json.dumps(b))
 with pytest.raises(ValueError):d.require_binding(config,p)


def test_first12_levels_and_acceptance_match_prior_guarded_rule():
 for j in range(12):
  assert d.trial_alpha('coarse12',j)==d.trial_alpha('fine20',j)==d.rules.trial_alpha(j)
  for mse in [.09,.1,math.nextafter(.1,math.inf)]:
   old=d.source.trial_decision(.02,.019,-.01,j,mse,.1,d.source.ARMS[1])
   assert d.trial_decision('coarse12',.02,.019,-.01,j,mse,.1)==old
   assert d.trial_decision('fine20',.02,.019,-.01,j,mse,.1)==old
 assert d.trial_alpha('fine20',12)==1/4096 and d.trial_alpha('fine20',19)==1/524288


@pytest.mark.parametrize('arm,index', [('coarse12',12),('fine20',20),('fine20',-1),('fine20',True),('fine20',12.),('other',0)])
def test_arm_does_not_get_unfrozen_grid_points(arm,index):
 with pytest.raises(ValueError):d.trial_alpha(arm,index)


def trial(arm,j,mse):
 return dict(trial_index=j,alpha=d.trial_alpha(arm,j),trainable_objective=.019,component3_mse=mse,
  **d.trial_decision(arm,.02,.019,-.01,j,mse,.1))


def test_coarse_exhausts_but_fine_accepts_13th_trial():
 common=[trial('coarse12',j,.11) for j in range(12)]
 assert d.validate_round('coarse12',.02,-.01,.1,common)['exhausted']
 fine=[trial('fine20',j,.11) for j in range(12)]+[trial('fine20',12,.09)]
 result=d.validate_round('fine20',.02,-.01,.1,fine)
 assert result=={'accepted_index':12,'exhausted':False,'extra_levels_tried':1}
 assert d.validate_round('fine20',.02,-.01,.1,[trial('fine20',j,.11) for j in range(20)])=={'accepted_index':None,'exhausted':True,'extra_levels_tried':8}
 with pytest.raises(ValueError):d.validate_round('fine20',.02,-.01,.1,fine[:12])
 assert not d.validate_round('fine20',.02,-.01,.1,[trial('fine20',0,.09)])['exhausted']
 with pytest.raises(ValueError):d.validate_round('fine20',.02,-.01,.1,[trial('fine20',0,.09),trial('fine20',1,.09)])


def test_mse_gate_and_armijo_bound_are_not_relaxed_in_tail():
 j=19;rhs=.02-.0001*d.trial_alpha('fine20',j)*.01
 assert d.trial_decision('fine20',.02,rhs,-.01,j,.1,.1)['accepted']
 assert not d.trial_decision('fine20',.02,math.nextafter(rhs,math.inf),-.01,j,.1,.1)['accepted']
 assert not d.trial_decision('fine20',.02,.019,-.01,j,math.nextafter(.1,math.inf),.1)['accepted']
 # The upper envelope is fixed at start; a rise from a hypothetical .08 prior point is allowed.
 assert d.trial_decision('fine20',.02,.019,-.01,j,.09,.1)['accepted']


@pytest.mark.parametrize('arm,updates,norm,slope,rejected,reason',[
 ('coarse12',3,1.,-1.,12,'line_search_exhausted'),('fine20',3,1.,-1.,12,None),
 ('fine20',3,1.,-1.,19,None),('fine20',3,1.,-1.,20,'line_search_exhausted'),
 ('fine20',50,0.,0.,0,'update_cap'),('fine20',0,0.,0.,0,'zero_gradient'),
 ('fine20',4,1.,0.,0,'non_descent_selected_direction'),
])
def test_same_priority_and_arm_specific_exhaustion(arm,updates,norm,slope,rejected,reason):
 assert d.stopping_reason(arm,updates,norm,slope,rejected)==reason


def pair(**changes):
 args=dict(coarse_updates=3,fine_updates=4,coarse_stop='line_search_exhausted',fine_stop='line_search_exhausted',
  coarse_t=.02,fine_t=.019,start_mse=.1,fine_mse=.0999,first_extended_index=12,first_extended_from_update=3,prefix_valid=True)
 args.update(changes);return args


def test_opening_needs_real_new_accepted_step_and_target_drop():
 result=d.pair_evidence(**pair())
 assert result['grid_opened'] and result['extra_updates']==1 and not result['automatic_Policy_promotion']
 # fine MSE may consume the prior coarse's hypothetical .0998 margin, while protecting .1 start.
 assert result['fine_mse_within_start']
 unchanged=d.pair_evidence(**pair(coarse_updates=0,fine_updates=0,coarse_t=.02,fine_t=.02,first_extended_index=None,first_extended_from_update=None))
 assert not unchanged['grid_opened']
 complete=d.pair_evidence(**pair(coarse_updates=50,fine_updates=50,coarse_stop='update_cap',fine_stop='update_cap',
  fine_t=.02,first_extended_index=None,first_extended_from_update=None))
 assert complete['unchanged_complete50'] and not complete['grid_opened']


@pytest.mark.parametrize('changes',[
 {'fine_updates':2},{'fine_t':.021},{'fine_mse':math.nextafter(.1,math.inf)},
 {'first_extended_index':11},{'first_extended_index':20},{'first_extended_index':True},
 {'first_extended_from_update':2},{'prefix_valid':False},{'prefix_valid':1},{'numeric_valid':False},
 {'coarse_stop':'non_descent_selected_direction'},
 {'fine_updates':3,'fine_t':.02,'first_extended_index':12},
])
def test_invalid_prefix_or_false_opening_rejected(changes):
 with pytest.raises(ValueError):d.pair_evidence(**pair(**changes))


def pair_rows(cfg,opened):
 rows=[];short_n=0
 for h in cfg['cohorts']:
  flag=h['cohort']=='short' and short_n<opened
  if h['cohort']=='short':short_n+=1
  rows.append({**{k:h[k] for k in d.IDS},'cohort':h['cohort'],'grid_opened':flag,'unchanged_complete50':h['cohort']=='full50'})
 return rows


@pytest.mark.parametrize('opened,label',[(0,'no_short_opening'),(2,'partial_short_opening'),(5,'all5_short_opened')])
def test_mechanism_counts_preserve_all18_and_fixed_cohorts(cfg,opened,label):
 rows=pair_rows(cfg,opened);summary=d.opening_summary(rows)
 assert summary['short_grid_opened_count']==opened and summary['short_mechanism_label']==label
 assert summary['zero_grid_opened_count']==0 and summary['full50_unchanged_count']==7
 zero=next(row for row in rows if row['cohort']=='zero');zero['grid_opened']=True
 assert d.opening_summary(rows)['zero_grid_opened_count']==1
 # A surprise zero-head opening is retained; unlike a full50 prefix violation it is not a technical failure.
 full=next(row for row in rows if row['cohort']=='full50');full['unchanged_complete50']=False
 with pytest.raises(ValueError):d.opening_summary(rows)
