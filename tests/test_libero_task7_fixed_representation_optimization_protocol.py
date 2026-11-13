"""Metadata, finite-step rules and synthetic scalars; no research array IO."""
import copy
import json
import math
from pathlib import Path
import pytest
from experiments import libero_task7_fixed_representation_optimization_protocol as d


@pytest.fixture
def cfg(): return d.create()


def test_finite_budget_includes_trials_updates_replay_and_preprocessing(cfg):
    b = cfg['budget']
    assert b['main_auxiliary_updates_max'] == b['audit_replay_updates_max'] == 1800
    assert b['parameter_coordinate_writes_max_per_stage'] == 462600
    assert b['gradient_states_max_per_stage'] == 36*51 == 1836
    assert b['audit_autograd_calls_max'] == 1836*3 == 5508
    assert b['gradient_coordinates_max_per_stage'] == 5508*257 == 1415556
    assert b['trial_objective_forms_max_per_stage'] == 36*50*12 == 21600
    assert b['all_linear_forms_max_per_stage'] == 25272
    assert b['all_linear_form_feature_multiplications_max_per_stage'] == 18063212544
    assert b['scaler_reconstructions_per_stage'] == b['paired_reuses_per_stage'] == 18
    assert b['fit_standardization_feature_coordinates_per_stage'] == 50256*256
    assert b['total_NPZ_loads'] == 43+43+36 == 122
    assert b['total_member_decodes'] == 305+305+36*7 == 862
    assert b['total_archive_SHA_checks'] == 86+86+36+72 == 280
    assert b['output_CSV_rows_max'] == sum(b['tables'].values()) == 42066
    assert b['original_Policy_CNN_environment_or_sealed_calls'] == 0


def test_fixed_inputs_own_loss_and_500_start(cfg):
    parent = d.parent.validate()
    assert cfg['inputs'] == parent['inputs']
    assert len(cfg['reference_points']) == 36
    assert cfg['optimization']['extra_updates_max'] == 50
    assert cfg['scope']['original_Policy_and_source_auxiliary_updates'] == 500
    assert cfg['scope']['auxiliary_probe_scoring'] == 0
    assert 'Adam_state' in cfg['optimization']['direction']
    assert cfg['head_members'] == parent['head_members']
    assert not {'predictions','probe_rows','sample_rows','final_adam_m_weight'} & set(cfg['head_members'])
    assert cfg['new_head_members'] == ['fit_rows','fit_demo_indices','scaler_mean','scaler_scale','additional_updates','weights','bias']


def test_factory_metadata_only_and_nested_objects_independent(cfg,monkeypatch):
    original = d.p.sha; seen = []
    def sha(path):
        assert Path(path).suffix in ('.py','.md','.json','.toml')
        seen.append(str(path)); return original(path)
    monkeypatch.setattr(d.p,'sha',sha)
    cfg['scope']['training_seeds'].append(42)
    cfg['inputs']['heads'][0]['training']['updates'] = 999
    new = d.create()
    assert new['scope']['training_seeds'] == [20261106,20261107,20261108]
    assert new['inputs']['heads'][0]['training']['updates'] == 500 and seen


@pytest.mark.parametrize('section,key,value',[
    ('scope','original_Policy_and_source_auxiliary_updates',550),
    ('scope','future_stabilization_steps',0), ('scope','auxiliary_probe_scoring',1),
    ('scope','new_loss_lambda_representation_or_data_weight_change',True),
    ('optimization','extra_updates_max',100), ('optimization','trials_per_round_max',13),
    ('optimization','armijo_c1',.01), ('optimization','final','best_probe_checkpoint'),
    ('optimization','direction','Adam_with_inherited_state'),
    ('decision','trainable_ratio_max',1.), ('decision','gradient_ratio_max',.5),
    ('execution','technical_retries',1), ('execution','relative_tolerance',1e-5),
    ('budget','scaler_reconstructions_per_stage',36), ('budget','new_member_decodes_audit',0),
    ('budget','trial_objective_forms_max_per_stage',0),
])
def test_changed_protocol_cannot_silently_pass(cfg,tmp_path,section,key,value):
    cfg[section][key] = value; path = tmp_path/'changed.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


@pytest.mark.parametrize('fault',['head','fit_probe','source_checkpoint','source_updates','members','reference'])
def test_changed_roles_or_source_lineage_fail(cfg,tmp_path,fault):
    if fault == 'head': cfg['inputs']['heads'].pop()
    elif fault == 'fit_probe': cfg['inputs']['folds'][0]['fit_rows'][0] = cfg['inputs']['folds'][0]['probe_rows'][0]
    elif fault == 'source_checkpoint': cfg['inputs']['models'][0]['checkpoint']['sha256'] = '0'*64
    elif fault == 'source_updates': cfg['inputs']['heads'][0]['training']['updates'] = 550
    elif fault == 'members': cfg['head_members'].append('sample_rows')
    else: cfg['reference_points'][0]['start500_gradient_l2'] = 0.
    path = tmp_path/'changed.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


def test_unbound_cannot_hash_or_decode_research_archives(cfg,tmp_path,monkeypatch):
    path = tmp_path/'science.json'; path.write_text(json.dumps(cfg)); sha0 = d.p.sha
    def sha(path):
        assert Path(path).suffix not in ('.npz','.npy','.pt','.csv','.h5','.hdf5')
        return sha0(path)
    monkeypatch.setattr(d.p,'sha',sha)
    with pytest.raises(FileNotFoundError): d.require_binding(path,tmp_path/'absent.json')


@pytest.mark.parametrize('fault',['unbound','empty_dependencies','budget','environment'])
def test_forged_binding_rejects_missing_implementation(cfg,tmp_path,fault):
    config = tmp_path/'science.json'; config.write_text(json.dumps(cfg))
    binding = {'status':'bound_before_payload','before_research_payload':True,'science_config':d.p.pin(config),
        'budget':copy.deepcopy(cfg['budget']),'CPU_environment':copy.deepcopy(cfg['execution']['environment']),
        'future_output':cfg['output'],'future_artifacts':cfg['artifacts'],'dependencies':[]}
    if fault == 'unbound': binding['status'] = 'scientific_freeze_only'
    elif fault == 'budget': binding['budget']['main_auxiliary_updates_max'] = 0
    elif fault == 'environment': binding['CPU_environment']['CUDA_VISIBLE_DEVICES'] = '0'
    path = tmp_path/'binding.json'; path.write_text(json.dumps(binding))
    with pytest.raises(ValueError): d.require_binding(config,path)


@pytest.mark.parametrize('index,value',[(0,1.),(1,.5),(11,1/2048)])
def test_fixed_trial_grid(index,value): assert d.trial_alpha(index) == value


@pytest.mark.parametrize('index',[-1,12,True,0.,math.nan])
def test_no_extra_trial_or_alternate_grid(index):
    with pytest.raises(ValueError): d.trial_alpha(index)


def test_armijo_boundary_strict_improvement_and_no_epsilon():
    rhs = 1.-.0001*2.
    assert d.armijo_accept(1.,rhs,-2.,0)
    assert not d.armijo_accept(1.,math.nextafter(rhs,math.inf),-2.,0)
    assert not d.armijo_accept(1.,1.,-2.,0)
    assert not d.armijo_accept(1.,1.01,-2.,0)
    assert d.armijo_accept(1.,.99,-2.,0)


@pytest.mark.parametrize('slope',[0.,.01,math.nan,math.inf,True])
def test_non_descent_subgradient_is_not_used_for_trial(slope):
    with pytest.raises(ValueError): d.armijo_accept(1.,.5,slope,0)


def test_full_gradient_direction_needs_true_one_sided_slope():
    # Parent toy: lambda1 bias gradient is -1/28, but d=-G is ascent at the hinge.
    old = d.parent.old
    ys = [1.,-.75]
    gradient = sum(old.scalar_subgradient(y,0.,0.,1.)['total'] for y in ys)/2
    slope = sum(old.scalar_directional_derivative(y,0.,0.,-gradient,1.) for y in ys)/2
    assert gradient != 0 and slope > 0
    assert d.stopping_reason(0,abs(gradient),slope) == 'non_descent_selected_direction'
    with pytest.raises(ValueError): d.armijo_accept(1.,.5,slope,0)


@pytest.mark.parametrize('updates,norm,slope,rejected,status',[
    (50,1.,-1.,0,'update_cap'), (50,0.,0.,0,'update_cap'),
    (0,0.,0.,0,'zero_gradient'), (1,1.,0.,0,'non_descent_selected_direction'),
    (1,1.,1.,0,'non_descent_selected_direction'),
    (49,1.,-1.,12,'line_search_exhausted'), (49,1.,-1.,11,None),
])
def test_prespecified_stop_and_cap_priority(updates,norm,slope,rejected,status):
    assert d.stopping_reason(updates,norm,slope,rejected) == status


@pytest.mark.parametrize('args',[(51,1.,-1.,0),(-1,1.,-1.,0),(1,1.,-1.,13),(1,0.,-1.,0),(1,1.,math.nan,0)])
def test_invalid_iteration_state_stops(args):
    with pytest.raises(ValueError): d.stopping_reason(*args)


@pytest.mark.parametrize('t0,tend,g0,gend,cue,status',[
    (1.,.99,1.,.1,True,'joint_optimization_cue'),
    (1.,.99,1.,.10000001,False,'partial_target_reduction'),
    (1.,.99000001,1.,.1,False,'partial_target_reduction'),
    (1.,1.,1.,.01,False,'no_target_reduction'),
    (1.,.5,1.,2.,False,'partial_target_reduction'),
    (0.,0.,0.,0.,False,'reference_zero'), (1.,1.,0.,0.,False,'reference_zero'),
])
def test_loss_and_gradient_joint_not_mean_loss_alone(t0,tend,g0,gend,cue,status):
    r = d.endpoint_evidence(t0,tend,g0,gend)
    assert r['joint_cue'] is cue and r['status'] == status
    assert 'optimized' not in r['status']
    if not g0: assert r['gradient_ratio'] is None
    if not t0: assert r['trainable_ratio'] is None


@pytest.mark.parametrize('args,kwargs',[
    ((1.,1.1,1.,.1),{}), ((1.,.9,-1.,.1),{}), ((1.,.9,1.,math.nan),{}),
    ((1.,.9,1.,.1),{'numeric_valid':False}), ((1.,.9,1.,.1),{'numeric_valid':1}),
])
def test_invalid_endpoint_or_failed_audit_is_not_a_cue(args,kwargs):
    with pytest.raises(ValueError): d.endpoint_evidence(*args,**kwargs)


def rows():
    return [dict(seed=s,fold=f,arm='A_original',mode='uniform_affine',joint_cue=True) for s in d.p.SEEDS for f in range(3)]


def test_all9_head_cue_keeps_seed_and_fold_coverage():
    r = rows(); assert d.joint_evidence(r)
    r[4]['joint_cue'] = False; assert not d.joint_evidence(r)


@pytest.mark.parametrize('fault',['missing','duplicate','mixed_mode','boolean'])
def test_bad_group_cannot_be_pooled(fault):
    r = rows()
    if fault == 'missing': r.pop()
    elif fault == 'duplicate': r[-1] = dict(r[0])
    elif fault == 'mixed_mode': r[-1]['mode'] = 'harm_protected_affine'
    else: r[-1]['joint_cue'] = 1
    with pytest.raises(ValueError): d.joint_evidence(r)
