"""Metadata and synthetic scalar geometry; real gradient CSVs remain closed."""
import builtins
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
from experiments import libero_task7_mse_gate_stop_direction_grid_protocol as d


@pytest.fixture
def cfg(): return d.create()


def test_fixed_cohort_roles_and_minimal_projection(cfg):
    assert len(cfg['heads']) == 11
    assert sum(h['cohort']=='zero' for h in cfg['heads']) == 6
    assert sum(h['cohort']=='short' for h in cfg['heads']) == 5
    assert len(cfg['full50_metadata_only']) == 7
    assert len({tuple(h[k] for k in d.IDS) for h in cfg['heads']+cfg['full50_metadata_only']}) == 18
    assert sorted(h['stop_extra_updates'] for h in cfg['heads'] if h['cohort']=='short') == [3,3,4,5,11]
    assert cfg['scope']['original_Policy_updates'] == cfg['scope']['source_auxiliary_updates'] == 500
    assert cfg['scope']['future_stabilization_steps'] == 5
    assert not cfg['scope']['probe_source_query_validation_heldout_qualification_reserved_final_access']
    assert set(cfg['inputs']) == {'endpoints','parameters','trials'}
    assert 'parameter' not in cfg['projections']['parameters']
    assert cfg['projections']['parameters'][-4:] == d.VECTORS
    assert cfg['design']['points'] == ['saved500','start500_mse_guarded_armijo_final']


def test_budget_separates_lexical_selected_and_projected_reads(cfg):
    b=cfg['budget']
    assert b['source_rows_per_stage'] == {'endpoints':54,'parameters':13878,'trials':10090}
    assert b['selected_rows_per_stage'] == {'endpoints':22,'parameters':22*257,'trials':11*12}
    assert {k:len(v) for k,v in cfg['projections'].items()} == {'endpoints':26,'parameters':11,'trials':21}
    assert sum(b['projected_cells_per_stage'].values()) == 65538
    assert b['total_projected_cells'] == 131076
    assert b['total_selected_rows'] == 11616 and b['total_source_rows'] == 48044
    assert b['lexical_cells_per_stage'] == 380100 and b['total_lexical_cells'] == 760200
    assert b['source_CSV_passes_per_stage'] == 3 and b['total_source_CSV_passes'] == 6
    assert b['source_SHA_per_stage'] == 6 and b['total_source_SHA'] == 12
    assert b['cached_gradient_vector_coordinates_per_stage'] == 22616
    assert b['gradient_coordinate_rows_checked_per_stage'] == 5654
    assert b['output_rows'] == sum(b['tables'].values()) == 177
    assert b['new_model_forward_gradient_update_array_or_sealed_calls'] == 0


def test_factory_does_not_read_or_hash_source_CSV_arrays(cfg,monkeypatch):
    sha0,open0,io0=d.p.sha,builtins.open,io.open
    seen=[]
    def check(path):
        if isinstance(path,(str,Path)):assert Path(path).suffix not in ('.csv','.npz','.npy','.pt','.h5','.hdf5')
    def sha(path):check(path);seen.append(str(path));return sha0(path)
    def opened(path,*args,**kwargs):check(path);return open0(path,*args,**kwargs)
    def io_opened(path,*args,**kwargs):check(path);return io0(path,*args,**kwargs)
    monkeypatch.setattr(d.p,'sha',sha);monkeypatch.setattr(builtins,'open',opened);monkeypatch.setattr(io,'open',io_opened)
    cfg['heads'][0]['cohort']='changed';cfg['scope']['training_seeds'].append(42)
    fresh=d.create()
    assert fresh['heads'][0]['cohort'] in ('zero','short')
    assert fresh['scope']['training_seeds']==[20261106,20261107,20261108] and seen


def test_metadata_module_in_fresh_process_imports_no_numeric_framework():
    code='from experiments import libero_task7_mse_gate_stop_direction_grid_protocol as d; import sys; d.create(); assert "numpy" not in sys.modules and "torch" not in sys.modules'
    result=subprocess.run([sys.executable,'-c',code],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
    assert result.returncode==0,result.stdout


@pytest.mark.parametrize('section,key,value',[
    ('scope','original_Policy_updates',550),('scope','future_stabilization_steps',0),
    ('scope','new_optimization_gradient_or_model_forward_environment_calls',1),
    ('scope','probe_source_query_validation_heldout_qualification_reserved_final_access',True),
    ('scope','loss_representation_search_direction_grid_budget_or_checkpoint_rule_changes',True),
    ('design','exact_total_slope','minus_norm_squared_is_always_exact'),
    ('design','exact_MSE_slope','always_use_dot_even_clip_ties'),
    ('design','slack','previous_accepted_MSE_minus_stop_MSE'),
    ('design','numeric_band','relax_original_acceptance_by_epsilon'),
    ('budget','total_source_CSV_passes',3),('budget','total_source_SHA',0),
    ('budget','total_projected_cells',0),('budget','cached_gradient_vector_coordinates_per_stage',0),
    ('execution','technical_retries',1),('execution','absolute_tolerance',1e-6),
])
def test_scientific_change_requires_new_freeze(cfg,tmp_path,section,key,value):
    cfg[section][key]=value;path=tmp_path/'changed.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):d.validate(path)


@pytest.mark.parametrize('fault',['missing','duplicate','full_as_focus','step','parameter','states','source_sha'])
def test_source_coverage_and_field_changes_fail(cfg,tmp_path,fault):
    if fault=='missing':cfg['heads'].pop()
    elif fault=='duplicate':cfg['heads'][-1]=dict(cfg['heads'][0])
    elif fault=='full_as_focus':cfg['heads'][0].update(cfg['full50_metadata_only'][0])
    elif fault=='step':cfg['heads'][0]['stop_extra_updates']+=1
    elif fault=='parameter':cfg['projections']['parameters'].append('parameter')
    elif fault=='states':cfg['inputs']['states']=cfg['inputs']['endpoints']
    else:cfg['inputs']['trials']['sha256']='0'*64
    path=tmp_path/'changed.json';path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):d.validate(path)


def test_missing_binding_rejects_before_source_SHA(cfg,tmp_path,monkeypatch):
    config=tmp_path/'science.json';config.write_text(json.dumps(cfg));original=d.p.sha
    def sha(path):
        assert Path(path).suffix in ('.py','.md','.json','.toml')
        return original(path)
    monkeypatch.setattr(d.p,'sha',sha)
    with pytest.raises(FileNotFoundError):d.require_binding(config,tmp_path/'absent.json')


@pytest.mark.parametrize('fault',['empty_dependencies','status','budget','environment'])
def test_forged_binding_cannot_release_source_projection(cfg,tmp_path,fault):
    config=tmp_path/'science.json';config.write_text(json.dumps(cfg))
    bound={'status':'bound_before_payload','before_research_payload':True,'science_config':d.p.pin(config),
        'budget':copy.deepcopy(cfg['budget']),'CPU_environment':copy.deepcopy(cfg['execution']['environment']),
        'future_output':cfg['output'],'future_artifacts':cfg['artifacts'],'dependencies':[]}
    if fault=='status':bound['status']='scientific_freeze_only'
    elif fault=='budget':bound['budget']['total_source_CSV_passes']=0
    elif fault=='environment':bound['CPU_environment']['CUDA_VISIBLE_DEVICES']='0'
    path=tmp_path/'binding.json';path.write_text(json.dumps(bound))
    with pytest.raises(ValueError):d.require_binding(config,path)


def vec(value):return [value]+[0.]*256


def geometry(gb=.125,gp=-.25,clip=0,hinge=0,slope=None):
    g=gb+gp;direction=-g
    return d.gradient_geometry(vec(gb),vec(gp),vec(g),vec(direction),clip,hinge,-g*g if slope is None else slope)


def test_smooth_direction_can_raise_MSE_and_lower_total():
    out=geometry()
    assert out['exact_mse_slope']==.109375 and out['cached_exact_total_slope']==-.015625
    assert out['exact_penalty_slope']==-.03125 and out['base_penalty_cosine']==-1.
    assert out['kink_gap']==0
    inference=d.feasibility_evidence(.1,.1,out['cached_exact_total_slope'],out['exact_mse_slope'])
    assert inference['inference_class']=='boundary_direction_conflict_near_zero_alpha'
    assert inference['linear_alpha_scale'] is None


def test_hinge_ties_need_cached_exact_total_but_not_unknown_MSE():
    out=geometry(gb=-.125,gp=0.,hinge=2,slope=.03125)
    assert out['dot_total_direction']==-.015625 and out['cached_exact_total_slope']>0
    assert out['exact_mse_slope']==-.109375
    assert out['exact_penalty_slope']==.046875 and out['kink_gap']==.046875
    assert out['base_penalty_cosine'] is None


def test_clip_ties_keep_MSE_and_penalty_slope_unresolved():
    out=geometry(clip=1,slope=-.01)
    assert out['exact_mse_slope'] is out['exact_penalty_slope'] is None
    assert out['mse_slope_status']=='unresolved_clip_tie'
    assert d.feasibility_evidence(.1,.1,-.01,None)['inference_class']=='boundary_clip_kink_unresolved'
    # Positive slack uses continuity even when a separate MSE slope is unknown.
    assert d.feasibility_evidence(.1,.09,-.01,None)['inference_class']=='positive_slack_local_feasible_by_continuity'


def test_zero_norm_has_null_cosine_without_epsilon():
    out=geometry(gb=0.,gp=0.)
    assert out['base_penalty_cosine'] is None and out['exact_mse_slope']==0
    assert d.safe_ratio(1.,0.) is None


@pytest.mark.parametrize('fault',['length','nonfinite','boolean','total','direction','clip_count','hinge_count','smooth_slope','hinge_gap'])
def test_bad_gradient_geometry_rejected(fault):
    args=[vec(.125),vec(-.25),vec(-.125),vec(.125),0,0,-.015625]
    if fault=='length':args[0].pop()
    elif fault=='nonfinite':args[0][0]=math.inf
    elif fault=='boolean':args[0][0]=True
    elif fault=='total':args[2][0]+=.01
    elif fault=='direction':args[3][0]+=.01
    elif fault=='clip_count':args[4]=True
    elif fault=='hinge_count':args[5]=-1
    elif fault=='smooth_slope':args[6]=0.
    else:args[5]=2;args[6]=-.1
    with pytest.raises(ValueError):d.gradient_geometry(*args)


@pytest.mark.parametrize('m0,m,sT,sM,label',[
    (.1,.09,-1.,2.,'positive_slack_local_feasible_by_continuity'),
    (.1,.1,-1.,-2.,'boundary_descent_local_feasible'),
    (.1,.1,-1.,0.,'boundary_near_zero_mse_slope_unresolved'),
    (.1,.1,-1.,d.TOL,'boundary_near_zero_mse_slope_unresolved'),
    (.1,.1,-d.TOL,1.,'near_zero_total_slope_unresolved'),
    (.1,math.nextafter(.1,0.),-1.,1.,'positive_slack_near_numeric_band'),
])
def test_prespecified_inference_classes(m0,m,sT,sM,label):
    out=d.feasibility_evidence(m0,m,sT,sM)
    assert out['inference_class']==label and not out['untried_step_evaluated'] and not out['global_infeasibility_claim']
    assert out['strict_slack_sign']==('positive' if m<m0 else 'zero')


def test_linear_alpha_is_only_scale_not_untried_forward():
    out=d.feasibility_evidence(.125,.0625,-1.,2.)
    assert out['linear_alpha_scale']==.03125 and out['linear_scale_over_smallest_grid_alpha']==64
    assert out['untried_step_evaluated'] is False


@pytest.mark.parametrize('args',[
    (.1,math.nextafter(.1,math.inf),-1.,1.),(.1,.1,0.,1.),(.1,.1,1.,1.),
    (.1,-.1,-1.,1.),(.1,.1,-1.,math.nan),(True,.1,-1.,1.),
])
def test_description_band_never_relaxes_old_gate(args):
    with pytest.raises(ValueError):d.feasibility_evidence(*args)


def test_cached_trial_contributions_and_first_order_residual():
    # stop M=.0625, trial M=.125, P decreases .02: total may decrease despite MSE rise.
    mt,m=.125,.0625;p0=.03;p1=.01;t0=m/7+p0;t1=mt/7+p1
    out=d.trial_terms(.125,m,p0,t0,mt,p1,t1,.5,.1)
    assert out['delta_mse_step']==.0625 and out['mse_gate_excess']==0
    assert out['delta_trainable']<0 and out['mse_first_order_prediction']==.05
    assert abs(out['mse_first_order_residual']-.0125)<1e-15
    assert abs(out['conservation_residual'])<1e-15
    unknown=d.trial_terms(.125,m,p0,t0,mt,p1,t1,.5,None)
    assert unknown['mse_first_order_prediction'] is unknown['mse_first_order_residual'] is None


@pytest.mark.parametrize('fault',['conservation','alpha_zero','alpha_negative','nonfinite','bad_envelope'])
def test_invalid_cached_trial_is_not_directional_evidence(fault):
    args=[.125,.0625,.03,.0625/7+.03,.125,.01,.125/7+.01,.5,.1]
    if fault=='conservation':args[6]+=.01
    elif fault=='alpha_zero':args[7]=0.
    elif fault=='alpha_negative':args[7]=-.1
    elif fault=='nonfinite':args[4]=math.nan
    else:args[0]=.01
    with pytest.raises(ValueError):d.trial_terms(*args)
