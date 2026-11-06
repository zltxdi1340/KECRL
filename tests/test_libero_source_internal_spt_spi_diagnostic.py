from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.run_libero_source_internal_spt_spi_diagnostic import (
    ROOT, preflight, sha256, validate_spec,
)

CONFIG=ROOT/'configs/libero-source-internal-spt-spi-diagnostic_v1.json'


def binding(tmp_path):
    files={
        'runner':'experiments/run_libero_source_internal_spt_spi_diagnostic.py',
        'fomaml':'src/algorithms/libero_bc_fomaml.py',
        'causal_dataset':'src/environments/libero_causal_t1_dataset.py',
        'base_dataset':'src/environments/libero_dataset.py',
        'policy':'src/algorithms/libero_bc_policy.py',
        'spt':'src/skills/libero_pick_place_spt.py','models':'src/skills/models.py','contracts':'src/skills/contracts.py'}
    value={'formal_result':False,'config_sha256':sha256(CONFIG),
           'files':{n:{'path':p,'sha256':sha256(ROOT/p)} for n,p in files.items()}}
    path=tmp_path/'binding.json';path.write_text(json.dumps(value))
    return path,value


def test_preflight_validates_metadata_without_real_data_or_model_access(tmp_path,monkeypatch):
    path,_=binding(tmp_path)
    import builtins
    original=builtins.__import__
    def restricted(name,*args,**kwargs):
        if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:
            pytest.fail(f'metadata preflight imported runtime {name}')
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',restricted)
    config,_=preflight(CONFIG,path)
    assert config['adaptation_updates']==500 and config['new_environment_steps']==0


@pytest.mark.parametrize('key,value',[
    ('adaptation_updates',1000),('source_tasks',[0,1,2,4,7]),
    ('folds',[{'probe':0,'source_update_tasks':[0,3,4,7]}]),
    ('validation_qualification_heldout_access',True),('formal_result',True),
    ('alignment','stored_same_index'),('candidate_acceptance',True),
    ('new_environment_steps',300),('source_inner_steps',2),('batch_size',16),
    ('gradient_clip_norm',1),('curve_updates',[0,500])])
def test_unfrozen_protocol_changes_fail_closed(key,value):
    config=json.loads(CONFIG.read_text()); config[key]=value
    with pytest.raises(ValueError):validate_spec(config)


@pytest.mark.parametrize('kind',['missing_code','bad_code_hash','bad_config_hash'])
def test_missing_or_mismatched_binding_is_rejected(tmp_path,kind):
    path,value=binding(tmp_path)
    if kind=='missing_code':del value['files']['fomaml']
    elif kind=='bad_code_hash':value['files']['runner']['sha256']='0'*64
    else:value['config_sha256']='0'*64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):preflight(CONFIG,path)


def test_budget_arithmetic_is_explicit_and_inner_writes_not_declared_matched():
    c=json.loads(CONFIG.read_text());validate_spec(c)
    n=len(c['folds'])*len(c['training_seeds']);tasks=4;steps=c['source_outer_updates']
    b=c['budget_totals']
    assert b['generator_optimizer_updates']==n*steps*2
    assert b['source_data_gradient_calls']==n*steps*tasks*2*2
    assert b['generator_vjp_calls']==n*steps*tasks*2
    assert b['disposable_source_sgd_updates']==n*steps*tasks
    assert b['probe_adaptation_optimizer_updates']==n*3*500
    assert b['probe_curve_checkpoints']==n*3*len(c['curve_updates'])
    assert c['budget_matching']['same_total_parameter_writes_including_disposable_inner_models'] is False
