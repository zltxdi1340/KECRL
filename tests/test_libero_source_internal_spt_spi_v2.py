import json

import pytest

from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import ROOT,preflight,sha256,validate_spec

CONFIG=ROOT/'configs/libero-source-internal-spt-spi-diagnostic_v2.json'


def test_v2_retains_scientific_budget_and_strong_determinism_requirement():
    v1=json.loads((ROOT/'configs/libero-source-internal-spt-spi-diagnostic_v1.json').read_text())
    v2=json.loads(CONFIG.read_text());validate_spec(v2)
    for key,value in v1.items():
        if key not in {'frozen_at_utc','references'}:assert v2[key]==value
    assert v2['execution_revision']=='v2_deterministic_equivalent_pool'
    assert 'preserved_v1_failure_audit' in v2['references']


def test_v2_binding_requires_the_pool_implementation(tmp_path):
    files={'runner':'experiments/run_libero_source_internal_spt_spi_diagnostic_v2.py',
           'fomaml':'src/algorithms/libero_bc_fomaml.py','causal_dataset':'src/environments/libero_causal_t1_dataset.py',
           'base_dataset':'src/environments/libero_dataset.py','policy':'src/algorithms/libero_bc_policy.py',
           'spt':'src/skills/libero_pick_place_spt.py','models':'src/skills/models.py','contracts':'src/skills/contracts.py'}
    binding={'formal_result':False,'config_sha256':sha256(CONFIG),
             'files':{k:{'path':p,'sha256':sha256(ROOT/p)} for k,p in files.items()}}
    path=tmp_path/'binding.json';path.write_text(json.dumps(binding))
    with pytest.raises(ValueError,match='missing'):preflight(CONFIG,path)
    p='src/algorithms/libero_deterministic_pool.py'
    binding['files']['deterministic_pool']={'path':p,'sha256':sha256(ROOT/p)}
    path.write_text(json.dumps(binding))
    observed,_=preflight(CONFIG,path)
    assert observed['adaptation_updates']==500


def test_v2_cannot_fall_back_to_v1_execution_revision():
    c=json.loads(CONFIG.read_text());c['execution_revision']='v1'
    with pytest.raises(ValueError):validate_spec(c)
