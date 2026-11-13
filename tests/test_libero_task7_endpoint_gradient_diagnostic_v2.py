"""Synthetic checks for the v1 preprocessing undercount and v2 correction."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from experiments import libero_task7_endpoint_gradient_protocol_v2 as d
from experiments import libero_task7_endpoint_gradient_access_v2 as a
from experiments import run_libero_task7_endpoint_gradient_diagnostic_v2 as main
from experiments import audit_libero_task7_endpoint_gradient_diagnostic_v2 as audit


@pytest.fixture
def cfg(): return d.create()


def test_same_scientific_question_separate_corrective_budget(cfg):
    old = d.old.validate()
    for key in ('budget', 'scope', 'execution', 'inputs', 'source_members', 'head_members'):
        assert cfg[key] == old[key]
    assert cfg['version'] == 2 and cfg['output'] != old['output']
    assert cfg['corrective_execution']['total_v1_plus_v2_NPZ_loads'] == 172
    assert cfg['corrective_execution']['same_v1_retry'] is False
    assert d.p.load_pin(d.DEVIATION)['status'].endswith('budget_nonconforming')


@pytest.mark.parametrize('field', ['budget', 'scope', 'design', 'inputs', 'corrective_execution'])
def test_freeze_rejects_changed_roles_or_budget(cfg, tmp_path, field):
    changed = copy.deepcopy(cfg); changed[field]['undeclared_change'] = True
    path = tmp_path/'bad-science.json'; path.write_text(json.dumps(changed))
    with pytest.raises(ValueError): d.validate(path)


def test_unbound_v2_never_opens_payload(cfg, tmp_path):
    path = tmp_path/'science.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(FileNotFoundError): a.preflight(path, tmp_path/'absent-binding.json')


def test_complete_preprocessing_has18_builds_and18_reuses(cfg):
    work = a.RepresentationWork(cfg)
    for key, n in work.expected.items():
        z = np.zeros((n,256), np.float64)
        work.begin(key,n); work.built(key,z); work.reuse(key,z)
    work.complete(); r = work.record()
    assert r['scaler_reconstructions'] == r['paired_reuses'] == 18
    assert r['head_admissions'] == 36 and r['fit_standardization_row_visits'] == 50256
    assert r['fit_standardization_feature_coordinates'] == 12865536


@pytest.mark.parametrize('fault', ['duplicate', 'rows', 'key', 'new_object', 'mutable', 'third', 'missing_reuse'])
def test_preprocessing_budget_is_enforced(cfg, fault):
    work = a.RepresentationWork(cfg); key,n = next(iter(work.expected.items()))
    if fault == 'rows':
        with pytest.raises(ValueError): work.begin(key,n+1)
        return
    if fault == 'key':
        with pytest.raises(ValueError): work.begin((0,'wrong',0),n)
        return
    z = np.zeros((n,256),np.float64); work.begin(key,n); work.built(key,z)
    with pytest.raises(ValueError):
        if fault == 'duplicate': work.begin(key,n)
        elif fault == 'new_object': work.reuse(key,z.copy())
        elif fault == 'mutable':
            z.flags.writeable = True; work.reuse(key,z)
        elif fault == 'third': work.reuse(key,z); work.reuse(key,z)
        else: work.complete()


def head_for(plan):
    return {'fit_rows':np.array(plan['fit_rows'],np.int32), 'fit_demo_indices':np.array(plan['fit_demos'],np.int32),
            'updates':np.array([0,100,250,500],np.int32), 'optimizer_steps':np.arange(1,501,dtype=np.int32),
            'scaler_mean':np.zeros(256), 'scaler_scale':np.ones(256),
            'weights':np.zeros((4,256),np.float32), 'bias':np.zeros(4,np.float32)}


@pytest.mark.parametrize('who', ['primary','audit'])
@pytest.mark.parametrize('fault', ['scaler', 'fit_probe', 'demos', 'updates', 'nonfinite', 'nonzero0'])
def test_second_loss_independently_admits_its_head(cfg, who, fault):
    plan = cfg['inputs']['folds'][0]; h = head_for(plan)
    bits = h['scaler_mean'].tobytes(), h['scaler_scale'].tobytes()
    if fault == 'scaler': h['scaler_scale'][0] = 2
    elif fault == 'fit_probe': h['fit_rows'][0] = plan['probe_rows'][0]
    elif fault == 'demos': h['fit_demo_indices'][0] = plan['probe_demos'][0]
    elif fault == 'updates': h['optimizer_steps'][-1] = 499
    elif fault == 'nonfinite': h['weights'][3,0] = np.nan
    else: h['bias'][0] = .1
    with pytest.raises(ValueError):
        if who == 'primary': main.admit_reused(h,plan,bits)
        else: audit.admit_reused(h,plan,cfg['inputs']['action_lengths'],b''.join(bits))


def test_pair_reuse_preserves_float32_training_representation(cfg):
    plan = cfg['inputs']['folds'][0]; h = head_for(plan)
    feature = np.zeros((4188,256),np.float32)
    z = main.admit(h,feature,plan); rows,other = audit.prepare(h,feature,plan,cfg['inputs']['action_lengths'])
    h2 = copy.deepcopy(h); h2['weights'][3] = .1
    main.admit_reused(h2,plan,(h['scaler_mean'].tobytes(),h['scaler_scale'].tobytes()))
    audit.admit_reused(h2,plan,cfg['inputs']['action_lengths'],h['scaler_mean'].tobytes()+h['scaler_scale'].tobytes())
    assert np.array_equal(z,other) and np.array_equal(rows,h['fit_rows'])


def test_full_synthetic_72points_counts_actual_preprocessing_calls(cfg,tmp_path):
    helper_path = Path(__file__).with_name('test_libero_task7_endpoint_gradient_diagnostic.py')
    spec = importlib.util.spec_from_file_location('endpoint_v1_synthetic_helpers',helper_path)
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    cfg = helper.synthetic_inputs(cfg,tmp_path); path = tmp_path/'science.json'; path.write_text(json.dumps(cfg))
    env = os.environ.copy(); env.update(cfg['execution']['environment'])
    for i,(name,operation,entry) in enumerate([
        ('run_libero_task7_endpoint_gradient_diagnostic_v2','admit','execute_payload'),
        ('audit_libero_task7_endpoint_gradient_diagnostic_v2','prepare','audit_payload')]):
        code = f'''import json
from experiments import {name} as module
cfg=json.load(open({str(path)!r}))
pin={{'path':'synthetic-only','sha256':{'a'*64!r}}}
actual=[]
original=module.{operation}
def counted(*args,**kwargs):
    actual.append(len(args[2]['fit_rows']))
    return original(*args,**kwargs)
module.{operation}=counted
result=module.{entry}(cfg,pin,pin)
assert len(actual)==18 and sum(actual)==50256, actual
assert result['representation_work']['scaler_reconstructions']==18
'''
        proc = subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=195)
        (tmp_path/f'child-{i}.log').write_text(proc.stdout); assert proc.returncode == 0,proc.stdout
    r = json.loads(Path(cfg['output']).read_text()); done = json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
    assert len(r['points']) == 72 and done['total_NPZ_loads'] == 86 and done['total_member_decodes'] == 610
    assert r['representation_work'] == done['representation_work_per_stage']
    assert r['representation_work']['scaler_reconstructions'] == 18
    assert r['representation_work']['fit_standardization_feature_coordinates'] == 12865536
    assert r['representation_work']['head_admissions'] == 36
