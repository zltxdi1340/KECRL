"""Scientific-contract tests use metadata and synthetic scalars only."""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
from experiments import libero_task7_endpoint_gradient_protocol as d


@pytest.fixture
def cfg(): return d.create()


def test_budget_fit_only_and_no_updates(cfg):
    assert cfg['budget']['total_NPZ_loads'] == 86 and cfg['budget']['total_member_decodes'] == 610
    assert cfg['budget']['total_archive_SHA_checks'] == 172
    assert cfg['budget']['fit_row_point_visits_per_stage'] == 201024
    assert cfg['budget']['linear_form_feature_multiplications_per_stage'] == 102924288
    assert cfg['budget']['gradient_coordinate_values_per_stage'] == 55512
    assert sum(cfg['budget']['tables'].values()) == cfg['budget']['total_output_CSV_rows'] == 18630
    assert cfg['budget']['new_optimizer_steps_parameter_updates_trial_steps_Policy_CNN_environment_or_sealed_calls'] == 0
    assert len(cfg['inputs']['heads']) == 36 and len(cfg['head_members']) == 8
    assert not set(cfg['head_members']) & {'predictions', 'probe_rows', 'sample_rows', 'step_loss', 'final_adam_m_weight'}


def test_factory_is_independent_and_metadata_only(cfg, monkeypatch):
    original = d.p.sha; seen = []
    def metadata_only(path):
        assert Path(path).suffix in {'.py', '.json', '.md', '.toml'}
        seen.append(str(path)); return original(path)
    monkeypatch.setattr(d.p, 'sha', metadata_only)
    cfg['design']['points'].append(100); cfg['inputs']['heads'][0]['training']['updates'] = 999
    frozen = d.create()
    assert frozen['design']['points'] == [0, 500] and frozen['inputs']['heads'][0]['training']['updates'] == 500
    assert seen


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'original_Policy_and_auxiliary_endpoints', 1000), ('scope', 'future_stabilization_steps', 0),
    ('scope', 'query_or_probe_numerical_scoring', 1), ('scope', 'new_optimizer_steps_parameter_updates_or_trial_steps', False),
    ('design', 'points', [0, 250, 500]), ('design', 'precision', 'float32_training_replay'),
    ('design', 'remaining_cue', 'selected_best_fold'), ('execution', 'technical_retries', 1),
    ('budget', 'audit_autograd_calls', 72), ('budget', 'total_member_decodes', 620),
])
def test_rejects_protocol_mutation(cfg, tmp_path, section, key, value):
    cfg[section][key] = value; path = tmp_path / 'changed.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


@pytest.mark.parametrize('kind', ['missing_head', 'swap_mode', 'bad_optimizer', 'unknown_member', 'probe_gradient'])
def test_rejects_input_or_member_mutation(cfg, tmp_path, kind):
    if kind == 'missing_head': cfg['inputs']['heads'].pop()
    elif kind == 'swap_mode': cfg['inputs']['heads'][0]['mode'] = 'harm_protected_affine'
    elif kind == 'bad_optimizer': cfg['inputs']['heads'][0]['training']['actual_optimizer']['lr'] = .003
    elif kind == 'unknown_member': cfg['head_members'].append('predictions')
    else: cfg['inputs']['folds'][0]['fit_rows'][0] = cfg['inputs']['folds'][0]['probe_rows'][0]
    path = tmp_path / 'changed.json'; path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError): d.validate(path)


def test_unbound_rejected_without_archive_SHA(cfg, tmp_path, monkeypatch):
    path = tmp_path / 'science.json'; path.write_text(json.dumps(cfg)); original = d.p.sha
    def metadata_only(p):
        assert Path(p).suffix not in ('.npz', '.npy', '.pt', '.csv', '.h5', '.hdf5')
        return original(p)
    monkeypatch.setattr(d.p, 'sha', metadata_only)
    with pytest.raises(FileNotFoundError): d.require_binding(path, tmp_path / 'missing-binding.json')


@pytest.mark.parametrize('kind', ['unbound', 'missing_roles', 'budget_changed'])
def test_forged_binding_cannot_bypass_missing_implementation(cfg, tmp_path, kind):
    config = tmp_path / 'science.json'; config.write_text(json.dumps(cfg))
    bound = {'status': 'bound_before_payload', 'before_research_payload': True, 'science_config': d.p.pin(config),
             'budget': copy.deepcopy(cfg['budget']), 'CPU_environment': cfg['execution']['environment'],
             'future_output': cfg['output'], 'future_artifacts': cfg['artifacts'], 'dependencies': []}
    if kind == 'unbound': bound['status'] = 'scientific_freeze_only'
    elif kind == 'budget_changed': bound['budget']['total_NPZ_loads'] += 1
    binding = tmp_path / 'binding.json'; binding.write_text(json.dumps(bound))
    with pytest.raises(ValueError): d.require_binding(config, binding)


@pytest.mark.parametrize('raw,y,p0,lam,expected', [
    (.5, 0., 0., 0., .25 / 7), (.5, 0., 0., 1., .5 / 7),
    (.25, .5, 0., 1., .0625 / 7), (2., 1., 0., 1., 0.),
    (-2., -1., 0., 1., 0.), (0., .5, 0., 1., .25 / 7),
])
def test_same_loss_scale_clip_and_penalty(raw, y, p0, lam, expected):
    row = d.scalar_terms(y, p0, raw, lam)
    assert row['full_objective'] == pytest.approx(expected)
    assert row['fixed_other6_floor'] == 0
    if raw == p0: assert row['penalty'] == 0


def test_fixed_other6_floor_does_not_hide_trainable_change():
    init = d.scalar_terms(.5, 0., 0., 0., 6.)
    final = d.scalar_terms(.5, 0., .25, 0., 6.)
    assert init['fixed_other6_floor'] == final['fixed_other6_floor'] == 6 / 7
    assert final['trainable_objective'] / init['trainable_objective'] == .25
    assert final['full_objective'] / init['full_objective'] > .9


@pytest.mark.parametrize('raw,expected', [(-2., 0.), (-1., -2 / 7), (0., 0.), (1., 2 / 7), (2., 0.)])
def test_clamp_selected_subgradient(raw, expected):
    row = d.scalar_subgradient(0., 0., raw, 0.)
    assert row['base'] == expected and row['total'] == row['base']


@pytest.mark.parametrize('y,p0,raw', [(1., 0., 0.), (-.75, 0., 0.), (.5, .25, .25)])
def test_zero_residual_penalty_gradient_is_zero(y, p0, raw):
    a, b = d.scalar_subgradient(y, p0, raw, 0.), d.scalar_subgradient(y, p0, raw, 1.)
    assert a['penalty'] == b['penalty'] == 0 and a['total'] == b['total']


@pytest.mark.parametrize('y,p0,raw,draw,lam', [
    (1., 0., 0., 1., 1.), (-.75, 0., 0., 1., 1.),
    (0., 1., 1., 1., 1.), (0., 1., 1., -1., 1.),
    (0., -1., -1., -1., 1.), (0., -1., -1., 1., 1.),
    (.5, 0., .25, -.5, 0.), (.5, 0., .75, .5, 1.),
    (0., 0., 2., -1., 1.),
])
def test_exact_one_sided_derivative_against_synthetic_difference(y, p0, raw, draw, lam):
    eps = 1e-7
    value = d.scalar_directional_derivative(y, p0, raw, draw, lam)
    numeric = (d.scalar_terms(y, p0, raw + eps * draw, lam)['full_objective'] - d.scalar_terms(y, p0, raw, lam)['full_objective']) / eps
    assert numeric == pytest.approx(value, abs=1e-6, rel=0)


def test_nonzero_selected_gradient_can_point_to_ascent_at_hinge():
    y = [1., -.75]
    g = sum(d.scalar_subgradient(v, 0., 0., 1.)['total'] for v in y) / 2
    right_derivative = sum(d.scalar_directional_derivative(v, 0., 0., 1., 1.) for v in y) / 2
    assert g == pytest.approx(-1 / 28) and right_derivative == pytest.approx(1 / 14)
    assert d.remaining_evidence(abs(g), abs(g), right_derivative)['status'] == 'no_descent_in_direction'


@pytest.mark.parametrize('g0,g500,dd,status,cue', [
    (1., .2, -.1, 'remaining_first_order_cue', True),
    (1., .2, -.099999, 'weak_descent_in_direction', False),
    (1., .2, -1e-10, 'no_descent_in_direction', False),
    (1., .2, .01, 'no_descent_in_direction', False),
    (1., 0., 0., 'zero_gradient', False),
    (0., .2, -.1, 'reference_zero', False),
])
def test_predeclared_relative_slope_screen(g0, g500, dd, status, cue):
    r = d.remaining_evidence(g0, g500, dd)
    assert r['status'] == status and r['remaining_cue'] is cue
    assert 'optimized' not in r['status']
    if not g0: assert r['relative_gradient'] is None and r['normalized_descent'] is None


@pytest.mark.parametrize('kind', ['negative_norm', 'nonfinite', 'false_gate', 'zero_gradient_nonzero_direction', 'overflow_ratio'])
def test_invalid_or_unstable_evidence_stops(kind):
    args, kwargs = (1., .5, -.1), {}
    if kind == 'negative_norm': args = (-1., .5, -.1)
    elif kind == 'nonfinite': args = (1., math.nan, -.1)
    elif kind == 'false_gate': kwargs['numeric_valid'] = False
    elif kind == 'zero_gradient_nonzero_direction': args = (0., 0., -.1)
    else: args = (1e-320, 1., -.1)
    with pytest.raises(ValueError): d.remaining_evidence(*args, **kwargs)


def test_joint_cue_keeps_all3seeds_all3folds_and_is_not_acceptance():
    rows = [{'seed': seed, 'fold': fold, 'remaining_cue': True} for seed in d.p.SEEDS for fold in range(3)]
    assert d.joint_remaining_evidence(rows)
    rows[-1]['remaining_cue'] = False; assert not d.joint_remaining_evidence(rows)
    with pytest.raises(ValueError): d.joint_remaining_evidence(rows[:-1])
    rows[-1] = rows[0].copy()
    with pytest.raises(ValueError): d.joint_remaining_evidence(rows)


@pytest.mark.parametrize('args', [(0., 0., 0., .5), (float('nan'), 0., 0., 0.), (2., 0., 0., 0.), (0., 0., 0., False)])
def test_invalid_scalar_loss_rejected(args):
    with pytest.raises(ValueError): d.scalar_terms(*args)


def test_local_torch_synthetic_boundary_conventions():
    env = os.environ.copy(); env.update(d.EXECUTION['environment'])
    code = '''import json,torch
assert torch.__version__=='2.4.1+cu124'
results=[]
for raw in (-2.,-1.,0.,1.,2.):
 x=torch.tensor(raw,dtype=torch.float64,requires_grad=True)
 results.append(float(torch.autograd.grad(torch.clamp(x,-1.,1.),x)[0]))
assert results==[0.,1.,1.,1.,0.]
x=torch.tensor(0.,dtype=torch.float64,requires_grad=True)
assert float(torch.autograd.grad(torch.relu(x),x)[0])==0.
print(json.dumps({'synthetic_clamp_gradients':results,'relu_at_zero':0.,'real_data_or_heads_read':0}))
'''
    child = subprocess.run([sys.executable, '-c', code], env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
    assert child.returncode == 0, child.stdout
    assert json.loads(child.stdout)['real_data_or_heads_read'] == 0
