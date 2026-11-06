"""Factorial signs, paired reference gates, fixed budgets and cached-only access."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from experiments import analyze_libero_task4_task7_source_stabilization_input_decomposition as d
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


def public(value=0):
    return {'agentview_rgb': np.full((30, 128, 128, 3), value, np.uint8),
            'eye_in_hand_rgb': np.full((30, 128, 128, 3), value, np.uint8),
            'joint_states': np.full((30, 7), value, np.float32),
            'gripper_states': np.full((30, 2), value, np.float32)}


def fake_forward(_model, obs):
    mask = int(obs['agentview_rgb'][0, 0, 0, 0]) + 2 * int(obs['eye_in_hand_rgb'][0, 0, 0, 0]) + 4 * int(obs['joint_states'][0, 0])
    return np.full((30, 7), mask / 10, np.float32)


def test_all_eight_combinations_preserve_joint_gripper_pairing():
    zero, five = public(), public(1)
    for mask in range(8):
        mixed = d.core.mix_observations(zero, five, mask)
        for key, bit in zip(d.core.OBS_KEYS, (1, 2, 4, 4)):
            assert mixed[key] is (five if mask & bit else zero)[key]


def test_both_natural_references_and_repeats_gate_mixed_batches():
    progress, saved, events = d.counters(), {}, []
    values = d.collect_outputs(None, public(), public(1), {0: np.zeros((30, 7), np.float32),
        7: np.full((30, 7), .7, np.float32)}, progress, lambda kind, **row: events.append((kind, row)), saved, fake_forward)
    assert values.shape == (8, 30, 7)
    assert progress['main_forward_attempts'] == progress['main_forward_calls'] == 8
    assert progress['repeat_forward_attempts'] == progress['repeat_forward_calls'] == 2
    assert progress['main_model_rows'] == 240 and progress['repeat_model_rows'] == 60
    assert progress['natural_endpoint_parity_exact'] == progress['repeated_endpoint_parity_exact'] == 2
    assert [row['mask'] for kind, row in events if kind == 'forward_complete'] == [0, 7, 0, 7, 1, 2, 3, 4, 5, 6]
    assert np.array_equal(saved['mask0'], saved['repeat0'])


def test_endpoint_mismatch_stops_before_any_mixed_input():
    progress, saved = d.counters(), {}
    with pytest.raises(ValueError, match='endpoint parity'):
        d.collect_outputs(None, public(), public(1), {0: np.ones((30, 7), np.float32)},
                          progress, lambda *a, **k: None, saved, fake_forward)
    assert progress['main_forward_calls'] == 1
    assert set(saved) == {'mask0'}  # retained for caller's failed-result archive


def test_repeat_mismatch_stops_before_mixed_inputs():
    progress, saved = d.counters(), {}
    def unstable(model, observations):
        value = fake_forward(model, observations)
        return value + np.float32(.01) if progress['repeat_forward_attempts'] else value
    with pytest.raises(ValueError, match='repeated endpoint'):
        d.collect_outputs(None, public(), public(1), {0: np.zeros((30, 7), np.float32),
            7: np.full((30, 7), .7, np.float32)}, progress, lambda *a, **k: None, saved, unstable)
    assert progress['main_forward_calls'] == 2 and progress['repeat_forward_calls'] == 1
    assert set(saved) == {'mask0', 'mask7', 'repeat0'}


def factorial_predictions():
    values = np.zeros((8, 30, 7), np.float32)
    for mask in range(8):
        a, h, p = (int(bool(mask & bit)) for bit in (1, 2, 4))
        values[mask, :, 0] = .01 * (a + 2 * h + 4 * p + 3 * a * h + 5 * h * p)
        values[mask, :, 6] = .1 * (a - p)
    return values


def test_signed_interactions_reconstruct_vectors_in_both_references():
    result = d.decompose(factorial_predictions())
    for anchor in ('zero', 'five'):
        row = result[anchor]
        assert row['signed_interactions']['camera']['mean_signed_action_delta'][0] == pytest.approx(.03)
        assert row['signed_interactions']['rgb_proprio']['mean_signed_action_delta'][0] == pytest.approx(.05)
        assert max(row['identity_residual_max_abs'].values()) <= 1e-12
    assert result['zero']['swaps']['full']['mean_signed_action_delta'][0] > 0
    assert result['five']['swaps']['full']['mean_signed_action_delta'][0] < 0
    row = result['zero']['swaps']
    assert row['full']['metrics']['action_l2']['mean'] != pytest.approx(
        row['rgb']['metrics']['action_l2']['mean'] + row['proprio']['metrics']['action_l2']['mean'])


def test_sign_changes_include_neutral_but_do_not_assign_interaction_action_signs():
    result = d.decompose(factorial_predictions())
    assert result['zero']['swaps']['agentview']['gripper_sign_changes'] == 30
    assert result['zero']['swaps']['full']['gripper_sign_changes'] == 0
    assert 'gripper_sign_changes' not in result['zero']['signed_interactions']['camera']


def test_both_references_and_all_three_seeds_required_for_comparison():
    models = {str(s): {'decomposition': d.decompose(factorial_predictions())} for s in d.SEEDS}
    assert d.decisions(models)['rgb_minus_proprio']['both_references_all_three_means_positive'] is True
    models[str(d.SEEDS[-1])]['decomposition']['five']['comparisons']['rgb_minus_proprio']['mean'] = -1.
    flags = d.decisions(models)['rgb_minus_proprio']
    assert flags['zero_reference_all_three_means_positive'] is True
    assert flags['five_reference_all_three_means_positive'] is False
    assert flags['both_references_all_three_means_positive'] is False


@pytest.mark.parametrize('bad', ['order', 'extra_private', 'rgb_dtype', 'nonfinite'])
def test_malformed_or_wrong_role_cache_rejected(bad):
    values = {**public(), 'source_demo_indices': np.arange(30)}
    if bad == 'order':
        values['source_demo_indices'] = np.arange(30, 60)
    elif bad == 'extra_private':
        values['sim_state'] = np.zeros(92)
    elif bad == 'rgb_dtype':
        values['agentview_rgb'] = values['agentview_rgb'].astype(np.float32)
    else:
        values['joint_states'][0, 0] = np.nan
    with pytest.raises(ValueError, match='cache|observation'):
        d.validate_public(values)


@pytest.mark.parametrize('key,value', [('checkpoint_updates', 1000), ('source_demo_indices', list(range(30, 50))),
                                      ('new_environment_resets_or_steps', 1), ('main_forward_calls', 49)])
def test_changed_protocol_rejected_before_input_access(tmp_path, monkeypatch, key, value):
    config = {'check_type': 'libero_source_stabilization_input_decomposition', 'formal_result': False,
              'scope': deepcopy(d.SCOPE), 'analysis': d.ANALYSIS}
    config['scope'][key] = value
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps(config))
    monkeypatch.setattr(d, 'sha', lambda _: pytest.fail('input hash read after rejected scope'))
    with pytest.raises(ValueError, match='frozen input decomposition'):
        d.load_protocol(p)


def test_changed_file_and_existing_outputs_rejected(tmp_path, monkeypatch):
    p = tmp_path / 'cache.npz'
    p.write_bytes(b'changed bytes')
    with pytest.raises(ValueError, match='hash differs'):
        d.verify_file({'path': str(p), 'sha256': '0' * 64})
    monkeypatch.setattr(d, 'load_protocol', lambda _: pytest.fail('must not load inputs'))
    with pytest.raises(ValueError, match='already exists'):
        d.run(tmp_path / 'unused', p, tmp_path / 'artifacts')
    assert p.read_bytes() == b'changed bytes'
    with pytest.raises(FileExistsError):
        d.save_npz(p, {'test': np.zeros(1)})


def test_actual_cpu_forward_preserves_weights_inputs_and_gradients():
    model = LiberoBCPolicy(LiberoBCPolicyConfig(cnn_channels=(2, 2, 2), spatial_pool_size=1,
        visual_embedding_dim=4, fusion_dim=8, recurrent_hidden_dim=8, memory_mode='cnn_only'))
    model.eval()
    values = public()
    before_weights, before_inputs = d.core._digest_weights(model.state_dict()), d.digest_batch(values)
    predictions = d.forward_batch(model, values)
    assert predictions.shape == (30, 7)
    assert d.core._digest_weights(model.state_dict()) == before_weights
    assert d.digest_batch(values) == before_inputs
    assert all(p.grad is None for p in model.parameters())


def test_import_requires_no_environment_or_hdf5_loader():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'h5py', 'libero', 'mujoco', 'robosuite', 'gym'}
    assert not name.startswith('src.environments')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import analyze_libero_task4_task7_source_stabilization_input_decomposition
"""
    subprocess.run([sys.executable, '-c', code], cwd=d.ROOT, check=True, capture_output=True, text=True)
