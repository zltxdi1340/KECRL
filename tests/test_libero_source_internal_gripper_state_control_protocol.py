from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import validate_libero_source_internal_gripper_state_control_protocol as a


def sample_fixture():
    actions = np.zeros((15, 7), np.float32)
    actions[:, 6] = [-1] * 3 + [1] * 4 + [-1] * 8
    masks, _ = a.phase.parent.expert_masks(actions)
    target = actions[1:].copy()
    data = {'targets': target, 'predictions': np.stack([target] * 9),
            'model_ids': np.array([f'{s}:{arm}' for s in a.SEEDS for arm in a.ARMS]),
            'demonstration_indices': np.zeros(14, np.int16),
            'action_steps': np.arange(1, 15, dtype=np.int32),
            'mask_near_open': masks['near_open'][1:]}
    return data, a.build_task_samples(0, data)


def test_event_anchor_uses_event_preaction_and_never_action0():
    _, sample = sample_fixture()
    rows = sample['selected']
    assert [r['action_step'] for r in rows] == list(range(8, 13))
    assert [r['source_row'] for r in rows] == list(range(7, 12))
    assert all(r['event_step'] == 7 and r['event_anchor_row'] == 6 for r in rows)
    assert [r['lag1_anchor_row'] for r in rows] == list(range(6, 11))
    assert sample['at_event_sham_rows'] == [6]
    assert sample['source_frame_ids'][0] == [0, 1]


def test_replacement_uses_immutable_history_and_preserves_sham_and_other_rows():
    _, sample = sample_fixture()
    original = np.arange(28, dtype=np.float32).reshape(14, 2)
    original.setflags(write=False)
    hold = a.replace_gripper_states(original, sample, a.CONDITIONS[1])
    lag = a.replace_gripper_states(original, sample, a.CONDITIONS[2])
    selected = [r['source_row'] for r in sample['selected']]
    np.testing.assert_array_equal(hold[selected], np.tile(original[6], (5, 1)))
    np.testing.assert_array_equal(lag[selected], original[6:11])
    np.testing.assert_array_equal(hold[selected[0]], lag[selected[0]])
    untreated = np.ones(14, bool)
    untreated[selected] = False
    np.testing.assert_array_equal(hold[untreated], original[untreated])
    np.testing.assert_array_equal(lag[untreated], original[untreated])
    np.testing.assert_array_equal(original, np.arange(28, dtype=np.float32).reshape(14, 2))
    assert not np.shares_memory(hold, original) and not np.shares_memory(lag, original)


def test_identity_copies_original_without_changes():
    _, sample = sample_fixture()
    original = np.ones((14, 2), np.float32)
    result = a.replace_gripper_states(original, sample, 'identity')
    np.testing.assert_array_equal(result, original)
    assert not np.shares_memory(result, original)


@pytest.mark.parametrize('corruption', ['future', 'other_demo', 'duplicate', 'out_of_range', 'wrong_event', 'lag6'])
def test_malformed_or_noncausal_anchors_reject(corruption):
    _, sample = sample_fixture()
    row = sample['selected'][0]
    if corruption == 'future':
        row['event_anchor_row'] = row['source_row']
    elif corruption == 'other_demo':
        sample['source_frame_ids'][row['event_anchor_row']][0] = 1
    elif corruption == 'duplicate':
        sample['selected'].append(deepcopy(row))
    elif corruption == 'out_of_range':
        row['source_row'] = 14
    elif corruption == 'wrong_event':
        row['event_anchor_row'] -= 1
    else:
        row['lag'] = 6
    with pytest.raises(ValueError, match='anchor'):
        a.replace_gripper_states(np.zeros((14, 2), np.float32), sample, a.CONDITIONS[1])


@pytest.mark.parametrize('corruption', ['float64', 'shape', 'nan', 'unsupported'])
def test_public_gripper_interface_rejects_invalid_inputs(corruption):
    _, sample = sample_fixture()
    values, condition = np.zeros((14, 2), np.float32), a.CONDITIONS[1]
    if corruption == 'float64':
        values = values.astype(np.float64)
    elif corruption == 'shape':
        values = np.zeros((14, 3), np.float32)
    elif corruption == 'nan':
        values[0, 0] = np.nan
    else:
        condition = 'zero_gripper'
    with pytest.raises(ValueError, match='interface'):
        a.replace_gripper_states(values, sample, condition)


def test_selection_and_repeat_do_not_depend_on_model_errors():
    data, before = sample_fixture()
    data['predictions'][:, :, 6] *= -1
    after = a.build_task_samples(0, data)
    assert before['repeat_batch'] == after['repeat_batch'] == 0
    assert before['repeat_valid_rows'] == after['repeat_valid_rows'] == 14
    assert before['repeat_network_slots'] == after['repeat_network_slots'] == 20
    assert before['source_frame_ids'] == after['source_frame_ids']
    for old, new in zip(before['selected'], after['selected']):
        assert {k: v for k, v in old.items() if k != 'original_predicted_gripper_signs'} == {
            k: v for k, v in new.items() if k != 'original_predicted_gripper_signs'}
    assert all(r['original_correct_n'] == 5 for r in before['model_labels'])
    assert all(r['original_wrong_close_n'] == 5 for r in after['model_labels'])


@pytest.mark.parametrize('corruption', ['dtype', 'nan', 'range', 'model_order', 'noncontiguous'])
def test_cached_frame_interface_rejects_malformed_provenance(corruption):
    data, _ = sample_fixture()
    if corruption == 'dtype':
        data['predictions'] = data['predictions'].astype(np.float64)
    elif corruption == 'nan':
        data['targets'][0, 0] = np.nan
    elif corruption == 'range':
        data['predictions'][0, 0, 0] = 1.1
    elif corruption == 'model_order':
        data['model_ids'] = data['model_ids'][::-1]
    else:
        data['action_steps'][-1] += 1
    with pytest.raises(ValueError):
        a.build_task_samples(0, data)


def test_sign_corrections_and_harms_are_separate_from_amplitude_error():
    target = np.zeros((3, 7), np.float32)
    target[:, 6] = -1
    original = target.copy()
    original[:, 6] = [1, -1, -1]
    changed = target.copy()
    changed[:, 6] = [-1, 1, -.5]
    changed[0, :3] = .5
    metrics = a.condition_metrics(original, changed, target, np.ones(3, bool))
    assert metrics['sign_error_delta'] == 0
    assert metrics['correct_to_wrong_n'] == metrics['wrong_to_correct_n'] == 1
    assert metrics['translation_mse_delta'] == pytest.approx(1 / 12)
    assert metrics['gripper_mse_delta'] == pytest.approx(1 / 12)
    empty = a.condition_metrics(original, changed, target, np.zeros(3, bool))
    assert empty['n'] == 0 and empty['sign_error_delta'] is None
    assert empty['translation_mse_delta'] is None


def test_joint_flag_retains_all_seeds_and_requires_strict_sign_gain_without_translation_harm():
    rows = [{'sign_error_delta': -.1, 'translation_mse_delta': 0.} for _ in range(3)]
    assert a.descriptive_joint_flag(rows) is True
    assert json.loads(json.dumps(a.descriptive_joint_flag(rows))) is True
    rows[0]['translation_mse_delta'] = .001
    assert a.descriptive_joint_flag(rows) is False
    rows[0] = {'sign_error_delta': 0., 'translation_mse_delta': -.01}
    assert a.descriptive_joint_flag(rows) is False
    rows[0]['sign_error_delta'] = None
    assert a.descriptive_joint_flag(rows) is None
    with pytest.raises(ValueError, match='all3 seeds'):
        a.descriptive_joint_flag(rows[:2])


@pytest.mark.parametrize('field,value', [('checkpoint_updates', 1000), ('closed_loop_stabilization_steps', 0),
    ('component', 'joint_states'), ('conditions', ['identity', 'zero_gripper']),
    ('source_role', 'meta_query'), ('new_environment_calls', 1), ('planned_main_forward_batches', 2458)])
def test_scope_changes_fail_before_provenance_reads(tmp_path, monkeypatch, field, value):
    config = json.loads(a.DEFAULT_CONFIG.read_text())
    config['scope'][field] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('invalid scope reached provenance reads'))
    with pytest.raises(ValueError, match='scientific scope differs'):
        a.validate(path)


def test_unbound_operator_change_fails_config_hash(tmp_path):
    config = json.loads(a.DEFAULT_CONFIG.read_text())
    config['analysis']['immutable_anchors'] = 'cascade replacements'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='design hash differs'):
        a.validate(path)


def test_freeze_cannot_claim_an_executed_result(tmp_path, monkeypatch):
    config = json.loads(a.DEFAULT_CONFIG.read_text())
    config['research_run_executed'] = True
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('execution claim reached provenance reads'))
    with pytest.raises(ValueError, match='executed result'):
        a.validate(path)


def test_protocol_import_uses_no_policy_data_or_simulator_libraries():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'h5py', 'libero', 'robosuite', 'mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import validate_libero_source_internal_gripper_state_control_protocol
'''
    subprocess.run([sys.executable, '-c', code], cwd=a.ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
