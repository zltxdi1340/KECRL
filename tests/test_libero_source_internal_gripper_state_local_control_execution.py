from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import run_libero_source_internal_gripper_state_local_control as r
from experiments import audit_libero_source_internal_gripper_state_local_control as audit
from tests.test_libero_source_internal_gripper_state_control_protocol import sample_fixture


def test_original_batch_layout_marks_padding_and_preserves_frame_order():
    _, sample = sample_fixture()
    batches = r.batch_frame_rows(sample)
    assert len(batches) == 1 and batches[0].shape == (2, 10)
    np.testing.assert_array_equal(batches[0][0], np.arange(10))
    np.testing.assert_array_equal(batches[0][1], [10, 11, 12, 13, -1, -1, -1, -1, -1, -1])


def test_changed_gripper_does_not_use_negative_padding_as_source_index():
    rows = np.array([[0, 1, -1]], np.int32)
    old = np.array([[[1., 2.], [3., 4.], [0., 0.]]], np.float32)
    changed = np.array([[10., 11.], [12., 13.]], np.float32)
    result = r.condition_batch_gripper(old, rows, changed)
    np.testing.assert_array_equal(result, [[[10., 11.], [12., 13.], [0., 0.]]])
    np.testing.assert_array_equal(old, [[[1., 2.], [3., 4.], [0., 0.]]])
    with pytest.raises(ValueError, match='invalid batch source'):
        r.condition_batch_gripper(old, np.array([[0, 2, -1]]), changed)


def test_frozen_strata_come_from_original_arm_and_all_three_seeds():
    data, sample = sample_fixture()
    selected = [v['source_row'] for v in sample['selected']]
    preds = data['predictions']
    preds[[1, 4, 7], selected[0], 6] = 1
    preds[1, selected[1], 6] = 1
    preds[4, selected[2], 6] = 0
    masks = r.fixed_masks(preds, 1, data['targets'], sample)
    assert masks['all_three_wrong_close'][selected[0]]
    assert masks['same_frame_mixed_seeds'][selected[1]]
    assert not masks['same_frame_mixed_seeds'][selected[2]]
    assert masks['original_wrong_close'].sum() == 2
    assert masks['original_correct'].sum() == 3
    assert sum(masks[f'lag_{i}'].sum() for i in range(1, 6)) == 5


def test_summary_retains_harmed_original_correct_rows_and_empty_demos():
    data, sample = sample_fixture()
    model = 1
    selected = [v['source_row'] for v in sample['selected']]
    predictions = np.stack([data['predictions'][model]] * 3)
    predictions[1, selected[0], 6] = 1
    result = r.analyze_model(data['predictions'], model, predictions, data['targets'], data['demonstration_indices'], sample)
    main = result[r.p.CONDITIONS[1]]
    assert main['selected_all']['sign_error_delta'] == .2
    assert main['original_correct']['correct_to_wrong_n'] == 1
    assert main['original_wrong_close']['n'] == 0
    assert main['selected_all']['per_demo']['1']['n'] == 0
    assert main['selected_all']['per_demo']['1']['sign_error_delta'] is None
    assert main['selected_all']['equal_demo']['sign_error_delta']['missing'] == 29
    assert main['original_correct']['n'] == 5


def test_joint_gate_does_not_allow_secondary_condition_to_rescue_main():
    rows = []
    for task in r.p.TASKS:
        for arm in r.p.ARMS:
            for seed in r.p.SEEDS:
                base = {'n': 2, 'sign_error_delta': -.1, 'translation_mse_delta': 0.,
                        'gripper_mse_delta': -.1, 'correct_to_wrong_n': 0, 'wrong_to_correct_n': 1}
                main = {**base, 'translation_mse_delta': .01 if seed == r.p.SEEDS[0] else 0.}
                rows.append({'task_id': task, 'arm': arm, 'training_seed': seed,
                    'conditions': {r.p.CONDITIONS[1]: {'selected_all': main}, r.p.CONDITIONS[2]: {'selected_all': base}}})
    result = r.summarize_models(rows)
    assert len(result) == 6
    assert all(row['primary_joint_cue'] is False and row['training_promotion'] is False for row in result)
    assert all(row['conditions'][r.p.CONDITIONS[2]]['joint_cue'] is True for row in result)


@pytest.mark.parametrize('repeat,maximum', [(False, 2457), (True, 54)])
def test_attempts_count_before_call_and_exhausted_budget_prevents_call(repeat, maximum):
    counts = r.new_counts()
    key = 'repeat' if repeat else 'main'
    events = []
    class FakePolicy:
        calls = 0
        def forward_sequence(self, *args):
            assert counts[key + '_forward_batches'] == 1
            self.calls += 1
            return np.zeros((1, 1, 7), np.float32), None
    policy = FakePolicy()
    r.execute_forward(policy, [], counts, repeat, lambda kind, **kw: events.append(kind), {})
    assert policy.calls == counts[key + '_completed_batches'] == 1
    assert events == [key + '_forward_attempt']
    counts[key + '_forward_batches'] = maximum
    with pytest.raises(ValueError, match='budget exhausted'):
        r.execute_forward(policy, [], counts, repeat, lambda *args, **kw: None, {})
    assert policy.calls == 1


def test_failed_forward_attempt_remains_counted():
    counts = r.new_counts()
    class FailingPolicy:
        def forward_sequence(self, *args):
            raise RuntimeError('synthetic error')
    with pytest.raises(RuntimeError, match='synthetic error'):
        r.execute_forward(FailingPolicy(), [], counts, False, lambda *args, **kw: None, {})
    assert counts['main_forward_batches'] == 1 and counts['main_completed_batches'] == 0


def test_independent_summary_matches_sign_harms_and_translation_delta():
    target = np.zeros((3, 7), np.float32)
    target[:, 6] = -1
    before = target.copy()
    before[0, 6] = 1
    after = target.copy()
    after[1, 6] = 1
    after[2, 6] = -.5
    after[0, :3] = .5
    expected = audit.scalar_summary(before, after, target, np.ones(3, bool))
    observed = r.metric_summary(before, after, target, np.ones(3, bool))
    assert audit.compare(expected, observed) > 20
    assert expected['correct_to_wrong_n'] == expected['wrong_to_correct_n'] == 1
    assert expected['sign_error_delta'] == 0
    empty = audit.scalar_summary(before, after, target, np.zeros(3, bool))
    assert empty['translation_mse_delta'] is None and empty['action_response']['mean_l2'] is None
    with pytest.raises(ValueError, match='arithmetic differs'):
        audit.compare(.1, .2)


@pytest.mark.parametrize('variable', list(r.LAUNCH))
def test_launch_environment_is_exact(monkeypatch, variable):
    for k, v in r.LAUNCH.items():
        monkeypatch.setenv(k, v)
    r.launch_gate()
    monkeypatch.setenv(variable, 'different')
    with pytest.raises(ValueError, match='launch environment'):
        r.launch_gate()


def test_existing_outputs_are_preserved_before_provenance_reads(tmp_path, monkeypatch):
    out = tmp_path / 'old.json'
    out.write_text('preserved')
    monkeypatch.setattr(r, 'preflight', lambda *args: pytest.fail('old output reached inputs'))
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, None, out, tmp_path / 'new-artifacts')
    assert out.read_text() == 'preserved'


def test_preflight_failure_is_saved_without_policy_or_source_reads(tmp_path, monkeypatch):
    out = tmp_path / 'new.json'
    artifacts = tmp_path / 'new-artifacts'
    monkeypatch.setattr(r, 'preflight', lambda *args: (_ for _ in ()).throw(ValueError('synthetic binding mismatch')))
    config, binding = tmp_path / 'config.json', tmp_path / 'binding.json'
    config.write_text('{}')
    binding.write_text('{}')
    with pytest.raises(ValueError, match='binding mismatch'):
        r.run(config, None, binding, out, artifacts)
    saved = json.loads((tmp_path / 'new-failed-partial.json').read_text())
    assert saved['status'] == 'failed_partial' and saved['stage'] == 'binding_preflight'
    assert all(v == 0 for v in saved['actual_counts'].values())
    assert not out.exists()
    assert len((artifacts / 'events.jsonl').read_text().splitlines()) == 2


def test_import_never_imports_policy_hdf5_or_simulators():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import run_libero_source_internal_gripper_state_local_control
'''
    subprocess.run([sys.executable, '-c', code], cwd=r.ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
