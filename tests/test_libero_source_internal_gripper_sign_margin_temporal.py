from copy import deepcopy
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_source_internal_gripper_sign_margin_temporal as a
from tests.test_libero_source_internal_gripper_state_control_protocol import sample_fixture


def fixture_values(before_values, after_values):
    data, sample = sample_fixture()
    before = data['targets'].copy()
    after = before.copy()
    rows = [r['source_row'] for r in sample['selected']]
    before[rows, 6] = before_values
    after[rows, 6] = after_values
    for r in sample['selected']:
        r['original_predicted_gripper_signs'][0] = int(np.sign(before[r['source_row'], 6]))
    return data['targets'], before, after, sample


def analyze(values):
    return a.analyze_model(0, a.SEEDS[0], a.ARMS[0], a.CONDITIONS[0], *values)


def test_small_margin_flip_can_extend_a_preexisting_wrong_run():
    result = analyze(fixture_values([-.002, .15, .2, -.8, -.9], [.001, .2, .3, -.8, -.9]))
    harm = [r for r in result['frames'] if r['is_harmed']]
    assert len(harm) == 1 and harm[0]['lag'] == 1
    assert harm[0]['original_margin'] == pytest.approx(.002)
    assert harm[0]['delta_g'] == pytest.approx(.003)
    assert harm[0]['delta_over_original_margin'] == pytest.approx(1.5)
    event = result['events'][0]
    assert event['runs']['original_wrong'][0]['length'] == 2
    assert event['runs']['changed_wrong'][0]['length'] == 3
    assert event['runs']['is_harmed'][0]['length'] == 1
    assert event['runs']['is_harmed'][0]['joins_existing_wrong_run'] is True
    assert result['summary']['cohorts']['wrong_both']['n'] == 2
    assert result['summary']['harmed_runs_joining_existing_wrong_run'] == 1


def test_isolated_new_wrong_run_is_distinct_from_still_wrong_frames():
    result = analyze(fixture_values([-1, -1, -.002, -1, -1], [-1, -1, .001, -1, -1]))
    assert result['summary']['harmed_runs_with_only_new_wrong_frames'] == 1
    assert result['summary']['cohorts']['wrong_both']['n'] == 0
    assert result['events'][0]['runs']['changed_wrong'][0]['length'] == 1
    assert result['summary']['cohorts']['wrong_both']['original_margin']['mean'] is None


def test_no_near_zero_cutoff_discards_a_large_crossing():
    result = analyze(fixture_values([-.9, -.8, -.7, -.6, -.5], [.1, -.8, -.7, -.6, -.5]))
    harm = [r for r in result['frames'] if r['is_harmed']][0]
    assert harm['original_margin'] == pytest.approx(.9)
    assert result['summary']['cohorts']['harmed']['n'] == 1


def test_exact_zero_is_retained_as_error_but_not_positive_close():
    result = analyze(fixture_values([-.01, 0, .2, -.4, .5], [0, .1, 0, -.4, -.5]))
    counts = result['sign_transition9']
    assert counts['neg_to_zero'] == counts['zero_to_pos'] == counts['pos_to_zero'] == 1
    assert counts['pos_to_neg'] == counts['neg_to_neg'] == 1
    cohorts = result['summary']['cohorts']
    assert cohorts['harmed']['n'] == 1 and cohorts['harmed']['changed_positive_close_n'] == 0
    assert cohorts['harmed']['changed_exact_zero_n'] == 1
    assert cohorts['wrong_both']['n'] == 2 and cohorts['repaired']['n'] == 1
    zero_original = next(r for r in result['frames'] if r['original_g'] == 0)
    assert zero_original['delta_over_original_margin'] is None
    assert sum(counts.values()) == 5


def run_frame(step, **overrides):
    row = {'task_id': 0, 'training_seed': a.SEEDS[0], 'arm': a.ARMS[0], 'condition': a.CONDITIONS[0],
           'demonstration_index': 0, 'event_step': 7, 'action_step': step, 'source_row': step - 1,
           'lag': step - 7, 'original_wrong': True, 'changed_wrong': True,
           'is_harmed': False, 'persistent_at_frame': True}
    return {**row, **overrides}


def test_missing_observed_step_never_connects_runs():
    runs = a.contiguous_runs([run_frame(8), run_frame(10)], 'original_wrong')
    assert [r['length'] for r in runs] == [1, 1]
    assert all(r['left_scope_censored'] and r['right_scope_censored'] for r in runs)


@pytest.mark.parametrize('field,value', [('training_seed', a.SEEDS[1]), ('arm', a.ARMS[1]),
    ('condition', a.CONDITIONS[1]), ('demonstration_index', 1), ('event_step', 8), ('task_id', 1)])
def test_runs_cannot_join_model_demo_or_event_scopes(field, value):
    with pytest.raises(ValueError, match='different model/event'):
        a.contiguous_runs([run_frame(8), run_frame(9, **{field: value})], 'original_wrong')


def test_internal_correct_frame_ends_run_and_window_edges_are_censored():
    records = [run_frame(8), run_frame(9, original_wrong=False), run_frame(10)]
    runs = a.contiguous_runs(records, 'original_wrong')
    assert len(runs) == 2
    assert runs[0]['left_scope_censored'] and not runs[0]['right_scope_censored']
    assert not runs[1]['left_scope_censored'] and runs[1]['right_scope_censored']
    assert a.contiguous_runs([], 'original_wrong') == []


def test_duplicate_temporal_frame_rejects():
    with pytest.raises(ValueError, match='duplicate'):
        a.contiguous_runs([run_frame(8), run_frame(8)], 'original_wrong')


def test_empty_selected_event_and_missing_demos_are_retained():
    values = fixture_values([-1] * 5, [-1] * 5)
    values[-1]['selected'] = []
    result = analyze(values)
    assert len(result['events']) == 1 and result['events'][0]['n'] == 0
    assert result['events'][0]['missing_lags'] == [1, 2, 3, 4, 5]
    assert result['summary']['zero_selected_event_windows'] == 1
    assert result['summary']['run_summary']['original_wrong']['length']['max'] is None
    assert len(result['per_demo']) == 30
    assert result['per_demo'][29]['cohorts']['all']['n'] == 0


def test_original_correct_reference_rank_uses_all_original_correct_frames():
    result = analyze(fixture_values([-.002, -.3, -.7, .5, .6], [.001, -.2, -.8, .6, .7]))
    harm = next(r for r in result['frames'] if r['is_harmed'])
    assert harm['original_correct_reference_n'] == 3
    assert harm['original_correct_margin_fraction_le'] == 1 / 3


@pytest.mark.parametrize('corruption', ['nan', 'range', 'dtype', 'label', 'frame_id', 'lag', 'duplicate_selected'])
def test_malformed_frozen_payload_fails_closed(corruption):
    target, before, after, sample = fixture_values([-1] * 5, [-1] * 5)
    if corruption == 'nan':
        after[0, 6] = np.nan
    elif corruption == 'range':
        after[0, 6] = 1.1
    elif corruption == 'dtype':
        after = after.astype(np.float64)
    elif corruption == 'label':
        sample['selected'][0]['original_predicted_gripper_signs'][0] = 1
    elif corruption == 'frame_id':
        sample['source_frame_ids'][sample['selected'][0]['source_row']][0] = 1
    elif corruption == 'lag':
        sample['selected'][0]['lag'] = 6
    else:
        sample['selected'].append(deepcopy(sample['selected'][0]))
    with pytest.raises(ValueError):
        analyze((target, before, after, sample))


@pytest.mark.parametrize('field,value', [('new_policy_forward_batches', 1), ('new_training_updates', 1),
    ('checkpoint_updates', 1000), ('closed_loop_stabilization_steps', 0), ('source_role', 'meta_query')])
def test_changed_budget_or_role_rejects_before_pins(tmp_path, monkeypatch, field, value):
    c = json.loads(a.DEFAULT_CONFIG.read_text())
    c['scope'][field] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(c))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('invalid scope reached provenance I/O'))
    with pytest.raises(ValueError, match='frozen scope differs'):
        a.load_inputs(path, tmp_path / 'binding')


def test_unbound_threshold_change_rejects(tmp_path):
    c = json.loads(a.DEFAULT_CONFIG.read_text())
    c['analysis']['margin'] = 'Choose a favorable cutoff'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(c))
    with pytest.raises(ValueError, match='config hash'):
        a.load_inputs(path, tmp_path / 'binding')


def test_existing_output_is_preserved_before_any_input_reads(tmp_path, monkeypatch):
    out = tmp_path / 'old.json'
    out.write_text('preserved')
    monkeypatch.setattr(a, 'load_inputs', lambda *args: pytest.fail('existing output reached inputs'))
    with pytest.raises(ValueError, match='already exists'):
        a.run(None, None, out)
    assert out.read_text() == 'preserved'


def test_readonly_import_never_imports_policy_hdf5_or_simulators():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import analyze_libero_source_internal_gripper_sign_margin_temporal
'''
    subprocess.run([sys.executable, '-c', code], cwd=a.ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
