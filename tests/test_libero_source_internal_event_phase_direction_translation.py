from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_source_internal_event_phase_direction_translation as a


def event_fixture(signs):
    actions = np.zeros((len(signs), 7), np.float32)
    actions[:, 6] = signs
    masks, _ = a.parent.expert_masks(actions)
    return actions[1:], np.zeros(len(signs) - 1, np.int16), np.arange(1, len(signs), dtype=np.int32), masks


def test_event_before_at_after_and_same_type_tie_are_unique():
    signs = np.ones(31, np.float32)
    signs[10:15] = -1
    signs[20:] = -1
    target, ids, times, masks = event_fixture(signs)
    assigned = a.assign_events(target, ids, times, masks['near_open'][1:], 'open')
    row = int(np.flatnonzero(times == 15)[0])
    assert assigned['event_step'][row] == 10 and assigned['lag'][row] == 5
    assert assigned['phases']['post'][row]
    assert sum(int(x.sum()) for x in assigned['phases'].values()) == int(masks['near_open'].sum()) == 21
    assert len(assigned['known_events']) == 2


def test_missing_action0_never_gets_a_guessed_event_label():
    target = np.zeros((8, 7), np.float32)
    target[:, 6] = -1
    times = np.arange(1, 9)
    mask = times <= 6
    assigned = a.assign_events(target, np.zeros(8, int), times, mask, 'open')
    assert assigned['known_events'] == []
    assert assigned['phases']['unresolved_boundary'].sum() == 6
    assert not assigned['phases']['pre'].any() and np.all(assigned['event_step'] == -1)


def test_nonboundary_unmatched_and_missing_known_windows_reject():
    target, ids, times, masks = event_fixture([1] * 10 + [-1] * 10)
    bad = masks['near_open'][1:].copy()
    bad[:] = False
    with pytest.raises(ValueError, match='known event window'):
        a.assign_events(target, ids, times, bad, 'open')
    target[:, 6] = -1
    bad[-1] = True
    with pytest.raises(ValueError, match='nonboundary'):
        a.assign_events(target, ids, times, bad, 'open')


def test_end_clipping_does_not_create_unobserved_post_frames():
    target, ids, times, masks = event_fixture([1] * 10 + [-1])
    out = a.assign_events(target, ids, times, masks['near_open'][1:], 'open')
    assert out['phases']['pre'].sum() == 5 and out['phases']['at'].sum() == 1
    assert out['phases']['post'].sum() == 0


def test_noncontiguous_frame_ids_reject():
    target, ids, times, masks = event_fixture([1] * 10 + [-1] * 10)
    times[-1] += 1
    with pytest.raises(ValueError, match='contiguous'):
        a.assign_events(target, ids, times, masks['near_open'][1:], 'open')


def test_neutral_mismatches_are_retained_in_four_way_partition():
    truth = np.array([-1, 1, 0, -1, 1])
    guessed = np.array([1, -1, 1, 0, 1])
    out = a.direction_masks(truth, guessed)
    assert [int(out[k].sum()) for k in a.ALL_CASES] == [1, 1, 2, 1]


def test_translation_control_matches_expert_command_not_all_correct_frames():
    truth, guessed = np.array([-1, -1, 1, 1]), np.array([1, -1, -1, 1])
    result = a.phase_summary(np.array([9., 1., 100., 100.]), truth, a.direction_masks(truth, guessed),
                             np.ones(4, bool), np.zeros(4, int))
    assert result['directions'][a.DIRECTIONS[0]]['translation_difference'] == 8
    assert result['directions'][a.DIRECTIONS[1]]['translation_difference'] == 0
    assert result['directions'][a.DIRECTIONS[0]]['conditional_error_rate'] == .5


def test_absent_expert_class_has_null_rate_not_zero():
    truth, guessed = np.ones(3), np.array([-1, -1, 1])
    row = a.phase_summary(np.ones(3), truth, a.direction_masks(truth, guessed), np.ones(3, bool), np.zeros(3, int))
    assert row['directions'][a.DIRECTIONS[0]]['expert_class_n'] == 0
    assert row['directions'][a.DIRECTIONS[0]]['conditional_error_rate'] is None
    assert row['directions'][a.DIRECTIONS[1]]['conditional_error_rate'] == 2 / 3


def test_within_demo_comparison_does_not_impute_between_demo_difference():
    truth, guessed = -np.ones(4), np.array([1, 1, -1, -1])
    row = a.phase_summary(np.array([9., 9., 1., 1.]), truth, a.direction_masks(truth, guessed),
                          np.ones(4, bool), np.array([0, 0, 1, 1]))
    m = row['directions'][a.DIRECTIONS[0]]
    assert m['translation_difference'] == 8 and m['per_demo_difference']['n'] == 0
    assert m['per_demo_difference']['missing'] == 2 and m['per_demo_difference']['mean'] is None


def test_cross_seed_translation_contrast_counts_each_same_frame_once():
    truth = -np.ones(3)
    guessed = np.array([[1, 1, -1], [1, -1, 1], [1, -1, -1]])
    cases = [a.direction_masks(truth, g) for g in guessed]
    assigned = {'event_step': np.array([10, 10, 10]), 'lag': np.array([0, 1, 2])}
    row = a.cross_seed_summary(np.array([[5., 9., 1.], [6., 1., 8.], [7., 3., 2.]]), truth, cases,
                               np.ones(3, bool), np.zeros(3, int), np.array([10, 11, 12]), assigned, a.DIRECTIONS[0])
    assert row['all_three_intersection_n'] == 1 and row['any_seed_union_n'] == 3
    assert row['all_three_jaccard'] == 1 / 3
    assert row['same_frame_mixed_n'] == 2
    assert [r['difference'] for r in row['same_frame_translation_contrasts']] == [7., 6.5]
    assert row['same_frame_translation_difference']['mean'] == 6.75


def test_empty_cross_seed_error_union_is_null_and_keeps_missing_counts():
    truth = -np.ones(3)
    cases = [a.direction_masks(truth, truth) for _ in range(3)]
    row = a.cross_seed_summary(np.zeros((3, 3)), truth, cases, np.ones(3, bool), np.zeros(3, int),
                               np.arange(1, 4), {'event_step': np.full(3, -1), 'lag': np.full(3, 99)}, a.DIRECTIONS[0])
    assert row['all_three_jaccard'] is None and row['same_frame_translation_difference']['mean'] is None
    assert row['common_frames'] == [] and row['same_frame_mixed_n'] == 0


def test_all_arms_phases_contexts_and_lags_are_retained():
    target, ids, times, masks = event_fixture([-1] * 5 + [1] * 10 + [-1] * 17)
    data = {'targets': target, 'demonstration_indices': ids, 'action_steps': times,
            'predictions': np.stack([target] * 9), 'model_ids': np.array([f'{s}:{arm}' for s in a.SEEDS for arm in a.ARMS]),
            'mask_near_open': masks['near_open'][1:], 'mask_near_close': masks['near_close'][1:]}
    before = data['predictions'].copy()
    result = a.analyze_task(0, data)
    assert np.array_equal(data['predictions'], before)
    assert len(result['phase_rows']) == 144 and len(result['lag_rows']) == 396 and len(result['cross_seed_rows']) == 96
    assert all(r['all_three_positive_translation_association'] is None for r in result['cross_seed_rows'])
    assert {r['phase'] for r in result['phase_rows']} == set(a.PHASES)
    assert {r['context'] for r in result['phase_rows']} == set(a.CONTEXTS)


@pytest.mark.parametrize('field,value', [('new_policy_forward_batches', 1), ('checkpoint_updates', 1000)])
def test_changed_budget_rejects_before_provenance_io(tmp_path, monkeypatch, field, value):
    cfg = json.loads(a.DEFAULT_CONFIG.read_text())
    cfg['scope'][field] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('invalid scope reached pin reads'))
    with pytest.raises(ValueError, match='definition differs'):
        a.load_inputs(path, tmp_path / 'no-binding')


def test_unbound_phase_definition_change_rejects(tmp_path):
    cfg = json.loads(a.DEFAULT_CONFIG.read_text())
    cfg['analysis']['assignment'] = 'choose_event_using_prediction_error'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='config hash'):
        a.load_inputs(path, tmp_path / 'no-binding')


def test_existing_output_is_preserved_before_input_processing(tmp_path, monkeypatch):
    path = tmp_path / 'old.json'
    path.write_text('preserved')
    monkeypatch.setattr(a, 'load_inputs', lambda *args: pytest.fail('existing output reached inputs'))
    with pytest.raises(ValueError, match='already exists'):
        a.run(a.DEFAULT_CONFIG, tmp_path / 'no-binding', path)
    assert path.read_text() == 'preserved'


def test_readonly_module_imports_no_model_data_or_simulator_libraries():
    code = '''
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from experiments import analyze_libero_source_internal_event_phase_direction_translation
'''
    subprocess.run([sys.executable, '-c', code], cwd=a.ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
