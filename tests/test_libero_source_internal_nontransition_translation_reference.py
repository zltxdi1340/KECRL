import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_source_internal_nontransition_translation_reference as a


def original_frame(errors, guessed, expert_sign=-1):
    wrong = [s for s, g in zip(a.SEEDS, guessed) if np.sign(g) == -expert_sign]
    correct = [s for s, g in zip(a.SEEDS, guessed) if np.sign(g) == expert_sign]
    ew = float(np.mean([e for e, g in zip(errors, guessed) if np.sign(g) == -expert_sign]))
    ec = float(np.mean([e for e, g in zip(errors, guessed) if np.sign(g) == expert_sign]))
    return {'demonstration_index': 0, 'action_step': 21, 'event_step': 20, 'lag': 1,
            'wrong_seeds': wrong, 'correct_seeds': correct,
            'wrong_translation_mse': ew, 'correct_translation_mse': ec, 'difference': ew - ec}


def test_fixed_additive_seed_difference_can_remove_raw_association():
    e, guessed = [.75, .25, .25], [1, -1, -1]
    row = a.adjust_frame(original_frame(e, guessed), e, guessed, -1, [.5, 0., 0.])
    assert row['raw_difference'] == .5 and row['reference_gap'] == .5
    assert row['centered_difference'] == 0 and row['identity_residual'] == 0


def test_local_difference_can_remain_after_fixed_reference_centering():
    e, guessed = [.875, .25, .25], [1, -1, -1]
    row = a.adjust_frame(original_frame(e, guessed), e, guessed, -1, [.5, 0., 0.])
    assert row['raw_difference'] == .625 and row['centered_difference'] == .125


def test_sign_reversal_and_negative_residuals_are_not_clipped():
    e, guessed = [.5, .25, .25], [1, -1, -1]
    row = a.adjust_frame(original_frame(e, guessed), e, guessed, -1, [.5, 0., 0.])
    assert row['centered_difference'] == -.25
    summary = a.summarize_adjusted([row], [0, 1])
    assert summary['mean_sign_reversal'] is True and summary['centered_positive_mean'] is False
    assert summary['per_demo'][1]['centered']['mean'] is None


def test_identical_references_cancel_between_seed_groups():
    e, guessed = [.75, .25, .5], [1, -1, -1]
    row = a.adjust_frame(original_frame(e, guessed), e, guessed, -1, [.125, .125, .125])
    assert row['reference_gap'] == 0 and row['centered_difference'] == row['raw_difference']


def test_multiple_wrong_seeds_form_one_group_mean_per_frame():
    e, guessed = [.5, .75, .25], [1, 1, -1]
    row = a.adjust_frame(original_frame(e, guessed), e, guessed, -1, [.25, .5, .125])
    assert row['raw_difference'] == .375 and row['reference_gap'] == .25 and row['centered_difference'] == .125


def test_neutral_seed_is_not_fabricated_as_correct_or_directional_error():
    e, guessed = [.75, .25, .5], [1, -1, 0]
    row = a.adjust_frame(original_frame(e, guessed), e, guessed, -1, [.5, 0., None])
    assert row['centered_difference'] == 0 and row['missing_reference_seeds'] == []


def test_missing_required_reference_keeps_raw_frame_without_fallback():
    e, guessed = [.75, .25, .25], [1, -1, -1]
    row = a.adjust_frame(original_frame(e, guessed), e, guessed, -1, [.5, None, 0.])
    assert row['raw_difference'] == .5 and row['centered_difference'] is None
    assert row['missing_reference_seeds'] == [a.SEEDS[1]]
    summary = a.summarize_adjusted([row], [0])
    assert summary['raw']['n'] == 1 and summary['centered']['n'] == 0 and summary['centered']['missing'] == 1
    assert summary['raw_positive_mean'] is None and summary['centered_positive_mean'] is None


def test_secondary_missing_expert_class_is_not_replaced_with_global_mean():
    ref = {'global': {'mean': .5}, 'by_expert_sign': {'-1': {'mean': None}}}
    assert a.reference_value(ref, a.MODES[0], -1) == .5
    assert a.reference_value(ref, a.MODES[1], -1) is None


def test_references_use_only_fixed_mask_and_retain_empty_neutral_class():
    target = np.zeros((4, 7), np.float32)
    target[:, 6] = [-1, -1, 1, 1]
    predictions = np.stack([target.copy() for _ in range(3)])
    for i in range(3):
        predictions[i, :, :3] = (i + 1) / 4
        predictions[i, 1, :3] = 1  # A large excluded-frame error must not enter the reference.
    refs, errors = a.compute_references(predictions, target, np.array([1, 0, 1, 0], bool),
                                       [f'{s}:A_original' for s in a.SEEDS])
    for i, seed in enumerate(a.SEEDS):
        ref = refs[seed, 'A_original']
        assert ref['global']['n'] == 2 and ref['global']['mean'] == ((i + 1) / 4) ** 2
        assert ref['global']['centered_mean'] == 0
        assert ref['by_expert_sign']['0']['n'] == 0 and ref['by_expert_sign']['0']['mean'] is None


def test_empty_primary_reference_stops_instead_of_imputing_zero():
    with pytest.raises(ValueError, match='nonempty'):
        a.compute_references(np.zeros((1, 2, 7), np.float32), np.zeros((2, 7), np.float32),
                              np.zeros(2, bool), [f'{a.SEEDS[0]}:A_original'])


@pytest.mark.parametrize('kind', ['seed_sets', 'raw_value', 'nonfinite_reference'])
def test_original_contrast_identity_and_seed_sets_fail_closed(kind):
    e, guessed = [.75, .25, .25], [1, -1, -1]
    record, ref = original_frame(e, guessed), [.5, 0., 0.]
    if kind == 'seed_sets':
        record['correct_seeds'] = record['correct_seeds'][::-1]
    elif kind == 'raw_value':
        record['difference'] += .01
    else:
        ref[0] = float('nan')
    with pytest.raises(ValueError):
        a.adjust_frame(record, e, guessed, -1, ref)


def test_same_seed_same_command_contrast_is_algebraically_unchanged():
    wrong, correct, baseline = .75, .25, 1.
    assert (wrong - baseline) - (correct - baseline) == wrong - correct


def test_empty_groups_keep_missing_demo_information():
    row = a.summarize_adjusted([], list(range(30)))
    assert row['raw']['n'] == row['centered']['n'] == 0
    assert row['raw_positive_mean'] is row['centered_positive_mean'] is None
    assert row['equal_demo_centered']['missing'] == 30


def test_complete_synthetic_task_retains_all_prior_groups_and_both_modes():
    full = np.zeros((41, 7), np.float32)
    full[:, 6] = [-1] * 10 + [1] * 10 + [-1] * 21
    masks, _ = a.phase.parent.expert_masks(full)
    target = full[1:]
    data = {'targets': target, 'predictions': np.stack([target] * 9),
            'model_ids': np.array([f'{s}:{arm}' for s in a.SEEDS for arm in a.ARMS]),
            'demonstration_indices': np.zeros(40, np.int16), 'action_steps': np.arange(1, 41),
            'mask_near_open': masks['near_open'][1:], 'mask_near_close': masks['near_close'][1:],
            'mask_away_from_transitions': masks['away_from_transitions'][1:]}
    original = a.phase.analyze_task(0, data)['cross_seed_rows']
    result = a.analyze_task(0, data, original)
    assert len(result['references']) == 9 and len(result['adjusted_groups']) == 192
    assert all(r['frames'] == [] and r['centered_positive_mean'] is None for r in result['adjusted_groups'])
    bad = {**data, 'mask_away_from_transitions': np.ones(40, bool)}
    with pytest.raises(ValueError, match='fixed complement'):
        a.analyze_task(0, bad, original)


@pytest.mark.parametrize('field,value', [('checkpoint_updates', 1000), ('closed_loop_stabilization_steps', 0),
                                       ('new_policy_forward_batches', 1), ('loss_observation_changed', True)])
def test_scope_changes_stop_before_provenance_reads(tmp_path, monkeypatch, field, value):
    config = json.loads(a.DEFAULT_CONFIG.read_text())
    config['scope'][field] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(a, 'sha', lambda *args: pytest.fail('invalid scope reached pin IO'))
    with pytest.raises(ValueError, match='definition differs'):
        a.load_inputs(path, tmp_path / 'no-binding')


def test_unbound_reference_change_is_rejected(tmp_path):
    config = json.loads(a.DEFAULT_CONFIG.read_text())
    config['analysis']['centering'] = 'clip_negative_residuals_to0'
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='config hash'):
        a.load_inputs(path, tmp_path / 'no-binding')


def test_historical_output_is_preserved_before_input_access(tmp_path, monkeypatch):
    output = tmp_path / 'old.json'
    output.write_text('preserved')
    monkeypatch.setattr(a, 'load_inputs', lambda *args: pytest.fail('old output reached inputs'))
    with pytest.raises(ValueError, match='already exists'):
        a.run(a.DEFAULT_CONFIG, tmp_path / 'binding', output)
    assert output.read_text() == 'preserved'


def test_import_requires_no_learning_data_or_simulator_packages():
    code = '''
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}:raise AssertionError(name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from experiments import analyze_libero_source_internal_nontransition_translation_reference
'''
    subprocess.run([sys.executable, '-c', code], cwd=a.ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
