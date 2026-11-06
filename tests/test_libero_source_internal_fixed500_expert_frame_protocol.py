from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import validate_libero_source_internal_fixed500_expert_frame_protocol as p
from experiments import run_libero_source_internal_fixed500_expert_frame_diagnostic as r


def actions(signs):
    result = np.zeros((len(signs), 7), np.float32)
    result[:, 6] = signs
    return result


def scored(signs, pred=None):
    full = actions(signs)
    masks, atoms = p.expert_masks(full)
    target = full[1:]
    if pred is None:
        pred = target.copy()
    return p.analyze_predictions(pred, target, np.zeros(len(target), np.int16), {k: v[1:] for k, v in masks.items()})


def metadata():
    lengths = {0: [81] + [96] * 23 + [106] * 6,
               1: [81] + [10 * w - (6 if i < 16 else 5) + 1 for i, w in enumerate([14] * 21 + [15] * 8)]}
    tasks = []
    for t in p.TASKS:
        demos = [{'demonstration_index': i, 'actions_shape': [n, 7], 'actions_dtype': 'float64',
                  'observations': {name: {'shape': [n, *shape], 'dtype': 'uint8' if name.endswith('_rgb') else 'float64'}
                                   for name, shape in p.OBS_SHAPES.items()}} for i, n in enumerate(lengths[t])]
        tasks.append({'task_id': t, 'demonstrations': demos,
                      'access_record': {'task_role': 'source_spi', 'demonstration_role': 'meta_support',
                                        'demonstration_indices': list(range(30))}})
    return {'tasks': tasks}


def test_action0_labels_transition_but_is_never_scored():
    masks, atoms = p.expert_masks(actions([-1, 1, 1, 1, 1, 1, 1, 1]))
    assert all(not mask[0] for mask in masks.values()) and atoms[0] == -1
    assert np.flatnonzero(masks['near_close']).tolist() == list(range(1, 7))
    assert not masks['near_open'].any()


def test_initial_positive_without_switch_is_not_invented_as_close_event():
    masks, _ = p.expert_masks(actions([1] * 35))
    assert not masks['near_close'].any()
    assert np.flatnonzero(masks['early_t1_to20']).tolist() == list(range(1, 21))
    assert np.flatnonzero(masks['late_t21_plus']).tolist() == list(range(21, 35))


def test_overlapping_windows_have_exhaustive_disjoint_atomic_partition():
    signs = [-1] * 30
    signs[10:12] = [1, 1]
    masks, atoms = p.expert_masks(actions(signs))
    assert masks['near_close'][10] and masks['near_open'][10]
    assert atoms[10] == 7
    assert np.all(sum(masks[a].astype(int) for a in p.ATOMS)[1:] == 1)
    assert not masks['near_open'][29]


def test_neutral_to_signed_commands_count_as_frozen_transitions():
    masks, _ = p.expert_masks(actions([0, 1, 0, -1, 0]))
    assert masks['near_close'][1] and masks['near_open'][3]
    assert masks['expert_neutral'][2] and not masks['expert_neutral'][0]


def test_more_frames_alone_do_not_create_concentration():
    target = actions([-1] * 41)[1:]
    pred = target.copy()
    pred[:, 0] = .5
    result = scored([-1] * 41, pred)
    c = result['concentration']['early_t1_to20']
    assert c['n'] == c['n_complement'] == 20
    assert c['metrics']['translation_mse']['concentration_ratio'] == pytest.approx(1)
    assert c['metrics']['translation_mse']['mean_difference'] == 0


def test_error_share_is_normalized_by_window_frame_share():
    target = actions([-1] * 41)[1:]
    pred = target.copy()
    pred[:20, 6] = 1
    result = scored([-1] * 41, pred)
    m = result['concentration']['early_t1_to20']['metrics']['gripper_sign_error']
    assert m == {'inside_mean': 1., 'outside_mean': 0., 'mean_difference': 1.,
                 'error_sum_share': 1., 'concentration_ratio': 2.}
    assert result['strata']['all']['gripper_sign_confusion'] == [[20, 0, 20], [0, 0, 0], [0, 0, 0]]


def test_empty_windows_and_zero_error_totals_have_null_ratios():
    result = scored([-1] * 41)
    empty = result['concentration']['near_close']['metrics']['gripper_mse']
    assert empty['inside_mean'] is None and empty['mean_difference'] is None
    assert empty['error_sum_share'] is None and empty['concentration_ratio'] is None
    assert result['strata']['near_close']['n'] == 0
    assert result['demo_equal']['near_close']['gripper_mse']['missing_demos'] == 1


def test_component_identity_and_sign_error_do_not_conflate_amplitude():
    target = actions([1] * 5)
    pred = target.copy()
    pred[:, :3] = .5
    pred[:, 3:6] = .25
    pred[:, 6] = .01
    errors, _ = p.row_errors(pred, target)
    assert np.all(errors['gripper_sign_error'] == 0) and np.all(errors['gripper_mse'] > .9)
    np.testing.assert_allclose(errors['mse'], (3 * errors['translation_mse'] +
                                             3 * errors['rotation_mse'] + errors['gripper_mse']) / 7, atol=1e-12)


def test_metadata_exact_batch_and_valid_padding_row_budget():
    b = p.metadata_budget(metadata())
    assert sum(x['batches'] for x in b.values()) * 9 == 819
    assert sum(x['valid_rows'] for x in b.values()) * 9 == 62586
    assert sum(x['network_slots'] for x in b.values()) * 9 == 65340
    assert sum(x['repeat_rows'] for x in b.values()) * 9 == 1440
    plan = p.window_plan([3] * 30)
    assert len(plan) == 30 and all(x['start'] == 1 and x['valid_length'] == 2 for x in plan)


@pytest.mark.parametrize('kind', ['length', 'role', 'shape', 'dtype'])
def test_metadata_mismatch_cannot_silently_change_budget(kind):
    m = metadata()
    if kind == 'length':
        m['tasks'][0]['demonstrations'][0]['actions_shape'][0] += 1
    elif kind == 'role':
        m['tasks'][0]['access_record']['demonstration_indices'] = list(range(30, 50))
    elif kind == 'shape':
        m['tasks'][0]['demonstrations'][0]['observations']['joint_states']['shape'][1] = 8
    else:
        m['tasks'][0]['demonstrations'][0]['observations']['agentview_rgb']['dtype'] = 'float32'
    with pytest.raises(ValueError):
        p.metadata_budget(m)


@pytest.mark.parametrize('key,value', [('checkpoint_updates', 1000), ('source_role', 'meta_query'),
                                       ('planned_main_forward_batches', 820), ('new_optimizer_updates', 1),
                                       ('qualification_accessed', True), ('default_closed_loop_stabilization_steps', 0)])
def test_budget_role_or_training_change_rejects_before_provenance_io(tmp_path, monkeypatch, key, value):
    config = json.loads(p.DEFAULT_CONFIG.read_text())
    config['scope'][key] = value
    path = tmp_path / 'changed.json'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(p, 'sha', lambda _: pytest.fail('invalid design reached provenance IO'))
    with pytest.raises(ValueError, match='scope differs'):
        p.validate(path)


def test_all_seeds_and_empty_concentration_flags_are_preserved():
    numerical = scored([-1] * 41)
    rows = [{'task_id': t, 'training_seed': s, 'arm': a, **deepcopy(numerical)}
            for t in p.TASKS for s in p.SEEDS for a in p.ARMS]
    result = p.across_models(rows)
    assert len(result['paired_rows']) == 12 and len(result['descriptive_concentration_flags']) == 72
    near_close = [f for f in result['descriptive_concentration_flags'] if f['window'] == 'near_close']
    assert all(f['all_three_seed_concentration'] is None and len(f['missing_seeds']) == 3 for f in near_close)
    assert all(v is None for row in result['paired_rows'] for v in row['strata']['near_close']['deltas'].values())


def test_historical_accuracy_is_exact_integer_ratio_not_rounded_complement():
    target = actions([-1] * 4)[1:]
    pred = target.copy()
    pred[-1, 6] = 1
    summary = scored([-1] * 4, pred)['strata']['all']
    old = {'valid_steps': 3, 'gripper_sign_accuracy': 2 / 3, **summary['metric_means']}
    r.historical_parity(summary, old)
    old['mse'] += 1e-8
    with pytest.raises(ValueError, match='parity'):
        r.historical_parity(summary, old)


def test_old_outputs_reject_before_binding_and_policy_access(tmp_path, monkeypatch):
    output = tmp_path / 'old.json'
    output.write_text('preserved')
    monkeypatch.setattr(p, 'validate', lambda *a: pytest.fail('old output reached preflight'))
    with pytest.raises(ValueError, match='already exist'):
        r.run(p.DEFAULT_CONFIG, tmp_path / 'binding', output, tmp_path / 'artifacts')
    assert output.read_text() == 'preserved'


@pytest.mark.parametrize('bad', [float('nan'), 1.001])
def test_nonfinite_or_out_of_interface_actions_are_rejected(bad):
    pred = actions([-1] * 3)
    pred[0, 0] = bad
    with pytest.raises(ValueError):
        p.row_errors(pred, actions([-1] * 3))


def test_import_does_not_open_learning_data_or_simulator_packages():
    code = '''
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch','h5py','libero','robosuite','mujoco'}: raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import run_libero_source_internal_fixed500_expert_frame_diagnostic
'''
    subprocess.run([sys.executable, '-c', code], cwd=p.ROOT, check=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''}, capture_output=True, text=True)
