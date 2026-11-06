import json
import subprocess
import sys

import numpy as np
import pytest

from experiments import analyze_libero_task7_expert_camera_component_tradeoff as a


def summary(t, g, mask):
    t, g = np.array(t, float), np.array(g, float)
    zero = np.zeros_like(t)
    delta = {'translation': t, 'rotation': zero, 'gripper': g, 'all7': (3 * t + g) / 7}
    return a.group_summary(np.array(mask, bool), delta, {'translation': zero, 'gripper': zero},
        {'translation': zero, 'gripper': zero}, zero, zero, np.ones_like(t), np.zeros(len(t), np.int8))


def test_pooled_tradeoff_does_not_imply_any_same_row_tradeoff():
    t, g = np.array([-.3, .1]), np.array([-.1, .3])
    assert t.mean() < 0 and g.mean() > 0
    assert a.direction_codes(t, g).tolist() == [0, 8]
    assert not np.any(a.direction_codes(t, g) == 2)


def test_nine_direction_cells_cover_zero_axes_and_reject_nonfinite():
    t = np.repeat([-1., 0., 1.], 3)
    g = np.tile([-1., 0., 1.], 3)
    assert a.direction_codes(t, g).tolist() == list(range(9))
    with pytest.raises(ValueError, match='direction'):
        a.direction_codes(np.array([np.nan]), np.array([1.]))


def test_partition_reconstructs_global_contributions_not_sum_of_group_means():
    t, g = [-.6, .1, .1, .1], [.6, -.1, -.1, -.1]
    first, rest, total = [summary(t, g, mask) for mask in ([1, 0, 0, 0], [0, 1, 1, 1], [1, 1, 1, 1])]
    a.validate_partition({'first': first, 'rest': rest}, total)
    assert first['delta_mse']['translation']['mean'] + rest['delta_mse']['translation']['mean'] != pytest.approx(total['delta_mse']['translation']['mean'])
    assert first['global_signed_contribution']['translation'] + rest['global_signed_contribution']['translation'] == pytest.approx(-.075)
    assert first['positive_gripper_harm_share'] == first['translation_benefit_share'] == 1.
    with pytest.raises(ValueError, match='partition row'):
        a.validate_partition({'duplicated': total, 'again': total}, total)


def test_empty_groups_and_zero_denominators_remain_explicit():
    group = summary([0., 0.], [0., 0.], [False, False])
    assert group['n'] == 0
    assert group['delta_mse']['translation']['mean'] is None
    assert group['delta_mse']['gripper']['median'] is None
    assert group['positive_gripper_harm_share'] is None
    assert group['translation_benefit_share'] is None
    assert group['global_signed_contribution']['all7'] == 0.


def test_gripper_amplitude_harm_is_separate_from_sign_error():
    base = np.array([-.5, -.1, .2, .5, 0.])
    changed = np.array([-.2, .2, -.4, .8, .1])
    target = np.array([-1., -1., -1., -1., 0.])
    assert a.sign_fates(base, changed, target).tolist() == list(range(5))
    assert (changed[0] - target[0]) ** 2 > (base[0] - target[0]) ** 2
    assert np.sign(base[0]) == np.sign(changed[0])


def label_fixture():
    lengths = [140] * 18 + [139] * 12
    demos = np.repeat(np.arange(30, dtype=np.int64), lengths)
    steps = np.concatenate([np.arange(1, n + 1, dtype=np.int64) for n in lengths])
    target = np.zeros((4188, 7), np.float32)
    target[:, 6] = np.where((steps >= 3) & (steps <= 4), 1, -1)
    previous = np.zeros(4188, np.float32)
    previous[:] = np.where((steps >= 4) & (steps <= 5), 1, -1)
    masks = np.column_stack([np.ones(4188, bool), steps == 1, target[:, 6] < 0,
        target[:, 6] > 0, np.zeros(4188, bool), steps <= 8, steps <= 10])
    return {'target_actions': target, 'previous_expert_gripper': previous, 'demonstration_indices': demos,
        'action_indices': steps, 'command_masks': masks, 'mask_names': np.array(a.STRATA), 'condition_names': np.array(a.CONDITIONS)}


def test_overlap_partition_does_not_double_count_existing_windows():
    arrays = label_fixture()
    _, windows = a.validate_labels(arrays)
    assert np.sum(windows == 3) == 30 * 8
    assert np.sum(windows == 1) == 30 * 2
    assert np.sum(windows == 2) == 0
    assert sum(np.sum(windows == i) for i in range(4)) == 4188
    assert arrays['command_masks'][:, 5].sum() + arrays['command_masks'][:, 6].sum() > np.sum(windows != 0)


def test_changed_cached_window_or_t0_indices_are_rejected():
    arrays = label_fixture()
    arrays['command_masks'][0, 6] = False
    with pytest.raises(ValueError, match='command masks'):
        a.validate_labels(arrays)
    arrays = label_fixture()
    arrays['action_indices'][0] = 0
    with pytest.raises(ValueError, match='t0 exclusion'):
        a.validate_labels(arrays)


@pytest.mark.parametrize('field,value', [('checkpoint_updates', 1000), ('source_role', 'meta_query'),
                                       ('new_policy_forwards', 1), ('task_ids', [4, 7])])
def test_scope_changes_rejected_before_source_access(tmp_path, monkeypatch, field, value):
    scope = dict(a.SCOPE)
    scope[field] = value
    p = tmp_path / 'bad.json'
    p.write_text(json.dumps({'check_type': 'libero_task7_cached_component_tradeoff', 'formal_result': False,
                            'scope': scope, 'analysis': a.ANALYSIS}))
    monkeypatch.setattr(a, 'sha', lambda _: pytest.fail('source hash read after rejected scope'))
    with pytest.raises(ValueError, match='read-only scope'):
        a.load_protocol(p)


def test_existing_outputs_are_preserved_before_loading_inputs(tmp_path, monkeypatch):
    p = tmp_path / 'old.json'
    p.write_text('preserved')
    monkeypatch.setattr(a, 'load_protocol', lambda _: pytest.fail('existing output reached inputs'))
    with pytest.raises(ValueError, match='already exists'):
        a.run(tmp_path / 'unused', p, tmp_path / 'memberships.npz')
    assert p.read_text() == 'preserved'


def test_readonly_import_has_no_policy_dataset_or_environment_dependency():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'h5py', 'libero', 'mujoco', 'robosuite', 'gym'}
    assert not name.startswith('src.')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from experiments import analyze_libero_task7_expert_camera_component_tradeoff
"""
    subprocess.run([sys.executable, '-c', code], cwd=a.ROOT, check=True, text=True, capture_output=True)
