"""Synthetic operator and protocol tests; no Policy or research dataset is executed."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments import validate_libero_task4_task7_expert_camera_target_error_protocol as p


def images(flat=False):
    rng = np.random.default_rng(17)
    return {'agentview_rgb': np.full((2, 128, 128, 3), 120, np.uint8) if flat else rng.integers(0, 256, (2, 128, 128, 3), dtype=np.uint8),
            'eye_in_hand_rgb': np.full((2, 128, 128, 3), 60, np.uint8) if flat else rng.integers(0, 256, (2, 128, 128, 3), dtype=np.uint8),
            'joint_states': np.zeros((2, 7), np.float32), 'gripper_states': np.zeros((2, 2), np.float32)}


def design():
    return {'schema_version': 1, 'check_type': 'libero_expert_camera_target_error_protocol', 'formal_result': False,
        'research_run_executed': False, 'protocol_status': 'scientific_design_frozen_execution_binding_pending',
        'scope': deepcopy(p.SCOPE), 'transforms': deepcopy(p.TRANSFORMS), 'analysis': deepcopy(p.ANALYSIS),
        'stop_rules': list(p.GATES), 'execution_binding': {'status': 'pending_runner_implementation_and_pre_run_validation',
            'must_pin': ['this_design_sha256', 'execution_runner_sha256', 'operator_validator_sha256', 'launch_environment'],
            'scientific_design_must_not_change': True, 'actual_experiment_result_is_separate': True}}


def test_matched_controls_share_pixel_mae_across_both_cameras():
    obs = images()
    before = {k: v.copy() for k, v in obs.items()}
    conditions, diagnostics = p.prepare_interventions(obs)
    assert list(conditions) == p.CONDITIONS
    assert all(diagnostics['active_rows'])
    for name, control in diagnostics['controls'].items():
        assert np.allclose(control['realized_pixel_mae'], diagnostics['common_pixel_mae'], rtol=0, atol=1e-7)
        assert all(0 < a <= .1 + 1e-12 for a in control['alpha'])
    for name, value in conditions.items():
        assert np.array_equal(value['joint_states'], before['joint_states'])
        assert np.array_equal(value['gripper_states'], before['gripper_states'])
        if name != 'identity_uint8':
            assert all(v.dtype == np.float32 and v.min() >= 0 and v.max() <= 1 for v in (value[k] for k in p.VIEWS))
    assert all(np.array_equal(obs[k], before[k]) for k in obs)


def test_photo_control_preserves_coordinates_and_channel_means():
    obs = images()
    values, diagnostics = p.prepare_interventions(obs)
    u = p.normalized(obs['eye_in_hand_rgb']).astype(np.float64)
    photo = values['eye_in_hand_photo_matched']['eye_in_hand_rgb']
    alpha = np.array(diagnostics['controls']['eye_in_hand_photo_matched']['alpha'])[:, None, None, None]
    mean = u.mean(axis=(1, 2), keepdims=True)
    assert np.allclose(photo - mean, (u - mean) * (1 - alpha), atol=1e-7, rtol=0)
    assert np.allclose(photo.mean(axis=(1, 2), dtype=np.float64), mean[:, 0, 0], atol=1e-7, rtol=0)
    assert np.array_equal(values['eye_in_hand_photo_matched']['agentview_rgb'], values['identity_float']['agentview_rgb'])


def test_full_spatial_probe_keeps_block_means_without_moving_blocks():
    obs = images()
    values, _ = p.prepare_interventions(obs)
    original = p.normalized(obs['eye_in_hand_rgb'])
    changed = values['eye_in_hand_spatial_full']['eye_in_hand_rgb']
    for row, col in ((0, 0), (7, 9), (15, 15)):
        block = changed[:, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
        expected = original[:, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8].mean(axis=(1, 2), dtype=np.float64)
        assert np.allclose(block, expected[:, None, None, :], atol=1e-7, rtol=0)


def test_flat_inputs_are_retained_as_zero_budget_controls():
    obs = images(flat=True)
    conditions, diagnostics = p.prepare_interventions(obs)
    assert diagnostics['common_pixel_mae'] == [0., 0.]
    assert diagnostics['active_rows'] == [False, False]
    for name in p.CONDITIONS[2:]:
        assert all(np.array_equal(conditions[name][key], conditions['identity_float'][key]) for key in p.VIEWS)


def test_float_input_is_not_silently_normalized_twice():
    obs = images()
    obs['eye_in_hand_rgb'] = obs['eye_in_hand_rgb'].astype(np.float32)
    with pytest.raises(ValueError, match='original RGB'):
        p.prepare_interventions(obs)


def test_identical_response_energy_can_improve_or_worsen_target_error():
    clean = np.zeros((2, 7), np.float32)
    clean[:, :3] = .5
    changed = clean.copy()
    changed[0, :3], changed[1, :3] = .25, .75
    target = np.zeros_like(clean)
    terms = p.error_terms(clean, changed, target)
    assert terms['response_energy_term'].tolist() == [.0625, .0625]
    assert terms['error_alignment_cross_term'].tolist() == [-.25, .25]
    assert terms['DeltaMSE'].tolist() == [-.1875, .3125]
    assert np.array_equal(clean[:, :3], np.full((2, 3), .5))


def test_component_mse_identity_uses_three_three_one_dimensions():
    clean = np.full((2, 7), .25, np.float32)
    changed, target = -clean, np.full((2, 7), .125, np.float32)
    terms = {k: p.error_terms(clean, changed, target, k) for k in p.COMPONENTS}
    for measure in terms['all7']:
        combined = (3 * terms['translation'][measure] + 3 * terms['rotation'][measure] + terms['gripper'][measure]) / 7
        assert np.allclose(terms['all7'][measure], combined, atol=1e-12, rtol=0)


@pytest.mark.parametrize('bad', [float('nan'), 1.1])
def test_invalid_target_action_rejected(bad):
    values = np.zeros((2, 7), np.float32)
    target = values.copy()
    target[0, 0] = bad
    with pytest.raises(ValueError, match='action values'):
        p.error_terms(values, values, target)


@pytest.mark.parametrize('key,value', [
    ('checkpoint_updates', 1000), ('default_closed_loop_stabilization_steps', 0), ('source_role', 'meta_query'),
    ('target_alignment', 'stabilized_observation_to_old_action0'), ('planned_main_forward_calls', 7201),
    ('new_environment_resets_or_steps', 1),
])
def test_changed_role_budget_or_target_semantics_rejected_before_io(tmp_path, monkeypatch, key, value):
    cfg = design()
    cfg['scope'][key] = value
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(p, 'sha', lambda _: pytest.fail('provenance read after rejected design'))
    with pytest.raises(ValueError, match='frozen roles'):
        p.validate(path)


def test_design_validation_does_not_claim_a_completed_experiment():
    cfg = design()
    p.verify_design(cfg)
    cfg['research_run_executed'] = True
    with pytest.raises(ValueError, match='design status'):
        p.verify_design(cfg)


def test_preprocessing_or_execution_binding_changes_require_new_protocol():
    cfg = design()
    cfg['transforms']['matched_budget'] = 'choose_best_strength_by_target_MSE'
    with pytest.raises(ValueError, match='frozen roles'):
        p.verify_design(cfg)
    cfg = design()
    cfg['execution_binding']['scientific_design_must_not_change'] = False
    with pytest.raises(ValueError, match='execution binding'):
        p.verify_design(cfg)
