import json

import numpy as np
import pytest

from experiments import run_libero_task1_b_fixed_head_expert_target_error as r


def test_action_change_energy_alone_cannot_identify_improvement():
    target, real = np.zeros((2, 7)), np.zeros((2, 7))
    target[:, 0], real[:, 0] = 1, .5
    better, worse = real.copy(), real.copy()
    better[:, 0], worse[:, 0] = .9, .1
    good, good_arrays = r.error_terms(real, better, target)
    bad, bad_arrays = r.error_terms(real, worse, target)
    assert good['translation']['repeat_minus_real_MSE'] == pytest.approx(-.08)
    assert bad['translation']['repeat_minus_real_MSE'] == pytest.approx(.56/3)
    assert good['translation']['response_energy'] == pytest.approx(bad['translation']['response_energy'])
    assert good['translation']['direction_term'] < 0 < bad['translation']['direction_term']
    for arrays in (good_arrays, bad_arrays):
        np.testing.assert_allclose(arrays['elementwise_mse_delta'],
            arrays['elementwise_direction_term']+arrays['elementwise_response_energy'], atol=1e-12, rtol=0)


def test_same_candidate_head_two_inputs_current_gripper_reused_without_third_call():
    rng = np.random.default_rng(330)
    current, history = [rng.normal(size=(9, 1280)).astype('f4') for _ in range(2)]
    weight, bias = rng.normal(0, .01, (7, 1280)).astype('f4'), np.zeros(7, 'f4')
    counts = r.new_counts()
    real, repeat, rl, cl = r.paired_inputs(current, history, weight, bias, counts)
    np.testing.assert_array_equal(real[:, 6], repeat[:, 6])
    np.testing.assert_array_equal(rl[:, 6], cl[:, 6])
    expected_real = np.tanh(history.astype('f8') @ weight.astype('f8').T)
    np.testing.assert_allclose(real[:, :6], expected_real[:, :6], atol=1e-14, rtol=0)
    assert counts['new_repeat_attempts'] == counts['new_repeat_completed'] == 1
    assert counts['history_replay_attempts'] == counts['history_replay_completed'] == 1
    assert counts['new_repeat_rows_completed'] == counts['history_replay_rows_completed'] == 9
    metrics, arrays = r.error_terms(real, repeat, np.ones((9, 7)))
    assert metrics['gripper']['repeat_minus_real_MSE'] == metrics['gripper']['response_energy'] == 0
    np.testing.assert_array_equal(arrays['elementwise_mse_delta'][:, 6], np.zeros(9))


def seed_rows(translation=(-.1, -.1, -.1), rotation=(0, 0, 0), gripper=(0, 0, 0)):
    rows = []
    for i, seed in enumerate(r.SEEDS):
        values = {key: 0 for key in r.PARTS}
        values.update(translation=translation[i], rotation=rotation[i], gripper=gripper[i], all7=translation[i]/7)
        rows.append({'training_seed': seed, 'strata': {'all': {k: {'repeat_minus_real_MSE': v} for k, v in values.items()}}})
    return rows


def test_all_seeds_and_secondary_tradeoffs_cannot_be_hidden_in_pooled_improvement():
    assert r.improvement_flags(seed_rows())['joint_descriptive_flag'] is True
    mixed = r.improvement_flags(seed_rows(translation=(-.5, -.5, .1)))
    assert mixed['translation_strictly_improved_all_three'] is False
    tradeoff = r.improvement_flags(seed_rows(rotation=(0, .001, 0)))
    assert tradeoff['translation_strictly_improved_all_three'] is True and tradeoff['joint_descriptive_flag'] is False
    with pytest.raises(ValueError, match='all three'):
        r.improvement_flags(seed_rows()[:2])
    with pytest.raises(ValueError, match='negative control'):
        r.improvement_flags(seed_rows(gripper=(0, 1e-15, 0)))


def test_history_rejects_future_cross_demo_padding_and_wrong_order():
    ids = np.repeat([30, 31], 6)
    steps = np.tile(np.arange(1, 7), 2)
    rows = np.array([4, 10])
    history = rows[:, None]-np.arange(5)
    r.validate_history(ids, steps, rows, history)
    for row, col, value in [(0, 0, 5), (0, 1, 5), (1, 4, 1), (0, 4, -1)]:
        broken = history.copy()
        broken[row, col] = value
        with pytest.raises(ValueError, match='history'):
            r.validate_history(ids, steps, rows, broken)


def test_frame_roles_rejected_before_history_planning(monkeypatch):
    monkeypatch.setattr(r.cached, 'make_plan', lambda *a: pytest.fail('invalid role reached frame planner'))
    with pytest.raises(ValueError, match='role/count'):
        r.common_plan(np.zeros((2598, 7), 'f4'), np.full(2598, 2), np.arange(1, 2599))


def test_linear_budget_attempts_are_bounded_and_failed_calls_do_not_complete(monkeypatch):
    x, w, b = np.ones((2, 1280), 'f4'), np.zeros((7, 1280), 'f4'), np.zeros(7, 'f4')
    counts = r.new_counts()
    def fail(_):
        raise RuntimeError('synthetic tanh failure')
    monkeypatch.setattr(np, 'tanh', fail)
    with pytest.raises(RuntimeError, match='synthetic'):
        r.linear(x, w, b, 'new_repeat', counts)
    assert counts['new_repeat_attempts'] == 1 and counts['new_repeat_completed'] == 0
    assert counts['new_repeat_rows_attempted'] == 2 and counts['new_repeat_rows_completed'] == 0
    counts['new_repeat_attempts'] = 3
    with pytest.raises(ValueError, match='budget'):
        r.linear(x, w, b, 'new_repeat', counts)


@pytest.mark.parametrize('key,value', [('task_id', 2), ('demonstration_role', 'meta_support'),
    ('demonstration_indices', list(range(20))), ('training_seeds', [20261024]),
    ('readout_updates', 1000), ('common_first_action_step', 1), ('new_repeat_batches', 4),
    ('history_replay_batches', 6), ('offline_stabilization_actions', 5),
    ('same_candidate_weights', False), ('sealed_roles_accessed', True), ('new_environment_calls', True)])
def test_frozen_scope_rejects_before_any_payload_hash(tmp_path, monkeypatch, key, value):
    config = json.loads(r.DEFAULT_CONFIG.read_text())
    config['scope'][key] = value
    p = tmp_path/'bad.json'
    p.write_text(json.dumps(config))
    monkeypatch.setattr(r, 'sha', lambda *a: pytest.fail('invalid scope reached input hashes'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path/'binding.json')


def test_metadata_preflight_decodes_no_npz_and_runs_no_readout(monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata decoded NPZ'))
    monkeypatch.setattr(r, 'paired_inputs', lambda *a: pytest.fail('metadata ran readout'))
    c, _ = r.preflight()
    assert c['scope']['seed_frame_pairs'] == 7554 and c['scope']['new_encoder_calls'] == 0


def test_failed_preflight_retains_zero_counts_and_never_overwrites(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic preflight rejection')
    monkeypatch.setattr(r, 'preflight', reject)
    out, artifacts = tmp_path/'result.json', tmp_path/'artifacts'
    with pytest.raises(ValueError, match='synthetic'):
        r.run(None, None, out, artifacts)
    failed = json.loads((tmp_path/'result-failed.json').read_text())
    assert failed['primary_valid'] is False and not any(failed['actual_counts_so_far'].values())
    assert not artifacts.exists()
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, out, artifacts)


def test_error_summary_retains_frame_counts_and_empty_strata():
    real = np.zeros((3, 7))
    repeat = real.copy()
    repeat[:, 0] = [-1, 1, 0]
    metrics, _ = r.error_terms(real, repeat, np.ones((3, 7)))
    assert metrics['translation']['frame_deltas']['improved_n'] == 1
    assert metrics['translation']['frame_deltas']['worsened_n'] == 1
    assert metrics['translation']['frame_deltas']['equal_n'] == 1
    empty, _ = r.error_terms(real[:0], repeat[:0], real[:0])
    assert empty['translation']['real_MSE'] is None and empty['translation']['frame_deltas']['n'] == 0
    with pytest.raises(ValueError, match='interface'):
        r.error_terms(real, np.full_like(real, np.nan), real)
