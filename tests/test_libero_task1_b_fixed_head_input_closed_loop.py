import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_libero_task1_b_fixed_head_input_closed_loop as r


class FakeAdapter:
    def __init__(self, stop=5, zeros=5):
        self.environment = SimpleNamespace(step=lambda action: None)
        self.stop, self.zeros, self.n, self.state = stop, zeros, 0, 0

    def observation(self):
        return {'agentview_rgb': np.zeros((128, 128, 3), np.uint8), 'eye_in_hand_rgb': np.zeros((128, 128, 3), np.uint8),
            'joint_states': np.full(7, self.state/100, np.float32), 'gripper_states': np.zeros(2, np.float32)}

    def reset(self, initial_state_index):
        self.state, self.n = initial_state_index, 0
        for _ in range(self.zeros):
            self.environment.step(np.zeros(7, np.float32))
        return self.observation(), {}

    def step(self, action):
        self.environment.step(action)
        self.n += 1
        done = self.n == self.stop
        return self.observation(), float(done), done, False, {'success': done, 'terminal_reason': 'success' if done else None}

    def close(self):
        pass


def fake_model(seed, arm):
    return {'reader': SimpleNamespace(reset=lambda: None), 'checkpoint': {'seed': seed}, 'readout_archives': {}, 'arm': arm, 'controller_identity': {'same_seed_weights': seed}}


def predict(model, obs, step, c):
    for key in ('encoder_forward_attempts', 'encoder_forwards_completed', 'readout_forward_attempts', 'readout_forwards_completed'):
        c[key] += 1
    meta = r.runtime.route_metadata(model['arm'], step)
    c['readout_primitive_linear_attempts'] += meta['primitive_linear_calls']
    c['readout_primitive_linear_calls'] += meta['primitive_linear_calls']
    return np.ones(256, np.float32), np.zeros(7, np.float32), meta


def test_complete_paired_budget_and_startup_with_early_termination(tmp_path):
    c = r.new_counts()
    episodes, files = r.execute_models(model_loader=fake_model, adapter_factory=lambda: FakeAdapter(stop=2), predictor=predict,
        checker=lambda m: None, archive=tmp_path, counts=c, event=lambda *args, **kwargs: None)
    assert len(episodes) == len(files) == 120
    assert c['policy_actions_completed'] == 240 and c['zero_steps_completed'] == 600
    assert c['native_steps_completed'] == 840
    assert c['reset_digest_references'] == 20 and c['reset_digest_comparisons'] == 100
    assert c['startup_action_comparisons'] == c['prefix_observation_comparisons'] == 120
    summary = r.paired_summary(episodes)
    assert summary['paired_outcomes']['both_success'] == 60
    assert summary['strict_repeat_current_improvement_all_three_seeds'] is False


def test_candidate_startup_mismatch_aborts_before_action_and_retains_episode(tmp_path):
    c = r.new_counts()
    def wrong(model, obs, step, counts):
        f, action, meta = predict(model, obs, step, counts)
        if model['arm'] == r.ARMS[1] and step == 1:
            action[0] = 1
        return f, action, meta
    with pytest.raises(ValueError, match='shared-startup action'):
        r.execute_models(model_loader=fake_model, adapter_factory=FakeAdapter, predictor=wrong, checker=lambda m: None,
                         archive=tmp_path, counts=c, event=lambda *args, **kwargs: None)
    assert c['episodes_completed'] == 20 and c['policy_actions_completed'] == 100
    failed = json.loads((tmp_path/f'seed{r.SEEDS[0]}-{r.ARMS[1]}-state00.json').read_text())
    assert failed['status'] == 'failed' and failed['completed_actions_in_episode'] == 0


def test_wrong_stabilization_aborts_before_encoder(tmp_path):
    c = r.new_counts()
    with pytest.raises(ValueError, match='exactly five'):
        r.execute_models(model_loader=fake_model, adapter_factory=lambda: FakeAdapter(zeros=4), predictor=predict,
                         checker=lambda m: None, archive=tmp_path, counts=c, event=lambda *args, **kwargs: None)
    assert c['encoder_forward_attempts'] == 0 and c['zero_steps_completed'] == 4


@pytest.mark.parametrize('key,value', [('initial_state_indices', list(range(20, 40))), ('task_id', 2),
    ('stabilization_steps', 0), ('episodes', 121), ('maximum_policy_actions', 37000), ('readout_updates', 1000),
    ('qualification_accessed', True), ('new_optimizer_updates', True), ('inference_device', 'cuda:0'), ('same_candidate_weights_between_inputs', False),
    ('maximum_primitive_linear_calls', 53761)])
def test_scope_mismatch_rejected_before_access(tmp_path, monkeypatch, key, value):
    c = json.loads(r.DEFAULT_CONFIG.read_text())
    c['scope'][key] = value
    p = tmp_path/'changed.json'
    p.write_text(json.dumps(c))
    monkeypatch.setattr(r, 'sha', lambda *args: pytest.fail('scope reached inputs'))
    with pytest.raises(ValueError, match='frozen scope'):
        r.preflight(p, tmp_path/'binding')


def test_renderer_guard_never_falls_back_when_gpu1_occupied(monkeypatch):
    responses = iter(['0, GPU-a, RTX 4090, 0, 0\n1, GPU-b, RTX 4090, 600, 0\n', ''])
    monkeypatch.setattr(r.subprocess, 'check_output', lambda *args, **kwargs: next(responses))
    with pytest.raises(ValueError, match='occupied'):
        r.idle_renderer()


def test_metadata_failure_is_zero_call_and_old_outputs_remain(tmp_path, monkeypatch):
    def reject(*args):
        raise ValueError('synthetic metadata failure')
    monkeypatch.setattr(r, 'preflight', reject)
    out, artifacts = tmp_path/'result.json', tmp_path/'artifacts'
    with pytest.raises(ValueError, match='synthetic'):
        r.run(None, None, out, artifacts)
    failed = json.loads((tmp_path/'result-failed.json').read_text())
    assert not any(failed['actual_counts_so_far'].values()) and failed['primary_success_summary_valid'] is False
    with pytest.raises(ValueError, match='already exist'):
        r.run(None, None, out, artifacts)


def synthetic_heads():
    rng = np.random.default_rng(52)
    mean, scale = rng.normal(0, .1, 256), rng.uniform(.5, 2, 256)
    return {a: {'weight': rng.normal(0, .02, (7, 1280)).astype('f4'),
        'bias': rng.normal(0, .01, 7).astype('f4'), 'mean': mean.copy(), 'scale': scale.copy()}
        for a in r.runtime.previous.ARMS}


def test_same_weights_input_only_routing_numpy_reference_and_parent_bridge():
    import torch
    torch.set_num_threads(1)
    heads = synthetic_heads()
    readers = [r.runtime.FixedHeadInputReadout(heads, arm) for arm in r.ARMS]
    old = r.runtime.previous.FrozenHistoryReadout(heads, r.runtime.previous.ARMS[1])
    for reader in [*readers, old]:
        reader.reset()
    hashes = [x.state_hash() for x in readers]
    assert hashes[0] == hashes[1]
    features = np.random.default_rng(22).normal(size=(9, 256)).astype('f4')
    operations = [[], []]
    for step, feature in enumerate(features, 1):
        values = [x.act(feature, step, operations[i].append) for i, x in enumerate(readers)]
        parent, _ = old.act(feature, step)
        np.testing.assert_array_equal(values[0][0], parent)
        if step <= 4:
            np.testing.assert_array_equal(values[0][0], values[1][0])
        else:
            assert values[0][0][6] == values[1][0][6]
            assert not np.array_equal(values[0][0][:6], values[1][0][:6])
        for i, (action, meta) in enumerate(values):
            assert meta == r.runtime.route_metadata(r.ARMS[i], step)
            ids = meta['motion_input_policy_steps']
            current_ids = [step] * 5
            transform = lambda idx: (((features[np.array(idx)-1].astype('f8') - heads['current_all7']['mean']) /
                heads['current_all7']['scale']).reshape(1, 1280) / np.sqrt(5)).astype('f4').astype('f8')
            head = heads['current_all7' if step <= 4 else 'history_motion_current_gripper']
            logits = transform(ids) @ head['weight'].astype('f8').T + head['bias']
            logits[:, 6] = (transform(current_ids) @ head['weight'].astype('f8').T + head['bias'])[:, 6]
            np.testing.assert_allclose(action, np.tanh(logits)[0], rtol=0, atol=1e-6)
    assert [len(op) for op in operations] == [28, 18]
    assert all(op[::2] == ['attempt'] * (len(op)//2) and op[1::2] == ['complete'] * (len(op)//2) for op in operations)
    assert [x.state_hash() for x in readers] == hashes
    for reader in readers:
        reader.reset()
        a, meta = reader.act(features[0], 1)
        np.testing.assert_array_equal(a, old_reset_action(old, features[0]))
        assert meta['history_policy_steps'] == [1] and len(reader.history) == 1


def old_reset_action(reader, feature):
    reader.reset()
    return reader.act(feature, 1)[0]


def test_runtime_rejects_missing_reset_noncausal_steps_and_bad_scaler():
    heads = synthetic_heads()
    reader = r.runtime.FixedHeadInputReadout(heads, r.ARMS[0])
    feature = np.ones(256, 'f4')
    with pytest.raises(ValueError, match='reset/sequential'):
        reader.act(feature, 1)
    reader.reset()
    for step in (0, 2, True):
        with pytest.raises(ValueError, match='reset/sequential'):
            reader.act(feature, step)
    for feature_bad in (feature.astype('f8'), feature[:-1], np.full(256, np.nan, 'f4')):
        with pytest.raises(ValueError, match='invalid public'):
            reader.act(feature_bad, 1)
    assert reader.next_step == 1 and len(reader.history) == 0
    heads['history_motion_current_gripper']['scale'][0] *= 2
    with pytest.raises(ValueError, match='scalers differ'):
        r.runtime.FixedHeadInputReadout(heads, r.ARMS[0])


def test_internal_linear_failure_counts_attempt_without_completion():
    reader = r.runtime.FixedHeadInputReadout(synthetic_heads(), r.ARMS[0])
    reader.reset()
    events = []
    def reject(kind):
        events.append(kind)
        raise RuntimeError('synthetic Linear interruption')
    with pytest.raises(RuntimeError, match='interruption'):
        reader.act(np.zeros(256, 'f4'), 1, reject)
    assert events == ['attempt']


@pytest.mark.parametrize('fault,message', [('route', 'input/weight route'), ('feature', 'feature prefix'),
    ('gripper', 'step5 gripper')])
def test_step5_pairing_fault_aborts_before_control_and_preserves_failed_episode(tmp_path, fault, message):
    counts = r.new_counts()
    def wrong(model, obs, step, c):
        feature, action, meta = predict(model, obs, step, c)
        if model['arm'] == r.ARMS[1] and step == 5:
            if fault == 'route':
                meta['motion_input_policy_steps'][0] = step+1
            elif fault == 'feature':
                feature[0] += 1
            else:
                action[6] = .1
        return feature, action, meta
    with pytest.raises(ValueError, match=message):
        r.execute_models(model_loader=fake_model, adapter_factory=FakeAdapter, predictor=wrong,
            checker=lambda m: None, archive=tmp_path, counts=counts, event=lambda *a, **k: None)
    assert counts['episodes_completed'] == 20 and counts['policy_actions_completed'] == 104
    failed = json.loads((tmp_path/f'seed{r.SEEDS[0]}-{r.ARMS[1]}-state00.json').read_text())
    assert failed['status'] == 'failed' and failed['completed_actions_in_episode'] == 4


def test_weight_identity_difference_rejects_second_adapter(tmp_path):
    counts = r.new_counts()
    def changed(seed, arm):
        model = fake_model(seed, arm)
        if arm == r.ARMS[1]:
            model['controller_identity']['changed'] = 'different candidate weights'
        return model
    with pytest.raises(ValueError, match='weights/scaler identity'):
        r.execute_models(model_loader=changed, adapter_factory=FakeAdapter, predictor=predict,
            checker=lambda m: None, archive=tmp_path, counts=counts, event=lambda *a, **k: None)
    assert counts['episodes_completed'] == 20 and counts['adapters_built'] == 1 and counts['models_loaded'] == 2


def test_all_four_paired_outcomes_and_process_changes_are_distinct():
    episodes = []
    for seed in r.SEEDS:
        for arm_index, arm in enumerate(r.ARMS):
            for i in range(20):
                success = i % 4 == 0 or i % 4 == (1 if arm_index == 0 else 2)
                episodes.append({'status': 'complete', 'training_seed': seed, 'arm': arm,
                    'initial_state_index': i, 'success': success, 'steps': 300,
                    'terminal_reason': 'success' if success else 'time_limit',
                    'action_trace_sha256': f'actions-{arm}'})
    summary = r.paired_summary(episodes)
    assert set(summary['paired_outcomes'].values()) == {15}
    assert summary['action_trace_different_pairs'] == 60
    assert summary['terminal_result_different_pairs'] == 30
    assert not summary['strict_repeat_current_improvement_all_three_seeds']
    with pytest.raises(ValueError, match='complete120'):
        r.paired_summary(episodes[:-1])
    episodes[-1] = episodes[0]
    with pytest.raises(ValueError, match='duplicate or missing'):
        r.paired_summary(episodes)


def test_valid_frozen_metadata_without_numeric_or_environment_reads(monkeypatch):
    monkeypatch.setattr(np, 'load', lambda *a, **k: pytest.fail('metadata decoded NPZ'))
    monkeypatch.setattr(r.runtime, 'load_encoder', lambda *a, **k: pytest.fail('metadata loaded encoder'))
    monkeypatch.setattr(r, 'idle_renderer', lambda: pytest.fail('metadata entered execution'))
    c, _ = r.preflight(r.DEFAULT_CONFIG, r.DEFAULT_BINDING)
    assert c['scope']['episodes'] == 120 and c['scope']['same_candidate_weights_between_inputs'] is True
