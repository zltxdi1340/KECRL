import json
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_libero_task1_b_routed_history_closed_loop as r


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
    return {'reader': SimpleNamespace(reset=lambda: None), 'checkpoint': {'seed': seed}, 'readout_archives': {}, 'arm': arm}


def predict(model, obs, step, c):
    for key in ('encoder_forward_attempts', 'encoder_forwards_completed', 'readout_forward_attempts', 'readout_forwards_completed'):
        c[key] += 1
    candidate = model['arm'] == r.ARMS[1] and step >= 5
    c['readout_primitive_linear_calls'] += 2 if candidate else 1
    return np.ones(256, np.float32), np.zeros(7, np.float32), {'policy_step': step, 'history_length': min(step, 5),
        'startup': step <= 4, 'history_policy_steps': list(range(step, max(0, step-5), -1)),
        'route': 'candidate_motion_history_current_gripper' if candidate else 'shared_baseline_startup' if step <= 4 else 'baseline_current'}


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
    assert summary['strict_candidate_improvement_all_three_seeds'] is False


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
    ('qualification_accessed', True), ('new_optimizer_updates', True), ('inference_device', 'cuda:0')])
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
