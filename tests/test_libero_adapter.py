from types import SimpleNamespace

import numpy as np
import pytest

from src.environments.libero_adapter import (
    POLICY_OBSERVATION_KEYS,
    LiberoEnvironmentAdapter,
)


class FakeLiberoEnvironment:
    action_spec = (np.full(7, -1.0), np.full(7, 1.0))

    def __init__(self, *, success_step=None, bad_done=False):
        self.env = SimpleNamespace(action_spec=self.action_spec)
        self.success_step = success_step
        self.bad_done = bad_done
        self.steps = 0
        self.closed = False
        self.seed_value = None
        self.initial_state = None

    def _observation(self):
        return {
            "agentview_image": np.full((16, 16, 3), self.steps, dtype=np.uint8),
            "robot0_eye_in_hand_image": np.full((16, 16, 3), 2, dtype=np.uint8),
            "robot0_joint_pos": np.arange(7, dtype=np.float64),
            "robot0_gripper_qpos": np.arange(2, dtype=np.float64),
            "object-state": np.arange(12),
            "black_bowl_to_robot0_eef_pos": np.arange(3),
        }

    def seed(self, value):
        self.seed_value = value

    def reset(self):
        self.steps = 0
        return self._observation()

    def set_init_state(self, state):
        self.initial_state = np.asarray(state).copy()
        return self._observation()

    def step(self, action):
        assert np.asarray(action).shape == (7,)
        self.steps += 1
        success = self.check_success()
        return self._observation(), float(success), success != self.bad_done, {"object-state": "private"}

    def check_success(self):
        return self.success_step is not None and self.steps >= self.success_step

    def close(self):
        self.closed = True


def make_adapter(environment=None, **kwargs):
    return LiberoEnvironmentAdapter(
        environment=environment or FakeLiberoEnvironment(),
        initial_states=np.arange(18).reshape(3, 6),
        task_name="pick_up_the_bowl",
        language_instruction="pick up the bowl",
        image_size=16,
        stabilization_steps=0,
        **kwargs,
    )


def test_reset_exposes_only_policy_observation_and_public_metadata():
    native = FakeLiberoEnvironment()
    adapter = make_adapter(native, seed=7)
    observation, info = adapter.reset(initial_state_index=2)
    assert tuple(observation) == POLICY_OBSERVATION_KEYS
    assert observation["joint_states"].dtype == np.float32
    assert observation["gripper_states"].shape == (2,)
    assert "object-state" not in observation
    assert not any("bowl_to" in key for key in observation)
    assert native.seed_value == 7
    assert np.array_equal(native.initial_state, np.arange(18).reshape(3, 6)[2])
    assert info["initial_state_index"] == 2
    assert info["success"] is False
    observation["joint_states"][0] = -99
    assert adapter.current_observation()["joint_states"][0] == 0


def test_success_and_time_limit_are_distinct():
    success_adapter = make_adapter(FakeLiberoEnvironment(success_step=2), max_episode_steps=3)
    success_adapter.reset()
    assert success_adapter.step(np.zeros(7))[2:4] == (False, False)
    _, reward, terminated, truncated, info = success_adapter.step(np.zeros(7))
    assert (reward, terminated, truncated) == (1.0, True, False)
    assert info["terminal_reason"] == "success"
    assert success_adapter.state()["terminal_reason"] == "success"
    with pytest.raises(RuntimeError, match="terminated or truncated"):
        success_adapter.step(np.zeros(7))

    timeout_adapter = make_adapter(max_episode_steps=1)
    timeout_adapter.reset()
    assert timeout_adapter.step(np.zeros(7))[2:4] == (False, True)
    assert timeout_adapter.state()["terminal_reason"] == "time_limit"


def test_success_on_last_allowed_step_is_not_reported_as_timeout():
    adapter = make_adapter(FakeLiberoEnvironment(success_step=1), max_episode_steps=1)
    adapter.reset()
    assert adapter.step(np.zeros(7))[2:4] == (True, False)
    assert adapter.state()["success"] is True
    assert adapter.state()["terminal_reason"] == "success"


@pytest.mark.parametrize("action", [
    np.zeros(6), np.zeros(8), np.full(7, np.nan), np.full(7, 1.01), np.full(7, -1.01),
])
def test_invalid_actions_fail_closed(action):
    adapter = make_adapter()
    adapter.reset()
    with pytest.raises(ValueError):
        adapter.step(action)


def test_reset_requires_valid_fixed_initial_state_index():
    adapter = make_adapter()
    for index in (-1, 3, True):
        with pytest.raises(ValueError, match="initial_state_index"):
            adapter.reset(initial_state_index=index)


def test_stabilization_is_outside_policy_step_budget():
    native = FakeLiberoEnvironment()
    adapter = LiberoEnvironmentAdapter(
        environment=native,
        initial_states=np.zeros((1, 3)),
        task_name="pick_up_the_bowl",
        language_instruction="pick up the bowl",
        image_size=16,
        stabilization_steps=5,
        max_episode_steps=1,
    )
    adapter.reset()
    assert native.steps == 5
    assert adapter.state()["step_count"] == 0
    assert adapter.step(np.zeros(7))[3] is True
    assert native.steps == 6


def test_native_done_success_disagreement_is_rejected():
    adapter = make_adapter(FakeLiberoEnvironment(success_step=1, bad_done=True))
    adapter.reset()
    with pytest.raises(RuntimeError, match="disagrees"):
        adapter.step(np.zeros(7))


def test_close_is_idempotent_and_prevents_reset():
    native = FakeLiberoEnvironment()
    adapter = make_adapter(native)
    adapter.close()
    adapter.close()
    assert native.closed is True
    with pytest.raises(RuntimeError, match="closed"):
        adapter.reset()
