import json

import numpy as np
import pytest
import torch

from experiments import run_libero_task4_ordered_prefix_reproduction_diagnostic as diagnostic


def observation(step):
    return {
        "agentview_rgb": np.full((128, 128, 3), step, np.uint8),
        "eye_in_hand_rgb": np.full((128, 128, 3), step + 1, np.uint8),
        "joint_states": np.full(7, step / 100, np.float32),
        "gripper_states": np.full(2, step / 1000, np.float32),
    }


def action(_observation):
    return np.asarray([.1, 0, 0, 0, 0, 0, -1], np.float32)


class FakeAdapter:
    def __init__(self):
        self.step_count = 0
        self.resets = []
        self.actions = []

    def reset(self, *, initial_state_index):
        self.resets.append(initial_state_index)
        self.step_count = 0
        return observation(0), {"step_count": 0, "initial_state_index": initial_state_index}

    def step(self, value):
        self.actions.append(value.copy())
        self.step_count += 1
        return observation(self.step_count), 0., False, self.step_count == 20, {
            "step_count": self.step_count, "success": False,
            "terminal_reason": "time_limit" if self.step_count == 20 else None,
        }


def source_episode():
    trace = []
    for step in range(21):
        obs = observation(step)
        row = {"step": step, "joint_state": obs["joint_states"].tolist(),
               "gripper_state": obs["gripper_states"].tolist()}
        if step:
            row["action"] = action(obs).tolist()
        trace.append(row)
    return {"steps": 20, "reset_observation_sha256": diagnostic.evidence._digest_observation(observation(0)),
            "stepwise_public_state_trace": trace, "success": False, "terminal_reason": "time_limit"}


def window():
    return diagnostic.collect_window(adapter=FakeAdapter(), state_index=0, action_fn=action)


def test_ordered_candidate_schedule_matches_frozen_56_window_budget():
    assert diagnostic.planned_states(20261023) == tuple(range(10))
    assert diagnostic.planned_states(20261024) == tuple(range(18))
    count = sum(len(diagnostic.planned_states(seed)) for seed in diagnostic.TARGETS) * 2
    assert count == diagnostic.SCOPE["new_episode_budget"] == 56
    assert count * 20 == diagnostic.SCOPE["max_new_policy_action_steps"]
    with pytest.raises(ValueError, match="training seed"):
        diagnostic.planned_states(20261022)


@pytest.mark.parametrize("field,value", [("policy_optimizer_updates", 1000),
                                         ("new_optimizer_updates", 1),
                                         ("new_episode_budget", 57),
                                         ("qualification_accessed", True)])
def test_budget_and_role_changes_fail_before_loading_weights(tmp_path, field, value):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["scope"][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="500-update reproduction scope"):
        diagnostic.load_protocol(path)


def test_skipping_preceding_states_is_not_silently_accepted(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["lifecycle"]["source_state_order_by_seed"]["20261023"] = [9]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="ordered candidate lifecycle"):
        diagnostic.load_protocol(path)


def test_capture_is_pre_action_and_uses_20_policy_actions():
    adapter = FakeAdapter()
    metadata, arrays = diagnostic.collect_window(adapter=adapter, state_index=9, action_fn=action)
    assert adapter.resets == [9] and len(adapter.actions) == 20
    assert np.array_equal(arrays["joint_states"][0], observation(0)["joint_states"])
    assert np.array_equal(arrays["joint_states"][19], observation(19)["joint_states"])
    assert metadata["post_step20_public_state"]["joint_states"] == observation(20)["joint_states"].tolist()
    assert metadata["pre_action_observation_sha256_by_step"][0] == metadata["reset_observation_sha256"]
    assert metadata["rgb_frame_sha256_by_step"]["eye_in_hand_rgb"][19] == diagnostic.evidence._digest_array(arrays["eye_in_hand_rgb"][19])
    diagnostic.evidence.require_source_match(diagnostic.verify_window(arrays, metadata, source_episode()))


def test_action_mismatch_cannot_be_saved_as_verified_archive(tmp_path):
    metadata, arrays = window()
    arrays["actions"][0, 0] = np.nextafter(arrays["actions"][0, 0], np.float32(1))
    audit = diagnostic.verify_window(arrays, metadata, source_episode())
    path = tmp_path / "wrong.npz"
    with pytest.raises(ValueError, match="source prefix mismatch"):
        diagnostic.save_verified_window(path, arrays, audit)
    assert not path.exists()


def test_reset_rgb_digest_mismatch_is_not_ignored(tmp_path):
    metadata, arrays = window()
    arrays["agentview_rgb"][0, 0, 0, 0] = 1
    metadata["reset_observation_sha256"] = diagnostic.evidence._digest_observation({k: arrays[k][0] for k in diagnostic.evidence.OBS_KEYS})
    audit = diagnostic.verify_window(arrays, metadata, source_episode())
    assert not audit["source_reset_digest_match"]
    with pytest.raises(ValueError, match="source prefix mismatch"):
        diagnostic.save_verified_window(tmp_path / "wrong.npz", arrays, audit)


def test_post_step20_drift_is_checked_in_addition_to_pre_action_states():
    metadata, arrays = window()
    source = source_episode()
    source["stepwise_public_state_trace"][20]["joint_state"][0] += 1e-8
    audit = diagnostic.verify_window(arrays, metadata, source)
    assert all(row["exact_match"] for row in audit["array_comparisons"].values())
    assert not audit["post_step20_public_state_exact_match"]
    assert not audit["source_prefix_exact_match"]


def test_verified_archive_retains_dtype_and_cannot_overwrite(tmp_path):
    metadata, arrays = window()
    path = tmp_path / "good.npz"
    audit = diagnostic.verify_window(arrays, metadata, source_episode())
    digest = diagnostic.save_verified_window(path, arrays, audit)
    assert diagnostic.sha256_file(path) == digest
    with np.load(path, allow_pickle=False) as saved:
        assert np.array_equal(saved["actions"], arrays["actions"])
        assert saved["agentview_rgb"].dtype == np.uint8
    with pytest.raises(FileExistsError):
        diagnostic.save_verified_window(path, arrays, audit)


def test_locked_state_index_is_rejected_before_loading_or_resetting(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("locked state request reached file loader")
    monkeypatch.setattr(torch, "load", forbidden)
    with pytest.raises(ValueError, match="candidate role"):
        diagnostic.load_candidate_states(tmp_path / "missing", {}, (20,))
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="candidate role"):
        diagnostic.collect_window(adapter=adapter, state_index=20, action_fn=action)
    assert adapter.resets == []


def test_candidate_container_checks_values_only_after_allowed_slice(tmp_path):
    synthetic = np.zeros((50, 92), np.float64)
    synthetic[20:] = np.nan
    path = tmp_path / "synthetic.pt"
    torch.save(synthetic, path)
    record = {"size_bytes": path.stat().st_size, "sha256": diagnostic.sha256_file(path)}
    states = diagnostic.load_candidate_states(path, record, (0, 9, 17))
    assert states.shape == (3, 92) and np.isfinite(states).all()


def test_early_termination_is_rejected_instead_of_filling_missing_frames():
    class EarlyAdapter(FakeAdapter):
        def step(self, value):
            obs, reward, _, _, info = super().step(value)
            return obs, reward, True, False, info
    with pytest.raises(ValueError, match="before its 20-step boundary"):
        diagnostic.collect_window(adapter=EarlyAdapter(), state_index=0, action_fn=action)


def test_existing_output_is_preserved_before_any_run(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("historical")
    monkeypatch.setattr(diagnostic, "load_protocol", lambda *args: pytest.fail("already-existing output reached protocol load"))
    with pytest.raises(ValueError, match="already exists"):
        diagnostic.run_diagnostic(tmp_path / "unused", output, tmp_path / "archive")
    assert output.read_text() == "historical"
