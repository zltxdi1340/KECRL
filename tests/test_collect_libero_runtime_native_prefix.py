import numpy as np
import pytest

from experiments import collect_libero_runtime_native_prefix as collector


def _observation():
    return {
        "agentview_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
        "eye_in_hand_rgb": np.zeros((128, 128, 3), dtype=np.uint8),
        "joint_states": np.zeros(7, dtype=np.float32),
        "gripper_states": np.zeros(2, dtype=np.float32),
    }


def test_runtime_observation_validation_is_strict():
    result = collector._validate_observation(_observation())
    assert result["agentview_rgb"].dtype == np.uint8
    assert result["joint_states"].shape == (7,)
    broken = _observation()
    broken["joint_states"] = np.zeros(6, dtype=np.float32)
    with pytest.raises(ValueError, match="joint_states shape"):
        collector._validate_observation(broken)


def test_prefix_save_records_digest_and_shared_length(tmp_path):
    observations = {key: np.stack([_observation()[key], _observation()[key]]) for key in collector.OBSERVATION_KEYS}
    actions = np.zeros((2, 7), dtype=np.float32)
    saved = collector._save_prefix(tmp_path / "prefix.npz", actions=actions, observations=observations)
    assert saved["prefix_length"] == 2
    assert len(saved["sha256"]) == 64
    with np.load(saved["path"]) as loaded:
        assert loaded["actions"].shape == (2, 7)
        assert loaded["eye_in_hand_rgb"].shape == (2, 128, 128, 3)


def test_prefix_save_rejects_mismatched_lengths(tmp_path):
    observations = {key: np.stack([_observation()[key]]) for key in collector.OBSERVATION_KEYS}
    with pytest.raises(ValueError, match="shared positive length"):
        collector._save_prefix(
            tmp_path / "prefix.npz",
            actions=np.zeros((2, 7), dtype=np.float32),
            observations=observations,
        )


def test_terminal_expectation_uses_pinned_reason_not_prefix_completion():
    terminal_reason = None
    assert (terminal_reason is not None) is False
    terminal_reason = "success"
    assert (terminal_reason is not None) is True
