import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import run_libero_bc_causal_alignment_diagnostic as diagnostic


def _protocol_stubs():
    base = {"task_id": 0, "seed": 20261021, "training": {"optimizer_updates": 500}}
    return base, {"policy": {}}, {}, {}, {}


def test_comparison_config_accepts_only_the_matched_two_arm_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "_load_protocols", lambda _path: _protocol_stubs())
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["base_diagnostic_config"] = "configs/libero_spatial_single_spi_bc_diagnostic_v1.json"
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(config))

    loaded = diagnostic.load_comparison_config(path)
    assert loaded[0]["training"]["optimizer_updates"] == 500

    config["training_arms"].reverse()
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="stored_same_index then causal_pre_action"):
        diagnostic.load_comparison_config(path)


def test_official_initial_states_use_pinned_legacy_torch_format(tmp_path: Path):
    path = tmp_path / "task.pruned_init"
    expected = np.arange(5 * 92, dtype=np.float64).reshape(5, 92)
    torch.save(expected, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    selected = diagnostic._capture_official_initial_states(
        initial_path=path,
        expected_size=path.stat().st_size,
        expected_sha256=digest,
        indices=(1, 3),
    )

    np.testing.assert_array_equal(selected, expected[[1, 3]])
    with pytest.raises(ValueError, match="pinned metadata"):
        diagnostic._capture_official_initial_states(
            initial_path=path,
            expected_size=path.stat().st_size,
            expected_sha256="0" * 64,
            indices=(1,),
        )


def test_official_initial_state_indices_cannot_escape_file(tmp_path: Path):
    path = tmp_path / "task.pruned_init"
    torch.save(np.zeros((2, 92), dtype=np.float64), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="evaluation state exceeds"):
        diagnostic._capture_official_initial_states(
            initial_path=path,
            expected_size=path.stat().st_size,
            expected_sha256=digest,
            indices=(2,),
        )


def test_hidden_state_protocol_pins_v1_models_and_unique_modes(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "_load_protocols", lambda _path: _protocol_stubs())
    config_path = diagnostic.ROOT / "configs/libero_spatial_bc_hidden_state_diagnostic_v1.json"
    config = json.loads(config_path.read_text())
    path = tmp_path / "hidden_state.json"
    path.write_text(json.dumps(config))

    loaded = diagnostic.load_comparison_config(path)
    assert loaded[0]["evaluation"]["policy_hidden_modes"] == list(diagnostic.HIDDEN_MODES)
    assert set(loaded[0]["expected_final_model_sha256_by_arm"]) == set(diagnostic.ALIGNMENTS)

    config["evaluation"]["policy_hidden_modes"].append("continuous")
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hidden-state modes are invalid"):
        diagnostic.load_comparison_config(path)


def test_continuous_teacher_forcing_carries_hidden_within_demo_and_resets_between_demos():
    class Window:
        def __init__(self, demonstration_index, start):
            self.demonstration_index = demonstration_index
            self.start = start

    class Dataset:
        stride = 2
        windows = (Window(0, 0), Window(0, 2), Window(1, 0))

        def __getitem__(self, _index):
            return {
                "agentview_rgb": torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
                "eye_in_hand_rgb": torch.zeros(2, 3, 2, 2, dtype=torch.uint8),
                "joint_states": torch.ones(2, 7),
                "gripper_states": torch.zeros(2, 2),
                "actions": torch.zeros(2, 7),
                "valid_mask": torch.ones(2, dtype=torch.bool),
            }

    class AccumulatingPolicy(torch.nn.Module):
        def forward_sequence(
            self, agentview_rgb, eye_in_hand_rgb, joint_states, gripper_states, hidden=None
        ):
            increments = joint_states[:, :, :1]
            previous = torch.zeros(1, len(joint_states), 1) if hidden is None else hidden
            values = torch.cumsum(increments, dim=1) + previous.transpose(0, 1)
            actions = values.expand(-1, -1, 7)
            return actions, values[:, -1:, :].transpose(0, 1)

    result = diagnostic._evaluate_hidden_state_teacher_forced(
        model=AccumulatingPolicy(), dataset=Dataset(), device=torch.device("cpu")
    )

    assert result["demonstration_count"] == 2
    continuous = result["by_hidden_mode"]["continuous"]["by_window_number"]
    reset = result["by_hidden_mode"]["reset_each_window"]["by_window_number"]
    assert continuous["0"]["mse"] == pytest.approx(2.5)
    assert continuous["1"]["mse"] == pytest.approx(12.5)
    assert reset["0"]["mse"] == pytest.approx(2.5)
    assert reset["1"]["mse"] == pytest.approx(2.5)


def test_trajectory_error_summary_tracks_action_and_state_drift_by_time_bin():
    observations = [
        {
            "agentview_rgb": np.full((2, 2, 3), value, dtype=np.uint8),
            "eye_in_hand_rgb": np.full((2, 2, 3), value + 1, dtype=np.uint8),
            "joint_states": np.full(7, value, dtype=np.float32),
            "gripper_states": np.full(2, value / 10, dtype=np.float32),
        }
        for value in (0, 10, 20)
    ]
    expert_actions = np.zeros((2, 7), dtype=np.float32)
    expert_actions[:, 6] = np.asarray([-1.0, 1.0])
    policy_actions = expert_actions.copy()
    policy_actions[0, 0] = 0.5
    policy_actions[1, 6] = -1.0
    policy_observations = [
        {key: value.copy() for key, value in observations[0].items()},
        {key: value + 2 for key, value in observations[1].items()},
        {key: value + 5 for key, value in observations[2].items()},
    ]

    result = diagnostic._trajectory_error_summary(
        expert_actions=expert_actions,
        policy_actions_on_expert_states=policy_actions,
        expert_observations=observations,
        policy_observations=policy_observations,
        time_bin_steps=1,
    )

    assert result["overlap_steps"] == 2
    assert result["teacher_forced_action_error"]["first_gripper_sign_mismatch_step"] == 1
    assert result["teacher_forced_action_error"]["gripper_sign_accuracy"] == pytest.approx(0.5)
    assert result["time_bins"][0]["agentview_rgb_mae"] == pytest.approx(0.0)
    assert result["time_bins"][1]["joint_state_l2_mean"] == pytest.approx(
        2 * np.sqrt(7)
    )
    assert result["free_rollout_state_drift_on_expert_horizon"]["agentview_rgb_mae"] == pytest.approx(1.0)
