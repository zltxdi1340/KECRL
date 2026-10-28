import json

import numpy as np
import pytest

from experiments import run_libero_runtime_native_prefix_cross_task_observation_diagnostic as diagnostic


def test_config_pins_cross_task_cases_and_observation_boundary():
    config, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_SOURCE_RESULT,
        diagnostic.DEFAULT_TRACE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["task_ids"] == [1, 3, 4, 7]
    assert config["observation_capture"]["steps_per_selected_episode"] == 20
    assert config["rollout"]["selected_states"]["4"] == [0, 2, 7]


def test_config_rejects_changed_selected_strata(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["rollout"]["selected_states"]["7"] = [0]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="selected strata"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_SOURCE_RESULT,
            diagnostic.DEFAULT_TRACE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_observation_digest_tracks_all_policy_inputs():
    first = {
        "agentview_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "eye_in_hand_rgb": np.zeros((4, 4, 3), dtype=np.uint8),
        "joint_states": np.zeros(7, dtype=np.float32),
        "gripper_states": np.zeros(2, dtype=np.float32),
    }
    second = {key: value.copy() for key, value in first.items()}
    assert diagnostic._observation_digest(first) == diagnostic._observation_digest(second)
    second["eye_in_hand_rgb"][0, 0, 0] = 1
    assert diagnostic._observation_digest(first) != diagnostic._observation_digest(second)
