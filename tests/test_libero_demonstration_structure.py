import importlib.util
from pathlib import Path

import numpy as np
import pytest


_PATH = Path(__file__).resolve().parents[1] / "scripts/audit_libero_demonstration_structure.py"
_SPEC = importlib.util.spec_from_file_location("audit_libero_demonstration_structure", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)


def _trajectory(gripper, positions=None):
    gripper = np.asarray(gripper, dtype=float)
    positions = np.zeros((len(gripper), 3)) if positions is None else np.asarray(positions, dtype=float)
    actions = np.zeros((len(gripper), 7), dtype=float)
    actions[:, -1] = gripper
    robot_states = np.zeros((len(gripper), 9), dtype=float)
    robot_states[:, 2:5] = positions
    return actions, robot_states


def test_analyze_trajectory_reports_coarse_cycle_and_motion_segments():
    actions, states = _trajectory(
        [-1, -1, 1, 1, 1, -1, -1],
        [[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [4, 0, 0], [5, 0, 0], [6, 0, 0]],
    )
    result = _AUDIT.analyze_trajectory(actions, states, np.zeros(7), np.zeros(7))
    assert result["gripper_pattern"] == [-1, 1, -1]
    assert result["coarse_open_close_open"] is True
    assert result["exact_single_open_close_open"] is True
    assert result["first_release_observed"] is True
    assert result["ends_open"] is True
    assert result["path_lengths"]["before_first_close"] > 0
    assert result["path_lengths"]["closed_until_first_release"] > 0


def test_analyze_trajectory_keeps_successful_closed_termination_distinct():
    actions, states = _trajectory([-1, -1, 1, 1])
    result = _AUDIT.analyze_trajectory(actions, states)
    assert result["gripper_pattern"] == [-1, 1]
    assert result["coarse_open_close_open"] is False
    assert result["first_close_observed"] is True
    assert result["first_release_observed"] is False
    assert result["ends_closed"] is True
    assert result["path_lengths"]["after_first_release"] is None


def test_analyze_trajectory_rejects_wrong_interfaces():
    actions, states = _trajectory([-1, 1])
    with pytest.raises(ValueError, match="shape \(T, 7\)"):
        _AUDIT.analyze_trajectory(actions[:, :6], states)
    with pytest.raises(ValueError, match="shape \(T, 9\)"):
        _AUDIT.analyze_trajectory(actions, states[:, :8])
