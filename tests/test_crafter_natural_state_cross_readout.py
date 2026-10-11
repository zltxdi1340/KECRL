import json
from pathlib import Path

import pytest

pytest.importorskip("crafter")

from experiments.run_crafter_wood3_natural_state_cross_readout import _aggregate, _category


def test_cross_readout_category_mapping_is_read_only():
    assert _category({"ready_to_collect": True, "awake_adjacent_opportunity": True,
                      "safe_approach_move_actions": [1], "sleep_override_active": False}) == ("ready", (5,))
    assert _category({"ready_to_collect": False, "awake_adjacent_opportunity": True,
                      "unblocked_tree_move_actions": [2, 4], "safe_approach_move_actions": [],
                      "sleep_override_active": False}) == ("turn", (2, 4))
    assert _category({"ready_to_collect": False, "awake_adjacent_opportunity": False,
                      "safe_approach_move_actions": [1], "sleep_override_active": False}) == ("approach", (1,))
    assert _category({"ready_to_collect": False, "awake_adjacent_opportunity": False,
                      "safe_approach_move_actions": [], "sleep_override_active": False}) == ("other", ())


def test_cross_readout_aggregate_keeps_path_and_policy_axes():
    rows = []
    for path_variant in ("baseline", "standardized_linear"):
        for policy_variant in ("baseline", "standardized_linear"):
            rows.append({
                "path_variant": path_variant, "policy_variant": policy_variant, "category": "ready",
                "trajectory_action_valid": True, "trajectory_action_probability": .5,
                "p_do": .25, "p_valid": .25, "greedy_valid": True, "entropy": 1.0,
            })
    result = _aggregate(rows)
    assert set(result) == {
        "baseline/baseline/ready", "baseline/standardized_linear/ready",
        "standardized_linear/baseline/ready", "standardized_linear/standardized_linear/ready",
    }
    assert result["baseline/standardized_linear/ready"]["mean_p_do"] == .25


def test_cross_readout_config_is_evaluation_only():
    config = json.loads(Path("configs/crafter_wood3_natural_state_cross_readout_cuda_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["evaluation_only"] is True
    assert config["policy_updates_allowed"] is False
    assert config["new_fit_or_selection_allowed"] is False
    assert config["module_registered"] is False
