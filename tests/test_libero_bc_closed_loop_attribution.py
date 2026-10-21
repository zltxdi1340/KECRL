import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
_PATH = ROOT / "experiments/run_libero_bc_closed_loop_attribution.py"
_SPEC = importlib.util.spec_from_file_location("run_libero_bc_closed_loop_attribution", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_ATTRIBUTION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_ATTRIBUTION)


def test_attribution_config_reuses_the_fixed_base_budget():
    attribution, base, policy, transfer, family, dataset = _ATTRIBUTION._load_config(
        ROOT / "configs/libero_spatial_bc_closed_loop_attribution_v1.json"
    )
    assert attribution["training"]["optimizer_updates"] == 500
    assert attribution["training"]["optimizer_updates"] == base["training"]["optimizer_updates"]
    assert attribution["training"]["checkpoint_selection"] is False
    assert policy["formal_result"] is False
    assert transfer["formal_result"] is False
    assert family["formal_result"] is False
    assert dataset["formal_result"] is False


def test_demonstration_state_scope_is_partition_restricted():
    _attribution, _base, _policy, transfer, _family, dataset = _ATTRIBUTION._load_config(
        ROOT / "configs/libero_spatial_bc_closed_loop_attribution_v1.json"
    )
    assert _ATTRIBUTION.validate_demonstration_state_scope(
        transfer,
        dataset,
        task_id=0,
        demonstration_role="meta_support",
        indices=[0, 1, 2],
    ) == (0, 1, 2)
    with pytest.raises(ValueError, match="authorized partition"):
        _ATTRIBUTION.validate_demonstration_state_scope(
            transfer,
            dataset,
            task_id=0,
            demonstration_role="meta_support",
            indices=[30],
        )
    with pytest.raises(PermissionError, match="locked"):
        _ATTRIBUTION.validate_demonstration_state_scope(
            transfer,
            dataset,
            task_id=transfer["task_roles"]["heldout_query_spi"][0],
            demonstration_role="adaptation_support",
            indices=[0],
        )


def test_hidden_modes_match_training_horizon_boundaries():
    assert not _ATTRIBUTION.should_reset_hidden("continuous", 10, 10)
    assert _ATTRIBUTION.should_reset_hidden("reset_at_training_horizon", 0, 10)
    assert not _ATTRIBUTION.should_reset_hidden("reset_at_training_horizon", 9, 10)
    assert _ATTRIBUTION.should_reset_hidden("reset_at_training_horizon", 10, 10)
    assert _ATTRIBUTION.should_reset_hidden("stateless", 7, 10)
    with pytest.raises(ValueError, match="unknown"):
        _ATTRIBUTION.should_reset_hidden("unknown", 0, 10)


def test_action_trace_keeps_arm_and_gripper_behavior_visible():
    actions = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    result = _ATTRIBUTION.summarize_action_trace(actions)
    assert result["steps"] == 3
    assert result["per_dimension"]["z"]["maximum_abs"] == 3.0
    assert result["gripper_negative_fraction"] == pytest.approx(1 / 3)
    assert result["gripper_sign_switches"] == 1
