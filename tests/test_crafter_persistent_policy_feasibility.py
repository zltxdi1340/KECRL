import json
from pathlib import Path

import pytest

from experiments.run_crafter_persistent_policy_feasibility import (
    _state_transfer_verified,
    _target_satisfied,
    _validate_config,
)


def test_persistent_policy_feasibility_candidate_is_non_formal_and_no_teacher():
    config = json.loads(Path("configs/crafter_persistent_policy_feasibility_pilot_v1.yaml").read_text())
    _validate_config(config)
    assert config["formal_result"] is False
    assert config["teacher_used"] is False
    assert config["policy_updated"] is False
    assert config["state_transfer_mode"] == "persistent_continual_world_per_seed"
    assert [task["task_id"] for task in config["tasks"]] == [
        "collect_wood", "setup_table", "obtain_wood_pickaxe", "collect_stone", "setup_furnace"
    ]


def test_persistent_policy_target_checks_use_public_boundary_state():
    assert _target_satisfied(
        {"inventory": {"wood": 3}, "world_object_setup": None},
        {"name": "inventory_at_least", "item": "wood", "threshold": 3},
    ) is True
    assert _target_satisfied(
        {"inventory": {"wood": 3}, "world_object_setup": None},
        {"name": "crafter_world_object_setup", "objects": ["table"]},
    ) == "unknown"
    assert _target_satisfied(
        {"inventory": {}, "world_object_setup": ["table"]},
        {"name": "crafter_world_object_setup", "objects": ["table"]},
    ) is True


def test_persistent_policy_validation_rejects_teacher_or_formal_candidate():
    config = json.loads(Path("configs/crafter_persistent_policy_feasibility_pilot_v1.yaml").read_text())
    config["teacher_used"] = True
    with pytest.raises(ValueError, match="must not use a teacher"):
        _validate_config(config)


def test_persistent_policy_transfer_audit_compares_public_boundaries():
    boundary = {
        "step_count": 3, "inventory": {"wood": 1},
        "world_object_setup": ["table"], "episode_done": False, "task_index": 1,
    }
    rows = [
        {"before": {**boundary, "task_index": 0}, "after": boundary},
        {"before": boundary, "after": {**boundary, "task_index": 2}},
    ]
    assert _state_transfer_verified(rows)
    rows[1]["before"] = {**boundary, "step_count": 4}
    assert not _state_transfer_verified(rows)
