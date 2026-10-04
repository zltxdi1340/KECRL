import copy
import json
from pathlib import Path

import pytest

from experiments.prepare_crafter_protocol import audit_manifest, build_manifest
from src.environments.crafter_tasks import (
    crafter_knowledge_evidence,
    crafter_transition_result,
    inventory_at_least,
)


@pytest.mark.parametrize("inventory", [None, {}, {"wood": None}, {"wood": -1}, {"wood": float("nan")}, {"wood": True}])
def test_missing_or_invalid_inventory_is_unknown(inventory):
    assert inventory_at_least(inventory, "wood", 1) == "unknown"


def test_public_goal_uses_state_not_achievement_or_recipe():
    assert inventory_at_least({"wood": 0}, "wood", 1) is False
    assert inventory_at_least({"wood": 2}, "wood", 2) is True
    assert inventory_at_least({"achievements": {"collect_wood": 1}}, "wood", 1) == "unknown"


def test_transition_and_evidence_keep_unknown_and_feedback_boundaries():
    target = {"name": "inventory_at_least", "item": "wood", "threshold": 1}
    unknown = crafter_transition_result(None, None, target, False)
    assert unknown.target_achieved == "unknown"
    assert unknown.execution_status == "unknown"
    continued = crafter_transition_result({"wood": 0}, {"wood": 0}, target, False)
    assert continued.target_achieved is False
    assert continued.execution_status == "continued"
    completed = crafter_transition_result({"wood": 0}, {"wood": 1}, target, False)
    assert completed.target_achieved is True
    assert completed.produced_capabilities == (target,)
    evidence = crafter_knowledge_evidence(None, {"wood": 0}, {"environment": "crafter"})
    assert evidence.evidence_validity == "unknown"
    assert evidence.intervention_metadata["performed"] is False
    assert "trajectory" not in evidence.before_state


def test_protocol_reproducibility_and_actual_seed_isolation():
    config = json.loads(Path("configs/crafter_task_protocol_v1.yaml").read_text())
    first = build_manifest(config)
    assert first == build_manifest(config)
    assert audit_manifest(first)["split_checks_passed"]
    assert not audit_manifest(first)["ready_for_formal_training"]
    duplicate = copy.deepcopy(first)
    duplicate["episodes"][1]["env_seed"] = duplicate["episodes"][0]["env_seed"]
    assert not audit_manifest(duplicate)["split_checks_passed"]
    config["seeds"] = [0, 0]
    with pytest.raises(ValueError):
        build_manifest(config)
