import json
from pathlib import Path

import pytest

from src.environments.crafter_task_plan import build_crafter_task_plan


def test_task_local_plan_keeps_primary_target_separate_from_prerequisites():
    config = json.loads(Path("configs/crafter_task_local_composition_candidate_v1.yaml").read_text())
    plan = build_crafter_task_plan(config, "collect_stone", {"environment": "crafter"})
    assert [step.role for step in plan.steps] == ["prerequisite"] * 5 + ["target"]
    assert plan.steps[-1].request.target_capability == {"name": "inventory_at_least", "item": "stone", "threshold": 1}
    assert plan.steps[-1].request.input_capabilities == ({"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1},)
    assert plan.steps[0].request.target_capability["threshold"] == 1
    assert [step.request.target_capability["item"] for step in plan.steps[:3]] == ["wood", "wood", "wood"]


def test_task_local_plan_rejects_unknown_auxiliary_step():
    config = json.loads(Path("configs/crafter_task_local_composition_candidate_v1.yaml").read_text())
    config["task_plans"]["collect_stone"][0] = "missing_step"
    with pytest.raises(KeyError):
        build_crafter_task_plan(config, "collect_stone", {"environment": "crafter"})
