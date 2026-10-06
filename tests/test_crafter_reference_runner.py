import pytest

pytest.importorskip("crafter")

from src.counterfactual.crafter_reference import InterventionSpec
from src.counterfactual.crafter_reference_runner import run_paired_reference


def test_real_paired_world_proves_wood_unavailable_for_pickaxe():
    result = run_paired_reference(
        seed=0,
        pair_id="pair:test:0",
        target={"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1},
        environment_scope={"environment": "crafter", "adapter": "rgb64_inventory_v1"},
        intervention=InterventionSpec(
            "necessity_ablation", {"name": "inventory_at_least", "item": "wood", "threshold": 1}, 256,
        ),
        max_steps=256,
    )
    assert result["baseline"].target_achieved is True
    assert result["intervention"].proven_unreachable is True
    assert result["verification"].outcome == "PROVEN_UNREACHABLE"
    assert result["knowledge_evidence"].evidence_validity == "invalid"


def test_runner_keeps_unsupported_target_unknown():
    result = run_paired_reference(
        seed=0,
        pair_id="pair:test:unsupported",
        target={"name": "inventory_at_least", "item": "diamond", "threshold": 1},
        environment_scope={"environment": "crafter", "adapter": "rgb64_inventory_v1"},
        intervention=InterventionSpec(
            "availability_intervention", {"name": "inventory_at_least", "item": "wood", "threshold": 1}, 16,
        ),
        max_steps=16,
    )
    assert result["verification"].outcome == "UNKNOWN"
    assert result["knowledge_evidence"].evidence_validity == "unknown"
