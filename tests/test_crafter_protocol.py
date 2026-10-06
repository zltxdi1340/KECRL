import copy
import json
from pathlib import Path

import pytest

from experiments.prepare_crafter_protocol import (
    _derived_world_seed,
    audit_manifest,
    build_manifest,
)
from src.environments.crafter_tasks import (
    crafter_knowledge_evidence,
    crafter_transition_result,
    inventory_at_least,
)
from src.environments.crafter_continual import (
    CrafterContinualSession,
    WorldObjectSetupContract,
    WorldObjectSetupObservation,
)
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
import torch


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


def test_real_policy_module_executor_checks_response_and_returns_public_transition():
    from src.environments.crafter_adapter import CrafterEnvironmentAdapter
    from src.skills.contracts import ImplementationContract, ImplementationResponse

    target = {"name": "inventory_at_least", "item": "wood", "threshold": 1}
    contract = ImplementationContract((), (), (target,), {}, {}, {}, {"environment": "crafter"})
    response = ImplementationResponse("reused_module", "module:test", "spi:test", "spt:test", contract)
    policy = CategoricalResourcePolicy(PolicyConfig(observation_dim=192, action_count=17))
    environment = CrafterEnvironmentAdapter(seed=0, length=1)
    executor = CrafterPolicyModuleExecutor("module:test", policy, environment, target, torch.device("cpu"), 1)
    result = executor.execute(response, {"inventory": None})
    assert result.target_achieved in (False, "unknown")
    assert executor.last_steps == 1
    environment.close()


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


def test_derived_world_seed_is_stable_and_process_independent():
    first = _derived_world_seed(12345, 1)
    second = _derived_world_seed(12345, 1)
    assert first == second
    assert first != _derived_world_seed(12345, 2)


def test_continual_session_and_world_object_contract_keep_state_boundary():
    from src.environments.crafter_adapter import CrafterEnvironmentAdapter
    environment = CrafterEnvironmentAdapter(seed=0, length=2)
    session = CrafterContinualSession(environment)
    session.start()
    setup = WorldObjectSetupContract(("table", "furnace"))
    task_state = session.begin_task("obtain_iron_pickaxe", setup)
    assert task_state["world_state_mode"] == "persistent_continual_world_per_seed"
    assert session.state()["task_id"] == "obtain_iron_pickaxe"
    assert WorldObjectSetupObservation(("table", "furnace")).satisfies(setup)
    session.end_task()
    assert session.state()["task_index"] == 1
    session.close()


def test_policy_module_executor_can_preserve_a_continual_world():
    from src.environments.crafter_adapter import CrafterEnvironmentAdapter
    from src.skills.contracts import ImplementationContract, ImplementationResponse
    target = {"name": "inventory_at_least", "item": "wood", "threshold": 1}
    contract = ImplementationContract((), (), (target,), {}, {}, {}, {"environment": "crafter"})
    response = ImplementationResponse("reused_module", "module:persistent", "spi:test", "spt:test", contract)
    policy = CategoricalResourcePolicy(PolicyConfig(observation_dim=192, action_count=17))
    environment = CrafterEnvironmentAdapter(seed=0, length=3)
    environment.reset()
    executor = CrafterPolicyModuleExecutor(
        "module:persistent", policy, environment, target, torch.device("cpu"), 1,
        reset_before_execute=False,
    )
    executor.execute(response, {"inventory": None})
    first_steps = environment.state()["step_count"]
    executor.execute(response, {"inventory": None})
    assert first_steps == 1
    assert environment.state()["step_count"] == 2
    environment.close()
