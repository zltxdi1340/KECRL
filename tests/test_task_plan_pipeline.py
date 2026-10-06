from src.continual_learning.contracts import (
    InMemoryContinualLearningPipeline,
    TaskPlan,
    TaskPlanStep,
    TransitionResult,
)
from src.knowledge.contracts import InMemoryKnowledgeBank, Mechanism
from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPI, SPT


def test_task_plan_executes_prerequisite_and_target_in_one_pipeline_context():
    scope = {"environment": "crafter", "adapter": "candidate"}
    prerequisite = {"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1}
    target = {"name": "inventory_at_least", "item": "stone", "threshold": 1}
    spt = SPT("spt:plan", "gather", "v1", {})
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig())
    mechanisms = []
    requests = []
    for index, (capability, inputs) in enumerate(((prerequisite, ()), (target, (prerequisite,)))):
        contract = ImplementationContract(inputs, (), (capability,), {}, {}, {}, scope)
        request = TransitionRequest(inputs, capability, {}, scope, {})
        spi = SPI(f"spi:plan:{index}", "gather", spt.spt_id, spt.version, {}, request, contract, {}, {}, {})
        assert library.qualify(spi, f"policy:plan:{index}", 1, 1, 1) is not None
        mechanisms.append(Mechanism(f"mechanism:plan:{index}", inputs, (), capability, scope, None, "confirmed"))
        requests.append(request)
    bank = InMemoryKnowledgeBank(tuple(mechanisms))
    feedback, evidence = [], []

    def execute(response, state):
        capability = response.implementation_contract.declared_output_capabilities[0]
        inventory = dict(state.get("inventory") or {})
        inventory[capability["item"]] = int(capability["threshold"])
        return TransitionResult(
            True,
            {"before": dict(state), "after": {"inventory": inventory}},
            {}, {}, (dict(capability),), "completed",
        )

    pipeline = InMemoryContinualLearningPipeline(
        bank, library, execute, evidence.append, feedback.append,
    )
    plan = TaskPlan(
        "collect_stone",
        (
            TaskPlanStep("wood_pickaxe", "prerequisite", requests[0]),
            TaskPlanStep("stone", "target", requests[1]),
        ),
    )
    status, transitions, state = pipeline.run_task_plan(plan, (), scope, {"inventory": {}})
    assert status == "completed"
    assert len(transitions) == 2
    assert state["inventory"]["stone"] == 1
    assert [item["plan_step_role"] for item in feedback] == ["prerequisite", "target"]
    assert [item["plan_step_id"] for item in feedback] == ["wood_pickaxe", "stone"]
    assert len(evidence) == 2
