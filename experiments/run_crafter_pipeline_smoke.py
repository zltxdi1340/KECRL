"""Execute one real Crafter transition through the KECRL pipeline contract."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import crafter

from src.continual_learning.contracts import InMemoryContinualLearningPipeline, TaskVersionView
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import crafter_knowledge_evidence, crafter_transition_result
from src.knowledge.contracts import InMemoryKnowledgeBank, Mechanism
from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPT, SPI


def run(output_path: str) -> dict:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    target = {"name": "inventory_at_least", "item": "wood", "threshold": 1}
    scope = {"environment": "crafter", "adapter": "rgb64_inventory_v1"}
    # This confirmed entry is a pipeline smoke fixture, not a learned Crafter
    # mechanism. Formal mechanism evidence requires a separate verifier.
    mechanism = Mechanism("smoke:collect_wood", (), (), target, scope, None, "confirmed")
    bank = InMemoryKnowledgeBank((mechanism,))
    spt = SPT("spt:smoke:gather", "gather", "smoke-v1", {"fixture": True})
    contract = ImplementationContract((), (), (target,), {}, {}, {}, scope)
    request = TransitionRequest((), target, {}, scope, {})
    spi = SPI("spi:smoke:collect_wood", "gather", spt.spt_id, spt.version, {}, request, contract, {}, {}, {})
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig(min_samples=1, success_threshold=1.0, contract_threshold=1.0))
    module = library.qualify(spi, "crafter-smoke-fixture", 1, 1, 1)
    if module is None:
        raise RuntimeError("pipeline smoke fixture qualification failed")
    environment = CrafterEnvironmentAdapter(seed=0, length=1)
    environment.reset()
    before_inventory = environment.state()["inventory"]
    evidence, feedback = [], []

    def execute(response, current_state):
        del response, current_state
        _, _, done, info = environment.step(0)
        after_inventory = info.get("inventory")
        return crafter_transition_result(before_inventory, after_inventory, target, done)

    pipeline = InMemoryContinualLearningPipeline(
        bank, library, execute, evidence.append, feedback.append,
        TaskVersionView("kb:crafter-smoke", {spt.skill_family: spt.version}),
    )
    task_result, transition = pipeline.run_transition(
        target, (), scope, request, {"inventory": before_inventory}
    )
    explicit_evidence = crafter_knowledge_evidence(
        before_inventory, transition.observed_state_changes["after"]["inventory"], scope
    )
    environment.close()
    result = {
        "status": "crafter_pipeline_smoke",
        "formal_result": False,
        "crafter_version": getattr(crafter, "__version__", "distribution-1.8.3"),
        "fixture_mechanism": True,
        "task_result": task_result,
        "transition": asdict(transition) if transition else None,
        "knowledge_evidence": [asdict(item) for item in evidence],
        "explicit_evidence_check": asdict(explicit_evidence),
        "skill_feedback": feedback,
        "task_view": {"knowledge_version": "kb:crafter-smoke", "spt_version": spt.version},
        "qualification": {"performed": False, "fixture_only": True},
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/crafter_pipeline_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
