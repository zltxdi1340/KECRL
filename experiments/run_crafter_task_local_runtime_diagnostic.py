"""Run one task-local Crafter plan through real execution boundaries.

This diagnostic uses fixture-qualified Modules only to exercise routing. It
does not train, qualify, register, or persist a formal Module and remains
explicitly non-formal.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from src.continual_learning.contracts import InMemoryContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_task_plan import build_crafter_task_plan
from src.knowledge.contracts import InMemoryKnowledgeBank, Mechanism
from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPI, SPT
from src.skills.crafter_task_plan_executor import CrafterTaskPlanExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _candidate_config() -> dict:
    return load_config("configs/crafter_task_local_composition_candidate_v1.yaml")


def _fixture_library(plan, scope):
    spt = SPT("spt:task-local-runtime-fixture", "task_local_fixture", "v1", {"fixture": True})
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig())
    mechanisms = []
    for index, step in enumerate(plan.steps):
        request = step.request
        contract = ImplementationContract(
            request.input_capabilities, (), (dict(request.target_capability),), {}, {}, {}, scope
        )
        spi = SPI(
            f"spi:task-local-runtime:{step.step_id}", "task_local_fixture", spt.spt_id,
            spt.version, {"step_id": step.step_id}, request, contract, {"fixture": True}, {}, {},
        )
        module = library.qualify(spi, f"policy:task-local-runtime:{step.step_id}", 1, 1, 1)
        if module is None:  # pragma: no cover - QualificationConfig(1, 1, 1) is deterministic.
            raise RuntimeError(f"fixture Module qualification failed for {step.step_id}")
        mechanisms.append(Mechanism(
            f"mechanism:task-local-runtime:{index}", request.input_capabilities, (),
            request.target_capability, scope, None, "confirmed",
        ))
    return library, InMemoryKnowledgeBank(mechanisms)


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False or config.get("fixture_modules") is not True:
        raise ValueError("runtime diagnostic requires formal_result=false and fixture_modules=true")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    candidate = _candidate_config()
    scope = {"environment": "crafter", "adapter": candidate["observation_interface"]}
    plan = build_crafter_task_plan(candidate, config["task_id"], scope)
    library, bank = _fixture_library(plan, scope)
    resolved = "cuda" if torch.cuda.is_available() else "cpu"
    if config["device"] == "cuda" and resolved != "cuda":
        raise RuntimeError("device=cuda requested but CUDA is unavailable; refusing CPU fallback")
    device = torch.device(resolved)
    start = time.perf_counter()
    episodes = []
    for seed in config["seeds"]:
        environment = CrafterEnvironmentAdapter(seed=int(seed), length=int(config["max_steps"]))
        policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
        executors = {}
        for step in plan.steps:
            module = next(item for item in library.modules.values() if item.spi_id.endswith(step.step_id))
            executors[module.module_id] = CrafterPolicyModuleExecutor(
                module.module_id, policy, environment, step.request.target_capability,
                device, int(config["max_steps"]), reset_before_execute=False,
            )
        plan_executor = CrafterTaskPlanExecutor(executors)
        evidence, feedback = [], []
        pipeline = InMemoryContinualLearningPipeline(
            bank, library, plan_executor.execute, evidence.append, feedback.append,
        )
        plan_executor.begin_episode()
        status, transitions, state = pipeline.run_task_plan(
            plan, (), scope, {"inventory": None, "world_object_setup": None},
        )
        episodes.append({
            "seed": int(seed), "task_result": status,
            "step_ids_executed": [item.get("plan_step_id") for item in feedback],
            "step_results": [item.execution_status for item in transitions],
            "target_achieved": [item.target_achieved for item in transitions],
            "episode_steps": environment.state()["step_count"],
            "final_public_state": dict(state),
            "knowledge_evidence_validities": [item.evidence_validity for item in evidence],
            "skill_feedback_count": len(feedback),
        })
        environment.close()
    result = {
        "status": config["status"], "formal_result": False, "fixture_modules": True,
        "config": config_path, "candidate_config": "configs/crafter_task_local_composition_candidate_v1.yaml",
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": bool(torch.cuda.is_available() and device.type == "cuda"),
        "task_id": config["task_id"], "plan_step_ids": [step.step_id for step in plan.steps],
        "episodes": episodes, "knowledge_evolution_updated": False,
        "formal_module_claim": False, "elapsed_seconds": time.perf_counter() - start,
        "diagnostic_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_task_local_runtime_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_task_local_runtime_diagnostic_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
