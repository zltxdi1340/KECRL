"""Exercise real Crafter qualification and Pipeline routing at diagnostic scale."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.continual_learning.contracts import InMemoryContinualLearningPipeline, TaskVersionView
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import crafter_knowledge_evidence
from src.knowledge.contracts import InMemoryKnowledgeBank, Mechanism
from src.skills.contracts import ImplementationContract, ImplementationResponse, TransitionRequest
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPI, SPT
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _target(task):
    return {"name": "inventory_at_least", "item": task["item"], "threshold": task["threshold"]}


def _scope():
    return {"environment": "crafter", "adapter": "rgb64_inventory_v1"}


def _build_objects(task, config):
    target = _target(task)
    scope = _scope()
    contract = ImplementationContract((), (), (target,), {}, {}, {}, scope)
    request = TransitionRequest((), target, {}, scope, {})
    spt = SPT(f"spt:diagnostic:{task['skill_family']}", task["skill_family"], "diagnostic-v1", {"task_family": task["skill_family"]})
    spi = SPI(
        f"spi:diagnostic:{task['task_id']}", task["skill_family"], spt.spt_id, spt.version,
        {"task_id": task["task_id"]}, request, contract, {"target": target}, {}, {},
    )
    qualification_config = {
        key: config["qualification"][key]
        for key in ("min_samples", "success_threshold", "contract_threshold")
    }
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig(**qualification_config))
    mechanism = Mechanism(
        f"fixture:{task['task_id']}", (), (), target, scope, None, "confirmed"
    )
    return target, scope, contract, request, spt, spi, library, InMemoryKnowledgeBank((mechanism,))


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False or config.get("fixture_mechanism") is not True:
        raise ValueError("diagnostic requires formal_result=false and fixture_mechanism=true")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    resolved = "cuda" if torch.cuda.is_available() else "cpu"
    if config["device"] == "cuda" and resolved != "cuda":
        raise RuntimeError("device=cuda requested but CUDA is unavailable; refusing CPU fallback")
    device = torch.device(resolved)
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    rows = []
    start = time.perf_counter()
    for task in config["tasks"]:
        target, scope, contract, request, spt, spi, library, bank = _build_objects(task, config)
        qualification_response = ImplementationResponse(
            "created_module_from_spi", f"module:candidate:{task['task_id']}", spi.spi_id, spt.spt_id, contract
        )
        qualification_rows = []
        for seed in config["seeds"]:
            for episode in range(int(config["qualification"]["episodes_per_seed"])):
                environment = CrafterEnvironmentAdapter(seed=int(seed) * 1000 + episode, length=config["max_steps"])
                executor = CrafterPolicyModuleExecutor(
                    qualification_response.module_id, policy, environment, target, device, config["max_steps"]
                )
                try:
                    transition = executor.execute(qualification_response, {"inventory": None})
                    contract_pass = transition.target_achieved in (True, False, "unknown")
                    error = None
                except Exception as exc:
                    transition = None
                    contract_pass = False
                    error = f"{type(exc).__name__}: {exc}"
                qualification_rows.append({
                    "seed": seed, "episode": episode,
                    "success": bool(transition is not None and transition.target_achieved is True),
                    "contract_pass": contract_pass,
                    "target_achieved": transition.target_achieved if transition else "unknown",
                    "execution_status": transition.execution_status if transition else "exception",
                    "steps": executor.last_steps, "error": error,
                })
                environment.close()
        samples = len(qualification_rows)
        successes = sum(row["success"] for row in qualification_rows)
        contract_passes = sum(row["contract_pass"] for row in qualification_rows)
        qualification = {
            "samples": samples, "successes": successes, "contract_passes": contract_passes,
            "success_rate": successes / max(samples, 1),
            "contract_rate": contract_passes / max(samples, 1),
        }
        module = library.qualify(spi, "policy:crafter:diagnostic", successes, contract_passes, samples)
        evidence, feedback, pipeline_rows = [], [], []
        for seed in config["seeds"]:
            for episode in range(int(config["query_episodes_per_seed"])):
                environment = CrafterEnvironmentAdapter(seed=100000 + int(seed) * 1000 + episode, length=config["max_steps"])
                executor = None
                if module is not None:
                    executor = CrafterPolicyModuleExecutor(module.module_id, policy, environment, target, device, config["max_steps"])

                def execute(response, current_state):
                    if executor is None:
                        raise RuntimeError("unavailable response must not reach executor")
                    return executor.execute(response, current_state)

                pipeline = InMemoryContinualLearningPipeline(
                    bank, library, execute, evidence.append, feedback.append,
                    TaskVersionView("kb:fixture-diagnostic", {spt.skill_family: spt.version}),
                )
                task_result, transition = pipeline.run_transition(
                    target, (), scope, request, {"inventory": None}
                )
                pipeline_rows.append({
                    "seed": seed, "episode": episode, "task_result": task_result,
                    "execution_status": transition.execution_status if transition else "unavailable",
                    "target_achieved": transition.target_achieved if transition else "unknown",
                    "module_registered": module is not None,
                })
                environment.close()
        rows.append({
            "task_id": task["task_id"], "skill_family": task["skill_family"],
            "qualification": qualification, "module_registered": module is not None,
            "module_id": module.module_id if module else None,
            "module_reuse": sum(row["module_registered"] for row in pipeline_rows),
            "pipeline_task_results": {status: sum(row["task_result"] == status for row in pipeline_rows) for status in ("completed", "continued", "unavailable", "unknown")},
            "qualification_rows": qualification_rows, "pipeline_rows": pipeline_rows,
            "knowledge_evidence_count": len(evidence), "skill_feedback_count": len(feedback),
            "knowledge_evidence_validities": [item.evidence_validity for item in evidence],
            "fixture_mechanism": True,
        })
    result = {
        "status": config["status"], "formal_result": False,
        "config": config_path, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": next(policy.parameters()).is_cuda,
        "fixture_mechanism": True, "knowledge_evolution_updated": False,
        "spt_pointer_switched": False, "formal_module_claim": False,
        "tasks": rows, "elapsed_seconds": time.perf_counter() - start,
        "diagnostic_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["task_id", "qualification_success_rate", "qualification_contract_rate", "module_registered", "module_reuse", "completed", "continued", "unavailable", "unknown"]
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({
                "task_id": row["task_id"],
                "qualification_success_rate": row["qualification"]["success_rate"],
                "qualification_contract_rate": row["qualification"]["contract_rate"],
                "module_registered": row["module_registered"], "module_reuse": row["module_reuse"],
                **row["pipeline_task_results"],
            })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_module_pipeline_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_module_pipeline_diagnostic_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
