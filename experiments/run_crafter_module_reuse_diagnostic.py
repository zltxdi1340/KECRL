"""Exercise a fixture-qualified Crafter Module through the real GPU Pipeline path."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.continual_learning.contracts import InMemoryContinualLearningPipeline, TaskVersionView
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.knowledge.contracts import InMemoryKnowledgeBank, Mechanism
from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPI, SPT
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _build_fixture(config):
    task = config["task"]
    target = {"name": "inventory_at_least", "item": task["item"], "threshold": task["threshold"]}
    scope = {"environment": "crafter", "adapter": "rgb64_inventory_v1"}
    contract = ImplementationContract((), (), (target,), {}, {}, {}, scope)
    request = TransitionRequest((), target, {}, scope, {})
    spt = SPT("spt:diagnostic:gather", task["skill_family"], "diagnostic-v1", {"fixture": True})
    spi = SPI(
        "spi:diagnostic:collect_wood", task["skill_family"], spt.spt_id, spt.version,
        {"task_id": task["task_id"]}, request, contract, {"target": target}, {}, {},
    )
    qualification = config["fixture_qualification"]
    library = InMemoryQualifiedSkillLibrary(
        spt, QualificationConfig(
            min_samples=int(qualification["min_samples"]),
            success_threshold=1.0,
            contract_threshold=1.0,
        )
    )
    module = library.qualify(
        spi, "policy:crafter:fixture", int(qualification["successes"]),
        int(qualification["contract_passes"]), int(qualification["min_samples"]),
    )
    if module is None:
        raise RuntimeError("fixture qualification unexpectedly failed")
    mechanism = Mechanism("fixture:collect_wood", (), (), target, scope, None, "confirmed")
    return target, scope, request, spt, spi, module, library, InMemoryKnowledgeBank((mechanism,))


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("diagnostic config must keep formal_result=false")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fixture reuse diagnostic requires CUDA and refuses CPU fallback")
    device = torch.device("cuda")
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    target, scope, request, spt, spi, module, library, bank = _build_fixture(config)
    response = library.request_implementation(request)
    if response.status != "reused_module" or response.module_id != module.module_id:
        raise RuntimeError("qualified fixture Module was not reused")
    environment = CrafterEnvironmentAdapter(seed=int(config["seed"]), length=int(config["max_steps"]))
    executor = CrafterPolicyModuleExecutor(
        module.module_id, policy, environment, target, device, int(config["max_steps"])
    )
    knowledge_evidence = []
    skill_feedback = []
    pipeline = InMemoryContinualLearningPipeline(
        bank, library,
        executor.execute,
        knowledge_evidence.append,
        skill_feedback.append,
        TaskVersionView("kb:fixture-diagnostic", {spt.skill_family: spt.version}),
    )
    start = time.perf_counter()
    task_result, transition = pipeline.run_transition(
        target, (), scope, request, {"inventory": None}
    )
    elapsed = time.perf_counter() - start
    environment.close()
    result = {
        "status": config["status"], "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, "cuda"), "device": "cuda",
        "cuda_tensor_verified": next(policy.parameters()).is_cuda,
        "fixture_qualification": True, "fixture_mechanism": True,
        "qualification_record": asdict(module) if module else None,
        "implementation_response": asdict(response),
        "task_result": task_result,
        "transition": asdict(transition) if transition else None,
        "knowledge_evidence": [asdict(item) for item in knowledge_evidence],
        "skill_feedback": skill_feedback,
        "task_view": {"knowledge_version": "kb:fixture-diagnostic", "spt_version": spt.version},
        "spt_pointer_switched": False, "knowledge_evolution_updated": False,
        "formal_module_claim": False,
        "elapsed_seconds": elapsed,
        "diagnostic_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_module_reuse_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_module_reuse_diagnostic_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
