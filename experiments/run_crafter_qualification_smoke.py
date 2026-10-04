"""Evaluate a candidate Crafter policy against the qualification contract."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import torch

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.skills.contracts import ImplementationContract, ImplementationResponse
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _candidate_response(task, module_id):
    target = {"name": "inventory_at_least", "item": task["item"], "threshold": task["threshold"]}
    contract = ImplementationContract(
        (), (), (target,), {}, {}, {}, {"environment": "crafter", "adapter": "rgb64_inventory_v1"}
    )
    return ImplementationResponse("created_module_from_spi", module_id, "spi:candidate", "spt:candidate", contract)


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    target = {"name": "inventory_at_least", "item": config["task"]["item"], "threshold": config["task"]["threshold"]}
    module_id = "module:crafter:candidate:qualification"
    response = _candidate_response(config["task"], module_id)
    rows = []
    start = time.perf_counter()
    for seed in config["seeds"]:
        environment = CrafterEnvironmentAdapter(seed=seed, length=config["max_steps"])
        executor = CrafterPolicyModuleExecutor(module_id, policy, environment, target, device, config["max_steps"])
        for episode_index in range(config["episodes_per_seed"]):
            try:
                transition = executor.execute(response, {"inventory": None})
                contract_pass = transition.target_achieved in (True, False, "unknown")
                error = None
            except Exception as exc:
                transition = None
                contract_pass = False
                error = f"{type(exc).__name__}: {exc}"
            rows.append({
                "seed": seed, "episode": episode_index, "task_id": config["task"]["task_id"],
                "success": bool(transition is not None and transition.target_achieved is True),
                "contract_pass": contract_pass,
                "target_achieved": transition.target_achieved if transition else "unknown",
                "execution_status": transition.execution_status if transition else "exception",
                "steps": executor.last_steps, "error": error,
            })
        environment.close()
    samples = len(rows)
    successes = sum(row["success"] for row in rows)
    contracts = sum(row["contract_pass"] for row in rows)
    qualification = config["qualification"]
    success_rate = successes / samples if samples else 0.0
    contract_rate = contracts / samples if samples else 0.0
    qualified = (
        samples >= qualification["min_samples"]
        and success_rate >= qualification["success_threshold"]
        and contract_rate >= qualification["contract_threshold"]
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    result = {
        "status": "crafter_policy_module_qualification_smoke",
        "formal_result": False,
        "config": config_path,
        "git_commit": commit,
        "metadata": runtime_metadata(config, resolved),
        "device": str(device),
        "cuda_tensor_verified": next(policy.parameters()).is_cuda,
        "candidate_module_id": module_id,
        "module_registered": False,
        "task": config["task"],
        "qualification": {
            "samples": samples, "successes": successes, "contract_passes": contracts,
            "success_rate": success_rate, "contract_rate": contract_rate,
            "qualified": qualified, "config": qualification,
        },
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": "Candidate qualification smoke; no Module is accepted into the Skill Library.",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "qualification.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_qualification_smoke.yaml")
    parser.add_argument("--output", default="results/crafter_qualification_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
