"""Measure Crafter task learnability at several non-formal train budgets.

Each task and budget receives a fresh RGB policy. Qualification uses an
independent seed range and the candidate qualification contract, but this
pilot never registers a persistent Module, updates Knowledge, or changes an
SPT pointer.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path

import torch

from experiments.run_crafter_qualification_feasibility import _objects, _rollout
from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.skills.contracts import ImplementationResponse
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if config.get("formal_result") is not False:
        raise ValueError("learnability budget pilot requires formal_result=false")
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    start = time.perf_counter()
    rows, task_summaries = [], []
    budgets = [int(value) for value in config["train_episode_budgets"]]
    for budget_index, train_episodes in enumerate(budgets):
        for task_index, task in enumerate(config["tasks"]):
            # Every budget/task pair starts from an independently initialized policy.
            policy_seed = int(config["seed"]) + budget_index * 10000 + task_index
            torch.manual_seed(policy_seed)
            policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
            train_successes, train_losses = 0, []
            for episode in range(train_episodes):
                env_seed = int(config["train_seed_base"]) + budget_index * 100000 + task_index * 1000 + episode
                torch.manual_seed(int(config["action_seed_base"]) + budget_index * 100000 + task_index * 1000 + episode)
                summary = _rollout(policy, env_seed, {"name": "inventory_at_least", "item": task["item"], "threshold": int(task["threshold"])}, device, config, True)
                train_successes += int(summary["success"])
                train_losses.append(float(summary["loss"]))
                rows.append({"budget": train_episodes, "task_id": task["task_id"], "role": "train", "episode": episode, "env_seed": env_seed, **summary})

            target, _scope, contract, spi, _library = _objects(task, {**config, "train_episodes": train_episodes})
            response = ImplementationResponse(
                "created_module_from_spi", f"module:budget-pilot:{train_episodes}:{task['task_id']}",
                spi.spi_id, spi.spt_id, contract,
            )
            qualification_rows = []
            for episode in range(int(config["qualification_episodes"])):
                env_seed = int(config["qualification_seed_base"]) + budget_index * 100000 + task_index * 1000 + episode
                torch.manual_seed(int(config["qualification_action_seed_base"]) + budget_index * 100000 + task_index * 1000 + episode)
                env = CrafterEnvironmentAdapter(seed=env_seed, length=int(config["max_steps"]))
                executor = CrafterPolicyModuleExecutor(response.module_id, policy, env, target, device, int(config["max_steps"]))
                try:
                    transition = executor.execute(response, {"inventory": None})
                    contract_pass = transition.target_achieved in (True, False)
                    unknown = transition.target_achieved == "unknown"
                    error = None
                except Exception as exc:  # pragma: no cover
                    transition, contract_pass, unknown = None, False, False
                    error = f"{type(exc).__name__}: {exc}"
                env.close()
                row = {
                    "budget": train_episodes, "task_id": task["task_id"], "role": "qualification",
                    "episode": episode, "env_seed": env_seed,
                    "success": bool(transition is not None and transition.target_achieved is True),
                    "contract_pass": contract_pass, "unknown": unknown,
                    "target_achieved": transition.target_achieved if transition else "unknown",
                    "steps": executor.last_steps, "native_reward": None, "loss": None, "error": error,
                }
                qualification_rows.append(row)
                rows.append(row)
            successes = sum(int(row["success"]) for row in qualification_rows)
            contracts = sum(int(row["contract_pass"]) for row in qualification_rows)
            unknowns = sum(int(row["unknown"]) for row in qualification_rows)
            qualified = successes / max(len(qualification_rows), 1) >= float(config["qualification_threshold"]) and contracts == len(qualification_rows)
            task_summaries.append({
                "budget": train_episodes,
                "task_id": task["task_id"],
                "train_episodes": train_episodes,
                "train_successes": train_successes,
                "train_success_rate": train_successes / max(train_episodes, 1),
                "mean_train_loss": sum(train_losses) / max(len(train_losses), 1),
                "qualification": {
                    "samples": len(qualification_rows), "successes": successes,
                    "contract_passes": contracts, "unknowns": unknowns,
                    "success_rate": successes / max(len(qualification_rows), 1),
                    "contract_rate": contracts / max(len(qualification_rows), 1),
                    "qualified": qualified,
                },
            })
    result = {
        "status": config["status"], "formal_result": False, "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": bool(torch.cuda.is_available() and device.type == "cuda"),
        "train_episode_budgets": budgets, "task_summaries": task_summaries, "rows": rows,
        "seed_partitions_disjoint": True, "module_registration_formal": False,
        "formal_training_allowed": False, "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": config["pilot_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "episodes.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_task_learnability_budget_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_task_learnability_budget_pilot_4942156")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
