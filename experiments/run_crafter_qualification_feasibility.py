"""Test candidate train/qualification budgets for real Crafter task Modules.

Each task gets an isolated RGB policy trained with the candidate 20 episode
budget, followed by independent qualification episodes. This is a feasibility
pilot: it may create in-memory diagnostic Modules, but it never changes an
SPT pointer, Knowledge state, or formal configuration.
"""
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
from src.environments.crafter_tasks import inventory_at_least
from src.skills.contracts import ImplementationContract, ImplementationResponse, TransitionRequest
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig, SPI, SPT
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _target(task: dict) -> dict:
    return {"name": "inventory_at_least", "item": task["item"], "threshold": int(task["threshold"])}


def _rollout(policy, seed: int, target: dict, device, config: dict, train: bool):
    env = CrafterEnvironmentAdapter(seed=int(seed), length=int(config["max_steps"]))
    observation = env.reset()
    log_probs, rewards = [], []
    success, native_reward = False, 0.0
    done = False
    for step in range(int(config["max_steps"])):
        features = CrafterPolicyModuleExecutor._observation_tensor(observation, device)
        distribution = policy.action_distribution(features, tuple(range(env.action_count)))
        action = distribution.sample()
        if train:
            log_probs.append(distribution.log_prob(action))
        before_inventory = info.get("inventory") if step > 0 else None
        observation, reward, done, info = env.step(int(action.item()))
        native_reward += float(reward)
        shaped = float(reward)
        if train and before_inventory is not None:
            before_value = before_inventory.get(target["item"])
            after_value = info.get("inventory", {}).get(target["item"])
            if isinstance(before_value, int) and isinstance(after_value, int) and after_value > before_value:
                shaped += float(config.get("progress_bonus", 0.0)) * (after_value - before_value)
        if not success and inventory_at_least(info.get("inventory"), target["item"], target["threshold"]) is True:
            success = True
            shaped += float(config["success_bonus"])
        rewards.append(shaped)
        if done or success:
            break
    loss = policy.update_episode(log_probs, rewards) if train else None
    env.close()
    return {"success": success, "steps": step + 1, "native_reward": native_reward, "loss": loss}


def _objects(task: dict, config: dict):
    target = _target(task)
    scope = {"environment": "crafter", "adapter": "rgb64_inventory_v1"}
    contract = ImplementationContract((), (), (target,), {}, {}, {}, scope)
    request = TransitionRequest((), target, {}, scope, {})
    spt = SPT(f"spt:qualification-pilot:{task['skill_family']}", task["skill_family"], "qualification-pilot-v1", {"pilot": True})
    spi = SPI(
        f"spi:qualification-pilot:{task['task_id']}", task["skill_family"], spt.spt_id,
        spt.version, {"task_id": task["task_id"]}, request, contract,
        {"target": target}, {}, {},
    )
    library = InMemoryQualifiedSkillLibrary(
        spt, QualificationConfig(
            min_samples=int(config["qualification_episodes"]),
            success_threshold=float(config["qualification_threshold"]),
            contract_threshold=1.0,
        )
    )
    return target, scope, contract, spi, library


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if config.get("formal_result") is not False:
        raise ValueError("qualification feasibility pilot requires formal_result=false")
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    start = time.perf_counter()
    rows, task_summaries = [], []
    for task_index, task in enumerate(config["tasks"]):
        torch.manual_seed(int(config["seed"]) + task_index)
        policy = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
        train_successes = 0
        train_losses = []
        for episode in range(int(config["train_episodes"])):
            seed = int(config["train_seed_base"]) + task_index * 1000 + episode
            torch.manual_seed(int(config["action_seed_base"]) + task_index * 1000 + episode)
            summary = _rollout(policy, seed, _target(task), device, config, True)
            train_successes += int(summary["success"])
            train_losses.append(float(summary["loss"]))
            rows.append({"task_id": task["task_id"], "role": "train", "episode": episode, "env_seed": seed, **summary})

        target, scope, contract, spi, library = _objects(task, config)
        response = ImplementationResponse(
            "created_module_from_spi", f"module:qualification-pilot:{task['task_id']}",
            spi.spi_id, spi.spt_id, contract,
        )
        qualification_rows = []
        for episode in range(int(config["qualification_episodes"])):
            seed = int(config["qualification_seed_base"]) + task_index * 1000 + episode
            env = CrafterEnvironmentAdapter(seed=seed, length=int(config["max_steps"]))
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
                "task_id": task["task_id"], "role": "qualification", "episode": episode,
                "env_seed": seed, "success": bool(transition is not None and transition.target_achieved is True),
                "contract_pass": contract_pass, "unknown": unknown,
                "target_achieved": transition.target_achieved if transition else "unknown",
                "steps": executor.last_steps, "native_reward": None, "loss": None, "error": error,
            }
            qualification_rows.append(row); rows.append(row)
        successes = sum(int(row["success"]) for row in qualification_rows)
        contracts = sum(int(row["contract_pass"]) for row in qualification_rows)
        unknowns = sum(int(row["unknown"]) for row in qualification_rows)
        module = library.qualify(spi, f"policy:qualification-pilot:{task['task_id']}", successes, contracts, len(qualification_rows))
        task_summaries.append({
            "task_id": task["task_id"], "train_episodes": config["train_episodes"],
            "train_successes": train_successes, "train_success_rate": train_successes / max(int(config["train_episodes"]), 1),
            "mean_train_loss": sum(train_losses) / max(len(train_losses), 1),
            "qualification": {
                "samples": len(qualification_rows), "successes": successes, "contract_passes": contracts,
                "unknowns": unknowns, "success_rate": successes / max(len(qualification_rows), 1),
                "contract_rate": contracts / max(len(qualification_rows), 1), "qualified": module is not None,
            },
            "module_id": module.module_id if module else None,
        })
    result = {
        "status": config["status"], "formal_result": False, "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved), "device": str(device),
        "cuda_tensor_verified": bool(torch.cuda.is_available() and device.type == "cuda"),
        "task_summaries": task_summaries, "rows": rows,
        "all_tasks_qualified": all(item["qualification"]["qualified"] for item in task_summaries),
        "formal_training_allowed": False, "module_registration_formal": False,
        "elapsed_seconds": time.perf_counter() - start,
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
    parser.add_argument("--config", default="configs/crafter_qualification_feasibility_pilot_v1.yaml")
    parser.add_argument("--output", default="results/crafter_qualification_feasibility_pilot_eba9b2f")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
