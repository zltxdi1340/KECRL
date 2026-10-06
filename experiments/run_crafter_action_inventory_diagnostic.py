"""Audit Crafter action mapping and public inventory transitions."""
from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.utils.config import load_config, runtime_metadata


def _inventory_delta(before, after, item):
    if not isinstance(before, dict) or not isinstance(after, dict):
        return None
    before_value, after_value = before.get(item), after.get(item)
    if not isinstance(before_value, int) or not isinstance(after_value, int):
        return None
    return after_value - before_value


def _termination_reason(done, steps, max_steps, inventory):
    if not done:
        return "running"
    if isinstance(inventory, dict) and inventory.get("health") == 0:
        return "dead"
    if steps >= max_steps:
        return "episode_limit"
    return "environment_done"


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("action diagnostic requires formal_result=false")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")

    one_step = []
    probe = CrafterEnvironmentAdapter(seed=int(config["one_step_seed"]), length=2)
    action_names = list(getattr(probe.environment, "action_names", ()))
    observation = probe.reset()
    del observation
    for action in range(probe.action_count):
        probe.reset()
        before = probe.state()["inventory"]
        _, reward, done, info = probe.step(action)
        after = info.get("inventory")
        one_step.append({
            "action": action,
            "action_name": action_names[action] if action < len(action_names) else None,
            "wood_delta": _inventory_delta(before, after, "wood"),
            "reward": float(reward),
            "done": bool(done),
            "inventory_available": isinstance(after, dict),
        })
    probe.close()

    rollouts = []
    for seed in config["rollout_seeds"]:
        env = CrafterEnvironmentAdapter(seed=int(seed), length=int(config["rollout_steps"]))
        env.reset()
        action_counts = Counter()
        total_reward = 0.0
        max_wood = 0
        steps = 0
        done = False
        while steps < int(config["rollout_steps"]) and not done:
            action = (int(seed) + steps * 7) % env.action_count
            before = env.state()["inventory"]
            _, reward, done, info = env.step(action)
            after = info.get("inventory")
            action_counts[str(action)] += 1
            total_reward += float(reward)
            if isinstance(after, dict) and isinstance(after.get("wood"), int):
                max_wood = max(max_wood, after["wood"])
            steps += 1
            del before
        final_inventory = env.state()["inventory"]
        rollouts.append({
            "seed": int(seed),
            "steps": steps,
            "done": bool(done),
            "termination_reason": _termination_reason(done, steps, int(config["rollout_steps"]), final_inventory),
            "total_reward": total_reward,
            "max_wood": max_wood,
            "action_counts": dict(action_counts),
            "final_health": final_inventory.get("health") if isinstance(final_inventory, dict) else None,
            "final_wood": final_inventory.get("wood") if isinstance(final_inventory, dict) else None,
        })
        env.close()

    result = {
        "status": config["status"],
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, "cpu"),
        "observation_shape": [64, 64, 3],
        "action_count": len(action_names) or 17,
        "action_names": action_names,
        "one_step": one_step,
        "rollouts": rollouts,
        "knowledge_evolution_updated": False,
        "module_registered": False,
        "formal_training_allowed": False,
        "diagnostic_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_action_inventory_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_action_inventory_diagnostic_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
