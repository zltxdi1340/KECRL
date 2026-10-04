"""Run a context-conditioned policy initialization smoke on Crafter."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_tasks import inventory_at_least
from src.skills.context_policy import ContextConditionedPolicyInitializer
from src.skills.crafter_policy_module import CrafterPolicyModuleExecutor
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _evaluate(policy, seed, target, device, max_steps):
    environment = CrafterEnvironmentAdapter(seed=seed, length=max_steps)
    observation = environment.reset()
    success = False
    native_reward = 0.0
    steps = 0
    with torch.no_grad():
        for steps in range(1, max_steps + 1):
            features = CrafterPolicyModuleExecutor._observation_tensor(observation, device)
            distribution = policy.action_distribution(features, tuple(range(environment.action_count)))
            observation, reward, done, info = environment.step(int(distribution.sample().item()))
            native_reward += float(reward)
            if inventory_at_least(info.get("inventory"), target["item"], target["threshold"]) is True:
                success = True
                break
            if done:
                break
    environment.close()
    return {"seed": seed, "success": success, "steps": steps, "native_reward": native_reward}


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    torch.manual_seed(int(config.get("seed", 0)))
    resolved = ContinualLearningPipeline.resolve_device(config["device"])
    device = torch.device(resolved)
    template = CategoricalResourcePolicy(PolicyConfig(**config["policy"])).to(device)
    initializer = ContextConditionedPolicyInitializer(template, int(config["context_dim"])).to(device)
    template_before = {name: value.detach().cpu().clone() for name, value in template.state_dict().items()}
    rows = {}
    start = time.perf_counter()
    for name, values in config["contexts"].items():
        context = torch.tensor(values, dtype=torch.float32, device=device)
        policy = initializer.initialize(context)
        rows[name] = {
            "context": values,
            "evaluation": [_evaluate(policy, int(seed), config["task"], device, int(config["max_steps"])) for seed in config["query_seeds"]],
            "logits_checksum": float(policy.network(torch.zeros(config["policy"]["observation_dim"], device=device)).sum().detach().cpu()),
        }
    unchanged = all(torch.equal(template_before[name], value.detach().cpu()) for name, value in template.state_dict().items())
    result = {
        "status": config["status"],
        "formal_result": False,
        "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, resolved),
        "device": str(device),
        "cuda_tensor_verified": next(template.parameters()).is_cuda,
        "template_unchanged": unchanged,
        "contexts_generate_distinct_initialization": len({row["logits_checksum"] for row in rows.values()}) == len(rows),
        "contexts": rows,
        "spt_candidate_updated": False,
        "module_registered": False,
        "elapsed_seconds": time.perf_counter() - start,
        "formal_result_note": config["smoke_note"],
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_context_policy_smoke.yaml")
    parser.add_argument("--output", default="results/crafter_context_policy_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
