"""Run one short CUDA policy smoke against the real Crafter environment."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import time
from pathlib import Path

import imageio.v2 as imageio
import torch
import torch.nn.functional as functional

from src.continual_learning.contracts import ContinualLearningPipeline
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
from src.utils.config import load_config, runtime_metadata


def _observation_tensor(observation, device):
    tensor = torch.as_tensor(observation, dtype=torch.float32, device=device)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0
    return functional.adaptive_avg_pool2d(tensor, (8, 8)).flatten()


def _git_state():
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())
    return commit, dirty


def run(config_path: str, output_path: str) -> dict:
    config = load_config(config_path)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite smoke output: {output}")
    output.mkdir(parents=True)
    device = torch.device(ContinualLearningPipeline.resolve_device(config["device"]))
    policy_config = PolicyConfig(**config["policy"])
    policy = CategoricalResourcePolicy(policy_config).to(device)
    environment = CrafterEnvironmentAdapter(seed=int(config["seed"]), length=int(config["max_steps"]))
    observation = environment.reset()
    initial_state = {name: value.detach().clone() for name, value in policy.state_dict().items()}
    log_probs, rewards = [], []
    total_reward = 0.0
    done = False
    first_frame = observation.copy()
    first_inventory = None
    final_inventory = None
    start = time.perf_counter()
    for _ in range(int(config["max_steps"])):
        tensor = _observation_tensor(observation, device)
        distribution = policy.action_distribution(tensor, tuple(range(environment.action_count)))
        action = int(distribution.sample().item())
        next_observation, reward, done, info = environment.step(action)
        log_probs.append(distribution.log_prob(torch.tensor(action, device=device)))
        rewards.append(float(reward))
        total_reward += float(reward)
        if first_inventory is None:
            first_inventory = None
        final_inventory = dict(info.get("inventory", {})) if "inventory" in info else None
        observation = next_observation
        if done:
            break
    update_loss = policy.update_episode(log_probs, rewards)
    parameter_delta = 0.0
    for name, value in policy.state_dict().items():
        parameter_delta += float((value.detach().cpu() - initial_state[name].cpu()).pow(2).sum())
    parameter_delta = parameter_delta**0.5
    elapsed = time.perf_counter() - start
    imageio.imwrite(output / "first_frame.png", first_frame)
    commit, dirty = _git_state()
    result = {
        "status": "crafter_environment_policy_smoke",
        "formal_result": False,
        "config": str(config_path),
        "git_commit": commit,
        "git_dirty_at_capture": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "cuda_tensor_verified": next(policy.parameters()).is_cuda,
        "observation_shape": list(environment.observation_shape),
        "observation_dtype": str(observation.dtype),
        "action_count": environment.action_count,
        "steps": len(rewards),
        "done": done,
        "total_reward": total_reward,
        "policy_update_loss": update_loss,
        "parameter_delta_l2": parameter_delta,
        "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(device),
        "elapsed_seconds": elapsed,
        "public_transition_evidence": {
            "before_state": {"inventory": first_inventory},
            "after_state": {"inventory": final_inventory},
            "public_observation": {"observation_shape": list(environment.observation_shape)},
            "environment_scope": {"environment": "crafter", "adapter": "rgb64_inventory_v1"},
            "intervention_metadata": {"performed": False},
            "evidence_validity": "unknown",
        },
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with (output / "result.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["steps", "done", "total_reward", "policy_update_loss", "parameter_delta_l2", "formal_result"])
        writer.writeheader()
        writer.writerow({key: result[key] for key in writer.fieldnames})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_gpu_smoke.yaml")
    parser.add_argument("--output", default="results/crafter_gpu_smoke")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
