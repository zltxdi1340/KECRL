"""Non-formal RGB behavior-cloning upper-bound diagnostic for Crafter wood.

The teacher may inspect Crafter's private world state only to emit action labels.
The student receives RGB frames and the public action allowlist; private state,
trajectories, and teacher internals never enter the student observation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from torch import nn

from experiments.run_crafter_auxiliary_ppo_pilot import _features
from src.continual_learning.contracts import ContinualLearningPipeline
from src.counterfactual.crafter_reference_runner import CrafterReferencePlanner
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.utils.config import load_config, runtime_metadata


class RecordingReferencePlanner(CrafterReferencePlanner):
    """Oracle planner that exposes only RGB/action training pairs."""

    def __init__(self, adapter, target, max_steps):
        super().__init__(adapter, target, max_steps)
        self.samples = []

    def _step(self, action_name: str) -> bool:
        observation = self.adapter.current_observation()
        action = self.environment.action_names.index(action_name)
        self.samples.append((observation.copy(), int(action)))
        return super()._step(action_name)


class RGBActionClassifier(nn.Module):
    def __init__(self, observation_dim: int, action_count: int, hidden_dim: int,
                 encoder: str = "avgpool", cnn_channels=(16, 32)):
        super().__init__()
        self.encoder_type = str(encoder)
        if self.encoder_type == "avgpool":
            self.network = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, action_count),
            )
            self.feature_encoder = None
        elif self.encoder_type == "spatial_cnn":
            first, second = (int(value) for value in cnn_channels)
            self.feature_encoder = nn.Sequential(
                nn.Conv2d(3, first, 5, stride=2, padding=2),
                nn.Tanh(),
                nn.Conv2d(first, second, 3, stride=2, padding=1),
                nn.Tanh(),
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
            )
            self.network = nn.Sequential(
                nn.Linear(second * 16, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, action_count),
            )
        else:
            raise ValueError(f"unsupported RGB imitation encoder: {self.encoder_type}")

    def logits(self, features):
        squeeze = False
        if self.encoder_type == "spatial_cnn":
            if features.ndim == 3:
                features = features.unsqueeze(0)
                squeeze = True
            features = self.feature_encoder(features)
        logits = self.network(features)
        return logits.squeeze(0) if squeeze else logits


def _target(task: dict) -> dict:
    return {
        "name": "inventory_at_least",
        "item": str(task["item"]),
        "threshold": int(task["threshold"]),
    }


def _teacher_episode(seed: int, target: dict, max_steps: int):
    environment = CrafterEnvironmentAdapter(seed=int(seed), length=int(max_steps))
    environment.reset()
    planner = RecordingReferencePlanner(environment, target, int(max_steps))
    result = planner.run()
    samples = list(planner.samples)
    environment.close()
    return result, samples


def _student_episode(policy, seed: int, target: dict, device, config: dict):
    environment = CrafterEnvironmentAdapter(seed=int(seed), length=int(config["max_steps"]))
    observation = environment.reset()
    allowlist = tuple(int(action) for action in config["action_allowlist"])
    success = False
    done = False
    steps = 0
    encoder = str(config["policy"].get("encoder", "avgpool"))
    with torch.no_grad():
        while steps < int(config["max_steps"]) and not done and not success:
            features = _features(observation, device, "cnn" if encoder == "spatial_cnn" else "avgpool")
            logits = policy.logits(features)
            masked = torch.full_like(logits, float("-inf"))
            masked[list(allowlist)] = logits[list(allowlist)]
            action = int(torch.argmax(masked).item())
            observation, _, done, info = environment.step(action)
            inventory = info.get("inventory")
            value = inventory.get(target["item"]) if isinstance(inventory, dict) else None
            success = isinstance(value, int) and value >= int(target["threshold"])
            steps += 1
    environment.close()
    return {"seed": int(seed), "success": bool(success), "steps": steps}


def run(config_path: str, output_path: str, action_loss_override: str | None = None) -> dict:
    config = load_config(config_path)
    if config.get("formal_result") is not False:
        raise ValueError("oracle imitation diagnostic requires formal_result=false")
    if action_loss_override is not None:
        config = dict(config)
        config["policy"] = dict(config["policy"])
        config["policy"]["action_loss"] = str(action_loss_override)
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    device = torch.device(ContinualLearningPipeline.resolve_device(config["device"]))
    target = _target(config["task"])
    start = time.perf_counter()
    teacher_rows = []
    observations, labels = [], []
    teacher_successes = 0
    encoder = str(config["policy"].get("encoder", "avgpool"))
    for seed in config["teacher_seeds"]:
        summary, samples = _teacher_episode(seed, target, int(config["max_steps"]))
        teacher_successes += int(summary.target_achieved)
        for observation, action in samples:
            features = _features(observation, device, "cnn" if encoder == "spatial_cnn" else "avgpool")
            observations.append(features.detach())
            labels.append(action)
        teacher_rows.append({"seed": int(seed), "target_achieved": summary.target_achieved,
                             "reference_steps": summary.reference_steps, "sample_count": len(samples)})
    policy = RGBActionClassifier(
        int(config["policy"]["observation_dim"]),
        int(config["policy"]["action_count"]),
        int(config["policy"]["hidden_dim"]),
        encoder=encoder,
        cnn_channels=config["policy"].get("cnn_channels", (16, 32)),
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(config["policy"]["learning_rate"]))
    if not observations:
        raise RuntimeError("oracle teacher produced no RGB/action samples")
    features = torch.stack(observations)
    targets = torch.tensor(labels, dtype=torch.long, device=device)
    torch.manual_seed(int(config["seed"]))
    action_counts = torch.bincount(
        targets, minlength=int(config["policy"]["action_count"])
    )
    loss_mode = str(config["policy"].get("action_loss", "uniform"))
    loss_weights = None
    if loss_mode == "inverse_sqrt":
        observed = action_counts > 0
        loss_weights = torch.zeros_like(action_counts, dtype=torch.float32)
        loss_weights[observed] = action_counts[observed].to(torch.float32).rsqrt()
        loss_weights[observed] *= observed.sum().to(torch.float32) / loss_weights[observed].sum()
    elif loss_mode != "uniform":
        raise ValueError(f"unsupported action_loss: {loss_mode}")
    losses = []
    for _ in range(int(config["epochs"])):
        logits = policy.logits(features)
        loss = nn.functional.cross_entropy(logits, targets, weight=loss_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    evaluation_rows = [_student_episode(policy, seed, target, device, config) for seed in config["evaluation_seeds"]]
    result = {
        "status": config["status"], "formal_result": False, "config": config_path,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "metadata": runtime_metadata(config, str(device)), "device": str(device),
        "cuda_tensor_verified": bool(torch.cuda.is_available() and device.type == "cuda"),
        "teacher": {"episodes": len(teacher_rows), "successes": teacher_successes,
                    "success_rate": teacher_successes / max(len(teacher_rows), 1),
                    "rows": teacher_rows, "samples": len(labels),
                    "action_counts": [int(value) for value in action_counts.cpu().tolist()],
                    "observed_action_count": int((action_counts > 0).sum().item())},
        "student": {"epochs": int(config["epochs"]), "final_loss": losses[-1],
                    "losses": losses, "evaluation": evaluation_rows,
                    "successes": sum(row["success"] for row in evaluation_rows),
                    "episodes": len(evaluation_rows),
                    "success_rate": sum(row["success"] for row in evaluation_rows) / max(len(evaluation_rows), 1)},
        "action_allowlist": list(config["action_allowlist"]),
        "student_encoder": encoder, "action_loss": loss_mode,
        "effective_overrides": ({"action_loss": loss_mode} if action_loss_override is not None else {}),
        "module_registered": False, "knowledge_evolution_updated": False,
        "formal_training_allowed": False, "elapsed_seconds": time.perf_counter() - start,
        "diagnostic_note": config["diagnostic_note"],
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_rgb_oracle_imitation_diagnostic_v1.yaml")
    parser.add_argument("--output", default="results/crafter_rgb_oracle_imitation_diagnostic_v1")
    parser.add_argument("--action-loss", choices=("uniform", "inverse_sqrt"))
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output, args.action_loss), indent=2))


if __name__ == "__main__":
    main()
