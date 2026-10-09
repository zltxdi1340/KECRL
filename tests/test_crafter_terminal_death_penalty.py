import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import run_crafter_wood3_spatial_representation_curve as spatial
from experiments.run_crafter_wood3_death_penalty_diagnostic import (
    _curve_summary,
    _validate_comparison,
)


class CapturePolicy:
    def __init__(self):
        self.returns = None

    def distribution_value(self, image, hidden, actions):
        return torch.distributions.Categorical(logits=torch.zeros(1)), torch.zeros(()), None

    def update(self, features, actions, log_probs, returns, advantages, *args, **kwargs):
        self.returns = returns.clone()
        return {"loss": 0.0}


def _rollout_config(penalty):
    return {
        "max_steps": 2, "environment_length": 10000,
        "total_train_steps": 2, "bootstrap_on_truncation": True,
        "collect_failure_diagnostics": True, "action_allowlist": [0],
        "success_bonus": 1.0, "progress_bonus": 0.5,
        "terminal_death_penalty": penalty,
        "policy": {"action_count": 1, "gamma": 0.5, "gae_lambda": 0.8},
    }


def _environment(monkeypatch, terminal, success=False):
    class FakeEnvironment:
        def __init__(self, **kwargs):
            self.step_count = 0

        def reset(self):
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def step(self, action):
            self.step_count += 1
            final = self.step_count == 2
            done = final and terminal != "external_truncation"
            return self.reset(), 0.0, done, {
                "inventory": {"wood": 3 if success and final else 0},
                "diagnostics": {"terminal_reason": terminal if done else None},
            }

        def close(self):
            pass

    monkeypatch.setattr(spatial, "CrafterEnvironmentAdapter", FakeEnvironment)


def test_death_penalty_is_added_once_to_final_training_reward(monkeypatch):
    _environment(monkeypatch, "death")
    baseline = CapturePolicy()
    penalized = CapturePolicy()
    target = {"item": "wood", "threshold": 3}
    device = torch.device("cpu")
    spatial._rollout(baseline, 0, target, device, _rollout_config(0.0), True)
    row = spatial._rollout(penalized, 0, target, device, _rollout_config(-1.0), True)

    assert torch.allclose(penalized.returns - baseline.returns, torch.tensor([-0.4, -1.0]))
    assert row["reward_components"]["terminal_death_penalty"] == -1.0
    assert row["native_reward"] == 0.0


@pytest.mark.parametrize("terminal,success", [("external_truncation", False), ("environment_horizon", False), ("death", True)])
def test_penalty_excludes_truncation_and_task_success(monkeypatch, terminal, success):
    _environment(monkeypatch, terminal, success)
    policy = CapturePolicy()
    row = spatial._rollout(policy, 0, {"item": "wood", "threshold": 3}, torch.device("cpu"), _rollout_config(-1.0), True)
    assert "terminal_death_penalty" not in row["reward_components"]


def test_evaluation_death_does_not_apply_penalty_or_update(monkeypatch):
    _environment(monkeypatch, "death")
    policy = CapturePolicy()
    row = spatial._rollout(policy, 0, {"item": "wood", "threshold": 3}, torch.device("cpu"), _rollout_config(-1.0), False)
    assert row["terminal_reason"] == "death"
    assert "terminal_death_penalty" not in row["reward_components"]
    assert policy.returns is None


def test_penalty_configuration_matches_v3_training_boundaries(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    base = json.loads(Path("configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml").read_text())
    config = json.loads(Path("configs/crafter_wood3_death_penalty_cuda_pilot_v1.yaml").read_text())
    spatial._validate_matched_config(config)
    changed = {key for key in base if base[key] != config[key]}
    assert changed == {"status", "pilot_note", "representations"}
    assert config["terminal_death_penalty"] == -1.0


@pytest.mark.parametrize("penalty", [1.0, float("nan"), float("inf")])
def test_penalty_config_rejects_positive_or_nonfinite_rewards(monkeypatch, penalty):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_death_penalty_cuda_pilot_v1.yaml").read_text())
    config["terminal_death_penalty"] = penalty
    with pytest.raises(ValueError, match="finite and non-positive"):
        spatial._validate_matched_config(config)


def test_diagnostic_rejects_an_additional_training_change(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    baseline = json.loads(Path("configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml").read_text())
    config = json.loads(Path("configs/crafter_wood3_death_penalty_cuda_pilot_v1.yaml").read_text())
    _validate_comparison(config, baseline)
    config["policy"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="baseline configuration differs"):
        _validate_comparison(config, baseline)


def test_curve_summary_compares_seed_rates_when_episode_counts_differ():
    runs = []
    for episodes, deaths, successes in [(10, 8, 1), (20, 4, 14)]:
        runs.append({
            "training_curve": [{
                "actual_interaction_steps": 100,
                "episodes_in_interval": episodes,
                "successes_in_interval": successes,
                "terminal_counts": {"death": deaths, "external_truncation": episodes - deaths - successes},
                "wood_milestone_episode_counts": {"1": episodes, "2": successes, "3": successes},
                "death_stage_counts": {"before_wood1": 0, "after_wood1_before_wood2": deaths, "after_wood2_before_wood3": 0},
                "mean_episode_steps": 100 / episodes,
                "mean_ppo_metrics": {name: 0.0 for name in ("entropy", "approx_kl", "clip_fraction", "value_loss", "explained_variance")},
            }],
            "development_evaluations": [{"development": {"success_rate": successes / episodes}}],
        })
    row = _curve_summary(runs, {"checkpoint_steps": [100]})[0]
    assert row["interval_death_rate_mean"] == pytest.approx(0.5)
    assert row["interval_training_success_rate_mean"] == pytest.approx(0.4)
    assert row["interval_external_truncation_rate_mean"] == pytest.approx(0.1)
    assert row["interval_wood_milestone_rates_mean"]["1"] == 1.0
