import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import run_crafter_wood3_spatial_representation_curve as spatial
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from experiments.run_crafter_wood3_do_wood_gain_auxiliary_diagnostic import _validate


def _config():
    return {"cnn_channels": [4, 8], "embedding_dim": 16, "recurrent_hidden_dim": 16,
            "action_count": 17, "learning_rate": 0.001, "clip_epsilon": 0.2,
            "value_coef": 0.5, "entropy_coef": 0.01, "update_epochs": 2}


def _policies(coefficient):
    torch.manual_seed(7)
    baseline = build_matched_spatial_policy(_config(), "cnn_only")
    baseline_rng = torch.get_rng_state().clone()
    torch.manual_seed(7)
    aux = build_matched_spatial_policy({**_config(), "auxiliary_target": "do_wood_gain",
                                       "auxiliary_loss_coef": coefficient}, "cnn_only")
    assert torch.equal(torch.get_rng_state(), baseline_rng)
    return baseline, aux


def _update(policy, images, actions, **kwargs):
    with torch.no_grad():
        dist, _ = policy.sequence_logits_values(images, tuple(range(7)))
    return policy.update(images, actions, dist.log_prob(actions).detach(), torch.ones(len(actions)),
                         torch.linspace(-1, 1, len(actions)), tuple(range(7)), return_metrics=True, **kwargs)


def test_zero_auxiliary_coefficient_preserves_policy_initialization_and_ppo_update():
    baseline, aux = _policies(0.0)
    images = torch.rand(6, 3, 64, 64)
    actions = torch.tensor([5, 1, 5, 2, 5, 3])
    base_metrics = _update(baseline, images, actions)
    aux_metrics = _update(aux, images, actions, auxiliary_targets=torch.ones(6), auxiliary_mask=torch.ones(6))
    assert all(base_metrics[key] == aux_metrics[key] for key in base_metrics)
    assert all(torch.equal(value, aux.state_dict()[name]) for name, value in baseline.state_dict().items())
    assert aux.auxiliary_head.weight.grad is None


def test_auxiliary_loss_updates_shared_rgb_encoder_without_label_input_to_actor():
    baseline, aux = _policies(0.1)
    images = torch.rand(6, 3, 64, 64)
    actions = torch.full((6,), 5)
    with torch.no_grad():
        before_dist, before_value, _ = baseline.distribution_value(images[0], None, tuple(range(7)))
        aux_dist, aux_value, _ = aux.distribution_value(images[0], None, tuple(range(7)))
    assert torch.equal(before_dist.logits, aux_dist.logits)
    assert torch.equal(before_value, aux_value)
    _update(baseline, images, actions)
    metrics = _update(aux, images, actions, auxiliary_targets=torch.tensor([0., 1., 0., 1., 0., 1.]),
                      auxiliary_mask=torch.ones(6))
    assert metrics["auxiliary_loss"] > 0
    assert aux.auxiliary_head.weight.grad is not None
    assert any(not torch.equal(value, aux.encoder.state_dict()[key]) for key, value in baseline.encoder.state_dict().items())


def test_unknown_and_non_do_targets_have_no_auxiliary_gradient():
    baseline, aux = _policies(0.1)
    images = torch.rand(6, 3, 64, 64)
    actions = torch.tensor([5, 1, 5, 2, 5, 3])
    _update(baseline, images, actions)
    metrics = _update(aux, images, actions, auxiliary_targets=torch.full((6,), float("nan")),
                      auxiliary_mask=torch.tensor([0., 1., 0., 1., 0., 1.]))
    assert metrics["auxiliary_loss"] == 0
    assert all(torch.equal(value, aux.state_dict()[name]) for name, value in baseline.state_dict().items())
    assert aux.auxiliary_head.weight.grad is None


@pytest.mark.parametrize("before,after,action,label,mask", [
    (None, {"wood": 1}, 5, 0., 0.),
    ({"wood": 1}, {"wood": 2}, 5, 1., 1.),
    ({"wood": 1}, {"wood": 1}, 5, 0., 1.),
    ({"wood": 1}, {"wood": 2}, 1, 1., 0.),
    ({"wood": 1}, {}, 5, 0., 0.),
    ({"wood": True}, {"wood": 2}, 5, 0., 0.),
])
def test_wood_gain_labels_preserve_unknown_and_action_masks(before, after, action, label, mask):
    assert spatial._wood_gain_label(before, after, action) == (label, mask)


def test_rollout_aligns_labels_with_pre_action_rgb_and_excludes_evaluation(monkeypatch):
    class FakeEnvironment:
        def __init__(self, **kwargs):
            self.index = 0

        def reset(self):
            return np.zeros((64, 64, 3), dtype=np.uint8)

        def step(self, action):
            wood = [0, 1, 2, None, 3][self.index]
            self.index += 1
            return np.full((64, 64, 3), self.index, dtype=np.uint8), 0., False, {
                "inventory": {} if wood is None else {"wood": wood}, "diagnostics": {}}

        def close(self):
            pass

    class CapturePolicy:
        def __init__(self):
            self.index = 0
            self.update_kwargs = None

        def distribution_value(self, image, hidden, allowed):
            action = [5, 1, 5, 5, 5][min(self.index, 4)]
            self.index += 1
            probs = torch.zeros(17)
            probs[action] = 1
            return torch.distributions.Categorical(probs=probs), torch.zeros(()), None

        def update(self, images, actions, log_probs, returns, advantages, *args, **kwargs):
            assert torch.allclose(images[:, 0, 0, 0], torch.arange(5) / 255.)
            self.update_kwargs = kwargs
            return {"loss": 0.}

    monkeypatch.setattr(spatial, "CrafterEnvironmentAdapter", FakeEnvironment)
    config = {"max_steps": 5, "total_train_steps": 5, "environment_length": 10000,
              "collect_failure_diagnostics": True, "auxiliary_target": "do_wood_gain",
              "action_allowlist": list(range(7)), "success_bonus": 1., "progress_bonus": .5,
              "policy": {"action_count": 17, "gamma": .99, "gae_lambda": .95}}
    policy = CapturePolicy()
    row = spatial._rollout(policy, 0, {"item": "wood", "threshold": 3}, torch.device("cpu"), config, True)
    assert torch.equal(policy.update_kwargs["auxiliary_targets"], torch.tensor([0., 1., 1., 0., 0.]))
    assert torch.equal(policy.update_kwargs["auxiliary_mask"], torch.tensor([0., 0., 1., 0., 0.]))
    assert row["auxiliary_samples"] == row["auxiliary_positive_samples"] == 1
    evaluation_policy = CapturePolicy()
    row = spatial._rollout(evaluation_policy, 0, {"item": "wood", "threshold": 3}, torch.device("cpu"), config, False)
    assert evaluation_policy.update_kwargs is None
    assert row["auxiliary_samples"] == 0


def test_auxiliary_pilot_config_matches_the_baseline(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    baseline = json.loads(Path("configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml").read_text())
    config = json.loads(Path("configs/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_v1.yaml").read_text())
    spatial._validate_matched_config(config)
    _validate(config, baseline)
    assert {key for key in baseline if baseline[key] != config[key]} == {"status", "pilot_note", "representations"}
    assert config["auxiliary_target"] == "do_wood_gain"
    assert config["auxiliary_loss_coef"] == 0.1
    config["auxiliary_loss_coef"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        spatial._validate_matched_config(config)
