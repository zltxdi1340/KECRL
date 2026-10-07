import json
from pathlib import Path

import torch

from experiments.run_crafter_auxiliary_ppo_pilot import PPOCrafterPolicy, _features, _inventory_features


def test_ppo_pilot_is_non_formal_and_uses_exact_unit_target():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["task"] == {"task_id": "gather_wood_1", "item": "wood", "threshold": 1}
    assert config["policy"]["gae_lambda"] == 0.95


def test_ppo_80_budget_keeps_independent_qualification():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_pilot_80_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["train_episodes"] == 80
    assert config["qualification_episodes"] == 20


def test_goal_action_allowlist_masks_only_diagnostic_actions():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["action_allowlist"] == [0, 1, 2, 3, 4, 5, 6]
    policy = PPOCrafterPolicy(config["policy"])
    distribution, _ = policy.distribution_value(torch.zeros(192), config["action_allowlist"])
    assert torch.equal(torch.nonzero(distribution.probs > 0).flatten(), torch.tensor(config["action_allowlist"]))


def test_temporal_pilot_uses_four_frame_feature_contract():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_temporal_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["frame_stack"] == 4
    observations = tuple(torch.zeros(64, 64, 3, dtype=torch.uint8).numpy() for _ in range(4))
    assert _features(observations, torch.device("cpu")).shape == (768,)


def test_cnn_pilot_preserves_spatial_rgb_contract():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_cnn_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    policy = PPOCrafterPolicy(config["policy"])
    observation = torch.zeros(3, 64, 64)
    distribution, value = policy.distribution_value(observation, config["action_allowlist"])
    assert distribution.probs.shape == (17,)
    assert value.shape == ()


def test_progress_pilot_keeps_training_only_inventory_bonus():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_progress_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["progress_bonus"] == 0.5
    assert config["action_allowlist"] == [0, 1, 2, 3, 4, 5, 6]


def test_inventory_observability_pilot_preserves_unknown_mask():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_inventory_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["policy"]["observation_dim"] == 194
    unknown = _inventory_features(None, ("wood",), torch.device("cpu"))
    known = _inventory_features({"wood": 1}, ("wood",), torch.device("cpu"))
    assert torch.equal(unknown, torch.tensor([0.0, 0.0]))
    assert torch.allclose(known, torch.tensor([1.0 / 9.0, 1.0]))
