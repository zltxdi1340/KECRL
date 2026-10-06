import json
from pathlib import Path

import torch

from experiments.run_crafter_auxiliary_ppo_pilot import PPOCrafterPolicy


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
