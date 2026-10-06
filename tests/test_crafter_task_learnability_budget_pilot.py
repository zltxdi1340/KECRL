import json
from pathlib import Path


def test_task_learnability_budget_pilot_is_non_formal_and_disjoint():
    config = json.loads(Path("configs/crafter_task_learnability_budget_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["train_episode_budgets"] == [20, 80]
    assert config["qualification_episodes"] == 20
    assert config["qualification_threshold"] == 0.8
    assert config["train_seed_base"] != config["qualification_seed_base"]
    assert config["action_seed_base"] != config["qualification_action_seed_base"]
    assert "Non-formal" in config["pilot_note"]
