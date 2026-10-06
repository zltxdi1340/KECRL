import json
from pathlib import Path


def test_representation_pilot_is_non_formal_and_uses_disjoint_roles():
    config = json.loads(Path("configs/crafter_policy_representation_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["representations"] == ["avgpool_linear", "cnn"]
    assert config["train_episodes"] == 40
    assert config["qualification_episodes"] == 10
    assert config["train_seed_base"] != config["qualification_seed_base"]
    assert "does not change the formal" in config["pilot_note"]


def test_horizon_pilot_keeps_candidate_horizon_non_formal():
    config = json.loads(Path("configs/crafter_policy_horizon_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["max_steps"] == 512
    assert config["representations"] == ["cnn"]
    assert {task["task_id"] for task in config["tasks"]} == {"collect_stone", "collect_coal"}
    assert "does not change the formal 256-step" in config["pilot_note"]
