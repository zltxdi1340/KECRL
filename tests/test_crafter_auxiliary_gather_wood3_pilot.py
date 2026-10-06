import json
from pathlib import Path


def test_auxiliary_gather_wood3_pilot_is_non_formal_and_isolates_first_prerequisite():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood3_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert [task["task_id"] for task in config["tasks"]] == ["gather_wood_3"]
    assert config["tasks"][0]["threshold"] == 3
    assert config["progress_bonus"] > 0


def test_auxiliary_gather_wood3_longer_budget_remains_non_formal():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood3_pilot_80_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["train_episodes"] == 80
    assert config["qualification_episodes"] == 20


def test_auxiliary_gather_wood3_representation_pilot_compares_rgb_backends():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood3_representation_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["representations"] == ["avgpool_linear", "cnn"]
    assert config["tasks"][0]["threshold"] == 3


def test_auxiliary_gather_wood3_horizon_pilot_changes_only_horizon():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood3_horizon512_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["max_steps"] == 512
    assert config["qualification_episodes"] == 20
