import json
from pathlib import Path


def test_auxiliary_gather_wood3_pilot_is_non_formal_and_isolates_first_prerequisite():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood3_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert [task["task_id"] for task in config["tasks"]] == ["gather_wood_3"]
    assert config["tasks"][0]["threshold"] == 3
    assert config["progress_bonus"] > 0
