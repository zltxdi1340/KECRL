import json
from pathlib import Path


def test_ppo_pilot_is_non_formal_and_uses_exact_unit_target():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["task"] == {"task_id": "gather_wood_1", "item": "wood", "threshold": 1}
    assert config["policy"]["gae_lambda"] == 0.95
