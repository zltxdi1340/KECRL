import json
from pathlib import Path


def test_action_inventory_diagnostic_is_non_formal():
    config = json.loads(Path("configs/crafter_action_inventory_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["rollout_steps"] == 256
