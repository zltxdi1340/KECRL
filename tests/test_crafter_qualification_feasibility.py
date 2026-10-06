import json
from pathlib import Path


def test_qualification_feasibility_pilot_preserves_formal_gate():
    config = json.loads(Path("configs/crafter_qualification_feasibility_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["train_episodes"] == 20
    assert config["qualification_episodes"] == 20
    assert config["qualification_threshold"] == 0.8
    assert "non-formal" in config["pilot_note"]
