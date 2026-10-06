import json
from pathlib import Path


def test_connected_diagnostic_config_is_explicitly_non_formal():
    config = json.loads(Path("configs/crafter_fomaml_module_pipeline_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["qualification"]["success_threshold"] < 0.8
    assert "diagnostic only" in config["diagnostic_note"]
    assert set(config["role_seed_bases"]) == {
        "train_support", "train_query", "adapt_support", "qualification", "evaluation"
    }
