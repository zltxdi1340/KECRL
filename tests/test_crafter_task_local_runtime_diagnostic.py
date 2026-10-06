import json
from pathlib import Path

from experiments.run_crafter_task_local_runtime_diagnostic import _candidate_config


def test_task_local_runtime_diagnostic_is_explicitly_non_formal():
    config = json.loads(Path("configs/crafter_task_local_runtime_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["fixture_modules"] is True
    assert _candidate_config()["plan_scope"] == "task_local_same_episode"
