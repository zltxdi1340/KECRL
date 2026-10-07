import json
from pathlib import Path


def test_oracle_pipeline_diagnostic_is_explicitly_non_formal():
    config = json.loads(Path("configs/crafter_task_local_oracle_pipeline_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["fixture_modules"] is True
    assert config["oracle_action_script"] is True
    assert config["public_boundary_only"] is True
