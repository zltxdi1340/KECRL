import json
from pathlib import Path

from experiments.audit_crafter_task_local_composition_candidate import audit


def test_task_local_composition_audit_preserves_primary_targets():
    config = json.loads(Path("configs/crafter_task_local_composition_candidate_v1.yaml").read_text())
    report = audit(config)
    assert report["checks_passed"]
    assert report["formal_training_allowed"] is False
    assert all(item["target"]["threshold"] == 1 for item in config["primary_tasks"])
