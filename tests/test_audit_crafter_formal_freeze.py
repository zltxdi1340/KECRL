import json
from pathlib import Path

from experiments.audit_crafter_formal_freeze import audit


def test_crafter_formal_freeze_candidate_is_internally_consistent():
    config = json.loads(Path("configs/crafter_formal_freeze_candidate_v1.yaml").read_text())
    result = audit(config)
    assert result["checks_passed"] is True
    assert result["formal_result"] is False
    assert result["ready_for_formal_training"] is False
    assert result["candidate_values_require_validation"] is True


def test_crafter_formal_freeze_rejects_training_enablement():
    config = json.loads(Path("configs/crafter_formal_freeze_candidate_v1.yaml").read_text())
    config["freeze_gates"]["formal_training_allowed"] = True
    result = audit(config)
    assert result["checks_passed"] is False
    assert result["ready_for_formal_training"] is False
