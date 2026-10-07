import json
from pathlib import Path

from experiments.audit_crafter_prerequisite_aware_candidate import audit


def test_dependency_aware_candidate_orders_inventory_prerequisites():
    config = json.loads(Path("configs/crafter_prerequisite_aware_protocol_candidate_v1.yaml").read_text())
    report = audit(config, "1.8.3")
    assert report["checks"]["inventory_dependency_order_valid"]
    assert report["checks"]["world_object_setup_declared"] is False
    assert report["formal_training_allowed"] is False
    assert report["checks_passed"] is False


def test_dependency_aware_v2_declares_setup_producers_without_allowing_formal_training():
    config = json.loads(Path("configs/crafter_prerequisite_aware_protocol_candidate_v2.yaml").read_text())
    report = audit(config, "1.8.3")
    assert report["checks"]["inventory_dependency_order_valid"]
    assert report["checks"]["world_object_setup_declared"]
    assert report["checks"]["formal_training_blocked"]
    assert report["formal_training_allowed"] is False
    assert report["checks_passed"] is True
