import json
from pathlib import Path

import pytest

from experiments.run_crafter_split_feasibility import _validate_task_split


def test_split_feasibility_config_has_explicit_disjoint_roles():
    config = json.loads(Path("configs/crafter_split_feasibility_pilot_v1.yaml").read_text())
    _validate_task_split(config)
    assert set(config["training_task_ids"]).issubset(config["evaluation_task_ids"])
    assert config["formal_result"] is False


def test_split_feasibility_rejects_unknown_task():
    config = json.loads(Path("configs/crafter_split_feasibility_pilot_v1.yaml").read_text())
    config["training_task_ids"] = ["missing_task"]
    with pytest.raises(ValueError, match="unknown task"):
        _validate_task_split(config)
