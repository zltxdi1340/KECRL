import importlib.util
import json
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
_PATH = ROOT / "experiments/run_libero_single_spi_bc_diagnostic.py"
_SPEC = importlib.util.spec_from_file_location("run_libero_single_spi_bc_diagnostic", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_DIAGNOSTIC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_DIAGNOSTIC)


def test_component_metrics_keep_gripper_and_arm_errors_separate():
    prediction = torch.zeros(1, 2, 7)
    target = torch.zeros_like(prediction)
    target[0, 0, :3] = 1
    target[0, 0, 6] = -1
    mask = torch.tensor([[True, False]])
    result = _DIAGNOSTIC._finish_metrics(
        _DIAGNOSTIC._metric_sums(prediction, target, mask)
    )
    assert result["valid_steps"] == 1
    assert result["translation_mse"] == pytest.approx(1.0)
    assert result["rotation_mse"] == 0.0
    assert result["gripper_mse"] == pytest.approx(1.0)
    assert result["gripper_sign_accuracy"] == 0.0


def test_online_scope_rejects_qualification_and_nonsource_tasks():
    transfer = json.loads(
        (ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json").read_text()
    )
    with pytest.raises(ValueError, match="candidate_nonregression"):
        _DIAGNOSTIC.validate_online_state_scope(transfer, 0, "qualification", [20])
    with pytest.raises(ValueError, match="source SPI"):
        _DIAGNOSTIC.validate_online_state_scope(
            transfer, transfer["task_roles"]["spt_validation_spi"][0],
            "candidate_nonregression", [0]
        )


def test_diagnostic_budget_is_fixed_and_logging_only():
    diagnostic, policy, transfer, family, dataset = _DIAGNOSTIC._load_protocols(
        ROOT / "configs/libero_spatial_single_spi_bc_diagnostic_v1.json"
    )
    assert diagnostic["training"]["optimizer_updates"] == 500
    assert diagnostic["training"]["log_checkpoints"][-1] == 500
    assert diagnostic["training"]["checkpoint_selection"] is False
    assert policy["formal_result"] is False
    assert transfer["formal_result"] is False
    assert family["formal_result"] is False
    assert dataset["formal_result"] is False
