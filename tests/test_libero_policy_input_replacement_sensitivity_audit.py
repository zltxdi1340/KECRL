import json

import numpy as np
import pytest

from experiments import run_libero_policy_input_replacement_sensitivity_audit as audit


def test_input_sensitivity_config_pins_reconstructed_tasks_and_scope():
    config, *_ = audit.load_audit_config(audit.DEFAULT_CONFIG)
    assert config["task_ids"] == [3, 7]
    assert config["training"]["reconstruct_pinned_policy_weights"] is True
    assert config["scope"]["policy_actions_executed_in_environment"] is False


def test_input_sensitivity_rejects_unfixed_hidden_mode(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["evaluation"]["hidden_comparison_modes"][0] = "free_running_hidden"
    path = tmp_path / "sensitivity.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="hidden comparison modes"):
        audit.load_audit_config(path)


def test_error_summary_detects_replacement_and_target_signs():
    historical = np.zeros((2, 7), dtype=np.float32)
    current = historical.copy()
    current[1, 0] = 0.5
    current[1, 6] = -0.5
    target = historical.copy()
    target[1, 6] = 0.5
    result = audit._error_summary(
        historical_actions=historical,
        current_actions=current,
        target_actions=target,
        first_difference_tolerance=1e-7,
        time_bin_steps=10,
    )
    assert result["replacement_delta"]["first_difference_step"] == 1
    assert result["replacement_delta"]["gripper_sign_flip_count"] == 1
    assert result["current_vs_target"]["gripper_sign_mismatch_count"] == 1
