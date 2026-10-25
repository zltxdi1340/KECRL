import json

import pytest

from experiments import run_libero_source_support_replay_feasibility_audit as audit


def test_support_replay_config_pins_all_source_support_demos():
    config, *_ = audit.load_audit_config(audit.DEFAULT_CONFIG)
    assert config["task_ids"] == [0, 1, 3, 4, 7]
    assert config["demonstration_indices"] == list(range(30))
    assert config["scope"]["policy_actions_executed_in_environment"] is False


def test_support_replay_config_rejects_validation_task(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["task_ids"][-1] = 2
    path = tmp_path / "support.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="five source-SPI tasks"):
        audit.load_audit_config(path)


def test_replay_summary_reports_prefix_coverage():
    result = audit._summarize([
        {"complete_action_prefix": True, "success": True, "coverage_ratio": 1.0,
         "hdf5_action_count": 4, "executed_action_count": 4, "terminal_reason": "success"},
        {"complete_action_prefix": False, "success": False, "coverage_ratio": 0.5,
         "hdf5_action_count": 4, "executed_action_count": 2, "terminal_reason": None},
    ])
    assert result["complete_action_prefix_count"] == 1
    assert result["complete_action_prefix_rate"] == pytest.approx(0.5)
    assert result["minimum_coverage_ratio"] == pytest.approx(0.5)
