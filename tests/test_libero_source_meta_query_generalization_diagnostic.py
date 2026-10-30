import json

import pytest

from experiments import run_libero_source_meta_query_generalization_diagnostic as diagnostic


def test_config_pins_disjoint_source_support_query_roles_and_scopes():
    config, _base, _policy, transfer, _family, _manifest = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG
    )
    assert config["data"]["training_demonstration_indices"] == list(range(30))
    assert config["data"]["evaluation_demonstration_indices"] == list(range(30, 50))
    assert set(diagnostic.TASKS).issubset(transfer["task_roles"]["source_spi"])
    assert diagnostic.ARMS == (
        "full_policy",
        "freeze_visual",
        "freeze_visual_fusion",
        "action_head_only",
    )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("training_seeds", [20261022, 20261023], "task IDs or training seeds"),
        ("readout_arms", ["full_policy"], "readout arms"),
        ("source_transfer_split_sha256", "0" * 64, "pinned source file"),
    ],
)
def test_config_rejects_protocol_drift(tmp_path, field, value, error):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config[field] = value
    path = tmp_path / "support-query.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=error):
        diagnostic._load_protocol(path)


def test_config_rejects_query_access_before_all_support_fits(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["evaluation"]["query_dataset_created_after_all_support_fits_per_task"] = False
    path = tmp_path / "support-query.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="evaluation boundary"):
        diagnostic._load_protocol(path)
