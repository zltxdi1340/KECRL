import json

import pytest

from experiments import run_libero_current_runtime_replay_audit as audit


def test_current_runtime_config_pins_read_only_scope():
    config, *_ = audit.load_runtime_config(audit.DEFAULT_CONFIG)
    assert config["repetitions_per_case"] == 3
    assert config["scope"]["official_hdf5_modified"] is False
    assert config["scope"]["private_state_used_as_policy_input"] is False


def test_current_runtime_config_rejects_unknown_case(tmp_path):
    config = json.loads(audit.DEFAULT_CONFIG.read_text())
    config["cases"][0] = "unknown_case"
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="match semantics cases"):
        audit.load_runtime_config(path)


def test_digest_is_deterministic_for_same_arrays():
    import numpy as np

    value = np.arange(6, dtype=np.float32).reshape(2, 3)
    assert audit._digest_arrays([value]) == audit._digest_arrays([value.copy()])
