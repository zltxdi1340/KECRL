import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import run_libero_bc_causal_alignment_diagnostic as diagnostic


def _protocol_stubs():
    base = {"task_id": 0, "seed": 20261021, "training": {"optimizer_updates": 500}}
    return base, {"policy": {}}, {}, {}, {}


def test_comparison_config_accepts_only_the_matched_two_arm_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostic, "_load_protocols", lambda _path: _protocol_stubs())
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["base_diagnostic_config"] = "configs/libero_spatial_single_spi_bc_diagnostic_v1.json"
    path = tmp_path / "comparison.json"
    path.write_text(json.dumps(config))

    loaded = diagnostic.load_comparison_config(path)
    assert loaded[0]["training"]["optimizer_updates"] == 500

    config["training_arms"].reverse()
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="stored_same_index then causal_pre_action"):
        diagnostic.load_comparison_config(path)


def test_official_initial_states_use_pinned_legacy_torch_format(tmp_path: Path):
    path = tmp_path / "task.pruned_init"
    expected = np.arange(5 * 92, dtype=np.float64).reshape(5, 92)
    torch.save(expected, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    selected = diagnostic._capture_official_initial_states(
        initial_path=path,
        expected_size=path.stat().st_size,
        expected_sha256=digest,
        indices=(1, 3),
    )

    np.testing.assert_array_equal(selected, expected[[1, 3]])
    with pytest.raises(ValueError, match="pinned metadata"):
        diagnostic._capture_official_initial_states(
            initial_path=path,
            expected_size=path.stat().st_size,
            expected_sha256="0" * 64,
            indices=(1,),
        )


def test_official_initial_state_indices_cannot_escape_file(tmp_path: Path):
    path = tmp_path / "task.pruned_init"
    torch.save(np.zeros((2, 92), dtype=np.float64), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="evaluation state exceeds"):
        diagnostic._capture_official_initial_states(
            initial_path=path,
            expected_size=path.stat().st_size,
            expected_sha256=digest,
            indices=(2,),
        )
