import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("crafter")

from experiments.run_crafter_wood3_natural_coverage import (
    _split_for_episode, _validate, assert_coverage_split_boundary,
)


def test_coverage_episode_split_is_contiguous_and_episode_level():
    counts = {"train": 20, "validation": 5, "heldout": 5}
    assert _split_for_episode(0, counts) == ("train", 0)
    assert _split_for_episode(19, counts) == ("train", 0)
    assert _split_for_episode(20, counts) == ("validation", 1)
    assert _split_for_episode(24, counts) == ("validation", 1)
    assert _split_for_episode(25, counts) == ("heldout", 2)
    assert _split_for_episode(29, counts) == ("heldout", 2)
    with pytest.raises(ValueError):
        _split_for_episode(30, counts)


def test_coverage_boundary_allows_same_environment_only_with_same_split():
    rows = [
        {"source_environment_seed": 10, "source_path_variant": "baseline", "source_episode": 0,
         "split": "train", "rgb_sha256": "a", "category": "ready"},
        {"source_environment_seed": 10, "source_path_variant": "standardized_linear", "source_episode": 0,
         "split": "train", "rgb_sha256": "b", "category": "turn"},
    ]
    assert_coverage_split_boundary(rows)["episode_isolated"] is True
    rows[1]["split"] = "heldout"
    with pytest.raises(ValueError, match="environment crosses"):
        assert_coverage_split_boundary(rows)


@pytest.mark.parametrize("field", ["coverage_heldout_used_for_fit_or_selection", "encoder_updates_allowed", "formal_training_allowed"])
def test_coverage_protocol_rejects_leakage_or_training(monkeypatch, field):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_natural_coverage_cuda_v1.yaml").read_text())
    _validate(config)
    with pytest.raises(ValueError):
        _validate({**config, field: True})


def test_coverage_protocol_requires_python_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    config = json.loads(Path("configs/crafter_wood3_natural_coverage_cuda_v1.yaml").read_text())
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        _validate(config)
