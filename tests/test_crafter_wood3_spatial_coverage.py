import copy
from pathlib import Path

import pytest

from experiments.run_crafter_wood3_collection_opportunity import _read
from experiments.run_crafter_wood3_spatial_coverage import _select_rows, _validate


def test_coverage_protocol_locks_episode_isolation_and_no_oracle(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _read(Path("configs/crafter_wood3_spatial_coverage_cuda_pilot_v1.yaml"))
    _validate(config)
    changed = copy.deepcopy(config)
    changed["collection_action_oracle_allowed"] = True
    with pytest.raises(ValueError, match="collection_action_oracle_allowed"):
        _validate(changed)
    changed = copy.deepcopy(config)
    changed["fresh_environment_seed_base"] += 1
    with pytest.raises(ValueError, match="fresh_environment_seed_base"):
        _validate(changed)


def test_coverage_selection_is_category_balanced_and_deterministic():
    records, images = [], []
    for category in ("ready", "turn", "approach"):
        for index in range(5):
            records.append({"row": len(records), "split": "train", "category": category,
                            "target_actions": [5], "rgb_sha256": str(len(records))})
            images.append([len(records)])
    selected, selected_images = _select_rows(records, images, "train",
                                             {"ready": 2, "turn": 2, "approach": 2})
    assert selected.tolist() == [0, 4, 5, 9, 10, 14]
    assert len(selected_images) == 6
