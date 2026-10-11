import json
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("crafter")

from experiments.run_crafter_wood3_natural_coverage_matched import (
    VARIANTS, _validate_config, evenly_spaced_indices, matched_training_rows,
)


def test_evenly_spaced_subsample_is_stable_and_exact_size():
    values = list(range(13, 38))
    first = evenly_spaced_indices(values, 8)
    second = evenly_spaced_indices(values, 8)
    assert len(first) == 8
    assert len(set(first.tolist())) == 8
    assert first.tolist() == second.tolist()
    assert first[0] == values[0]
    assert first[-1] == values[-1]


def test_baseline_and_mixed_training_rows_have_exact_category_budget():
    records = []
    row_index = 0
    for category in ("ready", "turn", "approach"):
        for path, count in (("baseline", 8), ("standardized_linear", 14)):
            for index in range(count):
                records.append({"row": row_index, "split": "train", "category": category,
                                "source_path_variant": path})
                row_index += 1
    baseline_counts, mixed_counts = [], []
    for category in ("ready", "turn", "approach"):
        baseline, mixed, quota = matched_training_rows(records, category)
        assert quota == 8
        assert len(baseline) == len(mixed) == 8
        assert all(records[index]["source_path_variant"] == "baseline" for index in baseline)
        assert sum(records[index]["source_path_variant"] == "baseline" for index in mixed) == 4
        assert sum(records[index]["source_path_variant"] == "standardized_linear" for index in mixed) == 4
        baseline_counts.append(len(baseline))
        mixed_counts.append(len(mixed))
    assert baseline_counts == mixed_counts == [8, 8, 8]
    assert VARIANTS == ("matched_baseline_only", "matched_mixed_coverage")


def test_matched_protocol_rejects_heldout_selection(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_natural_coverage_matched_cuda_v1.yaml").read_text())
    _validate_config(config)
    with pytest.raises(ValueError):
        _validate_config({**config, "formal_training_allowed": True})


def test_matched_protocol_requires_python_hash_seed(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "17")
    config = json.loads(Path("configs/crafter_wood3_natural_coverage_matched_cuda_v1.yaml").read_text())
    with pytest.raises(ValueError, match="PYTHONHASHSEED"):
        _validate_config(config)
