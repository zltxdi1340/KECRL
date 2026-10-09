import json
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.run_crafter_wood3_local_event_diagnostic import (
    _binary_metrics,
    _probe,
    _validate,
)


def test_local_event_diagnostic_matches_frozen_baseline_config(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_local_event_diagnostic_cuda_v1.yaml").read_text())
    baseline = json.loads(Path("configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml").read_text())
    _validate(config, baseline)
    config["policy"]["learning_rate"] *= 2
    with pytest.raises(ValueError, match="policy config"):
        _validate(config, baseline)


def test_binary_probe_metrics_detect_separable_event():
    train, test = [], []
    for episode in range(10):
        rows = train if episode < 8 else test
        for step in range(20):
            positive = int(step % 4 == 0)
            embedding = np.zeros(128, dtype=np.float32)
            embedding[0] = 2.0 if positive else -2.0
            rows.append({"embedding": embedding, "action": step % 7, "event": positive})
    action_only = _probe(train, test, "event", True)
    rgb_action = _probe(train, test, "event", False)
    assert rgb_action["roc_auc"] > 0.95
    assert rgb_action["balanced_accuracy"] > action_only["balanced_accuracy"]


def test_auc_is_undefined_if_one_event_class_is_absent():
    metrics = _binary_metrics(np.asarray([0.1, 0.2]), np.asarray([0, 0]))
    assert metrics["roc_auc"] is None
    assert metrics["balanced_accuracy"] is None
