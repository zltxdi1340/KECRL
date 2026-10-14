import copy
from pathlib import Path

import pytest

from experiments.run_crafter_wood3_collection_opportunity import _read
from experiments.run_crafter_wood3_spatial_actor_fresh import _validate


def test_locked_spatial_fresh_protocol_rejects_oracle_or_survival_override(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _read(Path("configs/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2.yaml"))
    _validate(config)
    changed = copy.deepcopy(config)
    changed["fresh_action_oracle_allowed"] = True
    with pytest.raises(ValueError, match="fresh_action_oracle_allowed"):
        _validate(changed)
    changed = copy.deepcopy(config)
    changed["survival_intervention_allowed"] = True
    with pytest.raises(ValueError, match="survival_intervention_allowed"):
        _validate(changed)


def test_locked_spatial_fresh_protocol_requires_fresh_seed_pairing(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _read(Path("configs/crafter_wood3_spatial_actor_fresh_cuda_pilot_v2.yaml"))
    config["fresh_action_seed_base"] += 1
    with pytest.raises(ValueError, match="fresh_action_seed_base"):
        _validate(config)
