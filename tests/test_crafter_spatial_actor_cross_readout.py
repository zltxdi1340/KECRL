import copy
from pathlib import Path

import pytest

from experiments.run_crafter_wood3_collection_opportunity import _read
from experiments.run_crafter_wood3_natural_state_cross_readout import _category
from experiments.run_crafter_wood3_spatial_actor_cross_readout import _validate


def test_spatial_cross_readout_protocol_locks_same_fresh_states(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _read(Path("configs/crafter_wood3_spatial_actor_cross_readout_cuda_pilot_v1.yaml"))
    _validate(config)
    changed = copy.deepcopy(config)
    changed["new_actions_sampled_or_executed"] = True
    with pytest.raises(ValueError, match="new_actions_sampled_or_executed"):
        _validate(changed)
    changed = copy.deepcopy(config)
    changed["episode_count_per_path_arm"] -= 1
    with pytest.raises(ValueError, match="episode_count_per_path_arm"):
        _validate(changed)


def test_cross_readout_categories_match_local_action_contract():
    ready = {"ready_to_collect": True, "awake_adjacent_opportunity": False,
             "safe_approach_move_actions": [], "sleep_override_active": False}
    assert _category(ready) == ("ready", (5,))
    turn = {"ready_to_collect": False, "awake_adjacent_opportunity": True,
            "unblocked_tree_move_actions": [2, 4], "safe_approach_move_actions": [],
            "sleep_override_active": False}
    assert _category(turn) == ("turn", (2, 4))
    approach = {"ready_to_collect": False, "awake_adjacent_opportunity": False,
                "safe_approach_move_actions": [1, 3], "sleep_override_active": False}
    assert _category(approach) == ("approach", (1, 3))
    assert _category({**approach, "sleep_override_active": True}) == ("other", ())
