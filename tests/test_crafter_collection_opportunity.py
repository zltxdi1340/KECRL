import copy
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("crafter")
from crafter import objects

from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_collection_opportunity import _stage_summary, _validate
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order


def _controlled_environment():
    adapter = CrafterEnvironmentAdapter(seed=0, diagnostics=True)
    adapter.reset()
    world, player = adapter.environment._world, adapter.environment._player
    for entity in world.objects:
        if entity is not player:
            world.remove(entity)
    for offset in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        world[tuple(player.pos + offset)] = "grass"
    return adapter, world, player


def test_turning_toward_a_tree_changes_facing_even_when_movement_is_blocked():
    adapter, world, player = _controlled_environment()
    world[tuple(player.pos + (0, -1))] = "tree"
    snapshot = collection_snapshot(adapter)
    assert snapshot["awake_adjacent_opportunity"]
    assert not snapshot["ready_to_collect"]
    assert snapshot["unblocked_tree_move_actions"] == [3]
    position = player.pos.copy()
    adapter.step(3)
    assert np.array_equal(position, player.pos)
    assert collection_snapshot(adapter)["ready_to_collect"]
    _, _, _, info = adapter.step(5)
    assert info["inventory"]["wood"] == 1
    adapter.close()


@pytest.mark.parametrize("energy,expected_ready,wood_after", [(8, False, 0), (9, True, 1)])
def test_sleep_override_accounts_for_waking_at_full_energy(energy, expected_ready, wood_after):
    adapter, world, player = _controlled_environment()
    world[tuple(player.pos + player.facing)] = "tree"
    player.sleeping = True
    player.inventory["energy"] = energy
    snapshot = collection_snapshot(adapter)
    assert snapshot["ready_to_collect"] is expected_ready
    assert snapshot["sleep_override_active"] is not expected_ready
    _, _, _, info = adapter.step(5)
    assert info["inventory"]["wood"] == wood_after
    adapter.close()


def test_object_on_tree_blocks_collection_and_selects_object_interaction():
    adapter, world, player = _controlled_environment()
    target = tuple(player.pos + player.facing)
    world[target] = "tree"
    cow = objects.Cow(world, target)
    world.add(cow)
    snapshot = collection_snapshot(adapter)
    assert snapshot["facing_tree"]
    assert snapshot["target_object"] == "Cow"
    assert not snapshot["ready_to_collect"]
    _, _, _, info = adapter.step(5)
    assert info["inventory"]["wood"] == 0
    assert cow.health == 2
    adapter.close()


def test_collection_labels_require_opt_in_and_stay_out_of_adapter_state():
    adapter, _world, _player = _controlled_environment()
    snapshot = collection_snapshot(adapter)
    assert "position" in snapshot
    assert "position" not in adapter.state()
    adapter.diagnostics = False
    with pytest.raises(ValueError, match="diagnostic mode"):
        collection_snapshot(adapter)
    assert set(adapter.step(0)[3]) == {"inventory"}
    adapter.close()


def test_local_view_and_hostile_radius_do_not_treat_lava_as_safe_approach():
    adapter, world, player = _controlled_environment()
    for dx in range(-4, 5):
        for dy in range(-3, 4):
            world[tuple(player.pos + (dx, dy))] = "grass"
    world[tuple(player.pos + (0, -3))] = "tree"
    world[tuple(player.pos + (5, 0))] = "tree"
    world[tuple(player.pos + (0, -1))] = "lava"
    world.add(objects.Zombie(world, player.pos + (3, 0), player))
    world.add(objects.Skeleton(world, player.pos + (4, 0), player))
    snapshot = collection_snapshot(adapter)
    assert len(snapshot["visible_trees"]) == 1
    assert snapshot["nearest_visible_unblocked_tree_distance"] == 3
    assert snapshot["safe_approach_move_actions"] == []
    assert snapshot["nearby_hostile_count"] == 1
    assert snapshot["nearby_hostiles"][0]["type"] == "Zombie"
    world[tuple(player.pos + (0, -1))] = "grass"
    assert collection_snapshot(adapter)["safe_approach_move_actions"] == [3]
    adapter.close()


def test_snapshots_preserve_world_rng_rgb_and_fixed_action_trajectory():
    with stable_crafter_object_order():
        observed = CrafterEnvironmentAdapter(seed=4, diagnostics=True)
        control = CrafterEnvironmentAdapter(seed=4, diagnostics=True)
        assert np.array_equal(observed.reset(), control.reset())
        for action in [2, 3, 5, 1, 4, 0] * 5:
            rng = copy.deepcopy(observed.environment._world.random.get_state())
            rgb = observed.current_observation().copy()
            collection_snapshot(observed)
            assert _state_equal(rng, observed.environment._world.random.get_state())
            assert np.array_equal(rgb, observed.current_observation())
            left, right = observed.step(action), control.step(action)
            assert np.array_equal(left[0], right[0])
            assert left[1:] == right[1:]
            if left[2]:
                break
        observed.close()
        control.close()


def _event(step, *, adjacent=False, ready=False, action=0, gain=0):
    return {"step": step, "action": action, "wood_gain": gain, "wood_after": gain,
            "p_do": .3, "p_turn_to_unblocked_tree": .2, "p_safe_approach": .4,
            "position_after": [0, 0], "before": {
                "wood": 0, "ready_to_collect": ready, "unblocked_adjacent_tree": adjacent,
                "awake_adjacent_opportunity": adjacent, "unblocked_tree_move_actions": [3] if adjacent else [],
                "safe_approach_move_actions": [], "nearest_visible_unblocked_tree_distance": 1 if adjacent else None,
                "nearby_hostile_count": 0, "sleep_override_active": False, "target_object": None,
                "position": [0, 0]}}


def test_stage_funnel_counts_ready_visits_and_terminal_entry_without_next_decision():
    episodes = [
        {"final_wood": 0, "terminal_reason": "death", "events": [_event(1)]},
        {"final_wood": 0, "terminal_reason": "death", "events": [_event(1, adjacent=True)]},
        {"final_wood": 0, "terminal_reason": "external_truncation", "events": [_event(1, adjacent=True, ready=True)]},
        {"final_wood": 1, "terminal_reason": "death", "events": [
            _event(1, adjacent=True, ready=True), _event(2, adjacent=True, ready=True, action=5, gain=1)]},
    ]
    stage0 = _stage_summary(episodes, 0)
    assert stage0["episodes_entered"] == 4
    assert stage0["episodes_completed"] == 1
    assert stage0["failed_stage_opportunity_classes"] == {
        "no_unblocked_adjacent_tree": 1, "awake_adjacent_never_ready": 1, "ready_but_no_do": 1}
    assert stage0["ready_visits"]["visits"] == 2
    assert stage0["ready_visits"]["collected_visits"] == 1
    assert stage0["decisions"]["ready_do_rate"] == pytest.approx(1 / 3)
    stage1 = _stage_summary(episodes, 1)
    assert stage1["episodes_entered"] == 1
    assert stage1["failed_stage_opportunity_classes"] == {"no_decision_step_after_entry": 1}


def test_fresh_diagnostic_roles_match_policies_and_reject_previous_seed_reuse(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_collection_opportunity_cuda_v1.yaml").read_text())
    arms = {"baseline": json.loads(Path("configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml").read_text()),
            "auxiliary": json.loads(Path("configs/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_v1.yaml").read_text())}
    previous = json.loads(Path("configs/crafter_wood3_local_event_diagnostic_cuda_v1.yaml").read_text())
    _validate(config, arms, previous)
    config["diagnostic_seed_base"] = previous["diagnostic_seed_base"]
    with pytest.raises(ValueError, match="overlaps"):
        _validate(config, arms, previous)
    config["policy_updates_allowed"] = True
    with pytest.raises(ValueError, match="explicitly false"):
        _validate(config, arms, previous)
