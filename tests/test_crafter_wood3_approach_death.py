import copy

import pytest

from experiments.analyze_crafter_wood3_approach_death import (
    _approach_event, _death_row, _episode_metrics, _failure_class, _local_state,
)


def _snapshot(*, nearest=None, visible=None, safe=(), adjacent=False, ready=False,
              sleep=False, health=9, food=9, drink=9, energy=9, hostiles=0):
    visible = [] if visible is None else visible
    return {
        "visible_trees": visible, "nearest_visible_unblocked_tree_distance": nearest,
        "safe_approach_move_actions": list(safe), "sleep_override_active": sleep,
        "awake_adjacent_opportunity": adjacent, "ready_to_collect": ready,
        "position": [0, 0], "facing": [0, 1], "health": health, "food": food,
        "drink": drink, "energy": energy, "sleeping": sleep, "daylight": 0.5,
        "nearby_hostile_count": hostiles, "unblocked_tree_move_actions": [1] if adjacent else [],
    }


def _event(before, *, action=0, sampled=None, after=None, step=1, done=False,
           reason=None, health_after=None, delta=0, replaced=False):
    if sampled is None:
        sampled = action
    if health_after is None:
        health_after = before["health"] + delta
    return {
        "step": step, "action": action, "executed_action": action, "sampled_action": sampled,
        "action_replaced": replaced, "before": before, "after": after,
        "position_after": [1, 0] if action in (1, 2, 3, 4) and not done else before["position"],
        "environment_done": done, "terminal_reason": reason, "health_after": health_after,
        "health_delta": delta, "damage_source_hint": "zombie" if done else None,
        "wood_before": 0, "wood_after": 0, "wood_gain": 0,
    }


def _episode(events, *, final_wood=0, terminal_reason="external_truncation"):
    return {
        "episode": 0, "environment_seed": 1, "success": False, "steps": len(events),
        "terminal_reason": terminal_reason, "final_wood": final_wood, "events": events,
    }


def test_local_state_separates_visible_tree_and_approach_boundaries():
    tree = [{"object": None, "offset": [0, -2], "distance": 2}]
    assert _local_state(_snapshot()) == "no_visible_unblocked_tree"
    assert _local_state(_snapshot(visible=[{"object": "Zombie", "offset": [0, -1], "distance": 1}])) == "no_visible_unblocked_tree"
    assert _local_state(_snapshot(nearest=3, visible=tree)) == "visible_no_safe_approach"
    assert _local_state(_snapshot(nearest=3, visible=tree, safe=(1,))) == "approach"
    assert _local_state(_snapshot(nearest=1, visible=tree, adjacent=True)) == "turn"
    assert _local_state(_snapshot(nearest=1, visible=tree, ready=True)) == "ready"


def test_approach_event_requires_valid_action_and_measures_distance_progress():
    before = _snapshot(nearest=3, visible=[{"object": None, "offset": [0, -3], "distance": 3}], safe=(1,))
    after = _snapshot(nearest=2, visible=[{"object": None, "offset": [0, -2], "distance": 2}], safe=(1,))
    row = _approach_event(_event(before, action=1, after=after), None)
    assert row["executed_valid_approach"] and row["position_changed"] and row["distance_reduced"]
    blocked = _approach_event(_event(before, action=2, after=before), None)
    assert not blocked["executed_valid_approach"] and not blocked["distance_reduced"]
    assert _approach_event(_event(_snapshot(nearest=3, visible=[{"object": None, "offset": [0, -3], "distance": 3}]), action=1), None) is None


@pytest.mark.parametrize(("snapshot_kwargs", "expected"), [
    ({}, "no_visible_unblocked_tree_observed"),
    ({"nearest": 3, "visible": [{"object": None, "offset": [0, -3], "distance": 3}]}, "visible_tree_but_no_safe_approach"),
    ({"nearest": 3, "visible": [{"object": None, "offset": [0, -3], "distance": 3}], "safe": (1,)}, "approach_available_never_selected"),
])
def test_failure_classification_separates_approach_boundaries(snapshot_kwargs, expected):
    before = _snapshot(**snapshot_kwargs)
    action = 2 if expected == "approach_available_never_selected" else 0
    episode = _episode([_event(before, action=action, done=True, reason="death", health_after=0, delta=-9)])
    assert _failure_class(episode) == expected


def test_failure_classification_records_progress_then_later_failure():
    before = _snapshot(nearest=3, visible=[{"object": None, "offset": [0, -3], "distance": 3}], safe=(1,))
    after = _snapshot(nearest=2, visible=[{"object": None, "offset": [0, -2], "distance": 2}], safe=(1,))
    terminal_before = _snapshot(nearest=None)
    episode = _episode([
        _event(before, action=1, after=after, step=1),
        _event(terminal_before, action=0, step=2, done=True, reason="death", health_after=0, delta=-9),
    ])
    assert _failure_class(episode) == "approach_progress_without_local_opportunity"


def test_death_row_keeps_resource_and_hostile_context_and_censors_truncation():
    before = _snapshot(nearest=3, visible=[{"object": None, "offset": [0, -3], "distance": 3}], safe=(1,), health=2,
                       food=1, drink=2, energy=3, hostiles=2)
    episode = _episode([_event(before, action=0, step=4, done=True, reason="death", health_after=0, delta=-2)], terminal_reason="death")
    metrics = _episode_metrics(episode, 0, "head", "turn_ready_do")
    row = _death_row(episode, metrics)
    assert row["death_before_local_state"] == "approach"
    assert row["death_food"] == 1 and row["death_drink"] == 2 and row["death_energy"] == 3
    assert row["death_nearby_hostile_count"] == 2 and row["death_damage_source_hint"] == "zombie"
    truncated = _episode([_event(before, action=0, step=4, done=False)], terminal_reason="external_truncation")
    assert _episode_metrics(truncated, 0, "head", "turn_ready_do")["terminal_reason"] == "external_truncation"
    with pytest.raises(ValueError):
        _death_row(truncated, _episode_metrics(truncated, 0, "head", "turn_ready_do"))


def test_episode_metrics_records_distance_and_terminal_context():
    before = _snapshot(nearest=3, visible=[{"object": None, "offset": [0, -3], "distance": 3}], safe=(1,))
    after = _snapshot(nearest=2, visible=[{"object": None, "offset": [0, -2], "distance": 2}], safe=(1,))
    episode = _episode([_event(before, action=1, after=after, step=1), _event(after, action=0, step=2, done=True, reason="death", health_after=0, delta=-9)], terminal_reason="death")
    metrics = _episode_metrics(episode, 0, "head", "turn_ready_do")
    assert metrics["ever_approach_progress"] and metrics["minimum_visible_tree_distance"] == 2
    assert metrics["terminal_local_state"] == "approach"
