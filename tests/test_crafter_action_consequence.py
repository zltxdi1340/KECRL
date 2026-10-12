import copy
import json

import pytest

from experiments.analyze_crafter_wood3_action_consequence import (
    VARIANTS, _digest, _event_rows, _load_trace, _sha256, _validate_episode,
    aggregate, local_category,
)


def _snapshot(category, wood=0):
    return {
        "wood": wood, "health": 9, "position": [32, 32], "facing": [1, 0],
        "ready_to_collect": category == "ready",
        "awake_adjacent_opportunity": category in ("ready", "turn"),
        "unblocked_tree_move_actions": [2] if category in ("ready", "turn") else [],
        "safe_approach_move_actions": [2] if category in ("ready", "turn", "approach") else [],
        "nearest_visible_unblocked_tree_distance": (
            1 if category in ("ready", "turn") else 2 if category == "approach" else None),
        "sleep_override_active": False, "nearby_hostile_count": 0,
    }


def _episode(categories, actions, terminal="external_truncation", seed=0):
    wood, events, first_wood_steps = 0, [], {}
    for index, (category, action) in enumerate(zip(categories, actions)):
        before = _snapshot(category, wood)
        if events:
            events[-1]["after"] = copy.deepcopy(before)
        gain = int(category == "ready" and action == 5)
        wood += gain
        if gain:
            first_wood_steps[str(wood)] = index + 1
        probabilities = [0.0] * 17
        probabilities[action] = 1.0
        done = index == len(actions) - 1 and terminal == "death"
        events.append({
            "step": index + 1, "action": action, "probabilities": probabilities,
            "wood_before": before["wood"], "wood_after": wood, "wood_gain": gain,
            "before": before, "after": None if done else _snapshot("other", wood),
            "rgb_before_digest": str(index), "rgb_after_digest": str(index + 1),
            "health_after": 0 if done else 9, "health_delta": -9 if done else 0,
            "environment_done": done, "terminal_reason": "death" if done else None,
            "position_after": [32, 32], "facing_after": [1, 0],
        })
    return {
        "policy_seed": seed, "variant": VARIANTS[0], "episode": 0,
        "environment_seed": 100 + seed * 1000, "action_seed": 200 + seed * 1000,
        "steps": len(events), "initial_snapshot": copy.deepcopy(events[0]["before"]),
        "initial_rgb_digest": "0", "final_wood": wood,
        "first_wood_steps": first_wood_steps, "success": wood >= 3,
        "terminal_reason": terminal, "events": events,
    }


def _config(episode):
    return {
        "seed_set": [0, 1, 2], "wood_threshold": 3,
        "fresh_max_steps": episode["steps"], "fresh_episode_count": 1,
        "fresh_environment_seed_base": 100, "fresh_action_seed_base": 200,
        "fresh_replicate_seed_stride": 1000, "action_allowlist": list(range(7)),
    }


def test_local_category_is_exclusive_and_approach_requires_awake_state():
    assert local_category(_snapshot("ready")) == "ready"
    assert local_category(_snapshot("turn")) == "turn"
    approach = _snapshot("approach")
    assert local_category(approach) == "approach"
    approach["sleep_override_active"] = True
    assert local_category(approach) == "other"


def test_turn_ready_collection_chain_and_current_action_are_distinct():
    episode = _episode(["turn", "ready", "approach"], [2, 5, 2])
    rows = _event_rows([episode], 0, VARIANTS[0], 8)
    turn, ready, approach = rows
    assert turn["selected_local_action"] and turn["turn_became_ready"]
    assert turn["wood_gain"] == 0 and turn["next_ready_do"]
    assert turn["followup_wood_gain"]
    assert ready["wood_gain"] == 1 and not ready["followup_wood_gain"]
    assert approach["wood_stage"] == 1
    assert all(not row["followup_complete"] for row in rows)
    result = aggregate(rows)
    assert result["followup_wood_gain_observed"] == 1
    assert result["followup_wood_gain_rate_lower_bound"] == pytest.approx(1 / 3)
    assert result["followup_wood_gain_rate_upper_bound"] == 1.0
    assert result["followup_wood_gain_rate_complete"] is None


def test_truncated_horizon_boundary_does_not_turn_unobserved_future_into_failure():
    episode = _episode(["other"] * 10, [0] * 10)
    rows = _event_rows([episode], 0, VARIANTS[0], 8)
    assert [row["followup_complete"] for row in rows] == [True, True] + [False] * 8
    result = aggregate(rows)
    assert result["followup_windows_eligible"] == 10
    assert result["followup_windows_complete"] == 2
    assert result["followup_windows_censored"] == 8
    assert result["followup_wood_gain_rate_upper_bound"] == 0.8


def test_native_death_closes_earlier_windows_and_excludes_dead_start():
    episode = _episode(["approach", "other", "other"], [2, 0, 0], "death")
    rows = _event_rows([episode], 0, VARIANTS[0], 8)
    assert [row["followup_complete"] for row in rows] == [True, True, False]
    assert [row["followup_death"] for row in rows] == [True, True, False]
    assert not rows[-1]["alive_after"] and not rows[-1]["followup_eligible"]
    result = aggregate(rows)
    assert result["terminal_events"] == 1
    assert result["followup_windows_eligible"] == result["followup_windows_complete"] == 2
    assert result["followup_death_rate_complete"] == 1.0
    assert result["followup_windows_censored"] == 0


def test_task_success_closes_earlier_windows_and_does_not_count_current_gain_twice():
    episode = _episode(["ready"] * 3, [5] * 3, "success")
    _validate_episode(episode, seed=0, variant=VARIANTS[0], index=0, config=_config(episode))
    rows = _event_rows([episode], 0, VARIANTS[0], 8)
    result = aggregate(rows)
    assert result["wood_gained"] == 3
    assert result["followup_windows_eligible"] == result["followup_wood_gain"] == 2
    assert result["followup_task_success"] == 2
    assert result["terminal_events"] == 0
    assert not rows[-1]["followup_eligible"] and not rows[-1]["followup_wood_gain"]


def test_pooled_episode_count_includes_policy_seed():
    first = _episode(["other"], [0], seed=0)
    second = _episode(["other"], [0], seed=1)
    rows = _event_rows([first], 0, VARIANTS[0], 8) + _event_rows([second], 1, VARIANTS[0], 8)
    assert aggregate(rows)["episodes"] == 2


def test_ready_blocked_moves_are_separate_from_collection_and_losing_readiness():
    episode = _episode(["ready", "ready", "ready"], [2, 2, 5])
    rows = _event_rows([episode], 0, VARIANTS[0], 8)
    stalled = aggregate(rows[:2])
    assert stalled["blocked_moves"] == stalled["awake_move_actions"] == 2
    assert stalled["after_ready"] == 2 and stalled["wood_gained"] == 0
    assert stalled["next_ready_action_observed"] == 2 and stalled["next_ready_do"] == 1
    collection = aggregate(rows[-1:])
    assert collection["blocked_moves"] == 0 and collection["wood_gained"] == 1


@pytest.mark.parametrize("mutation, error", [
    ("rgb", "discontinuous"), ("wood", "collection rule"),
    ("health", "health delta"), ("probability", "allowlist"),
    ("milestone", "milestone"), ("snapshot", "snapshot boundary"),
])
def test_transition_verification_rejects_corrupted_records(mutation, error):
    episode = _episode(["ready", "approach"], [5, 2])
    config = _config(episode)
    _validate_episode(episode, seed=0, variant=VARIANTS[0], index=0, config=config)
    if mutation == "rgb":
        episode["events"][1]["rgb_before_digest"] = "corrupt"
    elif mutation == "wood":
        episode["events"][0]["wood_gain"] = 0
    elif mutation == "health":
        episode["events"][1]["health_delta"] = -1
    elif mutation == "probability":
        episode["events"][1]["probabilities"][2] = 0
        episode["events"][1]["probabilities"][10] = 1
    elif mutation == "milestone":
        episode["first_wood_steps"]["1"] = 2
    else:
        episode["events"][1]["after"] = None
    with pytest.raises(ValueError, match=error):
        _validate_episode(episode, seed=0, variant=VARIANTS[0], index=0, config=config)


def test_loader_rejects_trace_tampering_before_summarizing(tmp_path):
    episode = _episode(["ready", "approach"], [5, 2])
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(episode) + "\n")
    run = {"trace": str(path), "trace_canonical_digest": _digest([episode]), "trace_sha256": _sha256(path)}
    episode["events"][0]["action"] = 0
    path.write_text(json.dumps(episode) + "\n")
    with pytest.raises(RuntimeError, match="trace digest differs"):
        _load_trace(tmp_path, run, seed=0, variant=VARIANTS[0], config=_config(episode))
