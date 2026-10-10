import copy

import pytest

pytest.importorskip("crafter")
pytest.importorskip("torch")

from experiments.analyze_crafter_wood3_natural_state_window import aggregate_windows, recompute_window


def _window():
    record = {"rgb_sha256": "frame0", "designated_position": [5, 4], "target_actions": [5],
              "snapshot": {"position": [4, 4], "facing": [1, 0], "wood": 1}}
    events = []
    for step in range(1, 9):
        collect = step == 1
        probabilities = [0.0] * 7
        probabilities[5 if collect else 0] = 1.0
        events.append({"step": step, "action": 5 if collect else 0, "probabilities": probabilities,
                       "rgb_before": f"frame{step - 1}", "rgb_after": f"frame{step}",
                       "wood_before": 1 if collect else 2, "wood_after": 2, "wood_gain": int(collect),
                       "before": {"wood": 1 if collect else 2, "health": 9, "position": [4, 4],
                                  "facing": [1, 0], "ready_to_collect": collect},
                       "position_after": [4, 4], "health_after": 9,
                       "designated_tree_gain": collect, "environment_done": False})
    return {"steps": 8, "initial_rgb_sha256": "frame0", "mode": "greedy", "events": events}, record


def test_unrelated_wood_does_not_count_as_designated_tree_success():
    row, record = _window()
    record["designated_position"] = [4, 5]
    row["events"][0]["designated_tree_gain"] = False
    metrics = recompute_window(row, record)
    aggregate = aggregate_windows([metrics])
    assert aggregate["any_wood_gain_windows"] == 1
    assert aggregate["other_tree_wood_gain"] == 1
    assert aggregate["designated_successes"] == 0


@pytest.mark.parametrize("mutation", ["collection_label", "wood_arithmetic", "early_stop"])
def test_analysis_rejects_corrupt_gain_or_premature_window(mutation):
    row, record = _window()
    row = copy.deepcopy(row)
    if mutation == "collection_label":
        row["events"][0]["designated_tree_gain"] = False
    elif mutation == "wood_arithmetic":
        row["events"][0]["wood_after"] = 3
    else:
        row["events"] = row["events"][:1]
        row["steps"] = 1
    with pytest.raises(ValueError):
        recompute_window(row, record)
