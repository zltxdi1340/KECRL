import json
from pathlib import Path

import pytest

pytest.importorskip("crafter")

from experiments.run_crafter_wood3_actor_natural_migration import (
    _event_context, _trajectory_payload, _validate,
)


def test_natural_migration_protocol_is_frozen_and_non_formal(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_actor_natural_migration_cuda_v1.yaml").read_text())
    head_config = {"formal_result": False, "teacher_used": True,
                   "baseline_result_root": config["source_result_root"]}
    head_summary = {"cross_process_repetition": {"passed": True}}
    source_config = json.loads(Path(
        "configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml"
    ).read_text())
    _validate(config, head_config, head_summary, source_config)
    for key, value in (("policy_updates_allowed", True), ("new_fit_or_selection_allowed", True),
                       ("formal_result", True), ("max_steps", 128)):
        with pytest.raises(ValueError):
            _validate({**config, key: value}, head_config, head_summary, source_config)


def test_natural_migration_event_context_and_replay_payload():
    before = {"sleep_override_active": False, "target_object": None,
              "unblocked_adjacent_tree": True}
    assert _event_context(before, 5, 0) == "adjacent_tree_wrong_facing"
    assert _event_context({**before, "sleep_override_active": True}, 5, 0) == "sleep_override"
    assert _event_context({**before, "target_object": "Cow"}, 5, 0) == "target_object_blocked"
    assert _event_context(before, 5, 1) is None
    episode = {
        "environment_seed": 1, "action_seed": 2, "success": False, "steps": 0,
        "final_wood": 0, "first_wood_steps": {}, "action_counts": {},
        "mean_action_entropy": 0.0, "initial_rgb_digest": "rgb", "events": [],
        "policy_seed": 0, "variant": "baseline", "episode": 0,
    }
    assert _trajectory_payload(episode)["events"] == []
    assert "variant" not in _trajectory_payload(episode)
