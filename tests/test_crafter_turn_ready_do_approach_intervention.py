import copy
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("crafter")

from experiments.analyze_crafter_wood3_turn_ready_do_approach_intervention import (
    _approach_outcomes, _load, _pairs, _validate_gates,
)
from experiments.crafter_controlled_collection_scene import CollectionScene, DIRECTIONS, QuietCollectionEnv
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest
from experiments.run_crafter_wood3_actor_natural_migration import _trajectory_payload
from experiments.run_crafter_wood3_collection_opportunity import _sha256
from experiments.run_crafter_wood3_turn_ready_do_approach_intervention import (
    _choose_action, _episode, _validate,
)
from src.environments.crafter_adapter import CrafterEnvironmentAdapter


class _RGBPolicy:
    def __init__(self, action):
        self.action, self.calls = action, 0

    def distribution_value(self, image, hidden, allowlist):
        assert image.shape == (3, 64, 64) and hidden is None and allowlist == list(range(7))
        self.calls += 1
        logits = torch.full((1, 17), -torch.inf, device=image.device)
        logits[:, self.action] = 0
        return torch.distributions.Categorical(logits=logits), torch.zeros(1), None


def _fixture(monkeypatch, *, direction=1, distance=2, mutate=None):
    scene = CollectionScene(0, direction, 4 if direction != 4 else 1, False)

    class Fixture(QuietCollectionEnv):
        def reset(self):
            super().reset()
            x, y = self._player.pos
            dx, dy = DIRECTIONS[direction]
            self._world[(x + distance * dx, y + distance * dy)] = "tree"
            if mutate is not None:
                mutate(self)
            return self._obs()

    def factory(**kwargs):
        native = Fixture(scene, seed=kwargs["seed"], length=kwargs["length"])
        return CrafterEnvironmentAdapter(environment=native, diagnostics=kwargs["diagnostics"])

    import experiments.run_crafter_wood3_turn_ready_do_approach_intervention as runner
    import experiments.run_crafter_wood3_turn_ready_do_intervention as reference
    monkeypatch.setattr(runner, "CrafterEnvironmentAdapter", factory)
    monkeypatch.setattr(reference, "CrafterEnvironmentAdapter", factory)
    return {"max_steps": 8, "environment_length": 10000, "initial_wood": 0,
            "task": {"item": "wood", "threshold": 3},
            "action_allowlist": list(range(7)), "hostile_radius": 3}


@pytest.mark.parametrize("direction", [1, 2, 3, 4])
def test_approach_uses_native_move_then_ready_do_and_one_draw_per_step(monkeypatch, direction):
    config = _fixture(monkeypatch, direction=direction)
    policy = _RGBPolicy(0)
    ep = _episode(policy, config, 17, 139, torch.device("cpu"), "turn_ready_do_approach")
    first, second = ep["events"][:2]
    assert policy.calls == ep["steps"]
    assert first["sampled_action"] == second["sampled_action"] == 0
    assert first["action"] == direction and first["approach_gate_triggered"]
    assert first["position_after"] != first["before"]["position"]
    assert first["after"]["nearest_visible_unblocked_tree_distance"] == 1
    assert first["after"]["ready_to_collect"] and first["wood_gain"] == 0
    assert second["action"] == 5 and second["wood_gain"] == 1
    assert ep["approach_gate_steps"] == ep["approach_replaced_action_steps"] == 1
    _validate_gates(ep, "turn_ready_do_approach", list(range(7)))


def test_control_matches_prior_turn_runner_and_rng(monkeypatch):
    from experiments.run_crafter_wood3_turn_ready_do_intervention import _episode as reference
    config = _fixture(monkeypatch, distance=1)
    control = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    rng = torch.get_rng_state().clone()
    prior = reference(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    assert _trajectory_payload(control) == _trajectory_payload(prior)
    assert torch.equal(rng, torch.get_rng_state())
    assert control["approach_gate_steps"] == control["approach_replaced_action_steps"] == 0
    assert control["turn_gate_steps"] == 1 and control["ready_gate_steps"] == 1
    _validate_gates(control, "turn_ready_do", list(range(7)))


def test_control_preserves_original_distant_tree_action(monkeypatch):
    config = _fixture(monkeypatch)
    ep = _episode(_RGBPolicy(2), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    assert ep["events"][0]["before"]["safe_approach_move_actions"] == [1]
    assert ep["events"][0]["action"] == 2
    assert ep["approach_gate_steps"] == 0


def test_selection_preserves_valid_sample_and_breaks_probability_ties_without_rng():
    before = {"ready_to_collect": False, "awake_adjacent_opportunity": False,
              "safe_approach_move_actions": [3, 1], "sleep_override_active": False}
    probabilities = [0, .1, 0, .8, 0, .1, 0]
    rng = torch.get_rng_state().clone()
    assert _choose_action(1, probabilities, before, "turn_ready_do_approach") == (1, "approach_to_tree")
    assert _choose_action(0, probabilities, before, "turn_ready_do_approach") == (3, "approach_to_tree")
    probabilities[1] = probabilities[3]
    assert _choose_action(0, probabilities, before, "turn_ready_do_approach") == (1, "approach_to_tree")
    assert torch.equal(rng, torch.get_rng_state())


def test_approach_gate_preserves_already_valid_sample(monkeypatch):
    ep = _episode(_RGBPolicy(1), _fixture(monkeypatch), 17, 139, torch.device("cpu"), "turn_ready_do_approach")
    assert ep["approach_gate_steps"] == 1 and ep["approach_replaced_action_steps"] == 0
    assert ep["events"][0]["action"] == ep["events"][0]["sampled_action"] == 1


@pytest.mark.parametrize("ready", [True, False])
def test_shared_ready_and_turn_take_priority_over_approach(ready):
    before = {"ready_to_collect": ready, "awake_adjacent_opportunity": True,
              "unblocked_tree_move_actions": [3], "safe_approach_move_actions": [1]}
    expected = (5, "ready_do") if ready else (3, "turn_to_tree")
    for condition in ("turn_ready_do", "turn_ready_do_approach"):
        assert _choose_action(0, [1 / 7] * 7, before, condition) == expected


def test_sleep_override_prevents_approach_even_with_geometric_action(monkeypatch):
    def asleep(env):
        env._player.sleeping = True
        env._player.inventory["energy"] = 0
    ep = _episode(_RGBPolicy(0), _fixture(monkeypatch, mutate=asleep), 17, 139,
                  torch.device("cpu"), "turn_ready_do_approach")
    first = ep["events"][0]
    assert first["before"]["sleep_override_active"]
    assert first["before"]["safe_approach_move_actions"] == [1]
    assert not first["oracle_gate_triggered"] and first["action"] == first["sampled_action"] == 0
    _validate_gates(ep, "turn_ready_do_approach", list(range(7)))


def test_local_obstacle_and_no_tree_do_not_trigger_exploration_override(monkeypatch):
    def blocked(env):
        x, y = env._player.pos
        env._world[(x - 1, y)] = "stone"
    config = _fixture(monkeypatch, mutate=blocked)
    ep = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do_approach")
    assert ep["events"][0]["before"]["nearest_visible_unblocked_tree_distance"] == 2
    assert ep["events"][0]["before"]["safe_approach_move_actions"] == []
    assert ep["approach_gate_steps"] == 0
    def no_tree(env):
        x, y = env._player.pos
        env._world[(x - 2, y)] = "grass"
    ep = _episode(_RGBPolicy(0), _fixture(monkeypatch, mutate=no_tree), 17, 139,
                  torch.device("cpu"), "turn_ready_do_approach")
    assert ep["oracle_gate_steps"] == 0


def test_pair_prefix_retains_shared_turn_and_ready_before_approach(monkeypatch):
    def another_tree(env):
        x, y = env._player.pos
        env._world[(x + 2, y)] = "tree"
    config = _fixture(monkeypatch, distance=1, mutate=another_tree)
    left = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    right = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do_approach")
    left["episode"] = right["episode"] = 0
    assert left["events"][0]["turn_gate_triggered"] and right["events"][0]["turn_gate_triggered"]
    assert left["events"][1]["ready_gate_triggered"] and right["events"][1]["ready_gate_triggered"]
    row = _pairs([left], [right], 0, "matched_baseline_only")[0]
    assert row["first_incremental_approach_step"] == 3
    assert row["prefix_before_incremental_approach_identical"] and row["paired_sampling_rng_identical"]
    corrupted = copy.deepcopy(right)
    corrupted["events"][0]["reward"] += 1
    with pytest.raises(ValueError, match="equal executed actions"):
        _pairs([left], [corrupted], 0, "matched_baseline_only")
    corrupted = copy.deepcopy(right)
    corrupted["events"][-1]["policy_rng_after_sample_digest"] = "changed"
    with pytest.raises(ValueError, match="RNG differs"):
        _pairs([left], [corrupted], 0, "matched_baseline_only")


def test_approach_followthrough_separates_censored_and_task_complete_windows(monkeypatch):
    config = _fixture(monkeypatch)
    ep = _episode(_RGBPolicy(0), {**config, "max_steps": 1}, 17, 139,
                  torch.device("cpu"), "turn_ready_do_approach")
    ep["episode"] = 0
    row = _approach_outcomes([ep])
    assert row["executed_valid_approach_steps"] == row["distance_reduced_steps"] == 1
    assert row["next_event_missing_external_truncation"] == 1
    assert row["followup8_effective_external_censored"] == 1
    assert row["followup8_effective_complete"] == row["next_event_observed_after_effective"] == 0
    ep = _episode(_RGBPolicy(0), {**config, "task": {"item": "wood", "threshold": 1}},
                  17, 139, torch.device("cpu"), "turn_ready_do_approach")
    ep["episode"] = 0
    row = _approach_outcomes([ep])
    assert row["followup8_effective_complete"] == row["followup8_effective_wood_gain"] == 1
    assert row["next_action_wood_gain_after_effective"] == 1
    assert row["effective_approach_native_terminal_steps"] == 0


def test_approach_death_denominator_only_counts_selected_actions(monkeypatch):
    config = _fixture(monkeypatch)
    ep = _episode(_RGBPolicy(0), {**config, "max_steps": 1}, 17, 139, torch.device("cpu"), "turn_ready_do")
    ep["episode"] = 0
    event = ep["events"][0]
    event.update(environment_done=True, terminal_reason="death", after=None, health_after=0, health_delta=-9)
    ep["terminal_reason"] = "death"
    row = _approach_outcomes([ep])
    assert row["opportunity_native_terminal_steps"] == 1
    assert row["effective_approach_native_terminal_steps"] == row["executed_valid_approach_steps"] == 0
    event["action"] = 1
    row = _approach_outcomes([ep])
    assert row["effective_approach_native_terminal_steps"] == row["next_event_missing_native_terminal"] == 1
    assert row["distance_observed_steps"] == row["followup8_effective_eligible"] == 0


def test_independent_validation_rejects_changed_approach_and_counts(monkeypatch):
    ep = _episode(_RGBPolicy(0), _fixture(monkeypatch), 17, 139, torch.device("cpu"), "turn_ready_do_approach")
    _validate_gates(ep, "turn_ready_do_approach", list(range(7)))
    corrupted = copy.deepcopy(ep)
    corrupted["events"][0]["approach_gate_triggered"] = False
    with pytest.raises(ValueError, match="boundary differs"):
        _validate_gates(corrupted, "turn_ready_do_approach", list(range(7)))
    corrupted = copy.deepcopy(ep)
    corrupted["approach_replaced_action_steps"] += 1
    with pytest.raises(ValueError, match="counts differ"):
        _validate_gates(corrupted, "turn_ready_do_approach", list(range(7)))


def test_trace_validation_rejects_changed_raw_hash(monkeypatch, tmp_path):
    config = _fixture(monkeypatch)
    ep = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do_approach")
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps(ep) + "\n")
    row = {"trace": str(path), "trace_sha256": _sha256(path), "trace_canonical_digest": _json_digest([ep])}
    path.write_text(json.dumps({**ep, "steps": 999}) + "\n")
    with pytest.raises(RuntimeError, match="trace hash differs"):
        _load(tmp_path, row, config, False)


@pytest.mark.parametrize("field,value", [
    ("teacher_used", False), ("policy_updates_allowed", True), ("encoder_updates_allowed", True),
    ("conditions", ["ready_do", "turn_ready_do_approach"]),
    ("approach_intervention_allowed", False), ("survival_intervention_allowed", True),
    ("approach_selection_rule", "lowest_action_id"),
])
def test_locked_approach_protocol_rejects_scope_changes(monkeypatch, field, value):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_turn_ready_do_approach_intervention_cuda_v1.yaml").read_text())
    _validate(config)
    with pytest.raises(ValueError):
        _validate({**config, field: value})
