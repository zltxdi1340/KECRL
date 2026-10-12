import copy
import json
from pathlib import Path

import pytest
import torch

pytest.importorskip("crafter")

from experiments.analyze_crafter_wood3_turn_ready_do_intervention import _pairs, _turn_outcomes, _validate_gates
from experiments.crafter_controlled_collection_scene import CollectionScene, QuietCollectionEnv
from experiments.run_crafter_wood3_actor_natural_migration import _trajectory_payload
from experiments.run_crafter_wood3_turn_ready_do_intervention import _choose_action, _episode, _validate
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


def _fixture(monkeypatch, scene, mutate=None):
    class Fixture(QuietCollectionEnv):
        def reset(self):
            observation = super().reset()
            if mutate is not None:
                mutate(self)
                observation = self._obs()
            return observation

    def factory(**kwargs):
        native = Fixture(scene, seed=kwargs["seed"], length=kwargs["length"])
        return CrafterEnvironmentAdapter(environment=native, diagnostics=kwargs["diagnostics"])

    import experiments.run_crafter_wood3_turn_ready_do_intervention as runner
    import experiments.run_crafter_wood3_ready_do_intervention as reference
    monkeypatch.setattr(runner, "CrafterEnvironmentAdapter", factory)
    monkeypatch.setattr(reference, "CrafterEnvironmentAdapter", factory)
    return {"max_steps": 8, "environment_length": 10000, "initial_wood": 0,
            "task": {"item": "wood", "threshold": 3}, "action_allowlist": list(range(7)), "hostile_radius": 3}


def test_ready_control_matches_prior_runner_and_rng(monkeypatch):
    from experiments.run_crafter_wood3_ready_do_intervention import _episode as previous
    config = _fixture(monkeypatch, CollectionScene(0, 1, 1, True))
    control = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "ready_do")
    rng = torch.get_rng_state().clone()
    reference = previous(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "ready_do")
    assert torch.equal(rng, torch.get_rng_state())
    assert _trajectory_payload(control) == _trajectory_payload(reference)
    assert control["turn_gate_steps"] == 0
    _validate_gates(control, "ready_do", list(range(7)))


@pytest.mark.parametrize("tree_direction", [1, 2, 3, 4])
def test_turn_then_do_uses_native_stationary_turn_and_one_sample_per_step(monkeypatch, tree_direction):
    facing = 4 if tree_direction != 4 else 1
    config = _fixture(monkeypatch, CollectionScene(0, tree_direction, facing, True))
    policy = _RGBPolicy(0)
    ep = _episode(policy, config, 17, 139, torch.device("cpu"), "turn_ready_do")
    assert policy.calls == ep["steps"] == 8
    first, second = ep["events"][:2]
    assert first["sampled_action"] == second["sampled_action"] == 0
    assert first["action"] == tree_direction and first["turn_gate_triggered"]
    assert first["position_after"] == first["before"]["position"]
    assert first["wood_gain"] == 0 and first["after"]["ready_to_collect"]
    assert second["action"] == 5 and second["wood_gain"] == 1 and second["ready_gate_triggered"]
    assert ep["turn_gate_steps"] == ep["turn_replaced_action_steps"] == 1
    assert ep["ready_gate_steps"] == ep["ready_replaced_action_steps"] == 1
    _validate_gates(ep, "turn_ready_do", list(range(7)))


def test_valid_sample_is_preserved_even_if_another_valid_direction_has_more_probability():
    before = {"ready_to_collect": False, "awake_adjacent_opportunity": True, "unblocked_tree_move_actions": [1, 3]}
    probabilities = [0.0, 0.1, 0.0, 0.8, 0.0, 0.1, 0.0]
    assert _choose_action(1, probabilities, before, "turn_ready_do") == (1, "turn_to_tree")
    assert _choose_action(0, probabilities, before, "turn_ready_do") == (3, "turn_to_tree")
    probabilities[1] = probabilities[3]
    assert _choose_action(0, probabilities, before, "turn_ready_do") == (1, "turn_to_tree")
    assert _choose_action(0, probabilities, before, "ready_do") == (0, None)


def test_turn_gate_does_not_replace_an_already_valid_sample(monkeypatch):
    config = _fixture(monkeypatch, CollectionScene(0, 1, 4, True))
    ep = _episode(_RGBPolicy(1), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    assert ep["turn_gate_steps"] == 1 and ep["turn_replaced_action_steps"] == 0
    assert ep["events"][0]["sampled_action"] == ep["events"][0]["action"] == 1
    assert ep["events"][1]["wood_gain"] == 1


def test_ready_has_priority_over_turn_and_preserves_sampled_do(monkeypatch):
    config = _fixture(monkeypatch, CollectionScene(0, 1, 1, True))
    ep = _episode(_RGBPolicy(5), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    assert ep["ready_gate_steps"] == 1 and ep["turn_gate_steps"] == 0
    assert ep["replaced_action_steps"] == 0


@pytest.mark.parametrize("label", ["no_tree", "object_blocked", "sleep_override", "capacity", "approach"])
def test_gates_leave_noneligible_states_unchanged(label):
    before = {"ready_to_collect": False, "awake_adjacent_opportunity": False,
              "unblocked_tree_move_actions": [] if label in ("no_tree", "object_blocked", "approach") else [1],
              "safe_approach_move_actions": [1] if label == "approach" else []}
    assert _choose_action(2, [1 / 7] * 7, before, "turn_ready_do") == (2, None)


def test_real_distant_tree_does_not_trigger_approach_override(monkeypatch):
    def distant(env):
        x, y = env._player.pos
        env._world[(x - 1, y)] = "grass"
        env._world[(x - 2, y)] = "tree"
    config = _fixture(monkeypatch, CollectionScene(0, 1, 4, True), distant)
    ep = _episode(_RGBPolicy(2), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    first = ep["events"][0]
    assert first["before"]["safe_approach_move_actions"] == [1]
    assert first["action"] == first["sampled_action"] == 2 and not first["oracle_gate_triggered"]
    _validate_gates(ep, "turn_ready_do", list(range(7)))


def test_real_sleep_override_prevents_turn_gate(monkeypatch):
    def asleep(env):
        env._player.sleeping = True
        env._player.inventory["energy"] = 0
    config = _fixture(monkeypatch, CollectionScene(0, 1, 4, True), asleep)
    ep = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    first = ep["events"][0]
    assert first["before"]["unblocked_adjacent_tree"] and first["before"]["sleep_override_active"]
    assert not first["oracle_gate_triggered"] and first["action"] == first["sampled_action"] == 0


def test_pair_prefix_ignores_shared_ready_replacement_then_detects_incremental_turn(monkeypatch):
    def second_tree(env):
        x, y = env._player.pos
        env._world[(x + 1, y)] = "tree"
    config = _fixture(monkeypatch, CollectionScene(0, 1, 1, True), second_tree)
    control = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "ready_do")
    turned = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    control["episode"] = turned["episode"] = 0
    assert control["events"][0]["action_replaced"] and turned["events"][0]["action_replaced"]
    row = _pairs([control], [turned], 0, "matched_baseline_only")[0]
    assert row["first_incremental_turn_step"] == 2
    assert row["prefix_before_incremental_turn_identical"] and row["paired_sampling_rng_identical"]
    corrupted = copy.deepcopy(turned)
    corrupted["events"][0]["reward"] += 1
    with pytest.raises(ValueError, match="equal executed actions"):
        _pairs([control], [corrupted], 0, "matched_baseline_only")
    corrupted = copy.deepcopy(turned)
    corrupted["events"][-1]["policy_rng_after_sample_digest"] = "changed"
    with pytest.raises(ValueError, match="RNG differs"):
        _pairs([control], [corrupted], 0, "matched_baseline_only")


def test_turn_continuation_distinguishes_last_step_truncation(monkeypatch):
    config = _fixture(monkeypatch, CollectionScene(0, 1, 4, True))
    ep = _episode(_RGBPolicy(0), {**config, "max_steps": 1}, 17, 139, torch.device("cpu"), "turn_ready_do")
    row = _turn_outcomes([ep])
    assert row["executed_valid_turns"] == row["effective_turn_after_ready"] == 1
    assert row["next_event_missing_external_truncation"] == 1 and row["next_event_observed"] == 0
    ep = _episode(_RGBPolicy(0), {**config, "max_steps": 2}, 17, 139, torch.device("cpu"), "turn_ready_do")
    row = _turn_outcomes([ep])
    assert row["next_event_observed"] == row["next_action_do"] == row["next_action_wood_gain"] == 1
    assert row["next_event_missing_external_truncation"] == row["effective_turn_native_terminal"] == 0


def test_independent_gate_validation_rejects_changed_turn_and_counts(monkeypatch):
    config = _fixture(monkeypatch, CollectionScene(0, 1, 4, True))
    ep = _episode(_RGBPolicy(0), config, 17, 139, torch.device("cpu"), "turn_ready_do")
    _validate_gates(ep, "turn_ready_do", list(range(7)))
    corrupted = copy.deepcopy(ep)
    corrupted["events"][0]["turn_gate_triggered"] = False
    with pytest.raises(ValueError, match="boundary differs"):
        _validate_gates(corrupted, "turn_ready_do", list(range(7)))
    corrupted = copy.deepcopy(ep)
    corrupted["turn_replaced_action_steps"] += 1
    with pytest.raises(ValueError, match="counts differ"):
        _validate_gates(corrupted, "turn_ready_do", list(range(7)))


@pytest.mark.parametrize("field,value", [("teacher_used", False), ("policy_updates_allowed", True),
                                          ("conditions", ["native", "turn_ready_do"]),
                                          ("approach_intervention_allowed", True), ("survival_intervention_allowed", True),
                                          ("turn_selection_rule", "lowest_action_id")])
def test_locked_turn_protocol_rejects_scope_changes(monkeypatch, field, value):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_turn_ready_do_intervention_cuda_v1.yaml").read_text())
    _validate(config)
    with pytest.raises(ValueError):
        _validate({**config, field: value})
