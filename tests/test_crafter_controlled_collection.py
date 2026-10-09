import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("crafter")

from experiments.crafter_controlled_collection_scene import (
    CollectionScene, DIRECTIONS, QuietCollectionEnv, scene_manifest,
)
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_controlled_collection import (
    _episode, _paired_responses, _response_summary, _summarize, _validate,
    uniform_success_probability,
)
from experiments.crafter_local_action_reference import constant_action_success_probability
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot


def _adapter(scene):
    return CrafterEnvironmentAdapter(environment=QuietCollectionEnv(scene, seed=17), diagnostics=True)


def test_every_geometry_and_wood_stage_uses_native_one_or_two_step_collection():
    scenes = scene_manifest()
    assert len(scenes) == 96
    assert len({scene.pair_id for scene in scenes}) == 48
    adapter = _adapter(scenes[0])
    try:
        for scene in scenes:
            if not scene.tree_present:
                continue
            adapter.environment.configure(scene, seed=17)
            observation = adapter.reset()
            assert observation.shape == (64, 64, 3) and observation.dtype == np.uint8
            before = collection_snapshot(adapter)
            position = before["position"]
            assert before["wood"] == scene.initial_wood
            assert before["unblocked_tree_move_actions"] == [scene.tree_action]
            assert before["ready_to_collect"] == (scene.alignment == "aligned")
            if scene.alignment == "needs_turn":
                _, _, _, info = adapter.step(5)
                assert info["inventory"]["wood"] == scene.initial_wood
                adapter.step(scene.tree_action)
                assert collection_snapshot(adapter)["position"] == position
                assert collection_snapshot(adapter)["ready_to_collect"]
            _, reward, done, info = adapter.step(5)
            assert not done
            assert reward == (1 if scene.initial_wood == 0 else 0)
            assert info["inventory"]["wood"] == scene.initial_wood + 1
            assert adapter.environment._world.count("tree") == 0
    finally:
        adapter.close()


def test_tree_removal_pair_changes_only_one_cell_and_its_rgb_patch():
    scene = CollectionScene(2, 3, 1, True)
    adapter = _adapter(scene)
    try:
        present_rgb = adapter.reset().copy()
        world = adapter.environment._world
        present_map = world._mat_map.copy()
        rng = copy.deepcopy(world.random.get_state())
        inventory = copy.deepcopy(adapter.environment._player.inventory)
        facing = tuple(adapter.environment._player.facing)
        adapter.environment.configure(replace(scene, tree_present=False), seed=17)
        absent_rgb = adapter.reset()
        assert np.count_nonzero(present_map != world._mat_map) == 1
        assert _state_equal(rng, world.random.get_state())
        assert adapter.environment._player.inventory == inventory
        assert tuple(adapter.environment._player.facing) == facing
        assert not np.array_equal(present_rgb, absent_rgb)
        changed = np.any(present_rgb != absent_rgb, axis=-1)
        assert changed.sum() <= 7 * 7
        assert not collection_snapshot(adapter)["ready_to_collect"]
        for action in (3, 5, 5, 5, 0, 6, 1, 5):
            _, _, done, info = adapter.step(action)
            assert not done and info["inventory"]["wood"] == 2
    finally:
        adapter.close()


def test_quiet_fixture_keeps_full_resources_and_has_no_spawns_at_native_balance_step():
    adapter = _adapter(CollectionScene(0, 1, 1, False))
    try:
        adapter.reset()
        for _ in range(10):
            adapter.step(6)
            snapshot = collection_snapshot(adapter)
            assert all(snapshot[key] == 9 for key in ("health", "food", "drink", "energy"))
            assert snapshot["daylight"] == 1 and not snapshot["sleeping"]
            assert len(adapter.environment._world.objects) == 1
    finally:
        adapter.close()


@pytest.mark.parametrize("values", [(3, 1, 1, True), (0, 0, 1, True), (0, 1, True, True), (0, 1, 1, 1)])
def test_scene_rejects_invalid_or_implicit_conditions(values):
    with pytest.raises(ValueError):
        CollectionScene(*values)


class _ConstantRGBPolicy:
    def __init__(self, action=None):
        self.action = action

    def distribution_value(self, image, hidden, allowlist):
        assert image.shape == (3, 64, 64) and hidden is None
        assert allowlist == list(range(7))
        logits = torch.zeros((1, 17), device=image.device)
        logits[:, 7:] = -torch.inf
        if self.action is not None:
            logits[:, :7] = -torch.inf
            logits[:, self.action] = 0
        return torch.distributions.Categorical(logits=logits), torch.zeros(1), None


def test_oracle_observer_preserves_sampled_trajectory_and_torch_rng():
    scene = CollectionScene(0, 1, 4, True)
    adapter = _adapter(scene)
    config = {"max_steps": 8, "action_allowlist": list(range(7))}
    try:
        policy = _ConstantRGBPolicy()
        control = _episode(adapter, policy, scene, config, 17, 129, "stochastic", "cpu", observe=False)
        rng = torch.get_rng_state().clone()
        observed = _episode(adapter, policy, scene, config, 17, 129, "stochastic", "cpu")
        assert torch.equal(rng, torch.get_rng_state())
        assert control["events"] == [{key: value for key, value in row.items() if key != "before"}
                                      for row in observed["events"]]
        assert control["success"] == observed["success"]
    finally:
        adapter.close()


def test_local_success_uses_increment_from_initial_stage_and_no_tree_is_negative_control():
    adapter = _adapter(CollectionScene(2, 1, 1, True))
    config = {"max_steps": 8, "action_allowlist": list(range(7))}
    try:
        rows = []
        for wood in (0, 1, 2):
            for present in (True, False):
                scene = CollectionScene(wood, 1, 1, present)
                row = _episode(adapter, _ConstantRGBPolicy(5), scene, config, 17, 129, "greedy", "cpu")
                assert row["success"] is present
                assert row["steps"] == (1 if present else 8)
                rows.append(row)
        summary = _summarize(rows)
        assert summary["by_mode"]["greedy"]["aligned"]["successes"] == 3
        assert summary["by_mode"]["greedy"]["no_tree"]["successes"] == 0
        assert summary["by_mode"]["greedy"]["aligned"]["ready_do_rate"] == 1
    finally:
        adapter.close()


def test_tree_presence_probability_contrasts_are_paired_not_marginal_action_frequencies():
    scene = CollectionScene(0, 3, 1, True)
    present = [0, .1, .1, .3, .1, .3, .1]
    absent = [0, .1, .1, .1, .1, .5, .1]
    responses = [{**scene.record(), "probabilities": present, "greedy_action": 3},
                 {**replace(scene, tree_present=False).record(), "probabilities": absent, "greedy_action": 5}]
    pairs = _paired_responses(responses)
    assert pairs[0]["delta_p_target_move"] == pytest.approx(.2)
    assert pairs[0]["delta_p_do"] == pytest.approx(-.2)
    summary = _response_summary(pairs)["needs_turn"]
    assert summary["greedy_optimal_count"] == 1
    assert summary["no_tree_greedy_action_counts"] == {"5": 1}


def test_exact_uniform_reference_matches_short_action_sequences_and_rotation_symmetry():
    assert uniform_success_probability(CollectionScene(0, 1, 1, True), 1) == pytest.approx(1 / 7)
    assert uniform_success_probability(CollectionScene(0, 1, 1, True), 2) == pytest.approx(10 / 49)
    assert uniform_success_probability(CollectionScene(0, 1, 4, True), 2) == pytest.approx(1 / 49)
    assert uniform_success_probability(CollectionScene(0, 1, 1, False), 8) == 0
    aligned, turn = [], []
    for tree in DIRECTIONS:
        for facing in DIRECTIONS:
            (aligned if tree == facing else turn).append(
                uniform_success_probability(CollectionScene(0, tree, facing, True), 8))
    assert max(aligned) - min(aligned) < 1e-12
    assert max(turn) - min(turn) < 1e-12


def test_protocol_rejects_training_unbalanced_scenes_and_previous_seed_reuse(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = json.loads(Path("configs/crafter_wood3_controlled_collection_cuda_v1.yaml").read_text())
    arms = {"baseline": json.loads(Path("configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml").read_text()),
            "auxiliary": json.loads(Path("configs/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_v1.yaml").read_text())}
    previous = [json.loads(Path(path).read_text()) for path in config["previous_diagnostic_configs"]]
    _validate(config, arms, previous)
    for field, value in (("policy_updates_allowed", True), ("initial_facing_actions", [1]),
                         ("max_steps", 16), ("diagnostic_action_seed_base", previous[-1]["diagnostic_action_seed_base"])):
        invalid = {**config, field: value}
        with pytest.raises(ValueError):
            _validate(invalid, arms, previous)


def test_constant_action_reference_matches_weighted_native_two_step_sequences():
    from itertools import product

    probabilities = [.03, .08, .13, .17, .21, .31, .07]
    adapter = _adapter(CollectionScene(2, 1, 1, True))
    try:
        for tree_action in DIRECTIONS:
            for facing_action in (tree_action, next(action for action in DIRECTIONS if action != tree_action)):
                scene = CollectionScene(2, tree_action, facing_action, True)
                probability = 0.0
                for left, right in product(range(7), repeat=2):
                    adapter.environment.configure(scene, seed=17)
                    adapter.reset()
                    adapter.step(left)
                    _, _, _, info = adapter.step(right)
                    if info['inventory']['wood'] == 3:
                        probability += probabilities[left] * probabilities[right]
                assert constant_action_success_probability(tree_action, facing_action, probabilities, 2) == pytest.approx(probability)
    finally:
        adapter.close()


def test_constant_action_reference_matches_eight_step_uniform_reference():
    for scene in scene_manifest():
        if scene.initial_wood != 0:
            continue
        value = constant_action_success_probability(scene.tree_action, scene.facing_action, [1 / 7] * 7, tree_present=scene.tree_present)
        assert value == pytest.approx(uniform_success_probability(scene, 8), abs=1e-12)
    with pytest.raises(ValueError, match='sum to one'):
        constant_action_success_probability(1, 1, [.1] * 7)
