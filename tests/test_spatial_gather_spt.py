from __future__ import annotations

import json

import pytest
import torch

from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.skills.spatial_gather_spt import (
    SpatialGatherFOMAML,
    SpatialGatherSPT,
    SpatialMetaTask,
    SpatialPPORollout,
    build_gather_spt,
    instantiate_gather_spi,
)
from src.skills.models import InMemoryQualifiedSkillLibrary, QualificationConfig


def _policy_config():
    return {
        "cnn_channels": [2, 2], "embedding_dim": 8, "action_count": 17,
        "learning_rate": 3e-4, "gamma": .99, "clip_epsilon": .2,
        "value_coef": .5, "entropy_coef": .01, "update_epochs": 1,
    }


def _rollout(seed=0):
    generator = torch.Generator().manual_seed(seed)
    return SpatialPPORollout(
        observations=torch.rand((1, 3, 64, 64), generator=generator),
        actions=torch.tensor([5]), old_log_probs=torch.tensor([-1.7]),
        returns=torch.tensor([1.0]), advantages=torch.tensor([1.0]),
    )


def test_material_bindings_create_versioned_spi_records_and_zero_residual_policies():
    torch.manual_seed(3)
    spt = build_gather_spt()
    active = SpatialGatherSPT(spt, build_matched_spatial_policy(_policy_config(), "cnn_only"))
    original = {name: value.detach().clone() for name, value in active.state_dict().items()}

    instances = [active.instantiate(instantiate_gather_spi(spt, material))
                 for material in ("tree", "stone", "coal")]
    assert [instance.spi.spi_id for instance in instances] == [
        "gather_wood_delta1", "gather_stone_delta1", "gather_coal_delta1"
    ]
    assert [instance.context.tolist() for instance in instances] == [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    ]
    assert instances[0].spi.transition_request.input_capabilities == ()
    assert instances[1].spi.transition_request.input_capabilities == (
        {"name": "inventory_at_least", "item": "wood_pickaxe", "threshold": 1},
    )
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig())
    assert all(library._compatible(instance.spi.transition_request,
                                   instance.spi.implementation_contract)
               for instance in instances)
    assert all(torch.equal(instances[0].policy.actor.weight, item.policy.actor.weight)
               for item in instances[1:])
    assert all(instance.policy.optimizer is not active.template.optimizer for instance in instances)
    assert all(not instance.policy.optimizer.state for instance in instances)
    assert all(torch.equal(original[name], value) for name, value in active.state_dict().items())


def test_initializer_uses_only_spi_binding_and_rejects_other_spt_versions():
    torch.manual_seed(5)
    spt = build_gather_spt()
    model = SpatialGatherSPT(spt, build_matched_spatial_policy(_policy_config(), "cnn_only"))
    with torch.no_grad():
        model.context_to_actor.weight.zero_()
        model.context_to_actor.weight[:, 1].fill_(0.01)
        model.context_to_actor.weight[:, 2].fill_(-0.02)
    wood = model.instantiate(instantiate_gather_spi(spt, "tree")).policy
    coal = model.instantiate(instantiate_gather_spi(spt, "coal")).policy
    assert not torch.equal(wood.actor.weight, coal.actor.weight)
    assert torch.equal(wood.encoder.net[0].weight, coal.encoder.net[0].weight)

    stale_spt = build_gather_spt()
    stale_spi = instantiate_gather_spi(stale_spt, "coal")
    object.__setattr__(stale_spi, "spt_version", "0.9.0")
    with pytest.raises(ValueError, match="different SPT"):
        model.instantiate(stale_spi)


def test_fomaml_updates_candidate_and_generator_without_touching_active_or_cold_column():
    torch.manual_seed(11)
    record = build_gather_spt()
    active = SpatialGatherSPT(record, build_matched_spatial_policy(_policy_config(), "cnn_only"))
    candidate = active.clone_candidate()
    active_before = {name: value.detach().clone() for name, value in active.state_dict().items()}
    candidate_before = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    cold_before = candidate.cold_context_column("coal")
    wood_spi = instantiate_gather_spi(record, "tree")

    probe = candidate.instantiate(wood_spi)
    assert not probe.policy.optimizer.state
    learner = SpatialGatherFOMAML(candidate, inner_lr=3e-4, meta_lr=1e-2)
    adapted = learner.adapt(wood_spi, _rollout(1))
    assert adapted.optimizer.state
    assert all(torch.equal(candidate_before[name], value) for name, value in candidate.state_dict().items())

    loss = learner.meta_update((SpatialMetaTask(wood_spi, _rollout(2), _rollout(3)),))
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(candidate_before["template.actor.weight"], candidate.template.actor.weight)
    assert not torch.equal(candidate_before["context_to_actor.weight"], candidate.context_to_actor.weight)
    assert torch.equal(cold_before, candidate.cold_context_column("coal"))
    assert all(torch.equal(active_before[name], value) for name, value in active.state_dict().items())


def test_ordinary_update_backpropagates_through_shared_generated_actor():
    torch.manual_seed(19)
    spt = build_gather_spt()
    candidate = SpatialGatherSPT(spt, build_matched_spatial_policy(_policy_config(), "cnn_only"))
    spi = instantiate_gather_spi(spt, "tree")
    learner = SpatialGatherFOMAML(candidate, inner_lr=3e-4, meta_lr=1e-2)
    before_template = candidate.template.actor.weight.detach().clone()
    before_generator = candidate.context_to_actor.weight.detach().clone()
    value = learner.ordinary_update((SpatialMetaTask(spi, _rollout(7), _rollout(8)),))
    assert torch.isfinite(torch.tensor(value))
    assert not torch.equal(before_template, candidate.template.actor.weight)
    assert not torch.equal(before_generator, candidate.context_to_actor.weight)


def test_rollout_requires_rgb_and_aligned_vectors():
    with pytest.raises(ValueError, match="observations"):
        SpatialPPORollout(torch.zeros(2, 3), torch.zeros(2, dtype=torch.long),
                          torch.zeros(2), torch.zeros(2), torch.zeros(2))


def test_protocol_scene_manifest_declares_real_spatial_generator():
    config = json.loads(open("configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml").read())
    assert config["config_kind"] == "protocol_manifest_not_runner_config"
    assert config["environment"]["fixture_version"] == "quiet-material-gather-v1"
    assert config["initialization"]["generated_parameter_names"] == ["actor.weight", "actor.bias"]


def test_material_scene_is_paired_and_uses_native_collection_requirements():
    pytest.importorskip("crafter")
    import numpy as np
    from experiments.crafter_material_gather_scene import (
        MATERIAL_TO_ITEM, MaterialGatherEnv, make_material_gather_scene,
    )

    scene = make_material_gather_scene(123)
    left = MaterialGatherEnv(scene, seed=17, length=32)
    right = MaterialGatherEnv(scene, seed=17, length=32)
    try:
        left_rgb = left.reset()
        right_rgb = right.reset()
        assert left_rgb.shape == (64, 64, 3) and left_rgb.dtype == np.uint8
        assert np.array_equal(left_rgb, right_rgb)
        assert np.array_equal(left._world._mat_map, right._world._mat_map)
        assert {name: int(left._world.count(name)) for name in MATERIAL_TO_ITEM} == {
            "tree": 1, "stone": 1, "coal": 1,
        }
        assert left._player.inventory["wood_pickaxe"] == 1
        for material, item in MATERIAL_TO_ITEM.items():
            target = scene.positions[material]
            assert sum(abs(target[index] - 32) for index in range(2)) in (1, 2, 3)
            material_positions = set(scene.positions.values())
            adjacent = next(
                (target[0] - dx, target[1] - dy, (dx, dy))
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1))
                if (target[0] - dx, target[1] - dy) not in material_positions
            )
            left._world.move(left._player, adjacent[:2])
            left._player.facing = adjacent[2]
            before = dict(left._player.inventory)
            if material in {"stone", "coal"}:
                left._player.inventory["wood_pickaxe"] = 0
                left.step(5)
                assert left._player.inventory[item] == before[item]
                left._player.inventory["wood_pickaxe"] = 1
            left.step(5)
            assert left._player.inventory[item] == before[item] + 1
            assert all(left._player.inventory[other] == before[other]
                       for other in ("wood", "stone", "coal") if other != item)
            left.reset()
    finally:
        for env in (left, right):
            close = getattr(env, "close", None)
            if callable(close):
                close()


def test_material_scene_initial_rgb_exposes_each_target_material():
    pytest.importorskip("crafter")
    from experiments.run_crafter_gather_spt_transfer_pilot import audit_scene_rgb_visibility

    audit = audit_scene_rgb_visibility(4)
    assert audit["material_count"] == 12
    assert audit["all_visible"]
    assert audit["minimum_changed_pixels"] > 0


def test_gather_pilot_audit_rejects_inconsistent_action_counts():
    from experiments.analyze_crafter_gather_spt_transfer_pilot import _check_evaluation

    evaluation = {
        "episodes": 1,
        "successes": 0,
        "terminal_counts": {"external_truncation": 1},
        "action_counts": {"0": 2},
        "first_target_gain_steps": [],
        "rows": [{"episode": 0, "success": False, "steps": 1,
                  "first_target_gain_step": None, "action_counts": {"0": 1}}],
    }
    errors = []
    _check_evaluation(evaluation, 1, "synthetic", errors)
    assert any("action count mismatch" in error for error in errors)


def test_gather_execution_chain_has_native_mechanical_headroom():
    pytest.importorskip("crafter")
    from experiments.crafter_material_gather_scene import make_material_gather_scene
    from experiments.run_crafter_gather_execution_chain_diagnostic import _run_chain

    rows = []
    for seed in range(4):
        scene = make_material_gather_scene(seed)
        rows.extend(_run_chain(scene, material, 8) for material in ("tree", "stone", "coal"))
    assert all(row["success"] for row in rows)
    assert max(row["route_length"] for row in rows) <= 8
    assert all(event["native_step_valid"] for row in rows for event in row["events"])
    assert all([event["remaining_shortest_steps"] for event in row["events"]]
               == list(range(row["route_length"], 0, -1)) for row in rows)


def test_gather_execution_chain_respects_initial_facing_and_tied_routes():
    pytest.importorskip("crafter")
    from experiments.crafter_material_gather_scene import MaterialGatherScene
    from experiments.run_crafter_gather_execution_chain_diagnostic import build_native_route

    materials = (("tree", (33, 32)), ("stone", (32, 30)), ("coal", (30, 32)))
    ready = MaterialGatherScene(0, materials, (1, 0))
    away = MaterialGatherScene(0, materials, (-1, 0))
    assert [row["action"] for row in build_native_route(ready, "tree")] == [5]
    assert [row["action"] for row in build_native_route(away, "tree")] == [2, 5]

    diagonal = MaterialGatherScene(0, (("tree", (30, 32)), ("stone", (32, 30)),
                                     ("coal", (33, 33))), (-1, 0))
    route = build_native_route(diagonal, "coal")
    assert len(route) == 3
    assert route[0]["shortest_actions"] == [2, 4]
