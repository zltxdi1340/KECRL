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


def test_e2_smoke_config_resolves_three_disjoint_holdout_folds():
    from experiments.run_crafter_gather_spt_transfer_e2 import resolve_fold_config

    config = json.loads(open("configs/crafter_gather_spt_transfer_e2_smoke_v1.yaml").read())
    resolved = [resolve_fold_config(config, fold) for fold in config["folds"]]
    assert [(item["fold_id"], tuple(item["source_materials"]), item["target_material"])
            for item in resolved] == [
        ("holdout_coal", ("tree", "stone"), "coal"),
        ("holdout_stone", ("tree", "coal"), "stone"),
        ("holdout_wood", ("stone", "coal"), "tree"),
    ]
    assert all(item["formal_result"] is False for item in resolved)
    assert all(item["action_rng_scheme"] == "per_episode_seed_v1" for item in resolved)
    undersized = dict(config)
    undersized["source_scene_pool_size"] = 3
    with pytest.raises(ValueError, match="one-step-per-episode"):
        resolve_fold_config(undersized, config["folds"][0])


def test_e2_candidate_preflight_matches_protocol_cost_and_replica_ranges():
    from experiments.analyze_crafter_gather_spt_transfer_e2_candidate import preflight

    result = preflight(
        "configs/crafter_gather_spt_transfer_e2_candidate_v1.yaml",
        "configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml",
    )
    assert result["valid"]
    assert result["formal_freeze_ready"] is False
    assert result["costs"]["e2_total_interaction_step_upper_bound"] == 6_846_720
    assert result["scene_pool_checks"]["cross_replica_role_ranges_disjoint"]
    assert result["scene_pool_checks"]["candidate_pool_capacity_sufficient"]
    assert result["scene_pool_checks"]["pool_size_per_role_per_slot"] == 256
    assert result["scene_pool_checks"]["action_seed_ranges"]["cross_role_and_replica_ranges_disjoint"]


def test_formal_manifest_requires_exact_hash_bound_snapshot(tmp_path):
    import hashlib
    import subprocess

    from experiments.crafter_formal_manifest import package_versions, verify_formal_manifest

    root = tmp_path
    config_path = root / "config.json"
    protocol_path = root / "protocol.json"
    source_path = root / "runner.py"
    manifest_path = root / "manifest.json"
    source_path.write_text("frozen runner\n", encoding="utf-8")
    protocol_path.write_text("{}\n", encoding="utf-8")
    config = {
        "formal_result": True,
        "diagnostic_only": False,
        "ready_for_formal_training": True,
        "formal_manifest_sealed": True,
        "formal_manifest": str(manifest_path),
        "formal_protocol": str(protocol_path),
        "replica_seeds": [101, 102],
    }
    config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "status": "sealed_formal_manifest_v1",
        "formal_result": True,
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "runtime": {"packages": package_versions()},
        "source_sha256": {str(source_path): sha(source_path)},
        "replica_seeds": [101, 102],
        "config_sha256": sha(config_path),
        "protocol_sha256": sha(protocol_path),
    }
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    manifest["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    manifest_path.with_suffix(".json.sha256").write_text(sha(manifest_path) + "\n", encoding="ascii")

    verified = verify_formal_manifest(config, config_path)
    assert verified["manifest_sha256"] == sha(manifest_path)
    source_path.write_text("changed runner\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source hash check failed"):
        verify_formal_manifest(config, config_path)


def test_formal_config_generation_freezes_thirty_replica_plan(tmp_path):
    from experiments.freeze_crafter_gather_spt_transfer_formal import generate_configs

    import shutil

    shutil.copytree("configs", tmp_path / "configs")
    generate_configs(tmp_path)
    e1 = json.loads((tmp_path / "configs/crafter_gather_spt_transfer_e1_formal_v1.json").read_text())
    e2 = json.loads((tmp_path / "configs/crafter_gather_spt_transfer_e2_formal_v1.json").read_text())
    protocol = json.loads((tmp_path / "configs/crafter_gather_spt_transfer_protocol_formal_v1.json").read_text())
    assert e1["formal_result"] is True and e1["diagnostic_only"] is False
    assert e2["replica_seeds"] == list(range(101, 131))
    assert e1["replica_seeds"] == e2["replica_seeds"]
    assert e1["action_rng_scheme"] == "per_episode_seed_v1"
    assert e1["source_scene_pool_size"] == 256
    assert protocol["formal_budget"]["values_frozen"] is True
    assert protocol["power_planning"]["minimum_meaningful_primary_effect"] == 0.10


def test_e2_candidate_manifest_plan_materializes_fold_slot_and_evaluation_seeds():
    from experiments.analyze_crafter_gather_spt_transfer_e2_candidate import build_candidate_manifest_plan

    config = json.loads(open("configs/crafter_gather_spt_transfer_e2_candidate_v1.yaml").read())
    protocol = json.loads(open("configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml").read())
    plan = build_candidate_manifest_plan(
        config,
        protocol,
        config_path="candidate-config",
        config_sha256="config-hash",
        protocol_sha256="protocol-hash",
        source_sha256={"runner": "runner-hash"},
        runtime={"python": "test"},
        git_commit="test-commit",
    )
    wood_fold = next(fold for fold in plan["folds"] if fold["fold_id"] == "holdout_wood")
    replica = next(row for row in wood_fold["replicas"] if row["replica_seed"] == 101)
    first_slot = replica["source_slots"][0]
    source_scene_base = 1_100_000 + 101 * 10_000_000
    source_action_base = 1_200_000 + 101 * 10_000_000
    assert first_slot["source_material"] == "stone"
    assert first_slot["support"]["scene_seed_start"] == source_scene_base
    assert first_slot["support"]["scene_seed_count"] == 256
    assert first_slot["meta_query"]["scene_seed_start"] == source_scene_base + 300
    assert first_slot["support"]["block_action_seed"] == source_action_base
    assert first_slot["meta_query"]["block_action_seed"] == source_action_base + 500
    assert replica["target_support_blocks"][0]["after_steps"] == 256
    assert replica["target_support_blocks"][0]["action_seed_count_upper_bound"] == 256
    assert replica["final_evaluations"][0]["episodes"][0] == {
        "episode": 0,
        "scene_seed": 1_300_000 + 101 * 10_000_000 + 1_000_000,
        "action_seed": 1_400_000 + 101 * 10_000_000 + 1_000_000,
    }
    assert plan["status"] == "candidate_schedule_not_sealed"
    assert len(plan["manifest_plan_sha256"]) == 64


def test_e2_primary_auc_and_seed_cluster_bootstrap_are_deterministic():
    from experiments.analyze_crafter_gather_spt_transfer_e2 import (
        _normalized_auc,
        _paired_seed_cluster_bootstrap,
    )

    auc = _normalized_auc([0, 1, 3], [0.0, 0.5, 1.0], budget=3)
    assert auc == pytest.approx(7 / 12)
    first = _paired_seed_cluster_bootstrap(
        [0.1, 0.2, 0.3], resamples=1000, level=0.95, random_seed=42
    )
    second = _paired_seed_cluster_bootstrap(
        [0.1, 0.2, 0.3], resamples=1000, level=0.95, random_seed=42
    )
    assert first == second
    assert first["method"] == "paired_seed_cluster_percentile_bootstrap"
    assert first["lower"] <= 0.2 <= first["upper"]
    with pytest.raises(ValueError, match="at least two"):
        _paired_seed_cluster_bootstrap(
            [0.2], resamples=1000, level=0.95, random_seed=42
        )


def test_e2_smoke_budget_is_not_misreported_as_primary_candidate_estimate():
    from experiments.analyze_crafter_gather_spt_transfer_e2 import _candidate_statistics

    config = json.loads(open("configs/crafter_gather_spt_transfer_e2_smoke_v1.yaml").read())
    protocol = json.loads(open("configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml").read())
    errors = []
    result = _candidate_statistics(config, {}, {}, protocol, errors)
    assert errors == []
    assert result["status"] == "not_estimable"
    assert result["formal_result"] is False
    assert any("checkpoints" in reason for reason in result["reasons"])
    assert any("episode count" in reason for reason in result["reasons"])
    assert any("replica seeds" in reason for reason in result["reasons"])


def test_e2_primary_analysis_averages_folds_inside_paired_seed_clusters(tmp_path):
    from experiments.analyze_crafter_gather_spt_transfer_e2 import _candidate_statistics

    protocol = json.loads(open("configs/crafter_gather_spt_transfer_protocol_candidate_v1.yaml").read())
    protocol["formal_budget_candidate"].update({
        "replica_seeds": [1, 2],
        "target_support_steps": 2,
        "evaluation_support_checkpoints": [0, 2],
        "evaluation_episodes_per_checkpoint": 2,
    })
    protocol["analysis_candidate"]["confidence_interval"].update({"resamples": 250, "analysis_seed": 17})
    protocol["_path"] = "synthetic-test-protocol"
    protocol["_sha256"] = "synthetic-test-hash"
    config = {
        "arms": list(protocol["arms"]),
        "replica_seeds": [1, 2],
        "target_checkpoints": [0, 2],
        "evaluation_episodes": 2,
    }
    success_counts = {
        ("A_initial_spt", 1): [0, 0],
        ("A_initial_spt", 2): [0, 0],
        ("B_fomaml_spt", 1): [1, 2],
        ("B_fomaml_spt", 2): [0, 1],
        ("C_ordinary_pretraining", 1): [0, 1],
        ("C_ordinary_pretraining", 2): [0, 0],
    }
    task_by_id = {task["spi_id"]: task for task in protocol["tasks"]}
    fold_aggregates = {}
    for fold in protocol["experiments"]["E2_leave_one_binding_out"]["folds"]:
        fold_id = fold["fold_id"]
        fold_config_path = tmp_path / f"{fold_id}.json"
        fold_config_path.write_text(json.dumps({"target_checkpoints": [0, 2]}), encoding="utf-8")
        runs = []
        for (arm, seed), counts in success_counts.items():
            target_rows = []
            for checkpoint_index, checkpoint in enumerate([0, 2]):
                rows = [
                    {
                        "scene_seed": 5000 + episode,
                        "action_seed": 6000 + episode,
                        "success": episode < counts[checkpoint_index],
                    }
                    for episode in range(2)
                ]
                successes = sum(row["success"] for row in rows)
                target_rows.append({
                    "support_steps": checkpoint,
                    "evaluation": {
                        "episodes": 2,
                        "successes": successes,
                        "success_rate": successes / 2,
                        "rows": rows,
                    },
                })
            runs.append({"arm": arm, "seed": seed, "target_rows": target_rows})
        fold_aggregates[fold_id] = {
            "config": str(fold_config_path),
                "source_materials": [task_by_id[spi_id]["material"] for spi_id in fold["source_spi_ids"]],
                "target_material": task_by_id[fold["target_spi_id"]]["material"],
            "runs": runs,
        }

    errors = []
    result = _candidate_statistics(config, {"git_commit": "synthetic"}, fold_aggregates, protocol, errors)
    assert errors == []
    assert result["status"] == "candidate_estimate_diagnostic_only"
    assert result["primary_effect"]["mean_B_minus_C_auc"] == pytest.approx(0.375)
    assert [row["B_minus_C"] for row in result["per_seed_primary_contrast"]] == pytest.approx([0.5, 0.25])
    assert all(row["paired_B_minus_C_auc"]["mean"] == pytest.approx(0.375)
               for row in result["per_fold_summary"])
    assert [point["support_steps"] for point in result["per_fold_summary"][0]["success_curve"]] == [0, 2]
    assert result["per_fold_summary"][0]["success_curve"][1]["success_rate_by_arm"]["B_fomaml_spt"]["mean"] == pytest.approx(0.75)
    assert result["primary_effect"]["confidence_interval"]["lower"] <= 0.375
    assert result["primary_effect"]["confidence_interval"]["upper"] >= 0.375


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


def test_collect_block_initializes_from_first_injective_schedule_seed():
    import hashlib

    from experiments.crafter_material_gather_scene import (
        INJECTIVE_SCENE_VERSION,
        MaterialGatherEnv,
        injective_scene_seed,
        make_material_gather_scene,
    )
    from experiments.run_crafter_gather_spt_transfer_pilot import collect_block

    class Policy:
        config = {"gamma": 0.99, "gae_lambda": 0.95}

        def distribution_value(self, *_args):
            logits = torch.zeros((1, 17))
            distribution = torch.distributions.Categorical(logits=logits)
            return distribution, torch.zeros(1), None

    class SPI:
        parameter_binding = {"item": "wood", "material": "tree"}

    seed = injective_scene_seed(101, "target_support", 0)
    expected_scene = make_material_gather_scene(seed, scheme=INJECTIVE_SCENE_VERSION)
    expected_hash = hashlib.sha256(
        MaterialGatherEnv(expected_scene, seed=seed, length=1).reset().tobytes()
    ).hexdigest()
    _rollout, summary = collect_block(
        Policy(), SPI(), steps=1, scene_seed_base=101 * 10_000_000 + 65_535,
        action_seed=7, device=torch.device("cpu"),
        max_episode_steps=1, training=False, role="target_support",
        scene_seed_schedule=(seed,), action_rng_scheme="per_episode_seed_v1",
        scene_scheme=INJECTIVE_SCENE_VERSION,
    )
    assert summary["episode_manifest"][0]["scene_seed"] == seed
    assert summary["episode_manifest"][0]["initial_rgb_sha256"] == expected_hash


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


def test_gather_pilot_respects_distinct_meta_and_ordinary_learning_rates():
    from experiments.run_crafter_gather_spt_transfer_pilot import _build_source_learner

    spt = build_gather_spt()
    candidate = SpatialGatherSPT(spt, build_matched_spatial_policy(_policy_config(), "cnn_only"))
    config = {"inner_learning_rate": 3e-4, "meta_learning_rate": 1e-3,
              "policy_learning_rate": 3e-4}
    for arm, expected in (("B_fomaml_spt", 1e-3), ("C_ordinary_pretraining", 3e-4)):
        learner = _build_source_learner(config, candidate, arm)
        assert learner.inner_lr == 3e-4
        assert {group["lr"] for optimizer in (learner.template_optimizer, learner.generator_optimizer)
                for group in optimizer.param_groups} == {expected}


def test_gather_pilot_preserves_legacy_e1_tree_source_default():
    from experiments.run_crafter_gather_spt_transfer_pilot import _source_material_bindings

    assert _source_material_bindings({"source_task_slots_per_round": 2}, 2) == ("tree", "tree")
    with pytest.raises(ValueError, match="must not repeat"):
        _source_material_bindings({"source_materials": ["tree", "tree"]}, 2)


def test_gather_pilot_audit_rejects_overridden_meta_learning_rate():
    from experiments.analyze_crafter_gather_spt_transfer_pilot import _check_source_optimizer_settings

    errors = []
    config = {"inner_learning_rate": 3e-4, "meta_learning_rate": 1e-3,
              "policy_learning_rate": 3e-4}
    settings = {"inner_learning_rate": 3e-4, "template_learning_rates": [3e-4],
                "generator_learning_rates": [3e-4]}
    assert not _check_source_optimizer_settings(settings, config, "B_fomaml_spt", "B", errors)
    assert len(errors) == 2
    assert not _check_source_optimizer_settings(None, config, "B_fomaml_spt", "legacy", [])


def test_gather_comparability_audit_rejects_missing_episode_manifest():
    from experiments.analyze_crafter_gather_spt_transfer_comparability import _check_block_manifest

    summary = {
        "role": "source_support",
        "requested_steps": 2,
        "actual_steps": 2,
        "episodes": 1,
        "episode_manifest": [],
    }
    errors = []
    _check_block_manifest(summary, "source_support", 2, "synthetic", errors, verify_rgb=False)
    assert any("episode manifest count mismatch" in error for error in errors)


def test_gather_comparability_audit_rejects_wrong_scene_record():
    from experiments.analyze_crafter_gather_spt_transfer_comparability import _check_block_manifest
    from experiments.crafter_material_gather_scene import make_material_gather_scene

    scene = make_material_gather_scene(0)
    summary = {
        "role": "source_support",
        "requested_steps": 1,
        "actual_steps": 1,
        "episodes": 1,
        "episode_manifest": [{
            "scene_seed": 0,
            "scene": make_material_gather_scene(1).record(),
            "initial_rgb_sha256": "synthetic",
        }],
    }
    errors = []
    _check_block_manifest(summary, "source_support", 1, "synthetic", errors, verify_rgb=False)
    assert any("scene record does not match scene seed" in error for error in errors)
    assert scene.record() != summary["episode_manifest"][0]["scene"]


def test_gather_comparability_audit_rejects_wrong_material_binding():
    from experiments.analyze_crafter_gather_spt_transfer_comparability import _check_block_manifest

    summary = {
        "role": "source_support",
        "material": "tree",
        "requested_steps": 1,
        "actual_steps": 1,
        "episodes": 0,
        "episode_manifest": [],
    }
    errors = []
    _check_block_manifest(summary, "source_support", 1, "synthetic", errors,
                          verify_rgb=False, expected_material="stone")
    assert any("material" in error for error in errors)


def test_gather_aligned_source_pools_pair_b_and_c_roles_without_overlap():
    from experiments.run_crafter_gather_spt_transfer_pilot import _source_scene_pool

    config = {"source_scene_pool_size": 64, "source_slot_stride": 200,
              "source_query_pool_offset": 100}
    support = set(_source_scene_pool(config, 2_100_000, 3, "source_support"))
    meta_query = set(_source_scene_pool(config, 2_100_000, 3, "source_meta_query"))
    ordinary_support = set(_source_scene_pool(config, 2_100_000, 3, "source_ordinary_support"))
    ordinary_query = set(_source_scene_pool(config, 2_100_000, 3, "source_ordinary_query"))
    assert len(support) == len(meta_query) == 64
    assert support.isdisjoint(meta_query)
    assert support == ordinary_support
    assert meta_query == ordinary_query


def test_injective_scene_seed_schedule_separates_roles_and_preserves_rgb_identity():
    import hashlib

    from experiments.crafter_material_gather_scene import (
        INJECTIVE_SCENE_VERSION,
        MaterialGatherEnv,
        injective_scene_seed,
        make_material_gather_scene,
    )

    seeds = {
        role: [injective_scene_seed(101, role, index) for index in range(8)]
        for role in ("source_support", "source_meta_query", "target_support", "final_evaluation")
    }
    assert all(set(left).isdisjoint(right)
               for index, left in enumerate(seeds.values())
               for right in list(seeds.values())[index + 1:])
    env = None
    digests = []
    for seed in (seeds["source_support"][0], seeds["source_meta_query"][0],
                 seeds["target_support"][0], seeds["final_evaluation"][0]):
        scene = make_material_gather_scene(seed, scheme=INJECTIVE_SCENE_VERSION)
        if env is None:
            env = MaterialGatherEnv(scene, seed=seed, length=1)
        else:
            env.configure(scene, seed=seed)
        digests.append(hashlib.sha256(env.reset().tobytes()).hexdigest())
    assert len(set(digests)) == len(digests)


def test_injective_scene_seed_schedule_fails_when_role_capacity_is_exceeded():
    import pytest

    from experiments.crafter_material_gather_scene import (
        INJECTIVE_SCENE_VERSION,
        SCENE_LAYOUT_ROLE_PARTITIONS,
        injective_scene_seed,
        make_material_gather_scene,
    )

    role_capacity = SCENE_LAYOUT_ROLE_PARTITIONS["final_evaluation"][1]
    seed = injective_scene_seed(101, "final_evaluation", role_capacity)
    with pytest.raises(ValueError, match="role counter exceeds"):
        make_material_gather_scene(seed, scheme=INJECTIVE_SCENE_VERSION)


def test_gather_comparability_audit_rejects_scene_outside_frozen_v2_role_schedule():
    import hashlib
    import pytest

    from experiments.analyze_crafter_gather_spt_transfer_comparability import _check_block_manifest
    from experiments.crafter_material_gather_scene import (
        INJECTIVE_SCENE_VERSION,
        MaterialGatherEnv,
        injective_scene_seed,
        make_material_gather_scene,
    )

    valid_seed = injective_scene_seed(101, "target_support", 0)
    wrong_seed = injective_scene_seed(101, "final_evaluation", 0)
    scene = make_material_gather_scene(wrong_seed, scheme=INJECTIVE_SCENE_VERSION)
    env = MaterialGatherEnv(scene, seed=wrong_seed, length=1)
    rgb_hash = hashlib.sha256(env.reset().tobytes()).hexdigest()
    summary = {
        "role": "target_support",
        "material": "coal",
        "requested_steps": 1,
        "actual_steps": 1,
        "episodes": 1,
        "episode_manifest": [{
            "scene_seed": wrong_seed,
            "scene": scene.record(),
            "initial_rgb_sha256": rgb_hash,
        }],
    }
    errors = []
    _check_block_manifest(summary, "target_support", 1, "synthetic", errors,
                          verify_rgb=False, replica_seed=101,
                          expected_scene_counter_start=0)
    assert wrong_seed != valid_seed
    assert any("differs from frozen role schedule" in error for error in errors)
