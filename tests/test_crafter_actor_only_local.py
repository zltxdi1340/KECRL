import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("crafter")

from experiments.crafter_actor_only_local import (
    action_metrics, evaluate_actor, fit_actor_copy, fixture_action_targets, install_actor_copy,
)
from experiments.crafter_controlled_collection_scene import CollectionScene
from experiments.crafter_spatial_readout_data import BackgroundCollectionEnv, scene_labels, scene_manifest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_actor_only_local import (
    _background_rejection, _episode, _generate_dataset, _rollout_stats, _validate,
)
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_determinism import stable_crafter_object_order


def _config():
    return json.loads(Path("configs/crafter_wood3_actor_only_local_cuda_v1.yaml").read_text())


def _labels():
    return np.asarray([scene_labels(scene) for scene in scene_manifest()], dtype=np.int64)


def test_fixture_targets_cover_six_actual_actions_with_no_tree_diagnostic_noop():
    labels = _labels()
    targets = fixture_action_targets(labels)
    assert np.bincount(targets).tolist() == [12, 9, 9, 9, 9, 12]
    assert np.all(targets[labels[:, 0] == 0] == 0)
    assert np.all(targets[labels[:, 2] == 1] == 5)
    assert np.array_equal(targets[labels[:, 2] == 2], labels[labels[:, 2] == 2, 0])


def test_balanced_actor_metrics_penalize_do_bias_sleep_and_illegal_predictions():
    targets = fixture_action_targets(_labels())
    logits = torch.zeros(len(targets), 17)
    logits[:, 5] = 20
    logits[:, 7:] = 100  # Never legal under the original policy's mask.
    metrics = action_metrics(logits, targets)
    assert metrics["balanced_accuracy"] == pytest.approx(1 / 6)
    assert metrics["tree_present_balanced_accuracy"] == pytest.approx(.2)
    assert metrics["ready_do_rate"] == 1
    assert metrics["unaligned_target_move_rate"] == 0
    logits[:, 6] = 30
    assert action_metrics(logits, targets)["balanced_accuracy"] == 0


def test_actor_fit_uses_actual_raw_head_and_isolates_encoder_critic_and_optimizer():
    config = {**_config(), "actor_fit_epochs": 60, "validation_interval": 10, "actor_learning_rates": [.03]}
    source = build_matched_spatial_policy(config["policy"], "cnn_only")
    # Populate optimizer momentum so checking only parameter values cannot pass.
    sum(parameter.square().sum() for parameter in source.parameters()).backward()
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    labels = _labels()
    targets = fixture_action_targets(labels)
    features = np.pad(np.eye(6, dtype=np.float32)[targets], ((0, 0), (0, 122)))
    features[:, -1] = .17  # An unchanged raw feature; no standardizer.
    head, artifact = fit_actor_copy(source, features, targets, features, targets, config, "cpu")
    fitted = install_actor_copy(source, head)
    assert isinstance(fitted.actor, torch.nn.Linear) and fitted.actor.in_features == 128 and fitted.actor.out_features == 17
    assert artifact["feature_preprocessing"] == "none" and not artifact["selection_uses_heldout"]
    assert artifact["supervised_optimizer_updates"] == 60
    assert artifact["masked_out_action_rows_unchanged"]
    assert _state_equal(before, (source.state_dict(), source.optimizer.state_dict()))
    assert _state_equal(source.optimizer.state_dict(), fitted.optimizer.state_dict())
    assert all(torch.equal(value, fitted.state_dict()[key]) for key, value in source.state_dict().items() if not key.startswith("actor."))
    assert not torch.equal(source.actor.weight, fitted.actor.weight)
    state = copy.deepcopy(fitted.state_dict())
    selected = (artifact["selected_epoch"], artifact["actor_state_canonical_digest"])
    metrics = evaluate_actor(fitted, features, labels, np.zeros(len(labels), dtype=np.int64), "cpu")
    perturbed = evaluate_actor(fitted, np.zeros_like(features), labels, np.zeros(len(labels), dtype=np.int64), "cpu")
    assert metrics["balanced_accuracy"] == 1
    assert perturbed["balanced_accuracy"] < metrics["balanced_accuracy"]
    assert _state_equal(state, fitted.state_dict())
    assert selected == (artifact["selected_epoch"], artifact["actor_state_canonical_digest"])
    assert all(p.grad is None and not p.requires_grad for p in fitted.parameters())


def test_fit_rejects_missing_targets_and_wrong_policy_architecture():
    config = {**_config(), "actor_fit_epochs": 1, "validation_interval": 1}
    source = build_matched_spatial_policy(config["policy"], "cnn_only")
    features = np.zeros((6, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="six fixture"):
        fit_actor_copy(source, features, np.zeros(6, dtype=np.int64), features, np.arange(6), config, "cpu")
    recurrent = build_matched_spatial_policy(config["policy"], "cnn_gru")
    with pytest.raises(ValueError, match="CNN-only"):
        fit_actor_copy(recurrent, features, np.arange(6), features, np.arange(6), config, "cpu")


class _ScriptedRGBPolicy:
    def __init__(self, actions):
        self.actions, self.index = actions, 0

    def distribution_value(self, observation, hidden, legal_actions):
        assert observation.shape == (3, 64, 64) and hidden is None
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        probs = torch.zeros((1, 17))
        probs[0, action] = 1
        return torch.distributions.Categorical(probs=probs), torch.zeros(1), None


def test_rollout_validates_native_turn_then_designated_collection_and_observer(tmp_path):
    config = _config()
    scene = CollectionScene(1, 3, 1, True)
    native = BackgroundCollectionEnv(scene, seed=0)
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    try:
        with stable_crafter_object_order():
            native.set_background(41000000)
            first = _episode(_ScriptedRGBPolicy([3, 5]), adapter, scene, 41000000, 45000000, "greedy", config, "cpu")
            control = _episode(_ScriptedRGBPolicy([3, 5]), adapter, scene, 41000000, 45000000, "greedy", config, "cpu", observe=False)
        assert first == control
        assert first["designated_tree_success"] and first["first_designated_collection_step"] == 2
        assert first["steps"] == 2 and not first["other_tree_wood_gain"]
        assert [row["action"] for row in first["events"]] == [3, 5]
        assert _rollout_stats([first])["designated_success_rate"] == 1
    finally:
        adapter.close()


def test_other_tree_gains_cannot_count_as_designated_success():
    config = _config()
    scene = CollectionScene(0, 3, 1, True)
    native = BackgroundCollectionEnv(scene, seed=0)
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    reset = native.reset

    def add_other_tree():
        reset()
        center = native._player.pos
        native._world[(center[0] - 1, center[1])] = "tree"
        return native._obs()

    try:
        with stable_crafter_object_order():
            native.set_background(41000000)
            native.reset = add_other_tree
            row = _episode(_ScriptedRGBPolicy([5, 0]), adapter, scene, 41000000, 45, "greedy", config, "cpu")
        assert row["any_wood_gain"] and row["other_tree_wood_gain"] == 1
        assert not row["designated_tree_success"] and row["steps"] == 8
    finally:
        adapter.close()


def test_no_adjacent_tree_fixture_can_gain_other_wood_without_false_target_success():
    config = _config()
    scene = CollectionScene(0, 1, 1, False)
    native = BackgroundCollectionEnv(scene, seed=0)
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    reset = native.reset

    def add_distant_tree():
        reset()
        center = native._player.pos
        native._world[(center[0] - 2, center[1])] = "tree"
        return native._obs()

    try:
        with stable_crafter_object_order():
            native.set_background(41000000)
            native.reset = add_distant_tree
            row = _episode(_ScriptedRGBPolicy([1, 5, 0]), adapter, scene, 41000000, 45, "greedy", config, "cpu")
        assert row["any_wood_gain"] and row["other_tree_wood_gain"] == 1
        assert not row["designated_tree_success"]
    finally:
        adapter.close()


def test_rollout_records_native_lava_death_outside_clear_center():
    config = _config()
    scene = CollectionScene(0, 3, 1, True)
    native = BackgroundCollectionEnv(scene, seed=0)
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    reset = native.reset

    def add_lava():
        reset()
        center = native._player.pos
        native._world[(center[0] - 2, center[1])] = "lava"
        return native._obs()

    try:
        with stable_crafter_object_order():
            native.set_background(41000000)
            native.reset = add_lava
            row = _episode(_ScriptedRGBPolicy([1]), adapter, scene, 41000000, 45, "greedy", config, "cpu")
        assert row["death"] and row["end_reason"] == "death"
        assert row["health_lost"] and not row["designated_tree_success"]
    finally:
        adapter.close()


def test_new_dataset_rejects_old_visual_backgrounds_before_splitting(tmp_path):
    config = {**_config(), "background_counts": {"train": 1, "validation": 1, "heldout": 1}}
    with stable_crafter_object_order():
        _, _, first = _generate_dataset(config, tmp_path, set())
        old = {json.loads(line)["background_rgb_sha256"] for line in (tmp_path / "scene_records.jsonl").read_text().splitlines()}
        second_root = tmp_path / "second"
        second_root.mkdir()
        data, records, second = _generate_dataset(config, second_root, old)
    assert first["rows"] == second["rows"] == 180
    assert not old & {row["background_rgb_sha256"] for row in records}
    assert all(row["reason"] == "previous_spatial_readout_background" for row in second["rejected_backgrounds"] if row["background_rgb_sha256"] in old)
    assert np.bincount(data["split_code"]).tolist() == [60, 60, 60]
    assert _background_rejection("old", set(), {"old"}) == "previous_spatial_readout_background"
    assert _background_rejection("new", {"new"}, set()) == "duplicate_in_current_dataset"


def test_fixed_protocol_rejects_teacher_scope_architecture_and_seed_leakage(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _config()
    baseline = json.loads(Path("configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml").read_text())
    previous = [json.loads(Path(p).read_text()) for p in config["previous_diagnostic_configs"]]
    readout = json.loads(Path("configs/crafter_wood3_spatial_readout_cuda_v1.yaml").read_text())
    _validate(config, baseline, previous, readout)
    for key, value in (("teacher_used", False), ("source_policy_updates_allowed", True),
                       ("encoder_updates_allowed", True), ("feature_preprocessing", "standardized"),
                       ("actor_architecture", "mlp64"), ("background_seed_base", readout["background_seed_base"]),
                       ("diagnostic_action_seed_base", baseline["action_seed_base"]),
                       ("background_counts", {"train": 40, "validation": 4, "heldout": 4})):
        with pytest.raises(ValueError):
            _validate({**config, key: value}, baseline, previous, readout)
