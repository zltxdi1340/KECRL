import copy
import json

import numpy as np
import pytest
import torch

from experiments import analyze_libero_task4_task7_expert_action_error_diagnostic as diagnostic


@pytest.mark.parametrize("field,value", [
    ("policy_optimizer_updates", 1000), ("new_optimizer_updates", 1), ("new_episodes", 1),
    ("first_scored_action_index", 0), ("qualification_accessed", True),
    ("training_seeds", [20261022, 20261023]),
])
def test_scope_changes_fail_before_data_or_weight_access(tmp_path, field, value):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["scope"][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="500-update zero-interaction"):
        diagnostic.load_protocol(path)


def test_same_index_alignment_cannot_replace_the_frozen_causal_alignment(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["analysis"]["alignment"] = "stored_same_index"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="alignment or metrics"):
        diagnostic.load_protocol(path)


def test_only_source_tasks_and_two_demo_roles_can_reach_authorization(monkeypatch):
    monkeypatch.setattr(diagnostic, "resolve_demonstration_access", lambda *args, **kwargs: pytest.fail("forbidden task/role reached resolver"))
    for task, role in [(2, "meta_support"), (5, "meta_query"), (4, "qualification")]:
        with pytest.raises(ValueError, match="outside source"):
            diagnostic.authorize_role({}, {}, task, role)


def test_repository_role_lock_is_preserved():
    transfer = json.loads((diagnostic.ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json").read_text())
    manifest = json.loads((diagnostic.ROOT / "configs/libero_spatial_dataset_manifest_v1.json").read_text())
    access = diagnostic.authorize_role(transfer, manifest, 4, "meta_query")
    assert access.demonstration_indices == tuple(range(30, 50))
    transfer["locked_access"]["before_method_freeze"].append("source_spi.meta_query")
    with pytest.raises(PermissionError, match="locked before method freeze"):
        diagnostic.authorize_role(transfer, manifest, 4, "meta_query")


def test_causal_batch_shifts_observations_by_one_and_never_reads_private_fields():
    length = 5
    observations = {}
    for name, (shape, dtype) in diagnostic.POLICY_OBSERVATION_SPECS.items():
        observations[name] = np.stack([np.full(shape, step if name.endswith("rgb") else step / 10, dtype) for step in range(length)])
    class ProtectedGroup:
        def __getitem__(self, name):
            assert name == "obs", "private simulator data was requested"
            return observations
        @property
        def attrs(self):
            pytest.fail("initial state attributes were requested")
    actions = np.zeros((length, 7), np.float32)
    batch = diagnostic.read_causal_batch(ProtectedGroup(), actions, 1, 5)
    assert batch["agentview_rgb"][:, 0, 0, 0].tolist() == [0, 1, 2, 3]
    assert batch["joint_states"][:, 0].tolist() == observations["joint_states"][:4, 0].tolist()
    with pytest.raises(ValueError, match="exclude t0"):
        diagnostic.read_causal_batch(ProtectedGroup(), actions, 0, 4)


def test_expert_command_masks_have_exact_transition_windows_and_exclude_t0():
    actions = np.zeros((30, 7), np.float32)
    actions[:, 6] = -1
    actions[10:20, 6] = 1
    masks = diagnostic.command_masks(actions)
    assert all(not mask[0] for mask in masks.values())
    assert np.flatnonzero(masks["first_scored_t1"]).tolist() == [1]
    assert np.flatnonzero(masks["around_close_transition"]).tolist() == list(range(5, 16))
    assert np.flatnonzero(masks["around_open_transition"]).tolist() == list(range(15, 26))
    assert masks["expert_close"].sum() == 10
    assert masks["expert_open"].sum() == 19
    assert masks["expert_neutral"].sum() == 0


def test_action0_is_used_only_as_previous_command_for_a_t1_transition():
    actions = np.zeros((9, 7), np.float32)
    actions[:, 6] = 1
    actions[0, 6] = -1
    masks = diagnostic.command_masks(actions)
    assert np.flatnonzero(masks["around_close_transition"]).tolist() == list(range(1, 7))
    assert not masks["expert_open"].any()


def test_component_mse_uses_three_three_one_dimension_weighting():
    target = np.zeros((2, 7), np.float64)
    prediction = np.tile([.1, .1, .1, .2, .2, .2, .3], (2, 1))
    metrics = diagnostic.error_metrics(prediction, target, np.ones(2, bool))
    assert metrics["component_mse"] == pytest.approx({"translation": .01, "rotation": .04, "gripper": .09})
    assert metrics["action_mse"] == pytest.approx(.24 / 7)
    assert sum(metrics["uniform7_mse_terms"].values()) == pytest.approx(metrics["action_mse"])
    assert sum(metrics["squared_error_fractions"].values()) == pytest.approx(1)


def test_empty_command_group_has_null_errors_and_rates_instead_of_perfect_scores():
    target = np.zeros((2, 7), np.float32)
    metrics = diagnostic.error_metrics(target.copy(), target, np.zeros(2, bool))
    assert metrics["valid_steps"] == 0 and metrics["action_mse"] is None
    assert all(value is None for value in metrics["component_mse"].values())
    assert metrics["gripper"]["sign_error_rate"] is None
    assert metrics["gripper"]["missed_close_rate"] is None
    assert sum(sum(row) for row in metrics["gripper"]["confusion_counts"]) == 0
    json.dumps(metrics, allow_nan=False)


def test_gripper_sign_error_and_directional_confusion_are_conditioned_on_expert_command():
    target = np.zeros((3, 7), np.float32)
    target[:, 6] = [1, -1, 0]
    prediction = target.copy()
    prediction[:, 6] = [-.4, .1, 0]
    metrics = diagnostic.error_metrics(prediction, target, np.ones(3, bool))["gripper"]
    assert metrics["nonzero_expert_steps"] == 2
    assert metrics["sign_error_rate"] == metrics["missed_close_rate"] == metrics["expert_open_pred_close_rate"] == 1
    assert metrics["confusion_counts"] == [[0, 0, 1], [0, 1, 0], [1, 0, 0]]


def test_nonfinite_action_and_wrong_mask_dtype_fail_closed():
    target = np.zeros((1, 7), np.float32)
    prediction = target.copy()
    prediction[0, 0] = np.nan
    with pytest.raises(ValueError, match="action interface"):
        diagnostic.error_metrics(prediction, target, np.ones(1, bool))
    with pytest.raises(ValueError, match="mask malformed"):
        diagnostic.error_metrics(target, target, np.ones(1, np.int64))


def public_batch():
    return {name: np.zeros((2, *shape), dtype=dtype) for name, (shape, dtype) in diagnostic.POLICY_OBSERVATION_SPECS.items()}


def test_two_arms_receive_the_same_shared_tensors():
    seen = []
    class RecordingPolicy:
        def forward_step(self, *args, hidden):
            seen.append(tuple(value.data_ptr() for value in args))
            return torch.zeros((len(args[0]), 7)), None
    observations = public_batch()
    result = diagnostic.predict_shared_batch({"baseline": RecordingPolicy(), "jitter": RecordingPolicy()}, observations)
    assert seen[0] == seen[1]
    assert np.array_equal(result["baseline"], result["jitter"])


def test_model_mutation_of_an_expert_observation_is_rejected():
    class MutatingPolicy:
        def forward_step(self, *args, hidden):
            args[2][0, 0] = .7
            return torch.zeros((len(args[0]), 7)), None
    with pytest.raises(ValueError, match="mutated shared expert inputs"):
        diagnostic.predict_shared_batch({"bad": MutatingPolicy()}, public_batch())


def test_mean_improvement_does_not_become_a_three_seed_consistency_claim():
    target = np.zeros((2, 7), np.float32)
    models = {}
    for index, seed in enumerate(diagnostic.SEEDS):
        for arm in diagnostic.ARMS:
            prediction = np.full((2, 7), .2 if arm == diagnostic.ARMS[0] else (.1 if index < 2 else .3), np.float32)
            metrics = diagnostic.error_metrics(prediction, target, np.ones(2, bool))
            models[diagnostic.model_key(seed, arm)] = {"metrics_by_mask": {name: copy.deepcopy(metrics) for name in diagnostic.MASKS}}
    summary = diagnostic.summarize_role(models)["all"]
    assert summary["component_seed_delta_stats"]["translation"]["mean"] < 0
    assert summary["component_all_three_seed_mse_improvement"]["translation"] is False


def test_prediction_archive_cannot_overwrite_existing_artifact(tmp_path):
    path = tmp_path / "historical.npz"
    path.write_bytes(b"historical")
    with pytest.raises(FileExistsError):
        diagnostic.save_archive(path, {"targets": np.zeros((1, 7))})
    assert path.read_bytes() == b"historical"


def test_existing_result_is_preserved_before_loading_protocol(tmp_path, monkeypatch):
    output = tmp_path / "historical.json"
    output.write_text("historical")
    monkeypatch.setattr(diagnostic, "load_protocol", lambda *args: pytest.fail("existing result reached input load"))
    with pytest.raises(ValueError, match="already exists"):
        diagnostic.run_analysis(tmp_path / "unused", output, tmp_path / "archive")
    assert output.read_text() == "historical"
