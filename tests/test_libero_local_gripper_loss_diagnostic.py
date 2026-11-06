import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import run_libero_task4_task7_local_gripper_loss_diagnostic as diagnostic
from src.algorithms.libero_bc_local_gripper_loss import expert_open_transition_mask, masked_local_gripper_mse
from src.algorithms.libero_bc_policy import masked_behavior_cloning_loss


def test_open_windows_are_unions_clip_at_demo_boundary_and_exclude_t0():
    commands = np.ones(24, np.float32)
    commands[2:4] = -1
    commands[7:9] = -1
    commands[22:] = -1
    mask = expert_open_transition_mask(commands)
    assert np.flatnonzero(mask).tolist() == list(range(1, 13)) + list(range(17, 24))
    assert mask.dtype == bool and not mask[0]
    assert not expert_open_transition_mask(np.full(24, -1., np.float32)).any()


@pytest.mark.parametrize("commands,radius", [(np.asarray([np.nan]), 5), (np.asarray([2.]), 5),
                                          (np.ones((2, 2)), 5), (np.ones(4), -1)])
def test_invalid_transition_labels_fail_closed(commands, radius):
    with pytest.raises(ValueError):
        expert_open_transition_mask(commands, radius=radius)


def test_local_loss_doubles_only_selected_valid_gripper_gradients():
    prediction = (torch.arange(56, dtype=torch.float64).reshape(2, 4, 7) / 100).requires_grad_()
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, True, True, False], [True, True, False, False]])
    local = torch.tensor([[False, True, False, True], [True, False, True, True]])
    baseline = masked_behavior_cloning_loss(prediction, target, valid)
    original_gradient = torch.autograd.grad(baseline, prediction, retain_graph=True)[0]
    weighted = masked_local_gripper_mse(prediction, target, valid, local)
    gradient = torch.autograd.grad(weighted, prediction)[0]
    expected = original_gradient.clone()
    expected[..., 6][local & valid] *= 2
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
    extra = (prediction[..., 6].square() * (valid & local)).sum() / (7 * valid.sum())
    torch.testing.assert_close(weighted, baseline + extra, rtol=0, atol=0)
    assert torch.count_nonzero(gradient[~valid]) == 0


def test_unit_weight_or_empty_window_returns_the_exact_original_loss():
    prediction = torch.linspace(-.5, .5, 28).reshape(1, 4, 7).requires_grad_()
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, True, True, False]])
    original = masked_behavior_cloning_loss(prediction, target, valid)
    assert torch.equal(masked_local_gripper_mse(prediction, target, valid, torch.ones_like(valid), gripper_weight=1), original)
    assert torch.equal(masked_local_gripper_mse(prediction, target, valid, torch.zeros_like(valid)), original)


def test_local_loss_rejects_wrong_masks_weight_and_no_valid_steps():
    prediction = torch.zeros(1, 3, 7)
    valid = torch.ones(1, 3, dtype=torch.bool)
    with pytest.raises(ValueError, match="boolean"):
        masked_local_gripper_mse(prediction, prediction, valid, valid.float())
    with pytest.raises(ValueError, match="at least 1"):
        masked_local_gripper_mse(prediction, prediction, valid, valid, gripper_weight=.5)
    with pytest.raises(ValueError, match="at least one"):
        masked_local_gripper_mse(prediction, prediction, ~valid, valid)


class FakeSupportDataset:
    sequence_length = 10
    def __init__(self):
        self.access = SimpleNamespace(task_id=4, task_role="source_spi", demonstration_role="meta_support", demonstration_indices=(0, 1))
        actions = np.zeros((25, 7), np.float32)
        actions[:, 6] = 1
        actions[11:, 6] = -1
        self.data = {"data": {"demo_0": {"actions": actions}, "demo_1": {"actions": np.ones((25, 7), np.float32)}}}
        self.windows = tuple(SimpleNamespace(demonstration_index=demo, start=start, valid_length=min(10, 25 - start))
                             for demo in (0, 1) for start in (0, 10, 20))
    def _open(self):
        return self.data
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, index):
        window = self.windows[index]
        valid = torch.zeros(10, dtype=torch.bool)
        valid[:window.valid_length] = True
        return {"valid_mask": valid, "demonstration_index": window.demonstration_index,
                "start_index": window.start, "actions": torch.zeros(10, 7), "joint_states": torch.zeros(10, 7)}


def test_dataset_computes_mask_before_chunking_and_does_not_leak_between_demos():
    base = FakeSupportDataset()
    wrapped = diagnostic.OpenTransitionDataset(base)
    assert wrapped[0]["open_transition_mask"].tolist() == [False] * 6 + [True] * 4
    assert wrapped[1]["open_transition_mask"].tolist() == [True] * 7 + [False] * 3
    assert not wrapped[2]["open_transition_mask"].any()
    assert not wrapped[3]["open_transition_mask"].any()
    for index in range(len(base)):
        actual, expected = wrapped[index], base[index]
        assert set(actual) == set(expected) | {"open_transition_mask"}
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(actual[key], value)
            else:
                assert actual[key] == value
        assert not actual["open_transition_mask"][~actual["valid_mask"]].any()


def test_query_dataset_cannot_supply_training_loss_labels():
    dataset = FakeSupportDataset()
    dataset.access.demonstration_role = "meta_query"
    dataset._open = lambda: pytest.fail("query data reached training-mask reader")
    with pytest.raises(ValueError, match="source meta_support"):
        diagnostic.OpenTransitionDataset(dataset)


@pytest.mark.parametrize("section,field,value", [("scope", "optimizer_updates_per_model", 1000),
    ("scope", "policy_rollout_episode_budget", 1), ("scope", "qualification_accessed", True),
    ("training", "candidate_gripper_weight", 6), ("training", "loss_denominator", "sum_weights")])
def test_training_protocol_changes_fail_before_model_or_environment_access(tmp_path, section, field, value):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="500-update paired training scope"):
        diagnostic.load_protocol(path)


def test_nonregression_threshold_cannot_change_after_the_protocol_is_frozen(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["stop_rules"]["query_all_translation_mse_max_seed_mean_ratio"] = 2.
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="stop/resource rules"):
        diagnostic.load_protocol(path)


def training_row():
    return {"optimizer_updates": 500, "initial_model_sha256": "initial", "batch_order_sha256": "order",
            "first_input_tensor_sha256": "input", "first_target_tensor_sha256": "target",
            "valid_step_updates": 100, "potentially_weighted_step_updates": 10, "baseline_final_weight_parity": True}


def test_query_gate_requires_all_six_endpoints_and_exact_pair_metadata():
    rows = {str(seed): {"arms": {arm: training_row() for arm in diagnostic.ARMS}} for seed in diagnostic.SEEDS}
    assert diagnostic.require_all_support_fits(rows) == 3000
    broken = copy.deepcopy(rows)
    del broken[str(diagnostic.SEEDS[-1])]
    with pytest.raises(ValueError, match="all three support seeds"):
        diagnostic.require_all_support_fits(broken)
    broken = copy.deepcopy(rows)
    broken[str(diagnostic.SEEDS[-1])]["arms"][diagnostic.ARMS[1]]["batch_order_sha256"] = "different"
    with pytest.raises(ValueError, match="paired training differs"):
        diagnostic.require_all_support_fits(broken)
    broken = copy.deepcopy(rows)
    broken[str(diagnostic.SEEDS[0])]["arms"][diagnostic.ARMS[0]]["baseline_final_weight_parity"] = False
    with pytest.raises(ValueError, match="baseline hash parity"):
        diagnostic.require_all_support_fits(broken)


def fake_evaluations():
    tasks = {}
    for task in diagnostic.TASKS:
        models = {}
        for seed in diagnostic.SEEDS:
            for arm in diagnostic.ARMS:
                candidate = arm == diagnostic.ARMS[1]
                models[f"seed{seed}__{arm}"] = {
                    "all": {"component_mse": {"translation": .101 if candidate else .1,
                                              "rotation": .0102 if candidate else .01, "gripper": .19 if candidate else .2},
                            "action_mse": .099 if candidate else .1},
                    "around_open_transition": {"component_mse": {"gripper": .7 if candidate else .8},
                                               "gripper": {"sign_error_rate": .34 if candidate else .35}}}
        tasks[str(task)] = {"meta_query": {"metrics_by_model": models}}
    return tasks


def test_offline_gate_only_allows_a_separately_frozen_followup():
    result = diagnostic.offline_followup_gate(fake_evaluations())
    assert result["passed"] and result["closed_loop_executed"] is False
    assert result["module_qualification"] is False


def test_one_seed_open_window_regression_stops_even_when_mean_improves():
    tasks = fake_evaluations()
    tasks["7"]["meta_query"]["metrics_by_model"][f"seed{diagnostic.SEEDS[-1]}__{diagnostic.ARMS[1]}"]["around_open_transition"]["component_mse"]["gripper"] = .81
    result = diagnostic.offline_followup_gate(tasks)
    assert not result["passed"] and result["decision"] == "stop_candidate_branch"


def test_single_seed_translation_regression_is_not_hidden_by_the_average():
    tasks = fake_evaluations()
    for index, seed in enumerate(diagnostic.SEEDS):
        tasks["4"]["meta_query"]["metrics_by_model"][f"seed{seed}__{diagnostic.ARMS[1]}"]["all"]["component_mse"]["translation"] = .098 if index < 2 else .106
    result = diagnostic.offline_followup_gate(tasks)["task_results"]["4"]
    assert result["checks"]["translation_mean_nonregression"] is True
    assert result["checks"]["translation_each_seed_nonregression"] is False


def test_occupied_gpu_is_rejected_before_training(monkeypatch):
    monkeypatch.setattr(diagnostic.subprocess, "check_output", lambda *args, **kwargs: "1, uuid, RTX4090, driver, 4096, 49140, 90")
    with pytest.raises(ValueError, match="GPU 1 is occupied"):
        diagnostic.check_gpu_available()


def test_initial_policy_reproduces_the_reference_hash_without_training():
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    _, _, policy, _, family, _ = diagnostic.source_query._load_protocol(diagnostic.ROOT / "configs/libero-runtime-native-task4-task7-support-query-hdf5-diagnostic_v1.json")
    model, _generator, _before, _context = diagnostic.make_policy(task=4, seed=20261022, policy=policy, family=family, device=torch.device("cpu"))
    assert diagnostic.causal._digest_tensors(model.state_dict()) == config["baseline_references"]["4"]["20261022"]["initial_model_sha256"]


def test_public_t0_archive_contains_no_initial_simulator_state(tmp_path):
    observations = {name: np.zeros(shape, dtype) for name, (shape, dtype) in diagnostic.expert.POLICY_OBSERVATION_SPECS.items()}
    dataset = SimpleNamespace(access=SimpleNamespace(demonstration_indices=(0,)), causal_initial_observations={0: observations})
    path = tmp_path / "public.npz"
    record = diagnostic.save_public_initials(dataset, path)
    assert record["observation_count"] == 1 and record["private_initial_state_values_recorded"] is False
    with np.load(path, allow_pickle=False) as saved:
        assert set(saved.files) == set(diagnostic.expert.core.OBS_KEYS) | {"demonstration_indices"}


def test_existing_artifacts_are_preserved_before_any_protocol_access(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("historical")
    monkeypatch.setattr(diagnostic, "load_protocol", lambda *args: pytest.fail("existing output reached protocol load"))
    with pytest.raises(ValueError, match="already exists"):
        diagnostic.run_diagnostic(tmp_path / "unused", output, tmp_path / "artifacts")
    assert output.read_text() == "historical"
