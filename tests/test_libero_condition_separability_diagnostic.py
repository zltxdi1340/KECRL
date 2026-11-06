import copy
import json

import numpy as np
import pytest
import torch

from experiments import analyze_libero_task4_task7_condition_separability as d


@pytest.mark.parametrize("section,field,value", [
    ("scope", "demonstration_role", "meta_query"), ("scope", "checkpoint_optimizer_updates", 1000),
    ("scope", "first_action_index", 0), ("scope", "new_optimizer_updates", 1),
    ("scope", "device", "cuda:0"), ("scope", "qualification_accessed", True),
    ("analysis", "neighbors", 1), ("analysis", "scaler", "include_query_demo"),
])
def test_frozen_scope_and_neighbor_rules_fail_before_payload_access(tmp_path, section, field, value):
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    config[section][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="scope or analysis"):
        d.load_protocol(path)


def test_query_or_non_source_never_reaches_role_resolver(monkeypatch):
    monkeypatch.setattr(d.expert, "authorize_role", lambda *a: pytest.fail("forbidden partition reached resolver"))
    for task, role in [(4, "meta_query"), (4, "qualification"), (2, "meta_support"), (5, "meta_support")]:
        with pytest.raises(ValueError, match="support-only"):
            d.authorize_support({}, {}, task, role)


def test_source_role_lock_is_kept():
    transfer = json.loads((d.ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json").read_text())
    manifest = json.loads((d.ROOT / "configs/libero_spatial_dataset_manifest_v1.json").read_text())
    assert d.authorize_support(transfer, manifest, 7).demonstration_indices == tuple(range(30))
    transfer["locked_access"]["before_method_freeze"].append("source_spi.meta_support")
    with pytest.raises(PermissionError, match="locked before"):
        d.authorize_support(transfer, manifest, 7)


def test_whole_query_demo_is_excluded_from_bank_and_scaler():
    features = np.concatenate((np.arange(5)[:, None], (100 + np.arange(5))[:, None])).astype(np.float32)
    demos = np.repeat([0, 1], 5)
    signs = np.repeat([-1, 1], 5).astype(np.int8)
    before = features.copy()
    result = d.leave_one_demo_knn(features, demos, signs)
    assert np.array_equal(features, before)
    assert np.all(demos[result["indices"]] != demos[:, None])
    assert result["folds"][0]["bank_demo_ids"] == [1]
    bank = features[5:].astype(np.float64)
    assert result["folds"][0]["bank_mean_sha256"] == d.core._digest_array(bank.mean(axis=0))
    assert result["folds"][0]["bank_scale_sha256"] == d.core._digest_array(bank.std(axis=0))
    assert result["majority_sign"][:5].tolist() == [1] * 5


def test_distance_ties_are_stable_and_never_use_same_demo_adjacent_frames():
    features = np.zeros((15, 2), np.float32)
    demos = np.repeat([0, 1, 2], 5)
    signs = np.ones(15, np.int8)
    result = d.leave_one_demo_knn(features, demos, signs)
    assert result["indices"][0].tolist() == list(range(5, 10))
    assert result["indices"][5].tolist() == list(range(5))
    assert np.array_equal(result["indices"], d.leave_one_demo_knn(features, demos, signs)["indices"])
    assert np.all(result["squared_distances"] == 0)


def test_query_labels_cannot_choose_its_neighbors_or_majority():
    features = np.arange(10, dtype=np.float32).reshape(-1, 1)
    demos = np.repeat([0, 1], 5)
    signs = np.ones(10, np.int8)
    first = d.leave_one_demo_knn(features, demos, signs)
    signs[:5] = -1
    second = d.leave_one_demo_knn(features, demos, signs)
    assert np.array_equal(first["indices"][:5], second["indices"][:5])
    assert np.array_equal(first["majority_sign"][:5], second["majority_sign"][:5])
    assert not np.array_equal(first["same_label_fraction"][:5], second["same_label_fraction"][:5])


def test_three_class_vote_tie_uses_nearest_among_tied_classes():
    neighbors = np.array([[0, 1, -1, 1, -1], [1, 0, -1, 0, -1]], np.int8)
    assert d.majority_labels(neighbors).tolist() == [1, 0]


def test_empty_stratum_is_null_not_perfect():
    values = np.ones(5)
    result = d.classifier_metrics(values, values, values, np.zeros(5, bool))
    assert result == {"rows": 0, "sign_error": None, "same_label_agreement": None}


def test_wrong_sign_saturation_has_strong_mse_gradient_suppression():
    prediction = np.zeros((1, 7), np.float32)
    prediction[0, 6] = .99
    target = np.zeros((1, 7), np.float32)
    target[0, 6] = -1
    metric = d.saturation_metrics(prediction, np.zeros_like(prediction), target, np.ones(1, bool))
    assert metric["wrong_sign_rows"] == metric["suppressed_wrong_rows"] == 1
    assert metric["suppressed_fraction_of_wrong"] == 1
    ratio = metric["wrong_mse_logit_gradient_abs_mean"] / metric["wrong_origin_matched_logit_bce_gradient_abs_mean"]
    assert ratio == pytest.approx(1 - float(prediction[0, 6]) ** 2)


def test_origin_matched_bce_derivative_matches_mse_at_zero_and_escapes_wrong_saturation():
    for z_value in (0., 10.):
        z = torch.tensor(z_value, dtype=torch.float64, requires_grad=True)
        mse = (torch.tanh(z) + 1).square()
        bce = 2 * torch.nn.functional.binary_cross_entropy_with_logits(2 * z, torch.zeros_like(z))
        g_mse = torch.autograd.grad(mse, z, retain_graph=True)[0]
        g_bce = torch.autograd.grad(bce, z)[0]
        if z_value == 0:
            assert torch.equal(g_mse, g_bce)
        else:
            assert g_mse < 1e-6 and g_bce > 3.99


def fake_models():
    result = {}
    for seed in d.SEEDS:
        result[str(seed)] = {"spaces": {name: {"around_open_transition": {"same_label_agreement": .8, "sign_error": .1}}
            for name in ("fusion", "projection")}, "policy": {"around_open_transition": {"gripper": {"sign_error_rate": .2}}},
            "saturation": {"around_open_transition": {"wrong_sign_rows": 30, "suppressed_fraction_of_wrong": .3}}}
    return result


def test_training_gate_requires_all_three_seeds_and_binary_target_contract():
    models = fake_models()
    assert d.classify_task(models, True)["eligible_for_separately_frozen_logit_loss_training"]
    assert not d.classify_task(models, False)["eligible_for_separately_frozen_logit_loss_training"]
    models[str(d.SEEDS[-1])]["saturation"]["around_open_transition"]["suppressed_fraction_of_wrong"] = .1
    assert not d.classify_task(models, True)["eligible_for_separately_frozen_logit_loss_training"]


def test_projection_condition_drop_blocks_a_readout_only_training_shortcut():
    models = fake_models()
    for model in models.values():
        model["spaces"]["fusion"]["around_open_transition"]["same_label_agreement"] = .95
    result = d.classify_task(models, True)
    assert result["consistent_projection_agreement_drop"]
    assert not result["eligible_for_separately_frozen_logit_loss_training"]


def test_actual_policy_hooks_preserve_actions_weights_inputs_and_are_removed():
    from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig
    model = LiberoBCPolicy(LiberoBCPolicyConfig(memory_mode="cnn_only")).eval()
    observations = {name: np.zeros((2, *shape), dtype=dtype) for name, (shape, dtype) in d.expert.POLICY_OBSERVATION_SPECS.items()}
    before = d.core._digest_weights(model.state_dict())
    captured = d.capture_features(model, observations)
    reference = d.expert.predict_shared_batch({"baseline": model}, observations)["baseline"]
    assert np.array_equal(captured["predictions"], reference)
    assert before == d.core._digest_weights(model.state_dict())
    assert captured["visual_concat"].shape == (2, 256)
    assert captured["projection"].shape == (2, 256)
    assert all(len(module._forward_hooks) == 0 for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())


def test_existing_artifact_guard_precedes_protocol_read(tmp_path, monkeypatch):
    output = tmp_path / "history.json"
    output.write_text("historical")
    monkeypatch.setattr(d, "load_protocol", lambda *a: pytest.fail("existing output reached protocol"))
    with pytest.raises(ValueError, match="already exists"):
        d.run_analysis(tmp_path / "unused", output, tmp_path / "archive")
    assert output.read_text() == "historical"
