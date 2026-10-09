import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("crafter")

from experiments.crafter_actor_head_ablation import (
    ActorMLP, fit_head_variant, fold_standardized_linear, train_standardizer, install_head_copy,
)
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head, _validate
from experiments.crafter_actor_only_local import fixture_action_targets
from experiments.crafter_spatial_readout_data import scene_labels, scene_manifest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


def _config():
    return json.loads(Path("configs/crafter_wood3_actor_head_ablation_cuda_v1.yaml").read_text())


def test_train_standardizer_uses_only_training_values_and_has_floor():
    values = np.array([[0., 5., 2.], [2., 5., 2.]], dtype=np.float32)
    mean, std = train_standardizer(values, .1)
    assert np.array_equal(mean, [[1., 5., 2.]])
    assert np.allclose(std, [[1., .1, .1]])


def test_standardized_linear_fold_is_logit_equivalent_and_keeps_masked_rows():
    rng = np.random.default_rng(4)
    mean = rng.normal(size=(1, 128)).astype(np.float32)
    std = np.exp(rng.normal(size=(1, 128))).astype(np.float32)
    state = {"weight": torch.randn(17, 128), "bias": torch.randn(17)}
    source = {"weight": torch.randn(17, 128), "bias": torch.randn(17)}
    state["weight"][7:] = source["weight"][7:]
    state["bias"][7:] = source["bias"][7:]
    folded = fold_standardized_linear(state, mean, std, source)
    features = torch.randn(19, 128)
    standard = (features - torch.as_tensor(mean)) / torch.as_tensor(std)
    raw_logits = features @ folded["weight"].T + folded["bias"]
    standardized_logits = standard @ state["weight"].T + state["bias"]
    assert torch.allclose(raw_logits[:, :7], standardized_logits[:, :7], atol=2e-5, rtol=2e-5)
    assert torch.equal(folded["weight"][7:], source["weight"][7:])
    assert torch.equal(folded["bias"][7:], source["bias"][7:])


def test_head_fit_variants_are_source_immutable_and_mlp_is_diagnostic_only():
    config = {**_config(), "actor_fit_epochs": 20, "validation_interval": 10,
              "actor_learning_rates": [.03], "mlp_init_seed": 13}
    source = build_matched_spatial_policy(config["policy"], "cnn_only")
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    labels = np.asarray([scene_labels(scene) for scene in scene_manifest()], dtype=np.int64)
    targets = fixture_action_targets(labels)
    features = np.pad(np.eye(6, dtype=np.float32)[targets], ((0, 0), (0, 122)))
    for variant in ("raw_linear", "standardized_linear", "mlp64"):
        fitted, artifact = fit_head_variant(source, features, targets, features, targets, config, "cpu", variant)
        assert artifact["variant"] == variant
        assert artifact["selection_uses_heldout"] is False
        assert artifact["source_policy_and_optimizer_unchanged"] is True
        assert _state_equal(before, (source.state_dict(), source.optimizer.state_dict()))
        assert all(parameter.grad is None and not parameter.requires_grad for parameter in fitted.parameters())
        if variant == "mlp64":
            assert isinstance(fitted, ActorMLP)
            assert artifact["masked_out_action_rows_unchanged"] is None
        else:
            assert isinstance(fitted, torch.nn.Linear)
            assert artifact["masked_out_action_rows_unchanged"] is True


def test_mlp_initialization_isolated_from_global_rng():
    config = {**_config(), "actor_fit_epochs": 1, "validation_interval": 1,
              "actor_learning_rates": [.03], "mlp_init_seed": 7}
    source = build_matched_spatial_policy(config["policy"], "cnn_only")
    labels = np.asarray([scene_labels(scene) for scene in scene_manifest()], dtype=np.int64)
    targets = fixture_action_targets(labels)
    features = np.zeros((len(targets), 128), dtype=np.float32)
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    fit_head_variant(source, features, targets, features, targets, config, "cpu", "mlp64")
    assert torch.equal(before, torch.random.get_rng_state())


def test_saved_diagnostic_mlp_reload_preserves_non_actor_state_optimizer_and_rng():
    config = _config()
    source = build_matched_spatial_policy(config['policy'], 'cnn_only')
    sum(parameter.square().sum() for parameter in source.parameters()).backward()
    source.optimizer.step()
    source.optimizer.zero_grad(set_to_none=True)
    before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
    head = ActorMLP()
    state = copy.deepcopy(head.state_dict())
    artifact = {'variant': 'mlp64', 'head_state': state,
                'head_state_canonical_digest': array_digest({key: value.numpy() for key, value in state.items()})}
    rng = torch.random.get_rng_state().clone()
    policy = load_saved_head(source, artifact)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert isinstance(policy.actor, ActorMLP)
    assert _state_equal(before, (source.state_dict(), source.optimizer.state_dict()))
    assert _state_equal(policy.optimizer.state_dict(), source.optimizer.state_dict())
    assert all(torch.equal(value, policy.state_dict()[key]) for key, value in source.state_dict().items() if not key.startswith('actor.'))
    assert all(p.grad is None and not p.requires_grad for p in policy.parameters())
    with pytest.raises(RuntimeError, match='digest'):
        load_saved_head(source, {**artifact, 'head_state_canonical_digest': 'bad'})


def test_standardized_fit_normalizer_and_selection_ignore_heldout_values():
    from experiments.crafter_actor_only_local import evaluate_actor
    config = {**_config(), 'actor_fit_epochs': 30, 'validation_interval': 10, 'actor_learning_rates': [.03]}
    source = build_matched_spatial_policy(config['policy'], 'cnn_only')
    labels = np.asarray([scene_labels(scene) for scene in scene_manifest()], dtype=np.int64)
    targets = fixture_action_targets(labels)
    features = np.pad(np.eye(6, dtype=np.float32)[targets] * .02 + .5, ((0, 0), (0, 122)))
    head, artifact = fit_head_variant(source, features, targets, features, targets, config, 'cpu', 'standardized_linear')
    mean, std = train_standardizer(features, config['feature_std_floor'])
    assert np.array_equal(mean, np.asarray(artifact['standardizer_mean'], dtype=np.float32))
    assert np.array_equal(std, np.asarray(artifact['standardizer_std'], dtype=np.float32))
    policy = install_head_copy(source, head)
    state = copy.deepcopy(policy.state_dict())
    selected = (artifact['selected_epoch'], artifact['head_state_canonical_digest'])
    clean = evaluate_actor(policy, features, labels, np.zeros(len(labels), dtype=np.int64), 'cpu')
    changed = evaluate_actor(policy, np.zeros_like(features), labels, np.zeros(len(labels), dtype=np.int64), 'cpu')
    assert clean['balanced_accuracy'] > changed['balanced_accuracy']
    assert _state_equal(state, policy.state_dict())
    assert selected == (artifact['selected_epoch'], artifact['head_state_canonical_digest'])


def test_fresh_evaluation_refuses_fit_selection_or_previous_background_seed():
    import os
    from unittest.mock import patch
    config = json.loads(Path('configs/crafter_wood3_actor_head_fresh_cuda_v1.yaml').read_text())
    with patch.dict(os.environ, {'PYTHONHASHSEED': '0'}):
        _validate(config)
        for key, value in [('new_fit_or_selection_allowed', True), ('evaluation_only', False),
                           ('policy_updates_allowed', True), ('background_seed_base', 41000000),
                           ('evaluation_modes', ['sample']), ('variants', ['standardized_linear'])]:
            with pytest.raises(ValueError):
                _validate({**config, key: value})
