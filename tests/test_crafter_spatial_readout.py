import copy
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("crafter")

from experiments.crafter_controlled_collection_scene import CollectionScene
from experiments.crafter_spatial_readout_data import (
    TARGETS, BackgroundCollectionEnv, array_digest, assert_group_boundary,
    local_rgb_features, scene_labels, scene_manifest, split_manifest,
)
from experiments.crafter_spatial_readouts import (
    classification_metrics, compose_predictions, evaluate_readout,
    fit_readout, standardizer,
)
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_spatial_readout import _actor_metrics, _validate
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot


def test_manifest_is_balanced_and_deduplicates_absent_tree_directions():
    scenes = scene_manifest()
    assert len(scenes) == 60
    labels = np.asarray([scene_labels(scene) for scene in scenes])
    assert np.bincount(labels[:, 0]).tolist() == [12] * 5
    assert np.bincount(labels[:, 1]).tolist() == [15] * 4
    assert np.bincount(labels[:, 2]).tolist() == [12, 12, 36]
    assert np.bincount(labels[:, 3]).tolist() == [12, 12, 9, 9, 9, 9]
    absent = [scene for scene in scenes if not scene.tree_present]
    assert len({(scene.initial_wood, scene.facing_action) for scene in absent}) == 12
    composed = compose_predictions(labels[:, 0], labels[:, 1])
    assert np.array_equal(composed['joint_state'], labels[:, 2])
    assert np.array_equal(composed['local_decision'], labels[:, 3])


def test_natural_background_fixtures_keep_all_variants_paired_and_native_rules():
    scene = CollectionScene(1, 3, 1, True)
    native = BackgroundCollectionEnv(scene, seed=0)
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    try:
        with pytest.raises(RuntimeError, match='set_background'):
            adapter.reset()
        background_images = []
        for seed in (31000000, 31000001):
            native.set_background(seed)
            native.configure(scene, seed=seed)
            present = adapter.reset().copy()
            materials = native._world._mat_map.copy()
            rng = copy.deepcopy(native._world.random.get_state())
            snapshot = collection_snapshot(adapter)
            assert snapshot['unblocked_tree_move_actions'] == [3]
            assert not snapshot['ready_to_collect']
            assert all(snapshot[key] == 9 for key in ('health', 'food', 'drink', 'energy'))
            assert len(native._world.objects) == 1
            assert 'position' not in adapter.state()
            adapter.step(3)
            _, _, done, info = adapter.step(5)
            assert not done and info['inventory']['wood'] == 2
            native.configure(replace(scene, tree_present=False), seed=seed)
            absent = adapter.reset().copy()
            assert np.count_nonzero(materials != native._world._mat_map) == 1
            assert _state_equal(rng, native._world.random.get_state())
            assert not collection_snapshot(adapter)['unblocked_tree_move_actions']
            features = local_rgb_features(np.stack([present, absent]))
            assert features.shape == (2, 1323)
            assert not np.array_equal(features[0], features[1])
            background_images.append(absent)
        assert not np.array_equal(*background_images)
        native.set_background(31000000)
        native.configure(replace(scene, tree_present=False), seed=31000000)
        assert np.array_equal(background_images[0], adapter.reset())
        with pytest.raises(ValueError, match='background seed'):
            native.configure(scene, seed=19)
    finally:
        adapter.close()


def _record(background, split, source, image):
    return {'background_id': background, 'split': split, 'background_rgb_sha256': source, 'rgb_sha256': image}


def test_split_boundary_rejects_paired_variants_background_duplicates_and_rgb_leakage():
    good = [_record(0, 'train', 'bg0', 'a'), _record(0, 'train', 'bg0', 'b'), _record(1, 'heldout', 'bg1', 'c')]
    assert assert_group_boundary(good)['backgrounds'] == 2
    for bad in ([_record(0, 'heldout', 'bg0', 'z')], [_record(2, 'validation', 'bg0', 'z')],
                [_record(2, 'validation', 'bg2', 'a')]):
        with pytest.raises(ValueError):
            assert_group_boundary(good + bad)
    manifest = split_manifest({'background_counts': {'train': 2, 'validation': 1, 'heldout': 1}, 'background_seed_base': 123})
    assert [row['split'] for row in manifest] == ['train', 'train', 'validation', 'heldout']
    assert [row['environment_seed'] for row in manifest] == [123, 124, 125, 126]


def test_balanced_accuracy_penalizes_majority_joint_state_prediction():
    labels = np.array([0] * 12 + [1] * 12 + [2] * 36)
    metrics = classification_metrics(np.full(60, 2), labels, 3)
    assert metrics['accuracy'] == .6
    assert metrics['balanced_accuracy'] == pytest.approx(1 / 3)
    assert metrics['confusion_matrix'] == [[0, 0, 12], [0, 0, 12], [0, 0, 36]]


def test_train_only_normalizer_does_not_include_other_distributions():
    train = torch.tensor([[0., 7.], [2., 7.]])
    mean, std = standardizer(train, .0001)
    assert torch.equal(mean, torch.tensor([[1., 7.]]))
    assert torch.allclose(std, torch.tensor([[1., .0001]]))


def test_readout_selection_and_state_are_independent_of_heldout_evaluation():
    labels = np.asarray([scene_labels(scene) for scene in scene_manifest()], dtype=np.int64)
    features = np.concatenate([np.eye(classes, dtype=np.float32)[labels[:, index]]
                               for index, classes in enumerate(TARGETS.values())], axis=1)
    config = {'feature_std_floor': .0001, 'readout_weight_decays': [.0001, .01],
              'readout_learning_rate': .03, 'readout_epochs': 40, 'validation_interval': 10,
              'label_shuffle_seed': 37}
    rng = torch.get_rng_state().clone()
    model, artifact = fit_readout(features, labels, features, labels, config, 'linear', 0, 'cpu')
    assert torch.equal(rng, torch.get_rng_state())
    before = copy.deepcopy(model.state_dict())
    selected = (artifact['epoch'], artifact['weight_decay'], artifact['state_and_normalizer_canonical_digest'])
    clean = evaluate_readout(model, artifact, features, labels, 'cpu')
    changed = evaluate_readout(model, artifact, np.zeros_like(features), labels, 'cpu')
    assert clean['mean_balanced_accuracy'] > .95
    assert changed['mean_balanced_accuracy'] < clean['mean_balanced_accuracy']
    assert _state_equal(before, model.state_dict())
    assert selected == (artifact['epoch'], artifact['weight_decay'], artifact['state_and_normalizer_canonical_digest'])
    assert np.array_equal(artifact['mean'].numpy(), features.mean(axis=0, keepdims=True))
    assert artifact['offline_optimizer_updates'] == 80


def test_actor_metrics_exclude_no_tree_from_claims_of_optimal_action():
    labels = np.asarray([scene_labels(scene) for scene in scene_manifest()], dtype=np.int64)
    probs = np.full((len(labels), 7), .01, dtype=np.float32)
    probs[:, 5] = .94
    metrics = _actor_metrics(probs, labels, np.full(len(labels), True))
    assert metrics['greedy_ready_do_rate'] == 1
    assert metrics['greedy_unaligned_target_move_rate'] == 0
    assert metrics['greedy_local_decision_accuracy_tree_present'] == .25
    assert metrics['greedy_local_decision_balanced_accuracy_tree_present'] == .2
    assert metrics['no_tree_is_excluded_from_optimal_action_metrics']


def test_dataset_digest_covers_dtype_shape_and_content():
    value = np.arange(4, dtype=np.int64)
    digest = array_digest({'labels': value})
    assert digest != array_digest({'labels': value.astype(np.float32)})
    assert digest != array_digest({'labels': value.reshape(2, 2)})
    assert digest != array_digest({'labels': value + 1})


def test_protocol_rejects_policy_supervision_split_changes_and_previous_seed_reuse(monkeypatch):
    monkeypatch.setenv('PYTHONHASHSEED', '0')
    config = json.loads(Path('configs/crafter_wood3_spatial_readout_cuda_v1.yaml').read_text())
    arms = {'baseline': json.loads(Path('configs/crafter_wood3_spatial_representation_deterministic_v3_cuda_pilot_v1.yaml').read_text()),
            'auxiliary': json.loads(Path('configs/crafter_wood3_do_wood_gain_auxiliary_cuda_pilot_v1.yaml').read_text())}
    previous = [json.loads(Path(path).read_text()) for path in config['previous_diagnostic_configs']]
    _validate(config, arms, previous)
    for key, value in (('policy_teacher_used', True), ('readout_oracle_supervision_used', False),
                       ('background_counts', {'train': 40, 'validation': 4, 'heldout': 4}),
                       ('background_seed_base', previous[-1]['diagnostic_seed_base'])):
        with pytest.raises(ValueError):
            _validate({**config, key: value}, arms, previous)
