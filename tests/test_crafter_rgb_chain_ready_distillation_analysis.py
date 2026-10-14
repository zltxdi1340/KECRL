import copy

import numpy as np
import pytest
import torch

from experiments.analyze_crafter_wood3_rgb_chain_ready_distillation import (
    _check_selection, _reconstruct_split, _same_search,
)
from experiments.crafter_natural_readout import fit_natural_head
from experiments.run_crafter_wood3_rgb_chain_ready_distillation import _fit_head


def _artifact():
    return {
        "trials": [{"learning_rate": .003, "history": [
            {"epoch": 0, "validation_macro_accuracy": .3, "validation_macro_set_nll": 2.},
            {"epoch": 50, "validation_macro_accuracy": .4, "validation_macro_set_nll": 3.},
            {"epoch": 100, "validation_macro_accuracy": .4, "validation_macro_set_nll": 2.},
        ]}],
        "selected_epoch": 100, "selected_learning_rate": .003, "selection_uses_heldout": False,
        "standardizer_mean": [[0.]], "standardizer_std": [[1.]],
        "train_category_counts": {"ready": 1, "turn": 1, "approach": 1},
        "source_checkpoint_sha256": "checkpoint", "source_pilot_training_rows_digest": "rows",
        "training_rgb_digest": "train", "validation_rgb_digest": "validation", "heldout_rgb_digest": "heldout",
        "supervised_optimizer_updates": 100, "distillation_weight": 0., "distillation_categories": [],
    }


def _config():
    return {"actor_learning_rates": [.003], "actor_fit_epochs": 100, "validation_interval": 50,
            "ready_distillation_weight": 1., "states_per_category_per_episode": 2}


def test_selection_audit_uses_category_accuracy_then_nll_with_stable_ties():
    artifact = _artifact()
    _check_selection(artifact, _config())
    artifact["selected_epoch"] = 50
    with pytest.raises(ValueError, match="validation criterion"):
        _check_selection(artifact, _config())
    artifact["trials"][0]["history"][1]["validation_macro_set_nll"] = 2.
    _check_selection(artifact, _config())
    artifact["selection_uses_heldout"] = True
    with pytest.raises(ValueError, match="validation criterion"):
        _check_selection(artifact, _config())


def test_distillation_comparison_rejects_different_data_or_budgets():
    plain = _artifact()
    distilled = {**copy.deepcopy(plain), "distillation_weight": 1., "distillation_categories": ["ready"]}
    _same_search(plain, distilled, _config())
    distilled["training_rgb_digest"] = "different"
    with pytest.raises(ValueError, match="training_rgb_digest"):
        _same_search(plain, distilled, _config())
    distilled["training_rgb_digest"] = plain["training_rgb_digest"]
    distilled["supervised_optimizer_updates"] = 101
    with pytest.raises(ValueError, match="supervised_optimizer_updates"):
        _same_search(plain, distilled, _config())


def test_split_reconstruction_preserves_first_distinct_geometry_rows_and_death():
    episode = {"episode": 0, "environment_seed": 10, "action_seed": 11, "success": False, "terminal_reason": "death"}
    labels = [{"category": "turn", "rgb_digest": name, "step": i + 1,
               "valid_actions": [2, 4], "before": {"health": 1}, "trajectory_action": 0}
              for i, name in enumerate(("old", "a", "a", "b", "c"))]
    images = np.arange(5).reshape(5, 1, 1, 1)
    records, selected = _reconstruct_split(images, labels, episode, "validation", {"old"}, set(), _config())
    assert [row["step"] for row in records] == [2, 4]
    assert all(row["target_actions"] == [2, 4] for row in records)
    assert [int(image.item()) for image in selected] == [1, 3]


def test_plain_arm_preserves_existing_action_set_fit_behavior():
    class FrozenSource(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.recurrent = False
            self.actor = torch.nn.Linear(4, 17)
            self.optimizer = torch.optim.Adam(self.parameters(), lr=.001)
            for parameter in self.parameters():
                parameter.requires_grad_(False)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1)
        source = FrozenSource()
    features = np.random.RandomState(3).normal(size=(12, 4)).astype(np.float32)
    categories = np.asarray([0] * 3 + [1] * 4 + [2] * 5)
    masks = np.zeros((12, 7), dtype=bool)
    masks[:3, 5] = True
    masks[3:7, [1, 4]] = True
    masks[7:, 2] = True
    config = {"actor_learning_rates": [.003, .03], "actor_fit_epochs": 10,
              "validation_interval": 5, "actor_weight_decay": 0.,
              "feature_std_floor": 1e-4, "ready_distillation_weight": 1.}
    with torch.no_grad():
        teacher = torch.softmax(source.actor(torch.as_tensor(features))[:, :7], dim=1).numpy()
    _, existing = fit_natural_head(source, features, masks, categories, features, masks, categories,
                                   config, "standardized_linear")
    plain = _fit_head(source, features, masks, categories, teacher, features, masks, categories,
                      config, "action_set")
    assert (plain["selected_epoch"], plain["selected_learning_rate"]) == (existing["selected_epoch"], existing["selected_learning_rate"])
    assert plain["supervised_optimizer_updates"] == existing["supervised_optimizer_updates"]
    for key in plain["head_state"]:
        assert torch.allclose(plain["head_state"][key], existing["head_state"][key], atol=1e-6, rtol=1e-6)
