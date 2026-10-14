from pathlib import Path

import pytest
import torch

from experiments.run_crafter_wood3_collection_opportunity import _read
from experiments.run_crafter_wood3_rgb_chain_ready_distillation import (
    _assert_new_split_boundary,
    _loss_components,
    _validate,
)


def _config():
    return _read(Path("configs/crafter_wood3_rgb_chain_ready_distillation_cuda_pilot_v1.yaml"))


def test_ready_distillation_adds_gradient_only_on_ready_rows():
    logits = torch.tensor([[0.0, 0.2, 0.1, -0.2, 0.0, 0.5, 0.0],
                           [0.0, 0.2, 0.1, -0.2, 0.0, 0.5, 0.0],
                           [0.0, 0.2, 0.1, -0.2, 0.0, 0.5, 0.0]], requires_grad=True)
    masks = torch.tensor([[False, False, False, False, False, True, False],
                          [False, True, True, False, False, False, False],
                          [True, False, False, False, False, False, False]])
    categories = torch.tensor([0, 1, 2])
    teacher = torch.full((3, 7), 1 / 7)
    teacher[0] = torch.tensor([0.05, 0.05, 0.05, 0.05, 0.05, 0.70, 0.05])
    total, set_loss, kl_loss = _loss_components(logits, masks, categories, teacher, 1.0)
    assert total.item() == pytest.approx(set_loss.item() + kl_loss.item())
    total.backward()
    distilled_gradient = logits.grad.detach().clone()
    logits_baseline = logits.detach().clone().requires_grad_(True)
    baseline, _, baseline_kl = _loss_components(logits_baseline, masks, categories, teacher, 0.0)
    baseline.backward()
    assert baseline_kl.item() > 0
    assert torch.allclose(distilled_gradient[1:], logits_baseline.grad[1:])
    assert not torch.allclose(distilled_gradient[0], logits_baseline.grad[0])


def test_ready_distillation_contract_locks_teacher_scope_and_training_budget(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    config = _config()
    _validate(config)
    for key, value in (("ready_distillation_weight", 0.5), ("death_episodes_excluded", True),
                       ("fresh_action_oracle_allowed", True), ("heldout_used_for_fit_or_selection", True)):
        changed = {**config, key: value}
        with pytest.raises(ValueError, match=key):
            _validate(changed)


def test_fresh_episode_seed_cannot_cross_supervision_boundary():
    records = [
        {"split": "train", "rgb_sha256": "train-rgb", "environment_seed": 1, "action_seed": 11},
        {"split": "validation", "rgb_sha256": "validation-rgb", "environment_seed": 2, "action_seed": 12},
        {"split": "heldout", "rgb_sha256": "heldout-rgb", "environment_seed": 3, "action_seed": 13},
    ]
    _assert_new_split_boundary(records, {4}, {14})
    records[-1]["action_seed"] = 14
    with pytest.raises(ValueError, match="fresh evaluation seeds"):
        _assert_new_split_boundary(records, {4}, {14})
