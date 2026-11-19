"""Blank architectures and invented inputs only; no weights/data downloads."""
from copy import deepcopy
import pytest
import torch
from torch import nn

from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig, masked_behavior_cloning_loss
from src.algorithms.libero_frozen_resnet18_policy import LiberoFrozenResNet18Policy
from src.algorithms.libero_windowed_resnet18_policy import WindowedResNet18Policy
from src.algorithms.libero_view_specific_readout_policy_v4 import (
    ARMS, VIEWS, VIEW_SEED_OFFSETS, READOUT_PARAMETERS, TRAINABLE_PARAMETERS,
    ViewSpecificReadoutPolicyV4, ViewCoordinateReadoutV4, make_view_specific_readout_policy_v4,
)
from experiments import libero_task7_view_specific_readout_protocol_v4 as p

torch.set_num_threads(1)
CFG = LiberoBCPolicyConfig(memory_mode='cnn_only')


def invented(arm, seed=12):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone = nn.Sequential(nn.AdaptiveAvgPool2d((4, 4)), nn.Conv2d(3, 512, 1), nn.BatchNorm2d(512))
        initialized = WindowedResNet18Policy(LiberoFrozenResNet18Policy(CFG, backbone), window=1, seed=seed)
        return ViewSpecificReadoutPolicyV4(initialized, arm=arm, seed=seed)


def data():
    g = torch.Generator().manual_seed(77)
    return [torch.randint(256, (2, 3, 3, 128, 128), dtype=torch.uint8, generator=g),
            torch.randint(256, (2, 3, 3, 128, 128), dtype=torch.uint8, generator=g),
            torch.randn(2, 3, 7, generator=g), torch.randn(2, 3, 2, generator=g)]


@pytest.mark.parametrize('seed', p.SEEDS)
def test_blank_full_architecture_exact_counts_common_initialization_and_independent_view_seeds(seed, monkeypatch):
    def forbidden(*args, **kwargs): pytest.fail('source weights must not be loaded')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    before = torch.get_rng_state().clone()
    control, candidate = [make_view_specific_readout_policy_v4(CFG, arm=arm, seed=seed) for arm in ARMS]
    assert torch.equal(before, torch.get_rng_state())
    for arm, model in zip(ARMS, [control, candidate]):
        assert sum(v.numel() for v in model.parameters() if v.requires_grad) == TRAINABLE_PARAMETERS[arm]
        assert sum(v.numel() for v in model.visual_encoder.projection.parameters()) == READOUT_PARAMETERS[arm]
        assert sum(v.numel() for v in model.parameters()) == 11176512+TRAINABLE_PARAMETERS[arm]
        assert model.fusion[0].in_features == 268
    common = lambda m: {k: v for k, v in m.state_dict().items() if not k.startswith('visual_encoder.projection.')}
    left, right = common(control), common(candidate)
    assert left.keys() == right.keys() and all(torch.equal(v, right[k]) for k, v in left.items())
    branches = candidate.visual_encoder.projection
    assert set(branches) == set(VIEWS)
    assert not ({id(v) for v in branches[VIEWS[0]].parameters()} & {id(v) for v in branches[VIEWS[1]].parameters()})
    assert not torch.equal(branches[VIEWS[0]].channel_projection.weight, branches[VIEWS[1]].channel_projection.weight)
    for view in VIEWS:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+VIEW_SEED_OFFSETS[view]); expected = ViewCoordinateReadoutV4()
        assert sum(v.numel() for v in branches[view].parameters()) == 520928
        assert all(torch.equal(v, expected.state_dict()[k]) for k, v in branches[view].state_dict().items())
        assert branches[view].coordinate_mlp[0].out_features == 608
        assert isinstance(branches[view].coordinate_mlp[-1], nn.Linear)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('arm', ARMS)
def test_pixel_cache_gradient_parity_frozen_backbone_and_xyz_zero(arm):
    model = invented(arm); cached = deepcopy(model); batch = data(); b, t = batch[0].shape[:2]
    before = deepcopy(model.visual_encoder.backbone.state_dict())
    features = [model.visual_encoder.encode_frozen(x.reshape(-1, 3, 128, 128)).reshape(b, t, 512, 4, 4) for x in batch[:2]]
    model.train(); cached.train()
    direct = model.forward_sequence(*batch)[0]; value = cached.forward_feature_sequence(*features, *batch[2:])[0]
    assert torch.equal(direct, value)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    for m, output in [(model, direct), (cached, value)]:
        masked_behavior_cloning_loss(output, torch.zeros_like(output), mask).backward()
        assert not m.visual_encoder.backbone.training
        assert not torch.count_nonzero(m.fusion[0].weight.grad[:, -3:])
        if arm == ARMS[1]:
            for view in VIEWS:
                gradient = m.visual_encoder.projection[view].channel_projection.weight.grad
                assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
            summary = m.entropy_summary(mask); p.audit_entropy_summary(summary, 3)
            assert summary['valid_rows'] == 3 and summary['heatmaps_per_view'] == 768
            assert all(not v.requires_grad and v.grad_fn is None for v in m._entropy_views)
        else: assert m.entropy_summary(mask) is None
    for (name, v), (other, w) in zip(model.named_parameters(), cached.named_parameters()):
        assert name == other
        if v.requires_grad: assert torch.allclose(v.grad, w.grad, atol=1e-9, rtol=1e-5)
        else: assert v.grad is None and w.grad is None
    assert all(torch.equal(v, before[k]) for k, v in model.visual_encoder.backbone.state_dict().items())
    with pytest.raises(TypeError): model.forward_sequence(*batch, eef_pos=torch.zeros(2, 3, 3))


def test_each_view_routes_to_its_branch_no_extra_entropy_forward_and_failure_is_not_stale():
    m = invented(ARMS[1]); seen = []
    def guard(key, fn, *args):
        seen.append((key, id(fn))); return fn(*args)
    m.compute_guard = guard
    g = torch.Generator().manual_seed(41)
    args = [torch.randn(1, 3, 512, 4, 4, generator=g) for _ in range(2)] + [torch.randn(1, 3, 7, generator=g), torch.randn(1, 3, 2, generator=g)]
    m.forward_feature_sequence(*args)
    assert seen == [('spatial_readout_'+view, id(m.visual_encoder.projection[view])) for view in VIEWS]
    first = m.entropy_summary(); assert m.entropy_summary() == first and len(seen) == 2
    mask = torch.tensor([[1, 0, 1]], dtype=torch.bool)
    record = m.entropy_summary(mask); p.audit_entropy_summary(record, 2)
    for index, label in enumerate(['agent', 'hand']):
        mean = float(m._entropy_views[index][[0, 2]].double().mean())
        assert record[label+'_entropy_nats_mean'] == pytest.approx(mean)
    def reject(key, fn, *args):
        if key.endswith('eye_in_hand'): raise RuntimeError('invented budget refusal')
        return fn(*args)
    m.compute_guard = reject
    with pytest.raises(RuntimeError): m.forward_feature_sequence(*args)
    with pytest.raises(ValueError): m.entropy_summary()


def test_independent_branches_do_not_alias_and_retained_entropy_bound_is_four_copies():
    m = invented(ARMS[1]); initial = {k: v.clone() for k, v in m.state_dict().items()}
    features = torch.zeros(1, 80, 512, 4, 4)
    m.forward_feature_sequence(features, features, torch.zeros(1, 80, 7), torch.zeros(1, 80, 2))
    branches = m.visual_encoder.projection
    retained = sum(branches[v]._last_entropy.numel() for v in VIEWS) + sum(v.numel() for v in m._entropy_views)
    assert retained == p.ENTROPY['max_retained_entropy_scalars'] == 81920
    assert all(torch.equal(v, initial[k]) for k, v in m.state_dict().items())
    old_hand = branches['eye_in_hand'].channel_projection.weight.clone()
    with torch.no_grad(): branches['agentview'].channel_projection.weight.add_(1)
    assert torch.equal(old_hand, branches['eye_in_hand'].channel_projection.weight)


def test_six_view_initialization_seeds_are_distinct():
    assert len({seed+offset for seed in p.SEEDS for offset in VIEW_SEED_OFFSETS.values()}) == 6
