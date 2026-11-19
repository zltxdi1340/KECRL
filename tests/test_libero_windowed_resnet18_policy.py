"""Invented tiny backbones and pixels; no downloads/research checkpoints."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
from src.algorithms.libero_frozen_resnet18_policy import LiberoFrozenResNet18Policy
from src.algorithms.libero_windowed_resnet18_policy import WindowedResNet18Policy


def tiny(window,seed=12):
    torch.manual_seed(seed)
    config=LiberoBCPolicyConfig(cnn_channels=(2,2,2),visual_embedding_dim=2,fusion_dim=4,recurrent_hidden_dim=4,memory_mode='cnn_only')
    backbone=nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Conv2d(3,512,1))
    return WindowedResNet18Policy(LiberoFrozenResNet18Policy(config,backbone),window=window,seed=seed)


def inputs(steps=13):
    generator=torch.Generator().manual_seed(44)
    return [torch.randint(0,256,(1,steps,3,128,128),generator=generator,dtype=torch.uint8),
        torch.randint(0,256,(1,steps,3,128,128),generator=generator,dtype=torch.uint8),
        torch.randn(1,steps,7,generator=generator),torch.randn(1,steps,2,generator=generator)]


@pytest.mark.parametrize('window',[1,10])
def test_bounded_online_output_equals_explicit_recent_window_and_episode_reset(window):
    model=tiny(window).eval();data=inputs();context=None
    with torch.no_grad():
        for i in range(13):
            action,context,rows=model.forward_window_step(*[v[:,i] for v in data],context)
            start=max(0,i-window+1);expected,_=model.forward_sequence(*[v[:,start:i+1] for v in data])
            assert torch.allclose(action,expected[:,-1],atol=1e-6,rtol=1e-6) and rows==min(i+1,window)
            if context is not None:assert context.shape[1]<=9 and not context.requires_grad
        first,_,_=model.forward_window_step(*[v[:,0] for v in data],None)
        expected,_=model.forward_sequence(*[v[:,:1] for v in data]);assert torch.equal(first,expected[:,0])


def test_old_observation_outside_window_cannot_change_action_and_future_is_causal():
    model=tiny(10).eval();a=inputs();b=[v.clone() for v in a];b[0][:,:3]=255-b[0][:,:3];b[2][:,:3]+=10
    def last(data):
        context=None
        for i in range(13):out,context,_=model.forward_window_step(*[v[:,i] for v in data],context)
        return out
    with torch.no_grad():
        assert torch.equal(last(a),last(b))
        original,_=model.forward_sequence(*a);changed=[v.clone() for v in a];changed[2][:,6:]+=100
        updated,_=model.forward_sequence(*changed);assert torch.equal(original[:,:6],updated[:,:6])


def test_only_memory_module_differs_at_initialization_and_cached_gradient_parity():
    a,b=tiny(1),tiny(10)
    for key,value in a.state_dict().items():
        if not key.startswith('recurrent.'):assert torch.equal(value,b.state_dict()[key])
    data=[v[:,:4] for v in inputs()];model=b
    agent=model.visual_encoder.encode_frozen(data[0].reshape(-1,3,128,128)).reshape(1,4,512,4,4)
    hand=model.visual_encoder.encode_frozen(data[1].reshape(-1,3,128,128)).reshape(1,4,512,4,4)
    clone=deepcopy(model);direct,_=model.forward_sequence(*data);cached,_=clone.forward_feature_sequence(agent,hand,data[2],data[3])
    assert torch.equal(direct,cached);direct.square().mean().backward();cached.square().mean().backward()
    for (key,value),(other_key,other) in zip(model.named_parameters(),clone.named_parameters()):
        assert key==other_key
        if value.requires_grad:assert torch.equal(value.grad,other.grad)
        else:assert value.grad is None and other.grad is None


@pytest.mark.parametrize('kind',['long','requires_grad','nan','wrong_shape'])
def test_invalid_or_leaky_context_rejected(kind):
    model=tiny(10);data=inputs(1);context=torch.zeros(1,3,4)
    if kind=='long':context=torch.zeros(1,10,4)
    if kind=='requires_grad':context.requires_grad_(True)
    if kind=='nan':context[0,0,0]=float('nan')
    if kind=='wrong_shape':context=torch.zeros(1,2,5)
    with pytest.raises(ValueError):model.forward_window_step(*[v[:,0] for v in data],context)
