"""Invented pixels, XYZ and tiny frozen backbones; no research payloads."""
from copy import deepcopy
import pytest
import torch
from src.algorithms.libero_cached_eef_policy import CachedEEFPolicy, ARMS
from test_libero_windowed_resnet18_policy import tiny as base_tiny, inputs as old_inputs


def tiny(arm,seed=12):
    return CachedEEFPolicy(base_tiny(1,seed),arm=arm,seed=seed)


def inputs(steps=3):
    return [*old_inputs(steps),torch.tensor([.1,-.2,.9]).expand(1,steps,3).clone()]


def test_pair_has_identical_full_initial_state_and_parameters_but_only_candidate_uses_xyz():
    a,b=[tiny(arm).eval() for arm in ARMS]
    assert a.fusion[0].in_features==16  # tiny visual2*2 + joint7 + gripper2 + XYZ3
    assert sum(v.numel() for v in a.parameters())==sum(v.numel() for v in b.parameters())
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    x=inputs();zero=[*x[:-1],torch.zeros_like(x[-1])];changed=[*x[:-1],x[-1]+.4]
    with torch.no_grad():
        assert torch.equal(a.forward_sequence(*x)[0],a.forward_sequence(*changed)[0])
        assert torch.equal(a.forward_sequence(*x)[0],b.forward_sequence(*zero)[0])
        assert not torch.equal(b.forward_sequence(*x)[0],b.forward_sequence(*changed)[0])


@pytest.mark.parametrize('arm',ARMS)
def test_cached_direct_gradient_parity_and_frozen_bn(arm):
    m=tiny(arm);c=deepcopy(m);data=inputs();b,t=data[0].shape[:2]
    features=[m.visual_encoder.encode_frozen(x.reshape(-1,3,128,128)).reshape(b,t,512,4,4) for x in data[:2]]
    expected=m.forward_sequence(*data)[0];actual=c.forward_feature_sequence(*features,*data[2:])[0]
    assert torch.equal(expected,actual)
    expected.square().mean().backward();actual.square().mean().backward()
    for (k,v),(q,w) in zip(m.named_parameters(),c.named_parameters()):
        assert k==q
        if v.requires_grad:assert torch.equal(v.grad,w.grad)
        else:assert v.grad is None and w.grad is None
    xyz_grad=m.fusion[0].weight.grad[:,-3:]
    assert bool(torch.count_nonzero(xyz_grad))==(arm==ARMS[1])
    m.train();assert not m.visual_encoder.backbone.training
    with torch.no_grad():
        for i in range(t):
            out,state,n=m.forward_window_step(*[v[:,i] for v in data])
            assert state is None and n==1 and torch.allclose(out,expected[:,i],atol=1e-6,rtol=1e-6)


@pytest.mark.parametrize('arm',ARMS)
@pytest.mark.parametrize('bad',['nan','width','integer','hidden'])
def test_invalid_xyz_and_history_rejected(arm,bad):
    m=tiny(arm);data=inputs()
    if bad=='nan':data[-1][0,0,0]=float('nan')
    if bad=='width':data[-1]=data[-1][:,:,:2]
    if bad=='integer':data[-1]=data[-1].long()
    with pytest.raises(ValueError):m.forward_sequence(*data,hidden=torch.zeros(1) if bad=='hidden' else None)
