"""Invented pixels/features and blank architectures; no research payloads."""
from copy import deepcopy
import math
import numpy as np
import pytest
import torch
from torch import nn

from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig,masked_behavior_cloning_loss
from src.algorithms.libero_frozen_resnet18_policy import LiberoFrozenResNet18Policy
from src.algorithms.libero_windowed_resnet18_policy import WindowedResNet18Policy
from src.algorithms.libero_spatial_readout_policy_v3 import (
    ARMS,READOUT_PARAMETERS,TRAINABLE_PARAMETERS,SpatialSoftmax2D,SpatialCoordinateReadout,
    VisualReadoutPolicyV3,make_visual_readout_policy_v3,
)
from experiments import libero_task7_spatial_readout_protocol_v3 as protocol

torch.set_num_threads(1)
CFG=LiberoBCPolicyConfig(memory_mode='cnn_only')


def invented(arm,seed=12):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        backbone=nn.Sequential(nn.AdaptiveAvgPool2d((4,4)),nn.Conv2d(3,512,1),nn.BatchNorm2d(512))
        base=WindowedResNet18Policy(LiberoFrozenResNet18Policy(CFG,backbone),window=1,seed=seed)
        return VisualReadoutPolicyV3(base,arm=arm,seed=seed)


def pixels():
    g=torch.Generator().manual_seed(181)
    return [torch.randint(256,(2,3,3,128,128),dtype=torch.uint8,generator=g),
        torch.randint(256,(2,3,3,128,128),dtype=torch.uint8,generator=g),
        torch.randn(2,3,7,generator=g),torch.randn(2,3,2,generator=g)]


@pytest.mark.parametrize('seed',[20261106,20261107,20261108])
def test_exact_full_architecture_parameter_counts_and_shared_initialization_without_loads(seed,monkeypatch):
    def forbidden(*a,**kw):pytest.fail('research weights/downloads must not be opened')
    monkeypatch.setattr(torch,'load',forbidden)
    monkeypatch.setattr(torch.hub,'load_state_dict_from_url',forbidden)
    before=torch.get_rng_state().clone()
    flat,spatial=[make_visual_readout_policy_v3(CFG,arm=a,seed=seed) for a in ARMS]
    assert torch.equal(before,torch.get_rng_state())
    for arm,m in zip(ARMS,[flat,spatial]):
        assert sum(v.numel() for v in m.parameters() if v.requires_grad)==TRAINABLE_PARAMETERS[arm]
        assert sum(v.numel() for v in m.visual_encoder.projection.parameters())==READOUT_PARAMETERS[arm]
        assert sum(v.numel() for v in m.parameters())==11176512+TRAINABLE_PARAMETERS[arm]
        assert m.fusion[0].in_features==268 and m.arm=='zero_xyz_mlp'
        assert not any(v.requires_grad for v in m.visual_encoder.backbone.parameters())
    common=lambda m:{k:v for k,v in m.state_dict().items() if not k.startswith('visual_encoder.projection.')}
    a,b=common(flat),common(spatial)
    assert set(a)==set(b) and all(torch.equal(v,b[k]) for k,v in a.items())
    assert set(flat.state_dict())!=set(spatial.state_dict())
    assert not any('temperature' in k for k,_ in spatial.named_parameters())
    assert isinstance(spatial.visual_encoder.projection.coordinate_mlp[-1],nn.Linear)


def test_numpy_independent_coordinates_and_entropy():
    values=np.random.default_rng(61).normal(size=(3,5,4,4))
    scores=torch.tensor(values,dtype=torch.float64,requires_grad=True)
    coordinates,entropy=SpatialSoftmax2D()(scores,return_entropy=True)
    flat=values.reshape(3,5,16);prob=np.exp(flat-flat.max(axis=-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
    axis=np.arange(4)*(2/3)-1;yy,xx=np.meshgrid(axis,axis,indexing='ij')
    expected=np.stack([(prob*xx.reshape(16)).sum(-1),(prob*yy.reshape(16)).sum(-1)],-1).reshape(3,10)
    expected_h=-(prob*np.log(prob)).sum(-1)
    np.testing.assert_allclose(coordinates.detach().numpy(),expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_allclose(entropy.numpy(),expected_h,atol=1e-12,rtol=1e-12)
    assert coordinates.requires_grad and not entropy.requires_grad


def test_uniform_peaked_shift_invariance_and_coordinate_orientation():
    op=SpatialSoftmax2D();zero=torch.zeros(2,3,4,4,dtype=torch.float64)
    out,h=op(zero,return_entropy=True)
    assert torch.allclose(out,torch.zeros_like(out),atol=1e-15)
    assert torch.allclose(h,torch.full_like(h,math.log(16)),atol=1e-15)
    scores=zero.clone();scores[:,:,1,2]=1000
    out,h=op(scores,return_entropy=True)
    expected=torch.tensor([1/3,-1/3],dtype=torch.float64).repeat(2,3)
    assert torch.allclose(out,expected,atol=1e-12) and torch.all(h==0)
    assert torch.allclose(op(scores+10000),out,atol=1e-12)
    reflected=op(scores.flip(-1)).reshape(2,3,2)
    assert torch.allclose(reflected[...,0],-out.reshape(2,3,2)[...,0],atol=1e-12)


def test_coordinate_gradcheck_and_fixed_temperature_guard():
    op=SpatialSoftmax2D();x=torch.randn(1,2,4,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(op,(x,),eps=1e-6,atol=1e-5,rtol=1e-4)
    assert not list(op.parameters())
    op.temperature.fill_(.5)
    with pytest.raises(ValueError,match='temperature'):op(x)


@pytest.mark.parametrize('value',[0.,.5,2.,float('nan'),True])
def test_temperature_variants_are_not_candidates(value):
    with pytest.raises(ValueError):SpatialSoftmax2D(temperature=value)


@pytest.mark.parametrize('kind',['shape','nan','inf','integer','half','empty'])
def test_invalid_response_maps_fail_closed(kind):
    x=torch.zeros(1,2,4,4)
    if kind=='shape':x=torch.zeros(1,2,8,2)
    if kind=='nan':x[0,0,0,0]=float('nan')
    if kind=='inf':x[0,0,0,0]=float('inf')
    if kind=='integer':x=x.long()
    if kind=='half':x=x.half()
    if kind=='empty':x=x[:0]
    with pytest.raises(ValueError):SpatialSoftmax2D()(x)


@pytest.mark.parametrize('arm',ARMS)
def test_pixel_cache_action_gradient_parity_frozen_bn_and_zero_xyz(arm):
    model=invented(arm);clone=deepcopy(model);data=pixels();b,t=data[0].shape[:2]
    frozen_before=deepcopy(model.visual_encoder.backbone.state_dict())
    cache=[model.visual_encoder.encode_frozen(v.reshape(-1,3,128,128)).reshape(b,t,512,4,4) for v in data[:2]]
    model.train();clone.train()
    direct=model.forward_sequence(*data)[0];cached=clone.forward_feature_sequence(*cache,*data[2:])[0]
    assert torch.equal(direct,cached)
    mask=torch.tensor([[1,1,0],[1,0,0]],dtype=torch.bool)
    target=torch.zeros_like(direct)
    masked_behavior_cloning_loss(direct,target,mask).backward()
    masked_behavior_cloning_loss(cached,target,mask).backward()
    for (key,v),(other,w) in zip(model.named_parameters(),clone.named_parameters()):
        assert key==other
        if v.requires_grad:assert torch.allclose(v.grad,w.grad,atol=1e-9,rtol=1e-5)
        else:assert v.grad is None and w.grad is None
    assert not model.visual_encoder.backbone.training
    assert all(torch.equal(v,frozen_before[k]) for k,v in model.visual_encoder.backbone.state_dict().items())
    assert not torch.count_nonzero(model.fusion[0].weight.grad[:,-3:])
    if arm==ARMS[1]:
        grad=model.visual_encoder.projection.channel_projection.weight.grad
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad)>0
        record=model.entropy_summary(mask);assert record['valid_rows']==3 and record['heatmaps_per_view']==768
        protocol.audit_entropy_summary(record,3)
        assert model.entropy_summary(mask)==record  # No extra forward or accumulation.
        assert all(not v.requires_grad and v.grad_fn is None for v in model._entropy_views)
    else:assert model.entropy_summary(mask) is None
    with pytest.raises(TypeError):model.forward_sequence(*data,eef_pos=torch.zeros(2,3,3))


def test_online_frame_matches_sequence_and_entropy_excludes_padding():
    m=invented(ARMS[1]).eval();data=pixels()
    with torch.no_grad():
        expected=m.forward_sequence(*data)[0]
        a,h=[v.clone() for v in m._entropy_views]
        mask=torch.tensor([[1,0,0],[1,0,0]],dtype=torch.bool)
        record=m.entropy_summary(mask)
        chosen=mask.reshape(-1)
        mean=float((a[chosen].double().sum()+h[chosen].double().sum())/(2*2*256))
        assert record['entropy_nats_mean']==mean
        for i in range(3):
            out,state,rows=m.forward_window_step(*[v[:,i] for v in data])
            assert state is None and rows==1 and torch.allclose(out,expected[:,i],atol=1e-6,rtol=1e-6)
    with pytest.raises(ValueError):m.entropy_summary(torch.ones(1,dtype=torch.bool))
    with pytest.raises(ValueError):m.forward_window_step(*[v[:,0] for v in data],context=torch.zeros(1))
    with pytest.raises(ValueError):m.entropy_summary()


def test_activation_snapshots_do_not_change_state_dict_and_reject_stale_or_empty_masks():
    m=invented(ARMS[1]);before={k:v.clone() for k,v in m.state_dict().items()}
    with pytest.raises(ValueError):m.entropy_summary()
    data=pixels();m.forward_sequence(*data)
    assert all(torch.equal(v,before[k]) for k,v in m.state_dict().items())
    with pytest.raises(ValueError):m.entropy_summary(torch.zeros(2,3,dtype=torch.bool))
    bad=list(data);bad[2]=bad[2].clone();bad[2][0,0,0]=float('nan')
    with pytest.raises(ValueError):m.forward_sequence(*bad)
    with pytest.raises(ValueError):m.entropy_summary()


def test_readout_rejects_extra_rows_before_convolution_and_snapshots_are_copies():
    r=SpatialCoordinateReadout()
    with pytest.raises(ValueError):r(torch.zeros(81,512,4,4))
    r(torch.randn(2,512,4,4));first=r.entropy_snapshot();first.fill_(-99)
    assert torch.all(r.entropy_snapshot()>=0)


@pytest.mark.parametrize('arm',ARMS)
def test_readout_call_guard_and_entropy_queries_do_not_add_forwards(arm):
    model=invented(arm);calls=[]
    def guard(key,fn,*args,**kwargs):
        calls.append(key);return fn(*args,**kwargs)
    model.compute_guard=guard
    g=torch.Generator().manual_seed(81)
    args=[torch.randn(1,2,512,4,4,generator=g),torch.randn(1,2,512,4,4,generator=g),
          torch.randn(1,2,7,generator=g),torch.randn(1,2,2,generator=g)]
    model.forward_feature_sequence(*args)
    key='spatial_readout' if arm==ARMS[1] else 'flat_readout'
    assert calls==[key,key]
    model.entropy_summary();model.entropy_summary()
    assert calls==[key,key]
    if arm==ARMS[1]:
        def refused(*args,**kwargs):raise RuntimeError('invented exhausted readout budget')
        model.compute_guard=refused
        with pytest.raises(RuntimeError):model.forward_feature_sequence(*args)
        with pytest.raises(ValueError):model.entropy_summary()
