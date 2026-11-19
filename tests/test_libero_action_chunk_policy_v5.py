"""Blank ResNet structures and invented causal windows only."""
from copy import deepcopy
import pytest
import torch
from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig,masked_behavior_cloning_loss
from src.algorithms.libero_spatial_readout_policy_v3 import make_visual_readout_policy_v3
from src.algorithms.libero_action_chunk_policy_v5 import (
    ARMS,REGISTERED_PARAMETERS,TRAINABLE_PARAMETERS,chunk_targets,supervision_mask,chunk_masked_mse,
    make_action_chunk_policy_v5,
)

torch.set_num_threads(1)
CFG=LiberoBCPolicyConfig(memory_mode='cnn_only')


@pytest.mark.parametrize('seed',[20261106,20261107,20261108])
def test_exact_registered_active_counts_shared_initial_state_and_local_rng(seed,monkeypatch):
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('research weights forbidden'))
    monkeypatch.setattr(torch.hub,'load_state_dict_from_url',lambda *a,**k:pytest.fail('download forbidden'))
    before=torch.get_rng_state().clone();left,right=[make_action_chunk_policy_v5(CFG,arm=a,seed=seed) for a in ARMS]
    assert torch.equal(before,torch.get_rng_state())
    assert left.state_dict().keys()==right.state_dict().keys()
    assert all(torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items())
    for arm,m in zip(ARMS,[left,right]):
        assert sum(v.numel() for v in m.parameters())==11176512+REGISTERED_PARAMETERS==12681053
        assert sum(v.numel() for v in m.parameters() if v.requires_grad)==TRAINABLE_PARAMETERS[arm]
        assert sum(v.numel() for v in m.future_action_heads.parameters())==5397
        assert all(v.requires_grad==(arm==ARMS[1]) for v in m.future_action_heads.parameters())
        assert m.fusion[0].in_features==268 and not m.visual_encoder.backbone.training
    assert left.action_head.weight.data_ptr()!=left.future_action_heads[0].weight.data_ptr()


def test_double_boundary_t7_t8_t9_and_demo_end():
    actions=torch.arange(2*10*7,dtype=torch.float32).reshape(2,10,7)/200
    mask=torch.tensor([[True]*10,[True]*8+[False]*2]);starts=torch.tensor([1,11]);ends=torch.tensor([100,19])
    target,anchor,future=chunk_targets(actions,mask,starts,ends)
    assert future[0,7].tolist()==[True,True,True,False]
    assert future[0,8].tolist()==[True,True,False,False]
    assert future[0,9].tolist()==[True,False,False,False]
    assert future[1,7].tolist()==[True,False,False,False]
    assert not future[1,8:].any() and not anchor[1,8:].any()
    assert future[0].sum()==34 and future[1].sum()==26
    assert torch.equal(target[0,7,:3],actions[0,7:10]) and not target[0,7,3].any()
    assert supervision_mask(anchor,future,ARMS[0]).sum()==18
    assert supervision_mask(anchor,future,ARMS[1]).sum()==60


@pytest.mark.parametrize('arm',ARMS)
def test_double_mask_seven_dimensional_mse_matches_scalar_and_isolates_invalid_gradients(arm):
    actions=torch.linspace(-.9,.9,70).reshape(1,10,7);valid=torch.tensor([[True]*8+[False]*2])
    targets,a,f=chunk_targets(actions,valid,torch.tensor([1]),torch.tensor([9]))
    pred=torch.linspace(-.5,.5,280).reshape(1,10,4,7).requires_grad_();mask=supervision_mask(a,f,arm)
    value=chunk_masked_mse(pred,targets,a,f,arm)
    scalar=sum((float(pred[0,t,h,j].detach())-float(targets[0,t,h,j]))**2 for t in range(10) for h in range(4) for j in range(7) if mask[0,t,h])/(7*int(mask.sum()))
    assert float(value)==pytest.approx(scalar,abs=1e-7)
    value.backward();assert not pred.grad[~mask].any()
    changed=pred.detach().clone();changed[~mask]=1000
    assert torch.equal(value.detach(),chunk_masked_mse(changed,targets,a,f,arm))


@pytest.mark.parametrize('fault',['start0','wrong_demo_end','hole','mask_dtype','nan'])
def test_bad_causal_metadata_rejected(fault):
    actions=torch.zeros(1,10,7);mask=torch.ones(1,10,dtype=torch.bool);s=torch.tensor([1]);n=torch.tensor([20])
    if fault=='start0':s[0]=0
    if fault=='wrong_demo_end':n[0]=9
    if fault=='hole':mask[0,3]=False
    if fault=='mask_dtype':mask=mask.int()
    if fault=='nan':actions[0,1,0]=float('nan')
    with pytest.raises(ValueError):chunk_targets(actions,mask,s,n)


def test_legacy_control_action_loss_gradients_and_adam_match_without_future_optimizer_state():
    seed=20261106;old=make_visual_readout_policy_v3(CFG,arm='spatial_coordinates_zero_xyz',seed=seed)
    control=make_action_chunk_policy_v5(CFG,arm=ARMS[0],seed=seed)
    frozen=deepcopy(control.future_action_heads.state_dict());g=torch.Generator().manual_seed(44)
    inputs=[torch.randn(1,10,512,4,4,generator=g) for _ in range(2)]+[torch.randn(1,10,7,generator=g),torch.randn(1,10,2,generator=g)]
    actions=torch.randn(1,10,7,generator=g).tanh();mask=torch.ones(1,10,dtype=torch.bool)
    target,a,f=chunk_targets(actions,mask,torch.tensor([1]),torch.tensor([30]))
    optimizers=[torch.optim.Adam([v for v in m.parameters() if v.requires_grad],lr=.0003) for m in [old,control]]
    for _ in range(2):
        for opt in optimizers:opt.zero_grad(set_to_none=True)
        left=old.forward_feature_sequence(*inputs)[0];right=control.forward_feature_sequence(*inputs)[0]
        assert torch.equal(left,right[:,:,0]) and not right[:,:,1:].any()
        l=masked_behavior_cloning_loss(left,actions,mask);r=chunk_masked_mse(right,target,a,f,ARMS[0]);assert torch.equal(l,r)
        l.backward();r.backward();new=dict(control.named_parameters())
        for name,v in old.named_parameters():
            if v.requires_grad:assert torch.equal(v.grad,new[name].grad),name
            else:assert new[name].grad is None
        for m,opt in zip([old,control],optimizers):
            torch.nn.utils.clip_grad_norm_([v for v in m.parameters() if v.requires_grad],1.);opt.step()
    assert all(torch.equal(v,control.state_dict()[k]) for k,v in old.state_dict().items())
    assert all(v.grad is None and v not in optimizers[1].state for v in control.future_action_heads.parameters())
    assert all(torch.equal(v,frozen[k]) for k,v in control.future_action_heads.state_dict().items())


def test_candidate_future_heads_train_only_current_observation_and_first_action_is_executed():
    m=make_action_chunk_policy_v5(CFG,arm=ARMS[1],seed=20261107);g=torch.Generator().manual_seed(52)
    features=[torch.randn(1,10,512,4,4,generator=g) for _ in range(2)];j=torch.randn(1,10,7,generator=g);q=torch.randn(1,10,2,generator=g)
    prediction=m.forward_feature_sequence(*features,j,q)[0]
    changed=[v.clone() for v in features];changed[0][:,8:]+=100
    assert torch.equal(prediction[:,:8],m.forward_feature_sequence(*changed,j,q)[0][:,:8])
    target,a,f=chunk_targets(torch.zeros(1,10,7),torch.ones(1,10,dtype=torch.bool),torch.tensor([1]),torch.tensor([40]))
    chunk_masked_mse(prediction,target,a,f,ARMS[1]).backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.future_action_heads.parameters())
    pixels=[torch.randint(256,(1,3,128,128),dtype=torch.uint8,generator=g) for _ in range(2)]
    with torch.no_grad():
        expected=m.forward_sequence(*[v.unsqueeze(1) for v in pixels],j[:,:1],q[:,:1])[0][:,0,0]
        value,context,count=m.forward_window_step(*pixels,j[:,0],q[:,0]);assert torch.equal(value,expected) and context is None and count==1
        for head in m.future_action_heads:head.bias.add_(100)
        assert torch.equal(value,m.forward_window_step(*pixels,j[:,0],q[:,0])[0])
    with pytest.raises(ValueError):m.forward_window_step(*pixels,j[:,0],q[:,0],context=torch.zeros(1))
    with pytest.raises(ValueError):m.entropy_summary()
