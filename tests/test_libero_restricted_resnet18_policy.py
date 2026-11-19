"""Invented ResNet states/RGB only; real autograd, optimizer and mask checks."""
from __future__ import annotations

import hashlib

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('torchvision')
from torchvision.models import resnet18

from experiments.libero_formal_execution import Ledger
from experiments import libero_restricted_vision_finetune_protocol as p
from experiments import audit_libero_restricted_vision_finetune as audit
from experiments.run_libero_restricted_vision_finetune import describe_mask, immutable_hash
from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig, masked_behavior_cloning_loss
from src.algorithms.libero_frozen_resnet18_policy import make_frozen_resnet18_policy
from src.algorithms.libero_restricted_resnet18_policy import (
    CONVOLUTION_PARAMETER_NAMES, make_restricted_resnet18_policy, parameter_manifest,
)


@pytest.fixture(scope='module')
def invented_weights(tmp_path_factory):
    path = tmp_path_factory.mktemp('invented_restricted_resnet')/'invented.pt'
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        model = resnet18(weights=None)
    torch.save(model.state_dict(),path)
    return path,hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def factory(invented_weights,monkeypatch):
    def forbidden(*args,**kwargs):
        raise AssertionError('synthetic tests must never download weights')
    monkeypatch.setattr(torch.hub,'download_url_to_file',forbidden)
    monkeypatch.setattr(torch.hub,'load_state_dict_from_url',forbidden)
    def make(mode=True):
        torch.manual_seed(71)
        return make_restricted_resnet18_policy(LiberoBCPolicyConfig(memory_mode='cnn_only'),
            *invented_weights,train_layer4=mode)
    return make


def inputs():
    images = torch.arange(2*3*128*128).remainder(256).to(torch.uint8).reshape(1,2,3,128,128)
    return images,images.flip(-1),torch.ones(1,2,7),torch.ones(1,2,2)


def test_same_architecture_and_all_initial_tensors_not_just_head(factory,invented_weights):
    frozen,tuned = factory(False),factory(True)
    torch.manual_seed(71)
    old = make_frozen_resnet18_policy(LiberoBCPolicyConfig(memory_mode='cnn_only'),*invented_weights)
    assert list(frozen.state_dict()) == list(tuned.state_dict()) == list(old.state_dict())
    for name,value in frozen.state_dict().items():
        assert torch.equal(value,tuned.state_dict()[name]) and torch.equal(value,old.state_dict()[name])
    a,b = describe_mask(frozen,p.ARMS[0]),describe_mask(tuned,p.ARMS[1])
    assert a['total'] == b['total'] == 12689864
    assert a['trainable'] == 1513352 and b['trainable'] == 9901960
    assert b['trainable']-a['trainable'] == 8388608
    assert set(b['names'])-set(a['names']) == set(CONVOLUTION_PARAMETER_NAMES)


@pytest.mark.parametrize('mode',[False,True])
def test_one_real_update_preserves_bn_and_exactly_admits_five_convolutions(factory,mode,invented_weights):
    model = factory(mode).train()
    before = {name:value.detach().clone() for name,value in model.state_dict().items()}
    mask = parameter_manifest(model)
    immutable_before = immutable_hash(model,mask['names'])
    optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=3e-4)
    output,hidden = model.forward_sequence(*inputs())
    loss = masked_behavior_cloning_loss(output,torch.full_like(output,.23),torch.ones(1,2,dtype=torch.bool))
    loss.backward()
    for name,value in model.named_parameters():
        assert (value.grad is not None) == value.requires_grad
        if name in CONVOLUTION_PARAMETER_NAMES:
            assert (value.grad is not None) == mode
            if mode: assert torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
    torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    assert hidden is None and not model.visual_encoder.backbone.training
    assert all(not module.training for module in model.visual_encoder.backbone.modules())
    assert all(not torch.equal(before[name],model.state_dict()[name]) for name in
        ['visual_encoder.projection.1.weight','action_head.weight'])
    assert immutable_hash(model,mask['names']) == immutable_before
    for name in CONVOLUTION_PARAMETER_NAMES:
        assert torch.equal(before[name],model.state_dict()[name]) != mode
    row = dict(arm=p.ARMS[int(mode)],parameter_mask=mask,trainable_parameters=mask['trainable'],
        immutable_before=immutable_before,immutable_after=immutable_before)
    public = torch.load(invented_weights[0],weights_only=True)
    audit.verify_backbone(model.state_dict(),public,row)
    bad = {name:value.clone() for name,value in model.state_dict().items()}
    bad['visual_encoder.backbone.7.0.bn1.running_mean'][0] += 1
    with pytest.raises(ValueError,match='immutable state'):
        audit.verify_backbone(bad,public,row)
    # Updating the self-reported digest must not hide a changed BN tensor.
    forged = dict(row,immutable_before=immutable_hash_state(bad,mask['names']),
                  immutable_after=immutable_hash_state(bad,mask['names']))
    with pytest.raises(ValueError,match='differs from public'):
        audit.verify_backbone(bad,public,forged)


def immutable_hash_state(state, names):
    return audit.state_sha({name:value for name,value in state.items() if name not in set(names)})


def test_cached_prefix_matches_raw_actions_and_candidate_gradients(factory):
    model = factory().train(); agent,hand,joints,gripper = inputs()
    encoder = model.visual_encoder
    prefix = [encoder.encode_prefix(images.reshape(2,3,128,128)).reshape(1,2,256,8,8).clone() for images in (agent,hand)]
    initial = [value.clone() for value in prefix]
    direct,_ = model.forward_sequence(agent,hand,joints,gripper)
    direct.square().sum().backward()
    gradients = {name:value.grad.clone() for name,value in model.named_parameters() if name in CONVOLUTION_PARAMETER_NAMES}
    model.zero_grad(set_to_none=True)
    cached,_ = model.forward_feature_sequence(*prefix,joints,gripper)
    cached.square().sum().backward()
    torch.testing.assert_close(direct,cached,atol=0,rtol=0)
    for name,value in model.named_parameters():
        if name in gradients: torch.testing.assert_close(value.grad,gradients[name],atol=0,rtol=0)
    assert all(torch.equal(a,b) for a,b in zip(prefix,initial))
    assert all(not value.requires_grad and value.grad_fn is None for value in prefix)


def test_cache_entry_rejects_old_final_features_or_attached_graph(factory):
    model = factory(); _,_,joints,gripper = inputs()
    old = torch.zeros(1,2,512,4,4)
    with pytest.raises(ValueError,match='old final features'):
        model.forward_feature_sequence(old,old,joints,gripper)
    prefix = torch.ones(1,2,256,8,8,requires_grad=True)
    with pytest.raises(ValueError,match='detached'):
        model.forward_feature_sequence(prefix,prefix.detach(),joints,gripper)
    with pytest.raises(ValueError,match='final-feature caching'):
        model.visual_encoder.encode_frozen(inputs()[0].reshape(2,3,128,128))


@pytest.mark.parametrize('kind',['prefix_forward','layer4_forward'])
def test_component_cap_denies_before_any_module_compute(factory,kind):
    model = factory(); ledger = Ledger(dict(prefix_forward=0,layer4_forward=0))
    model.visual_encoder.compute_guard = ledger.call
    stage = model.visual_encoder.backbone[0 if kind == 'prefix_forward' else 7]
    def forbidden(*args):raise AssertionError('operation ran before admission')
    hook = stage.register_forward_pre_hook(forbidden)
    try:
        with pytest.raises(RuntimeError,match=kind):
            if kind == 'prefix_forward':model.visual_encoder.encode_prefix(torch.zeros(1,3,128,128,dtype=torch.uint8))
            else:model.visual_encoder.forward_prefix(torch.zeros(1,256,8,8))
    finally:hook.remove()
    assert ledger.counts[kind] == dict(attempted=0,completed=0)


def test_step_sequence_uint8_float_and_eval_grad_modes(factory):
    model = factory().eval(); agent,hand,joints,gripper = inputs()
    full,_ = model.forward_sequence(agent,hand,joints,gripper)
    scaled,_ = model.forward_sequence(agent.float()/255.,hand.float()/255.,joints,gripper)
    single,hidden = model.forward_step(agent[:,0],hand[:,0],joints[:,0],gripper[:,0])
    torch.testing.assert_close(full,scaled,atol=0,rtol=0)
    torch.testing.assert_close(full[:,0],single,atol=1e-6,rtol=1e-6)
    full.sum().backward()
    assert hidden is None and dict(model.named_parameters())[CONVOLUTION_PARAMETER_NAMES[0]].grad is not None


@pytest.mark.parametrize('mode',[1,None,'layer4'])
def test_grad_mode_is_explicit_bool_before_weight_read(invented_weights,mode,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('mode rejected before file access')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='explicit bool'):
        make_restricted_resnet18_policy(LiberoBCPolicyConfig(memory_mode='cnn_only'),*invented_weights,train_layer4=mode)


def test_bad_sha_and_gru_fail_before_deserialization(invented_weights,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('invalid weights must not load')
    monkeypatch.setattr(torch,'load',forbidden)
    with pytest.raises(ValueError,match='SHA256 mismatch'):
        make_restricted_resnet18_policy(LiberoBCPolicyConfig(memory_mode='cnn_only'),invented_weights[0],'0'*64,train_layer4=True)
    with pytest.raises(ValueError,match='cnn_only'):
        make_restricted_resnet18_policy(LiberoBCPolicyConfig(memory_mode='gru'),*invented_weights,train_layer4=True)


@pytest.mark.parametrize('bad',[torch.ones(1,3,128,128)*255,torch.zeros(1,3,224,224),
                               torch.full((1,3,128,128),float('nan')),torch.zeros(1,3,128,128,dtype=torch.int16)])
def test_invalid_rgb_rejected(factory,bad):
    with pytest.raises(ValueError):factory().visual_encoder.encode_prefix(bad)
