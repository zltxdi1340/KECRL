from copy import deepcopy

import pytest
import torch

from src.algorithms.libero_bc_policy import LiberoBCPolicy,LiberoBCPolicyConfig
from src.algorithms.libero_deterministic_pool import DeterministicDivisibleAveragePool,make_deterministic_libero_policy


@pytest.mark.parametrize('shape,out', [((2,3,16,16),4),((1,2,12,16),4),((1,2,1,1),1)])
def test_equivalent_pool_matches_adaptive_output_and_backward(shape,out):
    torch.manual_seed(18)
    left=torch.randn(shape,dtype=torch.float64,requires_grad=True)
    right=left.detach().clone().requires_grad_(True)
    expected=torch.nn.AdaptiveAvgPool2d(out)(left)
    actual=DeterministicDivisibleAveragePool(out)(right)
    weights=torch.randn_like(expected)
    expected_grad,=torch.autograd.grad((expected*weights).sum(),(left,))
    actual_grad,=torch.autograd.grad((actual*weights).sum(),(right,))
    torch.testing.assert_close(actual,expected,atol=1e-12,rtol=0.)
    torch.testing.assert_close(actual_grad,expected_grad,atol=1e-12,rtol=0.)


def test_equivalent_pool_preserves_policy_parameters_outputs_and_gradients():
    config=LiberoBCPolicyConfig(cnn_channels=(2,3,4),spatial_pool_size=4,visual_embedding_dim=4,
                               fusion_dim=4,recurrent_hidden_dim=4,memory_mode='cnn_only')
    torch.manual_seed(31);old=LiberoBCPolicy(config)
    torch.manual_seed(31);new=make_deterministic_libero_policy(config)
    assert list(old.state_dict())==list(new.state_dict())
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in old.state_dict().items())
    args=(torch.zeros(1,1,3,128,128,dtype=torch.uint8),torch.ones(1,1,3,128,128,dtype=torch.uint8),
          torch.zeros(1,1,7),torch.zeros(1,1,2))
    a,_=old.forward_sequence(*args);b,_=new.forward_sequence(*args)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=0.)
    ag=torch.autograd.grad(a.square().sum(),tuple(old.parameters()))
    bg=torch.autograd.grad(b.square().sum(),tuple(new.parameters()))
    for x,y in zip(ag,bg):torch.testing.assert_close(x,y,atol=1e-6,rtol=0.)


def test_nondivisible_grid_fails_instead_of_approximating():
    with pytest.raises(ValueError):DeterministicDivisibleAveragePool(4)(torch.zeros(1,2,15,16))
