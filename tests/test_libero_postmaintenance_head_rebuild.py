"""Invented zeros only; current metadata supplies identities, never source arrays."""
import json
import numpy as np
import pytest
import torch

from experiments import rebuild_libero_postmaintenance_A_heads as rebuild


def cfg():
    old=json.loads((rebuild.ROOT/'configs/libero-task7-bounded-constrained-feasibility-diagnostic_v1.json').read_text())
    return {'inputs':{'action_lengths':old['inputs']['action_lengths'],'folds':old['inputs']['folds'],
            'models':[{'seed':s,'arm':'A_original'} for s in (20261106,20261107,20261108)]}}


def test_new_A_role_contract_preserves_whole_demo_and_forbids_early_probe():
    config=cfg();roles=rebuild.scoped_roles(config)
    assert len(roles.expected)==9
    with pytest.raises(ValueError,match='all nine'):
        roles.open_probe()
    weight=torch.nn.Parameter(torch.zeros(256));bias=torch.nn.Parameter(torch.zeros(1))
    key=(20261106,'A_original',0,'harm_protected_affine')
    with pytest.raises(ValueError,match='probe or invalid'):
        with roles.head(key,[weight,bias]):
            roles.batch(np.zeros(64,np.int32))
    assert roles.active is None and roles.done[key]==0


def test_real500_auxiliary_steps_on_invented_rows_with_registered_gradients_only():
    from experiments import run_libero_task7_residual_harm_protection as kernel
    torch.set_num_threads(1)
    config=cfg();roles=rebuild.scoped_roles(config);plan=config['inputs']['folds'][0]
    n=len(plan['fit_rows']);x=np.zeros((n,256),np.float32);target=np.zeros((n,7),np.float32)
    actual=dict(backward_attempted=0,backward_completed=0,optimizer_attempted=0,optimizer_completed=0)
    class Limits:
        def check(self):pass
    with rebuild.optimizer_guard(roles,actual):
        with pytest.raises(ValueError):
            (torch.ones(1,requires_grad=True)*2).sum().backward()
        arrays,metadata=kernel.fit_head(x,target,target,kernel.sampled_rows(20261106,0,plan),plan,
            (20261106,'A_original',0,'harm_protected_affine'),roles,Limits())
    assert set(actual.values())=={500}
    assert metadata['actual_state_steps']==[500,500]
    assert arrays['step_weights_before'].shape==(500,256)
    assert arrays['sample_rows'].shape==(500,64)
    assert roles.counts['optimizer_steps']==roles.counts['backward_calls']==500
    assert roles.done[(20261106,'A_original',0,'harm_protected_affine')]==500


def test_A_scoped_role_cannot_impersonate_B_or_relabel_fold_membership():
    bad=cfg();bad['inputs']['models'][0]['arm']='B_fomaml'
    with pytest.raises(ValueError,match='A cohort'):
        rebuild.scoped_roles(bad)
    bad=cfg();bad['inputs']['folds'][0]['fit_rows'][0]=0
    with pytest.raises(ValueError,match='whole-demo'):
        rebuild.scoped_roles(bad)
