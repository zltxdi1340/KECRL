"""Opt-in fixed visual backbone with one-frame MLP or bounded ten-frame GRU.

No downloads, file loads, training/evaluation authority, or changed defaults.
"""
from __future__ import annotations

from dataclasses import replace
import torch
from torch import nn

from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
from src.algorithms.libero_frozen_resnet18_policy import LiberoFrozenResNet18Policy


class WindowedResNet18Policy(LiberoFrozenResNet18Policy):
    def __init__(self,initialized,*,window,seed):
        if type(window) is not int or window not in (1,10):raise ValueError('only the paired windows1/10 are admitted')
        nn.Module.__init__(self)
        self.config=replace(initialized.config,memory_mode='cnn_only' if window==1 else 'gru')
        self.visual_encoder=initialized.visual_encoder;self.fusion=initialized.fusion;self.action_head=initialized.action_head
        self.window=window;self.compute_guard=None
        if window==1:self.recurrent=initialized.recurrent
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+9000001)
                self.recurrent=nn.GRU(self.config.fusion_dim,self.config.recurrent_hidden_dim,batch_first=True)

    def _features(self,agent,hand,joint,gripper):
        values=[]
        for rgb in [agent,hand]:
            fn=self.visual_encoder.encode_frozen
            frozen=fn(rgb) if self.compute_guard is None else self.compute_guard('frozen_encoder',fn,rgb)
            values.append(self.visual_encoder.projection(frozen))
        return self.fusion(torch.cat([*values,joint.float(),gripper.float()],dim=-1))

    def forward_sequence(self,agentview_rgb,eye_in_hand_rgb,joint_states,gripper_states,hidden=None):
        if agentview_rgb.ndim!=5 or agentview_rgb.shape!=eye_in_hand_rgb.shape:raise ValueError('paired RGB sequences required')
        b,t=agentview_rgb.shape[:2]
        if joint_states.shape!=(b,t,7) or gripper_states.shape!=(b,t,2):raise ValueError('public proprioception shapes differ')
        fused=self._features(agentview_rgb.reshape(-1,3,128,128),eye_in_hand_rgb.reshape(-1,3,128,128),
            joint_states.reshape(-1,7),gripper_states.reshape(-1,2)).reshape(b,t,-1)
        values,state=self.recurrent(fused,hidden)
        return torch.tanh(self.action_head(values)),state

    def forward_window_step(self,agent,hand,joint,gripper,context=None):
        """Context contains at most nine past fused observations, never actions."""
        current=self._features(agent,hand,joint,gripper).unsqueeze(1)
        if self.window==1:
            if context is not None:raise ValueError('frame control must not retain context')
            values,_=self.recurrent(current,None);return torch.tanh(self.action_head(values[:,0])),None,1
        if context is not None:
            if (context.ndim!=3 or context.shape[0]!=current.shape[0] or context.shape[2]!=self.config.fusion_dim
                or not 1<=context.shape[1]<=9 or context.requires_grad or context.dtype!=current.dtype
                or context.device!=current.device or not torch.isfinite(context).all()):raise ValueError('invalid bounded context')
            sequence=torch.cat([context,current],dim=1)
        else:sequence=current
        values,_=self.recurrent(sequence,None)
        return torch.tanh(self.action_head(values[:,-1])),sequence[:,-9:].detach().clone(),sequence.shape[1]


def make_windowed_policy(config,*,window,seed,public_state=None):
    """Construct from an already admitted in-memory public state, or blank for strict reload."""
    from torchvision.models import resnet18
    config=LiberoBCPolicyConfig.from_mapping(config) if isinstance(config,dict) else config
    if config.memory_mode!='cnn_only':raise ValueError('paired base config must be cnn_only')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        with torch.random.fork_rng(devices=[]):backbone=resnet18(weights=None)
        if public_state is not None:backbone.load_state_dict(public_state,strict=True)
        initialized=LiberoFrozenResNet18Policy(config,nn.Sequential(*list(backbone.children())[:-2]))
        return WindowedResNet18Policy(initialized,window=window,seed=seed)
