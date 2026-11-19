"""Matched frame controllers with zero or public cached world XYZ channels.

No downloads, simulator reads, data selection, or changed project defaults.
"""
from __future__ import annotations

import torch
from torch import nn

from src.algorithms.libero_windowed_resnet18_policy import WindowedResNet18Policy, make_windowed_policy

ARMS = ('zero_xyz_mlp', 'cached_xyz_mlp')


class CachedEEFPolicy(WindowedResNet18Policy):
    def __init__(self, initialized, *, arm, seed):
        if arm not in ARMS or initialized.window != 1:
            raise ValueError('matched frame MLP and explicit XYZ arm required')
        super().__init__(initialized, window=1, seed=seed)
        self.arm = arm
        original = self.fusion[0]
        # Preserve all old 265-channel weights, bias, projection and head.
        # The extra three columns have the same seeded initialization in both arms.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 9100001)
            widened = nn.Linear(original.in_features + 3, original.out_features)
        with torch.no_grad():
            widened.weight[:, :-3].copy_(original.weight)
            widened.bias.copy_(original.bias)
        self.fusion[0] = widened

    def _fuse(self, agent, hand, joint, gripper, eef):
        shape = agent.shape[:-1]
        if (hand.shape != agent.shape or joint.shape != (*shape, 7)
                or gripper.shape != (*shape, 2) or eef.shape != (*shape, 3)):
            raise ValueError('matched visual, joint7, gripper2 and cached XYZ3 required')
        if not all(v.is_floating_point() and torch.isfinite(v).all()
                   for v in (joint, gripper, eef)):
            raise ValueError('finite public floating proprioception required')
        xyz = eef.float() if self.arm == ARMS[1] else torch.zeros_like(eef, dtype=torch.float32)
        return self.fusion(torch.cat((agent, hand, joint.float(), gripper.float(), xyz), dim=-1))

    def _features(self, agent, hand, joint, gripper, eef):
        values = []
        for rgb in (agent, hand):
            fn = self.visual_encoder.encode_frozen
            frozen = fn(rgb) if self.compute_guard is None else self.compute_guard('frozen_encoder', fn, rgb)
            values.append(self.visual_encoder.projection(frozen))
        return self._fuse(*values, joint, gripper, eef)

    def forward_sequence(self, agentview_rgb, eye_in_hand_rgb, joint_states, gripper_states, eef_pos, hidden=None):
        if hidden is not None or agentview_rgb.ndim != 5 or eye_in_hand_rgb.shape != agentview_rgb.shape:
            raise ValueError('paired RGB sequences without history required')
        b, t = agentview_rgb.shape[:2]
        if joint_states.shape != (b,t,7) or gripper_states.shape != (b,t,2) or eef_pos.shape != (b,t,3):
            raise ValueError('public sequence shapes differ')
        fused = self._features(agentview_rgb.reshape(-1,3,128,128), eye_in_hand_rgb.reshape(-1,3,128,128),
                               joint_states.reshape(-1,7), gripper_states.reshape(-1,2), eef_pos.reshape(-1,3)).reshape(b,t,-1)
        values, _ = self.recurrent(fused, None)
        return torch.tanh(self.action_head(values)), None

    def forward_feature_sequence(self, agent, hand, joint, gripper, eef):
        if (agent.ndim != 5 or hand.shape != agent.shape or tuple(agent.shape[2:]) != (512,4,4)
                or agent.requires_grad or hand.requires_grad):
            raise ValueError('detached paired [B,T,512,4,4] convolution cache required')
        b,t = agent.shape[:2]
        project = self.visual_encoder.projection
        a = project(agent.reshape(-1,512,4,4)).reshape(b,t,-1)
        h = project(hand.reshape(-1,512,4,4)).reshape(b,t,-1)
        values, _ = self.recurrent(self._fuse(a,h,joint,gripper,eef), None)
        return torch.tanh(self.action_head(values)), None

    def forward_window_step(self, agent, hand, joint, gripper, eef, context=None):
        if context is not None:
            raise ValueError('frame MLP cannot receive history')
        fused = self._features(agent,hand,joint,gripper,eef)
        values, _ = self.recurrent(fused.unsqueeze(1), None)
        return torch.tanh(self.action_head(values[:,0])), None, 1


def make_cached_eef_policy(config, *, arm, seed, public_state=None):
    base = make_windowed_policy(config, window=1, seed=seed, public_state=public_state)
    return CachedEEFPolicy(base, arm=arm, seed=seed)
