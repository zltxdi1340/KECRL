"""H4 joint supervision, k1 feedback; cached causal windows only."""
from __future__ import annotations

import torch
from torch import nn

from src.algorithms.libero_bc_policy import masked_behavior_cloning_loss
from src.algorithms.libero_spatial_readout_policy_v3 import VisualReadoutPolicyV3
from src.algorithms.libero_windowed_resnet18_policy import make_windowed_policy

ARMS = ('h1_spatial_control', 'joint_h4_k1_spatial')
HORIZON = 4
FUTURE_SEED_OFFSETS = (9700001, 9701001, 9702001)
REGISTERED_PARAMETERS = 1504541
TRAINABLE_PARAMETERS = {ARMS[0]: 1499144, ARMS[1]: 1504541}


def chunk_targets(actions, valid_mask, starts, demo_lengths):
    """No cross-window reads: a[s+t+h] requires both window and demo validity."""
    if actions.ndim != 3 or actions.shape[-1] != 7 or not actions.is_floating_point():
        raise ValueError('floating [B,T,7] cached actions required')
    b,t,_ = actions.shape
    if not 1 <= t <= 10 or valid_mask.shape != (b,t) or valid_mask.dtype != torch.bool:
        raise ValueError('bounded prefix mask required')
    if starts.shape != (b,) or demo_lengths.shape != (b,) or starts.dtype != torch.int64 or demo_lengths.dtype != torch.int64:
        raise ValueError('integer start/demo lengths required')
    if any(x.device != actions.device for x in [valid_mask,starts,demo_lengths]):
        raise ValueError('mask metadata device differs')
    positions = torch.arange(t,device=actions.device)
    expected = starts[:,None]+positions[None,:] < demo_lengths[:,None]
    if bool((starts < 1).any()) or bool((starts >= demo_lengths).any()) or not torch.equal(valid_mask,expected):
        raise ValueError('causal demo/window prefix mask inconsistent')
    if not torch.isfinite(actions).all() or torch.abs(actions).max() > 1:
        raise ValueError('finite normalized cached actions required')
    offsets = torch.arange(HORIZON,device=actions.device)
    indices = positions[:,None]+offsets[None,:]
    inside_window = indices < t
    safe = indices.clamp(max=t-1)
    inside_demo = starts[:,None,None]+indices[None,:,:] < demo_lengths[:,None,None]
    future = inside_window[None,:,:] & inside_demo & valid_mask[:,safe]
    anchor = valid_mask
    targets = actions[:,safe,:]
    targets = torch.where((anchor[:,:,None] & future)[:,:,:,None],targets,torch.zeros_like(targets))
    return targets,anchor,future


def supervision_mask(anchor, future, arm):
    if arm not in ARMS or anchor.dtype != torch.bool or future.dtype != torch.bool or future.shape != (*anchor.shape,HORIZON):
        raise ValueError('arm and double masks required')
    active = torch.tensor([True,arm==ARMS[1],arm==ARMS[1],arm==ARMS[1]],device=anchor.device)
    return anchor[:,:,None] & future & active


def chunk_masked_mse(predictions, targets, anchor, future, arm):
    if predictions.shape != targets.shape or predictions.shape != (*anchor.shape,HORIZON,7):
        raise ValueError('matching [B,T,4,7] chunk predictions/targets required')
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError('nonfinite chunk loss tensor')
    mask = supervision_mask(anchor,future,arm)
    # Preserve the legacy H0 reduction/layout in the control arm exactly.
    if arm == ARMS[0]:
        return masked_behavior_cloning_loss(predictions[:,:,0].contiguous(),targets[:,:,0].contiguous(),mask[:,:,0].contiguous())
    b,t,_,_ = predictions.shape
    return masked_behavior_cloning_loss(predictions.reshape(b,t*HORIZON,7),targets.reshape(b,t*HORIZON,7),mask.reshape(b,t*HORIZON))


class ActionChunkPolicyV5(VisualReadoutPolicyV3):
    """Identical registered four-head state; control freezes/skips future heads."""
    def __init__(self, initialized, *, arm, seed):
        if arm not in ARMS:
            raise ValueError('unknown v5 supervision arm')
        super().__init__(initialized,arm='spatial_coordinates_zero_xyz',seed=seed)
        self.readout_arm = arm
        heads = []
        for offset in FUTURE_SEED_OFFSETS:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+offset)
                heads.append(nn.Linear(self.action_head.in_features,7))
        self.future_action_heads = nn.ModuleList(heads)
        if arm == ARMS[0]: self.future_action_heads.requires_grad_(False)

    def _project_pair(self, agent, hand):
        values,entropies = [],[]
        for features in [agent,hand]:
            fn = self.visual_encoder.projection
            values.append(fn(features) if self.compute_guard is None else self.compute_guard('spatial_readout',fn,features))
            entropies.append(fn.entropy_snapshot())
        self._entropy_views = tuple(entropies)
        return values

    def _actions(self, values):
        fn = self.action_head
        first = torch.tanh(fn(values) if self.compute_guard is None else self.compute_guard('action_head_0',fn,values))
        chunks = [first]
        if self.readout_arm == ARMS[1]:
            for i,fn in enumerate(self.future_action_heads,1):
                chunks.append(torch.tanh(fn(values) if self.compute_guard is None else self.compute_guard('action_head_'+str(i),fn,values)))
        else: chunks.extend(torch.zeros_like(first) for _ in range(3))
        result = torch.stack(chunks,dim=2)
        if not torch.isfinite(result).all(): raise ValueError('nonfinite chunk actions')
        return result

    def forward_feature_sequence(self, agent, hand, joint, gripper):
        self._entropy_views = None; self._completed_forward = False
        if (agent.ndim != 5 or hand.shape != agent.shape or tuple(agent.shape[2:]) != (512,4,4)
                or agent.requires_grad or hand.requires_grad or not torch.isfinite(agent).all() or not torch.isfinite(hand).all()):
            raise ValueError('detached finite paired frozen features required')
        b,t = agent.shape[:2]
        if not 1 <= b*t <= 80 or joint.shape != (b,t,7) or gripper.shape != (b,t,2):
            raise ValueError('bounded chunk feature shapes differ')
        a,h = self._project_pair(agent.reshape(-1,512,4,4),hand.reshape(-1,512,4,4))
        fused = self._fuse_zero(a.reshape(b,t,128),h.reshape(b,t,128),joint,gripper)
        values,_ = self.recurrent(fused,None)
        result = self._actions(values); self._completed_forward = True
        return result,None

    def forward_sequence(self, agentview_rgb, eye_in_hand_rgb, joint_states, gripper_states, hidden=None):
        self._entropy_views = None; self._completed_forward = False
        if hidden is not None or agentview_rgb.ndim != 5 or eye_in_hand_rgb.shape != agentview_rgb.shape or tuple(agentview_rgb.shape[2:]) != (3,128,128):
            raise ValueError('paired causal frame images without history required')
        b,t = agentview_rgb.shape[:2]
        if not 1 <= b*t <= 80 or joint_states.shape != (b,t,7) or gripper_states.shape != (b,t,2):
            raise ValueError('bounded chunk pixel shapes differ')
        fused = self._features(agentview_rgb.reshape(-1,3,128,128),eye_in_hand_rgb.reshape(-1,3,128,128),
                               joint_states.reshape(-1,7),gripper_states.reshape(-1,2)).reshape(b,t,-1)
        values,_ = self.recurrent(fused,None)
        result = self._actions(values); self._completed_forward = True
        return result,None

    def forward_window_step(self, agent, hand, joint, gripper, context=None):
        self._entropy_views = None; self._completed_forward = False
        if context is not None: raise ValueError('k1 chunk policy cannot retain history or queued actions')
        chunks,_ = self.forward_sequence(agent.unsqueeze(1),hand.unsqueeze(1),joint.unsqueeze(1),gripper.unsqueeze(1))
        return chunks[:,0,0].contiguous(),None,1


def make_action_chunk_policy_v5(config, *, arm, seed, public_state=None):
    base = make_windowed_policy(config,window=1,seed=seed,public_state=public_state)
    return ActionChunkPolicyV5(base,arm=arm,seed=seed)
