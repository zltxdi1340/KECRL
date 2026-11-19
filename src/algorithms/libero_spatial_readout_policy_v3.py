"""Opt-in matched-budget visual readouts on frozen 512x4x4 features.

No downloads, payload loading, research execution, or changed defaults.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from src.algorithms.libero_cached_eef_policy import CachedEEFPolicy
from src.algorithms.libero_windowed_resnet18_policy import make_windowed_policy

ARMS = ('flat_zero_xyz', 'spatial_coordinates_zero_xyz')
READOUT_PARAMETERS = {ARMS[0]: 1048704, ARMS[1]: 1033728}
TRAINABLE_PARAMETERS = {ARMS[0]: 1514120, ARMS[1]: 1499144}
MAX_IMAGE_ROWS = 80


class SpatialSoftmax2D(nn.Module):
    """Fixed-temperature expectations of feature-grid coordinates, x then y.

    x increases with column, y with row; both range from -1 to +1. These
    are feature-grid coordinates, not calibrated image or world positions.
    Entropy is diagnostic only and is detached from the training graph.
    """
    def __init__(self, *, temperature=1.0):
        super().__init__()
        if type(temperature) not in (int, float) or temperature != 1.0:
            raise ValueError('v3 admits only fixed temperature 1.0')
        self.register_buffer('temperature', torch.tensor(1.0, dtype=torch.float64))
        axis = torch.arange(4, dtype=torch.float64) * (2.0 / 3.0) - 1.0
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
        self.register_buffer('grid_x', xx.reshape(16))
        self.register_buffer('grid_y', yy.reshape(16))

    def forward(self, scores, *, return_entropy=False):
        if (scores.ndim != 4 or tuple(scores.shape[-2:]) != (4,4)
                or min(scores.shape[:2]) < 1 or scores.dtype not in (torch.float32,torch.float64)
                or not torch.isfinite(scores).all()):
            raise ValueError('finite floating [N,K,4,4] response maps required')
        if self.temperature.requires_grad or float(self.temperature) != 1.0:
            raise ValueError('temperature must remain a nontrainable 1.0 buffer')
        n, k = scores.shape[:2]
        logp = F.log_softmax(scores.reshape(n,k,16), dim=-1)
        probabilities = logp.exp()
        x = (probabilities * self.grid_x.to(scores)).sum(-1)
        y = (probabilities * self.grid_y.to(scores)).sum(-1)
        coordinates = torch.stack((x,y),dim=-1).reshape(n,2*k)
        # entr(0)=0; saturation never introduces 0*log(0) NaNs in telemetry.
        entropy = torch.special.entr(probabilities.detach()).sum(-1)
        if not torch.isfinite(coordinates).all() or not torch.isfinite(entropy).all():
            raise ValueError('nonfinite spatial expectations or entropy')
        return (coordinates,entropy) if return_entropy else coordinates


class SpatialCoordinateReadout(nn.Module):
    """512x4x4 -> 256 heatmaps -> 512 coordinates -> 1408 -> 128.

    The final layer is Linear, as specified by v3; there is no final ReLU.
    """
    def __init__(self):
        super().__init__()
        self.channel_projection = nn.Conv2d(512,256,1,bias=False)
        self.spatial = SpatialSoftmax2D()
        self.coordinate_mlp = nn.Sequential(nn.Linear(512,1408),nn.ReLU(),nn.Linear(1408,128))
        self._last_entropy = None

    def forward(self, features):
        self._last_entropy = None
        if (features.ndim != 4 or tuple(features.shape[1:]) != (512,4,4)
                or not 1 <= features.shape[0] <= MAX_IMAGE_ROWS
                or features.dtype not in (torch.float32,torch.float64)
                or not torch.isfinite(features).all()):
            raise ValueError('finite [1..80,512,4,4] frozen features required')
        coordinates, entropy = self.spatial(self.channel_projection(features),return_entropy=True)
        result = self.coordinate_mlp(coordinates)
        if not torch.isfinite(result).all():
            raise ValueError('nonfinite spatial readout')
        self._last_entropy = entropy.detach().clone()
        return result

    def entropy_snapshot(self):
        if self._last_entropy is None:
            raise ValueError('a completed spatial readout is required before diagnostics')
        return self._last_entropy.detach().clone()


class VisualReadoutPolicyV3(CachedEEFPolicy):
    """Frame-only policy; XYZ is generated as zero, never supplied by sensors."""
    def __init__(self, initialized, *, arm, seed):
        if arm not in ARMS:
            raise ValueError('unknown v3 readout arm')
        cfg = initialized.config
        if (cfg.visual_embedding_dim,cfg.fusion_dim,cfg.recurrent_hidden_dim,cfg.action_dimension) != (128,256,256,7):
            raise ValueError('v3 fixes visual128/fusion256/frame256/action7')
        super().__init__(initialized,arm='zero_xyz_mlp',seed=seed)
        self.readout_arm = arm
        self._entropy_views = None
        self._completed_forward = False
        if arm == ARMS[1]:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed+9200001)
                self.visual_encoder.projection = SpatialCoordinateReadout()

    def _project_pair(self, agent, hand):
        values, entropies = [], []
        for features in (agent,hand):
            projection = self.visual_encoder.projection
            key = 'spatial_readout' if self.readout_arm == ARMS[1] else 'flat_readout'
            output = projection(features) if self.compute_guard is None else self.compute_guard(key,projection,features)
            values.append(output)
            if self.readout_arm == ARMS[1]:entropies.append(projection.entropy_snapshot())
        self._entropy_views = tuple(entropies) if entropies else None
        return values

    def _fuse_zero(self, agent, hand, joint, gripper):
        zeros = torch.zeros((*joint.shape[:-1],3),dtype=joint.dtype,device=joint.device)
        return self._fuse(agent,hand,joint,gripper,zeros)

    def _features(self, agent, hand, joint, gripper):
        frozen = []
        for rgb in (agent,hand):
            fn = self.visual_encoder.encode_frozen
            frozen.append(fn(rgb) if self.compute_guard is None else self.compute_guard('frozen_encoder',fn,rgb))
        return self._fuse_zero(*self._project_pair(*frozen),joint,gripper)

    def forward_sequence(self, agentview_rgb, eye_in_hand_rgb, joint_states, gripper_states, hidden=None):
        self._entropy_views = None
        self._completed_forward = False
        if (hidden is not None or agentview_rgb.ndim != 5 or eye_in_hand_rgb.shape != agentview_rgb.shape
                or tuple(agentview_rgb.shape[2:]) != (3,128,128)):
            raise ValueError('paired RGB sequences without history required')
        b,t = agentview_rgb.shape[:2]
        if not 1 <= b*t <= MAX_IMAGE_ROWS or joint_states.shape != (b,t,7) or gripper_states.shape != (b,t,2):
            raise ValueError('bounded sequence shapes differ')
        fused = self._features(agentview_rgb.reshape(-1,3,128,128),eye_in_hand_rgb.reshape(-1,3,128,128),
                               joint_states.reshape(-1,7),gripper_states.reshape(-1,2)).reshape(b,t,-1)
        values,_ = self.recurrent(fused,None)
        action = torch.tanh(self.action_head(values))
        if not torch.isfinite(action).all():raise ValueError('nonfinite v3 actions')
        self._completed_forward = True
        return action,None

    def forward_feature_sequence(self, agent, hand, joint, gripper):
        self._entropy_views = None
        self._completed_forward = False
        if (agent.ndim != 5 or hand.shape != agent.shape or tuple(agent.shape[2:]) != (512,4,4)
                or agent.requires_grad or hand.requires_grad
                or agent.dtype not in (torch.float32,torch.float64) or hand.dtype != agent.dtype
                or not torch.isfinite(agent).all() or not torch.isfinite(hand).all()):
            raise ValueError('detached paired [B,T,512,4,4] convolution cache required')
        b,t = agent.shape[:2]
        if not 1 <= b*t <= MAX_IMAGE_ROWS or joint.shape != (b,t,7) or gripper.shape != (b,t,2):
            raise ValueError('bounded cached sequence shapes differ')
        a,h = self._project_pair(agent.reshape(-1,512,4,4),hand.reshape(-1,512,4,4))
        fused = self._fuse_zero(a.reshape(b,t,128),h.reshape(b,t,128),joint,gripper)
        values,_ = self.recurrent(fused,None)
        action = torch.tanh(self.action_head(values))
        if not torch.isfinite(action).all():raise ValueError('nonfinite v3 actions')
        self._completed_forward = True
        return action,None

    def forward_window_step(self, agent, hand, joint, gripper, context=None):
        self._completed_forward = False
        self._entropy_views = None
        if context is not None:
            raise ValueError('v3 frame MLP cannot receive history')
        values,_ = self.forward_sequence(agent.unsqueeze(1),hand.unsqueeze(1),joint.unsqueeze(1),gripper.unsqueeze(1))
        return values[:,0],None,1

    def entropy_summary(self, valid_mask=None):
        if self.readout_arm == ARMS[0]:
            return None  # No fabricated zero entropy for a control without heatmaps.
        if not self._completed_forward or self._entropy_views is None:
            raise ValueError('completed current forward required for entropy telemetry')
        a,h = self._entropy_views
        if valid_mask is None:
            mask = torch.ones(a.shape[0],dtype=torch.bool,device=a.device)
        else:
            if valid_mask.dtype != torch.bool or valid_mask.device != a.device or valid_mask.numel() != a.shape[0]:
                raise ValueError('entropy mask must cover exactly the current image rows')
            mask = valid_mask.reshape(-1)
        rows = int(mask.sum())
        if rows < 1:
            raise ValueError('at least one valid row required for entropy')
        count = rows*256
        a_sum,h_sum = float(a[mask].double().sum()),float(h[mask].double().sum())
        mean = (a_sum+h_sum)/(2*count)
        if not -1e-6 <= mean <= math.log(16)+1e-6:
            raise ValueError('spatial entropy outside numerical range')
        return dict(valid_rows=rows,heatmaps_per_view=count,agent_entropy_nats_sum=a_sum,
            hand_entropy_nats_sum=h_sum,agent_entropy_nats_mean=a_sum/count,hand_entropy_nats_mean=h_sum/count,
            entropy_nats_mean=mean,normalized_entropy_mean=mean/math.log(16),temperature=1.0)


def make_visual_readout_policy_v3(config, *, arm, seed, public_state=None):
    """Only an already-admitted in-memory public state or blank test/reload state."""
    base = make_windowed_policy(config,window=1,seed=seed,public_state=public_state)
    return VisualReadoutPolicyV3(base,arm=arm,seed=seed)
