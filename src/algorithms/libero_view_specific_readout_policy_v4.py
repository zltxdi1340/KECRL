"""Opt-in view-specific readout; no downloads or changed policy defaults."""
from __future__ import annotations

import torch
from torch import nn

from src.algorithms.libero_spatial_readout_policy_v3 import (
    SpatialSoftmax2D, SpatialCoordinateReadout, VisualReadoutPolicyV3,
)
from src.algorithms.libero_windowed_resnet18_policy import make_windowed_policy

ARMS = ('flat_zero_xyz', 'view_specific_spatial_zero_xyz')
VIEWS = ('agentview', 'eye_in_hand')
VIEW_SEED_OFFSETS = {'agentview': 9400001, 'eye_in_hand': 9500001}
READOUT_PARAMETERS = {ARMS[0]: 1048704, ARMS[1]: 1041856}
TRAINABLE_PARAMETERS = {ARMS[0]: 1514120, ARMS[1]: 1507272}


class ViewCoordinateReadoutV4(SpatialCoordinateReadout):
    """512x4x4 -> 256 heatmaps -> 512 coordinates -> 608 -> 128."""
    def __init__(self):
        nn.Module.__init__(self)
        self.channel_projection = nn.Conv2d(512, 256, 1, bias=False)
        self.spatial = SpatialSoftmax2D()
        self.coordinate_mlp = nn.Sequential(nn.Linear(512, 608), nn.ReLU(), nn.Linear(608, 128))
        self._last_entropy = None


class ViewSpecificReadoutPolicyV4(VisualReadoutPolicyV3):
    """Two independent readouts, shared frozen backbone, internally zero XYZ."""
    def __init__(self, initialized, *, arm, seed):
        if arm not in ARMS:
            raise ValueError('unknown v4 view-specific readout arm')
        # Retain the existing flat control and all downstream initialization.
        super().__init__(initialized, arm=ARMS[0], seed=seed)
        self.readout_arm = arm
        if arm == ARMS[1]:
            branches = {}
            for view in VIEWS:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed + VIEW_SEED_OFFSETS[view])
                    branches[view] = ViewCoordinateReadoutV4()
            self.visual_encoder.projection = nn.ModuleDict(branches)

    def _project_pair(self, agent, hand):
        if self.readout_arm == ARMS[0]:
            return super()._project_pair(agent, hand)
        values, entropies = [], []
        for view, features in zip(VIEWS, (agent, hand)):
            branch = self.visual_encoder.projection[view]
            output = branch(features) if self.compute_guard is None else self.compute_guard(
                'spatial_readout_' + view, branch, features)
            values.append(output)
            entropies.append(branch.entropy_snapshot())
        self._entropy_views = tuple(entropies)
        return values


def make_view_specific_readout_policy_v4(config, *, arm, seed, public_state=None):
    """Construct from admitted in-memory weights, or blank synthetic architecture."""
    base = make_windowed_policy(config, window=1, seed=seed, public_state=public_state)
    return ViewSpecificReadoutPolicyV4(base, arm=arm, seed=seed)
