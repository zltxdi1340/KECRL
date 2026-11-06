"""Role-restricted causal LIBERO windows excluding unreconstructed t=0.

Targets are action[t], inputs are stored post-action observations[t-1].
No simulator initial state or reconstructed initial observation is required.
"""
from __future__ import annotations

import numpy as np

from .libero_dataset import LiberoSequenceWindow, RoleRestrictedLiberoSequenceDataset


class CausalT1LiberoSequenceDataset(RoleRestrictedLiberoSequenceDataset):
    def __init__(self, *args, **kwargs):
        if "observation_action_alignment" in kwargs or "causal_initial_observations" in kwargs:
            raise ValueError("t>=1 causal alignment cannot be overridden")
        super().__init__(*args, **kwargs)
        self.observation_action_alignment = "causal_pre_action_t_ge_1"

    def _index_windows(self):
        windows = []
        data = self._open()["data"]
        for index in self.access.demonstration_indices:
            demo = data[f"demo_{index}"]
            length = int(demo["actions"].shape[0])
            if length < 2:
                raise ValueError("causal t>=1 demo must have at least two actions")
            for start in range(1, length, self.stride):
                windows.append(LiberoSequenceWindow(index, start, min(self.sequence_length, length-start)))
        return tuple(windows)

    def _observation_slice(self, demo, key, window):
        if window.start < 1:
            raise ValueError("causal t>=1 window must not access t0")
        return np.asarray(demo["obs"][key][window.start-1:window.start+window.valid_length-1])

    def load_initial_states(self):
        raise PermissionError("causal t>=1 learning never reads simulator initial states")

    @property
    def access_record(self):
        return {**super().access_record, "t0_excluded": True, "simulator_state_access": False}
