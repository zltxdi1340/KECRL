"""Opt-in public cached EEF extension; no native pose query or forced update."""
from __future__ import annotations

import contextlib
import sys
import numpy as np

from .libero_adapter import LiberoEnvironmentAdapter


class CachedEEFLiberoAdapter(LiberoEnvironmentAdapter):
    def _policy_observation(self, native):
        result = super()._policy_observation(native)
        xyz = np.asarray(native['robot0_eef_pos'])
        if xyz.shape != (3,) or not np.issubdtype(xyz.dtype,np.floating) or not np.isfinite(xyz).all():
            raise ValueError('cached public robot0_eef_pos must be finite XYZ')
        result['eef_pos'] = xyz.astype(np.float32,copy=True)
        return result


def build_cached_eef_adapter(*, suite_name, task_id, seed, image_size, max_episode_steps, stabilization_steps, initial_states):
    with contextlib.redirect_stdout(sys.stderr):
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
    task = suite.get_task(task_id)
    environment = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
        camera_heights=image_size,camera_widths=image_size,camera_names=['agentview','robot0_eye_in_hand'],
        has_renderer=False,has_offscreen_renderer=True,use_camera_obs=True,
        horizon=max_episode_steps+stabilization_steps)
    return CachedEEFLiberoAdapter(suite_name=suite_name,task_id=task_id,seed=seed,image_size=image_size,
        max_episode_steps=max_episode_steps,stabilization_steps=stabilization_steps,environment=environment,
        initial_states=initial_states,task_name=task.name,language_instruction=task.language)
