"""Fresh, injected LIBERO adapters with one bounded native reset attempt."""
from __future__ import annotations

import contextlib

import numpy as np

from src.environments.libero_adapter import LiberoEnvironmentAdapter


class BoundedControlEnvironment:
    def __init__(self, environment, ledger):
        self.control, self.ledger = environment, ledger
        self.reset_attempts = 0
        self.inside_reset = False
        self.zero_steps = 0

    def __getattr__(self, key):
        return getattr(self.control, key)

    def reset(self):
        if self.reset_attempts:
            raise RuntimeError('one native reset attempt per fresh adapter; retry forbidden')
        self.reset_attempts += 1
        # Deliberately avoid ControlEnv.reset's finally/continue retry loop.
        return self.ledger.call('native_reset', self.control.env.reset)

    def step(self, action):
        if self.inside_reset:
            if self.zero_steps >= 5 or not np.array_equal(np.asarray(action), np.zeros(7, np.float32)):
                raise ValueError('stabilization must be five exact zero controls')
            self.ledger.begin('zero_action'); self.zero_steps += 1
        value = self.ledger.call('native_control', self.control.step, action)
        if self.inside_reset:
            self.ledger.finish('zero_action')
        return value


def build_source_geometry_adapter(task, initial_row, ledger):
    if type(task) is not int or task not in [0, 1, 3, 4, 7]:
        raise PermissionError('source candidate task only')
    row = np.asarray(initial_row)
    if row.shape != (92,) or not np.isfinite(row).all():
        raise ValueError('exactly one authorized initial row required')
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    adapter = _build_injected_adapter(suite_name='libero_spatial', task_id=task, seed=20261022,
        image_size=128, max_episode_steps=300, stabilization_steps=5, initial_states=row[None].copy())
    adapter.environment = BoundedControlEnvironment(adapter.environment, ledger)
    return adapter


def reset_once(adapter):
    wrapper = adapter.environment
    wrapper.inside_reset = True
    try:
        value = adapter.reset(initial_state_index=0)
    finally:
        wrapper.inside_reset = False
    if wrapper.zero_steps != 5 or wrapper.reset_attempts != 1:
        raise ValueError('native reset/stabilization count differs')
    return value


@contextlib.contextmanager
def counted_native_physics(ledger):
    """Count integrations even if a constructor/reset creates a new MjSim."""
    from robosuite.utils.binding_utils import MjSim
    original, original_forward = MjSim.step, MjSim.forward

    def step(sim, *args, **kwargs):
        if getattr(ledger, 'observer_active', False):
            ledger.begin('forbidden_observer_native_call')
        return ledger.call('physics_step', original, sim, *args, **kwargs)

    def forward(sim, *args, **kwargs):
        if getattr(ledger, 'observer_active', False):
            ledger.begin('forbidden_observer_native_call')
        return original_forward(sim, *args, **kwargs)

    MjSim.step = step
    MjSim.forward = forward
    try:
        yield
    finally:
        MjSim.step = original
        MjSim.forward = original_forward
