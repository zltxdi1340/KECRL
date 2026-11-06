"""Fixed candidate weights with two explicit causal motion-input contracts."""
from __future__ import annotations

import numpy as np

from experiments import libero_routed_history_runtime as previous

ARMS = ('candidate_real_history', 'candidate_repeat_current')
require = previous.require
saved = previous.saved
load_heads, load_encoder, encode_step = previous.load_heads, previous.load_encoder, previous.encode_step


def route_metadata(arm, step):
    require(arm in ARMS and type(step) is int and step >= 1, 'invalid input arm/policy step')
    startup = step <= 4
    history = list(range(step, max(0, step - 5), -1))
    real = arm == ARMS[0] and not startup
    return {'policy_step': step, 'history_length': min(step, 5), 'startup': startup,
        'history_policy_steps': history,
        'weight_route': 'shared_baseline_500' if startup else 'fixed_candidate_500',
        'motion_input_policy_steps': history if real else [step] * 5,
        'gripper_input_policy_steps': [step] * 5,
        'primitive_linear_calls': 2 if real else 1,
        'route': 'shared_baseline_startup' if startup else
            'candidate_motion_real_history_current_gripper' if real else
            'candidate_motion_repeat_current_current_gripper'}


class FixedHeadInputReadout(previous.FrozenHistoryReadout):
    """Share all weights/scalers/startup; alter only motion input from action five."""

    def __init__(self, heads, arm):
        require(arm in ARMS, 'invalid fixed-head input arm')
        require(set(heads) == set(previous.ARMS) and
            all(np.array_equal(heads[previous.ARMS[0]][k], heads[previous.ARMS[1]][k])
                for k in ('mean', 'scale')), 'paired support scalers differ')
        super().__init__(heads, previous.ARMS[1])
        self.input_arm = arm

    def act(self, feature, step, linear_observer=None):
        import torch
        require(type(step) is int and self.next_step is not None and step == self.next_step,
            'explicit reset/sequential policy step required')
        require(feature.shape == (256,) and feature.dtype == np.float32 and np.isfinite(feature).all(),
            'invalid public-derived feature')
        meta = route_metadata(self.input_arm, step)
        self.history.appendleft(feature.copy())
        self.next_step += 1
        current = (feature.astype(np.float64) - self.mean) / self.scale
        current = np.ascontiguousarray((np.tile(current, 5) / np.sqrt(5)).astype(np.float32)[None])

        def linear(model, values):
            if linear_observer:
                linear_observer('attempt')
            logits = model(torch.from_numpy(values))
            if linear_observer:
                linear_observer('complete')
            return logits

        with torch.inference_mode():
            if meta['startup']:
                logits = linear(self.models[previous.ARMS[0]], current)
            else:
                require(len(self.history) == 5, 'candidate history not mature')
                head = self.models[previous.ARMS[1]]
                current_logits = linear(head, current)
                if self.input_arm == ARMS[0]:
                    z = (np.stack(self.history).astype(np.float64) - self.mean) / self.scale
                    hist = np.ascontiguousarray((z.reshape(1, 1280) / np.sqrt(5)).astype(np.float32))
                    logits = torch.cat((linear(head, hist)[:, :6], current_logits[:, 6:]), dim=1)
                else:
                    logits = current_logits
            action = torch.tanh(logits)[0].numpy().copy()
        require(action.shape == (7,) and action.dtype == np.float32 and np.isfinite(action).all()
            and (np.abs(action) <= 1).all(), 'invalid fixed-head runtime action')
        return action, meta
