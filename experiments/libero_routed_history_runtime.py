"""Experimental frozen-feature runtime; no environment or training side effects."""
from collections import deque

import numpy as np

from experiments import run_libero_task1_b_support_query_history_readout as saved

ARMS = ('current_all7', 'history_motion_current_gripper')
require = saved.require


def load_heads(inputs, seed):
    heads = {}
    for arm in ARMS:
        pin = inputs[f'{seed}_{arm}']
        require(saved.sha(pin['path']) == pin['sha256'], 'readout archive hash differs')
        with np.load(saved.resolve(pin['path']), allow_pickle=False) as z:
            require(z['updates'].tolist() == [0, 100, 250, 500], 'readout endpoint differs')
            heads[arm] = {'weight': z['weights'][-1], 'bias': z['biases'][-1],
                          'mean': z['scaler_mean'], 'scale': z['scaler_scale']}
    require(all(np.array_equal(heads[ARMS[0]][k], heads[ARMS[1]][k]) for k in ('mean', 'scale')), 'paired support scalers differ')
    return heads


class FrozenHistoryReadout:
    """Explicit episode reset and shared baseline startup for the first four actions."""

    def __init__(self, heads, arm):
        import torch
        require(arm in ARMS and set(heads) == set(ARMS), 'invalid arm/head bundle')
        self.arm, self.models = arm, {}
        self.mean, self.scale = [np.array(heads[ARMS[0]][k], copy=True) for k in ('mean', 'scale')]
        require(self.mean.shape == self.scale.shape == (256,) and np.isfinite(self.mean).all() and
                np.isfinite(self.scale).all() and (self.scale > 0).all(), 'invalid support scaler')
        for name, values in heads.items():
            w, b = values['weight'], values['bias']
            require(w.shape == (7, 1280) and b.shape == (7,) and w.dtype == b.dtype == np.float32 and
                    np.isfinite(w).all() and np.isfinite(b).all(), 'invalid frozen head')
            model = torch.nn.Linear(1280, 7, device='cpu', dtype=torch.float32)
            with torch.no_grad():
                model.weight.copy_(torch.from_numpy(w.copy()))
                model.bias.copy_(torch.from_numpy(b.copy()))
            self.models[name] = model.eval().requires_grad_(False)
        self.history = deque(maxlen=5)
        self.next_step = None

    def reset(self):
        self.history.clear()
        self.next_step = 1

    def act(self, feature, step):
        import torch
        require(type(step) is int and self.next_step is not None and step == self.next_step, 'explicit reset/sequential policy step required')
        require(feature.shape == (256,) and feature.dtype == np.float32 and np.isfinite(feature).all(), 'invalid public-derived feature')
        self.history.appendleft(feature.copy())
        self.next_step += 1
        current = ((feature.astype(np.float64) - self.mean) / self.scale)
        current = np.ascontiguousarray((np.tile(current, 5) / np.sqrt(5)).astype(np.float32)[None])
        startup = step <= 4
        with torch.inference_mode():
            if self.arm == ARMS[0] or startup:
                action = torch.tanh(self.models[ARMS[0]](torch.from_numpy(current)))
                route = 'shared_baseline_startup' if startup else 'baseline_current'
            else:
                require(len(self.history) == 5, 'candidate history not mature')
                z = (np.stack(self.history).astype(np.float64) - self.mean) / self.scale
                hist = np.ascontiguousarray((z.reshape(1, 1280) / np.sqrt(5)).astype(np.float32))
                head = self.models[ARMS[1]]
                logits = torch.cat((head(torch.from_numpy(hist))[:, :6], head(torch.from_numpy(current))[:, 6:]), dim=1)
                action = torch.tanh(logits)
                route = 'candidate_motion_history_current_gripper'
        value = action[0].numpy().copy()
        require(value.shape == (7,) and np.isfinite(value).all() and (np.abs(value) <= 1).all(), 'invalid runtime action')
        return value, {'policy_step': step, 'history_length': len(self.history), 'startup': startup, 'route': route,
                       'history_policy_steps': list(range(step, max(0, step - 5), -1))}

    def state_hash(self):
        values = {name + '_' + key: tensor.detach().numpy() for name, model in self.models.items() for key, tensor in model.state_dict().items()}
        values.update(scaler_mean=self.mean, scaler_scale=self.scale)
        return saved.parent.array_hash(values)


def load_encoder(checkpoint, learning):
    import torch
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
    require(saved.sha(checkpoint['path']) == checkpoint['sha256'], 'encoder file differs')
    state = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    require(saved.tensor_digest(state) == checkpoint['weight_hash'], 'encoder tensor hash differs')
    config = LiberoBCPolicyConfig.from_mapping({**learning['policy'], 'memory_mode': 'cnn_only'})
    policy = make_deterministic_libero_policy(config)
    policy.load_state_dict(state, strict=True)
    return policy.eval().requires_grad_(False)


def encode_step(policy, observation):
    import torch
    captured = []
    handle = policy.action_head.register_forward_pre_hook(lambda _m, args: captured.append(args[0].detach().clone()))
    try:
        image = lambda key: torch.from_numpy(np.ascontiguousarray(observation[key])).permute(2, 0, 1).unsqueeze(0)
        prop = lambda key: torch.from_numpy(np.ascontiguousarray(observation[key])).unsqueeze(0)
        with torch.inference_mode():
            action, hidden = policy.forward_step(image('agentview_rgb'), image('eye_in_hand_rgb'), prop('joint_states'), prop('gripper_states'), hidden=None)
        require(hidden is None and len(captured) == 1 and captured[0].shape == (1, 1, 256), 'runtime pre-head interface differs')
        feature, original = captured[0][0, 0].numpy().copy(), action[0].numpy().copy()
        require(np.isfinite(feature).all() and np.isfinite(original).all(), 'nonfinite encoder output')
        return feature, original
    finally:
        handle.remove()
