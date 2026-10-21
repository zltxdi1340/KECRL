"""Thin, fail-closed adapter for the external LIBERO environment.

The policy-facing observation contains the two official RGB views and robot
proprioception. Simulator state and object poses remain private to LIBERO.
"""
from __future__ import annotations

import contextlib
import sys
from typing import Any, Mapping

import numpy as np


POLICY_OBSERVATION_KEYS = (
    "agentview_rgb",
    "eye_in_hand_rgb",
    "joint_states",
    "gripper_states",
)
_NATIVE_OBSERVATION_KEYS = {
    "agentview_rgb": "agentview_image",
    "eye_in_hand_rgb": "robot0_eye_in_hand_image",
    "joint_states": "robot0_joint_pos",
    "gripper_states": "robot0_gripper_qpos",
}


class LiberoEnvironmentAdapter:
    """Expose a stable KECRL boundary around one LIBERO task instance."""

    def __init__(
        self,
        *,
        suite_name: str = "libero_spatial",
        task_id: int = 0,
        seed: int = 0,
        image_size: int = 128,
        max_episode_steps: int = 300,
        stabilization_steps: int = 5,
        environment: Any | None = None,
        initial_states: Any | None = None,
        task_name: str | None = None,
        language_instruction: str | None = None,
    ) -> None:
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id < 0:
            raise ValueError("task_id must be a non-negative integer")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
            raise ValueError("image_size must be a positive integer")
        if max_episode_steps <= 0 or stabilization_steps < 0:
            raise ValueError("episode and stabilization step counts are invalid")

        if environment is None:
            try:
                with contextlib.redirect_stdout(sys.stderr):
                    from libero.libero import benchmark
                    from libero.libero.envs import OffScreenRenderEnv

                    suite_type = benchmark.get_benchmark_dict()[suite_name]
                    suite = suite_type(task_order_index=0)
                if task_id >= suite.get_num_tasks():
                    raise ValueError(
                        f"task_id must be smaller than {suite.get_num_tasks()} for {suite_name}"
                    )
                task = suite.get_task(task_id)
                environment = OffScreenRenderEnv(
                    bddl_file_name=suite.get_task_bddl_file_path(task_id),
                    camera_heights=image_size,
                    camera_widths=image_size,
                    camera_names=["agentview", "robot0_eye_in_hand"],
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    use_camera_obs=True,
                    horizon=max_episode_steps + stabilization_steps,
                )
                initial_states = suite.get_task_init_states(task_id)
                task_name = task.name
                language_instruction = task.language
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "LIBERO is not installed; use the isolated LIBERO runtime"
                ) from exc
        elif initial_states is None or task_name is None or language_instruction is None:
            raise ValueError(
                "injected environments require initial_states, task_name, and language_instruction"
            )

        if initial_states is None or len(initial_states) < 1:
            raise ValueError("LIBERO task must provide at least one fixed initial state")
        self.environment = environment
        self.initial_states = initial_states
        self.suite_name = str(suite_name)
        self.task_id = task_id
        self.task_name = str(task_name)
        self.language_instruction = str(language_instruction)
        self.seed = seed
        self.image_size = image_size
        self.max_episode_steps = int(max_episode_steps)
        self.stabilization_steps = int(stabilization_steps)
        self.action_dimension = 7
        self.action_low = np.full(self.action_dimension, -1.0, dtype=np.float32)
        self.action_high = np.full(self.action_dimension, 1.0, dtype=np.float32)
        self._check_native_action_spec()
        self._observation: dict[str, np.ndarray] | None = None
        self._episode_done = False
        self._success = False
        self._terminal_reason: str | None = None
        self._closed = False
        self._step_count = 0
        self._initial_state_index: int | None = None

    def _check_native_action_spec(self) -> None:
        spec = getattr(self.environment, "action_spec", None)
        if spec is None:
            spec = getattr(getattr(self.environment, "env", None), "action_spec", None)
        if spec is None:
            return
        if not isinstance(spec, (tuple, list)) or len(spec) != 2:
            raise ValueError("unexpected LIBERO action specification")
        low, high = (np.asarray(bound) for bound in spec)
        if low.shape != (7,) or high.shape != (7,):
            raise ValueError("LIBERO action specification must be seven-dimensional")
        if not np.allclose(low, self.action_low) or not np.allclose(high, self.action_high):
            raise ValueError("LIBERO action bounds must be [-1, 1]")

    def _policy_observation(self, native: Mapping[str, Any]) -> dict[str, np.ndarray]:
        missing = set(_NATIVE_OBSERVATION_KEYS.values()) - set(native)
        if missing:
            raise ValueError(f"LIBERO observation is missing required keys: {sorted(missing)}")
        result = {
            public: np.asarray(native[source]).copy()
            for public, source in _NATIVE_OBSERVATION_KEYS.items()
        }
        for key in ("agentview_rgb", "eye_in_hand_rgb"):
            if result[key].shape != (self.image_size, self.image_size, 3):
                raise ValueError(f"unexpected {key} shape: {result[key].shape}")
            if result[key].dtype != np.uint8:
                raise ValueError(f"{key} must have uint8 dtype")
        for key, shape in (("joint_states", (7,)), ("gripper_states", (2,))):
            if result[key].shape != shape or not np.issubdtype(result[key].dtype, np.number):
                raise ValueError(f"unexpected {key} shape or dtype")
            if not np.isfinite(result[key]).all():
                raise ValueError(f"{key} contains non-finite values")
            result[key] = result[key].astype(np.float32)
        return result

    @staticmethod
    def _copy_observation(observation: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        return {key: value.copy() for key, value in observation.items()}

    def reset(self, *, initial_state_index: int = 0) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        if self._closed:
            raise RuntimeError("LIBERO environment is closed")
        if (
            isinstance(initial_state_index, bool)
            or not isinstance(initial_state_index, int)
            or not 0 <= initial_state_index < len(self.initial_states)
        ):
            raise ValueError("initial_state_index is outside the fixed initial-state set")
        self.environment.seed(self.seed)
        self.environment.reset()
        native = self.environment.set_init_state(self.initial_states[initial_state_index])
        for _ in range(self.stabilization_steps):
            native, _reward, native_done, _info = self.environment.step(
                np.zeros(self.action_dimension, dtype=np.float32)
            )
            if native_done or self.environment.check_success():
                raise RuntimeError("fixed initial state succeeds during stabilization")
        self._observation = self._policy_observation(native)
        self._episode_done = False
        self._success = False
        self._terminal_reason = None
        self._step_count = 0
        self._initial_state_index = initial_state_index
        return self._copy_observation(self._observation), self._public_info(
            success=False, terminal_reason=None
        )

    def _validate_action(self, action: Any) -> np.ndarray:
        values = np.asarray(action)
        if values.shape != (self.action_dimension,) or not np.issubdtype(values.dtype, np.number):
            raise ValueError("LIBERO action must be a numeric vector with shape (7,)")
        if not np.isfinite(values).all():
            raise ValueError("LIBERO action must contain only finite values")
        if np.any(values < self.action_low) or np.any(values > self.action_high):
            raise ValueError("LIBERO action values must lie in [-1, 1]")
        return values.astype(np.float32)

    def _public_info(self, *, success: bool, terminal_reason: str | None) -> dict[str, Any]:
        return {
            "suite": self.suite_name,
            "task_id": self.task_id,
            "task_name": self.task_name,
            "language_instruction": self.language_instruction,
            "initial_state_index": self._initial_state_index,
            "step_count": self._step_count,
            "success": bool(success),
            "terminal_reason": terminal_reason,
        }

    def step(
        self, action: Any
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._observation is None:
            raise RuntimeError("reset must be called before step")
        if self._episode_done:
            raise RuntimeError("LIBERO episode has terminated or truncated")
        native, reward, native_done, _native_info = self.environment.step(
            self._validate_action(action)
        )
        self._step_count += 1
        self._observation = self._policy_observation(native)
        success = bool(self.environment.check_success())
        if bool(native_done) != success:
            raise RuntimeError("LIBERO native done signal disagrees with success predicate")
        terminated = success
        truncated = self._step_count >= self.max_episode_steps and not terminated
        self._episode_done = terminated or truncated
        reason = "success" if terminated else "time_limit" if truncated else None
        self._success = success
        self._terminal_reason = reason
        return (
            self._copy_observation(self._observation),
            float(reward),
            terminated,
            truncated,
            self._public_info(success=success, terminal_reason=reason),
        )

    def current_observation(self) -> dict[str, np.ndarray]:
        if self._observation is None:
            raise RuntimeError("reset must be called before requesting an observation")
        return self._copy_observation(self._observation)

    def state(self) -> dict[str, Any]:
        """Return episode bookkeeping without exposing simulator or object state."""
        return self._public_info(
            success=self._success,
            terminal_reason=self._terminal_reason,
        )

    def close(self) -> None:
        if not self._closed:
            self.environment.close()
            self._closed = True
