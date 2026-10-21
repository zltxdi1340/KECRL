"""Run a bounded LIBERO environment and data-interface feasibility check.

This script is intentionally a smoke check.  It does not train a policy and
its output must not be interpreted as a benchmark result.

Run it with the dedicated LIBERO environment, for example::

    MUJOCO_GL=egl LIBERO_CONFIG_PATH=$HOME/.libero \
      python scripts/check_libero_environment.py --suite libero_spatial

The LIBERO checkout and its datasets are external dependencies.  They are
kept outside this repository so that the source tree does not contain large
robotics assets or demonstrations.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        default="libero_spatial",
        choices=("libero_spatial", "libero_object", "libero_goal", "libero_90", "libero_10"),
    )
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--demo",
        type=Path,
        default=None,
        help="Optional HDF5 demonstration to validate through the simulator.",
    )
    return parser.parse_args()


def _check_task(
    suite: Any,
    task_id: int,
    *,
    width: int,
    height: int,
    steps: int,
    seed: int,
    demo_path: Optional[Path],
) -> dict[str, Any]:
    from libero.libero.envs import OffScreenRenderEnv

    task = suite.get_task(task_id)
    states = suite.get_task_init_states(task_id)
    bddl = suite.get_task_bddl_file_path(task_id)
    env = None
    try:
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=height,
            camera_widths=width,
            camera_names=["agentview"],
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            horizon=max(steps, 1) + 1,
        )
        env.seed(seed)
        env.reset()
        obs = env.set_init_state(states[0])
        rewards = []
        done = False
        for _ in range(steps):
            obs, reward, done, _info = env.step(np.zeros(7, dtype=np.float32))
            rewards.append(float(reward))
        rgb = np.asarray(obs["agentview_image"])
        result = {
            "task_id": task_id,
            "task_name": task.name,
            "language": task.language,
            "bddl_exists": Path(bddl).exists(),
            "init_state_shape": list(states.shape),
            "rgb_shape": list(rgb.shape),
            "rgb_dtype": str(rgb.dtype),
            "rgb_min": int(rgb.min()),
            "rgb_max": int(rgb.max()),
            "rgb_mean": float(rgb.mean()),
            "rewards": rewards,
            "done": bool(done),
            "success_after_zero_action": bool(env.check_success()),
            "status": "pass",
        }
        if demo_path is not None:
            import h5py

            expected_name = f"{task.name}_demo.hdf5"
            if demo_path.name != expected_name:
                raise ValueError(
                    f"demo {demo_path.name!r} does not match task {task.name!r}"
                )
            with h5py.File(demo_path, "r") as handle:
                demo_name = sorted(handle["data"].keys())[0]
                final_state = handle["data"][demo_name]["states"][-1]
            env.set_init_state(final_state)
            result["demo_final_state_success"] = bool(env.check_success())
            _obs, demo_reward, demo_done, _info = env.step(np.zeros(7, dtype=np.float32))
            result["demo_replay_reward"] = float(demo_reward)
            result["demo_replay_done"] = bool(demo_done)
        return result
    finally:
        if env is not None:
            env.close()


def _check_demo(path: Path) -> dict[str, Any]:
    import h5py

    if not path.exists():
        return {"path": str(path), "status": "missing"}
    with h5py.File(path, "r") as handle:
        demo_names = sorted(handle["data"].keys())
        first = handle["data"][demo_names[0]]
        return {
            "path": str(path),
            "status": "pass",
            "demo_count": len(demo_names),
            "first_demo": demo_names[0],
            "states_shape": list(first["states"].shape),
            "actions_shape": list(first["actions"].shape),
            "observation_keys": sorted(first["obs"].keys()),
        }


def main() -> None:
    args = _parse_args()
    # LIBERO prints task-order and missing-path notices to stdout.  Keep the
    # machine-readable result on stdout and route those upstream notices to
    # stderr instead.
    with contextlib.redirect_stdout(sys.stderr):
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[args.suite](task_order_index=0)
        task_ids = [args.task_id] if args.task_id is not None else list(range(suite.get_num_tasks()))
        if any(task_id < 0 or task_id >= suite.get_num_tasks() for task_id in task_ids):
            raise SystemExit(f"task id must be in [0, {suite.get_num_tasks() - 1}]")
        demo_task_id = None
        if args.demo is not None:
            suffix = "_demo.hdf5"
            if not args.demo.name.endswith(suffix):
                raise SystemExit(f"demo filename must end in {suffix!r}: {args.demo.name}")
            demo_stem = args.demo.name[: -len(suffix)]
            matches = [
                task_id
                for task_id in range(suite.get_num_tasks())
                if suite.get_task(task_id).name == demo_stem
            ]
            if len(matches) != 1:
                raise SystemExit(f"cannot map demo filename to one task: {args.demo.name}")
            demo_task_id = matches[0]
            if args.task_id is not None and args.task_id != demo_task_id:
                raise SystemExit(
                    f"demo belongs to task {demo_task_id}, not requested task {args.task_id}"
                )

        tasks = [
            _check_task(
                suite,
                task_id,
                width=args.width,
                height=args.height,
                steps=args.steps,
                seed=args.seed,
                demo_path=args.demo if task_id == demo_task_id else None,
            )
            for task_id in task_ids
        ]
    result = {
        "check_type": "libero_environment_smoke",
        "formal_result": False,
        "suite": args.suite,
        "task_count_checked": len(tasks),
        "tasks": tasks,
        "demo": _check_demo(args.demo) if args.demo is not None else None,
        "libero_config_path": os.environ.get("LIBERO_CONFIG_PATH"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
