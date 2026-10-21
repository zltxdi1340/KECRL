"""Run a bounded integration smoke through the KECRL LIBERO adapter."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.environments.libero_adapter import POLICY_OBSERVATION_KEYS, LiberoEnvironmentAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="libero_spatial")
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--initial-state-index", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stabilization-steps", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=300)
    args = parser.parse_args()

    if args.task_id is None:
        if args.suite != "libero_spatial":
            parser.error("all-task smoke currently has an explicit count only for libero_spatial")
        task_ids = range(10)
    else:
        task_ids = [args.task_id]

    results = []
    for task_id in task_ids:
        adapter = None
        try:
            adapter = LiberoEnvironmentAdapter(
                suite_name=args.suite,
                task_id=task_id,
                seed=args.seed,
                image_size=args.image_size,
                max_episode_steps=args.max_episode_steps,
                stabilization_steps=args.stabilization_steps,
            )
            observation, reset_info = adapter.reset(
                initial_state_index=args.initial_state_index
            )
            next_observation, reward, terminated, truncated, step_info = adapter.step(
                np.zeros(7, dtype=np.float32)
            )
            results.append({
                "task_id": task_id,
                "task_name": reset_info["task_name"],
                "observation_keys": list(observation),
                "observation_shapes": {
                    key: list(value.shape) for key, value in observation.items()
                },
                "observation_dtypes": {
                    key: str(value.dtype) for key, value in observation.items()
                },
                "rgb_nonblank": all(
                    int(next_observation[key].max()) > int(next_observation[key].min())
                    for key in ("agentview_rgb", "eye_in_hand_rgb")
                ),
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "terminal_reason": step_info["terminal_reason"],
                "status": "pass",
            })
        finally:
            if adapter is not None:
                adapter.close()

    expected_keys = list(POLICY_OBSERVATION_KEYS)
    passed = all(
        item["status"] == "pass"
        and item["observation_keys"] == expected_keys
        and item["rgb_nonblank"]
        for item in results
    )
    payload = {
        "check_type": "libero_kecrl_adapter_smoke",
        "formal_result": False,
        "status": "pass" if passed else "fail",
        "suite": args.suite,
        "task_count_checked": len(results),
        "seed": args.seed,
        "initial_state_index": args.initial_state_index,
        "stabilization_steps": args.stabilization_steps,
        "policy_steps": 1,
        "tasks": results,
    }
    print(json.dumps(payload, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
