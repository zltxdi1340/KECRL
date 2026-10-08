"""Non-formal cross-process replay audit for Crafter collection ordering."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


def _digest(value) -> str:
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    elif not isinstance(value, (bytes, bytearray)):
        value = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(value).hexdigest()


def _run_trace(seed: int, actions: list[int], stable: bool) -> list[dict]:
    from src.environments.crafter_adapter import CrafterEnvironmentAdapter
    env = CrafterEnvironmentAdapter(seed=seed, length=len(actions) + 1)
    observation = env.reset()
    trace = [{"observation": _digest(observation)}]
    for action in actions:
        observation, reward, done, info = env.step(int(action))
        trace.append({
            "observation": _digest(observation),
            "reward": float(reward),
            "done": bool(done),
            "inventory": info.get("inventory"),
        })
        if done:
            break
    env.close()
    return trace


def _worker(seed: int, actions: list[int], stable: bool, output: Path) -> None:
    import crafter

    if stable:
        native_balance = crafter.Env._balance_object

        def stable_balance(self, chunk, objs, *args, **kwargs):
            def order_key(obj):
                try:
                    return int(self._world._obj_map[tuple(obj.pos)])
                except (KeyError, TypeError, ValueError):
                    return (obj.__class__.__name__, tuple(int(v) for v in obj.pos))
            return native_balance(self, chunk, sorted(objs, key=order_key), *args, **kwargs)

        crafter.Env._balance_object = stable_balance

    traces = [_run_trace(seed, actions, stable) for _ in range(3)]
    output.write_text(json.dumps({"seed": seed, "stable": stable, "traces": traces}, indent=2) + "\n")


def _run_worker(seed: int, actions: list[int], stable: bool, hash_seed: int, output: Path) -> None:
    command = [
        sys.executable, __file__, "--worker", "--seed", str(seed),
        "--actions", json.dumps(actions), "--stable", str(int(stable)),
        "--worker-output", str(output),
    ]
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(hash_seed)
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1]) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, check=True, env=environment)


def _first_divergence(traces: list[list[dict]]) -> int | None:
    if not traces:
        return None
    for index in range(min(len(trace) for trace in traces)):
        if any(trace[index] != traces[0][index] for trace in traces[1:]):
            return index
    if any(len(trace) != len(traces[0]) for trace in traces[1:]):
        return min(len(trace) for trace in traces)
    return None


def run(output_path: str) -> dict:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    actions_by_seed = {}
    rows = []
    seeds = [850000, 850001, 850002]
    hash_seeds = [0, 1, 2]
    for offset, seed in enumerate(seeds):
        rng = np.random.default_rng(870000 + offset)
        actions = [int(action) for action in rng.integers(0, 7, size=256)]
        actions_by_seed[str(seed)] = actions
        for stable in (False, True):
            traces = []
            for hash_seed in hash_seeds:
                trace_path = output / f"seed_{seed}_{'stable' if stable else 'native'}_hash_{hash_seed}.json"
                _run_worker(seed, actions, stable, hash_seed, trace_path)
                worker_traces = json.loads(trace_path.read_text())["traces"]
                traces.extend(worker_traces)
            rows.append({
                "seed": seed,
                "stable_order": stable,
                "processes": len(hash_seeds),
                "repeats_per_process": 3,
                "trace_lengths": [len(trace) for trace in traces],
                "first_divergence_index": _first_divergence(traces),
                "identical_public_trace": _first_divergence(traces) is None,
            })
    result = {
        "formal_result": False,
        "status": "crafter_replay_order_audit",
        "seeds": seeds,
        "hash_seeds": hash_seeds,
        "action_streams": actions_by_seed,
        "rows": rows,
        "native_cross_process_identical": all(
            row["identical_public_trace"] for row in rows if not row["stable_order"]
        ),
        "stable_cross_process_identical": all(
            row["identical_public_trace"] for row in rows if row["stable_order"]
        ),
        "installed_package_modified": False,
    }
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/crafter_replay_order_audit_20261008_v1")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--actions")
    parser.add_argument("--stable", type=int, default=0)
    parser.add_argument("--worker-output")
    args = parser.parse_args()
    if args.worker:
        _worker(args.seed, json.loads(args.actions), bool(args.stable), Path(args.worker_output))
    else:
        print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
