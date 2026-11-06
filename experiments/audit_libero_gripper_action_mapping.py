"""Audit the installed Panda command mapping without importing a simulator.

Only pinned source text and an existing task-7 prediction archive are read.
The isolated formatter is executed on dummy command accumulators, not physics.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def isolated_formatter(path: Path):
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "PandaGripper")
    methods = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"format_action", "speed", "dof"}]
    if len(methods) != 3:
        raise ValueError("installed Panda formatter interface differs")
    isolated = ast.Module(body=[ast.ClassDef(name="IsolatedPanda", bases=[], keywords=[], body=methods, decorator_list=[])], type_ignores=[])
    namespace = {"np": np}
    exec(compile(ast.fix_missing_locations(isolated), str(path), "exec"), namespace)
    return namespace["IsolatedPanda"]


def formatted(cls, value: float, previous: np.ndarray) -> np.ndarray:
    instance = cls()
    instance.current_action = previous.copy()
    return instance.format_action(np.array([value], dtype=np.float32)).copy()


def run(config_path: Path, output: Path) -> dict:
    started = time.monotonic()
    config = json.loads(config_path.read_text())
    if config.get("formal_result") is not False or config.get("new_environment_or_model_calls") != 0:
        raise ValueError("audit must remain read-only")
    if output.exists():
        raise FileExistsError(output)
    inputs = {}
    for name, pin in config["inputs"].items():
        path = Path(pin["path"])
        path = path if path.is_absolute() else ROOT / path
        if sha256(path) != pin["sha256"]:
            raise ValueError(f"pinned input differs: {name}")
        inputs[name] = path
    sources = {name: path.read_text() for name, path in inputs.items() if path.suffix == ".py"}
    required = {
        "adapter": ["self._validate_action(action)", "return values.astype(np.float32)"],
        "wrapper": ["return self.env.step(action)", 'gripper_types="default"'],
        "panda_robot": ['return "PandaGripper"'],
        "single_arm": ["gripper_action = action[self.controller.control_dim :]", "self.grip_action(gripper=self.gripper, gripper_action=gripper_action)"],
        "manipulator": ["gripper.format_action(gripper_action)", "bias + weight * gripper_action_actual"],
        "formatter": ["np.sign(action)", "self.current_action +", "return 0.01"],
    }
    for name, snippets in required.items():
        if any(snippet not in sources[name] for snippet in snippets):
            raise ValueError(f"command chain changed: {name}")
    cls = isolated_formatter(inputs["formatter"])
    states = [np.array(v, dtype=np.float64) for v in [(0, 0), (-1, 1), (1, -1), (-.99, .99), (.99, -.99)]]
    probes = []
    for state in states:
        for a, b in [(.1, .9), (-.1, -.9), (0., 0.)]:
            left, right = formatted(cls, a, state), formatted(cls, b, state)
            if not np.array_equal(left, right):
                raise RuntimeError("same-sign amplitude altered the isolated command")
            probes.append({"previous": state.tolist(), "actions": [a, b], "output": left.tolist(), "equal": True})
    rows = []
    with np.load(inputs["task7_predictions"], allow_pickle=False) as data:
        conditions = data["condition_names"].tolist()
        target = data["target_actions"][:, 6]
        if len(target) != 4188 or np.any(data["action_indices"] < 1):
            raise ValueError("task7 legal cached scope differs")
        for seed in config["training_seeds"]:
            predictions = data[f"predictions_seed{seed}"]
            original = predictions[conditions.index("identity_uint8"), :, 6]
            for condition in config["conditions"]:
                changed = predictions[conditions.index(condition), :, 6]
                if not np.isfinite(changed).all() or not np.isfinite(original).all():
                    raise ValueError("nonfinite cached action")
                same = np.sign(original) == np.sign(changed)
                correct = same & (np.sign(original) == np.sign(target))
                # At identical command history, all same-sign commands must agree.
                checked = 0
                for state in states:
                    left = np.clip(state[None] + np.array([-1., 1.])[None] * .01 * np.sign(original)[:, None], -1., 1.)
                    right = np.clip(state[None] + np.array([-1., 1.])[None] * .01 * np.sign(changed)[:, None], -1., 1.)
                    if not np.array_equal(left[same], right[same]):
                        raise RuntimeError("equal-sign command check failed")
                    checked += int(same.sum())
                rows.append({"seed": seed, "condition": condition, "frames": len(target),
                             "same_sign": int(same.sum()), "both_correct_sign": int(correct.sum()),
                             "different_sign": int((~same).sum()), "same_history_command_checks": checked})
    result = {"status": "complete", "formal_result": False, "check_type": "libero_gripper_action_mapping_source_audit",
              "configuration": config, "config_sha256": sha256(config_path), "script_sha256": sha256(Path(__file__)),
              "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "mapping": "clip(previous + [-1,+1]*0.01*sign(action[6]), -1,+1); then actuator-range affine map",
              "synthetic_formatter_probes": probes, "cached_comparisons": rows,
              "same_sign_equivalence_requires_identical_accumulator_history": True,
              "different_arm_motion_or_prior_sign_history_can_change_physics": True,
              "MSE_amplitude_change_is_not_physical_grip_strength": True,
              "new_environment_model_optimizer_HDF5_initial_state_checkpoint_calls": 0,
              "elapsed_seconds": time.monotonic()-started, "python": sys.version.split()[0], "numpy": np.__version__}
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(json.dumps({"status": result["status"], "elapsed_seconds": result["elapsed_seconds"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
