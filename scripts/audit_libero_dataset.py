"""Verify pinned LIBERO demonstrations without training or selecting a policy.

Run in the dedicated LIBERO environment. Large HDF5 files stay outside KECRL;
the small source manifest contains their expected sizes and SHA-256 hashes.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
from typing import Any

import numpy as np


DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "configs/libero_spatial_dataset_manifest_v1.json"
PROPRIO_DIMS = {"ee_ori": 3, "ee_pos": 3, "ee_states": 6, "gripper_states": 2, "joint_states": 7}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != 1 or manifest.get("formal_result") is not False:
        raise ValueError("expected a version-1, non-formal data manifest")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["dataset"]["revision"]):
        raise ValueError("dataset revision must be an immutable Git commit")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest["libero_revision"]):
        raise ValueError("LIBERO revision must be an immutable Git commit")
    files = manifest["files"]
    if not files or sorted(row["task_id"] for row in files) != list(range(len(files))):
        raise ValueError("task IDs must cover the suite once, without duplicates")
    if len({row["task_name"] for row in files}) != len(files):
        raise ValueError("duplicate task names")
    if manifest["dataset"]["expected_demo_count_per_task"] < 1:
        raise ValueError("expected demonstration count must be positive")
    for row in files:
        expected = PurePosixPath(manifest["suite"]) / (row["task_name"] + "_demo.hdf5")
        if row["path"] != str(expected) or ".." in expected.parts or expected.is_absolute():
            raise ValueError("dataset path does not match suite and task name")
        if row["size_bytes"] < 1 or not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("expected size and SHA-256 are required for every file")


def _text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def inspect_hdf5(path: Path, row: dict[str, Any], expected_demos: int) -> dict[str, Any]:
    import h5py

    lengths = []
    state_dims = set()
    action_min, action_max = float("inf"), float("-inf")
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        names = {"demo_" + str(index) for index in range(expected_demos)}
        if set(data.keys()) != names or int(data.attrs["num_demos"]) != expected_demos:
            raise ValueError("demonstration IDs/count do not match the manifest")
        if Path(_text(data.attrs["bddl_file_name"])).name != row["task_name"] + ".bddl":
            raise ValueError("HDF5 BDDL task identity mismatch")
        info = json.loads(_text(data.attrs["problem_info"]))
        if " ".join(info["language_instruction"].split()) != row["task_name"].replace("_", " "):
            raise ValueError("HDF5 language instruction does not match task identity")
        for index in range(expected_demos):
            demo = data["demo_" + str(index)]
            actions = np.asarray(demo["actions"])
            if actions.ndim != 2 or actions.shape[1] != 7 or actions.shape[0] < 1:
                raise ValueError("expected nonempty (T, 7) actions")
            length = len(actions)
            if not np.issubdtype(actions.dtype, np.floating) or not np.isfinite(actions).all():
                raise ValueError("actions must be finite floating-point values")
            if int(demo.attrs["num_samples"]) != length:
                raise ValueError("demonstration num_samples mismatch")
            for key in ("states", "robot_states", "rewards", "dones"):
                values = np.asarray(demo[key])
                if values.shape[0] != length or not np.isfinite(values).all():
                    raise ValueError(key + " has mismatched length or non-finite values")
                if key in ("rewards", "dones") and (values.shape != (length,) or not np.isin(values, [0, 1]).all()):
                    raise ValueError(key + " must be a binary vector")
            if demo["states"].ndim != 2 or demo["states"].shape[1] < 1:
                raise ValueError("states must be a nonempty matrix")
            initial = np.asarray(demo.attrs["init_state"])
            if initial.shape != demo["states"].shape[1:] or not np.isfinite(initial).all():
                raise ValueError("init_state does not match the state dimension")
            obs = demo["obs"]
            for key in ("agentview_rgb", "eye_in_hand_rgb"):
                if obs[key].shape != (length, 128, 128, 3) or obs[key].dtype != np.dtype("uint8"):
                    raise ValueError(key + " must be uint8 (T, 128, 128, 3)")
            for key, dimension in PROPRIO_DIMS.items():
                values = np.asarray(obs[key])
                if values.shape != (length, dimension) or not np.isfinite(values).all():
                    raise ValueError(key + " has an unexpected shape or non-finite values")
            lengths.append(length)
            state_dims.add(demo["states"].shape[1])
            action_min = min(action_min, float(actions.min()))
            action_max = max(action_max, float(actions.max()))
        if int(data.attrs["total"]) != sum(lengths):
            raise ValueError("dataset total does not match summed demonstration lengths")
    return {
        "demo_count": len(lengths),
        "transition_count": sum(lengths),
        "min_demo_steps": min(lengths),
        "max_demo_steps": max(lengths),
        "state_dimensions": sorted(state_dims),
        "action_dimension": 7,
        "action_min": action_min,
        "action_max": action_max,
        "rgb_shape": [128, 128, 3],
        "rgb_dtype": "uint8",
    }


def verify_task_map(manifest: dict[str, Any]) -> dict[str, Any]:
    with contextlib.redirect_stdout(sys.stderr):
        import libero.libero as libero_package
        from libero.libero import benchmark

        suite = benchmark.get_benchmark_dict()[manifest["suite"]](
            task_order_index=manifest["task_order_index"]
        )
    source_root = Path(libero_package.__file__).resolve().parents[2]
    revision = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != manifest["libero_revision"]:
        raise ValueError("installed LIBERO checkout does not match pinned revision")
    if suite.get_num_tasks() != len(manifest["files"]):
        raise ValueError("manifest task count does not match installed suite")
    for row in manifest["files"]:
        if suite.get_task(row["task_id"]).name != row["task_name"]:
            raise ValueError("manifest task ID does not match installed suite")
        if suite.get_task_demonstration(row["task_id"]) != row["path"]:
            raise ValueError("manifest demonstration path does not match installed suite")
    return {"status": "pass", "libero_revision": revision, "task_count": suite.get_num_tasks()}


def audit_dataset(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    validate_manifest(manifest)
    results = []
    for row in manifest["files"]:
        path = root / row["path"]
        item = {"task_id": row["task_id"], "task_name": row["task_name"], "path": row["path"]}
        try:
            if path.stat().st_size != row["size_bytes"]:
                raise ValueError("file size does not match pinned metadata")
            digest = sha256_file(path)
            if digest != row["sha256"]:
                raise ValueError("SHA-256 does not match pinned metadata")
            item.update({"size_bytes": path.stat().st_size, "sha256": digest})
            item.update(inspect_hdf5(path, row, manifest["dataset"]["expected_demo_count_per_task"]))
            item["status"] = "pass"
        except (OSError, KeyError, TypeError, ValueError) as error:
            item.update({"status": "fail", "error": str(error)})
        results.append(item)
    extras = sorted(
        str(path.relative_to(root))
        for path in (root / manifest["suite"]).glob("*.hdf5")
        if str(path.relative_to(root)) not in {row["path"] for row in manifest["files"]}
    )
    passed = all(item["status"] == "pass" for item in results) and not extras
    return {
        "check_type": "libero_pinned_dataset_audit",
        "formal_result": False,
        "status": "pass" if passed else "fail",
        "dataset": manifest["dataset"],
        "suite": manifest["suite"],
        "dataset_root": str(root.resolve()),
        "file_count_passed": sum(item["status"] == "pass" for item in results),
        "file_count_expected": len(results),
        "unexpected_files": extras,
        "total_size_bytes": sum(item.get("size_bytes", 0) for item in results),
        "total_demonstrations": sum(item.get("demo_count", 0) for item in results),
        "total_transitions": sum(item.get("transition_count", 0) for item in results),
        "files": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--verify-task-map", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error("output already exists; choose a new audit path")
    try:
        manifest = json.loads(args.manifest.read_text())
        result = audit_dataset(manifest, args.dataset_root)
        result["manifest_sha256"] = sha256_file(args.manifest)
        result["task_map"] = {"status": "not_checked"}
        if args.verify_task_map:
            result["task_map"] = verify_task_map(manifest)
    except (OSError, KeyError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        result = {"formal_result": False, "status": "fail", "error": str(error)}
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(payload)
    print(payload, end="")
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
