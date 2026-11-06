"""Decompose audited source-support initialization/endpoint errors, without replay.

Only historical JSON and bound code/documentation are opened. Checkpoint pins
are inherited metadata, never paths to load. No learning or simulator imports.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/libero-source-internal-initialization-endpoint-target-error-readonly-diagnostic_v1.json"
DEFAULT_BINDING = ROOT / "configs/libero-source-internal-initialization-endpoint-target-error-execution-binding_v1.json"
DEFAULT_OUTPUT = Path("/tmp/libero-source-internal-initialization-endpoint-target-error-20261103-v1.json")
CONFIG_SHA256 = "db6a83698c4fe1caa4924a0e7bbd441cdf305b8b63dbcd8210fb457416441d79"
TASKS = (0, 1)
SEEDS = (20261022, 20261023, 20261024)
ARMS = ("A_original", "B_fomaml", "C_pooled_bc")
POINTS = (0, 500)
METRICS = ("translation_mse", "rotation_mse", "gripper_mse", "mse", "gripper_sign_error")
STEPS = {0: 2895, 1: 4059}
WINDOWS = {0: 304, 1: 422}
INPUT_NAMES = {"source_result", "source_audit", "source_design", "source_binding", "transfer", "learning",
               "dataset_manifest", "paired_trace_result", "paired_trace_audit", "gripper_mapping_audit"}
TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path):
    value = json.loads(Path(path).read_text())
    require(isinstance(value, dict), "expected JSON object")
    return value


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_spec(config):
    require(config.get("schema_version") == 1 and config.get("formal_result") is False and
            config.get("check_type") == "libero_source_internal_initialization_endpoint_target_error_readonly",
            "readonly nonformal definition differs")
    scope = config["scope"]
    expected = {
        "task_ids": list(TASKS), "training_seeds": list(SEEDS), "arms": list(ARMS),
        "adaptation_points": list(POINTS), "demonstration_role": "meta_support",
        "demonstration_indices": list(range(30)), "policy_memory_mode": "cnn_only",
        "policy_endpoint_updates": 500, "closed_loop_stabilization_steps": 5,
        "historical_unique_scored_steps_by_task": {str(k): v for k, v in STEPS.items()},
        "historical_windows_by_task": {str(k): v for k, v in WINDOWS.items()},
        "historical_sequence_length": 10, "historical_stride": 10, "historical_evaluation_batch_size": 8,
        "historical_metric_groups_reused": 36, "historical_model_prediction_rows_reused": 125172,
        "new_main_forward_batches": 0, "new_repeat_forward_batches": 0, "new_backward_calls": 0,
        "new_optimizer_updates": 0, "new_environment_calls": 0,
        "new_HDF5_RGB_initial_checkpoint_payload_reads": 0,
        "sealed_roles_accessed": False, "checkpoint_selected": False, "active_spt_updated": False,
        "knowledge_updated": False, "module_registered": False,
    }
    for key, value in expected.items():
        require(scope.get(key) == value and type(scope.get(key)) is type(value), f"scope differs: {key}")
    require(set(config["inputs"]) == INPUT_NAMES, "input roles differ")
    require(config["analysis"]["primary_metrics"] == list(METRICS[:3]) and
            config["analysis"]["secondary_metrics"] == list(METRICS[3:]), "metric priority differs")
    require(config["analysis"]["comparisons"] == [[ARMS[1], ARMS[0]], [ARMS[1], ARMS[2]]] and
            config["analysis"]["identity_absolute_tolerance"] == TOLERANCE, "comparison definition differs")


def finite_number(value, label):
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/non-numeric {label}")
    return float(value)


def metrics(record, expected_steps):
    require(type(record.get("valid_steps")) is int and record["valid_steps"] == expected_steps,
            "source-support valid-step count differs")
    output = {name: finite_number(record[name], name) for name in METRICS[:-1]}
    require(all(value >= 0 for value in output.values()), "negative target MSE")
    accuracy = finite_number(record["gripper_sign_accuracy"], "gripper sign accuracy")
    require(0 <= accuracy <= 1, "gripper accuracy outside0..1")
    output["gripper_sign_error"] = 1 - accuracy
    require(abs(output["mse"] - (3 * output["translation_mse"] + 3 * output["rotation_mse"] +
                                output["gripper_mse"]) / 7) <= TOLERANCE, "seven-dimensional MSE identity differs")
    return output


def stats(values):
    require(bool(values), "expected all three seeds, not an empty summary")
    values = [finite_number(v, "summary value") for v in values]
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "minimum": min(values), "maximum": max(values)}


def decompose(left0, left500, right0, right500):
    left0, left500, right0, right500 = [finite_number(v, "error") for v in (left0, left500, right0, right500)]
    require(min(left0, left500, right0, right500) >= 0, "negative target error")
    initial = left0 - right0
    left_change, right_change = left500 - left0, right500 - right0
    relative = left_change - right_change
    endpoint = left500 - right500
    residual = endpoint - (initial + relative)
    require(abs(residual) <= TOLERANCE, "initial/endpoint decomposition identity failed")
    return {"left0": left0, "left500": left500, "right0": right0, "right500": right500,
            "initial_gap": initial, "left_adaptation_change": left_change,
            "right_adaptation_change": right_change, "relative_adaptation_change": relative,
            "endpoint_gap": endpoint, "identity_residual": residual}


def checkpoint_identity(task, seed, arm, point, entry):
    return {"task_id": task, "training_seed": seed, "arm": arm, "optimizer_updates": point,
            **{name: entry[name] for name in ("path", "sha256", "weight_hash")}}


def index_closed_context(paired):
    rows = paired["episode_rows"]
    expected = {(t, s, a, i) for t in TASKS for s in SEEDS for a in ARMS for i in range(20)}
    seen, counts = set(), {(t, s, a): 0 for t in TASKS for s in SEEDS for a in ARMS}
    for row in rows:
        key = (row["task_id"], row["training_seed"], row["arm"], row["initial_state_index"])
        require(key in expected and key not in seen and type(row["success"]) is bool,
                "closed-loop context matrix/label differs")
        seen.add(key)
        counts[key[:3]] += int(row["success"])
    require(seen == expected, "closed-loop context must retain all360 episodes")
    return counts


def analyze(source, paired, config):
    validate_spec(config)
    require(source.get("status") == "complete" and source.get("formal_result") is False,
            "original source run was not complete diagnostic")
    require(source["actual_counts"]["probe_policies"] == 18 and
            source["actual_counts"]["probe_adaptation_optimizer_updates"] == 9000,
            "original adaptation budget differs")
    require(source["config"]["arms"] == list(ARMS) and source["config"]["training_seeds"] == list(SEEDS)
            and source["config"]["adaptation_updates"] == 500, "original model matrix differs")
    require(source["config"]["alignment"] == "causal_pre_action_t_ge_1" and
            source["config"]["policy_memory_mode"] == "cnn_only", "original input interface differs")
    for task in TASKS:
        records = [row for row in source["access_records"] if row["task_id"] == task and
                   row["demonstration_role"] == "meta_support"]
        require(bool(records), "source-support access evidence missing")
        for row in records:
            require(row["task_role"] == "source_spi" and row["demonstration_indices"] == list(range(30))
                    and row["window_count"] == WINDOWS[task] and row["sequence_length"] == 10
                    and row["stride"] == 10 and row["t0_excluded"] is True
                    and row["file_sha256_verified_before_access"] is True
                    and row["read_all_authorized_t_ge_1_payload"] is True,
                    "historical support role/window evidence differs")
    closed = index_closed_context(paired)
    rows, identities, seen = [], [], set()
    require(len(source["fold_results"]) == 2, "must retain both folds")
    for fold in source["fold_results"]:
        task = fold["probe"]
        require(task in TASKS and fold["source_update_tasks"] == [t for t in (0, 1, 3, 4, 7) if t != task],
                "source-internal fold boundary differs")
        require(len(fold["seed_results"]) == 3, "must retain all seeds")
        for seed_record in fold["seed_results"]:
            seed = seed_record["seed"]
            require(seed in SEEDS and set(seed_record["arms"]) == set(ARMS), "seed/arm matrix differs")
            require(len({entry["sampling_sha256"] for entry in seed_record["arms"].values()}) == 1,
                    "original adaptation batches were not paired")
            for arm in ARMS:
                key = (task, seed, arm)
                require(key not in seen, "duplicated source model")
                seen.add(key)
                entry = seed_record["arms"][arm]
                values = {str(point): metrics(entry["curve"][str(point)]["meta_support"], STEPS[task]) for point in POINTS}
                pins = [checkpoint_identity(task, seed, arm, point, entry["curve"][str(point)]) for point in POINTS]
                identities.extend(pins)
                rows.append({"task_id": task, "training_seed": seed, "arm": arm,
                             "historical_support_valid_steps": STEPS[task], "points": values,
                             "adaptation_change": {m: values["500"][m] - values["0"][m] for m in METRICS},
                             "checkpoint_identity_inherited": pins,
                             "recorded_candidate_closed_loop": {"successes": closed[key], "episodes": 20}})
    require(seen == {(t, s, a) for t in TASKS for s in SEEDS for a in ARMS}, "missing source model")
    require(identities == config["checkpoints"] and len({p["path"] for p in identities}) == 36,
            "inherited checkpoint identity differs")
    indexed = {(row["task_id"], row["training_seed"], row["arm"]): row for row in rows}
    pairs = []
    for task in TASKS:
        for seed in SEEDS:
            for left_arm, right_arm in config["analysis"]["comparisons"]:
                left, right = indexed[task, seed, left_arm], indexed[task, seed, right_arm]
                pairs.append({"task_id": task, "training_seed": seed, "left_arm": left_arm, "right_arm": right_arm,
                              "metrics": {m: decompose(left["points"]["0"][m], left["points"]["500"][m],
                                                        right["points"]["0"][m], right["points"]["500"][m]) for m in METRICS},
                              "recorded_candidate_closed_loop": {"left": left["recorded_candidate_closed_loop"],
                                                                 "right": right["recorded_candidate_closed_loop"]}})
    arm_groups, pair_groups = [], []
    for task in TASKS:
        for arm in ARMS:
            group = [indexed[task, seed, arm] for seed in SEEDS]
            arm_groups.append({"task_id": task, "arm": arm,
                               "points": {str(p): {m: stats([r["points"][str(p)][m] for r in group]) for m in METRICS} for p in POINTS},
                               "adaptation_change": {m: stats([r["adaptation_change"][m] for r in group]) for m in METRICS}})
        for left_arm, right_arm in config["analysis"]["comparisons"]:
            group = [r for r in pairs if r["task_id"] == task and r["left_arm"] == left_arm and r["right_arm"] == right_arm]
            require(len(group) == 3, "paired summary must retain three seeds")
            fields = ("initial_gap", "left_adaptation_change", "right_adaptation_change", "relative_adaptation_change", "endpoint_gap")
            pair_groups.append({"task_id": task, "left_arm": left_arm, "right_arm": right_arm,
                                "metrics": {m: {field: {**stats([r["metrics"][m][field] for r in group]),
                                                         "strict_negative_seed_count": sum(r["metrics"][m][field] < 0 for r in group)}
                                                for field in fields} for m in METRICS}})
    return {"model_rows": rows, "paired_rows": pairs, "arm_groups": arm_groups, "paired_groups": pair_groups,
            "counts": {"models": 18, "historical_metric_groups_reused": 36, "paired_seed_comparisons": 12,
                       "historical_unique_expert_steps": sum(STEPS.values()),
                       "historical_model_prediction_rows_reused": sum(STEPS.values()) * 18,
                       "closed_loop_episodes_reused_as_context": 360,
                       "new_policy_forward_batches": 0, "new_repeat_batches": 0, "new_prediction_rows": 0,
                       "new_backward_calls": 0, "new_optimizer_updates": 0, "new_environment_calls": 0,
                       "new_HDF5_RGB_initial_checkpoint_payload_reads": 0}}


def load_inputs(config_path, binding_path):
    config = load_json(config_path)
    validate_spec(config)
    require(sha256(config_path) == CONFIG_SHA256, "frozen config content hash differs")
    binding = load_json(binding_path)
    require(binding.get("formal_result") is False and binding.get("config_sha256") == CONFIG_SHA256,
            "execution binding differs")
    require(set(binding.get("files", {})) == {"analyzer", "tests", "protocol"}, "analysis code binding missing")
    require(resolve(binding["files"]["analyzer"]["path"]).resolve() == Path(__file__).resolve(), "bound analyzer differs")
    for name, pin in {**config["inputs"], **binding["files"]}.items():
        require(sha256(resolve(pin["path"])) == pin["sha256"], f"input/binding hash differs: {name}")
    inputs = {name: load_json(resolve(pin["path"])) for name, pin in config["inputs"].items()}
    source, source_audit = inputs["source_result"], inputs["source_audit"]
    require(source_audit["status"] == "passed" and source_audit["result_sha256"] == config["inputs"]["source_result"]["sha256"]
            and source_audit["checkpoint_files_and_tensor_hashes_verified"] == 108
            and source_audit["metric_groups_verified"] == 180, "historical audit differs")
    require(source["config"] == inputs["source_design"] and source["binding"] == inputs["source_binding"]
            and source["config_sha256"] == config["inputs"]["source_design"]["sha256"]
            and source["binding_sha256"] == config["inputs"]["source_binding"]["sha256"], "original run provenance differs")
    require(inputs["source_binding"]["config_sha256"] == source["config_sha256"], "historical code/config binding differs")
    # These pins point only to original code and protocol documents, never to models/data.
    for name, pin in inputs["source_binding"]["files"].items():
        require(sha256(resolve(pin["path"])) == pin["sha256"], f"historical evaluation code differs: {name}")
    transfer = inputs["transfer"]
    require(transfer["task_roles"]["source_spi"] == [0, 1, 3, 4, 7] and
            transfer["demonstration_indices_by_task_role"]["source_spi"]["meta_support"] == list(range(30)),
            "source support role changed")
    manifest = {row["task_id"]: row for row in inputs["dataset_manifest"]["files"]}
    historical_files = {row["task"]: row for row in source["source_files_verified"]}
    for task in TASKS:
        require(manifest[task]["sha256"] == historical_files[task]["sha256"], "historical source dataset identity differs")
    audit = inputs["paired_trace_audit"]
    require(audit["status"] == "passed" and audit["analysis_sha256"] == config["inputs"]["paired_trace_result"]["sha256"]
            and audit["episode_feature_rows_verified"] == 360, "paired trace audit differs")
    return config, binding, inputs


def run(config_path, binding_path, output):
    output = Path(output)
    require(not output.exists(), "output already exists; preserve historical artifacts")
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    config, binding, inputs = load_inputs(config_path, binding_path)
    result = analyze(inputs["source_result"], inputs["paired_trace_result"], config)
    result.update({"status": "complete", "formal_result": False, "read_only_arithmetic": True,
                   "scope": config["scope"], "analysis_definition": config["analysis"],
                   "provenance": {"config_path": str(Path(config_path).resolve()), "config_sha256": sha256(config_path),
                                  "binding_path": str(Path(binding_path).resolve()), "binding_sha256": sha256(binding_path),
                                  "inputs": config["inputs"], "bound_files": binding["files"],
                                  "checkpoint_identities_independently_reverified_this_run": False,
                                  "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()},
                   "runtime": {"started_at_utc": started_utc, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                               "elapsed_seconds": time.monotonic() - started, "python": sys.version.split()[0],
                               "execution": "CPU standard-library arithmetic; no model device or training runtime"},
                   "interpretation": ["The zero point is before probe adaptation, not before B/C source training.",
                                      "The error decomposition is algebraic; initialization and adaptation interact.",
                                      "Historical support fit is not independent query performance or rollout correctness.",
                                      "Gripper magnitude error is distinct from runtime sign-based control.",
                                      "No checkpoint/loss selection, SPT acceptance, Knowledge update or Module qualification."]})
    with output.open("x") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binding", type=Path, default=DEFAULT_BINDING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true", help="Verify JSON/code provenance without writing a result")
    args = parser.parse_args()
    if args.validate_only:
        load_inputs(args.config, args.binding)
        print(json.dumps({"status": "provenance_validated", "new_model_data_environment_calls": 0}))
    else:
        result = run(args.config, args.binding, args.output)
        print(json.dumps({"status": result["status"], "output": str(args.output), "sha256": sha256(args.output),
                          "counts": result["counts"], "runtime": result["runtime"]}))


if __name__ == "__main__":
    main()
