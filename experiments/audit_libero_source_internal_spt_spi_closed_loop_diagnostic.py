"""Independently audit the fixed source-internal closed-loop artifacts.

No Policy is instantiated or evaluated. Initial-state containers and HDF5 are
never opened. Only saved public traces, metadata and CPU endpoint tensors are
read; failed experiments retain their failed status and have no primary rate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TASKS = [0, 1]
SEEDS = [20261022, 20261023, 20261024]
ARMS = ["A_original", "B_fomaml", "C_pooled_bc"]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def episode_key(ep):
    return ep["task_id"], ep["training_seed"], ep["arm"], ep["initial_state_index"]


def independent_summary(episodes):
    require(len(episodes) == 360, "primary summary requires all360 episodes")
    index = {episode_key(e): e for e in episodes}
    expected = {(t, s, a, i) for t in TASKS for s in SEEDS for a in ARMS for i in range(20)}
    require(len(index) == 360 and set(index) == expected, "primary pair keys are missing or duplicated")
    result = {}
    for task in TASKS:
        seed_results = {}
        for seed in SEEDS:
            totals = {a: sum(int(index[task, seed, a, i]["success"]) for i in range(20)) for a in ARMS}
            comparisons = {}
            for left, right in [("B_fomaml", "A_original"), ("B_fomaml", "C_pooled_bc")]:
                counts = {"both_success": 0, "left_only_success": 0, "right_only_success": 0, "both_failure": 0}
                for i in range(20):
                    l, r = index[task, seed, left, i]["success"], index[task, seed, right, i]["success"]
                    if l and r: counts["both_success"] += 1
                    elif l: counts["left_only_success"] += 1
                    elif r: counts["right_only_success"] += 1
                    else: counts["both_failure"] += 1
                delta = totals[left] - totals[right]
                comparisons[left + "_minus_" + right] = {"paired_outcomes": counts, "success_count_delta": delta, "success_rate_delta": delta / 20}
            seed_results[str(seed)] = {"episodes_per_arm": 20, "success_counts": totals,
                "success_rates": {a: n / 20 for a, n in totals.items()}, "comparisons": comparisons}
        totals = {a: sum(v["success_counts"][a] for v in seed_results.values()) for a in ARMS}
        result[str(task)] = {"seed_results": seed_results, "episodes_per_arm": 60, "success_counts": totals,
            "success_rates": {a: n / 60 for a, n in totals.items()},
            "strict_B_improvement_all_three_seeds": {a: all(v["success_counts"]["B_fomaml"] > v["success_counts"][a]
                for v in seed_results.values()) for a in ["A_original", "C_pooled_bc"]}}
    return result


def audit_public_episode(ep, endpoint):
    require(ep["status"] == "complete" and type(ep["success"]) is bool, "episode is not a completed diagnostic")
    require(ep["initial_state_role"] == "candidate_nonregression" and ep["stabilization_steps"] == 5 and
            ep["environment_seed"] == 20261022 and ep["initial_state_index"] in range(20), "episode scope drift")
    require(ep["checkpoint_sha256"] == endpoint["sha256"] and ep["checkpoint_weight_hash"] == endpoint["weight_hash"], "episode endpoint drift")
    n = ep["steps"]
    require(type(n) is int and 1 <= n <= 300 and len(ep["stepwise_public_state_trace"]) == n + 1, "episode step boundary differs")
    require(ep["terminal_reason"] == ("success" if ep["success"] else "time_limit") and (ep["success"] or n == 300), "invalid success/time-limit boundary")
    require(math.isfinite(ep["reward_sum"]) and len(ep["reset_observation_sha256"]) == 64, "invalid reward or reset digest")
    trace = ep["stepwise_public_state_trace"]
    require([row["step"] for row in trace] == list(range(n + 1)), "public trace is not contiguous")
    joints = np.asarray([row["joint_state"] for row in trace], dtype=np.float64)
    gripper = np.asarray([row["gripper_state"] for row in trace], dtype=np.float64)
    actions = np.asarray([row["action"] for row in trace[1:]], dtype=np.float64)
    require(joints.shape == (n + 1, 7) and gripper.shape == (n + 1, 2) and actions.shape == (n, 7), "public trace shape differs")
    require(np.isfinite(joints).all() and np.isfinite(gripper).all() and np.isfinite(actions).all() and
            np.all(np.abs(actions) <= 1), "invalid public action/state trace")
    joint_offsets, gripper_offsets = joints - joints[0], gripper - gripper[0]
    jnorm = np.linalg.norm(joint_offsets, axis=1)
    gnorm = np.linalg.norm(gripper_offsets, axis=1)
    observed_j = np.asarray([row["joint_offset_l2_from_start"] for row in trace])
    observed_g = np.asarray([row["gripper_offset_l2_from_start"] for row in trace])
    require(np.all(np.abs(jnorm - observed_j) <= 1e-12) and np.all(np.abs(gnorm - observed_g) <= 1e-12), "public offset norm reconstruction differs")
    for i, row in enumerate(trace[1:], 1):
        require(np.array_equal(np.asarray(row["joint_offset_from_start"]), joint_offsets[i]) and
                np.array_equal(np.asarray(row["gripper_offset_from_start"]), gripper_offsets[i]), "public offset vector reconstruction differs")
    require(np.array_equal(np.asarray(ep["last_attempted_action"]), actions[-1]), "last action provenance differs")
    return {"policy_actions": n, "trace_rows": n + 1,
            "max_joint_offset_l2": float(jnorm.max()), "max_gripper_offset_l2": float(gnorm.max())}


def audit(result_path, output):
    require(not output.exists(), "audit output already exists")
    started = time.perf_counter()
    result = json.loads(result_path.read_text())
    require(result.get("formal_result") is False and result["status"] in {"complete", "failed"}, "unexpected experiment status")
    prov = result["provenance"]
    require(sha(prov["config_path"]) == prov["config_sha256"] and sha(prov["binding_path"]) == prov["binding_sha256"], "protocol provenance changed")
    cfg, binding = [json.loads(Path(prov[k]).read_text()) for k in ["config_path", "binding_path"]]
    require(binding["config_sha256"] == prov["config_sha256"], "binding/config lineage mismatch")
    for name, pin in {**cfg["inputs"], **binding["files"]}.items():
        path = Path(pin["path"])
        path = path if path.is_absolute() else ROOT / path
        require(sha(path) == pin["sha256"], f"input/code pin changed: {name}")
    require(cfg["scope"]["episodes"] == 360 and cfg["scope"]["checkpoint_count"] == 18 and
            cfg["scope"]["initial_state_indices"] == list(range(20)) and cfg["scope"]["stabilization_steps"] == 5,
            "scientific scope changed")
    endpoints = {(p["task_id"], p["training_seed"], p["arm"]): p for p in cfg["endpoints"]}
    require(len(endpoints) == 18 and all(p["optimizer_updates"] == 500 for p in endpoints.values()), "endpoint count/budget changed")
    import torch
    weight_checks = []
    for key, pin in endpoints.items():
        require(sha(pin["path"]) == pin["sha256"], "endpoint file changed")
        state = torch.load(pin["path"], map_location="cpu", weights_only=True)
        h = hashlib.sha256()
        for name, value in state.items():
            require(bool(torch.isfinite(value).all()), "nonfinite endpoint tensor")
            h.update(name.encode()); h.update(value.contiguous().numpy().tobytes())
        require(h.hexdigest() == pin["weight_hash"], "endpoint tensor hash changed")
        weight_checks.append({"task_id": key[0], "seed": key[1], "arm": key[2], "weight_hash": h.hexdigest()})
    rows_decoded = 0
    for task, access in result["initial_state_access"].items():
        require(int(task) in TASKS and access["candidate_rows_numerically_decoded"] == list(range(20)) and
                access["qualification_or_reserved_rows_decoded"] == 0 and access["used_torch_or_pickle_deserialization"] is False,
                "initial-state role or deserialization boundary violated")
        require(access["decoded_candidate_bytes"] == 20 * 92 * 8 and
                access["utf8_payload_bytes_read"] < access["encoded_payload_bytes_total"], "initial-state prefix did not stop at candidate boundary")
        rows_decoded += 20
    require(sha(prov["event_log"]["path"]) == prov["event_log"]["sha256"], "event log changed")
    events = [json.loads(line) for line in Path(prov["event_log"]["path"]).read_text().splitlines()]
    calls = result["actual_calls"]
    runtime = result["runtime"]
    require(runtime["new_optimizer_updates"] == runtime["new_backward_calls"] == runtime["new_hdf5_payload_reads"] == 0,
            "fixed inference unexpectedly trained or read demonstrations")
    require(calls["policy_action_attempts"] <= 108000 and calls["fixed_zero_step_attempts"] <= 1800 and
            calls["native_env_step_attempts"] <= 109800, "execution exceeded frozen interaction caps")
    completed = result["status"] == "complete"
    if completed:
        require(rows_decoded == 40 and len(result["episode_files"]) == 360, "complete run missing source candidates or episodes")
        serialized = []
        for pin in result["episode_files"]:
            require(sha(pin["path"]) == pin["sha256"], "episode file changed")
            serialized.append(json.loads(Path(pin["path"]).read_text()))
        require(serialized == result["episodes"], "combined and individual episode records differ")
        public_checks, reset_groups = [], {}
        for ep in serialized:
            key = episode_key(ep)
            public_checks.append(audit_public_episode(ep, endpoints[key[:3]]))
            group = (key[0], key[3])
            if group not in reset_groups: reset_groups[group] = ep["reset_observation_sha256"]
            require(reset_groups[group] == ep["reset_observation_sha256"], "captured paired reset digest differs")
        require(len(reset_groups) == calls["reset_references"] == 40 and calls["reset_exact_comparisons"] == 320,
                "paired reset budget mismatch")
        total_actions = sum(row["policy_actions"] for row in public_checks)
        require(calls["episodes_completed"] == calls["resets_completed"] == calls["resets_attempted"] == calls["episodes_attempted"] == 360 and
                calls["adapters_built"] == 18, "completed episode/model budget mismatch")
        require(calls["policy_forward_attempts"] == calls["policy_forwards_completed"] == calls["policy_action_attempts"] ==
                calls["policy_actions_completed"] == total_actions and calls["fixed_zero_step_attempts"] == calls["fixed_zero_steps_completed"] == 1800 and
                calls["native_env_step_attempts"] == calls["native_env_steps_completed"] == total_actions + 1800,
                "actual interaction count mismatch")
        require(len(events) == 360 and all(e["event"] == "episode_complete" for e in events), "complete run contains missing or failed events")
        for event, ep in zip(events, serialized):
            require((event["task_id"], event["seed"], event["arm"], event["state"]) == episode_key(ep) and
                    event["success"] == ep["success"] and event["steps"] == ep["steps"], "event/episode mismatch")
        summary = independent_summary(serialized)
        require(summary == result["task_results"], "independent paired outcomes differ")
        extra = {"primary_summary_valid": True, "independent_task_results": summary,
            "episodes_and_files_verified": 360, "public_trace_rows_verified": sum(r["trace_rows"] for r in public_checks),
            "public_policy_actions_verified": total_actions, "reset_reference_groups_verified": 40,
            "reset_equalities_verified": 320,
            "terminal_reasons": dict(Counter(ep["terminal_reason"] for ep in serialized))}
    else:
        require(result.get("primary_success_summary_valid") is False and events[-1]["event"] == "technical_abort",
                "technical failure was relabelled or summarized as valid")
        for path in result["partial_episode_files"]:
            ep = json.loads(Path(path).read_text())
            require(ep["task_id"] in TASKS and ep["initial_state_index"] in range(20), "partial run left source scope")
        extra = {"primary_summary_valid": False, "preserved_failure": result["failure"],
            "partial_episode_files_preserved": len(result["partial_episode_files"]), "no_rate_computed_for_failed_run": True}
    answer = {"status": "passed", "experiment_status_preserved": result["status"], "formal_result": False,
        "result_path": str(result_path), "result_sha256": sha(result_path),
        "audit_script_sha256": sha(Path(__file__)), "checkpoint_files_and_tensor_hashes_verified": 18,
        "initial_candidate_rows_recorded": rows_decoded, "qualification_reserved_numeric_rows_recorded": 0,
        "new_policy_environment_optimizer_or_initial_bank_calls": 0,
        "public_offset_norm_absolute_tolerance": 1e-12, "public_offset_vector_comparison": "exact",
        "elapsed_seconds": time.perf_counter() - started, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        **extra}
    with output.open("x") as handle: json.dump(answer, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write("\n")
    return answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); result = audit(args.result, args.output)
    print(json.dumps({"status": result["status"], "experiment_status": result["experiment_status_preserved"],
        "primary_summary_valid": result["primary_summary_valid"], "task_results": result.get("independent_task_results")}, ensure_ascii=False))


if __name__ == "__main__": main()
