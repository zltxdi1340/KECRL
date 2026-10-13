"""Read-only approach and death-timing audit over a validated frozen trace set."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from experiments.analyze_crafter_wood3_turn_ready_do_intervention import _load
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest


HEADS = ("matched_baseline_only", "matched_mixed_coverage")
CONDITIONS = ("ready_do", "turn_ready_do")
STAGES = (0, 1, 2)
CARDINAL_ACTIONS = frozenset((1, 2, 3, 4))
FAILURE_CLASSES = (
    "no_stage_observation",
    "no_visible_unblocked_tree_observed",
    "visible_tree_but_no_safe_approach",
    "approach_available_never_selected",
    "approach_selected_without_observed_progress",
    "approach_progress_without_local_opportunity",
    "local_opportunity_reached_then_failed",
)


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _tree_counts(snapshot: dict) -> tuple[int, int, int | None]:
    visible = snapshot["visible_trees"]
    unblocked = [tree for tree in visible if tree["object"] is None]
    nearest = snapshot["nearest_visible_unblocked_tree_distance"]
    if nearest is not None and not isinstance(nearest, int):
        raise ValueError("nearest visible tree distance must be an integer or null")
    if nearest is None and unblocked:
        raise ValueError("unblocked visible tree list and nearest distance disagree")
    if nearest is not None and not unblocked:
        raise ValueError("nearest distance exists without an unblocked visible tree")
    return len(visible), len(unblocked), nearest


def _local_state(snapshot: dict) -> str:
    """Classify the local state without reading any future event."""
    _tree_counts(snapshot)
    if snapshot["ready_to_collect"]:
        return "ready"
    if snapshot["awake_adjacent_opportunity"]:
        return "turn"
    if snapshot["safe_approach_move_actions"] and not snapshot["sleep_override_active"]:
        return "approach"
    if snapshot["nearest_visible_unblocked_tree_distance"] is None:
        return "no_visible_unblocked_tree"
    if snapshot["sleep_override_active"]:
        return "sleep_override"
    return "visible_no_safe_approach"


def _approach_available(snapshot: dict) -> bool:
    return _local_state(snapshot) == "approach"


def _distance_reduced(before: dict, after: dict | None) -> bool:
    if after is None:
        return False
    before_distance = before["nearest_visible_unblocked_tree_distance"]
    after_distance = after["nearest_visible_unblocked_tree_distance"]
    return (before_distance is not None and after_distance is not None
            and after_distance < before_distance)


def _approach_event(event: dict, next_event: dict | None = None) -> dict | None:
    before, after = event["before"], event["after"]
    if not _approach_available(before):
        return None
    valid = tuple(int(action) for action in before["safe_approach_move_actions"])
    selected = int(event["action"]) in valid
    moved = selected and event["position_after"] != before["position"]
    reduced = selected and _distance_reduced(before, after)
    after_state = "terminal" if after is None else _local_state(after)
    return {
        "approach_available": True,
        "safe_approach_action_count": len(valid),
        "safe_approach_actions": ",".join(str(action) for action in valid),
        "sampled_valid_approach": int(event["sampled_action"]) in valid,
        "executed_valid_approach": selected,
        "approach_action": int(event["action"]),
        "approach_action_replaced": bool(event["action_replaced"]),
        "position_changed": moved,
        "distance_reduced": reduced,
        "before_distance": before["nearest_visible_unblocked_tree_distance"],
        "after_distance": None if after is None else after["nearest_visible_unblocked_tree_distance"],
        "after_local_state": after_state,
        "same_step_death": bool(event["environment_done"] and event["terminal_reason"] == "death"),
        "next_event_observed": next_event is not None,
        "next_action": None if next_event is None else int(next_event["action"]),
        "next_action_do": bool(next_event is not None and next_event["action"] == 5),
        "next_action_wood_gain": bool(next_event is not None and next_event["wood_gain"] == 1),
        "next_event_death": bool(next_event is not None and next_event["environment_done"]
                                and next_event["terminal_reason"] == "death"),
    }


def _stage_rows(episode: dict, stage: int) -> list[dict]:
    return [event for event in episode["events"] if event["wood_before"] == stage]


def _failure_class(episode: dict) -> str:
    if episode["success"]:
        raise ValueError("failure class requested for a successful episode")
    stage = int(episode["final_wood"])
    rows = _stage_rows(episode, stage)
    if not rows:
        return "no_stage_observation"
    states = [_local_state(event["before"]) for event in rows]
    approach_rows = [row for row in rows if _approach_available(row["before"])]
    selected_rows = [row for row in approach_rows
                     if row["action"] in row["before"]["safe_approach_move_actions"]]
    progress_rows = [row for row in selected_rows if _distance_reduced(row["before"], row["after"])]
    if any(state in ("ready", "turn") for state in states):
        return "local_opportunity_reached_then_failed"
    if progress_rows:
        return "approach_progress_without_local_opportunity"
    if selected_rows:
        return "approach_selected_without_observed_progress"
    if approach_rows:
        return "approach_available_never_selected"
    if any(state not in ("no_visible_unblocked_tree", "sleep_override") for state in states):
        return "visible_tree_but_no_safe_approach"
    if any(state == "sleep_override" for state in states):
        # Sleep/capacity is retained in the row-level metrics; keep failure
        # classes focused on the tree/approach boundary.
        return "visible_tree_but_no_safe_approach"
    return "no_visible_unblocked_tree_observed"


def _episode_metrics(episode: dict, seed: int, head: str, condition: str) -> dict:
    events = episode["events"]
    approach_rows = []
    for index, event in enumerate(events):
        row = _approach_event(event, events[index + 1] if index + 1 < len(events) else None)
        if row is not None:
            approach_rows.append((event, row))
    visible_steps = [event["step"] for event in events
                     if event["before"]["nearest_visible_unblocked_tree_distance"] is not None]
    local_steps = [event["step"] for event in events
                   if _local_state(event["before"]) in ("turn", "ready")]
    progress_steps = [event["step"] for event, row in approach_rows if row["distance_reduced"]]
    selected_steps = [event["step"] for event, row in approach_rows if row["executed_valid_approach"]]
    terminal = events[-1]
    return {
        "seed": seed, "head_variant": head, "condition": condition, "episode": episode["episode"],
        "environment_seed": episode["environment_seed"], "success": episode["success"],
        "terminal_reason": episode["terminal_reason"], "final_wood": episode["final_wood"],
        "steps": episode["steps"], "ever_visible_unblocked_tree": bool(visible_steps),
        "ever_approach_available": bool(approach_rows),
        "ever_approach_selected": bool(selected_steps), "ever_approach_progress": bool(progress_steps),
        "ever_local_opportunity": bool(local_steps),
        "visible_tree_steps": len(visible_steps), "approach_opportunity_steps": len(approach_rows),
        "approach_selected_steps": len(selected_steps), "approach_progress_steps": len(progress_steps),
        "minimum_visible_tree_distance": min(
            (event["before"]["nearest_visible_unblocked_tree_distance"] for event in events
             if event["before"]["nearest_visible_unblocked_tree_distance"] is not None), default=None),
        "first_visible_tree_step": min(visible_steps, default=None),
        "last_visible_tree_step": max(visible_steps, default=None),
        "first_approach_step": min((event["step"] for event, _ in approach_rows), default=None),
        "last_approach_progress_step": max(progress_steps, default=None),
        "last_local_opportunity_step": max(local_steps, default=None),
        "terminal_local_state": _local_state(terminal["before"]),
        "terminal_visible_unblocked_tree": terminal["before"]["nearest_visible_unblocked_tree_distance"] is not None,
        "terminal_nearest_tree_distance": terminal["before"]["nearest_visible_unblocked_tree_distance"],
        "terminal_safe_approach_action_count": len(terminal["before"]["safe_approach_move_actions"]),
        "failure_class": None if episode["success"] else _failure_class(episode),
    }


def _failure_row(episode: dict, metrics: dict) -> dict:
    if episode["success"]:
        raise ValueError("failure row requested for successful episode")
    terminal = episode["events"][-1]
    stage = int(episode["final_wood"])
    rows = _stage_rows(episode, stage)
    return {
        **metrics, "failure_stage": stage, "failure_stage_event_count": len(rows),
        "failure_stage_visible_tree_steps": sum(
            event["before"]["nearest_visible_unblocked_tree_distance"] is not None for event in rows),
        "failure_stage_approach_steps": sum(_approach_available(event["before"]) for event in rows),
        "failure_stage_selected_approach_steps": sum(
            _approach_available(event["before"])
            and event["action"] in event["before"]["safe_approach_move_actions"] for event in rows),
        "failure_stage_progress_steps": sum(
            _approach_available(event["before"])
            and event["action"] in event["before"]["safe_approach_move_actions"]
            and _distance_reduced(event["before"], event["after"]) for event in rows),
        "failure_stage_local_opportunity_steps": sum(
            _local_state(event["before"]) in ("turn", "ready") for event in rows),
        "terminal_step": terminal["step"],
        "terminal_action": terminal["action"],
        "terminal_action_is_cardinal": terminal["action"] in CARDINAL_ACTIONS,
        "terminal_position_changed": terminal["position_after"] != terminal["before"]["position"],
        "terminal_health_before": terminal["before"]["health"],
        "terminal_health_after": terminal["health_after"],
        "terminal_health_delta": terminal["health_delta"],
        "terminal_nearby_hostile_count": terminal["before"]["nearby_hostile_count"],
        "terminal_damage_source_hint": terminal["damage_source_hint"],
    }


def _death_row(episode: dict, metrics: dict) -> dict:
    if episode["terminal_reason"] != "death":
        raise ValueError("death row requested for a non-death episode")
    terminal = episode["events"][-1]
    before = terminal["before"]
    last_progress = metrics["last_approach_progress_step"]
    last_visible = metrics["last_visible_tree_step"]
    last_local = metrics["last_local_opportunity_step"]
    return {
        **metrics, "death_step": terminal["step"], "death_stage": episode["final_wood"],
        "death_before_local_state": _local_state(before),
        "death_visible_tree_count": _tree_counts(before)[0],
        "death_visible_unblocked_tree_count": _tree_counts(before)[1],
        "death_nearest_tree_distance": before["nearest_visible_unblocked_tree_distance"],
        "death_safe_approach_action_count": len(before["safe_approach_move_actions"]),
        "death_action": terminal["action"],
        "death_action_is_approach": terminal["action"] in before["safe_approach_move_actions"],
        "death_action_is_turn": terminal["action"] in before["unblocked_tree_move_actions"],
        "death_action_is_ready_do": terminal["action"] == 5 and before["ready_to_collect"],
        "death_position_changed": terminal["position_after"] != before["position"],
        "death_health_before": before["health"], "death_health_after": terminal["health_after"],
        "death_health_delta": terminal["health_delta"],
        "death_food": before["food"], "death_drink": before["drink"], "death_energy": before["energy"],
        "death_sleeping": before["sleeping"], "death_sleep_override_active": before["sleep_override_active"],
        "death_daylight": before["daylight"], "death_nearby_hostile_count": before["nearby_hostile_count"],
        "death_damage_source_hint": terminal["damage_source_hint"],
        "steps_since_last_visible_tree": None if last_visible is None else terminal["step"] - last_visible,
        "steps_since_last_approach_progress": None if last_progress is None else terminal["step"] - last_progress,
        "steps_since_last_local_opportunity": None if last_local is None else terminal["step"] - last_local,
    }


def _aggregate_approach(episodes: list[dict], seed: int, head: str, condition: str) -> list[dict]:
    rows_by_key = defaultdict(list)
    for episode in episodes:
        for index, event in enumerate(episode["events"]):
            row = _approach_event(event, episode["events"][index + 1] if index + 1 < episode["steps"] else None)
            if row is None:
                continue
            key = event["wood_before"]
            rows_by_key[key].append(row)
    output = []
    for stage in ("all", 0, 1, 2):
        rows = [row for key, values in rows_by_key.items() if stage == "all" or key == stage for row in values]
        output.append({
            "seed": seed, "head_variant": head, "condition": condition, "wood_stage": stage,
            "approach_opportunity_steps": len(rows),
            "sampled_valid_approach_steps": sum(row["sampled_valid_approach"] for row in rows),
            "executed_valid_approach_steps": sum(row["executed_valid_approach"] for row in rows),
            "position_changed_steps": sum(row["position_changed"] for row in rows),
            "distance_reduced_steps": sum(row["distance_reduced"] for row in rows),
            "after_approach_ready_or_turn_steps": sum(row["after_local_state"] in ("ready", "turn") for row in rows),
            "same_step_deaths": sum(row["same_step_death"] for row in rows),
            "next_event_observed": sum(row["next_event_observed"] for row in rows),
            "next_action_do": sum(row["next_action_do"] for row in rows),
            "next_action_wood_gain": sum(row["next_action_wood_gain"] for row in rows),
            "next_event_death": sum(row["next_event_death"] for row in rows),
        })
    return output


def _aggregate_diagnosis(metrics: list[dict], seed: int, head: str, condition: str) -> dict:
    rows = [row for row in metrics
            if (seed == "all" or row["seed"] == seed)
            and row["head_variant"] == head and row["condition"] == condition]
    failed = [row for row in rows if not row["success"]]
    return {
        "seed": seed, "head_variant": head, "condition": condition, "episodes": len(rows),
        "successes": sum(row["success"] for row in rows), "deaths": sum(row["terminal_reason"] == "death" for row in rows),
        "external_truncations": sum(row["terminal_reason"] == "external_truncation" for row in rows),
        "episodes_with_visible_unblocked_tree": sum(row["ever_visible_unblocked_tree"] for row in rows),
        "episodes_with_approach_available": sum(row["ever_approach_available"] for row in rows),
        "episodes_with_approach_selected": sum(row["ever_approach_selected"] for row in rows),
        "episodes_with_approach_progress": sum(row["ever_approach_progress"] for row in rows),
        "episodes_with_local_opportunity": sum(row["ever_local_opportunity"] for row in rows),
        "failed_episodes": len(failed),
        **{f"failure_{name}": sum(row["failure_class"] == name for row in failed) for name in FAILURE_CLASSES},
    }


def _validate_config(config: dict) -> None:
    expected = {
        "status": "crafter_wood3_approach_death_audit_v1", "formal_result": False,
        "diagnostic_only": True, "evaluation_only": True, "new_environment_interactions": 0,
        "policy_training_steps": 0, "encoder_updated": False, "new_fit_or_selection_performed": False,
        "intervention_executor_is_rgb_only": False, "module_registered": False,
        "knowledge_updated": False, "spt_updated": False, "causal_effect_claim": False,
        "input_result_root": "results/crafter_wood3_turn_ready_do_intervention_cuda_20261012_v1",
        "source_run_profile": "main", "heads": list(HEADS), "conditions": list(CONDITIONS),
        "wood_stages": list(STAGES),
        "approach_definition": "safe_approach_move_actions_nonempty_while_not_ready_or_adjacent_turn",
        "approach_distance_metric": "nearest_visible_unblocked_tree_manhattan_distance",
        "death_definition": "last_event_environment_done_with_terminal_reason_death",
        "external_truncation_censored": True,
        "event_statistics_are_correlated_descriptive_observations": True,
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"locked approach/death audit differs: {key}")


def _verify_source(root: Path, config: dict) -> tuple[dict, dict, dict, dict]:
    source_config = _read(root / "config.json")
    summary = _read(root / "summary.json")
    provenance = _read(root / "provenance.json")
    prior_verification = _read(root / "analysis_verified/verification.json")
    if (summary["run_profile"] != config["source_run_profile"] or summary["formal_result"]
            or summary["policy_training_steps"] != 0 or summary["supervised_optimizer_updates"] != 0
            or summary["encoder_updated"] or summary["new_fit_or_selection_performed"]
            or summary["intervention_executor_is_rgb_only"] is not False or summary["teacher_used"] is not True
            or summary["module_registered"] or summary["knowledge_updated"] or summary["spt_updated"]
            or summary["approach_intervention_used"] or summary["survival_intervention_used"]
            or not all(summary["files_unchanged"].values())):
        raise ValueError("source result is not a completed frozen diagnostic")
    if (not prior_verification["rgb_policy_sampling_and_environment_replay_passed"]
            or not prior_verification["reference_audits_passed"]
            or not prior_verification["cross_process_repetition_passed"]
            or not prior_verification["paired_prefix_before_incremental_turn_and_sampling_rng_checks_passed"]):
        raise ValueError("source independent verification is incomplete")
    hashes = provenance["sha256_before_evaluation"]
    if not all(_sha256(Path(path)) == digest for path, digest in hashes.items()):
        raise RuntimeError("source result input hash changed")
    return source_config, summary, provenance, prior_verification


def run(config_path: str, input_path: str, output_path: str) -> dict:
    config = _read(Path(config_path))
    _validate_config(config)
    root = Path(input_path)
    if str(root) != config["input_result_root"]:
        raise ValueError("input result root differs from locked config")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    source_config, summary, provenance, prior_verification = _verify_source(root, config)
    if source_config["status"] != "crafter_wood3_turn_ready_do_intervention_cuda_v1":
        raise ValueError("source config status differs")
    runs = summary["runs"]
    if len(runs) != len(HEADS) * len(CONDITIONS) * 3:
        raise ValueError("source run count differs")
    episodes_by_key = {}
    input_paths = [root / name for name in ("config.json", "summary.json", "provenance.json", "selection_lock.json",
                                             "analysis_verified/verification.json")]
    input_paths += [root / Path(row["trace"]).name for row in runs]
    input_hashes = {str(path): _sha256(path) for path in input_paths}
    for row in runs:
        episodes = _load(root, row, source_config, False)
        key = row["seed"], row["head_variant"], row["condition"]
        if key in episodes_by_key:
            raise ValueError("duplicate source run key")
        episodes_by_key[key] = episodes
    metrics = []
    failures = []
    deaths = []
    approach = []
    diagnosis = []
    for (seed, head, condition), episodes in sorted(episodes_by_key.items()):
        metrics.extend(_episode_metrics(ep, seed, head, condition) for ep in episodes)
        failures.extend(_failure_row(ep, metrics[-len(episodes) + index])
                         for index, ep in enumerate(episodes) if not ep["success"])
        deaths.extend(_death_row(ep, metrics[-len(episodes) + index])
                      for index, ep in enumerate(episodes) if ep["terminal_reason"] == "death")
        approach.extend(_aggregate_approach(episodes, seed, head, condition))
        diagnosis.append(_aggregate_diagnosis(metrics, seed, head, condition))
    pooled_metrics = []
    pooled_diagnosis = []
    pooled_approach = []
    for head in HEADS:
        for condition in CONDITIONS:
            rows = [row for row in metrics if row["head_variant"] == head and row["condition"] == condition]
            pooled_metrics.extend(rows)
            pooled_diagnosis.append(_aggregate_diagnosis(rows, "all", head, condition))
            grouped = [ep for (seed, h, c), eps in episodes_by_key.items() if h == head and c == condition for ep in eps]
            pooled_approach.extend(_aggregate_approach(grouped, "all", head, condition))
    all_failure_counts = Counter(row["failure_class"] for row in failures)
    all_death_counts = Counter(row["death_before_local_state"] for row in deaths)
    current_input_hashes = {str(path): _sha256(path) for path in input_paths}
    if current_input_hashes != input_hashes:
        raise RuntimeError("source inputs changed during read-only analysis")
    current_source_hashes = {path: _sha256(Path(path)) for path in provenance["sha256_before_evaluation"]}
    if current_source_hashes != provenance["sha256_before_evaluation"]:
        raise RuntimeError("sealed source files changed during read-only analysis")
    verification = {
        "formal_result": False, "diagnostic_only": True, "evaluation_only": True,
        "new_environment_interactions": 0, "policy_training_steps": 0, "encoder_updated": False,
        "new_fit_or_selection_performed": False, "module_registered": False, "knowledge_updated": False,
        "spt_updated": False, "causal_effect_claim": False, "external_truncation_censored": True,
        "source_independent_verification_reused": True, "source_prior_verification_sha256": _sha256(root / "analysis_verified/verification.json"),
        "source_input_files_unchanged": True, "source_trace_count": len(runs),
        "source_episode_count": len(metrics), "source_event_count": sum(row["steps"] for row in metrics),
        "failure_class_counts": dict(sorted(all_failure_counts.items())),
        "death_before_local_state_counts": dict(sorted(all_death_counts.items())),
        "input_hashes": input_hashes,
        "analysis_source_sha256": _sha256(Path(__file__)),
    }
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    _write_csv(output / "episode_metrics.csv", metrics)
    _write_csv(output / "failure_attribution.csv", failures)
    _write_csv(output / "death_timing.csv", deaths)
    _write_csv(output / "approach_outcomes.csv", approach + pooled_approach)
    _write_csv(output / "diagnosis_summary.csv", diagnosis + pooled_diagnosis)
    verification["output_sha256"] = {path.name: _sha256(path) for path in sorted(output.iterdir())}
    _write(output / "verification.json", verification)
    return {"verification": verification, "diagnosis_summary": diagnosis + pooled_diagnosis,
            "failure_class_counts": dict(sorted(all_failure_counts.items()))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.config, args.input, args.output)
    print(json.dumps({"diagnosis_summary": result["diagnosis_summary"],
                      "failure_class_counts": result["failure_class_counts"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
