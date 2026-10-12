"""Read-only transition audit of matched wood-3 natural-policy traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


VARIANTS = ("matched_baseline_only", "matched_mixed_coverage")
CATEGORIES = ("ready", "turn", "approach", "other")
STAGES = (0, 1, 2)


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def local_category(snapshot: dict) -> str:
    if snapshot["ready_to_collect"]:
        return "ready"
    if snapshot["awake_adjacent_opportunity"]:
        return "turn"
    if snapshot["safe_approach_move_actions"] and not snapshot["sleep_override_active"]:
        return "approach"
    return "other"


def _valid_local_actions(snapshot: dict, category: str) -> tuple[int, ...]:
    if category == "ready":
        return (5,)
    if category == "turn":
        return tuple(int(action) for action in snapshot["unblocked_tree_move_actions"])
    if category == "approach":
        return tuple(int(action) for action in snapshot["safe_approach_move_actions"])
    return ()


def _validate_episode(episode: dict, *, seed: int, variant: str, index: int,
                      config: dict) -> None:
    if (episode["policy_seed"] != seed or episode["variant"] != variant or episode["episode"] != index
            or episode["steps"] != len(episode["events"])
            or episode["initial_snapshot"]["wood"] != 0
            or episode["success"] != (episode["final_wood"] >= config["wood_threshold"])
            or episode["terminal_reason"] not in ("success", "death", "external_truncation")):
        raise ValueError(f"episode manifest/boundary differs: seed={seed} variant={variant} episode={index}")
    if not episode["events"] or episode["steps"] > config["fresh_max_steps"]:
        raise ValueError(f"empty or oversized episode: seed={seed} variant={variant} episode={index}")
    events = episode["events"]
    if episode["initial_rgb_digest"] != events[0]["rgb_before_digest"]:
        raise ValueError("initial RGB digest does not match first transition")
    if episode["initial_snapshot"] != events[0]["before"]:
        raise ValueError("initial snapshot does not match first transition")
    first_wood_steps = {}
    for position, event in enumerate(events):
        before, after = event["before"], event["after"]
        probabilities = event["probabilities"]
        if event["step"] != position + 1 or event["action"] not in config["action_allowlist"]:
            raise ValueError("step index or action allowlist differs")
        if len(probabilities) != 17 or any(not math.isfinite(p) or p < 0 for p in probabilities):
            raise ValueError("invalid action probability vector")
        if abs(sum(probabilities) - 1.0) > 1e-5:
            raise ValueError("action probabilities are not normalized")
        if any(p != 0 for action, p in enumerate(probabilities) if action not in config["action_allowlist"]):
            raise ValueError("action probabilities escape the allowlist")
        wood_before, wood_after = int(event["wood_before"]), int(event["wood_after"])
        if (wood_before not in STAGES or before["wood"] != wood_before
                or event["wood_gain"] != wood_after - wood_before
                or event["wood_gain"] not in (0, 1)
                or event["wood_gain"] != int(event["action"] == 5 and before["ready_to_collect"])):
            raise ValueError("wood transition differs from the Crafter collection rule")
        for stage in STAGES:
            if wood_after >= stage + 1:
                first_wood_steps.setdefault(str(stage + 1), event["step"])
        if not isinstance(event["health_after"], int) or not isinstance(event["environment_done"], bool):
            raise ValueError("health or terminal observation is missing")
        if event["environment_done"] != (after is None):
            raise ValueError("terminal observation/snapshot boundary differs")
        if event["health_delta"] != event["health_after"] - before["health"]:
            raise ValueError("health delta differs from observed health")
        if (event["environment_done"] != (event["terminal_reason"] == "death")
                or event["environment_done"] != (event["health_after"] <= 0)):
            raise ValueError("native terminal/death observation differs")
        if after is not None:
            if after["wood"] != wood_after or after["health"] != event["health_after"]:
                raise ValueError("post-transition inventory or health differs")
            if after["position"] != event["position_after"] or after["facing"] != event["facing_after"]:
                raise ValueError("post-transition position or facing differs")
        if position:
            previous = events[position - 1]
            if (previous["environment_done"]
                    or previous["rgb_after_digest"] != event["rgb_before_digest"]
                    or previous["wood_after"] != wood_before
                    or previous["after"] != before):
                raise ValueError("adjacent transition frames/snapshots are discontinuous")
        if event["environment_done"] and position != len(events) - 1:
            raise ValueError("events continue after native termination")
    last = events[-1]
    if (episode["final_wood"] != last["wood_after"]
            or first_wood_steps != episode["first_wood_steps"]):
        raise ValueError("episode final inventory or milestone record differs")
    if episode["success"]:
        if episode["terminal_reason"] != "success":
            raise ValueError("successful episode terminal label differs")
    elif last["environment_done"]:
        if episode["terminal_reason"] != last["terminal_reason"]:
            raise ValueError("native terminal label differs")
    elif episode["terminal_reason"] != "external_truncation":
        raise ValueError("nonterminal episode must be externally truncated")
    if episode["terminal_reason"] == "external_truncation" and len(events) != config["fresh_max_steps"]:
        raise ValueError("external truncation is not at the step limit")
    offset = config["seed_set"].index(seed) * config["fresh_replicate_seed_stride"] + index
    expected_env_seed = config["fresh_environment_seed_base"] + offset
    expected_action_seed = config["fresh_action_seed_base"] + offset
    if episode["environment_seed"] != expected_env_seed or episode["action_seed"] != expected_action_seed:
        raise ValueError("episode RNG seed manifest differs")


def _trace_path(root: Path, run: dict) -> Path:
    path = Path(run["trace"])
    if not path.is_file():
        path = root / path.name
    if path.resolve().parent != root.resolve():
        raise ValueError("trace path is outside its source result directory")
    return path


def _load_trace(root: Path, run: dict, *, seed: int, variant: str, config: dict) -> list[dict]:
    path = _trace_path(root, run)
    raw = path.read_bytes()
    episodes = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
    if (_digest(episodes) != run["trace_canonical_digest"]
            or hashlib.sha256(raw).hexdigest() != run["trace_sha256"]):
        raise RuntimeError(f"trace digest differs: {path}")
    if len(episodes) != config["fresh_episode_count"]:
        raise ValueError(f"episode count differs: {path}")
    for index, episode in enumerate(episodes):
        _validate_episode(episode, seed=seed, variant=variant, index=index, config=config)
    actual_summary = {
        "episodes": len(episodes),
        "interaction_steps": sum(row["steps"] for row in episodes),
        "successes": sum(bool(row["success"]) for row in episodes),
        "success_rate": sum(bool(row["success"]) for row in episodes) / len(episodes),
        "terminal_counts": {reason: sum(row["terminal_reason"] == reason for row in episodes)
                            for reason in sorted({row["terminal_reason"] for row in episodes})},
        "wood_milestone_episode_counts": {
            str(stage): sum(str(stage) in row["first_wood_steps"] for row in episodes)
            for stage in (1, 2, 3)},
    }
    if any(run["summary"].get(key) != value for key, value in actual_summary.items()):
        raise ValueError(f"stored run summary differs from trace: {path}")
    if run["policy_and_optimizer_unchanged"] is not True or run["all_gradients_absent"] is not True:
        raise ValueError(f"frozen-policy boundary differs: {path}")
    return episodes


def _event_rows(episodes: list[dict], seed: int, variant: str, horizon: int) -> list[dict]:
    if horizon < 1:
        raise ValueError("followthrough horizon must be positive")
    rows = []
    for episode in episodes:
        events = episode["events"]
        for index, event in enumerate(events):
            before = event["before"]
            category = local_category(before)
            valid_actions = _valid_local_actions(before, category)
            selected = event["action"] in valid_actions if valid_actions else None
            after = event["after"]
            after_category = local_category(after) if after is not None else "terminal"
            next_events = events[index + 1:index + 1 + horizon]
            followup_len = len(next_events)
            reached_episode_end = index + 1 + followup_len == len(events)
            ended_by_task_or_native = reached_episode_end and episode["terminal_reason"] in ("success", "death")
            alive_after = not event["environment_done"] and event["health_after"] > 0
            eligible = alive_after and event["wood_after"] < 3
            complete = eligible and (followup_len >= horizon or ended_by_task_or_native)
            followup_reason = "ineligible" if not eligible else "horizon" if followup_len >= horizon else (
                episode["terminal_reason"] if reached_episode_end and complete else
                "external_truncation" if reached_episode_end else "incomplete")
            nearest_before = before["nearest_visible_unblocked_tree_distance"]
            nearest_after = after["nearest_visible_unblocked_tree_distance"] if after is not None else None
            approach_distance_observed = (category == "approach" and nearest_before is not None
                                          and nearest_after is not None)
            next_ready_action_observed = after_category == "ready" and bool(next_events)
            rows.append({
                "seed": seed, "variant": variant, "episode": episode["episode"],
                "step": event["step"], "wood_stage": int(event["wood_before"]),
                "category": category, "selected_local_action": selected,
                "nearby_hostile": before["nearby_hostile_count"] > 0,
                "nearby_hostile_count": int(before["nearby_hostile_count"]),
                "sleep_override": bool(before["sleep_override_active"]),
                "action": int(event["action"]), "environment_done": event["environment_done"],
                "move_selected_awake": event["action"] in (1, 2, 3, 4) and not before["sleep_override_active"],
                "blocked_move": (event["action"] in (1, 2, 3, 4) and not before["sleep_override_active"]
                                 and event["position_after"] == before["position"]),
                "alive_after": alive_after,
                "damage_after": event["health_delta"] is not None and event["health_delta"] < 0,
                "wood_gain": int(event["wood_gain"]),
                "after_category": after_category,
                "after_ready": after_category == "ready",
                "after_any_local_opportunity": after_category in ("ready", "turn", "approach"),
                "turn_became_ready": category == "turn" and after_category == "ready",
                "next_ready_action_observed": next_ready_action_observed,
                "next_ready_do": next_ready_action_observed and next_events[0]["action"] == 5,
                "approach_distance_observed": approach_distance_observed,
                "approach_distance_reduced": (approach_distance_observed and nearest_after < nearest_before),
                "followup_wood_gain": any(row["wood_gain"] > 0 for row in next_events),
                "followup_death": any(row["environment_done"] and row["terminal_reason"] == "death"
                                       for row in next_events),
                "followup_task_success": any(row["wood_after"] >= 3 for row in next_events),
                "followup_steps": followup_len, "followup_complete": complete,
                "followup_eligible": eligible,
                "followup_reason": followup_reason,
            })
    return rows


def _mean_rate(count: int, denominator: int):
    return count / denominator if denominator else None


def _group(rows: list[dict], keys: tuple[str, ...]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    return [{**dict(zip(keys, key)), **aggregate(values)}
            for key, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0]))]


def aggregate(rows: list[dict]) -> dict:
    count = len(rows)
    after_observed = sum(row["after_category"] != "terminal" for row in rows)
    followup_eligible = [row for row in rows if row["followup_eligible"]]
    followup_complete = [row for row in followup_eligible if row["followup_complete"]]
    followup_censored = [row for row in followup_eligible if not row["followup_complete"]]
    observed_gain = sum(row["followup_wood_gain"] for row in followup_eligible)
    unobserved_gain = sum(not row["followup_wood_gain"] for row in followup_censored)
    approach_observed = [row for row in rows if row["approach_distance_observed"]]
    return {
        "transitions": count,
        "episodes": len({(row["seed"], row["variant"], row["episode"]) for row in rows}),
        "local_action_states": sum(row["category"] != "other" for row in rows),
        "selected_local_actions": sum(row["selected_local_action"] is True for row in rows),
        "local_action_selection_rate": _mean_rate(sum(row["selected_local_action"] is True for row in rows),
                                                   sum(row["category"] != "other" for row in rows)),
        "terminal_events": sum(row["environment_done"] for row in rows),
        "alive_after": sum(row["alive_after"] for row in rows),
        "damage_events": sum(row["damage_after"] for row in rows),
        "awake_move_actions": sum(row["move_selected_awake"] for row in rows),
        "blocked_moves": sum(row["blocked_move"] for row in rows),
        "wood_gain_events": sum(row["wood_gain"] > 0 for row in rows),
        "wood_gained": sum(row["wood_gain"] for row in rows),
        "after_snapshot_observed": after_observed,
        "after_ready": sum(row["after_ready"] for row in rows),
        "after_turn": sum(row["after_category"] == "turn" for row in rows),
        "after_approach": sum(row["after_category"] == "approach" for row in rows),
        "after_other": sum(row["after_category"] == "other" for row in rows),
        "after_any_local_opportunity": sum(row["after_any_local_opportunity"] for row in rows),
        "turn_to_ready": sum(row["turn_became_ready"] for row in rows),
        "next_ready_action_observed": sum(row["next_ready_action_observed"] for row in rows),
        "next_ready_do": sum(row["next_ready_do"] for row in rows),
        "approach_distance_observed": len(approach_observed),
        "approach_distance_reduced": sum(row["approach_distance_reduced"] for row in approach_observed),
        "followup_windows_eligible": len(followup_eligible),
        "followup_windows_complete": len(followup_complete),
        "followup_windows_censored": len(followup_censored),
        "followup_wood_gain_observed": observed_gain,
        "followup_wood_gain_rate_lower_bound": _mean_rate(observed_gain, len(followup_eligible)),
        "followup_wood_gain_rate_upper_bound": _mean_rate(observed_gain + unobserved_gain,
                                                         len(followup_eligible)),
        "followup_wood_gain": sum(row["followup_wood_gain"] for row in followup_complete),
        "followup_wood_gain_rate_complete": _mean_rate(
            sum(row["followup_wood_gain"] for row in followup_complete), len(followup_complete)),
        "followup_death": sum(row["followup_death"] for row in followup_complete),
        "followup_death_rate_complete": _mean_rate(
            sum(row["followup_death"] for row in followup_complete), len(followup_complete)),
        "followup_task_success": sum(row["followup_task_success"] for row in followup_complete),
        **{f"action_{action}": sum(row["action"] == action for row in rows) for action in range(7)},
    }


def _verify_run(root: Path, summary: dict, config: dict) -> tuple[list[dict], list[dict]]:
    if (summary["formal_result"] is not False or summary["diagnostic_only"] is not True
            or summary["policy_training_steps"] != 0 or summary["encoder_updated"] is not False
            or summary["coverage_heldout_used_for_fit_or_selection"] is not False
            or summary["training_data_budget_matched"] is not True
            or summary["shared_validation_across_variants"] is not True
            or not all(summary["files_unchanged"].values())):
        raise ValueError("matched source is not a frozen diagnostic with verified inputs")
    repetition = summary["cross_process_repetition"]
    if repetition is None or repetition["passed"] is not True:
        raise ValueError("matched source cross-process repetition did not pass")
    if (len(summary["seeds"]) != len(config["seed_set"])
            or {row["seed"] for row in summary["seeds"]} != set(config["seed_set"])):
        raise ValueError("matched source seed set differs")
    provenance = _read(root / "provenance.json")
    source_checks = {path: _sha256(Path(path)) == digest
                     for path, digest in provenance["sha256_before_evaluation"].items()}
    if not all(source_checks.values()):
        raise RuntimeError("matched source provenance hash differs")
    all_rows, trace_checks = [], []
    seed_rows = {row["seed"]: row for row in summary["seeds"]}
    for seed in config["seed_set"]:
        seed_row = seed_rows[seed]
        if (len(seed_row["runs"]) != len(VARIANTS)
                or {run["variant"] for run in seed_row["runs"]} != set(VARIANTS)):
            raise ValueError(f"source variants differ for seed {seed}")
        variants = {}
        for run in seed_row["runs"]:
            variant = run["variant"]
            episodes = _load_trace(root, run, seed=seed, variant=variant, config=config)
            variants[variant] = episodes
            trace_checks.append({"seed": seed, "variant": variant,
                                 "episodes": len(episodes), "trace_sha256": run["trace_sha256"],
                                 "trace_canonical_digest": run["trace_canonical_digest"]})
            all_rows.extend(_event_rows(episodes, seed, variant, config["followthrough_horizon_steps"]))
        left, right = (variants[name] for name in VARIANTS)
        for index, (baseline, mixed) in enumerate(zip(left, right)):
            if (baseline["environment_seed"] != mixed["environment_seed"]
                    or baseline["action_seed"] != mixed["action_seed"]
                    or baseline["initial_rgb_digest"] != mixed["initial_rgb_digest"]
                    or baseline["initial_snapshot"] != mixed["initial_snapshot"]
                    or index != baseline["episode"] or index != mixed["episode"]):
                raise ValueError(f"matched initial-state pair differs for seed {seed} episode {index}")
        audits = seed_row["observer_audits"]
        if len(audits) != len(VARIANTS) or not all(
                row["trajectory_without_collection_oracle_equal"] for row in audits):
            raise ValueError(f"source observer audit failed for seed {seed}")
    repeat_root = root / "cross_process_repeat"
    repeat_summary = _read(repeat_root / "summary.json")
    if repeat_summary["cross_process_repetition"] is not None:
        raise ValueError("repeat source unexpectedly nests another repeat")
    repeat_seed = repeat_summary["seeds"][0]
    main_seed = seed_rows[config["seed_set"][0]]
    repeat_runs = {row["variant"]: row for row in repeat_seed["runs"]}
    main_runs = {row["variant"]: row for row in main_seed["runs"]}
    if set(repeat_runs) != set(VARIANTS):
        raise ValueError("repeat variants differ")
    for variant in VARIANTS:
        repeated = _load_trace(repeat_root, repeat_runs[variant], seed=config["seed_set"][0],
                               variant=variant, config=config)
        if _digest(repeated) != main_runs[variant]["trace_canonical_digest"]:
            raise RuntimeError(f"independent-process trace differs: {variant}")
    return all_rows, trace_checks


def run(config_path: str, input_path: str, output_path: str) -> dict:
    config_file, root, output = Path(config_path), Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config = _read(config_file)
    expected = {
        "status": "crafter_wood3_action_consequence_analysis_v1", "formal_result": False,
        "evaluation_only": True, "new_environment_interactions": 0,
        "policy_variants": list(VARIANTS), "categories": list(CATEGORIES[:3]),
        "followthrough_horizon_steps": 8, "policy_input": "RGB64x64x3 only",
        "diagnostic_snapshots_used_after_action_selection": True, "causal_effect_claim": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"action consequence protocol differs: {key}")
    if root.resolve() != Path(config["source_result_root"]).resolve():
        raise ValueError("source result root differs from the pinned analysis config")
    source_config = _read(root / "config.json")
    source_summary = _read(root / "summary.json")
    if (source_config["fresh_variants"] != list(VARIANTS)
            or source_config["seed_set"] != [0, 1, 2]
            or source_config["fresh_episode_count"] != 30
            or source_config["fresh_max_steps"] != 256
            or source_config["initial_wood"] != 0):
        raise ValueError("pinned matched evaluation protocol differs")
    analysis_config = {
        "seed_set": source_config["seed_set"],
        "fresh_episode_count": source_config["fresh_episode_count"],
        "fresh_max_steps": source_config["fresh_max_steps"],
        "fresh_environment_seed_base": source_config["fresh_environment_seed_base"],
        "fresh_action_seed_base": source_config["fresh_action_seed_base"],
        "fresh_replicate_seed_stride": source_config["fresh_replicate_seed_stride"],
        "action_allowlist": source_config["action_allowlist"],
        "wood_threshold": 3,
        "followthrough_horizon_steps": config["followthrough_horizon_steps"],
    }
    repeat_root = root / "cross_process_repeat"
    repeat_summary = _read(repeat_root / "summary.json")
    source_verification_path = root / "analysis_verified/verification.json"
    source_verification = _read(source_verification_path)
    previous_analyzer = Path("experiments/analyze_crafter_wood3_natural_coverage_matched.py")
    if source_verification["analysis_source_sha256"] != _sha256(previous_analyzer):
        raise RuntimeError("matched independent analyzer source differs from its verification")
    paths = [config_file, Path(__file__), source_verification_path, previous_analyzer]
    for directory, summary in ((root, source_summary), (repeat_root, repeat_summary)):
        paths += [directory / name for name in ("config.json", "summary.json", "provenance.json")]
        paths += [_trace_path(directory, run) for seed_row in summary["seeds"] for run in seed_row["runs"]]
    provenance = _read(root / "provenance.json")
    paths += [Path(path) for path in provenance["sha256_before_evaluation"]]
    before_hashes = {str(path): _sha256(path) for path in paths}
    rows, trace_checks = _verify_run(root, source_summary, analysis_config)
    output.mkdir(parents=True)
    _write(output / "config.json", config)
    overall = _group(rows, ("variant", "wood_stage", "category"))
    pooled = [{"variant": variant, "wood_stage": "all", "category": "all",
               **aggregate([row for row in rows if row["variant"] == variant])}
              for variant in VARIANTS]
    overall.extend(pooled)
    by_seed = _group(rows, ("seed", "variant", "wood_stage", "category"))
    selected = _group(rows, ("variant", "wood_stage", "category", "selected_local_action"))
    selected += [{"wood_stage": "all", **row} for row in
                 _group(rows, ("variant", "category", "selected_local_action"))]
    selected_by_seed = _group(rows, ("seed", "variant", "category", "selected_local_action"))
    by_hostile = _group(rows, ("variant", "wood_stage", "nearby_hostile"))
    by_hostile += [{"wood_stage": "all", **row} for row in
                   _group(rows, ("variant", "category", "nearby_hostile"))]
    _write_csv(output / "transition_outcomes.csv", overall)
    _write_csv(output / "transition_outcomes_by_seed.csv", by_seed)
    _write_csv(output / "transition_outcomes_by_local_action.csv", selected)
    _write_csv(output / "local_action_outcomes_by_seed.csv", selected_by_seed)
    _write_csv(output / "hostile_exposure_association.csv", by_hostile)
    verification = {
        "formal_result": False, "evaluation_only": True, "new_environment_interactions": 0,
        "policy_training_steps": 0, "encoder_updated": False,
        "module_registered": False, "knowledge_updated": False, "spt_updated": False,
        "causal_effect_claim": False, "diagnostic_snapshots_used_after_action_selection": True,
        "independent_source_verification_passed": True,
        "source_trace_checks": trace_checks,
        "source_trace_count": len(trace_checks), "episodes_per_arm": source_config["fresh_episode_count"] * len(source_config["seed_set"]),
        "events_per_arm": {row["variant"]: row["transitions"] for row in pooled},
        "followthrough_horizon_steps_after_current_action": config["followthrough_horizon_steps"],
        "external_truncation_windows_censored": True,
        "followthrough_excludes_current_action": True,
        "followthrough_excludes_already_terminal_or_successful_starts": True,
        "step_statistics_are_correlated_descriptive_observations": True,
        "analysis_source_sha256": _sha256(Path(__file__)),
        "source_input_hashes": before_hashes,
        "analysis_output_sha256": {path.name: _sha256(path) for path in sorted(output.iterdir())},
    }
    after_hashes = {path: _sha256(Path(path)) for path in before_hashes}
    if before_hashes != after_hashes:
        raise RuntimeError("source files changed during read-only analysis")
    verification["source_inputs_unchanged_during_analysis"] = True
    _write(output / "verification.json", verification)
    return {"verification": verification, "pooled_transitions": pooled}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.config, args.input, args.output)
    print(json.dumps({"verification": result["verification"],
                      "pooled_transitions": result["pooled_transitions"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
