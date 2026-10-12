"""Replay and incremental paired analysis of local turn-to-ready intervention."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import torch

from experiments.analyze_crafter_wood3_action_consequence import _validate_episode
from experiments.analyze_crafter_wood3_ready_do_intervention import _outcome as ready_outcome, _replay, _stages, _write_csv
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from experiments.run_crafter_wood3_turn_ready_do_intervention import CONDITIONS, COUNTS, HEADS, _validate
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_determinism import stable_crafter_object_order


def _validate_gates(episode: dict, condition: str, allowlist: list[int]) -> None:
    if condition not in CONDITIONS:
        raise ValueError("unknown turn-ready-do condition")
    sampled_counts, executed_counts = Counter(), Counter()
    counts = dict.fromkeys(COUNTS, 0)
    for event in episode["events"]:
        before, sampled = event["before"], event["sampled_action"]
        ready = bool(before["ready_to_collect"])
        turn = bool(condition == "turn_ready_do" and not ready and before["awake_adjacent_opportunity"])
        valid = before["unblocked_tree_move_actions"]
        if turn and (not valid or any(a not in (1, 2, 3, 4) for a in valid)):
            raise ValueError("invalid cardinal turn opportunity")
        if ready:
            expected, kind = 5, "ready_do"
        elif turn:
            candidates = sorted(valid, key=lambda a: (-event["probabilities"][a], a))
            expected, kind = (sampled if sampled in valid else candidates[0]), "turn_to_tree"
        else:
            expected, kind = sampled, None
        replaced = expected != sampled
        if (sampled not in allowlist or event["action"] != expected or event["executed_action"] != expected
                or event["intervention_kind"] != kind or event["oracle_gate_triggered"] is not (ready or turn)
                or event["ready_gate_triggered"] is not ready or event["turn_gate_triggered"] is not turn
                or event["action_replaced"] is not replaced or event["oracle_policy_rng_unchanged"] is not True):
            raise ValueError("ready/turn intervention boundary differs")
        sampled_counts[str(sampled)] += 1
        executed_counts[str(expected)] += 1
        for key, value in zip(COUNTS, (ready or turn, ready, turn, replaced, ready and replaced, turn and replaced)):
            counts[key] += value
    if (any(episode.get(key) != value for key, value in counts.items())
            or dict(sorted(sampled_counts.items())) != episode["sampled_action_counts"]
            or dict(sorted(executed_counts.items())) != episode["action_counts"]):
        raise ValueError("episode gate/action counts differ")


def _load(root: Path, row: dict, config: dict, preflight: bool) -> list[dict]:
    path = root / Path(row["trace"]).name
    episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if _sha256(path) != row["trace_sha256"] or _json_digest(episodes) != row["trace_canonical_digest"]:
        raise RuntimeError(f"trace hash differs: {path}")
    count = config["preflight_episode_count"] if preflight else config["episode_count"]
    if len(episodes) != count:
        raise ValueError("trace episode count differs")
    check_config = {
        "seed_set": config["seed_set"], "fresh_max_steps": config["max_steps"], "wood_threshold": config["task"]["threshold"],
        "fresh_environment_seed_base": config["preflight_environment_seed_base"] if preflight else config["environment_seed_base"],
        "fresh_action_seed_base": config["preflight_action_seed_base"] if preflight else config["action_seed_base"],
        "fresh_replicate_seed_stride": config["replicate_seed_stride"], "action_allowlist": config["action_allowlist"],
    }
    for index, ep in enumerate(episodes):
        _validate_episode(ep, seed=row["seed"], variant=f"{row['head_variant']}/{row['condition']}", index=index, config=check_config)
        if ep["head_variant"] != row["head_variant"] or ep["condition"] != row["condition"]:
            raise ValueError("head/condition episode manifest differs")
        _validate_gates(ep, row["condition"], config["action_allowlist"])
    expected = {"episodes": len(episodes), "interaction_steps": sum(ep["steps"] for ep in episodes),
                "successes": sum(ep["success"] for ep in episodes),
                "terminal_counts": dict(sorted(Counter(ep["terminal_reason"] for ep in episodes).items())),
                **{key: sum(ep[key] for ep in episodes) for key in COUNTS}}
    if any(row["summary"].get(key) != value for key, value in expected.items()):
        raise ValueError("run summary differs from trace")
    return episodes


def _outcome(episodes: list[dict]) -> dict:
    return {**ready_outcome(episodes), **{key: sum(ep[key] for ep in episodes) for key in COUNTS}}


def _pairs(control: list[dict], turned: list[dict], seed: int, head: str) -> list[dict]:
    if len(control) != len(turned):
        raise ValueError("paired episode counts differ")
    rows = []
    for left, right in zip(control, turned):
        if any(left[key] != right[key] for key in (
                "environment_seed", "action_seed", "episode", "initial_rgb_digest", "initial_snapshot")):
            raise ValueError("paired initial state or seeds differ")
        prefix, first_difference = True, None
        for a, b in zip(left["events"], right["events"]):
            if a["policy_rng_after_sample_digest"] != b["policy_rng_after_sample_digest"]:
                raise ValueError("paired categorical sampling RNG differs")
            if prefix:
                if any(a[key] != b[key] for key in ("rgb_before_digest", "before", "sampled_action", "probabilities")):
                    raise ValueError("paired trajectories differ before the first incremental turn")
                # Both arms can replace a sample by do. Only an executed-action
                # difference caused by the added turn can end the shared prefix.
                if a["action"] != b["action"]:
                    if not b["turn_gate_triggered"] or not b["action_replaced"] or a["turn_gate_triggered"]:
                        raise ValueError("first paired action difference is not incremental turn")
                    prefix, first_difference = False, b["step"]
                elif any(a[key] != b[key] for key in (
                        "rgb_after_digest", "after", "wood_after", "reward", "environment_done", "terminal_reason")):
                    raise ValueError("paired transitions differ with equal executed actions")
        if prefix and left["steps"] != right["steps"]:
            raise ValueError("paired identical actions have different trajectory lengths")
        rows.append({"seed": seed, "head_variant": head, "episode": left["episode"],
                     "environment_seed": left["environment_seed"], "ready_do_success": left["success"],
                     "turn_ready_do_success": right["success"], "ready_do_final_wood": left["final_wood"],
                     "turn_ready_do_final_wood": right["final_wood"], "ready_do_steps": left["steps"],
                     "turn_ready_do_steps": right["steps"], "first_incremental_turn_step": first_difference,
                     "prefix_before_incremental_turn_identical": True, "paired_sampling_rng_identical": True})
    return rows


def _turn_outcomes(episodes: list[dict], stage: int | None = None) -> dict:
    opportunities = []
    for ep in episodes:
        for index, event in enumerate(ep["events"]):
            before = event["before"]
            if (stage is None or event["wood_before"] == stage) and (
                    before["awake_adjacent_opportunity"] and not before["ready_to_collect"]):
                following = ep["events"][index + 1] if index + 1 < len(ep["events"]) else None
                opportunities.append((event, following))
    effective = [(e, n) for e, n in opportunities if e["action"] in e["before"]["unblocked_tree_move_actions"]]
    return {
        "wood_stage": "all" if stage is None else stage, "turn_opportunity_steps": len(opportunities),
        "sampled_valid_turns": sum(e["sampled_action"] in e["before"]["unblocked_tree_move_actions"] for e, _ in opportunities),
        "executed_valid_turns": len(effective),
        "turn_gate_steps": sum(e["turn_gate_triggered"] for e, _ in opportunities),
        "turn_replacements": sum(e["turn_gate_triggered"] and e["action_replaced"] for e, _ in opportunities),
        "effective_turn_survived": sum(not e["environment_done"] for e, _ in effective),
        "effective_turn_native_terminal": sum(e["environment_done"] for e, _ in effective),
        "effective_turn_after_ready": sum(e["after"] is not None and e["after"]["ready_to_collect"] for e, _ in effective),
        "survived_effective_turn_not_ready": sum(e["after"] is not None and not e["after"]["ready_to_collect"] for e, _ in effective),
        "next_event_observed": sum(n is not None for _, n in effective),
        "next_event_missing_native_terminal": sum(n is None and e["environment_done"] for e, n in effective),
        "next_event_missing_external_truncation": sum(n is None and not e["environment_done"] for e, n in effective),
        "next_action_do": sum(n is not None and n["action"] == 5 for _, n in effective),
        "next_action_wood_gain": sum(n is not None and n["wood_gain"] == 1 for _, n in effective),
    }


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    _validate(config)
    if (summary["run_profile"] not in ("main", "preflight") or summary["formal_result"]
            or summary["policy_training_steps"] != 0 or summary["supervised_optimizer_updates"] != 0
            or summary["encoder_updated"] or summary["new_fit_or_selection_performed"]
            or summary["intervention_executor_is_rgb_only"] is not False or summary["teacher_used"] is not True
            or summary["module_registered"] or summary["knowledge_updated"] or summary["spt_updated"]
            or summary["approach_intervention_used"] or summary["survival_intervention_used"]
            or not all(summary["files_unchanged"].values())):
        raise ValueError("input is not a completed frozen local-turn diagnostic")
    preflight = summary["run_profile"] == "preflight"
    seeds = config["preflight_seed_set"] if preflight else config["seed_set"]
    if summary["selected_seeds"] != seeds or len(summary["runs"]) != len(seeds) * len(HEADS) * len(CONDITIONS):
        raise ValueError("seed/head/condition run manifest differs")
    if not preflight and (summary["cross_process_repetition"] is None or not summary["cross_process_repetition"]["passed"]):
        raise ValueError("main cross-process repeat must pass")
    hashes = _read(root / "provenance.json")["sha256_before_evaluation"]
    if not all(_sha256(Path(path)) == digest for path, digest in hashes.items()):
        raise RuntimeError("source/input hashes differ from the sealed run")
    paths = [root / name for name in ("config.json", "summary.json", "provenance.json", "selection_lock.json")]
    paths += [root / Path(row["trace"]).name for row in summary["runs"]]
    input_hashes = {str(path): _sha256(path) for path in paths}
    source_config = _read(Path(config["source_result_root"]) / "config.json")
    locks = _read(root / "selection_lock.json")
    if not locks["heads_locked_before_evaluation"] or locks["new_fit_or_selection_performed"]:
        raise ValueError("head selection boundary differs")
    _configure_torch_determinism(config)
    episodes_by_key, errors, frozen = {}, [], []
    with stable_crafter_object_order() as order:
        if summary["environment_order_version"] != order:
            raise RuntimeError("replay wrapper differs")
        for seed in seeds:
            seed_locks = [row for row in locks["heads"] if row["seed"] == seed]
            if _sha256(Path(seed_locks[0]["checkpoint"])) != seed_locks[0]["checkpoint_sha256"]:
                raise RuntimeError("checkpoint hash differs")
            checkpoint = torch.load(seed_locks[0]["checkpoint"], map_location="cuda", weights_only=False)
            if checkpoint["actual_interaction_steps"] != config["checkpoint_steps"] or checkpoint["environment_order_version"] != order:
                raise RuntimeError("checkpoint step budget or wrapper differs")
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for p in source.parameters():
                p.requires_grad_(False)
                p.grad = None
            original = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            for head in HEADS:
                lock = next(row for row in seed_locks if row["head_variant"] == head)
                if _sha256(Path(lock["artifact"])) != lock["artifact_sha256"]:
                    raise RuntimeError("head artifact hash differs")
                artifact = torch.load(lock["artifact"], map_location="cuda", weights_only=False)
                if (artifact["source_checkpoint_sha256"] != lock["checkpoint_sha256"]
                        or artifact["head_state_canonical_digest"] != lock["head_digest"]
                        or artifact["selection_uses_heldout"] is not False or artifact["diagnostic_variant"] != head):
                    raise RuntimeError("matched artifact provenance differs")
                policy = load_saved_head(source, artifact).eval()
                for p in policy.parameters():
                    p.requires_grad_(False)
                    p.grad = None
                before = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                for condition in CONDITIONS:
                    rows = [row for row in summary["runs"] if (row["seed"], row["head_variant"], row["condition"]) == (seed, head, condition)]
                    if len(rows) != 1:
                        raise ValueError("duplicate or absent head/condition run")
                    episodes = _load(root, rows[0], config, preflight)
                    max_error = max(_replay(policy, ep, config) for ep in episodes)
                    episodes_by_key[seed, head, condition] = episodes
                    errors.append({"seed": seed, "head_variant": head, "condition": condition,
                                   "episodes_replayed": len(episodes), "max_probability_error": max_error})
                    print(f"turn-ready-do verified {head}/{condition} seed {seed}", flush=True)
                if (not _state_equal(before, (policy.state_dict(), policy.optimizer.state_dict()))
                        or not _state_equal(original, (source.state_dict(), source.optimizer.state_dict()))
                        or any(p.grad is not None for p in policy.parameters()) or not next(policy.parameters()).is_cuda):
                    raise RuntimeError("frozen state or CUDA boundary failed during replay")
                frozen.append({"seed": seed, "head_variant": head, "policy_and_optimizer_unchanged": True,
                               "source_policy_and_optimizer_unchanged": True, "real_cuda": True})
    outcomes, stages, turns, pairs, pooled, paired = [], [], [], [], [], []
    for (seed, head, condition), episodes in episodes_by_key.items():
        group = {"seed": seed, "head_variant": head, "condition": condition}
        outcomes.append({**group, **_outcome(episodes)})
        stages += [{**group, **row} for row in _stages(episodes)]
        turns += [{**group, **_turn_outcomes(episodes, stage)} for stage in (None, 0, 1, 2)]
    for head in HEADS:
        for seed in seeds:
            pairs += _pairs(episodes_by_key[seed, head, "ready_do"], episodes_by_key[seed, head, "turn_ready_do"], seed, head)
        for condition in CONDITIONS:
            episodes = [ep for seed in seeds for ep in episodes_by_key[seed, head, condition]]
            group = {"seed": "all", "head_variant": head, "condition": condition}
            pooled.append({"head_variant": head, "condition": condition, **_outcome(episodes)})
            stages += [{**group, **row} for row in _stages(episodes)]
            turns += [{**group, **_turn_outcomes(episodes, stage)} for stage in (None, 0, 1, 2)]
        p = [row for row in pairs if row["head_variant"] == head]
        paired.append({"head_variant": head, "paired_episodes": len(p),
                       "ready_do_only_successes": sum(row["ready_do_success"] and not row["turn_ready_do_success"] for row in p),
                       "turn_ready_do_only_successes": sum(not row["ready_do_success"] and row["turn_ready_do_success"] for row in p),
                       "both_successes": sum(row["ready_do_success"] and row["turn_ready_do_success"] for row in p),
                       "neither_successes": sum(not row["ready_do_success"] and not row["turn_ready_do_success"] for row in p)})
    if not preflight:
        repeat_root = root / "cross_process_repeat"
        repeated = _read(repeat_root / "summary.json")
        if repeated["run_profile"] != "repeat_seed0" or len(repeated["runs"]) != len(HEADS) * len(CONDITIONS):
            raise ValueError("independent repeat manifest differs")
        repeated_keys = set()
        for row in repeated["runs"]:
            key = row["seed"], row["head_variant"], row["condition"]
            if row["seed"] != config["seed_set"][0] or key in repeated_keys:
                raise ValueError("duplicate or incorrect repeat key")
            repeated_keys.add(key)
            episodes = _load(repeat_root, row, config, False)
            if _json_digest(episodes) != _json_digest(episodes_by_key[key]):
                raise RuntimeError("independently loaded repeat trace differs")
    if len(summary["reference_audits"]) != len(seeds) * len(HEADS) or not all(
            row["ready_do_reference_runner_trajectory_identical"] for row in summary["reference_audits"]):
        raise ValueError("ready-do reference runner audit failed")
    if summary["evaluation_interaction_steps"] != sum(row["interaction_steps"] for row in outcomes):
        raise ValueError("total interaction step count differs")
    if not all(_sha256(Path(path)) == digest for path, digest in {**hashes, **input_hashes}.items()):
        raise RuntimeError("input or source changed during independent replay")
    output.mkdir(parents=True)
    _write_csv(output / "outcomes.csv", outcomes)
    _write_csv(output / "pooled_outcomes.csv", pooled)
    _write_csv(output / "stage_funnel.csv", stages)
    _write_csv(output / "turn_outcomes.csv", turns)
    _write_csv(output / "paired_episodes.csv", pairs)
    verification = {
        "formal_result": False, "run_profile": summary["run_profile"], "oracle_intervention_only": True,
        "module_registered": False, "policy_training_steps": 0, "encoder_updated": False,
        "approach_intervention_used": False, "survival_intervention_used": False,
        "input_files_unchanged": True, "rgb_policy_sampling_and_environment_replay_passed": True,
        "replay_checks": errors, "frozen_checks": frozen, "reference_audits_passed": True,
        "cross_process_repetition_passed": not preflight,
        "paired_prefix_before_incremental_turn_and_sampling_rng_checks_passed": True,
        "pooled_outcomes": pooled, "paired_outcomes": paired,
        "analysis_source_sha256": _sha256(Path(__file__)), "input_hashes": input_hashes,
        "output_sha256": {p.name: _sha256(p) for p in sorted(output.iterdir())},
    }
    _write(output / "verification.json", verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(json.dumps({"pooled_outcomes": result["pooled_outcomes"], "paired_outcomes": result["paired_outcomes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
