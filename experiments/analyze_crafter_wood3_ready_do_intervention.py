"""Independent RGB-policy replay and paired analysis for ready-to-do gating."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from experiments.analyze_crafter_wood3_action_consequence import _validate_episode
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism, _image_tensor
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_collection_diagnostics import collection_snapshot
from src.environments.crafter_determinism import stable_crafter_object_order


HEADS = ("matched_baseline_only", "matched_mixed_coverage")
CONDITIONS = ("native", "ready_do")


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load(root: Path, row: dict, config: dict, preflight: bool) -> list[dict]:
    path = root / Path(row["trace"]).name
    episodes = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if _sha256(path) != row["trace_sha256"] or _json_digest(episodes) != row["trace_canonical_digest"]:
        raise RuntimeError(f"trace hash differs: {path}")
    count = config["preflight_episode_count"] if preflight else config["episode_count"]
    if len(episodes) != count:
        raise ValueError("trace episode count differs")
    check_config = {
        "seed_set": config["seed_set"], "fresh_max_steps": config["max_steps"], "wood_threshold": 3,
        "fresh_environment_seed_base": config["preflight_environment_seed_base"] if preflight else config["environment_seed_base"],
        "fresh_action_seed_base": config["preflight_action_seed_base"] if preflight else config["action_seed_base"],
        "fresh_replicate_seed_stride": config["replicate_seed_stride"], "action_allowlist": config["action_allowlist"],
    }
    variant = f"{row['head_variant']}/{row['condition']}"
    for index, ep in enumerate(episodes):
        _validate_episode(ep, seed=row["seed"], variant=variant, index=index, config=check_config)
        if ep["head_variant"] != row["head_variant"] or ep["condition"] != row["condition"]:
            raise ValueError("head/condition episode manifest differs")
        sampled_counts, executed_counts = Counter(), Counter()
        for event in ep["events"]:
            gate = row["condition"] == "ready_do" and event["before"]["ready_to_collect"]
            expected_action = 5 if gate else event["sampled_action"]
            if (event["executed_action"] != expected_action or event["action"] != expected_action
                    or event["oracle_gate_triggered"] is not gate
                    or event["action_replaced"] != (expected_action != event["sampled_action"])
                    or event["oracle_policy_rng_unchanged"] is not True
                    or event["sampled_action"] not in config["action_allowlist"]):
                raise ValueError("ready-only action intervention boundary differs")
            sampled_counts[str(event["sampled_action"])] += 1
            executed_counts[str(event["action"])] += 1
        if (dict(sorted(sampled_counts.items())) != ep["sampled_action_counts"]
                or dict(sorted(executed_counts.items())) != ep["action_counts"]
                or ep["oracle_gate_steps"] != sum(x["oracle_gate_triggered"] for x in ep["events"])
                or ep["replaced_action_steps"] != sum(x["action_replaced"] for x in ep["events"])):
            raise ValueError("recorded sampled/executed action counts differ")
    expected = {
        "episodes": len(episodes), "interaction_steps": sum(ep["steps"] for ep in episodes),
        "successes": sum(ep["success"] for ep in episodes),
        "terminal_counts": dict(sorted(Counter(ep["terminal_reason"] for ep in episodes).items())),
        "oracle_gate_steps": sum(ep["oracle_gate_steps"] for ep in episodes),
        "replaced_action_steps": sum(ep["replaced_action_steps"] for ep in episodes),
    }
    if any(row["summary"].get(key) != value for key, value in expected.items()):
        raise ValueError("run summary differs from trace")
    return episodes


def _replay(policy, episode: dict, config: dict) -> float:
    adapter = CrafterEnvironmentAdapter(seed=episode["environment_seed"], length=config["environment_length"], diagnostics=True)
    torch.manual_seed(episode["action_seed"])
    max_error = 0.0
    try:
        observation = adapter.reset()
        if _json_digest(observation.tolist()) != episode["initial_rgb_digest"]:
            raise RuntimeError("replay initial RGB differs")
        for event in episode["events"]:
            if _json_digest(observation.tolist()) != event["rgb_before_digest"]:
                raise RuntimeError("replay before RGB differs")
            with torch.no_grad():
                distribution, _, _ = policy.distribution_value(
                    _image_tensor(observation, torch.device("cuda")), None, config["action_allowlist"])
                sampled = int(distribution.sample().reshape(-1)[0].item())
                probabilities = distribution.probs.reshape(-1).detach().cpu().tolist()
            rng = torch.cuda.get_rng_state().clone()
            if (sampled != event["sampled_action"]
                    or hashlib.sha256(rng.numpy().tobytes()).hexdigest() != event["policy_rng_after_sample_digest"]):
                raise RuntimeError("replayed original policy sample or sampling RNG differs")
            error = max(abs(left - right) for left, right in zip(probabilities, event["probabilities"]))
            max_error = max(max_error, error)
            if error > 1e-6:
                raise RuntimeError("replayed RGB action probabilities differ")
            before = collection_snapshot(adapter, hostile_radius=config["hostile_radius"])
            if before != event["before"] or not torch.equal(rng, torch.cuda.get_rng_state()):
                raise RuntimeError("replay snapshot or oracle sampling-RNG audit differs")
            observation, reward, done, info = adapter.step(event["executed_action"])
            after = collection_snapshot(adapter, hostile_radius=config["hostile_radius"]) if not done else None
            if (_json_digest(observation.tolist()) != event["rgb_after_digest"]
                    or after != event["after"] or bool(done) != event["environment_done"]
                    or reward != event["reward"] or int(info["inventory"]["wood"]) != event["wood_after"]):
                raise RuntimeError("replayed executed transition differs")
    finally:
        adapter.close()
    return max_error


def _outcome(episodes: list[dict]) -> dict:
    events = [e for ep in episodes for e in ep["events"]]
    return {"episodes": len(episodes), "successes": sum(ep["success"] for ep in episodes),
            "success_rate": sum(ep["success"] for ep in episodes) / len(episodes),
            "deaths": sum(ep["terminal_reason"] == "death" for ep in episodes),
            "external_truncations": sum(ep["terminal_reason"] == "external_truncation" for ep in episodes),
            "native_terminal_events": sum(e["environment_done"] for e in events),
            "successful_but_native_terminal_episodes": sum(ep["success"] and ep["events"][-1]["environment_done"] for ep in episodes),
            "interaction_steps": len(events), "wood_gained": sum(e["wood_gain"] for e in events),
            "oracle_gate_steps": sum(e["oracle_gate_triggered"] for e in events),
            "replaced_action_steps": sum(e["action_replaced"] for e in events),
            "ready_steps": sum(e["before"]["ready_to_collect"] for e in events),
            "ready_do_actions": sum(e["before"]["ready_to_collect"] and e["action"] == 5 for e in events),
            "ready_blocked_moves": sum(e["before"]["ready_to_collect"] and e["action"] in (1, 2, 3, 4)
                                       and e["position_after"] == e["before"]["position"] for e in events)}


def _stages(episodes: list[dict]) -> list[dict]:
    result = []
    for stage in (0, 1, 2):
        entered = [ep for ep in episodes if ep["final_wood"] >= stage]
        completed = [ep for ep in entered if ep["final_wood"] > stage]
        failed = [ep for ep in entered if ep["final_wood"] == stage]
        classes = Counter()
        for ep in failed:
            rows = [e for e in ep["events"] if e["wood_before"] == stage]
            label = "no_observed_stage_step" if not rows else (
                "no_unblocked_adjacent_tree_observed" if not any(e["before"]["unblocked_adjacent_tree"] for e in rows) else
                "sleep_or_capacity_blocked" if not any(e["before"]["awake_adjacent_opportunity"] for e in rows) else
                "never_facing_ready_tree" if not any(e["before"]["ready_to_collect"] for e in rows) else
                "ready_but_no_collection")
            classes[label] += 1
        result.append({"wood_stage": stage, "entered": len(entered), "completed": len(completed),
                       "completion_rate": len(completed) / len(entered) if entered else None,
                       "deaths": sum(ep["terminal_reason"] == "death" for ep in failed),
                       "external_truncations": sum(ep["terminal_reason"] == "external_truncation" for ep in failed),
                       **{f"failure_{name}": classes[name] for name in (
                           "no_observed_stage_step", "no_unblocked_adjacent_tree_observed",
                           "sleep_or_capacity_blocked", "never_facing_ready_tree", "ready_but_no_collection")}})
    return result


def _pairs(native: list[dict], gated: list[dict], seed: int, head: str) -> list[dict]:
    rows = []
    if len(native) != len(gated):
        raise ValueError("paired episode counts differ")
    for left, right in zip(native, gated):
        if any(left[key] != right[key] for key in (
                "environment_seed", "action_seed", "episode", "initial_rgb_digest", "initial_snapshot")):
            raise ValueError("paired initial state or seeds differ")
        prefix = True
        first_replaced_step = None
        for a, b in zip(left["events"], right["events"]):
            if a["policy_rng_after_sample_digest"] != b["policy_rng_after_sample_digest"]:
                raise ValueError("paired categorical sampling RNG rule differs")
            if prefix:
                if any(a[key] != b[key] for key in ("rgb_before_digest", "before", "sampled_action", "probabilities")):
                    raise ValueError("paired trajectories differ before the first executed action replacement")
                if b["action_replaced"]:
                    prefix = False
                    first_replaced_step = b["step"]
                elif any(a[key] != b[key] for key in ("action", "rgb_after_digest", "after", "wood_after", "environment_done")):
                    raise ValueError("paired transitions differ without an executed action replacement")
        if prefix and left["steps"] != right["steps"]:
            raise ValueError("unmodified paired trajectories have different lengths")
        rows.append({"seed": seed, "head_variant": head, "episode": left["episode"],
                     "environment_seed": left["environment_seed"], "native_success": left["success"],
                     "ready_do_success": right["success"], "native_final_wood": left["final_wood"],
                     "ready_do_final_wood": right["final_wood"], "native_steps": left["steps"],
                     "ready_do_steps": right["steps"], "first_replaced_step": first_replaced_step,
                     "prefix_before_replacement_identical": True, "paired_sampling_rng_identical": True})
    return rows


def run(input_path: str, output_path: str) -> dict:
    root, output = Path(input_path), Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    config, summary = _read(root / "config.json"), _read(root / "summary.json")
    if (summary["formal_result"] or summary["policy_training_steps"] != 0
            or summary["encoder_updated"] or summary["intervention_executor_is_rgb_only"] is not False
            or not all(summary["files_unchanged"].values())):
        raise ValueError("input is not a completed frozen oracle diagnostic")
    preflight = summary["run_profile"] == "preflight"
    seeds = config["preflight_seed_set"] if preflight else config["seed_set"]
    if summary["selected_seeds"] != seeds:
        raise ValueError("main/preflight selected seeds differ")
    if not preflight and (summary["cross_process_repetition"] is None or not summary["cross_process_repetition"]["passed"]):
        raise ValueError("main cross-process repeat must pass")
    if len(summary["runs"]) != len(seeds) * len(HEADS) * len(CONDITIONS):
        raise ValueError("head/condition run count differs")
    provenance = _read(root / "provenance.json")
    hashes = provenance["sha256_before_evaluation"]
    if not all(_sha256(Path(path)) == digest for path, digest in hashes.items()):
        raise RuntimeError("source/input hashes differ from the sealed run")
    paths = [root / name for name in ("config.json", "summary.json", "provenance.json", "selection_lock.json")]
    paths += [root / Path(row["trace"]).name for row in summary["runs"]]
    paths += [Path(__file__), Path("experiments/analyze_crafter_wood3_action_consequence.py")]
    input_hashes = {str(path): _sha256(path) for path in paths}
    source_config = _read(Path(config["source_result_root"]) / "config.json")
    locks = _read(root / "selection_lock.json")
    _configure_torch_determinism(config)
    episodes_by_key, errors, frozen = {}, [], []
    with stable_crafter_object_order() as order:
        if summary["environment_order_version"] != order:
            raise RuntimeError("replay wrapper differs")
        for seed in seeds:
            seed_locks = [row for row in locks["heads"] if row["seed"] == seed]
            checkpoint = torch.load(seed_locks[0]["checkpoint"], map_location="cuda", weights_only=False)
            source = build_matched_spatial_policy(source_config["policy"], "cnn_only").cuda()
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for p in source.parameters():
                p.requires_grad_(False)
                p.grad = None
            for head in HEADS:
                lock = next(row for row in seed_locks if row["head_variant"] == head)
                if _sha256(Path(lock["artifact"])) != lock["artifact_sha256"]:
                    raise RuntimeError("locked head artifact hash differs")
                artifact = torch.load(lock["artifact"], map_location="cuda", weights_only=False)
                policy = load_saved_head(source, artifact).eval()
                for p in policy.parameters():
                    p.requires_grad_(False)
                    p.grad = None
                before = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                for condition in CONDITIONS:
                    rows = [row for row in summary["runs"]
                            if (row["seed"], row["head_variant"], row["condition"]) == (seed, head, condition)]
                    if len(rows) != 1:
                        raise ValueError("duplicate or absent head/condition run")
                    episodes = _load(root, rows[0], config, preflight)
                    max_error = max(_replay(policy, ep, config) for ep in episodes)
                    episodes_by_key[seed, head, condition] = episodes
                    errors.append({"seed": seed, "head_variant": head, "condition": condition,
                                   "episodes_replayed": len(episodes), "max_probability_error": max_error})
                    print(f"ready-do verified {head}/{condition} seed {seed}", flush=True)
                unchanged = _state_equal(before, (policy.state_dict(), policy.optimizer.state_dict()))
                if not unchanged or any(p.grad is not None for p in policy.parameters()):
                    raise RuntimeError("policy state changed during independent replay")
                frozen.append({"seed": seed, "head_variant": head, "policy_and_optimizer_unchanged": True})
    outcomes, stages, pairs = [], [], []
    for (seed, head, condition), episodes in episodes_by_key.items():
        group = {"seed": seed, "head_variant": head, "condition": condition}
        outcomes.append({**group, **_outcome(episodes)})
        stages += [{**group, **row} for row in _stages(episodes)]
    pooled, paired = [], []
    for head in HEADS:
        for seed in seeds:
            pairs += _pairs(episodes_by_key[seed, head, "native"], episodes_by_key[seed, head, "ready_do"], seed, head)
        for condition in CONDITIONS:
            episodes = [ep for seed in seeds for ep in episodes_by_key[seed, head, condition]]
            pooled.append({"head_variant": head, "condition": condition, **_outcome(episodes)})
            stages += [{"seed": "all", "head_variant": head, "condition": condition, **row}
                       for row in _stages(episodes)]
        p = [row for row in pairs if row["head_variant"] == head]
        paired.append({"head_variant": head, "paired_episodes": len(p),
                       "native_only_successes": sum(row["native_success"] and not row["ready_do_success"] for row in p),
                       "ready_do_only_successes": sum(not row["native_success"] and row["ready_do_success"] for row in p),
                       "both_successes": sum(row["native_success"] and row["ready_do_success"] for row in p),
                       "neither_successes": sum(not row["native_success"] and not row["ready_do_success"] for row in p)})
    if not preflight:
        repeat_root = root / "cross_process_repeat"
        repeated = _read(repeat_root / "summary.json")
        if len(repeated["runs"]) != len(HEADS) * len(CONDITIONS):
            raise ValueError("independent repeat run count differs")
        for row in repeated["runs"]:
            episodes = _load(repeat_root, row, config, False)
            if _json_digest(episodes) != _json_digest(episodes_by_key[row["seed"], row["head_variant"], row["condition"]]):
                raise RuntimeError("independently loaded repeat trace differs")
    if len(summary["reference_audits"]) != len(seeds) * len(HEADS) or not all(
            row["old_runner_without_oracle_trajectory_identical"] for row in summary["reference_audits"]):
        raise ValueError("native old-runner reference audit failed")
    if not all(_sha256(Path(path)) == digest for path, digest in {**hashes, **input_hashes}.items()):
        raise RuntimeError("input or source changed during independent replay")
    output.mkdir(parents=True)
    _write_csv(output / "outcomes.csv", outcomes)
    _write_csv(output / "pooled_outcomes.csv", pooled)
    _write_csv(output / "stage_funnel.csv", stages)
    _write_csv(output / "paired_episodes.csv", pairs)
    verification = {"formal_result": False, "run_profile": summary["run_profile"],
                    "oracle_intervention_only": True, "module_registered": False,
                    "policy_training_steps": 0, "encoder_updated": False,
                    "input_files_unchanged": True, "rgb_policy_sampling_and_environment_replay_passed": True,
                    "replay_checks": errors, "frozen_checks": frozen,
                    "reference_audits_passed": True, "cross_process_repetition_passed": not preflight,
                    "paired_prefix_and_sampling_rng_checks_passed": True,
                    "pooled_outcomes": pooled, "paired_outcomes": paired,
                    "analysis_source_sha256": _sha256(Path(__file__)), "input_hashes": input_hashes,
                    "output_sha256": {p.name: _sha256(p) for p in sorted(output.iterdir())}}
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
