"""Paired-outcome and validation-curve summary for a verified coverage pilot."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from itertools import combinations
from pathlib import Path

from experiments.run_crafter_spatial_training_determinism_audit import _json_digest
from experiments.run_crafter_wood3_collection_opportunity import _read, _write


ARMS = ("source_cnn_only", "old_native_216", "new_native_216", "merged_native_432")


def exact_mcnemar(left_only: int, right_only: int) -> float:
    """Two-sided exact conditional binomial test for paired binary outcomes."""
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts cannot be negative")
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    lower = min(left_only, right_only)
    tail = sum(math.comb(discordant, index) for index in range(lower + 1)) / (2 ** discordant)
    return min(1.0, 2 * tail)


def wilson_interval(successes: int, episodes: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if episodes < 1 or not 0 <= successes <= episodes:
        raise ValueError("successes and episode count are inconsistent")
    rate = successes / episodes
    denominator = 1 + z * z / episodes
    center = (rate + z * z / (2 * episodes)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / episodes + z * z / (4 * episodes * episodes)) / denominator
    return center - half, center + half


def _read_episodes(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(input_path: str) -> dict:
    root = Path(input_path)
    analysis = root / "analysis_verified"
    verification = _read(analysis / "verification.json")
    if (verification.get("formal_result") is not False
            or verification.get("independent_fresh_episode_replay_passed") is not True
            or verification.get("cross_process_repeat_passed") is not True):
        raise ValueError("independent verification must pass before paired summary")
    summary = _read(root / "summary.json")
    if summary["cross_process_repetition"].get("passed") is not True:
        raise ValueError("runner cross-process repetition must pass")
    episodes_by_arm = {}
    for arm in ARMS:
        episodes = _read_episodes(root / f"seed0_{arm}_fresh.jsonl")
        trace = next(row for row in summary["fresh_traces"] if row["arm"] == arm)
        if len(episodes) != summary["fresh_arms"][arm]["episodes"]:
            raise ValueError(f"episode count differs for {arm}")
        if _json_digest(episodes) != trace["canonical_digest"]:
            raise ValueError(f"canonical trace digest differs for {arm}")
        episodes_by_arm[arm] = episodes
    count = len(episodes_by_arm[ARMS[0]])
    if any(len(episodes_by_arm[arm]) != count for arm in ARMS):
        raise ValueError("paired arms have different episode counts")
    manifest = []
    for index in range(count):
        seeds = {(episodes_by_arm[arm][index]["environment_seed"],
                  episodes_by_arm[arm][index]["action_seed"]) for arm in ARMS}
        if len(seeds) != 1:
            raise ValueError(f"fresh episode seed pairing differs at index {index}")
        environment_seed, action_seed = next(iter(seeds))
        row = {"episode": index, "environment_seed": environment_seed, "action_seed": action_seed}
        for arm in ARMS:
            episode = episodes_by_arm[arm][index]
            row[f"{arm}_success"] = bool(episode["success"])
            row[f"{arm}_final_wood"] = int(episode["final_wood"])
            row[f"{arm}_terminal_reason"] = episode["terminal_reason"]
        manifest.append(row)
    _write_csv(analysis / "paired_success_outcomes.csv", manifest)
    paired_rows = []
    paired_results = {}
    for left, right in combinations(ARMS, 2):
        left_only = sum(bool(row[f"{left}_success"]) and not bool(row[f"{right}_success"]) for row in manifest)
        right_only = sum(bool(row[f"{right}_success"]) and not bool(row[f"{left}_success"]) for row in manifest)
        p_value = exact_mcnemar(left_only, right_only)
        paired_rows.append({"left_arm": left, "right_arm": right,
                            "left_only_success": left_only, "right_only_success": right_only,
                            "discordant_pairs": left_only + right_only,
                            "two_sided_exact_mcnemar_p": p_value})
        paired_results[f"{left}_vs_{right}"] = paired_rows[-1]
    _write_csv(analysis / "paired_success_tests.csv", paired_rows)

    validation_rows, selection = [], {}
    for arm in ARMS[1:]:
        artifact = __import__("torch").load(root / f"seed0_{arm}.pt", map_location="cpu", weights_only=False)
        selection[arm] = {"selected_epoch": artifact["selected_epoch"],
                          "selected_learning_rate": artifact["selected_learning_rate"],
                          "selected_validation_metrics": artifact["validation_metrics"]}
        for trial in artifact["trials"]:
            for entry in trial["history"]:
                metric = entry["validation_metrics"]
                validation_rows.append({"arm": arm, "learning_rate": trial["learning_rate"],
                                        "epoch": entry["epoch"],
                                        "category_macro_accuracy": metric["category_macro_accuracy"],
                                        "category_macro_set_nll": metric["category_macro_set_nll"]})
    _write_csv(analysis / "validation_learning_curves.csv", validation_rows)

    arm_summary = {}
    for arm in ARMS:
        episodes = episodes_by_arm[arm]
        successes = sum(bool(row["success"]) for row in episodes)
        low, high = wilson_interval(successes, len(episodes))
        arm_summary[arm] = {"episodes": len(episodes), "successes": successes,
                             "success_rate": successes / len(episodes),
                             "wilson_95_percent_interval": [low, high],
                             "terminal_counts": summary["fresh_arms"][arm]["terminal_counts"],
                             "wood_milestone_episode_counts": summary["fresh_arms"][arm]["wood_milestone_episode_counts"],
                             "death_stage_counts": summary["fresh_arms"][arm]["death_stage_counts"]}
    output = {"formal_result": False, "diagnostic_only": True,
              "input_verification_passed": True,
              "episode_count_per_arm": count,
              "arms": arm_summary, "paired_success_tests": paired_results,
              "validation_head_selection": selection,
              "summarizer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    _write(analysis / "paired_outcome_summary.json", output)
    public = _read(analysis / "public_summary.json")
    public["paired_outcome_summary"] = output
    _write(analysis / "public_summary.json", public)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input), indent=2), flush=True)
