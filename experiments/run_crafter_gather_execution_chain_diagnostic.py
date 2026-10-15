"""Read-only execution-chain diagnostic for the gather transfer pilot.

The native scripted route establishes feasibility and exposes common RGB
states at approach, turn, and do boundaries. Frozen policies only report
probabilities; their outputs are not executed or used for parameter updates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_material_gather_scene import MaterialGatherEnv, make_material_gather_scene
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


CENTER = (32, 32)
DELTA_TO_ACTION = {(-1, 0): 1, (1, 0): 2, (0, -1): 3, (0, 1): 4}
ACTION_TO_DELTA = {value: key for key, value in DELTA_TO_ACTION.items()}
MATERIAL_TO_ITEM = {"tree": "wood", "stone": "stone", "coal": "coal"}


def _move(position, action: int, blocked):
    delta = ACTION_TO_DELTA[action]
    neighbor = (position[0] + delta[0], position[1] + delta[1])
    if neighbor in blocked or not (1 <= neighbor[0] < 63 and 1 <= neighbor[1] < 63):
        neighbor = position
    return neighbor, delta


def _shortest_action_set(position, facing, target, blocked) -> tuple[int, tuple[int, ...]]:
    """BFS over position AND facing; retain every first action on a shortest route.

    All movement uses the native turn-before-move rule. The final do costs one
    action. This geometry oracle is used only to label diagnostic states.
    """
    frontier = {(position, facing): set()}
    seen = set(frontier)
    depth = 0
    while frontier:
        good = set()
        ready = False
        for (point, direction), first_actions in frontier.items():
            if (point[0] + direction[0], point[1] + direction[1]) == target:
                ready = True
                good.update(first_actions)
        if ready:
            return depth + 1, tuple(sorted(good)) if depth else (5,)
        following = {}
        for (point, _direction), first_actions in frontier.items():
            for action in ACTION_TO_DELTA:
                state = _move(point, action, blocked)
                if state in seen:
                    continue
                following.setdefault(state, set()).update(first_actions or {action})
        seen.update(following)
        frontier = following
        depth += 1
    raise RuntimeError(f"no collection route to {target}")


def build_native_route(scene, material: str) -> list[dict]:
    """Return a shortest route, including initial facing and tied action sets."""
    if material not in scene.positions:
        raise ValueError(f"unknown material: {material}")
    target = scene.positions[material]
    blocked = set(scene.positions.values())
    position, facing = CENTER, scene.facing
    route = []
    while True:
        distance, actions = _shortest_action_set(position, facing, target, blocked)
        action = min(actions)
        after, direction = _move(position, action, blocked) if action != 5 else (position, facing)
        route.append({
            "stage": "do" if action == 5 else ("approach" if after != position else "turn"),
            "action": action,
            "shortest_actions": list(actions),
            "remaining_shortest_steps": distance,
            "expected_position_after": list(after),
            "expected_facing_after": list(direction),
        })
        if action == 5:
            return route
        position, facing = after, direction


def _image_tensor(observation: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(observation, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0


def _load_policy(checkpoint: Path, device: torch.device):
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    policy = build_matched_spatial_policy(payload["policy_config"], payload["representation"]).to(device)
    policy.load_state_dict(payload["state_dict"])
    policy.eval()
    return policy, payload


def _policy_probe(policy, observation: np.ndarray, device: torch.device) -> np.ndarray:
    with torch.no_grad():
        distribution, _value, _hidden = policy.distribution_value(
            _image_tensor(observation, device), None, tuple(range(7))
        )
    return distribution.probs.reshape(-1).detach().cpu().numpy()


def _policy_digest(policy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _run_chain(scene, material: str, horizon: int, policy=None, device=None) -> dict:
    item = MATERIAL_TO_ITEM[material]
    route = build_native_route(scene, material)
    env = MaterialGatherEnv(scene, seed=scene.layout_seed, length=horizon)
    events = []
    try:
        observation = env.reset()
        initial = int(env._player.inventory[item])
        for step, row in enumerate(route, start=1):
            probabilities = _policy_probe(policy, observation, device) if policy is not None else None
            rgb_sha256 = hashlib.sha256(observation.tobytes()).hexdigest()
            before_position = tuple(int(value) for value in env._player.pos)
            before_facing = tuple(int(value) for value in env._player.facing)
            before_item = int(env._player.inventory[item])
            observation, _reward, done, info = env.step(int(row["action"]))
            current = int(info["inventory"][item])
            event = {
                "layout_seed": int(scene.layout_seed),
                "material": material,
                "rgb_sha256": rgb_sha256,
                "step": step,
                **row,
                "position_before": list(before_position),
                "facing_before": list(before_facing),
                "position_after": [int(value) for value in env._player.pos],
                "facing_after": [int(value) for value in env._player.facing],
                "item_before": before_item,
                "item_after": current,
                "done": bool(done),
            }
            event["native_step_valid"] = bool(
                event["position_after"] == row["expected_position_after"]
                and event["facing_after"] == row["expected_facing_after"]
                and (current == before_item + 1 if row["stage"] == "do" else current == before_item)
            )
            if probabilities is not None:
                event.update({
                    "shortest_action_set_probability": float(sum(float(probabilities[action]) for action in row["shortest_actions"])),
                    "top_action": int(np.argmax(probabilities)),
                    "top_action_in_shortest_set": bool(int(np.argmax(probabilities)) in row["shortest_actions"]),
                    "allowed_action_probabilities": [float(value) for value in probabilities[:7]],
                    "do_probability": float(probabilities[5]),
                })
            events.append(event)
            if done or current > initial:
                break
        final_item = int(env._player.inventory[item])
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
    return {
        "layout_seed": int(scene.layout_seed),
        "material": material,
        "horizon": int(horizon),
        "route_length": len(route),
        "route_stages": [row["stage"] for row in route],
        "success": final_item > initial,
        "steps_executed": len(events),
        "first_gain_step": next((event["step"] for event in events if event["item_after"] > initial), None),
        "events": events,
    }


def _mechanical_summary(rows: list[dict]) -> dict:
    by_horizon = {}
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        subset = [row for row in rows if int(row["horizon"]) == horizon]
        by_horizon[str(horizon)] = {
            "cases": len(subset),
            "successes": sum(int(row["success"]) for row in subset),
            "success_rate": sum(int(row["success"]) for row in subset) / max(len(subset), 1),
        }
    stage_rows = [row for row in rows if int(row["horizon"]) == max(int(item["horizon"]) for item in rows)]
    stage_counts = Counter()
    stage_successes = Counter()
    for row in stage_rows:
        for event in row["events"]:
            stage_counts[event["stage"]] += 1
            stage_successes[event["stage"]] += int(event["native_step_valid"])
    return {
        "horizon_curves": by_horizon,
        "route_length_min": min(int(row["route_length"]) for row in rows),
        "route_length_max": max(int(row["route_length"]) for row in rows),
        "route_length_mean": sum(int(row["route_length"]) for row in rows) / max(len(rows), 1),
        "stage_single_step": {
            stage: {
                "executed": int(stage_counts[stage]),
                "native_successes": int(stage_successes[stage]),
                "success_rate": stage_successes[stage] / max(stage_counts[stage], 1),
            }
            for stage in ("approach", "turn", "do")
        },
    }


def _policy_summary(rows: list[dict]) -> dict:
    result = {}
    for stage in ("approach", "turn", "do"):
        subset = [row for row in rows if row["stage"] == stage]
        probabilities = [float(row["shortest_action_set_probability"]) for row in subset]
        result[stage] = {
            "observations": len(subset),
            "mean_shortest_action_set_probability": sum(probabilities) / max(len(probabilities), 1),
            "median_shortest_action_set_probability": float(np.median(probabilities)) if probabilities else None,
            "uniform_allowed_action_baseline": sum(len(row["shortest_actions"]) / 7 for row in subset) / max(len(subset), 1),
            "top1_in_shortest_set_rate": sum(int(row["top_action_in_shortest_set"]) for row in subset) / max(len(subset), 1),
        }
    return result


def run(input_root: str, output_path: str, *, layout_count: int = 32, device_name: str = "cuda") -> dict:
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if int(layout_count) <= 0:
        raise ValueError("layout_count must be positive")
    root = Path(input_root)
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    device = torch.device(device_name)
    mechanical_rows = []
    policy_summaries = {}
    policy_rows = []
    horizons = (1, 2, 4, 8, 16, 32)
    for layout_seed in range(int(layout_count)):
        scene = make_material_gather_scene(layout_seed)
        for material in ("tree", "stone", "coal"):
            for horizon in horizons:
                mechanical_rows.append(_run_chain(scene, material, horizon))
    for run_record in aggregate["runs"]:
        checkpoint = Path(run_record["target_policy_checkpoint"])
        policy, payload = _load_policy(checkpoint, device)
        if payload["arm"] != run_record["arm"] or int(payload["seed"]) != int(run_record["seed"]):
            raise ValueError(f"checkpoint identity mismatch: {checkpoint}")
        before_digest = _policy_digest(policy)
        rows = []
        for layout_seed in range(int(layout_count)):
            scene = make_material_gather_scene(layout_seed)
            row = _run_chain(scene, "coal", 32, policy=policy, device=device)
            rows.extend(row["events"])
        if _policy_digest(policy) != before_digest:
            raise RuntimeError("frozen policy parameters changed during probe")
        state_digest = hashlib.sha256(json.dumps([
            [row["layout_seed"], row["step"], row["rgb_sha256"]] for row in rows
        ]).encode()).hexdigest()
        policy_summaries[f"{run_record['arm']}/seed_{run_record['seed']}"] = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "policy_sha256_before_and_after": before_digest,
            "probe_states_sha256": state_digest,
            "spi_id": payload["spi_id"],
            "summary": _policy_summary(rows),
        }
        policy_rows.extend({"arm": run_record["arm"], "seed": int(run_record["seed"]), **row} for row in rows)
    result = {
        "formal_result": False,
        "diagnostic_only": True,
        "pilot_input": str(root),
        "pilot_git_commit": aggregate.get("git_commit"),
        "diagnostic_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "diagnostic_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "layout_count": int(layout_count),
        "horizons": list(horizons),
        "mechanical": _mechanical_summary(mechanical_rows),
        "mechanical_rows": mechanical_rows,
        "policy_probe": policy_summaries,
        "policy_rows": policy_rows,
        "interpretation": {
            "mechanical_horizon_8_all_success": all(row["success"] for row in mechanical_rows if row["horizon"] == 8),
            "policy_probe_is_read_only": True,
            "policy_parameters_unchanged": True,
            "all_policies_share_rgb_states": len({row["probe_states_sha256"] for row in policy_summaries.values()}) == 1,
            "probe_distribution": "native scripted shortest routes, not autonomous policy visits",
            "action_label": "all first actions on a shortest position-and-facing route including final do",
            "oracle_actions_executed_by_learned_policy": False,
            "spt_or_module_updated": False,
        },
    }
    if not result["interpretation"]["all_policies_share_rgb_states"]:
        raise RuntimeError("policies were probed on different RGB states")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layout-count", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run(args.input, args.output, layout_count=args.layout_count, device_name=args.device)
    print(json.dumps({"pilot_git_commit": result["pilot_git_commit"], "mechanical": result["mechanical"], "policy_probe": result["policy_probe"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
