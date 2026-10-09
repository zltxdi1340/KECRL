"""Frozen evaluation of all saved actor-head variants on unseen backgrounds."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_actor_head_ablation import ActorMLP, install_head_copy
from experiments.crafter_actor_only_local import evaluate_actor
from experiments.crafter_spatial_readout_data import BackgroundCollectionEnv, array_digest, scene_manifest
from experiments.run_crafter_spatial_training_determinism_audit import _json_digest, _state_equal
from experiments.run_crafter_wood3_actor_only_local import _episode, _generate_dataset, _rollout_stats
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_readout import _extract
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy
from src.environments.crafter_adapter import CrafterEnvironmentAdapter
from src.environments.crafter_determinism import stable_crafter_object_order


def load_saved_head(source, artifact):
    variant = artifact["variant"]
    if variant == "mlp64":
        # Loading a stored head must not consume the sampling RNG.
        with torch.random.fork_rng(devices=[]):
            head = ActorMLP()
        head = head.to(next(source.parameters()).device)
    elif variant in ("raw_linear", "standardized_linear"):
        head = copy.deepcopy(source.actor)
    else:
        raise ValueError("unrecognized saved head architecture")
    head.load_state_dict(artifact["head_state"], strict=True)
    digest = array_digest({key: value.detach().cpu().numpy() for key, value in head.state_dict().items()})
    if digest != artifact["head_state_canonical_digest"]:
        raise RuntimeError("saved head canonical digest changed")
    if not all(torch.isfinite(p).all() for p in head.parameters()):
        raise RuntimeError("non-finite saved head")
    return install_head_copy(source, head)


def fresh_rollouts(policy, records, config, path, device, seed_index):
    scenes = scene_manifest()
    native = BackgroundCollectionEnv(scenes[0], seed=0, length=config["environment_length"])
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    rows, current, observer, observer_steps = [], None, None, 0
    try:
        with path.open("w") as stream:
            for record in records:
                seed = record["environment_seed"]
                if seed != current:
                    native.set_background(seed)
                    current = seed
                action_seed = config["diagnostic_action_seed_base"] + seed_index * config["replicate_seed_stride"] + record["row"]
                scene = scenes[record["scene_index"]]
                row = _episode(policy, adapter, scene, seed, action_seed, "greedy", config, device,
                               expected_rgb=record["rgb_sha256"])
                if observer is None:
                    control = _episode(policy, adapter, scene, seed, action_seed, "greedy", config, device,
                                       expected_rgb=record["rgb_sha256"], observe=False)
                    observer_steps = control["steps"]
                    observer = {"trajectory_with_and_without_oracle_identical": _state_equal(row, control)}
                    if not observer["trajectory_with_and_without_oracle_identical"]:
                        raise RuntimeError("fresh oracle observer changed a trajectory")
                row.update({"background_id": record["background_id"], "scene_index": record["scene_index"], "repetition": 0})
                rows.append(row)
                stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    finally:
        adapter.close()
    metrics = {"tree_present": _rollout_stats([row for row in rows if row["tree_present"]]),
               **{condition: _rollout_stats([row for row in rows if row["condition"] == condition])
                  for condition in ("aligned", "needs_turn", "no_tree")},
               "by_background": {str(group): _rollout_stats([row for row in rows if row["tree_present"] and row["background_id"] == group])
                                 for group in sorted({row["background_id"] for row in rows})}}
    return {"metrics": metrics, "episodes": len(rows), "interaction_steps": sum(row["steps"] for row in rows),
            "canonical_digest": _json_digest(rows), "jsonl_sha256": _sha256(path),
            "observer_audit": observer, "observer_control_interaction_steps": observer_steps}


def _validate(config):
    required = {"formal_result": False, "evaluation_only": True, "policy_updates_allowed": False,
                "formal_training_allowed": False, "new_fit_or_selection_allowed": False,
                "device": "cuda", "seed_set": [0, 1, 2],
                "variants": ["baseline", "raw_linear", "standardized_linear", "mlp64"],
                "background_counts": {"train": 0, "validation": 0, "heldout": 16},
                "background_seed_base": 51000000, "background_candidate_limit": 2048,
                "max_steps": 8, "evaluation_modes": ["greedy"]}
    for key, value in required.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"frozen fresh-evaluation protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("configured PYTHONHASHSEED required")


def run(config_path, output_path, repeat=False):
    config = _read(Path(config_path))
    _validate(config)
    root = Path(config["head_result_root"])
    head_config, head_summary = _read(root / "config.json"), _read(root / "summary.json")
    source_root = Path(head_config["baseline_result_root"])
    prior_root = Path(head_config["source_actor_diagnostic_root"])
    if head_summary["cross_process_repetition"]["passed"] is not True:
        raise RuntimeError("paired head fits have not passed repetition")
    seeds = [0] if repeat else config["seed_set"]
    paths = [Path(config_path), root / "config.json", root / "summary.json", prior_root / "provenance.json",
             prior_root / "scene_dataset.npz", prior_root / "scene_records.jsonl", prior_root / "summary.json"]
    paths += [source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt" for seed in seeds]
    paths += [root / f"seed{seed}_{variant}.pt" for seed in seeds for variant in config["variants"] if variant != "baseline"]
    excluded = set()
    for excluded_root in config["exclude_background_roots"]:
        path = Path(excluded_root) / "scene_records.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        descriptor = _read(Path(excluded_root) / "summary.json")["dataset"]
        if _json_digest(records) != descriptor["records_canonical_digest"]:
            raise RuntimeError("excluded source manifest changed")
        excluded.update(row["background_rgb_sha256"] for row in records)
        paths.append(path)
    sources = ["experiments/run_crafter_wood3_actor_head_fresh.py", "experiments/crafter_actor_head_ablation.py",
               "experiments/run_crafter_wood3_actor_head_ablation.py", "experiments/crafter_actor_only_local.py",
               "experiments/run_crafter_wood3_actor_only_local.py", "experiments/crafter_spatial_readout_data.py",
               "experiments/crafter_controlled_collection_scene.py", "src/algorithms/spatial_crafter_policy.py",
               "src/environments/crafter_adapter.py", "src/environments/crafter_determinism.py",
               "src/environments/crafter_collection_diagnostics.py"]
    prior_provenance = _read(prior_root / "provenance.json")
    prior_unchanged = {path: _sha256(Path(path)) == digest
                       for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
                       for path, digest in prior_provenance[field].items()}
    if not all(prior_unchanged.values()):
        raise RuntimeError("original sources/checkpoints/package changed since actor-only run")
    paths += [Path(path) for path in sources] + [Path(path) for path in prior_unchanged]
    hashes = {str(path): _sha256(path) for path in paths}
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "config.json", config)
    _write(output / "provenance.json", {"sha256_before_fresh_evaluation": hashes,
           "prior_actor_only_files_unchanged": prior_unchanged,
           "paired_fit_new_sources_sealed_after_fit_completion": True,
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "python_executable": sys.executable})
    (output / "source_snapshot").mkdir()
    for path in sources:
        shutil.copy2(path, output / "source_snapshot" / Path(path).name)
    device = torch.device(config["device"])
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    settings = _configure_torch_determinism(config)
    evaluation, audits, features = [], [], {}
    with stable_crafter_object_order() as order_version:
        data, records, dataset = _generate_dataset(config, output, excluded)
        if not np.all(data["split_code"] == 2):
            raise RuntimeError("fresh set must be heldout-only")
        for seed in seeds:
            checkpoint_path = source_root / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if checkpoint["environment_order_version"] != order_version or checkpoint["actual_interaction_steps"] != 100000:
                raise RuntimeError("checkpoint budget or wrapper differs")
            source = build_matched_spatial_policy(head_config["policy"], "cnn_only").to(device)
            source.load_state_dict(checkpoint["policy"], strict=True)
            source.optimizer.load_state_dict(checkpoint["optimizer"])
            source.eval()
            for parameter in source.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
            arrays = _extract(source, data["images"], config, device)
            np.savez_compressed(output / f"seed{seed}_features.npz", **arrays)
            features[str(seed)] = {"canonical_digest": array_digest(arrays), "npz_sha256": _sha256(output / f"seed{seed}_features.npz")}
            for variant in config["variants"]:
                if variant == "baseline":
                    policy = source
                else:
                    artifact = torch.load(root / f"seed{seed}_{variant}.pt", map_location=device, weights_only=False)
                    if artifact["source_checkpoint_sha256"] != _sha256(checkpoint_path) or artifact["selection_uses_heldout"] is not False:
                        raise RuntimeError("saved head source or selection boundary differs")
                    policy = load_saved_head(source, artifact)
                snapshot = copy.deepcopy((policy.state_dict(), policy.optimizer.state_dict()))
                actions = evaluate_actor(policy, arrays["embedding"], data["labels"], data["background_id"], device)
                rollout = fresh_rollouts(policy, records, config, output / f"seed{seed}_{variant}_rollouts.jsonl", device, config["seed_set"].index(seed))
                audit = {"seed": seed, "variant": variant,
                         "evaluation_parameters_and_optimizer_unchanged": _state_equal(snapshot, (policy.state_dict(), policy.optimizer.state_dict())),
                         "all_gradients_absent": all(p.grad is None for p in policy.parameters()),
                         "encoder_critic_and_original_optimizer_unchanged":
                         all(torch.equal(v, source.state_dict()[k]) for k, v in policy.state_dict().items() if not k.startswith("actor."))
                         and _state_equal(policy.optimizer.state_dict(), source.optimizer.state_dict()),
                         "real_cuda": next(policy.parameters()).is_cuda}
                if not all(v for k, v in audit.items() if k not in ("seed", "variant")):
                    raise RuntimeError("fresh frozen evaluation boundary failed")
                audits.append(audit)
                evaluation.append({"policy_seed": seed, "variant": variant, "teacher_used_for_fit": variant != "baseline",
                                   "heldout_action_metrics": actions, "rollouts": rollout})
                print(f"Fresh seed{seed} {variant}: do={actions['ready_do_rate']:.3f}, turn={actions['unaligned_target_move_rate']:.3f}, greedy collection={rollout['metrics']['tree_present']['designated_success_rate']:.3f}", flush=True)
            if not _state_equal(before, (source.state_dict(), source.optimizer.state_dict())):
                raise RuntimeError("source changed during fresh evaluation")
    repetition = None
    if not repeat:
        with (output / "repeat.log").open("w") as stream:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_actor_head_fresh",
                            "--config", config_path, "--output", str(output / "cross_process_repeat"), "--repeat-seed0"],
                           check=True, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
        repeated = _read(output / "cross_process_repeat/summary.json")
        repetition = {"dataset_identical": dataset == repeated["dataset"],
                      "features_identical": features["0"] == repeated["features"]["0"],
                      "all_seed0_head_metrics_and_trajectories_identical": [row for row in evaluation if row["policy_seed"] == 0] == repeated["evaluation"]}
        repetition["passed"] = all(repetition.values())
        if not repetition["passed"]:
            raise RuntimeError("fresh cross-process evaluation diverged")
    unchanged = {path: _sha256(Path(path)) == digest for path, digest in hashes.items()}
    if not all(unchanged.values()):
        raise RuntimeError("input artifact or source changed during fresh evaluation")
    result = {"formal_result": False, "evaluation_only": True, "new_fit_or_selection_performed": False,
              "policy_input": "RGB64x64x3 only", "evaluation_modes": ["greedy"],
              "teacher_scope": config["teacher_scope"], "dataset": dataset, "features": features,
              "evaluation": evaluation, "frozen_boundary_audits": audits, "files_unchanged": unchanged,
              "cross_process_repetition": repetition, "environment_order_version": order_version,
              "training_determinism": settings, "head_result_root": str(root),
              "supervised_optimizer_updates": 0, "ppo_training_interaction_steps": 0,
              "diagnostic_rollout_episodes": sum(row["rollouts"]["episodes"] for row in evaluation),
              "diagnostic_rollout_interaction_steps": sum(row["rollouts"]["interaction_steps"] for row in evaluation),
              "observer_control_interaction_steps": sum(row["rollouts"]["observer_control_interaction_steps"] for row in evaluation),
              "fixture_validation_interaction_steps": dataset["fixture_validation_interaction_steps"],
              "module_registered": False, "knowledge_updated": False, "spt_updated": False}
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    run(args.config, args.output, args.repeat_seed0)


if __name__ == "__main__":
    main()
