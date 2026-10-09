"""Compare actor-head optimization and capacity on a frozen, audited dataset."""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_actor_head_ablation import fit_head_variant, install_head_copy
from experiments.crafter_actor_only_local import evaluate_actor
from experiments.run_crafter_wood3_actor_only_local import _episode, _json_digest, _rollout_stats
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from experiments.run_crafter_wood3_spatial_representation_curve import _configure_torch_determinism
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


VARIANTS = ("raw_linear", "standardized_linear", "mlp64")


def _validate(config, prior, baseline):
    expected = {
        "formal_result": False, "teacher_used": True,
        "teacher_scope": "supervised_actor_head_copies_on_constructed_local_fixtures_only",
        "source_policy_updates_allowed": False, "diagnostic_actor_updates_allowed": True,
        "encoder_updates_allowed": False, "formal_training_allowed": False,
        "device": "cuda", "seed_set": [0, 1, 2], "checkpoint_steps": 100000,
        "dataset_digest_source": "previous_actor_only_local_diagnostic",
        "dataset_manifest_is_frozen": True, "background_counts": {"train": 24, "validation": 8, "heldout": 16},
        "scenes_per_background": 60, "action_allowlist": [0, 1, 2, 3, 4, 5, 6],
        "feature_preprocessing_candidates": ["none", "train_mean_std_folded_to_raw_head"],
        "actor_head_variants": list(VARIANTS), "actor_learning_rates": [.003, .03],
        "actor_weight_decay": 0.0, "actor_fit_epochs": 3000, "validation_interval": 100,
        "feature_std_floor": .0001, "mlp_init_seed": 47000001,
        "selection_metric": "validation_balanced_accuracy_on_six_fixture_action_targets",
        "class_weights": "inverse_frequency_from_training_only", "max_steps": 8,
        "sample_repetitions_per_scene": 5, "greedy_repetitions_per_scene": 1,
        "observer_audit_episodes_per_policy": 1,
    }
    for key, value in expected.items():
        if config.get(key) != value or (isinstance(value, bool) and config.get(key) is not value):
            raise ValueError(f"fixed head-ablation protocol differs: {key}")
    if os.environ.get("PYTHONHASHSEED") != str(config["python_hash_seed"]):
        raise ValueError("configured PYTHONHASHSEED is required")
    for key in ("policy", "seed_set", "action_allowlist", "environment_length", "python_hash_seed", "cublas_workspace_config", "replicate_seed_stride"):
        if config.get(key, baseline.get(key)) != baseline[key]:
            raise ValueError(f"baseline boundary differs: {key}")
    if baseline.get("auxiliary_target") or baseline.get("terminal_death_penalty", 0):
        raise ValueError("head ablation requires unmodified baseline checkpoints")
    if prior["dataset"]["canonical_digest"] != _dataset_digest(prior):
        raise ValueError("prior actor-only dataset digest is not self-consistent")
    if prior["dataset"]["backgrounds"] != _read(Path(config["source_actor_diagnostic_root"]) / "dataset_summary.json")["backgrounds"]:
        raise ValueError("prior background manifest changed")


def _dataset_digest(summary):
    import hashlib
    with np.load(Path(summary["_root"]) / "scene_dataset.npz") as values:
        arrays = dict(values)
    digest = hashlib.sha256()
    for key, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(json.dumps([key, value.dtype.str, list(value.shape)], separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _rollouts(policy, records, config, output, device, seed_index):
    from experiments.crafter_controlled_collection_scene import CollectionScene
    from experiments.crafter_spatial_readout_data import BackgroundCollectionEnv, scene_manifest
    from src.environments.crafter_adapter import CrafterEnvironmentAdapter
    from src.environments.crafter_determinism import stable_crafter_object_order
    scenes = scene_manifest()
    native = BackgroundCollectionEnv(scenes[0], seed=0, length=10000)
    adapter = CrafterEnvironmentAdapter(environment=native, diagnostics=True)
    rows, observer, observer_steps, current = [], None, 0, None
    try:
        with stable_crafter_object_order():
            with output.open("w") as stream:
                for record in records:
                    if record["split"] != "heldout":
                        continue
                    seed = record["environment_seed"]
                    if current != seed:
                        native.set_background(seed)
                        current = seed
                    scene = scenes[record["scene_index"]]
                    for rep in range(6):
                        mode = "sample" if rep < 5 else "greedy"
                        action_seed = config["diagnostic_action_seed_base"] + seed_index * config["replicate_seed_stride"] + record["row"] * 6 + rep
                        row = _episode(policy, adapter, scene, seed, action_seed, mode, config, device,
                                       expected_rgb=record["rgb_sha256"])
                        row.update({"background_id": record["background_id"], "scene_index": record["scene_index"], "repetition": rep})
                        if observer is None:
                            control = _episode(policy, adapter, scene, seed, action_seed, mode, config, device,
                                               expected_rgb=record["rgb_sha256"], observe=False)
                            observer_steps = control["steps"]
                            observer = {"trajectory_with_and_without_oracle_identical":
                                        _json_digest({key: value for key, value in row.items() if key not in ("background_id", "scene_index", "repetition")})
                                        == _json_digest(control)}
                            if not observer["trajectory_with_and_without_oracle_identical"]:
                                raise RuntimeError("diagnostic observer changed a trajectory")
                        rows.append(row)
                        stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
    finally:
        adapter.close()
    metrics = {}
    for mode in ("sample", "greedy"):
        subset = [row for row in rows if row["mode"] == mode]
        metrics[mode] = {"tree_present": _rollout_stats([row for row in subset if row["tree_present"]]),
                         **{condition: _rollout_stats([row for row in subset if row["condition"] == condition])
                            for condition in ("aligned", "needs_turn", "no_tree")}}
    return {"episodes": len(rows), "interaction_steps": sum(row["steps"] for row in rows),
            "canonical_digest": _json_digest(rows), "jsonl_sha256": _sha256(output),
            "metrics": metrics, "observer_audit": observer, "observer_control_interaction_steps": observer_steps}


def run(config_path, output_path, repeat=False):
    config = _read(Path(config_path))
    prior_root = Path(config["source_actor_diagnostic_root"])
    prior_summary = _read(prior_root / "summary.json")
    prior_summary["_root"] = str(prior_root)
    baseline = _read(Path(config["baseline_result_root"]) / "config.json")
    _validate(config, prior_summary, baseline)
    with np.load(prior_root / "scene_dataset.npz") as values:
        data = dict(values)
    records = [json.loads(line) for line in (prior_root / "scene_records.jsonl").read_text().splitlines()]
    if _json_digest(records) != prior_summary["dataset"]["records_canonical_digest"]:
        raise RuntimeError("prior records digest changed")
    prior_config = _read(prior_root / "config.json")
    device = torch.device(config["device"])
    if not torch.cuda.is_available():
        raise RuntimeError("real CUDA required")
    settings = _configure_torch_determinism(config)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / "config.json", config)
    seeds = [0] if repeat else config["seed_set"]
    fits, evaluation, frozen_audits = [], [], []
    targets = __import__("experiments.crafter_actor_only_local", fromlist=["fixture_action_targets"]).fixture_action_targets(data["labels"])
    masks = {name: data["split_code"] == index for index, name in enumerate(("train", "validation", "heldout"))}
    for seed in seeds:
        checkpoint_path = Path(config["baseline_result_root"]) / "cnn_only" / f"seed_{seed}" / "checkpoints" / "interaction_0100000.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        source = build_matched_spatial_policy(config["policy"], "cnn_only").to(device)
        source.load_state_dict(checkpoint["policy"], strict=True)
        source.optimizer.load_state_dict(checkpoint["optimizer"])
        source.eval()
        for parameter in source.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        arrays = np.load(prior_root / f"seed{seed}_features.npz")
        features = arrays["embedding"]
        arrays.close()
        seed_fits = []
        # Evaluate the unchanged source once per seed. Every fitted head uses
        # the same heldout rows and action RNG manifest below.
        original_snapshot = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        original_metrics = evaluate_actor(source, features[masks["heldout"]], data["labels"][masks["heldout"]],
                                          data["background_id"][masks["heldout"]], device)
        original_rollout_file = output / f"seed{seed}_baseline_original_rollouts.jsonl"
        original_rollouts = _rollouts(source, records, config, original_rollout_file, device, config["seed_set"].index(seed))
        evaluation.append({"policy_seed": seed, "variant": "baseline", "evaluation_variant": "original",
                           "teacher_used": False, "heldout_action_metrics": original_metrics,
                           "rollouts": original_rollouts})
        original_audit = {"policy_seed": seed, "variant": "baseline", "evaluation_variant": "original",
                          "policy_and_optimizer_unchanged": _state_equal(original_snapshot, (source.state_dict(), source.optimizer.state_dict())),
                          "all_policy_gradients_absent": all(parameter.grad is None for parameter in source.parameters()),
                          "cuda_tensor_verified": next(source.parameters()).is_cuda,
                          "encoder_critic_and_ppo_optimizer_match_source": True}
        if not all(value for key, value in original_audit.items() if key not in ("policy_seed", "variant", "evaluation_variant")):
            raise RuntimeError("baseline evaluation boundary failed")
        frozen_audits.append(original_audit)
        for variant in VARIANTS:
            print(f"Seed {seed}: fitting {variant}", flush=True)
            head, artifact = fit_head_variant(source, features[masks["train"]], targets[masks["train"]],
                                              features[masks["validation"]], targets[masks["validation"]], config, device, variant)
            state = artifact.pop("head_state")
            artifact["policy_seed"] = seed
            artifact["source_checkpoint"] = str(checkpoint_path)
            artifact["source_checkpoint_sha256"] = _sha256(checkpoint_path)
            artifact["teacher_used"] = True
            artifact["teacher_scope"] = config["teacher_scope"]
            artifact["formal_result"] = False
            artifact["head_state"] = state
            torch.save(artifact, output / f"seed{seed}_{variant}.pt")
            loaded = torch.load(output / f"seed{seed}_{variant}.pt", map_location=device, weights_only=False)
            head.load_state_dict(loaded["head_state"], strict=True)
            fitted = install_head_copy(source, head)
            fit_record = {key: value for key, value in loaded.items() if key not in ("head_state",)}
            seed_fits.append(fit_record)
            snapshot = copy.deepcopy((fitted.state_dict(), fitted.optimizer.state_dict()))
            heldout_metrics = evaluate_actor(fitted, features[masks["heldout"]], data["labels"][masks["heldout"]],
                                              data["background_id"][masks["heldout"]], device)
            rollout_file = output / f"seed{seed}_{variant}_rollouts.jsonl"
            rollouts = _rollouts(fitted, records, config, rollout_file, device, config["seed_set"].index(seed))
            evaluation.append({"policy_seed": seed, "variant": variant, "evaluation_variant": variant,
                               "teacher_used": True, "heldout_action_metrics": heldout_metrics,
                               "rollouts": rollouts})
            audit = {"policy_seed": seed, "variant": variant, "evaluation_variant": variant,
                     "policy_and_optimizer_unchanged": _state_equal(snapshot, (fitted.state_dict(), fitted.optimizer.state_dict())),
                     "all_policy_gradients_absent": all(parameter.grad is None for parameter in fitted.parameters()),
                     "cuda_tensor_verified": next(fitted.parameters()).is_cuda,
                     "encoder_critic_and_ppo_optimizer_match_source":
                     all(torch.equal(value, source.state_dict()[key]) for key, value in fitted.state_dict().items() if not key.startswith("actor."))
                     and _state_equal(fitted.optimizer.state_dict(), source.optimizer.state_dict())}
            if not all(value for key, value in audit.items() if key not in ("policy_seed", "variant", "evaluation_variant")):
                raise RuntimeError("head ablation evaluation boundary failed")
            frozen_audits.append(audit)
            del fitted, head
        fits.extend(seed_fits)
        if not _state_equal(before, (source.state_dict(), source.optimizer.state_dict())):
            raise RuntimeError("source policy changed during head ablation")
        del source, checkpoint, before
    repetition = None
    if not repeat:
        with (output / "repeat.log").open("w") as log:
            subprocess.run([sys.executable, "-u", "-m", "experiments.run_crafter_wood3_actor_head_ablation", "--config", config_path,
                            "--output", str(output / "cross_process_repeat"), "--repeat-seed0"], check=True,
                           env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT)
        repeated = _read(output / "cross_process_repeat/summary.json")
        main_seed0 = [row for row in evaluation if row["policy_seed"] == 0]
        repetition = {"prior_dataset_digest_identical": repeated["dataset"]["canonical_digest"] == prior_summary["dataset"]["canonical_digest"],
                      "fit_selection_and_history_identical": fits[:3] == repeated["fits"],
                      "evaluation_metrics_and_rollouts_identical": main_seed0 == repeated["evaluation"]}
        repetition["passed"] = all(repetition.values())
        if not repetition["passed"]:
            raise RuntimeError("head ablation cross-process repetition diverged")
    result = {"formal_result": False, "status": config["status"], "teacher_used": True, "teacher_scope": config["teacher_scope"],
              "source_policy_updated": False, "encoder_updated": False, "ppo_training_interaction_steps": 0,
              "diagnostic_head_copies_updated": True, "supervised_optimizer_updates": sum(fit["supervised_optimizer_updates"] for fit in fits),
              "diagnostic_rollout_interaction_steps": sum(row["rollouts"]["interaction_steps"] for row in evaluation),
              "diagnostic_rollout_episodes": sum(row["rollouts"]["episodes"] for row in evaluation),
              "selection_uses_heldout": False, "dataset": {"canonical_digest": prior_summary["dataset"]["canonical_digest"],
              "records_canonical_digest": prior_summary["dataset"]["records_canonical_digest"], "rows": int(len(data["images"])),
              "backgrounds": prior_summary["dataset"]["backgrounds"]}, "fits": fits, "evaluation": evaluation,
              "frozen_boundary_audits": frozen_audits, "cross_process_repetition": repetition,
              "training_determinism": settings, "prior_actor_only_root": str(prior_root),
              "module_registered": False, "knowledge_updated": False, "spt_updated": False, "formal_training_allowed": False}
    _write(output / "summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeat-seed0", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.output, args.repeat_seed0)
    print(json.dumps({"fits": len(result["fits"]), "rollout_steps": result["diagnostic_rollout_interaction_steps"],
                      "cross_process_repetition": result["cross_process_repetition"]}, indent=2))


if __name__ == "__main__":
    main()
