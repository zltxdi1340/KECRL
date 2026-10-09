"""Reload actor-only artifacts and audit the saved source/evaluation boundary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_actor_only_local import evaluate_actor, install_actor_copy
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


def verify(root, output):
    summary, config = _read(root / "summary.json"), _read(root / "config.json")
    checks, actor_stats, artifact_hashes = {}, [], {}
    with np.load(root / "scene_dataset.npz") as arrays:
        data = dict(arrays)
    heldout = data["split_code"] == 2
    for fit in summary["actor_fits"]:
        seed = fit["policy_seed"]
        artifact_path = root / f"seed{seed}_supervised_actor.pt"
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
        checkpoint_path = Path(artifact["source_checkpoint"])
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        actor_digest = array_digest({key: value.numpy() for key, value in artifact["actor_state"].items()})
        checks[f"seed{seed}_artifact_digest_and_fit_match"] = actor_digest == fit["actor_state_canonical_digest"] and artifact["fit"] == fit
        checks[f"seed{seed}_source_checkpoint_sha_match"] = _sha256(checkpoint_path) == artifact["source_checkpoint_sha256"]
        checks[f"seed{seed}_teacher_scope_and_no_ppo_resume_explicit"] = artifact["teacher_used"] is True and artifact["teacher_scope"] == config["teacher_scope"] and artifact["resumable_ppo_checkpoint"] is False
        source = build_matched_spatial_policy(config["policy"], "cnn_only")
        source.load_state_dict(checkpoint["policy"], strict=True)
        source.optimizer.load_state_dict(checkpoint["optimizer"])
        head = torch.nn.Linear(128, 17)
        head.load_state_dict(artifact["actor_state"], strict=True)
        fitted = install_actor_copy(source, head)
        checks[f"seed{seed}_masked_out_actor_rows_unchanged"] = all(torch.equal(artifact["actor_state"][key][7:], source.actor.state_dict()[key][7:]) for key in ("weight", "bias"))
        with np.load(root / f"seed{seed}_features.npz") as arrays:
            features = arrays["embedding"]
        # Saved CUDA and CPU GEMMs can differ in the last bits. Compare metrics
        # within numerical tolerance; the independent CUDA process audit is exact.
        for variant, policy in (("original", source), ("teacher_actor_copy", fitted)):
            metrics = evaluate_actor(policy, features[heldout], data["labels"][heldout], data["background_id"][heldout], "cpu")
            saved = next(row["heldout_action_metrics"] for row in summary["evaluation"] if row["policy_seed"] == seed and row["variant"] == variant)
            checks[f"seed{seed}_{variant}_reloaded_head_predictions_match"] = metrics["confusion_matrix"] == saved["confusion_matrix"]
            checks[f"seed{seed}_{variant}_reloaded_head_probabilities_match"] = all(np.isclose(metrics[key], saved[key], rtol=1e-5, atol=1e-6)
                                                                                 for key in ("cross_entropy", "mean_entropy", "mean_correct_action_probability"))
        weight, original_weight = artifact["actor_state"]["weight"][:7], source.actor.weight.detach()[:7]
        actor_stats.append({"policy_seed": seed, "original_legal_actor_weight_l2": float(original_weight.norm()),
                            "supervised_legal_actor_weight_l2": float(weight.norm()),
                            "legal_actor_weight_change_l2": float((weight - original_weight).norm()),
                            "legal_actor_bias_change_l2": float((artifact["actor_state"]["bias"][:7] - source.actor.bias.detach()[:7]).norm())})
        artifact_hashes[str(artifact_path)] = _sha256(artifact_path)
    provenance = _read(root / "provenance.json")
    files = {path: _sha256(Path(path)) == digest for field in ("source_sha256", "installed_crafter_sha256", "checkpoint_and_config_sha256")
             for path, digest in provenance[field].items()}
    checks["source_package_checkpoint_config_and_previous_results_still_unchanged"] = all(files.values())
    result = {"checks": checks, "passed": all(checks.values()), "source_files": files,
              "actor_statistics": actor_stats, "actor_artifact_sha256": artifact_hashes}
    _write(output, result)
    if not result["passed"]:
        raise RuntimeError(f"artifact audit failed: {checks}")
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(Path(args.input), Path(args.output))


if __name__ == "__main__":
    main()
