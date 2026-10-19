"""Reproduce the disclosed post-training audit correction, without formal promotion.

The original manifest verifier and E2 statistic remain unchanged. A detached,
hash-bound erratum identifies the sole permitted source change and the exact
inputs/full-RGB audits. Original manifests are also checked in a temporary
historical checkout using their original verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from experiments.analyze_crafter_gather_spt_transfer_e2 import _candidate_statistics
from experiments.crafter_formal_manifest import sha256_file


AUDITOR = "experiments/analyze_crafter_gather_spt_transfer_comparability.py"
ANALYSIS_SOURCES = {
    AUDITOR,
    "experiments/analyze_crafter_gather_spt_transfer_erratum.py",
    "experiments/analyze_crafter_gather_spt_transfer_e2.py",
    "experiments/crafter_formal_manifest.py",
}


def _load_bound_json(path: str | Path, expected_hash: str) -> dict:
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_hash:
        raise ValueError(f"erratum input hash mismatch: {path}")
    return json.loads(payload)


def _validate_audit(audit: dict, root: str, config_path: str) -> None:
    if audit.get("verification_mode") != "reconstruct_scene_and_initial_rgb":
        raise ValueError(f"full RGB reconstruction is required: {root}")
    if any(audit.get(key) is not True for key in (
        "valid", "provenance_valid", "strict_source_scene_pool_equal",
    )) or audit.get("errors") != []:
        raise ValueError(f"corrected comparability audit failed: {root}")
    if Path(audit.get("input", "")).resolve() != Path(root).resolve():
        raise ValueError(f"audit input directory mismatch: {root}")
    if audit.get("config") != config_path:
        raise ValueError(f"audit config mismatch: {root}")


def _verify_original_checkout(commit: str, config_paths: list[str]) -> list[dict]:
    subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"], check=True)
    with tempfile.TemporaryDirectory(prefix="kecrl-original-manifest-") as temporary:
        checkout = Path(temporary) / "checkout"
        subprocess.run(
            ["git", "worktree", "add", "--detach", "--quiet", str(checkout), commit],
            check=True,
        )
        try:
            code = (
                "import json,sys; from pathlib import Path; "
                "from experiments.crafter_formal_manifest import verify_formal_manifest; "
                "print(json.dumps([verify_formal_manifest(json.loads(Path(p).read_text()), p) "
                "for p in sys.argv[1:]]))"
            )
            output = subprocess.check_output(
                [sys.executable, "-c", code, *config_paths], cwd=checkout, text=True,
            )
            return json.loads(output)
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(checkout)], check=True,
            )


def analyze(erratum_path: str | Path) -> dict:
    erratum_path = Path(erratum_path)
    expected_hash = erratum_path.with_suffix(erratum_path.suffix + ".sha256").read_text().strip()
    erratum = _load_bound_json(erratum_path, expected_hash)
    if erratum.get("version") != "crafter_gather_post_training_audit_erratum_v1":
        raise ValueError("unsupported audit erratum version")
    if erratum.get("formal_result") is not False:
        raise ValueError("audit erratum cannot promote the original formal result flag")
    changes = erratum.get("source_changes", {})
    if set(changes) != {AUDITOR}:
        raise ValueError("only the comparability auditor may differ from the sealed sources")
    if set(erratum.get("analysis_source_sha256", {})) != ANALYSIS_SOURCES:
        raise ValueError("erratum must bind the complete analysis source set")
    for path, digest in erratum["analysis_source_sha256"].items():
        if sha256_file(path) != digest:
            raise ValueError(f"erratum analysis source hash mismatch: {path}")

    manifests = []
    aggregates = {}
    if [experiment["id"] for experiment in erratum["experiments"]] != ["E1", "E2"]:
        raise ValueError("erratum must identify E1 and E2 once each")
    for experiment in erratum["experiments"]:
        manifest = _load_bound_json(experiment["manifest"], experiment["manifest_sha256"])
        if manifest.get("source_sha256", {}).get(AUDITOR) != changes[AUDITOR]["original_sha256"]:
            raise ValueError("erratum does not match the sealed auditor hash")
        for path, digest in manifest["source_sha256"].items():
            expected = changes[AUDITOR]["corrected_sha256"] if path == AUDITOR else digest
            if sha256_file(path) != expected:
                raise ValueError(f"unapproved source change: {path}")
        aggregate = _load_bound_json(experiment["aggregate"], experiment["aggregate_sha256"])
        if (aggregate.get("formal_result"), aggregate.get("diagnostic_only")) != (True, False):
            raise ValueError("erratum requires the completed formal training inputs")
        if aggregate.get("formal_manifest", {}).get("manifest_sha256") != experiment["manifest_sha256"]:
            raise ValueError("training input does not reference the original sealed manifest")
        if aggregate.get("config") != manifest["config_path"]:
            raise ValueError("training input config differs from the original manifest")
        if any(aggregate["formal_manifest"].get(key) != manifest[key] for key in (
            "source_sha256", "config_sha256", "protocol_sha256",
        )):
            raise ValueError("training provenance snapshot differs from the original manifest")
        aggregates[experiment["id"]] = aggregate
        manifests.append(manifest)

    snapshots = _verify_original_checkout(
        erratum["original_checkout"], [manifest["config_path"] for manifest in manifests],
    )
    for experiment, snapshot in zip(erratum["experiments"], snapshots):
        if snapshot["manifest_sha256"] != experiment["manifest_sha256"]:
            raise ValueError("original checkout manifest differs from the erratum")

    e2 = aggregates["E2"]
    config = json.loads(Path(e2["config"]).read_text())
    folds = {}
    audited = []
    expected_audits = {("E1", "direct_wood_to_coal")} | {
        ("E2", fold["fold_id"]) for fold in e2["folds"]
    }
    audit_keys = [(entry["experiment"], entry["fold_id"]) for entry in erratum["audits"]]
    if len(audit_keys) != len(expected_audits) or set(audit_keys) != expected_audits:
        raise ValueError("erratum must include exactly the E1 audit and all E2 fold audits")
    for entry in erratum["audits"]:
        audit = _load_bound_json(entry["path"], entry["sha256"])
        aggregate = _load_bound_json(entry["aggregate"], entry["aggregate_sha256"])
        root = str(Path(entry["aggregate"]).parent)
        _validate_audit(audit, root, aggregate["config"])
        if audit.get("aggregate_git_commit") != aggregate.get("git_commit"):
            raise ValueError(f"audit training commit mismatch: {root}")
        expected_pairs = {(arm, int(seed)) for arm in config["arms"] for seed in config["replica_seeds"]}
        audited_pairs = {(run["arm"], int(run["seed"])) for run in audit["run_summaries"]}
        if len(audit["run_summaries"]) != len(expected_pairs) or audited_pairs != expected_pairs:
            raise ValueError(f"audit arm/seed coverage mismatch: {root}")
        if entry["experiment"] == "E2":
            fold_id = entry["fold_id"]
            declaration = next((fold for fold in e2["folds"] if fold["fold_id"] == fold_id), None)
            if declaration is None or declaration["aggregate"] != entry["aggregate"] or fold_id in folds:
                raise ValueError(f"erratum fold mapping mismatch: {fold_id}")
            if aggregate.get("formal_manifest", {}).get("manifest_sha256") != e2["formal_manifest"]["manifest_sha256"]:
                raise ValueError(f"fold training manifest mismatch: {fold_id}")
            if (aggregate.get("formal_result"), aggregate.get("diagnostic_only")) != (True, False):
                raise ValueError(f"fold formal training mode mismatch: {fold_id}")
            if (aggregate.get("source_materials"), aggregate.get("target_material")) != (
                declaration["source_materials"], declaration["target_material"],
            ):
                raise ValueError(f"fold binding differs from root aggregate: {fold_id}")
            folds[fold_id] = aggregate
        elif entry["aggregate"] != erratum["experiments"][0]["aggregate"]:
            raise ValueError("E1 audit does not bind the E1 training input")
        audited.append({"path": entry["path"], "sha256": entry["sha256"], "valid": True})

    protocol_path = config["formal_protocol"]
    protocol = json.loads(Path(protocol_path).read_text())
    protocol.update({"_path": protocol_path, "_sha256": sha256_file(protocol_path)})
    errors: list[str] = []
    statistics = _candidate_statistics(config, e2, folds, protocol, errors, formal_result=False)
    if errors or statistics.get("status") != "candidate_estimate_diagnostic_only":
        raise ValueError(f"frozen statistical input checks failed: {errors or statistics.get('reasons')}")
    return {
        "status": "post_training_audit_correction",
        "formal_result": False,
        "diagnostic_only": True,
        "erratum": {"path": str(erratum_path), "sha256": expected_hash},
        "original_manifest_verification": snapshots,
        "corrected_full_rgb_audits": audited,
        "statistics": statistics,
        "interpretation": "unchanged prespecified statistic after a disclosed post-training audit correction; original formal flag remains false; no Module qualification",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--erratum", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.erratum)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"preserve previous analysis output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": result["status"], "formal_result": result["formal_result"],
                      "primary_effect": result["statistics"]["primary_effect"], "output": str(output)}))


if __name__ == "__main__":
    main()
