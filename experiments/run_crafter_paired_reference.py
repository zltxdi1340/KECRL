"""Run real Crafter paired-world reference verification (non-formal)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from pathlib import Path

from src.counterfactual.crafter_reference import InterventionSpec
from src.counterfactual.crafter_reference_runner import run_paired_reference


def run(config_path: str, output_path: str) -> dict:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config.get("formal_result") is not False or config.get("oracle_reference_run") is not True:
        raise ValueError("paired reference runner requires formal_result=false and oracle_reference_run=true")
    output = Path(output_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    intervention = InterventionSpec(**config["intervention"])
    cases = [run_paired_reference(
        seed=int(seed), pair_id=f"pair:crafter:{seed}", target=config["target"],
        environment_scope=config["environment_scope"], intervention=intervention,
        max_steps=int(config["max_steps"]),
    ) for seed in config["seeds"]]
    source_files = [Path(config_path), Path("src/counterfactual/crafter_reference.py"),
                    Path("src/counterfactual/crafter_reference_runner.py"), Path(__file__)]
    source_hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files}
    result = {
        "status": config["status"], "formal_result": False, "oracle_reference_run": True,
        "config": config_path, "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short", "--branch"], text=True).strip(),
        "source_sha256": source_hashes,
        "runtime": {"python": platform.python_version(), "crafter": importlib.metadata.version("crafter")},
        "cases": [{key: (asdict(value) if hasattr(value, "__dataclass_fields__") else value) for key, value in case.items()} for case in cases],
        "note": "Real Crafter oracle paired-world verification; not policy training, qualification, or a formal method result.",
    }
    output.mkdir(parents=True)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crafter_paired_reference_v1.yaml")
    parser.add_argument("--output", default="results/crafter_paired_reference_v1")
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == "__main__":
    main()
