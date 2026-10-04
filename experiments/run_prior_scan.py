"""Run independent controlled prior-strength diagnostics without overwriting results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.train_controlled import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/controlled_stage_prior_scan.yaml")
    parser.add_argument("--strengths", nargs="+", type=float, default=[0.0, 0.5, 1.5, 3.0])
    parser.add_argument("--root", default="results/prior_scan")
    args = parser.parse_args()
    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    index = {"formal_result": False, "strengths": args.strengths, "runs": []}
    for strength in args.strengths:
        config = dict(base)
        config["knowledge"] = dict(base["knowledge"])
        config["knowledge"]["action_prior_strength"] = strength
        config["results_root"] = str(Path(args.root) / f"strength_{strength:g}")
        temporary = Path(args.root) / f"config_strength_{strength:g}.json"
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        result = run(str(temporary))
        index["runs"].append({"strength": strength, "results_root": config["results_root"], "run_count": len(result)})
    output = Path(args.root)
    output.mkdir(parents=True, exist_ok=True)
    (output / "scan_index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
