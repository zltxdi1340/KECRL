"""Scan evidence budgets needed for the configured Knowledge thresholds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.knowledge.evolution import KnowledgeEvolution


def first_decision(observation, max_budget=1000):
    config = {"n_min": 10, "tau_confirm": 0.8, "tau_reject": 0.2, "confidence": 0.95, "budget": max_budget}
    for budget in range(1, max_budget + 1):
        evolution = KnowledgeEvolution(config)
        result = None
        for index in range(budget):
            result = evolution.record("claim", observation, {"evidence_id": f"{observation}:{index}"})
            if result["status"] in {"confirmed", "rejected"}:
                return {"observation": observation, "status": result["status"], "effective_n": result["n"], "budget_used": index + 1, "lower": result["lower"], "upper": result["upper"]}
    return {"observation": observation, "status": result["status"], "effective_n": result["n"], "budget_used": max_budget, "lower": result["lower"], "upper": result["upper"]}


def scan(max_budget=1000):
    unknown = first_decision("bottom", max_budget)
    return {"formal_result": False, "config": {"n_min": 10, "tau_confirm": 0.8, "tau_reject": 0.2, "confidence": 0.95}, "support": first_decision(1, max_budget), "counterevidence": first_decision(0, max_budget), "unknown": unknown}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("--max-budget", type=int, default=1000); parser.add_argument("--output", default="results/knowledge_budget_scan.json")
    args = parser.parse_args(); result = scan(args.max_budget); Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8"); print(json.dumps(result, indent=2))
