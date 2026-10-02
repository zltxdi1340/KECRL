"""Run the small reproducible CPU/GPU-compatible interface smoke experiment."""
from __future__ import annotations
import argparse, csv, json, random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from src.knowledge.contracts import Mechanism, InMemoryKnowledgeBank
from src.knowledge.evolution import KnowledgeEvolution
from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.models import SPT, SPI, InMemoryQualifiedSkillLibrary, QualificationConfig
from src.continual_learning.contracts import InMemoryContinualLearningPipeline, TaskVersionView
from src.environments.simulated import ControlledEnvironment
from src.utils.config import load_config, runtime_metadata

def run(config_path):
    config = load_config(config_path); random.seed(config.get("seed", 0))
    resolved_device = InMemoryContinualLearningPipeline.resolve_device(config.get("device", "auto"))
    target = {"name": "value_at_least", "value": 1}; scope = {"environment": "controlled"}
    contract = ImplementationContract(({"name": "value_at_least", "value": 0},), (), (target,), {}, {}, {}, scope)
    mechanism = Mechanism("increment", ({"name": "value_at_least", "value": 0},), (), target, scope, {"seconds": 1}, "confirmed")
    kb = InMemoryKnowledgeBank((mechanism,)); spt = SPT("spt:increment", "increment", "v1", {"family":"increment"})
    skills = InMemoryQualifiedSkillLibrary(spt, QualificationConfig(**config.get("qualification", {})))
    req = TransitionRequest(({"name":"value_at_least", "value":0},), target, {}, scope, {})
    spi = SPI("spi:increment:1", "increment", spt.spt_id, spt.version, {}, req, contract, {"target": target}, {}, {"steps": 1})
    module = skills.qualify(spi, "controlled-policy", 1, 1, 1)
    if module is None: raise RuntimeError("smoke module qualification unexpectedly failed")
    env = ControlledEnvironment(); knowledge_feedback=[]; skill_feedback=[]
    pipeline = InMemoryContinualLearningPipeline(kb, skills, env.execute, knowledge_feedback.append, skill_feedback.append, TaskVersionView("kb:v1", {spt.skill_family:spt.version}))
    outcome, transition = pipeline.run_transition(target, req.input_capabilities, scope, req, env.state())
    evolution = KnowledgeEvolution(config.get("knowledge", {"n_min":1,"tau_confirm":.5,"tau_reject":.2}))
    # Ordinary execution evidence is intentionally UNKNOWN and cannot update structural claims.
    evolution.record("increment_reachability", "bottom", knowledge_feedback[0])
    out_dir=Path(config.get("results_dir", "results/smoke")); out_dir.mkdir(parents=True, exist_ok=True)
    result={"metadata":runtime_metadata(config, resolved_device),"config":config,"task_result":outcome,"transition":asdict(transition) if transition else None,"knowledge_feedback":[asdict(x) if is_dataclass(x) else x for x in knowledge_feedback],"skill_feedback":skill_feedback,"knowledge_evolution":evolution.to_dict(),"module":asdict(module)}
    (out_dir/"result.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    with (out_dir/"result.csv").open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=["task_result","execution_status","target_achieved"]); w.writeheader(); w.writerow({"task_result":outcome,"execution_status":transition.execution_status if transition else "unavailable","target_achieved":transition.target_achieved if transition else "unknown"})
    return result

if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/smoke_cpu.yaml"); args=p.parse_args(); print(json.dumps(run(args.config), indent=2, default=str))
