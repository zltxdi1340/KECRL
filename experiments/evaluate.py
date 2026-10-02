"""Read saved experiment results and emit a compact evaluation summary."""
import argparse, json
from pathlib import Path
def evaluate(path):
    data=json.loads(Path(path).read_text(encoding="utf-8")); t=data.get("task_result")
    return {"task_result":t,"completed":t=="completed","knowledge_status":next(iter(data.get("knowledge_evolution",{}).get("propositions",{}).values()),{}).get("status","unknown")}
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("result", default="results/smoke/result.json", nargs="?"); print(json.dumps(evaluate(p.parse_args().result), indent=2))
