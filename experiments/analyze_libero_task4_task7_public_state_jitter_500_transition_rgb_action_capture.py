"""Offline RGB/action and input-swap analysis of the captured 20-step windows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_bc_closed_loop_attribution as policy_loop
from experiments import run_libero_task4_task7_full_policy_closed_loop_diagnostic as closed_loop
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


DEFAULT_INPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-20261101-v3.json")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-analysis-20261101-v1.json")
EXPECTED_INPUT_SHA256 = "f647ce8642b03f93f92c72c68241038c952c6f07684df266c4b7f2423291e2a3"
WINDOW = 20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _l2(values: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average empty values")
    return float(statistics.fmean(values))


def _load_model(path: Path, weight_sha: str, policy_spec: Mapping[str, Any], device: torch.device) -> LiberoBCPolicy:
    model = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    if causal._digest_tensors(model.state_dict()) != weight_sha:
        raise ValueError(f"weight hash differs: {path}")
    model.eval()
    return model


@torch.no_grad()
def _forward_variant(model: LiberoBCPolicy, arrays: Mapping[str, np.ndarray], device: torch.device, *, rgb_source: str, proprio_source: str, sequence_length: int) -> np.ndarray:
    hidden = None
    outputs = []
    for step in range(WINDOW):
        if policy_loop.should_reset_hidden("stateless", step, sequence_length):
            hidden = None
        observation = {
            "agentview_rgb": arrays[f"{rgb_source}_agentview_rgb"][step] if f"{rgb_source}_agentview_rgb" in arrays else arrays["agentview_rgb"][step],
            "eye_in_hand_rgb": arrays[f"{rgb_source}_eye_in_hand_rgb"][step] if f"{rgb_source}_eye_in_hand_rgb" in arrays else arrays["eye_in_hand_rgb"][step],
            "joint_states": arrays[f"{proprio_source}_joint_states"][step] if f"{proprio_source}_joint_states" in arrays else arrays["joint_states"][step],
            "gripper_states": arrays[f"{proprio_source}_gripper_states"][step] if f"{proprio_source}_gripper_states" in arrays else arrays["gripper_states"][step],
        }
        action, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
        outputs.append(action[0].detach().cpu().numpy().astype(np.float64))
    return np.asarray(outputs)


def _delta_metrics(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    return {
        "mean_l2": float(np.linalg.norm(delta, axis=1).mean()),
        "first_l2": float(np.linalg.norm(delta[0])),
        "translation_l2_mean": float(np.linalg.norm(delta[:, :3], axis=1).mean()),
        "rotation_l2_mean": float(np.linalg.norm(delta[:, 3:6], axis=1).mean()),
        "gripper_abs_mean": float(np.abs(delta[:, 6]).mean()),
        "maximum_l2": float(np.linalg.norm(delta, axis=1).max()),
    }


def analyze(input_path: Path = DEFAULT_INPUT, *, device_name: str = "cpu") -> dict[str, Any]:
    input_path = input_path.resolve()
    if sha256_file(input_path) != EXPECTED_INPUT_SHA256:
        raise ValueError("capture result hash differs")
    data = _load_json(input_path)
    if data.get("status") != "complete" or data.get("runtime", {}).get("episode_count") != 14:
        raise ValueError("capture input is incomplete")
    if not device_name.startswith("cpu") and not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    _original, _base, policy, _transfer, _family, _manifest = closed_loop._load_protocol((ROOT / "configs/libero-task4-task7-full-policy-closed-loop-diagnostic_v1.json").resolve())
    episodes = data["episode_rows"]
    grouped: dict[tuple[int, int, int], dict[str, Mapping[str, Any]]] = {}
    for row in episodes:
        grouped.setdefault((int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"])), {})[row["arm"]] = row
    if any(set(value) != {"source_baseline", "public_state_jitter"} for value in grouped.values()):
        raise ValueError("capture pairs are incomplete")
    checkpoint_by_key = {(int(row["task_id"]), int(row["training_seed"]), row["arm"]): row for row in data["archive"]["files"]}
    # The archive manifest points to arrays; checkpoint paths and weight hashes
    # are taken from the capture result's fixed-policy manifest.
    fixed = {(int(row["task_id"]), int(row["training_seed"]), row["arm"]): row for row in _load_json(ROOT / "configs/libero-task4-task7-public-state-jitter-500-transition-rgb-action-diagnostic_v1.json")["fixed_policy"]["checkpoint_files"]}
    transition_config = _load_json(ROOT / "configs/libero-task4-task7-public-state-jitter-500-transition-rgb-action-diagnostic_v1.json")
    transition_by_key = {
        (int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"])): row["transition"]
        for row in transition_config["transition_episodes"]
    }
    models: dict[tuple[int, int, str], LiberoBCPolicy] = {}
    for key, checkpoint in fixed.items():
        models[key] = _load_model(Path(checkpoint["path"]), checkpoint["weight_sha256"], policy["policy"], device)
    rows = []
    for key in sorted(grouped):
        pair = grouped[key]
        arrays_by_arm = {}
        for arm in ("source_baseline", "public_state_jitter"):
            row = pair[arm]
            archive = Path(row["archive_file"])
            if sha256_file(archive) != row["archive_file_sha256"]:
                raise ValueError(f"archive hash differs: {archive}")
            loaded = np.load(archive)
            arrays_by_arm[arm] = {name: loaded[name] for name in loaded.files}
        baseline_arrays = arrays_by_arm["source_baseline"]
        jitter_arrays = arrays_by_arm["public_state_jitter"]
        rgb_mae = {
            "agentview_rgb_mae_mean": float(np.abs(baseline_arrays["agentview_rgb"].astype(np.float32) - jitter_arrays["agentview_rgb"].astype(np.float32)).mean() / 255.0),
            "eye_in_hand_rgb_mae_mean": float(np.abs(baseline_arrays["eye_in_hand_rgb"].astype(np.float32) - jitter_arrays["eye_in_hand_rgb"].astype(np.float32)).mean() / 255.0),
            "agentview_rgb_mae_by_step": (np.abs(baseline_arrays["agentview_rgb"].astype(np.float32) - jitter_arrays["agentview_rgb"].astype(np.float32)).mean(axis=(1,2,3)) / 255.0).tolist(),
            "eye_in_hand_rgb_mae_by_step": (np.abs(baseline_arrays["eye_in_hand_rgb"].astype(np.float32) - jitter_arrays["eye_in_hand_rgb"].astype(np.float32)).mean(axis=(1,2,3)) / 255.0).tolist(),
        }
        baseline_model = models[(key[0], key[1], "source_baseline")]
        jitter_model = models[(key[0], key[1], "public_state_jitter")]
        # Add source prefixes so _forward_variant can switch modalities.
        variant_arrays = {
            "baseline_agentview_rgb": baseline_arrays["agentview_rgb"], "baseline_eye_in_hand_rgb": baseline_arrays["eye_in_hand_rgb"], "baseline_joint_states": baseline_arrays["joint_states"], "baseline_gripper_states": baseline_arrays["gripper_states"],
            "jitter_agentview_rgb": jitter_arrays["agentview_rgb"], "jitter_eye_in_hand_rgb": jitter_arrays["eye_in_hand_rgb"], "jitter_joint_states": jitter_arrays["joint_states"], "jitter_gripper_states": jitter_arrays["gripper_states"],
        }
        # Natural action reproduction is checked against archived action arrays.
        baseline_natural = _forward_variant(baseline_model, variant_arrays, device, rgb_source="baseline", proprio_source="baseline", sequence_length=int(policy["policy"]["sequence_length"]))
        jitter_natural = _forward_variant(jitter_model, variant_arrays, device, rgb_source="jitter", proprio_source="jitter", sequence_length=int(policy["policy"]["sequence_length"]))
        if not np.allclose(baseline_natural, baseline_arrays["actions"], atol=2e-4, rtol=0.0):
            raise ValueError("baseline natural action reproduction differs")
        if not np.allclose(jitter_natural, jitter_arrays["actions"], atol=2e-4, rtol=0.0):
            raise ValueError("jitter natural action reproduction differs")
        cross = {
            "baseline_policy_weight_delta_on_baseline_observation": _delta_metrics(baseline_natural, _forward_variant(jitter_model, variant_arrays, device, rgb_source="baseline", proprio_source="baseline", sequence_length=int(policy["policy"]["sequence_length"]))),
            "baseline_policy_rgb_only_swap": _delta_metrics(baseline_natural, _forward_variant(baseline_model, variant_arrays, device, rgb_source="jitter", proprio_source="baseline", sequence_length=int(policy["policy"]["sequence_length"]))),
            "baseline_policy_proprio_only_swap": _delta_metrics(baseline_natural, _forward_variant(baseline_model, variant_arrays, device, rgb_source="baseline", proprio_source="jitter", sequence_length=int(policy["policy"]["sequence_length"]))),
            "jitter_policy_full_observation_swap": _delta_metrics(jitter_natural, _forward_variant(jitter_model, variant_arrays, device, rgb_source="baseline", proprio_source="baseline", sequence_length=int(policy["policy"]["sequence_length"]))),
            "jitter_policy_rgb_only_swap": _delta_metrics(jitter_natural, _forward_variant(jitter_model, variant_arrays, device, rgb_source="baseline", proprio_source="jitter", sequence_length=int(policy["policy"]["sequence_length"]))),
            "jitter_policy_proprio_only_swap": _delta_metrics(jitter_natural, _forward_variant(jitter_model, variant_arrays, device, rgb_source="jitter", proprio_source="baseline", sequence_length=int(policy["policy"]["sequence_length"]))),
        }
        rows.append({"task_id":key[0],"training_seed":key[1],"initial_state_index":key[2],"transition":transition_by_key[key],"rgb_metrics":rgb_mae,"captured_action_delta":_delta_metrics(baseline_arrays["actions"],jitter_arrays["actions"]),"cross_readout":cross})
    for model in models.values(): del model
    if device.type == "cuda": torch.cuda.empty_cache()
    return {"check_type":"libero_task4_task7_public_state_jitter_500_transition_rgb_action_capture_analysis","status":"complete","formal_result":False,"diagnostic_result":True,"read_only_analysis":True,"training_run":False,"new_rollout_executed":False,"qualification_accessed":False,"validation_accessed":False,"heldout_accessed":False,"final_evaluation_accessed":False,"checkpoint_selected":False,"spt_updated":False,"knowledge_updated":False,"module_qualified":False,"provenance":{"input_path":str(input_path),"input_sha256":sha256_file(input_path),"analysis_script_sha256":sha256_file(Path(__file__).resolve()),"device":str(device)},"scope":{"paired_episode_count":len(rows),"capture_steps":WINDOW,"rgb_keys":["agentview_rgb","eye_in_hand_rgb"],"private_simulator_state_read":False},"rows":rows,"interpretation":{"rgb_action_alignment_is_descriptive":True,"cross_readout_is_fixed_policy_input_sensitivity":True,"no_causal_claim":True,"formal_spi_result":False,"module_qualification":False}}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",type=Path,default=DEFAULT_INPUT); parser.add_argument("--device",default="cpu"); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); args=parser.parse_args(); result=analyze(args.input,device_name=args.device); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n'); print(json.dumps({"status":result["status"],"pairs":len(result["rows"]),"output":str(args.output.resolve())},indent=2))


if __name__ == "__main__": main()
