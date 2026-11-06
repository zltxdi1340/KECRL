"""Capture paired 20-step RGB/action/proprioception windows for seven transitions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_bc_closed_loop_attribution as policy_loop
from experiments import run_libero_task4_task7_full_policy_closed_loop_diagnostic as closed_loop
from experiments.validate_libero_task4_task7_public_state_jitter_500_transition_rgb_action_protocol import (
    load_protocol,
    sha256_file,
)
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


DEFAULT_CONFIG = ROOT / "configs/libero-task4-task7-public-state-jitter-500-transition-rgb-action-diagnostic_v1.json"
DEFAULT_DATASET_ROOT = Path("/home/zl202621/third_party/LIBERO/datasets")
DEFAULT_OUTPUT = Path("/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-20261101-v1.json")
DEFAULT_ARCHIVE = Path("/tmp/libero-task4-task7-public-state-jitter-500-transition-rgb-action-archive-20261101-v1")
TASKS = (4, 7)
SEEDS = (20261022, 20261023, 20261024)
ARMS = ("source_baseline", "public_state_jitter")
CAPTURE_STEPS = 20


def _observation_digest(observation: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(observation):
        values = np.ascontiguousarray(observation[key])
        digest.update(key.encode("utf-8"))
        digest.update(str(values.dtype).encode("ascii"))
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _load_model(checkpoint: Mapping[str, Any], policy_spec: Mapping[str, Any], device: torch.device) -> tuple[LiberoBCPolicy, str]:
    path = Path(str(checkpoint["path"]))
    if sha256_file(path) != checkpoint["file_sha256"]:
        raise ValueError(f"checkpoint file hash differs: {path}")
    model = LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(policy_spec)).to(device)
    state_dict = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    weight_sha = causal._digest_tensors(model.state_dict())
    if weight_sha != checkpoint["weight_sha256"]:
        raise ValueError(f"checkpoint weight hash differs: {path}")
    model.eval()
    return model, weight_sha


def _capture_episode(*, model: LiberoBCPolicy, adapter: Any, task_id: int, training_seed: int, arm: str, state_index: int, device: torch.device, sequence_length: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    observation, reset_info = adapter.reset(initial_state_index=adapter._capture_local_state_index)
    initial_digest = _observation_digest(observation)
    observations = {key: [] for key in ("agentview_rgb", "eye_in_hand_rgb", "joint_states", "gripper_states")}
    actions: list[np.ndarray] = []
    hidden = None
    terminal_info: Mapping[str, Any] = {"success": False, "terminal_reason": None}
    with torch.no_grad():
        for step in range(CAPTURE_STEPS):
            for key in observations:
                observations[key].append(np.asarray(observation[key]).copy())
            if policy_loop.should_reset_hidden("stateless", step, sequence_length):
                hidden = None
            action_tensor, hidden = model.forward_step(*causal._online_tensors(observation, device), hidden=hidden)
            action = action_tensor[0].detach().cpu().numpy().astype(np.float32)
            actions.append(action.copy())
            observation, _reward, terminated, truncated, terminal_info = adapter.step(action)
            if terminated or truncated:
                if step + 1 < CAPTURE_STEPS:
                    raise RuntimeError(f"episode terminated before 20-step capture: task={task_id} seed={training_seed} arm={arm} state={state_index}")
                break
    if len(actions) != CAPTURE_STEPS:
        raise RuntimeError("capture did not produce exactly 20 actions")
    arrays = {key: np.stack(values, axis=0) for key, values in observations.items()}
    arrays["actions"] = np.stack(actions, axis=0)
    return {
        "task_id": task_id,
        "training_seed": training_seed,
        "arm": arm,
        "initial_state_index": state_index,
        "reset_observation_sha256": initial_digest,
        "reset_info": dict(reset_info),
        "capture_steps": CAPTURE_STEPS,
        "capture_terminal_reason": terminal_info.get("terminal_reason") or "capture_horizon",
        "capture_success": bool(terminal_info.get("success", False)),
    }, arrays


def run_diagnostic(*, config_path: Path, dataset_root: Path, device_name: str, output_path: Path, archive_dir: Path) -> dict[str, Any]:
    config_path = config_path.resolve(); dataset_root = dataset_root.resolve(); output_path = output_path.resolve(); archive_dir = archive_dir.resolve()
    if output_path.exists() or archive_dir.exists():
        raise ValueError("output or archive path already exists")
    config, _transition = load_protocol(config_path, allow_authorized_run=True)
    _original, _base, policy, transfer, _family, _manifest = closed_loop._load_protocol((ROOT / config["source_closed_loop_config"]).resolve())
    if not device_name.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("RGB/action capture requires CUDA")
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    from libero.libero import get_libero_path
    if dataset_root != Path(get_libero_path("datasets")).resolve():
        raise ValueError("dataset root differs from configured LIBERO dataset")
    init_root = Path(get_libero_path("init_states")).resolve()
    checkpoint_by_key = {(int(row["task_id"]), int(row["training_seed"]), row["arm"]): row for row in config["fixed_policy"]["checkpoint_files"]}
    selected = sorted(config["transition_episodes"], key=lambda row: (int(row["task_id"]), int(row["training_seed"]), int(row["initial_state_index"])))
    by_task_seed: dict[tuple[int, int], list[int]] = {}
    for row in selected:
        by_task_seed.setdefault((int(row["task_id"]), int(row["training_seed"])), []).append(int(row["initial_state_index"]))
    archive_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats(device)
    expected_digest: dict[tuple[int, int, int], str] = {}
    episode_rows: list[dict[str, Any]] = []; archive_manifest: list[dict[str, Any]] = []
    for (task_id, seed), state_indices in sorted(by_task_seed.items()):
        init_row = next(row for row in transfer["initial_state_files"] if int(row["task_id"]) == task_id)
        states = causal._capture_official_initial_states(initial_path=(init_root / init_row["path"]).resolve(), expected_size=int(init_row["size_bytes"]), expected_sha256=str(init_row["sha256"]), indices=tuple(state_indices))
        local_by_state = {state: index for index, state in enumerate(state_indices)}
        for arm in ARMS:
            checkpoint = checkpoint_by_key[(task_id, seed, arm)]
            model, weight_sha = _load_model(checkpoint, policy["policy"], device)
            adapter = causal._new_adapter(suite=transfer["suite"], task_id=task_id, seed=int(config["evaluation"]["environment_seed"]), image_size=int(policy["policy"]["image_size"]), max_steps=CAPTURE_STEPS, stabilization_steps=int(config["evaluation"]["stabilization_steps"]), states=states)
            try:
                for state_index in state_indices:
                    adapter._capture_local_state_index = local_by_state[state_index]
                    metadata, arrays = _capture_episode(model=model, adapter=adapter, task_id=task_id, training_seed=seed, arm=arm, state_index=state_index, device=device, sequence_length=int(policy["policy"]["sequence_length"]))
                    key = (task_id, seed, state_index)
                    if arm == "source_baseline":
                        expected_digest[key] = metadata["reset_observation_sha256"]
                    elif metadata["reset_observation_sha256"] != expected_digest[key]:
                        raise RuntimeError(f"paired reset observation differs: task={task_id} seed={seed} state={state_index}")
                    filename = f"task{task_id}-seed{seed}-state{state_index:02d}-arm-{arm}.npz"
                    archive_path = archive_dir / filename
                    np.savez_compressed(archive_path, **arrays)
                    archive_sha = sha256_file(archive_path)
                    metadata.update({"archive_file": str(archive_path), "archive_file_sha256": archive_sha, "weight_sha256": weight_sha, "archive_arrays": {key: {"shape": list(value.shape), "dtype": str(value.dtype)} for key, value in arrays.items()}})
                    episode_rows.append(metadata)
                    archive_manifest.append({"task_id":task_id,"training_seed":seed,"initial_state_index":state_index,"arm":arm,"path":str(archive_path),"sha256":archive_sha,"size_bytes":archive_path.stat().st_size})
            finally:
                adapter.close()
            del model; torch.cuda.empty_cache()
    archive_manifest.sort(key=lambda row:(row["task_id"],row["training_seed"],row["initial_state_index"],row["arm"]))
    archive_sha = hashlib.sha256(json.dumps(archive_manifest,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    episode_rows.sort(key=lambda row:(row["task_id"],row["training_seed"],row["initial_state_index"],row["arm"]))
    torch.cuda.synchronize(device)
    return {
        "check_type":"libero_task4_task7_public_state_jitter_500_transition_rgb_action_diagnostic","status":"complete","formal_result":False,"diagnostic_result":True,"training_run":False,"policy_actions_executed_in_environment":True,"candidate_observations_read":True,"new_rollout_executed":True,"checkpoint_selected":False,"spt_updated":False,"knowledge_updated":False,"module_qualified":False,
        "provenance":{"git_commit":subprocess.check_output(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True).strip(),"config_path":str(config_path),"config_sha256":sha256_file(config_path),"runner_sha256":sha256_file(Path(__file__).resolve()),"archive_manifest_sha256":archive_sha,"archive_file_count":len(archive_manifest),"training_budget_change":False,"episode_budget":14},
        "runtime":{"python":sys.version.split()[0],"torch":torch.__version__,"torch_cuda":torch.version.cuda,"device":str(device),"device_name":torch.cuda.get_device_name(device),"elapsed_seconds":time.perf_counter()-started,"episode_count":len(episode_rows),"paired_episode_count":14,"capture_steps":CAPTURE_STEPS,"peak_memory_allocated_bytes":int(torch.cuda.max_memory_allocated(device))},
        "scope":{**config["evaluation"],"new_rollout_executed":True,"new_training_executed":False,"archive_directory":str(archive_dir),"archive_manifest_sha256":archive_sha},
        "interpretation":config["interpretation_boundary"],"authorization":config["authorization"],"archive":{"directory":str(archive_dir),"manifest_sha256":archive_sha,"files":archive_manifest},"episode_rows":episode_rows,
    }


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--config",type=Path,default=DEFAULT_CONFIG); parser.add_argument("--dataset-root",type=Path,default=DEFAULT_DATASET_ROOT); parser.add_argument("--device",default="cuda:0"); parser.add_argument("--output",type=Path,default=DEFAULT_OUTPUT); parser.add_argument("--archive-dir",type=Path,default=DEFAULT_ARCHIVE); args=parser.parse_args()
    result=run_diagnostic(config_path=args.config,dataset_root=args.dataset_root,device_name=args.device,output_path=args.output,archive_dir=args.archive_dir); args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({"status":result["status"],"episodes":result["runtime"]["episode_count"],"archive":result["archive"]["directory"]},indent=2))


if __name__ == "__main__": main()
