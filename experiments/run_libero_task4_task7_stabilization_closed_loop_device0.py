"""Continue the paired diagnostic on the original CUDA/EGL device zero.

Reuse only the three complete, exactly reproduced task-4 five-step groups.
Repeat the incomplete task-7 group on device0 and then run all zero-step groups.
No old reproduction gate is relaxed and no model is retrained.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments import run_libero_task4_task7_stabilization_closed_loop as base

core, require = base.core, base.require
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-stabilization-closed-loop-device0-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-stabilization-closed-loop-device0-20261102-v1.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-stabilization-closed-loop-device0-artifacts-20261102-v1')
SCOPE = {'task_ids': [4, 7], 'training_seeds': list(base.SEEDS), 'checkpoint_updates': 500,
    'initial_state_role': 'candidate_nonregression', 'initial_state_indices': list(range(20)),
    'native_horizon_both_arms': 305, 'policy_action_limit': 300, 'environment_seed': 20261022,
    'reused_five_task4_episodes': 60, 'new_five_task7_episodes': 60, 'new_zero_episodes': 120,
    'new_episode_budget': 180, 'final_paired_episode_matrix': 240,
    'new_adapter_budget': 9, 'new_candidate_initial_rows_decoded': 40,
    'max_new_policy_action_steps': 54000, 'new_fixed_zero_steps': 300,
    'previous_policy_action_steps': 19761, 'previous_fixed_zero_steps': 335,
    'max_cumulative_policy_action_steps': 73761, 'cumulative_fixed_zero_steps': 635,
    'cumulative_episode_attempts': 247, 'cumulative_candidate_rows_decoded': 80,
    'inference_device': 'cuda:0', 'cuda_visible_devices': '0', 'egl_device_id': '0',
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_hdf5_payload_reads': 0,
    'qualification_accessed': False, 'reserved_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False, 'checkpoint_selected': False,
    'private_live_simulator_state_read': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}


def sha(path): return base.sha(Path(path))


def validate_parent(parent):
    require(parent['status'] == 'failed' and parent['failure'] == {'type': 'ValueError',
        'message': 'five-step baseline reproduction mismatch: task=7, state=6'}, 'unexpected parent failure')
    calls = parent['runtime']['actual_calls']
    require(calls == {'adapters_built': 4, 'episodes_completed': 67, 'five_episodes_exact': 66,
        'fixed_zero_stabilization_steps': 335, 'native_steps': 20096, 'policy_actions': 19761,
        'resets_attempted': 67, 'resets_completed': 67}, 'parent actual budget differs')
    require(parent['scope'] == base.SCOPE and parent['analysis'] == base.ANALYSIS and
            all(e['stabilization_steps'] == 5 for e in parent['episodes']), 'parent scope or arm differs')


def load_protocol(path):
    cfg = core.load_json(path)
    require(cfg.get('check_type') == 'libero_task4_task7_stabilization_device0_continuation' and
            cfg.get('formal_result') is False and cfg.get('scope') == SCOPE and
            cfg.get('analysis') == base.ANALYSIS, 'device0 frozen scope or analysis differs')
    require(cfg['runner_sha256'] == sha(Path(__file__)), 'device0 runner differs')
    require(set(cfg['inputs']) == {'parent', 'parent_config', 'base_code'}, 'device0 input roles differ')
    paths = {}
    for name, row in cfg['inputs'].items():
        p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(sha(p) == row['sha256'], f'device0 input pin differs: {name}'); paths[name] = p
    parent = core.load_json(paths['parent']); validate_parent(parent)
    require(sha(paths['parent_config']) == parent['provenance']['config_sha256'] and
            sha(paths['base_code']) == parent['provenance']['runner_sha256'] ==
            'fed40bd93ef152b00915b56d4317a332f0981fbef2a6f1b1b38f721de5255912', 'parent execution provenance differs')
    old_cfg = core.load_json(paths['parent_config']); require(old_cfg['scope'] == base.SCOPE and old_cfg['analysis'] == base.ANALYSIS, 'parent protocol differs')
    old_paths = {}
    for name, row in old_cfg['inputs'].items():
        p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(row['sha256'] == base.PINS[name] and sha(p) == row['sha256'], f'parent input differs: {name}'); old_paths[name] = p
    return cfg, paths, parent, old_paths


def reuse_task4(parent, baseline):
    require(len(parent['episodes']) == len(parent['episode_files']) == 67, 'parent episode files incomplete')
    episodes, files = [], []
    for ep, record in zip(parent['episodes'], parent['episode_files']):
        require(sha(record['path']) == record['sha256'] and core.load_json(Path(record['path'])) == ep, 'parent episode payload changed')
        if ep['task_id'] != 4: continue
        base.exact_baseline(ep, baseline['task_results']['4']['seed_results'][str(ep['training_seed'])]['candidate_nonregression']['episodes'][ep['initial_state_index']])
        episodes.append(ep); files.append(record)
    require([(e['training_seed'], e['initial_state_index']) for e in episodes] ==
            [(s, i) for s in base.SEEDS for i in range(20)], 'only complete task4 groups may be reused')
    return episodes, files


def run(config_path, output, artifacts):
    require(not output.exists() and not artifacts.exists(), 'device0 output/artifacts already exist')
    cfg, paths, parent, old_paths = load_protocol(config_path)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('MUJOCO_EGL_DEVICE_ID') == '0' and
            os.environ.get('MUJOCO_GL') == 'egl' and torch.cuda.is_available() and torch.cuda.device_count() == 1, 'original physical device0 required')
    torch.set_num_threads(1); torch.cuda.set_device(0); device = torch.device('cuda:0'); torch.cuda.reset_peak_memory_stats(device)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    baseline, source, transfer = [core.load_json(old_paths[name]) for name in ('baseline', 'source_analysis', 'transfer')]
    episodes, files = reuse_task4(parent, baseline)
    policy = {**core.load_json(old_paths['policy_config'])['policy'], 'memory_mode': 'cnn_only'}
    artifacts.mkdir(parents=True, exist_ok=False); started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    progress = dict.fromkeys(('adapters_built', 'resets_attempted', 'resets_completed', 'policy_actions',
        'native_steps', 'fixed_zero_stabilization_steps', 'episodes_completed', 'five_episodes_exact'), 0)
    access, failure = {}, None
    with (artifacts / 'events.jsonl').open('x') as log:
        def event(kind, **values):
            log.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}, sort_keys=True) + '\n'); log.flush()
        try:
            states = {}
            for task in base.TASKS:
                row = next(r for r in transfer['initial_state_files'] if r['task_id'] == task)
                states[task], access[str(task)] = base.load_candidate_prefix(Path(base.SCOPE['native_initial_root']) / row['path'], row, base.PREFIX_SHA[task])
            for arm, tasks in ((5, (7,)), (0, (4, 7))):
                if arm == 0: require(progress['five_episodes_exact'] == 60 and len(episodes) == 120, 'all five groups exact before zero')
                for task in tasks:
                    for seed in base.SEEDS:
                        base.original.causal._set_seed(seed)
                        checkpoint = source['task_results'][str(task)]['models'][str(seed)]['checkpoint']
                        reference = baseline['task_results'][str(task)]['seed_results'][str(seed)]
                        require(checkpoint['weight_sha256'] == reference['training']['final_model_sha256'] and checkpoint['optimizer_updates'] == 500, 'checkpoint differs')
                        model = core.load_policy(checkpoint, policy).to(device).eval(); before = core._digest_weights(model.state_dict())
                        adapter = base._build_injected_adapter(suite_name='libero_spatial', task_id=task, seed=20261022,
                            image_size=128, max_episode_steps=300, stabilization_steps=5, initial_states=states[task].copy())
                        progress['adapters_built'] += 1; adapter.stabilization_steps = arm
                        environment, in_reset = adapter.environment, [False]; step = environment.step
                        def counted(action):
                            if in_reset[0]:require(np.array_equal(action, np.zeros(7, np.float32)), 'nonzero stabilization command')
                            result = step(action); progress['native_steps'] += 1
                            if in_reset[0]:progress['fixed_zero_stabilization_steps'] += 1
                            return result
                        environment.step = counted; reset = adapter.reset
                        def resetting(*args, **kwargs):
                            in_reset[0] = True
                            try:return reset(*args, **kwargs)
                            finally:in_reset[0] = False
                        adapter.reset = resetting
                        try:
                            for state in range(20):
                                ep = base.rollout(model, adapter, task, seed, state, arm, device, progress)
                                path = artifacts / f'task{task}-seed{seed}-stabilization{arm}-state{state:02d}.json'
                                base.archive_episode(ep, path, files, episodes, progress,
                                    reference['candidate_nonregression']['episodes'][state] if arm == 5 else None)
                                event('episode_complete', task_id=task, seed=seed, state_index=state, stabilization_steps=arm,
                                      success=ep['success'], policy_steps=ep['steps'], exact_original_five=arm == 5)
                                print(json.dumps({'task': task, 'seed': seed, 'arm': arm, 'state': state,
                                    'success': ep['success'], 'new_episodes': progress['episodes_completed'], 'paired_rows': len(episodes)}), flush=True)
                        finally:environment.step = step; adapter.close()
                        require(before == core._digest_weights(model.state_dict()) == checkpoint['weight_sha256'] and
                                all(p.grad is None for p in model.parameters()), 'inference mutated fixed checkpoint')
                        del model; torch.cuda.empty_cache()
            require(len(episodes) == 240 and progress['episodes_completed'] == progress['resets_completed'] == 180 and
                    progress['adapters_built'] == 9 and progress['fixed_zero_stabilization_steps'] == 300 and
                    progress['five_episodes_exact'] == 60 and progress['policy_actions'] <= 54000 and
                    progress['native_steps'] == progress['policy_actions'] + 300, 'device0 budget differs')
        except Exception as exc:failure = {'type': type(exc).__name__, 'message': str(exc)}; event('technical_abort', **failure)
    torch.cuda.synchronize()
    result = {'check_type': 'libero_task4_task7_stabilization_device0_continuation', 'formal_result': False,
        'status': 'failed' if failure else 'complete', 'failure': failure, 'scope': SCOPE, 'analysis': base.ANALYSIS,
        'episodes': episodes, 'episode_files': files, 'initial_state_access': access,
        'task_results': base.paired_summary(episodes) if not failure else {},
        'parent_attempt': {'path': str(paths['parent']), 'sha256': sha(paths['parent']), 'runtime': parent['runtime'],
            'reused_task4_five_episodes': 60, 'excluded_incomplete_task7_attempts': 7,
            'reused_five_trajectories_exact_to_original_device0_reference': True},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'inputs': cfg['inputs'], 'original_input_paths': {name: str(p) for name, p in old_paths.items()},
            'started_at_utc': utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'event_log': {'path': str(artifacts / 'events.jsonl'), 'sha256': sha(artifacts / 'events.jsonl')},
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'egl_device_id': os.environ.get('MUJOCO_EGL_DEVICE_ID')},
        'runtime': {'actual_new_calls': progress, 'cumulative_episode_attempts': 67 + progress['episodes_completed'],
            'cumulative_policy_actions': 19761 + progress['policy_actions'], 'cumulative_fixed_zero_steps': 335 + progress['fixed_zero_stabilization_steps'],
            'device': str(device), 'device_name': torch.cuda.get_device_name(device), 'torch': torch.__version__,
            'python': sys.version.split()[0], 'cuda': torch.version.cuda, 'peak_cuda_memory_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
            'elapsed_seconds': time.perf_counter() - started, 'new_optimizer_updates': 0, 'new_backward_calls': 0}}
    with output.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    if failure:raise RuntimeError(f'device0 continuation failed: {failure}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args(); result = run(args.config.resolve(), args.output.resolve(), args.artifacts.resolve())
    print(json.dumps({'status': result['status'], 'task_results': result['task_results'], 'runtime': result['runtime']}))


if __name__ == '__main__':main()
