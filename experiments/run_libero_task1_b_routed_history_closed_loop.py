"""Fresh paired candidate-only control with frozen routed readouts and shared startup."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import libero_routed_history_runtime as runtime
from experiments import run_libero_source_internal_spt_spi_closed_loop_diagnostic as legacy

saved = runtime.saved
require, sha, load, resolve = saved.require, saved.sha, saved.load, saved.resolve
SEEDS, ARMS = saved.SEEDS, runtime.ARMS
PREFIX = 'libero-task1-b-routed-history-closed-loop'
DEFAULT_CONFIG = ROOT / f'configs/{PREFIX}-diagnostic_v1.json'
DEFAULT_BINDING = ROOT / f'configs/{PREFIX}-binding_v1.json'
DEFAULT_OUTPUT = Path(f'/tmp/{PREFIX}-20261104-v1.json')
DEFAULT_ARTIFACTS = Path(f'/tmp/{PREFIX}-artifacts-20261104-v1')
CONFIG_SHA = 'b69b72b5fffba3e239e34eddc1e8e2c6bbcd57148473f6bb982f245518aaa288'


def preflight(config_path, binding_path):
    c = load(config_path)
    expected = {'task_id': 1, 'training_seeds': list(SEEDS), 'arms': list(ARMS), 'encoder_updates': 500,
        'readout_updates': 500, 'models': 6, 'initial_state_role': 'candidate_nonregression',
        'initial_state_indices': list(range(20)), 'episodes_per_model': 20, 'episodes': 120, 'fresh_episodes': 120,
        'reused_success_results': 0, 'adapters': 6, 'environment_seed': 20261022, 'stabilization_steps': 5,
        'policy_action_limit': 300, 'native_horizon': 305, 'maximum_policy_actions': 36000,
        'fixed_zero_actions': 600, 'maximum_native_step_calls': 36600, 'candidate_initial_rows_decoded': 20,
        'qualification_reserved_rows_decoded': 0, 'reset_digest_references': 20, 'reset_digest_comparisons': 100,
        'maximum_shared_startup_action_comparisons': 240, 'maximum_shared_prefix_observation_comparisons': 300,
        'history_steps': 5, 'shared_baseline_startup_actions': 4, 'inference_device': 'cpu', 'cuda_visible_devices': '',
        'physical_renderer_gpu': 1, 'egl_device_id': '1', 'new_optimizer_updates': 0, 'new_backward_calls': 0,
        'new_HDF5_payload_reads': 0, 'qualification_accessed': False, 'reserved_accessed': False,
        'validation_accessed': False, 'heldout_accessed': False, 'final_evaluation_accessed': False,
        'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False,
        'private_live_state_recorded': False, 'private_initial_vector_values_recorded': False}
    require(c['formal_result'] is False, 'diagnostic only')
    for k, v in expected.items():
        require(c['scope'].get(k) == v and type(c['scope'].get(k)) is type(v), f'frozen scope differs: {k}')
    require(sha(config_path) == CONFIG_SHA, 'config hash differs')
    for name, pin in c['inputs'].items():
        require(sha(pin['path']) == pin['sha256'], f'input hash differs: {name}')
    b = load(binding_path)
    require(b['formal_result'] is False and b['config_sha256'] == CONFIG_SHA and set(b['files']) == {'runner', 'tests', 'protocol'}, 'binding differs')
    for pin in b['files'].values():
        require(sha(pin['path']) == pin['sha256'], 'execution code differs')
    require(resolve(b['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'runner binding differs')
    interface = load(c['inputs']['interface_result']['path'])
    audit = load(c['inputs']['interface_audit']['path'])
    require(interface['status'] == 'complete_routed_runtime_interface' and interface['passed'] is True and
            audit['status'] == 'complete_independent_routed_runtime_interface_audit' and
            audit['result_sha256'] == c['inputs']['interface_result']['sha256'], 'interface gate differs')
    for key in ('interface_binding', 'routed_binding'):
        for pin in load(c['inputs'][key]['path'])['files'].values():
            require(sha(pin['path']) == pin['sha256'], 'historical bound code differs')
    return c, b


def idle_renderer():
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,name,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True)
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory', '--format=csv,noheader,nounits'], text=True)
    row = next([v.strip() for v in line.split(',')] for line in rows.splitlines() if int(line.split(',')[0]) == 1)
    require(int(row[3]) <= 512 and int(row[4]) <= 5 and not any(row[1] in line for line in processes.splitlines()), 'frozen renderer GPU1 is occupied; no device fallback')
    return {'gpu_rows_before_run': rows, 'compute_processes_before_run': processes, 'renderer_gpu': 1}


def new_counts():
    return {k: 0 for k in ('adapters_built', 'episodes_attempted', 'episodes_completed', 'resets_attempted', 'resets_completed',
        'encoder_forward_attempts', 'encoder_forwards_completed', 'readout_forward_attempts', 'readout_forwards_completed',
        'policy_action_attempts', 'policy_actions_completed', 'native_step_attempts', 'native_steps_completed',
        'zero_step_attempts', 'zero_steps_completed', 'reset_digest_references', 'reset_digest_comparisons',
        'startup_action_references', 'startup_action_comparisons', 'prefix_observation_references', 'prefix_observation_comparisons',
        'readout_primitive_linear_calls', 'checkpoint_loads', 'new_optimizer_updates', 'new_backward_calls', 'new_HDF5_payload_reads')}


def paired_summary(episodes):
    require(len(episodes) == 120 and all(e['status'] == 'complete' for e in episodes), 'complete120 required for primary result')
    indexed = {(e['training_seed'], e['arm'], e['initial_state_index']): e for e in episodes}
    require(len(indexed) == 120 and set(indexed) == {(s, a, i) for s in SEEDS for a in ARMS for i in range(20)}, 'duplicate or missing episodes')
    rows = []
    for seed in SEEDS:
        total = {a: sum(int(indexed[seed, a, i]['success']) for i in range(20)) for a in ARMS}
        outcomes = {k: 0 for k in ('both_success', 'candidate_only_success', 'baseline_only_success', 'both_failure')}
        for i in range(20):
            a, b = [indexed[seed, arm, i]['success'] for arm in ARMS]
            key = 'both_success' if a and b else 'baseline_only_success' if a else 'candidate_only_success' if b else 'both_failure'
            outcomes[key] += 1
        rows.append({'training_seed': seed, 'episodes_per_arm': 20, 'success_counts': total,
            'success_rates': {a: n/20 for a, n in total.items()}, 'paired_outcomes': outcomes,
            'candidate_minus_baseline_success_n': total[ARMS[1]]-total[ARMS[0]]})
    totals = {a: sum(r['success_counts'][a] for r in rows) for a in ARMS}
    return {'seed_results': rows, 'episodes_per_arm': 60, 'success_counts': totals,
        'success_rates': {a: n/60 for a, n in totals.items()},
        'paired_outcomes': {k: sum(r['paired_outcomes'][k] for r in rows) for k in rows[0]['paired_outcomes']},
        'strict_candidate_improvement_all_three_seeds': all(r['candidate_minus_baseline_success_n'] > 0 for r in rows),
        'automatic_training_or_spt_promotion': False}


def execute_models(*, model_loader, adapter_factory, predictor, checker, archive, counts, event):
    episodes, files, resets, prefix_obs, prefix_actions = [], [], {}, {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            model = model_loader(seed, arm)
            adapter = adapter_factory()
            counts['adapters_built'] += 1
            native_step, in_reset, reset_zeros = adapter.environment.step, [False], [0]

            def counted_step(action):
                require(counts['native_step_attempts'] < 36600, 'native step limit exceeded')
                if in_reset[0]:
                    require(reset_zeros[0] < 5 and counts['zero_step_attempts'] < 600 and
                            np.array_equal(np.asarray(action), np.zeros(7, np.float32)), 'stabilization contract differs')
                    reset_zeros[0] += 1
                    counts['zero_step_attempts'] += 1
                counts['native_step_attempts'] += 1
                result = native_step(action)
                counts['native_steps_completed'] += 1
                if in_reset[0]:
                    counts['zero_steps_completed'] += 1
                return result

            adapter.environment.step = counted_step
            try:
                for state in range(20):
                    model['reader'].reset()
                    ep = {'task_id': 1, 'training_seed': seed, 'arm': arm, 'initial_state_role': 'candidate_nonregression',
                        'initial_state_index': state, 'environment_seed': 20261022, 'stabilization_steps': 5,
                        'encoder': model['checkpoint'], 'readout_archives': model['readout_archives'],
                        'status': 'running', 'stepwise_public_state_trace': [], 'shared_prefix_pre_observation_digests': []}
                    episode_path = archive / f'seed{seed}-{arm}-state{state:02d}.json'
                    feature_path = archive / f'seed{seed}-{arm}-state{state:02d}-public-derived-features.npz'
                    features, actions = [], []
                    counts['episodes_attempted'] += 1
                    try:
                        counts['resets_attempted'] += 1
                        reset_zeros[0] = 0
                        before_zeros = counts['zero_steps_completed']
                        in_reset[0] = True
                        try:
                            obs, _info = adapter.reset(initial_state_index=state)
                        finally:
                            in_reset[0] = False
                        counts['resets_completed'] += 1
                        require(counts['zero_steps_completed']-before_zeros == 5, 'reset needs exactly five zero actions')
                        digest = legacy.observation_digest(obs)
                        ep['reset_observation_sha256'] = digest
                        if state not in resets:
                            resets[state] = digest
                            counts['reset_digest_references'] += 1
                        else:
                            require(resets[state] == digest, f'reset differs before first action: seed{seed}/{arm}/state{state}')
                            counts['reset_digest_comparisons'] += 1
                        j0, g0 = [np.asarray(obs[k], np.float64).copy() for k in ('joint_states', 'gripper_states')]
                        ep['stepwise_public_state_trace'].append({'step': 0, 'joint_state': j0.tolist(), 'gripper_state': g0.tolist(),
                            'joint_offset_l2_from_start': 0., 'gripper_offset_l2_from_start': 0.})
                        reward_sum = 0.
                        for step in range(1, 301):
                            if step <= 5:
                                digest = legacy.observation_digest(obs)
                                key = (seed, state, step)
                                ep['shared_prefix_pre_observation_digests'].append(digest)
                                if arm == ARMS[0]:
                                    prefix_obs[key] = digest
                                    counts['prefix_observation_references'] += 1
                                else:
                                    require(prefix_obs.get(key) == digest, 'shared-startup observation prefix differs')
                                    counts['prefix_observation_comparisons'] += 1
                            require(counts['encoder_forward_attempts'] < 36000, 'encoder budget exceeded')
                            feature, action, meta = predictor(model, obs, step, counts)
                            require(feature.shape == (256,) and feature.dtype == np.float32 and action.shape == (7,) and
                                    action.dtype == np.float32 and np.isfinite(feature).all() and np.isfinite(action).all() and
                                    (np.abs(action) <= 1).all(), 'invalid runtime feature/action')
                            require(meta['policy_step'] == step and meta['history_length'] == min(step, 5) and
                                    meta['startup'] == (step <= 4) and meta['history_policy_steps'] == list(range(step, max(0, step-5), -1)), 'runtime history contract differs')
                            if step <= 4:
                                key = (seed, state, step)
                                if arm == ARMS[0]:
                                    prefix_actions[key] = action.copy()
                                    counts['startup_action_references'] += 1
                                else:
                                    require(key in prefix_actions and np.array_equal(prefix_actions[key], action), 'shared-startup action differs')
                                    counts['startup_action_comparisons'] += 1
                            features.append(feature)
                            actions.append(action)
                            ep['last_attempted_action'] = action.tolist()
                            counts['policy_action_attempts'] += 1
                            obs, reward, terminated, truncated, info = adapter.step(action)
                            counts['policy_actions_completed'] += 1
                            require(np.isfinite(reward), 'nonfinite reward')
                            reward_sum += float(reward)
                            j, g = [np.asarray(obs[k], np.float64) for k in ('joint_states', 'gripper_states')]
                            require(j.shape == (7,) and g.shape == (2,) and np.isfinite(j).all() and np.isfinite(g).all(), 'invalid public state')
                            ep['stepwise_public_state_trace'].append({'step': step, 'action': action.tolist(), 'joint_state': j.tolist(),
                                'gripper_state': g.tolist(), 'joint_offset_from_start': (j-j0).tolist(), 'gripper_offset_from_start': (g-g0).tolist(),
                                'joint_offset_l2_from_start': float(np.linalg.norm(j-j0)), 'gripper_offset_l2_from_start': float(np.linalg.norm(g-g0)),
                                'history': meta})
                            if terminated or truncated:
                                require(not (terminated and truncated) and bool(terminated) == bool(info['success']) and
                                    info['terminal_reason'] == ('success' if terminated else 'time_limit') and (terminated or step == 300), 'invalid terminal boundary')
                                ep.update(status='complete', success=bool(info['success']), terminal_reason=info['terminal_reason'],
                                          steps=step, reward_sum=reward_sum)
                                break
                        require(ep['status'] == 'complete', 'episode did not stop by300')
                        ep['public_derived_feature_archive'] = saved.write_npz(feature_path, {
                            'projection': np.stack(features), 'actions': np.stack(actions), 'policy_steps': np.arange(1, len(features)+1, dtype=np.int32)})
                        saved.write_json(episode_path, ep)
                        files.append({'path': str(episode_path), 'sha256': sha(episode_path)})
                        episodes.append({k: v for k, v in ep.items() if k not in ('stepwise_public_state_trace', 'last_attempted_action')})
                        counts['episodes_completed'] += 1
                        event('episode_complete', seed=seed, arm=arm, state=state, success=ep['success'], steps=ep['steps'], episodes=counts['episodes_completed'])
                    except Exception as exc:
                        ep.update(status='failed', error_type=type(exc).__name__, error=str(exc), completed_actions_in_episode=len(ep['stepwise_public_state_trace'])-1)
                        if features and not feature_path.exists():
                            ep['partial_public_derived_features'] = saved.write_npz(feature_path, {'projection': np.stack(features), 'actions': np.stack(actions)})
                        if not episode_path.exists():
                            saved.write_json(episode_path, ep)
                        event('episode_failed', path=str(episode_path), error=str(exc))
                        raise
                checker(model)
            finally:
                adapter.environment.step = native_step
                adapter.close()
    require(counts['episodes_completed'] == counts['resets_completed'] == 120 and counts['adapters_built'] == 6, 'episode/adapter budget differs')
    require(counts['reset_digest_references'] == 20 and counts['reset_digest_comparisons'] == 100 and counts['zero_steps_completed'] == 600, 'reset/zero budget differs')
    for attempted, completed in [('encoder_forward_attempts', 'encoder_forwards_completed'), ('readout_forward_attempts', 'readout_forwards_completed'),
        ('policy_action_attempts', 'policy_actions_completed'), ('native_step_attempts', 'native_steps_completed'), ('zero_step_attempts', 'zero_steps_completed')]:
        require(counts[attempted] == counts[completed], 'complete result contains failed operation attempts')
    require(counts['policy_actions_completed'] == counts['encoder_forwards_completed'] == counts['readout_forwards_completed'] <= 36000 and
            counts['native_steps_completed'] == counts['policy_actions_completed']+600, 'action budget differs')
    require(counts['startup_action_comparisons'] == counts['startup_action_references'] <= 240 and
            counts['prefix_observation_comparisons'] == counts['prefix_observation_references'] <= 300, 'prefix comparison budget differs')
    return episodes, files


def run(config_path=DEFAULT_CONFIG, binding_path=DEFAULT_BINDING, output=DEFAULT_OUTPUT, artifacts=DEFAULT_ARTIFACTS):
    output, artifacts = Path(output), Path(artifacts)
    failure = output.with_name(output.stem+'-failed.json')
    require(not any(p.exists() for p in (output, artifacts, failure)), 'outputs already exist')
    counts, start, event_handle = new_counts(), time.perf_counter(), None
    initial_access, stage, gpu = None, 'preflight', None
    try:
        c, _ = preflight(config_path, binding_path)
        require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and os.environ.get('MUJOCO_GL') == 'egl' and
                os.environ.get('MUJOCO_EGL_DEVICE_ID') == '1' and all(os.environ.get(k) == '1' for k in
                ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'fixed CPU/EGL launch differs')
        require(sys.version.split()[0] == c['runtime_versions']['python'], 'Python runtime differs')
        for key, version in c['runtime_versions'].items():
            if key != 'python':
                require(importlib.metadata.version(key) == version, f'runtime package differs: {key}')
        require(subprocess.check_output(['git', '-C', '/home/zl202621/third_party/LIBERO', 'rev-parse', 'HEAD'], text=True).strip() == c['LIBERO_revision'], 'LIBERO runtime revision differs')
        stage = 'renderer_resource_guard'
        gpu = idle_renderer()
        import torch
        require(torch.cuda.device_count() == 0, 'CPU inference must not initialize CUDA')
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        from experiments.run_libero_task4_task7_stabilization_closed_loop import load_candidate_prefix
        from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
        artifacts.mkdir(parents=True, exist_ok=False)
        event_handle = (artifacts/'events.jsonl').open('x')

        def event(kind, **fields):
            event_handle.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **fields}, allow_nan=False)+'\n')
            event_handle.flush()
            if kind in ('episode_complete', 'episode_failed'):
                print(json.dumps({'progress': kind, **fields}), flush=True)

        event('runtime_ready', renderer=gpu, inference_device='cpu')
        stage = 'candidate_prefix_only'
        bank = c['initial_bank']
        values, initial_access = load_candidate_prefix(Path(c['native_initial_root'])/bank['relative_path'], bank, bank['pickle_header_sha256'])
        learning = load(c['inputs']['learning']['path'])

        def model_loader(seed, arm):
            ck = next(x['checkpoint'] for x in c['fixed_encoders'] if x['checkpoint']['training_seed'] == seed)
            encoder = runtime.load_encoder(ck, learning)
            counts['checkpoint_loads'] += 1
            reader = runtime.FrozenHistoryReadout(runtime.load_heads(c['inputs'], seed), arm)
            return {'encoder': encoder, 'reader': reader, 'initial_reader_hash': reader.state_hash(), 'checkpoint': ck,
                'readout_archives': {a: c['inputs'][f'{seed}_{a}'] for a in ARMS}}

        def adapter_factory():
            random.seed(20261022)
            np.random.seed(20261022)
            torch.manual_seed(20261022)
            return _build_injected_adapter(suite_name='libero_spatial', task_id=1, seed=20261022, image_size=128,
                max_episode_steps=300, stabilization_steps=5, initial_states=values.copy())

        def predictor(model, obs, step, progress):
            progress['encoder_forward_attempts'] += 1
            feature, _original = runtime.encode_step(model['encoder'], obs)
            progress['encoder_forwards_completed'] += 1
            progress['readout_forward_attempts'] += 1
            action, meta = model['reader'].act(feature, step)
            progress['readout_forwards_completed'] += 1
            progress['readout_primitive_linear_calls'] += 2 if meta['route'] == 'candidate_motion_history_current_gripper' else 1
            return feature, action, meta

        def checker(model):
            require(saved.tensor_digest(model['encoder'].state_dict()) == model['checkpoint']['weight_hash'] and
                    model['reader'].state_hash() == model['initial_reader_hash'] and
                    all(p.grad is None and not p.requires_grad for p in model['encoder'].parameters()), 'frozen controller mutated')

        stage = 'paired120_closed_loop'
        episodes, files = execute_models(model_loader=model_loader, adapter_factory=adapter_factory, predictor=predictor,
            checker=checker, archive=artifacts, counts=counts, event=event)
        summary = paired_summary(episodes)
        require(counts['checkpoint_loads'] == 6, 'encoder load budget differs')
        for pin in c['inputs'].values():
            require(sha(pin['path']) == pin['sha256'], 'input mutated')
        event('complete', actual_counts=counts, summary=summary)
        result = {'status': 'complete_routed_history_closed_loop', 'formal_result': False,
            'created_at_utc': datetime.now(timezone.utc).isoformat(), 'config_sha256': sha(config_path), 'binding_sha256': sha(binding_path),
            'scope': c['scope'], 'analysis': c['analysis'], 'inputs': c['inputs'], 'initial_state_access': initial_access,
            'actual_counts': counts, 'summary': summary, 'episodes': episodes, 'episode_files': files,
            'event_log': {'path': str(artifacts/'events.jsonl'), 'sha256': sha(artifacts/'events.jsonl')},
            'runtime': {'python': sys.version.split()[0], 'versions': c['runtime_versions'], 'inference_device': 'cpu',
                'physical_renderer_gpu': 1, 'gpu_preflight': gpu, 'threads': 1, 'deterministic': True,
                'entry_elapsed_seconds': time.perf_counter()-start, 'argv': sys.argv}}
        saved.write_json(output, result)
        return result
    except Exception as exc:
        saved.write_json(failure, {'status': 'failed_routed_history_closed_loop', 'formal_result': False, 'stage': stage,
            'error_type': type(exc).__name__, 'error': str(exc), 'actual_counts_so_far': counts, 'initial_state_access': initial_access,
            'primary_success_summary_valid': False, 'partial_episode_files': [str(p) for p in sorted(artifacts.glob('seed*.json'))] if artifacts.exists() else [],
            'entry_elapsed_seconds': time.perf_counter()-start})
        raise
    finally:
        if event_handle:
            event_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key, value in [('config', DEFAULT_CONFIG), ('binding', DEFAULT_BINDING), ('output', DEFAULT_OUTPUT), ('artifacts', DEFAULT_ARTIFACTS)]:
        parser.add_argument('--'+key, type=Path, default=value)
    parser.add_argument('--execute', action='store_true')
    a = parser.parse_args()
    if a.execute:
        r = run(a.config, a.binding, a.output, a.artifacts)
        print(json.dumps({'status': r['status'], 'actual_counts': r['actual_counts'], 'summary': r['summary']}))
    else:
        c, _ = preflight(a.config, a.binding)
        print(json.dumps({'status': 'metadata_preflight_passed', 'scope': c['scope']}))


if __name__ == '__main__':
    main()
