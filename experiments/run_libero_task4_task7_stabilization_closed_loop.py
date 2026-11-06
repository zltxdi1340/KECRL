"""Fixed baseline500, paired stabilization 5/0 candidate-only closed loop.

Candidate initial vectors are decoded only for rows 0..19 of the pinned stored
NumPy pickle. Qualification/reserved numerical rows are never deserialized.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import run_libero_task4_task7_full_policy_closed_loop_diagnostic as original
from experiments import analyze_libero_task4_task7_evidence_decomposition as core
from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter

require, sha = core.require, core.sha256_file
TASKS, SEEDS = (4, 7), (20261022, 20261023, 20261024)
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-stabilization-closed-loop-diagnostic_v2.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-stabilization-closed-loop-20261102-v2.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-stabilization-closed-loop-artifacts-20261102-v2')
SCOPE = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'checkpoint_updates': 500,
    'arm': 'source_baseline', 'stabilization_arms_in_order': [5, 0],
    'initial_state_role': 'candidate_nonregression', 'initial_state_indices': list(range(20)),
    'initial_state_set_previously_used': True, 'environment_seed': 20261022,
    'policy_action_limit': 300, 'native_horizon_both_arms': 305,
    'builder_stabilization_both_arms': 5, 'adapters': 12, 'episodes': 240,
    'episodes_per_arm': 120, 'max_policy_action_steps': 72000, 'fixed_zero_stabilization_steps': 600,
    'candidate_initial_rows_decoded': 40, 'qualification_or_reserved_rows_decoded': 0,
    'native_initial_root': '/home/zl202621/third_party/LIBERO/libero/libero/init_files',
    'adapter_lifecycle': 'one_per_task_seed_arm; state0_to19_ascending; all_five_exact_before_any_zero',
    'inference_device': 'cuda:0', 'cuda_visible_devices': '1', 'egl_device_id': '1', 'torch_threads': 1,
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_hdf5_payload_reads': 0,
    'private_live_simulator_state_read': False, 'private_initial_vector_values_recorded': False,
    'qualification_accessed': False, 'reserved_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False,
    'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
ANALYSIS = {'five_arm_reset_action_public_state_and_terminal_exact_to_original': True,
    'no_tolerance_or_checkpoint_search_on_reproduction_failure': True,
    'primary': 'success_count_and_rate_per_task_seed_arm_and_paired_outcome_table',
    'paired_keys': ['task_id', 'training_seed', 'initial_state_index'],
    'consistent_zero_improvement': 'strictly_more_successes_in_all_three_seeds_per_task',
    'paired_counts': ['both_success', 'zero_only_success', 'five_only_success', 'both_failure'],
    'statistical_significance_test': False, 'automatic_default_or_training_change': False,
    'same_weights_and_native_horizon_and_policy_action_limit': True,
    'initial_public_state_offsets_are_not_expert_errors': True,
    'stabilization_changes_physics_and_fixed_zero_actions_not_just_model_input': True,
    'diagnostic_success_is_not_qualification_or_unseen_generalization': True}
PINS = {'baseline': '03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce',
    'source_analysis': 'a2d3df183cb2f830fdcfdd7a1621ac45792920b7ce18a56f8e1ed797b5229115',
    'source_audit': 'f183a6ae3417387e3c3508b5c4bc946d5ece58f2dc99853d7c99764601ec62bf',
    'policy_config': 'f5e57c895e9249ef7a3361ebee2119bf06c6cbb48da01c31980b1eb0af8feb1b',
    'transfer': '6cdc8c3eeaa90886ad0e15f0f4ebbe260aa972b6e69b1f122e95f65ab63dd7d6',
    'policy_code': 'a3098a68b5ceab21495207856c9d94fd4a6d26c1668737625ecb7b5d46977735',
    'adapter_code': '8dfa22a1e19b049d78c8e6ae05494291242cbde8a21194e0fbb321662d56512c',
    'builder_code': '767c738e7ab98501f21439f190fd8fc80c98a8c4ece6716476ddc4fa7e0b9557',
    'causal_code': '780e26d1a50d452609fb24b2a7d4740351f4f044dadc03a286e8523bfbe22e22',
    'offline_helpers': '9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4'}
PREFIX_SHA = {4: '22f103ab49c6f3be149e78cbda2075b4499fdbc19b15043497f3fdcadef24708',
              7: 'c70b3fd4769c65750584c42a6161a6bd45fb3bb5673a32f1e20691687c50ba96'}


def load_protocol(path):
    cfg = core.load_json(path)
    require(cfg.get('check_type') == 'libero_task4_task7_stabilization_closed_loop_diagnostic' and
            cfg.get('formal_result') is False and cfg.get('protocol_status') == 'frozen_paired_candidate_only', 'closed-loop protocol not frozen diagnostic')
    require(cfg.get('scope') == SCOPE and cfg.get('analysis') == ANALYSIS, 'closed-loop frozen scope or analysis differs')
    require(cfg['runner_sha256'] == sha(Path(__file__)), 'closed-loop runner hash differs')
    require(set(cfg['inputs']) == set(PINS), 'closed-loop input roles differ')
    paths = {}
    for name, pin in PINS.items():
        row = cfg['inputs'][name]; p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(row['sha256'] == pin and sha(p) == pin, f'closed-loop input pin differs: {name}'); paths[name] = p
    transfer = core.load_json(paths['transfer'])
    for task in TASKS:
        indices = original.validate_online_state_scope(transfer, task, 'candidate_nonregression', list(range(20)))
        require(indices == tuple(range(20)), 'candidate initial-state role differs')
    return cfg, paths, transfer


def decode_utf8_latin1_prefix(stream, byte_count):
    """Read only the first byte_count Latin-1 characters of a UTF-8 payload."""
    require(type(byte_count) is int and byte_count > 0, 'invalid candidate byte count')
    decoded = bytearray()
    encoded_count = 0
    while len(decoded) < byte_count:
        first = stream.read(1); require(len(first) == 1, 'truncated candidate prefix')
        value = first[0]; encoded_count += 1
        if value >= 128:
            require(value in (0xc2, 0xc3), 'non Latin-1 NumPy payload')
            second = stream.read(1); require(len(second) == 1 and 128 <= second[0] < 192, 'invalid UTF-8 continuation')
            value = ((value & 31) << 6) | (second[0] & 63); encoded_count += 1
        decoded.append(value)
    return bytes(decoded), encoded_count


def load_candidate_prefix(path, row, prefix_sha, indices=tuple(range(20))):
    require(indices == tuple(range(20)) and all(type(i) is int for i in indices), 'candidate prefix must be exactly rows0to19')
    path = Path(path)
    require(path.stat().st_size == row['size_bytes'] and sha(path) == row['sha256'], 'initial container hash differs')
    with zipfile.ZipFile(path) as z:
        item = z.getinfo('archive/data.pkl')
        require(item.compress_type == zipfile.ZIP_STORED and not item.flag_bits & 1, 'initial pickle must be stored and unencrypted')
        local_offset, pickle_size = item.header_offset, item.file_size
    with path.open('rb', buffering=0) as f:
        f.seek(local_offset); header = f.read(30)
        require(header[:4] == b'PK\x03\x04', 'invalid local zip header')
        name_len, extra_len = struct.unpack('<HH', header[26:30])
        name = f.read(name_len); require(name == b'archive/data.pkl', 'initial pickle entry differs')
        f.seek(extra_len, 1)
        prefix = f.read(193)
        require(hashlib.sha256(prefix).hexdigest() == prefix_sha and prefix[188] == 88, 'pinned NumPy float64 C-order header differs')
        encoded_size = struct.unpack('<I', prefix[189:193])[0]
        require(36800 <= encoded_size <= 73600 and encoded_size + 193 <= pickle_size, 'invalid encoded ndarray size')
        raw, encoded_read = decode_utf8_latin1_prefix(f, 20 * 92 * 8)
    states = np.frombuffer(raw, dtype='<f8').reshape(20, 92).copy()
    require(np.isfinite(states).all(), 'candidate prefix contains nonfinite values')
    return states, {'path': str(path), 'container_sha256': row['sha256'], 'checksum_reads_whole_container_bytes': True,
        'pickle_header_sha256': prefix_sha, 'candidate_rows_numerically_decoded': list(range(20)),
        'decoded_candidate_bytes': len(raw), 'utf8_payload_bytes_read': encoded_read,
        'encoded_payload_bytes_total': encoded_size, 'qualification_or_reserved_rows_decoded': 0,
        'used_torch_or_pickle_deserialization': False,
        'candidate_vector_sha256_by_index': {str(i): core._digest_array(s) for i, s in enumerate(states)}}


def exact_baseline(episode, reference):
    fields = ('reset_observation_sha256', 'success', 'steps', 'terminal_reason', 'stepwise_public_state_trace')
    require(all(episode[k] == reference[k] for k in fields),
            f'five-step baseline reproduction mismatch: task={episode["task_id"]}, state={episode["initial_state_index"]}')


def archive_episode(episode, path, files, episodes, progress, reference=None):
    # A completed episode remains available even if its reproduction gate fails.
    with path.open('x') as f: json.dump(episode, f, separators=(',', ':')); f.write('\n')
    files.append({'path': str(path), 'sha256': sha(path)})
    episodes.append(episode); progress['episodes_completed'] += 1
    if reference is not None:
        exact_baseline(episode, reference); progress['five_episodes_exact'] += 1


@torch.no_grad()
def rollout(model, adapter, task, seed, state, arm, device, progress):
    progress['resets_attempted'] += 1
    obs, _ = adapter.reset(initial_state_index=state); progress['resets_completed'] += 1
    start_j = np.asarray(obs['joint_states'], np.float64).copy()
    start_g = np.asarray(obs['gripper_states'], np.float64).copy()
    joints, grippers, actions = [start_j.copy()], [start_g.copy()], []
    digest = original._observation_digest(obs)
    trace = [{'step': 0, 'joint_state': start_j.tolist(), 'joint_offset_from_start': [0.] * 7,
              'gripper_state': start_g.tolist(), 'gripper_offset_from_start': [0.] * 2}]
    reward_sum = 0.; terminated = truncated = False
    info = {'success': False, 'terminal_reason': None}
    while not (terminated or truncated):
        action, hidden = model.forward_step(*original.causal._online_tensors(obs, device), hidden=None)
        require(hidden is None, 'closed-loop Policy is not stateless CNN-only')
        values = action[0].cpu().numpy().astype(np.float32)
        obs, reward, terminated, truncated, info = adapter.step(values)
        progress['policy_actions'] += 1
        actions.append(values.copy()); reward_sum += reward
        j, g = np.asarray(obs['joint_states'], np.float64).copy(), np.asarray(obs['gripper_states'], np.float64).copy()
        joints.append(j); grippers.append(g)
        trace.append({'step': len(actions), 'action': values.tolist(), 'joint_state': j.tolist(),
            'joint_step_delta': (j - joints[-2]).tolist(), 'joint_offset_from_start': (j - start_j).tolist(),
            'joint_offset_l2_from_start': float(np.linalg.norm(j - start_j)), 'gripper_state': g.tolist(),
            'gripper_step_delta': (g - grippers[-2]).tolist(), 'gripper_offset_from_start': (g - start_g).tolist(),
            'gripper_offset_l2_from_start': float(np.linalg.norm(g - start_g))})
    require(len(actions) <= 300 and info['terminal_reason'] in ('success', 'time_limit'), 'unexpected terminal boundary')
    result = original.closed_loop._rollout_record(actions=actions, joint_states=joints, gripper_states=grippers, info=info, reward_sum=reward_sum)
    result.update({'task_id': task, 'training_seed': seed, 'initial_state_role': 'candidate_nonregression',
        'initial_state_index': state, 'stabilization_steps': arm, 'reset_observation_sha256': digest,
        'state_trace_reference': 'post_stabilization_reset_observation', 'state_trace_is_expert_error': False,
        'stepwise_public_state_trace': trace})
    return result


def paired_summary(episodes):
    indexed = core.unique_index(episodes, ('task_id', 'training_seed', 'initial_state_index', 'stabilization_steps'))
    require(set(indexed) == {(t, s, i, a) for t in TASKS for s in SEEDS for i in range(20) for a in (5, 0)}, 'paired episode matrix incomplete')
    tasks = {}
    for task in TASKS:
        seed_rows, totals = {}, dict.fromkeys(('both_success', 'zero_only_success', 'five_only_success', 'both_failure'), 0)
        for seed in SEEDS:
            counts = dict.fromkeys(totals, 0)
            for state in range(20):
                z, f = bool(indexed[(task, seed, state, 0)]['success']), bool(indexed[(task, seed, state, 5)]['success'])
                name = 'both_success' if z and f else 'zero_only_success' if z else 'five_only_success' if f else 'both_failure'
                counts[name] += 1
            zero, five = counts['both_success'] + counts['zero_only_success'], counts['both_success'] + counts['five_only_success']
            seed_rows[str(seed)] = {'episodes_per_arm': 20, 'zero_successes': zero, 'five_successes': five,
                'zero_success_rate': zero / 20, 'five_success_rate': five / 20, 'success_difference_zero_minus_five': zero - five,
                'paired_outcomes': counts}
            for name in totals: totals[name] += counts[name]
        zero = sum(r['zero_successes'] for r in seed_rows.values()); five = sum(r['five_successes'] for r in seed_rows.values())
        tasks[str(task)] = {'seed_results': seed_rows, 'episodes_per_arm': 60, 'zero_successes': zero, 'five_successes': five,
            'zero_success_rate': zero / 60, 'five_success_rate': five / 60, 'paired_outcomes': totals,
            'strict_zero_improvement_in_all_three_seeds': all(r['success_difference_zero_minus_five'] > 0 for r in seed_rows.values())}
    return tasks


def run(config_path, output, artifacts):
    require(not output.exists() and not artifacts.exists(), 'closed-loop output/artifacts already exist')
    cfg, paths, transfer = load_protocol(config_path)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '1' and os.environ.get('MUJOCO_EGL_DEVICE_ID') == '1' and
            os.environ.get('MUJOCO_GL') == 'egl', 'frozen inference/render device differs')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'single visible CUDA device required')
    torch.set_num_threads(1); torch.cuda.set_device(0); device = torch.device('cuda:0')
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats(device)
    parent, baseline = core.load_json(paths['source_analysis']), core.load_json(paths['baseline'])
    policy = {**core.load_json(paths['policy_config'])['policy'], 'memory_mode': 'cnn_only'}
    artifacts.mkdir(parents=True, exist_ok=False)
    started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    progress = dict.fromkeys(('adapters_built', 'resets_attempted', 'resets_completed', 'policy_actions',
                             'native_steps', 'fixed_zero_stabilization_steps', 'episodes_completed', 'five_episodes_exact'), 0)
    episodes, files, initial_access, failure = [], [], {}, None
    with (artifacts / 'events.jsonl').open('x') as log:
        def event(kind, **values):
            log.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}, sort_keys=True) + '\n'); log.flush()
        try:
            states = {}
            for task in TASKS:
                row = next(r for r in transfer['initial_state_files'] if r['task_id'] == task)
                states[task], initial_access[str(task)] = load_candidate_prefix(Path(SCOPE['native_initial_root']) / row['path'], row, PREFIX_SHA[task])
            for arm in (5, 0):
                if arm == 0: require(progress['five_episodes_exact'] == 120, 'all five baseline episodes must reproduce before zero arm')
                for task in TASKS:
                    for seed in SEEDS:
                        original.causal._set_seed(seed)
                        checkpoint = parent['task_results'][str(task)]['models'][str(seed)]['checkpoint']
                        reference = baseline['task_results'][str(task)]['seed_results'][str(seed)]
                        require(checkpoint['optimizer_updates'] == 500 and checkpoint['arm'] == 'source_baseline' and
                                checkpoint['weight_sha256'] == reference['training']['final_model_sha256'], 'fixed baseline500 checkpoint differs')
                        model = core.load_policy(checkpoint, policy).to(device).eval(); before = core._digest_weights(model.state_dict())
                        adapter = _build_injected_adapter(suite_name='libero_spatial', task_id=task, seed=20261022,
                            image_size=128, max_episode_steps=300, stabilization_steps=5, initial_states=states[task].copy())
                        progress['adapters_built'] += 1; adapter.stabilization_steps = arm
                        environment, in_reset = adapter.environment, [False]
                        native_step = environment.step
                        def counted(action):
                            if in_reset[0]:
                                require(np.array_equal(action, np.zeros(7, np.float32)), 'nonzero stabilization command')
                            result = native_step(action); progress['native_steps'] += 1
                            if in_reset[0]: progress['fixed_zero_stabilization_steps'] += 1
                            return result
                        environment.step = counted
                        real_reset = adapter.reset
                        def resetting(*args, **kwargs):
                            in_reset[0] = True
                            try: return real_reset(*args, **kwargs)
                            finally: in_reset[0] = False
                        adapter.reset = resetting
                        try:
                            for state in range(20):
                                ep = rollout(model, adapter, task, seed, state, arm, device, progress)
                                episode_path = artifacts / f'task{task}-seed{seed}-stabilization{arm}-state{state:02d}.json'
                                archive_episode(ep, episode_path, files, episodes, progress,
                                    reference['candidate_nonregression']['episodes'][state] if arm == 5 else None)
                                event('episode_complete', task_id=task, seed=seed, stabilization_steps=arm, state_index=state,
                                      success=ep['success'], policy_steps=ep['steps'], exact_original_five=arm == 5)
                                print(json.dumps({'task': task, 'seed': seed, 'arm': arm, 'state': state, 'success': ep['success'], 'episodes': len(episodes)}), flush=True)
                        finally:
                            environment.step = native_step; adapter.close()
                        require(before == core._digest_weights(model.state_dict()) == checkpoint['weight_sha256'] and
                                all(p.grad is None for p in model.parameters()), 'rollout mutated checkpoint')
                        del model; torch.cuda.empty_cache()
            require(progress['episodes_completed'] == progress['resets_completed'] == 240 and progress['adapters_built'] == 12 and
                    progress['fixed_zero_stabilization_steps'] == 600 and progress['policy_actions'] <= 72000 and
                    progress['native_steps'] == progress['policy_actions'] + 600, 'closed-loop actual budget differs')
        except Exception as exc:
            failure = {'type': type(exc).__name__, 'message': str(exc)}; event('technical_abort', **failure)
    torch.cuda.synchronize()
    result = {'check_type': 'libero_task4_task7_stabilization_closed_loop_diagnostic', 'formal_result': False,
        'status': 'failed' if failure else 'complete', 'failure': failure, 'scope': SCOPE, 'analysis': ANALYSIS,
        'initial_state_access': initial_access, 'episodes': episodes, 'episode_files': files,
        'task_results': paired_summary(episodes) if not failure else {},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'inputs': cfg['inputs'], 'started_at_utc': utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'event_log': {'path': str(artifacts / 'events.jsonl'), 'sha256': sha(artifacts / 'events.jsonl')},
            'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'MUJOCO_GL', 'MUJOCO_EGL_DEVICE_ID', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'runtime': {'actual_calls': progress, 'device': str(device), 'device_name': torch.cuda.get_device_name(device),
            'python': sys.version.split()[0], 'torch': torch.__version__, 'cuda': torch.version.cuda,
            'peak_cuda_memory_allocated_bytes': int(torch.cuda.max_memory_allocated(device)),
            'elapsed_seconds': time.perf_counter() - started, 'new_optimizer_updates': 0, 'new_backward_calls': 0},
        'interpretation': {'diagnostic_only': True, 'no_checkpoint_or_default_setting_selected': True,
            'candidate_states_not_unseen': True, 'not_module_qualification': True}}
    with output.open('x') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    if failure: raise RuntimeError(f'closed-loop stopped: {failure}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args(); result = run(args.config.resolve(), args.output.resolve(), args.artifacts.resolve())
    print(json.dumps({'status': result['status'], 'task_results': result['task_results'], 'runtime': result['runtime']}))


if __name__ == '__main__': main()
