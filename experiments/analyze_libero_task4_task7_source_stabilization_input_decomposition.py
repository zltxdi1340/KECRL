"""Fixed baseline500 Policy responses to paired cached source 0/5 observations.

Only pinned JSON, public NPZ arrays and six existing weight files are opened.
No dataset, environment, optimizer or action target is imported or constructed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_evidence_decomposition as core

require, sha = core.require, core.sha256_file
TASKS, SEEDS = (4, 7), (20261022, 20261023, 20261024)
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-source-stabilization-input-decomposition-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-source-stabilization-input-decomposition-20261102-v1.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-source-stabilization-input-decomposition-artifacts-20261102-v1')
PINS = {
    'source_result': 'a2d3df183cb2f830fdcfdd7a1621ac45792920b7ce18a56f8e1ed797b5229115',
    'source_audit': 'f183a6ae3417387e3c3508b5c4bc946d5ece58f2dc99853d7c99764601ec62bf',
    'source_config': '385cd973794e6339042462227040e0a471a40fcdd572d8fa1e3fb092089882f6',
    'transfer': '6cdc8c3eeaa90886ad0e15f0f4ebbe260aa972b6e69b1f122e95f65ab63dd7d6',
    'policy_config': 'f5e57c895e9249ef7a3361ebee2119bf06c6cbb48da01c31980b1eb0af8feb1b',
    'policy_code': 'a3098a68b5ceab21495207856c9d94fd4a6d26c1668737625ecb7b5d46977735',
    'offline_helper': '9bf4ff6b3d18345be1b7d9e3b50d887a7ca78c1f34e9c039c10ced283472a4a4',
}
SCOPE = {
    'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'checkpoint_updates': 500,
    'policy_arm': 'source_baseline', 'memory_mode': 'cnn_only', 'source_role': 'meta_support',
    'source_demo_indices': list(range(30)), 'cached_public_observation_rows': 120,
    'source_demo_pairs': 60, 'checkpoint_loads': 6, 'cached_reference_prediction_rows': 360,
    'main_forward_calls': 48, 'repeat_forward_calls': 12, 'main_model_rows': 1440, 'repeat_model_rows': 360,
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_policy_control_steps': 0,
    'new_environment_resets_or_steps': 0, 'new_hdf5_or_initial_state_reads': 0,
    'new_candidate_observation_or_feature_rows': 0, 'new_expert_target_rows': 0,
    'query_dataset_read': False, 'private_simulator_state_read': False, 'official_state_files_read': False,
    'qualification_accessed': False, 'reserved_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False, 'checkpoint_selected': False,
    'device': 'cpu', 'torch_threads': 1, 'analysis_rng_seed': 20261022,
    'spt_updated': False, 'knowledge_updated': False, 'module_registered': False,
}
SWAPS = {'agentview': 1, 'eye_in_hand': 2, 'rgb': 3, 'proprio': 4, 'full': 7}
ANALYSIS = {
    'mask_bits': {'agentview_rgb': 1, 'eye_in_hand_rgb': 2, 'joint_and_gripper_together': 4},
    'mask0': 'all_zero_stabilization_cached_observation', 'mask7': 'all_five_stabilization_cached_observation',
    'masks': list(range(8)), 'references': {'zero': 0, 'five': 7}, 'swaps': SWAPS,
    'swap_delta': 'f(anchor XOR swap)-f(anchor); signed seven-dimensional vector per source demo',
    'camera_interaction': 'f(anchor XOR3)-f(anchor XOR1)-f(anchor XOR2)+f(anchor)',
    'rgb_proprio_interaction': 'f(anchor XOR7)-f(anchor XOR3)-f(anchor XOR4)+f(anchor)',
    'arithmetic': 'cast float32 predictions to float64 before signed differences and statistics',
    'metrics': ['action_l2', 'translation_l2', 'rotation_l2', 'gripper_abs'],
    'summary': 'n/mean/median/minimum/maximum; signed means and all per-demo vectors retained',
    'gripper_sign_switch': 'count sign(output_gripper)!=sign(anchor_gripper); zero is neutral',
    'primary': 'RGB_vs_proprio_mean_action_L2_separately_for_both_reference_anchors_and_all_three_seeds',
    'secondary': 'eye_in_hand_vs_agentview_same_comparison; no_camera_or_seed_selection',
    'consistent_larger': 'strictly larger per-seed mean for all three seeds in BOTH references',
    'all_masks_run_in_frozen_order': [0, 7, 'repeat0', 'repeat7', 1, 2, 3, 4, 5, 6],
    'natural_predictions_and_repeats': 'both mask0 and mask7 must exactly match parent and repeat batch30',
    'identity_residual_max_abs_le': 1e-12,
    'mixed_inputs_may_be_off_natural_manifold': True, 'norms_are_additive_contributions': False,
    'new_observations_have_no_expert_targets': True, 'source_demos_not_six_closed_loop_outcome_pairs': True,
    'no_significance_training_promotion_or_closed_loop_causal_claim': True,
}


def verify_file(record):
    path = Path(record['path'])
    path = path if path.is_absolute() else ROOT / path
    require(path.is_file() and sha(path) == record['sha256'], f'input file hash differs: {path}')
    return path


def load_protocol(path):
    cfg = core.load_json(path)
    require(cfg.get('check_type') == 'libero_source_stabilization_input_decomposition' and
            cfg.get('formal_result') is False and cfg.get('scope') == SCOPE and cfg.get('analysis') == ANALYSIS,
            'frozen input decomposition scope or definitions differ')
    require(cfg['runner_sha256'] == sha(Path(__file__)), 'input decomposition runner differs')
    require(set(cfg['inputs']) == set(PINS), 'input decomposition roles differ')
    paths = {}
    for name, pin in PINS.items():
        require(cfg['inputs'][name]['sha256'] == pin, f'input role pin differs: {name}')
        paths[name] = verify_file(cfg['inputs'][name])
    parent, audit, source_config, transfer = [core.load_json(paths[n]) for n in ('source_result', 'source_audit', 'source_config', 'transfer')]
    require(parent['status'] == audit['status'] == 'complete' and parent['formal_result'] is False and
            audit['result_sha256'] == PINS['source_result'] and parent['scope'] == source_config['scope'] and
            parent['provenance']['config_sha256'] == PINS['source_config'], 'source completion or provenance differs')
    require(parent['scope']['source_role'] == 'meta_support' and parent['scope']['source_demo_indices'] == list(range(30)) and
            parent['scope']['checkpoint_updates'] == 500, 'source data role or weight budget differs')
    require(all(t in transfer['task_roles']['source_spi'] for t in TASKS) and
            transfer['demonstration_indices_by_task_role']['source_spi']['meta_support'] == list(range(30)),
            'transfer source/support roles differ')
    for name in ('qualification_accessed', 'reserved_accessed', 'validation_accessed', 'heldout_accessed', 'final_evaluation_accessed'):
        require(parent['scope'][name] is False, f'source sealed boundary differs: {name}')
    public = parent['source_capture_parent']['observation_archives']
    expected_public = [{'task_id': t, 'stabilization_steps': a,
        'archive': next(r for r in public if Path(r['path']).name == f'task{t}-stabilization{a}-public-observations.npz')}
        for t in TASKS for a in (0, 5)]
    require(len(public) == 4 and cfg['public_caches'] == expected_public, 'public source cache matrix differs')
    expected_models = [{'task_id': t, 'training_seed': s, 'checkpoint': parent['task_results'][str(t)]['models'][str(s)]['checkpoint'],
        'reference': parent['task_results'][str(t)]['models'][str(s)]['archive']} for t in TASKS for s in SEEDS]
    require(cfg['model_inputs'] == expected_models, 'fixed Policy/reference matrix differs')
    for row in cfg['public_caches']:
        verify_file(row['archive'])
    for row in cfg['model_inputs']:
        cp = row['checkpoint']
        require(cp['task_id'] == row['task_id'] and cp['training_seed'] == row['training_seed'] and
                cp['optimizer_updates'] == 500 and cp['arm'] == 'source_baseline', 'checkpoint task/seed/arm/update differs')
        verify_file({'path': cp['path'], 'sha256': cp['file_sha256']})
        verify_file(row['reference'])
    return cfg, paths


def validate_public(arrays):
    require(set(arrays) == set(core.OBS_KEYS) | {'source_demo_indices'}, 'public cache interface differs')
    require(arrays['source_demo_indices'].dtype == np.int64 and np.array_equal(arrays['source_demo_indices'], np.arange(30)),
            'public cache support demonstration order differs')
    for name in core.OBS_KEYS:
        shape = (30, 128, 128, 3) if name in core.OBS_KEYS[:2] else (30, 7 if name == 'joint_states' else 2)
        dtype = np.uint8 if name in core.OBS_KEYS[:2] else np.float32
        require(arrays[name].shape == shape and arrays[name].dtype == dtype and np.isfinite(arrays[name]).all(),
                f'public observation shape/dtype/value differs: {name}')


def digest_batch(arrays):
    return {k: core._digest_array(arrays[k]) for k in core.OBS_KEYS}


@torch.no_grad()
def forward_batch(model, observations):
    args = [torch.from_numpy(observations[k]).permute(0, 3, 1, 2) for k in core.OBS_KEYS[:2]]
    args += [torch.from_numpy(observations[k]) for k in core.OBS_KEYS[2:]]
    predictions, hidden = model.forward_step(*args, hidden=None)
    require(hidden is None and model.config.memory_mode == 'cnn_only', 'requires stateless CNN-only Policy')
    values = predictions.cpu().numpy().copy()
    require(values.shape == (30, 7) and values.dtype == np.float32 and np.isfinite(values).all() and
            np.max(np.abs(values)) <= 1, 'invalid fixed Policy prediction')
    return values


def collect_outputs(model, zero, five, references, progress, event, saved, forward=forward_batch):
    def read(mask, repeated=False):
        kind = 'repeat' if repeated else 'main'
        progress[f'{kind}_forward_attempts'] += 1
        event('forward_attempt', mask=mask, readout_kind=kind, rows=30)
        values = forward(model, core.mix_observations(zero, five, mask))
        saved[f'repeat{mask}' if repeated else f'mask{mask}'] = values
        progress[f'{kind}_forward_calls'] += 1
        progress[f'{kind}_model_rows'] += 30
        event('forward_complete', mask=mask, readout_kind=kind, rows=30)
        return values
    for mask in (0, 7):
        require(np.array_equal(read(mask), references[mask]), f'natural endpoint parity failed: mask{mask}')
        progress['natural_endpoint_parity_exact'] += 1
    for mask in (0, 7):
        require(np.array_equal(read(mask, repeated=True), saved[f'mask{mask}']), f'repeated endpoint differs: mask{mask}')
        progress['repeated_endpoint_parity_exact'] += 1
    for mask in range(1, 7):
        read(mask)
    return np.stack([saved[f'mask{m}'] for m in range(8)])


def summary(values):
    v = np.asarray(values, dtype=np.float64)
    require(v.shape == (30,) and np.isfinite(v).all(), 'summary rows differ')
    return {'n': 30, 'mean': float(v.mean()), 'median': float(np.median(v)), 'minimum': float(v.min()), 'maximum': float(v.max())}


def delta_metrics(values):
    v = np.asarray(values, dtype=np.float64)
    require(v.shape == (30, 7) and np.isfinite(v).all(), 'signed delta rows differ')
    metrics = {'action_l2': np.linalg.norm(v, axis=1), 'translation_l2': np.linalg.norm(v[:, :3], axis=1),
               'rotation_l2': np.linalg.norm(v[:, 3:6], axis=1), 'gripper_abs': np.abs(v[:, 6])}
    return {'metrics': {k: summary(a) for k, a in metrics.items()}, 'mean_signed_action_delta': v.mean(axis=0).tolist(),
            'per_demo_signed_action_delta': v.tolist()}


def decompose(predictions):
    require(predictions.shape == (8, 30, 7) and predictions.dtype == np.float32 and np.isfinite(predictions).all(),
            'eight-cell prediction tensor differs')
    p = predictions.astype(np.float64)
    result = {}
    for name, anchor in ANALYSIS['references'].items():
        swaps = {}
        for label, mask in SWAPS.items():
            swaps[label] = {**delta_metrics(p[anchor ^ mask] - p[anchor]),
                'gripper_sign_changes': int(np.sum(np.sign(p[anchor ^ mask, :, 6]) != np.sign(p[anchor, :, 6])))}
        camera = p[anchor ^ 3] - p[anchor ^ 1] - p[anchor ^ 2] + p[anchor]
        rgbp = p[anchor ^ 7] - p[anchor ^ 3] - p[anchor ^ 4] + p[anchor]
        residuals = {'camera': float(np.max(np.abs((p[anchor ^ 3] - p[anchor]) -
            ((p[anchor ^ 1] - p[anchor]) + (p[anchor ^ 2] - p[anchor]) + camera)))),
            'rgb_proprio': float(np.max(np.abs((p[anchor ^ 7] - p[anchor]) -
            ((p[anchor ^ 3] - p[anchor]) + (p[anchor ^ 4] - p[anchor]) + rgbp))))}
        require(max(residuals.values()) <= ANALYSIS['identity_residual_max_abs_le'], 'signed decomposition identity failed')
        comparisons = {}
        for left, right in (('rgb', 'proprio'), ('eye_in_hand', 'agentview')):
            difference = np.linalg.norm(p[anchor ^ SWAPS[left]] - p[anchor], axis=1) - np.linalg.norm(p[anchor ^ SWAPS[right]] - p[anchor], axis=1)
            comparisons[f'{left}_minus_{right}'] = {**summary(difference), 'left_larger_rows': int(np.sum(difference > 0))}
        result[name] = {'swaps': swaps, 'signed_interactions': {'camera': delta_metrics(camera), 'rgb_proprio': delta_metrics(rgbp)},
                        'identity_residual_max_abs': residuals, 'comparisons': comparisons}
    return result


def decisions(models):
    require(set(models) == set(map(str, SEEDS)), 'three fixed seeds required')
    return {comparison: {'zero_reference_all_three_means_positive': all(models[str(s)]['decomposition']['zero']['comparisons'][comparison]['mean'] > 0 for s in SEEDS),
        'five_reference_all_three_means_positive': all(models[str(s)]['decomposition']['five']['comparisons'][comparison]['mean'] > 0 for s in SEEDS),
        'both_references_all_three_means_positive': all(models[str(s)]['decomposition'][a]['comparisons'][comparison]['mean'] > 0 for s in SEEDS for a in ('zero', 'five'))}
        for comparison in ('rgb_minus_proprio', 'eye_in_hand_minus_agentview')}


def save_npz(path, arrays):
    with path.open('xb') as handle:
        np.savez_compressed(handle, **arrays)
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def counters():
    return dict.fromkeys(('checkpoint_loads', 'main_forward_attempts', 'main_forward_calls', 'repeat_forward_attempts',
        'repeat_forward_calls', 'main_model_rows', 'repeat_model_rows', 'natural_endpoint_parity_exact',
        'repeated_endpoint_parity_exact', 'models_completed'), 0)


def run(config_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), 'decomposition output/archive already exists')
    cfg, paths = load_protocol(config_path)
    started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    archive_dir.mkdir(parents=True, exist_ok=False)
    progress, tasks, files, failure = counters(), {}, [], None
    event_path = archive_dir / 'events.jsonl'
    def event(kind, **values):
        with event_path.open('a') as f:
            f.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}, sort_keys=True) + '\n')
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    np.random.seed(20261022)
    event('start', config_sha256=sha(config_path))
    try:
        public = {}
        for row in cfg['public_caches']:
            with np.load(verify_file(row['archive']), allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            validate_public(arrays)
            public[row['task_id'], row['stabilization_steps']] = {k: arrays[k] for k in core.OBS_KEYS}
        input_digests = {key: digest_batch(v) for key, v in public.items()}
        policy = {**core.load_json(paths['policy_config'])['policy'], 'memory_mode': 'cnn_only'}
        for task in TASKS:
            models = {}
            tasks[str(task)] = {'models': models}
            for seed in SEEDS:
                row = next(r for r in cfg['model_inputs'] if (r['task_id'], r['training_seed']) == (task, seed))
                saved = {'source_demo_indices': np.arange(30)}
                model_record = {'checkpoint': row['checkpoint'], 'reference': row['reference'], 'status': 'started'}
                models[str(seed)] = model_record
                try:
                    with np.load(verify_file(row['reference']), allow_pickle=False) as old:
                        require(np.array_equal(old['source_demo_indices'], np.arange(30)), 'reference support order differs')
                        references = {m: old[f'stabilization{a}_predictions'] for m, a in ((0, 0), (7, 5))}
                    require(all(v.shape == (30, 7) and v.dtype == np.float32 and np.isfinite(v).all() for v in references.values()), 'reference predictions differ')
                    model = core.load_policy(row['checkpoint'], policy)
                    progress['checkpoint_loads'] += 1
                    before = core._digest_weights(model.state_dict())
                    def model_event(kind, **values):
                        event(kind, task_id=task, training_seed=seed, **values)
                    values = collect_outputs(model, public[task, 0], public[task, 5], references, progress, model_event, saved)
                    require(before == core._digest_weights(model.state_dict()) == row['checkpoint']['weight_sha256'] and
                            all(p.grad is None for p in model.parameters()), 'fixed Policy weights or gradients mutated')
                    require(all(digest_batch(public[task, a]) == input_digests[task, a] for a in (0, 5)), 'public inputs mutated')
                    model_record.update(status='complete', natural_endpoint_parity_exact=True, repeat_endpoint_parity_exact=True,
                        weight_sha256_unchanged=before, decomposition=decompose(values))
                    progress['models_completed'] += 1
                    print(json.dumps({'task': task, 'training_seed_complete': seed}), flush=True)
                    del model
                finally:
                    if len(saved) > 1:
                        record = save_npz(archive_dir / f'task{task}-seed{seed}-predictions.npz', saved)
                        model_record['archive'] = record
                        files.append(record)
            tasks[str(task)]['descriptive_comparisons'] = decisions(models)
        expected = {'checkpoint_loads': 6, 'main_forward_attempts': 48, 'main_forward_calls': 48,
            'repeat_forward_attempts': 12, 'repeat_forward_calls': 12, 'main_model_rows': 1440, 'repeat_model_rows': 360,
            'natural_endpoint_parity_exact': 12, 'repeated_endpoint_parity_exact': 12, 'models_completed': 6}
        require(progress == expected, f'fixed forward budget differs: {progress}')
        for row in cfg['public_caches']:
            verify_file(row['archive'])
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        event('failure', **failure)
    event('finished', status='failed' if failure else 'complete', actual_calls=progress)
    result = {'check_type': 'libero_source_stabilization_input_decomposition', 'status': 'failed' if failure else 'complete',
        'failure': failure, 'formal_result': False, 'scope': SCOPE, 'analysis': ANALYSIS, 'task_results': tasks, 'archives': files,
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'inputs': cfg['inputs'], 'public_caches': cfg['public_caches'], 'model_inputs': cfg['model_inputs'],
            'event_log': {'path': str(event_path), 'sha256': sha(event_path)}, 'started_at_utc': utc,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'runtime': {'device': 'cpu', 'python': platform.python_version(), 'platform': platform.platform(),
            'torch': torch.__version__, 'numpy': np.__version__, 'torch_threads': torch.get_num_threads(),
            'environment': {k: os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')},
            'mkldnn_enabled': torch.backends.mkldnn.enabled, 'actual_calls': progress, 'elapsed_seconds': time.perf_counter() - started},
        'interpretation': {'diagnostic_only': True, 'no_expert_target_scored': True, 'no_new_success_rate': True,
            'no_training_or_rollout_promoted': True, 'norms_not_additive_contributions': True,
            'mixed_inputs_not_natural_rollouts': True, 'source_demos_not_six_closed_loop_pairs': True}}
    with output.open('x') as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write('\n')
    if failure:
        raise RuntimeError(f'readout failed; result and partial outputs preserved: {failure}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve(), args.archive_dir.resolve())
    print(json.dumps({'status': result['status'], 'runtime': result['runtime']}))


if __name__ == '__main__':
    main()
