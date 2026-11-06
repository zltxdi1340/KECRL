"""Paired source-support reset observations at stabilization 0 and 5.

Only authorized demonstration initial vectors are used for reconstruction.
No Policy action is executed, no expert action target is scored, and cached
candidate features are reused without new candidate resets or CNN evaluation.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_cached_input_coverage as coverage
from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
from src.environments.libero_dataset import RoleRestrictedLiberoSequenceDataset

condition, core = coverage.condition, coverage.core
require, sha = coverage.require, coverage.sha
TASKS, SEEDS, SPACES = coverage.TASKS, coverage.SEEDS, coverage.SPACES
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-source-stabilization-control-diagnostic_v2.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-source-stabilization-control-20261102-v2.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-source-stabilization-control-artifacts-20261102-v2')
SCOPE = {
    'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'checkpoint_updates': 500,
    'policy_arm': 'source_baseline', 'source_role': 'meta_support', 'source_demo_indices': list(range(30)),
    'bank_demo_indices': list(range(20)), 'reference_demo_indices': list(range(20, 30)),
    'stabilization_steps': [0, 5], 'environment_seed': 20261022,
    'dataset_root': '/home/zl202621/third_party/LIBERO/datasets', 'image_size': 128,
    'native_horizon_both_arms': 300, 'wrapper_max_episode_steps_both_arms': 300,
    'builder_stabilization_steps_both_arms': 0,
    'adapter_lifecycle': 'one_per_task_arm; source_demo0_to29_in_order; all_zero_arms_verified_before_five_arms',
    'source_initial_vectors_read': 60, 'source_reset_budget': 120, 'adapter_budget': 4,
    'fixed_zero_stabilization_action_budget': 300, 'policy_control_action_budget': 0,
    'main_forward_calls': 12, 'repeat_forward_calls': 6, 'main_model_rows': 360, 'repeat_model_rows': 180,
    'candidate_features_reused': 2400, 'new_candidate_resets_or_forwards': 0,
    'unique_bank_scalers': 48, 'bank_evaluation_groups': 96,
    'candidate_knn_rows': 38400, 'reference_knn_rows': 960,
    'device': 'cpu', 'torch_threads': 1, 'analysis_seed': 20261022,
    'new_optimizer_updates': 0, 'new_backward_calls': 0,
    'source_action_or_observation_payload_rows_read': 0, 'query_dataset_read': False,
    'source_initial_vectors_used_only_for_reset': True, 'initial_vector_values_recorded': False,
    'live_private_simulator_state_read': False, 'official_state_files_read': False,
    'qualification_accessed': False, 'reserved_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False,
    'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False,
}
ANALYSIS = {
    'spaces': SPACES, 'neighbors': 5, 'bank_rows': 20, 'reference_rows': 10,
    'primary_space': 'fusion', 'primary_scaling': 'fixed_zero_scale',
    'scaling_modes': ['fixed_zero_scale', 'per_arm_scale'],
    'fixed_zero_scale': 'both_arms_use_only_zero_bank_mean_and_population_std; same_source_demo_ids',
    'per_arm_scale': 'secondary; each_arm_fits_only_its_own_bank; zero_scale_reused',
    'std_le1e-6_replaced_by1': True, 'distance': 'mean_of_five_squared_euclidean_distances',
    'neighbor_ties': 'stable_original_bank_demo_order', 'reference95': 'per_arm_reference_default_linear_quantile',
    'preaction_horizons': [0, 4, 9, 19],
    'consistent_distance_reduction': 'candidate_index0_distance_mean_five_over_zero_lt1_in_all_three_seeds',
    'consistent_exceedance_reduction': 'candidate_index0_exceedance_fraction_five_minus_zero_lt0_in_all_three_seeds',
    'both_required_for_descriptive_alignment_signal': True,
    'zero_public_observations_must_exactly_match_old_cache': True,
    'zero_features_and_predictions_must_exactly_match_old_cpu_cache': True,
    'zero_coverage_horizons_exact_to_old_runtime_t0': True,
    'repeat_zero_batch_exact': True,
    'no_expert_target_for_stabilized_observation': True, 'no_significance_or_ood_validity_claim': True,
    'no_training_or_rollout_promotion': True, 'no_closed_loop_failure_cause_claim': True,
}
INPUT_PINS = {
    'coverage_result': '0009843ce72f6fa802da4c9538e7ede206074b3201f5cd32d614861c8d056488',
    'coverage_audit': '373ef8b6576ea611595a490692cbd3447789d3edb12a7c3513818cdc1c1aabe1',
    'coverage_code': 'b09d1ca768f98aaf56690b3345fd02dbd73caf79b7f8e8b44e6fcc664aa96109',
    'condition_code': '58229e2684ad76bd9111b0c6099542b840380d12212d40eaa1c69943826929c7',
    'expert_code': 'bd44981057bbb09ecc485379ca4ad13563b3f770d892dbd9c73ce3e1af557425',
    'offline_helpers': condition.PINS['offline_helpers'], 'policy_code': condition.PINS['policy_code'],
    'policy_config': condition.PINS['policy_config'], 'dataset_code': condition.PINS['dataset_access_code'],
    'manifest': condition.PINS['dataset_manifest'], 'transfer': condition.PINS['transfer_config'],
    'adapter_code': '8dfa22a1e19b049d78c8e6ae05494291242cbde8a21194e0fbb321662d56512c',
    'adapter_builder_code': '767c738e7ab98501f21439f190fd8fc80c98a8c4ece6716476ddc4fa7e0b9557',
    'public_source_zero4': coverage.PINS['public_support_t0_4'],
    'public_source_zero7': coverage.PINS['public_support_t0_7'],
}


def load_protocol(path):
    cfg = core.load_json(path)
    require(cfg.get('schema_version') == 1 and cfg.get('formal_result') is False and
            cfg.get('check_type') == 'libero_task4_task7_source_stabilization_control_diagnostic' and
            cfg.get('protocol_status') == 'frozen_source_support_paired_reset', 'stabilization protocol is not frozen diagnostic')
    require(cfg.get('scope') == SCOPE and cfg.get('analysis') == ANALYSIS, 'stabilization frozen scope or analysis differs')
    require(cfg.get('runner_sha256') == sha(Path(__file__)), 'stabilization runner hash differs')
    require(set(cfg['inputs']) == set(INPUT_PINS), 'stabilization input roles differ')
    paths = {}
    for name, pin in INPUT_PINS.items():
        row = cfg['inputs'][name]; p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(row['sha256'] == pin and sha(p) == pin, f'stabilization input pin differs: {name}')
        paths[name] = p.resolve()
    return cfg, paths


def authorize_support(transfer, manifest, task, role='meta_support'):
    require(type(task) is int and task in TASKS and role == 'meta_support', 'stabilization access outside source support')
    return condition.authorize_support(transfer, manifest, task, role)


def read_source_initials(transfer, manifest, task, root):
    access = authorize_support(transfer, manifest, task)
    dataset = RoleRestrictedLiberoSequenceDataset(
        transfer, manifest, root, task_id=task, demonstration_role='meta_support',
        sequence_length=1, stride=1, access_phase='development', verify_sha256=True)
    try:
        require(dataset.access == access, 'source reconstruction access differs')
        states = dataset.load_initial_states()
        record = dataset.access_record
    finally:
        dataset.close()
    require(list(states) == list(range(30)), 'source initial reconstruction matrix differs')
    return states, {**record, 'payload_read': 'authorized_demo_init_state_attributes_only',
                    'initial_vector_sha256_by_demo': {str(d): core._digest_array(s) for d, s in states.items()},
                    'initial_vector_dimensions_by_demo': {str(d): int(len(s)) for d, s in states.items()}}


def counters():
    return dict.fromkeys(('source_initial_vectors_read', 'adapters_built', 'resets_attempted', 'resets_completed',
        'native_zero_steps_attempted', 'native_zero_steps_completed', 'zero_public_cache_matches',
        'main_forward_calls', 'repeat_forward_calls', 'main_model_rows', 'repeat_model_rows',
        'unique_bank_scalers', 'bank_evaluation_groups', 'candidate_knn_rows', 'reference_knn_rows'), 0)


def validate_observations(values):
    require(set(values) == set(core.OBS_KEYS), 'unexpected public observation interface')
    for name, (shape, dtype) in condition.expert.POLICY_OBSERVATION_SPECS.items():
        array = values[name]
        require(array.shape == (30, *shape) and array.dtype == dtype and np.isfinite(array).all(),
                f'source public observation shape, dtype or value differs: {name}')


def capture_arm(task, stabilization, states, reference, progress, event, factory=_build_injected_adapter):
    require(task in TASKS and type(stabilization) is int and stabilization in (0, 5), 'invalid source arm')
    require(list(states) == list(range(30)), 'source reset order or indices differ')
    validate_observations(reference)
    before_states = {d: core._digest_array(s) for d, s in states.items()}
    adapter = factory(suite_name='libero_spatial', task_id=task, seed=20261022, image_size=128,
                      max_episode_steps=300, stabilization_steps=0, initial_states=np.stack(list(states.values())))
    progress['adapters_built'] += 1
    # Build both native environments with horizon=300, then change only the
    # reset's stabilization count. The wrapper limit remains 300 in both arms.
    adapter.stabilization_steps = stabilization
    environment = adapter.environment
    original_step = environment.step
    def counted_step(action):
        require(np.array_equal(np.asarray(action), np.zeros(7, dtype=np.float32)), 'nonzero action in stabilization-only run')
        progress['native_zero_steps_attempted'] += 1
        result = original_step(action)
        progress['native_zero_steps_completed'] += 1
        return result
    environment.step = counted_step
    parts = {name: [] for name in core.OBS_KEYS}
    try:
        for demo in range(30):
            progress['resets_attempted'] += 1
            observation, _ = adapter.reset(initial_state_index=demo)
            progress['resets_completed'] += 1
            require(set(observation) == set(core.OBS_KEYS), 'reset public interface differs')
            digest = core._digest_observation(observation)
            expected = core._digest_observation({name: reference[name][demo] for name in core.OBS_KEYS})
            exact = digest == expected
            event('source_reset', task_id=task, source_demo_index=demo, stabilization_steps=stabilization,
                  observation_sha256=digest, zero_cache_sha256=expected, zero_cache_exact_match=exact)
            if stabilization == 0:
                require(exact, f'zero stabilization public cache mismatch: task={task}, demo={demo}')
                progress['zero_public_cache_matches'] += 1
            for name in core.OBS_KEYS:
                parts[name].append(np.asarray(observation[name]).copy())
    finally:
        environment.step = original_step
        adapter.close()
    require(before_states == {d: core._digest_array(s) for d, s in states.items()}, 'source reset mutated initial vectors')
    values = {name: np.stack(rows) for name, rows in parts.items()}
    validate_observations(values)
    return values


def collect_arms(states, reference, archive_dir, progress, event, files, factory=_build_injected_adapter):
    captured = {task: {} for task in TASKS}
    for stabilization in (0, 5):
        if stabilization == 5:
            require(progress['zero_public_cache_matches'] == 60, 'five-step arm requires all 60 exact zero caches')
        for task in TASKS:
            values = capture_arm(task, stabilization, states[task], reference[task], progress, event, factory)
            path = archive_dir / f'task{task}-stabilization{stabilization}-public-observations.npz'
            record = save(path, {**values, 'source_demo_indices': np.arange(30)})
            files.append(record); captured[task][stabilization] = values
    return captured


def save(path, arrays):
    digest = condition.expert.save_archive(path, arrays)
    return {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}


def verify_archive(record):
    path = Path(record['path'])
    require(sha(path) == record['sha256'], 'feature archive hash differs')
    return path


def summary(values):
    values = np.asarray(values, dtype=np.float64)
    require(values.ndim == 1 and len(values) and np.isfinite(values).all(), 'invalid paired metric')
    return {'rows': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'minimum': float(values.min()), 'maximum': float(values.max())}


def observation_delta(zero, five):
    result = {}
    for name in core.OBS_KEYS:
        delta = five[name].astype(np.float64) - zero[name].astype(np.float64)
        if name.endswith('_rgb'):
            metric = np.abs(delta).mean(axis=(1, 2, 3)) / 255.
            label = 'normalized_pixel_mae'
        else:
            metric = np.linalg.norm(delta, axis=1); label = 'state_l2'
        result[name] = {label: summary(metric), 'changed_demos': int(np.any(delta.reshape(30, -1) != 0, axis=1).sum())}
    return result


def action_delta(zero, five):
    require(zero.shape == five.shape == (30, 7), 'paired action matrix differs')
    delta = five.astype(np.float64) - zero.astype(np.float64)
    return {'action_l2': summary(np.linalg.norm(delta, axis=1)),
            'translation_l2': summary(np.linalg.norm(delta[:, :3], axis=1)),
            'rotation_l2': summary(np.linalg.norm(delta[:, 3:6], axis=1)),
            'gripper_abs': summary(np.abs(delta[:, 6])),
            'gripper_sign_changed_demos': int((np.sign(zero[:, 6]) != np.sign(five[:, 6])).sum())}


def scaled_bank(features, baseline, mode):
    require(mode in ANALYSIS['scaling_modes'], 'unknown scaling mode')
    if mode == 'per_arm_scale':
        return coverage.fit_bank(features)
    require(np.isfinite(features).all() and features.shape == baseline['normalized'].shape, 'matched bank shape differs')
    return {'mean': baseline['mean'].copy(), 'scale': baseline['scale'].copy(),
            'normalized': (features.astype(np.float64) - baseline['mean']) / baseline['scale']}


def evaluate_banks(features, candidate, steps, progress, saved):
    require(np.array_equal(steps, np.tile(np.arange(20), 20)), 'candidate cached horizon order differs')
    banks = {mode: {} for mode in ANALYSIS['scaling_modes']}
    for space in SPACES:
        base = coverage.fit_bank(features[0][space][:20]); progress['unique_bank_scalers'] += 1
        arm5 = scaled_bank(features[5][space][:20], base, 'per_arm_scale'); progress['unique_bank_scalers'] += 1
        for mode in ANALYSIS['scaling_modes']:
            banks[mode][space] = {}
            for stabilization in (0, 5):
                fitted = base if stabilization == 0 else (arm5 if mode == 'per_arm_scale' else
                         scaled_bank(features[5][space][:20], base, mode))
                ref_ids, ref_dist = coverage.bank_distances(fitted, features[stabilization][space][20:30])
                ids, distance = coverage.bank_distances(fitted, candidate[space])
                ref_values, values = ref_dist.mean(axis=1), distance.mean(axis=1)
                threshold = float(np.quantile(ref_values, .95))
                prefix = f'{mode}__{space}__stabilization{stabilization}'
                saved.update({f'{prefix}__mean': fitted['mean'], f'{prefix}__scale': fitted['scale'],
                    f'{prefix}__candidate_indices5': ids, f'{prefix}__candidate_distances5': distance,
                    f'{prefix}__reference_indices5': ref_ids, f'{prefix}__reference_distances5': ref_dist})
                banks[mode][space][str(stabilization)] = {
                    'bank_rows': 20, 'reference_rows': 10,
                    'bank_mean_sha256': core._digest_array(fitted['mean']), 'bank_scale_sha256': core._digest_array(fitted['scale']),
                    'reference': coverage.distance_summary(ref_values, threshold),
                    'candidate_all20_steps': coverage.distance_summary(values, threshold),
                    'horizons': {str(h): coverage.distance_summary(values[steps == h], threshold) for h in (0, 4, 9, 19)}}
                progress['bank_evaluation_groups'] += 1
                progress['candidate_knn_rows'] += len(values); progress['reference_knn_rows'] += len(ref_values)
    return banks


def decisions(models):
    ratios, fractions = {}, {}
    for seed in SEEDS:
        rows = models[str(seed)]['banks']['fixed_zero_scale']['fusion']
        zero, five = rows['0']['horizons']['0'], rows['5']['horizons']['0']
        ratios[str(seed)] = five['mean'] / zero['mean'] if zero['mean'] > 0 else None
        fractions[str(seed)] = five['exceedance_fraction'] - zero['exceedance_fraction']
    distance = all(r is not None and r < 1 for r in ratios.values())
    exceedance = all(r < 0 for r in fractions.values())
    return {'initial_distance_mean_five_over_zero_by_seed': ratios,
            'initial_exceedance_fraction_five_minus_zero_by_seed': fractions,
            'consistent_distance_reduction': distance, 'consistent_exceedance_reduction': exceedance,
            'descriptive_alignment_signal': distance and exceedance,
            'source_to_candidate_domains_fully_matched': False, 'training_or_rollout_promoted': False,
            'closed_loop_failure_cause_established': False}


def run(config_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), 'stabilization output/archive already exists')
    cfg, paths = load_protocol(config_path)
    started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    parent = core.load_json(paths['coverage_result'])
    require(parent['status'] == 'complete' and core.load_json(paths['coverage_audit'])['status'] == 'complete', 'coverage parent incomplete')
    manifest, transfer = core.load_json(paths['manifest']), core.load_json(paths['transfer'])
    torch.set_num_threads(1); torch.manual_seed(20261022); np.random.seed(20261022)
    archive_dir.mkdir(parents=True, exist_ok=False)
    progress, files, tasks, access_records = counters(), [], {}, {}
    failure = None
    with (archive_dir / 'events.jsonl').open('x') as events:
        def event(kind, **values):
            row = {'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values}
            events.write(json.dumps(row, sort_keys=True) + '\n'); events.flush()
        try:
            states, reference = {}, {}
            for task in TASKS:
                states[task], access_records[str(task)] = read_source_initials(transfer, manifest, task, Path(SCOPE['dataset_root']))
                progress['source_initial_vectors_read'] += len(states[task])
                with np.load(paths[f'public_source_zero{task}'], allow_pickle=False) as old:
                    require(set(old.files) == set(core.OBS_KEYS) | {'demonstration_indices'} and
                            np.array_equal(old['demonstration_indices'], np.arange(30)), 'old source cache role differs')
                    reference[task] = {name: old[name] for name in core.OBS_KEYS}
                validate_observations(reference[task])
            captured = collect_arms(states, reference, archive_dir, progress, event, files)
            policy = core.load_json(paths['policy_config'])['policy']
            for task in TASKS:
                models = {}
                tasks[str(task)] = {'public_observation_delta_five_minus_zero': observation_delta(captured[task][0], captured[task][5]),
                                    'models': models}
                for seed in SEEDS:
                    previous = parent['task_results'][str(task)]['seed_results'][str(seed)]
                    checkpoint = previous['checkpoint']
                    require(checkpoint['optimizer_updates'] == 500 and checkpoint['arm'] == 'source_baseline', 'checkpoint is not baseline500')
                    verify_archive(previous['archive'])
                    model = core.load_policy(checkpoint, {**policy, 'memory_mode': 'cnn_only'})
                    weight_hash = core._digest_weights(model.state_dict())
                    with np.load(previous['archive']['path'], allow_pickle=False) as old:
                        candidate = {name: old[f'candidate_{name}'] for name in SPACES}
                        steps = old['candidate_pre_action_indices']; state_ids = old['candidate_initial_state_indices']
                        require(np.array_equal(state_ids, np.repeat(np.arange(20), 20)), 'cached candidate role order differs')
                        features = {}
                        for arm in (0, 5):
                            features[arm] = condition.capture_features(model, captured[task][arm])
                            progress['main_forward_calls'] += 1; progress['main_model_rows'] += 30
                            if arm == 0:
                                repeated = condition.capture_features(model, captured[task][0])
                                progress['repeat_forward_calls'] += 1; progress['repeat_model_rows'] += 30
                                require(all(np.array_equal(features[0][k], repeated[k]) for k in repeated), 'zero readout repeat differs')
                                for name in (*SPACES, 'predictions', 'logits'):
                                    require(np.array_equal(features[0][name], old[f'source_t0_{name}']), f'old zero feature parity failed: {name}')
                    require(weight_hash == core._digest_weights(model.state_dict()) == checkpoint['weight_sha256'] and
                            all(p.grad is None for p in model.parameters()), 'stabilization readout mutated weights or gradients')
                    saved = {'source_demo_indices': np.arange(30), 'candidate_pre_action_indices': steps,
                             'candidate_initial_state_indices': state_ids}
                    for arm in (0, 5):
                        for name, values in features[arm].items(): saved[f'stabilization{arm}_{name}'] = values
                    banks = evaluate_banks(features, candidate, steps, progress, saved)
                    for mode in ANALYSIS['scaling_modes']:
                        for space in SPACES:
                            require(banks[mode][space]['0']['horizons'] == previous['banks']['runtime_t0'][space]['horizons'],
                                    'old zero-bank candidate coverage parity failed')
                    record = save(archive_dir / f'task{task}-seed{seed}-readouts-and-distances.npz', saved); files.append(record)
                    models[str(seed)] = {'checkpoint': checkpoint, 'old_candidate_feature_archive': previous['archive'],
                        'old_zero_feature_and_prediction_parity_exact': True, 'weight_sha256_unchanged': weight_hash,
                        'action_delta_five_minus_zero': action_delta(features[0]['predictions'], features[5]['predictions']),
                        'feature_l2_delta_five_minus_zero': {name: summary(np.linalg.norm(features[5][name].astype(np.float64) - features[0][name], axis=1)) for name in SPACES},
                        'banks': banks, 'archive': record}
                    event('model_complete', task_id=task, training_seed=seed)
                    print(json.dumps({'task': task, 'seed_complete': seed}), flush=True)
                    del model
                tasks[str(task)]['decision'] = decisions(models)
            expected = {'source_initial_vectors_read': 60, 'adapters_built': 4, 'resets_attempted': 120,
                'resets_completed': 120, 'native_zero_steps_attempted': 300, 'native_zero_steps_completed': 300,
                'zero_public_cache_matches': 60, 'main_forward_calls': 12, 'repeat_forward_calls': 6,
                'main_model_rows': 360, 'repeat_model_rows': 180, 'unique_bank_scalers': 48,
                'bank_evaluation_groups': 96, 'candidate_knn_rows': 38400, 'reference_knn_rows': 960}
            require(progress == expected, f'stabilization actual budget differs: {progress}')
        except Exception as exc:
            failure = {'type': type(exc).__name__, 'message': str(exc)}
            event('technical_abort', **failure)
    versions = {}
    for name in ('mujoco', 'robosuite', 'gym', 'h5py'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = None
    result = {'check_type': 'libero_task4_task7_source_stabilization_control_diagnostic', 'formal_result': False,
        'status': 'failed' if failure else 'complete', 'failure': failure, 'scope': SCOPE, 'analysis': ANALYSIS,
        'source_initial_access': access_records, 'task_results': tasks, 'archives': files,
        'provenance': {'config_path': str(config_path.resolve()), 'config_sha256': sha(config_path),
            'runner_sha256': sha(Path(__file__)), 'inputs': cfg['inputs'], 'started_at_utc': utc,
            'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'git_status': subprocess.check_output(['git', 'status', '--short', '--branch'], cwd=ROOT, text=True),
            'event_log': {'path': str(archive_dir / 'events.jsonl'), 'sha256': sha(archive_dir / 'events.jsonl')},
            'environment_variables': {k: os.environ.get(k) for k in ('MUJOCO_GL', 'PYOPENGL_PLATFORM', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}},
        'runtime': {'device': 'cpu', 'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
            'package_versions': versions, 'torch_threads': torch.get_num_threads(), 'actual_calls': progress,
            'elapsed_seconds': time.perf_counter() - started, 'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_policy_control_steps': 0},
        'interpretation': {'diagnostic_only': True, 'no_expert_action_target_scored': True, 'no_new_success_rate': True,
            'no_training_or_rollout_promoted': True, 'candidate_and_source_have_other_domain_differences': True}}
    with output.open('x') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    if failure: raise RuntimeError(f'stabilization diagnostic failed: {failure}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output.resolve(), args.archive_dir.resolve())
    print(json.dumps({'status': result['status'], 'output': str(args.output), 'runtime': result['runtime']}))


if __name__ == '__main__':
    main()
