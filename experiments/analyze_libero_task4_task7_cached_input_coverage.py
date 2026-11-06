"""Compare cached candidate observations with fixed source-support feature banks.

No environment, official state file or HDF5 payload is opened. Existing reset,
action and public-state prefixes must match the original closed-loop record.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_condition_separability as condition

core, require, sha = condition.core, condition.require, condition.sha256_file
TASKS, SEEDS, SPACES = condition.TASKS, condition.SEEDS, condition.SPACES
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-cached-input-coverage-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-cached-input-coverage-20261102-v1.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-cached-input-coverage-artifacts-20261102-v1')
PINS = {
    'parent_result': '66682c55850acde665ca09c84ffcffe9ba5f0a0e9ae6c743b2390394ddb2a51e',
    'parent_config': '85769119f0b10bc5c92049a621952577d1380abf29bb5998d1894d0a97b4cd00',
    'parent_code': '58229e2684ad76bd9111b0c6099542b840380d12212d40eaa1c69943826929c7',
    'parent_audit': 'a698a1b6423a11d4f9b24130c3310929cd7966708b64067d67ed15bb0d397894',
    'support_features4': '1d0e9cce3c8d087a2cfd57ccc3f2f430ecf8375ea2ef65ebbff486ad1791c2c5',
    'support_features7': '7c297fb04833c7588c098a0d5fb8d3bf71ca632593ec581788878c1565df8efb',
    'capture': '1f225135c3dd1da796660cbb470fc055551a6d4edb03e0794647afc1b96404a9',
    'source_closed_loop': '03439cc0172f3ba29ee4527d3c64d1da16edd93d2eaf63134344e4c8b15f49ce',
    'prefix_preflight': 'd3ad809d79fe0a456e532c40099a8c80e4613b7e488549bb9c09e25a0be7cef6',
    'public_t0_record': 'c231ced84aa4d35e573e0ad33f0923bc91e7d7bb6ef0bbd615743505efa4082a',
    'public_support_t0_4': '508f7f62b4ef9d8572f35fe6eecc42586c11de700a829d6a2b73c3c968c1dbb5',
    'public_support_t0_7': '4423fe03943cdab579e66511a68110c57181d398097deaf51154ae9b8bceec90',
    'policy_config': condition.PINS['policy_config'], 'policy_code': condition.PINS['policy_code'],
    'offline_helpers': condition.PINS['offline_helpers'], 'expert_code': condition.PINS['expert_code'],
    'overnight_plan': 'd6b434f0812ed9c328b5ea8c28968f4dce6f21093a41e06e63a5677eaac3aa61',
}
SCOPE = {
    'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arm': 'source_baseline', 'checkpoint_updates': 500,
    'candidate_role': 'cached_candidate_nonregression', 'candidate_indices': list(range(20)), 'preaction_indices': list(range(20)),
    'source_role': 'meta_support', 'source_demo_indices': list(range(30)), 'bank_demo_indices': list(range(20)),
    'reference_demo_indices': list(range(20, 30)), 'candidate_cached_frames': 2400, 'source_cached_t0_frames': 60,
    'main_forward_calls': 84, 'repeat_forward_calls': 6, 'main_model_rows': 2580, 'repeat_model_rows': 192,
    'batch_size': 32, 'device': 'cpu', 'torch_threads': 1, 'analysis_seed': 20261022,
    'bank_calibrations': 72, 'candidate_knn_rows': 28800,
    'new_optimizer_updates': 0, 'new_model_backward_calls': 0, 'new_environment_resets_or_actions': 0,
    'hdf5_payload_reads': 0, 'official_state_files_read': False, 'source_private_state_read': False,
    'qualification_accessed': False, 'validation_accessed': False, 'heldout_accessed': False, 'final_evaluation_accessed': False,
    'checkpoint_selected': False, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False,
}
ANALYSIS = {
    'spaces': SPACES, 'banks': ['historical_all', 'historical_prefix20', 'runtime_t0'],
    'bank_reference_split': 'source_demo0to19_bank; source_demo20to29_reference; not_official_state_indices',
    'scaler': 'bank_only_float64_mean_population_std; std_le1e-6_replaced_by1',
    'neighbors': 5, 'distance': 'mean_of_five_squared_euclidean_distances_on_bank_zscore',
    'neighbor_ties': 'stable_distance_sort_by_original_bank_row_order', 'query_chunk': 64,
    'reference_quantile': .95, 'quantile_method': 'numpy_default_linear',
    'preaction_horizons': [0, 4, 9, 19], 'exceedance': 'strictly_greater_than_reference95',
    'primary_space': 'fusion', 'primary_bank': 'historical_prefix20',
    'initial_gap': 'initial_exceedance_fraction_ge0.50_in_all_three_seeds',
    'feedback_growth': 'step19_minus_step0_exceedance_fraction_ge0.25_in_all_three_seeds',
    'runtime_t0_contrast': 'initial_gap_and_runtime_t0_initial_exceedance_fraction_le0.25_in_all_three_seeds',
    'source_runtime_t0_reference_also_compared_with_historical_banks': True,
    'cpu_vs_original_cuda_action_delta': 'descriptive_only; original_source_prefix_gate_remains_exact',
    'no_expert_action_target_for_candidate_frames': True,
    'no_ood_validity_threshold_or_closed_loop_causal_claim': True,
    'no_policy_training_or_checkpoint_selection_from_distance': True,
    'source_t0_stabilization0_candidate_stabilization5_is_a_confound': True,
}


def load_protocol(path):
    cfg = core.load_json(path)
    require(cfg.get('schema_version') == 1 and cfg.get('check_type') == 'libero_task4_task7_cached_input_coverage_diagnostic' and
            cfg.get('formal_result') is False, 'coverage protocol type differs')
    require(cfg.get('protocol_status') == 'frozen_cached_observation_diagnostic', 'coverage protocol is not frozen')
    require(cfg.get('scope') == SCOPE and cfg.get('analysis') == ANALYSIS, 'coverage frozen scope or analysis differs')
    require(cfg['runner_sha256'] == sha(Path(__file__)), 'coverage runner hash differs')
    require(set(cfg['inputs']) == set(PINS), 'coverage input roles differ')
    paths = {}
    for name, digest in PINS.items():
        row = cfg['inputs'][name]
        path = Path(row['path']); path = path if path.is_absolute() else ROOT / path
        require(row['sha256'] == digest and sha(path) == digest, f'coverage input hash differs: {name}')
        paths[name] = path.resolve()
    return cfg, paths


def fit_bank(bank):
    require(bank.ndim == 2 and len(bank) >= 5 and np.isfinite(bank).all(), 'invalid coverage bank')
    bank = bank.astype(np.float64)
    mean, scale = bank.mean(axis=0), bank.std(axis=0)
    scale[scale <= 1e-6] = 1.
    return {'mean': mean, 'scale': scale, 'normalized': (bank - mean) / scale}


def bank_distances(fitted, queries):
    require(queries.ndim == 2 and queries.shape[1] == len(fitted['mean']) and np.isfinite(queries).all(), 'invalid coverage query')
    bank = fitted['normalized']; norm = np.square(bank).sum(axis=1)
    neighbors, distances = [], []
    for start in range(0, len(queries), 64):
        q = (queries[start:start + 64].astype(np.float64) - fitted['mean']) / fitted['scale']
        squared = np.maximum(np.square(q).sum(axis=1)[:, None] + norm[None, :] - 2 * (q @ bank.T), 0.)
        nearest = np.argsort(squared, axis=1, kind='stable')[:, :5]
        neighbors.append(nearest)
        distances.append(np.take_along_axis(squared, nearest, axis=1))
    return np.concatenate(neighbors), np.concatenate(distances)


def distance_summary(values, threshold):
    require(values.ndim == 1 and len(values) and np.isfinite(values).all() and np.isfinite(threshold), 'empty or invalid coverage summary')
    return {'rows': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'minimum': float(values.min()), 'maximum': float(values.max()),
            'reference95': float(threshold), 'exceedance_fraction': float(np.mean(values > threshold))}


def coverage_decisions(seed_results):
    initial, growth, native = {}, {}, {}
    for seed in SEEDS:
        record = seed_results[str(seed)]['banks']
        source = record['historical_prefix20']['fusion']['horizons']
        initial[str(seed)] = source['0']['exceedance_fraction']
        growth[str(seed)] = source['19']['exceedance_fraction'] - initial[str(seed)]
        native[str(seed)] = record['runtime_t0']['fusion']['horizons']['0']['exceedance_fraction']
    gap = all(value >= .5 for value in initial.values())
    return {'initial_fraction_by_seed': initial, 'step19_minus_step0_fraction_by_seed': growth, 'runtime_t0_initial_fraction_by_seed': native,
            'consistent_initial_feature_coverage_gap': gap, 'consistent_feedback_fraction_growth': all(value >= .25 for value in growth.values()),
            'runtime_t0_reference_contrast': gap and all(value <= .25 for value in native.values()),
            'closed_loop_failure_cause_established': False, 'training_or_rollout_promoted': False}


def infer_batches(model, observations, calls, repeat_first=False):
    total = len(observations['joint_states'])
    parts = {name: [] for name in (*SPACES, 'predictions', 'logits')}
    for start in range(0, total, 32):
        batch = {name: values[start:start + 32] for name, values in observations.items()}
        output = condition.capture_features(model, batch)
        calls['main_forward_calls'] += 1; calls['main_model_rows'] += len(batch['joint_states'])
        if repeat_first and start == 0:
            repeated = condition.capture_features(model, batch)
            require(all(np.array_equal(output[name], repeated[name]) for name in output), 'coverage repeated feature readout differs')
            calls['repeat_forward_calls'] += 1; calls['repeat_model_rows'] += len(batch['joint_states'])
        for name, values in output.items(): parts[name].append(values)
    return {name: np.concatenate(values) for name, values in parts.items()}


def run(config_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), 'coverage output/archive already exists')
    cfg, paths = load_protocol(config_path)
    started, started_utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    parent, capture, original, local = (core.load_json(paths[name]) for name in ('parent_result', 'capture', 'source_closed_loop', 'public_t0_record'))
    require(parent['status'] == 'complete' and core.load_json(paths['parent_audit'])['status'] == 'complete', 'coverage parent or audit not complete')
    require(all(not task['decisions']['eligible_for_separately_frozen_logit_loss_training'] for task in parent['task_results'].values()), 'coverage plan requires retained failed training gates')
    prefix = core.load_json(paths['prefix_preflight'])['records']
    require(len(prefix) == 120 and all(row['source_reset_digest_match'] and row['source_prefix_exact_match'] for row in prefix), 'coverage source prefix preflight failed')
    index = core.unique_index(capture['episode_rows'], ('task_id', 'training_seed', 'initial_state_index'))
    require(set(index) == {(t, s, i) for t in TASKS for s in SEEDS for i in range(20)}, 'coverage candidate archive matrix differs')
    source = {(int(t), int(s), ep['initial_state_index']): ep for t, tr in original['task_results'].items()
              for s, sr in tr['seed_results'].items() for ep in sr['candidate_nonregression']['episodes']}
    policy = core.load_json(paths['policy_config'])['policy']
    torch.set_num_threads(1); torch.manual_seed(20261022); np.random.seed(20261022)
    archive_dir.mkdir(parents=True, exist_ok=False)
    calls = {'main_forward_calls': 0, 'repeat_forward_calls': 0, 'main_model_rows': 0, 'repeat_model_rows': 0, 'bank_calibrations': 0, 'candidate_knn_rows': 0}
    task_results, files, verified_prefixes, failure = {}, [], [], None
    try:
        for task in TASKS:
            with np.load(paths[f'public_support_t0_{task}'], allow_pickle=False) as cache:
                require(set(cache.files) == set(core.OBS_KEYS) | {'demonstration_indices'} and
                        np.array_equal(cache['demonstration_indices'], np.arange(30)), 'source public t0 cache role differs')
                t0_observations = {name: cache[name] for name in core.OBS_KEYS}
            metadata = local['task_results'][str(task)]['source_role_captures']['meta_support']['public_t0_archive']
            require(metadata['sha256'] == PINS[f'public_support_t0_{task}'] and not metadata['private_initial_state_values_recorded'], 'public source cache provenance differs')
            for demo in range(30):
                require(core._digest_observation({name: values[demo] for name, values in t0_observations.items()}) == metadata['observation_sha256_by_demo'][str(demo)], 'public t0 observation digest differs')
            seed_results = {}
            with np.load(paths[f'support_features{task}'], allow_pickle=False) as historical:
                demo_ids, action_ids = historical['demo_ids'], historical['action_ids']
                require(set(demo_ids.tolist()) == set(range(30)), 'coverage feature bank is not source support')
                for seed in SEEDS:
                    checkpoint = parent['task_results'][str(task)]['models'][str(seed)]['checkpoint']
                    require(checkpoint['optimizer_updates'] == 500 and checkpoint['arm'] == 'source_baseline', 'coverage checkpoint is not baseline500')
                    model = core.load_policy(checkpoint, {**policy, 'memory_mode': 'cnn_only'})
                    before = core._digest_weights(model.state_dict())
                    observation_parts = {name: [] for name in core.OBS_KEYS}
                    old_actions = []
                    for state in range(20):
                        row = index[(task, seed, state)]
                        require(sha(Path(row['archive_file'])) == row['archive_file_sha256'], 'candidate observation archive hash differs')
                        with np.load(row['archive_file'], allow_pickle=False) as cache:
                            arrays = {name: cache[name] for name in cache.files}
                        core.validate_archive(arrays)
                        audit = core.compare_source_prefix(arrays, row, source[(task, seed, state)])
                        core.require_source_match(audit)
                        verified_prefixes.append({'task_id': task, 'training_seed': seed, 'state_index': state, 'archive_sha256': row['archive_file_sha256'], 'exact_prefix': True})
                        for name in core.OBS_KEYS: observation_parts[name].append(arrays[name])
                        old_actions.append(arrays['actions'])
                    observations = {name: np.concatenate(parts) for name, parts in observation_parts.items()}
                    candidate = infer_batches(model, observations, calls, repeat_first=True)
                    runtime_t0 = infer_batches(model, t0_observations, calls)
                    require(core._digest_weights(model.state_dict()) == before == checkpoint['weight_sha256'], 'coverage inference mutated fixed checkpoint')
                    rows = np.repeat(np.arange(20), 20); steps = np.tile(np.arange(20), 20)
                    saved = {'candidate_initial_state_indices': rows, 'candidate_pre_action_indices': steps,
                             'candidate_predictions': candidate['predictions'], 'candidate_logits': candidate['logits'],
                             'source_t0_predictions': runtime_t0['predictions'], 'source_t0_logits': runtime_t0['logits']}
                    for space in SPACES:
                        saved[f'candidate_{space}'] = candidate[space]
                        saved[f'source_t0_{space}'] = runtime_t0[space]
                    banks_result = {}
                    for bank_name in ANALYSIS['banks']:
                        banks_result[bank_name] = {}
                        if bank_name == 'runtime_t0':
                            bank_mask, reference_mask = np.arange(30) < 20, np.arange(30) >= 20
                        else:
                            bank_mask, reference_mask = demo_ids < 20, demo_ids >= 20
                            if bank_name == 'historical_prefix20':
                                bank_mask, reference_mask = bank_mask & (action_ids <= 20), reference_mask & (action_ids <= 20)
                        for space in SPACES:
                            features = runtime_t0[space] if bank_name == 'runtime_t0' else historical[f'seed{seed}__{space}']
                            fitted = fit_bank(features[bank_mask])
                            ref_indices, ref_distances = bank_distances(fitted, features[reference_mask])
                            query_indices, query_distances = bank_distances(fitted, candidate[space])
                            t0_indices, t0_distances = bank_distances(fitted, runtime_t0[space][20:30])
                            reference_values, query_values = ref_distances.mean(axis=1), query_distances.mean(axis=1)
                            threshold = float(np.quantile(reference_values, .95))
                            key = f'{bank_name}__{space}'
                            saved[f'{key}__bank_row_ids'] = np.flatnonzero(bank_mask)
                            saved[f'{key}__reference_row_ids'] = np.flatnonzero(reference_mask)
                            saved[f'{key}__candidate_distances5'] = query_distances
                            saved[f'{key}__candidate_bank_indices5'] = query_indices
                            saved[f'{key}__reference_distances5'] = ref_distances
                            saved[f'{key}__reference_bank_indices5'] = ref_indices
                            saved[f'{key}__source_t0_reference_distances5'] = t0_distances
                            saved[f'{key}__source_t0_reference_bank_indices5'] = t0_indices
                            banks_result[bank_name][space] = {
                                'bank_rows': int(bank_mask.sum()), 'reference_rows': int(reference_mask.sum()),
                                'bank_mean_sha256': core._digest_array(fitted['mean']), 'bank_scale_sha256': core._digest_array(fitted['scale']),
                                'reference': distance_summary(reference_values, threshold),
                                'candidate_all20_steps': distance_summary(query_values, threshold),
                                'source_runtime_t0_reference': distance_summary(t0_distances.mean(axis=1), threshold),
                                'horizons': {str(step): distance_summary(query_values[steps == step], threshold) for step in (0, 4, 9, 19)},
                            }
                            calls['bank_calibrations'] += 1; calls['candidate_knn_rows'] += len(query_values)
                    path = archive_dir / f'task{task}-seed{seed}-coverage.npz'
                    digest = condition.expert.save_archive(path, saved)
                    record = {'path': str(path), 'sha256': digest, 'size_bytes': path.stat().st_size}
                    files.append(record)
                    seed_results[str(seed)] = {'checkpoint': checkpoint, 'banks': banks_result, 'archive': record,
                        'cpu_vs_recorded_cuda_action_max_abs': float(np.max(np.abs(candidate['predictions'].astype(np.float64) - np.concatenate(old_actions).astype(np.float64)))),
                        'weight_hash_unchanged': before}
                    print(json.dumps({'task': task, 'seed_complete': seed}), flush=True)
                    del model
            task_results[str(task)] = {'seed_results': seed_results, 'decisions': coverage_decisions(seed_results)}
        require(calls == {'main_forward_calls': 84, 'repeat_forward_calls': 6, 'main_model_rows': 2580, 'repeat_model_rows': 192,
                          'bank_calibrations': 72, 'candidate_knn_rows': 28800}, 'coverage compute budget differs')
        require(len(verified_prefixes) == 120, 'coverage candidate source prefixes incomplete')
    except Exception as error:
        failure = {'type': type(error).__name__, 'message': str(error)}
    return {'check_type': cfg['check_type'], 'status': 'complete' if failure is None else 'failed_cached_coverage_diagnostic',
        'formal_result': False, 'scope': SCOPE, 'analysis': ANALYSIS, 'failure': failure,
        'runtime': {'elapsed_seconds': time.perf_counter() - started, 'actual_calls': calls, 'python': sys.version.split()[0],
                    'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'torch_threads': torch.get_num_threads(),
                    'new_optimizer_updates': 0, 'new_environment_interactions': 0},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'started_at_utc': started_utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
            'git_status': subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain'], text=True),
            'inputs': {name: {'path': str(path), 'sha256': PINS[name]} for name, path in paths.items()}},
        'task_results': task_results, 'archives': files, 'verified_source_prefixes': verified_prefixes,
        'interpretation': {'distances_not_action_errors_or_success': True, 'source_demo20to29_not_official_qualification_states': True,
            'bank_size_phase_and_stabilization_confound_runtime_contrast': True, 'no_knowledge_or_module_result': True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = run(args.config, args.output, args.archive_dir)
    with args.output.open('x') as f: json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'failure': result['failure']}), flush=True)
    if result['failure']: raise SystemExit(1)


if __name__ == '__main__': main()
