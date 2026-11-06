"""Score cached public source-t0 predictions against recorded source action[0]."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_expert_action_error_diagnostic as expert

core, require, sha = expert.core, expert.require, expert.sha256_file
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-source-t0-target-error-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-source-t0-target-error-20261102-v1.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-source-t0-target-error-artifacts-20261102-v1')
SCOPE = {'task_ids': [4, 7], 'training_seeds': [20261022, 20261023, 20261024], 'arm': 'source_baseline',
    'checkpoint_updates': 500, 'source_role': 'meta_support', 'source_demo_ids': list(range(30)),
    'new_hdf5_action0_rows': 60, 'cached_t0_prediction_rows': 180,
    'new_model_forward_or_backward_calls': 0, 'new_optimizer_updates': 0, 'new_environment_interactions': 0,
    'private_simulator_state_read': False, 'source_init_state_read': False, 'query_dataset_read': False,
    'official_state_files_read': False, 'qualification_accessed': False, 'validation_accessed': False,
    'heldout_accessed': False, 'final_evaluation_accessed': False}
ANALYSIS = {'target_t0': 'recorded_source_HDF5_action0', 'observation_t0': 'previously_saved_public_runtime_source_t0_cache',
    'controls': ['source_t1', 'source_all_t_ge_1'], 'component_normalization': 'per_dimension3_3_1; uniform7_total',
    'initial_translation_excess_ratio': 1.5, 'consistent_gap_requires_all_three_seeds': True,
    'no_training_promotion_from_this_metric': True,
    't0_vs_t1_not_same_observation_or_same_target_counterfactual': True,
    'source_t0_target_error_not_candidate_correct_action_label': True}


def load_protocol(path):
    c = core.load_json(path)
    require(c.get('check_type') == 'libero_task4_task7_source_t0_target_error_diagnostic' and c.get('formal_result') is False,
            'source t0 protocol type differs')
    require(c.get('scope') == SCOPE and c.get('analysis') == ANALYSIS, 'source t0 frozen scope differs')
    require(c['runner_sha256'] == sha(Path(__file__)), 'source t0 runner hash differs')
    require(set(c['inputs']) == {'coverage_result', 'coverage_config', 'condition_result', 'condition_config', 'manifest', 'transfer', 'expert_code'},
            'source t0 input roles differ')
    paths = {}
    for name, row in c['inputs'].items():
        p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(sha(p) == row['sha256'], f'source t0 input pin differs: {name}')
        paths[name] = p
    require(c['inputs']['manifest']['sha256'] == expert.PINS['dataset_manifest'] and
            c['inputs']['transfer']['sha256'] == expert.PINS['transfer_config'] and
            c['inputs']['expert_code']['sha256'] == 'bd44981057bbb09ecc485379ca4ad13563b3f770d892dbd9c73ce3e1af557425', 'source t0 role helper pin differs')
    return c, paths


def authorize_support(transfer, manifest, task, role='meta_support'):
    require(type(task) is int and task in (4, 7) and role == 'meta_support', 'source t0 role outside support')
    return expert.authorize_role(transfer, manifest, task, role)


def read_initial_actions(handle, access):
    require(access.task_role == 'source_spi' and access.demonstration_role == 'meta_support' and
            access.demonstration_indices == tuple(range(30)), 'source t0 action access is not full support')
    values = [np.asarray(handle['data'][f'demo_{index}']['actions'][0], dtype=np.float32) for index in access.demonstration_indices]
    actions = np.stack(values)
    require(actions.shape == (30, 7) and np.isfinite(actions).all() and np.all(np.abs(actions) <= 1), 'invalid source first actions')
    return actions


def excess_decision(models):
    ratios = {}
    for seed in SCOPE['training_seeds']:
        row = models[str(seed)]
        base = row['source_all_t_ge_1']['component_mse']['translation']
        ratios[str(seed)] = row['source_t0']['component_mse']['translation'] / base if base > 0 else None
    return {'translation_t0_over_all_t_ge_1': ratios,
            'consistent_initial_translation_excess': all(value is not None and value >= 1.5 for value in ratios.values()),
            'candidate_action_correctness_established': False, 'training_promoted': False}


def run(config_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), 'source t0 output/archive already exists')
    config, paths = load_protocol(config_path)
    started = time.perf_counter()
    coverage, condition = (core.load_json(paths[name]) for name in ('coverage_result', 'condition_result'))
    require(coverage['status'] == condition['status'] == 'complete' and coverage['scope']['checkpoint_updates'] == 500 and
            condition['scope']['checkpoint_optimizer_updates'] == 500, 'source t0 parent is not complete baseline500')
    manifest, transfer = core.load_json(paths['manifest']), core.load_json(paths['transfer'])
    accesses = {task: authorize_support(transfer, manifest, task) for task in SCOPE['task_ids']}
    archive_dir.mkdir(parents=True, exist_ok=False)
    task_results, files, source_rows = {}, [], 0
    root = Path(condition['scope']['dataset_root']).resolve()
    for task in SCOPE['task_ids']:
        record = next(row for row in manifest['files'] if row['task_id'] == task)
        path = (root / record['path']).resolve()
        require(path.is_relative_to(root) and sha(path) == record['sha256'], 'source first-action HDF5 identity differs')
        with h5py.File(path, 'r') as handle: targets = read_initial_actions(handle, accesses[task])
        source_rows += len(targets)
        values = {'target_actions_t0': targets, 'source_demo_ids': np.arange(30, dtype=np.int64)}
        feature_path = condition['task_results'][str(task)]['archive']
        require(sha(Path(feature_path['path'])) == feature_path['sha256'], 'source t1/all cache hash differs')
        models = {}
        with np.load(feature_path['path'], allow_pickle=False) as previous:
            first = previous['action_ids'] == 1
            require(np.array_equal(previous['demo_ids'][first], np.arange(30)), 'source t1 control ordering differs')
            for seed in SCOPE['training_seeds']:
                native = coverage['task_results'][str(task)]['seed_results'][str(seed)]
                require(native['checkpoint']['arm'] == 'source_baseline' and native['checkpoint']['optimizer_updates'] == 500,
                        'source first-action checkpoint differs')
                require(sha(Path(native['archive']['path'])) == native['archive']['sha256'], 'source t0 prediction cache hash differs')
                with np.load(native['archive']['path'], allow_pickle=False) as cache:
                    prediction = cache['source_t0_predictions']
                values[f'prediction_seed{seed}'] = prediction
                all_prediction = previous[f'seed{seed}__predictions']
                models[str(seed)] = {
                    'source_t0': expert.error_metrics(prediction, targets, np.ones(30, bool)),
                    'source_t1': expert.error_metrics(all_prediction[first], previous['target_actions'][first], np.ones(30, bool)),
                    'source_all_t_ge_1': expert.error_metrics(all_prediction, previous['target_actions'], np.ones(len(all_prediction), bool)),
                    'weight_sha256': native['checkpoint']['weight_sha256']}
        artifact = archive_dir / f'task{task}-source-t0-targets-predictions.npz'
        digest = expert.save_archive(artifact, values)
        files.append({'path': str(artifact), 'sha256': digest})
        task_results[str(task)] = {'source_access': accesses[task].to_dict(), 'models': models, 'decision': excess_decision(models),
                                  'dataset_sha256': record['sha256']}
    require(source_rows == 60, 'source t0 target budget differs')
    return {'check_type': config['check_type'], 'status': 'complete', 'formal_result': False, 'scope': SCOPE, 'analysis': ANALYSIS,
        'runtime': {'elapsed_seconds': time.perf_counter() - started, 'source_action0_rows': source_rows,
                    'new_model_calls': 0, 'new_optimizer_or_environment_steps': 0, 'python': sys.version.split()[0], 'numpy': np.__version__},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': config['inputs']},
        'task_results': task_results, 'archives': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args(); result = run(args.config, args.output, args.archive_dir)
    with args.output.open('x') as f: json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__': main()
