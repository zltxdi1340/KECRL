"""Finish fixed-Policy readouts from complete paired public reset caches.

The original source capture failed at a Path/string file-hash boundary before
any model forward. This entry point performs no HDF5 or environment work and
retains the original failed result, configuration, log and source snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
from experiments import run_libero_task4_task7_source_stabilization_control as control

core, require, sha = control.core, control.require, control.sha
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-source-stabilization-analysis-diagnostic_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-source-stabilization-analysis-20261102-v1.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-source-stabilization-analysis-artifacts-20261102-v1')
SCOPE = {'task_ids': [4, 7], 'training_seeds': list(control.SEEDS), 'checkpoint_updates': 500,
    'source_role': 'meta_support', 'source_demo_indices': list(range(30)),
    'captured_public_source_rows': 120, 'candidate_feature_rows_reused': 2400,
    'new_source_initial_vector_or_hdf5_payload_reads': 0, 'new_environment_resets_or_steps': 0,
    'main_forward_calls': 12, 'repeat_forward_calls': 6, 'main_model_rows': 360, 'repeat_model_rows': 180,
    'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_policy_control_steps': 0,
    'official_state_files_read': False, 'qualification_accessed': False, 'reserved_accessed': False,
    'validation_accessed': False, 'heldout_accessed': False, 'final_evaluation_accessed': False,
    'device': 'cpu', 'torch_threads': 1, 'spt_updated': False, 'knowledge_updated': False, 'module_registered': False}
EXPECTED_CAPTURE = {'source_initial_vectors_read': 60, 'adapters_built': 4, 'resets_attempted': 120,
    'resets_completed': 120, 'native_zero_steps_attempted': 300, 'native_zero_steps_completed': 300,
    'zero_public_cache_matches': 60, 'main_forward_calls': 0, 'repeat_forward_calls': 0,
    'main_model_rows': 0, 'repeat_model_rows': 0, 'unique_bank_scalers': 0,
    'bank_evaluation_groups': 0, 'candidate_knn_rows': 0, 'reference_knn_rows': 0}


def validate_capture(data):
    require(data['status'] == 'failed' and data['formal_result'] is False and
            data['failure'] == {'type': 'AttributeError', 'message': "'str' object has no attribute 'open'"},
            'only the recorded pre-forward Path failure can be resumed')
    require(data['scope'] == control.SCOPE and data['analysis'] == control.ANALYSIS,
            'capture frozen roles or metrics differ')
    require(data['runtime']['actual_calls'] == EXPECTED_CAPTURE, 'source capture incomplete or readouts already attempted')
    names = {Path(r['path']).name for r in data['archives']}
    require(len(data['archives']) == 4 and names == {f'task{t}-stabilization{a}-public-observations.npz' for t in (4, 7) for a in (0, 5)},
            'complete paired source observation matrix is required')


def load_protocol(path):
    cfg = core.load_json(path)
    require(cfg.get('check_type') == 'libero_task4_task7_source_stabilization_analysis_diagnostic' and
            cfg.get('formal_result') is False and cfg.get('scope') == SCOPE and
            cfg.get('analysis') == control.ANALYSIS, 'frozen cache analysis scope or metrics differ')
    require(cfg['runner_sha256'] == sha(Path(__file__)), 'offline stabilization analysis hash differs')
    require(set(cfg['inputs']) == {'capture', 'capture_config', 'capture_code_snapshot', 'current_control_code',
                                 'launch_record', 'process_exit_record'}, 'offline continuation input roles differ')
    paths = {}
    for name, row in cfg['inputs'].items():
        p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(sha(p) == row['sha256'], f'continuation input hash differs: {name}'); paths[name] = p
    capture = core.load_json(paths['capture']); validate_capture(capture)
    require(capture['provenance']['runner_sha256'] == sha(paths['capture_code_snapshot']) ==
            '0f44ea3acecb735b9ad52b97b10f9dd4b01d7d26c8c840ce0b5b22b7e551c83a', 'original source execution code differs')
    original = core.load_json(paths['capture_config'])
    require(capture['provenance']['config_sha256'] == sha(paths['capture_config']) and
            original['scope'] == control.SCOPE and original['analysis'] == control.ANALYSIS and
            original['runner_sha256'] == sha(paths['capture_code_snapshot']), 'original capture protocol differs')
    require(set(original['inputs']) == set(control.INPUT_PINS), 'capture input roles differ')
    for name, row in original['inputs'].items():
        p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(row['sha256'] == control.INPUT_PINS[name] and sha(p) == row['sha256'], f'original input pin differs: {name}')
    process, launch = core.load_json(paths['process_exit_record']), core.load_json(paths['launch_record'])
    require(process['process_exit_code'] == 1 and process['launch_sha256'] == sha(paths['launch_record']) and
            sha(Path(process['log_path'])) == process['log_sha256'] and
            launch['config_sha256'] == sha(paths['capture_config']), 'original failed process provenance differs')
    return cfg, paths, capture, original


def run(config_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), 'offline stabilization output/archive already exists')
    cfg, paths, capture, original = load_protocol(config_path)
    started, utc = time.perf_counter(), datetime.now(timezone.utc).isoformat()
    archive_dir.mkdir(parents=True, exist_ok=False)
    parent = core.load_json(Path(original['inputs']['coverage_result']['path']))
    event_record = capture['provenance']['event_log']
    require(sha(Path(event_record['path'])) == event_record['sha256'], 'source capture event log changed')
    events = [json.loads(line) for line in Path(event_record['path']).read_text().splitlines()]
    resets = [r for r in events if r['event'] == 'source_reset']
    require([(r['stabilization_steps'], r['task_id'], r['source_demo_index']) for r in resets] ==
            [(a, t, i) for a in (0, 5) for t in (4, 7) for i in range(30)], 'source reset order differs')
    require(all(r['zero_cache_exact_match'] for r in resets[:60]), 'source zero preflight failed')
    torch.set_num_threads(1); torch.manual_seed(20261022); np.random.seed(20261022)
    progress, tasks, files = control.counters(), {}, []
    policy = core.load_json(ROOT / original['inputs']['policy_config']['path'])['policy']
    for task in control.TASKS:
        captured = {}
        for arm in (0, 5):
            record = next(r for r in capture['archives'] if Path(r['path']).name == f'task{task}-stabilization{arm}-public-observations.npz')
            path = control.verify_archive(record)
            with np.load(path, allow_pickle=False) as z:
                require(set(z.files) == set(core.OBS_KEYS) | {'source_demo_indices'} and
                        np.array_equal(z['source_demo_indices'], np.arange(30)), 'cached source observation roles differ')
                captured[arm] = {name: z[name] for name in core.OBS_KEYS}
            control.validate_observations(captured[arm])
            for demo in range(30):
                row = next(r for r in resets if (r['task_id'], r['stabilization_steps'], r['source_demo_index']) == (task, arm, demo))
                require(core._digest_observation({k: v[demo] for k, v in captured[arm].items()}) == row['observation_sha256'], 'source observation event digest differs')
        with np.load(Path(original['inputs'][f'public_source_zero{task}']['path']), allow_pickle=False) as old:
            require(all(np.array_equal(captured[0][name], old[name]) for name in core.OBS_KEYS), 'zero cache no longer exact')
        models = {}; delta = control.observation_delta(captured[0], captured[5])
        if str(task) in capture['task_results']:
            require(delta == capture['task_results'][str(task)]['public_observation_delta_five_minus_zero'], 'completed public metrics changed')
        for seed in control.SEEDS:
            previous = parent['task_results'][str(task)]['seed_results'][str(seed)]
            checkpoint = previous['checkpoint']; cache_path = control.verify_archive(previous['archive'])
            require(checkpoint['optimizer_updates'] == 500 and checkpoint['arm'] == 'source_baseline', 'not baseline500')
            model = core.load_policy(checkpoint, {**policy, 'memory_mode': 'cnn_only'})
            before = core._digest_weights(model.state_dict())
            with np.load(cache_path, allow_pickle=False) as old:
                candidate = {name: old[f'candidate_{name}'] for name in control.SPACES}
                steps, state_ids = old['candidate_pre_action_indices'], old['candidate_initial_state_indices']
                require(np.array_equal(state_ids, np.repeat(np.arange(20), 20)), 'candidate index order differs')
                features = {}
                for arm in (0, 5):
                    features[arm] = control.condition.capture_features(model, captured[arm])
                    progress['main_forward_calls'] += 1; progress['main_model_rows'] += 30
                    if arm == 0:
                        repeated = control.condition.capture_features(model, captured[0])
                        progress['repeat_forward_calls'] += 1; progress['repeat_model_rows'] += 30
                        require(all(np.array_equal(features[0][name], repeated[name]) for name in repeated), 'source repeat differs')
                        require(all(np.array_equal(features[0][name], old[f'source_t0_{name}']) for name in features[0]), 'old zero CPU feature parity failed')
            require(before == core._digest_weights(model.state_dict()) == checkpoint['weight_sha256'] and
                    all(p.grad is None for p in model.parameters()), 'model mutated during fixed readouts')
            saved = {'source_demo_indices': np.arange(30), 'candidate_pre_action_indices': steps,
                     'candidate_initial_state_indices': state_ids}
            for arm in (0, 5):
                for name, array in features[arm].items(): saved[f'stabilization{arm}_{name}'] = array
            banks = control.evaluate_banks(features, candidate, steps, progress, saved)
            for mode in control.ANALYSIS['scaling_modes']:
                for space in control.SPACES:
                    require(banks[mode][space]['0']['horizons'] == previous['banks']['runtime_t0'][space]['horizons'], 'old zero coverage parity failed')
            record = control.save(archive_dir / f'task{task}-seed{seed}-readouts-and-distances.npz', saved); files.append(record)
            models[str(seed)] = {'checkpoint': checkpoint, 'old_candidate_feature_archive': previous['archive'],
                'old_zero_feature_and_prediction_parity_exact': True, 'weight_sha256_unchanged': before,
                'action_delta_five_minus_zero': control.action_delta(features[0]['predictions'], features[5]['predictions']),
                'feature_l2_delta_five_minus_zero': {name: control.summary(np.linalg.norm(features[5][name].astype(np.float64) - features[0][name], axis=1)) for name in control.SPACES},
                'banks': banks, 'archive': record}
            print(json.dumps({'task': task, 'seed_complete': seed}), flush=True); del model
        tasks[str(task)] = {'public_observation_delta_five_minus_zero': delta, 'models': models, 'decision': control.decisions(models)}
    expected = {**dict.fromkeys(control.counters(), 0), 'main_forward_calls': 12, 'repeat_forward_calls': 6,
                'main_model_rows': 360, 'repeat_model_rows': 180, 'unique_bank_scalers': 48,
                'bank_evaluation_groups': 96, 'candidate_knn_rows': 38400, 'reference_knn_rows': 960}
    require(progress == expected, f'offline continuation budget differs: {progress}')
    result = {'check_type': 'libero_task4_task7_source_stabilization_analysis_diagnostic', 'status': 'complete',
        'formal_result': False, 'scope': SCOPE, 'analysis': control.ANALYSIS, 'task_results': tasks, 'archives': files,
        'source_capture_parent': {'path': str(paths['capture']), 'sha256': sha(paths['capture']), 'status': 'failed_at_offline_preflight_after_complete_source_capture',
            'source_initial_access': capture['source_initial_access'], 'runtime': capture['runtime'],
            'observation_archives': capture['archives'], 'original_provenance': capture['provenance']},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'inputs': cfg['inputs'], 'started_at_utc': utc, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'runtime': {'device': 'cpu', 'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__,
            'torch_threads': torch.get_num_threads(), 'actual_calls': progress, 'elapsed_seconds': time.perf_counter() - started,
            'new_optimizer_updates': 0, 'new_backward_calls': 0, 'new_environment_resets_or_steps': 0},
        'interpretation': {'diagnostic_only': True, 'no_expert_target_scored': True, 'no_new_success_rate': True,
            'no_training_or_rollout_promoted': True, 'original_source_capture_not_repeated': True}}
    with output.open('x') as f: json.dump(result, f, indent=2, sort_keys=True); f.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args(); result = run(args.config.resolve(), args.output.resolve(), args.archive_dir.resolve())
    print(json.dumps({'status': result['status'], 'runtime': result['runtime']}))


if __name__ == '__main__': main()
