"""Recover an interrupted JSON export from complete immutable readout archives.

Does not instantiate or call a Policy. Original incomplete JSON/log/code remain
untouched; all target metrics and decisions are recomputed into a new result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_gripper_feedback as feedback

require, sha = feedback.require, feedback.sha
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-gripper-feedback-serialization-recovery_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-gripper-feedback-20261102-v2.json')
SCOPE = {'task_ids': [4, 7], 'training_seeds': list(feedback.SEEDS), 'role': 'meta_support',
         'new_partial_policy_readouts': 0, 'new_cnn_forwards': 0, 'new_optimizer_updates': 0,
         'new_environment_interactions': 0, 'old_files_overwritten': False}


def complete_top_level_prefix(text):
    decoder, index, values = json.JSONDecoder(), 1, {}
    require(text.startswith('{'), 'partial export is not a JSON object')
    while True:
        while index < len(text) and (text[index].isspace() or text[index] == ','): index += 1
        if index >= len(text) or text[index] == '}': return values, None
        key, index = decoder.raw_decode(text, index)
        while text[index].isspace(): index += 1
        require(text[index] == ':', 'malformed partial export key separator')
        index += 1
        while index < len(text) and text[index].isspace(): index += 1
        try: value, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError: return values, key
        values[key] = value


def recover(config_path, output):
    require(not output.exists(), 'recovery output already exists')
    config = feedback.core.load_json(config_path)
    require(config.get('check_type') == 'libero_gripper_feedback_serialization_recovery' and config.get('formal_result') is False and
            config.get('scope') == SCOPE, 'recovery must remain a zero-readout export operation')
    require(config['runner_sha256'] == sha(Path(__file__)), 'recovery runner hash differs')
    for name, row in config['inputs'].items(): require(sha(Path(row['path'])) == row['sha256'], f'recovery pin differs: {name}')
    paths = {name: Path(row['path']) for name, row in config['inputs'].items()}
    require(set(paths) == {'partial_result', 'original_log', 'original_config', 'original_runner_snapshot', 'corrected_runner',
        'parent_result', 'support4', 'support7', 'readouts4', 'readouts7', 'overnight_plan'}, 'recovery input roles differ')
    started = time.perf_counter()
    prefix, stopped = complete_top_level_prefix(paths['partial_result'].read_text())
    require(stopped == 'task_results' and set(prefix) == {'analysis', 'archives', 'check_type', 'formal_result', 'interpretation', 'provenance', 'runtime', 'scope', 'status'},
            'partial export lacks complete original scope/runtime/provenance')
    require(prefix['scope'] == feedback.SCOPE and prefix['analysis'] == feedback.ANALYSIS and prefix['formal_result'] is False, 'recovery scientific settings differ')
    require(prefix['provenance']['config_sha256'] == sha(paths['original_config']) and
            prefix['provenance']['runner_sha256'] == sha(paths['original_runner_snapshot']), 'recovery original source provenance differs')
    require(prefix['runtime']['actual_partial_readouts'] == {'main_chains': 2700, 'repeat_chains': 6, 'main_rows': 77670, 'repeat_rows': 192},
            'readout execution was incomplete')
    log = paths['original_log'].read_text()
    require('Object of type bool_ is not JSON serializable' in log, 'recovery reason is not the recorded scalar export error')
    completed = [json.loads(line) for line in log.splitlines() if line.startswith('{') and 'seed_complete' in line]
    require({(row['task'], row['seed_complete']) for row in completed} == {(t, s) for t in (4, 7) for s in feedback.SEEDS}, 'six task-seed completion records missing')
    parent = feedback.core.load_json(paths['parent_result'])
    require(parent['status'] == 'complete' and parent['scope']['demonstration_role'] == 'meta_support', 'recovery parent role differs')
    results = {}
    for task in (4, 7):
        expected = next(row for row in prefix['archives'] if row['path'] == str(paths[f'readouts{task}']))
        require(expected['sha256'] == sha(paths[f'readouts{task}']), 'recovery readout archive differs')
        models = {}
        with np.load(paths[f'readouts{task}'], allow_pickle=False) as cached, np.load(paths[f'support{task}'], allow_pickle=False) as reference:
            require(all(np.array_equal(cached[name], reference[name]) for name in ('demo_ids', 'action_ids', 'target_actions', 'command_masks', 'stratum_names')), 'recovery target or role-index parity differs')
            require(set(cached['demo_ids'].tolist()) == set(range(30)), 'recovery cache not source support')
            for seed in feedback.SEEDS:
                require(np.array_equal(cached[f'prediction_seed{seed}__original'], reference[f'seed{seed}__predictions']), 'recovery original prediction parity differs')
                variants = {variant: {name: feedback.expert.error_metrics(cached[f'prediction_seed{seed}__{variant}'], cached['target_actions'], cached['command_masks'][:, i])
                    for i, name in enumerate(cached['stratum_names'].tolist())} for variant in feedback.VARIANTS}
                checkpoint = parent['task_results'][str(task)]['models'][str(seed)]['checkpoint']
                require(checkpoint['optimizer_updates'] == 500 and checkpoint['arm'] == 'source_baseline' and sha(Path(checkpoint['path'])) == checkpoint['file_sha256'], 'recovery baseline checkpoint differs')
                models[str(seed)] = {'checkpoint': checkpoint, 'variants': variants}
        results[str(task)] = {'models': models, 'decision': feedback.classify(models)}
    result = {**prefix, 'status': 'complete', 'task_results': results,
        'recovery': {'kind': 'numpy_boolean_export_type_only', 'original_export_status': 'failed_serialization',
            'original_partial_result': str(paths['partial_result']), 'original_partial_sha256': sha(paths['partial_result']),
            'original_execution_runner_snapshot': str(paths['original_runner_snapshot']), 'corrected_metrics_runner_sha256': sha(paths['corrected_runner']),
            'recovery_config_sha256': sha(config_path), 'recovery_runner_sha256': sha(Path(__file__)),
            'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'elapsed_seconds': time.perf_counter() - started,
            'new_partial_readouts': 0, 'new_cnn_forwards': 0, 'new_optimizer_or_environment_steps': 0, 'inputs': config['inputs']}}
    json.dumps(result, allow_nan=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(); result = recover(args.config, args.output)
    with args.output.open('x') as f: json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'new_readouts': 0, 'output': str(args.output)}))


if __name__ == '__main__': main()
