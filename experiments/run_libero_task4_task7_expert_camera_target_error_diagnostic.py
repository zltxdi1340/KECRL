"""Execute the frozen source-support camera/target-error design without training."""
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

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments import validate_libero_task4_task7_expert_camera_target_error_protocol as protocol
from experiments import analyze_libero_task4_task7_expert_action_error_diagnostic as expert

core, require, sha = expert.core, protocol.require, protocol.sha
DESIGN_SHA = '67654a94e8e41497fa24b0929849f2d9535d9b5ae518bd56c39a863182acfdb4'
OPERATOR_SHA = '681a3390bc894e753e945fa0bdab6b79b5dd4ee2a60cc2fccb9e18fb10b7d834'
DEFAULT_BINDING = ROOT / 'configs/libero-task4-task7-expert-camera-target-error-execution-binding_v1.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-expert-camera-target-error-20261102-v1.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-expert-camera-target-error-artifacts-20261102-v1')
ENVIRONMENT = {'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
EXPECTED = {'checkpoint_load_attempts': 6, 'checkpoint_loads': 6, 'source_files_opened': 2,
    'support_demonstrations_read': 60, 'source_action_payload_rows': 8690, 'source_observation_rows': 8630,
    't0_rows_excluded': 60, 'source_batch_digest_exact': 300, 'target_rows_exact': 8630,
    'command_mask_rows_exact': 8630, 'previous_gripper_rows_exact': 8630,
    'main_forward_attempts': 7200, 'main_forward_calls': 7200, 'main_prediction_rows': 207120,
    'repeat_forward_attempts': 48, 'repeat_forward_calls': 48, 'repeat_prediction_rows': 1536,
    'original_prediction_batch_exact': 900, 'uint8_float_identity_batch_exact': 900, 'repeat_prediction_batch_exact': 48,
    'pixel_matching_rows': 8630, 'weight_and_grad_demo_checks': 180, 'models_weights_unchanged': 6,
    'completed_task_archives': 2, 'new_optimizer_updates': 0, 'new_backward_calls': 0,
    'new_environment_resets_or_steps': 0, 'new_initial_state_or_private_state_reads': 0}


def utc():
    return datetime.now(timezone.utc).isoformat()


def counters():
    return dict.fromkeys(EXPECTED, 0)


def load_binding(path):
    binding = protocol.load_json(path)
    require(binding.get('check_type') == 'libero_expert_camera_target_error_execution_binding' and
        binding.get('formal_result') is False and binding.get('protocol_status') == 'execution_frozen_before_readouts',
        'execution binding type/status differs')
    require(binding.get('this_design_sha256') == DESIGN_SHA and binding.get('operator_validator_sha256') == OPERATOR_SHA and
        binding.get('execution_runner_sha256') == sha(Path(__file__)), 'execution code/design pin differs')
    require(binding.get('launch_environment') == ENVIRONMENT and binding.get('expected_actual_calls') == EXPECTED,
        'execution environment or budget differs')
    require({k: os.environ.get(k) for k in ENVIRONMENT} == ENVIRONMENT, 'actual launch environment differs')
    design_path = ROOT / binding['scientific_design_path']
    require(sha(design_path) == DESIGN_SHA and sha(Path(protocol.__file__)) == OPERATOR_SHA, 'frozen scientific design/operator changed')
    validation = protocol.validate(design_path)
    cfg = protocol.load_json(design_path)
    require(binding['runtime_versions'] == {'python': platform.python_version(), 'torch': torch.__version__,
        'numpy': np.__version__, 'h5py': h5py.__version__, 'mkldnn_enabled': torch.backends.mkldnn.enabled},
        'runtime versions/backend differ from bound execution')
    return binding, cfg, validation


def statistic(values):
    v = np.asarray(values, np.float64)
    require(v.ndim == 1 and np.isfinite(v).all(), 'nonfinite or malformed summary')
    return {'n': len(v), 'missing_values': 0, 'mean': float(v.mean()) if len(v) else None,
        'median': float(np.median(v)) if len(v) else None, 'minimum': float(v.min()) if len(v) else None,
        'maximum': float(v.max()) if len(v) else None}


def summarize_predictions(predictions, target, masks, demo_ids):
    require(predictions.shape == (8, len(target), 7) and predictions.dtype == np.float32 and np.isfinite(predictions).all(),
        'complete eight-condition predictions required')
    require(masks.shape == (len(target), 7) and masks.dtype == np.bool_, 'command strata differ')
    clean = predictions[0]
    terms, groups, per_demo, demo_equal = {}, {}, {}, {}
    residuals = {'paired_error': 0., 'uniform7': 0.}
    for index, condition in enumerate(protocol.CONDITIONS):
        terms[condition] = {component: protocol.error_terms(clean, predictions[index], target, component)
                            for component in protocol.COMPONENTS}
        current = terms[condition]
        for component, values in current.items():
            residuals['paired_error'] = max(residuals['paired_error'], float(np.max(np.abs(
                values['DeltaMSE'] - values['error_alignment_cross_term'] - values['response_energy_term']))))
        for measure in protocol.ANALYSIS['per_row_terms']:
            weighted = (3 * current['translation'][measure] + 3 * current['rotation'][measure] + current['gripper'][measure]) / 7
            residuals['uniform7'] = max(residuals['uniform7'], float(np.max(np.abs(current['all7'][measure] - weighted))))
        groups[condition] = {}
        for column, stratum in enumerate(protocol.ANALYSIS['strata']):
            selected = masks[:, column]
            groups[condition][stratum] = {'valid_steps': int(selected.sum()), 'unselected_steps': int((~selected).sum()),
                'error_metrics': expert.error_metrics(predictions[index], target, selected),
                'terms': {component: {measure: statistic(v[selected]) for measure, v in values.items()}
                          for component, values in current.items()}}
        per_demo[condition] = [{'demo_index': int(demo), 'valid_steps': int(np.sum(demo_ids == demo)),
            'terms': {component: {measure: statistic(v[demo_ids == demo]) for measure, v in values.items()}
                      for component, values in current.items()}} for demo in sorted(set(demo_ids.tolist()))]
        demo_equal[condition] = {component: {measure: statistic([row['terms'][component][measure]['mean']
            for row in per_demo[condition]]) for measure in protocol.ANALYSIS['per_row_terms']} for component in protocol.COMPONENTS}
    require(max(residuals.values()) <= 1e-12, 'error/component identity residual differs')
    contrasts = {}
    for component in protocol.COMPONENTS:
        contrasts[component] = {
            f'{view}_spatial_minus_photo': statistic(terms[f'{view}_spatial_matched'][component]['DeltaMSE'] -
                terms[f'{view}_photo_matched'][component]['DeltaMSE']) for view in ('eye_in_hand', 'agentview')}
        for kind in ('photo', 'spatial'):
            contrasts[component][f'{kind}_hand_minus_agent'] = statistic(
                terms[f'eye_in_hand_{kind}_matched'][component]['DeltaMSE'] - terms[f'agentview_{kind}_matched'][component]['DeltaMSE'])
    return {'by_condition_stratum': groups, 'per_demo': per_demo, 'demo_equal': demo_equal,
            'paired_contrasts': contrasts, 'max_identity_residuals': residuals}, terms


def cue_flags(models):
    require(set(models) == set(map(str, protocol.SEEDS)), 'all three fixed seeds required')
    flags = {}
    def positive(value):
        return value is not None and np.isfinite(value) and value > 0
    for component in ('translation', 'all7'):
        flags[component] = {}
        for camera in ('eye_in_hand', 'agentview'):
            rows = []
            for seed in protocol.SEEDS:
                result = models[str(seed)]['statistics']
                rows.append({'training_seed': seed,
                    'photo_DeltaMSE': result['by_condition_stratum'][f'{camera}_photo_matched']['all']['terms'][component]['DeltaMSE']['mean'],
                    'spatial_minus_photo_DeltaMSE': result['paired_contrasts'][component][f'{camera}_spatial_minus_photo']['mean'],
                    'full_spatial_DeltaMSE': result['by_condition_stratum'][f'{camera}_spatial_full']['all']['terms'][component]['DeltaMSE']['mean']})
            flags[component][camera] = {'per_seed': rows,
                'appearance_error_amplification_cue': all(positive(r['photo_DeltaMSE']) for r in rows),
                'spatial_detail_dependence_cue': all(positive(r['spatial_minus_photo_DeltaMSE']) and positive(r['full_spatial_DeltaMSE']) for r in rows)}
    return flags


def forward_counted(model, observations, metadata, progress, event, repeated=False, predict=None):
    kind = 'repeat' if repeated else 'main'
    limit = 48 if repeated else 7200
    require(progress[f'{kind}_forward_attempts'] < limit, f'{kind} forward budget exhausted')
    progress[f'{kind}_forward_attempts'] += 1
    event('forward_attempt', readout_kind=kind, **metadata)
    predict = predict or (lambda m, obs: expert.predict_shared_batch({'fixed': m}, obs)['fixed'])
    values = predict(model, observations)
    progress[f'{kind}_forward_calls'] += 1
    progress[f'{kind}_prediction_rows'] += len(values)
    event('forward_complete', readout_kind=kind, prediction_sha256=core._digest_array(values), **metadata)
    return values


def execute_batch(models, observations, references, progress, event, metadata, sink, repeat=False, prepared=None):
    before = expert.digest_batch(observations)
    variants, diagnostics = protocol.prepare_interventions(observations)
    n = len(observations['joint_states'])
    diagnostics['full_spatial_realized_pixel_mae'] = {}
    for view, prefix in (('eye_in_hand_rgb', 'eye_in_hand'), ('agentview_rgb', 'agentview')):
        diagnostics['full_spatial_realized_pixel_mae'][prefix] = np.abs(
            variants[f'{prefix}_spatial_full'][view].astype(np.float64) - variants['identity_float'][view]).mean(axis=(1, 2, 3)).tolist()
    if prepared is not None:
        prepared(diagnostics)
    progress['pixel_matching_rows'] += n
    for index, condition in enumerate(protocol.CONDITIONS):
        for seed in protocol.SEEDS:
            values = forward_counted(models[seed], variants[condition], {**metadata, 'training_seed': seed,
                'condition': condition, 'rows': n}, progress, event)
            sink(seed, index, values, False)
            if index == 0:
                require(np.array_equal(values, references[seed]), 'original prediction parity failed')
                progress['original_prediction_batch_exact'] += 1
            elif index == 1:
                require(np.array_equal(values, references[seed]), 'uint8/float identity prediction parity failed')
                progress['uint8_float_identity_batch_exact'] += 1
    if repeat:
        for index, condition in enumerate(protocol.CONDITIONS):
            for seed in protocol.SEEDS:
                values = forward_counted(models[seed], variants[condition], {**metadata, 'training_seed': seed,
                    'condition': condition, 'rows': n}, progress, event, repeated=True)
                sink(seed, index, values, True)
    require(expert.digest_batch(observations) == before, 'original public observations mutated')
    return diagnostics


def save_archive(path, arrays):
    with path.open('xb') as f:
        np.savez_compressed(f, **arrays)
    return {'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size}


def load_support_reference(row):
    record = row['archive']
    require(sha(Path(record['path'])) == record['sha256'], 'source support reference file differs')
    keys = ['target_actions', 'previous_expert_gripper', 'demonstration_indices', 'action_indices', 'command_masks', 'mask_names']
    keys += [f'prediction_{expert.model_key(seed, "source_baseline")}' for seed in protocol.SEEDS]
    with np.load(record['path'], allow_pickle=False) as data:
        arrays = {k: data[k] for k in keys}
    require(arrays['mask_names'].tolist() == protocol.ANALYSIS['strata'], 'reference command strata differ')
    require(arrays['target_actions'].shape == (sum(n - 1 for n in row['source_action_lengths']), 7), 'support target row count differs')
    for demo, length in enumerate(row['source_action_lengths']):
        selected = arrays['demonstration_indices'] == demo
        require(np.array_equal(arrays['action_indices'][selected], np.arange(1, length)), 'reference excludes wrong t0/action indices')
    require(set(arrays['demonstration_indices'].tolist()) == set(range(30)), 'reference support role differs')
    return arrays


def run(binding_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), 'execution output/archive already exists')
    started, start_utc = time.perf_counter(), utc()
    archive_dir.mkdir(parents=True, exist_ok=False)
    events = archive_dir / 'events.jsonl'
    def event(name, **fields):
        with events.open('a') as f:
            f.write(json.dumps({'event': name, 'utc': utc(), **fields}, sort_keys=True) + '\n')
    progress, tasks, files, failure = counters(), {}, [], None
    binding, cfg, preflight = None, None, None
    event('start', binding_path=str(binding_path))
    try:
        binding, cfg, preflight = load_binding(binding_path)
        event('binding_verified', binding_sha256=sha(binding_path), design_sha256=DESIGN_SHA)
        torch.set_num_threads(1)
        torch.manual_seed(protocol.SCOPE['analysis_seed'])
        np.random.seed(protocol.SCOPE['analysis_seed'])
        paths = {k: ROOT / v['path'] for k, v in cfg['inputs'].items()}
        transfer, manifest = protocol.load_json(paths['transfer']), protocol.load_json(paths['dataset_manifest'])
        parent = protocol.load_json(paths['expert_result'])
        policy_spec = {**protocol.load_json(paths['policy_config'])['policy'], 'memory_mode': 'cnn_only'}
        for task in protocol.TASKS:
            access = expert.authorize_role(transfer, manifest, task, 'meta_support')
            row = next(r for r in cfg['support_references'] if r['task_id'] == task)
            require(access.demonstration_indices == tuple(range(30)) and row['role'] == 'meta_support', 'support authorization differs')
            dataset = Path(row['dataset']['path'])
            require(sha(dataset) == row['dataset']['sha256'], 'source HDF5 file hash differs')
            reference = load_support_reference(row)
            ref_digest = {k: core._digest_array(v) for k, v in reference.items()}
            total = len(reference['target_actions'])
            saved = {k: v for k, v in reference.items() if not k.startswith('prediction_')}
            saved['condition_names'] = np.array(protocol.CONDITIONS, dtype='U64')
            predictions = {seed: np.full((8, total, 7), np.nan, np.float32) for seed in protocol.SEEDS}
            valid = {seed: np.zeros((8, total), np.bool_) for seed in protocol.SEEDS}
            for seed in protocol.SEEDS:
                saved[f'predictions_seed{seed}'] = predictions[seed]
                saved[f'prediction_valid_seed{seed}'] = valid[seed]
            task_record = {'status': 'started', 'access': access.to_dict(), 'dataset': row['dataset'],
                'support_reference': row['archive'], 'models': {}, 'batch_records': []}
            tasks[str(task)] = task_record
            transform_log = archive_dir / f'task{task}-transforms.jsonl'
            try:
                models = {}
                checkpoints = {}
                for seed in protocol.SEEDS:
                    cp = next(m['checkpoint'] for m in cfg['models'] if (m['task_id'], m['training_seed']) == (task, seed))
                    progress['checkpoint_load_attempts'] += 1
                    models[seed] = core.load_policy(cp, policy_spec)
                    progress['checkpoint_loads'] += 1
                    checkpoints[seed] = cp
                    task_record['models'][str(seed)] = {'checkpoint': cp, 'status': 'loaded'}
                offsets = np.r_[0, np.cumsum(np.array(row['source_action_lengths']) - 1)]
                batch_manifest = parent['task_results'][str(task)]['roles']['meta_support']['shared_input_batches']
                with h5py.File(dataset, 'r') as handle:
                    progress['source_files_opened'] += 1
                    for demo in access.demonstration_indices:
                        group = handle['data'][f'demo_{demo}']
                        stored = np.asarray(group['actions'])
                        require(stored.shape == (row['source_action_lengths'][demo], 7) and np.issubdtype(stored.dtype, np.floating), 'source actions shape/dtype differs')
                        require(np.isfinite(stored).all() and np.all(np.abs(stored) <= 1), 'invalid source actions')
                        actions = stored.astype(np.float32)
                        progress['source_action_payload_rows'] += len(actions)
                        progress['support_demonstrations_read'] += 1
                        progress['t0_rows_excluded'] += 1
                        start_row, stop_row = int(offsets[demo]), int(offsets[demo + 1])
                        target = reference['target_actions'][start_row:stop_row]
                        require(np.array_equal(actions[1:], target), 'original expert targets differ')
                        progress['target_rows_exact'] += len(target)
                        require(np.array_equal(actions[:-1, 6], reference['previous_expert_gripper'][start_row:stop_row]), 'previous expert command differs')
                        progress['previous_gripper_rows_exact'] += len(target)
                        mask_data = expert.command_masks(actions)
                        command_masks = np.stack([mask_data[k][1:] for k in protocol.ANALYSIS['strata']], axis=1)
                        require(np.array_equal(command_masks, reference['command_masks'][start_row:stop_row]), 'expert command masks differ')
                        progress['command_mask_rows_exact'] += len(target)
                        action_digest = core._digest_array(actions)
                        for old in [b for b in batch_manifest if b['demonstration_index'] == demo]:
                            start, stop = old['target_start'], old['target_stop_exclusive']
                            observations = expert.read_causal_batch(group, actions, start, stop)
                            progress['source_observation_rows'] += stop - start
                            require(expert.digest_batch(observations) == old['shared_observation_sha256'], 'original shared input batch digest differs')
                            progress['source_batch_digest_exact'] += 1
                            left, right = start_row + start - 1, start_row + stop - 1
                            metadata = {'task_id': task, 'demonstration_index': demo, 'target_start': start, 'target_stop_exclusive': stop}
                            def sink(seed, condition_index, values, repeated):
                                if repeated:
                                    saved[f'repeat_seed{seed}_condition{condition_index}'] = values.copy()
                                    require(np.array_equal(values, predictions[seed][condition_index, left:right]), 'repeated condition prediction differs')
                                    progress['repeat_prediction_batch_exact'] += 1
                                else:
                                    predictions[seed][condition_index, left:right] = values
                                    valid[seed][condition_index, left:right] = True
                            references = {seed: reference[f'prediction_{expert.model_key(seed, "source_baseline")}'][left:right] for seed in protocol.SEEDS}
                            def prepared(diagnostics):
                                with transform_log.open('a') as f:
                                    f.write(json.dumps({**old, 'first_archive_row': left, 'stop_archive_row': right,
                                        'stage': 'prepared_before_policy_calls', 'diagnostics': diagnostics}, sort_keys=True) + '\n')
                            execute_batch(models, observations, references, progress, event, metadata, sink,
                                repeat=(demo == 0 and start == 1), prepared=prepared)
                            task_record['batch_records'].append(old)
                            event('source_batch_complete', **metadata, original_observation_sha256=old['shared_observation_sha256'])
                        require(core._digest_array(actions) == action_digest, 'original expert actions mutated')
                        for seed, model in models.items():
                            require(core._digest_weights(model.state_dict()) == checkpoints[seed]['weight_sha256'] and
                                all(parameter.grad is None for parameter in model.parameters()), 'fixed weights or gradients changed')
                            progress['weight_and_grad_demo_checks'] += 1
                        print(json.dumps({'task': task, 'demo_complete': demo, 'main_forward_calls': progress['main_forward_calls'],
                            'repeat_forward_calls': progress['repeat_forward_calls']}), flush=True)
                require(task_record['batch_records'] == batch_manifest, 'source batch order/count differs')
                require(all(core._digest_array(v) == ref_digest[k] for k, v in reference.items()), 'reference arrays mutated')
                for seed in protocol.SEEDS:
                    require(valid[seed].all(), 'incomplete prediction rows')
                    statistics, terms = summarize_predictions(predictions[seed], reference['target_actions'], reference['command_masks'], reference['demonstration_indices'])
                    task_record['models'][str(seed)].update(status='complete', statistics=statistics,
                        weight_sha256_unchanged=checkpoints[seed]['weight_sha256'])
                    for condition, components in terms.items():
                        for component, measures in components.items():
                            for measure, values in measures.items():
                                saved[f'terms_seed{seed}_{condition}_{component}_{measure}'] = values
                    progress['models_weights_unchanged'] += 1
                task_record['descriptive_flags'] = cue_flags(task_record['models'])
                task_record['status'] = 'complete'
                progress['completed_task_archives'] += 1
                del models
            finally:
                record = save_archive(archive_dir / f'task{task}-predictions-and-error-terms.npz', saved)
                task_record['archive'] = record
                files.append(record)
                if transform_log.exists():
                    task_record['transform_log'] = {'path': str(transform_log), 'sha256': sha(transform_log)}
            event('task_complete', task_id=task)
        require(progress == EXPECTED, f'actual execution budget differs: {progress}')
    except Exception as exc:
        failure = {'type': type(exc).__name__, 'message': str(exc)}
        event('failure', **failure)
    event('finished', status='failed' if failure else 'complete', actual_calls=progress)
    result = {'check_type': 'libero_expert_camera_target_error_diagnostic', 'status': 'failed' if failure else 'complete',
        'failure': failure, 'formal_result': False, 'research_run_executed': True, 'training_run': False,
        'scope': protocol.SCOPE, 'analysis': protocol.ANALYSIS, 'task_results': tasks, 'archives': files,
        'provenance': {'binding_path': str(binding_path), 'binding_sha256': sha(binding_path) if binding_path.exists() else None,
            'scientific_design_sha256': DESIGN_SHA, 'operator_validator_sha256': OPERATOR_SHA,
            'execution_runner_sha256': sha(Path(__file__)), 'design_validation': preflight,
            'scientific_design_inputs': cfg['inputs'] if cfg else {}, 'event_log': {'path': str(events), 'sha256': sha(events)},
            'started_at_utc': start_utc, 'finished_at_utc': utc(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()},
        'runtime': {'device': 'cpu', 'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'h5py': h5py.__version__, 'torch_threads': torch.get_num_threads(), 'mkldnn_enabled': torch.backends.mkldnn.enabled,
            'launch_environment': {k: os.environ.get(k) for k in ENVIRONMENT}, 'actual_calls': progress,
            'elapsed_seconds': time.perf_counter() - started},
        'interpretation': {'diagnostic_only': True, 'source_support_fit_domain_only': True, 'corruptions_not_new_expert_labels': True,
            'spatial_probe_not_geometric_necessity_proof': True, 'no_closed_loop_or_qualification_result': True,
            'no_training_or_rollout_promotion': True}}
    with output.open('x') as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write('\n')
    if failure:
        raise RuntimeError(f'failed execution and partial artifacts preserved: {failure}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    result = run(args.binding.resolve(), args.output.resolve(), args.archive_dir.resolve())
    print(json.dumps({'status': result['status'], 'runtime': result['runtime']}))


if __name__ == '__main__':
    main()
