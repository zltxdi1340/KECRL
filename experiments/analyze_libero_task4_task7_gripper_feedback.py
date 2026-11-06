"""Replace only public gripper inputs in fixed cached Policy readouts.

No CNN forward, observation collection, HDF5 read, optimizer or environment.
One readout chain is fusion -> per-frame projection -> action head -> tanh.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from experiments import analyze_libero_task4_task7_condition_separability as condition

core, expert, require, sha = condition.core, condition.expert, condition.require, condition.sha256_file
SEEDS, TASKS = condition.SEEDS, condition.TASKS
VARIANTS = ('original', 'hold_demo_t1_gripper', 'zero_gripper')
DEFAULT_CONFIG = ROOT / 'configs/libero-task4-task7-gripper-feedback-diagnostic_v2.json'
DEFAULT_OUTPUT = Path('/tmp/libero-task4-task7-gripper-feedback-20261102-v2.json')
DEFAULT_ARCHIVE = Path('/tmp/libero-task4-task7-gripper-feedback-artifacts-20261102-v2')
SCOPE = {'task_ids': list(TASKS), 'training_seeds': list(SEEDS), 'arm': 'source_baseline', 'checkpoint_updates': 500,
    'role': 'meta_support', 'demo_ids': list(range(30)), 'first_action_index': 1, 'device': 'cpu', 'torch_threads': 1,
    'partial_readout_main_chains': 2700, 'partial_readout_repeat_chains': 6,
    'partial_readout_main_rows': 77670, 'partial_readout_repeat_rows': 192,
    'new_full_policy_or_cnn_forwards': 0, 'new_model_backward_calls': 0, 'new_optimizer_updates': 0,
    'new_environment_interactions': 0, 'hdf5_payload_reads': 0, 'private_state_reads': 0,
    'qualification_accessed': False, 'validation_accessed': False, 'heldout_accessed': False, 'final_evaluation_accessed': False}
ANALYSIS = {'variants': list(VARIANTS), 'primary_variant': 'hold_demo_t1_gripper',
    'replacement': 'last_two_of_nine_proprio_dimensions_only; visual_and_joint_values_fixed',
    'hold_anchor': 'same_demo_first_scored_pre_action_t1_gripper; causal_past_no_future_or_expert_label_input',
    'zero_anchor': 'two_zeros_secondary_opposite_anchor_control_not_selectable_as_primary',
    'batch_order': 'exact_parent_source_batch_manifest; no_cross_demo_batches',
    'baseline_prediction_parity': 'bitwise_exact_parent_original_full_Policy_predictions',
    'metrics': 'original_uniform7_target_error_and_component_error_by_parent_strata',
    'primary_signal': 'open_window_gripper_mse_lower_all_three_seeds; sign_error_nonincrease_all_three',
    'translation_mean_ratio_max': 1.02, 'translation_each_seed_ratio_max': 1.05, 'uniform7_mean_ratio_max': 1.02,
    'hybrid_observation_is_not_physical_intervention': True, 'no_automatic_training_or_qualification': True}


def load_protocol(path):
    config = core.load_json(path)
    require(config.get('check_type') == 'libero_task4_task7_gripper_feedback_diagnostic' and config.get('formal_result') is False,
            'feedback protocol type differs')
    require(config.get('protocol_status') == 'frozen_cached_feature_feedback_diagnostic', 'feedback protocol is not frozen')
    require(config.get('scope') == SCOPE and config.get('analysis') == ANALYSIS, 'feedback frozen scope differs')
    require(config['runner_sha256'] == sha(Path(__file__)), 'feedback runner hash differs')
    require(set(config['inputs']) == {'parent_result', 'parent_config', 'parent_code', 'support4', 'support7', 'source_batches',
        'overnight_plan', 'policy_code', 'policy_config', 'offline_helpers', 'expert_code'}, 'feedback input roles differ')
    paths = {}
    for name, row in config['inputs'].items():
        p = Path(row['path']); p = p if p.is_absolute() else ROOT / p
        require(sha(p) == row['sha256'], f'feedback pin differs: {name}')
        paths[name] = p
    require(config['inputs']['policy_code']['sha256'] == condition.PINS['policy_code'] and
            config['inputs']['parent_code']['sha256'] == '58229e2684ad76bd9111b0c6099542b840380d12212d40eaa1c69943826929c7', 'feedback Policy/helper identity differs')
    return config, paths


def replace_gripper(proprio, variant, anchor):
    require(proprio.ndim == 2 and proprio.shape[1] == 9 and variant in VARIANTS, 'invalid feedback replacement')
    result = proprio.copy()
    if variant == 'hold_demo_t1_gripper':
        require(anchor.shape == (2,), 'feedback anchor must have two public gripper coordinates')
        result[:, 7:] = anchor
    elif variant == 'zero_gripper': result[:, 7:] = 0
    require(np.array_equal(result[:, :7], proprio[:, :7]), 'feedback replacement changed joint inputs')
    return result


@torch.no_grad()
def cached_readout(model, visual, proprio):
    require(model.config.memory_mode == 'cnn_only' and visual.shape == (len(proprio), 256) and proprio.shape == (len(proprio), 9),
            'feedback cached features require original CNN-only interfaces')
    values = torch.cat((torch.from_numpy(visual), torch.from_numpy(proprio)), dim=-1).unsqueeze(1)
    fused = model.fusion(values)
    projection, hidden = model.recurrent(fused, hidden=None)
    require(hidden is None, 'cached projection unexpectedly recurrent')
    output = torch.tanh(model.action_head(projection))[:, 0].numpy().copy()
    require(np.isfinite(output).all() and all(p.grad is None for p in model.parameters()), 'feedback output/gradient malformed')
    return output


def classify(models):
    out = {}
    for variant in VARIANTS[1:]:
        mse_changes, sign_changes, ratios_t, ratios_a = {}, {}, {}, {}
        for seed in SEEDS:
            original = models[str(seed)]['variants']['original']
            candidate = models[str(seed)]['variants'][variant]
            mse_changes[str(seed)] = candidate['around_open_transition']['component_mse']['gripper'] - original['around_open_transition']['component_mse']['gripper']
            sign_changes[str(seed)] = candidate['around_open_transition']['gripper']['sign_error_rate'] - original['around_open_transition']['gripper']['sign_error_rate']
            ratios_t[str(seed)] = candidate['all']['component_mse']['translation'] / original['all']['component_mse']['translation']
            ratios_a[str(seed)] = candidate['all']['action_mse'] / original['all']['action_mse']
        original_t = np.mean([models[str(seed)]['variants']['original']['all']['component_mse']['translation'] for seed in SEEDS])
        candidate_t = np.mean([models[str(seed)]['variants'][variant]['all']['component_mse']['translation'] for seed in SEEDS])
        original_a = np.mean([models[str(seed)]['variants']['original']['all']['action_mse'] for seed in SEEDS])
        candidate_a = np.mean([models[str(seed)]['variants'][variant]['all']['action_mse'] for seed in SEEDS])
        out[variant] = {'open_gripper_mse_delta_by_seed': mse_changes, 'open_sign_error_delta_by_seed': sign_changes,
            'all_translation_ratio_by_seed': ratios_t, 'all_uniform7_ratio_by_seed': ratios_a,
            'translation_seed_mean_ratio': float(candidate_t / original_t), 'uniform7_seed_mean_ratio': float(candidate_a / original_a),
            'all_three_local_mse_improved': all(value < 0 for value in mse_changes.values()),
            'all_three_local_sign_error_not_increased': all(value <= 0 for value in sign_changes.values()),
            'nonregression': bool(candidate_t / original_t <= 1.02 and all(value <= 1.05 for value in ratios_t.values()) and candidate_a / original_a <= 1.02)}
    return {'primary_variant': 'hold_demo_t1_gripper', 'variants': out, 'training_or_deployment_promoted': False,
            'physical_failure_cause_established': False}


def run(config_path, output, archive_dir):
    require(not output.exists() and not archive_dir.exists(), 'feedback output/archive already exists')
    config, paths = load_protocol(config_path)
    started = time.perf_counter()
    parent = core.load_json(paths['parent_result'])
    require(parent['status'] == 'complete' and parent['scope']['demonstration_role'] == 'meta_support' and
            parent['scope']['checkpoint_optimizer_updates'] == 500, 'feedback parent not source baseline500')
    batches = [json.loads(line) for line in paths['source_batches'].read_text().splitlines()]
    policy = core.load_json(paths['policy_config'])['policy']
    torch.set_num_threads(1); torch.manual_seed(20261022)
    archive_dir.mkdir(parents=True, exist_ok=False)
    counters = {'main_chains': 0, 'repeat_chains': 0, 'main_rows': 0, 'repeat_rows': 0}
    task_results, archives = {}, []
    for task in TASKS:
        require(parent['task_results'][str(task)]['source_access']['demonstration_indices'] == list(range(30)), 'feedback demonstration role differs')
        task_batches = [row for row in batches if row['task_id'] == task]
        with np.load(paths[f'support{task}'], allow_pickle=False) as source:
            demos, ids, targets = source['demo_ids'], source['action_ids'], source['target_actions']
            require(set(demos.tolist()) == set(range(30)), 'feedback cache is not source support')
            saved = {'demo_ids': demos, 'action_ids': ids, 'target_actions': targets, 'command_masks': source['command_masks'], 'stratum_names': source['stratum_names']}
            models = {}
            for seed in SEEDS:
                checkpoint = parent['task_results'][str(task)]['models'][str(seed)]['checkpoint']
                model = core.load_policy(checkpoint, {**policy, 'memory_mode': 'cnn_only'})
                before = core._digest_weights(model.state_dict())
                visual, proprio = source[f'seed{seed}__visual_concat'], source[f'seed{seed}__proprio']
                before_visual, before_proprio = core._digest_array(visual), core._digest_array(proprio)
                predictions = {variant: np.empty_like(targets) for variant in VARIANTS}
                covered = np.zeros(len(targets), np.int64)
                for batch_index, batch in enumerate(task_batches):
                    demo = batch['demo']
                    selected = np.flatnonzero((demos == demo) & (ids >= batch['target_start']) & (ids < batch['target_stop']))
                    require(len(selected) == batch['target_stop'] - batch['target_start'], 'feedback source batch indices differ')
                    first = np.flatnonzero((demos == demo) & (ids == 1))
                    require(len(first) == 1, 'feedback causal initial anchor missing')
                    anchor = proprio[first[0], 7:]
                    for variant in VARIANTS:
                        p = replace_gripper(proprio[selected], variant, anchor)
                        readout = cached_readout(model, visual[selected], p)
                        counters['main_chains'] += 1; counters['main_rows'] += len(selected)
                        predictions[variant][selected] = readout
                        if variant == 'original':
                            require(np.array_equal(readout, source[f'seed{seed}__predictions'][selected]), 'cached partial readout baseline prediction parity differs')
                            if batch_index == 0:
                                repeated = cached_readout(model, visual[selected], p)
                                require(np.array_equal(readout, repeated), 'feedback repeated partial readout differs')
                                counters['repeat_chains'] += 1; counters['repeat_rows'] += len(selected)
                    covered[selected] += 1
                require(np.all(covered == 1), 'feedback source row coverage differs')
                require(before == core._digest_weights(model.state_dict()) == checkpoint['weight_sha256'] and
                        core._digest_array(visual) == before_visual and core._digest_array(proprio) == before_proprio, 'feedback probe changed weights or cached input')
                metrics = {}
                for variant, prediction in predictions.items():
                    saved[f'prediction_seed{seed}__{variant}'] = prediction
                    metrics[variant] = {name: expert.error_metrics(prediction, targets, source['command_masks'][:, i])
                                       for i, name in enumerate(source['stratum_names'].tolist())}
                models[str(seed)] = {'checkpoint': checkpoint, 'variants': metrics}
                print(json.dumps({'task': task, 'seed_complete': seed, 'partial_readouts': counters}), flush=True)
            path = archive_dir / f'task{task}-gripper-feedback-readouts.npz'
            digest = expert.save_archive(path, saved); archives.append({'path': str(path), 'sha256': digest})
            task_results[str(task)] = {'models': models, 'decision': classify(models)}
    require(counters == {'main_chains': 2700, 'repeat_chains': 6, 'main_rows': 77670, 'repeat_rows': 192}, 'feedback actual cached compute budget differs')
    return {'check_type': config['check_type'], 'status': 'complete', 'formal_result': False, 'scope': SCOPE, 'analysis': ANALYSIS,
        'runtime': {'elapsed_seconds': time.perf_counter() - started, 'actual_partial_readouts': counters, 'new_optimizer_or_environment_steps': 0,
                    'python': sys.version.split()[0], 'torch': torch.__version__, 'numpy': np.__version__, 'device': 'cpu', 'torch_threads': torch.get_num_threads()},
        'provenance': {'config_path': str(config_path), 'config_sha256': sha(config_path), 'runner_sha256': sha(Path(__file__)),
            'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'inputs': config['inputs']},
        'task_results': task_results, 'archives': archives,
        'interpretation': {'input_replacement_is_model_sensitivity_not_environment_cause': True, 'hybrid_observations_may_be_off_distribution': True,
                           'visual_gripper_feedback_was_not_removed': True}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--archive-dir', type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args(); result = run(args.config, args.output, args.archive_dir)
    with args.output.open('x') as f: json.dump(result, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({'status': result['status'], 'output': str(args.output)}))


if __name__ == '__main__': main()
