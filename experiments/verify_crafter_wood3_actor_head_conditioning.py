"""Reload all saved heads without fitting; audit features, metrics, and folding."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from experiments.crafter_actor_head_ablation import train_standardizer
from experiments.crafter_actor_only_local import action_metrics, evaluate_actor, fixture_action_targets
from experiments.crafter_spatial_readout_data import array_digest
from experiments.run_crafter_spatial_training_determinism_audit import _state_equal
from experiments.run_crafter_wood3_actor_head_fresh import load_saved_head
from experiments.run_crafter_wood3_collection_opportunity import _read, _sha256, _write
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


def _near(a, b, tolerance=3e-4):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_near(a[k], b[k], tolerance) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(_near(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, float):
        return bool(np.isclose(a, b, atol=tolerance, rtol=tolerance))
    return a == b


def _encode(source, images):
    batches = []
    with torch.no_grad():
        for start in range(0, len(images), 64):
            values = torch.as_tensor(images[start:start + 64].transpose(0, 3, 1, 2).copy(), dtype=torch.float32) / 255
            batches.append(source.encoder(values).cpu().numpy())
    return np.concatenate(batches)


def verify(paired_root, fresh_root, output):
    config, paired = _read(paired_root / 'config.json'), _read(paired_root / 'summary.json')
    fresh = _read(fresh_root / 'summary.json')
    prior_root = Path(config['source_actor_diagnostic_root'])
    prior = _read(prior_root / 'summary.json')
    datasets = {}
    for name, root in (('old', prior_root), ('fresh', fresh_root)):
        with np.load(root / 'scene_dataset.npz') as values:
            datasets[name] = dict(values)
    checks, statistics, hashes = {}, [], {}
    for seed in config['seed_set']:
        seed_fits = [f for f in paired['fits'] if f['policy_seed'] == seed]
        checkpoint_path = Path(seed_fits[0]['source_checkpoint'])
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        source = build_matched_spatial_policy(config['policy'], 'cnn_only')
        source.load_state_dict(checkpoint['policy'], strict=True)
        source.optimizer.load_state_dict(checkpoint['optimizer'])
        source.eval()
        for p in source.parameters():
            p.requires_grad_(False)
            p.grad = None
        before = copy.deepcopy((source.state_dict(), source.optimizer.state_dict()))
        checks[f'seed{seed}_source_configuration'] = (checkpoint['config']['policy'] == config['policy']
            and checkpoint['actual_interaction_steps'] == 100000
            and checkpoint['environment_order_version'] == fresh['environment_order_version'])
        features = {}
        for name, root, summary in (('old', prior_root, prior), ('fresh', fresh_root, fresh)):
            with np.load(root / f'seed{seed}_features.npz') as values:
                arrays = dict(values)
            features[name] = arrays['embedding']
            checks[f'seed{seed}_{name}_feature_integrity'] = (array_digest(arrays) == summary['features'][str(seed)]['canonical_digest']
                and _sha256(root / f'seed{seed}_features.npz') == summary['features'][str(seed)]['npz_sha256'])
            regenerated = _encode(source, datasets[name]['images'])
            checks[f'seed{seed}_{name}_features_from_rgb_encoder'] = bool(np.allclose(regenerated, features[name], atol=2e-5, rtol=2e-5))
        masks = {name: datasets['old']['split_code'] == code for code, name in enumerate(('train', 'validation', 'heldout'))}
        mean, std = train_standardizer(features['old'][masks['train']], config['feature_std_floor'])
        source_weight, source_bias = source.actor.weight.detach(), source.actor.bias.detach()
        values = torch.as_tensor(features['old'])
        z = (values - torch.as_tensor(mean)) / torch.as_tensor(std)
        with torch.no_grad():
            initial_raw = source.actor(values)[:, :7]
            initial_z = (z @ (source_weight * torch.as_tensor(std)).T
                         + source_bias + (source_weight * torch.as_tensor(mean)).sum(dim=1))[:, :7]
        checks[f'seed{seed}_initial_function_equivalent'] = torch.allclose(initial_raw, initial_z, atol=2e-5, rtol=2e-5)
        for variant in ('baseline', 'raw_linear', 'standardized_linear', 'mlp64'):
            prefix = f'seed{seed}_{variant}_'
            if variant == 'baseline':
                policy = source
            else:
                path = paired_root / f'seed{seed}_{variant}.pt'
                artifact = torch.load(path, map_location='cpu', weights_only=False)
                hashes[str(path)] = _sha256(path)
                fit = next(f for f in seed_fits if f['variant'] == variant)
                checks[prefix + 'artifact_matches_summary'] = {k: v for k, v in artifact.items() if k != 'head_state'} == fit
                checks[prefix + 'source_sha'] = _sha256(checkpoint_path) == artifact['source_checkpoint_sha256']
                checks[prefix + 'nonformal_teacher_scope'] = (artifact['formal_result'] is False and artifact['teacher_used'] is True
                    and artifact['teacher_scope'] == config['teacher_scope'] and artifact['selection_uses_heldout'] is False)
                checks[prefix + 'head_only_no_ppo_state'] = not ({'optimizer', 'policy', 'rng'} & artifact.keys())
                points = [(point['validation_balanced_accuracy'], trial['learning_rate'], point['epoch'])
                          for trial in fit['trials'] for point in trial['history']]
                best = max(p[0] for p in points)
                winner = next(p for p in points if p[0] == best)
                checks[prefix + 'validation_only_selection_recomputed'] = (winner == (fit['selection_score'], fit['selected_learning_rate'], fit['selected_epoch'])
                    and fit['supervised_optimizer_updates'] == len(config['actor_learning_rates']) * config['actor_fit_epochs'])
                policy = load_saved_head(source, artifact)
                if variant == 'standardized_linear':
                    checks[prefix + 'train_only_standardizer_exact'] = (np.array_equal(mean, np.asarray(artifact['standardizer_mean'], dtype=np.float32))
                        and np.array_equal(std, np.asarray(artifact['standardizer_std'], dtype=np.float32)))
                    with torch.no_grad():
                        raw_logits = policy.actor(values)[:, :7]
                        # Reconstruct standardized coordinates from the persisted raw affine head.
                        wz = policy.actor.weight * torch.as_tensor(std)
                        bz = policy.actor.bias + (policy.actor.weight * torch.as_tensor(mean)).sum(dim=1)
                        z_logits = (z @ wz.T + bz)[:, :7]
                    max_difference = float((raw_logits - z_logits).abs().max())
                    checks[prefix + 'folded_function_numerical_equivalence'] = torch.allclose(raw_logits, z_logits, atol=5e-3, rtol=5e-4)
                else:
                    max_difference = None
                if variant != 'mlp64':
                    checks[prefix + 'masked_rows_preserved'] = all(torch.equal(artifact['head_state'][key][7:], source.actor.state_dict()[key][7:])
                                                                  for key in ('weight', 'bias'))
                if variant == 'raw_linear':
                    previous = torch.load(prior_root / f'seed{seed}_supervised_actor.pt', map_location='cpu', weights_only=False)
                    checks[prefix + 'previous_raw_fit_exact'] = (_state_equal(artifact['head_state'], previous['actor_state'])
                        and all(artifact[key] == previous['fit'][key] for key in ('selected_learning_rate', 'selected_epoch', 'selection_score', 'trials')))
                targets = fixture_action_targets(datasets['old']['labels'])
                with torch.no_grad():
                    logits = policy.actor(values)
                for split in ('train', 'validation'):
                    metrics = action_metrics(logits[masks[split]], targets[masks[split]])
                    checks[prefix + split + '_saved_metrics_recomputed'] = _near(metrics, artifact[split + '_metrics'])
                statistics.append({'policy_seed': seed, 'variant': variant,
                    'head_parameters': sum(p.numel() for p in policy.actor.parameters()),
                    'max_raw_vs_reconstructed_standardized_logit_difference': max_difference,
                    'cpu_metrics_tolerance': 3e-4,
                    'feature_std_min': float(std.min()), 'feature_std_max': float(std.max()),
                    'feature_std_median': float(np.median(std)),
                    'feature_std_floor_dimensions': int((std <= config['feature_std_floor']).sum())})
            checks[prefix + 'non_actor_and_optimizer_preserved'] = (all(torch.equal(v, source.state_dict()[k]) for k, v in policy.state_dict().items() if not k.startswith('actor.'))
                and _state_equal(policy.optimizer.state_dict(), source.optimizer.state_dict()))
            for name, summary in (('old', paired), ('fresh', fresh)):
                mask = masks['heldout'] if name == 'old' else np.ones(len(features[name]), dtype=bool)
                metrics = evaluate_actor(policy, features[name][mask], datasets[name]['labels'][mask], datasets[name]['background_id'][mask], 'cpu')
                saved = next(row['heldout_action_metrics'] for row in summary['evaluation'] if row['policy_seed'] == seed and row['variant'] == variant)
                checks[prefix + name + '_saved_cpu_metrics_recomputed'] = _near(metrics, saved)
            checks[prefix + 'no_gradients'] = all(p.grad is None and not p.requires_grad for p in policy.parameters())
        checks[f'seed{seed}_source_unchanged'] = _state_equal(before, (source.state_dict(), source.optimizer.state_dict()))
    checks = {k: bool(v) for k, v in checks.items()}
    result = {'passed': all(checks.values()), 'checks': checks, 'artifact_sha256': hashes, 'head_statistics': statistics,
              'supervised_updates': 0, 'ppo_updates': 0, 'resumable_ppo_checkpoint': False,
              'cpu_cuda_last_bit_differences_allowed': True,
              'training_coordinate_weights_reconstructed_from_folded_head': True}
    _write(output, result)
    if not result['passed']:
        raise RuntimeError(f"saved artifact checks failed: {[k for k, v in checks.items() if not v]}")
    print(f"Saved-head verification passed: {len(checks)} checks; 9 fitted artifacts, 12 evaluated policy variants")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--paired', required=True)
    parser.add_argument('--fresh', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    verify(Path(args.paired), Path(args.fresh), Path(args.output))


if __name__ == '__main__':
    main()
