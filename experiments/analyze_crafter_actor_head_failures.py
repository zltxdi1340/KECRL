"""Read-only attribution of the fresh, frozen actor-head trajectories."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from experiments.analyze_crafter_wood3_actor_only_local import _csv, _json_digest, _read, _sha, _write


def analyze(root, output):
    summary = _read(root / 'summary.json')
    counters, background_rows, case_rows = [], [], []
    checks = {}
    for entry in summary['evaluation']:
        seed, variant = entry['policy_seed'], entry['variant']
        path = root / f'seed{seed}_{variant}_rollouts.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        checks[f'seed{seed}_{variant}_trajectory_integrity'] = (_sha(path) == entry['rollouts']['jsonl_sha256']
            and _json_digest(rows) == entry['rollouts']['canonical_digest'])
        counts = Counter({key: 0 for key in ('tree_present_episodes', 'successes', 'failures',
            'aligned_first_wrong', 'aligned_first_correct_then_fail', 'needs_turn_first_wrong',
            'needs_turn_correct_first_then_wrong_second', 'needs_turn_correct_first_and_do_then_fail')})
        for row in rows:
            if not row['tree_present']:
                continue
            counts['tree_present_episodes'] += 1
            if row['designated_tree_success']:
                counts['successes'] += 1
                continue
            counts['failures'] += 1
            target = 5 if row['condition'] == 'aligned' else row['tree_action']
            first = row['events'][0]['action']
            second = row['events'][1]['action'] if len(row['events']) > 1 else None
            if row['condition'] == 'aligned':
                bucket = 'aligned_first_wrong' if first != target else 'aligned_first_correct_then_fail'
            elif first != target:
                bucket = 'needs_turn_first_wrong'
            else:
                bucket = 'needs_turn_correct_first_then_wrong_second' if second != 5 else 'needs_turn_correct_first_and_do_then_fail'
            counts[bucket] += 1
            case_rows.append({'policy_seed': seed, 'variant': variant, 'background_id': row['background_id'],
                'scene_index': row['scene_index'], 'initial_wood': row['initial_wood'], 'condition': row['condition'],
                'first_target_action': target, 'first_action': first, 'second_action': second, 'failure_bucket': bucket})
        checks[f'seed{seed}_{variant}_partition'] = (counts['failures'] == sum(v for k, v in counts.items()
            if k not in ('tree_present_episodes', 'successes', 'failures'))
            and counts['failures'] + counts['successes'] == counts['tree_present_episodes'])
        counters.append({'policy_seed': seed, 'variant': variant, **dict(counts)})
        for bg in sorted({r['background_id'] for r in rows}):
            chosen = [r for r in rows if r['tree_present'] and r['background_id'] == bg]
            failed = [r for r in chosen if not r['designated_tree_success']]
            correct_first = sum(r['events'][0]['action'] == (5 if r['condition'] == 'aligned' else r['tree_action']) for r in failed)
            background_rows.append({'policy_seed': seed, 'variant': variant, 'background_id': bg,
                'episodes': len(chosen), 'successes': sum(r['designated_tree_success'] for r in chosen),
                'failures_with_first_action_correct': correct_first})
    if not all(checks.values()):
        raise RuntimeError('fresh failure attribution integrity audit failed')
    totals = {}
    for variant in ('baseline', 'raw_linear', 'standardized_linear', 'mlp64'):
        group = [r for r in counters if r['variant'] == variant]
        totals[variant] = {k: sum(r[k] for r in group) for k in group[0] if k not in ('policy_seed', 'variant')}
    _write(output / 'fresh_failure_attribution.json', {'formal_result': False, 'evaluation_only': True,
        'passed': True, 'checks': checks, 'counts_by_policy_seed': counters, 'totals': totals,
        'scope': 'constructed adjacent-tree fixtures, greedy, max eight steps; no survival or exploration claim'})
    _csv(output / 'fresh_failure_backgrounds.csv', background_rows)
    _csv(output / 'fresh_failed_scenes.csv', case_rows)
    _csv(output / 'fresh_failure_counts.csv', counters)
    from PIL import Image, ImageDraw
    records = [json.loads(line) for line in (root / 'scene_records.jsonl').read_text().splitlines()]
    with np.load(root / 'scene_dataset.npz') as data:
        images = data['images']
        sheet = Image.new('RGB', (200 * 4, 225), 'white')
        draw = ImageDraw.Draw(sheet)
        for column, bg in enumerate((0, 1, 11, 13)):
            record = next(r for r in records if r['background_id'] == bg and r['scene_index'] == 0)
            draw.text((column * 200 + 4, 4), f'Fresh bg {bg}; wood0, aligned left', fill='black')
            sheet.paste(Image.fromarray(images[record['row']]).resize((192, 192), Image.Resampling.NEAREST), (column * 200 + 4, 25))
        sheet.save(output / 'fresh_background_examples.png')
    print(json.dumps(totals['standardized_linear'], indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fresh', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    analyze(Path(args.fresh), Path(args.output))


if __name__ == '__main__':
    main()
