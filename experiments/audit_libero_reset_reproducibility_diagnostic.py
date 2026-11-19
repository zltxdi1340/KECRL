"""Independent byte/scalar audit; never reset physics or open initial banks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from experiments import libero_reset_reproducibility_protocol as p


def independent_summary(arrays, source):
    combined = hashlib.sha256(); parts = {}
    for key in sorted(p.SHAPES):
        value = arrays[key]
        p.require(value.shape == p.SHAPES[key]
            and value.dtype == (np.uint8 if key.endswith('rgb') else np.float32)
            and np.isfinite(value).all(), 'audit public interface differs')
        data = (key.encode() + str(value.dtype).encode()
            + struct.pack('<'+'q'*len(value.shape), *value.shape) + value.tobytes(order='C'))
        combined.update(data); parts[key] = hashlib.sha256(data).hexdigest()
    exact = {}; maximum = {}
    for field, original in [('joint_states', 'joint_state'), ('gripper_states', 'gripper_state')]:
        reference = source['stepwise_public_state_trace'][0][original]
        p.require(len(reference) == p.SHAPES[field][0], 'audit source public dimensions differ')
        actual = [float(x) for x in arrays[field]]
        exact[field] = all(a == b for a, b in zip(actual, reference))
        maximum[field] = max(abs(a-b) for a, b in zip(actual, reference))
    digest = combined.hexdigest(); match = digest == source['reset_observation_sha256']
    p.require(not match or all(exact.values()), 'audit source digest/public state conflict')
    category = ('all_public_match' if match else 'rgb_only_difference' if all(exact.values())
                else 'proprioception_differs_rgb_vs_original_unknown')
    return dict(observation_sha256=digest, component_sha256=parts,
        source_reset_sha256=source['reset_observation_sha256'], reset_digest_exact=match,
        state_exact=exact, state_max_absolute_error=maximum, difference_category=category)


def audit_probe(cfg, captured, *, source_loader):
    p.require(captured['status'] == 'reset_probe_complete_pending_audit'
        and captured['no_automatic_retry'] is True, 'complete reset capture required')
    base = Path(cfg['output']); rows = captured['records']
    expected = [('direct', 7)] + [('sequential', i) for i in range(8)]
    p.require([(r['branch'], r['initial_state_index']) for r in rows] == expected,
              'reset order or coverage differs')
    source_bytes = 0; sources = {}
    for descriptor in cfg['references']:
        source_bytes += Path(descriptor['path']).stat().st_size
        sources[descriptor['initial_state_index']] = source_loader(descriptor)
    item = captured['initial_access']
    p.require(p.pin(item['path']) == item and Path(item['path']).resolve() == base/'initial-access.json',
              'initial access receipt changed')
    access = json.loads(Path(item['path']).read_text()); p.validate_initial_access(access, cfg)
    observed = {}; checks = 0
    for row in rows:
        branch, index = row['branch'], row['initial_state_index']; item = row['observation']
        p.require(Path(item['path']).resolve() == base/f'{branch}-{index:02d}.npz'
            and p.pin(item['path']) == item and row['local_initial_state_index'] == index
            and row['source'] == {k: cfg['references'][index][k] for k in ('path', 'sha256')}
            and row['initial_row_sha256'] == access['selected_row_sha256'][str(index)],
            'reset file identity or row mapping differs')
        with np.load(item['path'], allow_pickle=False) as archive:
            p.require(set(archive.files) == set(p.SHAPES) and len(archive.files) == 4,
                      'public-only NPZ schema required')
            arrays = {key: archive[key] for key in p.SHAPES}
        summary = independent_summary(arrays, sources[index])
        for key, value in summary.items():
            p.require(row[key] == value, 'independent reset summary differs: '+key); checks += 1
        observed[branch, index] = arrays
    p.require({path.name for path in base.glob('*.npz')} == {f'{b}-{i:02d}.npz' for b, i in expected},
              'unexpected public payload file')
    ledger = captured['ledger']
    p.require(ledger['caps'] == p.CAPS and 0 <= ledger['elapsed_seconds'] <= 180, 'reset budget differs')
    for name, cap in p.CAPS.items():
        expected_count = source_bytes if name == 'source_episode_bytes' else cap
        p.require(ledger['counts'][name] == dict(attempted=expected_count, completed=expected_count)
            and expected_count <= cap, 'actual reset count differs: '+name); checks += 1
    direct = rows[0]; sequential = rows[-1]
    comparison = dict(direct_matches_source=direct['reset_digest_exact'],
        sequential_matches_source=sequential['reset_digest_exact'],
        branches_match_each_other=direct['observation_sha256'] == sequential['observation_sha256'],
        reset_only_probe=True, original_full_episode_history_reproduced=False,
        sole_cause_identified=False, gripper_timing_identified=False)
    p.require(comparison == captured['comparison'], 'target comparison differs')
    branch_differences = {}
    for field in p.SHAPES:
        a = observed['direct', 7][field]; b = observed['sequential', 7][field]
        # Explicit scalar arithmetic keeps descriptive differences separate from exact gates.
        pairs = list(zip(a.flat, b.flat))
        branch_differences[field] = dict(changed_values=sum(float(x) != float(y) for x, y in pairs),
            max_absolute_error=max(abs(float(x)-float(y)) for x, y in pairs))
    return dict(status='passed_independent_reset_probe_audit', formal_result=False,
        comparison=comparison, target_branch_differences=branch_differences,
        records_checked=9, exact_summary_and_budget_checks=checks,
        source_episode_parses=8, source_episode_bytes=source_bytes,
        initial_bank_reads=0, new_native_steps=0, policy_forwards=0,
        original_failed_video_retried=False, original_full_episode_history_reproduced=False,
        sole_cause_identified=False, gripper_timing_identified=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--binding', required=True); args = parser.parse_args()
    cfg, _ = p.preflight(Path(args.config), Path(args.binding)); base = Path(cfg['output'])
    execution = json.loads((base/'execution.json').read_text())
    p.require(execution['science'] == p.pin(args.config) and execution['binding'] == p.pin(args.binding)
        and execution['capture'] == p.pin(base/'reset-probe.json') and execution['model_reads'] == 0
        and execution['model_constructed'] is False and execution['previous_video_resumed'] is False
        and execution['new_policy_evaluation'] is False, 'reset execution provenance differs')
    def loader(item):
        raw = Path(item['path']).read_bytes()
        p.require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'source reset JSON changed in audit')
        return json.loads(raw)
    captured = json.loads((base/'reset-probe.json').read_text())
    try:
        result = audit_probe(cfg, captured, source_loader=loader)
        result.update(science=p.pin(args.config), binding=p.pin(args.binding), capture=p.pin(base/'reset-probe.json'))
        p.write_new(base/'independent-audit.json', result)
        p.write_new(base/'completion.json', dict(status='complete_audited_reset_reproducibility_probe',
            formal_result=False, result=p.pin(base/'reset-probe.json'), audit=p.pin(base/'independent-audit.json'),
            comparison=result['comparison'], sole_cause_identified=False, gripper_timing_identified=False))
        print(json.dumps(dict(status=result['status'], comparison=result['comparison'])))
    except BaseException as error:
        p.write_new(base/'audit-failure.json', dict(status='technical_reset_audit_failure', error=repr(error)))
        raise


if __name__ == '__main__':
    main()
