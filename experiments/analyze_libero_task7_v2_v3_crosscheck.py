"""Read audited endpoint metadata and nine saved action members; no rollout.

The scalar cross-check shares the admitted cache. It is not a second physics
audit. Action differences cannot identify object geometry or a failure cause.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from experiments import libero_eef_xyz_control_protocol as v2
from experiments import libero_spatial_readout_execution_v3 as v3

ROOT, require, pin, write_new = v3.ROOT, v3.require, v3.pin, v3.write_new
ARMS = {'v2': ('zero_xyz_mlp', 'cached_xyz_mlp'),
        'v3': ('flat_zero_xyz', 'spatial_coordinates_zero_xyz')}
MEMBERS = ['agentview_rgb', 'eye_in_hand_rgb', 'joint_states', 'gripper_states',
           'actions', 'rewards', 'terminated', 'truncated']


def overlap(pairs):
    require(len(pairs) == 22 and len({r['source_id'] for r in pairs}) == 22,
            'exact22 paired endpoints required')
    sets = {name: {r['source_id'] for r in pairs if r[key]} for name, key in
            [('control', 'control_success'), ('v2', 'v2_xyz_success'), ('v3', 'v3_spatial_success')]}
    a, b = sets['v2'], sets['v3']
    return dict(success_sets={k: sorted(s) for k, s in sets.items()}, intersection=sorted(a & b),
                union=sorted(a | b), v2_only=sorted(a - b), v3_only=sorted(b - a),
                per_seed={str(seed): {name: sum(r['seed'] == seed and r['source_id'] in s for r in pairs)
                                     for name, s in sets.items()} for seed in sorted({r['seed'] for r in pairs})})


def preflight(config, binding):
    cfg = json.loads(Path(config).read_text()); bound = json.loads(Path(binding).read_text())
    require(cfg['kind'] == 'libero_task7_v2_v3_readonly_overlap_actions' and cfg['version'] == 1
            and cfg['formal_result'] is False and cfg['combination_model_tested'] is False,
            'readonly crosscheck scope differs')
    require(bound['status'] == 'bound_before_v2_v3_crosscheck_action_payload'
            and bound['science'] == pin(config), 'new readonly binding required')
    for item in bound['code_dependencies'] + bound['validation']:
        require(pin(item['path']) == item, 'crosscheck dependency changed: ' + item['path'])
    closure = {r['path']: r['sha256'] for r in bound['code_dependencies']}
    require(closure.get(str(Path(__file__).resolve())) == pin(__file__)['sha256'], 'analysis code unbound')
    groups = {}
    for family, parent in cfg['parents'].items():
        require(family in ARMS, 'unknown parent')
        for item in parent.values():
            require(pin(item['path']) == item, 'parent metadata changed')
        _, old = (v2 if family == 'v2' else v3).preflight(parent['science']['path'], parent['binding']['path'], 'handoff')
        require({r['path']: r['sha256'] for r in old['code_dependencies']}.items() <= closure.items(), 'parent closure missing')
        proof = json.loads(Path(parent['completion']['path']).read_text())
        require(proof['status'] == 'complete_audited_postgrasp_handoff'
                and proof['result'] == parent['result'] and proof['audit'] == parent['audit'], 'audited endpoint proof differs')
        result = json.loads(Path(parent['result']['path']).read_text())
        require(result['status'] == 'postgrasp_handoff_complete_pending_audit' and len(result['episodes']) == 44, 'complete44 required')
        index = {}
        for row in result['episodes']:
            key = (row['source']['source_id'], row['controller'])
            require(key not in index and row['controller'] in ARMS[family], 'duplicate or unknown endpoint')
            index[key] = row
        groups[family] = index
    require(set(groups) == {'v2', 'v3'}, 'both parents required')
    overlap(cfg['pairs']); archives = []
    for pair in cfg['pairs']:
        source = pair['source_id']
        c2, r2 = [groups['v2'][(source, arm)] for arm in ARMS['v2']]
        c3, r3 = [groups['v3'][(source, arm)] for arm in ARMS['v3']]
        require(all(r['source'] == c2['source'] for r in [r2, c3, r3]), 'source identity differs')
        require(c2['success'] == c3['success'] and c2['steps'] == c3['steps'], 'controls differ')
        expected = dict(source_id=source, seed=c2['seed'], initial_state_index=c2['initial_state_index'],
                        anchor_step=c2['anchor_step'], control_success=c3['success'],
                        v2_xyz_success=r2['success'], v3_spatial_success=r3['success'],
                        control_steps=c3['steps'], v2_steps=r2['steps'], v3_steps=r3['steps'])
        require(pair == expected, 'paired endpoint metadata differs')
        if c3['success'] and not r2['success'] and not r3['success']:
            a = pair['anchor_step']
            for row in [r2, r3]:
                require(row['initial_row_sha256'] == c3['initial_row_sha256']
                        and row['handoff_observation_sha256'] == c3['handoff_observation_sha256']
                        and row['observation_sha256'][:a+1] == c3['observation_sha256'][:a+1], 'initial/prefix/handoff observations differ')
            for controller, family, row in [('control', 'v3', c3), ('v2_xyz', 'v2', r2), ('v3_spatial', 'v3', r3)]:
                archives.append(dict(source_id=source, controller=controller, family=family, seed=row['seed'],
                                     anchor_step=a, steps=row['steps'], success=row['success'], frames=row['frames']))
    require(cfg['archives'] == archives and len(archives) == 9, 'all three repeated losses required')
    expected_caps = dict(npz_parses=9, numeric_member_decodes=9,
                         numeric_scalars=sum(r['steps']*7 for r in archives),
                         source_container_bytes=sum(Path(r['frames']['path']).stat().st_size for r in archives),
                         **{k: 0 for k in ['rgb_pixel_decodes', 'eef_decodes', 'object_state_reads',
                                           'policy_forwards', 'model_constructions', 'checkpoint_reads', 'hdf5_reads',
                                           'initial_bank_reads', 'optimizer_updates', 'native_steps']})
    require(cfg['caps'] == expected_caps and Path(cfg['output']).resolve().is_relative_to(ROOT/'results'), 'budget/storage differs')
    item = bound['authorization']; require(pin(item['path']) == item, 'authority changed')
    grant = json.loads(Path(item['path']).read_text())
    require(grant['user_approved'] is True and grant['science'] == pin(config) and grant['caps'] == expected_caps
            and grant['source_archives'] == [r['frames'] for r in archives] and grant['numeric_members'] == ['actions']
            and grant['sealed_roles_allowed'] == [], 'readonly authority differs')
    return cfg


def load_actions(row, ledger):
    item = row['frames']; n = row['steps']; size = Path(item['path']).stat().st_size
    for key, amount in [('source_container_bytes', size), ('npz_parses', 1),
                        ('numeric_member_decodes', 1), ('numeric_scalars', n*7)]:
        ledger.begin(key, amount)
    require(pin(item['path']) == item, 'action archive changed')
    with np.load(item['path'], allow_pickle=False) as archive:
        members = MEMBERS + (['eef_pos'] if row['family'] == 'v2' else [])
        require(len(archive.files) == len(members) and set(archive.files) == set(members), 'archive member schema differs')
        # Inspect only the actions header before allocating its numeric payload.
        with archive.zip.open('actions.npy') as header:
            version = np.lib.format.read_magic(header)
            require(version in [(1, 0), (2, 0)], 'unsupported action header')
            reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
            shape, _, dtype = reader(header)
        require(shape == (n, 7) and dtype == np.float32, 'action shape/dtype differs')
        actions = archive['actions']
    require(np.isfinite(actions).all() and np.abs(actions).max() <= 1, 'nonfinite or out of range action')
    actions.flags.writeable = False
    for key, amount in [('source_container_bytes', size), ('npz_parses', 1),
                        ('numeric_member_decodes', 1), ('numeric_scalars', n*7)]:
        ledger.finish(key, amount)
    return actions


def metrics(control, candidate, anchor):
    require(0 <= anchor < min(len(control), len(candidate)) and control.shape[1:] == candidate.shape[1:] == (7,), 'comparison window differs')
    require(np.array_equal(control[:anchor], candidate[:anchor]), 'stored action prefix differs')
    end = min(len(control), len(candidate))
    c, x = control[anchor:end].astype(np.float64), candidate[anchor:end].astype(np.float64)
    delta = x-c; differing = np.flatnonzero(np.any(delta != 0, axis=1))
    def first(g, sign):
        indices = np.flatnonzero(g*sign > 0)
        return None if not len(indices) else anchor+1+int(indices[0])
    return dict(common_start_step=anchor+1, common_end_step=end, common_steps=end-anchor,
                first_action_difference=None if not len(differing) else anchor+1+int(differing[0]),
                first_control_action=c[0].tolist(), first_candidate_action=x[0].tolist(), first_delta=delta[0].tolist(),
                axis_mean_abs_delta=np.mean(np.abs(delta), axis=0).tolist(),
                translation_mean_abs_delta=float(np.mean(np.abs(delta[:, :3]))),
                rotation_mean_abs_delta=float(np.mean(np.abs(delta[:, 3:6]))),
                gripper_sign_disagreements=int(np.count_nonzero(np.sign(c[:, 6]) != np.sign(x[:, 6]))),
                control_first_open=first(c[:, 6], -1), candidate_first_open=first(x[:, 6], -1),
                control_first_close=first(c[:, 6], 1), candidate_first_close=first(x[:, 6], 1))


def scalar_metrics(control, candidate, anchor):
    require(control[:anchor].tolist() == candidate[:anchor].tolist(), 'scalar action prefix differs')
    end = min(len(control), len(candidate)); c = control[anchor:end].tolist(); x = candidate[anchor:end].tolist()
    changes = [[b-a for a, b in zip(left, right)] for left, right in zip(c, x)]
    different = [i+anchor+1 for i, row in enumerate(changes) if any(v != 0 for v in row)]
    def first(rows, sign):
        return next((i+anchor+1 for i, row in enumerate(rows) if row[6]*sign > 0), None)
    mean = [sum(abs(row[k]) for row in changes)/len(changes) for k in range(7)]
    sign = lambda x: (x > 0)-(x < 0)
    return dict(common_start_step=anchor+1, common_end_step=end, common_steps=len(c),
                first_action_difference=different[0] if different else None, first_control_action=c[0],
                first_candidate_action=x[0], first_delta=changes[0], axis_mean_abs_delta=mean,
                translation_mean_abs_delta=sum(mean[:3])/3, rotation_mean_abs_delta=sum(mean[3:6])/3,
                gripper_sign_disagreements=sum(sign(a[6]) != sign(b[6]) for a, b in zip(c, x)),
                control_first_open=first(c, -1), candidate_first_open=first(x, -1),
                control_first_close=first(c, 1), candidate_first_close=first(x, 1))


def crosscheck(left, right):
    require(type(left) is type(right), 'crosscheck type differs')
    if isinstance(left, dict):
        require(left.keys() == right.keys(), 'crosscheck keys differ')
        return sum(crosscheck(left[k], right[k]) for k in left)
    if isinstance(left, list):
        require(len(left) == len(right), 'crosscheck length differs')
        return sum(crosscheck(a, b) for a, b in zip(left, right))
    require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-12) if isinstance(left, float) else left == right,
            'scalar action statistic differs')
    return 1


def run(config, binding):
    cfg = preflight(config, binding); base = Path(cfg['output'])
    require(not base.exists(), 'new output directory required'); base.mkdir(parents=True)
    ledger = v3.Ledger(cfg['caps']); cache = {}; rows = []; checks = 0
    try:
        for row in cfg['archives']:
            key = (row['source_id'], row['controller']); require(key not in cache, 'duplicate action read')
            cache[key] = load_actions(row, ledger)
        for pair in cfg['pairs']:
            source = pair['source_id']
            if (source, 'control') not in cache:
                continue
            for arm in ['v2_xyz', 'v3_spatial']:
                arrays = cache[(source, 'control')], cache[(source, arm)]
                value = metrics(*arrays, pair['anchor_step'])
                checks += crosscheck(value, scalar_metrics(*arrays, pair['anchor_step']))
                rows.append(dict(source_id=source, seed=pair['seed'], candidate=arm, **value))
        require(all(r['completed'] == cfg['caps'][k] == r['attempted'] for k, r in ledger.counts.items()), 'exact readonly envelope differs')
        result = dict(status='complete_readonly_v2_v3_crosscheck', formal_result=False, science=pin(config), binding=pin(binding),
                      overlap=overlap(cfg['pairs']), endpoints=cfg['pairs'], command_comparisons=rows,
                      ledger=ledger.snapshot(), scalar_formula_checks=checks, combination_model_tested=False,
                      geometric_margin_identified=False, command_cause_identified=False,
                      interpretation='Endpoint union is descriptive, not a combined policy score. Scalar formulas share one cache. Container bytes count unique admitted archive sizes, not total filesystem IO.')
        write_new(base/'analysis.json', result)
        write_new(base/'completion.json', dict(status=result['status'], analysis=pin(base/'analysis.json'),
                                              physics_steps=0, optimizer_updates=0, model_forwards=0))
        print(json.dumps(dict(status=result['status'], union=len(result['overlap']['union']),
                              v2_only=result['overlap']['v2_only'], v3_only=result['overlap']['v3_only'], scalar_checks=checks)))
    except BaseException as error:
        write_new(base/'failure.json', dict(error=repr(error), ledger=ledger.snapshot(), completed_comparisons=rows))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True)
    args = parser.parse_args(); run(args.config, args.binding)
