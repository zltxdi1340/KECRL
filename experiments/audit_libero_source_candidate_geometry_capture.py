"""Independent saved-JSON geometry audit and descriptive event analysis.

No collector/observer, Torch, simulator, image, checkpoint or bank loader is
imported. Recalculate coordinates, contact counts, timing, pairing and budgets.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import groupby
import json
import math
from pathlib import Path
import statistics

from experiments import libero_source_candidate_geometry_capture_protocol as science

ROOT, require, sha = science.ROOT, science.require, science.sha


def expected_caps(config):
    b = config['design']['budget']
    mapping = {'model_load': 'checkpoint_deserializations_max', 'policy_object': 'models',
        'initial_container': 'initial_containers_max', 'numeric_row': 'numeric_initial_rows_max',
        'adapter': 'adapter_creations_max', 'native_reset': 'native_reset_attempts_max', 'episode': 'episodes',
        'native_control': 'native_control_steps_max', 'physics_step': 'native_physics_steps_max',
        'zero_action': 'zero_actions_max', 'policy_forward': 'policy_forwards_and_actions_max',
        'policy_action': 'policy_forwards_and_actions_max', 'shadow_allocation': 'shadow_allocations_max',
        'shadow_forward': 'shadow_position_forwards_max', 'geometry_row': 'geometry_rows_max',
        'public_row': 'public_rows_max', 'contact_record': 'saved_bowl_contact_records_max'}
    return {**{k: b[v] for k, v in mapping.items()}, **dict.fromkeys(
        ['forbidden_observer_native_call', 'optimizer_update', 'HDF5_read', 'sealed_payload_read', 'old_trajectory_read'], 0)}


def load(path):
    return json.loads(Path(path).read_text())


def scalar(value):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite/invalid scalar')
    return value


def close(actual, expected, tolerance=1e-12):
    require(abs(scalar(actual)-scalar(expected)) <= tolerance, 'numeric derivation/time mismatch')


def vector(value, size=3):
    require(type(value) is list and len(value) == size, 'vector shape differs')
    return [scalar(x) for x in value]


def norm(values):
    return math.sqrt(math.fsum(x*x for x in values))


def rotation(value):
    rows = [vector(r) for r in value]
    require(len(rows) == 3, 'rotation shape differs')
    for i in range(3):
        for j in range(3):close(math.fsum(a*b for a, b in zip(rows[i], rows[j])), int(i == j), 1e-6)
    determinant = (rows[0][0]*(rows[1][1]*rows[2][2]-rows[1][2]*rows[2][1])-
                   rows[0][1]*(rows[1][0]*rows[2][2]-rows[1][2]*rows[2][0])+
                   rows[0][2]*(rows[1][0]*rows[2][1]-rows[1][1]*rows[2][0]))
    close(determinant, 1, 1e-6)


def digest_valid(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'invalid SHA field')


def reset_digest(row):
    ignore = {'native_before_sha256', 'native_after_sha256', 'observation_before_sha256',
              'observation_after_sha256', 'copied_state_sha256', 'shadow_copied_state_sha256'}
    return hashlib.sha256(json.dumps({k: v for k, v in row.items() if k not in ignore},
                                    sort_keys=True, allow_nan=False).encode()).hexdigest()


def validate_episode(public, geometry, identity):
    require(public['status'] == geometry['status'] == 'complete' and
            public['identity'] == geometry['identity'] == identity, 'identity/status mismatch')
    n = public['steps'];success = public['prefix_success']
    require(type(n) is int and 1 <= n <= 150 and type(success) is bool and
            geometry['steps'] == n and geometry['prefix_success'] is success and
            public['capture_stop_reason'] == geometry['capture_stop_reason'] ==
                ('success' if success else 'diagnostic_prefix_limit') and (success or n == 150), 'prefix boundary differs')
    pr, gr = public['rows'], geometry['rows'];header = geometry['header']
    require(len(pr) == len(gr) == n+1 and [r['state_index'] for r in pr] ==
            [r['state_index'] for r in gr] == list(range(n+1)), 'state timeline differs')
    require(header['coordinate_frame'] == 'world' and header['distance_unit'] == 'metre' and
            header['rotation_format'] == '3x3_row_major' and
            header['contact_origin'] == 'shadow_mj_fwdPosition_geometric_collision_not_force', 'geometry semantics differ')
    close(header['physics_timestep_seconds'], .002);close(header['control_frequency_hz'], 20)
    digest_valid(header['model_xml_sha256'])
    require(header['bodies']['bowl']['object_name'] == 'akita_black_bowl_1' and
            header['bodies']['plate']['object_name'] == 'plate_1', 'wrong object binding')
    groups = header['collision_groups'];require(set(groups) == {'bowl', 'plate', 'left', 'right'}, 'collision scope differs')
    ids = {}; names = {}
    for group, rows in groups.items():
        require(rows and all(type(x['id']) is int and x['id'] >= 0 and type(x['name']) is str and x['name'] for x in rows),
                'empty/invalid collision group')
        ids[group] = {x['id'] for x in rows};require(len(ids[group]) == len(rows), 'duplicate collision ID')
        names.update({x['id']: x['name'] for x in rows})
    keys = list(ids);require(all(not ids[a] & ids[b] for j, a in enumerate(keys) for b in keys[j+1:]), 'overlapping collision groups')
    contacts_total = 0
    geometry_keys = set(('state_index sim_time eef_position_world eef_rotation_world bowl_position_world '
        'bowl_rotation_world plate_position_world plate_rotation_world eef_to_bowl_world eef_to_bowl_xyz '
        'eef_to_bowl_xy eef_minus_bowl_signed_z bowl_to_plate_world bowl_to_plate_xyz bowl_to_plate_xy '
        'bowl_height_change contacts touch_counts positive_margin_counts bilateral_touch_proxy '
        'native_before_sha256 native_after_sha256 observation_before_sha256 observation_after_sha256 '
        'copied_state_sha256 shadow_copied_state_sha256 native_vs_policy_joint_gap_l2 '
        'native_vs_policy_gripper_gap_l2').split())
    for k, (p, g) in enumerate(zip(pr, gr)):
        require(set(g) == geometry_keys, 'unexpected geometry/private field')
        require(set(p) == {'state_index', 'action', 'joint_state', 'gripper_state', 'observation_sha256',
                           'joint_offset_l2_from_start', 'gripper_offset_l2_from_start'}, 'unexpected public/private field')
        vector(p['joint_state'], 7);vector(p['gripper_state'], 2);digest_valid(p['observation_sha256'])
        require(p['action'] is None if k == 0 else all(abs(x) <= 1 for x in vector(p['action'], 7)), 'invalid action')
        close(p['joint_offset_l2_from_start'], math.dist(p['joint_state'], pr[0]['joint_state']))
        close(p['gripper_offset_l2_from_start'], math.dist(p['gripper_state'], pr[0]['gripper_state']))
        scalar(g['sim_time'])
        require(scalar(g['native_vs_policy_joint_gap_l2']) >= 0 and
                scalar(g['native_vs_policy_gripper_gap_l2']) >= 0, 'invalid sampled-cache gap')
        if k:close(g['sim_time']-gr[k-1]['sim_time'], .05, 1e-9)
        for a, b in [('native_before_sha256', 'native_after_sha256'),
                     ('observation_before_sha256', 'observation_after_sha256'),
                     ('copied_state_sha256', 'shadow_copied_state_sha256')]:
            digest_valid(g[a]);digest_valid(g[b]);require(g[a] == g[b], 'passivity/copy witness failed')
        require(g['observation_before_sha256'] == p['observation_sha256'], 'geometry/public join failed')
        eef, bowl, plate = [vector(g[x+'_position_world']) for x in ('eef', 'bowl', 'plate')]
        for name in ('eef', 'bowl', 'plate'):rotation(g[name+'_rotation_world'])
        delta = [b-a for a, b in zip(eef, bowl)]; target = [b-a for a, b in zip(bowl, plate)]
        for recorded, expected in [(g['eef_to_bowl_world'], delta), (g['bowl_to_plate_world'], target)]:
            for a, b in zip(vector(recorded), expected):close(a, b)
        for name, value in [('eef_to_bowl_xyz', norm(delta)), ('eef_to_bowl_xy', norm(delta[:2])),
                            ('eef_minus_bowl_signed_z', eef[2]-bowl[2]), ('bowl_to_plate_xyz', norm(target)),
                            ('bowl_to_plate_xy', norm(target[:2])), ('bowl_height_change', bowl[2]-gr[0]['bowl_position_world'][2])]:
            close(g[name], value)
        counts, margins = dict.fromkeys(('left', 'right', 'plate', 'other'), 0), dict.fromkeys(('left', 'right', 'plate', 'other'), 0)
        require(len(g['contacts']) <= 256, 'contact cap exceeded')
        for contact in g['contacts']:
            require(set(contact) == {'geom1', 'geom2', 'geom1_name', 'geom2_name', 'other_group', 'distance_m', 'position_world'},
                    'unexpected contact/private/force field')
            a, b = contact['geom1'], contact['geom2']
            require(type(a) is int and type(b) is int and min(a, b) >= 0 and {a, b} & ids['bowl'], 'unapproved contact')
            other = b if a in ids['bowl'] else a
            group = next((q for q in ('left', 'right', 'plate') if other in ids[q]), 'other')
            require(contact['other_group'] == group, 'incorrect contact group')
            for label, index in [('geom1_name', a), ('geom2_name', b)]:
                if index in names:require(contact[label] == names[index], 'contact geom name/ID differs')
            vector(contact['position_world'])
            (counts if scalar(contact['distance_m']) <= 0 else margins)[group] += 1
        contacts_total += len(g['contacts'])
        require(g['touch_counts'] == counts and g['positive_margin_counts'] == margins and
                type(g['bilateral_touch_proxy']) is bool and
                g['bilateral_touch_proxy'] == (counts['left'] > 0 and counts['right'] > 0), 'contact/touch counts differ')
    reset = {'public_sha256': pr[0]['observation_sha256'], 'geometry_sha256': reset_digest(gr[0])}
    require(public['reset'] == geometry['reset'] == reset, 'reset digest differs')
    return n, contacts_total


def episode_features(public, geometry):
    actions, geo = [r['action'] for r in public['rows'][1:]], geometry['rows']
    groups = [(s, list(items)) for s, items in groupby(enumerate(actions, 1), key=lambda x: (x[1][6] > 0)-(x[1][6] < 0))]
    runs = [{'sign': s, 'start': items[0][0], 'end': items[-1][0], 'length': len(items),
             'right_censored': items[-1][0] == public['steps']} for s, items in groups]
    first = next((r for r in runs if r['sign'] == 1), None)
    multi = next((r for r in runs if r['sign'] == 1 and r['length'] >= 2), None)
    opening = next((r['start'] for r in runs if r['sign'] == -1 and first is not None and r['start'] > first['start']), None)
    touch = next((k for k, g in enumerate(geo) if g['bilateral_touch_proxy']), None)
    k = None if first is None else first['start']
    previous = None if k is None else geo[k-1]
    change = [] if k is None else [geo[t]['eef_to_bowl_xyz']-geo[t-1]['eef_to_bowl_xyz'] for t in range(max(1, k-5), k)]
    return {'steps': public['steps'], 'prefix_success': public['prefix_success'], 'capture_stop_reason': public['capture_stop_reason'],
            'first_close': k, 'first_close_length': None if first is None else first['length'],
            'first_close_right_censored': None if first is None else first['right_censored'],
            'first_multi_close': None if multi is None else multi['start'], 'first_open_after_close': opening,
            'first_bilateral_touch_state': touch, 'initial_bilateral_touch': touch == 0,
            'close_to_touch_seconds': None if k is None or touch is None else geo[touch]['sim_time']-previous['sim_time'],
            'first_close_time': None if previous is None else previous['sim_time'],
            'pre_close_distance_xyz': None if previous is None else previous['eef_to_bowl_xyz'],
            'pre_close_distance_xy': None if previous is None else previous['eef_to_bowl_xy'],
            'pre_close_signed_z': None if previous is None else previous['eef_minus_bowl_signed_z'],
            'pre_close_bilateral_touch': None if previous is None else previous['bilateral_touch_proxy'],
            'pre_close_distance_delta_mean': statistics.fmean(change) if change else None,
            'pre_close_distance_delta_count': len(change), 'close_positive_steps': sum(r['length'] for r in runs if r['sign'] > 0),
            'maximum_bowl_height_change': max(g['bowl_height_change'] for g in geo),
            'bilateral_touch_states': sum(g['bilateral_touch_proxy'] for g in geo),
            'bilateral_touch_seen_during_or_after_close': None if k is None else any(g['bilateral_touch_proxy'] for g in geo[k-1:]),
            'runs': runs}


def analyze_pairs(indexed):
    episode_rows, pair_rows, windows, groups = [], [], [], []
    for task in science.TASKS:
        for seed in science.SEEDS:
            for arm in science.ARMS:
                for index in science.INDICES:
                    p, g = indexed[task, seed, arm, index]
                    features = episode_features(p, g)
                    episode_rows.append({'task': task, 'seed': seed, 'arm': arm, 'global_initial_index': index,
                                         **{k: v for k, v in features.items() if k != 'runs'}})
            for index in science.INDICES:
                left, right = [indexed[task, seed, arm, index] for arm in science.ARMS]
                lf, rf = [episode_features(*entry) for entry in (left, right)]
                pair = {'task': task, 'seed': seed, 'global_initial_index': index,
                        'prefix_outcome': 'both_success' if lf['prefix_success'] and rf['prefix_success'] else
                            'A_only' if lf['prefix_success'] else 'B_only' if rf['prefix_success'] else 'neither_success_within_prefix'}
                for arm, f in [('A', lf), ('B', rf)]:
                    pair.update({arm+'_'+key: f[key] for key in ['first_close', 'first_multi_close', 'first_open_after_close',
                        'pre_close_distance_xyz', 'pre_close_signed_z', 'first_bilateral_touch_state', 'close_to_touch_seconds']})
                pair_rows.append(pair)
                anchors = {}
                for arm, f in [('A', lf), ('B', rf)]:
                    for event in ('first_close', 'first_multi_close', 'first_open_after_close'):
                        if f[event] is not None:anchors.setdefault(f[event], []).append(arm+'_'+event)
                for anchor, aliases in sorted(anchors.items()):
                    for offset in range(-5, 6):
                        tick = anchor+offset
                        row = {'task': task, 'seed': seed, 'global_initial_index': index, 'anchor_step': anchor,
                               'anchor_aliases': ';'.join(aliases), 'offset': offset, 'absolute_action_step': tick,
                               'reset_side_truncated': tick < 1}
                        for arm, (p, g) in [('A', left), ('B', right)]:
                            ok = 1 <= tick <= p['steps']
                            row[arm+'_available'] = ok; row[arm+'_terminal_truncated'] = tick > p['steps']
                            for metric in ('eef_to_bowl_xyz', 'eef_to_bowl_xy', 'eef_minus_bowl_signed_z',
                                           'bowl_height_change', 'bowl_to_plate_xyz', 'bilateral_touch_proxy'):
                                for phase, state in [('pre', tick-1), ('post', tick)]:
                                    row[arm+'_'+phase+'_'+metric] = g['rows'][state][metric] if ok else None
                            row[arm+'_gripper_command'] = p['rows'][tick]['action'][6] if ok else None
                        windows.append(row)
    for task in science.TASKS:
        for seed in science.SEEDS:
            for arm in science.ARMS:
                rows = [r for r in episode_rows if (r['task'], r['seed'], r['arm']) == (task, seed, arm)]
                values = [r['pre_close_distance_xyz'] for r in rows if r['pre_close_distance_xyz'] is not None]
                groups.append({'task': task, 'seed': seed, 'arm': arm, 'prefixes': len(rows),
                               'close_observed': sum(r['first_close'] is not None for r in rows),
                               'touch_observed': sum(r['first_bilateral_touch_state'] is not None for r in rows),
                               'prefix_success_count': sum(r['prefix_success'] for r in rows),
                               'pre_close_distance_mean': statistics.fmean(values) if values else None,
                               'pre_close_distance_median': statistics.median(values) if values else None})
    require(len(episode_rows) == 120 and len(pair_rows) == 60 and len(windows) <= 3960, 'analysis accounting differs')
    return {'episodes': episode_rows, 'pairs': pair_rows, 'event_windows': windows, 'task_seed_arms': groups}


def write_table(path, rows):
    # All-no-close is a valid diagnostic, so an empty event table still needs
    # an explicit schema rather than an indexing error or invented event.
    fields = list(rows[0]) if rows else ['task', 'seed', 'global_initial_index', 'anchor_step',
        'anchor_aliases', 'offset', 'absolute_action_step', 'reset_side_truncated']
    if not rows:
        for arm in ('A', 'B'):
            fields += [arm+'_available', arm+'_terminal_truncated']
            fields += [arm+'_'+phase+'_'+metric for metric in ('eef_to_bowl_xyz', 'eef_to_bowl_xy',
                'eef_minus_bowl_signed_z', 'bowl_height_change', 'bowl_to_plate_xyz', 'bilateral_touch_proxy')
                for phase in ('pre', 'post')]
            fields += [arm+'_gripper_command']
    with Path(path).open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields);writer.writeheader();writer.writerows(rows)


def validate_capture(result, config, archive):
    require(result['status'] == 'complete_capture_pending_independent_audit' and result['new_training_updates'] == 0 and
            result['G_decision_unchanged'] == 'failed_keep_H_sealed', 'capture incomplete or G decision changed')
    require(result['ledger']['caps'] == expected_caps(config), 'scientific budget/caps changed')
    expected = {(t, s, a, i) for t in science.TASKS for s in science.SEEDS for a in science.ARMS for i in science.INDICES}
    endpoints = {(m['task'], m['seed'], m['arm']): m for m in config['model_endpoints']}
    ordered = [(m['task'], m['seed'], m['arm'], i) for m in config['model_endpoints'] for i in science.INDICES]
    require([(d['task'], d['seed'], d['arm'], d['global_initial_index']) for d in result['rollout']['episode_files']] == ordered,
            'frozen task/seed/arm/scene order differs')
    indexed, resets, headers = {}, {}, {}; steps = contacts = comparisons = 0
    for descriptor in result['rollout']['episode_files']:
        key = descriptor['task'], descriptor['seed'], descriptor['arm'], descriptor['global_initial_index']
        require(key in expected and key not in indexed, 'duplicate/unapproved episode')
        identity = {k: descriptor[k] for k in ('task', 'seed', 'arm', 'global_initial_index', 'selected_bank_local_index',
                    'adapter_local_initial_index', 'initial_state_role', 'checkpoint_update', 'checkpoint',
                    'stabilization_steps', 'environment_seed')}
        require(identity['initial_state_role'] == 'candidate_nonregression' and identity['checkpoint_update'] == 500 and
                identity['checkpoint'] == endpoints[key[:3]]['checkpoint'] and
                identity['selected_bank_local_index'] == key[3] and identity['adapter_local_initial_index'] == 0 and
                identity['stabilization_steps'] == 5 and identity['environment_seed'] == 20261022, 'source identity/endpoint differs')
        docs = []
        for name in ('public', 'geometry'):
            pin = descriptor[name];path = Path(pin['path'])
            wanted = Path(archive)/f'task{key[0]}-seed{key[1]}-{key[2]}-global{key[3]:02d}-{name}.json'
            require(path == wanted and sha(path) == pin['sha256'], 'artifact path/hash differs')
            docs.append(load(path))
        public, geometry = docs
        n, count = validate_episode(public, geometry, identity)
        require(all(descriptor[k] == public[k] for k in ('steps', 'prefix_success', 'capture_stop_reason')), 'descriptor boundary differs')
        steps += n; contacts += count; indexed[key] = public, geometry
        reset_key = key[0], key[3]
        if reset_key in resets:
            require(resets[reset_key] == public['reset'] and headers[reset_key] == geometry['header'], 'reset geometry mismatch')
            comparisons += 1
        else:
            resets[reset_key], headers[reset_key] = public['reset'], geometry['header']
    require(set(indexed) == expected and result['rollout']['reset_references'] == len(resets) == 20 and
            result['rollout']['reset_exact_comparisons'] == comparisons == 100, 'incomplete paired resets')
    verified = result['endpoints_verified_before_environment'];after = result['rollout']['weights_after']
    require(len(verified) == len(after) == 30, 'endpoint verification incomplete')
    for rows, phase in [(verified, 'before'), (after, 'after')]:
        seen = set()
        for row in rows:
            key = row['task'], row['seed'], row['arm'];require(key in endpoints and key not in seen, 'weight identity differs');seen.add(key)
            pin = endpoints[key]['checkpoint']
            if phase == 'before':require(row['verified_file_sha256'] == pin['sha256'] and row['verified_weight_hash'] == pin['weight_hash'], 'weight/file pin differs')
            else:require(row['weight_hash'] == pin['weight_hash'] and row['no_gradients_or_trainable_parameters'] is True, 'Policy mutated')
    require(set(result['initial_state_access']) == set(map(str, science.TASKS)), 'initial task scope differs')
    for task, access in result['initial_state_access'].items():
        pin = config['initial_banks'][task]
        require(access['numeric_rows_decoded'] == [0, 1, 2, 3] and access['skipped_rows_numerically_decoded'] == [] and
                access['skipped_rows_buffered'] is False and access['used_pickle_or_torch_deserialization'] is False and
                access['container_sha256'] == pin['sha256'] and access['pickle_header_sha256'] == pin['pickle_header_sha256'] and
                access['global_to_local_index'] == {str(i): i for i in range(4)}, 'initial row boundary differs')
    counts = result['ledger']['counts'];rows = steps+120
    expected_counts = {'model_load': 30, 'policy_object': 30, 'initial_container': 5, 'numeric_row': 20,
        'adapter': 120, 'native_reset': 120, 'episode': 120, 'native_control': steps+600,
        'physics_step': 25*(steps+600), 'zero_action': 600, 'policy_forward': steps, 'policy_action': steps,
        'shadow_allocation': 120, 'shadow_forward': rows, 'geometry_row': rows, 'public_row': rows,
        'contact_record': contacts, 'forbidden_observer_native_call': 0, 'optimizer_update': 0,
        'HDF5_read': 0, 'sealed_payload_read': 0, 'old_trajectory_read': 0}
    require(set(counts) == set(expected_counts), 'operation ledger scope differs')
    for key, n in expected_counts.items():
        require(counts[key] == {'attempted': n, 'completed': n} and n <= result['ledger']['caps'][key], 'actual operation count differs: '+key)
    require(result['ledger']['elapsed_seconds'] <= 3600, 'wall budget exceeded')
    return indexed, {'prefixes': 120, 'pairs': 60, 'policy_actions': steps, 'zero_actions': 600,
                     'native_physics_steps': 25*(steps+600), 'geometry_rows': rows, 'bowl_contact_records': contacts,
                     'exact_reset_comparisons': 100}


def audit(path, output, tables):
    require(not Path(output).exists() and not Path(tables).exists(), 'audit outputs must be new')
    result = load(path); provenance = result['provenance']
    config = science.preflight(provenance['config_path'], provenance['science_binding_path'])
    binding = load(provenance['execution_binding_path'])
    require(sha(provenance['config_path']) == provenance['config_sha256'] == binding['config_sha256'] and
            sha(provenance['science_binding_path']) == provenance['science_binding_sha256'] == binding['science_binding_sha256'] and
            sha(provenance['execution_binding_path']) == provenance['execution_binding_sha256'] and
            binding['execution_binding_complete'] is True and binding['status'] == 'source_geometry_execution_bound_before_payload',
            'execution binding provenance differs')
    for pin in binding['files'].values():
        p = Path(pin['path']);p = p if p.is_absolute() else ROOT/p
        require(sha(p) == pin['sha256'], 'bound implementation changed')
    require(Path(path) == Path(binding['output']) and result['ledger']['caps'] == binding['caps'], 'output/caps differ')
    require(binding['caps'] == expected_caps(config), 'bound scientific budget differs')
    for key, cap in binding['caps'].items():
        count = result['ledger']['counts'][key]
        require(0 <= count['completed'] <= count['attempted'] <= cap, 'invalid attempted/completed count')
    require(sha(result['event_log']['path']) == result['event_log']['sha256'], 'event log changed')
    events = [json.loads(line) for line in Path(result['event_log']['path']).read_text().splitlines()]
    if result['status'] != 'complete_capture_pending_independent_audit':
        report = {'status': 'audited_technical_failure_no_primary_summary', 'result_sha256': sha(path),
                  'failure': result.get('failure'), 'ledger': result['ledger'], 'new_policy_optimizer_environment_or_initial_calls': 0}
    else:
        event_counts = Counter(e['event'] for e in events)
        require(event_counts == {'all30_endpoints_verified_before_environment': 1, 'source_initial_rows_decoded': 5,
                                 'paired_reset_verified': 120, 'episode_complete': 120, 'model_complete': 30}, 'event accounting differs')
        require(events[0]['event'] == 'all30_endpoints_verified_before_environment', 'environment before all endpoint checks')
        indexed, counts = validate_capture(result, config, binding['artifacts'])
        tables_data = analyze_pairs(indexed);Path(tables).mkdir(parents=True, exist_ok=False)
        pins = {}
        for name, rows in tables_data.items():
            p = Path(tables)/(name+'.csv')
            write_table(p, rows)
            pins[name] = {'path': str(p), 'sha256': sha(p), 'rows': len(rows)}
        report = {'status': 'passed_independent_geometry_capture_audit', 'formal_result': False,
                  'result_sha256': sha(path), 'verified_at_utc': datetime.now(timezone.utc).isoformat(),
                  'actual_counts': counts, 'table_artifacts': pins,
                  'prefix_outcomes': dict(Counter(r['prefix_outcome'] for r in tables_data['pairs'])),
                  'episode_features': tables_data['episodes'], 'task_seed_arms': tables_data['task_seed_arms'],
                  'new_policy_optimizer_environment_or_initial_calls': 0, 'new_checkpoint_deserializations': 0,
                  'G_decision_unchanged': 'failed_keep_H_sealed',
                  'limits': ['Geometry/collision boundary proxies, not forces or stable grasp labels.',
                             'Fresh source prefixes, not old G/validation36 reproduction or 300-step success rates.',
                             'Saved-field, passivity-witness and code/metadata audit; no new physics replay.']}
    with Path(output).open('x') as handle:json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False);handle.write('\n')
    print(json.dumps({'status': report['status'], 'output': str(output), 'actual_counts': report.get('actual_counts')}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tables', type=Path, required=True)
    args = parser.parse_args();report = audit(args.result, args.output, args.tables)
    if report['status'] != 'passed_independent_geometry_capture_audit':raise SystemExit(1)


if __name__ == '__main__':
    main()
