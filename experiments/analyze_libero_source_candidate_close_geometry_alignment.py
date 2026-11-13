"""Read-only alignment of the existing source prefixes; no numerical runtime.

Every positive command run uses states start-1..end.  Touch is a geometric
boundary proxy, never a force/grasp label.  Paired rows use absolute control
indices and retain unavailable post-terminal values as null.
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
import platform
import statistics
import time

from experiments import audit_libero_source_candidate_geometry_capture as saved

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / 'configs/libero-source-candidate-close-geometry-alignment-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-source-candidate-close-geometry-alignment-binding_v2.json'
TASKS, SEEDS, ARMS = [0, 1, 3, 4, 7], [20261106, 20261107, 20261108], ['A_original', 'B_fomaml']
STATE_FIELDS = ['eef_to_bowl_xyz', 'eef_to_bowl_xy', 'eef_minus_bowl_signed_z',
                'bowl_height_change', 'bowl_to_plate_xyz', 'bowl_to_plate_xy', 'bilateral_touch_proxy']
NEW_CALLS = dict.fromkeys(['Policy_forward', 'optimizer_update', 'environment', 'reset', 'shadow_forward',
                          'checkpoint', 'initial_bank', 'HDF5', 'RGB', 'sealed_payload'], 0)
require = saved.require


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    return {'path': str(path), 'sha256': sha(path)}


def load_pin(item):
    path = Path(item['path'])
    path = path if path.is_absolute() else ROOT / path
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == item['sha256'], f'input hash differs: {path}')
    return json.loads(raw)


def sign(value):
    require(type(value) in (int, float) and math.isfinite(value), 'nonfinite command')
    return (value > 0) - (value < 0)


def stats(values):
    values = [v for v in values if v is not None]
    return {'observed': len(values), 'mean': statistics.fmean(values) if values else None,
            'median': statistics.median(values) if values else None,
            'minimum': min(values) if values else None, 'maximum': max(values) if values else None}


def command_runs(public):
    actions = public['rows'][1:]
    runs = []
    for ordinal, (s, items) in enumerate(groupby(enumerate(actions, 1), key=lambda r: sign(r[1]['action'][6])), 1):
        items = list(items)
        start, end = items[0][0], items[-1][0]
        runs.append({'run_ordinal': ordinal, 'sign': s, 'start': start, 'end': end,
                     'observed_length': end-start+1, 'right_censored': end == public['steps']})
    return runs


def touch_blocks(geo, start, end):
    blocks = []
    for touch, items in groupby(range(start, end+1), key=lambda t: geo[t]['bilateral_touch_proxy']):
        indices = list(items)
        if touch:
            blocks.append((indices[0], indices[-1]))
    return blocks


def close_features(public, geometry, run, identity):
    geo = geometry['rows']
    start, end = run['start'], run['end']
    lo = start-1
    before, after = geo[lo], geo[end]
    blocks = touch_blocks(geo, lo, end)
    first = blocks[0][0] if blocks else None
    first_block_end = blocks[0][1] if blocks else None
    following = None if end == public['steps'] else sign(public['rows'][end+1]['action'][6])
    immediate_open = end+1 if following == -1 else None
    row = {**identity, 'prefix_success': public['prefix_success'], **run,
           'pre_state': lo, 'post_state': end, 'duration_observed_seconds': geo[end]['sim_time']-before['sim_time'],
           'complete_one_step_pulse': run['observed_length'] == 1 and not run['right_censored'],
           'touch_at_pre': before['bilateral_touch_proxy'], 'touch_at_end': after['bilateral_touch_proxy'],
           'touch_blocks_in_close': len(blocks),
           'touch_boundary_count_in_close': sum(b-a+1 for a, b in blocks),
           'first_touch_state_in_close': first,
           'first_touch_seconds_from_close_onset': None if first is None else geo[first]['sim_time']-before['sim_time'],
           'first_touch_block_end_state': first_block_end,
           'first_touch_block_ends_inside_close': None if first is None else first_block_end < end,
           'following_command_sign': following, 'immediate_open_action': immediate_open,
           'first_touch_to_close_end_seconds': None if first is None else geo[end]['sim_time']-geo[first]['sim_time']}
    for field in STATE_FIELDS[:-1]:
        row['pre_'+field] = before[field]
        row['end_'+field] = after[field]
        row['delta_'+field] = after[field]-before[field]
        row['min_'+field] = min(g[field] for g in geo[lo:end+1])
        row['max_'+field] = max(g[field] for g in geo[lo:end+1])
    for field in ['bowl_height_change', 'bowl_to_plate_xyz', 'bowl_to_plate_xy']:
        row['touch_'+field] = None if first is None else geo[first][field]
        row['touch_to_end_delta_'+field] = None if first is None else after[field]-geo[first][field]
        row['touch_to_end_min_delta_'+field] = None if first is None else min(g[field] for g in geo[first:end+1])-geo[first][field]
        row['touch_to_end_max_delta_'+field] = None if first is None else max(g[field] for g in geo[first:end+1])-geo[first][field]
    for field in STATE_FIELDS:
        row['immediate_open_post_'+field] = None if immediate_open is None else geo[immediate_open][field]
    return row


def episode_features(public, geometry, close_rows, identity):
    geo = geometry['rows']
    touched = [g['state_index'] for g in geo if g['bilateral_touch_proxy']]
    first = touched[0] if touched else None
    last = touched[-1] if touched else None
    contact_closes = [r for r in close_rows if r['first_touch_state_in_close'] is not None]
    multi = next((r for r in close_rows if r['observed_length'] >= 2), None)
    row = {**identity, 'steps': public['steps'], 'prefix_success': public['prefix_success'],
           'capture_stop_reason': public['capture_stop_reason'], 'close_runs': len(close_rows),
           'multi_close_runs': sum(r['observed_length'] >= 2 for r in close_rows),
           'close_runs_with_touch': len(contact_closes), 'close_runs_without_touch': len(close_rows)-len(contact_closes),
           'first_touch_state': first, 'last_touch_state': last,
           'touch_boundary_count': len(touched), 'touch_blocks_in_prefix': len(touch_blocks(geo, 0, public['steps'])),
           'touch_at_terminal': geo[-1]['bilateral_touch_proxy'],
           'first_multi_pre_eef_distance': None if multi is None else multi['pre_eef_to_bowl_xyz'],
           'first_multi_pre_plate_distance_xy': None if multi is None else multi['pre_bowl_to_plate_xy'],
           'first_close_step': None if not close_rows else close_rows[0]['start'],
           'first_multi_close_step': None if multi is None else multi['start'],
           'complete_first_close_pulse': bool(close_rows and close_rows[0]['complete_one_step_pulse'])}
    for field in ['bowl_height_change', 'bowl_to_plate_xyz', 'bowl_to_plate_xy', 'eef_to_bowl_xyz']:
        row['reset_'+field] = geo[0][field]
        row['terminal_'+field] = geo[-1][field]
        row['terminal_minus_reset_'+field] = geo[-1][field]-geo[0][field]
        row['minimum_'+field] = min(g[field] for g in geo)
        row['maximum_'+field] = max(g[field] for g in geo)
        row['at_first_touch_'+field] = None if first is None else geo[first][field]
        row['first_touch_to_terminal_delta_'+field] = None if first is None else geo[-1][field]-geo[first][field]
        row['first_touch_to_terminal_min_delta_'+field] = None if first is None else min(g[field] for g in geo[first:])-geo[first][field]
        row['first_touch_to_terminal_max_delta_'+field] = None if first is None else max(g[field] for g in geo[first:])-geo[first][field]
    return row


def aligned_frames(indexed):
    rows = []
    for task in TASKS:
        for seed in SEEDS:
            for index in range(4):
                docs = [indexed[(task, seed, arm, index)] for arm in ARMS]
                n = [p['steps'] for p, _ in docs]
                for state in range(max(n)+1):
                    row = {'task': task, 'seed': seed, 'global_initial_index': index, 'state_index': state,
                           'jointly_available': state <= min(n)}
                    for arm, (public, geometry) in zip(('A', 'B'), docs):
                        available = state <= public['steps']
                        row[arm+'_available'] = available
                        row[arm+'_sim_time'] = geometry['rows'][state]['sim_time'] if available else None
                        row[arm+'_command_to_state'] = public['rows'][state]['action'][6] if available and state else None
                        for field in STATE_FIELDS:
                            row[arm+'_'+field] = geometry['rows'][state][field] if available else None
                    if row['jointly_available']:
                        saved.close(row['A_sim_time'], row['B_sim_time'], 1e-9)
                    for field in STATE_FIELDS[:-1]:
                        row['B_minus_A_'+field] = row['B_'+field]-row['A_'+field] if row['jointly_available'] else None
                    rows.append(row)
    return rows


def event_windows(indexed, close_rows, half_window=5):
    anchors = {}
    for row in close_rows:
        case = row['task'], row['seed'], row['global_initial_index']
        events = [('close_onset_pre', row['pre_state']), ('close_end_post', row['end']),
                  ('first_touch_in_close', row['first_touch_state_in_close']),
                  ('immediate_open_post', row['immediate_open_action'])]
        for label, state in events:
            if state is not None:
                anchors.setdefault((*case, state), []).append(f"{row['arm']}:run{row['run_ordinal']}:{label}")
    rows = []
    for (task, seed, index, anchor), labels in sorted(anchors.items()):
        for offset in range(-half_window, half_window+1):
            state = anchor+offset
            row = {'task': task, 'seed': seed, 'global_initial_index': index,
                   'anchor_state': anchor, 'anchor_aliases': '|'.join(sorted(labels)),
                   'offset': offset, 'absolute_state_index': state, 'reset_side_truncated': state < 0}
            for name, arm in zip(('A', 'B'), ARMS):
                public, geometry = indexed[(task, seed, arm, index)]
                available = 0 <= state <= public['steps']
                row[name+'_available'] = available
                row[name+'_terminal_truncated'] = state > public['steps']
                row[name+'_command_to_state'] = public['rows'][state]['action'][6] if available and state else None
                for field in STATE_FIELDS:
                    row[name+'_'+field] = geometry['rows'][state][field] if available else None
            rows.append(row)
    return rows


def group_summaries(episodes, closes):
    result = []
    keys = [(None, None, None)] + [(None, None, arm) for arm in ARMS]
    keys += [(task, None, arm) for task in TASKS for arm in ARMS]
    keys += [(task, seed, arm) for task in TASKS for seed in SEEDS for arm in ARMS]
    for task, seed, arm in keys:
        match = lambda r: (task is None or r['task'] == task) and (seed is None or r['seed'] == seed) and (arm is None or r['arm'] == arm)
        er, cr = list(filter(match, episodes)), list(filter(match, closes))
        touches = [r for r in er if r['first_touch_state'] is not None]
        tcr = [r for r in cr if r['first_touch_state_in_close'] is not None]
        result.append({'task': task, 'seed': seed, 'arm': arm, 'prefixes': len(er),
            'prefix_native_goals': sum(r['prefix_success'] for r in er),
            'no_close_prefixes': sum(r['close_runs'] == 0 for r in er), 'touch_prefixes': len(touches),
            'close_runs': len(cr), 'close_runs_with_touch': len(tcr),
            'close_runs_touch_block_ends_inside': sum(r['first_touch_block_ends_inside_close'] is True for r in cr),
            'close_runs_touch_at_end': sum(r['touch_at_end'] for r in cr),
            'close_runs_right_censored': sum(r['right_censored'] for r in cr),
            'multi_close_pre_distance': stats(r['first_multi_pre_eef_distance'] for r in er),
            'maximum_height_from_reset': stats(r['maximum_bowl_height_change'] for r in er),
            'terminal_plate_xy_from_reset': stats(r['terminal_minus_reset_bowl_to_plate_xy'] for r in er),
            'touch_to_terminal_height_delta': stats(r['first_touch_to_terminal_delta_bowl_height_change'] for r in er),
            'touch_to_terminal_max_height_delta': stats(r['first_touch_to_terminal_max_delta_bowl_height_change'] for r in er),
            'touch_to_terminal_plate_xy_delta': stats(r['first_touch_to_terminal_delta_bowl_to_plate_xy'] for r in er),
            'close_touch_to_end_height_delta': stats(r['touch_to_end_delta_bowl_height_change'] for r in cr),
            'close_touch_to_end_plate_xy_delta': stats(r['touch_to_end_delta_bowl_to_plate_xy'] for r in cr)})
    return result


def pair_summaries(indexed, episodes, frames):
    lookup = {(e['task'], e['seed'], e['arm'], e['global_initial_index']): e for e in episodes}
    frame_groups = {}
    for row in frames:
        if row['jointly_available']:
            frame_groups.setdefault((row['task'], row['seed'], row['global_initial_index']), []).append(row)
    result = []
    for case, rows in sorted(frame_groups.items()):
        task, seed, index = case
        a, b = [lookup[(task, seed, arm, index)] for arm in ARMS]
        row = {'task': task, 'seed': seed, 'global_initial_index': index,
               'common_last_state': rows[-1]['state_index'], 'common_state_count': len(rows),
               'A_prefix_success': a['prefix_success'], 'B_prefix_success': b['prefix_success'],
               'outcome': 'both' if a['prefix_success'] and b['prefix_success'] else 'A_only' if a['prefix_success'] else 'B_only' if b['prefix_success'] else 'neither'}
        for arm, doc, ep in zip(('A','B'), [indexed[(task, seed, q, index)] for q in ARMS], (a,b)):
            last = rows[-1]['state_index']
            common = doc[1]['rows'][:last+1]
            row[arm+'_first_multi_close_step'] = ep['first_multi_close_step']
            row[arm+'_first_multi_pre_distance'] = ep['first_multi_pre_eef_distance']
            row[arm+'_common_max_height'] = max(g['bowl_height_change'] for g in common)
            row[arm+'_common_terminal_plate_xy'] = common[-1]['bowl_to_plate_xy']
            row[arm+'_common_touch_states'] = sum(g['bilateral_touch_proxy'] for g in common)
        row['common_B_minus_A_max_height'] = row['B_common_max_height']-row['A_common_max_height']
        row['common_B_minus_A_terminal_plate_xy'] = row['B_common_terminal_plate_xy']-row['A_common_terminal_plate_xy']
        result.append(row)
    return result


def validate_parent_scope(raw, audit, done, prior_runs):
    require(raw['G_decision_unchanged'] == audit['G_decision_unchanged'] == 'failed_keep_H_sealed' and
            done['G_admission_remains_failed_H_sealed'] is True and raw['new_training_updates'] == 0 and
            audit['actual_counts']['prefixes'] == 120 and len(prior_runs) == 359, 'parent cohort differs')


def preflight(config_path, binding_path):
    config, binding = (json.loads(Path(p).read_text()) for p in (config_path, binding_path))
    require(binding['config_sha256'] == sha(config_path), 'analysis specification changed')
    require(config['tasks'] == TASKS and config['seeds'] == SEEDS and config['arms'] == ARMS and
            config['global_initial_indices'] == [0,1,2,3] and config['candidate_role'] == 'candidate_nonregression' and
            config['new_research_calls'] == NEW_CALLS and config['neighborhood_half_window'] == 5 and
            config['prior_outcomes_known'] is True and config['formal_result'] is False, 'analysis role/budget differs')
    require(set(config['inputs']) == {'capture','audit','completion','interpretation','prior_runs'}, 'unexpected inputs')
    for item in binding['files'].values():
        path = Path(item['path']); path = path if path.is_absolute() else ROOT/path
        require(sha(path) == item['sha256'], f'bound analysis file changed: {path}')
    require((ROOT/binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'wrong analysis runner')
    docs = {k: load_pin(p) for k, p in config['inputs'].items() if k != 'prior_runs'}
    p = Path(config['inputs']['prior_runs']['path'])
    require(sha(p) == config['inputs']['prior_runs']['sha256'], 'prior run table changed')
    with p.open(newline='') as f: docs['prior_runs'] = list(csv.DictReader(f))
    raw, audit, done = [docs[k] for k in ['capture','audit','completion']]
    require(audit['status'] == 'passed_independent_geometry_capture_audit' and
            done['status'] == 'complete_independently_audited_source_geometry_diagnostic' and
            audit['result_sha256'] == config['inputs']['capture']['sha256'] == done['result']['sha256'] and
            done['independent_audit']['sha256'] == config['inputs']['audit']['sha256'] and
            done['interpretation']['sha256'] == config['inputs']['interpretation']['sha256'], 'missing audited lineage')
    validate_parent_scope(raw, audit, done, docs['prior_runs'])
    return config, binding, docs


def read_cohort(config, docs):
    indexed, rows_read = {}, 0
    archive = Path(config['capture_root']).resolve()
    expected = {(t,s,a,i) for t in TASKS for s in SEEDS for a in ARMS for i in range(4)}
    for item in docs['capture']['rollout']['episode_files']:
        key = tuple(item[x] for x in ['task','seed','arm','global_initial_index'])
        require(key in expected and key not in indexed and item['initial_state_role'] == 'candidate_nonregression' and
                item['checkpoint_update'] == 500 and item['stabilization_steps'] == 5, 'source cohort changed')
        loaded = []
        for label in ['public','geometry']:
            path = Path(item[label]['path'])
            require(path.resolve().parent == archive and path.name.endswith('-'+label+'.json'), 'path outside capture allowlist')
            loaded.append(load_pin(item[label]))
        public, geometry = loaded
        identity = {k:v for k,v in item.items() if k not in ('steps','prefix_success','capture_stop_reason','public','geometry')}
        n, _ = saved.validate_episode(public, geometry, identity)
        require(n == item['steps'] and public['prefix_success'] is item['prefix_success'], 'parent episode mismatch')
        indexed[key] = public, geometry
        rows_read += n+1
    require(set(indexed) == expected and rows_read == 17247, 'incomplete source prefixes')
    return indexed, rows_read


def tables(indexed, prior_runs):
    close_rows, episode_rows, run_identity = [], [], []
    for (task,seed,arm,index), (public,geometry) in sorted(indexed.items()):
        identity = {'task':task,'seed':seed,'arm':arm,'global_initial_index':index}
        runs = command_runs(public)
        for r in runs:
            run_identity.append((task,seed,arm,index,r['run_ordinal'],r['sign'],r['start'],r['end'],r['observed_length'],r['right_censored']))
        cr = [close_features(public,geometry,r,identity) for r in runs if r['sign'] == 1]
        close_rows.extend(cr)
        episode_rows.append(episode_features(public,geometry,cr,identity))
    cached = [(int(r['task']),int(r['seed']),r['arm'],int(r['global_initial_index']),int(r['run_ordinal']),int(r['sign']),
               int(r['start']),int(r['end']),int(r['observed_length']),r['right_censored'] == 'True') for r in prior_runs]
    require(sorted(run_identity) == sorted(cached), 'saved run segmentation differs')
    require(len(close_rows) == 170 and len(episode_rows) == 120, 'run/prefix accounting differs')
    frames = aligned_frames(indexed)
    return {'close_runs':close_rows, 'episodes':episode_rows, 'pairs':pair_summaries(indexed,episode_rows,frames),
            'aligned_states':frames, 'event_windows':event_windows(indexed,close_rows)}


def write_csv(path, rows):
    require(rows, 'empty primary table')
    with Path(path).open('x',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(config_path, binding_path, output, directory):
    started=time.monotonic()
    output, directory = Path(output), Path(directory)
    require(not output.exists() and not directory.exists(), 'refuse existing analysis artifacts')
    config, binding, docs = preflight(config_path,binding_path)
    indexed, row_count = read_cohort(config,docs)
    data = tables(indexed,docs['prior_runs'])
    groups = group_summaries(data['episodes'],data['close_runs'])
    directory.mkdir(parents=True,exist_ok=False)
    artifacts={}
    for label, rows in data.items():
        path=directory/(label+'.csv');write_csv(path,rows)
        artifacts[label]={**pin(path),'rows':len(rows)}
    result={'status':'complete_readonly_close_geometry_alignment_pending_independent_audit',
        'formal_result':False,'diagnostic_only':True,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'config':pin(config_path),'binding':pin(binding_path),'inputs':config['inputs'],
        'rows_read':row_count,'capture_json_files_read':240,'resource_new_calls':NEW_CALLS,
        'tables':artifacts,'groups':groups,'prefix_outcomes':dict(Counter(r['outcome'] for r in data['pairs'])),
        'G_decision_unchanged':True,'SPT_Knowledge_Module_unchanged':True,
        'runtime':{'python':platform.python_version(),'device':'CPU_stdlib_only','elapsed_seconds':time.monotonic()-started},
        'limits':['Saved boundary collision proxy, not force or stable grasp.',
                  'Run-weighted and episode-weighted statistics are separate.',
                  'Same absolute paired time; no terminal fill or distance/phase thresholds.',
                  'Known source150 prefixes; not validation36, 300-step success rates or causal effects.']}
    with output.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    p.add_argument('--binding',type=Path,default=DEFAULT_BINDING)
    p.add_argument('--preflight',action='store_true')
    p.add_argument('--output',type=Path,default=Path('/tmp/libero-source-candidate-close-geometry-alignment-20261107-v1.json'))
    p.add_argument('--tables',type=Path,default=Path('/tmp/libero-source-candidate-close-geometry-alignment-tables-20261107-v1'))
    args=p.parse_args()
    if args.preflight:
        preflight(args.config,args.binding);print('passed_metadata_only_readonly_alignment_preflight');return
    result=run(args.config,args.binding,args.output,args.tables)
    print(json.dumps({'status':result['status'],'table_rows':{k:v['rows'] for k,v in result['tables'].items()},
                      'resource_new_calls':result['resource_new_calls']},ensure_ascii=False))


if __name__ == '__main__':main()
