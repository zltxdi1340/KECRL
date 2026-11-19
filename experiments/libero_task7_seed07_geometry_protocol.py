"""Nine archived seed07 trajectories; bounded geometry, no policy execution."""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import numpy as np

from experiments import analyze_libero_task7_v2_v3_crosscheck as cross
from experiments import libero_postgrasp_geometry_protocol_v2 as geometry
from experiments import analyze_libero_postgrasp_geometry as legacy

ROOT, require, pin, write_new = geometry.ROOT, geometry.require, geometry.pin, geometry.write_new
SHAPES, OBJECTS, previous = geometry.SHAPES, geometry.OBJECTS, geometry.previous
NATIVE, GEOMETRY, GEOMETRY_WIDTH = geometry.NATIVE, geometry.GEOMETRY, geometry.GEOMETRY_WIDTH
native, old, RGB_RULE = geometry.native, geometry.old, geometry.RGB_RULE
CAUSES = ['native_success'] + geometry.CAUSES
ARMS = ['control', 'v2_xyz', 'v3_spatial']
caps, worker_caps, geometry_arrays, rgb_metrics = geometry.caps, geometry.worker_caps, geometry.geometry_arrays, geometry.rgb_metrics


def selection(c):
    jobs, warm = [], []
    parents = {name: json.loads(Path(p['result']['path']).read_text()) for name, p in c['parents'].items()}
    for item in c['archives']:
        arm = {'control': cross.ARMS['v3'][0], 'v2_xyz': cross.ARMS['v2'][1], 'v3_spatial': cross.ARMS['v3'][1]}[item['controller']]
        result = parents[item['family']]
        row = next(r for r in result['episodes'] if r['source']['source_id'] == item['source_id'] and r['controller'] == arm)
        require(row['frames'] == item['frames'], 'selected frame descriptor differs')
        identity = item['source_id'] + '-' + item['controller']
        jobs.append(dict(problem_id=identity, source_id=item['source_id'], seed=row['seed'], controller=item['controller'],
                         source_controller=arm, family=item['family'], initial_state_index=row['initial_state_index'],
                         anchor_step=row['anchor_step'], steps=row['steps'], frames=row['frames'], success=row['success'],
                         terminal_reason=row['terminal_reason'], observation_sha256=row['observation_sha256'],
                         initial_row_sha256=row['initial_row_sha256']))
        rows = [r for r in result['warmups'] if r['problem_id'] == row['problem_id']]
        require([r['initial_state_index'] for r in rows] == list(range(row['initial_state_index'])), 'warm reset coverage differs')
        warm.extend({**r, 'problem_id': identity} for r in rows)
    return jobs, warm


def preflight(config, binding):
    c = json.loads(Path(config).read_text()); b = json.loads(Path(binding).read_text())
    require(c['kind'] == 'libero_task7_seed07_geometry' and c['version'] == 1 and c['task'] == 7
            and c['formal_result'] is False and c['initial_indices'] == list(range(19))
            and c['objects'] == OBJECTS and c['RGB_rule'] == RGB_RULE and c['cause_order'] == CAUSES
            and c['stabilization_steps'] == 5 and c['horizon'] == 300 and c['image_size'] == 128
            and c['environment_seed'] == 20261022 and c['wall_seconds_max'] == 1800
            and c['policy_inference'] is False and c['model_training_allowed'] is False
            and c['one_fresh_process_per_trajectory'] is True and c['no_automatic_retry'] is True
            and c['new_success_rate_estimate'] is False and c['original_private_state_recovered'] is False,
            'targeted geometry scope differs')
    require(b['status'] == 'bound_before_seed07_geometry_payload' and b['science'] == pin(config), 'new geometry binding required')
    for item in b['code_dependencies'] + b['validation'] + list(c['parents'].values()):
        require(pin(item['path']) == item, 'targeted dependency changed: ' + item['path'])
    cc = cross.preflight(c['parents']['cross_config']['path'], c['parents']['cross_binding']['path'])
    proof = json.loads(Path(c['parents']['cross_completion']['path']).read_text())
    require(proof['status'] == 'complete_readonly_v2_v3_crosscheck' and pin(proof['analysis']['path']) == proof['analysis'], 'completed crosscheck required')
    _, prior = geometry.preflight(c['parents']['geometry_config']['path'], c['parents']['geometry_binding']['path'])
    closure = {r['path']: r['sha256'] for r in b['code_dependencies']}
    for parent in [prior, json.loads(Path(c['parents']['cross_binding']['path']).read_text())]:
        require({r['path']: r['sha256'] for r in parent.get('dependencies', parent.get('code_dependencies', []))}.items() <= closure.items(), 'inherited code closure missing')
    for name in ['libero_task7_seed07_geometry_protocol.py', 'run_libero_task7_seed07_geometry.py',
                 'execute_libero_task7_seed07_geometry.py', 'audit_libero_task7_seed07_geometry.py']:
        path = ROOT/'experiments'/name
        require(closure.get(str(path)) == pin(path)['sha256'], 'targeted code unbound')
    jobs, warm = selection(cc)
    require(c['problems'] == jobs and c['warmup_references'] == warm and len(jobs) == 9, 'all nine selection differs')
    require(c['caps'] == caps(jobs) and c['caps']['native_steps'] == 2931
            and sum(j['anchor_step'] for j in jobs) == 900, 'authorized total2931 scope differs')
    parent_cfg = json.loads(Path(cc['parents']['v3']['science']['path']).read_text())
    for key in ['initial_bank', 'runtime_environment', 'versions', 'libero_revision']:
        require(c[key] == parent_cfg[key], 'source runtime/initial provenance differs')
    require(Path(c['output']).resolve().is_relative_to(ROOT/'results'), 'persistent output required')
    grant = b['authorization']; require(pin(grant['path']) == grant, 'geometry authority changed')
    g = json.loads(Path(grant['path']).read_text())
    require(g['user_approved'] is True and g['approved_native_steps'] == 2931 and g['science'] == pin(config)
            and g['caps'] == c['caps'] and g['initial_indices'] == list(range(19)) and g['objects'] == OBJECTS
            and g['source_archives'] == [j['frames'] for j in jobs] and g['sealed_roles_allowed'] == []
            and g['training_allowed'] is False and g['model_reads_allowed'] is False, 'geometry authorization differs')
    return c, b


def read_original(item, steps, ledger, *, warm=False, family=None, success=False):
    require(pin(item['path']) == item, 'original archive changed')
    numeric = ['joint_states', 'gripper_states'] if warm else previous.NUMERIC
    expected = list(SHAPES) if warm else previous.MEMBERS
    with np.load(item['path'], allow_pickle=False) as archive:
        members = set(archive.files)
        allowed = [set(expected), set(expected) | {'eef_pos'}] if warm else [set(expected) | ({'eef_pos'} if family == 'v2' else set())]
        require(len(archive.files) == len(members) and members in allowed, 'source member whitelist differs')
        arrays = {k: archive[k] for k in numeric}
        for k in ['agentview_rgb', 'eye_in_hand_rgb']:
            frames = 1 if warm else steps+1
            ledger.begin('source_RGB_members'); ledger.begin('source_rgb_decodes', frames)
            ledger.begin('source_RGB_pixel_values', frames*49152)
            arrays[k] = archive[k]
            require(arrays[k].shape == ((128, 128, 3) if warm else (steps+1, 128, 128, 3)) and arrays[k].dtype == np.uint8, 'source RGB schema differs')
            ledger.finish('source_RGB_pixel_values', frames*49152); ledger.finish('source_rgb_decodes', frames); ledger.finish('source_RGB_members')
    specs = [('joint_states', (7,) if warm else (steps+1, 7), np.float32),
             ('gripper_states', (2,) if warm else (steps+1, 2), np.float32)]
    if not warm:
        specs += [('actions', (steps, 7), np.float32), ('rewards', (steps,), np.float64),
                  ('terminated', (steps,), np.bool_), ('truncated', (steps,), np.bool_)]
    for k, shape, dtype in specs:
        require(arrays[k].shape == shape and arrays[k].dtype == dtype and np.isfinite(arrays[k]).all(), 'source numeric schema differs')
    if not warm:
        require(np.abs(arrays['actions']).max() <= 1 and not arrays['terminated'][:-1].any()
                and not arrays['truncated'][:-1].any() and bool(arrays['terminated'][-1]) == success
                and bool(arrays['truncated'][-1]) != success, 'source outcome differs')
    for a in arrays.values(): a.flags.writeable = False
    return arrays


def load_source(job, ledger):
    return read_original(job['frames'], job['steps'], ledger, family=job['family'], success=job['success'])


def load_warm(item, ledger):
    size = Path(item['path']).stat().st_size
    ledger.begin('source_container_bytes', size); ledger.begin('source_warm_numeric_members', 2); ledger.begin('source_warm_numeric_scalars', 9)
    value = ledger.call('source_warm_archives', read_original, item, 0, ledger, warm=True)
    ledger.finish('source_container_bytes', size); ledger.finish('source_warm_numeric_members', 2); ledger.finish('source_warm_numeric_scalars', 9)
    return value


def validate_geometry(arrays, job):
    geometry.validate_geometry(arrays, job)
    expected = np.zeros(len(arrays['step']), dtype=bool); expected[-1] = job['success']
    require(np.array_equal(arrays['native_On'], expected) and np.array_equal(arrays['native_success'], expected), 'geometry/source terminal success differs')


def trajectory(job, g, actions):
    validate_geometry(g, job)
    r = legacy.trajectory(job, {k: g[k] for k in old.GEOMETRY}, actions)
    minimum, command = r['windows']['all']['min_xy'], r['last_nonzero_gripper_before_loss']
    if job['success']: cause = 'native_success'
    elif not r['handoff_any_finger_contact']: cause = CAUSES[1]
    elif r['first_loss_with_descent'] is not None:
        cause = 'open_command_loss_descent' if command is not None and command < 0 else 'close_command_loss_descent' if command is not None and command > 0 else 'loss_descent_command_unknown'
    elif minimum >= .10: cause = 'xy_ge10cm'
    elif minimum >= .06: cause = 'xy_6to10cm'
    elif minimum >= .05: cause = 'xy_5to6cm'
    elif minimum >= .03: cause = 'xy_3to5cm'
    elif r['aligned']['0.03']['envelope_gap_min'] > .03 and r['aligned']['0.03']['contact_frames'] == 0: cause = 'aligned_high_no_contact'
    elif g['bowl_plate_contact'].any(): cause = 'plate_contact_without_On'
    else: cause = 'aligned_no_plate_contact'
    r['cause'] = cause
    touching = g['left_finger_contact'] | g['right_finger_contact']
    r.update(finger_contact_frames_all=int(touching.sum()), finger_contact_frames_last50=int(touching[-50:].sum()))
    z = g['plate_pos'][:, 2] <= g['bowl_pos'][:, 2]
    xy = np.linalg.norm((g['bowl_pos']-g['plate_pos'])[:, :2], axis=1) < .03
    for name, indices in [('all', range(len(z))), ('last50', range(max(0, len(z)-50), len(z)))]:
        r['native_component_patterns_'+name] = dict(Counter(f'Z{int(z[i])}_XY{int(xy[i])}_C{int(g["bowl_plate_contact"][i])}' for i in indices))
    r.update(terminal_xy_margin_m=.03-r['windows']['all']['final_xy'] if job['success'] else None,
             terminal_center_dz_m=float(g['bowl_pos'][-1, 2]-g['plate_pos'][-1, 2]))
    return r


def paired(jobs, cache):
    groups = {}
    for job in jobs: groups.setdefault(job['source_id'], {})[job['controller']] = job
    rows = []
    for source, group in groups.items():
        x = cache[group['control']['problem_id']]
        for arm in ARMS[1:]:
            y = cache[group[arm]['problem_id']]; n = min(len(x['step']), len(y['step'])); left, right = x['bowl_pos'][n-1]-x['plate_pos'][n-1], y['bowl_pos'][n-1]-y['plate_pos'][n-1]
            equal = all(native.same_bytes(x['native_before_'+k][0], y['native_before_'+k][0]) for k in NATIVE)
            require(equal and np.array_equal(x['step'][:n], y['step'][:n]), 'new paired native anchor/cadence differs')
            row = dict(source_id=source, candidate=arm, common_end_step=int(x['step'][n-1]), common_samples=n,
                       anchor_native_byte_equal=equal, anchor_max_native_abs_difference=0.,
                       control_xy_at_common_end=float(np.linalg.norm(left[:2])), candidate_xy_at_common_end=float(np.linalg.norm(right[:2])),
                       control_center_dz_at_common_end=float(left[2]), candidate_center_dz_at_common_end=float(right[2]),
                       control_contact_at_common_end=bool(x['bowl_plate_contact'][n-1]), candidate_contact_at_common_end=bool(y['bowl_plate_contact'][n-1]),
                       relative_bowl_offset_difference_at_common_end=(right-left).tolist())
            for obj in ['eef', 'bowl']:
                dist = np.linalg.norm(x[obj+'_pos'][:n]-y[obj+'_pos'][:n], axis=1)
                row.update({f'maximum_{obj}_separation': float(dist.max()), f'maximum_{obj}_step': int(x['step'][dist.argmax()]), f'final_{obj}_separation': float(dist[-1])})
            rows.append(row)
    return rows


def summary(rows):
    return {arm: dict(trajectories=sum(r['controller'] == arm for r in rows),
                      causes={c: sum(r['controller'] == arm and r['cause'] == c for r in rows) for c in CAUSES}) for arm in ARMS}
