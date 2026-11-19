"""Invented nine-episode 2931-step workflow; no research data or environment."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import runpy
import numpy as np
import pytest

from experiments import libero_task7_seed07_geometry_protocol as p
from experiments import run_libero_task7_seed07_geometry as run
from experiments import execute_libero_task7_seed07_geometry as driver
from experiments import audit_libero_task7_seed07_geometry as audit
from experiments.libero_flip_replay_protocol import describe

helpers = runpy.run_path(str(p.ROOT/'tests/test_libero_postgrasp_geometry_replay_v2.py'))
obs = helpers['obs']


class Adapter(helpers['Adapter']):
    def step(self, command):
        value, _, _, _, _ = super().step(command)
        terminal = self.t == self.job['steps']
        success = terminal and self.job['success']
        return value, float(success), success, terminal and not success, dict(success=success,
            terminal_reason=('success' if success else 'time_limit') if terminal else None)


class Geometry(helpers['Geometry']):
    def sample(self, step, observation):
        r = super().sample(step, observation)
        if step == self.job['steps'] and self.job['success']:
            r.update(native_On=True, native_success=True, bowl_plate_contact=True)
            r['bowl_pos'] = r['plate_pos'] + [.01, 0, .15]
            for phase in ['native_before_', 'native_after_']:
                r[phase+'bowl_pos'] = r['bowl_pos'].copy()
                r[phase+'bowl_joint_qpos'] = np.r_[r['bowl_pos'], [1., 0, 0, 0]]
        return r


def fixture(tmp_path):
    source = tmp_path/'source'; source.mkdir()
    base = tmp_path/'results/targeted'; base.mkdir(parents=True)
    states = np.arange(19*92, dtype=np.float64).reshape(19, 92)
    warm_pins = {}
    for family in ['v2', 'v3']:
        for i in range(19):
            arrays = obs(i, 0)
            if family == 'v2': arrays['eef_pos'] = np.zeros(3, np.float32)
            path = source/f'{family}-warm-{i}.npz'; np.savez_compressed(path, **arrays); warm_pins[family, i] = p.pin(path)
    jobs, warm = [], []
    for case, index, length in [('case0', 13, 130), ('case1', 13, 138), ('case2', 18, 158)]:
        for arm in p.ARMS:
            success = arm == 'control'; n = length if success else 300; family = 'v2' if arm == 'v2_xyz' else 'v3'
            identity = case+'-'+arm
            arrays = {k: np.stack([obs(index, t)[k] for t in range(n+1)]) for k in p.SHAPES}
            arrays.update(actions=np.full((n, 7), .05, np.float32), rewards=np.asarray([0.]*(n-1)+[float(success)]),
                          terminated=np.asarray([False]*(n-1)+[success]), truncated=np.asarray([False]*(n-1)+[not success]))
            if family == 'v2': arrays['eef_pos'] = np.zeros((n+1, 3), np.float32)
            path = source/(identity+'.npz'); np.savez_compressed(path, **arrays)
            jobs.append(dict(problem_id=identity, source_id=case, seed=20261107, controller=arm, family=family,
                initial_state_index=index, anchor_step=100, steps=n, success=success, terminal_reason='success' if success else 'time_limit',
                frames=p.pin(path), initial_row_sha256=hashlib.sha256(states[index].tobytes()).hexdigest(),
                observation_sha256=[describe(obs(index, t))['observation_sha256'] for t in range(n+1)]))
            warm.extend(dict(problem_id=identity, initial_state_index=i, observation=warm_pins[family, i],
                             observation_sha256=describe(obs(i, 0))['observation_sha256']) for i in range(index))
    cfg = dict(problems=jobs, warmup_references=warm, caps=p.caps(jobs), output=str(base), wall_seconds_max=1800,
               initial_indices=list(range(19)), initial_bank=dict(path='/invented/initial', sha256='a'*64, pickle_header_sha256='b'*64))
    access = dict(numeric_rows_decoded=list(range(19)), global_to_local_index={str(i): i for i in range(19)},
                  source_path='/invented/initial', container_sha256='a'*64, pickle_header_sha256='b'*64,
                  decoded_selected_bytes=19*92*8, skipped_rows_numerically_decoded=[], skipped_rows_buffered=False,
                  used_pickle_or_torch_deserialization=False,
                  selected_row_sha256={str(i): hashlib.sha256(states[i].tobytes()).hexdigest() for i in range(19)})
    p.write_new(base/'initial-access.json', access); np.savez_compressed(base/'initial-selected.npz', rows=states)
    prepared = dict(science=dict(path='invented', sha256='0'*64), binding=dict(path='invented_binding', sha256='1'*64),
                    access=p.pin(base/'initial-access.json'), ledger=dict(caps=dict(initial_row_loads=1, numeric_rows=19),
                    counts={k: dict(attempted=n, completed=n) for k, n in [('initial_row_loads', 1), ('numeric_rows', 19)]}))
    def initial(ledger):
        ledger.begin('numeric_rows', 19); ledger.begin('initial_cache_reads')
        ledger.finish('numeric_rows', 19); ledger.finish('initial_cache_reads')
        return states.copy(), deepcopy(access)
    return cfg, prepared, initial


def worker(data, job, mode='exact'):
    cfg, _, initial = data
    local = {**cfg, 'problems': [job], 'caps': p.worker_caps(job),
             'warmup_references': [w for w in cfg['warmup_references'] if w['problem_id'] == job['problem_id']]}
    return run.capture_dataset(local, adapter_factory=lambda task, states: Adapter(job, mode), initial_loader=initial,
        source_loader=p.load_source, geometry_factory=Geometry, destination=Path(cfg['output'])/job['problem_id'])


@pytest.fixture(scope='module')
def complete(tmp_path_factory):
    data = fixture(tmp_path_factory.mktemp('targeted-human')); cfg, prepared, _ = data
    workers = [worker(data, j, 'warning' if i == 1 else 'exact') for i, j in enumerate(cfg['problems'])]
    captured = driver.aggregate(cfg, prepared, workers, cfg['output'], 1.)
    result, receipt = audit.audit_capture(cfg, captured)
    return data, captured, result, receipt


def test_full9_exact_maximum_and_independent_audit(complete):
    (cfg, _, _), captured, result, receipt = complete
    assert cfg['caps']['native_steps'] == captured['ledger']['counts']['native_steps']['completed'] == 2931
    assert cfg['caps']['zero_steps'] == 705 and cfg['caps']['saved_actions'] == 2226
    assert cfg['caps']['geometry_samples'] == 1335 and receipt['geometry_samples'] == 1335
    assert len(result['rows']) == 9 and len(result['pairs']) == 6
    assert receipt['RGB_warning_camera_frames'] == 1 and receipt['new_native_steps'] == receipt['policy_forwards'] == 0
    assert all(r['anchor_native_byte_equal'] for r in result['pairs'])


def test_success_margin_and_short_last50(complete):
    _, _, result, _ = complete
    controls = [r for r in result['rows'] if r['controller'] == 'control']
    assert [r['windows']['last50']['On_frames'] for r in controls] == [1, 1, 1]
    for r, samples in zip(controls, [31, 39, 50]):
        assert r['cause'] == 'native_success'
        assert r['terminal_xy_margin_m'] == pytest.approx(.02)
        assert sum(r['native_component_patterns_last50'].values()) == samples


@pytest.mark.parametrize('mode', ['bad_state', 'amplitude2', 'pixels17', 'extra_zero', 'changed_command', 'duplicate_native', 'query_change'])
def test_immediate_stop_retains_failure(tmp_path, mode):
    data = fixture(tmp_path); job = data[0]['problems'][1]
    with pytest.raises(ValueError): worker(data, job, mode)
    failure = json.loads((Path(data[0]['output'])/job['problem_id']/'failure.json').read_text())
    assert failure['no_automatic_retry'] is True
    assert failure['ledger']['counts']['native_steps']['completed'] <= p.worker_caps(job)['native_steps']


def test_audit_rejects_changed_success_label(complete):
    (cfg, _, _), captured, _, _ = complete
    broken = deepcopy(cfg); broken['problems'][0]['success'] = False
    with pytest.raises(ValueError): audit.audit_capture(broken, captured)
