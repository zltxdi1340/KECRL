"""Synthetic models only; no real checkpoint, LIBERO scene or initial bank."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments import run_libero_source_candidate_geometry_capture as capture
from experiments.libero_formal_execution import Ledger
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest
from src.environments.libero_geometry_observer import LiberoGeometryObserver, fingerprint, reset_geometry_digest
from src.environments.libero_geometry_capture_adapter import BoundedControlEnvironment, reset_once
from src.environments.libero_adapter import LiberoEnvironmentAdapter
from experiments import audit_libero_source_candidate_geometry_capture as independent


def config():
    return json.loads(capture.DEFAULT_CONFIG.read_text())


def ledger():
    value = Ledger(capture.resource_caps(config()))
    value.observer_active = False
    return value


@pytest.fixture
def toy():
    import mujoco

    def make(*, bowl_geoms=1, finger_x=.025, margin=0):
        chain = ''.join(f'<body name="link{i}"><joint name="j{i}" type="hinge" axis="0 0 1"/>'
                        '<geom size=".001" mass=".1" contype="0" conaffinity="0"/>' for i in range(7))
        xml = '<mujoco><option timestep=".002" gravity="0 0 0"/><worldbody>'+chain
        xml += '<site name="grip_site" pos="0 0 .15" size=".001"/>'+'</body>'*7
        for group, x in [('left', -finger_x), ('right', finger_x)]:
            xml += f'<body name="{group}" pos="{x} 0 .05"><joint name="{group}_joint" type="slide" axis="1 0 0"/>'
            xml += f'<geom name="{group}_geom" type="box" size=".01 .03 .03" mass=".1"/></body>'
        xml += '<body name="bowl_body" pos="0 0 .05"><freejoint/>'
        xml += ''.join(f'<geom name="bowl_geom{i}" type="sphere" size=".02" mass=".01" margin="{margin}"/>' for i in range(bowl_geoms))
        xml += '</body><body name="plate_body" pos=".3 0 .05"><geom name="plate_geom" size=".04"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml); data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)  # Toy model only, never a LIBERO runtime.
        robot = SimpleNamespace(_ref_joint_pos_indexes=list(range(7)), _ref_gripper_joint_pos_indexes=[7, 8],
            gripper=SimpleNamespace(important_sites={'grip_site': 'grip_site'},
                                    important_geoms={'left_finger': ['left_geom'], 'right_finger': ['right_geom']}))
        objects = {'akita_black_bowl_1': SimpleNamespace(root_body='bowl_body', contact_geoms=[f'bowl_geom{i}' for i in range(bowl_geoms)]),
                   'plate_1': SimpleNamespace(root_body='plate_body', contact_geoms=['plate_geom'])}
        domain = SimpleNamespace(robots=[robot], objects_dict=objects, control_freq=20, model_timestep=.002,
            obj_body_id={k: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, v.root_body) for k, v in objects.items()},
            sim=SimpleNamespace(model=SimpleNamespace(_model=model, get_xml=lambda: xml), data=SimpleNamespace(_data=data)))

        def observation():
            return {'agentview_rgb': np.zeros((128, 128, 3), np.uint8),
                    'eye_in_hand_rgb': np.zeros((128, 128, 3), np.uint8),
                    'joint_states': data.qpos[:7].astype(np.float32).copy(),
                    'gripper_states': data.qpos[7:9].astype(np.float32).copy()}

        return domain, data, observation
    return make


def test_shadow_uses_current_qpos_keeps_all_native_buffers_and_public_inputs(toy):
    domain, native, obs = toy()
    progress = ledger(); observer = LiberoGeometryObserver(domain, observation_digest=observation_digest, ledger=progress)
    start = observer.sample(obs(), 0)
    assert start['bilateral_touch_proxy'] and start['touch_counts']['left'] > 0 and start['touch_counts']['right'] > 0
    old_cached = native.xpos[domain.obj_body_id['akita_black_bowl_1']].copy()
    native.qpos[9] += .15; native.time += .05
    before = fingerprint(native); observation = obs(); digest = observation_digest(observation)
    after = observer.sample(observation, 1)
    assert after['bowl_position_world'][0] == pytest.approx(.15)
    assert np.array_equal(native.xpos[domain.obj_body_id['akita_black_bowl_1']], old_cached)
    assert after['native_before_sha256'] == after['native_after_sha256'] == before
    assert observation_digest(observation) == digest
    assert after['copied_state_sha256'] == after['shadow_copied_state_sha256']
    assert not after['bilateral_touch_proxy']
    assert progress.counts['shadow_allocation'] == {'attempted': 1, 'completed': 1}
    assert progress.counts['shadow_forward'] == {'attempted': 2, 'completed': 2}
    assert reset_geometry_digest(start) != reset_geometry_digest(after)


def test_fingerprint_does_not_read_unbacked_inactive_solver_memory(toy):
    _, native, _ = toy()
    # Non-dual/no-island models expose inactive C NULL pointers as owned,
    # uninitialized Python arrays. They must not cause false passivity failures.
    value = fingerprint(native)
    assert all(fingerprint(native) == value for _ in range(25))
    native.qpos[0] += .1
    assert fingerprint(native) != value


def test_positive_margin_contacts_are_not_touch_or_force_labels(toy):
    domain, _, obs = toy(finger_x=.04, margin=.02)
    row = LiberoGeometryObserver(domain, observation_digest=observation_digest, ledger=ledger()).sample(obs(), 0)
    assert row['contacts'] and all(x['distance_m'] > 0 for x in row['contacts'])
    assert row['positive_margin_counts']['left'] and row['positive_margin_counts']['right']
    assert not row['bilateral_touch_proxy'] and not any(row['touch_counts'].values())
    assert not any('force' in key for key in row)


@pytest.mark.parametrize('kind', ['wrong_clock', 'native_mutation'])
def test_clock_and_passivity_fail_closed(toy, monkeypatch, kind):
    domain, native, obs = toy(); observer = LiberoGeometryObserver(domain, observation_digest=observation_digest, ledger=ledger())
    observer.sample(obs(), 0)
    native.time += .04 if kind == 'wrong_clock' else .05
    observation = obs()
    if kind == 'native_mutation':
        original = observer.mujoco.mj_fwdPosition
        def unsafe(model, shadow):
            original(model, shadow); native.qpos[0] += .1
        monkeypatch.setattr(observer.mujoco, 'mj_fwdPosition', unsafe)
    with pytest.raises(ValueError):
        observer.sample(observation, 1)


def test_original_sampled_observation_is_kept_and_cache_gap_is_recorded(toy):
    domain, native, obs = toy();observation = obs();observation['joint_states'][0] += .1
    digest = observation_digest(observation)
    row = LiberoGeometryObserver(domain, observation_digest=observation_digest, ledger=ledger()).sample(observation, 0)
    assert row['native_vs_policy_joint_gap_l2'] == pytest.approx(.1)
    assert row['native_before_sha256'] == row['native_after_sha256']
    assert observation_digest(observation) == digest


@pytest.mark.parametrize('kind', ['empty', 'overlap', 'missing', 'bad_body'])
def test_model_mapping_must_be_resolved_and_disjoint(toy, kind):
    domain, _, _ = toy()
    if kind == 'empty':domain.robots[0].gripper.important_geoms['left_finger'] = []
    elif kind == 'overlap':domain.robots[0].gripper.important_geoms['left_finger'] = ['bowl_geom0']
    elif kind == 'missing':domain.robots[0].gripper.important_sites['grip_site'] = 'missing_site'
    else:domain.obj_body_id['akita_black_bowl_1'] += 1
    with pytest.raises(ValueError):
        LiberoGeometryObserver(domain, observation_digest=observation_digest, ledger=ledger())


def test_contact_cap_stops_instead_of_truncating(toy):
    domain, _, obs = toy(bowl_geoms=129)
    observer = LiberoGeometryObserver(domain, observation_digest=observation_digest, ledger=ledger())
    with pytest.raises(ValueError, match='cap'):
        observer.sample(obs(), 0)


def test_bounded_reset_does_not_enter_vendor_retry_loop_or_retry_failure():
    calls = []
    def native_reset():calls.append('native');raise RuntimeError('synthetic reset failed')
    def forbidden():pytest.fail('vendor reset retry loop entered')
    control = SimpleNamespace(reset=forbidden, env=SimpleNamespace(reset=native_reset))
    wrapper = BoundedControlEnvironment(control, ledger())
    with pytest.raises(RuntimeError, match='synthetic'):wrapper.reset()
    with pytest.raises(RuntimeError, match='retry forbidden'):wrapper.reset()
    assert calls == ['native']


def test_zero_stabilization_is_exact_and_bounded():
    progress = ledger(); control = SimpleNamespace(step=lambda action: 1, env=SimpleNamespace(reset=lambda: None))
    wrapper = BoundedControlEnvironment(control, progress); wrapper.inside_reset = True
    for _ in range(5):assert wrapper.step(np.zeros(7, np.float32)) == 1
    with pytest.raises(ValueError):wrapper.step(np.zeros(7, np.float32))
    assert progress.counts['zero_action'] == {'attempted': 5, 'completed': 5}
    wrapper = BoundedControlEnvironment(control, ledger());wrapper.inside_reset = True
    with pytest.raises(ValueError):wrapper.step(np.ones(7, np.float32))


@pytest.fixture(scope='module')
def complete_capture(tmp_path_factory):
    make = toy.__wrapped__()
    cfg = config(); progress = ledger(); archive = tmp_path_factory.mktemp('synthetic-geometry-complete')
    for key, n in [('model_load', 30), ('policy_object', 30), ('initial_container', 5), ('numeric_row', 20)]:
        progress.begin(key, n);progress.finish(key, n)

    def factory(task, row):
        domain, native, observation = make()
        native.qpos[9] = .15
        steps = [0]; stop = 1000 if task == 0 and int(row[0]) == 0 else 7
        domain.reset = lambda: None

        def native_observation():
            values = observation()
            return {'agentview_image': values['agentview_rgb'], 'robot0_eye_in_hand_image': values['eye_in_hand_rgb'],
                    'robot0_joint_pos': values['joint_states'], 'robot0_gripper_qpos': values['gripper_states']}

        def step(action):
            steps[0] += 1; native.time += .05
            progress.begin('physics_step', 25);progress.finish('physics_step', 25)
            if steps[0] > 5:
                native.qpos[0] += .001
                native.qpos[9] = .1 if steps[0] == 6 else 0
            return native_observation(), 0., steps[0] >= stop, {}

        control = SimpleNamespace(env=domain, step=step, seed=lambda value: None,
            set_init_state=lambda value: native_observation(), check_success=lambda: steps[0] >= stop, close=lambda: None)
        return LiberoEnvironmentAdapter(environment=BoundedControlEnvironment(control, progress), initial_states=row[None],
            task_name='synthetic', language_instruction='synthetic', task_id=task, seed=20261022)

    initial = {t: np.zeros((4, 92)) for t in capture.science.TASKS}
    for rows in initial.values():rows[:, 0] = np.arange(4)
    models = {(r['task'], r['seed'], r['arm']): object() for r in cfg['model_endpoints']}

    def predict(model, observation):
        value = 1 if observation['joint_states'][0] < .0015 else -1
        return np.asarray([0, 0, 0, 0, 0, 0, value], np.float32)

    def weights(model, row):
        return {'task': row['task'], 'seed': row['seed'], 'arm': row['arm'],
                'weight_hash': row['checkpoint']['weight_hash'], 'no_gradients_or_trainable_parameters': True}

    rollout = capture.execute(cfg, initial_rows=initial, loaded_models=models, model_to_device=lambda x: x,
        adapter_factory=factory, resetter=reset_once,
        observer_factory=lambda adapter: LiberoGeometryObserver(adapter.environment.control.env,
            observation_digest=observation_digest, ledger=progress), predictor=predict, weight_checker=weights,
        observation_digest=observation_digest, reset_geometry_digest=reset_geometry_digest,
        ledger=progress, archive=archive, event=lambda *a, **kw: None)
    access = {t: {'numeric_rows_decoded': [0, 1, 2, 3], 'skipped_rows_numerically_decoded': [],
                 'skipped_rows_buffered': False, 'used_pickle_or_torch_deserialization': False,
                 'container_sha256': p['sha256'], 'pickle_header_sha256': p['pickle_header_sha256'],
                 'global_to_local_index': {str(i): i for i in range(4)}} for t, p in cfg['initial_banks'].items()}
    before = [{**r, 'verified_file_sha256': r['checkpoint']['sha256'],
               'verified_weight_hash': r['checkpoint']['weight_hash']} for r in cfg['model_endpoints']]
    result = {'status': 'complete_capture_pending_independent_audit', 'new_training_updates': 0,
              'G_decision_unchanged': 'failed_keep_H_sealed', 'rollout': rollout,
              'ledger': progress.snapshot(), 'initial_state_access': access, 'endpoints_verified_before_environment': before}
    return cfg, result, archive


def test_complete120_synthetic_capture_and_independent_geometry_analysis(complete_capture):
    cfg, result, archive = complete_capture
    indexed, counts = independent.validate_capture(result, cfg, archive)
    assert counts['prefixes'] == 120 and counts['exact_reset_comparisons'] == 100
    assert counts['policy_actions'] == 6*150+114*2
    tables = independent.analyze_pairs(indexed)
    assert len(tables['episodes']) == 120 and len(tables['pairs']) == 60
    assert len(tables['task_seed_arms']) == 30
    assert sum(r['prefix_outcome'] == 'neither_success_within_prefix' for r in tables['pairs']) == 3
    assert all(r['close_to_touch_seconds'] == pytest.approx(.1) for r in tables['episodes'])
    assert all(r['first_bilateral_touch_state'] == 2 and r['first_close'] == 1 for r in tables['episodes'])
    assert any(not r['A_available'] and r['reset_side_truncated'] for r in tables['event_windows'])
    assert any(not r['A_available'] and r['A_terminal_truncated'] for r in tables['event_windows'])


@pytest.mark.parametrize('kind', ['private_field', 'contact_force_field', 'shifted_clock', 'wrong_contact_label',
                                  'bad_rotation', 'passivity_changed', 'bad_native_count', 'qualification_role', 'missing_pair', 'cap_changed', 'order_changed'])
def test_independent_audit_rejects_semantic_drift_after_rehash(complete_capture, tmp_path, kind):
    import shutil
    cfg, original, source = complete_capture
    result = deepcopy(original);archive = tmp_path/'copy';shutil.copytree(source, archive)
    for d in result['rollout']['episode_files']:
        for name in ['public', 'geometry']:
            d[name]['path'] = str(archive/Path(d[name]['path']).name)
    d = result['rollout']['episode_files'][0];p = Path(d['geometry']['path']);g = json.loads(p.read_text())
    if kind == 'private_field':g['rows'][0]['qpos'] = [0.]*16
    elif kind == 'contact_force_field':g['rows'][2]['contacts'][0]['force'] = [0.]*6
    elif kind == 'shifted_clock':g['rows'][1]['sim_time'] += .001
    elif kind == 'wrong_contact_label':g['rows'][2]['bilateral_touch_proxy'] = False
    elif kind == 'bad_rotation':g['rows'][0]['eef_rotation_world'][0][0] = 2.
    elif kind == 'passivity_changed':g['rows'][0]['native_after_sha256'] = '0'*64
    elif kind == 'bad_native_count':result['ledger']['counts']['physics_step']['completed'] -= 1
    elif kind == 'qualification_role':d['initial_state_role'] = 'qualification'
    elif kind == 'cap_changed':result['ledger']['caps']['zero_action'] += 1
    elif kind == 'order_changed':result['rollout']['episode_files'].reverse()
    else:result['rollout']['episode_files'].pop()
    p.write_text(json.dumps(g));d['geometry']['sha256'] = capture.sha(p)
    with pytest.raises(ValueError):independent.validate_capture(result, cfg, archive)


def test_ledger_failure_does_not_get_replaced_or_exceed_action_cap():
    progress = Ledger({'policy_action': 1});progress.call('policy_action', lambda: None)
    with pytest.raises(RuntimeError):progress.call('policy_action', lambda: pytest.fail('budget exceeded'))
    assert progress.counts['policy_action'] == {'attempted': 1, 'completed': 1}


def test_collector_preflight_is_metadata_only_and_requires_full_binding(tmp_path, monkeypatch):
    import builtins
    cfg = config(); science_binding = capture.DEFAULT_SCIENCE
    monkeypatch.setattr(capture.science, 'preflight', lambda *a: cfg)
    files = {'runner': Path(capture.__file__).resolve()}
    for name in ['observer', 'bounded_adapter', 'independent_auditor', 'tests', 'execution_contract']:
        p = tmp_path/(name+'.py');p.write_text('# synthetic metadata\n');files[name] = p
    binding = {'status': 'source_geometry_execution_bound_before_payload', 'execution_binding_complete': True,
               'formal_result': False, 'config_sha256': capture.sha(capture.DEFAULT_CONFIG),
               'science_binding_sha256': capture.sha(science_binding), 'caps': capture.resource_caps(cfg),
               'output': str(tmp_path/'new.json'), 'artifacts': str(tmp_path/'new'),
               'device': {'physical_index': 1, 'uuid': 'GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff',
                          'cuda_visible_devices': '1', 'egl_device_id': '1', 'inference_device': 'cuda:0'},
               'files': {k: {'path': str(p), 'sha256': capture.sha(p)} for k, p in files.items()}}
    path = tmp_path/'binding.json';path.write_text(json.dumps(binding))
    old_import, old_open = builtins.__import__, Path.open
    def no_runtime(name, *a, **kw):
        assert name.split('.')[0] not in {'torch', 'numpy', 'h5py', 'libero', 'robosuite', 'mujoco', 'src'}
        return old_import(name, *a, **kw)
    def no_payload(p, *a, **kw):
        assert p.suffix not in {'.pt', '.pruned_init', '.hdf5', '.npz'}
        return old_open(p, *a, **kw)
    monkeypatch.setattr(builtins, '__import__', no_runtime);monkeypatch.setattr(Path, 'open', no_payload)
    assert capture.preflight(capture.DEFAULT_CONFIG, science_binding, path)[0] == cfg
    binding['caps']['policy_forward'] += 1;path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='caps'):capture.preflight(capture.DEFAULT_CONFIG, science_binding, path)
    binding['caps'] = capture.resource_caps(cfg);del binding['files']['observer'];path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='pins'):capture.preflight(capture.DEFAULT_CONFIG, science_binding, path)


def test_native_physics_guard_blocks_observer_step_forward_and_restores_methods(monkeypatch):
    import sys
    from src.environments.libero_geometry_capture_adapter import counted_native_physics
    calls = []
    class Sim:
        def step(self):calls.append('step')
        def forward(self):calls.append('forward')
    old_step, old_forward = Sim.step, Sim.forward
    monkeypatch.setitem(sys.modules, 'robosuite.utils.binding_utils', SimpleNamespace(MjSim=Sim))
    progress = ledger()
    with counted_native_physics(progress):
        Sim().step();progress.observer_active = True
        with pytest.raises(RuntimeError):Sim().step()
        with pytest.raises(RuntimeError):Sim().forward()
    assert calls == ['step'] and Sim.step is old_step and Sim.forward is old_forward


def test_existing_capture_output_stops_before_checkpoint_or_gpu(tmp_path, monkeypatch):
    path = tmp_path/'existing.json';path.write_text('preserved')
    monkeypatch.setattr(capture, 'preflight', lambda *a: (config(), {'output': str(path), 'artifacts': str(tmp_path/'unused')}))
    with pytest.raises(ValueError, match='no retry'):capture.run(None, None, None)
    assert path.read_text() == 'preserved'


def test_auditor_is_independent_of_collector_observer_and_numeric_runtime():
    import subprocess, sys
    program = '''
import builtins,sys
original = builtins.__import__
def checked(name,*a,**kw):
    assert name.split('.')[0] not in {'torch','numpy','h5py','mujoco','libero','robosuite','src'}
    return original(name,*a,**kw)
builtins.__import__ = checked
from experiments import audit_libero_source_candidate_geometry_capture
assert 'experiments.run_libero_source_candidate_geometry_capture' not in sys.modules
assert 'src.environments.libero_geometry_observer' not in sys.modules
'''
    subprocess.run([sys.executable, '-c', program], cwd=capture.ROOT, check=True)


def test_all_no_close_retains_pairs_and_exports_empty_event_schema(complete_capture, tmp_path):
    cfg, result, archive = complete_capture
    indexed, _ = independent.validate_capture(result, cfg, archive)
    indexed = deepcopy(indexed)
    for public, _ in indexed.values():
        for row in public['rows'][1:]:row['action'][6] = -1.
    tables = independent.analyze_pairs(indexed)
    assert len(tables['pairs']) == 60 and tables['event_windows'] == []
    assert all(row['first_close'] is None and row['close_to_touch_seconds'] is None for row in tables['episodes'])
    output = tmp_path/'empty.csv';independent.write_table(output, tables['event_windows'])
    assert len(output.read_text().splitlines()) == 1 and 'A_pre_bilateral_touch_proxy' in output.read_text()
