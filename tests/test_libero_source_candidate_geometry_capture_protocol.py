from copy import deepcopy
import builtins
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_source_candidate_geometry_capture_protocol as plan

CONFIG = plan.ROOT/'configs/libero-source-candidate-geometry-capture-diagnostic_v1.json'


def config():
    return json.loads(CONFIG.read_text())


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    c = config()
    training = {'status': 'complete_training', 'stage': 'G', 'models': []}
    for endpoint in c['model_endpoints']:
        training['models'].append({
            'task': endpoint['task'], 'seed': endpoint['seed'], 'arm': endpoint['arm'],
            'support_role': 'meta_support', 'query_role': None,
            'initial_state_role': 'candidate_nonregression', 'initial_state_indices': list(range(20)),
            'curve': {str(k): deepcopy(endpoint['checkpoint']) for k in plan.prior.CURVE},
        })
    paths = {k: tmp_path/(k+'.json') for k in plan.METADATA_PATHS}
    monkeypatch.setattr(plan, 'METADATA_PATHS', paths)
    sources = {k: tmp_path/(k+'.py') for k in plan.SOURCE_PATHS}
    for path in sources.values():
        path.write_text('# synthetic source, never imported\n')
    monkeypatch.setattr(plan, 'SOURCE_PATHS', sources)

    def save(name, value):
        paths[name].write_text(json.dumps(value))
        return {'path': str(paths[name]), 'sha256': plan.sha(paths[name])}

    train_pin = save('training', training)
    audit_pin = save('training_audit', {'status': 'passed_training_audit', 'training_sha256': train_pin['sha256']})
    result_pin = save('result', {'stage': 'G', 'training': train_pin, 'training_audit': audit_pin})
    decision = {'passed': False, 'next_stage': 'stop_keep_heldout_sealed'}
    complete_pin = save('complete_audit', {'status': 'passed_complete_audit', 'decision': decision,
                                         'result_sha256': result_pin['sha256']})
    save('completion', {'status': 'complete_audited_G', 'decision': decision,
                        'result': result_pin, 'audit': complete_pin})
    save('manifest', {'initial_banks': c['initial_banks']})
    save('family', {'tasks': [{'task_id': r['task'], 'task_name': r['task_name'],
                              'source_geometry': r['source_geometry'],
                              'source_initial_predicate': r['source_initial_predicate']}
                             for r in c['task_bindings']]})
    c['metadata_inputs'] = {k: {'path': str(p), 'sha256': plan.sha(p)} for k, p in paths.items()}
    c['source_references'] = {k: {'path': str(p), 'sha256': plan.sha(p)} for k, p in sources.items()}
    config_path = tmp_path/'config.json'
    report = tmp_path/'protocol.md'; report.write_text('Synthetic protocol\n')
    files = {'protocol_validator': Path(plan.__file__), 'protocol_report': report, 'tests': Path(__file__)}
    binding_path = tmp_path/'binding.json'

    def write_binding():
        config_path.write_text(json.dumps(c))
        binding = {'status': 'source_candidate_geometry_science_frozen',
                   'config_sha256': plan.sha(config_path), 'execution_binding_complete': False,
                   'capture_authorized': False,
                   'files': {k: {'path': str(p.resolve()), 'sha256': plan.sha(p)} for k, p in files.items()}}
        binding_path.write_text(json.dumps(binding))
        return binding

    write_binding()
    return c, config_path, binding_path, paths, training, write_binding


def test_metadata_preflight_never_opens_checkpoint_bank_or_imports_runtime(synthetic, monkeypatch):
    c, path, binding, _, _, _ = synthetic
    old_open, old_import = Path.open, builtins.__import__

    def checked_open(p, *args, **kwargs):
        assert p.suffix not in {'.pt', '.npz', '.h5', '.hdf5', '.pruned_init', '.pkl'}
        return old_open(p, *args, **kwargs)

    def checked_import(name, *args, **kwargs):
        assert name.split('.')[0] not in {'torch', 'numpy', 'h5py', 'libero', 'robosuite', 'mujoco', 'src'}
        return old_import(name, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', checked_open)
    monkeypatch.setattr(builtins, '__import__', checked_import)
    observed = plan.preflight(path, binding)
    assert observed == c and observed['design']['freeze_boundary']['capture_authorized_by_this_preflight'] is False


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'tasks', [0, 1, 3, 4, 7, 2]),
    ('scope', 'tasks', [5, 8, 9]),
    ('scope', 'seeds', [20261126]),
    ('scope', 'arms', ['B_fomaml', 'C_pooled_bc']),
    ('scope', 'global_initial_indices', [20, 21, 22, 23]),
    ('scope', 'initial_role', 'qualification'),
    ('fixed_policy', 'checkpoint_update', 1000),
    ('fixed_policy', 'loss', 'gripper_weighted'),
    ('capture', 'stabilization_steps', 0),
    ('capture', 'maximum_policy_actions_per_prefix', 300),
    ('capture', 'lifecycle', 'reuse_adapter_and_previous_scene_history'),
    ('capture', 'adapter_construction', 'default_adapter_loads_entire_initial_bank'),
    ('analysis', 'close_onset_time', 'state[k].sim_time'),
    ('observer', 'policy_receives_geometry', True),
    ('observer', 'site_check_contact_used', True),
    ('observer', 'native_simulation_writes_or_extra_forward_or_step', 1),
    ('budget', 'episodes', 121),
    ('budget', 'shadow_position_forwards_max', 18121),
    ('budget', 'technical_rerun_or_replacement_episodes', 1),
    ('freeze_boundary', 'execution_binding_complete', True),
])
def test_scope_and_budget_changes_reject_before_provenance_io(tmp_path, monkeypatch, section, key, value):
    c = config(); c['design'][section][key] = value
    path = tmp_path/'changed.json'; path.write_text(json.dumps(c))
    monkeypatch.setattr(plan, 'sha', lambda *a: pytest.fail('invalid design reached provenance I/O'))
    with pytest.raises(ValueError, match='design differs'):
        plan.preflight(path, tmp_path/'not_opened.json')


def test_payload_reference_rejected_before_any_checksum_read(tmp_path, monkeypatch):
    c = config(); c['metadata_inputs']['training']['path'] = '/unread/qualification.pruned_init'
    path = tmp_path/'bad.json'; path.write_text(json.dumps(c))
    monkeypatch.setattr(plan, 'sha', lambda *a: pytest.fail('payload path reached checksum read'))
    with pytest.raises(ValueError, match='unapproved input'):
        plan.preflight(path, tmp_path/'not_opened.json')


@pytest.mark.parametrize('kind', ['missing_model', 'duplicate_model', 'wrong_role', 'qualification_rows', 'missing500', 'wrong500_path'])
def test_endpoint_selection_cannot_substitute_missing_or_unauthorized_models(synthetic, kind):
    training = synthetic[4]
    if kind == 'missing_model':
        training['models'].pop()
    elif kind == 'duplicate_model':
        training['models'].append(deepcopy(training['models'][0]))
    elif kind == 'wrong_role':
        training['models'][0]['support_role'] = 'selection_query'
    elif kind == 'qualification_rows':
        training['models'][0]['initial_state_indices'] = list(range(20, 40))
    elif kind == 'missing500':
        del training['models'][0]['curve']['500']
    else:
        training['models'][0]['curve']['500']['path'] = '/unread/selected-u300.pt'
    with pytest.raises(ValueError):
        plan.model_endpoints(training)


@pytest.mark.parametrize('kind', ['metadata_changed', 'endpoint_hash_changed', 'execution_authorized', 'missing_code_pin'])
def test_bound_metadata_and_endpoint_or_execution_drift_fail_closed(synthetic, kind):
    c, path, binding_path, paths, _, write_binding = synthetic
    if kind == 'metadata_changed':
        paths['training'].write_text(paths['training'].read_text()+'\n')
    elif kind == 'endpoint_hash_changed':
        c['model_endpoints'][0]['checkpoint']['weight_hash'] = '0'*64
        write_binding()
    else:
        b = json.loads(binding_path.read_text())
        if kind == 'execution_authorized':
            b['capture_authorized'] = True
        else:
            del b['files']['tests']
        binding_path.write_text(json.dumps(b))
    with pytest.raises(ValueError):
        plan.preflight(path, binding_path)


def test_budget_includes_geometry_compute_and_initial_decoding_not_just_policy_actions():
    d = plan.design(); b = d['budget']; s = d['scope']; capture = d['capture']
    models = len(s['tasks'])*len(s['seeds'])*len(s['arms'])
    episodes = models*len(s['global_initial_indices'])
    assert b['models'] == models == 30 and b['episodes'] == episodes == 120
    assert b['paired_scenes'] == episodes//2
    assert b['policy_forwards_and_actions_max'] == episodes*capture['maximum_policy_actions_per_prefix']
    assert b['zero_actions_max'] == episodes*capture['stabilization_steps']
    assert b['native_control_steps_max'] == b['zero_actions_max']+b['policy_forwards_and_actions_max']
    assert b['native_physics_steps_max'] == b['native_control_steps_max']*capture['physics_substeps_per_control']
    assert b['geometry_rows_max'] == b['shadow_position_forwards_max'] == b['public_rows_max'] == episodes*151
    assert b['saved_bowl_contact_records_max'] == b['geometry_rows_max']*d['observer']['contacts_per_snapshot_cap']
    assert b['numeric_initial_rows_max'] == len(s['tasks'])*len(s['global_initial_indices'])


def test_import_is_metadata_only_in_clean_process():
    program = '''
import builtins
original = builtins.__import__
def checked(name, *args, **kwargs):
    assert name.split('.')[0] not in {'torch', 'numpy', 'h5py', 'libero', 'robosuite', 'mujoco', 'src'}
    return original(name, *args, **kwargs)
builtins.__import__ = checked
from experiments import libero_source_candidate_geometry_capture_protocol
'''
    subprocess.run([sys.executable, '-c', program], cwd=plan.ROOT, check=True)


def test_cli_never_overwrites_existing_artifact_or_reads_inputs(tmp_path, monkeypatch):
    output = tmp_path/'retained.json'; output.write_text('original')
    monkeypatch.setattr(sys, 'argv', ['preflight', '--config', 'missing.json', '--binding', 'missing.json', '--output', str(output)])
    monkeypatch.setattr(plan, 'preflight', lambda *a: pytest.fail('existing output caused input read'))
    with pytest.raises(ValueError, match='must be new'):
        plan.main()
    assert output.read_text() == 'original'
