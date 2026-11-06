from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import h5py
import pytest

from experiments import libero_formal_execution as execution
from experiments.prepare_libero_formal_execution import shape_metadata


@pytest.fixture
def bound(tmp_path, monkeypatch):
    plan = execution.stage_plan('G')
    metadata = tmp_path/'metadata.json'
    metadata.write_text('{}')
    pin = {'path': str(metadata), 'sha256': execution.sha(metadata)}
    transfer = tmp_path/'transfer.json'
    rows = [{'task_id': t, 'path': f'task{t}.pruned_init', 'size_bytes': 10, 'sha256': 'a'*64}
            for t in plan['initial_tasks']]
    transfer.write_text(json.dumps({'initial_state_files': rows}))
    lengths = {str(t): {str(i): 101 for i in range(50)} for t in plan['initial_tasks']}
    banks = {str(r['task_id']): {'relative_path': r['path'], 'size_bytes': r['size_bytes'],
                               'sha256': r['sha256'], 'pickle_header_sha256': 'b'*64} for r in rows}
    manifest = {'stage': 'G', 'plan': plan, 'inputs': {'scientific_config': pin, 'scientific_binding': pin,
                'transfer': {'path': str(transfer), 'sha256': execution.sha(transfer)}},
                'initial_banks': banks, 'action_lengths_by_task': lengths,
                'expected_offline_batches': execution.expected_evaluation_batches(plan, lengths)}
    names = ('runner', 'execution_guards', 'initial_reader', 'training_helpers', 'independent_auditor',
             'policy', 'fomaml', 'spt', 'causal_dataset', 'base_dataset', 'deterministic_pool', 'adapter', 'adapter_builder')
    binding = {'execution_binding_complete': True, 'files': {name: pin for name in names}}
    manifest_path, binding_path = tmp_path/'manifest.json', tmp_path/'binding.json'
    def write():
        manifest_path.write_text(json.dumps(manifest))
        binding['manifest_sha256'] = execution.sha(manifest_path)
        binding_path.write_text(json.dumps(binding))
    write()
    monkeypatch.setattr(execution.protocol, 'preflight', lambda *args: None)
    return manifest, binding, write, manifest_path, binding_path


def test_metadata_preflight_does_not_open_real_payloads_or_initial_banks(bound, monkeypatch):
    manifest, binding, write, path, binding_path = bound
    original = Path.open
    def guarded(self, *args, **kwargs):
        assert not str(self).startswith((execution.DATA_ROOT, execution.INITIAL_ROOT))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded)
    result, pins = execution.execution_preflight(path, binding_path, requested_stage='G')
    assert result == manifest and pins == binding


def test_G_binding_cannot_request_H(bound):
    *_, path, binding_path = bound
    with pytest.raises(ValueError, match='stage/plan'):
        execution.execution_preflight(path, binding_path, requested_stage='H')


@pytest.mark.parametrize('change', ['plan', 'bank_scope', 'bank_rows', 'length_scope', 'length_type',
                                   'batches', 'missing_code_pin', 'incomplete_binding', 'file_hash'])
def test_preflight_rejects_modified_scope_budget_and_dependencies(bound, change):
    manifest, binding, write, path, binding_path = bound
    if change == 'plan': manifest['plan']['models'][0]['initial_state_indices'] = list(range(20, 40))
    elif change == 'bank_scope': manifest['initial_banks']['5'] = manifest['initial_banks']['0']
    elif change == 'bank_rows': manifest['initial_banks']['2']['relative_path'] = 'task5.pruned_init'
    elif change == 'length_scope': manifest['action_lengths_by_task']['5'] = deepcopy(manifest['action_lengths_by_task']['0'])
    elif change == 'length_type': manifest['action_lengths_by_task']['0']['0'] = True
    elif change == 'batches': manifest['expected_offline_batches'] += 1
    elif change == 'missing_code_pin': del binding['files']['initial_reader']
    elif change == 'incomplete_binding': binding['execution_binding_complete'] = False
    elif change == 'file_hash': binding['files']['runner'] = {**binding['files']['runner'], 'sha256': '0'*64}
    write()
    with pytest.raises(ValueError): execution.execution_preflight(path, binding_path, requested_stage='G')


def test_metadata_import_path_needs_no_torch_numpy_hdf5_or_environment_modules():
    program = """
import builtins
original = builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in ('torch','numpy','h5py','src'):
        raise AssertionError('payload dependency imported: '+name)
    return original(name,*args,**kwargs)
builtins.__import__ = guarded
import experiments.run_libero_formal_transfer
import experiments.prepare_libero_formal_execution
"""
    subprocess.run([sys.executable, '-c', program], check=True, cwd=execution.ROOT)


def test_shape_collector_reads_no_dataset_values(tmp_path, monkeypatch):
    path = tmp_path/'synthetic.hdf5'
    with h5py.File(path, 'w') as handle:
        data = handle.create_group('data')
        for i in range(50):
            demo = data.create_group(f'demo_{i}')
            demo.create_dataset('actions', shape=(11+i, 7))
            obs = demo.create_group('obs')
            for key, tail in [('agentview_rgb', (128, 128, 3)), ('eye_in_hand_rgb', (128, 128, 3)),
                              ('joint_states', (7,)), ('gripper_states', (2,))]:
                obs.create_dataset(key, shape=(11+i, *tail))
    def forbidden(*args, **kwargs): pytest.fail('numeric payload value read during schema collection')
    monkeypatch.setattr(h5py.Dataset, '__getitem__', forbidden)
    assert shape_metadata(path) == {str(i): 11+i for i in range(50)}


def test_existing_output_is_rejected_before_preflight_or_model_import(tmp_path, monkeypatch):
    from experiments import run_libero_formal_transfer as runner
    output = tmp_path/'existing.json'; output.write_text('retained')
    monkeypatch.setattr(runner, 'execution_preflight', lambda *a, **kw: pytest.fail('preflight reached'))
    with pytest.raises(ValueError, match='must be new'):
        runner.run(None, None, stage='G', output=output, archive=tmp_path/'new')
    assert output.read_text() == 'retained' and not (tmp_path/'new').exists()
