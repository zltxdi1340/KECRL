"""Synthetic metadata tests; no real arrays, model weights or environment."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_component3_history_transition_protocol as d


def example():
    lengths = [155, 125, 124, 191, 136, 138, 133, 130, 139, 145, 140, 164, 124, 138, 120,
               137, 141, 152, 137, 128, 197, 124, 128, 126, 144, 129, 168, 132, 140, 133]
    endpoints, models = [], []
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            checkpoint = {'path': f'/never/open/{seed}-{arm}.pt', 'sha256': 'a' * 64, 'weight_hash': 'b' * 64}
            endpoint = {'task': 7, 'seed': seed, 'arm': arm, 'checkpoint_update': 500,
                        'support_role': 'meta_support', 'checkpoint': checkpoint}
            metrics = {phase: {'n': n, 'negative_n': nn, 'positive_n': pn, 'zero_prediction_n': 0}
                       for phase, (n, nn, pn) in {'all_common': (3598, 2273, 1325),
                                                  'tail20': (477, 402, 75),
                                                  'complement': (3121, 1871, 1250)}.items()}
            models.append({'task': 7, 'seed': seed, 'arm': arm, 'checkpoint': copy.deepcopy(checkpoint),
                           'archive': {'path': str(d.PARENT_DIRECTORY / f'task7-seed{seed}-{arm}-probe.npz'),
                                       'sha256': 'c' * 64},
                           'metrics': {mode: {'strata': copy.deepcopy(metrics)}
                                       for mode in ('current_balanced', 'history5_balanced')}})
            endpoints.append(endpoint)
    parents = {
        'science': {'formal_result': False, 'endpoints': endpoints, 'action_lengths': {'7': lengths}},
        'binding': {'formal_result': False, 'status': 'bound_before_payload', 'science_config': d.PARENTS['science']},
    }
    common = {'formal_result': False, 'science_config': d.PARENTS['science'], 'binding': d.PARENTS['binding']}
    parents['capture'] = {**common, 'status': 'complete_component3_feature_capture',
                          'shared': [{'task': 7, 'path': str(d.PARENT_DIRECTORY / 'task7-shared.npz'),
                                      'sha256': 'd' * 64}]}
    parents['result'] = {**common, 'status': 'complete_component3_observability_pending_independent_audit',
                         'capture': d.PARENTS['capture'], 'primary_results': models}
    parents['audit'] = {**common, 'status': 'passed_independent_component3_observability_audit',
                        'capture': d.PARENTS['capture'], 'result': d.PARENTS['result']}
    parents['completion'] = {**common, 'status': 'complete_independently_audited_component3_observability_diagnostic',
                             'result': d.PARENTS['result'], 'audit': d.PARENTS['audit']}
    config = {'kind': 'task7_component3_cached_history_transitions_readonly', 'version': 1,
              'formal_result': False, 'status': 'scientific_protocol_frozen_execution_unbound',
              'parents': copy.deepcopy(d.PARENTS), 'inputs': d.source_selection(parents),
              'scope': copy.deepcopy(d.SCOPE), 'analysis': copy.deepcopy(d.ANALYSIS),
              'allowed_members': copy.deepcopy(d.MEMBERS), 'execution': copy.deepcopy(d.EXECUTION),
              'budget': d.derive_budget(), 'output': d.OUTPUT, 'artifacts': d.ARTIFACTS}
    return config, copy.deepcopy(parents)


def test_complete_metadata_keeps_both_signs_all_models_and_four_transitions():
    config, parents = example()
    assert d.validate_design(config, parents)
    assert len(config['inputs']['models']) == 6
    assert config['scope']['demonstration_indices'] == list(range(30))
    assert config['analysis']['directions'] == ['negative', 'positive']
    assert len(config['analysis']['transitions']) == 4


def test_exact_budget_distinguishes_unique_labels_reused_predictions_and_numeric_loads():
    b = d.derive_budget()
    assert b['shared_tail20_positive_rows'] + b['shared_tail20_negative_rows'] == 477
    assert b['shared_material_negative_rows'] + b['shared_material_positive_rows'] == 3598
    assert b['shared_material_common_rows'] + b['shared_neutral_common_rows_excluded'] == 4068
    assert b['per_stage']['model_frame_records'] == 21588
    assert b['per_stage']['saved_neighbor_records_checked'] == 431760
    assert b['expected_table_rows'] == {'frame-transitions': 21588, 'model-transitions': 144,
                                      'demo-transitions': 4320, 'seed-agreement-frames': 7196,
                                      'seed-agreement-counts': 192, 'tail20-neighbors': 57240}
    assert b['analysis_and_audit_array_file_loads'] == 2 * b['per_stage']['total_array_file_loads'] == 14


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'task_ids', [7, 2]), ('scope', 'training_seeds', [20261106]),
    ('scope', 'checkpoint_updates', 1000), ('scope', 'closed_loop_default_stabilization_steps', 0),
    ('scope', 'component_index', 6), ('scope', 'material_absolute_margin_inclusive', .01),
    ('scope', 'new_training_updates', 1), ('scope', 'new_training_updates', False),
    ('scope', 'source_meta_query_accessed', True), ('scope', 'sealed_roles_accessed', True),
    ('scope', 'new_feature_or_full_library_distance_computations', 1),
    ('analysis', 'directions', ['positive']), ('analysis', 'transitions', ['corrected']),
    ('analysis', 'primary_stratum', 'best_seed_only'),
    ('analysis', 'material_run', 'skip_neutral_frames_to_manufacture_switch'),
    ('allowed_members', 'shared', d.MEMBERS['shared'] + ['public_proprio']),
    ('execution', 'technical_retries', 1), ('execution', 'output_overwrite', True),
    ('execution', 'device', 'cuda:0'), ('budget', 'analysis_and_audit_array_file_loads', 16),
])
def test_training_role_sampling_member_or_budget_drift_fails_closed(section, key, value):
    config, parents = example(); config[section][key] = value
    with pytest.raises(ValueError, match='frozen scientific'):
        d.validate_design(config, parents)


@pytest.mark.parametrize('change', ['missing_model', 'duplicate_model', 'wrong_checkpoint', 'wrong_role',
                                   'wrong_archive', 'wrong_counts', 'duplicate_shared', 'wrong_lengths'])
def test_source_identity_and_exact_counts_cannot_be_silently_replaced(change):
    config, parents = example()
    if change == 'missing_model': parents['result']['primary_results'].pop()
    elif change == 'duplicate_model': parents['result']['primary_results'][-1] = parents['result']['primary_results'][0]
    elif change == 'wrong_checkpoint': parents['result']['primary_results'][0]['checkpoint']['weight_hash'] = 'e' * 64
    elif change == 'wrong_role': parents['science']['endpoints'][0]['support_role'] = 'meta_query'
    elif change == 'wrong_archive': parents['result']['primary_results'][0]['archive']['path'] = '/never/open/public9.npz'
    elif change == 'wrong_counts': parents['result']['primary_results'][0]['metrics']['current_balanced']['strata']['tail20']['positive_n'] = 74
    elif change == 'duplicate_shared': parents['capture']['shared'].append(parents['capture']['shared'][0])
    elif change == 'wrong_lengths': parents['science']['action_lengths']['7'][0] += 1
    with pytest.raises(ValueError):
        d.validate_design(config, parents)


@pytest.mark.parametrize('key,field,value', [
    ('audit', 'status', 'not_passed'), ('completion', 'audit', {'path': '/never/open/other.json', 'sha256': 'a' * 64}),
    ('result', 'capture', {'path': '/never/open/other.json', 'sha256': 'b' * 64}),
    ('completion', 'formal_result', True),
])
def test_pending_or_mismatched_audit_cannot_authorize_analysis(key, field, value):
    config, parents = example(); parents[key][field] = value
    with pytest.raises(ValueError):
        d.validate_design(config, parents)


def test_input_pin_rehash_or_missing_seed_does_not_change_audited_selection():
    config, parents = example(); config['inputs']['models'][0]['archive']['sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='input identities'):
        d.validate_design(config, parents)


def test_parent_pins_and_version_types_are_not_user_editable_shortcuts():
    config, parents = example(); config['version'] = True
    with pytest.raises(ValueError, match='identity'):
        d.validate_design(config, parents)
    config, parents = example(); config['parents']['audit']['sha256'] = 'f' * 64
    with pytest.raises(ValueError, match='identity'):
        d.validate_design(config, parents)


def test_missing_execution_binding_blocks_before_payload(tmp_path, monkeypatch):
    config, _ = example(); monkeypatch.setattr(d, 'validate', lambda *_: config)
    with pytest.raises(ValueError, match='execution binding required before array payload'):
        d.require_execution_binding(tmp_path / 'science.json', tmp_path / 'absent.json')


def test_fake_complete_role_names_cannot_bind_unrelated_code(tmp_path, monkeypatch):
    config, _ = example(); monkeypatch.setattr(d, 'validate', lambda *_: config)
    source = tmp_path / 'science.json'; source.write_text('{}')
    fake = tmp_path / 'unrelated.py'; fake.write_text('raise RuntimeError("never execute")')
    binding = {'formal_result': False, 'status': 'bound_before_payload', 'science_config': d.p.pin(source),
               'budget': config['budget'], 'files': {role: d.p.pin(fake) for role in d.BINDING_FILES}}
    path = tmp_path / 'binding.json'; path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='bound execution role or code differs'):
        d.require_execution_binding(source, path)


def test_omitted_context_files_are_rejected_before_payload_or_hash_reads(tmp_path, monkeypatch):
    config, parents = example()
    config['context_code'] = []
    config['protocol'] = {'path': str(d.BINDING_FILES['protocol']), 'sha256': 'a' * 64}
    monkeypatch.setattr(d.p, 'load_pin', lambda ref: parents[next(k for k, r in d.PARENTS.items() if r == ref)])
    monkeypatch.setattr(d.p, 'sha', lambda *_: pytest.fail('unexpected hash or payload read'))
    path = tmp_path / 'science.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='context or protocol identities differ'):
        d.validate(path)


def test_default_preflight_and_explicit_byte_checks_do_not_decode_arrays(tmp_path, monkeypatch):
    config, parents = example()
    config['context_code'] = [{'path': str(path), 'sha256': 'a' * 64} for path in d.CONTEXT_FILES]
    config['protocol'] = {'path': str(d.BINDING_FILES['protocol']), 'sha256': 'a' * 64}
    monkeypatch.setattr(d.p, 'load_pin', lambda ref: parents[next(k for k, r in d.PARENTS.items() if r == ref)])
    calls = []
    arrays = [config['inputs']['shared']] + [m['archive'] for m in config['inputs']['models']]
    hashes = {ref['path']: ref['sha256'] for ref in arrays}
    def byte_hash(path):
        calls.append(str(path))
        return hashes.get(str(path), 'a' * 64)
    monkeypatch.setattr(d.p, 'sha', byte_hash)
    path = tmp_path / 'science.json'; path.write_text(json.dumps(config))
    assert d.validate(path)
    assert not any(p.endswith('.npz') for p in calls)
    calls.clear()
    assert d.validate(path, verify_array_hashes=True)
    assert sorted(p for p in calls if p.endswith('.npz')) == sorted(hashes)


def test_metadata_validation_cannot_import_numeric_runtime_or_open_payload(tmp_path):
    config, parents = example(); path = tmp_path / 'synthetic.json'
    path.write_text(json.dumps([config, parents]))
    code = '''
import builtins, importlib.abc, json, pathlib, sys
class BlockNumerical(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'numpy', 'torch', 'h5py', 'libero', 'mujoco', 'robosuite'}:
            raise AssertionError('forbidden numerical import: ' + fullname)
sys.meta_path.insert(0, BlockNumerical())
from experiments import libero_task7_component3_history_transition_protocol as d
cfg, parents = json.loads(pathlib.Path(sys.argv[1]).read_text())
old_open = builtins.open
def guarded_open(file, *args, **kwargs):
    if str(file).endswith(('.npz', '.npy', '.pt', '.hdf5', '.pkl')):
        raise AssertionError('forbidden array/weight/data open')
    return old_open(file, *args, **kwargs)
builtins.open = guarded_open
pathlib.Path.open = lambda self, *a, **kw: guarded_open(self, *a, **kw)
assert d.validate_design(cfg, parents)
assert not {'numpy', 'torch', 'h5py'} & set(sys.modules)
print('metadata-only passed')
'''
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    process = subprocess.run([sys.executable, '-c', code, str(path)], cwd=d.ROOT, env=env,
                             text=True, capture_output=True, timeout=20)
    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == 'metadata-only passed'
