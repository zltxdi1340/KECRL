"""Synthetic freeze checks; no real predictions, tag rows or numerical runtime."""
import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_policy_error_focus_alignment_protocol as d


def example():
    parents = {key: {'formal_result': False} for key in d.PARENTS}
    parents['lag_audit'].update(status='passed_independent_task7_lag_neighbor_audit', result=d.PARENTS['lag_result'],
                                science_config=d.PARENTS['lag_science'], binding=d.PARENTS['lag_binding'])
    parents['lag_completion'].update(status='complete_independently_audited_task7_lag_neighbor', result=d.PARENTS['lag_result'],
                                     audit=d.PARENTS['lag_audit'], science_config=d.PARENTS['lag_science'], binding=d.PARENTS['lag_binding'])
    parents['transition_audit'].update(status='passed_independent_task7_history_transition_audit', result=d.PARENTS['transition_result'])
    parents['transition_completion'].update(status='complete_independently_audited_task7_history_transitions',
                                            result=d.PARENTS['transition_result'], audit=d.PARENTS['transition_audit'])
    pin = lambda name: {'path': '/never/open/' + name, 'sha256': 'a' * 64}
    models, captures, legacy = [], [], []
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            checkpoint = {**pin(f'{seed}-{arm}.pt'), 'weight_hash': 'b' * 64}
            feature = pin(f'{seed}-{arm}-features.npz')
            metrics = {'mse': .04, 'translation_mse': .04, 'rotation_mse': .04, 'gripper_mse': .04,
                       'gripper_sign_accuracy': .75, 'translation_axis_mse': [.04] * 3, 'translation_axis_bias': [0.] * 3}
            strata = {phase: {**copy.deepcopy(metrics), 'valid_steps': n}
                      for phase, n in [('all', 4188), ('tail20', 600), ('complement', 3588)]}
            identity = {'task': 7, 'seed': seed, 'arm': arm, 'checkpoint': checkpoint}
            models.append({**identity, 'feature': feature})
            captures.append({**identity, 'features': feature, 'legacy_prediction_bit_exact': True, 'G500_parity': copy.deepcopy(strata['all'])})
            legacy.append({**identity, 'G500_parity_passed': True, 'strata': strata})
    parents['lag_science']['inputs'] = {'shared': pin('shared.npz'), 'models': models, 'action_lengths': [140] * 12 + [141] * 18}
    parents['capture']['models'] = captures; parents['legacy_result']['models'] = legacy
    tag = lambda name, key: {**pin(name), 'rows': 21588, 'columns': list(d.CSV_COLUMNS[key])}
    parents['transition_result']['tables'] = {'frame-transitions': tag('old.csv', 'transition_tags')}
    parents['lag_result']['tables'] = {'frames': tag('lag.csv', 'lag_tags')}
    cfg = {'kind': 'task7_original_Policy_action_error_focus_alignment', 'version': 1, 'formal_result': False,
           'status': 'scientific_frozen_execution_unbound', 'parents': copy.deepcopy(d.PARENTS),
           'scope': copy.deepcopy(d.SCOPE), 'design': copy.deepcopy(d.DESIGN), 'members': copy.deepcopy(d.MEMBERS),
           'CSV_columns': copy.deepcopy(d.CSV_COLUMNS), 'execution': copy.deepcopy(d.EXECUTION), 'budget': d.budget(),
           'inputs': d.select_inputs(parents), 'output': d.OUTPUT, 'artifacts': d.ARTIFACTS}
    return cfg, parents


def test_complete_synthetic_metadata_and_exact_load_row_budgets():
    cfg, parents = example(); assert d.validate_design(cfg, parents)
    b = cfg['budget']
    assert b['raw_Policy_prediction_rows'] == 6 * b['shared_tge1_target_rows'] == 25128
    assert b['raw_seven_axis_squared_error_coordinates'] == 25128 * 7
    assert b['per_stage']['source_NPZ_members'] == 5 + 6
    assert b['total_NPZ_loads'] == 14 and b['total_NPZ_member_decodes'] == 22
    assert b['total_tag_CSV_passes'] == 4 and b['per_stage']['projected_CSV_cells'] == 21588 * 24
    assert sum(b['tables'].values()) == 38990 and b['legacy_scalar_parity_checks'] == 6 * 3 * 12


def test_all_binary_tags_preserve_primary_partition_and_secondary_parent():
    for truth, current, history, hc in itertools.product((-1, 1), repeat=4):
        groups = d.group_memberships(truth, current, history, hc)
        assert groups[0] == 'all_same_sign' and groups[1] in d.ORIGINAL_GROUPS
        assert sum(name in groups for name in d.ORIGINAL_GROUPS) == 1
        expected = ('stays_correct' if history == truth else 'new_error') if current == truth else ('corrected' if history == truth else 'stays_wrong')
        assert groups[1] == expected
        assert (len(groups) == 3) == (current == history == -1)
        if len(groups) == 3:
            assert groups[2] == ('cancelled_internal_change' if hc == 1 else 'cancelled_internal_control')
            assert groups[1] == ('stays_wrong' if truth == 1 else 'stays_correct')


@pytest.mark.parametrize('value', [0, True, 1., '1'])
def test_nonbinary_or_noninteger_readout_tags_fail(value):
    with pytest.raises(ValueError, match='binary'): d.group_memberships(value, 1, 1, 1)


def test_references_match_target_sign_and_original_readout_correctness():
    assert d.contrast_groups('positive', 'readout_change') == ('corrected', 'stays_wrong')
    assert d.contrast_groups('negative', 'readout_change') == ('new_error', 'stays_correct')
    assert d.contrast_groups('negative', 'cancelled_internal_change') == ('cancelled_internal_change', 'cancelled_internal_control')
    with pytest.raises(ValueError): d.contrast_groups('zero', 'readout_change')


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'checkpoint_updates', 1000), ('scope', 'task_ids', [0, 7]),
    ('scope', 'source_meta_query_accessed', True), ('scope', 'sealed_roles_accessed', True),
    ('scope', 'new_CNN_Policy_or_environment_calls', 1), ('scope', 'new_optimizer_updates', False),
    ('scope', 'new_neighbor_search_feature_distance_or_lag_sweeps', 1),
    ('scope', 'closed_loop_default_stabilization_steps', 0),
    ('members', 'prediction', ['predictions', 'features']),
    ('CSV_columns', 'lag_tags', d.CSV_COLUMNS['lag_tags'] + ['HC_gap']),
    ('design', 'primary_reference', 'choose_lowest_error_control'),
    ('design', 'no_error_size_acceptance_threshold_or_pvalues', False),
    ('budget', 'total_NPZ_loads', 16), ('execution', 'technical_retries', 1),
])
def test_scope_fields_metrics_and_budget_drift_fail_closed(section, key, value):
    cfg, parents = example(); cfg[section][key] = value
    with pytest.raises(ValueError, match='frozen'): d.validate_design(cfg, parents)


@pytest.mark.parametrize('tamper', ['checkpoint', 'feature_hash', 'legacy_parity', 'G500', 'counts', 'metrics',
                                   'lengths', 'missing_model', 'duplicate_model', 'tag_projection', 'tag_rows'])
def test_parent_cache_tag_or_metric_mismatch_is_not_repaired(tamper):
    cfg, parents = example()
    if tamper == 'checkpoint': parents['legacy_result']['models'][0]['checkpoint'] = {'path': '/wrong.pt'}
    elif tamper == 'feature_hash': parents['capture']['models'][0]['features'] = {'path': '/wrong.npz', 'sha256': 'b' * 64}
    elif tamper == 'legacy_parity': parents['capture']['models'][0]['legacy_prediction_bit_exact'] = False
    elif tamper == 'G500': parents['legacy_result']['models'][0]['G500_parity_passed'] = False
    elif tamper == 'counts': parents['legacy_result']['models'][0]['strata']['tail20']['valid_steps'] -= 1
    elif tamper == 'metrics': parents['legacy_result']['models'][0]['strata']['tail20']['mse'] = .9
    elif tamper == 'lengths': parents['lag_science']['inputs']['action_lengths'][0] += 1
    elif tamper == 'missing_model': parents['lag_science']['inputs']['models'].pop()
    elif tamper == 'duplicate_model': parents['lag_science']['inputs']['models'][-1] = parents['lag_science']['inputs']['models'][0]
    elif tamper == 'tag_projection': parents['lag_result']['tables']['frames']['columns'].remove('HC_prediction')
    elif tamper == 'tag_rows': parents['transition_result']['tables']['frame-transitions']['rows'] -= 1
    with pytest.raises((ValueError, KeyError)): d.validate_design(cfg, parents)


def test_failed_or_unlinked_audit_cannot_define_focus_cohorts():
    cfg, parents = example(); parents['lag_completion']['audit'] = {'path': '/other', 'sha256': 'c' * 64}
    with pytest.raises(ValueError, match='lag parent'): d.validate_design(cfg, parents)


def test_missing_binding_rejects_before_numeric_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'validate', lambda *_: example()[0])
    with pytest.raises(ValueError, match='binding required'): d.require_binding(tmp_path/'science', tmp_path/'absent')


def test_complete_role_names_cannot_bind_unrelated_files(tmp_path, monkeypatch):
    cfg, _ = example(); monkeypatch.setattr(d, 'validate', lambda *_: cfg)
    science = tmp_path/'science.json'; science.write_text('{}')
    fake = tmp_path/'unrelated.py'; fake.write_text('raise RuntimeError()')
    binding = {'status': 'bound_before_payload', 'formal_result': False, 'science_config': d.p.pin(science),
               'budget': cfg['budget'], 'files': {key: d.p.pin(fake) for key in d.BINDING_FILES}}
    path = tmp_path/'binding.json'; path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='role/code'): d.require_binding(science, path)


def test_freeze_spec_requires_no_numeric_runtime_or_payload(tmp_path):
    cfg, parents = example(); path = tmp_path/'synthetic.json'; path.write_text(json.dumps([cfg, parents]))
    code = '''
import builtins,importlib.abc,json,pathlib,sys
class NoNumeric(importlib.abc.MetaPathFinder):
 def find_spec(self,name,*args):
  if name.split('.')[0] in {'numpy','torch','h5py','libero','mujoco','robosuite'}:raise AssertionError(name)
sys.meta_path.insert(0,NoNumeric())
from experiments import libero_task7_policy_error_focus_alignment_protocol as d
cfg,parents=json.loads(pathlib.Path(sys.argv[1]).read_text())
def deny(*a,**k):raise AssertionError('unexpected payload/file read')
builtins.open=deny;pathlib.Path.open=deny
assert d.validate_design(cfg,parents)
assert not {'numpy','torch','h5py'} & set(sys.modules)
print('metadata_only_passed')
'''
    env = os.environ.copy(); env['CUDA_VISIBLE_DEVICES'] = ''
    result = subprocess.run([sys.executable, '-c', code, str(path)], cwd=d.ROOT, env=env,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'metadata_only_passed'
