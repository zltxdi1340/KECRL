"""Synthetic metadata checks for the scientific freeze; no real data or models."""
import copy
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_source_expert_component3_observability_protocol as d


def example():
    counts = {str(t): [{'tail20': {'negative': 6, 'positive': 6}, 'complement': {'negative': 2, 'positive': 2}}
                      for _ in range(30)] for t in d.p.TASKS}
    source = {'action_lengths': {str(t): [25] * 30 for t in d.p.TASKS}, 'data_files': {}, 'data_root': '/never/open',
              'G_reference_selection': {'effective_policy_config': {'memory_mode': 'cnn_only', 'recurrent_hidden_dim': 256}},
              'endpoints': [{'task': t, 'seed': s, 'arm': a, 'checkpoint_update': 500, 'support_role': 'meta_support',
                             'checkpoint': {'path': '/never/open/model.pt', 'sha256': 'a' * 64, 'weight_hash': 'b' * 64}}
                            for t in d.p.TASKS for s in d.p.SEEDS for a in d.p.ARMS]}
    config = {'formal_result': False, 'status': 'scientific_protocol_frozen_execution_unbound',
              'scope': copy.deepcopy(d.SCOPE), 'readout': copy.deepcopy(d.READOUT), 'execution': copy.deepcopy(d.EXECUTION),
              'endpoints': copy.deepcopy(source['endpoints']), 'action_lengths': source['action_lengths'],
              'data_files': {}, 'data_root': source['data_root'], 'policy_config': source['G_reference_selection']['effective_policy_config'],
              'budget': d.derive_budget(source['action_lengths'], counts)}
    return config, source, counts


def test_budget_separates_exact_forward_counts_from_label_dependent_bounds():
    config, source, counts = example()
    assert d.validate_design(config, source, counts)
    budget = config['budget']; exact = budget['exact']; caps = budget['label_dependent_upper_bounds']
    assert exact['main_forward_batches'] == 90 and exact['main_valid_rows'] == 21600
    assert exact['common_rows_before_neutral_exclusion_shared'] == 3000
    assert caps['material_common_rows_shared'] == 1800  # With length25 all t>=5 are in the known tail20.
    assert caps['primary_scored_output_rows'] == 43200
    assert exact['primary_fold_mode_rule_records'] == 3600 and exact['secondary_fold_mode_rule_records'] == 600
    assert exact['total_capture_probe_audit_array_file_loads'] == 205
    assert exact['new_optimizer_updates'] == exact['new_environment_calls'] == 0


def test_early_history_exclusion_uses_known_tail_and_reference_class_lower_bound():
    config, _, _ = example(); row = config['budget']['per_task'][0]
    assert row['source_t_ge_1_rows'] == 720 and row['common_t_ge_5_rows'] == 600
    assert row['material_common_rows_lower'] == row['material_common_rows_upper'] == 360
    assert row['reference_class_rows_lower_in_any_fold'] == {'negative': 174, 'positive': 174}
    assert row['tail20_material_class_rows_exact'] == {'negative': 180, 'positive': 180}


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'new_optimizer_updates', 1), ('scope', 'source_query_accessed', True),
    ('scope', 'task_ids', [0, 1, 2, 4, 7]), ('scope', 'component_index', 6),
    ('scope', 'common_first_action_step', 1), ('scope', 'source_query_accessed', 0),
    ('readout', 'neighbors_per_class', 3), ('readout', 'scaler_fit_rows', 'include_query_demo'),
    ('readout', 'balanced_distance_tie', 1), ('execution', 'device_or_cpu_capture_fallback', True),
])
def test_scope_label_isolation_and_compute_rule_drift_fail_closed(section, key, value):
    config, source, counts = example(); config[section][key] = value
    with pytest.raises(ValueError, match='frozen scientific'):
        d.validate_design(config, source, counts)


def test_bound_cannot_be_raised_after_metadata_planning():
    config, source, counts = example(); config['budget']['exact']['main_forward_batches'] += 1
    with pytest.raises(ValueError, match='derived scientific budgets'):
        d.validate_design(config, source, counts)


def test_duplicate_or_changed_endpoints_are_not_replaced():
    config, source, counts = example(); config['endpoints'][1] = config['endpoints'][0]
    with pytest.raises(ValueError, match='all thirty'):
        d.validate_design(config, source, counts)


def test_insufficient_class_metadata_stops_instead_of_reducing_k():
    _, source, counts = example()
    for row in counts['0']:
        row['tail20']['positive'] = row['complement']['positive'] = 0
    with pytest.raises(ValueError, match='five references per class'):
        d.derive_budget(source['action_lengths'], counts)


def make_count_csv(path):
    rows = []
    for t in d.p.TASKS:
        for s in d.p.SEEDS:
            for a in d.p.ARMS:
                for demo in range(30):
                    for phase in ('tail20', 'complement'):
                        for direction in ('all', 'negative', 'zero', 'positive'):
                            for band in ('all', 'material'):
                                n = 0 if direction == 'zero' else (6 if phase == 'tail20' else 2)
                                if direction == 'all': n *= 2
                                rows.append({'task': t, 'seed': s, 'arm': a, 'demo': demo, 'phase': phase,
                                             'direction': direction, 'band': band, 'n': n})
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return rows


def test_shared_count_reader_retains_all_models_and_demos(tmp_path):
    path = tmp_path / 'counts.csv'; make_count_csv(path)
    counts = d.read_shared_counts(path)
    assert counts == example()[2]


@pytest.mark.parametrize('kind', ['duplicate', 'missing', 'shared_label_change'])
def test_count_metadata_cannot_silently_drop_or_disagree(tmp_path, kind):
    path = tmp_path / 'counts.csv'; rows = make_count_csv(path)
    if kind == 'duplicate': rows.append(rows[0])
    elif kind == 'missing': rows.pop()
    else:
        row = next(r for r in rows if r['task'] == 0 and r['seed'] == d.p.SEEDS[1] and r['arm'] == d.p.ARMS[0]
                   and r['demo'] == 0 and r['phase'] == 'tail20' and r['direction'] == 'positive' and r['band'] == 'material')
        row['n'] += 1
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with pytest.raises(ValueError): d.read_shared_counts(path)


def test_absent_execution_binding_blocks_before_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'validate', lambda *_: example()[0])
    with pytest.raises(ValueError, match='execution binding required'):
        d.require_execution_binding(tmp_path / 'science.json', tmp_path / 'missing-binding.json')


def test_fake_complete_role_names_cannot_bind_an_unrelated_file(tmp_path, monkeypatch):
    config, _, _ = example(); monkeypatch.setattr(d, 'validate', lambda *_: config)
    science = tmp_path / 'science.json'; science.write_text('{}')
    binding = {'formal_result': False, 'status': 'bound_before_payload', 'science_config': d.p.pin(science),
               'budget': config['budget'], 'files': {key: d.p.pin(science) for key in d.BINDING_FILES}}
    path = tmp_path / 'binding.json'; path.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='wrong implementation'):
        d.require_execution_binding(science, path)


def test_metadata_protocol_does_not_import_numeric_or_environment_runtime():
    code = '''
import builtins
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name.split('.')[0] in {'numpy','torch','h5py','libero','robosuite','mujoco'}:
        raise AssertionError(name)
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from experiments import libero_source_expert_component3_observability_protocol
'''
    subprocess.run([sys.executable, '-c', code], cwd=d.p.ROOT, check=True, capture_output=True, text=True,
                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})
