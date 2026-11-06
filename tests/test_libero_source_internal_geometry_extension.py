from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import run_libero_source_internal_geometry_extension_diagnostic as d


CONFIG = d.ROOT / 'configs/libero-source-internal-geometry-extension-diagnostic_v1.json'


def binding(tmp_path):
    previous = json.loads((d.ROOT / 'configs/libero-source-internal-spt-spi-execution-binding_v2.json').read_text())
    previous['files']['runner'] = {
        'path': 'experiments/run_libero_source_internal_geometry_extension_diagnostic.py',
        'sha256': d.sha256(Path(d.__file__))}
    previous['config_path'] = str(CONFIG)
    previous['config_sha256'] = d.sha256(CONFIG)
    path = tmp_path / 'binding.json'
    path.write_text(json.dumps(previous))
    return path


def test_geometry_extrapolation_preflight_reads_only_metadata(tmp_path, monkeypatch):
    pin = binding(tmp_path)
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix not in {'.hdf5', '.pruned_init', '.pt'}
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', guarded)
    config, _ = d.preflight(CONFIG, pin)
    assert [row['probe'] for row in config['folds']] == [4, 7]
    assert all(row['probe_geometry_absent_from_source_updates'] for row in config['geometry_strata'].values())


@pytest.mark.parametrize('key,value', [
    ('folds', [{'probe': 4, 'source_update_tasks': [0, 1, 3, 4]},
               {'probe': 7, 'source_update_tasks': [0, 1, 3, 4]}]),
    ('source_tasks', [0, 1, 2, 4, 7]), ('adaptation_updates', 1000),
    ('source_inner_steps', 2), ('formal_result', True),
    ('validation_qualification_heldout_access', True), ('candidate_acceptance', True),
    ('new_environment_steps', 1), ('official_initial_state_reads', 1),
    ('execution_revision', 'v2_deterministic_equivalent_pool'),
    ('geometry_strata', {}), ('policy_memory_mode', 'cnn_gru')])
def test_role_budget_or_geometry_changes_are_rejected(key, value):
    config = json.loads(CONFIG.read_text())
    config[key] = value
    with pytest.raises(ValueError):
        d.validate_spec(config)


def fit_records():
    return [{'seed': seed, 'arms': {arm: {'curve': {str(step): {} for step in d.CURVE}}
                                   for arm in d.ARMS}} for seed in d.SEEDS]


@pytest.mark.parametrize('fold_index,policies', [(1, 9), (2, 18)])
def test_query_gate_counts_completed_folds_instead_of_task_ids(fold_index, policies):
    d.validate_probe_query_gate(fit_records(), policies, fold_index)


@pytest.mark.parametrize('failure', ['one_seed_missing', 'duplicate_seed', 'arm_missing',
                                    'endpoint_missing', 'old_task_id_arithmetic'])
def test_query_payload_stays_closed_for_incomplete_support_fits(failure):
    records = deepcopy(fit_records())
    policies = 9
    if failure == 'one_seed_missing':
        records.pop()
    elif failure == 'duplicate_seed':
        records[-1]['seed'] = records[0]['seed']
    elif failure == 'arm_missing':
        records[0]['arms'].pop('C_pooled_bc')
    elif failure == 'endpoint_missing':
        records[0]['arms']['B_fomaml']['curve'].pop('500')
    else:
        policies = (4 + 1) * 9
    with pytest.raises(ValueError, match='query gate'):
        d.validate_probe_query_gate(records, policies, 1)


def test_old_metadata_gate_remains_valid_and_bound_runner_unchanged(tmp_path):
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import preflight
    original = d.ROOT / 'configs/libero-source-internal-spt-spi-diagnostic_v2.json'
    pin = d.ROOT / 'configs/libero-source-internal-spt-spi-execution-binding_v2.json'
    config, _ = preflight(original, pin)
    assert [row['probe'] for row in config['folds']] == [0, 1]
