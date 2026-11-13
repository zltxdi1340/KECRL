"""Metadata-only freeze checks; no numerical payload."""
import copy
import pytest
from experiments import libero_task7_lag_neighbor_decomposition_protocol as d


def test_budget_counts_distances_and_repeated_source_labels_separately():
    b = d.budget()
    assert b['per_stage']['paired_lag_distances'] == 6 * 3598 * 2 * 5 * 10
    assert b['per_stage']['coordinate_differences'] == b['per_stage']['paired_lag_distances'] * 256
    assert b['per_stage']['array_files'] == 13 and b['per_stage']['array_members'] == 81
    assert b['total_array_loads'] == 26 and b['total_member_decodes'] == 162
    assert b['tables']['distances'] == b['tables']['frames'] == 21588
    assert b['tables']['attribution'] == 6 * 3 * 2 * 4 * 5


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'checkpoint_updates', 1000), ('scope', 'task_ids', [2, 7]),
    ('scope', 'sealed_roles_accessed', True), ('scope', 'new_nearest_neighbor_searches', 1),
    ('design', 'cells', ['CC', 'HH']), ('design', 'lags', [0, 1]),
    ('design', 'significance_or_automatic_training_promotion', True),
    ('members', 'feature', ['features', 'predictions']), ('budget', 'total_array_loads', 28),
    ('execution', 'technical_retries', 1),
])
def test_scope_compute_or_role_drift_rejected_before_parents(section, key, value):
    config = {'kind': 'task7_cached_lag_fixed_neighbor_decomposition', 'version': 2, 'formal_result': False,
              'status': 'scientific_frozen_execution_unbound', 'parents': copy.deepcopy(d.PARENTS),
              'scope': copy.deepcopy(d.SCOPE), 'design': copy.deepcopy(d.DESIGN),
              'members': copy.deepcopy(d.MEMBERS), 'execution': copy.deepcopy(d.EXECUTION), 'budget': d.budget()}
    config[section][key] = value
    with pytest.raises(ValueError, match='frozen'): d.validate_design(config, {})


def test_no_feature_distance_work_is_mislabeled_as_zero_compute():
    assert 'new_feature_or_full_library_distance_computations' not in d.SCOPE
    assert d.SCOPE['selected_pair_lag_distances_per_stage'] > 0
    assert d.SCOPE['new_full_library_distances'] == 0 and d.SCOPE['new_nearest_neighbor_searches'] == 0


def test_missing_binding_stops_before_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(d, 'validate', lambda *_: {})
    with pytest.raises(ValueError, match='binding required'): d.require_binding(tmp_path/'science', tmp_path/'absent')
