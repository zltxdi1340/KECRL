import json
import numpy as np
import pytest
from experiments import analyze_libero_task4_task7_cached_input_coverage as d


@pytest.mark.parametrize('field,value', [('candidate_indices', list(range(21))), ('checkpoint_updates', 1000),
    ('official_state_files_read', True), ('new_environment_resets_or_actions', 1), ('new_optimizer_updates', 1),
    ('source_role', 'meta_query'), ('device', 'cuda:0')])
def test_role_and_compute_scope_changes_fail_before_cache_read(tmp_path, field, value):
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    config['scope'][field] = value
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen scope'):
        d.load_protocol(path)


def test_reference_threshold_or_primary_change_requires_new_protocol(tmp_path):
    config = json.loads(d.DEFAULT_CONFIG.read_text())
    config['analysis']['reference_quantile'] = .9
    path = tmp_path / 'bad.json'; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='frozen scope or analysis'):
        d.load_protocol(path)


def test_closed_form_bank_distances_and_nearest_order():
    bank = np.arange(5, dtype=np.float32).reshape(-1, 1)
    original = bank.copy()
    fitted = d.fit_bank(bank)
    indices, squared = d.bank_distances(fitted, np.array([[5.]], np.float32))
    assert indices[0].tolist() == [4, 3, 2, 1, 0]
    np.testing.assert_allclose(squared[0], [.5, 2., 4.5, 8., 12.5], atol=1e-12)
    assert np.array_equal(bank, original)
    assert fitted['mean'].tolist() == [2.]


def test_query_and_reference_cannot_change_fitted_bank_statistics():
    fitted = d.fit_bank(np.arange(5, dtype=np.float32).reshape(-1, 1))
    before = (fitted['mean'].copy(), fitted['scale'].copy())
    d.bank_distances(fitted, np.array([[10000.]], np.float32))
    assert np.array_equal(fitted['mean'], before[0]) and np.array_equal(fitted['scale'], before[1])


def test_constant_dimensions_are_finite_and_equal_distances_use_bank_order():
    fitted = d.fit_bank(np.zeros((6, 3), np.float32))
    ids, values = d.bank_distances(fitted, np.zeros((1, 3), np.float32))
    assert ids[0].tolist() == [0, 1, 2, 3, 4]
    assert np.all(values == 0) and np.all(fitted['scale'] == 1)


def test_exceedance_is_strict_and_empty_group_does_not_become_zero():
    assert d.distance_summary(np.array([0., 1., 2.]), 1.)['exceedance_fraction'] == pytest.approx(1 / 3)
    assert d.distance_summary(np.zeros(3), 0.)['exceedance_fraction'] == 0
    with pytest.raises(ValueError, match='empty or invalid'):
        d.distance_summary(np.array([]), 0.)


def test_nonfinite_features_and_too_small_bank_fail_closed():
    with pytest.raises(ValueError, match='invalid coverage bank'):
        d.fit_bank(np.zeros((4, 3)))
    with pytest.raises(ValueError, match='invalid coverage query'):
        d.bank_distances(d.fit_bank(np.zeros((5, 1))), np.array([[np.nan]]))


def fake_results(initial=.6, final=.9, t0=.2):
    return {str(seed): {'banks': {'historical_prefix20': {'fusion': {'horizons': {'0': {'exceedance_fraction': initial},
        '19': {'exceedance_fraction': final}}}}, 'runtime_t0': {'fusion': {'horizons': {'0': {'exceedance_fraction': t0}}}}}}
        for seed in d.SEEDS}


def test_one_favorable_seed_or_average_cannot_satisfy_three_seed_coverage_rule():
    records = fake_results()
    assert d.coverage_decisions(records)['consistent_initial_feature_coverage_gap']
    records[str(d.SEEDS[-1])]['banks']['historical_prefix20']['fusion']['horizons']['0']['exceedance_fraction'] = .4
    result = d.coverage_decisions(records)
    assert not result['consistent_initial_feature_coverage_gap']
    assert not result['runtime_t0_reference_contrast']
    assert result['closed_loop_failure_cause_established'] is False


def test_high_initial_coverage_gap_does_not_invent_more_exceedance_growth():
    result = d.coverage_decisions(fake_results(initial=1., final=1., t0=.2))
    assert result['consistent_initial_feature_coverage_gap']
    assert not result['consistent_feedback_fraction_growth']
    assert result['runtime_t0_reference_contrast']


def test_existing_artifact_is_preserved_before_any_protocol_or_data_access(tmp_path, monkeypatch):
    output = tmp_path / 'old.json'; output.write_text('historical')
    monkeypatch.setattr(d, 'load_protocol', lambda *a: pytest.fail('existing output reached data loading'))
    with pytest.raises(ValueError, match='already exists'):
        d.run(tmp_path / 'unused', output, tmp_path / 'archive')
    assert output.read_text() == 'historical'
