"""Classification boundaries, error budgets and immutable cache provenance."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from experiments import analyze_libero_source_expert_component3_error_types as a
from experiments import audit_libero_source_expert_component3_error_types as auditor
from experiments import audit_libero_source_expert_translation_amplitude as previous_audit
from experiments import libero_source_expert_late_translation_protocol as p
from test_libero_source_expert_translation_amplitude import cache, dump


def test_strict_partition_preserves_both_zero_cases_and_negative_magnitudes():
    truth = np.array([1., -1, 1, -1, -.5, .5, 0, 0, 1e-12])
    pred = np.array([-.1, .1, 0, -.5, -1, .5, 0, .2, -.2])
    assert a.classify(pred, truth).tolist() == [0, 0, 1, 2, 3, 4, 4, 5, 0]


def test_frequency_and_error_energy_are_distinct():
    truth = np.ones(11); pred = np.array([.9] * 10 + [-1.])
    row = a.statistics(pred, truth, a.classify(pred, truth))
    assert row['wrong_sign_frame_fraction'] == pytest.approx(1 / 11)
    assert row['wrong_sign_error_share'] == pytest.approx(4 / 4.1)
    assert sum(row[k + '_mse_contribution'] for k in a.CATEGORIES) == pytest.approx(row['mse'])
    assert sum(row[k + '_n'] for k in a.CATEGORIES) == 11


def test_material_margin_is_symmetric_and_does_not_relabel_strict_signs():
    truth = np.array([.025, -.025, 1, -1, .024, -.024])
    pred = np.array([-.01, .01, -1, 1, -1, 1])
    row = a.statistics(pred, truth, a.classify(pred, truth))
    assert row['wrong_sign_n'] == 6
    assert row['wrong_sign_small_prediction_n'] == 2
    assert row['wrong_sign_material_prediction_n'] == 2


def test_empty_direction_is_retained_without_epsilon_or_fake_fractions():
    row = a.statistics(np.array([]), np.array([]), np.array([], np.uint8))
    assert row['n'] == row['error_sse'] == 0
    assert row['mse'] is row['error_over_zero'] is None
    assert all(row[k + '_n'] == 0 and row[k + '_frame_fraction'] is None for k in a.CATEGORIES)


def test_exact_match_and_zero_target_error_energy():
    truth = np.zeros(2); pred = np.array([0., .3])
    row = a.statistics(pred, truth, a.classify(pred, truth))
    assert row['exact_match_n'] == row['zero_target_nonzero_n'] == 1
    assert row['zero_target_nonzero_error_share'] == 1
    assert row['error_over_zero'] is None


def test_concentration_ties_are_deterministic_and_keep_all_thirty_demos():
    row = a.concentration(np.ones(30), np.arange(30), np.ones(30, bool))
    assert row['top5_demos'] == '0;1;2;3;4'
    assert row['top5_error_share'] == pytest.approx(1 / 6)
    assert row['nonzero_error_demos'] == 30


@pytest.mark.parametrize('kind', ['nan', 'shape', 'infinity'])
def test_malformed_classification_fails_closed(kind):
    pred = np.array([np.nan]) if kind == 'nan' else np.array([np.inf]) if kind == 'infinity' else np.zeros((1, 1))
    with pytest.raises(ValueError, match='invalid component3'):
        a.classify(pred, np.zeros(1))


@pytest.fixture
def component_cache(cache):
    base_path, base = cache
    a.previous.analyze(base_path)
    old_audit = Path(base['artifacts']) / 'independent-audit.json'
    previous_audit.audit(base['output'], old_audit)
    directory = base_path.parent
    cfg = {'kind': 'source_support_fixed500_component3_error_types_readonly', 'formal_result': False,
           'source_tasks': p.TASKS, 'seeds': p.SEEDS, 'arms': p.ARMS, 'component_index': 2,
           'checkpoint_update': 500, 'categories': list(a.CATEGORIES), 'margin': a.MARGIN,
           'parents': {'amplitude_config': p.pin(base_path), 'amplitude_result': p.pin(base['output']), 'amplitude_audit': p.pin(old_audit)},
           'code': [p.pin(Path(a.__file__)), p.pin(Path(auditor.__file__)), p.pin(Path(__file__))],
           'budget': base['budget'], 'audit_budget': base['audit_budget'],
           'output': str(directory / 'component3-result.json'), 'artifacts': str(directory / 'component3')}
    path = directory / 'component3-config.json'; dump(path, cfg)
    return path, cfg


def test_complete_cohort_independent_audit_and_all_demo_directions(component_cache):
    path, cfg = component_cache
    result = a.analyze(path)
    assert result['prediction_rows'] == 21600
    assert result['prior_scalar_checks'] == 360
    assert result['tables']['demo-decomposition']['rows'] == 14400
    assert result['tables']['cross-seed-consensus']['rows'] == 240
    verified = auditor.audit(cfg['output'], Path(cfg['artifacts']) / 'audit.json')
    assert verified['scalar_checks'] > 500000
    assert verified['array_loads'] == {'label_npz_loads': 5, 'prediction_npy_loads': 30}


@pytest.mark.parametrize('kind', ['margin', 'component', 'parent_hash'])
def test_frozen_scope_or_lineage_drift_rejected_before_payload(component_cache, kind):
    path, cfg = component_cache
    if kind == 'margin': cfg['margin'] = .03
    elif kind == 'component': cfg['component_index'] = 1
    else: cfg['parents']['amplitude_result']['sha256'] = '0' * 64
    dump(path, cfg)
    with pytest.raises(ValueError): a.preflight(path)


def test_independent_audit_rejects_category_tamper_after_rehash(component_cache):
    path, cfg = component_cache; result = a.analyze(path)
    item = result['tables']['model-decomposition']
    with Path(item['path']).open(newline='') as handle: rows = list(csv.DictReader(handle))
    rows[0]['wrong_sign_n'] = str(int(rows[0]['wrong_sign_n']) + 1)
    with Path(item['path']).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    item['sha256'] = p.sha(item['path']); dump(cfg['output'], result)
    with pytest.raises(ValueError, match='independent component3 scalar differs'):
        auditor.audit(cfg['output'], Path(cfg['artifacts']) / 'audit.json')


def test_refuse_existing_output_preserves_completed_analysis(component_cache):
    path, cfg = component_cache; a.analyze(path)
    original = p.sha(cfg['output'])
    with pytest.raises(ValueError, match='refuse existing'): a.analyze(path)
    assert p.sha(cfg['output']) == original
