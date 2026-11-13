"""Human scalars/vectors and metadata only: no research payload or SHA reads."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from experiments import libero_task7_reference_joint_reachability_protocol as d


def theta(first):
    return [float(first)] + [0.] * 256


def refs():
    return {r: d.point(.7 + .07 * i, .03 + .01 * i) for i, r in enumerate(d.REFERENCES)}


def scalar_loss(parameter, target=0.):
    """One invented fit row, original prediction0, fixed feature1; no file input."""
    prediction = max(-1., min(1., parameter))
    mse = (prediction - target) ** 2
    old = target ** 2
    return d.point(mse, max(mse - old, 0.) / 7.)


@pytest.fixture(scope='module')
def canonical():
    return d.create()


def test_caps_keep_all_four_and_all_tied_roles():
    r = refs()
    r['bisector'] = d.point(.49, .2)
    assert d.caps(r) == {'M': .49, 'T': r['control']['T'], 'M_active': ['bisector'], 'T_active': ['control']}
    r = {name: d.point(.7, .03) for name in d.REFERENCES}
    assert d.caps(r)['M_active'] == d.caps(r)['T_active'] == d.REFERENCES


@pytest.mark.parametrize('m', [.1, .49, .7, math.nextafter(.7, math.inf), 1.])
@pytest.mark.parametrize('penalty', [0., .03, .2])
def test_same_parameter_caps_equivalent_to_all_eight_original_comparisons(m, penalty):
    r = refs()
    r['bisector'] = d.point(.49, .2)
    value = d.point(m, penalty)
    result = d.gates(value, r)
    assert result['all_four'] == all(m <= r[n]['M'] and value['T'] < r[n]['T'] for n in d.REFERENCES)
    assert result['all_four'] == (m <= min(v['M'] for v in r.values()) and value['T'] < min(v['T'] for v in r.values()))


def test_original_strict_T_and_nonstrict_M_are_raw_nextafter_boundaries():
    r = {name: d.point(.7, .03) for name in d.REFERENCES}
    assert d.gates(d.point(.7, 0.), r)['all_four']
    assert not d.gates(d.point(math.nextafter(.7, math.inf), 0.), r)['all_four']
    assert not d.gates(r['control'], r)['all_four']
    lower = dict(r['control'], T=math.nextafter(r['control']['T'], -math.inf))
    assert d.gates(lower, r)['all_four']
    assert not d.gates(dict(r['control'], T=math.nextafter(r['control']['T'], math.inf)), r)['all_four']


def test_cannot_combine_M_and_T_from_different_parameters_or_impose_new_P_bound():
    r = {name: d.point(.07, .09) for name in d.REFERENCES}
    first, second = d.point(.01, .1), d.point(.08, 0.)
    assert d.gates(first, r)['cap_pass'] == {'M': True, 'T': False}
    assert d.gates(second, r)['cap_pass'] == {'M': False, 'T': True}
    assert not any(d.gates(v, r)['all_four'] for v in [first, second])
    third = d.point(0., .095)
    assert third['P'] > r['control']['P'] and d.gates(third, r)['all_four']


@pytest.mark.parametrize('name', d.REFERENCES)
def test_missing_reference_fails_closed(name):
    r = refs()
    del r[name]
    with pytest.raises(ValueError):
        d.caps(r)


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1., 0, True, None])
def test_invalid_objective_input_fails_closed(value):
    with pytest.raises(ValueError):
        d.point(value, 0.)


def test_conservation_tolerance_is_not_a_decision_tolerance():
    x = d.point(.7, .03)
    with pytest.raises(ValueError):
        d.validate_point(dict(x, T=x['T'] + 1e-8))
    with pytest.raises(ValueError):
        d.validate_point(dict(x, injected=0.))
    r = {name: x.copy() for name in d.REFERENCES}
    assert not d.gates(dict(x, M=math.nextafter(x['M'], math.inf), T=x['T'] - 1e-14), r)['all_four']


def test_replay_prevents_self_reference_roundoff_witness_and_never_replaces_cached_cap():
    replay = {name: d.point(.7, .03) for name in d.REFERENCES}
    cached = {name: dict(v, T=math.nextafter(v['T'], math.inf)) for name, v in replay.items()}
    value = replay['control'].copy()
    result = d.replay_gates(value, cached, replay)
    assert result['raw_cached_pass'] and not result['replay_confirmed']
    assert result['cached']['caps']['T'] == cached['control']['T']
    assert result['replayed']['caps']['T'] == replay['control']['T']


def test_replay_mismatch_is_technical_failure_not_a_relaxed_threshold():
    r = refs()
    replay = copy.deepcopy(r)
    replay['control'] = d.point(r['control']['M'] + 1e-8, r['control']['P'])
    with pytest.raises(ValueError, match='reference replay'):
        d.replay_gates(d.point(.1, 0.), r, replay)


def test_same_parameter_raw_boolean_disagreement_retained_within_numeric_tolerance():
    r = {name: d.point(.7, .03) for name in d.REFERENCES}
    first = dict(r['control'], T=math.nextafter(r['control']['T'], -math.inf))
    second = r['control'].copy()
    result = d.same_parameter_witness(theta(0), theta(0), first, second, r, r, r)
    assert result['raw_flag_disagreement'] and result['confirmed_flag_disagreement'] and not result['verified']


@pytest.mark.parametrize('change', ['different_parameter', 'signed_zero', 'numeric_mismatch'])
def test_same_parameter_audit_cannot_be_replaced_by_a_nearby_path(change):
    r = refs()
    a, b = theta(0), theta(0)
    x, y = d.point(.1, .01), d.point(.1, .01)
    if change == 'different_parameter':
        b[0] = math.nextafter(a[0], math.inf)
    elif change == 'signed_zero':
        b[-1] = -0.
    else:
        y = d.point(.1 + 1e-8, .01)
    with pytest.raises(ValueError):
        d.same_parameter_witness(a, b, x, y, r, r, r)


def test_four_references_six_edges_190_slots_canonical_order_and_binary_exact_grid():
    rows = d.recipes()
    assert len(rows) == len({r['recipe_id'] for r in rows}) == 190
    assert [r['left'] for r in rows[:4]] == d.REFERENCES
    pairs = [(d.REFERENCES[i], right) for i in range(4) for right in d.REFERENCES[i + 1:]]
    for i, pair in enumerate(pairs):
        group = rows[4 + i * 31:4 + (i + 1) * 31]
        assert [(r['left'], r['right']) for r in group] == [pair] * 31
        assert [r['numerator'] for r in group] == list(range(1, 32))
        assert all(r['denominator'] == 32 and r['numerator'] / 32. == math.ldexp(float(r['numerator']), -5) for r in group)


def test_reference_copy_keeps_signed_zeros_and_no_metric_based_dedup():
    r = {name: theta(1) for name in d.REFERENCES}
    r['control'][0] = -0.
    value = d.interpolate(r, d.recipes()[0])
    assert value is not r['control'] and value[0].hex() == '-0x0.0p+0'
    duplicated = {name: theta(1) for name in d.REFERENCES}
    values = [d.interpolate(duplicated, recipe) for recipe in d.recipes()]
    assert len(values) == 190 and all(v == theta(1) for v in values)


@pytest.mark.parametrize('index', [4, 20, 34, 35, 65, 97, 158, 189])
def test_interpolation_preserves_explicit_operation_order(index):
    r = {name: [float(i + .125)] * 257 for i, name in enumerate(d.REFERENCES)}
    recipe = d.recipes()[index]
    value = d.interpolate(r, recipe)
    x, y = r[recipe['left']][0], r[recipe['right']][0]
    expected = x + (recipe['numerator'] / 32.) * (y - x)
    assert len(value) == 257 and all(v.hex() == expected.hex() for v in value)


@pytest.mark.parametrize('failure', ['small_vector', 'nonfinite', 'int_coordinate', 'overflow', 'extra_reference', 'grid64', 'bool_numerator'])
def test_grid_or_coordinate_changes_fail_closed(failure):
    r = {name: theta(i) for i, name in enumerate(d.REFERENCES)}
    recipe = copy.deepcopy(d.recipes()[4])
    if failure == 'small_vector':
        r['control'].pop()
    elif failure == 'nonfinite':
        r['zero'][1] = float('nan')
    elif failure == 'int_coordinate':
        r['control'][2] = 0
    elif failure == 'overflow':
        r['control'][0], r['margin'][0] = -1.7e308, 1.7e308
    elif failure == 'extra_reference':
        r['candidate'] = theta(0)
    elif failure == 'grid64':
        recipe['denominator'] = 64
    else:
        recipe['numerator'] = True
    with pytest.raises(ValueError):
        d.interpolate(r, recipe)


def test_invented_clip_objective_full_grid_has_an_independently_confirmable_witness():
    parameters = {r: theta(v) for r, v in zip(d.REFERENCES, [-2, -1, 1, 2])}
    r = {name: scalar_loss(v[0]) for name, v in parameters.items()}
    verified = []
    for recipe in d.recipes():
        value = d.interpolate(parameters, recipe)
        x = scalar_loss(value[0])
        # Separate closed-form scalar implementation for this invented zero-target row.
        pred = 1. if value[0] > 1. else -1. if value[0] < -1. else value[0]
        square = pred * pred
        other = dict(M=square, P=square / 7., T=square / 7. + square / 7.)
        verified.append(d.same_parameter_witness(value, list(value), x, other, r, r, r)['verified'])
    assert len(verified) == 190 and not any(verified[:4]) and any(verified[4:])
    assert d.result_label(verified) == 'verified_endpoint_witness'


def test_grid_negative_cannot_prove_unreachable_even_on_a_single_continuous_edge():
    # Invented feasible interval of width2e-6 around1/64, between frozen1/32 points.
    parameters = {name: theta(i) for i, name in enumerate(d.REFERENCES)}
    target = 1. / 64.
    feasible = lambda t: abs(t - target) < 1e-6
    hits = [feasible(d.interpolate(parameters, recipe)[0]) for recipe in d.recipes()]
    assert not any(hits) and feasible(target)
    assert d.result_label(hits) == 'not_found_on_frozen_grid'


@pytest.mark.parametrize('flags', [[], [False] * 189, [True] * 191, [0] * 190])
def test_no_early_stop_or_partial_grid_classification(flags):
    with pytest.raises(ValueError):
        d.result_label(flags)


def test_exact_budget_conservation_no_new_optimization_or_policy_calls():
    b = d.budget()
    assert b['recipe_slots_per_head'] == 4 + 6 * 31 == 190
    assert b['core_forwards_total'] == 18 * 190 * 4 == 13680
    assert b['forward_rows_total'] == 6 * (2782 + 2817 + 2777) * 190 * 4 == 38194560
    assert b['feature_multiplies_total'] == b['forward_rows_total'] * 256 == 9777807360
    assert b['parameter_forms_total'] == sum(b['parameter_forms_by_role'].values()) == 10260
    assert b['parameter_coordinate_writes_total'] == 2636820
    assert b['original_source_scalar_projections'] == 6 * 54 * len(d.HEADER) + 144 * len(d.ENDPOINT) + 36 * len(d.PAIR) == 4212
    assert b['original_source_parameter_coordinates'] == 2 * 18 * 4 * 257 == 37008
    assert b['original_source_total_projection_values'] == 4212 + 37008 == 41220
    assert b['parameter_coordinates_read_total'] == 37008 + 18504 == 55512
    assert b['NPZ_loads_total'] == 2 * (1 + 6 + 18) == 50
    assert b['NPZ_decodes_total'] == 2 * (5 + 6 * 2 + 18 * 6) == 250
    assert b['NPZ_SHA_total'] == 2 * b['NPZ_loads_total']
    assert b['logical_gate_comparisons_total'] == 5 * 18 * 190 * 20 == 342000
    assert b['logical_cap_reductions_total'] == 5 * 18 * 2 * 2 == 360
    assert b['logical_cap_input_scalars_total'] == 4 * b['logical_cap_reductions_total']
    assert b['probe_CSV_rows_total'] == sum(b['probe_CSV_rows_by_role'].values()) == 13680
    for name in ('original_Policy_CNN_environment_forwards', 'optimizer_constructions_updates_backward_autograd_gradients_trials',
                 'checkpoint_reads_or_writes', 'NPZ_writes', 'extra_source_CSV_HDF5_observation_or_sealed_reads_including_SHA',
                 'retries_overwrites_grid_refinements'):
        assert b[name] == 0


def test_frozen_scope_preserves_four_refs_loss_500_sealing_and_no_global_claim(canonical):
    cfg = canonical
    assert cfg['status'] == 'scientific_freeze_only_execution_binding_required' and cfg['formal_result'] is False
    assert len(cfg['identities']) == 18 and {h['task'] for h in cfg['identities']} == {7}
    assert {h['arm'] for h in cfg['identities']} == {'A_original', 'B_fomaml'}
    assert {h['seed'] for h in cfg['identities']} == {20261106, 20261107, 20261108}
    assert cfg['scope']['original_Policy_updates'] == 500 and cfg['scope']['future_stabilization_steps'] == 5
    assert cfg['scope']['loss_lambda'] == 1 and cfg['scope']['query_validation_heldout_qualification_reserved_final_sealed']
    assert cfg['scope']['SPT_Knowledge_Module_changes'] is False and cfg['decision']['no_global_infeasibility_or_global_optimality_claim']
    assert '0/18' in cfg['scope']['original_main_cue']
    assert cfg['references'] == d.REFERENCES and cfg['conditions']['cached_caps_never_replaced']
    assert cfg['conditions']['raw_sign_tolerance'] == 0 and cfg['conditions']['numerical_absolute_tolerance'] == 1e-10
    assert cfg['grid']['checkpoint_or_best_recipe_selection'] is False
    assert 'does_not_establish' in cfg['decision']['path']


def test_sources_project_only_original_reference_parameters_and_fit_normalization(canonical):
    cfg = canonical
    assert sum(len(family) for family in cfg['source_JSON'].values()) == 6
    assert cfg['source_roles']['latest']['select'] == {'control': 'stepwise_M_nonincrease_guarded_margin_final'}
    assert cfg['source_roles']['stepwise']['select'] == {'margin': 'norm_matched_T_priority_M_margin_final'}
    assert cfg['source_roles']['zero_T']['select'] == {'zero': 'norm_matched_T_priority_projection_final', 'bisector': 'norm_matched_joint_bisector_final'}
    assert len(cfg['projections']['head_schema']) == 8 and len(cfg['projections']['head_members_decoded']) == 6
    assert not {'weights', 'bias'} & set(cfg['projections']['head_members_decoded'])
    assert len(cfg['inputs']['models']) == 6 and len(cfg['inputs']['heads']) == 18
    assert {len(f['fit_demos']) for f in cfg['inputs']['folds']} == {20}
    assert {h['mode'] for h in cfg['inputs']['heads']} == {'harm_protected_affine'}
    for spec in cfg['source_roles'].values():
        assert d.p.pin(spec['science']['path']) == spec['science']
        assert d.p.pin(spec['binding']['path']) == spec['binding']


def test_factory_never_reads_or_hashes_research_payloads(monkeypatch, canonical):
    forbidden = d.forbidden_payload_paths(canonical)
    original_bytes, original_text, original_open = Path.read_bytes, Path.read_text, Path.open
    accesses = []

    def check(path):
        assert str(path) not in forbidden, f'forbidden payload access:{path}'
        assert path.suffix.lower() not in {'.npz', '.npy', '.pt', '.hdf5', '.h5', '.csv'}, f'forbidden payload type:{path}'
        accesses.append(str(path))

    def read_bytes(path):
        check(path)
        return original_bytes(path)

    def read_text(path, *args, **kwargs):
        check(path)
        return original_text(path, *args, **kwargs)

    def open_file(path, *args, **kwargs):
        check(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'read_bytes', read_bytes)
    monkeypatch.setattr(Path, 'read_text', read_text)
    monkeypatch.setattr(Path, 'open', open_file)
    assert d.create() == canonical and accesses
    # The guard also intercepts the repository SHA helper, before any real byte read.
    with pytest.raises(AssertionError, match='forbidden payload access'):
        d.p.pin(d.SOURCE_JSON['latest']['P']['path'])


@pytest.mark.parametrize('field', ['budget', 'scope', 'grid', 'conditions', 'source_JSON', 'decision'])
def test_frozen_configuration_changes_rejected(monkeypatch, tmp_path, canonical, field):
    cfg = copy.deepcopy(canonical)
    cfg[field] = {'mutated': True}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(d, 'CONFIG', path)
    with pytest.raises(ValueError, match='design changed'):
        d.validate()


def test_binding_closure_contains_full_previous929_without_payload_SHA(canonical):
    # No dependency generation until the new science file exists; use a temporary pin.
    from unittest.mock import patch
    expected = d.p.load(d.PARENTS['binding'])['dependencies']
    real_pin = d.p.pin
    synthetic = {'path': str(d.CONFIG), 'sha256': '0' * 64}
    with patch.object(d.p, 'pin', side_effect=lambda file: synthetic if Path(file) == d.CONFIG else real_pin(file)):
        refs_bound = d.freeze_dependencies(canonical)
    assert all(ref in refs_bound for ref in expected)
    assert len({r['path'] for r in refs_bound}) == len(refs_bound)
    assert not {r['path'] for r in refs_bound} & d.forbidden_payload_paths(canonical)


def test_scientific_freeze_cannot_authorize_payload_execution(monkeypatch, tmp_path, canonical):
    monkeypatch.setattr(d, 'validate_freeze', lambda: (canonical, {'dependencies': []}))
    monkeypatch.setattr(d, 'BINDING', tmp_path / 'missing-execution-binding.json')
    with pytest.raises(ValueError, match='science freeze alone denies'):
        d.require_binding()


@pytest.fixture
def human_execution(monkeypatch, tmp_path, canonical):
    science, scientific, contract = tmp_path / 'science.json', tmp_path / 'frozen.json', tmp_path / 'contract.md'
    science.write_text(json.dumps(canonical))
    scientific.write_text('{}')
    contract.write_text('human contract only')
    files = {key: tmp_path / (key + '.py') for key in d.FUTURE_FILES}
    for file in files.values():
        file.write_text('# human stub, not a research runner\n')
    monkeypatch.setattr(d, 'CONFIG', science)
    monkeypatch.setattr(d, 'FREEZE_BINDING', scientific)
    monkeypatch.setattr(d, 'FUTURE_FILES', files)
    monkeypatch.setattr(d, 'FUTURE_CONTRACT', contract)
    monkeypatch.setattr(d, 'validate_freeze', lambda: (canonical, {'dependencies': []}))
    path = tmp_path / 'execution.json'
    monkeypatch.setattr(d, 'BINDING', path)
    record = dict(status='bound_before_payload', before_payload=True, science=d.p.pin(science),
        scientific_binding=d.p.pin(scientific), budget=canonical['budget'], CPU_environment=canonical['execution']['environment'],
        dependencies=[d.p.pin(file) for file in [scientific, *files.values(), contract]],
        full18_synthetic_passed=True, maximum_budget_envelope_passed=True)
    return path, record


def test_complete_human_execution_manifest_passes_without_numerical_input(human_execution):
    path, record = human_execution
    path.write_text(json.dumps(record))
    assert d.require_binding()[1] == record


@pytest.mark.parametrize('key', ['before_payload', 'full18_synthetic_passed', 'maximum_budget_envelope_passed'])
def test_incomplete_execution_gates_fail_closed(human_execution, key):
    path, record = human_execution
    record[key] = False
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        d.require_binding()


@pytest.mark.parametrize('failure', ['payload_in_closure', 'missing_guard', 'duplicate_dependency', 'changed_budget', 'changed_code'])
def test_future_execution_closure_and_budget_fail_closed(human_execution, failure):
    path, record = human_execution
    if failure == 'payload_in_closure':
        record['dependencies'].append(d.SOURCE_JSON['latest']['P'])
    elif failure == 'missing_guard':
        record['dependencies'] = [r for r in record['dependencies'] if Path(r['path']).name != 'guard.py']
    elif failure == 'duplicate_dependency':
        record['dependencies'].append(record['dependencies'][0])
    elif failure == 'changed_budget':
        record['budget'] = dict(record['budget'], core_forwards_total=13681)
    else:
        d.FUTURE_FILES['audit'].write_text('# changed human code\n')
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        d.require_binding()
