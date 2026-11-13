"""Synthetic-only sign/magnitude and metadata checks; no real CSV rows or arrays."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_residual_sign_transition_protocol as d

LENGTHS = [155, 125, 124, 191, 136, 138, 133, 130, 139, 145, 140, 164, 124, 138, 120,
           137, 141, 152, 137, 128, 197, 124, 128, 126, 144, 129, 168, 132, 140, 133]


def example():
    parents = {key: {'formal_result': False} for key in d.PARENTS}
    science = parents['science']
    science.update(scope={'source_role': 'meta_support', 'source_meta_query_accessed': False,
                          'sealed_roles_accessed': False, 'encoder_has_seen_all30_support_demos': True,
                          'original_Policy_checkpoint_updates': 500}, design={'endpoint': 500})
    science['inputs'] = {'action_lengths': LENGTHS.copy(), 'shared': {'path': '/never/open/shared.npz'},
                         'models': [{'task': 7, 'seed': seed, 'arm': arm,
                                     'checkpoint': {'path': f'/never/open/{seed}-{arm}.pt', 'sha256': 'a' * 64}}
                                    for seed in d.p.SEEDS for arm in d.p.ARMS]}
    tables = {name: {'path': str(Path(d.previous.ARTIFACTS) / filename), 'sha256': 'b' * 64,
                     'rows': rows, 'columns': columns + ['unprojected_synthetic_column']}
              for name, filename, rows, columns in [
                  ('oof-frames', 'oof-frames.csv', 75384, d.FRAME_COLUMNS.copy()),
                  ('curves', 'curves.csv', 4608, d.CURVE_ID_COLUMNS + d.CURVE_METRICS)]}
    for name in ('result', 'audit', 'completion'):
        parents[name].update(science_config=copy.deepcopy(d.PARENTS['science']), binding=copy.deepcopy(d.PARENTS['binding']))
    parents['result'].update(status='complete_task7_current_residual_pending_audit', tables=tables,
                             inputs=copy.deepcopy(science['inputs']), G_failed_H_unrun_unchanged=True,
                             SPT_Knowledge_Module_changes=False)
    parents['audit'].update(status='passed_independent_task7_current_residual_audit', result=copy.deepcopy(d.PARENTS['result']),
                            model_flags=[{'cue': False, 'flags': {'numerical_sign_stable': True}} for _ in range(18)],
                            arm_flags=[{'joint_cue': False, 'automatic_promotion': False} for _ in range(4)])
    parents['completion'].update(status='complete_independently_audited_task7_current_residual',
                                 result=copy.deepcopy(d.PARENTS['result']), audit=copy.deepcopy(d.PARENTS['audit']),
                                 arm_flags=copy.deepcopy(parents['audit']['arm_flags']))
    cfg = {'kind': 'task7_cached_residual_action_sign_transition_readonly', 'version': 1, 'formal_result': False,
           'status': 'scientific_frozen_execution_unbound', 'parents': copy.deepcopy(d.PARENTS),
           'scope': copy.deepcopy(d.SCOPE), 'design': copy.deepcopy(d.DESIGN), 'execution': copy.deepcopy(d.EXECUTION),
           'budget': d.budget(), 'frame_columns': d.FRAME_COLUMNS.copy(), 'curve_ID_columns': d.CURVE_ID_COLUMNS.copy(),
           'curve_metric_columns': d.CURVE_METRICS.copy(), 'inputs': d.select_inputs(parents),
           'output': d.OUTPUT, 'artifacts': d.ARTIFACTS}
    return cfg, parents


def test_exact_csv_only_budgets_and_output_cardinality():
    cfg, parents = example()
    assert d.validate_design(cfg, parents)
    b = d.budget(); s = b['per_stage']
    assert len(d.FRAME_COLUMNS) == len(set(d.FRAME_COLUMNS)) == 17
    assert len(d.CURVE_ID_COLUMNS) == 9 and len(d.CURVE_METRICS) == 14
    assert s['frame_projected_cells'] == 6 * 3 * 4188 * 17
    assert s['curve_ID_projected_cells'] == 4608 * 9
    assert s['selected_curve_metric_cells'] == 6 * 3 * 2 * 4 * 4 * 14
    assert s['projected_cells'] == 1281528 + 41472 + 8064 == 1331064
    assert b['total_CSV_data_passes'] == 4 and b['total_CSV_lexical_rows'] == 2 * (75384 + 4608)
    assert b['total_projected_cells'] == 2 * s['projected_cells']
    assert b['pair_frame_rows'] == 6 * 3 * 3598 and b['cross_seed_frame_rows'] == 2 * 3 * 3598
    assert sum(b['tables'].values()) == 115728 and b['seed_summary_JSON_rows'] == 72
    assert b['NPZ_NPY_checkpoint_feature_or_other_array_reads_including_hashes'] == 0
    assert b['new_training_or_forward_environment_neighbor_or_sealed_calls'] == 0
    assert set(cfg['inputs']) == {'tables', 'action_lengths', 'endpoints'}
    assert set(cfg['inputs']['tables']) == {'frames', 'curves'}
    assert 'shared' not in cfg['inputs'] and len(cfg['inputs']['endpoints']) == 6


def test_source_id_layout_reconstructs_causal_steps_whole_demos_and_tail():
    rows = d.source_layout(LENGTHS)
    assert [r['source_row'] for r in rows] == list(range(4188))
    assert sum(r['action_step'] >= 5 for r in rows) == 4068
    assert sum(r['tail20'] for r in rows) == 600
    for demo, length in enumerate(LENGTHS):
        current = [r for r in rows if r['demo'] == demo]
        assert [r['action_step'] for r in current] == list(range(1, length))
        assert {r['probe_fold'] for r in current} == {demo // 10}
        assert sum(r['tail20'] for r in current) == 20
        assert current[-20]['action_step'] == length - 20 and not current[-21]['tail20']


@pytest.mark.parametrize('lengths', [LENGTHS[:-1], [140] * 30, [True] + LENGTHS[1:], [24] + LENGTHS[1:]])
def test_bad_source_lengths_rejected(lengths):
    with pytest.raises(ValueError, match='lengths'): d.source_layout(lengths)


@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('reference,candidate,cell', [(.2, .1, 'stays_correct'), (-.2, .1, 'corrected'),
                                                    (.2, -.1, 'new_error'), (-.2, -.1, 'stays_wrong'),
                                                    (.2, 0., 'new_error'), (0., .1, 'corrected'),
                                                    (0., 0., 'stays_wrong')])
def test_sign_four_cells_zero_and_error_count_conservation(sign, reference, candidate, cell):
    r = d.pair_quantities(sign * .1, sign * reference, sign * candidate, 0., sign * (candidate - reference))
    assert r['transition'] == cell
    assert int(r['candidate_error']) - int(r['reference_error']) == int(cell == 'new_error') - int(cell == 'corrected')
    assert r['reference_zero'] == (reference == 0) and r['candidate_zero'] == (candidate == 0)
    assert r['delta_sse'] == pytest.approx((candidate - .1) ** 2 - (reference - .1) ** 2)
    if cell == 'new_error': assert r['signed_reference_margin'] > 0 and r['signed_candidate_margin'] <= 0
    if cell == 'corrected': assert r['signed_reference_margin'] <= 0 and r['signed_candidate_margin'] > 0


@pytest.mark.parametrize('value,band', [(0., 'zero'), (-0., 'zero'), (1e-20, 'nonzero_abs_lt.025'),
                                      (-.0249999, 'nonzero_abs_lt.025'), (.025, 'abs_ge.025'),
                                      (-.025, 'abs_ge.025'), (1., 'abs_ge.025')])
def test_output_bands_have_exact_boundaries_no_epsilon(value, band):
    assert d.output_band(value) == band


def test_zero_denominator_is_null_and_ratio_ge_one_does_not_imply_harm():
    zero = d.pair_quantities(.1, 0., .2, 0., .2)
    assert zero['effective_shift_over_abs_reference'] is zero['raw_shift_over_abs_reference'] is None
    good = d.pair_quantities(.1, .01, .1, 0., .09)
    assert good['effective_shift_over_abs_reference'] > 1 and good['transition'] == 'stays_correct'
    assert good['delta_sse'] < 0
    harmful = d.pair_quantities(-.1, -.01, .01, 0., .02)
    assert harmful['transition'] == 'new_error' and harmful['signed_effective_shift'] < 0


def test_sign_and_continuous_error_can_move_in_opposite_directions():
    new_wrong_but_lower_mse = d.pair_quantities(.03, 1., -.01, 0., -1.01)
    corrected_but_higher_mse = d.pair_quantities(.03, -.01, 1., 0., 1.01)
    assert new_wrong_but_lower_mse['transition'] == 'new_error' and new_wrong_but_lower_mse['delta_sse'] < 0
    assert corrected_but_higher_mse['transition'] == 'corrected' and corrected_but_higher_mse['delta_sse'] > 0


@pytest.mark.parametrize('target,pred,category', [(.1, -.1, 'wrong_sign'), (-.1, .1, 'wrong_sign'),
                                                (.1, 0., 'prediction_zero'), (.2, .1, 'same_sign_under'),
                                                (-.1, -.2, 'same_sign_over'), (.1, .1, 'exact_match'),
                                                (0., 0., 'exact_match'), (0., .1, 'zero_target_nonzero')])
def test_strict_six_category_partition(target, pred, category):
    assert d.error_category(target, pred) == category


def test_saved_float32_reconstruction_and_rounding_gap_without_clipping():
    target, original = d.float32(.3), d.float32(.2)
    residual = .01; prediction = d.float32(original + residual)
    assert d.validate_action_scalar(target, prediction, residual, original, prediction - target, False, 'current_affine') == 'same_sign_under'
    r = d.pair_quantities(target, original, prediction, 0., residual)
    assert r['clip_or_rounding_gap'] != 0  # Float32 storage rounding, although no clamp.
    assert d.validate_action_scalar(target, original, 0., original, original - target, False, 'original') == 'same_sign_under'
    assert d.validate_action_scalar(target, 1., .9, original, 1. - target, True, 'current_affine') == 'same_sign_over'
    exact_boundary = .5
    assert d.validate_action_scalar(.25, 1., .5, exact_boundary, .75, False, 'bias_only') == 'same_sign_over'


@pytest.mark.parametrize('field,value', [('target', .3), ('prediction', .21), ('saved_error', .5),
                                        ('clipped', True), ('clipped', 0), ('mode', 'unlisted'),
                                        ('raw_residual', float('nan')), ('original', 1.1)])
def test_saved_scalar_mismatches_rejected(field, value):
    target, original = d.float32(.3), d.float32(.2)
    pred = d.float32(original + .01)
    args = dict(target=target, prediction=pred, raw_residual=.01, original=original,
                saved_error=pred-target, clipped=False, mode='current_affine')
    args[field] = value
    with pytest.raises(ValueError): d.validate_action_scalar(**args)


def test_original_cannot_hide_residual_change():
    with pytest.raises(ValueError, match='original identity'):
        d.validate_action_scalar(.5, .5, .25, .25, 0., False, 'original')


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), '0.1', None])
def test_non_numeric_or_nonfinite_scalars_rejected(value):
    with pytest.raises(ValueError): d.float32(value)
    with pytest.raises(ValueError): d.pair_quantities(.1, value, .1, 0., 0.)


@pytest.mark.parametrize('target', [0., .024999999, -.024999999, 1.1])
def test_nonmaterial_or_out_of_range_targets_rejected(target):
    with pytest.raises(ValueError): d.pair_quantities(target, .1, .1, 0., 0.)


@pytest.mark.parametrize('update', [0, 100, 250, 500])
@pytest.mark.parametrize('role', ['fit', 'probe'])
def test_only_final500_probe_metrics_are_authorized(update, role):
    assert d.curve_metric_allowed(update, role) is (update == 500 and role == 'probe')


@pytest.mark.parametrize('update,role', [(True, 'probe'), (500., 'probe'), (501, 'probe'), (500, 'query')])
def test_invalid_curve_roles_or_updates_rejected(update, role):
    with pytest.raises(ValueError): d.curve_metric_allowed(update, role)


@pytest.mark.parametrize('key,value', [('formal_result', True), ('version', True), ('status', 'executed'),
                                     ('output', '/tmp/other.json'), ('artifacts', '/tmp/other-artifacts')])
def test_scientific_identity_drift_rejected(key, value):
    cfg, parents = example(); cfg[key] = value
    with pytest.raises(ValueError): d.validate_design(cfg, parents)


@pytest.mark.parametrize('section,key,value', [('scope', 'source_meta_query_accessed', True),
                                            ('scope', 'readout_endpoint_updates', 250),
                                            ('scope', 'new_training_or_optimizer_updates', 1),
                                            ('scope', 'sealed_roles_accessed', True),
                                            ('design', 'comparisons_candidate_reference', [['current_affine', 'original']]),
                                            ('execution', 'technical_retries', 1),
                                            ('budget', 'total_CSV_data_passes', 5)])
def test_role_budget_comparison_or_retry_drift_rejected(section, key, value):
    cfg, parents = example(); cfg[section][key] = value
    with pytest.raises(ValueError): d.validate_design(cfg, parents)


@pytest.mark.parametrize('case', ['audit_status', 'parent_hash', 'model_cue', 'numeric_unstable', 'arm_cue',
                                 'automatic_promotion', 'done_cue', 'sealed', 'query', 'endpoint', 'G',
                                 'Knowledge', 'formal', 'CSV_path', 'CSV_rows', 'CSV_columns', 'CSV_duplicate_columns',
                                 'endpoint_missing', 'endpoint_duplicate', 'extra_array_input'])
def test_bad_provenance_cues_inputs_or_roles_rejected(case):
    cfg, parents = example()
    if case == 'audit_status': parents['audit']['status'] = 'pending'
    elif case == 'parent_hash': parents['completion']['audit']['sha256'] = 'c' * 64
    elif case == 'model_cue': parents['audit']['model_flags'][0]['cue'] = True
    elif case == 'numeric_unstable': parents['audit']['model_flags'][0]['flags']['numerical_sign_stable'] = False
    elif case == 'arm_cue': parents['audit']['arm_flags'][0]['joint_cue'] = True
    elif case == 'automatic_promotion': parents['audit']['arm_flags'][0]['automatic_promotion'] = True
    elif case == 'done_cue': parents['completion']['arm_flags'][0]['joint_cue'] = True
    elif case == 'sealed': parents['science']['scope']['sealed_roles_accessed'] = True
    elif case == 'query': parents['science']['scope']['source_meta_query_accessed'] = True
    elif case == 'endpoint': parents['science']['design']['endpoint'] = 250
    elif case == 'G': parents['result']['G_failed_H_unrun_unchanged'] = False
    elif case == 'Knowledge': parents['result']['SPT_Knowledge_Module_changes'] = True
    elif case == 'formal': parents['binding']['formal_result'] = True
    elif case == 'CSV_path': parents['result']['tables']['curves']['path'] = '/never/open/old.csv'
    elif case == 'CSV_rows': parents['result']['tables']['oof-frames']['rows'] -= 1
    elif case == 'CSV_columns': parents['result']['tables']['oof-frames']['columns'].remove('target_2')
    elif case == 'CSV_duplicate_columns': parents['result']['tables']['curves']['columns'].append('update')
    elif case == 'endpoint_missing': parents['science']['inputs']['models'].pop()
    elif case == 'endpoint_duplicate': parents['science']['inputs']['models'][-1] = copy.deepcopy(parents['science']['inputs']['models'][0])
    elif case == 'extra_array_input': cfg['inputs']['shared'] = {'path': '/never/open/shared.npz'}
    with pytest.raises(ValueError): d.validate_design(cfg, parents)


def local_metadata(tmp_path, monkeypatch):
    cfg, parents = example()
    files = [tmp_path / f'context{i}.py' for i in range(len(d.CONTEXT_FILES))]
    plan = tmp_path / 'protocol.md'
    for path in files + [plan]: path.write_text('synthetic metadata only\n')
    cfg.update(code=[d.p.pin(path) for path in files], protocol=d.p.pin(plan))
    monkeypatch.setattr(d, 'CONTEXT_FILES', files)
    monkeypatch.setitem(d.BINDING_FILES, 'protocol', plan)
    opened = []
    def load(ref):
        key = next(k for k, r in d.PARENTS.items() if r == ref)
        opened.append(key)
        return copy.deepcopy(parents[key])
    monkeypatch.setattr(d.p, 'load_pin', load)
    path = tmp_path / 'science.json'; path.write_text(json.dumps(cfg))
    return path, cfg, opened


def test_metadata_validation_does_not_recurse_or_read_csv_or_arrays(tmp_path, monkeypatch):
    path, cfg, opened = local_metadata(tmp_path, monkeypatch)
    monkeypatch.setattr(d.previous, 'validate', lambda *a, **kw: pytest.fail('parent recursive validation forbidden'))
    original_sha = d.p.sha; seen = []
    def checked_sha(value):
        item = Path(value); seen.append(item)
        assert item.parent == tmp_path and item.suffix in ('.py', '.md')
        return original_sha(item)
    monkeypatch.setattr(d.p, 'sha', checked_sha)
    assert d.validate(path) == cfg and opened == list(d.PARENTS)
    assert len(seen) == len(d.CONTEXT_FILES) + 1


def test_optional_csv_byte_checks_only_two_approved_paths(tmp_path, monkeypatch):
    path, cfg, _ = local_metadata(tmp_path, monkeypatch)
    approved = {ref['path']: ref['sha256'] for ref in cfg['inputs']['tables'].values()}
    original_sha = d.p.sha; seen = []
    def checked_sha(value):
        item = Path(value)
        assert item.suffix not in ('.npz', '.npy', '.pt', '.pth', '.hdf5')
        if item.suffix == '.csv':
            seen.append(str(item)); return approved[str(item)]  # No fixture data reader either.
        return original_sha(item)
    monkeypatch.setattr(d.p, 'sha', checked_sha)
    assert d.validate(path, check_csv_byte_hashes=True) == cfg
    assert set(seen) == set(approved) and len(seen) == 2
    monkeypatch.setattr(d.p, 'sha', lambda value: 'wrong' if str(value) in approved else original_sha(value))
    with pytest.raises(ValueError, match='CSV bytes'): d.validate(path, check_csv_byte_hashes=True)


def test_execution_requires_separate_pre_payload_binding(tmp_path, monkeypatch):
    cfg, _ = example(); path = tmp_path / 'science.json'; path.write_text(json.dumps(cfg))
    monkeypatch.setattr(d, 'validate', lambda path: copy.deepcopy(cfg))
    with pytest.raises(ValueError, match='execution binding required'):
        d.require_binding(path, tmp_path / 'missing.json')
    files = {role: tmp_path / (role + '.py') for role in d.BINDING_FILES}
    for file in files.values(): file.write_text('synthetic code\n')
    monkeypatch.setattr(d, 'BINDING_FILES', files)
    binding = {'status': 'bound_before_payload', 'formal_result': False, 'before_research_payload': True,
               'science_config': d.p.pin(path), 'budget': d.budget(), 'files': {k: d.p.pin(v) for k, v in files.items()}}
    target = tmp_path / 'binding.json'; target.write_text(json.dumps(binding))
    assert d.require_binding(path, target) == binding
    binding['budget']['total_CSV_data_passes'] = 6; target.write_text(json.dumps(binding))
    with pytest.raises(ValueError, match='execution identity'): d.require_binding(path, target)
    binding['budget'] = d.budget(); target.write_text(json.dumps(binding))
    files['guard'].write_text('changed\n')
    with pytest.raises(ValueError, match='execution role/code'): d.require_binding(path, target)


def test_protocol_import_is_stdlib_only_in_fresh_process():
    script = ('import sys; from experiments import libero_task7_residual_sign_transition_protocol; '
              'assert not set(sys.modules).intersection({"numpy","torch","h5py","csv","PIL","libero","mujoco"})')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'}
    result = subprocess.run([sys.executable, '-c', script], cwd=d.ROOT, env=env, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
