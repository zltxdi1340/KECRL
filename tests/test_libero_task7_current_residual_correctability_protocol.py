"""Synthetic protocol/role/budget/cue checks; no real caches or fitting."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import libero_task7_current_residual_correctability_protocol as d

LENGTHS = [155, 125, 124, 191, 136, 138, 133, 130, 139, 145, 140, 164, 124, 138, 120,
           137, 141, 152, 137, 128, 197, 124, 128, 126, 144, 129, 168, 132, 140, 133]


def example():
    pin = lambda name: {'path': '/never/open/' + name, 'sha256': 'a' * 64}
    parents = {name: {'formal_result': False} for name in d.PARENTS}; models = []; captured = []
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            checkpoint = {**pin(f'{seed}-{arm}.pt'), 'weight_hash': 'b' * 64}; archive = pin(f'{seed}-{arm}-features.npz')
            legacy = {'all': {'valid_steps': 4188, 'mse': .04}}
            row = {'task': 7, 'seed': seed, 'arm': arm, 'checkpoint': checkpoint, 'prediction_archive': archive, 'legacy_strata': legacy}
            models.append(row); captured.append({**{k: row[k] for k in ('task', 'seed', 'arm', 'checkpoint')},
                                                'features': archive, 'legacy_prediction_bit_exact': True, 'G500_parity': legacy['all']})
    source = {'shared': pin('shared.npz'), 'models': models, 'action_lengths': LENGTHS.copy(),
              'tags': {'unused': pin('must-not-open.csv')}}
    parents['alignment_science']['inputs'] = source; parents['capture']['models'] = captured
    parents['alignment_result'].update(status='complete_task7_policy_error_focus_pending_audit', science_config=d.PARENTS['alignment_science'],
                                       binding=d.PARENTS['alignment_binding'], inputs=copy.deepcopy(source))
    parents['alignment_audit'].update(status='passed_independent_task7_policy_error_focus_audit', science_config=d.PARENTS['alignment_science'],
                                      binding=d.PARENTS['alignment_binding'], result=d.PARENTS['alignment_result'])
    parents['alignment_completion'].update(status='complete_independently_audited_task7_policy_error_focus', science_config=d.PARENTS['alignment_science'],
                                           binding=d.PARENTS['alignment_binding'], result=d.PARENTS['alignment_result'], audit=d.PARENTS['alignment_audit'])
    cfg = {'kind': 'fixed_current_representation_continuous_component3_correctability', 'version': 1, 'formal_result': False,
           'status': 'scientific_frozen_execution_unbound', 'parents': copy.deepcopy(d.PARENTS),
           'scope': copy.deepcopy(d.SCOPE), 'design': copy.deepcopy(d.DESIGN), 'optimizer': copy.deepcopy(d.OPTIMIZER),
           'source_members': copy.deepcopy(d.MEMBERS), 'head_members': d.HEAD_MEMBERS.copy(),
           'execution': copy.deepcopy(d.EXECUTION), 'budget': d.budget(), 'inputs': d.select_inputs(parents),
           'output': d.OUTPUT, 'artifacts': d.ARTIFACTS}
    return cfg, parents


def measurement_cube():
    return {mode: {phase: {direction: {'component3_mse': value, 'material_sign_error_rate': value, 'all7_mse': value}
                          for direction in d.DIRECTIONS} for phase in d.PHASES}
            for mode, value in [('original', .4), ('bias_only', .2), ('current_affine', .1)]}


def record_cube():
    metrics = measurement_cube()
    return [{'seed': seed, 'arm': arm, **d.comparison_flags(metrics, c, r, numerical_sign_stable=True, other6_bit_exact=True)}
            for seed in d.p.SEEDS for arm in d.p.ARMS for c, r in d.CONTRASTS]


def test_complete_scientific_metadata_and_separated_optimizer_budgets():
    cfg, parents = example(); assert d.validate_design(cfg, parents)
    b = cfg['budget']; assert b['auxiliary_heads'] == 6 * 3 * 2 == 36
    assert b['auxiliary_optimizer_updates'] == b['auxiliary_backward_calls'] == b['auxiliary_train_forward_batches'] == 36 * 500
    assert b['auxiliary_train_row_visits'] == 36 * 500 * 64
    assert b['unique_sampling_tables'] == 3 * 3 and b['sample_RNG_row_draws'] == 9 * 500 * 64
    assert b['source_members_per_stage'] == 5 + 6 * 2 == 17
    assert len(set(d.HEAD_MEMBERS)) == b['members_per_head_NPZ'] == 19
    assert b['total_NPZ_loads_including_new_heads'] == 2 * 7 + 36 == 50
    assert b['total_NPZ_member_decodes_including_new_heads'] == 2 * 17 + 36 * 19 == 718
    assert b['scaler_fit_row_visits'] == 6 * 2 * 4188 and b['scaler_fit_feature_coordinates'] == b['scaler_fit_row_visits'] * 256
    assert b['full_cache_standardization_coordinates'] == 6 * 3 * 4188 * 256
    assert b['main_total_standardization_rows'] == b['main_fit_standardization_rows'] + b['full_cache_standardization_rows']
    assert b['main_total_standardization_coordinates'] == b['main_total_standardization_rows'] * 256
    assert b['snapshot_readout_rows_per_stage'] == 36 * 4 * 4188
    assert b['snapshot_sevenD_prediction_coordinates_per_stage'] == b['snapshot_readout_rows_per_stage'] * 7
    assert b['OOF_all_modes_all_snapshots_logical_rows'] == 6 * 3 * 4 * 4188
    assert b['tables']['curves'] == 6 * 3 * 2 * 4 * 2 * 4 * 4
    assert sum(b['tables'].values()) == 151434
    assert cfg['scope']['new_original_Policy_or_generator_optimizer_updates'] == b['input_tag_CSV_passes'] == 0


def test_whole_demo_threefold_disjointness_and_exact_once_OOF():
    plan = d.folds(LENGTHS); source = [(demo, step) for demo, length in enumerate(LENGTHS) for step in range(1, length)]
    seen = []
    assert [(r['fit_tge1_rows'], r['probe_tge1_rows'], r['probe_tge5_rows']) for r in plan] == [(2782, 1406, 1366), (2817, 1371, 1331), (2777, 1411, 1371)]
    for row in plan:
        assert not set(row['fit_rows']) & set(row['probe_rows'])
        assert sorted(row['fit_rows'] + row['probe_rows']) == list(range(4188))
        assert {source[i][0] for i in row['probe_rows']} == set(row['probe_demos'])
        assert {source[i][0] for i in row['fit_rows']} == set(row['fit_demos'])
        assert set(row['fit_demos']).isdisjoint(row['probe_demos']) and len(row['fit_demos']) == 20
        seen.extend(row['probe_rows'])
    assert sorted(seen) == list(range(4188)) and len(seen) == len(set(seen))


@pytest.mark.parametrize('lengths', [LENGTHS[:-1], [140] * 30, [24] + LENGTHS[1:], [True] + LENGTHS[1:]])
def test_malformed_source_lengths_rejected(lengths):
    with pytest.raises(ValueError, match='lengths'): d.folds(lengths)


def test_primary_positive_and_negative_global_guards_all_pass_for_complete_example():
    for c, r in d.CONTRASTS:
        row = d.comparison_flags(measurement_cube(), c, r, numerical_sign_stable=True, other6_bit_exact=True)
        assert row['cue'] is True and len(row['flags']) == 15 and row['automatic_promotion'] is False
    assert len(d.joint_flags(record_cube())) == 4
    assert all(row['joint_cue'] is True and row['automatic_promotion'] is False for row in d.joint_flags(record_cube()))


@pytest.mark.parametrize('phase,direction,key', [(phase, direction, key) for phase in d.CUE_PHASES
                                                for direction in ('positive', 'negative')
                                                for key in ('component3_mse', 'material_sign_error_rate')])
def test_every_direction_and_phase_guard_prevents_average_compensation(phase, direction, key):
    metrics = measurement_cube(); metrics['current_affine'][phase][direction][key] = .400000000001
    result = d.comparison_flags(metrics, 'current_affine', 'original', numerical_sign_stable=True, other6_bit_exact=True)
    assert result['cue'] is False and result['flags'][phase + '/' + direction + '/' + key] is False


@pytest.mark.parametrize('phase,key', [(phase, key) for phase in ('all_common', 'tail20') for key in ('component3_mse', 'material_sign_error_rate')])
def test_strict_positive_requires_improvement_not_tie(phase, key):
    cube = measurement_cube(); cube['current_affine'][phase]['positive'][key] = cube['original'][phase]['positive'][key]
    assert not d.comparison_flags(cube, 'current_affine', 'original', numerical_sign_stable=True, other6_bit_exact=True)['cue']


def test_negative_and_complement_ties_allowed_no_scientific_epsilon():
    cube = measurement_cube()
    for phase in d.CUE_PHASES:
        for key in ('component3_mse', 'material_sign_error_rate'):
            cube['current_affine'][phase]['negative'][key] = cube['original'][phase]['negative'][key]
            cube['current_affine'][phase]['positive'][key] = cube['original'][phase]['positive'][key] - (0 if phase == 'complement' else 1e-12)
    assert d.comparison_flags(cube, 'current_affine', 'original', numerical_sign_stable=True, other6_bit_exact=True)['cue']
    cube['current_affine']['all_common']['all']['all7_mse'] = .400000000001
    assert not d.comparison_flags(cube, 'current_affine', 'original', numerical_sign_stable=True, other6_bit_exact=True)['cue']


@pytest.mark.parametrize('value', [None, float('nan'), float('inf'), -1., True, 1.01])
def test_invalid_or_empty_primary_value_returns_false_with_reason(value):
    cube = measurement_cube(); cube['current_affine']['tail20']['positive']['material_sign_error_rate'] = value
    result = d.comparison_flags(cube, 'current_affine', 'original', numerical_sign_stable=True, other6_bit_exact=True)
    assert result['cue'] is False and result['missing_or_invalid']


def test_missing_primary_metric_is_false_not_silent_row_omission():
    cube = measurement_cube(); del cube['current_affine']['tail20']['positive']['component3_mse']
    result = d.comparison_flags(cube, 'current_affine', 'original', numerical_sign_stable=True, other6_bit_exact=True)
    assert not result['cue'] and 'tail20/positive/component3_mse' in result['missing_or_invalid']


@pytest.mark.parametrize('sign,invariant', [(False, True), (True, False), (False, False)])
def test_numerical_sign_and_other6_invariance_gate_cannot_be_averaged_away(sign, invariant):
    assert not d.comparison_flags(measurement_cube(), 'current_affine', 'original', numerical_sign_stable=sign, other6_bit_exact=invariant)['cue']


def test_boolean_evidence_and_registered_contrast_required():
    with pytest.raises(ValueError): d.comparison_flags(measurement_cube(), 'original', 'bias_only', numerical_sign_stable=True, other6_bit_exact=True)
    with pytest.raises(ValueError): d.comparison_flags(measurement_cube(), 'current_affine', 'original', numerical_sign_stable=1, other6_bit_exact=True)


def test_current_extra_cue_requires_bias_and_original_and_all3seeds():
    records = record_cube()
    bad = next(row for row in records if row['seed'] == d.p.SEEDS[-1] and row['arm'] == 'B_fomaml' and row['reference'] == 'bias_only')
    bad['flags']['tail20/negative/component3_mse'] = False; bad['cue'] = False
    joined = {(row['arm'], row['cue_type']): row for row in d.joint_flags(records)}
    assert joined['B_fomaml', 'current_affine_extra']['joint_cue'] is False
    assert joined['B_fomaml', 'bias_only']['joint_cue'] is True and joined['A_original', 'current_affine_extra']['joint_cue'] is True
    assert [row['cue'] for row in joined['B_fomaml', 'current_affine_extra']['per_seed']] == [True, True, False]


@pytest.mark.parametrize('problem', ['missing', 'duplicate', 'seed', 'flag', 'promotion'])
def test_incomplete_or_inconsistent_cue_provenance_rejected(problem):
    rows = record_cube()
    if problem == 'missing': rows.pop()
    elif problem == 'duplicate': rows[-1] = rows[0]
    elif problem == 'seed': rows[0]['seed'] = 1
    elif problem == 'flag': rows[0]['cue'] = False
    else: rows[0]['automatic_promotion'] = True
    with pytest.raises(ValueError): d.joint_flags(rows)


@pytest.mark.parametrize('section,key,value', [
    ('scope', 'original_Policy_checkpoint_updates', 1000), ('scope', 'source_meta_query_accessed', True),
    ('scope', 'sealed_roles_accessed', True), ('scope', 'new_original_Policy_or_generator_optimizer_updates', 1),
    ('scope', 'new_original_CNN_Policy_or_environment_calls', 1), ('scope', 'history_NN_tags_or_neighbor_search_used', True),
    ('scope', 'closed_loop_default_stabilization_steps', 0), ('scope', 'SPT_Knowledge_Module_changes', True),
    ('optimizer', 'updates_per_readout', 1000), ('optimizer', 'learning_rate', .003),
    ('optimizer', 'phase_direction_gripper_weights_regularization_or_target_normalization', True),
    ('source_members', 'current', ['features', 'predictions', 'logits']),
    ('budget', 'total_NPZ_loads_including_new_heads', 14), ('budget', 'auxiliary_optimizer_updates', 0),
    ('budget', 'input_tag_CSV_passes', 2), ('budget', 'parameters_by_mode', {'bias_only': 257, 'current_affine': 257}),
    ('design', 'fold_demo_indices', [list(range(9)), list(range(9, 20)), list(range(20, 30))]),
    ('design', 'endpoint', 250), ('design', 'checkpoints', [0, 500]),
    ('design', 'probe_gate', 'probe_during_fit_for_early_stop'), ('design', 'cue_weighting', 'choose_best_weighting'),
    ('execution', 'technical_retries', 1), ('execution', 'readout_reconstruction_absolute_tolerance', .001),
    ('execution', 'working_memory_bytes_max', 1073741824), ('execution', 'integer_ID_tag_and_other6_component_tolerance', 1e-5),
])
def test_scientific_drift_rejected(section, key, value):
    cfg, parents = example(); cfg[section][key] = value
    with pytest.raises(ValueError, match='frozen'): d.validate_design(cfg, parents)


@pytest.mark.parametrize('problem', ['audit_status', 'completion_audit', 'binding', 'science', 'role', 'source', 'endpoint', 'archive', 'bit_parity', 'G500', 'models', 'folds', 'CSV'])
def test_parent_endpoint_and_role_lineage_rejected(problem):
    cfg, parents = example()
    if problem == 'audit_status': parents['alignment_audit']['status'] = 'not_audited'
    elif problem == 'completion_audit': parents['alignment_completion']['audit'] = {}
    elif problem == 'binding': parents['alignment_result']['binding'] = {}
    elif problem == 'science': parents['alignment_result']['science_config'] = {}
    elif problem == 'role': parents['capture']['formal_result'] = True
    elif problem == 'source': parents['alignment_result']['inputs']['shared']['path'] = '/bad.npz'
    elif problem == 'endpoint': cfg['inputs']['models'][0]['checkpoint']['weight_hash'] = 'c' * 64
    elif problem == 'archive': cfg['inputs']['models'][0]['current_archive']['path'] = '/bad.npz'
    elif problem == 'bit_parity': parents['capture']['models'][0]['legacy_prediction_bit_exact'] = False
    elif problem == 'G500': parents['capture']['models'][0]['G500_parity'] = {}
    elif problem == 'models': cfg['inputs']['models'].pop()
    elif problem == 'folds': cfg['inputs']['folds'][0]['fit_rows'].append(cfg['inputs']['folds'][0]['probe_rows'][0])
    else: cfg['inputs']['tags'] = {'forbidden': '/bad.csv'}
    with pytest.raises(ValueError): d.validate_design(cfg, parents)


def metadata_harness(tmp_path, monkeypatch):
    cfg, parents = example(); context = tmp_path / 'metadata.py'; context.write_text('# synthetic context\n')
    protocol = tmp_path / 'protocol.md'; protocol.write_text('synthetic protocol\n')
    cfg['code'] = [d.p.pin(context)]; cfg['protocol'] = d.p.pin(protocol)
    monkeypatch.setattr(d, 'CONTEXT_FILES', [context]); monkeypatch.setitem(d.BINDING_FILES, 'protocol', protocol)
    by_path = {ref['path']: parents[name] for name, ref in d.PARENTS.items()}
    monkeypatch.setattr(d.p, 'load_pin', lambda ref: by_path[ref['path']]); monkeypatch.setattr(d.previous, 'validate', lambda *a, **k: None)
    path = tmp_path / 'config.json'; path.write_text(json.dumps(cfg))
    return cfg, path


def test_default_validation_avoids_source_reads_and_byte_checks_are_explicit(tmp_path, monkeypatch):
    cfg, path = metadata_harness(tmp_path, monkeypatch); native_sha = d.p.sha; source_calls = []
    def sha(p):
        if str(p).startswith('/never/open/'):
            source_calls.append(str(p)); return 'a' * 64
        return native_sha(p)
    monkeypatch.setattr(d.p, 'sha', sha)
    assert d.validate(path) == cfg and not source_calls
    d.validate(path, check_source_byte_hashes=True); assert len(source_calls) == 7
    monkeypatch.setattr(d.p, 'sha', lambda p: '0' * 64 if str(p).startswith('/never/open/') else native_sha(p))
    with pytest.raises(ValueError, match='cached source bytes'): d.validate(path, check_source_byte_hashes=True)


def test_unbound_execution_refuses_and_no_runtime_files_are_required_for_freeze(tmp_path, monkeypatch):
    cfg, path = metadata_harness(tmp_path, monkeypatch)
    assert d.validate(path)['status'] == 'scientific_frozen_execution_unbound'
    with pytest.raises(ValueError, match='execution binding required'): d.require_binding(path, tmp_path / 'missing.json')


def test_missing_or_changed_bound_role_and_budget_fail_closed(tmp_path, monkeypatch):
    cfg, path = metadata_harness(tmp_path, monkeypatch)
    files = {}
    for role in d.BINDING_FILES:
        file = tmp_path / (role + '.txt'); file.write_text(role); files[role] = file
    monkeypatch.setattr(d, 'BINDING_FILES', files); monkeypatch.setattr(d, 'validate', lambda *a, **k: cfg)
    record = {'status': 'bound_before_payload', 'formal_result': False, 'before_research_payload': True,
              'science_config': d.p.pin(path), 'budget': d.budget(), 'files': {key: d.p.pin(file) for key, file in files.items()}}
    binding = tmp_path / 'binding.json'; binding.write_text(json.dumps(record)); assert d.require_binding(path, binding) == record
    changed = copy.deepcopy(record); changed['budget']['total_NPZ_loads_including_new_heads'] = 14; binding.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='budget'): d.require_binding(path, binding)
    binding.write_text(json.dumps(record)); files['training'].write_text('changed')
    with pytest.raises(ValueError, match='code/role'): d.require_binding(path, binding)


def test_metadata_import_has_no_numerical_runtime_or_payload_in_fresh_process():
    command = "from experiments import libero_task7_current_residual_correctability_protocol as d; import sys; assert not {'numpy','torch','h5py'}.intersection(sys.modules); assert d.budget()['auxiliary_optimizer_updates']==18000; assert d.SCOPE['sealed_roles_accessed'] is False; print('stdlib_only_zero_payload')"
    result = subprocess.run([sys.executable, '-c', command], cwd=d.ROOT, env=os.environ.copy(), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=10)
    assert result.returncode == 0 and result.stdout.strip() == 'stdlib_only_zero_payload', result.stdout
