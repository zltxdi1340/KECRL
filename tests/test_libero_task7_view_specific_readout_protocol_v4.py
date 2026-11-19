"""Scientific metadata and invented endpoints; no research numeric payloads."""
from copy import deepcopy
import json
from pathlib import Path
import pytest
import torch
from experiments import libero_task7_view_specific_readout_protocol_v4 as p


def cfg():
    return json.loads(p.CONFIG.read_text())


def endpoints(c, stage, counts):
    used = {(arm, seed): 0 for arm in p.ARMS for seed in p.SEEDS}; rows = []
    for job in p.problems(c, stage):
        key = job['controller'], job['seed']; used[key] += 1
        rows.append(dict(problem_id=job['problem_id'], controller=job['controller'], seed=job['seed'],
                         success=used[key] <= counts[key]))
    return rows


def test_exact_science_scope_and_module_level_attribution():
    c = cfg(); p.validate_scope(c)
    assert c['candidate']['post_coordinate_mlp'] == [512, 608, 128]
    assert c['expected_trainable_parameters'] == dict(flat_zero_xyz=1514120, view_specific_spatial_zero_xyz=1507272)
    assert c['candidate']['per_view_readout_parameters']*2 == c['expected_readout_parameters'][p.ARMS[1]]
    assert not c['attribution']['isolated_weight_unsharing'] and c['attribution']['shared_v3_historical_context_only']
    assert c['readout_seed_offsets'] == dict(agentview=9400001, eye_in_hand=9500001)
    assert not c['scientific_freeze_authorizes_execution'] and c['stage_binding_required']


@pytest.mark.parametrize('path,value', [
    (('candidate', 'post_coordinate_mlp'), [512, 1408, 128]),
    (('candidate', 'shared_readout_parameters'), True), (('candidate', 'final_activation'), 'ReLU'),
    (('readout_seed_offsets', 'eye_in_hand'), 9400001), (('entropy', 'temperature'), .2),
    (('entropy', 'max_retained_entropy_scalars'), 61440), (('entropy', 'loss_weight'), .01),
    (('entropy', 'success_gate'), True), (('continuation', 'handoff_minimum_net_gain'), 3),
    (('continuation', 'fixed_absolute_candidate_minimum'), True),
    (('attribution', 'isolated_weight_unsharing'), True), (('attribution', 'qualified_KECRL_Module'), True),
    (('source_meta_query_allowed',), True), (('xyz_mode',), 'real_xyz'), (('updates_per_policy',), 501),
    (('stabilization_steps',), 4), (('loss',), 'weighted_mse'), (('formal_result',), 0),
    (('scientific_freeze_authorizes_execution',), True), (('entropy', 'temperature'), True),
])
def test_frozen_scientific_mutations_rejected(path, value):
    c = cfg(); node = c
    for key in path[:-1]: node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError): p.validate_scope(c)


def test_per_view_compute_and_unchanged_physics_envelopes():
    c = cfg(); t = p.train_caps()
    assert t['train_forward'] == t['backward'] == t['optimizer'] == 3000
    assert t['flat_readout'] == t['spatial_readout'] == 3006
    for view in p.VIEW_SEED_OFFSETS:
        assert t['spatial_readout_'+view] == 1503
        assert t['spatial_image_rows_'+view] == 120240
        assert t['spatial_heatmaps_'+view] == 30781440
    for stage, episodes, native in [('handoff', 44, 15980), ('full', 120, 36600)]:
        b = p.evaluation_caps(c, stage); jobs = p.problems(c, stage)
        assert len(jobs) == episodes and b['native_steps'] == native
        assert b['native_steps'] == 300*len(jobs)+5*b['resets']
        for total in ['spatial_readout', 'spatial_image_rows', 'spatial_heatmaps',
                      'spatial_probability_values', 'spatial_coordinate_values', 'spatial_entropy_values']:
            assert b[total] == sum(b[total+'_'+v] for v in p.VIEW_SEED_OFFSETS)
        assert b['eef_payload_reads'] == b['private_object_state_reads'] == b['optimizer_updates'] == 0


@pytest.mark.parametrize('control,candidate,regression,passed', [
    (0, 4, False, True), (8, 12, False, True), (9, 12, False, False), (9, 13, False, True),
    (8, 11, False, False), (8, 12, True, False),
])
def test_continue_gate_is_current_control_difference_not_fixed12(control, candidate, regression, passed):
    controls = [max(0, control-4), min(control, 4), 0]
    candidates = [candidate-3, 3, 0] if regression else [candidate-4, 4, 0]
    if control == 0: candidates = [candidate, 0, 0]
    counts = {(arm, s): values[i] for arm, values in zip(p.ARMS, [controls, candidates]) for i, s in enumerate(p.SEEDS)}
    result = p.decision(endpoints(cfg(), 'handoff', counts), 'handoff')
    assert result['passed'] is passed and result['required_net'] == 4


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'unpaired', 'integer_success', 'unknown_arm', 'wrong_seed'])
def test_pairing_and_endpoint_types_fail_closed(fault):
    counts = {(arm, seed): 0 for arm in p.ARMS for seed in p.SEEDS}
    rows = endpoints(cfg(), 'handoff', counts)
    if fault == 'missing': rows.pop()
    if fault == 'duplicate': rows[-1] = deepcopy(rows[0])
    if fault == 'unpaired': rows[-1]['problem_id'] = 'unpaired-'+p.ARMS[1]
    if fault == 'integer_success': rows[-1]['success'] = 1
    if fault == 'unknown_arm': rows[-1]['controller'] = 'shared_v3'
    if fault == 'wrong_seed': rows[-1]['seed'] = 5
    with pytest.raises(ValueError): p.decision(rows, 'handoff')


@pytest.mark.parametrize('seed', p.SEEDS)
def test_all500_batches_match_and_preserve_rng(seed):
    windows = [(i//15, 1+10*(i % 15)) for i in range(432)]
    before = torch.get_rng_state().clone()
    left, right = [p.sampling_plan(seed, windows) for arm in p.ARMS]
    assert len(left) == 500 and left == right and all(len(batch) == 8 for batch in left)
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('net,regression,passed', [(5, False, False), (6, False, True), (6, True, False)])
def test_conditional_full_stage_gate_stays_six_of60(net, regression, passed):
    counts = {(arm, seed): 0 for arm in p.ARMS for seed in p.SEEDS}
    counts[p.ARMS[0], p.SEEDS[1]] = int(regression)
    counts[p.ARMS[1], p.SEEDS[0]] = net+int(regression)
    result = p.decision(endpoints(cfg(), 'full', counts), 'full')
    assert result['passed'] is passed and result['required_net'] == 6


@pytest.mark.parametrize('stage', ['train', 'handoff', 'full'])
def test_scientific_freeze_never_authorizes_execution(stage, monkeypatch):
    monkeypatch.setattr(Path, 'read_bytes', lambda *_: pytest.fail('payload read before admission'))
    with pytest.raises(PermissionError, match='stage-specific execution binding required'):
        p.require_execution_binding(stage=stage, user_approved=True, binding='invented')


@pytest.mark.parametrize('fault', ['none', 'dependency', 'status', 'science', 'payload_dependency', 'audit_identity'])
def test_invented_parent_chain_integrity(tmp_path, monkeypatch, fault):
    monkeypatch.setattr(p, 'ROOT', tmp_path)
    def save(name, value):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value)); return p.pin(path)
    parents = {key: save(key+'.json', {}) for key in p.PARENT_KEYS}
    proof = save('result-proof.json', {})
    audit_science = parents['v3_science'] if fault != 'audit_identity' else {**parents['v3_science'], 'sha256': '0'*64}
    audit = save('v3-audit.json', dict(status='passed_independent_postgrasp_handoff_audit', stage='handoff',
        science=audit_science, binding=parents['v3_handoff_binding'], result=proof))
    parents['v3_completion'] = save('v3-complete.json', dict(status='complete_audited_postgrasp_handoff',
        result=proof, audit=audit))
    parents['geometry_completion'] = save('geo-complete.json', dict(status='complete_audited_targeted_seed07_geometry',
        science=parents['geometry_science'], binding=parents['geometry_binding'], audit=parents['geometry_audit']))
    monkeypatch.setattr(p.parent, 'FREEZE', Path(parents['v3_scientific_freeze']['path']))
    base = json.loads(p.parent.CONFIG.read_text()); c = p.make_science(base, parents)
    monkeypatch.setattr(p.prior_execution, 'preflight', lambda *args: (base, dict(code_dependencies=[])))
    monkeypatch.setattr(p.geometry, 'preflight', lambda *args: ({}, dict(code_dependencies=[])))
    dependencies = []
    for relative in p.CODE:
        path = tmp_path/relative; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('# invented component\n'); dependencies.append(p.pin(path))
    science = save('science.json', c)
    b = dict(status='scientific_frozen_execution_not_bound', science=science, dependencies=dependencies, validation=[])
    if fault == 'dependency': Path(dependencies[0]['path']).write_text('# changed invented component')
    if fault == 'status': b['status'] = 'execution_bound'
    if fault == 'science': b['science'] = {**science, 'sha256': '0'*64}
    if fault == 'payload_dependency': b['dependencies'].append(dict(path=str(tmp_path/'sealed.npz'), sha256='0'*64))
    binding = save('freeze.json', b)
    if fault == 'none': assert p.validate_freeze(science['path'], binding['path'])[0] == c
    else:
        with pytest.raises(ValueError): p.validate_freeze(science['path'], binding['path'])
