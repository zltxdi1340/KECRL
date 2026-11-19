"""Bounded source-development reset probe, separate from failed video v1."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.libero_gripper_approach_video_protocol import ROOT, RUNTIME, pin, require, write_new
from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import observation_digest

INDICES = list(range(8))
BRANCHES = [('direct', [7]), ('sequential', INDICES)]
SHAPES = dict(agentview_rgb=(128, 128, 3), eye_in_hand_rgb=(128, 128, 3),
              joint_states=(7,), gripper_states=(2,))
CAPS = dict(adapters=2, resets=9, native_steps=45, zero_steps=45,
            numeric_rows=8, initial_row_loads=1, observations=9, rgb_frames=18,
            raw_rgb_bytes=9*2*128*128*3, source_episode_parses=8,
            source_episode_bytes=8*1024**2, saved_actions=0, policy_forwards=0,
            backwards=0, optimizer_updates=0, model_creations=0,
            checkpoint_reads=0, hdf5_reads=0, extra_render_calls=0)


def validate_scope(cfg):
    expected = dict(kind='libero_source_reset_reproducibility_probe', version=1,
        formal_result=False, task=4, training_seed=20261106, arm='frozen_resnet18',
        indices=INDICES, target_index=7, role='candidate_nonregression',
        branches=[dict(name=name, indices=indices) for name, indices in BRANCHES],
        environment_seed=20261022, stabilization_steps=5, image_size=128,
        adapter_horizon=300, wall_seconds_max=180, caps=CAPS,
        runtime_environment=RUNTIME, no_automatic_retry=True,
        previous_episode_actions_replayed=False, mismatch_is_diagnostic_data=True,
        new_success_rate_estimate=False)
    for key, value in expected.items():
        require(json.dumps(cfg.get(key), sort_keys=True) == json.dumps(value, sort_keys=True),
                'reset probe scope differs: '+key)
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results'), 'persistent output required')
    refs = cfg['references']
    require(len(refs) == 8 and [r['initial_state_index'] for r in refs] == INDICES,
            'exact eight source reset references required')
    for item in refs:
        require(item['task_id'] == 4 and item['training_seed'] == 20261106
            and item['arm'] == 'frozen_resnet18' and item['initial_state_role'] == cfg['role']
            and Path(item['path']).resolve().parent == Path(cfg['source_archive'])/'episodes',
            'source reset reference identity differs')


def preflight(config_path, binding_path):
    cfg = json.loads(Path(config_path).read_text()); validate_scope(cfg)
    bound = json.loads(Path(binding_path).read_text())
    require(bound['status'] == 'bound_before_reset_payload' and bound['science'] == pin(config_path),
            'new reset execution binding required')
    for item in bound['code_dependencies'] + bound['validation']:
        require(pin(item['path']) == item, 'reset dependency changed: '+item['path'])
    require(pin(cfg['parent_science']['path']) == cfg['parent_science'], 'parent video science changed')
    require(pin(cfg['parent_freeze']['path']) == cfg['parent_freeze'], 'parent video freeze changed')
    parent = json.loads(Path(cfg['parent_freeze']['path']).read_text())
    parent_science = json.loads(Path(cfg['parent_science']['path']).read_text())
    require(cfg['initial_bank'] == parent_science['initial_banks']['4']
        and cfg['versions'] == parent_science['versions']
        and cfg['libero_revision'] == parent_science['libero_revision']
        and cfg['source_evaluation'] == parent_science['source_provenance']['evaluation']
        and cfg['source_archive'] == parent_science['source_archive'], 'reset parent provenance differs')
    closure = {r['path']: r['sha256'] for r in bound['code_dependencies']}
    require({r['path']: r['sha256'] for r in parent['metadata_dependencies']}.items() <= closure.items(),
            'parent code closure missing')
    required = [ROOT/'experiments'/name for name in (
        'libero_reset_reproducibility_protocol.py', 'run_libero_reset_reproducibility_diagnostic.py',
        'audit_libero_reset_reproducibility_diagnostic.py')]
    required += [ROOT/'tests/test_libero_reset_reproducibility_diagnostic.py',
                 ROOT/'docs/libero-reset-reproducibility-protocol-2026-11-14.md']
    require(all(closure.get(str(path)) == pin(path)['sha256'] for path in required),
            'reset implementation closure missing')
    item = bound['source_initial_authorization']
    require(pin(item['path']) == item, 'new reset authorization changed')
    grant = json.loads(Path(item['path']).read_text())
    require(grant['user_approved'] is True and grant['science'] == pin(config_path)
        and grant['tasks'] == [4] and grant['indices'] == INDICES
        and grant['role'] == cfg['role'] and grant['native_steps_max'] == 45
        and grant['resets_max'] == 9 and grant['numeric_rows_max'] == 8
        and grant['policy_forwards_max'] == grant['optimizer_updates_max'] == 0
        and grant['previous_video_permission_reused'] is False
        and grant['model_reads_allowed'] is False and grant['hdf5_allowed'] is False
        and grant['source_meta_query_allowed'] is False and grant['extra_cameras_allowed'] is False
        and grant['sealed_roles_allowed'] == [],
        'new reset authorization scope differs')
    require(pin(cfg['source_evaluation']['path']) == cfg['source_evaluation'], 'source evaluation changed')
    evaluation = json.loads(Path(cfg['source_evaluation']['path']).read_text())
    refs = sorted((r for r in evaluation['rollout']['episode_files']
        if r['task_id'] == 4 and r['training_seed'] == 20261106 and r['arm'] == 'frozen_resnet18'
        and r['initial_state_index'] in INDICES), key=lambda r: r['initial_state_index'])
    require(refs == cfg['references'], 'source reference metadata differs')
    return cfg, bound


def validate_initial_access(access, cfg):
    item = cfg['initial_bank']
    require(access['numeric_rows_decoded'] == INDICES
        and access['global_to_local_index'] == {str(i): i for i in INDICES}
        and access['source_path'] == item['path'] and access['container_sha256'] == item['sha256']
        and access['pickle_header_sha256'] == item['pickle_header_sha256']
        and access['decoded_selected_bytes'] == 8*92*8
        and access['skipped_rows_numerically_decoded'] == []
        and access['skipped_rows_buffered'] is False
        and access['used_pickle_or_torch_deserialization'] is False
        and set(access['selected_row_sha256']) == {str(i) for i in INDICES},
        'reset initial access differs')


def summarize(observation, source):
    """Exact gates stay exact; discrepancy magnitudes are descriptive only."""
    digest = observation_digest(observation)
    fields = {}; errors = {}; matches = {}
    for key in SHAPES:
        array = np.ascontiguousarray(observation[key])
        # A one-field digest with the same interface descriptor is independently auditable.
        import hashlib
        h = hashlib.sha256(); h.update(key.encode()); h.update(str(array.dtype).encode())
        h.update(np.asarray(array.shape, dtype=np.int64).tobytes()); h.update(array.tobytes())
        fields[key] = h.hexdigest()
    for actual, original in [('joint_states', 'joint_state'), ('gripper_states', 'gripper_state')]:
        ref = np.asarray(source['stepwise_public_state_trace'][0][original], np.float64)
        require(ref.shape == SHAPES[actual] and np.isfinite(ref).all(), 'invalid source reset public state')
        values = observation[actual].astype(np.float64)
        matches[actual] = bool(np.array_equal(values, ref))
        errors[actual] = float(np.max(np.abs(values-ref)))
    exact = digest == source['reset_observation_sha256']
    require(not exact or all(matches.values()), 'source digest and public reset reference conflict')
    category = ('all_public_match' if exact else 'rgb_only_difference'
                if all(matches.values()) else 'proprioception_differs_rgb_vs_original_unknown')
    return dict(observation_sha256=digest, component_sha256=fields,
        source_reset_sha256=source['reset_observation_sha256'], reset_digest_exact=exact,
        state_exact=matches, state_max_absolute_error=errors, difference_category=category)


def target_comparison(records):
    selected = {r['branch']: r for r in records if r['initial_state_index'] == 7}
    require(set(selected) == {'direct', 'sequential'}, 'both target resets required')
    return dict(direct_matches_source=selected['direct']['reset_digest_exact'],
        sequential_matches_source=selected['sequential']['reset_digest_exact'],
        branches_match_each_other=selected['direct']['observation_sha256'] == selected['sequential']['observation_sha256'],
        reset_only_probe=True, original_full_episode_history_reproduced=False,
        sole_cause_identified=False, gripper_timing_identified=False)
