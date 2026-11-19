"""Versioned sequential-reset saved-action capture and source admission."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import libero_reset_reproducibility_protocol as reset
from experiments.libero_gripper_approach_video_protocol import ROOT, RUNTIME, pin, require, write_new, event_indices

SHAPES = reset.SHAPES
ARMS = ['frozen_resnet18', 'layer4_conv_finetune']


def cases_for_phase(parent, phase):
    require(phase in ('task4_prefix_gate', 'task7_visual_cases'), 'unknown replay phase')
    cases = ([row for row in parent['cases'] if row['case_id'] == 'case01'] if phase == 'task4_prefix_gate'
             else [row for row in parent['cases'] if row['task'] == 7])
    require(len(cases) == (1 if phase == 'task4_prefix_gate' else 8), 'frozen replay cases differ')
    return cases


def budget(cases):
    prefixes = 2*len(cases); resets = 2*sum(row['initial_state_index']+1 for row in cases)
    maximum = max(row['initial_state_index'] for row in cases)
    observations = resets-prefixes+prefixes*101
    return dict(adapters=prefixes, resets=resets, replay_prefixes=prefixes,
        saved_actions=prefixes*100, zero_steps=resets*5, native_steps=prefixes*100+resets*5,
        initial_row_loads=1, numeric_rows=maximum+1, observations=observations,
        rgb_frames=2*observations, raw_rgb_bytes=2*observations*128*128*3,
        source_episode_parses=None, source_episode_bytes=8*1024**2,
        policy_forwards=0, backwards=0, optimizer_updates=0, model_creations=0,
        checkpoint_reads=0, hdf5_reads=0, extra_render_calls=0)


def reference_design(parent, evaluation, phase):
    cases = cases_for_phase(parent, phase); task = cases[0]['task']
    maximum = max(row['initial_state_index'] for row in cases)
    metadata = {(r['task_id'], r['training_seed'], r['arm'], r['initial_state_index']):r
                for r in evaluation['rollout']['episode_files']}
    warmups = [metadata[task, 20261106, ARMS[0], i] for i in range(maximum)]
    problems = []
    for case in cases:
        for slot in ('A', 'B'):
            arm = case['slots'][slot]; source = case['source_episodes'][arm]
            require(metadata[task, case['seed'], arm, case['initial_state_index']] == source
                and source['steps'] >= 100, 'target source metadata differs')
            for i in range(case['initial_state_index']):
                require(metadata[task, case['seed'], arm, i]['reset_observation_sha256']
                    == warmups[i]['reset_observation_sha256'], 'warmup reset references are not paired')
            problems.append(dict(problem_id=case['case_id']+'-'+slot, case_id=case['case_id'],
                slot=slot, task=task, seed=case['seed'], arm=arm,
                initial_state_index=case['initial_state_index'], source=source))
    sources = {row['path']:row for row in warmups}
    for problem in problems: sources[problem['source']['path']] = problem['source']
    return cases, problems, warmups, list(sources.values())


def validate_scope(cfg):
    expected = dict(kind='libero_sequential_reset_saved_action_replay', version=2,
        formal_result=False, initial_role='candidate_nonregression', environment_seed=20261022,
        stabilization_steps=5, prefix_actions=100, adapter_horizon=300, image_size=128,
        no_automatic_retry=True, policy_inference=False, previous_full_episodes_replayed=False,
        exact_public_prefix_required=True, new_success_rate_estimate=False, runtime_environment=RUNTIME)
    for key, value in expected.items():
        require(json.dumps(cfg.get(key), sort_keys=True) == json.dumps(value, sort_keys=True),
                'sequential replay scope differs: '+key)
    require(cfg['phase'] in ('task4_prefix_gate', 'task7_visual_cases'), 'invalid replay phase')
    caps = budget(cfg['cases']); caps['source_episode_parses'] = len(cfg['sources'])
    require(cfg['caps'] == caps and len({r['path'] for r in cfg['sources']}) == len(cfg['sources']),
            'replay cap/source schema differs')
    require(cfg['initial_indices'] == list(range(caps['numeric_rows']))
        and cfg['wall_seconds_max'] == (180 if cfg['phase'] == 'task4_prefix_gate' else 1800)
        and Path(cfg['output']).resolve().is_relative_to(ROOT/'results'), 'replay domain/storage differs')


def preflight(config_path, binding_path):
    cfg = json.loads(Path(config_path).read_text()); validate_scope(cfg)
    bound = json.loads(Path(binding_path).read_text())
    require(bound['status'] == 'bound_before_sequential_replay_payload' and bound['science'] == pin(config_path),
            'new sequential replay binding required')
    for item in bound['code_dependencies'] + bound['validation']:
        require(pin(item['path']) == item, 'replay dependency changed: '+item['path'])
    for field in ('parent_video_science', 'parent_reset_binding', 'source_evaluation'):
        require(pin(cfg[field]['path']) == cfg[field], 'replay parent pin changed: '+field)
    old = json.loads(Path(cfg['parent_reset_binding']['path']).read_text())
    closure = {r['path']:r['sha256'] for r in bound['code_dependencies']}
    require({r['path']:r['sha256'] for r in old['code_dependencies']}.items() <= closure.items(),
            'replay parent closure missing')
    for name in ('libero_sequential_replay_protocol.py', 'run_libero_sequential_replay.py',
                 'audit_libero_sequential_replay.py'):
        path = ROOT/'experiments'/name
        require(closure.get(str(path)) == pin(path)['sha256'], 'replay implementation missing')
    parent = json.loads(Path(cfg['parent_video_science']['path']).read_text())
    evaluation = json.loads(Path(cfg['source_evaluation']['path']).read_text())
    cases, problems, warmups, sources = reference_design(parent, evaluation, cfg['phase'])
    require((cfg['cases'], cfg['problems'], cfg['warmup_references'], cfg['sources'])
        == (cases, problems, warmups, sources), 'replay source selection changed')
    task = cases[0]['task']
    require(cfg['initial_bank'] == parent['initial_banks'][str(task)]
        and cfg['versions'] == parent['versions'] and cfg['libero_revision'] == parent['libero_revision']
        and cfg['source_archive'] == parent['source_archive']
        and cfg['source_evaluation'] == parent['source_provenance']['evaluation'], 'replay provenance differs')
    auth = bound['authorization']; require(pin(auth['path']) == auth, 'replay authorization changed')
    grant = json.loads(Path(auth['path']).read_text())
    require(grant['user_approved'] is True and grant['science'] == pin(config_path)
        and grant['tasks'] == [task] and grant['indices'] == cfg['initial_indices']
        and grant['role'] == cfg['initial_role'] and grant['caps'] == cfg['caps']
        and grant['sealed_roles_allowed'] == [] and grant['model_reads_allowed'] is False
        and grant['hdf5_allowed'] is False and grant['policy_inference_allowed'] is False
        and grant['previous_failed_run_resumed'] is False, 'replay authorization scope differs')
    if cfg['phase'] == 'task7_visual_cases':
        gate = cfg['task4_gate']; require(pin(gate['path']) == gate, 'prefix gate changed')
        receipt = json.loads(Path(gate['path']).read_text())
        require(receipt['status'] == 'complete_audited_sequential_replay'
            and receipt['phase'] == 'task4_prefix_gate', 'task4 replay gate has not passed')
        for field in ('capture', 'audit'):
            require(pin(receipt[field]['path']) == receipt[field], 'task4 gate provenance changed')
    return cfg, bound


def check_access(access, cfg, states=None):
    indices = cfg['initial_indices']; item = cfg['initial_bank']
    require(access['numeric_rows_decoded'] == indices
        and access['global_to_local_index'] == {str(i):i for i in indices}
        and access['source_path'] == item['path'] and access['container_sha256'] == item['sha256']
        and access['pickle_header_sha256'] == item['pickle_header_sha256']
        and access['decoded_selected_bytes'] == len(indices)*92*8
        and access['skipped_rows_numerically_decoded'] == [] and access['skipped_rows_buffered'] is False
        and access['used_pickle_or_torch_deserialization'] is False
        and set(access['selected_row_sha256']) == {str(i) for i in indices}, 'replay initial receipt differs')
    if states is not None:
        require(states.shape == (len(indices), 92) and states.dtype == np.float64 and np.isfinite(states).all()
            and all(hashlib.sha256(states[i].tobytes()).hexdigest() == access['selected_row_sha256'][str(i)]
                    for i in indices), 'injected initial rows differ')


def describe(observation, source_row):
    digest = reset.observation_digest(observation); parts = {}; exact = {}; errors = {}
    for key in SHAPES:
        value = np.ascontiguousarray(observation[key]); h = hashlib.sha256()
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(np.asarray(value.shape, dtype=np.int64).tobytes()); h.update(value.tobytes())
        parts[key] = h.hexdigest()
    for field, original in [('joint_states', 'joint_state'), ('gripper_states', 'gripper_state')]:
        ref = np.asarray(source_row[original], np.float64); current = observation[field].astype(np.float64)
        require(ref.shape == SHAPES[field] and np.isfinite(ref).all(), 'invalid source public state')
        exact[field] = bool(np.array_equal(current, ref)); errors[field] = float(np.max(np.abs(current-ref)))
    return dict(observation_sha256=digest, component_sha256=parts, state_exact=exact,
                state_max_absolute_error=errors)
