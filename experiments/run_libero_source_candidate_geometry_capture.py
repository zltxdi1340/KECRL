"""Bounded, fixed500 source-candidate geometry capture; preflight is metadata only."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

from experiments import libero_source_candidate_geometry_capture_protocol as science
from experiments.libero_formal_execution import Ledger

ROOT, require, sha = science.ROOT, science.require, science.sha
INITIAL_ROOT = Path('/home/zl202621/third_party/LIBERO/libero/libero/init_files')
DEFAULT_CONFIG = ROOT/'configs/libero-source-candidate-geometry-capture-diagnostic_v1.json'
DEFAULT_SCIENCE = ROOT/'configs/libero-source-candidate-geometry-capture-freeze-binding_v1.json'
DEFAULT_BINDING = ROOT/'configs/libero-source-candidate-geometry-capture-execution-binding_v1.json'


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False); handle.write('\n')


def resource_caps(config):
    b = config['design']['budget']
    return {'model_load': b['checkpoint_deserializations_max'], 'policy_object': b['models'],
            'initial_container': b['initial_containers_max'], 'numeric_row': b['numeric_initial_rows_max'],
            'adapter': b['adapter_creations_max'], 'native_reset': b['native_reset_attempts_max'],
            'episode': b['episodes'], 'native_control': b['native_control_steps_max'],
            'physics_step': b['native_physics_steps_max'], 'zero_action': b['zero_actions_max'],
            'policy_forward': b['policy_forwards_and_actions_max'], 'policy_action': b['policy_forwards_and_actions_max'],
            'shadow_allocation': b['shadow_allocations_max'], 'shadow_forward': b['shadow_position_forwards_max'],
            'geometry_row': b['geometry_rows_max'], 'public_row': b['public_rows_max'],
            'contact_record': b['saved_bowl_contact_records_max'], 'forbidden_observer_native_call': 0,
            'optimizer_update': 0, 'HDF5_read': 0, 'sealed_payload_read': 0, 'old_trajectory_read': 0}


def preflight(config_path, science_binding_path, execution_binding_path):
    config = science.preflight(config_path, science_binding_path)
    binding = json.loads(Path(execution_binding_path).read_text())
    require(binding.get('status') == 'source_geometry_execution_bound_before_payload' and
            binding.get('execution_binding_complete') is True and binding.get('formal_result') is False and
            binding.get('config_sha256') == sha(config_path) and
            binding.get('science_binding_sha256') == sha(science_binding_path), 'execution/science binding differs')
    require({'runner', 'observer', 'bounded_adapter', 'independent_auditor', 'tests', 'execution_contract'} <= set(binding['files']),
            'execution implementation/auditor pins missing')
    for pin in binding['files'].values():
        p = Path(pin['path']); p = p if p.is_absolute() else ROOT/p
        require(p.suffix in ('.py', '.json', '.md', '.bddl') and sha(p) == pin['sha256'], 'execution code/input changed')
    require((ROOT/binding['files']['runner']['path']).resolve() == Path(__file__).resolve(), 'wrong geometry entry')
    require(binding['caps'] == resource_caps(config), 'execution resource caps differ')
    for key in ('output', 'artifacts'):
        require(Path(binding[key]).is_absolute() and not Path(binding[key]).resolve().is_relative_to(ROOT),
                'raw outputs must remain outside repository')
    require(binding['device'] == {'physical_index': 1, 'uuid': 'GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff',
                                 'cuda_visible_devices': '1', 'egl_device_id': '1', 'inference_device': 'cuda:0'},
            'frozen GPU identity differs')
    return config, binding


def public_row(observation, state_index, action, joint0, gripper0, digest):
    import numpy as np
    j, g = observation['joint_states'], observation['gripper_states']
    return {'state_index': state_index, 'action': None if action is None else np.asarray(action).tolist(),
            'joint_state': j.tolist(), 'gripper_state': g.tolist(), 'observation_sha256': digest(observation),
            'joint_offset_l2_from_start': float(np.linalg.norm(j.astype(np.float64)-joint0)),
            'gripper_offset_l2_from_start': float(np.linalg.norm(g.astype(np.float64)-gripper0))}


def execute(config, *, initial_rows, loaded_models, model_to_device, adapter_factory, resetter,
            observer_factory, predictor, weight_checker, observation_digest, reset_geometry_digest,
            ledger, archive, event):
    """Callback core is tested with synthetic environments, never real smoke episodes."""
    import numpy as np
    archive = Path(archive)
    descriptors, resets, compiled, after_weights = [], {}, {}, []
    comparisons = 0
    for number, endpoint in enumerate(config['model_endpoints'], 1):
        task, seed, arm = endpoint['task'], endpoint['seed'], endpoint['arm']
        model = model_to_device(loaded_models[task, seed, arm])
        for index in science.INDICES:
            ledger.begin('episode')
            name = f'task{task}-seed{seed}-{arm}-global{index:02d}'
            pub_path, geo_path = archive/(name+'-public.json'), archive/(name+'-geometry.json')
            identity = {'task': task, 'seed': seed, 'arm': arm, 'global_initial_index': index,
                        'selected_bank_local_index': index, 'adapter_local_initial_index': 0,
                        'initial_state_role': 'candidate_nonregression', 'checkpoint_update': 500,
                        'checkpoint': endpoint['checkpoint'], 'stabilization_steps': 5, 'environment_seed': 20261022}
            public = {'status': 'running', 'identity': identity, 'rows': []}
            geometry = {'status': 'running', 'identity': identity, 'rows': []}
            adapter = observer = None
            try:
                random.seed(20261022); np.random.seed(20261022)
                # Torch is seeded by the injected runtime factory before construction.
                adapter = ledger.call('adapter', adapter_factory, task, initial_rows[task][index])
                observation, _ = resetter(adapter)
                j0 = np.asarray(observation['joint_states'], np.float64).copy()
                g0 = np.asarray(observation['gripper_states'], np.float64).copy()
                observer = observer_factory(adapter)
                geometry['header'] = observer.header

                def save_boundary(k, action):
                    ledger.observer_active = True
                    try:
                        row = ledger.call('geometry_row', observer.sample, observation, k)
                    finally:
                        ledger.observer_active = False
                    geometry['rows'].append(row)
                    public['rows'].append(ledger.call('public_row', public_row, observation, k, action,
                                                      j0, g0, observation_digest))

                save_boundary(0, None)
                reset = {'public_sha256': public['rows'][0]['observation_sha256'],
                         'geometry_sha256': reset_geometry_digest(geometry['rows'][0])}
                key = task, index
                if key in resets:
                    require(resets[key] == reset and compiled[key] == geometry['header'], 'paired reset/compiled geometry differs')
                    comparisons += 1
                else:
                    resets[key], compiled[key] = reset, geometry['header']
                public['reset'] = geometry['reset'] = reset
                event('paired_reset_verified', **identity, reset=reset, compiled_header=geometry['header'])
                success = False
                for step in range(1, 151):
                    action = np.asarray(ledger.call('policy_forward', predictor, model, observation), np.float32)
                    require(action.shape == (7,) and np.isfinite(action).all() and np.all(np.abs(action) <= 1), 'invalid action')
                    observation, reward, terminated, truncated, info = ledger.call('policy_action', adapter.step, action)
                    require(np.isfinite(reward) and not truncated and type(info['success']) is bool and
                            bool(terminated) == info['success'], 'unexpected native termination')
                    save_boundary(step, action)
                    if terminated:
                        success = True
                        break
                ending = {'steps': step, 'prefix_success': success,
                          'capture_stop_reason': 'success' if success else 'diagnostic_prefix_limit'}
                public.update(status='complete', **ending); geometry.update(status='complete', **ending)
                write_new(pub_path, public); write_new(geo_path, geometry)
                descriptor = {**identity, **ending,
                              'public': {'path': str(pub_path), 'sha256': sha(pub_path)},
                              'geometry': {'path': str(geo_path), 'sha256': sha(geo_path)}}
                descriptors.append(descriptor); ledger.finish('episode')
                event('episode_complete', **descriptor)
            except BaseException as error:
                for path, content in [(pub_path, public), (geo_path, geometry)]:
                    if not path.exists():
                        write_new(path, {**content, 'status': 'failed', 'error_type': type(error).__name__, 'error': str(error)})
                event('episode_failed', **identity, error_type=type(error).__name__, error=str(error))
                raise
            finally:
                if adapter is not None:
                    adapter.close()
                observer = None
                adapter = None
                gc.collect()
        after_weights.append(weight_checker(model, endpoint))
        event('model_complete', task=task, seed=seed, arm=arm, weight_verification=after_weights[-1])
        if hasattr(model, 'cpu'):
            model.cpu()
        print(json.dumps({'event': 'model_complete', 'models': number, 'episodes': len(descriptors),
                          'task': task, 'seed': seed, 'arm': arm, 'elapsed_seconds': ledger.snapshot()['elapsed_seconds']}), flush=True)
    return {'episode_files': descriptors, 'reset_references': len(resets), 'reset_exact_comparisons': comparisons,
            'weights_after': after_weights}


def run(config_path, science_binding_path, binding_path):
    config, binding = preflight(config_path, science_binding_path, binding_path)
    output, archive = Path(binding['output']), Path(binding['artifacts'])
    require(not output.exists() and not archive.exists(), 'capture outputs must be new; no retry')
    for key, value in binding['launch_environment'].items():
        require(os.environ.get(key) == value, 'launch environment differs: '+key)
    require(sys.version.split()[0] == '3.10.22', 'Python version differs')
    for name, version in {'torch': '2.4.1+cu124', 'numpy': '1.22.4', 'mujoco': '3.2.3', 'robosuite': '1.4.0'}.items():
        require(importlib.metadata.version(name) == version, 'runtime package differs: '+name)
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu = idle_gpu(1)
    require(gpu['target_gpu'][1] == binding['device']['uuid'], 'physical GPU UUID differs')
    require(subprocess.check_output(['git', '-C', '/home/zl202621/third_party/LIBERO', 'rev-parse', 'HEAD'],
                                    text=True).strip() == config['design']['runtime']['libero_revision'], 'LIBERO revision differs')
    archive.mkdir(parents=True, exist_ok=False)
    ledger = Ledger(binding['caps'], seconds=3600)
    ledger.observer_active = False
    events_path = archive/'events.jsonl'
    events = events_path.open('x')
    started_at = datetime.now(timezone.utc).isoformat()

    def event(kind, **values):
        events.write(json.dumps({'event': kind, 'utc': datetime.now(timezone.utc).isoformat(), **values},
                                allow_nan=False)+'\n'); events.flush()

    initial_access, loaded, verified, rollout = {}, {}, [], None
    originals = {}
    try:
        import numpy as np
        import torch
        import h5py
        from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
        from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy
        from src.environments.libero_initial_state_reader import load_initial_state_rows
        from src.environments.libero_geometry_observer import LiberoGeometryObserver, reset_geometry_digest
        from src.environments.libero_geometry_capture_adapter import build_source_geometry_adapter, reset_once, counted_native_physics
        from experiments.run_libero_source_internal_spt_spi_closed_loop_diagnostic import tensor_digest, observation_digest
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one selected CUDA GPU required')
        torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        originals['torch_load'], originals['HDF5_File'] = torch.load, h5py.File
        allowed = {row['checkpoint']['path'] for row in config['model_endpoints']}
        consumed = set()

        def guarded_load(path, *args, **kwargs):
            name = str(path)
            require(name in allowed and name not in consumed, 'unapproved/repeated checkpoint or initial deserialization')
            consumed.add(name)
            return ledger.call('model_load', originals['torch_load'], path, *args, **kwargs)

        def deny_HDF5(*args, **kwargs):
            ledger.begin('HDF5_read')

        torch.load, h5py.File = guarded_load, deny_HDF5
        training = json.loads(science.METADATA_PATHS['training'].read_text())
        policy_config = LiberoBCPolicyConfig.from_mapping(training['effective_policy_config'])
        for row in config['model_endpoints']:
            pin = row['checkpoint']
            require(sha(pin['path']) == pin['sha256'], 'checkpoint bytes changed')
            state = torch.load(pin['path'], map_location='cpu', weights_only=True)
            require(tensor_digest(state) == pin['weight_hash'] and all(torch.isfinite(v).all() for v in state.values()),
                    'checkpoint tensor hash/values differ')
            model = ledger.call('policy_object', make_deterministic_libero_policy, policy_config)
            model.load_state_dict(state, strict=True); model.eval().requires_grad_(False)
            loaded[row['task'], row['seed'], row['arm']] = model
            verified.append({**row, 'verified_file_sha256': pin['sha256'], 'verified_weight_hash': tensor_digest(model.state_dict())})
        event('all30_endpoints_verified_before_environment', models=verified)
        initial_rows = {}
        for task in science.TASKS:
            pin = config['initial_banks'][str(task)]
            ledger.begin('initial_container')

            def decoded(kind, rows):
                require(rows == [0, 1, 2, 3], 'numeric initial role/range differs')
                (ledger.begin if kind == 'attempted' else ledger.finish)('numeric_row', len(rows))

            initial_rows[task], initial_access[str(task)] = load_initial_state_rows(INITIAL_ROOT/pin['relative_path'], pin,
                [0, 1, 2, 3], authorized_indices=[0, 1, 2, 3], on_decode=decoded)
            ledger.finish('initial_container')
            event('source_initial_rows_decoded', task=task, access=initial_access[str(task)])

        def adapter_factory(task, initial_row):
            torch.manual_seed(20261022)
            return build_source_geometry_adapter(task, initial_row, ledger)

        def predictor(model, observation):
            image = lambda key: torch.from_numpy(np.ascontiguousarray(observation[key])).permute(2, 0, 1).unsqueeze(0).to('cuda:0')
            prop = lambda key: torch.from_numpy(np.ascontiguousarray(observation[key])).unsqueeze(0).to('cuda:0')
            with torch.inference_mode():
                action, hidden = model.forward_step(image('agentview_rgb'), image('eye_in_hand_rgb'),
                    prop('joint_states'), prop('gripper_states'), hidden=None)
            require(hidden is None, 'CNN-only returned hidden state')
            return action[0].cpu().numpy().copy()

        def weight_checker(model, row):
            value = tensor_digest(model.state_dict())
            require(value == row['checkpoint']['weight_hash'] and all(not p.requires_grad and p.grad is None for p in model.parameters()),
                    'fixed Policy weights/gradient flags changed')
            return {'task': row['task'], 'seed': row['seed'], 'arm': row['arm'], 'weight_hash': value,
                    'no_gradients_or_trainable_parameters': True}

        with counted_native_physics(ledger):
            rollout = execute(config, initial_rows=initial_rows, loaded_models=loaded,
                model_to_device=lambda model: model.to('cuda:0'), adapter_factory=adapter_factory, resetter=reset_once,
                observer_factory=lambda adapter: LiberoGeometryObserver(adapter.environment.control.env,
                    observation_digest=observation_digest, ledger=ledger), predictor=predictor, weight_checker=weight_checker,
                observation_digest=observation_digest, reset_geometry_digest=reset_geometry_digest,
                ledger=ledger, archive=archive, event=event)
        require(ledger.counts['episode']['completed'] == 120, 'incomplete capture')
        ledger.check_time()
        result = {'status': 'complete_capture_pending_independent_audit', 'formal_result': False,
                  'rollout': rollout, 'initial_state_access': initial_access, 'endpoints_verified_before_environment': verified}
    except BaseException as error:
        event('technical_abort', error_type=type(error).__name__, error=str(error))
        result = {'status': 'failed_capture_no_primary_summary', 'formal_result': False,
                  'failure': {'type': type(error).__name__, 'message': str(error)}, 'initial_state_access': initial_access,
                  'endpoints_verified_before_environment': verified,
                  'partial_artifact_paths': [str(p) for p in sorted(archive.glob('task*.json'))]}
    finally:
        if 'torch_load' in originals:
            torch.load = originals['torch_load']; h5py.File = originals['HDF5_File']
        events.close()
    result.update(ledger=ledger.snapshot(), event_log={'path': str(events_path), 'sha256': sha(events_path)},
                  provenance={'config_path': str(config_path), 'config_sha256': sha(config_path),
                    'science_binding_path': str(science_binding_path), 'science_binding_sha256': sha(science_binding_path),
                    'execution_binding_path': str(binding_path), 'execution_binding_sha256': sha(binding_path),
                    'started_at_utc': started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat(),
                    'gpu': gpu, 'argv': sys.argv, 'python': sys.version, 'torch': importlib.metadata.version('torch'),
                    'mujoco': importlib.metadata.version('mujoco'), 'robosuite': importlib.metadata.version('robosuite'),
                    'capture_implementation': 'fresh source prefixes; shadow geometry only; no training'},
                  G_decision_unchanged='failed_keep_H_sealed', new_training_updates=0)
    write_new(output, result)
    print(json.dumps({'status': result['status'], 'output': str(output), 'ledger': result['ledger']}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--science-binding', type=Path, default=DEFAULT_SCIENCE)
    parser.add_argument('--binding', type=Path, default=DEFAULT_BINDING)
    parser.add_argument('--mode', choices=['preflight', 'run'], default='preflight')
    args = parser.parse_args()
    if args.mode == 'preflight':
        c, b = preflight(args.config, args.science_binding, args.binding)
        print(json.dumps({'status': 'execution_metadata_preflight_passed', 'models': len(c['model_endpoints']),
                          'planned_episodes': 120, 'actual_new_research_calls': 0, 'output': b['output']}))
    else:
        result = run(args.config, args.science_binding, args.binding)
        if result['status'] != 'complete_capture_pending_independent_audit':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
