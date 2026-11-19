"""Run exactly two environment lifetimes and nine source-development resets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

from experiments import libero_reset_reproducibility_protocol as p
from experiments.libero_formal_execution import Ledger


def run_probe(cfg, *, adapter_factory, initial_loader, source_loader, destination):
    destination = Path(destination)
    p.require(not destination.exists(), 'new reset output required')
    destination.mkdir(parents=True)
    ledger = Ledger(p.CAPS, seconds=180); records = []; access = None; active = None
    try:
        sources = {}
        for descriptor in cfg['references']:
            size = Path(descriptor['path']).stat().st_size
            ledger.begin('source_episode_bytes', size)
            source = ledger.call('source_episode_parses', source_loader, descriptor)
            ledger.finish('source_episode_bytes', size)
            index = descriptor['initial_state_index']
            p.require((source['task_id'], source['training_seed'], source['arm'], source['initial_state_index'])
                == (4, 20261106, 'frozen_resnet18', index)
                and source['status'] == 'complete' and source['stabilization_steps'] == 5
                and source['initial_state_role'] == cfg['role']
                and source['reset_observation_sha256'] == descriptor['reset_observation_sha256']
                and source['stepwise_public_state_trace'][0]['step'] == 0, 'source reset identity differs')
            sources[index] = source
        states, access = initial_loader(ledger)
        p.require(states.shape == (8, 92) and states.dtype == np.float64 and np.isfinite(states).all(),
                  'selected initial rows differ')
        p.validate_initial_access(access, cfg)
        p.require(all(hashlib.sha256(states[i].tobytes()).hexdigest() == access['selected_row_sha256'][str(i)]
            for i in p.INDICES), 'initial receipt and injected rows differ')
        p.write_new(destination/'initial-access.json', access)
        for name, indices in p.BRANCHES:
            active = dict(branch=name, initial_state_index=None)
            adapter = ledger.call('adapters', adapter_factory, states.copy())
            native = adapter.environment.step; inside_reset = False; zero_count = 0
            def step(action):
                nonlocal zero_count
                p.require(inside_reset and zero_count < 5
                    and np.asarray(action).dtype == np.float32
                    and np.array_equal(np.asarray(action), np.zeros(7, np.float32)),
                    'only five stabilization zero actions per reset are permitted')
                zero_count += 1; ledger.begin('zero_steps')
                result = ledger.call('native_steps', native, action)
                ledger.finish('zero_steps'); return result
            adapter.environment.step = step
            try:
                for index in indices:
                    active = dict(branch=name, initial_state_index=index)
                    zero_count = 0; inside_reset = True
                    try:
                        observation, _ = ledger.call('resets', adapter.reset, initial_state_index=index)
                    finally:
                        inside_reset = False
                    p.require(zero_count == 5, 'five stabilization steps required')
                    p.observation_digest(observation)
                    ledger.begin('observations'); ledger.begin('rgb_frames', 2)
                    ledger.begin('raw_rgb_bytes', 2*128*128*3)
                    path = destination/f'{name}-{index:02d}.npz'
                    with path.open('xb') as handle:
                        np.savez_compressed(handle, **{k: observation[k].copy() for k in p.SHAPES})
                    ledger.finish('observations'); ledger.finish('rgb_frames', 2)
                    ledger.finish('raw_rgb_bytes', 2*128*128*3)
                    record = dict(branch=name, initial_state_index=index, local_initial_state_index=index,
                        observation=p.pin(path), source={k: cfg['references'][index][k] for k in ('path', 'sha256')},
                        initial_row_sha256=hashlib.sha256(states[index].tobytes()).hexdigest(),
                        **p.summarize(observation, sources[index]))
                    records.append(record)
                    with (destination/'reset-records.jsonl').open('a') as handle:
                        handle.write(json.dumps(record, allow_nan=False)+'\n')
                    print(json.dumps(dict(event='reset_recorded', branch=name, index=index,
                        reset_digest_exact=record['reset_digest_exact'],
                        difference_category=record['difference_category'])), flush=True)
            finally:
                adapter.environment.step = native; adapter.close()
        ledger.check_time()
        result = dict(status='reset_probe_complete_pending_audit', formal_result=False,
            records=records, initial_access=p.pin(destination/'initial-access.json'),
            ledger=ledger.snapshot(), comparison=p.target_comparison(records),
            no_automatic_retry=True, at_UTC=datetime.now(timezone.utc).isoformat())
        p.write_new(destination/'reset-probe.json', result)
        return result
    except BaseException as error:
        p.write_new(destination/'failure.json', dict(status='technical_reset_probe_failure',
            error=repr(error), active=active, records=records, initial_access=access,
            ledger=ledger.snapshot(), no_automatic_retry=True))
        raise


def capture(config, binding):
    cfg, _ = p.preflight(config, binding)
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results', os.W_OK), 'persistent writable data mount required')
    for key, value in cfg['runtime_environment'].items():
        p.require(os.environ.get(key) == value, 'reset runtime differs: '+key)
    for key, value in cfg['versions'].items():
        p.require(importlib.metadata.version(key) == value, 'reset package differs: '+key)
    p.require(sys.version.split()[0] == '3.10.22', 'Python differs')
    revision = subprocess.check_output(['git', '-C', str(p.ROOT/'third_party/LIBERO'), 'rev-parse', 'HEAD'], text=True).strip()
    p.require(revision == cfg['libero_revision'], 'LIBERO revision differs')
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu = idle_gpu(1)
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    def initial_loader(ledger):
        def decoded(kind, indices):
            (ledger.begin if kind == 'attempted' else ledger.finish)('numeric_rows', len(indices))
        item = cfg['initial_bank']
        return ledger.call('initial_row_loads', load_initial_state_rows, item['path'], item,
            p.INDICES, authorized_indices=p.INDICES, on_decode=decoded)
    def adapter_factory(states):
        random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial', task_id=4, seed=20261022,
            image_size=128, max_episode_steps=300, stabilization_steps=5, initial_states=states)
    allowed = {row['path']: row['sha256'] for row in cfg['references']}
    def source_loader(item):
        path = Path(item['path']).resolve()
        p.require(str(path) in allowed and path.parent == Path(cfg['source_archive'])/'episodes', 'source path outside reset scope')
        raw = path.read_bytes()
        p.require(hashlib.sha256(raw).hexdigest() == allowed[str(path)], 'source reset JSON changed')
        return json.loads(raw)
    result = run_probe(cfg, adapter_factory=adapter_factory, initial_loader=initial_loader,
                       source_loader=source_loader, destination=cfg['output'])
    p.write_new(Path(cfg['output'])/'execution.json', dict(science=p.pin(config), binding=p.pin(binding),
        capture=p.pin(Path(cfg['output'])/'reset-probe.json'), gpu=gpu,
        initial_bank=p.pin(Path(cfg['output'])/'initial-access.json'),
        new_policy_evaluation=False, model_constructed=False, model_reads=0,
        previous_video_resumed=False))
    return dict(status=result['status'], comparison=result['comparison'])


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', required=True)
    parser.add_argument('--binding', required=True); args = parser.parse_args()
    print(json.dumps(capture(Path(args.config), Path(args.binding)), ensure_ascii=False))


if __name__ == '__main__':
    main()
