"""Saved-action prefixes after sequential resets, with failure observations."""
from __future__ import annotations

import argparse
import csv
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

from experiments import libero_sequential_replay_protocol as p
from experiments.libero_formal_execution import Ledger


def capture_dataset(cfg, *, adapter_factory, initial_loader, source_loader, destination):
    destination = Path(destination); p.require(not destination.exists(), 'new replay output required')
    destination.mkdir(parents=True); ledger = Ledger(cfg['caps'], seconds=cfg['wall_seconds_max'])
    records = []; warmups = []; access = None; active = None; buffer = None; problem = None
    def save_buffer(suffix):
        path = destination/(problem['problem_id']+'-'+suffix+'.npz')
        arrays = {k:np.stack(buffer[k]) for k in p.SHAPES}
        arrays['actions'] = np.asarray(buffer['actions'], np.float32).reshape(-1, 7)
        with path.open('xb') as handle: np.savez_compressed(handle, **arrays)
        return p.pin(path)
    def observe(observation, source, row, *, kind, index, step):
        ledger.begin('observations'); ledger.begin('rgb_frames', 2)
        ledger.begin('raw_rgb_bytes', 2*128*128*3)
        p.reset.observation_digest(observation)
        if kind != 'warmup':
            for key in p.SHAPES: buffer[key].append(observation[key].copy())
        entry = dict(problem_id=problem['problem_id'], kind=kind, initial_state_index=index,
            step=step, source={k:source[k] for k in ('path', 'sha256')},
            **p.describe(observation, row))
        if kind == 'warmup':
            path = destination/f"{problem['problem_id']}-reset-{index:02d}.npz"
            with path.open('xb') as handle: np.savez_compressed(handle, **observation)
            entry['observation'] = p.pin(path)
        entry['reset_digest_exact'] = (entry['observation_sha256'] == source['reset_observation_sha256']
                                       if kind in ('warmup', 'target_reset') else None)
        ledger.finish('observations'); ledger.finish('rgb_frames', 2); ledger.finish('raw_rgb_bytes', 2*128*128*3)
        with (destination/'observations.jsonl').open('a') as handle:
            handle.write(json.dumps(entry, allow_nan=False)+'\n')
        if kind == 'warmup': warmups.append(entry)
        p.require(all(entry['state_exact'].values()), 'source public joint/gripper state differs')
        if kind in ('warmup', 'target_reset'):
            p.require(entry['reset_digest_exact'] is True, 'source reset RGB/proprioception digest differs')
        return entry
    try:
        sources = {}
        for descriptor in cfg['sources']:
            size = Path(descriptor['path']).stat().st_size; ledger.begin('source_episode_bytes', size)
            source = ledger.call('source_episode_parses', source_loader, descriptor); ledger.finish('source_episode_bytes', size)
            p.require(source['status'] == 'complete' and source['stabilization_steps'] == 5
                and source['initial_state_role'] == cfg['initial_role']
                and tuple(source[k] for k in ('task_id', 'training_seed', 'arm', 'initial_state_index'))
                    == tuple(descriptor[k] for k in ('task_id', 'training_seed', 'arm', 'initial_state_index'))
                and source['reset_observation_sha256'] == descriptor['reset_observation_sha256']
                and len(source['stepwise_public_state_trace']) == source['steps']+1
                and [r['step'] for r in source['stepwise_public_state_trace']] == list(range(source['steps']+1)),
                'source episode schema or identity differs')
            sources[descriptor['path']] = {**source, 'path':descriptor['path'], 'sha256':descriptor['sha256']}
        states, access = initial_loader(ledger); p.check_access(access, cfg, states)
        p.write_new(destination/'initial-access.json', access)
        for problem in cfg['problems']:
            active = dict(problem_id=problem['problem_id'], kind='construction', index=None, step=None)
            buffer = {k:[] for k in [*p.SHAPES, 'actions']}
            adapter = ledger.call('adapters', adapter_factory, problem['task'], states.copy())
            native = adapter.environment.step; inside_reset = False; zeros = 0; pending = None; used = False
            def step_native(action):
                nonlocal zeros, used
                if inside_reset:
                    p.require(zeros < 5 and np.asarray(action).dtype == np.float32
                        and np.array_equal(np.asarray(action), np.zeros(7, np.float32)), 'five stabilization zeros required')
                    zeros += 1; ledger.begin('zero_steps')
                else:
                    p.require(pending is not None and not used and np.asarray(action).dtype == np.float32
                        and np.array_equal(np.asarray(action), pending), 'saved action or native call count differs')
                    used = True
                value = ledger.call('native_steps', native, action)
                if inside_reset: ledger.finish('zero_steps')
                return value
            adapter.environment.step = step_native
            try:
                target = sources[problem['source']['path']]; target_index = problem['initial_state_index']
                prefix_events = []
                for index in range(target_index+1):
                    kind = 'warmup' if index < target_index else 'target_reset'
                    active = dict(problem_id=problem['problem_id'], kind=kind, index=index, step=0)
                    if kind == 'target_reset': ledger.begin('replay_prefixes')
                    zeros = 0; inside_reset = True
                    try: observation, _ = ledger.call('resets', adapter.reset, initial_state_index=index)
                    finally: inside_reset = False
                    p.require(zeros == 5, 'stabilization count differs')
                    reference = sources[cfg['warmup_references'][index]['path']] if kind == 'warmup' else target
                    entry = observe(observation, reference, reference['stepwise_public_state_trace'][0],
                                    kind=kind, index=index, step=0)
                    if kind == 'target_reset': prefix_events.append(entry)
                for step in range(1, 101):
                    active = dict(problem_id=problem['problem_id'], kind='step', index=target_index, step=step)
                    action = np.asarray(target['stepwise_public_state_trace'][step]['action'], np.float32)
                    p.require(action.shape == (7,) and np.isfinite(action).all() and np.abs(action).max() <= 1,
                              'invalid saved command')
                    pending = action.copy(); used = False
                    observation, reward, terminated, truncated, info = ledger.call('saved_actions', adapter.step, action.copy())
                    p.require(used and np.isfinite(reward), 'missing native command or invalid reward')
                    buffer['actions'].append(action.copy())
                    prefix_events.append(observe(observation, target, target['stepwise_public_state_trace'][step],
                                                 kind='step', index=target_index, step=step))
                    terminal = step == target['steps']
                    p.require(bool(terminated or truncated) == terminal, 'native terminal differs within prefix')
                    if terminal:
                        p.require(info['success'] == target['success'] and info['terminal_reason'] == target['terminal_reason'],
                                  'source terminal outcome differs')
                ledger.finish('replay_prefixes'); frames = save_buffer('prefix')
                first = next((i for i, action in enumerate(buffer['actions'], 1) if float(action[6]) > .1), None)
                records.append(dict(**problem, frames=frames, replay_steps=100,
                    first_close_step=first, event_frame_indices=p.event_indices(first, 100),
                    observation_sha256=[r['observation_sha256'] for r in prefix_events],
                    source_checkpoint_sha256=target['checkpoint_sha256'],
                    initial_row_sha256=access['selected_row_sha256'][str(target_index)],
                    exact_public_prefix=True, original_rgb_recovered=False))
                buffer = None
                print(json.dumps(dict(event='prefix_complete', problem_id=problem['problem_id'], first_close_step=first)), flush=True)
            finally:
                adapter.environment.step = native; adapter.close()
        ledger.check_time()
        result = dict(status='sequential_prefix_capture_complete_pending_audit', formal_result=False,
            phase=cfg['phase'], records=records, warmups=warmups,
            initial_access=p.pin(destination/'initial-access.json'), journal=p.pin(destination/'observations.jsonl'),
            ledger=ledger.snapshot(), no_automatic_retry=True, at_UTC=datetime.now(timezone.utc).isoformat())
        p.write_new(destination/'capture.json', result); return result
    except BaseException as error:
        partial = save_buffer('partial') if buffer is not None and buffer['joint_states'] else None
        p.write_new(destination/'failure.json', dict(status='technical_sequential_replay_failure',
            error=repr(error), active=active, partial_observations=partial, records=records, warmups=warmups,
            initial_access=access, ledger=ledger.snapshot(), no_automatic_retry=True))
        raise


def capture(config, binding):
    cfg, _ = p.preflight(config, binding)
    p.require(os.path.ismount('/data') and os.access(p.ROOT/'results', os.W_OK), 'persistent data mount required')
    for key, value in cfg['runtime_environment'].items(): p.require(os.environ.get(key) == value, 'replay runtime differs: '+key)
    for key, value in cfg['versions'].items(): p.require(importlib.metadata.version(key) == value, 'replay package differs: '+key)
    p.require(sys.version.split()[0] == '3.10.22', 'Python differs')
    revision = subprocess.check_output(['git', '-C', str(p.ROOT/'third_party/LIBERO'), 'rev-parse', 'HEAD'], text=True).strip()
    p.require(revision == cfg['libero_revision'], 'LIBERO revision differs')
    from experiments.run_libero_source_internal_spt_spi_diagnostic_v2 import idle_gpu
    gpu = idle_gpu(1)
    from src.environments.libero_initial_state_reader import load_initial_state_rows
    from experiments.run_libero_bc_closed_loop_attribution import _build_injected_adapter
    import torch
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    def initial_loader(ledger):
        def decoded(kind, indices): (ledger.begin if kind == 'attempted' else ledger.finish)('numeric_rows', len(indices))
        item = cfg['initial_bank']; indices = cfg['initial_indices']
        return ledger.call('initial_row_loads', load_initial_state_rows, item['path'], item, indices,
                           authorized_indices=indices, on_decode=decoded)
    def adapter_factory(task, states):
        random.seed(20261022); np.random.seed(20261022); torch.manual_seed(20261022)
        return _build_injected_adapter(suite_name='libero_spatial', task_id=task, seed=20261022,
            image_size=128, max_episode_steps=300, stabilization_steps=5, initial_states=states)
    allowed = {r['path']:r['sha256'] for r in cfg['sources']}
    def source_loader(item):
        path = Path(item['path']).resolve()
        p.require(str(path) in allowed and path.parent == Path(cfg['source_archive'])/'episodes', 'replay source outside scope')
        raw = path.read_bytes(); p.require(hashlib.sha256(raw).hexdigest() == allowed[str(path)], 'source episode changed')
        return json.loads(raw)
    result = capture_dataset(cfg, adapter_factory=adapter_factory, initial_loader=initial_loader,
        source_loader=source_loader, destination=cfg['output'])
    p.write_new(Path(cfg['output'])/'execution.json', dict(science=p.pin(config), binding=p.pin(binding),
        capture=p.pin(Path(cfg['output'])/'capture.json'), gpu=gpu, policy_forwards=0, model_reads=0,
        previous_failed_run_resumed=False))
    return dict(status=result['status'], prefixes=len(result['records']))


def export_media(config, binding):
    cfg, _ = p.preflight(config, binding); base = Path(cfg['output'])
    audit = json.loads((base/'independent-audit.json').read_text())
    p.require(audit['status'] == 'passed_independent_sequential_replay_audit'
        and audit['capture'] == p.pin(base/'capture.json'), 'passed independent capture audit required')
    import imageio_ffmpeg
    p.require(p.pin(imageio_ffmpeg.get_ffmpeg_exe()) == cfg['ffmpeg_binary'], 'media encoder differs')
    from experiments.run_libero_gripper_approach_video_diagnostic import export_prefix_media
    captured = json.loads((base/'capture.json').read_text()); media = []; annotations = []
    for row in captured['records']:
        p.require(p.pin(row['frames']['path']) == row['frames'], 'captured frames changed')
        with np.load(row['frames']['path'], allow_pickle=False) as archive: arrays = {k:archive[k] for k in archive.files}
        media.append(dict(problem_id=row['problem_id'], **export_prefix_media(arrays, row,
            case_id=row['case_id'], slot=row['slot'], directory=base/'media')))
        annotations.append(dict(case_id=row['case_id'], slot=row['slot'],
            first_close_step='' if row['first_close_step'] is None else row['first_close_step'], verdict='pending',
            target_visible='', gripper_visible='', evidence_frame_indices='[]', observer='', notes=''))
    with (base/'annotations-template.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotations[0])); writer.writeheader(); writer.writerows(annotations)
    page = '''<!doctype html><meta charset="utf-8"><title>顺序reset后的动作视频</title>
<style>body{font-family:sans-serif;max-width:1600px;margin:24px auto}.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px}video,img{width:100%}section{border-top:1px solid #aaa;padding:16px}</style>
<h1>顺序reset后的原动作回放</h1><p>原128×128双视角补录；左agentview、右手眼，3倍显示不增加信息。帧c−1为close命令执行前。A/B隐藏模型，案例结果已知，非盲测。无法看清时保留未知。当前页面没有证明夹爪时机或因果效果。</p>'''
    for case in cfg['cases']:
        name = case['case_id']; page += f'<section><h2>{name} · task{case["task"]}</h2><div class="pair">'
        for slot in ('A', 'B'):
            page += f'<div><h3>{slot}</h3><video controls muted src="media/{name}-{slot}.mp4"></video><img src="media/{name}-{slot}-event.png"></div>'
        page += '</div></section>'
    with (base/'review.html').open('x') as handle: handle.write(page)
    p.write_new(base/'media.json', dict(status='media_exported_visual_review_pending', media=media,
        capture_audit=p.pin(base/'independent-audit.json'), review_page=p.pin(base/'review.html'),
        annotations=p.pin(base/'annotations-template.csv'), visual_cause_identified=False))
    return dict(status='media_exported_visual_review_pending', prefixes=len(media))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True); parser.add_argument('--stage', choices=['capture', 'media'], required=True)
    args = parser.parse_args(); print(json.dumps((capture if args.stage == 'capture' else export_media)(args.config, args.binding)))
