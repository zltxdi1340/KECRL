"""Independent archive/scalar audit of sequential-reset action replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments import libero_sequential_replay_protocol as p
from experiments.audit_libero_reset_reproducibility_diagnostic import independent_summary


def audit_capture(cfg, captured, *, source_loader):
    p.require(captured['status'] == 'sequential_prefix_capture_complete_pending_audit'
        and captured['phase'] == cfg['phase'] and captured['no_automatic_retry'] is True,
        'complete sequential capture required')
    base = Path(cfg['output']); sources = {}; byte_count = 0
    for descriptor in cfg['sources']:
        sources[descriptor['path']] = source_loader(descriptor)
        byte_count += Path(descriptor['path']).stat().st_size
    for field in ('initial_access', 'journal'):
        p.require(p.pin(captured[field]['path']) == captured[field]
            and Path(captured[field]['path']).parent == base, 'capture receipt changed')
    access = json.loads(Path(captured['initial_access']['path']).read_text()); p.check_access(access, cfg)
    events = [json.loads(line) for line in Path(captured['journal']['path']).read_text().splitlines()]
    p.require(len(events) == cfg['caps']['observations'], 'observation journal coverage differs')
    rows = captured['records']; problems = cfg['problems']
    p.require([r['problem_id'] for r in rows] == [r['problem_id'] for r in problems], 'prefix order differs')
    warmups = captured['warmups']; warm_index = {(r['problem_id'], r['initial_state_index']):r for r in warmups}
    p.require(len(warm_index) == len(warmups) == cfg['caps']['resets']-len(rows), 'warmup coverage differs')
    expected_events = []; expected_npzs = set(); checks = 0
    for problem, row in zip(problems, rows):
        for key, value in problem.items(): p.require(row[key] == value, 'prefix identity differs: '+key)
        source = sources[problem['source']['path']]; index = problem['initial_state_index']
        for i in range(index):
            warm = warm_index[problem['problem_id'], i]; item = warm['observation']
            name = f"{problem['problem_id']}-reset-{i:02d}.npz"; expected_npzs.add(name)
            p.require(Path(item['path']) == base/name and p.pin(item['path']) == item, 'warmup archive changed')
            with np.load(item['path'], allow_pickle=False) as archive:
                p.require(set(archive.files) == set(p.SHAPES) and len(archive.files) == 4, 'warmup public schema differs')
                arrays = {k:archive[k] for k in p.SHAPES}
            reference = cfg['warmup_references'][i]; original = sources[reference['path']]
            summary = independent_summary(arrays, original)
            p.require(warm['source'] == {k:reference[k] for k in ('path', 'sha256')}
                and warm['kind'] == 'warmup' and warm['step'] == 0 and warm['reset_digest_exact'] is True,
                'warmup source or reset gate differs')
            for key in ('observation_sha256', 'component_sha256', 'state_exact', 'state_max_absolute_error'):
                p.require(warm[key] == summary[key], 'independent warmup summary differs'); checks += 1
            p.require(summary['reset_digest_exact'] and all(summary['state_exact'].values()), 'warmup differs from source')
            expected_events.append(warm)
        item = row['frames']; name = problem['problem_id']+'-prefix.npz'; expected_npzs.add(name)
        p.require(p.pin(item['path']) == item and Path(item['path']) == base/name
            and row['initial_row_sha256'] == access['selected_row_sha256'][str(index)], 'prefix archive or row mapping differs')
        with np.load(item['path'], allow_pickle=False) as archive:
            p.require(set(archive.files) == {*p.SHAPES, 'actions'} and len(archive.files) == 5, 'prefix schema differs')
            arrays = {k:archive[k] for k in archive.files}
        for key, shape in p.SHAPES.items():
            p.require(arrays[key].shape == (101, *shape), 'prefix array dimensions differ')
        p.require(arrays['actions'].shape == (100,7) and arrays['actions'].dtype == np.float32
            and np.isfinite(arrays['actions']).all(), 'saved actions schema differs')
        for i, actual in enumerate(arrays['actions'], 1):
            original = source['stepwise_public_state_trace'][i]['action']
            p.require(all(float(a) == b for a, b in zip(actual, original)) and len(original) == 7,
                      'saved command changed in replay'); checks += 7
        digests = []
        for i in range(101):
            obs = {key:arrays[key][i] for key in p.SHAPES}
            reference = dict(reset_observation_sha256=source['reset_observation_sha256'],
                stepwise_public_state_trace=[source['stepwise_public_state_trace'][i]])
            summary = independent_summary(obs, reference)
            p.require(all(summary['state_exact'].values()), 'independent replay state differs')
            if i == 0: p.require(summary['reset_digest_exact'], 'target reset digest differs')
            digests.append(summary['observation_sha256'])
            event = dict(problem_id=problem['problem_id'], kind='target_reset' if i == 0 else 'step',
                initial_state_index=index, step=i, source={k:problem['source'][k] for k in ('path', 'sha256')},
                **{k:summary[k] for k in ('observation_sha256','component_sha256','state_exact','state_max_absolute_error')},
                reset_digest_exact=True if i == 0 else None)
            expected_events.append(event); checks += 9
        first = next((i for i in range(1,101) if source['stepwise_public_state_trace'][i]['action'][6] > .1), None)
        p.require(row['observation_sha256'] == digests and row['replay_steps'] == 100
            and row['first_close_step'] == first and row['event_frame_indices'] == p.event_indices(first,100)
            and row['source_checkpoint_sha256'] == source['checkpoint_sha256']
            and row['exact_public_prefix'] is True and row['original_rgb_recovered'] is False,
            'prefix digest or event metadata differs')
    p.require(events == expected_events, 'observation journal order or contents differ')
    p.require({path.name for path in base.glob('*.npz')} == expected_npzs, 'unexpected capture archive')
    ledger = captured['ledger']; p.require(ledger['caps'] == cfg['caps']
        and 0 <= ledger['elapsed_seconds'] <= cfg['wall_seconds_max'], 'capture time or caps differ')
    for key, cap in cfg['caps'].items():
        count = byte_count if key == 'source_episode_bytes' else cap
        p.require(ledger['counts'][key] == dict(attempted=count, completed=count) and count <= cap,
                  'actual replay budget differs: '+key); checks += 1
    return dict(status='passed_independent_sequential_replay_audit', formal_result=False,
        phase=cfg['phase'], prefixes=len(rows), warmup_observations=len(warmups),
        checks=checks, source_episode_parses=len(sources), source_episode_bytes=byte_count,
        new_native_steps=0, initial_bank_reads=0, model_reads=0,
        gripper_timing_identified=False, private_state_equivalence_proven=False)


def audit_media(cfg, captured, manifest, annotations=None):
    import imageio.v2 as imageio
    from PIL import Image
    from experiments.libero_gripper_approach_video_protocol import validate_annotations
    base = Path(cfg['output']); frames = {r['problem_id']:r for r in captured['records']}
    p.require([r['problem_id'] for r in manifest['media']] == list(frames), 'media prefix coverage differs')
    for row in manifest['media']:
        original = frames[row['problem_id']]
        for field in ('video', 'event_sheet'): p.require(p.pin(row[field]['path']) == row[field], 'media changed')
        reader = imageio.get_reader(row['video']['path'], 'ffmpeg')
        try:
            meta = reader.get_meta_data(); count = 0
            p.require(meta['fps'] == 20 and meta['size'] == (768,416), 'video format differs')
            for frame in reader:
                p.require(frame.shape == (416,768,3), 'video frame size differs'); count += 1
            p.require(count == 101, 'video frame count differs')
        finally: reader.close()
        p.require(row['event_frame_indices'] == original['event_frame_indices'], 'event anchors differ')
        with Image.open(row['event_sheet']['path']) as picture:
            picture.load(); p.require(picture.size == (768,416*len(original['event_frame_indices'])), 'event sheet size differs')
    for field in ('review_page', 'annotations', 'capture_audit'):
        p.require(p.pin(manifest[field]['path']) == manifest[field], 'media receipt changed')
    template = list(csv.DictReader(Path(manifest['annotations']['path']).open()))
    for row in template:
        key = row['case_id']+'-'+row['slot']; source = frames[key]
        p.require(row['first_close_step'] == ('' if source['first_close_step'] is None else str(source['first_close_step']))
            and row['verdict'] == 'pending', 'annotation anchors or original template changed')
    review = validate_annotations(template if annotations is None else annotations, cfg)
    if annotations is not None:
        p.require([(r['case_id'],r['slot'],r['first_close_step']) for r in annotations]
            == [(r['case_id'],r['slot'],r['first_close_step']) for r in template], 'review command anchors changed')
    return dict(status='passed_media_integrity_visual_review_pending' if review['pending']
        else 'passed_visual_record_integrity', phase=cfg['phase'], review=review,
        decoded_frames=101*len(frames), gripper_causal_effect_established=False,
        new_native_steps=0, policy_forwards=0)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--binding', type=Path, required=True); parser.add_argument('--stage', choices=['capture','media'], required=True)
    parser.add_argument('--annotations', type=Path); args = parser.parse_args()
    cfg, _ = p.preflight(args.config, args.binding); base = Path(cfg['output'])
    captured = json.loads((base/'capture.json').read_text())
    try:
        if args.stage == 'capture':
            execution = json.loads((base/'execution.json').read_text())
            p.require(execution['science'] == p.pin(args.config) and execution['binding'] == p.pin(args.binding)
                and execution['capture'] == p.pin(base/'capture.json') and execution['model_reads'] == 0,
                'capture execution provenance differs')
            def loader(item):
                raw = Path(item['path']).read_bytes(); p.require(hashlib.sha256(raw).hexdigest() == item['sha256'], 'audit source changed')
                return json.loads(raw)
            result = audit_capture(cfg, captured, source_loader=loader)
            result.update(science=p.pin(args.config), binding=p.pin(args.binding), capture=p.pin(base/'capture.json'))
            p.write_new(base/'independent-audit.json', result)
            p.write_new(base/'completion.json', dict(status='complete_audited_sequential_replay', phase=cfg['phase'],
                capture=p.pin(base/'capture.json'), audit=p.pin(base/'independent-audit.json'),
                gripper_timing_identified=False, private_state_equivalence_proven=False))
        else:
            rows = list(csv.DictReader(args.annotations.open())) if args.annotations else None
            result = audit_media(cfg, captured, json.loads((base/'media.json').read_text()), rows)
            if args.annotations: result['annotations'] = p.pin(args.annotations)
            p.write_new(base/('visual-record-audit.json' if args.annotations else 'media-audit.json'), result)
        print(json.dumps(result))
    except BaseException as error:
        p.write_new(base/(args.stage+'-audit-failure.json'), dict(status='technical_audit_failure', error=repr(error)))
        raise


if __name__ == '__main__': main()
