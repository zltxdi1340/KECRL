"""Collect G-only schema/length/header metadata without reading numeric payloads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from experiments import libero_formal_transfer_protocol_v2 as protocol
from experiments.libero_formal_execution import (
    ROOT, DATA_ROOT, INITIAL_ROOT, expected_evaluation_batches, require, sha, stage_plan,
)


def write_new(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def shape_metadata(path):
    import h5py
    lengths = {}
    with h5py.File(path, 'r') as handle:
        data = handle['data']
        require(set(data) == {f'demo_{i}' for i in range(50)}, 'unexpected demonstration inventory')
        for index in range(50):
            demo = data[f'demo_{index}']
            shape = demo['actions'].shape
            require(len(shape) == 2 and shape[0] >= 2 and shape[1] == 7, 'invalid action schema')
            length = int(shape[0])
            for key, tail in [('agentview_rgb', (128, 128, 3)), ('eye_in_hand_rgb', (128, 128, 3)),
                              ('joint_states', (7,)), ('gripper_states', (2,))]:
                require(demo['obs'][key].shape == (length, *tail), 'observation schema differs')
            lengths[str(index)] = length
    return lengths


def prepare(manifest_path, metadata_path):
    require(not Path(manifest_path).exists() and not Path(metadata_path).exists(), 'outputs must be new')
    scientific = ROOT/'configs/libero-formal-transfer-protocol_v2.json'
    science_binding = ROOT/'configs/libero-formal-transfer-freeze-binding_v2.json'
    protocol.preflight(scientific, science_binding)
    config = json.loads(scientific.read_text())
    references = {name: config['references'][name] for name in ('transfer', 'dataset_manifest', 'family', 'learning')}
    load = lambda key: json.loads((ROOT/references[key]['path']).read_text())
    transfer, datasets = load('transfer'), load('dataset_manifest')
    plan = stage_plan('G')
    lengths, banks, inspected_files = {}, {}, []
    # Importing the reader here reads no bank. This command only calls its header inspector.
    from src.environments.libero_initial_state_reader import inspect_initial_bank_header
    for task in plan['initial_tasks']:
        pin = next(row for row in datasets['files'] if row['task_id'] == task)
        path = Path(DATA_ROOT)/pin['path']
        require(path.stat().st_size == pin['size_bytes'] and sha(path) == pin['sha256'], 'dataset bytes changed')
        lengths[str(task)] = shape_metadata(path)
        inspected_files.append({'task': task, 'path': str(path), 'sha256': pin['sha256'],
                                'inspected': 'group names, dataset shapes and trajectory lengths only'})
        initial = next(row for row in transfer['initial_state_files'] if row['task_id'] == task)
        header = inspect_initial_bank_header(Path(INITIAL_ROOT)/initial['path'], initial)
        banks[str(task)] = {'relative_path': initial['path'], 'size_bytes': initial['size_bytes'],
                            'sha256': initial['sha256'], 'pickle_header_sha256': header['pickle_header_sha256']}
    batches = expected_evaluation_batches(plan, lengths)
    require(batches <= 120000, 'G offline batch cap exceeded before execution')
    record = {'status': 'G_metadata_collected_no_payload_values', 'formal_result': False,
              'collected_at_utc': datetime.now(timezone.utc).isoformat(), 'collector_sha256': sha(Path(__file__)),
              'task_scope': plan['initial_tasks'], 'dataset_files': inspected_files,
              'action_lengths_by_task': lengths, 'initial_bank_headers': banks,
              'expected_offline_forward_batches': batches,
              'RGB_action_joint_gripper_value_reads': 0, 'numeric_initial_rows_decoded': [],
              'policy_optimizer_environment_calls': 0, 'heldout_files_opened': [],
              'checksum_reads_entire_container_bytes': True}
    write_new(metadata_path, record)
    inputs = {**references,
              'scientific_config': {'path': str(scientific.relative_to(ROOT)), 'sha256': sha(scientific)},
              'scientific_binding': {'path': str(science_binding.relative_to(ROOT)), 'sha256': sha(science_binding)},
              'metadata_collection': {'path': str(Path(metadata_path).resolve()), 'sha256': sha(metadata_path)}}
    manifest = {'schema_version': 1, 'status': 'G_plan_bound_before_payload_execution', 'formal_result': False,
                'stage': 'G', 'plan': plan, 'inputs': inputs, 'initial_banks': banks,
                'action_lengths_by_task': lengths, 'expected_offline_batches': batches,
                'automatic_H_execution': False, 'qualification_budget': 0}
    write_new(manifest_path, manifest)
    return {'manifest_sha256': sha(manifest_path), 'metadata_sha256': sha(metadata_path),
            'expected_offline_batches': batches, 'numeric_initial_rows_decoded': []}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--metadata', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.metadata)))


if __name__ == '__main__':
    main()
