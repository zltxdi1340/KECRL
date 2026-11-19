"""A lagged cached sensor is not the raw channel; real zero gates stay strict."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from experiments import libero_postgrasp_determinism_protocol as p
from experiments import audit_libero_postgrasp_determinism as original
from experiments import audit_libero_postgrasp_determinism_v2 as revised


def invented(tmp_path):
    spec=importlib.util.spec_from_file_location('determinism_invented_helpers',p.ROOT/'tests/test_libero_postgrasp_determinism.py')
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    cfg,prepared=helper.fixture(tmp_path);records={}
    for arm in p.ARMS:records[arm]=helper.run((cfg,prepared),arm)[0]
    for arm,record in records.items():
        for member in ['physics','endpoint']+(['post_query'] if arm==p.ARMS[1] else []):
            item=record['saved'][member]
            with np.load(item['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
            for key in ['joint_qpos','gripper_qpos']:
                name='after_robot_'+key if member=='endpoint' else key
                arrays[name]=arrays[name]+.001
            with Path(item['path']).open('wb') as f:np.savez_compressed(f,**arrays)
            item.update(p.pin(item['path']))
    return cfg,prepared,records


def test_cached_and_raw_channels_can_differ_while_each_exact_gate_holds(tmp_path):
    cfg,prepared,records=invented(tmp_path)
    with pytest.raises(ValueError,match='raw/public qpos projection differs'):original.audit(cfg,prepared,records)
    result=revised.audit(cfg,prepared,records)
    assert result['new_robot_qpos_qvel_byte_identical'] and result['both_prefixes_exact_to_original']
    assert result['raw_public_projection'][p.ARMS[0]]['joint_qpos']['different_frames']==154
    assert result['raw_public_projection'][p.ARMS[0]]['joint_qpos']['admission_gate'] is False


def test_one_ulp_new_physics_difference_still_rejected(tmp_path):
    cfg,prepared,records=invented(tmp_path);item=records[p.ARMS[1]]['saved']['physics']
    with np.load(item['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays['joint_qvel'][11,0]=np.nextafter(arrays['joint_qvel'][11,0],np.inf)
    with Path(item['path']).open('wb') as f:np.savez_compressed(f,**arrays)
    item.update(p.pin(item['path']))
    with pytest.raises(ValueError):revised.audit(cfg,prepared,records)


def test_one_byte_RGB_difference_still_rejected(tmp_path):
    cfg,prepared,records=invented(tmp_path);item=records[p.ARMS[1]]['saved']['public']
    with np.load(item['path'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    arrays['eye_in_hand_rgb'][153,86,108,1]+=1
    with Path(item['path']).open('wb') as f:np.savez_compressed(f,**arrays)
    item.update(p.pin(item['path']))
    with pytest.raises(ValueError):revised.audit(cfg,prepared,records)
