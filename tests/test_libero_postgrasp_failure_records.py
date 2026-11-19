"""Invented arrays only; no original images, models, initial banks or physics."""
from copy import deepcopy
import json

import numpy as np
import pytest

from experiments import analyze_libero_postgrasp_failure_records as a


def arrays(seed=19):
    rng=np.random.RandomState(seed)
    result=dict(actions=rng.uniform(-1,1,(300,7)).astype(np.float32),joint_states=rng.normal(size=(301,7)).astype(np.float32),
        gripper_states=rng.normal(size=(301,2)).astype(np.float32),rewards=np.zeros(300,np.float64),
        terminated=np.zeros(300,np.bool_),truncated=np.r_[np.zeros(299,np.bool_),True])
    return result


@pytest.mark.parametrize('seed',[1,7,10,19,22,77])
def test_numpy_and_scalar_events_and_descriptive_measures(seed):
    values=arrays(seed);row=dict(anchor_step=75,steps=300)
    a.crosscheck(a.numpy_metrics(row,values),a.scalar_metrics(row,values))


def test_positive_commands_never_called_physical_grasps_or_missing_release_failure():
    values=arrays();values['actions'][:,6]=1.;row=dict(anchor_step=100,steps=300)
    result=a.numpy_metrics(row,values);a.crosscheck(result,a.scalar_metrics(row,values))
    assert result['first_open_command'] is None and result['close_requests']==200
    assert result['closing_to_opening_steps']==[] and 'xy_distance' not in result and 'drop' not in result


def test_hold_is_not_open_and_open_at_handoff_is_not_a_new_release_transition():
    values=arrays();values['actions'][:,6]=-1.;values['actions'][80:100,6]=0.;values['actions'][100:150,6]=1.
    row=dict(anchor_step=75,steps=300);result=a.numpy_metrics(row,values)
    a.crosscheck(result,a.scalar_metrics(row,values))
    assert result['first_open_command']==76 and result['first_open_offset']==1
    assert result['hold_requests']==20 and result['closing_to_opening_steps']==[151]


@pytest.mark.parametrize('kind',['nan','bad_shape','bad_flags','action_range','extra_member','hash'])
def test_bad_or_out_of_scope_archive_rejected(tmp_path,kind):
    values=arrays();values.update(agentview_rgb=np.zeros(1,np.uint8),eye_in_hand_rgb=np.zeros(1,np.uint8))
    if kind=='nan':values['actions'][10,2]=np.nan
    if kind=='bad_shape':values['joint_states']=values['joint_states'][:-1]
    if kind=='bad_flags':values['terminated'][100]=True
    if kind=='action_range':values['actions'][100,0]=1.1
    if kind=='extra_member':values['eef_pos']=np.zeros((301,3),np.float32)
    path=tmp_path/'invented.npz'
    with path.open('wb') as handle:np.savez_compressed(handle,**values)
    row=dict(steps=300,frames=a.pin(path))
    if kind=='hash':row['frames']['sha256']='0'*64
    with pytest.raises(ValueError):a.load_numeric(row)


def metadata():
    rows=[]
    for i in range(22):
        for arm in a.parent.ARMS:
            success=(i>=18 and (i<21 or arm==a.parent.ARMS[1]))
            rows.append(dict(problem_id=f's{i}-{arm}',source=dict(source_id=f's{i}'),seed=20261106,controller=arm,
                anchor_step=75,steps=300,success=success,terminal_reason='success' if success else 'time_limit',frames=dict(path='/invented/unused.npz',sha256='a'*64)))
    return dict(status='postgrasp_handoff_complete_pending_audit',episodes=rows)


def test_selection_all18_pairs_and_no_successes_or_partial_parents():
    value=metadata();chosen=a.selected(value)
    assert len(chosen)==36 and {r['source_id'] for r in chosen}=={f's{i}' for i in range(18)}
    value['episodes'].pop()
    with pytest.raises(ValueError):a.selected(value)


def test_complete36_numeric_only_envelope_and_unknown_geometry(tmp_path,monkeypatch):
    rows=a.selected(metadata())
    for i,row in enumerate(rows):
        value=arrays(i);value.update(agentview_rgb=np.zeros(1,np.uint8),eye_in_hand_rgb=np.zeros(1,np.uint8))
        path=tmp_path/(row['problem_id']+'.npz')
        with path.open('wb') as handle:np.savez_compressed(handle,**value)
        row['frames']=a.pin(path)
    caps=dict(npz_parses=36,numeric_member_decodes=216,numeric_scalars=205524,container_bytes_max=2*1024**3,
        rgb_pixel_decodes=0,policy_forwards=0,native_steps=0,checkpoint_reads=0,hdf5_reads=0,initial_bank_reads=0)
    cfg=dict(records=rows,caps=caps,output=str(tmp_path/'results'))
    monkeypatch.setattr(a,'preflight',lambda *_:deepcopy(cfg))
    old_get=np.lib.npyio.NpzFile.__getitem__
    def get(archive,key):
        assert key not in ['agentview_rgb','eye_in_hand_rgb'],'image pixel decode forbidden in numeric phase'
        return old_get(archive,key)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',get)
    config=tmp_path/'config.json';binding=tmp_path/'binding.json';config.write_text('{}');binding.write_text('{}')
    a.run(config,binding)
    result=json.loads((tmp_path/'results/analysis.json').read_text())
    assert result['pairs']==18 and result['trajectories']==36 and result['geometric_unresolved_trajectories']==36
    assert result['actual']['numeric_scalars']==205524 and result['actual']['numeric_member_decodes']==216
    assert result['actual']['rgb_pixel_decodes']==result['actual']['native_steps']==0
    assert all(v['count'] is None for v in result['geometric_categories'].values())
    assert result['availability']['eef_pos'] is False and result['availability']['gripper_open_goal_required'] is False
