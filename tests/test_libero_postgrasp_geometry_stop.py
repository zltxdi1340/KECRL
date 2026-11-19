"""Selective pixel decoding at a rejected frame, using invented compressed arrays."""
import numpy as np
import pytest

from experiments import analyze_libero_postgrasp_geometry_stop as s


def test_only_admitted_frames_are_exposed(tmp_path):
    values=np.full((301,128,128,3),17,np.uint8);values[153,2,3,1]=18
    arrays={key:np.zeros(1) for key in s.p.previous.MEMBERS};arrays['eye_in_hand_rgb']=values
    path=tmp_path/'invented.npz';np.savez_compressed(path,**arrays)
    selected,counts=s.selected_rgb(s.p.pin(path),[152,153],301)
    assert selected.shape==(2,128,128,3) and selected[1,2,3,1]==18 and selected[0,2,3,1]==17
    assert counts['pixel_values_decoded']==98304 and counts['member_uncompressed_bytes_consumed']<154*49152+256
    result=s.quantify(np.full_like(selected,17),selected,[152,153])
    assert result[0]['exact'] and result[1]['different_channels']==result[1]['different_pixels']==1
    assert result[1]['maximum_channel_error']==1 and result[1]['first_differences'][0]['delta']==1


@pytest.mark.parametrize('frames',[[0,153],[153],[152,153,154]])
def test_disallowed_frames_rejected_before_archive_access(frames):
    with pytest.raises(ValueError):s.selected_rgb(dict(path='/does/not/exist',sha256='0'*64),frames,301)


def test_sha_rejection_before_decode(tmp_path):
    path=tmp_path/'invented.npz';path.write_bytes(b'not opened as ZIP')
    with pytest.raises(ValueError):s.selected_rgb(dict(path=str(path),sha256='0'*64),[152,153],301)


def test_journal_control_kind_and_isolated_wrist_mismatch():
    original=[dict(problem_id='invented',initial_state_index=5,step=i,kind='control',outcome=None,
        component_sha256={'eye_in_hand_rgb':'old','joint_states':'same'},observation_sha256='original') for i in [152,153]]
    new=[dict(r,kind='suffix',exact_public=True) for r in original]
    new[1]=dict(new[1],component_sha256={'eye_in_hand_rgb':'new','joint_states':'same'},observation_sha256='changed',exact_public=False)
    result=s.journal_comparison(original,new)
    assert result['public_events']==2 and result['matching_public_events']==1
    assert result['differences']==[dict(problem_id='invented',step=153,components=['eye_in_hand_rgb'])]


def test_missing_original_journal_event_stays_closed():
    row=dict(problem_id='invented',initial_state_index=5,step=153,kind='suffix')
    with pytest.raises(ValueError):s.journal_comparison([], [row])
