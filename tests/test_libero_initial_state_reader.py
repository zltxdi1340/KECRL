import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch

from src.environments.libero_initial_state_reader import inspect_initial_bank_header, load_initial_state_rows


def bank(tmp_path, values=None):
    values = np.arange(50*92,dtype=np.float64).reshape(50,92) if values is None else values
    path = tmp_path/'synthetic.pruned_init'
    with path.open('wb') as handle:
        torch.save(values,handle,pickle_protocol=2)
    pin = {'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    pin.update(inspect_initial_bank_header(path,pin))
    return path,pin,values


@pytest.mark.parametrize('start,stop',[(0,20),(10,30),(20,50)])
def test_only_authorized_rows_are_numerically_decoded_and_indices_are_preserved(tmp_path,monkeypatch,start,stop):
    path,pin,values=bank(tmp_path)
    original=np.frombuffer;seen=[]
    def tracked(raw,*args,**kwargs):
        seen.append(len(raw));return original(raw,*args,**kwargs)
    monkeypatch.setattr(np,'frombuffer',tracked)
    states,record=load_initial_state_rows(path,pin,range(start,stop),authorized_indices=range(start,stop))
    assert np.array_equal(states,values[start:stop])
    assert seen==[(stop-start)*92*8]
    assert record['numeric_rows_decoded']==list(range(start,stop))
    assert record['global_to_local_index']=={str(i):i-start for i in range(start,stop)}
    assert record['skipped_rows_numerically_decoded']==[] and not record['skipped_rows_buffered']
    assert not record['used_pickle_or_torch_deserialization']


def test_metadata_inspection_never_interprets_a_numpy_vector(tmp_path,monkeypatch):
    path,pin,_=bank(tmp_path)
    monkeypatch.setattr(np,'frombuffer',lambda *a,**kw:pytest.fail('header interpreted a state'))
    assert inspect_initial_bank_header(path,pin)['numeric_rows_decoded']==[]


@pytest.mark.parametrize('indices',[[0,2],list(range(21)),list(range(20,40)),[],[True],[-1]])
def test_bad_or_unauthorized_ranges_fail_before_any_container_access(tmp_path,monkeypatch,indices):
    monkeypatch.setattr(Path,'stat',lambda *a,**kw:pytest.fail('invalid scope opened container'))
    with pytest.raises(ValueError,match='permitted contiguous'):
        load_initial_state_rows(tmp_path/'missing',{},indices,authorized_indices=list(range(20)))


def test_whole_pickle_or_torch_deserialization_is_never_used(tmp_path,monkeypatch):
    path,pin,_=bank(tmp_path)
    import pickle
    monkeypatch.setattr(pickle,'load',lambda *a,**kw:pytest.fail('pickle.load used'))
    monkeypatch.setattr(pickle,'loads',lambda *a,**kw:pytest.fail('pickle.loads used'))
    monkeypatch.setattr(torch,'load',lambda *a,**kw:pytest.fail('torch.load used'))
    states,_=load_initial_state_rows(path,pin,range(10,30),authorized_indices=range(10,30))
    assert states.shape==(20,92)


def test_skipped_nonfinite_rows_do_not_get_interpreted_as_state_values(tmp_path):
    values=np.arange(50*92,dtype=np.float64).reshape(50,92);values[:10]=np.nan;values[30:]=np.nan
    path,pin,_=bank(tmp_path,values)
    states,_=load_initial_state_rows(path,pin,range(10,30),authorized_indices=range(10,30))
    assert np.isfinite(states).all()


def test_selected_nonfinite_rows_are_rejected(tmp_path):
    values=np.zeros((50,92));values[10,0]=np.nan
    path,pin,_=bank(tmp_path,values)
    with pytest.raises(ValueError,match='nonfinite'):
        load_initial_state_rows(path,pin,range(10,30),authorized_indices=range(10,30))


def test_unsupported_shape_is_rejected_in_metadata_before_row_decoding(tmp_path):
    values=np.zeros((50,93));path=tmp_path/'wrong.pruned_init'
    with path.open('wb') as handle:
        torch.save(values,handle,pickle_protocol=2)
    pin={'size_bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError,match='shape or dtype'):
        inspect_initial_bank_header(path,pin)


@pytest.mark.parametrize('change',['container','header'])
def test_hash_changes_fail_closed(tmp_path,change):
    path,pin,_=bank(tmp_path)
    if change=='container':pin['sha256']='0'*64
    else:pin['pickle_header_sha256']='0'*64
    with pytest.raises(ValueError):load_initial_state_rows(path,pin,range(20),authorized_indices=range(20))
