"""Reject reuse across incompatible endpoint provenance; no numeric payload."""
from copy import deepcopy
from pathlib import Path
import pytest
from experiments import audit_libero_source_expert_prediction_cache_availability as audit


def endpoint():
    return {'task':0,'seed':20261106,'arm':'A_original','checkpoint_update':500,
            'checkpoint':{'sha256':'a'*64,'weight_hash':'b'*64}}


def test_identical_endpoint_can_match_old_metadata_field_names():
    cached={'task_id':0,'training_seed':20261106,'arm':'A_original',
            'checkpoint':{'task_id':0,'training_seed':20261106,'arm':'A_original','optimizer_updates':500,'sha256':'a'*64,'weight_hash':'b'*64}}
    assert audit.matches(endpoint(),cached)


@pytest.mark.parametrize('key,value',[('seed',20261022),('task',1),('arm','B_fomaml')])
def test_identity_difference_rejected_even_with_same_hashes(key,value):
    other=endpoint();other[key]=value
    assert not audit.matches(endpoint(),other)


@pytest.mark.parametrize('key',['sha256','weight_hash'])
def test_same_task_seed_arm_not_enough_when_checkpoint_or_weight_differs(key):
    other=endpoint();other['checkpoint'][key]='c'*64
    assert not audit.matches(endpoint(),other)


def test_missing_weight_provenance_fails_closed():
    other=endpoint();del other['checkpoint']['weight_hash']
    with pytest.raises(KeyError):audit.matches(endpoint(),other)


def test_wrong_update_not_silently_reused():
    other=endpoint();other['checkpoint_update']=1000
    with pytest.raises(ValueError,match='update'):audit.matches(endpoint(),other)


def test_existing_result_rejected_before_metadata_load(tmp_path):
    p=tmp_path/'old.json';p.write_text('preserve')
    with pytest.raises(ValueError,match='existing'):audit.audit(p)
    assert p.read_text()=='preserve'
