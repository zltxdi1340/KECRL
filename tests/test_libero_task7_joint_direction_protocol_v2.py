"""Metadata freeze and exact v2 budget identities; no research array access."""
import builtins,copy,io,json
from pathlib import Path
import pytest
from experiments import libero_task7_joint_direction_protocol_v2 as d

@pytest.fixture(scope='module')
def canonical():return d.create()


def test_same18_same_optimization_and_decisions(canonical):
 old=json.loads(d.old.DEFAULT_CONFIG.read_text())
 for field in ('inputs','optimization','decision','reference_points','cohorts','historical_control'):
  assert canonical[field]==old[field]
 assert len(canonical['inputs']['heads'])==18 and canonical['version']==2 and canonical['formal_result'] is False
 assert canonical['audit_contract']['S'].startswith('Torch_targets')
 assert canonical['scope']['original_Policy_updates']==500 and canonical['scope']['future_stabilization_steps']==5


def test_full18_budget_includes_S_and_all_shadow_trials(canonical):
 b=canonical['budget'];assert b['total_gradient_states_max']==5508 and b['audit_total_autograd_max']==11016
 assert b['total_auxiliary_updates_max']==3600 and b['total_trial_evaluations_max']==108000
 assert b['total_linear_forms_max']==120852 and b['total_linear_rows_max']==337418784
 assert b['total_linear_feature_multiplications_max']==337418784*256
 assert (b['total_NPZ_loads'],b['total_member_decodes'],b['total_archive_SHA_checks'])==(68,574,172)
 assert (b['new_NPZ_writes'],b['new_member_writes'])==(36,630)
 assert len(canonical['new_head_members'])==14 and len(canonical['audit_head_members'])==21
 assert b['total_direction_constructions_max']==5508 and b['total_direction_norms_max']==22032
 assert b['local_vector_coordinate_checks_max']==1836*257*4
 assert b['total_CSV_rows_max']==89676 and b['local_S_updates']==0


@pytest.mark.parametrize('section,key,value',[
 ('execution','local_absolute_tolerance',1e-8),('execution','technical_retries',1),
 ('budget','local_S_updates',1),('budget','total_trial_evaluations_max',72000),
 ('optimization','extra_updates_max',100),('audit_contract','old_v1_remains_failed',False),
 ('scope','original_Policy_updates',1000),('scope','future_stabilization_steps',0),
])
def test_mutation_fails_closed(canonical,tmp_path,monkeypatch,section,key,value):
 changed=copy.deepcopy(canonical);changed[section][key]=value;p=tmp_path/'bad.json';p.write_text(json.dumps(changed))
 monkeypatch.setattr(d,'create',lambda:canonical)
 with pytest.raises(ValueError):d.validate(p)


def test_metadata_factory_denies_payload(monkeypatch,canonical):
 saved=(builtins.open,io.open,d.p.sha)
 def check(path):
  if isinstance(path,(str,Path)):assert Path(path).suffix not in ('.csv','.npz','.npy','.pt','.h5','.hdf5')
 def op(path,*a,**kw):check(path);return saved[0](path,*a,**kw)
 def ip(path,*a,**kw):check(path);return saved[1](path,*a,**kw)
 def sha(path):check(path);return saved[2](path)
 monkeypatch.setattr(builtins,'open',op);monkeypatch.setattr(io,'open',ip);monkeypatch.setattr(d.p,'sha',sha)
 monkeypatch.setattr(d.old,'direction',lambda *a:(_ for _ in ()).throw(AssertionError('no real directions')))
 assert d.create()==canonical


def test_unbound_rejects_before_archive(canonical,tmp_path,monkeypatch):
 p=tmp_path/'science.json';p.write_text(json.dumps(canonical));monkeypatch.setattr(d,'create',lambda:canonical)
 with pytest.raises(FileNotFoundError):d.require_binding(p,tmp_path/'missing.json')
