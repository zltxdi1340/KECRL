"""Only synthetic scalar paths/CSVs; source experiment files stay closed."""
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from collections import Counter
import pytest
from experiments import libero_task7_optimization_path_decomposition_protocol as d
from experiments import libero_task7_optimization_path_decomposition_access as a
from experiments import analyze_libero_task7_optimization_path_decomposition as main
from experiments import audit_libero_task7_optimization_path_decomposition as audit


@pytest.fixture
def cfg():return d.create()


def synthetic_rows(cfg):
 cfg=copy.deepcopy(cfg);data={name:[] for name in d.COLUMNS};global_round=0;lambda1_index=0
 for index,ident in enumerate(cfg['head_identities']):
  coefficient=d.CASES[ident['mode']];obj=index%4 in (0,2);grad=index%4 in (0,1)
  t0=.02;drop=.0004 if obj else .0001;gend=.005 if grad else .02;up=coefficient==1 and lambda1_index<11
  if coefficient:lambda1_index+=1
  n=cfg['fit_rows'][str(ident['fold'])];states=[];rounds=[]
  for k in range(51):
   fraction=k/50;t=t0-drop*fraction;g=.1+(gend-.1)*fraction
   mse7=t if coefficient==0 else .015+(drop/2 if up else -drop*.75)*fraction
   penalty=.002-.0005*fraction if coefficient==0 else t-mse7
   row={**ident,'extra_updates':k,'n':n,'fixed_other6_floor':.03,'component3_mse':mse7*7,'penalty':penalty,'trainable_objective':t,'total_objective':.03+t,
    'base_l2':g if coefficient==0 else .6*g,'penalty_l2':.002 if coefficient==0 else .4*g,'total_l2':g,'directional_derivative':-g*g}
   row.update({name:0 for name in d.COUNTS});row['clip_inside_n']=n;row['hinge_positive_n']=100;row['hinge_negative_n']=n-100
   states.append(row);data['states'].append(row)
  for k in range(50):
   count=7 if global_round<1485 else 6;global_round+=1;round_=[]
   for j in range(count):
    alpha=2.**(-j);accepted=j==count-1;t=states[k+1]['trainable_objective'] if accepted else states[k]['trainable_objective']
    trial={**ident,'from_extra_updates':k,'trial_index':j,'alpha':alpha,'trainable_objective':t,'total_objective':t+.03,'slope':states[k]['directional_derivative'],
     'armijo_bound':states[k]['trainable_objective']+1e-4*alpha*states[k]['directional_derivative'],'accepted':accepted}
    round_.append(trial);data['trials'].append(trial)
   rounds.append(round_)
  for condition,k in [('hold_500',0),('fullbatch_armijo',50)]:data['endpoints'].append({**states[k],'condition':condition})
  tr=states[50]['trainable_objective']/states[0]['trainable_objective'];gr=states[50]['total_l2']/states[0]['total_l2'];of=tr<=.99;gf=gr<=.1
  data['heads'].append({**ident,'extra_updates':50,'trials':sum(map(len,rounds)),'terminal_reason':'update_cap',
   'initial_objective':states[0]['total_objective'],'final_objective':states[50]['total_objective'],
   'initial_trainable_objective':states[0]['trainable_objective'],'final_trainable_objective':states[50]['trainable_objective'],
   'initial_gradient_l2':states[0]['total_l2'],'final_gradient_l2':states[50]['total_l2'],'trainable_ratio':tr,'gradient_ratio':gr,
   'objective_target_met':of,'gradient_target_met':gf,'joint_cue':of and gf,'status':'joint_optimization_cue' if of and gf else 'partial_target_reduction'})
 counts=dict(Counter(d.goal_class(h['objective_target_met'],h['gradient_target_met']) for h in data['heads']))
 cfg['expected_counts']={'heads':36,'lambda1_mse_up':11,'nonjoint':sum(not h['joint_cue'] for h in data['heads']),'goal_classes':counts}
 return cfg,data


def test_budget_and_metadata_no_source_csv_sha(cfg,monkeypatch):
 sha0=d.p.sha
 def sha(path):
  assert Path(path).suffix in ('.py','.json','.md','.toml');return sha0(path)
 monkeypatch.setattr(d.p,'sha',sha);fresh=d.create()
 assert fresh['budget']['total_projected_cells']==412578 and fresh['budget']['total_source_SHA']==16
 assert sum(fresh['budget']['rows_per_stage'].values())==14229 and sum(fresh['budget']['tables'].values())==2304
 assert {k:len(v) for k,v in fresh['projections'].items()}=={'states':24,'trials':13,'endpoints':25,'heads':20}
 assert fresh['scope']['new_training_forward_gradient_environment_or_sealed_calls']==0


@pytest.mark.parametrize('section,key,value',[
 ('scope','future_stabilization_steps',0),('scope','probe_source_query_validation_heldout_qualification_reserved_final_access',True),
 ('design','anchors',[0,25,50]),('execution','technical_retries',1),('execution','absolute_tolerance',.01),
 ('budget','total_source_CSV_passes',12),('budget','total_projected_cells',0),('expected_counts','nonjoint',20),
])
def test_freeze_rejects_changed_analysis(cfg,tmp_path,section,key,value):
 cfg[section][key]=value;p=tmp_path/'changed.json';p.write_text(json.dumps(cfg))
 with pytest.raises(ValueError):d.validate(p)


def test_unbound_apis_fail_before_source_reads(cfg,tmp_path,monkeypatch):
 path=tmp_path/'science.json';path.write_text(json.dumps(cfg));sp=d.p.pin(path);bp={'path':str(tmp_path/'missing.json'),'sha256':'0'*64};before=Path(cfg['output']).exists(),Path(cfg['artifacts']).exists()
 sha0=d.p.sha
 def sha(path):
  assert Path(path).suffix not in ('.csv','.npz','.pt','.h5','.hdf5');return sha0(path)
 monkeypatch.setattr(d.p,'sha',sha)
 with pytest.raises(FileNotFoundError):main.execute_payload(cfg,sp,bp)
 with pytest.raises(FileNotFoundError):audit.audit_payload(cfg,sp,bp)
 assert before==(Path(cfg['output']).exists(),Path(cfg['artifacts']).exists())


@pytest.mark.parametrize('values,kind',[
 ((.14,.01,.03,.147,.008,.029,1.),'mse_up_penalty_down'),
 ((.14,.01,.03,.133,.009,.028,1.),'both_down'),
 ((.14,.01,.03,.126,.011,.029,1.),'mse_down_penalty_up'),
 ((.14,.01,.02,.133,.5,.019,0.),'mse_down_penalty_up'),
])
def test_delta_conservation_and_lambda0_penalty_exclusion(values,kind):
 row=d.terms(*values);assert row['step_class']==kind and abs(row['conservation_residual'])<1e-10
 if values[-1]==0:assert row['weighted_delta_penalty']==0


def test_bad_conservation_and_ratios_fail_closed():
 with pytest.raises(ValueError):d.terms(.14,.01,.03,.147,.008,.03,1.)
 assert d.safe_ratio(1.,0.) is None
 with pytest.raises(ValueError):d.safe_ratio(math.nan,1.)
 with pytest.raises(ValueError):d.safe_ratio(1e308,1e-308)


def test_independent_full_scalar_reconstruction(cfg):
 cfg,data=synthetic_rows(cfg);left=main.analyze_rows(cfg,data);right=audit.reconstruct(cfg,data)
 audit.deep_check(audit.Checks(),left,right,'synthetic paths')
 audit.deep_check(audit.Checks(),main.summaries(left),audit.summaries(right),'synthetic summaries')
 assert {k:len(v) for k,v in left.items()}==cfg['budget']['tables']
 assert any(r['n']==0 and r['delta_trainable_mean'] is None for r in left['strata'])
 up=[h for h in left['heads'] if h['lambda1_mse_up']]
 assert len(up)==11 and all(0<h['compensation_ratio']<1 for h in up)
 assert all(h['weighted_delta_penalty']==0 for h in left['heads'] if h['mode']=='uniform_affine')


@pytest.mark.parametrize('fault',['conservation','goal','first_accept','alpha','missing_state','duplicate_endpoint','branch','trial_alignment','unknown_head'])
def test_bad_saved_path_rejected_by_both_implementations(cfg,fault):
 cfg,data=synthetic_rows(cfg)
 if fault=='conservation':data['states'][2]['component3_mse']+=.1
 elif fault=='goal':data['heads'][0]['joint_cue']=not data['heads'][0]['joint_cue']
 elif fault=='first_accept':data['trials'][0]['accepted']=True
 elif fault=='alpha':data['trials'][0]['alpha']=.5
 elif fault=='missing_state':data['states'].pop(20)
 elif fault=='duplicate_endpoint':data['endpoints'].append(dict(data['endpoints'][0]))
 elif fault=='branch':data['states'][0]['clip_inside_n']-=1
 elif fault=='trial_alignment':
  row=next(t for t in data['trials'] if t['accepted']);row['trainable_objective']-=1e-4;row['total_objective']-=1e-4
 else:data['heads'][0]['task']=8
 for fn in [main.analyze_rows,audit.reconstruct]:
  with pytest.raises(ValueError):fn(cfg,data)


def csv_fixture(cfg,tmp_path):
 cfg,data=synthetic_rows(cfg)
 for name,rows in data.items():
  columns=cfg['inputs'][name]['columns'];path=tmp_path/(name+'.csv')
  with path.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
   for row in rows:writer.writerow({k:row[k] if k in d.COLUMNS[name] else 'UNPROJECTED_NONNUMERIC' for k in columns})
  cfg['inputs'][name]={**d.p.pin(path),'rows':len(rows),'columns':columns}
 cfg['output']=str(tmp_path/'result.json');cfg['artifacts']=str(tmp_path/'artifacts');return cfg


@pytest.mark.parametrize('operation',['raw','write','repeat','projection','os_open','fileio','other_file','numpy','torch','direct_reader'])
def test_projection_guard_denies_bypass(cfg,tmp_path,operation):
 cfg=csv_fixture(cfg,tmp_path);guard=a.Guard(cfg,limits=a.base.Limits(120,2**40));path=Path(cfg['inputs']['heads']['path'])
 with pytest.raises(PermissionError):
  with guard:
   if operation=='raw':path.read_text()
   elif operation=='write':path.open('w')
   elif operation=='repeat':list(guard.project('heads'));list(guard.project('heads'))
   elif operation=='projection':list(guard.project('heads',['task']))
   elif operation=='os_open':os.open(path,os.O_RDONLY)
   elif operation=='fileio':__import__('io').FileIO(path)
   elif operation=='other_file':Path('/tmp/forbidden-policy.npz').read_bytes()
   elif operation in ('numpy','torch'):__import__(operation)
   else:
    guard.active_path=str(path)
    with path.open() as f:csv.reader(f)
 assert guard.denied


def test_fixed_projection_does_not_convert_other_columns(cfg,tmp_path):
 cfg=csv_fixture(cfg,tmp_path);guard=a.Guard(cfg,limits=a.base.Limits(120,2**40))
 with guard:
  rows={name:list(guard.project(name)) for name in d.COLUMNS};guard.complete()
 assert len(rows['states'])==1836 and guard.record()['unselected_CSV_columns_numeric_converted']==0
 assert guard.record()['lexical_cells']==219609 and sum(guard.record()['cells'].values())==206289


def test_full_synthetic_fresh_readonly_analysis_and_audit(cfg,tmp_path):
 cfg=csv_fixture(cfg,tmp_path);p=tmp_path/'science.json';p.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 for i,(name,entry) in enumerate([('analyze_libero_task7_optimization_path_decomposition','execute_payload'),('audit_libero_task7_optimization_path_decomposition','audit_payload')]):
  code=f'''import json
from pathlib import Path
from experiments import libero_task7_optimization_path_decomposition_access as a
from experiments import {name} as module
cfg=json.load(open({str(p)!r}))
def synthetic_only(cfg,sp,bp):
    assert all(Path(r['path']).is_relative_to(Path({str(tmp_path)!r})) for r in cfg['inputs'].values())
a.validate_call=synthetic_only
pin={{'path':'synthetic-only','sha256':{'a'*64!r}}}
module.{entry}(cfg,pin,pin)
'''
  proc=subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=135)
  (tmp_path/f'child-{i}.log').write_text(proc.stdout);assert proc.returncode==0,proc.stdout
 r=json.loads(Path(cfg['output']).read_text());done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
 assert done['total_source_CSV_passes']==8 and done['total_projected_cells']==412578 and done['total_source_SHA']==16
 assert sum(t['rows'] for t in r['tables'].values())==2304
