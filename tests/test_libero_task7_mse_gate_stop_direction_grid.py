"""Synthetic projections, saved geometry and fresh independent audit only."""
import copy
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
from experiments import libero_task7_mse_gate_stop_direction_grid_protocol as d
from experiments import libero_task7_mse_gate_stop_direction_grid_access as a
from experiments import analyze_libero_task7_mse_gate_stop_direction_grid as main
from experiments import audit_libero_task7_mse_gate_stop_direction_grid as audit


@pytest.fixture
def cfg():return d.create()


def synthetic_rows(cfg):
 cfg=copy.deepcopy(cfg);data={k:[] for k in d.COLUMNS}
 for index,h in enumerate(cfg['heads']):
  ident=a.identity(h);n=cfg['fit_rows'][str(h['fold'])];clip=1 if index==1 else 0;hinge=2 if index==2 else 0
  for condition in cfg['design']['points']:
   stop=condition!='saved500';m=.25-(.00001 if stop and h['cohort']=='short' else 0.);penalty=.01-(.001 if stop and h['cohort']=='short' else 0.)
   t=m/7+penalty
   point={**ident,'condition':condition,'extra_updates':h['stop_extra_updates'] if stop else 0,'n':n,
    'fixed_other6_floor':.02,'component3_mse':m,'penalty':penalty,'trainable_objective':t,'total_objective':.02+t,
    'base_l2':.125,'penalty_l2':.25,'total_l2':.125,'directional_derivative':-.015625+(.001 if hinge else 0.),'branch_digest':'a'*64}
   point.update({f:0 for f in d.COUNTS});point['clip_low_tie_n']=clip;point['clip_inside_n']=n-clip
   point['hinge_zero_n']=hinge;point['hinge_negative_n']=n-hinge
   data['endpoints'].append({f:point[f] for f in d.COLUMNS['endpoints']})
   for j in range(257):
    row={**ident,'condition':condition,'coordinate':j,
     **{f:value if j==0 else 0. for f,value in zip(d.VECTORS,[.125,-.25,-.125,.125])}}
    data['parameters'].append({f:row[f] for f in d.COLUMNS['parameters']})
  stop=next(row for row in data['endpoints'] if a.key(row)==a.key(h) and row['condition']==cfg['design']['points'][1])
  for j in range(12):
   alpha=2.**(-j);m=.25+.001*alpha;t=stop['trainable_objective']-.001*alpha;p=t-m/7
   row={**ident,'optimization_arm':d.parent.ARMS[1],'from_extra_updates':h['stop_extra_updates'],'trial_index':j,'alpha':alpha,
    'component3_mse':m,'penalty':p,'trainable_objective':t,'total_objective':.02+t,'start500_component3_mse':.25,
    'slope':stop['directional_derivative'],'armijo_bound':stop['trainable_objective']+.0001*alpha*stop['directional_derivative'],
    'armijo_pass':True,'mse_envelope_pass':False,'mse_gate_applied':True,'accepted':False,'reason':'mse_envelope_rejected'}
   data['trials'].append({f:row[f] for f in d.COLUMNS['trials']})
 return cfg,data


def csv_fixture(cfg,tmp_path):
 cfg,data=synthetic_rows(cfg);all_heads=cfg['heads']+cfg['full50_metadata_only'];points={}
 for row in data['endpoints']:points[a.key(row),row['condition']]=row
 for name in d.COLUMNS:
  path=tmp_path/(name+'.csv');columns=cfg['inputs'][name]['columns'];rows=[]
  if name in ('endpoints','parameters'):
   for h in all_heads:
    for condition in ['saved500',*[arm+'_final' for arm in d.parent.ARMS]]:
     base={**a.identity(h),'condition':condition}
     if name=='endpoints':
      projected=points.get((a.key(h),condition),{})
      rows.append({f:projected.get(f,base.get(f,'UNSELECTED_NONNUMERIC')) for f in columns})
     else:
      for j in range(257):
       selected=a.key(h) in {a.key(v) for v in cfg['heads']} and condition in cfg['design']['points']
       raw={**base,'coordinate':j}
       if selected:raw.update({f:value if j==0 else 0. for f,value in zip(d.VECTORS,[.125,-.25,-.125,.125])})
       # A weight string cannot be converted, including on selected rows.
       rows.append({f:raw.get(f,'UNPROJECTED_NONNUMERIC') for f in columns})
  else:
   rows=list(data['trials']);filler={**a.identity(cfg['full50_metadata_only'][0]),'optimization_arm':d.parent.ARMS[0]}
   rows += [{f:filler.get(f,'UNSELECTED_NONNUMERIC') for f in columns} for _ in range(10090-132)]
  with path.open('w',newline='') as f:
   writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)
  assert len(rows)==cfg['budget']['source_rows_per_stage'][name]
  cfg['inputs'][name]={**d.p.pin(path),'rows':len(rows),'columns':columns}
 cfg['output']=str(tmp_path/'result.json');cfg['artifacts']=str(tmp_path/'artifacts')
 return cfg


def test_independent_reconstruction_matches_all_outputs(cfg,monkeypatch):
 cfg,data=synthetic_rows(cfg);expected=main.analyze_rows(cfg,data)
 def deny(*args,**kwargs):raise AssertionError('audit reused primary numerical helper')
 for name in ['gradient_geometry','feasibility_evidence','trial_terms']:monkeypatch.setattr(d,name,deny)
 monkeypatch.setattr(d.parent,'trial_decision',deny)
 result=audit.reconstruct(cfg,data);audit.deep_check(audit.Checks(),expected,result,'saved synthetic geometry')
 assert {k:len(v) for k,v in result.items()}==cfg['budget']['tables']
 assert any(h['inference_class']=='boundary_clip_kink_unresolved' for h in result['heads'])
 assert any(h['inference_class']=='boundary_direction_conflict_near_zero_alpha' for h in result['heads'])
 assert sum(h['inference_class']=='positive_slack_local_feasible_by_continuity' for h in result['heads'])==5
 assert any(p['hinge_zero_n']>0 and p['kink_gap']>0 and p['exact_mse_slope'] is not None for p in result['points'])
 assert any(s['n']==0 and s['slack_mean'] is None for s in result['strata'])
 assert all(not t['untried_step_evaluated'] for t in result['trials'])


@pytest.mark.parametrize('fault',['duplicate_coordinate','missing_coordinate','wrong_norm','wrong_total','wrong_direction','branch_count','bad_slope','role','zero_changed','trial_flag','grid','from_update','envelope','trial_terms','duplicate_trial','weight_column'])
def test_malformed_cache_rejected_by_both(cfg,fault):
 cfg,data=synthetic_rows(cfg)
 if fault=='duplicate_coordinate':data['parameters'][1]=dict(data['parameters'][0])
 elif fault=='missing_coordinate':data['parameters'].pop()
 elif fault=='wrong_norm':data['endpoints'][0]['total_l2']+=.1
 elif fault=='wrong_total':data['parameters'][0]['total_gradient']+=.1
 elif fault=='wrong_direction':data['parameters'][0]['direction']+=.1
 elif fault=='branch_count':data['endpoints'][0]['clip_inside_n']-=1
 elif fault=='bad_slope':data['endpoints'][0]['directional_derivative']-=.1
 elif fault=='role':data['endpoints'][0]['condition']='original_armijo_final'
 elif fault=='zero_changed':
  zero=next(h for h in cfg['heads'] if h['cohort']=='zero');row=next(r for r in data['endpoints'] if a.key(r)==a.key(zero) and r['condition']!='saved500');row['branch_digest']='b'*64
 elif fault=='trial_flag':data['trials'][0]['accepted']=True
 elif fault=='grid':data['trials'][0]['alpha']=.5
 elif fault=='from_update':data['trials'][0]['from_extra_updates']+=1
 elif fault=='envelope':data['trials'][0]['start500_component3_mse']+=1e-12
 elif fault=='trial_terms':data['trials'][0]['penalty']+=.01
 elif fault=='duplicate_trial':data['trials'][1]=dict(data['trials'][0])
 else:data['parameters'][0]['parameter']=1.
 for fn in [main.analyze_rows,audit.reconstruct]:
  with pytest.raises(ValueError):fn(cfg,data)


def test_unbound_payload_APIs_fail_before_source_SHA_and_directory(cfg,tmp_path,monkeypatch):
 path=tmp_path/'science.json';path.write_text(json.dumps(cfg));sp=d.p.pin(path);bp={'path':str(tmp_path/'missing.json'),'sha256':'0'*64}
 before=Path(cfg['output']).exists(),Path(cfg['artifacts']).exists();sha0=d.p.sha
 def sha(path):
  assert Path(path).suffix not in ('.csv','.npz','.pt','.h5','.hdf5');return sha0(path)
 monkeypatch.setattr(d.p,'sha',sha)
 with pytest.raises(FileNotFoundError):main.execute_payload(cfg,sp,bp)
 with pytest.raises(FileNotFoundError):audit.audit_payload(cfg,sp,bp)
 assert before==(Path(cfg['output']).exists(),Path(cfg['artifacts']).exists())


@pytest.mark.parametrize('operation',['raw','write','repeat','projection','parameter_column','states','os_open','fileio','archive_sha','numpy','torch','direct_reader'])
def test_guard_denies_unlisted_access(cfg,tmp_path,operation):
 cfg=csv_fixture(cfg,tmp_path);guard=a.Guard(cfg,limits=a.base.Limits(120,2**40));path=Path(cfg['inputs']['parameters']['path'])
 with pytest.raises(PermissionError):
  with guard:
   if operation=='raw':path.read_text()
   elif operation=='write':path.open('w')
   elif operation=='repeat':list(guard.project('parameters'));list(guard.project('parameters'))
   elif operation=='projection':list(guard.project('parameters',['task']))
   elif operation=='parameter_column':list(guard.project('parameters',d.COLUMNS['parameters']+['parameter']))
   elif operation=='states':list(guard.project('states'))
   elif operation=='os_open':os.open(path,os.O_RDONLY)
   elif operation=='fileio':__import__('io').FileIO(path)
   elif operation=='archive_sha':d.p.sha(tmp_path/'forbidden.npz')
   elif operation in ('numpy','torch'):__import__(operation)
   else:
    guard.active_path=str(path)
    with path.open() as f:csv.reader(f)
 assert guard.denied


def test_lexical_selection_does_not_convert_unused_columns_or_rows(cfg,tmp_path):
 cfg=csv_fixture(cfg,tmp_path);guard=a.Guard(cfg,limits=a.base.Limits(120,2**40))
 with guard:
  data={name:list(guard.project(name)) for name in d.COLUMNS};main.analyze_rows(cfg,data);guard.complete()
 record=guard.record()
 assert record['selected_rows']=={'endpoints':22,'parameters':5654,'trials':132}
 assert record['lexical_cells']==380100 and sum(record['cells'].values())==65538
 assert record['parameter_weight_column_numeric_converted']==record['unselected_rows_or_columns_numeric_converted']==0
 assert len(record['byte_SHA_checks'])==6 and record['source_CSV_passes']['completed']==3


def test_source_byte_change_fails_before_numeric_decode(cfg,tmp_path):
 cfg=csv_fixture(cfg,tmp_path);path=Path(cfg['inputs']['endpoints']['path']);path.write_text(path.read_text()+'\n')
 guard=a.Guard(cfg,limits=a.base.Limits(120,2**40))
 with pytest.raises(ValueError):
  with guard:list(guard.project('endpoints'))
 assert guard.completed==[] and guard.selected_rows['endpoints']==0


def test_full_synthetic_projection_analysis_and_fresh_audit(cfg,tmp_path):
 cfg=csv_fixture(cfg,tmp_path);path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 for i,(name,entry) in enumerate([('analyze_libero_task7_mse_gate_stop_direction_grid','execute_payload'),('audit_libero_task7_mse_gate_stop_direction_grid','audit_payload')]):
  code=f'''import json
from pathlib import Path
from experiments import libero_task7_mse_gate_stop_direction_grid_access as a
from experiments import {name} as module
cfg=json.load(open({str(path)!r}))
def synthetic_only(cfg,sp,bp):
 assert all(Path(ref['path']).is_relative_to(Path({str(tmp_path)!r})) for ref in cfg['inputs'].values())
a.validate_call=synthetic_only
pin={{'path':'synthetic-only','sha256':{'a'*64!r}}}
module.{entry}(cfg,pin,pin)
'''
  result=subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=135)
  (tmp_path/f'child-{i}.log').write_text(result.stdout);assert result.returncode==0,result.stdout
 record=json.loads(Path(cfg['output']).read_text());done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
 assert done['total_source_CSV_passes']==6 and done['total_projected_cells']==131076 and done['total_source_SHA']==12
 assert sum(v['rows'] for v in record['tables'].values())==177
 assert record['access']['archive_head_checkpoint_reads_including_SHA']==0
 assert record['cost']['peak_RSS_bytes']<=268435456 and record['cost']['elapsed_seconds']<=120


def test_zip_member_reader_is_blocked_without_raw_IO(cfg,tmp_path):
 import zipfile
 cfg=csv_fixture(cfg,tmp_path);guard=a.Guard(cfg,limits=a.base.Limits(120,2**40))
 with pytest.raises(PermissionError):
  with guard:zipfile.ZipFile.open(None,'forbidden.npy')
 assert guard.denied[-1]['kind']=='array_model_or_alternate_reader'


def test_physical_row_budget_rejects_extra_unselected_record(cfg,tmp_path):
 cfg=csv_fixture(cfg,tmp_path);path=Path(cfg['inputs']['trials']['path'])
 with path.open() as f:rows=list(csv.reader(f))
 with path.open('a',newline='') as f:csv.writer(f).writerow(rows[-1])
 cfg['inputs']['trials'].update(d.p.pin(path))
 guard=a.Guard(cfg,limits=a.base.Limits(120,2**40))
 with pytest.raises(ValueError):
  with guard:list(guard.project('trials'))
 assert guard.rows['trials']==10091 and guard.selected_rows['trials']==132 and guard.completed==[]
