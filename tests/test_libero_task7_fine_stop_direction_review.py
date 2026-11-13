"""Saved-value projection/20-trial checks; no research arrays in tests."""
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
from experiments import libero_task7_fine_stop_direction_review_protocol as d
from experiments import libero_task7_fine_stop_direction_review_access as a
from experiments import analyze_libero_task7_fine_stop_direction_review as main
from experiments import audit_libero_task7_fine_stop_direction_review as audit


@pytest.fixture(scope='module')
def canonical():return d.create()


@pytest.fixture
def cfg(canonical):return copy.deepcopy(canonical)


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
  for j in range(20):
   alpha=2.**(-j);m=.25+.001*alpha;t=stop['trainable_objective']-.001*alpha;p=t-m/7
   row={**ident,'optimization_arm':d.parent.ARMS[1],'from_extra_updates':h['stop_extra_updates'],'trial_index':j,'alpha':alpha,
    'component3_mse':m,'penalty':p,'trainable_objective':t,'total_objective':.02+t,'start500_component3_mse':.25,
    'slope':stop['directional_derivative'],'armijo_bound':stop['trainable_objective']+.0001*alpha*stop['directional_derivative'],
    'armijo_pass':True,'mse_envelope_pass':False,'mse_gate_applied':True,'accepted':False,'reason':'mse_envelope_rejected'}
   data['trials'].append({f:row[f] for f in d.COLUMNS['trials']})
 return cfg,data


@pytest.mark.parametrize('fault',['duplicate_coordinate','missing_coordinate','wrong_norm','wrong_total','wrong_direction','branch_count','bad_slope','role','trial_flag','grid','from_update','envelope','trial_terms','duplicate_trial','weight_column'])
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
 elif fault=='trial_flag':data['trials'][0]['accepted']=True
 elif fault=='grid':data['trials'][0]['alpha']=.5
 elif fault=='from_update':data['trials'][0]['from_extra_updates']+=1
 elif fault=='envelope':data['trials'][0]['start500_component3_mse']+=1e-12
 elif fault=='trial_terms':data['trials'][0]['penalty']+=.01
 elif fault=='duplicate_trial':data['trials'][1]=dict(data['trials'][0])
 else:data['parameters'][0]['parameter']=1.
 for fn in [main.analyze_rows,audit.reconstruct]:
  with pytest.raises(ValueError):fn(cfg,data)


def parent_fixture(cfg,tmp_path):
 cfg,data=synthetic_rows(cfg);focus={a.key(h):h for h in cfg['heads']}
 allids=[dict(task=7,seed=s,arm=arm,fold=f,mode=d.parent.source.MODE) for s in d.p.SEEDS for arm in d.p.ARMS for f in range(3)]
 remaining=[h for h in allids if a.key(h) not in focus]
 bypoint={(a.key(r),r['condition']):r for r in data['endpoints']}
 params={(a.key(r),r['condition'],r['coordinate']):r for r in data['parameters']}
 endpoints=[];pairs=[];heads=[]
 for h in allids:
  k=a.key(h)
  for c in ('saved500','coarse12_final','fine20_final'):
   row={**h,'condition':c,'parameters':'NOT_FOR_NUMERICS'}
   if (k,c) in bypoint:
    row.update(bypoint[k,c]);row.update({v:[params[k,c,j][v] for j in range(257)] for v in d.VECTORS})
   endpoints.append(row)
  is_focus=k in focus;label='short' if is_focus else 'zero' if h in remaining[:6] else 'full50'
  count=focus[k]['stop_extra_updates'] if is_focus else 0 if label=='zero' else 50
  pairs.append(dict(h,cohort=label,candidate_updates=count,fine_updates=count,candidate_stop='line_search_exhausted' if count<50 else 'update_cap',grid_opened=is_focus,prefix_valid=True))
  heads.extend([dict(h,optimization_arm=opt) for opt in ('coarse12','fine20')])
 filler=dict(remaining[0],optimization_arm='fine20',from_extra_updates=0,unselected='NOT_NUMERIC')
 raw=dict(status='complete_mse_guard_finer_grid_pending_audit',formal_result=False,science_config=d.PARENTS['science'],binding=d.PARENTS['binding'],
  original_Policy_CNN_environment_or_sealed_calls=0,endpoints=endpoints,heads=heads,pairs=pairs,
  trials=data['trials']+[filler.copy() for _ in range(6569-100)])
 path=tmp_path/'saved-parent.json';path.write_text(json.dumps(raw));cfg['inputs']['source_JSON']=d.p.pin(path)
 cfg['output']=str(tmp_path/'result.json');cfg['artifacts']=str(tmp_path/'artifacts');return cfg


def test_saved_main_and_independent_reference_match_including_kinks(cfg):
 cfg,data=synthetic_rows(cfg);left=main.analyze_rows(cfg,data);right=audit.reconstruct(cfg,data)
 audit.deep_check(audit.Checks(),left,right,'all saved decompositions')
 assert {k:len(v) for k,v in left.items()}=={'points':10,'trials':100,'heads':5,'strata':6}
 assert any(x['stop_exact_mse_slope'] is None for x in left['heads'])
 assert any(x['n']==0 and x['slack_mean'] is None for x in left['strata'])
 assert all(t['last_trials']==20 for t in left['heads'])
 assert all(not t['untried_step_evaluated'] for t in left['trials'])


def test_positive_slack_fine_grid_ratio_is_not_old_grid():
 r=d.feasibility_evidence(.25,.25-1e-8,-.01,.125)
 assert r['linear_scale_over_smallest_grid_alpha']==r['linear_alpha_scale']/(2.**-19)
 assert r['strict_slack_sign']=='positive' and not r['untried_step_evaluated']


def test_sub_tolerance_excess_is_still_rejected(cfg):
 cfg,data=synthetic_rows(cfg);t=data['trials'][19];t['component3_mse']=.25+5e-12;t['penalty']=t['trainable_objective']-t['component3_mse']/7
 for fn in (main.analyze_rows,audit.reconstruct):
  rows=fn(cfg,data);row=rows['trials'][19]
  assert 0<row['mse_gate_excess']<1e-10 and not row['mse_envelope_pass'] and not row['accepted']


@pytest.mark.parametrize('field,value',[('trial_index',20),('alpha',2.**-20),('reason','armijo_rejected'),('mse_gate_applied',False)])
def test_twentieth_trial_cannot_change_role(cfg,field,value):
 cfg,data=synthetic_rows(cfg);data['trials'][19][field]=value
 for fn in (main.analyze_rows,audit.reconstruct):
  with pytest.raises(ValueError):fn(cfg,data)


@pytest.mark.parametrize('operation',['raw','write','repeat','projection','parameter','os_open','fileio','NPZ_SHA','CSV_SHA','numpy','torch','audit_write'])
def test_guard_blocks_unlisted_payloads(cfg,tmp_path,operation):
 cfg=parent_fixture(cfg,tmp_path);out=tmp_path/'table.csv';g=a.Guard(cfg,[out],limits=a.base.Limits(120,2**40),audit=operation=='audit_write');path=Path(cfg['inputs']['source_JSON']['path'])
 with pytest.raises(PermissionError):
  with g:
   if operation=='raw':path.read_text()
   elif operation=='write':path.open('w')
   elif operation=='repeat':g.project();g.project()
   elif operation in ('projection','parameter'):g.project({**d.COLUMNS,'parameters':d.COLUMNS['parameters']+['parameter']})
   elif operation=='os_open':os.open(path,os.O_RDONLY)
   elif operation=='fileio':__import__('io').FileIO(path)
   elif operation in ('NPZ_SHA','CSV_SHA'):d.p.sha(tmp_path/('forbidden.npz' if operation=='NPZ_SHA' else 'forbidden.csv'))
   elif operation in ('numpy','torch'):__import__(operation)
   else:out.open('w')
 assert g.denied


def test_only_selected_gradients_projected_and_parameter_poison_ignored(cfg,tmp_path):
 cfg=parent_fixture(cfg,tmp_path);g=a.Guard(cfg,limits=a.base.Limits(120,2**40))
 with g:
  data=g.project();main.analyze_rows(cfg,data);g.complete()
 assert g.rows=={'endpoints':10,'parameters':2570,'trials':100} and g.parses==1 and g.hash_checks==2
 assert all('parameter' not in row for row in data['parameters'])


def test_modified_input_fails_before_projection(cfg,tmp_path):
 cfg=parent_fixture(cfg,tmp_path);p=Path(cfg['inputs']['source_JSON']['path']);p.write_text(p.read_text()+'\n')
 g=a.Guard(cfg,limits=a.base.Limits(120,2**40))
 with pytest.raises(ValueError):
  with g:g.project()
 assert g.parses==0 and not g.rows


@pytest.mark.parametrize('field,value',[('trial_index',True),('armijo_pass',1),('component3_mse',float('inf'))])
def test_native_JSON_types_are_checked(cfg,field,value):
 cfg,data=synthetic_rows(cfg);row=data['trials'][0];row[field]=value
 with pytest.raises(ValueError):a.typed('trials',row)


def test_unbound_APIs_reject_before_payload_or_directory(cfg,tmp_path,monkeypatch):
 science=tmp_path/'science.json';science.write_text(json.dumps(cfg));sp=d.p.pin(science);bp={'path':str(tmp_path/'absent.json'),'sha256':'0'*64}
 def no_payload(*args,**kw):raise AssertionError('payload must not start')
 monkeypatch.setattr(a.Guard,'project',no_payload)
 for fn in (main.execute_payload,audit.audit_payload):
  with pytest.raises(FileNotFoundError):fn(cfg,sp,bp)
 assert not Path(cfg['output']).exists() and not Path(cfg['artifacts']).exists()


def test_full_synthetic_one_JSON_per_process_and_fresh_audit(cfg,tmp_path):
 cfg=parent_fixture(cfg,tmp_path);path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env=os.environ.copy();env.update(cfg['execution']['environment'])
 for i,(name,entry) in enumerate([('analyze_libero_task7_fine_stop_direction_review','execute_payload'),('audit_libero_task7_fine_stop_direction_review','audit_payload')]):
  code=f'''import json
from pathlib import Path
from experiments import libero_task7_fine_stop_direction_review_access as a
from experiments import {name} as module
cfg=json.load(open({str(path)!r}))
def synthetic_only(cfg,sp,bp):
 assert Path(cfg['inputs']['source_JSON']['path']).is_relative_to(Path({str(tmp_path)!r}))
a.validate_call=synthetic_only
pin={{'path':'synthetic-only','sha256':{'a'*64!r}}}
r=module.{entry}(cfg,pin,pin)
assert r['access']['parent_JSON_parses']==1 and r['access']['parent_JSON_SHA_checks']==2
'''
  result=subprocess.run([sys.executable,'-c',code],env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=135)
  (tmp_path/f'child-{i}.log').write_text(result.stdout);assert result.returncode==0,result.stdout
 r=json.loads(Path(cfg['output']).read_text());done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
 assert done['total_parent_JSON_parses']==2 and done['total_parent_JSON_SHA']==4 and len(r['records']['trials'])==100
