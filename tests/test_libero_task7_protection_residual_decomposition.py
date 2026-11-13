"""Synthetic projection failures, independent geometry and full bounded pipeline."""
from __future__ import annotations

import copy
import csv
import inspect
import json
import os
from pathlib import Path
import random
import struct
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_task7_protection_residual_decomposition_protocol as d
from experiments import libero_task7_protection_residual_decomposition_access as access
from experiments import analyze_libero_task7_protection_residual_decomposition as main
from experiments import audit_libero_task7_protection_residual_decomposition as audit

LENGTHS=[155,125,124,191,136,138,133,130,139,145,140,164,124,138,120,137,141,152,137,128,197,124,128,126,144,129,168,132,140,133]


class NoLimits:
    def check(self):pass


def row_value():
    return dict(task=7,seed=d.p.SEEDS[0],arm=d.p.ARMS[0],mode='original',update=500,source_row=0,probe_fold=0,demo=0,
        action_step=1,target_direction=0,tail20=False,component3_category='exact_match',raw_residual=0.,clip=False,target_2=0.,prediction_2=0.,error_2=0.)


def source_csv(path,rows):
    with Path(path).open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=d.HEADER);writer.writeheader()
        for row in rows:writer.writerow({key:row.get(key,'UNSELECTED_MUST_NOT_CONVERT') for key in d.HEADER})
    return {**d.p.pin(path),'rows':len(rows),'columns':d.HEADER.copy()}


@pytest.fixture
def source(tmp_path):
    return {'frames':source_csv(tmp_path/'frames.csv',[row_value()])}


def test_projection_only17_columns_leaves18_unselected_sentinels_unconverted(source):
    guard=access.Guard(source,limits=NoLimits())
    with guard:
        guard.hash_inputs();rows=list(guard.project('frames'))
    assert rows==[row_value()] and len(rows[0])==17
    assert guard.record()['CSV_projected_cells']=={'frames':17} and guard.record()['unselected_columns_numeric_converted']==0
    assert guard.record()['CSV_data_passes']==dict(attempted=1,completed=1)


@pytest.mark.parametrize('operation',['columns','repeat','unlisted','raw','reader','array','loadtxt','fromfile','zip','checkpoint','other_CSV','source_write'])
def test_field_file_and_alternate_reader_fail_closed(source,operation,tmp_path):
    path=source['frames']['path'];handle=Path(path).open()
    try:
        with access.Guard(source,limits=NoLimits()) as guard:
            with pytest.raises(PermissionError):
                if operation=='columns':list(guard.project('frames',d.COLUMNS+['prediction_0']))
                elif operation=='repeat':list(guard.project('frames'));list(guard.project('frames'))
                elif operation=='unlisted':list(guard.project('curves'))
                elif operation=='raw':Path(path).read_bytes()
                elif operation=='reader':list(csv.reader(handle))
                elif operation=='array':np.load('/never/open/cache.npz',allow_pickle=False)
                elif operation=='loadtxt':np.loadtxt(path,delimiter=',')
                elif operation=='fromfile':np.fromfile(path)
                elif operation=='zip':
                    import zipfile
                    with zipfile.ZipFile.__new__(zipfile.ZipFile) as archive:archive.open('x')
                elif operation=='checkpoint':Path('/never/open/original.pt').read_bytes()
                elif operation=='other_CSV':Path('/never/open/protection-groups.csv').read_text()
                else:Path(path).write_text('overwritten')
    finally:handle.close()


@pytest.mark.parametrize('field,value',[('task','4'),('seed','20261109'),('mode','bias_only'),('update','250'),('source_row','-1'),
    ('probe_fold','3'),('tail20','true'),('clip','0'),('target_2','nan'),('raw_residual','inf'),('component3_category','unknown'),('demo','01')])
def test_typed_projection_rejects_role_or_syntax_drift(field,value):
    row={key:str(value) for key,value in row_value().items()};row[field]=value
    with pytest.raises((ValueError,KeyError)):access.typed(row)


def test_hash_mismatch_rejects_before_any_column_conversion(source):
    bad=copy.deepcopy(source);bad['frames']['sha256']='0'*64
    with access.Guard(bad,limits=NoLimits()) as guard:
        with pytest.raises(ValueError,match='SHA'):list(guard.project('frames'))
    assert guard.cells=={'frames':0}


def test_partial_input_consumption_is_never_complete(tmp_path):
    inputs={'frames':source_csv(tmp_path/'two.csv',[row_value(),{**row_value(),'source_row':1,'action_step':2}])}
    with access.Guard(inputs,limits=NoLimits()) as guard:
        iterator=guard.project('frames');next(iterator);iterator.close()
        with pytest.raises(ValueError,match='projection budget'):guard.complete()
    assert guard.record()['CSV_data_passes']==dict(attempted=1,completed=0)


def triplet(y,p,b,c,rb=None,rc=None):
    original={**row_value(), 'source_row':0,'action_step':5,'target_2':y,'prediction_2':p,'error_2':p-y,
              'target_direction':1 if y>=.025 else -1 if y<=-.025 else 0,'component3_category':d.layout.error_category(y,p)}
    base={**original,'mode':'uniform_affine','prediction_2':b,'error_2':b-y,'raw_residual':b-p if rb is None else rb,
          'component3_category':d.layout.error_category(y,b)}
    candidate={**original,'mode':'harm_protected_affine','prediction_2':c,'error_2':c-y,'raw_residual':c-p if rc is None else rc,
               'component3_category':d.layout.error_category(y,c)}
    base['clip']=abs(p+base['raw_residual'])>1;candidate['clip']=abs(p+candidate['raw_residual'])>1
    return original,base,candidate


@pytest.mark.parametrize('values',[(.25,-.25,.125,-.0625),(-.25,-.125,-.25,-.5),(.25,.125,.0625,.1875),
    (.5,.875,1.,1.,.5,.25),(.25,.25,.25,0.),(0.,0.,.125,-.125)])
def test_independent_geometry_and_statistics_match_frozen_synthetic_cases(values):
    o,b,c=triplet(*values);actual=audit.frame(o,b,c)
    geo=d.frame_geometry(o['target_2'],o['prediction_2'],b['prediction_2'],c['prediction_2'])
    for key,value in geo.items():assert actual[key]==value
    for rep in d.REPRESENTATIONS:
        assert main.statistics([actual],8,rep)==audit.statistics([actual],8,rep)
    assert actual['closed_segment'] is False or actual['hC']<=actual['hB']


def test_empty_cells_full_cohort_contributions_and_quantile_exactness():
    for stats in (main.statistics,audit.statistics):
        value=stats([],10)
        assert value['baseline_mse'] is value['ratio_median'] is None and value['delta_harm_cohort_contribution']==0
        assert stats([],0)['delta_harm_cohort_contribution'] is None
    assert main.describe([0.,1.,2.,3.],'x')==audit.distribution([0.,1.,2.,3.],'x')==dict(x_mean=1.5,x_median=1.5,x_q25=.75,x_q75=2.25)


def test_summary_does_not_impute_missing_seed_support():
    records={}
    key=('movement','all_common','all','effective','zero_to_nonzero','')
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            records[seed,arm]={key:{field:1 if field in ('n','cohort_n') else .125 for field in main.SUMMARY_FIELDS}}
    records[d.p.SEEDS[-1],d.p.ARMS[0]][key]['n']=0
    # Exercise the support rule with expected full row count using synthetic keys.
    for part in records.values():
        source=copy.deepcopy(part[key])
        for j in range(671):part[('movement','all_common','all','effective','synthetic'+str(j),'')]=source.copy()
    left=main.seed_summary(records);right=audit.summary(records)
    assert left==right
    row=next(r for r in left if r['arm']==d.p.ARMS[0] and r['movement']=='zero_to_nonzero')
    assert not row['cell_supported_in_all3'] and row['metrics']['delta_abs_mean']['complete_mean'] is None


def make_synthetic(directory):
    directory=Path(directory);directory.mkdir(parents=True)
    plan=d.layout.source_layout(LENGTHS);body=[r['source_row'] for r in plan if r['action_step']>=5 and not r['tail20']]
    tail=[r['source_row'] for r in plan if r['tail20']];labels=[0]*4188
    for positions,negative,positive in [(body,1871,1250),(tail,402,75)]:
        for i in positions[:negative]:labels[i]=-1
        for i in positions[negative:negative+positive]:labels[i]=1
    rng=random.Random(131102);targets=[d.layout.float32(sign*rng.choice([.125,.25,.5])) for sign in labels]
    all_rows=[];endpoints=[];parity=[]
    for seed in d.p.SEEDS:
        for arm in d.p.ARMS:
            endpoint={'task':7,'seed':seed,'arm':arm,'checkpoint':{'path':'/never/open/original.pt','sha256':'b'*64,'weight_hash':'c'*64}}
            endpoints.append(endpoint);modes={mode:[] for mode in d.MODES};frames=[]
            for info,y in zip(plan,targets):
                p=d.layout.float32(max(-1.,min(1.,y+rng.uniform(-.45,.45))))
                rb=rng.uniform(-.4,.4);kind=info['source_row']%7
                if kind==0:rc=rb*.5
                elif kind==1:rc=rb*1.5
                elif kind==2:rc=-rb*.7
                elif kind==3:rc=0.
                elif kind==4:rc=rb
                elif kind==5:rb=0.;rc=.125
                else:rb=rc=0.
                b=d.layout.float32(max(-1.,min(1.,p+rb)));c=d.layout.float32(max(-1.,min(1.,p+rc)))
                rows=triplet(y,p,b,c,rb,rc)
                for mode,row in zip(d.MODES,rows):
                    row.update(task=7,seed=seed,arm=arm,**info);modes[mode].append(row)
                if info['action_step']>=5:frames.append(audit.frame(*rows))
            for mode in d.MODES:all_rows.extend(modes[mode])
            parity.append({**{key:endpoint[key] for key in ('task','seed','arm')},**audit.parity_projection(frames)})
    ref=source_csv(directory/'input.csv',all_rows)
    cfg={'inputs':{'frames':ref,'endpoints':endpoints,'action_lengths':LENGTHS.copy(),'parent_parity':parity},
         'scope':copy.deepcopy(d.SCOPE),'execution':copy.deepcopy(d.EXECUTION),'budget':d.budget(),'parents':{'synthetic':True},
         'output':str(directory/'result.json'),'artifacts':str(directory/'artifacts')}
    (directory/'config.json').write_text(json.dumps(cfg));return cfg


def test_independent_import_has_no_primary_numerics_or_Torch():
    command="from experiments import audit_libero_task7_protection_residual_decomposition as a; import sys; assert 'torch' not in sys.modules; assert 'experiments.analyze_libero_task7_protection_residual_decomposition' not in sys.modules; print('independent')"
    r=subprocess.run([sys.executable,'-c',command],text=True,capture_output=True,timeout=10)
    assert r.returncode==0 and r.stdout.strip()=='independent',r.stderr
    source=inspect.getsource(audit)
    assert 'd.frame_geometry(' not in source and 'd.movement(' not in source and 'd.ratio(' not in source


def test_preflight_wrong_science_never_opens_input_CSV(tmp_path):
    p=tmp_path/'bad.json';p.write_text('{}')
    with pytest.raises(ValueError,match='frozen science'):access.preflight(p)


def test_missing_binding_preflight_refuses_before_source_bytes(monkeypatch,tmp_path):
    monkeypatch.setattr(access.d.p,'sha',lambda path:access.SCIENCE_SHA)
    monkeypatch.setattr(access.d,'validate',lambda *a:{})
    def unbound(*a):raise ValueError('execution binding required before CSV byte hash')
    monkeypatch.setattr(access.d,'require_binding',unbound)
    with pytest.raises(ValueError,match='before CSV byte hash'):access.preflight(tmp_path/'cfg.json',tmp_path/'missing.json')


def test_saved_frame_output_comparison_does_not_relax_float32_values(tmp_path):
    path=tmp_path/'paired-frames.csv'
    with path.open('x',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['baseline']);writer.writerow([.125+1e-12])
    table=audit.StrictTable({'path':str(path),'columns':['baseline'],'rows':1},audit.Checks(),NoLimits())
    try:
        with pytest.raises(ValueError):table.row({'baseline':.125})
    finally:table.handle.close()


def test_full_sole_CSV_two_processes_and_independent_audit(tmp_path):
    directory=tmp_path/'full';make_synthetic(directory);cfg_path=directory/'config.json'
    first='''import copy,json
from pathlib import Path
from experiments import analyze_libero_task7_protection_residual_decomposition as m
from experiments import audit_libero_task7_protection_residual_decomposition as a
cfg=json.loads(Path(CONFIG).read_text());pin={'path':'synthetic-only','sha256':'a'*64}
r=m.analyze_payload(cfg,pin,pin)
assert r['access']['CSV_data_passes']==dict(attempted=1,completed=1)
assert r['access']['CSV_projected_cells']=={'frames':1281528}
assert sum(row['rows'] for row in r['tables'].values())==149418
for field,value in [('formal_result',True),('binding',{}),('parent_parity_checks',1548.0),('new_training_forward_environment_neighbor_query_sealed_calls',False)]:
 bad=copy.deepcopy(r);bad[field]=value
 try:a.validate_record(cfg,bad,pin,pin)
 except ValueError:pass
 else:raise AssertionError('malformed main provenance accepted '+field)
try:m.analyze_payload(cfg,pin,pin)
except ValueError:pass
else:raise AssertionError('analysis retry allowed')
'''
    second='''import json,sys
from pathlib import Path
from experiments import audit_libero_task7_protection_residual_decomposition as a
cfg=json.loads(Path(CONFIG).read_text());pin={'path':'synthetic-only','sha256':'a'*64}
assert 'experiments.analyze_libero_task7_protection_residual_decomposition' not in sys.modules
r=a.audit_payload(cfg,pin,pin)
assert r['access']['CSV_data_passes']==dict(attempted=1,completed=1) and r['parent_parity_checks']==1548
assert r['scalar_checks']>5000000 and 'torch' not in sys.modules
try:a.audit_payload(cfg,pin,pin)
except ValueError:pass
else:raise AssertionError('audit retry allowed')
Path(PROOF).write_text(json.dumps({'checks':r['scalar_checks'],'cost':r['cost']}))
'''
    env=os.environ.copy();env.update(d.EXECUTION['environment'])
    for label,script in [('analysis',first),('audit',second)]:
        code=script.replace('CONFIG',repr(str(cfg_path))).replace('PROOF',repr(str(directory/'synthetic-proof.json')))
        r=subprocess.run([sys.executable,'-c',code],cwd=d.ROOT,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=195)
        (tmp_path/(label+'.log')).write_text(r.stdout)
        assert r.returncode==0,label+'\n'+r.stdout
    assert json.loads((directory/'synthetic-proof.json').read_text())['checks']>5000000
