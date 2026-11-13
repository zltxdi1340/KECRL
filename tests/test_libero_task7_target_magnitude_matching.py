"""Synthetic projections, exact/coarse support, weighting and independent pipelines."""
from __future__ import annotations

import copy
import csv
import inspect
import itertools
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_task7_target_magnitude_matching_protocol as d
from experiments import libero_task7_target_magnitude_matching_access as access
from experiments import analyze_libero_task7_target_magnitude_matching as main
from experiments import audit_libero_task7_target_magnitude_matching as audit


class NoLimits:
    def check(self):pass


def make_synthetic(directory,sparse=False):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True);cfg=json.loads(d.DEFAULT_CONFIG.read_text())
    layout=d.previous.source_layout(cfg['inputs']['action_lengths']);n=len(layout);demo=np.asarray([r['demo'] for r in layout]);step=np.asarray([r['action_step'] for r in layout]);tail=np.asarray([r['tail20'] for r in layout])
    sign=np.zeros(n,np.int8);body=np.flatnonzero((step>=5)&~tail);end=np.flatnonzero(tail)
    sign[body[:1871]]=-1;sign[body[1871:3121]]=1;sign[end[:402]]=-1;sign[end[402:477]]=1
    ids=np.flatnonzero(sign!=0);source=np.arange(n)
    magnitude=(.03125+source*.0001).astype(np.float32) if sparse else np.asarray([.03125,.0625,.125,.25,.5,.75,1.],np.float32)[source%7]
    target=(sign*magnitude).astype(np.float32).astype(np.float64);payloads={}
    for number,endpoint in enumerate(cfg['inputs']['endpoints']):
        factor=np.asarray([-1.,0.,.25,1.25,1.,.75,-.2,.2])[(source+number)%8]
        base=np.clip(target*factor,-1,1).astype(np.float32).astype(np.float64)
        bias=np.asarray([.0625,-.0625,.03125],np.float32).astype(np.float64)[(demo//10+number)%3]
        affine=-.2+.05*((source+number)%9);affine[(source+number)%41==0]=2.
        raw=np.stack([np.zeros(n),bias,affine]);pred=np.clip(base[None,:]+raw,-1,1).astype(np.float32).astype(np.float64)
        clip=np.abs(base[None,:]+raw)>1
        payloads[endpoint['seed'],endpoint['arm']]={'p':pred,'raw':raw,'clip':clip}
    ref=cfg['inputs']['paired_frames'];path=directory/'paired.csv';columns=ref['columns']
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for endpoint in cfg['inputs']['endpoints']:
            data=payloads[endpoint['seed'],endpoint['arm']]
            for candidate,reference in d.COMPARISONS:
                c,r=d.previous.MODES.index(candidate),d.previous.MODES.index(reference)
                for i in ids:
                    y=float(target[i]);pr=float(data['p'][r,i]);pc=float(data['p'][c,i]);dr=float(data['raw'][r,i]);dc=float(data['raw'][c,i])
                    q=d.previous.pair_quantities(y,pr,pc,dr,dc)
                    row={key:'UNPROJECTED_DO_NOT_PARSE' for key in columns}
                    row.update(task=7,seed=endpoint['seed'],arm=endpoint['arm'],comparison=candidate+'-minus-'+reference,candidate=candidate,reference=reference,
                               source_row=int(i),demo=int(demo[i]),action_step=int(step[i]),target_sign=int(sign[i]),tail20=bool(tail[i]),target_2=y,
                               reference_prediction_2=pr,candidate_prediction_2=pc,reference_category=d.previous.error_category(y,pr),candidate_category=d.previous.error_category(y,pc),
                               reference_clip=bool(data['clip'][r,i]),candidate_clip=bool(data['clip'][c,i]),transition=q['transition'],reference_error=q['reference_error'],candidate_error=q['candidate_error'],
                               reference_band=q['reference_band'],candidate_band=q['candidate_band'],reference_sse=(pr-y)**2,candidate_sse=(pc-y)**2,delta_sse=q['delta_sse'],
                               reference_raw_residual=dr,candidate_raw_residual=dc)
                    writer.writerow(row)
    cfg['inputs']['paired_frames']={**ref,**d.p.pin(path)};cfg['parents']={'synthetic':True};cfg['output']=str(directory/'result.json');cfg['artifacts']=str(directory/'artifacts')
    return cfg


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):return make_synthetic(tmp_path_factory.mktemp('target-match-dense'))


def compare(actual,expected):audit.compare_tree(actual,expected,audit.Checks(),'synthetic')


@pytest.mark.parametrize('scheme',['exact_target','coarse_bin'])
@pytest.mark.parametrize('focus,control',[([],[]),([.03125],[]),([],[.0625]),([.03125]*9,[.015625]),([.03125,.0625],[.125])])
def test_independent_cell_and_demo_weights(scheme,focus,control):
    if scheme=='exact_target' and not focus and not control:return
    y=.125;key=d.target_bits(y) if scheme=='exact_target' else d.target_bin(y)
    rows=lambda values:[d.frame_metrics(y,value,-value,0.,-2*value) for value in values]
    f,c=rows(focus),rows(control);cell=d.compare_cell(f,c,scheme,key)
    independent=audit.independent_cell([[r[k] for k in d.METRICS] for r in f],[[r[k] for k in d.METRICS] for r in c],scheme,key)
    compare(cell,independent);compare(main.cell_row(cell),audit.encode_cell(independent))
    compare(d.summarize_cells([cell],scheme),audit.independent_demo([independent],scheme))


def test_simpson_reversal_and_equal_demo_independent_reconstruction():
    cells=[]
    for y,f,c in [(.125,[.03125]*9,[.015625]),(.5,[.25],[.125]*9)]:
        frows=[d.frame_metrics(y,p,-p,0.,-2*p) for p in f];crows=[d.frame_metrics(y,p,-p,0.,-2*p) for p in c]
        cells.append(d.compare_cell(frows,crows,'exact_target',d.target_bits(y)))
    actual=d.summarize_cells(cells,'exact_target');independent=audit.independent_demo(cells,'exact_target');compare(actual,independent)
    assert actual['pooled_full']['delta']['reference_abs']<0 and actual['equal_cell']['delta']['reference_abs']>0
    empty=d.summarize_cells([],'exact_target');demos=[{'demo':i,**(actual if i==0 else empty)} for i in range(30)]
    compare(d.summarize_demos(demos),audit.independent_model(demos))


def test_empty_support_and_intersection_reconstruction():
    f=d.frame_metrics(.125,.03125,-.03125,0.,-.0625);c=d.frame_metrics(.25,.125,-.125,0.,-.25)
    cells=[d.compare_cell([f],[],'exact_target',d.target_bits(.125)),d.compare_cell([],[c],'exact_target',d.target_bits(.25))]
    a=d.summarize_cells(cells,'exact_target');b=audit.independent_demo(cells,'exact_target');compare(a,b)
    assert a['missing_reason']=='no_common_exact_target' and a['pooled_common']['delta']['reference_abs'] is None
    assert d.common3_keys([{(0,1)},set(),{(0,1)}])==set()


def test_projection_28_fields_only_and_inherited_denial_contract(synthetic):
    guard=access.Guard(synthetic['inputs'],limits=NoLimits());n=0
    with guard:
        guard.hash_inputs()
        for row in guard.project():assert set(row)==set(d.COLUMNS);n+=1
        guard.complete()
    assert n==64764 and guard.record()['CSV_projected_cells']=={'paired_frames':1813392}
    assert guard.record()['byte_SHA_checks']==['paired_frames']*3


@pytest.mark.parametrize('operation',['source_raw','old_CSV','archive_hash','np_load','fromfile','memmap','loadtxt','FileIO','os_open','import_torch','import_environment','extra_fields','direct_reader','repeat','source_write'])
def test_inherited_bypasses_fail_closed(synthetic,operation):
    guard=access.Guard(synthetic['inputs'],limits=NoLimits());path=Path(synthetic['inputs']['paired_frames']['path'])
    with guard:
        with pytest.raises((PermissionError,ValueError)):
            if operation=='source_raw':path.read_text()
            elif operation=='old_CSV':Path('/never/open/curves.csv').read_text()
            elif operation=='archive_hash':d.p.sha('/never/open/features.npz')
            elif operation=='np_load':np.load('/never/open/weights.npz',allow_pickle=False)
            elif operation=='fromfile':np.fromfile('/never/open/features.npy')
            elif operation=='memmap':np.memmap('/never/open/features.npy')
            elif operation=='loadtxt':np.loadtxt(path)
            elif operation=='FileIO':__import__('io').FileIO(path)
            elif operation=='os_open':os.open(path,os.O_RDONLY)
            elif operation=='import_torch':__import__('torch')
            elif operation=='import_environment':__import__('src.environments.libero_adapter')
            elif operation=='extra_fields':next(guard.project(d.COLUMNS+['reference_abs']))
            elif operation=='direct_reader':
                gen=guard.project();next(gen);csv.reader(gen.gi_frame.f_locals['handle'])
            elif operation=='repeat':
                gen=guard.project();next(gen);gen.close();next(guard.project())
            elif operation=='source_write':path.open('w')
    assert guard.denied


@pytest.mark.parametrize('field,value',[('seed','20261106.0'),('action_step','4'),('target_sign','0'),('candidate','history'),
                                       ('comparison','current_affine-minus-current_affine'),('reference_sse','nan'),('tail20','1')])
def test_syntax_role_and_material_inputs_rejected(synthetic,field,value):
    with Path(synthetic['inputs']['paired_frames']['path']).open(newline='') as handle:raw=next(csv.DictReader(handle))
    row={key:raw[key] for key in d.COLUMNS};row[field]=value
    with pytest.raises(ValueError):access.typed(row)


class Rows:
    def __init__(self,rows):self.rows=rows
    def project(self):yield from copy.deepcopy(self.rows)


@pytest.mark.parametrize('case',['duplicate','demo','tail','label','SSE','band','category','float32'])
def test_independent_source_rejections(synthetic,case):
    with Path(synthetic['inputs']['paired_frames']['path']).open(newline='') as handle:raw=next(csv.DictReader(handle))
    row=access.typed({key:raw[key] for key in d.COLUMNS});rows=[row]
    if case=='duplicate':rows.append(copy.deepcopy(row))
    elif case=='demo':row['demo']=(row['demo']+1)%30
    elif case=='tail':row['tail20']=not row['tail20']
    elif case=='label':row['target_sign']=-row['target_sign']
    elif case=='SSE':row['delta_sse']+=1
    elif case=='band':row['reference_band']='zero' if row['reference_band']!='zero' else 'abs_ge.025'
    elif case=='category':row['candidate_category']='wrong_sign' if row['candidate_category']!='wrong_sign' else 'exact_match'
    elif case=='float32':row['target_2']=.1
    with pytest.raises(ValueError):main.load_source(Rows(rows),synthetic)
    with pytest.raises(ValueError):audit.reconstruct_source(Rows(rows),synthetic)


@pytest.mark.parametrize('case',['SHA','header','duplicate_header','width'])
def test_malformed_CSV_rejected_before_payload(synthetic,tmp_path,case):
    inputs=copy.deepcopy(synthetic['inputs']);path=tmp_path/'invalid.csv';columns=inputs['paired_frames']['columns']
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(columns[:-1] if case=='header' else columns+['task'] if case=='duplicate_header' else columns);writer.writerow(['bad_width'])
    inputs['paired_frames'].update(d.p.pin(path))
    if case=='SHA':inputs['paired_frames']['sha256']='f'*64
    guard=access.Guard(inputs,limits=NoLimits())
    with guard:
        with pytest.raises(ValueError):next(guard.project())
    assert guard.attempted==['paired_frames'] and not guard.completed


def test_unbound_execution_refused_before_projection(tmp_path):
    with pytest.raises(ValueError,match='execution binding required'):main.analyze(binding=tmp_path/'missing-binding.json')


def test_audit_uses_no_primary_or_metadata_numeric_functions():
    text=inspect.getsource(audit)
    for forbidden in ['from experiments import analyze_libero_task7_target_magnitude_matching','d.frame_metrics(', 'd.compare_cell(', 'd.summarize_cells(', 'd.summarize_demos(', 'd.target_bits(', 'd.target_bin(', 'd.common3_keys(']:assert forbidden not in text


@pytest.mark.parametrize('sparse',[False,True])
def test_fresh_full_pipeline_at_frozen_limits(synthetic,tmp_path,sparse):
    cfg=make_synthetic(tmp_path/'sparse',True) if sparse else copy.deepcopy(synthetic)
    if not sparse:cfg['output']=str(tmp_path/'result.json');cfg['artifacts']=str(tmp_path/'artifacts')
    path=tmp_path/'science.json';path.write_text(json.dumps(cfg));env={**os.environ,**cfg['execution']['environment']}
    script='''import importlib,json,pathlib,sys
m=importlib.import_module(sys.argv[1]);cfg=json.loads(pathlib.Path(sys.argv[2]).read_text())
science={"path":"synthetic-science","sha256":"0"*64};binding={"path":"synthetic-binding","sha256":"1"*64}
fn=m.analyze_payload if sys.argv[3]=="analysis" else m.audit_payload
r=fn(cfg,science,binding);print(json.dumps({"status":r["status"]}))
'''
    for stage,module in [('analysis','experiments.analyze_libero_task7_target_magnitude_matching'),('audit','experiments.audit_libero_task7_target_magnitude_matching')]:
        result=subprocess.run([sys.executable,'-c',script,module,str(path),stage],cwd=d.ROOT,env=env,text=True,capture_output=True,timeout=195)
        (tmp_path/(stage+'.log')).write_text(result.stdout+result.stderr);assert result.returncode==0,result.stdout+result.stderr
    root=Path(cfg['artifacts']);record=json.loads(Path(cfg['output']).read_text());audited=json.loads((root/'independent-audit.json').read_text());done=json.loads((root/'completion.json').read_text())
    assert done['total_source_CSV_data_passes']==2 and done['total_source_projected_cells']==3626784
    assert audited['scalar_checks']>2_000_000 and audited['primary_numerical_implementation_imported'] is False
    assert len(record['seed_summary'])==144
    assert record['cost']['peak_RSS_bytes']<=536870912 and audited['cost']['peak_RSS_bytes']<=536870912
    assert sum(Path(ref['path']).stat().st_size for ref in record['tables'].values())<=268435456
    for key,n in d.budget()['fixed_table_rows'].items():assert record['tables'][key]['rows']==n
    if sparse:
        assert record['tables']['exact-target-cells']['rows']==129528
        assert all(r['individual_support']['standardized']['reference_abs']['mean'] is None for r in record['seed_summary'] if r['scheme']=='exact_target')
    else:
        assert any(r['common_cell_n']>0 for r in record['seed_summary'] if r['scheme']=='exact_target')
    with pytest.raises(ValueError,match='existing run'):main.analyze_payload(cfg,record['science_config'],record['binding'])
    with pytest.raises(ValueError,match='existing audit'):audit.audit_payload(cfg,record['science_config'],record['binding'])
