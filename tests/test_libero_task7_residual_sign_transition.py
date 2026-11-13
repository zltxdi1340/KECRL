"""Synthetic CSV/field/role/numerics and full fresh-process independent-audit tests."""
from __future__ import annotations

import copy
import csv
import inspect
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from experiments import libero_task7_residual_sign_transition_protocol as d
from experiments import libero_task7_residual_sign_transition_access as access
from experiments import analyze_libero_task7_residual_sign_transition as main
from experiments import audit_libero_task7_residual_sign_transition as audit


class NoLimits:
    def check(self): pass


def make_synthetic(directory):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    cfg=json.loads(d.DEFAULT_CONFIG.read_text());lengths=cfg['inputs']['action_lengths']
    layout=d.source_layout(lengths);demo=np.asarray([r['demo'] for r in layout]);step=np.asarray([r['action_step'] for r in layout]);tail=np.asarray([r['tail20'] for r in layout])
    common=step>=5;body=np.flatnonzero(common&~tail);end=np.flatnonzero(tail);sign=np.zeros(4188,np.int8)
    sign[body[:1871]]=-1;sign[body[1871:3121]]=1;sign[end[:402]]=-1;sign[end[402:477]]=1
    rng=np.random.default_rng(4103);target=(sign*rng.uniform(.026,.8,4188)).astype(np.float32).astype(np.float64)
    source=np.arange(4188);payloads={}
    for number,model in enumerate(cfg['inputs']['endpoints']):
        factors=np.asarray([-1.,0.,.4,1.5,1.,.8])[(source+number)%6]
        original=np.clip(target*factors,-1,1).astype(np.float32).astype(np.float64)
        original[(source+number)%13==0]=np.float32(.000001)
        original[(source+number)%37==0]=np.float32(-.95)
        folds=demo//10;bias=np.asarray([.02,-.04,.1],np.float32).astype(np.float64)[(folds+number)%3]
        affine=.08*np.sin(source*.17+number)+.03
        affine[(source+number)%29==0]=1.9
        raw=np.stack([np.zeros(4188),bias,affine]);pred=np.clip(original[None,:]+raw,-1,1).astype(np.float32).astype(np.float64)
        clip=np.abs(original[None,:]+raw)>1
        payloads[model['seed'],model['arm']]={'p':pred,'raw':raw,'clip':clip}
    tables=cfg['inputs']['tables'];framepath=directory/'frames.csv';columns=tables['frames']['columns']
    with framepath.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for model in cfg['inputs']['endpoints']:
            data=payloads[model['seed'],model['arm']]
            for mode,name in enumerate(d.MODES):
                for i in range(4188):
                    row={key:'DO_NOT_PARSE_OTHER_AXIS' for key in columns}
                    row.update(task=7,seed=model['seed'],arm=model['arm'],mode=name,update=500,**layout[i],target_direction=int(sign[i]),
                               component3_category=d.error_category(float(target[i]),float(data['p'][mode,i])),raw_residual=float(data['raw'][mode,i]),clip=bool(data['clip'][mode,i]),
                               target_2=float(target[i]),prediction_2=float(data['p'][mode,i]),error_2=float(data['p'][mode,i]-target[i]))
                    writer.writerow(row)
    curvepath=directory/'curves.csv';columns=tables['curves']['columns']
    with curvepath.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=columns);writer.writeheader()
        for model in cfg['inputs']['endpoints']:
            data=payloads[model['seed'],model['arm']]
            for fold,mode,update,role,phase,direction in itertools.product(range(3),d.MODES[1:],(0,100,250,500),('fit','probe'),('all_source','all_common','tail20','complement'),('all','negative','positive','neutral')):
                row={key:'DO_NOT_PARSE_UNSELECTED_METRIC' for key in columns}
                row.update(task=7,seed=model['seed'],arm=model['arm'],fold=fold,mode=mode,update=update,role=role,phase=phase,direction=direction)
                if update==500 and role=='probe':
                    mask=demo//10==fold
                    if phase!='all_source':mask&=step>=5
                    if phase in ('tail20','complement'):mask&=tail==(phase=='tail20')
                    if direction!='all':mask&=sign=={'negative':-1,'positive':1,'neutral':0}[direction]
                    ids=np.flatnonzero(mask);m=d.MODES.index(mode);error=data['p'][m,ids]-target[ids];n=len(ids);material=sign[ids]!=0
                    count=int(material.sum());wrong=int(np.count_nonzero(material&(sign[ids]*data['p'][m,ids]<=0)));raw=data['raw'][m,ids]
                    row.update(n=n,material_n=count,component3_mse=float(np.sum(error**2)/n) if n else None,component3_sse=float(np.sum(error**2)),
                               component3_mae=float(np.sum(abs(error))/n) if n else None,component3_bias=float(np.sum(error)/n) if n else None,
                               material_sign_error_n=wrong,material_sign_error_rate=wrong/count if count else None,clip_n=int(data['clip'][m,ids].sum()),
                               clip_fraction=int(data['clip'][m,ids].sum())/n if n else None,raw_residual_mean=float(raw.mean()) if n else None,
                               raw_residual_abs_mean=float(np.abs(raw).mean()) if n else None,raw_residual_min=float(raw.min()) if n else None,raw_residual_max=float(raw.max()) if n else None)
                writer.writerow(row)
    cfg['inputs']['tables']={'frames':{**tables['frames'],**d.p.pin(framepath)},'curves':{**tables['curves'],**d.p.pin(curvepath)}}
    cfg['parents']={'synthetic':True};cfg['output']=str(directory/'result.json');cfg['artifacts']=str(directory/'artifacts')
    return cfg


@pytest.fixture(scope='module')
def synthetic(tmp_path_factory):return make_synthetic(tmp_path_factory.mktemp('residual-sign-CSV'))


def small_data():
    target=np.asarray([.03,.03,.2,-.2,-.2,.2,.2,0.],np.float64)
    original=np.asarray([1.,-.01,0.,-.1,.1,.03,-.03,.1],np.float64)
    bias=np.full(8,.02);affine=np.asarray([-1.01,1.01,.1,.3,-.2,-.04,.07,-.1])
    raw=np.stack([np.zeros(8),bias,affine]);pred=np.clip(original+raw,-1,1).astype(np.float32).astype(np.float64)
    sign=np.where(target>=.025,1,np.where(target<=-.025,-1,0))
    category=np.asarray([[d.CLASSES.index(d.error_category(float(target[i]),float(pred[m,i]))) for i in range(8)] for m in range(3)],np.int8)
    return {'target':target,'sign':sign}, {'prediction':pred,'residual':raw,'clip':np.abs(original+raw)>1,'category':category}


def audit_store(store):return {'p':store['prediction'],'d':store['residual'],'clip':store['clip'],'category':store['category']}


def assert_equal(actual,expected):
    assert set(actual)==set(expected)
    for key,value in expected.items():
        if isinstance(value,float):assert actual[key]==pytest.approx(value,abs=1e-10,rel=0),key
        else:assert actual[key]==value,key


@pytest.mark.parametrize('candidate,reference',d.COMPARISONS)
@pytest.mark.parametrize('ids',[[],[0],[1],[2],[3,4],list(range(7))])
def test_independent_stats_four_cells_zeros_and_error_reversals(candidate,reference,ids):
    shared,store=small_data();ctx=main.pair_context(shared,store,candidate,reference)
    independent=audit.build_pairs({'y':shared['target'],'sign':shared['sign']},audit_store(store),candidate,reference)
    actual=main.stats(ctx,np.asarray(ids,np.int64));expected=audit.statistics(independent,ids);assert_equal(actual,expected)
    assert_equal(main.cells(ctx,np.asarray(ids,np.int64)),audit.small_cells(independent,ids))
    if ids==[0] and reference=='original' and candidate=='current_affine':
        assert d.TRANSITIONS[ctx['transition'][0]]=='new_error' and actual['delta_sse']<0
    if ids==[1] and reference=='original' and candidate=='current_affine':
        assert d.TRANSITIONS[ctx['transition'][1]]=='corrected' and actual['delta_sse']>0


def test_demo_empties_equal_weights_and_nulls():
    shared,store=small_data();ctx=main.pair_context(shared,store,'current_affine','original')
    empty=main.stats(ctx,np.asarray([],int));full=main.stats(ctx,np.asarray([0,1],int))
    pairs=[main.contrast(full,full),main.contrast(full,empty),main.contrast(empty,full)]+[main.contrast(empty,empty)]*27
    assert_equal(main.standardized(pairs),audit.standardize(pairs))
    assert main.standardized(pairs)['paired_demo_n']==1
    assert main.standardized(pairs)['candidate_mse_null_demo_n']==29
    assert main.contrast(empty,empty)['missing_reason']=='both_empty'


@pytest.mark.parametrize('value',[.025,-.025,0.,-0.,1e-12,-1e-12,.2,-.2])
def test_independent_strict_scalar_classes_and_zero(value):
    for target in (.1,-.1,0.):assert d.CLASSES[audit.classify(value,target)]==d.error_category(target,value)


def test_conditional_projection_poisoned_nonpermitted_columns_untouched(synthetic):
    guard=access.Guard(synthetic['inputs'],limits=NoLimits());rows=selected=0
    with guard:
        guard.hash_inputs()
        for row in guard.project('frames'):
            assert set(row)==set(d.FRAME_COLUMNS);rows+=1
        for row in guard.project('curves'):
            permitted=row['update']==500 and row['role']=='probe'
            assert set(row)==set(d.CURVE_ID_COLUMNS+(d.CURVE_METRICS if permitted else []));selected+=permitted
        guard.complete()
    assert rows==75384 and selected==576
    assert sum(guard.record()['CSV_projected_cells'].values())==1331064
    assert guard.record()['byte_SHA_checks']==['frames','curves','frames','curves','frames','curves']


@pytest.mark.parametrize('operation',['raw_CSV','unlisted_CSV','archive_hash','np_load','fromfile','memmap','read_array','loadtxt','genfromtxt','os_open','FileIO','import_torch','import_environment','source_write','extra_fields','unknown_table','direct_reader','repeat'])
def test_access_bypasses_fail_closed(synthetic,tmp_path,operation):
    guard=access.Guard(synthetic['inputs'],limits=NoLimits());path=Path(synthetic['inputs']['tables']['frames']['path'])
    with guard:
        with pytest.raises((PermissionError,ValueError)):
            if operation=='raw_CSV':path.read_text()
            elif operation=='unlisted_CSV':Path('/never/open/other.csv').read_text()
            elif operation=='archive_hash':d.p.sha('/never/open/sealed.npz')
            elif operation=='np_load':np.load('/never/open/sealed.npz',allow_pickle=False)
            elif operation=='fromfile':np.fromfile('/never/open/sealed.npy')
            elif operation=='memmap':np.memmap('/never/open/sealed.npy')
            elif operation=='read_array':np.lib.format.read_array(None)
            elif operation=='loadtxt':np.loadtxt(path)
            elif operation=='genfromtxt':np.genfromtxt(path)
            elif operation=='os_open':os.open(path,os.O_RDONLY)
            elif operation=='FileIO':__import__('io').FileIO(path)
            elif operation=='import_torch':__import__('torch')
            elif operation=='import_environment':__import__('src.environments.libero_adapter')
            elif operation=='source_write':path.open('w')
            elif operation=='extra_fields':next(guard.project('frames',d.FRAME_COLUMNS+['target_0']))
            elif operation=='unknown_table':next(guard.project('old_NN'))
            elif operation=='direct_reader':
                gen=guard.project('frames');next(gen);csv.reader(gen.gi_frame.f_locals['handle'])
            elif operation=='repeat':
                gen=guard.project('frames');next(gen);gen.close();next(guard.project('frames'))
    assert guard.denied


@pytest.mark.parametrize('key,value',[('target_direction','positive'),('seed','20261106.0'),('task','4'),('tail20','1'),
                                     ('update','250'),('mode','history'),('target_2','nan'),('prediction_2','')])
def test_noncanonical_frame_syntax_rejected(key,value):
    row={name:'0' for name in d.FRAME_COLUMNS};row.update(task='7',seed=str(d.p.SEEDS[0]),arm=d.p.ARMS[0],mode='original',update='500',
                                                                       source_row='0',probe_fold='0',demo='0',action_step='1',target_direction='0',
                                                                       tail20='False',clip='False',component3_category='exact_match')
    row[key]=value
    with pytest.raises((ValueError,KeyError)):access.typed(row)


def test_projected_metrics_cannot_smuggle_extra_columns():
    with pytest.raises(ValueError,match='unlisted projected'):access.typed({**{key:'0' for key in d.CURVE_METRICS},'target_0':'secret'})


def test_incomplete_projection_not_counted_as_completed(synthetic):
    guard=access.Guard(synthetic['inputs'],limits=NoLimits())
    with guard:
        gen=guard.project('frames');next(gen);gen.close()
        assert guard.completed==[]
        with pytest.raises(ValueError,match='budget'):guard.complete()


@pytest.mark.parametrize('case',['SHA','header','duplicate_header','row_width'])
def test_malformed_CSV_stops_before_numeric_projection(synthetic,tmp_path,case):
    inputs=copy.deepcopy(synthetic['inputs']);path=tmp_path/'malformed.csv'
    columns=inputs['tables']['frames']['columns']
    with path.open('w',newline='') as handle:
        writer=csv.writer(handle)
        writer.writerow(columns[:-1] if case=='header' else columns+['task'] if case=='duplicate_header' else columns)
        writer.writerow(['bad_width'])
    inputs['tables']['frames'].update(d.p.pin(path))
    if case=='SHA':inputs['tables']['frames']['sha256']='f'*64
    guard=access.Guard(inputs,limits=NoLimits())
    with guard:
        with pytest.raises(ValueError):next(guard.project('frames'))
    assert guard.attempted==['frames'] and guard.completed==[]


class Rows:
    def __init__(self,rows):self.rows=rows
    def project(self,name):yield from copy.deepcopy(self.rows)


@pytest.mark.parametrize('case',['duplicate','demo','probe_fold','direction','target_across_mode'])
def test_independent_frame_ID_and_shared_target_guards(synthetic,case):
    with Path(synthetic['inputs']['tables']['frames']['path']).open(newline='') as handle:
        raw=next(csv.DictReader(handle));row=access.typed({k:raw[k] for k in d.FRAME_COLUMNS})
    rows=[row]
    if case=='duplicate':rows.append(copy.deepcopy(row))
    elif case=='demo':rows[0]['demo']=1
    elif case=='probe_fold':rows[0]['probe_fold']=2
    elif case=='direction':rows[0]['target_direction']=1
    else:
        other=copy.deepcopy(row);other.update(mode='bias_only',target_2=.5,target_direction=1);rows.append(other)
    with pytest.raises(ValueError):main.load_frames(Rows(rows),synthetic)
    with pytest.raises(ValueError):audit.reconstruct_frames(Rows(rows),synthetic)


@pytest.mark.parametrize('rows',[[],[0,0]])
def test_independent_curve_missing_or_duplicate_ID_rejected(rows):
    row={'task':7,'seed':d.p.SEEDS[0],'arm':d.p.ARMS[0],'fold':0,'mode':'bias_only','update':0,
         'role':'fit','phase':'all_source','direction':'all'}
    class Sink:
        def add(self,row):pytest.fail('unselected metrics must not produce output')
        def row(self,row):pytest.fail('unselected metrics must not produce output')
    payload=[row for _ in rows]
    with pytest.raises(ValueError):main.check_curves(Rows(payload),{}, {},Sink())
    with pytest.raises(ValueError):audit.audit_curves(Rows(payload),{}, {},Sink(),audit.Checks())


def test_audit_saved_float32_prediction_is_exact_even_near_zero(tmp_path):
    original=float(np.float32(1e-8));changed=float(np.nextafter(np.float32(original),np.float32(1.)))
    assert 0<changed-original<1e-10
    path=tmp_path/'paired.csv'
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['reference_prediction_2']);writer.writeheader();writer.writerow({'reference_prediction_2':changed})
    table=audit.StrictTable({**d.p.pin(path),'rows':1,'columns':['reference_prediction_2']},audit.Checks(),NoLimits())
    try:
        with pytest.raises(ValueError,match='exact'):table.row({'reference_prediction_2':original})
    finally:table.handle.close()


def test_output_read_allowlist_cannot_include_archive(synthetic):
    with pytest.raises(ValueError,match='only output CSV'):access.Guard(synthetic['inputs'],outputs=['/never/open/head.npz'],limits=NoLimits())


def test_unbound_execution_refused_before_payload(tmp_path):
    with pytest.raises(ValueError,match='execution binding required'):
        main.analyze(binding=tmp_path/'missing-binding.json')


def test_audit_does_not_import_primary_numerics():
    text=inspect.getsource(audit)
    assert 'from experiments import analyze_libero_task7_residual_sign_transition' not in text
    assert 'main.' not in text and 'd.pair_quantities(' not in text and 'd.validate_action_scalar(' not in text


def test_fresh_full_pipeline(synthetic,tmp_path):
    cfg=copy.deepcopy(synthetic);cfg['output']=str(tmp_path/'result.json');cfg['artifacts']=str(tmp_path/'artifacts')
    path=tmp_path/'science.json';path.write_text(json.dumps(cfg))
    env={**os.environ,**cfg['execution']['environment']}
    script='''import importlib,json,pathlib,sys
m=importlib.import_module(sys.argv[1]);cfg=json.loads(pathlib.Path(sys.argv[2]).read_text())
science={"path":"synthetic-science","sha256":"0"*64};binding={"path":"synthetic-binding","sha256":"1"*64}
fn=m.analyze_payload if sys.argv[3]=="analysis" else m.audit_payload
result=fn(cfg,science,binding);print(json.dumps({"status":result["status"]}))
'''
    for stage,module in [('analysis','experiments.analyze_libero_task7_residual_sign_transition'),('audit','experiments.audit_libero_task7_residual_sign_transition')]:
        result=subprocess.run([sys.executable,'-c',script,module,str(path),stage],cwd=d.ROOT,env=env,text=True,capture_output=True,timeout=195)
        (tmp_path/(stage+'.log')).write_text(result.stdout+result.stderr)
        assert result.returncode==0,result.stdout+result.stderr
    record=json.loads(Path(cfg['output']).read_text());audit_record=json.loads((Path(cfg['artifacts'])/'independent-audit.json').read_text())
    done=json.loads((Path(cfg['artifacts'])/'completion.json').read_text())
    assert done['total_input_CSV_data_passes']==4 and done['total_input_CSV_projected_cells']==2662128
    assert audit_record['scalar_checks']>5_000_000 and audit_record['primary_numerical_implementation_imported'] is False
    assert record['probe_scalar_parity_checks']==audit_record['probe_scalar_parity_checks']==8064
    assert record['cost']['peak_RSS_bytes']<=536870912 and audit_record['cost']['peak_RSS_bytes']<=536870912
    assert {name:ref['rows'] for name,ref in record['tables'].items()}==d.budget()['tables']
    assert len(record['seed_summary'])==72
    assert sum(Path(ref['path']).stat().st_size for ref in record['tables'].values())<=268435456
    with pytest.raises(ValueError,match='existing run'):main.analyze_payload(cfg,record['science_config'],record['binding'])
    with pytest.raises(ValueError,match='existing audit'):audit.audit_payload(cfg,record['science_config'],record['binding'])
