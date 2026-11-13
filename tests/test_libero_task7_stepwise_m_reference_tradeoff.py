"""Human scalar fixtures only; never decode the research P/U JSON in tests."""
import copy
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import pytest
from experiments import libero_task7_stepwise_m_reference_tradeoff_protocol as d
from experiments import libero_task7_stepwise_m_reference_tradeoff_access as a
from experiments import analyze_libero_task7_stepwise_m_reference_tradeoff as main
from experiments import audit_libero_task7_stepwise_m_reference_tradeoff as audit


@pytest.fixture(scope='module')
def canonical():return d.create()


def human(cfg,tmp_path):
    cfg=copy.deepcopy(cfg);inputs={}
    for engine in ('P','U'):
        values=dict(heads=[],pairs=[],zero=[],bisector=[])
        for i,spec in enumerate(cfg['identities']):
            ident={f:spec[f] for f in d.IDS};focus=(spec['arm'],spec['seed'],spec['fold']) in d.FOCUS
            start=main.point(1.,.3,1./7+.3);control=main.point(.9,.2,.9/7+.2)
            m,p=(.89,.199) if not focus else [(.91,.199),(.91,.197),(.89,.202),(.91,.202)][d.FOCUS.index((spec['arm'],spec['seed'],spec['fold']))%4]
            candidate=main.point(m,p,m/7+p);zero=main.point(.99,.29,.99/7+.29);bisector=main.point(.8,.23,.8/7+.23)
            for opt,end in zip(d.ARMS,(control,candidate)):
                values['heads'].append(dict(**ident,optimization_arm=opt,extra_updates=50,terminal_reason='update_cap',
                    initial_component3_mse=start['M'],initial_penalty=start['P'],initial_trainable_objective=start['T'],
                    final_component3_mse=end['M'],final_penalty=end['P'],final_trainable_objective=end['T'],
                    accepted_MSE_step_increases=(23 if i<17 else 16) if opt==d.ARMS[0] else 0,joint_cue=True))
            pair=dict(**ident,cohort=spec['cohort'],candidate_progress_from500=True,step_M_invariant_met=True)
            for name,reference in [('control',control),('zero',zero),('bisector',bisector)]:
                delta=main.delta(reference,candidate)
                pair['candidate_minus_'+name+'_MSE']=delta['dM'];pair['candidate_minus_'+name+'_T']=delta['dT']
                pair['margin_control_pair_cue' if name=='control' else name+'_anchor_cue']=delta['joint']
            pair['paired_joint_cue']=False;values['pairs'].append(pair)
            for name,point in [('zero',zero),('bisector',bisector)]:values[name].append(dict(**ident,extra_updates=50,component3_mse=point['M'],penalty=point['P'],trainable_objective=point['T']))
        raw=dict(status='complete_stepwise_M_gate_pending_same_parameter_audit' if engine=='P' else 'passed_independent_stepwise_M_gate_audit',
          formal_result=False,science_config=d.PARENTS['science'],binding=d.PARENTS['binding'],heads=values['heads'],pairs=values['pairs'],
          historical_control_parity={'fixed_anchors':{k:values[k] for k in ('zero','bisector')}},
          states=[{'unselected':'not semantically accessed'}],trials=[{'unselected':[999]}],parameters=[999])
        if engine=='U':raw['primary']=inputs['P']
        path=tmp_path/(engine+'.json');path.write_text(json.dumps(raw));inputs[engine]=d.p.pin(path)
    cfg['inputs']=inputs;return cfg


@pytest.mark.parametrize('m,p',[(.89,.202),(.91,.197),(.91,.202),(.9,.2),(.89,.199)])
def test_independent_gap_categories_and_conservation(m,p):
    x=main.point(.9,.2,.9/7+.2);y=main.point(m,p,m/7+p)
    assert audit.compare(main.delta(x,y),audit.difference(x,y))>5
    gap=main.delta(x,y);assert gap['dT']==pytest.approx(gap['dM_scaled']+gap['dP'])


def test_raw_tiny_positive_error_is_not_reclassified_by_accounting_tolerance():
    x={'M':1.,'P':.1,'T':1./7+.1};m=math.nextafter(1.,math.inf);y={'M':m,'P':.09,'T':m/7+.09}
    for fn in (main.delta,audit.difference):
        r=fn(x,y);assert r['dM']>0 and r['dM']<1e-10 and r['category']=='M_higher_T_lower' and not r['joint']


def test_zero_denominators_keep_ratios_null():
    x={'M':1.,'P':.1,'T':1./7+.1}
    for fn in (main.delta,audit.difference):
        r=fn(x,x);assert r['penalty_covers_M_cost'] is r['penalty_consumes_M_saving'] is None


@pytest.mark.parametrize('value',[True,-.1,float('nan'),float('inf')])
def test_invalid_or_nonconserving_points_fail_closed(value):
    for fn in (main.point,audit.terms):
        with pytest.raises(ValueError):fn(value,.1,.2)


def test_metadata_factory_never_opens_numeric_sources(canonical,monkeypatch):
    pin0=d.p.pin
    def pin(path):
        assert str(path) not in {r['path'] for r in d.INPUTS.values()};assert Path(path).suffix not in ('.npz','.csv','.hdf5')
        return pin0(path)
    monkeypatch.setattr(d.p,'pin',pin)
    assert d.create()==canonical
    assert canonical['budget']['projected_scalar_fields_per_stage']==2448


@pytest.mark.parametrize('field',['parameters','states','unknown'])
def test_projection_cannot_export_unlisted_fields(field):
    with pytest.raises(ValueError):a.scalar('heads',{},[field])


@pytest.mark.parametrize('operation',['read','write','os_open','FileIO','numpy','torch','source_SHA','CSV','HDF5'])
def test_guard_rejects_unlisted_io_and_runtime(canonical,tmp_path,operation):
    g=a.Guard(canonical);path=tmp_path/'unused.json';path.write_text('{}')
    with pytest.raises(PermissionError):
        with g:
            if operation=='read':path.read_text()
            elif operation=='write':path.write_text('new')
            elif operation=='os_open':os.open(path,os.O_RDONLY)
            elif operation=='FileIO':__import__('io').FileIO(path)
            elif operation in ('numpy','torch'):__import__(operation)
            else:d.p.sha(tmp_path/('head.npz' if operation=='source_SHA' else 'a.csv' if operation=='CSV' else 'a.hdf5'))
    assert g.denied and path.read_text()=='{}'


def test_all18_projection_and_independent_accounting(canonical,tmp_path):
    cfg=human(canonical,tmp_path);g=a.Guard(cfg);rows=[];focus=[];ind=[];chosen=[]
    with g:
        for engine in ('P','U'):
            data=g.project(engine);actual,selected=main.analyze(engine,data,cfg);expected,targets=audit.reconstruct(engine,data,cfg)
            audit.compare(actual,expected);audit.compare(selected,targets)
            assert all(set(v)==set(d.PROJECTIONS[table]) for table,values in data.items() for v in values)
            rows.extend(actual);focus.extend(selected);ind.extend(expected);chosen.extend(targets)
        g.complete()
    actual=main.finish_tables(rows,focus,cfg);expected=audit.reconstruct_tables(ind,chosen,cfg)
    assert audit.compare(actual,expected)>0
    assert {k:len(v) for k,v in actual.items()}==cfg['budget']['output_rows']
    assert g.record()['projected_rows']==180 and g.record()['projected_scalar_fields']==2448
    assert g.record()['source_JSON_passes']==2 and g.record()['source_SHA_checks']==4
    altered=copy.deepcopy(actual);altered['focus_current'][0]['dP']+=1e-6
    with pytest.raises(ValueError):audit.compare(altered,expected)


@pytest.mark.parametrize('corruption',['duplicate','missing','foreign','wrong_flag','wrong_delta','nonfinite','candidate_Mup'])
def test_malformed_projection_or_saved_cue_is_rejected(canonical,tmp_path,corruption):
    cfg=human(canonical,tmp_path);path=Path(cfg['inputs']['P']['path']);raw=json.loads(path.read_text())
    if corruption=='duplicate':raw['pairs'][0]=dict(raw['pairs'][1])
    elif corruption=='missing':raw['heads'].pop()
    elif corruption=='foreign':raw['pairs'][0]['task']=2
    elif corruption=='wrong_flag':raw['pairs'][0]['margin_control_pair_cue']=not raw['pairs'][0]['margin_control_pair_cue']
    elif corruption=='wrong_delta':raw['pairs'][0]['candidate_minus_control_MSE']+=1e-5
    elif corruption=='nonfinite':raw['heads'][0]['final_penalty']=float('nan')
    else:raw['heads'][1]['accepted_MSE_step_increases']=1
    path.write_text(json.dumps(raw));cfg['inputs']['P']=d.p.pin(path)
    with pytest.raises(ValueError):
        with a.Guard(cfg) as g:main.analyze('P',g.project('P'),cfg)


def test_missing_binding_stops_before_source_or_directory(canonical,tmp_path,monkeypatch):
    science=tmp_path/'science.json';science.write_text(json.dumps(canonical));monkeypatch.setattr(d,'CONFIG',science)
    monkeypatch.setattr(d,'BINDING',tmp_path/'missing.json');monkeypatch.setattr(d,'ARTIFACTS',tmp_path/'none')
    for fn in (main.execute,main.run,audit.run):
        with pytest.raises(FileNotFoundError):fn()
    assert not d.ARTIFACTS.exists()


def test_repeated_projection_and_incomplete_budget_fail_closed(canonical,tmp_path):
    cfg=human(canonical,tmp_path);g=a.Guard(cfg)
    with pytest.raises(PermissionError):
        with g:g.project('P');g.project('P')
    g=a.Guard(cfg)
    with pytest.raises(ValueError):
        with g:g.project('P');g.complete()


def test_fresh_full_pipeline_and_independent_audit(canonical,tmp_path):
    cfg=human(canonical,tmp_path);path=tmp_path/'human-science.json';path.write_text(json.dumps(cfg));binding=tmp_path/'human-binding.json';binding.write_text('{}')
    for module,fn in [('analyze_libero_task7_stepwise_m_reference_tradeoff','execute'),('audit_libero_task7_stepwise_m_reference_tradeoff','run')]:
        code=f'''import json
from pathlib import Path
from experiments import libero_task7_stepwise_m_reference_tradeoff_protocol as d
from experiments import {module} as module
cfg=json.loads(Path({str(path)!r}).read_text())
assert all(Path(r['path']).is_relative_to(Path({str(tmp_path)!r})) for r in cfg['inputs'].values())
d.CONFIG=Path({str(path)!r});d.BINDING=Path({str(binding)!r});d.ARTIFACTS=Path({str(tmp_path/'human-output')!r});d.OUTPUT=d.ARTIFACTS/'analysis.json';d.AUDIT=d.ARTIFACTS/'independent-audit.json'
d.preflight=lambda:(cfg,{{'CPU_environment':cfg['execution']['environment']}})
print(module.{fn}()['status'])
'''
        env=os.environ.copy();env.update(cfg['execution']['environment'])
        child=subprocess.run([sys.executable,'-c',code],env=env,capture_output=True,text=True,timeout=60)
        (tmp_path/(module+'.log')).write_text(child.stdout+child.stderr)
        assert child.returncode==0,child.stdout+child.stderr
    done=json.loads((tmp_path/'human-output/completion.json').read_text())
    assert done['source_JSON_passes']==4 and done['source_SHA_checks']==8 and done['checks']>0
    assert 'analyze_libero_task7_stepwise_m_reference_tradeoff' not in Path(audit.__file__).read_text().split('def run():')[0]


def test_direct_or_reordered_hash_api_rejected(canonical,tmp_path):
    cfg=human(canonical,tmp_path)
    for method,args in [('hash',(cfg['inputs']['P'],)),('source_hash',('U','before')),('source_hash',('P','after'))]:
        g=a.Guard(cfg)
        with pytest.raises(PermissionError):
            with g:getattr(g,method)(*args)


@pytest.mark.parametrize('field,value',[('output','/tmp/unfrozen.json'),('scope',{}),('budget',{})])
def test_changed_config_is_rejected_before_any_source(canonical,tmp_path,monkeypatch,field,value):
    changed=copy.deepcopy(canonical);changed[field]=value;path=tmp_path/'changed-science.json';path.write_text(json.dumps(changed))
    monkeypatch.setattr(d,'CONFIG',path)
    with pytest.raises(ValueError):d.preflight()
