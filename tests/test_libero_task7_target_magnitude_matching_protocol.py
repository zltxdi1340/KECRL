"""Synthetic-only bins, exact matching, weighting, coverage and metadata tests."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

from experiments import libero_task7_target_magnitude_matching_protocol as d

LENGTHS = [155,125,124,191,136,138,133,130,139,145,140,164,124,138,120,137,141,152,137,128,197,124,128,126,144,129,168,132,140,133]


def example():
    parents={key:{'formal_result':False} for key in d.PARENTS}
    science=parents['science'];science['scope']=copy.deepcopy(d.previous.SCOPE)
    science['inputs']={'action_lengths':LENGTHS.copy(), 'endpoints':[{'task':7,'seed':seed,'arm':arm,
        'checkpoint':{'path':f'/never/open/{seed}-{arm}.pt','sha256':'a'*64}} for seed in d.p.SEEDS for arm in d.p.ARMS]}
    ref={'path':str(Path(d.previous.ARTIFACTS)/'paired-frames.csv'),'sha256':'b'*64,'rows':64764,'columns':d.COLUMNS+['unused_poison']}
    tables={'paired-frames':ref}
    for key in ('result','audit','completion'):
        parents[key].update(science_config=copy.deepcopy(d.PARENTS['science']),binding=copy.deepcopy(d.PARENTS['binding']),
                            archive_array_checkpoint_reads_including_byte_hashes=0,new_training_forward_environment_neighbor_query_sealed_calls=0)
    parents['binding'].update(status='bound_before_payload',before_research_payload=True,science_config=copy.deepcopy(d.PARENTS['science']),
                              budget=d.previous.budget(),future_output=d.previous.OUTPUT,future_artifacts=d.previous.ARTIFACTS)
    parents['result'].update(status='complete_task7_residual_sign_transition_pending_audit',inputs=copy.deepcopy(science['inputs']),tables=tables,
                             all_parent_joint_cues_false_unchanged=True,G_failed_H_unrun_unchanged=True,SPT_Knowledge_Module_changes=False)
    parents['audit'].update(status='passed_independent_task7_residual_sign_transition_audit',result=copy.deepcopy(d.PARENTS['result']),
                            tables_verified=copy.deepcopy(tables),primary_numerical_implementation_imported=False,scalar_checks=6300492)
    parents['completion'].update(status='complete_independently_audited_task7_residual_sign_transition',result=copy.deepcopy(d.PARENTS['result']),
                                 audit=copy.deepcopy(d.PARENTS['audit']),all_parent_joint_cues_false_unchanged=True,G_failed_H_unrun_unchanged=True,
                                 SPT_Knowledge_Module_changes=False,total_input_CSV_data_passes=4,total_input_CSV_projected_cells=2662128,
                                 output_rows_verified=115728,seed_summary_rows_verified=72)
    cfg={'kind':'task7_derived_CSV_target_magnitude_within_demo_matching','version':1,'formal_result':False,
         'status':'scientific_frozen_execution_unbound','parents':copy.deepcopy(d.PARENTS),'scope':copy.deepcopy(d.SCOPE),
         'design':copy.deepcopy(d.DESIGN),'execution':copy.deepcopy(d.EXECUTION),'budget':d.budget(),'columns':d.COLUMNS.copy(),
         'inputs':d.select_inputs(parents),'output':d.OUTPUT,'artifacts':d.ARTIFACTS}
    return cfg,parents


def sample(target,reference):
    return d.frame_metrics(target,reference,-reference,0.,-2*reference)


def exact_cell(target,focus,control):
    return d.compare_cell([sample(target,x) for x in focus],[sample(target,x) for x in control],'exact_target',d.target_bits(target))


def test_exact_budgets_and_material_partition_bound():
    cfg,parents=example();assert d.validate_design(cfg,parents)
    b=d.budget();assert len(set(d.COLUMNS))==len(d.COLUMNS)==28 and len(d.METRICS)==12
    assert b['per_stage']['projected_cells']==64764*28==1813392
    assert b['total_input_CSV_data_passes']==2 and b['total_input_CSV_lexical_rows']==2*64764
    assert b['total_projected_cells']==3626784
    assert b['fixed_table_rows']['frame-strata']==6*3*3598
    assert b['fixed_table_rows']['bin-transitions']==6*3*3*2*7*4
    assert b['fixed_table_rows']['coarse-bin-cells']==6*3*3*2*2*30*7
    assert b['fixed_table_rows']['demo-matched']==6*3*3*2*2*2*30
    assert b['fixed_table_rows']['model-matched']==432 and b['fixed_table_rows']['common3-summary']==144
    assert b['variable_table_rows_max']['exact-target-cells']==6*3*2*3598==129528
    assert sum(b['fixed_table_rows'].values())+129528==b['total_output_table_rows_max']==256212
    assert set(cfg['inputs'])=={'paired_frames','endpoints','action_lengths'}
    assert not {'raw_shift','effective_shift','reference_abs'}.intersection(d.COLUMNS)


@pytest.mark.parametrize('target,expected',[(.025,0),(.049999,0),(.05,1),(.1,2),(.2,3),(.4,4),(.6,5),(.8,6),(1.,6)])
@pytest.mark.parametrize('sign',[-1,1])
def test_fixed_bins_exact_boundaries_and_symmetry(target,expected,sign):
    assert d.target_bin(sign*target)==expected


@pytest.mark.parametrize('target',[0.,.024999,-.024999,1.00001,float('nan'),float('inf'),True,'0.1',None])
def test_invalid_target_bin_inputs_rejected(target):
    with pytest.raises(ValueError):d.target_bin(target)


def test_float32_keys_no_rounding_caliper_or_cross_demo_matching():
    y=.125;next_y=struct.unpack('<f',struct.pack('<I',d.target_bits(y)+1))[0]
    assert d.target_bin(y)==d.target_bin(next_y)
    assert d.match_key(0,y,'exact_target')!=d.match_key(0,next_y,'exact_target')
    assert d.match_key(0,y,'coarse_bin')==d.match_key(0,next_y,'coarse_bin')
    assert d.match_key(0,y,'exact_target')!=d.match_key(1,y,'exact_target')
    assert d.match_key(0,y,'exact_target')!=d.match_key(0,-y,'exact_target')
    with pytest.raises(ValueError,match='key mismatch'):
        d.compare_cell([sample(y,.0625)],[sample(next_y,.0625)],'exact_target',d.target_bits(y))
    coarse=d.compare_cell([sample(y,.0625)],[sample(next_y,.0625)],'coarse_bin',d.target_bin(y))
    assert coarse['matched'] and coarse['delta']['target_abs']!=0
    with pytest.raises(ValueError,match='float32'):d.target_bits(.1)


@pytest.mark.parametrize('demo,scheme',[(True,'exact_target'),(30,'exact_target'),(-1,'coarse_bin'),(0,'nearest')])
def test_bad_matching_roles_rejected(demo,scheme):
    with pytest.raises(ValueError):d.match_key(demo,.125,scheme)


def test_metrics_normalization_and_zero_reference_without_epsilon():
    r=d.frame_metrics(-.125,0.,.25,0.,.25)
    assert r['target_abs']==.125 and r['reference_abs_over_target_abs']==0
    assert r['reference_relative_squared_error']==1 and r['candidate_relative_squared_error']==9
    assert r['delta_relative_squared_error']==8 and r['signed_effective_shift']==-.25
    assert r['absolute_effective_shift_over_target_abs']==2


@pytest.mark.parametrize('reference,candidate,raw',[(.1,.25,0.),(.125,1.1,0.),(.125,.25,float('nan')),(True,.25,0.)])
def test_invalid_frame_metric_scalars_rejected(reference,candidate,raw):
    with pytest.raises(ValueError):d.frame_metrics(.125,reference,candidate,0.,raw)


def test_exact_cell_uses_all_frames_and_is_symmetric():
    f=[.03125,.0625];c=[.015625]
    x=exact_cell(.125,f,c);y=exact_cell(.125,c,f)
    assert x['matched'] and x['focus_n']==2 and x['control_n']==1
    assert x['delta']['target_abs']==0 and x['focus_target_min']==x['control_target_max']==.125
    for key in d.METRICS:assert x['delta'][key]==pytest.approx(-y['delta'][key],abs=1e-12)
    duplicate=exact_cell(.125,f*10,c)
    assert duplicate['delta']==x['delta']  # Counts change; cell means do not gain weight.


def test_simpson_reversal_target_cells_and_weighting_are_not_raw_frame_effect():
    # In both exact target cells the focus output is larger; frame mixture reverses the raw sign.
    cells=[exact_cell(.125,[.03125]*9,[.015625]),exact_cell(.5,[.25],[.125]*9)]
    r=d.summarize_cells(cells,'exact_target')
    assert r['focus_coverage']==r['control_coverage']==1
    assert r['pooled_full']['delta']['reference_abs']<0
    assert r['pooled_common']['delta']['reference_abs']==r['pooled_full']['delta']['reference_abs']
    assert r['equal_cell']['delta']['reference_abs']==.0703125
    assert r['equal_cell']['delta']['target_abs']==0


def test_equal_demo_not_frame_or_cell_weight_and_all30_retained():
    demo0=d.summarize_cells([exact_cell(.125,[.03125]*9,[.015625]),exact_cell(.5,[.25],[.125]*9)],'exact_target')
    demo1=d.summarize_cells([exact_cell(.25,[.125],[.25])],'exact_target')
    empty=d.summarize_cells([],'exact_target')
    demos=[{'demo':i,**(demo0 if i==0 else demo1 if i==1 else empty)} for i in range(30)]
    r=d.summarize_demos(demos)
    assert r['eligible_demo_n']==2 and r['ineligible_demo_n']==28 and r['matched_cell_n']==3
    assert r['standardized']['delta']['reference_abs']==(.0703125-.125)/2
    assert r['standardized']['delta']['target_abs']==0
    assert r['focus_total_n']==r['control_total_n']==11 and r['focus_matched_n']==11
    assert r['pooled_full']['delta']['reference_abs']!=r['standardized']['delta']['reference_abs']
    with pytest.raises(ValueError,match='all30'):d.summarize_demos(demos[:-1])
    with pytest.raises(ValueError,match='all30'):d.summarize_demos([demos[0]]+demos[:-1])


def test_support_restriction_is_separately_recorded():
    cells=[exact_cell(.125,[.03125],[.0625]),exact_cell(.5,[.75]*9,[])]
    r=d.summarize_cells(cells,'exact_target')
    assert r['focus_total_n']==10 and r['focus_matched_n']==1 and r['focus_unmatched_n']==9
    assert r['focus_coverage']==.1 and r['control_coverage']==1
    assert r['pooled_full']['delta']['reference_abs']>0
    assert r['pooled_common']['delta']['reference_abs']==r['equal_cell']['delta']['reference_abs']==-.03125


def test_no_common_exact_key_is_null_even_when_both_groups_nonempty():
    cells=[exact_cell(.125,[.03125],[]),exact_cell(.25,[],[.125])]
    r=d.summarize_cells(cells,'exact_target')
    assert r['missing_reason']=='no_common_exact_target' and r['matched_cell_n']==0
    assert r['pooled_full']['delta']['reference_abs'] is not None
    assert r['pooled_common']['delta']['reference_abs'] is r['equal_cell']['delta']['reference_abs'] is None
    assert r['focus_coverage']==r['control_coverage']==0
    with pytest.raises(ValueError,match='duplicated'):d.summarize_cells([cells[0],cells[0]],'exact_target')


def test_coarse_empty_bins_preserved_and_no_matching_fallback():
    cells=[]
    for k in range(7):
        f=[sample(.125,.03125)] if k==2 else []
        c=[sample(.25,.125)] if k==3 else []
        cells.append(d.compare_cell(f,c,'coarse_bin',k))
    r=d.summarize_cells(cells,'coarse_bin')
    assert r['cell_n']==7 and r['missing_reason']=='no_common_target_bin' and r['matched_cell_n']==0
    assert sum(x['missing_reason']=='both_empty' for x in cells)==5
    with pytest.raises(ValueError,match='observed'):d.compare_cell([],[],'exact_target',d.target_bits(.125))


@pytest.mark.parametrize('focus,control,reason',[([],[],'both_empty'),([],[.0625],'focus_empty'),([.0625],[],'control_empty')])
def test_empty_reasons_and_zero_denominator_coverage_null(focus,control,reason):
    cell=d.compare_cell([sample(.125,x) for x in focus],[sample(.125,x) for x in control],'coarse_bin',2)
    r=d.summarize_cells([cell],'coarse_bin')
    assert r['missing_reason']==reason
    if not focus:assert r['focus_coverage'] is None
    if not control:assert r['control_coverage'] is None
    assert all(v is None for v in r['equal_cell']['delta'].values())


def test_all3_intersection_not_pairwise_or_any_seed():
    a=(0,1,d.target_bits(.125));b=(1,-1,d.target_bits(.25))
    assert d.common3_keys([{a,b},{a},{a,b}])=={a}
    assert d.common3_keys([{a},{a},set()])==set()
    with pytest.raises(ValueError,match='all3'):d.common3_keys([{a},{a}])


@pytest.mark.parametrize('section,key,value',[('scope','new_research_training_optimizer_or_forward_calls',1),('scope','original_Policy_updates',1000),
    ('scope','source_meta_query_accessed',True),('scope','sealed_roles_accessed',True),('design','bin_edges',[.025,.1,1.]),
    ('design','schemes',['coarse_bin']),('design','matching','nearest'),('execution','technical_retries',1),
    ('budget','total_input_CSV_data_passes',3)])
def test_scientific_or_access_drift_rejected(section,key,value):
    cfg,parents=example();cfg[section][key]=value
    with pytest.raises(ValueError):d.validate_design(cfg,parents)


@pytest.mark.parametrize('case',['audit_status','audit_hash','independence','cue','G','Knowledge','sealed','endpoint','binding',
    'old_budget','counter','archive_access','query_access','formal','CSV_path','CSV_rows','CSV_column','CSV_duplicate','endpoint_missing','extra_original_CSV'])
def test_provenance_roles_and_derived_input_fail_closed(case):
    cfg,parents=example()
    if case=='audit_status':parents['audit']['status']='pending'
    elif case=='audit_hash':parents['completion']['audit']['sha256']='f'*64
    elif case=='independence':parents['audit']['primary_numerical_implementation_imported']=True
    elif case=='cue':parents['result']['all_parent_joint_cues_false_unchanged']=False
    elif case=='G':parents['completion']['G_failed_H_unrun_unchanged']=False
    elif case=='Knowledge':parents['result']['SPT_Knowledge_Module_changes']=True
    elif case=='sealed':parents['science']['scope']['sealed_roles_accessed']=True
    elif case=='endpoint':parents['science']['scope']['readout_endpoint_updates']=250
    elif case=='binding':parents['binding']['before_research_payload']=False
    elif case=='old_budget':parents['binding']['budget']['total_CSV_data_passes']=5
    elif case=='counter':parents['completion']['total_input_CSV_projected_cells']=0
    elif case=='archive_access':parents['audit']['archive_array_checkpoint_reads_including_byte_hashes']=1
    elif case=='query_access':parents['result']['new_training_forward_environment_neighbor_query_sealed_calls']=1
    elif case=='formal':parents['science']['formal_result']=True
    elif case=='CSV_path':parents['result']['tables']['paired-frames']['path']='/never/open/curves.csv'
    elif case=='CSV_rows':parents['result']['tables']['paired-frames']['rows']=64763
    elif case=='CSV_column':parents['result']['tables']['paired-frames']['columns'].remove('target_2')
    elif case=='CSV_duplicate':parents['result']['tables']['paired-frames']['columns'].append('task')
    elif case=='endpoint_missing':parents['science']['inputs']['endpoints'].pop()
    elif case=='extra_original_CSV':cfg['inputs']['original_curves']={'path':'/never/open/curves.csv'}
    with pytest.raises(ValueError):d.validate_design(cfg,parents)


def local_metadata(tmp_path,monkeypatch):
    cfg,parents=example();files=[tmp_path/f'context{i}.py' for i in range(len(d.CONTEXT_FILES))];plan=tmp_path/'plan.md'
    for path in files+[plan]:path.write_text('synthetic metadata\n')
    cfg.update(code=[d.p.pin(path) for path in files],protocol=d.p.pin(plan))
    monkeypatch.setattr(d,'CONTEXT_FILES',files);monkeypatch.setitem(d.BINDING_FILES,'protocol',plan)
    loaded=[]
    def load(ref):
        key=next(key for key,value in d.PARENTS.items() if ref==value);loaded.append(key);return copy.deepcopy(parents[key])
    monkeypatch.setattr(d.p,'load_pin',load)
    path=tmp_path/'science.json';path.write_text(json.dumps(cfg));return path,cfg,loaded


def test_metadata_never_recursively_hashes_old_CSV_or_archives(tmp_path,monkeypatch):
    path,cfg,loaded=local_metadata(tmp_path,monkeypatch)
    monkeypatch.setattr(d.previous,'validate',lambda *a,**kw:pytest.fail('old source preflight forbidden'))
    sha=d.p.sha;seen=[]
    def checked(value):
        file=Path(value);assert file.parent==tmp_path and file.suffix in ('.py','.md');seen.append(file);return sha(file)
    monkeypatch.setattr(d.p,'sha',checked)
    assert d.validate(path)==cfg and loaded==list(d.PARENTS) and len(seen)==len(d.CONTEXT_FILES)+1


def test_byte_hash_option_only_one_approved_derived_CSV(tmp_path,monkeypatch):
    path,cfg,_=local_metadata(tmp_path,monkeypatch);ref=cfg['inputs']['paired_frames'];sha=d.p.sha;seen=[]
    def checked(value):
        file=Path(value);assert file.suffix not in ('.npz','.npy','.pt','.h5','.hdf5')
        if file.suffix=='.csv':
            assert str(file)==ref['path'];seen.append(str(file));return ref['sha256']
        return sha(file)
    monkeypatch.setattr(d.p,'sha',checked);assert d.validate(path,check_csv_byte_hash=True)==cfg and seen==[ref['path']]
    monkeypatch.setattr(d.p,'sha',lambda value:'bad' if str(value)==ref['path'] else sha(value))
    with pytest.raises(ValueError,match='CSV bytes'):d.validate(path,check_csv_byte_hash=True)


def test_separate_execution_binding_required_and_immutable(tmp_path,monkeypatch):
    cfg,_=example();path=tmp_path/'science.json';path.write_text(json.dumps(cfg));monkeypatch.setattr(d,'validate',lambda path:copy.deepcopy(cfg))
    with pytest.raises(ValueError,match='execution binding required'):d.require_binding(path,tmp_path/'missing.json')
    files={role:tmp_path/(role+'.py') for role in d.BINDING_FILES}
    for file in files.values():file.write_text('synthetic code\n')
    monkeypatch.setattr(d,'BINDING_FILES',files)
    binding={'status':'bound_before_payload','formal_result':False,'before_research_payload':True,'science_config':d.p.pin(path),
             'budget':d.budget(),'files':{k:d.p.pin(v) for k,v in files.items()}}
    target=tmp_path/'binding.json';target.write_text(json.dumps(binding));assert d.require_binding(path,target)==binding
    binding['budget']['total_projected_cells']+=1;target.write_text(json.dumps(binding))
    with pytest.raises(ValueError,match='execution identity'):d.require_binding(path,target)
    binding['budget']=d.budget();target.write_text(json.dumps(binding));files['guard'].write_text('changed\n')
    with pytest.raises(ValueError,match='execution role'):d.require_binding(path,target)


def test_import_stdlib_only_without_CSV_parser_in_fresh_process():
    script='import sys;from experiments import libero_task7_target_magnitude_matching_protocol;assert not set(sys.modules).intersection({"numpy","torch","csv","h5py","PIL","libero","mujoco"})'
    result=subprocess.run([sys.executable,'-c',script],cwd=d.ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'},text=True,capture_output=True,timeout=15)
    assert result.returncode==0,result.stderr
