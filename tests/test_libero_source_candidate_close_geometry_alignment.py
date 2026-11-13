"""Boundary arithmetic and censoring tests; synthetic saved dictionaries only."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import analyze_libero_source_candidate_close_geometry_alignment as analysis
from experiments import audit_libero_source_candidate_close_geometry_alignment as audit


IDENTITY={'task':0,'seed':20261106,'arm':'A_original','global_initial_index':0}


def prefix(signs,touches=(),heights=None,plates=None,success=False):
    n=len(signs)
    heights=heights or [0.01*k for k in range(n+1)]
    plates=plates or [0.4-0.02*k for k in range(n+1)]
    public={'steps':n,'prefix_success':success,'capture_stop_reason':'success' if success else 'diagnostic_prefix_limit',
            'rows':[{'state_index':0,'action':None}]+[
                {'state_index':k,'action':[0]*6+[v]} for k,v in enumerate(signs,1)]}
    geometry={'rows':[{'state_index':k,'sim_time':.25+.05*k,'eef_to_bowl_xyz':.4-.03*k,
                       'eef_to_bowl_xy':.3-.02*k,'eef_minus_bowl_signed_z':.2-.01*k,
                       'bowl_height_change':heights[k],'bowl_to_plate_xyz':plates[k]+.01,
                       'bowl_to_plate_xy':plates[k],'bilateral_touch_proxy':k in touches} for k in range(n+1)]}
    return public,geometry


def closes(p,g):
    return [analysis.close_features(p,g,r,IDENTITY) for r in analysis.command_runs(p) if r['sign']==1]


def test_single_pulse_and_later_multistep_use_distinct_pre_states():
    p,g=prefix([1,-1,-1,1,1,-1])
    c=closes(p,g)
    assert [r['start'] for r in c]==[1,4]
    assert c[0]['complete_one_step_pulse'] and not c[1]['complete_one_step_pulse']
    assert c[0]['pre_eef_to_bowl_xyz']==.4
    assert c[1]['pre_eef_to_bowl_xyz']==pytest.approx(.31)
    episode=analysis.episode_features(p,g,c,IDENTITY)
    assert episode['first_multi_close_step']==4
    assert episode['first_multi_pre_eef_distance']==pytest.approx(.31)


def test_post_action_touch_is_one_control_interval_from_onset():
    p,g=prefix([-1,1,1,-1],touches=[2,3])
    row=closes(p,g)[0]
    assert row['pre_state']==1 and row['first_touch_state_in_close']==2
    assert row['first_touch_seconds_from_close_onset']==pytest.approx(.05)
    assert row['touch_to_end_delta_bowl_height_change']==pytest.approx(.01)


def test_preexisting_proxy_and_loss_inside_close_are_separate_from_grasp():
    p,g=prefix([-1,-1,1,1,1,-1],touches=[2,4,5])
    row=closes(p,g)[0]
    assert row['touch_at_pre'] and row['touch_at_end']
    assert row['touch_blocks_in_close']==2 and row['touch_boundary_count_in_close']==3
    assert row['first_touch_block_end_state']==2
    assert row['first_touch_block_ends_inside_close']
    assert row['first_touch_seconds_from_close_onset']==0


def test_no_touch_is_missing_and_immediate_open_uses_post_open_state():
    p,g=prefix([1,1,-1],heights=[0,.01,.1,.02])
    row=closes(p,g)[0]
    assert row['first_touch_state_in_close'] is None
    assert row['touch_to_end_delta_bowl_height_change'] is None
    assert row['immediate_open_action']==3
    assert row['immediate_open_post_bowl_height_change']==.02
    assert row['end_bowl_height_change']==.1


def test_terminal_single_positive_is_censored_not_complete_pulse():
    p,g=prefix([-1,1],success=True)
    row=closes(p,g)[0]
    assert row['right_censored'] and row['observed_length']==1
    assert not row['complete_one_step_pulse']
    assert row['following_command_sign'] is None and row['immediate_open_action'] is None


def test_zero_splits_runs_and_is_not_open():
    p,g=prefix([1,0,-1,1])
    assert [r['sign'] for r in analysis.command_runs(p)]==[1,0,-1,1]
    row=closes(p,g)[0]
    assert row['following_command_sign']==0 and row['immediate_open_action'] is None


def test_no_close_can_have_touch_and_height_changes():
    p,g=prefix([-1,-1,-1],touches=[1,2])
    row=analysis.episode_features(p,g,[],IDENTITY)
    assert row['close_runs']==0 and row['first_multi_pre_eef_distance'] is None
    assert row['first_touch_state']==1 and row['touch_blocks_in_prefix']==1
    assert row['first_touch_to_terminal_delta_bowl_height_change']==pytest.approx(.02)


def small_scope(monkeypatch):
    monkeypatch.setattr(analysis,'TASKS',[0]);monkeypatch.setattr(analysis,'SEEDS',[20261106])
    values={}
    for i in range(4):
        values[(0,20261106,'A_original',i)]=prefix([1,-1],touches=[1],success=True)
        values[(0,20261106,'B_fomaml',i)]=prefix([-1,1,1,1],touches=[3,4])
    return values


def test_paired_common_time_and_unavailable_tail_never_forward_filled(monkeypatch):
    indexed=small_scope(monkeypatch);rows=analysis.aligned_frames(indexed)
    last=[r for r in rows if r['global_initial_index']==0 and r['state_index']==4][0]
    assert not last['A_available'] and last['B_available'] and not last['jointly_available']
    assert last['A_bowl_height_change'] is None and last['B_minus_A_bowl_height_change'] is None
    reset=rows[0]
    assert reset['A_command_to_state'] is None and reset['B_command_to_state'] is None
    episodes=[]
    for (t,s,a,i),(p,g) in indexed.items():
        identity={'task':t,'seed':s,'arm':a,'global_initial_index':i}
        cr=[analysis.close_features(p,g,r,identity) for r in analysis.command_runs(p) if r['sign']==1]
        episodes.append(analysis.episode_features(p,g,cr,identity))
    pairs=analysis.pair_summaries(indexed,episodes,rows)
    assert pairs[0]['common_last_state']==2
    assert pairs[0]['B_common_max_height']==.02
    assert pairs[0]['B_common_terminal_plate_xy']==pytest.approx(.36)


def test_event_windows_preserve_negative_reset_indices_and_terminal_missing(monkeypatch):
    indexed=small_scope(monkeypatch);p,g=indexed[(0,20261106,'A_original',0)]
    cr=closes(p,g)
    windows=analysis.event_windows(indexed,cr)
    left=next(r for r in windows if r['absolute_state_index']==-1)
    assert left['reset_side_truncated'] and not left['A_available'] and left['A_bowl_height_change'] is None
    right=next(r for r in windows if r['absolute_state_index']==4)
    assert right['A_terminal_truncated'] and right['B_available']
    assert 'A_original:run1:close_onset_pre' in windows[0]['anchor_aliases']


@pytest.mark.parametrize('signs,touches',[([1,-1,1,1],[]),([-1,1,1,-1],[1,2]),([1,0,1,-1],[0,2]),([-1,-1],[1])])
def test_independent_scan_and_derivations_match_synthetic_boundary_examples(signs,touches):
    p,g=prefix(signs,touches=touches)
    assert analysis.command_runs(p)==audit.scan_runs(p)
    a=closes(p,g)
    b=[audit.expected_close(p,g,r,IDENTITY) for r in audit.scan_runs(p) if r['sign']==1]
    assert a==b
    assert analysis.episode_features(p,g,a,IDENTITY)==audit.expected_episode(p,g,b,IDENTITY)


@pytest.mark.parametrize('actual,expected',[('0',None),('nan',.1),('False',True),('1',0),('filled','')])
def test_independent_cells_reject_imputed_nonfinite_or_wrong_values(actual,expected):
    with pytest.raises(ValueError):audit.cell(actual,expected)


def test_missing_statistics_preserve_observed_count():
    result=analysis.stats([None,.02,None,.04])
    assert result['observed']==2 and result['median']==pytest.approx(.03)
    assert analysis.stats([None])=={'observed':0,'mean':None,'median':None,'minimum':None,'maximum':None}


def test_existing_artifacts_rejected_before_any_parent_read(tmp_path,monkeypatch):
    target=tmp_path/'old.json';target.write_text('keep')
    monkeypatch.setattr(analysis,'preflight',lambda *a:pytest.fail('read after existing output'))
    with pytest.raises(ValueError,match='existing'):analysis.run('missing','missing',target,tmp_path/'tables')
    assert target.read_text()=='keep'


def test_changed_analysis_definition_rejected_before_payload(tmp_path):
    config=tmp_path/'config.json';binding=tmp_path/'binding.json'
    config.write_text('{}');binding.write_text(json.dumps({'config_sha256':'0'*64}))
    with pytest.raises(ValueError,match='specification'):analysis.preflight(config,binding)


def test_payload_hash_mismatch_fails_before_json_decode(tmp_path):
    p=tmp_path/'saved.json';p.write_text('not JSON')
    with pytest.raises(ValueError,match='hash'):analysis.load_pin({'path':str(p),'sha256':'0'*64})


def test_parent_gate_accepts_exact_saved_negative_G_status():
    raw={'G_decision_unchanged':'failed_keep_H_sealed','new_training_updates':0}
    audited={'G_decision_unchanged':'failed_keep_H_sealed','actual_counts':{'prefixes':120}}
    done={'G_admission_remains_failed_H_sealed':True}
    analysis.validate_parent_scope(raw,audited,done,[{}]*359)


@pytest.mark.parametrize('wrong',[True,'passed'])
def test_parent_gate_rejects_boolean_or_changed_G_decision(wrong):
    with pytest.raises(ValueError,match='cohort'):
        analysis.validate_parent_scope({'G_decision_unchanged':wrong,'new_training_updates':0},
            {'G_decision_unchanged':'failed_keep_H_sealed','actual_counts':{'prefixes':120}},
            {'G_admission_remains_failed_H_sealed':True},[{}]*359)


def test_parent_gate_rejects_reopened_H():
    with pytest.raises(ValueError,match='cohort'):
        analysis.validate_parent_scope({'G_decision_unchanged':'failed_keep_H_sealed','new_training_updates':0},
            {'G_decision_unchanged':'failed_keep_H_sealed','actual_counts':{'prefixes':120}},
            {'G_admission_remains_failed_H_sealed':False},[{}]*359)


def test_wrong_source_role_rejected_before_path_reads():
    fake={'task':2,'seed':20261106,'arm':'A_original','global_initial_index':0,'initial_state_role':'qualification'}
    with pytest.raises(ValueError,match='cohort'):
        analysis.read_cohort({'capture_root':'/tmp/never'},{'capture':{'rollout':{'episode_files':[fake]}}})


def test_analysis_and_independent_auditor_import_without_numerical_runtime():
    code="""
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'torch','numpy','mujoco','robosuite','libero','h5py','PIL','src'}:
        raise AssertionError('forbidden runtime import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from experiments import analyze_libero_source_candidate_close_geometry_alignment
from experiments import audit_libero_source_candidate_close_geometry_alignment
print('standard_library_only')
"""
    completed=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],text=True,capture_output=True)
    assert completed.returncode==0,completed.stderr
    assert completed.stdout.strip()=='standard_library_only'
