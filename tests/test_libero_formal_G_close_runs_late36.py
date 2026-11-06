from copy import deepcopy
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from experiments import analyze_libero_formal_G_close_runs_late36 as d
from test_libero_formal_G_validation_BC_traces import episode,model


def case(pulse=True):
    b=episode([-1,-1]+[1]*6)
    c=episode(([1,-1,-1,-1]+[1]*296) if pulse else [1]*300,arm='C_pooled_bc',success=False)
    return b,c,{'B_first_close_step':'3','C_first_close_step':'1'}


def test_single_step_then_multi_step_is_preserved_and_original_counterexample_not_erased():
    row,runs,windows,anchors=d.analyze_case(*case())
    assert row['original_first_positive_winner_later_preserved'] is True
    assert row['loser_first_positive_kind']=='one_step_completed'
    assert row['winner_first_positive_kind']=='multi_step_right_censored'
    assert row['loser_first_run_relation']=='completed_one_step_before_winner'
    assert row['winner_vs_loser_first_multi_order']=='observed_left_earlier'
    assert row['B_first_positive_start']==row['B_first_multi_step_start']==3
    assert row['C_first_positive_start']==1 and row['C_first_multi_step_start']==5
    assert anchors==3 and len(windows)==33
    assert len([r for r in runs if r['sign']>0])==3


def test_early_continuous_close_can_remain_a_genuine_multi_step_counterexample():
    row,_,_,_=d.analyze_case(*case(pulse=False))
    assert row['loser_first_positive_kind']=='multi_step_right_censored'
    assert row['loser_first_positive_run_still_active_at_winner_first_positive']
    assert row['loser_positive_age_at_winner_first_positive']==3
    assert row['winner_vs_loser_first_multi_order']=='observed_left_later'


def test_terminal_one_step_positive_is_censored_not_a_completed_short_pulse():
    ep=episode([-1,-1,1]);runs=d.command_segments(ep);feature=d.episode_features(ep,runs)
    assert feature['first_positive_kind']=='one_step_right_censored'
    assert feature['first_multi_step_start'] is None and feature['first_positive_observed_length']==1
    assert runs[-1]['right_censored'] is True


def test_neutral_splits_runs_and_run_paths_align_with_pre_action_state():
    ep=episode([1,0,1,1,-1]);runs=d.command_segments(ep)
    assert [(r['sign'],r['start'],r['end'],r['length']) for r in runs]==[(1,1,1,1),(0,2,2,1),(1,3,4,2),(-1,5,5,1)]
    assert runs[0]['joint_path']==pytest.approx(.01) and runs[2]['joint_path']==pytest.approx(.02)
    feature=d.episode_features(ep,runs)
    assert feature['first_positive_kind']=='one_step_completed' and feature['first_multi_step_start']==3


def test_no_close_is_explicit_and_not_given_a_multi_step_timestamp():
    ep=episode([-1,0,-1]);feature=d.episode_features(ep,d.command_segments(ep))
    assert feature['first_positive_kind']=='not_observed'
    assert feature['first_positive_start'] is feature['first_multi_step_start'] is None


def test_neighborhood_uses_absolute_steps_and_never_wraps_or_fills_ended_success():
    _,_,windows,_=d.analyze_case(*case())
    before=next(w for w in windows if w['anchor_step']==1 and w['offset']==-1)
    assert before['absolute_step']==0 and before['reset_side_truncated']
    assert not before['B_available'] and before['B_gripper_command'] is before['joint_gap_post_l2'] is None
    after=next(w for w in windows if w['anchor_step']==5 and w['offset']==5)
    assert after['absolute_step']==10 and after['B_terminal_truncated'] and after['C_available']
    assert after['B_joint_step_l2'] is after['translation_gap_l2'] is None and after['C_gripper_command']==1
    event=next(w for w in windows if w['anchor_step']==3 and w['offset']==0)
    assert event['B_command_sign']==1 and event['C_command_sign']==-1 and event['window_phase']=='event'


def test_same_absolute_anchor_is_deduplicated_with_all_aliases_retained():
    b,c,_=case(pulse=False);br,cr=d.command_segments(b),d.command_segments(c)
    rows,n=d.event_windows(b,c,d.episode_features(b,br),d.episode_features(c,cr),br,cr)
    assert n==2 and len(rows)==22
    assert next(r for r in rows if r['anchor_step']==1)['anchor_aliases']=='C_first_positive;C_first_multi_step'


@pytest.mark.parametrize('kind',['task','seed','arm','role','index','digest','prior_timing','both_success','no_close'])
def test_wrong_pair_selection_or_changed_original_close_fails_closed(kind):
    b,c,p=case()
    if kind=='task':c['task_id']=6
    elif kind=='seed':c['training_seed']+=1
    elif kind=='arm':c['arm']='A_original'
    elif kind=='role':c['initial_state_role']='qualification'
    elif kind=='index':c['initial_state_index']=11
    elif kind=='digest':c['reset_observation_sha256']='changed'
    elif kind=='prior_timing':p['B_first_close_step']='2'
    elif kind=='both_success':c['success']=True
    else:b=episode([-1]*8)
    with pytest.raises(ValueError):d.analyze_case(b,c,p)


def test_import_needs_no_learning_or_environment_libraries():
    program='''
import builtins
old=builtins.__import__
def check(name,*a,**kw):
 if name.split('.')[0] in ('torch','h5py','libero','robosuite','mujoco','src'):raise AssertionError(name)
 return old(name,*a,**kw)
builtins.__import__=check
from experiments import analyze_libero_formal_G_close_runs_late36
'''
    subprocess.run([sys.executable,'-c',program],cwd=d.ROOT,check=True)


def test_existing_output_is_not_overwritten_or_used_to_trigger_input_reads(tmp_path,monkeypatch):
    p=tmp_path/'prior.json';p.write_text('kept')
    monkeypatch.setattr(d,'preflight',lambda *a:pytest.fail('input read'))
    with pytest.raises(ValueError):d.run(None,None,p,tmp_path/'artifacts')
    assert p.read_text()=='kept'


@pytest.fixture
def bound_saved(tmp_path,monkeypatch):
    folder=tmp_path/'traces';folder.mkdir();monkeypatch.setattr(d.trace,'TRACE_ROOT',folder)
    config=json.loads((d.ROOT/'configs/libero-formal-G-close-runs-late36-readonly_v1.json').read_text())
    ids={tuple(k) for k in config['case_ids']};raw_rows=[];pairs=[];context=[];counter=[]
    for task in d.TASKS:
      for seed in d.SEEDS:
        for index in d.trace.INDICES:
          chosen=(task,seed,index) in ids
          pairs.append({'task':task,'seed':seed,'global_initial_index':index,'winner_close_order':'observed_left_later' if chosen else '',
                        'outcome':'B_only' if chosen else 'both_failure','B_first_close_step':3 if chosen else '', 'C_first_close_step':1 if chosen else ''})
          if chosen:counter.append({'category':'winner_closed_later','task':task,'seed':seed,'global_initial_index':index,'winner':'B',
                                    'outcome':'B_only','B_first_close_step':3,'C_first_close_step':1})
          for arm in d.ARMS:
            ep=episode(([-1,-1]+[1]*6) if arm=='B_fomaml' else ([1,-1,-1,-1]+[1]*296),
                       task=task,seed=seed,state=index,arm=arm,success=arm=='B_fomaml')
            p=folder/f'task{task}-seed{seed}-{arm}-global{index:02d}.json'
            if chosen:p.write_text(json.dumps(ep))
            raw_rows.append({'task_id':task,'training_seed':seed,'arm':arm,'initial_state_index':index,'local_initial_state_index':index-10,
                'initial_state_role':'candidate_acceptance','success':ep['success'] if chosen else False,
                'steps':ep['steps'] if chosen else 300,'terminal_reason':ep['terminal_reason'] if chosen else 'time_limit',
                'reset_observation_sha256':'c'*64,'path':str(p),'sha256':d.sha(p) if chosen else 'd'*64})
            context.append({'task_id':task,'training_seed':seed,'arm':arm,'initial_state_index':index,
                            'first_close_run_length':6 if chosen and arm=='B_fomaml' else 1 if chosen else '',
                            'first_close_run_right_censored':'True' if chosen and arm=='B_fomaml' else 'False' if chosen else ''})
    raw_rows += [{'task_id':0,'training_seed':d.SEEDS[0],'arm':'A_original','initial_state_index':0} for _ in range(4800)]
    def save_json(name,value):
        p=tmp_path/name;p.write_text(json.dumps(value));return {'path':str(p),'sha256':d.sha(p)}
    def save_csv(name,rows):
        p=tmp_path/name
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return {'path':str(p),'sha256':d.sha(p)}
    tr=save_json('training.json',{'stage':'G','models':[model(t,s,a) for t in d.TASKS for s in d.SEEDS for a in d.ARMS]})
    raw=save_json('result.json',{'stage':'G','training':tr,'rollout':{'episode_files':raw_rows}})
    decision={'status':'synthetic_negative_G','passed':False}
    audit=save_json('audit.json',{'status':'passed_complete_audit','stage':'G','decision':decision,'result_sha256':raw['sha256']})
    done=save_json('done.json',{'status':'complete_audited_G','decision':decision,'result':raw,'audit':audit})
    ps,es,cs=save_csv('pairs.csv',pairs),save_csv('episodes.csv',context),save_csv('counterexamples.csv',counter)
    prior=save_json('prior.json',{'G_decision_unchanged':decision,'csv_artifacts':{'pairs':ps,'episodes':es}})
    prior_summary=save_json('summary.json',{'counterexamples':cs})
    config['trace_root']=str(folder);config['read_budget']['policy_action_rows']=11088;config['read_budget']['public_state_rows']=11160
    config['read_budget']['maximum_sign_run_rows']=11088
    config['inputs']={'result':raw,'training':tr,'completion':done,'complete_audit':audit,'prior_trace_analysis':prior,
                      'prior_trace_summary':prior_summary,'counterexamples':cs,'prior_pairs':ps,'prior_episodes':es}
    cp=save_json('config.json',config)
    files={'runner':Path(d.__file__),'tests':Path(__file__),'trace_helpers':Path(d.trace.__file__),
           'command_metrics':Path(d.trace.metrics.__file__),'public_validator':Path(d.trace.public_audit.__file__)}
    binding={'config_sha256':cp['sha256'],'files':{k:{'path':str(p),'sha256':d.sha(p)} for k,p in files.items()}}
    bp=save_json('binding.json',binding)
    return Path(cp['path']),Path(bp['path']),config,binding


def test_complete72_trace_pipeline_does_not_read_the_other1528_raw_episodes(bound_saved,tmp_path):
    cp,bp,_,_=bound_saved
    r=d.run(cp,bp,tmp_path/'out.json',tmp_path/'artifacts')
    assert r['actual_access']['trace_files_validated']==72 and r['actual_access']['pairs_completed']==36
    assert r['actual_access']['policy_action_rows']==11088 and r['actual_access']['public_state_rows']==11160
    assert r['actual_access']['neighborhood_anchors']==108 and r['actual_access']['neighborhood_rows']==1188
    assert r['cached_context']['other_raw_episodes_opened']==0 and r['G_decision_unchanged']['passed'] is False
    assert r['groups'][0]['first_multi_orders']=={'observed_left_earlier':36}
    assert all(x['original_first_positive_winner_later_preserved'] for x in r['case_rows'])


@pytest.mark.parametrize('kind',['hash','case_ids','window','read_budget','input_hash'])
def test_preflight_rejects_scope_or_lineage_changes_without_opening_traces(bound_saved,kind,monkeypatch):
    cp,bp,config,binding=bound_saved
    if kind=='hash':binding['config_sha256']='0'*64
    elif kind=='case_ids':config['case_ids'].pop()
    elif kind=='window':config['neighborhood_half_window']=10
    elif kind=='read_budget':config['read_budget']['unique_episode_trace_files']=74
    else:config['inputs']['counterexamples']['sha256']='0'*64
    cp.write_text(json.dumps(config))
    if kind!='hash':binding['config_sha256']=d.sha(cp)
    bp.write_text(json.dumps(binding));original=Path.read_bytes
    def check(self,*a,**kw):
        assert self.parent!=d.trace.TRACE_ROOT,'trace read before preflight passed'
        return original(self,*a,**kw)
    monkeypatch.setattr(Path,'read_bytes',check)
    with pytest.raises(ValueError):d.preflight(cp,bp)
