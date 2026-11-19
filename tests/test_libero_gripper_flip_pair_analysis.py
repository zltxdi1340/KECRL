"""Invented trace pairs; no original results, models, initial banks or physics."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments import analyze_libero_gripper_flip_pairs as main
from experiments import audit_libero_gripper_flip_pairs as audit
from experiments import libero_gripper_flip_pair_protocol as p


def source(condition,success=False,early_raw_difference=False):
    steps=20 if success else 300;trace=[dict(step=0,joint_state=[0.]*7,gripper_state=[.04,-.04])];run=0
    for i in range(1,steps+1):
        raw=[.01]*6+[-1. if i<3 else 1.]
        if condition=='confirm5' and i>=4:raw[0]=.02
        if condition=='confirm5' and early_raw_difference and i==2:raw[0]=.015
        run=min(6,run+1) if raw[6]>0 else 0;masked=condition=='confirm5' and 0<run<=5
        applied=raw[:6]+[0. if masked else raw[6]]
        gap=.04 if i<(8 if condition=='confirm5' else 3) else .03
        trace.append(dict(step=i,raw_action=raw,applied_action=applied,joint_state=[0.]*7,
            gripper_state=[gap,-gap],positive_run=run,masked=masked))
    return dict(status='complete',task=7,seed=p.SEEDS[0],policy_arm=p.ARMS[0],condition=condition,
        initial_state_index=0,initial_role='candidate_nonregression',steps=steps,success=success,
        reset_observation_sha256='a'*64,trace=trace)


def case():return dict(case_id='flip00',seed=p.SEEDS[0],policy_arm=p.ARMS[0],initial_state_index=0,outcome='gained')


def test_expected_intervention_then_feedback_and_gripper_events():
    u,c=source('unchanged'),source('confirm5',True);row=main.analyze_pair(case(),u,c)
    assert row==audit.analyze_pair(case(),u,c)
    assert row['first_masked_step']==row['first_applied_difference']==3
    assert row['first_arm_command_difference']==4 and row['first_gripper_state_difference']==3
    assert row['pre_mask_recorded_prefix_exact'] is True and row['first_applied_change_gripper_only'] is True
    assert row['first_applied_positive_delta']==5 and row['episodes']['confirm5']['first_applied_positive']==8
    assert row['failure_stage']=='unknown_no_RGB_object_pose_or_contact' and not row['grasp_confirmed_from_finger_state']
    assert json.loads(json.dumps(row,allow_nan=False))==row


def test_preexisting_raw_difference_is_reported_not_assigned_to_confirmation():
    u,c=source('unchanged'),source('confirm5',True,True);row=main.analyze_pair(case(),u,c)
    assert row==audit.analyze_pair(case(),u,c)
    assert row['first_raw_difference']==2 and row['first_masked_step']==3
    assert row['pre_mask_recorded_prefix_exact'] is False and row['first_applied_change_gripper_only'] is False


def test_short_positive_requests_fully_filtered_and_never_called_grasps():
    c=source('confirm5');c['trace']=c['trace'][:6];c['steps']=5
    a=main.project(c);events=main.episode_events(c,a)
    assert events==audit.episode_events(c) and events['first_applied_positive'] is None
    assert events['short_positive_runs']==1 and events['masked_steps']==3


@pytest.mark.parametrize('kind',['nan','wrong_mask','extra_rgb','bad_identity','hash'])
def test_malformed_or_out_of_scope_source_rejected(tmp_path,kind):
    value=source('unchanged');descriptor_case=case()
    if kind=='nan':value['trace'][1]['raw_action'][0]=float('nan')
    elif kind=='wrong_mask':value['trace'][1]['masked']=True
    elif kind=='extra_rgb':value['trace'][1]['agentview_rgb']=[0]
    elif kind=='bad_identity':value['task']=8
    path=tmp_path/'invented.json';path.write_text(json.dumps(value));item=dict(p.pin(path),steps=value['steps'],success=value['success'])
    if kind=='hash':item['sha256']='0'*64
    with pytest.raises(ValueError):
        loaded,_=p.read_episode(item,descriptor_case,'unchanged');main.project(loaded)


def full_metadata():
    rows=[];pair=0
    for seed in p.SEEDS:
        for arm in p.ARMS:
            for i in range(20):
                for condition in p.CONDITIONS:
                    success=(condition=='confirm5') if pair<7 else (condition=='unchanged') if pair<13 else pair<20
                    rows.append(dict(seed=seed,policy_arm=arm,condition=condition,initial_state_index=i,
                        success=bool(success),steps=20 if success else 300,path=f'/invented/{seed}-{arm}-{condition}-{i}.json',sha256='c'*64))
                pair+=1
    return dict(episodes=rows)


def test_selection_is_all_thirteen_and_keeps_denominator():
    data=full_metadata();cases=p.select_cases(data)
    assert len(cases)==13 and sum(c['outcome']=='gained' for c in cases)==7 and len(data['episodes'])==240
    assert cases==p.select_cases(data)
    data['episodes'].pop()
    with pytest.raises(ValueError):p.select_cases(data)


def test_complete13_projected_envelope_independent_formulas_and_tamper(tmp_path):
    cases=p.select_cases(full_metadata());by_path={}
    for c in cases:
        for condition,item in c['episodes'].items():
            s=source(condition,item['success']);s.update(seed=c['seed'],policy_arm=c['policy_arm'],initial_state_index=c['initial_state_index'])
            path=tmp_path/(c['case_id']+'-'+condition+'.json');path.write_text(json.dumps(s));item.update(p.pin(path));by_path[str(path)]=s
    rows=[main.analyze_pair(c,by_path[c['episodes']['unchanged']['path']],by_path[c['episodes']['confirm5']['path']]) for c in cases]
    steps=sum(r['steps'] for c in cases for r in c['episodes'].values());size=sum(Path(r['path']).stat().st_size for c in cases for r in c['episodes'].values())
    caps=dict(source_episode_parses=26,source_episode_bytes=16*1024**2,public_scalar_projections=25*steps+26*9,
        policy_forwards=0,native_steps=0,model_reads=0,initial_bank_reads=0)
    result=dict(status='flip_pair_analysis_complete_pending_audit',cases=rows,summary=main.summary(rows),
        actual={**caps,'source_episode_bytes':size})
    cfg=dict(cases=cases,caps=caps)
    receipt=audit.audit(cfg,result,p.read_episode);assert receipt['summary']['failure_stage_unknown']==13
    result['cases'][0]['first_applied_positive_delta']=0
    with pytest.raises(ValueError):audit.audit(cfg,result,p.read_episode)
