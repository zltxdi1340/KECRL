"""Admission for all thirteen outcome-discordant saved trajectory pairs."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.libero_gripper_confirmation_protocol_v2 import ROOT,ARMS,SEEDS,CONDITIONS,pin,require,write_new


def select_cases(result):
    rows=result['episodes'];index={(r['seed'],r['policy_arm'],r['condition'],r['initial_state_index']):r for r in rows}
    require(len(rows)==len(index)==240 and set(index)=={(s,a,c,i) for s in SEEDS for a in ARMS for c in CONDITIONS for i in range(20)},'full240 metadata required')
    cases=[]
    for seed in SEEDS:
        for arm in ARMS:
            for i in range(20):
                u,c=(index[seed,arm,k,i] for k in CONDITIONS)
                if u['success']==c['success']:continue
                cases.append(dict(case_id=f'flip{len(cases):02d}',seed=seed,policy_arm=arm,initial_state_index=i,
                    outcome='gained' if c['success'] else 'lost',episodes=dict(unchanged=u,confirm5=c)))
    require(len(cases)==13 and sum(c['outcome']=='gained' for c in cases)==7,'frozen thirteen flips differ')
    return cases


def preflight(config,binding):
    cfg=json.loads(Path(config).read_text());bound=json.loads(Path(binding).read_text())
    require(cfg['kind']=='libero_gripper_all13_flip_readonly_analysis' and cfg['version']==1
        and cfg['task']==7 and cfg['selection']=='all_outcome_discordant_pairs'
        and cfg['first_positive_threshold']==0. and cfg['raw_positive_short_run_max']==5
        and cfg['stage_labels_from_robot_state_allowed'] is False,'flip analysis scope differs')
    require(bound['status']=='bound_before_flip_trajectory_payload' and bound['science']==pin(config),'new flip binding required')
    for item in bound['code_dependencies']+bound['validation']:
        require(pin(item['path'])==item,'flip dependency changed: '+item['path'])
    for key in ['parent_completion','parent_result','parent_audit','parent_execution_binding']:
        require(pin(cfg[key]['path'])==cfg[key],'flip parent metadata changed')
    complete=json.loads(Path(cfg['parent_completion']['path']).read_text())
    require(complete['status']=='complete_audited_fresh_gripper_confirmation_control'
        and complete['result']==cfg['parent_result'] and complete['audit']==cfg['parent_audit'],'audited parent required')
    cases=select_cases(json.loads(Path(cfg['parent_result']['path']).read_text()))
    require(cases==cfg['cases'],'all13 case selection changed')
    steps=sum(r['steps'] for c in cases for r in c['episodes'].values())
    require(cfg['caps']==dict(source_episode_parses=26,source_episode_bytes=16*1024**2,
        public_scalar_projections=25*steps+26*9,policy_forwards=0,native_steps=0,model_reads=0,initial_bank_reads=0),'flip budget differs')
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results'),'persistent flip output required')
    grant_pin=bound['authorization'];require(pin(grant_pin['path'])==grant_pin,'flip authorization changed')
    grant=json.loads(Path(grant_pin['path']).read_text())
    paths=[r['path'] for c in cases for r in c['episodes'].values()]
    require(grant['user_approved'] is True and grant['science']==pin(config) and grant['source_paths']==paths
        and grant['caps']==cfg['caps'] and grant['sealed_roles_allowed']==[],'flip authority differs')
    parent=json.loads(Path(cfg['parent_execution_binding']['path']).read_text())
    closure={r['path']:r['sha256'] for r in bound['code_dependencies']}
    require({r['path']:r['sha256'] for r in parent['code_dependencies']}.items()<=closure.items(),'flip parent code missing')
    for name in ['libero_gripper_flip_pair_protocol.py','analyze_libero_gripper_flip_pairs.py','audit_libero_gripper_flip_pairs.py']:
        path=ROOT/'experiments'/name;require(closure.get(str(path))==pin(path)['sha256'],'flip implementation missing')
    return cfg,bound


def read_episode(item,case,condition):
    import hashlib
    raw=Path(item['path']).read_bytes();require(hashlib.sha256(raw).hexdigest()==item['sha256'],'flip trajectory SHA differs')
    source=json.loads(raw)
    require(source['status']=='complete' and source['task']==7 and source['initial_role']=='candidate_nonregression'
        and (source['seed'],source['policy_arm'],source['condition'],source['initial_state_index'])
          ==(case['seed'],case['policy_arm'],condition,case['initial_state_index'])
        and type(source['success']) is bool and source['success']==item['success'] and source['steps']==item['steps']
        and len(source['trace'])==source['steps']+1 and [r['step'] for r in source['trace']]==list(range(source['steps']+1)),
        'flip source identity or trace coverage differs')
    require(set(source['trace'][0])=={'step','joint_state','gripper_state'},'reset public fields differ')
    require(all(set(r)=={'step','raw_action','applied_action','joint_state','gripper_state','positive_run','masked'}
        for r in source['trace'][1:]),'step public fields differ')
    return source,len(raw)
