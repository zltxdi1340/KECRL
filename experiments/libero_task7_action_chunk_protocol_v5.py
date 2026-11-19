"""Frozen H4/k1 supervision, causal masks and exact finite execution caps."""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

from experiments import libero_task7_spatial_readout_protocol_v3 as parent
from experiments import libero_view_specific_readout_execution_v4 as latest
from experiments import libero_spatial_readout_execution_v3 as base_execution

ROOT,require,pin,write_new=parent.ROOT,parent.require,parent.pin,parent.write_new
ARMS=['h1_spatial_control','joint_h4_k1_spatial']
SEEDS,HDF5_FIELDS=parent.SEEDS,parent.HDF5_FIELDS
CONFIG=ROOT/'configs/libero-task7-action-chunk_v5.json'
FREEZE=ROOT/'configs/libero-task7-action-chunk-freeze-binding_v5.json'
CODE=['src/algorithms/libero_action_chunk_policy_v5.py','experiments/libero_task7_action_chunk_protocol_v5.py']
ENTROPY={**parent.ENTROPY,'control_value':'measured_shared_spatial_entropy_both_arms'}
sampling_plan,audit_entropy_summary=parent.sampling_plan,parent.audit_entropy_summary
PARENT_KEYS=['latest_science','latest_handoff_binding','latest_completion','base_science','base_freeze',
             'base_handoff_binding','base_completion','base_training','base_training_audit','crosscheck_evidence']


def compute_caps(b,control,candidate,rows):
    b=dict(b)
    for key in ['flat_readout','flat_image_rows']:b.pop(key,None)
    all_forwards=control+candidate;heatmaps=2*all_forwards*rows*256
    b.update(spatial_readout=2*all_forwards,spatial_image_rows=2*all_forwards*rows,
        spatial_heatmaps=heatmaps,spatial_probability_values=16*heatmaps,
        spatial_coordinate_values=2*heatmaps,spatial_entropy_values=heatmaps,
        returned_chunk_values=all_forwards*rows*28,inactive_action_zero_writes=control*rows*21,
        action_output_values=(control+4*candidate)*rows*7)
    for arm,n in [('control',control),('candidate',candidate)]:
        b['spatial_readout_'+arm]=2*n
        for h in range(4):b[f'action_head_{h}_{arm}']=n if arm=='candidate' or h==0 else 0
    for h in range(4):b[f'action_head_{h}']=control+candidate if h==0 else candidate
    return b


def train_caps():
    n=3*(500+1);b=compute_caps(parent.train_caps(),n,n,80)
    b.update(entropy_training_summaries=3000,target_builds=3000,target_pair_slots=3000*80*4,
             target_action_values=3000*80*4*7,loss_evaluations=3000,
             supervised_pairs=1500*(80+8*34),supervised_action_values=1500*(80+8*34)*7)
    return b


def evaluation_caps(cfg,stage):
    b=parent.evaluation_caps(cfg,stage)
    n=(b['warm_forward']+b['policy_forward'])//2
    b=compute_caps(b,n,n,1);b['entropy_evaluation_summaries']=2*n
    return b


def problems(cfg,stage):
    if stage=='handoff':
        return [dict(problem_id=s['source_id']+'-'+arm,seed=s['seed'],controller=arm,
                     initial_state_index=s['initial_state_index'],anchor_step=s['anchor_step'],source=s)
                for s in cfg['handoff_sources'] for arm in ARMS]
    require(stage=='full','unknown v5 stage')
    return [dict(problem_id=f'{s}-{a}-{i:02d}',seed=s,controller=a,initial_state_index=i,anchor_step=0,source=None)
            for s in SEEDS for a in ARMS for i in range(20)]


def make_science(base,parents):
    parent.validate_scope(base);c=deepcopy(base)
    c.update(kind='libero_task7_joint_h4_k1_action_prediction_v5',version=5,controllers=ARMS,
        windows={a:1 for a in ARMS},expected_trainable_parameters={ARMS[0]:1499144,ARMS[1]:1504541},
        registered_nonbackbone_parameters=1504541,expected_total_parameters=12681053,
        expected_readout_parameters={a:1033728 for a in ARMS},
        candidate=dict(base_readout='shared_3A_512_1408_128',prediction_horizon=4,execution_stride=1,
            future_head_seed_offsets=[9700001,9701001,9702001],action_head_shapes=[256,7],
            registered_heads=4,control_active_heads=[0],candidate_active_heads=[0,1,2,3],
            control_future_heads_frozen=True,control_future_heads_in_optimizer=False,only_first_action_executed=True),
        dual_mask=dict(anchor='valid_mask[t]',future='t+h<T AND start+t+h<demo_length AND valid_mask[t+h]',
            control_head_mask=[True,False,False,False],candidate_head_mask=[True,True,True,True],
            target='cached_actions[t+h]',cross_window_targets=False,extra_HDF5_reads=False,
            formula='sum(M_anchor*M_future*A_arm*(prediction-target)^2)/(7*sum(M_anchor*M_future*A_arm))',
            control_reduction='legacy_H0_contiguous_masked_seven_dimensional_mse',
            supervision_distribution_changes=True,full_window_pair_count=34),
        entropy=ENTROPY,shared_initialization='all_registered_state_equal_active_masks_differ',whole_state_initialization_equal=True,
        loss='double_masked_uniform_seven_dimensional_mse',expected_candidate_dataset_pairs=14168,
        continuation=dict(net_gain=4,per_seed_nonregressing=True,current_run_control_only=True,
            absolute_candidate_minimum_fixed=False,illustration_only=dict(control=11,candidate_required=15),
            current_driver_stops_after44=True,full_episodes_this_run=0),
        parents=parents,train_caps=train_caps(),
        output=str(ROOT/'results/libero-task7-action-chunk-20261118-v5'),
        checkpoint_directory=str(ROOT/'checkpoints/libero-task7-action-chunk-20261118-v5'))
    c['handoff_caps']=evaluation_caps(c,'handoff');c['full_caps']=evaluation_caps(c,'full')
    return c


def validate_scope(c):
    require(set(c['parents'])==set(PARENT_KEYS),'v5 parent coverage differs')
    expected=make_science(json.loads(parent.CONFIG.read_text()),c['parents'])
    require(json.dumps(c,sort_keys=True)==json.dumps(expected,sort_keys=True),'v5 frozen scope differs')


def validate_freeze(config=CONFIG,binding=FREEZE):
    c=json.loads(Path(config).read_text());validate_scope(c);b=json.loads(Path(binding).read_text())
    require(b['status']=='scientific_frozen_execution_not_bound' and b['science']==pin(config),'new v5 scientific binding required')
    for item in b['dependencies']+b['validation']+list(c['parents'].values()):
        require(Path(item['path']).suffix.lower() not in ['.pt','.pth','.npz','.hdf5','.h5','.pruned_init','.safetensors'],'payload forbidden in scientific closure')
        require(pin(item['path'])==item,'v5 metadata/code changed: '+item['path'])
    old,ancestor=latest.preflight(c['parents']['latest_science']['path'],c['parents']['latest_handoff_binding']['path'],'handoff')
    parent.validate_freeze(c['parents']['base_science']['path'],c['parents']['base_freeze']['path'])
    base,base_bound=base_execution.preflight(c['parents']['base_science']['path'],c['parents']['base_handoff_binding']['path'],'handoff')
    for key in ['source','public_weights','policy','metadata','versions','libero_revision','data_root','initial_bank','handoff_sources','runtime_environment','training_runtime_environment']:
        require(c[key]==base[key]==old[key],'v5 base source/runtime differs: '+key)
    complete=json.loads(Path(c['parents']['base_completion']['path']).read_text());require(complete['status']=='complete_audited_postgrasp_handoff','base audited handoff required')
    for key in ['result','audit']:require(pin(complete[key]['path'])==complete[key],'base output metadata changed')
    t=json.loads(Path(c['parents']['base_training']['path']).read_text());a=json.loads(Path(c['parents']['base_training_audit']['path']).read_text())
    require(t['status']=='postgrasp_training_complete_pending_audit' and a['status']=='passed_independent_postgrasp_training_audit'
        and a['result']==c['parents']['base_training'] and a['science']==c['parents']['base_science'],'base training provenance differs')
    lengths=t['window_valid_lengths'];require(len(lengths)==432 and sum(lengths)==4188
        and sum(sum(min(4,L-i) for i in range(L)) for L in lengths)==14168,'frozen dual-mask dataset counts differ')
    recent=json.loads(Path(c['parents']['latest_completion']['path']).read_text());require(recent['status']=='complete_audited_postgrasp_handoff','audited latest context required')
    for key in ['result','audit']:require(pin(recent[key]['path'])==recent[key],'latest metadata changed')
    closure={r['path']:r['sha256'] for r in b['dependencies']}
    for prior in [ancestor,base_bound]:require({r['path']:r['sha256'] for r in prior['code_dependencies']}.items()<=closure.items(),'inherited closure missing')
    for relative in CODE:
        path=ROOT/relative;require(closure.get(str(path))==pin(path)['sha256'],'v5 module/protocol unbound')
    return c,b


def require_execution_binding(*args,**kwargs):
    raise PermissionError('new v5 stage execution binding required; scientific freeze denies research work')


def decision(rows,stage):
    require(stage in ['handoff','full'],'unknown decision stage')
    n=22 if stage=='handoff' else 60;threshold=4 if stage=='handoff' else 6
    require(len(rows)==2*n and len({r['problem_id'] for r in rows})==2*n
        and all(r['controller'] in ARMS and r['seed'] in SEEDS and type(r['success']) is bool for r in rows),
        'complete v3 paired boolean endpoints required')
    totals={a:sum(r['success'] for r in rows if r['controller']==a) for a in ARMS}
    require(all(sum(r['controller']==a for r in rows)==n for a in ARMS),'balanced paired arm coverage required')
    pairs={}
    for row in rows:
        if stage=='handoff':
            suffix='-'+row['controller'];require(row['problem_id'].endswith(suffix),'paired problem suffix differs')
            key=row['problem_id'][:-len(suffix)]
        else:
            prefix=f"{row['seed']}-{row['controller']}-"
            require(row['problem_id'].startswith(prefix),'full paired problem prefix differs')
            token=row['problem_id'][len(prefix):]
            require(len(token)==2 and token.isdecimal() and 0<=int(token)<20
                and token==f'{int(token):02d}','full paired initial index differs')
            key=(row['seed'],int(token))
        require(row['controller'] not in pairs.setdefault(key,{}),'duplicate paired controller')
        pairs[key][row['controller']]=row['seed']
    require(len(pairs)==n and all(set(v)==set(ARMS) and len(set(v.values()))==1 for v in pairs.values()),
        'complete same-seed endpoint pairs required')
    counts={SEEDS[0]:12,SEEDS[1]:8,SEEDS[2]:2} if stage=='handoff' else {s:20 for s in SEEDS}
    require(all(sum(r['controller']==a and r['seed']==s for r in rows)==counts[s] for a in ARMS for s in SEEDS),
        'fixed per-seed coverage differs')
    seeds={str(s):{a:sum(r['success'] for r in rows if r['seed']==s and r['controller']==a) for a in ARMS} for s in SEEDS}
    net=totals[ARMS[1]]-totals[ARMS[0]]
    gates=dict(net_gain_at_least_threshold=net>=threshold,
        each_seed_nonregressing=all(v[ARMS[1]]>=v[ARMS[0]] for v in seeds.values()))
    return dict(stage=stage,pairs=n,totals=totals,per_seed=seeds,required_net=threshold,
        gain_fraction=str(Fraction(net,n)),gates=gates,passed=all(gates.values()))
