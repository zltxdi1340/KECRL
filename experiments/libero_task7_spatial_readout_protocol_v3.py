"""Scientific freeze and invented-input helpers, with research execution denied.

This module never opens demonstrations, public weights, checkpoints, initial
states, images, or simulator state. New stage-specific execution bindings and
an independent audit implementation are still required before research work.
"""
from __future__ import annotations

import json
from fractions import Fraction
import math
from pathlib import Path

from experiments import libero_eef_xyz_control_protocol as parent

ROOT,require,pin,write_new = parent.ROOT,parent.require,parent.pin,parent.write_new
ARMS = ['flat_zero_xyz','spatial_coordinates_zero_xyz']
SEEDS = [20261106,20261107,20261108]
CONFIG = ROOT/'configs/libero-task7-spatial-readout_v3.json'
FREEZE = ROOT/'configs/libero-task7-spatial-readout-freeze-binding_v3.json'
CODE = ['src/algorithms/libero_spatial_readout_policy_v3.py','experiments/libero_task7_spatial_readout_protocol_v3.py']
HDF5_FIELDS = ['obs/agentview_rgb','obs/eye_in_hand_rgb','obs/joint_states','obs/gripper_states','actions']
ENTROPY = dict(temperature=1.0,temperature_trainable=False,log_base='e',maximum_nats='ln(16)',
    normalization='H/ln(16)',channels=256,views=2,exclude_padding=True,loss_weight=0.,
    training_source='cached_train_forward_before_optimizer_and_parity',control_value=None,
    diagnostic_only=True,success_gate=False,extra_forwards=0,range_tolerance_nats=1e-6,
    valid_row_weighting='equal_valid_frame_view_channel',full_heatmaps_saved=False,
    max_retained_entropy_scalars=3*80*256)
SHARED_PREFIXES = ['visual_encoder.backbone.','visual_encoder.rgb_','fusion.','recurrent.','action_head.']


def train_caps():
    per_arm_forwards=3*(500+1)  # One direct/cache parity check per model.
    readouts=2*per_arm_forwards
    image_rows=readouts*8*10
    heatmaps=image_rows*256
    return dict(train_forward=3000,backward=3000,optimizer=3000,checkpoint=6,model_create=7,
        dataset_items=550,source_sha=1,public_weight_sha=1,public_weight_load=1,
        frozen_encoder=150,parity_forward=6,hdf5_opens=2,hdf5_slices=2750,
        flat_readout=readouts,spatial_readout=readouts,flat_image_rows=image_rows,spatial_image_rows=image_rows,
        spatial_heatmaps=heatmaps,spatial_probability_values=16*heatmaps,
        spatial_coordinate_values=2*heatmaps,spatial_entropy_values=heatmaps,
        entropy_training_summaries=1500,zero_xyz_coordinate_writes=3*2*per_arm_forwards*80,
        initial_row_loads=0,numeric_rows=0,native_steps=0,policy_forward=0,
        eef_payload_reads=0,private_object_state_reads=0,extra_render_calls=0)


def problems(cfg,stage):
    if stage=='handoff':
        return [dict(problem_id=s['source_id']+'-'+a,seed=s['seed'],controller=a,
            initial_state_index=s['initial_state_index'],anchor_step=s['anchor_step'],source=s)
            for s in cfg['handoff_sources'] for a in ARMS]
    require(stage=='full','unknown v3 evaluation stage')
    return [dict(problem_id=f'{s}-{a}-{i:02d}',seed=s,controller=a,initial_state_index=i,anchor_step=0,source=None)
            for s in SEEDS for a in ARMS for i in range(20)]


def evaluation_caps(cfg,stage):
    jobs=problems(cfg,stage);handoff=stage=='handoff'
    resets=sum(j['initial_state_index']+1 for j in jobs) if handoff else len(jobs)
    prefix=sum(j['anchor_step'] for j in jobs);suffix=sum(300-j['anchor_step'] for j in jobs)
    warm=9*len(jobs) if handoff else 0;forwards=warm+suffix
    spatial_forwards=sum((9 if handoff else 0)+300-j['anchor_step'] for j in jobs if j['controller']==ARMS[1])
    flat_forwards=forwards-spatial_forwards;heatmaps=2*spatial_forwards*256
    obs=resets+prefix+suffix
    return dict(model_create=6,checkpoint_sha=6,checkpoint_load=6,model_weight_checks=12,
        adapters=len(jobs) if handoff else 6,resets=resets,episodes=len(jobs),prefix_actions=prefix,
        warm_forward=warm,policy_forward=suffix,policy_actions=suffix,frozen_encoder=2*forwards,temporal_rows=forwards,
        flat_readout=2*flat_forwards,spatial_readout=2*spatial_forwards,
        flat_image_rows=2*flat_forwards,spatial_image_rows=2*spatial_forwards,
        spatial_heatmaps=heatmaps,spatial_probability_values=16*heatmaps,spatial_coordinate_values=2*heatmaps,
        spatial_entropy_values=heatmaps,entropy_evaluation_summaries=spatial_forwards,
        zero_xyz_coordinate_writes=3*forwards,zero_steps=5*resets,native_steps=prefix+suffix+5*resets,
        initial_row_loads=1,numeric_rows=19 if handoff else 20,observations=obs,rgb_frames=2*obs,
        raw_rgb_bytes=2*obs*128*128*3,source_episode_parses=22 if handoff else 0,
        source_episode_bytes=16*1024**2 if handoff else 0,
        source_scalar_projections=sum(25*s['anchor_step']+9 for s in cfg['handoff_sources']) if handoff else 0,
        backwards=0,optimizer_updates=0,hdf5_reads=0,public_weight_reads=0,eef_payload_reads=0,
        private_object_state_reads=0,extra_render_calls=0)


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


def sampling_plan(seed,windows):
    """Pure invented/metadata schedule; no data loading and no optimizer work."""
    import torch
    require(seed in SEEDS and 1<=len(windows)<=550,'fixed seed and bounded window metadata required')
    require(len({tuple(v) for v in windows})==len(windows)
        and all(len(v)==2 and type(v[0]) is int and type(v[1]) is int and 0<=v[0]<30 and v[1]>=1 for v in windows),
        'source-support causal window metadata required')
    generator=torch.Generator().manual_seed(seed+7*10000+777)
    return [[windows[i] for i in torch.randint(len(windows),(8,),generator=generator).tolist()] for _ in range(500)]


def audit_entropy_summary(record,valid_rows):
    """Independent scalar algebra/bounds, without replaying a model or heatmaps."""
    require(type(valid_rows) is int and 1<=valid_rows<=80,'bounded valid entropy row count required')
    count=valid_rows*256
    require(type(record['valid_rows']) is int and type(record['heatmaps_per_view']) is int
        and record['valid_rows']==valid_rows and record['heatmaps_per_view']==count
        and type(record['temperature']) in (int,float) and record['temperature']==1.0,
        'entropy identity/temperature differs')
    a,h=record['agent_entropy_nats_sum'],record['hand_entropy_nats_sum']
    require(all(type(v) in (int,float) and math.isfinite(v) and -1e-6*count<=v<=(math.log(16)+1e-6)*count for v in [a,h]),
        'entropy sum outside finite numerical range')
    mean=(a+h)/(2*count)
    expected=dict(agent_entropy_nats_mean=a/count,hand_entropy_nats_mean=h/count,
        entropy_nats_mean=mean,normalized_entropy_mean=mean/math.log(16))
    for key,value in expected.items():
        require(type(record[key]) in (int,float) and math.isfinite(record[key])
            and abs(record[key]-value)<=1e-6,'independent entropy scalar formula differs: '+key)
    return expected


def validate_scope(cfg):
    expected=dict(kind='libero_task7_frozen_spatial_readout_module_v3',version=3,formal_result=False,
        task=7,seeds=SEEDS,controllers=ARMS,windows={a:1 for a in ARMS},fusion_input_dimension=268,
        xyz_mode='internally_generated_three_zeros_both_arms',demonstration_fields=HDF5_FIELDS,
        public_observation_fields=['agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states'],
        expected_trainable_parameters={ARMS[0]:1514120,ARMS[1]:1499144},
        expected_readout_parameters={ARMS[0]:1048704,ARMS[1]:1033728},
        candidate=dict(channels=256,conv_bias=False,grid=[4,4],coordinate_order='channel_interleaved_x_y',
            coordinate_system='feature_grid_minus1_plus1_x_columns_y_rows',post_coordinate_mlp=[512,1408,128],
            hidden_activation='ReLU',final_activation='identity'),
        entropy=ENTROPY,shared_initialization='all_state_except_visual_encoder_projection',
        whole_state_initialization_equal=False,readout_seed_offset=9200001,
        updates_per_policy=500,batch_size=8,sequence_length=10,stride=10,stabilization_steps=5,
        demonstrations=list(range(30)),source_role='meta_support',source_meta_query_allowed=False,
        formal_evaluation_allowed=False,initial_role='candidate_nonregression',initial_indices=list(range(20)),
        expected_training_windows=432,expected_valid_training_rows=4188,loss='uniform_masked_seven_dimensional_mse',
        optimizer=dict(name='Adam',lr=.0003,betas=[.9,.999],eps=1e-8,weight_decay=0.,amsgrad=False),
        gradient_clip_norm=1.,training_device='cpu',horizon=300,image_size=128,environment_seed=20261022,
        no_automatic_retry=True,confirmation_enabled=False,checkpoint='last500_only',
        training_wall_seconds_max=1800,handoff_wall_seconds_max=1800,full_wall_seconds_max=3600,
        train_caps=train_caps(),scientific_freeze_authorizes_execution=False,stage_binding_required=True)
    for key,value in expected.items():
        require(json.dumps(cfg.get(key),sort_keys=True)==json.dumps(value,sort_keys=True),'v3 frozen scope differs: '+key)
    require(len(cfg['handoff_sources'])==22 and cfg['handoff_caps']==evaluation_caps(cfg,'handoff')
        and cfg['full_caps']==evaluation_caps(cfg,'full'),'v3 fixed22/budgets differ')
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results'),'persistent v3 storage required')


def validate_freeze(config=CONFIG,binding=FREEZE):
    cfg=json.loads(Path(config).read_text());validate_scope(cfg)
    bound=json.loads(Path(binding).read_text())
    require(bound['status']=='scientific_frozen_execution_not_bound' and bound['science']==pin(config),
        'scientific-only v3 freeze binding required')
    for item in bound['dependencies']+bound['validation']+list(cfg['parents'].values()):
        require(Path(item['path']).suffix not in ['.pt','.pth','.npz','.hdf5','.pruned_init'],
            'research payload must not enter scientific metadata freeze')
        require(pin(item['path'])==item,'v3 frozen dependency changed: '+item['path'])
    old,ancestor=parent.preflight(cfg['parents']['v2_science']['path'],cfg['parents']['v2_handoff_binding']['path'],'handoff')
    for key in ['source','public_weights','policy','metadata','versions','libero_revision','data_root','initial_bank',
                'handoff_sources','runtime_environment','training_runtime_environment']:
        require(cfg[key]==old[key],'v3 inherited provenance differs: '+key)
    closure={r['path']:r['sha256'] for r in bound['dependencies']}
    require({r['path']:r['sha256'] for r in ancestor['code_dependencies']}.items()<=closure.items(),
        'v3 inherited code closure incomplete')
    for name in CODE:
        path=ROOT/name;require(closure.get(str(path))==pin(path)['sha256'],'v3 module/protocol unbound')
    completion=json.loads(Path(cfg['parents']['v2_completion']['path']).read_text())
    require(completion['status']=='complete_audited_postgrasp_handoff' and completion['decision']['passed'] is False,
        'v2 audited non-admission provenance differs')
    for key in ['result','audit']:require(pin(completion[key]['path'])==completion[key],'v2 result proof changed')
    return cfg,bound


def require_execution_binding(*args,**kwargs):
    """Scientific freeze is deliberately insufficient to open research payloads."""
    raise PermissionError('v3 scientific freeze only; new runner/guards/independent audit and stage execution binding required')
