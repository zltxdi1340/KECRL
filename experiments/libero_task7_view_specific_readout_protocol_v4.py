"""Scientific metadata and invented inputs only; all research stages denied."""
from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path

from experiments import libero_task7_spatial_readout_protocol_v3 as parent
from experiments import libero_spatial_readout_execution_v3 as prior_execution
from experiments import libero_task7_seed07_geometry_protocol as geometry

ROOT, require, pin, write_new = parent.ROOT, parent.require, parent.pin, parent.write_new
ARMS = ['flat_zero_xyz', 'view_specific_spatial_zero_xyz']
SEEDS, HDF5_FIELDS = parent.SEEDS, parent.HDF5_FIELDS
CONFIG = ROOT/'configs/libero-task7-view-specific-readout_v4.json'
FREEZE = ROOT/'configs/libero-task7-view-specific-readout-freeze-binding_v4.json'
CODE = ['src/algorithms/libero_view_specific_readout_policy_v4.py',
        'experiments/libero_task7_view_specific_readout_protocol_v4.py']
VIEW_SEED_OFFSETS = dict(agentview=9400001, eye_in_hand=9500001)
ENTROPY = {**parent.ENTROPY, 'max_retained_entropy_scalars': 4*80*256,
           'view_readout_parameters_shared': False, 'view_specialization_proven_by_entropy': False}
PARENT_KEYS = ['v3_science', 'v3_scientific_freeze', 'v3_handoff_binding', 'v3_completion',
               'v3_result_report', 'geometry_science', 'geometry_binding', 'geometry_completion',
               'geometry_audit', 'geometry_result_report']
sampling_plan, audit_entropy_summary = parent.sampling_plan, parent.audit_entropy_summary


def per_view_caps(caps):
    result = dict(caps)
    for view in VIEW_SEED_OFFSETS:
        for total in ['spatial_readout', 'spatial_image_rows', 'spatial_heatmaps',
                      'spatial_probability_values', 'spatial_coordinate_values', 'spatial_entropy_values']:
            require(caps[total] % 2 == 0, 'paired view budget must be even')
            result[total+'_'+view] = caps[total]//2
    return result


def train_caps():
    return per_view_caps(parent.train_caps())


def evaluation_caps(cfg, stage):
    return per_view_caps(parent.evaluation_caps(cfg, stage))


def problems(cfg, stage):
    if stage == 'handoff':
        return [dict(problem_id=s['source_id']+'-'+arm, seed=s['seed'], controller=arm,
                     initial_state_index=s['initial_state_index'], anchor_step=s['anchor_step'], source=s)
                for s in cfg['handoff_sources'] for arm in ARMS]
    require(stage == 'full', 'unknown v4 evaluation stage')
    return [dict(problem_id=f'{s}-{arm}-{i:02d}', seed=s, controller=arm,
                 initial_state_index=i, anchor_step=0, source=None)
            for s in SEEDS for arm in ARMS for i in range(20)]


def make_science(base, parents):
    """Derive fixed v4 metadata without reading or hashing any source payload."""
    parent.validate_scope(base)
    c = deepcopy(base)
    c.pop('readout_seed_offset')
    c.update(kind='libero_task7_view_specific_spatial_readout_v4', version=4, controllers=ARMS,
             windows={arm: 1 for arm in ARMS},
             expected_trainable_parameters={ARMS[0]: 1514120, ARMS[1]: 1507272},
             expected_readout_parameters={ARMS[0]: 1048704, ARMS[1]: 1041856},
             candidate=dict(channels=256, conv_bias=False, grid=[4, 4],
                 coordinate_order='channel_interleaved_x_y', coordinate_system='feature_grid_minus1_plus1_x_columns_y_rows',
                 post_coordinate_mlp=[512, 608, 128], hidden_activation='ReLU', final_activation='identity',
                 views=['agentview', 'eye_in_hand'], identical_view_structures=True,
                 shared_frozen_backbone=True, shared_readout_parameters=False,
                 per_view_readout_parameters=520928),
             readout_seed_offsets=VIEW_SEED_OFFSETS, entropy=ENTROPY,
             attribution=dict(scope='matched_budget_view_specific_readout_component',
                 isolated_weight_unsharing=False, isolated_spatial_softmax=False,
                 direct_comparator='new_paired_flat_zero_xyz', shared_v3_historical_context_only=True,
                 geometric_root_cause_proven=False, qualified_KECRL_Module=False),
             continuation=dict(handoff_pairs=22, handoff_minimum_net_gain=4, each_seed_nonregressing=True,
                 comparator='current_run_paired_control', fixed_absolute_candidate_minimum=False,
                 illustration_only=dict(control=8, candidate_required=12), independent_audit_required=True,
                 full_episodes_after_pass_only=120, full_pairs=60, full_minimum_net_gain=6),
             parents=parents, train_caps=train_caps(),
             output=str(ROOT/'results/libero-task7-view-specific-readout-20261118-v4'),
             checkpoint_directory=str(ROOT/'checkpoints/libero-task7-view-specific-readout-20261118-v4'))
    c['handoff_caps'] = evaluation_caps(c, 'handoff')
    c['full_caps'] = evaluation_caps(c, 'full')
    return c


def validate_scope(cfg):
    require(set(cfg['parents']) == set(PARENT_KEYS), 'v4 parent metadata coverage differs')
    base = json.loads(parent.CONFIG.read_text())
    expected = make_science(base, cfg['parents'])
    require(json.dumps(cfg, sort_keys=True) == json.dumps(expected, sort_keys=True), 'v4 frozen scientific scope differs')
    require(Path(cfg['output']).resolve().is_relative_to(ROOT/'results')
            and Path(cfg['checkpoint_directory']).resolve().is_relative_to(ROOT/'checkpoints'), 'persistent v4 storage required')


def validate_freeze(config=CONFIG, binding=FREEZE):
    cfg = json.loads(Path(config).read_text()); validate_scope(cfg)
    b = json.loads(Path(binding).read_text())
    require(b['status'] == 'scientific_frozen_execution_not_bound' and b['science'] == pin(config), 'scientific-only v4 binding required')
    for item in b['dependencies'] + b['validation'] + list(cfg['parents'].values()):
        require(Path(item['path']).suffix.lower() not in ['.pt', '.pth', '.npz', '.hdf5', '.h5', '.pruned_init', '.safetensors'],
                'source payload forbidden in scientific freeze')
        require(pin(item['path']) == item, 'v4 metadata/code dependency changed: ' + item['path'])
    old, ancestor = prior_execution.preflight(cfg['parents']['v3_science']['path'], cfg['parents']['v3_handoff_binding']['path'], 'handoff')
    require(cfg['parents']['v3_scientific_freeze'] == pin(parent.FREEZE), 'v3 scientific freeze provenance differs')
    for key in ['source', 'public_weights', 'policy', 'metadata', 'versions', 'libero_revision', 'data_root',
                'initial_bank', 'handoff_sources', 'runtime_environment', 'training_runtime_environment']:
        require(cfg[key] == old[key], 'v4 inherited source/runtime differs: ' + key)
    complete = json.loads(Path(cfg['parents']['v3_completion']['path']).read_text())
    require(complete['status'] == 'complete_audited_postgrasp_handoff', 'completed audited v3 context required')
    for key in ['result', 'audit']:
        require(pin(complete[key]['path']) == complete[key], 'v3 audited metadata changed')
    audit = json.loads(Path(complete['audit']['path']).read_text())
    require(audit['status'] == 'passed_independent_postgrasp_handoff_audit' and audit['stage'] == 'handoff'
            and audit['science'] == cfg['parents']['v3_science'] and audit['binding'] == cfg['parents']['v3_handoff_binding']
            and audit['result'] == complete['result'], 'v3 audited science/binding/result identity differs')
    _, geo_bound = geometry.preflight(cfg['parents']['geometry_science']['path'], cfg['parents']['geometry_binding']['path'])
    geo = json.loads(Path(cfg['parents']['geometry_completion']['path']).read_text())
    require(geo['status'] == 'complete_audited_targeted_seed07_geometry'
            and geo['science'] == cfg['parents']['geometry_science'] and geo['binding'] == cfg['parents']['geometry_binding']
            and geo['audit'] == cfg['parents']['geometry_audit'], 'completed geometry audit context required')
    closure = {r['path']: r['sha256'] for r in b['dependencies']}
    for prior in [ancestor, geo_bound]:
        require({r['path']: r['sha256'] for r in prior['code_dependencies']}.items() <= closure.items(), 'inherited v4 closure incomplete')
    for name in CODE:
        path = ROOT/name; require(closure.get(str(path)) == pin(path)['sha256'], 'v4 component code unbound')
    return cfg, b


def require_execution_binding(*args, **kwargs):
    raise PermissionError('v4 stage-specific execution binding required; scientific freeze authorizes no research work')


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
