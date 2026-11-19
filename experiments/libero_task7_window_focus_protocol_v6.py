"""Source-metadata scientific freeze for one fixed1:3 window distribution."""
from __future__ import annotations
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
from experiments import libero_task7_spatial_readout_protocol_v3 as parent
from experiments import libero_action_chunk_execution_v5 as latest
from experiments import libero_spatial_readout_execution_v3 as base_execution
from src.algorithms.libero_window_focus_sampling_v6 import ARMS as SAMPLING_ARMS,make_table

ROOT,require,pin,write_new=parent.ROOT,parent.require,parent.pin,parent.write_new
ARMS=list(SAMPLING_ARMS)
SEEDS,HDF5_FIELDS=parent.SEEDS,parent.HDF5_FIELDS
CONFIG=ROOT/'configs/libero-task7-window-focus_v6.json'
FREEZE=ROOT/'configs/libero-task7-window-focus-freeze-binding_v6.json'
CODE=['src/algorithms/libero_window_focus_sampling_v6.py','experiments/libero_task7_window_focus_protocol_v6.py']
ENTROPY={**parent.ENTROPY,'control_value':'measured_shared_spatial_entropy_both_arms'}
PARENT_KEYS=['latest_science','latest_handoff_binding','latest_completion','latest_evidence','latest_report','latest_readable',
    'base_science','base_freeze','base_handoff_binding','base_completion','base_training','base_training_audit','window_metadata']


def spatial_caps(b):
    b=dict(b);b.pop('flat_readout');b.pop('flat_image_rows')
    forwards=b.get('train_forward',0)+b.get('parity_forward',0)+b.get('policy_forward',0)+b.get('warm_forward',0)
    rows=80 if 'train_forward' in b else 1;heatmaps=2*forwards*rows*256
    b.update(spatial_readout=2*forwards,spatial_image_rows=2*forwards*rows,spatial_heatmaps=heatmaps,
        spatial_probability_values=16*heatmaps,spatial_coordinate_values=2*heatmaps,spatial_entropy_values=heatmaps,
        action_output_values=7*forwards*rows)
    for arm in ['control','candidate']:b['spatial_readout_'+arm]=forwards
    return b


def train_caps():
    b=spatial_caps(parent.train_caps());b.update(entropy_training_summaries=3000,sampling_table_builds=6,
        sampling_weight_coordinates=6*550,sampler_calls=3000,sampler_draws=24000,sampling_progress_summaries=3000)
    return b


def evaluation_caps(c,stage):
    b=spatial_caps(parent.evaluation_caps(c,stage));b['entropy_evaluation_summaries']=b['warm_forward']+b['policy_forward']
    return b


def problems(c,stage):
    if stage=='handoff':return [dict(problem_id=s['source_id']+'-'+a,seed=s['seed'],controller=a,
        initial_state_index=s['initial_state_index'],anchor_step=s['anchor_step'],source=s) for s in c['handoff_sources'] for a in ARMS]
    require(stage=='full','unknown v6 stage')
    return [dict(problem_id=f'{s}-{a}-{i:02d}',seed=s,controller=a,initial_state_index=i,anchor_step=0,source=None) for s in SEEDS for a in ARMS for i in range(20)]


def make_science(base,parents):
    parent.validate_scope(base);c=deepcopy(base);m=json.loads(Path(parents['window_metadata']['path']).read_text())
    c.update(kind='libero_task7_source_time_window_sampling_v6',version=6,controllers=ARMS,windows={a:1 for a in ARMS},
        expected_trainable_parameters={a:1499144 for a in ARMS},expected_total_parameters=12675656,
        expected_readout_parameters={a:1033728 for a in ARMS},candidate=deepcopy(base['candidate']),entropy=ENTROPY,
        whole_state_initialization_equal=True,shared_initialization='all_parameters_and_buffers_equal',
        minibatches_shared=False,network_factory='unchanged_v3_spatial_coordinates_zero_xyz',
        sampling=dict(variable='global_window_draw_distribution_only',progress='window_start/original_demo_action_length',
            late='2*start>=demo_action_length',control_weights='all1',candidate_early_weight=1,candidate_late_weight=3,
            normalization='global_over_all432_windows',replacement=True,batch_size=8,updates=500,
            method='integer_ticket_right_bisect',rng_seed_offset=70777,same_seed_independent_generators=True,
            classify_straddling_window='start_only',demo_length_known_source_metadata_only=True,
            no_phase_geometry_inference=True,extra_HDF5_reads=0,extra_forwards=0,
            summary=m['summary']),
        continuation=dict(net_gain=4,per_seed_nonregressing=True,current_run_control_only=True,
            illustration_only=dict(control=11,candidate_required=15),full_conditionally_authorized=True,
            full_requires_new_stage_binding_and_audited_handoff_pass=True,full_episode_cap=120,full_native_step_cap=36600),
        parents=parents,train_caps=train_caps(),output=str(ROOT/'results/libero-task7-window-focus-20261119-v6'),
        checkpoint_directory=str(ROOT/'checkpoints/libero-task7-window-focus-20261119-v6'))
    c['handoff_caps']=evaluation_caps(c,'handoff');c['full_caps']=evaluation_caps(c,'full')
    return c


def validate_scope(c):
    require(set(c['parents'])==set(PARENT_KEYS),'v6 parent coverage differs')
    require(json.dumps(c,sort_keys=True)==json.dumps(make_science(json.loads(parent.CONFIG.read_text()),c['parents']),sort_keys=True),'v6 frozen scope differs')


def validate_freeze(config=CONFIG,binding=FREEZE):
    c=json.loads(Path(config).read_text());validate_scope(c);b=json.loads(Path(binding).read_text())
    require(b['status']=='scientific_frozen_execution_not_bound' and b['science']==pin(config),'new v6 scientific binding required')
    for item in b['dependencies']+b['validation']+list(c['parents'].values()):
        require(Path(item['path']).suffix.lower() not in ['.pt','.pth','.npz','.hdf5','.h5','.pruned_init','.safetensors'],'payload forbidden in scientific closure')
        require(pin(item['path'])==item,'v6 metadata/code changed: '+item['path'])
    old,ancestor=latest.preflight(c['parents']['latest_science']['path'],c['parents']['latest_handoff_binding']['path'],'handoff')
    base,prior=base_execution.preflight(c['parents']['base_science']['path'],c['parents']['base_handoff_binding']['path'],'handoff')
    for key in ['source','public_weights','policy','metadata','versions','libero_revision','data_root','initial_bank','handoff_sources','runtime_environment','training_runtime_environment']:
        require(c[key]==base[key]==old[key],'v6 base provenance differs: '+key)
    for key in ['latest_completion','base_completion']:
        proof=json.loads(Path(c['parents'][key]['path']).read_text());require(proof['status']=='complete_audited_postgrasp_handoff','audited parent required')
        for field in ['result','audit']:require(pin(proof[field]['path'])==proof[field],'parent output metadata changed')
    evidence=json.loads(Path(c['parents']['latest_evidence']['path']).read_text())
    require(evidence['science']==c['parents']['latest_science'] and evidence['handoff_completion']==c['parents']['latest_completion'],
        'latest context evidence differs')
    t=json.loads(Path(c['parents']['base_training']['path']).read_text());a=json.loads(Path(c['parents']['base_training_audit']['path']).read_text())
    require(t['science']==c['parents']['base_science'] and a['science']==c['parents']['base_science']
        and a['status']=='passed_independent_postgrasp_training_audit' and a['result']==c['parents']['base_training'],'audited source lengths required')
    m=json.loads(Path(c['parents']['window_metadata']['path']).read_text())
    require(m['source_training']==c['parents']['base_training'] and m['source_audit']==c['parents']['base_training_audit']
        and m['demo_action_lengths']==t['demo_action_lengths'] and [[w['demo'],w['start']] for w in m['windows']]==t['window_ids']
        and [w['valid_length'] for w in m['windows']]==t['window_valid_lengths'],'window metadata provenance differs')
    table=make_table(t['window_ids'],t['window_valid_lengths'],t['demo_action_lengths'],arm=ARMS[1])
    require(table.total_weight==850 and sum(table.late)==209 and len(table.ids)==432 and sum(table.lengths)==4188,'frozen weight population differs')
    expected_summary=dict(windows=432,valid_rows=4188,early_windows=223,late_windows=209,total_weight=850,
        uniform_late_window_probability=209/432,candidate_late_window_probability=627/850,
        valid_frame_quartiles=[sum(q[i] for q in table.quartiles) for i in range(4)])
    require(m['summary']==expected_summary,'window summary algebra differs')
    require(all(w['weight']==table.weights[i] and w['late_window']==table.late[i] and w['valid_frame_quartiles']==list(table.quartiles[i])
        and w['demo_action_length']==m['demo_action_lengths'][str(w['demo'])] for i,w in enumerate(m['windows'])),'window weights/progress metadata differs')
    require(json.loads(Path(c['parents']['latest_report']['path']).read_text())==Path(c['parents']['latest_readable']['path']).read_text(),'historical readable derivative differs')
    closure={r['path']:r['sha256'] for r in b['dependencies']}
    for prior_bound in [ancestor,prior]:require({r['path']:r['sha256'] for r in prior_bound['code_dependencies']}.items()<=closure.items(),'inherited closure missing')
    for relative in CODE:
        path=ROOT/relative;require(closure.get(str(path))==pin(path)['sha256'],'v6 sampler/protocol unbound')
    return c,b


def require_execution_binding(*args,**kwargs):
    raise PermissionError('v6 science freeze denies research execution; new stage binding required')

def decision(rows,stage):
    require(stage in ['handoff','full'],'unknown decision stage')
    n=22 if stage=='handoff' else 60;threshold=4 if stage=='handoff' else 6
    require(len(rows)==2*n and len({r['problem_id'] for r in rows})==2*n
        and all(r['controller'] in ARMS and r['seed'] in SEEDS and type(r['success']) is bool for r in rows),
        'complete v6 paired boolean endpoints required')
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
