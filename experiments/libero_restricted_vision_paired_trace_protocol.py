"""Existing-output-only scope and bounded reads for an exploratory diagnosis.

Outcomes and one example trace were known before this metric definition.
This is not a blinded confirmatory test or causal identification protocol.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = [4,7]
SEEDS = [20261106,20261107,20261108]
ARMS = ['frozen_resnet18','layer4_conv_finetune']
HORIZONS = [1,5,20,50,100]
OUTCOMES = ['both_success','frozen_only','finetune_only','both_failure']
DEADBAND = .1
SATURATION = .95
SIGN_EPS = 1e-12
CAPS = dict(json_parses_per_role=244,source_bytes_per_role=256*1024**2,
            episodes=240,pairs=120,prefix_rows=600,training_models=12,
            policy_forwards=0,backwards=0,optimizer_updates=0,environment_steps=0,
            new_episodes=0,checkpoint_reads=0,hdf5_reads=0,initial_bank_reads=0)


def require(condition,message):
    if not condition:raise ValueError(message)


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write_new(path,value):
    with Path(path).open('x') as handle:
        json.dump(value,handle,ensure_ascii=False,allow_nan=False,indent=2);handle.write('\n')


def validate_scope(cfg):
    expected=dict(kind='libero_restricted_visual_existing_paired_trace_analysis',version=1,
        formal_result=False,status='metrics_fixed_after_outcomes_known',tasks=TASKS,seeds=SEEDS,arms=ARMS,
        initial_indices=list(range(20)),horizons=HORIZONS,primary_prefix=50,
        gripper_command_deadband=DEADBAND,saturation_absolute_threshold=SATURATION,
        delta_sign_zero_absolute_tolerance=SIGN_EPS,
        caps=CAPS,causal_identification=False,outcomes_previously_known=True,
        no_padding_or_time_warp=True,per_episode_equal_weight=True)
    for name,value in expected.items():
        require(json.dumps(cfg.get(name),sort_keys=True)==json.dumps(value,sort_keys=True),'analysis scope differs: '+name)
    require(set(cfg['sources'])=={'training','training_audit','evaluation','completion'},'source JSON set differs')
    base=Path(cfg['source_archive']).resolve()
    require(base.is_relative_to(ROOT/'results') and Path(cfg['output']).resolve().is_relative_to(ROOT/'results')
            and base!=Path(cfg['output']).resolve(),'persistent separate output required')
    for name,item in cfg['sources'].items():
        require(Path(item['path']).resolve()==base/(name.replace('_','-')+'.json'),'input path differs: '+name)


def preflight(config,binding):
    cfg=json.loads(Path(config).read_text());bound=json.loads(Path(binding).read_text());validate_scope(cfg)
    require(bound['status']=='bound_before_complete_trace_projection' and bound['science']==pin(config)
            and bound['synthetic_validation_passed'] is True,'analysis execution binding required')
    for item in bound['metadata_dependencies']+bound['validation']:
        require(pin(item['path'])==item,'analysis dependency changed: '+item['path'])
    return cfg,bound


class Sources:
    """Only four declared trial JSONs and their 240 pinned public episodes.

    Logical read admission, not an OS sandbox. No model/dataset/bank parser is
    imported. Counts and bytes are checked before reads; a file is parsed once.
    """
    def __init__(self,cfg):
        self.cfg=cfg;self.paths=set();self.parses=0;self.bytes=0
        self.fixed={str(Path(row['path']).resolve()):row for row in cfg['sources'].values()}

    def read(self,item,*,episode=False):
        path=Path(item['path']).resolve()
        require(path.suffix=='.json' and str(path) not in self.paths,'only unique JSON input reads admitted')
        if episode:
            require(path.parent==Path(self.cfg['source_archive']).resolve()/'episodes','episode outside archive')
        else:
            require(self.fixed.get(str(path))==item,'undeclared source JSON')
        size=path.stat().st_size
        require(self.parses+1<=CAPS['json_parses_per_role'] and self.bytes+size<=CAPS['source_bytes_per_role'],
                'source parse or byte cap exceeded')
        self.paths.add(str(path));self.parses+=1;self.bytes+=size
        raw=path.read_bytes()
        require(len(raw)==size and hashlib.sha256(raw).hexdigest()==item['sha256'],'source SHA/size differs')
        def invalid(value):raise ValueError('nonfinite JSON literal: '+value)
        return json.loads(raw,parse_constant=invalid)

    def snapshot(self):
        return dict(json_parses=self.parses,source_bytes=self.bytes,unique_paths=len(self.paths),
            policy_forwards=0,backwards=0,optimizer_updates=0,environment_steps=0,
            new_episodes=0,checkpoint_reads=0,hdf5_reads=0,initial_bank_reads=0)
