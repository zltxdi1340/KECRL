"""Stage admission and exact compute accounting for window-only sampling v6."""
from __future__ import annotations
import json
from pathlib import Path
from experiments import libero_task7_window_focus_protocol_v6 as science
from experiments.libero_formal_execution import Ledger

ROOT,require,pin,write_new=science.ROOT,science.require,science.pin,science.write_new
ARMS,SEEDS=science.ARMS,science.SEEDS
previous=science.base_execution.previous
SHAPES=previous.SHAPES
HDF5_FIELDS=science.HDF5_FIELDS
TRAIN_CAPS=science.train_caps()
problems,decision=science.problems,science.decision
CODE=['experiments/libero_window_focus_execution_v6.py','experiments/run_libero_window_focus_v6.py',
      'experiments/audit_libero_window_focus_v6.py','experiments/execute_libero_window_focus_v6.py']

def caps(cfg,stage):return science.evaluation_caps(cfg,stage)

def preflight(config,binding,stage):
    cfg,frozen=science.validate_freeze(config,science.FREEZE)
    b=json.loads(Path(binding).read_text())
    require(stage in ['train','handoff','full'] and b.get('stage')==stage
        and b.get('status')=='bound_before_window_focus_v6_payload' and b.get('science')==pin(config)
        and b.get('scientific_freeze')==pin(science.FREEZE),'new v6 stage execution binding required')
    for item in b['code_dependencies']+b['validation']:
        require(pin(item['path'])==item,'v6 execution dependency changed: '+item['path'])
    closure={r['path']:r['sha256'] for r in b['code_dependencies']}
    require({r['path']:r['sha256'] for r in frozen['dependencies']}.items()<=closure.items(),'v6 frozen closure missing')
    for relative in CODE:
        path=ROOT/relative;require(closure.get(str(path))==pin(path)['sha256'],'v6 execution code unbound')
    item=b['authorization'];require(pin(item['path'])==item,'v6 authorization changed')
    g=json.loads(Path(item['path']).read_text());expected=TRAIN_CAPS if stage=='train' else caps(cfg,stage)
    require(g['user_approved'] is True and g['science']==pin(config) and g['stage']==stage and g['tasks']==[7]
        and g['caps']==expected and g['sealed_roles_allowed']==[] and g['previous_episode_permission_reused'] is False
        and g['hdf5_fields']==HDF5_FIELDS and g['observation_fields']==list(SHAPES)
        and g['hdf5_allowed']==(stage=='train') and g['initial_indices']==([] if stage=='train' else list(range(19 if stage=='handoff' else 20))),
        'v6 stage authorization scope differs')
    if stage=='train':require(g['demonstrations']==list(range(30)) and g['source_paths']==[cfg['source']['path']],'v6 support scope differs')
    else:
        require(g['source_paths']==([s['source']['path'] for s in cfg['handoff_sources']] if stage=='handoff' else []),'v6 prefix scope differs')
        item=b['training_completion'];require(pin(item['path'])==item,'v6 training receipt changed')
        proof=json.loads(Path(item['path']).read_text());require(proof['status']=='complete_audited_postgrasp_training','audited v6 training required')
        for key in ['training','audit']:require(pin(proof[key]['path'])==proof[key],'v6 training proof changed')
        trained=json.loads(Path(proof['training']['path']).read_text())
        require(g['checkpoint_paths']==[r['checkpoint']['path'] for r in trained['models']],'v6 checkpoint authority differs')
    if stage=='full':
        item=b['handoff_completion'];require(pin(item['path'])==item,'v6 handoff receipt changed')
        proof=json.loads(Path(item['path']).read_text())
        require(proof['status']=='complete_audited_postgrasp_handoff' and proof['decision']['passed'] is True,'full denied without audited v6 handoff pass')
        for key in ['result','audit']:require(pin(proof[key]['path'])==proof[key],'v6 handoff proof changed')
        checked_handoff(cfg,config)
    return cfg,b


def checked_handoff(cfg,config):
    """Recheck this run's audited gate using JSON metadata only, before full."""
    base=Path(cfg['output'])/'handoff';proof=json.loads((base/'completion.json').read_text())
    require(proof['status']=='complete_audited_postgrasp_handoff' and proof['decision']['passed'] is True,
        'handoff gate has not admitted full evaluation')
    require(proof['result']==pin(base/'result.json') and proof['audit']==pin(base/'independent-audit.json'),
        'current handoff receipt paths/hash differ')
    result=json.loads((base/'result.json').read_text());audit=json.loads((base/'independent-audit.json').read_text())
    hand_binding=ROOT/'configs/libero-task7-window-focus-handoff-binding_v6.json'
    require(audit['status']=='passed_independent_postgrasp_handoff_audit' and audit['science']==pin(config)
        and audit['binding']==pin(hand_binding) and audit['result']==proof['result'],
        'current handoff audit provenance differs')
    jobs=problems(cfg,'handoff');rows=result['episodes']
    require(len(rows)==len(jobs) and all(all(row[k]==v for k,v in job.items()) for row,job in zip(rows,jobs)),
        'current handoff paired coverage differs')
    actual=decision(rows,'handoff')
    require(actual==result['decision']==audit['decision']==proof['decision'] and actual['passed'] is True,
        'current audited handoff decision differs')
    return proof


def make_policy(config,*,arm,seed,public_state=None):
    """Unchanged v3 shared3A factory; sampling_arm is execution metadata only."""
    require(arm in ARMS,'unknown sampling arm')
    from src.algorithms.libero_spatial_readout_policy_v3 import make_visual_readout_policy_v3
    model=make_visual_readout_policy_v3(config,arm='spatial_coordinates_zero_xyz',seed=seed,public_state=public_state)
    model.sampling_arm=arm
    return model


class ComputeLedger(Ledger):
    """Both arms use the same admitted network, with two spatial readouts."""
    optimizer_ref=None
    _frame=None
    last_prediction_entropy=None
    def call(self,key,fn,*args,**kwargs):
        if key=='optimizer':self.optimizer_ref=fn.__self__
        if key in ['train_forward','parity_forward','policy_forward','warm_forward']:
            require(self._frame is None,'nested policy forward forbidden')
            model=getattr(fn,'__self__',None) if key in ['train_forward','parity_forward'] else args[0]
            arm=model.get('controller') if isinstance(model,dict) else model.sampling_arm
            require(arm in ARMS,'unknown sampling arm')
            rows=int(args[0].shape[0]*args[0].shape[1]) if key in ['train_forward','parity_forward'] else 1
            require(1<=rows<=80,'forward row cap differs')
            self._frame=dict(arm=arm,rows=rows,readouts=0,encoders=0,expected_encoders=0 if key=='train_forward' else 2)
            extra=dict(zero_xyz_coordinate_writes=3*rows,action_output_values=7*rows)
            for name,amount in extra.items():self.begin(name,amount)
            try:
                result=super().call(key,fn,*args,**kwargs)
                require(self._frame['readouts']==2 and self._frame['encoders']==(0 if key=='train_forward' else 2),'per-forward compute count differs')
                shape=(*args[0].shape[:2],7) if key in ['train_forward','parity_forward'] else (7,)
                require(tuple(result[0].shape)==shape,'single-action output shape differs')
                for name,amount in extra.items():self.finish(name,amount)
                return result
            finally:self._frame=None
        if key=='frozen_encoder' and self._frame is not None:
            require(self._frame['encoders']<self._frame['expected_encoders'],'extra encoder forward')
            self._frame['encoders']+=1
        if key in ['flat_readout','spatial_readout']:
            frame=self._frame;require(frame is not None,'readout outside policy forward')
            require(key=='spatial_readout' and frame['readouts']<2,'wrong or extra visual readout')
            x=args[0];rows=int(x.shape[0]);require(tuple(x.shape[1:])==(512,4,4) and rows==frame['rows'],'readout rows differ')
            frame['readouts']+=1;heatmaps=rows*256
            extra=dict(spatial_image_rows=rows,spatial_heatmaps=heatmaps,spatial_probability_values=16*heatmaps,
                spatial_coordinate_values=2*heatmaps,spatial_entropy_values=heatmaps)
            extra['spatial_readout_'+('control' if frame['arm']==ARMS[0] else 'candidate')]=1
            for name,amount in extra.items():self.begin(name,amount)
            result=super().call(key,fn,*args,**kwargs)
            for name,amount in extra.items():self.finish(name,amount)
            return result
        return super().call(key,fn,*args,**kwargs)
