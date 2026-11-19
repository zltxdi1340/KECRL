"""Stage admission and exact compute accounting for frozen H4/k1 action supervision v5."""
from __future__ import annotations
import json
from pathlib import Path
from experiments import libero_task7_action_chunk_protocol_v5 as science
from experiments.libero_formal_execution import Ledger

ROOT,require,pin,write_new=science.ROOT,science.require,science.pin,science.write_new
ARMS,SEEDS=science.ARMS,science.SEEDS
previous=science.base_execution.previous
SHAPES=previous.SHAPES
HDF5_FIELDS=science.HDF5_FIELDS
TRAIN_CAPS=science.train_caps()
problems,decision=science.problems,science.decision
CODE=['experiments/libero_action_chunk_execution_v5.py','experiments/run_libero_action_chunk_v5.py',
      'experiments/audit_libero_action_chunk_v5.py','experiments/execute_libero_action_chunk_v5.py']

def caps(cfg,stage):return science.evaluation_caps(cfg,stage)

def preflight(config,binding,stage):
    cfg,frozen=science.validate_freeze(config,science.FREEZE)
    b=json.loads(Path(binding).read_text())
    require(stage in ['train','handoff','full'] and b.get('stage')==stage
        and b.get('status')=='bound_before_action_chunk_v5_payload' and b.get('science')==pin(config)
        and b.get('scientific_freeze')==pin(science.FREEZE),'new v5 stage execution binding required')
    for item in b['code_dependencies']+b['validation']:
        require(pin(item['path'])==item,'v5 execution dependency changed: '+item['path'])
    closure={r['path']:r['sha256'] for r in b['code_dependencies']}
    require({r['path']:r['sha256'] for r in frozen['dependencies']}.items()<=closure.items(),'v5 frozen closure missing')
    for relative in CODE:
        path=ROOT/relative;require(closure.get(str(path))==pin(path)['sha256'],'v5 execution code unbound')
    item=b['authorization'];require(pin(item['path'])==item,'v5 authorization changed')
    g=json.loads(Path(item['path']).read_text());expected=TRAIN_CAPS if stage=='train' else caps(cfg,stage)
    require(g['user_approved'] is True and g['science']==pin(config) and g['stage']==stage and g['tasks']==[7]
        and g['caps']==expected and g['sealed_roles_allowed']==[] and g['previous_episode_permission_reused'] is False
        and g['hdf5_fields']==HDF5_FIELDS and g['observation_fields']==list(SHAPES)
        and g['hdf5_allowed']==(stage=='train') and g['initial_indices']==([] if stage=='train' else list(range(19 if stage=='handoff' else 20))),
        'v5 stage authorization scope differs')
    if stage=='train':require(g['demonstrations']==list(range(30)) and g['source_paths']==[cfg['source']['path']],'v5 support scope differs')
    else:
        require(g['source_paths']==([s['source']['path'] for s in cfg['handoff_sources']] if stage=='handoff' else []),'v5 prefix scope differs')
        item=b['training_completion'];require(pin(item['path'])==item,'v5 training receipt changed')
        proof=json.loads(Path(item['path']).read_text());require(proof['status']=='complete_audited_postgrasp_training','audited v5 training required')
        for key in ['training','audit']:require(pin(proof[key]['path'])==proof[key],'v5 training proof changed')
        trained=json.loads(Path(proof['training']['path']).read_text())
        require(g['checkpoint_paths']==[r['checkpoint']['path'] for r in trained['models']],'v5 checkpoint authority differs')
    if stage=='full':
        item=b['handoff_completion'];require(pin(item['path'])==item,'v5 handoff receipt changed')
        proof=json.loads(Path(item['path']).read_text())
        require(proof['status']=='complete_audited_postgrasp_handoff' and proof['decision']['passed'] is True,'full denied without audited v5 handoff pass')
        for key in ['result','audit']:require(pin(proof[key]['path'])==proof[key],'v5 handoff proof changed')
    return cfg,b


class ComputeLedger(Ledger):
    """Admission before computation; both arms spatial, only candidate H1..H3."""
    optimizer_ref=None
    _frame=None
    last_prediction_entropy=None
    def call(self,key,fn,*args,**kwargs):
        if key=='optimizer':self.optimizer_ref=fn.__self__
        if key in ['train_forward','parity_forward','policy_forward','warm_forward']:
            require(self._frame is None,'nested policy forward forbidden')
            model=getattr(fn,'__self__',None) if key in ['train_forward','parity_forward'] else args[0]
            arm=model.get('controller') if isinstance(model,dict) else model.readout_arm
            require(arm in ARMS,'unknown action supervision arm')
            rows=int(args[0].shape[0]*args[0].shape[1]) if key in ['train_forward','parity_forward'] else 1
            require(1<=rows<=80,'forward row cap differs')
            self._frame=dict(arm=arm,rows=rows,readouts=0,encoders=0,heads=[],expected_encoders=0 if key=='train_forward' else 2)
            extra=dict(zero_xyz_coordinate_writes=3*rows,returned_chunk_values=28*rows)
            if arm==ARMS[0]:extra['inactive_action_zero_writes']=21*rows
            for name,amount in extra.items():self.begin(name,amount)
            try:
                result=super().call(key,fn,*args,**kwargs)
                require(self._frame['readouts']==2 and self._frame['encoders']==(0 if key=='train_forward' else 2)
                    and self._frame['heads']==([0] if arm==ARMS[0] else [0,1,2,3]),'per-forward encoder/readout/action-head count differs')
                for name,amount in extra.items():self.finish(name,amount)
                return result
            finally:self._frame=None
        if key=='frozen_encoder' and self._frame is not None:
            require(self._frame['encoders']<self._frame['expected_encoders'],'extra encoder forward')
            self._frame['encoders']+=1
        if key in ['flat_readout','spatial_readout']:
            frame=self._frame;require(frame is not None,'readout outside admitted policy forward')
            require(key=='spatial_readout' and frame['readouts']<2,'wrong or extra visual readout')
            x=args[0];rows=int(x.shape[0]);require(tuple(x.shape[1:])==(512,4,4) and rows==frame['rows'],'readout tensor rows differ')
            frame['readouts']+=1;heatmaps=rows*256
            extra=dict(spatial_image_rows=rows,spatial_heatmaps=heatmaps,spatial_probability_values=16*heatmaps,
                spatial_coordinate_values=2*heatmaps,spatial_entropy_values=heatmaps)
            extra['spatial_readout_'+('control' if frame['arm']==ARMS[0] else 'candidate')]=1
            for name,amount in extra.items():self.begin(name,amount)
            result=super().call(key,fn,*args,**kwargs)
            for name,amount in extra.items():self.finish(name,amount)
            return result
        if key.startswith('action_head_'):
            frame=self._frame;require(frame is not None and key in ['action_head_'+str(h) for h in range(4)],'unknown/unadmitted action head')
            h=int(key[-1]);expected=[0] if frame['arm']==ARMS[0] else [0,1,2,3]
            require(len(frame['heads'])<len(expected) and h==expected[len(frame['heads'])],'inactive, reordered or extra action head')
            require(args[0].numel()//args[0].shape[-1]==frame['rows'],'action head row count differs')
            frame['heads'].append(h)
            extra={key+'_'+('control' if frame['arm']==ARMS[0] else 'candidate'):1,'action_output_values':7*frame['rows']}
            for name,amount in extra.items():self.begin(name,amount)
            result=super().call(key,fn,*args,**kwargs)
            for name,amount in extra.items():self.finish(name,amount)
            return result
        return super().call(key,fn,*args,**kwargs)
