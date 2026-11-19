"""Stage admission and exact compute accounting for frozen spatial readout v4."""
from __future__ import annotations
import json
from pathlib import Path
from experiments import libero_task7_view_specific_readout_protocol_v4 as science
from experiments.libero_formal_execution import Ledger

ROOT,require,pin,write_new=science.ROOT,science.require,science.pin,science.write_new
ARMS,SEEDS=science.ARMS,science.SEEDS
VIEWS=['agentview','eye_in_hand']
previous=science.prior_execution.previous
SHAPES=previous.SHAPES
HDF5_FIELDS=science.HDF5_FIELDS
TRAIN_CAPS=science.train_caps()
problems,decision=science.problems,science.decision
CODE=['experiments/libero_view_specific_readout_execution_v4.py','experiments/run_libero_view_specific_readout_v4.py',
      'experiments/audit_libero_view_specific_readout_v4.py','experiments/execute_libero_view_specific_readout_v4.py']

def caps(cfg,stage):return science.evaluation_caps(cfg,stage)

def preflight(config,binding,stage):
    cfg,frozen=science.validate_freeze(config,science.FREEZE)
    b=json.loads(Path(binding).read_text())
    require(stage in ['train','handoff','full'] and b.get('stage')==stage
        and b.get('status')=='bound_before_view_specific_readout_v4_payload' and b.get('science')==pin(config)
        and b.get('scientific_freeze')==pin(science.FREEZE),'new v4 stage execution binding required')
    for item in b['code_dependencies']+b['validation']:
        require(pin(item['path'])==item,'v4 execution dependency changed: '+item['path'])
    closure={r['path']:r['sha256'] for r in b['code_dependencies']}
    require({r['path']:r['sha256'] for r in frozen['dependencies']}.items()<=closure.items(),'v4 frozen closure missing')
    for relative in CODE:
        path=ROOT/relative;require(closure.get(str(path))==pin(path)['sha256'],'v4 execution code unbound')
    item=b['authorization'];require(pin(item['path'])==item,'v4 authorization changed')
    g=json.loads(Path(item['path']).read_text());expected=TRAIN_CAPS if stage=='train' else caps(cfg,stage)
    require(g['user_approved'] is True and g['science']==pin(config) and g['stage']==stage and g['tasks']==[7]
        and g['caps']==expected and g['sealed_roles_allowed']==[] and g['previous_episode_permission_reused'] is False
        and g['hdf5_fields']==HDF5_FIELDS and g['observation_fields']==list(SHAPES)
        and g['hdf5_allowed']==(stage=='train') and g['initial_indices']==([] if stage=='train' else list(range(19 if stage=='handoff' else 20))),
        'v4 stage authorization scope differs')
    if stage=='train':require(g['demonstrations']==list(range(30)) and g['source_paths']==[cfg['source']['path']],'v4 support scope differs')
    else:
        require(g['source_paths']==([s['source']['path'] for s in cfg['handoff_sources']] if stage=='handoff' else []),'v4 prefix scope differs')
        item=b['training_completion'];require(pin(item['path'])==item,'v4 training receipt changed')
        proof=json.loads(Path(item['path']).read_text());require(proof['status']=='complete_audited_postgrasp_training','audited v4 training required')
        for key in ['training','audit']:require(pin(proof[key]['path'])==proof[key],'v4 training proof changed')
        trained=json.loads(Path(proof['training']['path']).read_text())
        require(g['checkpoint_paths']==[r['checkpoint']['path'] for r in trained['models']],'v4 checkpoint authority differs')
    if stage=='full':
        item=b['handoff_completion'];require(pin(item['path'])==item,'v4 handoff receipt changed')
        proof=json.loads(Path(item['path']).read_text())
        require(proof['status']=='complete_audited_postgrasp_handoff' and proof['decision']['passed'] is True,'full denied without audited v4 handoff pass')
        for key in ['result','audit']:require(pin(proof[key]['path'])==proof[key],'v4 handoff proof changed')
    return cfg,b


class ComputeLedger(Ledger):
    """Admit aggregate and view counters before their ordered branch call."""
    optimizer_ref=None
    _frame=None
    last_prediction_entropy=None

    def call(self,key,fn,*args,**kwargs):
        if key=='optimizer':self.optimizer_ref=fn.__self__
        if key in ['train_forward','parity_forward','policy_forward','warm_forward']:
            require(self._frame is None,'nested policy forward forbidden')
            model=getattr(fn,'__self__',None) if key in ['train_forward','parity_forward'] else args[0]
            arm=model.get('controller') if isinstance(model,dict) else model.readout_arm
            require(arm in ARMS,'unknown active readout arm')
            rows=int(args[0].shape[0]*args[0].shape[1]) if key in ['train_forward','parity_forward'] else 1
            require(1<=rows<=80,'forward row cap differs')
            projection=getattr(getattr(model,'visual_encoder',None),'projection',None)
            identifiers=None if projection is None else ([id(projection)]*2 if arm==ARMS[0] else [id(projection[v]) for v in VIEWS])
            self._frame=dict(arm=arm,rows=rows,readouts=0,encoders=0,
                             expected_encoders=0 if key=='train_forward' else 2,projection_ids=identifiers)
            self.begin('zero_xyz_coordinate_writes',3*rows)
            try:
                result=super().call(key,fn,*args,**kwargs)
                require(self._frame['readouts']==2 and self._frame['encoders']==self._frame['expected_encoders'],
                        'per-forward view/encoder count differs')
                self.finish('zero_xyz_coordinate_writes',3*rows)
                return result
            finally:self._frame=None
        if key=='frozen_encoder' and self._frame is not None:
            require(self._frame['encoders']<self._frame['expected_encoders'],'extra encoder forward')
            self._frame['encoders']+=1
        spatial_keys=['spatial_readout_'+view for view in VIEWS]
        if key in ['flat_readout','spatial_readout',*spatial_keys]:
            frame=self._frame;require(frame is not None and frame['readouts']<2,'extra or unframed readout')
            index=frame['readouts'];expected='flat_readout' if frame['arm']==ARMS[0] else spatial_keys[index]
            require(key==expected,'wrong, duplicate or reordered view readout')
            if frame['projection_ids'] is not None:require(id(fn)==frame['projection_ids'][index],'wrong readout module identity')
            x=args[0];rows=int(x.shape[0]);require(tuple(x.shape[1:])==(512,4,4) and rows==frame['rows'],'readout tensor rows differ')
            frame['readouts']+=1
            extra={'flat_image_rows':rows} if key=='flat_readout' else dict(spatial_readout=1,spatial_image_rows=rows,
                spatial_heatmaps=rows*256,spatial_probability_values=rows*256*16,
                spatial_coordinate_values=rows*256*2,spatial_entropy_values=rows*256)
            if key!='flat_readout':
                view=VIEWS[index]
                extra.update({name+'_'+view:value for name,value in extra.items() if name!='spatial_readout'})
            for name,amount in extra.items():self.begin(name,amount)
            result=super().call(key,fn,*args,**kwargs)
            for name,amount in extra.items():self.finish(name,amount)
            return result
        return super().call(key,fn,*args,**kwargs)


