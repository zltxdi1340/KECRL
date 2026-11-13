"""Metadata-only fixed500 source-support readout protocol and selective parser."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
TASKS=[0,1,3,4,7]
SEEDS=[20261106,20261107,20261108]
ARMS=['A_original','B_fomaml']
OBS=['agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states']
DEFAULT_CONFIG=ROOT/'configs/libero-source-expert-late-translation-readout_v1.json'
DEFAULT_BINDING=ROOT/'configs/libero-source-expert-late-translation-execution-binding_v1.json'
OUTPUT=Path('/tmp/libero-source-expert-late-translation-readout-20261107-v1.json')
ARCHIVE=Path('/tmp/libero-source-expert-late-translation-artifacts-20261107-v1')
DEVICE={'physical_index':1,'uuid':'GPU-d90bd7b2-16f6-c0fe-fbc1-7ed4060e95ff','inference_device':'cuda:0'}
LAUNCH_ENV={'CUDA_VISIBLE_DEVICES':'1','CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTHONHASHSEED':'0',
            'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
            'MUJOCO_GL':'egl','MUJOCO_EGL_DEVICE_ID':'1'}


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def pin(path):return {'path':str(path),'sha256':sha(path)}


def resolve(path):
    p=Path(path);return p if p.is_absolute() else ROOT/p


def load_pin(item):
    p=resolve(item['path']);require(sha(p)==item['sha256'],f'metadata hash differs: {p}')
    return json.loads(p.read_text())


def skip_ws(text,i):
    while i<len(text) and text[i].isspace():i+=1
    return i


def value_end(text,start):
    """Skip a value lexically without decoding its numeric/object contents."""
    i=skip_ws(text,start);require(i<len(text),'truncated JSON')
    if text[i]=='"':
        i+=1;escape=False
        while i<len(text):
            c=text[i];i+=1
            if escape:escape=False
            elif c=='\\':escape=True
            elif c=='"':return i
        raise ValueError('unterminated JSON string')
    if text[i] in '[{':
        stack=[text[i]];i+=1
        while stack and i<len(text):
            c=text[i]
            if c=='"':i=value_end(text,i);continue
            if c in '[{':stack.append(c)
            elif c in ']}':
                require(stack.pop()==('[' if c==']' else '{'),'unbalanced JSON container')
            i+=1
        require(not stack,'truncated JSON container');return i
    end=i
    while end<len(text) and text[end] not in ',]} \t\n\r':end+=1
    require(end>i,'missing JSON scalar');return end


def object_fields(text,start=0):
    i=skip_ws(text,start);require(text[i]=='{','expected JSON object');i=skip_ws(text,i+1)
    fields={}
    while text[i]!='}':
        end=value_end(text,i);key=json.loads(text[i:end]);require(type(key) is str and key not in fields,'duplicate JSON field')
        i=skip_ws(text,end);require(text[i]==':','missing JSON colon');lo=skip_ws(text,i+1);hi=value_end(text,lo)
        fields[key]=(lo,hi);i=skip_ws(text,hi)
        if text[i]=='}':break
        require(text[i]==',','missing JSON comma');i=skip_ws(text,i+1)
    return fields


def array_spans(text,start):
    i=skip_ws(text,start);require(text[i]=='[','expected JSON array');i=skip_ws(text,i+1)
    while text[i]!=']':
        end=value_end(text,i);yield i,end;i=skip_ws(text,end)
        if text[i]==']':break
        require(text[i]==',','missing array comma');i=skip_ws(text,i+1)


def select_training_metadata(text,endpoints):
    fields=object_fields(text)
    decode=lambda span:json.loads(text[span[0]:span[1]])
    require(decode(fields['status'])=='complete_training' and decode(fields['stage'])=='G','G training not complete')
    expected={(r['task'],r['seed'],r['arm']):r for r in endpoints};selected={};scanned=0
    for lo,hi in array_spans(text,fields['models'][0]):
        row_fields=object_fields(text,lo)
        identity=tuple(decode(row_fields[k]) for k in ['task','seed','arm']);scanned+=1
        if identity not in expected:continue
        require(identity not in selected,'duplicate selected model')
        model=json.loads(text[lo:hi]);ref=model['curve']['500'];endpoint=expected[identity]
        require(model['support_role']=='meta_support' and model['query_role'] is None and
                all(ref[k]==endpoint['checkpoint'][k] for k in ['path','sha256','weight_hash']),
                'selected source500 lineage differs')
        require(set(ref['meta_support'])=={'valid_steps','mse','translation_mse','rotation_mse','gripper_mse','gripper_sign_accuracy'},
                'G source500 metric schema differs')
        selected[identity]=ref['meta_support']
    require(set(selected)==set(expected) and len(selected)==30 and scanned==320,'incomplete G selected metadata')
    references=[{'task':t,'seed':s,'arm':a,'metrics':selected[t,s,a]} for t in TASKS for s in SEEDS for a in ARMS]
    return {'effective_policy_config':decode(fields['effective_policy_config']),'source500_references':references,
            'selection_audit':{'model_identity_records_scanned':scanned,'source_records_decoded':30,
                               'nonselected_model_records_decoded':0,'validation_curve_payloads_decoded':0}}


def window_plan(lengths):
    require(len(lengths)==30 and all(type(n) is int and n>=2 for n in lengths),'source demo lengths differ')
    return [{'demo':d,'start':k,'valid':min(10,n-k)} for d,n in enumerate(lengths) for k in range(1,n,10)]


def design():
    return {'tasks':TASKS,'seeds':SEEDS,'arms':ARMS,'role':'meta_support','demos':list(range(30)),
            'checkpoint_update':500,'new_optimizer_updates':0,'closed_loop_stabilization':5,
            'offline_extra_stabilization':0,'sequence_length':10,'stride':10,'batch_size':32,
            'tail_last_valid_actions_per_demo':20,'causal_alignment':'obs[t-1] -> action[t], t>=1',
            'dtype':'float32','deterministic':True,'TF32':False,'augmentation':False,
            'loss':'uniform_seven_dimensional_valid_step_masked_action_MSE','memory_mode':'cnn_only',
            'metrics':['mse','translation_mse','rotation_mse','gripper_mse','gripper_sign_accuracy'],
            'strata':['all','tail20','complement'],'G_parity_absolute_tolerance':1e-10,'G_parity_relative_tolerance':0.,
            'repeat_first_batch_per_model':1,'repeat_exact_including_padding':True,'new_training_or_checkpoint_selection':False,
            'main_forward_batches':378,'repeat_forward_batches':30,'active_seconds':600,
            'GPU_wait_seconds_max':3600,'GPU_wait_poll_seconds':30,'technical_retry_or_replacement':0,
            'host_cache_bytes_max':2*1024**3,'raw_RGB_or_private_state_archive':False,
            'SPT_Knowledge_Module_changes':False,'G_failed_H_unrun_and_sealed_roles_unchanged':True}


def budget(config):
    windows={t:window_plan(config['action_lengths'][str(t)]) for t in TASKS}
    valid=sum(sum(w['valid'] for w in windows[t]) for t in TASKS)
    main=6*sum(math.ceil(len(windows[t])/32) for t in TASKS)
    repeat_rows=6*sum(sum(w['valid'] for w in windows[t][:32]) for t in TASKS)
    require(valid==18648 and main==378,'readout payload budget differs')
    return {'model_load':30,'policy_object':30,'HDF5_open':10,'source_window':sum(map(len,windows.values())),
            **{'payload_rows_'+k:valid for k in OBS+['actions']},
            'main_forward':main,'repeat_forward':30,'main_valid_rows':6*valid,'repeat_valid_rows':repeat_rows,
            'main_slots':60*sum(map(len,windows.values())),'repeat_slots':9600,'repeat_exact':30,'G_parity':30,
            'input_unchanged':378,'input_paired':315,'weight_after':30,
            **dict.fromkeys(['backward','optimizer','environment','initial_state','private_state','sealed_payload','old_prediction_load'],0)}


def check_device_binding(binding):
    require(binding['device']==DEVICE and binding['launch_environment']==LAUNCH_ENV,'frozen GPU identity/launch environment differs')


def preflight(config_path=DEFAULT_CONFIG,binding_path=None):
    config=json.loads(Path(config_path).read_text())
    require(config['formal_result'] is False and config['design']==design(),'frozen diagnostic definition differs')
    parents={k:load_pin(v) for k,v in config['inputs'].items() if k!='G_training'}
    require(set(parents)=={'source_capture_config','G_manifest','transfer','dataset_manifest','native_goal_completion','cache_completion'},'unexpected protocol metadata')
    require(parents['native_goal_completion']['G_failed_H_not_run_sealed_roles_unchanged'] and
            parents['cache_completion']['G_failed_H_unrun_sealed_roles_and500_stabilization5_currentloss_unchanged'],
            'negative G or role boundary missing')
    source=parents['source_capture_config']
    require(config['endpoints']==source['model_endpoints'] and len(config['endpoints'])==30,'source500 endpoints changed')
    require(set(config['action_lengths'])==set(map(str,TASKS)),'unexpected task lengths')
    for task in TASKS:
        require(config['action_lengths'][str(task)]==[parents['G_manifest']['action_lengths_by_task'][str(task)][str(d)] for d in range(30)],'demo lengths changed')
    require(config['budget']==budget(config),'derived resource caps differ')
    item=config['inputs']['G_training'];require(sha(item['path'])==item['sha256'],'G training metadata changed')
    selected=select_training_metadata(Path(item['path']).read_text(),config['endpoints'])
    require(selected==config['G_reference_selection'],'selected source reference/config changed')
    require(selected['effective_policy_config']['memory_mode']=='cnn_only','non-CNN readout')
    binding=None
    if binding_path is not None:
        binding=json.loads(Path(binding_path).read_text())
        require(binding['config_sha256']==sha(config_path) and binding['status']=='source_expert_late_translation_execution_bound', 'execution binding differs')
        require({'runner','protocol','auditor','tests','contract'}<=set(binding['files']),'missing execution code')
        for item in binding['files'].values():require(sha(resolve(item['path']))==item['sha256'],'bound execution code changed')
        require(binding['budget']==config['budget'],'execution budget changed')
        check_device_binding(binding)
        require(resolve(binding['files']['runner']['path'])==ROOT/'experiments/run_libero_source_expert_late_translation_readout.py',
                'wrong fixed source readout entry')
    return config,parents,binding
