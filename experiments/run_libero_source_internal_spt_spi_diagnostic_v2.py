"""Version 2: deterministic equivalent pooling; source-internal A/B/C diagnostic; never a formal transfer result.

Default preflight reads only configuration/source metadata. A run requires an
exact code binding and an idle physical GPU. Qualification, validation,
held-out tasks, initial states and simulators have no execution path here.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARMS = ("A_original", "B_fomaml", "C_pooled_bc")
SEEDS = [20261022, 20261023, 20261024]
FOLDS = [{"probe": 0, "source_update_tasks": [1, 3, 4, 7]},
         {"probe": 1, "source_update_tasks": [0, 3, 4, 7]}]
CURVE = [0, 50, 150, 300, 500]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def validate_spec(config):
    expected = {
        "formal_result": False, "offline_only": True, "source_tasks": [0, 1, 3, 4, 7],
        "training_seeds": SEEDS, "folds": FOLDS, "arms": list(ARMS),
        "policy_memory_mode": "cnn_only", "alignment": "causal_pre_action_t_ge_1",
        "sequence_length": 10, "batch_size": 8, "stride": 10,
        "source_outer_updates": 50, "source_inner_steps": 1, "source_inner_lr": .001,
        "generator_outer_lr": .0003, "adaptation_updates": 500, "adaptation_lr": .0003,
        "gradient_clip_norm": 1., "curve_updates": CURVE,
        "new_environment_steps": 0, "official_initial_state_reads": 0,
        "validation_qualification_heldout_access": False,
        "candidate_acceptance": False, "module_registration": False, "knowledge_update": False,
        "physical_gpu": 1, "maximum_host_cache_bytes": 8*1024**3,
        "execution_revision": "v2_deterministic_equivalent_pool",
    }
    for key, value in expected.items():
        if config.get(key) != value or type(config.get(key)) is not type(value):
            raise ValueError(f"frozen source-internal setting differs: {key}")
    if config.get("budget_totals") != {
        "generator_optimizer_updates": 600, "disposable_source_sgd_updates": 1200,
        "source_data_gradient_calls": 4800, "generator_vjp_calls": 2400,
        "probe_adaptation_optimizer_updates": 9000, "probe_policies": 18,
        "probe_curve_checkpoints": 90,
    }:
        raise ValueError("declared total budget differs")


def preflight(config_path, binding_path):
    config = load(config_path)
    validate_spec(config)
    binding = load(binding_path)
    if binding.get("formal_result") is not False or binding.get("config_sha256") != sha256(config_path):
        raise ValueError("execution binding differs from the frozen config")
    required_files = {"runner", "fomaml", "causal_dataset", "base_dataset", "policy", "spt", "models", "contracts", "deterministic_pool"}
    if not required_files.issubset(binding.get("files", {})):
        raise ValueError("execution code binding is missing")
    for name, pin in {**config["references"], **binding["files"]}.items():
        path = Path(pin["path"])
        path = path if path.is_absolute() else ROOT/path
        if sha256(path) != pin["sha256"]:
            raise ValueError(f"pinned source differs: {name}")
    runner_path = Path(binding["files"]["runner"]["path"])
    runner_path = runner_path if runner_path.is_absolute() else ROOT/runner_path
    if runner_path.resolve() != Path(__file__).resolve():
        raise ValueError("runner binding points to a different entry")
    transfer = load(ROOT/config["references"]["transfer"]["path"])
    if transfer["task_roles"]["source_spi"] != config["source_tasks"]:
        raise ValueError("source task roles changed")
    partitions = transfer["demonstration_indices_by_task_role"]["source_spi"]
    if partitions != {"meta_support": list(range(30)), "meta_query": list(range(30, 50))}:
        raise ValueError("source demonstration roles changed")
    family = load(ROOT/config["references"]["family"]["path"])
    geometries = {row["task_id"]:row["source_geometry"] for row in family["tasks"]}
    for fold in config["folds"]:
        if fold["probe"] in fold["source_update_tasks"] or set(fold["source_update_tasks"]) | {fold["probe"]} != set(config["source_tasks"]):
            raise ValueError("probe leaked into source updates")
        if len({geometries[t] for t in fold["source_update_tasks"]}) != 4:
            raise ValueError("source geometry coverage changed")
    return config, binding


def idle_gpu(index):
    lines = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], text=True)
    row = next([v.strip() for v in line.split(',')] for line in lines.splitlines() if int(line.split(',')[0]) == index)
    if int(row[3]) > 512 or int(row[4]) > 5 or any(row[1] in line for line in processes.splitlines()):
        raise RuntimeError("target GPU is occupied; source training was not started")
    import os
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(index):
        raise ValueError("CUDA_VISIBLE_DEVICES must pin the frozen physical GPU")
    return {"gpu_rows_before_run": lines, "compute_processes_before_run": processes, "target_gpu": row}


def run(config_path, binding_path, output, artifacts):
    entry_started = time.monotonic()
    requested_at_utc = datetime.now(timezone.utc).isoformat()
    config, binding = preflight(config_path, binding_path)
    if output.exists() or artifacts.exists():
        raise FileExistsError("source diagnostic artifacts must be new")
    gpu = idle_gpu(config["physical_gpu"])
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, default_collate
    from src.algorithms.libero_bc_fomaml import LiberoBCFirstOrderLearner, LiberoBCFOMAMLConfig, LiberoBCMetaTask, bc_loss
    from src.algorithms.libero_bc_policy import LiberoBCPolicyConfig
    from src.algorithms.libero_deterministic_pool import make_deterministic_libero_policy as LiberoBCPolicy, cuda_pool_equivalence_gate
    from src.environments.libero_causal_t1_dataset import CausalT1LiberoSequenceDataset
    from src.skills.libero_pick_place_spt import LearnableLiberoPickPlaceSPT, build_libero_pick_place_spt, instantiate_libero_pick_place_spi
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one selected CUDA device is required")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    pool_equivalence = cuda_pool_equivalence_gate()
    transfer = load(ROOT/config["references"]["transfer"]["path"])
    manifest = load(ROOT/config["references"]["dataset_manifest"]["path"])
    family = load(ROOT/config["references"]["family"]["path"])
    learning = load(ROOT/config["references"]["learning"]["path"])
    learning["policy"]["memory_mode"] = "cnn_only"
    learning["policy"]["policy_family"] = "dual_view_shared_cnn_only_uniform_bc_diagnostic_v1"
    root = Path(config["dataset_root"])
    rows = {row["task_id"]: row for row in manifest["files"]}
    verified = []
    for task in config["source_tasks"]:
        path = (root/rows[task]["path"]).resolve()
        if not path.is_relative_to(root.resolve()) or path.stat().st_size != rows[task]["size_bytes"] or sha256(path) != rows[task]["sha256"]:
            raise ValueError(f"source dataset provenance differs: {task}")
        verified.append({"task":task,"path":str(path),"sha256":rows[task]["sha256"]})
    artifacts.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat()
    events_path = artifacts/"events.jsonl"
    events = events_path.open("x")
    count = {key:0 for key in config["budget_totals"]}
    forward_calls = {"source_training":0, "probe_training":0, "offline_evaluation":0}
    valid_prediction_rows = {"source_training":0, "probe_training":0, "offline_evaluation":0}
    access = []
    folded = []
    cache = {}
    cache_bytes = 0
    peak_cache_bytes = 0
    timings = {"dataset_cache_seconds":0., "source_update_loop_seconds":0.,
               "probe_fit_loop_including_checkpoint_logging_seconds":0., "offline_evaluation_seconds":0.}
    def event(kind, **values):
        events.write(json.dumps({"event":kind,"utc":datetime.now(timezone.utc).isoformat(),**values}, allow_nan=False)+"\n")
        events.flush()
    def tensor_digest(values):
        h=hashlib.sha256()
        for name,value in values.items():
            h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def dataset(task, role):
        nonlocal cache_bytes, peak_cache_bytes
        if (task,role) not in cache:
            cache_started=time.monotonic()
            ds=CausalT1LiberoSequenceDataset(transfer,manifest,root,task_id=task,demonstration_role=role,
                                            sequence_length=10,stride=10,verify_sha256=False)
            items=[ds[i] for i in range(len(ds))]
            size=sum(v.numel()*v.element_size() for item in items for v in item.values() if torch.is_tensor(v))
            cache_bytes+=size
            peak_cache_bytes=max(peak_cache_bytes,cache_bytes)
            if cache_bytes>config["maximum_host_cache_bytes"]:
                raise MemoryError("frozen host cache budget exceeded")
            access.append({**ds.access_record,"file_sha256_verified_before_access":True,
                           "read_all_authorized_t_ge_1_payload":True,"cache_bytes":size})
            ds.close(); cache[(task,role)]=items
            timings['dataset_cache_seconds']+=time.monotonic()-cache_started
            event("dataset_role_cached",task=task,role=role,rows=len(items),cache_bytes=cache_bytes)
        return cache[(task,role)]
    def to_device(batch):
        return {k:v.to(device) if torch.is_tensor(v) else v for k,v in batch.items()}
    def sample(items, rng):
        indexes=torch.randint(len(items),(8,),generator=rng).tolist()
        batch=default_collate([items[i] for i in indexes])
        keys=[(int(items[i]['demonstration_index']),int(items[i]['start_index'])) for i in indexes]
        return to_device(batch),keys
    def spi_for(generator, task, phase):
        spi,_=instantiate_libero_pick_place_spi(generator.record,family,learning,task)
        return replace(spi,adaptation_spec={**spi.adaptation_spec,"usage_phase":phase,
            "optimizer":"SGD" if phase=="source_inner" else "Adam",
            "optimizer_updates":1 if phase=="source_inner" else 500,
            "learning_rate":.001 if phase=="source_inner" else .0003,
            "training_role":"meta_support","query_role":"meta_query",
            "query_use":"source_outer_feedback" if phase=="source_inner" else "evaluation_only_after_all_fits",
            "observation_action_alignment":"causal_pre_action_t_ge_1"})
    def save_weights(path, model):
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        with path.open('xb') as handle:torch.save(state,handle)
        return {"path":str(path),"sha256":sha256(path),"weight_hash":tensor_digest(state)}
    @torch.no_grad()
    def evaluate(policy, items):
        policy.eval()
        total=np.zeros(7,dtype=np.float64); valid=0; correct=0
        for batch in DataLoader(items,batch_size=8,shuffle=False,num_workers=0):
            b=to_device(batch)
            prediction,_=policy.forward_sequence(*(b[k] for k in ('agentview_rgb','eye_in_hand_rgb','joint_states','gripper_states')))
            forward_calls['offline_evaluation']+=1
            mask=b['valid_mask'].bool(); pred=prediction[mask]; target=b['actions'][mask]
            valid_prediction_rows['offline_evaluation']+=int(mask.sum())
            if not torch.isfinite(pred).all():raise ValueError('nonfinite evaluation readout')
            total+=(pred.double()-target.double()).square().sum(0).cpu().numpy()
            valid+=int(mask.sum()); correct+=int((torch.sign(pred[:,6])==torch.sign(target[:,6])).sum())
        if not valid:raise ValueError('empty evaluation')
        return {'valid_steps':valid,'mse':float(total.sum()/(valid*7)),
                'translation_mse':float(total[:3].sum()/(valid*3)),
                'rotation_mse':float(total[3:6].sum()/(valid*3)),
                'gripper_mse':float(total[6]/valid),'gripper_sign_accuracy':correct/valid}
    try:
        for fold in FOLDS:
            probe=fold['probe']; cache={}; cache_bytes=0
            event('fold_start',probe=probe,source_update_tasks=fold['source_update_tasks'])
            source_data={t:{r:dataset(t,r) for r in ('meta_support','meta_query')} for t in fold['source_update_tasks']}
            probe_support=dataset(probe,'meta_support')
            seed_records=[]
            for seed in SEEDS:
                torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
                original=LearnableLiberoPickPlaceSPT(build_libero_pick_place_spt(family,learning),
                    LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(learning['policy']))).to(device)
                generators={ARMS[0]:original,ARMS[1]:original.clone_candidate(),ARMS[2]:original.clone_candidate()}
                initial_hash=tensor_digest(original.state_dict())
                for arm,g in generators.items():
                    g.record=replace(g.record,version=f'0.1.0-{arm}-probe{probe}-seed{seed}',
                                     omega={**g.record.omega,'status':'isolated_reference' if arm==ARMS[0] else 'isolated_unaccepted_candidate'})
                    if tensor_digest(g.state_dict())!=initial_hash:raise ValueError('unpaired generator initialization')
                learners={arm:LiberoBCFirstOrderLearner(generators[arm],LiberoBCFOMAMLConfig()) for arm in ARMS[1:]}
                rng=torch.Generator().manual_seed(seed+probe*10000)
                source_curve=[]
                source_started=time.monotonic()
                for step in range(1,51):
                    shared={}; ids={}
                    for task in fold['source_update_tasks']:
                        support,si=sample(source_data[task]['meta_support'],rng)
                        query,qi=sample(source_data[task]['meta_query'],rng)
                        shared[task]=(support,query); ids[str(task)]={'support':si,'query':qi}
                    record={'outer_update':step,'paired_window_ids':ids,'arms':{}}
                    for arm in ARMS[1:]:
                        tasks=tuple(LiberoBCMetaTask(spi_for(generators[arm],task,'source_inner'),*shared[task]) for task in fold['source_update_tasks'])
                        log=learners[arm].step(tasks,arm); record['arms'][arm]=log
                        count['generator_optimizer_updates']+=1
                        count['source_data_gradient_calls']+=log['data_gradient_calls']
                        forward_calls['source_training']+=log['data_gradient_calls']
                        valid_prediction_rows['source_training']+=sum(int(b['valid_mask'].sum()) for pair in shared.values() for b in pair)
                        count['generator_vjp_calls']+=log['generator_vjp_calls']
                        count['disposable_source_sgd_updates']+=log['disposable_policy_sgd_updates']
                    source_curve.append(record)
                timings['source_update_loop_seconds']+=time.monotonic()-source_started
                if tensor_digest(original.state_dict())!=initial_hash:raise ValueError('original SPT was mutated')
                source_endpoints={arm:save_weights(artifacts/f'p{probe}-s{seed}-{arm}-generator.pt',g) for arm,g in generators.items()}
                arm_records={}
                for arm,generator in generators.items():
                    spi=spi_for(generator,probe,'probe_adaptation')
                    policy=generator.instantiate(spi).policy.to(device)
                    optimizer=torch.optim.Adam(policy.parameters(),lr=.0003,betas=(.9,.999),eps=1e-8,weight_decay=0.)
                    rng=torch.Generator().manual_seed(seed+probe*10000+777)
                    curves={}; sampling=hashlib.sha256()
                    fit_started=time.monotonic()
                    for step in range(501):
                        if step in CURVE:
                            curves[str(step)]=save_weights(artifacts/f'p{probe}-s{seed}-{arm}-u{step}.pt',policy)
                            count['probe_curve_checkpoints']+=1
                        if step==500:break
                        b,ids=sample(probe_support,rng); sampling.update(json.dumps(ids).encode())
                        policy.train(); optimizer.zero_grad(set_to_none=True)
                        loss=bc_loss(policy,b); loss.backward()
                        forward_calls['probe_training']+=1
                        valid_prediction_rows['probe_training']+=int(b['valid_mask'].sum())
                        norm=torch.nn.utils.clip_grad_norm_(policy.parameters(),1.,error_if_nonfinite=True)
                        optimizer.step(); count['probe_adaptation_optimizer_updates']+=1
                        event('probe_adaptation_update',probe=probe,seed=seed,arm=arm,update=step+1,
                              loss=float(loss.detach()),gradient_norm=float(norm))
                    timings['probe_fit_loop_including_checkpoint_logging_seconds']+=time.monotonic()-fit_started
                    if tensor_digest(generator.state_dict())!=source_endpoints[arm]['weight_hash']:
                        raise ValueError('probe adaptation mutated its SPT generator')
                    count['probe_policies']+=1
                    arm_records[arm]={'spi':asdict(spi),'sampling_sha256':sampling.hexdigest(),'curve':curves,
                                      'generator':source_endpoints[arm],'initial_generator_weight_hash':initial_hash}
                if len({r['sampling_sha256'] for r in arm_records.values()})!=1:
                    raise ValueError('unpaired probe adaptation batch order')
                seed_records.append({'seed':seed,'arms':arm_records,'source_training_curve':source_curve})
                event('seed_all_support_fits_complete',probe=probe,seed=seed)
                print(json.dumps({'progress':'source_and_probe_fits_complete','probe':probe,'seed':seed,'counts':count}),flush=True)
            # The probe query payload/dataset is first created only after all
            # nine support fits in this fold; it can never feed a source update.
            if len(seed_records)!=3 or count['probe_policies'] != (probe+1)*9:
                raise ValueError('probe query gate: incomplete support fits')
            event('fold_all_support_fits_complete',probe=probe,models=9)
            probe_query=dataset(probe,'meta_query')
            evaluation_started=time.monotonic()
            for record in seed_records:
                for arm,entry in record['arms'].items():
                    policy=LiberoBCPolicy(LiberoBCPolicyConfig.from_mapping(learning['policy'])).to(device)
                    for checkpoint in entry['curve'].values():
                        if sha256(checkpoint['path'])!=checkpoint['sha256']:raise ValueError('checkpoint changed before evaluation')
                        policy.load_state_dict(torch.load(checkpoint['path'],map_location=device,weights_only=True))
                        checkpoint['meta_support']=evaluate(policy,probe_support)
                        checkpoint['meta_query']=evaluate(policy,probe_query)
            timings['offline_evaluation_seconds']+=time.monotonic()-evaluation_started
            folded.append({'probe':probe,'source_update_tasks':fold['source_update_tasks'],'seed_results':seed_records})
            event('fold_complete',probe=probe)
        if count!=config['budget_totals']:raise ValueError('actual source diagnostic budget mismatch')
        result={'status':'complete','formal_result':False,'diagnostic_result':True,'config':config,
                'config_sha256':sha256(config_path),'binding':binding,'binding_sha256':sha256(binding_path),
                'source_files_verified':verified,'access_records':access,'actual_counts':count,'fold_results':folded,
                'forward_calls':forward_calls,'valid_prediction_rows':valid_prediction_rows,
                'runtime':{'requested_at_utc':requested_at_utc,'started_at_utc':started_utc,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
                           'elapsed_seconds':time.monotonic()-started,'python':sys.version.split()[0],'torch':torch.__version__,
                           'total_entry_elapsed_seconds':time.monotonic()-entry_started,
                           'verification_and_setup_seconds':started-entry_started,'stage_timings':timings,
                           'peak_host_tensor_cache_bytes':peak_cache_bytes,
                           'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(0),
                           'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(0),
                           'numpy':np.__version__,'device':str(device),'cuda':torch.version.cuda,'gpu':gpu,
                           'gpu_name':torch.cuda.get_device_name(0),'deterministic_algorithms':True,'tf32':False,
                           'pool_equivalence_gate':pool_equivalence,
                           'launch_argv':sys.argv,
                           'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                           'git_status':subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True)},
                'new_environment_or_official_initial_state_calls':0,'active_spt_library_loaded_or_updated':False,
                'candidate_accepted':False,'module_registered':False,'knowledge_updated':False,
                'checkpoint_selection':False,'probe_tasks_previously_exposed_source_tasks':True,
                'formal_unseen_transfer_claim':False,'event_log':{'path':str(events_path)}}
        events.close(); result['event_log']['sha256']=sha256(events_path)
        with output.open('x') as handle:json.dump(result,handle,ensure_ascii=False,indent=2,allow_nan=False);handle.write('\n')
        return result
    except BaseException as error:
        event('failed',error_type=type(error).__name__,message=str(error),actual_counts=count)
        events.close()
        partial=output.with_name(output.stem+'-failed-partial.json')
        with partial.open('x') as handle:
            json.dump({'status':'failed','formal_result':False,'error':repr(error),'actual_counts':count,
                       'config_sha256':sha256(config_path),'binding_sha256':sha256(binding_path),
                       'forward_calls':forward_calls,'valid_prediction_rows':valid_prediction_rows,
                       'access_records':access,'completed_folds':folded,'event_log':str(events_path)},handle,indent=2,allow_nan=False)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--binding',type=Path,required=True)
    parser.add_argument('--mode',choices=['preflight','run'],default='preflight')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--artifacts',type=Path)
    args=parser.parse_args()
    config,binding=preflight(args.config,args.binding)
    if args.mode=='preflight':
        print(json.dumps({'status':'preflight_passed','config_sha256':sha256(args.config),
                          'binding_sha256':sha256(args.binding),'budget_totals':config['budget_totals'],
                          'new_model_HDF5_environment_calls':0}));return
    if args.output is None or args.artifacts is None:parser.error('run requires new --output and --artifacts paths')
    result=run(args.config,args.binding,args.output,args.artifacts)
    print(json.dumps({'status':result['status'],'actual_counts':result['actual_counts'],
                      'elapsed_seconds':result['runtime']['elapsed_seconds'],'output':str(args.output)}))


if __name__=='__main__':
    main()
