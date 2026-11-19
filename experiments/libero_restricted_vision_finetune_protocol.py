"""Metadata-only scope, stage admission and exact gates for the new contrast."""
from __future__ import annotations

from fractions import Fraction
import json
from pathlib import Path

from experiments.run_libero_pretrained_vision_control import POLICY, ROOT, pin


TASKS = [4, 7]
SEEDS = [20261106, 20261107, 20261108]
ARMS = ['frozen_resnet18', 'layer4_conv_finetune']
KIND = 'libero_restricted_visual_finetune_source_development'
RUNTIME = dict(CUDA_VISIBLE_DEVICES='1', MUJOCO_EGL_DEVICE_ID='1', MUJOCO_GL='egl',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               PYTHONHASHSEED='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')
TRAIN_CAPS = dict(train_forward=6000, backward=6000, optimizer=6000, checkpoint=12,
                  prefix_forward=302, layer4_forward=12024, parity_forward=12,
                  dataset_items=1100, source_sha=2, model_create=14,
                  adapter=0, reset=0, episode=0, policy_forward=0, policy_action=0,
                  zero_step=0, native_step=0, numeric_rows=0)
EVAL_CAPS = dict(train_forward=0, backward=0, optimizer=0, checkpoint=0,
                 prefix_forward=144000, layer4_forward=144000, parity_forward=0,
                 dataset_items=0, source_sha=0, model_create=12,
                 adapter=12, reset=240, episode=240, policy_forward=72000,
                 policy_action=72000, zero_step=1200, native_step=73200, numeric_rows=40)
OPTIMIZER = dict(name='Adam', lr=.0003, betas=[.9, .999], eps=1e-8,
                 weight_decay=0., amsgrad=False, global_gradient_clip=1.)
CACHE = dict(stage='layer3', shape=[256, 8, 8], dtype='float32', batch_windows=8,
             raw_bytes_per_task_max=2*1024**3, prefix_bytes_per_task_max=2*1024**3,
             immutable=True, final_feature_cache_allowed=False)
DECISION = dict(minimum_equal_weight_mean_gain_fraction='1/10',
                each_task_mean_nonregressing=True, each_seed_mean_nonregressing=True,
                paired_retention='descriptive_only_all_120_pairs',
                formal_transfer_admitted=False, stay_in_LIBERO=True)
MASK = dict(candidate='layer4_convolutions_only', frozen_BN_affine=True, frozen_BN_statistics=True,
            frozen_stem_and_layer1_to_3=True, control_trainable=1513352,
            candidate_trainable=9901960, total_parameters=12689864,
            extra_convolution_parameters=8388608, same_full_initial_state=True)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_scope(cfg):
    expected = dict(kind=KIND, version=1, formal_result=False, tasks=TASKS, seeds=SEEDS, arms=ARMS,
        updates_per_policy=500, stabilization_steps=5, demonstrations=list(range(30)),
        initial_indices=list(range(20)), loss='uniform_masked_seven_dimensional_mse',
        source_meta_query_allowed=False, formal_evaluation_allowed=False,
        source_role='meta_support', initial_role='candidate_nonregression', policy=POLICY,
        runtime_environment=RUNTIME, training_caps=TRAIN_CAPS, evaluation_caps=EVAL_CAPS,
        optimizer=OPTIMIZER, cache=CACHE, mask=MASK, decision=DECISION,
        training_wall_seconds_max=1800, evaluation_wall_seconds_max=7200,
        batch_size=8, sequence_length=10, stride=10, max_policy_steps=300,
        start='fresh_same_initialization_no_u500_resume')
    for name, value in expected.items():
        # JSON comparison also distinguishes booleans from integers.
        require(json.dumps(cfg.get(name), sort_keys=True) == json.dumps(value, sort_keys=True),
                'frozen scope differs: ' + name)
    require(Path(cfg['artifacts']).resolve().is_relative_to(ROOT / 'results'), 'persistent new output required')
    require(set(cfg['sources']) == {'4', '7'} and set(cfg['initial_banks']) == {'4', '7'}, 'source tasks changed')
    require(cfg['status'] == 'scientific_frozen_execution_unbound', 'new scientific freeze required')


def validate_science(config, freeze):
    """Only metadata/code pins are read. No dataset, weights or bank SHA here."""
    cfg = json.loads(Path(config).read_text())
    frozen = json.loads(Path(freeze).read_text())
    validate_scope(cfg)
    require(frozen['status'] == 'scientific_frozen_execution_unbound' and frozen['science'] == pin(config),
            'scientific freeze differs')
    require(frozen['execution_authorized'] is False and frozen['source_initial_authorized'] is False,
            'scientific freeze is not execution authority')
    for item in frozen['metadata_dependencies']:
        require(pin(item['path']) == item, 'scientific dependency changed: ' + item['path'])
    return cfg, frozen


def authorize_stage(bound, cfg, science, stage, *, authorization_loader):
    require(stage in ('train', 'evaluate'), 'unknown stage')
    require(bound.get('stage') == stage and bound.get('status') == 'bound_before_research_payload'
            and bound.get('science') == science and bound.get('synthetic_validation_passed') is True,
            'new stage execution binding required')
    scope = dict(tasks=TASKS, demonstration_role='meta_support', demonstration_indices=list(range(30)),
                 t_ge=1, source_meta_query=False, formal_evaluation=False)
    if stage == 'train':
        require(bound.get('source_support_authorization') == scope, 'explicit support-only execution scope required')
        return
    approval = bound.get('source_initial_authorization')
    require(isinstance(approval, dict), 'source initial states remain sealed without new authorization')
    granted = authorization_loader(approval)
    require(granted.get('user_approved') is True and granted.get('scientific_configuration') == science
            and granted.get('tasks') == TASKS and granted.get('indices') == list(range(20))
            and granted.get('role') == 'candidate_nonregression'
            and granted.get('episodes_max') == 240,
            'new trial source-initial authorization differs; V1 permission does not transfer')


def paired_decision(descriptors):
    expected = {(t, s, a, i) for t in TASKS for s in SEEDS for a in ARMS for i in range(20)}
    by_key = {}
    for row in descriptors:
        key = (row['task_id'], row['training_seed'], row['arm'], row['initial_state_index'])
        require(all(type(value) is int for value in (key[0], key[1], key[3]))
                and key in expected and key not in by_key and type(row['success']) is bool,
                'invalid or duplicate episode identity')
        by_key[key] = int(row['success'])
    require(set(by_key) == expected, 'complete240 paired episodes required')
    counts = {(t, s, a):sum(by_key[t, s, a, i] for i in range(20)) for t in TASKS for s in SEEDS for a in ARMS}
    tasks = {str(t):Fraction(sum(counts[t, s, ARMS[1]]-counts[t, s, ARMS[0]] for s in SEEDS),60) for t in TASKS}
    seeds = {str(s):Fraction(sum(counts[t, s, ARMS[1]]-counts[t, s, ARMS[0]] for t in TASKS),40) for s in SEEDS}
    delta = sum(tasks.values())/2
    gates = dict(mean_gain_at_least_ten_percentage_points=delta >= Fraction(1,10),
                 both_task_means_nonregressing=all(v >= 0 for v in tasks.values()),
                 all_seed_means_nonregressing=all(v >= 0 for v in seeds.values()))
    paired = dict(both_success=0, frozen_only=0, finetune_only=0, both_failure=0)
    for t in TASKS:
        for s in SEEDS:
            for i in range(20):
                frozen, tuned = (by_key[t,s,a,i] for a in ARMS)
                label = ('both_success' if tuned else 'frozen_only') if frozen else ('finetune_only' if tuned else 'both_failure')
                paired[label] += 1
    denominator = paired['both_success'] + paired['frozen_only']
    return dict(continue_to_all_source_development=all(gates.values()), gates=gates,
        mean_delta_fraction=str(delta), task_delta_fraction={k:str(v) for k,v in tasks.items()},
        seed_delta_fraction={k:str(v) for k,v in seeds.items()}, paired=paired,
        frozen_success_retention_fraction=str(Fraction(paired['both_success'],denominator)) if denominator else None,
        counts=[dict(task=t,seed=s,arm=a,successes=n,episodes=20) for (t,s,a),n in counts.items()],
        formal_transfer_or_module_admitted=False, stay_in_LIBERO=True)
