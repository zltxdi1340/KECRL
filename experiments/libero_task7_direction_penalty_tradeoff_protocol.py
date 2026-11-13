"""Metadata-only freeze for saved v2 scalar penalty accounting."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEM = 'libero-task7-direction-penalty-tradeoff'
CONFIG = ROOT / f'configs/{STEM}-readonly_v1.json'
BINDING = ROOT / f'configs/{STEM}-execution-binding_v1.json'
PLAN = ROOT / 'docs/libero-task7-direction-penalty-tradeoff-plan-2026-11-10.md'
ARTIFACTS = Path('/tmp/libero-task7-direction-penalty-tradeoff-20261110-v1')
OUTPUT = ARTIFACTS / 'analysis.json'
AUDIT = ARTIFACTS / 'independent-audit.json'
IDS = ['task', 'seed', 'arm', 'fold', 'mode']
ARMS = ['negative_total_gradient', 'norm_matched_joint_bisector']
STATE = IDS + ['optimization_arm', 'extra_updates', 'n', 'component3_mse', 'penalty', 'trainable_objective', 'exact_MSE_slope', 'exact_penalty_slope', 'directional_derivative', 'hinge_positive_n', 'branch_digest']
TRIAL = IDS + ['optimization_arm', 'from_extra_updates', 'trial_index', 'alpha', 'accepted']
FOCUS = [('A_original', 20261106, 0), ('B_fomaml', 20261107, 0), ('B_fomaml', 20261107, 2), ('B_fomaml', 20261108, 1)]
ANCHORS = [0, 1, 5, 10, 20, 30, 40, 50]
WINDOWS = [(0, 10), (10, 40), (40, 50)]
PARENTS = {
 'science': {'path': str(ROOT/'configs/libero-task7-joint-direction-diagnostic_v2.json'), 'sha256': '253061977b288bbec50a232e49b630f0f1e825370bdafeeabdc248e0106ac50c'},
 'binding': {'path': str(ROOT/'configs/libero-task7-joint-direction-execution-binding_v2.json'), 'sha256': 'd81444c3fc17c103343cd273941732c5e532ca434eba2b0956da4fee44a677d7'},
 'result': {'path': '/tmp/libero-task7-joint-direction-20261110-v2.json', 'sha256': '28615e083c656c5a5f6f779f958d6b831bc8c21c52272cc703e3f7247c4e5cc3'},
 'process': {'path': '/tmp/libero-task7-joint-direction-20261110-v2-process.json', 'sha256': 'c50646b9dbc0d60dfb0c15b8623cd5b453384f4afa00209c47a580e3a7cc1475'},
 'completion': {'path': '/tmp/libero-task7-joint-direction-artifacts-20261110-v2/completion.json', 'sha256': '2884652664aec65a8d170bc746d8fa3ec5a27d0e7f8e5fa5962014c0bfceecfb'},
 'report': {'path': str(ROOT/'docs/libero-task7-joint-direction-results-v2-2026-11-10.md'), 'sha256': 'e251b28445ed32efddf94b313b3dbb7a1307c57e975484d2390d5c838310522c'},
}
INPUTS = {
 'P': {'path': '/tmp/libero-task7-joint-direction-artifacts-20261110-v2/primary-result.json', 'sha256': 'f9d3c390ed1a2f96ef27fb1a7204f55ec2e88ae99d671a205aaa6d440355e058'},
 'U': {'path': '/tmp/libero-task7-joint-direction-artifacts-20261110-v2/independent-audit.json', 'sha256': '9be5913ecae646a8c9b1861b966c517396836ab15d340893fd7634b252fe6ca4'},
}
FILES = {name: ROOT / path for name, path in {
 'protocol': 'experiments/libero_task7_direction_penalty_tradeoff_protocol.py',
 'guard': 'experiments/libero_task7_direction_penalty_tradeoff_access.py',
 'analysis': 'experiments/analyze_libero_task7_direction_penalty_tradeoff.py',
 'audit': 'experiments/audit_libero_task7_direction_penalty_tradeoff.py',
 'tests': 'tests/test_libero_task7_direction_penalty_tradeoff.py',
 'plan': 'docs/libero-task7-direction-penalty-tradeoff-plan-2026-11-10.md',
}.items()}


def require(test, message):
    if not test:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path):
    return {'path': str(Path(path).resolve()), 'sha256': sha(path)}


def load(ref):
    require(pin(ref['path']) == ref, 'pinned metadata changed')
    return json.loads(Path(ref['path']).read_text())


def write(path, obj):
    with Path(path).open('x') as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write('\n')


def create():
    docs = {k: load(v) for k, v in PARENTS.items() if k != 'report'}
    require(pin(PARENTS['report']['path']) == PARENTS['report'], 'report changed')
    r, done = docs['result'], docs['completion']
    require(r['status'] == done['status'] == 'complete_independently_same_parameter_audited_direction_v2'
            and r['primary'] == INPUTS['P'] and r['audit'] == done['audit'] == INPUTS['U']
            and done['result'] == PARENTS['result'] and r['science_config'] == done['science_config'] == PARENTS['science']
            and r['binding'] == done['binding'] == PARENTS['binding']
            and all(x['exit_code'] == 0 for x in docs['process']['processes']), 'completed v2 provenance required')
    require(len(docs['binding']['dependencies']) == 804, 'parent closure differs')
    return dict(kind='task7_direction_v2_saved_scalar_penalty_tradeoff', version=1, formal_result=False,
                parents=PARENTS, inputs=INPUTS, identities=docs['science']['cohorts'],
                scope=dict(original_Policy_updates=500, future_stabilization_steps=5, loss_lambda=1,
                           original_Policy_training_forward_gradient_environment_and_sealed_access=0,
                           extra_array_CSV_checkpoint_HDF5_or_private_reads_including_SHA=0,
                           source='own20_fit_of_support0..29_tge1_encoder_seen_all30', outcome_conditioned=True),
                projections={'states': STATE, 'accepted_trials': TRIAL}, focus=[list(x) for x in FOCUS],
                anchors=ANCHORS, windows=[list(x) for x in WINDOWS],
                design=dict(reference='own500_matched_control_previous_accepted',
                            penalty='mean_relu(error_minus_originalPolicy_error)/7',
                            active_factorization='symmetric_product;zero_active_conditional_mean_null',
                            finite_remainder='observed_change_minus_alpha_times_saved_exact_directional_derivative',
                            strict_signs=True, conservation_absolute_tolerance=1e-10,
                            no_checkpoint_selection_no_pvalues_no_causal_claim=True),
                budget=dict(per_stage_source_JSON_passes=2, per_stage_states=2668,
                            per_stage_trial_flags=21840, per_stage_accepted_trials=2596,
                            per_stage_source_SHA_checks=4, stages=2, technical_retries=0,
                            output_rows={'own_endpoints':72,'pair_endpoints':36,'steps':2596,'matched_full50':714,'full50_paths':14}),
                execution=dict(python='3.10.22', device='CPU', seconds_per_stage=120, RSS_bytes=536870912,
                               overall_seconds=300, input_JSON_bytes_max=100663296, output_JSON_bytes_max=16777216,
                               environment={'CUDA_VISIBLE_DEVICES':'','PYTHONHASHSEED':'0','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}),
                roles={k: pin(v) for k,v in FILES.items()}, output=str(OUTPUT), audit=str(AUDIT))


def preflight():
    cfg = json.loads(CONFIG.read_text())
    require(cfg == create(), 'frozen scalar analysis protocol differs')
    binding = json.loads(BINDING.read_text())
    require(binding['before_payload'] is True and binding['science'] == pin(CONFIG)
            and binding['budget'] == cfg['budget'], 'execution binding required')
    refs = binding['dependencies']; paths = [r['path'] for r in refs]
    require(len(set(paths)) == len(paths) and all(pin(p) in refs for p in list(FILES.values())+[CONFIG]), 'new roles not bound')
    parent = load(PARENTS['binding'])
    require(all(r in refs for r in parent['dependencies']) and PARENTS['binding'] in refs, 'parent804 closure missing')
    for ref in refs:
        require(Path(ref['path']).suffix in {'.py','.md','.json','.toml'} and pin(ref['path']) == ref, 'bound code changed')
    return cfg, binding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        write(CONFIG, create())
    else:
        preflight()
    print(json.dumps({'status':'metadata_valid_no_P_U_payload_read'}))
