"""Separate corrective freeze: one fit normalization per paired model/fold."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from experiments import libero_task7_endpoint_gradient_protocol as old

p = old.p
ROOT = old.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-endpoint-gradient-diagnostic_v2.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-endpoint-gradient-execution-binding_v2.json'
OUTPUT = '/tmp/libero-task7-endpoint-gradient-20261109-v2.json'
ARTIFACTS = '/tmp/libero-task7-endpoint-gradient-artifacts-20261109-v2'
PLAN = ROOT / 'docs/libero-task7-endpoint-gradient-correction-plan-2026-11-09.md'
TESTS = ROOT / 'tests/test_libero_task7_endpoint_gradient_diagnostic_v2.py'
DEVIATION = {'path': '/tmp/libero-task7-endpoint-gradient-v1-budget-deviation-20261109-v1.json',
             'sha256': 'daced34af4a4bbd67794b670383e643dbf21727b890fb3b262293ec2eb12d991'}
PARENT_BINDING = {'path': str(old.DEFAULT_BINDING), 'sha256': 'ed48c481ef370516f458c47d4d5760c51e47050510f798cd612d0d67c8db596c'}
CASES, POINTS = old.CASES, old.POINTS
SOURCE_MEMBERS, HEAD_MEMBERS = old.SOURCE_MEMBERS, old.HEAD_MEMBERS
remaining_evidence, joint_remaining_evidence = old.remaining_evidence, old.joint_remaining_evidence
FUTURE_FILES = {
    'primary': ROOT / 'experiments/run_libero_task7_endpoint_gradient_diagnostic_v2.py',
    'audit': ROOT / 'experiments/audit_libero_task7_endpoint_gradient_diagnostic_v2.py',
    'guard': ROOT / 'experiments/libero_task7_endpoint_gradient_access_v2.py',
    'metadata': Path(__file__).resolve(), 'tests': TESTS, 'protocol': PLAN,
}
FUTURE_CONTRACT = PLAN


def create():
    cfg = copy.deepcopy(old.validate())
    deviation = p.load_pin(DEVIATION)
    p.require(deviation['status'] == 'v1_numerically_audited_but_execution_budget_nonconforming'
              and p.pin(old.DEFAULT_BINDING) == PARENT_BINDING, 'fixed v1 deviation lineage required')
    cfg.update(version=2, parents={**cfg['parents'], 'v1_deviation': DEVIATION, 'v1_binding': PARENT_BINDING},
               code={'metadata': p.pin(Path(__file__).resolve()), 'tests': p.pin(TESTS)}, plan=p.pin(PLAN),
               future_files={k: str(v) for k, v in FUTURE_FILES.items()}, future_contract=str(PLAN),
               output=OUTPUT, artifacts=ARTIFACTS)
    cfg['design']['normalization_execution'] = 'once_per_model_fold;paired_head_bits_equal;reuse_same_readonly_z;explicit18build18reuse_work_ledger'
    cfg['corrective_execution'] = {'separate_v2_budget_before_payload': True, 'same_v1_retry': False,
        'v1_retained_not_rewritten': True, 'numerical_design_loss_roles_thresholds_and_endpoints_unchanged': True,
        'total_v1_plus_v2_NPZ_loads': 172, 'total_v1_plus_v2_member_decodes': 1220,
        'total_v1_plus_v2_archive_SHA_checks': 344, 'cumulative_parameter_updates': 0}
    return cfg


def validate(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(json.dumps(cfg, sort_keys=True, allow_nan=False) == json.dumps(create(), sort_keys=True, allow_nan=False), 'v2 scientific freeze differs')
    return cfg


def require_binding(config=DEFAULT_CONFIG, binding=DEFAULT_BINDING):
    cfg = validate(config); b = json.loads(Path(binding).read_text())
    p.require(b['status'] == 'bound_before_payload' and b['before_research_payload'] is True
        and b['science_config'] == p.pin(config) and b['budget'] == cfg['budget']
        and b['CPU_environment'] == cfg['execution']['environment']
        and b['future_output'] == OUTPUT and b['future_artifacts'] == ARTIFACTS, 'v2 binding missing or changed')
    refs = b['dependencies']; paths = [p.resolve(r['path']).resolve() for r in refs]
    p.require(len(paths) == len(set(paths)) and set(FUTURE_FILES.values()).union({DEFAULT_CONFIG, old.DEFAULT_BINDING}).issubset(paths), 'corrective implementation not bound')
    p.require(all(r in refs for r in p.load_pin(PARENT_BINDING)['dependencies']), 'v1 closure missing')
    for ref in refs:
        path = p.resolve(ref['path'])
        p.require(path.suffix in ('.py', '.md', '.json', '.toml') and p.pin(path) == ref, 'bound dependency differs')
    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-freeze', action='store_true')
    parser.add_argument('--require-binding', action='store_true')
    args = parser.parse_args()
    if args.write_freeze:
        with DEFAULT_CONFIG.open('x') as f: json.dump(create(), f, ensure_ascii=False, indent=2); f.write('\n')
        result = {'status': 'v2_scientific_frozen_no_payload'}
    else:
        validate(); result = {'status': 'v2_metadata_valid_no_payload'}
        if args.require_binding: require_binding(); result['status'] = 'bound_no_payload'
    print(json.dumps(result))
