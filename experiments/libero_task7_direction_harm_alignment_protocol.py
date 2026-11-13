"""Metadata-only freeze of saved direction errors versus continuous-action harm."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
from experiments import libero_source_expert_late_translation_protocol as p

ROOT = p.ROOT
DEFAULT_CONFIG = ROOT / 'configs/libero-task7-direction-harm-alignment-readonly_v1.json'
DEFAULT_BINDING = ROOT / 'configs/libero-task7-direction-harm-alignment-execution-binding_v1.json'
OUTPUT = '/tmp/libero-task7-direction-harm-alignment-20261109-v1.json'
ARTIFACTS = '/tmp/libero-task7-direction-harm-alignment-artifacts-20261109-v1'
PARENTS = {
    'direction_science': {'path': str(ROOT / 'configs/libero-task7-needed-direction-diagnostic_v1.json'), 'sha256': '7f9308766a36a681320f6cdec3d0ce9fbde1234f88f01d5710a38430ddaaebdc'},
    'direction_result': {'path': '/tmp/libero-task7-needed-direction-20261109-v1.json', 'sha256': '6faa031dbf04e753eb286580161eeab3e000d2f67ca9c2854ee1158a7c978d73'},
    'direction_audit': {'path': '/tmp/libero-task7-needed-direction-artifacts-20261109-v1/independent-audit.json', 'sha256': '49c86819a73a9e82185afd9179037cd2c8a78f4a33c85e8d34fee29a74a327f8'},
    'direction_completion': {'path': '/tmp/libero-task7-needed-direction-artifacts-20261109-v1/completion.json', 'sha256': '8a9231c78f72f7ceec346edd9f53532dd7d30f623d76679a17fb49423964fe2f'},
    'direction_summary': {'path': '/tmp/libero-task7-needed-direction-artifacts-20261109-v1/report-summary.json', 'sha256': '76e9957f21e6f587e988707160807db0197f00bdbc4fa0fb87bf4da01cf0ca33'},
    'summary_audit': {'path': '/tmp/libero-task7-needed-direction-artifacts-20261109-v1/report-summary-audit.json', 'sha256': '63d1bbf9575c497f27a316b73819cd14699ba34f48d8d1851adbcc492c424e16'},
    'continuous_result': {'path': '/tmp/libero-task7-protection-residual-decomposition-20261108-v1.json', 'sha256': '74f1315025fcd789e64b9bc70a727cb2235826d2240adffda3ceb72d1b2db3ee'},
    'continuous_audit': {'path': '/tmp/libero-task7-protection-residual-decomposition-artifacts-20261108-v1/independent-audit.json', 'sha256': '8aae93307aab1e65d810044de46bb10add96258a08ac15114cb27771d24f2c88'},
    'continuous_completion': {'path': '/tmp/libero-task7-protection-residual-decomposition-artifacts-20261108-v1/completion.json', 'sha256': 'c5787ae0f9f24f3e27ccf928a2db9adba2f7286700133cd22600e830bee5d4ba'},
    'previous_binding': {'path': str(ROOT / 'configs/libero-task7-needed-direction-execution-binding_v1.json'), 'sha256': '78696d17457ba76536ddc52b22a3b15e6abbaae0cb4b7449a10ae3aaabf7cd85'},
}
VIEWS = ['p0_only', 'current_only', 'current_plus_p0']
PHASES = ['all_common', 'tail20', 'complement']
COHORTS = ['all', 'original_good', 'original_wrong']
LABELS = ['all', 'negative', 'hold', 'positive']
EVENTS = ['candidate_harmed', 'new_harm', 'lost_correction']
PREDICTIONS = LABELS.copy()
COLUMNS = {
    'direction': ['task', 'seed', 'arm', 'fold', 'source_row', 'demo', 'action_step', 'tail20', 'y2', 'p0', 'needed', 'truth', 'view', 'predicted'],
    'continuous': ['task', 'seed', 'arm', 'source_row', 'probe_fold', 'demo', 'action_step', 'target', 'original', 'baseline', 'candidate'],
}
PARITY_FIELDS = ['n', 'correct_n', 'negative_n', 'hold_n', 'positive_n'] + [a + '_as_' + b for a in LABELS[1:] for b in LABELS[1:]]
FILES = {
    'analysis': ROOT / 'experiments/analyze_libero_task7_direction_harm_alignment.py',
    'audit': ROOT / 'experiments/audit_libero_task7_direction_harm_alignment.py',
    'guard': ROOT / 'experiments/libero_task7_direction_harm_alignment_access.py',
    'metadata': Path(__file__).resolve(),
    'tests': ROOT / 'tests/test_libero_task7_direction_harm_alignment.py',
    'plan': ROOT / 'docs/libero-task7-direction-harm-alignment-plan-2026-11-09.md',
    'contract': ROOT / 'docs/libero-task7-direction-harm-alignment-execution-contract-2026-11-09.md',
}
SCOPE = {
    'tasks': [7], 'seeds': p.SEEDS, 'feature_source_arms': p.ARMS,
    'original_and_auxiliary_endpoints': 500, 'future_stabilization_steps': 5, 'loss_unchanged': True,
    'source': 'meta_support_demos0..29;all_common_tge5;encoder_has_seen_all30;auxiliary3fold_OOF',
    'new_optimizer_Policy_CNN_environment_initial_state_or_neighbor_calls': 0,
    'NPZ_NPY_head_checkpoint_HDF5_RGB_proprio_private_or_other_CSV_reads_including_SHA': 0,
    'source_meta_query_validation_heldout_qualification_reserved_final_access': False,
    'G_failed_H_unrun_unchanged': True, 'all_prior_joint_cues_false_unchanged': True,
    'Policy_SPT_Knowledge_Module_changes': False,
}
DESIGN = {
    'question': 'whether_saved_direction_misclassification_in_hold_or_tail_coincides_with_existing_continuous_action_harm',
    'join': 'exact_task_seed_arm_source_row_demo_step_fold;full6x4068_support;3views_each;float32_target_p0_bit_exact',
    'required_label': 'r=y2-p0_2;negative_r<=-.025;positive_r>=.025;otherwise_hold;inclusive',
    'primary_view': 'current_plus_p0', 'other_views': 'fixed_p0_only_current_only_descriptive_controls;no_new_readout',
    'class_error': 'saved_prediction_not_equal_required_label;hold_error_is_false_activation',
    'continuous': 'B=lambda0_uniform_affine;C=lambda1_harm_protected_affine;saved500_OOF_actions;component_index2',
    'errors': 'e0=(p0-y)^2;eB=(B-y)^2;eC=(C-y)^2;hB=max(eB-e0,0);hC=max(eC-e0,0)',
    'candidate_harmed': 'eC>e0;strict;not_action_sign_error',
    'new_harm': 'eB<=e0_and_eC>e0;strict',
    'lost_correction': 'abs(y)>=.025_and_sign(y)*p0<=0_and_sign(y)*B>0_and_sign(y)*C<=0',
    'cohorts': 'all;original_good:abs(y)>=.025_and_e0<y^2;original_wrong:abs(y)>=.025_and_sign(y)*p0<=0',
    'phases': 'tge5_common;last20_index_window;common_complement;not_physical_phases',
    'groups': 'all6_endpoints_x3views_x3phases_x3cohorts_x4required_classes_x4prediction_classes;retain_empty_cells',
    'association': 'within_identical_endpoint_phase_cohort_required_class;P(event|class_error)-P(event|class_correct)',
    'metrics': 'full2x2_counts;event_rates_RD_RR;error_exposure_and_event_overlap;enrichment;missing_or_zero_reference_RR_null_with_reason',
    'demo': 'all30;paired_equal_demo_rates_only_demos_with_both_error_and_correct_rows;record_support_coverage_and_nulls;frame_pooled_primary',
    'continuous_harm_geometry': 'among_strict_harmed:away_if_delta*r<0;at_target_move_if_r==0;overshoot_if_same_sign_abs_delta>2abs_r;otherwise_roundoff_boundary',
    'roundoff_gate': 'boundary_harm_above1e-12_stops;no_relabeling_harm_or_class_errors',
    'parity': '14_saved_confusion_counts_x54_view_phase_models;4_existing_harm_retention_counts_x18_model_phases=828_exact',
    'summary': 'all3raw_seed_values_per_arm;mean_median_only_all3_nonnull;positive_negative_tie_null_counts;no_seed_or_cell_selection',
    'interpretation': 'posthoc_same_frame_association;NN_output_was_not_used_to_generate_B_or_C;no_counterfactual_causal_or_closed_loop_claim',
    'structural_limit': 'lost_correction_cannot_occur_in_required_hold_on_material_original_wrong;retain_structural_zeros_not_as_protection_evidence',
    'decision': 'descriptive_only;no_pvalues_training_acceptance_or_new_jointcue;missing_reference_is_inconclusive_not_zero_risk;end_after_frozen_budget',
}
EXECUTION = {
    'runtime': {'python': '3.10.22', 'numpy': '1.22.4'},
    'environment': {'CUDA_VISIBLE_DEVICES': '', 'PYTHONHASHSEED': '0', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
    'device': 'CPU', 'seconds_per_stage_max': 180, 'overall_seconds_max': 420,
    'RSS_bytes_max': 536870912, 'output_CSV_bytes_max': 134217728,
    'access': 'each_stage_two_CSV_once;only_fixed14_and11_columns_numeric;SHA_before_pass_and_after_stage;no_original_arrays_or_other_CSV',
    'audit': 'separate_process_no_primary_numerical_import;independent_join_labels_actions_geometry_counts_metrics_demo_summary_and_parent_parity',
    'float_absolute_tolerance': 1e-10, 'relative_tolerance': 0,
    'IDs_signs_counts_saved_float32_join_tolerance': 0, 'technical_retries': 0, 'overwrite': False,
    'binding_required_before_source_CSV_SHA_or_projection': True,
}


def budget():
    return {'CSV_passes_per_stage': 2, 'rows_per_stage': {'direction': 73224, 'continuous': 24408},
        'projected_cells_per_stage': {'direction': 1025136, 'continuous': 268488},
        'lexical_cells_per_stage': 2562840, 'total_CSV_passes': 4, 'total_projected_cells': 2587248,
        'total_lexical_rows': 195264, 'total_lexical_cells': 5125680, 'source_SHA_checks_per_stage': 4,
        'joined_unique_frames_per_stage': 24408, 'direction_frame_decisions_per_stage': 73224,
        'parent_exact_parity_per_stage': 828,
        'tables': {'joined-frames': 24408, 'groups': 2592, 'associations': 1944, 'associations-demo': 58320, 'parity': 828},
        'output_CSV_rows_per_stage': 88092, 'seed_summary_records': 648,
        'source_array_head_checkpoint_new_training_forward_environment_sealed_calls': 0}


def select_inputs(parents):
    n = parents['direction_result']; c = parents['continuous_result']; s = parents['direction_science']
    models = [{k: r[k] for k in ('task', 'seed', 'arm', 'checkpoint')} for r in n['inputs']['models']]
    p.require(models == c['inputs']['endpoints'] and n['inputs']['action_lengths'] == c['inputs']['action_lengths'] == s['inputs']['action_lengths'], 'saved endpoints/roles/lengths differ')
    p.require(n['science_config'] == PARENTS['direction_science'] and n['parents']['geometry_result'] == PARENTS['continuous_result'], 'common endpoint lineage differs')
    for key in ('direction_audit', 'summary_audit', 'continuous_audit'):
        p.require(parents[key]['status'].startswith('passed_'), 'parent audit incomplete')
    for key in ('direction_completion', 'continuous_completion'):
        p.require(parents[key]['status'].startswith('complete_independently_audited'), 'parent completion missing')
    p.require(parents['summary_audit']['report'] == PARENTS['direction_summary'], 'summary lineage differs')
    return {'direction': n['tables']['frames'], 'continuous': c['tables']['paired-frames'],
        'endpoints': models, 'action_lengths': n['inputs']['action_lengths'],
        'classification_parity': [r for r in parents['direction_summary']['model_metrics'] if r['readout'] in VIEWS],
        'continuous_parity': c['inputs']['parent_parity']}


def create():
    parents = {name: p.load_pin(ref) for name, ref in PARENTS.items()}
    inputs = select_inputs(parents)
    p.require(inputs['direction']['rows'] == 73224 and len(inputs['direction']['columns']) == 21 and inputs['continuous']['rows'] == 24408 and len(inputs['continuous']['columns']) == 42, 'two canonical source tables required')
    return copy.deepcopy({'kind': 'task7_direction_harm_alignment_readonly', 'version': 1, 'formal_result': False,
        'status': 'frozen_execution_binding_required', 'parents': PARENTS, 'scope': SCOPE, 'design': DESIGN,
        'execution': EXECUTION, 'columns': COLUMNS, 'budget': budget(), 'inputs': inputs,
        'metadata': p.pin(FILES['metadata']), 'plan': p.pin(FILES['plan']), 'output': OUTPUT, 'artifacts': ARTIFACTS})


def validate(path=DEFAULT_CONFIG):
    cfg = json.loads(Path(path).read_text())
    p.require(json.dumps(cfg, sort_keys=True, allow_nan=False) == json.dumps(create(), sort_keys=True, allow_nan=False), 'frozen alignment protocol/provenance differs')
    return cfg


def require_binding(config=DEFAULT_CONFIG, binding=DEFAULT_BINDING):
    b = json.loads(Path(binding).read_text()); cfg = validate(config)
    p.require(b['status'] == 'bound_before_payload' and b['before_research_payload'] is True and b['science_config'] == p.pin(config)
              and b['budget'] == cfg['budget'] and b['CPU_environment'] == EXECUTION['environment']
              and b['future_output'] == OUTPUT and b['future_artifacts'] == ARTIFACTS, 'execution unbound or changed')
    refs = b['dependencies']; paths = [p.resolve(r['path']).resolve() for r in refs]
    p.require(len(paths) == len(set(paths)) and set(FILES.values()).issubset(paths), 'code coverage differs')
    p.require(all(ref in refs for ref in p.load_pin(PARENTS['previous_binding'])['dependencies']), 'parent code closure missing')
    for ref in refs:
        path = p.resolve(ref['path']); p.require(path.suffix in ('.py', '.md', '.json', '.toml'), 'payload in code closure')
        p.require(p.pin(path) == ref, 'bound code changed')
    return b


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--write-freeze', action='store_true'); args = parser.parse_args()
    if args.write_freeze:
        with DEFAULT_CONFIG.open('x') as f: json.dump(create(), f, ensure_ascii=False, indent=2); f.write('\n')
    else: validate()
    print(json.dumps({'status': 'metadata_frozen_no_payload' if args.write_freeze else 'metadata_valid_no_payload'}))
