import copy
import json

import numpy as np
import pytest
import torch
from torch import nn

from experiments import analyze_libero_task4_task7_gradient_interference_diagnostic as diagnostic


@pytest.mark.parametrize("field,value", [
    ("checkpoint_optimizer_updates", 1000), ("new_optimizer_updates", 1),
    ("demonstration_role", "meta_query"), ("first_action_index", 0),
    ("new_environment_action_steps", 1), ("qualification_accessed", True),
    ("batch_windows", 2), ("device", "cuda:0"),
])
def test_scope_changes_fail_before_data_or_checkpoints(tmp_path, field, value):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["scope"][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="support-only gradient scope"):
        diagnostic.load_protocol(path)


@pytest.mark.parametrize("field,value", [("loss_denominator", "component_dimensions"),
    ("global_clip_threshold", 2.), ("negative_cosine_threshold", -.1)])
def test_metric_or_decision_change_requires_new_protocol(tmp_path, field, value):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["analysis"][field] = value
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="decomposition or decision"):
        diagnostic.load_protocol(path)


def test_runner_hash_is_checked_before_input_reads(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["execution_script_sha256"] = "0" * 64
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="runner hash"):
        diagnostic.load_protocol(path)


def test_gradient_access_rejects_query_or_non_source_before_resolver(monkeypatch):
    monkeypatch.setattr(diagnostic.expert, "authorize_role", lambda *a: pytest.fail("forbidden data reached resolver"))
    for task, role in [(4, "meta_query"), (4, "qualification"), (2, "meta_support"), (5, "meta_support")]:
        with pytest.raises(ValueError, match="support-only scope"):
            diagnostic.authorize_support({}, {}, task, role)


def test_repository_role_lock_remains_active():
    transfer = json.loads((diagnostic.ROOT / "configs/libero_spatial_transfer_split_candidate_v1.json").read_text())
    manifest = json.loads((diagnostic.ROOT / "configs/libero_spatial_dataset_manifest_v1.json").read_text())
    assert diagnostic.authorize_support(transfer, manifest, 4).demonstration_indices == tuple(range(30))
    transfer["locked_access"]["before_method_freeze"].append("source_spi.meta_support")
    with pytest.raises(PermissionError, match="locked before method freeze"):
        diagnostic.authorize_support(transfer, manifest, 4)


@pytest.mark.parametrize("length", [2, 10, 11, 12, 21, 157])
def test_windows_cover_each_t_ge_1_exactly_once_without_cross_demo_or_t0(length):
    ranges = diagnostic.window_ranges(length)
    assert [index for start, stop in ranges for index in range(start, stop)] == list(range(1, length))
    assert all(1 <= stop - start <= 10 for start, stop in ranges)


def test_padding_cannot_create_expert_data_or_a_local_transition():
    observations = {name: np.zeros((3, *shape), dtype=dtype)
                    for name, (shape, dtype) in diagnostic.expert.POLICY_OBSERVATION_SPECS.items()}
    tensors, targets, valid, local = diagnostic.pack_window(observations, np.ones((3, 7), np.float32), np.array([False, True, False]))
    assert tensors[0].shape == (1, 10, 3, 128, 128)
    assert valid.sum() == 3 and local.sum() == 1
    assert not bool(local[0, 3:].any()) and not bool(valid[0, 3:].any())
    assert torch.count_nonzero(targets[0, 3:]) == 0
    assert tensors[0].dtype == targets.dtype == torch.float64


def test_precision_v2_preserves_original_float32_normalized_pixels_exactly():
    observations = {name: np.full((1, *shape), 13 if name.endswith("_rgb") else .1, dtype=dtype)
                    for name, (shape, dtype) in diagnostic.expert.POLICY_OBSERVATION_SPECS.items()}
    tensors, *_ = diagnostic.pack_window(observations, np.zeros((1, 7), np.float32), np.zeros(1, bool))
    original = torch.tensor(13, dtype=torch.uint8).float().div(255).double()
    assert torch.equal(tensors[0][0, 0, 0, 0, 0], original)


def test_component_losses_have_original_7n_denominator_and_gradient_locality():
    prediction = torch.full((1, 4, 7), .2, requires_grad=True)
    target = torch.zeros_like(prediction)
    valid = torch.tensor([[True, True, True, False]])
    local = torch.tensor([[False, True, False, True]])
    losses = diagnostic.component_losses(prediction, target, valid, local)
    assert float(losses["translation"]) == pytest.approx(3 * .04 / 7)
    assert float(losses["gripper"]) == pytest.approx(.04 / 7)
    assert float(losses["local_open"]) == pytest.approx(.04 / 21)
    assert torch.allclose(losses["baseline"], losses["translation"] + losses["rotation"] + losses["gripper"])
    extra = torch.autograd.grad(losses["weighted2"] - losses["baseline"], prediction)[0]
    assert torch.count_nonzero(extra) == 1
    assert float(extra[0, 1, 6]) == pytest.approx(.4 / 21)


def test_empty_local_selection_is_zero_not_an_extra_gripper_weight():
    prediction = torch.full((1, 3, 7), .2, requires_grad=True)
    losses = diagnostic.component_losses(prediction, torch.zeros_like(prediction), torch.ones((1, 3), dtype=torch.bool), torch.zeros((1, 3), dtype=torch.bool))
    assert float(losses["local_open"]) == 0
    assert torch.equal(losses["baseline"], losses["weighted2"])


def test_empty_valid_or_invalid_mask_fail_closed():
    prediction = torch.zeros((1, 3, 7))
    with pytest.raises(ValueError, match="at least one"):
        diagnostic.component_losses(prediction, prediction, torch.zeros((1, 3), dtype=torch.bool), torch.zeros((1, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        diagnostic.component_losses(prediction, prediction, torch.ones((1, 3)), torch.zeros((1, 3), dtype=torch.bool))


class CoupledPolicy(nn.Module):
    """Shared scalar trunk has a known translation/gripper conflict."""
    def __init__(self):
        super().__init__()
        self.visual_encoder = nn.Linear(1, 1, bias=False)
        self.fusion = nn.Linear(1, 1, bias=False)
        self.recurrent = nn.Linear(1, 1, bias=False)
        self.action_head = nn.Linear(1, 7, bias=False)
        with torch.no_grad():
            for value in self.parameters():
                value.fill_(1.)
        self.double()

    def forward_sequence(self, rgb, *unused, hidden=None):
        values = rgb[:, :, 0, 0, 0].unsqueeze(-1)
        return self.action_head(self.recurrent(self.fusion(self.visual_encoder(values)))), None


def probe_fixture(local_nonempty=True):
    tensors = [torch.full((1, 10, 1, 1, 1), .5, dtype=torch.float64)]
    targets = torch.full((1, 10, 7), .5, dtype=torch.float64)
    targets[..., :3], targets[..., 6] = 1., 0.
    valid = torch.ones((1, 10), dtype=torch.bool)
    local = valid.clone() if local_nonempty else torch.zeros_like(valid)
    progress = {name: 0 for name in ["forward_calls_attempted", "forward_calls_completed", "forward_valid_rows", "forward_padded_rows", "autograd_calls_attempted", "autograd_calls_completed"]}
    return tensors, targets, valid, local, progress


def test_probe_detects_shared_conflict_without_assigning_grad_or_updating_weights():
    model = CoupledPolicy().eval()
    before = diagnostic.core._digest_weights(model.state_dict())
    args = probe_fixture()
    row, gradients, prediction, layout = diagnostic.probe_window(model, *args)
    assert row["pairs"]["translation_vs_local_open"]["shared"]["cosine"] == pytest.approx(-1.)
    assert row["head_disjoint_dot"] == 0.
    np.testing.assert_allclose(gradients["baseline"], gradients["translation"] + gradients["rotation"] + gradients["gripper"],
                               atol=diagnostic.ANALYSIS["gradient_identity_atol"], rtol=diagnostic.ANALYSIS["gradient_identity_rtol"])
    assert all(value.grad is None for value in model.parameters())
    assert diagnostic.core._digest_weights(model.state_dict()) == before
    repeated, again, repeated_prediction, repeated_layout = diagnostic.probe_window(model, *args)
    assert row == repeated and layout == repeated_layout
    assert np.array_equal(prediction, repeated_prediction)
    assert all(np.array_equal(gradients[name], again[name]) for name in gradients)
    assert args[-1]["autograd_calls_completed"] == 12


def test_empty_local_window_skips_two_backward_calls_and_has_null_local_metrics():
    args = probe_fixture(False)
    row, gradients, *_ = diagnostic.probe_window(CoupledPolicy(), *args)
    assert row["autograd_calls"] == args[-1]["autograd_calls_completed"] == 4
    assert all(value is None for value in row["pairs"]["translation_vs_local_open"].values())
    assert np.array_equal(gradients["weighted2"], gradients["baseline"])
    summary = diagnostic.summarize_rows([row])
    assert summary["pairs"]["translation_vs_local_open"]["shared"]["cosine"]["mean"] is None


def test_real_cnn_only_policy_has_disjoint_head_rows_and_preserves_weights():
    from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig
    torch.manual_seed(17)
    model = LiberoBCPolicy(LiberoBCPolicyConfig(cnn_channels=(2, 3, 4), spatial_pool_size=1,
        visual_embedding_dim=3, fusion_dim=5, recurrent_hidden_dim=5, memory_mode="cnn_only")).eval().double()
    before = diagnostic.core._digest_weights(model.state_dict())
    tensors = [torch.randint(0, 255, (1, 10, 3, 8, 8), dtype=torch.uint8).float().div(255).double() for _ in range(2)]
    tensors += [torch.zeros((1, 10, 7), dtype=torch.float64), torch.zeros((1, 10, 2), dtype=torch.float64)]
    target = torch.zeros((1, 10, 7), dtype=torch.float64)
    valid = torch.tensor([[True] * 3 + [False] * 7])
    local = torch.tensor([[False, True] + [False] * 8])
    progress = probe_fixture()[-1]
    row, *_ = diagnostic.probe_window(model, tensors, target, valid, local, progress)
    assert row["valid_steps"] == 3 and row["padded_steps"] == 7
    assert row["head_disjoint_dot"] == 0
    assert diagnostic.core._digest_weights(model.state_dict()) == before
    assert all(value.grad is None for value in model.parameters())


def test_zero_norm_has_null_cosine_and_never_counts_as_conflict():
    metric = diagnostic.vector_metrics(np.zeros(3, np.float32), np.ones(3, np.float32), [(0, 3)])
    assert metric["dot"] == 0 and metric["cosine"] is None
    assert diagnostic.clipping(np.zeros(3))["coefficient"] == 1.


def test_global_clipping_changes_scale_but_preserves_direction():
    vector = np.array([3., 4.])
    clipped = diagnostic.clipping(vector)
    assert clipped["raw_norm"] == 5.
    assert clipped["coefficient"] == pytest.approx(1 / (5 + 1e-6))
    assert diagnostic.vector_metrics(vector, vector * clipped["coefficient"], [(0, 2)])["cosine"] == pytest.approx(1.)


def test_gradient_algebra_gate_rejects_a_real_denominator_error():
    with pytest.raises(ValueError, match="gradient identity"):
        diagnostic.require_gradient_identity(np.ones(4), np.ones(4) * 7 / 3, "wrong_normalization")


def test_nonfinite_gradient_metrics_are_rejected():
    with pytest.raises(ValueError, match="nonfinite gradient"):
        diagnostic.vector_metrics(np.array([np.nan]), np.ones(1), [(0, 1)])


def test_no_single_seed_or_favorable_mean_can_satisfy_three_seed_median_rule():
    seeds = {str(seed): {"summary": {"pairs": {"translation_vs_local_open": {"shared": {"cosine": {"median": value}}}}}}
             for seed, value in zip(diagnostic.SEEDS, [-.8, -.6, .1])}
    result = diagnostic.classify_task(seeds)
    assert result["all_three_seed_negative_medians"] is False
    assert result["status"] == "seed_dependent_local_conflict_at_fixed_endpoint"
    assert result["closed_loop_cause_established"] is False


def test_output_guard_preserves_historical_result_before_protocol_reads(tmp_path, monkeypatch):
    output = tmp_path / "history.json"
    output.write_text("historical")
    monkeypatch.setattr(diagnostic, "load_protocol", lambda *a: pytest.fail("existing output reached inputs"))
    with pytest.raises(ValueError, match="already exists"):
        diagnostic.run_analysis(tmp_path / "unused", output, tmp_path / "archive")
    assert output.read_text() == "historical"


def test_checkpoint_matrix_rejects_an_arm_or_budget_change_before_file_reads(monkeypatch):
    base = [{"task_id": task, "training_seed": seed, "arm": "source_baseline", "optimizer_updates": 500}
            for task in diagnostic.TASKS for seed in diagnostic.SEEDS]
    monkeypatch.setattr(diagnostic, "sha256_file", lambda *a: pytest.fail("bad checkpoint reached file read"))
    for field, value in [("arm", "public_state_jitter"), ("optimizer_updates", 1000)]:
        rows = copy.deepcopy(base)
        rows[0][field] = value
        with pytest.raises(ValueError, match="only baseline500"):
            diagnostic.fixed_checkpoints({"checkpoints": rows}, {"status": "complete"})


def test_numerical_audit_refuses_unbounded_window_before_source_access(tmp_path):
    from experiments import audit_libero_gradient_decomposition_precision as audit
    config = {"check_type": "libero_gradient_decomposition_precision_audit", "scope": {"task_id": 5}}
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="bounded numerical audit scope"):
        audit.run(path, tmp_path / "result", tmp_path / "archive")
