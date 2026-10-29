import json

import pytest
import torch

from experiments import run_libero_bc_causal_alignment_diagnostic as causal
from experiments import run_libero_runtime_native_prefix_task4_task7_module_freeze_readout_diagnostic as diagnostic
from src.algorithms.libero_bc_policy import LiberoBCPolicy, LiberoBCPolicyConfig


def _small_model():
    return LiberoBCPolicy(
        LiberoBCPolicyConfig(
            cnn_channels=(4, 8, 8),
            spatial_pool_size=2,
            visual_embedding_dim=8,
            fusion_dim=12,
            recurrent_hidden_dim=10,
        )
    )


def test_config_pins_four_module_freeze_scopes():
    config, cross, coverage, failure, prefix, *_ = diagnostic._load_protocol(
        diagnostic.DEFAULT_CONFIG,
        diagnostic.DEFAULT_CROSS_RESULT,
        diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
        diagnostic.DEFAULT_FAILURE_RESULT,
        diagnostic.DEFAULT_COVERAGE_RESULT,
        diagnostic.DEFAULT_PREFIX_RESULT,
    )
    assert config["readout_arms"] == [
        "full_policy",
        "freeze_visual",
        "freeze_visual_fusion",
        "action_head_only",
    ]
    assert cross["case_count"] == 9
    assert coverage["status"] == failure["status"] == prefix["status"] == "complete"


def test_config_rejects_causal_scope_interpretation(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["interpretation_boundary"]["frozen_scope_marginal_is_causal"] = True
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="interpretation boundary"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_CROSS_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_FAILURE_RESULT,
            diagnostic.DEFAULT_COVERAGE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_config_rejects_source_hash_drift(tmp_path):
    config = json.loads(diagnostic.DEFAULT_CONFIG.read_text())
    config["source_prefix_result_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="pinned prefix_result differs"):
        diagnostic._load_protocol(
            path,
            diagnostic.DEFAULT_CROSS_RESULT,
            diagnostic.DEFAULT_OBSERVATION_ARCHIVE,
            diagnostic.DEFAULT_FAILURE_RESULT,
            diagnostic.DEFAULT_COVERAGE_RESULT,
            diagnostic.DEFAULT_PREFIX_RESULT,
        )


def test_freeze_scope_parameter_names_are_nested():
    expected = {
        "full": lambda name: True,
        "freeze_visual": lambda name: not name.startswith("visual_encoder."),
        "freeze_visual_fusion": lambda name: not (
            name.startswith("visual_encoder.") or name.startswith("fusion.")
        ),
        "action_head_only": lambda name: name.startswith("action_head."),
    }
    for scope, predicate in expected.items():
        model = _small_model()
        names = causal._configure_trainable_scope(model, scope)
        assert names
        assert set(names) == {
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        }
        assert all(predicate(name) for name in names)


def test_unknown_freeze_scope_fails_closed():
    with pytest.raises(ValueError, match="unknown trainable scope"):
        causal._configure_trainable_scope(_small_model(), "freeze_cnn_only")
