import torch
import pytest

from experiments.run_crafter_wood3_spatial_representation_curve import _validate_matched_config
from experiments.run_crafter_survival_checkpoint_diagnostic import _depletion_pattern
from src.algorithms.spatial_crafter_policy import build_matched_spatial_policy


def _config():
    return {
        "cnn_channels": [4, 8],
        "embedding_dim": 16,
        "recurrent_hidden_dim": 6,
        "action_count": 5,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "update_epochs": 2,
    }


def test_representation_arms_share_the_same_spatial_encoder():
    config = _config()
    torch.manual_seed(17)
    cnn_only = build_matched_spatial_policy(config, "cnn_only")
    torch.manual_seed(17)
    cnn_gru = build_matched_spatial_policy(config, "cnn_gru")

    cnn_state = cnn_only.encoder.state_dict()
    gru_state = cnn_gru.encoder.state_dict()
    assert cnn_state.keys() == gru_state.keys()
    assert all(torch.equal(cnn_state[key], gru_state[key]) for key in cnn_state)


def test_cnn_only_and_gru_paths_have_matching_ppo_interfaces():
    config = _config()
    observations = torch.rand(4, 3, 64, 64)
    actions = torch.tensor([0, 1, 2, 3])
    legal_actions = (0, 1, 2, 3)
    for representation in ("cnn_only", "cnn_gru"):
        policy = build_matched_spatial_policy(config, representation)
        hidden = None
        distribution, value, next_hidden = policy.distribution_value(
            observations[0], hidden, legal_actions
        )
        assert distribution.logits.shape == (1, 5)
        assert value.shape == (1,)
        if representation == "cnn_only":
            assert next_hidden is None
        else:
            assert next_hidden.shape == (1, 1, 6)
        sequence_distribution, values = policy.sequence_logits_values(observations, legal_actions)
        assert sequence_distribution.logits.shape == (4, 5)
        assert values.shape == (4,)
        old_log_probs = sequence_distribution.log_prob(actions).detach()
        result = policy.update(
            observations,
            actions,
            old_log_probs,
            torch.ones(4),
            torch.zeros(4),
            legal_actions,
            return_metrics=True,
        )
        assert set(("loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "explained_variance")) <= result.keys()


def _matched_runner_config(hidden_dim):
    return {
        "representations": ["cnn_only", "cnn_gru"],
        "formal_result": False,
        "teacher_used": False,
        "update_mode": "episode",
        "python_hash_seed": 0,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": ":4096:8",
        "seed_set": [0],
        "replicate_seed_stride": 100,
        "total_train_steps": 10,
        "development_episodes": 1,
        "qualification_episodes": 1,
        "train_seed_base": 1000,
        "development_seed_base": 2000,
        "qualification_seed_base": 3000,
        "action_seed_base": 11000,
        "development_action_seed_base": 12000,
        "qualification_action_seed_base": 13000,
        "same_seed_manifest": True,
        "policy": {"embedding_dim": 128, "recurrent_hidden_dim": hidden_dim},
    }


def test_matched_runner_requires_equal_policy_head_widths(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    _validate_matched_config(_matched_runner_config(128))
    with pytest.raises(ValueError, match="same input width"):
        _validate_matched_config(_matched_runner_config(64))


def test_survival_diagnostic_classifies_terminal_depletion():
    row = {
        "life_summary": {
            "food": {"final": 0.0},
            "drink": {"final": 2.0},
            "energy": {"final": 0.0},
        }
    }

    assert _depletion_pattern(row) == "food+energy"
