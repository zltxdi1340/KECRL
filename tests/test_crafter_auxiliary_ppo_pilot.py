import json
import inspect
from pathlib import Path

import torch
import pytest

from experiments.run_crafter_auxiliary_ppo_pilot import PPOCrafterPolicy, _features, _inventory_features
from experiments.run_crafter_auxiliary_ppo_independent import _validate_role_seeds
from experiments.run_crafter_rgb_oracle_imitation_diagnostic import RGBActionClassifier, RecordingReferencePlanner
from experiments.run_crafter_wood3_spatial_gru_pilot import SpatialGRUPolicy
from experiments.run_crafter_wood3_spatial_gru_pilot import run as run_gru
from experiments.run_crafter_wood3_paired_budget_pilot import _run_seed as run_budget_seed


def test_ppo_pilot_is_non_formal_and_uses_exact_unit_target():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["task"] == {"task_id": "gather_wood_1", "item": "wood", "threshold": 1}
    assert config["policy"]["gae_lambda"] == 0.95


def test_ppo_80_budget_keeps_independent_qualification():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_pilot_80_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["train_episodes"] == 80
    assert config["qualification_episodes"] == 20


def test_goal_action_allowlist_masks_only_diagnostic_actions():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["action_allowlist"] == [0, 1, 2, 3, 4, 5, 6]
    policy = PPOCrafterPolicy(config["policy"])
    distribution, _ = policy.distribution_value(torch.zeros(192), config["action_allowlist"])
    assert torch.equal(torch.nonzero(distribution.probs > 0).flatten(), torch.tensor(config["action_allowlist"]))


def test_temporal_pilot_uses_four_frame_feature_contract():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_temporal_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["frame_stack"] == 4
    observations = tuple(torch.zeros(64, 64, 3, dtype=torch.uint8).numpy() for _ in range(4))
    assert _features(observations, torch.device("cpu")).shape == (768,)


def test_cnn_pilot_preserves_spatial_rgb_contract():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_cnn_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    policy = PPOCrafterPolicy(config["policy"])
    observation = torch.zeros(3, 64, 64)
    distribution, value = policy.distribution_value(observation, config["action_allowlist"])
    assert distribution.probs.shape == (17,)
    assert value.shape == ()


def test_progress_pilot_keeps_training_only_inventory_bonus():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_progress_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["progress_bonus"] == 0.5
    assert config["action_allowlist"] == [0, 1, 2, 3, 4, 5, 6]


def test_inventory_observability_pilot_preserves_unknown_mask():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_inventory_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["policy"]["observation_dim"] == 194
    unknown = _inventory_features(None, ("wood",), torch.device("cpu"))
    known = _inventory_features({"wood": 1}, ("wood",), torch.device("cpu"))
    assert torch.equal(unknown, torch.tensor([0.0, 0.0]))
    assert torch.allclose(known, torch.tensor([1.0 / 9.0, 1.0]))


def test_entropy_annealing_pilot_has_explicit_schedule():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood1_ppo_goal_actions_entropy_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["policy"]["entropy_coef_start"] > config["policy"]["entropy_coef_end"]
    assert config["action_allowlist"] == [0, 1, 2, 3, 4, 5, 6]


def test_oracle_imitation_diagnostic_is_non_formal_and_split():
    config = json.loads(Path("configs/crafter_rgb_oracle_imitation_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert set(config["teacher_seeds"]).isdisjoint(config["evaluation_seeds"])
    assert config["action_allowlist"] == [0, 1, 2, 3, 4, 5, 6]


def test_extended_oracle_imitation_keeps_teacher_eval_disjoint():
    config = json.loads(Path("configs/crafter_rgb_oracle_imitation_extended_diagnostic_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert len(config["teacher_seeds"]) == 100
    assert set(config["teacher_seeds"]).isdisjoint(config["evaluation_seeds"])


def test_spatial_oracle_imitation_preserves_rgb_and_reports_balanced_loss():
    config = json.loads(Path("configs/crafter_rgb_oracle_imitation_spatial_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["policy"]["encoder"] == "spatial_cnn"
    assert config["policy"]["action_loss"] == "inverse_sqrt"
    policy = RGBActionClassifier(192, 17, 64, encoder="spatial_cnn")
    assert policy.logits(torch.zeros(3, 64, 64)).shape == (17,)
    assert policy.logits(torch.zeros(2, 3, 64, 64)).shape == (2, 17)


def test_oracle_public_teacher_mode_is_explicit():
    assert "public_action_consistent" in inspect.signature(
        RecordingReferencePlanner
    ).parameters


def test_collect_wood_fomaml_pilot_excludes_unresolved_prerequisite_tasks():
    config = json.loads(Path("configs/crafter_collect_wood_fomaml_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert {task["target"]["item"] for task in config["tasks"]} == {"wood"}
    assert set(config["training_seeds"]["support"]).isdisjoint(config["training_seeds"]["query"])
    assert set(config["evaluation"]["support"]).isdisjoint(config["evaluation"]["query"])


def test_wood3_ppo_independent_pilot_matches_first_v2_boundary():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood3_ppo_goal_actions_pilot_v1.yaml").read_text())
    assert config["formal_result"] is False
    assert config["seed_set"] == [0, 1, 2, 3, 4]
    assert config["task"] == {"task_id": "gather_wood_3", "item": "wood", "threshold": 3}
    assert config["action_allowlist"] == [0, 1, 2, 3, 4, 5, 6]
    assert config["teacher_used"] is False
    assert config["policy_updated"] is True
    _validate_role_seeds(config, len(config["seed_set"]))


def test_wood3_ppo_horizon_candidate_declares_exploratory_budget():
    base = json.loads(Path("configs/crafter_auxiliary_gather_wood3_ppo_goal_actions_pilot_v1.yaml").read_text())
    horizon = json.loads(Path("configs/crafter_auxiliary_gather_wood3_ppo_goal_actions_horizon512_pilot_v1.yaml").read_text())
    assert horizon["formal_result"] is False
    assert horizon["max_steps"] == 512
    assert horizon["task"] == base["task"]
    assert horizon["action_allowlist"] == base["action_allowlist"]
    assert horizon["policy"] == base["policy"]
    assert horizon["train_episodes"] == 40
    assert base["train_episodes"] == 80
    assert horizon["train_seed_base"] != base["train_seed_base"]
    assert horizon["policy_updated"] is True
    _validate_role_seeds(horizon, len(horizon["seed_set"]))


def test_independent_ppo_rejects_overlapping_role_and_replica_seeds():
    config = json.loads(Path("configs/crafter_auxiliary_gather_wood3_ppo_goal_actions_pilot_v1.yaml").read_text())
    config["qualification_seed_base"] = config["train_seed_base"] + 10
    with pytest.raises(ValueError, match="overlap"):
        _validate_role_seeds(config, 5)
    config["qualification_seed_base"] = 850000
    config["replicate_seed_stride"] = 10
    with pytest.raises(ValueError, match="overlap"):
        _validate_role_seeds(config, 5)


def test_wood3_budget_pair_and_spatial_gru_pilots_are_non_formal_and_disjoint():
    budget = json.loads(Path("configs/crafter_wood3_paired_budget_pilot_v1.yaml").read_text())
    gru = json.loads(Path("configs/crafter_wood3_spatial_gru_pilot_v1.yaml").read_text())
    assert budget["formal_result"] is False
    assert budget["baseline_train_episodes"] < budget["extended_train_episodes"]
    assert budget["device"] == "cpu"
    assert "continuation" in budget["pilot_note"]
    assert gru["formal_result"] is False
    assert gru["teacher_used"] is False
    assert gru["task"] == {"task_id": "gather_wood_3", "item": "wood", "threshold": 3}
    _validate_role_seeds(gru, len(gru["seed_set"]))
    policy = SpatialGRUPolicy(gru["policy"])
    distribution, value, hidden = policy.distribution_value(
        torch.zeros(3, 64, 64), None, gru["action_allowlist"]
    )
    assert distribution.probs.shape == (17,)
    assert value.shape == ()
    assert hidden.shape == (1, 1, gru["policy"]["recurrent_hidden_dim"])


def test_cuda_wood3_configs_keep_budget_task_and_role_boundaries():
    budget = json.loads(Path("configs/crafter_wood3_budget_continuation_cuda_pilot_v1.yaml").read_text())
    gru = json.loads(Path("configs/crafter_wood3_spatial_gru_cuda_pilot_v1.yaml").read_text())
    assert budget["device"] == gru["device"] == "cuda"
    assert budget["formal_result"] is gru["formal_result"] is False
    assert budget["extended_train_episodes"] == gru["train_episodes"] == 160
    assert budget["max_steps"] == gru["max_steps"] == 256
    assert budget["seed_set"] == gru["seed_set"] == [0, 1, 2, 3, 4]
    assert gru["teacher_used"] is False
    _validate_role_seeds(gru, len(gru["seed_set"]))


def test_gru_sequence_replay_matches_stepwise_hidden_state():
    config = json.loads(Path("configs/crafter_wood3_spatial_gru_pilot_v1.yaml").read_text())
    policy = SpatialGRUPolicy(config["policy"])
    observations = torch.rand(4, 3, 64, 64)
    hidden = None
    probabilities, values = [], []
    with torch.no_grad():
        for observation in observations:
            distribution, value, hidden = policy.distribution_value(
                observation, hidden, config["action_allowlist"]
            )
            probabilities.append(distribution.probs)
            values.append(value)
        replay, replay_values = policy.sequence_logits_values(observations, config["action_allowlist"])
    assert torch.allclose(replay.probs, torch.stack(probabilities), atol=1e-6)
    assert torch.allclose(replay_values, torch.stack(values), atol=1e-6)


def test_wood3_runners_reject_unavailable_cuda_before_rollout(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = json.loads(Path("configs/crafter_wood3_spatial_gru_cuda_pilot_v1.yaml").read_text())
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        run_gru(str(path), str(tmp_path / "gru"))
    with pytest.raises(RuntimeError, match="refusing silent CPU fallback"):
        run_budget_seed(config, tmp_path / "budget")
    assert not (tmp_path / "gru").exists()
    assert not (tmp_path / "budget").exists()
