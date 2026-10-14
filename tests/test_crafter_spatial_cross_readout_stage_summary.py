from experiments.summarize_crafter_wood3_spatial_cross_readout import discounted_returns, episode_macro_rows


def _row(path, episode, step, wood, category, p_valid):
    return {
        "path_variant": path, "policy_variant": "source_cnn_only", "episode": episode,
        "step": step, "wood_before": wood, "category": category, "p_valid": p_valid,
        "trajectory_action_probability": p_valid, "trajectory_action_valid": False,
        "greedy_valid": False, "p_do": 0.4, "entropy": 0.8, "critic_value": 1.25,
        "discounted_mc_return": 1.0, "critic_value_abs_error": 0.25,
    }


def test_stage_and_predeath_summary_is_episode_balanced():
    predictions = [
        _row("source_cnn_only", 0, 1, 0, "turn", 0.9),
        *[_row("source_cnn_only", 1, step, 0, "turn", 0.1) for step in range(1, 10)],
        _row("source_cnn_only", 1, 9, 1, "approach", 0.5),
    ]
    metadata = {
        ("source_cnn_only", 0): {"terminal_reason": "success", "steps": 20},
        ("source_cnn_only", 1): {"terminal_reason": "death", "steps": 9},
    }
    rows = episode_macro_rows(predictions, metadata)
    turn = next(row for row in rows if row["window"] == "before_wood1" and row["category"] == "turn")
    predeath = next(row for row in rows if row["window"] == "predeath_last10" and row["category"] == "turn")
    approach = next(row for row in rows if row["window"] == "predeath_last10" and row["category"] == "approach")
    assert turn["states"] == 10 and turn["episodes_with_states"] == 2
    assert turn["episode_macro_p_valid"] == 0.5
    assert predeath["episodes_with_states"] == 1
    assert predeath["episode_macro_p_valid"] == 0.1
    assert approach["episode_macro_p_valid"] == 0.5


def test_discounted_returns_are_terminal_and_gamma_weighted():
    episodes = {"arm": [{"episode": 2, "events": [
        {"step": 1, "reward": 1.0}, {"step": 2, "reward": 2.0},
    ]}]}
    result = discounted_returns(episodes, 0.5)
    assert result[("arm", 2, 1)] == 2.0
    assert result[("arm", 2, 2)] == 2.0
