import pytest

from experiments.audit_crafter_material_gather_role_geometry import _total_variation, analyze


def test_geometry_comparison_distinguishes_proportions_from_pool_sizes():
    assert _total_variation({"near": 1, "far": 1}, {"near": 5, "far": 5}) == 0
    assert _total_variation({"near": 1}, {"far": 1}) == 1
    assert _total_variation({"near": 1, "far": 1}, {"far": 1}) == 0.5
    with pytest.raises(ValueError, match="must not be empty"):
        _total_variation({}, {"far": 1})


def test_frozen_v2_partitions_reveal_geometry_shift_without_policy_evaluation():
    result = analyze()
    assert result["formal_result"] is False
    roles = result["roles"]
    evaluation = roles["final_evaluation"]["materials"]["tree"]
    assert evaluation["position_counts"] == {"30,29": 2500}
    assert evaluation["mean_manhattan_distance"] == 5
    source_target = next(pair for pair in result["pairwise"]
                         if (pair["left"], pair["right"]) == ("source_support", "target_support"))
    assert source_target["materials"]["tree"]["position_support_intersection"] == 0
    assert source_target["materials"]["tree"]["position_total_variation"] == 1
