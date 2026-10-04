from experiments.prepare_dataset import build_manifest


def test_manifest_is_reproducible_and_roles_are_disjoint():
    first = build_manifest([0, 1], episodes_per_role=3)
    second = build_manifest([0, 1], episodes_per_role=3)
    assert first == second
    assert first["formal_result"] is False
    for split in first["splits"].values():
        ids = [episode for episodes in split.values() for episode in episodes]
        assert len(ids) == len(set(ids))
    for roles in first["episode_specs"].values():
        ids = [episode["episode_id"] for specs in roles.values() for episode in specs]
        assert len(ids) == len(set(ids))
        assert all(episode["role"] in roles for specs in roles.values() for episode in specs)


def test_manifest_lists_controlled_task_families():
    manifest = build_manifest([4])
    assert manifest["environment"] == "discrete_resource_v1"
    assert "craft" in manifest["skill_families"]
    assert "query" in manifest["roles"]
    assert manifest["episode_specs"]["4"]["query"][0]["target"]["name"] == "resource_at_least"
