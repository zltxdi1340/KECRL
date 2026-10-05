import json

from experiments.audit_formal_freeze import audit


def test_formal_freeze_config_is_machine_ready_but_training_blocked():
    result = audit()
    assert result["config_ready_for_runner_implementation"] is True
    assert result["ready_for_formal_training"] is False
    assert all(result["checks"].values())


def test_formal_freeze_rejects_role_overlap(tmp_path):
    manifest = {
        "formal_result": False,
        "seeds": [0, 1, 2, 3, 4],
        "episodes_per_role": 1,
        "task_ids": ["task"],
        "splits": {
            str(seed): {
                role: [{"episode_id": f"{seed}:{role}", "task_id": "task"}]
                for role in ["train", "support", "query", "qualification", "spt_validation"]
            }
            for seed in [0, 1, 2, 3, 4]
        },
    }
    manifest["splits"]["0"]["query"][0]["episode_id"] = "0:train"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = json.load(open("configs/controlled_torch_fomaml_formal_freeze_v1.yaml", encoding="utf-8"))
    config["environment"]["manifest"] = str(manifest_path)
    config["environment"]["task_ids"] = ["task"]
    config["episodes_per_role"] = 1
    config["qualification"]["episodes_per_task"] = 1
    config["qualification"]["min_samples_per_task"] = 1
    config["spt_acceptance"]["validation_episodes_per_task_per_batch"] = 0
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    result = audit(str(config_path))
    assert result["checks"]["role_sets_disjoint"] is False
    assert result["config_ready_for_runner_implementation"] is False
