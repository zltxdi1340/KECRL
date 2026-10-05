from experiments.train_controlled_torch_fomaml_formal_v1 import (
    _build_skill_library,
    _contract_execution_check,
    _module_reuse_evaluation,
    _review_spt,
    _task_contract,
)
from src.environments.discrete_resources import default_resource_tasks


def test_formal_runner_config_is_explicitly_diagnostic():
    import json

    config = json.load(open("configs/controlled_torch_fomaml_formal_v1.yaml", encoding="utf-8"))
    assert config["formal_result"] is False
    assert config["qualification"]["min_samples"] == 20
    assert config["spt"]["validation_batches"] == 2
    assert config["metrics"]["support_curve_checkpoints"] == [0, 50, 100, 200]


def test_formal_runner_module_exposes_spt_review_entrypoint():
    assert callable(_review_spt)


def test_contract_check_uses_controlled_transition():
    task = default_resource_tasks()[0]
    assert _contract_execution_check(task, {}) is True
    assert _contract_execution_check(default_resource_tasks()[1], {}) is False


def test_module_reuse_path_returns_reused_and_unavailable():
    tasks = {task.task_id: task for task in default_resource_tasks()[:2]}
    library = _build_skill_library(tasks, {
        "min_samples": 1, "success_threshold": 1.0, "contract_threshold": 1.0,
    }, "v1")
    _, contract, spi = _task_contract(tasks["gather_wood"], "v1")
    library.qualify(spi, "test-policy", 1, 1, 1)
    query = {
        "gather_wood": [{"episode_id": "q0", "initial_resources": {}}],
        "craft_tool": [{"episode_id": "q1", "initial_resources": {"wood": 1}}],
    }
    result = _module_reuse_evaluation(library, tasks, query, "v1")
    assert result["reused"] == 1
    assert result["unavailable"] == 1
    assert result["completed"] == 1


def test_support_curve_records_task_strata():
    import torch
    from experiments.train_controlled_torch_fomaml_formal_v1 import _support_curve
    from src.skills.context_fomaml import ContextConditionedPolicyFOMAML
    from src.skills.context_policy import ContextConditionedPolicyInitializer
    from src.skills.torch_fomaml import TorchFOMAMLConfig
    from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig

    tasks = {task.task_id: task for task in default_resource_tasks()}
    pools = {task_id: [{"episode_id": f"{task_id}:0", "initial_resources": {}}]
             for task_id in tasks}
    policy = CategoricalResourcePolicy(PolicyConfig(hidden_dim=4))
    initializer = ContextConditionedPolicyInitializer(policy, len(tasks))
    learner = ContextConditionedPolicyFOMAML(initializer, TorchFOMAMLConfig())
    config = {"metrics": {"support_curve_checkpoints": [0], "query_success_threshold": 0.8},
              "eval_query_episodes_per_task": 1}
    rows = _support_curve(initializer, learner, tasks, pools, pools, config,
                          torch.device("cpu"), 2, {}, 0.0, 0)
    assert set(rows[0]["task_query_success_rates"]) == set(tasks)
