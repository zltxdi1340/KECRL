from src.environments.discrete_resources import (
    DiscreteResourceEnvironment,
    default_resource_tasks,
    mechanism_observations,
    task_split,
)
from src.continual_learning.contracts import InMemoryContinualLearningPipeline, TaskVersionView
from src.skills.models import QualificationConfig
from experiments.train_controlled import _run_pipeline_transition
from src.skills.torch_policy import CategoricalResourcePolicy, PolicyConfig
import torch


def test_resource_tasks_expose_success_and_missing_requirement_paths():
    tasks = {task.task_id: task for task in default_resource_tasks()}
    env = DiscreteResourceEnvironment()

    missing = env.execute(tasks["craft_tool"])
    assert not missing.success
    assert missing.reason == "missing_requirement"

    gathered = env.execute(tasks["gather_wood"])
    assert gathered.success
    crafted = env.execute(tasks["craft_tool"])
    assert crafted.success
    assert crafted.consumed == {"wood": 1}
    assert crafted.produced == {"tool": 1}
    assert env.state() == {"wood": 0, "tool": 1}


def test_task_split_is_deterministic_and_roles_do_not_overlap():
    first = task_split(3)
    second = task_split(3)
    assert first == second
    role_sets = [set(values) for values in first.values()]
    assert sum(len(values) for values in role_sets) == len(set().union(*role_sets))


def test_mechanism_cases_cover_support_refutation_and_unknown():
    cases = mechanism_observations(4)
    assert [case.observation for case in cases] == [1, 0, "bottom"]
    assert [case.truth for case in cases] == [True, False, True]
    assert len({case.evidence_id for case in cases}) == len(cases)


def test_resource_task_runs_through_pipeline_with_split_feedback():
    task = {task.task_id: task for task in default_resource_tasks()}["gather_wood"]
    (task_result, transition), evidence, feedback = _run_pipeline_transition(task, {}, QualificationConfig())
    assert task_result == "completed"
    assert transition.execution_status == "completed"
    assert evidence[0].evidence_validity == "unknown"
    assert feedback and feedback[0]["execution_status"] == "completed"


def test_confirmed_knowledge_prior_changes_policy_logits_only_when_enabled():
    policy = CategoricalResourcePolicy(PolicyConfig())
    observation = torch.zeros(4)
    without = policy.action_distribution(observation, (0, 1, 2, 3)).logits
    with_prior = policy.action_distribution(observation, (0, 1, 2, 3), preferred_action=0, prior_strength=1.5).logits
    relative_without = without[0] - without[1]
    relative_with = with_prior[0] - with_prior[1]
    assert torch.isclose(relative_with - relative_without, torch.tensor(1.5))
