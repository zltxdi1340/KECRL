from dataclasses import asdict

from src.skills.contracts import ImplementationContract, TransitionRequest
from src.skills.evolution import (
    ContextConditionedFOMAML,
    FOMAMLConfig,
    SPIEpisode,
    SPTVersionManager,
)
from src.skills.models import (
    InMemoryQualifiedSkillLibrary,
    QualificationConfig,
    SPI,
    SPT,
)


def episode(target: float) -> SPIEpisode:
    return SPIEpisode(
        context=(1.0,),
        support_x=((1.0,),),
        support_y=(target,),
        query_x=((1.0,),),
        query_y=(target,),
    )


def active_spt() -> SPT:
    return SPT("spt:test", "test", "v1", {"family": "test"}, {"parameters": (0.0, 0.0)})


def test_candidate_is_accepted_when_query_loss_improves():
    learner = ContextConditionedFOMAML(1, FOMAMLConfig(meta_lr=0.01))
    candidate = learner.propose_candidate(
        "test",
        "v1",
        (episode(2.0),),
        candidate_version="v2",
        comparison_dataset="split:a",
    )
    manager = SPTVersionManager(active_spt())
    decision = manager.review(candidate)

    assert candidate.query_loss < candidate.baseline_query_loss
    assert decision.decision == "accepted"
    assert decision.acceptance_reason == "query_loss_improved"
    assert decision.comparison_dataset == "split:a"
    assert manager.active_spt.version == "v2"
    assert manager.previous_stable_version == "v1"


def test_candidate_is_rejected_without_query_loss_improvement():
    learner = ContextConditionedFOMAML(1, FOMAMLConfig(meta_lr=0.0))
    candidate = learner.propose_candidate(
        "test", "v1", (episode(2.0),), candidate_version="v2"
    )
    manager = SPTVersionManager(active_spt())
    decision = manager.review(candidate)

    assert decision.decision == "rejected"
    assert decision.rejection_reason == "insufficient_improvement"
    assert manager.active_spt.version == "v1"
    assert manager.previous_stable is None


def test_insufficient_evidence_keeps_active_spt():
    learner = ContextConditionedFOMAML(1, FOMAMLConfig(meta_lr=0.01))
    candidate = learner.propose_candidate(
        "test",
        "v1",
        (episode(2.0),),
        candidate_version="v2",
        comparison_dataset="split:small",
        min_evidence=2,
    )
    manager = SPTVersionManager(active_spt())
    decision = manager.review(candidate)

    assert decision.decision == "inconclusive"
    assert decision.inconclusive_reason == "insufficient_evidence"
    assert manager.active_spt.version == "v1"
    assert manager.candidate_spt.version == "v2"


def test_existing_spi_query_regression_rejects_candidate():
    learner = ContextConditionedFOMAML(1, FOMAMLConfig(meta_lr=0.01))
    candidate = learner.propose_candidate(
        "test",
        "v1",
        (episode(2.0),),
        candidate_version="v2",
        existing_spi_episodes={"spi:old": episode(-2.0)},
        max_existing_spi_regression=0.0,
    )
    manager = SPTVersionManager(active_spt())
    decision = manager.review(candidate)

    assert candidate.query_loss < candidate.baseline_query_loss
    assert decision.decision == "rejected"
    assert decision.rejection_reason == "existing_spi_regression"
    assert manager.active_spt.version == "v1"


def test_spt_update_does_not_rewrite_existing_module():
    spt = active_spt()
    target = {"name": "goal"}
    scope = {"environment": "controlled"}
    contract = ImplementationContract((), (), (target,), {}, {}, {}, scope)
    request = TransitionRequest((), target, {}, scope, {})
    spi = SPI("spi:test", "test", spt.spt_id, spt.version, {}, request, contract, {}, {}, {})
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig())
    module = library.qualify(spi, "policy:test", 1, 1, 1)
    module_before = asdict(module)

    learner = ContextConditionedFOMAML(1, FOMAMLConfig(meta_lr=0.01))
    candidate = learner.propose_candidate(
        "test", "v1", (episode(2.0),), candidate_version="v2"
    )
    manager = SPTVersionManager(spt)
    manager.review(candidate)

    assert asdict(module) == module_before
    assert module.metadata["spt_version"] == "v1"
    assert manager.active_spt.version == "v2"


def test_support_and_query_roles_are_not_mixed():
    learner = ContextConditionedFOMAML(1, FOMAMLConfig())
    first = SPIEpisode((1.0,), ((1.0,),), (2.0,), ((1.0,),), (100.0,))
    second = SPIEpisode((1.0,), ((1.0,),), (2.0,), ((1.0,),), (-100.0,))
    different_support = SPIEpisode((1.0,), ((1.0,),), (-2.0,), ((1.0,),), (100.0,))

    assert learner._adapt(first) == learner._adapt(second)
    assert learner.query_loss(first) != learner.query_loss(different_support)
