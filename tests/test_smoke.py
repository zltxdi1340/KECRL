import json
from experiments.run import run
from src.continual_learning.contracts import InMemoryContinualLearningPipeline
from src.knowledge.contracts import Mechanism, InMemoryKnowledgeBank
from src.knowledge.evolution import KnowledgeEvolution
from src.skills.contracts import ImplementationContract, TransitionRequest, ImplementationResponse
from src.skills.models import SPT, SPI, InMemoryQualifiedSkillLibrary, QualificationConfig
from src.skills.evolution import ContextConditionedFOMAML, FOMAMLConfig, SPIEpisode
def test_cpu_smoke(tmp_path):
    result=run("configs/smoke_cpu.yaml")
    assert result["task_result"] == "completed"
    assert result["knowledge_feedback"] and result["skill_feedback"]
    assert json.loads(open("results/smoke/result.json", encoding="utf-8").read())["task_result"] == "completed"

def test_cuda_request_does_not_fallback_when_unavailable():
    try:
        InMemoryContinualLearningPipeline.resolve_device("cuda", cuda_available=False)
    except RuntimeError:
        return
    raise AssertionError("cuda request silently fell back to CPU")

def test_knowledge_filters_confirmed_scope_and_deduplicates():
    target = {"name": "goal"}; scope = {"environment": "controlled"}
    mechanisms = [
        Mechanism("confirmed", (), (), target, scope, cognitive_status="confirmed"),
        Mechanism("testing", (), (), target, scope, cognitive_status="testing"),
        Mechanism("other-scope", (), (), target, {"environment": "other"}, cognitive_status="confirmed"),
    ]
    found = InMemoryKnowledgeBank(mechanisms).retrieve_mechanisms(target, (), scope)
    assert [m.id for m in found] == ["confirmed"]
    evo = KnowledgeEvolution({"n_min": 1, "tau_confirm": .5, "tau_reject": .2, "budget": 2})
    evidence = {"source": "controlled", "id": 1}
    evo.record("p", 1, evidence)
    evo.record("p", 1, evidence)
    assert evo.propositions["p"].support == 1
    assert evo.budget_remaining == 1

def test_mechanism_requires_all_claims_and_retests_after_refutation():
    evo = KnowledgeEvolution({"n_min": 2, "tau_confirm": .4, "tau_reject": .2, "confidence": .5, "budget": 20})
    evo.register_mechanism("m", ("reachability", "parent"))
    assert evo.assess_mechanism("m")["status"] == "candidate"
    evo.record("reachability", 1, {"id": "r1"})
    evo.record("reachability", 1, {"id": "r2"})
    assert evo.assess_mechanism("m")["status"] == "testing"
    evo.record("parent", 1, {"id": "p1"})
    evo.record("parent", 1, {"id": "p2"})
    assert evo.assess_mechanism("m")["status"] == "confirmed"
    for index in range(6):
        evo.record("parent", 0, {"id": f"counter-{index}"})
    assert evo.assess_mechanism("m")["status"] == "testing"

def test_skill_reuse_and_unavailable_paths():
    target = {"name": "goal"}; scope = {"environment": "controlled"}
    contract = ImplementationContract(({"name": "start"},), (), (target,), {}, {}, {}, scope)
    request = TransitionRequest(({"name": "start"},), target, {}, scope, {})
    spt = SPT("spt:test", "test", "v1", {})
    library = InMemoryQualifiedSkillLibrary(spt, QualificationConfig())
    spi = SPI("spi:test", "test", spt.spt_id, spt.version, {}, request, contract, {}, {}, {})
    module = library.qualify(spi, "policy:test", 1, 1, 1)
    response = library.request_implementation(request)
    assert module is not None and response.status == "reused_module"
    incompatible = TransitionRequest(({"name": "other"},), target, {}, scope, {})
    assert library.request_implementation(incompatible).status == "unavailable"

def test_fomaml_support_query_and_candidate_decision():
    episode = SPIEpisode(
        context=(1.0,), support_x=((1.0,),), support_y=(2.0,),
        query_x=((1.0,),), query_y=(2.0,),
    )
    learner = ContextConditionedFOMAML(1, FOMAMLConfig(inner_lr=.1, meta_lr=.01, inner_steps=1, device="cpu"))
    before = learner.query_loss(episode)
    candidate = learner.propose_candidate("test", "v1", (episode,), min_improvement=-1.0)
    assert float(before) >= 0.0
    assert candidate.family == "test"
    assert len(candidate.parameters) == 2
    assert candidate.accepted is True
    assert candidate.query_loss <= float(before) + 1e-6
