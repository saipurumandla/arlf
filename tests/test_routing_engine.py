import pytest
from alrf.classifier.base import ClassifierResult, QueryComplexity, QueryIntent
from alrf.routing.decision import RoutingDecision
from alrf.routing.engine import RoutingEngine


@pytest.fixture
def engine() -> RoutingEngine:
    return RoutingEngine()


def _result(complexity: QueryComplexity, needs_retrieval: bool = False) -> ClassifierResult:
    return ClassifierResult(
        complexity=complexity,
        intent=QueryIntent.qa,
        token_estimate=10,
        confidence=0.9,
        needs_retrieval=needs_retrieval,
        signals=["short_query"],
    )


def test_simple_routes_to_fast_openai(engine: RoutingEngine) -> None:
    d = engine.decide(_result(QueryComplexity.simple))
    assert d.tier == "fast"
    assert d.provider == "openai"
    assert d.model == "gpt-4o-mini"


def test_moderate_routes_to_fast(engine: RoutingEngine) -> None:
    assert engine.decide(_result(QueryComplexity.moderate)).tier == "fast"


def test_complex_routes_to_reasoning_anthropic(engine: RoutingEngine) -> None:
    d = engine.decide(_result(QueryComplexity.complex))
    assert d.tier == "reasoning"
    assert d.provider == "anthropic"


def test_use_rag_mirrors_needs_retrieval(engine: RoutingEngine) -> None:
    assert engine.decide(_result(QueryComplexity.moderate, needs_retrieval=True)).use_rag is True


def test_no_rag_when_not_needed(engine: RoutingEngine) -> None:
    assert engine.decide(_result(QueryComplexity.simple)).use_rag is False


def test_decision_has_reason(engine: RoutingEngine) -> None:
    assert len(engine.decide(_result(QueryComplexity.simple)).reason) > 0


def test_decision_carries_classifier_result(engine: RoutingEngine) -> None:
    r = _result(QueryComplexity.complex)
    assert engine.decide(r).classifier_result is r


def test_returns_routing_decision_instance(engine: RoutingEngine) -> None:
    assert isinstance(engine.decide(_result(QueryComplexity.simple)), RoutingDecision)
