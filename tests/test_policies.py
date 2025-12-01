import pytest
from alrf.classifier.base import ClassifierResult, QueryComplexity, QueryIntent
from alrf.models.config import RouterConfig
from alrf.routing.policies import CostAwarePolicy, LatencyFirstPolicy, QualityFirstPolicy


def _result(complexity: QueryComplexity, needs_retrieval: bool = False) -> ClassifierResult:
    return ClassifierResult(
        complexity=complexity,
        intent=QueryIntent.qa,
        token_estimate=10,
        confidence=0.9,
        needs_retrieval=needs_retrieval,
        signals=["short_query"],
    )


@pytest.fixture
def config() -> RouterConfig:
    return RouterConfig()


# cost_aware

def test_cost_aware_simple_to_local(config: RouterConfig) -> None:
    d = CostAwarePolicy().decide(_result(QueryComplexity.simple), config)
    assert d.tier == "local"
    assert d.provider == "ollama"


def test_cost_aware_moderate_to_fast(config: RouterConfig) -> None:
    d = CostAwarePolicy().decide(_result(QueryComplexity.moderate), config)
    assert d.tier == "fast"
    assert d.provider == "openai"


def test_cost_aware_complex_to_reasoning(config: RouterConfig) -> None:
    d = CostAwarePolicy().decide(_result(QueryComplexity.complex), config)
    assert d.tier == "reasoning"
    assert d.provider == "anthropic"


# quality_first

def test_quality_first_simple_to_fast(config: RouterConfig) -> None:
    assert QualityFirstPolicy().decide(_result(QueryComplexity.simple), config).tier == "fast"


def test_quality_first_moderate_to_reasoning(config: RouterConfig) -> None:
    assert QualityFirstPolicy().decide(_result(QueryComplexity.moderate), config).tier == "reasoning"


def test_quality_first_complex_to_reasoning(config: RouterConfig) -> None:
    assert QualityFirstPolicy().decide(_result(QueryComplexity.complex), config).tier == "reasoning"


# latency_first

def test_latency_first_simple_to_local(config: RouterConfig) -> None:
    assert LatencyFirstPolicy().decide(_result(QueryComplexity.simple), config).tier == "local"


def test_latency_first_complex_skips_reasoning(config: RouterConfig) -> None:
    d = LatencyFirstPolicy().decide(_result(QueryComplexity.complex), config)
    assert d.tier == "fast"


def test_latency_first_never_uses_reasoning(config: RouterConfig) -> None:
    for c in QueryComplexity:
        assert LatencyFirstPolicy().decide(_result(c), config).tier != "reasoning"


# shared behaviour

def test_policy_name_recorded(config: RouterConfig) -> None:
    assert CostAwarePolicy().decide(_result(QueryComplexity.moderate), config).policy == "cost_aware"


def test_use_rag_propagated(config: RouterConfig) -> None:
    d = CostAwarePolicy().decide(_result(QueryComplexity.moderate, needs_retrieval=True), config)
    assert d.use_rag is True


def test_model_comes_from_config() -> None:
    cfg = RouterConfig(fast_model="gpt-4o")
    assert CostAwarePolicy().decide(_result(QueryComplexity.moderate), cfg).model == "gpt-4o"


def test_local_model_comes_from_config() -> None:
    cfg = RouterConfig(local_model="mistral")
    assert CostAwarePolicy().decide(_result(QueryComplexity.simple), cfg).model == "mistral"
