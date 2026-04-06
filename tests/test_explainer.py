from alrf.classifier.base import ClassifierResult, QueryComplexity, QueryIntent
from alrf.explain.tracer import DecisionTrace, RoutingExplainer
from alrf.models.config import RouterConfig
from alrf.models.response import RouterResult
from alrf.routing.decision import RoutingDecision


def _clf(complexity: QueryComplexity = QueryComplexity.moderate) -> ClassifierResult:
    return ClassifierResult(
        complexity=complexity,
        intent=QueryIntent.debug,
        token_estimate=47,
        confidence=0.75,
        needs_retrieval=False,
        signals=["medium_query", "has_error_keyword"],
    )


def _decision(tier: str = "fast") -> RoutingDecision:
    return RoutingDecision(
        provider="openai", model="gpt-4o-mini",
        tier=tier, policy="cost_aware",
        use_rag=False, reason="test",
        classifier_result=_clf(),
    )


def _result(escalated: bool = False, route: str = "fast") -> RouterResult:
    return RouterResult(
        answer="Connection pool exhausted.",
        route=route,
        provider="anthropic" if escalated else "openai",
        model="claude-sonnet-4-6" if escalated else "gpt-4o-mini",
        cost_usd=0.003,
        latency_ms=1240,
        confidence=0.61,
        escalated=escalated,
        rag_used=False,
        decision_trace=[],
    )


def test_explainer_returns_five_steps() -> None:
    trace = RoutingExplainer().explain(
        "login failing", _clf(), _decision(), _result(), RouterConfig()
    )
    assert isinstance(trace, DecisionTrace)
    assert len(trace.steps) == 5


def test_step_labels_are_correct() -> None:
    trace = RoutingExplainer().explain(
        "login failing", _clf(), _decision(), _result(), RouterConfig()
    )
    labels = [s.label for s in trace.steps]
    assert labels == ["classifier", "policy", "confidence", "escalation", "final"]


def test_classifier_step_has_signals() -> None:
    trace = RoutingExplainer().explain(
        "login failing", _clf(), _decision(), _result(), RouterConfig()
    )
    clf_step = trace.steps[0]
    assert "has_error_keyword" in str(clf_step.details.get("signals", ""))


def test_escalated_trace_shows_upgraded_provider() -> None:
    trace = RoutingExplainer().explain(
        "login failing", _clf(), _decision("fast"),
        _result(escalated=True, route="reasoning"), RouterConfig()
    )
    esc_step = trace.steps[3]
    assert esc_step.details["escalated"] is True
    assert esc_step.details["final_provider"] == "anthropic"


def test_str_representation_readable() -> None:
    trace = RoutingExplainer().explain(
        "login failing", _clf(), _decision(), _result(), RouterConfig()
    )
    text = str(trace)
    assert "classifier" in text
    assert "final" in text
    assert "\n" in text
