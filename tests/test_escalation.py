import pytest
from alrf.classifier.base import ClassifierResult, QueryComplexity, QueryIntent
from alrf.evaluation.escalation import EscalationHandler
from alrf.models.config import RouterConfig
from alrf.routing.decision import RoutingDecision


def _decision(tier: str) -> RoutingDecision:
    clf = ClassifierResult(
        complexity=QueryComplexity.moderate,
        intent=QueryIntent.qa,
        token_estimate=10,
        confidence=0.9,
        needs_retrieval=False,
        signals=["medium_query"],
    )
    return RoutingDecision(
        provider="openai", model="gpt-4o-mini",
        tier=tier, policy="cost_aware",
        use_rag=False, reason="test",
        classifier_result=clf,
    )


@pytest.fixture
def config() -> RouterConfig:
    return RouterConfig(escalation_threshold=0.70, retry_budget=3)


@pytest.fixture
def handler() -> EscalationHandler:
    return EscalationHandler()


def test_should_escalate_when_below_threshold(handler: EscalationHandler, config: RouterConfig) -> None:
    assert handler.should_escalate(0.5, config, attempts=1) is True


def test_no_escalation_when_above_threshold(handler: EscalationHandler, config: RouterConfig) -> None:
    assert handler.should_escalate(0.8, config, attempts=1) is False


def test_no_escalation_when_budget_exhausted(handler: EscalationHandler, config: RouterConfig) -> None:
    assert handler.should_escalate(0.3, config, attempts=3) is False


def test_escalate_fast_to_reasoning(handler: EscalationHandler, config: RouterConfig) -> None:
    upgraded = handler.escalate(_decision("fast"), config)
    assert upgraded is not None
    assert upgraded.tier == "reasoning"
    assert upgraded.provider == "anthropic"


def test_escalate_local_to_fast(handler: EscalationHandler, config: RouterConfig) -> None:
    upgraded = handler.escalate(_decision("local"), config)
    assert upgraded is not None
    assert upgraded.tier == "fast"


def test_no_escalation_from_reasoning(handler: EscalationHandler, config: RouterConfig) -> None:
    assert handler.escalate(_decision("reasoning"), config) is None


def test_escalated_decision_carries_policy(handler: EscalationHandler, config: RouterConfig) -> None:
    upgraded = handler.escalate(_decision("fast"), config)
    assert upgraded is not None
    assert upgraded.policy == "cost_aware"


def test_escalated_reason_mentions_low_confidence(handler: EscalationHandler, config: RouterConfig) -> None:
    upgraded = handler.escalate(_decision("fast"), config)
    assert upgraded is not None
    assert "escalated" in upgraded.reason
