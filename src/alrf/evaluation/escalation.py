from alrf.models.config import RouterConfig
from alrf.routing.decision import RoutingDecision

_NEXT_TIER: dict[str, str] = {"local": "fast", "fast": "reasoning"}
_TIER_PROVIDER_MODEL: dict[str, tuple[str, str]] = {
    "fast":      ("openai",    "gpt-4o-mini"),
    "reasoning": ("anthropic", "claude-sonnet-4-6"),
}


class EscalationHandler:
    """Upgrade to a stronger tier when confidence falls below threshold."""

    def should_escalate(self, confidence: float, config: RouterConfig, attempts: int) -> bool:
        if attempts >= config.retry_budget:
            return False
        return confidence < config.escalation_threshold

    def escalate(self, decision: RoutingDecision, config: RouterConfig) -> RoutingDecision | None:
        next_tier = _NEXT_TIER.get(decision.tier)
        if next_tier is None:
            return None  # already at reasoning, nowhere to go
        provider, model = _TIER_PROVIDER_MODEL[next_tier]
        return RoutingDecision(
            provider=provider,
            model=model,
            tier=next_tier,
            policy=decision.policy,
            use_rag=decision.use_rag,
            reason=f"escalated from {decision.tier} (low confidence)",
            classifier_result=decision.classifier_result,
        )
